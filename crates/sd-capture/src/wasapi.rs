//! Windows backend: WASAPI shared-mode, event-driven capture.
//!
//! * Microphone: capture endpoint. System audio: render endpoint opened with
//!   `AUDCLNT_STREAMFLAGS_LOOPBACK`.
//! * The stream is initialized with `AUTOCONVERTPCM | SRC_DEFAULT_QUALITY` and an explicit
//!   float32 format, so the OS resamples and remixes any device to the configured format.
//! * Timestamps come from `IAudioCaptureClient::GetBuffer`'s `qpcPosition`, which is on the same
//!   QPC clock as [`HostTime::now`].
//! * `IMMNotificationClient` drives default-device following and removal handling.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use sd_core::ring::RingProducer;
use sd_core::{BlockFlags, HostTime, Source, StreamFormat};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{
    CloseHandle, HANDLE, PROPERTYKEY, RPC_E_CHANGED_MODE, WAIT_OBJECT_0,
};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
    AUDCLNT_STREAMOPTIONS_RAW, AudioCategory_Other, AudioClientProperties, DEVICE_STATE,
    DEVICE_STATE_ACTIVE, EDataFlow, ERole, IAudioCaptureClient, IAudioClient, IAudioClient2,
    IAudioClock, IAudioRenderClient, IMMDevice, IMMDeviceEnumerator, IMMNotificationClient,
    IMMNotificationClient_Impl, MMDeviceEnumerator, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    WAVEFORMATEXTENSIBLE_0, eCapture, eConsole, eRender,
};
use windows::Win32::Media::KernelStreaming::{
    SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize, STGM_READ,
};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent,
    WaitForMultipleObjects,
};
use windows::core::{HRESULT, HSTRING, Interface, PCWSTR, w};

use crate::feed::{Feed, Idle, Packet};
use crate::policy::StallWatchdog;
use crate::{
    CaptureConfig, CaptureError, CaptureEvent, CaptureStats, DeviceInfo, DeviceKind,
    DeviceSelector, PlaybackReport, Result, SwitchReason,
};
use windows::Win32::Media::Audio::Endpoints::IAudioMeterInformation;

/// `E_NOTFOUND` / `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`: no such endpoint.
const E_NOTFOUND: HRESULT = HRESULT(0x8007_0490_u32 as i32);

fn os_err(context: &str, e: windows::core::Error) -> CaptureError {
    CaptureError::Os(format!("{context}: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Small RAII helpers
// ---------------------------------------------------------------------------------------------

/// Initializes COM (MTA) for the current thread and balances it on drop.
struct ComGuard {
    uninit: bool,
}

impl ComGuard {
    fn new() -> Result<Self> {
        // SAFETY: plain COM initialization for the calling thread.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_ok() {
            Ok(Self { uninit: true })
        } else if hr == RPC_E_CHANGED_MODE {
            // The thread is already an STA; MMDevice APIs work there too.
            Ok(Self { uninit: false })
        } else {
            Err(CaptureError::Os(format!("CoInitializeEx failed: {hr:?}")))
        }
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.uninit {
            // SAFETY: balances the successful CoInitializeEx on this thread.
            unsafe { CoUninitialize() };
        }
    }
}

/// A Win32 event handle that may be shared with other threads.
#[derive(Clone, Copy)]
struct SharedHandle(HANDLE);
// SAFETY: Win32 event handles are process-wide kernel objects, usable from any thread.
unsafe impl Send for SharedHandle {}
// SAFETY: as above; SetEvent/WaitFor* are thread-safe.
unsafe impl Sync for SharedHandle {}

struct OwnedEvent(HANDLE);

impl OwnedEvent {
    fn new() -> Result<Self> {
        // SAFETY: creates an unnamed auto-reset event with default security.
        let h = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
            .map_err(|e| os_err("CreateEventW", e))?;
        Ok(Self(h))
    }
    fn shared(&self) -> SharedHandle {
        SharedHandle(self.0)
    }
}

impl Drop for OwnedEvent {
    fn drop(&mut self) {
        // SAFETY: we own this handle and close it exactly once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn signal(h: SharedHandle) {
    // SAFETY: `h` is a live event handle owned by the capture handle or thread.
    let _ = unsafe { SetEvent(h.0) };
}

/// Takes ownership of a COM-allocated wide string and converts it.
///
/// # Safety
/// `p` must be null or a NUL-terminated string allocated with `CoTaskMemAlloc`.
unsafe fn take_co_string(p: windows::core::PWSTR) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: guaranteed by the caller.
    let s = unsafe { p.to_string() }.unwrap_or_default();
    // SAFETY: allocated by COM, freed exactly once here.
    unsafe { CoTaskMemFree(Some(p.0 as *const _)) };
    s
}

// ---------------------------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------------------------

fn flow_of(kind: DeviceKind) -> EDataFlow {
    match kind {
        DeviceKind::Input => eCapture,
        DeviceKind::Output => eRender,
    }
}

fn kind_of(source: Source) -> DeviceKind {
    match source {
        Source::Mic => DeviceKind::Input,
        Source::System => DeviceKind::Output,
    }
}

fn enumerator() -> Result<IMMDeviceEnumerator> {
    // SAFETY: standard COM activation of the MMDevice enumerator on a COM-initialized thread.
    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
        .map_err(|e| os_err("create IMMDeviceEnumerator", e))
}

fn device_id(dev: &IMMDevice) -> Result<String> {
    // SAFETY: GetId returns a CoTaskMem string that take_co_string frees.
    unsafe { dev.GetId().map(|p| take_co_string(p)) }.map_err(|e| os_err("IMMDevice::GetId", e))
}

fn property_string(dev: &IMMDevice, key: &PROPERTYKEY) -> Option<String> {
    // SAFETY: read-only property store access; the PROPVARIANT is cleared on drop and the
    // allocated string is freed by take_co_string.
    unsafe {
        let store = dev.OpenPropertyStore(STGM_READ).ok()?;
        let value = store.GetValue(key).ok()?;
        let p = PropVariantToStringAlloc(&value).ok()?;
        Some(take_co_string(p))
    }
}

fn mix_format(dev: &IMMDevice) -> Option<StreamFormat> {
    // SAFETY: activates a temporary IAudioClient; GetMixFormat returns a CoTaskMem block that is
    // read (unaligned, the struct is packed) and freed exactly once.
    unsafe {
        let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).ok()?;
        let p = client.GetMixFormat().ok()?;
        let fmt = std::ptr::read_unaligned(p);
        CoTaskMemFree(Some(p as *const _));
        StreamFormat::new(fmt.nSamplesPerSec, fmt.nChannels).ok()
    }
}

fn is_active(dev: &IMMDevice) -> bool {
    // SAFETY: simple getter.
    unsafe { dev.GetState() }.is_ok_and(|s| s == DEVICE_STATE_ACTIVE)
}

pub fn list_devices(kind: DeviceKind) -> Result<Vec<DeviceInfo>> {
    let _com = ComGuard::new()?;
    let en = enumerator()?;
    // SAFETY: COM calls on a COM-initialized thread with valid arguments.
    let default_id = unsafe { en.GetDefaultAudioEndpoint(flow_of(kind), eConsole) }
        .ok()
        .and_then(|d| device_id(&d).ok());
    // SAFETY: as above.
    let coll = unsafe { en.EnumAudioEndpoints(flow_of(kind), DEVICE_STATE_ACTIVE) }
        .map_err(|e| os_err("EnumAudioEndpoints", e))?;
    // SAFETY: as above.
    let count = unsafe { coll.GetCount() }.map_err(|e| os_err("GetCount", e))?;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        // SAFETY: index is within GetCount.
        let dev = unsafe { coll.Item(i) }.map_err(|e| os_err("Item", e))?;
        let id = device_id(&dev)?;
        out.push(DeviceInfo {
            is_default: default_id.as_deref() == Some(id.as_str()),
            name: property_string(&dev, &PKEY_Device_FriendlyName).unwrap_or_else(|| id.clone()),
            mix_format: mix_format(&dev),
            id,
            kind,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Device notifications
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
enum Notice {
    DefaultChanged(Option<String>),
    Removed(String),
    StateChanged(String, bool),
    Added(String),
}

#[windows_core::implement(IMMNotificationClient)]
struct Notifier {
    flow: EDataFlow,
    tx: Sender<Notice>,
    wake: SharedHandle,
}

impl Notifier {
    fn send(&self, n: Notice) {
        if self.tx.send(n).is_ok() {
            signal(self.wake);
        }
    }
}

fn pcwstr_to_string(p: &PCWSTR) -> Option<String> {
    if p.is_null() {
        None
    } else {
        // SAFETY: the OS passes a valid NUL-terminated string for the callback's duration.
        Some(unsafe { p.to_string() }.unwrap_or_default())
    }
}

// Callbacks run on an OS thread: they only enqueue and signal, never block.
impl IMMNotificationClient_Impl for Notifier_Impl {
    fn OnDeviceStateChanged(&self, id: &PCWSTR, state: DEVICE_STATE) -> windows::core::Result<()> {
        if let Some(id) = pcwstr_to_string(id) {
            self.send(Notice::StateChanged(id, state == DEVICE_STATE_ACTIVE));
        }
        Ok(())
    }
    fn OnDeviceAdded(&self, id: &PCWSTR) -> windows::core::Result<()> {
        if let Some(id) = pcwstr_to_string(id) {
            self.send(Notice::Added(id));
        }
        Ok(())
    }
    fn OnDeviceRemoved(&self, id: &PCWSTR) -> windows::core::Result<()> {
        if let Some(id) = pcwstr_to_string(id) {
            self.send(Notice::Removed(id));
        }
        Ok(())
    }
    fn OnDefaultDeviceChanged(
        &self,
        flow: EDataFlow,
        role: ERole,
        id: &PCWSTR,
    ) -> windows::core::Result<()> {
        if flow == self.flow && role == eConsole {
            self.send(Notice::DefaultChanged(pcwstr_to_string(id)));
        }
        Ok(())
    }
    fn OnPropertyValueChanged(
        &self,
        _id: &PCWSTR,
        _key: &PROPERTYKEY,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Capture thread
// ---------------------------------------------------------------------------------------------

enum Control {
    SetDevice(DeviceSelector),
}

pub struct Handle {
    thread: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    stop_event: OwnedEvent,
    wake_event: OwnedEvent,
    ctrl: Sender<Control>,
}

// SAFETY: OwnedEvent only wraps kernel handles, which are usable from any thread.
unsafe impl Send for Handle {}

impl Handle {
    pub fn set_device(&self, device: DeviceSelector) {
        if self.ctrl.send(Control::SetDevice(device)).is_ok() {
            signal(self.wake_event.shared());
        }
    }

    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Release);
        signal(self.stop_event.shared());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn start(
    cfg: CaptureConfig,
    producer: RingProducer,
    events: Sender<CaptureEvent>,
    stats: Arc<CaptureStats>,
) -> Result<Handle> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_event = OwnedEvent::new()?;
    let wake_event = OwnedEvent::new()?;
    let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded();
    let ctx = ThreadCtx {
        cfg: cfg.clone(),
        // Loopback delivers nothing while nothing plays: fill silence quickly.
        feed: Feed::new(
            &cfg,
            producer,
            stats.clone(),
            Idle {
                after_ms: 60,
                margin_ms: 100,
            },
        ),
        events,
        stats,
        stop: stop.clone(),
        stop_event: stop_event.shared(),
        wake_event: wake_event.shared(),
        ctrl: ctrl_rx,
    };
    let thread = std::thread::Builder::new()
        .name(format!("sd-capture-{}", cfg.source))
        .spawn(move || ctx.run())
        .map_err(|e| CaptureError::Thread(e.to_string()))?;
    Ok(Handle {
        thread: Some(thread),
        stop,
        stop_event,
        wake_event,
        ctrl: ctrl_tx,
    })
}

struct ThreadCtx {
    cfg: CaptureConfig,
    feed: Feed,
    events: Sender<CaptureEvent>,
    stats: Arc<CaptureStats>,
    stop: Arc<AtomicBool>,
    stop_event: SharedHandle,
    wake_event: SharedHandle,
    ctrl: Receiver<Control>,
}

struct OpenStream {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    event: OwnedEvent,
    device_id: String,
    raw: bool,
    /// Loopback only: the output endpoint's peak meter, to tell "nothing playing" from a stall.
    meter: Option<IAudioMeterInformation>,
    /// The device's native (mix) rate. WASAPI reports capture positions in these frames even
    /// when AUTOCONVERTPCM delivers another rate (measured: 160 per 480-frame packet on a
    /// 16 kHz Bluetooth headset), so positions are scaled to stream frames before use.
    device_rate: u32,
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        // SAFETY: stopping an initialized client; errors (e.g. device gone) are irrelevant here.
        let _ = unsafe { self.client.Stop() };
    }
}

enum StreamExit {
    Stop,
    Switch(SwitchReason),
    Failed(String),
}

enum OpenError {
    NoDevice,
    Other(String),
}

/// Mutable state that survives device switches (the timeline itself is in [`Feed`]).
struct Session {
    preferred: DeviceSelector,
    on_fallback: bool,
    notices: Receiver<Notice>,
    /// Set when the stream should be reopened after the current packets (see GapFillPolicy).
    recover_requested: bool,
    /// The test stall (`test_stall_after_ms`) has been injected; it happens once per session.
    stall_injected: bool,
}

impl ThreadCtx {
    fn emit(&self, e: CaptureEvent) {
        let _ = self.events.send(e);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn channels(&self) -> usize {
        self.cfg.format.channels as usize
    }

    fn ns_of(&self, frames: u64) -> u64 {
        self.cfg.format.frames_to_ns(frames)
    }

    fn run(mut self) {
        let _com = match ComGuard::new() {
            Ok(c) => c,
            Err(e) => {
                self.emit(CaptureEvent::Retrying {
                    attempt: 0,
                    retry_in_ms: 0,
                    error: e.to_string(),
                });
                self.emit(CaptureEvent::Stopped);
                return;
            }
        };
        let mut task_index = 0u32;
        // SAFETY: registers this thread with MMCSS; reverted below. Failure is non-fatal.
        let mmcss = unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task_index) }.ok();

        let en = match enumerator() {
            Ok(e) => e,
            Err(e) => {
                self.emit(CaptureEvent::Retrying {
                    attempt: 0,
                    retry_in_ms: 0,
                    error: e.to_string(),
                });
                self.emit(CaptureEvent::Stopped);
                return;
            }
        };
        let (ntx, nrx) = crossbeam_channel::unbounded();
        let notifier: IMMNotificationClient = Notifier {
            flow: flow_of(kind_of(self.cfg.source)),
            tx: ntx,
            wake: self.wake_event,
        }
        .into();
        // SAFETY: registering a valid COM callback object; unregistered before return.
        let registered = unsafe { en.RegisterEndpointNotificationCallback(&notifier) }.is_ok();

        let mut sess = Session {
            preferred: self.cfg.device.clone(),
            on_fallback: false,
            notices: nrx,
            recover_requested: false,
            stall_injected: false,
        };
        // Both sources start their timeline at capture start, so frame 0 means the same instant.
        self.feed.start_at(HostTime::now());

        let mut attempt = 0u32;
        let mut reported_no_device = false;
        while !self.stopped() {
            match self.open(&en, &mut sess) {
                Ok(stream) => {
                    attempt = 0;
                    reported_no_device = false;
                    match self.pump(&stream, &mut sess) {
                        StreamExit::Stop => break,
                        StreamExit::Switch(reason) => {
                            self.stats.restarts.fetch_add(1, Ordering::Relaxed);
                            self.emit(CaptureEvent::Switching { reason });
                            continue;
                        }
                        StreamExit::Failed(err) => {
                            attempt += 1;
                            let delay = backoff_ms(attempt);
                            self.emit(CaptureEvent::Retrying {
                                attempt,
                                retry_in_ms: delay,
                                error: err,
                            });
                            drop(stream);
                            self.idle_wait(&mut sess, delay);
                        }
                    }
                }
                Err(OpenError::NoDevice) => {
                    if !reported_no_device {
                        self.emit(CaptureEvent::NoDevice);
                        reported_no_device = true;
                    }
                    attempt += 1;
                    self.idle_wait(&mut sess, backoff_ms(attempt).min(1_000));
                }
                Err(OpenError::Other(err)) => {
                    attempt += 1;
                    let delay = backoff_ms(attempt);
                    self.emit(CaptureEvent::Retrying {
                        attempt,
                        retry_in_ms: delay,
                        error: err,
                    });
                    self.idle_wait(&mut sess, delay);
                }
            }
        }

        if registered {
            // SAFETY: unregisters the callback registered above.
            let _ = unsafe { en.UnregisterEndpointNotificationCallback(&notifier) };
        }
        if let Some(h) = mmcss {
            // SAFETY: reverts the MMCSS registration made above.
            let _ = unsafe { AvRevertMmThreadCharacteristics(h) };
        }
        self.emit(CaptureEvent::Stopped);
    }

    fn open(&self, en: &IMMDeviceEnumerator, sess: &mut Session) -> Result<OpenStream, OpenError> {
        let flow = flow_of(kind_of(self.cfg.source));
        let preferred = match &sess.preferred {
            DeviceSelector::Id(id) => Some(id.clone()),
            DeviceSelector::Default => None,
        };
        let device = if let Some(id) = &preferred {
            // SAFETY: COM call with a valid HSTRING argument.
            match unsafe { en.GetDevice(&HSTRING::from(id.as_str())) } {
                Ok(d) if is_active(&d) => {
                    sess.on_fallback = false;
                    Some(d)
                }
                _ => {
                    if !sess.on_fallback {
                        sess.on_fallback = true;
                        self.emit(CaptureEvent::Switching {
                            reason: SwitchReason::DeviceRemoved,
                        });
                    }
                    None
                }
            }
        } else {
            None
        };
        let device = match device {
            Some(d) => d,
            // SAFETY: COM call with valid enum arguments.
            None => match unsafe { en.GetDefaultAudioEndpoint(flow, eConsole) } {
                Ok(d) => d,
                Err(e) if e.code() == E_NOTFOUND => return Err(OpenError::NoDevice),
                Err(e) => return Err(OpenError::Other(format!("GetDefaultAudioEndpoint: {e}"))),
            },
        };
        let id = device_id(&device).map_err(|e| OpenError::Other(e.to_string()))?;
        let stream = self.init_stream(&device, id.clone()).map_err(|e| {
            if e.code() == AUDCLNT_E_DEVICE_INVALIDATED || e.code() == E_NOTFOUND {
                OpenError::NoDevice
            } else {
                OpenError::Other(format!("initialize {}: {e}", self.cfg.source))
            }
        })?;
        self.emit(CaptureEvent::Started {
            device_name: property_string(&device, &PKEY_Device_FriendlyName)
                .unwrap_or_else(|| id.clone()),
            mix_format: mix_format(&device),
            device_id: id,
            raw: stream.raw,
        });
        Ok(stream)
    }

    fn init_stream(
        &self,
        device: &IMMDevice,
        device_id: String,
    ) -> windows::core::Result<OpenStream> {
        let fmt = float_format(self.cfg.format);
        let mut flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
            | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        if self.cfg.source == Source::System {
            flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
        }
        let buffer_100ns = self.cfg.os_buffer_ms as i64 * 10_000;
        // SAFETY: creates an unnamed auto-reset event; OwnedEvent closes it.
        let event = OwnedEvent(unsafe { CreateEventW(None, false, false, PCWSTR::null())? });
        // SAFETY: COM calls on a COM-initialized thread. `fmt` outlives Initialize (the client
        // copies it); the event handle stays alive as long as the OpenStream that owns it.
        unsafe {
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
            // Raw mode must be requested before Initialize. Failure is not fatal.
            let mut raw = false;
            if self.cfg.raw
                && let Ok(c2) = client.cast::<IAudioClient2>()
            {
                let props = AudioClientProperties {
                    cbSize: std::mem::size_of::<AudioClientProperties>() as u32,
                    bIsOffload: false.into(),
                    eCategory: AudioCategory_Other,
                    Options: AUDCLNT_STREAMOPTIONS_RAW,
                };
                raw = c2.SetClientProperties(&props).is_ok();
            }
            let mix = client.GetMixFormat()?;
            let device_rate = std::ptr::read_unaligned(mix).nSamplesPerSec;
            CoTaskMemFree(Some(mix as *const _));
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                flags,
                buffer_100ns,
                0,
                &fmt as *const WAVEFORMATEXTENSIBLE as *const WAVEFORMATEX,
                None,
            )?;
            client.SetEventHandle(event.0)?;
            let capture: IAudioCaptureClient = client.GetService()?;
            let meter = if self.cfg.source == Source::System {
                device
                    .Activate::<IAudioMeterInformation>(CLSCTX_ALL, None)
                    .ok()
            } else {
                None
            };
            client.Start()?;
            Ok(OpenStream {
                client,
                capture,
                event,
                device_id,
                raw,
                meter,
                device_rate,
            })
        }
    }

    /// Waits up to `ms` while keeping the timeline running with silence; returns early on stop,
    /// a control message, or a device notification that may make a device available.
    fn idle_wait(&mut self, sess: &mut Session, ms: u64) {
        let deadline = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < deadline && !self.stopped() {
            let handles = [self.stop_event.0, self.wake_event.0];
            // SAFETY: both handles are valid events owned by the capture handle.
            let r = unsafe { WaitForMultipleObjects(&handles, false, 20) };
            self.feed.fill_idle();
            if r == WAIT_OBJECT_0 {
                return;
            }
            if r.0 == WAIT_OBJECT_0.0 + 1 {
                let mut wake = false;
                while let Ok(n) = sess.notices.try_recv() {
                    wake |= matches!(
                        n,
                        Notice::Added(_)
                            | Notice::DefaultChanged(_)
                            | Notice::StateChanged(_, true)
                    );
                }
                while let Ok(Control::SetDevice(sel)) = self.ctrl.try_recv() {
                    sess.preferred = sel;
                    wake = true;
                }
                if wake {
                    return;
                }
            }
        }
    }

    fn pump(&mut self, stream: &OpenStream, sess: &mut Session) -> StreamExit {
        self.feed.new_stream(); // device positions restart with every stream
        let started = Instant::now();
        let mut watchdog = StallWatchdog::new(self.cfg.stall_timeout_ms, started);
        let stall_timeout = Duration::from_millis(self.cfg.stall_timeout_ms as u64);
        let mut playing_since: Option<Instant> = None;
        let inject_at = self
            .cfg
            .test_stall_after_ms
            .filter(|_| !sess.stall_injected)
            .map(|ms| started + Duration::from_millis(ms as u64));
        let mut pending_default_switch: Option<Instant> = None;
        let settle = Duration::from_millis(self.cfg.default_switch_settle_ms as u64);
        loop {
            let handles = [self.stop_event.0, self.wake_event.0, stream.event.0];
            // SAFETY: all three handles are valid events for the duration of this call.
            let r = unsafe { WaitForMultipleObjects(&handles, false, 20) };
            if r == WAIT_OBJECT_0 || self.stopped() {
                return StreamExit::Stop;
            }
            if r.0 == WAIT_OBJECT_0.0 + 1 {
                // Only the most recent device request matters.
                let mut requested = None;
                while let Ok(Control::SetDevice(sel)) = self.ctrl.try_recv() {
                    requested = Some(sel);
                }
                if let Some(sel) = requested {
                    sess.preferred = sel;
                    sess.on_fallback = false;
                    return StreamExit::Switch(SwitchReason::Requested);
                }
                while let Ok(n) = sess.notices.try_recv() {
                    match n {
                        Notice::DefaultChanged(new_id) => {
                            let follows_default = matches!(sess.preferred, DeviceSelector::Default)
                                || sess.on_fallback;
                            if follows_default
                                && new_id.as_deref() != Some(stream.device_id.as_str())
                            {
                                pending_default_switch = Some(Instant::now() + settle);
                            }
                        }
                        Notice::Removed(id) | Notice::StateChanged(id, false)
                            if id == stream.device_id =>
                        {
                            return StreamExit::Switch(SwitchReason::DeviceRemoved);
                        }
                        Notice::Added(id) | Notice::StateChanged(id, true) => {
                            if sess.on_fallback
                                && matches!(&sess.preferred, DeviceSelector::Id(p) if *p == id)
                            {
                                return StreamExit::Switch(SwitchReason::Requested);
                            }
                        }
                        _ => {}
                    }
                }
            }
            let now = Instant::now();
            let injected = inject_at.is_some_and(|t| now >= t);
            if injected {
                sess.stall_injected = true;
            }
            let read = if injected {
                Ok(0) // fault injection: behave as if the driver delivered nothing
            } else {
                self.drain(stream, sess)
            };
            if let Ok(n) = read {
                if n > 0 {
                    watchdog.packet(now);
                    playing_since = None;
                } else if let Some(quiet) = watchdog.stalled_for(now) {
                    let expected = match &stream.meter {
                        None => true, // a microphone always delivers packets
                        Some(m) => {
                            // SAFETY: plain COM getter on a valid meter interface.
                            let peak = unsafe { m.GetPeakValue() }.unwrap_or(0.0);
                            if peak > 1e-4 {
                                let since = *playing_since.get_or_insert(now);
                                now.duration_since(since) >= stall_timeout
                            } else {
                                playing_since = None;
                                false
                            }
                        }
                    };
                    if expected {
                        self.emit(CaptureEvent::Stalled {
                            quiet_ms: quiet.as_millis() as u64,
                        });
                        return StreamExit::Switch(SwitchReason::Recovery);
                    }
                }
            }
            match read {
                Ok(0) => self.feed.fill_idle(),
                Ok(_) if sess.recover_requested => {
                    sess.recover_requested = false;
                    return StreamExit::Switch(SwitchReason::Recovery);
                }
                Ok(_) => {}
                Err(e) if e.code() == AUDCLNT_E_DEVICE_INVALIDATED => {
                    return StreamExit::Switch(SwitchReason::DeviceInvalidated);
                }
                Err(e) => return StreamExit::Failed(format!("read {}: {e}", self.cfg.source)),
            }
            if pending_default_switch.is_some_and(|t| Instant::now() >= t) {
                return StreamExit::Switch(SwitchReason::DefaultDeviceChanged);
            }
        }
    }

    /// Reads every available packet. Returns the number of packets read.
    fn drain(&mut self, stream: &OpenStream, sess: &mut Session) -> windows::core::Result<u32> {
        let ch = self.channels();
        let mut packets = 0;
        loop {
            // SAFETY: COM call on the thread that owns the stream.
            let next = unsafe { stream.capture.GetNextPacketSize()? };
            if next == 0 {
                return Ok(packets);
            }
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut frames = 0u32;
            let mut flags = 0u32;
            let mut qpc = 0u64;
            let mut devpos = 0u64;
            // SAFETY: out-pointers are valid locals; the buffer stays valid until ReleaseBuffer.
            unsafe {
                stream.capture.GetBuffer(
                    &mut data,
                    &mut frames,
                    &mut flags,
                    Some(&mut devpos),
                    Some(&mut qpc),
                )?
            };
            packets += 1;
            if frames > 0 {
                let mut bflags = BlockFlags::empty();
                let ts_error =
                    flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32 != 0 || qpc == 0;
                let t = if ts_error {
                    bflags.insert(BlockFlags::TIMESTAMP_ESTIMATED);
                    HostTime::from_ns(
                        HostTime::now()
                            .as_ns()
                            .saturating_sub(self.ns_of(frames as u64)),
                    )
                } else {
                    HostTime::from_100ns_units(qpc)
                };
                if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 {
                    bflags.insert(BlockFlags::DISCONTINUITY);
                }
                let silent = flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 || data.is_null();
                if silent {
                    bflags.insert(BlockFlags::SILENT);
                }
                let position = (stream.device_rate > 0).then(|| {
                    devpos as f64 * self.cfg.format.sample_rate as f64 / stream.device_rate as f64
                });
                let samples = (!silent).then(|| {
                    // SAFETY: WASAPI guarantees `frames * nBlockAlign` readable bytes at `data`;
                    // the format is float32 x `ch` as requested, so the buffer holds
                    // `frames * ch` f32 values (WASAPI buffers are 4-byte aligned). The slice
                    // is used before ReleaseBuffer.
                    unsafe { std::slice::from_raw_parts(data as *const f32, frames as usize * ch) }
                });
                if let Some(filled_ms) = self.feed.packet(Packet {
                    t,
                    frames,
                    position,
                    flags: bflags,
                    data: samples,
                }) {
                    self.emit(CaptureEvent::GapFillBudgetExhausted { filled_ms });
                    sess.recover_requested = true;
                }
            }
            // SAFETY: releases exactly the frames obtained from GetBuffer.
            unsafe { stream.capture.ReleaseBuffer(frames)? };
        }
    }
}

/// Interleaved float32 format at `f`'s rate and channel count.
fn float_format(f: StreamFormat) -> WAVEFORMATEXTENSIBLE {
    let (ch, rate) = (f.channels, f.sample_rate);
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
            nChannels: ch,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * ch as u32 * 4,
            nBlockAlign: ch * 4,
            wBitsPerSample: 32,
            cbSize: (std::mem::size_of::<WAVEFORMATEXTENSIBLE>()
                - std::mem::size_of::<WAVEFORMATEX>()) as u16,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 {
            wValidBitsPerSample: 32,
        },
        dwChannelMask: if ch == 1 {
            SPEAKER_FRONT_CENTER
        } else {
            SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
        },
        SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    }
}

// ---------------------------------------------------------------------------------------------
// Playback (test signals: loopback verification now, echo-path measurement later)
// ---------------------------------------------------------------------------------------------

pub fn play_blocking(
    device: &DeviceSelector,
    format: StreamFormat,
    samples: &[f32],
) -> Result<PlaybackReport> {
    let ch = format.channels as usize;
    if !samples.len().is_multiple_of(ch) {
        return Err(CaptureError::Config(
            "sample count is not a multiple of channels".into(),
        ));
    }
    let total = (samples.len() / ch) as u64;
    let _com = ComGuard::new()?;
    let en = enumerator()?;
    // SAFETY: COM calls on a COM-initialized thread; every pointer passed is valid for the call,
    // GetBuffer returns room for exactly the requested frames, which are fully written before
    // ReleaseBuffer, and the event handle outlives the client (dropped after Stop).
    unsafe {
        let dev = match device {
            DeviceSelector::Id(id) => en.GetDevice(&HSTRING::from(id.as_str())),
            DeviceSelector::Default => en.GetDefaultAudioEndpoint(eRender, eConsole),
        }
        .map_err(|e| os_err("open output device", e))?;
        let client: IAudioClient = dev
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| os_err("Activate", e))?;
        let fmt = float_format(format);
        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                    | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                    | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                1_000_000,
                0,
                &fmt as *const WAVEFORMATEXTENSIBLE as *const WAVEFORMATEX,
                None,
            )
            .map_err(|e| os_err("Initialize render", e))?;
        let event = OwnedEvent::new()?;
        client
            .SetEventHandle(event.0)
            .map_err(|e| os_err("SetEventHandle", e))?;
        let buffer = client
            .GetBufferSize()
            .map_err(|e| os_err("GetBufferSize", e))?;
        let render: IAudioRenderClient = client
            .GetService()
            .map_err(|e| os_err("render client", e))?;
        let clock: IAudioClock = client.GetService().map_err(|e| os_err("audio clock", e))?;
        let freq = clock
            .GetFrequency()
            .map_err(|e| os_err("GetFrequency", e))? as f64;

        let write = |n: u32, written: &mut u64| -> Result<()> {
            if n == 0 {
                return Ok(());
            }
            let p = render
                .GetBuffer(n)
                .map_err(|e| os_err("render GetBuffer", e))? as *mut f32;
            let dst = std::slice::from_raw_parts_mut(p, n as usize * ch);
            let start = (*written as usize * ch).min(samples.len());
            let avail = (samples.len() - start).min(dst.len());
            dst[..avail].copy_from_slice(&samples[start..start + avail]);
            dst[avail..].fill(0.0);
            render
                .ReleaseBuffer(n, 0)
                .map_err(|e| os_err("render ReleaseBuffer", e))?;
            *written += n as u64;
            Ok(())
        };
        let mut written = 0u64;
        write(buffer, &mut written)?;
        client.Start().map_err(|e| os_err("render Start", e))?;
        let mut offsets = Vec::new();
        let deadline = Instant::now()
            + Duration::from_secs_f64(total as f64 / format.sample_rate as f64 + 5.0);
        loop {
            let _ = WaitForMultipleObjects(&[event.0], false, 200);
            let mut pos = 0u64;
            let mut qpc = 0u64;
            if clock.GetPosition(&mut pos, Some(&mut qpc)).is_ok() && pos > 0 && qpc > 0 {
                // Host time at which frame 0 passed the device position.
                let played_s = pos as f64 / freq;
                offsets.push(qpc as f64 * 100.0 - played_s * 1e9);
                if played_s * format.sample_rate as f64
                    >= total as f64 + format.sample_rate as f64 * 0.05
                {
                    break;
                }
            }
            if Instant::now() > deadline {
                break;
            }
            let padding = client
                .GetCurrentPadding()
                .map_err(|e| os_err("GetCurrentPadding", e))?;
            write(buffer - padding, &mut written)?;
        }
        let _ = client.Stop();
        if offsets.is_empty() {
            return Err(CaptureError::Os("render clock never advanced".into()));
        }
        offsets.sort_by(f64::total_cmp);
        Ok(PlaybackReport {
            first_frame_host_time: HostTime::from_ns(offsets[offsets.len() / 2] as u64),
            frames: total,
            clock_samples: offsets.len(),
            underruns: 0,
        })
    }
}

fn backoff_ms(attempt: u32) -> u64 {
    (100u64 << attempt.saturating_sub(1).min(6)).min(5_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_ms(1), 100);
        assert_eq!(backoff_ms(2), 200);
        assert_eq!(backoff_ms(4), 800);
        assert_eq!(backoff_ms(20), 5_000);
    }
}
