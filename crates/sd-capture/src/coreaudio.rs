//! macOS backend: Core Audio (HAL).
//!
//! * Microphone: an IOProc on the input device.
//! * System audio: a **Core Audio process tap** (macOS 14.2+), a private global stereo tap of
//!   every process's output, read through a private aggregate device whose clock is the output
//!   device. The tap functions are looked up at run time, so the binary still runs (mic only)
//!   on older macOS, where system audio reports [`CaptureError::Unsupported`].
//! * The IOProc runs on Core Audio's real-time thread and only copies samples and timestamps
//!   into lock-free rings; the capture thread converts them to the configured format
//!   ([`Converter`]: channel mapping and resampling, since Core Audio delivers each device's
//!   own rate) and places them on the timeline ([`Feed`]).
//! * Timestamps: the IOProc's input time, `mHostTime` (mach ticks, converted to ns on the
//!   same clock as [`HostTime::now`]). Device positions: `mSampleTime`; a jump is flagged as a
//!   discontinuity so the timeline aligner measures the gap.
//! * Property listeners drive default-device following (200 ms settle), removal and fallback.
//! * Permissions: the microphone's TCC status is read from AVFoundation (a denial is reported
//!   as [`CaptureEvent::PermissionNeeded`]). System audio needs "System Audio Recording"; macOS
//!   has no public query for it, and a tap without it delivers silence, so ten seconds of
//!   exact silence are reported as a possible missing permission.

use std::ffi::{CStr, c_void};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2_core_audio as ca;
use objc2_core_audio::{AudioObjectID, AudioObjectPropertyAddress};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, AudioTimeStampFlags,
    kAudioFormatFlagIsFloat, kAudioFormatLinearPCM,
};
use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSObject, NSProcessInfo, NSString};
use sd_core::ring::RingProducer;
use sd_core::{BlockFlags, HostTime, Source, StreamFormat};

use crate::convert::Converter;
use crate::feed::{Feed, Idle, Packet};
use crate::policy::StallWatchdog;
use crate::{
    CaptureConfig, CaptureError, CaptureEvent, CaptureStats, DeviceInfo, DeviceKind,
    DeviceSelector, PlaybackReport, Result, SwitchReason,
};

/// Prints each step of opening a stream when `SD_COREAUDIO_DIAG` is set (for CI).
fn diag(step: &str) {
    if std::env::var_os("SD_COREAUDIO_DIAG").is_some() {
        eprintln!("coreaudio diag: {step}");
    }
}

const SYSTEM_OBJECT: AudioObjectID = ca::kAudioObjectSystemObject as AudioObjectID;
const TICK: Duration = Duration::from_millis(10);
/// How long tap creation may block (on the permission prompt) before it is reported.
const TAP_PROMPT_HINT: Duration = Duration::from_secs(2);
/// Seconds of exact silence from a tap after which a missing permission is suspected.
const TAP_SILENCE_HINT_S: f64 = 10.0;

// ---------------------------------------------------------------------------------------------
// Property helpers
// ---------------------------------------------------------------------------------------------

fn address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: ca::kAudioObjectPropertyElementMain,
    }
}

/// An OSStatus as text: four-character codes are how Core Audio errors are documented.
fn status_text(status: i32) -> String {
    let b = (status as u32).to_be_bytes();
    if b.iter().all(|c| c.is_ascii_graphic()) {
        format!("'{}' ({status})", String::from_utf8_lossy(&b))
    } else {
        status.to_string()
    }
}

fn check(status: i32, what: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {}", status_text(status)))
    }
}

/// A fixed-size property value.
fn get<T: Copy>(object: AudioObjectID, selector: u32, scope: u32) -> Result<T, String> {
    let a = address(selector, scope);
    let mut value = MaybeUninit::<T>::uninit();
    let mut size = size_of::<T>() as u32;
    // SAFETY: `value` has room for `size` bytes; Core Audio writes at most that much.
    let status = unsafe {
        ca::AudioObjectGetPropertyData(
            object,
            NonNull::from(&a),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(value.as_mut_ptr()).cast(),
        )
    };
    check(status, "AudioObjectGetPropertyData")?;
    if size as usize != size_of::<T>() {
        return Err(format!("property size {size}, expected {}", size_of::<T>()));
    }
    // SAFETY: the call succeeded and filled all of `T`.
    Ok(unsafe { value.assume_init() })
}

/// A variable-size property value, as 8-byte aligned words.
fn get_bytes(object: AudioObjectID, selector: u32, scope: u32) -> Result<(Vec<u64>, u32), String> {
    let a = address(selector, scope);
    let mut size = 0u32;
    // SAFETY: plain size query.
    let status = unsafe {
        ca::AudioObjectGetPropertyDataSize(
            object,
            NonNull::from(&a),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
        )
    };
    check(status, "AudioObjectGetPropertyDataSize")?;
    let mut buf = vec![0u64; (size as usize).div_ceil(8).max(1)];
    // SAFETY: `buf` holds at least `size` bytes.
    let status = unsafe {
        ca::AudioObjectGetPropertyData(
            object,
            NonNull::from(&a),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(buf.as_mut_ptr()).cast(),
        )
    };
    check(status, "AudioObjectGetPropertyData")?;
    Ok((buf, size))
}

/// A CFString property (the caller owns the returned reference).
fn get_string(object: AudioObjectID, selector: u32, scope: u32) -> Option<String> {
    let p: *const CFString = get(object, selector, scope).ok()?;
    // SAFETY: Core Audio returns a +1 CFStringRef for string properties; CFRetained releases it.
    let s = unsafe { CFRetained::from_raw(NonNull::new(p as *mut CFString)?) };
    Some(s.to_string())
}

fn device_ids() -> Vec<AudioObjectID> {
    match get_bytes(
        SYSTEM_OBJECT,
        ca::kAudioHardwarePropertyDevices,
        ca::kAudioObjectPropertyScopeGlobal,
    ) {
        Ok((buf, size)) => {
            // SAFETY: the property is an array of `size / 4` AudioObjectIDs.
            let ids = unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const AudioObjectID, size as usize / 4)
            };
            ids.to_vec()
        }
        Err(_) => Vec::new(),
    }
}

/// Channel count of each buffer of the device's streams in `scope`.
fn buffer_channels(device: AudioObjectID, scope: u32) -> Vec<usize> {
    let Ok((buf, size)) = get_bytes(device, ca::kAudioDevicePropertyStreamConfiguration, scope)
    else {
        return Vec::new();
    };
    if (size as usize) < size_of::<u32>() {
        return Vec::new();
    }
    // SAFETY: the property is an AudioBufferList of `mNumberBuffers` buffers within `size`
    // bytes; `buf` is 8-byte aligned like the struct.
    unsafe {
        let list = buf.as_ptr() as *const AudioBufferList;
        let n = (*list).mNumberBuffers as usize;
        let first = std::ptr::addr_of!((*list).mBuffers) as *const AudioBuffer;
        let end = first as usize + n * size_of::<AudioBuffer>();
        if end > buf.as_ptr() as usize + size as usize {
            return Vec::new();
        }
        std::slice::from_raw_parts(first, n)
            .iter()
            .map(|b| b.mNumberChannels as usize)
            .collect()
    }
}

fn channels(device: AudioObjectID, scope: u32) -> usize {
    buffer_channels(device, scope).iter().sum()
}

fn device_uid(device: AudioObjectID) -> Option<String> {
    get_string(
        device,
        ca::kAudioDevicePropertyDeviceUID,
        ca::kAudioObjectPropertyScopeGlobal,
    )
}

fn device_name(device: AudioObjectID) -> Option<String> {
    get_string(
        device,
        ca::kAudioObjectPropertyName,
        ca::kAudioObjectPropertyScopeGlobal,
    )
}

fn nominal_rate(device: AudioObjectID) -> Option<f64> {
    get::<f64>(
        device,
        ca::kAudioDevicePropertyNominalSampleRate,
        ca::kAudioObjectPropertyScopeGlobal,
    )
    .ok()
    .filter(|r| *r > 0.0)
}

fn is_alive(device: AudioObjectID) -> bool {
    get::<u32>(
        device,
        ca::kAudioDevicePropertyDeviceIsAlive,
        ca::kAudioObjectPropertyScopeGlobal,
    )
    .is_ok_and(|v| v != 0)
}

/// Whether some client runs the device's IO (for an output: something is playing to it).
/// Unknown counts as running.
fn running_somewhere(device: AudioObjectID) -> bool {
    !matches!(
        get::<u32>(
            device,
            ca::kAudioDevicePropertyDeviceIsRunningSomewhere,
            ca::kAudioObjectPropertyScopeGlobal,
        ),
        Ok(0)
    )
}

/// A tap whose output nothing plays to. Core Audio may then not call the tap's IOProc at all
/// (seen on macOS 27: no callbacks for as long as the output was idle), which is idle, not a
/// stall. The answer is cached for a second.
fn tap_idle(stream: &OpenStream, cache: &mut Option<(Instant, bool)>, now: Instant) -> bool {
    if stream.tap.is_none() {
        return false;
    }
    if let Some((at, idle)) = *cache
        && now.duration_since(at) < Duration::from_secs(1)
    {
        return idle;
    }
    let idle = !running_somewhere(stream.device);
    *cache = Some((now, idle));
    idle
}

fn scope_of(kind: DeviceKind) -> u32 {
    match kind {
        DeviceKind::Input => ca::kAudioObjectPropertyScopeInput,
        DeviceKind::Output => ca::kAudioObjectPropertyScopeOutput,
    }
}

fn kind_of(source: Source) -> DeviceKind {
    match source {
        Source::Mic => DeviceKind::Input,
        Source::System => DeviceKind::Output,
    }
}

fn default_device(kind: DeviceKind) -> Option<AudioObjectID> {
    let selector = match kind {
        DeviceKind::Input => ca::kAudioHardwarePropertyDefaultInputDevice,
        DeviceKind::Output => ca::kAudioHardwarePropertyDefaultOutputDevice,
    };
    get::<AudioObjectID>(SYSTEM_OBJECT, selector, ca::kAudioObjectPropertyScopeGlobal)
        .ok()
        .filter(|&d| d != 0)
}

fn find_device(uid: &str, kind: DeviceKind) -> Option<AudioObjectID> {
    device_ids()
        .into_iter()
        .find(|&d| channels(d, scope_of(kind)) > 0 && device_uid(d).as_deref() == Some(uid))
}

fn format_of(device: AudioObjectID, kind: DeviceKind) -> Option<StreamFormat> {
    let ch = channels(device, scope_of(kind)).min(StreamFormat::MAX_CHANNELS as usize);
    StreamFormat::new(nominal_rate(device)? as u32, ch as u16).ok()
}

/// Checks that every stream of `device` in `scope` delivers 32-bit float (the HAL's usual
/// virtual format; anything else is reported instead of misread).
fn check_float_streams(device: AudioObjectID, scope: u32) -> Result<(), String> {
    let Ok((buf, size)) = get_bytes(device, ca::kAudioDevicePropertyStreams, scope) else {
        return Ok(()); // no stream list: nothing to check
    };
    // SAFETY: the property is an array of AudioObjectIDs.
    let streams = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const AudioObjectID, size as usize / 4)
    };
    for &s in streams {
        let f: AudioStreamBasicDescription = get(
            s,
            ca::kAudioStreamPropertyVirtualFormat,
            ca::kAudioObjectPropertyScopeGlobal,
        )?;
        if f.mFormatID != kAudioFormatLinearPCM
            || f.mFormatFlags & kAudioFormatFlagIsFloat == 0
            || f.mBitsPerChannel != 32
        {
            return Err(format!(
                "unsupported stream format (id {}, flags {:#x}, {} bits); 32-bit float expected",
                status_text(f.mFormatID as i32),
                f.mFormatFlags,
                f.mBitsPerChannel
            ));
        }
    }
    Ok(())
}

pub fn list_devices(kind: DeviceKind) -> Result<Vec<DeviceInfo>> {
    let default = default_device(kind);
    Ok(device_ids()
        .into_iter()
        .filter(|&d| channels(d, scope_of(kind)) > 0)
        .filter_map(|d| {
            let id = device_uid(d)?;
            Some(DeviceInfo {
                is_default: default == Some(d),
                name: device_name(d).unwrap_or_else(|| id.clone()),
                mix_format: format_of(d, kind),
                id,
                kind,
            })
        })
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Property listeners
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Notice {
    DefaultInput,
    DefaultOutput,
    /// The device list changed.
    Devices,
    /// A watched device's "is alive" changed.
    Alive,
}

unsafe extern "C-unwind" fn on_property(
    _object: AudioObjectID,
    count: u32,
    addresses: NonNull<AudioObjectPropertyAddress>,
    client: *mut c_void,
) -> i32 {
    // SAFETY: `client` is the Sender registered by `Listeners`, alive until it unregisters;
    // `addresses` holds `count` entries for the duration of the call.
    let (tx, addresses) = unsafe {
        (
            &*(client as *const Sender<Notice>),
            std::slice::from_raw_parts(addresses.as_ptr(), count as usize),
        )
    };
    for a in addresses {
        let n = match a.mSelector {
            ca::kAudioHardwarePropertyDefaultInputDevice => Notice::DefaultInput,
            ca::kAudioHardwarePropertyDefaultOutputDevice => Notice::DefaultOutput,
            ca::kAudioHardwarePropertyDevices => Notice::Devices,
            ca::kAudioDevicePropertyDeviceIsAlive => Notice::Alive,
            _ => continue,
        };
        let _ = tx.send(n);
    }
    0
}

/// Registered property listeners, removed on drop.
struct Listeners {
    tx: *mut Sender<Notice>,
    registered: Vec<(AudioObjectID, AudioObjectPropertyAddress)>,
}

impl Listeners {
    fn new(tx: Sender<Notice>) -> Self {
        Self {
            tx: Box::into_raw(Box::new(tx)),
            registered: Vec::new(),
        }
    }

    fn add(&mut self, object: AudioObjectID, selector: u32) {
        let a = address(selector, ca::kAudioObjectPropertyScopeGlobal);
        // SAFETY: `self.tx` stays valid until the listener is removed in `remove`/`drop`.
        let status = unsafe {
            ca::AudioObjectAddPropertyListener(
                object,
                NonNull::from(&a),
                Some(on_property),
                self.tx as *mut c_void,
            )
        };
        if status == 0 {
            self.registered.push((object, a));
        }
    }

    /// Removes the listeners on `object`.
    fn remove(&mut self, object: AudioObjectID) {
        let tx = self.tx;
        self.registered.retain(|(o, a)| {
            if *o != object {
                return true;
            }
            // SAFETY: removes a listener registered above with the same arguments.
            unsafe {
                ca::AudioObjectRemovePropertyListener(
                    *o,
                    NonNull::from(a),
                    Some(on_property),
                    tx as *mut c_void,
                );
            }
            false
        });
    }
}

impl Drop for Listeners {
    fn drop(&mut self) {
        for (o, a) in std::mem::take(&mut self.registered) {
            // SAFETY: as in `remove`.
            unsafe {
                ca::AudioObjectRemovePropertyListener(
                    o,
                    NonNull::from(&a),
                    Some(on_property),
                    self.tx as *mut c_void,
                );
            }
        }
        // SAFETY: allocated in `new`; no listener refers to it any more.
        drop(unsafe { Box::from_raw(self.tx) });
    }
}

// ---------------------------------------------------------------------------------------------
// IOProc: real-time thread -> rings
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Meta {
    host_ticks: u64,
    sample_time: f64,
    frames: u32,
    host_valid: bool,
    sample_valid: bool,
}

/// Owned by the IOProc (only it touches the producers).
struct IoState {
    samples: rtrb::Producer<f32>,
    metas: rtrb::Producer<Meta>,
    /// Input buffers to skip (an aggregate lists its sub-device's own inputs before the tap).
    skip_buffers: usize,
    /// Interleaved channels delivered (sum over the used buffers).
    channels: usize,
    overflows: Arc<AtomicU64>,
    callbacks: Arc<AtomicU64>,
}

unsafe extern "C-unwind" fn input_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    // SAFETY: `client` is the IoState owned by the OpenStream, which destroys the IOProc
    // before freeing it; Core Audio calls the IOProc from one thread at a time. The buffer list
    // and time stamp are valid for the duration of the call.
    unsafe {
        let st = &mut *(client as *mut IoState);
        st.callbacks.fetch_add(1, Ordering::Relaxed);
        let list = input.as_ptr();
        let n = (*list).mNumberBuffers as usize;
        let buffers = std::slice::from_raw_parts(
            std::ptr::addr_of!((*list).mBuffers) as *const AudioBuffer,
            n,
        );
        let used = buffers.get(st.skip_buffers..).unwrap_or(&[]);
        let Some(first) = used.first() else {
            return 0;
        };
        let frames = if first.mNumberChannels == 0 {
            0
        } else {
            first.mDataByteSize as usize / (4 * first.mNumberChannels as usize)
        };
        let channels: usize = used.iter().map(|b| b.mNumberChannels as usize).sum();
        if frames == 0 || channels != st.channels {
            return 0;
        }
        if st.samples.slots() < frames * channels || st.metas.slots() < 1 {
            st.overflows.fetch_add(1, Ordering::Relaxed);
            return 0;
        }
        let Ok(chunk) = st.samples.write_chunk_uninit(frames * channels) else {
            st.overflows.fetch_add(1, Ordering::Relaxed);
            return 0;
        };
        // Interleave across buffers: frame by frame, buffer by buffer, channel by channel.
        let iter = (0..frames).flat_map(|f| {
            used.iter().flat_map(move |b| {
                let nch = b.mNumberChannels as usize;
                let data = b.mData as *const f32;
                let len = b.mDataByteSize as usize / 4;
                (0..nch).map(move |c| {
                    let i = f * nch + c;
                    if data.is_null() || i >= len {
                        0.0
                    } else {
                        *data.add(i)
                    }
                })
            })
        });
        chunk.fill_from_iter(iter);
        let t = &*input_time.as_ptr();
        let _ = st.metas.push(Meta {
            host_ticks: t.mHostTime,
            sample_time: t.mSampleTime,
            frames: frames as u32,
            host_valid: t.mFlags.contains(AudioTimeStampFlags::HostTimeValid),
            sample_valid: t.mFlags.contains(AudioTimeStampFlags::SampleTimeValid),
        });
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Process tap (macOS 14.2+), resolved at run time
// ---------------------------------------------------------------------------------------------

type CreateTapFn = unsafe extern "C" fn(*const c_void, *mut AudioObjectID) -> i32;
type DestroyTapFn = unsafe extern "C" fn(AudioObjectID) -> i32;

fn tap_functions() -> Option<(CreateTapFn, DestroyTapFn)> {
    // SAFETY: dlsym with NUL-terminated names; the symbols, when present, have these C
    // signatures (CoreAudio/AudioHardware.h, macOS 14.2+).
    unsafe {
        let create = libc::dlsym(
            libc::RTLD_DEFAULT,
            c"AudioHardwareCreateProcessTap".as_ptr(),
        );
        let destroy = libc::dlsym(
            libc::RTLD_DEFAULT,
            c"AudioHardwareDestroyProcessTap".as_ptr(),
        );
        if create.is_null() || destroy.is_null() {
            return None;
        }
        Some((
            std::mem::transmute::<*mut c_void, CreateTapFn>(create),
            std::mem::transmute::<*mut c_void, DestroyTapFn>(destroy),
        ))
    }
}

fn macos_at_least(major: isize, minor: isize) -> bool {
    let v = NSProcessInfo::processInfo().operatingSystemVersion();
    (v.majorVersion, v.minorVersion) >= (major, minor)
}

/// A process tap and the private aggregate device it is read through.
struct Tap {
    tap: AudioObjectID,
    aggregate: AudioObjectID,
    destroy_tap: DestroyTapFn,
}

impl Drop for Tap {
    fn drop(&mut self) {
        // SAFETY: both objects were created by `Tap::new` and are destroyed once.
        unsafe {
            ca::AudioHardwareDestroyAggregateDevice(self.aggregate);
            (self.destroy_tap)(self.tap);
        }
    }
}

fn ns(s: &str) -> Retained<NSObject> {
    NSString::from_str(s).into_super()
}

fn key(k: &CStr) -> Retained<NSString> {
    NSString::from_str(k.to_str().unwrap_or_default())
}

impl Tap {
    /// A global stereo tap (all processes) on an aggregate clocked by `output` (UID).
    fn new(output_uid: &str) -> Result<Self, OpenError> {
        if !macos_at_least(14, 2) {
            return Err(OpenError::Unsupported(
                "system audio capture needs macOS 14.2 or later (Core Audio process taps)".into(),
            ));
        }
        diag("tap: macOS version ok");
        let (create, destroy) = tap_functions().ok_or_else(|| {
            OpenError::Unsupported("AudioHardwareCreateProcessTap is not available".into())
        })?;
        let none: Retained<NSArray<NSNumber>> = NSArray::new();
        // SAFETY: CATapDescription exists on macOS 14.2+ (checked above); the setters take
        // plain values.
        let desc = unsafe {
            let d = ca::CATapDescription::initStereoGlobalTapButExcludeProcesses(
                ca::CATapDescription::alloc(),
                &none,
            );
            d.setPrivate(true);
            d.setMuteBehavior(ca::CATapMuteBehavior::Unmuted);
            d.setName(&NSString::from_str("SonicDiar system audio"));
            d
        };
        diag("tap: description made");
        // SAFETY: plain getter.
        let tap_uid = unsafe { desc.UUID() }.UUIDString().to_string();
        let mut tap: AudioObjectID = 0;
        // SAFETY: `desc` is a valid CATapDescription for the call; `tap` is writable.
        let status = unsafe { create(Retained::as_ptr(&desc) as *const c_void, &mut tap) };
        check(status, "AudioHardwareCreateProcessTap").map_err(OpenError::Other)?;
        diag(&format!("tap: created {tap} ({tap_uid})"));

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let aggregate_uid = format!(
            "sonicdiar.tap.{}.{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let sub_device: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[&*key(ca::kAudioSubDeviceUIDKey)],
                &[ns(output_uid)],
            );
        let tap_entry: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[
                    &*key(ca::kAudioSubTapUIDKey),
                    &*key(ca::kAudioSubTapDriftCompensationKey),
                ],
                &[
                    ns(&tap_uid),
                    NSNumber::new_bool(true).into_super().into_super(),
                ],
            );
        let sub_list = NSArray::from_retained_slice(&[sub_device]).into_super();
        let tap_list = NSArray::from_retained_slice(&[tap_entry]).into_super();
        let description: Retained<NSDictionary<NSString, NSObject>> =
            NSDictionary::from_retained_objects(
                &[
                    &*key(ca::kAudioAggregateDeviceNameKey),
                    &*key(ca::kAudioAggregateDeviceUIDKey),
                    &*key(ca::kAudioAggregateDeviceMainSubDeviceKey),
                    &*key(ca::kAudioAggregateDeviceIsPrivateKey),
                    &*key(ca::kAudioAggregateDeviceTapAutoStartKey),
                    &*key(ca::kAudioAggregateDeviceSubDeviceListKey),
                    &*key(ca::kAudioAggregateDeviceTapListKey),
                ],
                &[
                    ns("SonicDiar system audio"),
                    ns(&aggregate_uid),
                    ns(output_uid),
                    NSNumber::new_bool(true).into_super().into_super(),
                    NSNumber::new_bool(true).into_super().into_super(),
                    sub_list,
                    tap_list,
                ],
            );
        let mut aggregate: AudioObjectID = 0;
        // SAFETY: NSDictionary is toll-free bridged with CFDictionary; it outlives the call.
        let status = unsafe {
            let cf = &*(Retained::as_ptr(&description) as *const CFDictionary);
            ca::AudioHardwareCreateAggregateDevice(cf, NonNull::from(&mut aggregate))
        };
        diag(&format!("tap: aggregate status {status}, id {aggregate}"));
        if let Err(e) = check(status, "AudioHardwareCreateAggregateDevice") {
            // SAFETY: created above.
            unsafe { destroy(tap) };
            return Err(OpenError::Other(e));
        }
        Ok(Self {
            tap,
            aggregate,
            destroy_tap: destroy,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MicPermission {
    Granted,
    /// Not asked yet: a request was made; audio is silent until the user answers.
    Requested,
    Denied,
}

fn mic_permission() -> MicPermission {
    use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};
    // SAFETY: AVMediaTypeAudio is a framework constant; the class methods are thread-safe.
    unsafe {
        let Some(audio) = AVMediaTypeAudio else {
            return MicPermission::Granted; // cannot tell; let the HAL decide
        };
        match AVCaptureDevice::authorizationStatusForMediaType(audio) {
            AVAuthorizationStatus::Authorized => MicPermission::Granted,
            AVAuthorizationStatus::NotDetermined => {
                let handler = block2::RcBlock::new(|_granted: objc2::runtime::Bool| {});
                AVCaptureDevice::requestAccessForMediaType_completionHandler(audio, &handler);
                MicPermission::Requested
            }
            _ => MicPermission::Denied,
        }
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
    ctrl: Sender<Control>,
}

impl Handle {
    pub fn set_device(&self, device: DeviceSelector) {
        let _ = self.ctrl.send(Control::SetDevice(device));
    }

    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Release);
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
    if cfg.format.channels > 2 {
        return Err(CaptureError::Config(
            "capture format must be mono or stereo".into(),
        ));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded();
    let ctx = ThreadCtx {
        // Core Audio delivers continuously while a device runs: a pause is not silence.
        feed: Feed::new(
            &cfg,
            producer,
            stats.clone(),
            Idle {
                after_ms: 500,
                margin_ms: 500,
            },
        ),
        cfg: cfg.clone(),
        events,
        stats,
        stop: stop.clone(),
        ctrl: ctrl_rx,
    };
    let thread = std::thread::Builder::new()
        .name(format!("sd-capture-{}", cfg.source))
        .spawn(move || ctx.run())
        .map_err(|e| CaptureError::Thread(e.to_string()))?;
    Ok(Handle {
        thread: Some(thread),
        stop,
        ctrl: ctrl_tx,
    })
}

struct ThreadCtx {
    cfg: CaptureConfig,
    feed: Feed,
    events: Sender<CaptureEvent>,
    stats: Arc<CaptureStats>,
    stop: Arc<AtomicBool>,
    ctrl: Receiver<Control>,
}

struct OpenStream {
    /// The device the IOProc runs on (the mic, or the tap's aggregate).
    io_device: AudioObjectID,
    proc_id: ca::AudioDeviceIOProcID,
    io: *mut IoState,
    samples: rtrb::Consumer<f32>,
    metas: rtrb::Consumer<Meta>,
    channels: usize,
    /// The physical device (mic or output), watched for removal.
    device: AudioObjectID,
    device_id: String,
    converter: Converter,
    next_sample_time: Option<f64>,
    overflows: Arc<AtomicU64>,
    overflows_seen: u64,
    callbacks: Arc<AtomicU64>,
    /// Kept alive while the stream runs (dropped after the IOProc).
    tap: Option<Tap>,
    /// Seconds of exact silence so far (taps only).
    silent_s: f64,
    hinted: bool,
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        // SAFETY: stops and destroys the IOProc created in `open`, then frees its state, which
        // the IOProc can no longer reach. The tap (a field) is dropped afterwards.
        unsafe {
            ca::AudioDeviceStop(self.io_device, self.proc_id);
            ca::AudioDeviceDestroyIOProcID(self.io_device, self.proc_id);
            drop(Box::from_raw(self.io));
        }
    }
}

enum OpenError {
    NoDevice,
    Unsupported(String),
    PermissionDenied(String),
    Other(String),
    /// Capture was stopped while waiting.
    Stopped,
}

enum StreamExit {
    Stop,
    Switch(SwitchReason),
    Failed(String),
}

struct Session {
    preferred: DeviceSelector,
    on_fallback: bool,
    recover_requested: bool,
    stall_injected: bool,
    notices: Receiver<Notice>,
    listeners: Listeners,
}

impl ThreadCtx {
    fn emit(&self, e: CaptureEvent) {
        let _ = self.events.send(e);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn run(mut self) {
        let (ntx, nrx) = crossbeam_channel::unbounded();
        let mut listeners = Listeners::new(ntx);
        listeners.add(
            SYSTEM_OBJECT,
            match self.cfg.source {
                Source::Mic => ca::kAudioHardwarePropertyDefaultInputDevice,
                Source::System => ca::kAudioHardwarePropertyDefaultOutputDevice,
            },
        );
        listeners.add(SYSTEM_OBJECT, ca::kAudioHardwarePropertyDevices);
        let mut sess = Session {
            preferred: self.cfg.device.clone(),
            on_fallback: false,
            recover_requested: false,
            stall_injected: false,
            notices: nrx,
            listeners,
        };
        // Both sources start their timeline at capture start, so frame 0 means the same instant.
        self.feed.start_at(HostTime::now());
        let mut attempt = 0u32;
        let mut reported = false;
        while !self.stopped() {
            match self.open(&mut sess) {
                Ok(mut stream) => {
                    attempt = 0;
                    reported = false;
                    let exit = self.pump(&mut stream, &mut sess);
                    sess.listeners.remove(stream.device);
                    drop(stream);
                    match exit {
                        StreamExit::Stop => break,
                        StreamExit::Switch(reason) => {
                            self.stats.restarts.fetch_add(1, Ordering::Relaxed);
                            self.emit(CaptureEvent::Switching { reason });
                        }
                        StreamExit::Failed(error) => {
                            attempt += 1;
                            let delay = backoff_ms(attempt);
                            self.emit(CaptureEvent::Retrying {
                                attempt,
                                retry_in_ms: delay,
                                error,
                            });
                            self.idle_wait(&mut sess, delay);
                        }
                    }
                }
                Err(OpenError::NoDevice) => {
                    if !reported {
                        self.emit(CaptureEvent::NoDevice);
                        reported = true;
                    }
                    attempt += 1;
                    self.idle_wait(&mut sess, backoff_ms(attempt).min(1_000));
                }
                Err(OpenError::PermissionDenied(what)) => {
                    if !reported {
                        self.emit(CaptureEvent::PermissionNeeded {
                            what,
                            confirmed: true,
                        });
                        reported = true;
                    }
                    // Keep the timeline running and look again now and then.
                    self.idle_wait(&mut sess, 2_000);
                }
                Err(OpenError::Stopped) => break,
                Err(OpenError::Unsupported(e)) => {
                    if !reported {
                        self.emit(CaptureEvent::Retrying {
                            attempt: 0,
                            retry_in_ms: 0,
                            error: e,
                        });
                        reported = true;
                    }
                    self.idle_wait(&mut sess, 5_000);
                }
                Err(OpenError::Other(error)) => {
                    attempt += 1;
                    let delay = backoff_ms(attempt);
                    self.emit(CaptureEvent::Retrying {
                        attempt,
                        retry_in_ms: delay,
                        error,
                    });
                    self.idle_wait(&mut sess, delay);
                }
            }
        }
        drop(sess);
        self.emit(CaptureEvent::Stopped);
    }

    fn open(&mut self, sess: &mut Session) -> Result<OpenStream, OpenError> {
        let kind = kind_of(self.cfg.source);
        while sess.notices.try_recv().is_ok() {} // the state is read fresh below
        if self.cfg.source == Source::Mic && mic_permission() == MicPermission::Denied {
            return Err(OpenError::PermissionDenied(
                "microphone (System Settings > Privacy & Security > Microphone)".into(),
            ));
        }
        let preferred = match &sess.preferred {
            DeviceSelector::Id(id) => Some(id.clone()),
            DeviceSelector::Default => None,
        };
        let found = preferred.as_deref().and_then(|id| find_device(id, kind));
        let fallback = preferred.is_some() && found.is_none();
        if fallback && !sess.on_fallback {
            self.emit(CaptureEvent::Switching {
                reason: SwitchReason::DeviceRemoved,
            });
        }
        sess.on_fallback = fallback;
        let device = found
            .or_else(|| default_device(kind))
            .ok_or(OpenError::NoDevice)?;
        let uid = device_uid(device).ok_or(OpenError::NoDevice)?;

        // Where the IOProc runs and which of its input buffers carry our audio.
        let (tap, io_device, skip_buffers) = match self.cfg.source {
            Source::Mic => (None, device, 0),
            Source::System => {
                let tap = self.create_tap(&uid)?;
                // The aggregate lists the output device's own inputs (if any) before the tap.
                let skip = buffer_channels(device, ca::kAudioObjectPropertyScopeInput).len();
                let agg = tap.aggregate;
                (Some(tap), agg, skip)
            }
        };
        let input_buffers = buffer_channels(io_device, ca::kAudioObjectPropertyScopeInput);
        let channels: usize = input_buffers.iter().skip(skip_buffers).sum();
        if channels == 0 {
            return Err(OpenError::Other(format!("{uid}: no input channels")));
        }
        check_float_streams(io_device, ca::kAudioObjectPropertyScopeInput)
            .map_err(|e| OpenError::Other(format!("{uid}: {e}")))?;
        let rate = nominal_rate(io_device)
            .ok_or_else(|| OpenError::Other(format!("{uid}: no sample rate")))?;

        // Two seconds of device audio between the real-time thread and this one.
        let (samples_tx, samples) = rtrb::RingBuffer::<f32>::new(rate as usize * 2 * channels);
        let (metas_tx, metas) = rtrb::RingBuffer::<Meta>::new(1_024);
        let overflows = Arc::new(AtomicU64::new(0));
        let callbacks = Arc::new(AtomicU64::new(0));
        let io = Box::into_raw(Box::new(IoState {
            samples: samples_tx,
            metas: metas_tx,
            skip_buffers,
            channels,
            overflows: overflows.clone(),
            callbacks: callbacks.clone(),
        }));
        let mut proc_id: ca::AudioDeviceIOProcID = None;
        // SAFETY: `io` lives until the OpenStream destroys the IOProc.
        let status = unsafe {
            ca::AudioDeviceCreateIOProcID(
                io_device,
                Some(input_proc),
                io as *mut c_void,
                NonNull::from(&mut proc_id),
            )
        };
        if let Err(e) = check(status, "AudioDeviceCreateIOProcID") {
            // SAFETY: never handed to Core Audio.
            drop(unsafe { Box::from_raw(io) });
            return Err(OpenError::Other(e));
        }
        let stream = OpenStream {
            io_device,
            proc_id,
            io,
            samples,
            metas,
            channels,
            device,
            device_id: uid.clone(),
            converter: Converter::new(
                rate.round() as u32,
                channels,
                self.cfg.format.sample_rate,
                self.cfg.format.channels as usize,
            ),
            next_sample_time: None,
            overflows,
            overflows_seen: 0,
            callbacks,
            tap,
            silent_s: 0.0,
            hinted: false,
        };
        diag(&format!(
            "open: {uid}: io device {io_device}, {channels} channels at {rate} Hz, skip {skip_buffers} buffers"
        ));
        // SAFETY: starts the IOProc created above (stopped again in OpenStream::drop).
        let status = unsafe { ca::AudioDeviceStart(io_device, proc_id) };
        check(status, "AudioDeviceStart").map_err(OpenError::Other)?;
        diag("open: started");
        sess.listeners
            .add(device, ca::kAudioDevicePropertyDeviceIsAlive);
        self.emit(CaptureEvent::Started {
            device_name: device_name(device).unwrap_or_else(|| uid.clone()),
            mix_format: format_of(device, kind),
            device_id: uid,
            raw: false, // Core Audio input is unprocessed unless voice processing is enabled
        });
        Ok(stream)
    }

    /// Creates the tap on a helper thread: macOS blocks `AudioHardwareCreateProcessTap` until
    /// the user answers the System Audio Recording prompt. Meanwhile the timeline keeps running,
    /// a missing permission is reported after [`TAP_PROMPT_HINT`], and capture can stop at any
    /// time (a tap that arrives after that is destroyed by the helper).
    fn create_tap(&mut self, output_uid: &str) -> Result<Tap, OpenError> {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let uid = output_uid.to_string();
        std::thread::Builder::new()
            .name("sd-capture-tap".into())
            .spawn(move || {
                // If the capture has gone, the send fails and the tap is dropped (destroyed).
                let _ = tx.send(Tap::new(&uid));
            })
            .map_err(|e| OpenError::Other(format!("tap thread: {e}")))?;
        let started = Instant::now();
        let mut hinted = false;
        loop {
            match rx.recv_timeout(TICK) {
                Ok(result) => return result,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    return Err(OpenError::Other("tap thread failed".into()));
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if self.stopped() {
                        return Err(OpenError::Stopped);
                    }
                    self.feed.fill_idle();
                    if !hinted && started.elapsed() >= TAP_PROMPT_HINT {
                        hinted = true;
                        self.emit(CaptureEvent::PermissionNeeded {
                            what: "system audio recording: macOS is waiting for it to be allowed (System Settings > Privacy & Security > Screen & System Audio Recording)".into(),
                            confirmed: false,
                        });
                    }
                }
            }
        }
    }

    /// Waits up to `ms` while keeping the timeline running with silence; returns early on stop,
    /// a control message, or a device change.
    fn idle_wait(&mut self, sess: &mut Session, ms: u64) {
        let deadline = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < deadline && !self.stopped() {
            std::thread::sleep(TICK);
            self.feed.fill_idle();
            let mut wake = false;
            while sess.notices.try_recv().is_ok() {
                wake = true;
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

    fn pump(&mut self, stream: &mut OpenStream, sess: &mut Session) -> StreamExit {
        self.feed.new_stream();
        let started = Instant::now();
        let mut watchdog = StallWatchdog::new(self.cfg.stall_timeout_ms, started);
        let inject_at = self
            .cfg
            .test_stall_after_ms
            .filter(|_| !sess.stall_injected)
            .map(|ms| started + Duration::from_millis(ms as u64));
        let mut pending_default_switch: Option<Instant> = None;
        let settle = Duration::from_millis(self.cfg.default_switch_settle_ms as u64);
        let kind = kind_of(self.cfg.source);
        let mut idle_cache = None;
        loop {
            if self.stopped() {
                return StreamExit::Stop;
            }
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
                    Notice::DefaultInput | Notice::DefaultOutput => {
                        let follows =
                            matches!(sess.preferred, DeviceSelector::Default) || sess.on_fallback;
                        if follows && default_device(kind).is_some_and(|d| d != stream.device) {
                            pending_default_switch = Some(Instant::now() + settle);
                        }
                    }
                    Notice::Devices | Notice::Alive => {
                        if !is_alive(stream.device) || !device_ids().contains(&stream.device) {
                            return StreamExit::Switch(SwitchReason::DeviceRemoved);
                        }
                        if sess.on_fallback
                            && let DeviceSelector::Id(p) = &sess.preferred
                            && find_device(p, kind).is_some()
                        {
                            return StreamExit::Switch(SwitchReason::Requested);
                        }
                    }
                }
            }
            std::thread::sleep(TICK);
            let now = Instant::now();
            let injected = inject_at.is_some_and(|t| now >= t);
            if injected {
                sess.stall_injected = true;
            }
            let packets = if injected {
                0
            } else {
                self.drain(stream, sess)
            };
            if packets > 0 {
                watchdog.packet(now);
                if sess.recover_requested {
                    sess.recover_requested = false;
                    return StreamExit::Switch(SwitchReason::Recovery);
                }
            } else {
                if let Some(quiet) = watchdog.stalled_for(now) {
                    if tap_idle(stream, &mut idle_cache, now) {
                        watchdog.packet(now);
                    } else {
                        self.emit(CaptureEvent::Stalled {
                            quiet_ms: quiet.as_millis() as u64,
                        });
                        return StreamExit::Switch(SwitchReason::Recovery);
                    }
                }
                self.feed.fill_idle();
            }
            if pending_default_switch.is_some_and(|t| Instant::now() >= t) {
                return StreamExit::Switch(SwitchReason::DefaultDeviceChanged);
            }
            if stream.callbacks.load(Ordering::Relaxed) == 0
                && now.duration_since(started) > Duration::from_secs(5)
                && !tap_idle(stream, &mut idle_cache, now)
            {
                return StreamExit::Failed(format!(
                    "{}: the device never called back",
                    stream.device_id
                ));
            }
        }
    }

    /// Moves everything the IOProc delivered onto the timeline. Returns the packets read.
    fn drain(&mut self, s: &mut OpenStream, sess: &mut Session) -> usize {
        let mut packets = 0;
        let overflows = s.overflows.load(Ordering::Relaxed);
        let mut glitch = overflows != s.overflows_seen;
        s.overflows_seen = overflows;
        let rate = s.converter_rate();
        let mut samples = Vec::new();
        while let Ok(meta) = s.metas.pop() {
            packets += 1;
            let n = meta.frames as usize * s.channels;
            samples.clear();
            if let Ok(chunk) = s.samples.read_chunk(n) {
                let (a, b) = chunk.as_slices();
                samples.extend_from_slice(a);
                samples.extend_from_slice(b);
                chunk.commit_all();
            } else {
                // Cannot happen (samples are written before their meta); resynchronize.
                glitch = true;
                continue;
            }
            // Device positions: a jump means the device lost or skipped audio.
            if meta.sample_valid {
                if let Some(next) = s.next_sample_time
                    && (meta.sample_time - next).abs() > 1.5
                {
                    glitch = true;
                }
                s.next_sample_time = Some(meta.sample_time + meta.frames as f64);
            }
            let host_ns = if meta.host_valid {
                // SAFETY: pure conversion of a mach host time.
                unsafe { ca::AudioConvertHostTimeToNanos(meta.host_ticks) }
            } else {
                HostTime::now()
                    .as_ns()
                    .saturating_sub((meta.frames as f64 * 1e9 / rate) as u64)
            };
            // Taps without permission deliver exact zeros.
            if s.tap.is_some() && !s.hinted {
                if samples.iter().all(|&v| v == 0.0) {
                    s.silent_s += meta.frames as f64 / rate;
                    if s.silent_s >= TAP_SILENCE_HINT_S {
                        s.hinted = true;
                        self.emit(CaptureEvent::PermissionNeeded {
                            what: "system audio recording, if audio was playing (System Settings > Privacy & Security > Screen & System Audio Recording)".into(),
                            confirmed: false,
                        });
                    }
                } else {
                    s.hinted = true; // real audio: permission is there
                }
            }
            let mut flags = BlockFlags::empty();
            if glitch {
                flags.insert(BlockFlags::DISCONTINUITY);
            }
            if !meta.host_valid {
                flags.insert(BlockFlags::TIMESTAMP_ESTIMATED);
            }
            let feed = &mut self.feed;
            let mut exhausted = None;
            s.converter.push(host_ns, &samples, |chunk| {
                if let Some(ms) = feed.packet(Packet {
                    t: HostTime::from_ns(chunk.host_ns),
                    frames: chunk.frames as u32,
                    position: Some(chunk.position as f64),
                    flags,
                    data: Some(chunk.data),
                }) {
                    exhausted = Some(ms);
                }
                flags = BlockFlags::empty(); // a discontinuity applies to the first chunk only
            });
            if flags.bits() != 0 {
                // No chunk was emitted yet: carry the flag to the next packet.
                glitch = flags.contains(BlockFlags::DISCONTINUITY);
            } else {
                glitch = false;
            }
            if let Some(filled_ms) = exhausted {
                self.emit(CaptureEvent::GapFillBudgetExhausted { filled_ms });
                sess.recover_requested = true;
            }
        }
        packets
    }
}

impl OpenStream {
    fn converter_rate(&self) -> f64 {
        nominal_rate(self.io_device).unwrap_or(48_000.0)
    }
}

// ---------------------------------------------------------------------------------------------
// Playback (test signals)
// ---------------------------------------------------------------------------------------------

struct PlayState {
    /// Interleaved, `channels` per frame, at the device rate.
    data: Vec<f32>,
    channels: usize,
    position: AtomicUsize,
    /// (output host ticks, frame index at that time) per callback.
    stamps: rtrb::Producer<(u64, u64)>,
    /// Device sample time expected for the next callback.
    next_sample_time: Option<f64>,
    /// Callbacks whose output time jumped: the device played something else (silence) in
    /// between, so the played signal has a gap.
    skips: AtomicU64,
}

unsafe extern "C-unwind" fn output_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    _input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    output: NonNull<AudioBufferList>,
    output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    // SAFETY: `client` is the PlayState kept alive by `play_blocking` until the IOProc is
    // destroyed; the buffer list and time stamp are valid for the call.
    unsafe {
        let st = &mut *(client as *mut PlayState);
        let list = output.as_ptr();
        let n = (*list).mNumberBuffers as usize;
        let buffers = std::slice::from_raw_parts_mut(
            std::ptr::addr_of_mut!((*list).mBuffers) as *mut AudioBuffer,
            n,
        );
        let total = st.data.len() / st.channels;
        let pos = st.position.load(Ordering::Relaxed);
        let mut frames = 0;
        let mut channel_base = 0;
        for b in buffers.iter_mut() {
            let nch = b.mNumberChannels as usize;
            if nch == 0 || b.mData.is_null() {
                continue;
            }
            let out =
                std::slice::from_raw_parts_mut(b.mData as *mut f32, b.mDataByteSize as usize / 4);
            frames = out.len() / nch;
            for f in 0..frames {
                for c in 0..nch {
                    let g = channel_base + c;
                    out[f * nch + c] = if g < st.channels && pos + f < total {
                        st.data[(pos + f) * st.channels + g]
                    } else {
                        0.0
                    };
                }
            }
            channel_base += nch;
        }
        let t = &*output_time.as_ptr();
        if t.mFlags.contains(AudioTimeStampFlags::HostTimeValid) {
            let _ = st.stamps.push((t.mHostTime, pos as u64));
        }
        if t.mFlags.contains(AudioTimeStampFlags::SampleTimeValid) {
            if let Some(next) = st.next_sample_time
                && (t.mSampleTime - next).abs() > 1.5
                && pos > 0
                && pos < total
            {
                st.skips.fetch_add(1, Ordering::Relaxed);
            }
            st.next_sample_time = Some(t.mSampleTime + frames as f64);
        }
        st.position.store(pos + frames, Ordering::Relaxed);
    }
    0
}

pub fn play_blocking(
    device: &DeviceSelector,
    format: StreamFormat,
    samples: &[f32],
) -> Result<PlaybackReport> {
    let ch = format.channels as usize;
    if !samples.len().is_multiple_of(ch) || ch > 2 {
        return Err(CaptureError::Config(
            "samples must be whole mono or stereo frames".into(),
        ));
    }
    let total = (samples.len() / ch) as u64;
    let dev = match device {
        DeviceSelector::Id(id) => find_device(id, DeviceKind::Output),
        DeviceSelector::Default => default_device(DeviceKind::Output),
    }
    .ok_or_else(|| CaptureError::Os("no output device".into()))?;
    check_float_streams(dev, ca::kAudioObjectPropertyScopeOutput).map_err(CaptureError::Os)?;
    let rate = nominal_rate(dev).ok_or_else(|| CaptureError::Os("no sample rate".into()))?;
    let out_ch = channels(dev, ca::kAudioObjectPropertyScopeOutput).clamp(1, 2);
    // Convert the signal to the device's rate and (up to two) channels.
    let mut conv = Converter::new(format.sample_rate, ch, rate.round() as u32, out_ch);
    let mut data = Vec::with_capacity(samples.len() * 2);
    conv.push(0, samples, |c| data.extend_from_slice(c.data));
    // Flush the resampler's last partial chunk with silence.
    let pad = vec![0.0; format.sample_rate as usize / 10 * ch];
    conv.push(0, &pad, |c| data.extend_from_slice(c.data));
    let (stamps_tx, mut stamps) = rtrb::RingBuffer::new(4_096);
    let state = Box::into_raw(Box::new(PlayState {
        data,
        channels: out_ch,
        position: AtomicUsize::new(0),
        stamps: stamps_tx,
        next_sample_time: None,
        skips: AtomicU64::new(0),
    }));
    let mut proc_id: ca::AudioDeviceIOProcID = None;
    // SAFETY: `state` lives until the IOProc is destroyed below.
    let status = unsafe {
        ca::AudioDeviceCreateIOProcID(
            dev,
            Some(output_proc),
            state as *mut c_void,
            NonNull::from(&mut proc_id),
        )
    };
    if let Err(e) = check(status, "AudioDeviceCreateIOProcID") {
        // SAFETY: never handed to Core Audio.
        drop(unsafe { Box::from_raw(state) });
        return Err(CaptureError::Os(e));
    }
    // SAFETY: starts the IOProc created above.
    let status = unsafe { ca::AudioDeviceStart(dev, proc_id) };
    let mut offsets = Vec::new();
    let end_frames = (total as f64 * rate / format.sample_rate as f64 + rate * 0.05) as usize;
    let deadline =
        Instant::now() + Duration::from_secs_f64(total as f64 / format.sample_rate as f64 + 5.0);
    if status == 0 {
        loop {
            std::thread::sleep(TICK);
            while let Ok((ticks, frame)) = stamps.pop() {
                // SAFETY: pure conversion.
                let host = unsafe { ca::AudioConvertHostTimeToNanos(ticks) } as f64;
                offsets.push(host - frame as f64 * 1e9 / rate);
            }
            // SAFETY: `state` is alive; only the atomic is read.
            let pos = unsafe { (*state).position.load(Ordering::Relaxed) };
            if pos >= end_frames || Instant::now() > deadline {
                break;
            }
        }
    }
    // SAFETY: stops and destroys the IOProc, after which `state` is unreachable and freed.
    unsafe {
        ca::AudioDeviceStop(dev, proc_id);
        ca::AudioDeviceDestroyIOProcID(dev, proc_id);
    }
    // SAFETY: the IOProc is destroyed, so nothing else refers to `state`.
    let skips = unsafe { Box::from_raw(state) }
        .skips
        .load(Ordering::Relaxed) as u32;
    check(status, "AudioDeviceStart").map_err(CaptureError::Os)?;
    if offsets.is_empty() {
        return Err(CaptureError::Os("output clock never advanced".into()));
    }
    offsets.sort_by(f64::total_cmp);
    Ok(PlaybackReport {
        first_frame_host_time: HostTime::from_ns(offsets[offsets.len() / 2].max(0.0) as u64),
        frames: total,
        clock_samples: offsets.len(),
        // The IOProc fills from memory and cannot run dry, but the device may skip a cycle.
        underruns: skips,
    })
}

fn backoff_ms(attempt: u32) -> u64 {
    (100u64 << attempt.saturating_sub(1).min(6)).min(5_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_print_as_four_char_codes() {
        assert_eq!(status_text(0x73746f70), "'stop' (1937010544)");
        assert_eq!(status_text(-50), "-50");
    }

    /// Lists devices on a real Mac (CI runners may have none).
    #[test]
    fn lists_devices() {
        for kind in [DeviceKind::Input, DeviceKind::Output] {
            let devices = list_devices(kind).unwrap();
            eprintln!("{kind:?}: {devices:?}");
            assert!(devices.iter().filter(|d| d.is_default).count() <= 1);
        }
    }
}
