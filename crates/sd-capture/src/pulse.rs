//! Linux backend: the PulseAudio client API (libpulse).
//!
//! One backend serves both Linux sound servers: PulseAudio itself, and PipeWire through its
//! PulseAudio-compatible server (`pipewire-pulse`, installed by default wherever PipeWire runs
//! the desktop's audio). It also works in Flatpak with the standard `--socket=pulseaudio`.
//!
//! * Microphone: a record stream on a source. System audio: a record stream on the monitor
//!   source of a sink (what the sink plays).
//! * Streams ask for float32 at the configured rate and channel count; the server resamples
//!   and remixes.
//! * Streams are connected to an explicit device with `DONT_MOVE`, so the server never moves
//!   them silently: following the default device, removal and fallback are handled here, with
//!   the same events and settle time as on Windows.
//! * Timestamps: arrival time (`CLOCK_MONOTONIC`) minus the queued data's duration and the
//!   source's buffering latency, smoothed to the lower envelope over the stream position
//!   ([`ArrivalClock`]); the frames read so far are the device position. Server-side overruns
//!   are reported as glitches.
//! * The libpulse main loop runs on the capture thread, polled every 20 ms.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use libpulse_binding as pa;
use pa::callbacks::ListResult;
use pa::context::subscribe::{Facility, InterestMaskSet, Operation as SubOp};
use pa::context::{Context, FlagSet as ContextFlags, State as ContextState};
use pa::def::BufferAttr;
use pa::mainloop::standard::Mainloop;
use pa::operation::State as OpState;
use pa::proplist::Proplist;
use pa::sample::{Format, Spec};
use pa::stream::{FlagSet as StreamFlags, PeekResult, SeekMode, State as StreamState};
use pa::time::MicroSeconds;
use sd_core::ring::RingProducer;
use sd_core::{BlockFlags, HostTime, Source, StreamFormat};

use crate::arrival::ArrivalClock;
use crate::feed::{Feed, Idle, Packet};
use crate::policy::StallWatchdog;
use crate::{
    CaptureConfig, CaptureError, CaptureEvent, CaptureStats, DeviceInfo, DeviceKind,
    DeviceSelector, PlaybackReport, Result, SwitchReason,
};

/// How long to wait for the server to answer a request or bring up a connection/stream.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// Main-loop poll interval; also the reaction time to stop and control requests.
const TICK_MS: u64 = 20;
/// Silence is synthesized only after this long without data (see [`Idle`]).
const IDLE_AFTER_MS: u64 = 1_000;

// ---------------------------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------------------------

/// Server state changes the capture reacts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Notice {
    /// Server defaults may have changed.
    Server,
    /// A source or sink appeared.
    Added,
    /// A source or sink went away.
    Removed,
}

/// A connected libpulse context with its main loop. Not `Send`: it lives on one thread.
struct Conn {
    // Field order matters: the context must go before the main loop it runs on.
    ctx: Context,
    ml: Mainloop,
    notices: Rc<RefCell<VecDeque<Notice>>>,
}

impl Drop for Conn {
    fn drop(&mut self) {
        self.ctx.disconnect();
    }
}

#[derive(Debug, Clone)]
struct SourceDesc {
    name: String,
    description: String,
    /// The sink this source monitors, if it is a monitor.
    monitor_of: Option<u32>,
    spec: Option<StreamFormat>,
}

#[derive(Debug, Clone)]
struct SinkDesc {
    name: String,
    description: String,
    monitor_source: Option<String>,
    spec: Option<StreamFormat>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Defaults {
    sink: Option<String>,
    source: Option<String>,
}

fn format_of(spec: &Spec) -> Option<StreamFormat> {
    StreamFormat::new(spec.rate, spec.channels as u16).ok()
}

impl Conn {
    /// Connects to the server (never autospawns one).
    fn new(name: &str) -> Result<Self> {
        let mut ml = Mainloop::new().ok_or_else(|| CaptureError::Os("pa_mainloop_new".into()))?;
        let mut props =
            Proplist::new().ok_or_else(|| CaptureError::Os("pa_proplist_new".into()))?;
        let _ = props.set_str(pa::proplist::properties::APPLICATION_NAME, "SonicDiar");
        let mut ctx = Context::new_with_proplist(&ml, name, &props)
            .ok_or_else(|| CaptureError::Os("pa_context_new".into()))?;
        ctx.connect(None, ContextFlags::NOAUTOSPAWN, None)
            .map_err(|e| CaptureError::Os(format!("no PulseAudio/PipeWire server: {e}")))?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            match ctx.get_state() {
                ContextState::Ready => break,
                ContextState::Failed | ContextState::Terminated => {
                    return Err(CaptureError::Os(format!(
                        "no PulseAudio/PipeWire server: {}",
                        ctx.errno()
                    )));
                }
                _ => {}
            }
            if Instant::now() > deadline {
                return Err(CaptureError::Os("server connection timed out".into()));
            }
            iterate(&mut ml, TICK_MS).map_err(CaptureError::Os)?;
        }
        Ok(Self {
            ctx,
            ml,
            notices: Rc::new(RefCell::new(VecDeque::new())),
        })
    }

    fn ready(&self) -> bool {
        self.ctx.get_state() == ContextState::Ready
    }

    fn iterate(&mut self, ms: u64) -> Result<(), String> {
        iterate(&mut self.ml, ms)
    }

    /// Runs the main loop until `op` finishes.
    fn wait<C: ?Sized>(&mut self, op: &pa::operation::Operation<C>) -> Result<(), String> {
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        while op.get_state() == OpState::Running {
            if !self.ready() {
                return Err("server connection lost".into());
            }
            if Instant::now() > deadline {
                return Err("server request timed out".into());
            }
            self.iterate(TICK_MS)?;
        }
        Ok(())
    }

    /// Queues [`Notice`]s for server, source and sink changes.
    fn subscribe(&mut self) -> Result<(), String> {
        let q = self.notices.clone();
        self.ctx
            .set_subscribe_callback(Some(Box::new(move |facility, op, _index| {
                let n = match (facility, op) {
                    (Some(Facility::Server), _) => Notice::Server,
                    (Some(Facility::Source | Facility::Sink), Some(SubOp::New)) => Notice::Added,
                    (Some(Facility::Source | Facility::Sink), Some(SubOp::Removed)) => {
                        Notice::Removed
                    }
                    _ => return,
                };
                q.borrow_mut().push_back(n);
            })));
        let op = self.ctx.subscribe(
            InterestMaskSet::SERVER | InterestMaskSet::SOURCE | InterestMaskSet::SINK,
            |_| {},
        );
        self.wait(&op)
    }

    fn take_notices(&self) -> Vec<Notice> {
        self.notices.borrow_mut().drain(..).collect()
    }

    fn defaults(&mut self) -> Result<Defaults, String> {
        let out = Rc::new(RefCell::new(Defaults::default()));
        let o = out.clone();
        let op = self.ctx.introspect().get_server_info(move |i| {
            *o.borrow_mut() = Defaults {
                sink: i.default_sink_name.as_ref().map(|s| s.to_string()),
                source: i.default_source_name.as_ref().map(|s| s.to_string()),
            };
        });
        self.wait(&op)?;
        Ok(out.take())
    }

    fn sources(&mut self) -> Result<Vec<SourceDesc>, String> {
        let out = Rc::new(RefCell::new(Vec::new()));
        let o = out.clone();
        let op = self.ctx.introspect().get_source_info_list(move |r| {
            if let ListResult::Item(i) = r {
                o.borrow_mut().push(SourceDesc {
                    name: i.name.as_deref().unwrap_or_default().to_string(),
                    description: i.description.as_deref().unwrap_or_default().to_string(),
                    monitor_of: i.monitor_of_sink,
                    spec: format_of(&i.sample_spec),
                });
            }
        });
        self.wait(&op)?;
        Ok(out.take())
    }

    fn sinks(&mut self) -> Result<Vec<SinkDesc>, String> {
        let out = Rc::new(RefCell::new(Vec::new()));
        let o = out.clone();
        let op = self.ctx.introspect().get_sink_info_list(move |r| {
            if let ListResult::Item(i) = r {
                o.borrow_mut().push(SinkDesc {
                    name: i.name.as_deref().unwrap_or_default().to_string(),
                    description: i.description.as_deref().unwrap_or_default().to_string(),
                    monitor_source: i.monitor_source_name.as_ref().map(|s| s.to_string()),
                    spec: format_of(&i.sample_spec),
                });
            }
        });
        self.wait(&op)?;
        Ok(out.take())
    }
}

/// One main-loop iteration waiting at most `ms` for events.
fn iterate(ml: &mut Mainloop, ms: u64) -> Result<(), String> {
    ml.prepare(Some(MicroSeconds(ms * 1_000)))
        .map_err(|e| format!("pa_mainloop_prepare: {e}"))?;
    ml.poll().map_err(|e| format!("pa_mainloop_poll: {e}"))?;
    ml.dispatch()
        .map_err(|e| format!("pa_mainloop_dispatch: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------------------------

pub fn list_devices(kind: DeviceKind) -> Result<Vec<DeviceInfo>> {
    let mut conn = Conn::new("sonicdiar device list")?;
    let defaults = conn.defaults().map_err(CaptureError::Os)?;
    Ok(match kind {
        DeviceKind::Input => conn
            .sources()
            .map_err(CaptureError::Os)?
            .into_iter()
            // Monitors are system audio, not microphones.
            .filter(|s| s.monitor_of.is_none())
            .map(|s| DeviceInfo {
                is_default: defaults.source.as_deref() == Some(s.name.as_str()),
                name: if s.description.is_empty() {
                    s.name.clone()
                } else {
                    s.description
                },
                mix_format: s.spec,
                id: s.name,
                kind,
            })
            .collect(),
        DeviceKind::Output => conn
            .sinks()
            .map_err(CaptureError::Os)?
            .into_iter()
            .map(|s| DeviceInfo {
                is_default: defaults.sink.as_deref() == Some(s.name.as_str()),
                name: if s.description.is_empty() {
                    s.name.clone()
                } else {
                    s.description
                },
                mix_format: s.spec,
                id: s.name,
                kind,
            })
            .collect(),
    })
}

/// The device a capture records from.
struct Target {
    /// Reported device id: the source (mic) or the sink (system audio).
    id: String,
    name: String,
    mix_format: Option<StreamFormat>,
    /// The source the stream records: the mic itself, or the sink's monitor.
    record_from: String,
}

enum OpenError {
    NoDevice,
    /// The server connection is gone; reconnect.
    Disconnected(String),
    Other(String),
}

/// Finds the device to record for `source`: the preferred one if it exists, else the default.
/// Returns the target and whether the preferred device was missing (fallback).
fn resolve(
    conn: &mut Conn,
    source: Source,
    preferred: Option<&str>,
) -> Result<(Target, bool), OpenError> {
    let lost = |e: String| {
        if conn_lost(&e) {
            OpenError::Disconnected(e)
        } else {
            OpenError::Other(e)
        }
    };
    let defaults = conn.defaults().map_err(lost)?;
    match source {
        Source::Mic => {
            let sources = conn.sources().map_err(lost)?;
            let mics: Vec<&SourceDesc> =
                sources.iter().filter(|s| s.monitor_of.is_none()).collect();
            let found = preferred.and_then(|p| mics.iter().find(|s| s.name == p));
            // With no microphone, PulseAudio makes a sink monitor the default source; that is
            // not a microphone, so it counts as no device.
            let chosen = found.or_else(|| {
                defaults
                    .source
                    .as_deref()
                    .and_then(|d| mics.iter().find(|s| s.name == d))
            });
            let s = chosen.ok_or(OpenError::NoDevice)?;
            Ok((
                Target {
                    id: s.name.clone(),
                    name: s.description.clone(),
                    mix_format: s.spec,
                    record_from: s.name.clone(),
                },
                preferred.is_some() && found.is_none(),
            ))
        }
        Source::System => {
            let sinks = conn.sinks().map_err(lost)?;
            let found = preferred.and_then(|p| sinks.iter().find(|s| s.name == p));
            let chosen = found.or_else(|| {
                defaults
                    .sink
                    .as_deref()
                    .and_then(|d| sinks.iter().find(|s| s.name == d))
            });
            let s = chosen.ok_or(OpenError::NoDevice)?;
            let monitor = s.monitor_source.clone().ok_or(OpenError::NoDevice)?;
            Ok((
                Target {
                    id: s.name.clone(),
                    name: s.description.clone(),
                    mix_format: s.spec,
                    record_from: monitor,
                },
                preferred.is_some() && found.is_none(),
            ))
        }
    }
}

fn conn_lost(e: &str) -> bool {
    e.contains("connection lost")
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
    let stop = Arc::new(AtomicBool::new(false));
    let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded();
    let ctx = ThreadCtx {
        // Sources deliver continuously (silence included) unless suspended, so a pause in
        // delivery is server scheduling, not silence: fill only after a long one.
        feed: Feed::new(
            &cfg,
            producer,
            stats.clone(),
            Idle {
                after_ms: IDLE_AFTER_MS,
                margin_ms: IDLE_AFTER_MS,
            },
        ),
        cfg: cfg.clone(),
        events,
        stats,
        stop: stop.clone(),
        ctrl: ctrl_rx,
        scratch: Vec::new(),
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
    /// Samples converted from the server's bytes.
    scratch: Vec<f32>,
}

struct OpenStream {
    stream: pa::stream::Stream,
    device_id: String,
    /// Frames read from this stream so far (holes included): the device position.
    position: u64,
    /// Set by the overflow callback: the server dropped data we did not read in time.
    overflow: Rc<Cell<bool>>,
    /// The source's buffering latency (from the first timing update), added to arrival times.
    source_latency_ns: Option<u64>,
    /// Removes scheduling jitter from arrival-based timestamps.
    clock: ArrivalClock,
    diag: Option<Diag>,
}

/// Delivery statistics of one stream, printed when it ends if `SD_PULSE_DIAG` is set
/// (for diagnosing servers in CI).
#[derive(Default)]
struct Diag {
    fragments: u64,
    min_frames: usize,
    max_frames: usize,
    last_data: Option<Instant>,
    max_interval_ms: u128,
    /// Reported latency per fragment, microseconds.
    latency_us: Vec<i64>,
    errors: u32,
    overflows: u32,
    holes: u32,
    /// Most frames found queued at one drain.
    queued_max: u64,
}

impl Diag {
    fn line(&mut self, what: &str) -> String {
        self.latency_us.sort_unstable();
        let q = |p: f64| {
            self.latency_us
                .get(((self.latency_us.len().max(1) - 1) as f64 * p) as usize)
                .copied()
                .unwrap_or(-1)
                / 1000
        };
        format!(
            "pulse diag {what}: {} fragments of {}..{} frames, max interval {} ms, source latency ms min/p50/max {}/{}/{}, no timing yet {}, max queued {} frames, overflows {}, holes {}",
            self.fragments,
            self.min_frames,
            self.max_frames,
            self.max_interval_ms,
            q(0.0),
            q(0.5),
            q(1.0),
            self.errors,
            self.queued_max,
            self.overflows,
            self.holes
        )
    }
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        let _ = self.stream.disconnect();
    }
}

enum StreamExit {
    Stop,
    Switch(SwitchReason),
    /// The server connection is gone; reconnect.
    Disconnected(String),
    Failed(String),
}

/// Mutable state that survives device switches (the timeline itself is in [`Feed`]).
struct Session {
    preferred: DeviceSelector,
    on_fallback: bool,
    recover_requested: bool,
    stall_injected: bool,
}

impl ThreadCtx {
    fn emit(&self, e: CaptureEvent) {
        let _ = self.events.send(e);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn frame_bytes(&self) -> usize {
        self.cfg.format.channels as usize * 4
    }

    fn run(mut self) {
        let mut sess = Session {
            preferred: self.cfg.device.clone(),
            on_fallback: false,
            recover_requested: false,
            stall_injected: false,
        };
        // Both sources start their timeline at capture start, so frame 0 means the same instant.
        self.feed.start_at(HostTime::now());
        let mut conn: Option<Conn> = None;
        let mut attempt = 0u32;
        let mut reported_no_device = false;
        while !self.stopped() {
            if conn.as_ref().is_none_or(|c| !c.ready()) {
                conn = None;
                match Conn::new(&format!("sonicdiar {}", self.cfg.source))
                    .and_then(|mut c| c.subscribe().map(|_| c).map_err(CaptureError::Os))
                {
                    Ok(c) => conn = Some(c),
                    Err(e) => {
                        attempt += 1;
                        let delay = backoff_ms(attempt);
                        self.emit(CaptureEvent::Retrying {
                            attempt,
                            retry_in_ms: delay,
                            error: e.to_string(),
                        });
                        self.idle_wait(None, &mut sess, delay);
                        continue;
                    }
                }
            }
            let Some(c) = conn.as_mut() else { continue };
            match self.open(c, &mut sess) {
                Ok(mut stream) => {
                    attempt = 0;
                    reported_no_device = false;
                    match self.pump(c, &mut stream, &mut sess) {
                        StreamExit::Stop => break,
                        StreamExit::Switch(reason) => {
                            self.stats.restarts.fetch_add(1, Ordering::Relaxed);
                            self.emit(CaptureEvent::Switching { reason });
                        }
                        StreamExit::Disconnected(err) => {
                            drop(stream);
                            conn = None;
                            attempt += 1;
                            self.emit(CaptureEvent::Retrying {
                                attempt,
                                retry_in_ms: 0,
                                error: err,
                            });
                        }
                        StreamExit::Failed(err) => {
                            drop(stream);
                            attempt += 1;
                            let delay = backoff_ms(attempt);
                            self.emit(CaptureEvent::Retrying {
                                attempt,
                                retry_in_ms: delay,
                                error: err,
                            });
                            self.idle_wait(conn.as_mut(), &mut sess, delay);
                        }
                    }
                }
                Err(OpenError::NoDevice) => {
                    if !reported_no_device {
                        self.emit(CaptureEvent::NoDevice);
                        reported_no_device = true;
                    }
                    attempt += 1;
                    self.idle_wait(conn.as_mut(), &mut sess, backoff_ms(attempt).min(1_000));
                }
                Err(OpenError::Disconnected(err)) => {
                    conn = None;
                    attempt += 1;
                    self.emit(CaptureEvent::Retrying {
                        attempt,
                        retry_in_ms: 0,
                        error: err,
                    });
                }
                Err(OpenError::Other(err)) => {
                    attempt += 1;
                    let delay = backoff_ms(attempt);
                    self.emit(CaptureEvent::Retrying {
                        attempt,
                        retry_in_ms: delay,
                        error: err,
                    });
                    self.idle_wait(conn.as_mut(), &mut sess, delay);
                }
            }
        }
        drop(conn);
        self.emit(CaptureEvent::Stopped);
    }

    fn open(&mut self, conn: &mut Conn, sess: &mut Session) -> Result<OpenStream, OpenError> {
        let preferred = match &sess.preferred {
            DeviceSelector::Id(id) => Some(id.clone()),
            DeviceSelector::Default => None,
        };
        conn.take_notices(); // the state is read fresh below
        let (target, fallback) = resolve(conn, self.cfg.source, preferred.as_deref())?;
        if fallback && !sess.on_fallback {
            self.emit(CaptureEvent::Switching {
                reason: SwitchReason::DeviceRemoved,
            });
        }
        sess.on_fallback = fallback;

        let spec = Spec {
            format: Format::F32le,
            rate: self.cfg.format.sample_rate,
            channels: self.cfg.format.channels as u8,
        };
        let mut stream = pa::stream::Stream::new(
            &mut conn.ctx,
            &format!("sonicdiar {}", self.cfg.source),
            &spec,
            None,
        )
        .ok_or_else(|| OpenError::Other("pa_stream_new failed".into()))?;
        // Deliver about every 20 ms (at most a fifth of the OS buffer); the server keeps up to
        // its default maximum buffered if we fall behind.
        let frag_ms = (self.cfg.os_buffer_ms / 5).clamp(5, 20);
        let fragsize = (self.cfg.format.sample_rate as usize * frag_ms as usize / 1000
            * self.frame_bytes()) as u32;
        let attr = BufferAttr {
            maxlength: u32::MAX,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize,
        };
        let overflow = Rc::new(Cell::new(false));
        let o = overflow.clone();
        stream.set_overflow_callback(Some(Box::new(move || o.set(true))));
        stream
            .connect_record(
                Some(&target.record_from),
                Some(&attr),
                StreamFlags::ADJUST_LATENCY
                    | StreamFlags::AUTO_TIMING_UPDATE
                    | StreamFlags::INTERPOLATE_TIMING
                    | StreamFlags::DONT_MOVE,
            )
            .map_err(|e| OpenError::Other(format!("connect {}: {e}", target.record_from)))?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            match stream.get_state() {
                StreamState::Ready => break,
                StreamState::Failed | StreamState::Terminated => {
                    let e = conn.ctx.errno();
                    return Err(if !conn.ready() {
                        OpenError::Disconnected(format!("server connection lost: {e}"))
                    } else {
                        // The device vanished between listing and connecting.
                        OpenError::Other(format!("record from {}: {e}", target.record_from))
                    });
                }
                _ => {}
            }
            if Instant::now() > deadline {
                return Err(OpenError::Other(format!(
                    "record from {}: timed out",
                    target.record_from
                )));
            }
            conn.iterate(TICK_MS).map_err(OpenError::Disconnected)?;
        }
        self.emit(CaptureEvent::Started {
            device_id: target.id.clone(),
            device_name: if target.name.is_empty() {
                target.id.clone()
            } else {
                target.name
            },
            mix_format: target.mix_format,
            raw: false, // PulseAudio has no raw mode; sources are unprocessed unless the user
                        // loads an effect module.
        });
        Ok(OpenStream {
            stream,
            device_id: target.id,
            position: 0,
            overflow,
            source_latency_ns: None,
            clock: ArrivalClock::default(),
            diag: std::env::var_os("SD_PULSE_DIAG").map(|_| Diag {
                min_frames: usize::MAX,
                ..Default::default()
            }),
        })
    }

    /// Waits up to `ms` while keeping the timeline running with silence; returns early on stop,
    /// a control message, or a server change that may make a device available.
    fn idle_wait(&mut self, mut conn: Option<&mut Conn>, sess: &mut Session, ms: u64) {
        let deadline = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < deadline && !self.stopped() {
            match conn.as_deref_mut() {
                Some(c) if c.ready() => {
                    if c.iterate(TICK_MS).is_err() {
                        std::thread::sleep(Duration::from_millis(TICK_MS));
                    }
                }
                _ => std::thread::sleep(Duration::from_millis(TICK_MS)),
            }
            self.feed.fill_idle();
            let mut wake = false;
            if let Some(c) = conn.as_deref_mut() {
                wake |= c
                    .take_notices()
                    .iter()
                    .any(|n| matches!(n, Notice::Added | Notice::Server));
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

    fn pump(&mut self, conn: &mut Conn, stream: &mut OpenStream, sess: &mut Session) -> StreamExit {
        let exit = self.pump_inner(conn, stream, sess);
        if let Some(d) = stream.diag.as_mut() {
            let s = self.stats.snapshot();
            eprintln!(
                "{}; feed idle {} gap {} trimmed {} skipped {} frames",
                d.line(&format!("{} {}", self.cfg.source, stream.device_id)),
                s.idle_frames,
                s.gap_frames,
                s.trimmed_frames,
                s.skipped_frames
            );
        }
        exit
    }

    fn pump_inner(
        &mut self,
        conn: &mut Conn,
        stream: &mut OpenStream,
        sess: &mut Session,
    ) -> StreamExit {
        self.feed.new_stream(); // positions restart with every stream
        let started = Instant::now();
        let mut watchdog = StallWatchdog::new(self.cfg.stall_timeout_ms, started);
        let inject_at = self
            .cfg
            .test_stall_after_ms
            .filter(|_| !sess.stall_injected)
            .map(|ms| started + Duration::from_millis(ms as u64));
        let mut pending_default_switch: Option<Instant> = None;
        let settle = Duration::from_millis(self.cfg.default_switch_settle_ms as u64);
        loop {
            if self.stopped() {
                return StreamExit::Stop;
            }
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
            if let Err(e) = conn.iterate(TICK_MS) {
                return StreamExit::Disconnected(e);
            }
            if !conn.ready() {
                return StreamExit::Disconnected(format!(
                    "server connection lost: {}",
                    conn.ctx.errno()
                ));
            }
            match stream.stream.get_state() {
                StreamState::Ready => {}
                // With DONT_MOVE the server kills the stream when its device goes away.
                StreamState::Failed | StreamState::Terminated => {
                    return StreamExit::Switch(SwitchReason::DeviceRemoved);
                }
                _ => {}
            }
            let notices = conn.take_notices();
            if notices.contains(&Notice::Server) {
                match conn.defaults() {
                    Ok(d) => {
                        let follows_default =
                            matches!(sess.preferred, DeviceSelector::Default) || sess.on_fallback;
                        let new_default = match self.cfg.source {
                            Source::Mic => &d.source,
                            Source::System => &d.sink,
                        };
                        if follows_default
                            && new_default.is_some()
                            && new_default.as_deref() != Some(stream.device_id.as_str())
                        {
                            pending_default_switch = Some(Instant::now() + settle);
                        }
                    }
                    Err(e) => return StreamExit::Disconnected(e),
                }
            }
            // PulseAudio kills a DONT_MOVE stream whose device goes away; a server may also
            // leave it unlinked, so check that the device still exists.
            if notices.contains(&Notice::Removed) {
                let exists = match self.cfg.source {
                    Source::Mic => conn
                        .sources()
                        .map(|v| v.iter().any(|s| s.name == stream.device_id)),
                    Source::System => conn
                        .sinks()
                        .map(|v| v.iter().any(|s| s.name == stream.device_id)),
                };
                match exists {
                    Ok(false) => return StreamExit::Switch(SwitchReason::DeviceRemoved),
                    Ok(true) => {}
                    Err(e) => return StreamExit::Disconnected(e),
                }
            }
            if notices.contains(&Notice::Added)
                && sess.on_fallback
                && let DeviceSelector::Id(p) = &sess.preferred
            {
                let back = match self.cfg.source {
                    Source::Mic => conn.sources().is_ok_and(|v| v.iter().any(|s| &s.name == p)),
                    Source::System => conn.sinks().is_ok_and(|v| v.iter().any(|s| &s.name == p)),
                };
                if back {
                    return StreamExit::Switch(SwitchReason::Requested);
                }
            }

            let now = Instant::now();
            let injected = inject_at.is_some_and(|t| now >= t);
            if injected {
                sess.stall_injected = true;
            }
            let read = if injected {
                Ok(0) // fault injection: behave as if the server delivered nothing
            } else {
                self.drain(stream, sess)
            };
            match read {
                Ok(0) => {
                    // A suspended device (e.g. an idle sink's monitor) legitimately delivers
                    // nothing; otherwise silence for the timeout is a stall.
                    let suspended = stream.stream.is_suspended().unwrap_or(false);
                    if suspended {
                        watchdog.packet(now);
                    } else if let Some(quiet) = watchdog.stalled_for(now) {
                        self.emit(CaptureEvent::Stalled {
                            quiet_ms: quiet.as_millis() as u64,
                        });
                        return StreamExit::Switch(SwitchReason::Recovery);
                    }
                    self.feed.fill_idle();
                }
                Ok(_) => {
                    watchdog.packet(now);
                    if sess.recover_requested {
                        sess.recover_requested = false;
                        return StreamExit::Switch(SwitchReason::Recovery);
                    }
                }
                Err(e) => return StreamExit::Failed(format!("read {}: {e}", self.cfg.source)),
            }
            if pending_default_switch.is_some_and(|t| Instant::now() >= t) {
                return StreamExit::Switch(SwitchReason::DefaultDeviceChanged);
            }
        }
    }

    /// Reads every available fragment. Returns the number of fragments read.
    fn drain(&mut self, s: &mut OpenStream, sess: &mut Session) -> Result<u32, String> {
        let fb = self.frame_bytes();
        let mut fragments = 0;
        // Arrival-based timestamps: the oldest queued frame arrived one queue-duration before
        // now, and was captured the source's buffering latency before that (fixed per stream).
        // `pa_stream_get_latency` is not used: for monitor sources it subtracts the sink's
        // playback buffer, so it moves (even below zero) as playback streams come and go.
        let now = HostTime::now().as_ns();
        let queued = (s.stream.readable_size().unwrap_or(0) / fb) as u64;
        if s.source_latency_ns.is_none()
            && let Some(ti) = s.stream.get_timing_info()
        {
            let us = (ti.source_usec.0 + ti.transport_usec.0).min(500_000);
            s.source_latency_ns = Some(us * 1_000);
        }
        let base = now
            .saturating_sub(self.cfg.format.frames_to_ns(queued))
            .saturating_sub(s.source_latency_ns.unwrap_or(0));
        let mut consumed = 0u64;
        loop {
            let peeked = s
                .stream
                .peek()
                .map_err(|e| format!("pa_stream_peek: {e}"))?;
            let (frames, hole) = match peeked {
                PeekResult::Empty => return Ok(fragments),
                PeekResult::Hole(n) => (n / fb, true),
                PeekResult::Data(bytes) => {
                    let n = bytes.len() / fb;
                    self.scratch.clear();
                    self.scratch.extend(
                        bytes[..n * fb]
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                    );
                    (n, false)
                }
            };
            s.stream
                .discard()
                .map_err(|e| format!("pa_stream_drop: {e}"))?;
            fragments += 1;
            if frames == 0 {
                continue;
            }
            let mut flags = BlockFlags::empty();
            let arrival = base + self.cfg.format.frames_to_ns(consumed);
            consumed += frames as u64;
            let t = HostTime::from_ns(s.clock.stamp(
                self.cfg.format.frames_to_ns(s.position) as f64,
                arrival as f64,
            ));
            let overflowed = s.overflow.replace(false);
            if overflowed {
                flags.insert(BlockFlags::DISCONTINUITY);
            }
            if hole {
                flags.insert(BlockFlags::SILENT | BlockFlags::DISCONTINUITY);
            }
            if let Some(d) = s.diag.as_mut() {
                d.fragments += 1;
                d.min_frames = d.min_frames.min(frames);
                d.max_frames = d.max_frames.max(frames);
                let now_i = Instant::now();
                if let Some(last) = d.last_data {
                    d.max_interval_ms = d.max_interval_ms.max((now_i - last).as_millis());
                }
                d.last_data = Some(now_i);
                match s.source_latency_ns {
                    Some(ns) => d.latency_us.push((ns / 1_000) as i64),
                    None => d.errors += 1,
                }
                d.queued_max = d.queued_max.max(queued);
                d.overflows += overflowed as u32;
                d.holes += hole as u32;
            }
            let position = s.position as f64;
            s.position += frames as u64;
            let data = (!hole).then_some(&self.scratch[..]);
            if let Some(filled_ms) = self.feed.packet(Packet {
                t,
                frames: frames as u32,
                position: Some(position),
                flags,
                data,
            }) {
                self.emit(CaptureEvent::GapFillBudgetExhausted { filled_ms });
                sess.recover_requested = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Playback (test signals: loopback verification, echo-path measurement)
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
    let mut conn = Conn::new("sonicdiar playback")?;
    let spec = Spec {
        format: Format::F32le,
        rate: format.sample_rate,
        channels: format.channels as u8,
    };
    let mut stream = pa::stream::Stream::new(&mut conn.ctx, "sonicdiar test signal", &spec, None)
        .ok_or_else(|| CaptureError::Os("pa_stream_new failed".into()))?;
    let dev = match device {
        DeviceSelector::Id(id) => Some(id.as_str()),
        DeviceSelector::Default => None,
    };
    // Half a second buffered on the server, so a busy machine does not underrun the signal.
    let tlength = (format.sample_rate as usize / 2 * ch * 4) as u32;
    let attr = BufferAttr {
        maxlength: u32::MAX,
        tlength,
        prebuf: u32::MAX,
        minreq: u32::MAX,
        fragsize: u32::MAX,
    };
    stream
        .connect_playback(
            dev,
            Some(&attr),
            // Started corked and uncorked once the buffer is full, so playback cannot run dry
            // right at the start.
            StreamFlags::ADJUST_LATENCY
                | StreamFlags::AUTO_TIMING_UPDATE
                | StreamFlags::INTERPOLATE_TIMING
                | StreamFlags::START_CORKED,
            None,
            None,
        )
        .map_err(|e| CaptureError::Os(format!("connect playback: {e}")))?;
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    loop {
        match stream.get_state() {
            StreamState::Ready => break,
            StreamState::Failed | StreamState::Terminated => {
                return Err(CaptureError::Os(format!(
                    "open output device: {}",
                    conn.ctx.errno()
                )));
            }
            _ => {}
        }
        if Instant::now() > deadline {
            return Err(CaptureError::Os("open output device: timed out".into()));
        }
        conn.iterate(TICK_MS).map_err(CaptureError::Os)?;
    }

    let underflows = Rc::new(Cell::new(0u32));
    let u = underflows.clone();
    stream.set_underflow_callback(Some(Box::new(move || u.set(u.get() + 1))));
    let mut written = 0u64; // frames, including trailing silence
    let mut corked = true;
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets = Vec::new();
    let end =
        Instant::now() + Duration::from_secs_f64(total as f64 / format.sample_rate as f64 + 5.0);
    loop {
        conn.iterate(5).map_err(CaptureError::Os)?;
        if let Some(room) = stream.writable_size().filter(|&n| n > 0) {
            let n = room / (ch * 4);
            bytes.clear();
            for i in 0..n {
                let frame = written as usize + i;
                for c in 0..ch {
                    let v = samples.get(frame * ch + c).copied().unwrap_or(0.0);
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            stream
                .write(&bytes, None, 0, SeekMode::Relative)
                .map_err(|e| CaptureError::Os(format!("pa_stream_write: {e}")))?;
            written += n as u64;
        }
        if corked && written > 0 {
            let _ = stream.uncork(None);
            corked = false;
        }
        // Host time at which frame 0 played: now minus the stream's playback time.
        if let Ok(Some(played)) = stream.get_time()
            && played.0 > 0
        {
            let now = HostTime::now().as_ns() as f64;
            offsets.push(now - played.0 as f64 * 1_000.0);
            if played.0 as f64 * 1e-6 * format.sample_rate as f64
                >= total as f64 + format.sample_rate as f64 * 0.05
            {
                break;
            }
        }
        if Instant::now() > end {
            break;
        }
    }
    let _ = stream.disconnect();
    if underflows.get() > 0 {
        tracing::warn!(
            "test signal playback underran {} times; the played signal has gaps",
            underflows.get()
        );
    }
    if offsets.is_empty() {
        return Err(CaptureError::Os("playback clock never advanced".into()));
    }
    offsets.sort_by(f64::total_cmp);
    Ok(PlaybackReport {
        first_frame_host_time: HostTime::from_ns(offsets[offsets.len() / 2] as u64),
        frames: total,
        clock_samples: offsets.len(),
        underruns: underflows.get(),
    })
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
        assert_eq!(backoff_ms(4), 800);
        assert_eq!(backoff_ms(20), 5_000);
    }

    /// Needs a running PulseAudio or PipeWire server; skipped (passes) without one.
    #[test]
    fn lists_devices_when_a_server_runs() {
        match list_devices(DeviceKind::Output) {
            Ok(sinks) => {
                for s in &sinks {
                    assert!(!s.id.is_empty());
                }
            }
            Err(e) => eprintln!("no sound server, skipped: {e}"),
        }
    }
}
