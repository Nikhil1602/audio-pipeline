//! Native audio capture: microphone and system audio (loopback) as separate streams.
//!
//! Each capture runs on its own thread and pushes fixed-format `f32` blocks into an
//! [`sd_core::ring`] ring. The format never changes for the life of a capture, even when the
//! underlying device is switched, because the OS converter (WASAPI `AUTOCONVERTPCM` on
//! Windows) is asked for exactly [`CaptureConfig::format`]. Every block carries a host-clock
//! timestamp, and the timeline stays continuous across glitches, loopback silence and device
//! switches (see [`sd_core::timeline`]).
//!
//! Platform status: **Windows (WASAPI)**, **Linux (PulseAudio API, served by PulseAudio or
//! PipeWire)** and **macOS (Core Audio; system audio through a process tap, macOS 14.2+)**.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use sd_core::{Source, StreamFormat};
use thiserror::Error;

// Smooths arrival-time stamps (Linux).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod arrival;
// Converts device-format audio (Core Audio).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod convert;
// Used by every backend; unused only on platforms without one yet.
#[cfg(target_os = "macos")]
mod coreaudio;
#[cfg_attr(
    not(any(windows, target_os = "linux", target_os = "macos")),
    allow(dead_code)
)]
mod feed;
pub mod policy;
#[cfg(target_os = "linux")]
mod pulse;
#[cfg(windows)]
mod wasapi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// Capture endpoints (microphones).
    Input,
    /// Render endpoints (speakers/headphones); captured through loopback.
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable OS identifier, usable with [`DeviceSelector::Id`].
    pub id: String,
    pub name: String,
    pub kind: DeviceKind,
    pub is_default: bool,
    /// The device's shared-mode mix format (what the OS mixes at), for information.
    pub mix_format: Option<StreamFormat>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelector {
    /// Follow the OS default device, switching when the user changes it.
    Default,
    /// A specific device. If it disappears, capture falls back to the default device.
    Id(String),
}

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub source: Source,
    pub device: DeviceSelector,
    /// Output format delivered to the ring (the OS converts to it).
    pub format: StreamFormat,
    /// Ring capacity in milliseconds of audio.
    pub ring_ms: u32,
    /// OS buffer duration in milliseconds.
    pub os_buffer_ms: u32,
    /// Default-device changes are applied after this settle time (devices often fire several
    /// notifications in a row).
    pub default_switch_settle_ms: u32,
    /// Gaps longer than this are not filled with silence in the ring; the timeline is advanced
    /// instead (e.g. after system sleep).
    pub max_gap_fill_ms: u32,
    /// Total silence inserted for gaps per capture session; beyond it, gaps are skipped, the
    /// stream is restarted once and [`CaptureEvent::GapFillBudgetExhausted`] is reported.
    /// Loopback idle fill does not count. See [`policy::GapFillPolicy`].
    pub max_session_gap_fill_ms: u32,
    /// A started stream that delivers nothing for this long is reopened and
    /// [`CaptureEvent::Stalled`] is reported. Loopback counts as stalled only while the output
    /// endpoint's peak meter shows audio playing for the whole period. See
    /// [`policy::StallWatchdog`].
    pub stall_timeout_ms: u32,
    /// Fault injection for tests: stop reading the first stream this many ms after it starts,
    /// as a hung driver would. Never set in production.
    #[doc(hidden)]
    pub test_stall_after_ms: Option<u32>,
    /// Ask the OS for an unprocessed ("raw") stream, bypassing driver/OS effects such as the
    /// device's own echo cancellation and noise suppression (Windows: `AUDCLNT_STREAMOPTIONS_RAW`).
    /// Falls back to the processed stream if the device does not support it; the
    /// [`CaptureEvent::Started`] event reports which mode is in effect.
    pub raw: bool,
}

impl CaptureConfig {
    pub fn new(source: Source) -> Self {
        let channels = match source {
            Source::Mic => 1,
            Source::System => 2,
        };
        Self {
            source,
            device: DeviceSelector::Default,
            format: StreamFormat::new(48_000, channels).expect("valid"),
            ring_ms: 2_000,
            os_buffer_ms: 100,
            default_switch_settle_ms: 200,
            max_gap_fill_ms: 2_000,
            max_session_gap_fill_ms: 300_000,
            stall_timeout_ms: 2_000,
            test_stall_after_ms: None,
            raw: false,
        }
    }
}

/// Lifecycle and device events from a running capture.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureEvent {
    /// A device stream started (initially and after every switch/restart).
    Started {
        device_id: String,
        device_name: String,
        mix_format: Option<StreamFormat>,
        /// True if the raw (unprocessed) stream was requested and accepted.
        raw: bool,
    },
    /// Capture is moving to another device.
    Switching {
        reason: SwitchReason,
    },
    /// Opening or reading the device failed; capture retries after `retry_in_ms`.
    Retrying {
        attempt: u32,
        retry_in_ms: u64,
        error: String,
    },
    /// No usable device exists. Capture keeps the timeline running and retries.
    NoDevice,
    /// The device reported so many gaps that the session's silence-fill budget is used up
    /// (`filled_ms` of silence inserted). Further gaps are left as gaps, and the stream is
    /// restarted once.
    GapFillBudgetExhausted {
        filled_ms: u64,
    },
    /// The stream delivered nothing for `quiet_ms` while audio was expected; it is reopened.
    Stalled {
        quiet_ms: u64,
    },
    /// The OS may be withholding audio for lack of a privacy permission (macOS). `confirmed`
    /// is true when the OS reported a denial; false when it is inferred (a system-audio tap
    /// delivering only silence). `what` names the permission and where to grant it.
    PermissionNeeded {
        what: String,
        confirmed: bool,
    },
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchReason {
    DefaultDeviceChanged,
    DeviceRemoved,
    DeviceInvalidated,
    Requested,
    /// Reopening the same device to recover from misbehaviour (e.g. gap-fill budget used up).
    Recovery,
}

/// Counters updated by the capture thread.
#[derive(Debug, Default)]
pub struct CaptureStats {
    pub packets: AtomicU64,
    pub frames_captured: AtomicU64,
    /// Device reported a data discontinuity (glitch).
    pub device_glitches: AtomicU64,
    pub timestamp_errors: AtomicU64,
    /// Frames of silence inserted for timeline gaps (glitches, restarts).
    pub gap_frames: AtomicU64,
    /// Frames of silence synthesized while a loopback stream delivered nothing.
    pub idle_frames: AtomicU64,
    /// Frames trimmed because they overlapped already-emitted audio.
    pub trimmed_frames: AtomicU64,
    /// Gap frames too long to fill, skipped on the timeline instead.
    pub skipped_frames: AtomicU64,
    pub restarts: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureStatsSnapshot {
    pub packets: u64,
    pub frames_captured: u64,
    pub device_glitches: u64,
    pub timestamp_errors: u64,
    pub gap_frames: u64,
    pub idle_frames: u64,
    pub trimmed_frames: u64,
    pub skipped_frames: u64,
    pub restarts: u64,
}

impl CaptureStats {
    pub fn snapshot(&self) -> CaptureStatsSnapshot {
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        CaptureStatsSnapshot {
            packets: l(&self.packets),
            frames_captured: l(&self.frames_captured),
            device_glitches: l(&self.device_glitches),
            timestamp_errors: l(&self.timestamp_errors),
            gap_frames: l(&self.gap_frames),
            idle_frames: l(&self.idle_frames),
            trimmed_frames: l(&self.trimmed_frames),
            skipped_frames: l(&self.skipped_frames),
            restarts: l(&self.restarts),
        }
    }
}

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("audio capture is not supported on this platform yet")]
    Unsupported,
    #[error("invalid capture configuration: {0}")]
    Config(String),
    #[error("OS audio error: {0}")]
    Os(String),
    #[error("capture thread failed to start: {0}")]
    Thread(String),
}

pub type Result<T, E = CaptureError> = std::result::Result<T, E>;

/// A running capture. Dropping it stops the capture thread.
pub struct CaptureHandle {
    inner: Option<platform::Handle>,
    events: crossbeam_channel::Receiver<CaptureEvent>,
    stats: Arc<CaptureStats>,
    format: StreamFormat,
    source: Source,
}

impl CaptureHandle {
    pub fn events(&self) -> &crossbeam_channel::Receiver<CaptureEvent> {
        &self.events
    }
    pub fn stats(&self) -> CaptureStatsSnapshot {
        self.stats.snapshot()
    }
    /// Shared counters, readable from other threads while the capture runs.
    pub fn stats_handle(&self) -> Arc<CaptureStats> {
        self.stats.clone()
    }
    pub fn format(&self) -> StreamFormat {
        self.format
    }
    pub fn source(&self) -> Source {
        self.source
    }
    /// Switches to another device without interrupting the timeline.
    pub fn set_device(&self, device: DeviceSelector) {
        if let Some(h) = &self.inner {
            h.set_device(device);
        }
    }
    /// Stops capture and waits for the thread to exit.
    pub fn stop(mut self) {
        if let Some(h) = self.inner.take() {
            h.stop();
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        if let Some(h) = self.inner.take() {
            h.stop();
        }
    }
}

/// Result of [`play`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackReport {
    /// Host time at which the first frame reached the device (from the device clock).
    pub first_frame_host_time: sd_core::HostTime,
    pub frames: u64,
    /// Number of device-clock readings the estimate is based on.
    pub clock_samples: usize,
    /// Times playback ran out of data, leaving a gap in the played signal (Linux; always 0 on
    /// Windows, whose API does not report it).
    pub underruns: u32,
}

/// Plays interleaved `f32` samples on an output device and blocks until they have been played.
/// Intended for test signals (loopback verification, echo-path measurement).
pub fn play(
    device: &DeviceSelector,
    format: StreamFormat,
    samples: &[f32],
) -> Result<PlaybackReport> {
    #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
    {
        platform::play_blocking(device, format, samples)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = (device, format, samples);
        Err(CaptureError::Unsupported)
    }
}

/// Lists active devices of `kind`.
pub fn list_devices(kind: DeviceKind) -> Result<Vec<DeviceInfo>> {
    platform::list_devices(kind)
}

/// Starts capturing. Returns the handle and the consumer end of the audio ring.
pub fn start_capture(cfg: CaptureConfig) -> Result<(CaptureHandle, sd_core::ring::RingConsumer)> {
    if cfg.ring_ms < 100 {
        return Err(CaptureError::Config("ring_ms must be at least 100".into()));
    }
    let ring_frames = cfg.format.sample_rate as usize * cfg.ring_ms as usize / 1000;
    let max_blocks = (cfg.ring_ms as usize).max(64); // ≥ 1 block per ms of capacity
    let (producer, consumer) =
        sd_core::ring::audio_ring(cfg.format.channels, ring_frames, max_blocks)
            .map_err(|e| CaptureError::Config(e.to_string()))?;
    let (tx, rx) = crossbeam_channel::unbounded();
    let stats = Arc::new(CaptureStats::default());
    let format = cfg.format;
    let source = cfg.source;
    let inner = platform::start(cfg, producer, tx, stats.clone())?;
    Ok((
        CaptureHandle {
            inner: Some(inner),
            events: rx,
            stats,
            format,
            source,
        },
        consumer,
    ))
}

#[cfg(target_os = "macos")]
use coreaudio as platform;
#[cfg(target_os = "linux")]
use pulse as platform;
#[cfg(windows)]
use wasapi as platform;

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod platform {
    use super::*;

    pub struct Handle;
    impl Handle {
        pub fn set_device(&self, _d: DeviceSelector) {}
        pub fn stop(self) {}
    }
    pub fn list_devices(_k: DeviceKind) -> Result<Vec<DeviceInfo>> {
        Err(CaptureError::Unsupported)
    }
    pub fn start(
        _cfg: CaptureConfig,
        _p: sd_core::ring::RingProducer,
        _tx: crossbeam_channel::Sender<CaptureEvent>,
        _s: Arc<CaptureStats>,
    ) -> Result<Handle> {
        Err(CaptureError::Unsupported)
    }
}
