//! Helpers shared by the end-to-end capture tests (signals, a capture collector, checks).
#![allow(dead_code)] // each test file uses a different subset

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sd_capture::{
    CaptureConfig, CaptureEvent, CaptureHandle, DeviceSelector, PlaybackReport, start_capture,
};
use sd_core::{BlockFlags, HostTime, Source};

pub const RATE: u32 = 48_000;

pub fn enabled() -> bool {
    let on = std::env::var("SD_CAPTURE_E2E").is_ok_and(|v| v == "1");
    if !on {
        eprintln!("SD_CAPTURE_E2E is not set: skipped");
    }
    on
}

/// Deterministic noise, band-limited to below ~10 kHz by a short moving average (so server
/// resampling cannot change it much), at about -20 dBFS; identical on both channels.
pub fn signal(seconds: f64, channels: usize) -> Vec<f32> {
    let n = (seconds * RATE as f64) as usize;
    let mut state = 0x1234_5678u32;
    let raw: Vec<f32> = (0..n + 4)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect();
    let mono: Vec<f32> = (0..n)
        .map(|i| 0.2 * (raw[i] + raw[i + 1] + raw[i + 2] + raw[i + 3]) / 4.0)
        .collect();
    mono.iter()
        .flat_map(|&v| std::iter::repeat_n(v, channels))
        .collect()
}

/// Everything a capture delivered, placed on its timeline (mono, channels averaged).
#[derive(Default)]
pub struct Captured {
    /// Samples by timeline frame (skipped frames stay zero).
    pub mono: Vec<f32>,
    /// (first frame, host time) per block that came from the device.
    pub stamps: Vec<(u64, HostTime)>,
    pub gap_fill_frames: u64,
    pub blocks: usize,
    /// First frames that did not follow the previous block (timeline jumps).
    pub jumps: usize,
    /// The capture's counters when it stopped.
    pub stats: sd_capture::CaptureStatsSnapshot,
}

pub struct Capture {
    pub handle: CaptureHandle,
    stop: Arc<AtomicBool>,
    reader: JoinHandle<Captured>,
    started_at: Instant,
}

pub fn capture(source: Source, device: DeviceSelector) -> Capture {
    let mut cfg = CaptureConfig::new(source);
    cfg.device = device;
    let channels = cfg.format.channels as usize;
    let (handle, mut consumer) = start_capture(cfg).expect("start capture");
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    let reader = std::thread::spawn(move || {
        let mut c = Captured::default();
        let mut buf = Vec::new();
        let mut next: Option<u64> = None;
        loop {
            let done = s.load(Ordering::Acquire);
            while let Some(meta) = consumer.pop(&mut buf) {
                c.blocks += 1;
                if next.is_some_and(|n| n != meta.first_frame) {
                    c.jumps += 1;
                }
                next = Some(meta.first_frame + meta.frames as u64);
                let at = meta.first_frame as usize;
                let end = at + meta.frames as usize;
                if c.mono.len() < end {
                    c.mono.resize(end, 0.0);
                }
                for f in 0..meta.frames as usize {
                    let frame = &buf[f * channels..(f + 1) * channels];
                    c.mono[at + f] = frame.iter().sum::<f32>() / channels as f32;
                }
                if meta.flags.contains(BlockFlags::GAP_FILL) {
                    c.gap_fill_frames += meta.frames as u64;
                } else {
                    c.stamps.push((meta.first_frame, meta.host_time));
                }
            }
            if done {
                return c;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    Capture {
        handle,
        stop,
        reader,
        started_at: Instant::now(),
    }
}

impl Capture {
    /// Waits for an event matching `pred`, returning the events seen until then.
    pub fn wait_for(
        &self,
        timeout: Duration,
        pred: impl Fn(&CaptureEvent) -> bool,
    ) -> Vec<CaptureEvent> {
        let deadline = Instant::now() + timeout;
        let mut seen = Vec::new();
        while Instant::now() < deadline {
            if let Ok(e) = self.handle.events().recv_timeout(Duration::from_millis(50)) {
                let hit = pred(&e);
                seen.push(e);
                if hit {
                    return seen;
                }
            }
        }
        panic!("event not seen within {timeout:?}; events: {seen:?}");
    }

    pub fn finish(self) -> (Captured, Duration) {
        let elapsed = self.started_at.elapsed();
        let stats = self.handle.stats();
        self.handle.stop();
        self.stop.store(true, Ordering::Release);
        let mut c = self.reader.join().expect("reader");
        c.stats = stats;
        (c, elapsed)
    }
}

pub fn started_on(e: &CaptureEvent, id: &str) -> bool {
    matches!(e, CaptureEvent::Started { device_id, .. } if device_id == id)
}

/// Normalised correlation of `a` and `b` (same length).
pub fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa * bb).sqrt().max(1e-12)
}

/// Where `reference` (mono) starts in `captured`, by a 100 ms probe from its middle.
pub fn find(reference: &[f32], captured: &[f32]) -> (usize, f64) {
    let probe_at = reference.len() / 2;
    let probe = &reference[probe_at..probe_at + RATE as usize / 10];
    let mut best = (0usize, f64::MIN);
    for lag in 0..captured.len().saturating_sub(probe.len()) {
        let c = correlation(probe, &captured[lag..lag + probe.len()]);
        if c > best.1 {
            best = (lag, c);
        }
    }
    (best.0.saturating_sub(probe_at), best.1)
}

/// Checks that the played signal is in the capture, intact, at the time the timestamps say.
pub fn verify(name: &str, played: &[f32], report: &PlaybackReport, c: &Captured) {
    let first_played = report.first_frame_host_time;
    let (start, probe_corr) = find(played, &c.mono);
    assert!(
        probe_corr > 0.95,
        "{name}: signal not found (best probe correlation {probe_corr:.3})"
    );
    assert!(
        start + played.len() <= c.mono.len(),
        "{name}: capture ends inside the signal"
    );
    // Every 100 ms window must match: a lost or duplicated packet breaks one. The first window
    // is not checked: PulseAudio rewinds a sink when a stream starts playing, and the monitor
    // does not see the rewound start (a server property; PipeWire has no rewinds).
    let win = RATE as usize / 10;
    let corr: Vec<f64> = (1..played.len() / win)
        .map(|w| {
            let r = &played[w * win..(w + 1) * win];
            correlation(r, &c.mono[start + w * win..start + (w + 1) * win])
        })
        .collect();
    let worst = corr.iter().copied().fold(f64::MAX, f64::min);
    // For mismatching windows: where the signal actually is (within 200 ms). A later shift
    // means the played signal paused (playback underrun); an earlier one, captured frames
    // missing from the timeline.
    let reach = RATE as usize / 5;
    let bad: Vec<String> = corr
        .iter()
        .enumerate()
        .filter(|(_, c)| **c <= 0.95)
        .take(6)
        .map(|(k, cw)| {
            let w = k + 1; // corr starts at window 1
            let r = &played[w * win..(w + 1) * win];
            let at = start + w * win;
            let (mut best, mut best_c) = (0i64, f64::MIN);
            for lag in at.saturating_sub(reach)..(at + reach).min(c.mono.len() - win) {
                let cc = correlation(r, &c.mono[lag..lag + win]);
                if cc > best_c {
                    best = lag as i64 - at as i64;
                    best_c = cc;
                }
            }
            format!(
                "{:.1}s:{cw:.2}->shift {:+.1} ms ({best_c:.2})",
                (w * win) as f64 / RATE as f64,
                best as f64 * 1000.0 / RATE as f64
            )
        })
        .collect();
    if !bad.is_empty() {
        println!(
            "{name}: {} of {} windows mismatch; first: {}",
            corr.iter().filter(|c| **c <= 0.95).count(),
            corr.len(),
            bad.join(", ")
        );
    }
    // Alignment: the timeline frame where the timestamps put the first played frame.
    let (f0, t0) = c.stamps[0];
    let predicted =
        f0 as f64 + (first_played.as_ns() as f64 - t0.as_ns() as f64) * RATE as f64 / 1e9;
    let offset_ms = (start as f64 - predicted) * 1000.0 / RATE as f64;
    println!(
        "{name}: signal at frame {start}, probe correlation {probe_corr:.4}, worst 100 ms window {worst:.4}, \
         timestamps vs audio {offset_ms:+.1} ms, blocks {}, gap fill {} frames, timeline jumps {}, \
         device glitches {}, gap {} idle {} trimmed {} frames, player skips {}",
        c.blocks,
        c.gap_fill_frames,
        c.jumps,
        c.stats.device_glitches,
        c.stats.gap_frames,
        c.stats.idle_frames,
        c.stats.trimmed_frames,
        report.underruns
    );
    assert_eq!(
        report.underruns, 0,
        "{name}: the test player ran dry or the device skipped a cycle, so the played signal itself has gaps"
    );
    assert!(
        worst > 0.95,
        "{name}: a window does not match ({worst:.3}): loss or glitch"
    );
    assert!(
        offset_ms.abs() < 40.0,
        "{name}: timestamps are {offset_ms:+.1} ms off the audio"
    );
}

pub fn mono_of(stereo: &[f32]) -> Vec<f32> {
    stereo
        .chunks_exact(2)
        .map(|f| (f[0] + f[1]) / 2.0)
        .collect()
}
