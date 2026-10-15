//! Real-time processing stage: aligns mic and system audio on host time, runs the mic chain
//! (echo gate + AEC), converts both sources to 16 kHz mono, runs VAD, and publishes processed
//! audio and speech segments.
//!
//! It runs on its own thread and receives copies of drained blocks through a bounded queue, so
//! it can never stall capture or journaling. If it falls behind, blocks are dropped and counted
//! (the journal still has everything for later reprocessing).
//!
//! **Alignment and drift.** For every 10 ms mic frame, the reference (system audio) position is
//! looked up by *host time* using the system stream's per-block timestamps. A phase-locked read
//! pointer follows that target: it advances by one frame's worth of samples plus a small
//! correction proportional to the error, and the reference is read at fractional positions with
//! linear interpolation. Clock drift is therefore followed smoothly (whole-sample slips measured
//! 20 dB ERLE at 200 ppm; see the tests), and errors beyond 2 ms (gaps, restarts) resync at once.
//! With no drift the pointer advances exactly one sample per sample and reads are exact.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use sd_core::{BlockMeta, HostTime, Source, StreamFormat};
use sd_dsp::SystemPath;
use sd_dsp::mic::{EchoState, MicConfig, MicProcessor};
use sd_vad::{Segmenter, SegmenterConfig, SileroVad};

use crate::RecorderEvent;
use sd_asr::runner::AsrInput;

/// The engine's source as speech recognition's source.
pub(crate) fn asr_source(s: Source) -> sd_asr::stream::Source {
    match s {
        Source::Mic => sd_asr::stream::Source::Mic,
        Source::System => sd_asr::stream::Source::System,
    }
}

/// Processing options. `vad_model = None` disables VAD.
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub mic: MicConfig,
    pub vad_model: Option<PathBuf>,
    pub segmenter: SegmenterConfig,
    /// Processed audio is published in chunks of this many 10 ms frames.
    pub chunk_frames: usize,
    /// Capacity of the processed-audio channel, in chunks; a consumer that does not keep up
    /// loses the oldest-arriving chunks (counted).
    pub output_capacity: usize,
    /// Mic audio is held at most this long waiting for the matching reference before it is
    /// processed with silence as the reference (e.g. the system stream stalled).
    pub max_reference_wait_ms: u32,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            mic: MicConfig::default(),
            vad_model: Some(PathBuf::from("models/silero_vad_16k_op15.onnx")),
            segmenter: SegmenterConfig::default(),
            chunk_frames: 10,
            output_capacity: 600,
            max_reference_wait_ms: 400,
        }
    }
}

/// 16 kHz mono processed audio for one source.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedChunk {
    pub source: Source,
    /// Index of the first sample on the source's 16 kHz processed timeline.
    pub first_sample: u64,
    /// Host time the first sample's *content* was captured (processing latency removed).
    pub host_time: HostTime,
    pub samples: Vec<f32>,
}

/// Counters for the processing stage.
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub frames: [AtomicU64; 2],
    pub dropped_input_blocks: AtomicU64,
    pub dropped_output_chunks: AtomicU64,
    /// Relative rate of the reference vs. the mic (ppm, f64 bits) as tracked by the read loop.
    pub reference_ppm_bits: AtomicU64,
    pub reference_resyncs: AtomicU64,
    /// Mic frames processed without a reference because it did not arrive in time while the
    /// system stream was running.
    pub reference_timeouts: AtomicU64,
    /// Mic frames after the system stream's last audio when the session ended (the two
    /// streams never stop at exactly the same instant); processed with a silent reference.
    pub reference_ended_frames: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProcessingSnapshot {
    pub mic_frames: u64,
    pub system_frames: u64,
    pub dropped_input_blocks: u64,
    pub dropped_output_chunks: u64,
    pub reference_ppm: f64,
    pub reference_resyncs: u64,
    pub reference_timeouts: u64,
    pub reference_ended_frames: u64,
}

impl ProcessingStats {
    pub fn snapshot(&self) -> ProcessingSnapshot {
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        ProcessingSnapshot {
            mic_frames: l(&self.frames[0]),
            system_frames: l(&self.frames[1]),
            dropped_input_blocks: l(&self.dropped_input_blocks),
            dropped_output_chunks: l(&self.dropped_output_chunks),
            reference_ppm: f64::from_bits(l(&self.reference_ppm_bits)),
            reference_resyncs: l(&self.reference_resyncs),
            reference_timeouts: l(&self.reference_timeouts),
            reference_ended_frames: l(&self.reference_ended_frames),
        }
    }
}

/// A block handed from the drain thread to processing.
pub(crate) struct ProcBlock {
    pub source: Source,
    pub meta: BlockMeta,
    pub samples: Vec<f32>,
}

/// Interleaved samples of one source with their timeline position and host-time anchors.
struct Fifo {
    format: StreamFormat,
    data: VecDeque<f32>,
    /// Timeline frame index of `data[0]`.
    base: u64,
    /// One past the last frame received.
    end: u64,
    started: bool,
    /// (frame, host ns) from recent blocks, oldest first.
    anchors: VecDeque<(u64, u64)>,
}

impl Fifo {
    fn new(format: StreamFormat) -> Self {
        Self {
            format,
            data: VecDeque::new(),
            base: 0,
            end: 0,
            started: false,
            anchors: VecDeque::new(),
        }
    }

    fn push(&mut self, meta: &BlockMeta, samples: &[f32]) {
        let ch = self.format.channels as usize;
        if !self.started {
            self.started = true;
            self.base = meta.first_frame;
            self.end = meta.first_frame;
        }
        if meta.first_frame > self.end {
            // Dropped upstream (ring or queue overflow): keep the timeline with silence.
            let gap = (meta.first_frame - self.end) as usize * ch;
            self.data.extend(std::iter::repeat_n(0.0, gap));
        }
        if meta.first_frame >= self.end {
            self.data.extend(samples.iter().copied());
            self.end = meta.first_frame + meta.frames as u64;
        }
        self.anchors
            .push_back((meta.first_frame, meta.host_time.as_ns()));
    }

    fn frames_available_from(&self, frame: u64) -> u64 {
        self.end.saturating_sub(frame)
    }

    /// Host time of `frame` from the nearest preceding anchor.
    fn host_of(&self, frame: u64) -> Option<u64> {
        let &(af, ah) = self
            .anchors
            .iter()
            .rev()
            .find(|(f, _)| *f <= frame)
            .or(self.anchors.front())?;
        let d = (frame as i128 - af as i128) * 1_000_000_000 / self.format.sample_rate as i128;
        Some((ah as i128 + d).max(0) as u64)
    }

    /// Fractional frame captured at host time `t`.
    fn frame_at_f64(&self, t: u64) -> Option<f64> {
        let &(af, ah) = self
            .anchors
            .iter()
            .rev()
            .find(|(_, h)| *h <= t)
            .or(self.anchors.front())?;
        Some(af as f64 + (t as f64 - ah as f64) * self.format.sample_rate as f64 / 1e9)
    }

    /// Reads `n` frames at fractional positions `pos + k * spacing`, interpolating with a 4-point
    /// cubic (Catmull-Rom) kernel. Integer positions are returned exactly.
    fn read_frac(&self, pos: f64, spacing: f64, n: usize, out: &mut [f32]) {
        let ch = self.format.channels as usize;
        let sample = |f: i64, c: usize| -> f32 {
            if f >= self.base as i64 && f < self.end as i64 {
                self.data[((f as u64 - self.base) as usize) * ch + c]
            } else {
                0.0
            }
        };
        for k in 0..n {
            let p = pos + k as f64 * spacing;
            let i = p.floor();
            let t = (p - i) as f32;
            let i = i as i64;
            for c in 0..ch {
                let y1 = sample(i, c);
                out[k * ch + c] = if t == 0.0 {
                    y1
                } else {
                    let (y0, y2, y3) = (sample(i - 1, c), sample(i + 1, c), sample(i + 2, c));
                    let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                    let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                    let cc = -0.5 * y0 + 0.5 * y2;
                    ((a * t + b) * t + cc) * t + y1
                };
            }
        }
    }

    /// Copies `n` frames starting at `frame` into `out` (zeros where unavailable).
    fn read(&self, frame: u64, n: usize, out: &mut [f32]) {
        let ch = self.format.channels as usize;
        for (k, o) in out.iter_mut().enumerate().take(n * ch) {
            let f = frame + (k / ch) as u64;
            *o = if f >= self.base && f < self.end {
                self.data[((f - self.base) as usize) * ch + k % ch]
            } else {
                0.0
            };
        }
    }

    /// Discards everything before `frame`.
    fn trim_before(&mut self, frame: u64) {
        if frame > self.base {
            let n = ((frame.min(self.end) - self.base) as usize) * self.format.channels as usize;
            self.data.drain(..n);
            self.base = frame.min(self.end);
        }
        // Anchors live exactly as long as their audio (the one covering `base` is kept). A
        // count cap would drop anchors still needed when one source runs far ahead in a burst.
        while self.anchors.len() > 1 && self.anchors[1].0 <= self.base {
            self.anchors.pop_front();
        }
    }
}

struct SourceOut {
    source: Source,
    vad: Option<SileroVad>,
    segmenter: Segmenter,
    probs: Vec<f32>,
    pending: Vec<f32>,
    pending_first: u64,
    pending_host: u64,
    next_sample: u64,
    /// Per emitted 10 ms frame (from `unverified_base`): echo possibly present but not removed.
    unverified: VecDeque<bool>,
    unverified_base: u64,
}

/// Frames of `echo_unverified` history kept for segments still open (VAD segments are far
/// shorter; a longer segment is checked against what remains).
const UNVERIFIED_HISTORY: usize = 6_000;

impl SourceOut {
    /// Whether any 10 ms frame overlapping `[start, end)` (16 kHz samples) was unverified.
    fn unverified_in(&self, start: u64, end: u64) -> bool {
        let first = (start / 160).max(self.unverified_base);
        let last = end.div_ceil(160);
        (first..last).any(|f| {
            self.unverified
                .get((f - self.unverified_base) as usize)
                .copied()
                .unwrap_or(false)
        })
    }
}

pub(crate) struct Stage {
    cfg: ProcessingConfig,
    mic: Option<(Fifo, MicProcessor)>,
    sys: Option<(Fifo, SystemPath)>,
    outs: Vec<SourceOut>,
    mic_next: u64,
    /// Read position of the AEC reference in the system stream (fractional, phase-locked).
    sys_pos: Option<f64>,
    /// Read position of the 16 kHz system output (independent of the reference).
    sys_out_next: Option<u64>,
    events: Sender<RecorderEvent>,
    audio: Sender<ProcessedChunk>,
    stats: Arc<ProcessingStats>,
    aec_active: bool,
    /// Smoothed relative rate of the reference read loop, ppm.
    ref_ppm: f64,
    ref_buf: Vec<f32>,
    mic_buf: Vec<f32>,
    out_buf: Vec<f32>,
    /// Live stage: giving up on a late reference also requires the wait to have lasted
    /// `max_reference_wait_ms` of wall-clock time. Offline (blocks in host-time order) uses
    /// host time only, which keeps it deterministic.
    realtime: bool,
    /// When the current wait for the reference began (wall clock, live only).
    waiting_since: Option<Instant>,
    /// Speech recognition input: processed audio and VAD events, if transcription is on.
    asr: Option<Sender<AsrInput>>,
}

impl Stage {
    pub(crate) fn new(
        cfg: ProcessingConfig,
        formats: &[(Source, StreamFormat)],
        events: Sender<RecorderEvent>,
        audio: Sender<ProcessedChunk>,
        stats: Arc<ProcessingStats>,
    ) -> Result<Self, String> {
        let fmt_of = |s: Source| formats.iter().find(|(x, _)| *x == s).map(|(_, f)| *f);
        let sys_fmt = fmt_of(Source::System);
        let mic = match fmt_of(Source::Mic) {
            Some(f) => {
                let mut mc = cfg.mic.clone();
                mc.input_rate = f.sample_rate;
                mc.reference_channels = sys_fmt.map_or(1, |s| s.channels);
                Some((
                    Fifo::new(f),
                    MicProcessor::new(mc).map_err(|e| e.to_string())?,
                ))
            }
            None => None,
        };
        let sys = match sys_fmt {
            Some(f) => Some((
                Fifo::new(f),
                SystemPath::new(f.sample_rate, f.channels).map_err(|e| e.to_string())?,
            )),
            None => None,
        };
        let mut outs = Vec::new();
        for &(source, _) in formats {
            // VAD is optional: if the model or the ONNX Runtime library is unavailable the
            // engine keeps recording and processing audio, and reports why.
            let vad = match &cfg.vad_model {
                Some(p) => match SileroVad::new(p) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        let _ = events.send(RecorderEvent::Warning {
                            message: format!("VAD disabled for {source}: {e}"),
                        });
                        None
                    }
                },
                None => None,
            };
            outs.push(SourceOut {
                source,
                vad,
                segmenter: Segmenter::new(cfg.segmenter),
                probs: Vec::new(),
                pending: Vec::new(),
                pending_first: 0,
                pending_host: 0,
                next_sample: 0,
                unverified: VecDeque::new(),
                unverified_base: 0,
            });
        }
        let ref_len = sys_fmt.map_or(480, |f| {
            (f.sample_rate / 100) as usize * f.channels as usize
        });
        let mic_len = fmt_of(Source::Mic).map_or(480, |f| (f.sample_rate / 100) as usize);
        Ok(Self {
            cfg,
            mic,
            sys,
            outs,
            mic_next: 0,
            sys_pos: None,
            sys_out_next: None,
            events,
            audio,
            stats,
            aec_active: false,
            ref_ppm: 0.0,
            ref_buf: vec![0.0; ref_len],
            mic_buf: vec![0.0; mic_len],
            out_buf: vec![0.0; 160],
            realtime: true,
            waiting_since: None,
            asr: None,
        })
    }

    /// Feeds processed audio and VAD events to speech recognition from now on.
    pub(crate) fn set_asr(&mut self, tx: Sender<AsrInput>) {
        self.asr = Some(tx);
    }

    /// Sends a recorder event; speech events also go to speech recognition.
    fn send_event(&self, ev: RecorderEvent) {
        if let Some(tx) = &self.asr {
            let input = match &ev {
                RecorderEvent::SpeechStart {
                    source,
                    start_sample,
                } => Some(AsrInput::SpeechStart {
                    source: asr_source(*source),
                    start: *start_sample,
                }),
                RecorderEvent::SpeechDiscarded { source, .. } => Some(AsrInput::SpeechDiscarded {
                    source: asr_source(*source),
                }),
                RecorderEvent::Speech {
                    source,
                    start_sample,
                    end_sample,
                    echo_unverified,
                } => Some(AsrInput::SpeechEnd {
                    source: asr_source(*source),
                    start: *start_sample,
                    end: *end_sample,
                    echo_unverified: *echo_unverified,
                }),
                _ => None,
            };
            if let Some(i) = input {
                let _ = tx.send(i);
            }
        }
        let _ = self.events.send(ev);
    }

    /// Publishes a processed chunk to the application and to speech recognition.
    fn send_chunk(&self, chunk: ProcessedChunk, at_finish: bool) {
        if let Some(tx) = &self.asr {
            let _ = tx.send(AsrInput::Audio {
                source: asr_source(chunk.source),
                first_sample: chunk.first_sample,
                samples: chunk.samples.clone(),
            });
        }
        if at_finish {
            let _ = self.audio.try_send(chunk);
        } else if let Err(TrySendError::Full(_)) = self.audio.try_send(chunk) {
            self.stats
                .dropped_output_chunks
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether to keep waiting for the reference of the current mic frame. `ref_lag_ms` is how
    /// far the reference trails the mic in host time. Live, that alone can mislead: after a
    /// scheduling stall the drain thread may deliver a burst of one source's blocks before the
    /// other's. So the live stage gives up only once the reference has also been missing for
    /// the same time on the wall clock.
    fn keep_waiting(&mut self, ref_lag_ms: u64) -> bool {
        let max = self.cfg.max_reference_wait_ms as u64;
        if ref_lag_ms < max {
            return true;
        }
        if !self.realtime {
            return false;
        }
        let since = *self.waiting_since.get_or_insert_with(Instant::now);
        since.elapsed() < Duration::from_millis(max)
    }

    /// Adds a block without processing (the thread drains its queue first, then runs once).
    pub(crate) fn add(&mut self, b: ProcBlock) {
        match b.source {
            Source::Mic => {
                if let Some((f, _)) = &mut self.mic {
                    if !f.started {
                        self.mic_next = b.meta.first_frame;
                    }
                    f.push(&b.meta, &b.samples);
                }
            }
            Source::System => {
                if let Some((f, _)) = &mut self.sys {
                    f.push(&b.meta, &b.samples);
                }
            }
        }
    }

    /// Processes everything that can be processed; with `flush`, does not wait for references.
    pub(crate) fn run(&mut self, flush: bool) {
        if self.mic.is_some() {
            self.run_mic(flush);
        }
        self.run_system_out();
        // Keep 1 s behind the slowest reader so the reference can slip backwards.
        if let Some((sf, _)) = &mut self.sys {
            let keep = [self.sys_pos.map(|p| p.max(0.0) as u64), self.sys_out_next]
                .into_iter()
                .flatten()
                .min();
            if let Some(k) = keep {
                sf.trim_before(k.saturating_sub(sf.format.sample_rate as u64));
            }
        }
    }

    fn run_mic(&mut self, flush: bool) {
        loop {
            let Some((mf, _)) = &self.mic else { return };
            let n = mf.format.sample_rate as u64 / 100;
            if mf.frames_available_from(self.mic_next) < n {
                return;
            }
            let Some(t) = mf.host_of(self.mic_next) else {
                return;
            };
            // How far (in host time) the system stream's newest audio trails the mic's newest.
            // Waiting for the reference is bounded by this lag, not by how much mic audio is
            // buffered, so a burst on one source (e.g. after a scheduling hiccup) is not
            // mistaken for a stalled reference.
            let mic_latest = mf.host_of(mf.end).unwrap_or(t);
            let sys_latest = self
                .sys
                .as_ref()
                .filter(|(sf, _)| sf.started)
                .and_then(|(sf, _)| sf.host_of(sf.end));
            let ref_lag_ms = match sys_latest {
                Some(sl) => mic_latest.saturating_sub(sl) / 1_000_000,
                None => {
                    mic_latest.saturating_sub(mf.host_of(self.mic_next).unwrap_or(t)) / 1_000_000
                }
            };

            // Reference frames for this mic frame (by host time).
            let mut have_ref = false;
            if let Some((sf, _)) = &self.sys
                && sf.started
                && let Some(target) = sf.frame_at_f64(t)
            {
                let sn = sf.format.sample_rate as u64 / 100;
                if target < 0.0 {
                    // This mic frame predates the system stream: the reference is silence.
                    self.ref_buf.iter_mut().for_each(|v| *v = 0.0);
                    have_ref = true;
                } else {
                    let (pos, resync) = match self.sys_pos {
                        None => (target, false),
                        // Beyond 2 ms: an event (gap, restart), not drift.
                        Some(cur)
                            if (target - cur).abs() > sf.format.sample_rate as f64 / 500.0 =>
                        {
                            (target, true)
                        }
                        Some(cur) => (cur, false),
                    };
                    // Phase-locked advance: one frame plus a correction toward the target,
                    // bounded to ±0.5 sample per frame (±1000 ppm at 48 kHz).
                    let err = target - pos;
                    let advance = sn as f64 + (0.05 * err).clamp(-0.5, 0.5);
                    // Cubic interpolation reads 2 frames ahead. When flushing nothing more will
                    // arrive, so whatever reference remains is used (missing samples read as 0).
                    let avail = sf.frames_available_from(pos.floor() as u64);
                    // The target was derived from the newest system block starting at or before
                    // `t`. Only once a block starting *after* `t` has arrived is that block
                    // certain to be present (blocks of one source arrive in order), which makes
                    // the target, and so the output, independent of delivery timing: live and
                    // offline replay then agree bit for bit.
                    let settled = flush || sf.anchors.back().is_some_and(|&(_, h)| h > t);
                    if settled && (avail >= sn + 2 || (flush && avail > 0)) {
                        sf.read_frac(pos, advance / sn as f64, sn as usize, &mut self.ref_buf);
                        if resync {
                            self.stats.reference_resyncs.fetch_add(1, Ordering::Relaxed);
                        }
                        self.sys_pos = Some(pos + advance);
                        let ppm = (advance / sn as f64 - 1.0) * 1e6;
                        self.ref_ppm = 0.99 * self.ref_ppm + 0.01 * ppm;
                        self.stats
                            .reference_ppm_bits
                            .store(self.ref_ppm.to_bits(), Ordering::Relaxed);
                        have_ref = true;
                    } else if !flush && self.keep_waiting(ref_lag_ms) {
                        return; // wait for the reference to arrive
                    }
                }
            }
            if !have_ref
                && self.sys.as_ref().is_some_and(|(sf, _)| !sf.started)
                && !flush
                && self.keep_waiting(ref_lag_ms)
            {
                return; // system stream not started yet: wait briefly for it
            }
            if have_ref {
                self.waiting_since = None;
            } else {
                if self.sys.is_some() {
                    // When flushing, no more reference audio will arrive: the system stream
                    // ended first (or never started). Otherwise the reference was late.
                    let counter = if flush {
                        &self.stats.reference_ended_frames
                    } else {
                        &self.stats.reference_timeouts
                    };
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.ref_buf.iter_mut().for_each(|v| *v = 0.0);
            }

            let (mf, mp) = self.mic.as_mut().expect("mic");
            mf.read(self.mic_next, n as usize, &mut self.mic_buf);
            let ok = mp
                .process(&self.mic_buf, &self.ref_buf, &mut self.out_buf)
                .is_ok();
            let latency_ns = mp.latency() as u64 * 1_000_000_000 / 16_000;
            let status = mp.status();
            self.mic_next += n;
            mf.trim_before(self.mic_next);
            self.stats.frames[0].fetch_add(1, Ordering::Relaxed);
            if status.aec_active != self.aec_active {
                self.aec_active = status.aec_active;
                let _ = self.events.send(RecorderEvent::EchoCanceller {
                    active: status.aec_active,
                    correlation: status.echo_correlation,
                });
            }
            let out = std::mem::take(&mut self.out_buf);
            if ok {
                let unverified = status.echo_state == EchoState::Unknown && status.far_active;
                self.emit(Source::Mic, &out, t.saturating_sub(latency_ns), unverified);
            }
            self.out_buf = out;
        }
    }

    /// Produces the 16 kHz system output sequentially, independent of the AEC reference.
    fn run_system_out(&mut self) {
        let mut buf = vec![0.0f32; self.ref_buf.len()];
        loop {
            let Some((sf, sp)) = self.sys.as_mut() else {
                return;
            };
            if !sf.started {
                return;
            }
            let n = sf.format.sample_rate as u64 / 100;
            let next = *self.sys_out_next.get_or_insert(sf.base);
            if sf.frames_available_from(next) < n {
                return;
            }
            sf.read(next, n as usize, &mut buf);
            let t = sf.host_of(next).unwrap_or(0);
            self.sys_out_next = Some(next + n);
            let mut out = [0.0f32; 160];
            if sp.process(&buf, &mut out).is_ok() {
                self.stats.frames[1].fetch_add(1, Ordering::Relaxed);
                self.emit(Source::System, &out, t, false);
            }
        }
    }

    /// Runs VAD and publishes processed audio in chunks.
    fn emit(&mut self, source: Source, frame16k: &[f32], host_ns: u64, unverified: bool) {
        let Some(o) = self.outs.iter_mut().find(|o| o.source == source) else {
            return;
        };
        o.unverified.push_back(unverified);
        while o.unverified.len() > UNVERIFIED_HISTORY {
            o.unverified.pop_front();
            o.unverified_base += 1;
        }
        let mut speech = Vec::new();
        if let Some(vad) = &mut o.vad {
            o.probs.clear();
            if vad.push(frame16k, &mut o.probs).is_ok() {
                for &p in &o.probs {
                    let was_open = o.segmenter.open_start();
                    let done = o.segmenter.push(p);
                    match (was_open, o.segmenter.open_start(), &done) {
                        (None, Some(start), _) => speech.push(RecorderEvent::SpeechStart {
                            source,
                            start_sample: start,
                        }),
                        (Some(start), None, None) => speech.push(RecorderEvent::SpeechDiscarded {
                            source,
                            start_sample: start,
                        }),
                        _ => {}
                    }
                    if let Some(seg) = done {
                        speech.push(RecorderEvent::Speech {
                            source,
                            start_sample: seg.start,
                            end_sample: seg.end,
                            echo_unverified: o.unverified_in(seg.start, seg.end),
                        });
                    }
                }
            }
        }
        if o.pending.is_empty() {
            o.pending_first = o.next_sample;
            o.pending_host = host_ns;
        }
        o.pending.extend_from_slice(frame16k);
        o.next_sample += frame16k.len() as u64;
        let chunk = (o.pending.len() >= self.cfg.chunk_frames * 160).then(|| ProcessedChunk {
            source,
            first_sample: o.pending_first,
            host_time: HostTime::from_ns(o.pending_host),
            samples: std::mem::take(&mut o.pending),
        });
        // Audio before the VAD events it completes, so recognition never waits needlessly.
        if let Some(chunk) = chunk {
            self.send_chunk(chunk, false);
        }
        for ev in speech {
            self.send_event(ev);
        }
    }

    /// Flushes open VAD segments and partial chunks at the end of a session.
    pub(crate) fn finish(&mut self) {
        self.run(true);
        let mut tail = Vec::new();
        for o in &mut self.outs {
            let chunk = (!o.pending.is_empty()).then(|| ProcessedChunk {
                source: o.source,
                first_sample: o.pending_first,
                host_time: HostTime::from_ns(o.pending_host),
                samples: std::mem::take(&mut o.pending),
            });
            let speech = o.segmenter.flush().map(|seg| RecorderEvent::Speech {
                source: o.source,
                start_sample: seg.start,
                end_sample: seg.end,
                echo_unverified: o.unverified_in(seg.start, seg.end),
            });
            tail.push((chunk, speech));
        }
        for (chunk, speech) in tail {
            if let Some(c) = chunk {
                self.send_chunk(c, true);
            }
            if let Some(ev) = speech {
                self.send_event(ev);
            }
        }
        // Dropping the sender lets the recognition thread finish its remaining work.
        self.asr = None;
    }
}

pub(crate) fn processing_thread(mut stage: Stage, rx: Receiver<ProcBlock>) {
    while let Ok(b) = rx.recv() {
        stage.add(b);
        // Take everything already queued (both sources) before aligning.
        while let Ok(b) = rx.try_recv() {
            stage.add(b);
        }
        stage.run(false);
    }
    stage.finish();
}

/// Runs the processing stage over already-recorded audio (e.g. a journal), as fast as the CPU
/// allows. Used for catch-up after an overload or crash, for reprocessing with different
/// settings, and for evaluation. Produces exactly what the live stage would.
pub struct OfflineProcessor {
    stage: Stage,
    audio: Receiver<ProcessedChunk>,
    events: Receiver<RecorderEvent>,
    events_tx: Sender<RecorderEvent>,
    stats: Arc<ProcessingStats>,
    asr: Option<std::thread::JoinHandle<()>>,
}

impl OfflineProcessor {
    pub fn new(cfg: ProcessingConfig, formats: &[(Source, StreamFormat)]) -> Result<Self, String> {
        let (atx, arx) = crossbeam_channel::unbounded();
        let (etx, erx) = crossbeam_channel::unbounded();
        let stats = Arc::new(ProcessingStats::default());
        let mut stage = Stage::new(cfg, formats, etx.clone(), atx, stats.clone())?;
        stage.realtime = false;
        Ok(Self {
            stage,
            audio: arx,
            events: erx,
            events_tx: etx,
            stats,
            asr: None,
        })
    }

    /// Adds one block of interleaved samples (any order across sources; timestamps align them).
    pub fn push(&mut self, source: Source, meta: BlockMeta, samples: &[f32]) {
        self.stage.add(ProcBlock {
            source,
            meta,
            samples: samples.to_vec(),
        });
        self.stage.run(false);
    }

    /// Takes the processed audio and events produced so far.
    pub fn drain(&self) -> (Vec<ProcessedChunk>, Vec<RecorderEvent>) {
        (
            self.audio.try_iter().collect(),
            self.events.try_iter().collect(),
        )
    }

    /// Transcribes the processed audio as it is produced, exactly as the live recorder does.
    /// Results arrive as [`RecorderEvent::Asr`] from [`OfflineProcessor::drain`] and
    /// [`OfflineProcessor::finish`] (which waits for recognition to complete).
    pub fn with_asr(mut self, setup: crate::AsrSetup) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        self.stage.set_asr(tx);
        self.asr = Some(crate::spawn_asr(setup, rx, self.events_tx.clone()));
        self
    }

    /// Flushes everything (open speech segments, partial chunks) and returns the remainder.
    pub fn finish(mut self) -> (Vec<ProcessedChunk>, Vec<RecorderEvent>, ProcessingSnapshot) {
        self.stage.finish();
        if let Some(h) = self.asr.take() {
            let _ = h.join();
        }
        let (a, e) = self.drain();
        (a, e, self.stats.snapshot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unverified_lookup_covers_overlapping_frames() {
        let o = SourceOut {
            source: Source::Mic,
            vad: None,
            segmenter: Segmenter::new(SegmenterConfig::default()),
            probs: Vec::new(),
            pending: Vec::new(),
            pending_first: 0,
            pending_host: 0,
            next_sample: 0,
            // Frames 10..14; only frame 12 unverified.
            unverified: [false, false, true, false].into_iter().collect(),
            unverified_base: 10,
        };
        assert!(!o.unverified_in(10 * 160, 12 * 160), "frames 10-11");
        assert!(
            o.unverified_in(11 * 160 + 5, 12 * 160 + 1),
            "touches frame 12"
        );
        assert!(!o.unverified_in(0, 10 * 160), "before the kept history");
        assert!(!o.unverified_in(13 * 160, 20 * 160), "after it");
    }
}
