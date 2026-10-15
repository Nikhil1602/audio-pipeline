//! The platform-independent half of every backend: device packets in, a continuous timeline
//! into the ring out.
//!
//! A backend reads packets from the OS (with a host timestamp, a device position if it has
//! one, and flags) and hands them to [`Feed::packet`]; while the device delivers nothing it
//! calls [`Feed::fill_idle`]. The feed keeps the timeline continuous with
//! [`TimelineAligner`]: gaps are filled with silence (within the [`GapFillPolicy`] budget) or
//! skipped, overlaps are trimmed, and idle periods become silence. Counters go to
//! [`CaptureStats`].

use std::sync::Arc;
use std::sync::atomic::Ordering;

use sd_core::ring::RingProducer;
use sd_core::timeline::{AlignerConfig, TimelineAligner};
use sd_core::{BlockFlags, HostTime, StreamFormat};

use crate::policy::{GapAction, GapFillPolicy};
use crate::{CaptureConfig, CaptureStats};

/// One packet from the device.
pub(crate) struct Packet<'a> {
    /// Host time of the first frame.
    pub t: HostTime,
    pub frames: u32,
    /// Device position of the first frame, in stream frames, if the API reports one.
    pub position: Option<f64>,
    /// `DISCONTINUITY` (device glitch), `TIMESTAMP_ESTIMATED`, `SILENT`.
    pub flags: BlockFlags,
    /// Interleaved samples (`frames * channels`); `None` for a silent packet.
    pub data: Option<&'a [f32]>,
}

/// When to fill silence for a device that delivers nothing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Idle {
    pub after_ms: u64,
    pub margin_ms: u64,
}

pub(crate) struct Feed {
    format: StreamFormat,
    max_gap_fill_ms: u32,
    producer: RingProducer,
    stats: Arc<CaptureStats>,
    aligner: TimelineAligner,
    gaps: GapFillPolicy,
    zeros: Vec<f32>,
}

impl Feed {
    /// `idle` is how long the device may deliver nothing before silence is synthesized, and
    /// how far behind now that silence stays (see [`Idle`]).
    pub fn new(
        cfg: &CaptureConfig,
        producer: RingProducer,
        stats: Arc<CaptureStats>,
        idle: Idle,
    ) -> Self {
        let mut aligner_cfg = AlignerConfig::new(cfg.format.sample_rate);
        aligner_cfg.idle_after_ns = idle.after_ms * 1_000_000;
        aligner_cfg.idle_margin_ns = idle.margin_ms * 1_000_000;
        let ch = cfg.format.channels as usize;
        Self {
            format: cfg.format,
            max_gap_fill_ms: cfg.max_gap_fill_ms,
            producer,
            stats,
            aligner: TimelineAligner::new(aligner_cfg),
            gaps: GapFillPolicy::new(
                cfg.format.sample_rate,
                cfg.max_gap_fill_ms,
                cfg.max_session_gap_fill_ms,
            ),
            zeros: vec![0.0; (cfg.format.sample_rate as usize / 10).max(4_800) * ch],
        }
    }

    /// Starts the timeline at `t` (both sources start at capture start, so frame 0 means the
    /// same instant).
    pub fn start_at(&mut self, t: HostTime) {
        self.aligner.start_at(t);
    }

    /// A new device stream begins (after a switch or restart): its positions restart.
    pub fn new_stream(&mut self) {
        self.aligner.new_stream();
    }

    fn ns_of(&self, frames: u64) -> u64 {
        self.format.frames_to_ns(frames)
    }

    fn channels(&self) -> usize {
        self.format.channels as usize
    }

    /// Pushes `frames` of silence starting at `start` (in 10 ms blocks), or advances the
    /// timeline without data when the gap is too long to be worth filling.
    fn emit_silence(&mut self, start: HostTime, frames: u64, flags: BlockFlags) {
        if frames == 0 {
            return;
        }
        let max_fill = self.format.sample_rate as u64 * self.max_gap_fill_ms as u64 / 1000;
        if frames > max_fill {
            self.producer.skip(frames);
            self.stats
                .skipped_frames
                .fetch_add(frames, Ordering::Relaxed);
            return;
        }
        let ch = self.channels();
        let block = (self.format.sample_rate as u64 / 100).max(1);
        let mut done = 0u64;
        while done < frames {
            let n = block.min(frames - done) as usize;
            let t = start.saturating_add_ns(self.ns_of(done));
            self.producer.push(&self.zeros[..n * ch], t, flags);
            done += n as u64;
        }
    }

    /// Keeps the timeline running with silence while the device delivers nothing.
    pub fn fill_idle(&mut self) {
        let start = self.aligner.expected_next();
        let frames = self.aligner.on_idle(HostTime::now());
        if frames > 0 {
            self.stats.idle_frames.fetch_add(frames, Ordering::Relaxed);
            let start = start.unwrap_or_else(HostTime::now);
            self.emit_silence(start, frames, BlockFlags::GAP_FILL | BlockFlags::SILENT);
        }
    }

    /// Places one device packet on the timeline. Returns the silence used so far (ms) when
    /// this packet's gap exhausted the session's fill budget: the backend reports
    /// [`crate::CaptureEvent::GapFillBudgetExhausted`] and reopens the stream.
    pub fn packet(&mut self, p: Packet<'_>) -> Option<u64> {
        if p.frames == 0 {
            return None;
        }
        let stats = &self.stats;
        stats.packets.fetch_add(1, Ordering::Relaxed);
        stats
            .frames_captured
            .fetch_add(p.frames as u64, Ordering::Relaxed);
        if p.flags.contains(BlockFlags::TIMESTAMP_ESTIMATED) {
            stats.timestamp_errors.fetch_add(1, Ordering::Relaxed);
        }
        let glitch = p.flags.contains(BlockFlags::DISCONTINUITY);
        if glitch {
            stats.device_glitches.fetch_add(1, Ordering::Relaxed);
        }
        let al = self
            .aligner
            .on_device_packet(p.t, p.frames, p.position, glitch);
        let mut exhausted = None;
        match self.gaps.decide(al.gap_frames) {
            _ if al.gap_frames == 0 => {}
            GapAction::Fill => {
                self.stats
                    .gap_frames
                    .fetch_add(al.gap_frames, Ordering::Relaxed);
                let gap_start =
                    HostTime::from_ns(p.t.as_ns().saturating_sub(self.ns_of(al.gap_frames)));
                self.emit_silence(
                    gap_start,
                    al.gap_frames,
                    BlockFlags::GAP_FILL | BlockFlags::SILENT,
                );
            }
            action => {
                self.producer.skip(al.gap_frames);
                self.stats
                    .skipped_frames
                    .fetch_add(al.gap_frames, Ordering::Relaxed);
                if action == GapAction::SkipBudgetExhausted {
                    exhausted =
                        Some(self.gaps.used_frames() * 1000 / self.format.sample_rate as u64);
                }
            }
        }
        if al.trim_frames > 0 {
            self.stats
                .trimmed_frames
                .fetch_add(al.trim_frames as u64, Ordering::Relaxed);
        }
        let keep = (p.frames - al.trim_frames) as usize;
        if keep > 0 {
            let ch = self.channels();
            let t_keep = p.t.saturating_add_ns(self.ns_of(al.trim_frames as u64));
            match p.data {
                Some(all) if !p.flags.contains(BlockFlags::SILENT) => {
                    self.producer
                        .push(&all[al.trim_frames as usize * ch..], t_keep, p.flags);
                }
                _ => {
                    let flags = p.flags | BlockFlags::SILENT;
                    let mut done = 0usize;
                    while done < keep {
                        let n = (keep - done).min(self.zeros.len() / ch);
                        let tt = t_keep.saturating_add_ns(self.ns_of(done as u64));
                        self.producer.push(&self.zeros[..n * ch], tt, flags);
                        done += n;
                    }
                }
            }
        }
        exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_core::Source;

    fn feed(source: Source) -> (Feed, sd_core::ring::RingConsumer, Arc<CaptureStats>) {
        let cfg = CaptureConfig::new(source);
        let frames = cfg.format.sample_rate as usize * 2;
        let (p, c) = sd_core::ring::audio_ring(cfg.format.channels, frames, 2_000).unwrap();
        let stats = Arc::new(CaptureStats::default());
        (
            Feed::new(
                &cfg,
                p,
                stats.clone(),
                Idle {
                    after_ms: 60,
                    margin_ms: 100,
                },
            ),
            c,
            stats,
        )
    }

    /// Packets of 10 ms at 48 kHz, stamped `t0 + i * 10 ms` unless `skip` drops one.
    fn packets(f: &mut Feed, t0: u64, n: u32, skip: Option<u32>) {
        let data = vec![0.5f32; 480];
        let mut pos = 0.0;
        for i in 0..n {
            if Some(i) == skip {
                pos += 480.0;
                continue;
            }
            f.packet(Packet {
                t: HostTime::from_ns(t0 + i as u64 * 10_000_000),
                frames: 480,
                position: Some(pos),
                flags: BlockFlags::empty(),
                data: Some(&data),
            });
            pos += 480.0;
        }
    }

    #[test]
    fn continuous_packets_reach_the_ring_unchanged() {
        let (mut f, _c, stats) = feed(Source::Mic);
        f.start_at(HostTime::from_ns(1_000_000_000));
        packets(&mut f, 1_000_000_000, 100, None);
        let s = stats.snapshot();
        assert_eq!(s.packets, 100);
        assert_eq!(s.frames_captured, 48_000);
        assert_eq!(
            s.gap_frames + s.skipped_frames + s.trimmed_frames,
            0,
            "{s:?}"
        );
    }

    #[test]
    fn a_lost_packet_becomes_filled_silence() {
        let (mut f, _c, stats) = feed(Source::Mic);
        f.start_at(HostTime::from_ns(1_000_000_000));
        packets(&mut f, 1_000_000_000, 50, Some(20));
        let s = stats.snapshot();
        assert_eq!(s.gap_frames, 480, "{s:?}");
        assert_eq!(s.packets, 49);
    }

    #[test]
    fn silent_packets_without_data_still_advance_the_timeline() {
        let (mut f, _c, stats) = feed(Source::System);
        f.start_at(HostTime::from_ns(0));
        for i in 0..10u64 {
            f.packet(Packet {
                t: HostTime::from_ns(i * 10_000_000),
                frames: 480,
                position: None,
                flags: BlockFlags::SILENT,
                data: None,
            });
        }
        let s = stats.snapshot();
        assert_eq!(s.frames_captured, 4_800);
        assert_eq!(s.gap_frames + s.trimmed_frames, 0, "{s:?}");
    }
}
