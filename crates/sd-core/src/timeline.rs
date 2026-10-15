//! Keeps a capture stream on a continuous timeline using device timestamps.
//!
//! Device APIs deliver packets with a host-clock timestamp of their first frame (WASAPI
//! `qpcPosition`, Core Audio `mHostTime`, PipeWire `nsec`). Four things break naive frame
//! counting:
//!
//! * **Loopback silence.** WASAPI loopback delivers *no packets at all* while nothing is
//!   playing. The timeline must keep advancing, so downstream consumers (echo cancellation,
//!   the mic/system aligner) still see audio-rate data.
//! * **Glitches.** A late callback or a device hiccup loses frames; the next packet's
//!   timestamp jumps ahead.
//! * **Overlap.** Silence synthesized during an idle period may overrun the moment packets
//!   resume; the overlapping head of the packet must be trimmed so time never runs backwards.
//! * **Structured timestamp jitter.** Some drivers stamp packets in a sawtooth. Measured with
//!   OnePlus Nord Buds 3r in Bluetooth hands-free mode: packets carrying 10 ms of audio are
//!   stamped ~9 ms apart, then one ~18 ms interval resynchronizes. No audio is lost; only the
//!   stamps wobble.
//!
//! [`TimelineAligner`] counts frames (the timeline is the audio itself) and compares each
//! packet's timestamp with the expected time of its first frame. That difference is tracked as
//! a *baseline*: a smoothed level, followed at a capped slew rate, so it absorbs slow clock
//! drift and averages out jitter. Only a sudden jump away from the baseline larger than the
//! tolerance counts as a gap (insert silence) or an overlap (trim). Errors from undetected
//! sub-tolerance gaps stay bounded by the tolerance instead of accumulating, and structured
//! jitter never inflates or shrinks the timeline.

use crate::clock::HostTime;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignerConfig {
    pub sample_rate: u32,
    /// Jumps away from the timestamp baseline smaller than this are treated as jitter. Keep it
    /// below one packet duration so a single lost packet is detected.
    pub tolerance_ns: u64,
    /// With device positions, a device-flagged packet whose timestamp jumps past the baseline by
    /// more than this is treated as real loss even if positions look continuous.
    pub glitch_tolerance_ns: u64,
    /// Time constant of the baseline's level estimate (averages jitter).
    pub level_tau_ns: u64,
    /// Maximum rate at which the baseline follows the level, in ppm. Must exceed real clock
    /// drift (devices are within a few hundred ppm) but stay far below gap-sized jumps.
    pub max_slew_ppm: f64,
    /// With no packets for this long, [`TimelineAligner::on_idle`] starts producing silence.
    pub idle_after_ns: u64,
    /// Synthesized silence stays this far behind "now" so resuming packets rarely overlap it.
    pub idle_margin_ns: u64,
}

impl AlignerConfig {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            tolerance_ns: 8_000_000,
            glitch_tolerance_ns: 15_000_000,
            level_tau_ns: 500_000_000,
            max_slew_ppm: 1_000.0,
            idle_after_ns: 60_000_000,
            idle_margin_ns: 100_000_000,
        }
    }
}

/// What to do with an incoming packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Alignment {
    /// Frames of silence to emit before the packet (a gap in the device stream).
    pub gap_frames: u64,
    /// Frames to drop from the front of the packet (they overlap audio already emitted).
    pub trim_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AlignerStats {
    pub packets: u64,
    pub gaps: u64,
    pub gap_frames: u64,
    pub overlaps: u64,
    pub trimmed_frames: u64,
    pub idle_frames: u64,
    /// Device-flagged glitches where the timestamp gap overrode continuous positions.
    pub timestamp_overrides: u64,
}

#[derive(Debug, Clone)]
pub struct TimelineAligner {
    cfg: AlignerConfig,
    /// Host time of the next frame on the frame-counted timeline, once started.
    expected_next: Option<u64>,
    /// Host time the last real packet arrived (for idle detection).
    last_packet_host: Option<u64>,
    /// Smoothed (timestamp - expected) offset, in ns.
    level: f64,
    /// Slew-limited follower of `level`; jumps are measured against it.
    baseline: f64,
    /// Device position (in stream frames) expected for the next packet of the current stream.
    position_next: Option<f64>,
    /// Device positions of the current stream proved unusable; use timestamps only.
    positions_unreliable: bool,
    /// Silence was synthesized since the last packet, so device positions no longer line up
    /// with the timeline and the next packet must be placed by its timestamp.
    idle_since_packet: bool,
    stats: AlignerStats,
}

impl TimelineAligner {
    pub fn new(cfg: AlignerConfig) -> Self {
        Self {
            cfg,
            expected_next: None,
            last_packet_host: None,
            level: 0.0,
            baseline: 0.0,
            position_next: None,
            positions_unreliable: false,
            idle_since_packet: false,
            stats: AlignerStats::default(),
        }
    }

    /// Marks the start of a new device stream (after a switch or restart): device positions
    /// restart, so the next packet is placed by its timestamp. The timeline itself continues.
    pub fn new_stream(&mut self) {
        self.position_next = None;
        self.positions_unreliable = false;
    }

    pub fn stats(&self) -> AlignerStats {
        self.stats
    }

    /// Host time at which the next frame is expected, once the timeline has started.
    pub fn expected_next(&self) -> Option<HostTime> {
        self.expected_next.map(HostTime)
    }

    /// Current timestamp baseline (device stamps minus frame-counted time), in ns.
    pub fn baseline_ns(&self) -> f64 {
        self.baseline
    }

    /// Forgets the current position (after a device change the new stream starts fresh).
    pub fn reset(&mut self) {
        self.expected_next = None;
        self.last_packet_host = None;
        self.level = 0.0;
        self.baseline = 0.0;
        self.position_next = None;
        self.positions_unreliable = false;
        self.idle_since_packet = false;
    }

    /// Starts the timeline at `t` without a packet (e.g. stream start for a loopback that may
    /// stay silent). Has no effect once the timeline is running.
    pub fn start_at(&mut self, t: HostTime) {
        if self.expected_next.is_none() {
            self.expected_next = Some(t.as_ns());
            self.last_packet_host = Some(t.as_ns());
        }
    }

    fn frames_to_ns(&self, frames: u64) -> u64 {
        (frames as u128 * 1_000_000_000 / self.cfg.sample_rate as u128) as u64
    }

    fn ns_to_frames(&self, ns: u64) -> u64 {
        ((ns as u128 * self.cfg.sample_rate as u128 + 500_000_000) / 1_000_000_000) as u64
    }

    /// A packet of `frames` frames whose first frame was captured at `t`, placed by timestamp
    /// only. Prefer [`Self::on_packet_at`] when the device reports positions.
    pub fn on_packet(&mut self, t: HostTime, frames: u32) -> Alignment {
        self.on_packet_at(t, frames, None)
    }

    /// A packet of `frames` frames whose first frame was captured at `t` and sits at device
    /// position `position` (in *stream* frames; scale device-rate positions before calling).
    ///
    /// Within a stream, lost frames are measured exactly from position discontinuities, and
    /// timestamps are not used for that decision: some drivers' stamps wobble by ±10 ms
    /// (Bluetooth hands-free), which would otherwise be mistaken for gaps and overlaps. The
    /// timestamp path (jump against the baseline) is used for the first packet of a stream, the
    /// first packet after synthesized silence, and when positions are absent or implausible.
    pub fn on_packet_at(&mut self, t: HostTime, frames: u32, position: Option<f64>) -> Alignment {
        self.on_device_packet(t, frames, position, false)
    }

    /// Like [`Self::on_packet_at`], plus the device's own discontinuity flag (WASAPI
    /// `AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY`).
    ///
    /// Device positions count *delivered* frames, so they miss audio the OS dropped while
    /// reconfiguring (measured: a Realtek mic lost 25–38 ms each time a Bluetooth output was
    /// removed or added, with continuous positions but a flagged packet and a matching timestamp
    /// jump). When the flag is set and the timestamp jumps past the baseline by more than
    /// [`AlignerConfig::glitch_tolerance_ns`], the timestamp-measured gap wins. That threshold
    /// sits above the jitter of drivers that set the flag on every timestamp resync
    /// (Bluetooth hands-free: about ±8 ms, no loss).
    pub fn on_device_packet(
        &mut self,
        t: HostTime,
        frames: u32,
        position: Option<f64>,
        device_glitch: bool,
    ) -> Alignment {
        let t = t.as_ns();
        self.stats.packets += 1;
        self.last_packet_host = Some(t);
        let mut out = Alignment::default();
        let position_next = self.position_next.take();
        self.position_next = position.map(|p| p + frames as f64);
        let after_idle = std::mem::replace(&mut self.idle_since_packet, false);
        let Some(mut exp) = self.expected_next else {
            self.expected_next = Some(t + self.frames_to_ns(frames as u64));
            return out;
        };
        let mut dev = t as f64 - exp as f64;

        // Positive = frames missing before this packet; negative = overlap with emitted audio.
        let mut missing_frames: Option<f64> = None;
        if let (Some(p), Some(pn)) = (position, position_next)
            && !after_idle
            && !self.positions_unreliable
        {
            let delta = p - pn;
            let one_second = self.cfg.sample_rate as f64;
            if delta > 60.0 * one_second || delta < -4.0 * frames as f64 {
                // Not a plausible glitch: this driver's positions can't be trusted.
                self.positions_unreliable = true;
            } else {
                // Sub-frame differences are resampler rounding, not loss.
                let by_position = if delta.abs() < 1.5 { 0.0 } else { delta };
                let jump_ns = dev - self.baseline;
                let by_timestamp = jump_ns * self.cfg.sample_rate as f64 / 1e9;
                missing_frames = Some(
                    if device_glitch
                        && jump_ns > self.cfg.glitch_tolerance_ns as f64
                        && by_timestamp > by_position
                    {
                        self.stats.timestamp_overrides += 1;
                        by_timestamp
                    } else {
                        by_position
                    },
                );
            }
        }
        let missing = missing_frames.unwrap_or_else(|| {
            let jump = dev - self.baseline;
            let tol = self.cfg.tolerance_ns as f64;
            if jump.abs() > tol {
                jump * self.cfg.sample_rate as f64 / 1e9
            } else {
                0.0
            }
        });

        if missing > 0.0 {
            out.gap_frames = missing.round() as u64;
            self.stats.gaps += 1;
            self.stats.gap_frames += out.gap_frames;
            let gap_ns = self.frames_to_ns(out.gap_frames);
            exp += gap_ns;
            dev -= gap_ns as f64;
        } else if missing < 0.0 {
            let overlap = ((-missing).round() as u64).min(frames as u64) as u32;
            out.trim_frames = overlap;
            self.stats.overlaps += 1;
            self.stats.trimmed_frames += overlap as u64;
            if overlap == frames {
                // The whole packet lies in the past: nothing to emit, timeline unchanged.
                return out;
            }
            dev += self.frames_to_ns(overlap as u64) as f64;
        }
        let kept = (frames - out.trim_frames) as u64;
        let dur = self.frames_to_ns(kept);
        // Track the offset: a smoothed level, followed by a slew-limited baseline.
        let alpha = (dur as f64 / self.cfg.level_tau_ns as f64).min(1.0);
        self.level += alpha * (dev - self.level);
        let max_step = self.cfg.max_slew_ppm * 1e-6 * dur as f64;
        self.baseline += (self.level - self.baseline).clamp(-max_step, max_step);
        self.expected_next = Some(exp + dur);
        out
    }

    /// Called periodically when no packet has arrived. Returns frames of silence to emit now.
    pub fn on_idle(&mut self, now: HostTime) -> u64 {
        let (Some(exp), Some(last)) = (self.expected_next, self.last_packet_host) else {
            return 0;
        };
        // Timeline position that corresponds to host time `now - margin`.
        let target = now.as_ns() as f64 - self.cfg.idle_margin_ns as f64 - self.baseline;
        if now.as_ns() < last + self.cfg.idle_after_ns || target <= exp as f64 {
            return 0;
        }
        let frames = self.ns_to_frames((target - exp as f64) as u64);
        if frames == 0 {
            return 0;
        }
        self.expected_next = Some(exp + self.frames_to_ns(frames));
        self.stats.idle_frames += frames;
        self.idle_since_packet = true;
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;
    const T0: u64 = 1_000_000 * MS;

    fn aligner() -> TimelineAligner {
        TimelineAligner::new(AlignerConfig::new(48_000))
    }

    /// Feeds `n` contiguous 10 ms packets from `start`, with optional per-packet jitter.
    fn feed(a: &mut TimelineAligner, start: u64, n: u64, jitter: impl Fn(u64) -> i64) -> u64 {
        let mut emitted = 0;
        for i in 0..n {
            let t = (start + i * 10 * MS) as i64 + jitter(i);
            let al = a.on_packet(HostTime(t as u64), 480);
            emitted += al.gap_frames + (480 - al.trim_frames) as u64;
        }
        emitted
    }

    #[test]
    fn continuous_stream_has_no_gaps_even_with_jitter() {
        let mut a = aligner();
        let emitted = feed(&mut a, T0, 1_000, |i| {
            ((i * 7919) % 7) as i64 * 500_000 - 1_500_000
        });
        assert_eq!(emitted, 480_000);
        assert_eq!((a.stats().gaps, a.stats().overlaps), (0, 0));
    }

    /// Replays the timestamp pattern measured with Bluetooth hands-free earbuds: eight 10 ms
    /// packets stamped ~9 ms apart, then an ~18 ms resync. No audio is lost, so nothing may be
    /// inserted or trimmed, and the frame-counted timeline must track host time.
    #[test]
    fn bluetooth_sawtooth_timestamps_are_not_gaps() {
        let mut a = aligner();
        let mut t = T0;
        let mut emitted = 0u64;
        let cycles = 2_000u64; // 180 s
        for _ in 0..cycles {
            for k in 0..9 {
                let al = a.on_packet(HostTime(t), 480);
                emitted += al.gap_frames + (480 - al.trim_frames) as u64;
                t += if k == 8 { 18 * MS } else { 9 * MS };
            }
        }
        assert_eq!(
            (a.stats().gaps, a.stats().overlaps),
            (0, 0),
            "{:?}",
            a.stats()
        );
        assert_eq!(emitted, cycles * 9 * 480);
        let span_host = (t - T0) as f64;
        let span_timeline = (a.expected_next().unwrap().as_ns() - T0) as f64;
        assert!(
            (span_host - span_timeline).abs() < 8.0 * MS as f64,
            "{span_host} vs {span_timeline}"
        );
    }

    /// One real lost packet (10 ms) every second is detected every time, never absorbed.
    #[test]
    fn repeated_single_packet_losses_are_all_detected() {
        let mut a = aligner();
        let mut t = T0;
        for i in 0..6_000u64 {
            if i % 100 == 99 {
                t += 10 * MS; // this packet is lost
            }
            a.on_packet(HostTime(t), 480);
            t += 10 * MS;
        }
        assert_eq!(a.stats().gaps, 60); // i = 99, 199, ..., 5999
        assert_eq!(a.stats().gap_frames, 60 * 480);
        assert_eq!(a.stats().overlaps, 0);
    }

    /// Replays the second, messier Bluetooth pattern (±6 ms swings and bursts of packets 2 ms
    /// apart). With continuous device positions nothing may be inserted or trimmed, however
    /// badly the timestamps behave.
    #[test]
    fn positions_override_chaotic_timestamps() {
        let mut a = aligner();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut emitted = 0u64;
        for i in 0..20_000u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let jitter = (rng % 24_000_000) as i64 - 12_000_000; // ±12 ms
            let t = (T0 + i * 10 * MS) as i64 + jitter;
            let al = a.on_packet_at(HostTime(t as u64), 480, Some(i as f64 * 480.0));
            emitted += al.gap_frames + (480 - al.trim_frames) as u64;
        }
        assert_eq!(
            (a.stats().gaps, a.stats().overlaps),
            (0, 0),
            "{:?}",
            a.stats()
        );
        assert_eq!(emitted, 20_000 * 480);
    }

    /// A real loss shows up as a position jump and yields an exact gap, regardless of stamps.
    #[test]
    fn position_jump_gives_exact_gap() {
        let mut a = aligner();
        for i in 0..10u64 {
            a.on_packet_at(HostTime(T0 + i * 10 * MS), 480, Some(i as f64 * 480.0));
        }
        // 2 packets (960 frames) lost; timestamp deliberately off by 3 ms.
        let al = a.on_packet_at(HostTime(T0 + 123 * MS), 480, Some(12.0 * 480.0));
        assert_eq!(
            al,
            Alignment {
                gap_frames: 960,
                trim_frames: 0
            }
        );
        // Device-rate rounding (fractional positions after scaling) is not loss.
        let al = a.on_packet_at(HostTime(T0 + 130 * MS), 480, Some(13.0 * 480.0 + 0.4));
        assert_eq!(al, Alignment::default());
    }

    /// After synthesized silence the next packet is placed by timestamp, not by position
    /// (the device position kept counting during silence, the timeline did too).
    #[test]
    fn after_idle_timestamps_place_the_packet() {
        let mut a = aligner();
        for i in 0..10u64 {
            a.on_packet_at(HostTime(T0 + i * 10 * MS), 480, Some(i as f64 * 480.0));
        }
        a.on_idle(HostTime(T0 + 1_100 * MS)); // silence up to T0 + 1000 ms
        // Positions say 100 packets later; timestamps say 1050 ms. Timestamp wins: 50 ms gap.
        let al = a.on_packet_at(HostTime(T0 + 1_050 * MS), 480, Some(105.0 * 480.0));
        assert_eq!(al.gap_frames, 2_400);
    }

    #[test]
    fn implausible_positions_fall_back_to_timestamps() {
        let mut a = aligner();
        for i in 0..10u64 {
            a.on_packet_at(HostTime(T0 + i * 10 * MS), 480, Some(i as f64 * 480.0));
        }
        // Position goes far backwards (driver bug): ignore positions, use the timestamp.
        let al = a.on_packet_at(HostTime(T0 + 100 * MS), 480, Some(5.0));
        assert_eq!(al, Alignment::default());
        let al = a.on_packet_at(HostTime(T0 + 130 * MS), 480, Some(5.0 + 480.0));
        assert_eq!(al.gap_frames, 960, "timestamp path detects the 20 ms gap");
    }

    /// Measured on a Realtek mic while a Bluetooth output was removed: positions continuous,
    /// packet flagged, timestamp +25 ms beyond the packet. That is real loss: insert it.
    #[test]
    fn flagged_glitch_with_large_timestamp_jump_beats_positions() {
        let mut a = aligner();
        for i in 0..200u64 {
            a.on_device_packet(
                HostTime(T0 + i * 10 * MS),
                480,
                Some(i as f64 * 480.0),
                false,
            );
        }
        let al = a.on_device_packet(HostTime(T0 + 2_025 * MS), 480, Some(200.0 * 480.0), true);
        assert_eq!(
            al.gap_frames, 1_200,
            "25 ms lost while positions stayed continuous"
        );
        assert_eq!(a.stats().timestamp_overrides, 1);
        // Back to normal afterwards.
        let al = a.on_device_packet(HostTime(T0 + 2_035 * MS), 480, Some(201.0 * 480.0), false);
        assert_eq!(al, Alignment::default());
    }

    /// The Bluetooth hands-free driver flags every resync (~90 ms) with jitter of ±8 ms and no
    /// loss. Flags plus continuous positions must not create gaps.
    #[test]
    fn flagged_bluetooth_resyncs_are_not_gaps() {
        let mut a = aligner();
        let mut t = T0;
        let mut n = 0u64;
        for _ in 0..2_000 {
            for k in 0..9 {
                let al = a.on_device_packet(HostTime(t), 480, Some(n as f64 * 160.0 * 3.0), k == 8);
                assert_eq!(al, Alignment::default(), "packet {n}");
                n += 1;
                t += if k == 8 { 18 * MS } else { 9 * MS };
            }
        }
        assert_eq!(a.stats().timestamp_overrides, 0);
    }

    #[test]
    fn slow_drift_never_accumulates_into_gaps() {
        // Device runs 300 ppm fast relative to nominal: timestamps advance 0.03 % slower.
        let mut a = aligner();
        let mut emitted = 0;
        for i in 0..60_000u64 {
            let t = T0 + (i as f64 * 10.0 * MS as f64 / 1.0003) as u64;
            let al = a.on_packet(HostTime(t), 480);
            emitted += al.gap_frames + (480 - al.trim_frames) as u64;
        }
        // Ten minutes: the 300 ppm (180 ms) drift is followed by the slew-limited baseline,
        // not turned into gaps or trims.
        assert_eq!(a.stats().gaps + a.stats().overlaps, 0, "{:?}", a.stats());
        assert_eq!(emitted, 60_000 * 480);
        assert!(
            (a.baseline_ns() + 180.0 * MS as f64).abs() < 2.0 * MS as f64,
            "{}",
            a.baseline_ns()
        );
    }

    #[test]
    fn glitch_inserts_gap() {
        let mut a = aligner();
        feed(&mut a, T0, 10, |_| 0);
        // 3 packets (30 ms) lost.
        let al = a.on_packet(HostTime(T0 + 130 * MS), 480);
        assert_eq!(
            al,
            Alignment {
                gap_frames: 1_440,
                trim_frames: 0
            }
        );
        let al = a.on_packet(HostTime(T0 + 140 * MS), 480);
        assert_eq!(al, Alignment::default());
    }

    #[test]
    fn loopback_silence_is_synthesized_then_resumes_cleanly() {
        let mut a = aligner();
        feed(&mut a, T0, 10, |_| 0); // audio until T0 + 100 ms
        // Too soon to call it idle.
        assert_eq!(a.on_idle(HostTime(T0 + 140 * MS)), 0);
        // 1 s later with no packets: silence up to now - 100 ms margin = T0 + 1000 ms.
        let idle = a.on_idle(HostTime(T0 + 1_100 * MS));
        assert_eq!(idle, 48_000 * 900 / 1_000);
        // Packets resume at T0 + 1050 ms: 50 ms gap after the synthesized silence.
        let al = a.on_packet(HostTime(T0 + 1_050 * MS), 480);
        assert_eq!(al.gap_frames, 2_400);
        assert_eq!(al.trim_frames, 0);
    }

    #[test]
    fn overlap_after_idle_is_trimmed() {
        let mut a = aligner();
        a.start_at(HostTime(T0));
        let idle = a.on_idle(HostTime(T0 + 500 * MS)); // silence to T0 + 400 ms
        assert_eq!(idle, 19_200);
        // A packet stamped T0 + 390 ms overlaps the silence by 10 ms = 480 frames: all of it.
        let al = a.on_packet(HostTime(T0 + 390 * MS), 480);
        assert_eq!(
            al,
            Alignment {
                gap_frames: 0,
                trim_frames: 480
            }
        );
        // The next packet continues exactly where the timeline is.
        let al = a.on_packet(HostTime(T0 + 400 * MS), 480);
        assert_eq!(al, Alignment::default());
        let mut b = aligner();
        b.start_at(HostTime(T0));
        b.on_idle(HostTime(T0 + 500 * MS));
        let al = b.on_packet(HostTime(T0 + 380 * MS), 1_440); // 30 ms packet, 20 ms overlap
        assert_eq!(al.trim_frames, 960);
    }

    #[test]
    fn silent_loopback_from_start() {
        let mut a = aligner();
        a.start_at(HostTime(T0));
        let mut total = 0;
        for step in 1..=100u64 {
            total += a.on_idle(HostTime(T0 + step * 20 * MS));
        }
        // 2 s elapsed, timeline 100 ms behind: 1.9 s of silence.
        assert_eq!(total, 91_200);
    }

    #[test]
    fn reset_starts_fresh() {
        let mut a = aligner();
        feed(&mut a, T0, 5, |_| 0);
        a.reset();
        let al = a.on_packet(HostTime(T0 + 10_000 * MS), 480);
        assert_eq!(al, Alignment::default());
    }
}
