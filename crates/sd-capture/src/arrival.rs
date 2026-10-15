//! Smooth capture timestamps from arrival times.
//!
//! Some APIs report no capture time per packet (PulseAudio for monitor sources), so a backend
//! stamps packets by when they arrive. Arrival is late by a scheduling delay that varies by a
//! few ms and is never negative, but a server may also hand over a burst of audio at once
//! (PulseAudio's null sink renders ahead of real time at stream start), which looks *early*.
//!
//! [`ArrivalClock`] therefore estimates the offset `arrival - position` as a low percentile
//! ([`PERCENTILE`]) of the last [`WINDOW_NS`] of packets: scheduling jitter sits above it and
//! isolated early bursts below it.
//! * During the first window of a stream the offset is that estimate.
//! * After it, the offset follows the estimate at most at [`MAX_SLEW_PPM`] (enough for real
//!   clock drift), so timestamps stay smooth.
//! * [`RESET_PACKETS`] packets in a row more than [`RESET_NS`] away from the offset are a
//!   discontinuity (the stream was suspended, stalled or restarted): the estimate restarts
//!   from them.
//!
//! Timestamps are `position + offset`: smooth and on the device's clock.

use std::collections::VecDeque;

/// Window of recent packets the offset is estimated from.
pub const WINDOW_NS: f64 = 2_000_000_000.0;
/// The offset is this percentile of the window (low: delay only ever adds).
pub const PERCENTILE: f64 = 0.1;
/// How fast the offset may move without a reset (clock drift between device and host).
pub const MAX_SLEW_PPM: f64 = 500.0;
/// Packets this far from the offset ...
pub const RESET_NS: f64 = 50_000_000.0;
/// ... this many times in a row restart the estimate.
pub const RESET_PACKETS: usize = 3;

#[derive(Debug, Clone, Default)]
pub(crate) struct ArrivalClock {
    /// host time - position time, in ns.
    offset: Option<f64>,
    last_position_ns: f64,
    /// (position, arrival - position) of recent packets.
    window: VecDeque<(f64, f64)>,
    scratch: Vec<f64>,
    /// Position where the current estimate started (warm-up lasts one window).
    start_ns: f64,
    /// Consecutive packets far from the offset.
    far: usize,
}

impl ArrivalClock {
    /// The timestamp of a packet whose first frame is at `position_ns` (stream time) and was
    /// estimated, from its arrival, to have been captured at `arrival_ns` (host time).
    pub fn stamp(&mut self, position_ns: f64, arrival_ns: f64) -> u64 {
        let x = arrival_ns - position_ns;
        if self.offset.is_none() {
            self.start_ns = position_ns;
        }
        match self.offset {
            Some(o) if (x - o).abs() > RESET_NS => self.far += 1,
            _ => self.far = 0,
        }
        if self.far >= RESET_PACKETS {
            // A discontinuity: estimate again from the packets after it.
            let keep = self.window.len().saturating_sub(RESET_PACKETS - 1);
            self.window.drain(..keep);
            self.start_ns = self.window.front().map_or(position_ns, |&(p, _)| p);
            self.offset = None;
            self.far = 0;
        }
        self.window.push_back((position_ns, x));
        while self
            .window
            .front()
            .is_some_and(|&(p, _)| position_ns - p > WINDOW_NS)
        {
            self.window.pop_front();
        }
        self.scratch.clear();
        self.scratch.extend(self.window.iter().map(|&(_, x)| x));
        self.scratch.sort_by(f64::total_cmp);
        let target = self.scratch[((self.scratch.len() - 1) as f64 * PERCENTILE) as usize];
        let warming_up = position_ns - self.start_ns < WINDOW_NS;
        let o = match self.offset {
            None => target,
            Some(_) if warming_up => target,
            Some(o) if self.far > 0 => o, // possibly a discontinuity: hold
            Some(o) => {
                let elapsed = (position_ns - self.last_position_ns).max(0.0);
                let step = elapsed * MAX_SLEW_PPM * 1e-6;
                o + (target - o).clamp(-step, step)
            }
        };
        self.offset = Some(o);
        self.last_position_ns = position_ns;
        (position_ns + o).max(0.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAG: f64 = 20_000_000.0; // 20 ms fragments

    /// Pseudo-random delay of 0..6 ms.
    fn delay(i: usize) -> f64 {
        ((i * 7_919) % 13) as f64 * 0.5e6
    }

    #[test]
    fn jitter_is_removed() {
        let mut c = ArrivalClock::default();
        let true_offset = 1e12;
        let mut worst: f64 = 0.0;
        for i in 0..1_000 {
            let pos = i as f64 * FRAG;
            let t = c.stamp(pos, pos + true_offset + delay(i));
            if i > 150 {
                worst = worst.max((t as f64 - pos - true_offset).abs());
            }
        }
        assert!(worst < 1.5e6, "worst error {worst} ns");
    }

    #[test]
    fn follows_drift_within_the_slew_limit() {
        let mut c = ArrivalClock::default();
        // Device clock 200 ppm slower than the host: the offset grows by 4 us per 20 ms.
        let mut err = 0.0;
        for i in 0..5_000 {
            let pos = i as f64 * FRAG;
            let truth = pos * (1.0 + 200e-6) + 1e9;
            let t = c.stamp(pos, truth + delay(i));
            err = t as f64 - truth;
        }
        assert!(err.abs() < 2e6, "error after 100 s: {err} ns");
    }

    #[test]
    fn an_early_burst_does_not_bias_the_clock() {
        let mut c = ArrivalClock::default();
        // The first 100 ms arrive 15 ms "early" (rendered ahead of real time), then steady.
        let mut stamps = Vec::new();
        for i in 0..1_500 {
            let pos = i as f64 * FRAG;
            let early = if i < 5 { 15e6 } else { 0.0 };
            stamps.push((pos, c.stamp(pos, pos + 1e9 + delay(i) - early) as f64));
        }
        // Slope of timestamps vs position over the last 20 s: no apparent drift.
        let (p0, t0) = stamps[500];
        let (p1, t1) = stamps[1_499];
        let ppm = ((t1 - t0) / (p1 - p0) - 1.0) * 1e6;
        assert!(ppm.abs() < 20.0, "apparent drift {ppm} ppm");
        let err = t1 - p1 - 1e9;
        assert!(err.abs() < 3e6, "offset error {err} ns");
    }

    #[test]
    fn a_large_jump_resets_at_once() {
        let mut c = ArrivalClock::default();
        for i in 0..200 {
            let pos = i as f64 * FRAG;
            c.stamp(pos, pos + 1e9);
        }
        // The stream was suspended for 2 s without the position advancing: after
        // RESET_PACKETS packets the clock follows.
        let mut last = 0;
        for i in 200..200 + RESET_PACKETS as u64 + 1 {
            let pos = i as f64 * FRAG;
            last = c.stamp(pos, pos + 3e9) as i64 - (pos + 3e9) as i64;
        }
        assert_eq!(last, 0);
    }
}
