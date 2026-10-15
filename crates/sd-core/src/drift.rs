//! Device clock drift estimation.
//!
//! A sound card's real sample rate differs from its nominal rate by tens of ppm, and two
//! devices (mic vs. render endpoint) drift independently. Over a two-hour meeting, 50 ppm is
//! 360 ms of skew. This estimator fits `frame_position = a + rate * host_seconds` by least
//! squares over a sliding window of (host time, frame position) observations taken from device
//! callbacks, which absorbs per-callback scheduling jitter.

use std::collections::VecDeque;

use crate::clock::HostTime;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftConfig {
    pub nominal_rate: f64,
    /// Observations older than this are discarded.
    pub window_ns: u64,
    /// Observations closer than this to the previous one are ignored (bounds memory and CPU).
    pub min_interval_ns: u64,
    /// No estimate is produced until the window spans at least this long.
    pub min_span_ns: u64,
}

impl DriftConfig {
    pub fn new(nominal_rate: u32) -> Self {
        Self {
            nominal_rate: nominal_rate as f64,
            window_ns: 120_000_000_000,
            min_interval_ns: 100_000_000,
            min_span_ns: 5_000_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftEstimate {
    /// Estimated true frame rate in frames per second of host time.
    pub rate_hz: f64,
    /// Deviation from nominal in parts per million (positive = device runs fast).
    pub ppm: f64,
    /// Standard error of `ppm`. Short or sparse data (e.g. loopback that only runs while audio
    /// plays) gives large errors; consumers should not correct drift from imprecise estimates.
    pub ppm_stderr: f64,
    /// RMS of residuals, expressed in nanoseconds of host time.
    pub jitter_rms_ns: f64,
    /// Number of observations used.
    pub points: usize,
    t0_ns: u64,
    mean_t: f64,
    mean_f: f64,
}

impl DriftEstimate {
    /// Host time at which `frame` was (or will be) captured, according to the fit.
    pub fn host_time_of_frame(&self, frame: u64) -> HostTime {
        let t = self.mean_t + (frame as f64 - self.mean_f) / self.rate_hz;
        HostTime((self.t0_ns as f64 + t * 1e9).max(0.0) as u64)
    }

    /// Frame position (fractional) at host time `t`, according to the fit.
    pub fn frame_at(&self, t: HostTime) -> f64 {
        let secs = (t.as_ns() as f64 - self.t0_ns as f64) / 1e9;
        self.mean_f + (secs - self.mean_t) * self.rate_hz
    }
}

#[derive(Debug, Clone)]
pub struct DriftEstimator {
    cfg: DriftConfig,
    /// (host ns, frame position, segment id). Segments are contiguous runs of real device data.
    points: VecDeque<(u64, u64, u32)>,
    segment: u32,
    pending_break: bool,
}

impl DriftEstimator {
    pub fn new(cfg: DriftConfig) -> Self {
        let cap = (cfg.window_ns / cfg.min_interval_ns.max(1)) as usize + 2;
        Self {
            cfg,
            points: VecDeque::with_capacity(cap),
            segment: 0,
            pending_break: false,
        }
    }

    /// Discards all history (call after a device restart or format change).
    pub fn reset(&mut self) {
        self.points.clear();
        self.pending_break = false;
    }

    /// Starts a new segment: the next observation is not assumed to continue the previous run.
    ///
    /// Call this at gaps and discontinuities. The relationship between frame position and host
    /// time can shift by tens of milliseconds across a gap (e.g. WASAPI loopback resuming after
    /// silence), which would bend a single regression line. Rates are fitted *within* segments
    /// (pooled regression), so such offsets do not bias the estimate.
    pub fn break_segment(&mut self) {
        self.pending_break = true;
    }

    /// Records that frame index `frame_pos` was captured at `host`.
    pub fn observe(&mut self, host: HostTime, frame_pos: u64) {
        let t = host.as_ns();
        if let Some(&(last_t, last_f, _)) = self.points.back() {
            if t < last_t || frame_pos < last_f {
                // Time or position went backwards: the device restarted. Start over.
                self.points.clear();
                self.pending_break = false;
            } else if !self.pending_break && t - last_t < self.cfg.min_interval_ns {
                return;
            }
        }
        if self.pending_break {
            self.segment = self.segment.wrapping_add(1);
            self.pending_break = false;
        }
        self.points.push_back((t, frame_pos, self.segment));
        while let Some(&(front_t, _, _)) = self.points.front() {
            if t - front_t > self.cfg.window_ns {
                self.points.pop_front();
            } else {
                break;
            }
        }
    }

    /// Per-segment (start index, end index exclusive) ranges with at least two points.
    fn segments(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0;
        for i in 1..=self.points.len() {
            if i == self.points.len() || self.points[i].2 != self.points[start].2 {
                if i - start >= 2 {
                    out.push((start, i));
                }
                start = i;
            }
        }
        out
    }

    pub fn estimate(&self) -> Option<DriftEstimate> {
        let &(t0, f0, _) = self.points.front()?;
        let segs = self.segments();
        let rel = |i: usize| {
            let (t, f, _) = self.points[i];
            ((t - t0) as f64 / 1e9, (f - f0) as f64)
        };
        // Per-segment means (both axes centred to keep f64 precision over long sessions).
        let mut means = Vec::with_capacity(segs.len());
        let (mut n, mut span) = (0usize, 0u64);
        for &(a, b) in &segs {
            let (mut st, mut sf) = (0.0, 0.0);
            for i in a..b {
                let (t, f) = rel(i);
                st += t;
                sf += f;
            }
            let k = (b - a) as f64;
            means.push((st / k, sf / k));
            n += b - a;
            span += self.points[b - 1].0 - self.points[a].0;
        }
        if n < 3 || span < self.cfg.min_span_ns {
            return None;
        }
        let (mut stt, mut stf) = (0.0, 0.0);
        for (&(a, b), &(mt, mf)) in segs.iter().zip(&means) {
            for i in a..b {
                let (t, f) = rel(i);
                stt += (t - mt) * (t - mt);
                stf += (t - mt) * (f - mf);
            }
        }
        if stt <= 0.0 {
            return None;
        }
        let rate = stf / stt;
        if !rate.is_finite() || rate <= 0.0 {
            return None;
        }
        let mut ss = 0.0;
        for (&(a, b), &(mt, mf)) in segs.iter().zip(&means) {
            for i in a..b {
                let (t, f) = rel(i);
                let resid_s = ((f - mf) - rate * (t - mt)) / rate;
                ss += resid_s * resid_s;
            }
        }
        // Standard error of the pooled slope: residual variance in frames over the degrees of
        // freedom left after fitting one slope plus one intercept per segment.
        let dof = n.saturating_sub(segs.len() + 1).max(1) as f64;
        let ppm_stderr = (ss * rate * rate / dof / stt).sqrt() / self.cfg.nominal_rate * 1e6;
        // The most recent segment anchors absolute frame <-> time conversions.
        let &(mean_t, mean_f) = means.last()?;
        Some(DriftEstimate {
            rate_hz: rate,
            ppm: (rate / self.cfg.nominal_rate - 1.0) * 1e6,
            ppm_stderr,
            jitter_rms_ns: (ss / n as f64).sqrt() * 1e9,
            points: n,
            t0_ns: t0,
            mean_t,
            mean_f: mean_f + f0 as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const BASE_NS: u64 = 5_000_000_000_000; // arbitrary boot-relative epoch

    fn simulate(ppm: f64, jitter_ms: f64, seconds: f64) -> DriftEstimate {
        let rate = 48_000.0 * (1.0 + ppm / 1e6);
        let mut est = DriftEstimator::new(DriftConfig::new(48_000));
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut frame = 0u64;
        while (frame as f64) / rate < seconds {
            let true_t = frame as f64 / rate;
            let jitter = (rng.next_f64() * 2.0 - 1.0) * jitter_ms / 1e3;
            let host = BASE_NS + ((true_t + jitter.max(-true_t)) * 1e9) as u64;
            est.observe(HostTime(host), frame);
            frame += 480;
        }
        est.estimate().expect("estimate")
    }

    #[test]
    fn recovers_drift_under_jitter() {
        for ppm in [-120.0, -35.0, 0.0, 50.0, 200.0] {
            let e = simulate(ppm, 2.0, 120.0);
            assert!(
                (e.ppm - ppm).abs() < 5.0,
                "true {ppm} ppm, estimated {:.2} ppm",
                e.ppm
            );
            assert!(e.jitter_rms_ns < 2.0e6, "jitter rms {}", e.jitter_rms_ns);
            // The reported uncertainty must cover the actual error and be small here.
            let err = (e.ppm - ppm).abs();
            assert!(
                err < 3.0 * e.ppm_stderr + 0.5,
                "err {err} vs se {}",
                e.ppm_stderr
            );
            assert!(e.ppm_stderr < 2.0, "stderr {}", e.ppm_stderr);
        }
    }

    #[test]
    fn no_estimate_before_min_span() {
        let mut est = DriftEstimator::new(DriftConfig::new(48_000));
        for i in 0..20u64 {
            est.observe(HostTime(i * 100_000_000), i * 4_800);
        }
        assert!(est.estimate().is_none()); // spans 1.9 s < 5 s
    }

    #[test]
    fn resets_when_device_restarts() {
        let mut est = DriftEstimator::new(DriftConfig::new(48_000));
        for i in 0..100u64 {
            est.observe(HostTime(i * 100_000_000), i * 4_800);
        }
        assert!(est.estimate().is_some());
        // Position restarts at zero after a device re-open.
        est.observe(HostTime(10_100_000_000), 0);
        assert!(est.estimate().is_none());
    }

    /// Bursty data (like loopback that only delivers while audio plays) whose timestamp offset
    /// jumps by up to ±40 ms between bursts: pooled within-segment fitting stays accurate.
    #[test]
    fn segment_offsets_do_not_bias_rate() {
        let ppm = 60.0;
        let rate = 48_000.0 * (1.0 + ppm / 1e6);
        let mut est = DriftEstimator::new(DriftConfig::new(48_000));
        let mut rng = Rng(0xD1B5_4A32_D192_ED03); // well-mixed seed (small seeds start near 0)
        for burst in 0..8u64 {
            let offset_s = (rng.next_f64() * 2.0 - 1.0) * 0.040;
            est.break_segment();
            // 5 s of packets every 15 s.
            let start_frame = burst * 15 * 48_000;
            for k in 0..500u64 {
                let frame = start_frame + k * 480;
                let t = frame as f64 / rate + offset_s + (rng.next_f64() - 0.5) * 0.002;
                est.observe(HostTime(BASE_NS + (t * 1e9) as u64), frame);
            }
        }
        let e = est.estimate().expect("estimate");
        assert!((e.ppm - ppm).abs() < 10.0, "estimated {:.2} ppm", e.ppm);
        let err = (e.ppm - ppm).abs();
        assert!(err < 3.0 * e.ppm_stderr, "err {err} vs se {}", e.ppm_stderr);
    }

    #[test]
    fn predicts_host_time_of_frame() {
        let e = simulate(80.0, 0.5, 60.0);
        let rate = 48_000.0 * (1.0 + 80.0 / 1e6);
        let frame = 2_000_000u64;
        let expected = BASE_NS as f64 + frame as f64 / rate * 1e9;
        let got = e.host_time_of_frame(frame).as_ns() as f64;
        assert!(
            (got - expected).abs() < 1.0e6,
            "off by {} ns",
            got - expected
        );
        assert!((e.frame_at(HostTime(expected as u64)) - frame as f64).abs() < 50.0);
    }
}
