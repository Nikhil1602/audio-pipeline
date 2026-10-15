//! Platform-independent capture policies shared by every backend.

/// What to do with a timeline gap reported by the aligner (a device glitch or restart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapAction {
    /// Insert this many frames of silence.
    Fill,
    /// Advance the timeline without data (the journal records a gap).
    Skip,
    /// As `Skip`, and this gap exhausted the session's fill budget: report it once and
    /// restart the device stream.
    SkipBudgetExhausted,
}

/// Decides whether gaps are filled with silence.
///
/// * A single gap longer than `max_gap_frames` (e.g. after system sleep) is skipped.
/// * Silence inserted for gaps is capped per capture session. A device that keeps reporting
///   gaps would otherwise inflate the timeline with silence; the cap is 300 s per
///   session, and then the device is retried. After
///   the cap, further gaps are skipped.
///
/// Idle fill (loopback silence while nothing plays) is not a gap and is not counted.
#[derive(Debug, Clone)]
pub struct GapFillPolicy {
    max_gap_frames: u64,
    budget_frames: u64,
    used_frames: u64,
    exhausted: bool,
}

impl GapFillPolicy {
    pub fn new(sample_rate: u32, max_gap_ms: u32, session_budget_ms: u32) -> Self {
        let frames = |ms: u32| sample_rate as u64 * ms as u64 / 1000;
        Self {
            max_gap_frames: frames(max_gap_ms),
            budget_frames: frames(session_budget_ms),
            used_frames: 0,
            exhausted: false,
        }
    }

    pub fn decide(&mut self, gap_frames: u64) -> GapAction {
        if gap_frames == 0 || gap_frames > self.max_gap_frames {
            return GapAction::Skip;
        }
        if self.exhausted {
            return GapAction::Skip;
        }
        if self.used_frames + gap_frames > self.budget_frames {
            self.exhausted = true;
            return GapAction::SkipBudgetExhausted;
        }
        self.used_frames += gap_frames;
        GapAction::Fill
    }

    /// Silence inserted for gaps so far, in frames.
    pub fn used_frames(&self) -> u64 {
        self.used_frames
    }

    pub fn exhausted(&self) -> bool {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_until_budget_then_skips() {
        // 1 kHz for easy numbers: max single gap 2 s, budget 5 s.
        let mut p = GapFillPolicy::new(1_000, 2_000, 5_000);
        assert_eq!(p.decide(0), GapAction::Skip);
        assert_eq!(p.decide(2_500), GapAction::Skip, "single gap too long");
        assert_eq!(p.used_frames(), 0, "skipped gaps use no budget");
        for _ in 0..2 {
            assert_eq!(p.decide(2_000), GapAction::Fill);
        }
        assert_eq!(p.decide(999), GapAction::Fill);
        assert_eq!(p.used_frames(), 4_999);
        assert_eq!(
            p.decide(2),
            GapAction::SkipBudgetExhausted,
            "would exceed 5 s"
        );
        assert!(p.exhausted());
        assert_eq!(p.decide(1), GapAction::Skip, "reported only once");
        assert_eq!(p.used_frames(), 4_999);
    }

    #[test]
    fn default_budget_is_five_minutes() {
        let cfg = crate::CaptureConfig::new(sd_core::Source::Mic);
        let mut p = GapFillPolicy::new(
            cfg.format.sample_rate,
            cfg.max_gap_fill_ms,
            cfg.max_session_gap_fill_ms,
        );
        let mut filled_s = 0.0;
        while p.decide(48_000) == GapAction::Fill {
            filled_s += 1.0;
        }
        assert_eq!(filled_s, 300.0);
    }
}

/// Detects a started stream that stopped delivering audio.
///
/// A microphone stream delivers packets continuously, so a long silence from the device means
/// it stalled (driver hang, endpoint suspended). A loopback stream legitimately delivers
/// nothing while nothing plays, so backends report a loopback stall only when audio is
/// playing: the endpoint's peak meter shows it (WASAPI), or some client runs the output device
/// (Core Audio).
#[derive(Debug, Clone)]
pub struct StallWatchdog {
    timeout: std::time::Duration,
    last_packet: std::time::Instant,
}

impl StallWatchdog {
    pub fn new(timeout_ms: u32, now: std::time::Instant) -> Self {
        Self {
            timeout: std::time::Duration::from_millis(timeout_ms as u64),
            last_packet: now,
        }
    }

    pub fn packet(&mut self, now: std::time::Instant) {
        self.last_packet = now;
    }

    /// How long nothing has arrived, if that exceeds the timeout.
    pub fn stalled_for(&self, now: std::time::Instant) -> Option<std::time::Duration> {
        let quiet = now.saturating_duration_since(self.last_packet);
        (quiet > self.timeout).then_some(quiet)
    }
}

#[cfg(test)]
mod watchdog_tests {
    use super::StallWatchdog;
    use std::time::{Duration, Instant};

    #[test]
    fn reports_only_after_timeout_and_resets_on_packets() {
        let t0 = Instant::now();
        let mut w = StallWatchdog::new(2_000, t0);
        assert_eq!(w.stalled_for(t0 + Duration::from_millis(1_999)), None);
        assert_eq!(
            w.stalled_for(t0 + Duration::from_millis(2_500)),
            Some(Duration::from_millis(2_500))
        );
        w.packet(t0 + Duration::from_millis(2_500));
        assert_eq!(w.stalled_for(t0 + Duration::from_millis(4_000)), None);
    }
}
