//! Host-monotonic clock.
//!
//! Every captured block is stamped with a [`HostTime`] in nanoseconds on the same clock the OS
//! audio stack reports device positions in, so mic and system streams can be aligned:
//!
//! * Windows: `QueryPerformanceCounter` (WASAPI `qpcPosition` is QPC scaled to 100 ns units).
//! * macOS: `CLOCK_UPTIME_RAW` (same time base as `mach_absolute_time` / Core Audio host time).
//! * Linux and other Unix: `CLOCK_MONOTONIC` (PipeWire and ALSA report timestamps on it).

use std::time::Duration;

/// A point on the host-monotonic clock, in nanoseconds since an arbitrary, boot-relative epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HostTime(pub u64);

impl HostTime {
    /// Current host time. Safe to call from real-time threads (no allocation, no locks).
    #[inline]
    pub fn now() -> Self {
        HostTime(platform::now_ns())
    }

    #[inline]
    pub const fn from_ns(ns: u64) -> Self {
        HostTime(ns)
    }

    #[inline]
    pub const fn as_ns(self) -> u64 {
        self.0
    }

    /// Converts a WASAPI `qpcPosition` (QPC expressed in 100 ns units) to host time.
    #[inline]
    pub const fn from_100ns_units(v: u64) -> Self {
        HostTime(v.saturating_mul(100))
    }

    /// Elapsed time from `earlier` to `self`, saturating at zero.
    #[inline]
    pub fn saturating_duration_since(self, earlier: HostTime) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    /// Signed difference `self - other` in nanoseconds.
    #[inline]
    pub fn signed_diff_ns(self, other: HostTime) -> i64 {
        (self.0 as i128 - other.0 as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    #[inline]
    pub fn saturating_add_ns(self, ns: u64) -> HostTime {
        HostTime(self.0.saturating_add(ns))
    }
}

#[cfg(windows)]
mod platform {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };

    fn frequency() -> u64 {
        static FREQ: OnceLock<u64> = OnceLock::new();
        *FREQ.get_or_init(|| {
            let mut f = 0i64;
            // SAFETY: `f` is a valid, writable i64. The call cannot fail on Windows XP or later.
            unsafe { QueryPerformanceFrequency(&mut f) };
            (f as u64).max(1)
        })
    }

    /// Converts raw QPC ticks to nanoseconds.
    pub fn qpc_ticks_to_ns(ticks: u64) -> u64 {
        (ticks as u128 * 1_000_000_000 / frequency() as u128) as u64
    }

    pub fn now_ns() -> u64 {
        let mut t = 0i64;
        // SAFETY: `t` is a valid, writable i64. The call cannot fail on Windows XP or later.
        unsafe { QueryPerformanceCounter(&mut t) };
        qpc_ticks_to_ns(t as u64)
    }
}

#[cfg(unix)]
mod platform {
    #[cfg(target_vendor = "apple")]
    const CLOCK: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_vendor = "apple"))]
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC;

    pub fn now_ns() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid, writable timespec and CLOCK is supported on this platform.
        let rc = unsafe { libc::clock_gettime(CLOCK, &mut ts) };
        debug_assert_eq!(rc, 0);
        (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64
    }
}

#[cfg(windows)]
pub use platform::qpc_ticks_to_ns;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_and_advancing() {
        let a = HostTime::now();
        std::thread::sleep(Duration::from_millis(20));
        let b = HostTime::now();
        assert!(b > a);
        let d = b.saturating_duration_since(a);
        assert!(d >= Duration::from_millis(15), "elapsed {d:?}");
        assert!(d < Duration::from_secs(2), "elapsed {d:?}");
        assert_eq!(a.saturating_duration_since(b), Duration::ZERO);
        assert!(a.signed_diff_ns(b) < 0);
    }

    #[test]
    fn hundred_ns_units() {
        assert_eq!(HostTime::from_100ns_units(12).as_ns(), 1_200);
        assert_eq!(HostTime::from_100ns_units(u64::MAX).as_ns(), u64::MAX);
    }

    #[cfg(windows)]
    #[test]
    fn wasapi_qpc_position_matches_now() {
        // WASAPI reports qpcPosition as QPC * 10^7 / frequency; it must land on the same clock.
        let mut t = 0i64;
        // SAFETY: valid out-pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut t) };
        let as_100ns = qpc_ticks_to_ns(t as u64) / 100;
        let now = HostTime::now();
        let diff = now.signed_diff_ns(HostTime::from_100ns_units(as_100ns));
        assert!((0..50_000_000).contains(&diff), "diff {diff} ns");
    }
}
