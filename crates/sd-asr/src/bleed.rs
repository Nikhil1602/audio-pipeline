//! Removes mic transcript text that is really the far end leaking into the microphone.
//!
//! Echo cancellation removes most of it at the audio level (M2). This is the safety net for
//! what is left: chiefly speech detected before the echo gate decided (`echo_unverified`,
//! M2.5) and residual echo. Rules:
//!
//! * **Flagged** mic segments (`echo_unverified`): the mic text is dropped if it matches a
//!   substring of system text from within ±4 s with an edit distance of at most
//!   max(⌈0.35 · length⌉, 2) characters, after normalization.
//! * **Unflagged** mic segments passed through AEC, so real double talk is likely. They are
//!   dropped only for a near-copy: at least 3 words and at least 80 % character similarity to
//!   a system substring.

use crate::metrics::normalize;

/// System-audio text near a mic segment, with its time span (ms on the session clock).
#[derive(Debug, Clone)]
pub struct SystemText<'a> {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: &'a str,
}

/// How far around the mic segment system text is considered.
pub const WINDOW_MS: i64 = 4_000;

/// Why a mic segment was judged to be far-end bleed, if it was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BleedMatch {
    /// Character edit distance to the best-matching system substring.
    pub distance: usize,
    /// 1 − distance / mic length.
    pub similarity: f64,
}

/// Decides whether the mic text `mic` (spanning `start_ms..end_ms`) is far-end bleed.
pub fn is_bleed(
    mic: &str,
    start_ms: i64,
    end_ms: i64,
    echo_unverified: bool,
    system: &[SystemText<'_>],
) -> Option<BleedMatch> {
    let m: Vec<char> = normalize(mic).chars().collect();
    let words = m.iter().filter(|c| **c == ' ').count() + usize::from(!m.is_empty());
    if m.is_empty() {
        return None;
    }
    let nearby: String = system
        .iter()
        .filter(|s| s.end_ms >= start_ms - WINDOW_MS && s.start_ms <= end_ms + WINDOW_MS)
        .map(|s| normalize(s.text))
        .collect::<Vec<_>>()
        .join(" ");
    if nearby.is_empty() {
        return None;
    }
    let s: Vec<char> = nearby.chars().collect();
    let distance = substring_distance(&m, &s);
    let similarity = 1.0 - distance as f64 / m.len() as f64;
    let found = BleedMatch {
        distance,
        similarity,
    };
    let limit = ((0.35 * m.len() as f64).ceil() as usize).max(2);
    if echo_unverified {
        (distance <= limit).then_some(found)
    } else {
        (words >= 3 && similarity >= 0.8).then_some(found)
    }
}

/// Smallest edit distance between `pattern` and any substring of `text` (Sellers' algorithm:
/// the match may start and end anywhere in `text`).
pub fn substring_distance(pattern: &[char], text: &[char]) -> usize {
    // prev[j]: distance between pattern[..i] and the best substring of text ending at j.
    let mut prev = vec![0usize; text.len() + 1];
    let mut cur = vec![0usize; text.len() + 1];
    for (i, p) in pattern.iter().enumerate() {
        cur[0] = i + 1;
        for (j, t) in text.iter().enumerate() {
            let sub = prev[j] + usize::from(p != t);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev.into_iter().min().unwrap_or(pattern.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(t: &str, start: i64, end: i64) -> SystemText<'_> {
        SystemText {
            start_ms: start,
            end_ms: end,
            text: t,
        }
    }

    #[test]
    fn substring_distance_basics() {
        let c = |s: &str| s.chars().collect::<Vec<_>>();
        assert_eq!(substring_distance(&c("cat"), &c("the cat sat")), 0);
        assert_eq!(substring_distance(&c("cot"), &c("the cat sat")), 1);
        assert_eq!(substring_distance(&c("xyz"), &c("")), 3);
    }

    #[test]
    fn flagged_echo_of_far_end_is_dropped() {
        let system = [sys(
            "Today we reviewed the project timeline.",
            10_000,
            13_000,
        )];
        // The leaked far end, slightly misrecognized on the mic path.
        let m = is_bleed(
            "today we review the project time line",
            10_200,
            13_100,
            true,
            &system,
        );
        assert!(m.is_some(), "{m:?}");
    }

    #[test]
    fn real_speech_during_far_end_is_kept() {
        let system = [sys(
            "Today we reviewed the project timeline.",
            10_000,
            13_000,
        )];
        let near = "please send me the meeting notes by friday";
        assert!(is_bleed(near, 10_500, 13_500, true, &system).is_none());
        assert!(is_bleed(near, 10_500, 13_500, false, &system).is_none());
    }

    #[test]
    fn unflagged_needs_a_near_copy() {
        let system = [sys("we will measure accuracy before choosing", 0, 3_000)];
        // A loose paraphrase passes the flagged rule's fuzzy threshold but not the strict one.
        let loose = "we measure the accuracy first";
        assert!(is_bleed(loose, 0, 3_000, false, &system).is_none());
        let copy = "we will measure accuracy before choosing";
        assert!(is_bleed(copy, 0, 3_000, false, &system).is_some());
        // Too short to call a copy without the flag.
        assert!(is_bleed("okay", 0, 1_000, false, &[sys("okay", 0, 1_000)]).is_none());
    }

    #[test]
    fn only_nearby_system_text_counts() {
        let far_away = [sys(
            "today we reviewed the project timeline",
            60_000,
            63_000,
        )];
        assert!(
            is_bleed(
                "today we reviewed the project timeline",
                10_000,
                13_000,
                true,
                &far_away
            )
            .is_none()
        );
    }

    #[test]
    fn works_for_indic_text() {
        let system = [sys("आज हमने परियोजना की समय-सीमा की समीक्षा की।", 0, 4_000)];
        assert!(
            is_bleed(
                "आज हमने परियोजना की समय सीमा की समीक्षा की",
                100,
                4_100,
                true,
                &system
            )
            .is_some()
        );
        assert!(is_bleed("कृपया शुक्रवार तक बैठक के नोट्स भेजें", 100, 4_100, true, &system).is_none());
    }
}
