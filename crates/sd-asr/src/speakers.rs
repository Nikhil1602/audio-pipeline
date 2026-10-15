//! Speaker information on finals (from a [`crate::runner::SpeakerTracker`]) and the split of
//! a final's text where the speaker changes.

use crate::Segment;

/// A time span of one speaker, in 16 kHz samples of the source's timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerSpan {
    pub start: u64,
    pub end: u64,
    pub speaker: u32,
}

/// What the tracker knows about one utterance.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeakerInfo {
    /// Majority speaker.
    pub speaker: Option<u32>,
    /// Who speaks when within the utterance, in time order.
    pub turns: Vec<SpeakerSpan>,
    /// Second speakers where two people speak at once.
    pub overlaps: Vec<SpeakerSpan>,
}

/// A stretch of a final's text spoken by one speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerPart {
    pub start: u64,
    pub end: u64,
    pub speaker: u32,
    pub text: String,
}

/// Parts shorter than this many words and [`MIN_PART_MS`] are merged into a neighbour, so a
/// single misattributed window does not break a sentence apart.
pub const MIN_PART_WORDS: usize = 3;
pub const MIN_PART_MS: i64 = 1500;

const SAMPLES_PER_MS: i64 = 16;

fn speaker_at(info: &SpeakerInfo, sample: u64) -> Option<u32> {
    info.turns
        .iter()
        .find(|t| t.start <= sample && sample < t.end)
        .or_else(|| {
            info.turns
                .iter()
                .min_by_key(|t| t.start.abs_diff(sample).min(t.end.abs_diff(sample)))
        })
        .map(|t| t.speaker)
        .or(info.speaker)
}

/// Splits a final (`text`, sample range `start..end`, absolute word timings in ms) into
/// speaker parts. One part covering the whole final when the speaker does not change, the
/// engine gave no word timings, or there is no speaker.
pub fn split(
    text: &str,
    start: u64,
    end: u64,
    words: &[Segment],
    info: &SpeakerInfo,
) -> Vec<SpeakerPart> {
    let Some(primary) = info.speaker else {
        return Vec::new();
    };
    let whole = || {
        vec![SpeakerPart {
            start,
            end,
            speaker: primary,
            text: text.to_string(),
        }]
    };
    if words.is_empty() || info.turns.len() < 2 {
        return whole();
    }
    // (speaker, first word, word count)
    let mut parts: Vec<(u32, usize, usize)> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        let mid = ((w.start_ms + w.end_ms) / 2).max(0) * SAMPLES_PER_MS;
        let s = speaker_at(info, mid as u64).unwrap_or(primary);
        match parts.last_mut() {
            Some(p) if p.0 == s => p.2 += 1,
            _ => parts.push((s, i, 1)),
        }
    }
    let duration = |p: &(u32, usize, usize)| words[p.1 + p.2 - 1].end_ms - words[p.1].start_ms;
    // Merge short parts into a neighbour, preferring the one that makes both sides agree.
    loop {
        if parts.len() < 2 {
            break;
        }
        let Some(k) = parts
            .iter()
            .position(|p| p.2 < MIN_PART_WORDS && duration(p) < MIN_PART_MS)
        else {
            break;
        };
        let into = if k == 0 {
            1
        } else if k + 1 == parts.len()
            || parts[k - 1].0 == parts[k + 1].0
            || parts[k - 1].2 >= parts[k + 1].2
        {
            k - 1
        } else {
            k + 1
        };
        let removed = parts.remove(k);
        let target = if into > k { into - 1 } else { into };
        let t = &mut parts[target];
        t.1 = t.1.min(removed.1);
        t.2 += removed.2;
        // Neighbours that now share a speaker become one part.
        let mut merged: Vec<(u32, usize, usize)> = Vec::with_capacity(parts.len());
        for p in parts {
            match merged.last_mut() {
                Some(m) if m.0 == p.0 => m.2 += p.2,
                _ => merged.push(p),
            }
        }
        parts = merged;
    }
    if parts.len() < 2 {
        return whole();
    }
    // Scripts written without spaces (Chinese) join words directly.
    let sep = if text.contains(' ') { " " } else { "" };
    // Parts are contiguous: each boundary lies midway between the words on either side,
    // inside the final and never before the previous one (engine word times can be slightly
    // off or run past the audio).
    let mut bounds = vec![start];
    for pair in parts.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let mid_ms = (words[a.1 + a.2 - 1].end_ms + words[b.1].start_ms) / 2;
        let at = (mid_ms.max(0) * SAMPLES_PER_MS) as u64;
        let prev = *bounds.last().unwrap_or(&start);
        bounds.push(at.clamp(prev, end.max(prev)));
    }
    bounds.push(end.max(*bounds.last().unwrap_or(&start)));
    parts
        .iter()
        .enumerate()
        .map(|(k, &(speaker, first, n))| {
            let ws = &words[first..first + n];
            SpeakerPart {
                start: bounds[k],
                end: bounds[k + 1],
                speaker,
                text: ws
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(sep),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start_ms: i64, end_ms: i64, text: &str) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
            no_speech_prob: 0.0,
        }
    }

    fn info(turns: &[(u64, u64, u32)], speaker: u32) -> SpeakerInfo {
        SpeakerInfo {
            speaker: Some(speaker),
            turns: turns
                .iter()
                .map(|&(start, end, speaker)| SpeakerSpan {
                    start,
                    end,
                    speaker,
                })
                .collect(),
            overlaps: vec![],
        }
    }

    #[test]
    fn splits_at_a_speaker_change() {
        let words: Vec<Segment> = [
            "so", "the", "plan", "is", "ready", "great", "thanks", "a", "lot",
        ]
        .iter()
        .enumerate()
        .map(|(i, t)| w(i as i64 * 500, i as i64 * 500 + 400, t))
        .collect();
        // Speaker 0 for the first 2.5 s, then speaker 1.
        let i = info(&[(0, 40_000, 0), (40_000, 80_000, 1)], 0);
        let parts = split(
            "so the plan is ready great thanks a lot",
            0,
            72_000,
            &words,
            &i,
        );
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!(
            (parts[0].speaker, parts[0].text.as_str()),
            (0, "so the plan is ready")
        );
        assert_eq!(
            (parts[1].speaker, parts[1].text.as_str()),
            (1, "great thanks a lot")
        );
        assert_eq!(parts[0].start, 0);
        assert_eq!(parts[1].end, 72_000);
    }

    #[test]
    fn parts_stay_inside_the_final_when_word_times_overrun() {
        // The engine's last words claim times past the end of the audio (2 s).
        let words: Vec<Segment> = (0..8).map(|i| w(i * 400, i * 400 + 300, "w")).collect();
        let i = info(&[(0, 19_200, 0), (19_200, 64_000, 1)], 0);
        let parts = split("w w w w w w w w", 0, 32_000, &words, &i);
        assert_eq!(parts.len(), 2, "{parts:?}");
        for p in &parts {
            assert!(p.start <= p.end && p.end <= 32_000, "{parts:?}");
        }
        assert_eq!(parts[0].end, parts[1].start);
    }

    #[test]
    fn a_single_misattributed_word_does_not_split() {
        let words: Vec<Segment> = (0..8).map(|i| w(i * 400, i * 400 + 300, "word")).collect();
        // One window in the middle labelled speaker 1 covers a single word.
        let i = info(
            &[(0, 25_000, 0), (25_000, 30_000, 1), (30_000, 60_000, 0)],
            0,
        );
        let parts = split("eight words", 0, 60_000, &words, &i);
        assert_eq!(parts.len(), 1, "{parts:?}");
        assert_eq!(parts[0].text, "eight words");
    }

    #[test]
    fn whole_final_without_words_or_speaker() {
        let i = info(&[(0, 10, 0), (10, 20, 1)], 0);
        assert_eq!(split("hi", 0, 20, &[], &i).len(), 1);
        assert!(split("hi", 0, 20, &[], &SpeakerInfo::default()).is_empty());
    }
}
