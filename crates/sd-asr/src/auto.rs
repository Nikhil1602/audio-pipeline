//! Auto language selection across a session.
//!
//! Language identification is least reliable on short segments, while a meeting rarely changes
//! language from one segment to the next. [`SessionLanguage`] therefore combines each
//! segment's detection with a decaying prior built from earlier decisions:
//!
//! * a confident detection on a long enough segment is taken as is;
//! * otherwise the segment's probabilities are multiplied by the prior and renormalized, so
//!   short or uncertain segments lean toward the session's language;
//! * the prior decays, so a real switch (e.g. Hindi to English) wins after a segment or two.
//!
//! Thresholds are defaults to be tuned on measured LID output (docs/m3-plan.md).

use std::collections::BTreeMap;

use crate::lang::Language;

/// Spoken languages that Auto treats as one, written in the second language's script: a
/// decision for `.0` is written as `.1`. Spoken Hindi and Urdu are close enough that language
/// identification splits them almost at random (measured on FLEURS Hindi: IndicConformer
/// 48% Hindi / 50% Urdu, Whisper 86% / 11%), and what differs is the script. Auto therefore
/// writes Hindi (Devanagari); choosing Urdu explicitly gives Urdu script.
///
/// The language is decided on the unmerged probabilities and relabelled afterwards. Adding
/// Urdu's probability to Hindi's before deciding was measured to pull Punjabi and Gujarati
/// speech into Hindi (FLEURS, 15-way Auto: Punjabi 50%, Gujarati 82%; relabelling after the
/// decision: 93% and 98%, 99.0% over all 15 languages).
pub const SAME_SPEECH: [(Language, Language); 1] = [(Language::Urdu, Language::Hindi)];

/// The language a decision for `language` is written in: `to` for a `(from, to)` pair in
/// `merges` when `to` is among `candidates` (so restricting Auto to Urdu still yields Urdu),
/// otherwise `language` itself.
pub fn same_speech_target(
    language: Language,
    merges: &[(Language, Language)],
    candidates: &[Language],
) -> Language {
    merges
        .iter()
        .find(|(from, to)| *from == language && candidates.contains(to))
        .map_or(language, |m| m.1)
}

#[derive(Debug, Clone, Copy)]
pub struct AutoConfig {
    /// A detection at least this probable on a segment at least `min_confident_s` long is
    /// accepted without the prior.
    pub confident_prob: f32,
    pub min_confident_s: f32,
    /// Weight of the prior (0 = ignore, 1 = full Bayesian combination).
    pub prior_weight: f32,
    /// Each new segment multiplies older evidence by this.
    pub decay: f32,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            confident_prob: 0.8,
            min_confident_s: 3.0,
            prior_weight: 0.7,
            decay: 0.7,
        }
    }
}

/// The decision for one segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDecision {
    pub language: Language,
    /// Combined probability of the chosen language.
    pub probability: f32,
    /// True if the session prior changed the segment's own top choice.
    pub prior_overrode: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SessionLanguage {
    cfg: AutoConfig,
    /// Decayed, duration-weighted evidence per language.
    evidence: BTreeMap<Language, f32>,
}

impl SessionLanguage {
    pub fn new(cfg: AutoConfig) -> Self {
        Self {
            cfg,
            evidence: BTreeMap::new(),
        }
    }

    /// The language the session has mostly been in so far, if any.
    pub fn current(&self) -> Option<Language> {
        self.evidence
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(l, _)| *l)
    }

    /// Chooses a language for a segment from its detection `probs` (any order; need not sum
    /// to 1) and its duration, then records the decision.
    pub fn decide(&mut self, probs: &[(Language, f32)], duration_s: f32) -> Option<AutoDecision> {
        let own = probs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).copied()?;
        let total_evidence: f32 = self.evidence.values().sum();
        let decision = if (own.1 >= self.cfg.confident_prob
            && duration_s >= self.cfg.min_confident_s)
            || total_evidence <= 0.0
        {
            AutoDecision {
                language: own.0,
                probability: own.1,
                prior_overrode: false,
            }
        } else {
            // posterior ∝ p(segment) · prior^w, with a floor so unseen languages stay possible.
            let floor = 0.05;
            let mut scored: Vec<(Language, f32)> = probs
                .iter()
                .map(|&(l, p)| {
                    let prior = self.evidence.get(&l).copied().unwrap_or(0.0) / total_evidence;
                    (l, p * (prior + floor).powf(self.cfg.prior_weight))
                })
                .collect();
            let z: f32 = scored.iter().map(|(_, s)| s).sum();
            scored
                .iter_mut()
                .for_each(|(_, s)| *s /= z.max(f32::MIN_POSITIVE));
            let best = scored
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("non-empty");
            AutoDecision {
                language: best.0,
                probability: best.1,
                prior_overrode: best.0 != own.0,
            }
        };
        // The prior accumulates each segment's *own* detection, not the decision: feeding back
        // decisions would let the prior lock itself in and block a real language switch.
        self.evidence
            .values_mut()
            .for_each(|e| *e *= self.cfg.decay);
        let z: f32 = probs
            .iter()
            .map(|(_, p)| p)
            .sum::<f32>()
            .max(f32::MIN_POSITIVE);
        for &(l, p) in probs {
            *self.evidence.entry(l).or_default() += duration_s.max(0.5) * p / z;
        }
        Some(decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urdu_decisions_are_written_as_hindi_when_hindi_is_allowed() {
        use Language::*;
        let all = [Hindi, Urdu, English, Punjabi];
        assert_eq!(same_speech_target(Urdu, &SAME_SPEECH, &all), Hindi);
        assert_eq!(same_speech_target(Punjabi, &SAME_SPEECH, &all), Punjabi);
        assert_eq!(
            same_speech_target(Urdu, &SAME_SPEECH, &[Urdu, English]),
            Urdu,
            "Hindi not allowed: Urdu stays"
        );
    }
    use Language::*;

    #[test]
    fn short_uncertain_segments_follow_the_session() {
        let mut s = SessionLanguage::new(AutoConfig::default());
        for _ in 0..3 {
            let d = s.decide(&[(Hindi, 0.9), (Urdu, 0.1)], 6.0).unwrap();
            assert_eq!(d.language, Hindi);
        }
        // A short segment that LID tilts toward Urdu (Hindi and Urdu sound alike).
        let d = s.decide(&[(Urdu, 0.55), (Hindi, 0.45)], 1.2).unwrap();
        assert_eq!(d.language, Hindi);
        assert!(d.prior_overrode);
    }

    #[test]
    fn confident_long_segments_switch_immediately() {
        let mut s = SessionLanguage::new(AutoConfig::default());
        for _ in 0..5 {
            s.decide(&[(Hindi, 0.9), (English, 0.1)], 6.0);
        }
        let d = s.decide(&[(English, 0.95), (Hindi, 0.05)], 5.0).unwrap();
        assert_eq!(d.language, English);
        assert!(!d.prior_overrode);
    }

    #[test]
    fn a_sustained_switch_wins_even_when_unsure() {
        let mut s = SessionLanguage::new(AutoConfig::default());
        for _ in 0..4 {
            s.decide(&[(Hindi, 0.9), (Gujarati, 0.1)], 6.0);
        }
        // Moderately confident Gujarati, segment after segment.
        let mut chosen = Vec::new();
        for _ in 0..6 {
            chosen.push(
                s.decide(&[(Gujarati, 0.7), (Hindi, 0.3)], 2.0)
                    .unwrap()
                    .language,
            );
        }
        // Decisions follow by the third segment; the dominant (decayed) language by the sixth.
        assert_eq!(&chosen[2..], &[Gujarati; 4], "{chosen:?}");
        assert_eq!(s.current(), Some(Gujarati));
    }

    #[test]
    fn first_segment_has_no_prior() {
        let mut s = SessionLanguage::new(AutoConfig::default());
        assert_eq!(s.current(), None);
        let d = s.decide(&[(Tamil, 0.4), (Malayalam, 0.35)], 1.0).unwrap();
        assert_eq!(d.language, Tamil);
        assert!(s.decide(&[], 1.0).is_none());
    }
}
