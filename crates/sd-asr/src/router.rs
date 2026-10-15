//! Engines behind one interface, and the per-language routing between them.

use std::sync::Arc;

use crate::lang::Language;
use crate::{AsrError, Result, Transcript};

/// A speech recognizer for some set of languages.
pub trait Recognizer: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, language: Language) -> bool;
    /// Transcribes 16 kHz mono audio in `language`.
    fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript>;
}

/// Spoken-language identification over a set of allowed languages.
pub trait LanguageDetector: Send + Sync {
    /// Probabilities over the allowed languages it can identify, highest first.
    fn detect(&self, pcm: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>>;
}

/// Chooses the engine for each language: the first engine in `preference` order that supports
/// it. The order is set from measured accuracy (docs/m3-report.md); for now Whisper only.
#[derive(Clone, Default)]
pub struct Router {
    engines: Vec<Arc<dyn Recognizer>>,
    /// Per-language override: engine names tried first for that language.
    overrides: Vec<(Language, Vec<String>)>,
}

impl Router {
    pub fn new(engines: Vec<Arc<dyn Recognizer>>) -> Self {
        Self {
            engines,
            overrides: Vec::new(),
        }
    }

    /// Tries `engines` (by name) first for `language`.
    pub fn prefer(mut self, language: Language, engines: &[&str]) -> Self {
        self.overrides
            .push((language, engines.iter().map(|s| s.to_string()).collect()));
        self
    }

    pub fn pick(&self, language: Language) -> Option<&Arc<dyn Recognizer>> {
        let preferred = self
            .overrides
            .iter()
            .filter(|(l, _)| *l == language)
            .flat_map(|(_, names)| names.iter())
            .filter_map(|n| self.engines.iter().find(|e| e.name() == n));
        preferred
            .chain(self.engines.iter())
            .find(|e| e.supports(language))
    }

    /// Languages at least one engine supports.
    pub fn supported(&self) -> Vec<Language> {
        Language::ALL
            .into_iter()
            .filter(|l| self.engines.iter().any(|e| e.supports(*l)))
            .collect()
    }

    pub fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript> {
        self.pick(language)
            .ok_or(AsrError::Unsupported(language))?
            .transcribe(pcm, language)
    }
}

#[cfg(feature = "whisper")]
impl Recognizer for crate::whisper::WhisperEngine {
    fn name(&self) -> &str {
        "whisper"
    }
    fn supports(&self, language: Language) -> bool {
        language.whisper_code().is_some()
    }
    fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript> {
        crate::whisper::WhisperEngine::transcribe(
            self,
            pcm,
            crate::lang::LanguageChoice::Fixed(language),
            &[language],
        )
    }
}

#[cfg(feature = "whisper")]
impl LanguageDetector for crate::whisper::WhisperEngine {
    fn detect(&self, pcm: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
        self.detect_language(pcm, allowed)
    }
}

#[cfg(feature = "indic")]
impl Recognizer for crate::indic::IndicConformer {
    fn name(&self) -> &str {
        "indicconformer"
    }
    fn supports(&self, language: Language) -> bool {
        self.languages().contains(&language)
    }
    fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript> {
        crate::indic::IndicConformer::transcribe(self, pcm, language)
    }
}

#[cfg(feature = "indic")]
impl LanguageDetector for crate::indic::IndicConformer {
    fn detect(&self, pcm: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
        crate::indic::IndicConformer::detect(self, pcm, allowed)
    }
}

/// Two-stage language identification: `first` decides among all allowed languages; the
/// probability it gives to the `refined` group is then redistributed within the group by
/// `second`.
///
/// Used for Auto mode: Whisper separates English/Chinese/Spanish/French from the Indic
/// languages well but confuses the Indic languages with each other (and cannot identify
/// Odia); IndicConformer, whose vocabulary is per language, tells the Indic languages apart.
pub struct CascadeDetector {
    first: Arc<dyn LanguageDetector>,
    second: Arc<dyn LanguageDetector>,
    refined: Vec<Language>,
    /// The second stage runs only when the group holds at least this much probability.
    pub min_group_prob: f32,
}

impl CascadeDetector {
    pub fn new(
        first: Arc<dyn LanguageDetector>,
        second: Arc<dyn LanguageDetector>,
        refined: Vec<Language>,
    ) -> Self {
        Self {
            first,
            second,
            refined,
            min_group_prob: 0.2,
        }
    }
}

impl LanguageDetector for CascadeDetector {
    fn detect(&self, pcm: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
        let first = self.first.detect(pcm, allowed)?;
        let group: Vec<Language> = allowed
            .iter()
            .copied()
            .filter(|l| self.refined.contains(l))
            .collect();
        let in_group = |l: &Language| group.contains(l);
        // Group languages the first stage cannot identify (Odia for Whisper) are reached
        // through the group's share: Odia speech lands on the other Indic languages.
        let mass: f32 = first.iter().filter(|(l, _)| in_group(l)).map(|p| p.1).sum();
        if group.is_empty() || mass < self.min_group_prob {
            return Ok(first);
        }
        let second = self.second.detect(pcm, &group)?;
        if second.is_empty() {
            return Ok(first);
        }
        let mut out: Vec<(Language, f32)> =
            first.into_iter().filter(|(l, _)| !in_group(l)).collect();
        out.extend(second.into_iter().map(|(l, p)| (l, p * mass)));
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        group_first(&mut out, &in_group, mass);
        Ok(out)
    }
}

/// When a language outside the group is on top but the group's share as a whole is larger
/// (Indic speech with English words: Whisper gives English, say, 0.33 and the Indic languages
/// 0.6 split eleven ways), the group wins: its languages are rescaled so that the best one
/// carries the whole share. Measured on the live session (docs/m3-report.md) and on FLEURS
/// (tools/eval/lid_rules.py: 99.0% -> 99.2% per clip, no language worse in meetings).
/// Nothing changes when a group language is already on top, so the session prior keeps
/// correcting close pairs such as Punjabi/Hindi.
fn group_first(out: &mut [(Language, f32)], in_group: &impl Fn(&Language) -> bool, mass: f32) {
    let Some(&(top_lang, top_p)) = out.first() else {
        return;
    };
    if in_group(&top_lang) || mass <= top_p {
        return;
    }
    let best_group = out
        .iter()
        .filter(|(l, _)| in_group(l))
        .map(|p| p.1)
        .fold(0.0f32, f32::max);
    if best_group <= 0.0 {
        return;
    }
    for (l, p) in out.iter_mut() {
        if in_group(l) {
            *p *= mass / best_group;
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str, Vec<Language>);
    impl Recognizer for Fake {
        fn name(&self) -> &str {
            self.0
        }
        fn supports(&self, l: Language) -> bool {
            self.1.contains(&l)
        }
        fn transcribe(&self, _: &[f32], language: Language) -> Result<Transcript> {
            Ok(Transcript {
                text: self.0.into(),
                language,
                language_probs: vec![],
                segments: vec![],
                words: vec![],
            })
        }
    }

    #[test]
    fn routes_by_support_and_preference() {
        use Language::*;
        let whisper: Arc<dyn Recognizer> = Arc::new(Fake("whisper", vec![English, Hindi]));
        let indic: Arc<dyn Recognizer> = Arc::new(Fake("indic", vec![Hindi, Odia]));
        let r = Router::new(vec![whisper, indic]);
        assert_eq!(r.pick(English).unwrap().name(), "whisper");
        assert_eq!(r.pick(Odia).unwrap().name(), "indic");
        assert_eq!(
            r.pick(Hindi).unwrap().name(),
            "whisper",
            "first supporting engine"
        );
        let r = r.prefer(Hindi, &["indic"]);
        assert_eq!(r.pick(Hindi).unwrap().name(), "indic");
        assert!(r.pick(Tamil).is_none());
        assert_eq!(r.supported(), vec![English, Hindi, Odia]);
        assert!(matches!(
            r.transcribe(&[], Tamil),
            Err(AsrError::Unsupported(Tamil))
        ));
        assert_eq!(r.transcribe(&[], Odia).unwrap().text, "indic");
    }

    struct FixedLid(Vec<(Language, f32)>);
    impl LanguageDetector for FixedLid {
        fn detect(&self, _: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
            Ok(self
                .0
                .iter()
                .copied()
                .filter(|(l, _)| allowed.contains(l))
                .collect())
        }
    }

    #[test]
    fn cascade_redistributes_the_indic_share() {
        use Language::*;
        let whisper = Arc::new(FixedLid(vec![(Tamil, 0.6), (English, 0.3), (Telugu, 0.1)]));
        let indic = Arc::new(FixedLid(vec![(Telugu, 0.9), (Tamil, 0.1)]));
        let c = CascadeDetector::new(whisper, indic, vec![Tamil, Telugu, Odia]);
        let p = c.detect(&[], &[English, Tamil, Telugu]).unwrap();
        assert_eq!(p[0].0, Telugu);
        assert!((p[0].1 - 0.63).abs() < 1e-6, "{p:?}");
        assert_eq!(p[1], (English, 0.3));
        assert!((p[2].1 - 0.07).abs() < 1e-6);
    }

    #[test]
    fn indic_share_beats_a_weak_english_lead() {
        use Language::*;
        // Bengali speech with English loanwords: English leads at 0.40, but the Indic
        // languages hold 0.60 together.
        let whisper = Arc::new(FixedLid(vec![(English, 0.4), (Hindi, 0.3), (Bengali, 0.3)]));
        let indic = Arc::new(FixedLid(vec![(Bengali, 0.5), (Odia, 0.3), (Hindi, 0.2)]));
        let c = CascadeDetector::new(whisper, indic, vec![Hindi, Bengali, Odia]);
        let p = c.detect(&[], &[English, Hindi, Bengali, Odia]).unwrap();
        assert_eq!(p[0].0, Bengali, "{p:?}");
        assert!((p[0].1 - 0.6).abs() < 1e-6);
        // A confident English segment is left alone.
        let whisper = Arc::new(FixedLid(vec![(English, 0.7), (Hindi, 0.3)]));
        let indic = Arc::new(FixedLid(vec![(Hindi, 1.0)]));
        let c = CascadeDetector::new(whisper, indic, vec![Hindi]);
        assert_eq!(c.detect(&[], &[English, Hindi]).unwrap()[0].0, English);
    }

    #[test]
    fn cascade_keeps_the_first_stage_when_it_is_sure_of_a_non_group_language() {
        use Language::*;
        let whisper = Arc::new(FixedLid(vec![(English, 0.95), (Hindi, 0.05)]));
        let indic = Arc::new(FixedLid(vec![(Telugu, 1.0)]));
        let c = CascadeDetector::new(whisper, indic, vec![Hindi, Telugu]);
        let p = c.detect(&[], &[English, Hindi, Telugu]).unwrap();
        assert_eq!(p, vec![(English, 0.95), (Hindi, 0.05)]);
    }
}
