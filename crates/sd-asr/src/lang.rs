//! The product's languages and how each maps onto the engines and the benchmark.
//!
//! Coverage was verified at the source (docs/m3-plan.md §1). Whisper has no Odia;
//! IndicConformer-600M has no English, Chinese, Spanish or French.

use std::fmt;
use std::str::FromStr;

/// A language the user can pick. `Auto` (the default) is not a language; see [`LanguageChoice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Language {
    English,
    Chinese,
    Spanish,
    French,
    Bengali,
    Gujarati,
    Hindi,
    Kannada,
    Malayalam,
    Marathi,
    Punjabi,
    Tamil,
    Telugu,
    Odia,
    Urdu,
}

/// What the user selected: a fixed language, or automatic detection (the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LanguageChoice {
    #[default]
    Auto,
    Fixed(Language),
}

impl Language {
    pub const ALL: [Language; 15] = [
        Language::English,
        Language::Chinese,
        Language::Spanish,
        Language::French,
        Language::Bengali,
        Language::Gujarati,
        Language::Hindi,
        Language::Kannada,
        Language::Malayalam,
        Language::Marathi,
        Language::Punjabi,
        Language::Tamil,
        Language::Telugu,
        Language::Odia,
        Language::Urdu,
    ];

    /// ISO 639-1 code (also the code used by Whisper and IndicConformer where supported).
    pub fn code(self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Chinese => "zh",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::Bengali => "bn",
            Language::Gujarati => "gu",
            Language::Hindi => "hi",
            Language::Kannada => "kn",
            Language::Malayalam => "ml",
            Language::Marathi => "mr",
            Language::Punjabi => "pa",
            Language::Tamil => "ta",
            Language::Telugu => "te",
            Language::Odia => "or",
            Language::Urdu => "ur",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Chinese => "Chinese",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::Bengali => "Bengali",
            Language::Gujarati => "Gujarati",
            Language::Hindi => "Hindi",
            Language::Kannada => "Kannada",
            Language::Malayalam => "Malayalam",
            Language::Marathi => "Marathi",
            Language::Punjabi => "Punjabi",
            Language::Tamil => "Tamil",
            Language::Telugu => "Telugu",
            Language::Odia => "Odia",
            Language::Urdu => "Urdu",
        }
    }

    /// Whisper's code for this language, if Whisper supports it (all but Odia).
    pub fn whisper_code(self) -> Option<&'static str> {
        (self != Language::Odia).then(|| self.code())
    }

    /// IndicConformer-600M multilingual's code, if it supports the language (the Indic ones).
    pub fn indic_code(self) -> Option<&'static str> {
        match self {
            Language::English | Language::Chinese | Language::Spanish | Language::French => None,
            other => Some(other.code()),
        }
    }

    /// The FLEURS test configuration used for evaluation.
    pub fn fleurs_config(self) -> &'static str {
        match self {
            Language::English => "en_us",
            Language::Chinese => "cmn_hans_cn",
            Language::Spanish => "es_419",
            Language::French => "fr_fr",
            Language::Bengali => "bn_in",
            Language::Gujarati => "gu_in",
            Language::Hindi => "hi_in",
            Language::Kannada => "kn_in",
            Language::Malayalam => "ml_in",
            Language::Marathi => "mr_in",
            Language::Punjabi => "pa_in",
            Language::Tamil => "ta_in",
            Language::Telugu => "te_in",
            Language::Odia => "or_in",
            Language::Urdu => "ur_pk",
        }
    }

    /// Languages written without spaces between words, scored by character error rate.
    pub fn scored_by_characters(self) -> bool {
        self == Language::Chinese
    }

    pub fn from_code(code: &str) -> Option<Language> {
        Language::ALL.into_iter().find(|l| l.code() == code)
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Language {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim().to_lowercase();
        Language::ALL
            .into_iter()
            .find(|l| l.code() == t || l.name().to_lowercase() == t || l.fleurs_config() == t)
            .ok_or_else(|| format!("unknown language '{s}'"))
    }
}

impl FromStr for LanguageChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim().eq_ignore_ascii_case("auto") {
            Ok(LanguageChoice::Auto)
        } else {
            s.parse().map(LanguageChoice::Fixed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_consistent() {
        assert_eq!(Language::ALL.len(), 15);
        for l in Language::ALL {
            assert_eq!(Language::from_code(l.code()), Some(l));
            assert_eq!(l.name().parse::<Language>(), Ok(l));
            assert_eq!(l.fleurs_config().parse::<Language>(), Ok(l));
            // Every language is covered by at least one engine.
            assert!(
                l.whisper_code().is_some() || l.indic_code().is_some(),
                "{l}"
            );
        }
        assert_eq!(Language::Odia.whisper_code(), None);
        assert_eq!(Language::English.indic_code(), None);
        assert_eq!(
            Language::ALL
                .iter()
                .filter(|l| l.whisper_code().is_some())
                .count(),
            14
        );
        assert_eq!(
            Language::ALL
                .iter()
                .filter(|l| l.indic_code().is_some())
                .count(),
            11
        );
        assert_eq!(LanguageChoice::default(), LanguageChoice::Auto);
        assert_eq!("AUTO".parse::<LanguageChoice>(), Ok(LanguageChoice::Auto));
        assert_eq!(
            "gujarati".parse::<LanguageChoice>(),
            Ok(LanguageChoice::Fixed(Language::Gujarati))
        );
    }
}
