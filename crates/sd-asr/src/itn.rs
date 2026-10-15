//! Writes spoken numbers as digits ("दो हज़ार सात" → "2007", "இரண்டாயிரத்து ஐந்து" → "2005").
//!
//! Tables (`data/numbers/<code>.tsv`, built by tools/testdata/build_number_tables.py from
//! AI4Bharat indic-numtowords, CLDR, NeMo-text-processing and Wikidata; provenance is in
//! each file) list number phrases of two kinds:
//! * complete numbers (0–999 with their colloquial variants, and round thousands, lakhs and
//!   crores);
//! * combining forms that must be followed by a smaller number: Tamil "இரண்டாயிரத்து"
//!   (2000 + …), Hindi "उन्नीस सौ" (1900 + …), "तीन लाख" (300000 + …).
//!
//! A number is the longest chain of combining forms with decreasing units, ending in a
//! complete number. It is written as digits only when it is clearly a number: two or more
//! words, or a value of at least [`ItnConfig::min_value`]. A lone "एक" (also "a") stays a
//! word. Words are matched after [`crate::metrics::normalize_lenient`], so spelling variants
//! (nukta, chandrabindu, zero-width joiners) match.
//!
//! Which languages it runs on is decided by measurement ([`ENABLED`]).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::lang::Language;
use crate::metrics::normalize_lenient;

/// Languages whose table lowered WER and CER on both FLEURS dev and test, and whose written
/// numbers were rarely wrong (docs/m3-report.md). Kannada is left out: its generated forms
/// contain errors that produce wrong values (the "thousand-of" form used as a round
/// thousand), so it keeps spoken numbers until its table is checked by a native speaker.
pub const ENABLED: [Language; 10] = [
    Language::Hindi,
    Language::Bengali,
    Language::Gujarati,
    Language::Marathi,
    Language::Punjabi,
    Language::Tamil,
    Language::Telugu,
    Language::Malayalam,
    Language::Odia,
    Language::Urdu,
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItnConfig {
    /// A single number word is written as digits only from this value up.
    pub min_value: u64,
}

impl Default for ItnConfig {
    fn default() -> Self {
        Self { min_value: 10 }
    }
}

/// One language's number phrases.
#[derive(Debug, Default)]
pub struct NumberWords {
    full: HashMap<String, u64>,
    /// Combining form -> (value, unit): must be followed by a number smaller than unit.
    prefix: HashMap<String, (u64, u64)>,
    max_words: usize,
    /// Every single word that occurs in a phrase (to recognise runs of number words).
    vocab: std::collections::HashSet<String>,
}

/// Matching form of a word or phrase: [`normalize_lenient`], plus Urdu noon ghunna (ں) read
/// as noon (ن), the two spellings of the nasal in "پانچ" / "پاںچ".
fn key(text: &str) -> String {
    normalize_lenient(text).replace('\u{06BA}', "\u{0646}")
}

fn table(language: Language) -> Option<&'static str> {
    Some(match language {
        Language::Hindi => include_str!("../data/numbers/hi.tsv"),
        Language::Bengali => include_str!("../data/numbers/bn.tsv"),
        Language::Gujarati => include_str!("../data/numbers/gu.tsv"),
        Language::Marathi => include_str!("../data/numbers/mr.tsv"),
        Language::Punjabi => include_str!("../data/numbers/pa.tsv"),
        Language::Tamil => include_str!("../data/numbers/ta.tsv"),
        Language::Telugu => include_str!("../data/numbers/te.tsv"),
        Language::Kannada => include_str!("../data/numbers/kn.tsv"),
        Language::Malayalam => include_str!("../data/numbers/ml.tsv"),
        Language::Odia => include_str!("../data/numbers/or.tsv"),
        Language::Urdu => include_str!("../data/numbers/ur.tsv"),
        _ => return None,
    })
}

impl NumberWords {
    /// Parses a table: "phrase<TAB>value<TAB>kind<TAB>unit", kind `n` (complete number) or
    /// `p` (combining form); `#` lines are comments.
    pub fn parse(tsv: &str) -> Self {
        let mut t = NumberWords::default();
        for line in tsv
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let cols: Vec<&str> = line.split('\t').collect();
            let (Some(phrase), Some(Ok(value))) =
                (cols.first(), cols.get(1).map(|v| v.trim().parse::<u64>()))
            else {
                continue;
            };
            let key = key(phrase);
            if key.is_empty() {
                continue;
            }
            t.max_words = t.max_words.max(key.split(' ').count());
            t.vocab.extend(key.split(' ').map(str::to_string));
            match (
                cols.get(2).copied(),
                cols.get(3).map(|u| u.trim().parse::<u64>()),
            ) {
                (Some("p"), Some(Ok(unit))) if unit > 0 => {
                    t.prefix.entry(key).or_insert((value, unit));
                }
                _ => {
                    t.full.entry(key).or_insert(value);
                }
            }
        }
        t
    }

    /// The table for `language`, if one exists (whether or not it is [`ENABLED`]).
    pub fn for_language(language: Language) -> Option<&'static NumberWords> {
        static TABLES: OnceLock<HashMap<Language, NumberWords>> = OnceLock::new();
        TABLES
            .get_or_init(|| {
                Language::ALL
                    .into_iter()
                    .filter_map(|l| table(l).map(|t| (l, NumberWords::parse(t))))
                    .collect()
            })
            .get(&language)
    }

    /// The table for `language` if numbers are written as digits in that language.
    pub fn enabled(language: Language) -> Option<&'static NumberWords> {
        ENABLED
            .contains(&language)
            .then(|| Self::for_language(language))
            .flatten()
    }

    /// Longest number starting at `words[i]` with a value below `limit`: (value, words used).
    fn parse_at(&self, words: &[String], i: usize, limit: u64) -> Option<(u64, usize)> {
        let mut best: Option<(u64, usize)> = None;
        let mut better = |c: (u64, usize)| {
            if best.is_none_or(|b| c.1 > b.1) {
                best = Some(c);
            }
        };
        let longest = self.max_words.min(words.len().saturating_sub(i));
        for len in 1..=longest {
            if words[i..i + len].iter().any(|w| w.is_empty()) {
                break;
            }
            let key = words[i..i + len].join(" ");
            if let Some(&(value, unit)) = self.prefix.get(&key)
                && value < limit
                && let Some((rest, used)) = self.parse_at(words, i + len, unit)
            {
                better((value + rest, len + used));
            }
            if let Some(&value) = self.full.get(&key)
                && value < limit
            {
                better((value, len));
            }
        }
        best
    }

    /// A word that is, or starts with, a number word (inflected forms such as Telugu
    /// "తొమ్మిదిన", "on nine", count too).
    fn numeric_word(&self, w: &str, min_prefix_chars: usize) -> bool {
        self.vocab.contains(w)
            || self
                .vocab
                .iter()
                .any(|v| v.chars().count() >= min_prefix_chars && w.starts_with(v.as_str()))
    }

    /// Rewrites the numbers in `text` (words separated by spaces) as digits.
    ///
    /// A run of adjacent number words is converted only if it reads as one number: when it
    /// would come out as several numbers, or as a number next to leftover number words
    /// ("17 100 55" for a year form the table lacks), the whole run stays as words.
    pub fn rewrite(&self, text: &str, cfg: &ItnConfig) -> String {
        let words: Vec<&str> = text.split(' ').filter(|w| !w.is_empty()).collect();
        self.rewrite_spans(&words, cfg)
            .into_iter()
            .map(|(t, _, _)| t)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// As [`NumberWords::rewrite`] over a word list: each output token with the range of
    /// input words it replaces (one word, or the words of a number).
    pub fn rewrite_spans(&self, words: &[&str], cfg: &ItnConfig) -> Vec<(String, usize, usize)> {
        let norm: Vec<String> = words.iter().map(|w| key(w)).collect();
        // Segments: (first word, word count, value if converted).
        let mut segs: Vec<(usize, usize, Option<u64>)> = Vec::new();
        let mut i = 0;
        while i < words.len() {
            match self.parse_at(&norm, i, u64::MAX) {
                Some((value, used)) if used >= 2 || value >= cfg.min_value => {
                    segs.push((i, used, Some(value)));
                    i += used;
                }
                _ => {
                    segs.push((i, 1, None));
                    i += 1;
                }
            }
        }
        // A converted round number followed by an inflected number word ("दोन हजार
        // आठपासून", "since 2008") is unfinished: shorter prefixes count there.
        let numeric_at = |k: usize| {
            let s = &segs[k];
            if s.2.is_some() {
                return true;
            }
            let after_round = k > 0 && segs[k - 1].2.is_some_and(|v| v % 100 == 0);
            s.1 == 1 && self.numeric_word(&norm[s.0], if after_round { 2 } else { 3 })
        };
        let mut out: Vec<(String, usize, usize)> = Vec::with_capacity(words.len());
        let mut j = 0;
        while j < segs.len() {
            let mut end = j + 1;
            if numeric_at(j) {
                while end < segs.len() && numeric_at(end) {
                    end += 1;
                }
            }
            let run = &segs[j..end];
            let keep_words = run.len() > 1 && run.iter().any(|s| s.2.is_some());
            for &(start, len, value) in run {
                match value {
                    Some(v) if !keep_words => out.push((v.to_string(), start, start + len)),
                    _ => out.extend((start..start + len).map(|k| (words[k].to_string(), k, k + 1))),
                }
            }
            j = end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(l: Language) -> &'static NumberWords {
        NumberWords::for_language(l).expect("table")
    }

    #[test]
    fn writes_hindi_numbers_as_digits() {
        let c = ItnConfig::default();
        let hi = table(Language::Hindi);
        assert_eq!(hi.rewrite("दो हज़ार सात सालों में", &c), "2007 सालों में");
        assert_eq!(hi.rewrite("उन्नीस सौ अस्सी", &c), "1980");
        assert_eq!(hi.rewrite("सौ साल", &c), "100 साल");
        assert_eq!(hi.rewrite("तीन लाख पचास हज़ार लोग", &c), "350000 लोग");
        assert_eq!(hi.rewrite("एक करोड़ दो लाख", &c), "10200000");
        assert_eq!(hi.rewrite("दो सौ पचास", &c), "250");
        // Spelling variant without nukta.
        assert_eq!(hi.rewrite("दो हजार", &c), "2000");
    }

    #[test]
    fn keeps_words_that_are_not_clearly_numbers() {
        let c = ItnConfig::default();
        let hi = table(Language::Hindi);
        // "एक" alone is also the article "a".
        assert_eq!(hi.rewrite("एक दिन", &c), "एक दिन");
        // Two units in a row are not one number ("two-three days").
        assert_eq!(hi.rewrite("दो तीन दिन", &c), "दो तीन दिन");
        assert_eq!(hi.rewrite("पच्चीस साल", &c), "25 साल");
    }

    #[test]
    fn runs_that_are_not_one_number_stay_words() {
        let c = ItnConfig::default();
        let hi = table(Language::Hindi);
        // Two numbers side by side ("ten-twenty people") are ambiguous.
        assert_eq!(hi.rewrite("दस बीस लोग", &c), "दस बीस लोग");
        // An inflected last word ("since 2008") must not leave a wrong "2000".
        let mr = table(Language::Marathi);
        assert_eq!(mr.rewrite("दोन हजार आठपासून", &c), "दोन हजार आठपासून");
        assert_eq!(mr.rewrite("दोन हजार दहा मध्ये", &c), "2010 मध्ये");
        // Urdu year with the nasal spelled either way.
        let ur = table(Language::Urdu);
        assert_eq!(ur.rewrite("انیس سو پچانوے", &c), "1995");
        assert_eq!(ur.rewrite("انیس ہزار پانچ سو", &c), "19500");
    }

    #[test]
    fn combining_forms_in_dravidian_languages() {
        let c = ItnConfig::default();
        let ta = table(Language::Tamil);
        assert_eq!(ta.rewrite("இரண்டாயிரத்து ஐந்து", &c), "2005");
        assert_eq!(ta.rewrite("மூணு லட்சத்து அம்பதாயிரம்", &c), "350000");
        assert_eq!(ta.rewrite("இருபத்தைந்து பேர்", &c), "25 பேர்");
        // A generator error mapped Kannada "ಇನ್ನೂ" ("still") to several numbers; the
        // table build drops such entries.
        assert_eq!(table(Language::Kannada).rewrite("ಇನ್ನೂ", &c), "ಇನ್ನೂ");
    }

    #[test]
    fn every_indic_language_has_a_table_and_enabled_ones_are_measured() {
        for l in Language::ALL
            .into_iter()
            .filter(|l| l.indic_code().is_some())
        {
            assert!(NumberWords::for_language(l).is_some(), "{l}");
        }
        assert!(NumberWords::for_language(Language::English).is_none());
        assert!(NumberWords::enabled(Language::Hindi).is_some());
        assert!(NumberWords::enabled(Language::Kannada).is_none());
    }
}
