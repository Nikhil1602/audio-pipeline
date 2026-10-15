//! Word and character error rates with a script-safe text normalizer.
//!
//! Whisper's reference `BasicTextNormalizer` replaces every Unicode mark (category M) with a
//! space. That is harmless for Latin text but destroys Indic scripts, where vowel signs and the
//! virama are marks, and it inflates Indic WER. This normalizer removes only punctuation (P)
//! and symbols (S), case-folds, and collapses whitespace; marks and letters are kept.

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

/// Lower-cases, drops bracketed annotations, replaces punctuation/symbols with spaces and
/// collapses whitespace.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0u32;
    for c in text.chars() {
        match c {
            '[' | '(' | '<' => {
                depth += 1;
                continue;
            }
            ']' | ')' | '>' => {
                depth = depth.saturating_sub(1);
                continue;
            }
            _ if depth > 0 => continue,
            _ => {}
        }
        if is_punct_or_symbol(c) {
            out.push(' ');
        } else {
            out.extend(c.to_lowercase());
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_punct_or_symbol(c: char) -> bool {
    use GeneralCategory::*;
    matches!(
        get_general_category(c),
        ConnectorPunctuation
            | DashPunctuation
            | OpenPunctuation
            | ClosePunctuation
            | InitialPunctuation
            | FinalPunctuation
            | OtherPunctuation
            | MathSymbol
            | CurrencySymbol
            | ModifierSymbol
            | OtherSymbol
    )
}

/// [`normalize`] plus folding of spelling variants that read the same, for Indic scripts:
///
/// * Unicode canonical composition (NFC): e.g. Tamil/Malayalam two-part vowel signs typed as
///   two code points, and nukta letters stored precomposed or as base + nukta, compare equal;
/// * zero-width joiners and other invisible format characters are removed;
/// * native digits (Devanagari, Bengali, ..., Arabic-Indic) become ASCII digits;
/// * the nukta is dropped and chandrabindu is folded into anusvara (writers use both forms
///   for the same pronunciation, and references are inconsistent).
///
/// This is close in spirit to "orthographically informed" WER: it forgives spelling variants
/// but never a different word. Numbers written as digits versus words still count as errors.
pub fn normalize_lenient(text: &str) -> String {
    let composed: String = text.nfc().collect();
    let mut out = String::with_capacity(composed.len());
    for c in composed.chars() {
        if get_general_category(c) == GeneralCategory::Format {
            continue;
        }
        match fold_indic(c) {
            Some(f) => out.push(f),
            None if is_nukta(c) => {}
            None => out.push(c),
        }
    }
    normalize(&out)
}

fn is_nukta(c: char) -> bool {
    matches!(
        c,
        '\u{093C}' | '\u{09BC}' | '\u{0A3C}' | '\u{0ABC}' | '\u{0B3C}' | '\u{0CBC}'
    )
}

/// Native digit to ASCII; chandrabindu to anusvara.
fn fold_indic(c: char) -> Option<char> {
    let u = c as u32;
    // Decimal digit blocks: digit zero of each script.
    for zero in [
        0x0660u32, 0x06F0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66,
    ] {
        if (zero..zero + 10).contains(&u) {
            return char::from_digit(u - zero, 10);
        }
    }
    match c {
        '\u{0901}' => Some('\u{0902}'), // Devanagari
        '\u{0981}' => Some('\u{0982}'), // Bengali
        '\u{0A01}' => Some('\u{0A02}'), // Gurmukhi
        '\u{0A81}' => Some('\u{0A82}'), // Gujarati
        '\u{0B01}' => Some('\u{0B02}'), // Odia
        '\u{0C00}' => Some('\u{0C02}'), // Telugu
        '\u{0C80}' => Some('\u{0C82}'), // Kannada
        '\u{0D01}' => Some('\u{0D02}'), // Malayalam
        _ => None,
    }
}

/// Word errors after [`normalize_lenient`].
pub fn word_errors_lenient(reference: &str, hypothesis: &str) -> ErrorCount {
    tokens_errors(
        &normalize_lenient(reference),
        &normalize_lenient(hypothesis),
    )
}

/// Character errors after [`normalize_lenient`], ignoring spaces.
pub fn char_errors_lenient(reference: &str, hypothesis: &str) -> ErrorCount {
    let r: Vec<char> = normalize_lenient(reference)
        .chars()
        .filter(|c| *c != ' ')
        .collect();
    let h: Vec<char> = normalize_lenient(hypothesis)
        .chars()
        .filter(|c| *c != ' ')
        .collect();
    ErrorCount {
        errors: edit_distance(&r, &h),
        reference: r.len(),
    }
}

fn tokens_errors(r: &str, h: &str) -> ErrorCount {
    let rw: Vec<&str> = r.split(' ').filter(|w| !w.is_empty()).collect();
    let hw: Vec<&str> = h.split(' ').filter(|w| !w.is_empty()).collect();
    ErrorCount {
        errors: edit_distance(&rw, &hw),
        reference: rw.len(),
    }
}

/// Edit (Levenshtein) distance between two token sequences.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(x != y);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Errors and reference length for one utterance, so rates can be pooled over a corpus.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ErrorCount {
    pub errors: usize,
    pub reference: usize,
}

impl ErrorCount {
    pub fn rate(&self) -> f64 {
        self.errors as f64 / self.reference.max(1) as f64
    }
    pub fn add(&mut self, other: ErrorCount) {
        self.errors += other.errors;
        self.reference += other.reference;
    }
}

/// Word errors after normalization.
pub fn word_errors(reference: &str, hypothesis: &str) -> ErrorCount {
    tokens_errors(&normalize(reference), &normalize(hypothesis))
}

/// Character errors after normalization, ignoring spaces.
pub fn char_errors(reference: &str, hypothesis: &str) -> ErrorCount {
    let r: Vec<char> = normalize(reference).chars().filter(|c| *c != ' ').collect();
    let h: Vec<char> = normalize(hypothesis)
        .chars()
        .filter(|c| *c != ' ')
        .collect();
    ErrorCount {
        errors: edit_distance(&r, &h),
        reference: r.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_keeps_indic_marks_and_drops_punctuation() {
        assert_eq!(normalize("Hello, World! (laughs)"), "hello world");
        // Devanagari vowel signs and virama survive; the danda is punctuation.
        assert_eq!(normalize("नमस्ते दुनिया।"), "नमस्ते दुनिया");
        // Gujarati, Tamil, Odia: marks kept.
        assert_eq!(normalize("કેમ છો?"), "કેમ છો");
        assert_eq!(normalize("வணக்கம்"), "வணக்கம்");
        assert_eq!(normalize("ନମସ୍କାର"), "ନମସ୍କାର");
        // Urdu (Arabic script): the Arabic question mark is punctuation.
        assert_eq!(normalize("آپ کیسے ہیں؟"), "آپ کیسے ہیں");
        assert_eq!(normalize("你好，世界。"), "你好 世界");
    }

    #[test]
    fn lenient_normalizer_folds_spelling_variants_only() {
        // Precomposed nukta letter vs base + nukta; nukta then dropped.
        assert_eq!(
            normalize_lenient("\u{0958}"),
            normalize_lenient("\u{0915}\u{093C}")
        );
        assert_eq!(normalize_lenient("ज़रूर"), normalize_lenient("जरूर"));
        // Chandrabindu vs anusvara.
        assert_eq!(normalize_lenient("हँसना"), normalize_lenient("हंसना"));
        // Zero-width joiner/non-joiner removed.
        assert_eq!(normalize_lenient("क्\u{200D}ष"), normalize_lenient("क्ष"));
        // Tamil two-part vowel sign typed as two code points.
        assert_eq!(
            normalize_lenient("\u{0B95}\u{0BC6}\u{0BBE}"),
            normalize_lenient("\u{0B95}\u{0BCA}")
        );
        // Native digits become ASCII; a different word is still an error.
        assert_eq!(normalize_lenient("१०० साल"), "100 साल");
        assert_eq!(word_errors_lenient("सौ साल", "100 साल").errors, 1);
        assert_eq!(word_errors_lenient("एंटीक", "एंटिक").errors, 1);
    }

    #[test]
    fn error_rates() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 3]), 1);
        assert_eq!(edit_distance::<u8>(&[], &[1, 2]), 2);
        let w = word_errors("the cat sat", "the cat sat down");
        assert_eq!((w.errors, w.reference), (1, 3));
        let w = word_errors("Hello, world.", "hello world");
        assert_eq!(w.errors, 0);
        let c = char_errors("你好世界", "你好时间");
        assert_eq!((c.errors, c.reference), (2, 4));
        let mut total = ErrorCount::default();
        total.add(ErrorCount {
            errors: 1,
            reference: 4,
        });
        total.add(ErrorCount {
            errors: 1,
            reference: 6,
        });
        assert!((total.rate() - 0.2).abs() < 1e-12);
    }
}
