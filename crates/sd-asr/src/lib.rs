//! Speech recognition for sonicdiar.
//!
//! * [`lang`]: the product's 15 languages and their mapping onto engines and benchmarks.
//! * [`metrics`]: WER/CER with a normalizer that is safe for Indic scripts.
//! * [`whisper`]: Whisper (whisper.cpp) for every product language except Odia.
//! * [`auto`]: Auto language selection with a session prior.
//! * [`bleed`]: drops mic text that is the far end leaking into the microphone.
//! * [`stream`]: the streaming policy (partials, finals, splitting, bleed hold).
//!
//! * [`indic`]: IndicConformer-600M for the 11 Indic product languages, Odia included.
//! * [`ctc_beam`]: CTC prefix beam search with language-model fusion.
//! * [`itn`]: spoken numbers written as digits (languages with verified tables).
//! * [`lm`]: word n-gram language models (Kneser-Ney) for beam-search decoding.
//! * [`router`]: picks the engine per language; cascades language detection.

pub mod auto;
pub mod bleed;
pub mod ctc_beam;
#[cfg(feature = "indic")]
pub mod indic;
pub mod itn;
pub mod lang;
pub mod lm;
pub mod metrics;
pub mod router;
pub mod runner;
pub mod speakers;
pub mod stream;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use lang::{Language, LanguageChoice};
pub use router::{LanguageDetector, Recognizer, Router};

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("model file not found: {0} (run `sdctl models fetch --group asr`)")]
    MissingModel(String),
    #[error("{0} is not supported by this engine")]
    Unsupported(Language),
    #[error("engine error: {0}")]
    Engine(String),
}

pub type Result<T> = std::result::Result<T, AsrError>;

/// A time-stamped piece of recognized text, relative to the start of the transcribed audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// The engine's probability that this segment is not speech, where available.
    pub no_speech_prob: f32,
}

/// The result of transcribing one stretch of audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The language used for decoding (detected in Auto mode).
    pub language: Language,
    /// Auto mode: probabilities over the allowed languages, highest first. Empty when fixed.
    pub language_probs: Vec<(Language, f32)>,
    pub segments: Vec<Segment>,
    /// Word timings (relative to the transcribed audio), in text order, when the engine
    /// provides them; used to split a final at speaker changes.
    pub words: Vec<Segment>,
}
