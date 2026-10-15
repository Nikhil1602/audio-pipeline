//! Speech recognition engines behind one router: Whisper (English, Chinese, Spanish, French,
//! and the Indian languages when IndicConformer is not loaded) and IndicConformer (11 Indian
//! languages), with the matching language detector for Auto.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sd_asr::ctc_beam::BeamConfig;
use sd_asr::indic::{Decoding, IndicConfig, IndicConformer, IndicLm};
use sd_asr::router::CascadeDetector;
use sd_asr::whisper::{WhisperConfig, WhisperEngine};
use sd_asr::{Language, LanguageDetector, Recognizer, Router};

use crate::{Error, Result};

/// IndicConformer decoding options.
#[derive(Debug, Clone)]
pub struct IndicOptions {
    pub decoding: Decoding,
    /// Directory of word language models `<code>.bin`: CTC decoding then uses beam search with
    /// the model of the language, where one exists.
    pub lm: Option<PathBuf>,
    pub lm_beam: usize,
    pub lm_alpha: f32,
    pub lm_beta: f32,
    pub lm_unk: f32,
    /// Write spoken numbers as digits where a verified number table exists.
    pub numbers_as_digits: bool,
}

impl Default for IndicOptions {
    fn default() -> Self {
        Self {
            decoding: Decoding::Ctc,
            lm: None,
            lm_beam: 32,
            lm_alpha: 0.2,
            lm_beta: 0.5,
            lm_unk: -5.0,
            numbers_as_digits: true,
        }
    }
}

/// Whisper decoding options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperOptions {
    /// Beam size (1: greedy, the fastest).
    pub beam: usize,
    pub gpu: bool,
    pub gpu_device: i32,
    /// Transcription encodes the utterance plus this many 20 ms frames instead of the full
    /// 30 s window; `None`: the full window. Models whose results mostly need the full window
    /// anyway (large-v3-turbo) stop shortening it by themselves (see `sd_asr::whisper`).
    pub ctx_margin: Option<u32>,
    /// Language detection encodes this many 20 ms frames; 0: the full 30 s.
    pub detect_ctx: u32,
}

impl WhisperOptions {
    /// Whisper's original encoder window: the full 30 s for transcription and detection.
    pub fn full_window(self) -> Self {
        Self {
            ctx_margin: None,
            detect_ctx: 0,
            ..self
        }
    }
}

/// The encoder windows were measured on FLEURS (CPU, Auto, docs/asr-speed.md): 2.6 s of margin
/// and a 10 s detection window made `small` 2.1x faster and large-v3-turbo 1.2-1.7x faster,
/// with the same accuracy and language identification as the full window.
impl Default for WhisperOptions {
    fn default() -> Self {
        Self {
            beam: 1,
            gpu: false,
            gpu_device: 0,
            ctx_margin: Some(128),
            detect_ctx: 500,
        }
    }
}

/// The loaded engines.
pub struct Engines {
    pub router: Router,
    pub detector: Arc<dyn LanguageDetector>,
    pub supported: Vec<Language>,
    /// The individual language detectors behind `detector`, by engine name.
    pub stages: Vec<(&'static str, Arc<dyn LanguageDetector>)>,
}

impl Engines {
    /// Loads Whisper (from a ggml model file) and/or IndicConformer (from its assets
    /// directory). Indic languages go to IndicConformer when it is loaded; with both engines,
    /// Auto runs Whisper's language detection, then IndicConformer's within the Indic
    /// languages.
    pub fn load(
        whisper: Option<&Path>,
        indic: Option<&Path>,
        indic_options: &IndicOptions,
        whisper_options: &WhisperOptions,
        threads: Option<usize>,
    ) -> Result<Self> {
        let asr = |e: sd_asr::AsrError| Error::Engine(e.to_string());
        let mut engines: Vec<Arc<dyn Recognizer>> = Vec::new();
        let mut detectors: Vec<Arc<dyn LanguageDetector>> = Vec::new();
        let mut indic_langs = Vec::new();
        let mut names = Vec::new();
        if let Some(model) = whisper {
            let mut cfg = WhisperConfig::new(model);
            if let Some(t) = threads {
                cfg.threads = t;
            }
            cfg.beam_size = whisper_options.beam;
            cfg.use_gpu = whisper_options.gpu;
            cfg.gpu_device = whisper_options.gpu_device;
            cfg.ctx_margin = whisper_options.ctx_margin;
            cfg.detect_ctx = whisper_options.detect_ctx;
            let w = Arc::new(WhisperEngine::new(cfg).map_err(asr)?);
            engines.push(w.clone());
            detectors.push(w);
            names.push("whisper");
        }
        if let Some(dir) = indic {
            let mut cfg = IndicConfig::new(dir);
            if let Some(t) = threads {
                cfg.threads = t;
            }
            cfg.decoding = indic_options.decoding;
            if !indic_options.numbers_as_digits {
                cfg.itn = None;
            }
            cfg.lm = indic_options.lm.as_ref().map(|dir| IndicLm {
                dir: dir.clone(),
                beam: BeamConfig {
                    beam: indic_options.lm_beam,
                    alpha: indic_options.lm_alpha,
                    beta: indic_options.lm_beta,
                    unk_penalty: indic_options.lm_unk,
                    ..Default::default()
                },
            });
            let m = Arc::new(IndicConformer::new(cfg).map_err(asr)?);
            indic_langs = m.languages();
            engines.push(m.clone());
            detectors.push(m);
            names.push("indicconformer");
        }
        let mut router = Router::new(engines);
        for l in &indic_langs {
            router = router.prefer(*l, &["indicconformer"]);
        }
        let detector: Arc<dyn LanguageDetector> = match detectors.as_slice() {
            [one] => one.clone(),
            [first, second] => Arc::new(CascadeDetector::new(
                first.clone(),
                second.clone(),
                indic_langs,
            )),
            _ => {
                return Err(Error::Config(
                    "transcription needs a Whisper model and/or the IndicConformer directory"
                        .into(),
                ));
            }
        };
        let supported = router.supported();
        let stages = names.into_iter().zip(detectors).collect();
        Ok(Self {
            router,
            detector,
            supported,
            stages,
        })
    }
}
