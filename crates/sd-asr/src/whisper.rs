//! Whisper (whisper.cpp 1.8.3 via whisper-rs) as a batch engine over 16 kHz mono segments.
//!
//! Language handling follows the product rules: a fixed language is passed through; `Auto`
//! runs Whisper's language identification and keeps only the allowed languages that Whisper
//! knows (all product languages but Odia), so it can never pick a language outside the list.
//!
//! Encoder context: Whisper always encodes a 30 s window (1500 frames of 20 ms), however short
//! the speech, so on a CPU a 2 s utterance costs as much as 30 s. whisper.cpp can encode fewer
//! frames (`audio_ctx`): transcription uses the utterance's length plus a margin
//! ([`WhisperConfig::ctx_margin`]); language detection, which runs on its own decoding state,
//! uses a fixed window ([`WhisperConfig::detect_ctx`]).
//!
//! A shortened window has one failure mode: not seeing the silence after the speech, the
//! decoder sometimes runs on (repeats the sentence, or adds a phrase such as "thanks for
//! watching"). Those segments are impossible on their own timestamps (past the audio, faster
//! than speech, or repeating earlier text), so such a result is decoded again with the full
//! window ([`implausible`]).

use std::path::PathBuf;
use std::sync::{Mutex, Once};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::lang::{Language, LanguageChoice};
use crate::{AsrError, Result, Segment, Transcript};

#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub model: PathBuf,
    pub threads: usize,
    pub use_gpu: bool,
    /// GPU index in whisper.cpp's device order (integrated GPUs may come first).
    pub gpu_device: i32,
    /// Beam width; 0 or 1 uses greedy decoding.
    pub beam_size: usize,
    /// Transcription encodes the utterance's frames plus this many (20 ms each); `None`
    /// encodes the full 30 s window.
    pub ctx_margin: Option<u32>,
    /// Language detection encodes this many frames (20 ms each) from the start; 0 or 1500:
    /// the full 30 s window.
    pub detect_ctx: u32,
}

impl WhisperConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
            use_gpu: false,
            gpu_device: 0,
            beam_size: 5,
            ctx_margin: None,
            detect_ctx: 0,
        }
    }
}

/// Encoder frames (20 ms) in Whisper's 30 s window.
const FULL_CTX: u32 = 1500;
/// Samples (16 kHz) per encoder frame.
const SAMPLES_PER_FRAME: usize = 320;

/// The encoder context for `samples` of audio with `margin` frames to spare, rounded up to a
/// multiple of 32 frames; 0 (whisper.cpp's default, the full window) when that reaches 30 s.
pub fn audio_ctx(samples: usize, margin: u32) -> u32 {
    let frames = samples.div_ceil(SAMPLES_PER_FRAME) as u32 + margin;
    let rounded = frames.div_ceil(32) * 32;
    if rounded >= FULL_CTX { 0 } else { rounded }
}

pub struct WhisperEngine {
    /// Decoding state, created once: creating one per call re-initializes the GPU backend
    /// (measured ~1.2 s per call on Vulkan). Calls on one engine are serialized.
    state: Mutex<WhisperState>,
    /// Language detection's own state, whose encoder context stays at `detect_ctx`
    /// (whisper.cpp keeps the last `audio_ctx` in the state, and detection has no parameter).
    detect_state: Mutex<WhisperState>,
    short_window: Mutex<ShortWindow>,
    cfg: WhisperConfig,
    /// Token ids from end-of-text up are special (timestamps, language, task tokens).
    eot: i32,
}

static LOG_HOOKS: Once = Once::new();

impl WhisperEngine {
    pub fn new(cfg: WhisperConfig) -> Result<Self> {
        // whisper.cpp logs to stderr by default; route it through `tracing` instead.
        LOG_HOOKS.call_once(whisper_rs::install_logging_hooks);
        if !cfg.model.exists() {
            return Err(AsrError::MissingModel(cfg.model.display().to_string()));
        }
        let mut params = WhisperContextParameters::default();
        params.use_gpu(cfg.use_gpu);
        params.gpu_device(cfg.gpu_device);
        let path = cfg
            .model
            .to_str()
            .ok_or_else(|| AsrError::Engine("model path is not UTF-8".into()))?;
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| AsrError::Engine(format!("loading {path}: {e}")))?;
        let state = ctx.create_state().map_err(engine)?;
        let mut detect_state = ctx.create_state().map_err(engine)?;
        if cfg.detect_ctx > 0 && cfg.detect_ctx < FULL_CTX {
            // The only way to set a state's encoder context: one short decode with it.
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(cfg.threads as i32);
            params.set_language(Some("en"));
            params.set_audio_ctx(cfg.detect_ctx as i32);
            params.set_single_segment(true);
            params.set_max_tokens(1);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_special(false);
            params.set_print_timestamps(false);
            detect_state
                .full(params, &vec![0.0; 16_000])
                .map_err(engine)?;
        }
        let eot = ctx.token_eot();
        Ok(Self {
            state: Mutex::new(state),
            detect_state: Mutex::new(detect_state),
            short_window: Mutex::new(ShortWindow::default()),
            cfg,
            eot,
        })
    }

    /// Probabilities over `allowed` (restricted to languages Whisper knows, renormalized),
    /// highest first. Uses the first 30 s of `pcm` (16 kHz mono).
    pub fn detect_language(
        &self,
        pcm: &[f32],
        allowed: &[Language],
    ) -> Result<Vec<(Language, f32)>> {
        let mut state = self.detect_state.lock().unwrap_or_else(|p| p.into_inner());
        state.pcm_to_mel(pcm, self.cfg.threads).map_err(engine)?;
        let (_, probs) = state.lang_detect(0, self.cfg.threads).map_err(engine)?;
        let mut out: Vec<(Language, f32)> = allowed
            .iter()
            .filter_map(|&l| {
                let code = l.whisper_code()?;
                let id = whisper_rs::get_lang_id(code)?;
                Some((l, *probs.get(id as usize)?))
            })
            .collect();
        let sum: f32 = out.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            out.iter_mut().for_each(|(_, p)| *p /= sum);
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }

    /// Transcribes one segment (16 kHz mono, any length; whisper.cpp windows internally).
    pub fn transcribe(
        &self,
        pcm: &[f32],
        choice: LanguageChoice,
        allowed: &[Language],
    ) -> Result<Transcript> {
        let (language, language_probs) = match choice {
            LanguageChoice::Fixed(l) => {
                if l.whisper_code().is_none() {
                    return Err(AsrError::Unsupported(l));
                }
                (l, Vec::new())
            }
            LanguageChoice::Auto => {
                let probs = self.detect_language(pcm, allowed)?;
                let best = probs.first().map(|(l, _)| *l).ok_or_else(|| {
                    AsrError::Engine("no allowed language Whisper supports".into())
                })?;
                (best, probs)
            }
        };

        let strategy = if self.cfg.beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size: self.cfg.beam_size as i32,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.cfg.threads as i32);
        params.set_language(language.whisper_code());
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_no_context(true);
        params.set_token_timestamps(true);
        // Steers Chinese to Simplified characters (Whisper otherwise mixes in Traditional).
        if language == Language::Chinese {
            params.set_initial_prompt("以下是普通话的句子。");
        }
        let mut window = self.short_window.lock().unwrap_or_else(|p| p.into_inner());
        let short = self
            .cfg
            .ctx_margin
            .filter(|_| window.enabled())
            .map(|m| audio_ctx(pcm.len(), m))
            .filter(|&c| c > 0);

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(ctx) = short {
            let mut p = params.clone();
            p.set_audio_ctx(ctx as i32);
            state.full(p, pcm).map_err(engine)?;
            let why = implausible(&state, pcm.len());
            if window.record(why.is_some()) {
                tracing::info!(
                    "whisper: {} of {} results with a shortened window needed the full one; full window from now on",
                    window.fallbacks,
                    window.tries
                );
            }
            if let Some(why) = why {
                tracing::debug!("whisper: audio_ctx {ctx}: {why}; decoding the full window");
                params.set_audio_ctx(0);
                state.full(params, pcm).map_err(engine)?;
            }
        } else {
            params.set_audio_ctx(0);
            state.full(params, pcm).map_err(engine)?;
        }
        let mut segments = Vec::new();
        let mut words: Vec<Segment> = Vec::new();
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let text = seg.to_str_lossy().map_err(engine)?.trim().to_string();
            if is_annotation(&text) {
                continue;
            }
            // Words from token timestamps: a token starting with a space starts a word
            // (scripts without spaces get one "word" per token). Bytes are joined before
            // decoding, so characters split across tokens stay intact.
            let mut word: Option<(Vec<u8>, i64, i64)> = None;
            for k in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(k) else {
                    continue;
                };
                if tok.token_id() >= self.eot {
                    continue;
                }
                let data = tok.token_data();
                let bytes = tok.to_bytes().map_err(engine)?;
                let starts_word = bytes.first() == Some(&b' ') || word.is_none();
                match (&mut word, starts_word) {
                    (Some(w), false) => {
                        w.0.extend_from_slice(bytes);
                        w.2 = data.t1;
                    }
                    _ => {
                        if let Some((b, t0, t1)) = word.take() {
                            push_word(&mut words, &b, t0, t1);
                        }
                        word = Some((bytes.to_vec(), data.t0, data.t1));
                    }
                }
            }
            if let Some((b, t0, t1)) = word.take() {
                push_word(&mut words, &b, t0, t1);
            }
            segments.push(Segment {
                start_ms: seg.start_timestamp() * 10,
                end_ms: seg.end_timestamp() * 10,
                text,
                no_speech_prob: seg.no_speech_probability(),
            });
        }
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Transcript {
            text,
            language,
            language_probs,
            segments,
            words,
        })
    }
}

/// Whisper's non-speech annotations, learned from subtitles: "[BLANK_AUDIO]", "(music)",
/// "♪ ♪", "*laughs*". A segment made only of them is not speech.
fn is_annotation(text: &str) -> bool {
    let mut close = None;
    let mut speech = false;
    for c in text.chars() {
        match close {
            Some(end) if c == end => close = None,
            Some(_) => {}
            None => match c {
                '[' => close = Some(']'),
                '(' => close = Some(')'),
                '*' => close = Some('*'),
                c if c.is_alphanumeric() => speech = true,
                _ => {}
            },
        }
    }
    !text.is_empty() && !speech
}

/// Whether shortened windows pay off for the loaded model. A result that needs the full window
/// is decoded twice, so the shortened window is given up once most results need the full one.
/// large-v3-turbo is such a model: with a shortened window its text is right but its timestamps
/// run far past the audio (24 of 25 FLEURS English clips); `small` needed it for 8 of 125.
#[derive(Debug, Default)]
struct ShortWindow {
    tries: u32,
    fallbacks: u32,
    off: bool,
}

impl ShortWindow {
    /// Results seen before deciding.
    const PROBE: u32 = 8;

    fn enabled(&self) -> bool {
        !self.off
    }

    /// Records one result; true when this turned the shortened window off.
    fn record(&mut self, fell_back: bool) -> bool {
        self.tries += 1;
        self.fallbacks += u32::from(fell_back);
        if !self.off && self.tries >= Self::PROBE && self.fallbacks * 2 > self.tries {
            self.off = true;
            return true;
        }
        false
    }
}

/// A decoded result that cannot be right: a segment ending well past the audio, text faster
/// than anyone speaks, or a segment repeating text already decoded. Seen when the encoder
/// window is shortened (the decoder runs on past the speech).
fn implausible(state: &WhisperState, samples: usize) -> Option<String> {
    let audio_ms = (samples as i64) * 1000 / 16_000;
    let mut earlier = String::new();
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let (t0, t1) = (seg.start_timestamp() * 10, seg.end_timestamp() * 10);
        let text = seg
            .to_str_lossy()
            .map(|t| t.into_owned())
            .unwrap_or_default();
        let norm: String = text
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        let chars = norm.chars().count() as i64;
        let why = if t1 > audio_ms + 500 {
            "ends after the audio"
        } else if chars >= 8 && (t1 - t0) * 40 < chars * 1000 {
            // Faster than ~40 characters a second (speech is about 15 in Latin scripts, 5 in
            // Chinese).
            "too fast"
        } else if chars >= 8 && earlier.contains(norm.as_str()) {
            "repeats earlier text"
        } else {
            earlier.push_str(&norm);
            continue;
        };
        return Some(format!(
            "segment {i} {why} ({t0}-{t1} ms of {audio_ms} ms): {}",
            text.trim()
        ));
    }
    None
}

/// Appends a word (whisper.cpp times are in 10 ms units).
fn push_word(words: &mut Vec<Segment>, bytes: &[u8], t0: i64, t1: i64) {
    let text = String::from_utf8_lossy(bytes).trim().to_string();
    if !text.is_empty() {
        words.push(Segment {
            start_ms: t0 * 10,
            end_ms: t1.max(t0) * 10,
            text,
            no_speech_prob: 0.0,
        });
    }
}

fn engine(e: whisper_rs::WhisperError) -> AsrError {
    AsrError::Engine(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annotations_are_not_speech() {
        for a in [
            "[BLANK_AUDIO]",
            "(music)",
            " [Music] ",
            "♪ ♪",
            "*laughs*",
            "[inaudible",
            "♪",
        ] {
            assert!(is_annotation(a), "{a}");
        }
        for s in [
            "",
            "Okay.",
            "Hello [laughs]",
            "Okay. Um... ♪ Eh ♪",
            "(1) first",
            "ठीक है",
        ] {
            assert!(!is_annotation(s), "{s}");
        }
    }

    #[test]
    fn the_shortened_window_is_given_up_when_most_results_need_the_full_one() {
        let mut w = ShortWindow::default();
        // A model that mostly needs the full window: off after the probe.
        for i in 0..ShortWindow::PROBE {
            assert!(w.enabled());
            assert_eq!(w.record(true), i + 1 == ShortWindow::PROBE);
        }
        assert!(!w.enabled());
        // A model that rarely does: stays on.
        let mut w = ShortWindow::default();
        for i in 0..200 {
            w.record(i % 16 == 0);
        }
        assert!(w.enabled());
    }

    #[test]
    fn audio_ctx_covers_the_audio_and_falls_back_to_the_full_window() {
        // (see `implausible` for the decode-side guard; it needs a model to test)
        // 2 s = 100 frames + 64 margin = 164 -> 192.
        assert_eq!(audio_ctx(32_000, 64), 192);
        assert_eq!(audio_ctx(1, 0), 32);
        // 28 s + margin passes 1500 frames: the full window.
        assert_eq!(audio_ctx(28 * 16_000, 128), 0);
        assert_eq!(audio_ctx(31 * 16_000, 0), 0);
    }
}
