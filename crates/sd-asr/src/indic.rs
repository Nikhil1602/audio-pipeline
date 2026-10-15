//! IndicConformer-600M multilingual (AI4Bharat, MIT) on ONNX Runtime, CTC decoding.
//!
//! Protocol (read from the model's own files, `models/indicconformer/assets`):
//! * Features: NeMo `AudioToMelSpectrogramPreprocessor` as exported in `preprocessor.ts`:
//!   pre-emphasis 0.97, reflect padding of 256, STFT n_fft 512 / hop 160 / Hann window 400
//!   (centred in 512), power spectrum, 80 mel bands, `ln(x + 2^-24)`, then per-band mean and
//!   (n−1) standard deviation normalization over the utterance. The window and mel matrix are
//!   read from the TorchScript archive, so they are the model's exact constants.
//! * `encoder.onnx`: `audio_signal` [B, 80, T] f32 + `length` [B] i64 → `outputs`
//!   [B, 1024, T/8] + `encoded_lengths`. One encoder frame is 80 ms.
//! * `ctc_decoder.onnx`: `encoder_output` → `logprobs` [B, T/8, 5633]: 22 languages × 256
//!   tokens, plus one shared blank (5632). `language_masks.json` gives each language's 257
//!   positions, `vocab.json` the matching token strings (SentencePiece, `▁` = word start).
//!
//! * RNN-T (optional, [`Decoding::Rnnt`]): `joint_enc.onnx` maps encoder frames to the joint
//!   space; `rnnt_decoder.onnx` is the prediction network (2-layer LSTM, 640, state in/out);
//!   `joint_pred.onnx`, `joint_pre_net.onnx` and `joint_post_net_<lang>.onnx` complete the
//!   joint network, the last giving 257 logits in the language's own vocabulary (256 = blank,
//!   also the start symbol). Greedy decoding, at most 10 symbols per frame, as the reference.
//!
//! CTC decoding masks the output to a language's 257 positions and takes the greedy path.
//! Because the vocabulary is per language, the unmasked output also identifies the language:
//! [`IndicConformer::detect`] sums each language's token probability over the utterance.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array3};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use realfft::RealFftPlanner;

use crate::lang::Language;
use crate::{AsrError, Result, Segment, Transcript};

const HOP: usize = 160;
const N_FFT: usize = 512;
const WIN: usize = 400;
const N_MELS: usize = 80;
const BINS: usize = N_FFT / 2 + 1;
/// Encoder output frame duration (8× subsampling of the 10 ms feature hop).
pub const FRAME_MS: i64 = 80;
/// A pause at least this long between tokens starts a new [`Segment`].
const SEGMENT_GAP_FRAMES: usize = 8;

fn engine_err(e: impl std::fmt::Display) -> AsrError {
    AsrError::Engine(e.to_string())
}

/// How the encoder output is turned into text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Decoding {
    /// Greedy CTC: one pass over the frames, cheapest.
    #[default]
    Ctc,
    /// Greedy RNN-T: conditions each token on the previous ones; needs the RNN-T files.
    Rnnt,
}

impl std::str::FromStr for Decoding {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "ctc" => Ok(Decoding::Ctc),
            "rnnt" => Ok(Decoding::Rnnt),
            other => Err(format!("decoding must be ctc or rnnt, not {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndicConfig {
    /// The model's `assets` directory.
    pub dir: PathBuf,
    /// Threads for the encoder.
    pub threads: usize,
    pub decoding: Decoding,
    /// CTC beam search with word language models `<dir>/<code>.bin` (see [`crate::lm`]).
    /// Languages without a model file use greedy decoding.
    pub lm: Option<IndicLm>,
    /// Write spoken numbers as digits where a verified number table exists ([`crate::itn`]).
    pub itn: Option<crate::itn::ItnConfig>,
}

#[derive(Debug, Clone)]
pub struct IndicLm {
    pub dir: PathBuf,
    pub beam: crate::ctc_beam::BeamConfig,
}

impl IndicConfig {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 8);
        Self {
            dir: dir.into(),
            threads,
            decoding: Decoding::default(),
            lm: None,
            itn: Some(crate::itn::ItnConfig::default()),
        }
    }
}

/// Per-frame log-probabilities over one language's tokens (see
/// [`IndicConformer::language_frames`]).
pub struct LangFrames {
    pub language: Language,
    pub frames: usize,
    rows: Vec<f32>,
}

/// One language's slice of the joint vocabulary.
struct LangVocab {
    /// Positions in the joint output, in vocabulary order; the last is the blank.
    positions: Vec<usize>,
    tokens: Vec<String>,
}

/// CTC log-probabilities over the joint vocabulary for one utterance, [frames × vocab], and
/// for RNN-T the encoder output projected into the joint space, [frames × JOINT].
struct Logits {
    frames: usize,
    vocab: usize,
    data: Vec<f32>,
    joint_enc: Option<Vec<f32>>,
}

const JOINT: usize = 640;
const PRED_LAYERS: usize = 2;
/// RNN-T: blank and start symbol in each language's own 257-token vocabulary.
const RNNT_BLANK: usize = 256;
const RNNT_MAX_SYMBOLS: usize = 10;

struct Rnnt {
    joint_enc: Session,
    decoder: Session,
    joint_pred: Session,
    pre_net: Session,
    post_net: HashMap<Language, Session>,
}

impl Logits {
    fn frame(&self, t: usize) -> &[f32] {
        &self.data[t * self.vocab..(t + 1) * self.vocab]
    }
}

struct Sessions {
    encoder: Session,
    ctc: Session,
    rnnt: Option<Rnnt>,
}

pub struct IndicConformer {
    sessions: Mutex<Sessions>,
    decoding: Decoding,
    lm: Option<IndicLm>,
    itn: Option<crate::itn::ItnConfig>,
    /// Language models loaded on first use (None: no model file for that language).
    lms: Mutex<HashMap<Language, Option<Arc<crate::lm::NgramLm>>>>,
    langs: HashMap<Language, LangVocab>,
    blank: usize,
    window: Vec<f32>,
    /// Mel filterbank, [BINS × N_MELS] row-major (as stored by the model).
    mel: Vec<f32>,
    /// The last utterance's output, so detecting and then transcribing the same audio runs
    /// the encoder once.
    last: Mutex<Option<(u64, Arc<Logits>)>>,
}

fn read_constant(archive: &Path, name: &str, floats: usize) -> Result<Vec<f32>> {
    let file = std::fs::File::open(archive)
        .map_err(|_| AsrError::MissingModel(archive.display().to_string()))?;
    let mut zip = zip::ZipArchive::new(file).map_err(engine_err)?;
    let mut entry = zip
        .by_name(&format!("preprocessor/constants/{name}"))
        .map_err(|e| engine_err(format!("{}: constant {name}: {e}", archive.display())))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).map_err(engine_err)?;
    if bytes.len() != floats * 4 {
        return Err(engine_err(format!(
            "{}: constant {name} has {} bytes, expected {}",
            archive.display(),
            bytes.len(),
            floats * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| AsrError::MissingModel(path.display().to_string()))?;
    serde_json::from_str(&text).map_err(|e| engine_err(format!("{}: {e}", path.display())))
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    if !path.is_file() {
        return Err(AsrError::MissingModel(path.display().to_string()));
    }
    let env = sd_vad::runtime().map_err(engine_err)?;
    let b = |e: ort::Error<ort::session::builder::SessionBuilder>| engine_err(e);
    Session::builder(env)
        .map_err(engine_err)?
        .with_intra_threads(threads)
        .map_err(b)?
        .with_inter_threads(1)
        .map_err(b)?
        // Input lengths vary per utterance: no memory pattern, and no arena that would keep
        // the largest utterance's buffers for the rest of the session.
        .with_memory_pattern(false)
        .map_err(b)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(b)?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(b)?
        .commit_from_file(path)
        .map_err(engine_err)
}

impl IndicConformer {
    pub fn new(cfg: IndicConfig) -> Result<Self> {
        let dir = &cfg.dir;
        let archive = dir.join("preprocessor.ts");
        let window = read_constant(&archive, "3", WIN)?;
        let mel = read_constant(&archive, "4", BINS * N_MELS)?;

        let masks = read_json(&dir.join("language_masks.json"))?;
        let vocab = read_json(&dir.join("vocab.json"))?;
        let masks = masks
            .as_object()
            .ok_or_else(|| engine_err("language_masks.json: expected an object"))?;
        let mut langs = HashMap::new();
        let mut blank = None;
        for lang in Language::ALL {
            let Some(code) = lang.indic_code() else {
                continue;
            };
            let (Some(mask), Some(tokens)) = (
                masks.get(code).and_then(|m| m.as_array()),
                vocab.get(code).and_then(|v| v.as_array()),
            ) else {
                continue;
            };
            let positions: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|(_, b)| b.as_bool() == Some(true))
                .map(|(i, _)| i)
                .collect();
            let tokens: Vec<String> = tokens
                .iter()
                .map(|t| t.as_str().unwrap_or_default().to_string())
                .collect();
            if positions.len() != tokens.len() || positions.is_empty() {
                return Err(engine_err(format!(
                    "{code}: {} mask positions but {} tokens",
                    positions.len(),
                    tokens.len()
                )));
            }
            let b = *positions.last().expect("non-empty");
            if *blank.get_or_insert(b) != b {
                return Err(engine_err(format!("{code}: blank at {b}, not shared")));
            }
            langs.insert(lang, LangVocab { positions, tokens });
        }
        let blank = blank.ok_or_else(|| engine_err("no supported language in the model"))?;

        let rnnt = match cfg.decoding {
            Decoding::Ctc => None,
            Decoding::Rnnt => {
                let mut post_net = HashMap::new();
                for lang in langs.keys() {
                    let code = lang.indic_code().expect("Indic language");
                    post_net.insert(
                        *lang,
                        session(&dir.join(format!("joint_post_net_{code}.onnx")), 1)?,
                    );
                }
                Some(Rnnt {
                    joint_enc: session(&dir.join("joint_enc.onnx"), cfg.threads)?,
                    decoder: session(&dir.join("rnnt_decoder.onnx"), 1)?,
                    joint_pred: session(&dir.join("joint_pred.onnx"), 1)?,
                    pre_net: session(&dir.join("joint_pre_net.onnx"), 1)?,
                    post_net,
                })
            }
        };
        let sessions = Sessions {
            encoder: session(&dir.join("encoder.onnx"), cfg.threads)?,
            ctc: session(&dir.join("ctc_decoder.onnx"), 1)?,
            rnnt,
        };
        Ok(Self {
            sessions: Mutex::new(sessions),
            decoding: cfg.decoding,
            lm: cfg.lm,
            itn: cfg.itn,
            lms: Mutex::new(HashMap::new()),
            langs,
            blank,
            window,
            mel,
            last: Mutex::new(None),
        })
    }

    /// Log-mel features, [N_MELS × frames] row-major, normalized per band.
    fn features(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let n = pcm.len();
        let frames = n / HOP + 1;
        // Pre-emphasis.
        let mut x = Vec::with_capacity(n);
        let mut prev = 0.0f32;
        for &s in pcm {
            x.push(s - 0.97 * prev);
            prev = s;
        }
        // Reflect padding of N_FFT/2 on both sides (torch "reflect").
        let pad = N_FFT / 2;
        let reflect = |i: isize| -> f32 {
            if n == 1 {
                return x[0];
            }
            let period = 2 * (n as isize - 1);
            let mut j = i.rem_euclid(period);
            if j >= n as isize {
                j = period - j;
            }
            x[j as usize]
        };
        let padded: Vec<f32> = (0..n + 2 * pad)
            .map(|i| reflect(i as isize - pad as isize))
            .collect();

        let mut window = vec![0.0f32; N_FFT];
        let off = (N_FFT - WIN) / 2;
        window[off..off + WIN].copy_from_slice(&self.window);

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let mut frame = fft.make_input_vec();
        let mut spec = fft.make_output_vec();
        let mut scratch = fft.make_scratch_vec();
        let mut power = vec![0.0f32; BINS];
        let mut feats = vec![0.0f32; N_MELS * frames];
        for t in 0..frames {
            let start = t * HOP;
            for (k, v) in frame.iter_mut().enumerate() {
                *v = padded[start + k] * window[k];
            }
            fft.process_with_scratch(&mut frame, &mut spec, &mut scratch)
                .expect("buffer sizes from the plan");
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.norm_sqr();
            }
            for m in 0..N_MELS {
                let mut acc = 0.0f32;
                for (k, p) in power.iter().enumerate() {
                    acc += p * self.mel[k * N_MELS + m];
                }
                feats[m * frames + t] = (acc + 5.960_464_5e-8).ln();
            }
        }
        // Per-band normalization (unbiased variance, floor 2^-24, epsilon 1e-5).
        if frames > 1 {
            for band in feats.chunks_exact_mut(frames) {
                let mean = band.iter().map(|&v| v as f64).sum::<f64>() / frames as f64;
                let var = band.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>()
                    / (frames - 1) as f64;
                let std = var.max(5.960_464_477_539_063e-8).sqrt();
                for v in band.iter_mut() {
                    *v = ((*v as f64 - mean) / (std + 1e-5)) as f32;
                }
            }
        }
        (feats, frames)
    }

    fn key(pcm: &[f32]) -> u64 {
        // FNV-1a over the samples' bits: cheap next to the encoder, and exact.
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ pcm.len() as u64;
        for s in pcm {
            h ^= s.to_bits() as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn logits(&self, pcm: &[f32]) -> Result<Arc<Logits>> {
        let key = Self::key(pcm);
        if let Some((k, l)) = &*self.last.lock().expect("not poisoned")
            && *k == key
        {
            return Ok(l.clone());
        }
        let (feats, frames) = self.features(pcm);
        let audio = Array3::from_shape_vec((1, N_MELS, frames), feats).map_err(engine_err)?;
        let length = Array1::from_elem(1, frames as i64);
        let mut guard = self.sessions.lock().expect("not poisoned");
        let s = &mut *guard;
        let enc = s
            .encoder
            .run(ort::inputs![
                "audio_signal" => TensorRef::from_array_view(audio.view()).map_err(engine_err)?,
                "length" => TensorRef::from_array_view(length.view()).map_err(engine_err)?,
            ])
            .map_err(engine_err)?;
        let (shape, encoded) = enc["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(engine_err)?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let encoded = ndarray::ArrayView3::from_shape((shape[0], shape[1], shape[2]), encoded)
            .map_err(engine_err)?;
        let ctc = s
            .ctc
            .run(ort::inputs![
                "encoder_output" => TensorRef::from_array_view(encoded).map_err(engine_err)?,
            ])
            .map_err(engine_err)?;
        let (shape, lp) = ctc["logprobs"]
            .try_extract_tensor::<f32>()
            .map_err(engine_err)?;
        let (frames, vocab, data) = (shape[1] as usize, shape[2] as usize, lp.to_vec());
        drop(ctc);
        let joint_enc = match &mut s.rnnt {
            Some(r) => {
                // [1, 1024, T] -> [1, T, 1024]
                let bt = encoded
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_owned();
                let out = r
                    .joint_enc
                    .run(ort::inputs![
                        "input" => TensorRef::from_array_view(bt.view()).map_err(engine_err)?,
                    ])
                    .map_err(engine_err)?;
                let (_, je) = out["output"]
                    .try_extract_tensor::<f32>()
                    .map_err(engine_err)?;
                Some(je.to_vec())
            }
            None => None,
        };
        let logits = Arc::new(Logits {
            frames,
            vocab,
            data,
            joint_enc,
        });
        drop(enc);
        drop(guard);
        *self.last.lock().expect("not poisoned") = Some((key, logits.clone()));
        Ok(logits)
    }

    /// Languages this model can transcribe, among the product languages.
    pub fn languages(&self) -> Vec<Language> {
        Language::ALL
            .into_iter()
            .filter(|l| self.langs.contains_key(l))
            .collect()
    }

    /// Greedy decoding (CTC or RNN-T, per [`IndicConfig::decoding`]) of 16 kHz mono audio in
    /// `language`.
    pub fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript> {
        let vocab = self
            .langs
            .get(&language)
            .ok_or(AsrError::Unsupported(language))?;
        let logits = self.logits(pcm)?;
        let path = match (self.decoding, &self.lm) {
            (Decoding::Ctc, Some(cfg)) => match self.language_model(language, cfg)? {
                Some(lm) => Self::beam_path(&logits, vocab, Some(&lm), &cfg.beam),
                None => Self::ctc_path(&logits, vocab),
            },
            (Decoding::Ctc, None) => Self::ctc_path(&logits, vocab),
            (Decoding::Rnnt, _) => self.rnnt_path(&logits, language)?,
        };
        Ok(Self::to_transcript(&path, vocab, language, self.itn))
    }

    fn language_model(
        &self,
        language: Language,
        cfg: &IndicLm,
    ) -> Result<Option<Arc<crate::lm::NgramLm>>> {
        let mut lms = self.lms.lock().expect("not poisoned");
        if let Some(lm) = lms.get(&language) {
            return Ok(lm.clone());
        }
        let code = language.indic_code().expect("Indic language");
        let path = cfg.dir.join(format!("{code}.bin"));
        let lm = if path.is_file() {
            Some(Arc::new(
                crate::lm::NgramLm::load(&path).map_err(engine_err)?,
            ))
        } else {
            None
        };
        lms.insert(language, lm.clone());
        Ok(lm)
    }

    /// CTC prefix beam search over the language's own vocabulary with LM fusion.
    fn beam_path(
        logits: &Logits,
        vocab: &LangVocab,
        lm: Option<&crate::lm::NgramLm>,
        cfg: &crate::ctc_beam::BeamConfig,
    ) -> Vec<(usize, usize, usize)> {
        let rows = Self::language_rows(logits, vocab);
        let n = vocab.positions.len();
        crate::ctc_beam::decode(
            logits.frames,
            |t| &rows[t * n..(t + 1) * n],
            n - 1,
            &vocab.tokens,
            lm,
            cfg,
        )
    }

    /// Each frame renormalized over the language's tokens (as the reference's masked
    /// log-softmax does), [frames × 257].
    fn language_rows(logits: &Logits, vocab: &LangVocab) -> Vec<f32> {
        let n = vocab.positions.len();
        let mut rows = vec![0f32; logits.frames * n];
        for t in 0..logits.frames {
            let f = logits.frame(t);
            let row = &mut rows[t * n..(t + 1) * n];
            for (r, &p) in row.iter_mut().zip(&vocab.positions) {
                *r = f[p];
            }
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            row.iter_mut().for_each(|v| *v -= lse);
        }
        rows
    }

    /// The encoder's per-frame log-probabilities over `language`'s tokens, for decoding many
    /// times with different settings (decoder tuning) without re-running the encoder.
    pub fn language_frames(&self, pcm: &[f32], language: Language) -> Result<LangFrames> {
        let vocab = self
            .langs
            .get(&language)
            .ok_or(AsrError::Unsupported(language))?;
        let logits = self.logits(pcm)?;
        Ok(LangFrames {
            language,
            frames: logits.frames,
            rows: Self::language_rows(&logits, vocab),
        })
    }

    /// Decodes [`LangFrames`]: beam search with `lm` and `cfg`, or greedy when `cfg` is None.
    pub fn decode_frames(
        &self,
        f: &LangFrames,
        lm: Option<&crate::lm::NgramLm>,
        cfg: Option<&crate::ctc_beam::BeamConfig>,
    ) -> Transcript {
        let vocab = &self.langs[&f.language];
        let n = vocab.positions.len();
        let path = match cfg {
            Some(cfg) => crate::ctc_beam::decode(
                f.frames,
                |t| &f.rows[t * n..(t + 1) * n],
                n - 1,
                &vocab.tokens,
                lm,
                cfg,
            ),
            None => {
                let mut path: Vec<(usize, usize, usize)> = Vec::new();
                let mut prev = n - 1;
                for t in 0..f.frames {
                    let row = &f.rows[t * n..(t + 1) * n];
                    let best = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map_or(n - 1, |(i, _)| i);
                    if best != n - 1 && best != prev {
                        path.push((best, t, t + 1));
                    }
                    prev = best;
                }
                path
            }
        };
        Self::to_transcript(&path, vocab, f.language, None)
    }

    /// Greedy CTC path: (token, first frame, frame after last).
    fn ctc_path(logits: &Logits, vocab: &LangVocab) -> Vec<(usize, usize, usize)> {
        let blank = vocab.positions.len() - 1;
        let mut path: Vec<(usize, usize, usize)> = Vec::new();
        let mut prev = blank;
        for t in 0..logits.frames {
            let f = logits.frame(t);
            let (best, _) = vocab.positions.iter().enumerate().fold(
                (blank, f32::NEG_INFINITY),
                |acc, (i, &p)| {
                    if f[p] > acc.1 { (i, f[p]) } else { acc }
                },
            );
            if best != blank {
                if best == prev {
                    if let Some(last) = path.last_mut() {
                        last.2 = t + 1;
                    }
                } else {
                    path.push((best, t, t + 1));
                }
            }
            prev = best;
        }
        path
    }

    /// Greedy RNN-T path: (token, frame, frame + 1). The prediction network runs only when a
    /// token is emitted; on blank its output is reused (the reference recomputes it with the
    /// same inputs, which gives the same result).
    fn rnnt_path(&self, logits: &Logits, language: Language) -> Result<Vec<(usize, usize, usize)>> {
        let f_all = logits.joint_enc.as_ref().ok_or_else(|| {
            engine_err("RNN-T decoding was not enabled when the model was loaded")
        })?;
        let mut guard = self.sessions.lock().expect("not poisoned");
        let r = guard
            .rnnt
            .as_mut()
            .expect("joint_enc implies RNN-T sessions");
        let post = r
            .post_net
            .get_mut(&language)
            .ok_or(AsrError::Unsupported(language))?;
        let mut h = Array3::<f32>::zeros((PRED_LAYERS, 1, JOINT));
        let mut c = Array3::<f32>::zeros((PRED_LAYERS, 1, JOINT));
        let mut last = RNNT_BLANK as i32;
        // Prediction-network output for `last` (joint space), with its new LSTM state.
        let mut pred: Option<(Vec<f32>, Array3<f32>, Array3<f32>)> = None;
        let mut path = Vec::new();
        let mut joint = ndarray::Array3::<f32>::zeros((1, 1, JOINT));
        for t in 0..logits.frames {
            let f = &f_all[t * JOINT..(t + 1) * JOINT];
            for _ in 0..RNNT_MAX_SYMBOLS {
                if pred.is_none() {
                    let targets = ndarray::Array2::from_elem((1, 1), last);
                    let target_len = Array1::from_elem(1, 1i32);
                    let out = r
                        .decoder
                        .run(ort::inputs![
                            "targets" => TensorRef::from_array_view(targets.view()).map_err(engine_err)?,
                            "target_length" => TensorRef::from_array_view(target_len.view()).map_err(engine_err)?,
                            "states.1" => TensorRef::from_array_view(h.view()).map_err(engine_err)?,
                            "onnx::Slice_3" => TensorRef::from_array_view(c.view()).map_err(engine_err)?,
                        ])
                        .map_err(engine_err)?;
                    // outputs [1, 640, 1] is laid out like [1, 1, 640].
                    let (_, g) = out["outputs"]
                        .try_extract_tensor::<f32>()
                        .map_err(engine_err)?;
                    let g = ndarray::Array3::from_shape_vec((1, 1, JOINT), g.to_vec())
                        .map_err(engine_err)?;
                    let (_, h2) = out["states"]
                        .try_extract_tensor::<f32>()
                        .map_err(engine_err)?;
                    let h2 = Array3::from_shape_vec((PRED_LAYERS, 1, JOINT), h2.to_vec())
                        .map_err(engine_err)?;
                    let (_, c2) = out["162"].try_extract_tensor::<f32>().map_err(engine_err)?;
                    let c2 = Array3::from_shape_vec((PRED_LAYERS, 1, JOINT), c2.to_vec())
                        .map_err(engine_err)?;
                    drop(out);
                    let out = r
                        .joint_pred
                        .run(ort::inputs![
                            "input" => TensorRef::from_array_view(g.view()).map_err(engine_err)?,
                        ])
                        .map_err(engine_err)?;
                    let (_, gp) = out["output"]
                        .try_extract_tensor::<f32>()
                        .map_err(engine_err)?;
                    pred = Some((gp.to_vec(), h2, c2));
                }
                let (gp, _, _) = pred.as_ref().expect("set above");
                for ((j, a), b) in joint.iter_mut().zip(f).zip(gp) {
                    *j = a + b;
                }
                let out = r
                    .pre_net
                    .run(ort::inputs![
                        "input" => TensorRef::from_array_view(joint.view()).map_err(engine_err)?,
                    ])
                    .map_err(engine_err)?;
                let (_, hidden) = out["output"]
                    .try_extract_tensor::<f32>()
                    .map_err(engine_err)?;
                let hidden = ndarray::Array3::from_shape_vec((1, 1, JOINT), hidden.to_vec())
                    .map_err(engine_err)?;
                drop(out);
                let out = post
                    .run(ort::inputs![
                        "input" => TensorRef::from_array_view(hidden.view()).map_err(engine_err)?,
                    ])
                    .map_err(engine_err)?;
                let (_, z) = out["output"]
                    .try_extract_tensor::<f32>()
                    .map_err(engine_err)?;
                let k = z
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(RNNT_BLANK, |(i, _)| i);
                drop(out);
                if k == RNNT_BLANK {
                    break;
                }
                path.push((k, t, t + 1));
                let (_, h2, c2) = pred.take().expect("set above");
                h = h2;
                c = c2;
                last = k as i32;
            }
        }
        Ok(path)
    }

    /// Words from the token path (a `▁` piece starts a word), spoken numbers written as
    /// digits when `itn` is set and the language has an enabled table, and segments split at
    /// pauses of [`SEGMENT_GAP_FRAMES`] between words.
    fn to_transcript(
        path: &[(usize, usize, usize)],
        vocab: &LangVocab,
        language: Language,
        itn: Option<crate::itn::ItnConfig>,
    ) -> Transcript {
        // (text, first frame, frame after last)
        let mut raw: Vec<(String, usize, usize)> = Vec::new();
        for &(tok, a, b) in path {
            let piece = &vocab.tokens[tok];
            if piece == "<unk>" {
                continue;
            }
            match (piece.strip_prefix('▁'), raw.last_mut()) {
                (None, Some(w)) => {
                    w.0.push_str(piece);
                    w.2 = b;
                }
                (Some(rest), _) => raw.push((rest.to_string(), a, b)),
                (None, None) => raw.push((piece.clone(), a, b)),
            }
        }
        raw.retain(|w| !w.0.trim().is_empty());
        let words: Vec<(String, usize, usize)> =
            match (itn, crate::itn::NumberWords::enabled(language)) {
                (Some(cfg), Some(table)) => {
                    let texts: Vec<&str> = raw.iter().map(|w| w.0.as_str()).collect();
                    table
                        .rewrite_spans(&texts, &cfg)
                        .into_iter()
                        .map(|(t, s, e)| (t, raw[s].1, raw[e - 1].2))
                        .collect()
                }
                _ => raw,
            };
        let mut segments: Vec<Segment> = Vec::new();
        let mut current: Option<(usize, usize, Vec<&str>)> = None;
        for (text, a, b) in &words {
            match &mut current {
                Some((_, end, ws)) if a.saturating_sub(*end) < SEGMENT_GAP_FRAMES => {
                    ws.push(text);
                    *end = *b;
                }
                _ => {
                    if let Some((s, e, ws)) = current.take() {
                        segments.push(make_segment(s, e, ws.join(" ")));
                    }
                    current = Some((*a, *b, vec![text.as_str()]));
                }
            }
        }
        if let Some((s, e, ws)) = current.take() {
            segments.push(make_segment(s, e, ws.join(" ")));
        }
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Transcript {
            text,
            language,
            language_probs: Vec::new(),
            segments,
            words: words
                .into_iter()
                .map(|(t, a, b)| make_segment(a, b, t))
                .collect(),
        }
    }

    /// Language identification among `allowed` (those this model supports): each language's
    /// share of the non-blank probability mass over the utterance, highest first.
    pub fn detect(&self, pcm: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
        let candidates: Vec<(Language, &LangVocab)> = allowed
            .iter()
            .filter_map(|l| self.langs.get(l).map(|v| (*l, v)))
            .collect();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let logits = self.logits(pcm)?;
        let mut mass = vec![0.0f64; candidates.len()];
        for t in 0..logits.frames {
            let f = logits.frame(t);
            let max = f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f64 = f.iter().map(|&v| ((v - max) as f64).exp()).sum();
            for (m, (_, v)) in mass.iter_mut().zip(&candidates) {
                let tokens = &v.positions[..v.positions.len() - 1];
                *m += tokens
                    .iter()
                    .map(|&p| ((f[p] - max) as f64).exp())
                    .sum::<f64>()
                    / total;
            }
        }
        let sum: f64 = mass.iter().sum();
        let mut out: Vec<(Language, f32)> = candidates
            .iter()
            .zip(&mass)
            .map(|((l, _), m)| {
                let p = if sum > 0.0 {
                    m / sum
                } else {
                    1.0 / candidates.len() as f64
                };
                (*l, p as f32)
            })
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }

    /// The shared blank's position in the joint vocabulary.
    pub fn blank(&self) -> usize {
        self.blank
    }
}

fn make_segment(a: usize, b: usize, text: String) -> Segment {
    Segment {
        start_ms: a as i64 * FRAME_MS,
        end_ms: b as i64 * FRAME_MS,
        text,
        no_speech_prob: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        sd_vad::set_runtime_library(models.join(sd_vad::RUNTIME_LIBRARY));
        let d = models.join("indicconformer/assets");
        d.join("encoder.onnx").is_file().then_some(d)
    }

    #[test]
    fn features_have_one_frame_per_hop_and_unit_variance() {
        let Some(dir) = model_dir() else {
            eprintln!("IndicConformer not downloaded; skipped");
            return;
        };
        let m = IndicConformer::new(IndicConfig::new(dir)).unwrap();
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.07).sin() * 0.1 + ((i * 7919) % 13) as f32 * 1e-3)
            .collect();
        let (f, frames) = m.features(&pcm);
        assert_eq!(frames, 101);
        let band = &f[40 * frames..41 * frames];
        let mean: f32 = band.iter().sum::<f32>() / frames as f32;
        let var: f32 = band.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (frames - 1) as f32;
        assert!(mean.abs() < 1e-4, "{mean}");
        assert!((var - 1.0).abs() < 1e-2, "{var}");
    }

    #[test]
    fn vocabulary_layout_matches_the_model() {
        let Some(dir) = model_dir() else {
            return;
        };
        let m = IndicConformer::new(IndicConfig::new(dir)).unwrap();
        assert_eq!(
            m.languages().len(),
            11,
            "every Indic product language, Odia included"
        );
        assert_eq!(m.blank(), 5632);
        for v in m.langs.values() {
            assert_eq!(v.positions.len(), 257);
        }
    }
}
