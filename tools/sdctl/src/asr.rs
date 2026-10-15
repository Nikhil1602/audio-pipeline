//! `sdctl asr eval`: measure an ASR engine on the FLEURS subsets (tools/testdata/fetch_fleurs.py).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use sd_asr::Language;
use sd_asr::ctc_beam::BeamConfig;
use sd_asr::indic::{Decoding, IndicConfig, IndicConformer};
use sd_asr::metrics::{ErrorCount, char_errors, word_errors};

use crate::CliResult;

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Word/character error rate, language-ID accuracy and speed per language.
    Eval(EvalArgs),
    /// Rescore `asr eval --out` TSVs with stricter and more lenient text comparison.
    Score(ScoreArgs),
    /// Build a word trigram language model (Kneser-Ney) from a text corpus.
    LmBuild(LmBuildArgs),
    /// Tune IndicConformer's LM decoding (weight, word bonus, unknown-word penalty) on a
    /// FLEURS split: the encoder runs once per clip, then every setting is decoded.
    TuneLm(TuneLmArgs),
    /// Transcribe WAV files with Whisper and print segments, words and timings.
    Transcribe(TranscribeArgs),
}

#[derive(clap::Args)]
pub struct TranscribeArgs {
    /// 16 kHz mono WAV files.
    #[arg(required = true)]
    wavs: Vec<PathBuf>,
    #[arg(long, default_value = "models/ggml-small.bin")]
    model: PathBuf,
    /// auto or a language code.
    #[arg(long, default_value = "auto")]
    language: String,
    #[arg(long, default_value_t = 1)]
    beam: usize,
    #[arg(long, default_value_t = 128)]
    ctx_margin: u32,
    #[arg(long, default_value_t = 500)]
    detect_ctx: u32,
    /// The original full 30 s window (slower).
    #[arg(long)]
    full_window: bool,
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(clap::Args)]
pub struct TuneLmArgs {
    /// IndicConformer assets directory.
    #[arg(long, default_value = "models/indicconformer/assets")]
    indic: PathBuf,
    /// Directory of `<code>.bin` language models.
    #[arg(long, default_value = "models/lm")]
    lm_dir: PathBuf,
    /// FLEURS split to tune on (use dev; report on test).
    #[arg(long, default_value = "bench/data/fleurs-dev")]
    data: PathBuf,
    /// Comma-separated languages.
    #[arg(long)]
    langs: String,
    #[arg(long, default_value_t = 100)]
    limit: usize,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value = "0.1,0.2,0.3,0.5,0.7", value_delimiter = ',')]
    alphas: Vec<f32>,
    #[arg(
        long,
        default_value = "0,0.5,1,2",
        value_delimiter = ',',
        allow_hyphen_values = true
    )]
    betas: Vec<f32>,
    #[arg(
        long,
        default_value = "0,-2,-5",
        value_delimiter = ',',
        allow_hyphen_values = true
    )]
    unks: Vec<f32>,
    #[arg(long, default_value_t = 32)]
    beam: usize,
}

/// IndicConformer decoding options.
#[derive(clap::Args, Debug, Clone)]
pub struct IndicOpts {
    /// IndicConformer decoding: ctc or rnnt.
    #[arg(long, default_value = "ctc")]
    pub indic_decoding: Decoding,
    /// Directory of word language models `<code>.bin` (`sdctl asr lm-build`): CTC decoding
    /// then uses beam search with the model of the language, where one exists.
    #[arg(long)]
    pub indic_lm: Option<PathBuf>,
    /// Beam width for LM decoding.
    #[arg(long, default_value_t = 32)]
    pub lm_beam: usize,
    /// Language-model weight.
    #[arg(long, default_value_t = 0.2)]
    pub lm_alpha: f32,
    /// Word insertion bonus (nats per word).
    #[arg(long, default_value_t = 0.5)]
    pub lm_beta: f32,
    /// Penalty (nats) for words outside the LM vocabulary.
    #[arg(long, default_value_t = -5.0, allow_hyphen_values = true)]
    pub lm_unk: f32,
    /// Keep spoken numbers as words (by default they are written as digits where a
    /// verified number table exists).
    #[arg(long)]
    pub no_itn: bool,
}

#[derive(clap::Args)]
pub struct LmBuildArgs {
    /// Text corpus, one sentence per line (e.g. bench/data/indiccorp/hi.txt).
    #[arg(long)]
    corpus: PathBuf,
    /// Output model file (e.g. models/lm/hi.bin).
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 3)]
    order: usize,
    /// Vocabulary size (most frequent words).
    #[arg(long, default_value_t = 200_000)]
    vocab: usize,
    /// Drop bigrams/trigrams seen fewer times than this from the stored model.
    #[arg(long, default_value_t = 2)]
    min_count: u32,
}

#[derive(clap::Args)]
pub struct ScoreArgs {
    /// TSV files written by `sdctl asr eval --out` (or by other engines in the same format).
    #[arg(required = true)]
    tsv: Vec<PathBuf>,
    /// Write spoken numbers in the hypotheses as digits before scoring (sd_asr::itn), with
    /// this minimum value for a single number word.
    #[arg(long)]
    itn_min: Option<u64>,
    /// With --itn-min: also write each row (language, reference, hypothesis after
    /// conversion) to this TSV, for inspection.
    #[arg(long)]
    rewritten: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct EvalArgs {
    /// Whisper ggml model file.
    #[arg(long)]
    model: Option<PathBuf>,
    /// IndicConformer assets directory (models/indicconformer/assets). With --model too,
    /// Indic languages are routed to IndicConformer and Auto uses Whisper then IndicConformer.
    #[arg(long)]
    indic: Option<PathBuf>,
    #[command(flatten)]
    indic_opts: IndicOpts,
    /// Directory with one FLEURS subset per language (bench/data/fleurs).
    #[arg(long, default_value = "bench/data/fleurs")]
    data: PathBuf,
    /// Comma-separated languages (codes or names), or "all".
    #[arg(long, default_value = "all")]
    langs: String,
    /// "fixed": decode with the true language; "auto": detect among the product languages the
    /// engines support, then decode; "lid": language detection only (no decoding).
    #[arg(long, default_value = "fixed")]
    mode: String,
    /// Utterances per language.
    #[arg(long, default_value_t = 100)]
    limit: usize,
    #[arg(long)]
    threads: Option<usize>,
    /// Beam width (1 = greedy).
    #[arg(long, default_value_t = 5)]
    beam: usize,
    #[arg(long)]
    gpu: bool,
    /// GPU index (whisper.cpp device order).
    #[arg(long, default_value_t = 0)]
    gpu_device: i32,
    /// Whisper: encode each utterance plus this many 20 ms frames instead of the full 30 s.
    #[arg(long, default_value_t = 128)]
    ctx_margin: u32,
    /// Whisper: language detection encodes this many 20 ms frames (0: the full 30 s).
    #[arg(long, default_value_t = 500)]
    detect_ctx: u32,
    /// Whisper: the original full 30 s window for transcription and detection (slower).
    #[arg(long)]
    full_window: bool,
    /// Write every hypothesis to this TSV.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Transcription options shared by `sdctl process` and `sdctl record`.
#[derive(clap::Args, Debug, Clone)]
pub struct AsrArgs {
    /// Transcribe with this Whisper ggml model (English, Chinese, Spanish, French; the Indic
    /// languages too when --indic is not given).
    #[arg(long)]
    pub asr_model: Option<PathBuf>,
    /// Transcribe the Indic languages (Odia included) with IndicConformer from this assets
    /// directory (models/indicconformer/assets). Auto then identifies the language in two
    /// stages: Whisper, then IndicConformer within the Indic languages.
    #[arg(long)]
    pub indic: Option<PathBuf>,
    #[command(flatten)]
    pub indic_opts: IndicOpts,
    /// Transcription language: auto (default) or a language name/code.
    #[arg(long, default_value = "auto")]
    pub language: String,
    /// Run Whisper on the GPU (Vulkan builds).
    #[arg(long)]
    pub gpu: bool,
    #[arg(long, default_value_t = 1)]
    pub gpu_device: i32,
    /// CPU threads per engine.
    #[arg(long)]
    pub asr_threads: Option<usize>,
    /// Also print partial results.
    #[arg(long)]
    pub partials: bool,
    /// Speaker diarization of the transcripts (per source), with revisions.
    #[arg(long)]
    pub diarize: bool,
    /// WeSpeaker embedding model for --diarize.
    #[arg(long, default_value = "models/wespeaker_en_voxceleb_resnet34_LM.onnx")]
    pub diar_model: PathBuf,
    /// Overlap detection model for --diarize (marks where a second speaker talks at once);
    /// skipped if the file is missing or with --no-overlap.
    #[arg(long, default_value = "models/pyannote_segmentation_3_0.onnx")]
    pub seg_model: PathBuf,
    #[arg(long)]
    pub no_overlap: bool,
}

impl AsrArgs {
    pub fn enabled(&self) -> bool {
        self.asr_model.is_some() || self.indic.is_some()
    }

    /// The engine setup for the recorder or the offline processor; None when no engine is
    /// given.
    pub fn setup(&self) -> Result<Option<sd_engine::AsrSetup>, Box<dyn std::error::Error>> {
        if !self.enabled() {
            return Ok(None);
        }
        let engines = Engines::load(
            self.asr_model.as_deref(),
            self.indic.as_deref(),
            &self.indic_opts.options(),
            &sonicdiar::engines::WhisperOptions {
                gpu: self.gpu,
                gpu_device: self.gpu_device,
                ..Default::default()
            },
            self.asr_threads,
        )?;
        Ok(Some(sd_engine::AsrSetup {
            config: sd_asr::runner::AsrConfig {
                language: self.language.parse()?,
                allowed: engines.supported.clone(),
                partials: self.partials,
                ..Default::default()
            },
            router: engines.router,
            detector: Some(engines.detector),
            speakers: if self.diarize {
                let seg = (!self.no_overlap && self.seg_model.exists())
                    .then_some(self.seg_model.as_path());
                if seg.is_none() && !self.no_overlap {
                    eprintln!(
                        "note: {} not found, overlapped speech is not marked",
                        self.seg_model.display()
                    );
                }
                Some(sd_engine::diarization::speaker_factory(
                    &self.diar_model,
                    seg,
                    2,
                    sd_diar::online::OnlineConfig::default(),
                )?)
            } else {
                None
            },
        }))
    }
}

pub use sonicdiar::engines::Engines;

impl IndicOpts {
    /// The engine options these flags describe.
    pub fn options(&self) -> sonicdiar::engines::IndicOptions {
        sonicdiar::engines::IndicOptions {
            decoding: self.indic_decoding,
            lm: self.indic_lm.clone(),
            lm_beam: self.lm_beam,
            lm_alpha: self.lm_alpha,
            lm_beta: self.lm_beta,
            lm_unk: self.lm_unk,
            numbers_as_digits: !self.no_itn,
        }
    }
}

pub fn run(cmd: Cmd) -> CliResult {
    match cmd {
        Cmd::Eval(a) => eval(a),
        Cmd::Score(a) => score(a),
        Cmd::LmBuild(a) => lm_build(a),
        Cmd::TuneLm(a) => tune_lm(a),
        Cmd::Transcribe(a) => transcribe(a),
    }
}

fn transcribe(a: TranscribeArgs) -> CliResult {
    let mut cfg = sd_asr::whisper::WhisperConfig::new(&a.model);
    cfg.beam_size = a.beam;
    if !a.full_window {
        cfg.ctx_margin = Some(a.ctx_margin);
        cfg.detect_ctx = a.detect_ctx;
    }
    if let Some(t) = a.threads {
        cfg.threads = t;
    }
    let w = sd_asr::whisper::WhisperEngine::new(cfg)?;
    let choice: sd_asr::LanguageChoice = a.language.parse()?;
    for wav in &a.wavs {
        let pcm = read_wav_16k(wav)?;
        let t0 = Instant::now();
        let t = w.transcribe(&pcm, choice, &sd_asr::Language::ALL)?;
        println!(
            "{} ({:.2} s audio, {:.2} s compute) [{}]",
            wav.display(),
            pcm.len() as f64 / 16_000.0,
            t0.elapsed().as_secs_f64(),
            t.language.code()
        );
        for s in &t.segments {
            println!(
                "  {:>6.2}-{:>6.2}  {}",
                s.start_ms as f64 / 1000.0,
                s.end_ms as f64 / 1000.0,
                s.text
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn read_wav_16k(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let r = hound::WavReader::open(path)?;
    let spec = r.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!("{}: expected 16 kHz mono", path.display()).into());
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => r
            .into_samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => r.into_samples::<f32>().collect::<Result<_, _>>()?,
    })
}

fn eval(a: EvalArgs) -> CliResult {
    let langs: Vec<Language> = if a.langs == "all" {
        Language::ALL.to_vec()
    } else {
        a.langs
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?
    };
    let (auto, lid_only) = match a.mode.as_str() {
        "fixed" => (false, false),
        "auto" => (true, false),
        "lid" => (true, true),
        other => return Err(format!("--mode must be fixed, auto or lid, not {other}").into()),
    };
    let t_load = Instant::now();
    let Engines {
        router,
        detector,
        supported,
        stages,
    } = Engines::load(
        a.model.as_deref(),
        a.indic.as_deref(),
        &a.indic_opts.options(),
        &{
            let o = sonicdiar::engines::WhisperOptions {
                beam: a.beam,
                gpu: a.gpu,
                gpu_device: a.gpu_device,
                ctx_margin: Some(a.ctx_margin),
                detect_ctx: a.detect_ctx,
            };
            if a.full_window { o.full_window() } else { o }
        },
        a.threads,
    )?;
    if a.model.is_some() {
        println!("whisper.cpp: {}", whisper_rs::print_system_info().trim());
    }
    println!(
        "engines loaded in {:.1} s; mode {}; languages {}",
        t_load.elapsed().as_secs_f64(),
        a.mode,
        supported
            .iter()
            .map(|l| l.code())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut tsv = match &a.out {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            let mut f = std::fs::File::create(p)?;
            use std::io::Write;
            write!(
                f,
                "language\tfile\tdetected\terrors\tref_len\treference\thypothesis"
            )?;
            if auto && stages.len() > 1 {
                for (name, _) in &stages {
                    write!(f, "\tprobs_{name}")?;
                }
            }
            writeln!(f)?;
            Some(f)
        }
        None => None,
    };

    println!(
        "{:<10} {:>4} {:>8} {:>8} {:>7}  {}",
        "language",
        "n",
        "metric",
        "rate",
        "rtf",
        if auto {
            "LID accuracy / confusions"
        } else {
            ""
        }
    );
    let (mut all_audio, mut all_time) = (0.0f64, 0.0f64);
    for lang in langs {
        if !supported.contains(&lang) {
            println!(
                "{:<10}  not supported by the engines (skipped)",
                lang.name()
            );
            continue;
        }
        let dir = a.data.join(lang.fleurs_config());
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).map_err(|e| {
            format!(
                "{}: {e} (run tools/testdata/fetch_fleurs.py)",
                dir.display()
            )
        })?;
        let mut total = ErrorCount::default();
        let (mut audio_s, mut time_s) = (0.0f64, 0.0f64);
        let (mut n, mut lid_ok) = (0usize, 0usize);
        // What Auto should answer: Urdu speech is written as Hindi by design.
        let expected =
            sd_asr::auto::same_speech_target(lang, &sd_asr::auto::SAME_SPEECH, &supported);
        let mut confusions: BTreeMap<Language, usize> = BTreeMap::new();
        for line in manifest.lines().skip(1).take(a.limit) {
            let mut cols = line.split('\t');
            let (Some(file), Some(reference)) = (cols.next(), cols.next()) else {
                continue;
            };
            let pcm = read_wav_16k(&dir.join(file))?;
            let t = Instant::now();
            // Auto: each stage's own probabilities, for offline analysis of the combination.
            let mut stage_probs = Vec::new();
            if auto && tsv.is_some() && stages.len() > 1 {
                for (_, d) in &stages {
                    stage_probs.push(d.detect(&pcm, &supported)?);
                }
            }
            let language = if auto {
                let probs = detector.detect(&pcm, &supported)?;
                let top = probs.first().map_or(Language::English, |p| p.0);
                sd_asr::auto::same_speech_target(top, &sd_asr::auto::SAME_SPEECH, &supported)
            } else {
                lang
            };
            let tr = if lid_only {
                sd_asr::Transcript {
                    text: String::new(),
                    language,
                    language_probs: Vec::new(),
                    segments: Vec::new(),
                    words: Vec::new(),
                }
            } else {
                router.transcribe(&pcm, language)?
            };
            time_s += t.elapsed().as_secs_f64();
            audio_s += pcm.len() as f64 / 16_000.0;
            let e = if lang.scored_by_characters() {
                char_errors(reference, &tr.text)
            } else {
                word_errors(reference, &tr.text)
            };
            total.add(e);
            n += 1;
            if tr.language == expected {
                lid_ok += 1;
            } else {
                *confusions.entry(tr.language).or_default() += 1;
            }
            if let Some(f) = &mut tsv {
                use std::io::Write;
                write!(
                    f,
                    "{}\t{file}\t{}\t{}\t{}\t{reference}\t{}",
                    lang.code(),
                    tr.language.code(),
                    e.errors,
                    e.reference,
                    tr.text.replace(['\t', '\n'], " ")
                )?;
                for probs in &stage_probs {
                    let cells: Vec<String> = probs
                        .iter()
                        .map(|(l, p)| format!("{}:{p:.4}", l.code()))
                        .collect();
                    write!(f, "\t{}", cells.join(","))?;
                }
                writeln!(f)?;
            }
        }
        all_audio += audio_s;
        all_time += time_s;
        let lid = if auto {
            let mut c: Vec<_> = confusions.into_iter().collect();
            c.sort_by(|x, y| y.1.cmp(&x.1));
            let top: Vec<String> = c
                .iter()
                .take(3)
                .map(|(l, k)| format!("{}×{k}", l.code()))
                .collect();
            format!(
                "{:>5.1}%  {}",
                100.0 * lid_ok as f64 / n.max(1) as f64,
                top.join(" ")
            )
        } else {
            String::new()
        };
        println!(
            "{:<10} {:>4} {:>8} {:>7.1}% {:>7.3}  {lid}",
            lang.name(),
            n,
            if lang.scored_by_characters() {
                "CER"
            } else {
                "WER"
            },
            100.0 * total.rate(),
            time_s / audio_s.max(1e-9),
        );
    }
    println!(
        "overall real-time factor {:.3} ({:.0} s of audio in {:.0} s)",
        all_time / all_audio.max(1e-9),
        all_audio,
        all_time
    );
    Ok(ExitCode::SUCCESS)
}

/// Per language: strict WER (as `eval`), lenient WER, lenient CER, and lenient WER on the
/// utterances whose reference has no digits (engines write numbers as words or digits, and
/// FLEURS references mix both, so this isolates the number-format effect).
fn score(a: ScoreArgs) -> CliResult {
    use sd_asr::metrics::{char_errors_lenient, word_errors_lenient};
    use std::io::Write;
    let mut rewritten = match &a.rewritten {
        Some(p) => {
            let mut f = std::fs::File::create(p)?;
            writeln!(f, "language\treference\thypothesis_before\thypothesis")?;
            Some(f)
        }
        None => None,
    };
    for path in &a.tsv {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
        let col = |name: &str| {
            header
                .iter()
                .position(|h| *h == name)
                .ok_or_else(|| format!("{}: no column {name}", path.display()))
        };
        let (c_lang, c_ref, c_hyp) = (col("language")?, col("reference")?, col("hypothesis")?);
        #[derive(Default)]
        struct Acc {
            n: usize,
            strict: ErrorCount,
            words: ErrorCount,
            chars: ErrorCount,
            no_digits: ErrorCount,
            n_no_digits: usize,
            /// Numbers written as digits by --itn-min, and how many of them occur in the
            /// reference.
            digits_written: usize,
            digits_in_reference: usize,
        }
        let mut per: BTreeMap<String, Acc> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for line in lines {
            let cells: Vec<&str> = line.split('\t').collect();
            let (Some(lang), Some(r), Some(h)) =
                (cells.get(c_lang), cells.get(c_ref), cells.get(c_hyp))
            else {
                continue;
            };
            let language: Option<Language> = lang.parse().ok();
            if !order.iter().any(|o| o == lang) {
                order.push(lang.to_string());
            }
            let itn_h;
            let h = match (
                a.itn_min,
                language.and_then(sd_asr::itn::NumberWords::for_language),
            ) {
                (Some(min_value), Some(words)) => {
                    itn_h = words.rewrite(h, &sd_asr::itn::ItnConfig { min_value });
                    itn_h.as_str()
                }
                _ => *h,
            };
            if let Some(f) = &mut rewritten {
                writeln!(f, "{lang}\t{r}\t{}\t{h}", cells[c_hyp])?;
            }
            let acc = per.entry(lang.to_string()).or_default();
            if a.itn_min.is_some() {
                let numbers = |t: &str| -> Vec<String> {
                    sd_asr::metrics::normalize_lenient(t)
                        .split(' ')
                        .filter(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_digit()))
                        .map(str::to_string)
                        .collect()
                };
                let before = numbers(cells[c_hyp]);
                let reference = numbers(r);
                for n in numbers(h) {
                    if !before.contains(&n) {
                        acc.digits_written += 1;
                        acc.digits_in_reference += usize::from(reference.contains(&n));
                    }
                }
            }
            acc.n += 1;
            let by_chars = language.is_some_and(|l| l.scored_by_characters());
            acc.strict.add(if by_chars {
                char_errors(r, h)
            } else {
                word_errors(r, h)
            });
            acc.words.add(word_errors_lenient(r, h));
            acc.chars.add(char_errors_lenient(r, h));
            if !sd_asr::metrics::normalize_lenient(r)
                .chars()
                .any(|c| c.is_ascii_digit())
            {
                acc.n_no_digits += 1;
                acc.no_digits.add(word_errors_lenient(r, h));
            }
        }
        println!("{}", path.display());
        println!(
            "  {:<10} {:>4} {:>11} {:>9} {:>9} {:>17}{}",
            "language",
            "n",
            "strict",
            "lenient",
            "CER",
            "lenient, no digits",
            if a.itn_min.is_some() {
                "   numbers written (in reference)"
            } else {
                ""
            }
        );
        for lang in &order {
            let acc = &per[lang];
            let name = lang
                .parse::<Language>()
                .map_or(lang.clone(), |l| l.name().to_string());
            println!(
                "  {:<10} {:>4} {:>10.1}% {:>8.1}% {:>8.1}% {:>9.1}% (n={})",
                name,
                acc.n,
                100.0 * acc.strict.rate(),
                100.0 * acc.words.rate(),
                100.0 * acc.chars.rate(),
                100.0 * acc.no_digits.rate(),
                acc.n_no_digits
            );
            if a.itn_min.is_some() {
                println!(
                    "{:>88}",
                    format!(
                        "{} ({}, {:.0}%)",
                        acc.digits_written,
                        acc.digits_in_reference,
                        100.0 * acc.digits_in_reference as f64 / acc.digits_written.max(1) as f64
                    )
                );
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn lm_build(a: LmBuildArgs) -> CliResult {
    let t = Instant::now();
    let cfg = sd_asr::lm::BuildConfig {
        order: a.order,
        vocab: a.vocab,
        min_count: a.min_count,
    };
    let (lm, rep) = sd_asr::lm::build(&a.corpus, &cfg)?;
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    lm.save(&a.out)?;
    let bytes = std::fs::metadata(&a.out)?.len();
    println!(
        "{}: {} sentences, {} words ({} distinct, {:.2}% outside the {}-word vocabulary)",
        a.corpus.display(),
        rep.sentences,
        rep.tokens,
        rep.distinct_words,
        100.0 * rep.oov_rate,
        a.vocab
    );
    println!(
        "  n-grams stored per order {:?}; discounts {:?}",
        rep.sizes,
        rep.discounts
            .iter()
            .map(|d| format!("{:.2}/{:.2}/{:.2}", d[0], d[1], d[2]))
            .collect::<Vec<_>>()
    );
    println!(
        "  wrote {} ({:.0} MB) in {:.0} s",
        a.out.display(),
        bytes as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

/// Lenient WER of `hyps` against `refs`.
fn corpus_wer(refs: &[String], hyps: &[String]) -> f64 {
    let mut total = ErrorCount::default();
    for (r, h) in refs.iter().zip(hyps) {
        total.add(sd_asr::metrics::word_errors_lenient(r, h));
    }
    total.rate()
}

fn tune_lm(a: TuneLmArgs) -> CliResult {
    let mut cfg = IndicConfig::new(&a.indic);
    if let Some(t) = a.threads {
        cfg.threads = t;
    }
    let model = IndicConformer::new(cfg)?;
    for code in a.langs.split(',') {
        let lang: Language = code.parse()?;
        let lm_path = a
            .lm_dir
            .join(format!("{}.bin", lang.indic_code().unwrap_or(code)));
        let lm = sd_asr::lm::NgramLm::load(&lm_path)?;
        let dir = a.data.join(lang.fleurs_config());
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv"))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let t = Instant::now();
        let mut frames = Vec::new();
        let mut refs = Vec::new();
        for line in manifest.lines().skip(1).take(a.limit) {
            let mut cols = line.split('\t');
            let (Some(file), Some(reference)) = (cols.next(), cols.next()) else {
                continue;
            };
            let pcm = read_wav_16k(&dir.join(file))?;
            frames.push(model.language_frames(&pcm, lang)?);
            refs.push(reference.to_string());
        }
        let greedy: Vec<String> = frames
            .iter()
            .map(|f| model.decode_frames(f, None, None).text)
            .collect();
        println!(
            "{} ({} clips from {}, encoded in {:.0} s): greedy WER {:.1}%",
            lang.name(),
            frames.len(),
            dir.display(),
            t.elapsed().as_secs_f64(),
            100.0 * corpus_wer(&refs, &greedy)
        );
        let mut grid = Vec::new();
        for &alpha in &a.alphas {
            for &beta in &a.betas {
                for &unk in &a.unks {
                    grid.push((alpha, beta, unk));
                }
            }
        }
        // Settings are independent: decode them on all cores.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::sync::Mutex::new(Vec::<(f64, f32, f32, f32)>::new());
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(alpha, beta, unk)) = grid.get(i) else {
                            break;
                        };
                        let bc = BeamConfig {
                            beam: a.beam,
                            alpha,
                            beta,
                            unk_penalty: unk,
                            ..Default::default()
                        };
                        let hyps: Vec<String> = frames
                            .iter()
                            .map(|f| model.decode_frames(f, Some(&lm), Some(&bc)).text)
                            .collect();
                        let w = corpus_wer(&refs, &hyps);
                        results
                            .lock()
                            .expect("not poisoned")
                            .push((w, alpha, beta, unk));
                    }
                });
            }
        });
        let mut results = results.into_inner().expect("not poisoned");
        results.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (w, alpha, beta, unk) in results.iter().take(5) {
            println!(
                "  WER {:5.1}%  --lm-alpha {alpha} --lm-beta {beta} --lm-unk {unk}",
                100.0 * w
            );
        }
        if let Some(worst) = results.last() {
            println!("  (worst {:.1}%)", 100.0 * worst.0);
        }
    }
    Ok(ExitCode::SUCCESS)
}
