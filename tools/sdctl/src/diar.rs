//! `sdctl diar eval`: diarization error rate on AMI (tools/testdata/fetch_ami.py).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use sd_diar::cluster::min_cluster_size;
use sd_diar::der::{DerResult, Turn, der, read_rttm, write_rttm};
use sd_diar::embed::Embedder;
use sd_diar::pipeline::{DiarConfig, dendrogram, embed_regions, turns};

use crate::CliResult;

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// DER per meeting for a sweep of clustering thresholds (and with the true speaker count).
    Eval(EvalArgs),
    /// Live diarization replayed utterance by utterance: DER of the labels as first emitted
    /// and after re-clustering, and how much speech was relabelled.
    OnlineEval(OnlineArgs),
}

#[derive(clap::Args)]
pub struct OnlineArgs {
    #[arg(long, default_value = "models/wespeaker_en_voxceleb_resnet34_LM.onnx")]
    model: PathBuf,
    #[arg(long, default_value = "bench/data/ami")]
    data: PathBuf,
    /// Distance for joining an existing speaker live.
    #[arg(long, value_delimiter = ',', default_value = "0.9,1.0,1.1")]
    assign: Vec<f32>,
    /// Re-clustering interval (seconds of audio).
    #[arg(long, default_value_t = 60.0)]
    recluster_every: f32,
    /// Small clusters are absorbed only into a large cluster closer than this (inf = always).
    #[arg(long, default_value_t = f32::INFINITY)]
    absorb_max: f32,
    /// Utterances longer than this are split (the streaming ASR splits at 28 s).
    #[arg(long, default_value_t = 28.0)]
    max_utterance: f32,
    /// Also mark overlapped speech in each utterance (pyannote segmentation-3.0), as the live
    /// pipeline does with --diarize, and score it.
    #[arg(long)]
    overlap: bool,
    #[arg(long, default_value = "models/pyannote_segmentation_3_0.onnx")]
    seg_model: PathBuf,
    /// Also transcribe each utterance with this Whisper model (English) and score the
    /// transcript split at speaker changes by word timings, as the live output is.
    #[arg(long)]
    asr_model: Option<PathBuf>,
    #[arg(long)]
    gpu: bool,
    #[arg(long, default_value_t = 0)]
    gpu_device: i32,
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

#[derive(clap::Args)]
pub struct EvalArgs {
    /// WeSpeaker ONNX embedding model.
    #[arg(long, default_value = "models/wespeaker_en_voxceleb_resnet34_LM.onnx")]
    model: PathBuf,
    /// Directory with <meeting>.wav (16 kHz mono) and <meeting>.rttm.
    #[arg(long, default_value = "bench/data/ami")]
    data: PathBuf,
    /// Speech regions: "vad" (Silero), "seg" (pyannote segmentation-3.0, which also gives
    /// overlap), or "oracle" (the reference's speech, to measure clustering alone).
    #[arg(long, default_value = "seg")]
    speech: String,
    /// pyannote segmentation-3.0 ONNX model (for --speech seg and --overlap).
    #[arg(long, default_value = "models/pyannote_segmentation_3_0.onnx")]
    seg_model: PathBuf,
    /// Window shift of the segmentation model, seconds.
    #[arg(long, default_value_t = 2.5)]
    seg_shift: f32,
    /// Give overlap frames (two speakers per the segmentation model) a second speaker.
    #[arg(long)]
    overlap: bool,
    /// Euclidean thresholds between unit embeddings (1.0 ≈ cosine 0.5).
    #[arg(long, value_delimiter = ',', default_value = "0.9,0.95,1.0,1.05,1.1")]
    thresholds: Vec<f32>,
    /// Minimum cluster size in windows; 0 = min(12, ⌈n/10⌉).
    #[arg(long, default_value_t = 0)]
    min_cluster: usize,
    /// Small clusters are absorbed only into a large cluster closer than this (inf = always).
    #[arg(long, default_value_t = f32::INFINITY)]
    absorb_max: f32,
    #[arg(long, default_value_t = 3.0)]
    window: f32,
    #[arg(long, default_value_t = 1.5)]
    step: f32,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Write the hypothesis RTTMs (at the first threshold) into this directory.
    #[arg(long)]
    rttm_out: Option<PathBuf>,
}

pub fn run(cmd: Cmd) -> CliResult {
    match cmd {
        Cmd::Eval(a) => eval(a),
        Cmd::OnlineEval(a) => online_eval(a),
    }
}

fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

fn vad_regions(pcm: &[f32]) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
    let mut vad = sd_vad::SileroVad::new("models/silero_vad_16k_op15.onnx")?;
    let mut seg = sd_vad::Segmenter::new(sd_vad::SegmenterConfig::default());
    let mut probs = Vec::new();
    vad.push(pcm, &mut probs)?;
    let mut out: Vec<(u64, u64)> = probs
        .into_iter()
        .filter_map(|p| seg.push(p))
        .map(|s| (s.start, s.end))
        .collect();
    out.extend(seg.flush().map(|s| (s.start, s.end)));
    Ok(out)
}

fn oracle_regions(reference: &[Turn]) -> Vec<(u64, u64)> {
    let mut spans: Vec<(f64, f64)> = reference.iter().map(|t| (t.start, t.end)).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
        .into_iter()
        .map(|(a, b)| ((a * 16_000.0) as u64, (b * 16_000.0) as u64))
        .collect()
}

fn eval(a: EvalArgs) -> CliResult {
    let embedder = Embedder::new(&a.model, a.threads)?;
    let seg_model = if a.speech == "seg" || a.overlap {
        Some(sd_diar::segment::SegmentationModel::new(
            &a.seg_model,
            a.threads,
        )?)
    } else {
        None
    };
    let cfg = DiarConfig {
        window_s: a.window,
        step_s: a.step,
        ..Default::default()
    };
    let mut meetings: Vec<String> = std::fs::read_dir(&a.data)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.strip_suffix(".rttm").map(str::to_string)
        })
        .collect();
    meetings.sort();
    if let Some(d) = &a.rttm_out {
        std::fs::create_dir_all(d)?;
    }
    println!(
        "model {}; speech {}{}; windows {} s / step {} s; DER with 0.25 s collar (overlap scored / not scored)",
        a.model.display(),
        a.speech,
        if a.overlap { " + overlap" } else { "" },
        a.window,
        a.step
    );
    // Per threshold (and K = true count, last): (all, no-overlap).
    let n_settings = a.thresholds.len() + 1;
    let mut totals = vec![(DerResult::default(), DerResult::default()); n_settings];
    let mut audio_s = 0.0;
    let mut compute_s = 0.0;
    for m in &meetings {
        let reference = read_rttm(&std::fs::read_to_string(a.data.join(format!("{m}.rttm")))?);
        let k = {
            let mut s: Vec<&str> = reference.iter().map(|t| t.speaker.as_str()).collect();
            s.sort();
            s.dedup();
            s.len()
        };
        let pcm = read_wav(&a.data.join(format!("{m}.wav")))?;
        audio_s += pcm.len() as f64 / 16_000.0;
        let t = Instant::now();
        let count = match &seg_model {
            Some(m) => Some(m.count(&pcm, a.seg_shift)?),
            None => None,
        };
        let regions = match a.speech.as_str() {
            "vad" => vad_regions(&pcm)?,
            "seg" => count
                .as_ref()
                .expect("segmentation model loaded")
                .speech_regions(0.1, 0.1),
            "oracle" => oracle_regions(&reference),
            other => {
                return Err(format!("--speech must be vad, seg or oracle, not {other}").into());
            }
        };
        let emb = embed_regions(&pcm, &regions, &embedder, &cfg)?;
        let merges = dendrogram(&emb);
        compute_s += t.elapsed().as_secs_f64();
        let min_cluster = if a.min_cluster == 0 {
            min_cluster_size(emb.embeddings.len())
        } else {
            a.min_cluster
        };
        let mut line = format!("{m:<9} {k} spk {:>5} windows", emb.windows.len());
        for (i, setting) in a
            .thresholds
            .iter()
            .map(|&th| (th, None))
            .chain(std::iter::once((0.0, Some(k))))
            .enumerate()
        {
            let overlap = if a.overlap { count.as_ref() } else { None };
            let hyp = turns(
                &emb,
                &merges,
                setting.0,
                setting.1,
                min_cluster,
                a.absorb_max,
                overlap,
            );
            let n_hyp = {
                let mut s: Vec<&str> = hyp.iter().map(|t| t.speaker.as_str()).collect();
                s.sort();
                s.dedup();
                s.len()
            };
            let all = der(&reference, &hyp, 0.25, false);
            let no_ov = der(&reference, &hyp, 0.25, true);
            totals[i].0.add(all);
            totals[i].1.add(no_ov);
            line += &format!(
                " | {} {:>5.1}/{:>5.1}% ({n_hyp})",
                setting
                    .1
                    .map_or(format!("{:.2}", setting.0), |k| format!("K={k}")),
                100.0 * all.der(),
                100.0 * no_ov.der()
            );
            if i == 0
                && let Some(d) = &a.rttm_out
            {
                std::fs::write(d.join(format!("{m}.rttm")), write_rttm(m, &hyp))?;
            }
        }
        println!("{line}");
    }
    println!(
        "\nTotal ({} meetings, {:.0} min):",
        meetings.len(),
        audio_s / 60.0
    );
    for (i, (all, no_ov)) in totals.iter().enumerate() {
        let name = a
            .thresholds
            .get(i)
            .map_or("true speaker count".to_string(), |t| {
                format!("threshold {t:.2}")
            });
        println!(
            "  {name:<20} DER {:5.1}% (miss {:4.1} FA {:4.1} confusion {:4.1})   without overlap {:5.1}% (confusion {:4.1})",
            100.0 * all.der(),
            100.0 * all.miss / all.total,
            100.0 * all.false_alarm / all.total,
            100.0 * all.confusion / all.total,
            100.0 * no_ov.der(),
            100.0 * no_ov.confusion / no_ov.total,
        );
    }
    println!(
        "speech detection + embedding: {:.0} s for {:.0} s of audio (RTF {:.3})",
        compute_s,
        audio_s,
        compute_s / audio_s.max(1e-9)
    );
    Ok(ExitCode::SUCCESS)
}

fn online_eval(a: OnlineArgs) -> CliResult {
    use sd_diar::online::{OnlineConfig, OnlineDiarizer};
    use std::collections::HashMap;
    use std::sync::Arc;
    let embedder = Arc::new(Embedder::new(&a.model, a.threads)?);
    let segmenter = if a.overlap {
        Some(Arc::new(sd_diar::segment::SegmentationModel::new(
            &a.seg_model,
            a.threads,
        )?))
    } else {
        None
    };
    let whisper = match &a.asr_model {
        Some(m) => {
            let mut cfg = sd_asr::whisper::WhisperConfig::new(m);
            cfg.threads = a.threads;
            cfg.use_gpu = a.gpu;
            cfg.gpu_device = a.gpu_device;
            Some(sd_asr::whisper::WhisperEngine::new(cfg)?)
        }
        None => None,
    };
    // Absolute word timings per (meeting, utterance), transcribed once.
    let mut words_cache: HashMap<(String, usize), (String, Vec<sd_asr::Segment>)> = HashMap::new();
    let mut meetings: Vec<String> = std::fs::read_dir(&a.data)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.strip_suffix(".rttm").map(str::to_string)
        })
        .collect();
    meetings.sort();
    println!(
        "utterances: Silero VAD regions split at {} s (as the streaming ASR); speaker per utterance; re-cluster every {} s; DER with 0.25 s collar, overlap scored",
        a.max_utterance, a.recluster_every
    );
    for &assign in &a.assign {
        let cfg = OnlineConfig {
            assign_threshold: assign,
            recluster_every_s: a.recluster_every,
            absorb_max: a.absorb_max,
            ..Default::default()
        };
        let (mut first_total, mut final_total) = (DerResult::default(), DerResult::default());
        let (mut win_first_total, mut win_final_total) =
            (DerResult::default(), DerResult::default());
        let mut overlap_total = DerResult::default();
        let (mut text_total, mut text_utt_total) = (DerResult::default(), DerResult::default());
        let (mut finals_split, mut finals_n) = (0usize, 0usize);
        let (mut relabelled, mut speech) = (0.0f64, 0.0f64);
        let mut revisions = 0usize;
        let t = Instant::now();
        let mut audio = 0.0;
        for m in &meetings {
            let reference = read_rttm(&std::fs::read_to_string(a.data.join(format!("{m}.rttm")))?);
            let pcm = read_wav(&a.data.join(format!("{m}.wav")))?;
            audio += pcm.len() as f64 / 16_000.0;
            let max = (a.max_utterance * 16_000.0) as u64;
            let mut utts: Vec<(u64, u64)> = Vec::new();
            for (s, e) in vad_regions(&pcm)? {
                let mut x = s;
                while e - x > max {
                    utts.push((x, x + max));
                    x += max;
                }
                utts.push((x, e));
            }
            let mut dia = OnlineDiarizer::new(embedder.clone(), cfg);
            if let Some(seg) = &segmenter {
                dia = dia.with_segmentation(seg.clone());
            }
            let mut first: HashMap<u64, u32> = HashMap::new();
            for (i, &(s, e)) in utts.iter().enumerate() {
                let end = (e as usize).min(pcm.len());
                let (spk, rev) = dia.add(i as u64, s, &pcm[s as usize..end])?;
                first.insert(i as u64, spk);
                revisions += rev.len();
            }
            revisions += dia.recluster().len();
            let finals: HashMap<u64, u32> = dia.speakers().into_iter().collect();
            let to_turns = |labels: &HashMap<u64, u32>| -> Vec<Turn> {
                utts.iter()
                    .enumerate()
                    .map(|(i, &(s, e))| Turn {
                        start: s as f64 / 16_000.0,
                        end: e as f64 / 16_000.0,
                        speaker: format!("spk{}", labels[&(i as u64)]),
                    })
                    .collect()
            };
            let d_first = der(&reference, &to_turns(&first), 0.25, false);
            let d_final = der(&reference, &to_turns(&finals), 0.25, false);
            let w_first = der(&reference, &dia.turns(true), 0.25, false);
            let w_final = der(&reference, &dia.turns(false), 0.25, false);
            win_first_total.add(w_first);
            win_final_total.add(w_final);
            if segmenter.is_some() {
                // Per utterance after re-clustering: its turns plus the second speakers.
                let mut hyp: Vec<Turn> = Vec::new();
                for i in 0..utts.len() {
                    let Some(u) = dia.utterance(i as u64) else {
                        continue;
                    };
                    for (s, e, spk) in u.turns.into_iter().chain(u.overlaps) {
                        hyp.push(Turn {
                            start: s as f64 / 16_000.0,
                            end: e as f64 / 16_000.0,
                            speaker: format!("spk{spk}"),
                        });
                    }
                }
                overlap_total.add(der(&reference, &hyp, 0.25, false));
            }
            if let Some(w) = &whisper {
                use sd_asr::Recognizer;
                use sd_asr::speakers::{SpeakerInfo, SpeakerSpan, split};
                let mut hyp: Vec<Turn> = Vec::new();
                let mut hyp_utt: Vec<Turn> = Vec::new();
                for (i, &(s, e)) in utts.iter().enumerate() {
                    let key = (m.clone(), i);
                    if !words_cache.contains_key(&key) {
                        let end = (e as usize).min(pcm.len());
                        let t = Recognizer::transcribe(
                            w,
                            &pcm[s as usize..end],
                            sd_asr::Language::English,
                        )?;
                        let base = (s / 16) as i64;
                        let words = t
                            .words
                            .into_iter()
                            .map(|x| sd_asr::Segment {
                                start_ms: base + x.start_ms,
                                end_ms: base + x.end_ms,
                                ..x
                            })
                            .collect();
                        words_cache.insert(key.clone(), (t.text, words));
                    }
                    let (text, words) = &words_cache[&key];
                    let Some(u) = dia.utterance(i as u64) else {
                        continue;
                    };
                    let spans = |v: &[(u64, u64, u32)]| -> Vec<SpeakerSpan> {
                        v.iter()
                            .map(|&(start, end, speaker)| SpeakerSpan {
                                start,
                                end,
                                speaker,
                            })
                            .collect()
                    };
                    let info = SpeakerInfo {
                        speaker: Some(u.speaker),
                        turns: spans(&u.turns),
                        overlaps: spans(&u.overlaps),
                    };
                    if text.trim().is_empty() {
                        continue;
                    }
                    let parts = split(text, s, e, words, &info);
                    finals_n += 1;
                    if parts.len() > 1 {
                        finals_split += 1;
                    }
                    let turn = |a: u64, b: u64, spk: u32| Turn {
                        start: a as f64 / 16_000.0,
                        end: b as f64 / 16_000.0,
                        speaker: format!("spk{spk}"),
                    };
                    hyp_utt.push(turn(s, e, u.speaker));
                    for p in &parts {
                        hyp.push(turn(p.start, p.end, p.speaker));
                    }
                    for o in &info.overlaps {
                        hyp.push(turn(o.start, o.end, o.speaker));
                        hyp_utt.push(turn(o.start, o.end, o.speaker));
                    }
                }
                text_total.add(der(&reference, &hyp, 0.25, false));
                text_utt_total.add(der(&reference, &hyp_utt, 0.25, false));
            }
            for (i, &(s, e)) in utts.iter().enumerate() {
                let dur = (e - s) as f64 / 16_000.0;
                speech += dur;
                if first[&(i as u64)] != finals[&(i as u64)] {
                    relabelled += dur;
                }
            }
            let n_first = first
                .values()
                .collect::<std::collections::HashSet<_>>()
                .len();
            let n_final = finals
                .values()
                .collect::<std::collections::HashSet<_>>()
                .len();
            println!(
                "  assign {assign:.2} {m:<9} {} utterances: as emitted {:5.1}% ({n_first} ids), after re-clustering {:5.1}% ({n_final} ids)",
                utts.len(),
                100.0 * d_first.der(),
                100.0 * d_final.der()
            );
            first_total.add(d_first);
            final_total.add(d_final);
        }
        println!(
            "assign {assign:.2}: DER as emitted {:.1}% (confusion {:.1}), after re-clustering {:.1}% (confusion {:.1}); {:.1}% of speech relabelled; {revisions} revisions; {:.0} s compute for {:.0} s audio",
            100.0 * first_total.der(),
            100.0 * first_total.confusion / first_total.total,
            100.0 * final_total.der(),
            100.0 * final_total.confusion / final_total.total,
            100.0 * relabelled / speech.max(1e-9),
            t.elapsed().as_secs_f64(),
            audio
        );
        println!(
            "            split at speaker changes inside utterances: DER as emitted {:.1}% (confusion {:.1}), after re-clustering {:.1}% (confusion {:.1})",
            100.0 * win_first_total.der(),
            100.0 * win_first_total.confusion / win_first_total.total,
            100.0 * win_final_total.der(),
            100.0 * win_final_total.confusion / win_final_total.total,
        );
        let parts = |d: &DerResult| {
            format!(
                "{:.1}% (miss {:.1}, false alarm {:.1}, confusion {:.1})",
                100.0 * d.der(),
                100.0 * d.miss / d.total,
                100.0 * d.false_alarm / d.total,
                100.0 * d.confusion / d.total
            )
        };
        println!(
            "            after re-clustering, split: {}",
            parts(&win_final_total)
        );
        if segmenter.is_some() {
            println!(
                "            ... plus overlapped second speakers: {}",
                parts(&overlap_total)
            );
        }
        if whisper.is_some() {
            println!(
                "            transcripts (non-empty finals, with overlaps), one speaker per final: {}",
                parts(&text_utt_total)
            );
            println!(
                "            transcripts split at speaker changes by word timings: {}; {finals_split} of {finals_n} finals split",
                parts(&text_total)
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}
