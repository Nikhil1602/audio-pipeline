//! `sdctl process`: run the processing stage offline over a recorded journal.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use sd_core::Source;
use sd_engine::{OfflineProcessor, ProcessingConfig, RecorderEvent};
use sd_journal::{JournalReader, RecoveredBlock};

use crate::CliResult;

/// Verified blocks of one journal source, in timeline order.
type BlockStream = Box<dyn Iterator<Item = sd_journal::Result<RecoveredBlock>>>;

#[derive(clap::Args)]
pub struct Args {
    /// Journal directory.
    journal: PathBuf,
    /// Output prefix for processed WAVs (`<out>.mic.wav`, `<out>.system.wav`).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Echo canceller mode: auto, on, off.
    #[arg(long, default_value = "auto")]
    aec: String,
    /// Disable mic gain control (AGC2).
    #[arg(long)]
    no_agc: bool,
    /// AGC2 initial gain (dB).
    #[arg(long, default_value_t = 10.0)]
    agc_initial_gain_db: f32,
    /// Echo canceller output blend: frames where the suppressor removes more than this many
    /// dB use the fully suppressed output, others the linear output (0 = always full; a large
    /// value such as 999 = always linear).
    #[arg(long, default_value_t = 20.0)]
    hybrid_threshold_db: f32,
    /// Noise suppression on the mic: off, low, moderate, high, veryhigh.
    #[arg(long, default_value = "off")]
    ns: String,
    /// Transcription (streaming policy, as live).
    #[command(flatten)]
    asr: crate::asr::AsrArgs,
    /// Write final transcripts to this TSV.
    #[arg(long)]
    transcript: Option<PathBuf>,
}

pub fn run(args: Args) -> CliResult {
    let reader = JournalReader::open(&args.journal)?;
    let formats = reader.sources();
    let aec = match args.aec.as_str() {
        "auto" => sd_dsp::mic::AecMode::Auto,
        "on" => sd_dsp::mic::AecMode::On,
        "off" => sd_dsp::mic::AecMode::Off,
        other => return Err(format!("--aec must be auto, on or off, not {other}").into()),
    };
    let cfg = ProcessingConfig {
        mic: sd_dsp::mic::MicConfig {
            aec,
            gain_control: !args.no_agc,
            agc_initial_gain_db: args.agc_initial_gain_db,
            hybrid_threshold_db: args.hybrid_threshold_db,
            noise_suppression: match args.ns.as_str() {
                "off" => None,
                "low" => Some(sd_dsp::NsLevel::Low),
                "moderate" => Some(sd_dsp::NsLevel::Moderate),
                "high" => Some(sd_dsp::NsLevel::High),
                "veryhigh" => Some(sd_dsp::NsLevel::VeryHigh),
                other => {
                    return Err(format!(
                        "--ns must be off, low, moderate, high or veryhigh, not {other}"
                    )
                    .into());
                }
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut p = OfflineProcessor::new(cfg, &formats)?;
    if let Some(setup) = args.asr.setup()? {
        p = p.with_asr(setup);
    }

    // Interleave the two sources in host-time order, as the live engine receives them.
    let mut iters: Vec<(Source, BlockStream)> = Vec::new();
    for &(s, _) in &formats {
        iters.push((s, Box::new(reader.blocks(s)?)));
    }
    let mut heads: Vec<Option<RecoveredBlock>> = iters
        .iter_mut()
        .map(|(_, it)| it.next().transpose())
        .collect::<Result<_, _>>()?;
    let mut buf = Vec::new();
    let mut audio_secs = 0.0;
    let started = Instant::now();
    let mut outputs: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut events = Vec::new();
    loop {
        let Some(i) = (0..heads.len())
            .filter(|&i| heads[i].is_some())
            .min_by_key(|&i| heads[i].as_ref().unwrap().meta.host_time)
        else {
            break;
        };
        let b = heads[i].take().unwrap();
        let (source, fmt) = (iters[i].0, formats[i].1);
        buf.clear();
        buf.extend(b.samples.iter().map(|&v| sd_journal::i16_to_f32(v)));
        if source == Source::Mic {
            audio_secs += b.meta.frames as f64 / fmt.sample_rate as f64;
        }
        p.push(source, b.meta, &buf);
        heads[i] = iters[i].1.next().transpose()?;
        let (a, e) = p.drain();
        for c in a {
            outputs[c.source as usize].extend_from_slice(&c.samples);
        }
        events.extend(e);
    }
    let (a, e, stats) = p.finish();
    for c in a {
        outputs[c.source as usize].extend_from_slice(&c.samples);
    }
    events.extend(e);
    let el = started.elapsed().as_secs_f64();

    let mut asr_stats = AsrStats::default();
    for ev in &events {
        match ev {
            RecorderEvent::Speech {
                source,
                start_sample,
                end_sample,
                echo_unverified,
            } => {
                println!(
                    "  speech {source:<6} {:>7.2} - {:>7.2} s{}",
                    *start_sample as f64 / 16_000.0,
                    *end_sample as f64 / 16_000.0,
                    if *echo_unverified {
                        "  [echo unverified]"
                    } else {
                        ""
                    }
                )
            }
            RecorderEvent::EchoCanceller {
                active,
                correlation,
            } => {
                println!(
                    "  echo canceller {} (correlation {correlation:?})",
                    if *active { "ON" } else { "off" }
                )
            }
            RecorderEvent::Warning { message } => println!("  ! {message}"),
            RecorderEvent::Asr(e) => print_asr(e, &mut asr_stats),
            _ => {}
        }
    }
    if args.asr.enabled() {
        println!("{}", asr_stats.summary());
        if let Some(path) = &args.transcript {
            asr_stats.write_tsv(path)?;
        }
    }
    let level = |x: &[f32]| {
        10.0 * (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64 + 1e-20)
            .log10()
    };
    println!(
        "processed {audio_secs:.1} s in {el:.2} s ({:.3}x real time); mic out {:.1} dBFS rms, system out {:.1} dBFS rms; {stats:?}",
        el / audio_secs.max(1e-9),
        level(&outputs[0]),
        level(&outputs[1])
    );
    if let Some(out) = &args.out {
        for (i, name) in ["mic", "system"].iter().enumerate() {
            if outputs[i].is_empty() {
                continue;
            }
            let path = out.with_extension(format!("{name}.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&path, spec)?;
            for &v in &outputs[i] {
                w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
            }
            w.finalize()?;
            println!("  wrote {}", path.display());
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Default)]
pub(crate) struct AsrStats {
    /// source, start, end, language, text, top language-ID probabilities, utterance id.
    finals: Vec<(String, f64, f64, String, String, String, u64)>,
    /// Speakers per (source, utterance id), as last emitted or revised.
    speakers: std::collections::HashMap<(String, u64), Speakers>,
    revisions: usize,
    final_latency_ms: Vec<u64>,
    partial_latency_ms: Vec<u64>,
    compute_ms: u64,
    audio_ms: u64,
}

/// Majority speaker, parts and overlaps of a final.
struct Speakers {
    speaker: u32,
    parts: Vec<sd_asr::speakers::SpeakerPart>,
    overlaps: Vec<sd_asr::speakers::SpeakerSpan>,
}

fn secs(s: u64) -> f64 {
    s as f64 / 16_000.0
}

/// Speaker label as printed and written ("S1" is speaker 0).
fn label(speaker: u32) -> String {
    format!("S{}", speaker + 1)
}

/// The parts (when the speaker changes within a final) and overlaps, one line each.
fn print_speakers(s: &Speakers) {
    if s.parts.len() > 1 {
        for p in &s.parts {
            println!(
                "      | {} {:>7.2}-{:>7.2} {}",
                label(p.speaker),
                secs(p.start),
                secs(p.end),
                p.text
            );
        }
    }
    for o in &s.overlaps {
        println!(
            "      + {} also speaking {:>7.2}-{:>7.2}",
            label(o.speaker),
            secs(o.start),
            secs(o.end)
        );
    }
}

pub(crate) fn print_asr(e: &sd_asr::runner::AsrEvent, st: &mut AsrStats) {
    use sd_asr::runner::AsrEvent;
    use sd_asr::stream::Output;
    let secs = |s: &u64| secs(*s);
    match e {
        AsrEvent::Transcript(Output::Partial {
            source,
            start,
            end,
            text,
            ..
        }) => println!(
            "  ~ {source:?} {:>7.2}-{:>7.2}  {text}",
            secs(start),
            secs(end)
        ),
        AsrEvent::Transcript(Output::Final {
            source,
            start,
            end,
            id,
            text,
            language,
            language_probs,
            echo_unverified,
            speaker,
            parts,
            overlaps,
            ..
        }) => {
            let lid = language_probs
                .iter()
                .take(4)
                .map(|(l, p)| format!("{}:{p:.2}", l.code()))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "  > {source:?} {:>7.2}-{:>7.2} [{}]{}{} {text}",
                secs(start),
                secs(end),
                language.code(),
                speaker.map_or(String::new(), |s| format!(" {}", label(s))),
                if *echo_unverified {
                    " (echo unverified)"
                } else {
                    ""
                }
            );
            if let Some(s) = speaker {
                let spk = Speakers {
                    speaker: *s,
                    parts: parts.clone(),
                    overlaps: overlaps.clone(),
                };
                print_speakers(&spk);
                st.speakers.insert((format!("{source:?}"), *id), spk);
            }
            st.finals.push((
                format!("{source:?}"),
                secs(start),
                secs(end),
                language.code().to_string(),
                text.clone(),
                lid,
                *id,
            ));
        }
        AsrEvent::Transcript(Output::Dropped {
            source,
            start,
            end,
            text,
            reason,
            ..
        }) => println!(
            "  x {source:?} {:>7.2}-{:>7.2} dropped as far-end bleed (similarity {:.2}): {text}",
            secs(start),
            secs(end),
            reason.similarity
        ),
        AsrEvent::Skipped {
            source,
            start,
            end,
            reason,
            ..
        } => {
            println!(
                "  ! {source:?} {:>7.2}-{:>7.2} not transcribed: {reason}",
                secs(start),
                secs(end)
            )
        }
        AsrEvent::SpeakerRevision {
            source,
            id,
            speaker,
            parts,
            overlaps,
        } => {
            st.revisions += 1;
            let spk = Speakers {
                speaker: *speaker,
                parts: parts.clone(),
                overlaps: overlaps.clone(),
            };
            println!(
                "  * {source:?} utterance {id} is now speaker {}",
                label(*speaker)
            );
            print_speakers(&spk);
            st.speakers.insert((format!("{source:?}"), *id), spk);
        }
        AsrEvent::Timing {
            partial,
            audio_ms,
            compute_ms,
            latency_ms,
            ..
        } => {
            if *partial {
                st.partial_latency_ms.push(*latency_ms);
            } else {
                st.final_latency_ms.push(*latency_ms);
            }
            st.compute_ms += compute_ms;
            st.audio_ms += audio_ms;
        }
    }
}

impl AsrStats {
    pub(crate) fn write_tsv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(
            f,
            "source\tstart_s\tend_s\tspeaker\tlanguage\ttext\tlid\toverlap"
        )?;
        // One row per speaker part; overlaps as "speaker@start-end".
        for (src, a, b, l, t, lid, id) in &self.finals {
            let Some(s) = self.speakers.get(&(src.clone(), *id)) else {
                writeln!(f, "{src}\t{a:.2}\t{b:.2}\t\t{l}\t{t}\t{lid}\t")?;
                continue;
            };
            let parts: Vec<(f64, f64, u32, &str)> = if s.parts.is_empty() {
                vec![(*a, *b, s.speaker, t.as_str())]
            } else {
                s.parts
                    .iter()
                    .map(|p| (secs(p.start), secs(p.end), p.speaker, p.text.as_str()))
                    .collect()
            };
            for (pa, pb, speaker, text) in parts {
                let overlap: Vec<String> = s
                    .overlaps
                    .iter()
                    .filter(|o| secs(o.end) > pa && secs(o.start) < pb)
                    .map(|o| {
                        format!(
                            "{src}-{}@{:.2}-{:.2}",
                            label(o.speaker),
                            secs(o.start).max(pa),
                            secs(o.end).min(pb)
                        )
                    })
                    .collect();
                writeln!(
                    f,
                    "{src}\t{pa:.2}\t{pb:.2}\t{src}-{}\t{l}\t{text}\t{lid}\t{}",
                    label(speaker),
                    overlap.join(" ")
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn summary(&mut self) -> String {
        let pct = |v: &mut Vec<u64>, q: f64| {
            if v.is_empty() {
                return 0;
            }
            v.sort_unstable();
            v[((v.len() - 1) as f64 * q).round() as usize]
        };
        let mut s = format!(
            "ASR: {} finals; decoded {:.0} s of audio in {:.0} s; final latency p50 {} ms p95 {} ms; partial latency p50 {} ms p95 {} ms",
            self.finals.len(),
            self.audio_ms as f64 / 1000.0,
            self.compute_ms as f64 / 1000.0,
            pct(&mut self.final_latency_ms, 0.5),
            pct(&mut self.final_latency_ms, 0.95),
            pct(&mut self.partial_latency_ms, 0.5),
            pct(&mut self.partial_latency_ms, 0.95),
        );
        if !self.speakers.is_empty() {
            let mut per: std::collections::BTreeMap<&str, std::collections::BTreeSet<u32>> =
                Default::default();
            for ((src, _), s) in &self.speakers {
                let set = per.entry(src).or_default();
                set.insert(s.speaker);
                set.extend(s.parts.iter().map(|p| p.speaker));
            }
            let counts: Vec<String> = per
                .iter()
                .map(|(src, set)| format!("{src} {}", set.len()))
                .collect();
            s += &format!(
                "; speakers: {}; {} revisions",
                counts.join(", "),
                self.revisions
            );
        }
        s
    }
}
