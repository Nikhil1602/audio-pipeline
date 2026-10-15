use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use sd_core::Source;
use sd_journal::{Codec, ExportOptions, JournalConfig, JournalReader, JournalWriter, export_wav};
use serde::Serialize;

mod asr;
mod diar;
mod echotest;
mod loopback;
mod models;
mod process;
mod record;
mod replay;
mod simulate;

#[derive(Parser)]
#[command(name = "sdctl", version, about = "SonicDiar command-line tool")]
struct Cli {
    /// Log filter, e.g. "info" or "sd_journal=debug" (also read from RUST_LOG).
    #[arg(long, global = true, default_value = "warn")]
    log: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inspect or export an audio journal.
    #[command(subcommand)]
    Journal(JournalCmd),
    /// Push synthetic mic + system audio through ring buffers into a journal at N x real time,
    /// then verify the journal (no-loss check).
    Simulate(simulate::Args),
    /// List capture (microphone) and output (loopback) devices.
    Devices,
    /// Record microphone and system audio from real devices into a journal.
    Record(record::Args),
    /// Play a known test signal and verify the system-audio (loopback) capture of it.
    LoopbackTest(loopback::Args),
    /// Measure echo cancellation over the real speaker-to-mic path (plays speech aloud).
    EchoTest(echotest::Args),
    /// Download or verify pinned model files (models/manifest.json).
    #[command(subcommand)]
    Models(models::Cmd),
    /// Play a 16-bit WAV file on an output device (for tests).
    Play {
        file: PathBuf,
        /// Output device: "default" or an id.
        #[arg(long, default_value = "default")]
        device: String,
    },
    /// Run the processing stage (echo gate/AEC, AGC, VAD) offline over a recorded journal.
    Process(process::Args),
    /// Speech recognition: evaluation on benchmark data.
    #[command(subcommand)]
    Asr(asr::Cmd),
    /// Speaker diarization: evaluation on benchmark data.
    #[command(subcommand)]
    Diar(diar::Cmd),
    /// Developer tools.
    #[command(subcommand)]
    Debug(DebugCmd),
}

#[derive(Subcommand)]
enum DebugCmd {
    /// Re-run the timeline aligner over the packets recorded in a journal.
    Replay(replay::Args),
    /// Run VAD over a 16 kHz mono WAV and print speech segments.
    Vad {
        wav: PathBuf,
        /// Gain applied before VAD, in dB.
        #[arg(long, default_value_t = 0.0)]
        gain_db: f32,
    },
}

#[derive(Subcommand)]
enum JournalCmd {
    /// Verify every block and print a recovery report.
    Inspect {
        dir: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Export one source to a 16-bit WAV file.
    Export {
        dir: PathBuf,
        #[arg(long)]
        source: Source,
        #[arg(long)]
        out: PathBuf,
        /// Do not fill timeline gaps with silence.
        #[arg(long)]
        no_fill: bool,
    },
    /// Rewrite a journal with another codec, verify it sample by sample, and report sizes.
    Recode {
        dir: PathBuf,
        out: PathBuf,
        /// lossless or pcm.
        #[arg(long, default_value = "lossless")]
        codec: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    let result = match cli.cmd {
        Cmd::Journal(JournalCmd::Inspect { dir, json }) => inspect(dir, json),
        Cmd::Journal(JournalCmd::Export {
            dir,
            source,
            out,
            no_fill,
        }) => export(dir, source, out, no_fill),
        Cmd::Journal(JournalCmd::Recode { dir, out, codec }) => recode(dir, out, codec),
        Cmd::Simulate(args) => simulate::run(args),
        Cmd::Devices => record::devices(),
        Cmd::Record(args) => record::run(args),
        Cmd::LoopbackTest(args) => loopback::run(args),
        Cmd::Debug(DebugCmd::Replay(args)) => replay::run(args),
        Cmd::Debug(DebugCmd::Vad { wav, gain_db }) => debug_vad(wav, gain_db),
        Cmd::EchoTest(args) => echotest::run(args),
        Cmd::Models(cmd) => models::run(cmd),
        Cmd::Process(args) => process::run(args),
        Cmd::Asr(cmd) => asr::run(cmd),
        Cmd::Diar(cmd) => diar::run(cmd),
        Cmd::Play { file, device } => play(file, device),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

type CliResult = Result<ExitCode, Box<dyn std::error::Error>>;

#[derive(Serialize)]
struct InspectSource {
    source: String,
    sample_rate: u32,
    channels: u16,
    segments: usize,
    blocks: u64,
    frames: u64,
    duration_secs: f64,
    gaps: u64,
    gap_frames: u64,
    discontinuity_flags: u64,
    rejected_records: u64,
    problems: Vec<String>,
}

fn inspect(dir: PathBuf, json: bool) -> CliResult {
    let reader = JournalReader::open(&dir)?;
    let mut out = Vec::new();
    for (source, _) in reader.sources() {
        let r = reader.scan(source)?;
        out.push(InspectSource {
            source: source.to_string(),
            sample_rate: r.format.sample_rate,
            channels: r.format.channels,
            segments: r.segments.len(),
            blocks: r.blocks,
            frames: r.frames,
            duration_secs: r.duration_secs(),
            gaps: r.gaps,
            gap_frames: r.gap_frames,
            discontinuity_flags: r.discontinuity_flags,
            rejected_records: r.rejected_records(),
            problems: r
                .segments
                .iter()
                .filter_map(|s| {
                    s.stop_reason
                        .as_ref()
                        .map(|why| format!("segment {}: {why}", s.seq))
                })
                .collect(),
        });
    }
    let clean = reader.manifest().finished && out.iter().all(|s| s.problems.is_empty());
    if json {
        let doc = serde_json::json!({ "finished": reader.manifest().finished, "sources": out });
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        println!(
            "journal {} ({})",
            dir.display(),
            if reader.manifest().finished {
                "finished cleanly"
            } else {
                "NOT finished cleanly"
            }
        );
        for s in &out {
            println!(
                "  {:<6} {} Hz x{}  {:>8.2} s  {} blocks in {} segments  gaps {} ({} frames)  rejected {}",
                s.source,
                s.sample_rate,
                s.channels,
                s.duration_secs,
                s.blocks,
                s.segments,
                s.gaps,
                s.gap_frames,
                s.rejected_records
            );
            for p in &s.problems {
                println!("         ! {p}");
            }
        }
    }
    Ok(if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn export(dir: PathBuf, source: Source, out: PathBuf, no_fill: bool) -> CliResult {
    let reader = JournalReader::open(&dir)?;
    let opts = ExportOptions {
        fill_gaps: !no_fill,
        ..Default::default()
    };
    let r = export_wav(&reader, source, &out, opts)?;
    println!(
        "wrote {} ({} frames: {} audio, {} gap fill, {} gaps not filled)",
        out.display(),
        r.frames_written,
        r.audio_frames,
        r.gap_frames_filled,
        r.gaps_not_filled
    );
    Ok(ExitCode::SUCCESS)
}

fn recode(dir: PathBuf, out: PathBuf, codec: String) -> CliResult {
    let codec = match codec.as_str() {
        "lossless" => Codec::Lossless,
        "pcm" => Codec::PcmS16le,
        other => return Err(format!("--codec must be lossless or pcm, not {other}").into()),
    };
    let reader = JournalReader::open(&dir)?;
    let sources = reader.sources();
    let mut cfg = JournalConfig::new(sources.clone());
    cfg.segment_seconds = reader.manifest().segment_seconds;
    cfg.codec = codec;
    // One sync at the end: this measures encoding, not the disk.
    cfg.sync_interval = std::time::Duration::from_secs(3600);
    let mut w = JournalWriter::create(&out, cfg)?;
    let mut buf = Vec::new();
    let mut write_time = std::time::Duration::ZERO;
    let mut audio_secs = 0.0;
    for &(s, fmt) in &sources {
        for b in reader.blocks(s)? {
            let b = b?;
            buf.clear();
            buf.extend(b.samples.iter().map(|&v| v as f32 / 32767.0));
            let t = std::time::Instant::now();
            w.write(s, &b.meta, &buf)?;
            write_time += t.elapsed();
            audio_secs += b.meta.frames as f64 / fmt.sample_rate as f64;
        }
    }
    let stats = w.finish()?;

    let copy = JournalReader::open(&out)?;
    for &(s, _) in &sources {
        let mut a = reader.blocks(s)?;
        let mut b = copy.blocks(s)?;
        loop {
            match (a.next().transpose()?, b.next().transpose()?) {
                (None, None) => break,
                (Some(x), Some(y)) if x == y => {}
                _ => return Err(format!("{s}: recoded journal differs from the original").into()),
            }
        }
    }
    let (mut bytes, mut pcm) = (0, 0);
    for (s, st) in &stats {
        println!(
            "  {s:<6} {:>12} bytes ({:.1}% of {} PCM bytes)",
            st.bytes,
            100.0 * st.bytes as f64 / st.pcm_bytes.max(1) as f64,
            st.pcm_bytes
        );
        bytes += st.bytes;
        pcm += st.pcm_bytes;
    }
    println!(
        "verified identical. total {:.1}% of PCM; writing {audio_secs:.1} s of audio (all sources) took {:.3} s ({:.4}x real time)",
        100.0 * bytes as f64 / pcm.max(1) as f64,
        write_time.as_secs_f64(),
        write_time.as_secs_f64() / audio_secs.max(1e-9)
    );
    Ok(ExitCode::SUCCESS)
}

fn play(file: PathBuf, device: String) -> CliResult {
    let r = hound::WavReader::open(&file)?;
    let spec = r.spec();
    let samples: Vec<f32> = r
        .into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    let format = sd_core::StreamFormat::new(spec.sample_rate, spec.channels)?;
    let sel = if device == "default" {
        sd_capture::DeviceSelector::Default
    } else {
        sd_capture::DeviceSelector::Id(device)
    };
    let rep = sd_capture::play(&sel, format, &samples)?;
    println!(
        "played {} frames of {} ({} Hz x{})",
        rep.frames,
        file.display(),
        spec.sample_rate,
        spec.channels
    );
    Ok(ExitCode::SUCCESS)
}

fn debug_vad(wav: PathBuf, gain_db: f32) -> CliResult {
    let r = hound::WavReader::open(&wav)?;
    if r.spec().sample_rate != 16_000 || r.spec().channels != 1 {
        return Err("expected a 16 kHz mono WAV".into());
    }
    let g = 10f32.powf(gain_db / 20.0);
    let x: Vec<f32> = r
        .into_samples::<i16>()
        .map(|s| s.map(|v| (v as f32 / 32768.0 * g).clamp(-1.0, 1.0)))
        .collect::<Result<_, _>>()?;
    let mut vad = sd_vad::SileroVad::new("models/silero_vad_16k_op15.onnx")?;
    let mut probs = Vec::new();
    vad.push(&x, &mut probs)?;
    let mut seg = sd_vad::Segmenter::new(sd_vad::SegmenterConfig::default());
    let mut n = 0;
    for &p in &probs {
        if let Some(s) = seg.push(p) {
            n += 1;
            println!(
                "  speech {:>7.2} - {:>7.2} s",
                s.start as f64 / 16_000.0,
                s.end as f64 / 16_000.0
            );
        }
    }
    if let Some(s) = seg.flush() {
        n += 1;
        println!(
            "  speech {:>7.2} - {:>7.2} s",
            s.start as f64 / 16_000.0,
            s.end as f64 / 16_000.0
        );
    }
    let max = probs.iter().cloned().fold(0.0f32, f32::max);
    println!("{n} segments, max speech probability {max:.3} (gain {gain_db:+.0} dB)");
    Ok(ExitCode::SUCCESS)
}
