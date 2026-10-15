//! `sdctl devices` and `sdctl record`: real-device capture into a journal.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use sd_capture::{DeviceKind, DeviceSelector, list_devices};
use sd_core::Source;
use sd_engine::{Recorder, RecorderConfig, RecorderEvent};
use sd_journal::{ExportOptions, JournalReader, export_wav};

use crate::CliResult;

pub fn devices() -> CliResult {
    for (kind, label) in [
        (DeviceKind::Input, "input (microphones)"),
        (DeviceKind::Output, "output (system audio via loopback)"),
    ] {
        println!("{label}:");
        let list = list_devices(kind)?;
        if list.is_empty() {
            println!("  (none)");
        }
        for d in list {
            let mix = d
                .mix_format
                .map(|f| format!("{} Hz x{}", f.sample_rate, f.channels))
                .unwrap_or_else(|| "?".into());
            println!(
                "  {} {}  [{mix}]\n      id: {}",
                if d.is_default { "*" } else { " " },
                d.name,
                d.id
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(clap::Args)]
pub struct Args {
    /// Recording length in seconds (Ctrl+C is not handled; pick a duration).
    #[arg(long, default_value_t = 30)]
    seconds: u64,
    /// Journal directory (must not exist).
    #[arg(long)]
    out: PathBuf,
    /// Microphone: "default", "none", or a device id from `sdctl devices`.
    #[arg(long, default_value = "default")]
    mic: String,
    /// System audio (loopback of an output device): "default", "none", or a device id.
    #[arg(long, default_value = "default")]
    system: String,
    /// Also export each source to a WAV file next to the journal.
    #[arg(long)]
    wav: bool,
    /// Capture the mic in raw mode (bypass driver/OS effects such as device echo cancellation).
    #[arg(long)]
    raw_mic: bool,
    /// Fault injection: stop reading the mic stream this many ms after it starts (tests the
    /// stall watchdog).
    #[arg(long, hide = true)]
    test_stall_after: Option<u32>,
    /// Force a device restart (re-open of the same device selection) every N seconds, to
    /// exercise the switch/recovery path. 0 disables.
    #[arg(long, default_value_t = 0)]
    restart_every: u64,
    /// Scheduled device switch, "SECONDS:SOURCE=DEVICE" (SOURCE is mic or system; DEVICE is
    /// "default" or an id). Repeatable, e.g. --at 10:mic=<id> --at 20:mic=default
    #[arg(long = "at", value_parser = parse_switch)]
    switches: Vec<(u64, Source, DeviceSelector)>,
    /// Run real-time processing (echo gate + AEC, 16 kHz, VAD) and write processed WAVs.
    #[arg(long)]
    process: bool,
    /// Echo canceller mode with --process: auto (echo gate), on, off.
    #[arg(long, default_value = "auto")]
    aec: String,
    /// Mic gain control (AGC2) with --process.
    #[arg(long)]
    agc: bool,
    /// Live transcription (needs --process).
    #[command(flatten)]
    asr: crate::asr::AsrArgs,
    /// Write final transcripts to this TSV.
    #[arg(long)]
    transcript: Option<PathBuf>,
}

fn parse_switch(s: &str) -> Result<(u64, Source, DeviceSelector), String> {
    let (secs, rest) = s.split_once(':').ok_or("expected SECONDS:SOURCE=DEVICE")?;
    let (src, dev) = rest
        .split_once('=')
        .ok_or("expected SECONDS:SOURCE=DEVICE")?;
    let secs = secs.parse().map_err(|e| format!("bad seconds: {e}"))?;
    let src: Source = src.parse().map_err(|e| format!("{e}"))?;
    let dev = match dev {
        "default" => DeviceSelector::Default,
        id => DeviceSelector::Id(id.to_string()),
    };
    Ok((secs, src, dev))
}

fn selector(s: &str) -> Option<DeviceSelector> {
    match s {
        "none" => None,
        "default" => Some(DeviceSelector::Default),
        id => Some(DeviceSelector::Id(id.to_string())),
    }
}

pub fn run(mut args: Args) -> CliResult {
    let mut cfg = RecorderConfig::new(&args.out);
    cfg.mic = selector(&args.mic);
    cfg.system = selector(&args.system);
    cfg.mic_raw = args.raw_mic;
    cfg.test_mic_stall_after_ms = args.test_stall_after;
    if args.process {
        let aec = match args.aec.as_str() {
            "auto" => sd_dsp::mic::AecMode::Auto,
            "on" => sd_dsp::mic::AecMode::On,
            "off" => sd_dsp::mic::AecMode::Off,
            other => return Err(format!("--aec must be auto, on or off, not {other}").into()),
        };
        cfg.processing = Some(sd_engine::ProcessingConfig {
            mic: sd_dsp::mic::MicConfig {
                aec,
                gain_control: args.agc,
                noise_suppression: None,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    if args.asr.enabled() {
        if cfg.processing.is_none() {
            return Err("live transcription needs --process".into());
        }
        cfg.asr = args.asr.setup()?;
    }
    let mut asr_stats = crate::process::AsrStats::default();
    let rec = Recorder::start(cfg)?;
    let events = rec.events().clone();
    let proc_rx = rec.processed().cloned();
    let mut processed = ProcessedWavs::new(&args.out);
    println!(
        "recording {} s into {} ...",
        args.seconds,
        args.out.display()
    );

    let start = Instant::now();
    let deadline = start + Duration::from_secs(args.seconds);
    let mut last_level: [(f32, f32); 2] = [(-180.0, -180.0); 2];
    let mut max_peak: [f32; 2] = [-180.0; 2];
    let mut problems = 0u32;
    let mut next_restart =
        (args.restart_every > 0).then(|| start + Duration::from_secs(args.restart_every));
    while Instant::now() < deadline {
        if let Some(t) = next_restart
            && Instant::now() >= t
        {
            for (source, sel) in [
                (Source::Mic, selector(&args.mic)),
                (Source::System, selector(&args.system)),
            ] {
                if let Some(sel) = sel {
                    rec.set_device(source, sel);
                }
            }
            next_restart = Some(t + Duration::from_secs(args.restart_every));
        }
        let elapsed = start.elapsed().as_secs();
        for (at, source, dev) in args.switches.iter().filter(|(at, _, _)| *at <= elapsed) {
            println!(
                "[{:>6.1}s]  >>> switching {source} to {dev:?} (scheduled at {at}s)",
                start.elapsed().as_secs_f64()
            );
            rec.set_device(*source, dev.clone());
        }
        args.switches.retain(|(at, _, _)| *at > elapsed);
        if let Some(rx) = &proc_rx {
            for c in rx.try_iter() {
                processed.add(&c)?;
            }
        }
        let Ok(ev) = rec.events().recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        let t = start.elapsed().as_secs_f64();
        match ev {
            RecorderEvent::Level {
                source,
                rms_dbfs,
                peak_dbfs,
                ..
            } => {
                last_level[source as usize] = (rms_dbfs, peak_dbfs);
                max_peak[source as usize] = max_peak[source as usize].max(peak_dbfs);
            }
            RecorderEvent::Health(h) => {
                let mut line = format!("[{t:>6.1}s]");
                for s in &h.sources {
                    let src = s.source.expect("source");
                    let (rms, peak) = last_level[src as usize];
                    let drift = s
                        .drift_ppm
                        .zip(s.drift_ppm_stderr)
                        .map(|(p, e)| format!("{p:+.1}±{e:.1} ppm"))
                        .unwrap_or_else(|| "  ...   ".into());
                    let cap = s.capture.unwrap_or_default();
                    line += &format!(
                        "  {src:<6} rms {rms:>6.1} pk {peak:>6.1} dBFS  drift {drift}  pkts {} glitch {} gap {} idle {}",
                        cap.packets, cap.device_glitches, cap.gap_frames, cap.idle_frames
                    );
                }
                println!("{line}");
            }
            RecorderEvent::Capture { source, event } => {
                println!("[{t:>6.1}s]  {source}: {event:?}")
            }
            RecorderEvent::Speech {
                source,
                start_sample,
                end_sample,
                echo_unverified,
            } => {
                println!(
                    "[{t:>6.1}s]  speech {source:<6} {:>7.2} - {:>7.2} s ({:.2} s){}",
                    start_sample as f64 / 16_000.0,
                    end_sample as f64 / 16_000.0,
                    (end_sample - start_sample) as f64 / 16_000.0,
                    if echo_unverified {
                        "  [echo unverified]"
                    } else {
                        ""
                    }
                )
            }
            RecorderEvent::Asr(e) => {
                print!("[{t:>6.1}s]");
                crate::process::print_asr(&e, &mut asr_stats);
            }
            RecorderEvent::EchoCanceller {
                active,
                correlation,
            } => {
                println!(
                    "[{t:>6.1}s]  echo canceller {} (echo correlation {correlation:?})",
                    if active { "ON" } else { "off" }
                )
            }
            e @ (RecorderEvent::SpeechStart { .. } | RecorderEvent::SpeechDiscarded { .. }) => {
                println!("[{t:>6.1}s]  {e:?}");
            }
            other => {
                problems += 1;
                println!("[{t:>6.1}s]  ! {other:?}");
            }
        }
    }
    let summary = rec.stop()?;
    println!("\nstopped after {:.2} s", summary.elapsed.as_secs_f64());
    if args.asr.enabled() {
        // Finals flushed at stop.
        for ev in events.try_iter() {
            if let RecorderEvent::Asr(e) = ev {
                print!("[  stop ]");
                crate::process::print_asr(&e, &mut asr_stats);
            }
        }
        println!("  {}", asr_stats.summary());
        if let Some(path) = &args.transcript {
            asr_stats.write_tsv(path)?;
        }
    }
    if let Some(rx) = &proc_rx {
        for c in rx.try_iter() {
            processed.add(&c)?;
        }
        if let Some(ps) = summary.health.processing {
            println!("  processing: {ps:?}");
        }
    }
    processed.finish()?;

    let reader = JournalReader::open(&summary.journal_dir)?;
    let mut ok = problems == 0;
    for (source, format) in reader.sources() {
        let r = reader.scan(source)?;
        let h = summary
            .health
            .sources
            .iter()
            .find(|s| s.source == Some(source));
        let dropped = h
            .map(|h| h.ring.dropped_frames + h.queue_dropped_frames)
            .unwrap_or(0);
        let cap = h.and_then(|h| h.capture).unwrap_or_default();
        println!(
            "  {source:<6} journal {:.3} s ({} Hz x{}), gaps in journal {}, dropped (ring+queue) {dropped}, device glitches {}, gap-fill {:.3} s, idle-fill {:.3} s, skipped {:.3} s, restarts {}, drift {}, max peak {:.1} dBFS, trimmed {:.3} s",
            r.duration_secs(),
            format.sample_rate,
            format.channels,
            r.gaps,
            cap.device_glitches,
            cap.gap_frames as f64 / format.sample_rate as f64,
            cap.idle_frames as f64 / format.sample_rate as f64,
            cap.skipped_frames as f64 / format.sample_rate as f64,
            cap.restarts,
            h.and_then(|h| h.drift_ppm.zip(h.drift_ppm_stderr))
                .map(|(p, e)| format!("{p:+.2} ± {e:.2} ppm"))
                .unwrap_or_else(|| "n/a".into()),
            max_peak[source as usize],
            cap.trimmed_frames as f64 / format.sample_rate as f64,
        );
        ok &= dropped == 0 && r.gaps == 0 && r.rejected_records() == 0;
        if args.wav {
            let out = args.out.with_extension(format!("{source}.wav"));
            let rep = export_wav(&reader, source, &out, ExportOptions::default())?;
            println!(
                "         wrote {} ({:.3} s)",
                out.display(),
                rep.frames_written as f64 / format.sample_rate as f64
            );
        }
    }
    let drift_of = |src: Source| {
        summary
            .health
            .sources
            .iter()
            .find(|s| s.source == Some(src))
            .and_then(|s| s.drift_ppm.zip(s.drift_ppm_stderr))
    };
    if let (Some((m, me)), Some((s, se))) = (drift_of(Source::Mic), drift_of(Source::System)) {
        let d = m - s;
        let e = (me * me + se * se).sqrt();
        println!(
            "  mic vs system clock drift: {d:+.2} ± {e:.2} ppm ({:+.1} ms per hour if uncorrected){}",
            d * 3.6,
            if d.abs() < 2.0 * e {
                " (not significant)"
            } else {
                ""
            }
        );
    }
    println!(
        "  verdict: {}",
        if ok {
            "PASS (continuous timeline, nothing dropped)"
        } else {
            "CHECK (see above)"
        }
    );
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

/// Processed 16 kHz audio streamed to `<out>.processed.{mic,system}.wav` as it arrives, so
/// long recordings do not accumulate it in memory.
struct ProcessedWavs {
    base: PathBuf,
    writers: [Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>; 2],
    samples: [u64; 2],
}

impl ProcessedWavs {
    fn new(base: &std::path::Path) -> Self {
        Self {
            base: base.to_path_buf(),
            writers: [None, None],
            samples: [0; 2],
        }
    }

    fn path(&self, i: usize) -> PathBuf {
        let name = ["mic", "system"][i];
        self.base.with_extension(format!("processed.{name}.wav"))
    }

    fn add(&mut self, c: &sd_engine::ProcessedChunk) -> Result<(), hound::Error> {
        let i = c.source as usize;
        if self.writers[i].is_none() {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            self.writers[i] = Some(hound::WavWriter::create(self.path(i), spec)?);
        }
        let w = self.writers[i].as_mut().expect("created above");
        for &v in &c.samples {
            w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        self.samples[i] += c.samples.len() as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<(), hound::Error> {
        for i in 0..2 {
            if let Some(w) = self.writers[i].take() {
                w.finalize()?;
                println!(
                    "  wrote {} ({:.2} s at 16 kHz)",
                    self.path(i).display(),
                    self.samples[i] as f64 / 16_000.0
                );
            }
        }
        Ok(())
    }
}
