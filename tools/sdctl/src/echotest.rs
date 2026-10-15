//! `sdctl echo-test`: echo cancellation over the real speaker → room → microphone path.
//!
//! 1. Plays far-end speech (bench/data/far_en.wav) on the default output while recording the
//!    mic and the output's loopback into a journal.
//! 2. Aligns mic and loopback on host time and runs [`sd_dsp::aec16k::Aec16k`] over them.
//! 3. Reports single-talk echo removal (ERLE) per output (full / linear / hybrid).
//! 4. Double talk on the real echo: near-end speech (bench/data/near_en.wav) is added digitally
//!    to the recorded mic, and near-end preservation is measured against the same near-end sent
//!    through the same path with no far-end (matched references).
//! 5. Writes 16 kHz WAVs of each signal for listening.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use sd_capture::DeviceSelector;
use sd_core::{HostTime, Source, StreamFormat};
use sd_dsp::aec16k::{Aec16k, AsrMixer, FULL_LATENCY, LINEAR_LATENCY};
use sd_dsp::metrics::{power_ratio_db, rms_dbfs, si_sdr};
use sd_engine::{Recorder, RecorderConfig};
use sd_journal::JournalReader;

use crate::CliResult;

#[derive(clap::Args)]
pub struct Args {
    /// Far-end speech, 48 kHz mono 16-bit WAV.
    #[arg(long, default_value = "bench/data/far_en.wav")]
    far: PathBuf,
    /// Near-end speech for the double-talk test, 48 kHz mono 16-bit WAV.
    #[arg(long, default_value = "bench/data/near_en.wav")]
    near: PathBuf,
    /// Near-end level for the double-talk test, dBFS RMS (while speaking).
    #[arg(long, default_value_t = -30.0)]
    near_dbfs: f64,
    /// Output directory for the journal and WAVs (must not exist).
    #[arg(long)]
    out: PathBuf,
    /// Reuse an existing recording in `out` instead of playing and recording again.
    #[arg(long)]
    reuse: bool,
    /// Capture the mic in raw mode (bypass the device's own echo cancellation / effects).
    #[arg(long)]
    raw_mic: bool,
    /// Hybrid mixer echo threshold (dB of suppression).
    #[arg(long, default_value_t = 20.0)]
    hybrid_threshold_db: f32,
}

const RATE: usize = 48_000;
const OUT: usize = 16_000;

/// (output, frame AEC engaged, last echo correlation).
type AutoRun = Result<(Vec<f32>, Option<usize>, Option<f32>), Box<dyn std::error::Error>>;

fn read_wav(p: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let r = hound::WavReader::open(p)?;
    if r.spec().sample_rate != RATE as u32 || r.spec().channels != 1 {
        return Err(format!("{} must be 48 kHz mono", p.display()).into());
    }
    Ok(r.into_samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?)
}

fn write_wav(p: &Path, x: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: OUT as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(p, spec)?;
    for &v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Loads one source from a journal: (samples interleaved, channels, host time of frame 0).
fn load(
    reader: &JournalReader,
    source: Source,
) -> Result<(Vec<f32>, usize, HostTime), Box<dyn std::error::Error>> {
    let ch = reader.format_of(source)?.channels as usize;
    let mut v = Vec::new();
    let mut h0 = None;
    for b in reader.blocks(source)? {
        let b = b?;
        // Host time of frame 0 extrapolated from this block (timeline is continuous).
        h0.get_or_insert(HostTime::from_ns(
            b.meta
                .host_time
                .as_ns()
                .saturating_sub(b.meta.first_frame * 1_000_000_000 / RATE as u64),
        ));
        v.extend(b.samples.iter().map(|&s| s as f32 / 32767.0));
    }
    Ok((v, ch, h0.ok_or("empty source")?))
}

/// Runs mic (mono) + reference (interleaved) through a fresh Aec16k; returns (full, linear,
/// hybrid), each compensated to the linear output's latency.
fn process(
    mic: &[f32],
    reference: &[f32],
    ref_ch: usize,
    thr: f32,
) -> Result<[Vec<f32>; 3], Box<dyn std::error::Error>> {
    let mut a = Aec16k::new(RATE as u32, ref_ch as u16)?;
    let n = RATE / 100;
    let frames = mic.len() / n;
    let (mut full, mut lin, mut hyb) = (
        vec![0.0; frames * 160],
        vec![0.0; frames * 160],
        vec![0.0; frames * 160],
    );
    let mut mixer = AsrMixer::default();
    mixer.echo_threshold_db = thr;
    for k in 0..frames {
        let r = k * 160..(k + 1) * 160;
        a.process(
            &mic[k * n..(k + 1) * n],
            &reference[k * n * ref_ch..(k + 1) * n * ref_ch],
            &mut full[r.clone()],
            &mut lin[r.clone()],
        )?;
        mixer.push(&lin[r.clone()], &full[r.clone()], &mut hyb[r]);
    }
    let shift = |x: &[f32], d: usize| {
        let mut v = x[d.min(x.len())..].to_vec();
        v.resize(x.len(), 0.0);
        v
    };
    Ok([
        shift(&full, FULL_LATENCY),
        shift(&lin, LINEAR_LATENCY),
        shift(&hyb, FULL_LATENCY),
    ])
}

pub fn run(args: Args) -> CliResult {
    let far = read_wav(&args.far)?;
    let near_src = read_wav(&args.near)?;
    let journal = args.out.join("journal");
    if !args.reuse {
        std::fs::create_dir_all(&args.out)?;
        let mut cfg = RecorderConfig::new(&journal);
        cfg.mic = Some(DeviceSelector::Default);
        cfg.system = Some(DeviceSelector::Default);
        cfg.mic_raw = args.raw_mic;
        let rec = Recorder::start(cfg)?;
        std::thread::sleep(Duration::from_millis(1_000));
        while let Ok(ev) = rec.events().try_recv() {
            if let sd_engine::RecorderEvent::Capture {
                source,
                event:
                    sd_capture::CaptureEvent::Started {
                        device_name, raw, ..
                    },
            } = ev
            {
                println!("{source}: {device_name} (raw mode: {raw})");
            }
        }
        println!(
            "playing {:.1} s of far-end speech on the default output ...",
            far.len() as f64 / RATE as f64
        );
        let stereo: Vec<f32> = far.iter().flat_map(|&s| [s, s]).collect();
        sd_capture::play(
            &DeviceSelector::Default,
            StreamFormat::new(RATE as u32, 2)?,
            &stereo,
        )?;
        std::thread::sleep(Duration::from_millis(1_000));
        rec.stop()?;
    }

    // Align mic and loopback on host time.
    let reader = JournalReader::open(&journal)?;
    let (mic_all, _, hm) = load(&reader, Source::Mic)?;
    let (sys_all, sch, hs) = load(&reader, Source::System)?;
    let offset = (hs.signed_diff_ns(hm) as f64 * RATE as f64 / 1e9).round() as i64; // sys frame 0 at mic index `offset`
    let (mic_start, sys_start) = if offset >= 0 {
        (offset as usize, 0)
    } else {
        (0, (-offset) as usize)
    };
    let len = (mic_all.len() - mic_start).min(sys_all.len() / sch - sys_start);
    let mic = &mic_all[mic_start..mic_start + len];
    let sys = &sys_all[sys_start * sch..(sys_start + len) * sch];
    let sys_mono: Vec<f32> = sys
        .chunks_exact(sch)
        .map(|f| f.iter().sum::<f32>() / sch as f32)
        .collect();

    // Far-end activity span from the loopback itself.
    let frame_rms: Vec<f64> = sys_mono.chunks(RATE / 100).map(rms_dbfs).collect();
    let active: Vec<usize> = frame_rms
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r > -60.0)
        .map(|(i, _)| i)
        .collect();
    let (Some(&f_first), Some(&f_last)) = (active.first(), active.last()) else {
        return Err("no far-end audio found in the loopback; is the default output muted?".into());
    };
    let (t0, t1) = (f_first as f64 / 100.0, f_last as f64 / 100.0);
    let seg = |x: &[f32], a: f64, b: f64| {
        x[((a * OUT as f64) as usize).min(x.len())..((b * OUT as f64) as usize).min(x.len())]
            .to_vec()
    };
    println!(
        "aligned {:.1} s (loopback starts {:+.1} ms vs mic); far-end active {t0:.1}-{t1:.1} s; loopback {:.1} dBFS, mic {:.1} dBFS",
        len as f64 / RATE as f64,
        offset as f64 * 1e3 / RATE as f64,
        rms_dbfs(&sys_mono[(t0 * RATE as f64) as usize..(t1 * RATE as f64) as usize]),
        rms_dbfs(&mic[(t0 * RATE as f64) as usize..(t1 * RATE as f64) as usize]),
    );

    // Single talk on the real echo path.
    let silent = vec![0.0f32; sys.len()];
    let [full, lin, hyb] = process(mic, sys, sch, args.hybrid_threshold_db)?;
    let raw = process(mic, &silent, sch, args.hybrid_threshold_db)?[1].clone(); // same path, no AEC effect
    let (a, b) = (t0 + 4.0, t1);
    println!("single talk (far-end only, {a:.1}-{b:.1} s, after 4 s convergence):");
    println!(
        "  echo in mic {:.1} dBFS ({:.1} dB below loopback)",
        rms_dbfs(&seg(&raw, a, b)),
        -power_ratio_db(&seg(&raw, a, b), &crate::echotest::decim(&sys_mono, a, b))
    );
    for (name, x) in [("full", &full), ("linear", &lin), ("hybrid", &hyb)] {
        println!(
            "  {name:<7} ERLE {:>5.1} dB  (residual {:.1} dBFS)",
            power_ratio_db(&seg(&raw, a, b), &seg(x, a, b)),
            rms_dbfs(&seg(x, a, b))
        );
    }

    // Double talk: add near-end speech digitally to the real recording, from t0 + 8 s.
    let start = ((t0 + 8.0) * RATE as f64) as usize;
    let gain = 10f64.powf((args.near_dbfs - rms_dbfs(&near_src)) / 20.0) as f32;
    let mut near = vec![0.0f32; len];
    for (i, &v) in near_src.iter().enumerate() {
        if start + i < len {
            near[start + i] = v * gain;
        }
    }
    let mic_dt: Vec<f32> = mic.iter().zip(&near).map(|(&m, &n)| m + n).collect();
    let [dfull, dlin, dhyb] = process(&mic_dt, sys, sch, args.hybrid_threshold_db)?;
    let raw_dt = process(&mic_dt, &silent, sch, args.hybrid_threshold_db)?[1].clone();
    let near_ref = process(&near, &silent, sch, args.hybrid_threshold_db)?[1].clone();
    let dt_end = ((start + near_src.len()).min(len) as f64 / RATE as f64).min(t1);
    let (da, db) = (t0 + 9.0, dt_end);
    println!(
        "double talk ({da:.1}-{db:.1} s; near-end {:.1} dBFS, echo {:.1} dBFS in that span):",
        rms_dbfs(&seg(&near_ref, da, db)),
        rms_dbfs(&seg(&raw, da, db))
    );
    println!(
        "  raw     SI-SDR {:>5.1} dB",
        si_sdr(&seg(&raw_dt, da, db), &seg(&near_ref, da, db))
    );
    for (name, x) in [("full", &dfull), ("linear", &dlin), ("hybrid", &dhyb)] {
        let (o, n) = (seg(x, da, db), seg(&near_ref, da, db));
        let dot: f64 = o.iter().zip(&n).map(|(&p, &q)| p as f64 * q as f64).sum();
        let nn: f64 = n.iter().map(|&q| (q as f64).powi(2)).sum();
        println!(
            "  {name:<7} SI-SDR {:>5.1} dB, near-end gain {:+.1} dB",
            si_sdr(&o, &n),
            20.0 * (dot / nn).abs().log10()
        );
    }

    // The production chain: MicProcessor in Auto mode (echo gate), no NS/AGC so levels compare.
    let auto = |m: &[f32]| -> AutoRun {
        let mut p = sd_dsp::mic::MicProcessor::new(sd_dsp::mic::MicConfig {
            reference_channels: sch as u16,
            gain_control: false,
            noise_suppression: None,
            ..Default::default()
        })?;
        let n = RATE / 100;
        let frames = m.len() / n;
        let mut out = vec![0.0f32; frames * 160];
        let mut engaged = None;
        for k in 0..frames {
            p.process(
                &m[k * n..(k + 1) * n],
                &sys[k * n * sch..(k + 1) * n * sch],
                &mut out[k * 160..(k + 1) * 160],
            )?;
            if engaged.is_none() && p.status().aec_active {
                engaged = Some(k);
            }
        }
        let lat = p.latency();
        let mut v = out[lat..].to_vec();
        v.resize(out.len(), 0.0);
        Ok((v, engaged, p.status().echo_correlation))
    };
    let (auto_st, eng_st, corr_st) = auto(mic)?;
    let (auto_dt, eng_dt, corr_dt) = auto(&mic_dt)?;
    let raw_full = process(mic, &silent, sch, args.hybrid_threshold_db)?[0].clone();
    let raw_dt_full = process(&mic_dt, &silent, sch, args.hybrid_threshold_db)?[0].clone();
    let near_ref_full = process(&near, &silent, sch, args.hybrid_threshold_db)?[0].clone();
    let fmt_eng = |e: Option<usize>| {
        e.map(|k| format!("engaged at {:.1} s", k as f64 / 100.0))
            .unwrap_or_else(|| "never engaged".into())
    };
    println!("production chain (MicProcessor, AEC auto):");
    println!(
        "  single talk: AEC {} (echo correlation {:?}); ERLE {:.1} dB",
        fmt_eng(eng_st),
        corr_st,
        power_ratio_db(&seg(&raw_full, a, b), &seg(&auto_st, a, b))
    );
    println!(
        "  double talk: AEC {} (echo correlation {:?}); SI-SDR {:.1} dB (raw {:.1} dB)",
        fmt_eng(eng_dt),
        corr_dt,
        si_sdr(&seg(&auto_dt, da, db), &seg(&near_ref_full, da, db)),
        si_sdr(&seg(&raw_dt_full, da, db), &seg(&near_ref_full, da, db))
    );
    write_wav(&args.out.join("auto_single.wav"), &auto_st)?;
    write_wav(&args.out.join("auto_double.wav"), &auto_dt)?;

    for (name, x) in [
        ("mic_raw", &raw),
        ("aec_full", &full),
        ("aec_linear", &lin),
        ("aec_hybrid", &hyb),
        ("dt_raw", &raw_dt),
        ("dt_full", &dfull),
        ("dt_linear", &dlin),
        ("dt_hybrid", &dhyb),
    ] {
        write_wav(&args.out.join(format!("{name}.wav")), x)?;
    }
    println!("WAVs written to {}", args.out.display());
    Ok(ExitCode::SUCCESS)
}

/// Loopback mono at 48 kHz → crude 16 kHz power-equivalent segment for level comparison.
pub(crate) fn decim(x: &[f32], a: f64, b: f64) -> Vec<f32> {
    let s =
        &x[((a * RATE as f64) as usize).min(x.len())..((b * RATE as f64) as usize).min(x.len())];
    sd_dsp::metrics::decimate(s, 3)
}
