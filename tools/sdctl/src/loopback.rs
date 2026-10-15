//! `sdctl loopback-test`: end-to-end verification of system-audio capture.
//!
//! Plays a known stereo noise signal on an output device while the recorder captures that
//! device's loopback into a journal, then cross-correlates the journal with the reference:
//!
//! * content fidelity: normalized correlation per 100 ms window (L and R separately);
//! * continuity: the best-matching lag must be identical in every window (a dropped or
//!   duplicated packet, or uncorrected timeline drift, shifts the lag);
//! * timing: the lag predicted from timestamps (render device clock vs. capture `qpcPosition`)
//!   is compared with the measured lag.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use sd_capture::{CaptureEvent, DeviceSelector};
use sd_core::{HostTime, Source, StreamFormat};
use sd_engine::{Recorder, RecorderConfig, RecorderEvent};
use sd_journal::JournalReader;

use crate::CliResult;

#[derive(clap::Args)]
pub struct Args {
    /// Output device: "default" or an id from `sdctl devices`.
    #[arg(long, default_value = "default")]
    device: String,
    /// Test signal length in seconds.
    #[arg(long, default_value_t = 8)]
    seconds: u32,
    /// Test signal RMS level in dBFS (it is audible at normal volume).
    #[arg(long, default_value_t = -30.0)]
    level_dbfs: f32,
    /// Fidelity is judged above this frequency. Driver effects (bass boost, loudness, crossovers)
    /// can alter low frequencies before the loopback tap; on the development laptop their
    /// influence extends to ~1 kHz (spectral coherence 0.9991 at 300-1000 Hz, 0.9999 above).
    #[arg(long, default_value_t = 1000.0)]
    fidelity_hz: f64,
    /// Journal directory (default: a new directory under the OS temp dir, removed on success).
    #[arg(long)]
    out: Option<PathBuf>,
}

const RATE: u32 = 48_000;
const WIN: usize = 4_800; // 100 ms

fn noise(seconds: u32, level_dbfs: f32) -> Vec<f32> {
    // Uniform noise with the requested RMS: uniform[-a, a] has RMS a / sqrt(3).
    let a = 10f32.powf(level_dbfs / 20.0) * 3f32.sqrt();
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    (0..seconds as usize * RATE as usize * 2)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
        })
        .collect()
}

/// 4th-order Butterworth high-pass (two cascaded RBJ biquads), applied in place.
fn highpass(x: &mut [f32], cutoff_hz: f64) {
    for q in [0.541_196_1, 1.306_563_0] {
        let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / RATE as f64;
        let alpha = w0.sin() / (2.0 * q);
        let cos = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos) / 2.0 / a0;
        let b1 = -(1.0 + cos) / a0;
        let b2 = b0;
        let a1 = -2.0 * cos / a0;
        let a2 = (1.0 - alpha) / a0;
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for v in x.iter_mut() {
            let x0 = *v as f64;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0 as f32;
        }
    }
}

/// Normalized cross-correlation of `r` with `c[off..]` (both de-interleaved, same length).
fn ncc(r: &[f32], c: &[f32]) -> f64 {
    let (mut rc, mut rr, mut cc) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in r.iter().zip(c) {
        rc += a as f64 * b as f64;
        rr += a as f64 * a as f64;
        cc += b as f64 * b as f64;
    }
    if rr == 0.0 || cc == 0.0 {
        0.0
    } else {
        rc / (rr * cc).sqrt()
    }
}

fn best_lag(r: &[f32], c: &[f32], lo: i64, hi: i64) -> (i64, f64) {
    let mut best = (lo, f64::MIN);
    for lag in lo..=hi {
        if lag < 0 || lag as usize + r.len() > c.len() {
            continue;
        }
        let v = ncc(r, &c[lag as usize..lag as usize + r.len()]);
        if v > best.1 {
            best = (lag, v);
        }
    }
    best
}

/// Fits an NLMS FIR filter from reference to capture and returns the residual-to-capture energy
/// ratio (dB) over the second half. A strongly negative value means the capture is a *linear*
/// filtering of the reference (e.g. an equalizer in the audio driver), not corrupted audio.
fn linear_fit_residual_db(r: &[f32], c: &[f32], taps: usize) -> f64 {
    let n = r.len().min(c.len());
    let mut w = vec![0.0f64; taps];
    let (mut e_res, mut e_cap) = (0.0f64, 0.0f64);
    let half = n / 2;
    for i in taps..n {
        let x = &r[i + 1 - taps..=i];
        let y: f64 = w
            .iter()
            .zip(x.iter().rev())
            .map(|(a, &b)| a * b as f64)
            .sum();
        let err = c[i] as f64 - y;
        let pow: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() + 1e-9;
        let mu = 0.5 / pow;
        for (wk, &xk) in w.iter_mut().zip(x.iter().rev()) {
            *wk += mu * err * xk as f64;
        }
        if i >= half {
            e_res += err * err;
            e_cap += (c[i] as f64).powi(2);
        }
    }
    10.0 * (e_res / e_cap.max(1e-30)).log10()
}

/// How long to wait for capture to start (it includes answering the permission prompt).
const START_TIMEOUT: Duration = Duration::from_secs(120);

pub fn run(args: Args) -> CliResult {
    let device = if args.device == "default" {
        DeviceSelector::Default
    } else {
        DeviceSelector::Id(args.device.clone())
    };
    let dir = args.out.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("sdctl-loopback-{}", HostTime::now().as_ns()))
    });
    let reference = noise(args.seconds, args.level_dbfs);
    let format = StreamFormat::new(RATE, 2)?;

    let mut cfg = RecorderConfig::new(&dir);
    cfg.mic = None;
    cfg.system = Some(device.clone());
    let rec = Recorder::start(cfg)?;
    // Play only once capture runs: on macOS opening the tap waits for the user to allow
    // system audio recording.
    let deadline = Instant::now() + START_TIMEOUT;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match rec.events().recv_timeout(left) {
            Ok(RecorderEvent::Capture { event, .. }) => match event {
                CaptureEvent::Started { .. } => break,
                CaptureEvent::PermissionNeeded { what, .. } => {
                    println!("waiting for permission: {what}")
                }
                CaptureEvent::Retrying { error, .. } => println!("capture retrying: {error}"),
                CaptureEvent::NoDevice => println!("capture: no output device"),
                _ => {}
            },
            Ok(_) => {}
            Err(_) => {
                rec.stop()?;
                return Err(format!(
                    "system audio capture did not start within {} s",
                    START_TIMEOUT.as_secs()
                )
                .into());
            }
        }
    }
    std::thread::sleep(Duration::from_millis(600));
    println!(
        "playing {} s of noise at {} dBFS RMS ...",
        args.seconds, args.level_dbfs
    );
    let play = sd_capture::play(&device, format, &reference)?;
    std::thread::sleep(Duration::from_millis(600));
    let summary = rec.stop()?;

    // Load the captured system stream (continuous timeline) as de-interleaved f32.
    let reader = JournalReader::open(&summary.journal_dir)?;
    let (mut cl, mut cr) = (Vec::new(), Vec::new());
    let mut first: Option<(u64, HostTime)> = None;
    let mut expected = None;
    let mut journal_gaps = 0;
    for b in reader.blocks(Source::System)? {
        let b = b?;
        first.get_or_insert((b.meta.first_frame, b.meta.host_time));
        if expected.is_some_and(|e| e != b.meta.first_frame) {
            journal_gaps += 1;
        }
        expected = Some(b.meta.first_frame + b.meta.frames as u64);
        for f in b.samples.chunks_exact(2) {
            cl.push(f[0] as f32 / 32767.0);
            cr.push(f[1] as f32 / 32767.0);
        }
    }
    let (f0, h0) = first.ok_or("no system audio captured")?;
    let (rl, rr): (Vec<f32>, Vec<f32>) = reference.chunks_exact(2).map(|f| (f[0], f[1])).unzip();
    let (rl_full, cl_full) = (rl.clone(), cl.clone());

    // Lag predicted by timestamps: where reference frame 0 should sit in the capture.
    let predicted = (play.first_frame_host_time.signed_diff_ns(h0) as f64 * RATE as f64 / 1e9)
        as i64
        - f0 as i64;
    // Coarse search ±300 ms around the prediction, using a 100 ms window 1 s into the signal.
    let probe = RATE as usize;
    let (lag_probe, c0) = best_lag(
        &rl[probe..probe + WIN],
        &cl,
        predicted + probe as i64 - 14_400,
        predicted + probe as i64 + 14_400,
    );
    let lag = lag_probe - probe as i64;
    println!(
        "global lag {lag} frames (timestamps predicted {predicted}; difference {:+.2} ms), probe correlation {c0:.5}",
        (lag - predicted) as f64 * 1e3 / RATE as f64
    );

    // Driver audio effects (bass boost, loudness) may alter low frequencies before the loopback
    // tap. Fidelity is therefore judged above `fidelity_hz`; full-band numbers are reported too.
    let full_band_probe = c0;
    for v in [&mut cl, &mut cr] {
        highpass(v, args.fidelity_hz);
    }
    let (mut rl, mut rr) = (rl, rr);
    highpass(&mut rl, args.fidelity_hz);
    highpass(&mut rr, args.fidelity_hz);

    // Per-window check, both channels, ±5 ms local search (on high-passed signals).
    let windows = rl.len() / WIN;
    let (mut min_corr, mut lag_changes, mut swapped) = (f64::MAX, 0usize, 0usize);
    let mut gains = Vec::new();
    let mut low_windows = Vec::new();
    for w in 0..windows {
        let s = w * WIN;
        let (wl, wcl) = best_lag(
            &rl[s..s + WIN],
            &cl,
            lag + s as i64 - 240,
            lag + s as i64 + 240,
        );
        let (wr, wcr) = best_lag(
            &rr[s..s + WIN],
            &cr,
            lag + s as i64 - 240,
            lag + s as i64 + 240,
        );
        if wl - s as i64 != lag || wr - s as i64 != lag {
            lag_changes += 1;
        }
        let at = (lag + s as i64) as usize;
        if at + WIN <= cr.len() && ncc(&rl[s..s + WIN], &cr[at..at + WIN]) > wcr {
            swapped += 1;
        }
        min_corr = min_corr.min(wcl).min(wcr);
        if wcl.min(wcr) < 0.99 {
            low_windows.push((w, wcl.min(wcr)));
        }
        if at + WIN <= cl.len() {
            let e = |x: &[f32]| x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            gains.push(10.0 * (e(&cl[at..at + WIN]) / e(&rl[s..s + WIN])).log10());
        }
    }
    gains.sort_by(f64::total_cmp);
    let gain = gains.get(gains.len() / 2).copied().unwrap_or(f64::NAN);
    if !low_windows.is_empty() {
        let list: Vec<String> = low_windows
            .iter()
            .map(|(w, c)| format!("{:.1}s:{c:.3}", *w as f64 * 0.1))
            .collect();
        println!(
            "windows below 0.99 (time into signal: correlation): {}",
            list.join(", ")
        );
    }
    let h = summary.health.sources.first().cloned().unwrap_or_default();
    let cap = h.capture.unwrap_or_default();
    println!(
        "above {} Hz, {windows} windows: min correlation {min_corr:.5}, lag changes {lag_changes}, channel swaps {swapped}, median gain {gain:+.2} dB",
        args.fidelity_hz
    );
    println!(
        "capture: packets {}, device glitches {}, gap-fill {} frames, trimmed {}, journal gaps {journal_gaps}, ring drops {}, render clock samples {}",
        cap.packets,
        cap.device_glitches,
        cap.gap_frames,
        cap.trimmed_frames,
        h.ring.dropped_frames,
        play.clock_samples
    );

    // Linear-filter explanation over the whole signal (64-tap NLMS, left channel, centred so the
    // filter can model up to ±0.67 ms of delay/phase around the measured lag).
    let at = (lag - 32).max(0) as usize;
    let fit_db = if at + rl_full.len() <= cl_full.len() {
        linear_fit_residual_db(&rl_full, &cl_full[at..at + rl_full.len()], 64)
    } else {
        f64::NAN
    };
    let mut corrs: Vec<f64> = Vec::new();
    for w in 0..windows {
        let s0 = w * WIN;
        let a0 = (lag + s0 as i64) as usize;
        if a0 + WIN <= cl_full.len() {
            corrs.push(ncc(&rl_full[s0..s0 + WIN], &cl_full[a0..a0 + WIN]));
        }
    }
    corrs.sort_by(f64::total_cmp);
    if !corrs.is_empty() {
        println!(
            "full band: probe correlation {full_band_probe:.4}, per-window p10 {:.4} / median {:.4} / p90 {:.4}, 64-tap linear-fit residual {fit_db:.1} dB",
            corrs[corrs.len() / 10],
            corrs[corrs.len() / 2],
            corrs[corrs.len() * 9 / 10]
        );
    }

    // Fidelity is judged above `fidelity_hz` (see above); continuity checks are strict.
    // Linear-fit residual above the fidelity cutoff (128 taps, centred on the lag): the driver may apply a
    // linear filter (EQ / crossover phase), which correlation penalizes but which is harmless.
    let at_hp = (lag - 64).max(0) as usize;
    let fit_hp_db = if at_hp + rl.len() <= cl.len() {
        linear_fit_residual_db(&rl, &cl[at_hp..at_hp + rl.len()], 128)
    } else {
        f64::NAN
    };
    println!(
        "above {} Hz: 128-tap linear-fit residual {fit_hp_db:.1} dB (capture = linear filter of playback if < -30 dB)",
        args.fidelity_hz
    );
    let faithful = min_corr > 0.99 || fit_hp_db < -30.0;
    let driver_processing = corrs.get(corrs.len() / 2).is_some_and(|&m| m < 0.99);
    println!(
        "loopback clock drift vs host (estimated over the session): {}",
        h.drift_ppm
            .map(|p| format!("{p:+.2} ppm"))
            .unwrap_or_else(|| "n/a (session too short)".into())
    );
    let ok = faithful
        && lag_changes == 0
        && swapped == 0
        && journal_gaps == 0
        && h.ring.dropped_frames == 0;
    println!(
        "verdict: {}",
        if ok {
            if driver_processing || min_corr <= 0.99 {
                "PASS (capture = linear filter of playback above the fidelity cutoff, no gaps, channels correct; driver effects alter the signal before the loopback tap)"
            } else {
                "PASS (faithful full band up to gain, no gaps, channels correct)"
            }
        } else {
            "FAIL (other audio playing, audio enhancements/spatial sound, or a capture defect; see numbers above)"
        }
    );
    if ok && args.out.is_none() {
        std::fs::remove_dir_all(&dir).ok();
    } else {
        println!("journal kept at {}", dir.display());
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
