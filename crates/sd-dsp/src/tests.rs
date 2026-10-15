//! Synthetic echo-path tests. Deterministic, hardware-free; the real-hardware measurement is
//! `sdctl echo-test`.

use super::*;
use crate::metrics::{power_ratio_db, si_sdr};

const RATE: usize = 48_000;
const OUT: usize = 16_000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Speech-like signal: noise through a one-pole low-pass (spectral tilt), modulated by a
/// syllable-rate envelope (~4 Hz), active only inside `active` (seconds).
fn speechlike(seed: u64, seconds: f32, active: &[(f32, f32)], tilt: f32, level: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    let n = (seconds * RATE as f32) as usize;
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            lp = tilt * lp + (1.0 - tilt) * rng.next();
            let on = active.iter().any(|&(a, b)| t >= a && t < b);
            let syll = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
            if on { lp * syll * level * 4.0 } else { 0.0 }
        })
        .collect()
}

/// Echo path: 30 ms delay, direct path plus decaying reflections over ~60 ms, scaled so the
/// echo is `erl_db` below the far-end.
fn echo_path(far: &[f32], erl_db: f32) -> Vec<f32> {
    let mut rng = Rng(77);
    let delay = RATE * 30 / 1000;
    let taps: Vec<(usize, f32)> = std::iter::once((0usize, 1.0f32))
        .chain((1..40).map(|k| {
            (
                k * 72 + (rng.next().abs() * 40.0) as usize,
                rng.next() * (-(k as f32) / 12.0).exp(),
            )
        }))
        .collect();
    let mut y = vec![0.0f32; far.len()];
    for &(off, g) in &taps {
        let d = delay + off;
        for i in d..far.len() {
            y[i] += g * far[i - d];
        }
    }
    let scale = 10f32.powf(-erl_db / 20.0)
        * (crate::metrics::power(far) / crate::metrics::power(&y)).sqrt() as f32;
    y.iter_mut().for_each(|v| *v *= scale);
    y
}

/// Runs mic/ref through a processor in 10 ms frames; returns the 16 kHz output.
fn run(p: &mut Processor, mic: &[f32], far: &[f32]) -> Vec<f32> {
    let (cf, of) = (p.capture_frame_len(), p.output_frame_len());
    let mut out = vec![0.0f32; mic.len() / cf * of];
    let mut ref_frame = vec![0.0f32; p.render_frame_len()];
    for (k, (m, o)) in mic
        .chunks_exact(cf)
        .zip(out.chunks_exact_mut(of))
        .enumerate()
    {
        for (i, s) in far[k * cf..(k + 1) * cf].iter().enumerate() {
            ref_frame[2 * i] = *s;
            ref_frame[2 * i + 1] = *s;
        }
        p.process_render(&ref_frame).unwrap();
        p.process_capture(m, o).unwrap();
    }
    out
}

fn bypass() -> Processor {
    Processor::new(DspConfig {
        echo_cancellation: false,
        noise_suppression: None,
        gain_control: false,
        ..Default::default()
    })
    .unwrap()
}

fn aec_only() -> Processor {
    Processor::new(DspConfig {
        noise_suppression: None,
        gain_control: false,
        ..Default::default()
    })
    .unwrap()
}

fn seg(x: &[f32], a: f32, b: f32) -> &[f32] {
    &x[(a * OUT as f32) as usize..(b * OUT as f32) as usize]
}

/// Removes the processor's latency so `x[i]` lines up with the bypass reference.
fn compensate(x: &[f32], latency: usize) -> Vec<f32> {
    let mut v = x[latency..].to_vec();
    v.resize(x.len(), 0.0);
    v
}

#[test]
fn measured_latency_matches_known_value() {
    let lat = Processor::measure_latency(&DspConfig::default()).unwrap();
    assert_eq!(lat, 128, "AEC path latency at 48 kHz -> 16 kHz");
    let bypass_cfg = DspConfig {
        echo_cancellation: false,
        noise_suppression: None,
        gain_control: false,
        ..Default::default()
    };
    assert_eq!(Processor::measure_latency(&bypass_cfg).unwrap(), 0);
}

#[test]
fn rejects_bad_config_and_frames() {
    assert!(
        Processor::new(DspConfig {
            capture_rate: 44_100,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        Processor::new(DspConfig {
            render_channels: 3,
            ..Default::default()
        })
        .is_err()
    );
    let mut p = aec_only();
    let mut out = vec![0.0; 160];
    assert!(p.process_capture(&[0.0; 479], &mut out).is_err());
    assert!(p.process_render(&[0.0; 10]).is_err());
}

/// Far-end talks for 20 s (single talk 0-12 s, double talk 12-18 s). Measures echo
/// suppression and near-end preservation, and prints the numbers for the report.
#[test]
fn aec_suppresses_echo_and_preserves_near_end() {
    // Independent talkers (different syllable rates); far end 0.5-18 s, near end 12-18 s.
    let secs = 20.0;
    let far = talker(1, secs, &[(0.5, 18.0)], 3.7, 0.05);
    let near = talker(2, secs, &[(12.0, 18.0)], 5.3, 0.04);
    let echo = echo_path(&far, 10.0);
    let mut rng = Rng(9);
    let mic: Vec<f32> = (0..far.len())
        .map(|i| echo[i] + near[i] + rng.next() * 3e-4)
        .collect();
    let silent = vec![0.0f32; far.len()];
    let path = || crate::aec16k::Aec16k::new(48_000, 2).unwrap();
    let (full, lin) = run16(&mut path(), &mic, &far);
    let full = compensate(&full, crate::aec16k::FULL_LATENCY);
    let lin = compensate(&lin, crate::aec16k::LINEAR_LATENCY);
    let raw = compensate(
        &run16(&mut path(), &mic, &silent).1,
        crate::aec16k::LINEAR_LATENCY,
    );
    let near16 = compensate(
        &run16(&mut path(), &near, &silent).1,
        crate::aec16k::LINEAR_LATENCY,
    );
    let erle_full = power_ratio_db(seg(&raw, 4.0, 12.0), seg(&full, 4.0, 12.0));
    let erle_lin = power_ratio_db(seg(&raw, 4.0, 12.0), seg(&lin, 4.0, 12.0));
    let dt = |x: &[f32]| si_sdr(seg(x, 13.0, 18.0), seg(&near16, 13.0, 18.0));
    println!(
        "synthetic: single-talk ERLE full {erle_full:.1} dB, linear {erle_lin:.1} dB; double-talk SI-SDR raw {:.1}, full {:.1}, linear {:.1} dB",
        dt(&raw),
        dt(&full),
        dt(&lin)
    );
    // The design guarantees (see docs/m2-report.md): the full output removes single-talk echo
    // strongly; the linear output keeps more of the near end than the full output in double
    // talk (AEC3's suppressor ducks the near end, in sonora and in the C++ original alike).
    assert!(
        erle_full >= 25.0,
        "full-output single-talk ERLE {erle_full:.1} dB below 25 dB"
    );
    assert!(
        erle_lin >= 10.0,
        "linear-output single-talk ERLE {erle_lin:.1} dB"
    );
    assert!(
        dt(&lin) > dt(&full),
        "linear must preserve double talk better than full"
    );
}

#[test]
fn transparent_without_far_end() {
    let secs = 6.0;
    let near = speechlike(3, secs, &[(0.5, 5.5)], 0.3, 0.05);
    let silent = vec![0.0f32; near.len()];
    let reference = run(&mut bypass(), &near, &silent);
    let mut p = aec_only();
    let lat = Processor::measure_latency(p.config()).unwrap();
    let out = compensate(&run(&mut p, &near, &silent), lat);
    let v = si_sdr(seg(&out, 1.0, 5.5), seg(&reference, 1.0, 5.5));
    assert!(v > 20.0, "near-end only, no echo: SI-SDR {v:.1} dB");
}

#[test]
#[ignore]
fn debug_double_talk() {
    for (label, phase) in [("in-phase syllables", 0.0f32), ("offset syllables", 0.125)] {
        let secs = 20.0;
        let far = speechlike(1, secs, &[(0.5, 18.0)], 0.6, 0.05);
        let near_raw = speechlike(2, secs + phase, &[(12.0 + phase, 18.0 + phase)], 0.3, 0.05);
        let shift = (phase * RATE as f32) as usize;
        let near: Vec<f32> = near_raw[shift..shift + far.len()].to_vec();
        let echo = echo_path(&far, 10.0);
        let mic: Vec<f32> = (0..far.len()).map(|i| echo[i] + near[i]).collect();
        let silent = vec![0.0; far.len()];
        let near_ref = run(&mut bypass(), &near, &silent);
        let echo_ref = run(&mut bypass(), &echo, &silent);
        let mut p = aec_only();
        let lat = Processor::measure_latency(p.config()).unwrap();
        let out = compensate(&run(&mut p, &mic, &far), lat);
        let (o, n, e) = (
            seg(&out, 13.0, 18.0),
            seg(&near_ref, 13.0, 18.0),
            seg(&echo_ref, 13.0, 18.0),
        );
        let dot: f64 = o.iter().zip(n).map(|(&a, &b)| a as f64 * b as f64).sum();
        let nn: f64 = n.iter().map(|&b| (b as f64).powi(2)).sum();
        let alpha = dot / nn;
        let resid: Vec<f32> = o
            .iter()
            .zip(n)
            .map(|(&a, &b)| a - (alpha as f32) * b)
            .collect();
        println!(
            "{label}: near gain {:+.1} dB (alpha {alpha:.3}), residual {:.1} dB rel. echo, echo-to-near in mic {:.1} dB, SI-SDR {:.1}",
            20.0 * alpha.abs().log10(),
            power_ratio_db(&resid, e),
            power_ratio_db(e, n),
            si_sdr(o, n)
        );
    }
}

fn load_speech(name: &str) -> Option<Vec<f32>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/data")
        .join(name);
    let r = hound::WavReader::open(path).ok()?;
    assert_eq!(r.spec().sample_rate, RATE as u32);
    Some(
        r.into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    )
}

/// Real speech (Windows TTS, generated by tools/testdata/make_tts.ps1) over the synthetic echo
/// path. Ignored by default: needs the local bench/data files.
#[test]
#[ignore]
fn speech_double_talk_report() {
    let (Some(far), Some(near_src)) = (load_speech("far_en.wav"), load_speech("near_en.wav"))
    else {
        println!("bench/data speech files missing; run tools/testdata/make_tts.ps1");
        return;
    };
    let start = 12 * RATE;
    let mut near = vec![0.0f32; far.len()];
    for (i, v) in near_src.iter().enumerate().take(far.len() - start) {
        near[start + i] = *v;
    }
    for erl in [6.0f32, 10.0, 20.0] {
        let echo = echo_path(&far, erl);
        let mic: Vec<f32> = (0..far.len()).map(|i| echo[i] + near[i]).collect();
        let silent = vec![0.0; far.len()];
        let raw = run(&mut bypass(), &mic, &far);
        let near_ref = run(&mut bypass(), &near, &silent);
        let echo_ref = run(&mut bypass(), &echo, &silent);
        let mut p = aec_only();
        let lat = Processor::measure_latency(p.config()).unwrap();
        let out = compensate(&run(&mut p, &mic, &far), lat);
        let erle = power_ratio_db(seg(&raw, 4.0, 12.0), seg(&out, 4.0, 12.0));
        let end = (far.len() / RATE) as f32 - 1.0;
        let (o, n, e) = (
            seg(&out, 13.0, end),
            seg(&near_ref, 13.0, end),
            seg(&echo_ref, 13.0, end),
        );
        let dot: f64 = o.iter().zip(n).map(|(&a, &b)| a as f64 * b as f64).sum();
        let nn: f64 = n.iter().map(|&b| (b as f64).powi(2)).sum();
        let alpha = dot / nn;
        let resid: Vec<f32> = o
            .iter()
            .zip(n)
            .map(|(&a, &b)| a - (alpha as f32) * b)
            .collect();
        println!(
            "speech, ERL {erl} dB: single-talk ERLE {erle:.1} dB | double talk: near gain {:+.1} dB, residual {:.1} dB rel. echo, SI-SDR raw {:.1} -> {:.1} dB | backend {:?}",
            20.0 * alpha.abs().log10(),
            power_ratio_db(&resid, e),
            si_sdr(seg(&raw, 13.0, end), n),
            si_sdr(o, n),
            p.stats()
        );
    }
}

/// Runs the C++ WebRTC reference harness (bench/webrtc-cpp/apm_cli.exe) on the same input.
fn run_cxx(mic: &[f32], far: &[f32], aec: bool) -> Option<Vec<f32>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/webrtc-cpp");
    let exe = dir.join("apm_cli.exe");
    if !exe.exists() {
        return None;
    }
    let tmp = std::env::temp_dir().join(format!("apdsp-cxx-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).ok()?;
    let write = |name: &str, v: &[f32]| {
        let p = tmp.join(name);
        std::fs::write(
            &p,
            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        p
    };
    let m = write("mic.f32", mic);
    let stereo: Vec<f32> = far.iter().flat_map(|&s| [s, s]).collect();
    let r = write("ref.f32", &stereo);
    let o = tmp.join("out.f32");
    let st = std::process::Command::new(exe)
        .args([m.as_os_str(), r.as_os_str(), o.as_os_str()])
        .args([if aec { "1" } else { "0" }, "-1", "0", "1"])
        .output()
        .ok()?;
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    let bytes = std::fs::read(&o).ok()?;
    std::fs::remove_dir_all(&tmp).ok();
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

struct DtReport {
    erle: f64,
    near_gain_db: f64,
    resid_rel_echo_db: f64,
    sisdr: f64,
}

fn double_talk_metrics(
    out: &[f32],
    raw: &[f32],
    near_ref: &[f32],
    echo_ref: &[f32],
    end: f32,
) -> DtReport {
    let erle = power_ratio_db(seg(raw, 4.0, 12.0), seg(out, 4.0, 12.0));
    let (o, n, e) = (
        seg(out, 13.0, end),
        seg(near_ref, 13.0, end),
        seg(echo_ref, 13.0, end),
    );
    let dot: f64 = o.iter().zip(n).map(|(&a, &b)| a as f64 * b as f64).sum();
    let nn: f64 = n.iter().map(|&b| (b as f64).powi(2)).sum();
    let alpha = dot / nn;
    let resid: Vec<f32> = o
        .iter()
        .zip(n)
        .map(|(&a, &b)| a - (alpha as f32) * b)
        .collect();
    DtReport {
        erle,
        near_gain_db: 20.0 * alpha.abs().log10(),
        resid_rel_echo_db: power_ratio_db(&resid, e),
        sisdr: si_sdr(o, n),
    }
}

/// sonora vs the original WebRTC C++ on identical real-speech double-talk input.
/// Ignored: needs bench/data speech and bench/webrtc-cpp/apm_cli.exe.
#[test]
#[ignore]
fn compare_sonora_with_webrtc_cpp() {
    let (Some(far), Some(near_src)) = (load_speech("far_en.wav"), load_speech("near_en.wav"))
    else {
        println!("speech files missing");
        return;
    };
    let start = 12 * RATE;
    let mut near = vec![0.0f32; far.len()];
    for (i, v) in near_src.iter().enumerate().take(far.len() - start) {
        near[start + i] = *v;
    }
    let end = (far.len() / RATE) as f32 - 1.0;
    // Latency of each engine's AEC path, from a probe (near-end noise, no far-end).
    let probe = speechlike(5, 4.0, &[(0.0, 4.0)], 0.0, 0.05);
    let zeros = vec![0.0f32; probe.len()];
    let Some(cxx_on) = run_cxx(&probe, &zeros, true) else {
        println!("apm_cli.exe missing; see bench/webrtc-cpp/README.md");
        return;
    };
    let cxx_off = run_cxx(&probe, &zeros, false).unwrap();
    let cxx_lat = crate::metrics::find_lag(&cxx_on[OUT..], &cxx_off[OUT..], 320, OUT);
    let rust_lat = Processor::measure_latency(&DspConfig {
        noise_suppression: None,
        gain_control: false,
        ..Default::default()
    })
    .unwrap();
    println!("AEC path latency @16k: sonora {rust_lat} samples, C++ {cxx_lat} samples");

    for erl in [6.0f32, 10.0, 20.0] {
        let echo = echo_path(&far, erl);
        let mic: Vec<f32> = (0..far.len()).map(|i| echo[i] + near[i]).collect();
        let silent = vec![0.0; far.len()];
        let raw = run(&mut bypass(), &mic, &far);
        let near_ref = run(&mut bypass(), &near, &silent);
        let echo_ref = run(&mut bypass(), &echo, &silent);
        let rust = compensate(&run(&mut aec_only(), &mic, &far), rust_lat);
        let cxx = compensate(&run_cxx(&mic, &far, true).unwrap(), cxx_lat);
        for (name, out) in [("sonora", &rust), ("webrtc-cpp", &cxx)] {
            let r = double_talk_metrics(out, &raw, &near_ref, &echo_ref, end);
            println!(
                "ERL {erl:>4} dB  {name:<10}  single-talk ERLE {:>5.1} dB | double talk: near {:+.1} dB, residual {:+.1} dB rel. echo, SI-SDR {:.1} dB (raw {:.1})",
                r.erle,
                r.near_gain_db,
                r.resid_rel_echo_db,
                r.sisdr,
                si_sdr(seg(&raw, 13.0, end), seg(&near_ref, 13.0, end))
            );
        }
        println!(
            "ERL {erl:>4} dB  sonora vs C++ output SI-SDR (agreement): {:.1} dB",
            si_sdr(seg(&rust, 2.0, end), seg(&cxx, 2.0, end))
        );
    }
}

fn run16(a: &mut crate::aec16k::Aec16k, mic: &[f32], far: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = a.input_frame_len();
    let frames = mic.len() / n;
    let (mut full, mut lin) = (vec![0.0f32; frames * 160], vec![0.0f32; frames * 160]);
    let mut rf = vec![0.0f32; n * 2];
    for k in 0..frames {
        for (i, s) in far[k * n..(k + 1) * n].iter().enumerate() {
            rf[2 * i] = *s;
            rf[2 * i + 1] = *s;
        }
        a.process(
            &mic[k * n..(k + 1) * n],
            &rf,
            &mut full[k * 160..(k + 1) * 160],
            &mut lin[k * 160..(k + 1) * 160],
        )
        .unwrap();
    }
    (full, lin)
}

/// Full vs linear AEC3 output at 16 kHz on real-speech double talk. Ignored (needs bench/data).
#[test]
#[ignore]
fn linear_vs_full_aec_report() {
    let (Some(far), Some(near_src)) = (load_speech("far_en.wav"), load_speech("near_en.wav"))
    else {
        println!("speech files missing");
        return;
    };
    let start = 12 * RATE;
    let mut near = vec![0.0f32; far.len()];
    for (i, v) in near_src.iter().enumerate().take(far.len() - start) {
        near[start + i] = *v;
    }
    let end = (far.len() / RATE) as f32 - 1.0;
    // Latency of the 16 kHz path vs the bypass resampler, from a near-end-only probe.
    let probe = speechlike(5, 4.0, &[(0.0, 4.0)], 0.0, 0.05);
    let zeros = vec![0.0f32; probe.len()];
    let (_, lin_probe) = run16(
        &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
        &probe,
        &zeros,
    );
    let base_probe = run(&mut bypass(), &probe, &zeros);
    let lat = crate::metrics::find_lag(&lin_probe[OUT..], &base_probe[OUT..], 320, OUT);
    let rust_lat = Processor::measure_latency(&DspConfig {
        noise_suppression: None,
        gain_control: false,
        ..Default::default()
    })
    .unwrap();
    println!("latency @16k vs bypass: 48k pipeline {rust_lat}, aec16k {lat} samples");

    for erl in [6.0f32, 10.0, 20.0] {
        let echo = echo_path(&far, erl);
        let mic: Vec<f32> = (0..far.len()).map(|i| echo[i] + near[i]).collect();
        let silent = vec![0.0; far.len()];
        let raw = run(&mut bypass(), &mic, &far);
        let near_ref = run(&mut bypass(), &near, &silent);
        let echo_ref = run(&mut bypass(), &echo, &silent);
        let p48 = compensate(&run(&mut aec_only(), &mic, &far), rust_lat);
        let mut a = crate::aec16k::Aec16k::new(48_000, 2).unwrap();
        let (full, lin) = run16(&mut a, &mic, &far);
        let (full, lin) = (
            compensate(&full, crate::aec16k::FULL_LATENCY),
            compensate(&lin, lat),
        );
        // References through the same 16 kHz path (same resampler and HPF), far-end silent.
        let near16 = compensate(
            &run16(
                &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
                &near,
                &silent,
            )
            .1,
            lat,
        );
        let echo16 = compensate(
            &run16(
                &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
                &echo,
                &silent,
            )
            .1,
            lat,
        );
        let raw16 = compensate(
            &run16(
                &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
                &mic,
                &silent,
            )
            .1,
            lat,
        );
        println!(
            "    16k path transparency (near through path vs 48k bypass ref) SI-SDR {:.1} dB; raw16 double-talk SI-SDR {:.1} dB",
            si_sdr(seg(&near16, 13.0, end), seg(&near_ref, 13.0, end)),
            si_sdr(seg(&raw16, 13.0, end), seg(&near16, 13.0, end))
        );
        // Hybrid for ASR: rerun raw outputs through the mixer (aligned to FULL_LATENCY).
        let (full_raw, lin_raw) = run16(
            &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
            &mic,
            &far,
        );
        let hybrid = |thr: f32| {
            let mut mixer = crate::aec16k::AsrMixer::default();
            mixer.echo_threshold_db = thr;
            let mut hyb = vec![0.0f32; full_raw.len()];
            for k in 0..full_raw.len() / 160 {
                let r = k * 160..(k + 1) * 160;
                mixer.push(&lin_raw[r.clone()], &full_raw[r.clone()], &mut hyb[r]);
            }
            compensate(&hyb, crate::aec16k::FULL_LATENCY)
        };
        let (h10, h20, h30) = (hybrid(10.0), hybrid(20.0), hybrid(30.0));
        for (name, out) in [
            ("16k full*", &full),
            ("16k linear*", &lin),
            ("hybrid@10*", &h10),
            ("hybrid@20*", &h20),
            ("hybrid@30*", &h30),
        ] {
            let r = double_talk_metrics(out, &raw16, &near16, &echo16, end);
            println!(
                "    {name:<11} single-talk ERLE {:>5.1} dB | double talk: near {:+.1} dB, residual {:+.1} dB rel. echo, SI-SDR {:.1} dB  (*matched refs)",
                r.erle, r.near_gain_db, r.resid_rel_echo_db, r.sisdr
            );
        }
        println!(
            "ERL {erl:>4} dB  raw SI-SDR in double talk {:.1} dB; aec16k stats {:?}",
            si_sdr(seg(&raw, 13.0, end), seg(&near_ref, 13.0, end)),
            a.stats()
        );
        for (name, out) in [
            ("48k full", &p48),
            ("16k full", &full),
            ("16k linear", &lin),
        ] {
            let r = double_talk_metrics(out, &raw, &near_ref, &echo_ref, end);
            println!(
                "    {name:<10}  single-talk ERLE {:>5.1} dB | double talk: near {:+.1} dB, residual {:+.1} dB rel. echo, SI-SDR {:.1} dB",
                r.erle, r.near_gain_db, r.resid_rel_echo_db, r.sisdr
            );
        }
    }
}

/// Locks the measured latencies of both 16 kHz outputs (near-end only, so AEC is transparent).
#[test]
fn aec16k_latencies() {
    let probe = speechlike(5, 4.0, &[(0.0, 4.0)], 0.0, 0.05);
    let zeros = vec![0.0f32; probe.len()];
    let (full, lin) = run16(
        &mut crate::aec16k::Aec16k::new(48_000, 2).unwrap(),
        &probe,
        &zeros,
    );
    let base = run(&mut bypass(), &probe, &zeros);
    let lag = |x: &[f32]| crate::metrics::find_lag(&x[OUT..], &base[OUT..], 320, OUT);
    assert_eq!(lag(&lin), crate::aec16k::LINEAR_LATENCY);
    assert_eq!(lag(&full), crate::aec16k::FULL_LATENCY);
}

/// Speech-like signal with its own syllable rate (independent talkers must not share rhythm).
fn talker(seed: u64, seconds: f32, active: &[(f32, f32)], syll_hz: f32, level: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    let n = (seconds * RATE as f32) as usize;
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            lp = 0.5 * lp + 0.5 * rng.next();
            let on = active.iter().any(|&(a, b)| t >= a && t < b);
            let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * syll_hz * t).sin()).powi(2);
            if on { lp * env * level * 4.0 } else { 0.0 }
        })
        .collect()
}

fn run_mic(p: &mut crate::mic::MicProcessor, mic: &[f32], far: &[f32]) -> (Vec<f32>, Vec<bool>) {
    let n = 480;
    let frames = mic.len() / n;
    let mut out = vec![0.0f32; frames * 160];
    let mut active = Vec::with_capacity(frames);
    let mut rf = vec![0.0f32; n * 2];
    for k in 0..frames {
        for (i, s) in far[k * n..(k + 1) * n].iter().enumerate() {
            rf[2 * i] = *s;
            rf[2 * i + 1] = *s;
        }
        p.process(
            &mic[k * n..(k + 1) * n],
            &rf,
            &mut out[k * 160..(k + 1) * 160],
        )
        .unwrap();
        active.push(p.status().aec_active);
    }
    (out, active)
}

fn mic_cfg(aec: crate::mic::AecMode) -> crate::mic::MicConfig {
    crate::mic::MicConfig {
        aec,
        gain_control: false,
        noise_suppression: None,
        ..Default::default()
    }
}

#[test]
fn echo_gate_turns_off_without_echo_and_passes_near_end() {
    // Far end talks for 40 s, but the mic hears only the local talker (headphones / device AEC).
    let secs = 40.0;
    let far = talker(11, secs, &[(0.5, 39.5)], 3.7, 0.05);
    let near = talker(12, secs, &[(5.0, 39.5)], 5.3, 0.03);
    let mut p = crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::Auto)).unwrap();
    let lat = p.latency();
    let (out, active) = run_mic(&mut p, &near, &far);
    let out = compensate(&out, lat);
    let reference = compensate(
        &run_mic(
            &mut crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::Off)).unwrap(),
            &near,
            &far,
        )
        .0,
        lat,
    );
    println!("no echo: correlation {:?}", p.status().echo_correlation);
    assert_eq!(p.status().echo_state, crate::mic::EchoState::NoEcho);
    assert!(
        active.iter().all(|&a| !a),
        "AEC must never engage without echo"
    );
    let v = si_sdr(seg(&out, 2.0, 39.0), seg(&reference, 2.0, 39.0));
    assert!(
        v > 40.0,
        "with AEC gated off the near-end must pass through: SI-SDR {v:.1} dB"
    );
}

#[test]
fn echo_gate_stays_on_with_echo_and_removes_it() {
    let secs = 30.0;
    let far = talker(21, secs, &[(0.5, 29.5)], 3.7, 0.05);
    let echo = echo_path(&far, 10.0);
    let mut p = crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::Auto)).unwrap();
    let lat = p.latency();
    let (out, active) = run_mic(&mut p, &echo, &far);
    let out = compensate(&out, lat);
    let raw = compensate(
        &run_mic(
            &mut crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::Off)).unwrap(),
            &echo,
            &far,
        )
        .0,
        lat,
    );
    let erle = power_ratio_db(seg(&raw, 10.0, 29.0), seg(&out, 10.0, 29.0));
    println!(
        "echo: ERLE {erle:.1} dB, correlation {:?}",
        p.status().echo_correlation
    );
    assert_eq!(p.status().echo_state, crate::mic::EchoState::Echo);
    let first_on = active
        .iter()
        .position(|&a| a)
        .expect("gate must engage with echo");
    println!("echo: AEC engaged after {:.1} s", first_on as f32 / 100.0);
    assert!(first_on <= 500, "engaged after {first_on} frames");
    assert!(
        active[first_on..].iter().all(|&a| a),
        "gate must stay on while echo is present"
    );
    assert!(erle > 25.0, "ERLE {erle:.1} dB");
}

#[test]
fn echo_gate_unit() {
    let mut g = crate::mic::EchoGate::new(false);
    assert_eq!(
        g.state(),
        crate::mic::EchoState::Unknown,
        "undecided until evidence"
    );
    // Correlated envelopes (echo): mic follows ref 50 ms later.
    let env: Vec<f32> = (0..1_500)
        .map(|i| -40.0 + 20.0 * ((i as f32) * 0.07).sin() + ((i * 7919) % 13) as f32)
        .collect();
    for i in 0..1_500 {
        g.push(if i >= 5 { env[i - 5] - 30.0 } else { -90.0 }, env[i]);
    }
    assert!(
        g.enabled(),
        "echo must enable the gate: {:?}",
        g.last_correlation()
    );
    // Then independent envelopes for 20 s.
    for i in 0..2_000 {
        g.push(-50.0 + ((i * 104_729) % 17) as f32, env[i % 1_500]);
    }
    assert!(
        !g.enabled(),
        "uncorrelated mic must disable the gate: {:?}",
        g.last_correlation()
    );
}

/// The bypass and AEC paths must stay sample-aligned (the crossfade mixes them): with AEC on
/// and no far end, the AEC output must match the bypass output.
#[test]
fn bypass_and_aec_paths_are_aligned() {
    let near = talker(31, 6.0, &[(0.2, 5.8)], 5.3, 0.04);
    let silent = vec![0.0f32; near.len()];
    let on = run_mic(
        &mut crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::On)).unwrap(),
        &near,
        &silent,
    )
    .0;
    let off = run_mic(
        &mut crate::mic::MicProcessor::new(mic_cfg(crate::mic::AecMode::Off)).unwrap(),
        &near,
        &silent,
    )
    .0;
    let v = si_sdr(seg(&on, 1.0, 5.5), seg(&off, 1.0, 5.5));
    assert!(
        v > 40.0,
        "AEC path vs bypass path SI-SDR {v:.1} dB (misaligned?)"
    );
}
