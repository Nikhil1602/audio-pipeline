use super::*;

fn model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/silero_vad_16k_op15.onnx")
}

fn vad() -> Option<SileroVad> {
    set_runtime_library(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models")
            .join(RUNTIME_LIBRARY),
    );
    match SileroVad::new(model_path()) {
        Ok(v) => Some(v),
        Err(e @ (VadError::MissingModel(_) | VadError::RuntimeMissing(_))) => {
            println!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn probs_to_segments(probs: &[f32], cfg: SegmenterConfig) -> Vec<Segment> {
    let mut s = Segmenter::new(cfg);
    let mut out: Vec<Segment> = probs.iter().filter_map(|&p| s.push(p)).collect();
    out.extend(s.flush());
    out
}

#[test]
fn segmenter_basic_hysteresis_and_padding() {
    // 1 s silence, 1 s speech, 1 s silence (31.25 chunks per second).
    let mut probs = vec![0.05f32; 31];
    probs.extend(vec![0.9f32; 31]);
    probs.extend(vec![0.05f32; 31]);
    let segs = probs_to_segments(&probs, SegmenterConfig::default());
    assert_eq!(segs.len(), 1);
    let pad = 1_600; // 100 ms
    assert_eq!(
        segs[0],
        Segment {
            start: 31 * 512 - pad,
            end: 62 * 512 + pad
        }
    );
}

#[test]
fn segmenter_drops_short_bursts_and_bridges_short_pauses() {
    let cfg = SegmenterConfig::default();
    // 128 ms burst: shorter than 250 ms minimum -> dropped.
    let mut probs = vec![0.0f32; 20];
    probs.extend(vec![0.9f32; 4]);
    probs.extend(vec![0.0f32; 20]);
    assert!(probs_to_segments(&probs, cfg).is_empty());
    // Speech with a 160 ms pause (< 300 ms min silence) and a dip to 0.4 (between thresholds)
    // stays one segment.
    let mut probs = vec![0.9f32; 20];
    probs.extend(vec![0.1f32; 5]);
    probs.extend(vec![0.9f32; 10]);
    probs.extend(vec![0.4f32; 30]);
    probs.extend(vec![0.9f32; 10]);
    probs.extend(vec![0.0f32; 20]);
    assert_eq!(probs_to_segments(&probs, cfg).len(), 1);
}

#[test]
fn segmenter_flush_closes_open_segment() {
    let mut s = Segmenter::new(SegmenterConfig::default());
    for _ in 0..30 {
        assert!(s.push(0.95).is_none());
    }
    assert!(s.in_speech());
    let seg = s.flush().expect("open segment");
    assert_eq!(seg.start, 0);
    assert_eq!(seg.end, 30 * 512 + 1_600);
}

#[test]
fn silence_and_noise_are_not_speech() {
    let Some(mut v) = vad() else { return };
    let mut probs = Vec::new();
    v.push(&vec![0.0f32; 16_000 * 3], &mut probs).unwrap();
    assert_eq!(probs.len(), 16_000 * 3 / 512);
    assert!(
        probs.iter().all(|&p| p < 0.2),
        "silence max prob {:?}",
        probs.iter().cloned().fold(0.0, f32::max)
    );
    v.reset();
    let mut rng = Rng(7);
    let noise: Vec<f32> = (0..16_000 * 3).map(|_| rng.next() * 0.03).collect();
    probs.clear();
    v.push(&noise, &mut probs).unwrap();
    let speechy = probs.iter().filter(|&&p| p >= 0.5).count();
    assert!(
        speechy * 20 < probs.len(),
        "{speechy} of {} white-noise chunks flagged",
        probs.len()
    );
}

/// Output preallocation (the leak fix) must not change any probability.
#[test]
fn preallocated_outputs_give_identical_probabilities() {
    let Some(mut a) = vad() else { return };
    let mem = SessionMemory {
        cpu_arena: true,
        memory_pattern: true,
        optimize: true,
        preallocate_outputs: false,
    };
    let mut b = SileroVad::with_memory(model_path(), mem).unwrap();
    let mut rng = Rng(9);
    let mut x: Vec<f32> = (0..16_000 * 5)
        .map(|i| rng.next() * 0.2 * ((i as f32 * 0.0007).sin()))
        .collect();
    if let Some(speech) = load_speech_16k("near_en.wav") {
        x.extend_from_slice(&speech);
    }
    let (mut pa, mut pb) = (Vec::new(), Vec::new());
    for c in x.chunks(160) {
        a.push(c, &mut pa).unwrap();
        b.push(c, &mut pb).unwrap();
    }
    assert!(pa.len() > 100);
    assert_eq!(pa, pb);
}

#[test]
fn chunking_is_independent_of_push_sizes() {
    let Some(mut a) = vad() else { return };
    let mut b = vad().unwrap();
    let mut rng = Rng(3);
    let x: Vec<f32> = (0..16_000 * 2)
        .map(|i| rng.next() * 0.1 * ((i as f32 * 0.001).sin()))
        .collect();
    let (mut pa, mut pb) = (Vec::new(), Vec::new());
    a.push(&x, &mut pa).unwrap();
    for c in x.chunks(160) {
        b.push(c, &mut pb).unwrap();
    }
    assert_eq!(
        pa, pb,
        "results must not depend on how audio is split into pushes"
    );
}

fn load_speech_16k(name: &str) -> Option<Vec<f32>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/data")
        .join(name);
    let r = hound::WavReader::open(path).ok()?;
    assert_eq!(r.spec().sample_rate, 48_000);
    let x: Vec<f32> = r
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut rs = sonora_common_audio::push_sinc_resampler::PushSincResampler::new(480, 160);
    let mut out = vec![0.0f32; x.len() / 480 * 160];
    for (i, o) in x.chunks_exact(480).zip(out.chunks_exact_mut(160)) {
        rs.resample(i, o);
    }
    Some(out)
}

/// Frame-level accuracy on real speech with additive noise at several SNRs. Ground truth: 32 ms
/// frames of the clean speech whose energy is within 40 dB of the loudest frame. Ignored: needs
/// bench/data (tools/testdata/make_tts.ps1).
#[test]
#[ignore]
fn speech_in_noise_report() {
    let (Some(far), Some(near)) = (
        load_speech_16k("far_en.wav"),
        load_speech_16k("near_en.wav"),
    ) else {
        println!("speech files missing");
        return;
    };
    let mut clean = far;
    clean.extend(vec![0.0f32; 16_000 * 3]);
    clean.extend(near);
    let frames: Vec<f64> = clean
        .chunks_exact(CHUNK)
        .map(|c| c.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / CHUNK as f64)
        .collect();
    let peak = frames.iter().cloned().fold(0.0, f64::max);
    let truth: Vec<bool> = frames.iter().map(|&e| e > peak * 1e-4).collect();
    let speech_power = frames
        .iter()
        .zip(&truth)
        .filter(|(_, t)| **t)
        .map(|(e, _)| e)
        .sum::<f64>()
        / truth.iter().filter(|t| **t).count() as f64;
    println!(
        "frames {}, speech frames {}",
        truth.len(),
        truth.iter().filter(|t| **t).count()
    );
    for snr in [f64::INFINITY, 20.0, 10.0, 5.0, 0.0] {
        // Pink-ish noise (one-pole low-pass of white) scaled to the requested SNR.
        let mut rng = Rng(42);
        let mut lp = 0.0f32;
        let mut noise: Vec<f32> = (0..clean.len())
            .map(|_| {
                lp = 0.9 * lp + 0.1 * rng.next();
                lp
            })
            .collect();
        let np = noise.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / noise.len() as f64;
        let g = if snr.is_finite() {
            (speech_power / np / 10f64.powf(snr / 10.0)).sqrt() as f32
        } else {
            0.0
        };
        noise.iter_mut().for_each(|v| *v *= g);
        let mixed: Vec<f32> = clean.iter().zip(&noise).map(|(a, b)| a + b).collect();
        let mut v = vad().unwrap();
        let mut probs = Vec::new();
        v.push(&mixed, &mut probs).unwrap();
        let (mut tp, mut fp, mut fn_) = (0, 0, 0);
        for (p, t) in probs.iter().zip(&truth) {
            match (*p >= 0.5, *t) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                _ => {}
            }
        }
        let prec = tp as f64 / (tp + fp).max(1) as f64;
        let rec = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = 2.0 * prec * rec / (prec + rec).max(1e-9);
        let segs = probs_to_segments(&probs, SegmenterConfig::default());
        println!(
            "SNR {snr:>5} dB: frame precision {prec:.3} recall {rec:.3} F1 {f1:.3}; {} segments",
            segs.len()
        );
    }
}

/// Throughput: real-time factor of the VAD on one thread.
#[test]
#[ignore]
fn vad_speed_report() {
    let Some(mut v) = vad() else { return };
    let mut rng = Rng(5);
    let x: Vec<f32> = (0..16_000 * 60).map(|_| rng.next() * 0.05).collect();
    let mut probs = Vec::new();
    let t = std::time::Instant::now();
    v.push(&x, &mut probs).unwrap();
    let el = t.elapsed().as_secs_f64();
    println!(
        "VAD: 60 s of audio in {:.3} s -> {:.4}x real time ({:.1} us per 32 ms chunk)",
        el,
        el / 60.0,
        el * 1e6 / probs.len() as f64
    );
}
