use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use sd_dsp::{DspConfig, Processor};

/// Per-10 ms cost of the full capture pipeline (render + capture), 48 kHz in, 16 kHz out.
fn full_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("dsp");
    for (name, cfg) in [
        ("aec_ns_agc_hpf", DspConfig::default()),
        (
            "aec_only",
            DspConfig {
                noise_suppression: None,
                gain_control: false,
                ..Default::default()
            },
        ),
        (
            "bypass",
            DspConfig {
                echo_cancellation: false,
                noise_suppression: None,
                gain_control: false,
                ..Default::default()
            },
        ),
    ] {
        let mut p = Processor::new(cfg).unwrap();
        let mic: Vec<f32> = (0..480).map(|i| ((i * 37) % 100) as f32 / 1000.0).collect();
        let far: Vec<f32> = (0..960).map(|i| ((i * 53) % 100) as f32 / 1000.0).collect();
        let mut out = vec![0.0f32; 160];
        g.bench_function(name, |b| {
            b.iter(|| {
                p.process_render(black_box(&far)).unwrap();
                p.process_capture(black_box(&mic), &mut out).unwrap();
            })
        });
    }
    g.finish();
}

/// Per-10 ms cost of the production mic chain (16 kHz AEC3 + gate + mixer + bypass path).
fn mic_processor(c: &mut Criterion) {
    use sd_dsp::mic::{AecMode, MicConfig, MicProcessor};
    let mut g = c.benchmark_group("mic");
    for (name, aec, agc) in [
        ("aec_on_agc", AecMode::On, true),
        ("aec_on", AecMode::On, false),
        ("aec_off", AecMode::Off, false),
    ] {
        let mut p = MicProcessor::new(MicConfig {
            aec,
            gain_control: agc,
            ..Default::default()
        })
        .unwrap();
        let mic: Vec<f32> = (0..480).map(|i| ((i * 37) % 100) as f32 / 1000.0).collect();
        let far: Vec<f32> = (0..960).map(|i| ((i * 53) % 100) as f32 / 1000.0).collect();
        let mut out = vec![0.0f32; 160];
        g.bench_function(name, |b| {
            b.iter(|| {
                p.process(black_box(&mic), black_box(&far), &mut out)
                    .unwrap()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, full_pipeline, mic_processor);
criterion_main!(benches);
