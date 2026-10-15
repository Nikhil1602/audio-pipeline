//! Diagnostic: runs the recorder on synthetic mic + system audio as fast as the pipeline allows
//! (paced at ~20x real time) so memory growth per hour of audio can be measured externally.
//!
//! `cargo run --release -p sd-engine --example leakprobe -- <minutes> <none|proc|vad> <dir>`
//!
//! Prints `minute N` after each simulated minute and keeps draining events and processed audio
//! the way an application would.

use std::time::{Duration, Instant};

use sd_core::ring::audio_ring;
use sd_core::{BlockFlags, HostTime, Source, StreamFormat};
use sd_engine::{ProcessingConfig, Recorder, RecorderConfig, SourceInput};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let minutes: u64 = args[0].parse().unwrap();
    let mode = args[1].as_str();
    let mut cfg = RecorderConfig::new(&args[2]);
    if mode != "none" {
        cfg.processing = Some(ProcessingConfig {
            vad_model: (mode == "vad").then(|| "models/silero_vad_16k_op15.onnx".into()),
            ..Default::default()
        });
    }
    let (pm, cm) = audio_ring(1, 48_000 * 4, 4_096).unwrap();
    let (ps, cs) = audio_ring(2, 48_000 * 4, 4_096).unwrap();
    let (mut pm, mut ps) = (pm, ps);
    let rec = Recorder::from_inputs(
        cfg,
        vec![
            SourceInput {
                source: Source::Mic,
                format: StreamFormat::new(48_000, 1).unwrap(),
                ring: cm,
                capture: None,
            },
            SourceInput {
                source: Source::System,
                format: StreamFormat::new(48_000, 2).unwrap(),
                ring: cs,
                capture: None,
            },
        ],
    )
    .unwrap();
    let processed = rec.processed().cloned();
    let t0 = 1_000_000_000_000u64;
    let mut seed = 1u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 9) as f32 / (1u32 << 23) as f32 * 0.02 - 0.01
    };
    let (mut mic, mut sys) = (vec![0.0f32; 480], vec![0.0f32; 960]);
    let start = Instant::now();
    for i in 0..minutes * 6_000 {
        mic.iter_mut().for_each(|v| *v = noise());
        sys.iter_mut().for_each(|v| *v = noise());
        let t = HostTime(t0 + i * 10_000_000);
        pm.push(&mic, t, BlockFlags::empty());
        ps.push(&sys, t, BlockFlags::empty());
        if i % 100 == 99 {
            // Pace at ~20x real time: 1 s of audio per 50 ms.
            let due = start + Duration::from_millis((i + 1).div_ceil(2));
            while Instant::now() < due {
                let _ = rec.events().try_iter().count();
                if let Some(p) = &processed {
                    let _ = p.try_iter().count();
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        if i % 6_000 == 5_999 {
            println!("minute {}", i / 6_000 + 1);
        }
    }
    drop((pm, ps));
    let s = rec.stop().unwrap();
    println!("done; journaled {} frames", s.health.journaled_frames);
}
