//! Diagnostic: runs Silero VAD over `minutes` of synthetic audio as fast as possible with the
//! given ONNX Runtime memory options, so memory growth can be sampled externally.
//! `cargo run --release -p sd-vad --example vadmem -- <minutes> <arena> <pattern> <optimize> <prealloc>`
//! (each flag 0 or 1)
use sd_vad::{SessionMemory, SileroVad};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let minutes: u64 = a[0].parse().unwrap();
    let mem = SessionMemory {
        cpu_arena: a[1] == "1",
        memory_pattern: a[2] == "1",
        optimize: a[3] == "1",
        preallocate_outputs: a[4] == "1",
    };
    let mut vad = SileroVad::with_memory("models/silero_vad_16k_op15.onnx", mem).unwrap();
    let mut seed = 1u32;
    let mut frame = vec![0.0f32; 160];
    let mut probs = Vec::new();
    for i in 0..minutes * 6_000 {
        for v in frame.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (seed >> 9) as f32 / (1u32 << 23) as f32 * 0.2 - 0.1;
        }
        probs.clear();
        vad.push(&frame, &mut probs).unwrap();
        if i % 6_000 == 5_999 {
            println!("minute {}", i / 6_000 + 1);
        }
    }
}
