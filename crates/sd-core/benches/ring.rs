use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sd_core::ring::audio_ring;
use sd_core::{BlockFlags, HostTime};

/// Push + pop of one 10 ms stereo 48 kHz block: the per-callback cost on the capture thread
/// plus the matching cost on the DSP thread.
fn push_pop_10ms_stereo(c: &mut Criterion) {
    let (mut p, mut cons) = audio_ring(2, 48_000, 256).unwrap();
    let block = vec![0.25f32; 480 * 2];
    let mut out = vec![0.0f32; 480 * 2];
    let mut g = c.benchmark_group("ring");
    g.throughput(Throughput::Elements(480));
    g.bench_function("push_pop_480x2", |b| {
        b.iter(|| {
            p.push(black_box(&block), HostTime(0), BlockFlags::empty());
            black_box(cons.pop_into(&mut out).unwrap());
        })
    });
    g.finish();
}

criterion_group!(benches, push_pop_10ms_stereo);
criterion_main!(benches);
