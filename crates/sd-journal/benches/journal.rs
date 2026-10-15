use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sd_core::{BlockFlags, BlockMeta, HostTime, Source, StreamFormat};
use sd_journal::{Codec, JournalConfig, JournalWriter};

/// Cost of journaling one 10 ms block of 48 kHz stereo (the heaviest live source), including
/// the amortized 1 s fsync, for each codec. The signal is a tone plus noise at about
/// -20 dBFS with different channels, so compression has real work to do.
fn write_10ms_stereo(c: &mut Criterion) {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    // 100 distinct blocks (1 s), cycled.
    let blocks: Vec<Vec<f32>> = (0..100)
        .map(|b| {
            (0..480)
                .flat_map(|i| {
                    let t = (b * 480 + i) as f32 / 48_000.0;
                    let tone = (t * 2.0 * std::f32::consts::PI * 220.0).sin() * 0.1;
                    [tone + noise() * 0.02, tone * 0.7 + noise() * 0.02]
                })
                .collect()
        })
        .collect();

    let mut g = c.benchmark_group("journal");
    g.throughput(Throughput::Elements(480));
    for (name, codec) in [
        ("write_480x2_pcm", Codec::PcmS16le),
        ("write_480x2_lossless", Codec::Lossless),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = JournalConfig::new(vec![(
            Source::System,
            StreamFormat::new(48_000, 2).unwrap(),
        )]);
        cfg.sync_interval = Duration::from_secs(1);
        cfg.codec = codec;
        let mut w = JournalWriter::create(dir.path(), cfg).unwrap();
        let mut frame = 0u64;
        g.bench_function(name, |b| {
            b.iter(|| {
                let meta = BlockMeta {
                    first_frame: frame,
                    frames: 480,
                    host_time: HostTime(frame),
                    flags: BlockFlags::empty(),
                };
                let samples = &blocks[(frame / 480 % 100) as usize];
                w.write(Source::System, black_box(&meta), black_box(samples))
                    .unwrap();
                frame += 480;
            })
        });
        w.finish().unwrap();
    }
    g.finish();
}

criterion_group!(benches, write_10ms_stereo);
criterion_main!(benches);
