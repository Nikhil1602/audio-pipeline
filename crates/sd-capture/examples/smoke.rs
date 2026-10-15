//! Hardware smoke test: lists devices, captures mic + system audio for N seconds, and reports
//! events, stats, peak levels and timeline continuity.
//!
//! cargo run -p sd-capture --example smoke -- 5

use std::time::{Duration, Instant};

use sd_capture::{CaptureConfig, DeviceKind, list_devices, start_capture};
use sd_core::{BlockFlags, Source};

fn main() {
    let secs: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    for kind in [DeviceKind::Input, DeviceKind::Output] {
        println!("{kind:?} devices:");
        for d in list_devices(kind).expect("list devices") {
            println!(
                "  {} {} [{}] mix={:?}",
                if d.is_default { "*" } else { " " },
                d.name,
                d.id,
                d.mix_format.map(|f| (f.sample_rate, f.channels))
            );
        }
    }

    let mut caps = Vec::new();
    for source in [Source::Mic, Source::System] {
        match start_capture(CaptureConfig::new(source)) {
            Ok((h, c)) => caps.push((h, c, 0u64, 0f32, None::<u64>, 0u64, 0u64)),
            Err(e) => println!("{source}: failed to start: {e}"),
        }
    }

    let start = Instant::now();
    let mut buf = Vec::new();
    while start.elapsed() < Duration::from_secs(secs) {
        for (h, c, frames, peak, next, gaps, gapfill) in caps.iter_mut() {
            while let Ok(ev) = h.events().try_recv() {
                println!(
                    "[{:>6.3}s] {}: {ev:?}",
                    start.elapsed().as_secs_f64(),
                    h.source()
                );
            }
            while let Some(meta) = c.pop(&mut buf) {
                if let Some(n) = *next
                    && meta.first_frame != n
                {
                    *gaps += 1;
                }
                *next = Some(meta.first_frame + meta.frames as u64);
                *frames += meta.frames as u64;
                if meta.flags.contains(BlockFlags::GAP_FILL) {
                    *gapfill += meta.frames as u64;
                }
                for &s in &buf {
                    *peak = peak.max(s.abs());
                }
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let wall = start.elapsed().as_secs_f64();
    for (h, c, frames, peak, _, gaps, gapfill) in caps {
        let rate = h.format().sample_rate as f64;
        println!(
            "{}: {:.3} s of audio in {:.3} s wall ({} frames, {} gap-fill), timeline jumps {}, peak {:.1} dBFS, ring {:?}, capture {:?}",
            h.source(),
            frames as f64 / rate,
            wall,
            frames,
            gapfill,
            gaps,
            20.0 * peak.max(1e-9).log10(),
            c.stats(),
            h.stats()
        );
        h.stop();
    }
}
