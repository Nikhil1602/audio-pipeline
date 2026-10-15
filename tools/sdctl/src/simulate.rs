//! Synthetic end-to-end check of the capture → ring → journal path.
//!
//! Two producer threads stand in for device callbacks: they push 10 ms blocks of a
//! deterministic signal at `speed` x real time (in bursts, like real devices). As in the
//! recording engine, a drain thread empties both rings into a bounded queue and a separate
//! journal thread writes it, so disk stalls (fsync) are absorbed by the queue, not the rings.
//! Afterwards every recovered sample is compared with the signal. This measures the no-loss
//! guarantee and journal throughput without hardware.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sd_core::ring::{RingConsumer, RingProducer, audio_ring};
use sd_core::{BlockFlags, HostTime, Source, StreamFormat};
use sd_journal::{JournalConfig, JournalReader, JournalWriter};

use crate::CliResult;

#[derive(clap::Args)]
pub struct Args {
    /// Simulated session length in seconds.
    #[arg(long, default_value_t = 600)]
    seconds: u64,
    /// Speed relative to real time (e.g. 60 = one simulated minute per wall second).
    #[arg(long, default_value_t = 60.0)]
    speed: f64,
    /// Ring capacity per source, in milliseconds of *real-time* audio. It is multiplied by
    /// `speed` so the wall-clock headroom matches a real session (a 2 s ring at 60x would
    /// otherwise absorb only 33 ms of consumer stall).
    #[arg(long, default_value_t = 2_000)]
    ring_ms: u64,
    /// Drain → journal queue capacity in seconds of real-time audio (scaled by `speed` like
    /// the ring). The recording engine uses at least 30 s.
    #[arg(long, default_value_t = 30)]
    queue_secs: u64,
    /// Stall the journal thread once for this many wall-clock ms after 1 s, as a slow disk or
    /// a long fsync would, to check that the queue absorbs it.
    #[arg(long, default_value_t = 0)]
    stall_ms: u64,
    /// Journal directory (must not exist). Defaults to a new directory under the OS temp dir.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Keep the journal after a successful run.
    #[arg(long)]
    keep: bool,
}

const RATE: u32 = 48_000;
const BLOCK_FRAMES: usize = 480; // 10 ms

fn signal(source: Source, frame: u64, ch: u16) -> f32 {
    // Cheap deterministic pseudo-noise in [-0.9, 0.9]; distinct per source/channel.
    let mut x =
        frame.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((source as u64) << 48) ^ ((ch as u64) << 40);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x & 0xFFFF) as f32 / 65_535.0 * 2.0 - 1.0) * 0.9
}

fn to_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

struct ProducerReport {
    frames: u64,
    dropped_frames: u64,
}

fn produce(
    mut p: RingProducer,
    source: Source,
    channels: u16,
    total_frames: u64,
    speed: f64,
) -> ProducerReport {
    let mut buf = vec![0.0f32; BLOCK_FRAMES * channels as usize];
    let start = Instant::now();
    let mut frame = 0u64;
    while frame < total_frames {
        // Push every block that "should" have arrived by now, then sleep ~1 ms (burst delivery).
        let due = ((start.elapsed().as_secs_f64() * speed * RATE as f64) as u64).min(total_frames);
        while frame < due {
            let n = BLOCK_FRAMES.min((total_frames - frame) as usize);
            for f in 0..n {
                for c in 0..channels {
                    buf[f * channels as usize + c as usize] = signal(source, frame + f as u64, c);
                }
            }
            p.push(
                &buf[..n * channels as usize],
                HostTime::now(),
                BlockFlags::empty(),
            );
            frame += n as u64;
        }
        thread::sleep(Duration::from_millis(1));
    }
    let s = p.stats();
    ProducerReport {
        frames: s.pushed_frames + s.dropped_frames,
        dropped_frames: s.dropped_frames,
    }
}

pub fn run(args: Args) -> CliResult {
    if !args.speed.is_finite() || args.speed <= 0.0 || args.seconds == 0 {
        return Err("--speed must be > 0 and --seconds must be > 0".into());
    }
    let dir = args.out.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("sdctl-sim-{}", HostTime::now().as_ns()))
    });
    let sources = [(Source::Mic, 1u16), (Source::System, 2u16)];
    let total_frames = args.seconds * RATE as u64;
    let ring_frames = (args.ring_ms as f64 * args.speed * RATE as f64 / 1000.0).ceil() as usize;

    let mut cfg = JournalConfig::new(
        sources
            .iter()
            .map(|&(s, ch)| (s, StreamFormat::new(RATE, ch).unwrap()))
            .collect(),
    );
    cfg.segment_seconds = 60;
    let mut writer = JournalWriter::create(&dir, cfg)?;
    let queue_blocks =
        ((args.queue_secs as f64 * args.speed * 100.0).ceil() as usize).max(16) * sources.len();
    let (qtx, qrx) =
        crossbeam_channel::bounded::<(Source, sd_core::BlockMeta, Vec<f32>)>(queue_blocks);

    println!(
        "simulating {} s of mic (48 kHz x1) + system (48 kHz x2) at {}x real time, ring {} ms real-time ({:.0} ms wall)",
        args.seconds, args.speed, args.ring_ms, args.ring_ms as f64
    );
    println!(
        "queue {} s real-time ({} blocks)",
        args.queue_secs, queue_blocks
    );
    let wall = Instant::now();
    let done = Arc::new(AtomicBool::new(false));
    let mut consumers: Vec<(Source, RingConsumer)> = Vec::new();
    let mut producers = Vec::new();
    for &(source, ch) in &sources {
        let max_blocks = (ring_frames / BLOCK_FRAMES).max(16) + 16;
        let (p, c) = audio_ring(ch, ring_frames, max_blocks)?;
        consumers.push((source, c));
        let speed = args.speed;
        producers.push(thread::spawn(move || {
            produce(p, source, ch, total_frames, speed)
        }));
    }

    let done2 = done.clone();
    let drain = thread::spawn(move || -> (usize, u64) {
        let mut buf = Vec::new();
        let mut max_queued = 0usize;
        let mut queue_dropped = 0u64;
        loop {
            let mut idle = true;
            for (source, c) in &mut consumers {
                while let Some(meta) = c.pop(&mut buf) {
                    idle = false;
                    if qtx.try_send((*source, meta, buf.clone())).is_err() {
                        queue_dropped += meta.frames as u64;
                    }
                }
            }
            max_queued = max_queued.max(qtx.len());
            if idle {
                if done2.load(Ordering::Acquire)
                    && consumers.iter().all(|(_, c)| c.queued_blocks() == 0)
                {
                    return (max_queued, queue_dropped); // dropping qtx ends the journal thread
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
    });
    let stall = Duration::from_millis(args.stall_ms);
    let journal = thread::spawn(move || -> Result<(JournalWriter, Duration), String> {
        let mut max_write = Duration::ZERO;
        let mut stall_at = (!stall.is_zero()).then(|| Instant::now() + Duration::from_secs(1));
        for (source, meta, samples) in qrx {
            if stall_at.is_some_and(|t| Instant::now() >= t) {
                stall_at = None;
                thread::sleep(stall);
            }
            let t = Instant::now();
            writer
                .write(source, &meta, &samples)
                .map_err(|e| e.to_string())?;
            max_write = max_write.max(t.elapsed());
        }
        Ok((writer, max_write))
    });

    let reports: Vec<ProducerReport> = producers
        .into_iter()
        .map(|h| h.join().expect("producer"))
        .collect();
    done.store(true, Ordering::Release);
    let (max_queued, queue_dropped) = drain.join().expect("drain thread");
    let (writer, max_write) = journal.join().expect("journal thread")?;
    let stats = writer.finish()?;
    let elapsed = wall.elapsed();

    // Verify every recovered sample against the signal.
    let verify_start = Instant::now();
    let reader = JournalReader::open(&dir)?;
    let mut ok = true;
    let mut bytes = 0u64;
    println!("\nresults");
    for (i, &(source, ch)) in sources.iter().enumerate() {
        let mut frames = 0u64;
        let mut mismatches = 0u64;
        for blk in reader.blocks(source)? {
            let blk = blk?;
            for (k, &s) in blk.samples.iter().enumerate() {
                let f = blk.meta.first_frame + (k / ch as usize) as u64;
                if s != to_i16(signal(source, f, (k % ch as usize) as u16)) {
                    mismatches += 1;
                }
            }
            frames += blk.meta.frames as u64;
        }
        let r = &reports[i];
        let st = stats
            .iter()
            .find(|(s, _)| *s == source)
            .map(|(_, s)| *s)
            .unwrap_or_default();
        bytes += st.bytes;
        let lost = r.frames - frames;
        println!(
            "  {source:<6} produced {} frames, journaled {}, dropped in ring {}, lost {}, sample mismatches {}, segments {}",
            r.frames, frames, r.dropped_frames, lost, mismatches, st.segments
        );
        ok &= lost == 0 && mismatches == 0 && r.dropped_frames == 0;
    }
    ok &= queue_dropped == 0;
    let audio_secs = args.seconds as f64;
    println!(
        "  wall {:.2} s ({:.1}x real time achieved), journal {:.1} MiB ({:.1} MiB/s), max queued blocks {} of {}, dropped at queue {} frames, max single write {:.2} ms, verify {:.2} s",
        elapsed.as_secs_f64(),
        audio_secs / elapsed.as_secs_f64(),
        bytes as f64 / 1_048_576.0,
        bytes as f64 / 1_048_576.0 / elapsed.as_secs_f64(),
        max_queued,
        queue_blocks,
        queue_dropped,
        max_write.as_secs_f64() * 1e3,
        verify_start.elapsed().as_secs_f64()
    );
    println!(
        "  verdict: {}",
        if ok {
            "PASS (no audio lost, all samples verified)"
        } else {
            "FAIL"
        }
    );

    if ok && !args.keep {
        std::fs::remove_dir_all(&dir).ok();
    } else {
        println!("  journal kept at {}", dir.display());
    }
    Ok(if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
