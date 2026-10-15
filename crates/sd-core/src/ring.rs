//! Lock-free single-producer / single-consumer audio ring for real-time capture callbacks.
//!
//! The producer side is called from the OS audio thread. It never allocates, locks, or blocks:
//! a block is either written completely or dropped completely. Dropped blocks are counted and
//! still advance the frame timeline, so downstream consumers see an explicit gap
//! ([`BlockFlags::DISCONTINUITY`]) instead of silently compressed time.
//!
//! Internally there are two `rtrb` rings: one for interleaved `f32` samples and one for
//! [`BlockMeta`]. Samples are committed before their metadata, so when the consumer sees a
//! metadata entry the matching samples are guaranteed to be readable.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::clock::HostTime;
use crate::error::{Error, Result};
use crate::types::{BlockFlags, BlockMeta};

/// Counters shared between both ends of a ring. All updates are `Relaxed`; they are statistics.
#[derive(Debug, Default)]
pub struct RingStats {
    pushed_blocks: AtomicU64,
    pushed_frames: AtomicU64,
    dropped_blocks: AtomicU64,
    dropped_frames: AtomicU64,
}

/// A point-in-time copy of [`RingStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RingStatsSnapshot {
    pub pushed_blocks: u64,
    pub pushed_frames: u64,
    pub dropped_blocks: u64,
    pub dropped_frames: u64,
}

impl RingStats {
    pub fn snapshot(&self) -> RingStatsSnapshot {
        RingStatsSnapshot {
            pushed_blocks: self.pushed_blocks.load(Ordering::Relaxed),
            pushed_frames: self.pushed_frames.load(Ordering::Relaxed),
            dropped_blocks: self.dropped_blocks.load(Ordering::Relaxed),
            dropped_frames: self.dropped_frames.load(Ordering::Relaxed),
        }
    }
}

/// Result of [`RingProducer::push`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    Written,
    /// The consumer fell behind (or the block exceeds capacity); the block was dropped.
    Dropped,
}

/// Creates a ring holding up to `capacity_frames` frames of `channels`-channel audio in at most
/// `max_blocks` blocks.
pub fn audio_ring(
    channels: u16,
    capacity_frames: usize,
    max_blocks: usize,
) -> Result<(RingProducer, RingConsumer)> {
    if channels == 0 {
        return Err(Error::InvalidArgument("channels must be > 0".into()));
    }
    if capacity_frames == 0 || max_blocks == 0 {
        return Err(Error::InvalidArgument("ring capacity must be > 0".into()));
    }
    let samples = capacity_frames
        .checked_mul(channels as usize)
        .ok_or_else(|| Error::InvalidArgument("ring capacity overflows".into()))?;
    let (sp, sc) = rtrb::RingBuffer::new(samples);
    let (mp, mc) = rtrb::RingBuffer::new(max_blocks);
    let stats = Arc::new(RingStats::default());
    Ok((
        RingProducer {
            samples: sp,
            metas: mp,
            channels: channels as usize,
            next_frame: 0,
            pending_discontinuity: false,
            stats: stats.clone(),
        },
        RingConsumer {
            samples: sc,
            metas: mc,
            channels: channels as usize,
            stats,
        },
    ))
}

pub struct RingProducer {
    samples: rtrb::Producer<f32>,
    metas: rtrb::Producer<BlockMeta>,
    channels: usize,
    next_frame: u64,
    pending_discontinuity: bool,
    stats: Arc<RingStats>,
}

impl RingProducer {
    /// Pushes one block of interleaved samples captured at `host_time`.
    ///
    /// Real-time safe. `interleaved.len()` must be a multiple of the channel count; a trailing
    /// partial frame is ignored.
    pub fn push(
        &mut self,
        interleaved: &[f32],
        host_time: HostTime,
        mut flags: BlockFlags,
    ) -> PushOutcome {
        let frames = interleaved.len() / self.channels;
        let used = frames * self.channels;
        let first_frame = self.next_frame;
        self.next_frame += frames as u64;

        if frames == 0 {
            return PushOutcome::Written;
        }
        if self.samples.slots() < used || self.metas.slots() < 1 || frames > u32::MAX as usize {
            self.pending_discontinuity = true;
            self.stats.dropped_blocks.fetch_add(1, Ordering::Relaxed);
            self.stats
                .dropped_frames
                .fetch_add(frames as u64, Ordering::Relaxed);
            return PushOutcome::Dropped;
        }
        if self.pending_discontinuity {
            flags.insert(BlockFlags::DISCONTINUITY);
            self.pending_discontinuity = false;
        }
        // Space was checked above and only this producer adds data, so neither push can fail.
        let samples_ok = self.samples.push_entire_slice(&interleaved[..used]).is_ok();
        debug_assert!(samples_ok);
        let meta = BlockMeta {
            first_frame,
            frames: frames as u32,
            host_time,
            flags,
        };
        let meta_ok = self.metas.push(meta).is_ok();
        debug_assert!(meta_ok);
        self.stats.pushed_blocks.fetch_add(1, Ordering::Relaxed);
        self.stats
            .pushed_frames
            .fetch_add(frames as u64, Ordering::Relaxed);
        PushOutcome::Written
    }

    /// Advances the timeline by `frames` without data, e.g. when the device reports a glitch.
    /// The next pushed block is flagged as a discontinuity. Real-time safe.
    pub fn skip(&mut self, frames: u64) {
        if frames > 0 {
            self.next_frame += frames;
            self.pending_discontinuity = true;
        }
    }

    /// Frame index the next pushed block will start at.
    pub fn next_frame(&self) -> u64 {
        self.next_frame
    }

    pub fn stats(&self) -> RingStatsSnapshot {
        self.stats.snapshot()
    }

    /// True if the consumer was dropped.
    pub fn is_abandoned(&self) -> bool {
        self.metas.is_abandoned()
    }
}

pub struct RingConsumer {
    samples: rtrb::Consumer<f32>,
    metas: rtrb::Consumer<BlockMeta>,
    channels: usize,
    stats: Arc<RingStats>,
}

impl RingConsumer {
    /// Pops the next block into `out` (resized to fit). Returns `None` when the ring is empty.
    pub fn pop(&mut self, out: &mut Vec<f32>) -> Option<BlockMeta> {
        let len = self.metas.peek().ok()?.frames as usize * self.channels;
        out.resize(len, 0.0);
        self.pop_into(out).expect("buffer sized to fit")
    }

    /// Pops the next block into the front of `out` without allocating.
    ///
    /// Returns `Ok(None)` when empty, or an error (leaving the block queued) if `out` is too
    /// small; the needed length is `meta.frames * channels`.
    pub fn pop_into(&mut self, out: &mut [f32]) -> Result<Option<BlockMeta>> {
        let Ok(&meta) = self.metas.peek() else {
            return Ok(None);
        };
        let len = meta.frames as usize * self.channels;
        if out.len() < len {
            return Err(Error::InvalidArgument(format!(
                "output buffer holds {} samples, block needs {len}",
                out.len()
            )));
        }
        // Samples were committed before their metadata, so they are available.
        let ok = self.samples.pop_entire_slice(&mut out[..len]).is_ok();
        debug_assert!(ok, "ring invariant violated: metadata without samples");
        let _ = self.metas.pop();
        Ok(Some(meta))
    }

    /// Number of queued blocks.
    pub fn queued_blocks(&self) -> usize {
        self.metas.slots()
    }

    pub fn channels(&self) -> u16 {
        self.channels as u16
    }

    pub fn stats(&self) -> RingStatsSnapshot {
        self.stats.snapshot()
    }

    /// True if the producer was dropped (capture stopped). Remaining blocks can still be read.
    pub fn is_abandoned(&self) -> bool {
        self.metas.is_abandoned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(start: u64, frames: usize, ch: usize) -> Vec<f32> {
        (0..frames * ch)
            .map(|i| (start as usize * ch + i) as f32)
            .collect()
    }

    #[test]
    fn round_trip_preserves_data_and_timeline() {
        let (mut p, mut c) = audio_ring(2, 4_800, 64).unwrap();
        let mut buf = Vec::new();
        for i in 0..10u64 {
            let b = block(i * 480, 480, 2);
            assert_eq!(
                p.push(&b, HostTime(i * 10_000_000), BlockFlags::empty()),
                PushOutcome::Written
            );
            let meta = c.pop(&mut buf).unwrap();
            assert_eq!(meta.first_frame, i * 480);
            assert_eq!(meta.frames, 480);
            assert_eq!(meta.host_time, HostTime(i * 10_000_000));
            assert!(!meta.flags.contains(BlockFlags::DISCONTINUITY));
            assert_eq!(buf, b);
        }
        assert!(c.pop(&mut buf).is_none());
    }

    #[test]
    fn overflow_drops_whole_block_and_flags_gap() {
        let (mut p, mut c) = audio_ring(1, 1_000, 64).unwrap();
        assert_eq!(
            p.push(&block(0, 600, 1), HostTime(0), BlockFlags::empty()),
            PushOutcome::Written
        );
        // Only 400 slots left: a 600-frame block must be dropped, not truncated.
        assert_eq!(
            p.push(&block(600, 600, 1), HostTime(1), BlockFlags::empty()),
            PushOutcome::Dropped
        );
        let mut buf = Vec::new();
        c.pop(&mut buf).unwrap();
        assert_eq!(
            p.push(&block(1200, 100, 1), HostTime(2), BlockFlags::empty()),
            PushOutcome::Written
        );
        let meta = c.pop(&mut buf).unwrap();
        assert_eq!(
            meta.first_frame, 1_200,
            "timeline must include the dropped frames"
        );
        assert!(meta.flags.contains(BlockFlags::DISCONTINUITY));
        let s = c.stats();
        assert_eq!((s.dropped_blocks, s.dropped_frames), (1, 600));
        assert_eq!((s.pushed_blocks, s.pushed_frames), (2, 700));
    }

    #[test]
    fn metadata_ring_full_also_drops() {
        let (mut p, _c) = audio_ring(1, 10_000, 2).unwrap();
        assert_eq!(
            p.push(&[0.0; 10], HostTime(0), BlockFlags::empty()),
            PushOutcome::Written
        );
        assert_eq!(
            p.push(&[0.0; 10], HostTime(0), BlockFlags::empty()),
            PushOutcome::Written
        );
        assert_eq!(
            p.push(&[0.0; 10], HostTime(0), BlockFlags::empty()),
            PushOutcome::Dropped
        );
    }

    #[test]
    fn skip_advances_timeline() {
        let (mut p, mut c) = audio_ring(1, 1_000, 8).unwrap();
        p.skip(480);
        p.push(&[0.5; 10], HostTime(0), BlockFlags::empty());
        let meta = c.pop(&mut Vec::new()).unwrap();
        assert_eq!(meta.first_frame, 480);
        assert!(meta.flags.contains(BlockFlags::DISCONTINUITY));
    }

    #[test]
    fn pop_into_too_small_keeps_block() {
        let (mut p, mut c) = audio_ring(2, 1_000, 8).unwrap();
        p.push(&[1.0; 20], HostTime(0), BlockFlags::empty());
        let mut small = [0.0f32; 10];
        assert!(c.pop_into(&mut small).is_err());
        let mut big = [0.0f32; 64];
        assert_eq!(c.pop_into(&mut big).unwrap().unwrap().frames, 10);
    }

    #[test]
    fn rejects_bad_config() {
        assert!(audio_ring(0, 10, 10).is_err());
        assert!(audio_ring(1, 0, 10).is_err());
        assert!(audio_ring(1, 10, 0).is_err());
    }

    /// Concurrent stress: a producer thread pushes variable-sized blocks while a consumer
    /// drains them. Every frame must be either delivered intact or accounted for as dropped,
    /// and the delivered timeline must be strictly increasing with gaps only after drops.
    #[test]
    fn concurrent_stress_accounts_for_every_frame() {
        const CH: usize = 2;
        const BLOCKS: u64 = 50_000;
        let (mut p, mut c) = audio_ring(CH as u16, 2_048, 32).unwrap();
        let producer = std::thread::spawn(move || {
            let mut scratch = vec![0.0f32; 1_024 * CH];
            let mut total = 0u64;
            for i in 0..BLOCKS {
                let frames = 1 + (i as usize * 7919) % 1_000;
                let start = p.next_frame();
                for (k, s) in scratch[..frames * CH].iter_mut().enumerate() {
                    // Encode the frame index so the consumer can verify content.
                    *s = ((start + (k / CH) as u64) % 1_000_003) as f32;
                }
                p.push(&scratch[..frames * CH], HostTime(i), BlockFlags::empty());
                total += frames as u64;
                if i % 64 == 0 {
                    std::thread::yield_now();
                }
            }
            total
        });

        let mut buf = Vec::new();
        let mut received = 0u64;
        let mut expected_next = 0u64;
        loop {
            match c.pop(&mut buf) {
                Some(meta) => {
                    assert!(meta.first_frame >= expected_next);
                    if meta.first_frame > expected_next {
                        assert!(meta.flags.contains(BlockFlags::DISCONTINUITY));
                    }
                    for (k, s) in buf.iter().enumerate() {
                        let frame = meta.first_frame + (k / CH) as u64;
                        assert_eq!(*s, (frame % 1_000_003) as f32);
                    }
                    expected_next = meta.first_frame + meta.frames as u64;
                    received += meta.frames as u64;
                }
                None if c.is_abandoned() && c.queued_blocks() == 0 => break,
                None => std::thread::yield_now(),
            }
        }
        let total = producer.join().unwrap();
        let s = c.stats();
        assert_eq!(received + s.dropped_frames, total);
        assert_eq!(received, s.pushed_frames);
    }
}
