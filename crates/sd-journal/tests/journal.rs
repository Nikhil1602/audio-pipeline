//! End-to-end journal tests: round trips, segment rotation, crash simulation, corruption,
//! timeline gaps and WAV export.

use std::fs;
use std::time::Duration;

use sd_core::{BlockFlags, BlockMeta, HostTime, Source, StreamFormat};
use sd_journal::{
    Codec, ExportOptions, JournalConfig, JournalError, JournalReader, JournalWriter, export_wav,
};

const MIC: (Source, u32, u16) = (Source::Mic, 16_000, 1);
const SYS: (Source, u32, u16) = (Source::System, 48_000, 2);

fn fmt(rate: u32, ch: u16) -> StreamFormat {
    StreamFormat::new(rate, ch).unwrap()
}

fn config(segment_seconds: u32) -> JournalConfig {
    let mut c = JournalConfig::new(vec![(MIC.0, fmt(MIC.1, MIC.2)), (SYS.0, fmt(SYS.1, SYS.2))]);
    c.segment_seconds = segment_seconds;
    c.sync_interval = Duration::from_secs(3600); // tests call sync() explicitly
    c
}

/// `(byte_offset, stored_len)` of every block, read from a segment's index file.
fn block_ranges(idx: &[u8]) -> Vec<(u64, u64)> {
    idx.chunks_exact(48)
        .map(|r| {
            let off = u64::from_le_bytes(r[8..16].try_into().unwrap());
            let len = u32::from_le_bytes(r[40..44].try_into().unwrap());
            (off, len as u64)
        })
        .collect()
}

/// Deterministic test signal: sample value depends on (source, frame, channel).
fn signal(source: Source, frame: u64, ch: u16) -> f32 {
    let x = (frame as f32 * 0.013 + ch as f32 * 1.7 + source as u8 as f32 * 0.5).sin();
    x * 0.9
}

fn block(source: Source, first_frame: u64, frames: u32, channels: u16) -> (BlockMeta, Vec<f32>) {
    let samples = (0..frames as u64)
        .flat_map(|f| (0..channels).map(move |c| signal(source, first_frame + f, c)))
        .collect();
    let meta = BlockMeta {
        first_frame,
        frames,
        host_time: HostTime::from_ns(1_000_000_000 + first_frame * 1_000),
        flags: BlockFlags::empty(),
    };
    (meta, samples)
}

fn to_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Writes `blocks` 10 ms blocks per source, interleaving sources like a live session.
fn write_blocks(w: &mut JournalWriter, start_block: u64, blocks: u64) {
    for b in start_block..start_block + blocks {
        for (s, rate, ch) in [MIC, SYS] {
            let frames = rate / 100;
            let (meta, samples) = block(s, b * frames as u64, frames, ch);
            w.write(s, &meta, &samples).unwrap();
        }
    }
}

/// Asserts every recovered block matches the expected signal exactly and the timeline is a
/// contiguous prefix starting at frame 0. Returns recovered frames.
fn verify_prefix(reader: &JournalReader, source: Source, channels: u16) -> u64 {
    let mut next = 0u64;
    for blk in reader.blocks(source).unwrap() {
        let blk = blk.unwrap();
        assert_eq!(
            blk.meta.first_frame, next,
            "{source}: recovered data must be a contiguous prefix"
        );
        for (i, &s) in blk.samples.iter().enumerate() {
            let f = blk.meta.first_frame + (i / channels as usize) as u64;
            let c = (i % channels as usize) as u16;
            assert_eq!(
                s,
                to_i16(signal(source, f, c)),
                "{source}: sample mismatch at frame {f}"
            );
        }
        next += blk.meta.frames as u64;
    }
    next
}

#[test]
fn round_trip_with_segment_rotation() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = JournalWriter::create(dir.path(), config(1)).unwrap();
    write_blocks(&mut w, 0, 350); // 3.5 s per source
    let stats = w.finish().unwrap();
    for (s, st) in stats {
        assert_eq!(st.blocks, 350, "{s}");
        assert_eq!(st.segments, 4, "{s}: 1 s segments for 3.5 s of audio");
        assert_eq!(st.io_errors, 0);
    }

    let r = JournalReader::open(dir.path()).unwrap();
    assert!(r.manifest().finished);
    assert_eq!(verify_prefix(&r, Source::Mic, 1), 56_000);
    assert_eq!(verify_prefix(&r, Source::System, 2), 168_000);
    let rep = r.scan(Source::System).unwrap();
    assert_eq!((rep.blocks, rep.gaps, rep.rejected_records()), (350, 0, 0));
    assert_eq!(rep.segments.len(), 4);
    assert!((rep.duration_secs() - 3.5).abs() < 1e-9);
    assert!(rep.segments.iter().all(|s| s.stop_reason.is_none()));
}

#[test]
fn crash_keeps_everything_synced() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = JournalWriter::create(dir.path(), config(60)).unwrap();
    write_blocks(&mut w, 0, 200); // 2 s
    w.sync().unwrap();
    write_blocks(&mut w, 200, 50); // unsynced: may or may not survive
    w.simulate_crash();

    let r = JournalReader::open(dir.path()).unwrap();
    assert!(
        !r.manifest().finished,
        "crashed journal must not claim a clean finish"
    );
    let mic = verify_prefix(&r, Source::Mic, 1);
    let sys = verify_prefix(&r, Source::System, 2);
    assert!(mic >= 32_000, "lost synced mic audio: {mic} frames");
    assert!(sys >= 96_000, "lost synced system audio: {sys} frames");
}

/// Truncates the last segment's data and index at many byte positions (torn writes). The
/// reader must never return wrong audio and must keep everything before the tear.
#[test]
fn torn_writes_at_every_offset_are_detected() {
    for codec in [Codec::Lossless, Codec::PcmS16le] {
        let dir = tempfile::tempdir().unwrap();
        let mut c = config(60);
        c.codec = codec;
        let mut w = JournalWriter::create(dir.path(), c).unwrap();
        write_blocks(&mut w, 0, 100);
        w.finish().unwrap();

        let pcm = dir.path().join("system/000000.pcm");
        let idx = dir.path().join("system/000000.idx");
        let (pcm_full, idx_full) = (fs::read(&pcm).unwrap(), fs::read(&idx).unwrap());
        let ranges = block_ranges(&idx_full);
        assert_eq!(ranges.len(), 100);
        if codec == Codec::PcmS16le {
            assert!(ranges.iter().all(|&(_, len)| len == 480 * 4));
        }

        for cut in (64..pcm_full.len() as u64).step_by(397) {
            fs::write(&pcm, &pcm_full[..cut as usize]).unwrap();
            let r = JournalReader::open(dir.path()).unwrap();
            let got = verify_prefix(&r, Source::System, 2);
            let whole_blocks = ranges.iter().filter(|&&(o, l)| o + l <= cut).count() as u64;
            assert_eq!(got, whole_blocks * 480, "{codec:?}: pcm cut at {cut}");
        }
        fs::write(&pcm, &pcm_full).unwrap();

        for cut in (0..idx_full.len()).step_by(13) {
            fs::write(&idx, &idx_full[..cut]).unwrap();
            let r = JournalReader::open(dir.path()).unwrap();
            let got = verify_prefix(&r, Source::System, 2);
            assert_eq!(got, (cut / 48) as u64 * 480, "{codec:?}: idx cut at {cut}");
        }
    }
}

#[test]
fn corrupted_data_stops_segment_but_later_segments_survive() {
    for codec in [Codec::Lossless, Codec::PcmS16le] {
        let dir = tempfile::tempdir().unwrap();
        let mut c = config(1);
        c.codec = codec;
        let mut w = JournalWriter::create(dir.path(), c).unwrap();
        write_blocks(&mut w, 0, 300); // 3 segments
        w.finish().unwrap();

        // Flip one byte in the middle of block 50 of the first mic segment.
        let ranges = block_ranges(&fs::read(dir.path().join("mic/000000.idx")).unwrap());
        let path = dir.path().join("mic/000000.pcm");
        let mut bytes = fs::read(&path).unwrap();
        let (off, len) = ranges[50];
        bytes[(off + len / 2) as usize] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();

        let r = JournalReader::open(dir.path()).unwrap();
        let rep = r.scan(Source::Mic).unwrap();
        assert_eq!(rep.segments[0].valid_blocks, 50, "{codec:?}");
        assert_eq!(rep.segments[0].rejected_records, 50);
        assert!(
            rep.segments[0]
                .stop_reason
                .as_deref()
                .unwrap()
                .contains("checksum")
        );
        assert_eq!(rep.segments[1].valid_blocks, 100);
        assert_eq!(rep.segments[2].valid_blocks, 100);
        assert_eq!(rep.gaps, 1, "the corrupted region becomes a reported gap");
        assert_eq!(rep.gap_frames, 50 * 160);
    }
}

/// Lossless blocks are smaller than PCM for a tonal signal and digital silence, and decode to
/// the same samples.
#[test]
fn lossless_codec_is_smaller_and_exact() {
    let mut sizes = Vec::new();
    for codec in [Codec::PcmS16le, Codec::Lossless] {
        let dir = tempfile::tempdir().unwrap();
        let mut c = config(60);
        c.codec = codec;
        let mut w = JournalWriter::create(dir.path(), c).unwrap();
        write_blocks(&mut w, 0, 200);
        let silence = vec![0.0f32; 960];
        for b in 200..300u64 {
            let meta = BlockMeta {
                first_frame: b * 480,
                frames: 480,
                host_time: HostTime::from_ns(1_000_000_000 + b * 480_000),
                flags: BlockFlags::empty(),
            };
            w.write(Source::System, &meta, &silence).unwrap();
        }
        let stats = w.finish().unwrap();
        let sys = stats.iter().find(|(s, _)| *s == Source::System).unwrap().1;
        assert_eq!(sys.pcm_bytes, 300 * 480 * 4);
        sizes.push(sys.bytes);

        let r = JournalReader::open(dir.path()).unwrap();
        assert_eq!(r.manifest().codec, codec);
        assert_eq!(verify_prefix(&r, Source::Mic, 1), 200 * 160);
        let blocks: Vec<_> = r
            .blocks(Source::System)
            .unwrap()
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(blocks.len(), 300);
        assert!(
            blocks[200..]
                .iter()
                .all(|b| b.samples.iter().all(|&v| v == 0))
        );
    }
    assert_eq!(sizes[0], 300 * 480 * 4);
    assert!(
        sizes[1] * 2 < sizes[0],
        "lossless {} bytes vs pcm {} bytes",
        sizes[1],
        sizes[0]
    );
}

#[test]
fn gaps_are_reported_and_filled_on_export() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(60);
    c.sources.truncate(1); // mic only
    let mut w = JournalWriter::create(dir.path(), c).unwrap();
    let (m1, s1) = block(Source::Mic, 0, 1_600, 1);
    let (mut m2, s2) = block(Source::Mic, 4_800, 1_600, 1); // 3_200-frame gap
    m2.flags = BlockFlags::DISCONTINUITY;
    w.write(Source::Mic, &m1, &s1).unwrap();
    w.write(Source::Mic, &m2, &s2).unwrap();
    w.finish().unwrap();

    let r = JournalReader::open(dir.path()).unwrap();
    let rep = r.scan(Source::Mic).unwrap();
    assert_eq!(
        (rep.gaps, rep.gap_frames, rep.discontinuity_flags),
        (1, 3_200, 1)
    );

    let wav = dir.path().join("mic.wav");
    let er = export_wav(&r, Source::Mic, &wav, ExportOptions::default()).unwrap();
    assert_eq!(
        (er.audio_frames, er.gap_frames_filled, er.frames_written),
        (3_200, 3_200, 6_400)
    );
    let read = hound::WavReader::open(&wav).unwrap();
    assert_eq!(read.spec().sample_rate, 16_000);
    assert_eq!(read.duration(), 6_400);
    let samples: Vec<i16> = read.into_samples().map(|s| s.unwrap()).collect();
    assert!(samples[1_600..4_800].iter().all(|&s| s == 0));
    assert_eq!(samples[4_800], to_i16(signal(Source::Mic, 4_800, 0)));

    let no_fill = ExportOptions {
        fill_gaps: false,
        ..Default::default()
    };
    let er = export_wav(&r, Source::Mic, dir.path().join("b.wav"), no_fill).unwrap();
    assert_eq!((er.frames_written, er.gaps_not_filled), (3_200, 1));
}

#[test]
fn rejects_invalid_use() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(60);
    c.sources.truncate(1);
    let mut w = JournalWriter::create(dir.path(), c.clone()).unwrap();
    let (meta, samples) = block(Source::System, 0, 10, 2);
    assert!(matches!(
        w.write(Source::System, &meta, &samples),
        Err(JournalError::UnknownSource(_))
    ));
    let (meta, samples) = block(Source::Mic, 0, 10, 1);
    assert!(matches!(
        w.write(Source::Mic, &meta, &samples[..9]),
        Err(JournalError::BadBlock { .. })
    ));
    w.finish().unwrap();
    assert!(matches!(
        JournalWriter::create(dir.path(), c),
        Err(JournalError::AlreadyExists(_))
    ));

    let mut dup = config(60);
    dup.sources.push((Source::Mic, fmt(16_000, 1)));
    assert!(matches!(
        JournalWriter::create(dir.path().join("x"), dup),
        Err(JournalError::Config(_))
    ));
}

#[test]
fn clipping_is_counted_not_wrapped() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(60);
    c.sources.truncate(1);
    let mut w = JournalWriter::create(dir.path(), c).unwrap();
    let meta = BlockMeta {
        first_frame: 0,
        frames: 4,
        host_time: HostTime(0),
        flags: BlockFlags::empty(),
    };
    w.write(Source::Mic, &meta, &[1.5, -3.0, 0.25, f32::INFINITY])
        .unwrap();
    let st = w.finish().unwrap();
    assert_eq!(st[0].1.clipped_samples, 3);
    let r = JournalReader::open(dir.path()).unwrap();
    let b = r.blocks(Source::Mic).unwrap().next().unwrap().unwrap();
    assert_eq!(b.samples, vec![32767, -32767, 8192, 0]);
}

#[test]
fn missing_index_file_is_tolerated() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = JournalWriter::create(dir.path(), config(1)).unwrap();
    write_blocks(&mut w, 0, 200);
    w.finish().unwrap();
    fs::remove_file(dir.path().join("mic/000000.idx")).unwrap();
    let r = JournalReader::open(dir.path()).unwrap();
    let rep = r.scan(Source::Mic).unwrap();
    assert_eq!(
        rep.segments[0].stop_reason.as_deref(),
        Some("index file missing")
    );
    assert_eq!(rep.segments[1].valid_blocks, 100);
}
