//! Binary layouts of segment headers and index records, plus the JSON manifest.

use sd_core::{BlockFlags, BlockMeta, HostTime, Source, StreamFormat};
use serde::{Deserialize, Serialize};

/// Version written by this build. Version 1 (raw PCM only) is still read.
pub const FORMAT_VERSION: u32 = 2;
pub const MIN_FORMAT_VERSION: u32 = 1;
pub const MANIFEST_FILE: &str = "journal.json";

pub const SEGMENT_MAGIC: [u8; 8] = *b"APJSEG01";
pub const SEGMENT_HEADER_LEN: usize = 64;
pub const SAMPLE_FORMAT_I16LE: u8 = 1;
pub const SAMPLE_FORMAT_LOSSLESS: u8 = 2;
pub const BYTES_PER_SAMPLE: usize = 2;

pub const INDEX_MAGIC: u32 = u32::from_le_bytes(*b"APIX");
pub const INDEX_RECORD_LEN: usize = 48;

/// How block samples are stored in segment files.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Codec {
    /// Interleaved little-endian `i16` (format version 1 journals are always this).
    #[default]
    PcmS16le,
    /// [`crate::codec`]: per-block lossless prediction + Rice coding.
    Lossless,
}

impl Codec {
    fn byte(self) -> u8 {
        match self {
            Codec::PcmS16le => SAMPLE_FORMAT_I16LE,
            Codec::Lossless => SAMPLE_FORMAT_LOSSLESS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub format_version: u32,
    /// Wall-clock creation time, Unix milliseconds (for humans; timelines use host time).
    pub created_unix_ms: u64,
    /// Host-monotonic time at creation, to relate host timestamps to `created_unix_ms`.
    pub created_host_ns: u64,
    pub segment_seconds: u32,
    pub sources: Vec<SourceManifest>,
    /// Codec of new segments (each segment header records its own; absent in version 1).
    #[serde(default)]
    pub codec: Codec,
    /// Set by [`crate::JournalWriter::finish`]. A journal without it was not closed cleanly.
    #[serde(default)]
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceManifest {
    pub source: String,
    pub sample_rate: u32,
    pub channels: u16,
}

impl SourceManifest {
    pub fn source(&self) -> Option<Source> {
        self.source.parse().ok()
    }
    pub fn format(&self) -> Option<StreamFormat> {
        StreamFormat::new(self.sample_rate, self.channels).ok()
    }
}

pub fn segment_file_names(seq: u32) -> (String, String) {
    (format!("{seq:06}.pcm"), format!("{seq:06}.idx"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentHeader {
    pub source: Source,
    pub format: StreamFormat,
    pub seq: u32,
    pub codec: Codec,
}

impl SegmentHeader {
    pub fn encode(&self) -> [u8; SEGMENT_HEADER_LEN] {
        let mut b = [0u8; SEGMENT_HEADER_LEN];
        b[0..8].copy_from_slice(&SEGMENT_MAGIC);
        b[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        b[12] = self.source as u8;
        b[13] = self.codec.byte();
        b[14..16].copy_from_slice(&self.format.channels.to_le_bytes());
        b[16..20].copy_from_slice(&self.format.sample_rate.to_le_bytes());
        b[20..24].copy_from_slice(&self.seq.to_le_bytes());
        let crc = crc32fast::hash(&b[..60]);
        b[60..64].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < SEGMENT_HEADER_LEN || b[0..8] != SEGMENT_MAGIC {
            return None;
        }
        if crc32fast::hash(&b[..60]) != u32::from_le_bytes(b[60..64].try_into().ok()?) {
            return None;
        }
        let version = u32::from_le_bytes(b[8..12].try_into().ok()?);
        let codec = match (version, b[13]) {
            (MIN_FORMAT_VERSION..=FORMAT_VERSION, SAMPLE_FORMAT_I16LE) => Codec::PcmS16le,
            (2..=FORMAT_VERSION, SAMPLE_FORMAT_LOSSLESS) => Codec::Lossless,
            _ => return None,
        };
        let source = Source::from_u8(b[12])?;
        let channels = u16::from_le_bytes(b[14..16].try_into().ok()?);
        let sample_rate = u32::from_le_bytes(b[16..20].try_into().ok()?);
        let format = StreamFormat::new(sample_rate, channels).ok()?;
        let seq = u32::from_le_bytes(b[20..24].try_into().ok()?);
        Some(Self {
            source,
            format,
            seq,
            codec,
        })
    }
}

/// One index record: where a block's samples live in the segment and its timeline metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexRecord {
    /// Offset of the block's first stored byte from the start of the `.pcm` file.
    pub byte_offset: u64,
    /// Stored (possibly compressed) length in bytes. 0 in version 1 records, where the
    /// length is `frames * channels * 2`.
    pub stored_len: u32,
    pub meta: BlockMeta,
    pub data_crc: u32,
}

impl IndexRecord {
    pub fn encode(&self) -> [u8; INDEX_RECORD_LEN] {
        let mut b = [0u8; INDEX_RECORD_LEN];
        b[0..4].copy_from_slice(&INDEX_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&self.meta.flags.bits().to_le_bytes());
        b[8..16].copy_from_slice(&self.byte_offset.to_le_bytes());
        b[16..24].copy_from_slice(&self.meta.first_frame.to_le_bytes());
        b[24..28].copy_from_slice(&self.meta.frames.to_le_bytes());
        b[28..32].copy_from_slice(&self.data_crc.to_le_bytes());
        b[32..40].copy_from_slice(&self.meta.host_time.as_ns().to_le_bytes());
        b[40..44].copy_from_slice(&self.stored_len.to_le_bytes());
        let crc = crc32fast::hash(&b[..44]);
        b[44..48].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < INDEX_RECORD_LEN {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if u32_at(0) != INDEX_MAGIC || crc32fast::hash(&b[..44]) != u32_at(44) {
            return None;
        }
        Some(Self {
            byte_offset: u64_at(8),
            stored_len: u32_at(40),
            meta: BlockMeta {
                first_frame: u64_at(16),
                frames: u32_at(24),
                host_time: HostTime::from_ns(u64_at(32)),
                flags: BlockFlags::from_bits_truncate(u32_at(4)),
            },
            data_crc: u32_at(28),
        })
    }
}

/// Inverse of [`f32_to_i16`] for one sample (the scale readers and reprocessing use).
pub fn i16_to_f32(v: i16) -> f32 {
    v as f32 / 32767.0
}

/// Converts float samples to `i16` with rounding and clipping. Non-finite samples become 0.
/// Returns how many samples were out of range or non-finite.
pub fn f32_to_i16(input: &[f32], out: &mut Vec<i16>) -> u64 {
    out.clear();
    out.reserve(input.len());
    let mut clipped = 0u64;
    for &x in input {
        // Out-of-range and non-finite samples (NaN/Inf mean an upstream bug) are both counted.
        if !(-1.0..=1.0).contains(&x) {
            clipped += 1;
        }
        let x = if x.is_finite() { x } else { 0.0 };
        out.push((x.clamp(-1.0, 1.0) * 32767.0).round() as i16);
    }
    clipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_and_corruption() {
        let h = SegmentHeader {
            source: Source::System,
            format: StreamFormat::new(48_000, 2).unwrap(),
            seq: 7,
            codec: Codec::Lossless,
        };
        let mut b = h.encode();
        assert_eq!(SegmentHeader::decode(&b), Some(h));
        b[17] ^= 1;
        assert_eq!(SegmentHeader::decode(&b), None);
    }

    #[test]
    fn record_round_trip_and_corruption() {
        let r = IndexRecord {
            byte_offset: 64 + 1920,
            stored_len: 1234,
            meta: BlockMeta {
                first_frame: 123_456_789,
                frames: 480,
                host_time: HostTime::from_ns(987_654_321_000),
                flags: BlockFlags::DISCONTINUITY | BlockFlags::SILENT,
            },
            data_crc: 0xDEAD_BEEF,
        };
        let mut b = r.encode();
        assert_eq!(IndexRecord::decode(&b), Some(r));
        for i in [0, 9, 20, 30, 33, 41, 46] {
            let saved = b[i];
            b[i] ^= 0x40;
            assert_eq!(
                IndexRecord::decode(&b),
                None,
                "flip at byte {i} not detected"
            );
            b[i] = saved;
        }
    }

    #[test]
    fn sample_conversion() {
        let mut v = Vec::new();
        let clipped = f32_to_i16(&[0.0, 1.0, -1.0, 0.5, 2.0, f32::NAN], &mut v);
        assert_eq!(v, vec![0, 32767, -32767, 16384, 32767, 0]);
        assert_eq!(clipped, 2);
    }
}
