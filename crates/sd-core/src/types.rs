use crate::clock::HostTime;
use crate::error::{Error, Result};

/// Where a stream of audio comes from. Mic and system audio are kept separate end to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Source {
    /// Local microphone (near end).
    Mic = 0,
    /// System / loopback output (far end, also the echo reference).
    System = 1,
}

impl Source {
    pub const ALL: [Source; 2] = [Source::Mic, Source::System];

    pub fn as_str(self) -> &'static str {
        match self {
            Source::Mic => "mic",
            Source::System => "system",
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Source::Mic),
            1 => Some(Source::System),
            _ => None,
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Source {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "mic" => Ok(Source::Mic),
            "system" => Ok(Source::System),
            other => Err(Error::InvalidArgument(format!("unknown source '{other}'"))),
        }
    }
}

/// Sample rate and channel count of an interleaved stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl StreamFormat {
    pub const MAX_CHANNELS: u16 = 32;
    pub const MIN_RATE: u32 = 8_000;
    pub const MAX_RATE: u32 = 384_000;

    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if !(Self::MIN_RATE..=Self::MAX_RATE).contains(&sample_rate) {
            return Err(Error::InvalidFormat(format!(
                "sample rate {sample_rate} Hz out of range"
            )));
        }
        if channels == 0 || channels > Self::MAX_CHANNELS {
            return Err(Error::InvalidFormat(format!(
                "channel count {channels} out of range"
            )));
        }
        Ok(Self {
            sample_rate,
            channels,
        })
    }

    /// Duration of `frames` frames in nanoseconds.
    pub fn frames_to_ns(&self, frames: u64) -> u64 {
        (frames as u128 * 1_000_000_000 / self.sample_rate as u128) as u64
    }

    /// Number of whole frames in `ns` nanoseconds.
    pub fn ns_to_frames(&self, ns: u64) -> u64 {
        (ns as u128 * self.sample_rate as u128 / 1_000_000_000) as u64
    }
}

/// Per-block flags describing timeline continuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockFlags(u32);

impl BlockFlags {
    /// Frames were lost or the device glitched immediately before this block.
    pub const DISCONTINUITY: BlockFlags = BlockFlags(1 << 0);
    /// The device reported this block as silent (e.g. WASAPI AUDCLNT_BUFFERFLAGS_SILENT).
    pub const SILENT: BlockFlags = BlockFlags(1 << 1);
    /// The block was synthesized to fill a gap; it holds no captured audio.
    pub const GAP_FILL: BlockFlags = BlockFlags(1 << 2);
    /// The device timestamp was unreliable and was replaced by an estimate.
    pub const TIMESTAMP_ESTIMATED: BlockFlags = BlockFlags(1 << 3);

    pub const fn empty() -> Self {
        BlockFlags(0)
    }
    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn from_bits_truncate(bits: u32) -> Self {
        BlockFlags(bits & 0b1111)
    }
    pub const fn contains(self, other: BlockFlags) -> bool {
        self.0 & other.0 == other.0
    }
    pub fn insert(&mut self, other: BlockFlags) {
        self.0 |= other.0;
    }
}

impl std::ops::BitOr for BlockFlags {
    type Output = BlockFlags;
    fn bitor(self, rhs: Self) -> Self {
        BlockFlags(self.0 | rhs.0)
    }
}

/// Metadata for one contiguous block of interleaved frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMeta {
    /// Index of the first frame of this block on the source's timeline (counts dropped frames too).
    pub first_frame: u64,
    /// Number of frames (not samples) in the block.
    pub frames: u32,
    /// Host time at which the first frame was captured (device position), in host-clock ns.
    pub host_time: HostTime,
    pub flags: BlockFlags,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_validation() {
        assert!(StreamFormat::new(48_000, 2).is_ok());
        assert!(StreamFormat::new(4_000, 2).is_err());
        assert!(StreamFormat::new(48_000, 0).is_err());
        assert!(StreamFormat::new(48_000, 33).is_err());
    }

    #[test]
    fn frame_ns_round_trip() {
        let f = StreamFormat::new(48_000, 2).unwrap();
        assert_eq!(f.frames_to_ns(48_000), 1_000_000_000);
        assert_eq!(f.ns_to_frames(10_000_000), 480);
        // 24 h at 48 kHz does not overflow.
        let frames = 48_000u64 * 86_400;
        assert_eq!(f.ns_to_frames(f.frames_to_ns(frames)), frames);
    }

    #[test]
    fn source_round_trip() {
        for s in Source::ALL {
            assert_eq!(Source::from_u8(s as u8), Some(s));
            assert_eq!(s.as_str().parse::<Source>().unwrap(), s);
        }
        assert!("speaker".parse::<Source>().is_err());
    }

    #[test]
    fn flags() {
        let mut f = BlockFlags::empty();
        assert!(!f.contains(BlockFlags::DISCONTINUITY));
        f.insert(BlockFlags::DISCONTINUITY);
        assert!(f.contains(BlockFlags::DISCONTINUITY));
        assert!(!f.contains(BlockFlags::GAP_FILL));
        assert_eq!(BlockFlags::from_bits_truncate(0xffff_ffff).bits(), 0b1111);
    }
}
