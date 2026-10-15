//! Crash-safe on-disk audio journal.
//!
//! Captured audio is appended here *before* any heavy processing, so a crash, power loss or a
//! slow transcription backlog never loses audio: processing can always catch up from disk.
//!
//! # Layout
//!
//! ```text
//! <session>/
//!   journal.json                 manifest (format version, per-source stream formats)
//!   mic/000000.pcm  000000.idx   segment data + its index
//!   mic/000001.pcm  000001.idx
//!   system/...
//! ```
//!
//! * `.pcm`: a 64-byte header (with CRC and the segment's codec) followed by one stored block
//!   after another: by default each block is losslessly compressed on its own ([`codec`]);
//!   with [`Codec::PcmS16le`] it is interleaved little-endian `i16` samples.
//! * `.idx`: fixed 48-byte records, one per block, each with a CRC over the record and a CRC
//!   over the block's stored bytes.
//!
//! # Durability rules
//!
//! 1. On [`JournalWriter::sync`], segment data is flushed and fsync'd **before** the index.
//! 2. Every segment has its own index, so a torn write damages at most the tail of one
//!    segment. After an I/O error (e.g. disk full) the writer starts a fresh segment.
//! 3. The reader accepts an index record only if its CRC matches, its byte range lies inside
//!    the data file, and the data CRC matches. Reading a segment stops at the first bad
//!    record; later segments are still read.
//!
//! Everything accepted by the last successful `sync` survives a crash. Blocks written after it
//! survive if the OS happened to persist them, and are verified by CRC either way.

pub mod codec;
mod error;
mod export;
mod format;
mod reader;
mod writer;

pub use error::{JournalError, Result};
pub use export::{ExportOptions, ExportReport, export_wav};
pub use format::{Codec, FORMAT_VERSION, Manifest, SourceManifest, f32_to_i16, i16_to_f32};
pub use reader::{JournalReader, RecoveredBlock, SegmentReport, SourceReport};
pub use writer::{JournalConfig, JournalWriter, WriterStats};
