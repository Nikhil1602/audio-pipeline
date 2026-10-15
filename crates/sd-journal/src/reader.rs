use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sd_core::{BlockFlags, BlockMeta, HostTime, Source, StreamFormat};

use crate::error::{JournalError, Result};
use crate::format::{
    BYTES_PER_SAMPLE, Codec, FORMAT_VERSION, INDEX_RECORD_LEN, IndexRecord, MANIFEST_FILE,
    MIN_FORMAT_VERSION, Manifest, SEGMENT_HEADER_LEN, SegmentHeader, segment_file_names,
};
use crate::writer::frame_bytes;

/// A verified block read back from the journal.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveredBlock {
    pub meta: BlockMeta,
    /// Interleaved samples.
    pub samples: Vec<i16>,
}

/// Outcome of verifying one segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentReport {
    pub seq: u32,
    pub valid_blocks: u64,
    pub valid_frames: u64,
    /// Index records after the first invalid one (not trusted, not returned).
    pub rejected_records: u64,
    /// Why reading stopped early, if it did.
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceReport {
    pub source: Source,
    pub format: StreamFormat,
    pub segments: Vec<SegmentReport>,
    pub blocks: u64,
    pub frames: u64,
    /// Timeline gaps between consecutive blocks (dropped or skipped frames).
    pub gaps: u64,
    pub gap_frames: u64,
    pub first_frame: Option<u64>,
    /// One past the last recovered frame.
    pub end_frame: Option<u64>,
    pub first_host_time: Option<HostTime>,
    pub last_host_time: Option<HostTime>,
    pub discontinuity_flags: u64,
}

impl SourceReport {
    pub fn rejected_records(&self) -> u64 {
        self.segments.iter().map(|s| s.rejected_records).sum()
    }
    pub fn duration_secs(&self) -> f64 {
        self.frames as f64 / self.format.sample_rate as f64
    }
}

pub struct JournalReader {
    dir: PathBuf,
    manifest: Manifest,
}

impl JournalReader {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join(MANIFEST_FILE);
        let bytes = fs::read(&path).map_err(|e| JournalError::io(&path, e))?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| JournalError::BadManifest {
                path: path.clone(),
                reason: e.to_string(),
            })?;
        if !(MIN_FORMAT_VERSION..=FORMAT_VERSION).contains(&manifest.format_version) {
            return Err(JournalError::BadManifest {
                path,
                reason: format!("unsupported format version {}", manifest.format_version),
            });
        }
        for s in &manifest.sources {
            if s.source().is_none() || s.format().is_none() {
                return Err(JournalError::BadManifest {
                    path,
                    reason: format!("invalid source entry {s:?}"),
                });
            }
        }
        Ok(Self { dir, manifest })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn sources(&self) -> Vec<(Source, StreamFormat)> {
        self.manifest
            .sources
            .iter()
            .filter_map(|s| Some((s.source()?, s.format()?)))
            .collect()
    }

    pub fn format_of(&self, source: Source) -> Result<StreamFormat> {
        self.sources()
            .into_iter()
            .find(|(s, _)| *s == source)
            .map(|(_, f)| f)
            .ok_or(JournalError::UnknownSource(source))
    }

    /// Iterates over all verified blocks of `source` in order.
    pub fn blocks(&self, source: Source) -> Result<BlockIter> {
        let format = self.format_of(source)?;
        let sdir = self.dir.join(source.as_str());
        let mut seqs = Vec::new();
        if sdir.exists() {
            for entry in fs::read_dir(&sdir).map_err(|e| JournalError::io(&sdir, e))? {
                let entry = entry.map_err(|e| JournalError::io(&sdir, e))?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(stem) = name.strip_suffix(".pcm")
                    && let Ok(seq) = stem.parse::<u32>()
                {
                    seqs.push(seq);
                }
            }
        }
        seqs.sort_unstable();
        Ok(BlockIter {
            dir: sdir,
            source,
            format,
            seqs,
            next_seg: 0,
            current: None,
            reports: Vec::new(),
            last_end_frame: None,
        })
    }

    /// Verifies every block of `source` and summarizes what is recoverable.
    pub fn scan(&self, source: Source) -> Result<SourceReport> {
        let format = self.format_of(source)?;
        let mut it = self.blocks(source)?;
        let mut r = SourceReport {
            source,
            format,
            segments: Vec::new(),
            blocks: 0,
            frames: 0,
            gaps: 0,
            gap_frames: 0,
            first_frame: None,
            end_frame: None,
            first_host_time: None,
            last_host_time: None,
            discontinuity_flags: 0,
        };
        for block in it.by_ref() {
            let m = block?.meta;
            if let Some(end) = r.end_frame {
                if m.first_frame > end {
                    r.gaps += 1;
                    r.gap_frames += m.first_frame - end;
                }
            } else {
                r.first_frame = Some(m.first_frame);
                r.first_host_time = Some(m.host_time);
            }
            if m.flags.contains(BlockFlags::DISCONTINUITY) {
                r.discontinuity_flags += 1;
            }
            r.blocks += 1;
            r.frames += m.frames as u64;
            r.end_frame = Some(m.first_frame + m.frames as u64);
            r.last_host_time = Some(m.host_time);
        }
        r.segments = it.into_reports();
        Ok(r)
    }
}

/// Upper bound on frames in one block (blocks are normally 10 ms), so a damaged index
/// cannot make the reader allocate huge buffers.
const MAX_BLOCK_FRAMES: u32 = 1 << 20;

struct CurrentSegment {
    pcm: File,
    codec: Codec,
    pcm_len: u64,
    records: Vec<IndexRecord>,
    next: usize,
    report: SegmentReport,
}

/// Iterator over verified blocks. Corruption ends the current segment (recorded in its
/// [`SegmentReport`]); only real I/O errors are yielded as `Err`.
pub struct BlockIter {
    dir: PathBuf,
    source: Source,
    format: StreamFormat,
    seqs: Vec<u32>,
    next_seg: usize,
    current: Option<CurrentSegment>,
    reports: Vec<SegmentReport>,
    last_end_frame: Option<u64>,
}

impl BlockIter {
    /// Per-segment verification reports for everything iterated so far.
    pub fn into_reports(mut self) -> Vec<SegmentReport> {
        if let Some(cur) = self.current.take() {
            self.reports.push(cur.report);
        }
        self.reports
    }

    fn open_next_segment(&mut self) -> Result<Option<()>> {
        let Some(&seq) = self.seqs.get(self.next_seg) else {
            return Ok(None);
        };
        self.next_seg += 1;
        let (pcm_name, idx_name) = segment_file_names(seq);
        let pcm_path = self.dir.join(pcm_name);
        let idx_path = self.dir.join(idx_name);
        let mut report = SegmentReport {
            seq,
            valid_blocks: 0,
            valid_frames: 0,
            rejected_records: 0,
            stop_reason: None,
        };

        let mut pcm = File::open(&pcm_path).map_err(|e| JournalError::io(&pcm_path, e))?;
        let pcm_len = pcm
            .metadata()
            .map_err(|e| JournalError::io(&pcm_path, e))?
            .len();
        let mut header = [0u8; SEGMENT_HEADER_LEN];
        let header = (pcm_len >= SEGMENT_HEADER_LEN as u64 && pcm.read_exact(&mut header).is_ok())
            .then(|| SegmentHeader::decode(&header))
            .flatten()
            .filter(|h| h.source == self.source && h.format == self.format && h.seq == seq);
        let header_ok = header.is_some();

        let mut records = Vec::new();
        if !header_ok {
            report.stop_reason = Some("missing or invalid segment header".into());
        } else {
            match fs::read(&idx_path) {
                Ok(bytes) => {
                    let whole = bytes.len() / INDEX_RECORD_LEN;
                    let mut stopped = false;
                    for (i, chunk) in bytes.chunks_exact(INDEX_RECORD_LEN).enumerate() {
                        match IndexRecord::decode(chunk) {
                            Some(r) => records.push(r),
                            None => {
                                report.rejected_records = (whole - i) as u64;
                                report.stop_reason = Some(format!("corrupt index record {i}"));
                                stopped = true;
                                break;
                            }
                        }
                    }
                    if !stopped && bytes.len() % INDEX_RECORD_LEN != 0 {
                        report.stop_reason = Some("torn index record at end".into());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    report.stop_reason = Some("index file missing".into());
                }
                Err(e) => return Err(JournalError::io(&idx_path, e)),
            }
        }
        self.current = Some(CurrentSegment {
            pcm,
            codec: header.map_or(Codec::PcmS16le, |h| h.codec),
            pcm_len,
            records,
            next: 0,
            report,
        });
        Ok(Some(()))
    }

    fn next_block(&mut self) -> Result<Option<RecoveredBlock>> {
        let fb = frame_bytes(self.format) as u64;
        loop {
            if self.current.is_none() && self.open_next_segment()?.is_none() {
                return Ok(None);
            }
            let cur = self.current.as_mut().expect("segment open");
            let Some(&rec) = cur.records.get(cur.next) else {
                let done = self.current.take().expect("segment open");
                self.reports.push(done.report);
                continue;
            };

            let pcm_len = rec.meta.frames as u64 * fb;
            let len = match (cur.codec, rec.stored_len) {
                (Codec::PcmS16le, 0) => pcm_len,
                (_, n) => n as u64,
            };
            let reject = |cur: &mut CurrentSegment, why: String| {
                cur.report.rejected_records += (cur.records.len() - cur.next) as u64;
                cur.report.stop_reason = Some(why);
                cur.next = cur.records.len();
            };
            if rec.meta.frames == 0
                || rec.meta.frames > MAX_BLOCK_FRAMES
                || (cur.codec == Codec::PcmS16le && len != pcm_len)
                || rec.byte_offset < SEGMENT_HEADER_LEN as u64
                || rec.byte_offset + len > cur.pcm_len
            {
                reject(
                    cur,
                    format!("record {} points outside the data file", cur.next),
                );
                continue;
            }
            if self
                .last_end_frame
                .is_some_and(|end| rec.meta.first_frame < end)
            {
                reject(cur, format!("record {} goes backwards in time", cur.next));
                continue;
            }
            let mut bytes = vec![0u8; len as usize];
            cur.pcm
                .seek(SeekFrom::Start(rec.byte_offset))
                .and_then(|_| cur.pcm.read_exact(&mut bytes))
                .map_err(|e| {
                    JournalError::io(self.dir.join(format!("{:06}.pcm", cur.report.seq)), e)
                })?;
            if crc32fast::hash(&bytes) != rec.data_crc {
                reject(cur, format!("record {} data checksum mismatch", cur.next));
                continue;
            }
            let samples = match cur.codec {
                Codec::PcmS16le => bytes
                    .chunks_exact(BYTES_PER_SAMPLE)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                Codec::Lossless => match crate::codec::decode(
                    &bytes,
                    self.format.channels as usize,
                    rec.meta.frames as usize,
                ) {
                    Some(s) => s,
                    None => {
                        reject(cur, format!("record {} does not decode", cur.next));
                        continue;
                    }
                },
            };
            cur.next += 1;
            cur.report.valid_blocks += 1;
            cur.report.valid_frames += rec.meta.frames as u64;
            self.last_end_frame = Some(rec.meta.first_frame + rec.meta.frames as u64);
            return Ok(Some(RecoveredBlock {
                meta: rec.meta,
                samples,
            }));
        }
    }
}

impl Iterator for BlockIter {
    type Item = Result<RecoveredBlock>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_block().transpose()
    }
}
