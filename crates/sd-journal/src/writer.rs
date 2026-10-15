use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sd_core::{BlockMeta, HostTime, Source, StreamFormat};

use crate::error::{JournalError, Result};
use crate::format::{
    BYTES_PER_SAMPLE, Codec, FORMAT_VERSION, INDEX_RECORD_LEN, IndexRecord, MANIFEST_FILE,
    Manifest, SEGMENT_HEADER_LEN, SegmentHeader, SourceManifest, f32_to_i16, segment_file_names,
};

#[derive(Debug, Clone)]
pub struct JournalConfig {
    pub sources: Vec<(Source, StreamFormat)>,
    /// A new segment is started after this much audio per source.
    pub segment_seconds: u32,
    /// Maximum time between automatic fsyncs (the crash-loss window).
    pub sync_interval: Duration,
    /// How blocks are stored. Lossless compression is the default.
    pub codec: Codec,
}

impl JournalConfig {
    pub fn new(sources: Vec<(Source, StreamFormat)>) -> Self {
        Self {
            sources,
            segment_seconds: 60,
            sync_interval: Duration::from_secs(1),
            codec: Codec::Lossless,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriterStats {
    pub blocks: u64,
    pub frames: u64,
    /// Bytes stored on disk for block data.
    pub bytes: u64,
    /// The same data as raw 16-bit PCM (`bytes / pcm_bytes` is the compression ratio).
    pub pcm_bytes: u64,
    pub clipped_samples: u64,
    pub segments: u32,
    pub io_errors: u64,
}

pub struct JournalWriter {
    dir: PathBuf,
    manifest: Manifest,
    streams: Vec<Stream>,
    sync_interval: Duration,
    last_sync: Instant,
    pcm: Vec<i16>,
    scratch: Vec<u8>,
    finished: bool,
}

struct Stream {
    source: Source,
    format: StreamFormat,
    codec: Codec,
    dir: PathBuf,
    next_seq: u32,
    segment_frames: u64,
    open: Option<OpenSegment>,
    stats: WriterStats,
}

struct OpenSegment {
    pcm: BufWriter<File>,
    idx: BufWriter<File>,
    pcm_path: PathBuf,
    idx_path: PathBuf,
    pcm_len: u64,
    frames: u64,
}

impl JournalWriter {
    /// Creates a new journal in `dir` (created if missing; must not already hold a journal).
    pub fn create(dir: impl AsRef<Path>, cfg: JournalConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if cfg.sources.is_empty() {
            return Err(JournalError::Config(
                "at least one source is required".into(),
            ));
        }
        if cfg.segment_seconds == 0 {
            return Err(JournalError::Config("segment_seconds must be > 0".into()));
        }
        for (i, (s, _)) in cfg.sources.iter().enumerate() {
            if cfg.sources[..i].iter().any(|(o, _)| o == s) {
                return Err(JournalError::Config(format!("source {s} listed twice")));
            }
        }
        fs::create_dir_all(&dir).map_err(|e| JournalError::io(&dir, e))?;
        if dir.join(MANIFEST_FILE).exists() {
            return Err(JournalError::AlreadyExists(dir));
        }

        let mut streams = Vec::with_capacity(cfg.sources.len());
        for &(source, format) in &cfg.sources {
            let sdir = dir.join(source.as_str());
            fs::create_dir_all(&sdir).map_err(|e| JournalError::io(&sdir, e))?;
            streams.push(Stream {
                source,
                format,
                codec: cfg.codec,
                dir: sdir,
                next_seq: 0,
                segment_frames: format.sample_rate as u64 * cfg.segment_seconds as u64,
                open: None,
                stats: WriterStats::default(),
            });
        }

        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            created_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            created_host_ns: HostTime::now().as_ns(),
            segment_seconds: cfg.segment_seconds,
            sources: cfg
                .sources
                .iter()
                .map(|(s, f)| SourceManifest {
                    source: s.as_str().to_string(),
                    sample_rate: f.sample_rate,
                    channels: f.channels,
                })
                .collect(),
            codec: cfg.codec,
            finished: false,
        };
        write_manifest(&dir, &manifest)?;
        sync_dir(&dir)?;

        Ok(Self {
            dir,
            manifest,
            streams,
            sync_interval: cfg.sync_interval,
            last_sync: Instant::now(),
            pcm: Vec::new(),
            scratch: Vec::new(),
            finished: false,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Appends one block. `samples` are interleaved floats in [-1, 1] (clipped otherwise).
    ///
    /// On an I/O error the current segment is abandoned and the next call starts a new one,
    /// so the journal stays readable. Check [`JournalError::is_disk_full`] to pause and retry.
    pub fn write(&mut self, source: Source, meta: &BlockMeta, samples: &[f32]) -> Result<()> {
        let idx = self
            .streams
            .iter()
            .position(|s| s.source == source)
            .ok_or(JournalError::UnknownSource(source))?;
        let stream = &mut self.streams[idx];
        let ch = stream.format.channels;
        if !samples.len().is_multiple_of(ch as usize)
            || samples.len() / ch as usize != meta.frames as usize
        {
            return Err(JournalError::BadBlock {
                source_name: source,
                got: samples.len(),
                channels: ch,
            });
        }
        if meta.frames == 0 {
            return Ok(());
        }

        let clipped = f32_to_i16(samples, &mut self.pcm);
        match stream.codec {
            Codec::PcmS16le => {
                self.scratch.clear();
                self.scratch
                    .extend(self.pcm.iter().flat_map(|v| v.to_le_bytes()));
            }
            Codec::Lossless => crate::codec::encode(&self.pcm, ch as usize, &mut self.scratch),
        }
        if let Err(e) = stream.append(meta, &self.scratch) {
            stream.stats.io_errors += 1;
            stream.open = None; // abandon this segment; the reader stops at its last valid record
            return Err(e);
        }
        stream.stats.blocks += 1;
        stream.stats.frames += meta.frames as u64;
        stream.stats.bytes += self.scratch.len() as u64;
        stream.stats.pcm_bytes += (self.pcm.len() * BYTES_PER_SAMPLE) as u64;
        stream.stats.clipped_samples += clipped;

        if self.last_sync.elapsed() >= self.sync_interval {
            self.sync()?;
        }
        Ok(())
    }

    /// Makes everything written so far durable: data is fsync'd before its index.
    pub fn sync(&mut self) -> Result<()> {
        let mut first_err = None;
        for stream in &mut self.streams {
            if let Some(seg) = &mut stream.open
                && let Err(e) = seg.sync()
            {
                stream.stats.io_errors += 1;
                stream.open = None;
                first_err.get_or_insert(e);
            }
        }
        self.last_sync = Instant::now();
        first_err.map_or(Ok(()), Err)
    }

    /// Syncs everything and marks the journal as cleanly finished.
    pub fn finish(mut self) -> Result<Vec<(Source, WriterStats)>> {
        self.sync()?;
        self.manifest.finished = true;
        write_manifest(&self.dir, &self.manifest)?;
        self.finished = true;
        Ok(self.stats())
    }

    pub fn stats(&self) -> Vec<(Source, WriterStats)> {
        self.streams.iter().map(|s| (s.source, s.stats)).collect()
    }

    /// Test hook: closes files *without* flushing user-space buffers, as if the process had
    /// been killed. Data accepted after the last `sync` may be lost, exactly like a crash.
    #[doc(hidden)]
    pub fn simulate_crash(mut self) {
        for stream in &mut self.streams {
            if let Some(seg) = stream.open.take() {
                let (_pcm, _) = seg.pcm.into_parts();
                let (_idx, _) = seg.idx.into_parts();
            }
        }
        self.finished = true;
    }
}

impl Drop for JournalWriter {
    fn drop(&mut self) {
        if !self.finished
            && let Err(e) = self.sync()
        {
            tracing::warn!(error = %e, "journal sync on drop failed");
        }
    }
}

impl Stream {
    fn append(&mut self, meta: &BlockMeta, bytes: &[u8]) -> Result<()> {
        if self
            .open
            .as_ref()
            .is_some_and(|s| s.frames >= self.segment_frames)
        {
            let mut full = self.open.take().expect("checked");
            full.sync()?;
        }
        if self.open.is_none() {
            self.open = Some(self.open_segment()?);
        }
        let seg = self.open.as_mut().expect("opened above");
        let record = IndexRecord {
            byte_offset: seg.pcm_len,
            stored_len: u32::try_from(bytes.len()).map_err(|_| {
                JournalError::Config(format!("block of {} bytes is too large", bytes.len()))
            })?,
            meta: *meta,
            data_crc: crc32fast::hash(bytes),
        };
        seg.pcm
            .write_all(bytes)
            .map_err(|e| JournalError::io(&seg.pcm_path, e))?;
        seg.idx
            .write_all(&record.encode())
            .map_err(|e| JournalError::io(&seg.idx_path, e))?;
        seg.pcm_len += bytes.len() as u64;
        seg.frames += meta.frames as u64;
        Ok(())
    }

    fn open_segment(&mut self) -> Result<OpenSegment> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let (pcm_name, idx_name) = segment_file_names(seq);
        let (pcm_path, idx_path) = (self.dir.join(pcm_name), self.dir.join(idx_name));
        let open = |p: &Path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(p)
                .map_err(|e| JournalError::io(p, e))
        };
        let mut pcm = BufWriter::with_capacity(256 * 1024, open(&pcm_path)?);
        let idx = BufWriter::with_capacity(64 * INDEX_RECORD_LEN * 16, open(&idx_path)?);
        let header = SegmentHeader {
            source: self.source,
            format: self.format,
            seq,
            codec: self.codec,
        };
        pcm.write_all(&header.encode())
            .map_err(|e| JournalError::io(&pcm_path, e))?;
        sync_dir(&self.dir)?;
        self.stats.segments += 1;
        Ok(OpenSegment {
            pcm,
            idx,
            pcm_path,
            idx_path,
            pcm_len: SEGMENT_HEADER_LEN as u64,
            frames: 0,
        })
    }
}

impl OpenSegment {
    fn sync(&mut self) -> Result<()> {
        self.pcm
            .flush()
            .map_err(|e| JournalError::io(&self.pcm_path, e))?;
        self.pcm
            .get_ref()
            .sync_data()
            .map_err(|e| JournalError::io(&self.pcm_path, e))?;
        self.idx
            .flush()
            .map_err(|e| JournalError::io(&self.idx_path, e))?;
        self.idx
            .get_ref()
            .sync_data()
            .map_err(|e| JournalError::io(&self.idx_path, e))?;
        Ok(())
    }
}

/// Bytes of PCM per frame for a stream format.
pub(crate) fn frame_bytes(format: StreamFormat) -> usize {
    format.channels as usize * BYTES_PER_SAMPLE
}

fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    let tmp = dir.join(format!("{MANIFEST_FILE}.tmp"));
    let final_path = dir.join(MANIFEST_FILE);
    let json = serde_json::to_vec_pretty(m).expect("manifest serializes");
    {
        let mut f = File::create(&tmp).map_err(|e| JournalError::io(&tmp, e))?;
        f.write_all(&json).map_err(|e| JournalError::io(&tmp, e))?;
        f.sync_all().map_err(|e| JournalError::io(&tmp, e))?;
    }
    fs::rename(&tmp, &final_path).map_err(|e| JournalError::io(&final_path, e))?;
    Ok(())
}

/// Persists directory entries for newly created files (needed on Unix; NTFS journals metadata).
fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| JournalError::io(dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}
