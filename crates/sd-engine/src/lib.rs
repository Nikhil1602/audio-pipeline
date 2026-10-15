//! Recording session orchestration.
//!
//! ```text
//!  capture thread (per source) ──ring──► drain thread ──bounded queue (≥30 s)──► journal thread
//!                                          │  levels, drift, timeline checks         │ write + fsync,
//!                                          └──────────── RecorderEvent ◄─────────────┘ disk-full retry
//! ```
//!
//! The drain thread never touches the disk, so an fsync stall or a full disk cannot back up
//! into the real-time rings (finding 1 of the M0 report). The journal thread retries on disk-full
//! while the queue absorbs audio; only if the queue itself fills is audio dropped, and that is
//! counted and reported.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use sd_capture::{
    CaptureConfig, CaptureEvent, CaptureHandle, CaptureStatsSnapshot, DeviceSelector,
};
use sd_core::drift::{DriftConfig, DriftEstimator};
use sd_core::ring::{RingConsumer, RingStatsSnapshot};
use sd_core::{BlockFlags, BlockMeta, HostTime, Source, StreamFormat};
use sd_journal::{JournalConfig, JournalWriter, WriterStats};
use thiserror::Error;

pub mod processing;

pub use processing::{OfflineProcessor, ProcessedChunk, ProcessingConfig, ProcessingSnapshot};
use processing::{ProcBlock, ProcessingStats};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("capture: {0}")]
    Capture(#[from] sd_capture::CaptureError),
    #[error("journal: {0}")]
    Journal(#[from] sd_journal::JournalError),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("internal thread failed: {0}")]
    Thread(String),
}

pub type Result<T, E = EngineError> = std::result::Result<T, E>;

/// Final per-source journal statistics.
pub type JournalStats = Vec<(Source, WriterStats)>;

/// Free-space thresholds for the journal's volume.
#[derive(Debug, Clone)]
pub struct DiskPolicy {
    /// Below this, [`RecorderEvent::DiskSpaceLow`] is reported (once until space recovers).
    pub low_bytes: u64,
    /// Below this, journal writes pause *before* the disk is completely full; audio waits in
    /// the queue (`queue_seconds`) and [`RecorderEvent::JournalPaused`] is reported.
    pub reserve_bytes: u64,
    /// Writing resumes once at least this much is free again.
    pub resume_bytes: u64,
    pub check_interval: Duration,
}

impl Default for DiskPolicy {
    fn default() -> Self {
        const MIB: u64 = 1 << 20;
        Self {
            low_bytes: 500 * MIB,
            reserve_bytes: 50 * MIB,
            resume_bytes: 100 * MIB,
            check_interval: Duration::from_secs(2),
        }
    }
}

/// Optional maximum session length. The recorder only reports; stopping is the application's
/// decision.
#[derive(Debug, Clone, Copy)]
pub struct SessionLimit {
    pub max: Duration,
    /// [`RecorderEvent::SessionLimitApproaching`] is sent this long before `max`.
    pub warn_before: Duration,
}

/// Speech recognition for a recording: the engines (injected by the application) and the
/// streaming policy. Requires [`RecorderConfig::processing`] with VAD.
#[derive(Clone)]
pub struct AsrSetup {
    pub config: sd_asr::runner::AsrConfig,
    pub router: sd_asr::Router,
    /// Needed for `LanguageChoice::Auto`.
    pub detector: Option<Arc<dyn sd_asr::LanguageDetector>>,
    /// Speaker diarization of finals ([`diarization`]); None leaves speakers unset.
    pub speakers: Option<sd_asr::runner::SpeakerFactory>,
}

impl std::fmt::Debug for AsrSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrSetup")
            .field("config", &self.config)
            .field("languages", &self.router.supported())
            .field("detector", &self.detector.is_some())
            .field("speakers", &self.speakers.is_some())
            .finish()
    }
}

/// Live speaker diarization for transcripts (sd-diar), one tracker per session.
pub mod diarization {
    use std::path::Path;
    use std::sync::Arc;

    use sd_asr::runner::{SpeakerFactory, SpeakerTracker};
    use sd_asr::speakers::{SpeakerInfo, SpeakerSpan};
    use sd_asr::stream::Source;
    use sd_diar::embed::Embedder;
    use sd_diar::online::{OnlineConfig, OnlineDiarizer};
    use sd_diar::segment::SegmentationModel;

    /// Mic and system are diarized separately: speaker ids are per source.
    struct Tracker {
        sources: [OnlineDiarizer; 2],
    }

    fn info(d: &OnlineDiarizer, id: u64) -> SpeakerInfo {
        let Some(u) = d.utterance(id) else {
            return SpeakerInfo::default();
        };
        let spans = |v: Vec<(u64, u64, u32)>| -> Vec<SpeakerSpan> {
            v.into_iter()
                .map(|(start, end, speaker)| SpeakerSpan {
                    start,
                    end,
                    speaker,
                })
                .collect()
        };
        SpeakerInfo {
            speaker: Some(u.speaker),
            turns: spans(u.turns),
            overlaps: spans(u.overlaps),
        }
    }

    impl SpeakerTracker for Tracker {
        fn add(
            &mut self,
            source: Source,
            id: u64,
            start: u64,
            pcm: &[f32],
        ) -> (SpeakerInfo, Vec<(u64, SpeakerInfo)>) {
            let d = &mut self.sources[source as usize];
            match d.add(id, start, pcm) {
                Ok((_, revisions)) => {
                    let revised = revisions
                        .into_iter()
                        .filter(|&(r, _)| r != id)
                        .map(|(r, _)| (r, info(d, r)))
                        .collect();
                    (info(d, id), revised)
                }
                Err(e) => {
                    tracing::warn!("diarization failed for {source:?} utterance {id}: {e}");
                    (SpeakerInfo::default(), Vec::new())
                }
            }
        }

        fn finish(&mut self) -> Vec<(Source, u64, SpeakerInfo)> {
            let mut out = Vec::new();
            for (source, d) in [Source::Mic, Source::System]
                .into_iter()
                .zip(&mut self.sources)
            {
                for (id, _) in d.recluster() {
                    out.push((source, id, info(d, id)));
                }
            }
            out
        }
    }

    /// A factory of trackers sharing one embedding model (WeSpeaker ONNX) and, if given, one
    /// overlap detector (pyannote segmentation-3.0 ONNX).
    pub fn speaker_factory(
        model: &Path,
        segmentation: Option<&Path>,
        threads: usize,
        cfg: OnlineConfig,
    ) -> sd_diar::Result<SpeakerFactory> {
        let embedder = Arc::new(Embedder::new(model, threads)?);
        let segmenter = match segmentation {
            Some(p) => Some(Arc::new(SegmentationModel::new(p, threads)?)),
            None => None,
        };
        Ok(Arc::new(move || {
            let make = || {
                let d = OnlineDiarizer::new(embedder.clone(), cfg);
                match &segmenter {
                    Some(s) => d.with_segmentation(s.clone()),
                    None => d,
                }
            };
            Box::new(Tracker {
                sources: [make(), make()],
            }) as Box<dyn SpeakerTracker>
        }))
    }
}

/// Starts the recognition thread; its events become [`RecorderEvent::Asr`].
pub(crate) fn spawn_asr(
    setup: AsrSetup,
    inputs: Receiver<sd_asr::runner::AsrInput>,
    events: Sender<RecorderEvent>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("sd-asr".into())
        .spawn(move || {
            let mut runner =
                sd_asr::runner::AsrRunner::new(setup.config, setup.router, setup.detector);
            if let Some(factory) = &setup.speakers {
                runner = runner.with_speakers(factory());
            }
            runner.run_with(inputs, |e| {
                let _ = events.send(RecorderEvent::Asr(e));
            });
        })
        .expect("spawn ASR thread")
}

/// Reports free bytes on the journal's volume. Replaceable for tests.
#[derive(Clone)]
pub struct FreeSpaceProbe(pub Arc<dyn Fn() -> Option<u64> + Send + Sync>);

impl std::fmt::Debug for FreeSpaceProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FreeSpaceProbe")
    }
}

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub journal_dir: PathBuf,
    /// `None` disables microphone capture.
    pub mic: Option<DeviceSelector>,
    /// `None` disables system-audio capture.
    pub system: Option<DeviceSelector>,
    /// In-memory buffer between capture and disk, in seconds of audio per source.
    pub queue_seconds: u32,
    pub level_interval: Duration,
    pub health_interval: Duration,
    pub journal_sync_interval: Duration,
    pub journal_segment_seconds: u32,
    /// Request the raw (unprocessed) mic stream; see [`sd_capture::CaptureConfig::raw`].
    pub mic_raw: bool,
    /// Fault injection for tests: the mic stream stops being read this many ms after it starts.
    #[doc(hidden)]
    pub test_mic_stall_after_ms: Option<u32>,
    /// Real-time processing (echo cancellation, 16 kHz conversion, VAD). `None` records only.
    pub processing: Option<ProcessingConfig>,
    pub disk: DiskPolicy,
    /// `None` (default): no limit.
    pub session_limit: Option<SessionLimit>,
    /// Speech recognition (needs `processing` with VAD). `None`: no transcription.
    pub asr: Option<AsrSetup>,
    /// Overrides how free space is measured (tests). Default: the journal directory's volume.
    #[doc(hidden)]
    pub free_space_probe: Option<FreeSpaceProbe>,
}

impl RecorderConfig {
    pub fn new(journal_dir: impl Into<PathBuf>) -> Self {
        Self {
            journal_dir: journal_dir.into(),
            mic: Some(DeviceSelector::Default),
            system: Some(DeviceSelector::Default),
            queue_seconds: 30,
            level_interval: Duration::from_millis(100),
            health_interval: Duration::from_secs(1),
            journal_sync_interval: Duration::from_secs(1),
            journal_segment_seconds: 60,
            mic_raw: false,
            test_mic_stall_after_ms: None,
            disk: DiskPolicy::default(),
            session_limit: None,
            asr: None,
            free_space_probe: None,
            processing: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecorderEvent {
    Capture {
        source: Source,
        event: CaptureEvent,
    },
    Level {
        source: Source,
        rms_dbfs: f32,
        peak_dbfs: f32,
        at: HostTime,
    },
    Health(HealthReport),
    /// Writing the journal failed; audio is held in memory and writing is retried.
    JournalError {
        message: String,
        disk_full: bool,
    },
    /// Writing resumed after an error.
    JournalRecovered,
    /// The in-memory queue was full and audio was dropped (disk stalled for > `queue_seconds`).
    QueueOverflow {
        source: Source,
        dropped_frames: u64,
    },
    /// Speech began (16 kHz sample position, padding included). Followed by [`Self::Speech`]
    /// when it completes, or [`Self::SpeechDiscarded`] if it was too short to count.
    SpeechStart {
        source: Source,
        start_sample: u64,
    },
    /// The speech that began at `start_sample` ended too soon to be a segment.
    SpeechDiscarded {
        source: Source,
        start_sample: u64,
    },
    /// A speech segment completed (16 kHz sample positions on the source's processed timeline).
    Speech {
        source: Source,
        start_sample: u64,
        end_sample: u64,
        /// Mic only: part of the segment fell in a period where the far end was active but
        /// the echo gate had not yet decided whether there is acoustic echo, so echo there was
        /// not removed. The segment may be (partly) the far end leaking into the mic, and
        /// should be checked against the system transcript before use.
        echo_unverified: bool,
    },
    /// Free space on the journal's volume fell below [`DiskPolicy::low_bytes`].
    DiskSpaceLow {
        free_bytes: u64,
    },
    /// Journal writes paused to keep [`DiskPolicy::reserve_bytes`] free; audio is held in the
    /// queue (and dropped, with [`RecorderEvent::QueueOverflow`], if the pause outlasts it).
    JournalPaused {
        free_bytes: u64,
    },
    /// Free space recovered to [`DiskPolicy::resume_bytes`]; queued audio is written.
    JournalResumed {
        free_bytes: u64,
    },
    /// The session will reach [`SessionLimit::max`] in `remaining` (sent once).
    SessionLimitApproaching {
        remaining: Duration,
    },
    /// The session reached [`SessionLimit::max`] (sent once). Capture continues until the
    /// application stops the recorder.
    SessionLimitReached {
        elapsed: Duration,
    },
    /// Speech recognition output: partial and final transcripts, skips, timing.
    Asr(sd_asr::runner::AsrEvent),
    /// A non-fatal problem (e.g. an optional component could not start).
    Warning {
        message: String,
    },
    /// The mic echo canceller switched on or off (echo gate decision).
    EchoCanceller {
        active: bool,
        correlation: Option<f32>,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SourceHealth {
    pub source: Option<Source>,
    pub capture: Option<CaptureStatsSnapshot>,
    pub ring: RingStatsSnapshot,
    /// Frames seen on the timeline (including gap-fill silence).
    pub timeline_frames: u64,
    /// Device clock deviation from nominal vs. the host clock, once enough data exists.
    pub drift_ppm: Option<f64>,
    /// Standard error of `drift_ppm`.
    pub drift_ppm_stderr: Option<f64>,
    pub timestamp_jitter_ms: Option<f64>,
    pub queued_blocks: usize,
    pub queue_dropped_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct HealthReport {
    pub elapsed: Duration,
    pub sources: Vec<SourceHealth>,
    pub journaled_frames: u64,
    pub journal_errors: u64,
    pub processing: Option<ProcessingSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordingSummary {
    pub journal_dir: PathBuf,
    pub elapsed: Duration,
    pub health: HealthReport,
    pub journal: JournalStats,
}

/// An audio input for the recorder: a ring plus (optionally) the capture driving it.
pub struct SourceInput {
    pub source: Source,
    pub format: StreamFormat,
    pub ring: RingConsumer,
    pub capture: Option<CaptureHandle>,
}

struct QueuedBlock {
    source: Source,
    meta: BlockMeta,
    samples: Vec<f32>,
}

struct Shared {
    stop: AtomicBool,
    /// While set, captured audio is replaced by silence (see [`Recorder::set_paused`]).
    paused: AtomicBool,
    journaled_frames: AtomicU64,
    journal_errors: AtomicU64,
}

pub struct Recorder {
    captures: Vec<CaptureHandle>,
    drain: Option<JoinHandle<HealthReport>>,
    journal: Option<JoinHandle<Result<JournalStats>>>,
    shared: Arc<Shared>,
    events: Receiver<RecorderEvent>,
    journal_dir: PathBuf,
    started: Instant,
    processing: Option<JoinHandle<()>>,
    /// Speech recognition thread; ends after processing (its input is the processing stage).
    asr: Option<JoinHandle<()>>,
    processing_stats: Option<Arc<ProcessingStats>>,
    processed: Option<Receiver<ProcessedChunk>>,
}

impl Recorder {
    /// Starts capturing the configured devices into a new journal.
    pub fn start(cfg: RecorderConfig) -> Result<Self> {
        let mut inputs = Vec::new();
        for (source, sel) in [(Source::Mic, &cfg.mic), (Source::System, &cfg.system)] {
            if let Some(sel) = sel {
                let mut cc = CaptureConfig::new(source);
                cc.device = sel.clone();
                cc.raw = source == Source::Mic && cfg.mic_raw;
                if source == Source::Mic {
                    cc.test_stall_after_ms = cfg.test_mic_stall_after_ms;
                }
                let (handle, ring) = sd_capture::start_capture(cc)?;
                inputs.push(SourceInput {
                    source,
                    format: handle.format(),
                    ring,
                    capture: Some(handle),
                });
            }
        }
        Self::from_inputs(cfg, inputs)
    }

    /// Records from caller-provided rings (synthetic audio, files, app-supplied buffers).
    pub fn from_inputs(cfg: RecorderConfig, inputs: Vec<SourceInput>) -> Result<Self> {
        if inputs.is_empty() {
            return Err(EngineError::Config("no sources enabled".into()));
        }
        let mut jcfg = JournalConfig::new(inputs.iter().map(|i| (i.source, i.format)).collect());
        jcfg.sync_interval = cfg.journal_sync_interval;
        jcfg.segment_seconds = cfg.journal_segment_seconds;
        let writer = JournalWriter::create(&cfg.journal_dir, jcfg)?;

        // Queue capacity in blocks: assume blocks of ≥ 1 ms worst case would be excessive; size
        // for 10 ms blocks, which is what every backend delivers or larger.
        let blocks_per_source = (cfg.queue_seconds as usize * 100).max(100);
        let (qtx, qrx) =
            crossbeam_channel::bounded::<QueuedBlock>(blocks_per_source * inputs.len());
        let (etx, erx) = crossbeam_channel::unbounded();
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            journaled_frames: AtomicU64::new(0),
            journal_errors: AtomicU64::new(0),
        });

        // Optional processing stage, fed copies of drained blocks through a bounded queue.
        let formats: Vec<(Source, StreamFormat)> =
            inputs.iter().map(|i| (i.source, i.format)).collect();
        let mut processing = None;
        let mut processed = None;
        let mut proc_link = None;
        let mut processing_stats = None;
        let mut asr = None;
        if cfg.asr.is_some()
            && cfg
                .processing
                .as_ref()
                .is_none_or(|p| p.vad_model.is_none())
        {
            return Err(EngineError::Config(
                "speech recognition needs processing with a VAD model".into(),
            ));
        }
        if let Some(pcfg) = cfg.processing.clone() {
            let (atx, arx) = crossbeam_channel::bounded(pcfg.output_capacity.max(1));
            let stats = Arc::new(ProcessingStats::default());
            let mut stage = processing::Stage::new(pcfg, &formats, etx.clone(), atx, stats.clone())
                .map_err(EngineError::Config)?;
            if let Some(setup) = cfg.asr.clone() {
                let (tx, rx) = crossbeam_channel::unbounded();
                stage.set_asr(tx);
                asr = Some(spawn_asr(setup, rx, etx.clone()));
            }
            let (ptx, prx) =
                crossbeam_channel::bounded::<ProcBlock>(blocks_per_source * formats.len());
            processing = Some(
                std::thread::Builder::new()
                    .name("sd-process".into())
                    .spawn(move || processing::processing_thread(stage, prx))
                    .map_err(|e| EngineError::Thread(e.to_string()))?,
            );
            processed = Some(arx);
            processing_stats = Some(stats.clone());
            proc_link = Some((ptx, stats));
        }

        let mut captures = Vec::new();
        let mut drains = Vec::new();
        for input in inputs {
            if let Some(c) = input.capture {
                captures.push(c);
            }
            drains.push(DrainSource::new(
                input.source,
                input.format,
                input.ring,
                &cfg,
            ));
        }
        let capture_views: Vec<CaptureView> = captures
            .iter()
            .map(|c| CaptureView {
                source: c.source(),
                events: c.events().clone(),
                stats: c.stats_handle(),
            })
            .collect();

        let journal = {
            let shared = shared.clone();
            let etx = etx.clone();
            let disk = cfg.disk.clone();
            let probe = cfg.free_space_probe.clone().unwrap_or_else(|| {
                let dir = cfg.journal_dir.clone();
                FreeSpaceProbe(Arc::new(move || fs4::available_space(&dir).ok()))
            });
            std::thread::Builder::new()
                .name("sd-journal".into())
                .spawn(move || journal_thread(writer, qrx, etx, shared, disk, probe))
                .map_err(|e| EngineError::Thread(e.to_string()))?
        };
        let started = Instant::now();
        let session_limit = cfg.session_limit;
        let drain = {
            let shared = shared.clone();
            let health_interval = cfg.health_interval;
            std::thread::Builder::new()
                .name("sd-drain".into())
                .spawn(move || {
                    drain_thread(
                        drains,
                        capture_views,
                        DrainOutputs {
                            queue: qtx,
                            events: etx,
                            proc_link,
                            session_limit,
                        },
                        shared,
                        health_interval,
                        started,
                    )
                })
                .map_err(|e| EngineError::Thread(e.to_string()))?
        };

        Ok(Self {
            captures,
            drain: Some(drain),
            journal: Some(journal),
            shared,
            events: erx,
            journal_dir: cfg.journal_dir,
            started,
            processing,
            asr,
            processing_stats,
            processed,
        })
    }

    /// Processed 16 kHz audio (mic after echo handling, system downmixed), if processing is
    /// enabled. Chunks a consumer does not read in time are dropped and counted.
    pub fn processed(&self) -> Option<&Receiver<ProcessedChunk>> {
        self.processed.as_ref()
    }

    pub fn events(&self) -> &Receiver<RecorderEvent> {
        &self.events
    }

    /// Switches a source to another device; the timeline stays continuous.
    pub fn set_device(&self, source: Source, device: DeviceSelector) {
        for c in &self.captures {
            if c.source() == source {
                c.set_device(device.clone());
            }
        }
    }

    /// Pauses or resumes. While paused, capture keeps running but its audio is replaced by
    /// silence before anything else sees it (levels, journal, processing): nothing is recorded
    /// or transcribed, speech in progress ends, and the timelines stay continuous and aligned,
    /// so times after a pause still count from the start.
    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::Relaxed)
    }

    /// Stops capture, flushes everything queued to disk and finishes the journal.
    pub fn stop(mut self) -> Result<RecordingSummary> {
        tracing::debug!("stop: captures");
        for c in self.captures.drain(..) {
            c.stop();
        }
        self.shared.stop.store(true, Ordering::Release);
        tracing::debug!("stop: drain thread");
        let mut health = self
            .drain
            .take()
            .expect("drain thread")
            .join()
            .map_err(|_| EngineError::Thread("drain thread panicked".into()))?;
        tracing::debug!("stop: journal thread");
        let journal = self
            .journal
            .take()
            .expect("journal thread")
            .join()
            .map_err(|_| EngineError::Thread("journal thread panicked".into()))??;
        // The drain thread's last report predates the journal thread writing its backlog.
        health.journaled_frames = self.shared.journaled_frames.load(Ordering::Relaxed);
        health.journal_errors = self.shared.journal_errors.load(Ordering::Relaxed);
        // The drain thread owned the processing queue's sender, so processing now drains its
        // remaining input, flushes open segments and exits.
        tracing::debug!("stop: processing thread");
        if let Some(p) = self.processing.take() {
            p.join()
                .map_err(|_| EngineError::Thread("processing thread panicked".into()))?;
        }
        // Recognition finishes its queued finals once processing has closed its input.
        tracing::debug!("stop: ASR thread");
        if let Some(a) = self.asr.take() {
            a.join()
                .map_err(|_| EngineError::Thread("ASR thread panicked".into()))?;
        }
        // Likewise for processing, which handles the end of the session after the drain stops.
        if let Some(s) = &self.processing_stats {
            health.processing = Some(s.snapshot());
        }
        tracing::debug!("stop: done");
        Ok(RecordingSummary {
            journal_dir: self.journal_dir.clone(),
            elapsed: self.started.elapsed(),
            health,
            journal,
        })
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for c in self.captures.drain(..) {
            c.stop();
        }
        if let Some(d) = self.drain.take() {
            let _ = d.join();
        }
        if let Some(j) = self.journal.take() {
            let _ = j.join();
        }
        if let Some(p) = self.processing.take() {
            let _ = p.join();
        }
        if let Some(a) = self.asr.take() {
            let _ = a.join();
        }
    }
}

struct CaptureView {
    source: Source,
    events: Receiver<CaptureEvent>,
    stats: Arc<sd_capture::CaptureStats>,
}

/// Per-source state owned by the drain thread.
struct DrainSource {
    source: Source,
    format: StreamFormat,
    ring: RingConsumer,
    drift: DriftEstimator,
    level_frames: u64,
    level_window: u64,
    sum_sq: f64,
    peak: f32,
    timeline_frames: u64,
    queue_dropped_frames: u64,
    pending_drop_report: u64,
}

impl DrainSource {
    fn new(source: Source, format: StreamFormat, ring: RingConsumer, cfg: &RecorderConfig) -> Self {
        Self {
            source,
            format,
            ring,
            drift: DriftEstimator::new(DriftConfig::new(format.sample_rate)),
            level_frames: 0,
            level_window: format
                .ns_to_frames(cfg.level_interval.as_nanos() as u64)
                .max(1),
            sum_sq: 0.0,
            peak: 0.0,
            timeline_frames: 0,
            queue_dropped_frames: 0,
            pending_drop_report: 0,
        }
    }
}

fn dbfs(x: f64) -> f32 {
    (20.0 * x.max(1e-9).log10()) as f32
}

/// Where the drain thread sends things: the journal queue, events, and (optionally) the
/// processing stage with its counters.
struct DrainOutputs {
    queue: Sender<QueuedBlock>,
    events: Sender<RecorderEvent>,
    proc_link: Option<(Sender<ProcBlock>, Arc<ProcessingStats>)>,
    session_limit: Option<SessionLimit>,
}

fn drain_thread(
    mut sources: Vec<DrainSource>,
    captures: Vec<CaptureView>,
    outputs: DrainOutputs,
    shared: Arc<Shared>,
    health_interval: Duration,
    started: Instant,
) -> HealthReport {
    let DrainOutputs {
        queue,
        events,
        proc_link,
        session_limit,
    } = outputs;
    let (mut limit_warned, mut limit_reached) = (false, false);
    let proc_stats = proc_link.as_ref().map(|(_, s)| s.clone());
    let mut buf = Vec::new();
    let mut next_health = Instant::now() + health_interval;
    loop {
        let stopping = shared.stop.load(Ordering::Acquire);
        let mut any = false;
        for cv in &captures {
            while let Ok(event) = cv.events.try_recv() {
                let _ = events.send(RecorderEvent::Capture {
                    source: cv.source,
                    event,
                });
            }
        }
        for s in &mut sources {
            while let Some(meta) = s.ring.pop(&mut buf) {
                any = true;
                if shared.paused.load(Ordering::Relaxed) {
                    buf.fill(0.0);
                }
                s.timeline_frames = meta.first_frame + meta.frames as u64;
                let f = meta.flags;
                if f.contains(BlockFlags::GAP_FILL) {
                    // Only fit drift within continuous runs of real device data. Device
                    // DISCONTINUITY flags alone are not breaks: some drivers raise them on every
                    // timestamp resync (Bluetooth hands-free: every ~90 ms) without losing audio;
                    // the aligner decides what is a real gap and marks it GAP_FILL.
                    s.drift.break_segment();
                }
                if !f.contains(BlockFlags::GAP_FILL) && !f.contains(BlockFlags::TIMESTAMP_ESTIMATED)
                {
                    s.drift.observe(meta.host_time, meta.first_frame);
                }
                // Level metering over all channels.
                let ch = s.format.channels as usize;
                for frame in buf.chunks_exact(ch) {
                    for &x in frame {
                        s.sum_sq += (x as f64) * (x as f64);
                        s.peak = s.peak.max(x.abs());
                    }
                    s.level_frames += 1;
                    if s.level_frames >= s.level_window {
                        let rms = (s.sum_sq / (s.level_frames * ch as u64) as f64).sqrt();
                        let _ = events.send(RecorderEvent::Level {
                            source: s.source,
                            rms_dbfs: dbfs(rms),
                            peak_dbfs: dbfs(s.peak as f64),
                            at: meta.host_time,
                        });
                        s.level_frames = 0;
                        s.sum_sq = 0.0;
                        s.peak = 0.0;
                    }
                }
                if let Some((tx, stats)) = &proc_link
                    && tx
                        .try_send(ProcBlock {
                            source: s.source,
                            meta,
                            samples: journal_precision(&buf),
                        })
                        .is_err()
                {
                    stats.dropped_input_blocks.fetch_add(1, Ordering::Relaxed);
                }
                let block = QueuedBlock {
                    source: s.source,
                    meta,
                    samples: std::mem::take(&mut buf),
                };
                match queue.try_send(block) {
                    Ok(()) => {}
                    Err(TrySendError::Full(b)) | Err(TrySendError::Disconnected(b)) => {
                        s.queue_dropped_frames += b.meta.frames as u64;
                        s.pending_drop_report += b.meta.frames as u64;
                        buf = b.samples;
                    }
                }
            }
            if s.pending_drop_report > 0 {
                let _ = events.send(RecorderEvent::QueueOverflow {
                    source: s.source,
                    dropped_frames: s.pending_drop_report,
                });
                s.pending_drop_report = 0;
            }
        }

        if let Some(limit) = session_limit {
            let elapsed = started.elapsed();
            if !limit_warned && elapsed + limit.warn_before >= limit.max {
                limit_warned = true;
                let _ = events.send(RecorderEvent::SessionLimitApproaching {
                    remaining: limit.max.saturating_sub(elapsed),
                });
            }
            if !limit_reached && elapsed >= limit.max {
                limit_reached = true;
                let _ = events.send(RecorderEvent::SessionLimitReached { elapsed });
            }
        }

        let done = stopping && !any && sources.iter().all(|s| s.ring.queued_blocks() == 0);
        if done || Instant::now() >= next_health {
            let report = health(
                &sources,
                &captures,
                &queue,
                &shared,
                started,
                proc_stats.as_deref(),
            );
            if done {
                return report;
            }
            let _ = events.send(RecorderEvent::Health(report));
            next_health += health_interval;
        }
        if !any {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn health(
    sources: &[DrainSource],
    captures: &[CaptureView],
    queue: &Sender<QueuedBlock>,
    shared: &Shared,
    started: Instant,
    proc_stats: Option<&ProcessingStats>,
) -> HealthReport {
    HealthReport {
        processing: proc_stats.map(|s| s.snapshot()),
        elapsed: started.elapsed(),
        sources: sources
            .iter()
            .map(|s| {
                let est = s.drift.estimate();
                SourceHealth {
                    source: Some(s.source),
                    capture: captures
                        .iter()
                        .find(|c| c.source == s.source)
                        .map(|c| c.stats.snapshot()),
                    ring: s.ring.stats(),
                    timeline_frames: s.timeline_frames,
                    drift_ppm: est.map(|e| e.ppm),
                    drift_ppm_stderr: est.map(|e| e.ppm_stderr),
                    timestamp_jitter_ms: est.map(|e| e.jitter_rms_ns / 1e6),
                    queued_blocks: queue.len(),
                    queue_dropped_frames: s.queue_dropped_frames,
                }
            })
            .collect(),
        journaled_frames: shared.journaled_frames.load(Ordering::Relaxed),
        journal_errors: shared.journal_errors.load(Ordering::Relaxed),
    }
}

/// The samples exactly as the journal will store them (16-bit), so that reprocessing the
/// journal with [`OfflineProcessor`] reproduces the live output bit for bit.
fn journal_precision(samples: &[f32]) -> Vec<f32> {
    let mut q = Vec::with_capacity(samples.len());
    sd_journal::f32_to_i16(samples, &mut q);
    q.into_iter().map(sd_journal::i16_to_f32).collect()
}

fn journal_thread(
    mut writer: JournalWriter,
    queue: Receiver<QueuedBlock>,
    events: Sender<RecorderEvent>,
    shared: Arc<Shared>,
    disk: DiskPolicy,
    probe: FreeSpaceProbe,
) -> Result<JournalStats> {
    let mut failing = false;
    let mut paused = false;
    let mut low_reported = false;
    let mut next_check = Instant::now();
    loop {
        if Instant::now() >= next_check {
            next_check = Instant::now() + disk.check_interval;
            if let Some(free) = (probe.0)() {
                if free < disk.low_bytes && !low_reported {
                    low_reported = true;
                    let _ = events.send(RecorderEvent::DiskSpaceLow { free_bytes: free });
                } else if free >= disk.low_bytes + disk.low_bytes / 5 {
                    low_reported = false; // hysteresis: re-arm only after a clear recovery
                }
                if !paused && free < disk.reserve_bytes {
                    paused = true;
                    let _ = writer.sync(); // persist everything accepted so far
                    let _ = events.send(RecorderEvent::JournalPaused { free_bytes: free });
                } else if paused && free >= disk.resume_bytes {
                    paused = false;
                    let _ = events.send(RecorderEvent::JournalResumed { free_bytes: free });
                }
            }
        }
        // While paused, leave audio in the queue. On stop, try to write anyway: the retry loop
        // below gives up on blocks that still fail.
        if paused && !shared.stop.load(Ordering::Acquire) {
            std::thread::sleep(disk.check_interval.min(Duration::from_millis(200)));
            continue;
        }
        // Ends when the drain thread exits and drops the sender, after draining everything.
        let wait = disk.check_interval.min(Duration::from_millis(200));
        let block = match queue.recv_timeout(wait) {
            Ok(b) => b,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        let mut attempt = 0u32;
        loop {
            match writer.write(block.source, &block.meta, &block.samples) {
                Ok(()) => {
                    shared
                        .journaled_frames
                        .fetch_add(block.meta.frames as u64, Ordering::Relaxed);
                    if failing {
                        failing = false;
                        let _ = events.send(RecorderEvent::JournalRecovered);
                    }
                    break;
                }
                Err(e) => {
                    shared.journal_errors.fetch_add(1, Ordering::Relaxed);
                    attempt += 1;
                    if !failing || attempt == 1 {
                        let _ = events.send(RecorderEvent::JournalError {
                            message: e.to_string(),
                            disk_full: e.is_disk_full(),
                        });
                    }
                    failing = true;
                    // Retry with backoff; the queue keeps absorbing audio meanwhile. Give up on
                    // this block only after stop was requested and ~10 s of retries failed.
                    if shared.stop.load(Ordering::Acquire) && attempt >= 20 {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis((100 * attempt as u64).min(1_000)));
                }
            }
        }
    }
    Ok(writer.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_core::BlockFlags;
    use sd_core::ring::audio_ring;
    use sd_journal::JournalReader;

    fn synthetic_input(
        source: Source,
        channels: u16,
    ) -> (SourceInput, sd_core::ring::RingProducer) {
        let format = StreamFormat::new(48_000, channels).unwrap();
        let (p, c) = audio_ring(channels, 400_000, 2_048).unwrap(); // holds the whole burst
        (
            SourceInput {
                source,
                format,
                ring: c,
                capture: None,
            },
            p,
        )
    }

    #[test]
    fn session_limit_warns_then_reports_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = RecorderConfig::new(dir.path().join("j"));
        cfg.session_limit = Some(SessionLimit {
            max: Duration::from_millis(600),
            warn_before: Duration::from_millis(300),
        });
        let (mic, pm) = synthetic_input(Source::Mic, 1);
        let rec = Recorder::from_inputs(cfg, vec![mic]).unwrap();
        let t0 = Instant::now();
        let mut seen = Vec::new();
        while t0.elapsed() < Duration::from_millis(1_500) {
            if let Ok(ev) = rec.events().recv_timeout(Duration::from_millis(20)) {
                match ev {
                    RecorderEvent::SessionLimitApproaching { remaining } => {
                        assert!(remaining <= Duration::from_millis(300));
                        seen.push(("approaching", t0.elapsed()));
                    }
                    RecorderEvent::SessionLimitReached { elapsed } => {
                        assert!(elapsed >= Duration::from_millis(600));
                        seen.push(("reached", t0.elapsed()));
                    }
                    _ => {}
                }
            }
        }
        drop(pm);
        rec.stop().unwrap();
        let names: Vec<_> = seen.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["approaching", "reached"], "{seen:?}");
        assert!(seen[0].1 >= Duration::from_millis(250), "{seen:?}");
    }

    /// While paused, the audio is replaced by silence: the timeline stays continuous and only
    /// the paused stretch is silent.
    #[test]
    fn pause_replaces_audio_with_silence() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = RecorderConfig::new(dir.path().join("j"));
        cfg.health_interval = Duration::from_millis(20);
        let (mic, mut pm) = synthetic_input(Source::Mic, 1);
        let rec = Recorder::from_inputs(cfg, vec![mic]).unwrap();
        let block = vec![0.5f32; 480];
        let mut i = 0u64;
        let mut push = |n: u64| {
            for _ in 0..n {
                let t = HostTime(1_000_000_000_000 + i * 10_000_000);
                pm.push(&block, t, BlockFlags::empty());
                i += 1;
            }
        };
        let journaled = |rec: &Recorder, n: u64| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if let Ok(RecorderEvent::Health(h)) =
                    rec.events().recv_timeout(Duration::from_millis(50))
                    && h.journaled_frames == n
                {
                    return;
                }
            }
            panic!("never journaled {n} frames");
        };
        push(50);
        journaled(&rec, 50 * 480);
        rec.set_paused(true);
        assert!(rec.is_paused());
        push(50);
        journaled(&rec, 100 * 480);
        rec.set_paused(false);
        push(50);
        journaled(&rec, 150 * 480);
        drop(pm);
        let summary = rec.stop().unwrap();

        let reader = JournalReader::open(&summary.journal_dir).unwrap();
        let mut samples = Vec::new();
        for b in reader.blocks(Source::Mic).unwrap() {
            samples.extend(b.unwrap().samples);
        }
        assert_eq!(samples.len(), 150 * 480);
        let (before, rest) = samples.split_at(50 * 480);
        let (paused, after) = rest.split_at(50 * 480);
        assert!(before.iter().all(|&v| v != 0), "audio before the pause");
        assert!(paused.iter().all(|&v| v == 0), "silence while paused");
        assert!(after.iter().all(|&v| v != 0), "audio after resuming");
    }

    #[test]
    fn real_free_space_probe_reports_a_value() {
        let dir = tempfile::tempdir().unwrap();
        let free = fs4::available_space(dir.path()).expect("free space");
        assert!(free > 0);
    }

    /// Low disk space pauses journal writes before the disk is full; audio waits in the queue
    /// and is written once space returns, so nothing is lost.
    #[test]
    fn low_disk_pauses_and_resumes_journal_without_loss() {
        const MIB: u64 = 1 << 20;
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = RecorderConfig::new(dir.path().join("j"));
        let free = Arc::new(AtomicU64::new(1024 * MIB));
        let f = free.clone();
        cfg.free_space_probe = Some(FreeSpaceProbe(Arc::new(move || {
            Some(f.load(Ordering::Relaxed))
        })));
        cfg.disk.check_interval = Duration::from_millis(20);
        cfg.health_interval = Duration::from_millis(50);
        let (mic, mut pm) = synthetic_input(Source::Mic, 1);
        let rec = Recorder::from_inputs(cfg, vec![mic]).unwrap();
        let block = vec![0.1f32; 480];
        let mut i = 0u64;
        let mut push = |n: u64| {
            for _ in 0..n {
                let t = HostTime(1_000_000_000_000 + i * 10_000_000);
                pm.push(&block, t, BlockFlags::empty());
                i += 1;
            }
        };
        // Collects events until the predicate holds for the latest one (or 10 s pass).
        let wait = |rec: &Recorder, done: &dyn Fn(&RecorderEvent) -> bool| {
            let mut got = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if let Ok(ev) = rec.events().recv_timeout(Duration::from_millis(50)) {
                    let stop = done(&ev);
                    got.push(ev);
                    if stop {
                        return got;
                    }
                }
            }
            panic!("timed out; events: {got:?}");
        };
        let journaled_is = |n: u64| move |ev: &RecorderEvent| matches!(ev, RecorderEvent::Health(h) if h.journaled_frames == n);

        push(100);
        wait(&rec, &journaled_is(100 * 480));

        free.store(400 * MIB, Ordering::Relaxed); // low, not yet critical
        wait(&rec, &|ev| matches!(ev, RecorderEvent::DiskSpaceLow { .. }));
        free.store(10 * MIB, Ordering::Relaxed); // below the 50 MiB reserve: pause
        wait(&rec, &|ev| {
            matches!(ev, RecorderEvent::JournalPaused { .. })
        });
        push(200);
        std::thread::sleep(Duration::from_millis(300));
        let during: Vec<u64> = rec
            .events()
            .try_iter()
            .filter_map(|ev| match ev {
                RecorderEvent::Health(h) => Some(h.journaled_frames),
                _ => None,
            })
            .collect();
        assert!(!during.is_empty());
        assert!(
            during.iter().all(|&n| n == 100 * 480),
            "nothing written while paused: {during:?}"
        );

        free.store(200 * MIB, Ordering::Relaxed); // above the 100 MiB resume level
        wait(&rec, &|ev| {
            matches!(ev, RecorderEvent::JournalResumed { .. })
        });
        wait(&rec, &journaled_is(300 * 480));
        drop(pm);
        let summary = rec.stop().unwrap();
        assert_eq!(
            summary.health.journaled_frames,
            300 * 480,
            "queued audio written"
        );
        let r = JournalReader::open(&summary.journal_dir).unwrap();
        assert_eq!(r.scan(Source::Mic).unwrap().frames, 300 * 480);
    }

    #[test]
    fn records_synthetic_sources_to_journal_with_levels_and_drift() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = RecorderConfig::new(dir.path().join("j"));
        cfg.health_interval = Duration::from_millis(200);
        let (mic, mut pm) = synthetic_input(Source::Mic, 1);
        let (sys, mut ps) = synthetic_input(Source::System, 2);
        let rec = Recorder::from_inputs(cfg, vec![mic, sys]).unwrap();

        // 8 s of audio, timestamped on a 48 kHz clock running 100 ppm fast, delivered quickly.
        let t0 = 1_000_000_000_000u64;
        let rate = 48_000.0 * (1.0 + 100e-6);
        let mono = vec![0.5f32; 480];
        let stereo = vec![0.25f32; 960];
        for i in 0..800u64 {
            let t = HostTime(t0 + (i as f64 * 480.0 / rate * 1e9) as u64);
            assert_eq!(
                pm.push(&mono, t, BlockFlags::empty()),
                sd_core::ring::PushOutcome::Written
            );
            assert_eq!(
                ps.push(&stereo, t, BlockFlags::empty()),
                sd_core::ring::PushOutcome::Written
            );
            if i % 50 == 0 {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        std::thread::sleep(Duration::from_millis(300));
        let mut levels = 0;
        let mut health = None;
        while let Ok(ev) = rec.events().try_recv() {
            match ev {
                RecorderEvent::Level {
                    source: Source::Mic,
                    rms_dbfs,
                    ..
                } => {
                    assert!((rms_dbfs - (-6.02)).abs() < 0.1, "mic rms {rms_dbfs}");
                    levels += 1;
                }
                RecorderEvent::Health(h) => health = Some(h),
                _ => {}
            }
        }
        assert_eq!(levels, 80, "one level event per 100 ms of mic audio");
        let h = health.expect("health events");
        let mic_h = h
            .sources
            .iter()
            .find(|s| s.source == Some(Source::Mic))
            .unwrap();
        let ppm = mic_h.drift_ppm.expect("drift estimate");
        assert!((ppm - 100.0).abs() < 1.0, "drift {ppm}");

        drop((pm, ps));
        let summary = rec.stop().unwrap();
        assert_eq!(summary.health.journaled_frames, 2 * 800 * 480);
        let r = JournalReader::open(&summary.journal_dir).unwrap();
        assert!(r.manifest().finished);
        assert_eq!(r.scan(Source::Mic).unwrap().frames, 384_000);
        assert_eq!(r.scan(Source::System).unwrap().frames, 384_000);
    }

    #[test]
    fn no_sources_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            Recorder::from_inputs(RecorderConfig::new(dir.path()), vec![]),
            Err(EngineError::Config(_))
        ));
    }
}

#[cfg(test)]
mod processing_tests {
    use super::*;
    use sd_core::ring::audio_ring;
    use sd_dsp::mic::{AecMode, MicConfig};

    const R: f64 = 48_000.0;

    /// Deterministic broadband "speech-like" far-end defined on true time (seconds), so clocks
    /// with different rates can sample it consistently. Linear interpolation of a noise grid.
    struct Far {
        grid: Vec<f32>,
    }
    impl Far {
        fn new(seconds: f64) -> Self {
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            // Band-limited like real device audio (two one-pole low-passes, ~1 kHz corner):
            // white noise up to 24 kHz would make the test's own grid interpolation, not the
            // drift tracking, the limiting factor once clocks differ.
            let (mut lp, mut lp2) = (0.0f32, 0.0f32);
            let grid = (0..(seconds * R) as usize + 4800)
                .map(|i| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    lp = 0.85 * lp + 0.15 * ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0);
                    lp2 = 0.85 * lp2 + 0.15 * lp;
                    let lp = lp2 * 4.0;
                    let t = i as f64 / R;
                    let env =
                        (0.5 + 0.5 * (2.0 * std::f64::consts::PI * 3.7 * t).sin()).powi(2) as f32;
                    lp * env * 0.2
                })
                .collect();
            Self { grid }
        }
        fn at(&self, t: f64) -> f32 {
            if t < 0.0 {
                return 0.0;
            }
            let p = t * R;
            let i = p as usize;
            if i + 1 >= self.grid.len() {
                return 0.0;
            }
            let f = (p - i as f64) as f32;
            self.grid[i] * (1.0 - f) + self.grid[i + 1] * f
        }
    }

    struct Outcome {
        mic: Vec<ProcessedChunk>,
        sys: Vec<ProcessedChunk>,
        echo_on: bool,
        stats: ProcessingSnapshot,
        /// The journal reprocessed with `OfflineProcessor` (threaded runs only).
        replay: Option<(Vec<ProcessedChunk>, Vec<ProcessedChunk>)>,
    }

    /// Reprocesses a journal the way `sdctl process` does: blocks of both sources interleaved
    /// in host-time order.
    fn replay(
        dir: &std::path::Path,
        cfg: ProcessingConfig,
    ) -> (Vec<ProcessedChunk>, Vec<ProcessedChunk>) {
        let r = sd_journal::JournalReader::open(dir).unwrap();
        let formats = r.sources();
        let mut p = OfflineProcessor::new(cfg, &formats).unwrap();
        let mut blocks: Vec<(Source, sd_journal::RecoveredBlock)> = Vec::new();
        for &(s, _) in &formats {
            blocks.extend(r.blocks(s).unwrap().map(|b| (s, b.unwrap())));
        }
        // Stable sort keeps each source's own order for equal timestamps.
        blocks.sort_by_key(|(_, b)| b.meta.host_time);
        let (mut mic, mut sys) = (Vec::new(), Vec::new());
        let mut take = |a: Vec<ProcessedChunk>| {
            for c in a {
                match c.source {
                    Source::Mic => mic.push(c),
                    Source::System => sys.push(c),
                }
            }
        };
        for (s, b) in blocks {
            let x: Vec<f32> = b
                .samples
                .iter()
                .map(|&v| sd_journal::i16_to_f32(v))
                .collect();
            p.push(s, b.meta, &x);
            take(p.drain().0);
        }
        take(p.finish().0);
        (mic, sys)
    }

    /// Feeds `seconds` of mic (echo of the far end, 30 ms delay, -6 dB) and system audio
    /// (starting `sys_offset_ms` later, clock `sys_ppm` fast) through a Recorder with processing.
    fn run(seconds: f64, sys_offset_ms: f64, sys_ppm: f64) -> Outcome {
        let dir = tempfile::tempdir().unwrap();
        let far = Far::new(seconds + 1.0);
        let mut cfg = RecorderConfig::new(dir.path().join("j"));
        cfg.health_interval = Duration::from_millis(100);
        let pcfg = ProcessingConfig {
            mic: MicConfig {
                aec: AecMode::Auto,
                gain_control: false,
                noise_suppression: None,
                ..Default::default()
            },
            vad_model: None,
            output_capacity: 10_000,
            ..Default::default()
        };
        cfg.processing = Some(pcfg.clone());
        let mic_fmt = StreamFormat::new(48_000, 1).unwrap();
        let sys_fmt = StreamFormat::new(48_000, 2).unwrap();
        let (mut pm, cm) = audio_ring(1, 48_000 * 4, 4_096).unwrap();
        let (mut ps, cs) = audio_ring(2, 48_000 * 4, 4_096).unwrap();
        let rec = Recorder::from_inputs(
            cfg,
            vec![
                SourceInput {
                    source: Source::Mic,
                    format: mic_fmt,
                    ring: cm,
                    capture: None,
                },
                SourceInput {
                    source: Source::System,
                    format: sys_fmt,
                    ring: cs,
                    capture: None,
                },
            ],
        )
        .unwrap();
        let t0 = 1_000_000_000_000u64;
        let sys_rate = R * (1.0 + sys_ppm * 1e-6);
        let off = sys_offset_ms / 1e3;
        let blocks = (seconds * 100.0) as u64;
        let mut mic = vec![0.0f32; 480];
        let mut sys = vec![0.0f32; 960];
        for i in 0..blocks {
            for (k, m) in mic.iter_mut().enumerate() {
                let t = (i * 480 + k as u64) as f64 / R;
                *m = 0.5 * far.at(t - 0.030);
            }
            pm.push(
                &mic,
                HostTime(t0 + (i as f64 * 0.01 * 1e9) as u64),
                BlockFlags::empty(),
            );
            for k in 0..480 {
                let t = off + (i * 480 + k as u64) as f64 / sys_rate;
                let v = far.at(t);
                sys[2 * k] = v;
                sys[2 * k + 1] = v;
            }
            ps.push(
                &sys,
                HostTime(t0 + ((off + (i * 480) as f64 / sys_rate) * 1e9) as u64),
                BlockFlags::empty(),
            );
            // ~5x real time: realistic interleaving without making the test slow.
            std::thread::sleep(Duration::from_millis(2));
        }
        drop((pm, ps));
        let arx = rec.processed().unwrap().clone();
        let erx = rec.events().clone();
        let summary = rec.stop().unwrap();
        let (mut mic_c, mut sys_c) = (Vec::new(), Vec::new());
        while let Ok(c) = arx.try_recv() {
            match c.source {
                Source::Mic => mic_c.push(c),
                Source::System => sys_c.push(c),
            }
        }
        let echo_on = erx
            .try_iter()
            .any(|e| matches!(e, RecorderEvent::EchoCanceller { active: true, .. }));
        Outcome {
            mic: mic_c,
            sys: sys_c,
            echo_on,
            stats: summary.health.processing.unwrap(),
            replay: Some(replay(&summary.journal_dir, pcfg)),
        }
    }

    fn continuous(chunks: &[ProcessedChunk]) -> usize {
        let mut next = 0;
        for c in chunks {
            assert_eq!(c.first_sample, next, "{:?} chunk out of sequence", c.source);
            next += c.samples.len() as u64;
        }
        next as usize
    }

    fn power_db(x: &[f32]) -> f64 {
        10.0 * (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64 + 1e-20).log10()
    }

    #[test]
    fn processing_aligns_cancels_echo_and_streams_continuously() {
        let o = run(20.0, 37.0, 0.0);
        let mic_n = continuous(&o.mic);
        let sys_n = continuous(&o.sys);
        assert!(
            mic_n >= 19 * 16_000 && sys_n >= 19 * 16_000,
            "mic {mic_n} sys {sys_n} samples"
        );
        assert!(o.echo_on, "echo gate must engage");
        assert_eq!(o.stats.reference_timeouts, 0, "{:?}", o.stats);
        assert_eq!(o.stats.reference_resyncs, 0, "{:?}", o.stats);
        let all: Vec<f32> = o
            .mic
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect();
        let sys: Vec<f32> = o
            .sys
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect();
        // Echo reaches the mic at -6 dB; after convergence it must be largely removed.
        let erle = power_db(&sys[12 * 16_000..19 * 16_000])
            - 6.0
            - power_db(&all[12 * 16_000..19 * 16_000]);
        println!("aligned: ERLE ~{erle:.1} dB, stats {:?}", o.stats);
        assert!(erle > 20.0, "ERLE {erle:.1} dB");

        // With nothing dropped, late or resynced, reprocessing the journal must reproduce the
        // live output exactly (the live stage sees samples at journal precision).
        let (rm, rs) = o.replay.as_ref().unwrap();
        assert_eq!(rm.len(), o.mic.len(), "replayed mic chunks");
        assert_eq!(rs.len(), o.sys.len(), "replayed system chunks");
        for (a, b) in o.mic.iter().chain(&o.sys).zip(rm.iter().chain(rs)) {
            assert_eq!(a.first_sample, b.first_sample, "{:?}", a.source);
            assert!(
                a.samples == b.samples,
                "{:?} chunk at {} differs on replay",
                a.source,
                a.first_sample
            );
        }
    }

    /// Same scenario through `OfflineProcessor`, with blocks delivered in strict host-time
    /// order: deterministic on any machine (thread scheduling on slow CI runners can reorder
    /// live deliveries, which is covered by the threaded test above).
    /// The system stream stops `sys_stops_early` blocks before the mic, as at a real stop.
    fn run_offline(
        seconds: f64,
        sys_offset_ms: f64,
        sys_ppm: f64,
        sys_stops_early: u64,
    ) -> Outcome {
        let far = Far::new(seconds + 1.0);
        let cfg = ProcessingConfig {
            mic: MicConfig {
                aec: AecMode::Auto,
                gain_control: false,
                noise_suppression: None,
                ..Default::default()
            },
            vad_model: None,
            ..Default::default()
        };
        let mic_fmt = StreamFormat::new(48_000, 1).unwrap();
        let sys_fmt = StreamFormat::new(48_000, 2).unwrap();
        let mut p =
            OfflineProcessor::new(cfg, &[(Source::Mic, mic_fmt), (Source::System, sys_fmt)])
                .unwrap();
        let t0 = 1_000_000_000_000u64;
        let sys_rate = R * (1.0 + sys_ppm * 1e-6);
        let off = sys_offset_ms / 1e3;
        let (mut mic_c, mut sys_c, mut echo_on) = (Vec::new(), Vec::new(), false);
        let mut collect = |a: Vec<ProcessedChunk>, e: Vec<RecorderEvent>| {
            for c in a {
                match c.source {
                    Source::Mic => mic_c.push(c),
                    Source::System => sys_c.push(c),
                }
            }
            echo_on |= e
                .iter()
                .any(|e| matches!(e, RecorderEvent::EchoCanceller { active: true, .. }));
        };
        let (mut mic, mut sys) = (vec![0.0f32; 480], vec![0.0f32; 960]);
        let blocks_total = (seconds * 100.0) as u64;
        for i in 0..blocks_total {
            for (k, m) in mic.iter_mut().enumerate() {
                *m = 0.5 * far.at((i * 480 + k as u64) as f64 / R - 0.030);
            }
            let mt = t0 + (i as f64 * 0.01 * 1e9) as u64;
            for k in 0..480 {
                let v = far.at(off + (i * 480 + k as u64) as f64 / sys_rate);
                sys[2 * k] = v;
                sys[2 * k + 1] = v;
            }
            let st = t0 + ((off + (i * 480) as f64 / sys_rate) * 1e9) as u64;
            let meta = |first_frame: u64, host: u64| BlockMeta {
                first_frame,
                frames: 480,
                host_time: HostTime(host),
                flags: BlockFlags::empty(),
            };
            // Deliver in host-time order.
            let blocks = [
                (Source::Mic, meta(i * 480, mt), &mic),
                (Source::System, meta(i * 480, st), &sys),
            ];
            let mut order: Vec<usize> = if mt <= st { vec![0, 1] } else { vec![1, 0] };
            if i + sys_stops_early >= blocks_total {
                order.retain(|&o| o == 0);
            }
            for o in order {
                let (src, m, data) = &blocks[o];
                p.push(*src, *m, data);
            }
            let (a, e) = p.drain();
            collect(a, e);
        }
        let (a, e, stats) = p.finish();
        collect(a, e);
        Outcome {
            mic: mic_c,
            sys: sys_c,
            echo_on,
            stats,
            replay: None,
        }
    }

    /// At a real stop the two streams end at slightly different times. The mic tail after the
    /// reference's end is processed with silence and counted separately, not as timeouts or
    /// resyncs (it used to be counted as both, once per frame).
    #[test]
    fn reference_ending_first_is_not_a_timeout() {
        let o = run_offline(5.0, 0.0, 0.0, 7);
        continuous(&o.mic);
        assert_eq!(o.stats.mic_frames, 500, "{:?}", o.stats);
        assert_eq!(o.stats.reference_resyncs, 0, "{:?}", o.stats);
        assert_eq!(o.stats.reference_timeouts, 0, "{:?}", o.stats);
        assert_eq!(o.stats.reference_ended_frames, 7, "{:?}", o.stats);
    }

    #[test]
    fn processing_follows_clock_drift() {
        let o = run_offline(30.0, 20.0, 200.0, 0);
        continuous(&o.mic);
        assert!(o.echo_on);
        assert_eq!(o.stats.reference_resyncs, 0, "{:?}", o.stats);
        assert_eq!(o.stats.reference_timeouts, 0, "{:?}", o.stats);
        // The read loop must have locked onto the 200 ppm faster reference clock.
        assert!(
            (o.stats.reference_ppm - 200.0).abs() < 20.0,
            "{:?}",
            o.stats
        );
        let all: Vec<f32> = o
            .mic
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect();
        let sys: Vec<f32> = o
            .sys
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect();
        let n = all.len().min(sys.len());
        let erle = power_db(&sys[n - 6 * 16_000..n]) - 6.0 - power_db(&all[n - 6 * 16_000..n]);
        println!(
            "drift 200 ppm: ERLE ~{erle:.1} dB at the end, stats {:?}",
            o.stats
        );
        assert!(erle > 30.0, "ERLE {erle:.1} dB after 30 s of drift");
    }
}
