//! A recording session: capture, processing, transcription and diarization, delivering
//! [`Event`]s until it is stopped.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sd_capture::DeviceSelector;
use sd_core::Source;
use sd_engine::{AsrSetup, Recorder, RecorderConfig};
use serde::Serialize;

use crate::config::Config;
use crate::engines::{Engines, IndicOptions, WhisperOptions};
use crate::events::{self, Event, SourceHealth};
use crate::{Error, Result};

/// A running session. Dropping it stops it (without the summary [`Session::stop`] returns).
///
/// [`Session::stop`] finishes the transcription of speech still queued, which produces more
/// events; keep calling [`Session::next_event`] after it until it returns `None`.
pub struct Session {
    recorder: Option<Recorder>,
    events: Events,
    journal_dir: PathBuf,
    started: Instant,
}

/// A session's event stream, usable from another thread than the session (clone it from
/// [`Session::events`]): one thread can wait for events while another stops the session.
#[derive(Clone)]
pub struct Events {
    rx: crossbeam_channel::Receiver<sd_engine::RecorderEvent>,
    levels: bool,
    health: bool,
}

impl Events {
    /// The next event, waiting up to `timeout`; `None` if none arrived in time. Once the
    /// session has stopped and every event was read, `None` at once.
    pub fn next(&self, timeout: Duration) -> Option<Event> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let e = self.rx.recv_timeout(left).ok()?;
            if let Some(e) = events::convert(e, self.levels, self.health) {
                return Some(e);
            }
        }
    }
}

/// What [`Session::stop`] reports.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub journal_dir: PathBuf,
    /// Seconds from start to stop.
    pub duration: f64,
    pub sources: Vec<SourceHealth>,
    /// Frames written to the journal (all sources).
    pub journaled_frames: u64,
    pub journal_errors: u64,
    /// Nothing was dropped between capture and disk.
    pub complete: bool,
}

fn selector(s: &str, what: &str) -> Result<Option<DeviceSelector>> {
    Ok(match s {
        "none" => None,
        "default" => Some(DeviceSelector::Default),
        "" => return Err(Error::Config(format!("{what}: empty device"))),
        id => Some(DeviceSelector::Id(id.to_string())),
    })
}

fn parse_source(s: &str) -> Result<Source> {
    match s {
        "mic" => Ok(Source::Mic),
        "system" => Ok(Source::System),
        other => Err(Error::Config(format!(
            "source must be \"mic\" or \"system\", not {other:?}"
        ))),
    }
}

/// A new journal directory under the system's temporary directory.
fn default_journal_dir() -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("sonicdiar-{}-{stamp}", std::process::id()))
}

fn asr_setup(cfg: &Config) -> Result<Option<AsrSetup>> {
    let Some(t) = &cfg.transcription else {
        return Ok(None);
    };
    if !cfg.processing {
        return Err(Error::Config(
            "transcription needs processing (voice activity detection)".into(),
        ));
    }
    let whisper = t.whisper_model.as_ref().map(|p| cfg.model_path(p));
    let indic = t
        .indic
        .then(|| cfg.models_dir.join("indicconformer").join("assets"));
    for p in whisper.iter().chain(indic.iter()) {
        if !p.exists() {
            return Err(Error::Config(format!(
                "{} not found; fetch the models first (group \"asr\" or \"indic\")",
                p.display()
            )));
        }
    }
    let options = IndicOptions {
        lm: t.indic_lm.as_ref().map(|p| cfg.model_path(p)),
        numbers_as_digits: t.numbers_as_digits,
        ..Default::default()
    };
    let engines = Engines::load(
        whisper.as_deref(),
        indic.as_deref(),
        &options,
        &WhisperOptions {
            gpu: t.gpu,
            gpu_device: t.gpu_device,
            ..Default::default() // greedy decoding keeps live latency low
        },
        t.threads,
    )?;
    let language = t
        .language
        .parse()
        .map_err(|e| Error::Config(format!("language: {e}")))?;
    let speakers = match &cfg.diarization {
        None => None,
        Some(d) => {
            let embed = cfg
                .models_dir
                .join("wespeaker_en_voxceleb_resnet34_LM.onnx");
            if !embed.exists() {
                return Err(Error::Config(format!(
                    "{} not found; fetch the models first (group \"diar\")",
                    embed.display()
                )));
            }
            let seg = cfg.models_dir.join("pyannote_segmentation_3_0.onnx");
            let seg = (d.overlap && seg.exists()).then_some(seg);
            Some(
                sd_engine::diarization::speaker_factory(
                    &embed,
                    seg.as_deref(),
                    2,
                    sd_diar::online::OnlineConfig::default(),
                )
                .map_err(|e| Error::Engine(format!("diarization: {e}")))?,
            )
        }
    };
    Ok(Some(AsrSetup {
        config: sd_asr::runner::AsrConfig {
            language,
            allowed: engines.supported.clone(),
            partials: t.partials,
            ..Default::default()
        },
        router: engines.router,
        detector: Some(engines.detector),
        speakers,
    }))
}

impl Session {
    /// Starts recording (and processing, transcription, diarization, as configured).
    pub fn start(cfg: Config) -> Result<Self> {
        sd_vad::set_runtime_library(cfg.models_dir.join(sd_vad::RUNTIME_LIBRARY));
        let journal_dir = cfg.journal_dir.clone().unwrap_or_else(default_journal_dir);
        if journal_dir.exists() {
            return Err(Error::Config(format!(
                "journal directory {} already exists",
                journal_dir.display()
            )));
        }
        if cfg.diarization.is_some() && cfg.transcription.is_none() {
            return Err(Error::Config(
                "diarization labels transcripts: it needs transcription".into(),
            ));
        }
        let mut rc = RecorderConfig::new(&journal_dir);
        rc.mic = selector(&cfg.mic, "mic")?;
        rc.system = selector(&cfg.system, "system")?;
        if rc.mic.is_none() && rc.system.is_none() {
            return Err(Error::Config("both mic and system are \"none\"".into()));
        }
        if cfg.processing {
            let aec = match cfg.echo_cancellation.as_str() {
                "auto" => sd_dsp::mic::AecMode::Auto,
                "on" => sd_dsp::mic::AecMode::On,
                "off" => sd_dsp::mic::AecMode::Off,
                other => {
                    return Err(Error::Config(format!(
                        "echo_cancellation must be \"auto\", \"on\" or \"off\", not {other:?}"
                    )));
                }
            };
            let vad = cfg.models_dir.join("silero_vad_16k_op15.onnx");
            if !vad.exists() {
                return Err(Error::Config(format!(
                    "{} not found; fetch the models first (group \"core\")",
                    vad.display()
                )));
            }
            rc.processing = Some(sd_engine::ProcessingConfig {
                mic: sd_dsp::mic::MicConfig {
                    aec,
                    gain_control: cfg.gain_control,
                    noise_suppression: None,
                    ..Default::default()
                },
                vad_model: Some(vad),
                ..Default::default()
            });
        }
        rc.asr = asr_setup(&cfg)?;
        let recorder = Recorder::start(rc).map_err(|e| Error::Engine(e.to_string()))?;
        Ok(Self {
            events: Events {
                rx: recorder.events().clone(),
                levels: cfg.levels,
                health: cfg.health,
            },
            recorder: Some(recorder),
            journal_dir,
            started: Instant::now(),
        })
    }

    /// The next event, waiting up to `timeout`; `None` if none arrived in time. After
    /// [`Session::stop`], it returns the remaining events, then `None` at once.
    pub fn next_event(&self, timeout: Duration) -> Option<Event> {
        self.events.next(timeout)
    }

    /// A handle on the event stream for another thread.
    pub fn events(&self) -> Events {
        self.events.clone()
    }

    /// Switches `source` (`"mic"` or `"system"`) to a device id or `"default"`, without
    /// interrupting the timeline.
    pub fn set_device(&self, source: &str, device: &str) -> Result<()> {
        let source = parse_source(source)?;
        let Some(sel) = selector(device, "device")? else {
            return Err(Error::Config(
                "a running source cannot be switched to \"none\"".into(),
            ));
        };
        if let Some(r) = &self.recorder {
            r.set_device(source, sel);
        }
        Ok(())
    }

    /// Pauses: capture keeps running, but its audio is replaced by silence, so nothing is
    /// recorded or transcribed until [`Session::resume`]. Speech in progress ends (and is
    /// transcribed); event times keep counting from the start, pauses included.
    pub fn pause(&self) -> Result<()> {
        self.recorder()?.set_paused(true);
        Ok(())
    }

    /// Resumes after [`Session::pause`].
    pub fn resume(&self) -> Result<()> {
        self.recorder()?.set_paused(false);
        Ok(())
    }

    /// Whether the session is paused.
    pub fn is_paused(&self) -> bool {
        self.recorder.as_ref().is_some_and(|r| r.is_paused())
    }

    fn recorder(&self) -> Result<&Recorder> {
        self.recorder
            .as_ref()
            .ok_or_else(|| Error::Engine("the session has stopped".into()))
    }

    /// The journal directory (the crash-safe recording; `sdctl journal export` turns it into
    /// WAV files).
    pub fn journal_dir(&self) -> &Path {
        &self.journal_dir
    }

    /// Stops capture, finishes the transcription of queued speech and closes the journal. The
    /// events produced meanwhile stay available from [`Session::next_event`]. Stopping can take
    /// as long as the recognizer needs for its backlog (seconds on a GPU; on a slow CPU, up to
    /// the length of the speech still queued).
    pub fn stop(&mut self) -> Result<Summary> {
        let rec = self
            .recorder
            .take()
            .ok_or_else(|| Error::Engine("already stopped".into()))?;
        let s = rec.stop().map_err(|e| Error::Engine(e.to_string()))?;
        let sources = events::health(&s.health);
        let complete = sources.iter().all(|x| x.dropped_frames == 0);
        Ok(Summary {
            journal_dir: s.journal_dir,
            duration: self.started.elapsed().as_secs_f64(),
            sources,
            journaled_frames: s.health.journaled_frames,
            journal_errors: s.health.journal_errors,
            complete,
        })
    }
}
