//! # SonicDiar
//!
//! Meeting audio capture as a library: microphone and system audio (what the computer plays)
//! on Windows, macOS and Linux, with echo cancellation, voice activity detection, live
//! transcription in 15 languages and speaker labels.
//!
//! ```no_run
//! use std::time::Duration;
//! use sonicdiar::{Config, Event, Session, models};
//!
//! // Once: the voice-activity model and this platform's ONNX Runtime (about 50 MB).
//! models::fetch("models".as_ref(), "core", None, &mut |_| {})?;
//! let mut cfg = Config::default();
//! cfg.transcription = None; // or Some(Default::default()) after fetching group "asr"
//! let mut session = Session::start(cfg)?;
//! let until = std::time::Instant::now() + Duration::from_secs(10);
//! while std::time::Instant::now() < until {
//!     if let Some(event) = session.next_event(Duration::from_millis(200)) {
//!         println!("{}", serde_json::to_string(&event)?);
//!     }
//! }
//! let summary = session.stop()?;
//! // Transcripts of speech still queued at the stop:
//! while let Some(event) = session.next_event(Duration::ZERO) {
//!     println!("{}", serde_json::to_string(&event)?);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The same API is available from C (`sonicdiar.h`), Python and Node.js; every binding
//! delivers the same events, as JSON objects (see [`Event`]).

pub mod config;
pub mod engines;
pub mod events;
pub mod models;
mod session;

use serde::Serialize;

pub use config::{Config, Diarization, Transcription};
pub use events::Event;
pub use session::{Events, Session, Summary};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("{0}")]
    Engine(String),
    #[error("models: {0}")]
    Models(String),
    #[error("devices: {0}")]
    Devices(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The library version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A language transcription can be set to ([`Transcription::language`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LanguageInfo {
    pub code: &'static str,
    pub name: &'static str,
    /// Whisper transcribes it.
    pub whisper: bool,
    /// IndicConformer transcribes it (`indic` on).
    pub indic: bool,
}

/// The languages, in the order a picker should list them (`"auto"` is not one of them).
pub fn languages() -> Vec<LanguageInfo> {
    sd_asr::Language::ALL
        .iter()
        .map(|&l| LanguageInfo {
            code: l.code(),
            name: l.name(),
            whisper: l.whisper_code().is_some(),
            indic: l.indic_code().is_some(),
        })
        .collect()
}

/// An audio device.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Device {
    /// Stable id, usable in [`Config::mic`] / [`Config::system`].
    pub id: String,
    pub name: String,
    /// `"input"` (microphones) or `"output"` (what system audio is captured from).
    pub kind: &'static str,
    pub is_default: bool,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
}

/// Lists `"input"` or `"output"` devices.
pub fn list_devices(kind: &str) -> Result<Vec<Device>> {
    let k = match kind {
        "input" => sd_capture::DeviceKind::Input,
        "output" => sd_capture::DeviceKind::Output,
        other => {
            return Err(Error::Config(format!(
                "kind must be \"input\" or \"output\", not {other:?}"
            )));
        }
    };
    let devices = sd_capture::list_devices(k).map_err(|e| Error::Devices(e.to_string()))?;
    Ok(devices
        .into_iter()
        .map(|d| Device {
            id: d.id,
            name: d.name,
            kind: if k == sd_capture::DeviceKind::Input {
                "input"
            } else {
                "output"
            },
            is_default: d.is_default,
            sample_rate: d.mix_format.map(|f| f.sample_rate),
            channels: d.mix_format.map(|f| f.channels),
        })
        .collect())
}
