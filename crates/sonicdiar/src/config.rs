//! Session configuration. Every field has a default, so `{}` (JSON) or
//! `Config::default()` is a valid configuration: mic and system audio from the default devices,
//! processing on, no transcription.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What to record and how to process it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Models and the ONNX Runtime library (see [`crate::models::fetch`]). Model paths below are
    /// relative to it unless absolute.
    pub models_dir: PathBuf,
    /// Where the crash-safe audio journal is written (a new directory; it must not exist).
    /// Default: a new directory under the system's temporary directory.
    pub journal_dir: Option<PathBuf>,
    /// Microphone: `"default"` (follows the system default), `"none"`, or a device id from
    /// [`crate::list_devices`].
    pub mic: String,
    /// System audio (what the computer plays): `"default"`, `"none"`, or an output device id.
    pub system: String,
    /// Echo cancellation, gain control and voice activity detection; needed for transcription.
    pub processing: bool,
    /// Mic echo cancellation: `"auto"` (on when the speakers' sound reaches the mic), `"on"`,
    /// `"off"`.
    pub echo_cancellation: String,
    /// Automatic gain control on the microphone.
    pub gain_control: bool,
    /// Speech recognition; `None` (JSON `null` or absent): off.
    pub transcription: Option<Transcription>,
    /// Who spoke when; needs transcription. `None`: off.
    pub diarization: Option<Diarization>,
    /// Emit [`crate::Event::Level`] about ten times a second per source.
    pub levels: bool,
    /// Emit [`crate::Event::Health`] once a second.
    pub health: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models"),
            journal_dir: None,
            mic: "default".into(),
            system: "default".into(),
            processing: true,
            echo_cancellation: "auto".into(),
            gain_control: false,
            transcription: None,
            diarization: None,
            levels: false,
            health: false,
        }
    }
}

/// Speech recognition settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Transcription {
    /// `"auto"` or a language: `en`, `zh`, `es`, `fr`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `pa`,
    /// `ta`, `te`, `or`, `ur` (or the language's English name).
    pub language: String,
    /// Whisper ggml model file. `None`: Whisper off (then `indic` must be on).
    pub whisper_model: Option<PathBuf>,
    /// IndicConformer for the Indian languages, from `<models_dir>/indicconformer/assets`
    /// (fetch group `indic`).
    pub indic: bool,
    /// Directory of IndicConformer word language models (optional, improves accuracy).
    pub indic_lm: Option<PathBuf>,
    /// Emit partial results while someone is speaking.
    pub partials: bool,
    /// Write spoken numbers as digits where a verified number table exists (Indian languages).
    pub numbers_as_digits: bool,
    /// Run Whisper on the GPU (builds with GPU support only).
    pub gpu: bool,
    pub gpu_device: i32,
    /// CPU threads per engine; `None`: the engine's default.
    pub threads: Option<usize>,
}

impl Default for Transcription {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            whisper_model: Some(PathBuf::from("ggml-large-v3-turbo-q5_0.bin")),
            indic: false,
            indic_lm: None,
            partials: true,
            numbers_as_digits: true,
            gpu: false,
            gpu_device: 0,
            threads: None,
        }
    }
}

/// Speaker diarization settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Diarization {
    /// Also mark where a second speaker talks at the same time (needs the segmentation model).
    pub overlap: bool,
}

impl Default for Diarization {
    fn default() -> Self {
        Self { overlap: true }
    }
}

impl Config {
    /// `path` inside the models directory unless it is absolute.
    pub fn model_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.models_dir.join(path)
        }
    }

    /// Parses a configuration from JSON (unknown fields are an error, so typos are caught).
    pub fn from_json(json: &str) -> crate::Result<Self> {
        serde_json::from_str(json).map_err(|e| crate::Error::Config(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_is_the_default() {
        assert_eq!(Config::from_json("{}").unwrap(), Config::default());
    }

    #[test]
    fn partial_sections_take_defaults_and_typos_are_errors() {
        let c = Config::from_json(r#"{"system": "none", "transcription": {"language": "hi"}}"#)
            .unwrap();
        assert_eq!(c.system, "none");
        let t = c.transcription.unwrap();
        assert_eq!(t.language, "hi");
        assert!(t.whisper_model.is_some());
        assert!(Config::from_json(r#"{"sytem": "none"}"#).is_err());
    }
}
