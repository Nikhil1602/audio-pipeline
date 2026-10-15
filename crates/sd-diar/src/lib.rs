//! Speaker diarization ("who spoke when") for sonicdiar.
//!
//! * [`fbank`]: Kaldi-compatible 80-band log-mel filterbanks (the WeSpeaker front end).
//! * [`embed`]: speaker embeddings from WeSpeaker ONNX models (ResNet34, CAM++), L2-normalised.
//! * [`cluster`]: average-linkage agglomerative clustering, cut at a distance or at K.
//! * [`pipeline`]: offline diarization of speech regions (windows -> embeddings -> clusters).
//! * [`online`]: live per-source diarization with periodic re-clustering and stable ids.
//! * [`segment`]: speech and overlap detection (pyannote segmentation-3.0).
//! * [`der`]: diarization error rate (missed speech, false alarm, speaker confusion).
//!
//! The baseline:
//! WeSpeaker ResNet34 embeddings, Euclidean distance between unit vectors, average linkage,
//! cut around 1.0 (cosine similarity about 0.5).

pub mod cluster;
pub mod der;
pub mod embed;
pub mod fbank;
pub mod online;
pub mod pipeline;
pub mod segment;

#[derive(Debug, thiserror::Error)]
pub enum DiarError {
    #[error("model file not found: {0} (run `sdctl models fetch --group diar`)")]
    MissingModel(String),
    #[error("engine error: {0}")]
    Engine(String),
}

pub type Result<T> = std::result::Result<T, DiarError>;

pub const SAMPLE_RATE: u32 = 16_000;
