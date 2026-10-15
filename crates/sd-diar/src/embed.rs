//! Speaker embeddings from WeSpeaker ONNX models: input `feats` [B, T, 80] (mean-normalised
//! Kaldi fbank), output `embs` [B, D] (D = 256 for ResNet34, 512 for CAM++), L2-normalised
//! here.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::fbank::{DIM, Fbank};
use crate::{DiarError, Result};

fn engine_err(e: impl std::fmt::Display) -> DiarError {
    DiarError::Engine(e.to_string())
}

pub struct Embedder {
    session: Mutex<Session>,
    fbank: Fbank,
}

impl Embedder {
    pub fn new(model: &Path, threads: usize) -> Result<Self> {
        if !model.is_file() {
            return Err(DiarError::MissingModel(model.display().to_string()));
        }
        let env = sd_vad::runtime().map_err(engine_err)?;
        let b = |e: ort::Error<ort::session::builder::SessionBuilder>| engine_err(e);
        let session = Session::builder(env)
            .map_err(engine_err)?
            .with_intra_threads(threads.max(1))
            .map_err(b)?
            .with_inter_threads(1)
            .map_err(b)?
            .with_memory_pattern(false)
            .map_err(b)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(b)?
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(b)?
            .commit_from_file(model)
            .map_err(engine_err)?;
        Ok(Self {
            session: Mutex::new(session),
            fbank: Fbank::new(),
        })
    }

    /// The L2-normalised embedding of 16 kHz mono audio (at least 25 ms); None if too short.
    pub fn embed(&self, pcm: &[f32]) -> Result<Option<Vec<f32>>> {
        let (feats, frames) = self.fbank.compute(pcm);
        if frames == 0 {
            return Ok(None);
        }
        let input = Array3::from_shape_vec((1, frames, DIM), feats).map_err(engine_err)?;
        let mut s = self.session.lock().expect("not poisoned");
        let out = s
            .run(ort::inputs![
                "feats" => TensorRef::from_array_view(input.view()).map_err(engine_err)?,
            ])
            .map_err(engine_err)?;
        let (_, e) = out["embs"]
            .try_extract_tensor::<f32>()
            .map_err(engine_err)?;
        let mut v = e.to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
        Ok(Some(v))
    }
}

/// Euclidean distance between two vectors (for unit vectors, sqrt(2 − 2·cos)).
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}
