//! Speech and overlap detection with pyannote segmentation-3.0 (MIT, © CNRS), ONNX export.
//!
//! Model protocol (pyannote's specification, recorded in the export script): 10 s windows of
//! 16 kHz audio, input [N, 1, 160000]; output [N, 589, 7] scores per frame, frames
//! every 270 samples (16.875 ms) with a 991-sample receptive field. The 7 classes are a
//! powerset over 3 local speakers: none, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}.
//!
//! Windows slide over the audio; each frame's number of active speakers is averaged over
//! the windows covering it and rounded, as pyannote does. Frames with one or more speakers
//! are speech; frames with two are overlap.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::{DiarError, Result};

pub const WINDOW: usize = 160_000;
pub const FRAME_STEP: usize = 270;
pub const RECEPTIVE_FIELD: usize = 991;
/// Speakers active in each powerset class.
const ACTIVE: [u8; 7] = [0, 1, 1, 1, 2, 2, 2];
const BATCH: usize = 8;

fn engine_err(e: impl std::fmt::Display) -> DiarError {
    DiarError::Engine(e.to_string())
}

/// Number of active speakers per model frame over a whole recording.
#[derive(Debug, Clone)]
pub struct SpeakerCount {
    pub counts: Vec<u8>,
}

impl SpeakerCount {
    /// Centre of frame `i`, in samples.
    pub fn centre(i: usize) -> u64 {
        (i * FRAME_STEP + RECEPTIVE_FIELD / 2) as u64
    }

    /// The count at sample position `s` (nearest frame).
    pub fn at(&self, s: u64) -> u8 {
        let i = (s.saturating_sub((RECEPTIVE_FIELD / 2) as u64) as f64 / FRAME_STEP as f64).round()
            as usize;
        self.counts.get(i).copied().unwrap_or(0)
    }

    /// Speech regions (sample ranges) where one or more speakers are active, with gaps
    /// shorter than `min_gap_s` closed and regions shorter than `min_on_s` dropped.
    pub fn speech_regions(&self, min_gap_s: f32, min_on_s: f32) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();
        let mut start: Option<usize> = None;
        for (i, &c) in self.counts.iter().chain(std::iter::once(&0)).enumerate() {
            match (c > 0, start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    out.push((
                        Self::centre(s) - FRAME_STEP as u64 / 2,
                        Self::centre(i) - FRAME_STEP as u64 / 2,
                    ));
                    start = None;
                }
                _ => {}
            }
        }
        let gap = (min_gap_s * 16_000.0) as u64;
        let min_on = (min_on_s * 16_000.0) as u64;
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (a, b) in out {
            match merged.last_mut() {
                Some(last) if a.saturating_sub(last.1) < gap => last.1 = b,
                _ => merged.push((a, b)),
            }
        }
        merged.retain(|(a, b)| b - a >= min_on);
        merged
    }
}

pub struct SegmentationModel {
    session: Mutex<Session>,
    /// Input and output names (exports differ: `x`/`y` or `input_values`/`logits`).
    input: String,
    output: String,
}

impl SegmentationModel {
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
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(b)?
            .commit_from_file(model)
            .map_err(engine_err)?;
        let input = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| engine_err("model has no input"))?;
        let output = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| engine_err("model has no output"))?;
        Ok(Self {
            session: Mutex::new(session),
            input,
            output,
        })
    }

    /// Speaker count per frame for 16 kHz mono `pcm`, with windows every `shift_s` seconds.
    pub fn count(&self, pcm: &[f32], shift_s: f32) -> Result<SpeakerCount> {
        let shift = ((shift_s * 16_000.0) as usize).max(FRAME_STEP);
        let mut starts: Vec<usize> = (0..pcm.len().saturating_sub(WINDOW) + 1)
            .step_by(shift)
            .collect();
        if pcm.len() > WINDOW && starts.last() != Some(&(pcm.len() - WINDOW)) {
            starts.push(pcm.len() - WINDOW);
        }
        let total_frames = (pcm.len().max(WINDOW) - RECEPTIVE_FIELD) / FRAME_STEP + 1;
        let mut sum = vec![0f32; total_frames];
        let mut covered = vec![0f32; total_frames];
        let mut session = self.session.lock().expect("not poisoned");
        for batch in starts.chunks(BATCH) {
            let mut input = Array3::<f32>::zeros((batch.len(), 1, WINDOW));
            for (k, &s) in batch.iter().enumerate() {
                let end = (s + WINDOW).min(pcm.len());
                input
                    .slice_mut(ndarray::s![k, 0, ..end - s])
                    .assign(&ndarray::ArrayView1::from(&pcm[s..end]));
            }
            let out = session
                .run(ort::inputs![
                    self.input.as_str() => TensorRef::from_array_view(input.view()).map_err(engine_err)?,
                ])
                .map_err(engine_err)?;
            let (shape, y) = out[self.output.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(engine_err)?;
            let (frames, classes) = (shape[1] as usize, shape[2] as usize);
            for (k, &s) in batch.iter().enumerate() {
                let first = (s as f64 / FRAME_STEP as f64).round() as usize;
                for f in 0..frames {
                    let row = &y[(k * frames + f) * classes..(k * frames + f + 1) * classes];
                    let best = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map_or(0, |(i, _)| i);
                    if let Some(slot) = sum.get_mut(first + f) {
                        *slot += f32::from(ACTIVE[best.min(6)]);
                        covered[first + f] += 1.0;
                    }
                }
            }
        }
        let counts = sum
            .iter()
            .zip(&covered)
            .map(|(s, c)| if *c > 0.0 { (s / c + 0.5) as u8 } else { 0 })
            .collect();
        Ok(SpeakerCount { counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_from_counts() {
        let mut counts = vec![0u8; 200];
        counts[10..60].iter_mut().for_each(|c| *c = 1);
        counts[62..120].iter_mut().for_each(|c| *c = 2);
        counts[150..152].iter_mut().for_each(|c| *c = 1);
        let sc = SpeakerCount { counts };
        let r = sc.speech_regions(0.1, 0.2);
        // The 2-frame gap (34 ms) is closed; the 34 ms burst at 150 is dropped.
        assert_eq!(r.len(), 1, "{r:?}");
        assert_eq!(sc.at(SpeakerCount::centre(80)), 2);
        assert_eq!(sc.at(SpeakerCount::centre(5)), 0);
    }
}
