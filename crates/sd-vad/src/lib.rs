//! Voice activity detection: Silero VAD (v6.2.3, MIT) on ONNX Runtime, plus a streaming
//! segmenter that turns per-chunk speech probabilities into speech start/end events.
//!
//! Model protocol (verified against the model's declared inputs/outputs): 16 kHz audio in
//! chunks of 512 samples (32 ms), each prefixed by the previous chunk's last 64 samples;
//! a recurrent `state` of shape [2, 1, 128] carried between calls; `sr` = 16000.

use std::path::Path;
use std::sync::OnceLock;

use ndarray::{Array0, Array2, Array3};
use ort::memory::Allocator;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
use ort::value::{Tensor, TensorRef};
use thiserror::Error;

pub const SAMPLE_RATE: u32 = 16_000;
pub const CHUNK: usize = 512;
pub const CONTEXT: usize = 64;

#[derive(Debug, Error)]
pub enum VadError {
    #[error("ONNX Runtime: {0}")]
    Runtime(#[from] ort::Error),
    #[error(
        "ONNX Runtime library not available ({0}); run `sdctl models fetch` or set SD_ORT_DYLIB"
    )]
    RuntimeMissing(String),
    #[error("model file {0} not found; run `sdctl models fetch`")]
    MissingModel(String),
    #[error("unexpected model output: {0}")]
    Output(String),
}

pub type Result<T, E = VadError> = std::result::Result<T, E>;

/// File name of the ONNX Runtime library on this platform.
pub const RUNTIME_LIBRARY: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

static RUNTIME_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

/// Sets where to load the ONNX Runtime library from. Call before the first model is created;
/// later calls are ignored.
pub fn set_runtime_library(path: impl Into<std::path::PathBuf>) {
    let _ = RUNTIME_PATH.set(path.into());
}

/// Candidate locations, in order: [`set_runtime_library`], `SD_ORT_DYLIB`, next to the
/// executable, `models/` under the working directory.
fn runtime_candidates() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = RUNTIME_PATH.get() {
        v.push(p.clone());
    }
    if let Some(p) = std::env::var_os("SD_ORT_DYLIB") {
        v.push(p.into());
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.to_path_buf()))
    {
        v.push(dir.join(RUNTIME_LIBRARY));
    }
    v.push(std::path::Path::new("models").join(RUNTIME_LIBRARY));
    v
}

/// ONNX Runtime allows one environment per process; every model in the engine shares it. The
/// library is loaded on first use; if it is missing, models fail with
/// [`VadError::RuntimeMissing`] while the rest of the engine keeps working.
pub fn runtime() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> =
        OnceLock::new();
    ENV.get_or_init(|| {
        let candidates = runtime_candidates();
        let path = candidates
            .iter()
            .find(|p| p.is_file())
            .ok_or_else(|| format!("{RUNTIME_LIBRARY} not found in {candidates:?}"))?;
        ort::init_from(path)
            .map_err(|e| format!("loading {}: {e}", path.display()))?
            .with_name("SonicDiar")
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| VadError::RuntimeMissing(e.clone()))
}

/// Streaming Silero VAD. Feed 16 kHz mono audio of any length; get one speech probability per
/// 512-sample chunk.
pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    context: [f32; CONTEXT],
    pending: Vec<f32>,
    input: Array2<f32>,
    sr: Array0<i64>,
    run_options: Option<RunOptions<HasSelectedOutputs>>,
}

/// ONNX Runtime session options that affect memory over long runs.
///
/// Measured (M2.5 soak, ORT 1.31, 30 min of audio per run): with ORT allocating the outputs on
/// every run, private memory grew by ~0.75 MB per minute of audio per VAD instance whatever the
/// arena, memory-pattern or graph-optimization setting. Binding the outputs to tensors
/// allocated once kept it flat. Preallocated outputs are therefore the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMemory {
    pub cpu_arena: bool,
    pub memory_pattern: bool,
    /// Graph optimizations (ORT default: all).
    pub optimize: bool,
    /// Bind the outputs to tensors allocated once, instead of letting ORT allocate per run.
    pub preallocate_outputs: bool,
}

impl SileroVad {
    pub fn new(model: impl AsRef<Path>) -> Result<Self> {
        Self::with_memory(
            model,
            SessionMemory {
                cpu_arena: true,
                memory_pattern: true,
                optimize: true,
                preallocate_outputs: true,
            },
        )
    }

    pub fn with_memory(model: impl AsRef<Path>, mem: SessionMemory) -> Result<Self> {
        let model = model.as_ref();
        if !model.exists() {
            return Err(VadError::MissingModel(model.display().to_string()));
        }
        // One thread each: the VAD is tiny and must not compete with ASR for cores.
        let builder_err = |e: ort::Error<ort::session::builder::SessionBuilder>| {
            VadError::Output(format!("session options: {e}"))
        };
        let session = Session::builder(runtime()?)?
            .with_intra_threads(1)
            .map_err(builder_err)?
            .with_inter_threads(1)
            .map_err(builder_err)?
            .with_memory_pattern(mem.memory_pattern)
            .map_err(builder_err)?
            .with_optimization_level(if mem.optimize {
                GraphOptimizationLevel::Level3
            } else {
                GraphOptimizationLevel::Disable
            })
            .map_err(builder_err)?
            .with_execution_providers([ort::ep::CPU::default()
                .with_arena_allocator(mem.cpu_arena)
                .build()])
            .map_err(builder_err)?
            .commit_from_file(model)?;
        let run_options = if mem.preallocate_outputs {
            let alloc = Allocator::default();
            Some(
                RunOptions::new()?.with_outputs(
                    OutputSelector::no_default()
                        .with("output")
                        .with("stateN")
                        .preallocate("output", Tensor::<f32>::new(&alloc, [1usize, 1])?)
                        .preallocate("stateN", Tensor::<f32>::new(&alloc, [2usize, 1, 128])?),
                ),
            )
        } else {
            None
        };
        Ok(Self {
            session,
            state: Array3::zeros((2, 1, 128)),
            context: [0.0; CONTEXT],
            pending: Vec::with_capacity(CHUNK * 2),
            input: Array2::zeros((1, CONTEXT + CHUNK)),
            sr: Array0::from_elem((), SAMPLE_RATE as i64),
            run_options,
        })
    }

    /// Clears the recurrent state (call between unrelated audio streams).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context = [0.0; CONTEXT];
        self.pending.clear();
    }

    /// Appends audio; pushes one probability to `probs` per complete 512-sample chunk.
    pub fn push(&mut self, samples: &[f32], probs: &mut Vec<f32>) -> Result<()> {
        self.pending.extend_from_slice(samples);
        let mut offset = 0;
        while self.pending.len() - offset >= CHUNK {
            let chunk = &self.pending[offset..offset + CHUNK];
            {
                let row = self.input.as_slice_mut().expect("contiguous");
                row[..CONTEXT].copy_from_slice(&self.context);
                row[CONTEXT..].copy_from_slice(chunk);
            }
            self.context.copy_from_slice(&chunk[CHUNK - CONTEXT..]);
            let inputs = ort::inputs![
                // Zero-copy views of our own buffers: no tensor allocation per chunk.
                "input" => TensorRef::from_array_view(self.input.view())?,
                "state" => TensorRef::from_array_view(self.state.view())?,
                "sr" => TensorRef::from_array_view(self.sr.view())?,
            ];
            let outputs = match &self.run_options {
                Some(o) => self.session.run_with_options(inputs, o)?,
                None => self.session.run(inputs)?,
            };
            let (_, p) = outputs["output"].try_extract_tensor::<f32>()?;
            let prob = *p
                .first()
                .ok_or_else(|| VadError::Output("empty output".into()))?;
            let (_, st) = outputs["stateN"].try_extract_tensor::<f32>()?;
            if st.len() != self.state.len() {
                return Err(VadError::Output(format!("state has {} values", st.len())));
            }
            self.state
                .as_slice_mut()
                .expect("contiguous")
                .copy_from_slice(st);
            probs.push(prob);
            offset += CHUNK;
        }
        self.pending.drain(..offset);
        Ok(())
    }
}

/// Segmenter parameters (Silero's reference defaults, in streaming form).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmenterConfig {
    /// Probability at or above which a chunk counts as speech.
    pub threshold: f32,
    /// Probability below which a chunk counts as silence (hysteresis); default threshold − 0.15.
    pub neg_threshold: f32,
    /// Shorter speech bursts are discarded.
    pub min_speech_ms: u32,
    /// Silence must last this long to end a segment.
    pub min_silence_ms: u32,
    /// Padding added before the start and after the end of each segment.
    pub speech_pad_ms: u32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 250,
            min_silence_ms: 300,
            speech_pad_ms: 100,
        }
    }
}

/// A detected speech segment, in 16 kHz sample positions on the stream's timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start: u64,
    pub end: u64,
}

/// Turns per-chunk probabilities into speech segments as they complete.
#[derive(Debug, Clone)]
pub struct Segmenter {
    cfg: SegmenterConfig,
    pos: u64,
    triggered: bool,
    start: u64,
    temp_end: Option<u64>,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self {
            cfg,
            pos: 0,
            triggered: false,
            start: 0,
            temp_end: None,
        }
    }

    fn ms(&self, ms: u32) -> u64 {
        ms as u64 * SAMPLE_RATE as u64 / 1000
    }

    /// True while inside a speech segment.
    pub fn in_speech(&self) -> bool {
        self.triggered
    }

    /// Start (with padding) of the segment currently open, if any.
    pub fn open_start(&self) -> Option<u64> {
        self.triggered
            .then(|| self.start.saturating_sub(self.ms(self.cfg.speech_pad_ms)))
    }

    /// Feeds one chunk probability; returns a segment when one completes.
    pub fn push(&mut self, prob: f32) -> Option<Segment> {
        let chunk_start = self.pos;
        self.pos += CHUNK as u64;
        if prob >= self.cfg.threshold {
            self.temp_end = None;
            if !self.triggered {
                self.triggered = true;
                self.start = chunk_start;
            }
            return None;
        }
        if self.triggered && prob < self.cfg.neg_threshold {
            let end = *self.temp_end.get_or_insert(chunk_start);
            if self.pos - end >= self.ms(self.cfg.min_silence_ms) {
                self.triggered = false;
                self.temp_end = None;
                if end - self.start >= self.ms(self.cfg.min_speech_ms) {
                    return Some(self.padded(self.start, end));
                }
            }
        }
        None
    }

    /// Ends any open segment (e.g. at stream end).
    pub fn flush(&mut self) -> Option<Segment> {
        if !self.triggered {
            return None;
        }
        self.triggered = false;
        let end = self.temp_end.take().unwrap_or(self.pos);
        (end - self.start >= self.ms(self.cfg.min_speech_ms)).then(|| self.padded(self.start, end))
    }

    fn padded(&self, start: u64, end: u64) -> Segment {
        let pad = self.ms(self.cfg.speech_pad_ms);
        Segment {
            start: start.saturating_sub(pad),
            end: end + pad,
        }
    }
}

#[cfg(test)]
mod tests;
