//! Echo cancellation at 16 kHz with access to AEC3's *linear* output.
//!
//! AEC3 removes echo in two stages: an adaptive linear filter that predicts and subtracts the
//! echo, then a nonlinear suppressor that attenuates whatever echo is left. The suppressor is
//! what makes single-talk echo inaudible, but during double talk it also attenuates and
//! distorts the local talker (measured in M2 on real speech: near-end −1.8…−3.1 dB and
//! SI-SDR worse than doing nothing; the original WebRTC C++ was worse still). Speech
//! recognition prefers an intact near-end with a little residual echo over a ducked one.
//!
//! [`Aec16k`] therefore returns both signals for every 10 ms frame:
//! * `full`: linear filter + suppressor (for listening, and for single-talk echo removal),
//! * `linear`: linear filter only (for speech recognition).
//!
//! It runs AEC3 at 16 kHz, where it uses a single band, so no band-splitting filter bank is
//! needed and CPU cost is lower; the mic and reference are resampled from the capture rate.
//! The framing mirrors sonora's `EchoCanceller3` wrapper (80-sample sub-frames ↔ 64-sample
//! blocks).

use sonora::high_pass_filter::HighPassFilter;
use sonora_aec3::block::Block;
use sonora_aec3::block_framer::BlockFramer;
use sonora_aec3::block_processor::BlockProcessor;
use sonora_aec3::config::EchoCanceller3Config;
use sonora_aec3::frame_blocker::FrameBlocker;
use sonora_common_audio::push_sinc_resampler::PushSincResampler;

use crate::{DspError, Result};

const RATE: usize = 16_000;
const FRAME: usize = 160;
const SUB: usize = 80;
/// WebRTC's internal float format is "FloatS16": floats in the int16 range. AEC3's activity and
/// saturation thresholds assume it, so audio in [-1, 1] is scaled up on the way in and down on
/// the way out (the `AudioProcessing` wrapper does the same internally).
const S16: f32 = 32768.0;

/// Latency of the `linear` output at 16 kHz, in samples (one 64-sample block from framing),
/// relative to the resampled input. Measured; locked by `aec16k_latencies` test.
pub const LINEAR_LATENCY: usize = 64;
/// Latency of the `full` output at 16 kHz, in samples: the suppressor adds one more block.
pub const FULL_LATENCY: usize = 128;

/// Combines the two outputs for speech recognition, frame by frame.
///
/// Where AEC3's suppressor attenuates heavily, the frame is (residual) echo and the full output
/// is used; otherwise the near-end is active and the linear output is used, so the local
/// talker is not ducked. The decision is a smoothed crossfade so switching never clicks.
/// Inputs must already be latency-aligned (full delayed by `FULL_LATENCY - LINEAR_LATENCY`
/// relative to linear is handled internally by [`AsrMixer::push`]).
pub struct AsrMixer {
    /// Suppression (linear power / full power) above which a frame counts as echo, in dB.
    pub echo_threshold_db: f32,
    lin_delay: Vec<f32>,
    weight: f32,
}

impl Default for AsrMixer {
    fn default() -> Self {
        Self {
            echo_threshold_db: 10.0,
            lin_delay: vec![0.0; FULL_LATENCY - LINEAR_LATENCY],
            weight: 0.0,
        }
    }
}

impl AsrMixer {
    /// Mixes one 160-sample frame. `linear` and `full` are this frame's raw [`Aec16k`]
    /// outputs; the result is aligned with `full` (latency [`FULL_LATENCY`]).
    pub fn push(&mut self, linear: &[f32], full: &[f32], out: &mut [f32]) {
        // Delay linear by one block so both streams line up.
        let d = self.lin_delay.len();
        let mut lin = Vec::with_capacity(linear.len());
        lin.extend_from_slice(&self.lin_delay);
        lin.extend_from_slice(&linear[..linear.len() - d]);
        self.lin_delay.copy_from_slice(&linear[linear.len() - d..]);

        let p = |x: &[f32]| x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() + 1e-12;
        let suppression_db = 10.0 * (p(&lin) / p(full)).log10() as f32;
        let target = if suppression_db > self.echo_threshold_db {
            1.0f32
        } else {
            0.0
        };
        // Ramp the full-output weight within the frame (attack/release ~10 ms).
        let start = self.weight;
        for (i, o) in out.iter_mut().enumerate() {
            let w = start + (target - start) * (i + 1) as f32 / full.len() as f32;
            *o = w * full[i] + (1.0 - w) * lin[i];
        }
        self.weight = target;
    }
}

/// Metrics reported by the AEC3 block processor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aec16kStats {
    pub erl_db: f64,
    pub erle_db: f64,
    pub delay_ms: i32,
}

pub struct Aec16k {
    input_rate: usize,
    render_channels: usize,
    export_linear: bool,
    bp: BlockProcessor,
    render_blocker: FrameBlocker,
    capture_blocker: FrameBlocker,
    out_framer: BlockFramer,
    lin_framer: BlockFramer,
    render_block: Block,
    capture_block: Block,
    linear_block: Block,
    mic_rs: PushSincResampler,
    ref_rs: PushSincResampler,
    hpf: HighPassFilter,
    mic16: Vec<Vec<f32>>,
    ref_mono: Vec<f32>,
    ref16: Vec<f32>,
    sub_out: Vec<Vec<Vec<f32>>>,
    sub_lin: Vec<Vec<Vec<f32>>>,
}

impl Aec16k {
    /// `input_rate`: rate of mic and reference (8/16/32/48 kHz); `render_channels`: 1 or 2.
    pub fn new(input_rate: u32, render_channels: u16) -> Result<Self> {
        Self::with_linear_export(input_rate, render_channels, true)
    }

    /// As [`Self::new`]; `export_linear` = false leaves `linear` equal to the (resampled,
    /// high-passed) input. Used to check that exporting does not change the full output.
    pub fn with_linear_export(
        input_rate: u32,
        render_channels: u16,
        export_linear: bool,
    ) -> Result<Self> {
        if ![8_000, 16_000, 32_000, 48_000].contains(&input_rate) {
            return Err(DspError::Config(format!(
                "unsupported input rate {input_rate}"
            )));
        }
        if !(1..=2).contains(&render_channels) {
            return Err(DspError::Config("render must be mono or stereo".into()));
        }
        let mut cfg = EchoCanceller3Config::default();
        cfg.filter.export_linear_aec_output = export_linear;
        let in_frame = input_rate as usize / 100;
        Ok(Self {
            input_rate: input_rate as usize,
            render_channels: render_channels as usize,
            export_linear,
            bp: BlockProcessor::new(&cfg, RATE, 1, 1),
            render_blocker: FrameBlocker::new(1, 1),
            capture_blocker: FrameBlocker::new(1, 1),
            out_framer: BlockFramer::new(1, 1),
            lin_framer: BlockFramer::new(1, 1),
            render_block: Block::new(1, 1),
            capture_block: Block::new(1, 1),
            linear_block: Block::new(1, 1),
            mic_rs: PushSincResampler::new(in_frame, FRAME),
            ref_rs: PushSincResampler::new(in_frame, FRAME),
            hpf: HighPassFilter::new(RATE as i32, 1),
            mic16: vec![vec![0.0; FRAME]],
            ref_mono: vec![0.0; in_frame],
            ref16: vec![0.0; FRAME],
            sub_out: vec![vec![vec![0.0; SUB]]],
            sub_lin: vec![vec![vec![0.0; SUB]]],
        })
    }

    pub fn input_frame_len(&self) -> usize {
        self.input_rate / 100
    }

    /// Processes one 10 ms frame. `mic`: mono at the input rate; `reference`: interleaved at the
    /// input rate. Writes 160 samples (16 kHz) to each of `full` and `linear`.
    pub fn process(
        &mut self,
        mic: &[f32],
        reference: &[f32],
        full: &mut [f32],
        linear: &mut [f32],
    ) -> Result<()> {
        let n = self.input_frame_len();
        if mic.len() != n {
            return Err(DspError::FrameSize {
                got: mic.len(),
                expected: n,
            });
        }
        if reference.len() != n * self.render_channels {
            return Err(DspError::FrameSize {
                got: reference.len(),
                expected: n * self.render_channels,
            });
        }
        if full.len() != FRAME || linear.len() != FRAME {
            return Err(DspError::FrameSize {
                got: full.len().min(linear.len()),
                expected: FRAME,
            });
        }

        // Reference: downmix to mono, resample to 16 kHz, buffer as render blocks.
        let ch = self.render_channels;
        for (i, f) in reference.chunks_exact(ch).enumerate() {
            self.ref_mono[i] = f.iter().sum::<f32>() / ch as f32;
        }
        self.ref_rs.resample(&self.ref_mono, &mut self.ref16);
        self.ref16.iter_mut().for_each(|v| *v *= S16);
        for s in 0..2 {
            let sub = [vec![&self.ref16[s * SUB..(s + 1) * SUB]]];
            self.render_blocker
                .insert_sub_frame_and_extract_block(&sub, &mut self.render_block);
            self.bp.buffer_render(&self.render_block);
        }
        if self.render_blocker.is_block_available() {
            self.render_blocker.extract_block(&mut self.render_block);
            self.bp.buffer_render(&self.render_block);
        }

        // Mic: resample, high-pass (AEC3 expects DC-free input), then two sub-frames.
        self.mic_rs.resample(mic, &mut self.mic16[0]);
        self.mic16[0].iter_mut().for_each(|v| *v *= S16);
        self.hpf.process_channels(&mut self.mic16);
        for s in 0..2 {
            let sub = [vec![&self.mic16[0][s * SUB..(s + 1) * SUB]]];
            self.capture_blocker
                .insert_sub_frame_and_extract_block(&sub, &mut self.capture_block);
            self.process_block();
            self.out_framer
                .insert_block_and_extract_sub_frame(&self.capture_block, &mut self.sub_out);
            self.lin_framer
                .insert_block_and_extract_sub_frame(&self.linear_block, &mut self.sub_lin);
            for (d, v) in full[s * SUB..(s + 1) * SUB]
                .iter_mut()
                .zip(&self.sub_out[0][0])
            {
                *d = v / S16;
            }
            for (d, v) in linear[s * SUB..(s + 1) * SUB]
                .iter_mut()
                .zip(&self.sub_lin[0][0])
            {
                *d = v / S16;
            }
        }
        if self.capture_blocker.is_block_available() {
            self.capture_blocker.extract_block(&mut self.capture_block);
            self.process_block();
            self.out_framer.insert_block(&self.capture_block);
            self.lin_framer.insert_block(&self.linear_block);
        }
        Ok(())
    }

    fn process_block(&mut self) {
        // If AEC3 skips processing (no render yet), the linear output must equal the input.
        self.linear_block
            .view_mut(0, 0)
            .copy_from_slice(self.capture_block.view(0, 0));
        let lin = if self.export_linear {
            Some(&mut self.linear_block)
        } else {
            None
        };
        self.bp
            .process_capture(false, false, lin, &mut self.capture_block);
    }

    pub fn stats(&self) -> Aec16kStats {
        let m = self.bp.get_metrics();
        Aec16kStats {
            erl_db: m.echo_return_loss,
            erle_db: m.echo_return_loss_enhancement,
            delay_ms: m.delay_ms,
        }
    }
}
