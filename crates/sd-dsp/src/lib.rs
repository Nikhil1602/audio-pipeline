//! Capture-side audio processing: acoustic echo cancellation (AEC3), noise suppression,
//! automatic gain control (AGC2) and high-pass filtering.
//!
//! The DSP backend is [sonora], a pure-Rust port of the WebRTC audio processing module (M145),
//! chosen so the engine builds on every platform without a C++ toolchain. The microphone is the
//! *capture* stream; system audio (the loopback of what is being played) is the *render*
//! stream, i.e. the echo reference.
//!
//! [`Processor`] works on 10 ms frames: the mic at the capture rate (48 kHz in this project) in,
//! processed mono audio at the output rate (16 kHz for speech recognition) out.
//!
//! [sonora]: https://crates.io/crates/sonora

pub mod aec16k;
pub mod metrics;
pub mod mic;

use sonora::config::{
    AdaptiveDigital, EchoCanceller, GainController2, HighPassFilter, NoiseSuppression,
    NoiseSuppressionLevel,
};
use sonora::{AudioProcessing, Config, StreamConfig};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DspError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("frame has {got} samples, expected {expected}")]
    FrameSize { got: usize, expected: usize },
    #[error("audio processing failed: {0:?}")]
    Backend(sonora::Error),
}

pub type Result<T, E = DspError> = std::result::Result<T, E>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsLevel {
    Low,
    Moderate,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DspConfig {
    /// Mic (capture) sample rate.
    pub capture_rate: u32,
    /// Reference (render) sample rate and channel count.
    pub render_rate: u32,
    pub render_channels: u16,
    /// Rate of the processed output.
    pub output_rate: u32,
    pub echo_cancellation: bool,
    pub noise_suppression: Option<NsLevel>,
    /// Adaptive digital gain (AGC2) with limiter.
    pub gain_control: bool,
    pub high_pass_filter: bool,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            capture_rate: 48_000,
            render_rate: 48_000,
            render_channels: 2,
            output_rate: 16_000,
            echo_cancellation: true,
            noise_suppression: Some(NsLevel::Moderate),
            gain_control: true,
            high_pass_filter: true,
        }
    }
}

/// Echo-canceller statistics as reported by the backend (all optional; `None` while unknown).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DspStats {
    pub erl_db: Option<f64>,
    pub erle_db: Option<f64>,
    pub delay_ms: Option<i32>,
    pub delay_median_ms: Option<i32>,
    pub divergent_filter_fraction: Option<f64>,
}

pub struct Processor {
    cfg: DspConfig,
    apm: AudioProcessing,
    capture_in: StreamConfig,
    capture_out: StreamConfig,
    render_cfg: StreamConfig,
    render_planes: Vec<Vec<f32>>,
    render_scratch: Vec<Vec<f32>>,
}

const VALID_RATES: [u32; 4] = [8_000, 16_000, 32_000, 48_000];

impl Processor {
    pub fn new(cfg: DspConfig) -> Result<Self> {
        for (name, rate) in [
            ("capture", cfg.capture_rate),
            ("render", cfg.render_rate),
            ("output", cfg.output_rate),
        ] {
            if !VALID_RATES.contains(&rate) {
                return Err(DspError::Config(format!(
                    "{name} rate {rate} Hz is not one of {VALID_RATES:?}"
                )));
            }
        }
        if !(1..=2).contains(&cfg.render_channels) {
            return Err(DspError::Config("render must be mono or stereo".into()));
        }
        let config = Config {
            echo_canceller: cfg.echo_cancellation.then(EchoCanceller::default),
            noise_suppression: cfg.noise_suppression.map(|l| NoiseSuppression {
                level: match l {
                    NsLevel::Low => NoiseSuppressionLevel::Low,
                    NsLevel::Moderate => NoiseSuppressionLevel::Moderate,
                    NsLevel::High => NoiseSuppressionLevel::High,
                    NsLevel::VeryHigh => NoiseSuppressionLevel::VeryHigh,
                },
                ..Default::default()
            }),
            // Adaptive digital gain must be enabled explicitly (WebRTC's default is fixed 0 dB only).
            gain_controller2: cfg.gain_control.then(|| GainController2 {
                adaptive_digital: Some(AdaptiveDigital::default()),
                ..Default::default()
            }),
            high_pass_filter: cfg.high_pass_filter.then(HighPassFilter::default),
            ..Default::default()
        };
        let capture_in = StreamConfig::new(cfg.capture_rate, 1);
        let capture_out = StreamConfig::new(cfg.output_rate, 1);
        let render_cfg = StreamConfig::new(cfg.render_rate, cfg.render_channels);
        let apm = AudioProcessing::builder()
            .config(config)
            .capture_config(capture_in)
            .render_config(render_cfg)
            .build();
        let rf = (cfg.render_rate / 100) as usize;
        Ok(Self {
            render_planes: vec![vec![0.0; rf]; cfg.render_channels as usize],
            render_scratch: vec![vec![0.0; rf]; cfg.render_channels as usize],
            cfg,
            apm,
            capture_in,
            capture_out,
            render_cfg,
        })
    }

    pub fn config(&self) -> &DspConfig {
        &self.cfg
    }

    /// Processing latency of the capture path in output samples, measured once per
    /// configuration by sending a test signal through a scratch instance (with AGC disabled so
    /// gain changes cannot distort the measurement). With AEC on at 48 kHz, sonora's band-split
    /// path delays output by 128 samples at 16 kHz (8 ms) relative to the bypass path.
    pub fn measure_latency(cfg: &DspConfig) -> Result<usize> {
        let mut probe_cfg = cfg.clone();
        probe_cfg.gain_control = false;
        probe_cfg.noise_suppression = None;
        let mut p = Processor::new(probe_cfg)?;
        let (cf, of) = (p.capture_frame_len(), p.output_frame_len());
        // Deterministic broadband probe: 0.5 s of noise at -30 dBFS, then silence.
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let frames = 100;
        let mut input = vec![0.0f32; cf * frames];
        for v in input.iter_mut().take(cf * 50) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.05;
        }
        let silent_ref = vec![0.0f32; p.render_frame_len()];
        let mut out = vec![0.0f32; of * frames];
        for (m, o) in input.chunks_exact(cf).zip(out.chunks_exact_mut(of)) {
            p.process_render(&silent_ref)?;
            p.process_capture(m, o)?;
        }
        // Reference: the same probe through a bypass instance (the resampler's own delay is
        // common to both and therefore excluded; we report the extra delay AEC introduces).
        let mut b = Processor::new(DspConfig {
            echo_cancellation: false,
            noise_suppression: None,
            gain_control: false,
            ..cfg.clone()
        })?;
        let mut base = vec![0.0f32; of * frames];
        for (m, o) in input.chunks_exact(cf).zip(base.chunks_exact_mut(of)) {
            b.process_render(&silent_ref)?;
            b.process_capture(m, o)?;
        }
        let start = of * 5;
        Ok(metrics::find_lag(
            &out[start..],
            &base[start..],
            of * 10,
            of * 20,
        ))
    }

    /// Samples per 10 ms mic frame.
    pub fn capture_frame_len(&self) -> usize {
        (self.cfg.capture_rate / 100) as usize
    }

    /// Interleaved samples per 10 ms reference frame.
    pub fn render_frame_len(&self) -> usize {
        (self.cfg.render_rate / 100) as usize * self.cfg.render_channels as usize
    }

    /// Samples per 10 ms output frame.
    pub fn output_frame_len(&self) -> usize {
        (self.cfg.output_rate / 100) as usize
    }

    /// Feeds 10 ms of the echo reference (interleaved). Call once per frame, *before*
    /// [`Self::process_capture`] for the mic frame recorded at the same time.
    pub fn process_render(&mut self, interleaved: &[f32]) -> Result<()> {
        if interleaved.len() != self.render_frame_len() {
            return Err(DspError::FrameSize {
                got: interleaved.len(),
                expected: self.render_frame_len(),
            });
        }
        let ch = self.cfg.render_channels as usize;
        for (i, frame) in interleaved.chunks_exact(ch).enumerate() {
            for (c, &s) in frame.iter().enumerate() {
                self.render_planes[c][i] = s;
            }
        }
        let src: Vec<&[f32]> = self.render_planes.iter().map(|p| p.as_slice()).collect();
        let mut dst: Vec<&mut [f32]> = self
            .render_scratch
            .iter_mut()
            .map(|p| p.as_mut_slice())
            .collect();
        let rc = self.render_cfg;
        self.apm
            .process_render_f32_with_config(&src, &rc, &rc, &mut dst)
            .map_err(DspError::Backend)
    }

    /// Processes 10 ms of mono mic audio; writes 10 ms at the output rate to `out`.
    pub fn process_capture(&mut self, mic: &[f32], out: &mut [f32]) -> Result<()> {
        if mic.len() != self.capture_frame_len() {
            return Err(DspError::FrameSize {
                got: mic.len(),
                expected: self.capture_frame_len(),
            });
        }
        if out.len() != self.output_frame_len() {
            return Err(DspError::FrameSize {
                got: out.len(),
                expected: self.output_frame_len(),
            });
        }
        let (ci, co) = (self.capture_in, self.capture_out);
        self.apm
            .process_capture_f32_with_config(&[mic], &ci, &co, &mut [out])
            .map_err(DspError::Backend)
    }

    pub fn stats(&self) -> DspStats {
        let s = self.apm.statistics();
        DspStats {
            erl_db: s.echo_return_loss,
            erle_db: s.echo_return_loss_enhancement,
            delay_ms: s.delay_ms,
            delay_median_ms: s.delay_median_ms,
            divergent_filter_fraction: s.divergent_filter_fraction,
        }
    }
}

#[cfg(test)]
mod tests;

/// System-audio path for speech recognition: downmix to mono and resample to 16 kHz with the
/// same resampler the mic path uses. No other processing (the far end is already clean).
pub struct SystemPath {
    channels: usize,
    mono: Vec<f32>,
    rs: sonora_common_audio::push_sinc_resampler::PushSincResampler,
}

impl SystemPath {
    pub fn new(input_rate: u32, channels: u16) -> Result<Self> {
        if !VALID_RATES.contains(&input_rate) || channels == 0 {
            return Err(DspError::Config(format!(
                "unsupported system format {input_rate} Hz x{channels}"
            )));
        }
        let n = input_rate as usize / 100;
        Ok(Self {
            channels: channels as usize,
            mono: vec![0.0; n],
            rs: sonora_common_audio::push_sinc_resampler::PushSincResampler::new(n, 160),
        })
    }

    /// 10 ms of interleaved input → 160 samples at 16 kHz.
    pub fn process(&mut self, interleaved: &[f32], out: &mut [f32]) -> Result<()> {
        if interleaved.len() != self.mono.len() * self.channels || out.len() != 160 {
            return Err(DspError::FrameSize {
                got: interleaved.len(),
                expected: self.mono.len() * self.channels,
            });
        }
        for (m, f) in self
            .mono
            .iter_mut()
            .zip(interleaved.chunks_exact(self.channels))
        {
            *m = f.iter().sum::<f32>() / self.channels as f32;
        }
        self.rs.resample(&self.mono, out);
        Ok(())
    }
}
