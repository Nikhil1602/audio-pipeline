//! The complete microphone chain used for speech recognition.
//!
//! ```text
//! mic 48 kHz ─┐                    ┌─ full ──┐
//!             ├─► Aec16k (AEC3) ───┤         ├─► AsrMixer ─┐
//! ref 48 kHz ─┘   + EchoGate       └─ linear ┘             ├─► crossfade ─► NS / AGC2 ─► 16 kHz out
//!                 (echo present?) ── bypass (mic, delayed) ─┘
//! ```
//!
//! Why a gate: M2 measurements on a laptop whose driver already cancels speaker echo showed
//! that running AEC anyway degrades clean near-end speech (SI-SDR 41.6 dB → 9–14 dB). The gate
//! enables our AEC only while the mic's loudness envelope actually follows the far end's
//! (echo is present); otherwise the mic passes through untouched.

use sonora::config::{AdaptiveDigital, GainController2, NoiseSuppression, NoiseSuppressionLevel};
use sonora::high_pass_filter::HighPassFilter;
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora_common_audio::push_sinc_resampler::PushSincResampler;

use crate::aec16k::{Aec16k, AsrMixer, FULL_LATENCY};
use crate::{DspError, NsLevel, Result};

const FRAME: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecMode {
    /// Never cancel echo (e.g. the device already does it).
    Off,
    /// Cancel echo only while echo is detected (default).
    Auto,
    /// Always cancel echo.
    On,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MicConfig {
    pub input_rate: u32,
    pub reference_channels: u16,
    pub aec: AecMode,
    /// Hybrid mixer threshold (dB of suppression above which a frame is treated as echo).
    pub hybrid_threshold_db: f32,
    pub noise_suppression: Option<NsLevel>,
    pub gain_control: bool,
    /// AGC2 gain applied before its level estimate settles (dB). WebRTC's default is 15 dB;
    /// on a real recording that drove the first second of speech ~11 dB above the settled
    /// level and into the limiter, while 10 dB did not (docs/m2-report.md).
    pub agc_initial_gain_db: f32,
}

impl Default for MicConfig {
    fn default() -> Self {
        Self {
            input_rate: 48_000,
            reference_channels: 2,
            aec: AecMode::Auto,
            hybrid_threshold_db: 20.0,
            noise_suppression: None,
            gain_control: true,
            agc_initial_gain_db: 10.0,
        }
    }
}

/// What the echo gate currently knows about the acoustic echo path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoState {
    /// Not enough far-end activity yet to decide. Mic audio passes through untouched, so if
    /// there *is* echo it is not removed yet: speech found here is flagged for downstream
    /// checks (see `RecorderEvent::Speech::echo_unverified`).
    Unknown,
    /// Echo detected (or AEC forced on); the AEC output is used.
    Echo,
    /// Sustained evidence of no echo (or AEC forced off); the mic passes through.
    NoEcho,
}

/// Echo-presence detector: correlation between the mic's and the reference's 10 ms log-power
/// envelopes, at the best lag in 0..=300 ms, over far-end-active frames of a sliding window.
///
/// Calibration (M2, real hardware): device-cancelled echo 0.00–0.09; strong laptop echo
/// 0.46–0.60 per 10 s window; two independent talkers in double talk up to 0.22.
#[derive(Debug, Clone)]
pub struct EchoGate {
    mic_db: Vec<f32>,
    ref_db: Vec<f32>,
    pos: usize,
    filled: usize,
    frames_since_eval: usize,
    enabled: bool,
    low_evals: u32,
    last_correlation: Option<f32>,
    state: EchoState,
}

impl EchoGate {
    pub const WINDOW: usize = 1_000; // 10 s of 10 ms frames
    pub const MAX_LAG: usize = 30; // 300 ms
    pub const EVAL_EVERY: usize = 100; // 1 s
    pub const ACTIVE_DB: f32 = -60.0;
    pub const MIN_ACTIVE: usize = 300; // 3 s of far-end activity
    pub const ON: f32 = 0.35; // independent talkers reached 0.22; real echo 0.46-0.60
    pub const OFF: f32 = 0.15;
    pub const OFF_EVALS: u32 = 10; // ~10 s of sustained low correlation

    pub fn new(start_enabled: bool) -> Self {
        Self {
            mic_db: vec![-120.0; Self::WINDOW],
            ref_db: vec![-120.0; Self::WINDOW],
            pos: 0,
            filled: 0,
            frames_since_eval: 0,
            enabled: start_enabled,
            low_evals: 0,
            last_correlation: None,
            state: if start_enabled {
                EchoState::Echo
            } else {
                EchoState::Unknown
            },
        }
    }

    pub fn state(&self) -> EchoState {
        self.state
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn last_correlation(&self) -> Option<f32> {
        self.last_correlation
    }

    /// Feeds one frame's mic and reference power (dBFS). Returns whether AEC should be on.
    pub fn push(&mut self, mic_db: f32, ref_db: f32) -> bool {
        self.mic_db[self.pos] = mic_db;
        self.ref_db[self.pos] = ref_db;
        self.pos = (self.pos + 1) % Self::WINDOW;
        self.filled = (self.filled + 1).min(Self::WINDOW);
        self.frames_since_eval += 1;
        if self.frames_since_eval >= Self::EVAL_EVERY {
            self.frames_since_eval = 0;
            if let Some(c) = self.correlation() {
                self.last_correlation = Some(c);
                if c > Self::ON {
                    self.enabled = true;
                    self.state = EchoState::Echo;
                    self.low_evals = 0;
                } else if c < Self::OFF {
                    self.low_evals += 1;
                    if self.low_evals >= Self::OFF_EVALS {
                        self.enabled = false;
                        self.state = EchoState::NoEcho;
                    }
                } else {
                    self.low_evals = 0;
                }
            }
        }
        self.enabled
    }

    /// Best Pearson correlation over lags, or None without enough far-end activity.
    fn correlation(&self) -> Option<f32> {
        let n = self.filled;
        // Oldest-to-newest views.
        let at = |buf: &[f32], i: usize| buf[(self.pos + Self::WINDOW - n + i) % Self::WINDOW];
        let mut best: Option<f32> = None;
        for lag in 0..=Self::MAX_LAG.min(n.saturating_sub(1)) {
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut k) =
                (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
            for i in lag..n {
                let r = at(&self.ref_db, i - lag);
                if r <= Self::ACTIVE_DB {
                    continue;
                }
                let m = at(&self.mic_db, i) as f64;
                let r = r as f64;
                sx += r;
                sy += m;
                sxx += r * r;
                syy += m * m;
                sxy += r * m;
                k += 1;
            }
            if k < Self::MIN_ACTIVE {
                continue;
            }
            let kf = k as f64;
            let cov = sxy - sx * sy / kf;
            let vx = sxx - sx * sx / kf;
            let vy = syy - sy * sy / kf;
            if vx <= 0.0 || vy <= 0.0 {
                continue;
            }
            let c = (cov / (vx * vy).sqrt()) as f32;
            best = Some(best.map_or(c, |b: f32| b.max(c)));
        }
        best
    }
}

/// Per-frame status of the chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicStatus {
    pub aec_active: bool,
    pub echo_correlation: Option<f32>,
    pub echo_state: EchoState,
    /// The reference (far end) was active in this frame.
    pub far_active: bool,
}

pub struct MicProcessor {
    cfg: MicConfig,
    aec: Aec16k,
    mixer: AsrMixer,
    gate: EchoGate,
    post: Option<AudioProcessing>,
    bypass_delay: Vec<f32>,
    weight: f32,
    full: Vec<f32>,
    linear: Vec<f32>,
    hybrid: Vec<f32>,
    bypass: Vec<f32>,
    /// Bypass path: the same resampler and high-pass filter as the AEC path, then a delay
    /// line of `FULL_LATENCY` samples so both paths stay sample-aligned for the crossfade.
    bypass_rs: PushSincResampler,
    bypass_hpf: HighPassFilter,
    resampled: Vec<Vec<f32>>,
    status: MicStatus,
}

fn power_db(x: &[f32]) -> f32 {
    let p = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len().max(1) as f64;
    (10.0 * (p + 1e-12).log10()) as f32
}

impl MicProcessor {
    pub fn new(cfg: MicConfig) -> Result<Self> {
        let post = if cfg.noise_suppression.is_some() || cfg.gain_control {
            let config = Config {
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
                    adaptive_digital: Some(AdaptiveDigital {
                        initial_gain_db: cfg.agc_initial_gain_db,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            Some(
                AudioProcessing::builder()
                    .config(config)
                    .capture_config(StreamConfig::new(16_000, 1))
                    .render_config(StreamConfig::new(16_000, 1))
                    .build(),
            )
        } else {
            None
        };
        // Auto starts with AEC off: switching on takes ~3-4 s of far-end activity once echo is
        // evident, whereas switching off (sustained no-echo evidence) takes ~15 s, during which
        // clean near-end speech would be degraded. Echo-free setups (headphones, devices with
        // their own AEC) are common, so they get the clean path from the start.
        let start_enabled = cfg.aec == AecMode::On;
        let mut mixer = AsrMixer::default();
        mixer.echo_threshold_db = cfg.hybrid_threshold_db;
        Ok(Self {
            aec: Aec16k::new(cfg.input_rate, cfg.reference_channels)?,
            bypass_rs: PushSincResampler::new((cfg.input_rate / 100) as usize, FRAME),
            bypass_hpf: HighPassFilter::new(16_000, 1),
            mixer,
            gate: EchoGate::new(start_enabled),
            post,
            bypass_delay: vec![0.0; FULL_LATENCY],
            weight: if start_enabled { 1.0 } else { 0.0 },
            full: vec![0.0; FRAME],
            linear: vec![0.0; FRAME],
            hybrid: vec![0.0; FRAME],
            bypass: vec![0.0; FRAME],
            resampled: vec![vec![0.0; FRAME]],
            status: MicStatus {
                aec_active: start_enabled,
                echo_correlation: None,
                echo_state: match cfg.aec {
                    AecMode::On => EchoState::Echo,
                    AecMode::Off => EchoState::NoEcho,
                    AecMode::Auto => EchoState::Unknown,
                },
                far_active: false,
            },
            cfg,
        })
    }

    /// Output latency in 16 kHz samples relative to the input (constant in every mode).
    pub fn latency(&self) -> usize {
        FULL_LATENCY
    }

    pub fn status(&self) -> MicStatus {
        self.status
    }

    /// Processes 10 ms: `mic` mono and `reference` interleaved at the input rate; writes 160
    /// samples at 16 kHz to `out`.
    pub fn process(&mut self, mic: &[f32], reference: &[f32], out: &mut [f32]) -> Result<()> {
        if out.len() != FRAME {
            return Err(DspError::FrameSize {
                got: out.len(),
                expected: FRAME,
            });
        }
        self.aec
            .process(mic, reference, &mut self.full, &mut self.linear)?;
        self.mixer.push(&self.linear, &self.full, &mut self.hybrid);

        // Bypass path: same resampler + HPF as the AEC path, delayed to FULL_LATENCY.
        self.bypass_rs.resample(mic, &mut self.resampled[0]);
        self.bypass_hpf.process_channels(&mut self.resampled);
        let cur = &self.resampled[0];
        let d = self.bypass_delay.len();
        self.bypass[..d].copy_from_slice(&self.bypass_delay);
        self.bypass[d..].copy_from_slice(&cur[..FRAME - d]);
        self.bypass_delay.copy_from_slice(&cur[FRAME - d..]);

        // Gate on the (aligned) input mic vs. reference envelopes.
        let ch = self.cfg.reference_channels as usize;
        let ref_mono_db = {
            let p = reference.chunks_exact(ch).map(|f| {
                let m = f.iter().sum::<f32>() / ch as f32;
                m as f64 * m as f64
            });
            let n = reference.len() / ch;
            (10.0 * (p.sum::<f64>() / n.max(1) as f64 + 1e-12).log10()) as f32
        };
        let target_on = match self.cfg.aec {
            AecMode::Off => false,
            AecMode::On => true,
            AecMode::Auto => self.gate.push(power_db(&self.resampled[0]), ref_mono_db),
        };
        self.status = MicStatus {
            aec_active: target_on,
            echo_correlation: self.gate.last_correlation(),
            echo_state: match self.cfg.aec {
                AecMode::On => EchoState::Echo,
                AecMode::Off => EchoState::NoEcho,
                AecMode::Auto => self.gate.state(),
            },
            far_active: ref_mono_db > EchoGate::ACTIVE_DB,
        };

        let target = if target_on { 1.0f32 } else { 0.0 };
        let start = self.weight;
        for (i, (o, (h, b))) in out
            .iter_mut()
            .zip(self.hybrid.iter().zip(&self.bypass))
            .enumerate()
        {
            let w = start + (target - start) * (i + 1) as f32 / FRAME as f32;
            *o = w * h + (1.0 - w) * b;
        }
        self.weight = target;

        if let Some(p) = &mut self.post {
            let input = out.to_vec();
            let cfg = StreamConfig::new(16_000, 1);
            p.process_render_f32_with_config(
                &[&[0.0f32; FRAME]],
                &cfg,
                &cfg,
                &mut [&mut [0.0f32; FRAME]],
            )
            .map_err(DspError::Backend)?;
            p.process_capture_f32_with_config(&[&input], &cfg, &cfg, &mut [out])
                .map_err(DspError::Backend)?;
        }
        Ok(())
    }
}
