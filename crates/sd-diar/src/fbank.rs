//! Kaldi-compatible log-mel filterbank features, as WeSpeaker's front end computes them
//! (`torchaudio.compliance.kaldi.fbank`): 16 kHz input scaled to the int16 range, 25 ms
//! frames every 10 ms (`snip_edges`), DC offset removed, pre-emphasis 0.97, Hamming window,
//! 512-point FFT, power spectrum, 80 triangular mel bins between 20 Hz and 8 kHz on Kaldi's
//! mel scale, `ln(max(e, FLT_EPSILON))`, dither 0. The features are then mean-normalised per
//! dimension over the segment (CMN), as WeSpeaker does before the embedding model.
//! Implemented from the Kaldi feature definition.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const DIM: usize = 80;
const FRAME: usize = 400;
const SHIFT: usize = 160;
const NFFT: usize = 512;
const PREEMPH: f32 = 0.97;

fn mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

pub struct Fbank {
    window: Vec<f32>,
    /// Per mel bin: first FFT bin and its weights.
    banks: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Self {
        let window = (0..FRAME)
            .map(|i| {
                (0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME - 1) as f64).cos())
                    as f32
            })
            .collect();
        let num_fft_bins = NFFT / 2;
        let bin_width = 16_000.0 / NFFT as f64;
        let (lo, hi) = (mel(20.0), mel(8000.0));
        let delta = (hi - lo) / (DIM + 1) as f64;
        let banks = (0..DIM)
            .map(|b| {
                let left = lo + b as f64 * delta;
                let center = left + delta;
                let right = center + delta;
                let mut first = None;
                let mut weights = Vec::new();
                for i in 0..num_fft_bins {
                    let m = mel(bin_width * i as f64);
                    if m > left && m < right {
                        let w = if m <= center {
                            (m - left) / (center - left)
                        } else {
                            (right - m) / (right - center)
                        };
                        first.get_or_insert(i);
                        weights.push(w as f32);
                    } else if first.is_some() {
                        break;
                    }
                }
                (first.unwrap_or(0), weights)
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(NFFT);
        Self { window, banks, fft }
    }

    /// Number of frames for `samples` samples (Kaldi `snip_edges`).
    pub fn frames(samples: usize) -> usize {
        if samples < FRAME {
            0
        } else {
            1 + (samples - FRAME) / SHIFT
        }
    }

    /// Features of 16 kHz mono audio in [-1, 1]: [frames × DIM] row-major, mean-normalised.
    pub fn compute(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let (mut out, frames) = self.log_mel(pcm);
        // Cepstral mean normalisation over the segment.
        if frames > 0 {
            for d in 0..DIM {
                let m = (0..frames).map(|t| out[t * DIM + d]).sum::<f32>() / frames as f32;
                for t in 0..frames {
                    out[t * DIM + d] -= m;
                }
            }
        }
        (out, frames)
    }

    /// Log-mel energies before mean normalisation.
    fn log_mel(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let frames = Self::frames(pcm.len());
        let mut out = vec![0f32; frames * DIM];
        let mut buf = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut frame = [0f32; FRAME];
        for t in 0..frames {
            let src = &pcm[t * SHIFT..t * SHIFT + FRAME];
            for (d, s) in frame.iter_mut().zip(src) {
                *d = s * 32768.0;
            }
            let mean = frame.iter().sum::<f32>() / FRAME as f32;
            frame.iter_mut().for_each(|v| *v -= mean);
            for i in (1..FRAME).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            buf.iter_mut().for_each(|v| *v = 0.0);
            for ((b, f), w) in buf.iter_mut().zip(&frame).zip(&self.window) {
                *b = f * w;
            }
            self.fft
                .process_with_scratch(&mut buf, &mut spec, &mut scratch)
                .expect("buffer sizes from the plan");
            let row = &mut out[t * DIM..(t + 1) * DIM];
            for (r, (first, weights)) in row.iter_mut().zip(&self.banks) {
                let e: f32 = weights
                    .iter()
                    .zip(&spec[*first..])
                    .map(|(w, c)| w * c.norm_sqr())
                    .sum();
                *r = e.max(f32::EPSILON).ln();
            }
        }
        (out, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_and_mean_normalisation() {
        assert_eq!(Fbank::frames(399), 0);
        assert_eq!(Fbank::frames(400), 1);
        assert_eq!(Fbank::frames(16_000), 98);
        let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let (f, n) = Fbank::new().compute(&pcm);
        assert_eq!(n, 98);
        for d in 0..DIM {
            let m: f32 = (0..n).map(|t| f[t * DIM + d]).sum::<f32>() / n as f32;
            assert!(m.abs() < 1e-3, "dim {d}: {m}");
        }
    }

    #[test]
    fn a_tone_lands_in_the_mel_bin_centred_on_it() {
        let fb = Fbank::new();
        assert!(fb.banks.windows(2).all(|w| w[0].0 <= w[1].0));
        assert!(fb.banks.iter().all(|b| !b.1.is_empty()));
        for hz in [300.0f32, 1000.0, 4000.0] {
            let pcm: Vec<f32> = (0..16_000)
                .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 16_000.0).sin() * 0.5)
                .collect();
            let (f, n) = fb.log_mel(&pcm);
            let peak = (0..DIM)
                .max_by(|&a, &b| f[(n / 2) * DIM + a].total_cmp(&f[(n / 2) * DIM + b]))
                .unwrap();
            let (lo, hi) = (mel(20.0), mel(8000.0));
            let center_mel = lo + (peak + 1) as f64 * (hi - lo) / (DIM + 1) as f64;
            let center_hz = 700.0 * ((center_mel / 1127.0).exp() - 1.0);
            let bin_width = (hz as f64) * 0.15 + 40.0;
            assert!(
                (center_hz - hz as f64).abs() < bin_width,
                "{hz} Hz -> bin {peak} at {center_hz:.0} Hz"
            );
        }
    }
}
