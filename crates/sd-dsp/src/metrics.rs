//! Objective measures used by tests and benchmarks.

/// Mean power of `x` (mean square).
pub fn power(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64
}

/// RMS level in dBFS (full-scale sine = -3 dBFS); -inf-safe (floors at -200 dB).
pub fn rms_dbfs(x: &[f32]) -> f64 {
    10.0 * power(x).max(1e-20).log10()
}

/// Ratio of powers in dB: `10 log10(P(a) / P(b))`.
pub fn power_ratio_db(a: &[f32], b: &[f32]) -> f64 {
    10.0 * (power(a).max(1e-20) / power(b).max(1e-20)).log10()
}

/// Scale-invariant signal-to-distortion ratio (dB) of `estimate` against `reference`
/// (Le Roux et al., 2019). Both must be time-aligned and equally long.
pub fn si_sdr(estimate: &[f32], reference: &[f32]) -> f64 {
    let n = estimate.len().min(reference.len());
    let (e, r) = (&estimate[..n], &reference[..n]);
    let dot: f64 = e.iter().zip(r).map(|(&a, &b)| a as f64 * b as f64).sum();
    let rr: f64 = r.iter().map(|&b| (b as f64) * (b as f64)).sum();
    if rr == 0.0 {
        return f64::NEG_INFINITY;
    }
    let alpha = dot / rr;
    let (mut target, mut noise) = (0.0f64, 0.0f64);
    for (&a, &b) in e.iter().zip(r) {
        let t = alpha * b as f64;
        target += t * t;
        noise += (a as f64 - t) * (a as f64 - t);
    }
    10.0 * (target.max(1e-30) / noise.max(1e-30)).log10()
}

/// Finds the lag (in samples, `0..=max_lag`) by which `delayed` trails `reference`, by
/// maximizing normalized cross-correlation over the first `window` samples of `reference`.
pub fn find_lag(delayed: &[f32], reference: &[f32], max_lag: usize, window: usize) -> usize {
    let w = window.min(reference.len());
    let r = &reference[..w];
    let rr: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mut best = (0usize, f64::MIN);
    for lag in 0..=max_lag {
        if lag + w > delayed.len() {
            break;
        }
        let d = &delayed[lag..lag + w];
        let (mut dot, mut dd) = (0.0f64, 0.0f64);
        for (&a, &b) in d.iter().zip(r) {
            dot += a as f64 * b as f64;
            dd += (a as f64) * (a as f64);
        }
        let ncc = if dd > 0.0 && rr > 0.0 {
            dot / (dd * rr).sqrt()
        } else {
            0.0
        };
        if ncc > best.1 {
            best = (lag, ncc);
        }
    }
    best.0
}

/// Decimates by an integer factor after a crude moving-average anti-alias filter. Only for
/// comparing signals in metrics (not for audio output).
pub fn decimate(x: &[f32], factor: usize) -> Vec<f32> {
    x.chunks(factor)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_ratios() {
        let sine: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin()).collect();
        assert!((rms_dbfs(&sine) + 3.01).abs() < 0.05);
        let half: Vec<f32> = sine.iter().map(|v| v * 0.5).collect();
        assert!((power_ratio_db(&sine, &half) - 6.02).abs() < 0.01);
    }

    #[test]
    fn si_sdr_is_scale_invariant_and_penalizes_noise() {
        let r: Vec<f32> = (0..16_000)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let scaled: Vec<f32> = r.iter().map(|v| v * 0.3).collect();
        assert!(si_sdr(&scaled, &r) > 100.0);
        let noisy: Vec<f32> = r
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let v = si_sdr(&noisy, &r);
        assert!(v > 20.0 && v < 40.0, "{v}");
    }
}
