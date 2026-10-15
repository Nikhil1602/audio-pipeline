//! Compression ratio of the journal codec on 16-bit WAV files, coded in 10 ms blocks.
//!
//! `cargo run --release -p sd-journal --example codec_ratio -- a.wav [b.wav ...]`
//! Optional `--from SECONDS --to SECONDS` restricts the measured range.

use std::time::Instant;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut range = (0.0f64, f64::INFINITY);
    if let Some(i) = args.iter().position(|a| a == "--from") {
        range.0 = args[i + 1].parse().expect("--from seconds");
        args.drain(i..i + 2);
    }
    if let Some(i) = args.iter().position(|a| a == "--to") {
        range.1 = args[i + 1].parse().expect("--to seconds");
        args.drain(i..i + 2);
    }
    for path in args {
        let r = hound::WavReader::open(&path).expect("open wav");
        let spec = r.spec();
        assert_eq!(spec.bits_per_sample, 16, "{path}: 16-bit WAV expected");
        let ch = spec.channels as usize;
        let all: Vec<i16> = r.into_samples::<i16>().map(|s| s.unwrap()).collect();
        let rate = spec.sample_rate as f64;
        let from = ((range.0 * rate) as usize * ch).min(all.len());
        let to = ((range.1 * rate).min(1e12) as usize * ch).min(all.len());
        let x = &all[from..to];
        let block = spec.sample_rate as usize / 100 * ch;
        let (mut enc, mut total) = (Vec::new(), 0usize);
        let t = Instant::now();
        for b in x.chunks(block) {
            let b = &b[..b.len() / ch * ch];
            sd_journal::codec::encode(b, ch, &mut enc);
            total += enc.len();
        }
        let el = t.elapsed().as_secs_f64();
        let secs = x.len() as f64 / ch as f64 / rate;
        let rms = (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64)
            .sqrt()
            / 32768.0;
        println!(
            "{path}: {secs:.1} s, {} Hz x{ch}, {:.1} dBFS rms -> {:.1}% of PCM ({:.0} kbit/s), encode {:.4}x real time",
            spec.sample_rate,
            20.0 * (rms + 1e-12).log10(),
            100.0 * total as f64 / (x.len() * 2) as f64,
            total as f64 * 8.0 / secs / 1000.0,
            el / secs
        );
    }
}
