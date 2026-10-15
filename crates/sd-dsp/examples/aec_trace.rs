//! Diagnostic: per-10 ms trace of the AEC3 linear filter on a recorded mic + reference pair.
//!
//! `cargo run --release -p sd-dsp --example aec_trace -- mic.wav system.wav [offset_frames] > trace.tsv`
//!
//! Both WAVs are 16-bit at 48 kHz (mic mono, system mono or stereo), aligned so that sample 0
//! of each was captured at the same time, after shifting the system by `offset_frames` 10 ms
//! frames. Columns: time s, reference dBFS, mic dBFS (16 kHz, high-passed), linear dBFS,
//! full dBFS, AEC3 ERLE dB, AEC3 delay ms.

use sd_dsp::aec16k::Aec16k;

fn read(path: &str) -> (Vec<f32>, usize) {
    let r = hound::WavReader::open(path).expect("open wav");
    let ch = r.spec().channels as usize;
    assert_eq!(r.spec().sample_rate, 48_000, "{path}: 48 kHz expected");
    let x = r
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    (x, ch)
}

fn db(x: &[f32]) -> f32 {
    let p = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len().max(1) as f64;
    (10.0 * (p + 1e-12).log10()) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mic, _) = read(&args[0]);
    let (mut sys, ch) = read(&args[1]);
    let offset: i64 = args.get(2).map_or(0, |s| s.parse().unwrap());
    if offset > 0 {
        let mut pad = vec![0.0; offset as usize * 480 * ch];
        pad.extend_from_slice(&sys);
        sys = pad;
    } else if offset < 0 {
        sys.drain(..(-offset) as usize * 480 * ch);
    }
    let mut aec = Aec16k::new(48_000, ch as u16).unwrap();
    let mut hp = Aec16k::with_linear_export(48_000, ch as u16, false).unwrap();
    let (mut full, mut lin) = (vec![0.0; 160], vec![0.0; 160]);
    let (mut hp_full, mut hp_in) = (vec![0.0; 160], vec![0.0; 160]);
    let frames = (mic.len() / 480).min(sys.len() / (480 * ch));
    println!("t\tref_db\tmic_db\tlin_db\tfull_db\terle_db\tdelay_ms");
    for f in 0..frames {
        let m = &mic[f * 480..(f + 1) * 480];
        let r = &sys[f * 480 * ch..(f + 1) * 480 * ch];
        aec.process(m, r, &mut full, &mut lin).unwrap();
        // Without linear export, `linear` is the resampled, high-passed input: the mic as AEC
        // sees it, at the same latency as `linear`.
        hp.process(m, r, &mut hp_full, &mut hp_in).unwrap();
        let s = aec.stats();
        println!(
            "{:.2}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}",
            f as f64 / 100.0,
            db(r),
            db(&hp_in),
            db(&lin),
            db(&full),
            s.erle_db,
            s.delay_ms
        );
    }
}
