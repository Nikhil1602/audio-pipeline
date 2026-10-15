//! `sdctl debug replay`: re-run the timeline aligner over the packets recorded in a journal.
//!
//! Each block that holds real device audio (not GAP_FILL) is treated as one packet: its host
//! timestamp, frame count and DISCONTINUITY flag are fed to a fresh [`TimelineAligner`], with
//! device positions synthesized as continuous (what the device reported while recording).
//! Useful to check aligner changes against real driver behaviour without the hardware.

use std::path::PathBuf;
use std::process::ExitCode;

use sd_core::timeline::{AlignerConfig, TimelineAligner};
use sd_core::{BlockFlags, Source};
use sd_journal::JournalReader;

use crate::CliResult;

#[derive(clap::Args)]
pub struct Args {
    dir: PathBuf,
    #[arg(long)]
    source: Source,
}

pub fn run(args: Args) -> CliResult {
    let reader = JournalReader::open(&args.dir)?;
    let rate = reader.format_of(args.source)?.sample_rate;
    let mut aligner = TimelineAligner::new(AlignerConfig::new(rate));
    let mut position = 0f64;
    let mut first_host = None;
    let mut last_end_host = 0u64;
    let mut events = Vec::new();
    for b in reader.blocks(args.source)? {
        let m = b?.meta;
        if m.flags.contains(BlockFlags::GAP_FILL) {
            // Silence the recorder synthesized; the device delivered nothing here. Treat as a
            // new stream boundary only if it was idle fill (positions unknown across it).
            aligner.new_stream();
            continue;
        }
        let t0 = *first_host.get_or_insert(m.host_time.as_ns());
        if aligner.expected_next().is_none() {
            aligner.start_at(m.host_time);
        }
        let glitch = m.flags.contains(BlockFlags::DISCONTINUITY);
        let al = aligner.on_device_packet(m.host_time, m.frames, Some(position), glitch);
        position += m.frames as f64;
        if al.gap_frames > 0 || al.trim_frames > 0 {
            events.push(format!(
                "  t={:8.3}s gap {:>6} frames ({:.1} ms) trim {:>4}{}",
                (m.host_time.as_ns() - t0) as f64 / 1e9,
                al.gap_frames,
                al.gap_frames as f64 * 1e3 / rate as f64,
                al.trim_frames,
                if glitch { "  [device flag]" } else { "" }
            ));
        }
        last_end_host = m.host_time.as_ns() + m.frames as u64 * 1_000_000_000 / rate as u64;
    }
    let st = aligner.stats();
    for e in &events {
        println!("{e}");
    }
    let timeline_end = aligner.expected_next().map(|t| t.as_ns()).unwrap_or(0);
    println!(
        "packets {}, gaps {} ({:.1} ms), trims {} ({:.1} ms), timestamp overrides {}; timeline end vs last packet end: {:+.1} ms",
        st.packets,
        st.gaps,
        st.gap_frames as f64 * 1e3 / rate as f64,
        st.overlaps,
        st.trimmed_frames as f64 * 1e3 / rate as f64,
        st.timestamp_overrides,
        (timeline_end as f64 - last_end_host as f64) / 1e6
    );
    Ok(ExitCode::SUCCESS)
}
