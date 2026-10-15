use std::path::Path;

use sd_core::Source;

use crate::error::Result;
use crate::reader::JournalReader;

#[derive(Debug, Clone, Copy)]
pub struct ExportOptions {
    /// Write silence for timeline gaps so the WAV stays aligned with the session clock.
    pub fill_gaps: bool,
    /// Gaps longer than this are not filled (guards against writing hours of silence when a
    /// device position jumps). They are still reported.
    pub max_gap_fill_secs: f64,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            fill_gaps: true,
            max_gap_fill_secs: 600.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExportReport {
    pub frames_written: u64,
    pub audio_frames: u64,
    pub gap_frames_filled: u64,
    pub gaps_not_filled: u64,
}

/// Exports one source of a journal to a 16-bit PCM WAV file.
pub fn export_wav(
    reader: &JournalReader,
    source: Source,
    out: impl AsRef<Path>,
    opts: ExportOptions,
) -> Result<ExportReport> {
    let format = reader.format_of(source)?;
    let spec = hound::WavSpec {
        channels: format.channels,
        sample_rate: format.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(out.as_ref(), spec)?;
    let max_fill = (opts.max_gap_fill_secs.max(0.0) * format.sample_rate as f64) as u64;
    let mut rep = ExportReport::default();
    let mut expected: Option<u64> = None;

    for block in reader.blocks(source)? {
        let block = block?;
        if let Some(exp) = expected {
            let gap = block.meta.first_frame.saturating_sub(exp);
            if gap > 0 {
                if opts.fill_gaps && gap <= max_fill {
                    for _ in 0..gap * format.channels as u64 {
                        w.write_sample(0i16)?;
                    }
                    rep.gap_frames_filled += gap;
                    rep.frames_written += gap;
                } else {
                    rep.gaps_not_filled += 1;
                }
            }
        }
        for &s in &block.samples {
            w.write_sample(s)?;
        }
        rep.audio_frames += block.meta.frames as u64;
        rep.frames_written += block.meta.frames as u64;
        expected = Some(block.meta.first_frame + block.meta.frames as u64);
    }
    w.finalize()?;
    Ok(rep)
}
