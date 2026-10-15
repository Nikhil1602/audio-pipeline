//! Offline diarization of known speech regions: each region is cut into overlapping windows,
//! each window gets a speaker embedding, the embeddings are clustered, and every 10 ms frame
//! of speech takes the speaker of the window whose centre is nearest.

use crate::cluster::{Merge, absorb_small, average_linkage, cut};
use crate::der::Turn;
use crate::embed::Embedder;
use crate::segment::SpeakerCount;
use crate::{Result, SAMPLE_RATE};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarConfig {
    pub window_s: f32,
    pub step_s: f32,
    /// Regions shorter than this are not embedded on their own; they take the speaker of
    /// the nearest embedded window.
    pub min_window_s: f32,
}

impl Default for DiarConfig {
    fn default() -> Self {
        Self {
            // Measured on AMI (docs/m4-report.md): 3 s windows every 1.5 s beat 1.5 s and 6 s.
            window_s: 3.0,
            step_s: 1.5,
            min_window_s: 0.4,
        }
    }
}

/// Speech windows (sample ranges) with their embeddings.
pub struct Embedded {
    pub windows: Vec<(u64, u64)>,
    pub embeddings: Vec<Vec<f32>>,
    /// Speech regions, in samples.
    pub regions: Vec<(u64, u64)>,
}

/// Cuts `regions` (sample ranges of 16 kHz `pcm`) into windows and embeds them.
pub fn embed_regions(
    pcm: &[f32],
    regions: &[(u64, u64)],
    embedder: &Embedder,
    cfg: &DiarConfig,
) -> Result<Embedded> {
    let sr = SAMPLE_RATE as f32;
    let (win, step, min) = (
        (cfg.window_s * sr) as u64,
        (cfg.step_s * sr) as u64,
        (cfg.min_window_s * sr) as u64,
    );
    let mut windows = Vec::new();
    let mut embeddings = Vec::new();
    for &(a, b) in regions {
        let b = b.min(pcm.len() as u64);
        if b <= a || b - a < min {
            continue;
        }
        let mut starts = Vec::new();
        if b - a <= win {
            starts.push(a);
        } else {
            let mut s = a;
            while s + win < b {
                starts.push(s);
                s += step;
            }
            starts.push(b - win);
        }
        for s in starts {
            let e = (s + win).min(b);
            if let Some(v) = embedder.embed(&pcm[s as usize..e as usize])? {
                windows.push((s, e));
                embeddings.push(v);
            }
        }
    }
    Ok(Embedded {
        windows,
        embeddings,
        regions: regions.to_vec(),
    })
}

/// The clustering dendrogram of the embedded windows (compute once, cut many times).
pub fn dendrogram(e: &Embedded) -> Vec<Merge> {
    average_linkage(&e.embeddings)
}

/// Speaker turns: clusters cut at `threshold` (Euclidean distance between unit embeddings)
/// or into `num_speakers` when known; then clusters with fewer than `min_cluster` windows
/// are absorbed into the nearest large cluster.
///
/// With `overlap`, frames where two speakers are active also get the nearest window in
/// time (within [`OVERLAP_REACH_S`]) whose speaker differs from the frame's own.
pub fn turns(
    e: &Embedded,
    merges: &[Merge],
    threshold: f32,
    num_speakers: Option<usize>,
    min_cluster: usize,
    absorb_max: f32,
    overlap: Option<&SpeakerCount>,
) -> Vec<Turn> {
    let labels = cut(e.embeddings.len(), merges, threshold, num_speakers);
    let labels = absorb_small(&e.embeddings, &labels, min_cluster, absorb_max);
    let mut out = label_frames(e, &labels);
    if let Some(count) = overlap {
        out.extend(second_speakers(e, &labels, count));
        out.sort_by(|a, b| a.start.total_cmp(&b.start));
    }
    out
}

/// How far (seconds) to look for the second speaker of an overlap frame.
pub const OVERLAP_REACH_S: f64 = 5.0;

/// Turns of the second speaker on overlap frames.
fn second_speakers(e: &Embedded, labels: &[usize], count: &SpeakerCount) -> Vec<Turn> {
    let hop = (SAMPLE_RATE / 100) as u64;
    let reach = (OVERLAP_REACH_S * SAMPLE_RATE as f64) as u64;
    let centres: Vec<u64> = e.windows.iter().map(|(a, b)| (a + b) / 2).collect();
    let mut out: Vec<Turn> = Vec::new();
    if centres.is_empty() {
        return out;
    }
    for &(a, b) in &e.regions {
        let mut f = a;
        while f < b {
            let c = f + hop / 2;
            if count.at(c) >= 2 {
                let j = nearest(&centres, c);
                let own = labels[j];
                // Nearest window in time with a different speaker, on either side.
                let mut best: Option<(u64, usize)> = None;
                for k in (0..j).rev() {
                    if centres[k] + reach < c {
                        break;
                    }
                    if labels[k] != own {
                        best = Some((c - centres[k], labels[k]));
                        break;
                    }
                }
                for k in j + 1..centres.len() {
                    if centres[k] > c + reach {
                        break;
                    }
                    if labels[k] != own {
                        let d = centres[k] - c;
                        if best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, labels[k]));
                        }
                        break;
                    }
                }
                if let Some((_, other)) = best {
                    let speaker = format!("spk{other}");
                    let (s, t) = (
                        f as f64 / SAMPLE_RATE as f64,
                        (f + hop).min(b) as f64 / SAMPLE_RATE as f64,
                    );
                    match out.last_mut() {
                        Some(last) if last.speaker == speaker && (s - last.end).abs() < 1e-6 => {
                            last.end = t
                        }
                        _ => out.push(Turn {
                            start: s,
                            end: t,
                            speaker,
                        }),
                    }
                }
            }
            f += hop;
        }
    }
    out
}

fn nearest(centres: &[u64], c: u64) -> usize {
    match centres.binary_search(&c) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= centres.len() => centres.len() - 1,
        Err(i) => {
            if c - centres[i - 1] <= centres[i] - c {
                i - 1
            } else {
                i
            }
        }
    }
}

/// Each 10 ms speech frame takes the label of the window whose centre is nearest.
pub fn label_frames(e: &Embedded, labels: &[usize]) -> Vec<Turn> {
    let hop = (SAMPLE_RATE / 100) as u64;
    let centres: Vec<u64> = e.windows.iter().map(|(a, b)| (a + b) / 2).collect();
    let mut out: Vec<Turn> = Vec::new();
    if centres.is_empty() {
        return out;
    }
    for &(a, b) in &e.regions {
        let mut f = a;
        let mut j = 0usize;
        while f < b {
            let c = f + hop / 2;
            // Windows are in time order; advance to the nearest centre.
            while j + 1 < centres.len() && centres[j + 1].abs_diff(c) <= centres[j].abs_diff(c) {
                j += 1;
            }
            let speaker = format!("spk{}", labels[j]);
            let (s, t) = (
                f as f64 / SAMPLE_RATE as f64,
                (f + hop).min(b) as f64 / SAMPLE_RATE as f64,
            );
            match out.last_mut() {
                Some(last) if last.speaker == speaker && (s - last.end).abs() < 1e-6 => {
                    last.end = t
                }
                _ => out.push(Turn {
                    start: s,
                    end: t,
                    speaker,
                }),
            }
            f += hop;
        }
    }
    out
}
