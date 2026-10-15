//! Live diarization of one audio source, utterance by utterance.
//!
//! * Each finished utterance (a final transcript's audio) is embedded with the measured
//!   windows (3 s every 1.5 s) and assigned at once to the nearest speaker centroid if it
//!   is within [`OnlineConfig::assign_threshold`], otherwise to a new speaker. Utterances
//!   too short for one window provisionally take the previous utterance's speaker.
//! * Every [`OnlineConfig::recluster_every_s`] of audio, and at the end, all windows so far
//!   are re-clustered offline (average linkage, small-cluster absorption). The new
//!   clusters are matched to the current speaker ids by maximum window overlap (Hungarian),
//!   so ids stay stable; utterances whose speaker changes are returned as revisions.
//! * Each window also has its own speaker (live: the nearest centroid if close enough,
//!   else the utterance's; after re-clustering: its cluster), so [`OnlineDiarizer::turns`]
//!   can split an utterance where the speaker changes within it.
//! * With a segmentation model ([`OnlineDiarizer::with_segmentation`]), frames of an
//!   utterance where two people speak at once are recorded; [`OnlineDiarizer::utterance`]
//!   gives them the nearest other speaker in time (within [`OVERLAP_REACH_S`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::cluster::{absorb_small, average_linkage, cut, min_cluster_size};
use crate::embed::{Embedder, distance};
use crate::segment::{FRAME_STEP, SegmentationModel, SpeakerCount};
use crate::{Result, SAMPLE_RATE};

/// How far (seconds) to look for the second speaker of an overlap.
pub const OVERLAP_REACH_S: f64 = 5.0;

/// One utterance's speakers, in absolute samples of the source's timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct UtteranceSpeakers {
    /// Majority speaker.
    pub speaker: u32,
    /// Who speaks when within the utterance: (start, end, speaker), in time order.
    pub turns: Vec<(u64, u64, u32)>,
    /// Second speakers where two people speak at once: (start, end, speaker).
    pub overlaps: Vec<(u64, u64, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineConfig {
    pub window_s: f32,
    pub step_s: f32,
    /// Shortest audio embedded on its own.
    pub min_embed_s: f32,
    /// Euclidean distance between unit vectors for joining an existing speaker live.
    pub assign_threshold: f32,
    /// Re-clustering cut (Euclidean, average linkage).
    pub cluster_threshold: f32,
    pub recluster_every_s: f32,
    /// Small clusters are absorbed only into a large cluster closer than this.
    pub absorb_max: f32,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            window_s: 3.0,
            step_s: 1.5,
            min_embed_s: 1.0,
            assign_threshold: 1.0,
            cluster_threshold: 1.0,
            recluster_every_s: 60.0,
            // AMI: 1.2 costs 0.1 point of live DER versus always absorbing, and keeps brief
            // distinct speakers apart (docs/m4-report.md).
            absorb_max: 1.2,
        }
    }
}

#[derive(Debug, Clone)]
struct Utterance {
    id: u64,
    speaker: u32,
    /// Indices of this utterance's windows.
    windows: Vec<usize>,
}

/// A window's (start, end) within an utterance and its embedding.
type Window = ((usize, usize), Vec<f32>);

pub struct OnlineDiarizer {
    embedder: Arc<Embedder>,
    cfg: OnlineConfig,
    embeddings: Vec<Vec<f32>>,
    window_utt: Vec<usize>,
    /// Absolute sample range of each window, its speaker now and as first emitted.
    windows: Vec<(u64, u64)>,
    window_speaker: Vec<u32>,
    window_first: Vec<u32>,
    /// Absolute sample range of each utterance.
    spans: Vec<(u64, u64)>,
    utterances: Vec<Utterance>,
    /// Per speaker id: sum of member embeddings and count (centroid = normalised sum).
    centroids: Vec<(u32, Vec<f32>, usize)>,
    next_speaker: u32,
    audio_since_recluster: u64,
    segmenter: Option<Arc<SegmentationModel>>,
    /// Per utterance: absolute sample ranges where two speakers are active.
    overlap_spans: Vec<Vec<(u64, u64)>>,
    index: HashMap<u64, usize>,
}

impl OnlineDiarizer {
    pub fn new(embedder: Arc<Embedder>, cfg: OnlineConfig) -> Self {
        Self {
            embedder,
            cfg,
            embeddings: Vec::new(),
            window_utt: Vec::new(),
            windows: Vec::new(),
            window_speaker: Vec::new(),
            window_first: Vec::new(),
            spans: Vec::new(),
            utterances: Vec::new(),
            centroids: Vec::new(),
            next_speaker: 0,
            audio_since_recluster: 0,
            segmenter: None,
            overlap_spans: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Also detects overlapped speech in each utterance (pyannote segmentation-3.0).
    pub fn with_segmentation(mut self, model: Arc<SegmentationModel>) -> Self {
        self.segmenter = Some(model);
        self
    }

    /// Absolute ranges of `pcm` (starting at `start`) where two speakers are active.
    fn overlap_ranges(&self, start: u64, pcm: &[f32]) -> Result<Vec<(u64, u64)>> {
        let Some(seg) = &self.segmenter else {
            return Ok(Vec::new());
        };
        let count = seg.count(pcm, 2.5)?;
        let half = (FRAME_STEP / 2) as u64;
        let end = start + pcm.len() as u64;
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (i, &c) in count.counts.iter().enumerate() {
            if c < 2 {
                continue;
            }
            let centre = start + SpeakerCount::centre(i);
            let (a, b) = (centre.saturating_sub(half), (centre + half).min(end));
            if a >= end {
                break;
            }
            match out.last_mut() {
                Some(last) if a <= last.1 + 1 => last.1 = b,
                _ => out.push((a, b)),
            }
        }
        Ok(out)
    }

    /// Windows of `pcm`: (start, end) relative to `pcm`, and the embedding.
    fn embed_windows(&self, pcm: &[f32]) -> Result<Vec<Window>> {
        let sr = SAMPLE_RATE as f32;
        let (win, step) = (
            (self.cfg.window_s * sr) as usize,
            (self.cfg.step_s * sr) as usize,
        );
        if pcm.len() < (self.cfg.min_embed_s * sr) as usize {
            return Ok(Vec::new());
        }
        let mut starts = Vec::new();
        if pcm.len() <= win {
            starts.push(0);
        } else {
            let mut s = 0;
            while s + win < pcm.len() {
                starts.push(s);
                s += step;
            }
            starts.push(pcm.len() - win);
        }
        let mut out = Vec::new();
        for s in starts {
            let e = (s + win).min(pcm.len());
            if let Some(v) = self.embedder.embed(&pcm[s..e])? {
                out.push(((s, e), v));
            }
        }
        Ok(out)
    }

    fn nearest_speaker(&self, v: &[f32]) -> Option<(u32, f32)> {
        self.centroids
            .iter()
            .map(|(id, sum, _)| {
                let n = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let c: Vec<f32> = sum.iter().map(|x| x / n).collect();
                (*id, distance(&c, v))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn add_to_centroid(&mut self, speaker: u32, v: &[f32], n: usize) {
        match self.centroids.iter_mut().find(|c| c.0 == speaker) {
            Some((_, sum, count)) => {
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += x * n as f32;
                }
                *count += n;
            }
            None => self
                .centroids
                .push((speaker, v.iter().map(|x| x * n as f32).collect(), n)),
        }
    }

    /// Adds a finished utterance (16 kHz mono audio starting at sample `start` of the
    /// source's timeline) and returns its speaker id now, plus any revisions from a
    /// periodic re-clustering: (utterance id, new speaker).
    pub fn add(&mut self, id: u64, start: u64, pcm: &[f32]) -> Result<(u32, Vec<(u64, u32)>)> {
        let windows = self.embed_windows(pcm)?;
        let speaker = if windows.is_empty() {
            match self.utterances.last().map(|u| u.speaker) {
                Some(s) => s,
                None => self.fresh_speaker(),
            }
        } else {
            let dim = windows[0].1.len();
            let mut mean = vec![0f32; dim];
            for (_, w) in &windows {
                for (m, x) in mean.iter_mut().zip(w) {
                    *m += x;
                }
            }
            let n = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            mean.iter_mut().for_each(|x| *x /= n);
            let s = match self.nearest_speaker(&mean) {
                Some((s, d)) if d < self.cfg.assign_threshold => s,
                _ => self.fresh_speaker(),
            };
            self.add_to_centroid(s, &mean, windows.len());
            s
        };
        let u = self.utterances.len();
        let first = self.embeddings.len();
        for ((a, b), w) in windows {
            // A window joins its own nearest speaker when close enough (a speaker change
            // inside the utterance), else the utterance's.
            let ws = match self.nearest_speaker(&w) {
                Some((s, d)) if d < self.cfg.assign_threshold => s,
                _ => speaker,
            };
            self.windows.push((start + a as u64, start + b as u64));
            self.window_speaker.push(ws);
            self.window_first.push(ws);
            self.embeddings.push(w);
            self.window_utt.push(u);
        }
        self.spans.push((start, start + pcm.len() as u64));
        let overlaps = self.overlap_ranges(start, pcm)?;
        self.overlap_spans.push(overlaps);
        self.index.insert(id, u);
        self.utterances.push(Utterance {
            id,
            speaker,
            windows: (first..self.embeddings.len()).collect(),
        });
        self.audio_since_recluster += pcm.len() as u64;
        let revisions = if self.audio_since_recluster as f32
            >= self.cfg.recluster_every_s * SAMPLE_RATE as f32
        {
            self.recluster()
        } else {
            Vec::new()
        };
        Ok((speaker, revisions))
    }

    fn fresh_speaker(&mut self) -> u32 {
        self.next_speaker += 1;
        self.next_speaker - 1
    }

    /// The current speaker of every utterance: (utterance id, speaker).
    pub fn speakers(&self) -> Vec<(u64, u32)> {
        self.utterances.iter().map(|u| (u.id, u.speaker)).collect()
    }

    /// Current speakers of one utterance: majority, turns from its windows, overlaps.
    pub fn utterance(&self, id: u64) -> Option<UtteranceSpeakers> {
        let &u = self.index.get(&id)?;
        let utt = &self.utterances[u];
        let span = self.spans[u];
        let turns: Vec<(u64, u64, u32)> = if utt.windows.is_empty() {
            vec![(span.0, span.1, utt.speaker)]
        } else {
            let e = crate::pipeline::Embedded {
                windows: utt.windows.iter().map(|&w| self.windows[w]).collect(),
                embeddings: Vec::new(),
                regions: vec![span],
            };
            let labels: Vec<usize> = utt
                .windows
                .iter()
                .map(|&w| self.window_speaker[w] as usize)
                .collect();
            crate::pipeline::label_frames(&e, &labels)
                .into_iter()
                .map(|t| {
                    let spk: u32 = t
                        .speaker
                        .trim_start_matches("spk")
                        .parse()
                        .unwrap_or(utt.speaker);
                    (
                        (t.start * SAMPLE_RATE as f64).round() as u64,
                        (t.end * SAMPLE_RATE as f64).round() as u64,
                        spk,
                    )
                })
                .collect()
        };
        let overlaps = self.second_speakers(&self.overlap_spans[u], &turns, utt.speaker);
        Some(UtteranceSpeakers {
            speaker: utt.speaker,
            turns,
            overlaps,
        })
    }

    /// For each 10 ms of the overlap ranges: the nearest window in time whose speaker differs
    /// from the primary speaker there.
    fn second_speakers(
        &self,
        ranges: &[(u64, u64)],
        turns: &[(u64, u64, u32)],
        fallback: u32,
    ) -> Vec<(u64, u64, u32)> {
        let hop = (SAMPLE_RATE / 100) as u64;
        let reach = (OVERLAP_REACH_S * SAMPLE_RATE as f64) as u64;
        let centres: Vec<u64> = self.windows.iter().map(|(a, b)| (a + b) / 2).collect();
        let mut out: Vec<(u64, u64, u32)> = Vec::new();
        for &(a, b) in ranges {
            let mut f = a;
            while f < b {
                let c = f + hop / 2;
                let primary = turns
                    .iter()
                    .find(|t| t.0 <= c && c < t.1)
                    .map_or(fallback, |t| t.2);
                let j = centres.partition_point(|&x| x < c);
                let mut best: Option<(u64, u32)> = None;
                for k in (0..j).rev() {
                    if centres[k] + reach < c {
                        break;
                    }
                    if self.window_speaker[k] != primary {
                        best = Some((c - centres[k], self.window_speaker[k]));
                        break;
                    }
                }
                for (&centre, &speaker) in centres[j..].iter().zip(&self.window_speaker[j..]) {
                    if centre > c + reach {
                        break;
                    }
                    if speaker != primary {
                        let d = centre - c;
                        if best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, speaker));
                        }
                        break;
                    }
                }
                if let Some((_, second)) = best {
                    let e = (f + hop).min(b);
                    match out.last_mut() {
                        Some(last) if last.2 == second && last.1 == f => last.1 = e,
                        _ => out.push((f, e, second)),
                    }
                }
                f += hop;
            }
        }
        out
    }

    /// Speaker turns over all utterances at 10 ms resolution, from the windows' speakers:
    /// as now (`first == false`) or as first emitted.
    pub fn turns(&self, first: bool) -> Vec<crate::der::Turn> {
        let labels: Vec<usize> = if first {
            &self.window_first
        } else {
            &self.window_speaker
        }
        .iter()
        .map(|&s| s as usize)
        .collect();
        let e = crate::pipeline::Embedded {
            windows: self.windows.clone(),
            embeddings: Vec::new(),
            regions: self.spans.clone(),
        };
        crate::pipeline::label_frames(&e, &labels)
    }

    /// Re-clusters everything so far; returns the utterances whose speakers changed (their
    /// majority speaker or any window's speaker) with their majority speaker now.
    pub fn recluster(&mut self) -> Vec<(u64, u32)> {
        self.audio_since_recluster = 0;
        let n = self.embeddings.len();
        if n < 2 {
            return Vec::new();
        }
        let merges = average_linkage(&self.embeddings);
        let labels = cut(n, &merges, self.cfg.cluster_threshold, None);
        let labels = absorb_small(
            &self.embeddings,
            &labels,
            min_cluster_size(n),
            self.cfg.absorb_max,
        );
        let k = labels.iter().copied().max().map_or(0, |m| m + 1);
        // Match new clusters to current ids by shared windows.
        let ids: Vec<u32> = {
            let mut v: Vec<u32> = self.utterances.iter().map(|u| u.speaker).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let mut overlap = vec![vec![0.0f64; ids.len()]; k];
        for (w, &l) in labels.iter().enumerate() {
            let cur = self.utterances[self.window_utt[w]].speaker;
            if let Ok(j) = ids.binary_search(&cur) {
                overlap[l][j] += 1.0;
            }
        }
        let assignment = crate::der::assign_max(&overlap);
        let mut cluster_id: Vec<u32> = assignment
            .iter()
            .map(|a| a.map_or(u32::MAX, |j| ids[j]))
            .collect();
        for c in cluster_id.iter_mut() {
            if *c == u32::MAX {
                *c = self.fresh_speaker();
            }
        }
        let old_windows = self.window_speaker.clone();
        // Each utterance takes the majority cluster of its windows.
        let mut changed_set: HashSet<usize> = HashSet::new();
        for (ui, u) in self.utterances.iter_mut().enumerate() {
            if u.windows.is_empty() {
                continue;
            }
            let mut votes = vec![0usize; k];
            for &w in &u.windows {
                votes[labels[w]] += 1;
            }
            let best = (0..k).max_by_key(|&c| votes[c]).expect("k > 0");
            let s = cluster_id[best];
            if s != u.speaker {
                u.speaker = s;
                changed_set.insert(ui);
            }
        }
        // Utterances without windows follow the previous utterance.
        for i in 1..self.utterances.len() {
            if self.utterances[i].windows.is_empty() {
                let prev = self.utterances[i - 1].speaker;
                if self.utterances[i].speaker != prev {
                    self.utterances[i].speaker = prev;
                    changed_set.insert(i);
                }
            }
        }
        for (w, l) in labels.iter().enumerate() {
            self.window_speaker[w] = cluster_id[*l];
            if self.window_speaker[w] != old_windows[w] {
                changed_set.insert(self.window_utt[w]);
            }
        }
        let mut changed_idx: Vec<usize> = changed_set.into_iter().collect();
        changed_idx.sort_unstable();
        let changed: Vec<(u64, u32)> = changed_idx
            .into_iter()
            .map(|i| (self.utterances[i].id, self.utterances[i].speaker))
            .collect();
        // Centroids follow the new clustering.
        self.centroids.clear();
        for w in 0..n {
            let s = cluster_id[labels[w]];
            let v = self.embeddings[w].clone();
            self.add_to_centroid(s, &v, 1);
        }
        changed
    }
}
