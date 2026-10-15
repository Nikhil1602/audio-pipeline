//! Streaming transcription policy: what to decode when, independent of any engine.
//!
//! [`StreamCore`] is single-threaded and deterministic. It takes the processing stage's
//! 16 kHz audio and VAD events, hands out [`Job`]s, and turns engine results into
//! [`Output`]s. A runner thread executes the jobs on a recognizer.
//!
//! **Policy.**
//! * While a speech segment is open, a **partial** decodes the segment so far, at most every
//!   `partial_interval_ms` of new audio. Only the newest partial per source matters; it is
//!   never queued behind another.
//! * When the segment ends, a **final** decodes it. Finals are never dropped and run before
//!   partials.
//! * Segments are cut at `max_segment_ms` (Whisper's window is 30 s), at the quietest
//!   100 ms in the last `split_search_ms`, even while speech continues.
//! * **Mic finals** are held until the system stream has moved `bleed::WINDOW_MS` past them,
//!   then checked against nearby system text with [`crate::bleed`]. A bleed match becomes
//!   [`Output::Dropped`] (kept for diagnostics, not shown as a transcript).
//!
//! Positions are 16 kHz sample indices on each source's processed timeline. Mic and system
//! timelines are aligned by the processing stage to within the reference alignment
//! (milliseconds), which is negligible against the ±4 s bleed window.

use std::collections::VecDeque;

use crate::bleed::{self, BleedMatch, SystemText};
use crate::lang::Language;
use crate::{Segment, Transcript};

pub const RATE: u64 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    Mic,
    System,
}

impl Source {
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    pub partial_interval_ms: u64,
    /// No partial before the open segment has this much audio.
    pub min_partial_ms: u64,
    pub max_segment_ms: u64,
    pub split_search_ms: u64,
    /// Audio kept per source for decoding (must exceed `max_segment_ms` plus queueing delay).
    pub keep_audio_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            partial_interval_ms: 1_000,
            min_partial_ms: 800,
            max_segment_ms: 28_000,
            split_search_ms: 5_000,
            keep_audio_ms: 120_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobKind {
    Partial,
    Final { echo_unverified: bool },
}

/// One decoding request: `start..end` of `source`'s audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job {
    pub source: Source,
    /// Utterance id, unique per source; partials and the final of one utterance share it.
    pub id: u64,
    pub start: u64,
    pub end: u64,
    pub kind: JobKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    Partial {
        source: Source,
        id: u64,
        start: u64,
        end: u64,
        text: String,
        language: Language,
    },
    Final {
        source: Source,
        id: u64,
        start: u64,
        end: u64,
        text: String,
        language: Language,
        language_probs: Vec<(Language, f32)>,
        /// Segment timestamps converted to absolute 16 kHz sample positions.
        segments: Vec<Segment>,
        /// Word timings, absolute like `segments` (empty if the engine gives none).
        words: Vec<Segment>,
        echo_unverified: bool,
        /// Speaker id within this source (diarization), if a speaker tracker is set. Later
        /// re-clustering may change it ([`crate::runner::AsrEvent::SpeakerRevision`]).
        speaker: Option<u32>,
        /// The text split where the speaker changes (one part when it does not; empty
        /// without a speaker tracker).
        parts: Vec<crate::speakers::SpeakerPart>,
        /// Where a second speaker talks at the same time (needs overlap detection).
        overlaps: Vec<crate::speakers::SpeakerSpan>,
    },
    /// A mic final judged to be far-end bleed.
    Dropped {
        source: Source,
        id: u64,
        start: u64,
        end: u64,
        text: String,
        reason: BleedMatch,
    },
}

#[derive(Debug, Default)]
struct Lane {
    audio: VecDeque<f32>,
    /// Sample index of `audio[0]`.
    base: u64,
    open: Option<Open>,
    next_id: u64,
    finals: VecDeque<Job>,
    partial_due: bool,
}

#[derive(Debug, Clone, Copy)]
struct Open {
    id: u64,
    start: u64,
    last_partial_end: u64,
}

impl Lane {
    fn end(&self) -> u64 {
        self.base + self.audio.len() as u64
    }
}

/// A mic final waiting for the bleed check.
#[derive(Debug)]
struct HeldFinal {
    job: Job,
    transcript: Transcript,
}

#[derive(Debug)]
pub struct StreamCore {
    cfg: StreamConfig,
    lanes: [Lane; 2],
    held: VecDeque<HeldFinal>,
    /// Recent system finals (start, end, text) for the bleed check.
    system_text: VecDeque<(u64, u64, String)>,
    /// Set by [`StreamCore::finish`]: finals no longer wait for their audio.
    closing: bool,
}

fn ms(ms: u64) -> u64 {
    ms * RATE / 1000
}

impl StreamCore {
    pub fn new(cfg: StreamConfig) -> Self {
        Self {
            cfg,
            lanes: Default::default(),
            held: VecDeque::new(),
            system_text: VecDeque::new(),
            closing: false,
        }
    }

    /// Appends processed audio. A gap (dropped chunk) is filled with silence.
    pub fn push_audio(&mut self, source: Source, first_sample: u64, samples: &[f32]) {
        let keep = ms(self.cfg.keep_audio_ms);
        let lane = &mut self.lanes[source.idx()];
        if lane.audio.is_empty() && lane.base == 0 {
            lane.base = first_sample;
        }
        if first_sample > lane.end() {
            let gap = (first_sample - lane.end()) as usize;
            lane.audio.extend(std::iter::repeat_n(0.0, gap));
        }
        let skip = lane.end().saturating_sub(first_sample) as usize;
        lane.audio.extend(samples.iter().skip(skip).copied());
        // Keep enough history for the oldest outstanding job and the open segment.
        let mut keep_from = lane.end().saturating_sub(keep);
        if let Some(o) = lane.open {
            keep_from = keep_from.min(o.start);
        }
        if let Some(j) = lane.finals.front() {
            keep_from = keep_from.min(j.start);
        }
        if keep_from > lane.base {
            let n = ((keep_from - lane.base) as usize).min(lane.audio.len());
            lane.audio.drain(..n);
            lane.base += n as u64;
        }
        if lane.open.is_some() {
            self.maybe_split(source);
            let lane = &mut self.lanes[source.idx()];
            if let Some(o) = lane.open
                && lane.end().saturating_sub(o.start) >= ms(self.cfg.min_partial_ms)
                && lane.end().saturating_sub(o.last_partial_end) >= ms(self.cfg.partial_interval_ms)
            {
                lane.partial_due = true;
            }
        }
    }

    pub fn speech_start(&mut self, source: Source, start: u64) {
        let lane = &mut self.lanes[source.idx()];
        let id = lane.next_id;
        lane.next_id += 1;
        lane.open = Some(Open {
            id,
            start,
            last_partial_end: start,
        });
        lane.partial_due = false;
    }

    pub fn speech_discarded(&mut self, source: Source) {
        let lane = &mut self.lanes[source.idx()];
        lane.open = None;
        lane.partial_due = false;
    }

    /// The open segment ended at `end` (VAD padding included).
    pub fn speech_end(&mut self, source: Source, start: u64, end: u64, echo_unverified: bool) {
        let lane = &mut self.lanes[source.idx()];
        let (id, start) = match lane.open.take() {
            // A split may already have finalized the beginning of this utterance.
            Some(o) => (o.id, o.start.max(start)),
            None => {
                let id = lane.next_id;
                lane.next_id += 1;
                (id, start)
            }
        };
        lane.partial_due = false;
        if end > start {
            lane.finals.push_back(Job {
                source,
                id,
                start,
                end,
                kind: JobKind::Final { echo_unverified },
            });
        }
    }

    /// Cuts an open segment that reached `max_segment_ms` at its quietest point.
    fn maybe_split(&mut self, source: Source) {
        let max = ms(self.cfg.max_segment_ms);
        let lane = &mut self.lanes[source.idx()];
        let Some(o) = lane.open else { return };
        if lane.end().saturating_sub(o.start) < max {
            return;
        }
        let window = ms(100) as usize;
        let lo = o.start + max - ms(self.cfg.split_search_ms).min(max);
        let hi = o.start + max - window as u64;
        let mut best = (f32::INFINITY, hi);
        let mut t = lo;
        while t <= hi {
            let i = (t - lane.base) as usize;
            let e: f32 = lane.audio.range(i..i + window).map(|v| v * v).sum();
            if e < best.0 {
                best = (e, t);
            }
            t += window as u64 / 2;
        }
        let cut = best.1 + window as u64 / 2;
        lane.finals.push_back(Job {
            source,
            id: o.id,
            start: o.start,
            end: cut,
            kind: JobKind::Final {
                echo_unverified: false,
            },
        });
        // The rest of the utterance continues as a new id.
        let id = lane.next_id;
        lane.next_id += 1;
        lane.open = Some(Open {
            id,
            start: cut,
            last_partial_end: cut,
        });
    }

    /// The next job: finals first (oldest first), then the newest partial per source.
    ///
    /// A final is handed out only once its audio has arrived: VAD events can overtake the
    /// audio, which travels in chunks. After [`StreamCore::finish`] finals no longer wait.
    pub fn next_job(&mut self) -> Option<Job> {
        let closing = self.closing;
        let oldest_final = (0..2)
            .filter_map(|i| {
                let lane = &self.lanes[i];
                lane.finals
                    .front()
                    .filter(|j| closing || lane.end() >= j.end)
                    .map(|j| (j.end, i))
            })
            .min()
            .map(|(_, i)| i);
        if let Some(i) = oldest_final {
            return self.lanes[i].finals.pop_front();
        }
        for (i, source) in [(0, Source::Mic), (1, Source::System)] {
            let max = ms(self.cfg.max_segment_ms);
            let lane = &mut self.lanes[i];
            if !lane.partial_due {
                continue;
            }
            lane.partial_due = false;
            let Some(o) = lane.open.as_mut() else {
                continue;
            };
            let end = lane.base + lane.audio.len() as u64;
            o.last_partial_end = end;
            return Some(Job {
                source,
                id: o.id,
                start: o.start.max(end.saturating_sub(max)),
                end,
                kind: JobKind::Partial,
            });
        }
        None
    }

    /// The audio of a job (silence where history was already discarded).
    pub fn audio(&self, job: &Job) -> Vec<f32> {
        let lane = &self.lanes[job.source.idx()];
        (job.start..job.end)
            .map(|t| {
                t.checked_sub(lane.base)
                    .and_then(|i| lane.audio.get(i as usize).copied())
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Turns a job's transcript into outputs (plus any held mic finals now decidable).
    pub fn complete(&mut self, job: Job, transcript: Transcript) -> Vec<Output> {
        let mut out = Vec::new();
        match job.kind {
            JobKind::Partial => {
                // Stale if the utterance already ended or moved on.
                let current = self.lanes[job.source.idx()].open.map(|o| o.id);
                if current == Some(job.id) && !transcript.text.is_empty() {
                    out.push(Output::Partial {
                        source: job.source,
                        id: job.id,
                        start: job.start,
                        end: job.end,
                        text: transcript.text,
                        language: transcript.language,
                    });
                }
            }
            JobKind::Final { .. } if job.source == Source::Mic => {
                self.held.push_back(HeldFinal { job, transcript });
            }
            JobKind::Final { echo_unverified } => {
                if !transcript.text.is_empty() {
                    self.system_text
                        .push_back((job.start, job.end, transcript.text.clone()));
                }
                out.push(final_output(job, transcript, echo_unverified));
            }
        }
        out.extend(self.release_held(false));
        // System text older than any mic final could need is no longer useful.
        let horizon = self.lanes[Source::Mic.idx()]
            .base
            .saturating_sub(ms(bleed::WINDOW_MS as u64) * 2);
        while self.system_text.front().is_some_and(|t| t.1 < horizon) {
            self.system_text.pop_front();
        }
        out
    }

    /// Releases held mic finals whose bleed check can be decided (all of them if `force`).
    pub fn release_held(&mut self, force: bool) -> Vec<Output> {
        let window = ms(bleed::WINDOW_MS as u64);
        let sys = &self.lanes[Source::System.idx()];
        let sys_end = sys.end();
        // System audio that may still turn into text near a held final: an open segment, or
        // finals not decoded yet.
        let sys_pending_from = sys
            .open
            .map(|o| o.start)
            .into_iter()
            .chain(sys.finals.iter().map(|j| j.start))
            .min();
        let mut out = Vec::new();
        while let Some(h) = self.held.front() {
            let decidable = force
                || (sys_end >= h.job.end + window
                    && sys_pending_from.is_none_or(|s| s > h.job.end + window));
            if !decidable {
                break;
            }
            let h = self.held.pop_front().expect("front exists");
            let JobKind::Final { echo_unverified } = h.job.kind else {
                unreachable!("only finals are held")
            };
            let to_ms = |s: u64| (s * 1000 / RATE) as i64;
            let nearby: Vec<SystemText<'_>> = self
                .system_text
                .iter()
                .map(|(s, e, t)| SystemText {
                    start_ms: to_ms(*s),
                    end_ms: to_ms(*e),
                    text: t,
                })
                .collect();
            match bleed::is_bleed(
                &h.transcript.text,
                to_ms(h.job.start),
                to_ms(h.job.end),
                echo_unverified,
                &nearby,
            ) {
                Some(reason) => out.push(Output::Dropped {
                    source: h.job.source,
                    id: h.job.id,
                    start: h.job.start,
                    end: h.job.end,
                    text: h.transcript.text,
                    reason,
                }),
                // A flagged utterance can begin with leaked far-end speech and continue with the
                // local talker (the gate engaged mid-utterance). Check each timed sub-segment.
                None if echo_unverified && h.transcript.segments.len() > 1 => {
                    out.extend(split_bleed(h.job, h.transcript, &nearby));
                }
                None => out.push(final_output(h.job, h.transcript, echo_unverified)),
            }
        }
        out
    }

    /// Ends the session: closes open segments as finals; queued finals stop waiting for audio.
    pub fn finish(&mut self) {
        self.closing = true;
        for source in [Source::Mic, Source::System] {
            let lane = &self.lanes[source.idx()];
            if let Some(o) = lane.open {
                let end = lane.end();
                self.speech_end(source, o.start, end, false);
            }
        }
    }

    /// Finals queued for `source` and not yet handed out.
    pub fn queued_finals(&self, source: Source) -> usize {
        self.lanes[source.idx()].finals.len()
    }

    /// Jobs still waiting (for the runner's shutdown).
    pub fn has_work(&self) -> bool {
        self.lanes
            .iter()
            .any(|l| !l.finals.is_empty() || l.partial_due)
    }
}

/// Drops the sub-segments of a flagged mic utterance that are far-end bleed; returns the
/// dropped parts (for diagnostics) and the final made of what remains.
fn split_bleed(job: Job, mut t: Transcript, nearby: &[SystemText<'_>]) -> Vec<Output> {
    let base_ms = (job.start * 1000 / RATE) as i64;
    let words = std::mem::take(&mut t.words);
    let mut kept = Vec::new();
    let mut out = Vec::new();
    for seg in t.segments {
        let (a, b) = (base_ms + seg.start_ms, base_ms + seg.end_ms);
        match bleed::is_bleed(&seg.text, a, b, true, nearby) {
            Some(reason) => out.push(Output::Dropped {
                source: job.source,
                id: job.id,
                start: (a.max(0) as u64) * RATE / 1000,
                end: (b.max(0) as u64) * RATE / 1000,
                text: seg.text,
                reason,
            }),
            None => kept.push(seg),
        }
    }
    if !kept.is_empty() {
        let text = kept
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        // The final now starts where the kept speech starts.
        let kept_start = (base_ms + kept[0].start_ms).max(0) as u64 * RATE / 1000;
        let shift = kept_start as i64 * 1000 / RATE as i64 - base_ms;
        let kept_words: Vec<Segment> = words
            .into_iter()
            .filter(|w| {
                let mid = (w.start_ms + w.end_ms) / 2;
                kept.iter()
                    .any(|s: &Segment| s.start_ms <= mid && mid <= s.end_ms)
            })
            .map(|w| Segment {
                start_ms: w.start_ms - shift,
                end_ms: w.end_ms - shift,
                ..w
            })
            .collect();
        let rebased = kept
            .into_iter()
            .map(|s| Segment {
                start_ms: s.start_ms - (kept_start as i64 * 1000 / RATE as i64 - base_ms),
                end_ms: s.end_ms - (kept_start as i64 * 1000 / RATE as i64 - base_ms),
                ..s
            })
            .collect::<Vec<_>>();
        let kept = rebased;
        out.push(final_output(
            Job {
                start: kept_start.max(job.start),
                ..job
            },
            Transcript {
                text,
                segments: kept,
                words: kept_words,
                ..t
            },
            true,
        ));
    }
    out
}

fn final_output(job: Job, t: Transcript, echo_unverified: bool) -> Output {
    let base_ms = (job.start * 1000 / RATE) as i64;
    let absolute = |v: Vec<Segment>| -> Vec<Segment> {
        v.into_iter()
            .map(|s| Segment {
                start_ms: base_ms + s.start_ms,
                end_ms: base_ms + s.end_ms,
                ..s
            })
            .collect()
    };
    Output::Final {
        source: job.source,
        id: job.id,
        start: job.start,
        end: job.end,
        text: t.text,
        language: t.language,
        language_probs: t.language_probs,
        segments: absolute(t.segments),
        words: absolute(t.words),
        echo_unverified,
        speaker: None,
        parts: Vec::new(),
        overlaps: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(text: &str) -> Transcript {
        Transcript {
            text: text.into(),
            language: Language::English,
            language_probs: vec![],
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 500,
                text: text.into(),
                no_speech_prob: 0.0,
            }],
            words: vec![],
        }
    }

    /// Feeds `secs` of 10 ms frames of constant `level` to a source starting at `from`.
    fn feed(c: &mut StreamCore, s: Source, from: u64, secs: f64, level: f32) -> u64 {
        let n = (secs * 100.0) as u64;
        for k in 0..n {
            c.push_audio(s, from + k * 160, &[level; 160]);
        }
        from + n * 160
    }

    #[test]
    fn partials_then_final_in_order() {
        let mut c = StreamCore::new(StreamConfig::default());
        c.speech_start(Source::System, 0);
        let mut t = feed(&mut c, Source::System, 0, 0.5, 0.1);
        assert_eq!(c.next_job(), None, "no partial before 0.8 s");
        t = feed(&mut c, Source::System, t, 0.6, 0.1);
        let p = c.next_job().expect("partial");
        assert_eq!(p.kind, JobKind::Partial);
        assert_eq!((p.start, p.end), (0, t));
        assert_eq!(c.next_job(), None, "next partial only after another second");
        let out = c.complete(p, tr("hello"));
        assert!(matches!(out.as_slice(), [Output::Partial { text, .. }] if text == "hello"));
        c.speech_end(Source::System, 0, t, false);
        let f = c.next_job().expect("final");
        assert_eq!(
            f.kind,
            JobKind::Final {
                echo_unverified: false
            }
        );
        assert_eq!(f.id, p.id);
        let out = c.complete(f, tr("hello world"));
        match &out[..] {
            [Output::Final { text, segments, .. }] => {
                assert_eq!(text, "hello world");
                assert_eq!(segments[0].end_ms, 500);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn finals_wait_for_their_audio_until_finish() {
        let mut c = StreamCore::new(StreamConfig::default());
        c.speech_start(Source::System, 0);
        feed(&mut c, Source::System, 0, 1.0, 0.1);
        // The VAD reports the end at 1.5 s, before the last 0.5 s of audio arrived.
        c.speech_end(Source::System, 0, ms(1_500), false);
        assert_eq!(c.next_job(), None, "waits for audio up to 1.5 s");
        feed(&mut c, Source::System, ms(1_000), 0.5, 0.1);
        assert!(matches!(
            c.next_job(),
            Some(Job {
                kind: JobKind::Final { .. },
                ..
            })
        ));
        // At shutdown, a final whose audio never came is still handed out.
        c.speech_end(Source::System, ms(2_000), ms(3_000), false);
        assert_eq!(c.next_job(), None);
        c.finish();
        assert!(c.next_job().is_some());
    }

    #[test]
    fn stale_partials_are_not_emitted() {
        let mut c = StreamCore::new(StreamConfig::default());
        c.speech_start(Source::System, 0);
        let t = feed(&mut c, Source::System, 0, 1.2, 0.1);
        let p = c.next_job().unwrap();
        c.speech_end(Source::System, 0, t, false);
        assert!(c.complete(p, tr("late")).is_empty());
    }

    #[test]
    fn long_speech_is_split_at_the_quietest_point() {
        let mut c = StreamCore::new(StreamConfig::default());
        c.speech_start(Source::System, 0);
        let mut t = feed(&mut c, Source::System, 0, 25.0, 0.2);
        t = feed(&mut c, Source::System, t, 0.3, 0.0); // a short pause at 25.0-25.3 s
        feed(&mut c, Source::System, t, 3.5, 0.2);
        let mut jobs = Vec::new();
        while let Some(j) = c.next_job() {
            jobs.push(j);
        }
        let f = jobs
            .iter()
            .find(|j| matches!(j.kind, JobKind::Final { .. }))
            .expect("split final");
        let cut_s = f.end as f64 / RATE as f64;
        assert!((25.0..25.3).contains(&cut_s), "cut at {cut_s} s");
        assert!(f.end - f.start <= ms(28_000));
    }

    #[test]
    fn mic_finals_wait_for_system_and_bleed_is_dropped() {
        let mut c = StreamCore::new(StreamConfig::default());
        // System says something at 1-4 s.
        c.speech_start(Source::System, ms(1_000));
        feed(&mut c, Source::System, 0, 4.0, 0.1);
        c.speech_end(Source::System, ms(1_000), ms(4_000), false);
        // The mic picks it up (flagged: echo gate undecided) and also says its own thing later.
        feed(&mut c, Source::Mic, 0, 4.2, 0.1);
        c.speech_end(Source::Mic, ms(1_100), ms(4_100), true);
        // Oldest final first: the system one ends at 4.0 s, the mic one at 4.1 s.
        let sys = c.next_job().unwrap();
        assert_eq!(sys.source, Source::System, "oldest final first");
        let out = c.complete(sys, tr("Today we reviewed the project timeline."));
        assert_eq!(out.len(), 1, "system final: {out:?}");
        let mic_echo = c.next_job().unwrap();
        assert_eq!(mic_echo.source, Source::Mic);
        let out = c.complete(mic_echo, tr("today we review the project time line"));
        assert!(
            out.is_empty(),
            "mic final held until the system stream moves on: {out:?}"
        );
        // Once the system stream is 4 s past the mic final, the check runs.
        feed(&mut c, Source::System, ms(4_000), 4.5, 0.0);
        let out = c.release_held(false);
        assert!(
            matches!(out.as_slice(), [Output::Dropped { .. }]),
            "{out:?}"
        );
    }

    #[test]
    fn leaked_prefix_of_a_mixed_mic_utterance_is_removed() {
        let mut c = StreamCore::new(StreamConfig::default());
        feed(&mut c, Source::System, 0, 3.0, 0.1);
        c.speech_end(Source::System, 0, ms(3_000), false);
        let j = c.next_job().unwrap();
        c.complete(j, tr("Thanks for joining the weekly planning call."));
        feed(&mut c, Source::Mic, 0, 8.0, 0.1);
        c.speech_end(Source::Mic, ms(500), ms(8_000), true);
        let j = c.next_job().unwrap();
        let mixed = Transcript {
            text: "Thanks for joining the weekly planning call. Today we review the timeline."
                .into(),
            language: Language::English,
            language_probs: vec![],
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 2_500,
                    text: "Thanks for joining the weekly planning call.".into(),
                    no_speech_prob: 0.0,
                },
                Segment {
                    start_ms: 2_600,
                    end_ms: 7_000,
                    text: "Today we review the timeline.".into(),
                    no_speech_prob: 0.0,
                },
            ],
            words: vec![
                Segment {
                    start_ms: 100,
                    end_ms: 2_400,
                    text: "Thanks".into(),
                    no_speech_prob: 0.0,
                },
                Segment {
                    start_ms: 2_700,
                    end_ms: 6_900,
                    text: "Today".into(),
                    no_speech_prob: 0.0,
                },
            ],
        };
        assert!(c.complete(j, mixed).is_empty());
        feed(&mut c, Source::System, ms(3_000), 9.5, 0.0);
        let out = c.release_held(false);
        assert!(
            matches!(&out[0], Output::Dropped { text, .. } if text.starts_with("Thanks")),
            "{out:?}"
        );
        match &out[1] {
            Output::Final {
                text,
                segments,
                words,
                start,
                ..
            } => {
                assert_eq!(*start, ms(3_100), "starts at the kept speech");
                assert_eq!(words.len(), 1, "the dropped prefix's words go too");
                assert_eq!(words[0].start_ms, 500 + 2_700, "absolute time");
                assert_eq!(text, "Today we review the timeline.");
                assert_eq!(segments.len(), 1);
                assert_eq!(segments[0].start_ms, 500 + 2_600, "absolute time");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn real_mic_speech_is_released_as_final() {
        let mut c = StreamCore::new(StreamConfig::default());
        feed(&mut c, Source::Mic, 0, 3.0, 0.1);
        c.speech_end(Source::Mic, 0, ms(3_000), false);
        let j = c.next_job().unwrap();
        assert!(c.complete(j, tr("my own words")).is_empty());
        feed(&mut c, Source::System, 0, 7.5, 0.0);
        let out = c.release_held(false);
        assert!(matches!(out.as_slice(), [Output::Final { text, .. }] if text == "my own words"));
    }

    #[test]
    fn audio_history_is_trimmed_but_kept_for_open_segments() {
        let cfg = StreamConfig {
            keep_audio_ms: 2_000,
            ..Default::default()
        };
        let mut c = StreamCore::new(cfg);
        feed(&mut c, Source::System, 0, 5.0, 0.1);
        assert!(c.lanes[1].audio.len() as u64 <= ms(2_000));
        c.speech_start(Source::System, ms(5_000));
        feed(&mut c, Source::System, ms(5_000), 6.0, 0.1);
        assert_eq!(c.lanes[1].base, ms(5_000), "open segment's audio kept");
        let j = Job {
            source: Source::System,
            id: 0,
            start: ms(5_000),
            end: ms(5_010),
            kind: JobKind::Partial,
        };
        assert_eq!(c.audio(&j), vec![0.1; 160]);
    }
}
