//! The ASR thread: feeds [`StreamCore`] with audio and VAD events, executes its jobs on the
//! [`Router`], chooses languages (fixed or Auto with a session prior per source), and emits
//! [`AsrEvent`]s.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

use crate::auto::{AutoConfig, SessionLanguage};
use crate::lang::{Language, LanguageChoice};
use crate::router::{LanguageDetector, Router};
use crate::speakers::{SpeakerInfo, SpeakerPart, SpeakerSpan};
use crate::stream::{Job, JobKind, Output, RATE, Source, StreamConfig, StreamCore};

#[derive(Debug)]
pub enum AsrInput {
    Audio {
        source: Source,
        first_sample: u64,
        samples: Vec<f32>,
    },
    SpeechStart {
        source: Source,
        start: u64,
    },
    SpeechDiscarded {
        source: Source,
    },
    SpeechEnd {
        source: Source,
        start: u64,
        end: u64,
        echo_unverified: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AsrEvent {
    Transcript(Output),
    /// A job could not be transcribed (no engine for the language, or an engine error).
    Skipped {
        source: Source,
        id: u64,
        start: u64,
        end: u64,
        reason: String,
    },
    /// Timing of one decode, for latency and load monitoring.
    Timing {
        source: Source,
        id: u64,
        partial: bool,
        audio_ms: u64,
        /// Time spent in language identification and decoding.
        compute_ms: u64,
        /// From the moment the job became available to its result.
        latency_ms: u64,
    },
    /// Re-clustering changed the speakers of an already emitted final: its majority speaker,
    /// parts and overlaps as they are now.
    SpeakerRevision {
        source: Source,
        id: u64,
        speaker: u32,
        parts: Vec<SpeakerPart>,
        overlaps: Vec<SpeakerSpan>,
    },
}

/// Assigns speakers to final utterances, per source (implemented with sd-diar by the
/// engine, so this crate does not depend on it).
pub trait SpeakerTracker: Send {
    /// The speakers of a final utterance of `source` whose audio starts at sample `start`,
    /// and revisions of earlier utterances as (utterance id, speakers now).
    fn add(
        &mut self,
        source: Source,
        id: u64,
        start: u64,
        pcm: &[f32],
    ) -> (SpeakerInfo, Vec<(u64, SpeakerInfo)>);
    /// Final re-clustering when the session ends: revisions as (source, utterance id,
    /// speakers now).
    fn finish(&mut self) -> Vec<(Source, u64, SpeakerInfo)>;
}

/// What is needed to split an emitted final again when its speakers are revised.
struct EmittedText {
    text: String,
    start: u64,
    end: u64,
    words: Vec<crate::Segment>,
}

/// Creates a speaker tracker for a session.
pub type SpeakerFactory = Arc<dyn Fn() -> Box<dyn SpeakerTracker> + Send + Sync>;

#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub language: LanguageChoice,
    /// Languages Auto may choose from (default: all 15).
    pub allowed: Vec<Language>,
    pub stream: StreamConfig,
    pub auto: AutoConfig,
    /// Emit partial results (finals are always produced).
    pub partials: bool,
    /// Auto: spoken languages treated as one (default: Urdu is written as Hindi); see
    /// [`crate::auto::SAME_SPEECH`].
    pub same_speech: Vec<(Language, Language)>,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            language: LanguageChoice::Auto,
            allowed: Language::ALL.to_vec(),
            stream: StreamConfig::default(),
            auto: AutoConfig::default(),
            partials: true,
            same_speech: crate::auto::SAME_SPEECH.to_vec(),
        }
    }
}

pub struct AsrRunner {
    cfg: AsrConfig,
    router: Router,
    detector: Option<Arc<dyn LanguageDetector>>,
    core: StreamCore,
    sessions: [SessionLanguage; 2],
    /// When each source's newest audio arrived (for partial latency).
    available_at: [Option<Instant>; 2],
    speakers: Option<Box<dyn SpeakerTracker>>,
    /// Speakers per (source, utterance id), and the finals already emitted.
    speaker_of: HashMap<(usize, u64), SpeakerInfo>,
    emitted: HashMap<(usize, u64), EmittedText>,
    /// When each queued final became available, per source, in queue order (finals of one
    /// source are handed out in order, whether from a segment end or a long-speech split).
    final_ready: [VecDeque<Instant>; 2],
}

impl AsrRunner {
    pub fn new(
        cfg: AsrConfig,
        router: Router,
        detector: Option<Arc<dyn LanguageDetector>>,
    ) -> Self {
        Self {
            core: StreamCore::new(cfg.stream),
            sessions: [
                SessionLanguage::new(cfg.auto),
                SessionLanguage::new(cfg.auto),
            ],
            cfg,
            router,
            detector,
            available_at: [None, None],
            speakers: None,
            speaker_of: HashMap::new(),
            emitted: HashMap::new(),
            final_ready: [VecDeque::new(), VecDeque::new()],
        }
    }

    /// Assigns a speaker to every final ([`Output::Final::speaker`]).
    pub fn with_speakers(mut self, tracker: Box<dyn SpeakerTracker>) -> Self {
        self.speakers = Some(tracker);
        self
    }

    /// Fills in the speakers of a final as it is emitted.
    fn output(&mut self, mut o: Output) -> AsrEvent {
        if let Output::Final {
            source,
            id,
            start,
            end,
            text,
            words,
            speaker,
            parts,
            overlaps,
            ..
        } = &mut o
        {
            let key = (*source as usize, *id);
            if let Some(info) = self.speaker_of.get(&key) {
                *speaker = info.speaker;
                *parts = crate::speakers::split(text, *start, *end, words, info);
                *overlaps = within(&info.overlaps, *start, *end);
            }
            if self.speakers.is_some() {
                self.emitted.insert(
                    key,
                    EmittedText {
                        text: text.clone(),
                        start: *start,
                        end: *end,
                        words: words.clone(),
                    },
                );
            }
        }
        AsrEvent::Transcript(o)
    }

    fn revisions(
        &mut self,
        source: Source,
        changes: Vec<(u64, SpeakerInfo)>,
        emit: &mut impl FnMut(AsrEvent),
    ) {
        for (id, info) in changes {
            let key = (source as usize, id);
            if let (Some(e), Some(speaker)) = (self.emitted.get(&key), info.speaker) {
                let parts = crate::speakers::split(&e.text, e.start, e.end, &e.words, &info);
                let overlaps = within(&info.overlaps, e.start, e.end);
                let before = self.speaker_of.get(&key).map(|old| {
                    (
                        old.speaker,
                        crate::speakers::split(&e.text, e.start, e.end, &e.words, old),
                        within(&old.overlaps, e.start, e.end),
                    )
                });
                // Window labels can change without changing what is shown.
                if before != Some((Some(speaker), parts.clone(), overlaps.clone())) {
                    emit(AsrEvent::SpeakerRevision {
                        source,
                        id,
                        speaker,
                        parts,
                        overlaps,
                    });
                }
            }
            self.speaker_of.insert(key, info);
        }
    }

    /// Runs until `inputs` disconnects, then finishes all pending work. Events go to a
    /// channel; see [`AsrRunner::run_with`] for a callback.
    pub fn run(self, inputs: Receiver<AsrInput>, events: Sender<AsrEvent>) {
        self.run_with(inputs, move |e| {
            let _ = events.send(e);
        });
    }

    /// As [`AsrRunner::run`], delivering events to `emit`.
    pub fn run_with(mut self, inputs: Receiver<AsrInput>, mut emit: impl FnMut(AsrEvent)) {
        loop {
            // Wait for input when idle, and also when the queued work cannot run yet (a final
            // whose audio has not all arrived), instead of spinning.
            let idle = !self.core.has_work();
            let wait = if idle { 100 } else { 0 };
            match inputs.recv_timeout(Duration::from_millis(wait)) {
                Ok(i) => self.apply(i),
                Err(RecvTimeoutError::Timeout) => {}
                // No more input: what is queued is finished below (finals no longer wait for
                // audio that will not come, e.g. VAD padding past the last chunk).
                Err(RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(i) = inputs.try_recv() {
                self.apply(i);
            }
            for o in self.core.release_held(false) {
                let e = self.output(o);
                emit(e);
            }
            match self.core.next_job() {
                Some(job) => self.execute(job, &mut emit),
                None if !idle => match inputs.recv_timeout(Duration::from_millis(10)) {
                    Ok(i) => self.apply(i),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                },
                None => {}
            }
        }
        self.core.finish();
        while let Some(job) = self.core.next_job() {
            self.execute(job, &mut emit);
        }
        for o in self.core.release_held(true) {
            let e = self.output(o);
            emit(e);
        }
        if let Some(mut tracker) = self.speakers.take() {
            let mut by_source: [Vec<(u64, SpeakerInfo)>; 2] = [Vec::new(), Vec::new()];
            for (source, id, info) in tracker.finish() {
                by_source[source as usize].push((id, info));
            }
            let [mic, system] = by_source;
            self.revisions(Source::Mic, mic, &mut emit);
            self.revisions(Source::System, system, &mut emit);
        }
    }

    pub fn apply(&mut self, input: AsrInput) {
        let before = [Source::Mic, Source::System].map(|s| self.core.queued_finals(s));
        self.apply_inner(input);
        for (i, s) in [Source::Mic, Source::System].into_iter().enumerate() {
            for _ in before[i]..self.core.queued_finals(s) {
                self.final_ready[i].push_back(Instant::now());
            }
        }
    }

    fn apply_inner(&mut self, input: AsrInput) {
        match input {
            AsrInput::Audio {
                source,
                first_sample,
                samples,
            } => {
                self.core.push_audio(source, first_sample, &samples);
                self.available_at[source as usize] = Some(Instant::now());
            }
            AsrInput::SpeechStart { source, start } => self.core.speech_start(source, start),
            AsrInput::SpeechDiscarded { source } => self.core.speech_discarded(source),
            AsrInput::SpeechEnd {
                source,
                start,
                end,
                echo_unverified,
            } => {
                self.core.speech_end(source, start, end, echo_unverified);
            }
        }
    }

    /// Picks the decoding language for a job; returns it with the detection probabilities.
    fn language_for(
        &mut self,
        job: &Job,
        audio: &[f32],
    ) -> Result<(Language, Vec<(Language, f32)>), String> {
        let LanguageChoice::Auto = self.cfg.language else {
            let LanguageChoice::Fixed(l) = self.cfg.language else {
                unreachable!()
            };
            return Ok((l, Vec::new()));
        };
        let candidates: Vec<Language> = self
            .cfg
            .allowed
            .iter()
            .copied()
            .filter(|l| self.router.pick(*l).is_some())
            .collect();
        let written =
            |l: Language| crate::auto::same_speech_target(l, &self.cfg.same_speech, &candidates);
        let session = &mut self.sessions[job.source as usize];
        if job.kind == JobKind::Partial
            && let Some(l) = session.current()
        {
            return Ok((written(l), Vec::new()));
        }
        let detector = self
            .detector
            .as_ref()
            .ok_or("Auto language needs a language detector")?;
        let probs = detector
            .detect(audio, &candidates)
            .map_err(|e| e.to_string())?;
        if job.kind == JobKind::Partial {
            let top = probs.first().ok_or("no language detected")?.0;
            return Ok((written(top), probs));
        }
        let duration_s = audio.len() as f32 / RATE as f32;
        let d = session
            .decide(&probs, duration_s)
            .ok_or("no language detected")?;
        Ok((written(d.language), probs))
    }

    fn execute(&mut self, job: Job, emit: &mut impl FnMut(AsrEvent)) {
        let ready = match job.kind {
            JobKind::Partial => self.available_at[job.source as usize],
            JobKind::Final { .. } => self.final_ready[job.source as usize].pop_front(),
        }
        .unwrap_or_else(Instant::now);
        if job.kind == JobKind::Partial && !self.cfg.partials {
            return;
        }
        let audio = self.core.audio(&job);
        let t0 = Instant::now();
        let result = self.language_for(&job, &audio).and_then(|(lang, probs)| {
            self.router
                .transcribe(&audio, lang)
                .map(|mut t| {
                    if t.language_probs.is_empty() {
                        t.language_probs = probs;
                    }
                    t
                })
                .map_err(|e| e.to_string())
        });
        let compute_ms = t0.elapsed().as_millis() as u64;
        match result {
            Ok(t) => {
                if let (JobKind::Final { .. }, Some(tracker)) = (job.kind, self.speakers.as_mut()) {
                    let (info, changes) = tracker.add(job.source, job.id, job.start, &audio);
                    self.speaker_of.insert((job.source as usize, job.id), info);
                    self.revisions(job.source, changes, emit);
                }
                for o in self.core.complete(job, t) {
                    let e = self.output(o);
                    emit(e);
                }
            }
            Err(reason) => {
                emit(AsrEvent::Skipped {
                    source: job.source,
                    id: job.id,
                    start: job.start,
                    end: job.end,
                    reason,
                });
            }
        }
        emit(AsrEvent::Timing {
            source: job.source,
            id: job.id,
            partial: job.kind == JobKind::Partial,
            audio_ms: (job.end - job.start) * 1000 / RATE,
            compute_ms,
            latency_ms: ready.elapsed().as_millis() as u64,
        });
    }
}

/// The spans clipped to `start..end`.
fn within(spans: &[SpeakerSpan], start: u64, end: u64) -> Vec<SpeakerSpan> {
    spans
        .iter()
        .filter(|s| s.end > start && s.start < end)
        .map(|s| SpeakerSpan {
            start: s.start.max(start),
            end: s.end.min(end),
            speaker: s.speaker,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::Recognizer;
    use crate::{Result, Transcript};

    /// Writes the language it was asked for and the audio length.
    struct Echo;
    impl Recognizer for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn supports(&self, l: Language) -> bool {
            l != Language::Odia
        }
        fn transcribe(&self, pcm: &[f32], language: Language) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("{} {}", language.code(), pcm.len()),
                language,
                language_probs: vec![],
                segments: vec![],
                words: vec![],
            })
        }
    }
    struct Says(Language);
    impl LanguageDetector for Says {
        fn detect(&self, _: &[f32], allowed: &[Language]) -> Result<Vec<(Language, f32)>> {
            Ok(allowed
                .iter()
                .map(|&l| (l, if l == self.0 { 0.9 } else { 0.01 }))
                .collect())
        }
    }

    fn run_with(cfg: AsrConfig, detected: Language, inputs: Vec<AsrInput>) -> Vec<AsrEvent> {
        let router = Router::new(vec![Arc::new(Echo)]);
        let runner = AsrRunner::new(cfg, router, Some(Arc::new(Says(detected))));
        let (itx, irx) = crossbeam_channel::unbounded();
        let (etx, erx) = crossbeam_channel::unbounded();
        for i in inputs {
            itx.send(i).unwrap();
        }
        drop(itx);
        runner.run(irx, etx);
        erx.try_iter().collect()
    }

    fn speech(source: Source, secs: f64) -> Vec<AsrInput> {
        let n = (secs * RATE as f64) as u64;
        vec![
            AsrInput::SpeechStart { source, start: 0 },
            AsrInput::Audio {
                source,
                first_sample: 0,
                samples: vec![0.1; n as usize],
            },
            AsrInput::SpeechEnd {
                source,
                start: 0,
                end: n,
                echo_unverified: false,
            },
        ]
    }

    #[test]
    fn auto_detects_and_transcribes_a_final() {
        let ev = run_with(
            AsrConfig::default(),
            Language::Hindi,
            speech(Source::System, 3.0),
        );
        let finals: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                AsrEvent::Transcript(Output::Final {
                    text,
                    language,
                    language_probs,
                    ..
                }) => Some((text.clone(), *language, language_probs.len())),
                _ => None,
            })
            .collect();
        assert_eq!(finals.len(), 1, "{ev:?}");
        assert_eq!(finals[0].0, "hi 48000");
        assert_eq!(finals[0].1, Language::Hindi);
        // Odia is excluded from candidates: no engine supports it here.
        assert_eq!(finals[0].2, 14);
        assert!(
            ev.iter()
                .any(|e| matches!(e, AsrEvent::Timing { partial: false, .. }))
        );
    }

    #[test]
    fn auto_writes_detected_urdu_as_hindi_but_fixed_urdu_stays_urdu() {
        let langs = |ev: Vec<AsrEvent>| -> Vec<Language> {
            ev.into_iter()
                .filter_map(|e| match e {
                    AsrEvent::Transcript(Output::Final { language, .. }) => Some(language),
                    _ => None,
                })
                .collect()
        };
        let auto = run_with(
            AsrConfig::default(),
            Language::Urdu,
            speech(Source::System, 3.0),
        );
        assert_eq!(langs(auto), vec![Language::Hindi]);
        let fixed = run_with(
            AsrConfig {
                language: LanguageChoice::Fixed(Language::Urdu),
                ..Default::default()
            },
            Language::Hindi,
            speech(Source::System, 3.0),
        );
        assert_eq!(langs(fixed), vec![Language::Urdu]);
    }

    fn one(speaker: u32) -> SpeakerInfo {
        SpeakerInfo {
            speaker: Some(speaker),
            ..Default::default()
        }
    }

    struct Alternating;
    impl SpeakerTracker for Alternating {
        fn add(
            &mut self,
            _: Source,
            id: u64,
            _: u64,
            _: &[f32],
        ) -> (SpeakerInfo, Vec<(u64, SpeakerInfo)>) {
            // Speaker = id parity; the second utterance revises the first to speaker 7.
            let revisions = if id == 1 { vec![(0, one(7))] } else { vec![] };
            (one(id as u32 % 2), revisions)
        }
        fn finish(&mut self) -> Vec<(Source, u64, SpeakerInfo)> {
            vec![(Source::System, 1, one(9)), (Source::System, 0, one(7))]
        }
    }

    #[test]
    fn finals_carry_speakers_and_revisions_follow() {
        let mut inputs = speech(Source::System, 3.0);
        let n = 3 * 16_000;
        inputs.push(AsrInput::Audio {
            source: Source::System,
            first_sample: n,
            samples: vec![0.1; n as usize],
        });
        inputs.push(AsrInput::SpeechStart {
            source: Source::System,
            start: n,
        });
        inputs.push(AsrInput::SpeechEnd {
            source: Source::System,
            start: n,
            end: 2 * n,
            echo_unverified: false,
        });
        let router = Router::new(vec![Arc::new(Echo)]);
        let runner = AsrRunner::new(
            AsrConfig {
                partials: false,
                ..Default::default()
            },
            router,
            Some(Arc::new(Says(Language::English))),
        )
        .with_speakers(Box::new(Alternating));
        let (tx, rx) = crossbeam_channel::unbounded();
        for i in inputs {
            tx.send(i).unwrap();
        }
        drop(tx);
        let mut events = Vec::new();
        runner.run_with(rx, |e| events.push(e));
        let finals: Vec<(u64, Option<u32>, usize)> = events
            .iter()
            .filter_map(|e| match e {
                AsrEvent::Transcript(Output::Final {
                    id, speaker, parts, ..
                }) => Some((*id, *speaker, parts.len())),
                _ => None,
            })
            .collect();
        assert_eq!(finals.len(), 2, "{events:?}");
        assert_eq!(finals[1], (1, Some(1), 1));
        let revisions: Vec<(u64, u32)> = events
            .iter()
            .filter_map(|e| match e {
                AsrEvent::SpeakerRevision { id, speaker, .. } => Some((*id, *speaker)),
                _ => None,
            })
            .collect();
        // The revision of utterance 0 is reported only if it was already emitted; the
        // end-of-session revision of utterance 1 always is. Repeating speaker 7 for
        // utterance 0 changes nothing shown, so it is reported at most once.
        assert!(revisions.contains(&(1, 9)), "{revisions:?}");
        assert!(
            revisions.iter().filter(|r| r.0 == 0).count() <= 1,
            "{revisions:?}"
        );
    }

    /// VAD padding can end a segment after the last audio chunk; when the input then closes,
    /// the run must finish that final instead of waiting for audio forever.
    #[test]
    fn a_final_ending_past_the_audio_finishes_when_input_closes() {
        let n = 3 * 16_000;
        let inputs = vec![
            AsrInput::SpeechStart {
                source: Source::System,
                start: 0,
            },
            AsrInput::Audio {
                source: Source::System,
                first_sample: 0,
                samples: vec![0.1; n],
            },
            AsrInput::SpeechEnd {
                source: Source::System,
                start: 0,
                end: n as u64 + 8_000,
                echo_unverified: false,
            },
        ];
        let (tx, rx) = crossbeam_channel::unbounded();
        let done = std::thread::spawn(move || {
            run_with(
                AsrConfig {
                    partials: false,
                    ..Default::default()
                },
                Language::English,
                inputs,
            )
        });
        let ev = std::thread::spawn(move || {
            let _ = tx.send(done.join().unwrap());
        });
        let events = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the runner did not finish");
        ev.join().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AsrEvent::Transcript(Output::Final { .. }))),
            "{events:?}"
        );
    }

    #[test]
    fn fixed_unsupported_language_is_skipped_not_dropped_silently() {
        let cfg = AsrConfig {
            language: LanguageChoice::Fixed(Language::Odia),
            ..Default::default()
        };
        let ev = run_with(cfg, Language::Odia, speech(Source::Mic, 2.0));
        assert!(
            ev.iter()
                .any(|e| matches!(e, AsrEvent::Skipped { reason, .. } if reason.contains("Odia"))),
            "{ev:?}"
        );
    }

    #[test]
    fn mic_final_is_released_at_shutdown() {
        let ev = run_with(
            AsrConfig::default(),
            Language::English,
            speech(Source::Mic, 2.0),
        );
        assert!(ev.iter().any(|e| matches!(
            e,
            AsrEvent::Transcript(Output::Final {
                source: Source::Mic,
                ..
            })
        )));
    }
}
