//! The event model: what a session reports, as plain data that serializes to JSON the same way
//! in every language binding (`{"type": "final", "source": "mic", ...}`).
//!
//! Times are seconds since the session started, on the source's own timeline (the journal's).
//! Speakers are labelled `S1`, `S2`, ... per source.

use serde::Serialize;

use sd_asr::runner::AsrEvent;
use sd_asr::speakers::{SpeakerPart, SpeakerSpan};
use sd_asr::stream::Output;
use sd_capture::{CaptureEvent, SwitchReason};
use sd_core::Source;
use sd_engine::{HealthReport, RecorderEvent};

/// Which stream an event is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceName {
    Mic,
    System,
}

impl From<sd_asr::stream::Source> for SourceName {
    fn from(s: sd_asr::stream::Source) -> Self {
        match s {
            sd_asr::stream::Source::Mic => SourceName::Mic,
            sd_asr::stream::Source::System => SourceName::System,
        }
    }
}

impl From<Source> for SourceName {
    fn from(s: Source) -> Self {
        match s {
            Source::Mic => SourceName::Mic,
            Source::System => SourceName::System,
        }
    }
}

/// A word with its time span.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// A stretch of a final's text spoken by one speaker.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Part {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
    pub text: String,
}

/// Where a second speaker talks at the same time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Overlap {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// Per-source health numbers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceHealth {
    pub source: SourceName,
    /// Seconds on the source's timeline (silence filled in for gaps included).
    pub timeline: f64,
    /// Audio lost between capture and disk (frames); 0 in a healthy session.
    pub dropped_frames: u64,
    pub device_glitches: u64,
    pub restarts: u64,
    /// The device clock's deviation from nominal, once measured.
    pub drift_ppm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Capture started on a device (initially and after every switch).
    DeviceStarted {
        source: SourceName,
        device_id: String,
        device_name: String,
    },
    /// Capture is moving to another device: `default_changed`, `removed`, `requested`,
    /// `recovery`.
    DeviceSwitching {
        source: SourceName,
        reason: String,
    },
    /// No usable device; capture keeps the timeline running and retries.
    NoDevice {
        source: SourceName,
    },
    /// The OS withholds audio for lack of a privacy permission (macOS). `confirmed`: the OS
    /// reported a denial (otherwise it is inferred). `what` says where to grant it.
    PermissionNeeded {
        source: SourceName,
        what: String,
        confirmed: bool,
    },
    /// A capture problem the backend is recovering from (retrying, stalled device, ...).
    CaptureProblem {
        source: SourceName,
        message: String,
    },
    /// Audio level (when `levels` is on).
    Level {
        source: SourceName,
        rms_dbfs: f32,
        peak_dbfs: f32,
    },
    /// A completed speech segment (voice activity detection).
    Speech {
        source: SourceName,
        start: f64,
        end: f64,
    },
    /// Text so far for an utterance still in progress (replaced by its final).
    Partial {
        source: SourceName,
        utterance: u64,
        start: f64,
        end: f64,
        text: String,
        language: String,
    },
    /// The final text of an utterance.
    Final {
        source: SourceName,
        utterance: u64,
        start: f64,
        end: f64,
        text: String,
        language: String,
        /// Top language-identification candidates: (code, probability).
        language_probabilities: Vec<(String, f32)>,
        words: Vec<Word>,
        /// Majority speaker (with diarization).
        speaker: Option<String>,
        /// The text split where the speaker changes (with diarization).
        parts: Vec<Part>,
        overlaps: Vec<Overlap>,
        /// Mic only: possibly the far end leaking into the mic (echo not yet verified).
        echo_unverified: bool,
    },
    /// Re-clustering changed the speakers of an already emitted final.
    SpeakerRevision {
        source: SourceName,
        utterance: u64,
        speaker: String,
        parts: Vec<Part>,
        overlaps: Vec<Overlap>,
    },
    /// A mic utterance judged to be the far end's audio leaking into the mic; not a final.
    BleedRemoved {
        source: SourceName,
        utterance: u64,
        start: f64,
        end: f64,
        text: String,
    },
    /// An utterance that could not be transcribed.
    TranscriptionSkipped {
        source: SourceName,
        utterance: u64,
        start: f64,
        end: f64,
        reason: String,
    },
    /// Audio was dropped (the disk stalled for longer than the in-memory queue lasts).
    AudioDropped {
        source: SourceName,
        frames: u64,
    },
    DiskSpaceLow {
        free_bytes: u64,
    },
    /// Writing the journal failed; audio is held in memory and writing is retried.
    JournalError {
        message: String,
        disk_full: bool,
    },
    JournalRecovered,
    JournalPaused {
        free_bytes: u64,
    },
    JournalResumed {
        free_bytes: u64,
    },
    /// A non-fatal problem (e.g. an optional component could not start).
    Warning {
        message: String,
    },
    /// Health numbers (when `health` is on), once a second.
    Health {
        elapsed: f64,
        sources: Vec<SourceHealth>,
    },
}

const SAMPLE_RATE: f64 = 16_000.0;

fn secs(samples: u64) -> f64 {
    samples as f64 / SAMPLE_RATE
}

fn speaker(s: u32) -> String {
    format!("S{}", s + 1)
}

fn parts(p: &[SpeakerPart]) -> Vec<Part> {
    p.iter()
        .map(|p| Part {
            start: secs(p.start),
            end: secs(p.end),
            speaker: speaker(p.speaker),
            text: p.text.clone(),
        })
        .collect()
}

fn overlaps(o: &[SpeakerSpan]) -> Vec<Overlap> {
    o.iter()
        .map(|o| Overlap {
            start: secs(o.start),
            end: secs(o.end),
            speaker: speaker(o.speaker),
        })
        .collect()
}

fn reason(r: &SwitchReason) -> &'static str {
    match r {
        SwitchReason::DefaultDeviceChanged => "default_changed",
        SwitchReason::DeviceRemoved | SwitchReason::DeviceInvalidated => "removed",
        SwitchReason::Requested => "requested",
        SwitchReason::Recovery => "recovery",
    }
}

pub(crate) fn health(h: &HealthReport) -> Vec<SourceHealth> {
    h.sources
        .iter()
        .filter_map(|s| {
            let source = s.source?;
            let cap = s.capture.unwrap_or_default();
            Some(SourceHealth {
                source: source.into(),
                // Both sources capture at 48 kHz.
                timeline: s.timeline_frames as f64 / 48_000.0,
                dropped_frames: s.ring.dropped_frames + s.queue_dropped_frames,
                device_glitches: cap.device_glitches,
                restarts: cap.restarts,
                drift_ppm: s.drift_ppm,
            })
        })
        .collect()
}

/// Converts an engine event; `None` for events the SDK does not expose (or switched off).
pub(crate) fn convert(e: RecorderEvent, levels: bool, with_health: bool) -> Option<Event> {
    Some(match e {
        RecorderEvent::Capture { source, event } => {
            let source = source.into();
            match event {
                CaptureEvent::Started {
                    device_id,
                    device_name,
                    ..
                } => Event::DeviceStarted {
                    source,
                    device_id,
                    device_name,
                },
                CaptureEvent::Switching { reason: r } => Event::DeviceSwitching {
                    source,
                    reason: reason(&r).into(),
                },
                CaptureEvent::NoDevice => Event::NoDevice { source },
                CaptureEvent::PermissionNeeded { what, confirmed } => Event::PermissionNeeded {
                    source,
                    what,
                    confirmed,
                },
                CaptureEvent::Retrying { error, .. } => Event::CaptureProblem {
                    source,
                    message: error,
                },
                CaptureEvent::Stalled { quiet_ms } => Event::CaptureProblem {
                    source,
                    message: format!("the device delivered nothing for {quiet_ms} ms; reopening"),
                },
                CaptureEvent::GapFillBudgetExhausted { filled_ms } => Event::CaptureProblem {
                    source,
                    message: format!(
                        "the device kept losing audio ({filled_ms} ms filled); reopening"
                    ),
                },
                CaptureEvent::Stopped => return None,
            }
        }
        RecorderEvent::Level {
            source,
            rms_dbfs,
            peak_dbfs,
            ..
        } => {
            if !levels {
                return None;
            }
            Event::Level {
                source: source.into(),
                rms_dbfs,
                peak_dbfs,
            }
        }
        RecorderEvent::Health(h) => {
            if !with_health {
                return None;
            }
            Event::Health {
                elapsed: h.elapsed.as_secs_f64(),
                sources: health(&h),
            }
        }
        RecorderEvent::JournalError { message, disk_full } => {
            Event::JournalError { message, disk_full }
        }
        RecorderEvent::JournalRecovered => Event::JournalRecovered,
        RecorderEvent::QueueOverflow {
            source,
            dropped_frames,
        } => Event::AudioDropped {
            source: source.into(),
            frames: dropped_frames,
        },
        RecorderEvent::Speech {
            source,
            start_sample,
            end_sample,
            ..
        } => Event::Speech {
            source: source.into(),
            start: secs(start_sample),
            end: secs(end_sample),
        },
        RecorderEvent::SpeechStart { .. } | RecorderEvent::SpeechDiscarded { .. } => return None,
        RecorderEvent::DiskSpaceLow { free_bytes } => Event::DiskSpaceLow { free_bytes },
        RecorderEvent::JournalPaused { free_bytes } => Event::JournalPaused { free_bytes },
        RecorderEvent::JournalResumed { free_bytes } => Event::JournalResumed { free_bytes },
        RecorderEvent::SessionLimitApproaching { remaining } => Event::Warning {
            message: format!(
                "the session limit is reached in {:.0} s",
                remaining.as_secs_f64()
            ),
        },
        RecorderEvent::SessionLimitReached { elapsed } => Event::Warning {
            message: format!(
                "the session limit was reached after {:.0} s",
                elapsed.as_secs_f64()
            ),
        },
        RecorderEvent::Warning { message } => Event::Warning { message },
        RecorderEvent::EchoCanceller { .. } => return None,
        RecorderEvent::Asr(a) => return asr(a),
    })
}

fn asr(a: AsrEvent) -> Option<Event> {
    Some(match a {
        AsrEvent::Transcript(Output::Partial {
            source,
            id,
            start,
            end,
            text,
            language,
        }) => Event::Partial {
            source: source.into(),
            utterance: id,
            start: secs(start),
            end: secs(end),
            text,
            language: language.code().into(),
        },
        AsrEvent::Transcript(Output::Final {
            source,
            id,
            start,
            end,
            text,
            language,
            language_probs,
            words,
            echo_unverified,
            speaker: spk,
            parts: p,
            overlaps: o,
            ..
        }) => Event::Final {
            source: source.into(),
            utterance: id,
            start: secs(start),
            end: secs(end),
            text,
            language: language.code().into(),
            language_probabilities: language_probs
                .iter()
                .take(3)
                .map(|(l, p)| (l.code().to_string(), *p))
                .collect(),
            words: words
                .into_iter()
                .map(|w| Word {
                    start: w.start_ms as f64 / 1000.0,
                    end: w.end_ms as f64 / 1000.0,
                    text: w.text,
                })
                .collect(),
            speaker: spk.map(speaker),
            parts: parts(&p),
            overlaps: overlaps(&o),
            echo_unverified,
        },
        AsrEvent::Transcript(Output::Dropped {
            source,
            id,
            start,
            end,
            text,
            ..
        }) => Event::BleedRemoved {
            source: source.into(),
            utterance: id,
            start: secs(start),
            end: secs(end),
            text,
        },
        AsrEvent::SpeakerRevision {
            source,
            id,
            speaker: spk,
            parts: p,
            overlaps: o,
        } => Event::SpeakerRevision {
            source: source.into(),
            utterance: id,
            speaker: speaker(spk),
            parts: parts(&p),
            overlaps: overlaps(&o),
        },
        AsrEvent::Skipped {
            source,
            id,
            start,
            end,
            reason,
        } => Event::TranscriptionSkipped {
            source: source.into(),
            utterance: id,
            start: secs(start),
            end: secs(end),
            reason,
        },
        AsrEvent::Timing { .. } => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_serialize_as_tagged_json() {
        let e = Event::DeviceSwitching {
            source: SourceName::System,
            reason: "removed".into(),
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"type":"device_switching","source":"system","reason":"removed"}"#
        );
        assert_eq!(
            serde_json::to_string(&Event::JournalRecovered).unwrap(),
            r#"{"type":"journal_recovered"}"#
        );
    }

    #[test]
    fn speech_positions_become_seconds() {
        let e = convert(
            RecorderEvent::Speech {
                source: Source::Mic,
                start_sample: 16_000,
                end_sample: 40_000,
                echo_unverified: false,
            },
            false,
            false,
        );
        assert_eq!(
            e,
            Some(Event::Speech {
                source: SourceName::Mic,
                start: 1.0,
                end: 2.5
            })
        );
    }
}
