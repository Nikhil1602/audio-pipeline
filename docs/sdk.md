# SonicDiar SDK: integration guide

SonicDiar records a meeting from the computer it runs on: the **microphone** and the
**system audio** (what the computer plays, i.e. the other participants), as two separate
streams. Optionally it cancels the echo of the speakers in the microphone, detects speech,
transcribes it live in 15 languages and labels who spoke. Everything runs locally; nothing is
sent anywhere.

The same API exists in four languages. Rust is the native one. C, Python and Node.js wrap it
and exchange the same configuration and events as JSON.

| Language | Package | Install |
|---|---|---|
| Rust | crate `sonicdiar` (`crates/sonicdiar`) | path or git dependency |
| C / C++ / any FFI | `sonicdiar-<version>-<platform>` (library + `sonicdiar.h`) | link `sonicdiar` |
| Python 3.9+ | wheel `sonicdiar-<version>-*.whl` | `pip install <wheel>` |
| Node.js 18+ / Electron | `sonicdiar-<version>.tgz` (prebuilt for all platforms) | `npm install <tgz>` |

Downloads: the GitHub Releases of the repository.

**Trying it without writing code:** `sonicdiar-ui` (`tools/ui`) opens a test page in the browser. It
has start, pause, resume and stop, the settings (language, model, devices, speaker labels) and
the live transcript labelled by speaker. See [§9](#9-the-test-page-sonicdiar-ui).

## 1. Quick start

**Python**
```python
import sonicdiar as sd

sd.fetch_models("models", "core")                         # once: voice detection (~50 MB)
sd.fetch_models("models", only="whisper-small")           # for transcription (~490 MB)

config = {"transcription": {"whisper_model": "ggml-small.bin"}}
with sd.Session(config) as session:
    for event in session.events(seconds=60):
        if event["type"] == "final":
            print(f'{event["source"]} {event["speaker"] or ""}: {event["text"]}')
print(session.summary)
```

**Node.js**
```js
const sd = require('sonicdiar');
await sd.fetchModels('models', 'core');
const session = new sd.Session({ transcription: { whisper_model: 'ggml-small.bin' } });
for await (const e of session.events({ seconds: 60 })) if (e.type === 'final') console.log(e.text);
```

**C**
```c
#include "sonicdiar.h"
sd_models_fetch("models", "core", NULL, NULL, NULL);
sd_session *s;
if (sd_session_start("{\"transcription\": {\"whisper_model\": \"ggml-small.bin\"}}", &s) != SD_OK) {
    char *e = sd_last_error(); fprintf(stderr, "%s\n", e); sd_string_free(e); return 1;
}
char *ev;
while (running) {
    if (sd_session_next_event(s, 200, &ev) == SD_OK && ev) { handle(ev); sd_string_free(ev); }
}
char *summary;
sd_session_stop(s, &summary);
while (sd_session_next_event(s, 0, &ev) == SD_OK && ev) { handle(ev); sd_string_free(ev); }
sd_string_free(summary);
sd_session_free(s);
```

**Rust**
```rust
use sonicdiar::{Config, Session, models};
models::fetch("models".as_ref(), "core", None, &mut |_| {})?;
let mut session = Session::start(Config::from_json(r#"{"transcription": null}"#)?)?;
while let Some(event) = session.next_event(std::time::Duration::from_secs(1)) { /* ... */ }
let summary = session.stop()?;
```

**Stopping.** Stopping finishes the transcription of speech that was still queued, and those
transcripts arrive as events **after** `stop` returns. Keep reading events until there are no
more. Python's `events()` and Node's `events()` iterators do this for you.

## 2. Configuration

Every field is optional; `{}` records the default microphone and system audio with
processing on and no transcription. Unknown fields are an error, so typos are caught.

| Field | Default | Meaning |
|---|---|---|
| `models_dir` | `"models"` | Where models and the ONNX Runtime library are. Model paths are relative to it. |
| `journal_dir` | a new temp directory | The crash-safe recording on disk; must not exist yet. |
| `mic` | `"default"` | `"default"` (follows the system default), `"none"`, or a device `id` from the device list. |
| `system` | `"default"` | The output device whose sound is captured: `"default"`, `"none"` or an id. On macOS the capture hears all apps, whichever device they play on. |
| `processing` | `true` | Echo cancellation, gain control, voice activity detection (needed for transcription). |
| `echo_cancellation` | `"auto"` | `"auto"` (on when the speakers' sound reaches the mic), `"on"`, `"off"`. |
| `gain_control` | `false` | Automatic gain control on the microphone. |
| `transcription` | `null` (off) | An object (below) turns live transcription on. |
| `diarization` | `null` (off) | `{"overlap": true}` labels speakers (needs transcription). |
| `levels` | `false` | `level` events about 10 times a second per source. |
| `health` | `false` | `health` events once a second. |

`transcription`:

| Field | Default | Meaning |
|---|---|---|
| `language` | `"auto"` | `"auto"` or `en`, `zh`, `es`, `fr`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `pa`, `ta`, `te`, `or`, `ur`. |
| `whisper_model` | `"ggml-large-v3-turbo-q5_0.bin"` | Whisper model file (group `asr`); `null` for IndicConformer only. |
| `indic` | `false` | IndicConformer for the 11 Indian languages (group `indic`, 2.5 GB, gated). |
| `indic_lm` | `null` | Directory of word language models for IndicConformer. |
| `partials` | `true` | `partial` events while someone speaks. |
| `numbers_as_digits` | `true` | Spoken numbers as digits (Indian languages). |
| `gpu`, `gpu_device` | `false`, `0` | Whisper on the GPU (GPU builds only). |
| `threads` | engine default | CPU threads per engine. |

## 3. Events

Every event is a JSON object with a `type`. Times are seconds since the session started, on
that source's timeline. Speakers are `S1`, `S2`, ... per source.

| `type` | Fields | When |
|---|---|---|
| `device_started` | `source`, `device_id`, `device_name` | Capture started on a device (initially and after every switch). |
| `device_switching` | `source`, `reason` (`default_changed`, `removed`, `requested`, `recovery`) | Moving to another device; the timeline continues. |
| `no_device` | `source` | No usable device; capture keeps retrying. |
| `permission_needed` | `source`, `what`, `confirmed` | macOS withholds audio for lack of a permission; `what` says where to grant it. |
| `capture_problem` | `source`, `message` | The backend is recovering from a device problem. |
| `level` | `source`, `rms_dbfs`, `peak_dbfs` | With `levels`. |
| `speech` | `source`, `start`, `end` | A speech segment (voice activity detection). |
| `partial` | `source`, `utterance`, `start`, `end`, `text`, `language` | Text so far of an utterance in progress. |
| `final` | `source`, `utterance`, `start`, `end`, `text`, `language`, `language_probabilities`, `words` [{`start`,`end`,`text`}], `speaker`, `parts` [{`start`,`end`,`speaker`,`text`}], `overlaps` [{`start`,`end`,`speaker`}], `echo_unverified` | The final text of an utterance. `parts` splits it where the speaker changes; `overlaps` marks a second speaker talking at once. `text` is empty when nothing recognizable was said (it still replaces the partial). |
| `speaker_revision` | `source`, `utterance`, `speaker`, `parts`, `overlaps` | Re-clustering changed the speakers of an earlier final. Update your display. |
| `bleed_removed` | `source`, `utterance`, `start`, `end`, `text` | A mic utterance recognized as the other side leaking into the mic; not a final. |
| `transcription_skipped` | `source`, `utterance`, `start`, `end`, `reason` | An utterance that could not be transcribed. |
| `audio_dropped` | `source`, `frames` | The disk stalled longer than the in-memory buffer (30 s). |
| `disk_space_low`, `journal_paused`, `journal_resumed` | `free_bytes` | Disk space handling. |
| `journal_error`, `journal_recovered` | `message`, `disk_full` | Write errors and recovery. |
| `warning` | `message` | Non-fatal problems. |
| `health` | `elapsed`, `sources` [{`source`,`timeline`,`dropped_frames`,`device_glitches`,`restarts`,`drift_ppm`}] | With `health`. |

**Pause and resume** (`pause()` / `resume()`, `sd_session_pause` / `sd_session_resume`): while
paused, capture keeps running but its audio is replaced by silence before anything sees it.
Nothing is recorded or transcribed, speech in progress ends (and is transcribed), and the mic
and system timelines stay aligned. Event times keep counting from the start, pauses included.

The summary that `stop` returns contains `journal_dir`, `duration`, `sources` (as in `health`),
`journaled_frames`, `journal_errors` and `complete` (true when nothing was dropped).

## 4. Models

Models are never bundled. `fetch_models` / `sd_models_fetch` downloads them on first use, and
`list_models` shows what is present. Every file is checked against a size and SHA-256 that are
compiled into the library, so a tampered or partial download is never used.

| Group | Contents | Size | Licence |
|---|---|---|---|
| `core` | Silero VAD + this platform's ONNX Runtime | ~50 MB | MIT |
| `asr` | Whisper models (pick one with `only`, e.g. `whisper-small`, `whisper-large-v3-turbo-q5`) | 0.5–1.6 GB each | MIT |
| `diar` | WeSpeaker ResNet34 (speaker embeddings) + pyannote segmentation-3.0 | ~33 MB | CC BY 4.0 (attribution: WeSpeaker / VoxCeleb), MIT |
| `indic` | IndicConformer (11 Indian languages) | 2.5 GB | MIT; gated on Hugging Face: needs a token (`HF_TOKEN`) |

## 5. Threads

All functions may be called from any thread. Reading events, stopping and downloading block:
- Python releases the GIL during them.
- Node runs them off the JavaScript thread (Promises).
- C and Rust can read events on one thread while another stops the session (in Rust, use
  `Session::events()`).

Starting a session loads the speech models, which can take a few seconds.

## 6. Platforms

| | Windows 10/11 x64 | macOS 12+ (Apple silicon and Intel) | Linux x64 (glibc 2.35+) |
|---|---|---|---|
| Microphone | WASAPI | Core Audio | PulseAudio API (PulseAudio or PipeWire) |
| System audio | WASAPI loopback | Core Audio process tap, **macOS 14.2+** | monitor of the output device |
| Permissions | microphone privacy setting | **Microphone** and **System Audio Recording** prompts (granted to the app, or to Terminal for scripts) | none |
| Validated on | real hardware | CI on macOS 14/15 (virtual devices); real Macs pending | CI on PulseAudio and PipeWire (virtual devices); real hardware pending |

- **Electron on macOS:** add `NSMicrophoneUsageDescription` and `NSAudioCaptureUsageDescription` to the app's `Info.plist`; otherwise macOS refuses the prompts.
- **Linux packaging:** the library loads the system's `libpulse.so.0` (present on every desktop). Flatpak needs `--socket=pulseaudio`. The Python wheel is tagged `linux_x86_64`, not manylinux, because it depends on that system library.
- **GPU:** the release builds run Whisper on the CPU. GPU (Vulkan) builds are a build option (`--features vulkan`).

## 7. Privacy

- Audio, transcripts and models stay on the machine. The only network access is
  `fetch_models`, which downloads the pinned files from GitHub and Hugging Face. The Hugging
  Face token, if used, is sent only to huggingface.co.
- The journal (`journal_dir`) holds the full recording until you delete it. Sessions never
  delete it themselves.
- Recording other people may need their consent where you are; the SDK does not decide that.

## 8. Known limits

- **Without a GPU, use Whisper `small` for live transcription.** On a laptop CPU (i5-12450HX,
  12 threads), `small` runs at 0.35× real time with the shortened encoder windows. The large
  models stay 2–3× slower than real time, so their transcripts fall behind and `stop` waits
  for the backlog. Measurements: [asr-speed.md](asr-speed.md).
- **Diarization** labels are refined by re-clustering (`speaker_revision`); the first label
  of an utterance can change.
- **macOS 13–14.1:** system audio is not available; the microphone is.
- See the milestone reports in `docs/` for measured accuracy and the open items per platform.

## 9. The test page (`sonicdiar-ui`)

`sonicdiar-ui` serves a page on `http://127.0.0.1:8765/` and opens it in the browser.
- **What it is:** a thin client of the Rust SDK. Everything the page shows comes from the SDK's events.
- **Who can reach it:** only this computer. Each API call needs a token embedded in the page, so other web sites cannot drive it.

```bash
cargo run --release -p sonicdiar-ui -- --models models
```

**Options:**
- `--models <dir>`: the models directory (default `models`).
- `--port <n>`: the port (default 8765; `0` picks any free port).
- `--no-browser`: don't open the browser.

**What the page has:**
- **Controls:** Start, Pause, Resume and Stop, a session clock, and level meters for the mic and system audio.
- **Settings:**
  - language (Auto-detect by default, or one of the 15 languages);
  - speech model (any downloaded Whisper model, or off);
  - IndicConformer;
  - microphone and system audio device (both can also be switched while recording);
  - speaker labels and overlap marking;
  - live partial text;
  - echo cancellation;
  - mic gain control;
  - CPU threads.

  The exact configuration sent to `Session::start` is shown under "Session config".
- **Live transcript:**
  - Both sources are merged on one timeline and grouped into turns per speaker ("Mic · S1", "System · S2"). Click a name to rename the speaker.
  - Partial text is shown in grey until its final arrives.
  - Speaker revisions relabel earlier turns.
  - Pauses are marked in the transcript.
  - After stop, a summary shows whether anything was dropped and where the recording (journal) is.
  - Copy, Save .txt and Save events (every event, as JSON).
- **Side panels:**
  - speakers and their talk time;
  - models, with a Download button for missing ones;
  - health once a second;
  - the event log (device changes, permissions, speech segments, warnings).
