# SonicDiar

[![ci](https://github.com/Nikhil1602/sonicdiar/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikhil1602/sonicdiar/actions/workflows/ci.yml)

A standalone, cross-platform audio engine for meeting capture, exposed as a reusable SDK:
- simultaneous microphone and system audio;
- echo cancellation, noise suppression, AGC and VAD;
- streaming transcription in **15 languages**, with **Auto** language detection as the default: English, Chinese, Spanish, French and 11 Indian languages (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Odia, Urdu);
- speaker diarization: who spoke when, live on transcripts, with mic and system audio diarized separately.


## Status

**M0–M4 are done and validated on real hardware**, each with documented open items:
- Windows mic and system capture, echo cancellation, AGC, VAD and drift-corrected alignment;
- a compressed, crash-safe journal;
- live transcription with partial and final results in 15 languages;
- live speaker labels on transcripts.

**M6:** Linux capture works on PulseAudio and PipeWire and is validated in CI with virtual devices; a run on real Linux hardware is still open.

**M7:** macOS capture (microphone, and system audio through a Core Audio process tap on macOS 14.2+) is validated in CI on macOS 14 and 15 with virtual devices; a run on a real Mac is still open.

**M8:** the SDK ships for Rust, C, Python and Node.js as GitHub Releases ([guide](docs/sdk.md)).

| Milestone | Scope | State |
|---|---|---|
| M0 | Core types, host clock, drift estimation, real-time ring buffer, crash-safe journal, CLI, CI | **done** ([report](docs/m0-report.md)) |
| M1 | Windows capture (WASAPI mic + loopback, devices, hot-plug, default-follow, recording engine) | **done, with open items** ([report](docs/m1-report.md)) |
| M2 | AEC3 / NS / AGC2 / VAD, drift-corrected mic/system alignment, journal compression | **done, with open items** ([report](docs/m2-report.md)) |
| M2.5 | Hardening: silence-fill cap, stall watchdog, disk tiers, echo-state flag, session limit, 4 h soak | **done, with open items** ([report](docs/m2.5-report.md)) |
| M3 | Streaming ASR: Whisper (en/zh/es/fr) + IndicConformer (11 Indian languages), two-stage Auto language ID, word LM, numbers as digits, mic-bleed dedupe | **done, with open items** ([report](docs/m3-report.md)) |
| M4 | Diarization: WeSpeaker + average-linkage clustering, pyannote segmentation for speech and overlap, live labels with re-clustering and stable ids | **done, with open items** ([report](docs/m4-report.md)) |
| M5 | Multilingual and code-switching | **mostly pulled into M3**; remaining items are in the M3 report |
| M6 | Linux capture (one backend on the PulseAudio API for PulseAudio and PipeWire), CI end-to-end with virtual devices on both servers | **done in CI; real hardware open** ([report](docs/m6-report.md)) |
| M7 | macOS capture (Core Audio mic, process tap for system audio on macOS 14.2+), CI end-to-end on macOS 14 and 15 with virtual devices | **done in CI; a real Mac is open** ([report](docs/m7-report.md)) |
| M8 | SDK: Rust API, C library, Python and Node.js bindings, release workflow (wheels, npm package, licence report, SBOM, attestations) | **done** ([report](docs/m8-report.md), [guide](docs/sdk.md)) |

The plan from M2.5 on, including the cross-platform backend contract and the test strategy, is in [docs/roadmap.md](docs/roadmap.md).

## Using it as a library

Download a release from the repository's **Releases** page (C library, Python wheel or npm
package), then:

```python
import sonicdiar as sd

sd.fetch_models("models", "core")                 # once: voice detection (~50 MB)
sd.fetch_models("models", only="whisper-small")   # for transcription (~490 MB)
with sd.Session({"transcription": {"whisper_model": "ggml-small.bin"}}) as s:
    for event in s.events(seconds=60):
        if event["type"] == "final":
            print(event["source"], event["text"])
```

Node.js, C and Rust work the same way, with the same configuration and events. The guide is [docs/sdk.md](docs/sdk.md).

## Transcription at a glance (M3)

| | |
|---|---|
| English, Chinese, Spanish, French | Whisper large-v3-turbo q5 (whisper.cpp; GPU through Vulkan, or CPU). WER 3–8% on FLEURS |
| 11 Indian languages | IndicConformer-600M (ONNX Runtime, CPU) with a word trigram LM per language. WER 6.8% (Hindi) to 20.2%; CER 2.9–5.4%, Tamil 14.6% |
| Auto (default) | Whisper language ID, then IndicConformer within the Indian languages, plus a per-meeting language memory. 99.2% per clip, 97–100% per meeting |
| Latency (live) | final text 1.8 s after speech ends (p50), partials 0.5 s (p50) |
| Output conventions | Urdu speech in Auto is written in Hindi script (choose Urdu for Urdu script). English words inside Indian speech are written in that language's script. Spoken numbers become digits in 10 Indian languages (all but Kannada) |

All transcription models and data are MIT, CC0, Unicode or Apache-2.0 licensed; details in [docs/m3-report.md](docs/m3-report.md).

## Diarization at a glance (M4)

| | |
|---|---|
| Offline | DER 15–16% on AMI meetings (0.25 s collar, overlap scored); speaker confusion about 4% |
| Live | each final transcript gets a speaker (S1, S2, … per source) when it is transcribed; re-clustering every 60 s emits revisions with stable ids; confusion on transcribed speech 5.6% |
| In the text | a final is split where the speaker changes (word timings), and overlapped speech is marked with the second speaker; on AMI with Whisper this lowers DER from 31.2% to 26.0% |
| Cost | about 0.03× real time on the CPU |
| Models | WeSpeaker ResNet34-LM speaker embeddings (**CC BY 4.0, attribution: WeSpeaker / VoxCeleb**); pyannote segmentation-3.0 (MIT, © CNRS) |

Details in [docs/m4-report.md](docs/m4-report.md).

## Layout

```
crates/sd-core      types, host clock, drift estimator, timeline aligner, lock-free audio ring
crates/sd-journal   crash-safe on-disk audio journal (+ WAV export)
crates/sd-capture   native capture (Windows: WASAPI; Linux: PulseAudio API for PulseAudio/PipeWire;
                    macOS: Core Audio + process tap), device list, test playback
crates/sd-dsp       echo cancellation (AEC3), echo gate, NS, AGC2, resampling (pure-Rust WebRTC APM)
crates/sd-vad       Silero VAD on ONNX Runtime (loaded at run time) + speech segmenter
crates/sd-asr       speech recognition: Whisper, IndicConformer, Auto language ID, router,
                    streaming policy, word LM + CTC beam search, numbers as digits, metrics
crates/sd-diar      speaker diarization: Kaldi fbank, WeSpeaker embeddings, clustering, pyannote
                    segmentation (speech/overlap), live diarizer, DER
crates/sd-engine    recording sessions: capture -> drain -> journal, processing, ASR thread, health
crates/sonicdiar  the SDK: Session, Config, events (JSON), devices, models
crates/sd-ffi       C library (sonicdiar.h)
bindings/python     Python package (pyo3, maturin)
bindings/node       Node.js package (napi-rs)
tools/sdctl         CLI: devices, record, process, asr (eval/score/tune), diar (eval), journal, models,
                    simulate
tools/ui            sonicdiar-ui: a local test page for the SDK (start/pause/resume/stop, settings,
                    live transcript with speaker labels)
tools/eval          evaluation scripts (language-ID rules, recordings, live playlist)
tools/testdata      fetchers/builders for test data, LM text and number tables
models/             manifest and lock files of pinned (sha256) models and ONNX Runtime libraries
docs/               architecture proposal, roadmap, milestone reports
```

## Build and test

Requires Rust 1.94 (pinned in `rust-toolchain.toml`).

On Windows, building with Whisper (whisper.cpp) also needs:
- CMake (Visual Studio's bundled copy works);
- libclang, with `LIBCLANG_PATH` set to its folder;
- the release-flag fix in `tools/dev/env-windows.sh`, which you source in Git Bash before building.

GPU builds also need the Vulkan SDK (`VULKAN_SDK`), and are built with `--features vulkan`.

On Linux (Ubuntu/Debian package names), building needs CMake, clang (libclang) and `libpulse0` (present on every desktop; `libpulse-dev` also works). Capture needs a running PulseAudio or PipeWire (`pipewire-pulse`) server, which desktops have. To run the Linux capture tests on your own machine (they add and remove virtual devices on the running server): `tools/ci/linux-audio-e2e.sh existing`.

```bash
. tools/dev/env-windows.sh
```

```bash
cargo test --workspace
```

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Models

Models and the ONNX Runtime library are downloaded and hash-checked, never committed.

Core (VAD and ONNX Runtime):

```bash
cargo run --release -p sdctl -- models fetch
```

Whisper turbo (English, Chinese, Spanish, French):

```bash
cargo run --release -p sdctl -- models fetch --only whisper-large-v3-turbo-q5
```

IndicConformer (2.5 GB) is gated on Hugging Face:
1. Accept its terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual.
2. Log in with `hf auth login`, or set `HF_TOKEN`.
3. Run:

```bash
cargo run --release -p sdctl -- models fetch --group indic
```

The word language models are built locally. This fetches public-domain (CC0) IndicCorp text samples, then builds one trigram model per language into `models/lm/`, at about 30–60 s each:

```bash
python tools/testdata/fetch_indiccorp.py
```

```bash
for l in bn gu hi kn ml mr pa ta te or ur; do cargo run --release -p sdctl -- asr lm-build --corpus bench/data/indiccorp/$l.txt --out models/lm/$l.bin; done
```

Speaker diarization models (WeSpeaker ResNet34-LM, pyannote segmentation-3.0):

```bash
cargo run --release -p sdctl -- models fetch --group diar
```

Check everything:

```bash
cargo run --release -p sdctl -- models verify --group all
```

### Trying the SDK in the browser

```bash
cargo run --release -p sonicdiar-ui -- --models models
```

The page opens at `http://127.0.0.1:8765/`. It has Start, Pause, Resume and Stop, a language
setting (Auto by default) and the live transcript labelled by speaker. On a CPU without a GPU,
pick Whisper `small` ([docs/asr-speed.md](docs/asr-speed.md)). More in
[docs/sdk.md §9](docs/sdk.md#9-the-test-page-sonicdiar-ui).

### Recording and transcription (Windows)

```bash
cargo run --release -p sdctl -- devices
```

Record 60 s and export WAVs:

```bash
cargo run --release -p sdctl -- record --seconds 60 --out rec1 --wav
```

Record with live processing and transcription (Auto language, partial results, transcript TSV):

```bash
cargo run --release -p sdctl --features vulkan -- record --seconds 120 --out rec2 --process --asr-model models/ggml-large-v3-turbo-q5_0.bin --indic models/indicconformer/assets --indic-lm models/lm --gpu --partials --transcript rec2.tsv
```

Replay a recording through processing and transcription, exactly as live:

```bash
cargo run --release -p sdctl -- process rec2 --asr-model models/ggml-large-v3-turbo-q5_0.bin --indic models/indicconformer/assets --indic-lm models/lm --transcript rec2-replay.tsv
```

Without `--gpu`, Whisper runs on the CPU.

**Other options:**
- `--diarize` adds speaker labels (per source) and revisions to the transcripts. A final is split where the speaker changes, and overlapped speech is marked (`--no-overlap` turns that off).
- `--language <code>` fixes the language instead of Auto.
- `--no-itn` keeps spoken numbers as words.
- `--indic-decoding rnnt` uses RNN-T instead of CTC.

### Evaluation

FLEURS test sets (CC-BY 4.0, evaluation only) and WER / CER / language-ID accuracy per language:

```bash
python tools/testdata/fetch_fleurs.py
```

```bash
cargo run --release -p sdctl -- asr eval --model models/ggml-large-v3-turbo-q5_0.bin --indic models/indicconformer/assets --indic-lm models/lm --mode auto --out target/eval/auto.tsv
```

```bash
cargo run --release -p sdctl -- asr score target/eval/auto.tsv
```

Diarization on AMI meetings (CC BY 4.0, about 180 MB):

```bash
python tools/testdata/fetch_ami.py
```

```bash
cargo run --release -p sdctl -- diar eval --overlap
```

```bash
cargo run --release -p sdctl -- diar online-eval
```

Other checks:

```bash
cargo run --release -p sdctl -- simulate --seconds 3600 --speed 120
```

```bash
cargo run --release -p sdctl -- journal inspect <journal-dir>
```

```bash
cargo run --release -p sdctl -- loopback-test
```

`loopback-test` plays about 8 s of quiet noise through the default speakers and verifies the system-audio capture of it.

## Platform support

| | Windows | macOS | Linux |
|---|---|---|---|
| Code builds and tests pass | verified locally (Windows 11 x64) and in CI | verified in CI (`macos-latest`) | verified in CI (`ubuntu-latest`) |
| Audio capture | WASAPI mic + loopback, validated on real hardware | Core Audio mic + process tap (macOS 14.2+): validated in CI with virtual devices; a real Mac not yet tested | PulseAudio API (PulseAudio or PipeWire): validated in CI with virtual devices; real hardware not yet tested |
| Transcription | validated on real hardware (Whisper on Vulkan GPU or CPU; IndicConformer on CPU) | builds; not validated | builds; not validated |
| Diarization | validated (AMI and a live session) | builds; not validated | builds; not validated |

## Licence

SonicDiar is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Its third-party dependencies are all permissively licensed; CI checks this with `cargo-deny` (`deny.toml`), and every release includes `THIRD-PARTY-LICENSES.md`. Models are downloaded separately under their own licences (see `models/manifest.json`; WeSpeaker requires attribution).
