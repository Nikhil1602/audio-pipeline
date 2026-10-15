# Architecture proposal (draft for review)

Status: **approved 2026-10-13** (Rust core; Whisper + IndicConformer; Windows-first with CI for macOS/Linux). M0 done, see [m0-report.md](m0-report.md).

## 1. Design principles

1. **Never lose audio.** Captured audio is written to a crash-safe on-disk journal before any heavy processing. ASR/diarization can fall behind and catch up; capture never waits on them.
2. **Real-time threads do no work they can't bound.** Device callbacks only copy into lock-free ring buffers. No allocation, locks, logging or I/O on those threads.
3. **Keep sources separate.** Mic and system audio stay distinct streams end to end. That gives free coarse diarization ("local" vs "remote") and a clean far-end reference for echo cancellation.
4. **One clock.** Every frame gets a host-monotonic timestamp. Drift between devices is estimated and corrected, so mic and system timelines align over hours.
5. **Measured, not claimed.** Each component ships with a benchmark and acceptance thresholds. Anything unmeasured is documented as such.
6. **Local-first and private by default.** No network traffic except an explicit, checksum-verified model download. Cloud ASR is an opt-in plug-in.

## 2. Technology choices

| Concern | Choice | Why / alternatives |
|---|---|---|
| Core language | **Rust** (stable ≥ 1.91) | Memory safety for a long-running real-time engine, first-class C ABI, napi-rs (Node) and PyO3 (Python) bindings. Rust 1.94 + MSVC are already on this machine. Alternative: C++20 (more audio libraries are native C++, but worse safety/tooling). |
| Windows capture | WASAPI via `windows` crate: event-driven shared mode; system = loopback on render endpoint; optional per-process loopback (Win10 2004+) | IMMNotificationClient for hot-plug/default changes; MMCSS "Pro Audio" thread priority. |
| macOS capture | Core Audio AUHAL (mic); **Core Audio process tap** (macOS 14.2+) for system audio; ScreenCaptureKit audio fallback (13.x) | Taps need `NSAudioCaptureUsageDescription`; there is no permission-status API, so the engine must probe and surface a clear error. |
| Linux capture | **PipeWire** native client (`pipewire` crate); PulseAudio client (`libpulse-binding`) fallback; sink `.monitor` for system audio | No shelling out to `parec`/`ffmpeg`. ALSA-only systems: mic only, documented. |
| Resampling / drift | `rubato` (async sinc resampler, MIT) driven by a drift estimator | Corrects ppm-level clock drift between mic and render devices. |
| AEC / NS / AGC | **WebRTC audio processing (AEC3, NS, AGC2, HPF)**. Candidates: `sonora` (pure-Rust port, BSD-3, v0.2.0 Jul 2026, young) vs C++ `webrtc-audio-processing` 2.x via FFI (mature, packaging issues reported). **Decide by benchmark in milestone M2.** | Far-end reference = system loopback stream. Fallback NS: RNNoise (BSD). |
| VAD | **Silero VAD v5** (MIT, ONNX, ~2 MB) for streaming gating; pyannote segmentation-3.0 (MIT) for overlap-aware segmentation inside diarization | Silero is cheap per 32 ms frame; seg-3.0 detects overlap. |
| ONNX inference | ONNX Runtime via `ort` crate (CPU default; DirectML / CoreML / CUDA optional) | One runtime for VAD, segmentation, embeddings, LID, Indic ASR. |
| ASR (general multilingual) | **whisper.cpp** via `whisper-rs` (MIT), quantized ggml models (tiny → large-v3-turbo); Vulkan / Metal / CUDA optional | Streaming via sliding window + LocalAgreement-style stable-prefix commit → partial + final results with word timestamps. |
| ASR (Indic, incl. Gujarati) | **AI4Bharat IndicConformer-600M** (MIT, gated download, 22 Indic languages, CTC/RNNT) via ONNX | Whisper lists Gujarati but quality must be measured. IndicConformer has **no English**, so Hinglish/Gujlish needs routing (below). |
| Language ID | Whisper language-detect logits per VAD segment, with an optional dedicated LID model (e.g. VoxLingua107 ECAPA) if benchmarks justify it | Feeds the language router. |
| Diarization | pyannote seg-3.0 + speaker embeddings (WeSpeaker ResNet34 or 3D-Speaker CAM++, Apache-2.0) via ONNX; **online** centroid assignment for live labels + periodic **global AHC re-clustering** with label reconciliation | Speaker profiles (opt-in) stored locally as embeddings in SQLite. |
| Storage | Audio journal: segmented WAV/FLAC with fsync'd index; metadata: SQLite (`rusqlite`) | Optional AES-GCM encryption at rest with a host-supplied key. |
| Logging / metrics | `tracing` with structured fields; content redaction by default; built-in counters (xruns, drops, queue depth, RTF, latency) | No telemetry is sent anywhere. |

All candidates are permissive (MIT / BSD / Apache-2.0). Model weights have their own terms; IndicConformer requires accepting AI4Bharat's gated-access terms, so the SDK will **not** bundle it and the user/host app downloads it.

## 3. Component architecture

```
            ┌──────────────────────────── sd-capture ─────────────────────────────┐
 Device ──► │ backend (WASAPI | CoreAudio/Tap/SCK | PipeWire/Pulse)                │
 callbacks  │   └► SPSC ring (per source, host-timestamped frames)                │
            └──────────────┬──────────────────────────────────────┬──────────────┘
                           │ mic                                   │ system (far-end ref)
                     ┌─────▼──────────── sd-dsp (DSP thread, 10 ms frames, 48k→16k) ───────┐
                     │ drift-correct ─► HPF ─► AEC3(ref=system) ─► NS ─► AGC2 ─► VAD(Silero) │
                     └─────┬─────────────────────────────────────────────┬────────────────┘
                           │                                             │
              ┌────────────▼─────────────┐                 ┌─────────────▼────────────┐
              │ sd-journal (crash-safe   │                 │ segmenter: speech regions │
              │ raw + processed audio)   │                 │ per source, bounded queue │
              └──────────────────────────┘                 └─────────────┬────────────┘
                                                                          │
                     ┌──────────────── sd-lang: LID per segment ─► router ┤
                     │                                                    │
           ┌─────────▼──────────┐                              ┌──────────▼──────────┐
           │ sd-asr: whisper    │                              │ sd-asr: Indic engine │
           │ streaming (partial │                              │ (Gujarati, Hindi …)  │
           │ + final, word ts)  │                              └──────────┬──────────┘
           └─────────┬──────────┘                                         │
                     └───────────────────────┬────────────────────────────┘
                                             ▼
                     sd-diar: seg-3.0 + embeddings → online labels → periodic re-cluster
                                             ▼
                     sd-engine: event bus → TranscriptEvent{partial|final|revision},
                                SpeakerEvent, DeviceEvent, LevelEvent, HealthEvent
                                             ▼
                     sd-ffi (C ABI)  →  Rust API  /  Node (napi-rs)  /  Python (PyO3)
```

### Threading and back-pressure
- Device threads → lock-free SPSC rings (≥ 2 s capacity). An overflow is counted and reported, never silent.
- One DSP thread (real-time priority) processes 10 ms frames for both sources in lock-step, so AEC sees aligned reference/near-end frames.
- The journal writer is a dedicated thread with buffered, periodic fsync. **This is the no-loss guarantee.**
  The DSP/drain thread never waits on it: blocks pass through a bounded in-memory queue (≥ 30 s of audio), because fsync stalls of 30–240 ms were measured on NVMe in M0 and can reach seconds on slow or scanned disks (see [m0-report.md](m0-report.md)).
- Journal storage format: 16-bit PCM at 48 kHz costs ~989 MiB/h for mic + stereo system; journal at 16 kHz and/or FLAC segments (decided by measurement in M1/M2).
- The ASR / diarization worker pool consumes from bounded queues. When it falls behind, the policy is: widen partial intervals → switch to a smaller model tier → defer to catch-up from the journal. Audio is never dropped.

### Mic / system synchronization
- Each callback buffer is stamped with host time at the device position (QPC / `mach_absolute_time` / `CLOCK_MONOTONIC`).
- The drift estimator (windowed linear regression of device frames vs host time) drives the async resampler, so both streams land on one 16 kHz timeline.
- Gaps (device glitch or switch) are filled with flagged silence so timestamps never compress.

### Streaming ASR policy (partial vs final)
- Sliding window over the current speech region (e.g. 1–2 s hop, ≤ 30 s window).
- A **partial** is the newest hypothesis. A token prefix becomes **final** once N consecutive hypotheses agree, or the VAD end-of-speech fires.
- Finals are immutable, except an explicit `revision` event emitted after global re-clustering changes speaker labels.
- Word timestamps from whisper DTW / CTC alignment, mapped to the session clock.

### Multilingual and code-switching
- LID runs per VAD segment (short segments use the language prior from neighbouring segments).
- Router:
  - en / most languages → Whisper (language forced to the LID result when confidence is high, else auto).
  - Gujarati / other Indic → IndicConformer when installed (Whisper otherwise).
  - Hindi → whichever engine wins the benchmark.
- Code-switching: intra-sentence switching is decoded by Whisper in auto mode with an Indic-aware prompt. **Word-level switching accuracy is a documented, measured limitation, not a promise.**

### Diarization
- First split by source: mic and system are diarized independently, and their speaker ids never collide.
- Mic path: usually one person, but a conference room can have several, so the same algorithm runs there.
- Online assignment by cosine distance to speaker centroids with a new-speaker threshold.
- Every ~60 s (and at the end): global re-clustering over all embeddings, with Hungarian matching to keep ids stable, and `revision` events for changed labels.
- Overlap: seg-3.0 powerset regions with two active speakers are emitted as overlapping turns (both speakers, overlap flag).
- Speaker profiles (opt-in, local): enrolled centroids matched across sessions.

## 4. Public SDK surface (sketch)

```c
sd_engine*  sd_engine_create(const sd_config* cfg, sd_error* err);
int         sd_devices_list(sd_engine*, sd_device_kind, sd_device_info** out, size_t* n);
sd_session* sd_session_start(sd_engine*, const sd_session_config*, sd_error* err);
int         sd_session_poll_event(sd_session*, sd_event* out, uint32_t timeout_ms); // or callback
int         sd_session_set_device(sd_session*, sd_source, const char* device_id);
int         sd_session_pause(sd_session*, sd_source);  /* resume / stop / finalize */
void        sd_session_free(sd_session*);  void sd_engine_free(sd_engine*);
```

Event types: `TranscriptPartial`, `TranscriptFinal`, `TranscriptRevision`, `SpeakerAssigned`, `LanguageDetected`, `DeviceChanged`, `DeviceLost`, `Level`, `Health` (xruns, queue depth, RTF), `Error`.
Idiomatic wrappers: Rust `async` stream, Node `EventEmitter` / async iterator, Python iterator.

## 5. Repository layout

```
sonicdiar/
  crates/ sd-core  sd-capture  sd-dsp  sd-journal  sd-asr  sd-lang  sd-diar  sd-engine  sd-ffi
  bindings/ node/  python/
  tools/ sdctl (CLI: devices, record, transcribe, bench)
  models/ manifest.json (URLs, sha256, licences; weights not committed)
  bench/ datasets manifests, harness, reports
  docs/  .github/workflows/
```

## 6. Validation plan (acceptance metrics)

| Component | Metric | Initial target (to confirm after baselining) |
|---|---|---|
| Capture | dropped frames / xruns over 8 h soak; mic↔system drift after 2 h | 0 lost frames written to journal; residual skew < 20 ms |
| Device recovery | time to resume after unplug / default switch | < 1 s, no crash, gap flagged |
| AEC | ERLE on synthetic echo (system→speaker→mic simulation, real room IRs); double-talk preservation | ERLE ≥ 25 dB steady state; near-end not suppressed in double-talk |
| NS | DNSMOS / SI-SDR gain on noisy speech | improvement without WER regression |
| VAD | frame F1 vs labels | ≥ 0.9 on test set |
| ASR | WER/CER on FLEURS (en, hi, gu …) + code-switched set; partial latency p50/p95; final latency | compare against batch Whisper baseline; partial p95 < 1.5 s on mid-tier CPU |
| Diarization | DER (with/without overlap) on AMI / VoxConverse subset; speaker-id stability | report vs pyannote-3.1 reference |
| Resources | CPU %, RSS, RTF on a "low-end" profile (4 cores, 8 GB, no GPU) | real-time with tiny/base models; RSS < 1 GB excluding model |

## 7. Milestones (each ends with measured results before moving on)

> **Superseded from M3 on by [roadmap.md](roadmap.md)** (2026-10-13). It adds an M2.5 hardening step and splits Linux (M6) from macOS (M7). It is based on the M0–M2 results.

| # | Deliverable | Where it can be validated |
|---|---|---|
| M0 | Workspace, core types, ring buffers, clock, journal, CI skeleton, `sdctl` | All OSes in CI |
| M1 | Windows capture: mic + loopback, device enumeration, hot-plug, default-follow, drift correction | This Windows machine (real devices) |
| M2 | DSP: AEC3 / NS / AGC2 / HPF + Silero VAD; sonora vs C++ WebRTC benchmark | Offline test vectors + this machine |
| M3 | Streaming whisper ASR: partial/final, word timestamps, model tiers, recovery | FLEURS subsets, latency bench |
| M4 | Diarization: online + global re-cluster, overlap, profiles | AMI / VoxConverse subsets |
| M5 | Multilingual: LID, router, IndicConformer engine, code-switch eval | FLEURS hi/gu + code-switched set |
| M6 | Linux (PipeWire/Pulse) and macOS (CoreAudio, process tap, SCK) backends | Needs real Linux/macOS machines or CI runners with virtual devices |
| M7 | C ABI freeze, Node/Python bindings, packaging, signed release workflows, full docs, benchmark report | CI matrix |

## 8. Known constraints and risks

- **Test hardware:** only Windows is available locally. Linux/macOS capture needs real machines, VMs with audio, or CI runners. GitHub-hosted runners have no sound devices, so tests there use virtual devices (PulseAudio null sink, BlackHole) and cannot prove real-hardware behavior.
- **macOS taps** are a young API with reported issues (e.g. all-zero buffers on a 26.x beta); needs an SCK fallback and health detection.
- **AEC with loopback reference** works when the far-end is played through the captured render device. Audio played through a *different* device than the one captured cannot be cancelled. This will be documented.
- **Gujarati / code-switching quality** depends on available models; it will be reported as measured WER, not asserted.
- **Model sizes vs low-end hardware:** IndicConformer-600M is heavy; it may need int8 quantization and may not run in real time on low-end CPUs (to be measured).
