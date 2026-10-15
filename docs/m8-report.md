# M8 report: SDK and release

Date: 2026-10-15. Plan: [roadmap.md](roadmap.md) §M8. Integration guide: [sdk.md](sdk.md).

## 1. Summary

- **One SDK, four languages.**
  - The Rust crate `sonicdiar` (`crates/sonicdiar`) wraps the engine in a small API:
    - `Session::start(Config)`, `next_event`, `set_device`, `stop`;
    - `list_devices`;
    - `models::fetch/list/verify`.
  - **C** (`crates/sd-ffi`, `sonicdiar.h`), **Python** (`bindings/python`, pyo3) and **Node.js** (`bindings/node`, napi-rs) wrap it with the same configuration and events, as JSON.
- **Stable event model.** Tagged JSON events cover:
  - devices, permissions, speech;
  - partial and final transcripts, with words, speakers, parts and overlaps;
  - speaker revisions, bleed removal, journal and disk state, and health.

  Configurations are JSON with defaults for every field, and typos are rejected.
- **Models are fetched, never bundled.** The manifest and lock files are compiled into the library. Every download is checked against its pinned size and SHA-256 before it is used.
- **Validated:**
  - **On this laptop (Windows, real devices):** the Rust example, the C smoke program (MinGW gcc), Python and Node all record. Live transcription works through Rust and Python.
  - **In CI (`sdk` job):** the C, Python and Node smoke tests run on Windows, macOS and Linux. Linux also records from a virtual device.
- **Release.** Tags `v<version>` build and publish a GitHub Release (§4).
- **Licence.** SonicDiar is **MIT OR Apache-2.0**. All 185 third-party crates it ships are permissive, which `cargo-deny` enforces in CI.

## 2. Found and fixed on the way

- **Hang at stop (engine, since M3).**
  - **Symptom:** the ASR thread only noticed that its input had closed when it had no queued work. At shutdown, voice detection can end a segment slightly past the last audio chunk (its padding). That final waited forever for audio that would never come, so the thread spun at full CPU and `stop` never returned. It happened in about 3 of 4 runs with partials off.
  - **Fix:** the runner now waits on its input whenever nothing can run, and finishes the queue when the input closes. A regression test reproduces the case.
  - **Side effect:** the same loop also busy-waited for audio in normal operation, so live CPU use dropped.
- **Lost transcripts at stop (SDK design).** The transcripts of speech still queued at the stop are produced *during* `stop`. The SDK keeps them readable: `stop` does not consume the session, and every binding's event iterator yields them.

## 3. Results (Windows laptop, real devices)

| Check | Result |
|---|---|
| Rust example, 8 s, no transcription | devices started, health events every second, summary `complete: true` |
| Rust example, English clip through the speakers, Whisper small (CPU), 15 s, three runs | 6–7 finals each, all correct ("Good morning everyone.", "Thanks for joining the weekly planning call.", …), stop completed every time |
| C smoke (gcc, `sonicdiar.dll`) | errors and messages, JSON lists, a real session, double stop rejected: SMOKE PASSED |
| Python smoke + live transcription | SMOKE PASSED; finals with words, including those finished during the stop |
| Node smoke (Node 22) | SMOKE PASSED |

CI (`sdk` job): the C smoke program (compiled with each platform's compiler), the Python wheel built with maturin, and the Node addon pass on Windows, macOS and Linux. On Linux all three also record from a PulseAudio virtual device. The `licenses` job (cargo-deny) passes.

## 4. Release workflow (`.github/workflows/release.yml`)

Pushing a tag `v<version>` (it must equal the version in `Cargo.toml`) builds on Windows x64, Linux x64 (Ubuntu 22.04, glibc 2.35) and macOS (Apple silicon and Intel), then publishes a GitHub Release with:

| File | Contents |
|---|---|
| `sonicdiar-<v>-windows-x64.zip`, `-linux-x64.tar.gz`, `-macos-universal.tar.gz` | the C library (shared, plus static on Linux), `sonicdiar.h`, licences, the smoke example |
| `sonicdiar-<v>-cp39-abi3-*.whl` | Python 3.9+ wheels (macOS universal2) |
| `sonicdiar-<v>.tgz` | the npm package with prebuilt addons for win32-x64, linux-x64, darwin-arm64, darwin-x64 |
| `THIRD-PARTY-LICENSES.md` | every third-party crate and its licence (cargo-about) |
| `sonicdiar-sbom.cdx.json` | CycloneDX software bill of materials |
| `SHA256SUMS` | checksums |

Every file has a **build-provenance attestation**: a keyless Sigstore signature made by the workflow, verifiable with `gh attestation verify <file> -R Nikhil1602/sonicdiar`. This replaces the roadmap's "signed tags". Signing tags themselves needs your GPG or SSH key, which only you have.

**First release: [v0.1.0](https://github.com/Nikhil1602/sonicdiar/releases/tag/v0.1.0)** (all 11 files above; the first attempt failed at the SBOM step, which was fixed and the tag re-created before anything was published). The published Windows wheel was installed into a fresh virtual environment, and the published npm package into a new Node project, after checking their SHA-256 against `SHA256SUMS`; both record from this laptop's devices.

## 5. Not done in M8 (open items)

1. **Publishing to PyPI and npm:** releases are on GitHub only (your decision). Adding the registries later needs their accounts and tokens.
2. **CPU transcription speed.** Done after M8 ([asr-speed.md](asr-speed.md)): shortened encoder windows made Whisper `small` 2.1× faster (RTF 0.73 → 0.35, same accuracy), so it keeps up live on a laptop CPU. large-v3-turbo gained 1.2–1.7× but stays slower than real time without a GPU. Still open: the low-spec benchmark (2 threads, peak RAM).
3. **GPU builds** (Vulkan) are a build option, not a release artifact.
4. **Linux wheels** are tagged `linux_x86_64`, not manylinux, because they depend on the system's libpulse (LGPL), which must not be bundled.
5. **Code signing of binaries:** Windows Authenticode and Apple notarization need paid certificates. Unsigned libraries work when an application loads them; an app that ships them is signed by its own developer.
6. **API stability:** `SD_API_VERSION` 1, and events and configuration at their first version. From now on, changes should only add fields.
