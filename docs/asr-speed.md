# Whisper on a CPU: shorter encoder windows

Whisper's encoder always processes a 30 s window, and the shorter speech in it is padded with
silence. Live utterances last a few seconds, so most of that work is spent on silence. In Auto
mode the encoder ran twice per utterance: once for language detection and once for
transcription. On a laptop CPU this made live transcripts fall behind (M8 open item 2).

## What changed (`crates/sd-asr/src/whisper.rs`)

1. **Transcription encodes the utterance plus 2.56 s** (`ctx_margin` = 128 frames of 20 ms)
   instead of 30 s. whisper.cpp's `audio_ctx` makes this possible.
2. **Language detection encodes 10 s** (`detect_ctx` = 500 frames). It runs on its own
   decoding state, because whisper.cpp keeps the last window size in the state and detection
   has no parameter for it.
3. **A shortened window has one failure mode,** and it is checked for. Without the silence
   after the speech, the decoder sometimes runs on: it repeats the sentence or adds a phrase.
   Those segments are impossible on their own timestamps:
   - they end more than 0.5 s after the audio;
   - or they are faster than 40 characters a second;
   - or they repeat earlier text.

   Such a result is decoded again with the full window.
4. **The shortened window turns itself off for models where it does not pay.** After 8 results,
   if more than half needed the full window, that engine uses the full window from then on.
   large-v3-turbo is such a model, and the engine logs when this happens.
   - With a shortened window, turbo's text is right, but its timestamps run past the audio by roughly 1500/ctx times. For example, a 10.6 s clip ended at 24.6 s.
   - That happened on 24 of 25 English clips, so every result was decoded twice.
5. **Chinese:** an initial prompt steers Whisper to Simplified characters ("以下是普通话的句子。"). Without it, Whisper mixed in Traditional characters, which the references don't use. This change is independent of the window, and it lowered the Chinese CER on the full window from 19.2 % to 15.6 %.

**Defaults:** `WhisperOptions::default()` uses the shortened windows.
- **Turn them off:** `sdctl asr eval --full-window` restores the original 30 s windows.
- **Tune them:** `--ctx-margin` and `--detect-ctx`.

## Measurements

**Test setup:**
- **Data:** FLEURS test set; the first 25 clips per language (10 for some turbo runs; noted below).
- **Mode:** Auto (language detected per clip), greedy decoding.
- **Hardware:** an i5-12450HX laptop CPU with 12 threads, no GPU.
- **Metrics:** WER (CER for Chinese).
- **Speed:** the real-time factor (RTF), compute time divided by audio duration. Below 1 is faster than real time.
- **Fairness:** every comparison uses the same clips.

### Whisper small (`ggml-small.bin`)

| | en WER | es WER | fr WER | zh CER | hi WER | Language ID | RTF |
|---|---|---|---|---|---|---|---|
| Full window (zh prompt) | 7.4 % | 4.8 % | 12.5 % | 15.6 % | 58.6 % | 100 % | 0.732 |
| Shortened (default) | 7.0 % | 5.0 % | 12.9 % | 16.7 % | 58.5 % | 100 % | **0.353** |

- **Speed:** 2.1× faster, and well under real time.
- **Accuracy:** within ±1.1 points in every language. 8 of 125 clips needed the full window.
- **Other margins tried:**
  - 64 frames: RTF 0.372; 13 clips needed the full window; WER es 5.3 %, fr 13.0 %.
  - 32 frames: RTF 0.409; 23 clips needed the full window.
  - A 6 s detection window (300 frames) confused one Hindi clip; 10 s did not.
- **Without the plausibility check,** a 64-frame margin raised the Spanish WER to 10.1 % and the French WER to 17.2 % (run-on text).

### Whisper large-v3-turbo q5 (`ggml-large-v3-turbo-q5_0.bin`, the SDK's default model)

| | en WER | es WER | fr WER | zh CER | hi WER | Language ID | RTF |
|---|---|---|---|---|---|---|---|
| Full window | 5.2 % | 1.7 % | 9.5 % | 9.3 % | 29.6 % | 100 % | 3.6–3.9 |
| Default (detection window only, after the self-check) | 5.2 % | 1.7 % | 9.5 % | 9.3 % | 29.6 % | 100 % | 2.2–3.1 |

Accuracy is identical (10 clips per language, paired).
- **Speed:** 1.2–1.7× faster:
  - Spanish 3.91 → 3.12;
  - French 3.73 → 3.11;
  - Hindi 3.91 → 2.35;
  - Chinese 3.64 → 2.24.

  The English figure includes the 8 probe clips that were decoded twice before the self-check.
- **One discarded run:** a first Chinese run measured 7.7 while the machine was busy with something else. A repeat gave 2.24.
- **Bottom line:** on this CPU, turbo stays 2–3× slower than real time.

## What this means

| Machine | Live transcription |
|---|---|
| CPU only (laptop, 12 threads) | Use Whisper `small`: 0.35× real time, so it keeps up live. The large models fall behind. |
| GPU (Vulkan build) | large-v3-turbo keeps up: final text p50 1.8 s after the speech ends (M3 report §6). |

From these numbers, a short utterance (11 s on average in FLEURS) now takes about 4 s with
`small`, down from about 8 s.

The test page (`sonicdiar-ui`) showed it working live, with Whisper small on CPU:
- **The test:** a 90 s excerpt of an AMI meeting (four speakers), played through the speakers.
- **Transcript:** finals followed the speech, labelled S1–S4.
- **Stopping:** after the stop, the queued speech was finished in about 16 s.
