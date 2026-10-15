# M4 report: speaker diarization

Date: 2026-10-14. Plan: [roadmap.md](roadmap.md) §M4.

All numbers were measured on this laptop's CPU (i5-12450HX). The test data is the **AMI Meeting Corpus** (CC BY 4.0):
- 4 meetings of the standard test split (ES2004a, IS1009a, TS3003a, EN2002a);
- 92 minutes in total, 4 speakers each;
- Mix-Headset audio, scored against the standard "only_words" reference RTTMs (BUTSpeechFIT/AMI-diarization-setup).

DER (diarization error rate) = (missed speech + false alarm + speaker confusion) ÷ reference speech. Our scorer uses 10 ms frames, a 0.25 s collar and the best one-to-one speaker mapping. Overlapped speech is scored unless stated otherwise.

## 1. Summary

- **Offline:** speech and overlap from pyannote segmentation-3.0, WeSpeaker ResNet34 embeddings on 3 s windows, and average-linkage clustering cut at 1.0. **DER 15–16%**, of which speaker confusion is 3.8–4.7%. This compares with **40.5% for a plain baseline** (Silero VAD, 1.5 s windows, the same clustering) measured the same way.
- **Live:** every final transcript gets a speaker as soon as it is transcribed. Re-clustering every 60 s (and at the end) corrects labels while keeping ids stable, emitting `SpeakerRevision` events. On transcribed speech, confusion is **6.9% as first shown and 5.6% after re-clustering**.
- **Speaker changes and overlap in transcripts:** a final's text is split where the speaker changes (by word timings), and stretches where a second person talks at the same time are marked. On AMI with Whisper transcripts this lowers DER from **31.2% to 26.0%** (confusion 11.2 → 7.2).
- **Cost:** about **0.03× real time** offline, including segmentation, and 0.02× live.
- **Separate sources:** mic and system audio are diarized separately.

## 2. Components (`crates/sd-diar`)

| Part | What | Source / licence |
|---|---|---|
| `fbank` | Kaldi-compatible 80-band log-mel features (25/10 ms, Hamming, pre-emphasis 0.97, dither 0, Kaldi mel scale 20 Hz–8 kHz), mean-normalised. These are WeSpeaker's front-end settings | implemented from the Kaldi definition |
| `embed` | WeSpeaker **ResNet34-LM** (256-d), L2-normalised | VoxCeleb-trained, **CC BY 4.0** (attribution required); ONNX from k2-fsa/sherpa-onnx |
| `cluster` | Average linkage (nearest-neighbour-chain, tested equal to textbook UPGMA). Euclidean distance between unit vectors. Small-cluster absorption | — |
| `segment` | pyannote **segmentation-3.0**: speakers per 16.875 ms frame (powerset over 3 local speakers), averaged over 10 s windows every 2.5 s | **MIT** (© CNRS); onnx-community export, output identical to the sherpa-onnx export on AMI |
| `online` | Live per-source diarizer with periodic re-clustering and stable ids; per-utterance turns and overlaps | — |
| `der` | DER (collar, overlap option, Hungarian mapping) and RTTM I/O | — |

The models are pinned by SHA-256 in `models/manifest.json` (group `diar`).

## 3. Offline results (AMI, 92 min)

| Configuration | Miss | FA | Confusion | **DER** |
|---|---|---|---|---|
| Baseline: Silero VAD, 1.5 s windows, cut 1.0, no small-cluster rule | 22.4 | 0.7 | 33.7 | 56.8% |
| + small-cluster rule (min(12, ⌈n/10⌉) windows) | 22.4 | 0.7 | 17.4 | 40.5% |
| Silero VAD, cut 1.2 | 22.4 | 0.7 | 4.3 | 27.3% |
| Segmentation-3.0 speech, cut 1.2 | 14.7 | 1.3 | 5.4 | 21.4% |
| + overlap (second speaker on overlap frames) | 8.4 | 2.1 | 7.4 | 17.8% |
| **+ 3 s windows every 1.5 s, cut 1.0** | 9.1 | 2.0 | 3.7 | **14.7%** |
| Same, absorbing small clusters only within 1.2 (product setting) | 9.3 | 2.0 | 4.7 | 16.0% |
| Reference: oracle speech + overlap (1.5 s windows, cut 1.2) | 5.3 | 0.8 | 9.7 | 15.8% |

**Findings:**
- **Window length matters most.** Clustering is only as good as the per-window embeddings:
  - **3 s windows beat both 1.5 s and 6 s.** With 3 s windows, the baseline's cut of 1.0 (cosine ≈ 0.5) is the best threshold, and 1.0–1.05 is a stable plateau (14.7–14.8%).
  - **2 s windows:** best 17.9%.
  - **CAM++ model** (instead of ResNet34): 18.7%.
- **The small-cluster rule is essential.** Without it, a 4-speaker meeting splits into hundreds of clusters.
- **Remaining error is mostly missed overlapped speech** (9%): only two speakers per frame are labelled, and the second is a heuristic.
- **Giving the true speaker count is worse** (29.5%), because the small-cluster rule then has fewer slots. The product uses the threshold.

## 4. Live diarization

How it works:
- **Live assignment.** Each final transcript's audio is cut into 3 s windows. The utterance joins the nearest speaker centroid within distance 1.0, otherwise it starts a new speaker. Each window also gets its own speaker, so an utterance can be split where the speaker changes. Utterances too short for a window take the previous speaker.
- **Re-clustering.** Every 60 s of audio, and at the end of the session, all windows are re-clustered. The new clusters are matched to the current ids by maximum shared windows (Hungarian), so "S2" stays "S2". Finals whose speaker changes produce `AsrEvent::SpeakerRevision`.
- **Small-cluster absorption** only joins a large cluster closer than **1.2**. On a recorded session where 10 different voices each spoke for 7–16 s, the always-absorb rule merged them into 4 speakers; with the limit, 7 remain, and 4 revisions instead of 11. On AMI the limit costs 0.1 point of live DER.

**AMI, replayed utterance by utterance** (utterances = Silero VAD regions split at 28 s, as the streaming ASR produces them):

| | DER | Confusion |
|---|---|---|
| One speaker per utterance, as emitted | 34.4% | 11.4% |
| One speaker per utterance, after re-clustering | 33.0% | 10.0% |
| **Split at speaker changes, as emitted** | 30.0% | **6.9%** |
| **Split at speaker changes, after re-clustering** | 28.6% | **5.6%** |

- **Relabelled:** 9.6% of the speech got a different speaker after re-clustering.
- **Most of the live DER is missed speech** (about 22%) that the live VAD never transcribes, mainly short backchannels and overlap. No transcript exists to label there. For transcripts, the meaningful figure is the confusion.

### Speaker changes and overlap within a final

- **Word timings.** Whisper gives token timestamps (a token starting with a space starts a word). IndicConformer gives the CTC frames of each word, carried through the number rewriting. Both are 10–40 ms resolution.
- **Text split.** Each word takes the speaker of the 3 s window covering its midpoint. Consecutive words of one speaker form a part. A part of fewer than 3 words and under 1.5 s is merged into a neighbour, so one stray window does not break a sentence. Part boundaries lie midway between words and inside the final. Chinese text is joined without spaces.
- **Overlap.** pyannote segmentation-3.0 runs on each final's audio. Frames with two or more active speakers get the nearest other speaker in time (within 5 s) as the second speaker. These come out as `overlaps` on the final.
- **Revisions.** When re-clustering changes an emitted final's speaker, parts or overlaps, a `SpeakerRevision` carries all three again. Changes that alter nothing shown are not reported.

**AMI, with transcripts** (Whisper large-v3-turbo on each utterance; DER over the non-empty finals, after re-clustering, overlaps included):

| | Miss | FA | Confusion | **DER** |
|---|---|---|---|---|
| One speaker per final | 18.7 | 1.3 | 11.2 | 31.2% |
| **Text split at speaker changes** | 17.3 | 1.4 | **7.2** | **26.0%** |
| Reference: window-level labels, no text (all utterances) | 17.1 | 1.5 | 7.1 | 25.7% |
| Reference: window-level labels, no overlap marks | 22.4 | 0.7 | 5.6 | 28.6% |

- **100 of the 1,194 finals were split.** The merging of short parts costs only 0.3 points compared with the raw window labels.
- **Overlap marks cut missed speech** from 22.4% to 17.1%, at the cost of 1.5 points more confusion (the second speaker is a guess).
- **Live session replay** (below, single speakers in turn): no final was split and no overlap was marked, so the split does not invent speaker changes on clean speech.
- **Cost:** live diarization with overlap detection took 164 s of CPU for the 92 min (0.03× real time), against 0.02× without it.

**Live session** (multilingual playlist of 10 voices through the speakers, plus the user on the mic):
- Live labels mostly separated the voices: Marathi, Hindi, Tamil, Bengali, Chinese and English speakers got distinct ids.
- The user's clean mic lines were consistently Mic-S1.
- After re-clustering, two clips of 2–3 windows each stayed merged with another voice.

## 5. Not done in M4 (open items)

1. ~~Overlap in live transcripts~~ and 2. ~~splitting a final's text at speaker changes~~: done (§4). The overlap's second speaker is the nearest other speaker in time, which is a heuristic; a proper per-frame speaker assignment would need the segmentation model's local speaker tracks matched to the embeddings.
3. **Mic finals dropped as far-end bleed** are embedded before they are dropped. They should be removed from the mic diarizer.
4. **Second dataset:** VoxConverse (CC BY 4.0) DER, and a multilingual or Indian conversational set if a permissively licensed one exists.
5. **Opt-in local speaker profiles** for cross-meeting identity (roadmap: optional).
6. **Speaker naming** in the product (e.g. "You" for a single mic speaker) is left to the SDK user.

## 6. How to run

Models:
```bash
cargo run --release -p sdctl -- models fetch --group diar
```

AMI data (CC BY 4.0, about 180 MB):
```bash
python tools/testdata/fetch_ami.py
```

Offline DER sweep (thresholds, windows, overlap):
```bash
cargo run --release -p sdctl -- diar eval --overlap
```

Live replay (add `--overlap` for overlap marks, and `--asr-model models/ggml-large-v3-turbo-q5_0.bin --gpu` to score the transcript split):
```bash
cargo run --release -p sdctl -- diar online-eval --assign 1.0 --absorb-max 1.2 --overlap
```

Transcripts with speakers: add `--diarize` to `sdctl record --process ...` or `sdctl process ...`. Parts are printed as `| S2 …` lines and overlaps as `+ S3 also speaking …`; the TSV (`--transcript`) has one row per part and an `overlap` column. `--no-overlap` skips overlap detection.
