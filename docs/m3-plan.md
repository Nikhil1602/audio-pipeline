# M3 plan: speech-to-text for the 15 product languages

Date: 2026-10-14.

**Required languages** (the product's language picker; **Auto is the default**): Auto, English, Chinese, Spanish, French, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Odia, Urdu.

## 1. Verified coverage

All entries below were checked at the source on 2026-10-14:
- Whisper: whisper.cpp's language table.
- IndicConformer: the Hugging Face model repository (files and licence).
- VoxLingua107: the model's label file.
- MMS-LID: model card licence.
- FLEURS: dataset splits.

| Language | Code | Whisper (MIT) | IndicConformer-600M multilingual (MIT) | VoxLingua107 LID (Apache-2.0) | FLEURS test set |
|---|---|---|---|---|---|
| English | en | ✓ | ✗ | ✓ | en_us |
| Chinese | zh | ✓ | ✗ | ✓ | cmn_hans_cn |
| Spanish | es | ✓ | ✗ | ✓ | es_419 |
| French | fr | ✓ | ✗ | ✓ | fr_fr |
| Bengali | bn | ✓ | ✓ | ✓ | bn_in |
| Gujarati | gu | ✓ | ✓ | ✓ | gu_in |
| Hindi | hi | ✓ | ✓ | ✓ | hi_in |
| Kannada | kn | ✓ | ✓ | ✓ | kn_in |
| Malayalam | ml | ✓ | ✓ | ✓ | ml_in |
| Marathi | mr | ✓ | ✓ | ✓ | mr_in |
| Punjabi | pa | ✓ | ✓ | ✓ | pa_in |
| Tamil | ta | ✓ | ✓ | ✓ | ta_in |
| Telugu | te | ✓ | ✓ | ✓ | te_in |
| **Odia** | or | **✗** | ✓ | **✗** | or_in |
| Urdu | ur | ✓ | ✓ | ✓ | ur_pk |

**What this means:**
- **No single engine covers all 15.** Whisper lacks Odia; IndicConformer has no English, Chinese, Spanish or French. The design is therefore **two engines with a router**, as already planned in the architecture proposal.
- **No permissively licensed language identifier covers all 15.** VoxLingua107 lacks Odia. Meta's MMS-LID covers Odia but is licensed CC-BY-NC-4.0 (non-commercial), so it is excluded.
- **IndicConformer is distributed as ONNX** (encoder, CTC decoder, RNN-T parts) with `language_masks.json`, which restricts its shared vocabulary to one language. Running the encoder once and scoring the masked CTC output for each Indic language is a candidate way to identify **which Indic language** is spoken, Odia included. This must be measured, not assumed.
- **FLEURS (CC-BY-4.0) has test sets for all 15 languages**, so every claim can be measured on the same benchmark.

## 2. Design

```
VAD segment (16 kHz mono, per source)
   │
   ├─ language fixed by the user ──────────────► engine for that language
   │
   └─ Auto (default):
        stage 1  15-way acoustic LID (Whisper's LID restricted to the 14 languages it knows,
                 or VoxLingua107 restricted the same way; chosen by measurement)
        stage 2  if stage 1 says an Indic language, or is unsure: IndicConformer encoder once
                 + masked CTC score per Indic language (adds Odia, separates confusable pairs)
        prior    session language prior: a meeting rarely switches language every segment,
                 so short or uncertain segments use their neighbours' language
   ▼
 router:  en / zh / es / fr                → Whisper (whisper.cpp)
          Indic languages, including Odia  → IndicConformer (CTC first; RNN-T if it is clearly
                                              more accurate and fast enough)
          Hindi / Urdu / others             → whichever engine wins on WER in this table
   ▼
 streaming policy: hypotheses every 1–2 s over the open VAD segment → partials (stable prefix)
                   → final at end of speech; word timestamps on the session clock
   ▼
 mic-bleed safety net: mic segments flagged `echo_unverified` (M2.5) are dropped if their text
                       matches the system transcript within ±4 s
```

**Code-switching (Hindi–English and similar).**
- Whisper decodes mixed speech in one pass when given the dominant language.
- IndicConformer's vocabulary is per language, so English words come out transliterated.
- Which is better for mixed segments is measured on recorded code-switched speech, not assumed.

## 3. Steps

1. **M3a, measurement before choosing.**
   - **Data:** FLEURS test subsets for all 15 languages (about 100 utterances each).
   - **Whisper via whisper.cpp:** small, medium, large-v3-turbo; CPU and RTX 2050 (CUDA or Vulkan).
   - **IndicConformer via ONNX Runtime:** CTC and RNN-T, CPU.
   - **Report:**
     - WER (CER for Chinese and for scripts where words are ambiguous) per language and engine;
     - real-time factor and memory;
     - LID accuracy for stage 1, stage 2 and combined, on 15-way FLEURS, Odia included;
     - the routing table that results.
2. **M3b, streaming engine in the SDK.** `sd-asr` crate: segment queue from the processing stage, partial and final events, word timestamps, model tiers by hardware, a watchdog, and the mic-bleed safety net.
3. **M3c, real recordings.** WER on our recorded sessions with known text (A, A2), with and without noise suppression, and with each AEC output, to settle the M2 open questions.

## 4. Needed from the owner

- **IndicConformer download access.** The repository is "gated (auto)": it is free and approved instantly, but each user must accept the terms on Hugging Face while logged in, and downloads need that account's access token.
  - Once accepted, a read-only token in the `HF_TOKEN` environment variable lets `sdctl models fetch` download and SHA-pin the files like the other models.
  - The token is never stored in the repository.

## 5. Risks

- **Odia in Auto mode** depends on the stage-2 scoring working. If it measures poorly, picking Odia explicitly is the fallback, and that will be documented.
- **Model sizes.** Whisper large-v3-turbo is about 1.6 GB and IndicConformer-600M about 2.4 GB (to be confirmed). Running both on a 12 GB, 4 GB-VRAM laptop needs measuring; smaller tiers may be required on low-end machines.
- **Accuracy varies widely by language.** Results are reported per language; weak languages are documented, not hidden.
