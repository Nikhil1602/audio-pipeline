# M3 report: streaming speech-to-text for the 15 product languages

Date: 2026-10-14. Plan: [m3-plan.md](m3-plan.md). All numbers were measured on this laptop:
- CPU: i5-12450HX, 12 GB RAM;
- GPU: RTX 2050, 4 GB, through Vulkan.

The main test set is FLEURS (CC-BY 4.0): **100 test clips per language**. Tuning used only the separate dev split.

## 1. Summary

- **All 15 languages transcribe**, with Auto as the default:
  - **Whisper large-v3-turbo** (q5, GPU) handles English, Chinese, Spanish and French.
  - **IndicConformer-600M** (MIT, ONNX Runtime, CPU) handles the 11 Indian languages, Odia included.
- **Auto picks the right language on 99.2% of single clips** across all 15 languages. In simulated meetings it is right **97–100% of the time for every language**, including switches between languages.
- **Live latency** in a real recording:
  - **final text arrives 1.8 s after speech ends** (median; 95th percentile 3.1 s);
  - **partial text arrives within 0.5 s** (median).
- **Indian-language accuracy has three layers:**
  - **base model:** greedy CTC decoding;
  - **language model:** a word trigram per language, built from public-domain text;
  - **numbers as digits:** for 10 of the 11 (all but Kannada).

  Hindi is at **6.8% WER / 2.9% CER**, and 10 of 11 languages have CER of 3.6–6.1%.
- **Licences:** every component shipped is MIT, CC0, Unicode or Apache-2.0. Indic-Transcribe was evaluated but rejected because of its licence (§7).

## 2. Engines and routing

| Languages | Engine | Why |
|---|---|---|
| English, Chinese, Spanish, French | Whisper large-v3-turbo q5_0 (whisper.cpp, Vulkan GPU) | WER 4.5% / CER 8.2% / 3.2% / 6.0%; language ID 100% |
| Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Odia, Urdu | IndicConformer-600M multilingual, CTC + word LM + digits | 3–6× fewer errors than Whisper (table below); Whisper cannot do Odia |

**Whisper turbo on Indian languages** (FLEURS test, true language given), WER: Bengali 102.9%, Gujarati 96.7%, Hindi 32.5%, Kannada 89.3%, Malayalam 116.5%, Marathi 84.5%. **Whisper is therefore never used to transcribe Indian languages.**

## 3. Indian-language accuracy (FLEURS test, 100 clips per language)

WER here is "lenient": spelling variants that read the same are folded (Unicode composition, zero-width joiners, nukta, chandrabindu, native digits). A different word still counts as an error. CER is the error rate per character.

| Language | Start (greedy CTC) | + language model | + numbers as digits | CER (final) |
|---|---|---|---|---|
| Hindi | 8.7% | 7.6% | **6.8%** | 2.9% |
| Punjabi | 13.5% | 11.9% | **11.0%** | 3.8% |
| Bengali | 15.2% | 14.6% | **13.6%** | 3.7% |
| Urdu | 17.1% | 15.7% | **14.8%** | 5.0% |
| Kannada | 16.1% | **15.3%** | not enabled (§5.3) | 3.9% |
| Gujarati | 18.2% | 17.4% | **16.8%** | 5.1% |
| Odia | 20.3% | 18.2% | **17.7%** | 4.9% |
| Marathi | 20.3% | 18.5% | **18.3%** | 5.4% |
| Telugu | 20.7% | 20.0% | **18.6%** | 4.5% |
| Malayalam | 21.8% | 21.1% | **20.2%** | 4.4% |
| Tamil | 33.3% | 30.8% | **30.8%** | 14.6% |

**Reading these numbers:**
- **Word error rate overstates errors in languages that build long words.** Malayalam, Tamil, Telugu and Kannada join words, so one wrong letter makes the whole word wrong. CER is the better measure there.
- **The Tamil test set contains repeated readings.** 4 of its 100 clips hold the sentence read 3–5 times. The model transcribes every reading, but the reference contains one, so the extra words all count as errors. Without those 4 clips, Tamil greedy WER is 25.6% rather than 33.3%. Much of the rest comes from sandhi consonant doubling ("கருத்தைத் தவிர" vs "கருத்தை தவிர").
- **The digits column** applies to the 10 languages where numbers are written as digits (§5.3).

Speed: **0.08–0.1× real time on the CPU** (8 threads), with or without the language model. The process uses about 2.4 GB of memory, mostly the f32 encoder weights.

## 4. Auto language identification

### 4.1 Two stages

1. **Stage 1, Whisper:** language ID over the languages it knows. It separates English, Chinese, Spanish and French from the Indian languages perfectly. Within the Indian group it is weak (Kannada, Telugu → Tamil; Punjabi → Hindi; 0% on Odia):

   | Language | Whisper turbo language-ID accuracy |
   |---|---|
   | Hindi | 86% |
   | Urdu | 52% |
   | Gujarati | 36% |
   | Marathi | 31% |
   | Bengali | 21% |
   | Kannada, Malayalam, Telugu, Punjabi, Odia | 0% |
   | Tamil, English, Chinese, Spanish, French | 100% |

2. **Stage 2, IndicConformer:** if the Indian share of stage 1 is at least 0.2, IndicConformer redistributes it. Its vocabulary is per language, so each language's share of the output identifies the language. Among the Indian languages alone it scores 94–100% (Punjabi 94%, Gujarati 98%, the rest 100%), except Hindi vs Urdu.

### 4.2 Hindi and Urdu

Spoken Hindi and Urdu are split almost at random: IndicConformer labels FLEURS Hindi as 48% Hindi and 50% Urdu. What differs is the script. **Product decision:** Auto writes Hindi (Devanagari). Choosing Urdu explicitly still gives Urdu script.

The first implementation added Urdu's score to Hindi's *before* deciding. That pulled Punjabi (50%) and Gujarati (82%) into Hindi. Deciding first and relabelling Urdu afterwards fixed it: **99.0% per clip** over all 15 languages.

### 4.3 Group-first rule (from the live session)

In the live session (§6), Indian speech containing English words was routed to English. Whisper gave English 0.30–0.69, while the Indian languages held more in total but split eleven ways. Those segments were then transcribed as *English translations*.

The fix: when a non-Indian language leads but the Indian group as a whole is larger, the group wins.

| | Before | After |
|---|---|---|
| FLEURS, per clip | 99.0% | **99.2%** |
| Simulated meetings | — | no language worse; Hindi 97.2% → 98.3% |
| Live code-switched mic line | English translation | **Hindi** |
| Live Bengali clip with loanwords, CER | 63.8% | **28.3%** |

### 4.4 Per meeting

Auto keeps a decaying memory of each source's language, so short or unclear segments lean towards the meeting's language. `tools/eval/lid_rules.py` replays the real per-clip scores through a mirror of that logic:

| Scenario | Accuracy |
|---|---|
| Single-language meetings | 98.3–100% for every language |
| Switches to and from English | 98.1–100% |
| Switches between two Indian languages (Hindi↔Marathi, Bengali↔Odia, …) | 95.7–99.4% (Hindi↔Punjabi is the hardest) |

## 5. Decoder improvements

### 5.1 RNN-T: not adopted

IndicConformer's RNN-T head gains 0.1–1.2 WER points on 10 languages (Odia +0.1) and costs 40% more CPU (RTF 0.15 vs 0.10). It remains available with `--indic-decoding rnnt`.

### 5.2 Word language model: adopted

- **What:** an interpolated modified Kneser-Ney trigram per language, implemented from the published method. Prefix beam search with 32 beams.
- **Training text:** IndicCorp v2 (AI4Bharat, **CC0**). About 400 MB per language, sampled from 200 byte ranges of a pinned revision, because the files are sorted. Any line sharing an 8-word run with a FLEURS test or dev sentence is dropped.
- **Settings:** weight 0.2, word bonus 0.5, unknown-word penalty −5. Tuned on FLEURS dev, where they were at or near the best for almost every language.
- **Size:** 45–115 MB per language, loaded on first use.
- **Pitfall fixed:** hypotheses were gluing words together to postpone their language-model score. The unknown-word penalty is now applied as soon as a partial word can no longer become a vocabulary word.
- **Tried and not adopted** (Tamil and Marathi, dev set):

  | Variant | Tamil | Marathi |
  |---|---|---|
  | Current (400 MB text, 200k words) | 20.4% | 16.2% |
  | 500k-word vocabulary | 20.9% | 16.2% |
  | 1M-word vocabulary | 21.0% | 16.5% |
  | 4× more text (1.6 GB) | 20.4% | 16.3% |
  | 4× more text, 500k words | 20.6% | 16.1% |

  The word-level model has saturated.

### 5.3 Numbers as digits: adopted for 10 of 11 Indian languages

Spoken numbers become digits: "दो हज़ार सात" → 2007, "उन्नीस सौ अस्सी" → 1980, Tamil "இரண்டாயிரத்து ஐந்து" → 2005, "மூணு லட்சத்து அம்பதாயிரம்" → 350000.
- **Parsing:** a number is the longest chain of combining forms ("…thousand-and", "…lakh-and", "nineteen hundred") ending in a complete number.
- **Lone small words stay words:** a single word below 10 is not converted, because "एक" is also "a". The threshold was chosen on dev (1 / 10 / 20 / 100 tried).
- **Unclear runs stay words:** if a run of number words does not read as one number, the whole run is left as words. That covers adjacent numbers ("दस बीस", ten-twenty), leftovers ("17 100 55"), and an inflected last word ("दोन हजार आठपासून", "since 2008").

**Tables** (`crates/sd-asr/data/numbers/`, provenance and licence texts included):
- **Main source:** AI4Bharat **indic-numtowords** (MIT; the makers of IndicConformer). It is a number-to-words generator with colloquial variants (Tamil "இரண்டு"/"ரெண்டு"); it is run and its output inverted.
- **Merged in:**
  - Unicode CLDR (Hindi, Gujarati);
  - NeMo-text-processing (Hindi variants, Marathi);
  - Wikidata (Bengali).
- **Year-style hundreds** ("seventeen hundred") are derived from each language's own hundred word.
- **Urdu** matches both spellings of the nasal (پانچ / پاںچ).

**Checks against bad source data:**
- **Contradictions are dropped:** a phrase mapping to two numbers is removed. This caught Kannada "ಇನ್ನೂ" ("still"), which the generator produced for 210–290.
- **Very common words are dropped:** a single word for a value of 10 or more that is among the 500 most frequent words in the language's text is removed. This caught Urdu "اسی" (80, also "the same") and "بہتر" (72, also "better").

**Measured** (FLEURS, LM on; lenient WER without → with digits):

| | hi | bn | gu | mr | pa | ta | te | kn | ml | or | ur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dev | 9.5→8.9 | 14.4→13.8 | 16.1→15.5 | 16.2→15.6 | 10.6→10.2 | 20.8→20.7 | 19.7→18.2 | 17.7→16.5 | 25.2→24.3 | 18.4→17.9 | 15.2→13.7 |
| Test | 7.6→6.8 | 14.6→13.6 | 17.4→16.8 | 18.5→18.3 | 11.9→11.0 | 30.8→30.8 | 20.0→18.6 | 15.3→14.7 | 21.1→20.2 | 18.2→17.7 | 15.7→14.8 |
| Written numbers not in a digit-bearing reference (dev+test) | 7% | 0% | 14% | 4% | 4% | 8% | 9% | **20%** | 12% | 5% | 7% |

**Reading the last row:** it overstates converter errors. It also counts numbers the recogniser misheard, and correct conversions next to other digits in the reference. Inspected Gujarati and Malayalam cases were mostly correct ("સો ટકા" → 100 %). Their genuine misses are spelling variants that break a longer number ("ઓગણીસ સો પંચાણુ" → "1900 પંચાણુ").

**Kannada is not enabled.** Its generated forms use "ಸಾವಿರದ" ("thousand-of") as a round thousand, which produces wrong values ("ಹದಿನೈದು ಸಾವಿರದ ಒಂಬೈನೂರನಲವತ್ತರಂದು", "15th, 1940", → "15000 …"). It keeps spoken numbers until a native speaker checks the table.

**Ordinals** ("ग्यारहवीं") stay as words.

## 6. Live session (real capture, laptop speakers and mic)

The session lasted 240 s:
- **System audio:** a 10-sentence FLEURS playlist (English, Hindi, Tamil, Bengali, Chinese, two each) played through the speakers.
- **Mic:** the user read 5 Hindi sentences containing English words.
- **ASR:** Auto, Whisper on the GPU and IndicConformer on the CPU, with partials.

| | Result |
|---|---|
| Final latency (speech end → final text) | **p50 1.8 s, p95 3.1 s** |
| Partial latency | **p50 0.5 s, p95 2.1 s** |
| Decoding load | 300 s of audio (both sources) decoded in 98 s |
| Capture | 0 frames dropped; mic–system drift −0.9 ± 0.2 ppm |
| Played clips, right language | CER 0–9% (Chinese 0%, Hindi 2.5%, English 3.4%, Tamil 6.7–9.0%) |

Notes:
- **Unrelated audio at the start.** Something else played loud Marathi speech during the first 13 s. It masked the first English clip, and it was (correctly) transcribed as Marathi.
- **Short segments are the weak spot for language ID.** With the language changing every sentence, segments of about 2 s were still sometimes routed to English (Whisper English 0.69 and 0.42). In a real meeting the session memory reduces this. The memory is also what carried the opening Marathi into the first Hindi clip (CER 26.9%).
- **Replay is exact.** Replaying the recorded journal reproduces the live results, so `target/live3` serves as a regression case.

### Code-switching (Hindi with English words)

- **Language right → good result.** When Auto picks Hindi, IndicConformer writes the English words in Devanagari, as Hinglish is commonly written in that script. A clean reading of line 2 was word-perfect: "डेडलाइन अगले फ्राइडे की है इसलिए टेस्टिंग इसी हफ्ते पूरी करनी होगी". Line 1 had one inserted word.
- **Language wrong → serious failure.** Whisper translates the sentence into English, losing what was said. The group-first rule (§4.3) addresses this.
- **Decided by the owner (2026-10-14):** English words inside Indian-language speech are written in that language's script ("मीटिंग", "डेडलाइन"), which is what IndicConformer produces.

## 7. Real recordings and processing settings (M2 open questions)

The earlier double-talk recordings were replayed through the full pipeline (`tools/eval/recording_wer.py`):
- `dt2`: raw mic, our echo canceller active;
- `dt1`: driver echo cancellation.

| Setting | `dt2` mic WER | `dt1` mic WER | System (far end) WER |
|---|---|---|---|
| **Default:** echo gate auto, blended output (20 dB), no noise suppression | **10.9%** | 45.2%* | 0% |
| Noise suppression moderate / high | 10.9% / 10.9% | 48.4% / 48.4% | 0% |
| Echo cancellation forced on: full / blended / linear output | 20.0% / 12.7% / 16.4% | 45.2% / 45.2% / 53.2% | 0% |
| Echo cancellation off | 14.5% | 45.2% | 0% |

\* `dt1` includes speech that was not in the script (an introduction, remarks between readings). It is transcribed correctly but counts against the script.

`dt2`'s remaining 10.9% is mostly reading deviations ("we are reviewing" for "we reviewed"); it is not separable without a hand transcript.

**Conclusions:**
- **The defaults are the best setting.** The fully suppressed output doubles mic errors, and the linear output lets echo through (an inserted "Oh, my gosh").
- **Noise suppression does not help recognition** on these recordings, so it stays off by default.
- **The far end is transcribed perfectly** in every setting.

## 8. Evaluated and rejected

- **Indic-Transcribe-core** (Bodhan AI, 1.2B parameters, PyTorch only):
  - Accuracy, on the 7 languages it finished: 0.9–4.4 WER points better than IndicConformer greedy.
  - Licence: the Indic Open Model License restricts hosted use (written approval needed) and requires a separate licence above 500M monthly users or $250M revenue.
  - Cost: 2× the size, needs a GPU.

  The owner chose IndicConformer for its MIT licence. Evaluation only; nothing from it is shipped.
- **Bigger vocabularies and more LM text:** no gain (§5.2).
- **Whisper small / medium / large-v3:** the sweep was stopped. Turbo-q5 already gives 3–8% on the four non-Indian languages and 100% language ID; small remains a candidate CPU fallback.

## 9. Licences and data

| Component | Licence | Notes |
|---|---|---|
| Whisper models (ggml) | MIT | |
| IndicConformer | MIT | Gated download, pinned by SHA-256 in `models/indicconformer.lock.json` |
| IndicCorp v2 (LM training text) | CC0 | |
| Number tables (CLDR, NeMo, Wikidata) | Unicode / Apache-2.0 / CC0 | Licence texts and provenance in `crates/sd-asr/data/numbers/` |
| FLEURS | CC-BY 4.0 | Evaluation only, not shipped |

## 10. Open items

1. **Short-segment language ID** (~2 s segments): consider merging very short segments with neighbours before deciding, or relying more on the session memory for them.
2. **Hindi↔Punjabi switching** (95.7%) is the weakest pair in the meeting simulation.
3. **Kannada number table:** needs a native-speaker check before enabling (§5.3). Spelling variants that break year-style numbers (Gujarati "પંચાણુ", Malayalam "പത്തൊമ്പതി…") could be added the same way.
4. **Marathi with the merged table** gains less on test (18.5 → 18.3) than the earlier NeMo-only table (→ 17.9), with dev equal (15.6). The inflection rule now leaves some runs as words; worth a look with more data.
5. **IndicConformer memory** (about 2.4 GB, f32 encoder): try int8 quantization for low-RAM machines, measuring accuracy.
6. **Model tiers by hardware** (planned in M3) are not implemented yet. This laptop runs Whisper turbo on its GPU and IndicConformer on its CPU. CPU-only and low-RAM machines need a measured smaller tier, for example Whisper small, whose language ID is better than turbo's on several Indian languages but whose speed was not measured on CPU, plus a quantized IndicConformer.
7. **Offline replay ASR timing** reports large "latency" because replay runs faster than real time. Only the live numbers (§6) are latencies.
