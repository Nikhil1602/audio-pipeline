# Number-phrase tables

Used by `sd_asr::itn` to write spoken numbers as digits. Built by
`tools/testdata/build_number_tables.py`; each table's header names its sources.

Format: `phrase<TAB>value<TAB>kind<TAB>unit`.
- `n`: a complete number.
- `p`: a combining form that must be followed by a number smaller than `unit`. For example Tamil "இரண்டாயிரத்து" = 2000 + …, or Hindi "उन्नीस सौ" = 1900 + ….

| Source | Used for | Licence |
|---|---|---|
| AI4Bharat indic-numtowords `167927b2` (generator run and inverted) | all 11 tables | MIT (`LICENSE-MIT-AI4Bharat.txt`) |
| Unicode CLDR release-48-2 `common/rbnf/{hi,gu}.xml` | Hindi and Gujarati 0–99 and the hundred word | Unicode License v3 (`LICENSE-Unicode.txt`) |
| NVIDIA NeMo-text-processing `ddadfb2a` ITN data | Hindi variants; Marathi 0–99 and the hundred word | Apache-2.0 (`LICENSE-Apache-2.0.txt`) |
| Wikidata numeral lexemes | Bengali 0–99 | CC0 1.0 |

Copyright notices:
- AI4Bharat indic-numtowords: © 2022 AI4Bhārat.
- Unicode CLDR data: © Unicode, Inc.
- NeMo-text-processing: © NVIDIA CORPORATION & AFFILIATES.

The tables are derived data (phrase and value columns); no source code from these projects is included.

**Build checks:**
- A phrase that maps to two different numbers is dropped. This removed AI4Bharat's Kannada generator errors, e.g. "ಇನ್ನೂ" ("still") produced for 210–290, and a Malayalam 53/54 duplicate.
- A single word for a value of 10 or more that is among the 500 most frequent words of the language's text sample is dropped. For example Urdu "اسی" (80, also "the same") and "بہتر" (72, also "better").

**Enabled languages** (`sd_asr::itn::ENABLED`), chosen by measurement on FLEURS dev and test (docs/m3-report.md):
- **Enabled:** Hindi, Bengali, Gujarati, Marathi, Punjabi, Tamil, Telugu, Malayalam, Odia, Urdu.
- **Not enabled: Kannada.** Its generated forms use the "thousand-of" form as a round thousand, producing wrong values (20% of written numbers suspicious, against 0–14% elsewhere). The table is kept for when a native speaker has checked it.
