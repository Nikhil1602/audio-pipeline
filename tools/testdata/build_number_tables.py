"""Build the number-phrase tables used to write spoken numbers as digits (sd_asr::itn).

Sources (pinned, permissively licensed):

* AI4Bharat indic-numtowords (MIT), commit 167927b2: a number-to-words generator with
  colloquial variants for all 11 Indic product languages. It is downloaded into target/ and
  run here; its output is inverted into phrase tables.
* Unicode CLDR release-48-2 rbnf (Unicode License v3): Hindi and Gujarati words for 0-99 and
  their hundred word.
* NVIDIA NeMo-text-processing ITN data (Apache-2.0), commit ddadfb2a: Hindi variants; Marathi
  words for 0-99 and its hundred word.
* Wikidata numeral lexemes (CC0): Bengali 0-99.

Checks, so a wrong entry cannot silently rewrite ordinary words:
1. A phrase that maps to two different numbers is dropped.
2. A single word mapped to a value of 10 or more that is among the 500 most frequent words of
   the language's IndicCorp sample (bench/data/indiccorp/<code>.txt) is dropped: real number
   words for such values are never that common, while generator mistakes such as Kannada
   "ಇನ್ನೂ" ("still") for 250 are. Every dropped entry is reported.
Whether a language's table is enabled is decided by measurement on FLEURS dev and test
(docs/m3-report.md), not here.

Output: crates/sd-asr/data/numbers/<code>.tsv with lines "phrase<TAB>value<TAB>kind<TAB>unit":
  kind n: a complete number (phrase = value);
  kind p: a combining form that must be followed by a number smaller than unit
          (Tamil "இரண்டாயிரத்து" = 2000 + ..., Hindi "उन्नीस सौ" = 1900 + ...).

Usage: python tools/testdata/build_number_tables.py
"""
import collections
import datetime
import io
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "crates", "sd-asr", "data", "numbers")
CORPUS = os.path.join(ROOT, "bench", "data", "indiccorp")
CLDR = "https://raw.githubusercontent.com/unicode-org/cldr/release-48-2/common/rbnf"
NEMO = ("https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/"
        "ddadfb2a38d2bc6b8cc6232c4f915eb60f500688/nemo_text_processing/inverse_text_normalization")
NUMTOWORDS_REV = "167927b24ba9c843a1c62021d47fcbe71f6f177f"
NUMTOWORDS_ZIP = f"https://github.com/AI4Bharat/indic-numtowords/archive/{NUMTOWORDS_REV}.zip"
LANGS = ["hi", "bn", "gu", "mr", "pa", "ta", "te", "kn", "ml", "or", "ur"]
MULTIPLIERS = (1000, 100000, 10000000)
COMMON_WORDS = 500
PUNCT = "\"'()[]{}<>,.;:!?-।॥۔،"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sonicdiar/0.1 (number tables)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def clean(text: str) -> str:
    text = re.sub(r"[​‌‍]", "", unicodedata.normalize("NFC", text))
    return " ".join(text.split())


# ------------------------------------------------------------------ AI4Bharat generator
def load_numtowords():
    dest = os.path.join(ROOT, "target", "numtowords", f"indic-numtowords-{NUMTOWORDS_REV}")
    if not os.path.isdir(dest):
        zipfile.ZipFile(io.BytesIO(get(NUMTOWORDS_ZIP))).extractall(os.path.dirname(dest))
    sys.path.insert(0, dest)
    from indic_numtowords import num2words  # noqa: E402  (pinned, audited: pure Python)
    return num2words


def forms(num2words, n: int, lang: str) -> list:
    out = num2words(n, lang=lang, variations=True)
    out = [out] if isinstance(out, str) else out
    return sorted({clean(f) for f in out if f and f.strip()})


def ai4bharat(num2words, lang: str) -> list:
    """(phrase, value, kind, unit) entries generated from the AI4Bharat converter."""
    entries = []
    for n in range(1000):
        entries += [(f, n, "n", 0) for f in forms(num2words, n, lang)]
    one = forms(num2words, 1, lang)
    for h in range(1, 10):
        for f in forms(num2words, h * 100 + 1, lang):
            entries += [(f[: -len(o) - 1], h * 100, "p", 100) for o in one if f.endswith(" " + o)]
    for unit in MULTIPLIERS:
        for k in range(1, 100):
            entries += [(f, k * unit, "n", 0) for f in forms(num2words, k * unit, lang)]
            for f in forms(num2words, k * unit + 1, lang):
                entries += [(f[: -len(o) - 1], k * unit, "p", unit) for o in one if f.endswith(" " + o)]
    # Year-style hundreds ("seventeen hundred fifty-five"): the generator only makes 1-9
    # hundred, so take the language's own hundred words from its "two hundred (and) ..." and
    # "two hundred" forms, where they are a separate word after the word for two, and attach
    # them to every form of 10-99.
    two = set(forms(num2words, 2, lang))
    combining = {p.split(" ")[1] for p, v, k, _ in entries
                 if k == "p" and v == 200 and len(p.split(" ")) == 2 and p.split(" ")[0] in two}
    terminal = {f.split(" ")[1] for f in forms(num2words, 200, lang)
                if len(f.split(" ")) == 2 and f.split(" ")[0] in two}
    for k in range(10, 100):
        for w in forms(num2words, k, lang):
            entries += [(f"{w} {h}", k * 100, "p", 100) for h in combining]
            entries += [(f"{w} {h}", k * 100, "n", 0) for h in terminal]
    return entries


# ------------------------------------------------------------------ other sources
def cldr(lang: str):
    """Words for 0-99 and the hundred word from CLDR %spellout-cardinal."""
    xml = get(f"{CLDR}/{lang}.xml").decode("utf-8")
    body = re.search(r"%spellout-cardinal:\n(.*?)(?=\n%|\]\]>)", xml, re.S).group(1)
    units, hundred = [], None
    for rule in body.split("\n"):
        m = re.fullmatch(r"(\d+): ([^<>=\[\];]+);", rule.strip())
        if m and int(m.group(1)) < 100:
            units.append((clean(m.group(2)), int(m.group(1))))
        m = re.fullmatch(r"100: << ([^<>=\[\];]+)\[ >>\];", rule.strip())
        if m:
            hundred = clean(m.group(1))
    return units, hundred


def devanagari_int(s: str) -> int:
    return int("".join(str(ord(c) - 0x0966) if 0x0966 <= ord(c) <= 0x096F else c for c in s))


def nemo(lang: str, names: list) -> list:
    out = []
    for name in names:
        for line in get(f"{NEMO}/{lang}/data/numbers/{name}").decode("utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) >= 2 and cols[0].strip():
                out.append((clean(cols[1]), devanagari_int(cols[0].strip())))
    return out


def wikidata_bn() -> list:
    query = """
SELECT ?value ?lemma (GROUP_CONCAT(DISTINCT ?rep; separator="|") AS ?forms) WHERE {
  ?l dct:language wd:Q9610; wikibase:lexicalCategory wd:Q63116; wikibase:lemma ?lemma;
     ontolex:sense ?s. ?s wdt:P5137 ?item. ?item wdt:P1181 ?value.
  OPTIONAL { ?l ontolex:lexicalForm ?f. ?f ontolex:representation ?rep. }
} GROUP BY ?value ?lemma"""
    rows = json.loads(get("https://query.wikidata.org/sparql?format=json&query="
                          + urllib.parse.quote(query)))["results"]["bindings"]
    out = []
    for r in rows:
        try:
            v = float(r["value"]["value"])
        except ValueError:
            continue
        if v != int(v) or not 0 <= v < 100:
            continue
        lemma = clean(r["lemma"]["value"])
        forms_ = [clean(f) for f in r.get("forms", {}).get("value", "").split("|") if f]
        # Lemma plus short forms (দু for দুই); longer forms are inflections (genitive একের).
        out += [(w, int(v)) for w in [lemma] + [f for f in forms_ if len(f) < len(lemma)]]
    return out


def hundreds_of(units: list, hundred: str) -> list:
    """Year-style hundreds: "<1-99> <hundred word>" as a combining form (उन्नीस सौ अस्सी)."""
    out = []
    for w, v in units:
        if 1 <= v <= 99:
            out.append((f"{w} {hundred}", v * 100, "p", 100))
            out.append((f"{w} {hundred}", v * 100, "n", 0))
    out += [(hundred, 100, "n", 0), (hundred, 100, "p", 100)]
    return out


# ------------------------------------------------------------------ checks and output
def common_words(code: str) -> set:
    path = os.path.join(CORPUS, f"{code}.txt")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} missing: run tools/testdata/fetch_indiccorp.py first")
    counts = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200_000:
                break
            # Split on whitespace only: Indic vowel signs are not "word" characters to re.
            counts.update(w for w in (t.strip(PUNCT) for t in line.lower().split()) if w)
    return {w for w, _ in counts.most_common(COMMON_WORDS)}


def finish(code: str, entries: list, sources: list) -> None:
    by_phrase = collections.defaultdict(set)
    for phrase, value, kind, unit in entries:
        if phrase:
            by_phrase[phrase].add((value, kind, unit))
    common = common_words(code)
    kept, dropped = [], []
    for phrase, meanings in sorted(by_phrase.items()):
        values = {v for v, _, _ in meanings}
        if len(values) > 1:
            dropped.append((phrase, sorted(values), "maps to several numbers"))
            continue
        value = next(iter(values))
        if " " not in phrase and value >= 10 and phrase.lower() in common:
            dropped.append((phrase, [value], f"among the {COMMON_WORDS} most common words"))
            continue
        kept += [(phrase, v, k, u) for v, k, u in sorted(meanings)]
    values_n = {v for _, v, k, _ in kept if k == "n"}
    missing = [v for v in range(100) if v not in values_n]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{code}.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for s in sources:
            f.write(f"# {s}\n")
        f.write(f"# Built {datetime.date.today()} by tools/testdata/build_number_tables.py. "
                f"Dropped {len(dropped)} entries (listed in the build output).\n")
        if missing:
            f.write(f"# Values 0-99 without a phrase: {missing}\n")
        for phrase, value, kind, unit in kept:
            f.write(f"{phrase}\t{value}\t{kind}\t{unit}\n")
    print(f"{code}: {len(kept)} entries, {len(dropped)} dropped, 0-99 missing {missing or 'none'}")
    for phrase, values, why in dropped:
        print(f"    dropped {phrase!r} -> {values}: {why}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    num2words = load_numtowords()
    a4b = f"AI4Bharat indic-numtowords {NUMTOWORDS_REV[:8]} (MIT, Copyright (c) 2022 AI4Bharat)"
    for code in LANGS:
        entries = ai4bharat(num2words, code)
        sources = [f"Number phrases. Source: {a4b}, generated forms inverted."]
        if code == "hi":
            units, hundred = cldr("hi")
            units += nemo("hi", ["digit.tsv", "teens_and_ties.tsv"])
            entries += [(w, v, "n", 0) for w, v in units] + hundreds_of(units, hundred)
            sources.append("Also: Unicode CLDR release-48-2 rbnf/hi.xml (Unicode License v3, "
                           "Copyright Unicode, Inc.); NVIDIA NeMo-text-processing ddadfb2a hi ITN "
                           "data (Apache-2.0, Copyright NVIDIA CORPORATION).")
        elif code == "gu":
            units, hundred = cldr("gu")
            entries += [(w, v, "n", 0) for w, v in units] + hundreds_of(units, hundred)
            sources.append("Also: Unicode CLDR release-48-2 rbnf/gu.xml (Unicode License v3, "
                           "Copyright Unicode, Inc.).")
        elif code == "mr":
            units = nemo("mr", ["digits.tsv", "tens.tsv", "zero.tsv"])
            hundred = nemo("mr", ["hundred.tsv"])[0][0]
            entries += [(w, v, "n", 0) for w, v in units] + hundreds_of(units, hundred)
            sources.append("Also: NVIDIA NeMo-text-processing ddadfb2a mr ITN data (Apache-2.0, "
                           "Copyright NVIDIA CORPORATION).")
        elif code == "bn":
            entries += [(w, v, "n", 0) for w, v in wikidata_bn()]
            sources.append(f"Also: Wikidata numeral lexemes (CC0), queried {datetime.date.today()}.")
        finish(code, entries, sources)


if __name__ == "__main__":
    main()
