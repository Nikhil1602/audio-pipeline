"""Compare Auto language-ID rules offline, on `sdctl asr eval --mode lid` output with
per-stage probabilities (columns probs_whisper, probs_indicconformer).

    python tools/eval/lid_rules.py target/eval/cascade-lid.tsv [bench/data/fleurs]

Per clip (no context), two ways of treating Hindi/Urdu as one spoken language written as
Hindi:
  sum   - cascade, Urdu's probability added to Hindi's, then argmax
  label - cascade, argmax over the unmerged probabilities, then Urdu relabelled as Hindi
each with two thresholds for running stage 2 (the Indic share of stage 1).

Per meeting: the product's session logic (sd_asr::auto::SessionLanguage, mirrored below with
its defaults) over sequences of clips, as a meeting presents them:
  single - 10 consecutive clips of one language
  switch - 5 clips of one language, then 5 of English (and back): checks that the session
           prior follows a real language change.
"""
import csv
import os
import random
import sys
from collections import Counter, defaultdict

INDIC = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te", "or", "ur"]
MERGE = {"ur": "hi"}
FLEURS = {
    "en": "en_us", "zh": "cmn_hans_cn", "es": "es_419", "fr": "fr_fr", "bn": "bn_in",
    "gu": "gu_in", "hi": "hi_in", "kn": "kn_in", "ml": "ml_in", "mr": "mr_in", "pa": "pa_in",
    "ta": "ta_in", "te": "te_in", "or": "or_in", "ur": "ur_pk",
}


def parse(cell):
    out = {}
    for part in filter(None, cell.split(",")):
        lang, p = part.split(":")
        out[lang] = float(p)
    return out


def cascade(whisper, indic, min_group):
    mass = sum(p for l, p in whisper.items() if l in INDIC)
    if mass < min_group or not indic:
        return dict(whisper)
    out = {l: p for l, p in whisper.items() if l not in INDIC}
    for l, p in indic.items():
        out[l] = p * mass
    return out


def group_first(probs):
    """If the Indic languages together outweigh the best non-Indic language, rescale them so
    the best Indic language carries the whole Indic share (the decision is "Indic" first, then
    which Indic language)."""
    mass = sum(p for l, p in probs.items() if l in INDIC)
    other = max((p for l, p in probs.items() if l not in INDIC), default=0.0)
    top = max((p for l, p in probs.items() if l in INDIC), default=0.0)
    if mass <= other or top <= 0 or top >= other:
        # Only when a non-Indic language is on top: then the Indic share as a whole decides.
        return probs
    return {l: (p * mass / top if l in INDIC else p) for l, p in probs.items()}


def merged(probs):
    m = defaultdict(float)
    for l, p in probs.items():
        m[MERGE.get(l, l)] += p
    return dict(m)


def decide(probs, rule):
    if rule == "group":
        probs = group_first(probs)
    if rule == "sum":
        m = merged(probs)
        return max(m, key=m.get)
    top = max(probs, key=probs.get)
    return MERGE.get(top, top)


class Session:
    """Mirror of sd_asr::auto::SessionLanguage with AutoConfig::default()."""

    def __init__(self, confident=0.8, min_confident_s=3.0, prior_weight=0.7, decay=0.7):
        self.c, self.min_s, self.w, self.decay = confident, min_confident_s, prior_weight, decay
        self.evidence = {}

    def decide(self, probs, dur):
        own = max(probs.items(), key=lambda kv: kv[1])
        total = sum(self.evidence.values())
        if (own[1] >= self.c and dur >= self.min_s) or total <= 0:
            choice = own[0]
        else:
            scored = {
                l: p * (self.evidence.get(l, 0.0) / total + 0.05) ** self.w
                for l, p in probs.items()
            }
            choice = max(scored, key=scored.get)
        for l in self.evidence:
            self.evidence[l] *= self.decay
        z = max(sum(probs.values()), 1e-30)
        for l, p in probs.items():
            self.evidence[l] = self.evidence.get(l, 0.0) + max(dur, 0.5) * p / z
        return choice


def clip_level(rows, langs):
    for rule in ("label", "group"):
        for min_group in (0.2,):
            print(f"\n== per clip: rule {rule}, stage-2 threshold {min_group}")
            total_ok = 0
            for lang in langs:
                rs = [r for r in rows if r["language"] == lang]
                expected = MERGE.get(lang, lang)
                got = Counter(decide(r["cascade"][min_group], rule) for r in rs)
                ok = got[expected]
                total_ok += ok
                wrong = ", ".join(f"{l}x{n}" for l, n in got.most_common() if l != expected)
                print(f"  {lang:<3} {100 * ok / len(rs):5.1f}%  {wrong}")
            print(f"  all {100 * total_ok / len(rows):5.1f}%")


PAIRS = [("hi", "mr"), ("hi", "pa"), ("hi", "gu"), ("bn", "or"), ("ta", "ml"), ("te", "kn"),
         ("mr", "gu"), ("ur", "pa")]


def meeting_level(rows, langs, rule, min_group, trials=200, prior_weight=0.7, decay=0.7):
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    rng = random.Random(7)

    def run(seq):
        s = Session(prior_weight=prior_weight, decay=decay)
        out = []
        for r in seq:
            p = r["cascade"][min_group]
            if rule == "sum":
                p = merged(p)
            if rule == "group":
                p = group_first(p)
            choice = s.decide(p, r["dur"])
            out.append(MERGE.get(choice, choice))
        return out

    print(f"\n== per meeting (session prior w={prior_weight} decay={decay}): rule {rule}")
    print("  lang  single-language meeting   switching to/from English")
    for lang in langs:
        expected = MERGE.get(lang, lang)
        ok = n = 0
        sw_ok = sw_n = 0
        for _ in range(trials):
            seq = rng.sample(by_lang[lang], 10)
            ok += sum(c == expected for c in run(seq))
            n += 10
            if lang != "en":
                first = rng.sample(by_lang[lang], 5)
                second = rng.sample(by_lang["en"], 5)
                for a, b, ea, eb in ((first, second, expected, "en"), (second, first, "en", expected)):
                    res = run(a + b)
                    sw_ok += sum(c == ea for c in res[:5]) + sum(c == eb for c in res[5:])
                    sw_n += 10
        sw = f"{100 * sw_ok / sw_n:5.1f}%" if sw_n else "    -"
        print(f"  {lang:<3}   {100 * ok / n:5.1f}%                    {sw}")
    print("  switching between two Indic languages (5 + 5 clips, both orders):")
    for x, y in PAIRS:
        ok = n = 0
        for _ in range(trials):
            for a, b in ((x, y), (y, x)):
                ea, eb = MERGE.get(a, a), MERGE.get(b, b)
                res = run(rng.sample(by_lang[a], 5) + rng.sample(by_lang[b], 5))
                ok += sum(c == ea for c in res[:5]) + sum(c == eb for c in res[5:])
                n += 10
        print(f"    {x}<->{y}  {100 * ok / n:5.1f}%")


def main(path, fleurs):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    for r in rows:
        w, i = parse(r["probs_whisper"]), parse(r["probs_indicconformer"])
        r["cascade"] = {g: cascade(w, i, g) for g in (0.2, 0.05)}
        wav = os.path.join(fleurs, FLEURS[r["language"]], r["file"])
        r["dur"] = (os.path.getsize(wav) - 80) / 4 / 16000  # float32 mono, 16 kHz
    langs = list(dict.fromkeys(r["language"] for r in rows))
    clip_level(rows, langs)
    for rule in ("label", "group"):
        meeting_level(rows, langs, rule, 0.2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "bench/data/fleurs")
