"""Word error rate on our own recordings, per processing setting.

Replays each recorded journal through `sdctl process` (echo canceller, optional noise
suppression, VAD, streaming ASR exactly as live), joins each source's final transcripts in
time order and writes them, with the known reference text, as an `sdctl asr eval`-style TSV
that `sdctl asr score` scores.

    python tools/eval/recording_wer.py [--sdctl PATH] [--out target/eval/recordings.tsv]

Recordings (local, not committed; docs/m2-report.md "Live session"):
  dt1  default mic mode (driver echo cancellation on); far end = far_en.wav; the user read
       the paragraph twice (over the far end, then alone).
  dt2  raw mic (driver processing bypassed); the user read line 1 over the far end, then
       line 2 alone.
"""
import argparse
import csv
import os
import subprocess
import sys

FAR = ("Good morning everyone, thanks for joining the weekly planning call. Let us start with "
       "the release schedule. The first build is ready for testing, and the team has fixed most "
       "of the issues reported last week. We still need to confirm the deployment window with "
       "operations. After that, we will review the budget for the next quarter, and finally we "
       "will go through the open questions from the customer meeting on Tuesday. If anyone has "
       "an urgent item, please raise it now so we can make time for it at the end of the call.")
LINE1 = ("Today we reviewed the project timeline and agreed to finish the audio tests this week. "
         "The next step is speech recognition, and we will measure accuracy before choosing the "
         "final settings.")
LINE2 = ("Please send me the meeting notes by Friday. I will check the recording quality and "
         "share the results with the team on Monday morning.")
RECORDINGS = {
    "dt1": {"Mic": LINE1 + " " + LINE1, "System": FAR},
    "dt2": {"Mic": LINE1 + " " + LINE2, "System": FAR},
}
# name -> extra `sdctl process` arguments
SETTINGS = {
    "default": [],  # AEC auto (echo gate), hybrid 20 dB, no NS, AGC on
    "ns-moderate": ["--ns", "moderate"],
    "ns-high": ["--ns", "high"],
    "aec-full": ["--aec", "on", "--hybrid-threshold-db", "0"],
    "aec-hybrid": ["--aec", "on"],
    "aec-linear": ["--aec", "on", "--hybrid-threshold-db", "999"],
    "aec-off": ["--aec", "off"],
}
ASR = ["--asr-model", "models/ggml-large-v3-turbo-q5_0.bin", "--indic", "models/indicconformer/assets",
       "--indic-lm", "models/lm", "--gpu", "--gpu-device", "1"]


def main():
    ap = argparse.ArgumentParser()
    sd.add_argument("--sdctl", default=r"D:\t\h\release\sdctl.exe")
    sd.add_argument("--out", default="target/eval/recordings.tsv")
    sd.add_argument("--recordings", default=",".join(RECORDINGS))
    sd.add_argument("--settings", default=",".join(SETTINGS))
    a = sd.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = r"C:\VulkanSDK\1.4.363.0\Bin;" + env.get("PATH", "")
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write("language\tfile\tdetected\terrors\tref_len\treference\thypothesis\n")
        for rec in a.recordings.split(","):
            for name in a.settings.split(","):
                tsv = f"target/eval/rec-{rec}-{name}.transcript.tsv"
                cmd = [a.sdctl, "process", f"target/{rec}", "--transcript", tsv] + ASR + SETTINGS[name]
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", env=env)
                if r.returncode != 0:
                    sys.exit(f"{rec} {name}: sdctl failed\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
                for source, reference in RECORDINGS[rec].items():
                    finals = sorted((float(x["start_s"]), x["text"], x["language"])
                                    for x in rows if x["source"] == source)
                    hyp = " ".join(t for _, t, _ in finals).replace("\t", " ")
                    langs = ",".join(sorted({l for _, _, l in finals})) or "-"
                    f.write(f"en\t{rec}-{name}-{source}\t{langs}\t0\t0\t{reference}\t{hyp}\n")
                f.flush()
                summary = [l for l in r.stdout.splitlines() if l.startswith("ASR:")]
                print(f"{rec} {name}: {len(rows)} finals; {summary[0] if summary else ''}", flush=True)
    report(a.out)


def words(text):
    import unicodedata
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, y in enumerate(h, 1):
            cur[j] = min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1], len(r)


def report(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="	"))
    print(f"{'recording / setting':<28} {'mic WER':>8} {'system WER':>11}")
    by = {}
    for r in rows:
        name, source = r["file"].rsplit("-", 1)
        e, n = wer(r["reference"], r["hypothesis"])
        by.setdefault(name, {})[source] = 100.0 * e / max(n, 1)
    for name, v in by.items():
        print(f"{name:<28} {v.get('Mic', float('nan')):7.1f}% {v.get('System', float('nan')):10.1f}%")


if __name__ == "__main__":
    main()
