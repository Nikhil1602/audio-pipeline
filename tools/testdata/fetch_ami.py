"""Fetch AMI meetings with speaker annotations for diarization evaluation.

AMI Meeting Corpus (CC BY 4.0): https://groups.inf.ed.ac.uk/ami/corpus/
Audio: the Mix-Headset recording (all headset channels summed, 16 kHz).
Reference: the standard "only_words" RTTMs of BUTSpeechFIT/AMI-diarization-setup (pinned),
i.e. speech regions from the word-level annotations, as used in published AMI DER results.

Writes bench/data/ami/<meeting>.wav and <meeting>.rttm (bench/data/ is not committed).

Usage: python tools/testdata/fetch_ami.py [meeting ...]
       (default: 4 meetings of the standard test split, about 2 hours)
"""
import os
import sys
import urllib.request

AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
SETUP_REV = "main"
RTTM = ("https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/"
        f"{SETUP_REV}/only_words/rttms/test/{{m}}.rttm")
DEFAULT = ["ES2004a", "IS1009a", "TS3003a", "EN2002a"]
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "data", "ami")


def get(url: str, path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "sonicdiar/0.1 (eval data)"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(path + ".part", path)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for m in sys.argv[1:] or DEFAULT:
        wav, rttm = os.path.join(OUT, f"{m}.wav"), os.path.join(OUT, f"{m}.rttm")
        if not os.path.isfile(rttm):
            get(RTTM.format(m=m), rttm)
        if not os.path.isfile(wav):
            get(AUDIO.format(m=m), wav)
        speakers = {line.split()[7] for line in open(rttm) if line.startswith("SPEAKER")}
        print(f"{m}: {os.path.getsize(wav) / 1e6:.0f} MB audio, {len(speakers)} speakers")


if __name__ == "__main__":
    main()
