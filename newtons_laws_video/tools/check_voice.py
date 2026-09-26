"""QA: run every narrated sentence through an offline speech recogniser (PocketSphinx) and report
the sentences it understands worst, which flags words the synthetic voice may mispronounce.

PocketSphinx itself makes errors even on clean speech, so read the output as a ranked list of
sentences to listen to, not as a pass/fail test."""
import glob
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from num2words import num2words
from pocketsphinx import Decoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lesson.narration import _synth_sentence  # noqa: E402


def norm(s: str) -> list[str]:
    s = s.lower().replace("—", " ").replace("–", " ")
    s = re.sub(r"(\d+)\.(\d+)", lambda m: num2words(int(m.group(1))) + " point " +
               " ".join(num2words(int(d)) for d in m.group(2)), s)
    s = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), s)
    return re.sub(r"[^a-z' ]", " ", s.replace("-", " ")).split()


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1, -1] / max(len(ref), 1)


dec = Decoder(samprate=16000)
rows = []
tmp = ROOT / "build" / "asr16.wav"
for f in sorted(glob.glob(str(ROOT / "build" / "captions" / "*_480p15.json"))):
    for c in json.load(open(f))["captions"]:
        wav = _synth_sentence(c["text"])
        subprocess.run(["sox", str(wav), "-r", "16000", "-c", "1", "-b", "16", str(tmp)], check=True)
        with wave.open(str(tmp)) as w:
            raw = w.readframes(w.getnframes())
        dec.start_utt()
        dec.process_raw(raw, full_utt=True)
        dec.end_utt()
        hyp = dec.hyp().hypstr if dec.hyp() else ""
        rows.append((wer(norm(c["text"]), norm(hyp)), c["text"], hyp))
w = [r[0] for r in rows]
print(f"sentences: {len(rows)}   mean WER {np.mean(w):.3f}   median {np.median(w):.3f}")
for r in sorted(rows, reverse=True)[:15]:
    print(f"\nWER {r[0]:.2f}\n  REF: {r[1]}\n  ASR: {r[2]}")
