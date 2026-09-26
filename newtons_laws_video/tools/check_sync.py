"""QA: check that narration in the final video starts exactly where the captions say it does.

For every subtitle cue that begins a sentence, the audio should be (near) silent just before the
cue and contain speech just after it. Also reports loudness statistics of the final track."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
video = ROOT / "output" / "newtons_laws_of_motion.mp4"
srt = ROOT / "output" / "newtons_laws_of_motion.srt"
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                      "-f", "s16le", "-"], capture_output=True, check=True).stdout
x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
sr = 16000


def rms(a, b):
    seg = x[int(a * sr):int(b * sr)]
    return float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0


def ts(s):
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


cues = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\n(.+?)\n\n", srt.read_text(), re.S)
starts, prev_end = [], -1.0
for a, b, text in cues:
    a, b = ts(a), ts(b)
    if a - prev_end > 0.2:  # a new sentence (not a continuation piece of the same sentence)
        starts.append(a)
    prev_end = b
speech_level = np.percentile(np.abs(x), 99)
bad = []
for t in starts:
    before, after = rms(t - 0.22, t - 0.04), rms(t + 0.03, t + 0.45)
    if not (after > 4 * max(before, 1e-4) and after > 0.02):
        bad.append((round(t, 2), round(before, 4), round(after, 4)))
print(f"sentence onsets checked: {len(starts)}; out of sync: {len(bad)}")
for b in bad[:20]:
    print("  suspicious onset at", b)
# The narration is a mono track; measure it as played on stereo speakers (dual mono), per EBU R128 practice.
ebur = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(video), "-map", "0:a", "-af",
                       "pan=stereo|c0=c0|c1=c0,ebur128=peak=true", "-f", "null", "-"],
                      capture_output=True, text=True).stderr
summary = ebur[ebur.rfind("Summary:"):]
print("loudness (dual-mono playback):", " ".join(summary.split()))
sys.exit(1 if bad else 0)
