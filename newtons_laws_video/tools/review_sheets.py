"""QA: contact sheets of the assembled video, one frame at the end of every narration block
(where the most elements are on screen), labelled with the time and the caption text."""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
video = ROOT / "output" / "newtons_laws_of_motion.mp4"
srt = (ROOT / "output" / "newtons_laws_of_motion.srt").read_text()
per_sheet, cols = int(sys.argv[1]) if len(sys.argv) > 1 else 12, 4


def ts(s):
    h, m, r = s.split(":")
    sec, ms = r.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


cues = [(ts(a), ts(b), t.replace("\n", " ")) for a, b, t in
        re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\n(.+?)\n\n", srt, re.S)]
samples = []
for i, (a, b, t) in enumerate(cues):
    nxt = cues[i + 1][0] if i + 1 < len(cues) else None
    if nxt is None or nxt - b > 0.36:  # end of a narration block
        samples.append((b + 0.3, t))
out = ROOT / "build" / "qa" / "review"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.png"):
    f.unlink()
font = ImageFont.load_default()
frames = []
for k, (t, text) in enumerate(samples):
    f = out / f"_f{k:03d}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(f)],
                   check=True)
    frames.append((f, t, text))
for s in range(0, len(frames), per_sheet):
    chunk = frames[s:s + per_sheet]
    ims = [Image.open(f) for f, _, _ in chunk]
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (im, (_, t, text)) in enumerate(zip(ims, chunk)):
        x, y = (k % cols) * w, (k // cols) * (h + 30)
        sheet.paste(im, (x, y))
        lab = f"{int(t // 60)}:{t % 60:04.1f}  " + text
        for j, line in enumerate(textwrap.wrap(lab, width=w // 6)[:2]):
            d.text((x + 4, y + h + 2 + 13 * j), line, fill="black", font=font)
    sheet.save(out / f"sheet_{s // per_sheet:02d}.png")
for f, _, _ in frames:
    f.unlink()
print(f"{len(samples)} frames -> {len(range(0, len(frames), per_sheet))} sheets in {out}")
