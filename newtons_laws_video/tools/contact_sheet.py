"""QA helper: tile frames sampled at the end of each narrated sentence.

usage: python tools/contact_sheet.py SCENE_NAME [--video PATH] [--cols 3] [--per-sheet 6]
Writes build/qa/<SCENE>_<n>.png contact sheets with the time and caption of each frame.
"""
import argparse
import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def find_video(scene: str) -> Path:
    hits = sorted((ROOT / "build" / "media" / "videos").glob(f"*/*/{scene}.mp4"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no rendered video for {scene}")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--video")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--times", help="comma-separated times instead of caption ends")
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames (e.g. 0.5 for 1080p renders)")
    args = ap.parse_args()

    video = Path(args.video) if args.video else find_video(args.scene)
    folder = video.parent.name  # e.g. 480p15 or 1080p30
    caps = json.loads((ROOT / "build" / "captions" / f"{args.scene}_{folder}.json").read_text())
    if args.times:
        samples = [(float(t), "") for t in args.times.split(",")]
    else:
        samples = [(max(c["end"] - 0.05, 0), c["text"]) for c in caps["captions"]]
        samples.append((caps["duration"] - 0.25, "[end of scene]"))

    out_dir = ROOT / "build" / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (t, text) in enumerate(samples):
        f = out_dir / f"_frame_{i:03d}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                        "-frames:v", "1", str(f)], check=True)
        frames.append((f, t, text))

    font = ImageFont.load_default()
    sheet_paths = []
    for s in range(0, len(frames), args.per_sheet):
        chunk = frames[s:s + args.per_sheet]
        ims = [Image.open(f) for f, _, _ in chunk]
        if args.scale != 1.0:
            ims = [im.resize((int(im.width * args.scale), int(im.height * args.scale)), Image.LANCZOS) for im in ims]
        w, h = ims[0].size
        cap_h = 46
        rows = (len(ims) + args.cols - 1) // args.cols
        sheet = Image.new("RGB", (args.cols * w, rows * (h + cap_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (im, (_, t, text)) in enumerate(zip(ims, chunk)):
            x, y = (k % args.cols) * w, (k // args.cols) * (h + cap_h)
            sheet.paste(im, (x, y))
            label = f"t={t:6.2f}s  " + text
            for j, line in enumerate(textwrap.wrap(label, width=w // 6)[:3]):
                draw.text((x + 4, y + h + 2 + 14 * j), line, fill="black", font=font)
        path = out_dir / f"{args.scene}_{s // args.per_sheet:02d}.png"
        sheet.save(path)
        sheet_paths.append(path)
    for f, _, _ in frames:
        f.unlink()
    print("\n".join(str(p) for p in sheet_paths))


if __name__ == "__main__":
    main()
