"""Render every scene and assemble the final lesson video.

    python build.py                 # 1080p30 render of all scenes, then assemble
    python build.py --quality low   # fast 480p15 preview
    python build.py --assemble-only # reuse existing scene renders
    python build.py --scenes S03_SecondLaw S04_Impulse   # re-render some scenes, then assemble

Outputs (in ./output):
    newtons_laws_of_motion.mp4   video + AAC audio + soft English subtitles + chapter markers
    newtons_laws_of_motion.srt   the same subtitles as a sidecar file
    transcript.md                full narration with timestamps, grouped by chapter
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
OUT = ROOT / "output"
MANIM = shutil.which("manim") or str(Path(sys.executable).with_name("manim"))

# (module, scene class, chapter title shown in the player)
SCENES = [
    ("s00_intro", "S00_Intro", "Introduction"),
    ("s01_galileo", "S01_Galileo", "1. What keeps things moving?"),
    ("s02_first_law", "S02_FirstLaw", "2. Newton's first law"),
    ("s03_second_law", "S03_SecondLaw", "3. Newton's second law"),
    ("s04_impulse", "S04_Impulse", "4. Impulse"),
    ("s05_third_law", "S05_ThirdLaw", "5. Newton's third law"),
    ("s06_momentum", "S06_Momentum", "6. Conservation of momentum"),
    ("s07_free_body", "S07_FreeBody", "7. Free-body diagrams; Example 1"),
    ("s08_lift", "S08_Lift", "Example 2: apparent weight in a lift"),
    ("s09_connected", "S09_Connected", "Example 3: connected blocks"),
    ("s10_summary", "S10_Summary", "8. Summary and common mistakes"),
    ("s11_quiz", "S11_Quiz", "Quiz and wrap-up"),
]
QUALITY = {"high": ("1920,1080", 30, "1080p30"), "low": ("854,480", 15, "480p15")}
SAMPLE_RATE = 48_000


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def render_scene(module: str, scene: str, quality: str) -> Path:
    res, fps, folder = QUALITY[quality]
    log = BUILD / "logs" / f"{scene}_{quality}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    cmd = [MANIM, "render", "--resolution", res, "--frame_rate", str(fps), "--media_dir", str(BUILD / "media"),
           "--disable_caching", str(ROOT / "lesson" / "scenes" / f"{module}.py"), scene]
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{scene} failed; see {log}")
    video = BUILD / "media" / "videos" / module / folder / f"{scene}.mp4"
    print(f"  rendered {scene} -> {video.relative_to(ROOT)}", flush=True)
    return video


def probe_frames(video: Path) -> tuple[int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    return int(st["nb_read_packets"]), num / den


# ---------------------------------------------------------------------------------- subtitles
def split_for_subtitles(text: str, max_chars: int = 84) -> list[str]:
    """Split a sentence into subtitle-sized pieces, preferring clause boundaries near the middle."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    mid = len(text) / 2
    cands = [m.end() for m in re.finditer(r"[,;:]\s|\s[—–-]\s", text)]
    good = [c for c in cands if 0.25 * len(text) < c < 0.75 * len(text)]
    if good:
        cut = min(good, key=lambda c: abs(c - mid))
    else:
        spaces = [m.start() for m in re.finditer(r"\s", text)]
        cut = min(spaces, key=lambda c: abs(c - mid))
    left, right = text[:cut].strip(), text[cut:].strip()
    return split_for_subtitles(left, max_chars) + split_for_subtitles(right, max_chars)


def wrap2(text: str, width: int = 42) -> str:
    """Wrap into at most two balanced lines."""
    if len(text) <= width:
        return text
    words, best = text.split(), None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = max(len(a), len(b))
        if best is None or score < best[0]:
            best = (score, a + "\n" + b)
    return best[1]


def fmt_srt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(t: float) -> str:
    t = int(t)
    return f"{t // 60:d}:{t % 60:02d}"


# ---------------------------------------------------------------------------------- assembly
def assemble(quality: str):
    _, fps, folder = QUALITY[quality]
    work = BUILD / "assemble"
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)

    segments = []
    offset = 0.0
    for module, scene, chapter in SCENES:
        video = BUILD / "media" / "videos" / module / folder / f"{scene}.mp4"
        if not video.exists():
            raise SystemExit(f"missing render: {video}")
        frames, rate = probe_frames(video)
        dur = frames / rate
        caps = json.loads((BUILD / "captions" / f"{scene}_{folder}.json").read_text())
        segments.append(dict(scene=scene, chapter=chapter, video=video, start=offset, duration=dur,
                             captions=caps["captions"]))
        offset += dur
    total = offset
    print(f"  total duration {fmt_clock(total)} ({total:.2f} s)")

    # video: lossless concatenation with exact per-segment durations
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{s['video']}'\nduration {s['duration']:.6f}\n" for s in segments))
    video_only = work / "video.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0",
         "-c", "copy", str(video_only)])

    # audio: each scene's track padded/trimmed to its exact video length, then joined
    parts = []
    for i, s in enumerate(segments):
        wav = work / f"a{i:02d}.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(s["video"]), "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
             "-af", f"apad,atrim=0:{s['duration']:.6f}", "-c:a", "pcm_s16le", str(wav)])
        parts.append(wav)
    joined = work / "joined.wav"
    (work / "alist.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "alist.txt"),
         "-c", "copy", str(joined)])

    # two-pass EBU R128 loudness normalisation to -16 LUFS (speech for online video)
    target = "I=-16:TP=-1.5:LRA=11"
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(joined), "-af",
                           f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    m = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    norm = work / "audio.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-af",
         f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
         f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
         f"linear=true:print_format=summary", "-ar", str(SAMPLE_RATE), str(norm)])

    # subtitles
    cues = []
    for s in segments:
        for c in s["captions"]:
            start, end = s["start"] + c["start"], s["start"] + c["end"]
            pieces = split_for_subtitles(c["text"])
            weights = [len(p) for p in pieces]
            t = start
            for p, w in zip(pieces, weights):
                d = (end - start) * w / sum(weights)
                cues.append((t, t + d, wrap2(p)))
                t += d
    srt = OUT / "newtons_laws_of_motion.srt"
    srt.write_text("".join(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{txt}\n\n"
                           for i, (a, b, txt) in enumerate(cues, 1)))

    # chapters
    meta = work / "chapters.txt"
    lines = [";FFMETADATA1", "title=Newton's Laws of Motion - a grade 12 physics lesson",
             "comment=Narrated lesson on Newton's three laws of motion, rendered with Manim."]
    for s in segments:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(s['start'] * 1000)}",
                  f"END={int((s['start'] + s['duration']) * 1000)}", f"title={s['chapter']}"]
    meta.write_text("\n".join(lines) + "\n")

    final = OUT / "newtons_laws_of_motion.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(video_only), "-i", str(norm), "-i", str(srt),
         "-i", str(meta), "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3",
         "-map_chapters", "3", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text",
         "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng",
         "-metadata:s:s:0", "title=English", "-disposition:s:0", "default", "-movflags", "+faststart",
         str(final)])

    # transcript
    md = ["# Newton's Laws of Motion: narration transcript", "",
          f"Total running time: {fmt_clock(total)}", "", "## Chapters", ""]
    md += [f"- `{fmt_clock(s['start'])}` {s['chapter']}" for s in segments]
    for s in segments:
        md += ["", f"## {s['chapter']}  (`{fmt_clock(s['start'])}`)", ""]
        para = []
        for c in s["captions"]:
            para.append(c["text"])
        md.append(" ".join(para))
    (OUT / "transcript.md").write_text("\n".join(md) + "\n")
    (OUT / "chapters.txt").write_text("".join(f"{fmt_clock(s['start'])} {s['chapter']}\n" for s in segments))
    print(f"  wrote {final.relative_to(ROOT)}  ({final.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=QUALITY, default="high")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--scenes", nargs="*", help="only (re)render these scene classes")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    if not args.assemble_only:
        todo = [s for s in SCENES if not args.scenes or s[1] in args.scenes]
        # longest scenes first keeps the worker pool busy
        order = ["S05_ThirdLaw", "S02_FirstLaw", "S03_SecondLaw", "S11_Quiz", "S07_FreeBody", "S01_Galileo"]
        todo.sort(key=lambda s: order.index(s[1]) if s[1] in order else len(order))
        print(f"rendering {len(todo)} scene(s) at {args.quality} quality with {args.jobs} workers")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda s: render_scene(s[0], s[1], args.quality), todo))
    print("assembling")
    assemble(args.quality)


if __name__ == "__main__":
    main()
