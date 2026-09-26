"""Voice-over for the lesson.

* Text is synthesised offline, one sentence at a time, with RHVoice (the most
  intelligible voice available in this environment; see README).  Results are
  cached on disk, keyed by engine, voice, rate and text.
* ``VoiceScene.say(text)`` plays a narration block and keeps the animations
  that run inside the ``with`` block in step with it.  When the block exits the
  scene waits for the audio to finish, so narration never overlaps.
* Each sentence's start/end time is logged so the build can write subtitles.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from manim import MovingCameraScene, config

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
TTS_CACHE = BUILD / "tts_cache"
CAPTIONS_DIR = BUILD / "captions"

VOICE = os.environ.get("LESSON_VOICE", "bdl")
RATE = int(os.environ.get("LESSON_RATE", "80"))  # RHVoice: 100 = default speed (~195 wpm)
SAMPLE_RATE = 48_000
SENTENCE_GAP = 0.30  # silence between sentences inside one block (s)

# Captions keep normal spelling; the synthesiser gets these respellings, chosen
# after checking the voice with an offline speech recogniser.
PRONUNCIATIONS = [
    (r"\b1687\b", "sixteen eighty-seven"),
    (r"\bPrincipia\b", "Prin-sip-ee-uh"),
    (r"metre", "meter"),  # metre(s), kilometre(s)
    (r"centre", "center"),
    (r"\btyres\b", "tires"),
    (r"\bdp/dt\b", "dee p by dee t"),
    (r"\bdv/dt\b", "dee v by dee t"),
    (r"—", ", "),
    (r"–", " "),
]


def to_speech(text: str) -> str:
    for pattern, repl in PRONUNCIATIONS:
        text = re.sub(pattern, repl, text)
    return text


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _synth_sentence(sentence: str) -> Path:
    speech = to_speech(sentence)
    key = hashlib.sha1(f"rhvoice|{VOICE}|{RATE}|{speech}".encode()).hexdigest()[:20]
    out = TTS_CACHE / f"s_{key}.wav"
    if out.exists():
        return out
    if shutil.which("RHVoice-test") is None:
        raise RuntimeError("RHVoice-test not found: apt-get install rhvoice rhvoice-english")
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.wav")
    subprocess.run(
        ["RHVoice-test", "-p", VOICE, "-r", str(RATE), "-o", str(raw)],
        input=speech.encode(), check=True, capture_output=True,
    )
    # Trim leading/trailing silence and resample; short fades remove edge clicks.
    trimmed = out.with_name(out.stem + "_trim.wav")
    subprocess.run(
        ["sox", str(raw), "-b", "16", str(trimmed),
         "silence", "1", "0.02", "0.3%", "reverse", "silence", "1", "0.02", "0.3%", "reverse",
         "rate", "-v", str(SAMPLE_RATE)],
        check=True, capture_output=True,
    )
    data, sr = sf.read(trimmed, dtype="float64")
    if data.ndim > 1:
        data = data[:, 0]
    n_in, n_out = int(0.005 * sr), int(0.03 * sr)
    data[:n_in] *= np.linspace(0.0, 1.0, n_in)
    data[-n_out:] *= np.linspace(1.0, 0.0, n_out)
    sf.write(out, data, sr, subtype="PCM_16")
    raw.unlink(missing_ok=True)
    trimmed.unlink(missing_ok=True)
    return out


@dataclass
class Clip:
    """A synthesised narration block."""

    path: Path
    duration: float
    sentences: list[tuple[float, float, str]] = field(default_factory=list)
    t0: float = 0.0  # scene time at which the clip started playing

    def start_of(self, i: int) -> float:
        return self.sentences[i][0]

    def end_of(self, i: int) -> float:
        return self.sentences[i][1]


def make_clip(text: str) -> Clip:
    sentences = split_sentences(text)
    files = [_synth_sentence(s) for s in sentences]
    key = hashlib.sha1(("|".join(f.name for f in files) + f"|{SENTENCE_GAP}").encode()).hexdigest()[:20]
    out = TTS_CACHE / f"b_{key}.wav"
    gap = np.zeros(int(SENTENCE_GAP * SAMPLE_RATE), dtype=np.int16)
    pieces, timings, t = [], [], 0.0
    for i, (s, f) in enumerate(zip(sentences, files)):
        data, sr = sf.read(f, dtype="int16")
        assert sr == SAMPLE_RATE, (f, sr)
        if data.ndim > 1:
            data = data[:, 0]
        dur = len(data) / SAMPLE_RATE
        timings.append((t, t + dur, s))
        pieces.append(data)
        t += dur
        if i < len(sentences) - 1:
            pieces.append(gap)
            t += SENTENCE_GAP
    if not out.exists():
        sf.write(out, np.concatenate(pieces), SAMPLE_RATE, subtype="PCM_16")
    return Clip(out, t, timings)


class VoiceScene(MovingCameraScene):
    """Scene (with a movable camera) plus synchronised narration and caption logging."""

    def setup(self):
        super().setup()
        self._captions: list[dict] = []

    @property
    def now(self) -> float:
        return self.renderer.time

    @contextmanager
    def say(self, text: str, pause: float = 0.45):
        clip = make_clip(text)
        clip.t0 = self.now
        # Scene.add_sound() silently drops sounds while renderer.skip_animations is set, which Manim
        # leaves on after replaying a cached animation; write the sound to the timeline directly.
        self.renderer.file_writer.add_sound(str(clip.path), self.now)
        yield clip
        self.wait_until(clip, clip.duration + pause)
        for start, end, sentence in clip.sentences:
            self._captions.append({"start": clip.t0 + start, "end": clip.t0 + end, "text": sentence})

    def narrate(self, text: str, pause: float = 0.45):
        """Narration with no animation of its own."""
        with self.say(text, pause):
            pass

    def wait_until(self, clip: Clip, t: float):
        """Wait until clip-relative time ``t`` (no-op if already past it)."""
        dt = clip.t0 + t - self.now
        if dt > 1.0 / config.frame_rate:
            self.wait(dt)

    def at_sentence(self, clip: Clip, i: int, offset: float = 0.0):
        self.wait_until(clip, clip.start_of(i) + offset)

    def remaining(self, clip: Clip, until: float | None = None) -> float:
        """Seconds left before clip-relative time ``until`` (default: end of clip)."""
        end = clip.duration if until is None else until
        return max(clip.t0 + end - self.now, 1.0 / config.frame_rate)

    def tear_down(self):
        super().tear_down()
        CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{config.pixel_height}p{int(round(config.frame_rate))}"
        (CAPTIONS_DIR / f"{type(self).__name__}_{tag}.json").write_text(
            json.dumps({"scene": type(self).__name__, "duration": self.now,
                        "captions": self._captions}, indent=1)
        )
