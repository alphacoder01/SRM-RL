# Newton's Laws of Motion: a narrated video lesson for Grade 12

An animated, narrated physics lesson (33 minutes) that teaches Newton's three laws of motion
at grade-12 depth (CBSE class 11–12, AP Physics 1/C, A-level, IB). It builds the ideas from observation,
states each law precisely, works through quantitative examples with free-body diagrams, clears up
the classic misconceptions, and ends with a quiz.

**Watch:** [`output/newtons_laws_of_motion.mp4`](output/newtons_laws_of_motion.mp4)
(1920×1080, 30 fps, H.264 video + mono AAC narration, English subtitles on by default, chapter markers)

Also in `output/`:

| file | contents |
|---|---|
| `newtons_laws_of_motion.srt` | the subtitles as a separate file |
| `transcript.md` | the full narration, chapter by chapter, with timestamps |
| `chapters.txt` | chapter start times (paste into a YouTube description to get chapters there too) |

## Chapters

<!-- CHAPTERS:START -->
| start | chapter |
|---|---|
| 0:00 | Introduction |
| 1:11 | 1. What keeps things moving? |
| 3:34 | 2. Newton's first law |
| 7:55 | 3. Newton's second law |
| 12:04 | 4. Impulse |
| 13:52 | 5. Newton's third law |
| 19:18 | 6. Conservation of momentum |
| 21:18 | 7. Free-body diagrams; Example 1 |
| 24:13 | Example 2: apparent weight in a lift |
| 26:24 | Example 3: connected blocks |
| 27:58 | 8. Summary and common mistakes |
| 29:56 | Quiz and wrap-up |
<!-- CHAPTERS:END -->

## What the lesson covers

1. **What keeps things moving?** Aristotle's view; friction as the reason things stop (sliding
   blocks with real μg decelerations); Galileo's double-ramp thought experiment, animated with the
   rolling-ball kinematics a = (5/7) g sin θ.
2. **First law.** Precise statement; constant *velocity* vs constant speed (circular motion needs a net
   force); balanced forces and the vector sum; the cruising car; inertia and mass; coin–card–glass;
   passengers in a bus; the same event seen from the road and from inside an accelerating bus, which leads
   to **inertial and non-inertial frames**.
3. **Second law.** Momentum p = mv; F<sub>net</sub> = dp/dt; F<sub>net</sub> = ma for constant mass;
   double-force and double-mass cart races; the newton; component form, shown with a projectile; the law is
   instantaneous (synchronised F–t, a–t and v–t graphs).
4. **Impulse.** J = ∫F dt = F<sub>avg</sub>Δt = Δp; catching a cricket ball with stiff vs yielding hands
   (400 N vs 40 N, to scale); airbags, crumple zones, landing mats.
5. **Third law.** Forces are interactions; the four properties of every pair; skaters pushing apart;
   walking and rockets; the Earth–apple pair; the *book-on-a-table trap* (weight and normal force are
   **not** a pair); the horse-and-cart puzzle.
6. **Conservation of momentum**, derived from the second and third laws; the skaters' momenta; recoil.
7. **Problem solving.** A five-step free-body method; the common forces; three worked examples:
   an angled pull (N ≠ mg), apparent weight in a lift (all four cases), and connected blocks (system
   vs. single-body diagrams, tension).
8. **Summary, misconceptions, limits and quiz.** Four common mistakes; where Newtonian mechanics stops
   working (relativity, quantum mechanics); three quiz questions with pauses, the last a full
   Atwood-machine calculation.

## How the technical accuracy was checked

* **Every number spoken in the video is recomputed** from first principles in
  [`verify_physics.py`](verify_physics.py) (run `python verify_physics.py`; all checks pass).
* **Animations follow the actual equations of motion.** Positions come from kinematics with the stated
  forces and masses (friction decelerations, the rolling ball, carts, projectile, lift, Atwood machine).
  Force arrows marked "to scale" are proportional to the forces. Any slow motion is labelled on screen.
* **Idealisations are stated out loud:** frictionless surfaces, light strings, no air resistance,
  g = 9.8 m/s².
* **Every narrated sentence was proofread for physics accuracy.** Frames were reviewed with contact
  sheets ([`tools/contact_sheet.py`](tools/contact_sheet.py), [`tools/review_sheets.py`](tools/review_sheets.py)).
  The sync between narration and captions in the final file was measured
  ([`tools/check_sync.py`](tools/check_sync.py)). Every sentence of the synthetic voice was also run through
  an offline speech recogniser to catch mispronounced words ([`tools/check_voice.py`](tools/check_voice.py)).

## About the narration voice

The voice is **RHVoice "bdl"** (US English), synthesised offline. The build environment's network policy
blocked the download of neural text-to-speech models (Hugging Face and GitHub releases), so the available
voices were compared with an offline speech recogniser on lesson sentences. "bdl" was the most
intelligible (16 % word-error rate, against 21–53 % for the others). Numbers and a few words are
respelled for the synthesiser only (for example, "1687" is spoken as "sixteen eighty-seven"), so the
captions keep normal spelling. Captions are on by default because the voice is synthetic.

**To re-voice the video**, replace `_synth_sentence()` in [`lesson/narration.py`](lesson/narration.py) with
any TTS that writes a WAV file (for example Piper or Kokoro), or change `LESSON_VOICE` / `LESSON_RATE`.
Then re-run the build. All animation timings and captions adapt to the new audio automatically.

## Rebuilding the video

```bash
# system packages (Ubuntu 24.04)
sudo apt-get install ffmpeg sox rhvoice rhvoice-english texlive-latex-base texlive-latex-recommended \
    texlive-latex-extra texlive-fonts-recommended texlive-science dvisvgm cm-super fonts-lato \
    libpango1.0-dev libcairo2-dev pkg-config
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python verify_physics.py          # check the numbers
python build.py --quality low     # quick 480p15 preview (a few minutes)
python build.py --crf 20          # final 1080p30 render + assembly, as published (tens of minutes on 4 cores)
python build.py --scenes S05_ThirdLaw --crf 20   # re-render one chapter, then re-assemble
```

The build renders each chapter with [Manim Community](https://www.manim.community/) 0.21. It then joins
the chapters with exact per-chapter audio alignment, and with `--crf` re-encodes the video once with
x264 (tuned for animation), which is visually identical and about half the size. It normalises the
narration to −16 LUFS (EBU R128, two-pass, −3 dBTP peak ceiling) and muxes the subtitles and chapter
markers.

## Project layout

```
newtons_laws_video/
├── build.py              render all scenes in parallel and assemble output/
├── verify_physics.py     independent check of every quoted number
├── lesson/
│   ├── narration.py      TTS, caching, VoiceScene (keeps animation in sync with speech, logs captions)
│   ├── style.py          colour code (weight red, normal blue, tension yellow, friction pink, ...), fonts
│   ├── objects.py        drawing helpers: force arrows, blocks, carts, figures, lift, pulley, rocket, ...
│   ├── common.py         chapter cards
│   └── scenes/           one file per chapter (s00_intro.py ... s11_quiz.py)
├── tools/
│   ├── contact_sheet.py  QA: frames at the end of every narrated sentence of a scene
│   ├── review_sheets.py  QA: frames at the end of every narration block of the final video
│   ├── check_sync.py     QA: speech onsets vs caption times; loudness report
│   └── check_voice.py    QA: offline speech recognition of every narrated sentence
└── output/               the finished video, subtitles, transcript, chapter list
```
