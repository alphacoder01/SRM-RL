"""Part 8 -- Summary of the three laws, four common myths, and where the laws apply."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, ball, check_mark, cross_mark, panel, v
from lesson.style import *


class S10_Summary(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "8", "Summary and quiz")
        self.summary()
        self.myths()
        self.limits()

    # ------------------------------------------------------------------ the three laws
    def summary(self):
        head = bold("Newton's three laws", 40, ACCENT).move_to(v(0, 2.95))
        cards = VGroup(*[panel(4.25, 3.6) for _ in range(3)]).arrange(RIGHT, buff=0.25).move_to(v(0, 0.2))
        data = [
            ("First law", r"\vec F_{\text{net}} = \vec 0 \Leftrightarrow \vec v = \text{const.}",
             "inertia · mass measures it\nholds in inertial frames"),
            ("Second law", r"\vec F_{\text{net}} = \frac{d\vec p}{dt} = m\vec a",
             "net force causes acceleration\n(m a form: constant mass)"),
            ("Third law", r"\vec F_{12} = -\,\vec F_{21}",
             "forces come in pairs\nacting on different bodies"),
        ]
        contents = VGroup()
        for card, (t, eq, note) in zip(cards, data):
            tt = bold(t, 30, ACCENT).next_to(card.get_top(), DOWN, buff=0.3)
            e = tex(eq, size=36)
            if e.width > 3.9:
                e.scale_to_fit_width(3.9)
            e.move_to(card.get_center() + UP * 0.1)
            n = txt(note, 22, MUTED, line_spacing=0.85).next_to(card.get_bottom(), UP, buff=0.3)
            contents.add(VGroup(tt, e, n))
        bottom = mtxt(f"Together, the second and third laws give {hl('impulse', C_MOM)} "
                      f"(J = Δp) and {hl('conservation of momentum', C_MOM)}.", 28).move_to(v(0, -2.35))
        if bottom.width > 13.2:
            bottom.scale_to_fit_width(13.2)
        with self.say("Let's pull it all together. First law: if the net force on a body is zero, its velocity "
                      "stays constant. This is inertia, and it holds in inertial frames. Second law: the net force "
                      "equals the rate of change of momentum, which, for constant mass, is mass times acceleration. "
                      "Third law: forces come in equal and opposite pairs, acting on different bodies. And from the "
                      "second and third laws together, we get impulse, and the conservation of momentum.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(cards[0]), FadeIn(contents[0]), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeIn(cards[1]), FadeIn(contents[1]), run_time=0.8)
            self.at_sentence(c, 4)
            self.play(FadeIn(cards[2]), FadeIn(contents[2]), run_time=0.8)
            self.at_sentence(c, 5)
            self.play(FadeIn(bottom), run_time=0.8)
        self.play(FadeOut(VGroup(head, cards, contents, bottom)), run_time=0.6)

    # ------------------------------------------------------------------ myths
    def myths(self):
        head = bold("Four common mistakes", 40, ACCENT).move_to(v(0, 2.95))
        myths = [
            ("A moving object needs a force to keep it moving.",
             "A force is needed to change velocity, not to keep it."),
            ("Heavier objects fall faster because gravity pulls them harder.",
             "The bigger force acts on a bigger mass: a = mg/m = g for all (no air resistance)."),
            ("Action and reaction cancel each other.",
             "They act on different bodies, so they can never cancel."),
            ("The net force always points in the direction of motion.",
             "It points along the acceleration: a ball thrown up moves up, but the net force is down."),
        ]
        rows = VGroup()
        for i, (m, f) in enumerate(myths):
            y = 1.95 - 1.28 * i
            mx = VGroup(cross_mark(0.26), mtxt(m, 25, TEXT)).arrange(RIGHT, buff=0.25)
            fx = VGroup(check_mark(0.3), mtxt(f, 23, GOOD)).arrange(RIGHT, buff=0.25)
            pair = VGroup(mx, fx).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            if pair.width > 13.2:
                pair.scale_to_fit_width(13.2)
            pair.move_to(v(0, y)).to_edge(LEFT, buff=0.6)
            rows.add(pair)
        texts = [
            "Before the quiz, let's clear up four common mistakes. Mistake one: a moving object needs a force to "
            "keep it moving. In fact, a force is needed to change velocity, not to keep it.",
            "Mistake two: heavier objects fall faster, because gravity pulls them harder. In fact, the bigger force "
            "acts on a bigger mass, so the acceleration, m g divided by m, is the same g for every object, as long "
            "as air resistance is negligible.",
            "Mistake three: action and reaction cancel each other. In fact, they act on different bodies, so they "
            "can never cancel.",
            "Mistake four: the net force always points in the direction of motion. In fact, it points in the "
            "direction of the acceleration. A ball thrown upward is moving up, but the net force on it, its "
            "weight, points down.",
        ]
        self.play(FadeIn(head), run_time=0.6)
        for i, (row, t) in enumerate(zip(rows, texts)):
            with self.say(t) as c:
                k = 1 if i == 0 else 0
                self.at_sentence(c, k)
                self.play(FadeIn(row[0], shift=RIGHT * 0.2), run_time=0.7)
                self.at_sentence(c, k + 1)
                self.play(FadeIn(row[1], shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeOut(VGroup(head, rows)), run_time=0.6)

    # ------------------------------------------------------------------ where the laws apply
    def limits(self):
        head = bold("How far do Newton's laws go?", 40, ACCENT).move_to(v(0, 2.8))
        good = VGroup(check_mark(0.4), txt("everyday objects, machines, planets, spacecraft: superb accuracy",
                                          28)).arrange(RIGHT, buff=0.3)
        rel = VGroup(txt("speeds close to the speed of light", 28), txt("→", 30, MUTED),
                     bold("Einstein's relativity", 28, C_VEL)).arrange(RIGHT, buff=0.3)
        qm = VGroup(txt("atoms and smaller", 28), txt("→", 30, MUTED),
                    bold("quantum mechanics", 28, C_MOM)).arrange(RIGHT, buff=0.3)
        grp = VGroup(good, rel, qm).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(v(0, 0.2))
        with self.say("One last remark. Newton's laws describe everything from cricket balls to planets and "
                      "spacecraft with superb accuracy. They need to be modified only at speeds close to the speed "
                      "of light, where we use Einstein's relativity, and for atoms and smaller, where we use quantum "
                      "mechanics.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(good), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(rel), run_time=0.8)
            self.wait_until(c, c.start_of(2) + 0.55 * (c.end_of(2) - c.start_of(2)))
            self.play(FadeIn(qm), run_time=0.8)
        self.play(FadeOut(VGroup(head, grp)), FadeOut(self.label), run_time=0.7)
