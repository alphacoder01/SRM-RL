"""Title, three motivating questions, and the lesson roadmap."""
import numpy as np
from manim import *

from lesson.narration import VoiceScene
from lesson.objects import (arrow, bathroom_scale, book, lift_car, panel, person,
                            rocket, surface, v)
from lesson.style import *


class S00_Intro(VoiceScene):
    def construct(self):
        # ---------------------------------------------------------------- title
        title = bold("Newton's Laws of Motion", 68)
        rule = Line(LEFT * 4.2, RIGHT * 4.2, color=ACCENT, stroke_width=3)
        sub = txt("Grade 12 Physics  ·  force, inertia and interaction", 30, MUTED)
        head = VGroup(title, rule, sub).arrange(DOWN, buff=0.32)
        with self.say("Newton's laws of motion are the foundation of classical mechanics. "
                      "In this lesson, we'll build them up carefully, one law at a time."):
            self.play(Write(title), run_time=1.8)
            self.play(GrowFromCenter(rule), FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.play(FadeOut(head, shift=UP * 0.3), run_time=0.7)

        # ---------------------------------------------------------------- three questions
        panels = VGroup(*[panel(4.25, 3.9) for _ in range(3)]).arrange(RIGHT, buff=0.3)
        panels.move_to(v(0, -0.35))
        q_texts = ["Why does a puck glide\nso far on ice?",
                   "How does a rocket speed\nup in empty space?",
                   "Why do you feel heavier\nin a lift going up?"]
        questions = VGroup(*[txt(q, 26, TEXT, line_spacing=0.9).next_to(p, UP, buff=0.25)
                             for q, p in zip(q_texts, panels)])

        # Panel 1: puck on ice keeps going, book on a table stops quickly.
        p1 = panels[0]
        x0, x1 = p1.get_left()[0] + 0.25, p1.get_right()[0] - 0.25
        ice = surface(x0, x1, p1.get_center()[1] + 0.35, "#7DD3FC")
        wood = surface(x0, x1, p1.get_center()[1] - 1.05, "#B45309")
        ice_lab = txt("ice", 22, "#7DD3FC").next_to(ice, DOWN, buff=0.12).align_to(ice, LEFT)
        wood_lab = txt("table", 22, "#F59E0B").next_to(wood, DOWN, buff=0.12).align_to(wood, LEFT)
        puck = Ellipse(width=0.55, height=0.2, fill_color="#0B1220", fill_opacity=1,
                       stroke_color=OBJ_STROKE, stroke_width=2)
        puck_start = v(x0 + 0.4, ice[1].get_y() + 0.1)
        puck.move_to(puck_start)
        bk = book(0.8, 0.22)
        book_start = v(x0 + 0.5, wood[1].get_y() + 0.11)
        bk.move_to(book_start)

        # Panel 2: rocket among stars.
        p2 = panels[1]
        rng = np.random.default_rng(7)
        stars = VGroup(*[Dot(p2.get_center() + v(rng.uniform(-1.95, 1.95), rng.uniform(-1.75, 1.75)),
                             radius=rng.uniform(0.012, 0.03), color=WHITE).set_opacity(rng.uniform(0.4, 1))
                         for _ in range(45)])
        rk = rocket(1.5).move_to(p2.get_center() + v(0, -0.3))

        # Panel 3: person on a scale in a lift that starts moving up.
        p3 = panels[2]
        frame = lift_car(1.9, 2.9).frame
        frame.move_to(p3.get_center() + v(-0.45, -0.25))
        cable_top = v(frame.get_x(), p3.get_top()[1] - 0.05)
        cable = always_redraw(lambda: Line(frame.get_top(), cable_top, color=MUTED, stroke_width=3))
        sc = bathroom_scale(0.8, 0.16).move_to(frame.get_bottom() + v(0, 0.12))
        guy = person(1.55).move_to(sc.get_top(), aligned_edge=DOWN)
        acc = arrow(v(p3.get_right()[0] - 0.55, p3.get_center()[1] - 0.9),
                    v(p3.get_right()[0] - 0.55, p3.get_center()[1] + 0.5), C_ACC)
        acc_lab = tex(r"\vec a", size=36, color=C_ACC).next_to(acc, LEFT, buff=0.1)
        lift_group = VGroup(frame, sc, guy)

        t = ValueTracker(0.0)

        def book_x(tt):
            v0, dec = 1.3, 2.0  # decelerates to rest (friction)
            tt = min(tt, v0 / dec)
            return v0 * tt - 0.5 * dec * tt**2

        puck.add_updater(lambda m: m.move_to(puck_start + RIGHT * min(0.95 * t.get_value(), 2.9)))
        bk.add_updater(lambda m: m.move_to(book_start + RIGHT * book_x(t.get_value())))

        with self.say("Why does a hockey puck glide so far across the ice, while a book pushed "
                      "across a table stops almost at once? How can a rocket speed up in the "
                      "emptiness of space, where there is nothing to push against? And why do you "
                      "feel heavier in a lift that is just starting to go up?") as c:
            self.play(FadeIn(panels[0]), FadeIn(questions[0]), FadeIn(ice), FadeIn(wood),
                      FadeIn(ice_lab), FadeIn(wood_lab), FadeIn(puck), FadeIn(bk), run_time=0.8)
            self.play(t.animate.set_value(3.2), run_time=3.2, rate_func=linear)
            self.at_sentence(c, 1)
            self.play(FadeIn(panels[1]), FadeIn(questions[1]), FadeIn(stars), FadeIn(rk), run_time=0.8)
            self.play(rk.animate.shift(UP * 0.6), run_time=2.2, rate_func=rate_functions.ease_in_quad)
            self.at_sentence(c, 2)
            self.play(FadeIn(panels[2]), FadeIn(questions[2]), FadeIn(lift_group), FadeIn(cable),
                      run_time=0.8)
            self.play(GrowArrow(acc), FadeIn(acc_lab), lift_group.animate.shift(UP * 0.25),
                      run_time=1.6, rate_func=rate_functions.ease_in_quad)
        puck.clear_updaters()
        bk.clear_updaters()

        cable.clear_updaters()
        everything = VGroup(panels, questions, ice, wood, ice_lab, wood_lab, puck, bk, stars, rk,
                            lift_group, cable, acc, acc_lab)

        # ---------------------------------------------------------------- Newton
        name = bold("Isaac Newton", 54)
        book_title = txt("Philosophiæ Naturalis Principia Mathematica", 32, TEXT, slant=ITALIC)
        year = txt("(Mathematical Principles of Natural Philosophy, 1687)", 26, MUTED)
        credit = VGroup(name, book_title, year).arrange(DOWN, buff=0.28).move_to(v(0, 0.9))
        three = VGroup(*[panel(3.6, 1.2) for _ in range(3)]).arrange(RIGHT, buff=0.35).move_to(v(0, -1.7))
        three_labels = VGroup(*[bold(s, 32, ACCENT).move_to(p) for s, p in
                                zip(["First law", "Second law", "Third law"], three)])
        with self.say("All three questions have the same answer: three short laws, published by "
                      "Isaac Newton in 1687, in his book called the Principia. They describe how "
                      "forces change the motion of everything from cricket balls to planets.") as c:
            self.play(FadeOut(everything), run_time=0.7)
            self.play(Write(name), run_time=1.0)
            self.play(FadeIn(book_title, shift=UP * 0.15), FadeIn(year, shift=UP * 0.15), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(p), FadeIn(l))
                                    for p, l in zip(three, three_labels)], lag_ratio=0.35), run_time=1.6)
        self.play(FadeOut(VGroup(credit, three, three_labels)), run_time=0.6)

        # ---------------------------------------------------------------- roadmap
        heading = bold("The plan", 46).to_edge(UP, buff=0.55)
        items = ["What keeps things moving?", "The first law: inertia",
                 "The second law: force and acceleration", "Impulse",
                 "The third law: forces come in pairs", "Conservation of momentum",
                 "Solving problems with free-body diagrams", "Summary and quiz"]
        rows = VGroup()
        for i, it in enumerate(items, 1):
            num = bold(str(i), 30, ACCENT)
            rows.add(VGroup(num, txt(it, 30)).arrange(RIGHT, buff=0.35))
        for i, r in enumerate(rows):  # fixed row pitch, independent of descenders
            r.shift(v(-4.2, 1.75 - 0.6 * i) - r[0].get_center())
        rows.set_x(0)
        with self.say("Here is our plan. We start with the key idea of inertia. Then we take each "
                      "law in turn, and see how the second and third laws lead to impulse and to the "
                      "conservation of momentum. Next, we learn a step-by-step method for solving "
                      "problems with free-body diagrams, and we finish with a summary and a short quiz.") as c:
            self.play(FadeIn(heading, shift=DOWN * 0.2), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.5),
                      run_time=self.remaining(c) * 0.8)
        self.play(FadeOut(VGroup(heading, rows)), run_time=0.6)
