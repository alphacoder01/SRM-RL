"""Part 7 (continued) -- Worked example 2: apparent weight in a lift."""
import numpy as np
from manim import *

from lesson.narration import VoiceScene
from lesson.objects import arrow, bathroom_scale, force, panel, person, shown, v
from lesson.style import *

G = 9.8
M = 60.0
CASES = [  # (label, acceleration in m/s^2, up = +; comment)
    ("constant velocity", 0.0, "true weight"),
    ("accelerating upward", 2.0, "feels heavier"),
    ("accelerating downward", -2.0, "feels lighter"),
    ("free fall", -G, "weightless"),
]


class S08_Lift(VoiceScene):
    def construct(self):
        self.label = chapter_label("7", "Solving problems")
        self.play(FadeIn(self.label), run_time=0.5)
        self.setup_picture()
        self.fbd_and_formula()
        self.cases()
        self.direction_note()

    # ------------------------------------------------------------------ picture
    def setup_picture(self):
        prob_box = panel(13.2, 1.55).move_to(v(0, 2.6))
        prob = lines(f"{hl('Example 2.')} A 60 kg student stands on a bathroom scale in a lift. What does the scale read",
                     "when the lift (a) moves at constant velocity, (b) accelerates upward at 2 m/s²,",
                     "(c) accelerates downward at 2 m/s², (d) falls freely?  (g = 9.8 m/s²)",
                     size=24, buff=0.1).move_to(prob_box)
        if prob.width > 12.8:
            prob.scale_to_fit_width(12.8)
        shaft = VGroup(Line(v(-6.55, -3.7), v(-6.55, 1.75), color=DIM, stroke_width=3),
                       Line(v(-3.45, -3.7), v(-3.45, 1.75), color=DIM, stroke_width=3))
        car = Rectangle(width=2.5, height=3.0, stroke_color=OBJ_STROKE, stroke_width=4,
                        fill_color="#111827", fill_opacity=1).move_to(v(-5.0, -1.9))
        sc = bathroom_scale(1.0, 0.18).move_to(car.get_bottom() + v(0, 0.13))
        guy = person(1.9).move_to(sc.get_top(), aligned_edge=DOWN)
        self.base_y = car.get_y()
        self.lift = VGroup(car, sc, guy)
        self.car, self.scale_m = car, sc
        self.cable = always_redraw(lambda: Line(self.car.get_top(), v(self.car.get_x(), 1.75), color=MUTED,
                                                stroke_width=3))
        self.reading = ValueTracker(M * G)
        self.readout = always_redraw(lambda: VGroup(
            RoundedRectangle(width=2.5, height=0.62, corner_radius=0.1, fill_color="#0B1220", fill_opacity=1,
                             stroke_color=C_NORMAL, stroke_width=2),
            txt(f"scale: {self.reading.get_value():.0f} N", 26, C_NORMAL)).move_to(v(-5.0, 0.95)))
        self.prob = VGroup(prob_box, prob)
        with self.say("Example two: the lift. A 60 kilogram student stands on a bathroom scale inside a lift. A "
                      "scale measures how hard it pushes up on you, which is the normal force. By the third law, "
                      "that's also how hard you push down on it. So, what does the scale read when the lift moves "
                      "at constant velocity, when it accelerates upward at 2 metres per second squared, when it "
                      "accelerates downward at 2 metres per second squared, and when it falls freely?") as c:
            self.play(FadeIn(self.prob), run_time=0.8)
            self.play(FadeIn(shaft), FadeIn(self.lift), FadeIn(self.cable), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(self.readout), run_time=0.6)
        self.shaft = shaft

    # ------------------------------------------------------------------ FBD and formula
    def fbd_and_formula(self):
        self.k = 1 / 340.0  # units per newton
        O = v(-1.55, -1.3)
        self.O = O
        self.fbd_dot = Dot(O, radius=0.08, color=TEXT)
        fbd_t = txt("free-body diagram\nof the student", 22, MUTED, line_spacing=0.8).move_to(v(-1.55, 1.45))
        self.fW = force(O, DOWN * M * G * self.k, C_WEIGHT, r"mg = 588\ \text{N}", label_dir=RIGHT, label_size=28)
        self.Nval = ValueTracker(M * G)
        self.fN = always_redraw(lambda: shown(
            force(self.O, UP * max(self.Nval.get_value(), 1e-3) * self.k, C_NORMAL, r"N", label_dir=RIGHT,
                  label_size=32), self.Nval.get_value() > 1.0))
        eq1 = tex(r"N - mg = ma", size=40)
        eq2 = tex(r"N = m\,(g + a)", size=46)
        up = txt("(up is positive)", 22, MUTED)
        eqs = VGroup(eq1, eq2, up).arrange(DOWN, buff=0.25).move_to(v(3.8, 0.75))
        box2 = SurroundingRectangle(eq2, color=ACCENT, buff=0.15, corner_radius=0.1)
        self.fbd_t = fbd_t
        with self.say("The free-body diagram of the student has just two forces: the normal force N from the "
                      "scale, pointing up, and the weight, m g, which is 60 times 9.8, or 588 newtons, pointing "
                      "down. Taking up as positive, the second law gives N minus m g equals m a. So N equals m "
                      "times, g plus a.") as c:
            self.play(FadeOut(self.prob), run_time=0.5)
            self.play(FadeIn(fbd_t), FadeIn(self.fbd_dot), run_time=0.5)
            self.add(self.fN)
            self.play(GrowArrow(self.fW[0]), FadeIn(self.fW[1]), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(Write(eq1), FadeIn(up), run_time=1.2)
            self.at_sentence(c, 2)
            self.play(Write(eq2), Create(box2), run_time=1.2)
        self.eqs = VGroup(eqs, box2)
        self.play(self.eqs.animate.scale(0.8).move_to(v(3.8, 2.55)), run_time=0.7)

    # ------------------------------------------------------------------ the four cases
    def cases(self):
        rows = VGroup()
        head = VGroup(bold("motion of the lift", 22, MUTED), bold("a (m/s²)", 22, MUTED),
                      bold("scale reading N", 22, MUTED))
        xs = [1.05, 4.05, 5.75]
        y0 = 1.35
        for j, h in enumerate(head):
            h.move_to(v(xs[j], y0))
        head[0].align_to(v(0.2, 0), LEFT)
        rule = Line(v(0.2, y0 - 0.3), v(6.9, y0 - 0.3), color=DIM, stroke_width=2)
        for i, (lab, a, comment) in enumerate(CASES):
            y = y0 - 0.85 - 0.95 * i
            N = M * (G + a)
            c1 = txt(lab, 22).move_to(v(xs[0], y)).align_to(v(0.2, 0), LEFT)
            c2 = txt({0.0: "0", 2.0: "+2", -2.0: "−2"}.get(a, "−9.8"), 22).move_to(v(xs[1], y))
            c3 = VGroup(bold(f"{N:.0f} N", 24, C_NORMAL), txt(comment, 19, MUTED)).arrange(DOWN, buff=0.05)
            c3.move_to(v(xs[2], y))
            rows.add(VGroup(c1, c2, c3))
        self.play(FadeIn(head), Create(rule), run_time=0.6)

        acc_arrow = always_redraw(lambda: VGroup())
        texts = [
            "At constant velocity, a is zero, so N is 588 newtons. The scale shows your true weight, even "
            "though you are moving.",
            "Accelerating upward at 2 metres per second squared, N equals 60 times 11.8, which is 708 newtons. "
            "You feel heavier.",
            "Accelerating downward at 2 metres per second squared, N equals 60 times 7.8, which is 468 newtons. "
            "You feel lighter.",
            "And in free fall, the acceleration is g, downward, so N equals 60 times zero: zero. The scale reads "
            "nothing, and you feel weightless, even though gravity still pulls on you with 588 newtons.",
        ]
        calcs = [r"N = 60\,(9.8 + 0) = 588\ \text{N}", r"N = 60\,(9.8 + 2) = 708\ \text{N}",
                 r"N = 60\,(9.8 - 2) = 468\ \text{N}", r"N = 60\,(9.8 - 9.8) = 0"]
        for i, ((lab, a, comment), text, calc) in enumerate(zip(CASES, texts, calcs)):
            N = M * (G + a)
            calc_m = tex(calc, size=32).move_to(v(-1.55, -3.55))
            # a short, physically consistent motion of the lift car (1 unit = 1 m; clipped to stay in view)
            clock = ValueTracker(0.0)
            v0 = 0.6 if a == 0 else 0.0
            T = 1.6 if a != -G else 0.45
            y_start = self.base_y - 0.3 if a >= 0 else self.base_y + 0.3
            self.lift.move_to(v(self.lift.get_x(), y_start - (self.car.get_y() - self.lift.get_y())))
            start = self.lift.get_center()

            def y_of(t, v0=v0, a=a):
                return 0.25 * (v0 * t + 0.5 * a * t * t)  # drawn at 1 unit = 4 m

            upd = lambda m, start=start: m.move_to(start + UP * y_of(clock.get_value()))
            acc_vec = UP * np.sign(a) * (0.5 + 0.1 * abs(a)) if a != 0 else ORIGIN
            a_arrow = VGroup()
            if a != 0:
                a_arrow = VGroup(arrow(v(-3.0, -1.2), v(-3.0, -1.2) + acc_vec, C_ACC),
                                 tex(r"\vec a", size=32, color=C_ACC).next_to(v(-3.0, -1.2) + acc_vec / 2, RIGHT, buff=0.1))
            else:
                a_arrow = txt("a = 0", 26, C_ACC).move_to(v(-2.75, -1.2))
            with self.say(text) as c:
                self.lift.add_updater(upd)
                self.play(FadeIn(a_arrow), FadeIn(calc_m), self.Nval.animate.set_value(N),
                          self.reading.animate.set_value(N), run_time=1.0)
                self.play(clock.animate.set_value(T), run_time=max(T / 0.5, 1.0), rate_func=linear)
                self.lift.remove_updater(upd)
                self.play(FadeIn(rows[i], shift=RIGHT * 0.2), run_time=0.7)
                if i == 3:
                    self.at_sentence(c, 1)
                    self.play(Indicate(self.fW, color=C_WEIGHT), run_time=1.0)
            self.play(FadeOut(a_arrow), FadeOut(calc_m), run_time=0.4)
            self.lift.move_to(v(self.lift.get_x(), self.base_y - (self.car.get_y() - self.lift.get_y())))
        self.table = VGroup(head, rule, rows)

    # ------------------------------------------------------------------ direction, not motion
    def direction_note(self):
        note = panel(6.7, 1.2, fill="#3B2F0B", stroke=ACCENT).move_to(v(3.55, -3.25))
        nt = lines(f"What matters is the direction of {hl('a')}, not of v.",
                   "Moving down but slowing to a stop: a points UP, so N > mg.",
                   size=22, buff=0.12, align=ORIGIN).move_to(note)
        if nt.width > 6.4:
            nt.scale_to_fit_width(6.4)
        self.Nval.set_value(M * G)
        self.reading.set_value(M * G)
        with self.say("Notice that what matters is the direction of the acceleration, not the direction of "
                      "motion. A lift that is moving down, but slowing to a stop, is accelerating upward, so you "
                      "feel heavier, just as when it starts going up.") as c:
            self.play(FadeIn(note), FadeIn(nt), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(Indicate(self.table[2][1], color=ACCENT), run_time=1.2)
        self.fN.clear_updaters()
        self.cable.clear_updaters()
        self.readout.clear_updaters()
        self.play(FadeOut(VGroup(note, nt, self.table, self.eqs, self.fW, self.fN, self.fbd_dot, self.fbd_t,
                                 self.lift, self.cable, self.readout, self.shaft)), FadeOut(self.label),
                  run_time=0.7)
