"""Part 7 (continued) -- Worked example 3: two blocks joined by a string."""
import numpy as np
from manim import *

from lesson.narration import VoiceScene
from lesson.objects import arrow, block, boxed, check_mark, force, panel, surface, v
from lesson.style import *

K = 0.12  # scene units per newton for the force arrows


class S09_Connected(VoiceScene):
    def construct(self):
        self.label = chapter_label("7", "Solving problems")
        self.play(FadeIn(self.label), run_time=0.5)
        prob_box = panel(13.2, 1.2).move_to(v(0, 2.7))
        prob = lines(f"{hl('Example 3.')} Blocks of 4 kg (front) and 2 kg (behind) rest on a frictionless floor, joined by a",
                     "light string. A 12 N horizontal force pulls the front block. Find the acceleration and the tension.",
                     size=24, buff=0.1).move_to(prob_box)
        if prob.width > 12.8:
            prob.scale_to_fit_width(12.8)
        fy = 0.25
        flr = surface(-6.9, 6.9, fy, "#94A3B8", depth=0.22, opacity=0.35)
        rear = block(1.2, 0.9, "2 kg").move_to(v(-2.4, fy + 0.45))
        front = block(1.6, 1.2, "4 kg", fill=OBJ_FILL_2).move_to(v(0.9, fy + 0.6))
        string = Line(rear.get_right(), v(front.get_left()[0], rear.get_y()), color=MUTED, stroke_width=3)
        pull = force(front.get_right(), RIGHT * 12 * K, C_FORCE, r"12\ \text{N}", label_dir=UP)
        picture = VGroup(flr, rear, front, string, pull)
        with self.say("Example three: connected bodies. Two blocks rest on a frictionless floor, joined by a "
                      "light string: a 4 kilogram block in front, and a 2 kilogram block behind. A 12 newton force "
                      "pulls the front block. Find the acceleration, and the tension in the string.") as c:
            self.play(FadeIn(prob_box), FadeIn(prob), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(flr), FadeIn(rear), FadeIn(front), Create(string), run_time=0.9)
            self.at_sentence(c, 2)
            self.play(GrowArrow(pull[0]), FadeIn(pull[1]), run_time=0.8)

        # ---- step 1: the whole system
        sys_box = DashedVMobject(SurroundingRectangle(VGroup(rear, front), buff=0.25, corner_radius=0.15),
                                 num_dashes=60).set_stroke(ACCENT, 2.5)
        sys_lab = txt("system: both blocks, 6 kg", 24, ACCENT).next_to(sys_box, DOWN, buff=0.45)
        internal = txt("the string's pulls are internal: they cancel", 22, MUTED).next_to(sys_lab, DOWN, buff=0.12)
        eq_sys = tex(r"a = \frac{F}{m_{\text{total}}} = \frac{12\ \text{N}}{6\ \text{kg}} = 2\ \text{m/s}^2",
                     size=40).move_to(v(4.3, -1.55))
        with self.say("First, treat both blocks together as one system. The pulls of the string on the two blocks "
                      "are internal forces: they are equal and opposite, so they cancel within the system. The only "
                      "external horizontal force is the 12 newton pull, acting on a total mass of 6 kilograms. So "
                      "the acceleration is 12 divided by 6: 2 metres per second squared.") as c:
            self.play(Create(sys_box), FadeIn(sys_lab), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(FadeIn(internal), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(Write(eq_sys), run_time=1.6)
        self.play(FadeOut(VGroup(sys_box, sys_lab, internal)), FadeOut(prob_box), FadeOut(prob),
                  picture.animate.shift(UP * 1.0), eq_sys.animate.scale(0.85).move_to(v(-4.6, 2.75)), run_time=0.8)

        # ---- step 2: free-body diagram of the rear block
        y2 = -2.2
        rb = block(1.2, 0.9, "2 kg").move_to(v(-5.3, y2))
        T_r = force(rb.get_right(), RIGHT * 4 * K * 2.2, C_TENSION, r"T", label_dir=UP)
        t1 = txt("rear block alone", 24, MUTED).next_to(rb, UP, buff=0.55).align_to(rb, LEFT)
        eq_r = tex(r"T = m\,a = 2 \times 2 = 4\ \text{N}", size=38).next_to(rb, RIGHT, buff=1.6)
        note_r = txt("(vertical forces balance, so they're not drawn)", 20, MUTED).next_to(eq_r, DOWN, buff=0.15).align_to(eq_r, LEFT)
        with self.say("To find the tension, draw a free-body diagram of the rear block alone. The only "
                      "horizontal force on it is the tension, T. So T equals 2 kilograms times 2 metres per "
                      "second squared, which is 4 newtons.") as c:
            self.play(FadeIn(t1), FadeIn(rb), run_time=0.7)
            self.at_sentence(c, 1)
            self.play(GrowArrow(T_r[0]), FadeIn(T_r[1]), FadeIn(note_r), run_time=0.9)
            self.at_sentence(c, 2)
            self.play(Write(eq_r), run_time=1.4)

        # ---- step 3: check with the front block
        fb = block(1.6, 1.2, "4 kg", fill=OBJ_FILL_2).move_to(v(-4.9, -3.35))
        self.play(VGroup(t1, rb, T_r, eq_r, note_r).animate.shift(UP * 0.55), run_time=0.5)
        fb.move_to(v(-4.9, -3.2))
        F_f = force(fb.get_right(), RIGHT * 12 * K, C_FORCE, r"12\ \text{N}", label_dir=UP, label_size=28)
        T_f = force(fb.get_left(), LEFT * 4 * K * 2.2, C_TENSION, r"T", label_dir=UP)
        eq_f = tex(r"12 - T = 12 - 4 = 8\ \text{N} = 4 \times 2\ \checkmark", size=38).move_to(v(1.9, -3.2))
        with self.say("Check with the front block: 12 newtons forward, minus 4 newtons of tension backward, leaves "
                      "8 newtons, which is exactly 4 kilograms times 2. It works. And notice that the tension is less "
                      "than 12 newtons, because the string only has to accelerate the rear block.") as c:
            self.play(FadeIn(fb), GrowArrow(F_f[0]), FadeIn(F_f[1]), GrowArrow(T_f[0]), FadeIn(T_f[1]), run_time=1.0)
            self.play(Write(eq_f), run_time=1.6)
            self.at_sentence(c, 2)
            tens = mtxt(f"{hl('T = 4 N &lt; 12 N', C_TENSION)}: the string only accelerates the rear block",
                        24).move_to(v(0.9, 0.3))
            self.play(FadeIn(tens), run_time=0.8)
        # ---- the pair accelerates together (1 unit = 1 m, shown at 0.5x speed)
        clock = ValueTracker(0.0)
        movers = VGroup(rear, front, string, pull)  # the floor stays put
        base = movers.get_center()
        movers.add_updater(lambda m: m.move_to(base + RIGHT * 0.5 * 2.0 * clock.get_value() ** 2))
        vel = always_redraw(lambda: arrow(rear.get_top() + v(0.2, 0.3),
                                          rear.get_top() + v(0.2 + 0.35 * 2.0 * clock.get_value(), 0.3),
                                          C_VEL, stroke=5, tip=0.16))
        vel2 = always_redraw(lambda: arrow(front.get_top() + v(0.2, 0.3),
                                           front.get_top() + v(0.2 + 0.35 * 2.0 * clock.get_value(), 0.3),
                                           C_VEL, stroke=5, tip=0.16))
        same = txt("both move together with a = 2 m/s²  (shown at 0.5× speed)", 22, C_VEL).move_to(v(2.6, 3.3))
        with self.say("Both blocks share the same velocity and the same acceleration, because the string keeps "
                      "them together."):
            self.add(vel, vel2)
            self.play(FadeIn(same), run_time=0.5)
            self.play(clock.animate.set_value(1.6), run_time=3.2, rate_func=linear)
        for m_ in (movers, vel, vel2):
            m_.clear_updaters()
        self.play(FadeOut(VGroup(picture, vel, vel2, same, eq_sys, t1, rb, T_r, eq_r, note_r, fb, F_f, T_f, eq_f,
                                 tens)), FadeOut(self.label), run_time=0.7)
