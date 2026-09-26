"""Part 7 -- Solving problems: the five-step free-body method, and worked example 1."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, block, boxed, force, ground, panel, surface, v
from lesson.style import *

G = 9.8
STEPS = [
    ("Choose the body", "the object (or system) you want to study"),
    ("Draw its free-body diagram", "every external force acting ON that body, and nothing else"),
    ("Choose axes", "ideally with one axis along the acceleration"),
    ("Apply the second law", r"\sum F_x = m a_x \qquad \sum F_y = m a_y"),
    ("Solve and check", "units, signs, and whether the answer makes sense"),
]


def steps_panel(active: int | None = None, size: float = 28) -> VGroup:
    rows = VGroup()
    for i, (head, detail) in enumerate(STEPS, 1):
        num = bold(str(i), size + 4, ACCENT)
        h = bold(head, size)
        d = tex(detail, size=size + 4, color=MUTED) if "\\" in detail else txt(detail, size - 4, MUTED)
        rows.add(VGroup(num, VGroup(h, d).arrange(DOWN, aligned_edge=LEFT, buff=0.08)).arrange(RIGHT, buff=0.3,
                                                                                            aligned_edge=UP))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    return rows


class S07_FreeBody(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "7", "Solving problems", "free-body diagrams")
        self.method()
        self.force_types()
        self.example_one()

    # ------------------------------------------------------------------ the method
    def method(self):
        head = bold("Five steps for any force problem", 38, ACCENT).move_to(v(0, 3.0))
        rows = steps_panel().next_to(head, DOWN, buff=0.45)
        rows.set_x(0)
        with self.say("Now let's put the laws to work. Almost every problem about forces and motion can be "
                      "solved with the same five steps. One: choose the body you want to study. Two: draw its "
                      "free-body diagram, showing every external force acting on that body, and nothing else. "
                      "Three: choose axes, ideally with one axis along the direction of the acceleration. Four: "
                      "apply Newton's second law along each axis. Five: solve, and check that your answer "
                      "makes sense.") as c:
            self.play(FadeIn(head), run_time=0.8)
            for i in range(5):
                self.at_sentence(c, 2 + i)
                self.play(FadeIn(rows[i], shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeOut(VGroup(head, rows)), run_time=0.6)

    # ------------------------------------------------------------------ common forces
    def force_types(self):
        head = bold("The forces you'll meet most often", 36).move_to(v(0, 2.9))
        cards = VGroup(*[panel(2.55, 3.6) for _ in range(5)]).arrange(RIGHT, buff=0.2).move_to(v(0, -0.2))
        titles = ["Weight", "Normal force", "Tension", "Friction", "Applied force"]
        subs = ["W = mg,\nstraight down", "perpendicular to\nthe surface", "along the string,\npulling",
                "along the surface,\nopposing sliding", "a push or a pull"]
        cols = [C_WEIGHT, C_NORMAL, C_TENSION, C_FRICTION, C_FORCE]
        items = VGroup()
        for card, t, sb, col in zip(cards, titles, subs, cols):
            tt = bold(t, 24, col).next_to(card.get_top(), DOWN, buff=0.2)
            st = txt(sb, 19, MUTED, line_spacing=0.8).next_to(card.get_bottom(), UP, buff=0.2)
            items.add(VGroup(tt, st))
        c0 = cards[0].get_center() + UP * 0.25
        b0 = block(0.8, 0.6).move_to(c0 + UP * 0.35)
        a0 = arrow(b0.get_center(), b0.get_center() + DOWN * 1.1, C_WEIGHT)
        c1 = cards[1].get_center() + DOWN * 0.15
        s1 = Line(c1 + v(-1.0, -0.3), c1 + v(1.0, 0.3), color=GROUND, stroke_width=4)
        nrm = normalize(v(-0.3, 1.0))
        b1 = block(0.7, 0.5).rotate(np.arctan2(0.6, 2.0)).move_to(c1 + nrm * 0.28)
        a1 = arrow(b1.get_center(), b1.get_center() + nrm * 0.85, C_NORMAL)
        c2 = cards[2].get_center()
        rope = Line(c2 + UP * 1.2, c2 + DOWN * 0.1, color=MUTED, stroke_width=3)
        b2 = block(0.7, 0.55).next_to(rope, DOWN, buff=0)
        a2 = arrow(b2.get_center(), b2.get_center() + UP * 1.0, C_TENSION)
        c3 = cards[3].get_center() + DOWN * 0.1
        s3 = Line(c3 + v(-1.0, -0.3), c3 + v(1.0, -0.3), color=GROUND, stroke_width=4)
        b3 = block(0.8, 0.5).move_to(c3 + v(0.2, -0.05))
        v3 = arrow(b3.get_top() + v(-0.3, 0.2), b3.get_top() + v(0.4, 0.2), C_VEL, stroke=4, tip=0.15)
        a3 = arrow(b3.get_bottom() + UP * 0.02, b3.get_bottom() + v(-0.95, 0.02), C_FRICTION)
        c4 = cards[4].get_center()
        b4 = block(0.8, 0.6).move_to(c4 + LEFT * 0.35)
        a4 = arrow(b4.get_right(), b4.get_right() + RIGHT * 0.9 + UP * 0.35, C_FORCE)
        pics = [VGroup(b0, a0), VGroup(s1, b1, a1), VGroup(rope, b2, a2), VGroup(s3, b3, v3, a3), VGroup(b4, a4)]
        with self.say("These are the forces you will meet most often. Weight, m g, acting straight down. The "
                      "normal force, perpendicular to a surface in contact. Tension, pulling along a string or "
                      "rope. Friction, along a surface, opposing sliding. And any applied push or pull.") as c:
            self.play(FadeIn(head), run_time=0.6)
            for i in range(5):
                self.at_sentence(c, 1 + i)
                self.play(FadeIn(cards[i]), FadeIn(items[i]), FadeIn(pics[i]), run_time=0.7)
        self.play(FadeOut(VGroup(head, cards, items, *pics)), run_time=0.6)

    # ------------------------------------------------------------------ example 1
    def example_one(self):
        m, P, th = 5.0, 20.0, np.radians(30)
        W = m * G
        Px, Py = P * np.cos(th), P * np.sin(th)
        a = Px / m
        N = W - Py
        prob_box = panel(13.2, 1.2).move_to(v(0, 2.7))
        prob = lines(f"{hl('Example 1.')} A 5 kg box rests on a smooth (frictionless) floor. A rope pulls it with",
                     "a 20 N force at 30° above the horizontal. Find its acceleration and the normal force. (g = 9.8 m/s²)",
                     size=25, buff=0.12).move_to(prob_box)
        if prob.width > 12.8:
            prob.scale_to_fit_width(12.8)
        # picture (left)
        fy = -1.6
        flr = surface(-6.8, -1.2, fy, "#94A3B8", depth=0.25, opacity=0.35)
        bx = block(1.4, 1.0, "5 kg").move_to(v(-4.6, fy + 0.5))
        rope_dir = v(np.cos(th), np.sin(th))
        attach = bx.get_right() + UP * 0.1
        rope = Line(attach, attach + rope_dir * 2.4, color=MUTED, stroke_width=3)
        pull = force(attach, rope_dir * 1.6, C_FORCE, r"20\ \text{N}", label_dir=UP)
        ang = Angle(Line(attach, attach + RIGHT), Line(attach, attach + rope_dir), radius=0.7, color=TEXT)
        ang_l = tex(r"30^\circ", size=28).next_to(ang, RIGHT, buff=0.08).shift(UP * 0.05)
        hline = DashedLine(attach, attach + RIGHT * 1.2, color=DIM, stroke_width=2)
        picture = VGroup(flr, bx, rope, hline, pull, ang, ang_l)
        # free-body diagram (right), 1 unit = 20 N
        k = 1 / 25
        O = v(2.9, -0.55)
        dot = Dot(O, radius=0.08, color=TEXT)
        fW = force(O, DOWN * W * k, C_WEIGHT, r"W = 49\ \text{N}", label_dir=RIGHT, label_size=30)
        fN = force(O, UP * N * k, C_NORMAL, r"N", label_dir=RIGHT, label_size=32)
        fP = force(O, rope_dir * P * k, C_FORCE, r"20\ \text{N}", label_dir=UR, label_size=30)
        xax = Arrow(O + LEFT * 1.8, O + RIGHT * 3.0, buff=0, color=DIM, stroke_width=2, tip_length=0.16)
        yax = Arrow(O + DOWN * 2.3, O + UP * 1.85, buff=0, color=DIM, stroke_width=2, tip_length=0.16)
        xl = tex("x", size=30, color=MUTED).next_to(xax.get_end(), DOWN, buff=0.12)
        yl = tex("y", size=30, color=MUTED).next_to(yax.get_end(), LEFT, buff=0.12)
        fbd_t = txt("free-body diagram of the box (arrows to scale)", 22, MUTED).move_to(v(2.9, 1.75))
        comp_x = DashedLine(O, O + RIGHT * Px * k, color=C_FORCE, stroke_width=3, dash_length=0.08)
        comp_y = DashedLine(O + RIGHT * Px * k, O + RIGHT * Px * k + UP * Py * k, color=C_FORCE,
                            stroke_width=3, dash_length=0.08)
        cx_l = tex(r"20\cos 30^\circ \approx 17.3\ \text{N}", size=28, color=C_FORCE)
        cx_l.next_to(O, DR, buff=0.12).shift(DOWN * 0.05 + RIGHT * 0.15)
        cy_l = tex(r"20\sin 30^\circ = 10\ \text{N}", size=28, color=C_FORCE).next_to(comp_y, RIGHT, buff=0.15)
        cy_l.shift(UP * 0.12)

        with self.say("Example one. A 5 kilogram box rests on a smooth, frictionless floor. A rope pulls it with "
                      "a force of 20 newtons, at 30 degrees above the horizontal. Find the acceleration of the "
                      "box, and the normal force from the floor. Take g as 9.8 metres per second squared.") as c:
            self.play(FadeIn(prob_box), FadeIn(prob), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(flr), FadeIn(bx), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(Create(rope), GrowArrow(pull[0]), FadeIn(pull[1]), Create(hline), Create(ang),
                      FadeIn(ang_l), run_time=1.2)
        with self.say("Step one: the body is the box. Step two, the free-body diagram. Its weight is 5 times "
                      "9.8, which is 49 newtons, straight down. The floor pushes up with a normal force, N, "
                      "which we don't know yet. And the rope pulls with 20 newtons, at 30 degrees.") as c:
            self.play(Indicate(bx, color=ACCENT), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(fbd_t), FadeIn(dot), run_time=0.6)
            self.add_foreground_mobject(dot)
            self.at_sentence(c, 2)
            self.play(GrowArrow(fW[0]), FadeIn(fW[1]), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(GrowArrow(fN[0]), FadeIn(fN[1]), run_time=0.8)
            self.at_sentence(c, 4)
            self.play(GrowArrow(fP[0]), FadeIn(fP[1]), run_time=0.8)
        with self.say("Step three: take x along the floor, because that's the direction in which the box can "
                      "accelerate, and y straight up. Then split the pull into components: 20 times the cosine "
                      "of 30 degrees, about 17.3 newtons, along x; and 20 times the sine of 30 degrees, which is "
                      "10 newtons, along y.") as c:
            self.bring_to_back(xax, yax)
            self.play(GrowArrow(xax), GrowArrow(yax), FadeIn(xl), FadeIn(yl), run_time=0.8)
            self.bring_to_back(xax, yax)
            self.at_sentence(c, 1)
            self.play(Create(comp_x), FadeIn(cx_l), run_time=1.0)
            self.play(Create(comp_y), FadeIn(cy_l), run_time=1.0)
        # --- solve (equations replace the picture on the left)
        eqx = VGroup(txt("along x:", 26, MUTED),
                     tex(r"17.3 = 5\,a \;\Rightarrow\; a \approx 3.46\ \text{m/s}^2", size=38)).arrange(RIGHT, buff=0.3)
        eqy = VGroup(txt("along y:", 26, MUTED),
                     tex(r"N + 10 - 49 = 0 \;\Rightarrow\; N = 39\ \text{N}", size=38)).arrange(RIGHT, buff=0.3)
        eqs = VGroup(eqx, eqy).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(v(-3.3, -0.3))
        why_y = txt("(no vertical acceleration: the box stays on the floor)", 20, MUTED).next_to(eqy, DOWN, buff=0.15).align_to(eqy, LEFT)
        with self.say("Step four. Along x, the only force is 17.3 newtons, so the acceleration is 17.3 divided "
                      "by 5: about 3.46 metres per second squared. Along y, the box does not accelerate, because "
                      "it stays on the floor, so the forces balance: N plus 10, minus 49, equals zero. So N is "
                      "39 newtons.") as c:
            self.play(FadeOut(picture), run_time=0.5)
            self.play(FadeIn(eqx[0]), Write(eqx[1]), run_time=1.6)
            self.at_sentence(c, 2)
            self.play(FadeIn(eqy[0]), Write(eqy[1]), FadeIn(why_y), run_time=1.8)
        check = panel(7.0, 1.5, fill="#3B2F0B", stroke=ACCENT).move_to(v(-3.3, -2.55))
        check_t = lines(f"{hl('N = 39 N')}, not mg = 49 N:",
                        "the normal force is not always equal to mg!", size=24, buff=0.1, align=ORIGIN).move_to(check)
        with self.say("Step five: check. The normal force is 39 newtons, less than the weight of 49 newtons, "
                      "because the rope is already supporting 10 newtons of it. So remember: the normal force is "
                      "not always equal to m g. It is whatever the surface needs to supply.") as c:
            self.play(Indicate(fN, color=C_NORMAL), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(FadeIn(check), FadeIn(check_t), run_time=0.8)
        self.remove_foreground_mobject(dot)
        self.play(FadeOut(VGroup(prob_box, prob, dot, fW, fN, fP, xax, yax, xl, yl, fbd_t, comp_x, comp_y,
                                 cx_l, cy_l, eqs, why_y, check, check_t)), FadeOut(self.label), run_time=0.7)
