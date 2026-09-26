"""Part 4 -- Impulse: J = F dt = dp; catching a cricket ball."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, boxed, panel, shown, v
from lesson.style import *

STIFF, SOFT = "#F87171", "#4ADE80"


class S04_Impulse(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "4", "Impulse", "force acting over time")
        self.definition()
        self.catch()
        self.applications()

    # ------------------------------------------------------------------ definition
    def definition(self):
        eq1 = tex(r"\vec F_{\text{net}}", r"=", r"\frac{d\vec p}{dt}", size=54).move_to(v(-3.6, 2.2))
        eq2 = tex(r"\Delta\vec p", r"=", r"\vec F\,\Delta t", size=54).move_to(v(3.3, 2.2))
        eq2[0].set_color(C_MOM)
        cst = txt("(constant force)", 24, MUTED).next_to(eq2, DOWN, buff=0.2)
        ax = Axes(x_range=[0, 1.0, 0.2], y_range=[0, 10, 2], x_length=6.0, y_length=3.2,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False,
                               "include_ticks": False}).move_to(v(-2.9, -1.2))
        xl = txt("time", 22, MUTED).next_to(ax.x_axis, DOWN, buff=0.15).align_to(ax.x_axis, RIGHT)
        yl = txt("force", 22, C_FORCE).next_to(ax.y_axis, UP, buff=0.12)

        def f(t):  # a smooth collision pulse between t = 0.15 and t = 0.85
            if t <= 0.15 or t >= 0.85:
                return 0.0
            return 8.5 * np.sin(np.pi * (t - 0.15) / 0.7) ** 2

        curve = ax.plot(f, x_range=[0, 1.0, 0.005], color=C_FORCE, stroke_width=4)
        area = ax.get_area(curve, x_range=[0.15, 0.85], color=C_MOM, opacity=0.45)
        f_avg = 8.5 / 2  # mean of sin^2 over the pulse
        rect = Polygon(ax.c2p(0.15, 0), ax.c2p(0.15, f_avg), ax.c2p(0.85, f_avg), ax.c2p(0.85, 0),
                       stroke_color=TEXT, stroke_width=2.5, fill_opacity=0)
        rect_lab = tex(r"F_{\text{avg}}", size=30).next_to(rect, LEFT, buff=0.1).shift(UP * 0.35)
        brace = BraceBetweenPoints(ax.c2p(0.15, 0), ax.c2p(0.85, 0), DOWN, color=MUTED)
        dt = tex(r"\Delta t", size=30, color=MUTED).next_to(brace, DOWN, buff=0.08)
        area_lab = txt("area = impulse", 26, C_MOM).move_to(ax.c2p(0.5, 2.0))
        defs = VGroup(
            tex(r"\vec J", r"=", r"\int \vec F\,dt", r"=", r"\vec F_{\text{avg}}\,\Delta t", size=46),
            tex(r"\vec J", r"=", r"\Delta \vec p", size=56),
            txt("unit: newton second (N·s) = kg·m/s", 26, MUTED),
        ).arrange(DOWN, buff=0.35).move_to(v(3.7, -1.3))
        defs[1][0].set_color(C_MOM)
        defs[1][2].set_color(C_MOM)
        bx = boxed(defs[1])
        with self.say("The momentum form of the second law leads to a very useful idea. If a constant "
                      "net force acts on a body for a time delta t, the momentum changes by the force "
                      "times delta t.") as c:
            self.play(Write(eq1), run_time=1.2)
            self.at_sentence(c, 1)
            self.play(Write(eq2), FadeIn(cst), run_time=1.4)
        with self.say("If the force varies, as it does in any collision, the change in momentum equals "
                      "the area under the force against time graph. We can also write it as the average "
                      "force times delta t. This quantity is called impulse, and its unit is the newton "
                      "second, which is the same as a kilogram metre per second.") as c:
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
            self.play(Create(curve), run_time=1.2)
            self.play(FadeIn(area), FadeIn(area_lab), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(Create(rect), FadeIn(rect_lab), GrowFromCenter(brace), FadeIn(dt), run_time=1.2)
            self.at_sentence(c, 2)
            self.play(Write(defs[0]), run_time=1.2)
            self.play(Write(defs[1]), Create(bx), run_time=1.0)
            self.play(FadeIn(defs[2]), run_time=0.6)
        self.play(FadeOut(VGroup(eq1, eq2, cst, ax, xl, yl, curve, area, rect, rect_lab, brace, dt,
                                 area_lab, defs, bx)), run_time=0.6)

    # ------------------------------------------------------------------ catching a ball
    def catch(self):
        m, u = 0.16, 25.0
        head = bold("Catching a cricket ball", 36).move_to(v(0, 3.0))
        calc = tex(r"\Delta p = m v = 0.16 \times 25 = 4\ \text{kg\,m/s}", size=40).move_to(v(0, 2.2))
        # Kinematics in real seconds; drawn at 1 unit = 0.5 m and shown at 1/50 speed.
        unit = 0.5
        slow = 1 / 50
        rows = [("stiff hands", 0.01, 0.9, STIFF), ("hands drawn back", 0.10, -1.7, SOFT)]
        x_contact = -2.9
        clock = ValueTracker(-0.06)
        mobs = VGroup()
        balls = []

        def x_ball(t, tau):
            dec = u / tau
            if t <= 0:
                return x_contact + (u * t) / unit
            tt = min(t, tau)
            return x_contact + (u * tt - 0.5 * dec * tt * tt) / unit

        for name, tau, y, col in rows:
            lab = bold(name, 26, col).move_to(v(-6.6, y + 0.75), aligned_edge=LEFT)
            ball_m = VGroup(Circle(radius=0.2, fill_color="#DC2626", fill_opacity=1, stroke_width=0),
                            Arc(radius=0.2, start_angle=-PI / 3, angle=2 * PI / 3, color="#FEE2E2",
                                stroke_width=2).shift(LEFT * 0.08))
            glove = RoundedRectangle(width=0.32, height=0.95, corner_radius=0.14, fill_color="#A16207",
                                     fill_opacity=1, stroke_color="#FDE68A", stroke_width=2)
            ball_m.add_updater(lambda mob, tau=tau, y=y: mob.move_to(v(x_ball(clock.get_value(), tau), y)))
            glove.add_updater(lambda mob, tau=tau, y=y: mob.move_to(
                v(max(x_ball(clock.get_value(), tau), x_contact) + 0.36, y)))
            F = m * u / tau  # average (constant) stopping force, N
            farrow = always_redraw(lambda b=ball_m, F=F, tau=tau, col=col: shown(
                arrow(b.get_left(), b.get_left() + LEFT * F / 150.0, col, stroke=6, tip=0.2),
                0 < clock.get_value() < tau))
            flab = tex(rf"F = \frac{{4}}{{{tau:g}}} = {F:.0f}\ \text{{N}}", size=34, color=col)
            flab.move_to(v(-4.2, y - 0.7))
            balls.append((ball_m, glove, farrow, flab))
            mobs.add(lab, ball_m, glove, farrow)
        slow_tag = txt("shown at 1/50 speed · force arrows to scale", 22, MUTED).to_corner(DL, buff=0.3)

        ax = Axes(x_range=[0, 0.12, 0.02], y_range=[0, 450, 100], x_length=5.4, y_length=3.6,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False, "font_size": 20},
                  x_axis_config={"numbers_to_include": [0.02, 0.04, 0.06, 0.08, 0.10],
                                 "decimal_number_config": {"num_decimal_places": 2}},
                  y_axis_config={"numbers_to_include": [100, 200, 300, 400]}).move_to(v(3.7, -0.9))
        xl = txt("t (s)", 20, MUTED).next_to(ax.x_axis, DOWN, buff=0.4).align_to(ax.x_axis, RIGHT)
        yl = txt("F (N)", 22, C_FORCE).next_to(ax.y_axis, UP, buff=0.12)
        r1 = Polygon(ax.c2p(0, 0), ax.c2p(0, 400), ax.c2p(0.01, 400), ax.c2p(0.01, 0),
                     fill_color=STIFF, fill_opacity=0.55, stroke_color=STIFF, stroke_width=2)
        r2 = Polygon(ax.c2p(0, 0), ax.c2p(0, 40), ax.c2p(0.10, 40), ax.c2p(0.10, 0),
                     fill_color=SOFT, fill_opacity=0.55, stroke_color=SOFT, stroke_width=2)
        l1 = tex(r"400\ \text{N} \times 0.01\ \text{s} = 4\ \text{N\,s}", size=28, color=STIFF)
        l1.next_to(ax.c2p(0.012, 400), RIGHT, buff=0.15)
        l2 = tex(r"40\ \text{N} \times 0.1\ \text{s} = 4\ \text{N\,s}", size=28, color=SOFT)
        l2.next_to(ax.c2p(0.05, 40), UP, buff=0.35)
        same = txt("equal areas: same impulse", 24, TEXT).next_to(ax, UP, buff=0.1).shift(RIGHT * 0.9)

        with self.say("Here's why it matters. A fielder catches a cricket ball of mass 0.16 kilograms, "
                      "travelling at 25 metres per second, and brings it to rest. The ball's change in "
                      "momentum is 0.16 times 25, which is 4 kilogram metres per second, however the ball "
                      "is caught.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.play(FadeIn(mobs), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(Write(calc), run_time=1.4)
        with self.say("If the fielder keeps her hands stiff, the ball stops in about a hundredth of a "
                      "second. The average force is then 4 divided by 0.01, which is 400 newtons. That "
                      "stings! But if she draws her hands back as she catches, and stops the ball over a "
                      "tenth of a second, the average force is only 40 newtons.") as c:
            self.play(FadeIn(slow_tag), run_time=0.4)
            self.play(clock.animate.set_value(0.11), run_time=0.17 / slow, rate_func=linear)
            self.at_sentence(c, 1)
            self.play(FadeIn(balls[0][3]), Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
            self.play(DrawBorderThenFill(r1), FadeIn(l1), run_time=1.0)
            self.at_sentence(c, 3)
            self.play(FadeIn(balls[1][3]), DrawBorderThenFill(r2), FadeIn(l2), run_time=1.0)
            self.play(FadeIn(same), run_time=0.6)
        for b in balls:
            for mob in b[:3]:
                mob.clear_updaters()
        self.remove(*[b[2] for b in balls])
        self.catch_mobs = VGroup(head, calc, mobs, *[b[3] for b in balls], slow_tag, ax, xl, yl, r1, r2, l1, l2, same)

    # ------------------------------------------------------------------ applications
    def applications(self):
        self.play(FadeOut(self.catch_mobs), run_time=0.6)
        head = mtxt(f"Same {hl('Δp', C_MOM)}  ·  {hl('10×', ACCENT)} the time  ·  {hl('1/10', ACCENT)} the force", 38)
        head.move_to(v(0, 2.5))
        items = lines("•  crumple zones and airbags in cars",
                      "•  thick soft mats in a high-jump pit",
                      "•  bending your knees when you land from a jump",
                      "•  a fielder drawing the hands back", size=32, buff=0.3)
        items.move_to(v(0, 0.0))
        rule = mtxt(f"longer stopping time  →  {hl('smaller average force')}", 34).move_to(v(0, -2.5))
        with self.say("Same change in momentum, ten times the time, one tenth of the force. The same idea "
                      "explains crumple zones and airbags in cars, the thick soft mats in a high-jump pit, "
                      "and why you bend your knees when you land from a jump. In every case, a longer "
                      "stopping time means a smaller average force.") as c:
            self.play(FadeIn(head), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in items], lag_ratio=0.6), run_time=3.0)
            self.at_sentence(c, 2)
            self.play(FadeIn(rule), run_time=0.8)
        self.play(FadeOut(VGroup(head, items, rule)), FadeOut(self.label), run_time=0.7)
