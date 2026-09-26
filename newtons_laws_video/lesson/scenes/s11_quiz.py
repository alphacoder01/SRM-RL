"""Part 8 (continued) -- Quiz: three questions (the last one a full Atwood-machine calculation), and closing."""
import numpy as np
from manim import *

from lesson.narration import VoiceScene
from lesson.objects import arrow, block, boxed, force, panel, pause_badge, pulley, rocket, v
from lesson.style import *

G = 9.8


def countdown(seconds: float) -> tuple[VGroup, Animation]:
    """A shrinking ring used during 'pause and think' moments."""
    ring = Circle(radius=0.32, color=ACCENT, stroke_width=6)
    track = Circle(radius=0.32, color=DIM, stroke_width=6)
    grp = VGroup(track, ring)
    anim = Uncreate(ring, run_time=seconds, rate_func=linear)
    return grp, anim


def truck() -> VGroup:
    box = RoundedRectangle(width=2.4, height=1.3, corner_radius=0.08, fill_color="#64748B", fill_opacity=1,
                           stroke_width=0)
    cab = RoundedRectangle(width=0.9, height=1.0, corner_radius=0.1, fill_color="#475569", fill_opacity=1,
                           stroke_width=0).next_to(box, RIGHT, buff=0.05).align_to(box, DOWN)
    win = Rectangle(width=0.45, height=0.35, fill_color="#BFDBFE", fill_opacity=0.9, stroke_width=0)
    win.move_to(cab.get_center() + v(0.15, 0.2))
    wheels = VGroup(*[Circle(radius=0.26, fill_color="#0B1220", fill_opacity=1, stroke_color=OBJ_STROKE,
                             stroke_width=2).move_to(box.get_bottom() + v(dx, -0.05)) for dx in (-0.7, 0.6)])
    wheels.add(Circle(radius=0.26, fill_color="#0B1220", fill_opacity=1, stroke_color=OBJ_STROKE,
                      stroke_width=2).move_to(cab.get_bottom() + v(0.05, -0.05)))
    return VGroup(box, cab, win, wheels)


def mosquito() -> VGroup:
    body = Ellipse(width=0.34, height=0.1, fill_color="#E2E8F0", fill_opacity=1, stroke_width=0)
    wings = VGroup(Ellipse(width=0.22, height=0.12, fill_color="#94A3B8", fill_opacity=0.7, stroke_width=0)
                   .rotate(0.6).move_to(v(-0.02, 0.1)),
                   Ellipse(width=0.22, height=0.12, fill_color="#94A3B8", fill_opacity=0.7, stroke_width=0)
                   .rotate(-0.6).move_to(v(0.06, 0.1)))
    legs = VGroup(*[Line(v(x, -0.02), v(x + 0.06, -0.14), color="#E2E8F0", stroke_width=1.5) for x in (-0.08, 0, 0.08)])
    return VGroup(wings, body, legs)


class S11_Quiz(VoiceScene):
    def construct(self):
        self.label = chapter_label("8", "Summary and quiz")
        self.play(FadeIn(self.label), run_time=0.5)
        self.q1()
        self.q2()
        self.q3()
        self.closing()

    def pause_for(self, seconds: float, where=v(0, -3.1)):
        badge = pause_badge().move_to(where + LEFT * 0.6)
        ring, anim = countdown(seconds)
        ring.next_to(badge, RIGHT, buff=0.35)
        self.play(FadeIn(badge), FadeIn(ring), run_time=0.4)
        self.play(anim)
        self.play(FadeOut(badge), FadeOut(ring[0]), run_time=0.4)

    # ------------------------------------------------------------------ question 1
    def q1(self):
        tag = bold("Question 1", 30, ACCENT).move_to(v(0, 3.0))
        q = txt("A space probe, far from any star or planet, switches off its engines.\nWhat happens to its motion?",
                30, line_spacing=0.9).move_to(v(0, 2.1))
        rng = np.random.default_rng(3)
        stars = VGroup(*[Dot(v(rng.uniform(-7, 7), rng.uniform(-3.4, 1.2)), radius=rng.uniform(0.01, 0.03),
                             color=WHITE).set_opacity(rng.uniform(0.3, 0.9)) for _ in range(70)])
        probe = rocket(1.3, flame=False).rotate(-PI / 2).move_to(v(-4.5, -1.0))
        with self.say("Now, test yourself. Question one. A space probe, far from any star or planet, switches off "
                      "its engines. What happens to its motion? Pause the video and decide.") as c:
            self.play(FadeIn(tag), run_time=0.5)
            self.at_sentence(c, 2)
            self.play(FadeIn(q), FadeIn(stars), FadeIn(probe), run_time=0.9)
        self.pause_for(5.0)
        vel = always_redraw(lambda: VGroup(
            arrow(probe.get_right() + RIGHT * 0.2, probe.get_right() + RIGHT * 1.3, C_VEL, stroke=5),
            tex(r"\vec v = \text{constant}", size=28, color=C_VEL).next_to(probe.get_right() + RIGHT * 0.75, UP, buff=0.15)))
        ans = lines(f"{hl('Answer:', GOOD)} it keeps moving in a straight line at constant speed.",
                    "No net force means no change in velocity. It does not slow down and stop.",
                    size=28, buff=0.15, align=ORIGIN).move_to(v(0, -3.0))
        probe.add_updater(lambda m, dt: m.shift(RIGHT * 0.7 * dt))
        with self.say("Answer: it keeps moving in a straight line at constant speed, until some force acts on "
                      "it. It does not slow down and stop: with no net force, its velocity doesn't change.") as c:
            self.add(vel)
            self.play(FadeIn(ans), run_time=0.8)
        probe.clear_updaters()
        vel.clear_updaters()
        self.play(FadeOut(VGroup(tag, q, stars, probe, vel, ans)), run_time=0.6)

    # ------------------------------------------------------------------ question 2
    def q2(self):
        tag = bold("Question 2", 30, ACCENT).move_to(v(0, 3.0))
        q = txt("A truck and a mosquito collide head-on.\nWhich feels the bigger force? Which has the bigger acceleration?",
                30, line_spacing=0.9).move_to(v(0, 2.1))
        tr = truck().move_to(v(-2.2, -0.6))
        mq = mosquito().scale(1.6).move_to(v(tr.get_right()[0] + 0.3, -0.35))
        with self.say("Question two. A truck and a mosquito collide head-on. Which one feels the bigger force? "
                      "And which one has the bigger acceleration? Pause and think.") as c:
            self.play(FadeIn(tag), FadeIn(tr), FadeIn(mq), run_time=0.7)
            self.play(FadeIn(q), run_time=0.8)
        self.pause_for(5.0)
        contact = v(tr.get_right()[0] + 0.12, -0.35)
        on_truck = force(contact + DOWN * 0.3, LEFT * 1.5, C_FORCE)
        on_mq = force(contact + UP * 0.3, RIGHT * 1.5, C_FORCE)
        l1 = txt("force on truck", 22, C_FORCE).move_to(v(on_truck.get_x(), tr.get_bottom()[1] - 0.3))
        l2 = txt("force on mosquito", 22, C_FORCE).next_to(on_mq, UP, buff=0.12)
        ans = VGroup(
            mtxt(f"{hl('Forces:', GOOD)} exactly equal and opposite (third law).", 28),
            mtxt(f"{hl('Accelerations:', GOOD)} a = F/m, so the tiny mosquito's is enormous,", 28),
            mtxt("and the truck's is far too small to notice (second law).", 28),
        ).arrange(DOWN, buff=0.15).move_to(v(0, -2.75))
        with self.say("Answer: the forces are exactly equal, by the third law. But the mosquito's mass is tiny, "
                      "so, by the second law, its acceleration is enormous, while the truck's is far too small "
                      "to notice.") as c:
            self.play(GrowArrow(on_truck[0]), GrowArrow(on_mq[0]), FadeIn(l1), FadeIn(l2), FadeIn(ans[0]), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(FadeIn(ans[1:]), run_time=0.9)
        self.play(FadeOut(VGroup(tag, q, tr, mq, on_truck, on_mq, l1, l2, ans)), run_time=0.6)

    # ------------------------------------------------------------------ question 3: Atwood machine
    def q3(self):
        m1, m2 = 3.0, 5.0
        a = (m2 - m1) * G / (m1 + m2)          # 2.45 m/s^2
        T = 2 * m1 * m2 * G / (m1 + m2)        # 36.75 N
        tag = bold("Question 3", 30, ACCENT).move_to(v(0, 3.0))
        q = lines("Masses of 3 kg and 5 kg hang from a light string over a frictionless pulley",
                  "(an Atwood machine). Find the acceleration and the tension.  (g = 9.8 m/s²)",
                  size=27, align=ORIGIN, buff=0.12).move_to(v(0, 2.2))
        # machine on the left; 1 unit = 1 m for the motion
        pc = v(-4.6, 0.3)
        R = 0.7
        pl = pulley(R, 0.55).move_to(pc, aligned_edge=ORIGIN)
        pl.shift(pc - pl.wheel.get_center())
        ceiling = Line(v(-5.8, pc[1] + R + 0.55), v(-3.4, pc[1] + R + 0.55), color=GROUND, stroke_width=4)
        y0 = -1.9
        T_RUN = 0.9  # s of motion shown (rising block stays below the pulley)
        clock = ValueTracker(0.0)

        def y_left(t):
            return y0 + 0.5 * a * t * t   # 3 kg rises

        def y_right(t):
            return y0 - 0.5 * a * t * t   # 5 kg falls

        b1 = block(0.85, 0.75, "3 kg", label_size=24)
        b2 = block(1.0, 0.95, "5 kg", label_size=24, fill=OBJ_FILL_2)
        b1.add_updater(lambda m: m.move_to(v(pc[0] - R, y_left(clock.get_value()))))
        b2.add_updater(lambda m: m.move_to(v(pc[0] + R, y_right(clock.get_value()))))
        strings = always_redraw(lambda: VGroup(
            Line(v(pc[0] - R, pc[1]), b1.get_top(), color=MUTED, stroke_width=3),
            Line(v(pc[0] + R, pc[1]), b2.get_top(), color=MUTED, stroke_width=3)))
        machine = VGroup(ceiling, pl)
        with self.say("Question three, a calculation. Two masses, 3 kilograms and 5 kilograms, hang from the ends "
                      "of a light string that passes over a frictionless pulley. This is called an Atwood machine. "
                      "Find the acceleration of the masses, and the tension in the string. Use g equals 9.8. Pause "
                      "the video, and work it out with free-body diagrams.") as c:
            self.play(FadeIn(tag), run_time=0.5)
            self.at_sentence(c, 1)
            self.play(FadeIn(q), FadeIn(machine), FadeIn(b1), FadeIn(b2), FadeIn(strings), run_time=1.0)
        self.pause_for(8.0)

        # free-body diagrams, 1 unit = 25 N
        k = 1 / 25
        o1, o2 = v(-1.6, -0.2), v(0.4, -0.2)
        fb1 = VGroup(Dot(o1, radius=0.07, color=TEXT), txt("3 kg", 22, MUTED).move_to(o1 + DOWN * (m1 * G * k + 0.45)),
                     force(o1, UP * T * k, C_TENSION, "T", label_dir=RIGHT, label_size=30),
                     force(o1, DOWN * m1 * G * k, C_WEIGHT, r"29.4\ \text{N}", label_dir=RIGHT, label_size=26))
        fb2 = VGroup(Dot(o2, radius=0.07, color=TEXT), txt("5 kg", 22, MUTED).move_to(o2 + DOWN * (m2 * G * k + 0.45)),
                     force(o2, UP * T * k, C_TENSION, "T", label_dir=RIGHT, label_size=30),
                     force(o2, DOWN * m2 * G * k, C_WEIGHT, r"49\ \text{N}", label_dir=RIGHT, label_size=26))
        up1 = arrow(o1 + v(-0.55, 0.3), o1 + v(-0.55, 1.1), C_ACC, stroke=5, tip=0.16)
        dn2 = arrow(o2 + v(-0.55, -0.3), o2 + v(-0.55, -1.1), C_ACC, stroke=5, tip=0.16)
        e1 = tex(r"5\text{ kg (down +)}:\ \ 5g - T = 5a", size=32)
        e2 = tex(r"3\text{ kg (up +)}:\ \ T - 3g = 3a", size=32)
        e3 = tex(r"\text{add:}\ \ 2g = 8a \;\Rightarrow\; a = \tfrac{g}{4} = 2.45\ \text{m/s}^2", size=32)
        e4 = tex(r"T = 3\,(g + a) = 3 \times 12.25 = 36.75\ \text{N}", size=32)
        eqs = VGroup(e1, e2, e3, e4).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        if eqs.width > 5.3:
            eqs.scale_to_fit_width(5.3)
        eqs.move_to(v(4.05, -0.55))
        ab = boxed(VGroup(e3, e4), GOOD)
        chk = mtxt(f"check: 29.4 N &lt; {hl('T = 36.75 N', C_TENSION)} &lt; 49 N", 26).move_to(v(2.9, -3.3))
        with self.say("Here's the solution. The 5 kilogram mass moves down, and the 3 kilogram mass moves up, with "
                      "accelerations of the same size.") as c:
            self.play(FadeOut(q), run_time=0.4)
            self.play(clock.animate.set_value(T_RUN), run_time=T_RUN / 0.5, rate_func=linear)
            self.play(FadeIn(fb1[:2]), FadeIn(fb2[:2]), GrowArrow(up1), GrowArrow(dn2), run_time=0.8)
        with self.say("For the 5 kilogram mass, taking down as positive: 5 g, minus T, equals 5 times the "
                      "acceleration. For the 3 kilogram mass, taking up as positive: T, minus 3 g, equals 3 times "
                      "the acceleration.") as c:
            self.play(GrowArrow(fb2[3][0]), FadeIn(fb2[3][1]), GrowArrow(fb2[2][0]), FadeIn(fb2[2][1]), run_time=0.8)
            self.play(Write(e1), run_time=1.2)
            self.at_sentence(c, 1)
            self.play(GrowArrow(fb1[3][0]), FadeIn(fb1[3][1]), GrowArrow(fb1[2][0]), FadeIn(fb1[2][1]), run_time=0.8)
            self.play(Write(e2), run_time=1.2)
        with self.say("Add the two equations, and the tension cancels: 2 g equals 8 times the acceleration. So the "
                      "acceleration is g over 4, which is 2.45 metres per second squared. Then T equals 3 times, g "
                      "plus the acceleration, which is 3 times 12.25: "
                      "36.75 newtons.") as c:
            self.play(Write(e3), run_time=1.6)
            self.at_sentence(c, 2)
            self.play(Write(e4), run_time=1.6)
            self.play(Create(ab), run_time=0.6)
        with self.say("Check: the tension lies between the two weights, 29.4 and 49 newtons, just as it must: less "
                      "than the weight of the falling mass, and more than the weight of the rising one.") as c:
            self.play(FadeIn(chk), run_time=0.8)
            clock.set_value(0)
            self.play(clock.animate.set_value(T_RUN), run_time=T_RUN / 0.5, rate_func=linear)
        for m_ in (b1, b2, strings):
            m_.clear_updaters()
        self.play(FadeOut(VGroup(tag, machine, b1, b2, strings, fb1, fb2, up1, dn2, eqs, ab, chk)), run_time=0.6)

    # ------------------------------------------------------------------ closing
    def closing(self):
        head = bold("Newton's laws of motion", 48).move_to(v(0, 2.3))
        l1 = mtxt(f"{hl('1')}  Motion needs no cause; only {hl('changes')} of motion do.", 32)
        l2 = mtxt(f"{hl('2')}  A net force changes momentum:  F = dp/dt = ma.", 32)
        l3 = mtxt(f"{hl('3')}  Forces are {hl('interactions')}: they come in equal and opposite pairs.", 32)
        grp = VGroup(l1, l2, l3).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(v(0, 0.4))
        if grp.width > 13.0:
            grp.scale_to_fit_width(13.0)
        tip = txt("Master free-body diagrams, and you can solve a huge range of mechanics problems.", 26, MUTED)
        tip.move_to(v(0, -1.8))
        thanks = bold("Thanks for watching!", 40, ACCENT).move_to(v(0, -2.9))
        with self.say("That's Newton's three laws of motion. The first tells us that motion itself needs no cause: "
                      "only changes of motion do. The second tells us how a net force changes momentum. And the third "
                      "tells us that forces always come from interactions, in pairs. Master free-body diagrams, and you "
                      "can solve a huge range of problems in mechanics. Thanks for watching!") as c:
            self.play(FadeIn(head), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(l1, shift=RIGHT * 0.2), run_time=0.7)
            self.at_sentence(c, 2)
            self.play(FadeIn(l2, shift=RIGHT * 0.2), run_time=0.7)
            self.at_sentence(c, 3)
            self.play(FadeIn(l3, shift=RIGHT * 0.2), run_time=0.7)
            self.at_sentence(c, 4)
            self.play(FadeIn(tip), run_time=0.7)
            self.at_sentence(c, 5)
            self.play(FadeIn(thanks, scale=1.1), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(VGroup(head, grp, tip, thanks)), FadeOut(self.label), run_time=1.0)
        self.wait(0.5)
