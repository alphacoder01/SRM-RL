"""Part 5 -- Newton's third law: pairs, skaters, walking/rockets, gravity, book-on-table, horse & cart."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import (apple, arrow, block, book, boxed, cart, check_mark, cross_mark,
                            earth_cap, force, ground, horse, panel, pause_badge, person, rocket,
                            shown, table, v)
from lesson.style import *


def pair_arrow(tail, vec, color, text, label_dir, size=24, buff=0.12):
    """Force arrow with a small word label."""
    return force(tail, vec, color, txt(text, size, color), label_dir=label_dir, buff=buff)


class S05_ThirdLaw(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "5", "Newton's third law", "forces come in pairs")
        self.statement()
        self.wall()
        self.facts()
        self.skaters()
        self.walking_rocket()
        self.apple_earth()
        self.book_trap()
        self.horse_cart()
        self.recap()

    # ------------------------------------------------------------------ statement
    def statement(self):
        box = panel(12.8, 2.1).move_to(v(0, 1.55))
        tag = bold("Newton's third law", 28, ACCENT).next_to(box, UP, buff=0.15).align_to(box, LEFT)
        law = lines(f"When one body exerts a force on a second body, the second body {hl('simultaneously')}",
                    f"exerts a force on the first that is {hl('equal in magnitude')} and {hl('opposite in direction')}.",
                    size=30, align=ORIGIN, buff=0.2).move_to(box)
        if law.width > 12.3:
            law.scale_to_fit_width(12.3)
        eq = tex(r"\vec F_{12} = -\,\vec F_{21}", size=60).move_to(v(-3.3, -1.0))
        legend = VGroup(tex(r"\vec F_{12}", size=34), txt(": force on body 1 due to body 2", 26, MUTED),
                        ).arrange(RIGHT, buff=0.1).next_to(eq, DOWN, buff=0.35)
        b1 = block(1.3, 1.0, "1").move_to(v(2.55, -1.3))
        b2 = block(1.3, 1.0, "2", fill=OBJ_FILL_2).next_to(b1, RIGHT, buff=0)
        f12 = force(b1.get_center(), LEFT * 1.6, C_FORCE, r"\vec F_{12}", label_dir=UP)
        f21 = force(b2.get_center(), RIGHT * 1.6, C_FORCE, r"\vec F_{21}", label_dir=UP)
        with self.say("Newton's third law says: whenever one body exerts a force on a second body, the "
                      "second body simultaneously exerts a force on the first that is equal in magnitude "
                      "and opposite in direction. You may have heard it as: to every action, there is an "
                      "equal and opposite reaction.") as c:
            self.play(FadeIn(box), FadeIn(tag), run_time=0.6)
            self.play(Write(law), run_time=min(5.0, self.remaining(c, c.end_of(0))))
            self.play(Write(eq), FadeIn(legend), run_time=1.2)
            self.play(FadeIn(b1), FadeIn(b2), run_time=0.5)
            self.play(GrowArrow(f12[0]), GrowArrow(f21[0]), FadeIn(f12[1]), FadeIn(f21[1]), run_time=0.9)
        self.play(FadeOut(VGroup(box, tag, law, eq, legend, b1, b2, f12, f21)), run_time=0.6)

    # ------------------------------------------------------------------ pushing a wall
    def wall(self):
        floor_y = -2.3
        flr = ground(-7.0, 7.0, floor_y)
        wl = Rectangle(width=0.7, height=4.4, fill_color="#57534E", fill_opacity=1, stroke_color="#A8A29E",
                       stroke_width=2).move_to(v(1.05, floor_y + 2.2))
        guy = person(2.0, arms="push_right").move_to(v(0, floor_y), aligned_edge=DOWN)
        guy.shift(RIGHT * (wl.get_left()[0] - (guy.get_right()[0])))
        hand = v(wl.get_left()[0], floor_y + 0.68 * 2.0)
        on_wall = pair_arrow(hand + UP * 0.25, RIGHT * 1.5, C_FORCE, "your hand pushes the wall", RIGHT)
        on_hand = pair_arrow(hand + DOWN * 0.25, LEFT * 1.5, C_NORMAL, "the wall pushes your hand", LEFT)
        head = bold("Forces are interactions", 38).move_to(v(0, 2.9))
        d1 = Circle(radius=0.55, fill_color=OBJ_FILL, fill_opacity=1, stroke_color=OBJ_STROKE).move_to(v(-2.2, 0.4))
        d2 = Circle(radius=0.55, fill_color=OBJ_FILL_2, fill_opacity=1, stroke_color=OBJ_STROKE).move_to(v(2.2, 0.4))
        n1, n2 = txt("1", 30).move_to(d1), txt("2", 30).move_to(d2)
        a12 = force(d1.get_right(), RIGHT * 1.2, C_FORCE, r"\vec F_{12}", label_dir=UP)
        a21 = force(d2.get_left(), LEFT * 1.2, C_FORCE, r"\vec F_{21}", label_dir=UP)
        one_pair = txt("one interaction  =  one pair of forces", 28, MUTED).move_to(v(0, -1.2))
        pic = VGroup(d1, d2, n1, n2, a12, a21, one_pair)
        with self.say("So a force is never a one-sided thing. It is always an interaction between two "
                      "bodies, and it always comes as a pair. Push on a wall, and the wall pushes back on "
                      "your hand, just as hard.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(VGroup(d1, d2, n1, n2)), run_time=0.5)
            self.play(GrowArrow(a12[0]), GrowArrow(a21[0]), FadeIn(a12[1]), FadeIn(a21[1]), run_time=0.8)
            self.play(FadeIn(one_pair), run_time=0.5)
            self.at_sentence(c, 2)
            self.play(FadeOut(pic), run_time=0.4)
            self.play(FadeIn(flr), FadeIn(wl), FadeIn(guy), run_time=0.6)
            self.play(GrowArrow(on_wall[0]), FadeIn(on_wall[1]), run_time=0.8)
            self.play(GrowArrow(on_hand[0]), FadeIn(on_hand[1]), run_time=0.8)
        self.play(FadeOut(VGroup(flr, wl, guy, on_wall, on_hand, head)), run_time=0.6)

    # ------------------------------------------------------------------ four facts
    def facts(self):
        head = bold("Four facts about every action–reaction pair", 36, ACCENT).move_to(v(0, 2.8))
        items = [
            f"{hl('1')}  They act on {hl('different bodies')}.",
            f"{hl('2')}  They are always {hl('equal in size')} and {hl('opposite in direction')},",
            "     whether the bodies are at rest, moving, or accelerating.",
            f"{hl('3')}  They act at the {hl('same instant')}: neither causes the other.",
            f"{hl('4')}  They are the {hl('same kind')} of force (both gravitational, both contact, ...).",
        ]
        rows = VGroup(*[mtxt(i, 30) for i in items]).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rows.next_to(head, DOWN, buff=0.55)
        rows[2].shift(UP * 0.1).align_to(rows[1][1], LEFT)
        if rows.width > 13.0:
            rows.scale_to_fit_width(13.0)
        never = panel(12.0, 1.1, fill="#3B2F0B", stroke=ACCENT).move_to(v(0, -2.8))
        never_t = mtxt(f"So an action–reaction pair can {hl('never cancel')}: "
                       f"forces cancel only when they act on the {hl('same')} body.", 29).move_to(never)
        if never_t.width > 11.6:
            never_t.scale_to_fit_width(11.6)
        with self.say("Four facts about these pairs are worth remembering. One: the two forces act on "
                      "different bodies. Two: they are always equal in size and opposite in direction, "
                      "whatever the bodies are doing: at rest, moving, or accelerating. Three: they act at "
                      "the same instant. Neither one causes the other, so the words action and reaction "
                      "are only labels. Four: they are the same kind of force. If one is gravitational, so "
                      "is the other. If one is a contact force, so is the other.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(rows[0], shift=RIGHT * 0.2), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(FadeIn(rows[1], shift=RIGHT * 0.2), FadeIn(rows[2], shift=RIGHT * 0.2), run_time=0.7)
            self.at_sentence(c, 3)
            self.play(FadeIn(rows[3], shift=RIGHT * 0.2), run_time=0.6)
            self.at_sentence(c, 5)
            self.play(FadeIn(rows[4], shift=RIGHT * 0.2), run_time=0.6)
        with self.say("Because the two forces act on different bodies, they can never cancel each other. "
                      "Forces cancel only when they act on the same body."):
            self.play(Indicate(rows[0], color=ACCENT, scale_factor=1.04), run_time=1.0)
            self.play(FadeIn(never), FadeIn(never_t), run_time=0.8)
        self.play(FadeOut(VGroup(head, rows, never, never_t)), run_time=0.6)

    # ------------------------------------------------------------------ skaters
    def skaters(self):
        ice_y, h = -1.9, 1.9
        ice = VGroup(Rectangle(width=14.4, height=0.35, fill_color="#7DD3FC", fill_opacity=0.25,
                               stroke_width=0).move_to(v(0, ice_y - 0.175)),
                     Line(v(-7.2, ice_y), v(7.2, ice_y), color="#7DD3FC", stroke_width=3))
        F, mA, mB, t_push = 750.0, 50.0, 75.0, 0.2
        aA, aB = F / mA, F / mB
        xA0, xB0 = -0.45, 0.45
        vA, vB = aA * t_push, aB * t_push
        clock = ValueTracker(0.0)

        def xa(t):
            if t <= t_push:
                return xA0 - 0.5 * aA * t * t
            return xA0 - 0.5 * aA * t_push**2 - vA * (t - t_push)

        def xb(t):
            if t <= t_push:
                return xB0 + 0.5 * aB * t * t
            return xB0 + 0.5 * aB * t_push**2 + vB * (t - t_push)

        def figures():
            t = clock.get_value()
            a_x, b_x = xa(t), xb(t)
            A = person(h, arms="none").move_to(v(a_x, ice_y), aligned_edge=DOWN)
            B = person(h, arms="none").move_to(v(b_x, ice_y), aligned_edge=DOWN)
            sa, sb = v(a_x, ice_y + 0.70 * h), v(b_x, ice_y + 0.70 * h)
            if t <= t_push:
                mid = (sa + sb) / 2
                arms = VGroup(Line(sa, mid + UP * 0.03), Line(sa, mid + DOWN * 0.05),
                              Line(sb, mid + UP * 0.03), Line(sb, mid + DOWN * 0.05))
            else:
                arms = VGroup(*[Line(s, s + v(dx, -0.26 * h)) for s in (sa, sb) for dx in (-0.17 * h, 0.17 * h)])
            arms.set_stroke(TEXT, 6)
            names = VGroup(txt("Anna, 50 kg", 24, TEXT).next_to(A, LEFT, buff=0.35).shift(DOWN * 0.62),
                           txt("Ben, 75 kg", 24, TEXT).next_to(B, RIGHT, buff=0.35).shift(DOWN * 0.62))
            return VGroup(A, B, arms, names)

        figs = always_redraw(figures)
        scale_f = 1.5 / F

        def push_arrows():
            t = clock.get_value()
            cA = v(xa(t), ice_y + 0.6 * h)
            cB = v(xb(t), ice_y + 0.6 * h)
            onA = force(cA + DOWN * 0.35, LEFT * F * scale_f, C_FORCE)
            onB = force(cB + DOWN * 0.35, RIGHT * F * scale_f, C_FORCE)
            return shown(VGroup(onA, onB), t <= t_push)

        pushes = always_redraw(push_arrows)
        labA = txt("force on Anna by Ben: 750 N", 24, C_FORCE).move_to(v(-4.1, 1.75))
        labB = txt("force on Ben by Anna: 750 N", 24, C_FORCE).move_to(v(4.1, 1.75))

        def vel_arrows():
            t = clock.get_value()
            k = 0.35
            top = ice_y + h + 0.35
            return shown(VGroup(arrow(v(xa(t), top), v(xa(t) - k * vA, top), C_VEL, stroke=5, tip=0.18),
                                arrow(v(xb(t), top), v(xb(t) + k * vB, top), C_VEL, stroke=5, tip=0.18)),
                         t > t_push)

        vels = always_redraw(vel_arrows)
        acc = VGroup(tex(r"a_{\text{Anna}} = \frac{750\ \text{N}}{50\ \text{kg}} = 15\ \text{m/s}^2", size=34, color=C_ACC),
                     tex(r"a_{\text{Ben}} = \frac{750\ \text{N}}{75\ \text{kg}} = 10\ \text{m/s}^2", size=34, color=C_ACC)
                     ).arrange(RIGHT, buff=1.2).move_to(v(0, 2.85))
        speeds = txt("after the 0.2 s push:  Anna 3 m/s left,  Ben 2 m/s right", 26, C_VEL).move_to(v(0, 2.1))
        tag = txt("push shown at 0.1× speed", 22, MUTED).to_corner(DR, buff=0.3)
        tag2 = txt("glide shown at real speed", 22, MUTED).to_corner(DR, buff=0.3)
        with self.say("Watch two skaters, standing still on smooth ice. Anna, with a mass of 50 kilograms, "
                      "pushes Ben, who has a mass of 75 kilograms. At every instant during the push, the "
                      "force on Ben and the force on Anna are equal and opposite, so both of them start to "
                      "move, in opposite directions.") as c:
            self.play(FadeIn(ice), FadeIn(figs), run_time=0.8)
            self.at_sentence(c, 2)
            self.add(pushes)
            self.play(FadeIn(labA), FadeIn(labB), FadeIn(tag), run_time=0.6)
            self.play(clock.animate.set_value(t_push), run_time=t_push / 0.1, rate_func=linear)
            self.add(vels)
            self.play(FadeOut(tag), FadeIn(tag2), run_time=0.3)
            self.play(clock.animate.set_value(t_push + 1.2), run_time=1.2, rate_func=linear)
        with self.say("But equal forces don't mean equal effects. With a 750 newton push, Anna accelerates at "
                      "15 metres per second squared, while Ben, who is heavier, accelerates at only 10. So "
                      "after the push, which lasts a fifth of a second, Anna glides away at 3 metres per "
                      "second, and Ben at 2.") as c:
            self.play(FadeOut(labA), FadeOut(labB), FadeOut(tag2), run_time=0.4)
            clock.set_value(0)
            self.wait(0.3)
            self.play(FadeIn(acc), run_time=0.8)
            self.play(clock.animate.set_value(t_push), run_time=t_push / 0.1, rate_func=linear)
            self.play(clock.animate.set_value(t_push + 1.2), run_time=1.2, rate_func=linear)
            self.at_sentence(c, 2)
            self.play(FadeIn(speeds), run_time=0.8)
        for m in (figs, pushes, vels):
            m.clear_updaters()
        self.play(FadeOut(VGroup(ice, figs, pushes, vels, acc, speeds)), run_time=0.6)

    # ------------------------------------------------------------------ walking and rockets
    def walking_rocket(self):
        pnl = VGroup(panel(6.6, 5.4), panel(6.6, 5.4)).arrange(RIGHT, buff=0.35).move_to(v(0, -0.45))
        t1 = bold("Walking", 30).next_to(pnl[0].get_top(), DOWN, buff=0.25)
        t2 = bold("Rocket", 30).next_to(pnl[1].get_top(), DOWN, buff=0.25)
        gy = pnl[0].get_bottom()[1] + 1.1
        gnd = ground(pnl[0].get_left()[0] + 0.3, pnl[0].get_right()[0] - 0.3, gy)
        walker = person(2.0, arms="stride").move_to(v(pnl[0].get_x() + 0.3, gy), aligned_edge=DOWN)
        back_foot = v(walker.get_left()[0] + 0.02, gy)
        on_ground = force(back_foot + DOWN * 0.32, LEFT * 1.3, C_FRICTION)
        on_foot = force(back_foot + UP * 0.12, RIGHT * 1.3, C_FRICTION)
        lg = txt("you push the ground back", 22, C_FRICTION).next_to(on_ground, DOWN, buff=0.12)
        lf = txt("the ground pushes you forward", 22, C_FRICTION).move_to(v(pnl[0].get_x(), gy + 2.9))
        lf_line = DashedLine(lf.get_bottom() + DOWN * 0.05, on_foot[0].get_center() + UP * 0.1,
                             color=C_FRICTION, stroke_width=1.5, dash_length=0.06)
        on_ground = VGroup(on_ground[0], lg)
        on_foot = VGroup(on_foot[0], VGroup(lf, lf_line))
        rk = rocket(2.0).move_to(pnl[1].get_center() + v(-1.75, 0.15))
        gas = VGroup(*[Circle(radius=r, fill_color="#94A3B8", fill_opacity=0.35, stroke_width=0)
                       .move_to(rk.get_bottom() + v(dx, dy))
                       for r, dx, dy in ((0.28, 0, -0.3), (0.22, -0.25, -0.75), (0.25, 0.22, -0.8),
                                         (0.18, 0, -1.2))])
        on_rocket = force(rk.get_center() + v(0.75, -0.2), UP * 1.3, C_FORCE)
        on_gas = force(rk.get_bottom() + v(0.75, 0.3), DOWN * 1.3, C_FORCE)
        on_rocket = VGroup(on_rocket[0], txt("gases push the rocket up", 22, C_FORCE).next_to(on_rocket[0], RIGHT, buff=0.15))
        on_gas = VGroup(on_gas[0], txt("rocket pushes the gases down", 22, C_FORCE).next_to(on_gas[0], RIGHT, buff=0.15))
        note = txt("no air needed: it works in the vacuum of space", 22, MUTED).next_to(t2, DOWN, buff=0.2)
        with self.say("The third law explains how people, animals and vehicles set themselves in motion. When you "
                      "walk, your foot pushes "
                      "backward on the ground, and the ground pushes forward on you. That forward push, a "
                      "friction force, is what accelerates you, which is why walking on ice is so hard.") as c:
            self.play(FadeIn(pnl[0]), FadeIn(t1), FadeIn(gnd), FadeIn(walker), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(GrowArrow(on_ground[0]), FadeIn(on_ground[1]), run_time=0.8)
            self.play(GrowArrow(on_foot[0]), FadeIn(on_foot[1]), run_time=0.8)
        with self.say("A rocket works the same way. It pushes its exhaust gases backward at high speed, and "
                      "the gases push the rocket forward. A rocket doesn't need air to push against, which "
                      "is why it works in the vacuum of space.") as c:
            self.play(FadeIn(pnl[1]), FadeIn(t2), FadeIn(rk), FadeIn(gas), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(GrowArrow(on_gas[0]), FadeIn(on_gas[1]), run_time=0.8)
            self.play(GrowArrow(on_rocket[0]), FadeIn(on_rocket[1]), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(note), run_time=0.6)
        self.play(FadeOut(VGroup(pnl, t1, t2, gnd, walker, on_ground, on_foot, rk, gas, on_rocket, on_gas, note)),
                  run_time=0.6)

    # ------------------------------------------------------------------ apple and Earth
    def apple_earth(self):
        earth = earth_cap(center_x=-4.4, top_y=-2.4, radius=9.0)
        elab = txt("Earth", 30, "#93C5FD").move_to(v(-4.4, -3.3))
        ap = apple(0.24).move_to(v(-4.4, 1.2))
        on_apple = force(ap.get_center() + DOWN * 0.3, DOWN * 1.4, C_WEIGHT)
        on_earth = force(v(-4.4, -2.4), UP * 1.4, C_WEIGHT)
        la = txt("Earth pulls apple: 0.98 N", 24, C_WEIGHT).next_to(on_apple, RIGHT, buff=0.2)
        le = txt("apple pulls Earth: 0.98 N", 24, C_WEIGHT).next_to(on_earth, RIGHT, buff=0.2)
        calc = VGroup(
            tex(r"a_{\text{apple}} = \frac{0.98\ \text{N}}{0.1\ \text{kg}} = 9.8\ \text{m/s}^2", size=36),
            tex(r"a_{\text{Earth}} = \frac{0.98\ \text{N}}{6.0\times 10^{24}\ \text{kg}}"
                r"\approx 1.6\times 10^{-25}\ \text{m/s}^2", size=36),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        calc.scale_to_fit_width(min(calc.width, 6.4)).move_to(v(3.55, 1.9))
        tiny = txt("far too small to notice", 26, MUTED).next_to(calc, DOWN, buff=0.35)
        with self.say("Even gravity obeys the third law. The Earth pulls a 0.1 kilogram apple down with a "
                      "force of about 0.98 newtons, and the apple pulls the Earth up with exactly the same "
                      "force. The apple accelerates at 9.8 metres per second squared. But the Earth, with its "
                      "enormous mass, accelerates by only about 1.6 times ten to the minus twenty-five metres "
                      "per second squared: far too small ever to notice.") as c:
            self.play(FadeIn(earth), FadeIn(elab), FadeIn(ap), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(GrowArrow(on_apple[0]), FadeIn(la), run_time=0.8)
            self.play(GrowArrow(on_earth[0]), FadeIn(le), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(Write(calc[0]), run_time=1.2)
            self.at_sentence(c, 3)
            self.play(Write(calc[1]), run_time=1.4)
            self.play(FadeIn(tiny), run_time=0.6)
        self.play(FadeOut(VGroup(earth, elab, ap, on_apple, on_earth, la, le, calc, tiny)), run_time=0.6)

    # ------------------------------------------------------------------ the book-on-table trap
    def book_trap(self):
        top_y = -0.2
        tb = table(4.2, top_y=top_y, x=-3.2, leg_h=1.7)
        bk = book(1.6, 0.42).move_to(v(-3.2, top_y + 0.21))
        c0 = bk.get_center()
        W = force(c0 + LEFT * 0.25, DOWN * 1.3, C_WEIGHT, r"\vec W", label_dir=LEFT)
        N = force(c0 + RIGHT * 0.25, UP * 1.3, C_NORMAL, r"\vec N", label_dir=RIGHT)
        q = mtxt(f"Equal and opposite... so are {hl('W')} and {hl('N')} an action–reaction pair?", 30).move_to(v(0, 3.0))
        badge = pause_badge().move_to(v(3.4, 1.0))
        with self.say("Now, a classic trap. A book rests on a table. Two forces act on the book: its weight, "
                      "pulling down, and the normal force from the table, pushing up. They're equal and "
                      "opposite. So, are they an action–reaction pair? Pause and think.") as c:
            self.play(FadeIn(tb), FadeIn(bk), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(GrowArrow(W[0]), FadeIn(W[1]), run_time=0.7)
            self.play(GrowArrow(N[0]), FadeIn(N[1]), run_time=0.7)
            self.at_sentence(c, 4)
            self.play(FadeIn(q), run_time=0.8)
            self.at_sentence(c, 5)
            self.play(FadeIn(badge), run_time=0.5)
        self.wait(3.0)
        self.play(FadeOut(badge), run_time=0.4)

        no = VGroup(cross_mark(0.45), bold("No!", 40, BAD)).arrange(RIGHT, buff=0.25).move_to(v(0, 1.7)).align_to(v(0.2, 0), LEFT)
        reasons = lines("•  both act on the same body: the book",
                        "•  different kinds: gravitational vs contact",
                        "•  equal here only because the book isn't",
                        "   accelerating (first law, not third law)", size=28, buff=0.2)
        if reasons.width > 6.6:
            reasons.scale_to_fit_width(6.6)
        reasons.next_to(no, DOWN, buff=0.4).align_to(no, LEFT)
        with self.say("No, they are not. Both forces act on the same body, the book, and they are different "
                      "kinds of force: one is gravitational, the other is a contact force. They're equal "
                      "here only because the book isn't accelerating. That's the first law at work, not the "
                      "third.") as c:
            self.play(FadeIn(no), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(reasons[0]), run_time=0.6)
            self.play(FadeIn(reasons[1]), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(FadeIn(reasons[2:]), run_time=0.6)
        self.play(FadeOut(VGroup(no, reasons, q)), run_time=0.5)

        earth = earth_cap(center_x=-3.2, top_y=-2.6, radius=12.0)
        elab = txt("Earth", 26, "#93C5FD").move_to(v(-6.2, -3.4))
        on_earth = force(v(-2.95, -2.6), UP * 1.0, C_WEIGHT)
        on_table = force(c0 + v(0.25, -0.21), DOWN * 1.0, C_NORMAL)
        tab_top = tb.top
        key = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=C_WEIGHT, stroke_width=6),
                   mtxt(f"{hl('gravitational pair', C_WEIGHT)}: Earth pulls book  ·  book pulls Earth", 24)),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=C_NORMAL, stroke_width=6),
                   mtxt(f"{hl('contact pair', C_NORMAL)}: table pushes book  ·  book pushes table", 24)),
        )
        for k in key:
            k.arrange(RIGHT, buff=0.2)
        key.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(v(0.4, 2.6))
        if key.width > 13.0:
            key.scale_to_fit_width(13.0)
        pair_lab = VGroup(txt("on Earth", 22, C_WEIGHT).next_to(on_earth, RIGHT, buff=0.1),
                          txt("on table", 22, C_NORMAL).next_to(on_table[0].get_end(), RIGHT, buff=0.15))
        with self.say("The true partner of the book's weight is the book's gravitational pull on the Earth, "
                      "acting on the Earth, and pointing up. And the partner of the normal force is the push "
                      "of the book down on the table. Here are the two genuine pairs.") as c:
            self.play(FadeIn(earth), FadeIn(elab), run_time=0.6)
            self.play(GrowArrow(on_earth[0]), FadeIn(pair_lab[0]), FadeIn(key[0]), run_time=0.9)
            self.at_sentence(c, 1)
            self.play(GrowArrow(on_table[0]), FadeIn(pair_lab[1]), FadeIn(key[1]), run_time=0.9)
            self.at_sentence(c, 2)
            self.play(Indicate(VGroup(W, on_earth), color=C_WEIGHT, scale_factor=1.05),
                      Indicate(VGroup(N, on_table), color=C_NORMAL, scale_factor=1.05), run_time=1.2)
        self.play(FadeOut(VGroup(tb, bk, W, N, earth, elab, on_earth, on_table, key, pair_lab)), run_time=0.6)

    # ------------------------------------------------------------------ horse and cart
    def horse_cart(self):
        gy = -1.6
        gnd = ground(-7.0, 7.0, gy)
        hs = horse(scale=1.0).move_to(v(1.4, gy), aligned_edge=DOWN)
        ct = cart(2.4, 0.9, None, wheel_r=0.3).move_to(v(-2.9, gy), aligned_edge=DOWN)
        shaft = Line(ct.body.get_right() + UP * 0.1, hs.get_left() + v(0.6, 1.25), color="#A16207", stroke_width=6)
        q = mtxt(f"The cart pulls back on the horse just as hard... {hl('so how do they move?')}", 30).move_to(v(0, 3.0))
        badge = pause_badge().move_to(v(0, 2.0))
        with self.say("Here's one more puzzle. A horse pulls a cart. By the third law, the cart pulls back on "
                      "the horse just as hard. So how can they ever start moving? Pause the video and think "
                      "about it.") as c:
            self.play(FadeIn(gnd), FadeIn(ct), FadeIn(shaft), FadeIn(hs), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(q), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeIn(badge), run_time=0.5)
        self.wait(4.0)
        self.play(FadeOut(badge), FadeOut(q), run_time=0.4)

        T, f_cart, P = 1.6, 1.0, 2.3  # arrow lengths: pull, friction on cart, ground's push on horse
        yc = ct.body.get_center()[1] + 0.05
        on_cart_T = force(ct.body.get_right() + v(-0.2, 0.0), RIGHT * T, C_TENSION)
        on_cart_f = force(v(ct.get_center()[0], gy + 0.05), LEFT * f_cart, C_FRICTION)
        on_horse_T = force(v(hs.get_left()[0] + 0.65, yc + 0.55), LEFT * T, C_TENSION)
        hoof = v(hs.get_x() + 0.5, gy)
        on_horse_P = force(hoof + UP * 0.05, RIGHT * P, C_FRICTION)
        cart_box = VGroup(bold("Forces on the cart", 26, TEXT),
                          mtxt(f"{hl('pull of horse', C_TENSION)} forward  &gt;  {hl('friction', C_FRICTION)} back", 22),
                          mtxt("so the cart accelerates forward", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        horse_box = VGroup(bold("Forces on the horse", 26, TEXT),
                           mtxt(f"{hl('ground pushes hooves', C_FRICTION)} forward  &gt;  {hl('pull of cart', C_TENSION)} back", 22),
                           mtxt("so the horse accelerates forward", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        cart_box.move_to(v(0, 2.3)).to_edge(LEFT, buff=0.4)
        horse_box.move_to(v(0, 2.3)).to_edge(RIGHT, buff=0.4)
        pair_note = mtxt(f"The two {hl('yellow', C_TENSION)} forces are a third-law pair: "
                         "they act on different bodies, so they never cancel.", 26).move_to(v(0, -3.1))
        if pair_note.width > 13.2:
            pair_note.scale_to_fit_width(13.2)
        with self.say("The key is to look at the forces on each body separately. On the cart, there's the "
                      "horse's forward pull, and friction from the ground holding it back. If the pull is "
                      "bigger, the cart accelerates forward. On the horse, there's the cart's backward pull, "
                      "and the forward push of the ground on its hooves, because the horse pushes backward "
                      "on the ground. If the ground's push is bigger, the horse accelerates forward too. The "
                      "pair of forces between horse and cart acts on two different bodies, so it never "
                      "cancels.") as c:
            self.at_sentence(c, 1)
            self.play(FadeIn(cart_box[0]), GrowArrow(on_cart_T[0]), GrowArrow(on_cart_f[0]), run_time=0.9)
            self.play(FadeIn(cart_box[1]), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(FadeIn(cart_box[2]), run_time=0.6)
            self.at_sentence(c, 3)
            self.play(FadeIn(horse_box[0]), GrowArrow(on_horse_T[0]), GrowArrow(on_horse_P[0]), run_time=0.9)
            self.play(FadeIn(horse_box[1]), run_time=0.6)
            self.at_sentence(c, 4)
            self.play(FadeIn(horse_box[2]), run_time=0.6)
            self.at_sentence(c, 5)
            self.play(FadeIn(pair_note), Indicate(VGroup(on_cart_T, on_horse_T), color=C_TENSION), run_time=1.2)
        self.play(FadeOut(VGroup(gnd, hs, ct, shaft, on_cart_T, on_cart_f, on_horse_T, on_horse_P,
                                 cart_box, horse_box, pair_note)), run_time=0.6)

    # ------------------------------------------------------------------ recap
    def recap(self):
        box = panel(11.8, 3.6).move_to(v(0, 0.2))
        ttl = bold("Third law in one line", 32, ACCENT).next_to(box.get_top(), DOWN, buff=0.35)
        eq = tex(r"\vec F_{12} = -\,\vec F_{21}", size=60).next_to(ttl, DOWN, buff=0.35)
        sub = txt("equal · opposite · simultaneous · same kind · on different bodies", 28, MUTED)
        sub.next_to(eq, DOWN, buff=0.35)
        with self.say("So, the third law in one line: the force on body one due to body two is equal and "
                      "opposite to the force on body two due to body one. The two forces act at the same "
                      "instant, are of the same kind, and act on different bodies."):
            self.play(FadeIn(box), FadeIn(ttl), run_time=0.6)
            self.play(Write(eq), run_time=1.2)
            self.play(FadeIn(sub), run_time=0.6)
        self.play(FadeOut(VGroup(box, ttl, eq, sub)), FadeOut(self.label), run_time=0.7)
