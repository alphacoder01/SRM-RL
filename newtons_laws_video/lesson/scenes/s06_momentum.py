"""Part 6 -- Conservation of momentum, derived from the second and third laws."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, block, boxed, force, panel, person, rocket, v
from lesson.style import *


def gun(scale: float = 1.0) -> VGroup:
    s = scale
    barrel = Rectangle(width=2.2 * s, height=0.28 * s, fill_color="#64748B", fill_opacity=1, stroke_width=0)
    body = Rectangle(width=1.0 * s, height=0.42 * s, fill_color="#475569", fill_opacity=1, stroke_width=0)
    body.next_to(barrel, DOWN, buff=0).align_to(barrel, LEFT)
    grip = Polygon(v(0, 0), v(0.38 * s, 0), v(0.18 * s, -0.75 * s), v(-0.2 * s, -0.75 * s),
                   fill_color="#334155", fill_opacity=1, stroke_width=0)
    grip.next_to(body, DOWN, buff=0).align_to(body, LEFT).shift(RIGHT * 0.12 * s)
    return VGroup(barrel, body, grip)


class S06_Momentum(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "6", "Conservation of momentum")
        self.derive()
        self.skaters_check()
        self.recoil()

    # ------------------------------------------------------------------ derivation
    def derive(self):
        l1 = VGroup(txt("third law:", 28, MUTED),
                    tex(r"\vec F_{12} = -\,\vec F_{21}", size=46)).arrange(RIGHT, buff=0.4)
        l2 = VGroup(txt("second law:", 28, MUTED),
                    tex(r"\frac{d\vec p_1}{dt} = \vec F_{12}", r"\qquad", r"\frac{d\vec p_2}{dt} = \vec F_{21}",
                        size=46)).arrange(RIGHT, buff=0.4)
        l3 = VGroup(txt("add:", 28, MUTED),
                    tex(r"\frac{d}{dt}\left(\vec p_1 + \vec p_2\right) = \vec F_{12} + \vec F_{21} = \vec 0",
                        size=46)).arrange(RIGHT, buff=0.4)
        rows = VGroup(l1, l2, l3).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(v(0, 1.0))
        for r in rows:
            r[0].set_x(rows.get_left()[0] + r[0].width / 2)
        result = tex(r"\vec p_1 + \vec p_2 = \text{constant}", size=60, color=C_MOM).move_to(v(0, -2.0))
        bx = boxed(result, C_MOM)
        header = mtxt(f"{hl('second law')}  +  {hl('third law')}  ⇒  ?", 40).move_to(v(0, 2.5))
        sys_box = DashedVMobject(RoundedRectangle(width=6.4, height=2.6, corner_radius=0.3), num_dashes=70)
        sys_box.set_stroke(MUTED, 2).move_to(v(0, 0.1))
        b1 = block(1.2, 0.9, "1").move_to(v(-0.62, 0.1))
        b2 = block(1.2, 0.9, "2", fill=OBJ_FILL_2).move_to(v(0.62, 0.1))
        f12 = force(b1.get_center(), LEFT * 1.5, C_FORCE, r"\vec F_{12}", label_dir=UP)
        f21 = force(b2.get_center(), RIGHT * 1.5, C_FORCE, r"\vec F_{21}", label_dir=UP)
        iso = txt("an isolated system: the two bodies interact only with each other", 26, MUTED)
        iso.next_to(sys_box, DOWN, buff=0.3)
        pic = VGroup(header, sys_box, b1, b2, f12, f21, iso)
        with self.say("Put the second and third laws together, and something remarkable follows. Take two "
                      "bodies that interact only with each other. By the third law, at every instant, the "
                      "force on body one due to body two is equal and opposite to the force on body two due "
                      "to body one. By the second law, each force equals the rate of change of momentum of "
                      "the body it acts on.") as c:
            self.play(FadeIn(header), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(Create(sys_box), FadeIn(b1), FadeIn(b2), run_time=0.9)
            self.play(GrowArrow(f12[0]), GrowArrow(f21[0]), FadeIn(f12[1]), FadeIn(f21[1]), FadeIn(iso),
                      run_time=0.9)
            self.at_sentence(c, 2)
            self.play(FadeOut(pic), run_time=0.5)
            self.play(FadeIn(l1[0]), Write(l1[1]), run_time=1.2)
            self.at_sentence(c, 3)
            self.play(FadeIn(l2[0]), Write(l2[1]), run_time=1.6)
        with self.say("Now add them. The rate of change of the total momentum is the sum of the two forces, "
                      "which is zero. So the total momentum never changes.") as c:
            self.play(FadeIn(l3[0]), Write(l3[1]), run_time=2.0)
            self.at_sentence(c, 2)
            self.play(Write(result), Create(bx), run_time=1.4)
        self.play(FadeOut(rows), VGroup(result, bx).animate.move_to(v(0, 1.9)), run_time=0.8)
        law = panel(12.6, 1.9).move_to(v(0, -0.6))
        law_t = lines(f"{hl('Conservation of linear momentum', C_MOM)}: if no net external force acts on a",
                      "system, its total momentum stays constant, whatever happens inside it.",
                      size=30, align=ORIGIN, buff=0.18).move_to(law)
        if law_t.width > 12.1:
            law_t.scale_to_fit_width(12.1)
        ex = txt("collisions  ·  explosions  ·  pushes  ·  recoil  ·  rockets", 28, MUTED).move_to(v(0, -2.3))
        with self.say("This is the law of conservation of linear momentum. If no net external force acts on "
                      "a system, its total momentum stays constant, whatever happens inside it: collisions, "
                      "explosions, or pushes.") as c:
            self.play(FadeIn(law), Write(law_t), run_time=2.0)
            self.at_sentence(c, 1, 2.5)
            self.play(FadeIn(ex), run_time=0.8)
        self.play(FadeOut(VGroup(result, bx, law, law_t, ex)), run_time=0.6)

    # ------------------------------------------------------------------ skaters
    def skaters_check(self):
        ice_y = -0.3
        ice = VGroup(Rectangle(width=14.4, height=0.3, fill_color="#7DD3FC", fill_opacity=0.25,
                               stroke_width=0).move_to(v(0, ice_y - 0.15)),
                     Line(v(-7.2, ice_y), v(7.2, ice_y), color="#7DD3FC", stroke_width=3))
        A = person(1.7).move_to(v(-3.6, ice_y), aligned_edge=DOWN)
        B = person(1.7).move_to(v(2.4, ice_y), aligned_edge=DOWN)
        na = txt("Anna, 50 kg", 24).next_to(A, LEFT, buff=0.3).shift(DOWN * 0.45)
        nb = txt("Ben, 75 kg", 24).next_to(B, RIGHT, buff=0.3).shift(DOWN * 0.45)
        k = 0.012  # units per kg m/s
        top = ice_y + 2.05
        pa = arrow(v(A.get_x(), top), v(A.get_x() - 150 * k, top), C_MOM)
        pb = arrow(v(B.get_x(), top), v(B.get_x() + 150 * k, top), C_MOM)
        va = arrow(v(A.get_x(), top + 0.5), v(A.get_x() - 3 * 0.35, top + 0.5), C_VEL, stroke=5, tip=0.18)
        vb = arrow(v(B.get_x(), top + 0.5), v(B.get_x() + 2 * 0.35, top + 0.5), C_VEL, stroke=5, tip=0.18)
        keys = VGroup(VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity", 22, C_VEL)),
                      VGroup(arrow(ORIGIN, RIGHT * 0.6, C_MOM, stroke=5, tip=0.18), txt("momentum", 22, C_MOM)))
        for kk in keys:
            kk.arrange(RIGHT, buff=0.15)
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(UR, buff=0.4).shift(DOWN * 0.3)
        pos = txt("take right as positive  →", 24, MUTED).move_to(v(-4.3, 3.05))
        before = tex(r"\text{before:}\quad p_{\text{total}} = 0", size=38).move_to(v(0, -1.35))
        after = VGroup(
            tex(r"p_{\text{Anna}} = 50 \times (-3) = -150\ \text{kg\,m/s}", size=36),
            tex(r"p_{\text{Ben}} = 75 \times (+2) = +150\ \text{kg\,m/s}", size=36),
            tex(r"\text{after:}\quad p_{\text{total}} = -150 + 150 = 0", size=40, color=C_MOM),
        ).arrange(DOWN, buff=0.22).move_to(v(0, -2.75))
        with self.say("Let's check this with our skaters. Before the push, both are at rest, so the total "
                      "momentum is zero. After the push, Anna moves left at 3 metres per second. Taking right "
                      "as positive, her momentum is 50 times minus 3: minus 150 kilogram metres per second. Ben "
                      "moves right at 2 metres per second: 75 times 2, which is plus 150. The total is still "
                      "exactly zero.") as c:
            self.play(FadeIn(ice), FadeIn(A), FadeIn(B), FadeIn(na), FadeIn(nb), FadeIn(pos), FadeIn(keys),
                      run_time=0.8)
            self.at_sentence(c, 1)
            self.play(Write(before), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(FadeOut(before), GrowArrow(va), GrowArrow(vb), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(GrowArrow(pa), Write(after[0]), run_time=1.2)
            self.at_sentence(c, 4)
            self.play(GrowArrow(pb), Write(after[1]), run_time=1.2)
            self.at_sentence(c, 5)
            self.play(Write(after[2]), run_time=1.2)
        with self.say("The lighter skater moves faster, but the two momenta are equal and opposite, so "
                      "they cancel.") as c:
            self.play(Indicate(VGroup(pa, pb), color=C_MOM, scale_factor=1.08), run_time=1.2)
        self.play(FadeOut(VGroup(ice, A, B, na, nb, pa, pb, va, vb, keys, pos, after)), run_time=0.6)

    # ------------------------------------------------------------------ recoil and rockets
    def recoil(self):
        pnl = VGroup(panel(6.6, 4.8), panel(6.6, 4.8)).arrange(RIGHT, buff=0.35).move_to(v(0, -0.4))
        t1 = bold("Recoil of a gun", 30).next_to(pnl[0].get_top(), DOWN, buff=0.25)
        t2 = bold("Rocket", 30).next_to(pnl[1].get_top(), DOWN, buff=0.25)
        g = gun(0.75)
        g.shift(v(-5.0, 0.1) - g[0].get_left())
        yb = g[0].get_y()
        bullet = RoundedRectangle(width=0.36, height=0.14, corner_radius=0.06, fill_color="#FBBF24",
                                  fill_opacity=1, stroke_width=0).move_to(v(-2.7, yb))
        gl, br = g[0].get_left()[0], bullet.get_right()[0]
        p_g = arrow(v(gl, 0.72), v(gl - 1.1, 0.72), C_MOM)
        p_b = arrow(v(br, 0.72), v(br + 1.1, 0.72), C_MOM)
        v_g = arrow(v(gl, 1.15), v(gl - 0.3, 1.15), C_VEL, stroke=5, tip=0.16)
        v_b = arrow(v(br, 1.15), v(br + 2.0, 1.15), C_VEL, stroke=5, tip=0.18)
        cap1 = lines("equal and opposite momenta:", "the heavy gun recoils slowly", size=22, color=MUTED,
                     align=ORIGIN, buff=0.08).move_to(v(pnl[0].get_x(), -2.25))
        rk = rocket(1.9).move_to(v(1.9, -0.25))
        gas = VGroup(*[Circle(radius=r, fill_color="#94A3B8", fill_opacity=0.35, stroke_width=0)
                       .move_to(rk.get_bottom() + v(dx, dy))
                       for r, dx, dy in ((0.24, 0, -0.26), (0.18, -0.2, -0.64), (0.2, 0.18, -0.66))])
        p_r = arrow(v(2.75, -0.1), v(2.75, 1.2), C_MOM)
        p_x = arrow(v(2.75, -1.05), v(2.75, -2.35), C_MOM)
        lr = txt("rocket: momentum up", 22, C_MOM).next_to(p_r, RIGHT, buff=0.15)
        lx = txt("gases: momentum down", 22, C_MOM).next_to(p_x, RIGHT, buff=0.15)
        keys = VGroup(VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity", 22, C_VEL)),
                      VGroup(arrow(ORIGIN, RIGHT * 0.6, C_MOM, stroke=5, tip=0.18), txt("momentum", 22, C_MOM)))
        for kk in keys:
            kk.arrange(RIGHT, buff=0.15)
        keys.arrange(RIGHT, buff=0.6).move_to(v(0, 2.7))
        with self.say("The same reasoning explains the recoil of a gun. The bullet gets a large momentum "
                      "forward, so the gun gets an equal momentum backward. Because the gun is much heavier, "
                      "it recoils slowly.") as c:
            self.play(FadeIn(pnl[0]), FadeIn(t1), FadeIn(g), FadeIn(bullet), FadeIn(keys), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(GrowArrow(p_b), GrowArrow(p_g), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(GrowArrow(v_b), GrowArrow(v_g), FadeIn(cap1), run_time=1.0)
        with self.say("And it's how a rocket keeps accelerating: the exhaust gases carry momentum backward, "
                      "so the rocket gains momentum forward.") as c:
            self.play(FadeIn(pnl[1]), FadeIn(t2), FadeIn(rk), FadeIn(gas), run_time=0.8)
            self.play(GrowArrow(p_x), FadeIn(lx), run_time=0.8)
            self.play(GrowArrow(p_r), FadeIn(lr), run_time=0.8)
        self.play(FadeOut(VGroup(pnl, t1, t2, g, bullet, p_b, p_g, v_b, v_g, cap1, rk, gas, p_r, p_x, lr, lx, keys)),
                  FadeOut(self.label), run_time=0.7)
