"""Part 3 -- Newton's second law: momentum, F = dp/dt, F = ma, components, instantaneity."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, ball, block, boxed, cart, dimension_label, force, ground, panel, shown, v
from lesson.style import *

G = 9.8


class S03_SecondLaw(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "3", "Newton's second law", "force and acceleration")
        self.momentum()
        self.statement()
        self.race("force")
        self.race("mass")
        self.unit()
        self.components()
        self.instantaneous()
        self.recap()

    # ------------------------------------------------------------------ momentum
    def momentum(self):
        head = bold("Momentum", 44, C_MOM).move_to(v(0, 2.7))
        eq = tex(r"\vec p", r"=", r"m", r"\vec v", size=64).next_to(head, DOWN, buff=0.4)
        eq[0].set_color(C_MOM)
        eq[3].set_color(C_VEL)
        unit = txt("a vector along the velocity  ·  unit: kg·m/s", 28, MUTED).next_to(eq, DOWN, buff=0.3)
        rows = [("bicycle + rider", "80 kg", "5 m/s", "400"),
                ("loaded truck", "20 000 kg", "5 m/s", "100 000"),
                ("slow cricket ball", "0.16 kg", "10 m/s", "1.6"),
                ("fast cricket ball", "0.16 kg", "40 m/s", "6.4")]
        header = ["", "mass", "speed", "momentum (kg·m/s)"]
        xs = [-4.3, -0.9, 1.2, 4.0]
        table_rows = VGroup()
        for j, htxt in enumerate(header):
            table_rows.add(bold(htxt, 26, MUTED).move_to(v(xs[j], -0.55)))
        for i, row in enumerate(rows):
            y = -1.15 - 0.55 * i
            for j, cell in enumerate(row):
                col = C_MOM if j == 3 else TEXT
                t = (bold(cell, 28, col) if j == 3 else txt(cell, 28, col)).move_to(v(xs[j], y))
                if j == 0:
                    t.move_to(v(xs[0], y)).align_to(v(-6.2, 0), LEFT)
                table_rows.add(t)
        rule = Line(v(-6.3, -0.82), v(6.3, -0.82), color=DIM, stroke_width=2)
        sep = Line(v(-6.3, -2.02), v(6.3, -2.02), color=DIM, stroke_width=1.5).set_opacity(0.6)
        r1 = VGroup(bold("First law:", 34, MUTED),
                    tex(r"\vec F_{\text{net}} = \vec 0 \;\Rightarrow\; \vec v = \text{constant}", size=44))
        r2 = VGroup(bold("Second law:", 34, ACCENT),
                    tex(r"\vec F_{\text{net}} \neq \vec 0 \;\Rightarrow\; \;?", size=44))
        for r in (r1, r2):
            r.arrange(RIGHT, buff=0.4)
        rows = VGroup(r1, r2).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(v(0, 0.6))
        qm = VGroup(txt("Newton's “quantity of motion”", 32, TEXT),
                    txt("(quantitas motus, in the Latin of the Principia)", 24, MUTED)).arrange(DOWN, buff=0.15)
        qm.move_to(v(0, -1.9))
        with self.say("The first law tells us what happens when the net force is zero. The second law "
                      "tells us what happens when it is not. To state it, Newton used a quantity he called "
                      "the quantity of motion. Today we call it momentum.") as c:
            self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(qm), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeOut(rows), FadeOut(qm), FadeIn(head, shift=DOWN * 0.2), run_time=0.8)
        with self.say("Momentum is mass times velocity. It is a vector, pointing the same way as the "
                      "velocity, and it is measured in kilogram metres per second. A loaded truck is much "
                      "harder to stop than a bicycle moving at the same speed, and a fast cricket ball is "
                      "harder to stop than a slow one. Momentum captures both effects.") as c:
            self.play(Write(eq), run_time=1.2)
            self.at_sentence(c, 1)
            self.play(FadeIn(unit), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(table_rows[:4]), Create(rule), run_time=0.6)
            self.play(FadeIn(table_rows[4:12]), run_time=1.0)
            self.play(FadeIn(sep), FadeIn(table_rows[12:]), run_time=1.0)
        self.play(FadeOut(VGroup(head, eq, unit, table_rows, rule, sep)), run_time=0.6)

    # ------------------------------------------------------------------ statement and F = ma
    def statement(self):
        box = panel(12.8, 2.1).move_to(v(0, 1.5))
        tag = bold("Newton's second law", 28, ACCENT).next_to(box, UP, buff=0.15).align_to(box, LEFT)
        law = lines(f"The {hl('rate of change of momentum', C_MOM)} of a body is proportional to the",
                    f"{hl('net external force', C_FORCE)} on it, and takes place in the direction of that force.",
                    size=32, align=ORIGIN, buff=0.2).move_to(box)
        if law.width > 12.3:
            law.scale_to_fit_width(12.3)
        eq1 = tex(r"\vec F_{\text{net}}", r"=", r"\frac{d\vec p}{dt}", size=60).move_to(v(0, 0.1))
        eq1[0].set_color(C_FORCE)
        eq1[2].set_color(C_MOM)
        si = txt("(in SI units the constant of proportionality is exactly 1)", 26, MUTED).next_to(eq1, DOWN, buff=0.3)
        with self.say("Newton's second law states: the rate of change of momentum of a body is "
                      "proportional to the net external force acting on it, and takes place in the "
                      "direction of that force. In SI units, the constant of proportionality is exactly "
                      "one. So, the net force equals dp/dt.") as c:
            self.play(FadeIn(box), FadeIn(tag), run_time=0.6)
            self.play(Write(law), run_time=min(4.5, self.remaining(c, c.end_of(0))))
            self.at_sentence(c, 1)
            self.play(FadeIn(si), run_time=0.6)
            self.at_sentence(c, 2)
            self.play(Write(eq1), run_time=1.4)

        deriv = tex(r"\vec F_{\text{net}}", r"=", r"\frac{d\vec p}{dt}", r"=", r"\frac{d(m\vec v)}{dt}",
                    r"=", r"m\,\frac{d\vec v}{dt}", r"=", r"m\vec a", size=52).move_to(v(0, -0.45))
        deriv[0].set_color(C_FORCE)
        deriv[2].set_color(C_MOM)
        deriv[8].set_color(C_ACC)
        cond = txt("(mass constant)", 26, MUTED).next_to(deriv[6], DOWN, buff=0.25)
        final = tex(r"\vec F_{\text{net}} = m\vec a", size=72).move_to(v(0, -2.55))
        final[0][0:4].set_color(C_FORCE)
        final[0][5:7].set_color(C_ACC)
        fbox = boxed(final)
        with self.say("If the mass of the body stays constant, the mass comes outside the derivative. The "
                      "rate of change of m v is m times the rate of change of v, and the rate of change of "
                      "velocity is the acceleration. So, for constant mass, the net force equals mass times "
                      "acceleration. This is the form you will use most often.") as c:
            self.play(FadeOut(si), ReplacementTransform(eq1, deriv[:3]), run_time=1.0)
            self.play(Write(deriv[3:5]), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(Write(deriv[5:7]), FadeIn(cond), run_time=1.2)
            self.play(Write(deriv[7:]), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(Write(final), Create(fbox), run_time=1.4)
        self.play(FadeOut(VGroup(box, tag, law, deriv, cond)), VGroup(final, fbox).animate.move_to(v(0, 2.3)),
                  run_time=0.8)

        a_eq = tex(r"\vec a", r"=", r"\frac{\vec F_{\text{net}}}{m}", size=64).move_to(v(-2.6, -0.3))
        a_eq[0].set_color(C_ACC)
        notes = lines(f"proportional to the {hl('net force', C_FORCE)}",
                      f"inversely proportional to the {hl('mass')}",
                      f"points in the {hl('direction of the net force', C_FORCE)}",
                      "(not necessarily the direction of motion)",
                      size=28, buff=0.25).next_to(a_eq, RIGHT, buff=0.8)
        zero = mtxt(f"If {hl('F<sub>net</sub> = 0', C_FORCE)}, then {hl('a = 0', C_ACC)}: consistent with the first law.",
                    28).move_to(v(0, -2.7))
        with self.say("Rearranged, the acceleration equals the net force divided by the mass. So the "
                      "acceleration is proportional to the net force, inversely proportional to the mass, "
                      "and it always points in the direction of the net force, which is not necessarily "
                      "the direction of motion. And if the net force is zero, the acceleration is zero, "
                      "which agrees with the first law.") as c:
            self.play(Write(a_eq), run_time=1.2)
            self.at_sentence(c, 1)
            for row in notes:
                self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.7)
            self.at_sentence(c, 2)
            self.play(FadeIn(zero), run_time=0.8)
        self.play(FadeOut(VGroup(final, fbox, a_eq, notes, zero)), run_time=0.6)

    # ------------------------------------------------------------------ cart races
    def race(self, which: str):
        scale = 2.0  # units per metre
        T = 2.0
        if which == "force":
            specs = [("1 kg", 1.0, 1.0, 1.25), ("1 kg", 1.0, 2.0, -1.35)]  # (label, m, F, y)
            text = ("Let's see this in action. Two identical one kilogram carts start from rest on a "
                    "frictionless track. The top one is pulled with a net force of one newton, the bottom "
                    "one with two newtons. Double the force gives double the acceleration: one metre per "
                    "second squared, compared with two. So in the same time, the bottom cart travels "
                    "twice as far.")
            title = "Same mass, double the force"
        else:
            specs = [("1 kg", 1.0, 2.0, 1.25), ("2 kg", 2.0, 2.0, -1.35)]
            text = ("Now keep the force fixed at two newtons, but double the mass. The two kilogram cart "
                    "accelerates at only one metre per second squared: half as much. Doubling the "
                    "inertia halves the acceleration.")
            title = "Same force, double the mass"
        heading = bold(title, 34).move_to(v(0, 3.1))
        clock = ValueTracker(0.0)
        x_start = -5.6
        mobs = VGroup()
        carts, finals = [], []
        for lab, m, F, y in specs:
            trk = ground(-7.0, 7.0, y)
            w = 1.3 if m < 1.5 else 1.9
            c = cart(w, 0.55 if m < 1.5 else 0.75, lab)
            c.move_to(v(x_start, y), aligned_edge=DOWN)
            c.start = c.get_center()
            a = F / m
            c.a = a
            c.add_updater(lambda mob: mob.move_to(mob.start + RIGHT * 0.5 * mob.a * clock.get_value() ** 2 * scale))
            pull = always_redraw(lambda c=c, F=F: force(c.body.get_right(), RIGHT * 0.9 * F, C_FORCE,
                                                        f"{F:.0f}\\,\\text{{N}}", label_dir=UP, label_size=30))
            vel = always_redraw(lambda c=c: arrow(c.get_top() + UP * 0.3,
                                                  c.get_top() + UP * 0.3 + RIGHT * 0.45 * c.a * clock.get_value(),
                                                  C_VEL, stroke=5, tip=0.18))
            acc_lab = tex(rf"a = \frac{{{F:.0f}\ \text{{N}}}}{{{m:.0f}\ \text{{kg}}}} = {a:.0f}\ \text{{m/s}}^2",
                          size=32, color=C_ACC).move_to(v(4.9, y - 0.75))
            carts.append(c)
            finals.append((c, a))
            mobs.add(trk, c, pull, vel)
            c.acc_lab = acc_lab
        timer = always_redraw(lambda: tex(rf"t = {clock.get_value():.1f}\ \text{{s}}", size=34, color=MUTED)
                              .move_to(v(5.6, 3.1)))
        with self.say(text) as c:
            self.play(FadeIn(heading), FadeIn(mobs), FadeIn(timer), run_time=0.8)
            self.at_sentence(c, 2 if which == "force" else 1)
            self.play(*[FadeIn(cc.acc_lab) for cc in carts], run_time=0.8)
            self.play(clock.animate.set_value(T), run_time=T / 0.5, rate_func=linear)
            dims = VGroup()
            for cc, a in finals:
                d = 0.5 * a * T**2
                p0 = cc.start + DOWN * (cc.height / 2)
                dims.add(dimension_label(p0, p0 + RIGHT * d * scale, f"{d:.0f} m in {T:.0f} s",
                                         color=TEXT, offset=DOWN * 0.45, size=26))
            self.play(FadeIn(dims), run_time=0.8)
        for m in carts + list(mobs):
            m.clear_updaters()
        timer.clear_updaters()
        self.play(FadeOut(VGroup(heading, mobs, timer, dims, *[cc.acc_lab for cc in carts])), run_time=0.6)

    # ------------------------------------------------------------------ the newton
    def unit(self):
        head = bold("The unit of force: the newton", 38).move_to(v(0, 2.3))
        defn = mtxt(f"{hl('1 newton')} is the net force that gives a {hl('1 kg')} mass",
                    32).next_to(head, DOWN, buff=0.5)
        defn2 = mtxt(f"an acceleration of {hl('1 m/s²')}.", 32).next_to(defn, DOWN, buff=0.18)
        eq = tex(r"1\ \text{N} = 1\ \text{kg}\cdot\text{m/s}^2", size=64).next_to(defn2, DOWN, buff=0.6)
        bx = boxed(eq)
        with self.say("This also defines the unit of force. One newton is the net force that gives a mass "
                      "of one kilogram an acceleration of one metre per second squared. So one newton "
                      "equals one kilogram metre per second squared.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(defn), FadeIn(defn2), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(Write(eq), Create(bx), run_time=1.4)
        self.play(FadeOut(VGroup(head, defn, defn2, eq, bx)), run_time=0.6)

    # ------------------------------------------------------------------ components and a projectile
    def components(self):
        eqs = tex(r"\sum F_x = m a_x", r"\qquad", r"\sum F_y = m a_y", size=46).move_to(v(0, 3.0))
        vx, vy0 = 4.0, 6.0  # m/s
        s = 1.5  # units per metre
        origin = v(-5.4, -2.6)
        T = 2 * vy0 / G
        grnd = ground(-7.0, 7.0, origin[1] - 0.2)
        clock = ValueTracker(0.0)

        def pos(t):
            return origin + s * v(vx * t, vy0 * t - 0.5 * G * t * t)

        bl = ball(0.18, "#F59E0B", stripe=False).move_to(pos(0))
        bl.add_updater(lambda m: m.move_to(pos(clock.get_value())))
        k = 0.2  # units per (m/s) for velocity arrows

        def arrows():
            t = clock.get_value()
            p = pos(t)
            vy = vy0 - G * t
            hx = arrow(p, p + RIGHT * k * vx, C_VEL, stroke=5, tip=0.18)
            hy = arrow(p, p + UP * k * vy, C_VEL, stroke=5, tip=0.18)
            w = arrow(p + LEFT * 0.16, p + LEFT * 0.16 + DOWN * 0.9, C_WEIGHT, stroke=5, tip=0.18)
            # once the ball has landed it is no longer a projectile: hide the arrows
            return shown(VGroup(w, hx, hy), t < T - 1e-6)

        arr = always_redraw(arrows)
        ghosts = VGroup(*[ball(0.18, "#F59E0B", stripe=False).move_to(pos(t)).set_opacity(0.3)
                          for t in np.arange(0.1, T - 1e-6, 0.1)])
        ghost_ticks = VGroup(*[DashedLine(g.get_center(), v(g.get_x(), origin[1] - 0.2), color=DIM,
                                          stroke_width=1.5, dash_length=0.06) for g in ghosts])
        keys = VGroup(
            VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity components", 24, C_VEL)),
            VGroup(arrow(ORIGIN, RIGHT * 0.6, C_WEIGHT, stroke=5, tip=0.18), txt("weight: the only force", 24, C_WEIGHT)))
        for kk in keys:
            kk.arrange(RIGHT, buff=0.15)
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.4).shift(DOWN * 0.9)
        n_x = mtxt(f"horizontal: no force, so {hl('v<sub>x</sub> stays constant', C_VEL)}", 26)
        n_y = mtxt(f"vertical: weight gives {hl('a<sub>y</sub> = −g', C_ACC)}", 26)
        notes = VGroup(n_x, n_y).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(v(-2.6, 2.05))
        slow = txt("0.4× speed · ghost images every 0.1 s", 22, MUTED)
        slow.next_to(keys, DOWN, buff=0.25).align_to(keys, RIGHT)
        with self.say("Force and acceleration are vectors, so the second law holds separately along each "
                      "axis: the sum of the x components of the forces equals m times the x component of "
                      "the acceleration, and the same for y. A thrown ball shows this beautifully.") as c:
            self.play(Write(eqs), run_time=1.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(grnd), FadeIn(bl), run_time=0.6)
        with self.say("Once it leaves your hand, if we ignore air resistance, the only force on it is its "
                      "weight, straight down. Horizontally there is no force, so the horizontal velocity "
                      "stays constant: that's the first law at work. Vertically, the weight gives a "
                      "constant downward acceleration, g. Together, they make the curved path of a "
                      "projectile.") as c:
            self.add(arr)
            self.play(FadeIn(keys), FadeIn(slow), run_time=0.5)
            self.play(clock.animate.set_value(T), run_time=T / 0.4, rate_func=linear)
            self.at_sentence(c, 1)
            self.play(FadeIn(ghosts), FadeIn(ghost_ticks), FadeIn(n_x), run_time=1.0)
            self.at_sentence(c, 2)
            self.play(FadeIn(n_y), run_time=0.8)
            self.at_sentence(c, 3)
            clock.set_value(0)
            self.play(clock.animate.set_value(T), run_time=T / 0.4, rate_func=linear)
        bl.clear_updaters()
        arr.clear_updaters()
        self.play(FadeOut(VGroup(eqs, grnd, bl, arr, ghosts, ghost_ticks, keys, notes, slow)), run_time=0.6)

    # ------------------------------------------------------------------ instantaneous
    def instantaneous(self):
        # 1 kg cart, 1 N net force for 0 <= t < 2 s, then no force; runs to t = 4 s.
        clock = ValueTracker(0.0)
        scale = 1.55
        trk = ground(-7.0, 7.0, 1.3)
        crt = cart(1.3, 0.5, "1 kg").move_to(v(-6.0, 1.3), aligned_edge=DOWN)
        start = crt.get_center()

        def xpos(t):
            return 0.5 * t * t if t <= 2 else 2 + 2 * (t - 2)

        crt.add_updater(lambda m: m.move_to(start + RIGHT * scale * xpos(clock.get_value())))
        pull = always_redraw(lambda: shown(force(crt.body.get_right(), RIGHT * 1.0, C_FORCE, r"1\,\text{N}",
                                                 label_dir=UP, label_size=28), clock.get_value() < 2))
        vel = always_redraw(lambda: arrow(crt.get_top() + UP * 0.3, crt.get_top() + UP * 0.3 +
                                          RIGHT * 0.5 * min(clock.get_value(), 2), C_VEL, stroke=5, tip=0.18))
        specs = [("F (N)", 1.5, C_FORCE, lambda t: 1.0 if t < 2 else 0.0),
                 ("a (m/s²)", 1.5, C_ACC, lambda t: 1.0 if t < 2 else 0.0),
                 ("v (m/s)", 2.5, C_VEL, lambda t: min(t, 2.0))]
        axes_list, curves = VGroup(), VGroup()
        for i, (ylab, ymax, col, f) in enumerate(specs):
            ax = Axes(x_range=[0, 4.4, 1], y_range=[0, ymax, 1 if ymax < 2 else 1], x_length=3.9, y_length=2.1,
                      axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False,
                                   "font_size": 22, "include_numbers": True},
                      ).move_to(v(-4.45 + 4.45 * i, -1.55))
            xl = txt("t (s)", 20, MUTED).next_to(ax.x_axis, DOWN, buff=0.35).align_to(ax.x_axis, RIGHT)
            yl = txt(ylab, 22, col).next_to(ax.y_axis, UP, buff=0.12)
            axes_list.add(VGroup(ax, xl, yl))

            def make(ax=ax, f=f, col=col):
                t_now = max(clock.get_value(), 1e-3)
                ts = np.linspace(0, t_now, 120)
                pts = []
                for t in ts:
                    if len(pts) and ts[0] <= 2 <= t and not any(abs(p_[0] - 2) < 1e-9 for p_ in pts):
                        pts.append((2.0, f(1.999)))
                        pts.append((2.0, f(2.0)))
                    pts.append((t, f(t)))
                curve = VMobject(stroke_color=col, stroke_width=5).set_points_as_corners(
                    [ax.c2p(x, y) for x, y in pts])
                return curve

            curves.add(always_redraw(make))
        cursor = always_redraw(lambda: VGroup(*[DashedLine(a[0].c2p(clock.get_value(), 0),
                                                           a[0].c2p(clock.get_value(), a[0].y_range[1]),
                                                           color=TEXT, stroke_width=1.5, dash_length=0.06)
                                                for a in axes_list]))
        note = mtxt(f"force switched off: {hl('a drops to zero at once', C_ACC)}, "
                    f"{hl('v stays constant', C_VEL)}", 28).move_to(v(0, 3.05))
        with self.say("One more important point. The second law links the net force to the acceleration "
                      "at the same instant. If the force suddenly stops, the acceleration drops to zero at "
                      "once, but the velocity does not. The body simply carries on with the velocity it "
                      "had, just as the first law says.") as c:
            self.play(FadeIn(trk), FadeIn(crt), FadeIn(axes_list), run_time=0.8)
            self.add(pull, vel, curves, cursor)
            self.at_sentence(c, 1)
            self.play(clock.animate.set_value(2.0), run_time=2.0 / 0.5, rate_func=linear)
            self.play(clock.animate.set_value(4.0), run_time=2.0 / 0.5, rate_func=linear)
            self.play(FadeIn(note), run_time=0.8)
        for m in (crt, pull, vel, cursor, *curves):
            m.clear_updaters()
        self.play(FadeOut(VGroup(trk, crt, pull, vel, axes_list, curves, cursor, note)), run_time=0.6)

    # ------------------------------------------------------------------ recap
    def recap(self):
        box = panel(11.5, 3.6).move_to(v(0, 0.2))
        ttl = bold("Second law in one line", 32, ACCENT).next_to(box.get_top(), DOWN, buff=0.35)
        eq = tex(r"\vec F_{\text{net}} = \frac{d\vec p}{dt} = m\vec a", size=56).next_to(ttl, DOWN, buff=0.35)
        sub = txt("(m a form: constant mass)   ·   a points along the net force", 28, MUTED).next_to(eq, DOWN, buff=0.35)
        with self.say("So, the second law in one line: the net force equals the rate of change of "
                      "momentum, which, for constant mass, is mass times acceleration. And the "
                      "acceleration always points along the net force."):
            self.play(FadeIn(box), FadeIn(ttl), run_time=0.6)
            self.play(Write(eq), run_time=1.4)
            self.play(FadeIn(sub), run_time=0.6)
        self.play(FadeOut(VGroup(box, ttl, eq, sub)), FadeOut(self.label), run_time=0.7)
