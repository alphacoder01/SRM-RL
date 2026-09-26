"""Part 2 -- Newton's first law: statement, net force, inertia, inertial frames."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import (arrow, ball, block, book, boxed, bus, car_side, cart, force,
                            glass, ground, panel, person, shown, surface, table, v)
from lesson.style import *

G = 9.8


class S02_FirstLaw(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "2", "Newton's first law", "the law of inertia")
        self.statement()
        self.circular()
        self.balanced()
        self.cruising_car()
        self.inertia()
        self.coin_card()
        self.bus_ride()
        self.frames()
        self.recap()

    # ------------------------------------------------------------------ statement
    def statement(self):
        box = panel(12.6, 2.2).move_to(v(0, 1.35))
        law = lines(f'A body stays {hl("at rest")}, or keeps moving {hl("in a straight line")}',
                    f'{hl("at constant speed")}, unless a {hl("net external force")} acts on it.',
                    size=36, align=ORIGIN, buff=0.22)
        law.move_to(box)
        if law.width > 12.0:
            law.scale_to_fit_width(12.0)
        tag = bold("Newton's first law", 28, ACCENT).next_to(box, UP, buff=0.2).align_to(box, LEFT)
        eq = tex(r"\vec F_{\text{net}} = \vec 0", r"\quad\Longleftrightarrow\quad",
                 r"\vec v = \text{constant}", size=52).move_to(v(0, -1.1))
        eq[0].set_color(C_FORCE)
        eq[2].set_color(C_VEL)
        with self.say("Newton's first law says: a body stays at rest, or keeps moving in a straight line "
                      "at constant speed, unless a net external force acts on it. In symbols: if the net "
                      "force on a body is zero, its velocity is constant. And if its velocity is "
                      "constant, the net force on it must be zero.") as c:
            self.play(FadeIn(box), FadeIn(tag), run_time=0.6)
            self.play(Write(law), run_time=min(4.0, self.remaining(c, c.end_of(0))))
            self.at_sentence(c, 1)
            self.play(Write(eq[0]), run_time=1.0)
            self.play(Write(eq[1]), Write(eq[2]), run_time=1.2)
            self.at_sentence(c, 2)
            self.play(Indicate(eq[2], color=C_VEL, scale_factor=1.08), run_time=1.0)
        self.play(FadeOut(VGroup(box, law, tag, eq)), run_time=0.6)

    # ------------------------------------------------------------------ circular motion
    def circular(self):
        centre, R = v(3.3, -0.55), 2.1
        track = DashedVMobject(Circle(radius=R, color=MUTED, stroke_width=2).move_to(centre), num_dashes=48)
        cdot = Dot(centre, radius=0.05, color=MUTED)
        cl = txt("centre", 22, MUTED).next_to(cdot, DOWN, buff=0.1)
        angle = ValueTracker(-PI / 2)

        def car_at():
            th = angle.get_value()
            pos = centre + R * v(np.cos(th), np.sin(th))
            body = RoundedRectangle(width=0.62, height=0.32, corner_radius=0.08, fill_color="#2563EB",
                                    fill_opacity=1, stroke_color=OBJ_STROKE, stroke_width=2)
            body.rotate(th + PI / 2).move_to(pos)
            tangent = v(-np.sin(th), np.cos(th))
            vel = arrow(pos + tangent * 0.36, pos + tangent * 1.55, C_VEL, stroke=5)
            fnet = arrow(pos, pos + (centre - pos) / R * 1.1, C_FRICTION, stroke=5)
            return VGroup(fnet, vel, body)

        car = always_redraw(car_at)
        v_key = VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity", 24, C_VEL))
        f_key = VGroup(arrow(ORIGIN, RIGHT * 0.6, C_FRICTION, stroke=5, tip=0.18),
                       txt("net force (friction)", 24, C_FRICTION))
        for k in (v_key, f_key):
            k.arrange(RIGHT, buff=0.15)
        keys = VGroup(v_key, f_key).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR, buff=0.4).shift(DOWN * 0.35)
        head = bold("Constant speed is not enough", 36).move_to(v(-3.4, 2.2))
        facts = lines(f"speed: {hl('constant', C_VEL)}",
                      f"direction: {hl('changing', C_FRICTION)}",
                      f"so the velocity is {hl('changing', C_FRICTION)}",
                      f"so a {hl('net force', C_FRICTION)} must act,",
                      "pointing towards the centre", size=30, buff=0.22)
        facts.next_to(head, DOWN, buff=0.45).align_to(head, LEFT)

        angle.add_updater(lambda m, dt: m.increment_value(0.9 * dt))  # constant speed
        with self.say("Pay attention to the words: in a straight line. Constant velocity means constant "
                      "speed and constant direction. A car going round a curve at a steady 50 kilometres "
                      "per hour does not have constant velocity, because its direction keeps changing. "
                      "So a net force must act on it. Here, it's the sideways friction of the road on the "
                      "tyres, pointing towards the centre of the curve.") as c:
            self.add(angle)
            self.play(FadeIn(head), FadeIn(track), FadeIn(cdot), FadeIn(cl), run_time=0.8)
            self.add(car)
            self.play(FadeIn(keys), run_time=0.4)
            self.at_sentence(c, 1)
            self.play(FadeIn(facts[0], shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeIn(facts[1], shift=RIGHT * 0.2), run_time=0.6)
            self.at_sentence(c, 2, 2.0)
            self.play(FadeIn(facts[2], shift=RIGHT * 0.2), run_time=0.6)
            self.at_sentence(c, 3)
            self.play(FadeIn(facts[3], shift=RIGHT * 0.2), FadeIn(facts[4], shift=RIGHT * 0.2), run_time=0.8)
        angle.clear_updaters()
        car.clear_updaters()
        self.play(FadeOut(VGroup(head, track, cdot, cl, car, keys, facts)), run_time=0.6)
        self.remove(angle)

    # ------------------------------------------------------------------ balanced forces: book
    def balanced(self):
        head = bold("Balanced forces", 38).move_to(v(0, 2.6))
        # three forces on one point whose vector sum is zero
        P, Q = v(-3.2, -0.4), v(1.6, -1.5)
        F = [v(2.0, 0.7), v(-1.3, 1.3)]
        F.append(-(F[0] + F[1]))
        cols = [C_FORCE, C_TENSION, C_NORMAL]
        names = [r"\vec F_1", r"\vec F_2", r"\vec F_3"]
        on_point = VGroup(Dot(P, radius=0.07, color=TEXT),
                          *[force(P, f, col, nm, label_size=32) for f, col, nm in zip(F, cols, names)])
        tip, tri = Q.copy(), VGroup()
        for f, col in zip(F, cols):
            tri.add(arrow(tip, tip + f, col))
            tip = tip + f
        tri_lab = tex(r"\vec F_1 + \vec F_2 + \vec F_3 = \vec 0", size=36).next_to(tri, DOWN, buff=0.35)
        closed = txt("head to tail, they close up: zero net force", 26, MUTED).next_to(tri, UP, buff=0.35)

        tb = table(3.6, top_y=-1.2, x=-4.6, leg_h=1.5)
        bk = book(1.6, 0.42).move_to(v(-4.6, -1.2 + 0.21))
        fbd_c = v(-1.0, -0.6)
        fbd_dot = Dot(fbd_c, radius=0.08, color=TEXT)
        fbd_lab = txt("the book", 22, MUTED).next_to(fbd_dot, LEFT, buff=0.15)
        w = force(fbd_c, DOWN * 1.4, C_WEIGHT, r"\vec W", label_dir=RIGHT)
        n = force(fbd_c, UP * 1.4, C_NORMAL, r"\vec N", label_dir=RIGHT)
        fbd_title = txt("free-body diagram", 24, MUTED).next_to(fbd_c + UP * 1.4, UP, buff=0.45)
        info = lines(f"{hl('weight', C_WEIGHT)}: Earth pulls the book down",
                     f"{hl('normal force', C_NORMAL)}: table pushes it up",
                     "equal in size, opposite in direction", size=28, buff=0.22)
        info.move_to(v(3.7, 0.9))
        eq = tex(r"\vec F_{\text{net}} = \vec W + \vec N = \vec 0", size=40).next_to(info, DOWN, buff=0.5)
        res = txt("so the book stays at rest", 30, ACCENT).next_to(eq, DOWN, buff=0.35)
        with self.say("Now pay attention to the word: net. Many forces can act on a body at once. If "
                      "they balance, so that their vector sum is zero, the body moves exactly as if no "
                      "force acted on it at all. A book lying on a table feels two forces: its weight, "
                      "pulling down, and the normal force from the table, pushing up. They balance, so "
                      "the book stays at rest.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(on_point[0]), *[GrowArrow(f[0]) for f in on_point[1:]],
                      *[FadeIn(f[1]) for f in on_point[1:]], run_time=1.2)
            self.at_sentence(c, 2)
            self.play(LaggedStart(*[TransformFromCopy(on_point[i + 1][0], tri[i]) for i in range(3)],
                                  lag_ratio=0.6), run_time=2.2)
            self.play(FadeIn(tri_lab), FadeIn(closed), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeOut(VGroup(on_point, tri, tri_lab, closed)), run_time=0.5)
            self.play(FadeIn(tb), FadeIn(bk), run_time=0.6)
            self.play(FadeIn(fbd_dot), FadeIn(fbd_lab), FadeIn(fbd_title), run_time=0.5)
            self.play(GrowArrow(w[0]), FadeIn(w[1]), FadeIn(info[0]), run_time=0.9)
            self.play(GrowArrow(n[0]), FadeIn(n[1]), FadeIn(info[1]), run_time=0.9)
            self.play(FadeIn(info[2]), run_time=0.5)
            self.at_sentence(c, 4)
            self.play(Write(eq), run_time=1.2)
            self.play(FadeIn(res), run_time=0.6)
        self.play(FadeOut(VGroup(head, tb, bk, fbd_dot, fbd_lab, fbd_title, w, n, info, eq, res)), run_time=0.6)

    # ------------------------------------------------------------------ cruising car
    def cruising_car(self):
        road_y = -1.6
        speed = 1.6  # scroll speed of the road markings (the camera moves with the car)
        road = Line(v(-7.3, road_y), v(7.3, road_y), color=GROUND, stroke_width=3)
        span = 14.4
        dashes = VGroup(*[Line(v(x, road_y - 0.45), v(x + 0.5, road_y - 0.45), color=MUTED, stroke_width=3)
                          for x in np.arange(-7.2, 7.2, 1.2)])
        posts = VGroup(*[VGroup(Line(v(x, road_y), v(x, road_y + 0.55), color=DIM, stroke_width=4),
                                Dot(v(x, road_y + 0.6), radius=0.07, color=DIM))
                         for x in np.arange(-7.2, 7.2, 3.6)])

        def scroll(group):
            def upd(m, dt):
                for piece in m:
                    piece.shift(LEFT * speed * dt)
                    if piece.get_center()[0] < -7.4:
                        piece.shift(RIGHT * span)
            return upd

        dashes.add_updater(scroll(dashes))
        posts.add_updater(scroll(posts))
        car = car_side(2.6).move_to(v(0.0, road_y), aligned_edge=DOWN)
        cm = car.body.get_center() + UP * 0.1
        fwd = force(cm, RIGHT * 1.5, C_FORCE)
        back = force(cm, LEFT * 1.5, C_FRICTION)
        vel = arrow(car.get_top() + v(-0.6, 0.35), car.get_top() + v(0.6, 0.35), C_VEL, stroke=5)
        cmdot = Dot(cm, radius=0.05, color=TEXT)
        keys = VGroup(
            VGroup(arrow(ORIGIN, RIGHT * 0.6, C_FORCE, stroke=5, tip=0.18),
                   txt("road pushes car forward (friction on the tyres)", 24, C_FORCE)),
            VGroup(arrow(ORIGIN, RIGHT * 0.6, C_FRICTION, stroke=5, tip=0.18),
                   txt("air resistance + rolling resistance", 24, C_FRICTION)),
            VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity", 24, C_VEL)))
        for k in keys:
            k.arrange(RIGHT, buff=0.15)
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(v(-3.2, 2.2))
        verdict = tex(r"\vec F_{\text{net}} = \vec 0 \;\Rightarrow\; \vec v = \text{constant}", size=40)
        verdict.move_to(v(3.9, 2.2))
        camnote = txt("camera moving along with the car", 22, MUTED).to_corner(DR, buff=0.3)
        with self.say("A car cruising at a steady speed on a straight, level road is the same. The road "
                      "pushes the car forward, through friction on its driven wheels, while air "
                      "resistance and rolling resistance push it backward. When these forces balance, "
                      "the net force is zero, and the velocity stays constant. With no resistive forces at "
                      "all, the car would cruise on with no engine force. The engine is needed only to cancel "
                      "the resistive forces.") as c:
            self.play(FadeIn(road), FadeIn(dashes), FadeIn(posts), FadeIn(car), FadeIn(camnote), run_time=0.6)
            self.play(GrowArrow(vel), FadeIn(keys[2]), run_time=0.5)
            self.at_sentence(c, 1)
            self.play(FadeIn(cmdot), GrowArrow(fwd[0]), FadeIn(keys[0]), run_time=0.8)
            self.play(GrowArrow(back[0]), FadeIn(keys[1]), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(Write(verdict), run_time=1.0)
        dashes.clear_updaters()
        posts.clear_updaters()
        self.play(FadeOut(VGroup(road, dashes, posts, car, fwd, back, vel, cmdot, keys, verdict, camnote)),
                  run_time=0.6)

    # ------------------------------------------------------------------ inertia and mass
    def inertia(self):
        head = bold("Inertia", 44, ACCENT).move_to(v(0, 2.75))
        defn = mtxt("the tendency of a body to <b>resist any change</b> in its motion", 32).next_to(head, DOWN, buff=0.2)
        meas = mtxt(f'{hl("Mass")} is the measure of inertia.', 32).next_to(defn, DOWN, buff=0.2)
        y1, y2 = 0.45, -2.45
        track1, track2 = ground(-7.0, 7.0, y1), ground(-7.0, 7.0, y2)
        c1 = cart(1.5, 0.5, "empty").move_to(v(-4.8, y1), aligned_edge=DOWN)
        c2 = cart(1.5, 0.5).move_to(v(-4.8, y2), aligned_edge=DOWN)
        bricks = VGroup(*[Rectangle(width=0.42, height=0.22, fill_color="#B45309", fill_opacity=1,
                                    stroke_color="#78350F", stroke_width=2) for _ in range(9)])
        bricks.arrange_in_grid(3, 3, buff=0.02).next_to(c2.body, UP, buff=0.0)
        c2.add(bricks)
        lab2 = txt("loaded: 10× the mass", 24, TEXT).next_to(c2, DOWN, buff=0.35).align_to(c2, LEFT)
        # Same impulse J = F dt for both: dv = J/m  ->  2.0 m/s (m) vs 0.2 m/s (10 m).
        t_push, v1, v2 = 0.3, 2.0, 0.2
        clock = ValueTracker(0.0)

        def x_of(t, vf):
            a = vf / t_push
            if t <= t_push:
                return 0.5 * a * t * t
            return 0.5 * a * t_push**2 + vf * (t - t_push)

        s1, s2 = c1.get_center(), c2.get_center()
        c1.add_updater(lambda m: m.move_to(s1 + RIGHT * x_of(clock.get_value(), v1)))
        c2.add_updater(lambda m: m.move_to(s2 + RIGHT * x_of(clock.get_value(), v2)))
        pushes = always_redraw(lambda: VGroup(*[shown(force(cc.get_left() + LEFT * 1.1 + UP * 0.1, RIGHT * 1.1,
                                                            C_FORCE, txt("push", 22, C_FORCE), label_dir=UP),
                                                      clock.get_value() < t_push)
                                                for cc in (c1, c2)]))
        vels = always_redraw(lambda: VGroup(*[shown(arrow(cc.get_top() + UP * 0.25,
                                                          cc.get_top() + UP * 0.25 + RIGHT * 0.9 * vf, C_VEL, stroke=5),
                                                    clock.get_value() >= t_push)
                                              for cc, vf in ((c1, v1), (c2, v2))]))
        with self.say("The first law is also called the law of inertia. Inertia is the "
                      "tendency of a body to resist any change in its motion. And mass is the measure of "
                      "inertia. Give an empty trolley and a heavily loaded one exactly the same short "
                      "push. The empty one shoots off, while the loaded one barely moves. The more mass "
                      "a body has, the harder it is to speed it up, to slow it down, or to turn it.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.at_sentence(c, 1)
            self.play(FadeIn(defn), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(meas), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeIn(VGroup(track1, track2, c1, c2, lab2)), run_time=0.8)
            self.add(pushes, vels)
            slowtag = txt("push shown at 0.3× speed", 20, MUTED).to_corner(DR, buff=0.3)
            self.play(FadeIn(slowtag), run_time=0.4)
            self.play(clock.animate.set_value(t_push), run_time=t_push / 0.3, rate_func=linear)
            self.play(clock.animate.set_value(t_push + 5.0), run_time=5.0, rate_func=linear)
        for m in (c1, c2, pushes, vels):
            m.clear_updaters()
        self.play(FadeOut(VGroup(head, defn, meas, track1, track2, c1, c2, lab2, pushes, vels, slowtag)),
                  run_time=0.6)

    # ------------------------------------------------------------------ coin, card and glass
    def coin_card(self):
        # Glass 10 cm tall drawn 1.9 units tall: 1 unit = 5.26 cm. Shown at 0.25x speed.
        unit_m = 0.10 / 1.9
        gl = glass(1.5, 1.9).move_to(v(0, -1.3))
        rim_y = gl.get_top()[1]
        card = Rectangle(width=2.2, height=0.06, fill_color="#F8FAFC", fill_opacity=1, stroke_width=0)
        card.move_to(v(0, rim_y + 0.03))
        coin = RoundedRectangle(width=0.55, height=0.1, corner_radius=0.04, fill_color="#FBBF24",
                                fill_opacity=1, stroke_width=0).move_to(v(0, rim_y + 0.06 + 0.05))
        bottom_y = gl.get_bottom()[1] + 0.05
        head = bold("Try it: the coin, the card and the glass", 34).move_to(v(0, 2.6))
        flick = force(card.get_left() + LEFT * 1.3, RIGHT * 1.1, C_FORCE, txt("flick", 24, C_FORCE), label_dir=UP)
        slow = txt("shown at 0.25× speed", 22, MUTED).to_corner(DR, buff=0.3)
        coin_y0 = coin.get_center()[1]
        fall = coin_y0 - (bottom_y + 0.05)
        t_fall = np.sqrt(2 * fall * unit_m / G)  # real seconds
        with self.say("Here's a demonstration you can try at home. Put a card on top of a glass, and a "
                      "coin on the card. Flick the card away sharply. The card flies off, but the coin's "
                      "inertia keeps it almost exactly where it was, and it drops straight into the glass.") as c:
            self.play(FadeIn(head), FadeIn(gl), FadeIn(card), FadeIn(coin), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(slow), GrowArrow(flick[0]), FadeIn(flick[1]), run_time=0.5)
            self.play(FadeOut(flick), card.animate.shift(RIGHT * 9.5), coin.animate.shift(RIGHT * 0.04),
                      run_time=0.6, rate_func=rate_functions.ease_in_quad)
            self.play(coin.animate(rate_func=rate_functions.ease_in_quad, run_time=t_fall / 0.25).shift(DOWN * fall))
        self.play(FadeOut(VGroup(head, gl, card, coin, slow)), run_time=0.6)

    # ------------------------------------------------------------------ bus ride
    def bus_ride(self):
        road_y = -2.3
        road = ground(-7.2, 7.2, road_y)
        b = bus(5.4, 2.1).move_to(v(-3.4, road_y + 0.3), aligned_edge=DOWN)
        b.shift(UP * (road_y - b[3].get_bottom()[1]))
        p = person(1.45).move_to(b.floor.get_center() + v(0.3, 0), aligned_edge=DOWN)
        rider = VGroup(b, p)
        head = bold("Inertia on a bus", 36).move_to(v(0, 2.6))
        note = txt("", 28)
        # kinematics of the bus: accelerate, cruise, brake (1 unit = 1 m)
        a1, t1, t2, a3 = 1.6, 1.4, 1.0, 2.4
        v1 = a1 * t1
        t3 = v1 / a3
        x1 = 0.5 * a1 * t1**2
        x2 = x1 + v1 * t2

        def pos(t):
            if t <= t1:
                return 0.5 * a1 * t * t
            if t <= t1 + t2:
                return x1 + v1 * (t - t1)
            tt = min(t - t1 - t2, t3)
            return x2 + v1 * tt - 0.5 * a3 * tt * tt

        def acc(t):
            return a1 if t < t1 else (0.0 if t < t1 + t2 else (-a3 if t < t1 + t2 + t3 else 0.0))

        clock = ValueTracker(0.0)
        base = b.get_center()
        lean = {"now": 0.0}

        def upd(m):
            t = clock.get_value()
            b.move_to(base + RIGHT * pos(t))
            # relative to the bus, the upper body lags behind the floor's acceleration
            target = 0.2 * acc(t)
            lean["now"] += (target - lean["now"]) * 0.35
            p.become(person(1.45).move_to(b.floor.get_center() + v(0.3, 0), aligned_edge=DOWN))
            p.rotate(lean["now"], about_point=p.get_bottom())

        labels = {
            "start": mtxt(f'bus speeds up: you lurch {hl("backward")}', 30),
            "stop": mtxt(f'bus brakes: you lurch {hl("forward")}', 30),
        }
        for l in labels.values():
            l.move_to(v(0, 1.6))
        with self.say("You feel your own inertia whenever you ride in a bus. When the bus starts "
                      "suddenly, the floor carries your feet forward, but your upper body tends to stay "
                      "where it was, so you lurch backward. When the bus brakes suddenly, your body "
                      "tends to keep moving forward. That is exactly why seat belts save lives.") as c:
            self.play(FadeIn(head), FadeIn(road), FadeIn(rider), run_time=0.8)
            self.at_sentence(c, 1)
            rider.add_updater(upd)
            slow = txt("shown at 0.5× speed", 22, MUTED).to_corner(DR, buff=0.3)
            self.play(FadeIn(labels["start"]), FadeIn(slow), run_time=0.4)
            self.play(clock.animate.set_value(t1), run_time=t1 / 0.5, rate_func=linear)
            self.play(clock.animate.set_value(t1 + t2), run_time=t2 / 0.5, rate_func=linear)
            self.at_sentence(c, 2)
            self.play(FadeOut(labels["start"]), FadeIn(labels["stop"]), run_time=0.4)
            self.play(clock.animate.set_value(t1 + t2 + t3 + 0.6), run_time=(t3 + 0.6) / 0.5, rate_func=linear)
        rider.clear_updaters()
        self.play(FadeOut(VGroup(head, road, rider, labels["stop"], slow)), run_time=0.6)

    # ------------------------------------------------------------------ inertial frames
    def frames(self):
        a = 1.4  # bus acceleration, m/s^2 (1 unit = 1 m)
        T = 2.4
        top_y, bot_y = 0.55, -2.75
        tops = []
        mobs = VGroup()
        clock = ValueTracker(0.0)
        views = []
        for y, title, col in ((top_y, "From the road  (inertial frame)", GOOD),
                              (bot_y, "From inside the bus  (accelerating frame)", BAD)):
            road = ground(-7.2, 7.2, y)
            b = bus(5.2, 1.7).move_to(v(-2.6, y), aligned_edge=DOWN)
            b.shift(UP * (y - b[3].get_bottom()[1]))
            bl = ball(0.2, "#F59E0B", stripe=False).move_to(v(-0.6, b.floor.get_y() + 0.2))
            ttl = bold(title, 26, col).move_to(v(0, y + 2.45)).to_edge(LEFT, buff=0.5)
            mobs.add(road, b, bl, ttl)
            views.append((b, bl))
        (b_road, ball_road), (b_bus, ball_bus) = views
        b0, bb0 = b_road.get_center(), ball_bus.get_center()
        b_road.add_updater(lambda m: m.move_to(b0 + RIGHT * 0.5 * a * clock.get_value() ** 2))
        ball_bus.add_updater(lambda m: m.move_to(bb0 + LEFT * 0.5 * a * clock.get_value() ** 2))
        acc_arrow = always_redraw(lambda: VGroup(
            arrow(b_road.shell.get_right() + v(0.25, 0.3), b_road.shell.get_right() + v(1.15, 0.3), C_ACC, stroke=5),
            tex(r"\vec a", size=30, color=C_ACC).next_to(b_road.shell.get_right() + v(0.7, 0.3), UP, buff=0.08)))
        still = txt("no force: the ball stays put", 26, GOOD).move_to(v(0, top_y + 2.45)).to_edge(RIGHT, buff=0.5)
        weird = txt("accelerates with no force?!", 26, BAD).move_to(v(0, bot_y + 2.45)).to_edge(RIGHT, buff=0.5)
        mobs.add(acc_arrow)
        with self.say("The bus teaches us something deeper. Suppose a ball rests on the floor of the bus, "
                      "which is so smooth that it exerts no horizontal force on the ball. Now the bus "
                      "speeds up. Seen from the roadside, nothing pushes the ball horizontally, so it "
                      "stays exactly where it was, just as the first law says. But seen from inside the "
                      "bus, the ball suddenly starts moving backward, although no force pushes it. For "
                      "someone riding in the bus, the first law seems to fail.") as c:
            self.play(FadeIn(mobs), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(clock.animate.set_value(T), run_time=T / 0.6, rate_func=linear)
            self.at_sentence(c, 3)
            self.play(FadeIn(still), run_time=0.5)
            self.at_sentence(c, 4)
            clock.set_value(0)
            self.wait(0.4)
            self.play(clock.animate.set_value(T), run_time=T / 0.6, rate_func=linear)
            self.play(FadeIn(weird), run_time=0.5)
        for m in (b_road, ball_bus, acc_arrow):
            m.clear_updaters()
        self.play(FadeOut(VGroup(mobs, still, weird)), run_time=0.6)

        head = bold("Inertial frames of reference", 40, ACCENT).move_to(v(0, 2.5))
        d1 = lines(f"{hl('Inertial frame', GOOD)}: a frame in which the first law holds.",
                   "The ground (very nearly), or anything moving at constant velocity relative to it.",
                   size=29, buff=0.15)
        d2 = lines(f"{hl('Non-inertial frame', BAD)}: an accelerating frame, like the bus speeding up.",
                   "Objects seem to accelerate with no force acting on them.", size=29, buff=0.15)
        d3 = mtxt(f"Newton's laws, in their usual form, hold only in {hl('inertial frames', GOOD)}.", 29)
        grp = VGroup(d1, d2, d3).arrange(DOWN, buff=0.5, aligned_edge=LEFT).next_to(head, DOWN, buff=0.55)
        if grp.width > 13.2:
            grp.scale_to_fit_width(13.2)
        with self.say("Frames of reference in which the first law holds are called inertial frames. The "
                      "ground is, to a very good approximation, an inertial frame, and so is any frame "
                      "moving at constant velocity relative to it. An accelerating bus is a non-inertial "
                      "frame. Newton's laws, in the form we use, hold only in inertial frames. So the "
                      "first law does more than describe what happens with zero force: it tells us "
                      "which frames the laws of motion work in.") as c:
            self.play(FadeIn(head), run_time=0.6)
            self.play(FadeIn(d1[0]), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(d1[1]), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(d2), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeIn(d3), run_time=0.8)
        self.play(FadeOut(VGroup(head, grp)), run_time=0.6)

    # ------------------------------------------------------------------ recap
    def recap(self):
        box = panel(11.5, 3.6).move_to(v(0, 0.2))
        ttl = bold("First law in one line", 32, ACCENT).next_to(box.get_top(), DOWN, buff=0.35)
        eq = tex(r"\vec F_{\text{net}} = \vec 0 \;\Longleftrightarrow\; \vec v = \text{constant}", size=50)
        eq.next_to(ttl, DOWN, buff=0.4)
        sub = txt("(in an inertial frame)   ·   mass measures inertia", 28, MUTED).next_to(eq, DOWN, buff=0.35)
        with self.say("So, the first law in one line: zero net force, if and only if the velocity is "
                      "constant, in an inertial frame. And the greater the mass, the greater the inertia."):
            self.play(FadeIn(box), FadeIn(ttl), run_time=0.6)
            self.play(Write(eq), run_time=1.4)
            self.play(FadeIn(sub), run_time=0.6)
        self.play(FadeOut(VGroup(box, ttl, eq, sub)), FadeOut(self.label), run_time=0.7)
