"""Part 1 -- What keeps things moving?  Aristotle, friction, Galileo's ramps."""
import numpy as np
from manim import *

from lesson.common import open_chapter
from lesson.narration import VoiceScene
from lesson.objects import arrow, ball, block, book, force, shown, surface, v
from lesson.style import *

G = 9.8  # m/s^2; in this scene 1 scene unit = 1 metre unless noted
ROLL = 5 / 7  # a = (5/7) g sin(theta) for a solid sphere rolling without slipping


class S01_Galileo(VoiceScene):
    def construct(self):
        self.label = open_chapter(self, "1", "What keeps things moving?")
        self.aristotle()
        self.friction_tracks()
        self.ramps()
        self.conclusion()

    # ------------------------------------------------------------------ Aristotle
    def aristotle(self):
        floor_y = -1.3
        table = surface(-7.2, 7.2, floor_y, "#B45309", depth=0.35, opacity=0.5)
        who = bold("Aristotle", 40).move_to(v(0, 2.55))
        when = txt("Greek philosopher, 4th century BCE", 26, MUTED).next_to(who, DOWN, buff=0.15)
        claim = mtxt('“A moving object needs a <b>continuous push</b> to keep moving.”', 32)
        claim.next_to(when, DOWN, buff=0.35)

        bk = book(1.8, 0.5)
        # The book enters already sliding at v0, pushed so that push = friction (steady speed);
        # the push then stops and kinetic friction (mu = 0.25) brings it to rest.
        start = v(-8.4, floor_y + 0.25)
        v0, t_push, mu = 2.0, 3.0, 0.25
        dec = mu * G
        t_stop = v0 / dec

        def x_of(t):
            if t <= t_push:
                return v0 * t
            tt = min(t - t_push, t_stop)
            return v0 * t_push + v0 * tt - 0.5 * dec * tt**2

        T_END = t_push + t_stop + 0.4
        clock = ValueTracker(0.0)
        origin = [start]  # the replay starts nearer the centre so every label stays in view
        bk.move_to(start)
        bk.add_updater(lambda m: m.move_to(origin[0] + RIGHT * x_of(clock.get_value())))
        push = always_redraw(lambda: shown(
            force(bk.get_corner(UL) + v(-1.25, -0.13), RIGHT * 1.25, C_FORCE,
                  txt("push", 26, C_FORCE), label_dir=UP, buff=0.15),
            clock.get_value() < t_push))

        with self.say("For nearly two thousand years, most people accepted the ideas of the Greek "
                      "philosopher Aristotle. He taught that a moving object needs a continuous push "
                      "to keep it moving. Stop pushing, and it stops.") as c:
            self.play(FadeIn(who), FadeIn(when), FadeIn(table), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(Write(claim), run_time=1.6)
            self.add(bk, push)
            self.play(clock.animate.set_value(T_END), run_time=T_END, rate_func=linear)

        galileo = bold("Galileo Galilei", 40).move_to(who)
        g_when = txt("Italian scientist, early 1600s", 26, MUTED).next_to(galileo, DOWN, buff=0.15)
        insight = mtxt(f'The book stops because a force acts on it: {hl("friction", C_FRICTION)}.', 32)
        insight.move_to(claim)
        fric = always_redraw(lambda: shown(
            force(bk.get_bottom(), LEFT * 1.25, C_FRICTION, txt("friction", 26, C_FRICTION),
                  label_dir=LEFT, buff=0.15),
            clock.get_value() < t_push + t_stop - 1e-3))

        with self.say("That seems sensible, because it matches everyday experience. But in the early "
                      "sixteen-hundreds, Galileo Galilei noticed what Aristotle had missed. The book does "
                      "not stop by itself. It stops because a force acts on it: friction from the table, "
                      "which opposes its motion.") as c:
            self.at_sentence(c, 1)
            self.play(FadeOut(VGroup(who, when), shift=UP * 0.2),
                      FadeIn(VGroup(galileo, g_when), shift=UP * 0.2), run_time=0.8)
            self.at_sentence(c, 3)
            self.play(FadeOut(claim), FadeIn(insight), run_time=0.8)

        slow = txt("replay at 0.6× speed", 22, MUTED).next_to(table, DOWN, buff=0.3)
        with self.say("Watch again. While the push and the friction are equal in size, the book slides "
                      "at a steady speed. Once the push stops, friction is the only horizontal force "
                      "left, and it slows the book down.") as c:
            clock.set_value(0)
            origin[0] = v(-3.6, floor_y + 0.25)
            self.add(fric)
            self.play(FadeIn(slow), run_time=0.5)
            self.play(clock.animate.set_value(t_push), run_time=t_push / 0.6, rate_func=linear)
            self.at_sentence(c, 2)
            self.play(clock.animate.set_value(T_END), run_time=(T_END - t_push) / 0.6, rate_func=linear)
        bk.clear_updaters()
        push.clear_updaters()
        fric.clear_updaters()
        self.play(FadeOut(VGroup(bk, push, fric, table, galileo, g_when, insight, slow)), run_time=0.6)

    # ------------------------------------------------------------------ three tracks
    def friction_tracks(self):
        scale = 1.2  # scene units per metre
        v0 = 2.0
        x_start, x_end = -5.3, 6.6
        rows = [("rough wood", "strong friction", 0.30, "#B45309", 1.35),
                ("polished floor", "weaker friction", 0.10, "#94A3B8", -0.35),
                ("ice", "very weak friction", 0.02, "#7DD3FC", -2.05)]
        clock = ValueTracker(0.0)
        mobs = VGroup()
        movers = []
        for name, desc, mu, col, y in rows:
            srf = surface(x_start - 1.4, x_end, y, col, depth=0.2, opacity=0.45)
            lab = VGroup(bold(name, 26, col), txt(desc, 22, MUTED)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
            lab.next_to(srf, DOWN, buff=0.12).align_to(srf, LEFT)
            blk = block(0.62, 0.42, fill=OBJ_FILL_2)
            base = v(x_start + 0.31, y + 0.21)
            blk.move_to(base)
            dec = mu * G

            def x_of(t, dec=dec):
                t = min(t, v0 / dec)
                return (v0 * t - 0.5 * dec * t * t) * scale

            def vel(t, dec=dec):
                return max(v0 - dec * t, 0.0)

            blk.add_updater(lambda m, base=base, x_of=x_of: m.move_to(base + RIGHT * x_of(clock.get_value())))
            varrow = always_redraw(lambda blk=blk, vel=vel: arrow(
                blk.get_top() + UP * 0.2, blk.get_top() + UP * 0.2 + RIGHT * 0.55 * vel(clock.get_value()),
                C_VEL, stroke=5, tip=0.18))
            movers += [blk, varrow]
            mobs.add(srf, lab, blk, varrow)
        key = VGroup(arrow(ORIGIN, RIGHT * 0.6, C_VEL, stroke=5, tip=0.18), txt("velocity", 24, C_VEL))
        key.arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.4).shift(DOWN * 0.3)

        with self.say("The smoother the surface, the weaker the friction, and the farther an object "
                      "slides. Here, three identical blocks start with the same speed: on rough wood, "
                      "on a polished floor, and on ice.") as c:
            self.play(FadeIn(mobs), FadeIn(key), run_time=1.0)
            self.at_sentence(c, 1, 1.0)
            self.play(clock.animate.set_value(6.0), run_time=6.0, rate_func=linear)
        self.narrate("On ice, the friction is so weak that the block is still sliding, long after the "
                     "other two have stopped. So what would happen with no friction at all?")
        for m in movers:
            m.clear_updaters()
        self.play(FadeOut(mobs), FadeOut(key), run_time=0.6)

    # ------------------------------------------------------------------ Galileo's ramps
    def ramps(self):
        y0, xb, h, r = -2.6, -3.3, 2.0, 0.22
        th1 = np.radians(35)
        d1 = v(np.cos(th1), -np.sin(th1))
        n1 = v(np.sin(th1), np.cos(th1))
        B = v(xb, y0)
        left_top = B - d1 * ((h + 0.45) / np.sin(th1))
        contact0 = B - d1 * (h / np.sin(th1))
        C0 = contact0 + r * n1  # ball centre at release
        FAR = 7.6       # right end of the sloping ramps
        X_END = 34.0    # the flat track is long; the camera follows the ball along it
        SLOW = 0.6

        def right_end(th2):
            if th2 == 0:
                return v(X_END, y0)
            L = min((FAR - xb) / np.cos(th2), (h + 0.8) / np.sin(th2))
            return B + L * v(np.cos(th2), np.sin(th2))

        def track_shape(th2):
            e = right_end(th2)
            pts = [left_top, B, e, v(e[0], y0 - 0.35), v(left_top[0], y0 - 0.35)]
            poly = Polygon(*pts, fill_color="#1E293B", fill_opacity=1, stroke_width=0)
            top = VMobject(stroke_color=GROUND, stroke_width=5).set_points_as_corners([left_top, B, e])
            return VGroup(poly, top)

        def path(th2, x_stop=None):
            """Kinematics of the ball centre for second-ramp angle th2 (radians)."""
            d2 = v(np.cos(th2), np.sin(th2))
            n2 = v(-np.sin(th2), np.cos(th2))
            # intersection of the two offset lines (the ball touches both ramps there)
            A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
            s1, _ = np.linalg.solve(A, (B + r * n2 - C0)[:2])
            Q = C0 + s1 * d1
            a1 = ROLL * G * np.sin(th1)
            t1 = np.sqrt(2 * s1 / a1)
            vq = a1 * t1
            a2 = ROLL * G * np.sin(th2)
            t2 = vq / a2 if a2 > 0 else (x_stop - Q[0]) / vq

            def pos(t):
                if t <= t1:
                    s = 0.5 * a1 * t * t
                    return C0 + s * d1, s
                tt = min(t - t1, t2)
                u = vq * tt - 0.5 * a2 * tt * tt
                return Q + u * d2, s1 + u

            return pos, t1 + t2, t1, vq, Q

        track = track_shape(np.radians(35))
        bl = ball(r, "#F59E0B").move_to(C0)
        bl.angle = 0.0
        dashed = DashedLine(v(-7.1, C0[1]), v(X_END, C0[1]), color=ACCENT, stroke_width=2, dash_length=0.12)
        dlab = txt("starting height", 24, ACCENT).next_to(v(6.8, C0[1]), UP, buff=0.08, aligned_edge=RIGHT)
        heading = bold("Galileo's thought experiment", 38).move_to(v(0, 2.9))
        slow_tag = txt(f"shown at {SLOW}× speed", 22, MUTED).to_corner(DR, buff=0.3)

        def run(th2_deg, x_stop=None):
            pos, T, _, _, _ = path(np.radians(th2_deg), x_stop)
            clock = ValueTracker(0.0)

            def upd(m):
                p, dist = pos(clock.get_value())
                ang = -dist / r
                m.rotate(ang - m.angle)
                m.angle = ang
                m.move_to(p)

            bl.add_updater(upd)
            self.play(clock.animate(run_time=T / SLOW, rate_func=linear).set_value(T))
            bl.remove_updater(upd)

        def reset_ball():
            self.play(FadeOut(bl), run_time=0.3)
            bl.rotate(-bl.angle)
            bl.angle = 0.0
            bl.move_to(C0)
            self.play(FadeIn(bl), run_time=0.3)

        with self.say("Galileo answered with a brilliant thought experiment. Let a ball roll down a ramp, "
                      "and up a second ramp facing it. If no energy is lost along the way, the ball "
                      "climbs back to exactly the height it started from.") as c:
            self.play(FadeIn(heading), FadeIn(track), FadeIn(bl), FadeIn(slow_tag), run_time=1.0)
            self.at_sentence(c, 1)
            self.play(Create(dashed), FadeIn(dlab), run_time=0.8)
            run(35)

        new_track = {deg: track_shape(np.radians(deg)) for deg in (20, 12, 0)}
        with self.say("Now make the second ramp less steep. To reach the same height, the ball has to "
                      "roll farther."):
            reset_ball()
            self.play(Transform(track, new_track[20]), run_time=0.8)
            run(20)
        with self.say("Make it gentler still, and the ball rolls farther again."):
            reset_ball()
            self.play(Transform(track, new_track[12]), run_time=0.8)
            run(12)

        # ---- the flat track: the camera follows the ball; ghost images every 0.5 s (real time)
        x_stop = X_END - 4.0
        _, _, t1, vq, Q = path(0.0, x_stop)
        ghosts = VGroup(*[ball(r, "#F59E0B").move_to(Q + RIGHT * vq * 0.5 * k).set_opacity(0)
                          for k in range(0, int((x_stop - Q[0]) / (vq * 0.5)) + 1)])
        ticks = VGroup(*[Line(g.get_center() + DOWN * (r + 0.02), g.get_center() + DOWN * (r + 0.3),
                              color=MUTED, stroke_width=2).set_opacity(0) for g in ghosts])
        vel = always_redraw(lambda: shown(VGroup(
            arrow(bl.get_center() + UP * 0.5, bl.get_center() + UP * 0.5 + RIGHT * 1.2, C_VEL, stroke=5),
            tex(r"\vec v = \text{constant}", size=30, color=C_VEL).next_to(
                bl.get_center() + UP * 0.5 + RIGHT * 0.6, UP, buff=0.12)),
            bl.get_center()[1] < y0 + r + 0.05))
        note = VGroup(txt("equal distances in equal times", 28, C_VEL),
                      txt("(ghost images every 0.5 s)", 22, MUTED)).arrange(DOWN, buff=0.1)
        note.move_to(v(0, 1.3)).set_opacity(0)

        frame = self.camera.frame
        fixed = [self.label, slow_tag, heading, note]
        offsets = {id(m): m.get_center() - frame.get_center() for m in fixed}

        def follow(fr):
            x = max(0.0, bl.get_center()[0] - 1.5)
            fr.set_x(x)
            for m in fixed:
                m.move_to(fr.get_center() + offsets[id(m)])
            for g, tk in zip(ghosts, ticks):
                if bl.get_center()[0] >= g.get_center()[0] - 1e-6 and bl.get_center()[1] < y0 + r + 0.05:
                    g.set_opacity(0.3)
                    g[1].set_opacity(0.3)
                    tk.set_opacity(1)
            if sum(g.get_fill_opacity() > 0 for g in ghosts) >= 3:
                note.set_opacity(1)

        with self.say("And if the second ramp is perfectly flat? The ball can never get back to its "
                      "starting height, so it never stops. It keeps rolling in a straight line, at "
                      "constant speed, forever.") as c:
            reset_ball()
            self.play(Transform(track, new_track[0]), run_time=0.8)
            self.add(ghosts, ticks, vel, note)
            self.bring_to_front(bl, vel)
            frame.add_updater(follow)
            self.add(frame)
            run(0, x_stop)
            frame.remove_updater(follow)
        self.play(FadeOut(VGroup(track, bl, vel, ghosts, ticks, dashed, dlab, heading, slow_tag, note)),
                  run_time=0.6)
        vel.clear_updaters()
        self.remove(vel)
        frame.move_to(ORIGIN)
        self.label.move_to(frame.get_center() + offsets[id(self.label)])

    # ------------------------------------------------------------------ conclusion
    def conclusion(self):
        head = bold("Galileo's great insight", 40, ACCENT).move_to(v(0, 2.4))
        l1 = mtxt("Motion does <b>not</b> need a force.", 46)
        l2 = mtxt(f'A {hl("change")} in motion does.', 46)
        l3 = txt("speed up  ·  slow down  ·  change direction", 30, MUTED)
        grp = VGroup(l1, l2, l3).arrange(DOWN, buff=0.45).move_to(v(0, 0.0))
        with self.say("This was Galileo's great insight. An object does not need a force to keep moving. "
                      "A force is needed only to change its motion: to speed it up, to slow it down, or "
                      "to change its direction. Newton made this idea the first of his three laws.") as c:
            self.play(FadeIn(head, shift=DOWN * 0.2), run_time=0.8)
            self.at_sentence(c, 1)
            self.play(FadeIn(l1, shift=UP * 0.2), run_time=0.8)
            self.at_sentence(c, 2)
            self.play(FadeIn(l2, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(l3), run_time=0.8)
        self.play(FadeOut(grp), FadeOut(head), FadeOut(self.label), run_time=0.7)
