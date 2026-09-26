"""Reusable drawings: force arrows, blocks, carts, ground, figures, props."""
from __future__ import annotations

import numpy as np
from manim import (BOLD, DL, DOWN, DR, LEFT, ORIGIN, PI, RIGHT, UL, UP, UR, Arc, Arrow, Circle,
                   Dot, Ellipse, Line, Mobject, Polygon, Rectangle,
                   RoundedRectangle, SurroundingRectangle, VGroup, VMobject,
                   normalize)

from .style import (ACCENT, BAD, BG, C_FORCE, GOOD, GROUND, MUTED, OBJ_FILL,
                    OBJ_FILL_2, OBJ_STROKE, PANEL, PANEL_EDGE, TEXT, bold, tex,
                    txt)


def v(x: float, y: float = 0.0) -> np.ndarray:
    return np.array([x, y, 0.0])


# --------------------------------------------------------------------------- arrows
def arrow(start, end, color=C_FORCE, stroke: float = 6, tip: float = 0.24) -> VMobject:
    """Straight arrow with a constant stroke and tip size (tip shrinks only for tiny arrows).

    A zero-length arrow is returned as an invisible stub (never an empty mobject), so it can
    be used safely inside ``always_redraw``."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if np.linalg.norm(end - start) < 1e-3:
        return Arrow(start, start + RIGHT * 0.01, buff=0, color=color, stroke_width=stroke,
                     tip_length=tip, max_tip_length_to_length_ratio=0.45,
                     max_stroke_width_to_length_ratio=1000).set_opacity(0)
    return Arrow(start, end, buff=0, color=color, stroke_width=stroke, tip_length=tip,
                 max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=1000)


def shown(mob: Mobject, visible: bool) -> Mobject:
    """For always_redraw: keep the same structure, hide with zero opacity."""
    return mob if visible else mob.set_opacity(0)


def force(tail, vec, color=C_FORCE, label: str | None = None, label_dir=None,
          label_size: float = 34, buff: float = 0.12, stroke: float = 6) -> VGroup:
    """Force arrow from ``tail`` along ``vec`` with an optional label at the head.

    ``label`` may be a LaTeX string or a ready-made Mobject (e.g. a word label)."""
    tail = np.asarray(tail, float)
    vec = np.asarray(vec, float)
    head = tail + vec
    group = VGroup(arrow(tail, head, color, stroke))
    if label is not None:
        lab = label if isinstance(label, Mobject) else tex(label, size=label_size, color=color)
        direction = label_dir if label_dir is not None else normalize(vec)
        lab.next_to(head, direction, buff=buff)
        group.add(lab)
    return group


# --------------------------------------------------------------------------- bodies
def block(w: float = 1.2, h: float = 0.8, label=None, fill: str = OBJ_FILL,
          stroke: str = OBJ_STROKE, label_size: float = 30, radius: float = 0.07) -> VGroup:
    body = RoundedRectangle(width=w, height=h, corner_radius=radius, fill_color=fill,
                            fill_opacity=1, stroke_color=stroke, stroke_width=2.5)
    group = VGroup(body)
    if label is not None:
        lab = label if isinstance(label, Mobject) else txt(label, label_size)
        lab.move_to(body)
        group.add(lab)
    group.body = body
    return group


def cart(w: float = 1.6, h: float = 0.55, label=None, fill: str = OBJ_FILL_2,
         wheel_r: float = 0.16, label_size: float = 28) -> VGroup:
    """Laboratory dynamics cart; its lowest point is the bottom of the wheels."""
    body = RoundedRectangle(width=w, height=h, corner_radius=0.08, fill_color=fill,
                            fill_opacity=1, stroke_color=OBJ_STROKE, stroke_width=2.5)
    wheels = VGroup()
    for dx in (-0.3 * w, 0.3 * w):
        c = body.get_bottom() + v(dx, -0.9 * wheel_r)
        wheel = Circle(radius=wheel_r, fill_color="#0B1220", fill_opacity=1,
                       stroke_color=OBJ_STROKE, stroke_width=2.5).move_to(c)
        wheels.add(VGroup(wheel, Dot(c, radius=0.035, color=OBJ_STROKE)))
    group = VGroup(wheels, body)
    if label is not None:
        lab = label if isinstance(label, Mobject) else txt(label, label_size)
        lab.move_to(body)
        group.add(lab)
    group.body = body
    group.wheels = wheels
    return group


def ground(x0: float, x1: float, y: float = 0.0, color: str = GROUND, hatch: bool = True,
           stroke: float = 3) -> VGroup:
    line = Line(v(x0, y), v(x1, y), color=color, stroke_width=stroke)
    group = VGroup(line)
    if hatch:
        marks = VGroup(*[Line(v(x, y), v(x - 0.16, y - 0.16), color=color, stroke_width=2)
                         for x in np.arange(x0 + 0.2, x1 + 1e-6, 0.3)])
        group.add(marks)
    return group


def surface(x0: float, x1: float, y: float, color: str, depth: float = 0.22,
            opacity: float = 0.35) -> VGroup:
    """A coloured slab whose top face is at height ``y`` (table, floor, ice...)."""
    slab = Rectangle(width=x1 - x0, height=depth, stroke_width=0, fill_color=color,
                     fill_opacity=opacity).move_to(v((x0 + x1) / 2, y - depth / 2))
    top = Line(v(x0, y), v(x1, y), color=color, stroke_width=3)
    return VGroup(slab, top)


def ball(radius: float = 0.25, color: str = "#F59E0B", stripe: bool = True) -> VGroup:
    """A ball with a stripe so that rolling (rotation) is visible."""
    disc = Circle(radius=radius, fill_color=color, fill_opacity=1, stroke_color=TEXT,
                  stroke_width=2)
    group = VGroup(disc)
    if stripe:
        group.add(Line(v(-radius * 0.95, 0), v(radius * 0.95, 0), color=BG, stroke_width=3))
    return group


def person(h: float = 1.8, color: str = TEXT, arms: str = "down", stroke: float = 6) -> VGroup:
    """Stick figure with feet at y = 0 (relative), centred on x = 0.

    arms: 'down', 'push_right', 'push_left', 'up', 'none' (draw arms yourself; shoulder at 0.70 h)
    """
    head_r = 0.1 * h
    hip, neck = v(0, 0.46 * h), v(0, 0.76 * h)
    shoulder = v(0, 0.70 * h)
    head = Circle(radius=head_r, stroke_color=color, stroke_width=stroke,
                  fill_color=BG, fill_opacity=1).move_to(neck + v(0, head_r * 1.08))
    parts = [Line(neck, hip), Line(hip, v(-0.13 * h, 0)), Line(hip, v(0.13 * h, 0))]
    if arms == "down":
        parts += [Line(shoulder, v(-0.17 * h, 0.44 * h)), Line(shoulder, v(0.17 * h, 0.44 * h))]
    elif arms == "push_right":
        parts += [Line(shoulder, v(0.33 * h, 0.70 * h)), Line(shoulder, v(0.31 * h, 0.64 * h))]
    elif arms == "push_left":
        parts += [Line(shoulder, v(-0.33 * h, 0.70 * h)), Line(shoulder, v(-0.31 * h, 0.64 * h))]
    elif arms == "up":
        parts += [Line(shoulder, v(-0.2 * h, 0.98 * h)), Line(shoulder, v(0.2 * h, 0.98 * h))]
    elif arms == "stride":  # walking: arms swinging, legs apart (feet at +-0.2 h)
        parts = [Line(neck, hip), Line(hip, v(-0.2 * h, 0)), Line(hip, v(0.2 * h, 0)),
                 Line(shoulder, v(-0.16 * h, 0.47 * h)), Line(shoulder, v(0.16 * h, 0.47 * h))]
    body = VGroup(*parts).set_stroke(color, stroke)
    return VGroup(body, head)


def rocket(height: float = 2.0, flame: bool = True) -> VGroup:
    s = height / 2.0
    body = RoundedRectangle(width=0.55 * s, height=1.3 * s, corner_radius=0.08 * s,
                            fill_color="#E2E8F0", fill_opacity=1, stroke_color=OBJ_STROKE,
                            stroke_width=2)
    nose = Polygon(v(-0.275 * s, 0.65 * s), v(0.275 * s, 0.65 * s), v(0, 1.15 * s),
                   fill_color="#F87171", fill_opacity=1, stroke_width=0)
    fin_l = Polygon(v(-0.275 * s, -0.25 * s), v(-0.55 * s, -0.7 * s), v(-0.275 * s, -0.65 * s),
                    fill_color="#F87171", fill_opacity=1, stroke_width=0)
    fin_r = fin_l.copy().flip(UP).shift(v(0.55 * s, 0))
    window = Circle(radius=0.12 * s, fill_color="#38BDF8", fill_opacity=1,
                    stroke_color="#0F172A", stroke_width=2).move_to(v(0, 0.25 * s))
    group = VGroup(body, nose, fin_l, fin_r, window)
    if flame:
        outer = Polygon(v(-0.2 * s, -0.65 * s), v(0.2 * s, -0.65 * s), v(0, -1.35 * s),
                        fill_color="#FB923C", fill_opacity=1, stroke_width=0)
        inner = Polygon(v(-0.1 * s, -0.65 * s), v(0.1 * s, -0.65 * s), v(0, -1.05 * s),
                        fill_color="#FDE047", fill_opacity=1, stroke_width=0)
        group.add_to_back(VGroup(outer, inner))
        group.flame = VGroup(outer, inner)
    return group


def apple(r: float = 0.22) -> VGroup:
    fruit = Circle(radius=r, fill_color="#EF4444", fill_opacity=1, stroke_width=0)
    stem = Line(v(0, r * 0.8), v(0.05, r * 1.35), color="#92400E", stroke_width=4)
    leaf = Ellipse(width=r * 0.9, height=r * 0.4, fill_color="#22C55E", fill_opacity=1,
                   stroke_width=0).rotate(PI / 6).move_to(v(r * 0.45, r * 1.2))
    return VGroup(fruit, stem, leaf)


def earth_cap(center_x: float = 0.0, top_y: float = -3.0, radius: float = 14.0) -> VGroup:
    """The top of a very large Earth, whose surface peaks at ``top_y``."""
    disc = Circle(radius=radius, fill_color="#1E3A8A", fill_opacity=1, stroke_color="#60A5FA",
                  stroke_width=3).move_to(v(center_x, top_y - radius))
    return VGroup(disc)


def table(w: float = 4.0, top_y: float = -1.0, x: float = 0.0, leg_h: float = 1.6) -> VGroup:
    top = Rectangle(width=w, height=0.18, fill_color="#92400E", fill_opacity=1,
                    stroke_color="#D97706", stroke_width=2).move_to(v(x, top_y - 0.09))
    legs = VGroup(*[Rectangle(width=0.16, height=leg_h, fill_color="#78350F", fill_opacity=1,
                              stroke_width=0).move_to(v(x + dx, top_y - 0.18 - leg_h / 2))
                    for dx in (-w / 2 + 0.3, w / 2 - 0.3)])
    group = VGroup(legs, top)
    group.top = top
    return group


def book(w: float = 1.5, h: float = 0.36, color: str = "#7C3AED") -> VGroup:
    cover = RoundedRectangle(width=w, height=h, corner_radius=0.04, fill_color=color,
                             fill_opacity=1, stroke_color="#C4B5FD", stroke_width=2)
    pages = Rectangle(width=w * 0.92, height=h * 0.5, fill_color="#F8FAFC", fill_opacity=1,
                      stroke_width=0).move_to(cover).shift(v(w * 0.03, 0))
    return VGroup(cover, pages)


def horse(color: str = "#B45309", scale: float = 1.0) -> VGroup:
    """Side view of a (cartoon) horse facing right; hooves at y = 0 relative."""
    s = scale
    body = RoundedRectangle(width=1.9 * s, height=0.72 * s, corner_radius=0.3 * s,
                            fill_color=color, fill_opacity=1, stroke_width=0).move_to(v(0, 1.25 * s))
    neck = Polygon(v(0.55 * s, 1.5 * s), v(0.95 * s, 1.55 * s), v(1.25 * s, 2.25 * s),
                   v(0.95 * s, 2.35 * s), fill_color=color, fill_opacity=1, stroke_width=0)
    head = RoundedRectangle(width=0.78 * s, height=0.34 * s, corner_radius=0.14 * s,
                            fill_color=color, fill_opacity=1, stroke_width=0)
    head.rotate(-0.55).move_to(v(1.38 * s, 2.12 * s))
    ear = Polygon(v(1.02 * s, 2.36 * s), v(1.12 * s, 2.62 * s), v(1.2 * s, 2.36 * s),
                  fill_color=color, fill_opacity=1, stroke_width=0)
    legs = VGroup(*[Line(v(x * s, 1.0 * s), v(x * s, 0), color=color, stroke_width=12)
                    for x in (-0.75, -0.5, 0.5, 0.75)])
    tail = Arc(radius=0.45 * s, start_angle=PI * 0.55, angle=PI * 0.5, color="#78350F",
               stroke_width=9).move_to(v(-1.1 * s, 1.15 * s))
    eye = Dot(v(1.3 * s, 2.2 * s), radius=0.035 * s, color=BG)
    mane = Line(v(0.72 * s, 1.6 * s), v(1.0 * s, 2.3 * s), color="#78350F", stroke_width=8)
    return VGroup(tail, legs, body, neck, mane, head, ear, eye)


def lift_car(w: float = 2.6, h: float = 3.6) -> VGroup:
    frame = Rectangle(width=w, height=h, stroke_color=OBJ_STROKE, stroke_width=4,
                      fill_color="#111827", fill_opacity=1)
    cable = Line(frame.get_top(), frame.get_top() + v(0, 4.5), color=MUTED, stroke_width=3)
    group = VGroup(cable, frame)
    group.frame = frame
    return group


def bathroom_scale(w: float = 1.1, h: float = 0.22) -> VGroup:
    base = RoundedRectangle(width=w, height=h, corner_radius=0.06, fill_color="#CBD5E1",
                            fill_opacity=1, stroke_width=0)
    window = RoundedRectangle(width=w * 0.35, height=h * 0.55, corner_radius=0.03,
                              fill_color="#0F172A", fill_opacity=1, stroke_width=0).move_to(base)
    return VGroup(base, window)


def pulley(radius: float = 0.45, support_len: float = 0.8) -> VGroup:
    wheel = Circle(radius=radius, stroke_color=OBJ_STROKE, stroke_width=4,
                   fill_color=OBJ_FILL, fill_opacity=1)
    axle = Dot(radius=0.06, color=OBJ_STROKE)
    support = Line(ORIGIN, v(0, radius + support_len), color=OBJ_STROKE, stroke_width=4)
    group = VGroup(support, wheel, axle)
    group.wheel = wheel
    return group


# --------------------------------------------------------------------------- UI bits
def panel(w: float, h: float, fill: str = PANEL, stroke: str = PANEL_EDGE,
          radius: float = 0.18, opacity: float = 1.0) -> RoundedRectangle:
    return RoundedRectangle(width=w, height=h, corner_radius=radius, fill_color=fill,
                            fill_opacity=opacity, stroke_color=stroke, stroke_width=2)


def boxed(mob: Mobject, color: str = ACCENT, buff: float = 0.22) -> SurroundingRectangle:
    return SurroundingRectangle(mob, color=color, buff=buff, corner_radius=0.12, stroke_width=3)


def check_mark(size: float = 0.5, color: str = GOOD) -> VMobject:
    return VMobject(stroke_color=color, stroke_width=9).set_points_as_corners(
        [v(-0.5 * size, 0), v(-0.15 * size, -0.38 * size), v(0.55 * size, 0.45 * size)])


def cross_mark(size: float = 0.5, color: str = BAD) -> VGroup:
    return VGroup(Line(v(-size / 2, -size / 2), v(size / 2, size / 2)),
                  Line(v(-size / 2, size / 2), v(size / 2, -size / 2))).set_stroke(color, 9)


def pause_badge(text: str = "Pause and think") -> VGroup:
    bars = VGroup(*[Rectangle(width=0.12, height=0.42, fill_color=ACCENT, fill_opacity=1,
                              stroke_width=0) for _ in range(2)]).arrange(RIGHT, buff=0.1)
    label = bold(text, 30, ACCENT)
    content = VGroup(bars, label).arrange(RIGHT, buff=0.25)
    frame = SurroundingRectangle(content, color=ACCENT, buff=0.22, corner_radius=0.15,
                                 stroke_width=3)
    return VGroup(frame, content)


def dimension_label(p0, p1, text: str, color: str = MUTED, offset=DOWN * 0.35,
                    size: float = 26) -> VGroup:
    p0, p1 = np.asarray(p0, float) + offset, np.asarray(p1, float) + offset
    line = Line(p0, p1, color=color, stroke_width=2)
    ticks = VGroup(*[Line(p + UP * 0.1, p + DOWN * 0.1, color=color, stroke_width=2) for p in (p0, p1)])
    lab = txt(text, size, color).next_to(line, normalize(offset), buff=0.1)
    return VGroup(line, ticks, lab)


def car_side(w: float = 2.6, color: str = "#2563EB") -> VGroup:
    """Side view of a car facing right; lowest point (tyres) at the group's bottom."""
    h = 0.42 * w
    body = RoundedRectangle(width=w, height=0.36 * h * 1.2, corner_radius=0.08 * w,
                            fill_color=color, fill_opacity=1, stroke_width=0)
    cabin = Polygon(v(-0.28 * w, 0), v(0.18 * w, 0), v(0.08 * w, 0.2 * w), v(-0.2 * w, 0.2 * w),
                    fill_color=color, fill_opacity=1, stroke_width=0)
    cabin.next_to(body, UP, buff=-0.02).shift(LEFT * 0.04 * w)
    win = Polygon(v(-0.24 * w, 0.02 * w), v(0.14 * w, 0.02 * w), v(0.06 * w, 0.17 * w),
                  v(-0.18 * w, 0.17 * w), fill_color="#BFDBFE", fill_opacity=0.9, stroke_width=0)
    win.move_to(cabin).shift(DOWN * 0.005 * w)
    wheels = VGroup()
    for dx in (-0.3 * w, 0.3 * w):
        c = body.get_bottom() + v(dx, 0)
        wheels.add(Circle(radius=0.11 * w, fill_color="#0B1220", fill_opacity=1,
                          stroke_color=OBJ_STROKE, stroke_width=2.5).move_to(c))
    group = VGroup(body, cabin, win, wheels)
    group.body = body
    return group


def bus(w: float = 5.4, h: float = 1.9, color: str = "#EAB308") -> VGroup:
    """Side view of a bus facing right. ``bus.floor_y`` gives the y of the inside floor."""
    shell = RoundedRectangle(width=w, height=h, corner_radius=0.18, fill_color=color,
                             fill_opacity=0.18, stroke_color=color, stroke_width=4)
    floor = Line(shell.get_corner(DL) + v(0.1, 0.28), shell.get_corner(DR) + v(-0.1, 0.28),
                 color=color, stroke_width=4)
    wins = VGroup(*[RoundedRectangle(width=0.7, height=0.5, corner_radius=0.06,
                                     stroke_color=color, stroke_width=2, fill_opacity=0)
                    .move_to(shell.get_top() + v(-w / 2 + 0.75 + i * 0.95, -0.45))
                    for i in range(int((w - 0.8) / 0.95))])
    wheels = VGroup()
    for dx in (-0.32 * w, 0.32 * w):
        c = shell.get_bottom() + v(dx, 0)
        wheels.add(Circle(radius=0.3, fill_color="#0B1220", fill_opacity=1,
                          stroke_color=OBJ_STROKE, stroke_width=3).move_to(c))
    group = VGroup(shell, floor, wins, wheels)
    group.shell = shell
    group.floor = floor
    return group


def glass(w: float = 1.2, h: float = 1.6) -> VGroup:
    """Drinking glass (open top); ``glass.rim_y`` is the y of the rim."""
    outline = VMobject(stroke_color="#BAE6FD", stroke_width=4).set_points_as_corners(
        [v(-w / 2, h / 2), v(-0.4 * w, -h / 2), v(0.4 * w, -h / 2), v(w / 2, h / 2)])
    fill = Polygon(v(-w / 2, h / 2), v(-0.4 * w, -h / 2), v(0.4 * w, -h / 2), v(w / 2, h / 2),
                   fill_color="#BAE6FD", fill_opacity=0.08, stroke_width=0)
    return VGroup(fill, outline)
