"""Colours, fonts and text helpers shared by every scene.

Colour code (kept identical in every diagram of the video):
  weight = red, normal = blue, tension = yellow, friction = pink,
  applied force = orange, velocity = cyan, acceleration = green, momentum = purple.
"""
from manim import (BOLD, DOWN, LEFT, UL, MarkupText, MathTex, Tex, Text,
                   VGroup, config)

BG = "#0F172A"
PANEL = "#1E293B"
PANEL_EDGE = "#334155"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
DIM = "#475569"
ACCENT = "#FBBF24"
GOOD = "#4ADE80"
BAD = "#F87171"

C_FORCE = "#FB923C"
C_WEIGHT = "#F87171"
C_NORMAL = "#60A5FA"
C_TENSION = "#FDE047"
C_FRICTION = "#F472B6"
C_VEL = "#22D3EE"
C_ACC = "#4ADE80"
C_MOM = "#C084FC"

OBJ_FILL = "#334155"
OBJ_FILL_2 = "#475569"
OBJ_STROKE = "#CBD5E1"
GROUND = "#64748B"

FONT = "Lato"

config.background_color = BG


def txt(s: str, size: float = 32, color: str = TEXT, weight=None, **kw) -> Text:
    if weight is not None:
        kw["weight"] = weight
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def bold(s: str, size: float = 32, color: str = TEXT, **kw) -> Text:
    return txt(s, size, color, weight=BOLD, **kw)


def mtxt(markup: str, size: float = 32, color: str = TEXT, **kw) -> MarkupText:
    """Pango markup text, e.g. 'a <span foreground="#FBBF24">net</span> force'."""
    return MarkupText(markup, font=FONT, font_size=size, color=color, **kw)


def hl(s: str, color: str = ACCENT, b: bool = True) -> str:
    """Markup helper: highlighted (and optionally bold) span."""
    inner = f"<b>{s}</b>" if b else s
    return f'<span foreground="{color}">{inner}</span>'


def tex(*s: str, size: float = 40, color: str = TEXT, **kw) -> MathTex:
    return MathTex(*s, font_size=size, color=color, **kw)


def textex(*s: str, size: float = 36, color: str = TEXT, **kw) -> Tex:
    return Tex(*s, font_size=size, color=color, **kw)


def lines(*rows: str, size: float = 30, color: str = TEXT, buff: float = 0.18,
          markup: bool = True, align=LEFT) -> VGroup:
    """Stack several text rows (markup allowed) as a left-aligned paragraph."""
    make = mtxt if markup else txt
    group = VGroup(*[make(r, size, color) for r in rows])
    group.arrange(DOWN, buff=buff, aligned_edge=align)
    return group


def chapter_label(number: str, title: str) -> Text:
    return txt(f"{number}  ·  {title}", 22, MUTED).to_corner(UL, buff=0.32)
