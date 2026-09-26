"""Colours, fonts and text helpers shared by every scene.

Colour code (kept identical in every diagram of the video):
  weight = red, normal = blue, tension = yellow, friction = pink,
  applied force = orange, velocity = cyan, acceleration = green, momentum = purple.
"""
from xml.sax.saxutils import escape

import manimpango
from manim import (BOLD, DOWN, LEFT, UL, ManimColor, MarkupText, MathTex, Tex,
                   VGroup, config)
from manim.mobject.text import text_mobject as _tm

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


class CrispText(MarkupText):
    """MarkupText laid out at SUPER x the requested size, without line wrapping, then scaled down.

    Manim lays text out in Pango at font_size / 4.8 points, so a size-23 label is set at under 5 pt.
    Hinted glyph advances are rounded at that tiny size, which makes word spacing erratic (spaces
    can vanish entirely). Laying out large and scaling down gives even spacing at every size.
    """

    SUPER = 4.0

    def __init__(self, text: str, font_size: float = 32, **kw):
        super().__init__(text, font_size=font_size * self.SUPER, **kw)
        self.scale(1 / self.SUPER)

    def _text2hash(self, color) -> str:
        return "crisp_" + super()._text2hash(color)

    def _text2svg(self, color) -> str:
        color = ManimColor(color)
        size = self._font_size / _tm.TEXT2SVG_ADJUSTMENT_FACTOR
        line_spacing = self.line_spacing / _tm.TEXT2SVG_ADJUSTMENT_FACTOR
        text_dir = config.get_dir("text_dir")
        text_dir.mkdir(parents=True, exist_ok=True)
        svg = text_dir / (self._text2hash(color) + ".svg")
        if not svg.exists():
            final = f'<span foreground="{color.to_hex()}">{self.text}</span>' if color is not None else self.text
            manimpango.MarkupUtils.text2svg(
                final, self.font, self.slant, self.weight, size, line_spacing, self.disable_ligatures,
                str(svg.resolve()), _tm.START_X, _tm.START_Y, 20000, 4000,  # large canvas, no wrapping
                justify=self.justify, pango_width=None)
        return str(svg.resolve())


def txt(s: str, size: float = 32, color: str = TEXT, weight=None, **kw) -> CrispText:
    """Plain text (markup characters are escaped)."""
    if weight is not None:
        kw["weight"] = weight
    return CrispText(escape(s), font=FONT, font_size=size, color=color, **kw)


def bold(s: str, size: float = 32, color: str = TEXT, **kw) -> CrispText:
    return txt(s, size, color, weight=BOLD, **kw)


def mtxt(markup: str, size: float = 32, color: str = TEXT, **kw) -> CrispText:
    """Pango markup text, e.g. 'a <span foreground="#FBBF24">net</span> force'."""
    return CrispText(markup, font=FONT, font_size=size, color=color, **kw)


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


def chapter_label(number: str, title: str) -> CrispText:
    return txt(f"{number}  ·  {title}", 22, MUTED).to_corner(UL, buff=0.32)
