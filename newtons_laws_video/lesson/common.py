"""Scene-level helpers shared by the chapters."""
from manim import (DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, GrowFromCenter, Line,
                   VGroup)

from .style import ACCENT, MUTED, bold, chapter_label, txt


def chapter_card(number: str, title: str, subtitle: str | None = None) -> VGroup:
    part = bold(f"PART {number}", 30, ACCENT)
    head = bold(title, 58)
    rule = Line(LEFT * 3.2, RIGHT * 3.2, color=ACCENT, stroke_width=3)
    group = VGroup(part, head, rule)
    if subtitle:
        group.add(txt(subtitle, 30, MUTED))
    return group.arrange(DOWN, buff=0.3)


def open_chapter(scene, number: str, title: str, subtitle: str | None = None,
                 hold: float = 1.6):
    """Show the chapter card, then leave a small chapter label in the corner."""
    card = chapter_card(number, title, subtitle)
    scene.play(FadeIn(card[:2], shift=UP * 0.2), GrowFromCenter(card[2]),
               *[FadeIn(m, shift=UP * 0.2) for m in card[3:]], run_time=0.9)
    scene.wait(hold)
    label = chapter_label(number, title)
    scene.play(FadeOut(card, shift=UP * 0.3), FadeIn(label), run_time=0.7)
    return label
