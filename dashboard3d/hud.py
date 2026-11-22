"""Flat overlays drawn on top of the 3D view: the neon HUD across the top and the translucent log."""
from __future__ import annotations

from dashboard.scene import DIM, _header


def draw_hud(cv, v, w):
    _header(cv, v, w)


def draw_log(cv, v, w, h):
    cv.rect(0, 0, w, h, (4, 6, 10), alpha=190)
    cv.rect(0, 0, w, 2, (57, 255, 160))
    cv.text("LOG", 14, 6, DIM, 12)
    cv.text(v.replay_label or "NO REPLAY DATA", w - 14, 6, (255, 190, 60), 12, "r")
    y = 26
    for col, text in list(v.log)[-6:]:
        cv.text(text, 14, y, col, 14)
        y += 18
