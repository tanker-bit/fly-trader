"""What is shown on the 3D monitor: drawn with the same 2D canvas interface as the flat dashboard (PIL or pygame), then used
as the texture of the screen mesh."""
from __future__ import annotations

import math

import numpy as np

from dashboard.scene import AMBER, BG, CYAN, DIM, GREEN, RED, TEXT, mix

SW, SH = 512, 320


def _price_line(ticker, clock, n=64):
    """Smooth random walk seeded by the ticker symbol, scrolling left. Illustrative: the dashboard has no price feed."""
    seed = sum(ord(c) * (i + 3) for i, c in enumerate(ticker or "FLY")) % 100000
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(size=400))
    off = int(clock * 14) % (len(walk) - n)
    seg = walk[off:off + n]
    seg = (seg - seg.min()) / max(np.ptp(seg), 1e-6)
    return seg


def draw_screen(cv, v):
    p, c = v.pose, v.clock
    st = p.state
    cv.rect(0, 0, SW, SH, (4, 12, 8))
    for y in range(0, SH, 4):
        cv.line(0, y, SW, y, (6, 18, 12))                                  # scanlines
    if st == "RESEARCH":
        blink = int(c * 3) % 2 == 0
        cv.text("CRAWLING SEC EDGAR..." if blink else "CRAWLING SEC EDGAR   ", 14, 10, GREEN, 22)
        cv.text(v.research_status or "DOWNLOADING SEC XBRL...", 14, 40, AMBER, 15)
        lines = v.research_lines
        if not lines:
            cv.text("(no SEC cache activity detected)", 14, 80, DIM, 14)
        for k in range(14):
            if lines:
                ln = lines[(int(c * 14) + k) % len(lines)]
                tk, tag, rest = ln[:6].strip(), ln[7:41].strip()[:14], ln[42:].split(" filed")[0]
                txt = f"{tk:<5} {tag:<14} {rest}"
            else:
                txt = "".join(chr(33 + (int(c * 20) * 7 + k * 13 + j * 5) % 90) for j in range(40))
            while txt and cv.text_width(txt, 13) > SW - 24:
                txt = txt[:-1]
            cv.text(txt, 14, 66 + k * 17, mix((4, 12, 8), GREEN if k % 2 == 0 else TEXT, 1 - 0.55 * k / 14), 13)
    elif st in ("BUY", "SELL"):
        col = GREEN if st == "BUY" else RED
        cv.rect(0, 0, SW, SH, col, alpha=int(40 + 170 * p.flash))
        cv.text("LONG" if st == "BUY" else "SHORT", SW // 2, 70, (255, 255, 255), 54, "c")
        cv.text(p.ticker, SW // 2, 140, (255, 255, 255), 44, "c")
        cv.text("[ ORDER SENT ]", SW // 2, 220, (255, 255, 255), 20, "c")
    elif st == "HOLD":
        cv.text(f"HOLD  {p.ticker}", 14, 10, CYAN, 24)
        cv.text("(illustrative price line - no price feed in the dashboard)", 14, 44, DIM, 11)
        seg = _price_line(p.ticker, c)
        n = len(seg)
        for gx in range(0, SW, 64):
            cv.line(gx, 70, gx, SH - 20, (10, 40, 24))
        for gy in range(70, SH - 20, 40):
            cv.line(0, gy, SW, gy, (10, 40, 24))
        for i in range(1, n):
            x0, x1 = (i - 1) * SW / (n - 1), i * SW / (n - 1)
            y0, y1 = SH - 30 - seg[i - 1] * 190, SH - 30 - seg[i] * 190
            cv.line(x0, y0, x1, y1, GREEN, 3)
        cv.rect(SW - 14, int(SH - 30 - seg[-1] * 190) - 5, 10, 10, (255, 255, 255))
    elif st == "SILENT":
        cv.text("NO SIGNAL", SW // 2, 110, AMBER, 44, "c")
        cv.text("circuit is dead - flagged for culling", SW // 2, 170, DIM, 16, "c")
        for k in range(10):
            yy = (int(c * 90) + k * 31) % SH
            cv.line(0, yy, SW, yy, (60, 40, 10), 2)
    else:
        cv.text("FLY-TRADER LEAGUE", SW // 2, 100, GREEN, 34, "c")
        cv.text("STANDBY", SW // 2, 150, DIM, 20, "c")
        for k, (tk, sc) in enumerate(v.tape[-5:]):
            cv.text(f"{tk:<6} {sc:+.1f}", 24, 200 + k * 20, GREEN if sc >= 0 else RED, 15)
    cv.rect(0, 0, SW, SH, None, outline=(40, 120, 70), width=3)
