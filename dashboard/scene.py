"""8-bit pixel-art fly, arcade console and the animation state machine. Pure drawing logic: it talks to a tiny canvas interface
(rect / ellipse / line / polygon / text), so the same scene is drawn by pygame in the window and by PIL in the tests and
screenshots. No pygame import here.

States: IDLE, RESEARCH, BUY, SELL, HOLD, SILENT.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

W, H = 960, 720
CX, CY = W // 2, H // 2            # the fly is centred here
PX = 10                            # one sprite cell = 10x10 screen pixels

BG = (10, 12, 18)
PANEL = (20, 24, 34)
PANEL2 = (32, 38, 54)
GRID = (18, 22, 32)
TEXT = (210, 220, 235)
DIM = (110, 122, 145)
GREEN = (57, 255, 20)              # neon green
RED = (255, 40, 50)
CYAN = (80, 220, 255)
AMBER = (255, 190, 60)
CHARCOAL = (44, 46, 54)
CHARCOAL2 = (66, 70, 82)
WING = (200, 205, 215)
EYE = (150, 20, 40)
NEON = (57, 255, 160)

TERM = (16, 80, 350, 400)          # crawler terminal: x, y, w, h (350 x 400)
RAIL_BUY = (430, 640, 524)         # x0, x1, y of the long lever path
RAIL_SELL = (710, 920, 524)
TAPE = (400, 458, 534, 30)         # ticker tape strip


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ animation state machine
DURATION = {"BUY": 1.5, "SELL": 1.5, "HOLD": 1.8, "SILENT": 3.0}
FX, FY = CX + 48, CY - 45          # thorax centre of the 3D fly
FLOOR = FY + 112
SHOULDER = {"BUY": (FX + 30, FY + 46), "SELL": (FX + 30, FY + 46)}      # the front leg is the arm


@dataclass
class Pose:
    state: str = "IDLE"
    t: float = 0.0
    color: tuple = CHARCOAL
    hand: tuple = (CX, CY + 60)
    arm: str = ""
    reach: float = 0.0
    lever_buy: float = 0.0           # 0 up .. 1 pressed
    lever_sell: float = 0.0
    flash: float = 0.0
    eyes_closed: bool = False
    wing: float = 0.0
    bob: float = 0.0
    scan: float = 0.0                # magnifier sweep 0..1 during HOLD
    ticker: str = ""
    zzz: float = 0.0


class FlyAnimator:
    """Feed it Decision objects (dashboard.state.Decision) and a research flag; call update(dt) each frame; read .pose()."""

    def __init__(self, log_lines=10, max_queue=12):
        self.state, self.t, self.cur = "IDLE", 0.0, None
        self.queue = deque(maxlen=max_queue)
        self.log = deque(maxlen=log_lines)          # (color, text)
        self.research = False
        self.clock = 0.0
        self._logged = False

    def set_research(self, on):
        self.research = bool(on)

    def push(self, decision):
        self.queue.append(decision)

    @staticmethod
    def message(d):
        s = d.score
        if d.action == "BUY":
            return GREEN, f"[FLY BUY] Entered Long Position on {d.ticker} - MBON Hz: {s:+.1f} (buy-sell)"
        if d.action == "SELL":
            return RED, f"[FLY SELL] Covered Short Position on {d.ticker} - MBON Hz: {s:+.1f} (buy-sell)"
        if d.action == "HOLD":
            return CYAN, f"[FLY HOLD] Defending Position - Maintaining neutral holding on {d.ticker}"
        return AMBER, f"[FLY SILENT] {d.agent}: constant score, circuit is dead - flagged for culling"

    def update(self, dt):
        self.clock += dt
        if self.cur is None and self.queue:
            self.cur = self.queue.popleft()
            self.state, self.t, self._logged = self.cur.action, 0.0, False
        if self.cur is not None:
            self.t += dt
            if not self._logged and self.t >= (0.55 if self.state in ("BUY", "SELL") else 0.25):
                self.log.append(self.message(self.cur))
                self._logged = True
            if self.t >= DURATION[self.state]:
                self.cur, self.t = None, 0.0
        if self.cur is None:
            self.state = "RESEARCH" if self.research else "IDLE"

    def pose(self):
        c = self.clock
        p = Pose(state=self.state, t=self.t, wing=math.sin(c * 40), bob=round(math.sin(c * 2.2) * 1.5) * 2,
                 eyes_closed=(c % 4.0) < 0.12)
        d = self.cur
        if self.state in ("BUY", "SELL") and d is not None:
            sh = SHOULDER[self.state]
            rail = RAIL_BUY if self.state == "BUY" else RAIL_SELL
            rest = (sh[0] + 16, sh[1] + 52)
            t = self.t
            reach = ease(t / 0.4) if t < 0.7 else 1 - ease((t - 0.7) / 0.5)
            press = ease((t - 0.4) / 0.15) if t < 0.7 else 1 - ease((t - 0.7) / 0.3)
            press = max(0.0, press)
            head_x = rail[0] + press * (rail[1] - rail[0])
            tgt = (head_x, rail[2] - 20)
            p.hand = (lerp(rest[0], tgt[0], reach), lerp(rest[1], tgt[1], reach))
            p.arm = self.state
            p.reach = reach
            if self.state == "BUY":
                p.lever_buy = press
            else:
                p.lever_sell = press
            p.color = GREEN if self.state == "BUY" else RED
            p.flash = max(0.0, min(1.0, (ease(t / 0.3) if t < 0.9 else 1 - ease((t - 0.9) / 0.6))))
            p.ticker = d.ticker
        elif self.state == "HOLD" and d is not None:
            p.scan = self.t / DURATION["HOLD"]
            p.color, p.flash, p.ticker = CYAN, 0.35, d.ticker
        elif self.state == "SILENT":
            p.eyes_closed = True
            p.color, p.flash, p.zzz = AMBER, 0.0, self.t
            p.bob = 6
            p.wing = 0.0
        elif self.state == "RESEARCH":
            p.color, p.flash = AMBER, 0.15
        return p


# ------------------------------------------------------------------ drawing
@dataclass
class View:
    header: dict = field(default_factory=dict)       # round, universe, top_id, top_ic, ridge_ic, untrained_ic, mode
    pose: Pose = field(default_factory=Pose)
    log: list = field(default_factory=list)          # [(color, text)]
    tape: list = field(default_factory=list)         # [(ticker, score)]
    research_lines: list = field(default_factory=list)
    research_status: str = ""
    clock: float = 0.0
    replay_label: str = ""


def _fmt(x):
    return "n/a" if x is None or x != x else f"{x:+.3f}"


def draw_scene(cv, v: View):
    cv.rect(0, 0, W, H, BG)
    for x in range(0, W, 40):
        cv.line(x, 72, x, 556, GRID)
    for y in range(72, 556, 40):
        cv.line(0, y, W, y, GRID)
    _header(cv, v)
    cv.rect(8, 72, W - 16, 484, None, outline=NEON, width=2)            # interior neon boundary
    _terminal(cv, v)
    _console(cv, v)
    _fly(cv, v)
    _log(cv, v)


def _header(cv, v, w=W):
    h = v.header
    cv.rect(0, 0, w, 64, PANEL)
    cv.rect(0, 62, w, 2, NEON)
    cv.text("FLY-TRADER LEAGUE", 16, 8, NEON, 20)
    mode = h.get("mode", "idle").upper()
    col = {"RESEARCH": AMBER, "LEAGUE": GREEN, "IDLE": DIM}.get(mode, DIM)
    cv.rect(w - 170, 8, 154, 22, BG, outline=col)
    cv.text(mode, w - 93, 10, col, 14, "c")
    top, ic, ridge = h.get("top_id") or "-", h.get("top_ic"), h.get("ridge_ic")
    beats = ic is not None and ridge is not None and ic == ic and ridge == ridge and ic > ridge
    cells = [("ROUND", str(h.get("round", "-")), TEXT), ("UNIVERSE", f"{h.get('universe', 0)} stocks", TEXT),
             ("TOP FLY", top, NEON), ("3-RD AVG IC", _fmt(ic), GREEN if beats else AMBER),
             ("RIDGE", _fmt(ridge), TEXT), ("UNTRAINED", _fmt(h.get("untrained_ic")), DIM)]
    x = 16
    for label, val, c in cells:
        cv.text(label, x, 34, DIM, 11)
        cv.text(val, x, 46, c, 14)
        x += 150 if label != "UNIVERSE" else 135


def _terminal(cv, v):
    x, y, w, h = TERM
    research = v.pose.state == "RESEARCH"
    cv.rect(x, y, w, h, (20, 120, 50), alpha=70)                         # green-tinted transparent terminal
    cv.rect(x, y, w, h, None, outline=mix(NEON, BG, 0.2 if research else 0.55), width=2)
    cv.text("SEC EDGAR CRAWLER", x + 10, y + 8, AMBER if research else DIM, 12)
    status = v.research_status if research and v.research_status else ("SEARCHING SEC EDGAR..." if research else "STANDBY")
    cv.text(status, x + 10, y + 28, GREEN if research else DIM, 17)
    lines = v.research_lines
    top, step = y + 64, 16
    n = (h - 72) // step
    if not lines:
        cv.text("(no SEC cache activity detected)", x + 10, top, DIM, 13)
        return
    rate = 12 if research else 2                                         # rows per second: fast when crawling
    start = int(v.clock * rate)
    for k in range(n):
        ln = lines[(start + k) % len(lines)]
        tk, tag, rest = ln[:6].strip(), ln[7:41].strip()[:15], ln[42:].split(" filed")[0]
        fade = 1.0 - 0.6 * (k / n)
        if not research:
            fade *= 0.55
        txt = f"{tk:<5} {tag:<15} {rest}"
        while txt and cv.text_width(txt, 12) > w - 20:
            txt = txt[:-1]
        cv.text(txt, x + 10, top + k * step, mix(BG, GREEN if k % 2 == 0 else TEXT, fade), 12)


def _console(cv, v):
    p = v.pose
    cv.rect(390, 444, 554, 104, PANEL, outline=NEON, width=2)             # arcade console table
    cv.rect(390, 444, 554, 6, PANEL2)
    tx, ty, tw, th = TAPE
    cv.rect(tx, ty, tw, th, (6, 8, 12), outline=PANEL2)
    if v.tape:
        items = [f"{tk} {sc:+.1f} {'^' if sc >= 0 else 'v'}" for tk, sc in v.tape]
        widths = [cv.text_width(i, 16) + 28 for i in items]
        total = sum(widths)
        x0 = tx + 6 - (v.clock * 80) % total
        for _ in range(int((tw + 20) // total) + 3):
            for it, wd, (tk, sc) in zip(items, widths, v.tape):
                if x0 >= tx + 4 and x0 + wd - 28 <= tx + tw - 4:         # whole items only: no clipping needed
                    cv.text(it, x0, ty + 6, GREEN if sc >= 0 else RED, 16)
                x0 += wd
    else:
        cv.text("NO TAPE", tx + tw // 2, ty + 7, DIM, 14, "c")
    if p.state == "HOLD":                                                # monitoring: a box sweeps along the tape
        bx = tx + 4 + int(p.scan * (tw - 90))
        cv.rect(bx, ty + 1, 86, th - 2, None, outline=CYAN, width=2)
        cv.text(p.ticker, bx + 43, ty - 14, CYAN, 12, "c")
    for rail, name, col, press in ((RAIL_BUY, "LONG", GREEN, p.lever_buy), (RAIL_SELL, "SHORT", RED, p.lever_sell)):
        x0, x1, y = rail
        cv.rect(x0 - 16, y - 6, x1 - x0 + 32, 12, (8, 10, 16), outline=mix(col, BG, 0.5), width=2)
        for gx in range(x0 + 20, x1 - 10, 36):
            cv.rect(gx, y - 1, 12, 2, mix(col, BG, 0.65))
        cv.text(name, x0 - 16, y - 24, col, 12)
        hx = int(x0 + press * (x1 - x0))
        glow = press > 0.5
        if glow:
            cv.ellipse(hx - 22, y - 22, 44, 44, mix(BG, col, 0.35))
        cv.ellipse(hx - 14, y - 14, 28, 28, col if not glow else mix(col, (255, 255, 255), 0.5), outline=(255, 255, 255))
        cv.rect(hx - 6, y - 8, 6, 6, (255, 255, 255))


def _sphere(cv, cx, cy, rx, ry, dark, light, steps=9, lx=-0.35, ly=-0.45, spec=None):
    """Banded shading: nested ellipses, each smaller, lighter and shifted toward the light. Looks like a lit pixel-art ball."""
    for i in range(steps):
        t = i / (steps - 1)
        k = 1.0 - 0.8 * t
        w, h = 2 * rx * k, 2 * ry * k
        x = cx + lx * rx * t * 0.9 - w / 2
        y = cy + ly * ry * t * 0.9 - h / 2
        cv.ellipse(x, y, w, h, mix(dark, light, t ** 0.9))
    if spec:
        sx, sy = cx + lx * rx * 0.55, cy + ly * ry * 0.6
        cv.rect(sx - 1, sy - 6, 3, 13, spec)                              # four-point sparkle
        cv.rect(sx - 6, sy - 1, 13, 3, spec)


def _wing(cv, bx, by, length, width, angle, tilt=0.0):
    """Translucent leaf-shaped wing from base (bx, by) pointing at `angle` (radians, 0 = right), with veins."""
    ca, sa = math.cos(angle), math.sin(angle)
    top, bot, mid = [], [], []
    for i in range(15):
        t = i / 14
        w = width * (math.sin(math.pi * min(1.0, t * 0.92 + 0.04)) ** 0.75) * (0.55 + 0.45 * t)
        px, py = length * t, w * (0.35 + tilt)
        top.append((bx + px * ca - py * sa, by + px * sa + py * ca))
        bot.append((bx + px * ca + (w * (0.65 - tilt)) * sa, by + px * sa - (w * (0.65 - tilt)) * ca))
    pts = top + bot[::-1]
    cv.polygon(pts, (150, 185, 235), outline=(205, 230, 255), alpha=85)
    for k in (3, 6, 9, 12):                                              # veins fan out from the base
        cv.line(bx, by, (top[k][0] + bot[k][0]) / 2, (top[k][1] + bot[k][1]) / 2, (190, 215, 245), 1)
    cv.line(bx, by, top[11][0], top[11][1], (190, 215, 245), 1)


def _leg(cv, hip, knee, foot, w=3, col=(168, 184, 206)):
    cv.line(hip[0], hip[1], knee[0], knee[1], col, w)
    cv.line(knee[0], knee[1], foot[0], foot[1], col, w)
    cv.ellipse(knee[0] - 3, knee[1] - 3, 7, 7, (215, 228, 245))
    cv.line(foot[0], foot[1], foot[0] + 8, foot[1] + 1, col, 2)


def _fly(cv, v):
    p = v.pose
    c = v.clock
    cx, cy = FX, FY + p.bob
    silent = p.state == "SILENT"
    rim = (30, 70, 255)
    # glossy floor: soft shadow and a faint blue reflection (dashes), no 3D object ever crosses the console
    cv.ellipse(cx - 150, FLOOR + 2, 330, 14, (8, 10, 22))
    for k in range(5):
        cv.rect(cx - 150 + k * 28, FLOOR + 6 + k * 3, 220 - k * 24, 2, mix(BG, rim, 0.55 - k * 0.1))
    # back legs first (darker), then the body, then the front legs
    walk = 0.0 if silent else math.sin(c * 3.0) * 5
    hips = [(cx - 24, cy + 46), (cx + 2, cy + 50), (cx + 28, cy + 46)]
    for i, hp in enumerate(hips[:2]):
        ph = walk * (1 if i % 2 else -1)
        _leg(cv, hp, (hp[0] - 34 + ph, hp[1] + 22), (hp[0] - 58 + ph, FLOOR), col=(95, 108, 130))
    # blue rim glow under the abdomen, abdomen with stripes
    cv.ellipse(cx - 172, cy - 6, 150, 88, mix(BG, rim, 0.3))
    ax, ay, arx, ary = cx - 100, cy + 14, 78, 46
    _sphere(cv, ax, ay, arx, ary, (24, 22, 28), (92, 80, 74), steps=8, lx=-0.1, ly=-0.55)
    for k in range(-3, 4):
        x = ax + k * 18
        hh = ary * math.sqrt(max(0.0, 1 - ((x - ax) / arx) ** 2)) * 0.88
        cv.line(x, ay - hh + 8, x, ay + hh, (16, 14, 18), 2)
    # wings (flap), anchored on the thorax
    fl = 0.0 if silent else p.wing * 0.12
    base = (cx - 24, cy - 40)
    if silent:
        _wing(cv, base[0], base[1], 150, 30, math.radians(198), 0.1)
        _wing(cv, base[0] + 6, base[1] + 6, 120, 26, math.radians(205), 0.1)
    else:
        _wing(cv, base[0], base[1], 200, 42, math.radians(205) + fl, 0.12)
        _wing(cv, base[0] + 8, base[1] + 10, 150, 36, math.radians(165) + fl * 1.5, 0.1)
    # thorax with bristles
    for i in range(9):
        a = math.radians(205 + i * 15)
        bx, by = cx + 54 * math.cos(a), cy + 50 * math.sin(a)
        cv.line(bx, by, bx + 12 * math.cos(a), by + 12 * math.sin(a), (200, 210, 225), 1)
    _sphere(cv, cx, cy, 56, 52, (74, 86, 112), (186, 198, 216), steps=10)
    cv.ellipse(cx - 50, cy - 8, 14, 14, (205, 218, 236), outline=(130, 150, 180))       # side joint
    # head with the compound eye
    hx, hy = cx + 52, cy + 6 + (12 if silent else 0)
    _sphere(cv, hx, hy, 32, 30, (80, 92, 118), (178, 190, 210), steps=8)
    _sphere(cv, hx - 8, hy - 30, 18, 14, (110, 8, 34), (230, 60, 80), steps=6)
    ex, ey = hx + 16, hy - 6
    flash = 0.5 + 0.5 * math.sin(c * 22)
    if p.state == "BUY":
        dark, light = (0, 90, 30), mix(GREEN, (230, 255, 230), flash * 0.6)
    elif p.state == "SELL":
        dark, light = (110, 4, 20), mix(RED, (255, 180, 180), 0.25 + 0.2 * flash)
    else:
        dark, light = (96, 6, 32), (240, 55, 85)
    if p.eyes_closed:
        cv.ellipse(ex - 26, ey - 3, 54, 12, (60, 30, 40))
        cv.line(ex - 24, ey + 3, ex + 26, ey + 3, (0, 0, 0), 4)                          # closed: horizontal black line
    else:
        _sphere(cv, ex, ey, 27, 30, dark, light, steps=9, spec=(255, 255, 255))
    if p.state == "HOLD":                                                                # binocular goggles over the eye
        for off, sz in ((-18, 30), (0, 34)):
            gx, gy = ex - sz + off - 4, ey - sz + (off // 3)
            cv.rect(gx, gy, 2 * sz, 2 * sz, (30, 34, 44), outline=(140, 152, 175), width=3)
            cv.ellipse(gx + 8, gy + 8, 2 * sz - 16, 2 * sz - 16, mix(BG, CYAN, 0.7), outline=(220, 245, 255))
            cv.rect(gx + 14, gy + 14, 8, 8, (240, 255, 255))
        cv.rect(ex - 40, ey - 8, 14, 8, (140, 152, 175))
    # front leg: standing leg, or the arm that reaches the lever
    fk = hips[2]
    if p.arm:
        sx, sy = SHOULDER[p.arm]
        hxp, hyp = p.hand
        elbow = ((sx + hxp) / 2 - 6, min(sy, (sy + hyp) / 2) - 26)
        col = mix((168, 184, 206), p.color, 0.45)
        cv.line(sx, sy, elbow[0], elbow[1], col, 4)
        cv.line(elbow[0], elbow[1], hxp, hyp, col, 4)
        cv.ellipse(elbow[0] - 4, elbow[1] - 4, 9, 9, (215, 228, 245))
        cv.ellipse(hxp - 8, hyp - 8, 17, 17, mix((168, 184, 206), p.color, 0.7), outline=p.color)
    else:
        ph = -walk
        _leg(cv, fk, (fk[0] + 38 + ph, fk[1] + 12), (fk[0] + 64 + ph, FLOOR))
    # state props
    if silent:
        for k in range(3):
            a = (p.zzz * 0.9 + k * 0.5) % 1.5
            cv.text("Z" if k % 2 == 0 else "z", hx + 30 + k * 18 + int(a * 8), hy - 70 - int(a * 60) - k * 10,
                    mix(BG, AMBER, 1 - a / 1.5), 18 + k * 6)
        cv.text("DEAD CIRCUIT - CULL WARNING", FX + 40, 236, AMBER, 13, "c")
    if p.state == "RESEARCH":
        for k in range(3):
            cv.rect(hx + 38 + k * 14, hy - 62 - k * 14, 8, 8, AMBER)
    lab = {"BUY": GREEN, "SELL": RED, "HOLD": CYAN, "SILENT": AMBER, "RESEARCH": AMBER}.get(p.state, DIM)
    cv.text(p.state, FX + 40, 252 if silent else 238, lab, 14, "c")


def _log(cv, v):
    cv.rect(0, 562, W, H - 562, (6, 8, 12))
    cv.rect(0, 562, W, 2, PANEL2)
    cv.text("LOG", 16, 568, DIM, 12)
    cv.text(v.replay_label or "NO REPLAY DATA", W - 16, 568, AMBER, 12, "r")
    y = 588
    for col, text in list(v.log)[-7:]:
        cv.text(text, 16, y, col, 14)
        y += 18