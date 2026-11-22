"""Builds one frame of the cyberpunk desk scene as triangle batches, posed from dashboard.scene.Pose.

World: y up, desk top at y = 0. The low-poly fruit fly sits on a chair at the left edge of the desk, facing +x; the console
with the green (long) and red (short) levers is in front of it, the monitor stands at the right and faces back toward the
fly and the camera. Camera looks down at the desk from the front-left (isometric-ish, narrow field of view)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dashboard3d.m3d import Batch, apply, box_mesh, prism_mesh, quad_mesh, rot_x, rot_y, rot_z, scale, trans, two_bone_ik

EYE = (-3.6, 5.6, 10.8)
TARGET = (-0.6, 0.8, 0.0)
FOV = 34.0
UP_LIGHT = np.array([-0.35, 0.85, 0.45])
UP_LIGHT = UP_LIGHT / np.linalg.norm(UP_LIGHT)

SEAT = np.array([-3.0, -0.35, 0.0])
LEVER_BASE = {"BUY": np.array([-1.25, 0.14, 0.55]), "SELL": np.array([-1.25, 0.14, -0.55])}
LEVER_LEN = 0.95
MON_M = trans(1.8, 1.45, -0.15) @ rot_y(math.radians(-70))          # monitor local frame: +z = screen normal
SCR_W, SCR_H = 1.2, 0.74                                           # half-sizes of the screen quad (local)

PAL_WIN = [(0, 255, 255), (255, 60, 200), (255, 225, 90), (130, 130, 255), (60, 255, 140)]
BODY = (62, 70, 92)
BODY2 = (92, 104, 132)
STRIPE = (36, 40, 56)
EYE_IDLE = (205, 30, 60)
CHAIR = (34, 30, 56)
MAGENTA = (255, 40, 200)
CYAN = (40, 230, 255)
GREEN = (57, 255, 20)
RED = (255, 40, 50)
AMBER = (255, 190, 60)


@dataclass
class Frame:
    main: Batch
    front: Batch
    screen_quad: np.ndarray            # (4,3) world corners: TL, TR, BR, BL
    tint: tuple = (0, 0, 0)
    flash: float = 0.0


def _skyline():
    rng = np.random.default_rng(7)
    b = Batch()
    b.add(prism_mesh(3.4, 0.05, 28), trans(5.0, 4.2, -17.5) @ rot_x(math.pi / 2), (255, 90, 140), emissive=True)     # synthwave sun
    for k in range(5):
        b.add(box_mesh(7.2, 0.14 + 0.05 * k, 0.05), trans(5.0, 2.2 + 0.55 * k, -17.4), (20, 6, 40), emissive=True)
    x = -17.0
    while x < 17.0:
        w = rng.uniform(1.0, 2.0)
        h = rng.uniform(2.5, 12.0)
        z = -8.0 - rng.uniform(0, 4.0)
        shade = int(rng.uniform(26, 48))
        b.add(box_mesh(w, h, 1.6), trans(x, -2.2 + h / 2, z), (shade, shade - 4, shade + 14))
        for yy in np.arange(-1.8, -2.2 + h - 0.3, 0.5):
            for xx in np.arange(x - w / 2 + 0.25, x + w / 2 - 0.15, 0.4):
                if rng.random() < 0.5:
                    c = PAL_WIN[rng.integers(len(PAL_WIN))]
                    b.add(quad_mesh((xx, yy, z + 0.82), (xx + 0.2, yy, z + 0.82), (xx + 0.2, yy + 0.3, z + 0.82), (xx, yy + 0.3, z + 0.82)),
                          None, c, emissive=True)
        x += w + rng.uniform(0.05, 0.5)
    return b


_SKY = None
_ENV = None


def _environment():
    """Static scene: floor with neon grid, desk, chair, console, monitor stand. Built once."""
    b = Batch()
    b.box((0, -2.3, -2), (34, 0.1, 22), (12, 8, 28))
    for gx in np.arange(-14, 14.1, 1.0):
        b.add(quad_mesh((gx - .02, -2.24, -13), (gx + .02, -2.24, -13), (gx + .02, -2.24, 9), (gx - .02, -2.24, 9)), None, (120, 30, 150), emissive=True)
    for gz in np.arange(-13, 9.1, 1.0):
        b.add(quad_mesh((-14, -2.24, gz - .02), (14, -2.24, gz - .02), (14, -2.24, gz + .02), (-14, -2.24, gz + .02)), None, (120, 30, 150), emissive=True)
    # desk
    b.box((0.4, -0.1, 0), (5.8, 0.2, 3.2), (38, 34, 66))
    for sx in (-2.3, 3.1):
        for sz in (-1.4, 1.4):
            b.box((sx, -1.2, sz), (0.2, 2.2, 0.2), (26, 24, 46))
    b.box((0.4, 0.015, 1.6), (5.8, 0.05, 0.05), MAGENTA, emissive=True)
    b.box((-2.5, 0.015, 0), (0.05, 0.05, 3.2), CYAN, emissive=True)
    # console deck with two lever bases
    b.box((-1.25, 0.06, 0), (1.5, 0.12, 2.0), (28, 26, 52))
    b.box((-1.25, 0.13, 0), (1.46, 0.02, 1.96), (60, 56, 100))
    for k, col in (("BUY", GREEN), ("SELL", RED)):
        base = LEVER_BASE[k]
        b.add(prism_mesh(0.2, 0.1, 8), trans(*base), (20, 20, 30))
        b.box((base[0], 0.145, base[2] * 1.9), (1.1, 0.02, 0.05), col, emissive=True)      # lever rail light
    # chair
    b.box(tuple(SEAT + (0, -0.06, 0)), (0.95, 0.12, 0.95), CHAIR)
    b.box((SEAT[0] - 0.5, 0.15, 0), (0.12, 1.1, 0.95), CHAIR)
    b.box((SEAT[0] - 0.46, 0.15, 0), (0.02, 1.05, 0.9), (255, 40, 200), emissive=True)
    b.add(prism_mesh(0.09, 1.5, 6), trans(SEAT[0], -1.15, 0), (50, 50, 70))
    for a in (0.0, 1.2, 2.4, 3.6, 4.8):
        b.limb((SEAT[0], -1.85, 0), (SEAT[0] + 0.8 * math.cos(a), -2.15, 0.8 * math.sin(a)), 0.09, (50, 50, 70))
    # monitor body (screen quad is added per frame)
    b.add(box_mesh(0.9, 0.08, 0.6), MON_M @ trans(0, -1.43, 0), (20, 20, 34))
    b.add(box_mesh(0.2, 0.75, 0.2), MON_M @ trans(0, -1.02, -0.12), (30, 30, 48))
    b.add(box_mesh(2.8, 1.8, 0.2), MON_M, (24, 24, 40))
    return b


def _lever(front, name, col, tilt):
    base = LEVER_BASE[name]
    tip = base + LEVER_LEN * np.array([math.sin(tilt), math.cos(tilt), 0.0])
    front.limb(base, tip, 0.08, (150, 150, 170))
    front.add(prism_mesh(0.17, 0.2, 6), trans(*tip), col, emissive=True)
    return tip


def _limb_chain(b, shoulder, elbow, hand, col=BODY2, thick=0.13):
    b.limb(shoulder, elbow, thick, col)
    b.limb(elbow, hand, thick * 0.85, col)
    b.add(box_mesh(0.18, 0.14, 0.14), trans(*hand), BODY)


def build_frame(pose, clock, ticker="", hud_flash_color=None):
    global _SKY, _ENV
    if _SKY is None:
        _SKY, _ENV = _skyline(), _environment()
    main, front = Batch(), Batch()
    main.tris, main.col, main.emi, main.alpha = (list(x) for x in (_SKY.tris, _SKY.col, _SKY.emi, _SKY.alpha))
    for l_, s_ in ((main.tris, _ENV.tris), (main.col, _ENV.col), (main.emi, _ENV.emi), (main.alpha, _ENV.alpha)):
        l_.extend(s_)
    st = pose.state
    flash = float(pose.flash)
    scol = {"BUY": GREEN, "SELL": RED, "HOLD": (80, 220, 255), "SILENT": AMBER, "RESEARCH": (60, 255, 140)}.get(st, (90, 90, 120))

    # ---- levers
    tilt_b = pose.lever_buy * 0.62
    tilt_s = -pose.lever_sell * 0.62
    tip = {"BUY": _lever(front, "BUY", GREEN, tilt_b), "SELL": _lever(front, "SELL", RED, tilt_s)}

    # ---- fly placement
    lean = {"BUY": 0.14, "SELL": -0.1, "HOLD": 0.12, "RESEARCH": 0.08}.get(st, 0.0)
    if st == "SILENT":
        G = trans(-2.15, 0.52, -0.55) @ rot_x(1.5)
    else:
        G = trans(*SEAT) @ trans(0, math.sin(clock * 2.2) * 0.02, 0)
    U = rot_z(-lean)                                           # upper body pitch about the hip
    P = lambda p: apply(G, apply(U, np.asarray(p, float))[None])[0]
    body = Batch()

    def box(c, s, col, emissive=False, alpha=1.0):
        body.add(box_mesh(*s), G @ U @ trans(*c), col, emissive, alpha)

    box((0, 0.85, 0), (0.85, 0.8, 0.72), BODY)                                  # thorax
    box((0.05, 1.05, 0), (0.6, 0.1, 0.5), BODY2)
    for i in range(3):                                                           # abdomen segments
        body.add(box_mesh(0.55, 0.55 - 0.08 * i, 0.55 - 0.08 * i),
                 G @ trans(-0.55 - 0.42 * i, 0.62 - 0.12 * i, 0) @ rot_z(-0.25 * (i + 1) * 0.4),
                 STRIPE if i % 2 else BODY2)
    hy = 1.42
    box((0.6, hy, 0), (0.52, 0.5, 0.58), BODY)                                  # head
    box((0.92, hy - 0.2, 0), (0.32, 0.1, 0.1), (140, 140, 160))                 # proboscis
    for sz in (-1, 1):
        box((0.62, hy + 0.34, 0.14 * sz), (0.05, 0.3, 0.05), (140, 140, 160))   # antennae
    ecol = EYE_IDLE
    if st == "BUY":
        ecol = tuple(int(c) for c in (np.array(GREEN) * (0.7 + 0.3 * math.sin(clock * 22) ** 2)))
    elif st == "SELL":
        ecol = RED
    elif st == "HOLD":
        ecol = (255, 255, 255)
    for sz in (-1, 1):
        if pose.eyes_closed or st == "SILENT":
            box((0.8, hy + 0.04, 0.3 * sz), (0.05, 0.04, 0.26), (10, 10, 14))     # closed: a thin dark slit
        else:
            box((0.7, hy + 0.04, 0.36 * sz), (0.3, 0.38, 0.24), ecol, emissive=True)
    flap = 0.0 if st == "SILENT" else pose.wing * 0.22
    for sz in (-1, 1):                                                            # wings (translucent)
        r = [(-0.1, 1.25, 0.2 * sz), (0.15, 1.25, 0.2 * sz), (-0.9, 1.95 + flap, 1.1 * sz), (-1.7, 1.85 + flap, 0.9 * sz)]
        body.add(quad_mesh(*[P(q) for q in r]) if False else quad_mesh(*[apply(G @ U, np.array(q)[None])[0] for q in r]),
                 None, (190, 215, 245), alpha=0.5)
    # legs: the two rear/mid pairs hang down to the chair
    for sz in (-1, 1):
        for dx in (-0.2, 0.1):
            a, k, f = P((dx, 0.5, 0.3 * sz)), P((dx - 0.15, 0.05, 0.75 * sz)), P((dx + 0.15, -0.3, 0.85 * sz))
            body.limb(a, k, 0.09, BODY2)
            body.limb(k, f, 0.08, BODY2)
    # front arms: right arm (z+) works the green lever, left arm (z-) the red one; both tap the deck while researching
    rest = {1: np.array([-2.15, 0.5, 0.5]), -1: np.array([-2.15, 0.5, -0.5])}
    for sz, name in ((1, "BUY"), (-1, "SELL")):
        sh = P((0.3, 0.95, 0.42 * sz))
        tgt = rest[sz].copy()
        if st == "SILENT":
            tgt = sh + np.array([0.1, -0.7, 0.2 * sz])
        elif st == "RESEARCH":
            tgt = np.array([-1.7 + 0.15 * math.sin(clock * 9 + sz), 0.17 + abs(math.sin(clock * 16 + (0 if sz > 0 else 1.6))) * 0.1, 0.45 * sz])
        elif st == name:
            grip = tip[name] + np.array([-0.06, -0.05, 0.0])
            tgt = rest[sz] + (grip - rest[sz]) * pose.reach
        elbow, hand = two_bone_ik(sh, tgt, 1.0, 1.0, bend=(0, 1, 0.35 * sz))
        _limb_chain(body, sh, elbow, hand)
    front_items = body
    for t_, c_, e_, a_ in zip(front_items.tris, front_items.col, front_items.emi, front_items.alpha):
        front.tris.append(t_), front.col.append(c_), front.emi.append(e_), front.alpha.append(a_)

    # ---- state glow: LED strip under the monitor, halo around the screen
    led = scol if st in ("BUY", "SELL", "HOLD", "RESEARCH", "SILENT") else (90, 90, 120)
    main.add(box_mesh(2.8, 0.06, 0.06), MON_M @ trans(0, -0.93, 0.1), led, emissive=True)
    if flash > 0.02 and st in ("BUY", "SELL"):
        for k, (sc, al) in enumerate(((1.1, 0.22), (1.2, 0.10))):
            main.add(quad_mesh((-1.4 * sc, 0.9 * sc, -0.05), (1.4 * sc, 0.9 * sc, -0.05), (1.4 * sc, -0.9 * sc, -0.05), (-1.4 * sc, -0.9 * sc, -0.05)),
                     MON_M, scol, emissive=True, alpha=al * flash)
    # ---- Zzz: blocky 3D 'Z' glyphs rising from the head
    if st == "SILENT":
        head = P((0.6, hy + 0.5, 0.0))
        for k in range(3):
            ph = (pose.zzz * 0.7 + k * 0.5) % 1.5
            s = 0.18 + 0.07 * k
            org = head + np.array([0.25 + 0.2 * k + ph * 0.2, 0.15 + ph * 1.1, 0.4])
            fade = 1.0 - ph / 1.5
            for (c, sz_) in (((0, s, 0), (s * 1.6, s * 0.28, 0.06)), ((0, -s, 0), (s * 1.6, s * 0.28, 0.06))):
                front.add(box_mesh(*sz_), trans(*(org + np.array(c))), AMBER, emissive=True, alpha=max(0.15, fade))
            front.add(box_mesh(s * 2.1, s * 0.28, 0.06), trans(*org) @ rot_z(-0.9), AMBER, emissive=True, alpha=max(0.15, fade))

    corners = np.array([(-SCR_W, SCR_H, 0.105), (SCR_W, SCR_H, 0.105), (SCR_W, -SCR_H, 0.105), (-SCR_W, -SCR_H, 0.105)])
    return Frame(main, front, apply(MON_M, corners), tint=scol, flash=flash)
