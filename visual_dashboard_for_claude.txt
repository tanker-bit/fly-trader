"""
experiments/visual_dashboard.py  --  STONKFLY 3D visual overlay dashboard

Self-contained ModernGL + Pygame scene (no .obj / texture files).
All geometry is generated mathematically into native vertex buffers.

Build status (grown in parts):
  [x] Part 1  : core engine (math, shaders, Mesh, Camera, main loop)
  [x] Part 2  : shared state + async poller thread
  [x] Part 3  : solid geometry (box, UV sphere, cylinder/cone, transform/concat) + adapter fix
  [x] Part 3.5: wireframes, tube edges, ring, limb/joint-chain helpers, two-bone IK
  [x] Part 4a : skyline: sky gradient, depth-layered low-poly towers, setbacks, antennas, haze
  [x] Part 4b : skyline: scattered neon window pixels (4 twinkle groups) + antenna beacons
  [x] Part 5a : desk, floor, macro-pad console deck (charcoal, solid)
  [x] Part 5b : neon cyan edge striping on the desk and console + purple floor grid
  [x] Part 6a : monitor (+ skyline fixes)
  [x] Part 6b : PC tower with dual fan rings, mug
  [x] Part 7a : fly body (head/thorax/abdomen), crimson eyes with cross-hatching
  [x] Part 7b : translucent wireframe wings + office chair
  [x] Part 8a : jointed gray legs (5 resting) + IK arm in rest pose
  [x] Part 8b : macro-pad keys + BUY/SELL lever, arm driven by the ViewModel
  [x] Part 9a : screen texture path + scrolling candlestick chart (illustrative, labelled)
  [x] Part 9b : raw metrics grid, flashing CRAWLING SEC EDGAR... panel, SILENT dim scanlines
  [x] Part 9c : bigger main monitor + fullscreen + any window shape
  [x] Part 9d : long desk, second monitor (decision tape)
  [x] Part 9e : long side of the desk faces the fly; three monitors in a row (the side ones angled toward the camera)
  [x] Part 9f : third monitor (rolling IC history) at the right end of the row
        9f-a IcContent + render_ic + tests (done); 9f-b IcAccumulator feeds ic_history from LeagueBridge/SimFeed, third monitor drawn (done)
  [x] Part 10 : HUD + log stream
        10a HUD top bar (done); 10b log stream + floating Zzz (done)
  [x] Part 9g : double-size text on all three screens
  [x] Part 11 : rig animation driven by ViewModel (the state machine itself lives in Part 2)
        11a body pose: HOLD lean-in, SILENT slump, eyes close, arm stays on the lever (done); 11b wing droop/flutter, bob, desk glow pool (done)
  [x] Part 12 : integration + polish
        12a bloom (4x MSAA offscreen target, bright-pass + 2 blurs, B / +/- keys, --no-bloom); trading date dd/mm/yyyy on the monitors (done)
        12b end-to-end fake-GL frame test over every state, window shape and bloom failure path (done)

Run:   python experiments/visual_dashboard.py --league-root league_data_fundamentals [--panel P] [--sec-root S]
       python experiments/visual_dashboard.py --sim          (synthetic feed, no league needed)
Check: python experiments/visual_dashboard.py --selftest   (no GPU needed)
Keys:  TAB next agent | SPACE pause | B bloom on/off, +/- bloom strength | ESC quit        (fullscreen by default; --windowed [--size 1600x900])
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
WINDOW_SIZE = (1280, 720)
TARGET_FPS = 60
SKY_TOP = (0.05, 0.03, 0.13)       # deep navy-purple
SKY_BOTTOM = (0.16, 0.07, 0.27)    # purple glow near horizon
FOG_COLOR = (0.13, 0.06, 0.24)     # distant buildings fade into this purple haze
FOG_DENSITY = 0.011                # ~3% fog at the desk, ~50% at 60 units

# Palette (linear-ish RGB, 0..1)
COL_CHARCOAL = (0.12, 0.12, 0.15)
COL_NEON_CYAN = (0.00, 0.95, 1.00)
COL_CRIMSON = (0.80, 0.05, 0.10)
COL_GREEN = (0.10, 1.00, 0.35)
COL_RED = (1.00, 0.15, 0.20)
COL_GRAY = (0.62, 0.62, 0.66)
COL_VIOLET = (0.75, 0.30, 0.85)


# --------------------------------------------------------------------------
# Matrix math (column-vector convention: clip = P @ V @ M @ v)
# --------------------------------------------------------------------------
def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = identity()
    m[:3, 3] = (x, y, z)
    return m


def scale(x: float, y: float | None = None, z: float | None = None) -> np.ndarray:
    y = x if y is None else y
    z = x if z is None else z
    return np.diag([x, y, z, 1.0]).astype(np.float32)


def rotate_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rotate_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rotate_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = identity()
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -(m[:3, :3] @ eye)
    return m


def gl_bytes(m: np.ndarray) -> bytes:
    """Column-major bytes for a GLSL mat4 uniform."""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


# --------------------------------------------------------------------------
# Shaders  (one program: optional lambert lighting, tint, emissive boost)
# --------------------------------------------------------------------------
VERT_SRC = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = mat3(u_model) * in_normal;   // uniform scale assumed
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAG_SRC = """
#version 330
uniform vec3  u_light_dir;   // direction light travels (world)
uniform vec3  u_ambient;
uniform vec3  u_tint;        // multiplies vertex colour
uniform float u_emissive;    // 0 = normal, >0 adds self-illumination
uniform float u_lit;         // 1 = lambert, 0 = flat/unlit
uniform float u_alpha;
uniform vec3  u_eye;          // camera position (world), for distance fog
uniform vec3  u_fog_color;
uniform float u_fog_density;
uniform float u_fog_k;        // 1 = fogged, 0 = never fogged (sky gradient, HUD)
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;
out vec4 f_color;
void main() {
    vec3 base = v_color * u_tint;
    vec3 col = base;
    if (u_lit > 0.5) {
        float d = max(dot(normalize(v_normal), -normalize(u_light_dir)), 0.0);
        col = base * (u_ambient + vec3(d) * 0.85);
    }
    col += base * u_emissive;
    float fd = length(v_world - u_eye) * u_fog_density;
    col = mix(col, u_fog_color, (1.0 - exp(-fd * fd)) * u_fog_k);
    f_color = vec4(col, u_alpha);
}
"""


# --------------------------------------------------------------------------
# Mesh: interleaved [pos3 normal3 color3] float32 vertex buffer
# --------------------------------------------------------------------------
class Mesh:
    """GPU mesh built from a numpy (N, 9) array. mode: 'tri' or 'line'."""

    STRIDE = 9

    def __init__(self, ctx, program, data: np.ndarray, mode: str = "tri"):
        import moderngl  # local import so --selftest works without GPU libs

        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, self.STRIDE)
        self.count = len(data)
        self.mode = moderngl.TRIANGLES if mode == "tri" else moderngl.LINES
        self.program = program
        self.vbo = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(
            program, [(self.vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")]
        )

    def draw(self, scene: "Scene", model: np.ndarray, tint=(1, 1, 1),
             emissive: float = 0.0, lit: bool = True, alpha: float = 1.0,
             fog: float = 1.0, mvp: np.ndarray | None = None):
        """mvp overrides proj_view @ model (used for the clip-space sky gradient)."""
        p = self.program
        p["u_mvp"].write(gl_bytes(scene.proj_view @ model if mvp is None else mvp))
        p["u_fog_k"].value = float(fog)
        p["u_model"].write(gl_bytes(model))
        p["u_tint"].value = tuple(tint)
        p["u_emissive"].value = float(emissive)
        p["u_lit"].value = 1.0 if lit else 0.0
        p["u_alpha"].value = float(alpha)
        self.vao.render(self.mode)

    def update(self, data: np.ndarray):
        """Re-upload vertices of the same count (per-frame animated meshes such as the IK arm)."""
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, self.STRIDE)
        if len(data) != self.count:
            raise ValueError(f"Mesh.update needs {self.count} vertices, got {len(data)}")
        self.vbo.write(data.tobytes())

    def release(self):
        self.vao.release()
        self.vbo.release()


# --------------------------------------------------------------------------
# Camera: fixed perspective, high and slightly left of the desk
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, aspect: float):
        self.eye = np.array([-12.49, 8.0, 7.19], dtype=np.float32)      # 9e: behind-left of the fly, looking along +x at the monitor row
        self.target = np.array([0.3, 1.0, 1.5], dtype=np.float32)
        self.base_fov = 36.0
        self.view = look_at(self.eye, self.target)
        self.set_aspect(aspect)

    def set_aspect(self, aspect: float):
        """Any window shape: the 16:9 framing is the minimum horizontal view, so narrower windows (16:10, 4:3) widen the
        vertical fov instead of cropping the desk; wider windows (21:9) keep the vertical fov and show more city."""
        self.aspect = float(aspect)
        ref = 16.0 / 9.0
        if self.aspect >= ref:
            self.fov = self.base_fov
        else:
            self.fov = math.degrees(2.0 * math.atan(math.tan(math.radians(self.base_fov) / 2.0) * ref / self.aspect))
        self.proj = perspective(self.fov, self.aspect, 0.1, 400.0)

    @property
    def proj_view(self) -> np.ndarray:
        return self.proj @ self.view


class Scene:
    """Per-frame render context shared by every drawable."""

    def __init__(self, camera: Camera):
        self.camera = camera
        self.time = 0.0
        self.proj_view = camera.proj_view
        self.light_dir = np.array([0.35, -0.8, -0.5], dtype=np.float32)
        self.ambient = np.array([0.38, 0.36, 0.5], dtype=np.float32)

    def begin_frame(self, program, t: float):
        self.time = t
        self.proj_view = self.camera.proj_view
        program["u_light_dir"].value = tuple(float(v) for v in self.light_dir)
        program["u_ambient"].value = tuple(float(v) for v in self.ambient)
        program["u_eye"].value = tuple(float(v) for v in self.camera.eye)
        program["u_fog_color"].value = FOG_COLOR
        program["u_fog_density"].value = FOG_DENSITY


# --------------------------------------------------------------------------
# Minimal vertex-building helpers (full generators arrive in Part 3)
# --------------------------------------------------------------------------
def quad_xz(w: float, d: float, y: float, color) -> np.ndarray:
    """Upward-facing quad on the XZ plane centred at origin."""
    hw, hd = w / 2, d / 2
    n = (0, 1, 0)
    c = tuple(color)
    pts = [(-hw, y, -hd), (-hw, y, hd), (hw, y, hd),
           (-hw, y, -hd), (hw, y, hd), (hw, y, -hd)]
    return np.array([[*p, *n, *c] for p in pts], dtype=np.float32)


def test_cube(size: float, color) -> np.ndarray:
    """Flat-shaded cube centred at origin, used to verify the pipeline."""
    h = size / 2
    faces = [
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
    ]
    rows = []
    for n, q in faces:
        for i in (0, 1, 2, 0, 2, 3):
            rows.append([*q[i], *n, *color])
    return np.array(rows, dtype=np.float32)


# --------------------------------------------------------------------------
# Part 3: solid geometry generators
#
# Every generator returns a float32 (N, 9) array of [pos3, normal3, colour3]
# triangles (CCW seen from outside), exactly what Mesh() expects. Nothing is
# loaded from disk. `flat=True` gives per-triangle normals (a visibly faceted,
# low/medium-poly look); `flat=False` gives smooth normals.
# --------------------------------------------------------------------------
def _finish(tris, color, normals=None) -> np.ndarray:
    """tris (T,3,3); normals (T,3,3) per-vertex or None for flat faces. Drops zero-area triangles."""
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(fn, axis=1)
    keep = ln > 1e-12
    tris, fn, ln = tris[keep], fn[keep], ln[keep]
    if normals is None:
        nrm = np.repeat((fn / ln[:, None])[:, None, :], 3, axis=1)
    else:
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3, 3)[keep]
    col = np.broadcast_to(np.asarray(color, dtype=np.float64), tris.shape)
    return np.concatenate([tris, nrm, col], axis=2).reshape(-1, 9).astype(np.float32)


def box_geo(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Axis-aligned box centred at the origin, flat-shaded (36 vertices)."""
    x, y, z = sx / 2, sy / 2, sz / 2
    quads = np.array([
        [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],          # +z
        [(x, -y, -z), (-x, -y, -z), (-x, y, -z), (x, y, -z)],      # -z
        [(x, -y, z), (x, -y, -z), (x, y, -z), (x, y, z)],          # +x
        [(-x, -y, -z), (-x, -y, z), (-x, y, z), (-x, y, -z)],      # -x
        [(-x, y, z), (x, y, z), (x, y, -z), (-x, y, -z)],          # +y
        [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)],      # -y
    ], dtype=np.float64)
    return _finish(quads[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3), color)


def uv_sphere_geo(radius: float = 1.0, slices: int = 16, stacks: int = 12,
                  color=(1.0, 1.0, 1.0), flat: bool = False) -> np.ndarray:
    """UV sphere, poles on +/-Y. Triangles: slices * (2 * stacks - 2)."""
    th = np.linspace(0.0, np.pi, stacks + 1)
    ph = np.linspace(0.0, 2.0 * np.pi, slices + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    u = np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], axis=-1)   # unit sphere (S+1, L+1, 3)
    a, b, c, d = u[:-1, :-1], u[:-1, 1:], u[1:, 1:], u[1:, :-1]
    unit = np.concatenate([np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
                           np.stack([a, c, d], axis=2).reshape(-1, 3, 3)])
    return _finish(unit * radius, color, None if flat else unit)


def cylinder_geo(radius_bottom: float = 0.5, radius_top: float | None = None, height: float = 1.0,
                 segments: int = 12, color=(1.0, 1.0, 1.0), caps: bool = True, flat: bool = False) -> np.ndarray:
    """Upright (axis = Y) cylinder or cone frustum centred at the origin. radius_top=0 makes a cone."""
    rb = radius_bottom
    rt = radius_bottom if radius_top is None else radius_top
    ph = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    c, s = np.cos(ph), np.sin(ph)
    hb, ht = -height / 2, height / 2
    bot = np.stack([rb * c, np.full_like(c, hb), rb * s], axis=-1)
    top = np.stack([rt * c, np.full_like(c, ht), rt * s], axis=-1)
    a, b, cc, d = bot[:-1], bot[1:], top[1:], top[:-1]
    side = np.concatenate([np.stack([a, d, cc], axis=1), np.stack([a, cc, b], axis=1)])
    nv = np.stack([c, np.full_like(c, (rb - rt) / max(height, 1e-9)), s], axis=-1)
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    na, nb = nv[:-1], nv[1:]
    side_n = np.concatenate([np.stack([na, na, nb], axis=1), np.stack([na, nb, nb], axis=1)])
    parts = [_finish(side, color, None if flat else side_n)]
    if caps:
        ct, cb = np.array([0.0, ht, 0.0]), np.array([0.0, hb, 0.0])
        tri_top = np.stack([np.broadcast_to(ct, top[:-1].shape), top[1:], top[:-1]], axis=1)
        tri_bot = np.stack([np.broadcast_to(cb, bot[:-1].shape), bot[:-1], bot[1:]], axis=1)
        parts += [_finish(tri_top, color), _finish(tri_bot, color)]
    return np.concatenate(parts)


def transform_geo(data: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to positions and (inverse-transpose) to normals. Returns a new array.
    A mirroring matrix (negative determinant) flips triangle winding; avoid it."""
    d = np.asarray(data, dtype=np.float32).reshape(-1, 9).copy()
    M = np.asarray(M, dtype=np.float64)
    pos = d[:, :3].astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    n = d[:, 3:6].astype(np.float64) @ np.linalg.inv(M[:3, :3])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    d[:, :3], d[:, 3:6] = pos, n
    return d


def concat_geo(*parts) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1, 9) for p in parts])


def recolor_geo(data: np.ndarray, color) -> np.ndarray:
    d = np.asarray(data, dtype=np.float32).reshape(-1, 9).copy()
    d[:, 6:9] = np.asarray(color, dtype=np.float32)
    return d


# --------------------------------------------------------------------------
# Part 3.5: wireframes, rings, limbs, IK
#
# *_wire_geo / circle_wire_geo return LINE-LIST vertices: build them with
# Mesh(ctx, prog, data, mode="line") and draw with lit=False. Core-profile GL
# ignores line widths above 1 px, so for thick neon edges use wire_box_tubes_geo
# (12 thin solid boxes) and draw it emissive.
# --------------------------------------------------------------------------
def _lines(segs, color) -> np.ndarray:
    """segs (K,2,3) -> (2K, 9) vertices; normal is a +Y placeholder (draw unlit)."""
    s = np.asarray(segs, dtype=np.float64).reshape(-1, 3)
    n = np.tile((0.0, 1.0, 0.0), (len(s), 1))
    c = np.broadcast_to(np.asarray(color, dtype=np.float64), s.shape)
    return np.concatenate([s, n, c], axis=1).astype(np.float32)


def wire_box_geo(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """All 12 edges of a box centred at the origin (24 vertices)."""
    x, y, z = sx / 2, sy / 2, sz / 2
    c = np.array([(a, b, d) for a in (-x, x) for b in (-y, y) for d in (-z, z)], dtype=np.float64)   # index = 4*ix + 2*iy + iz
    edges = [(c[i], c[i | bit]) for i in range(8) for bit in (4, 2, 1) if not i & bit]
    return _lines(edges, color)


def wire_sphere_geo(radius: float = 1.0, slices: int = 12, stacks: int = 8, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Latitude rings + meridians: the geometric cross-hatching on the compound eyes."""
    ph = np.linspace(0.0, 2.0 * np.pi, slices + 1)
    th = np.linspace(0.0, np.pi, stacks + 1)
    segs = []
    for k in range(1, stacks):
        t = np.pi * k / stacks
        ring = np.stack([np.sin(t) * np.cos(ph), np.full_like(ph, np.cos(t)), np.sin(t) * np.sin(ph)], axis=-1) * radius
        segs.append(np.stack([ring[:-1], ring[1:]], axis=1))
    for j in range(slices):
        m = np.stack([np.sin(th) * np.cos(ph[j]), np.cos(th), np.sin(th) * np.sin(ph[j])], axis=-1) * radius
        segs.append(np.stack([m[:-1], m[1:]], axis=1))
    return _lines(np.concatenate(segs), color)


def circle_wire_geo(radius: float = 1.0, segments: int = 32, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Circle outline in the XY plane (faces +Z), 2*segments vertices."""
    a = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    p = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], axis=-1)
    return _lines(np.stack([p[:-1], p[1:]], axis=1), color)


def ring_geo(outer: float = 1.0, inner: float = 0.7, segments: int = 32, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Flat annulus in the XY plane facing +Z: one glowing fan ring of the PC tower (2*segments triangles)."""
    a = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    cs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
    o, i = cs * outer, cs * inner
    tris = np.concatenate([np.stack([o[:-1], o[1:], i[1:]], axis=1), np.stack([o[:-1], i[1:], i[:-1]], axis=1)])
    return _finish(tris, color)


def wire_box_tubes_geo(sx: float, sy: float, sz: float, thickness: float, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Thick neon edges: 12 thin boxes along the edges of an sx*sy*sz box (outer size grows by `thickness`)."""
    x, y, z, t = sx / 2, sy / 2, sz / 2, thickness
    parts = []
    for b in (-y, y):
        for d in (-z, z):
            parts.append(transform_geo(box_geo(sx + t, t, t, color), translate(0, b, d)))
    for a in (-x, x):
        for d in (-z, z):
            parts.append(transform_geo(box_geo(t, sy + t, t, color), translate(a, 0, d)))
    for a in (-x, x):
        for b in (-y, y):
            parts.append(transform_geo(box_geo(t, t, sz + t, color), translate(a, b, 0)))
    return concat_geo(*parts)


def align_y(d) -> np.ndarray:
    """4x4 rotation taking +Y onto direction d (Rodrigues; handles parallel and anti-parallel)."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    M = np.eye(4)
    if n < 1e-12:
        return M
    d = d / n
    c = d[1]
    if c > 1.0 - 1e-9:
        return M
    if c < -1.0 + 1e-9:
        return rotate_x(np.pi).astype(np.float64)
    v = np.cross((0.0, 1.0, 0.0), d)
    s = np.linalg.norm(v)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M[:3, :3] = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return M


def limb_geo(p0, p1, radius: float, color=(1.0, 1.0, 1.0), segments: int = 8,
             radius_end: float | None = None, flat: bool = False) -> np.ndarray:
    """Cylinder (optionally tapered) from p0 to p1. radius is at p0, radius_end at p1."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.zeros((0, 9), dtype=np.float32)
    base = cylinder_geo(radius, radius_end, 1.0, segments, color, True, flat)
    return transform_geo(base, translate(*((p0 + p1) / 2)) @ align_y(d) @ scale(1.0, L, 1.0))


def joint_chain_geo(points, radius, color=(1.0, 1.0, 1.0), joint_scale: float = 1.25,
                    segments: int = 8, flat: bool = False) -> np.ndarray:
    """Jointed leg/arm: tapered limb per pair of points plus a sphere at every joint.
    radius: scalar, or one value per point (limbs taper between neighbours)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,)).copy()
    unit = uv_sphere_geo(1.0, segments, max(3, segments // 2), color, flat)
    parts = [limb_geo(pts[i], pts[i + 1], r[i], color, segments, r[i + 1], flat) for i in range(n - 1)]
    parts += [transform_geo(unit, translate(*pts[i]) @ scale(r[i] * joint_scale)) for i in range(n)]
    return concat_geo(*parts)


def two_bone_ik(shoulder, target, l1: float, l2: float, bend=(0.0, 1.0, 0.0)):
    """(elbow, hand) for a two-segment arm reaching `target`; the elbow bends toward `bend`.
    Unreachable targets are clamped to the arm's reach."""
    s = np.asarray(shoulder, dtype=np.float64)
    d = np.asarray(target, dtype=np.float64) - s
    dn = max(float(np.linalg.norm(d)), 1e-9)
    dist = float(np.clip(dn, abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
    u = d / dn
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    b = np.asarray(bend, dtype=np.float64)
    pole = b - u * float(b @ u)
    pole /= max(float(np.linalg.norm(pole)), 1e-9)
    return s + u * a + pole * h, s + u * dist


# --------------------------------------------------------------------------
# Part 4: night skyline
#
# build_skyline() is pure numpy (testable without a GPU) and returns:
#   sky        vertical purple/navy gradient backdrop (unlit)
#   buildings  dark low-poly towers with roof setbacks and antennas (lit)
#   windows    tiny square neon pixels on the +Z and -X faces (the two faces the
#              camera sees), split into 4 groups that pulse out of phase (unlit)
#   beacons    blinking antenna lights (unlit)
# Far layers are tinted toward the horizon glow for atmospheric depth.
# --------------------------------------------------------------------------
WINDOW_PX_REF_H = 720.0            # reference window height used by the "tiny pixel" size test
NEON_PIXELS = ((0.00, 0.95, 1.00), (1.00, 0.20, 0.80), (1.00, 0.85, 0.30), (0.60, 0.40, 1.00), (0.30, 1.00, 0.55))
SKY_LAYERS = (   # (z, (min_h, max_h), (min_w, max_w)) nearest first; tower base is y = SKY_BASE_Y
    (-13.0, (3.0, 9.0), (1.6, 3.0)),
    (-19.0, (5.0, 13.0), (2.0, 3.6)),
    (-27.0, (8.0, 19.0), (2.4, 4.2)),
    (-38.0, (11.0, 26.0), (3.0, 5.0)),
)
SKY_BASE_Y = -4.0
SKY_YAW_DEG = -90.0           # 9e: the city lies behind the monitor row (the monitors face -x toward the fly)
SKY_FOOT_Y = -40.0          # towers extend down to here so no void shows under the floor edge
SKY_WINDOW_Y0 = -20.0       # windows start here (limits quad count)
WINDOW_HALF = 0.055         # window half-size: 0.11-unit squares, tiny pixels


@dataclasses.dataclass
class Skyline:
    sky: np.ndarray
    buildings: np.ndarray
    windows: list              # 4 arrays (blink groups)
    beacons: np.ndarray
    boxes: np.ndarray          # (B, 6) main tower boxes: cx, cy, cz, w, h, d
    window_slots: int          # grid positions that could hold a window (lit or not)


def _quads(centers, u, v, half: float, normal, colors) -> np.ndarray:
    """K small squares in the plane spanned by unit axes u, v (u x v == normal) -> (6K, 9)."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if len(c) == 0:
        return np.zeros((0, 9), dtype=np.float32)
    hu, hv = np.asarray(u, dtype=np.float64) * half, np.asarray(v, dtype=np.float64) * half
    corners = np.stack([c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv], axis=1)
    tris = corners[:, [0, 1, 2, 0, 2, 3]]
    n = np.broadcast_to(np.asarray(normal, dtype=np.float64), tris.shape)
    col = np.broadcast_to(np.asarray(colors, dtype=np.float64)[:, None, :], tris.shape)
    return np.concatenate([tris, n, col], axis=2).reshape(-1, 9).astype(np.float32)


def sky_gradient_geo(top=SKY_TOP, bottom=SKY_BOTTOM) -> np.ndarray:
    """Full-screen gradient quad in clip space at the far plane (navy on top, purple glow at the bottom).
    Draw with mvp=identity, unlit, fog=0: it always covers the whole window whatever the camera sees."""
    z = 0.99999
    v = [(-1, -1, bottom), (1, -1, bottom), (1, 1, top), (-1, -1, bottom), (1, 1, top), (-1, 1, top)]
    return np.array([[x, y, z, 0, 0, 1, *c] for x, y, c in v], dtype=np.float32)


def build_skyline(seed: int = 7, sky_top=SKY_TOP, sky_bottom=SKY_BOTTOM, x_range=(-25.0, 130.0),
                  layers=SKY_LAYERS, base_y: float = SKY_BASE_Y, neon: bool = True,
                  foot_y: float = SKY_FOOT_Y, win_y0: float = SKY_WINDOW_Y0, yaw_deg: float = 0.0) -> Skyline:
    """neon=False returns empty window/beacon arrays. The random stream is identical either way,
    so the towers are the same with or without the neon pixels."""
    rng = np.random.default_rng(seed)
    horizon = np.asarray(sky_bottom, dtype=np.float64)
    bodies, boxes, beacons = [], [], []
    zc, zcol, zg = [], [], []          # windows on the +Z face: centres, colours, blink group
    xc, xcol, xg = [], [], []          # windows on the -X face
    slots = 0
    pitch_x, pitch_y, half = 0.42, 0.55, WINDOW_HALF
    drop = base_y - foot_y
    for li, (z, (h0, h1), (w0, w1)) in enumerate(layers):
        depth_f = li / max(len(layers) - 1, 1)                         # 0 near .. 1 far
        x = x_range[0] + rng.uniform(0.0, 1.5)
        while x < x_range[1]:
            w, d, h = rng.uniform(w0, w1), rng.uniform(2.0, 3.2), rng.uniform(h0, h1)
            cx, cy = x + w / 2, base_y + (h - drop) / 2
            body = np.array([0.07, 0.06, 0.17]) * rng.uniform(0.8, 1.3) + np.array([0.0, 0.0, 0.05]) * rng.random()
            body = body * (1 - 0.55 * depth_f) + horizon * (0.55 * depth_f)          # atmospheric tint
            bodies.append(transform_geo(box_geo(w, h + drop, d, body), translate(cx, cy, z)))
            boxes.append((cx, cy, z, w, h + drop, d))
            top_y = base_y + h
            if rng.random() < 0.4:                                                    # roof setback
                sh = h * rng.uniform(0.08, 0.16)
                bodies.append(transform_geo(box_geo(w * 0.6, sh, d * 0.6, body * 0.85), translate(cx, top_y + sh / 2, z)))
                top_y += sh
            if rng.random() < 0.18:                                                   # antenna + beacon
                ah = rng.uniform(1.4, 3.2)
                bodies.append(transform_geo(box_geo(0.08, ah, 0.08, body * 1.4), translate(cx, top_y + ah / 2, z)))
                bc = (1.0, 0.15, 0.35) if rng.random() < 0.6 else (1.0, 0.25, 0.9)
                beacons.append(transform_geo(box_geo(0.24, 0.24, 0.24, bc), translate(cx, top_y + ah + 0.1, z)))
            p_lit = rng.uniform(0.15, 0.45)
            dim = 1.0 - 0.35 * depth_f
            ys = np.arange(win_y0, base_y + h - 0.4, pitch_y)
            if w > 0.7 and len(ys):                                                   # +Z face (faces the camera)
                xs = cx - w / 2 + 0.3 + np.arange(int((w - 0.6) / pitch_x) + 1) * pitch_x
                gx, gy = np.meshgrid(xs, ys)
                slots += gx.size
                m = rng.random(gx.shape) < p_lit
                k = int(m.sum())
                zc.append(np.stack([gx[m], gy[m], np.full(k, z + d / 2 + 0.01)], axis=1))
                zcol.append(np.array(NEON_PIXELS)[rng.integers(0, len(NEON_PIXELS), k)] * (rng.uniform(0.65, 1.0, k) * dim)[:, None])
                zg.append(rng.integers(0, 4, k))
            if d > 0.7 and len(ys):                                                   # -X face (left side, seen from the left)
                zs = z - d / 2 + 0.3 + np.arange(int((d - 0.6) / pitch_x) + 1) * pitch_x
                gz, gy = np.meshgrid(zs, ys)
                slots += gz.size
                m = rng.random(gz.shape) < p_lit
                k = int(m.sum())
                xc.append(np.stack([np.full(k, cx - w / 2 - 0.01), gy[m], gz[m]], axis=1))
                xcol.append(np.array(NEON_PIXELS)[rng.integers(0, len(NEON_PIXELS), k)] * (rng.uniform(0.65, 1.0, k) * dim)[:, None])
                xg.append(rng.integers(0, 4, k))
            x += w + rng.uniform(0.05, 0.6)
    cat = lambda parts, shape: np.concatenate(parts) if parts else np.zeros(shape)
    zc, zcol, zg = cat(zc, (0, 3)), cat(zcol, (0, 3)), cat(zg, (0,))
    xc, xcol, xg = cat(xc, (0, 3)), cat(xcol, (0, 3)), cat(xg, (0,))
    groups = []
    for g in range(4):
        groups.append(np.concatenate([
            _quads(zc[zg == g], (1, 0, 0), (0, 1, 0), half, (0, 0, 1), zcol[zg == g]),
            _quads(xc[xg == g], (0, 0, 1), (0, 1, 0), half, (-1, 0, 0), xcol[xg == g])]))
    empty = np.zeros((0, 9), dtype=np.float32)
    sky = Skyline(sky=sky_gradient_geo(sky_top, sky_bottom), buildings=concat_geo(*bodies),
                  windows=groups if neon else [empty] * 4,
                  beacons=(concat_geo(*beacons) if beacons else empty) if neon else empty,
                  boxes=np.array(boxes), window_slots=slots)
    return rotate_skyline(sky, yaw_deg) if yaw_deg else sky


def rotate_skyline(sky: Skyline, yaw_deg: float) -> Skyline:
    """Turn the whole city about the world y axis (multiples of 90 deg only, so the tower boxes stay axis-aligned).
    The clip-space sky gradient is unaffected."""
    k = (yaw_deg / 90.0)
    assert abs(k - round(k)) < 1e-9, "skyline yaw must be a multiple of 90 degrees"
    M = rotate_y(math.radians(yaw_deg))
    rot = lambda g: transform_geo(g, M) if len(g) else g
    b = sky.boxes.copy()
    a = math.radians(yaw_deg)
    x, z = b[:, 0].copy(), b[:, 2].copy()
    b[:, 0] = x * math.cos(a) + z * math.sin(a)
    b[:, 2] = -x * math.sin(a) + z * math.cos(a)
    if int(round(k)) % 2:
        b[:, [3, 5]] = b[:, [5, 3]]
    return Skyline(sky=sky.sky, buildings=rot(sky.buildings), windows=[rot(g) for g in sky.windows],
                   beacons=rot(sky.beacons), boxes=b, window_slots=sky.window_slots)


class SkylineRenderer:
    """Uploads a Skyline once and draws it each frame; windows pulse in 4 groups, beacons blink."""

    def __init__(self, ctx, program, sky: Skyline):
        mk = lambda a, mode="tri": Mesh(ctx, program, a, mode) if len(a) else None
        self.sky = mk(sky.sky)
        self.buildings = mk(sky.buildings)
        self.windows = [mk(g) for g in sky.windows]
        self.beacons = mk(sky.beacons)

    def draw(self, scene, t: float):
        I = identity()
        if self.sky:
            self.sky.draw(scene, I, lit=False, fog=0.0, mvp=I)
        if self.buildings:
            self.buildings.draw(scene, I)
        for g, m in enumerate(self.windows):
            if m:
                pulse = 0.35 if g == 0 else 0.30 + 0.40 * (0.5 + 0.5 * math.sin(t * (0.9 + 0.4 * g) + g * 2.1))
                m.draw(scene, I, lit=False, emissive=pulse)
        if self.beacons:
            self.beacons.draw(scene, I, lit=False, emissive=0.2 + 1.2 * (math.sin(t * 2.2) > 0.6))

    def release(self):
        for m in [self.sky, self.buildings, self.beacons, *self.windows]:
            if m:
                m.release()


# --------------------------------------------------------------------------
# Part 5: workstation desk, floor, macro-pad console deck
#
# World layout, shared by every later part (y up, desk top surface is y = 0):
#   desk      5.8 x 0.2 x 3.2 charcoal slab centred at (0.4, -0.1, 0), four legs to the floor
#   console   1.5 x 0.12 x 2.0 deck on the desk's left with a lighter pad plate: the macro-pad the fly's arm works
#   floor     rooftop slab whose top is y = -2.3, ending before the skyline's nearest row (z > -11.4)
# build_desk(neon=True) also returns the cyan edge striping (Part 5b).
# --------------------------------------------------------------------------
DESK_CENTER = (-0.9, -0.1, 0.0)           # 9d: long desk (left edge stays at x = -2.5), room for a second monitor
DESK_SIZE = (3.2, 0.2, 9.4)           # 9e: the LONG side (z) faces the fly; the monitors stand in a row along the far edge
FLOOR_TOP_Y = -2.3
FLOOR_SIZE = (28.0, 0.1, 40.0)          # wide enough that 21:9 and 4:3 windows never look past the floor edge; ends before the city (x = 11)
FLOOR_CENTER_X = -4.0
FLOOR_CENTER_Z = 0.0
LEG_SIZE = (0.2, 2.2, 0.2)
LEG_XZ = ((-2.3, -4.2), (-2.3, 4.2), (0.5, -4.2), (0.5, 4.2))
CONSOLE_CENTER = (-1.25, 0.06, 0.0)
CONSOLE_SIZE = (1.5, 0.12, 2.0)
PAD_SIZE = (1.46, 0.02, 1.96)
COL_DESK = (0.12, 0.12, 0.15)
COL_LEG = (0.08, 0.08, 0.10)
COL_FLOOR = (0.05, 0.04, 0.10)
COL_CONSOLE = (0.10, 0.10, 0.14)
COL_PAD = (0.20, 0.20, 0.25)


COL_GRID = (0.47, 0.12, 0.59)
NEON_EDGE = 0.06                   # thickness of the cyan desk-edge tubes
NEON_EDGE_CONSOLE = 0.035


def floor_grid_geo(xs, zs, x_ext, z_ext, y: float, width: float = 0.04, color=COL_GRID) -> np.ndarray:
    """Thin upward-facing quads: one line at every x in xs (running over z_ext) and every z in zs (over x_ext)."""
    w = width / 2
    quads = [[(x - w, y, z_ext[0]), (x - w, y, z_ext[1]), (x + w, y, z_ext[1]), (x + w, y, z_ext[0])] for x in xs]
    quads += [[(x_ext[0], y, z - w), (x_ext[0], y, z + w), (x_ext[1], y, z + w), (x_ext[1], y, z - w)] for z in zs]
    return _finish(np.asarray(quads, dtype=np.float64)[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3), color)


@dataclasses.dataclass
class Desk:
    solid: np.ndarray      # lit geometry: top, 4 legs, console, pad (7 boxes)
    floor: np.ndarray      # floor slab (1 box)
    neon: np.ndarray       # emissive cyan striping: all 12 desk-slab edges + all 12 console-deck edges
    boxes: np.ndarray      # rows cx, cy, cz, sx, sy, sz: floor, top, leg0-3, console, pad
    grid: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 9), dtype=np.float32))   # purple floor grid


def build_desk(neon: bool = True) -> Desk:
    rows = [((FLOOR_CENTER_X, FLOOR_TOP_Y - FLOOR_SIZE[1] / 2, FLOOR_CENTER_Z), FLOOR_SIZE, COL_FLOOR),
            (DESK_CENTER, DESK_SIZE, COL_DESK)]
    rows += [((x, FLOOR_TOP_Y + LEG_SIZE[1] / 2, z), LEG_SIZE, COL_LEG) for x, z in LEG_XZ]
    rows += [(CONSOLE_CENTER, CONSOLE_SIZE, COL_CONSOLE),
             ((CONSOLE_CENTER[0], CONSOLE_CENTER[1] + CONSOLE_SIZE[1] / 2 + PAD_SIZE[1] / 2, CONSOLE_CENTER[2]), PAD_SIZE, COL_PAD)]
    geo = [transform_geo(box_geo(*size, col), translate(*c)) for c, size, col in rows]
    empty = np.zeros((0, 9), dtype=np.float32)
    stripes, grid = empty, empty
    if neon:
        # tubes are centred on the box edges, so they stand proud of the faces by half their thickness (no z-fighting)
        stripes = concat_geo(
            transform_geo(wire_box_tubes_geo(*DESK_SIZE, NEON_EDGE, COL_NEON_CYAN), translate(*DESK_CENTER)),
            transform_geo(wire_box_tubes_geo(*CONSOLE_SIZE, NEON_EDGE_CONSOLE, COL_NEON_CYAN), translate(*CONSOLE_CENTER)))
        grid = floor_grid_geo(np.arange(-17.0, 9.5, 1.0), np.arange(-19.0, 19.5, 1.0), (-18.0, 10.0), (-20.0, 20.0), FLOOR_TOP_Y + 0.005)
    return Desk(solid=concat_geo(*geo[1:]), floor=geo[0], neon=stripes, grid=grid,
                boxes=np.array([[*c, *size] for c, size, _ in rows], dtype=np.float64))


class DeskRenderer:
    def __init__(self, ctx, program, desk: Desk):
        mk = lambda a: Mesh(ctx, program, a) if len(a) else None
        self.floor, self.solid, self.neon, self.grid = mk(desk.floor), mk(desk.solid), mk(desk.neon), mk(desk.grid)

    def draw(self, scene, t: float):
        I = identity()
        if self.floor:
            self.floor.draw(scene, I)
        if self.grid:
            self.grid.draw(scene, I, lit=False, emissive=0.25 + 0.1 * math.sin(t * 0.8))
        if self.solid:
            self.solid.draw(scene, I)
        if self.neon:
            self.neon.draw(scene, I, lit=False, emissive=0.35 + 0.15 * math.sin(t * 1.7), fog=0.3)

    def release(self):
        for m in (self.floor, self.solid, self.neon, self.grid):
            if m:
                m.release()


# --------------------------------------------------------------------------
# Part 6a: monitor (wide, flat-shaded, angled inward toward the fly)
# Local frame: origin = screen-body centre, +z = out of the screen. Placed by MON_POS / MON_YAW_DEG.
# --------------------------------------------------------------------------
MON_SCALE = 1.15                    # 9c: bigger screen (was 1.0 at 139 px wide on a 720p window)
MON_POS = (0.15, 1.47 * MON_SCALE, 0.0)   # y: the stand's base bottom rests exactly on the desk top (y = 0)
MON_YAW_DEG = -90.0                 # turned toward the fly but mostly toward the camera, so the screen is readable
MON_BODY = (2.8, 1.8, 0.2)
SCR_HALF = (1.2, 0.74)              # 2.4 x 1.48 screen (16:10)
SCR_Z = 0.105
COL_MON_BODY = (0.094, 0.094, 0.157)
COL_MON_BASE = (0.078, 0.078, 0.133)
COL_MON_NECK = (0.118, 0.118, 0.188)
COL_MON_SCREEN = (0.02, 0.05, 0.07)
MON_BASE_BOX = ((0.0, -1.43, 0.0), (0.9, 0.08, 0.6))
MON_NECK_BOX = ((0.0, -1.02, -0.12), (0.2, 0.75, 0.2))
MON_LED_BOX = ((0.0, -0.93, 0.1), (2.8, 0.06, 0.06))


@dataclasses.dataclass(frozen=True)
class MonitorSpec:
    name: str
    x: float
    z: float
    yaw_deg: float
    scale: float

    @property
    def pos(self):
        return (self.x, 1.47 * self.scale, self.z)         # the stand's base bottom rests on the desk top (y = 0)


MAIN_MONITOR = MonitorSpec("main", MON_POS[0], MON_POS[2], MON_YAW_DEG, MON_SCALE)
SIDE_MONITOR = MonitorSpec("side", -0.2, -3.25, -55.0, 0.9)   # left of the main one (decision tape)
IC_MONITOR = MonitorSpec("ic", 0.15, 3.25, -75.0, 0.9)           # right of it (rolling IC history, Part 9f)   # 9d: second monitor on the long desk, right of the main one
MONITORS = (MAIN_MONITOR, SIDE_MONITOR, IC_MONITOR)


def monitor_matrix(spec: MonitorSpec | None = None) -> np.ndarray:
    spec = spec or MAIN_MONITOR
    return translate(*spec.pos) @ rotate_y(math.radians(spec.yaw_deg)) @ scale(spec.scale)


@dataclasses.dataclass
class Monitor:
    solid: np.ndarray            # body + base + neck (flat shaded)
    led: np.ndarray              # state LED strip (tinted by state in Part 11)
    screen: np.ndarray           # screen quad, 6 verts
    screen_quad: np.ndarray      # (4,3) local corners TL, TR, BR, BL (for Part 9's texture)
    boxes: list                  # [(centre, size)] local-frame solid boxes: body, base, neck


def build_monitor() -> Monitor:
    bx, by = SCR_HALF
    quad = np.array([(-bx, by, SCR_Z), (bx, by, SCR_Z), (bx, -by, SCR_Z), (-bx, -by, SCR_Z)], dtype=np.float64)
    tl, tr, br, bl = quad
    screen = _finish(np.array([[bl, br, tr], [bl, tr, tl]]), COL_MON_SCREEN)      # CCW seen from +z
    parts = [((0.0, 0.0, 0.0), MON_BODY, COL_MON_BODY), (*MON_BASE_BOX, COL_MON_BASE), (*MON_NECK_BOX, COL_MON_NECK)]
    solid = concat_geo(*[transform_geo(box_geo(*size, col), translate(*c)) for c, size, col in parts])
    led = transform_geo(box_geo(*MON_LED_BOX[1], COL_NEON_CYAN), translate(*MON_LED_BOX[0]))
    return Monitor(solid=solid, led=led, screen=screen, screen_quad=quad,
                   boxes=[(c, size) for c, size, _ in parts])


class MonitorRenderer:
    def __init__(self, ctx, program, mon: Monitor, flat_screen: bool = True, model: np.ndarray | None = None, react: bool = True):
        self.mon = mon
        self.flat_screen = flat_screen
        self.react = react                       # False: secondary monitors keep a neutral LED
        self.solid = Mesh(ctx, program, mon.solid)
        self.led = Mesh(ctx, program, mon.led)
        self.screen = Mesh(ctx, program, mon.screen)
        self.model = monitor_matrix() if model is None else model

    def draw(self, scene, t: float, vm=None):
        tint = (1.0, 1.0, 1.0)
        glow = 0.0
        if vm is not None and self.react:
            tint = {"BUY": COL_GREEN, "SELL": COL_RED}.get(vm.state, (1.0, 1.0, 1.0))
            glow = 0.9 * getattr(vm, "flash", 0.0)
        self.solid.draw(scene, self.model)
        if self.flat_screen:
            self.screen.draw(scene, self.model, tint=tint, emissive=0.15 + glow)
        self.led.draw(scene, self.model, tint=tint, lit=False, emissive=0.8 + 0.2 * math.sin(t * 3.0))

    def release(self):
        for m in (self.solid, self.led, self.screen):
            m.release()


# --------------------------------------------------------------------------
# Part 6b: PC tower (floor, bottom right) with two glowing fan rings + violet/pink mug
# Tower local frame: origin = body centre, +z = face. Placed by TOWER_POS / TOWER_YAW_DEG.
# --------------------------------------------------------------------------
TOWER_SIZE = (1.1, 2.2, 1.5)                                # w, h, d
TOWER_POS = (0.5, FLOOR_TOP_Y + TOWER_SIZE[1] / 2, 5.5)      # at the desk's near right end, turned toward the camera
TOWER_YAW_DEG = -81.0                                       # face turned toward the camera
COL_TOWER = (0.10, 0.10, 0.14)
COL_TOWER_FACE = (0.06, 0.06, 0.09)
FAN_Y = (0.5, -0.5)                                         # fan centres on the face (local y)
FAN_RINGS = ((0.43, 0.38), (0.24, 0.19))                    # (outer, inner) of the two concentric rings
FAN_HUB_R = 0.07
FACE_Z = TOWER_SIZE[2] / 2
RING_Z = FACE_Z + 0.012
MUG_POS = (-1.25, 0.0, 1.4)                                 # base centre on the desk top, right of the keyboard pad
MUG_R, MUG_H = 0.17, 0.38
COL_MUG = (0.50, 0.22, 0.85)
COL_MUG_PINK = (1.0, 0.25, 0.75)


def tower_matrix() -> np.ndarray:
    return translate(*TOWER_POS) @ rotate_y(math.radians(TOWER_YAW_DEG))


@dataclasses.dataclass
class Tower:
    body: np.ndarray             # chassis + front plate (lit)
    rings: np.ndarray            # 2 fans x 2 concentric rings (neon, unlit)
    hubs: np.ndarray             # 2 hub discs
    spokes: np.ndarray           # one fan's 4 blades centred on the origin (drawn once per fan, spinning)
    led: np.ndarray             # power LED


def build_tower(neon: bool = True) -> Tower:
    w, h, d = TOWER_SIZE
    body = concat_geo(box_geo(w, h, d, COL_TOWER),
                      transform_geo(box_geo(w - 0.1, h - 0.1, 0.02, COL_TOWER_FACE), translate(0, 0, FACE_Z + 0.005)))
    rings, hubs = [], []
    for fy in FAN_Y:
        for outer, inner in FAN_RINGS:
            rings.append(transform_geo(ring_geo(outer, inner, 40, COL_NEON_CYAN), translate(0, fy, RING_Z)))
        hubs.append(transform_geo(ring_geo(FAN_HUB_R, 0.0, 16, COL_NEON_CYAN), translate(0, fy, RING_Z)))
    blade = lambda a: transform_geo(box_geo(FAN_RINGS[1][1] * 2 - 0.02, 0.03, 0.01, COL_NEON_CYAN), rotate_z(a))
    spokes = concat_geo(*[blade(a) for a in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)])
    led = transform_geo(box_geo(0.07, 0.07, 0.02, (0.2, 1.0, 0.4)), translate(w / 2 - 0.15, -h / 2 + 0.14, RING_Z))
    empty = np.zeros((0, 9), dtype=np.float32)
    return Tower(body=body, rings=concat_geo(*rings) if neon else empty, hubs=concat_geo(*hubs) if neon else empty,
                 spokes=spokes if neon else empty, led=led)


@dataclasses.dataclass
class Mug:
    body: np.ndarray             # violet cup + handle (lit, faceted)
    rim: np.ndarray              # pink rim + drink surface (emissive accent)
    top_y: float


def build_mug() -> Mug:
    seg = 10
    cup = transform_geo(cylinder_geo(MUG_R, MUG_R * 0.9, MUG_H, seg, COL_MUG, flat=True), translate(0, MUG_H / 2, 0))
    hx = MUG_R + 0.07
    handle = concat_geo(
        transform_geo(box_geo(0.14, 0.04, 0.04, COL_MUG), translate(hx - 0.04, MUG_H * 0.8, 0)),
        transform_geo(box_geo(0.14, 0.04, 0.04, COL_MUG), translate(hx - 0.04, MUG_H * 0.25, 0)),
        transform_geo(box_geo(0.04, MUG_H * 0.6, 0.04, COL_MUG), translate(hx + 0.03, MUG_H * 0.525, 0)))
    rim = concat_geo(
        transform_geo(ring_geo(MUG_R, MUG_R - 0.025, seg, COL_MUG_PINK), translate(0, MUG_H + 0.003, 0) @ rotate_x(-math.pi / 2)),
        transform_geo(ring_geo(MUG_R - 0.025, 0.0, seg, (0.9, 0.2, 0.6)), translate(0, MUG_H - 0.04, 0) @ rotate_x(-math.pi / 2)))
    return Mug(body=concat_geo(cup, handle), rim=rim, top_y=MUG_H)


class PropsRenderer:
    """PC tower + mug."""
    def __init__(self, ctx, program, tower: Tower, mug: Mug):
        mk = lambda a: Mesh(ctx, program, a) if len(a) else None
        self.body, self.rings, self.hubs, self.spokes, self.led = (mk(tower.body), mk(tower.rings), mk(tower.hubs),
                                                                    mk(tower.spokes), mk(tower.led))
        self.mug_body, self.mug_rim = mk(mug.body), mk(mug.rim)
        self.tower_m = tower_matrix()
        self.mug_m = translate(*MUG_POS)

    def draw(self, scene, t: float, vm=None):
        M = self.tower_m
        self.body.draw(scene, M)
        pulse = 0.9 + 0.35 * math.sin(t * 2.2)
        if self.rings:
            self.rings.draw(scene, M, lit=False, emissive=pulse, fog=0.3)
            self.hubs.draw(scene, M, lit=False, emissive=pulse, fog=0.3)
            for i, fy in enumerate(FAN_Y):
                spin = rotate_z(t * (2.4 if i == 0 else -2.0))
                self.spokes.draw(scene, M @ translate(0, fy, RING_Z + 0.004) @ spin, lit=False, emissive=0.8 * pulse, fog=0.3)
        self.led.draw(scene, M, lit=False, emissive=1.0)
        self.mug_body.draw(scene, self.mug_m)
        self.mug_rim.draw(scene, self.mug_m, lit=False, emissive=0.8 + 0.2 * math.sin(t * 1.3))

    def release(self):
        for m in (self.body, self.rings, self.hubs, self.spokes, self.led, self.mug_body, self.mug_rim):
            if m:
                m.release()


# --------------------------------------------------------------------------
# Part 7a: fly body (head, thorax, abdomen, proboscis, antennae) + crimson compound eyes
# Fly local frame: origin = the point on the chair seat under the fly, +x = facing the monitor, +y up, z = sideways.
# --------------------------------------------------------------------------
FLY_POS = (-3.0, -0.35, 0.0)
COL_FLY_THORAX = (0.13, 0.13, 0.16)          # charcoal
COL_FLY_HEAD = (0.38, 0.38, 0.44)            # smooth gray
COL_FLY_BAND_A = (0.30, 0.30, 0.36)          # abdomen stripes
COL_FLY_BAND_B = (0.12, 0.12, 0.15)
COL_EYE = (0.78, 0.04, 0.12)                 # crimson
COL_EYE_HATCH = (0.30, 0.0, 0.05)
THORAX_C, THORAX_R = (0.05, 1.10, 0.0), (0.55, 0.50, 0.45)
HEAD_C, HEAD_R = (0.62, 1.50, 0.0), (0.34, 0.36, 0.40)
ABD_C, ABD_R, ABD_TILT_DEG = (-0.50, 0.75, 0.0), (0.45, 0.80, 0.45), 20.0   # long axis (y) is laid rearward and down
EYE_OFFSET, EYE_R = (0.10, 0.05, 0.28), (0.20, 0.26, 0.19)               # relative to HEAD_C; z mirrored
ABD_BANDS = 6


@dataclasses.dataclass
class FlyBody:
    head: np.ndarray             # smooth, higher-facet
    thorax: np.ndarray           # smooth charcoal
    abdomen: np.ndarray          # smooth, striped
    extras: np.ndarray           # proboscis + antennae
    eyes: np.ndarray             # two faceted crimson ellipsoids
    eye_hatch: np.ndarray        # line geometry: geometric cross-hatching over the eyes
    eye_centers: np.ndarray      # (2, 3) local centres [+z eye, -z eye]


def _ellipsoid(radii, center, color, slices, stacks, flat=False, pre=None):
    g = uv_sphere_geo(1.0, slices, stacks, color, flat=flat)
    if pre is not None:
        g = pre(g)
    M = translate(*center) @ scale(*radii)
    return transform_geo(g, M)


def build_fly_body() -> FlyBody:
    thorax = _ellipsoid(THORAX_R, THORAX_C, COL_FLY_THORAX, 28, 18)
    head = _ellipsoid(HEAD_R, HEAD_C, COL_FLY_HEAD, 28, 18)
    # abdomen: stripes by latitude along the long axis, then laid rearward/down
    ab = uv_sphere_geo(1.0, 28, 24, COL_FLY_BAND_A)
    band = np.floor((ab[:, 1] + 1.0) / 2.0 * ABD_BANDS * 0.999999).astype(int) % 2
    ab[:, 6:9] = np.where(band[:, None] == 0, np.array(COL_FLY_BAND_A), np.array(COL_FLY_BAND_B)).astype(np.float32)
    a = math.radians(90.0 + ABD_TILT_DEG)
    abdomen = transform_geo(ab, translate(*ABD_C) @ rotate_z(a) @ scale(*ABD_R))
    hc = np.array(HEAD_C)
    extras = concat_geo(
        limb_geo(hc + (0.28, -0.12, 0.0), hc + (0.52, -0.40, 0.0), 0.05, (0.25, 0.25, 0.30), 8, 0.03),
        limb_geo(hc + (0.18, 0.30, 0.10), hc + (0.36, 0.52, 0.13), 0.028, (0.25, 0.25, 0.30), 6),
        limb_geo(hc + (0.18, 0.30, -0.10), hc + (0.36, 0.52, -0.13), 0.028, (0.25, 0.25, 0.30), 6))
    eyes, hatch, cen = [], [], []
    for sz in (1.0, -1.0):
        c = hc + (EYE_OFFSET[0], EYE_OFFSET[1], EYE_OFFSET[2] * sz)
        cen.append(c)
        eyes.append(_ellipsoid(EYE_R, c, COL_EYE, 14, 10, flat=True))
        hatch.append(transform_geo(wire_sphere_geo(1.0, 14, 10, COL_EYE_HATCH), translate(*c) @ scale(*(np.array(EYE_R) * 1.03))))
    return FlyBody(head=head, thorax=thorax, abdomen=abdomen, extras=extras, eyes=concat_geo(*eyes),
                   eye_hatch=concat_geo(*hatch), eye_centers=np.array(cen))


class FlyRenderer:
    def __init__(self, ctx, program, body: FlyBody):
        self.body = body
        mk = lambda a, **k: Mesh(ctx, program, a, **k) if len(a) else None
        self.solid = mk(concat_geo(body.head, body.thorax, body.abdomen, body.extras))
        self.eyes = mk(body.eyes)
        self.hatch = mk(body.eye_hatch, mode="line")
        self.model = translate(*FLY_POS)

    def draw(self, scene, t: float, vm=None, pose=None):
        M = self.model @ (identity() if pose is None else pose) @ translate(0, body_bob(vm, t), 0)
        self.solid.draw(scene, M)
        tint, emissive = eye_style(vm)
        self.eyes.draw(scene, M, tint=tint, emissive=emissive)
        self.hatch.draw(scene, M, lit=False, emissive=0.5)

    def release(self):
        for m in (self.solid, self.eyes, self.hatch):
            if m:
                m.release()


# --------------------------------------------------------------------------
# Part 7b: translucent wireframe wings (fly-local frame) + minimalist office chair (world frame)
# --------------------------------------------------------------------------
WING_LEN, WING_HALF_W, WING_STEPS = 1.55, 0.24, 18
WING_ROOT = (-0.12, 0.40, 0.16)               # relative to THORAX_C; z mirrored
WING_DIR = (-0.93, 0.18, 0.34)                # span direction (rearward, slightly up, outward); z mirrored
COL_WING_PLATE = (0.76, 0.79, 0.86)
COL_WING_WIRE = (0.88, 0.90, 0.97)
WING_ALPHA = 0.24
CHAIR_XZ = (FLY_POS[0], FLY_POS[2])
SEAT_R, SEAT_T = 0.80, 0.12
SEAT_TOP_Y = FLY_POS[1]
BACK_X, BACK_SIZE, BACK_TILT_DEG = CHAIR_XZ[0] - 1.65, (0.10, 1.0, 1.1), 8.0
COL_CHAIR_SEAT = (0.17, 0.15, 0.26)
COL_CHAIR_METAL = (0.10, 0.10, 0.13)
COL_CHAIR_WHEEL = (0.05, 0.05, 0.07)
COL_CHAIR_TRIM = (0.95, 0.25, 0.80)


def _wing_halfwidth(s):
    return WING_HALF_W * np.sin(np.pi * np.clip(s, 0.0, 1.0) ** 0.75) ** 0.8


@dataclasses.dataclass
class Wings:
    plate: np.ndarray            # local frame: span = +x, chord = z, normal +y (one wing; drawn twice)
    wire: np.ndarray             # outline + longitudinal and cross veins (lines)
    bases: list                  # two 4x4 fly-local matrices (+z wing, -z wing)
    roots: np.ndarray            # (2, 3) wing roots, fly-local


def build_wings() -> Wings:
    s = np.linspace(0.0, 1.0, WING_STEPS + 1)
    u, w = s * WING_LEN, _wing_halfwidth(s)
    bl = np.stack([u, np.zeros_like(u), -w], 1)
    tl = np.stack([u, np.zeros_like(u), w], 1)
    tris = []
    for i in range(WING_STEPS):
        tris.append([bl[i], tl[i + 1], bl[i + 1]])
        tris.append([bl[i], tl[i], tl[i + 1]])
    plate = _finish(np.array(tris), COL_WING_PLATE)
    lift = np.array([0.0, 0.004, 0.0])
    segs = []
    for i in range(WING_STEPS):                                            # outline
        segs.append([bl[i] + lift, bl[i + 1] + lift])
        segs.append([tl[i] + lift, tl[i + 1] + lift])
    for k in (-0.6, -0.2, 0.2, 0.6):                                       # longitudinal veins
        for i in range(1, WING_STEPS - 1):
            segs.append([[u[i], 0.004, k * w[i]], [u[i + 1], 0.004, k * w[i + 1]]])
    for sc in (0.45, 0.70):                                                # cross veins
        i = int(round(sc * WING_STEPS))
        segs.append([[u[i], 0.004, -w[i]], [u[i], 0.004, w[i]]])
    wire = _lines(np.array(segs), COL_WING_WIRE)
    bases, roots = [], []
    up = np.array([0.0, 1.0, 0.0])
    for sz in (1.0, -1.0):
        d = np.array([WING_DIR[0], WING_DIR[1], WING_DIR[2] * sz])
        d /= np.linalg.norm(d)
        n = up - (up @ d) * d
        n /= np.linalg.norm(n)
        c = np.cross(d, n)
        root = np.array(THORAX_C) + np.array([WING_ROOT[0], WING_ROOT[1], WING_ROOT[2] * sz])
        B = identity().astype(np.float64)
        B[:3, 0], B[:3, 1], B[:3, 2], B[:3, 3] = d, n, c, root
        bases.append(B)
        roots.append(root)
    return Wings(plate=plate, wire=wire, bases=bases, roots=np.array(roots))


class WingsRenderer:
    def __init__(self, ctx, program, wings: Wings):
        self.ctx, self.wings = ctx, wings
        self.plate, self.wire = Mesh(ctx, program, wings.plate), Mesh(ctx, program, wings.wire, mode="line")

    def draw(self, scene, t: float, vm=None, base=None, pose=None, droop: float = 0.0):
        base = translate(*FLY_POS) @ (identity() if pose is None else pose) if base is None else base
        for i, B in enumerate(self.wings.bases):
            M = base @ wing_droop_matrix(self.wings.roots[i], droop) @ B @ rotate_x(flap_angle(vm, t, i))
            self.ctx.depth_mask = False                 # translucent plate: test depth, don't write it
            self.plate.draw(scene, M, lit=False, emissive=0.15, alpha=WING_ALPHA)
            self.ctx.depth_mask = True
            self.wire.draw(scene, M, lit=False, emissive=0.5)

    def release(self):
        self.plate.release()
        self.wire.release()


@dataclasses.dataclass
class Chair:
    solid: np.ndarray            # seat, backrest, post, base, casters (lit)
    trim: np.ndarray             # neon backrest edge tubes (unlit)
    seat: np.ndarray             # seat alone (for tests)
    back: np.ndarray             # backrest panel alone (for tests)


def build_chair() -> Chair:
    cx, cz = CHAIR_XZ
    seat = transform_geo(cylinder_geo(SEAT_R, SEAT_R, SEAT_T, 12, COL_CHAIR_SEAT, flat=True),
                         translate(cx, SEAT_TOP_Y - SEAT_T / 2, cz))
    post_top = SEAT_TOP_Y - SEAT_T
    hub_y = FLOOR_TOP_Y + 0.2
    post = limb_geo((cx, hub_y, cz), (cx, post_top, cz), 0.07, COL_CHAIR_METAL, 8)
    collar = limb_geo((cx, post_top - 0.30, cz), (cx, post_top, cz), 0.11, COL_CHAIR_METAL, 8)
    arms, wheels = [], []
    for k in range(5):
        a = 2.0 * math.pi * k / 5.0
        end = (cx + 0.58 * math.cos(a), FLOOR_TOP_Y + 0.12, cz + 0.58 * math.sin(a))
        arms.append(limb_geo((cx, hub_y, cz), end, 0.055, COL_CHAIR_METAL, 6, 0.045))
        wheels.append(transform_geo(uv_sphere_geo(0.12, 10, 8, COL_CHAIR_WHEEL), translate(end[0], FLOOR_TOP_Y + 0.12, end[2])))
    bmat = translate(BACK_X, SEAT_TOP_Y + 0.75, cz) @ rotate_z(math.radians(BACK_TILT_DEG))
    back = transform_geo(box_geo(*BACK_SIZE, COL_CHAIR_SEAT), bmat)
    trim = transform_geo(wire_box_tubes_geo(*BACK_SIZE, 0.03, COL_CHAIR_TRIM), bmat)
    bar = lambda z: limb_geo((cx - SEAT_R + 0.1, SEAT_TOP_Y - 0.06, z), (BACK_X + 0.05, SEAT_TOP_Y + 0.35, z * 0.9), 0.045, COL_CHAIR_METAL, 6)
    solid = concat_geo(seat, back, post, collar, bar(0.3), bar(-0.3), *arms, *wheels)
    return Chair(solid=solid, trim=trim, seat=seat, back=back)


class ChairRenderer:
    def __init__(self, ctx, program, chair: Chair):
        self.solid, self.trim = Mesh(ctx, program, chair.solid), Mesh(ctx, program, chair.trim)

    def draw(self, scene, t: float, vm=None):
        I = identity()
        self.solid.draw(scene, I)
        self.trim.draw(scene, I, lit=False, emissive=0.5 + 0.15 * math.sin(t * 1.4), fog=0.3)

    def release(self):
        self.solid.release()
        self.trim.release()


# --------------------------------------------------------------------------
# Part 8a: jointed legs (fly-local frame). Five legs rest on the chair seat / desk edge; the sixth
# (front +z) is the ARM, solved with two_bone_ik. 8a draws the arm in its rest pose; 8b drives it.
# --------------------------------------------------------------------------
COL_FLY_LEG = (0.46, 0.46, 0.52)                 # smooth gray
LEG_RADII = (0.055, 0.045, 0.032, 0.022)     # hip, knee, foot, toe
LEG_SEGMENTS = 8
FOOT_Y = 0.05                                # feet rest just above the seat (local y = 0 is the seat top)
LEG_L = (0.60, 0.60)                         # bone lengths, resting legs
ARM_L = (0.85, 0.85)                         # bone lengths, the reaching arm
ARM_HIP = (0.42, 0.85, 0.26)
ARM_BEND = (0.0, 1.0, 0.6)
ARM_REST_TARGET = (1.55, 0.66, 0.45)         # hovering over the macro-pad
# name: (hip, foot, bend, tip_dir)
LEG_SPECS = {
    "hind+": ((-0.20, 0.78, 0.26), (-0.35, FOOT_Y, 0.66), (0.0, 1.0, 0.8), (0.1, 0.0, 0.0)),
    "hind-": ((-0.20, 0.78, -0.26), (-0.35, FOOT_Y, -0.66), (0.0, 1.0, -0.8), (0.1, 0.0, 0.0)),
    "mid+": ((0.05, 0.72, 0.32), (0.15, FOOT_Y, 0.70), (0.0, 1.0, 0.8), (0.1, 0.0, 0.0)),
    "mid-": ((0.05, 0.72, -0.32), (0.15, FOOT_Y, -0.70), (0.0, 1.0, -0.8), (0.1, 0.0, 0.0)),
    "front-": ((0.42, 0.85, -0.26), (0.78, 0.37, -0.55), (0.0, 1.0, -0.5), (0.1, 0.0, 0.0)),   # resting on the desk edge
}


def leg_points(hip, target, l1: float, l2: float, bend, tip_dir=None):
    """[hip, knee, foot, toe] for one leg. tip_dir None: the toe continues the lower bone's direction."""
    knee, foot = two_bone_ik(hip, target, l1, l2, bend)
    if tip_dir is None:
        t = foot - knee
        tip = foot + 0.12 * t / max(float(np.linalg.norm(t)), 1e-9)
    else:
        tip = foot + np.asarray(tip_dir, dtype=np.float64)
    return np.array([np.asarray(hip, dtype=np.float64), knee, foot, tip])


def build_arm(target=ARM_REST_TARGET) -> np.ndarray:
    """Arm geometry for a hand target (fly-local). Same vertex count for every target, so it can be re-uploaded per frame."""
    pts = leg_points(ARM_HIP, target, ARM_L[0], ARM_L[1], ARM_BEND)
    return joint_chain_geo(pts, LEG_RADII, COL_FLY_LEG, 1.25, LEG_SEGMENTS)


@dataclasses.dataclass
class Legs:
    static: np.ndarray           # the five resting legs
    arm: np.ndarray              # the arm in its rest pose
    chains: dict                 # name -> (4,3) points, including "arm"


def build_legs() -> Legs:
    chains = {n: leg_points(h, f, LEG_L[0], LEG_L[1], b, t) for n, (h, f, b, t) in LEG_SPECS.items()}
    static = concat_geo(*[joint_chain_geo(c, LEG_RADII, COL_FLY_LEG, 1.25, LEG_SEGMENTS) for c in chains.values()])
    chains["arm"] = leg_points(ARM_HIP, ARM_REST_TARGET, ARM_L[0], ARM_L[1], ARM_BEND)
    return Legs(static=static, arm=build_arm(), chains=chains)


class LegsRenderer:
    def __init__(self, ctx, program, legs: Legs):
        self.legs = legs
        self.static, self.arm = Mesh(ctx, program, legs.static), Mesh(ctx, program, legs.arm)
        self.model = translate(*FLY_POS)

    def draw(self, scene, t: float, vm=None, pose=None):
        M = self.model @ (identity() if pose is None else pose) @ translate(0, body_bob(vm, t), 0)
        self.static.draw(scene, M)
        if vm is not None:
            self.arm.update(build_arm(arm_target_in_pose(vm, pose)))
        self.arm.draw(scene, M)

    def release(self):
        self.static.release()
        self.arm.release()


# --------------------------------------------------------------------------
# Part 8b: macro-pad console (keys, BUY/SELL lever) + arm driven by the ViewModel
# BUY pulls the lever toward the fly (-x), SELL pushes it back (+x); reach 0..1 blends rest hover -> grip.
# --------------------------------------------------------------------------
PAD_TOP_Y = CONSOLE_CENTER[1] + CONSOLE_SIZE[1] / 2 + PAD_SIZE[1]          # 0.14
LEVER_PLATE_SIZE = (1.0, 0.03, 0.30)
LEVER_BASE_W = (CONSOLE_CENTER[0], PAD_TOP_Y, 0.55)                       # world: centre of the lever plate's underside
LEVER_PIVOT_W = (LEVER_BASE_W[0], PAD_TOP_Y + LEVER_PLATE_SIZE[1], LEVER_BASE_W[2])
LEVER_LEN, KNOB_R, LEVER_MAX_DEG = 0.45, 0.07, 35.0
COL_LEVER_PLATE = (0.07, 0.07, 0.10)
COL_LEVER = (0.62, 0.62, 0.70)
COL_KNOB = (0.92, 0.85, 0.95)
KEY_SIZE = (0.18, 0.05, 0.18)
KEY_XZ = tuple((CONSOLE_CENTER[0] + dx, z) for z in (-0.78, -0.48) for dx in (-0.6, -0.35, -0.10))
KEY_COLORS = ((0.0, 0.95, 1.0), (1.0, 0.20, 0.80), (0.60, 0.40, 1.0), (1.0, 0.85, 0.30), (0.30, 1.0, 0.55), (0.9, 0.9, 1.0))
LED_OFFSET = 0.42


def _cl(x) -> float:
    return float(min(max(x, 0.0), 1.0))


def lever_angle(vm) -> float:
    """Radians from vertical, positive = tipped toward +x (pushed back, SELL); negative = pulled toward the fly (BUY)."""
    return math.radians(LEVER_MAX_DEG) * (_cl(getattr(vm, "lever_sell", 0.0)) - _cl(getattr(vm, "lever_buy", 0.0)))


def lever_tip_world(theta: float) -> np.ndarray:
    return np.array(LEVER_PIVOT_W, dtype=np.float64) + LEVER_LEN * np.array([math.sin(theta), math.cos(theta), 0.0])


def arm_target(vm) -> np.ndarray:
    """Hand target (fly-local): rest hover when reach = 0, gripping the lever knob when reach = 1."""
    r = _cl(getattr(vm, "reach", 0.0))
    grip = lever_tip_world(lever_angle(vm)) - np.array(FLY_POS)
    return np.array(ARM_REST_TARGET, dtype=np.float64) * (1.0 - r) + grip * r


@dataclasses.dataclass
class Console:
    solid: np.ndarray            # lever plate + pivot ball
    lever: np.ndarray            # stick + knob, built about the pivot (rotated at draw time)
    keys: list                   # 6 macro keys, each a box (world frame)
    leds: list                   # [BUY (green, fly side), SELL (red, far side)] boxes


def build_console() -> Console:
    px, py, pz = LEVER_BASE_W
    plate = transform_geo(box_geo(*LEVER_PLATE_SIZE, COL_LEVER_PLATE), translate(px, py + LEVER_PLATE_SIZE[1] / 2, pz))
    ball = transform_geo(uv_sphere_geo(0.055, 12, 8, COL_LEVER), translate(*LEVER_PIVOT_W))
    stick = limb_geo((0, 0, 0), (0, LEVER_LEN, 0), 0.03, COL_LEVER, 8)
    knob = transform_geo(uv_sphere_geo(KNOB_R, 14, 10, COL_KNOB), translate(0, LEVER_LEN, 0))
    keys = [transform_geo(box_geo(*KEY_SIZE, KEY_COLORS[i % len(KEY_COLORS)]), translate(x, PAD_TOP_Y + KEY_SIZE[1] / 2, z))
            for i, (x, z) in enumerate(KEY_XZ)]
    led = lambda dx, c: transform_geo(box_geo(0.09, 0.04, 0.09, c), translate(px + dx, PAD_TOP_Y + 0.02, pz))
    return Console(solid=concat_geo(plate, ball), lever=concat_geo(stick, knob), keys=keys,
                   leds=[led(-LED_OFFSET, COL_GREEN), led(LED_OFFSET, COL_RED)])


class ConsoleRenderer:
    def __init__(self, ctx, program, con: Console):
        self.solid, self.lever = Mesh(ctx, program, con.solid), Mesh(ctx, program, con.lever)
        self.keys = [Mesh(ctx, program, k) for k in con.keys]
        self.leds = [Mesh(ctx, program, l) for l in con.leds]

    def draw(self, scene, t: float, vm=None):
        I = identity()
        self.solid.draw(scene, I)
        theta = lever_angle(vm) if vm is not None else 0.0
        self.lever.draw(scene, translate(*LEVER_PIVOT_W) @ rotate_z(-theta))
        for i, k in enumerate(self.keys):
            k.draw(scene, I, lit=False, emissive=0.15 + 0.15 * math.sin(t * 2.0 + i), fog=0.5)
        buy = _cl(getattr(vm, "lever_buy", 0.0)) if vm is not None else 0.0
        sell = _cl(getattr(vm, "lever_sell", 0.0)) if vm is not None else 0.0
        self.leds[0].draw(scene, I, lit=False, emissive=0.2 + 1.6 * buy, fog=0.5)
        self.leds[1].draw(scene, I, lit=False, emissive=0.2 + 1.6 * sell, fog=0.5)

    def release(self):
        for m in (self.solid, self.lever, *self.keys, *self.leds):
            m.release()


# --------------------------------------------------------------------------
# Part 9a: monitor screen texture path + scrolling candlestick chart
# The screen image is rasterised on the CPU with numpy (own 5x7 bitmap font, no font files, no pygame needed),
# uploaded to a mip-mapped texture, and drawn on the monitor's screen quad with a tiny UV shader.
# The chart is ILLUSTRATIVE: the league places no real trades and has no price feed, so the screen says so.
# --------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 384, 240                 # 16:10 like the screen quad
TITLE_RECT = (0, 0, SCREEN_W, 22)             # x, y, w, h  (y down); 9g: every screen's text is 2x (see TAPE_LIST_RECT)
CHART_RECT = (6, 26, 180, 122)
METRICS_RECT = (192, 26, 186, 122)
STATUS_RECT = (6, 154, 372, 82)
COL_SCR_BG = (6, 10, 18)
COL_SCR_PANEL = (14, 22, 36)
COL_SCR_BORDER = (0, 150, 170)
COL_SCR_TEXT = (0, 230, 245)
COL_SCR_DIM = (70, 110, 130)
COL_CANDLE_UP = (40, 220, 90)
COL_CANDLE_DN = (235, 50, 60)
COL_SCR_GRID = (24, 38, 56)
CANDLE_W, CANDLE_GAP = 4, 1
CHART_LABEL_W = 62                            # right margin reserved for the last-price label
CHART_CANDLES = (CHART_RECT[2] - 3 - CHART_LABEL_W + CANDLE_GAP) // (CANDLE_W + CANDLE_GAP)   # 41
CANDLE_STEP_S = 0.4                           # seconds per new candle
SCREEN_FPS = 15.0

_FONT_ROWS = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    ".": "00000 00000 00000 00000 00000 01100 01100", ",": "00000 00000 00000 00000 01100 00100 01000",
    "-": "00000 00000 00000 11111 00000 00000 00000", "+": "00000 00100 00100 11111 00100 00100 00000",
    ":": "00000 01100 01100 00000 01100 01100 00000", "/": "00001 00010 00010 00100 01000 01000 10000",
    "%": "11001 11010 00010 00100 01000 01011 10011", "[": "01110 01000 01000 01000 01000 01000 01110",
    "]": "01110 00010 00010 00010 00010 00010 01110", "(": "00010 00100 01000 01000 01000 00100 00010",
    ")": "01000 00100 00010 00010 00010 00100 01000", "=": "00000 00000 11111 00000 11111 00000 00000",
    "_": "00000 00000 00000 00000 00000 00000 11111", "!": "00100 00100 00100 00100 00100 00000 00100",
    "$": "00100 01111 10100 01110 00101 11110 00100", " ": "00000 00000 00000 00000 00000 00000 00000",
}
FONT = {ch: np.array([[c == "1" for c in row] for row in spec.split()], dtype=bool) for ch, spec in _FONT_ROWS.items()}


def text_width(text: str, scale: int = 1) -> int:
    return max(0, (6 * len(text) - 1) * scale)


class Canvas:
    """CPU pixel buffer (H, W, 3) uint8; y grows downward. Every helper clips, nothing raises on out-of-range input."""
    def __init__(self, w: int = SCREEN_W, h: int = SCREEN_H, color=COL_SCR_BG):
        self.w, self.h = w, h
        self.px = np.empty((h, w, 3), dtype=np.uint8)
        self.px[:] = color

    def fill(self, color):
        self.px[:] = color

    def rect(self, x, y, w, h, color):
        x0, y0, x1, y1 = max(int(x), 0), max(int(y), 0), min(int(x + w), self.w), min(int(y + h), self.h)
        if x1 > x0 and y1 > y0:
            self.px[y0:y1, x0:x1] = color

    def frame(self, x, y, w, h, color):
        self.rect(x, y, w, 1, color)
        self.rect(x, y + h - 1, w, 1, color)
        self.rect(x, y, 1, h, color)
        self.rect(x + w - 1, y, 1, h, color)

    def text(self, x, y, s: str, color=COL_SCR_TEXT, scale: int = 1):
        for i, ch in enumerate(str(s).upper()):
            g = FONT.get(ch)
            if g is None:
                continue
            gy, gx = np.nonzero(g)
            for dy, dx in zip(gy, gx):
                self.rect(x + (i * 6 + dx) * scale, y + dy * scale, scale, scale, color)


class CandleSeries:
    """Deterministic ILLUSTRATIVE OHLC random walk (not market data). step() appends one candle, drops the oldest."""
    def __init__(self, seed: int = 11, n: int = CHART_CANDLES, start: float = 100.0):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.o = np.zeros(n); self.h = np.zeros(n); self.l = np.zeros(n); self.c = np.zeros(n)
        price = start
        for i in range(n):
            price = self._make(i, price)
        self.last_close = price

    def _make(self, i: int, o: float) -> float:
        drift = self.rng.normal(0.0, 0.8)
        c = max(o + drift, 1.0)
        self.o[i], self.c[i] = o, c
        self.h[i] = max(o, c) + abs(self.rng.normal(0.0, 0.45))
        self.l[i] = max(min(o, c) - abs(self.rng.normal(0.0, 0.45)), 0.5)
        return c

    def step(self):
        for a in (self.o, self.h, self.l, self.c):
            a[:-1] = a[1:]
        self.last_close = self._make(self.n - 1, self.c[-2])


NO_DATE = "--/--/----"
TRADE_DATE_KEYS = ("trade_date", "date", "as_of", "asof", "decision_date", "current_date", "sim_date", "day")


def format_trade_date(raw) -> str:
    """Any plausible date (date/datetime/Timestamp, 'YYYY-MM-DD[...]', 'YYYY/MM/DD', 'YYYYMMDD', 20251103) -> 'dd/mm/yyyy'.
    Anything else (None, junk, impossible dates, the ambiguous dd/mm vs mm/dd forms) -> '' so the screens show --/--/----."""
    import datetime as _dt
    try:
        if raw is None or isinstance(raw, bool):
            return ""
        if hasattr(raw, "year") and hasattr(raw, "month") and hasattr(raw, "day"):
            d = _dt.date(int(raw.year), int(raw.month), int(raw.day))
        else:
            txt = str(raw).strip()
            m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?$", txt) or re.match(r"^(\d{4})(\d{2})(\d{2})$", txt)
            if not m:
                return ""
            d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if not 1000 <= d.year <= 9999:
            return ""
        return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
    except (ValueError, TypeError, OverflowError):
        return ""


def extract_trade_date(header, view=None) -> str:
    """The date the fly is trading on, from the header (or an attribute on the view); '' if the feed does not say."""
    hd = header if isinstance(header, dict) else {}
    for k in TRADE_DATE_KEYS:
        out = format_trade_date(hd.get(k))
        if out:
            return out
    for k in TRADE_DATE_KEYS:
        out = format_trade_date(getattr(view, k, None)) if view is not None else ""
        if out:
            return out
    return ""


def date_txt(vm) -> str:
    return str(getattr(vm, "trade_date", "") or "") or NO_DATE


def render_chart(cv: Canvas, series: CandleSeries, ticker: str = "", date: str = ""):
    """Title bar + candlestick panel. Draws only inside TITLE_RECT and CHART_RECT."""
    cv.rect(*TITLE_RECT, COL_SCR_PANEL)
    cv.text(6, 4, (ticker or "----")[:6], COL_SCR_TEXT, 2)
    dtxt = date or NO_DATE
    dx = SCREEN_W - 6 - text_width(dtxt, 2)
    cv.text(dx, 4, dtxt, COL_SCR_VALUE, 2)                                   # the day the fly is trading, dd/mm/yyyy
    tag = "ILLUSTRATIVE DATA"
    cv.text(dx - 8 - text_width(tag), 8, tag, (255, 200, 60))
    x, y, w, h = CHART_RECT
    cv.rect(x, y, w, h, COL_SCR_PANEL)
    for k in range(1, 5):
        cv.rect(x, y + k * h // 5, w, 1, COL_SCR_GRID)
    cv.frame(x, y, w, h, COL_SCR_BORDER)
    lo, hi = float(series.l.min()), float(series.h.max())
    pad = (hi - lo) * 0.06 + 1e-9
    lo, hi = lo - pad, hi + pad
    inner_y, inner_h = y + 2, h - 4
    ypx = lambda v: inner_y + int(round((hi - v) / (hi - lo) * (inner_h - 1)))
    x0 = x + 3
    for i in range(series.n):
        cx = x0 + i * (CANDLE_W + CANDLE_GAP)
        if cx + CANDLE_W > x + w - CHART_LABEL_W:
            break
        up = series.c[i] >= series.o[i]
        col = COL_CANDLE_UP if up else COL_CANDLE_DN
        yo, yc, yh, yl = ypx(series.o[i]), ypx(series.c[i]), ypx(series.h[i]), ypx(series.l[i])
        cv.rect(cx + CANDLE_W // 2, yh, 1, yl - yh + 1, col)                      # wick
        cv.rect(cx, min(yo, yc), CANDLE_W, abs(yo - yc) + 1, col)                  # body
    ylast = ypx(series.c[-1])
    cv.rect(x + 1, ylast, w - 2, 1, COL_SCR_DIM)
    label = f"{series.c[-1]:.1f}"
    cv.text(x + w - 3 - text_width(label, 2), max(y + 2, min(y + h - 16, ylast - 7)), label, COL_SCR_TEXT, 2)


COL_SCR_VALUE = (225, 235, 245)
COL_SCR_AMBER = (255, 200, 60)
METRIC_ROW_H = 16
METRIC_COLS = 15                               # characters per grid row at 2x (12 px each)
METRIC_ABBR = {"Revenues": "REV", "NetIncomeLoss": "NI", "Assets": "ASSETS", "StockholdersEquity": "EQUITY",
               "OperatingIncomeLoss": "OPINC", "GrossProfit": "GP", "Liabilities": "LIAB", "CashAndCashEquivalents": "CASH"}
SCROLL_ROWS_PER_S = 3.0
BLINK_HZ = 2.0
SILENT_DIM = 0.30
SILENT_SEED = 5
_METRIC_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(-?\d+(?:\.\d+)?)")


def compact_number(v: float) -> str:
    a = abs(v)
    for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= lim:
            return f"{v / lim:.0f}{suf}" if a / lim >= 100 else f"{v / lim:.1f}{suf}"        # <= 5 chars: 123M, 12.3M
    return f"{v:.0f}"


def abbreviate_tag(tag: str) -> str:
    if tag in METRIC_ABBR:
        return METRIC_ABBR[tag]
    caps = "".join(ch for ch in tag if ch.isupper())
    return (caps if len(caps) >= 2 else tag)[:7]


def format_metric_row(line: str) -> str:
    """'AAPL   Revenues   123456789  FY2025 ...' -> 'AAPL  REV 123.5M' (<= METRIC_COLS chars).
    Lines that don't look like <ticker> <tag> <number> are shown raw, truncated."""
    m = _METRIC_RE.match(line)
    if not m:
        return line.strip()[:METRIC_COLS]
    tk, tag, val = m.group(1)[:5], abbreviate_tag(m.group(2)), compact_number(float(m.group(3)))
    room = METRIC_COLS - len(tk) - len(val) - 2
    return f"{tk} {tag[:max(room, 1)]:<{max(room, 1)}} {val}"[:METRIC_COLS].rstrip()


def render_metrics(cv: Canvas, lines, scroll: float, live: bool):
    x, y, w, h = METRICS_RECT
    cv.rect(x, y, w, h, COL_SCR_PANEL)
    cv.frame(x, y, w, h, COL_SCR_BORDER)
    cv.text(x + 4, y + 3, "XBRL DATA", COL_SCR_TEXT if live else COL_SCR_DIM, 2)
    cv.rect(x + 3, y + 20, w - 6, 1, COL_SCR_GRID)
    rows = (h - 24) // METRIC_ROW_H
    if not lines:
        cv.text(x + 4, y + 25, "NO DATA", COL_SCR_DIM, 2)
        return
    base = int(scroll)
    for i in range(rows):
        row = format_metric_row(str(lines[(base + i) % len(lines)]))
        col = COL_SCR_VALUE if live else COL_SCR_DIM
        if live and i == 0:
            col = COL_SCR_AMBER                      # the line being parsed right now
        cv.text(x + 4, y + 25 + i * METRIC_ROW_H, row, col, 2)


STATE_STATUS = {"BUY": "BUY ORDER {tk}", "SELL": "SELL ORDER {tk}", "HOLD": "TRACKING MARKET FLOW {tk}",
                "IDLE": "STANDING BY", "SILENT": "AGENT SILENT - CULL FILTER"}


def render_status(cv: Canvas, vm, t: float):
    x, y, w, h = STATUS_RECT
    cv.rect(x, y, w, h, COL_SCR_PANEL)
    cv.frame(x, y, w, h, COL_SCR_BORDER)
    state = getattr(vm, "state", "IDLE")
    if state == "RESEARCH":
        if int(t * BLINK_HZ * 2) % 2 == 0:                                         # flashing
            cv.text(x + 8, y + 8, "CRAWLING SEC EDGAR...", COL_SCR_AMBER, 2)
        cv.text(x + 8, y + 30, str(getattr(vm, "research_status", ""))[:(w - 16) // 12], COL_SCR_TEXT, 2)
        bar_w = 60
        bx = x + 4 + int((t * 90.0) % (w - 8 - bar_w))
        cv.rect(x + 4, y + h - 14, w - 8, 6, COL_SCR_GRID)
        cv.rect(bx, y + h - 14, bar_w, 6, COL_SCR_TEXT)
    else:
        tk = getattr(vm, "ticker", "") or ""
        msg = STATE_STATUS.get(state, state).format(tk=tk).strip()
        col = {"BUY": COL_CANDLE_UP, "SELL": COL_CANDLE_DN, "SILENT": COL_SCR_DIM}.get(state, COL_SCR_TEXT)
        cv.text(x + 8, y + 10, msg[:(w - 16) // 12], col, 2)
        cv.text(x + 8, y + 36, "ILLUSTRATIVE", COL_SCR_DIM, 2)
        cv.text(x + 8, y + 54, "NO LIVE TRADES", COL_SCR_DIM, 2)


def apply_silent(cv: Canvas):
    """SILENT: dim screen with a static horizontal scanline pattern (fixed per-row brightness, no animation)."""
    rng = np.random.default_rng(SILENT_SEED)
    rowgain = rng.uniform(0.75, 1.25, cv.h)
    rowgain[::2] *= 0.5
    cv.px[:] = np.clip(cv.px.astype(np.float32) * (SILENT_DIM * rowgain)[:, None, None], 0, 255).astype(np.uint8)



class ScreenContent:
    """Pure-numpy screen state machine (no GL): owns the canvas and series, redraws at SCREEN_FPS."""
    def __init__(self, seed: int = 11):
        self.series = CandleSeries(seed)
        self.canvas = Canvas()
        self._last_step = 0.0
        self._last_draw = -1e9
        self._sig = None
        self._prev_t = None
        self.scroll = 0.0
        self.draws = 0

    def update(self, vm, t: float) -> bool:
        state = getattr(vm, "state", "IDLE")
        dt = max(t - self._prev_t, 0.0) if self._prev_t is not None else 0.0
        self._prev_t = t
        if state == "RESEARCH":
            self.scroll += dt * SCROLL_ROWS_PER_S                                       # metrics only crawl while researching
        if state != "SILENT" and t - self._last_step >= CANDLE_STEP_S:
            self._last_step = t
            self.series.step()
        sig = (state, getattr(vm, "ticker", ""), date_txt(vm))
        if t - self._last_draw < 1.0 / SCREEN_FPS and sig == self._sig:
            return False
        self._last_draw, self._sig = t, sig
        self.canvas.fill(COL_SCR_BG)
        render_chart(self.canvas, self.series, getattr(vm, "ticker", ""), date_txt(vm))
        render_metrics(self.canvas, getattr(vm, "research_lines", None) or [], self.scroll, state == "RESEARCH")
        render_status(self.canvas, vm, t)
        if state == "SILENT":
            apply_silent(self.canvas)
        self.draws += 1
        return True



# --------------------------------------------------------------------------
# Part 9d: side monitor = decision tape (recent decisions and their MBON scores, newest first)
# --------------------------------------------------------------------------
# 9g: all text on the side screens is drawn at scale 2 (10 x 14 px glyphs): at fullscreen the screen is only ~265 px wide on a
# 1080p display, so 1x glyphs (5 x 7 texels) came out ~3 px wide, unreadable.
TITLE2_RECT = TITLE_RECT
TAPE_LIST_RECT = (6, 26, 372, 168)
TAPE_FOOT_RECT = (6, 198, 372, 38)
TAPE_ROW_H = 17
TAPE_ROWS = (TAPE_LIST_RECT[3] - 6) // TAPE_ROW_H          # 9
TAPE_BAR_CX = 262                                          # zero line of the score bars
TAPE_BAR_HALF = 104


def _fnum(v, fmt: str) -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "--"


def render_tape(cv: Canvas, vm):
    cv.fill(COL_SCR_BG)
    cv.rect(*TITLE2_RECT, COL_SCR_PANEL)
    cv.text(6, 4, "DECISION TAPE", COL_SCR_TEXT, 2)
    lab = "MBON HZ"
    cv.text(SCREEN_W - 6 - text_width(lab, 2), 4, lab, COL_SCR_DIM, 2)
    x, y, w, h = TAPE_LIST_RECT
    cv.rect(x, y, w, h, COL_SCR_PANEL)
    cv.frame(x, y, w, h, COL_SCR_BORDER)
    tape = list(getattr(vm, "tape", None) or [])[-TAPE_ROWS:][::-1]            # newest first
    cv.rect(TAPE_BAR_CX, y + 2, 1, h - 4, COL_SCR_GRID)
    if not tape:
        cv.text(x + 8, y + 8, "WAITING FOR", COL_SCR_DIM, 2)
        cv.text(x + 8, y + 26, "DECISIONS...", COL_SCR_DIM, 2)
    scores = []
    for _, sc in tape:
        try:
            scores.append(float(sc))
        except (TypeError, ValueError):
            scores.append(0.0)
    top = max([abs(v) for v in scores] + [1e-9])
    for i, ((tk, _), sc) in enumerate(zip(tape, scores)):
        ry = y + 4 + i * TAPE_ROW_H
        col = COL_CANDLE_UP if sc >= 0 else COL_CANDLE_DN
        cv.text(x + 8, ry + 1, str(tk)[:5], COL_SCR_VALUE if i else COL_SCR_AMBER, 2)
        cv.text(x + 74, ry + 1, f"{sc:+.1f}", COL_SCR_VALUE, 2)
        bw = max(1, int(round(TAPE_BAR_HALF * abs(sc) / top)))
        cv.rect(TAPE_BAR_CX if sc >= 0 else TAPE_BAR_CX - bw + 1, ry + 4, bw, 9, col)
    fx, fy, fw, fh = TAPE_FOOT_RECT
    cv.rect(fx, fy, fw, fh, COL_SCR_PANEL)
    cv.frame(fx, fy, fw, fh, COL_SCR_BORDER)
    hd = getattr(vm, "header", None) or {}
    cv.text(fx + 6, fy + 4, f"{str(hd.get('top_id', '--'))[:8]} IC {_fnum(hd.get('top_ic'), '+.3f')}", COL_SCR_TEXT, 2)
    cv.text(fx + 6, fy + 21, f"RIDGE {_fnum(hd.get('ridge_ic'), '+.3f')}", COL_SCR_VALUE, 2)
    dtxt = date_txt(vm)
    cv.text(fx + fw - 6 - text_width(dtxt, 2), fy + 21, dtxt, COL_SCR_VALUE, 2)  # trading date, dd/mm/yyyy
    note = str(getattr(vm, "replay_label", "") or "").upper()[:20]
    if note:                                                                   # honest source label, e.g. SIMULATED FEED / REPLAY
        cv.text(fx + fw - 6 - text_width(note), fy + 8, note, COL_SCR_DIM)


class TapeContent:
    """Side-monitor screen. Redraws only when the tape / header / state changes (cheap)."""
    def __init__(self):
        self.canvas = Canvas()
        self._sig = None
        self.draws = 0

    def update(self, vm, t: float) -> bool:
        tape = list(getattr(vm, "tape", None) or [])
        hd = getattr(vm, "header", None) or {}
        sig = (len(tape), tape[-1] if tape else None, hd.get("top_id"), hd.get("top_ic"), hd.get("ridge_ic"), getattr(vm, "state", ""), date_txt(vm))
        if sig == self._sig:
            return False
        self._sig = sig
        render_tape(self.canvas, vm)
        if getattr(vm, "state", "") == "SILENT":
            apply_silent(self.canvas)
        self.draws += 1
        return True


# --------------------------------------------------------------------------
# Part 9f-a: IC history screen (third monitor). Reads vm.ic_history = [(round, top_ic, ridge_ic, untrained_ic), ...]
# (9f-b fills it from the league). Every number is a real IC from the league, or the screen says it is waiting.
# --------------------------------------------------------------------------
IC_PLOT_RECT = (6, 26, 372, 146)
IC_FOOT_RECT = (6, 176, 372, 60)
IC_LABEL_W = 66                               # left margin of the plot for the y labels
IC_MAX_POINTS = 60                            # newest rounds shown
IC_AVG_N = 3                                  # the rolling average the HUD uses
COL_IC_TOP = (0, 230, 245)
COL_IC_AVG = (255, 200, 60)
COL_IC_RIDGE = (190, 120, 255)
COL_IC_UNTR = (120, 135, 150)
COL_IC_ZERO = (60, 100, 120)


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rolling_mean(vals, n: int = IC_AVG_N):
    """Trailing mean over up to n values; None (missing) entries are skipped; a window with no data gives None."""
    out = []
    for i in range(len(vals)):
        w = [v for v in (_finite(x) for x in vals[max(0, i - n + 1):i + 1]) if v is not None]
        out.append(sum(w) / len(w) if w else None)
    return out


def _clean_history(hist):
    rows = []
    for r in list(hist or [])[-IC_MAX_POINTS:]:
        try:
            rows.append((r[0], _finite(r[1]), _finite(r[2]), _finite(r[3])))
        except (TypeError, IndexError, KeyError):
            continue
    return rows


def _line(cv: Canvas, pts, color, thick: int = 1):
    """Polyline through (x, y) pixel points, clipped by Canvas.rect; None breaks the line."""
    prev = None
    for p in pts:
        if p is None:
            prev = None
            continue
        if prev is not None:
            n = int(max(abs(p[0] - prev[0]), abs(p[1] - prev[1]))) + 1
            for t in np.linspace(0.0, 1.0, n + 1):
                cv.rect(round(prev[0] + (p[0] - prev[0]) * t), round(prev[1] + (p[1] - prev[1]) * t), thick, thick, color)
        else:
            cv.rect(round(p[0]), round(p[1]), thick, thick, color)
        prev = p


def render_ic(cv: Canvas, vm):
    cv.fill(COL_SCR_BG)
    cv.rect(*TITLE2_RECT, COL_SCR_PANEL)
    cv.text(6, 4, "IC HISTORY", COL_SCR_TEXT, 2)
    lab = "3 RD AVG"
    cv.text(SCREEN_W - 6 - text_width(lab, 2), 4, lab, COL_IC_AVG, 2)
    x, y, w, h = IC_PLOT_RECT
    cv.rect(x, y, w, h, COL_SCR_PANEL)
    cv.frame(x, y, w, h, COL_SCR_BORDER)
    fx, fy, fw, fh = IC_FOOT_RECT
    cv.rect(fx, fy, fw, fh, COL_SCR_PANEL)
    cv.frame(fx, fy, fw, fh, COL_SCR_BORDER)
    rows = _clean_history(getattr(vm, "ic_history", None))
    top = [r[1] for r in rows]
    avg = rolling_mean(top)
    ridge = [r[2] for r in rows]
    untr = [r[3] for r in rows]
    series = (("TOP", top, COL_IC_TOP, 1), ("AVG", avg, COL_IC_AVG, 2), ("RIDGE", ridge, COL_IC_RIDGE, 1), ("UNTR", untr, COL_IC_UNTR, 1))
    vals = [v for _, ss, _, _ in series for v in ss if v is not None]
    if not vals:
        cv.text(x + 8, y + 8, "STANDBY", COL_SCR_DIM, 2)
        cv.text(x + 8, y + 26, "WAITING FOR ROUNDS", COL_SCR_DIM, 2)
        cv.text(fx + 6, fy + 4, "NO IC HISTORY YET", COL_SCR_DIM, 2)
    else:
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = max((hi - lo) * 0.12, 1e-3)
        lo, hi = lo - pad, hi + pad
        px0, px1 = x + IC_LABEL_W, x + w - 6
        py0, py1 = y + 8, y + h - 8
        ypix = lambda v: py1 - (v - lo) / (hi - lo) * (py1 - py0)
        n = len(rows)
        xpix = (lambda i: (px0 + px1) / 2) if n == 1 else (lambda i: px0 + i * (px1 - px0) / (n - 1))
        zy = int(round(ypix(0.0)))
        for g in np.linspace(lo, hi, 5)[1:-1]:                                  # faint horizontal grid + y labels
            gy = int(round(ypix(g)))
            if abs(gy - zy) < 18 or gy - 7 < y + 2 or gy + 7 > y + h - 2:       # would collide with the zero label / frame
                continue
            cv.rect(px0, gy, px1 - px0, 1, COL_SCR_GRID)
            cv.text(x + 4, gy - 7, f"{g:+.2f}", COL_SCR_DIM, 2)
        cv.rect(px0, zy, px1 - px0, 1, COL_IC_ZERO)                             # IC = 0 line
        cv.text(x + 4, zy - 7, "0", COL_SCR_TEXT, 2)
        for _, ss, col, th in series[::-1]:                                     # avg drawn on top
            _line(cv, [None if v is None else (xpix(i), ypix(v)) for i, v in enumerate(ss)], col, th + 1)
        last = rows[-1]
        cv.text(fx + 6, fy + 4, f"TOP {_fnum(last[1], '+.3f')}", COL_IC_TOP, 2)
        cv.text(fx + 196, fy + 4, f"AVG {_fnum(avg[-1], '+.3f')}", COL_IC_AVG, 2)
        cv.text(fx + 6, fy + 22, f"RIDGE {_fnum(last[2], '+.3f')}", COL_IC_RIDGE, 2)
        cv.text(fx + 196, fy + 22, f"UNTR {_fnum(last[3], '+.3f')}", COL_IC_UNTR, 2)
        cv.text(fx + 6, fy + 40, f"ROUND {str(last[0])[:6]}", COL_SCR_VALUE, 2)
    dtxt = date_txt(vm)
    cv.text(fx + fw - 6 - text_width(dtxt, 2), fy + 40, dtxt, COL_SCR_VALUE, 2)    # trading date, dd/mm/yyyy
    note = str(getattr(vm, "replay_label", "") or "").upper()[:20]
    if note:
        cv.text(fx + 112, fy + 44, note, COL_SCR_DIM)


class IcContent:
    """Third-monitor screen. Redraws only when the history / state changes."""
    def __init__(self):
        self.canvas = Canvas()
        self._sig = None
        self.draws = 0

    def update(self, vm, t: float) -> bool:
        hist = list(getattr(vm, "ic_history", None) or [])
        sig = (len(hist), repr(hist[-1]) if hist else None, getattr(vm, "state", ""), getattr(vm, "replay_label", ""), date_txt(vm))
        if sig == self._sig:
            return False
        self._sig = sig
        render_ic(self.canvas, vm)
        if getattr(vm, "state", "") == "SILENT":
            apply_silent(self.canvas)
        self.draws += 1
        return True


SCR_VERT = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
SCR_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform vec3 u_tint;
uniform float u_glow;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb * u_tint;
    f_color = vec4(c * (1.0 + u_glow), 1.0);
}
"""


def screen_quad_geo(quad) -> np.ndarray:
    """(6, 5) float32 [x y z u v] for the screen quad given TL, TR, BR, BL corners.
    Image row 0 (top) is texture v = 0, so TL has v = 0. Triangles are CCW seen from +z."""
    tl, tr, br, bl = [np.asarray(c, dtype=np.float32) for c in quad]
    uv = {"tl": (0, 0), "tr": (1, 0), "br": (1, 1), "bl": (0, 1)}
    v = [(bl, uv["bl"]), (br, uv["br"]), (tr, uv["tr"]), (bl, uv["bl"]), (tr, uv["tr"]), (tl, uv["tl"])]
    return np.array([[*p, *t] for p, t in v], dtype=np.float32)


class ScreenRenderer:
    """Monitor screen: textured quad. Content comes from ScreenContent; the texture is re-uploaded only when it changed."""
    def __init__(self, ctx, quad, model: np.ndarray, content=None, react: bool = True):
        import moderngl
        self.content = content or ScreenContent()
        self.model = model
        self.react = react
        self.prog = ctx.program(vertex_shader=SCR_VERT, fragment_shader=SCR_FRAG)
        self.vbo = ctx.buffer(screen_quad_geo(quad).tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 2f", "in_pos", "in_uv")])
        self.tex = ctx.texture((SCREEN_W, SCREEN_H), 3, self.content.canvas.px.tobytes())
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            self.tex.anisotropy = 8.0                      # the screen is seen at a steep angle
        except Exception:
            pass
        self.tex.build_mipmaps()

    def draw(self, scene, t: float, vm=None):
        if vm is not None and self.content.update(vm, t):
            self.tex.write(self.content.canvas.px.tobytes())
            self.tex.build_mipmaps()
        tint, glow = (1.0, 1.0, 1.0), 0.0
        if vm is not None and self.react:
            tint = {"BUY": (0.55, 1.0, 0.6), "SELL": (1.0, 0.5, 0.5), "SILENT": (0.7, 0.7, 0.8)}.get(vm.state, (1.0, 1.0, 1.0))
            glow = 0.8 * getattr(vm, "flash", 0.0)
        self.tex.use(0)
        self.prog["u_tex"].value = 0
        self.prog["u_mvp"].write(gl_bytes(scene.proj_view @ self.model))
        self.prog["u_tint"].value = tuple(tint)
        self.prog["u_glow"].value = float(glow)
        self.vao.render()

    def release(self):
        for o in (self.vao, self.vbo, self.tex, self.prog):
            o.release()


# --------------------------------------------------------------------------
# Part 10a: HUD top bar (screen-space): round, active universe, top fly id, rolling 3-round average IC.
# Rasterised on the CPU like the monitor screens, drawn as a clip-space quad across the top of the window, sized so one
# texel is a square number of window pixels at any aspect ratio. Nothing here is invented: missing data shows "--".
# --------------------------------------------------------------------------
HUD_W, HUD_H = 960, 28
HUD_CELL_W = HUD_W // 4
COL_HUD_BG = (4, 8, 16)


def hud_quad_ndc(aspect: float):
    """TL, TR, BR, BL corners in clip space; the bar is HUD_W:HUD_H on screen, full width, glued to the top edge."""
    h = 2.0 * float(aspect) * HUD_H / HUD_W
    h = min(h, 0.6)                                                      # never swallow the view on tall windows
    return [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0 - h, 0.0), (-1.0, 1.0 - h, 0.0)]


def hud_values(vm) -> dict:
    hd = getattr(vm, "header", None) or {}
    rows = _clean_history(getattr(vm, "ic_history", None))
    avg = rolling_mean([r[1] for r in rows])[-1] if rows else None
    rnd = hd.get("round")
    uni = hd.get("universe")
    return dict(round="--" if rnd is None else str(rnd)[:8], universe="--" if uni is None else str(uni)[:8],
                top_id=str(hd.get("top_id") or "--")[:14], avg_ic=avg)


def render_hud(cv: Canvas, vm):
    cv.fill(COL_HUD_BG)
    v = hud_values(vm)
    avg = v["avg_ic"]
    avg_txt = "--" if avg is None else f"{avg:+.3f}"
    avg_col = COL_SCR_DIM if avg is None else (COL_CANDLE_UP if avg >= 0 else COL_CANDLE_DN)
    cells = (("ROUND", v["round"], COL_SCR_VALUE), ("ACTIVE UNIVERSE", v["universe"], COL_SCR_VALUE),
             ("TOP FLY", v["top_id"], COL_SCR_AMBER), ("3 ROUND AVG IC", avg_txt, avg_col))
    for i, (label, value, col) in enumerate(cells):
        x0 = i * HUD_CELL_W
        if i:
            cv.rect(x0, 3, 1, HUD_H - 8, COL_SCR_GRID)
        cv.text(x0 + 10, 3, label, COL_SCR_DIM)
        cv.text(x0 + 10, 12, value, col, 2)
    cv.rect(0, HUD_H - 1, HUD_W, 1, COL_SCR_BORDER)


class HudContent:
    """Redraws only when a shown value changes."""
    def __init__(self):
        self.canvas = Canvas(HUD_W, HUD_H, COL_HUD_BG)
        self._sig = None
        self.draws = 0

    def update(self, vm, t: float = 0.0) -> bool:
        v = hud_values(vm)
        sig = (v["round"], v["universe"], v["top_id"], None if v["avg_ic"] is None else round(v["avg_ic"], 3))
        if sig == self._sig:
            return False
        self._sig = sig
        render_hud(self.canvas, vm)
        self.draws += 1
        return True


OVL_FRAG = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(u_tex, v_uv); }
"""


class HudRenderer:
    """Screen-space overlay quad. content.canvas.px is RGB (HUD), or content.pixels() gives RGBA (log panel, alpha-blended)."""
    def __init__(self, ctx, content=None, quad_fn=None, size=(HUD_W, HUD_H), rgba: bool = False):
        import moderngl
        self.ctx = ctx
        self.content = content or HudContent()
        self.quad_fn = quad_fn or hud_quad_ndc
        self.rgba = rgba
        self.prog = ctx.program(vertex_shader=SCR_VERT, fragment_shader=OVL_FRAG if rgba else SCR_FRAG)
        self.vbo = ctx.buffer(screen_quad_geo(self.quad_fn(16 / 9)).tobytes(), dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 2f", "in_pos", "in_uv")])
        self.tex = ctx.texture(tuple(size), 4 if rgba else 3, self._pixels().tobytes())
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)                   # crisp pixels, no blur
        self._aspect = 16 / 9
        self._moderngl = moderngl

    def _pixels(self):
        return self.content.pixels() if self.rgba else self.content.canvas.px

    def draw(self, aspect: float, vm):
        if self.content.update(vm):
            self.tex.write(self._pixels().tobytes())
        if abs(aspect - self._aspect) > 1e-6:
            self._aspect = aspect
            self.vbo.write(screen_quad_geo(self.quad_fn(aspect)).tobytes())
        self.ctx.disable(self._moderngl.DEPTH_TEST)
        self.tex.use(0)
        self.prog["u_tex"].value = 0
        self.prog["u_mvp"].write(gl_bytes(np.eye(4, dtype=np.float32)))
        if not self.rgba:
            self.prog["u_tint"].value = (1.0, 1.0, 1.0)
            self.prog["u_glow"].value = 0.0
        self.vao.render()
        self.ctx.enable(self._moderngl.DEPTH_TEST)

    def release(self):
        for o in (self.vao, self.vbo, self.tex, self.prog):
            o.release()


# --------------------------------------------------------------------------
# Part 10b: log stream (screen-space, bottom-left) + floating "Zzz" (3D, over the fly's head while SILENT)
# The log shows the lines the feed already wrote in the spec's wording; nothing is made up here.
# --------------------------------------------------------------------------
LOG_W, LOG_H = 216, 94                         # texels; one texel is W/960 of the window width (like the HUD)
LOG_ROW_H = 9
LOG_ROWS = (LOG_H - 4) // LOG_ROW_H            # 10
LOG_COLS = 34                                  # characters per row (6 px each)
LOG_MARGIN = 0.02                              # clip-space gap to the left window edge
LOG_BOTTOM = 0.11                              # clip-space gap above the bottom edge: clears a Windows taskbar drawn over the window (~4.5%)
COL_LOG_BG = (4, 8, 16)
LOG_BG_ALPHA = 150


def log_quad_ndc(aspect: float):
    w = 2.0 * LOG_W / HUD_W
    h = min(2.0 * float(aspect) * LOG_H / HUD_W, 0.6)
    x0, y0 = -1.0 + LOG_MARGIN, -1.0 + LOG_BOTTOM
    return [(x0, y0 + h, 0.0), (x0 + w, y0 + h, 0.0), (x0 + w, y0, 0.0), (x0, y0, 0.0)]


def wrap_log(text, cols: int = LOG_COLS) -> list:
    """Word-wrap to `cols` characters; continuation rows are indented two spaces. Never loses a character of the text."""
    words = str(text).split()
    rows, cur = [], ""
    for w in words:
        while len(w) > cols - (2 if rows else 0):                          # a single over-long word is split
            room = cols - (2 if rows else 0) - len(cur) - (1 if cur else 0)
            if room <= 0:
                rows.append(("  " if rows else "") + cur)
                cur = ""
                continue
            cur = (cur + " " if cur else "") + w[:room]
            w = w[room:]
            rows.append(("  " if rows else "") + cur)
            cur = ""
        lim = cols - (2 if rows else 0)
        if cur and len(cur) + 1 + len(w) > lim:
            rows.append(("  " if rows else "") + cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur or not rows:
        rows.append(("  " if rows else "") + cur)
    return rows


def log_rows(log, max_rows: int = LOG_ROWS) -> list:
    """[(rgb, row_text, rank)] oldest first, newest last. Whole messages only, newest messages win, rank 0 = newest."""
    msgs = []
    for e in list(log or []):
        try:
            rgb, text = e
            rgb = tuple(max(0, min(255, int(c))) for c in rgb)[:3]
            if len(rgb) != 3:
                raise ValueError
            msgs.append((rgb, str(text)))
        except (TypeError, ValueError):
            continue
    out, used = [], 0
    for rank, (rgb, text) in enumerate(reversed(msgs)):
        rows = wrap_log(text)
        if used + len(rows) > max_rows:
            break
        out = [(rgb, r, rank) for r in rows] + out
        used += len(rows)
    return out


def render_log(cv: Canvas, vm) -> bool:
    """Draws the rows; returns False (and leaves the canvas empty) when there is nothing to show."""
    cv.fill(COL_LOG_BG)
    rows = log_rows(getattr(vm, "log", None))
    if not rows:
        return False
    top = LOG_H - 2 - len(rows) * LOG_ROW_H
    for i, (rgb, text, rank) in enumerate(rows):
        fade = 1.0 if rank == 0 else max(0.45, 1.0 - 0.12 * rank)
        col = tuple(int(c * fade) for c in rgb)
        cv.text(4, top + i * LOG_ROW_H + 1, text, col)
    return True


class LogContent:
    def __init__(self):
        self.canvas = Canvas(LOG_W, LOG_H, COL_LOG_BG)
        self._sig = None
        self._has = False
        self.draws = 0

    def update(self, vm, t: float = 0.0) -> bool:
        sig = tuple(repr(e) for e in list(getattr(vm, "log", None) or [])[-LOG_ROWS:])
        if sig == self._sig:
            return False
        self._sig = sig
        self._has = render_log(self.canvas, vm)
        self.draws += 1
        return True

    def pixels(self) -> np.ndarray:
        """RGBA: panel background is translucent, text opaque; fully transparent when there is no log yet."""
        px = self.canvas.px
        a = np.where((px != np.array(COL_LOG_BG, dtype=np.uint8)).any(2), 255, LOG_BG_ALPHA if self._has else 0).astype(np.uint8)
        return np.dstack([px, a])


ZZZ_COUNT = 3
ZZZ_PERIOD = 1.6                               # seconds for one Z to drift up and fade
ZZZ_STAGGER = 0.55
COL_ZZZ = (0.72, 0.86, 1.0)
ZZZ_ORIGIN = (FLY_POS[0] + HEAD_C[0], FLY_POS[1] + HEAD_C[1] + 0.55, FLY_POS[2] + HEAD_C[2])   # just above the head


def zzz_params(zzz: float) -> list:
    """Up to three Zs, staggered, each drifting up and sideways, growing, fading in and out. zzz = seconds into SILENT."""
    out = []
    try:
        z = float(zzz)
    except (TypeError, ValueError):
        return out
    if not math.isfinite(z) or z <= 0.0:
        return out
    for i in range(ZZZ_COUNT):
        age = z - i * ZZZ_STAGGER
        if age < 0.0:
            continue
        ph = (age % ZZZ_PERIOD) / ZZZ_PERIOD
        out.append(dict(offset=(0.10 * ph, 0.9 * ph, 0.18 * math.sin(ph * math.pi * 2.0)),
                        size=0.16 + 0.16 * ph, alpha=math.sin(ph * math.pi) ** 0.8 * 0.95))
    return out


def z_glyph_geo(color=COL_ZZZ) -> np.ndarray:
    """A 'Z' in the XY plane (front faces +z): top bar, bottom bar and the diagonal, unit size (1 wide, 1 tall)."""
    bar = 0.17
    top = transform_geo(box_geo(1.0, bar, 0.06, color), translate(0.0, 0.5 - bar / 2, 0.0))
    bot = transform_geo(box_geo(1.0, bar, 0.06, color), translate(0.0, -0.5 + bar / 2, 0.0))
    dx, dy = 1.0 - 2 * 0.09 - 0.0, 1.0 - bar
    diag = transform_geo(box_geo(math.hypot(dx, dy) + 0.08, bar * 0.95, 0.06, color), rotate_z(math.atan2(dy, dx)))
    return concat_geo(top, bot, diag)


def zzz_matrix(par: dict, eye, origin=None) -> np.ndarray:
    """Y-axis billboard: the glyph's front (+z) turns toward the camera. origin: where the Zs start (follows the head)."""
    pos = np.array(ZZZ_ORIGIN if origin is None else origin, dtype=np.float64) + np.array(par["offset"])
    yaw = math.atan2(float(eye[0]) - pos[0], float(eye[2]) - pos[2])
    return translate(*pos) @ rotate_y(yaw) @ scale(par["size"])


class ZzzRenderer:
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.mesh = Mesh(ctx, program, z_glyph_geo())

    def draw(self, scene, t: float, vm=None, origin=None):
        eye = scene.camera.eye
        self.ctx.depth_mask = False                    # translucent: test depth, don't write it
        for par in zzz_params(getattr(vm, "zzz", 0.0) if vm is not None else 0.0):
            self.mesh.draw(scene, zzz_matrix(par, eye, origin), lit=False, emissive=1.0, alpha=par["alpha"], fog=0.0)
        self.ctx.depth_mask = True

    def release(self):
        self.mesh.release()


# --------------------------------------------------------------------------
# Part 12a: bloom. The 3D scene is drawn into an offscreen (4x MSAA) target, bright pixels are extracted, blurred at half
# resolution and added back, so the neon and the screens glow. HUD and log panel are drawn afterwards, crisp, on the window.
# If anything in the bloom setup fails the app silently falls back to the plain path. Key B toggles it, --no-bloom disables it.
# --------------------------------------------------------------------------
BLOOM_THRESHOLD = 0.50
BLOOM_KNEE = 0.20
BLOOM_STRENGTH = 1.10
BLOOM_SPREAD = 2.0                              # blur tap spacing in half-res texels: a wide halo for the same 13 taps
BLOOM_RADIUS = 6                                # blur taps each side, at half resolution
BLOOM_STRENGTH_STEP = 0.25
BLOOM_STRENGTH_MAX = 3.0
BLOOM_SAMPLES = 4


def gauss_weights(radius: int = BLOOM_RADIUS) -> np.ndarray:
    """Normalised 1-D gaussian, 2*radius+1 taps (index radius is the centre)."""
    r = max(int(radius), 1)
    sigma = r / 2.2
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


def half_size(size) -> tuple:
    return max(int(size[0]) // 2, 1), max(int(size[1]) // 2, 1)


def bloom_bright(img: np.ndarray) -> np.ndarray:
    """CPU reference of the bright-pass shader: soft-knee threshold on the brightest channel, colour preserved."""
    c = np.asarray(img, dtype=np.float64)
    b = c.max(axis=-1, keepdims=True)
    soft = np.clip(b - (BLOOM_THRESHOLD - BLOOM_KNEE), 0.0, 2.0 * BLOOM_KNEE)
    soft = soft * soft / (4.0 * BLOOM_KNEE + 1e-5)
    contrib = np.maximum(soft, b - BLOOM_THRESHOLD) / np.maximum(b, 1e-4)
    return c * contrib


def bloom_blur(img: np.ndarray, radius: int = BLOOM_RADIUS) -> np.ndarray:
    """CPU reference of the two separable blur passes (edge pixels are clamped like the GL texture)."""
    w = gauss_weights(radius)
    out = np.asarray(img, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for k, wk in enumerate(w):
            acc += wk * np.take(p, np.arange(out.shape[axis]) + k, axis=axis)
        out = acc
    return out


def bloom_composite(scene: np.ndarray, blurred: np.ndarray, strength: float = BLOOM_STRENGTH) -> np.ndarray:
    return np.clip(np.asarray(scene, dtype=np.float64) + float(strength) * np.asarray(blurred, dtype=np.float64), 0.0, 1.0)


BLOOM_VERT = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);     // one oversized triangle covers the screen
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
BLOOM_BRIGHT_FRAG = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb;
    float b = max(max(c.r, c.g), c.b);
    float soft = clamp(b - (%(T).6f - %(K).6f), 0.0, 2.0 * %(K).6f);
    soft = soft * soft / (4.0 * %(K).6f + 1e-5);
    float contrib = max(soft, b - %(T).6f) / max(b, 1e-4);
    f_color = vec4(c * contrib, 1.0);
}
""" % dict(T=BLOOM_THRESHOLD, K=BLOOM_KNEE)
BLOOM_BLUR_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_dir;
in vec2 v_uv;
out vec4 f_color;
const float W[%(N)d] = float[%(N)d](%(W)s);
void main() {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < %(N)d; i++) acc += W[i] * texture(u_tex, v_uv + u_dir * float(i - %(R)d)).rgb;
    f_color = vec4(acc, 1.0);
}
""" % dict(N=2 * BLOOM_RADIUS + 1, R=BLOOM_RADIUS, W=", ".join(f"{w:.8f}" for w in gauss_weights()))
BLOOM_COMPOSITE_FRAG = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_strength;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb + u_strength * texture(u_bloom, v_uv).rgb;
    f_color = vec4(min(c, vec3(1.0)), 1.0);
}
"""


class BloomPipeline:
    def __init__(self, ctx, size, samples: int = BLOOM_SAMPLES):
        import moderngl
        self.ctx, self.mgl, self.size = ctx, moderngl, (int(size[0]), int(size[1]))
        hw, hh = half_size(self.size)
        self.half = (hw, hh)
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer(self.size, 4, samples=samples)],
                                    depth_attachment=ctx.depth_renderbuffer(self.size, samples=samples))
        self.scene_tex = ctx.texture(self.size, 4)
        self.resolve = ctx.framebuffer(color_attachments=[self.scene_tex])
        self.tex_a, self.tex_b = ctx.texture(self.half, 4), ctx.texture(self.half, 4)
        for tx in (self.scene_tex, self.tex_a, self.tex_b):
            tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tx.repeat_x = tx.repeat_y = False
        self.fbo_a, self.fbo_b = ctx.framebuffer(color_attachments=[self.tex_a]), ctx.framebuffer(color_attachments=[self.tex_b])
        mk = lambda frag: ctx.program(vertex_shader=BLOOM_VERT, fragment_shader=frag)
        self.p_bright, self.p_blur, self.p_comp = mk(BLOOM_BRIGHT_FRAG), mk(BLOOM_BLUR_FRAG), mk(BLOOM_COMPOSITE_FRAG)
        self.vaos = [ctx.vertex_array(p, []) for p in (self.p_bright, self.p_blur, self.p_comp)]

    def _tri(self, i):
        self.vaos[i].render(self.mgl.TRIANGLES, vertices=3)

    def begin(self):
        self.msaa.use()
        self.ctx.viewport = (0, 0, *self.size)

    def present(self, strength: float = BLOOM_STRENGTH):
        ctx = self.ctx
        ctx.copy_framebuffer(self.resolve, self.msaa)                       # resolves the multisampled scene
        ctx.disable(self.mgl.DEPTH_TEST | self.mgl.BLEND)
        self.fbo_a.use(); ctx.viewport = (0, 0, *self.half)
        self.scene_tex.use(0); self.p_bright["u_tex"].value = 0; self._tri(0)
        self.fbo_b.use()
        self.tex_a.use(0); self.p_blur["u_tex"].value = 0; self.p_blur["u_dir"].value = (BLOOM_SPREAD / self.half[0], 0.0); self._tri(1)
        self.fbo_a.use()
        self.tex_b.use(0); self.p_blur["u_dir"].value = (0.0, BLOOM_SPREAD / self.half[1]); self._tri(1)
        ctx.screen.use(); ctx.viewport = (0, 0, *self.size)
        self.scene_tex.use(0); self.tex_a.use(1)
        self.p_comp["u_scene"].value = 0; self.p_comp["u_bloom"].value = 1; self.p_comp["u_strength"].value = float(strength)
        self._tri(2)
        ctx.enable(self.mgl.DEPTH_TEST | self.mgl.BLEND)

    def release(self):
        for o in (*self.vaos, self.p_bright, self.p_blur, self.p_comp, self.fbo_a, self.fbo_b, self.tex_a, self.tex_b, self.resolve, self.scene_tex, self.msaa):
            try:
                o.release()
            except Exception:
                pass


def bloom_step(strength: float, direction: int) -> float:
    """+/- keys: change the glow strength in fixed steps, clamped to 0..BLOOM_STRENGTH_MAX (0 = off, but B still toggles)."""
    v = float(strength) + BLOOM_STRENGTH_STEP * (1 if direction > 0 else -1)
    return round(min(max(v, 0.0), BLOOM_STRENGTH_MAX), 4)


def make_bloom(ctx, size):
    """BloomPipeline, or None (with a one-line note) if this GPU/driver cannot build it: the app then runs without bloom."""
    try:
        return BloomPipeline(ctx, size)
    except Exception as e:
        print(f"[visual_dashboard] bloom disabled ({type(e).__name__}: {e})", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Part 11b: wing flutter / droop, breathing bob and the coloured glow pool on the desk in front of the monitor.
# --------------------------------------------------------------------------
WING_DROOP_RAD = 0.5                            # silent: both wings fold down about their roots
BOB_MAX = 0.05
GLOW_POOL_ALPHA = 0.30
GLOW_POOL_RECT = (-2.15, -0.05, 3.0)           # x0, x1 (world, desk top), half-width in z: the desk in front of the main monitor


def body_bob(vm, t: float) -> float:
    """Vertical breathing offset: the feed's own `bob` if it supplies one (clamped), else a gentle sine."""
    b = getattr(vm, "bob", 0.0) if vm is not None else 0.0
    try:
        b = float(b)
    except (TypeError, ValueError):
        b = 0.0
    if not math.isfinite(b) or b == 0.0:
        return 0.012 * math.sin(t * 2.0)
    return max(-BOB_MAX, min(BOB_MAX, b))


def flap_angle(vm, t: float, i: int) -> float:
    """Twist of wing i about its span axis. Still when SILENT, busier while RESEARCHing, hard flicks on a flash; the two wings
    are mirrored (opposite sign). Uses the feed's `wing` value when present, otherwise its own 16 rad/s sine."""
    state = getattr(vm, "state", "IDLE") if vm is not None else "IDLE"
    if state == "SILENT":
        return 0.0
    amp = 0.05 + (0.05 if state == "RESEARCH" else 0.0) + 0.35 * _cl(getattr(vm, "flash", 0.0) if vm is not None else 0.0)
    w = getattr(vm, "wing", 0.0) if vm is not None else 0.0
    try:
        w = float(w)
    except (TypeError, ValueError):
        w = 0.0
    if not math.isfinite(w) or w == 0.0:
        w = math.sin(t * 16.0)
    return (1.0 if i % 2 == 0 else -1.0) * amp * max(-1.0, min(1.0, w))


def wing_droop_matrix(root, droop: float) -> np.ndarray:
    """Fly-local: swing the wing down about its root (rotation about the lateral z axis through the root)."""
    r = np.asarray(root, dtype=np.float64)
    return translate(*r) @ rotate_z(float(droop)) @ translate(*(-r))


def glow_pool(vm):
    """(rgb, alpha): the monitor's light spilling onto the desk. Green on BUY, red on SELL, scaled by the flash; none otherwise."""
    if vm is None:
        return COL_GREEN, 0.0
    col = {"BUY": COL_GREEN, "SELL": COL_RED}.get(getattr(vm, "state", ""))
    if col is None:
        return COL_GREEN, 0.0
    return col, GLOW_POOL_ALPHA * _cl(getattr(vm, "flash", 0.0))


def glow_pool_geo() -> np.ndarray:
    """Three nested flat rectangles lying just above the desk top; drawn with the same alpha they stack into a soft-edged pool."""
    x0, x1, hz = GLOW_POOL_RECT
    cx, cz, hx = (x0 + x1) / 2.0, 0.0, (x1 - x0) / 2.0
    parts = []
    for k, f in enumerate((1.0, 0.72, 0.44)):
        parts.append(transform_geo(box_geo(2 * hx * f, 0.002, 2 * hz * f, (1.0, 1.0, 1.0)), translate(cx, 0.003 + 0.0005 * k, cz)))
    return concat_geo(*parts)


class GlowPoolRenderer:
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.mesh = Mesh(ctx, program, glow_pool_geo())

    def draw(self, scene, t: float, vm=None):
        col, a = glow_pool(vm)
        if a <= 0.004:
            return
        self.ctx.depth_mask = False
        self.mesh.draw(scene, identity(), tint=col, lit=False, emissive=1.0, alpha=a * 0.6, fog=0.0)
        self.ctx.depth_mask = True

    def release(self):
        self.mesh.release()


# --------------------------------------------------------------------------
# Part 11a: body pose. HOLD leans in, SILENT slumps over the desk, BUY/SELL shift the shoulders, eyes close when silent or blinking.
# One eased pose per frame (BodyPose) is applied to body, legs, wings and the Zzz origin; the arm target is re-expressed in the
# leaned frame so the hand still lands on the lever.
# --------------------------------------------------------------------------
LEAN_PIVOT = (-0.10, 0.0, 0.0)                 # fly-local hip, on the seat
POSE_TARGETS = {                               # state -> (lean deg, forward = +x; drop in y)
    "IDLE": (0.0, 0.0), "RESEARCH": (5.0, 0.0), "BUY": (4.0, 0.0), "SELL": (-3.0, 0.0),
    "HOLD": (11.0, 0.0), "SILENT": (26.0, 0.12),
}
POSE_RATE = {"SILENT": 1.4, "HOLD": 3.0}       # 1/s easing speed per target state (slump slowly), default below
POSE_RATE_DEFAULT = 5.0
COL_EYE_CLOSED = (0.25, 0.10, 0.12)


def pose_target(state: str):
    return POSE_TARGETS.get(str(state), POSE_TARGETS["IDLE"])


def pose_matrix(lean_deg: float, drop: float = 0.0) -> np.ndarray:
    """Fly-local rigid transform: lean forward (+x) about the hip pivot, then sink by `drop`."""
    px, py, pz = LEAN_PIVOT
    return translate(0.0, -float(drop), 0.0) @ translate(px, py, pz) @ rotate_z(-math.radians(float(lean_deg))) @ translate(-px, -py, -pz)


class BodyPose:
    """Exponentially eased lean/drop toward the current state's target; frame-rate independent, never overshoots."""
    def __init__(self):
        self.lean = 0.0
        self.drop = 0.0
        self.droop = 0.0                 # wings fold down when silent
        self._t = None

    def update(self, vm, t: float):
        dt = 0.0 if self._t is None else min(max(float(t) - self._t, 0.0), 0.25)
        self._t = float(t)
        state = getattr(vm, "state", "IDLE")
        tl, td = pose_target(state)
        k = 1.0 - math.exp(-POSE_RATE.get(state, POSE_RATE_DEFAULT) * dt)
        self.lean += (tl - self.lean) * k
        self.drop += (td - self.drop) * k
        self.droop += ((WING_DROOP_RAD if state == "SILENT" else 0.0) - self.droop) * k
        return self.matrix()

    def matrix(self) -> np.ndarray:
        return pose_matrix(self.lean, self.drop)

    def head_world(self) -> np.ndarray:
        return np.array(FLY_POS) + (self.matrix().astype(np.float64) @ np.array([*HEAD_C, 1.0]))[:3]

    def zzz_origin(self) -> np.ndarray:
        return self.head_world() + np.array([0.0, ZZZ_ORIGIN[1] - (FLY_POS[1] + HEAD_C[1]), 0.0])


def arm_target_in_pose(vm, pose) -> np.ndarray:
    """arm_target is in the un-leaned fly frame; the arm mesh is drawn inside `pose`, so pre-apply its inverse."""
    tgt = arm_target(vm)
    if pose is None:
        return tgt
    return (np.linalg.inv(np.asarray(pose, dtype=np.float64)) @ np.append(tgt, 1.0))[:3]


def eye_style(vm):
    """(tint, emissive) for the eyes: state colour and flash glow, dark and dim while closed (SILENT or a blink)."""
    if vm is None:
        return (1.0, 1.0, 1.0), 0.30
    if bool(getattr(vm, "eyes_closed", False)) or getattr(vm, "state", "") == "SILENT":
        return COL_EYE_CLOSED, 0.02
    tint = {"BUY": COL_GREEN, "SELL": COL_RED}.get(getattr(vm, "state", ""), (1.0, 1.0, 1.0))
    return tint, 0.30 + 1.0 * _cl(getattr(vm, "flash", 0.0))


# --------------------------------------------------------------------------
# Part 2: async data bridge
#
# Two interchangeable feeds, both polled off the render thread:
#   LeagueBridge : wraps dashboard.app.DashboardApp (your existing read-only
#                  poller thread + FlyAnimator). Never writes to league folders.
#   SimFeed      : synthetic fallback with its own producer thread.
# The render loop only ever calls feed.update(dt) and feed.view(); neither
# touches the disk or blocks, so the viewport stays smooth.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))          # so `import dashboard` works from experiments/

UI_GREEN = (57, 255, 20)
UI_RED = (255, 40, 50)
UI_CYAN = (80, 220, 255)
UI_AMBER = (255, 190, 60)

DURATION = {"BUY": 1.5, "SELL": 1.5, "HOLD": 1.8, "SILENT": 3.0}
ACTIONS = ("BUY", "SELL", "HOLD", "SILENT")
RESEARCH_STATUS = ("SEARCHING SEC EDGAR...", "DOWNLOADING SEC XBRL...", "PARSING TICKER {tk}...")


def format_log(action: str, ticker: str, score: float, agent: str = ""):
    """(rgb 0-255, text) log line in the exact wording of the spec."""
    if action == "BUY":
        return UI_GREEN, f"[FLY BUY] Bought {ticker} - MBON Hz: {score:+.1f}"
    if action == "SELL":
        return UI_RED, f"[FLY SELL] Sold {ticker} from holdings - MBON Hz: {score:+.1f}"
    if action == "HOLD":
        return UI_CYAN, f"[FLY HOLD] Holding {ticker}"
    return UI_AMBER, f"[FLY SILENT] {agent}: constant score, circuit dead - flagged for culling"


_HOLD_RE = re.compile(r"^\[FLY HOLD\].*? (?:on|holding on) (\S+)\s*$")
_BUY_RE = re.compile(r"^\[FLY BUY\] Entered Long Position on (\S+)(.*)$")
_SELL_RE = re.compile(r"^\[FLY SELL\] Covered Short Position on (\S+)(.*)$")


def spec_text(text: str) -> str:
    """The accounts are long-only (no shorting, no calls), so the log says what really happens: a BUY is a purchase, a SELL sells
    shares the fly owns, a HOLD keeps a position. Older wordings ('Entered Long Position', 'Covered Short Position',
    'Defending Position ... Maintaining neutral holding on X') coming from your FlyAnimator are rewritten to the same text."""
    m = _BUY_RE.match(text)
    if m:
        return f"[FLY BUY] Bought {m.group(1)}{m.group(2)}"
    m = _SELL_RE.match(text)
    if m:
        return f"[FLY SELL] Sold {m.group(1)} from holdings{m.group(2)}"
    m = _HOLD_RE.match(text)
    return f"[FLY HOLD] Holding {m.group(1)}" if m else text


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def envelopes(state: str, t: float):
    """(reach, flash) for the current action. reach: arm out and back (0..1..0); flash: sharp attack then decay.
    Your Pose.flash / Pose.reach are never set by FlyAnimator, so they are derived here from state + elapsed time."""
    dur = DURATION.get(state, 0.0)
    if state in ("BUY", "SELL") and dur > 0:
        reach = _smooth(t / 0.5) * _smooth((dur - t) / 0.4)
        flash = min(1.0, t / 0.12) * max(0.0, 1.0 - t / dur) ** 0.7
        return reach, flash
    return 0.0, 0.0


@dataclasses.dataclass
class ViewModel:
    """Plain snapshot the renderer reads each frame (no pygame, no GL, no disk)."""
    state: str = "IDLE"          # IDLE | RESEARCH | BUY | SELL | HOLD | SILENT
    t: float = 0.0               # seconds into the current action
    ticker: str = ""
    flash: float = 0.0           # 0..1 screen/eye glow intensity
    reach: float = 0.0           # 0..1 arm extension toward the lever
    lever_buy: float = 0.0
    lever_sell: float = 0.0
    eyes_closed: bool = False
    wing: float = 0.0
    bob: float = 0.0
    zzz: float = 0.0
    clock: float = 0.0
    header: dict = dataclasses.field(default_factory=dict)       # round, universe, top_id, top_ic, ridge_ic, untrained_ic, mode
    log: list = dataclasses.field(default_factory=list)          # [(rgb255, text)]
    tape: list = dataclasses.field(default_factory=list)         # [(ticker, score)]
    ic_history: list = dataclasses.field(default_factory=list)   # [(round, top_ic, ridge_ic, untrained_ic)], 9f-b fills it
    research_lines: list = dataclasses.field(default_factory=list)
    research_status: str = ""
    replay_label: str = ""
    source: str = "sim"
    trade_date: str = ""         # dd/mm/yyyy, the day the fly is trading ('' = the feed does not say)


def adapt_view(v, source: str = "league", native: bool = False) -> ViewModel:
    """dashboard.scene.View -> ViewModel. native=True: the animator fills Pose.reach/flash/lever_* itself, use them as is;
    otherwise (older scene.py) derive reach/flash from state + elapsed time."""
    p = v.pose
    if native:
        reach, flash, lever_buy, lever_sell = p.reach, p.flash, p.lever_buy, p.lever_sell
    else:
        reach, flash = envelopes(p.state, p.t)
        lever_buy = reach if p.state == "BUY" else 0.0
        lever_sell = reach if p.state == "SELL" else 0.0
    return ViewModel(
        state=p.state, t=p.t, ticker=p.ticker, flash=flash, reach=reach,
        lever_buy=lever_buy, lever_sell=lever_sell,
        eyes_closed=bool(p.eyes_closed), wing=p.wing, bob=p.bob, zzz=p.zzz, clock=v.clock,
        header=dict(v.header), log=[(tuple(c), spec_text(s)) for c, s in v.log],
        tape=list(v.tape), research_lines=list(v.research_lines), research_status=v.research_status,
        replay_label=v.replay_label, source=source, trade_date=extract_trade_date(v.header, v))


def _animator_is_native(cls) -> bool:
    """True if this FlyAnimator class fills Pose.reach / Pose.flash itself (newer scene.py). Probed once, on a throwaway instance."""
    try:
        from types import SimpleNamespace
        a = cls()
        a.push(SimpleNamespace(action="BUY", ticker="TEST", score=1.0, agent="probe", when="", round=0))
        a.update(0.3)
        p = a.pose()
        return bool(p.reach > 0 or p.flash > 0)
    except Exception:
        return False


class IcAccumulator:
    """Builds the IC history from the header snapshots the feed reports: one row per round
    (round, top_ic, ridge_ic, untrained_ic). A repeated round updates its row in place; rows are never invented."""
    def __init__(self, maxlen: int = 200):
        self.maxlen = maxlen
        self.rows: list = []

    def add(self, header) -> list:
        hd = header or {}
        rnd = hd.get("round")
        vals = [_finite(hd.get(k)) for k in ("top_ic", "ridge_ic", "untrained_ic")]
        if rnd is not None and any(v is not None for v in vals):
            row = (rnd, *vals)
            if self.rows and self.rows[-1][0] == rnd:
                self.rows[-1] = row
            else:
                self.rows.append(row)
                del self.rows[:-self.maxlen]
        return list(self.rows)


class LeagueBridge:
    """Real data. DashboardApp starts its own daemon threads (panel index + 2 s poll of the league folders)."""
    source = "league"

    def __init__(self, league_root, panel=None, sec_root=None, universe_dir=None, agent=None, poll=2.0):
        from dashboard.app import DashboardApp          # ImportError -> caller falls back to SimFeed
        self.app = DashboardApp(league_root, panel, sec_root, universe_dir, agent, poll, start_threads=True)
        self.native = _animator_is_native(type(self.app.animator))
        self.ic = IcAccumulator()

    def update(self, dt: float):
        self.app.update(dt)

    def view(self) -> ViewModel:
        v = self.app.view()
        vm = adapt_view(v, "league", self.native)
        own = getattr(v, "ic_history", None)                      # use the app's own history if a newer scene.py provides one
        vm.ic_history = list(own) if own else self.ic.add(vm.header)
        return vm

    def next_agent(self):
        self.app.next_agent()

    def toggle_pause(self):
        self.app.toggle_pause()

    def close(self):
        self.app.quit = True


@dataclasses.dataclass
class SimDecision:
    action: str
    ticker: str
    score: float
    agent: str = "sim_fly"


SIM_START_DATE = (2020, 1, 6)                  # the simulated league "trades" one week per round from here (made up, like the rest of the sim)


def sim_trade_date(rnd) -> str:
    import datetime as _dt
    try:
        return format_trade_date(_dt.date(*SIM_START_DATE) + _dt.timedelta(days=7 * int(rnd)))
    except (TypeError, ValueError, OverflowError):
        return ""


class SimFeed:
    """Synthetic stand-in: research phase (~6 s) -> ten random decisions -> round += 1 -> repeat.
    A producer thread fills a queue; the render thread drains it in update()."""
    source = "sim"
    TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "V", "MA", "HD", "PG", "COST",
               "ZTS", "ZBRA", "LLY", "AVGO", "ORCL", "CRM", "ADBE", "KO", "PEP", "WMT", "BAC", "CVX", "NFLX", "AMD")
    TAGS = ("Revenues", "NetIncomeLoss", "Assets", "StockholdersEquity", "OperatingIncomeLoss", "GrossProfit")

    def __init__(self, seed: int = 7, start_thread: bool = True, poll: float = 0.5):
        self.rng = random.Random(seed)
        self.poll = poll
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending = collections.deque(maxlen=12)
        self._data = dict(round=0, mode="research", top_id="fly_017", top_ic=0.021, ridge_ic=0.004,
                          untrained_ic=-0.001, universe=499)
        self._ticks = 0
        self._made = 0
        self.ic = IcAccumulator()
        self._ic_rows = self.ic.add(self._data)                    # the simulated league starts with its round-0 reading
        self.rlines = [f"{self.rng.choice(self.TICKERS):<6} {self.rng.choice(self.TAGS):<34} "
                       f"{self.rng.randrange(10**7, 10**11)}  FY2025 filed 2025-11-0{self.rng.randint(1, 9)}"
                       for _ in range(40)]
        # foreground animation state (render thread only)
        self.cur = None
        self.state = "RESEARCH"
        self.t = 0.0
        self.clock = 0.0
        self.paused = False
        self.log = collections.deque(maxlen=10)
        self.tape = collections.deque(maxlen=12)
        self._logged = False
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="sim-feed", daemon=True)
            self._thread.start()

    # -- producer (background thread) ------------------------------------
    def _make_decision(self) -> SimDecision:
        a = self.rng.choices(ACTIONS, weights=(3, 3, 3, 1))[0]
        tk = self.rng.choice(self.TICKERS)
        if a == "BUY":
            return SimDecision(a, tk, self.rng.uniform(0.5, 3.0))
        if a == "SELL":
            return SimDecision(a, tk, self.rng.uniform(-3.0, -0.5))
        if a == "HOLD":
            return SimDecision(a, tk, self.rng.uniform(-0.3, 0.3))
        return SimDecision(a, "", 0.0)

    def produce_once(self):
        with self._lock:
            d = self._data
            if d["mode"] == "research":
                self._ticks += 1
                if self._ticks >= 12:                                   # ~6 s at poll=0.5
                    d["mode"], self._ticks, self._made = "league", 0, 0
            else:
                if len(self._pending) < 2 and self._made < 10:
                    self._pending.append(self._make_decision())
                    self._made += 1
                if self._made >= 10 and not self._pending:
                    d["mode"] = "research"
                    d["round"] += 1
                    d["top_ic"] = round(d["top_ic"] + self.rng.gauss(0, 0.004), 4)
                    d["ridge_ic"] = round(d["ridge_ic"] + self.rng.gauss(0, 0.002), 4)
                    d["untrained_ic"] = round(d["untrained_ic"] + self.rng.gauss(0, 0.002), 4)
                    self._ic_rows = self.ic.add(d)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.produce_once()
            except Exception:
                pass
            self._stop.wait(self.poll)

    # -- foreground (cheap) ----------------------------------------------
    def update(self, dt: float):
        if self.paused:
            return
        self.clock += dt
        if self.cur is None:
            with self._lock:
                d = self._pending.popleft() if self._pending else None
            if d is not None:
                self.cur, self.state, self.t, self._logged = d, d.action, 0.0, False
                if d.action != "SILENT":
                    self.tape.append((d.ticker, d.score))
        if self.cur is not None:
            self.t += dt
            if not self._logged and self.t >= 0.25:
                self.log.append(format_log(self.cur.action, self.cur.ticker, self.cur.score, self.cur.agent))
                self._logged = True
            if self.t >= DURATION[self.state]:
                self.cur, self.t = None, 0.0
        if self.cur is None:
            with self._lock:
                research = self._data["mode"] == "research"
            self.state = "RESEARCH" if research else "IDLE"

    def view(self) -> ViewModel:
        with self._lock:
            d = dict(self._data)
            ic_rows = list(self._ic_rows)
        c = self.clock
        reach, flash = envelopes(self.state, self.t)
        tk = self.rlines[int(c * 12) % len(self.rlines)][:6].strip()
        return ViewModel(
            state=self.state, t=self.t, ticker=self.cur.ticker if self.cur else "", flash=flash, reach=reach,
            lever_buy=reach if self.state == "BUY" else 0.0, lever_sell=reach if self.state == "SELL" else 0.0,
            eyes_closed=self.state == "SILENT" or (c % 4.0) < 0.12, wing=math.sin(c * 40),
            bob=math.sin(c * 2.5) * 0.015, zzz=self.t if self.state == "SILENT" else 0.0, clock=c,
            header=dict(round=d["round"], universe=d["universe"], top_id=d["top_id"], top_ic=d["top_ic"],
                        ridge_ic=d["ridge_ic"], untrained_ic=d["untrained_ic"], mode=d["mode"]),
            log=list(self.log), tape=list(self.tape), ic_history=ic_rows, research_lines=self.rlines,
            research_status=RESEARCH_STATUS[int(c * 1.2) % 3].format(tk=tk),
            replay_label="SIMULATED FEED (no league data attached)", source="sim",
            trade_date=sim_trade_date(d.get("round", 0)))

    def next_agent(self):
        pass

    def toggle_pause(self):
        self.paused = not self.paused

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def make_feed(args):
    """Real league if its state.json exists and the dashboard package imports; otherwise the simulator."""
    if not args.sim:
        root = Path(args.league_root)
        if (root / "state.json").exists():
            try:
                feed = LeagueBridge(root, args.panel, args.sec_root, args.universe_dir, args.agent, args.poll)
                print(f"[dashboard] live league feed: {root}")
                return feed
            except Exception as e:
                print(f"[dashboard] league bridge unavailable ({e}); using simulator", file=sys.stderr)
        else:
            print(f"[dashboard] no state.json under {root}; using simulator", file=sys.stderr)
    print("[dashboard] simulated feed")
    return SimFeed()


# --------------------------------------------------------------------------
# Dashboard app (window, GL context, loop)
# --------------------------------------------------------------------------
class VisualDashboard:
    def __init__(self, feed, fullscreen: bool = True, size=WINDOW_SIZE, bloom: bool = True):
        import moderngl
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("STONKFLY // Visual Dashboard")
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else 0)
        pygame.display.set_mode((0, 0) if fullscreen else tuple(size), flags)      # (0, 0) = the desktop resolution
        try:
            win = pygame.display.get_window_size()
        except Exception:
            win = tuple(size)
        self.win_size = (max(int(win[0]), 1), max(int(win[1]), 1))
        if fullscreen:
            pygame.mouse.set_visible(False)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.prog = self.ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        self.ctx.viewport = (0, 0, *self.win_size)
        self.camera = Camera(self.win_size[0] / self.win_size[1])
        self.scene = Scene(self.camera)
        self.clock = pygame.time.Clock()
        self.running = True
        self.t0 = time.perf_counter()
        self.feed = feed
        self.vm = feed.view()
        self._caption_t = 0.0

        # Scene objects, in the order the frame draws them (see render())
        self.desk = DeskRenderer(self.ctx, self.prog, build_desk())
        self.chair = ChairRenderer(self.ctx, self.prog, build_chair())
        self.fly = FlyRenderer(self.ctx, self.prog, build_fly_body())
        self.wings = WingsRenderer(self.ctx, self.prog, build_wings())
        self.legs = LegsRenderer(self.ctx, self.prog, build_legs())
        self.console = ConsoleRenderer(self.ctx, self.prog, build_console())
        self.skyline = SkylineRenderer(self.ctx, self.prog, build_skyline(x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG))
        mon = build_monitor()
        self.monitor = MonitorRenderer(self.ctx, self.prog, mon, flat_screen=False)
        self.screen = ScreenRenderer(self.ctx, mon.screen_quad, monitor_matrix())
        side_m = monitor_matrix(SIDE_MONITOR)
        self.side_monitor = MonitorRenderer(self.ctx, self.prog, mon, flat_screen=False, model=side_m, react=False)
        self.side_screen = ScreenRenderer(self.ctx, mon.screen_quad, side_m, content=TapeContent(), react=False)
        ic_m = monitor_matrix(IC_MONITOR)
        self.ic_monitor = MonitorRenderer(self.ctx, self.prog, mon, flat_screen=False, model=ic_m, react=False)
        self.ic_screen = ScreenRenderer(self.ctx, mon.screen_quad, ic_m, content=IcContent(), react=False)
        self.props = PropsRenderer(self.ctx, self.prog, build_tower(), build_mug())
        self.hud = HudRenderer(self.ctx)
        self.log_panel = HudRenderer(self.ctx, LogContent(), log_quad_ndc, (LOG_W, LOG_H), rgba=True)
        self.zzz = ZzzRenderer(self.ctx, self.prog)
        self.body_pose = BodyPose()
        self.glow_pool = GlowPoolRenderer(self.ctx, self.prog)
        self.bloom = make_bloom(self.ctx, self.win_size) if bloom else None
        self.bloom_on = self.bloom is not None
        self.bloom_strength = BLOOM_STRENGTH

    # -- lifecycle ---------------------------------------------------------
    def handle_events(self):
        for e in self.pygame.event.get():
            if e.type == self.pygame.QUIT:
                self.running = False
            elif e.type == self.pygame.KEYDOWN:
                if e.key == self.pygame.K_ESCAPE:
                    self.running = False
                elif e.key == self.pygame.K_TAB:
                    self.feed.next_agent()
                elif e.key == self.pygame.K_SPACE:
                    self.feed.toggle_pause()
                elif e.key == self.pygame.K_b and self.bloom is not None:
                    self.bloom_on = not self.bloom_on
                elif e.key in (self.pygame.K_EQUALS, self.pygame.K_PLUS, self.pygame.K_KP_PLUS):
                    self.bloom_strength = bloom_step(self.bloom_strength, +1)
                elif e.key in (self.pygame.K_MINUS, self.pygame.K_KP_MINUS):
                    self.bloom_strength = bloom_step(self.bloom_strength, -1)

    def update(self, t: float, dt: float):
        self.feed.update(dt)
        self.vm = self.feed.view()
        self._caption_t += dt
        if self._caption_t > 0.25:
            self._caption_t = 0.0
            h = self.vm.header
            self.pygame.display.set_caption(
                f"STONKFLY | {self.vm.source} | round {h.get('round', '-')} | {self.vm.state} {self.vm.ticker}")

    def _drop_bloom(self, e):
        print(f"[visual_dashboard] bloom failed mid-frame, switching it off ({type(e).__name__}: {e})", file=sys.stderr)
        self.bloom_on = False
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.win_size)

    def render(self, t: float):
        use_bloom = self.bloom is not None and self.bloom_on
        if use_bloom:
            try:
                self.bloom.begin()
            except Exception as e:
                self._drop_bloom(e)
                use_bloom = False
        self.ctx.clear(*SKY_TOP, 1.0)
        self.scene.begin_frame(self.prog, t)
        vm = self.vm
        self.skyline.draw(self.scene, t)
        self.desk.draw(self.scene, t)
        self.glow_pool.draw(self.scene, t, vm)      # monitor light on the desk, under the props
        self.monitor.draw(self.scene, t, vm)
        self.screen.draw(self.scene, t, vm)
        self.side_monitor.draw(self.scene, t, vm)
        self.side_screen.draw(self.scene, t, vm)
        self.ic_monitor.draw(self.scene, t, vm)
        self.ic_screen.draw(self.scene, t, vm)
        self.props.draw(self.scene, t, vm)
        self.chair.draw(self.scene, t, vm)
        pose = self.body_pose.update(vm, t)     # eased lean / slump for this frame
        self.fly.draw(self.scene, t, vm, pose)
        self.console.draw(self.scene, t, vm)
        self.legs.draw(self.scene, t, vm, pose)
        self.wings.draw(self.scene, t, vm, pose=pose, droop=self.body_pose.droop)      # translucent: last
        self.zzz.draw(self.scene, t, vm, self.body_pose.zzz_origin())        # translucent 3D, after the wings
        if use_bloom:
            try:
                self.bloom.present(self.bloom_strength)      # resolve + glow, onto the window
            except Exception as e:
                self._drop_bloom(e)
        self.hud.draw(self.win_size[0] / self.win_size[1], vm)      # screen-space overlays, over everything, never bloomed
        self.log_panel.draw(self.win_size[0] / self.win_size[1], vm)

    def run(self):
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt, last = now - last, now
            t = now - self.t0
            self.handle_events()
            self.update(t, dt)
            self.render(t)
            self.pygame.display.flip()
            self.clock.tick(TARGET_FPS)
        self.feed.close()
        self.pygame.quit()


# --------------------------------------------------------------------------
# Self-test (pure math; no window or GPU required)
# --------------------------------------------------------------------------
_FAILS: list = []


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            _FAILS.append(name)
        ok &= bool(cond)

    print("visual_dashboard selftest")
    p = translate(1, 2, 3) @ np.array([0, 0, 0, 1], dtype=np.float32)
    check("translate", np.allclose(p[:3], (1, 2, 3)))

    p = rotate_y(math.pi / 2) @ np.array([1, 0, 0, 1], dtype=np.float32)
    check("rotate_y +90deg maps +X -> -Z", np.allclose(p[:3], (0, 0, -1), atol=1e-6))

    cam = Camera(16 / 9)
    clip = cam.proj_view @ np.array([*cam.target, 1.0], dtype=np.float32)
    ndc = clip[:3] / clip[3]
    check("camera target projects to screen centre", abs(ndc[0]) < 1e-4 and abs(ndc[1]) < 1e-4)
    check("camera target inside depth range", -1.0 < ndc[2] < 1.0)

    cube = test_cube(1.0, COL_NEON_CYAN)
    check("cube has 36 verts x 9 floats", cube.shape == (36, 9))
    check("gl_bytes size", len(gl_bytes(identity())) == 64)
    ok &= _selftest_geometry(check)
    ok &= _selftest_wire(check)
    ok &= _selftest_skyline(check)
    ok &= _selftest_neon(check)
    ok &= _selftest_desk(check)
    ok &= _selftest_desk_neon(check)
    ok &= _selftest_monitor(check)
    ok &= _selftest_props(check)
    ok &= _selftest_fly_body(check)
    ok &= _selftest_wings_chair(check)
    ok &= _selftest_legs(check)
    ok &= _selftest_console_arm(check)
    ok &= _selftest_screen(check)
    ok &= _selftest_screen_panels(check)
    ok &= _selftest_aspect(check)
    ok &= _selftest_side_monitor(check)
    ok &= _selftest_ic_screen(check)
    ok &= _selftest_ic_feed(check)
    ok &= _selftest_hud(check)
    ok &= _selftest_log_zzz(check)
    ok &= _selftest_body_pose(check)
    ok &= _selftest_glow_wings(check)
    ok &= _selftest_trade_date(check)
    ok &= _selftest_bloom(check)
    ok &= _selftest_full_frame(check)
    ok &= _selftest_bridge(check)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _outward_ok(d: np.ndarray) -> bool:
    """For solids centred on the origin: every triangle's winding normal and stored normal point away from the centre."""
    t = np.asarray(d).reshape(-1, 3, 9)
    p = t[:, :, :3].astype(np.float64)
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return bool(((fn * p.mean(1)).sum(1) > 0).all() and ((fn * t[:, :, 3:6].mean(1)).sum(1) > 0).all())


def _selftest_geometry(check) -> bool:
    start = len(_FAILS)
    print(" geometry")
    bx = box_geo(2, 1, 4, (1, 0, 0))
    check("box: 36 verts, float32, (N,9)", bx.shape == (36, 9) and bx.dtype == np.float32)
    check("box: bounds", np.allclose(bx[:, :3].max(0), (1, 0.5, 2)) and np.allclose(bx[:, :3].min(0), (-1, -0.5, -2)))
    check("box: outward winding + unit normals", _outward_ok(bx) and np.allclose(np.linalg.norm(bx[:, 3:6], axis=1), 1, atol=1e-5))
    check("box: colour carried", np.allclose(bx[:, 6:9], (1, 0, 0)))

    sp = uv_sphere_geo(2.0, 16, 12)
    check("sphere: triangle count slices*(2*stacks-2)", len(sp) == 16 * 22 * 3)
    check("sphere: all verts on radius", np.allclose(np.linalg.norm(sp[:, :3], axis=1), 2.0, atol=1e-5))
    check("sphere: outward winding", _outward_ok(sp))
    check("sphere: smooth normals are radial", np.allclose(sp[:, 3:6], sp[:, :3] / 2.0, atol=1e-5))
    fl = uv_sphere_geo(1.0, 12, 8, flat=True)
    tri_n = fl[:, 3:6].reshape(-1, 3, 3)
    check("sphere flat: one normal per triangle", np.allclose(tri_n[:, 0], tri_n[:, 1]) and np.allclose(tri_n[:, 0], tri_n[:, 2]))
    check("sphere flat: faceted (not radial)", not np.allclose(fl[:, 3:6], fl[:, :3], atol=1e-2))
    check("sphere flat: outward winding", _outward_ok(fl))

    cy = cylinder_geo(0.5, None, 2.0, 12)
    check("cylinder: 4*segments triangles with caps", len(cy) == 4 * 12 * 3)
    check("cylinder: height bounds", np.allclose([cy[:, 1].min(), cy[:, 1].max()], (-1.0, 1.0)))
    check("cylinder: outward winding", _outward_ok(cy))
    check("cylinder: no caps -> 2*segments triangles", len(cylinder_geo(0.5, None, 1.0, 12, caps=False)) == 2 * 12 * 3)
    cone = cylinder_geo(0.5, 0.0, 1.0, 10)
    check("cone: degenerate apex triangles dropped", len(cone) == 2 * 10 * 3 and _outward_ok(cone))
    check("cone: slanted side normals tilt up", (cone[:, 4][np.abs(cone[:, 4]) < 0.999] > 0).all())

    m = translate(1, 2, 3) @ rotate_y(0.7) @ scale(2.0, 1.0, 0.5)
    ft = transform_geo(uv_sphere_geo(1.0, 12, 8, flat=True), m)
    p = ft[:, :3].reshape(-1, 3, 3).astype(np.float64)
    gn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    gn /= np.linalg.norm(gn, axis=1, keepdims=True)
    check("transform: normals stay perpendicular under non-uniform scale",
          np.allclose(ft[:, 3:6].reshape(-1, 3, 3)[:, 0], gn, atol=1e-4))
    check("transform: centre moved", np.allclose(ft[:, :3].mean(0), (1, 2, 3), atol=0.2))
    both = concat_geo(bx, sp)
    check("concat", both.shape == (len(bx) + len(sp), 9))
    check("recolor", np.allclose(recolor_geo(bx, (0, 1, 0))[:, 6:9], (0, 1, 0)) and np.allclose(bx[:, 6:9], (1, 0, 0)))
    return len(_FAILS) == start


def _selftest_wire(check) -> bool:
    start = len(_FAILS)
    print(" wireframes / limbs / IK")
    wb = wire_box_geo(2, 1, 4)
    check("wire box: 12 edges = 24 verts", wb.shape == (24, 9))
    check("wire box: 8 distinct corners", len({tuple(np.round(v, 5)) for v in wb[:, :3]}) == 8)
    e = wb[:, :3].reshape(-1, 2, 3)
    check("wire box: every edge is axis-aligned, lengths 2/1/4 x4 each",
          sorted(np.round(np.linalg.norm(e[:, 1] - e[:, 0], axis=1), 5).tolist()) == [1.0] * 4 + [2.0] * 4 + [4.0] * 4)

    ws = wire_sphere_geo(1.5, 12, 8)
    check("wire sphere: segment count", len(ws) == 2 * ((8 - 1) * 12 + 12 * 8))
    check("wire sphere: verts on radius", np.allclose(np.linalg.norm(ws[:, :3], axis=1), 1.5, atol=1e-5))

    cw = circle_wire_geo(2.0, 24)
    check("circle: 2*segments verts, on radius, in z=0",
          len(cw) == 48 and np.allclose(np.hypot(cw[:, 0], cw[:, 1]), 2.0, atol=1e-5) and np.allclose(cw[:, 2], 0))

    rg = ring_geo(1.0, 0.7, 32)
    r = np.hypot(rg[:, 0], rg[:, 1])
    check("ring: 2*segments triangles", len(rg) == 2 * 32 * 3)
    check("ring: radii within [inner, outer]", r.min() >= 0.7 - 1e-5 and r.max() <= 1.0 + 1e-5)
    check("ring: normals +Z and winding faces +Z", np.allclose(rg[:, 3:6], (0, 0, 1)) and
          (np.cross(rg[:, :3].reshape(-1, 3, 3)[:, 1] - rg[:, :3].reshape(-1, 3, 3)[:, 0],
                    rg[:, :3].reshape(-1, 3, 3)[:, 2] - rg[:, :3].reshape(-1, 3, 3)[:, 0])[:, 2] > 0).all())

    tb = wire_box_tubes_geo(2, 1, 4, 0.1)
    check("tube box: 12 boxes = 432 verts", tb.shape == (12 * 36, 9))
    check("tube box: bounds grow by thickness",
          np.allclose(tb[:, :3].max(0), (1.05, 0.55, 2.05), atol=1e-5) and np.allclose(tb[:, :3].min(0), (-1.05, -0.55, -2.05), atol=1e-5))

    rng = np.random.default_rng(1)
    good = True
    for d in list(rng.normal(size=(20, 3))) + [(0, 1, 0), (0, -1, 0), (1e-3, 1, 0)]:
        R = align_y(d)
        good &= np.allclose(R @ np.array([0, 1, 0, 1.0]), [*(np.asarray(d) / np.linalg.norm(d)), 1.0], atol=1e-6)
        good &= abs(np.linalg.det(R[:3, :3]) - 1.0) < 1e-6
    check("align_y: maps +Y to direction, proper rotation (incl. +/-Y)", good)

    p0, p1 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 1.0])
    L = np.linalg.norm(p1 - p0)
    lb = limb_geo(p0, p1, 0.2, segments=8)
    rel = lb[:, :3].astype(np.float64) - p0
    ax = (p1 - p0) / L
    along = rel @ ax
    radial = np.linalg.norm(rel - np.outer(along, ax), axis=1)
    check("limb: spans exactly p0..p1", abs(along.min()) < 1e-5 and abs(along.max() - L) < 1e-5)
    check("limb: radius 0.2", abs(radial.max() - 0.2) < 1e-5)
    t = lb.reshape(-1, 3, 9)
    pp = t[:, :, :3].astype(np.float64)
    check("limb: stored normals agree with winding",
          ((np.cross(pp[:, 1] - pp[:, 0], pp[:, 2] - pp[:, 0]) * t[:, :, 3:6].mean(1)).sum(1) > 0).all())
    tp = limb_geo(p0, p1, 0.3, segments=8, radius_end=0.1)
    rel = tp[:, :3].astype(np.float64) - p0
    along = rel @ ax
    radial = np.linalg.norm(rel - np.outer(along, ax), axis=1)
    check("limb taper: 0.3 at p0, 0.1 at p1",
          abs(radial[along < 1e-4].max() - 0.3) < 1e-5 and abs(radial[along > L - 1e-4].max() - 0.1) < 1e-5)
    check("limb: zero length -> empty", limb_geo(p0, p0, 0.1).shape == (0, 9))
    check("joint chain: 2 limbs + 3 joint spheres", joint_chain_geo([(0, 0, 0), (1, 1, 0), (2, 0, 0)], 0.1, segments=8).shape == (2 * 96 + 3 * 144, 9))

    s = np.array([0.0, 0.0, 0.0])
    elbow, hand = two_bone_ik(s, (1.2, 0.5, 0.0), 1.0, 1.0, bend=(0, 1, 0))
    check("IK: hand reaches reachable target", np.allclose(hand, (1.2, 0.5, 0.0), atol=1e-9))
    check("IK: bone lengths preserved", abs(np.linalg.norm(elbow - s) - 1) < 1e-9 and abs(np.linalg.norm(hand - elbow) - 1) < 1e-9)
    check("IK: elbow bends toward +Y", (elbow - (s + hand) / 2)[1] > 0.1)
    e2, h2 = two_bone_ik(s, (5.0, 0.0, 0.0), 1.0, 1.0)
    check("IK: unreachable target clamps to full reach", abs(np.linalg.norm(h2 - s) - 2.0) < 2e-3)
    e3, _ = two_bone_ik(s, (1.0, 0.0, 0.0), 1.0, 1.0, bend=(1, 0, 0))
    check("IK: bend parallel to reach does not crash", np.isfinite(e3).all())
    return len(_FAILS) == start


def _selftest_skyline(check) -> bool:
    import inspect
    start = len(_FAILS)
    print(" skyline (4a)")
    a, b, c = build_skyline(7), build_skyline(7), build_skyline(8)
    check("deterministic for a seed", np.array_equal(a.buildings, b.buildings) and np.array_equal(a.sky, b.sky))
    check("different seed -> different city", not np.array_equal(a.buildings, c.buildings))
    off = build_skyline(7, neon=False)
    check("neon=False: 4 empty window groups, no beacons",
          len(off.windows) == 4 and all(len(g) == 0 for g in off.windows) and len(off.beacons) == 0)
    check("neon flag does not move the towers", np.array_equal(a.buildings, off.buildings))

    sk = a.sky
    check("sky: clip-space quad at the far plane, 6 verts",
          sk.shape == (6, 9) and np.allclose(sk[:, 2], 0.99999) and set(np.round(sk[:, 1])) == {-1.0, 1.0})
    check("sky: purple glow at the bottom, navy at the top",
          np.allclose(sk[sk[:, 1] < 0][:, 6:9], SKY_BOTTOM, atol=1e-6) and np.allclose(sk[sk[:, 1] > 0][:, 6:9], SKY_TOP, atol=1e-6))

    bl = a.buildings
    check("towers: >50 main boxes, whole boxes only, finite", len(a.boxes) > 50 and len(bl) % 36 == 0 and np.isfinite(bl).all())
    check("towers: all behind the desk zone (z < -8)", (bl[:, 2] < -8.0).all())
    check("towers: reach down to the foot line", bl[:, 1].min() >= SKY_FOOT_Y - 1e-4)
    check("towers: dark navy/purple bodies", bl[:, 6:9].max() < 0.4)
    check("towers: one row per depth layer", len(np.unique(np.round(a.boxes[:, 2], 3))) == len(SKY_LAYERS))
    near = a.boxes[np.isclose(a.boxes[:, 2], SKY_LAYERS[0][0])]
    far = a.boxes[np.isclose(a.boxes[:, 2], SKY_LAYERS[-1][0])]
    check("towers: far row is taller than the near row", far[:, 4].mean() > near[:, 4].mean())

    pv = Camera(16 / 9).proj_view.astype(np.float64)
    cols = np.linspace(-1.0, 1.0, 200)
    covered = np.zeros(200, dtype=bool)
    n_vis = 0
    for cx, cy, cz, w, h, d in build_skyline(7, x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG).boxes:
        corners = np.array([[cx + sx * w / 2, cy + sy * h / 2, cz + sz * d / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        clip = np.concatenate([corners, np.ones((8, 1))], 1) @ pv.T
        if (clip[:, 3] <= 0.1).any():
            continue
        x = clip[:, 0] / clip[:, 3]
        if x.max() < -1 or x.min() > 1:
            continue
        n_vis += 1
        covered |= (cols >= x.min()) & (cols <= x.max())
    check(f"camera sees {n_vis} towers", n_vis > 30)
    check(f"towers span the screen width ({covered.mean():.0%} of columns)", covered.mean() > 0.85)

    names = ("u_eye", "u_fog_color", "u_fog_density", "u_fog_k")
    check("fog uniforms declared in the shader", all(n in FRAG_SRC for n in names))
    src = inspect.getsource(Scene.begin_frame) + inspect.getsource(Mesh.draw)
    check("fog uniforms written by Scene/Mesh (no inactive-uniform KeyError)", all(n in src for n in names))
    d = float(np.linalg.norm(Camera(16 / 9).eye))
    check("fog: light at the desk, heavy far away",
          1 - np.exp(-(d * FOG_DENSITY) ** 2) < 0.08 and 1 - np.exp(-(70 * FOG_DENSITY) ** 2) > 0.4)
    return len(_FAILS) == start


def _selftest_neon(check) -> bool:
    start = len(_FAILS)
    print(" skyline neon pixels (4b)")
    a = build_skyline(7)
    allw = np.concatenate(a.windows)
    n_w = len(allw) // 6
    check("windows: 4 non-empty twinkle groups", len(a.windows) == 4 and all(len(g) > 0 for g in a.windows))
    check("windows: whole quads, finite, colours in 0..1",
          len(allw) % 6 == 0 and np.isfinite(allw).all() and allw[:, 6:9].min() >= 0 and allw[:, 6:9].max() <= 1.0)
    q = allw[:, :3].reshape(-1, 6, 3)
    ext = q.max(1) - q.min(1)
    check("windows: tiny squares (0.11 units)", ext.max() <= 2 * WINDOW_HALF + 1e-3 and np.isclose(ext.max(1), 2 * WINDOW_HALF, atol=1e-4).all())
    nrm = allw[:, 3:6].reshape(-1, 6, 3)[:, 0]
    check("windows: unit axis-aligned normals only on +Z / -X faces",
          np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5) and all(tuple(n) in {(0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)} for n in np.unique(np.round(nrm, 5), axis=0)))
    t = q.reshape(-1, 3, 3)
    gn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    check("windows: winding agrees with the stored normal", (np.sum(gn * allw[:, 3:6].reshape(-1, 3, 3)[:, 0], axis=1) > 0).all())
    cen = q.mean(1)
    half = a.boxes[:, 3:6] / 2
    inside = (np.abs(cen[:, None, :] - a.boxes[None, :, :3]) < half[None] - 0.005).all(2).any(1)
    check("windows: none buried inside a tower", not inside.any())
    lit = n_w / a.window_slots
    check(f"windows: scattered, not a full grid (lit {lit:.0%} of slots)", 0.10 < lit < 0.50)
    share = np.array([len(g) for g in a.windows]) / len(allw)
    check("windows: groups roughly balanced (each > 15%)", (share > 0.15).all())
    mx = allw[:, 6:9].reshape(-1, 6, 3)[:, 0].max(1)
    check("windows: every pixel is neon (a channel >= 0.42 = far-row dimming 0.65 x jitter 0.65)", (mx >= 0.42).all())
    hues = np.unique(np.round(allw[:, 6:9].reshape(-1, 6, 3)[:, 0].argmax(1)))
    check("windows: more than one neon hue family", len(hues) >= 2)

    pv = Camera(16 / 9).proj_view.astype(np.float64)
    rs = build_skyline(7, x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG)              # what the app actually draws (9e: rotated)
    cen = np.concatenate(rs.windows)[:, :3].reshape(-1, 6, 3).mean(1)
    clip = np.concatenate([cen, np.ones((len(cen), 1))], 1) @ pv.T
    ndc = clip[:, :3] / clip[:, 3:4]
    seen = (clip[:, 3] > 0.1) & (np.abs(ndc[:, 0]) < 1) & (np.abs(ndc[:, 1]) < 1)
    check(f"camera sees {int(seen.sum())} window pixels", seen.sum() > 1000)
    px = (2 * WINDOW_HALF) / (2 * np.tan(np.radians(Camera(16 / 9).fov / 2)) * clip[seen, 3]) * WINDOW_PX_REF_H
    check(f"on-screen size {px.min():.1f}-{px.max():.1f}px at 720p (tiny, but not invisible)", px.min() > 0.8 and px.max() < 8)

    b = a.beacons
    check("beacons: whole boxes, 0.24 units, red/magenta", len(b) > 0 and len(b) % 36 == 0 and
          np.isclose((b[:, :3].reshape(-1, 36, 3).max(1) - b[:, :3].reshape(-1, 36, 3).min(1)).max(), 0.24, atol=1e-4) and
          (b[:, 6] >= 0.99).all())
    top = (a.boxes[:, 1] + a.boxes[:, 4] / 2).max()
    check("beacons sit above their towers", b[:, 1].min() > SKY_BASE_Y and b[:, 1].max() > top * 0.5)
    return len(_FAILS) == start


def _selftest_monitor(check) -> bool:
    start = len(_FAILS)
    print(" monitor (6a)")
    m = build_monitor()
    check("monitor: solid = body + base + neck (108 verts), LED = 1 box", m.solid.shape == (108, 9) and m.led.shape == (36, 9))
    for i, (c, size) in enumerate(m.boxes):
        check(f"monitor box {i}: outward winding", _outward_ok(m.solid[i * 36:(i + 1) * 36] - np.array([*c, 0, 0, 0, 0, 0, 0], dtype=np.float32)))
    M = monitor_matrix().astype(np.float64)
    w = lambda p: (M @ np.append(p, 1.0))[:3]
    base_c, base_s = m.boxes[1]
    check("monitor: base bottom rests on the desk top (y = 0)", abs(w(np.array(base_c) - [0, base_s[1] / 2, 0])[1]) < 1e-6)
    pts = np.array([w(np.array(c) + np.array(s) * sg / 2) for c, s in m.boxes for sg in
                    [(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]])
    dc, ds = np.array(DESK_CENTER), np.array(DESK_SIZE)
    check("monitor: inside the desk footprint",
          (pts[:, 0] > dc[0] - ds[0] / 2).all() and (pts[:, 0] < dc[0] + ds[0] / 2).all() and
          (pts[:, 2] > -ds[2] / 2).all() and (pts[:, 2] < ds[2] / 2).all())
    q = m.screen_quad
    check("monitor: screen planar, 2.4 x 1.48 (16:10)",
          np.isclose(np.linalg.norm(q[1] - q[0]), 2.4) and np.isclose(np.linalg.norm(q[0] - q[3]), 1.48) and np.allclose(q[:, 2], SCR_Z))
    sp = m.screen[:, :3].astype(np.float64).reshape(-1, 3, 3)
    fn = np.cross(sp[:, 1] - sp[:, 0], sp[:, 2] - sp[:, 0])
    check("screen: faces +z (local)", (fn[:, 2] > 0).all() and (m.screen[:, 5] > 0).all())
    n = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    check(f"screen normal {np.round(n, 2)} points slightly left (toward the fly side) and stays level", n[0] < -0.15 and abs(n[1]) < 1e-9)
    to_fly = (np.array(FLY_POS) - w(np.zeros(3))) * [1.0, 0.0, 1.0]
    check("screen angled inward toward the fly (horizontal normal . direction-to-fly > 0.25)", float(n @ (to_fly / np.linalg.norm(to_fly))) > 0.25)
    check("monitor uniform scale applied to the stand too", np.isclose(np.linalg.norm(M[:3, 0]), MON_SCALE) and np.isclose(np.linalg.norm(M[:3, 1]), MON_SCALE))
    cam = Camera(16 / 9)
    check("screen normal faces the camera", float(n @ (cam.eye - w(np.array([0, 0, SCR_Z])))) > 0)
    check("main monitor is right of the macro-pad console", w(np.zeros(3))[0] > CONSOLE_CENTER[0] + CONSOLE_SIZE[0] / 2)
    half_w = MON_BODY[0] / 2
    seg0, seg1 = w(np.array([-half_w, 0.0, 0.0]))[[0, 2]], w(np.array([half_w, 0.0, 0.0]))[[0, 2]]
    mgp = np.array([MUG_POS[0], MUG_POS[2]])
    dd = seg1 - seg0
    tt = float(np.clip((mgp - seg0) @ dd / (dd @ dd), 0.0, 1.0))
    gap = float(np.linalg.norm(seg0 + tt * dd - mgp))
    check(f"monitor body clear of the mug (gap {gap:.2f} >= 0.4 incl. mug radius and half body depth)", gap >= 0.4)
    pv = cam.proj_view.astype(np.float64)
    sc = np.array([w(c) for c in q])
    clip = np.concatenate([sc, np.ones((4, 1))], 1) @ pv.T
    ndc = clip[:, :2] / clip[:, 3:4]
    check("screen corners on screen, leaving the top 10% for the HUD", (np.abs(ndc[:, 0]) < 0.95).all() and (ndc[:, 1] < 0.8).all() and (ndc[:, 1] > -0.95).all())
    wd, ht = np.ptp(ndc[:, 0]) * 640, np.ptp(ndc[:, 1]) * 360
    print(f"     (screen spans ~{wd:.0f} x {ht:.0f} px at 1280x720)")
    check("screen is big enough to read (> 170 px wide at 720p, vs the old 139)", wd > 170)
    return len(_FAILS) == start


def _selftest_props(check) -> bool:
    start = len(_FAILS)
    print(" tower + mug (6b)")
    tw = build_tower()
    check("tower: body = chassis + face plate, outward winding", tw.body.shape == (72, 9) and _outward_ok(tw.body[:36]))
    check("tower: 4 neon rings (2 fans x 2 concentric), cyan", len(tw.rings) == 4 * 80 * 3 and np.allclose(tw.rings[:, 6:9], COL_NEON_CYAN))
    r = tw.rings[:, :3].reshape(4, -1, 3)
    cen = (r.max(1) + r.min(1)) / 2
    check("rings: concentric in pairs, 2 distinct fan centres", np.allclose(cen[0], cen[1], atol=1e-5) and np.allclose(cen[2], cen[3], atol=1e-5)
          and abs(cen[0][1] - cen[2][1]) > 0.8)
    rad = (r.max(1) - r.min(1))[:, 0] / 2
    check("rings: inner ring smaller than outer", rad[1] < rad[0] and rad[3] < rad[2])
    check("rings: sit just proud of the face, normals +z", (tw.rings[:, 2] > FACE_Z).all() and (tw.rings[:, 5] > 0.99).all())
    check("rings: wound CCW toward +z", bool((np.cross(*(np.diff(tw.rings[:, :3].reshape(-1, 3, 3), axis=1)[:, 0], np.diff(tw.rings[:, :3].reshape(-1, 3, 3), axis=1)[:, 1]))[:, 2] > 0).all()))
    check("fan spokes: 4 blades, fit inside the inner ring", tw.spokes.shape == (4 * 36, 9) and np.abs(tw.spokes[:, :2]).max() <= FAN_RINGS[1][1])
    off = build_tower(neon=False)
    check("neon=False: no rings/hubs/spokes, body unchanged", len(off.rings) == 0 and len(off.hubs) == 0 and np.array_equal(off.body, tw.body))
    M = tower_matrix().astype(np.float64)
    w = lambda p: (M @ np.append(p, 1.0))[:3]
    W, H, D = TOWER_SIZE
    corners = np.array([w((a * W / 2, b * H / 2, c * D / 2)) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    check("tower: stands on the floor (bottom y = floor top)", abs(corners[:, 1].min() - FLOOR_TOP_Y) < 1e-6)
    legs = np.array(LEG_XZ)
    dmin = min(np.hypot(corners[:, 0] - lx, corners[:, 2] - lz).min() for lx, lz in legs)
    check(f"tower clear of the desk legs (gap {dmin:.2f})", dmin > 0.5)
    check("tower: outside the desk footprint in front/right", (corners[:, 0] > DESK_CENTER[0] + DESK_SIZE[0] / 2).all() or (corners[:, 2] > DESK_SIZE[2] / 2).all() or (corners[:, 0] < DESK_CENTER[0] - DESK_SIZE[0] / 2).all())
    cam = Camera(16 / 9)
    n = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
    check("tower face points toward the camera", float(n @ (cam.eye - w(np.array([0, 0, FACE_Z]))) / np.linalg.norm(cam.eye - w(np.array([0, 0, FACE_Z])))) > 0.8)
    pv = cam.proj_view.astype(np.float64)
    allp = np.vstack([corners, [w((0, FAN_Y[0] + 0.43, RING_Z)), w((0, FAN_Y[1] - 0.43, RING_Z))]])
    clip = np.concatenate([allp, np.ones((len(allp), 1))], 1) @ pv.T
    ndc = clip[:, :2] / clip[:, 3:4]
    check("tower + both fans fully on screen (|ndc| < 0.98)", (clip[:, 3] > 0).all() and (np.abs(ndc) < 0.98).all())
    check("tower is in the bottom right (centre ndc x > 0.3, y < -0.2)", ndc[:8, 0].mean() > 0.3 and ndc[:8, 1].mean() < -0.2)
    fan_px = (np.ptp(ndc[:, 1]) * 360)
    print(f"     (tower centre ndc ~ {np.round(ndc[:8].mean(0), 2)}, height ~ {fan_px:.0f} px at 720p)")

    mg = build_mug()
    check("mug: body + handle, finite, faceted", len(mg.body) % 3 == 0 and np.isfinite(mg.body).all() and len(mg.body) > 100)
    check("mug: violet body, pink rim", np.allclose(mg.body[:, 6:9][:3], COL_MUG) and (mg.rim[:, 6] > 0.85).all() and (mg.rim[:, 8] > 0.5).all())
    mp = mg.body[:, :3] + np.array(MUG_POS)
    check("mug: sits on the desk top (min y = 0, height = MUG_H)", abs(mp[:, 1].min()) < 1e-6 and abs(mg.body[:, 1].max() - MUG_H) < 1e-6)
    pad_right = CONSOLE_CENTER[0] + CONSOLE_SIZE[0] / 2
    check("mug: just right of the keyboard (fly's right, clear of the console, left of the monitors)",
          mp[:, 2].min() > CONSOLE_CENTER[2] + CONSOLE_SIZE[2] / 2 + 0.05 and mp[:, 0].max() < 1.2)
    check("mug: well inside the desk top", abs(mp[:, 2]).max() < DESK_SIZE[2] / 2 - 0.1)
    mc = np.concatenate([mp[:: 7], np.ones((len(mp[::7]), 1))], 1) @ pv.T
    mn = mc[:, :2] / mc[:, 3:4]
    check("mug on screen", (np.abs(mn) < 0.95).all())
    check("mug rim: ring faces up (+y)", (mg.rim[:, 4] > 0.99).all())
    return len(_FAILS) == start


def _selftest_fly_body(check) -> bool:
    start = len(_FAILS)
    print(" fly body + eyes (7a)")
    f = build_fly_body()
    for name, g in (("head", f.head), ("thorax", f.thorax), ("abdomen", f.abdomen), ("eyes", f.eyes), ("extras", f.extras)):
        check(f"{name}: whole triangles, finite, float32", len(g) % 3 == 0 and np.isfinite(g).all() and g.dtype == np.float32 and len(g) > 0)
    tri = lambda g: g.reshape(-1, 3, 9)
    smooth = lambda g: (np.abs(tri(g)[:, 0, 3:6] - tri(g)[:, 1, 3:6]).max(1) > 1e-4).mean()
    check("head/thorax/abdomen are smooth shaded (vertex normals differ)", min(smooth(f.head), smooth(f.thorax), smooth(f.abdomen)) > 0.5)
    check("eyes are faceted (one normal per triangle)", smooth(f.eyes) == 0.0)
    check("head is higher-facet than an eye", len(f.head) // 3 > 3 * (len(f.eyes) // 6) and len(f.head) // 3 > 600)
    for name, g, c in (("head", f.head, HEAD_C), ("thorax", f.thorax, THORAX_C), ("abdomen", f.abdomen, ABD_C)):
        gg = g.copy()
        gg[:, :3] -= np.array(c, dtype=np.float32)
        check(f"{name}: outward winding", _outward_ok(gg))
    for i, c in enumerate(f.eye_centers):
        gg = tri(f.eyes)[i * (len(f.eyes) // 6):(i + 1) * (len(f.eyes) // 6)].reshape(-1, 9).copy()
        gg[:, :3] -= c.astype(np.float32)
        check(f"eye {i}: outward winding", _outward_ok(gg))
    check("eyes: crimson", (f.eyes[:, 6] > 0.6).all() and (f.eyes[:, 7] < 0.2).all() and (f.eyes[:, 8] < 0.2).all())
    ec = f.eye_centers
    check("eyes: mirrored on either side of the head, in front of its centre", np.isclose(ec[0, 2], -ec[1, 2]) and ec[0, 2] > 0.2 and (ec[:, 0] > HEAD_C[0]).all())
    hp = f.eye_hatch[:, :3].reshape(2, -1, 3)
    check("cross-hatching: line pairs on both eyes", len(f.eye_hatch) % 2 == 0 and len(f.eye_hatch) > 100)
    outside = [(((hp[i] - ec[i]) / np.array(EYE_R)) ** 2).sum(1).min() for i in range(2)]
    check("cross-hatching sits just outside the eye surface (no z-fighting)", min(outside) >= 1.0 and max(outside) < 1.2)
    check("abdomen: two stripe colours", len(np.unique(f.abdomen[:, 6:9].round(3), axis=0)) == 2)
    check("thorax charcoal, head gray", np.allclose(f.thorax[:, 6:9], COL_FLY_THORAX) and np.allclose(f.head[:, 6:9], COL_FLY_HEAD))
    allb = concat_geo(f.head, f.thorax, f.abdomen, f.extras, f.eyes)
    check("body sits on/above the seat (min y >= 0)", allb[:, 1].min() >= -0.01)
    check("abdomen trails behind the thorax, head leads", f.abdomen[:, 0].mean() < f.thorax[:, 0].mean() < f.head[:, 0].mean())
    wp = allb[:, :3] + np.array(FLY_POS, dtype=np.float32)
    check("fly is left of the monitor and not inside the console deck", wp[:, 0].max() < 0.0 and not
          ((wp[:, 0] > CONSOLE_CENTER[0] - CONSOLE_SIZE[0] / 2) & (wp[:, 1] < 0.12)).any())
    cam = Camera(16 / 9)
    pv = cam.proj_view.astype(np.float64)
    allw = np.concatenate([wp.astype(np.float64), np.ones((len(wp), 1))], 1) @ pv.T
    nd = allw[:, :2] / allw[:, 3:4]
    check("fly fully on screen (|ndc| < 0.97)", (allw[:, 3] > 0).all() and (np.abs(nd) < 0.97).all())
    print(f"     (fly spans ~{np.ptp(nd[:, 0]) * 640:.0f} x {np.ptp(nd[:, 1]) * 360:.0f} px at 1280x720; left edge ndc {nd[:, 0].min():.2f})")
    ew = ec + np.array(FLY_POS)
    outn = np.array([0.0, 0.0, 1.0])
    check("+z eye is visible from the camera (faces it)", float(outn @ (cam.eye - ew[0])) > 0)
    return len(_FAILS) == start


def _selftest_wings_chair(check) -> bool:
    start = len(_FAILS)
    print(" wings + chair (7b)")
    wg = build_wings()
    pl = wg.plate.reshape(-1, 3, 9)
    check("wing plate: ~2 triangles per step, finite", len(pl) >= 2 * WING_STEPS - 2 and np.isfinite(wg.plate).all())
    check("wing plate: normals +y (local), flat", (pl[:, :, 4] > 0.999).all())
    fn = np.cross(pl[:, 1, :3] - pl[:, 0, :3], pl[:, 2, :3] - pl[:, 0, :3])
    check("wing plate: winding agrees with the normal", (fn[:, 1] > 0).all())
    ext = np.ptp(wg.plate[:, :3], axis=0)
    check(f"wing plate: {ext[0]:.2f} long, {ext[2]:.2f} wide, flat", np.isclose(ext[0], WING_LEN) and 0.3 < ext[2] <= 2 * WING_HALF_W + 1e-6 and ext[1] < 1e-6)
    check("wing plate: symmetric about the span axis", np.isclose(wg.plate[:, 2].max(), -wg.plate[:, 2].min(), atol=1e-6))
    check("wing plate: light gray", (wg.plate[:, 6:9] > 0.7).all() and np.ptp(wg.plate[:, 6:9], axis=1).max() < 0.15)
    check("wing wire: even line vertex count, veins + outline", len(wg.wire) % 2 == 0 and len(wg.wire) // 2 > 4 * WING_STEPS)
    check("wing wire: stays within the plate outline",
          (np.abs(wg.wire[:, 2]) <= WING_HALF_W + 1e-6).all() and wg.wire[:, 0].min() >= -1e-6 and wg.wire[:, 0].max() <= WING_LEN + 1e-6)
    check("wing alpha is translucent", 0.1 <= WING_ALPHA <= 0.5)
    b0, b1 = wg.bases
    check("wings: proper rotations (det +1, orthonormal)", all(abs(np.linalg.det(B[:3, :3]) - 1) < 1e-9 and np.allclose(B[:3, :3].T @ B[:3, :3], np.eye(3), atol=1e-9) for B in wg.bases))
    check("wings: mirrored about z = 0", np.isclose(wg.roots[0][2], -wg.roots[1][2]) and np.allclose(b0[:3, 0] * [1, 1, -1], b1[:3, 0]))
    check("wing roots on the upper rear of the thorax", (wg.roots[:, 1] > THORAX_C[1] + 0.3).all() and (wg.roots[:, 0] < THORAX_C[0]).all())
    check("wings sweep rearward and up over the abdomen", (b0[:3, 0][0] < -0.8) and b0[:3, 0][1] > 0)
    fly = build_fly_body()
    am = (translate(*ABD_C) @ rotate_z(math.radians(90.0 + ABD_TILT_DEG)) @ scale(*ABD_R)).astype(np.float64)
    inv = np.linalg.inv(am)
    allp = []
    for B in wg.bases:
        pts = np.concatenate([wg.plate[:, :3].astype(np.float64), np.ones((len(wg.plate), 1))], 1) @ B.T
        allp.append(pts[:, :3])
        q = np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1) @ inv.T
        check("wing is clear of the abdomen (no clipping)", ((q[:, :3] ** 2).sum(1) > 1.0).all())
        far = np.linalg.norm(pts[:, :3] - np.array(THORAX_C), axis=1) > 0.75
        tq = (((pts[far][:, :3] - np.array(THORAX_C)) / np.array(THORAX_R)) ** 2).sum(1)
        check("wing (beyond the root) is clear of the thorax", (tq > 1.0).all())
    allp = np.concatenate(allp)
    check("wings reach back past the abdomen mid-point", allp[:, 0].min() < ABD_C[0])
    ChairT = build_chair()
    seat = ChairT.seat
    check("chair: seat top is at the fly's seat height", np.isclose(seat[:, 1].max(), SEAT_TOP_Y) and np.isclose(seat[:, 1].min(), SEAT_TOP_Y - SEAT_T))
    gs = seat.copy()
    gs[:, :3] -= np.array([CHAIR_XZ[0], SEAT_TOP_Y - SEAT_T / 2, CHAIR_XZ[1]], dtype=np.float32)
    check("chair seat: outward winding", _outward_ok(gs))
    check("chair: casters rest on the floor", np.isclose(ChairT.solid[:, 1].min(), FLOOR_TOP_Y, atol=1e-5))
    check("chair: finite whole triangles", len(ChairT.solid) % 3 == 0 and np.isfinite(ChairT.solid).all() and np.isfinite(ChairT.trim).all())
    check("chair: neon trim is whole boxes, magenta", len(ChairT.trim) == 12 * 36 and (ChairT.trim[:, 6] > 0.9).all())
    wb = fly.abdomen[:, :3] + np.array(FLY_POS, dtype=np.float32)
    bk = ChairT.back[:, 0].max()
    check(f"backrest is clear of the abdomen tip (gap {wb[:, 0].min() - bk:.2f})", wb[:, 0].min() - bk > 0.08)
    check("backrest is behind the fly and above the seat", ChairT.back[:, 0].max() < FLY_POS[0] - 1.0 and ChairT.back[:, 1].min() > SEAT_TOP_Y - 0.05)
    check("seat sits below the desk slab (no clipping)", seat[:, 1].max() < DESK_CENTER[1] - DESK_SIZE[1] / 2)
    lg = np.array(LEG_XZ)
    ds = min(np.hypot(ChairT.solid[:, 0] - lx, ChairT.solid[:, 2] - lz).min() for lx, lz in lg)
    check(f"chair clear of the desk legs (gap {ds:.2f})", ds > 0.3)
    cam = Camera(16 / 9)
    pv = cam.proj_view.astype(np.float64)
    pts = np.concatenate([ChairT.solid[:, :3].astype(np.float64), allp + np.array(FLY_POS)])
    c = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ pv.T
    nd = c[:, :2] / c[:, 3:4]
    check("chair and wings fully on screen (|ndc| < 0.98)", (c[:, 3] > 0).all() and (np.abs(nd) < 0.98).all())
    print(f"     (chair+wings ndc x {nd[:, 0].min():.2f}..{nd[:, 0].max():.2f}, y {nd[:, 1].min():.2f}..{nd[:, 1].max():.2f})")
    return len(_FAILS) == start


def _selftest_legs(check) -> bool:
    start = len(_FAILS)
    print(" legs + arm (8a)")
    lg = build_legs()
    L = lg.chains
    check("6 legs: 5 resting + arm", len(L) == 6 and "arm" in L)
    one = len(joint_chain_geo(L["hind+"], LEG_RADII, COL_FLY_LEG, 1.25, LEG_SEGMENTS))
    check("legs: static mesh = 5 identical-size legs, finite", lg.static.shape == (5 * one, 9) and np.isfinite(lg.static).all() and len(lg.arm) == one)
    check("legs: smooth gray", np.allclose(lg.static[:, 6:9], COL_FLY_LEG) and np.allclose(lg.arm[:, 6:9], COL_FLY_LEG))
    for n, (h, f, b, t) in LEG_SPECS.items():
        c = L[n]
        check(f"{n}: bone lengths exact ({LEG_L[0]}/{LEG_L[1]})", np.isclose(np.linalg.norm(c[1] - c[0]), LEG_L[0], atol=1e-6) and np.isclose(np.linalg.norm(c[2] - c[1]), LEG_L[1], atol=1e-6))
        check(f"{n}: foot reaches its target", np.allclose(c[2], f, atol=1e-6))
    a = L["arm"]
    check("arm: bone lengths exact", np.isclose(np.linalg.norm(a[1] - a[0]), ARM_L[0], atol=1e-6) and np.isclose(np.linalg.norm(a[2] - a[1]), ARM_L[1], atol=1e-6))
    check("arm: hand reaches the rest target", np.allclose(a[2], ARM_REST_TARGET, atol=1e-6))
    check("arm: target within reach", np.linalg.norm(np.array(ARM_REST_TARGET) - np.array(ARM_HIP)) < sum(ARM_L))
    far = leg_points(ARM_HIP, (9.0, 0.5, 0.5), *ARM_L, ARM_BEND)
    check("arm: unreachable target is clamped to full reach, not NaN", np.isfinite(far).all() and np.linalg.norm(far[2] - far[0]) <= sum(ARM_L) + 1e-6)
    t1, t2 = build_arm((1.2, 0.6, 0.3)), build_arm((2.0, 0.9, 0.6))
    check("arm: constant vertex count for any target (per-frame upload)", t1.shape == t2.shape == lg.arm.shape and not np.allclose(t1, t2))
    tm = lambda c: (THORAX_C, THORAX_R)
    th = lambda p: ((((np.asarray(p) - THORAX_C) / np.array(THORAX_R)) ** 2).sum(-1))
    ab_inv = np.linalg.inv((translate(*ABD_C) @ rotate_z(math.radians(90.0 + ABD_TILT_DEG)) @ scale(*ABD_R)).astype(np.float64))
    ab = lambda p: (((np.append(p, 1.0) @ ab_inv.T)[:3]) ** 2).sum()
    for n, c in L.items():
        check(f"{n}: hip is on the thorax underside", th(c[0]) < 1.25 and c[0][1] < THORAX_C[1] + 0.1)
        check(f"{n}: knee clear of thorax and abdomen", th(c[1]) > 1.0 and ab(c[1]) > 1.0)
    for n in ("hind+", "hind-", "mid+", "mid-"):
        c = L[n]
        check(f"{n}: foot on the seat", abs(c[2][1] - FOOT_Y) < 1e-9 and np.hypot(c[2][0], c[2][2]) < SEAT_R - 0.05)
        check(f"{n}: knee above the foot, bent outward", c[1][1] > c[2][1] + 0.2 and abs(c[1][2]) > abs(c[0][2]))
    for a_, b_ in (("hind+", "hind-"), ("mid+", "mid-")):
        check(f"{a_}/{b_}: mirrored about z = 0", np.allclose(L[a_] * [1, 1, -1], L[b_], atol=1e-6))
    fw = L["front-"][2] + np.array(FLY_POS)
    check("front- foot rests on the desk top edge", fw[0] >= DESK_CENTER[0] - DESK_SIZE[0] / 2 and fw[1] >= 0.0 and fw[1] < 0.1)
    hw = a[2] + np.array(FLY_POS)
    cl, cs = np.array(CONSOLE_CENTER), np.array(CONSOLE_SIZE)
    check("arm hand hovers over the console deck",
          abs(hw[0] - cl[0]) < cs[0] / 2 and abs(hw[2] - cl[2]) < cs[2] / 2 and hw[1] > cl[1] + cs[1] / 2 + 0.05)
    allv = concat_geo(lg.static, lg.arm)
    wp = allv[:, :3] + np.array(FLY_POS, dtype=np.float32)
    check("no leg pokes through the seat or the desk top", not ((wp[:, 1] < SEAT_TOP_Y - 0.01) & (np.hypot(wp[:, 0] - CHAIR_XZ[0], wp[:, 2]) < SEAT_R)).any())
    cam = Camera(16 / 9)
    c = np.concatenate([wp.astype(np.float64), np.ones((len(wp), 1))], 1) @ cam.proj_view.astype(np.float64).T
    nd = c[:, :2] / c[:, 3:4]
    check("legs and arm fully on screen (|ndc| < 0.97)", (c[:, 3] > 0).all() and (np.abs(nd) < 0.97).all())
    return len(_FAILS) == start


def _selftest_console_arm(check) -> bool:
    start = len(_FAILS)
    print(" console + arm drive (8b)")
    V = lambda **k: ViewModel(**k)
    n, b, sl = V(), V(lever_buy=1.0), V(lever_sell=1.0)
    check("lever: neutral is vertical", lever_angle(n) == 0.0)
    check("lever: BUY pulls toward the fly (tip x < pivot x), SELL pushes back (tip x > pivot x)",
          lever_tip_world(lever_angle(b))[0] < LEVER_PIVOT_W[0] < lever_tip_world(lever_angle(sl))[0])
    check("lever: swing limited to +/-35 deg", np.isclose(abs(lever_angle(b)), math.radians(LEVER_MAX_DEG)) and abs(lever_angle(V(lever_buy=5.0, lever_sell=-3.0))) <= math.radians(LEVER_MAX_DEG) + 1e-9)
    check("lever: tip length is constant", all(np.isclose(np.linalg.norm(lever_tip_world(a) - np.array(LEVER_PIVOT_W)), LEVER_LEN) for a in (-0.6, 0.0, 0.6)))
    check("arm target: reach 0 = rest pose", np.allclose(arm_target(n), ARM_REST_TARGET))
    for name, vm in (("neutral", V(reach=1.0)), ("BUY", V(reach=1.0, lever_buy=1.0)), ("SELL", V(reach=1.0, lever_sell=1.0))):
        check(f"arm target: reach 1 ({name}) = knob centre", np.allclose(arm_target(vm) + np.array(FLY_POS), lever_tip_world(lever_angle(vm)), atol=1e-9))
    hip = np.array(ARM_HIP)
    grid = [V(reach=r, lever_buy=bb, lever_sell=ss) for r in (0.0, 0.5, 1.0) for bb, ss in ((0, 0), (1, 0), (0, 1), (0.5, 0))]
    check("arm: every pose is within IK reach", all(np.linalg.norm(arm_target(v) - hip) < sum(ARM_L) - 0.02 for v in grid))
    th = lambda p: ((((np.asarray(p) - THORAX_C) / np.array(THORAX_R)) ** 2).sum(-1))
    hd = lambda p: ((((np.asarray(p) - HEAD_C) / np.array(HEAD_R)) ** 2).sum(-1))
    ok_elbow, ok_pad, ok_hand, ok_cnt = True, True, True, True
    cl, cs = np.array(CONSOLE_CENTER), np.array(CONSOLE_SIZE)
    for v in grid:
        tg = arm_target(v)
        pts = leg_points(ARM_HIP, tg, ARM_L[0], ARM_L[1], ARM_BEND)
        g = build_arm(tg)
        ok_cnt &= g.shape == build_arm().shape and bool(np.isfinite(g).all())
        ok_hand &= bool(np.allclose(pts[2], tg, atol=1e-6))
        ok_elbow &= bool(th(pts[1]) > 1.0 and hd(pts[1]) > 1.0)
        w = g[:, :3] + np.array(FLY_POS, dtype=np.float32)
        over = (np.abs(w[:, 0] - cl[0]) < cs[0] / 2) & (np.abs(w[:, 2] - cl[2]) < cs[2] / 2)
        ok_pad &= bool((w[over][:, 1] > PAD_TOP_Y - 0.005).all())
    check("arm: constant vertex count + finite for every pose", ok_cnt)
    check("arm: hand lands exactly on the target for every pose", ok_hand)
    check("arm: elbow never inside the thorax or head", ok_elbow)
    check("arm: never dips through the console deck", ok_pad)
    e_b = leg_points(ARM_HIP, arm_target(V(reach=1.0, lever_buy=1.0)), *ARM_L, ARM_BEND)[2]
    e_s = leg_points(ARM_HIP, arm_target(V(reach=1.0, lever_sell=1.0)), *ARM_L, ARM_BEND)[2]
    check(f"arm: BUY hand is {e_s[0] - e_b[0]:.2f} units behind-to-ahead of SELL hand (it really moves)", e_s[0] - e_b[0] > 0.4)
    con = build_console()
    check("console: 6 macro keys, whole boxes", len(con.keys) == 6 and all(k.shape == (36, 9) for k in con.keys))
    kc = np.array([k[:, :3].mean(0) for k in con.keys])
    check("keys sit on the pad top, inside the pad", (np.abs(kc[:, 1] - (PAD_TOP_Y + KEY_SIZE[1] / 2)) < 1e-6).all()
          and (np.abs(kc[:, 0] - CONSOLE_CENTER[0]) < PAD_SIZE[0] / 2 - 0.1).all() and (np.abs(kc[:, 2]) < PAD_SIZE[2] / 2 - 0.1).all())
    check("keys: neon colours, more than one hue", len({tuple(np.round(k[0, 6:9], 2)) for k in con.keys}) >= 4)
    pc = np.array([LEVER_BASE_W[0], LEVER_BASE_W[2]])
    check("keys clear of the lever plate", (np.abs(kc[:, 2] - pc[1]) > LEVER_PLATE_SIZE[2] / 2 + KEY_SIZE[2] / 2).all())
    check("lever plate inside the pad", abs(LEVER_BASE_W[0] - CONSOLE_CENTER[0]) + LEVER_PLATE_SIZE[0] / 2 < PAD_SIZE[0] / 2 and abs(LEVER_BASE_W[2]) + LEVER_PLATE_SIZE[2] / 2 < PAD_SIZE[2] / 2)
    lg, lr = con.leds[0].mean(0), con.leds[1].mean(0)
    check("LEDs: green BUY on the fly side, red SELL on the far side, both outside the lever sweep",
          lg[0] < LEVER_PIVOT_W[0] < lr[0] and con.leds[0][0, 7] > 0.9 and con.leds[1][0, 6] > 0.9
          and abs(lg[0] - LEVER_PIVOT_W[0]) > LEVER_LEN * math.sin(math.radians(LEVER_MAX_DEG)) + KNOB_R)
    check("lever geometry finite, tip at +y", np.isfinite(con.lever).all() and np.isclose(con.lever[:, 1].max(), LEVER_LEN + KNOB_R, atol=1e-3))
    mg = np.array(MUG_POS)
    check("mug is clear of the lever plate", mg[0] - MUG_R > LEVER_BASE_W[0] + LEVER_PLATE_SIZE[0] / 2 or abs(mg[2] - LEVER_BASE_W[2]) > 0.6)

    class _Vbo:
        def __init__(self): self.n = 0
        def write(self, b): self.n = len(b)
    m = object.__new__(Mesh)
    m.count, m.vbo = 72, _Vbo()
    m.update(np.zeros((72, 9)))
    check("Mesh.update uploads same-size data", m.vbo.n == 72 * 9 * 4)
    try:
        m.update(np.zeros((70, 9)))
        raised = False
    except ValueError:
        raised = True
    check("Mesh.update rejects a different vertex count", raised)
    cam = Camera(16 / 9)
    pts = np.concatenate([con.solid[:, :3], np.concatenate(con.keys)[:, :3], con.lever[:, :3] + np.array(LEVER_PIVOT_W, dtype=np.float32)])
    c = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1))], 1) @ cam.proj_view.astype(np.float64).T
    nd = c[:, :2] / c[:, 3:4]
    check("console props on screen", (c[:, 3] > 0).all() and (np.abs(nd) < 0.95).all())
    return len(_FAILS) == start


def _selftest_screen(check) -> bool:
    start = len(_FAILS)
    print(" screen texture + chart (9a)")
    check("font: every glyph is 7 rows x 5 columns", all(g.shape == (7, 5) for g in FONT.values()))
    vis = [k for k in FONT if k != " "]
    check("font: all glyphs are distinct", len({FONT[k].tobytes() for k in vis}) == len(vis))
    need = set("ILLUSTRATIVE DATA CRAWLING SEC EDGAR... [FLY BUY] ENTERED LONG POSITION ON MBON HZ: 0123456789.,-+%/$()=")
    check("font: covers the on-screen and log wording", need <= set(FONT))
    cv = Canvas(40, 20, (0, 0, 0))
    cv.text(1, 1, "A", (255, 255, 255))
    check("font: draw_text writes the glyph's pixels", int((cv.px[:, :, 0] == 255).sum()) == int(FONT["A"].sum()))
    cv.text(-30, -30, "HELLO", (1, 2, 3)); cv.text(500, 500, "HELLO", (1, 2, 3)); cv.rect(-5, -5, 9999, 3, (9, 9, 9)); cv.frame(30, 10, 99, 99, (5, 5, 5))
    check("canvas: clipping never raises, shape/dtype stable", cv.px.shape == (20, 40, 3) and cv.px.dtype == np.uint8)
    cv2 = Canvas(100, 20, (0, 0, 0)); cv2.text(0, 0, "ab", (255, 255, 255), scale=2)
    check("font: scale 2 doubles the glyph and lowercase maps to upper", int((cv2.px[:, :, 0] == 255).sum()) == 4 * (int(FONT["A"].sum()) + int(FONT["B"].sum())) and text_width("ab", 2) == 22)
    check("screen is 16:10", SCREEN_W * 10 == SCREEN_H * 16)
    rects = [TITLE_RECT, CHART_RECT, METRICS_RECT, STATUS_RECT]
    inside = all(r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= SCREEN_W and r[1] + r[3] <= SCREEN_H for r in rects)
    def ov(a, b): return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
    check("layout: all panels inside the texture and non-overlapping", inside and not any(ov(rects[i], rects[j]) for i in range(4) for j in range(i + 1, 4)))
    s1, s2, s3 = CandleSeries(11), CandleSeries(11), CandleSeries(12)
    check("candles: deterministic per seed, different across seeds", np.array_equal(s1.c, s2.c) and not np.array_equal(s1.c, s3.c))
    ok = lambda s: (s.h >= np.maximum(s.o, s.c) - 1e-12).all() and (s.l <= np.minimum(s.o, s.c) + 1e-12).all() and (s.l > 0).all()
    check("candles: valid OHLC (high >= body >= low > 0)", ok(s1))
    old = s1.c.copy(); s1.step()
    check("candles: step scrolls by one and chains open = previous close", np.array_equal(s1.c[:-2], old[1:-1]) and np.isclose(s1.o[-1], s1.c[-2]) and ok(s1) and len(s1.c) == s1.n)
    sent = (3, 4, 5)
    cv = Canvas(); cv.fill(sent)
    render_chart(cv, CandleSeries(11), "META")
    changed = np.any(cv.px != np.array(sent, dtype=np.uint8), axis=2)
    mask = np.zeros_like(changed)
    for r in (TITLE_RECT, CHART_RECT):
        mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = True
    check("chart: draws only inside the title bar and chart panel", not (changed & ~mask).any())
    up = (cv.px == np.array(COL_CANDLE_UP, dtype=np.uint8)).all(2).sum()
    dn = (cv.px == np.array(COL_CANDLE_DN, dtype=np.uint8)).all(2).sum()
    check(f"chart: green and red candles both present ({up} / {dn} px)", up > 100 and dn > 100)
    t_ref = Canvas(); t_ref.fill((0, 0, 0)); t_ref.text(SCREEN_W - 6 - text_width(NO_DATE, 2) - 8 - text_width("ILLUSTRATIVE DATA"), 8, "ILLUSTRATIVE DATA", (255, 200, 60))
    check("chart: 'ILLUSTRATIVE DATA' label is drawn in the title bar", ((t_ref.px[:, :, 0] == 255) <= (cv.px[:, :, 0] == 255)).all())
    ser = CandleSeries(11)
    x0 = CHART_RECT[0] + 3
    colcols = lambda i: cv.px[CHART_RECT[1] + 2:CHART_RECT[1] + CHART_RECT[3] - 2, x0 + i * 5 + 2]
    hi_i = int(np.argmax(ser.h)); lo_i = int(np.argmin(ser.l))
    def top_row(i):
        col = colcols(i); m = (col == np.array(COL_CANDLE_UP)).all(1) | (col == np.array(COL_CANDLE_DN)).all(1)
        return np.nonzero(m)[0][[0, -1]]
    check("chart: the series high is nearer the top than the series low", top_row(hi_i)[0] < top_row(lo_i)[1])
    check("chart: every candle's wick+body is drawn (all CHART_CANDLES columns)", all(len(np.nonzero(((colcols(i) == np.array(COL_CANDLE_UP)).all(1) | (colcols(i) == np.array(COL_CANDLE_DN)).all(1)))[0]) > 0 for i in range(ser.n)))
    sc = ScreenContent(11)
    vm = ViewModel(state="RESEARCH", ticker="AAPL")
    check("screen: first update draws", sc.update(vm, 1.0) is True)
    check("screen: throttled to SCREEN_FPS when nothing changed", sc.update(vm, 1.0 + 0.2 * CANDLE_STEP_S / 10) is False)
    before = sc.series.c.copy()
    sc.update(vm, 1.0 + CANDLE_STEP_S + 0.01)
    check("screen: RESEARCH scrolls a new candle every CANDLE_STEP_S", not np.array_equal(before, sc.series.c))
    before = sc.series.c.copy()
    sc.update(ViewModel(state="SILENT", ticker="AAPL"), 5.0)
    check("screen: SILENT freezes the chart", np.array_equal(before, sc.series.c))
    n = sc.draws
    sc.update(ViewModel(state="BUY", ticker="AAPL"), 5.0 + 1e-3)
    check("screen: a state change redraws immediately", sc.draws == n + 1)
    check("screen: ticker shown in the title bar", (sc.canvas.px[5:12, 5:30] == np.array(COL_SCR_TEXT, dtype=np.uint8)).all(2).any())
    q = build_monitor().screen_quad
    g = screen_quad_geo(q)
    check("screen quad: 6 vertices, (x y z u v), uv inside 0..1", g.shape == (6, 5) and g[:, 3:].min() == 0 and g[:, 3:].max() == 1)
    tl = q[0].astype(np.float32); br = q[2].astype(np.float32)
    check("screen quad: top-left corner is uv (0,0), bottom-right is (1,1)",
          any(np.allclose(r[:3], tl) and tuple(r[3:]) == (0, 0) for r in g) and any(np.allclose(r[:3], br) and tuple(r[3:]) == (1, 1) for r in g))
    tri = g[:, :3].reshape(2, 3, 3).astype(np.float64)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    check("screen quad: CCW toward +z (faces out of the monitor)", (fn[:, 2] > 0).all())
    check("screen shaders declare the expected inputs/uniforms",
          all(k in SCR_VERT for k in ("in_pos", "in_uv", "u_mvp")) and all(k in SCR_FRAG for k in ("u_tex", "u_tint", "u_glow")))
    cam = Camera(16 / 9)
    M = monitor_matrix().astype(np.float64)
    c = np.array([np.append(M @ np.append(p.astype(np.float64), 1.0), 0)[:4] for p in q])
    clip = c @ cam.proj_view.astype(np.float64).T
    nd = clip[:, :2] / clip[:, 3:4]
    wpx = (np.ptp(nd[:, 0]) * 640)
    print(f"     (screen is ~{wpx:.0f} px wide at 720p for {SCREEN_W} texels: {SCREEN_W / wpx:.1f} texels per pixel; mip-mapped + anisotropic)")
    return len(_FAILS) == start


def _selftest_screen_panels(check) -> bool:
    start = len(_FAILS)
    print(" screen panels (9b)")
    check("compact_number", [compact_number(v) for v in (950, 12345, 2.5e6, 7.2e9, 3.1e12, -4.0e6)] == ["950", "12.3K", "2.5M", "7.2B", "3.1T", "-4.0M"])
    check("abbreviate_tag: known + CamelCase fallback", abbreviate_tag("Revenues") == "REV" and abbreviate_tag("NetIncomeLoss") == "NI"
          and abbreviate_tag("ResearchAndDevelopmentExpense") == "RADE" and abbreviate_tag("Cash") == "Cash")
    row = format_metric_row("AAPL   Revenues                           123456789  FY2025 filed 2025-11-03")
    check(f"metric row '{row}' parsed to ticker/tag/value", row.startswith("AAPL") and "REV" in row and row.endswith("123M") and len(row) <= METRIC_COLS)
    check("metric row: unparseable line shown raw, truncated", format_metric_row("some other text that is long") == "some other text")
    check("metric rows never exceed the grid width", all(len(format_metric_row(f"MSFTX  StockholdersEquity  {10 ** k}")) <= METRIC_COLS for k in range(1, 14)))
    check("grid text fits inside the panel", METRIC_COLS * 6 - 1 + 4 <= METRICS_RECT[2] - 2)
    feed = SimFeed(start_thread=False)
    vm = feed.view()
    check("sim feed provides parsable raw-metric lines", len(vm.research_lines) > 10 and all(_METRIC_RE.match(l) for l in vm.research_lines))
    R = ViewModel(state="RESEARCH", ticker="AAPL", research_lines=vm.research_lines, research_status="SEARCHING SEC EDGAR...", clock=1.0)

    def shot(v, t, sc=None):
        sc = sc or ScreenContent(11)
        sc.update(v, t)
        return sc

    bg = np.array(COL_SCR_BG, dtype=np.uint8)
    sc = shot(R, 0.01)
    union = np.zeros((SCREEN_H, SCREEN_W), bool)
    for r in (TITLE_RECT, CHART_RECT, METRICS_RECT, STATUS_RECT):
        union[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = True
    check("RESEARCH: everything is drawn inside the four panels", not ((sc.canvas.px != bg).any(2) & ~union).any())
    mx, my, mw, mh = METRICS_RECT
    reg = sc.canvas.px[my + 14:my + mh - 2, mx + 2:mx + mw - 2]
    check("RESEARCH: metrics grid has text rows", (reg == np.array(COL_SCR_VALUE, dtype=np.uint8)).all(2).any() and (reg == np.array(COL_SCR_AMBER, dtype=np.uint8)).all(2).any())
    ref = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0))
    ref.text(STATUS_RECT[0] + 8, STATUS_RECT[1] + 8, "CRAWLING SEC EDGAR...", (255, 255, 255), 2)
    refm = ref.px[:, :, 0] == 255
    on = lambda c: (c.canvas.px[refm] == np.array(COL_SCR_AMBER, dtype=np.uint8)).all(1).all()
    off = lambda c: (c.canvas.px[refm] == np.array(COL_SCR_PANEL, dtype=np.uint8)).all(1).all()
    check("RESEARCH: 'CRAWLING SEC EDGAR...' drawn at scale 2 (flash on)", on(shot(R, 0.01)))
    check("RESEARCH: ... and gone half a blink period later (flashes)", off(shot(R, 1.0 / (BLINK_HZ * 2) + 0.01)))
    flips = [on(shot(R, k / 20.0 + 0.001)) for k in range(0, 20)]
    check(f"RESEARCH: blink toggles at {BLINK_HZ:g} Hz ({sum(flips)}/20 samples on)", 6 <= sum(flips) <= 14)
    check("status text fits inside the panel", text_width("CRAWLING SEC EDGAR...", 2) + 8 < STATUS_RECT[2] - 8 and 6 * 60 - 1 < STATUS_RECT[2])
    a = shot(R, 0.0)
    a.update(R, 1.0)
    first = a.canvas.px[my:my + mh, mx:mx + mw].copy()
    a.update(R, 2.0)
    check("RESEARCH: metrics grid scrolls over time", a.scroll > 2.5 and not np.array_equal(first, a.canvas.px[my:my + mh, mx:mx + mw]))
    H = ViewModel(state="HOLD", ticker="AAPL", research_lines=vm.research_lines)
    b = shot(R, 0.0)
    b.update(R, 1.0)
    sc0 = b.scroll
    b.update(H, 3.0)
    check("HOLD: grid stops scrolling and dims", b.scroll == sc0 and not (b.canvas.px[my + 14:my + mh - 2, mx + 2:mx + mw - 2] == np.array(COL_SCR_VALUE, dtype=np.uint8)).all(2).any())
    tkref = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0))
    tkref.text(STATUS_RECT[0] + 8, STATUS_RECT[1] + 10, "TRACKING MARKET FLOW AAPL"[:(STATUS_RECT[2] - 16) // 12], (255, 255, 255), 2)
    check("HOLD: status panel says tracking market flow (and no crawl)", ((tkref.px[:, :, 0] == 255) <= (b.canvas.px == np.array(COL_SCR_TEXT, dtype=np.uint8)).all(2)).all())
    for st, word in (("BUY", "BUY ORDER AAPL"), ("SELL", "SELL ORDER AAPL")):
        c = shot(ViewModel(state=st, ticker="AAPL", research_lines=vm.research_lines), 0.0)
        rr = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0)); rr.text(STATUS_RECT[0] + 8, STATUS_RECT[1] + 10, word, (255, 255, 255), 2)
        col = np.array(COL_CANDLE_UP if st == "BUY" else COL_CANDLE_DN, dtype=np.uint8)
        check(f"{st}: status panel shows '{word}' in {'green' if st == 'BUY' else 'red'}", (c.canvas.px[rr.px[:, :, 0] == 255] == col).all(1).all())
    S = ViewModel(state="SILENT", ticker="AAPL", research_lines=vm.research_lines)
    live = shot(R, 0.01).canvas.px.astype(np.float64)
    s1 = shot(S, 0.01)
    dim = s1.canvas.px.astype(np.float64)
    check(f"SILENT: screen is much dimmer ({dim.mean() / live.mean():.2f} x RESEARCH)", dim.mean() < 0.5 * live.mean())
    check("SILENT: static horizontal scanlines (even rows darker, every row constant gain)",
          dim[::2].mean() < 0.8 * dim[1::2].mean())
    s2 = ScreenContent(11)
    s2.update(S, 0.01)
    s2.update(S, 2.0)
    check("SILENT: pattern is static (identical frames over time)", np.array_equal(s1.canvas.px, s2.canvas.px))
    check("SILENT: no 'CRAWLING' text, panel reads cull filter", not on(s1) and (s1.canvas.px[STATUS_RECT[1] + 8:STATUS_RECT[1] + 24, STATUS_RECT[0] + 8:STATUS_RECT[0] + 300] > 0).any())
    flat = Canvas(40, 40, (200, 200, 200)); apply_silent(flat)
    rows = flat.px[:, 0, 0].astype(float)
    check("apply_silent: scanlines vary row to row, every column identical", rows.std() > 3 and (flat.px == flat.px[:, :1]).all())
    check("illustrative notice visible in non-research states", (b.canvas.px[STATUS_RECT[1] + 36:STATUS_RECT[1] + 43, STATUS_RECT[0] + 8:STATUS_RECT[0] + 150] == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2).any())
    return len(_FAILS) == start


def _selftest_aspect(check) -> bool:
    start = len(_FAILS)
    print(" window shapes / fullscreen camera (9c)")
    ref = 16 / 9
    c169, c43, c1610, c219 = Camera(ref), Camera(4 / 3), Camera(16 / 10), Camera(21 / 9)
    check("camera: 16:9 keeps the 36 deg vertical fov", np.isclose(c169.fov, 36.0))
    check("camera: wider than 16:9 keeps the vertical fov (shows more city)", np.isclose(c219.fov, 36.0))
    hfov = lambda c: 2 * math.atan(math.tan(math.radians(c.fov) / 2) * c.aspect)
    check("camera: narrower windows keep the 16:9 horizontal fov (taller view, nothing cropped)",
          all(np.isclose(hfov(c), hfov(c169)) for c in (c43, c1610)) and c43.fov > c1610.fov > 38.0)
    c1610.set_aspect(ref)
    check("camera: set_aspect is reversible", np.isclose(c1610.fov, 36.0) and np.isclose(c1610.aspect, ref))
    d = build_desk()
    mon = build_monitor()
    Mm = monitor_matrix().astype(np.float64)
    pts = [np.array([[*(np.array(b[:3]) + np.array(b[3:6]) * [sx, sy, sz] / 2)] for b in d.boxes[1:2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]),
           np.array([(Mm @ np.append(q, 1.0))[:3] for q in mon.screen_quad]),
           build_fly_body().head[:, :3] + np.array(FLY_POS), build_chair().solid[:, :3].astype(np.float64)]
    pts = np.concatenate([np.asarray(x, dtype=np.float64) for x in pts])
    Mt = tower_matrix().astype(np.float64)
    W_, H_, D_ = TOWER_SIZE
    pts = np.concatenate([pts, np.array([(Mt @ np.array([a * W_ / 2, b * H_ / 2, c * D_ / 2, 1.0]))[:3] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])])
    for name, asp in (("21:9", 21 / 9), ("16:9", 16 / 9), ("16:10", 16 / 10), ("4:3", 4 / 3), ("5:4", 5 / 4)):
        cam = Camera(asp)
        cl = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ cam.proj_view.astype(np.float64).T
        nd = cl[:, :2] / cl[:, 3:4]
        check(f"{name}: desk, monitor, fly, chair and tower all fit on screen", (cl[:, 3] > 0).all() and (np.abs(nd[:, 0]) < 1.0).all() and (np.abs(nd[:, 1]) < 1.0).all())
        inv = np.linalg.inv(cam.proj_view.astype(np.float64))
        B = build_skyline(x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG).boxes
        lo = np.stack([B[:, 0] - B[:, 3] / 2, B[:, 1] - B[:, 4] / 2, B[:, 2] - B[:, 5] / 2], 1)
        hi = np.stack([B[:, 0] + B[:, 3] / 2, B[:, 1] + B[:, 4] / 2, B[:, 2] + B[:, 5] / 2], 1)
        eye = cam.eye.astype(np.float64)
        holes = 0
        for nx in np.linspace(-1, 1, 25):
            for ny in np.linspace(-1, 1, 25):
                pp = inv @ np.array([nx, ny, 1.0, 1.0])
                dr = pp[:3] / pp[3] - eye
                dr /= np.linalg.norm(dr)
                if dr[1] > -0.05:
                    continue
                with np.errstate(all="ignore"):
                    t1, t2 = (lo - eye) / dr, (hi - eye) / dr
                hit = (np.maximum(t1, t2).min(1) >= np.maximum(np.minimum(t1, t2).max(1), 0)).any()
                tt = (FLOOR_TOP_Y - eye[1]) / dr[1]
                fp = eye + dr * tt
                floor = abs(fp[0] - FLOOR_CENTER_X) < FLOOR_SIZE[0] / 2 and FLOOR_CENTER_Z - FLOOR_SIZE[2] / 2 < fp[2] < FLOOR_CENTER_Z + FLOOR_SIZE[2] / 2
                holes += int(not hit and not floor)
        check(f"{name}: no empty void in the lower half of the view ({holes} stray rays)", holes == 0)
    return len(_FAILS) == start


def _selftest_side_monitor(check) -> bool:
    start = len(_FAILS)
    print(" long desk + side monitor + decision tape (9d)")
    d = build_desk()
    top = d.boxes[1]
    check("desk legs are dark (the fly's leg colour must not leak into the desk)", np.allclose(d.solid[36 * 1:36 * 1 + 36, 6:9], COL_LEG) and max(COL_LEG) < 0.2 and COL_FLY_LEG != COL_LEG)
    check("long desk: four legs, one per corner, outermost under the right end", len(LEG_XZ) == 4 and max(x for x, _ in LEG_XZ) > top[0] + top[3] / 2 - 0.5)
    cam = Camera(16 / 9)
    pv = cam.proj_view.astype(np.float64)
    q = build_monitor().screen_quad
    boxes = build_monitor().boxes
    nds = {}
    for sp in MONITORS:
        M = monitor_matrix(sp).astype(np.float64)
        w = lambda p: (M @ np.append(p, 1.0))[:3]
        base_c, base_s = boxes[1]
        check(f"{sp.name} monitor: base bottom on the desk top (y = 0)", abs(w(np.array(base_c) - [0, base_s[1] / 2, 0])[1]) < 1e-6)
        pts = np.array([w(np.array(c) + np.array(sz) * [a, b, e] / 2) for c, sz in boxes for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)])
        check(f"{sp.name} monitor: fully on the desk slab", (pts[:, 0] > top[0] - top[3] / 2).all() and (pts[:, 0] < top[0] + top[3] / 2).all() and (np.abs(pts[:, 2]) < top[5] / 2).all())
        sc = np.array([w(c) for c in q])
        n = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
        n /= np.linalg.norm(n)
        check(f"{sp.name} screen faces the camera", float(n @ (cam.eye - sc.mean(0))) > 0)
        clip = np.concatenate([sc, np.ones((4, 1))], 1) @ pv.T
        nd = clip[:, :2] / clip[:, 3:4]
        nds[sp.name] = nd
        wpx = np.ptp(nd[:, 0]) * 640
        check(f"{sp.name} screen: on screen, below the HUD strip, {wpx:.0f} px wide at 720p (> 110)", (np.abs(nd[:, 0]) < 0.95).all() and (nd[:, 1] < 0.8).all() and wpx > 110)
    a, b = nds["main"], nds["side"]
    check("side and main screens do not overlap in the view", a[:, 0].max() <= b[:, 0].min() + 0.01 or b[:, 0].max() <= a[:, 0].min() + 0.01)
    Mm, Ms = monitor_matrix(MAIN_MONITOR).astype(np.float64), monitor_matrix(SIDE_MONITOR).astype(np.float64)
    seg = lambda M: ((M @ np.array([-1.4, 0, 0, 1.0]))[[0, 2]], (M @ np.array([1.4, 0, 0, 1.0]))[[0, 2]])
    (a0, a1), (b0, b1) = seg(Mm), seg(Ms)
    cross = lambda o, p, q_: (p[0] - o[0]) * (q_[1] - o[1]) - (p[1] - o[1]) * (q_[0] - o[0])
    inter = cross(a0, a1, b0) * cross(a0, a1, b1) < 0 and cross(b0, b1, a0) * cross(b0, b1, a1) < 0
    dmin = min(np.linalg.norm(p - q_) for p in (a0, a1) for q_ in (b0, b1))
    check(f"monitor bodies don't intersect (closest ends {dmin:.2f} apart)", not inter and dmin > 0.3)
    check("both monitors are the same model (shared geometry) at different scales", MAIN_MONITOR.scale != SIDE_MONITOR.scale == IC_MONITOR.scale and len(MONITORS) == 3)

    sim = SimFeed(start_thread=False)
    V = lambda **k: ViewModel(**k)
    tape = [("AAPL", 2.0), ("MSFT", -4.0), ("NVDA", 1.0)]
    vm = V(state="HOLD", tape=tape, header=dict(top_id="fly_017", top_ic=0.0213, ridge_ic=0.004), replay_label="SIMULATED FEED")
    cv = Canvas()
    render_tape(cv, vm)
    px = cv.px
    lx, ly, lw, lh = TAPE_LIST_RECT
    row = lambda i: slice(ly + 4 + i * TAPE_ROW_H + 4, ly + 4 + i * TAPE_ROW_H + 13)
    green, red = np.array(COL_CANDLE_UP, dtype=np.uint8), np.array(COL_CANDLE_DN, dtype=np.uint8)
    r0 = (px[row(0), :] == red).all(2)            # newest first: NVDA (+1.0)... tape[-1]
    g0 = (px[row(0), :] == green).all(2)
    check("tape: newest decision first (NVDA +1.0 on the first row, green bar right of zero, none on the left)", g0[:, TAPE_BAR_CX:].any() and not g0[:, lx + 100:TAPE_BAR_CX].any())
    r1 = (px[row(1), :] == red).all(2)
    check("tape: negative score (MSFT -4.0) is a red bar left of the zero line", r1[:, :TAPE_BAR_CX].any() and not r1[:, TAPE_BAR_CX + 1:].any())
    bar = lambda m: int(m.any(0).sum())
    wmax = int((px[row(1), :] == red).all(2).any(0)[:TAPE_BAR_CX].sum())
    check(f"tape: largest |score| gets the full bar ({TAPE_BAR_HALF} px), others proportional", wmax >= TAPE_BAR_HALF - 1)
    w2 = int((px[row(2), :] == green).all(2).any(0)[TAPE_BAR_CX:].sum())
    check("tape: AAPL +2.0 bar is half of the MSFT -4.0 bar", abs(w2 - TAPE_BAR_HALF // 2) <= 2)
    amber = (px[row(0), lx + 8:lx + 62] == np.array(COL_SCR_AMBER, dtype=np.uint8)).all(2).any()
    check("tape: newest ticker highlighted amber", amber)
    foot = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0))
    foot.text(TAPE_FOOT_RECT[0] + 6, TAPE_FOOT_RECT[1] + 4, "fly_017 IC +0.021", (255, 255, 255), 2)
    ridge_m = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0))
    ridge_m.text(TAPE_FOOT_RECT[0] + 6, TAPE_FOOT_RECT[1] + 21, "RIDGE +0.004", (255, 255, 255), 2)
    check("tape footer: top fly id and IC at 2x, ridge IC on the next line",
          ((foot.px[:, :, 0] == 255) <= (px == np.array(COL_SCR_TEXT, dtype=np.uint8)).all(2)).all() and
          ((ridge_m.px[:, :, 0] == 255) <= (px == np.array(COL_SCR_VALUE, dtype=np.uint8)).all(2)).all())
    lab = Canvas(SCREEN_W, SCREEN_H, (0, 0, 0))
    lab.text(TAPE_FOOT_RECT[0] + TAPE_FOOT_RECT[2] - 6 - text_width("SIMULATED FEED"), TAPE_FOOT_RECT[1] + 8, "SIMULATED FEED", (255, 255, 255))
    check("tape footer: shows the honest source label", ((lab.px[:, :, 0] == 255) <= (px == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2)).all())
    union = np.zeros((SCREEN_H, SCREEN_W), bool)
    for r in (TITLE2_RECT, TAPE_LIST_RECT, TAPE_FOOT_RECT):
        union[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = True
    check("tape: everything drawn inside the title bar, list and footer", not ((px != np.array(COL_SCR_BG, dtype=np.uint8)).any(2) & ~union).any())
    big = [(f"T{i}", float(i) - 7.0) for i in range(40)]
    cv2 = Canvas(); render_tape(cv2, V(tape=big))
    check(f"tape: long tapes are clipped to {TAPE_ROWS} rows without overdraw", not ((cv2.px != np.array(COL_SCR_BG, dtype=np.uint8)).any(2) & ~union).any())
    cv3 = Canvas(); render_tape(cv3, V(tape=[("BAD", "x"), ("OK", None)], header={}))
    check("tape: junk scores and an empty header never raise", cv3.px.shape == (SCREEN_H, SCREEN_W, 3))
    cv4 = Canvas(); render_tape(cv4, V())
    check("tape: empty tape says it is waiting", (cv4.px[ly + 8:ly + 22, lx + 8:lx + 160] == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2).any())
    tc = TapeContent()
    check("TapeContent: first update draws", tc.update(vm, 0.0) is True)
    check("TapeContent: unchanged tape is not redrawn", tc.update(vm, 0.5) is False)
    vm2 = V(state="HOLD", tape=tape + [("TSLA", 3.0)], header=vm.header, replay_label="SIMULATED FEED")
    check("TapeContent: a new decision redraws", tc.update(vm2, 1.0) is True and tc.draws == 2)
    live = tc.canvas.px.astype(float).mean()
    tc.update(V(state="SILENT", tape=tape, header=vm.header), 2.0)
    check("TapeContent: SILENT dims the side screen too", tc.canvas.px.astype(float).mean() < 0.5 * live)
    check("sim feed provides the tape as a list for the side screen", isinstance(sim.view().tape, list))
    return len(_FAILS) == start


def _selftest_ic_screen(check) -> bool:
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" IC history screen (9f-a)")
    bg = np.array(COL_SCR_BG, dtype=np.uint8)
    hist = [(r, 0.02 + 0.01 * r, 0.01 * r, -0.005 * r) for r in range(1, 21)]
    rm = rolling_mean([1.0, 2.0, 3.0, 4.0, None, 6.0])
    check("rolling mean: trailing 3 window, ramps in", np.allclose(rm[:4], [1.0, 1.5, 2.0, 3.0]))
    check("rolling mean: missing values skipped, all-missing window is None", np.isclose(rm[5], 5.0) and rolling_mean([None, float("nan")]) == [None, None])
    check("ViewModel carries ic_history (empty by default)", ViewModel().ic_history == [])
    cv0 = Canvas(); render_ic(cv0, NS())
    px, py, pw, ph = IC_PLOT_RECT
    check("ic: empty history says STANDBY", (cv0.px[py + 8:py + 22, px + 8:px + 200] == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2).any())
    cv = Canvas(); render_ic(cv, NS(ic_history=hist, replay_label="SIM FEED"))
    for nm, col in (("top", COL_IC_TOP), ("3rd avg", COL_IC_AVG), ("ridge", COL_IC_RIDGE), ("untrained", COL_IC_UNTR)):
        check(f"ic: {nm} line is drawn in the plot", (cv.px[py:py + ph, px:px + pw] == np.array(col, dtype=np.uint8)).all(2).any())
    union = np.zeros((SCREEN_H, SCREEN_W), dtype=bool)
    for rx, ry, rw, rh in (TITLE2_RECT, IC_PLOT_RECT, IC_FOOT_RECT):
        union[ry:ry + rh, rx:rx + rw] = True
    check("ic: everything drawn inside the title bar, plot and footer", not ((cv.px != bg).any(2) & ~union).any())
    check("ic: plot frame stays intact", (cv.px[py, px:px + pw] == np.array(COL_SCR_BORDER, dtype=np.uint8)).all())
    # zero line: with symmetric data +/-0.1 the zero line sits mid-plot
    sym = [(r, v, v, v) for r, v in enumerate([0.1, -0.1, 0.1, -0.1], 1)]
    cs = Canvas(); render_ic(cs, NS(ic_history=sym))
    rows_z = np.nonzero((cs.px[py:py + ph, px + IC_LABEL_W + 5:px + pw - 8] == np.array(COL_IC_ZERO, dtype=np.uint8)).all(2).any(1))[0]
    check("ic: IC = 0 line is drawn near the middle for symmetric data", len(rows_z) >= 1 and abs(rows_z.mean() - ph / 2) < 8)
    check("ic: footer shows the latest top IC", (cv.px[IC_FOOT_RECT[1] + 4:IC_FOOT_RECT[1] + 18, IC_FOOT_RECT[0] + 6:IC_FOOT_RECT[0] + 190] == np.array(COL_IC_TOP, dtype=np.uint8)).all(2).any())
    check("ic: honest source label in the footer", (cv.px[IC_FOOT_RECT[1] + 44:IC_FOOT_RECT[1] + 51, IC_FOOT_RECT[0] + 112:IC_FOOT_RECT[0] + 112 + 60] == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2).any())
    for name, bad in (("junk rows", [("a", "x", None, float("nan")), (2,), None, ("b", 1e9, -1e9, 0.0), (3, float("inf"), 0.1, 0.2)]),
                      ("one point", [(1, 0.05, 0.02, 0.0)]), ("all None", [(1, None, None, None)]), ("huge history", [(r, 0.01 * (r % 7), 0.0, 0.0) for r in range(500)])):
        c = Canvas(); render_ic(c, NS(ic_history=bad))
        check(f"ic: {name} never raises and stays on the canvas", c.px.shape == (SCREEN_H, SCREEN_W, 3))
    big = Canvas(); render_ic(big, NS(ic_history=[(r, 0.01 * (r % 7), 0.0, 0.0) for r in range(500)]))
    check("ic: only the newest rounds are plotted (footer count capped)", (big.px != bg).any())
    ic = IcContent()
    check("IcContent: first update draws", ic.update(NS(ic_history=hist, state="IDLE"), 0.0) and ic.draws == 1)
    check("IcContent: unchanged history is not redrawn", not ic.update(NS(ic_history=hist, state="IDLE"), 0.1) and ic.draws == 1)
    check("IcContent: a new round redraws", ic.update(NS(ic_history=hist + [(21, 0.3, 0.1, 0.0)], state="IDLE"), 0.2) and ic.draws == 2)
    before = ic.canvas.px.copy()
    ic.update(NS(ic_history=hist + [(21, 0.3, 0.1, 0.0)], state="SILENT"), 0.3)
    check("IcContent: SILENT dims the IC screen too", ic.canvas.px.astype(int).sum() < before.astype(int).sum() * 0.7)
    return len(_FAILS) == start


def _selftest_ic_feed(check) -> bool:
    import inspect
    start = len(_FAILS)
    print(" IC history feed + third monitor (9f-b)")
    acc = IcAccumulator()
    check("ic feed: empty header adds nothing", acc.add({}) == [] and acc.add({"round": 1}) == [] and acc.add(None) == [])
    r = acc.add({"round": 1, "top_ic": 0.02, "ridge_ic": 0.01, "untrained_ic": 0.0})
    check("ic feed: first round makes a row", r == [(1, 0.02, 0.01, 0.0)])
    r = acc.add({"round": 1, "top_ic": 0.03, "ridge_ic": 0.01, "untrained_ic": 0.0})
    check("ic feed: same round updates its row in place", len(r) == 1 and r[0][1] == 0.03)
    r = acc.add({"round": 2, "top_ic": "x", "ridge_ic": None, "untrained_ic": 0.1})
    check("ic feed: a new round appends; junk values become None, not rows of text", len(r) == 2 and r[1] == (2, None, None, 0.1))
    small = IcAccumulator(maxlen=5)
    for i in range(20):
        small.add({"round": i, "top_ic": i * 0.01})
    check("ic feed: history is capped (newest kept)", len(small.rows) == 5 and small.rows[-1][0] == 19)
    sim = SimFeed(start_thread=False)
    v0 = sim.view()
    check("sim: IC history starts with the round-0 reading", len(v0.ic_history) == 1 and v0.ic_history[0][0] == 0)
    sim._data["mode"] = "league"
    sim._made = 10
    sim.produce_once()
    v1 = sim.view()
    check("sim: finishing a round appends a new IC row", len(v1.ic_history) == 2 and v1.ic_history[-1][0] == 1)
    check("sim: ridge and untrained IC drift too", v1.ic_history[-1][2:] != v0.ic_history[-1][2:])
    check("sim: view history is a copy (renderer cannot corrupt the feed)", v1.ic_history is not sim._ic_rows)
    src = inspect.getsource(VisualDashboard.render) + inspect.getsource(VisualDashboard.__init__)
    check("app: third monitor and its IC screen are built and drawn", "ic_monitor" in src and "ic_screen" in src and "IcContent()" in src)
    check("app: third monitor uses the IC_MONITOR placement", "monitor_matrix(IC_MONITOR)" in src)
    return len(_FAILS) == start


def _selftest_hud(check) -> bool:
    import inspect
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" HUD top bar (10a)")
    for asp in (16 / 9, 16 / 10, 21 / 9, 4 / 3):
        q = np.array(hud_quad_ndc(asp))
        px_w, px_h = 1000.0 * asp, 1000.0
        bar_px_h = (q[0, 1] - q[2, 1]) / 2 * px_h
        check(f"hud quad {asp:.2f}: full width, glued to the top, texel-square ({bar_px_h:.0f} px tall for {px_w:.0f} wide)",
              np.allclose(q[:, 0], (-1, 1, 1, -1)) and q[0, 1] == 1.0 and abs(bar_px_h - px_w * HUD_H / HUD_W) < 1e-6)
    check("hud quad: clamped on very tall windows", hud_quad_ndc(0.2)[0][1] - hud_quad_ndc(0.2)[2][1] <= 0.6 + 1e-9 and hud_quad_ndc(0.2)[0][1] - hud_quad_ndc(50.0)[2][1] <= 0.6 + 1e-9)
    hist = [(1, 0.10, 0, 0), (2, 0.20, 0, 0), (3, 0.30, 0, 0), (4, 0.60, 0, 0)]
    vm = NS(header=dict(round=12, universe=499, top_id="fly_017"), ic_history=hist)
    v = hud_values(vm)
    check("hud values: round, universe, top fly", v["round"] == "12" and v["universe"] == "499" and v["top_id"] == "fly_017")
    check("hud values: avg IC is the 3-round rolling mean of the top IC (0.2, 0.3, 0.6 -> 0.3667)", abs(v["avg_ic"] - (0.2 + 0.3 + 0.6) / 3) < 1e-9)
    e = hud_values(NS())
    check("hud values: nothing known shows '--', never a made-up number", e == dict(round="--", universe="--", top_id="--", avg_ic=None))
    cv = Canvas(HUD_W, HUD_H); render_hud(cv, vm)
    pxs = cv.px
    check("hud: positive average IC is green", (pxs[12:26, 3 * HUD_CELL_W:] == np.array(COL_CANDLE_UP, dtype=np.uint8)).all(2).any())
    cn = Canvas(HUD_W, HUD_H); render_hud(cn, NS(header={}, ic_history=[(1, -0.1, 0, 0)]))
    check("hud: negative average IC is red", (cn.px[12:26, 3 * HUD_CELL_W:] == np.array(COL_CANDLE_DN, dtype=np.uint8)).all(2).any())
    check("hud: top fly is amber", (pxs[12:26, 2 * HUD_CELL_W:3 * HUD_CELL_W] == np.array(COL_SCR_AMBER, dtype=np.uint8)).all(2).any())
    bg = np.array(COL_HUD_BG, dtype=np.uint8)
    for i in range(4):
        lab = (pxs[3:10, i * HUD_CELL_W + 10:(i + 1) * HUD_CELL_W] == np.array(COL_SCR_DIM, dtype=np.uint8)).all(2).any()
        check(f"hud cell {i}: has its small label", lab)
    worst = NS(header=dict(round="99999999", universe="99999999", top_id="x" * 30), ic_history=[(1, -0.123, 0, 0)])
    cw = Canvas(HUD_W, HUD_H); render_hud(cw, worst)
    seps = [i * HUD_CELL_W for i in range(1, 4)]
    clean = all((cw.px[3:HUD_H - 5, sx] == np.array(COL_SCR_GRID, dtype=np.uint8)).all() for sx in seps)
    check("hud: the longest values stay inside their cells (separators untouched)", clean)
    check("hud: bottom border line drawn", (cw.px[HUD_H - 1] == np.array(COL_SCR_BORDER, dtype=np.uint8)).all())
    for name, bad in (("junk header", NS(header=dict(round=object(), universe=[1], top_id=None), ic_history=[("a", "b", None, None), None])), ("none", NS(header=None, ic_history=None))):
        c = Canvas(HUD_W, HUD_H); render_hud(c, bad)
        check(f"hud: {name} never raises", c.px.shape == (HUD_H, HUD_W, 3))
    hc = HudContent()
    check("HudContent: first update draws", hc.update(vm) is True and hc.draws == 1)
    check("HudContent: unchanged values are not redrawn", hc.update(vm) is False and hc.draws == 1)
    vm2 = NS(header=dict(round=13, universe=499, top_id="fly_017"), ic_history=hist)
    check("HudContent: a new round redraws", hc.update(vm2) is True and hc.draws == 2)
    src = inspect.getsource(VisualDashboard.render) + inspect.getsource(VisualDashboard.__init__) + inspect.getsource(HudRenderer.draw)
    check("app: HUD built, drawn last, with depth test off while drawing", "HudRenderer(" in src and "self.hud.draw(" in src and "DEPTH_TEST" in src)
    check("app: HUD drawn after the wings (on top of everything)", src.index("self.wings.draw") < src.index("self.hud.draw("))
    return len(_FAILS) == start


def _selftest_log_zzz(check) -> bool:
    import inspect
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" log stream + floating Zzz (10b)")
    for asp in (16 / 9, 16 / 10, 21 / 9, 4 / 3):
        q = np.array(hud_quad_ndc(asp)), np.array(log_quad_ndc(asp))
        lq = q[1]
        w_px, h_px = 1000.0 * asp, 1000.0
        qw, qh = (lq[1, 0] - lq[0, 0]) / 2 * w_px, (lq[0, 1] - lq[3, 1]) / 2 * h_px
        check(f"log quad {asp:.2f}: bottom-left, on screen, texel-square ({qw / LOG_W:.3f} vs {w_px / HUD_W:.3f} px per texel)",
              (np.abs(lq[:, :2]) <= 1.0).all() and lq[3, 0] < -0.9 and lq[3, 1] < -0.8 and lq[3, 1] >= -1.0 + 0.09 and abs(qw / LOG_W - qh / LOG_H) < 1e-6)
    long_msg = "[FLY BUY] Entered Long Position on NVDA - MBON Hz: +1.5"
    rows = wrap_log(long_msg)
    check("wrap_log: rows fit the panel and continuation rows are indented", all(len(r) <= LOG_COLS for r in rows) and len(rows) >= 2 and rows[1].startswith("  "))
    check("wrap_log: no text lost", "".join(long_msg.split()) == "".join("".join(rows).split()))
    check("wrap_log: short line stays one row, empty text one blank row", wrap_log("[FLY HOLD] Defending Position on AAPL") == ["[FLY HOLD] Defending Position on AAPL"][:1] or len(wrap_log("[FLY HOLD] Defending Position on AAPL")) <= 2)
    wl = wrap_log("x" * 100)
    check("wrap_log: a single over-long word is split, nothing lost or overlong", all(len(r) <= LOG_COLS for r in wl) and "".join("".join(wl).split()) == "x" * 100)
    check("wrap_log: empty text gives one blank row", wrap_log("") == [""])
    green, red = (57, 255, 20), (255, 40, 50)
    log = [(green, "[FLY BUY] Entered Long Position on NVDA - MBON Hz: +1.5"), (red, "[FLY SELL] Covered Short Position on AAPL - MBON Hz: -2.0")]
    lr = log_rows(log)
    first_sell = [r for r in lr if r[1].startswith("[FLY SELL]")]
    check("log rows: oldest first, newest last, rank 0 is the newest", lr[-1][2] == 0 and lr[0][2] == 1 and len(first_sell) == 1 and first_sell[0][2] == 0 and lr[0][1].startswith("[FLY BUY]"))
    many = [(green, f"[FLY BUY] Entered Long Position on T{i} - MBON Hz: +1.0") for i in range(30)]
    lm = log_rows(many)
    check(f"log rows: capped at {LOG_ROWS} rows, whole messages only, newest kept", len(lm) <= LOG_ROWS and "T29" in "".join(r[1] for r in lm) and "T0 " not in "".join(r[1] for r in lm) and lm[0][1].startswith("[FLY"))
    check("log rows: junk entries are skipped without raising", log_rows([None, 5, ("x",), ((1, 2), "a"), (green, "ok"), ("rgb", "b")]) [-1][1] == "ok")
    check("log rows: empty or None log gives no rows", log_rows([]) == [] and log_rows(None) == [])
    cv = Canvas(LOG_W, LOG_H, COL_LOG_BG)
    shown = render_log(cv, NS(log=log))
    check("log: both colours drawn (newest at full brightness)", shown and (cv.px == np.array(red, dtype=np.uint8)).all(2).any() and ((cv.px.astype(int) > 0) & (cv.px.astype(int) <= np.array(green))).all(2).any())
    ce = Canvas(LOG_W, LOG_H, COL_LOG_BG)
    check("log: nothing to show leaves the canvas empty", render_log(ce, NS(log=[])) is False and (ce.px == np.array(COL_LOG_BG, dtype=np.uint8)).all())
    rows_y = np.nonzero((cv.px != np.array(COL_LOG_BG, dtype=np.uint8)).any(2).any(1))[0]
    check("log: text stays inside the panel, bottom-aligned", rows_y.min() >= 0 and rows_y.max() < LOG_H - 1 and rows_y.max() > LOG_H - 14)
    lc = LogContent()
    check("LogContent: no log means fully transparent", lc.update(NS(log=[])) and (lc.pixels()[:, :, 3] == 0).all())
    check("LogContent: first log redraws", lc.update(NS(log=log)) is True)
    check("LogContent: unchanged log is not redrawn", lc.update(NS(log=log)) is False)
    pxl = lc.pixels()
    check("LogContent: RGBA, translucent panel, opaque text", pxl.shape == (LOG_H, LOG_W, 4) and (pxl[:, :, 3] == LOG_BG_ALPHA).any() and (pxl[:, :, 3] == 255).any())
    check("LogContent: a new line redraws", lc.update(NS(log=log + [(green, "new")])) is True)
    sim = SimFeed(start_thread=False)
    sim.log.append(format_log("BUY", "NVDA", 1.5, "a"))
    check("log: the sim feed's own wording renders", "[FLY BUY] Bought" in log_rows(sim.view().log)[0][1] and "NVDA" in "".join(r[1] for r in log_rows(sim.view().log)))
    # Zzz
    check("zzz: nothing when awake / junk", zzz_params(0.0) == [] and zzz_params(-1) == [] and zzz_params(None) == [] and zzz_params(float("nan")) == [])
    check("zzz: one Z at first, then up to three", len(zzz_params(0.1)) == 1 and len(zzz_params(1.0)) == 2 and len(zzz_params(5.0)) == ZZZ_COUNT)
    ok_alpha = all(0.0 <= p["alpha"] <= 1.0 for z in np.linspace(0.01, 8, 80) for p in zzz_params(z))
    check("zzz: alpha always in 0..1", ok_alpha)
    a1, a2 = zzz_params(0.3)[0], zzz_params(0.9)[0]
    check("zzz: a Z drifts up and grows as it ages", a2["offset"][1] > a1["offset"][1] and a2["size"] > a1["size"])
    check("zzz: fades in from nothing", zzz_params(0.001)[0]["alpha"] < 0.05)
    g = z_glyph_geo()
    check("zzz glyph: finite, whole triangles, stride 9, 3 boxes", g.shape[1] == 9 and len(g) == 3 * 36 and np.isfinite(g).all())
    check("zzz glyph: about 1 wide and 1 tall, centred", abs(g[:, 0].max() - 0.5) < 0.1 and abs(g[:, 0].min() + 0.5) < 0.1 and abs(g[:, 1].max() - 0.5) < 0.01 and abs(g[:, 1].min() + 0.5) < 0.01)
    # Z shape: the diagonal runs from the top-right to the bottom-left
    diag = g[72:]
    check("zzz glyph: the diagonal leans from the top-right down to the bottom-left", diag[diag[:, 1] > 0, 0].mean() > diag[diag[:, 1] < 0, 0].mean() + 0.3)
    eye = Camera(16 / 9).eye
    par = zzz_params(0.8)[0]
    M = zzz_matrix(par, eye).astype(np.float64)
    pos = M[:3, 3]
    front = M[:3, :3] @ np.array([0.0, 0.0, 1.0])
    to_cam = eye - pos
    fh, ch = front[[0, 2]], to_cam[[0, 2]]
    check("zzz: billboard front faces the camera (turns about the vertical axis only)", front[1] == 0.0 and float(fh @ ch) / (np.linalg.norm(fh) * np.linalg.norm(ch)) > 0.999)
    check("zzz: floats above the fly's head", pos[1] > FLY_POS[1] + HEAD_C[1] + HEAD_R[1] and abs(pos[0] - (FLY_POS[0] + HEAD_C[0])) < 0.5)
    clip = Camera(16 / 9).proj_view.astype(np.float64) @ np.append(pos, 1.0)
    nd = clip[:2] / clip[3]
    check("zzz: on screen, below the HUD strip", (np.abs(nd) < 0.9).all() and nd[1] < 0.8)
    src = inspect.getsource(VisualDashboard.render) + inspect.getsource(VisualDashboard.__init__)
    check("app: log panel + Zzz built and drawn", "LogContent()" in src and "ZzzRenderer(" in src and "self.zzz.draw(" in src and "self.log_panel.draw(" in src)
    check("app: Zzz after the wings, HUD and log last", src.index("self.wings.draw") < src.index("self.zzz.draw(") < src.index("self.hud.draw(") < src.index("self.log_panel.draw("))
    return len(_FAILS) == start


def _selftest_body_pose(check) -> bool:
    import inspect
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" body pose (11a)")
    lean = lambda st: pose_target(st)[0]
    check("pose targets: SILENT slumps most, then HOLD leans in, RESEARCH a little, IDLE none", lean("SILENT") > lean("HOLD") > lean("RESEARCH") > lean("IDLE") == 0.0)
    check("pose targets: SELL pushes back (negative lean), BUY leans slightly in", lean("SELL") < 0 < lean("BUY") < lean("HOLD"))
    check("pose targets: only SILENT sinks; unknown state = IDLE", pose_target("SILENT")[1] > 0 and all(pose_target(k)[1] == 0 for k in ("IDLE", "BUY", "SELL", "HOLD", "RESEARCH")) and pose_target("???") == (0.0, 0.0))
    for st in POSE_TARGETS:
        check(f"pose targets: every state ({st}) has a target", st in POSE_TARGETS)
    I = pose_matrix(0.0, 0.0).astype(np.float64)
    check("pose matrix: zero lean is the identity", np.allclose(I, np.eye(4), atol=1e-6))
    for dg in (-3.0, 11.0, 26.0):
        M = pose_matrix(dg, 0.1).astype(np.float64)
        R = M[:3, :3]
        check(f"pose matrix {dg:+.0f} deg: rigid (orthonormal, det +1)", np.allclose(R @ R.T, np.eye(3), atol=1e-5) and abs(np.linalg.det(R) - 1.0) < 1e-5)
    Mp = pose_matrix(26.0, 0.0).astype(np.float64)
    pv = np.array(LEAN_PIVOT)
    check("pose matrix: the hip pivot stays put when there is no sink", np.allclose((Mp @ np.append(pv, 1.0))[:3], pv, atol=1e-6))
    head = np.array(HEAD_C)
    head2 = (Mp @ np.append(head, 1.0))[:3]
    check("pose matrix: leaning forward moves the head toward +x (the desk) and lowers it", head2[0] > head[0] + 0.4 and head2[1] < head[1] - 0.3)
    check("pose matrix: the leaned head stays above the desk-top (y=0 world) and seat", FLY_POS[1] + (pose_matrix(26.0, 0.12).astype(np.float64) @ np.append(head, 1.0))[1] > 0.3)
    check("pose matrix: the slumped head stays clear of the monitors (world x < 0)", FLY_POS[0] + head2[0] < -1.0)
    bp = BodyPose()
    vm = NS(state="HOLD")
    bp.update(vm, 0.0)
    prev, mono, t = 0.0, True, 0.0
    for k in range(1, 400):
        t = k / 60.0
        bp.update(vm, t)
        mono &= bp.lean >= prev - 1e-12 and bp.lean <= 11.0 + 1e-9
        prev = bp.lean
    check("BodyPose: eases toward the HOLD lean, monotonic, never overshoots", mono and abs(bp.lean - 11.0) < 0.05)
    a, b = BodyPose(), BodyPose()
    for k in range(0, 61):
        a.update(NS(state="HOLD"), k / 60.0)
    for k in range(0, 31):
        b.update(NS(state="HOLD"), k / 30.0)
    check("BodyPose: frame-rate independent (60 fps vs 30 fps agree within 0.1 deg)", abs(a.lean - b.lean) < 0.1)
    c, d = BodyPose(), BodyPose()
    c.update(NS(state="HOLD"), 0.0); d.update(NS(state="SILENT"), 0.0)
    for k in range(1, 61):
        c.update(NS(state="HOLD"), k / 60.0); d.update(NS(state="SILENT"), k / 60.0)
    check("BodyPose: the slump is slower than the lean-in (after 1 s HOLD is >90% there, SILENT is not)", c.lean / 11.0 > 0.90 and d.lean / 26.0 < 0.85)
    e = BodyPose()
    e.update(NS(state="SILENT"), 0.0)
    for k in range(1, 300):
        e.update(NS(state="SILENT"), k / 60.0)
    before = e.lean
    e.update(NS(state="BUY"), 299 / 60.0 + 1 / 60.0)
    check("BodyPose: a state change moves smoothly (no jump)", 0.0 < before - e.lean < 3.0)
    g = BodyPose()
    g.update(NS(state="HOLD"), 0.0)
    g.update(NS(state="HOLD"), 100.0)
    check("BodyPose: a huge time gap (pause) cannot overshoot or blow up", 0.0 <= g.lean <= 11.0 and np.isfinite(g.lean))
    z = BodyPose(); z.update(NS(state="HOLD"), 1.0); l0 = z.lean; z.update(NS(state="HOLD"), 1.0)
    check("BodyPose: zero dt changes nothing; junk vm is IDLE", z.lean == l0 and np.isfinite(BodyPose().update(NS(), 0.0)).all())
    # arm still lands on the lever
    vmb = NS(reach=1.0, lever_buy=1.0, lever_sell=0.0)
    for dg in (0.0, 4.0, 11.0):
        P = pose_matrix(dg, 0.0).astype(np.float64)
        tgt_local = arm_target_in_pose(vmb, P)
        hand_world = FLY_POS + (P @ np.append(tgt_local, 1.0))[:3]
        grip_world = lever_tip_world(lever_angle(vmb))
        check(f"arm: hand stays on the lever knob under a {dg:g} deg lean ({np.linalg.norm(hand_world - grip_world):.4f} off)", np.linalg.norm(hand_world - grip_world) < 1e-4)
    check("arm: no pose leaves the target unchanged", np.allclose(arm_target_in_pose(vmb, None), arm_target(vmb)))
    # eyes
    open_t, open_e = eye_style(NS(state="IDLE", eyes_closed=False, flash=0.0))
    cl_t, cl_e = eye_style(NS(state="IDLE", eyes_closed=True, flash=0.0))
    check("eyes: closed (blink) is dark and dim", cl_e < 0.1 and cl_t == COL_EYE_CLOSED and open_e >= 0.3)
    check("eyes: SILENT keeps them closed even if the feed forgot", eye_style(NS(state="SILENT", eyes_closed=False, flash=1.0))[1] < 0.1)
    check("eyes: BUY glows green, SELL red, and the flash adds glow", eye_style(NS(state="BUY", flash=0.5))[0] == COL_GREEN and eye_style(NS(state="SELL", flash=0.5))[0] == COL_RED and eye_style(NS(state="BUY", flash=1.0))[1] > eye_style(NS(state="BUY", flash=0.0))[1])
    check("eyes: no vm gives a sane default; flash out of range is clamped", eye_style(None)[1] == 0.30 and eye_style(NS(state="BUY", flash=9.0))[1] <= 1.30 + 1e-9)
    # zzz follows the head
    bz = BodyPose(); bz.lean, bz.drop = 26.0, 0.12
    o0, o1 = BodyPose().zzz_origin(), bz.zzz_origin()
    check("zzz origin: at rest it is the old fixed spot", np.allclose(o0, ZZZ_ORIGIN, atol=1e-6))
    check("zzz origin: follows the slumped head (forward and lower), still above it", o1[0] > o0[0] + 0.3 and o1[1] < o0[1] and o1[1] > bz.head_world()[1] + 0.4)
    src = inspect.getsource(VisualDashboard.render) + inspect.getsource(VisualDashboard.__init__)
    check("app: one eased pose per frame feeds body, legs, wings and the Zzz", "BodyPose()" in src and "self.body_pose.update(" in src and "self.fly.draw(self.scene, t, vm, pose)" in src and "self.legs.draw(self.scene, t, vm, pose)" in src and "pose=pose" in src and "self.body_pose.zzz_origin()" in src)
    check("app: the pose is computed before the fly is drawn", src.index("self.body_pose.update(") < src.index("self.fly.draw("))
    return len(_FAILS) == start


def _selftest_glow_wings(check) -> bool:
    import inspect
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" wings, bob and glow pool (11b)")
    check("bob: no feed value gives the gentle default sine", abs(body_bob(None, 0.0)) < 1e-9 and abs(body_bob(NS(bob=0.0), math.pi / 4)) <= 0.012 + 1e-9)
    check("bob: the feed's value is used, clamped to +/-0.05", body_bob(NS(bob=0.015), 0.0) == 0.015 and body_bob(NS(bob=9.0), 0.0) == BOB_MAX and body_bob(NS(bob=-9.0), 0.0) == -BOB_MAX)
    check("bob: junk and NaN fall back to the default", abs(body_bob(NS(bob="x"), 0.3)) <= 0.012 + 1e-9 and abs(body_bob(NS(bob=float("nan")), 0.3)) <= 0.012 + 1e-9)
    check("wings: SILENT holds still", all(flap_angle(NS(state="SILENT", wing=1.0, flash=1.0), t, i) == 0.0 for t in (0.0, 0.3, 1.1) for i in (0, 1)))
    check("wings: the two wings are mirrored (opposite sign)", all(abs(flap_angle(NS(state="IDLE", wing=0.7, flash=0.0), 0.2, 0) + flap_angle(NS(state="IDLE", wing=0.7, flash=0.0), 0.2, 1)) < 1e-12 for _ in (0,)))
    big = abs(flap_angle(NS(state="BUY", wing=1.0, flash=1.0), 0.0, 0))
    check("wings: a flash flicks them harder than idle, RESEARCH a little busier", big > 0.3 and abs(flap_angle(NS(state="RESEARCH", wing=1.0, flash=0.0), 0, 0)) > abs(flap_angle(NS(state="IDLE", wing=1.0, flash=0.0), 0, 0)))
    check("wings: amplitude stays bounded for wild wing values", all(abs(flap_angle(NS(state="BUY", wing=w, flash=9.0), 0, 0)) <= 0.40 + 1e-9 for w in (-50.0, 50.0, float("inf"), "x")))
    check("wings: with no wing value from the feed they still flutter on their own", flap_angle(NS(state="IDLE", wing=0.0, flash=0.0), 0.1, 0) != 0.0 and flap_angle(None, 0.1, 0) != 0.0)
    w = build_wings()
    tips = []
    for i, B in enumerate(w.bases):
        tip0 = (B @ np.array([WING_LEN, 0, 0, 1.0]))[:3]
        tip1 = (wing_droop_matrix(w.roots[i], WING_DROOP_RAD) @ B @ np.array([WING_LEN, 0, 0, 1.0]))[:3]
        tips.append((tip0, tip1))
    check("wings: the silent droop lowers both wing tips", all(t1[1] < t0[1] - 0.4 for t0, t1 in tips))
    check("wings: the droop is mirror-symmetric and keeps the roots fixed", np.isclose(tips[0][1][1], tips[1][1][1]) and np.isclose(tips[0][1][2], -tips[1][1][2]) and
          all(np.allclose((wing_droop_matrix(w.roots[i], 0.5) @ np.append(w.roots[i], 1.0))[:3], w.roots[i], atol=1e-6) for i in (0, 1)))
    check("wings: zero droop is the identity", np.allclose(wing_droop_matrix(w.roots[0], 0.0), np.eye(4), atol=1e-6))
    bp = BodyPose()
    bp.update(NS(state="SILENT"), 0.0)
    for k in range(1, 400):
        bp.update(NS(state="SILENT"), k / 60.0)
    check("BodyPose: the wings droop when silent and lift again after", abs(bp.droop - WING_DROOP_RAD) < 0.01)
    for k in range(400, 800):
        bp.update(NS(state="IDLE"), k / 60.0)
    check("BodyPose: ... and recover when awake", abs(bp.droop) < 0.01)
    gb, ab = glow_pool(NS(state="BUY", flash=1.0))
    gs, as_ = glow_pool(NS(state="SELL", flash=0.5))
    check("glow pool: green on BUY, red on SELL, proportional to the flash", gb == COL_GREEN and gs == COL_RED and np.isclose(ab, GLOW_POOL_ALPHA) and np.isclose(as_, GLOW_POOL_ALPHA / 2))
    check("glow pool: none for IDLE/HOLD/SILENT/RESEARCH, no vm, or no flash", all(glow_pool(NS(state=st, flash=1.0))[1] == 0.0 for st in ("IDLE", "HOLD", "SILENT", "RESEARCH")) and glow_pool(None)[1] == 0.0 and glow_pool(NS(state="BUY", flash=0.0))[1] == 0.0)
    check("glow pool: alpha is clamped for wild flash values", glow_pool(NS(state="BUY", flash=7.0))[1] <= GLOW_POOL_ALPHA + 1e-12 and glow_pool(NS(state="BUY", flash=-3.0))[1] == 0.0)
    g = glow_pool_geo()
    d = build_desk()
    top = d.boxes[1]
    check("glow pool: whole triangles, finite", g.shape[1] == 9 and len(g) == 3 * 36 and np.isfinite(g).all())
    check("glow pool: lies on the desk top (just above it, within its footprint)", g[:, 1].min() > 0.0 and g[:, 1].max() < 0.02 and g[:, 0].min() >= top[0] - top[3] / 2 and g[:, 0].max() <= top[0] + top[3] / 2 and np.abs(g[:, 2]).max() <= top[5] / 2)
    check("glow pool: in front of the main monitor (between the fly and the screen)", g[:, 0].max() < MON_POS[0] - 0.05 and g[:, 0].min() > FLY_POS[0])
    src = inspect.getsource(VisualDashboard.render) + inspect.getsource(VisualDashboard.__init__)
    check("app: glow pool built, drawn after the desk and before the props", "GlowPoolRenderer(" in src and src.index("self.desk.draw(") < src.index("self.glow_pool.draw(") < src.index("self.props.draw("))
    check("app: wings get the eased droop", "droop=self.body_pose.droop" in src)
    return len(_FAILS) == start


def _selftest_trade_date(check) -> bool:
    import datetime as _dt
    import inspect
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" trading date on the monitors (dd/mm/yyyy)")
    good = {"2025-11-03": "03/11/2025", "2025-11-03T09:30:00": "03/11/2025", "2025/1/5": "05/01/2025", "20251103": "03/11/2025", 20251103: "03/11/2025",
            _dt.date(2019, 12, 31): "31/12/2019", _dt.datetime(2020, 2, 29, 16, 0): "29/02/2020"}
    check("format_trade_date: ISO, slash, compact, int, date and datetime inputs", all(format_trade_date(k) == v for k, v in good.items()))
    check("format_trade_date: always dd/mm/yyyy with 2+2+4 digits", all(re.fullmatch(r"\d\d/\d\d/\d{4}", format_trade_date(k)) for k in good))
    bad = [None, "", "junk", "03/11/2025", "11/03/2025", "2025-13-01", "2025-02-30", "2023-02-29", 0, 1.5, True, float("nan"), "0000-01-01", [], {}]
    check("format_trade_date: junk, impossible and ambiguous dd/mm vs mm/dd forms give '' (never a guess)", all(format_trade_date(b) == "" for b in bad))
    check("format_trade_date: leap day valid, day/month order not mixed up (2025-04-03 is 3 April)", format_trade_date("2024-02-29") == "29/02/2024" and format_trade_date("2025-04-03") == "03/04/2025")
    check("extract_trade_date: header keys in priority order", extract_trade_date({"date": "2021-05-06", "trade_date": "2021-05-07"}) == "07/05/2021" and extract_trade_date({"as_of": "2018-01-02"}) == "02/01/2018")
    check("extract_trade_date: falls back to a view attribute, then to ''", extract_trade_date({}, NS(date="2022-03-04")) == "04/03/2022" and extract_trade_date({}, NS()) == "" and extract_trade_date(None) == "" and extract_trade_date({"date": "x"}, None) == "")
    check("extract_trade_date: skips an unreadable key and uses the next", extract_trade_date({"trade_date": "junk", "date": "2020-07-08"}) == "08/07/2020")
    check("ViewModel: trade_date defaults to ''", ViewModel().trade_date == "")
    check("date_txt: blank shows --/--/----", date_txt(NS()) == NO_DATE and date_txt(NS(trade_date="")) == NO_DATE and date_txt(NS(trade_date="03/11/2025")) == "03/11/2025")
    check("sim: dates advance one week per round, dd/mm/yyyy", sim_trade_date(0) == "06/01/2020" and sim_trade_date(1) == "13/01/2020" and sim_trade_date(52) == "04/01/2021" and sim_trade_date("x") == "")
    sim = SimFeed(start_thread=False)
    v0 = sim.view()
    check("sim: the view carries the date and it changes with the round", v0.trade_date == "06/01/2020")
    sim._data["mode"] = "league"; sim._made = 10; sim.produce_once()
    check("sim: a new round moves the date forward", sim.view().trade_date == "13/01/2020")
    check("adapt_view: the date is taken from the league header", "extract_trade_date(v.header, v)" in inspect.getsource(adapt_view))
    d = "03/11/2025"
    on = lambda cv, text, x, y, sc, col: ((lambda ref: ((ref.px[:, :, 0] == 255) <= (cv.px == np.array(col, dtype=np.uint8)).all(2)).all())(
        (lambda r: (r.text(x, y, text, (255, 255, 255), sc), r)[1])(Canvas(SCREEN_W, SCREEN_H, (0, 0, 0)))))
    cm = Canvas(); render_chart(cm, CandleSeries(11), "META", d)
    check("main screen: the date is in the title bar at 2x, right-aligned", on(cm, d, SCREEN_W - 6 - text_width(d, 2), 4, 2, COL_SCR_VALUE))
    cm2 = Canvas(); render_chart(cm2, CandleSeries(11), "META", "")
    check("main screen: unknown date shows --/--/----", on(cm2, NO_DATE, SCREEN_W - 6 - text_width(NO_DATE, 2), 4, 2, COL_SCR_VALUE))
    check("main screen: the date, ticker and ILLUSTRATIVE tag do not overlap (ticker < tag < date)", 6 + text_width("MMMMMM", 2) < SCREEN_W - 6 - text_width(d, 2) - 8 - text_width("ILLUSTRATIVE DATA"))
    ct = Canvas(); render_tape(ct, NS(tape=[("AAPL", 1.0)], header=dict(top_id="fly_017", top_ic=0.02, ridge_ic=0.004), replay_label="SIMULATED FEED", trade_date=d))
    fx, fy, fw, fh = TAPE_FOOT_RECT
    check("tape screen: the date is in the footer at 2x", on(ct, d, fx + fw - 6 - text_width(d, 2), fy + 21, 2, COL_SCR_VALUE))
    check("tape screen: footer text does not run into the date (ridge line < date)", fx + 6 + text_width("RIDGE +0.004", 2) < fx + fw - 6 - text_width(d, 2) - 6 and 6 + text_width("MMMMMMMM IC +0.000", 2) + 6 < fw - 6 - text_width("SIMULATED FEED")) 
    ci = Canvas(); render_ic(ci, NS(ic_history=[(r, 0.01 * r, 0.0, 0.0) for r in range(1, 9)], replay_label="SIM FEED", trade_date=d))
    ix, iy, iw, ih = IC_FOOT_RECT
    check("ic screen: the date is in the footer at 2x", on(ci, d, ix + iw - 6 - text_width(d, 2), iy + 40, 2, COL_SCR_VALUE))
    check("ic screen: ROUND, source label and date fit side by side", ix + 6 + text_width("ROUND 999999", 2) < ix + 112 + 20 * 6 and ix + 112 + 20 * 6 < ix + iw - 6 - text_width(d, 2) + 1)
    sc = ScreenContent(); sc.update(NS(state="IDLE", ticker="AAPL", trade_date="01/01/2020", research_lines=[]), 1.0)
    check("main screen: a changed date forces a redraw", sc.update(NS(state="IDLE", ticker="AAPL", trade_date="08/01/2020", research_lines=[]), 1.0001) is True)
    tc, ic = TapeContent(), IcContent()
    a1 = tc.update(NS(tape=[], header={}, state="IDLE", trade_date="01/01/2020"), 0.0); a2 = tc.update(NS(tape=[], header={}, state="IDLE", trade_date="08/01/2020"), 0.1)
    b1 = ic.update(NS(ic_history=[], state="IDLE", trade_date="01/01/2020"), 0.0); b2 = ic.update(NS(ic_history=[], state="IDLE", trade_date="08/01/2020"), 0.1)
    check("tape and IC screens: a changed date forces a redraw", a1 and a2 and b1 and b2)
    return len(_FAILS) == start


def _selftest_bloom(check) -> bool:
    import inspect
    from unittest import mock
    start = len(_FAILS)
    print(" bloom (12a)")
    w = gauss_weights()
    check("gauss: normalised, 2R+1 taps, symmetric, strictly falling from the centre", np.isclose(w.sum(), 1.0) and len(w) == 2 * BLOOM_RADIUS + 1 and np.allclose(w, w[::-1]) and (np.diff(w[:BLOOM_RADIUS + 1]) > 0).all())
    check("gauss: radius below 1 is clamped, tails are tiny but positive", len(gauss_weights(0)) == 3 and w.min() > 0 and w.min() < 0.02)
    check("half_size: halves, floors, never below 1", half_size((1920, 1080)) == (960, 540) and half_size((1281, 721)) == (640, 360) and half_size((1, 1)) == (1, 1) and half_size((0, 0)) == (1, 1))
    chk = np.linspace(0, 1, 101)
    gray = np.stack([chk, chk, chk], -1)
    br = bloom_bright(gray)[:, 0]
    check("bright pass: nothing below the soft knee, never negative, never brighter than the input", (br[chk <= BLOOM_THRESHOLD - BLOOM_KNEE] == 0).all() and (br >= 0).all() and (br <= chk + 1e-12).all())
    check("bright pass: monotonic in brightness and continuous across the knee (no jumps)", (np.diff(br) >= -1e-12).all() and np.abs(np.diff(br)).max() < 0.05)
    col = np.array([[1.0, 0.4, 0.1]])
    bc = bloom_bright(col)[0]
    check("bright pass: keeps the hue (channel ratios unchanged)", np.allclose(bc / bc[0], col[0] / col[0, 0]))
    check("bright pass: black stays black, a dim scene produces no bloom at all", (bloom_bright(np.zeros((4, 4, 3))) == 0).all() and (bloom_bright(np.full((8, 8, 3), 0.2)) == 0).all())
    imp = np.zeros((40, 40, 3)); imp[20, 20] = 1.0
    bl = bloom_blur(imp)
    check("blur: conserves energy for a point away from the edges", np.isclose(bl.sum(), imp.sum()))
    check("blur: symmetric, centred, spreads exactly R pixels each way and no further", np.allclose(bl[20 - BLOOM_RADIUS:21 + BLOOM_RADIUS, 20 - BLOOM_RADIUS:21 + BLOOM_RADIUS], bl[20 - BLOOM_RADIUS:21 + BLOOM_RADIUS, 20 - BLOOM_RADIUS:21 + BLOOM_RADIUS][::-1, ::-1]) and bl[20, 20, 0] == bl.max() and (bl[20, 20 + BLOOM_RADIUS + 1:] == 0).all() and bl[20, 20 + BLOOM_RADIUS, 0] > 0)
    check("blur: a constant image stays constant (edges clamp)", np.allclose(bloom_blur(np.full((9, 9, 3), 0.3)), 0.3))
    scene = np.full((40, 40, 3), 0.2); scene[20, 20] = 1.0
    out = bloom_composite(scene, bloom_blur(bloom_bright(scene)))
    check("composite: a bright pixel lights up its neighbours but not the far field", out[20, 22, 0] > 0.2 + 1e-4 and np.isclose(out[0, 0, 0], 0.2) and out[20, 20 + BLOOM_RADIUS + 3, 0] == 0.2)
    check("composite: strength 0 is the scene, and the result never exceeds 1", np.allclose(bloom_composite(scene, bloom_blur(bloom_bright(scene)), 0.0), scene) and bloom_composite(np.ones((3, 3, 3)), np.ones((3, 3, 3)), 5.0).max() == 1.0)
    check("composite: a scene below the threshold is returned untouched (no haze)", np.allclose(bloom_composite(np.full((6, 6, 3), 0.2), bloom_blur(bloom_bright(np.full((6, 6, 3), 0.2)))), 0.2))
    wts = [float(x) for x in re.search(r"float\[\d+\]\((.*?)\)", BLOOM_BLUR_FRAG).group(1).split(",")]
    check("shader: blur weights are the gaussian, 13 taps, sum 1", len(wts) == 13 and np.allclose(wts, w, atol=1e-7) and abs(sum(wts) - 1.0) < 1e-6)
    check("shader: bright pass uses the same threshold and knee as the reference", f"{BLOOM_THRESHOLD:.6f}" in BLOOM_BRIGHT_FRAG and f"{BLOOM_KNEE:.6f}" in BLOOM_BRIGHT_FRAG and "%" not in BLOOM_BRIGHT_FRAG)
    check("shader: full-screen triangle, composite clamps to 1", "gl_VertexID" in BLOOM_VERT and "min(c, vec3(1.0))" in BLOOM_COMPOSITE_FRAG and all(f.lstrip().startswith("#version 330") for f in (BLOOM_VERT, BLOOM_BRIGHT_FRAG, BLOOM_BLUR_FRAG, BLOOM_COMPOSITE_FRAG)))
    fake_gl = mock.patch.dict(sys.modules, {"moderngl": mock.MagicMock()})      # the GL library is not needed to test the call sequence
    fake_gl.start()
    ctx = mock.MagicMock()
    for fn in ("framebuffer", "texture", "renderbuffer", "depth_renderbuffer", "program", "vertex_array"):
        getattr(ctx, fn).side_effect = lambda *a, **k: mock.MagicMock()                  # every GL object is its own mock
    order = []
    ctx.disable.side_effect = lambda *a: order.append("off")
    ctx.enable.side_effect = lambda *a: order.append("on")
    pipe = BloomPipeline(ctx, (1920, 1080))
    sizes = [c.args[0] for c in ctx.texture.call_args_list]
    check("pipeline: one full-size scene texture and two half-size blur textures", sizes == [(1920, 1080), (960, 540), (960, 540)])
    check("pipeline: the scene renders into a 4x multisampled target with depth", ctx.renderbuffer.call_args.kwargs.get("samples") == 4 and ctx.depth_renderbuffer.call_args.kwargs.get("samples") == 4)
    ctx.reset_mock()
    pipe.begin()
    check("pipeline.begin: binds the offscreen target at full size", pipe.msaa.use.called and ctx.viewport == (0, 0, 1920, 1080))
    pipe.present(0.4)
    check("pipeline.present: resolves MSAA into the scene texture first", ctx.copy_framebuffer.call_args.args == (pipe.resolve, pipe.msaa))
    check("pipeline.present: bright -> blur H -> blur V on the half-size targets, then the window", pipe.fbo_a.use.call_count == 2 and pipe.fbo_b.use.call_count == 1 and ctx.screen.use.called)
    check("pipeline.present: three full-screen triangles drawn (bright, blur, blur) + composite", sum(v.render.call_count for v in pipe.vaos) == 4)
    check("pipeline.present: the blur taps are spread BLOOM_SPREAD texels apart", pipe.p_blur["u_dir"].value == (0.0, BLOOM_SPREAD / 540) and BLOOM_SPREAD > 1.0)
    check("pipeline.present: composite uses the strength given, ends at full size on the window", pipe.p_comp["u_strength"].value == 0.4 and ctx.viewport == (0, 0, 1920, 1080))
    check("pipeline.present: depth test and blending are turned off for the passes and back on after", order == ["off", "on"])
    pipe.release()
    check("pipeline.release: never raises, even if one object fails", True)
    bad = mock.MagicMock(); bad.framebuffer.side_effect = RuntimeError("no fbo")
    with mock.patch("sys.stderr"):
        check("make_bloom: a GPU that cannot build it gives None (the app keeps running)", make_bloom(bad, (640, 360)) is None)
    check("make_bloom: working context gives a pipeline", isinstance(make_bloom(mock.MagicMock(), (640, 360)), BloomPipeline))
    fake_gl.stop()
    src = inspect.getsource(VisualDashboard.render)
    check("app: scene drawn into the bloom target before the first draw call", src.index("self.bloom.begin()") < src.index("self.ctx.clear(") < src.index("self.skyline.draw("))
    check("app: bloom resolved after the translucent 3D, before the HUD and log panel (those stay crisp)", src.index("self.zzz.draw(") < src.index("self.bloom.present(") < src.index("self.hud.draw(") < src.index("self.log_panel.draw("))
    check("app: a failing bloom pass is dropped and rendering carries on", "_drop_bloom" in src and "self.ctx.screen.use()" in inspect.getsource(VisualDashboard._drop_bloom))
    init = inspect.getsource(VisualDashboard.__init__) + inspect.getsource(VisualDashboard.handle_events)
    check("bloom_step: fixed steps, clamped to 0..max, direction only matters by sign", bloom_step(1.1, 1) == 1.35 and bloom_step(0.1, -1) == 0.0 and bloom_step(0.0, -1) == 0.0 and bloom_step(BLOOM_STRENGTH_MAX, 1) == BLOOM_STRENGTH_MAX and bloom_step(1.0, -7) == 0.75)
    check("app: +/- change the live strength, which is what present() receives", "bloom_step(self.bloom_strength, +1)" in init and "bloom_step(self.bloom_strength, -1)" in init and "self.bloom.present(self.bloom_strength)" in src)
    check("bloom defaults are visible: strength >= 1, threshold <= 0.55, wide spread", BLOOM_STRENGTH >= 1.0 and BLOOM_THRESHOLD <= 0.55 and BLOOM_SPREAD >= 2.0)
    check("app: --no-bloom switches it off and key B toggles it", "make_bloom(" in init and "K_b" in init and "bloom_on = not self.bloom_on" in init and "bloom: bool = True" in init)
    check("main: --no-bloom flag exists and is passed through", "--no-bloom" in inspect.getsource(parse_args if "parse_args" in globals() else main) or "--no-bloom" in open(__file__, encoding="utf-8").read())
    return len(_FAILS) == start


def _selftest_full_frame(check) -> bool:
    """End-to-end: the real VisualDashboard (every renderer, the bloom pipeline, the overlays) run for whole frames against a fake
    GL / pygame. Catches wiring mistakes (wrong arguments, missing attributes, wrong draw order) that unit tests cannot see.
    It cannot judge how anything looks: that needs a real GPU."""
    import dataclasses
    from unittest import mock
    start = len(_FAILS)
    print(" full frame, every state, fake GL (12b)")
    states = [("RESEARCH", {}), ("BUY", dict(flash=1.0, reach=1.0, lever_buy=1.0)), ("SELL", dict(flash=0.6, reach=1.0, lever_sell=1.0)),
              ("HOLD", {}), ("SILENT", dict(zzz=2.0, eyes_closed=True)), ("IDLE", {})]
    check("every state the feed can produce has a pose target and a text line", all(st in POSE_TARGETS for st, _ in states))

    def build(size, bloom=True):
        gl, pg = mock.MagicMock(), mock.MagicMock()
        pg.display.get_window_size.return_value = size
        pg.event.get.return_value = []
        feed = SimFeed(start_thread=False)
        with mock.patch.dict(sys.modules, {"moderngl": gl, "pygame": pg}):
            app = VisualDashboard(feed, fullscreen=False, size=size, bloom=bloom)
        return app, feed, gl, pg

    def run(app, feed, frames_per_state=12, vm_extra=None):
        base = feed.view()
        t, err = 0.0, None
        try:
            for st, kw in states:
                for _ in range(frames_per_state):
                    t += 1 / 60
                    vm = dataclasses.replace(base, state=st, ticker="NVDA", trade_date="03/11/2025", log=[((57, 255, 20), "[FLY BUY] Entered Long Position on NVDA - MBON Hz: +1.5")], **kw)
                    feed.view = lambda vm=vm: vm
                    app.update(t, 1 / 60)
                    app.render(t)
        except Exception as e:                                        # noqa: BLE001 - report any failure as a test result
            err = e
        return err, t

    app, feed, gl, pg = build((1280, 720))
    order = []
    for name, obj in (("bloom.begin", app.bloom), ("bloom.present", app.bloom)):
        meth = name.split(".")[1]
        orig = getattr(obj, meth)
        setattr(obj, meth, (lambda o=orig, n=name: (lambda *a, **k: (order.append(n), o(*a, **k))[1]))())
    for name in ("skyline", "desk", "glow_pool", "monitor", "screen", "side_screen", "ic_screen", "fly", "legs", "wings", "zzz", "hud", "log_panel"):
        obj = getattr(app, name)
        orig = obj.draw
        setattr(obj, "draw", (lambda o=orig, n=name: (lambda *a, **k: (order.append(n), o(*a, **k))[1]))())
    orig_clear = app.ctx.clear
    app.ctx.clear = lambda *a, **k: (order.append("clear"), orig_clear(*a, **k))[1]
    err, _ = run(app, feed)
    check(f"full frame: 6 states x 12 frames render without an exception ({type(err).__name__ if err else 'ok'}: {err})" if err else "full frame: 6 states x 12 frames render without an exception", err is None)
    frames = order.count("clear")
    check("full frame: exactly one clear and one bloom begin/present per frame", frames == 72 and order.count("bloom.begin") == 72 and order.count("bloom.present") == 72)
    clears = [i for i, n in enumerate(order) if n == "clear"]
    one = order[:clears[1] - 1] if len(clears) > 1 else order          # frame 1 only (the next frame starts with bloom.begin)
    pos = {n: one.index(n) for n in ("bloom.begin", "clear", "skyline", "desk", "glow_pool", "screen", "side_screen", "ic_screen", "fly", "legs", "wings", "zzz", "bloom.present", "hud", "log_panel") if n in one}
    check("full frame: draw order is bloom target, clear, world, wings, Zzz, bloom resolve, then HUD and log on top",
          len(pos) == 15 and pos["bloom.begin"] < pos["clear"] < pos["skyline"] < pos["desk"] < pos["glow_pool"] < pos["fly"] < pos["wings"] < pos["zzz"] < pos["bloom.present"] < pos["hud"] < pos["log_panel"])
    check("full frame: all three monitor screens are drawn each frame", all(n in one for n in ("screen", "side_screen", "ic_screen")))
    check("full frame: the HUD and log are never inside the bloom (drawn after the resolve)", pos["bloom.present"] < pos["hud"] and pos["bloom.present"] < pos["log_panel"])
    check("full frame: the pose has settled toward the last state (IDLE) after the SILENT slump started", abs(app.body_pose.lean) < 26.0 and np.isfinite(app.body_pose.lean))
    for name, size in (("21:9", (2560, 1080)), ("4:3", (1024, 768)), ("5:4", (1280, 1024)), ("tiny", (320, 200)), ("portrait", (720, 1280))):
        a2, f2, _, _ = build(size)
        e2, _ = run(a2, f2, frames_per_state=2)
        check(f"full frame at {name} ({size[0]}x{size[1]}): renders without an exception" + (f" [{type(e2).__name__}: {e2}]" if e2 else ""), e2 is None)
    a3, f3, _, _ = build((1280, 720), bloom=False)
    e3, _ = run(a3, f3, frames_per_state=3)
    check("full frame: with bloom off (--no-bloom) the plain path renders", e3 is None and a3.bloom is None)
    a4, f4, _, _ = build((1280, 720))
    a4.bloom.present = mock.MagicMock(side_effect=RuntimeError("driver said no"))
    with mock.patch("sys.stderr"):
        e4, _ = run(a4, f4, frames_per_state=3)
    check("full frame: a bloom failure mid-frame is survived and bloom is switched off", e4 is None and a4.bloom_on is False)
    a5, f5, _, _ = build((1280, 720))
    a5.bloom.begin = mock.MagicMock(side_effect=RuntimeError("no fbo"))
    with mock.patch("sys.stderr"):
        e5, _ = run(a5, f5, frames_per_state=3)
    check("full frame: a bloom failure at the start of a frame is survived too", e5 is None and a5.bloom_on is False)
    a6, f6, gl6, pg6 = build((1280, 720))
    pg6.event.get.return_value = [mock.MagicMock(type=pg6.KEYDOWN, key=pg6.K_b)]
    a6.handle_events()
    check("keys: B toggles the bloom off", a6.bloom_on is False)
    a6.bloom_on = True
    pg6.event.get.return_value = [mock.MagicMock(type=pg6.KEYDOWN, key=pg6.K_EQUALS)]
    a6.handle_events()
    check("keys: + raises the bloom strength by one step", abs(a6.bloom_strength - (BLOOM_STRENGTH + BLOOM_STRENGTH_STEP)) < 1e-9)
    pg6.event.get.return_value = [mock.MagicMock(type=pg6.KEYDOWN, key=pg6.K_ESCAPE)]
    a6.handle_events()
    check("keys: ESC stops the main loop", a6.running is False)
    return len(_FAILS) == start


def _selftest_desk(check) -> bool:
    start = len(_FAILS)
    print(" desk (5a)")
    d = build_desk()
    bx = d.boxes
    nb = len(LEG_XZ) + 3                      # slab + legs + console + pad (lit)
    check(f"desk: {nb + 1} boxes ({nb} lit + 1 floor), finite", bx.shape == (nb + 1, 6) and len(d.solid) == nb * 36 and len(d.floor) == 36 and np.isfinite(d.solid).all())
    off = build_desk(neon=False)
    check("desk: neon=False has no striping and no grid", len(off.neon) == 0 and len(off.grid) == 0 and np.array_equal(off.solid, d.solid))
    check("desk: every box has outward winding", all(
        _outward_ok(np.concatenate([(lambda c: np.concatenate([c[:, :3] - bx[1 + i, :3], c[:, 3:]], 1))(d.solid[i * 36:(i + 1) * 36])]))
        for i in range(nb)))
    top = bx[1]
    check("desk: top surface is y = 0", abs(top[1] + top[4] / 2) < 1e-9)
    check(f"desk: {DESK_SIZE[0]} x {DESK_SIZE[2]} footprint (long desk, left edge stays at x = -2.5)", np.allclose(top[3:6:2], (DESK_SIZE[0], DESK_SIZE[2])) and np.isclose(top[0] - top[3] / 2, -2.5))
    legs = bx[2:2 + len(LEG_XZ)]
    check("desk: legs stand on the floor", np.allclose(legs[:, 1] - legs[:, 4] / 2, FLOOR_TOP_Y))
    check("desk: legs sit under the slab", (np.abs(legs[:, 0] - top[0]) < top[3] / 2).all() and (np.abs(legs[:, 2]) < top[5] / 2).all())
    con, pad = bx[2 + len(LEG_XZ)], bx[3 + len(LEG_XZ)]
    check("console: rests on the desk top, inside its footprint",
          abs(con[1] - con[4] / 2) < 1e-9 and abs(con[0] - top[0]) + con[3] / 2 < top[3] / 2 and con[0] - con[3] / 2 > top[0] - top[3] / 2 and abs(con[2]) + con[5] / 2 < top[5] / 2)
    check("pad: rests on the console", abs((pad[1] - pad[4] / 2) - (con[1] + con[4] / 2)) < 1e-9)
    check("desk colours: dark neutral charcoal", np.array(COL_DESK).max() < 0.2 and abs(COL_DESK[0] - COL_DESK[1]) < 0.02)
    check("floor: top is y = -2.3", abs(bx[0][1] + bx[0][4] / 2 - FLOOR_TOP_Y) < 1e-9)
    sky = build_skyline(7, x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG)
    check("floor ends before the nearest skyline row", bx[0][0] + bx[0][3] / 2 < (sky.boxes[:, 0] - sky.boxes[:, 3] / 2).min() - 0.1)

    pv = Camera(16 / 9).proj_view.astype(np.float64)
    corners = np.array([[top[0] + a * top[3] / 2, 0.0, b * top[5] / 2] for a in (-1, 1) for b in (-1, 1)])
    c = np.concatenate([corners, np.ones((4, 1))], 1) @ pv.T
    n = c[:, :3] / c[:, 3:4]
    check("desk top fully on screen", (np.abs(n[:, 0]) < 0.95).all() and (np.abs(n[:, 1]) < 0.95).all())
    check(f"desk sits in the lower part of the frame (top centre y {n[:, 1].mean():.2f}, near edge {n[:, 1].min():.2f})",
          -0.45 < n[:, 1].mean() < -0.2 and n[:, 1].min() < -0.5)
    check(f"desk is wide on screen ({n[:, 0].max() - n[:, 0].min():.2f} of the 2.0 width)", n[:, 0].max() - n[:, 0].min() > 0.6)
    return len(_FAILS) == start


def _selftest_desk_neon(check) -> bool:
    start = len(_FAILS)
    print(" desk neon + floor grid (5b)")
    d = build_desk()
    n = d.neon
    top = d.boxes[1]
    check("stripes: 24 thin boxes (12 desk edges + 12 console edges)", n.shape == (24 * 36, 9) and np.isfinite(n).all())
    check("stripes: neon cyan, nothing else", np.allclose(n[:, 6:9], COL_NEON_CYAN))
    s = n[:12 * 36, :3]
    half = NEON_EDGE / 2
    check("stripes: desk slab outline stands proud of every face by half the thickness",
          np.allclose(s.max(0), (top[0] + top[3] / 2 + half, top[1] + top[4] / 2 + half, top[5] / 2 + half), atol=1e-5)
          and np.allclose(s.min(0), (top[0] - top[3] / 2 - half, top[1] - top[4] / 2 - half, -top[5] / 2 - half), atol=1e-5))
    tubes = s.reshape(12, 36, 3)
    ext = tubes.max(1) - tubes.min(1)
    lengths = sorted(np.round(ext.max(1), 4).tolist())
    check("stripes: 4 tubes per axis, lengths = edge + thickness",
          np.allclose(lengths, sorted([DESK_SIZE[0] + NEON_EDGE] * 4 + [0.2 + NEON_EDGE] * 4 + [DESK_SIZE[2] + NEON_EDGE] * 4), atol=1e-4))
    check("stripes: the top-edge tubes reach above the desk surface (y > 0)", (tubes.max(1)[:, 1] > 0).sum() >= 4)
    con = n[12 * 36:, :3]
    c = d.boxes[2 + len(LEG_XZ)]
    check("stripes: console outline hugs the console deck",
          np.allclose(con.max(0), (c[0] + c[3] / 2 + NEON_EDGE_CONSOLE / 2, c[1] + c[4] / 2 + NEON_EDGE_CONSOLE / 2, c[5] / 2 + NEON_EDGE_CONSOLE / 2), atol=1e-5))

    pv = Camera(16 / 9).proj_view.astype(np.float64)

    def vis(pts):
        k = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ pv.T
        q = k[:, :3] / k[:, 3:4]
        return (k[:, 3] > 0.1) & (np.abs(q[:, 0]) < 1) & (np.abs(q[:, 1]) < 1)

    check("stripes: every desk-edge tube is on screen", vis(tubes.mean(1)).all())

    g = d.grid
    f = d.boxes[0]
    fx, fz = f[3] / 2, f[5] / 2
    check("grid: 27 + 39 line quads", g.shape == (66 * 6, 9) and np.isfinite(g).all())
    check("grid: lies just above the floor top", np.allclose(g[:, 1], FLOOR_TOP_Y + 0.005))
    check("grid: stays within the floor slab", (np.abs(g[:, 0] - f[0]) <= fx + 1e-6).all() and (np.abs(g[:, 2] - f[2]) <= fz + 1e-6).all())
    check("grid: normals +Y and winding faces up", np.allclose(g[:, 3:6], (0, 1, 0)) and
          (np.cross(g[:, :3].reshape(-1, 3, 3)[:, 1] - g[:, :3].reshape(-1, 3, 3)[:, 0],
                    g[:, :3].reshape(-1, 3, 3)[:, 2] - g[:, :3].reshape(-1, 3, 3)[:, 0])[:, 1] > 0).all())
    gx, gz = g[:, 0].reshape(-1, 6), g[:, 2].reshape(-1, 6)
    along_z = np.isclose(gz.max(1) - gz.min(1), 40.0)               # lines that run the full depth sit at a fixed x
    xl = np.sort(np.round(gx[along_z].mean(1), 3))
    check("grid: lines one unit apart", np.allclose(np.diff(xl), 1.0) and len(xl) == 27)
    check("grid: purple, dimmer than the desk neon", np.allclose(g[:, 6:9], COL_GRID) and COL_GRID[2] > COL_GRID[1] and max(COL_GRID) < 0.8)
    centres = g[:, :3].reshape(-1, 6, 3).mean(1)
    check(f"grid: {int(vis(centres).sum())} of 66 lines visible", vis(centres).sum() >= 20)
    sky = build_skyline(7, x_range=(-70.0, 70.0), yaw_deg=SKY_YAW_DEG)
    check("grid: stays clear of the skyline", g[:, 0].max() < (sky.boxes[:, 0] - sky.boxes[:, 3] / 2).min())
    return len(_FAILS) == start


def _selftest_bridge(check) -> bool:
    import tempfile
    import json as _json
    start = len(_FAILS)
    print(" bridge")

    r, f = envelopes("BUY", 0.5)
    check("BUY reach is 1 at 0.5 s, flash > 0", abs(r - 1.0) < 1e-6 and f > 0.3)
    r, f = envelopes("SELL", DURATION["SELL"])
    check("SELL reach and flash return to 0", r < 1e-6 and f < 1e-6)
    check("HOLD has no reach/flash", envelopes("HOLD", 0.5) == (0.0, 0.0))
    check("HOLD log rewritten to spec",
          spec_text("[FLY HOLD] Defending Position - Maintaining neutral holding on AAPL")
          == "[FLY HOLD] Holding AAPL")
    check("old BUY/SELL wording rewritten to what a long-only account does",
          spec_text("[FLY BUY] Entered Long Position on X - MBON Hz: +1.0") == "[FLY BUY] Bought X - MBON Hz: +1.0"
          and spec_text("[FLY SELL] Covered Short Position on X - MBON Hz: -1.0") == "[FLY SELL] Sold X from holdings - MBON Hz: -1.0"
          and spec_text("[FLY BUY] Bought X - MBON Hz: +1.0") == "[FLY BUY] Bought X - MBON Hz: +1.0")
    check("format_log wording",
          format_log("SELL", "ZTS", -2.04)[1] == "[FLY SELL] Sold ZTS from holdings - MBON Hz: -2.0")

    # deterministic simulator: drive producer + renderer by hand, no threads
    s = SimFeed(seed=3, start_thread=False)
    seen, texts = set(), set()
    for i in range(6000):
        if i % 5 == 0:
            s.produce_once()
        s.update(0.1)
        seen.add(s.state)
        for _, tx in s.log:
            texts.add(tx.split("]")[0] + "]")
    check("sim visits every state", {"RESEARCH", "BUY", "SELL", "HOLD", "SILENT"} <= seen)
    check("sim logs every action", {"[FLY BUY]", "[FLY SELL]", "[FLY HOLD]", "[FLY SILENT]"} <= texts)
    check("sim round advances", s.view().header["round"] >= 1)
    check("sim universe is 499", s.view().header["universe"] == 499)

    # real producer thread keeps running and shuts down cleanly
    s2 = SimFeed(seed=1, poll=0.05)
    time.sleep(0.3)
    s2.update(0.1)
    check("sim thread alive", s2._thread.is_alive())
    s2.close()
    check("sim thread stops", not s2._thread.is_alive())

    # your real dashboard package, if importable
    try:
        import dashboard.app  # noqa: F401
        from dashboard.state import Decision
    except Exception as e:
        print(f"  [skip] dashboard package not importable ({e})")
    else:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "agents" / "f1").mkdir(parents=True)
            (root / "rounds").mkdir()
            (root / "state.json").write_text(_json.dumps({"round": 3}))
            (root / "agents" / "f1" / "state.json").write_text(_json.dumps({
                "id": "f1", "status": "alive", "gen": 1, "last_round_done": 3,
                "history": [{"ic": 0.03}, {"ic": 0.02}, {"ic": 0.04}]}))
            (root / "rounds" / "round_03.json").write_text(_json.dumps(
                {"controls": {"ridge": {"ic": 0.01}, "untrained fly": {"ic": 0.0}}}))
            b = LeagueBridge(root, poll=0.1)
            time.sleep(0.5)
            b.update(0.05)
            h = b.view().header
            check("real bridge: round", h["round"] == 3)
            check("real bridge: top fly", h["top_id"] == "f1")
            check("real bridge: rolling avg IC", abs(h["top_ic"] - 0.03) < 1e-9)
            b.app.animator.push(Decision("2025-01-02", "AAPL", "BUY", 2.4, "f1", 3))
            for _ in range(18):                      # 0.9 s: past the 0.55 s log point of the newer animator
                b.update(0.05)
            vm = b.view()
            check("real bridge: BUY state", vm.state == "BUY" and vm.ticker == "AAPL")
            check("real bridge: BUY log line", any(t.startswith("[FLY BUY] Bought AAPL") for _, t in vm.log))
            check("real bridge: BUY reach/flash present", vm.reach > 0 and vm.flash > 0)
            check(f"real bridge: animator native={b.native} -> lever_buy matches",
                  (abs(vm.lever_buy - b.app.animator.pose().lever_buy) < 0.2) if b.native else (vm.lever_buy == vm.reach))
            b.close()
    return len(_FAILS) == start


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="STONKFLY 3D visual dashboard")
    ap.add_argument("--league-root", default=os.environ.get("STONKFLY_LEAGUE", "league_data_fundamentals"),
                    help="folder holding state.json, agents/, rounds/")
    ap.add_argument("--panel", help="panel parquet/pkl (gives the active-universe count and decision replay)")
    ap.add_argument("--sec-root", help="SEC cache root (its raw/ folder drives RESEARCH mode)")
    ap.add_argument("--universe-dir", help="folder with constituents.csv (cik -> ticker)")
    ap.add_argument("--agent", help="pin one fly id instead of following the top fly")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between league polls")
    ap.add_argument("--sim", action="store_true", help="force the synthetic feed")
    ap.add_argument("--nice", type=int, default=5, help="raise this process's niceness (POSIX) so training keeps priority")
    ap.add_argument("--windowed", action="store_true", help="open a normal window instead of fullscreen")
    ap.add_argument("--size", default="1600x900", help="window size for --windowed, WxH")
    ap.add_argument("--no-bloom", action="store_true", help="skip the glow post-process (older GPUs)")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        sys.exit(selftest())
    if args.nice:
        try:
            os.nice(args.nice)
        except (AttributeError, OSError):
            pass                                  # Windows / not permitted: harmless
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        w, h = WINDOW_SIZE
    VisualDashboard(make_feed(args), fullscreen=not args.windowed, size=(w, h), bloom=not args.no_bloom).run()


if __name__ == "__main__":
    main()
