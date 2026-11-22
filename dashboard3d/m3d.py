"""Tiny 3D toolkit: 4x4 matrices, low-poly meshes and a triangle batch. Pure numpy, no GL: the same geometry is drawn by the
ModernGL renderer (GPU) and by the software renderer (tests, screenshots, fallback). Right-handed, y up."""
from __future__ import annotations

import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)


def trans(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0])


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def apply(M, pts):
    p = np.asarray(pts, np.float64)
    q = p @ M[:3, :3].T + M[:3, 3]
    return q


# ------------------------------------------------------------------ meshes: (T, 3, 3) float arrays of triangles
def box_mesh(sx=1.0, sy=1.0, sz=1.0):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], np.float64)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (3, 6, 2), (3, 7, 6), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return v[np.array(f)]


def prism_mesh(radius=0.5, height=1.0, sides=6):
    """Upright n-gon prism (low-poly cylinder), centred, axis = y."""
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    top = np.stack([radius * np.cos(a), np.full(sides, height / 2), radius * np.sin(a)], 1)
    bot = top * [1, -1, 1]
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris += [[top[i], top[j], bot[j]], [top[i], bot[j], bot[i]],
                 [(0, height / 2, 0), top[j], top[i]], [(0, -height / 2, 0), bot[i], bot[j]]]
    return np.array(tris, np.float64)


def quad_mesh(p0, p1, p2, p3):
    p = [np.asarray(q, np.float64) for q in (p0, p1, p2, p3)]
    return np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])


def segment_matrix(p0, p1, up=(0, 1, 0)):
    """Matrix placing a unit-length-x box (scaled by the caller to length 1) between p0 and p1."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    L = np.linalg.norm(d)
    ax = d / max(L, 1e-9)
    az = np.cross(ax, up)
    if np.linalg.norm(az) < 1e-6:
        az = np.cross(ax, (0, 0, 1))
    az /= np.linalg.norm(az)
    ay = np.cross(az, ax)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = ax, ay, az, (p0 + p1) / 2
    return m, L


class Batch:
    """Triangles with a colour each, an 'emissive' flag (drawn unlit: neon, screens) and an alpha."""

    def __init__(self):
        self.tris, self.col, self.emi, self.alpha = [], [], [], []

    def add(self, mesh, M=None, color=(200, 200, 200), emissive=False, alpha=1.0):
        t = np.asarray(mesh, np.float64)
        if M is not None:
            t = apply(M, t.reshape(-1, 3)).reshape(-1, 3, 3)
        n = len(t)
        self.tris.append(t)
        self.col.append(np.tile(np.asarray(color, np.float32) / 255.0, (n, 1)))
        self.emi.append(np.full(n, emissive))
        self.alpha.append(np.full(n, alpha, np.float32))

    def box(self, center, size, color, M=None, emissive=False, alpha=1.0):
        m = trans(*center) @ scale(1, 1, 1)
        if M is not None:
            m = M @ m
        self.add(box_mesh(*size), m, color, emissive, alpha)

    def limb(self, p0, p1, thick, color, emissive=False):
        m, L = segment_matrix(p0, p1)
        self.add(box_mesh(L, thick, thick), m, color, emissive)

    def arrays(self):
        if not self.tris:
            return np.zeros((0, 3, 3)), np.zeros((0, 3), np.float32), np.zeros(0, bool), np.zeros(0, np.float32)
        return (np.concatenate(self.tris), np.concatenate(self.col), np.concatenate(self.emi), np.concatenate(self.alpha))


def two_bone_ik(shoulder, target, l1, l2, bend=(0, 1, 0)):
    """Elbow position for a two-segment arm reaching `target` (bends toward `bend`)."""
    s, t = np.asarray(shoulder, np.float64), np.asarray(target, np.float64)
    d = t - s
    dist = float(np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
    dirv = d / max(np.linalg.norm(d), 1e-9)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole = np.asarray(bend, np.float64) - dirv * float(np.dot(bend, dirv))
    pole /= max(np.linalg.norm(pole), 1e-9)
    elbow = s + dirv * a + pole * h
    hand = s + dirv * dist
    return elbow, hand
