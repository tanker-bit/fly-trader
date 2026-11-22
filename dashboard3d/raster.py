"""Z-buffer software rasterizer (numpy only): per-pixel depth test, multi-light flat shading, distance fog, supersampling.

Chunk 1 of the 3D dashboard upgrade. It replaces the painter's-algorithm drawing in render_soft.py: triangles can now be
drawn in any order and still occlude each other correctly, the monitor texture is mapped into the same depth buffer
(perspective-correct), and translucent triangles (wings, glow, Zzz) are blended last against the opaque depth.

Public API (drop-in compatible with render_soft.render):
    render(frame, view, size=(1280, 720), ssaa=2, lighting=None) -> PIL.Image
Building blocks for later chunks (post-fx, GL parity):
    Lighting, DirLight, PointLight, default_lighting(), Rasterizer, shade_faces()

Conventions: world is right-handed, y up (see m3d.py). Colours are float 0..1 inside, uint8 at the edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from dashboard.canvas_pil import PilCanvas
from dashboard3d import hud, screen
from dashboard3d.m3d import look_at, perspective
from dashboard3d.scene3d import EYE, FOV, TARGET

NEAR = 0.5          # near clip distance (camera units)
FAR = 80.0


# ------------------------------------------------------------------ lighting
@dataclass
class DirLight:
    direction: tuple          # vector pointing TOWARD the light
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0


@dataclass
class PointLight:
    pos: tuple
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    radius: float = 4.0       # light reaches zero at this distance (smooth quadratic falloff)


@dataclass
class Lighting:
    sky_ambient: tuple = (0.16, 0.14, 0.30)       # ambient for faces pointing up
    ground_ambient: tuple = (0.07, 0.05, 0.12)    # ambient for faces pointing down
    dirs: list = field(default_factory=list)
    points: list = field(default_factory=list)
    fog_color: tuple = (0.05, 0.03, 0.12)
    fog_density: float = 0.040
    fog_start: float = 7.0
    sky_top: tuple = (8 / 255, 4 / 255, 26 / 255)         # background gradient, same as the old renderer
    sky_bottom: tuple = (46 / 255, 12 / 255, 66 / 255)


def default_lighting() -> Lighting:
    """Cold key light from the camera side, magenta rim from behind the monitor, cyan/magenta neon point lights on the desk."""
    return Lighting(
        dirs=[DirLight((-0.35, 0.85, 0.45), (0.55, 0.65, 1.00), 0.75),
              DirLight((0.60, 0.30, -0.55), (1.00, 0.25, 0.80), 0.40)],
        points=[PointLight((-2.5, 0.5, 0.0), (0.15, 0.90, 1.00), 1.3, 4.5),      # cyan strip at the desk's left edge
                PointLight((0.4, 0.5, 1.6), (1.00, 0.15, 0.80), 1.1, 4.5),       # magenta strip on the desk's front edge
                PointLight((1.2, 1.5, 0.1), (0.35, 0.55, 1.00), 1.0, 4.0)],      # monitor glow
    )


def shade_faces(tris, col, emi, eye, lighting: Lighting):
    """Flat shading: one colour per triangle. Normals are flipped to face the viewer (two-sided), emissive faces are unlit.
    Returns (rgb float (N,3), centroids (N,3))."""
    tris = np.asarray(tris, np.float64)
    cen = tris.mean(1)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    to_eye = np.asarray(eye, np.float64) - cen
    n = np.where(((n * to_eye).sum(1) < 0)[:, None], -n, n)
    up = (n[:, 1:2] * 0.5 + 0.5)
    light = np.asarray(lighting.ground_ambient) * (1 - up) + np.asarray(lighting.sky_ambient) * up
    for d in lighting.dirs:
        L = np.asarray(d.direction, np.float64)
        L = L / np.linalg.norm(L)
        light = light + np.maximum(n @ L, 0.0)[:, None] * (np.asarray(d.color) * d.intensity)
    for p in lighting.points:
        v = np.asarray(p.pos, np.float64) - cen
        dist = np.maximum(np.linalg.norm(v, axis=1), 1e-6)
        lam = np.maximum((n * v).sum(1) / dist, 0.0)
        att = np.clip(1.0 - dist / p.radius, 0.0, 1.0) ** 2
        light = light + (lam * att)[:, None] * (np.asarray(p.color) * p.intensity)
    rgb = np.where(np.asarray(emi, bool)[:, None], col, col * light)
    return np.clip(rgb, 0.0, 1.5).astype(np.float32), cen


# ------------------------------------------------------------------ camera
def _clip_near(cam, attrs):
    """Clip one triangle (3,3 camera-space verts, depth = -z) against the near plane. Returns a list of (verts, attrs) triangles."""
    d = -cam[:, 2]
    inside = d >= NEAR
    if inside.all():
        return [(cam, attrs)]
    if not inside.any():
        return []
    out_v, out_a = [], []
    for i in range(3):
        j = (i + 1) % 3
        if inside[i]:
            out_v.append(cam[i]); out_a.append(attrs[i])
        if inside[i] != inside[j]:
            t = (d[i] - NEAR) / (d[i] - d[j])
            out_v.append(cam[i] + t * (cam[j] - cam[i])); out_a.append(attrs[i] + t * (attrs[j] - attrs[i]))
    out_v, out_a = np.array(out_v), np.array(out_a)
    return [(out_v[[0, k, k + 1]], out_a[[0, k, k + 1]]) for k in range(1, len(out_v) - 1)]


class Rasterizer:
    """Colour + inverse-depth buffers at ssaa x the output size. Larger inverse depth = nearer."""

    def __init__(self, size=(1280, 720), ssaa=2, eye=EYE, target=TARGET, fov=FOV, lighting: Lighting | None = None):
        self.out_w, self.out_h = size
        self.ssaa = int(ssaa)
        self.W, self.H = self.out_w * self.ssaa, self.out_h * self.ssaa
        self.eye = np.asarray(eye, np.float64)
        self.V = look_at(eye, target)
        self.P = perspective(fov, self.out_w / self.out_h, NEAR, FAR)
        self.light = lighting or default_lighting()
        self.f = self.P[1, 1]
        self.aspect = self.out_w / self.out_h
        self.color = self._sky()
        self.iw = np.zeros((self.H, self.W), np.float32)     # 1 / view depth, 0 = nothing drawn
        self._late = []                                       # translucent triangles, blended in resolve()

    # --- helpers
    def _sky(self):
        t = (np.arange(self.H, dtype=np.float32) / self.H)[:, None, None]
        top, bot = np.asarray(self.light.sky_top, np.float32), np.asarray(self.light.sky_bottom, np.float32)
        return np.broadcast_to(top + (bot - top) * t, (self.H, self.W, 3)).copy()

    def _to_cam(self, pts):
        return pts @ self.V[:3, :3].T + self.V[:3, 3]

    def _to_screen(self, cam):
        d = -cam[..., 2]
        sx = (cam[..., 0] * self.f / self.aspect / d * 0.5 + 0.5) * self.W
        sy = (1 - (cam[..., 1] * self.f / d * 0.5 + 0.5)) * self.H
        return sx, sy, 1.0 / d

    def _prepare(self, tris, attrs=None):
        """World triangles (N,3,3) -> list of (screen verts (3,2), 1/depth (3,), attrs (3,k) | None, source index), near-clipped."""
        cam = self._to_cam(np.asarray(tris, np.float64).reshape(-1, 3)).reshape(-1, 3, 3)
        d = -cam[..., 2]
        out = []
        full = (d >= NEAR).all(1)
        sx, sy, iw = self._to_screen(np.where(full[:, None, None], cam, cam * 0 + [0, 0, -1.0]))
        for i in np.nonzero(full)[0]:
            out.append((np.stack([sx[i], sy[i]], 1), iw[i], None if attrs is None else attrs[i], i))
        for i in np.nonzero(~full & (d >= NEAR).any(1))[0]:        # partly behind the camera
            a = np.zeros((3, 1)) if attrs is None else attrs[i]
            for v, at in _clip_near(cam[i], a):
                x, y, w = self._to_screen(v)
                out.append((np.stack([x, y], 1), w, None if attrs is None else at, i))
        return out

    # --- coverage of one triangle: returns (slice y, slice x, barycentrics l0,l1,l2, mask) or None
    def _cover(self, s, iw, bias=0.0):
        x0 = max(int(np.floor(s[:, 0].min())), 0); x1 = min(int(np.ceil(s[:, 0].max())) + 1, self.W)
        y0 = max(int(np.floor(s[:, 1].min())), 0); y1 = min(int(np.ceil(s[:, 1].max())) + 1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        (ax, ay), (bx, by), (cx, cy) = s
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            return None
        px = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        py = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        l0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        l1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        l2 = 1.0 - l0 - l1
        z = (l0 * iw[0] + l1 * iw[1] + l2 * iw[2]) * (1.0 + bias)
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        return slice(y0, y1), slice(x0, x1), l0, l1, l2, z, inside

    # --- drawing
    def draw_opaque(self, tris, rgb):
        """rgb: (N,3) float 0..1 colour per triangle (already lit)."""
        for s, iw, _, i in self._prepare(tris):
            c = self._cover(s, iw)
            if c is None:
                continue
            ys, xs, _, _, _, z, inside = c
            zb = self.iw[ys, xs]
            m = inside & (z > zb)
            if m.any():
                zb[m] = z[m]
                self.color[ys, xs][m] = rgb[i]

    def draw_textured_quad(self, quad, texture, bias=3e-4):
        """quad: world corners TL,TR,BR,BL. texture: PIL image. Perspective-correct UVs, bilinear sampling, depth-tested.
        The tiny depth bias lets a screen sitting on a bezel win the z-fight."""
        tex = np.asarray(texture.convert("RGB"), np.float32) / 255.0
        th, tw = tex.shape[:2]
        quad = np.asarray(quad, np.float64)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
        for idx in ([0, 1, 2], [0, 2, 3]):
            tri, a = quad[idx][None], uv[idx][None]
            for s, iw, at, _ in self._prepare(tri, a):
                c = self._cover(s, iw, bias)
                if c is None:
                    continue
                ys, xs, l0, l1, l2, z, inside = c
                zb = self.iw[ys, xs]
                m = inside & (z > zb)
                if not m.any():
                    continue
                w0, w1, w2 = l0[m] * iw[0], l1[m] * iw[1], l2[m] * iw[2]
                den = w0 + w1 + w2
                u = (w0 * at[0, 0] + w1 * at[1, 0] + w2 * at[2, 0]) / den
                v = (w0 * at[0, 1] + w1 * at[1, 1] + w2 * at[2, 1]) / den
                fx = np.clip(u * tw - 0.5, 0, tw - 1.001); fy = np.clip(v * th - 0.5, 0, th - 1.001)
                x0, y0 = fx.astype(int), fy.astype(int)
                tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
                samp = (tex[y0, x0] * (1 - tx) * (1 - ty) + tex[y0, x0 + 1] * tx * (1 - ty)
                        + tex[y0 + 1, x0] * (1 - tx) * ty + tex[y0 + 1, x0 + 1] * tx * ty)
                zb[m] = z[m]
                self.color[ys, xs][m] = samp

    def defer_translucent(self, tris, rgb, alpha):
        """Queue translucent triangles; they are blended in resolve() back-to-front against the opaque depth, without writing depth."""
        for s, iw, _, i in self._prepare(tris):
            self._late.append((1.0 / float(iw.mean()), s, iw, rgb[i], float(alpha[i])))

    def fog_factor(self, depth):
        L = self.light
        return np.clip(1.0 - np.exp(-L.fog_density * np.maximum(depth - L.fog_start, 0.0)), 0.0, 0.95)

    def resolve(self) -> np.ndarray:
        """Apply fog to opaque pixels, blend translucents, downsample. Returns float32 (out_h, out_w, 3), 0..1."""
        fc = np.asarray(self.light.fog_color, np.float32)
        hit = self.iw > 0
        depth = 1.0 / np.maximum(self.iw, 1e-9)
        f = np.where(hit, self.fog_factor(depth), 0.0).astype(np.float32)[..., None]
        self.color = self.color * (1 - f) + fc * f
        for depth_f, s, iw, c, a in sorted(self._late, key=lambda t: -t[0]):
            cov = self._cover(s, iw)
            if cov is None:
                continue
            ys, xs, _, _, _, z, inside = cov
            m = inside & (z > self.iw[ys, xs])
            if m.any():
                ff = float(self.fog_factor(np.float64(depth_f)))
                cc = np.asarray(c, np.float32) * (1 - ff) + fc * ff
                px = self.color[ys, xs]
                px[m] = px[m] * (1 - a) + cc * a
        self._late = []
        img = np.clip(self.color, 0.0, 1.0)
        k = self.ssaa
        return img.reshape(self.out_h, k, self.out_w, k, 3).mean((1, 3)).astype(np.float32)


# ------------------------------------------------------------------ frame -> image
def render(frame, view, size=(1280, 720), ssaa=2, lighting: Lighting | None = None, post=None):
    """Same contract as render_soft.render: returns a PIL RGB image with the HUD and log overlaid.
    post: optional dashboard3d.postfx.PostFX; the effects hit the 3D picture only, the HUD and log stay crisp."""
    w, h = size
    rz = Rasterizer(size, ssaa, lighting=lighting)
    batches = []
    for b in (frame.main, frame.front):
        tris, col, emi, alp = b.arrays()
        if len(tris):
            rgb, _ = shade_faces(tris, col, emi, rz.eye, rz.light)
            batches.append((tris, rgb, alp))
    for tris, rgb, alp in batches:                                       # opaque, any order (z-buffer sorts it out)
        solid = alp >= 0.999
        if solid.any():
            rz.draw_opaque(tris[solid], rgb[solid])
    tex_cv = PilCanvas((screen.SW, screen.SH))                           # monitor content goes into the same depth buffer
    screen.draw_screen(tex_cv, view)
    rz.draw_textured_quad(frame.screen_quad, tex_cv.img)
    for tris, rgb, alp in batches:                                       # translucent last
        glass = alp < 0.999
        if glass.any():
            rz.defer_translucent(tris[glass], rgb[glass], alp[glass])
    pic = rz.resolve()
    if post is not None:
        from dashboard3d import postfx
        pic = postfx.apply(pic, t=float(getattr(view, "clock", 0.0) or 0.0), cfg=post)
    img = Image.fromarray((pic * 255 + 0.5).astype(np.uint8), "RGB")
    top = PilCanvas((w, 64))
    hud.draw_hud(top, view, w)
    img.paste(top.img, (0, 0))
    lh = 150
    lg = PilCanvas((w, lh))
    hud.draw_log(lg, view, w, lh)
    img.paste(lg.img, (0, h - lh))
    return img
