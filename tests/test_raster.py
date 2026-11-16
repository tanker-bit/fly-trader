"""Z-buffer rasterizer (dashboard3d/raster.py): occlusion, texture mapping, clipping, fog, translucency. Headless, numpy only.
Run:  python -m tests.test_raster"""
from __future__ import annotations

import numpy as np
from PIL import Image

from dashboard.scene import FlyAnimator, View
from dashboard.state import Decision
from dashboard3d import raster, scene3d
from dashboard3d.m3d import box_mesh, trans

SZ = (160, 90)
FLAT = raster.Lighting(sky_ambient=(1, 1, 1), ground_ambient=(1, 1, 1), dirs=[], points=[], fog_density=0.0)   # no shading, no fog


def _tri(z, half=1.5):
    """A big triangle facing the camera at depth z along the view axis (camera at EYE looking at TARGET)."""
    e, t = np.asarray(scene3d.EYE, float), np.asarray(scene3d.TARGET, float)
    f = (t - e) / np.linalg.norm(t - e)
    r = np.cross(f, (0, 1, 0)); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c = e + f * z
    return np.array([[c - r * half - u * half, c + r * half - u * half, c + u * half * 1.5]])


def _centre(rz_img):
    h, w = rz_img.shape[:2]
    return rz_img[h // 2, w // 2]


def test_nearer_triangle_wins_in_either_draw_order():
    near, far = _tri(6.0), _tri(10.0, half=3.0)
    red, blue = np.array([[1, 0, 0]], np.float32), np.array([[0, 0, 1]], np.float32)
    for order in ((0, 1), (1, 0)):
        rz = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
        jobs = [(near, red), (far, blue)]
        for k in order:
            rz.draw_opaque(*jobs[k])
        c = _centre(rz.resolve())
        assert c[0] > 0.9 and c[2] < 0.1, (order, c)          # red (near) on top whichever was drawn first


def test_ssaa_output_size_and_antialiasing():
    rz = raster.Rasterizer(SZ, ssaa=2, lighting=FLAT)
    rz.draw_opaque(_tri(6.0), np.array([[1, 1, 1]], np.float32))
    img = rz.resolve()
    assert img.shape == (SZ[1], SZ[0], 3)
    edge = img[(img > 0.05) & (img < 0.95)]
    assert edge.size > 0                                       # edge pixels have in-between values (supersampled)


def test_triangle_behind_camera_is_dropped_and_straddling_one_is_clipped():
    rz = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
    e = np.asarray(scene3d.EYE, float)
    behind = _tri(-5.0)
    rz.draw_opaque(behind, np.array([[1, 0, 0]], np.float32))
    assert (rz.iw == 0).all()
    e, t = np.asarray(scene3d.EYE, float), np.asarray(scene3d.TARGET, float)
    f = (t - e) / np.linalg.norm(t - e)
    strad = _tri(6.0).copy()
    strad[0, 2] = e - f * 3.0                                   # one vertex 3 units BEHIND the eye, two in front
    rz2 = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
    rz2.draw_opaque(strad, np.array([[0, 1, 0]], np.float32))
    assert np.isfinite(rz2.iw).all() and np.isfinite(rz2.color).all()
    assert (rz2.iw > 0).any()                                   # the visible part was drawn, not silently discarded


def test_textured_quad_keeps_corner_orientation():
    tex = Image.new("RGB", (64, 40), (0, 0, 0))
    for (x0, y0, col) in ((0, 0, (255, 0, 0)), (32, 0, (0, 255, 0)), (32, 20, (0, 0, 255)), (0, 20, (255, 255, 0))):
        tex.paste(col, (x0, y0, x0 + 32, y0 + 20))              # TL red, TR green, BR blue, BL yellow
    e, t = np.asarray(scene3d.EYE, float), np.asarray(scene3d.TARGET, float)
    f = (t - e) / np.linalg.norm(t - e); r = np.cross(f, (0, 1, 0)); r /= np.linalg.norm(r); u = np.cross(r, f)
    c = e + f * 8
    quad = np.array([c - r * 2 + u, c + r * 2 + u, c + r * 2 - u, c - r * 2 - u])    # TL, TR, BR, BL facing the camera
    rz = raster.Rasterizer((320, 180), ssaa=1, lighting=FLAT)
    rz.draw_textured_quad(quad, tex)
    img = rz.resolve()
    h, w = img.shape[:2]
    tl, tr, br, bl = img[h // 2 - 8, w // 2 - 20], img[h // 2 - 8, w // 2 + 20], img[h // 2 + 8, w // 2 + 20], img[h // 2 + 8, w // 2 - 20]
    assert tl[0] > 0.8 and tl[1] < 0.2 and tr[1] > 0.8 and tr[0] < 0.2 and br[2] > 0.8 and bl[0] > 0.8 and bl[1] > 0.8


def test_textured_quad_is_hidden_behind_nearer_geometry():
    tex = Image.new("RGB", (8, 8), (255, 0, 255))
    rz = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
    rz.draw_opaque(_tri(5.0, half=3.0), np.array([[0, 0, 1]], np.float32))      # blue wall in front
    e, t = np.asarray(scene3d.EYE, float), np.asarray(scene3d.TARGET, float)
    f = (t - e) / np.linalg.norm(t - e); r = np.cross(f, (0, 1, 0)); r /= np.linalg.norm(r); u = np.cross(r, f)
    c = e + f * 9
    rz.draw_textured_quad(np.array([c - r + u, c + r + u, c + r - u, c - r - u]), tex)
    px = _centre(rz.resolve())
    assert px[2] > 0.9 and px[0] < 0.1                                           # still blue, not magenta


def test_fog_blends_far_surfaces_toward_fog_colour():
    L = raster.Lighting(sky_ambient=(1, 1, 1), ground_ambient=(1, 1, 1), fog_color=(0, 0, 0), fog_density=0.2, fog_start=2.0)
    res = []
    for z in (4.0, 14.0):
        rz = raster.Rasterizer(SZ, ssaa=1, lighting=L)
        rz.draw_opaque(_tri(z, half=z), np.array([[1, 1, 1]], np.float32))
        res.append(_centre(rz.resolve()).mean())
    assert res[0] > res[1] + 0.3


def test_translucent_blends_over_opaque_but_not_behind_it():
    over = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
    over.draw_opaque(_tri(10.0, half=4.0), np.array([[0, 0, 1]], np.float32))     # blue far
    over.defer_translucent(_tri(6.0), np.array([[1, 0, 0]], np.float32), np.array([0.5], np.float32))   # red glass in front
    c = _centre(over.resolve())
    assert 0.4 < c[0] < 0.6 and 0.4 < c[2] < 0.6
    behind = raster.Rasterizer(SZ, ssaa=1, lighting=FLAT)
    behind.draw_opaque(_tri(5.0, half=4.0), np.array([[0, 0, 1]], np.float32))    # blue near
    behind.defer_translucent(_tri(9.0), np.array([[1, 0, 0]], np.float32), np.array([0.5], np.float32))  # glass behind it
    c = _centre(behind.resolve())
    assert c[2] > 0.95 and c[0] < 0.05


def test_lighting_is_two_sided_and_emissive_ignores_it():
    tris = box_mesh(1, 1, 1)[None, 0] + 0
    col = np.array([[0.5, 0.5, 0.5]], np.float32)
    L = raster.default_lighting()
    lit, _ = raster.shade_faces(tris, col, np.array([False]), scene3d.EYE, L)
    flipped, _ = raster.shade_faces(tris[:, ::-1], col, np.array([False]), scene3d.EYE, L)
    assert np.allclose(lit, flipped, atol=1e-5)                                   # winding does not matter
    emi, _ = raster.shade_faces(tris, col, np.array([True]), scene3d.EYE, L)
    assert np.allclose(emi, col)


def test_full_scene_renders_every_state_not_blank():
    hdr = dict(round=9, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league")
    for act in ("BUY", "SELL", "HOLD", "SILENT", None):
        a = FlyAnimator()
        if act:
            a.push(Decision("2022-03-15", "AAPL", act, 3.2, "a1", 8))
        for _ in range(38):
            a.update(0.02)
        v = View(header=hdr, pose=a.pose(), log=list(a.log), clock=a.clock)
        im = raster.render(scene3d.build_frame(v.pose, a.clock, "AAPL"), v, size=(640, 360), ssaa=1)
        arr = np.asarray(im)
        assert im.size == (640, 360) and arr.std() > 10 and (arr.sum(2) > 60).mean() > 0.2


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all raster tests passed")
