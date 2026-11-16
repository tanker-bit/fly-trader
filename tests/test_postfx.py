"""Post-processing effects (dashboard3d/postfx.py). Headless, numpy only.  Run:  python -m tests.test_postfx"""
from __future__ import annotations

import numpy as np

from dashboard3d import postfx
from dashboard3d.postfx import PostFX

OFF = PostFX(bloom=0, pixel=1, rain=0, scanline=0, vignette=0, grain=0)


def _img(h=90, w=160, v=0.2, seed=1):
    return np.clip(np.random.default_rng(seed).uniform(0, 0.6, (h, w, 3)) + v - 0.2, 0, 1).astype(np.float32)


def test_everything_off_is_identity():
    a = _img()
    assert np.allclose(postfx.apply(a, 1.3, OFF), a)


def test_box_blur_matches_brute_force_and_keeps_constants():
    a = np.random.default_rng(3).uniform(0, 1, (12, 15, 3)).astype(np.float32)
    got = postfx._box1d(a, 2, 1)
    ref = np.empty_like(a)
    for x in range(15):
        idx = np.clip(np.arange(x - 2, x + 3), 0, 14)                 # edge-replicated window
        ref[:, x] = a[:, idx].mean(1)
    assert np.allclose(got, ref, atol=1e-5)
    c = np.full((20, 30, 3), 0.4, np.float32)
    assert np.allclose(postfx.blur(c, 3), 0.4, atol=1e-5)


def test_bloom_spreads_light_from_a_bright_spot_but_not_from_dark_areas():
    a = np.full((90, 160, 3), 0.05, np.float32)
    a[40:52, 74:86] = 1.0                                           # a bright 12x12 patch (a neon sign)
    out = postfx.bloom(a, 0.8, 0.5, 0.02)
    assert out[46, 90, 0] > a[46, 90, 0] + 0.01                      # neighbours gained light
    assert abs(out[5, 5, 0] - a[5, 5, 0]) < 1e-3                     # far away: untouched
    dark = np.full((90, 160, 3), 0.3, np.float32)
    assert np.allclose(postfx.bloom(dark, 0.8, 0.5, 0.02), dark)     # below threshold: nothing glows


def test_pixelate_makes_constant_blocks_and_handles_odd_sizes():
    a = _img(91, 163)
    out = postfx.pixelate(a, 4)
    assert out.shape == a.shape
    assert np.allclose(out[0:4, 0:4], out[0, 0]) and not np.allclose(out[0:4, 0:4], a[0:4, 0:4])


def test_scanlines_darken_every_period_row_only():
    a = np.ones((9, 5, 3), np.float32)
    out = postfx.scanlines(a, 0.3, 3)
    assert np.allclose(out[2::3], 0.7) and np.allclose(out[0::3], 1.0) and np.allclose(out[1::3], 1.0)


def test_vignette_darkens_corners_not_centre():
    out = postfx.vignette(np.ones((90, 160, 3), np.float32), 0.5)
    assert out[45, 80, 0] > 0.99 and out[0, 0, 0] < 0.6


def test_rain_is_reproducible_moves_with_time_and_only_adds_light():
    a = np.zeros((90, 160, 3), np.float32)
    r1, r1b, r2 = (postfx.rain(a, 120, 0.4, t, 7) for t in (0.5, 0.5, 0.9))
    assert np.array_equal(r1, r1b) and not np.array_equal(r1, r2)
    assert r1.min() >= 0 and r1.max() > 0.1 and (r1 > 0).any()


def test_grain_changes_over_time_with_zero_mean():
    a = np.full((90, 160, 3), 0.5, np.float32)
    g1, g2 = postfx.grain(a, 0.05, 0.0, 7), postfx.grain(a, 0.05, 1.0, 7)
    assert not np.array_equal(g1, g2) and abs(g1.mean() - 0.5) < 0.01


def test_full_pipeline_is_finite_in_range_and_deterministic():
    a = _img(180, 320)
    o1, o2 = postfx.apply(a, 2.0, PostFX()), postfx.apply(a, 2.0, PostFX())
    assert o1.shape == a.shape and np.isfinite(o1).all() and o1.min() >= 0 and o1.max() <= 1 and np.array_equal(o1, o2)


def test_raster_render_accepts_post_and_changes_only_the_3d_area():
    from dashboard.scene import FlyAnimator, View
    from dashboard3d import raster, scene3d
    a = FlyAnimator()
    for _ in range(10):
        a.update(0.02)
    hdr = dict(round=9, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league")
    v = View(header=hdr, pose=a.pose(), log=[], clock=a.clock)
    fr = scene3d.build_frame(v.pose, a.clock, "X")
    base = np.asarray(raster.render(fr, v, (640, 360), ssaa=1)).astype(int)
    fx = np.asarray(raster.render(fr, v, (640, 360), ssaa=1, post=PostFX())).astype(int)
    assert np.abs(base[80:200] - fx[80:200]).mean() > 1.0              # the 3D picture changed
    assert np.array_equal(base[:62], fx[:62]) and np.array_equal(base[-148:], fx[-148:])   # HUD and log rows are untouched


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all postfx tests passed")
