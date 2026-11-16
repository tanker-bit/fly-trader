"""3D geometry, posing and the software renderer (no GPU needed)."""
from __future__ import annotations

import numpy as np

from dashboard.scene import FlyAnimator, View
from dashboard.state import Decision
from dashboard3d import m3d, scene3d
from dashboard3d.render_soft import homography, render


def D(act, tk="AAPL", sc=3.2):
    return Decision("2022-03-15", tk, act, sc, "a1", 8)


def posed(act=None, secs=0.8, research=False):
    a = FlyAnimator()
    a.set_research(research)
    if act:
        a.push(D(act))
    for _ in range(int(secs / 0.02)):
        a.update(0.02)
    return a


def test_primitives_and_ik():
    assert m3d.box_mesh().shape == (12, 3, 3) and m3d.prism_mesh(sides=6).shape == (24, 3, 3)
    b = m3d.Batch()
    b.box((1, 2, 3), (2, 2, 2), (255, 0, 0))
    t, c, e, al = b.arrays()
    assert np.allclose(t.reshape(-1, 3).mean(0), (1, 2, 3)) and c.shape == (12, 3) and not e.any() and (al == 1).all()
    elbow, hand = m3d.two_bone_ik((0, 0, 0), (1.2, 0.3, 0), 1.0, 1.0)
    assert np.allclose(hand, (1.2, 0.3, 0)) and abs(np.linalg.norm(elbow) - 1) < 1e-9 and abs(np.linalg.norm(elbow - hand) - 1) < 1e-9
    far, h2 = m3d.two_bone_ik((0, 0, 0), (5, 0, 0), 1.0, 1.0)                   # out of reach: stretches, no NaN
    assert np.isfinite(far).all() and np.linalg.norm(h2) <= 2.0


def test_every_state_builds_finite_geometry_and_levers_move_the_right_way():
    tips = {}
    for act, research in (("BUY", False), ("SELL", False), ("HOLD", False), ("SILENT", False), (None, True), (None, False)):
        a = posed(act, 0.75 if act in ("BUY", "SELL") else 0.9, research)
        f = scene3d.build_frame(a.pose(), a.clock, "AAPL")
        for batch in (f.main, f.front):
            t, c, e, al = batch.arrays()
            assert np.isfinite(t).all() and len(t) == len(c) == len(e) == len(al)
        assert f.screen_quad.shape == (4, 3)
        tips[act or ("research" if research else "idle")] = scene3d._lever  # smoke only
    # BUY pushes the green lever forward (+x), SELL pulls the red lever back toward the fly (-x)
    base = scene3d.LEVER_BASE
    b = m3d.Batch()
    tb = scene3d._lever(b, "BUY", scene3d.GREEN, +0.6)
    ts = scene3d._lever(b, "SELL", scene3d.RED, -0.6)
    assert tb[0] > base["BUY"][0] + 0.3 and ts[0] < base["SELL"][0] - 0.3


def test_silent_fly_is_lower_and_sideways_than_awake_fly():
    fa = scene3d.build_frame(posed("HOLD").pose(), 1.0, "AAPL")
    fs = scene3d.build_frame(posed("SILENT", 1.0).pose(), 1.0, "AAPL")
    def top(f):                       # highest point of the opaque, unlit-free body parts (ignores glow, wings, Zzz and levers)
        t, c, e, al = f.front.arrays()
        keep = (~e) & (al >= 0.999) & (t[:, :, 0].mean(1) < -1.8)         # the fly's own side of the desk
        return t[keep][:, :, 1].max()
    ya, ys = top(fa), top(fs)
    assert ys < ya                                                                # slumped: the highest point drops
    # Zzz glyphs exist only when silent (amber emissive translucent boxes)
    assert (fs.front.arrays()[2] & (fs.front.arrays()[3] < 1)).any()
    assert not (fa.front.arrays()[2] & (fa.front.arrays()[3] < 1)).any()


def test_homography_maps_quad_corners_to_texture_corners():
    dst = [(100, 50), (400, 80), (380, 300), (90, 260)]
    c = homography(dst, (512, 320))
    for (x, y), (u, v) in zip(dst, [(0, 0), (512, 0), (512, 320), (0, 320)]):
        den = c[6] * x + c[7] * y + 1
        assert abs((c[0] * x + c[1] * y + c[2]) / den - u) < 1e-6 and abs((c[3] * x + c[4] * y + c[5]) / den - v) < 1e-6


def test_soft_renderer_draws_every_state():
    hdr = dict(round=9, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league")
    for act, research in (("BUY", False), ("SELL", False), ("HOLD", False), ("SILENT", False), (None, True)):
        a = posed(act, 0.75, research)
        v = View(header=hdr, pose=a.pose(), log=list(a.log), clock=a.clock,
                 research_lines=["AAPL   Revenues                          383285000000  FY2023 filed 2023-11-03"] * 3)
        im = render(scene3d.build_frame(v.pose, a.clock, v.pose.ticker), v, size=(640, 360))
        arr = np.asarray(im)
        assert im.size == (640, 360) and arr.std() > 10 and (arr.sum(2) > 60).mean() > 0.2   # not blank
    # BUY lights the screen green, SELL red
    def screen_color(act):
        a = posed(act, 0.75)
        v = View(header=hdr, pose=a.pose(), log=[], clock=a.clock)
        arr = np.asarray(render(scene3d.build_frame(v.pose, a.clock, "X"), v, size=(640, 360))).astype(int)
        reg = arr[60:230, 330:560]                                                    # where the monitor is
        return reg[..., 0].mean(), reg[..., 1].mean()
    r_b, g_b = screen_color("BUY")
    r_s, g_s = screen_color("SELL")
    assert g_b > r_b and r_s > g_s


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all dashboard3d tests passed")
