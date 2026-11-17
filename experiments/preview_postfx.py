"""Writes previews/postfx_demo.png: the z-buffer render without and with post-processing (bloom, pixelation, rain, scanlines,
vignette, grain), side by side. Headless. Run from the repo root:  python -m experiments.preview_postfx"""
from __future__ import annotations

import os
import time

from PIL import Image, ImageDraw

from dashboard.scene import FlyAnimator, View
from dashboard.state import Decision
from dashboard3d import raster, scene3d
from dashboard3d.postfx import PostFX


def main(out="previews/postfx_demo.png", size=(1280, 720)):
    a = FlyAnimator()
    a.push(Decision("2022-03-15", "AAPL", "BUY", 3.2, "a1", 8))
    for _ in range(37):
        a.update(0.02)
    hdr = dict(round=9, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league")
    v = View(header=hdr, pose=a.pose(), log=list(a.log), clock=a.clock)
    frame = scene3d.build_frame(v.pose, a.clock, "AAPL")
    t0 = time.time(); plain = raster.render(frame, v, size, ssaa=2); t1 = time.time()
    fx = raster.render(frame, v, size, ssaa=2, post=PostFX()); t2 = time.time()
    half = (size[0] // 2, size[1] // 2)
    sheet = Image.new("RGB", (half[0] * 2, half[1] + 28), (10, 10, 14))
    sheet.paste(plain.resize(half, Image.LANCZOS), (0, 28))
    sheet.paste(fx.resize(half, Image.LANCZOS), (half[0], 28))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"z-buffer only ({t1 - t0:.2f}s)", fill=(255, 190, 60))
    d.text((half[0] + 8, 8), f"+ postfx ({t2 - t1:.2f}s)", fill=(57, 255, 160))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    fx.save(out.replace("_demo", "_full"))
    print("wrote", out, "and", out.replace("_demo", "_full"))


if __name__ == "__main__":
    main()
