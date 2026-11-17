"""Before/after screenshot: old painter's renderer (render_soft) vs the new z-buffer rasterizer (raster), same BUY frame.
Headless, writes previews/zbuffer_before_after.png (before on the left, after on the right).
Run from the repo root:  python -m experiments.compare_renderers"""
from __future__ import annotations

import os
import time

from PIL import Image, ImageDraw

from dashboard.scene import FlyAnimator, View
from dashboard.state import Decision
from dashboard3d import raster, render_soft, scene3d


def main(out="previews/zbuffer_before_after.png", size=(1280, 720)):
    a = FlyAnimator()
    a.push(Decision("2022-03-15", "AAPL", "BUY", 3.2, "a1", 8))
    for _ in range(37):
        a.update(0.02)
    hdr = dict(round=9, universe=499, top_id="a1", top_ic=0.05, ridge_ic=0.02, mode="league")
    v = View(header=hdr, pose=a.pose(), log=list(a.log), clock=a.clock)
    frame = scene3d.build_frame(v.pose, a.clock, "AAPL")
    t0 = time.time(); before = render_soft.render(frame, v, size); t1 = time.time()
    after = raster.render(frame, v, size, ssaa=2); t2 = time.time()
    half = (size[0] // 2, size[1] // 2)
    sheet = Image.new("RGB", (half[0] * 2, half[1] + 28), (10, 10, 14))
    sheet.paste(before.resize(half, Image.LANCZOS), (0, 28))
    sheet.paste(after.resize(half, Image.LANCZOS), (half[0], 28))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"BEFORE: painter's algorithm ({t1 - t0:.2f}s)", fill=(255, 190, 60))
    d.text((half[0] + 8, 8), f"AFTER: z-buffer, lights, fog, 2x supersampling ({t2 - t1:.2f}s)", fill=(57, 255, 160))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("wrote", out, f"(before {t1 - t0:.2f}s, after {t2 - t1:.2f}s)")


if __name__ == "__main__":
    main()
