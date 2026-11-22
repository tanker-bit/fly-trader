"""Software renderer (PIL + numpy): painter's algorithm over the low-poly triangles, homography warp for the monitor texture.
Used for tests, screenshots and as a fallback when OpenGL is unavailable. The GL renderer draws the identical Frame."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from dashboard.canvas_pil import PilCanvas
from dashboard3d import hud, screen
from dashboard3d.m3d import look_at, perspective
from dashboard3d.scene3d import EYE, FOV, TARGET, UP_LIGHT


def project(tris, size):
    w, h = size
    V = look_at(EYE, TARGET)
    P = perspective(FOV, w / h, 0.5, 80.0)
    pts = tris.reshape(-1, 3)
    cam = pts @ V[:3, :3].T + V[:3, 3]
    clip = np.concatenate([cam, np.ones((len(cam), 1))], 1) @ P.T
    wv = clip[:, 3]
    ok = (wv > 0.3).reshape(-1, 3).all(1)
    ndc = clip[:, :2] / np.maximum(wv[:, None], 1e-6)
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * h
    scr = np.stack([sx, sy], 1).reshape(-1, 3, 2)
    depth = (-cam[:, 2]).reshape(-1, 3).mean(1)
    return scr, depth, ok


def shade(tris, col, emi):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    lam = np.abs(n @ UP_LIGHT)
    f = np.where(emi, 1.0, 0.38 + 0.62 * lam)[:, None]
    return np.clip(col * f, 0, 1)


def _draw(img, batch, size, alpha_layer):
    tris, col, emi, alp = batch.arrays()
    if not len(tris):
        return
    scr, depth, ok = project(tris, size)
    rgb = (shade(tris, col, emi) * 255).astype(np.uint8)
    order = np.argsort(-depth)
    d = ImageDraw.Draw(img)
    ad = ImageDraw.Draw(alpha_layer)
    for i in order:
        if not ok[i]:
            continue
        pts = [tuple(p) for p in scr[i]]
        if alp[i] >= 0.999:
            d.polygon(pts, fill=tuple(int(x) for x in rgb[i]))
        else:
            ad.polygon(pts, fill=tuple(int(x) for x in rgb[i]) + (int(255 * alp[i]),))


def homography(dst, src_size):
    """Coefficients for PIL's PERSPECTIVE transform: maps output (dst quad) pixels back to the source texture."""
    sw, sh = src_size
    src = [(0, 0), (sw, 0), (sw, sh), (0, sh)]
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float))


def render(frame, view, size=(1280, 720)):
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(h):                                                       # dusk gradient
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(8 + 38 * t), int(4 + 8 * t), int(26 + 40 * t)))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    _draw(img, frame.main, size, layer)
    # monitor screen: draw the 2D screen content, warp it onto the projected quad
    tex_cv = PilCanvas((screen.SW, screen.SH))
    screen.draw_screen(tex_cv, view)
    q, _, ok = project(frame.screen_quad[None, [0, 1, 2]], size)
    quad, _, _ = project(np.stack([frame.screen_quad[[0, 1, 2]], frame.screen_quad[[0, 2, 3]]]), size)
    dst = [tuple(quad[0][0]), tuple(quad[0][1]), tuple(quad[0][2]), tuple(quad[1][2])]
    coef = homography(dst, tex_cv.img.size)
    warped = tex_cv.img.transform(size, Image.PERSPECTIVE, tuple(coef), Image.BILINEAR)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(dst, fill=255)
    img.paste(warped, (0, 0), mask)
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    layer2 = Image.new("RGBA", size, (0, 0, 0, 0))
    _draw(img, frame.front, size, layer2)
    img = Image.alpha_composite(img.convert("RGBA"), layer2).convert("RGB")
    top = PilCanvas((w, 64))
    hud.draw_hud(top, view, w)
    img.paste(top.img, (0, 0))
    lh = 150
    lg = PilCanvas((w, lh))
    hud.draw_log(lg, view, w, lh)
    img.paste(lg.img, (0, h - lh))
    return img
