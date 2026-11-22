"""Post-processing for the software renderer (numpy + PIL): bloom, pixelation, rain, scanlines, vignette, film grain.

Chunk 2 of the 3D dashboard upgrade. Works on a float32 image (H, W, 3) in 0..1, the output of Rasterizer.resolve().
Every effect has an off switch, so PostFX(bloom=0, pixel=1, rain=0, scanline=0, vignette=0, grain=0) returns the image unchanged.

    from dashboard3d.postfx import PostFX, apply
    out = apply(img_float, t=clock, cfg=PostFX())        # t (seconds) animates rain and grain; same t -> same picture
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PostFX:
    bloom: float = 0.60              # strength of the glow added around bright pixels (0 = off)
    bloom_threshold: float = 0.55    # only pixels brighter than this glow
    bloom_radius: float = 0.010      # blur radius as a fraction of the image width
    pixel: int = 2                   # pixelation block size in image pixels (1 = off)
    rain: int = 170                  # number of rain streaks (0 = off)
    rain_alpha: float = 0.32
    scanline: float = 0.16           # darkness of every `scan_period`-th row (0 = off)
    scan_period: int = 3
    vignette: float = 0.50           # corner darkening (0 = off)
    grain: float = 0.035             # film grain amplitude (0 = off)
    seed: int = 7


# ------------------------------------------------------------------ helpers
def _box1d(a, r, axis):
    """Moving average of width 2r+1 along `axis` (edge-replicated), via cumulative sums."""
    if r < 1:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a, r, passes=3):
    """Approximate Gaussian blur: `passes` separable box blurs of radius r."""
    for _ in range(passes):
        a = _box1d(_box1d(a, r, 0), r, 1)
    return a


def _resize(a, size, resample):
    """Resize a float (H, W, 3) array with PIL (per channel, mode 'F' keeps values above 1)."""
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(a[..., k]), "F").resize(size, resample)) for k in range(3)]
    return np.stack(chans, -1).astype(np.float32)


def bloom(img, strength, threshold, radius_frac):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    f = 4                                                          # blur at quarter resolution: 16x cheaper
    sw, sh = max(w // f, 1), max(h // f, 1)
    bright = np.clip(img - threshold, 0, None) / max(1.0 - threshold, 1e-6)
    small = _resize(bright, (sw, sh), Image.BOX)
    r = max(int(round(radius_frac * w / f)), 1)
    glow = _resize(blur(small, r), (w, h), Image.BICUBIC)
    return img + strength * np.clip(glow, 0, None)


def pixelate(img, p):
    if p <= 1:
        return img
    h, w = img.shape[:2]
    ph, pw = -h % p, -w % p
    a = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = a.shape[:2]
    small = a.reshape(H // p, p, W // p, p, 3).mean((1, 3))
    return np.repeat(np.repeat(small, p, 0), p, 1)[:h, :w]


def rain(img, n, alpha, t, seed):
    """n slanted streaks falling at different speeds; positions depend only on (seed, t) so frames are reproducible."""
    if n <= 0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, w, n)
    y0 = rng.uniform(0, h, n)
    speed = rng.uniform(0.9, 1.8, n) * h * 0.9                     # pixels / second
    length = rng.integers(max(h // 40, 4), max(h // 14, 8), n)
    bright = rng.uniform(0.5, 1.0, n)
    slant = 0.22                                                    # x shift per y pixel
    y_head = (y0 + speed * t) % (h + 60)
    out = img.copy()
    col = np.array([0.60, 0.75, 1.00], np.float32)
    for k in range(int(length.max())):
        live = length > k
        yy = np.round(y_head[live] - k).astype(int)
        xx = np.round(x0[live] - slant * (y_head[live] - k)).astype(int) % w      # round first, then wrap
        ok = (yy >= 0) & (yy < h)
        fade = (1.0 - k / length[live]) * bright[live] * alpha     # brightest at the head, fading along the tail
        np.add.at(out, (yy[ok], xx[ok]), fade[ok, None] * col)
    return out


def scanlines(img, strength, period):
    if strength <= 0:
        return img
    m = np.ones(img.shape[0], np.float32)
    m[period - 1::period] = 1.0 - strength
    return img * m[:, None, None]


_VIG = {}


def vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    key = (h, w)
    if key not in _VIG:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((x / w - 0.5) * 2) ** 2 + ((y / h - 0.5) * 2) ** 2) / np.sqrt(2)      # 0 centre .. 1 corner
        _VIG[key] = np.clip((d - 0.35) / 0.65, 0, 1) ** 2
    return img * (1.0 - strength * _VIG[key])[..., None]


def grain(img, amount, t, seed):
    if amount <= 0:
        return img
    rng = np.random.default_rng([seed, int(t * 24)])                # new noise 24 times a second
    return img + rng.normal(0.0, amount, img.shape[:2]).astype(np.float32)[..., None]


# ------------------------------------------------------------------ pipeline
def apply(img, t=0.0, cfg: PostFX | None = None):
    """Bloom -> pixelate -> rain -> scanlines -> vignette -> grain. Returns float32 clipped to 0..1."""
    cfg = cfg or PostFX()
    a = np.asarray(img, np.float32)
    a = bloom(a, cfg.bloom, cfg.bloom_threshold, cfg.bloom_radius)
    a = pixelate(a, cfg.pixel)
    a = rain(a, cfg.rain, cfg.rain_alpha, t, cfg.seed)
    a = scanlines(a, cfg.scanline, cfg.scan_period)
    a = vignette(a, cfg.vignette)
    a = grain(a, cfg.grain, t, cfg.seed)
    return np.clip(a, 0.0, 1.0)
