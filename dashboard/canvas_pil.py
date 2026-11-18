"""PIL canvas for tests and screenshots (the window uses dashboard/canvas_pygame.py with the same interface)."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from dashboard.scene import H, W


class PilCanvas:
    def __init__(self, size=(W, H)):
        self.img = Image.new("RGB", size)
        self.d = ImageDraw.Draw(self.img)
        self.calls = 0
        self._fonts = {}

    def font(self, size):
        if size not in self._fonts:
            for name in ("DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc"):
                try:
                    self._fonts[size] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    continue
            else:
                self._fonts[size] = ImageFont.load_default(size)
        return self._fonts[size]

    def rect(self, x, y, w, h, fill, outline=None, width=1, alpha=None):
        self.calls += 1
        if fill is not None and alpha is not None:                       # translucent fill
            self.img.paste(fill, (int(x), int(y), int(x + w), int(y + h)), mask=Image.new("L", (int(w), int(h)), alpha))
            fill = None
        self.d.rectangle([x, y, x + w - 1, y + h - 1], fill=fill, outline=outline, width=width if outline else 0)

    def ellipse(self, x, y, w, h, fill, outline=None, width=1):
        self.calls += 1
        self.d.ellipse([x, y, x + w, y + h], fill=fill, outline=outline, width=width if outline else 0)

    def line(self, x1, y1, x2, y2, color, width=1):
        self.calls += 1
        self.d.line([x1, y1, x2, y2], fill=color, width=int(width))

    def polygon(self, pts, fill, outline=None, alpha=None):
        self.calls += 1
        if alpha is not None and fill is not None:                       # translucent fill (wings)
            mask = Image.new("L", self.img.size, 0)
            ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in pts], fill=int(alpha))
            self.img.paste(fill, (0, 0), mask=mask)
            fill = None
        self.d.polygon(pts, fill=fill, outline=outline)

    def text(self, s, x, y, color, size=14, anchor="l"):
        self.calls += 1
        f = self.font(size)
        w = self.d.textlength(s, font=f)
        x = x - w / 2 if anchor == "c" else x - w if anchor == "r" else x
        self.d.text((x, y), s, fill=color, font=f)

    def text_width(self, s, size=14):
        return self.d.textlength(s, font=self.font(size))

    def save(self, path):
        self.img.save(path)
