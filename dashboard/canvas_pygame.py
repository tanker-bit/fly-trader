"""pygame backend for the scene: every call maps straight onto pygame.draw (rect, ellipse, line, polygon) and font rendering.
Same interface as dashboard/canvas_pil.py, so the scene code is identical in the window and in the tests."""
from __future__ import annotations


class PygameCanvas:
    def __init__(self, surface, pygame):
        self.s, self.pg = surface, pygame
        self._fonts = {}

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = self.pg.font.SysFont("consolas,dejavusansmono,couriernew,menlo,monospace", size)
        return self._fonts[size]

    def rect(self, x, y, w, h, fill, outline=None, width=1, alpha=None):
        r = (int(x), int(y), int(w), int(h))
        if fill is not None:
            if alpha is not None:                                    # translucent fill (wings, terminal)
                layer = self.pg.Surface((r[2], r[3]), self.pg.SRCALPHA)
                layer.fill((fill[0], fill[1], fill[2], int(alpha)))
                self.s.blit(layer, (r[0], r[1]))
            else:
                self.pg.draw.rect(self.s, fill, r)
        if outline is not None:
            self.pg.draw.rect(self.s, outline, r, int(width))

    def ellipse(self, x, y, w, h, fill, outline=None, width=1):
        r = (int(x), int(y), max(1, int(w)), max(1, int(h)))
        if fill is not None:
            self.pg.draw.ellipse(self.s, fill, r)
        if outline is not None:
            self.pg.draw.ellipse(self.s, outline, r, int(width))

    def line(self, x1, y1, x2, y2, color, width=1):
        self.pg.draw.line(self.s, color, (int(x1), int(y1)), (int(x2), int(y2)), max(1, int(width)))

    def polygon(self, pts, fill, outline=None, alpha=None):
        pts = [(int(x), int(y)) for x, y in pts]
        if fill is not None:
            if alpha is not None:
                # Compute bounding box of the polygon to create a minimal temporary surface
                min_x = min(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_x = max(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                w, h = max_x - min_x + 1, max_y - min_y + 1
                
                layer = self.pg.Surface((w, h), self.pg.SRCALPHA)
                # Offset points relative to the temporary surface
                local_pts = [(p[0] - min_x, p[1] - min_y) for p in pts]
                self.pg.draw.polygon(layer, (fill[0], fill[1], fill[2], int(alpha)), local_pts)
                if outline is not None:
                    self.pg.draw.polygon(layer, (*outline, int(alpha)), local_pts, 1)
                self.s.blit(layer, (min_x, min_y))
                return
            else:
                self.pg.draw.polygon(self.s, fill, pts)
        if outline is not None and alpha is None:
            self.pg.draw.polygon(self.s, outline, pts, 1)

    def text_width(self, s, size=14):
        return self.font(size).size(s)[0]

    def text(self, s, x, y, color, size=14, anchor="l"):
        if not s:
            return
        img = self.font(size).render(s, False, color)                # no anti-aliasing: crisp retro look
        w = img.get_width()
        x = x - w // 2 if anchor == "c" else x - w if anchor == "r" else x
        self.s.blit(img, (int(x), int(y)))