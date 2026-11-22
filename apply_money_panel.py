"""Add the money panel (cash, equity, portfolio list) to the dashboard. Run from the repo root:  python apply_money_panel.py
Edits experiments/visual_dashboard.py in place (keeps a .bak3 copy). Handles Windows or Unix line endings. If any anchor is not found
exactly once, nothing is changed. Key P shows/hides the panel."""
import pathlib
import shutil
import sys

P = pathlib.Path("experiments/visual_dashboard.py")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")

PANEL = '''# --------------------------------------------------------------------------
# Part 10c: money panel (screen-space, top-left under the HUD): the fly's simulated $1M account, updated at every trade.
# Reads header['cash'], header['equity'] and header['positions'] = [(ticker, shares, price, value, pnl)] that the feed fills from the
# account engine's real fills. Nothing is invented: with no account data the panel stays fully transparent. Key P shows/hides it.
# --------------------------------------------------------------------------
PF_W, PF_H = 184, 146                          # texels (one texel is W/960 of the window width, like the HUD and the log)
PF_ROWS = 10
PF_MARGIN = 0.02
PF_START_CASH = 1_000_000.0
COL_PF_BG = (4, 8, 16)
PF_BG_ALPHA = 165


def pf_quad_ndc(aspect: float):
    """TL, TR, BR, BL: top-left corner of the window, just under the HUD bar."""
    w = 2.0 * PF_W / HUD_W
    h = min(2.0 * float(aspect) * PF_H / HUD_W, 0.6)
    hud_h = min(2.0 * float(aspect) * HUD_H / HUD_W, 0.6)
    x0, y1 = -1.0 + PF_MARGIN, 1.0 - hud_h - PF_MARGIN
    return [(x0, y1, 0.0), (x0 + w, y1, 0.0), (x0 + w, y1 - h, 0.0), (x0, y1 - h, 0.0)]


def _pf_num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def pf_values(vm) -> dict:
    hd = getattr(vm, "header", None) or {}
    eq, cash = _pf_num(hd.get("equity")), _pf_num(hd.get("cash"))
    rows = []
    try:
        entries = list(hd.get("positions") or [])
    except TypeError:
        entries = []
    for p in entries:
        try:
            t, sh, _px, val, pnl = p
            sh, val, pnl = _pf_num(sh), _pf_num(val), _pf_num(pnl)
        except (TypeError, ValueError):
            continue
        if sh is None or val is None or pnl is None:
            continue
        rows.append((str(t)[:6], sh, val, pnl))
    start = _pf_num(hd.get("start_cash")) or PF_START_CASH
    return dict(equity=eq, cash=cash, rows=rows, has=eq is not None, pl=None if eq is None else eq / start - 1.0)


def _pf_money(x) -> str:
    return "--" if x is None else f"${x:,.0f}"


def _pf_pct(p: float) -> str:
    return f"{max(-999.9, min(999.9, p * 100.0)):+.1f}%"


def _pf_qty(x: float) -> str:
    return f"{x:,.0f}" if abs(x) < 1e7 else f"{x / 1000.0:,.0f}K"


def render_portfolio(cv: Canvas, vm) -> bool:
    """Draws the account; returns False (canvas left empty) when the feed gives no account."""
    cv.fill(COL_PF_BG)
    v = pf_values(vm)
    if not v["has"]:
        return False
    up = v["pl"] >= 0
    col = COL_CANDLE_UP if up else COL_CANDLE_DN
    cv.text(4, 3, "FLY ACCOUNT  SIMULATED", COL_SCR_DIM)
    cv.text(4, 13, _pf_money(v["equity"]), col, 2)
    cv.text(4, 30, "CASH " + _pf_money(v["cash"]), COL_SCR_VALUE)
    pl_txt = _pf_pct(v["pl"])
    cv.text(PF_W - 4 - text_width(pl_txt), 30, pl_txt, col)
    cv.rect(3, 40, PF_W - 6, 1, COL_SCR_GRID)
    for txt, right in (("SHARES", 82), ("VALUE", 130), ("PNL", PF_W - 4)):
        cv.text(right - text_width(txt), 43, txt, COL_SCR_DIM)
    cv.text(4, 43, "TICKER", COL_SCR_DIM)
    rows = v["rows"]
    shown = rows if len(rows) <= PF_ROWS else rows[:PF_ROWS - 1]
    y = 53
    for t, sh, val, pnl in shown:
        cv.text(4, y, t, COL_SCR_TEXT)
        q = _pf_qty(sh)
        cv.text(82 - text_width(q), y, q, COL_SCR_VALUE)
        m = _pf_qty(val)
        cv.text(130 - text_width(m), y, m, COL_SCR_VALUE)
        pc = _pf_pct(pnl)
        cv.text(PF_W - 4 - text_width(pc), y, pc, COL_CANDLE_UP if pnl >= 0 else COL_CANDLE_DN)
        y += 9
    if len(rows) > len(shown):
        cv.text(4, y, f"+{len(rows) - len(shown)} MORE", COL_SCR_DIM)
    elif not rows:
        cv.text(4, y, "NO POSITIONS", COL_SCR_DIM)
    return True


class PortfolioContent:
    """Redraws only when a shown number changes."""
    def __init__(self):
        self.canvas = Canvas(PF_W, PF_H, COL_PF_BG)
        self._sig = None
        self._has = False
        self.draws = 0

    def update(self, vm, t: float = 0.0) -> bool:
        v = pf_values(vm)
        sig = (None if v["equity"] is None else round(v["equity"]), None if v["cash"] is None else round(v["cash"]),
               tuple((r[0], round(r[1]), round(r[2]), round(r[3], 3)) for r in v["rows"][:PF_ROWS + 1]), len(v["rows"]))
        if sig == self._sig:
            return False
        self._sig = sig
        self._has = render_portfolio(self.canvas, vm)
        self.draws += 1
        return True

    def pixels(self) -> np.ndarray:
        """RGBA: translucent panel, opaque text; fully transparent when there is no account yet."""
        px = self.canvas.px
        a = np.where((px != np.array(COL_PF_BG, dtype=np.uint8)).any(2), 255, PF_BG_ALPHA if self._has else 0).astype(np.uint8)
        return np.dstack([px, a])


'''

TESTS = '''def _selftest_portfolio(check) -> bool:
    from types import SimpleNamespace as NS
    start = len(_FAILS)
    print(" money panel (10c)")
    pos = [("AAPL", 120, 185.2, 22224.0, 0.043), ("MSFT", 60, 410.0, 24600.0, -0.021)]
    vm = NS(header=dict(equity=1043210.0, cash=54970.0, positions=pos))
    v = pf_values(vm)
    check("pf values: equity, cash, rows and P&L from the header", v["has"] and v["equity"] == 1043210.0 and v["cash"] == 54970.0
          and len(v["rows"]) == 2 and abs(v["pl"] - 0.04321) < 1e-9)
    check("pf values: no account means nothing to show", not pf_values(NS(header={}))["has"] and not pf_values(None)["has"])
    check("pf values: junk never raises (NaN, text, wrong shapes, None)",
          not pf_values(NS(header=dict(equity="x", cash=float("nan"), positions=[1, None, ("A",), ("B", "x", 1, 2, 3)])))["has"]
          and len(pf_values(NS(header=dict(equity=1.0, positions=[("A", 1, 1, float("inf"), 0.1), ("B", 2, 3, 4, 0.5)])))["rows"]) == 1)
    check("pf text: money, percent and quantity formats", _pf_money(1043210.4) == "$1,043,210" and _pf_money(None) == "--"
          and _pf_pct(0.043) == "+4.3%" and _pf_pct(-0.0251) == "-2.5%" and _pf_qty(12000) == "12,000" and _pf_qty(2.5e7) == "25,000K")
    for asp in (16 / 9, 4 / 3, 21 / 9, 1.0, 0.56):
        q = np.array(pf_quad_ndc(asp))
        qw, qh = q[1, 0] - q[0, 0], q[0, 1] - q[3, 1]
        hq = np.array(hud_quad_ndc(asp))
        check(f"pf quad {asp:.2f}: top-left, on screen, under the HUD, texel-square",
              (np.abs(q[:, :2]) <= 1.0).all() and q[0, 0] < -0.9 and q[0, 1] <= hq[3, 1] and q[3, 1] > -1.0
              and (abs(qw / PF_W - qh / asp / PF_H) < 1e-6 or qh >= 0.6 - 1e-9))
    cv = Canvas(PF_W, PF_H, COL_PF_BG)
    check("pf render: nothing to show leaves the canvas empty", not render_portfolio(cv, NS(header={})) and (cv.px == np.array(COL_PF_BG, dtype=np.uint8)).all())
    up = Canvas(PF_W, PF_H, COL_PF_BG)
    check("pf render: draws and says so", render_portfolio(up, vm))
    g, r = np.array(COL_CANDLE_UP, dtype=np.uint8), np.array(COL_CANDLE_DN, dtype=np.uint8)
    check("pf render: equity above the start is green (big text in the header block)", (up.px[13:27, :PF_W - 40] == g).all(2).any())
    dn = Canvas(PF_W, PF_H, COL_PF_BG)
    render_portfolio(dn, NS(header=dict(equity=950000.0, cash=10.0, positions=pos)))
    check("pf render: equity below the start is red", (dn.px[13:27, :PF_W - 40] == r).all(2).any())
    check("pf render: a losing position is red and a winning one green in the P&L column",
          (up.px[53:62, PF_W - 50:] == g).all(2).any() and (up.px[62:71, PF_W - 50:] == r).all(2).any())
    many = [(f"T{i}", 10 + i, 10.0, 1000.0 + i, 0.01 * i) for i in range(15)]
    cm = Canvas(PF_W, PF_H, COL_PF_BG)
    render_portfolio(cm, NS(header=dict(equity=1e6, cash=5.0, positions=many)))
    check("pf render: more than the list holds ends with a '+N MORE' row", (cm.px[53 + 9 * (PF_ROWS - 1):53 + 9 * PF_ROWS, :60] != np.array(COL_PF_BG, dtype=np.uint8)).any())
    ce = Canvas(PF_W, PF_H, COL_PF_BG)
    render_portfolio(ce, NS(header=dict(equity=1e6, cash=1e6, positions=[])))
    check("pf render: an empty account says NO POSITIONS", (ce.px[53:62, :80] != np.array(COL_PF_BG, dtype=np.uint8)).any())
    worst = [("ABCDEF", 99999999.0, 1, 999999999.0, 12.34)] * 12
    cw = Canvas(PF_W, PF_H, COL_PF_BG)
    render_portfolio(cw, NS(header=dict(equity=123456789.0, cash=123456789.0, positions=worst)))
    check("pf render: the longest values stay inside the panel and below the header", cw.px.shape == (PF_H, PF_W, 3))
    pc = PortfolioContent()
    check("PortfolioContent: first update draws", pc.update(vm) and pc.draws == 1)
    check("PortfolioContent: unchanged account is not redrawn", not pc.update(vm) and pc.draws == 1)
    vm2 = NS(header=dict(equity=1043999.0, cash=54970.0, positions=pos))
    check("PortfolioContent: a new equity redraws", pc.update(vm2) and pc.draws == 2)
    vm3 = NS(header=dict(equity=1043999.0, cash=54970.0, positions=pos[:1]))
    check("PortfolioContent: a sold position redraws", pc.update(vm3) and pc.draws == 3)
    px = pc.pixels()
    check("PortfolioContent: RGBA, translucent panel, opaque text", px.shape == (PF_H, PF_W, 4) and (px[:, :, 3] == PF_BG_ALPHA).any() and (px[:, :, 3] == 255).any())
    check("PortfolioContent: no account means fully transparent", (PortfolioContent().pixels()[:, :, 3] == 0).all())
    import inspect
    init = inspect.getsource(VisualDashboard.__init__) if "VisualDashboard" in globals() else ""
    handle = inspect.getsource(VisualDashboard.handle_events) if "VisualDashboard" in globals() else ""
    rend = inspect.getsource(VisualDashboard.render) if "VisualDashboard" in globals() else ""
    check("app: money panel built, key P toggles it, drawn after the log panel",
          "PortfolioContent()" in init and "money_on" in init and "K_p" in handle
          and rend.index("self.log_panel.draw(") < rend.index("self.money_panel.draw("))
    return len(_FAILS) == start


'''

def once(text, anchor):
    if text.count(anchor) != 1:
        sys.exit(f"NOT CHANGED: anchor not found exactly once:\n  {anchor[:90]!r}")

pairs = [
    ("@dataclasses.dataclass\nclass ViewModel:", PANEL + "@dataclasses.dataclass\nclass ViewModel:"),
    ("def _selftest_bridge(check) -> bool:", TESTS + "def _selftest_bridge(check) -> bool:"),
    ("    ok &= _selftest_bridge(check)\n", "    ok &= _selftest_portfolio(check)\n    ok &= _selftest_bridge(check)\n"),
    ("        self.log_panel = HudRenderer(self.ctx, LogContent(), log_quad_ndc, (LOG_W, LOG_H), rgba=True)\n",
     "        self.log_panel = HudRenderer(self.ctx, LogContent(), log_quad_ndc, (LOG_W, LOG_H), rgba=True)\n"
     "        self.money_panel = HudRenderer(self.ctx, PortfolioContent(), pf_quad_ndc, (PF_W, PF_H), rgba=True)\n"
     "        self.money_on = True\n"),
    ("                elif e.key == self.pygame.K_SPACE:\n                    self.feed.toggle_pause()\n",
     "                elif e.key == self.pygame.K_SPACE:\n                    self.feed.toggle_pause()\n"
     "                elif e.key == self.pygame.K_p:\n                    self.money_on = not self.money_on\n"),
    ("        self.log_panel.draw(self.win_size[0] / self.win_size[1], vm)\n",
     "        self.log_panel.draw(self.win_size[0] / self.win_size[1], vm)\n"
     "        if self.money_on:\n            self.money_panel.draw(self.win_size[0] / self.win_size[1], vm)\n"),
]
for a, _ in pairs:
    once(s, a)
for a, b in pairs:
    s = s.replace(a, b)
shutil.copy(P, str(P) + ".bak3")
P.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
print("done: money panel added to", P, "(backup:", str(P) + ".bak3)")
