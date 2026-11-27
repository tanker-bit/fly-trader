import json
from types import SimpleNamespace

import pytest

import experiments.visual_dashboard as vd
from broker.live_feed import Feed, last_run, read_new


# ---------------------------------------------------------------- the event file
def test_feed_file_is_tailed_line_by_line(tmp_path):
    f = tmp_path / "live_feed.jsonl"
    assert read_new(f, 0) == ([], 0)                                       # no file yet
    feed = Feed(f, run="r1")
    feed("start", mode="preview")
    feed("search", symbol="JBL", side="BUY", shares=13)
    ev, off = read_new(f, 0)
    assert [e["kind"] for e in ev] == ["start", "search"] and ev[1]["run"] == "r1" and ev[1]["symbol"] == "JBL"
    with open(f, "a", encoding="utf-8") as h:
        h.write('{"kind": "quote", "symbol": "JB')                          # a line still being written
    assert read_new(f, off) == ([], off)
    with open(f, "a", encoding="utf-8") as h:
        h.write('L", "bid": 1.0}\n')
    ev, off2 = read_new(f, off)
    assert [e["kind"] for e in ev] == ["quote"] and off2 > off


def test_last_run_is_the_newest_start(tmp_path):
    f = tmp_path / "live_feed.jsonl"
    Feed(f, run="a")("start", mode="trading")
    Feed(f, run="a")("search", symbol="OLD")
    Feed(f, run="b")("start", mode="preview")
    Feed(f, run="b")("search", symbol="NEW")
    assert [e.get("symbol") for e in last_run(f)] == [None, "NEW"]


def test_a_broken_feed_never_breaks_trading(tmp_path):
    Feed(tmp_path / "a_folder_not_a_file" / ".." / "..")("start")          # unwritable path: silently ignored


# ---------------------------------------------------------------- the dashboard playing the events
def play(feed, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        feed.update(dt)


def run_of_events():
    return [dict(kind="start", mode="trading"),
            dict(kind="phase", text="PLANNED 2 ORDERS", asof="2026-11-25"),
            dict(kind="account", equity=100000.0, cash=100000.0, positions=[]),
            dict(kind="search", symbol="JBL", side="BUY", shares=13, rank=3, score=1.21, reason="new holding: rank 3"),
            dict(kind="quote", symbol="JBL", bid=235.02, ask=235.45, day_high=237.97, day_low=231.95, volume="285.163k"),
            dict(kind="preview", symbol="JBL", side="BUY", shares=13, est_price=235.45, est_total=3060.85),
            dict(kind="result", symbol="JBL", side="BUY", shares=13, status="submitted", detail="placed", score=1.21),
            dict(kind="result", symbol="BE", side="BUY", shares=19, status="skipped", detail="price moved +6.2%"),
            dict(kind="done", text="SUBMITTED 1 SKIPPED 1")]


def test_the_fly_searches_reads_the_quote_and_pulls_the_buy_lever(tmp_path):
    feed = vd.TradeFeed(tmp_path / "none.jsonl", start_thread=False)
    for e in run_of_events():
        feed.push(e)
    seen = {}
    for _ in range(600):                                                   # 30 s of frames
        feed.update(0.05)
        v = feed.view()
        seen.setdefault(v.research_status, v)
        if v.state == "BUY":
            seen.setdefault("BUY", v)
    s = seen["SEARCHING JBL"]
    assert s.state == "RESEARCH" and s.ticker == "JBL" and s.source == "trade" and "RANK 3" in s.research_lines
    q = seen["READING JBL QUOTE"]
    assert "BID 235.02" in q.research_lines and "ASK 235.45" in q.research_lines and "VOL 285.163k" in q.research_lines
    assert seen["BUY"].ticker == "JBL"                                      # the BUY animation was played for the submitted order
    v = feed.view()
    assert v.tape == [("JBL", 1.21)] and v.trade_date == "25/11/2026" and v.state == "IDLE"
    texts = [t for _, t in v.log]
    assert "[FLY BUY] Bought 13 JBL (order sent)" in texts and any(t.startswith("[FLY SKIP] BE") for t in texts)
    assert vd.pf_values(v)["has"] and vd.pf_values(v)["equity"] == 100000.0          # the account panel shows the real account


def test_the_lever_moves_during_a_submitted_buy(tmp_path):
    feed = vd.TradeFeed(tmp_path / "none.jsonl", start_thread=False)
    feed.push(dict(kind="result", symbol="AAPL", side="SELL", shares=5, status="submitted", score=-0.4))
    feed.update(0.05)
    peak = 0.0
    for _ in range(40):
        feed.update(0.025)
        peak = max(peak, feed.view().lever_sell)
    assert feed.view().state == "SELL" and peak > 0.5


def test_preview_only_runs_are_labelled_and_never_look_like_trades(tmp_path):
    feed = vd.TradeFeed(tmp_path / "none.jsonl", start_thread=False)
    for e in (dict(kind="start", mode="preview"), dict(kind="result", symbol="JBL", side="BUY", shares=13, status="previewed")):
        feed.push(e)
    play(feed, 1.0)
    v = feed.view()
    assert v.replay_label == "PREVIEW ONLY" and v.state == "HOLD" and v.tape == []
    assert "[FLY PREVIEW ONLY] JBL not submitted" in [t for _, t in v.log]


def test_live_mode_shows_only_new_events_and_replay_shows_the_last_run(tmp_path):
    f = tmp_path / "live_feed.jsonl"
    old = Feed(f, run="old")
    for e in run_of_events():
        old(**e)
    live = vd.TradeFeed(f, start_thread=False)
    live.poll_once()
    play(live, 2.0)
    assert live.events_shown == 0                                          # history is not replayed in live mode
    Feed(f, run="new")("search", symbol="NVDA", side="BUY", shares=1, rank=1, score=2.0)
    live.poll_once()
    play(live, 0.2)
    assert live.view().research_status == "SEARCHING NVDA"
    rep = vd.TradeFeed(f, replay=True, start_thread=False)
    play(rep, 1.0)
    assert rep.events_shown >= 1 and rep.view().replay_label.startswith("REPLAY")


def test_the_screens_render_the_trade_feed():
    feed = vd.TradeFeed("missing.jsonl", start_thread=False)
    for e in run_of_events()[:5]:
        feed.push(e)
    sc, tape, log = vd.ScreenContent(), vd.TapeContent(), vd.LogContent()
    t = 0.0
    for _ in range(200):
        feed.update(0.05)
        t += 0.05
        v = feed.view()
        sc.update(v, t)
        tape.update(v, t)
        log.update(v, t)
    assert sc.draws > 0 and sc.canvas.px.any()


def test_make_feed_picks_the_trade_feed(tmp_path):
    args = vd.parse_args(["--trade-feed", str(tmp_path / "f.jsonl"), "--windowed"])
    feed = vd.make_feed(args)
    try:
        assert isinstance(feed, vd.TradeFeed) and feed.replay is False
    finally:
        feed.close()
    assert vd.parse_args(["--trade-feed"]).trade_feed == "orders/live_feed.jsonl"


# ---------------------------------------------------------------- the bot really emits the events
def test_run_orders_emits_search_quote_preview_result():
    from broker.trade_ui import run_orders

    events = []

    class Site:
        emit = staticmethod(lambda kind, **d: events.append((kind, d)))

        def open_trade(self):
            pass

        def enter(self, o):
            self.emit("search", symbol=o["game_symbol"])
            self.emit("quote", symbol=o["game_symbol"], bid=10.0, ask=10.0)
            return dict(bid=10.0, ask=10.0)

        def preview(self):
            self.emit("preview", symbol="AAA")
            return dict(type="Stock: Buy at Market", symbol="AAA", qty=1, est_price=10.0, est_total=10.0)

        def buying_power(self):
            return 1000.0

        def shot(self, name):
            pass

        def back_out(self):
            pass

        def submit(self):
            return "Your order has been placed"

    order = dict(side="BUY", symbol="AAA", game_symbol="AAA", shares="1", est_price="10.0", score="0.7", reason="x")
    res = run_orders(Site(), [order], log=lambda m: None)
    assert res[0]["status"] == "submitted"
    assert [k for k, _ in events] == ["search", "quote", "preview", "result"] and events[-1][1]["score"] == 0.7


# ---------------------------------------------------------------- the 4th monitor: the fly's browser
def test_web_screen_waits_then_shows_the_newest_picture_with_a_label(tmp_path):
    import os
    import numpy as np
    from PIL import Image
    f = tmp_path / "live_screen.jpg"
    now = [1_000_000.0]
    c = vd.WebContent(f, check_every=0.0, clock=lambda: now[0])
    assert c.update(None, 0.0) and c._state == "waiting"
    assert not c.update(None, 1.0)                                         # nothing new: no redraw
    Image.new("RGB", (1280, 900), (200, 30, 30)).save(f, quality=90)
    os.utime(f, (now[0], now[0]))
    assert c.update(None, 2.0) and c._state == "live"
    mid = c.canvas.px[vd.WEB_H // 2, vd.WEB_W // 2]
    assert mid[0] > 150 and mid[1] < 80                                     # the picture is on the screen
    assert (c.canvas.px[:vd.WEB_LABEL_H] == np.array(vd.COL_SCR_AMBER, np.uint8)).all(axis=2).any()   # LIVE label in amber
    now[0] += vd.WEB_STALE_S + 1
    assert c.update(None, 3.0) and c._state == "stale"                     # an old picture is labelled LAST SEEN
    f.write_bytes(b"not a jpeg")
    os.utime(f, (now[0], now[0]))
    assert not c.update(None, 4.0) and c._state == "stale"                 # a broken file keeps the last good picture


def test_web_screen_path_follows_the_trade_feed(tmp_path):
    assert vd.web_screen_path(vd.parse_args([])) is None
    assert str(vd.web_screen_path(vd.parse_args(["--trade-feed"]))).replace("\\", "/") == "orders/live_screen.jpg"
    assert vd.web_screen_path(vd.parse_args(["--trade-feed", str(tmp_path / "x" / "f.jsonl")])) == tmp_path / "x" / "live_screen.jpg"
    assert vd.web_screen_path(vd.parse_args(["--trade-replay", "--no-web-screen"])) is None
    assert vd.web_screen_path(vd.parse_args(["--web-screen", "pic.jpg"])).name == "pic.jpg"


def test_the_wall_monitor_floats_above_the_desk_and_faces_the_camera():
    import numpy as np
    m = vd.web_monitor_matrix()
    main = vd.monitor_matrix(vd.MAIN_MONITOR)
    assert m[1, 3] > main[1, 3] + vd.MON_BODY[1] * vd.MON_SCALE / 2       # higher than the main monitor's top edge
    normal = m[:3, :3] @ np.array([0.0, 0.0, 1.0])                         # the screen's face
    to_cam = np.array(vd.CAM_EYE) - m[:3, 3]
    assert float(normal @ to_cam) / (np.linalg.norm(normal) * np.linalg.norm(to_cam)) > 0.999      # head-on, not slanted
    assert len(vd.build_wall_monitor().solid) < len(vd.build_monitor().solid)          # no stand


def test_the_flat_website_panel_keeps_the_picture_shape_and_fits_under_the_hud():
    for aspect in (16 / 9, 16 / 10, 21 / 9):
        tl, tr, br, bl = vd.web_overlay_quad(aspect)
        w, h = tr[0] - tl[0], tl[1] - bl[1]
        assert tr[0] <= 1.0 and bl[1] >= -1.0 and tl[1] < 1.0               # inside the window, under the top bar
        px_w, px_h = w * aspect, h                                          # NDC -> same-scale units
        assert abs(px_w / px_h - vd.WEB_W / vd.WEB_H) < 0.02 or bl[1] + 1.0 < 0.05     # picture shape kept unless the height caps it


def test_mirror_writes_the_picture_atomically_and_throttles(tmp_path):
    from broker.live_feed import Mirror
    shots = []

    class Page:
        def screenshot(self, path, type, quality):
            shots.append(path)
            open(path, "wb").write(b"jpeg")

    t = [0.0]
    m = Mirror(tmp_path / "live_screen.jpg", every=0.6)
    m._clock = lambda: t[0]
    assert m(Page()) and (tmp_path / "live_screen.jpg").read_bytes() == b"jpeg" and not (tmp_path / "live_screen.jpg.tmp").exists()
    t[0] = 0.3
    assert not m(Page()) and m(Page(), force=True)                        # throttled unless forced
    assert not Mirror(tmp_path / "x.jpg")(object())                        # a page that cannot screenshot: ignored
