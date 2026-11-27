import datetime as dt

import pandas as pd
import pytest

from broker import live_universe as lu
from env import universe_us as uni


def main_page(symbols):
    head = "<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters</th>" \
           "<th>Date added</th><th>CIK</th><th>Founded</th></tr>"
    rows = "".join(f"<tr><td>{s}</td><td>{s} Inc</td><td>Tech</td><td>Software</td><td>Here</td><td>2001-01-0{1 + i % 9}</td>"
                   f"<td>{1000 + i:010d}</td><td>1990</td></tr>" for i, s in enumerate(symbols))
    return f"<html><body><table>{head}{rows}</table></body></html>"


def history_page(changes):
    head = "<tr><th>Effective Date</th><th colspan=2>Added</th><th colspan=2>Removed</th><th>Reason</th></tr>" \
           "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
    rows = "".join(f"<tr><td>{d}</td><td>{a}</td><td>{a} Co</td><td>{r}</td><td>{r} Co</td><td>why</td></tr>" for d, a, r in changes)
    return f"<html><body><table>{head}{rows}</table></body></html>"


SYMS = [f"S{i:03d}" for i in range(502)] + ["BRK.B"]


def fake_web(symbols=SYMS, changes=(("2026-10-30", "S000", "OLD1"), ("2026-11-20", "S001", "OLD2")), history=True):
    pages = {uni.WIKI_URL: main_page(symbols), uni.HIST_URLS[0]: history_page(changes) if history else "<html></html>",
             uni.HIST_URLS[1]: "<html></html>"}
    seen = []

    def fetch(url, ua):
        seen.append(ua)
        return pages[url]
    return fetch, seen


def research(tmp_path, members):
    d = tmp_path / "research"
    d.mkdir()
    pd.DataFrame(dict(symbol=members, security="x", date_added="2001-01-01", cik=1)).to_csv(d / "constituents.csv", index=False)
    pd.DataFrame(dict(date=["2020-01-02"], added=["S002"], removed=["OLD0"])).to_csv(d / "changes.csv", index=False)
    (d / "cik_overrides.csv").write_text("ticker,cik\nOLD1,42\n")
    return d


def test_refresh_writes_a_fresh_list_and_reports_what_changed(tmp_path):
    out = tmp_path / "live"
    res_dir = research(tmp_path, [s.replace(".", "-") for s in SYMS if s != "S001"] + ["OLD2"])
    fetch, seen = fake_web()
    r = lu.refresh_membership(out, res_dir, fetch=fetch, now=dt.datetime(2026, 11, 27, 14, 0), log=lambda m: None)
    assert r["members"] == 503 and r["newest_change"] == "2026-11-20"
    assert r["added_since_research"] == ["S001"] and r["removed_since_research"] == ["OLD2"]
    iv = pd.read_csv(out / "intervals.csv")
    assert iv.loc[iv.symbol == "S001", "start"].item() == "2026-11-20" and iv.loc[iv.symbol == "OLD2", "end"].item() == "2026-11-20"
    assert "BRK-B" in set(pd.read_csv(out / "constituents.csv").symbol)                 # our ticker form
    assert (out / "cik_overrides.csv").read_text().startswith("ticker,cik")
    assert lu.fetched_at(out) == dt.datetime(2026, 11, 27, 14, 0) and lu.universe_dir(out) == out
    assert all("@" not in ua for ua in seen)                                            # no personal contact data sent


def test_a_suspicious_page_writes_nothing(tmp_path):
    out = tmp_path / "live"
    fetch, _ = fake_web(symbols=SYMS[:100])
    with pytest.raises(ValueError, match="100 constituents"):
        lu.refresh_membership(out, research(tmp_path, SYMS), fetch=fetch, log=lambda m: None)
    assert not out.exists() and lu.universe_dir(out) is None


def test_missing_change_log_falls_back_to_the_research_one(tmp_path):
    out, msgs = tmp_path / "live", []
    fetch, _ = fake_web(history=False)
    r = lu.refresh_membership(out, research(tmp_path, SYMS), fetch=fetch, log=msgs.append)
    assert r["changes"] == 1 and any("research change log" in m for m in msgs)


def test_weekly_staleness(tmp_path):
    out = tmp_path / "live"
    assert lu.is_stale(out)                                                             # never fetched
    fetch, _ = fake_web()
    lu.refresh_membership(out, research(tmp_path, SYMS), fetch=fetch, now=dt.datetime(2026, 11, 27), log=lambda m: None)
    assert not lu.is_stale(out, now=dt.datetime(2026, 12, 3))
    assert lu.is_stale(out, now=dt.datetime(2026, 12, 4))                               # 7 days later
