import pandas as pd

from experiments.universe_coverage import coverage, report


def test_coverage_counts_missing_removed_names(tmp_path):
    u = tmp_path / "universe"
    (tmp_path / "prices").mkdir()
    u.mkdir()
    pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"], "start": ["2005-01-01", "2005-01-01", "2005-01-01", "2006-01-01"],
                  "end": ["", "2012-01-01", "2010-01-01", ""]}).to_csv(u / "intervals.csv", index=False)
    pd.DataFrame({"symbol": ["AAA", "DDD"], "security": ["a", "d"], "date_added": ["2005-01-01", "2006-01-01"],
                  "cik": [1, 4]}).to_csv(u / "constituents.csv", index=False)
    pd.DataFrame({"ticker": ["BBB"], "cik": [2]}).to_csv(u / "cik_overrides.csv", index=False)
    for t in ("AAA", "DDD"):
        (tmp_path / "prices" / f"{t}.pkl").write_bytes(b"x")
    (tmp_path / "prices" / "BBB.empty").write_text("empty")
    per, yr = coverage(tmp_path, today="2020-01-01")
    st = dict(zip(per.symbol, per.status))
    assert st["AAA"] == "has prices" and st["DDD"] == "has prices"
    assert st["BBB"] == "Yahoo has no prices"
    assert st["CCC"].startswith("no SEC CIK")
    y2008 = yr[yr.year == 2008].iloc[0]
    assert y2008.members == 4 and y2008.covered == 2                  # AAA, BBB, CCC, DDD all members; only AAA, DDD have prices
    txt = report(per, yr)
    assert "largest gaps" in txt and "CCC" in txt
