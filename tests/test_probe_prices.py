import pandas as pd

from experiments.probe_prices import summarize


def test_summarize_flags_usable_partial_and_empty(tmp_path):
    idx = pd.bdate_range("2008-01-01", "2012-12-31")
    pd.DataFrame({"Adj Close": 1.0}, index=idx).to_pickle(tmp_path / "FULL.pkl")
    pd.DataFrame({"Adj Close": 1.0}, index=idx[-300:]).to_pickle(tmp_path / "PART.pkl")
    (tmp_path / "GONE.empty").write_text("empty")
    iv = pd.DataFrame({"symbol": ["FULL", "PART", "GONE", "NEW"], "start": ["2009-01-01"] * 4, "end": ["2012-01-01"] * 4})
    d = summarize(["FULL", "PART", "GONE", "NEW"], tmp_path, iv).set_index("ticker")
    assert d.loc["FULL", "usable"] and not d.loc["PART", "usable"] and not d.loc["GONE", "usable"]
    assert d.loc["GONE", "yahoo"] == "no history" and d.loc["NEW", "yahoo"] == "not downloaded"
    assert d.loc["PART", "years_covered"] < 0.8 * d.loc["PART", "member_years"]
