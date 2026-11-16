import pandas as pd

from experiments.resolve_ciks import propose, apply_overrides

SEC = {"0": {"cik_str": 24545, "ticker": "TAP", "title": "MOLSON COORS BEVERAGE CO"},
       "1": {"cik_str": 99, "ticker": "OLD", "title": "SOME OTHER COMPANY"},
       "2": {"cik_str": 7, "ticker": "BRK-B", "title": "BERKSHIRE"}}


def test_propose_matches_recent_only_and_skips_non_sec(tmp_path):
    gaps = pd.DataFrame({"symbol": ["TAP", "OLD", "DEAD", "BRK.B", "HAS"], "years": [16.9, 5.0, 8.0, 3.0, 4.0],
                         "status": ["no SEC CIK (dropped before prices were tried)"] * 4 + ["Yahoo has no prices"]})
    iv = pd.DataFrame({"symbol": ["TAP", "OLD", "DEAD", "BRK.B", "HAS"], "start": "2005-01-01",
                       "end": ["2025-06-01", "2009-01-01", "2024-01-01", "2024-06-01", "2024-01-01"]})
    p = propose(gaps, iv, SEC, "2026-11-16", max_age=6)
    assert list(p.ticker) == ["TAP", "BRK.B"]                    # OLD left 17 years ago (ticker may be reused), DEAD not in SEC file
    assert p.set_index("ticker").loc["TAP", "cik"] == 24545
    assert set(propose(gaps, iv, SEC, "2026-11-16", max_age=None).ticker) == {"TAP", "OLD", "BRK.B"}


def test_apply_overrides_appends_without_duplicates(tmp_path):
    f = tmp_path / "cik_overrides.csv"
    pd.DataFrame({"ticker": ["AAA"], "cik": [1]}).to_csv(f, index=False)
    new = pd.DataFrame({"ticker": ["AAA", "TAP"], "cik": [1, 24545]})
    assert apply_overrides(f, new) == 1 and apply_overrides(f, new) == 0
    assert list(pd.read_csv(f).ticker) == ["AAA", "TAP"]
