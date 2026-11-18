import pandas as pd
import pytest

from experiments.fetch_sectors import sectors_from_html
from tournament.labels import load_sectors

HTML = """<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters Location</th>
<th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Hardware</td><td>Cupertino</td><td>1982-11-30</td><td>0000320193</td><td>1977</td></tr>
<tr><td>XOM</td><td>Exxon</td><td>Energy</td><td>Oil</td><td>Spring</td><td>1957-03-04</td><td>0000034088</td><td>1999</td></tr></table>"""


def test_sectors_from_html_and_loader(tmp_path):
    df = sectors_from_html(HTML)
    assert list(df.symbol) == ["AAPL", "XOM"] and list(df.sector) == ["Information Technology", "Energy"]
    df.to_csv(tmp_path / "sectors.csv", index=False)
    assert load_sectors(tmp_path / "sectors.csv") == {"AAPL": "Information Technology", "XOM": "Energy"}


def test_loader_explains_a_constituents_file_without_sectors(tmp_path):
    pd.DataFrame({"symbol": ["A"], "security": ["x"], "date_added": ["2000-01-01"], "cik": [1]}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(SystemExit) as e:
        load_sectors(tmp_path / "c.csv")
    assert "fetch_sectors" in str(e.value)
