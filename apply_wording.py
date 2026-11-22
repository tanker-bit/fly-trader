"""Rewrite the dashboard's trade wording for long-only accounts. Run from the repo root:  python apply_wording.py
Edits experiments/visual_dashboard.py in place (keeps a .bak copy). Works with Windows (CRLF) or Unix line endings."""
import pathlib
import shutil
import sys

P = pathlib.Path("experiments/visual_dashboard.py")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")

SPEC_NEW = '''_HOLD_RE = re.compile(r"^\\[FLY HOLD\\].*? (?:on|holding on) (\\S+)\\s*$")
_BUY_RE = re.compile(r"^\\[FLY BUY\\] Entered Long Position on (\\S+)(.*)$")
_SELL_RE = re.compile(r"^\\[FLY SELL\\] Covered Short Position on (\\S+)(.*)$")


def spec_text(text: str) -> str:
    """The accounts are long-only (no shorting, no calls), so the log says what really happens: a BUY is a purchase, a SELL sells
    shares the fly owns, a HOLD keeps a position. Older wordings ('Entered Long Position', 'Covered Short Position',
    'Defending Position ... Maintaining neutral holding on X') coming from your FlyAnimator are rewritten to the same text."""
    m = _BUY_RE.match(text)
    if m:
        return f"[FLY BUY] Bought {m.group(1)}{m.group(2)}"
    m = _SELL_RE.match(text)
    if m:
        return f"[FLY SELL] Sold {m.group(1)} from holdings{m.group(2)}"
    m = _HOLD_RE.match(text)
    return f"[FLY HOLD] Holding {m.group(1)}" if m else text


'''

pairs = [
    ('"BUY": "LONG ORDER {tk}", "SELL": "COVER ORDER {tk}"', '"BUY": "BUY ORDER {tk}", "SELL": "SELL ORDER {tk}"'),
    ('f"[FLY BUY] Entered Long Position on {ticker} - MBON Hz: {score:+.1f}"', 'f"[FLY BUY] Bought {ticker} - MBON Hz: {score:+.1f}"'),
    ('f"[FLY SELL] Covered Short Position on {ticker} - MBON Hz: {score:+.1f}"', 'f"[FLY SELL] Sold {ticker} from holdings - MBON Hz: {score:+.1f}"'),
    ('f"[FLY HOLD] Defending Position on {ticker}"', 'f"[FLY HOLD] Holding {ticker}"'),
    ('("BUY", "LONG ORDER AAPL"), ("SELL", "COVER ORDER AAPL")', '("BUY", "BUY ORDER AAPL"), ("SELL", "SELL ORDER AAPL")'),
    ('format_log("SELL", "ZTS", -2.04)[1] == "[FLY SELL] Covered Short Position on ZTS - MBON Hz: -2.0")',
     'format_log("SELL", "ZTS", -2.04)[1] == "[FLY SELL] Sold ZTS from holdings - MBON Hz: -2.0")'),
    ('t.startswith("[FLY BUY] Entered Long Position on AAPL")', 't.startswith("[FLY BUY] Bought AAPL")'),
    ('''          == "[FLY HOLD] Defending Position on AAPL")
    check("BUY log untouched", spec_text("[FLY BUY] Entered Long Position on X - MBON Hz: +1.0").startswith("[FLY BUY]"))''',
     '''          == "[FLY HOLD] Holding AAPL")
    check("old BUY/SELL wording rewritten to what a long-only account does",
          spec_text("[FLY BUY] Entered Long Position on X - MBON Hz: +1.0") == "[FLY BUY] Bought X - MBON Hz: +1.0"
          and spec_text("[FLY SELL] Covered Short Position on X - MBON Hz: -1.0") == "[FLY SELL] Sold X from holdings - MBON Hz: -1.0"
          and spec_text("[FLY BUY] Bought X - MBON Hz: +1.0") == "[FLY BUY] Bought X - MBON Hz: +1.0")'''),
]
missing = []
for a, b in pairs:
    if s.count(a) != 1:
        missing.append(a[:70])
    else:
        s = s.replace(a, b)
i, j = s.find("_HOLD_RE = re.compile"), s.find("def _smooth")
if i < 0 or j < i:
    missing.append("_HOLD_RE / spec_text block")
else:
    s = s[:i] + SPEC_NEW + s[j:]
if missing:
    sys.exit("NOT CHANGED (these places were not found exactly once, file left as is):\n  " + "\n  ".join(missing))
shutil.copy(P, str(P) + ".bak")
P.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
print("done: wording updated in", P, "(backup:", str(P) + ".bak)")
