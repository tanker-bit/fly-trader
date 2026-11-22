"""One-line self-test fix for the new wording. Run from the repo root:  python fix_selftest.py"""
import pathlib
import shutil
import sys

P = pathlib.Path("experiments/visual_dashboard.py")
raw = P.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")
old = '"[FLY BUY] Entered Long Position on" in log_rows(sim.view().log)[0][1]'
new = '"[FLY BUY] Bought" in log_rows(sim.view().log)[0][1]'
if s.count(old) != 1:
    sys.exit("not changed: the old test text was not found exactly once (already fixed?)")
shutil.copy(P, str(P) + ".bak2")
P.write_bytes((s.replace(old, new).replace("\n", "\r\n") if crlf else s.replace(old, new)).encode("utf-8"))
print("done: self-test now expects '[FLY BUY] Bought'")
