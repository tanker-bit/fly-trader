# Lab session handover (2026-11-27)

For the AI assistant (and the user) at home. Read `PROJECT_HANDOVER_CURRENT.md` first for the project itself; this file only covers what was done on the lab PC and how to bring it into the home clone.

## 1. Where this work happened

| Item | Value |
|---|---|
| Machine | a lab PC (not the home PC) |
| Folder | `D:\github-copy\` = a copy of the home `D:\github\` folder, brought in on a hard drive |
| Repo in the copy | `D:\github-copy\fly-trader\fly-trader` |
| Starting point | branch `trading-t1-account` at `cec8e1a` (same as GitHub), with the same 7 modified tracked files and untracked files that `PROJECT_HANDOVER_CURRENT.md` section 15 lists |
| New branch | `lab-work`, created from `cec8e1a`; all lab work goes here |
| AI used | Claude Code, working directly on the files (no sandbox, no push) |

The copy's files are owned by the home Windows account, so git on the lab PC refused to run ("dubious ownership"). Fix applied **on the lab PC only**: `git config --global --add safe.directory D:/github-copy/fly-trader/fly-trader`. Nothing about this needs doing at home.

## 2. Changes made

| # | File | Change | Status |
|---|---|---|---|
| 1 | `.gitignore` | Rewritten (details below) | committed on `lab-work` |
| 2 | `LAB_SESSION_HANDOVER.md` | this file | committed on `lab-work` |
| 3 | `accounts/policy_state.py` | **new, V4a**: saves and restores `HoldPolicy`'s memory between daily runs (details below) | committed on `lab-work`, tested |
| 4 | `tests/test_policy_state.py` | **new**: 10 tests for V4a | committed on `lab-work`, tested |
| 5 | `tournament/live.py` | **new, V4b-1**: pick, load and score a league's trained flies on live feature rows (details below) | committed on `lab-work`, tested |
| 6 | `tests/test_live_scores.py` | **new**: 6 tests for V4b-1 (2 use the real `league_v8`, connectome csvs and hybrid panel; skipped where absent) | committed on `lab-work`, tested |
| 7 | `tournament/live_features.py` | **new, V4b-2**: today's feature rows from the local caches, built exactly like `panel_hybrid.parquet` (details below) | committed on `lab-work`, tested |
| 8 | `tests/test_live_features.py` | **new**: 8 tests for V4b-2 (3 use the real `data_fundamentals` caches; skipped where absent) | committed on `lab-work`, tested |

The lab commits hold lab work only. The 7 modified and 26 untracked files from home were deliberately left out (still uncommitted in the copy). Use Route A below.

No existing source code, tests, data, the 3D dashboard or the Investopedia library folder were changed. No secrets were opened. `--final` was not run. No orders.

### Python on the lab PC and test results

The lab PC had no Python. The AI's own silent install of the official `python-3.15.2-amd64.exe` failed twice (the installer never saved its `core.msi` part; error 0x80070003). No lab policy was found to explain it; the likeliest cause is the AI's command sandbox. The user then installed it themselves from `cmd` and it worked:
- Python 3.15.2 for this user only, not on PATH: `%LOCALAPPDATA%\Programs\Python\Python315`
- a separate environment outside the repo: `D:\github-copy\fly-trader\.venv-lab`, with the home PC's package versions (numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, pyarrow 26.0.1, gymnasium 1.4.0, neuprint-python 0.6.4, yfinance 1.8.0, pytest 9.2.0, pillow 13.0.0, pygame-ce 2.5.8, moderngl 5.13.0). Its `requests` is 2.34.2, newer than the old pin stuck in the home PC's main Python (handover section 14, item 10).

**Tests on the lab PC:** the first run gave 281 passed, 2 failed. Both failures were a real Windows bug in V4a: `os.replace` raised `PermissionError: [WinError 5] Access is denied` when swapping in a state file written moments earlier (a scanner or indexer briefly holds new files). Fixed: `save_state` now retries the swap for up to about 10 seconds before giving up, with a new test for it. After the fix: **284 passed** (274 + 10 V4a tests), and the V4a tests passed 5 runs in a row. At home, after the merge, run `python -m pytest -q tests` once to confirm (expect 284).

### V4a details: `accounts/policy_state.py`

Why: `HoldPolicy` keeps its memory (`entry`, `peak`, `until` = cooldowns, `tact` = sleeve names, `started`) as integer day indices into one price frame, which only lives for one backtest. The V4 daily runner rebuilds the policy every day on a new price frame.

What it does:
- `PolicyState`: the memory as **dates** (entry and sleeve buy dates, peaks, cooldown days left at the save date, dates of the last full rebalance and last refill), plus a `version`.
- `capture(policy, dates, i, mode, prev)` and `restore(state, policy, dates, i)`: policy <-> state. Dates before the new frame's first day are counted back in weekdays (negative indices), so a short daily price window still gives correct held-day counts. Dict order is kept (the policy sells in that order).
- `mode_today(state, dates, i, params)`: `'full'`, `'fill'` or `None`. Same rhythm as `accounts.sim.run_account` (full every `every`, refill every `fill_every` trading days) but counted from the last rebalance, so a missed day catches up instead of skipping.
- `save_state` / `load_state`: JSON, written to a temp file then swapped in (the swap is retried briefly on Windows' "Access is denied"). A missing file = first day (`None`). A damaged file or a state dated after today raises instead of silently starting over (starting over would force a full rebalance and forget stops and cooldowns).

Key tests: a policy rebuilt from the file every morning makes **exactly** the same trades as one that never stops, both with the full price history and with a 100-day rolling window. Others cover cooldowns across frames, the rebalance rhythm, file safety, and the ticker `NA`.

Known limitation (found in review; the tests do not hit it): dates **before** the price frame's first day are counted in weekdays, not exchange trading days, so each market holiday in that stretch makes a held-day or cooldown count one day off. The same happens if the saved date is missing from the frame because of a data gap (it maps to the next trading day). Keep the daily price frame longer than `lt_days` (252) plus a margin so normal holdings stay inside it, or pass an exchange holiday list to `np.busday_count` / `np.busday_offset` (`holidays=`) in `_pos` / `_date_at` if exact counts matter.

### V4b-1 details: `tournament/live.py`

- `pick_flies(store, which)`: `"kept"` = living flies scored (not silent) in the league's latest round with weights on disk. For `league_v8` that is exactly the 10 KEPT in round 18 (`a0180 a0182 a0184 a0185 a0187 a0191 a0192 a0193 a0195 a0196`); never-scored newborn children with inherited weights (`a0204`-`a0208`) are left out. `"best"` = top selection score (`a0192`). Or a list of ids.
- `load_flies(store, cfg, recs)`: rebuilds each circuit (connectome csvs in `data/`, paths relative to the project root) and loads the trained weights; refuses missing or wrongly shaped weights.
- `score_rows(flies, X, feats)`: one score column per fly plus `committee` (mean of per-fly z-scores, as the league reports it). Refuses missing columns, gaps, duplicate tickers and values outside [-1, 1] (features must already be cross-sectional ranks). Column and row order of X do not matter.
- Checked on the lab PC: scoring a round-18 validation day through this module reproduces the scores the league saved **exactly** (difference 0.0), and repeat runs are identical. Speed: about 1.7 s per fly for about 480 stocks, so the 10-fly committee takes about 17 s.
- No training, no labels; the holdout is not touched (the kept flies were trained through 2023-01-12; the panel's holdout starts 2024-08-13).

### V4b-2 details: `tournament/live_features.py`

- `load_inputs("data_fundamentals")`: the same local caches the hybrid panel was built from (prices + usable extras, `universe/intervals.csv` membership, CIKs). No downloads.
- `feature_rows(inp, asof=None)`: the 25 league features for every S&P 500 member on `asof`, through the **same code** as the panel (`experiments.build_dataset_price.build_price_panel` + `raw_fundamentals`); nothing re-implemented. Ranking is among all members of the day, as in training; filtering to game-offered stocks must come **after** (V4c).
- Two findings that matter for exactness:
  1. **Splits:** the fundamentals must be aligned on dates running to the **end of the price cache**, not cut at `asof`. Yahoo's `Close` is adjusted for every split up to the download, and `env.fundamentals.split_after` only undoes splits inside the dates it is given. Cutting at `asof` changed `earnings_yield`, `book_to_price` and `fcf_yield` for a 2023 day (max rank error 1.7). Fixed and tested.
  2. **Partly refreshed price cache:** in this copy, prices after 2026-11-06 exist for only 103 of 606 tickers (44 on 2026-11-16). The "last date" would have given 0 rows, and dates 1-3 days after a stock's last price would silently reuse forward-filled stale prices. Now the default `asof` is the newest date on which at least 95% of the members have a real price (here 2026-11-06, 6 newer dates skipped and reported), and an explicitly requested day below 95% is refused. Single missing quotes are kept (as the panel does) but listed in `no_price_today`. **The home cache is probably in the same state**: refresh prices before a live run (V4d).
- Checked: rebuilt rows equal the stored panel **exactly** on 2023-04-14 and on the panel's last day (2026-11-06), the result does not depend on the window length, and rebuilt rows -> `tournament.live` -> fly `a0192` reproduce its saved round-18 scores exactly.
- Live demo on 2026-11-06 (lab PC): 500 of 503 members got rows (`FDXF`, `HONA`, `VYLR` dropped: under 260 days of history), about 9 s to build; the 10-fly committee scored them in about 17 s; mean correlation between the flies' scores 0.46. (The committee's top names are not shown here on purpose: there is no demonstrated edge.)
- Not done: refreshing the caches (prices from Yahoo, S&P 500 membership from Wikipedia, SEC filings) is part of V4d.

Design note for V4c/V4d: save the new state only after the day's orders have really been placed (V5). A V4 dry run must not overwrite the real state file, or the next day would think it already rebalanced.

### `.gitignore` details (handover tasks 2 and part of 1)

Old file: 16 lines with duplicates (`.env`, `league_data/`, `__pycache__/` twice); did not ignore `data_us/` bulk, `*.pkl`, `probe_out.json`, login files or scratch files.

New rules, grouped with comments:
- Python: `__pycache__/`, `*.pyc`, `.venv/`, `.ipynb_checkpoints/`, `.pytest_cache/` (new)
- Secrets and logins (all new except `.env`): `.env`, `bridge_key.txt`, `browser_profile/`, `session_state.json`, `access_token.json`, `credentials.json`, `auth.json`, `probe_out.json`. None of these are in the repo; the rules are a guard for when `token_keeper.mjs` is copied into `broker/`.
- Bulk data: `data/`, `data_fundamentals/`, `data_us/prices/` (new), `data_us/sec/` (new), `*.parquet`, `*.npz`, `*.pkl` (new)
- Run outputs: `results/`, `league_data/`, `league_data_fundamentals/`, `handover/`, `/*.csv` (new: root result csvs)
- Scratch: `/*_out.txt`, `/env_names.txt`, `/local_audit.txt` (new)

Deliberate differences from `PROJECT_HANDOVER_CURRENT.md` task 2:
- Ignored `data_us/prices/` and `data_us/sec/`, **not** all of `data_us/`, so the three small new csvs (`baseline_ic.csv`, `game_availability.csv`, `sectors_us.csv`) stay visible for the user to decide on.
- No `league_v*/` rule: nothing untracked was there, and `*.npz` was already ignored.

Verified on the lab copy with `git status --short --untracked-files=all`: before, about 5,513 entries; after, **34** (the `.gitignore` itself, the 7 modified files, and 26 untracked code, test, doc and small csv files). `git add -A --dry-run` lists the same 34.

Known leftover (not changed): 75 files **already tracked** now match the ignore rules (about 52 `data_us/prices/*.pkl` files and the root result csvs). Ignore rules do not untrack them. Untracking the `.pkl` files (`git rm --cached`) is an open decision for the user.

## 3. Bringing the work home

The work is committed on `lab-work`, so use Route A. Route B is a fallback.

### Route A: merge the `lab-work` branch (preferred)

The lab commits touch only the files in section 2: `.gitignore` (unmodified at home) and new files that do not exist at home, so the home clone's own uncommitted and untracked files are not in the way and stay as they are.

Run in `cmd` at home, from `D:\github\fly-trader\fly-trader` (replace `E:` with the drive's letter):

1. Check the start state: `git status` and `git log --oneline -1`. Expect branch `trading-t1-account` at `cec8e1a`, with `.gitignore` not listed as modified. If the home branch has new commits, the merge below is a normal merge rather than a fast-forward; review it before continuing. If `.gitignore` was edited at home, commit or set that edit aside first.
2. `git remote add lab E:/github-copy/fly-trader/fly-trader`
3. `git fetch lab lab-work`
4. `git log --oneline cec8e1a..lab/lab-work` to see the lab commit(s), and `git diff --stat HEAD lab/lab-work` to see the files (expect only the files in section 2).
5. `git merge lab/lab-work`
6. `git status --short --untracked-files=all`: the price and SEC data should be gone from the list (about 35 entries left, all code, tests, docs and small csvs).
7. `python -m pytest -q tests` from the project root (274 passed before the lab session; expect 298: 274 + 10 V4a + 6 V4b-1 + 8 V4b-2, as on the lab PC).
8. When done with the drive: `git remote remove lab`.

The merged commit lands on whatever branch is checked out at home (normally `trading-t1-account`). Nothing is pushed; the user pushes when ready.

### Route B: copy files by hand (fallback if git is not wanted)

Copy from `E:\github-copy\fly-trader\fly-trader\` to `D:\github\fly-trader\fly-trader\`, overwriting:
- `.gitignore`
- `LAB_SESSION_HANDOVER.md`
- `accounts/policy_state.py`
- `tests/test_policy_state.py`
- `tournament/live.py`
- `tests/test_live_scores.py`
- `tournament/live_features.py`
- `tests/test_live_features.py`

(These are the only files the lab session changed. Every other entry in the lab copy's `git status` came from home unchanged.)

## 4. Notes for the AI at home

- Update `PROJECT_HANDOVER_CURRENT.md` once this is merged: section 10 (`.gitignore` gaps), section 14 item 7, section 15 "Risk" line and section 17 task 2 are now partly or fully done. Task 2's second half (copy `token_keeper.mjs` into `broker/` without secrets) is **not done**.
- V4 plan agreed in the lab (each part waits for the user's "go"): **V4a** saved policy state (done, tested on the lab PC); **V4b** today's fly scores from a league folder (V4b-1 done and tested: load and score flies; V4b-2 done and tested: build today's feature rows from the local caches; open decision: score only S&P 500 stocks the game offers, as the leagues were trained on, or all offered US stocks; the lab AI recommended S&P 500 only); **V4c** order planner: shadow `Account` seeded with the game's real holdings and cash (pass cash 100,000 explicitly), one `HoldPolicy` day with MIX, diff -> order list with reasons; **V4d** `daily_run` command: read-only bridge reads, availability filter, writes `orders/YYYY-MM-DD.csv` and a summary. No ordering code (that is V5).
- Open questions for the user from the lab session: pushed the post-U2 work yet? full availability run finished? terms of service read?
- Still open from section 17: task 1 (commit and push the post-U2 work; safe now that `.gitignore` is fixed), task 3 (cleanup), 3b, 4 (terms of service), 5 (full availability run), 6 (V4 order list) and beyond.
- The user commits and pushes. Follow the working rules in `PROJECT_HANDOVER_CURRENT.md` section 19.
