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
| 3 | `accounts/policy_state.py` | **new, V4a**: saves and restores `HoldPolicy`'s memory between daily runs (details below) | committed on `lab-work`, **NOT RUN** |
| 4 | `tests/test_policy_state.py` | **new**: 9 tests for V4a | committed on `lab-work`, **NOT RUN** |

The lab commits hold lab work only. The 7 modified and 26 untracked files from home were deliberately left out (still uncommitted in the copy). Use Route A below.

No existing source code, tests, data, the 3D dashboard or the Investopedia library folder were changed. No secrets were opened. `--final` was not run. No orders.

### No Python on the lab PC: V4a is untested

The lab PC has no Python (only the Microsoft Store placeholder). The official `python-3.15.2-amd64.exe` (signed by the Python Software Foundation) was downloaded with the user's approval, but installing failed twice the same way: the installer never saved its `core.msi` part to disk, then stopped with error 0x80070003 (path not found). That points to a lab policy or security tool; it was not worked around. The partial install was removed with the installer's own uninstall.

So **nothing in V4a has been executed**. The code was checked by reading only: the lab AI and an independent review pass, which recomputed every exact assertion by hand and found no bugs. First job at home, after the merge: `python -m pytest -q tests` (expect 274 + 9 = 283 passed). If a V4a test fails, fix V4a; do not loosen the test without understanding why.

### V4a details: `accounts/policy_state.py`

Why: `HoldPolicy` keeps its memory (`entry`, `peak`, `until` = cooldowns, `tact` = sleeve names, `started`) as integer day indices into one price frame, which only lives for one backtest. The V4 daily runner rebuilds the policy every day on a new price frame.

What it does:
- `PolicyState`: the memory as **dates** (entry and sleeve buy dates, peaks, cooldown days left at the save date, dates of the last full rebalance and last refill), plus a `version`.
- `capture(policy, dates, i, mode, prev)` and `restore(state, policy, dates, i)`: policy <-> state. Dates before the new frame's first day are counted back in weekdays (negative indices), so a short daily price window still gives correct held-day counts. Dict order is kept (the policy sells in that order).
- `mode_today(state, dates, i, params)`: `'full'`, `'fill'` or `None`. Same rhythm as `accounts.sim.run_account` (full every `every`, refill every `fill_every` trading days) but counted from the last rebalance, so a missed day catches up instead of skipping.
- `save_state` / `load_state`: JSON, written to a temp file then swapped in. A missing file = first day (`None`). A damaged file or a state dated after today raises instead of silently starting over (starting over would force a full rebalance and forget stops and cooldowns).

Key tests: a policy rebuilt from the file every morning makes **exactly** the same trades as one that never stops, both with the full price history and with a 100-day rolling window. Others cover cooldowns across frames, the rebalance rhythm, file safety, and the ticker `NA`.

Known limitation (found in review; the tests do not hit it): dates **before** the price frame's first day are counted in weekdays, not exchange trading days, so each market holiday in that stretch makes a held-day or cooldown count one day off. The same happens if the saved date is missing from the frame because of a data gap (it maps to the next trading day). Keep the daily price frame longer than `lt_days` (252) plus a margin so normal holdings stay inside it, or pass an exchange holiday list to `np.busday_count` / `np.busday_offset` (`holidays=`) in `_pos` / `_date_at` if exact counts matter.

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

The lab commits touch only the four files in section 2: `.gitignore` (unmodified at home) and three files that do not exist at home, so the home clone's own uncommitted and untracked files are not in the way and stay as they are.

Run in `cmd` at home, from `D:\github\fly-trader\fly-trader` (replace `E:` with the drive's letter):

1. Check the start state: `git status` and `git log --oneline -1`. Expect branch `trading-t1-account` at `cec8e1a`, with `.gitignore` not listed as modified. If the home branch has new commits, the merge below is a normal merge rather than a fast-forward; review it before continuing. If `.gitignore` was edited at home, commit or set that edit aside first.
2. `git remote add lab E:/github-copy/fly-trader/fly-trader`
3. `git fetch lab lab-work`
4. `git log --oneline cec8e1a..lab/lab-work` to see the lab commit(s), and `git diff --stat HEAD lab/lab-work` to see the files (expect only the four in section 2).
5. `git merge lab/lab-work`
6. `git status --short --untracked-files=all`: the price and SEC data should be gone from the list (about 35 entries left, all code, tests, docs and small csvs).
7. `python -m pytest -q tests` from the project root (274 passed before the lab session; expect 283 with the 9 new V4a tests, which have **never been run**).
8. When done with the drive: `git remote remove lab`.

The merged commit lands on whatever branch is checked out at home (normally `trading-t1-account`). Nothing is pushed; the user pushes when ready.

### Route B: copy files by hand (fallback if git is not wanted)

Copy from `E:\github-copy\fly-trader\fly-trader\` to `D:\github\fly-trader\fly-trader\`, overwriting:
- `.gitignore`
- `LAB_SESSION_HANDOVER.md`
- `accounts/policy_state.py`
- `tests/test_policy_state.py`

(These are the only files the lab session changed. Every other entry in the lab copy's `git status` came from home unchanged.)

## 4. Notes for the AI at home

- Update `PROJECT_HANDOVER_CURRENT.md` once this is merged: section 10 (`.gitignore` gaps), section 14 item 7, section 15 "Risk" line and section 17 task 2 are now partly or fully done. Task 2's second half (copy `token_keeper.mjs` into `broker/` without secrets) is **not done**.
- V4 plan agreed in the lab (each part waits for the user's "go"): **V4a** saved policy state (written, untested); **V4b** today's fly scores from a league folder (not started; open decision: score only S&P 500 stocks the game offers, as the leagues were trained on, or all offered US stocks; the lab AI recommended S&P 500 only); **V4c** order planner: shadow `Account` seeded with the game's real holdings and cash (pass cash 100,000 explicitly), one `HoldPolicy` day with MIX, diff -> order list with reasons; **V4d** `daily_run` command: read-only bridge reads, availability filter, writes `orders/YYYY-MM-DD.csv` and a summary. No ordering code (that is V5).
- Open questions for the user from the lab session: pushed the post-U2 work yet? full availability run finished? terms of service read?
- Still open from section 17: task 1 (commit and push the post-U2 work; safe now that `.gitignore` is fixed), task 3 (cleanup), 3b, 4 (terms of service), 5 (full availability run), 6 (V4 order list) and beyond.
- The user commits and pushes. Follow the working rules in `PROJECT_HANDOVER_CURRENT.md` section 19.
