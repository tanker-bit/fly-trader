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
| 9 | `accounts/order_plan.py` | **new, V4c**: game account + scores -> one hold-policy day on a shadow account -> orders with reasons | committed on `lab-work`, tested |
| 10 | `broker/live_prices.py` | **new, V4d**: Yahoo refresh into a separate `data_live/prices` folder, overlaid on the research caches; drops today's unfinished bar | committed on `lab-work`, tested; **real refresh ran on the lab PC** (506 of 506 ok) |
| 11 | `broker/game_portfolio.py` | **new, V4d**: READ-ONLY game account (live through the bridge, snapshot file, or fresh) | committed on `lab-work`; converters tested; **live read not run** (bridge + login are at home) |
| 12 | `broker/daily_run.py` | **new, V4d**: the daily dry-run command and `--accept` | committed on `lab-work`, tested end to end on the local caches |
| 13 | `tests/test_order_plan.py`, `tests/test_daily_run.py` | **new**: 11 + 9 tests | committed on `lab-work`, tested |
| 14 | `.gitignore` | added `orders/`, `data_live/`, `game_snapshot*.json` (the user's game account and order lists) | committed on `lab-work` |
| 15 | `broker/trade_ui.py` | **new, V5**: places a day's orders on the Investopedia website in the bot's own visible Edge window (Playwright); submits by default | committed on `lab-work`; tested on a local stand-in page; **preview-only run on the real site: 24 of 26 previewed, 2 skipped by the price guard, 0 errors** |
| 16 | `tests/test_trade_ui.py`, `tests/fake_trade_page.html` | **new**: 11 tests (5 need Playwright + Edge; skipped otherwise) | committed on `lab-work`, tested |

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

### V4c details: `accounts/order_plan.py`

- `plan_day(prices, scores, holdings, params=MIX, state=None, buyable=None)`: seeds a shadow `Account` (no commission, as in the game) with the game's cash, shares and purchase prices; restores the policy memory (V4a); runs **one** `HoldPolicy` day (stops and the 30% ETF floor always; refill / full rebalance when `mode_today` says so); returns `Plan` with `orders` (sells first), the new `state`, before/after snapshots and notes.
- Every order carries a reason: `new holding: rank N`, `short-term sleeve: rank N`, `ETF core: top up ...`, `stop-loss`, `trailing stop`, `short-term sleeve stop/exit`, `rank fell: rank N (sell when worse than L; held D trading days)`, `trim ...`. Stocks the buy guards pass over are listed with the reason (falling 30%+ over 63 days, a one-day move over 35%, cooldown).
- Ranks are among the buyable stocks (`buyable` = what the game offers); held names can always be sold. Holdings the rules cannot trade (e.g. an inverse ETF) or without any price are reported and left alone. No saved state + existing holdings -> they count as bought today (noted).
- `plan_text`, `orders_frame` (adds `game_symbol`, e.g. `BRK.B`), `apply_fills` (what-if / tests).

### V4d details: the daily dry run

```
python -m broker.daily_run                                   (fresh $100,000 account, local prices as they are)
python -m broker.daily_run --refresh-prices                  (download fresh Yahoo prices first, into data_live\prices)
python -m broker.daily_run --portfolio game_snapshot.json    (plan against a saved copy of the game account)
python -m broker.daily_run --portfolio live --lib-dir D:\github\investopedia_simulator_api --save-snapshot game_snapshot.json
python -m broker.daily_run --accept 2026-11-06               (after placing that day's orders: make its memory the real one)
```
- Writes `orders\<date>\orders.csv, plan.txt, scores.csv, run.json, state_after.json`. The real policy memory `orders\policy_state.json` changes **only** with `--accept`, so dry runs never make the next day think orders happened.
- Prices for the planner are adjusted closes, with the last row replaced by the real close of the day (orders are sized with it; stops compare it with real purchase prices).
- Warns when the price date is more than 4 days old; reports skipped incomplete dates, never-checked stocks, guard skips.
- `--refresh-prices` downloads members + core ETFs + held names into `data_live/prices` only; `data_fundamentals` is **never rewritten** (the exactness tests depend on it).
- `--portfolio live` uses the bridge read-only (`broker/session_guard.py`), no trade class is imported, and the working folder is restored afterwards. Snapshot files carry no game ids.
- Lab run (2026-11-27, fresh account, full committee, no refresh): data date 2026-11-06 (9 thinner newer dates skipped), full rebalance, 26 buys (20 stocks, 3 sleeve names, VTI/RSP/QQQ), cash left 2,829, ETF core 34.5%; 12 high-ranked names passed over by the falling-knife guard. Tests: **317 passed**.

**Real price refresh on the lab PC (2026-11-27):** `python -m broker.daily_run --refresh-prices` downloaded 506 of 506 symbols (S&P 500 members, VTI/RSP/QQQ) in about 3 minutes into `data_live/prices` (not committed; git-ignored).
- **Bug found and fixed:** the first refresh ran at 12:47 New York time while the market was open (a half day). Yahoo returned a partial bar for 2026-11-27 (AAPL at about a quarter of a normal day's volume), and the plan was built on intraday prices. `refresh_one` now drops a bar dated today until 16:15 New York time (`CLOSED_AFTER`; half days are simply treated as unfinished until then). Test added; the flawed `orders\2026-11-27` output was deleted and the refresh rerun.
- Corrected dry run: data date **2026-11-25** (Thanksgiving 11-26 closed), 499 of 503 members scored, full rebalance, 26 buys, cash left 2,565, ETF core 34.2%. `SNDK`'s file ends 2026-11-19, so it is scored on its last known price; the run now names such stocks in its notes and `run.json` (`no_price_today`).

Open points for V4d (not verified on the lab PC):
- **Live game read** (`--portfolio live`) never ran: needs the bridge and login at home. Check that `purchase_price` from the library is a **per-share** price (the planner flags odd ratios); the stop-loss depends on it.
- **Availability**: `data_us/game_availability.csv` has only the 50-stock test run (all yes). 493 of 500 scored stocks are "unknown" and allowed by default; run the full V3 availability check or use `--offered-only`. The core ETFs' availability in the game is also unchecked.
- **Membership and SEC data are not refreshed**: S&P 500 changes after `universe/intervals.csv` was built, and filings after the SEC cache, are missing. Refreshing SEC needs a User-Agent with a name and email (ask the user first).
- The game uses market orders at the next open with a 20-minute delay; the plan's prices are estimates.

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
7. `python -m pytest -q tests` from the project root (274 passed before the lab session; expect 329: 274 + 10 V4a + 6 V4b-1 + 8 V4b-2 + 11 V4c + 9 V4d + 11 V5, as on the lab PC; install Playwright first, `pip install playwright`, or 5 V5 tests are skipped).
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
- `accounts/order_plan.py`
- `tests/test_order_plan.py`
- `broker/live_prices.py`
- `broker/game_portfolio.py`
- `broker/daily_run.py`
- `tests/test_daily_run.py`
- `broker/trade_ui.py`
- `tests/test_trade_ui.py`
- `tests/fake_trade_page.html`

(These are the only files the lab session changed. Every other entry in the lab copy's `git status` came from home unchanged.)

## 4. Notes for the AI at home

- Update `PROJECT_HANDOVER_CURRENT.md` once this is merged: section 10 (`.gitignore` gaps), section 14 item 7, section 15 "Risk" line and section 17 task 2 are now partly or fully done. Task 2's second half (copy `token_keeper.mjs` into `broker/` without secrets) is **not done**.
- **V4 is built and tested in the lab** (V4a state, V4b-1 scoring, V4b-2 features, V4c planner, V4d daily dry run); see section 2. The universe is the S&P 500 (what the leagues trained on), buying limited to stocks the game is not known to refuse. The user later said in the lab: no need to work in small parts there, since the AI can run Python and the tests itself; the hard rules still apply.
- **First things to do at home with V4:** (1) `python -m broker.daily_run --refresh-prices` (best after 16:15 New York time; earlier runs use the previous day); (2) with the bridge running, `python -m broker.daily_run --refresh-prices --portfolio live --lib-dir D:\github\investopedia_simulator_api --save-snapshot game_snapshot.json` and check the account read (cash, positions, per-share purchase prices); (3) the full availability run (V3) or `--offered-only`; (4) only then think about V5.
- **Investopedia Terms of Service read in the lab (2026-11-27)** (handover task 4). The simulator's sign-in page links to the People Inc. brands Terms of Service (`https://www.people.inc/brands-termsofservice`), section 3.3 "Restrictions on Use of Services": (e) no "manual or automated software, devices, or other processes (including ... robots, scrapers ...)" to scrape, harvest or download data from the Services; (f) no use of data from the Services to develop software or train AI; (l) no use "in any way not expressly permitted"; (m) no violating or attempting to violate security features. A bot that logs in, opens the Trade tab, reads prices and company data and enters orders is automated software on the Services; the Cloudflare-avoiding bridge and the scripted availability check are in the same territory. Breaking these terms can end the account ("Any unauthorized use ... immediately terminates the licenses"). The user asked for V5 as a browser bot after this; the lab AI stopped and put the decision to the user (see section 5).
- **The user then decided (lab, 2026-11-27): "i dont really care about investopedias conditions since this is just a research project", asked for V5 as a website bot that submits by default ("i want the ai to be able to do most stuff without my input").** Built as `broker/trade_ui.py` (section 5 below). Kept on purpose: the user's own visible window, a few seconds per step, no disguise tricks, only Buy/Sell Market Day Only.

### V5 details: `broker/trade_ui.py` (website order bot)

```
python -m broker.trade_ui --login                          (once per profile: sign in in the window that opens; the login is kept)
python -m broker.trade_ui --date 2026-11-25                (place orders\2026-11-25\orders.csv: SUBMITS by default)
python -m broker.trade_ui --date 2026-11-25 --preview-only (fill in and preview every order, never submit)
```
- Needs `pip install playwright` (1.65.2 in the lab) and Microsoft Edge; no browser download (`channel="msedge"`). The bot's own Edge profile defaults to `..\data_live\edge_profile` (outside the repository, it holds the live login). At home, run `--login` once.
- Per order (sells first): Trade tab -> type the symbol -> pick the suggestion whose ticker is exactly it -> read the stock panel (bid/ask, day and 52-week range, volume; the page also shows EPS, market cap, P/E, dividend yield, next earnings) -> Buy or Sell -> shares -> Market, Day Only -> PREVIEW ORDER -> checks -> SUBMIT ORDER.
- Checks before every submit (else skipped and logged): the preview shows exactly this symbol, "Buy/Sell at Market" and share count; the live ask (buy) or bid (sell) is within `--max-drift` (5%) of the plan's price; a buy's estimated total fits the buying power shown; no `orders\STOP` file.
- Logs `orders\<date>\trade_log.csv` and screenshots in `orders\<date>\screens\`. If every order was submitted or skipped by a guard (no errors, at least one submitted), the day is accepted automatically (same as `broker.daily_run --accept`).
- Page layout used (data-cy names, seen 2026-11-27): `symbol-search-field`, `symbol-info-form`, `action-select` (Buy / Sell / Short / Buy To Cover), `quantity-input`, `order-select`, `duration-select`, `preview-button`, `preview-order-dialog` (`preview-order-type`, `preview-stock-symbol`, `preview-stock-quantity`, `preview-list-item`), `change-order-button`, `submit-order-button`, `buying-power`, `cash`. Suggestions are `[role=option]`. A late "allow notifications" pop-up (`.pushly-prompt-btn-dismiss`) once swallowed the Preview click; the bot now dismisses it and retries once.
- **Lab results on the real site (2026-11-27, account $100,000 cash, no positions):** preview-only run of the 2026-11-25 plan: 24 of 26 orders filled and previewed correctly, BE (+6.2%) and ON (+6.3%) skipped by the price guard, 0 errors; every symbol, including VTI/RSP/QQQ, exists in the game. **The SUBMIT path has only been tested on the local stand-in page**; what the real site shows after SUBMIT ORDER is not yet known (the bot records whatever message appears and treats error words as a failure).
- Not built yet: reading the holdings table from the Portfolio page (the account had no positions to map). Until then `broker.daily_run` reads the account from a snapshot file or the bridge. It would read `orders\<date>\orders.csv`, need the bridge started with `--allow-orders`, and should begin with one tiny order the user watches; afterwards `--accept` the day.
- Open questions for the user from the lab session: pushed the post-U2 work yet? full availability run finished? terms of service read?
- Still open from section 17: task 1 (commit and push the post-U2 work; safe now that `.gitignore` is fixed), task 3 (cleanup), 3b, 4 (terms of service), 5 (full availability run), 6 (V4 order list) and beyond.
- The user commits and pushes. Follow the working rules in `PROJECT_HANDOVER_CURRENT.md` section 19.
