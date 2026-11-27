# Lab session handover (2026-11-27)

For the AI assistant and the user at home. Read `PROJECT_HANDOVER_CURRENT.md` first for the project itself; this file covers what was done on the lab PC, the state it was left in, and how to bring it into the home clone.

## 1. Summary

- **Built and tested in the lab:** the whole daily pipeline from research to the website. V4a (policy memory between days), V4b (live features + fly scores, reproducing the training panel and the league's saved scores exactly), V4c (order planner with reasons), V4d (daily dry run + price refresh), V5 (a bot that places orders through the Investopedia Trade tab in a visible Edge window), the **autopilot** (the fly reads its own account on the website, plans and trades a whole day unattended), and the **3D dashboard** showing the fly at work: live ticker searches, quotes, lever pulls, and a 4th monitor with the fly's browser.
- **Real-site runs:** preview only. Account reading, price refresh, planning and the Trade-tab flow all worked on the real site. **No order has been submitted by anyone.** The submit step itself is tested only on a local stand-in page.
- **Tests:** 274 at the start; **356 passed** at the end on the lab PC (plus the dashboard's own `--selftest`).
- **Git:** 20 commits (the last one is this rewrite of the handover) on the local branch **`lab-work`** in `D:\github-copy\fly-trader\fly-trader`, on top of `cec8e1a`. **Nothing was pushed to GitHub.** See section 3.
- **Research reminder:** no tradable edge has been shown (see `PROJECT_HANDOVER_CURRENT.md` section 1). This is a play-money research run.

## 2. Where this work happened

| Item | Value |
|---|---|
| Machine | a public lab PC (not the home PC) |
| Folder | `D:\github-copy\` = a copy of the home `D:\github\` folder, brought in on a hard drive |
| Repo in the copy | `D:\github-copy\fly-trader\fly-trader` |
| Starting point | branch `trading-t1-account` at `cec8e1a` (same as GitHub), with the same 7 modified tracked files and untracked files that `PROJECT_HANDOVER_CURRENT.md` section 15 lists (still uncommitted, deliberately left out of the lab commits) |
| Branch with the lab work | `lab-work`, created from `cec8e1a` |
| AI | Claude Code, working directly on the files and running the tests itself |
| Git on the lab PC | the copy is owned by the home Windows account, so git refused to run ("dubious ownership"). Fixed on the lab PC only with `git config --global --add safe.directory D:/github-copy/fly-trader/fly-trader` |
| Python on the lab PC | the AI's silent install failed twice (the installer never saved `core.msi`; error 0x80070003; probably the AI's command sandbox). The user installed Python 3.15.2 from `cmd` for their user only, not on PATH: `%LOCALAPPDATA%\Programs\Python\Python315`. Environment outside the repo: `D:\github-copy\fly-trader\.venv-lab` with the home versions (numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, pyarrow 26.0.1, gymnasium 1.4.0, neuprint-python 0.6.4, yfinance 1.8.0, pytest 9.2.0, pillow 13.0.0, pygame-ce 2.5.8, moderngl 5.13.0) plus **playwright 1.65.2**. Its `requests` is 2.34.2, newer than the old pin in the home PC's main Python (main handover section 14, item 10) |

### Commits on `lab-work` (oldest first)

| Commit | What |
|---|---|
| `72f22d3` | `.gitignore` gaps fixed; this handover started |
| `fc65ecf` | V4a: save and restore the hold policy's memory between daily runs |
| `14c9611` | V4a: retry the state-file swap on Windows "Access is denied" |
| `8b2b3c2` | V4b-1: pick, load and score a league's trained flies on live rows |
| `43f2ea7` | V4b-2: today's feature rows exactly like the hybrid panel |
| `d24924f` | V4c + V4d: order planner and the daily dry-run command |
| `1f79890` | V4d: drop today's unfinished Yahoo bar; name stale-priced stocks |
| `ab7962a` | Handover: what Investopedia's Terms of Service say about bots |
| `3c12f0d` | V5: website order bot for the Investopedia simulator |
| `b9423b8` | Autopilot: the fly reads its account on the website and trades by itself |
| `7dd8279` | Handover: results of the fly's preview-only autopilot run |
| `d5b2a0d` | Dashboard: watch the fly search tickers and trade on the website |
| `581e67c` | Dashboard 4th monitor: the fly's browser |
| `fe0c264` | Dashboard 4th monitor: legible website view |
| `46cf3d2`, `3bcdb78` | Handover: dashboard command lines (a broken fix, then repaired) |
| `7abdefe` | Dashboard: website panel as a split screen that never covers a monitor |
| `5caf565` | Handover: keep the bot's Edge window visible next to the dashboard |
| `c5d7c2f` | Bot: always reopen Edge on the screen unless `--offscreen` |
| (last) | Handover: rewritten in full with the final state |

24 files changed against `cec8e1a`: `.gitignore` and `experiments/visual_dashboard.py` modified, everything else new (list in section 3, Route B).

## 3. Bringing the work home (or to GitHub)

Nothing is on GitHub. Two ways:

**Push from the lab PC** (creates `lab-work` on the public repo; needs a GitHub sign-in in git on this public PC, so sign out afterwards):
```
cd /d D:\github-copy\fly-trader\fly-trader
git push -u origin lab-work
```

**Route A: take it home and merge** (preferred). Copy all of `D:\github-copy` to the drive, including the hidden `.git` folder in `fly-trader\fly-trader`. The lab commits touch only `.gitignore` (unmodified at home), `experiments/visual_dashboard.py` (unmodified at home, tracked) and new files, so the home clone's own uncommitted and untracked files are not in the way. In `cmd` at home, from `D:\github\fly-trader\fly-trader` (replace `E:` with the drive's letter):
1. `git status` and `git log --oneline -1`: expect `trading-t1-account` at `cec8e1a`, with `.gitignore` and `experiments/visual_dashboard.py` not modified. If the home branch has new commits, the merge is a normal merge; review it. If either file was edited at home, commit or set that aside first.
2. `git remote add lab E:/github-copy/fly-trader/fly-trader`
3. `git fetch lab lab-work`
4. `git log --oneline cec8e1a..lab/lab-work` (the 20 commits) and `git diff --stat HEAD lab/lab-work` (the 24 files)
5. `git merge lab/lab-work`
6. `pip install playwright` in the project environment (Edge is used as installed; no browser download)
7. `python -m pytest -q tests` from the project root: expect **356 passed** (274 + 10 V4a + 6 V4b-1 + 8 V4b-2 + 11 V4c + 9 V4d + 11 V5 + 11 autopilot + 16 dashboard). Without Playwright, 5 browser tests are skipped.
8. `python experiments\visual_dashboard.py --selftest` (expect PASS)
9. When done with the drive: `git remote remove lab`. Then push from home when ready.

**Route B: copy files by hand** (fallback). From `E:\github-copy\fly-trader\fly-trader\` to `D:\github\fly-trader\fly-trader\`: `.gitignore`, `LAB_SESSION_HANDOVER.md`, `run_autopilot.cmd`, `accounts/policy_state.py`, `accounts/order_plan.py`, `tournament/live.py`, `tournament/live_features.py`, `broker/live_prices.py`, `broker/game_portfolio.py`, `broker/daily_run.py`, `broker/trade_ui.py`, `broker/site_account.py`, `broker/autopilot.py`, `broker/live_feed.py`, `experiments/visual_dashboard.py`, `tests/test_policy_state.py`, `tests/test_live_scores.py`, `tests/test_live_features.py`, `tests/test_order_plan.py`, `tests/test_daily_run.py`, `tests/test_trade_ui.py`, `tests/fake_trade_page.html`, `tests/test_autopilot.py`, `tests/test_trade_feed.py`.

**Not in git, but in the copy (and so on the drive):** `orders\` (plans, logs, the policy memory, the dashboard feed), `data_live\prices\` (506 refreshed Yahoo files) inside the repo, and `D:\github-copy\fly-trader\data_live\` outside it (the bot's Edge profile with the **live Investopedia login**, plus page maps and screenshots from mapping the site). All are git-ignored or outside the repo.

## 4. How it fits together

```
python -m broker.autopilot            (one trading day, no person needed; run_autopilot.cmd for a scheduled task)
  1. broker/site_account.py   read the account on the website (Portfolio + Order Status pages)       -> stop if open orders
  2. broker/live_prices.py    refresh Yahoo prices into data_live/prices (today's bar only after 16:15 New York)
  3. tournament/live_features.py   today's 25 features for every S&P 500 member (same code as the training panel)
  4. tournament/live.py       scores from league_v8's 10 kept flies + their committee
  5. accounts/order_plan.py   one hold-policy day (MIX) on a shadow copy of the account -> orders with reasons
     broker/daily_run.py      steps 2-5 as a command; files in orders/<date>/
  6. broker/trade_ui.py       each order through the Trade tab in the bot's Edge window; checks; submit
  7. accounts/policy_state.py accept the day's policy memory when nothing went wrong
  broker/live_feed.py         events + browser pictures for the dashboard (never fails a run)
python experiments\visual_dashboard.py --trade-feed --web-overlay      (watch it in 3D; W = website panel)
```

### V4a: `accounts/policy_state.py` (policy memory between days)
- `HoldPolicy` keeps its memory (`entry`, `peak`, `until` = cooldowns, `tact` = sleeve names, `started`) as day indices into one price frame. `PolicyState` stores it as **dates**; `capture` / `restore` convert; dates before the frame are counted back in weekdays; dict order is kept (the policy sells in that order).
- `mode_today`: `'full'`, `'fill'` or `None`, the backtest's rhythm but counted from the last rebalance, so a missed day catches up.
- `save_state` / `load_state`: JSON, temp file then swap, **retried on Windows "Access is denied"** (a real failure seen in the lab: a scanner holds new files for a moment). A missing file = first day; a damaged or future-dated file raises.
- Proven: a policy rebuilt from the file every morning makes exactly the same trades as one that never stops (full history and a 100-day window).
- Known limitation: dates before the price frame are counted in weekdays, not exchange days (one day off per holiday in that stretch). Keep the daily frame longer than 252 days plus a margin, or pass a holiday list to `np.busday_*`.

### V4b-1: `tournament/live.py` (load and score the flies)
- `pick_flies(store, "kept")` = alive, scored (not silent) in the latest round, weights on disk. For `league_v8`: exactly the 10 KEPT in round 18 (`a0180 a0182 a0184 a0185 a0187 a0191 a0192 a0193 a0195 a0196`). `"best"` = `a0192`. Newborns `a0204`-`a0208` are left out.
- `score_rows`: one column per fly plus `committee` (mean of per-fly z-scores, as the league reports it). Strict input checks (columns, gaps, duplicates, values in [-1, 1]).
- Proven: a round-18 validation day reproduces the league's saved scores **exactly** (difference 0.0). About 1.7 s per fly for about 480 stocks (17 s for the committee). No training, no labels; the holdout is not touched.

### V4b-2: `tournament/live_features.py` (today's features)
- `feature_rows(load_inputs("data_fundamentals"))`: the 25 league features for every S&P 500 member on a day, through the panel's own code (`build_price_panel` + `raw_fundamentals`). Ranked among all members, as in training; filtering to game-offered stocks happens later.
- Findings: (1) the fundamentals must be aligned on dates running to the **end of the price cache** (Yahoo's `Close` is split-adjusted up to the download); cutting at the day broke three market-value features for a 2023 day. (2) The research price cache was only partly refreshed after 2026-11-06 (103 of 606 tickers), so the default day is the newest one with real prices for 95% of members; a thinner day is refused; single missing quotes are kept but named.
- Proven: rebuilt rows equal the stored panel exactly on 2023-04-14 and 2026-11-06, and feed straight into V4b-1 to reproduce the league's scores.

### V4c: `accounts/order_plan.py` (order planner)
- `plan_day(prices, scores, holdings, params=MIX, state, buyable)`: a shadow `Account` (no commission, as in the game) seeded with the real cash, shares and purchase prices; one `HoldPolicy` day; orders sells first, each with a reason (`new holding: rank N`, `short-term sleeve`, `ETF core: top up`, `stop-loss`, `trailing stop`, `rank fell ...`, `trim ...`); stocks passed over by the buy guards are named with the reason.

### V4d: `broker/daily_run.py`, `broker/live_prices.py`, `broker/game_portfolio.py` (daily dry run)
```
python -m broker.daily_run [--refresh-prices] [--portfolio fresh|<snapshot.json>|live --lib-dir ...] [--offered-only]
python -m broker.daily_run --accept <date>          (make that day's state_after.json the real policy memory)
```
- Files in `orders\<date>\`: `orders.csv`, `plan.txt`, `scores.csv`, `run.json`, `state_after.json`. The real memory `orders\policy_state.json` changes only on accept.
- Refresh goes to `data_live/prices` only; `data_fundamentals` is never rewritten (the exactness tests depend on it). Today's bar is dropped before 16:15 New York (a real bug: the first refresh, at 12:47 on a half day, planned on intraday prices).
- Warns on stale prices (over 4 days), names stale-priced stocks (`SNDK` ended 2026-11-19), never-checked stocks and guard skips.
- `--portfolio live` reads through the old bridge (never run in the lab); the autopilot reads the account from the website instead (below).

### V5: `broker/trade_ui.py` (website order bot)
```
python -m broker.trade_ui --login                          (once per profile: sign in in the window that opens)
python -m broker.trade_ui --date <date> [--preview-only] [--offscreen] [--max-drift 0.05]
```
- Playwright driving the installed Edge (`channel="msedge"`), its own profile (default `..\data_live\edge_profile`, outside the repo: it holds the live login).
- Per order, sells first: Trade tab -> type the symbol -> pick the exact suggestion -> read the stock panel (bid/ask, day and 52-week range, volume) -> Buy/Sell -> shares -> Market, Day Only -> PREVIEW ORDER -> checks -> SUBMIT ORDER (default; `--preview-only` stops at the preview). Only Buy and Sell, never Short / Buy To Cover.
- Checks before each submit, else skipped and logged: the preview shows exactly this symbol, side and share count; the live price is within 5% of the plan's; a buy fits the buying power; no `orders\STOP` file.
- Logs `orders\<date>\trade_log.csv` and screenshots `orders\<date>\screens\`; accepts the day automatically when nothing went wrong.
- Page layout (data-cy names, 2026-11-27): `symbol-search-field`, `symbol-info-form`, `action-select`, `quantity-input`, `order-select`, `duration-select`, `preview-button`, `preview-order-dialog` (`preview-order-type`, `preview-stock-symbol`, `preview-stock-quantity`, `preview-list-item`), `change-order-button`, `submit-order-button`, `buying-power`, `cash`; suggestions are `[role=option]`. A late "allow notifications" pop-up (`.pushly-prompt-btn-dismiss`) once swallowed a click; it is dismissed and the click retried.
- **Headless does not work:** a headless Edge got Cloudflare's "Just a moment..." page. Not worked around. The bot always uses a normal window. `--offscreen` places it off the screen; otherwise it is placed on screen (Edge remembers the last position in the profile, so after an `--offscreen` run it reopened off screen until this was fixed in `c5d7c2f`).

### Autopilot: `broker/autopilot.py`, `broker/site_account.py`, `run_autopilot.cmd`
```
python -m broker.autopilot [--preview-only] [--offscreen] [--no-refresh] [--allow-stale] [--offered-only]
run_autopilot.cmd [same options]        (finds ..\.venv-lab or ..\.venv; for a scheduled task)
```
- Reads the account from the website: Portfolio page (`account-value-text`, `buying-power-text`, `cash-text`, the `holdings-table`: Symbol, Description, Current Price, Today's Change, Purchase Price, QTY, Total Value, Total Gain/Loss, Trade Actions) and Order Status (`open-trade-table`, `failed-trade-table`). Cash + holdings must equal the account value within 1% (or $50), else it stops.
- Does not trade when: orders are still open on the website, prices are stale, that price date was already traded, or `orders\STOP` exists. A quiet day (no orders) is accepted so the memory moves forward. Everything goes to `orders\autopilot.log`.

### Dashboard: `experiments/visual_dashboard.py` + `broker/live_feed.py`
```
python experiments\visual_dashboard.py --trade-feed --web-overlay     (live, fullscreen; W toggles the website panel)
python experiments\visual_dashboard.py --trade-replay --web-overlay   (replay the last run)
```
- The user lifted the "do not change the dashboard" rule for this. The 3D scene is unchanged except for the new wall screen; a third feed `TradeFeed` was added next to `LeagueBridge` and `SimFeed`.
- `broker/live_feed.py`: the bot and the autopilot append events to `orders/live_feed.jsonl` (start, phase, account, search, quote, preview, result, done), and `Mirror` saves the bot's browser to `orders/live_screen.jpg` about once a second.
- On screen: the main monitor shows SEARCHING <ticker> and QUOTE DATA (bid, ask, ranges, volume, rank, score); a submitted order plays the BUY/SELL lever and lands on the decision tape; the log and account panel follow the run; with this feed the wording says LIVE GAME / PREVIEW ONLY / REPLAY, SEARCHING TRADE TAB, QUOTE DATA, PREVIEW ORDER <ticker> instead of the league's illustrative texts.
- **4th monitor:** a wall screen floating above the right-hand monitor, facing the camera, drawn after the glow pass, showing the newest browser picture (1024x632, sharpened, more contrast) under `INVESTOPEDIA LIVE hh:mm:ss` (`LAST SEEN` after 120 s). **W** switches to a split screen: the 3D view in the left 58% (framed 1.3x tighter) and a large, fully readable website panel on the right, so nothing covers a monitor. `--web-screen PATH`, `--no-web-screen` exist.
- Framing was checked by rendering into a hidden window (no screen capture). Best viewed fullscreen (the default).

### `.gitignore`
Rewritten and grouped: Python caches; **secrets and logins** (`.env`, `bridge_key.txt`, `browser_profile/`, session/token/credential files, `auth.json`, `probe_out.json`, `game_snapshot*.json`); bulk data (`data/`, `data_fundamentals/`, `data_us/prices/`, `data_us/sec/`, `*.parquet`, `*.npz`, `*.pkl`); run outputs (`results/`, `league_data*/`, `handover/`, root `*.csv`, **`orders/`, `data_live/`**); audit scratch files. Before: about 5,513 entries in `git status`; after: 34. Left as decisions for the user: the three small csvs in `data_us/` stay visible; 75 already-tracked files (about 52 price `.pkl` and the root result csvs) now match the rules but stay tracked until `git rm --cached`.

## 5. Real runs in the lab (all on 2026-11-27, account $100,000 cash, no positions)

| Run | Result |
|---|---|
| Dry run, local prices | data date 2026-11-06 (newer dates too thin), 26 planned buys, ETF core 34.5% |
| First real price refresh (12:47 NY) | 506 of 506 ok, but today's partial bar was used: bug, fixed, refresh rerun |
| Corrected dry run | data date 2026-11-25, 499 of 503 members scored, 26 buys (20 stocks, 3 sleeve, VTI/RSP/QQQ), cash left 2,565, core 34.2% |
| Site mapping (read only) | Trade tab, order flow up to the preview (closed with CHANGE ORDER), Portfolio, Order Status |
| V5 preview-only, 26 orders | 24 previewed, BE (+6.2%) and ON (+6.3%) skipped by the price check, 0 errors; every symbol exists in the game |
| Autopilot preview-only (13:25-13:32 NY) | account read OK, refresh, plan, 23 previewed, BE/NTAP/ON skipped, 0 errors, nothing accepted |
| 3-order demos (`orders/demo`) | used to check the dashboard feed, the 4th monitor, `--offscreen` and the on-screen fix; preview only |

**Never done:** a real SUBMIT on the website; reading an account that has holdings (the holdings-table row parser is tested on stand-in pages only); the bridge-based `--portfolio live`.

## 6. Decisions the user made in the lab

- Work in whole features, not small parts, since the AI can run Python and the tests in the lab. The hard rules still apply (no `--final`, no secrets, user commits/pushes).
- Universe: the S&P 500 (what the leagues trained on); buying limited to stocks the game is not known to refuse.
- **Investopedia Terms of Service** (read in the lab; the sign-in page links to `https://www.people.inc/brands-termsofservice`, section 3.3): (e) no robots, scrapers or other automated software to scrape or download data; (f) no use of site data to develop software or train AI; (l) no use not expressly permitted; (m) no getting around security features; unauthorized use ends the licence. The AI raised this and recommended an official paper-trading API (e.g. Alpaca). The user decided: "i dont really care about investopedias conditions since this is just a research project". Kept anyway: the user's own visible window, a few seconds per step, no disguise tricks, no Cloudflare workarounds.
- Submit by default ("i want the ai to be able to do most stuff without my input"); `--preview-only` as the exception.
- "we have to get our fly to do the order not you": the AI never submits; the fly's autopilot does.
- The dashboard may be changed to show the fly trading; the bot's Edge window should stay visible on the PC as well as on the dashboard.

## 7. Open points and next steps

1. **First real day:** run `python -m broker.autopilot` after 16:15 New York time (fresh prices, a fresh plan), with the dashboard open. Watch the first SUBMIT closely: what the site shows afterwards is unknown; the bot records the message and treats error words as a failure. Check `orders\autopilot.log` and the Portfolio / Order Status pages afterwards.
2. **Old plan:** `orders\2026-11-25\orders.csv` is from 2026-11-25 prices; more of its orders will be skipped by the 5% check every day. Let the autopilot make a new plan rather than trading that one.
3. **Holdings read:** after the first fills, run the autopilot in `--preview-only` once and check `orders\web_account.json` (symbols, quantities, per-share purchase prices; the planner flags odd purchase/current ratios).
4. **Availability:** `data_us/game_availability.csv` has only 50 stocks; every symbol the bot touched so far existed. Run the full V3 check or use `--offered-only` if failures appear.
5. **Not refreshed:** S&P 500 membership (`universe/intervals.csv`) and the SEC cache. Refreshing SEC needs a User-Agent with a name and email: ask the user first.
6. **Scheduling** (not set up): one run per trading day after 16:15 New York, e.g. `schtasks /Create /TN "STONKFLY autopilot" /TR "D:\github\fly-trader\fly-trader\run_autopilot.cmd" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:45` (adjust to the PC's time zone). The PC must be on and the bot's profile signed in.
7. From the main handover: push the post-U2 work (safe now that `.gitignore` is fixed); cleanup (task 3); `token_keeper.mjs` into `broker/` without secrets (task 2, second half); update `PROJECT_HANDOVER_CURRENT.md` sections 10, 14, 15, 17 and 19 with this file's results; decide whether the research continues (task 9). Never run `--final`.

## 8. Before leaving the lab PC (public computer)

- **The bot's Edge profile holds the live Investopedia login:** `D:\github-copy\fly-trader\data_live\edge_profile` (about 175 MB). Take it on the drive if you want to reuse the login at home, then **delete it from the lab PC** (or sign out of Investopedia in that window first: `python -m broker.trade_ui --login` opens it).
- If you pushed from here: sign out of GitHub in git / the browser.
- Optional cleanup: uninstall Python 3.15 (Settings, Apps), delete `D:\github-copy\fly-trader\.venv-lab`, remove the git exception: `git config --global --unset safe.directory D:/github-copy/fly-trader/fly-trader`.
- Privacy note: while checking the dashboard, the AI once captured the whole screen while the user was using the PC; those images were deleted straight away and later checks rendered into a hidden window instead.

## 9. Tests (lab PC, 2026-11-27)

```
D:\github-copy\fly-trader\.venv-lab\Scripts\activate
cd /d D:\github-copy\fly-trader\fly-trader
python -m pytest -q tests                                  (356 passed, about 1.5 minutes)
python experiments\visual_dashboard.py --selftest          (PASS)
```
New test files: `test_policy_state` (10), `test_live_scores` (6), `test_live_features` (8), `test_order_plan` (11), `test_daily_run` (9), `test_trade_ui` (11, uses `fake_trade_page.html`), `test_autopilot` (11), `test_trade_feed` (16). Tests that need the real caches, `league_v8`, Playwright or Edge are skipped where those are missing.
