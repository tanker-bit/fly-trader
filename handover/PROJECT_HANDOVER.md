# fly-trader / STONKFLY: PROJECT HANDOVER (v2, written 2026-11-09)

This REPLACES the 2026-11-05 handover (which stopped at the Stage-0 evolution work). Everything from the league, fundamentals, evaluation suite and dashboards onward is new here. Companion files: FILE_INVENTORY.md, D_DRIVE_GITHUB_INVENTORY.md, NEXT_STEPS.md, PROJECT_FILES.zip (all delivered separately, in that order).

Legend: [VERIFIED-USER] = user ran it and pasted output. [TESTED-SANDBOX] = I ran it in my Linux sandbox (synthetic data or headless). [UNTESTED] = written, never run. [LOST-DETAIL] = I know it happened (from my own running summary) but the exact code is no longer in my sandbox.

**Honesty note about this handover.** My sandbox copy of the newer repo (tournament, evalkit, fundamentals, reports, dashboard state) was wiped when the session was compacted. I do NOT have those files now, so I cannot paste their code and I will not pretend to. The user's D: drive and GitHub are authoritative. What I do have verbatim: the dashboard/dashboard3d files re-attached in chat, the original 37-file zip, and the user's pasted terminal outputs.

---

## 0. WHAT I WAS WORKING ON WHEN THIS HANDOVER WAS REQUESTED (read first)

**Task in flight: make the 3D dashboard look as good as the user's reference image.** Nothing was written yet for this; I had only inspected the environment and the reference image.

- User request (2026-11-09 11:11): "higher definition like that image ... as well rendered as this image above", and "do it in many steps ... break everything down to bite size chunks send them to me and continue on the second and so on."
- Reference image (attached in chat as `b97deafc-image.png`, a copy is in the handover attachments): moody cyberpunk room. Low-poly faceted flat-shaded fly (grey body, big red faceted eyes, translucent grey wings, segmented legs) on a dark desk with cyan neon edge lights; keyboard, mug, speaker with concentric cyan rings; huge monitor showing a Coinbase-style exchange UI (candlestick chart, order book, blue buy button); rainy dark skyline with lit windows; rain streaks, fog, neon strips, scanlines, pixelated retro look, bloom.
- Current renderer (`dashboard3d/render_soft.py`) is a painter's-algorithm PIL renderer, about 1,874 triangles, flat colours, no depth buffer, no post-processing. It is far cruder than the reference.
- **Planned chunks (none implemented):**
  1. `dashboard3d/raster.py`: numpy z-buffer triangle rasterizer, per-face shading from several lights, distance fog, supersampling. Test + before/after screenshot.
  2. `dashboard3d/postfx.py`: bloom, scanlines, pixelation, vignette, grain, rain streaks.
  3. Scene upgrade in `scene3d.py`: faceted fly with big red faceted eyes and veined wings, keyboard, mug, speaker with cyan rings, neon edge strips, denser skyline and windows, dark wet desk.
  4. Exchange-UI monitor texture in `screen.py` (candles, order book, buy/sell buttons), keeping the BUY/SELL/HOLD/SILENT overlays and the research terminal.
  5. ModernGL renderer + `experiments/visual_dashboard.py` options `--renderer gl|soft|2d` with fallback, headless `--selftest`, wired to `DashboardApp` (async polling, replay).
- Constraints: sandbox has numpy 2.5.3, PIL 12.3.0, scipy 1.18.1, 2 CPUs; NO pygame, moderngl or numba; no network installs. The user's machine has pygame-ce 2.5.8, moderngl and numpy. So the ModernGL path can only be tested by the user; everything else is testable headless.
- **Best next action for you:** do chunk 1 (z-buffer rasterizer) in the dashboard3d package, using the files in section 5 below as the base, then send it to the user before starting chunk 2. The user explicitly wants small parts delivered one at a time because of usage limits (they hit the 5-hour limit this session).

---

## 1. Project overview

**Name:** fly-trader (marketing banner "STONKFLY"). **Repo:** https://github.com/tanker-bit/fly-trader, branch `main`. **Local:** `D:\github\fly-trader\fly-trader` (the INNER folder is the repo; the outer `D:\github\fly-trader\.venv` is a stray venv).
**Purpose:** a research project in which a simulated fruit-fly mushroom body (hemibrain connectome: Kenyon cells KC -> mushroom body output neurons MBON, dopamine-gated plasticity) scores and ranks stocks. Long-term hope: paper trade, then real trade. Reality so far: **no statistically detectable edge** (see section 4).
**User's goals (stated):** keep iterating (v1 will be bad, v2..v5 better); a LEAGUE of flies (about 10 kept, rest culled, children bred from winners, weights saved so good flies persist); make it the "most feature-rich project in its niche"; point-in-time fundamentals; honest evaluation; a live visual dashboard.
**Stack:** Python 3.14 on Windows (Command Prompt), numpy, scipy, pandas, pyarrow, gymnasium, neuprint-python, yfinance, python-dotenv, pygame-ce 2.5.8, moderngl. Data: neuPrint (hemibrain), Yahoo Finance prices, SEC EDGAR XBRL fundamentals, Wikipedia S&P 500 membership tables. No database, no deployment.
**Similar project the user pointed to:** https://github.com/gajua/fly-earn-better-than-you (user asked if they are "downloading" it; they were not, I only compared ideas).

## 2. Development history (since the 2026-11-05 handover)

1. **Evolution Stage-0 (synthetic) and its speed problem** were superseded by the league approach. The user asked for "saving best performers" instead of throwaway runs: persistent league, 10 kept.
2. **League** (`tournament/` package + `experiments/run_league.py`): 20 flies at start (keep 10), each round trains through a date and validates on the next window, ranks by validation IC, culls the bottom 10, breeds 10 children (9 inherit parent weights), saves weights per agent. 21-day forward label, 21-day gap between train end and validation (no overlap leakage), annealed learning rate, counterfactual (full-information) dopamine update, committee scoring, per-round controls (ridge, momentum 12-1, reversal, random, untrained fly) on the same window, a "luck bar" (expected best of N noise draws), untouched final holdout of the last 500 dates, run once with `--final`. [VERIFIED-USER: 15 rounds ran, about 600-620 s per round with `--workers 4`, 5,405 labelled dates 2004-04-07 to 2026-09-30, league uses up to 2024-10-02.]
3. **Silent-agent problem:** culled/kept flies that output nothing (IC exactly 0.000) kept appearing and could win ties. Fix: silent-agent guard plus a `kc_drive` floor of 0.9. [LOST-DETAIL code]
4. **Committee scores not saved** in round 8 output; fixed in `run_agent_round`; saved scores live in `league_data_fundamentals/rounds/scores/rXX_<id>.npz` (rounds 8 and up) and are what the dashboard replays.
5. **Diagnostics v3 / v4 / v5** (`diagnose_fly.py`, `v3_counterfactual.py`, `v4_anneal.py`, `fly_panel_v5.py`): fly learning curves; finding was fly ~ untrained random sparse code ~ ridge, all near +0.01 IC.
6. **Universe expansion to about 499 S&P 500 names including removed constituents**, point-in-time membership from Wikipedia. Wikipedia changes-table layout moved, so `parse_changes` + `HIST_URLS` fetch and a stack-based parser were added. `certificate has expired` on Wikipedia was fixed with a certifi-based SSL context (verification stays ON). Survivorship gap REMAINS (delisted names without prices are still missing).
7. **SEC EDGAR point-in-time fundamentals** (`env/fundamentals.py`, `env/data_us.py`, `env/universe_us.py`, `experiments/build_dataset_fund.py`): XBRL facts aligned by FILED date with a 1-day lag, TTM/YoY, 11 features (earnings_yield, book_to_price, fcf_yield, gross_margin, accruals, eps_growth_yoy, quality, days_since_filing, ...). Requires a real User-Agent with a name and contact email, at most 8 requests per second. Tests in `tests/test_fundamentals.py`.
8. **Evaluation suite** (`evalkit/`, `experiments/eval_suite.py`, `reports/`): 33 signals on 120 non-overlapping monthly dates (2013-12-12 to 2023-11-16). First version gave p = 0.000 for nearly everything because a within-date label permutation assumes independent stocks (wrong). Replaced by a **sign-flip test on the monthly IC series**, plus Holm, Benjamini-Hochberg and studentized maxT corrections, power from real IC dispersion, sector-neutralization and decile long-short with costs. The user was told to ignore the earlier report.
9. **Control experiment** (`control_features.py`): random dense features + ridge and random KC code + ridge, to see whether the fly's connectome matters. Result: they match the fly.
10. **2D dashboard** (pygame, 960x720, 10-px "pixel" look; BUY/SELL/HOLD/SILENT animations; research-mode terminal; ticker tape; log) then **3D dashboard** (software renderer; see section 5). Dashboard reads a read-only replay of saved out-of-sample scores, never places trades, labelled "REPLAY".
11. **Housekeeping:** `.gitignore` now has `.env`, `league_data/`, `league_data_fundamentals/`, `data_fundamentals/`, `__pycache__/`. Commits confirmed in pasted output: up to the "tournament + fundamentals" commit (the `git add .gitignore env experiments tournament tests` step); later work is likely NOT pushed (section 8).

**Rejected / superseded:** whole-brain simulation; per-stock brains; raw single-stock daily returns; daily rebalancing (costs); replay-epoch training (memorization); permutation test over stocks (invalid); the old Stage-0 evolution package as the main direction; the earlier painter-style 2D "terminal only" visuals as the final look.

## 3. Architecture (current)

```
neuPrint -> data/mb_*.csv -> connectome/load.py -> sim/lif.py LIFNetwork
Yahoo + SEC EDGAR + Wikipedia -> env/{data_us,fundamentals,universe_us}.py
   -> experiments/build_dataset_fund.py -> data_fundamentals/panel.parquet
panel -> tournament/ (league: train, validate, cull, breed, save weights)
   -> league_data_fundamentals/{agents,rounds,scores,...}
panel + league -> evalkit/ + experiments/eval_suite.py -> reports/ (markdown)
league_data_fundamentals/ --(read-only poll thread)--> dashboard/app.py
   -> dashboard/scene.py (2D logic: Pose, FlyAnimator, View) -> canvas_pygame / canvas_pil
   -> dashboard3d/ (m3d geometry, scene3d, screen texture, hud, render_soft) -> experiments/visual_dashboard.py
```
- **FlyAgent** (unchanged core): sparse random projection of features into KCs, k-winner-take-all (5%), LIF simulation, MBON groups = sell/hold/buy, score = buy rate minus sell rate, three-factor dopamine rule on KC->MBON weights only.
- **Dashboard design rule:** all drawing is written against a tiny canvas interface (rect, ellipse, line, polygon, text, text_width, `alpha=` on rect) so the same scene runs through pygame or PIL. FlyAnimator is a state machine (BUY/SELL 1.5 s, HOLD 1.8 s, SILENT 3.0 s, queue max 12). The polling thread only reads files, so it cannot slow the multiprocessing workers.
- **Security:** neuPrint token only in `.env` (gitignored). TLS verification must stay on. SEC User-Agent: use a generic client name plus a contact the USER supplies; do not put the user's account email in request headers without asking.

## 4. Scientific status (honest)

- League, 15 rounds [VERIFIED-USER]: best single-round IC about +0.12 in round 0 (noise-level: validation se about 0.045), later rounds around +0.00 to +0.02. Last pasted round: best IC +0.020, luck bar for best of 160 agents 0.140 -> NOT above the bar. Controls that round: ridge +0.009, momentum +0.016, reversal -0.052, random -0.026, untrained fly +0.004.
- Eval suite [VERIFIED-USER]: 33 signals, no signal significant after correction (best raw: random dense + ridge +0.016 IC, p 0.083). Fly ~ random KC code ~ ridge.
- **Conclusion: no tradable edge demonstrated.** Highest-value work is better data (survivorship-free universe), not more rounds. Paper trade only if an edge appears. The `--final` holdout has NOT been run (do not run it casually; it is one-time).

## 5. Dashboard files that DO exist verbatim (in PROJECT_FILES.zip under `fly-trader/dashboard`, `dashboard3d`, `experiments`, `tests`)

`dashboard/scene.py`, `canvas_pil.py`, `canvas_pygame.py`, `app.py`; `dashboard3d/__init__.py`, `m3d.py` (matrices, Batch, two_bone_ik), `scene3d.py` (build_frame), `screen.py` (monitor texture 512x320), `hud.py`, `render_soft.py`; `experiments/visual_dashboard.py` (2D launcher, keys Space/Tab/Esc/Q, `--selftest`); `tests/test_dashboard_app.py`, `tests/test_dashboard3d.py`.
**Missing from my copy:** `dashboard/__init__.py` and `dashboard/state.py` (Decision dataclass: when, ticker, action, score, agent, round; snapshot/replay/research_feed/detect_mode/date_ticker_index/load_scores/cik_ticker_map), `tests/test_dashboard_state.py`, `tests/test_dashboard_scene.py`. They exist on the user's drive. `dashboard/app.py` and the tests import them, so a clean checkout of only my zip will not run the dashboard.
**Known limits:** the 3D renderer ([TESTED-SANDBOX] 5 tests passed, headless) is the crude painter's version; HOLD screen price line is an illustrative random walk and is labelled so; the 3D path is not yet wired into `visual_dashboard.py` (`--renderer` option pending); pygame/moderngl paths never ran in my sandbox.

## 6. Environment variables
Only `NEUPRINT_TOKEN` (in `.env`, mandatory for `connectome/fetch_mb.py`). `.env.example` has a placeholder. **The real token was pasted into chat earlier: rotate it at neuprint.janelia.org.** SEC EDGAR needs a `--user-agent "Name email"` CLI argument (not an env var). No other secrets.

## 7. Build / run (Command Prompt, `D:\github\fly-trader\fly-trader`, venv active)
```
pip install -r requirements.txt
python -m experiments.build_dataset_fund --user-agent "Your Name you@example.com"
python -m experiments.run_league --workers 4
python -m experiments.eval_suite --panel data_fundamentals/panel.parquet
python -m experiments.visual_dashboard            (2D pygame dashboard)
python -m experiments.visual_dashboard --selftest (headless check)
python -m tests.test_dashboard3d
```
`--final` on `run_league` is the one-time holdout test. Verified by the user: build_dataset_fund (after SSL fix), run_league (15 rounds), eval_suite. Not verified by the user: the dashboards (2D was written and sandbox-tested; the user installed pygame-ce and moderngl but I never saw them run it).

## 8. GitHub state (uncertain, check first)
Pasted output shows `main` up to date with `origin/main` and then `git add .gitignore env experiments tournament tests` for fundamentals/league code. Since then I created or changed, and the user has not confirmed pushing: `evalkit/`, `reports/`, `experiments/eval_suite.py`, `control_features.py`, `dashboard/`, `dashboard3d/`, `experiments/visual_dashboard.py`, `tests/test_dashboard*.py`, the fixed `tournament/league.py` (committee scores, silent guard). Safe check: `git status`, `git fetch origin`, `git log origin/main --oneline -5`. Commands to push are in NEXT_STEPS.md. Never force-push. Commit message footer for commits made by an AI: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## 9. Working preferences (evidenced in the conversation)
- Windows Command Prompt syntax, one copy-paste command per step; say what output to expect.
- Say exactly which folder each delivered file is saved in ("where am i saving each file" was asked twice).
- Deliver complete files, not patches.
- **Small chunks, sent immediately** ("you used up all the usage for nothing split it into parts and send them to me"; "do it in many steps ... bite size chunks"). Do not start giant builds.
- Honest results; negative findings are accepted ("v1 will be bad ... iterate").
- Ambitious scope ("make this the most feature rich of the niche"), but progress must be visible.
- Commit/push after each step; user asks "give me the code to push to git".
- Never put the neuPrint token in a file.

## 10. Instructions for the next AI
You are taking over an ongoing project. First read this file, then verify the real state (`git status`, `git log`, `dir`), because my sandbox lacks parts of the newer code and the user's drive is authoritative. Preserve working code, avoid rewrites, work in small deliverable chunks, say where each file goes, never claim something works unless it was run (and say where: user machine vs headless sandbox), keep TLS verification on, keep secrets out of files, and stay candid that no edge has been shown. Start with section 0: build the z-buffer rasterizer chunk of the 3D dashboard.

## 11. Completeness audit
Preserved: objectives, history since 11-05, decisions and rejections, architecture, scientific status, env var names, run commands, GitHub uncertainty, preferences, in-flight work. NOT preserved (inaccessible to me now): source of `tournament/`, `evalkit/`, `reports/`, `env/fundamentals.py`, `env/data_us.py`, `env/universe_us.py`, `dashboard/state.py`, most experiment scripts after the 11-05 zip, their test files, exact league numbers beyond the pasted rounds, and the real directory tree of `D:\GitHub`. For those, the user's drive/GitHub is the source; I have not invented them.
