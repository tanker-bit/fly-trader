# Start here at home (after the lab session of 2026-11-27)

The lab work was done in a school Claude account whose chat is not available at home. Everything needed is in this repository:
- `PROJECT_HANDOVER_CURRENT.md`: the whole project (sections 1-19), what changed in the lab (section 21).
- `LAB_SESSION_HANDOVER.md`: the full lab record: every module, commands, real runs, decisions, open points.
- this file: the steps, and a message to paste into your AI assistant at home.

## 1. Get the lab work (cmd, home PC)

```
cd /d D:\github\fly-trader\fly-trader
ren PROJECT_HANDOVER_CURRENT.md PROJECT_HANDOVER_CURRENT.home.md
git fetch origin lab-work
git merge origin/lab-work
```
The `ren` is needed because your home copy of `PROJECT_HANDOVER_CURRENT.md` was never committed and `lab-work` now contains the
updated version (it includes everything from the old one). If `git merge` complains about another file, stop and ask your AI with
the exact message.

If you copied `D:\github-copy` from the lab to a drive, also copy these (not in git) into `D:\github\fly-trader\fly-trader\` if
you want them: `orders\` (lab plans and logs; optional), `data_live\prices` and `data_live\universe` (refreshed data; optional, the
autopilot downloads them again). The bot's Edge profile with the Investopedia login was `D:\github-copy\fly-trader\data_live\edge_profile`:
copy it to `D:\github\fly-trader\data_live\edge_profile` to keep the login, or simply sign in again (step 2).

## 2. Set up

```
D:\github\fly-trader\.venv\Scripts\activate
pip install playwright
python -m broker.trade_ui --login
```
`--login` opens the bot's Edge window: sign in to Investopedia there once (email code or Google). Playwright uses the Edge already on
the PC; no browser download.

## 3. Check

```
python -m pytest -q tests
python experiments\visual_dashboard.py --selftest
```
Expect **364 passed** and **PASS**. (274 before the lab; 5 browser tests are skipped without Playwright or Edge.)

## 4. The fly's first real trading day

After **16:15 New York time** on a trading day, in two cmd windows from `D:\github\fly-trader\fly-trader` (after `activate`):
```
python experiments\visual_dashboard.py --trade-feed --web-overlay
python -m broker.autopilot
```
The fly reads its account on the website, refreshes prices and the S&P 500 list, scores, plans one day and **submits its own
orders** (market orders for the next open) in a visible Edge window. Nobody has seen a real submit yet: watch the first ones, then
read `orders\autopilot.log` and check the Portfolio and Order Status pages on the website. Create a file `orders\STOP` to halt it.
To rehearse first without submitting: `python -m broker.autopilot --preview-only`.
The next day: `python -m broker.autopilot --preview-only` once, and check `orders\web_account.json` (holdings, per-share purchase prices).

## 5. Message to paste into your AI assistant at home

> I'm continuing the fly-trader (STONKFLY) project. The previous AI sessions can't be reached, so everything is in the repo at
> `D:\github\fly-trader\fly-trader`. Please read, in this order: `HOME_START.md`, `PROJECT_HANDOVER_CURRENT.md` (sections 1-21,
> especially 19 and 21), and `LAB_SESSION_HANDOVER.md` (sections 1-7). Then tell me in a few lines where the project stands and what
> you suggest next. Hard rules: never run `--final`; never open, print or commit secrets (`.env`, `bridge_key.txt`,
> `browser_profile/`, `data_live/edge_profile`, session and token files, `orders/`); never send my email anywhere without asking;
> you never submit orders yourself (the fly's autopilot does); never claim something works unless a test or a run shows it.
> How I like to work: build whole features, run the tests yourself when you can, tell me in plain words what you changed, and give
> me the exact cmd commands to test each change myself (I use Windows cmd). I commit and push unless I ask you to.

## 6. What is next (short; details in `LAB_SESSION_HANDOVER.md` section 7)

1. The first real trading day (step 4), watched.
2. Check the holdings read the next day.
3. Schedule the autopilot (Windows task with `run_autopilot.cmd`, weekdays after 16:15 New York).
4. Data upkeep: refresh the SEC cache (needs your name and email in the User-Agent: your choice), the full game-availability check
   if orders fail on unknown symbols.
5. Housekeeping from `PROJECT_HANDOVER_CURRENT.md` section 17 (push the older home work, cleanup), and decide whether the research
   continues (no tradable edge has been shown; `--final` stays off until you decide).
