# fly-trader: FILE_INVENTORY (v2, 2026-11-09)

Paths relative to `D:\github\fly-trader\fly-trader`. This extends the 2026-11-05 inventory (still valid for the 37 original files: connectome, sim, learning, env/data|features|portfolio|trading_env|universe, baselines, original experiments). Only NEW or CHANGED files are listed here.

Columns: **In zip** = present verbatim in PROJECT_FILES.zip. **Pushed** = confirmed on GitHub from pasted terminal output ("unconfirmed" = I do not know; check `git status`). "Code unavailable" = I know it exists (from the conversation/user's git output) but its source is no longer in my sandbox; get it from the user's drive.

## Confirmed present on the user's machine (from the user's `git status` paste)
| File | Purpose | In zip | Pushed | Notes |
|---|---|---|---|---|
| `env/data_us.py` | US price download | no (code unavailable) | staged in that paste; later push unconfirmed | |
| `env/fundamentals.py` | SEC EDGAR XBRL point-in-time fundamentals, filed-date alignment, 1-day lag, TTM/YoY | no | as above | Needs real User-Agent, <=8 req/s |
| `env/universe_us.py` | S&P 500 point-in-time membership (Wikipedia, certifi SSL, parse_changes, HIST_URLS) | no | as above | Survivorship gap remains |
| `env/features.py` | modified: 21-day labels + 11 fundamental features | no (zip has older version) | as above | Do not overwrite with the zip copy |
| `experiments/build_dataset_fund.py` | builds `data_fundamentals/panel.parquet` | no | as above | CLI `--user-agent`, `--limit` |
| `experiments/diagnose_fly.py`, `v3_counterfactual.py`, `v4_anneal.py`, `fly_panel_v5.py`, `fund_signal_check.py` | diagnostics v3/v4/v5, fundamentals signal check | no | as above | Run by user |
| `experiments/run_league.py` | league CLI (`--workers`, `--final`) | no | as above | Run by user, 15 rounds |
| `tournament/` | league engine (agents, rounds, cull/breed, controls, luck bar, holdout lock) | no | as above | league.py later FIXED (committee scores, silent guard): fix unconfirmed pushed |
| `tests/test_fundamentals.py`, `tests/test_league.py` | tests | no | as above | |
| `.gitignore` | adds `.env`, `league_data/`, `league_data_fundamentals/`, `data_fundamentals/`, `__pycache__/` | zip has older | yes | |

## Created after that paste (push status unconfirmed; assume NOT pushed)
| File | Purpose | In zip | Notes |
|---|---|---|---|
| `evalkit/` (package) | sign-flip test, Holm/BH/maxT, power, sector-neutralization, decile long-short | no | Code unavailable |
| `experiments/eval_suite.py` | runs 33 signals over 120 monthly dates; writes report | no | Run by user |
| `reports/` | generated markdown reports | no | |
| `control_features.py` | random dense / random KC control features | no | location per user's tree |
| `league_data_fundamentals/` | league state, saved weights, `rounds/scores/rXX_<id>.npz` | no | gitignored data, on user's drive only |
| `dashboard/__init__.py`, `dashboard/state.py` | Decision dataclass, snapshot/replay/research_feed/detect_mode/date_ticker_index/load_scores/cik_ticker_map | NO (missing) | Required by app.py and tests |
| `tests/test_dashboard_state.py`, `tests/test_dashboard_scene.py` | tests | NO (missing) | |

## Dashboard files present verbatim in the zip
| File | Purpose | Status | Tested | Known problems |
|---|---|---|---|---|
| `dashboard/scene.py` | 2D scene logic: constants, Pose, FlyAnimator, View, draw_scene, `_header` | complete | sandbox | `_header` reused by 3D |
| `dashboard/canvas_pil.py` | PIL canvas backend (headless) | complete | sandbox | |
| `dashboard/canvas_pygame.py` | pygame canvas backend | complete | NEVER run (no pygame in sandbox) | |
| `dashboard/app.py` | DashboardApp: async read-only index/poll thread, update, view, next_agent, toggle_pause, replay_label | complete | sandbox | imports dashboard.state (missing in zip) |
| `dashboard3d/__init__.py` | package marker | done | | |
| `dashboard3d/m3d.py` | matrices, look_at, perspective, box/prism/quad meshes, Batch, two_bone_ik | complete | sandbox | |
| `dashboard3d/scene3d.py` | room, skyline, desk, fly poses per state, lever, build_frame | works but crude | sandbox | far below the reference image |
| `dashboard3d/screen.py` | monitor texture 512x320 per state | works | sandbox | needs exchange-UI redesign |
| `dashboard3d/hud.py` | top HUD and bottom log | works | sandbox | |
| `dashboard3d/render_soft.py` | painter's-algorithm PIL renderer, homography monitor warp | works, low quality | sandbox | no z-buffer, no post-FX |
| `experiments/visual_dashboard.py` | pygame launcher for the 2D scene, `--selftest` | complete | selftest in sandbox | not wired to 3D |
| `tests/test_dashboard_app.py`, `tests/test_dashboard3d.py` | tests (3D: 5 passed in sandbox) | complete | sandbox | |

## Not source files
The zip also holds screenshots in `previews/` (`hold3.png`, `sheet3d.png`, `sell3d_final.png`, the reference image) and a `handover/` folder. `.env` is never included.
