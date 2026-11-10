# fly-trader: NEXT_STEPS (v2, 2026-11-09)

Paths relative to `D:\github\fly-trader\fly-trader`. Order reflects the evidence: no edge shown, so prefer cheap honest checks and data quality over new machinery.

## Step 0: push what is not on GitHub yet (do this first, one command at a time)
```
D:
cd D:\github\fly-trader\fly-trader
git status
git fetch origin
git log origin/main --oneline -5
```
Check `git status` lists (untracked or modified) some of: `evalkit`, `reports`, `dashboard`, `dashboard3d`, `experiments\eval_suite.py`, `experiments\visual_dashboard.py`, `control_features.py`, `tournament\league.py`, `tests\test_dashboard*.py`. If it says "nothing to commit", it is already pushed. Otherwise:
```
git add .gitignore env experiments tournament tests evalkit reports dashboard dashboard3d control_features.py
git status
git commit -m "Add eval suite, dashboards, league fixes, handover docs"
git push
```
- If `git add` says a path "did not match any files", that path is somewhere else in your tree: skip it or use `git add .` after confirming `git status` shows no `.env`, `data*`, or `league_data*` entries (those must stay ignored).
- Never `git add .env`. Never `--force`.
- If `git push` is rejected: `git pull --rebase origin main`, then `git push`.
- Also add the handover files: put them in a `handover\` folder and `git add handover`.
- Housekeeping: rotate the neuPrint token (it was pasted into chat); delete stray `D:\github\fly-trader\.venv`.

## CRITICAL: the single best next task
**3D dashboard chunk 1: z-buffer rasterizer** (what I was doing; see PROJECT_HANDOVER.md section 0). New file `dashboard3d\raster.py` (numpy z-buffer triangles, multi-light flat shading, fog, supersampling) + a test + a before/after screenshot against the reference image. Done when: `python -m tests.test_dashboard3d` passes headless and the screenshot visibly shows correct occlusion without painter's-order artifacts. Send it to the user, then continue.

## HIGH (3D dashboard chunks, each sent separately)
2. `dashboard3d\postfx.py`: bloom, scanlines, pixelation, vignette, grain, rain.
3. Scene upgrade (`scene3d.py`): faceted fly, big red faceted eyes, veined wings, keyboard, mug, speaker with cyan rings, neon strips, denser skyline.
4. Exchange-UI monitor texture in `screen.py` (candles, order book, buy/sell buttons) keeping state overlays.
5. ModernGL renderer + `experiments\visual_dashboard.py --renderer gl|soft|2d` with fallback, headless `--selftest`, wired to `DashboardApp`. The user must test this one (moderngl and pygame-ce are installed on their machine, not in my sandbox).
6. Recreate or retrieve `dashboard\state.py`, `dashboard\__init__.py` and their tests from the user's drive so the zip is self-contained.

## MEDIUM (research)
7. **Survivorship-free universe**: add prices for delisted/removed S&P names (biggest known bias).
8. Do not run `run_league --final` until a candidate exists that beat the luck bar on rolling validation; it is a one-time test.
9. Add evaluation of the committee score through `eval_suite` with sector-neutral decile long-short net of costs.
10. Try different information (longer horizons, cross-sectional fundamentals interactions) before more rounds: more rounds of near-identical agents only rank noise.

## LOW (only if an edge survives)
11. `broker\` adapter, paper trading for months, position limits, kill switch.
12. README rewrite from these files; pin dependency versions; RESULTS.md with the honest numbers.

## Verification map
| Task | Done when |
|---|---|
| Step 0 | `git status` clean, `git log origin/main` shows the new commit |
| 1-5 | headless tests pass in sandbox; user confirms the window runs and looks right |
| 7 | universe count includes removed names; eval_suite rerun shows changed IC |
