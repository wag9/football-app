# Premier League Match Predictor
Live demo: https://wag9.github.io/football-app/

> **Note:** the API runs on a free hosting tier, so the first request after a quiet period can take 30+ seconds while the server wakes up.

<img src="docs/screenshot.png" alt="Front page" width="600">

Elo-based Home/Draw/Away probability model for the top five European leagues,
with a FastAPI backend, a plain HTML/JS frontend, and a public log
of locked-in predictions.

## Results (walk-forward validation, test seasons 2023/24 to 2025/26)

| League | Baseline | Elo only | Elo + shots form | Bookmaker | Gap to bookmaker |
|---|---|---|---|---|---|
| Premier League | 1.0736 | 0.9831 | **0.9776** | 0.9660 | 0.012 |
| La Liga | 1.0654 | 0.9727 | **0.9693** | 0.9555 | 0.014 |
| Bundesliga | 1.0804 | 0.9911 | **0.9856** | 0.9634 | 0.022 |
| Serie A | 1.0915 | 0.9916 | **0.9887** | 0.9667 | 0.022 |
| Ligue 1 | 1.0751 | 1.0080 | **1.0037** | 0.9818 | 0.022 |

Avg log loss over 3 walk-forward folds (test seasons 2023/24 to 2025/26), lower is better.
Same config in every league (K=30, home advantage 75, promoted-team start 1400), no per-league tuning.
The shots-on-target feature improved 14 of 15 folds.

## Approach
- Elo ratings updated match by match; the rating *before* each match is used as the feature (no leakage).
- Logistic regression on the Elo difference.
- Walk-forward validation by season, never random splits.
- Team pages: Elo rating and rank, form, season results, Elo history chart, and logged predictions per team.
- Elo + rolling shots-on-target net diff (walk-forward log loss 0.9776 vs 0.9831 Elo only)
- Feature: rolling-10 mean of (shots on target for minus against), shifted by one match (no leakage).
- One Elo state and one prediction log per league.
- Match preview page: model probabilities, Elo, recent form, head-to-head and shots form.

## What did not help (tested, flat or worse)
Longer history, mean reversion between seasons, margin-of-victory Elo,
extra rolling-form features, a draw-specific feature.
Full experiment log: [results.md](results.md)
Poisson attack/defence model with time decay (avg log loss 0.9929 vs Elo 0.9831).
Total shots and corners on top of shots-on-target (0.9774 / 0.9791 vs 0.9776).

## Live prediction log
Predictions are written to `data/predictions_log.csv` before kickoff and
never overwritten. Results are compared against the bookmaker.

## Weekly workflow
1. `python weekly.py` downloads the fixtures and the current season's results for all five leagues, rebuilds the match tables, runs the data checks (`audit_data.py`), scores earlier predictions (`check.py`) and logs predictions for upcoming fixtures (`upcoming.py`). Use `python weekly.py E0 SP1` for selected leagues and `--no-download` to reuse the files already in `data/`.
2. Read the summary table at the end. If it says `NOTHING UPDATED` or `PARTIAL`, do not commit blindly. A league whose data fails the checks gets no new predictions logged.
3. Look at `git status` and `git diff --stat`, then `git add data/`, commit and push before the next kickoff. The host redeploys and retrains automatically.
4. Every August (new season): change `CURRENT_SEASON` in `weekly.py` and the season range in `combine.py` and `combine_league.py`.

## Limitations
- The model trails bookmaker odds, which include lineups and injuries.
- In every pairing of current teams in all five leagues (October 2026) the draw probability stays below 31% (highest value per league: 27-31%) and is never the most likely outcome, so draws are never the top pick.
- Pooled over five leagues (5,256 test matches), the model over-predicts home wins by about 2.6 points and under-predicts away wins by about 2.4 points. Draw rates are well calibrated.
- Model-vs-bookmaker gap is about 0.012 (Premier League) and 0.014 (La Liga).
- The model trails bookmakers by about 0.012-0.014 in the Premier League and La Liga and about 0.022 in the other three leagues.

## Run locally
```
pip install -r requirements.txt          # enough to run the API
pip install -r requirements-dev.txt      # also needed for weekly.py (requests) and the Poisson experiment (scipy)
```
Start the API from the repository root, because it reads `data/` with relative paths:
```
uvicorn api:app --reload
```
In a second terminal, serve the pages:
```
python -m http.server 3000
```
Then open http://localhost:3000. Do not open `index.html` by double-clicking it: a page loaded from `file://` is not an allowed CORS origin, so the browser blocks its calls to the API. The allowed origins are `https://wag9.github.io`, `http://localhost:3000` and `http://127.0.0.1:3000`. On localhost the pages talk to the API at `http://127.0.0.1:8000`.

## Tests
```
python -m unittest
```
