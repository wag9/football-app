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
1. After a matchweek: download the latest `prem2627.csv` from football-data.co.uk into `data/`.
2. `python combine.py` rebuilds `data/all_matches.csv`.
3. `python check.py` scores earlier predictions against actual results.
4. Before the next matchweek: update `data/fixtures.csv`, then run `python upcoming.py` to log new predictions.
5. Commit and push the updated CSVs; the host redeploys and retrains automatically.

## Limitations
- The model trails bookmaker odds, which include lineups and injuries.
- Draw probability never exceeds ~27%, so draws are never the top pick.
- Pooled over five leagues (5,256 test matches), the model over-predicts home wins by about 2.6 points and under-predicts away wins by about 2.4 points. Draw rates are well calibrated.
- Model-vs-bookmaker gap is about 0.012 (Premier League) and 0.014 (La Liga).
- The model trails bookmakers by about 0.012-0.014 in the Premier League and La Liga and about 0.022 in the other three leagues.

## Run locally
```
pip install -r requirements.txt
uvicorn api:app --reload
```
Then open `index.html` in your browser.
