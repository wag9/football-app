# Premier League Match Predictor
Live demo: https://wag9.github.io/football-app/

> **Note:** the API runs on a free hosting tier, so the first request after a quiet period can take 30+ seconds while the server wakes up.

<img src="docs/screenshot.png" alt="Front page" width="600">

Elo-based Home/Draw/Away probability model for the Premier League and La Liga,
with a FastAPI backend, a plain HTML/JS frontend, and a public log
of locked-in predictions.

## Results (walk-forward validation, test seasons 2023/24 to 2025/26)

| Model | Premier League | La Liga |
|---|---|---|
| Uniform guess | 1.0986 | 1.0986 |
| Class-frequency baseline | 1.0736 | 1.0654 |
| Elo only (K=30, home adv=75) | 0.9831 | 0.9727 |
| **Elo + shots-on-target form** | **0.9776** | **0.9693** |
| Bookmaker average odds | 0.9660 | 0.9555 |

Avg log loss over 3 folds, lower is better. The shots feature improved all 3 folds in the
Premier League and replicated on La Liga with the same config and no tuning.

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
- Probabilities in the 0.7-0.8 range were overconfident in calibration checks.
- Model-vs-bookmaker gap is about 0.012 (Premier League) and 0.014 (La Liga).

## Run locally
```
pip install -r requirements.txt
uvicorn api:app --reload
```
Then open `index.html` in your browser.
