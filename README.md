# Premier League Match Predictor

Elo-based Home/Draw/Away probability model for the Premier League,
with a FastAPI backend, a plain HTML/JS frontend, and a public log
of locked-in predictions.

## Results (walk-forward validation, test seasons 2023/24 to 2025/26)

| Model | Avg log loss |
|---|---|
| Uniform guess | 1.0986 |
| Class-frequency baseline | ~1.08 |
| Rolling-form features (12) | ~1.02 |
| **Elo only (K=30, home adv=75)** | **0.983** |
| Bookmaker average odds | ~0.97 |

## Approach
- Elo ratings updated match by match; the rating *before* each match is used as the feature (no leakage).
- Logistic regression on the Elo difference.
- Walk-forward validation by season, never random splits.

## What did not help (tested, flat or worse)
Longer history, mean reversion between seasons, margin-of-victory Elo,
extra rolling-form features, a draw-specific feature.

## Live prediction log
Predictions are written to `data/predictions_log.csv` before kickoff and
never overwritten. Results are compared against the bookmaker.

## Limitations
- The model trails bookmaker odds, which include lineups and injuries.
- Draw probability never exceeds ~27%, so draws are never the top pick.
- Probabilities in the 0.7-0.8 range were overconfident in calibration checks.

## Run locally
```
pip install -r requirements.txt
uvicorn api:app --reload
```
Then open `index.html` in your browser.
