# Experiment log

Metric: multiclass log loss (lower is better). All Elo experiments use
walk-forward validation: train on everything before a season, test on that season.

## 1. Rolling-form features (single split, test = last 2 seasons)

| Setup | Model | Baseline | Bookmaker |
|---|---|---|---|
| 12 rolling-form features | 1.0240 | 1.0814 | 0.9908 |
| + Elo difference | 1.0065 | 1.0814 | 0.9908 |
| Elo difference only (same rows) | 1.0088 | 1.0814 | 0.9908 |

## 2. Elo tuning (data 2019+, 3 folds: 2023/24, 2024/25, 2025/26)

| Experiment | Values tried (avg log loss) | Outcome |
|---|---|---|
| K factor | 10: 0.9915, 20: 0.9816, 30: 0.9803, 40: 0.9816 | K=10 worse; 20-40 flat |
| Home advantage (Elo points) | 0: 0.9807, 40: 0.9805, 75: 0.9803, 110: 0.9802 | No effect; the regression intercept absorbs it |
| Season-start mean reversion | 1.0: 0.9803, 0.9: 0.9813, 0.8: 0.9828, 0.6: 0.9868 | Worse; team quality persists across seasons |
| Promoted-team start rating | 1500: 0.9803, 1450: 0.9799, 1400: 0.9798, 1350: 0.9798 | Within noise; 1400 kept as a sensible default |

## 3. Extra features on top of Elo (data 2019+)

| Features | Avg log loss |
|---|---|
| Elo only | 0.9816 |
| Elo + 4 goals features | 0.9801 |
| Elo + 12 rolling features | 0.9793 |

Gains of 0.0015-0.0023 are within noise, so the simpler Elo-only model was kept.

## 4. Later experiments (data 2003+, final fold setup)

| Experiment | Result | Outcome |
|---|---|---|
| Longer history (2003+ instead of 2019+) | 0.9831 vs 0.9798 | No gain |
| Training window (4y / 8y / 12y / all) | 0.9828 / 0.9818 / 0.9818 / 0.9831 | Flat |
| Draw feature (abs Elo difference) | 0.9846 vs 0.9831 | Slightly worse |
| Margin-of-victory Elo | K=20: 0.9837, K=30: 0.9841 vs plain 0.9834 / 0.9831 | No gain |

## Final model
Elo only, K=30, home advantage 75, promoted-team start 1400, full history.
Average log loss 0.9831, baseline 1.0736, bookmaker 0.9660.

## Live prediction log
First 10 locked predictions (matchweek of 10 Oct 2026): model log loss 1.0944,
bookmaker 1.0212, top-pick accuracy 5/10. Far too small a sample to conclude
anything; the log is being extended weekly.

## Poisson attack/defence (independent goals, halflife 730d, refit 28d)
alpha 0.003: 0.9929 | 0.01: 0.9955 | 0.03: 1.0037 (Elo: 0.9831)
Folds (alpha 0.003): [0.9324, 1.0138, 1.0326]. Equal to Elo in folds 1 and 3, much worse in fold 2 (+0.029).
Draw max 0.315 at alpha 0.003. Fold 3 had 380 rows vs 374 (row-set mismatch).
Result: worse than Elo, not adopted.

## Elo + rolling-10 shots-on-target net diff (data 2003+)
Elo only 0.9831 -> Elo + sot_diff 0.9776 (-0.0055). Folds: [0.9256, 0.9812, 1.0260] vs [0.9335, 0.9849, 1.0310].
All 3 folds improved. NaN filled with 0, scaled x100. No tuning done yet.

## Elo + rolling shots-on-target net diff (data 2003+)
Feature: rolling mean of (SoT for - SoT against), shift(1), min_periods=3, NaN->0, x100.
Window 5: 0.9795 | 10: 0.9776 | 20: 0.9786 (Elo only: 0.9831).
All windows beat Elo; window 10 improved all 3 folds ([0.9256, 0.9812, 1.0260]).
Adopted window 10 (middle value, no further tuning).

## Total shots and corners on top of Elo + SoT (data 2003+)
Baseline Elo + SoT10: 0.9776. + total shots net (rolling-10): 0.9774 (-0.0002). + corners net: 0.9791 (+0.0015).
Folds, shots: [0.9235, 0.9803, 1.0283]; corners: [0.9257, 0.9844, 1.0272].
SoT already captures the shot information (total shots contains SoT). Neither adopted.