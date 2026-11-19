# Case study: predicting football results honestly

A short account of what I built, how I checked it, and what I found. For the code and the run instructions, see the [README](README.md).

## The problem

Predict Home / Draw / Away probabilities for matches in the top five European leagues, and measure honestly how good those probabilities are. I did not set out to beat the bookmakers. Their odds are a very strong baseline, and I wanted to know exactly how far behind a simple model is.

## The model

- **Elo ratings** per league, updated match by match. The rating before each match is the feature, so nothing leaks from the future.
- **Shots-on-target form**: rolling mean of (shots on target for minus against) over the last 10 matches, shifted by one match.
- **Logistic regression** on those two features.
- Same settings in all five leagues (K=30, home advantage 75, promoted-team start 1400). No per-league tuning.

I kept it simple on purpose: with about 1,750 test matches per season across five leagues, a complex model has little room to prove itself.

## How I validated it

- **Walk-forward validation**: train only on matches before each test season (2023/24, 2024/25, 2025/26), never random splits.
- **Metric**: multiclass log loss, compared against a class-frequency baseline and against bookmaker odds (normalised to remove the margin).
- **Significance**: paired standard errors on the same matches, not just differences in averages.

## What I found

- The model beats the baseline clearly and trails the bookmaker in every league, by 0.012 to 0.022 log loss.
- Adding shots on target helped a little: -0.0042 log loss pooled over 5,256 test matches (paired SE 0.0011, 95% CI -0.0063 to -0.0021). It improved 14 of 15 league-season folds, but no single fold is significant on its own.
- Several ideas did not help and are logged in [results.md](results.md): longer history, mean reversion between seasons, margin-of-victory Elo, extra rolling features, a draw feature, a Poisson attack/defence model, total shots and corners.
- An ordered logit and gradient boosting, compared with the same nested design, did not beat the logistic regression (gradient boosting was significantly worse on the held-out season).

## Checking my own bias

K, home advantage and the shots window were first chosen while looking at the same test folds I reported. To measure how much that mattered, I ran a nested check: tune 140 configurations on the first two test seasons, then test once on the third.

- The reported configuration ranked 63rd of 140 on the held-out season, close to the median, so it was not a lucky pick.
- The configuration chosen without the held-out season scored 0.0029 better (paired SE 0.0016), which is within noise.
- The shots gain on the held-out season was -0.0043 (SE 0.0018), about the same as before.

This covers only those three settings. The choice of features and the Elo setup were not tested this way, and that season had been looked at before.

## Live evidence

Every prediction is written to an append-only log before kickoff, with a timestamp. A [track record page](https://wag9.github.io/football-app/track-record.html) scores only predictions whose timestamp is before kickoff, using the same code as the command-line check, and compares them with the bookmaker on the same matches. Below 30 scored matches it warns that the numbers are too early to read. Older predictions without timestamps are kept but scored separately and never counted as proof.

## Engineering

- A weekly pipeline: download, rebuild match tables, audit the data, score old predictions, log new ones. A league whose data fails the audit gets no new predictions.
- Data checks that report impossible values and missing odds instead of hiding them.
- Nearly 200 unit tests that use synthetic data only, run on GitHub Actions.
- FastAPI backend on Render, plain HTML/JS frontend on GitHub Pages.

Bugs I found and fixed along the way: a mismatch between how the shots feature was computed in training and in prediction, a season label that silently turned into a number and emptied the league tables, and a prediction that had been logged after the match and had to be removed and documented.

## Limitations

- Bookmaker odds include lineups and injuries, which the model never sees.
- Draws are never the top pick, because the draw probability stays below about 31%.
- Home wins are over-predicted by about 2.6 points. I measured this but have not applied a fix.
- The live sample is small. Even at a few dozen matches, the standard error is larger than the usual gap to the bookmaker.

## What I would do next

- Test a recalibration for the home-win bias, again nested.
- Keep the live log running. It is the only fully clean test, and it gets stronger every week.