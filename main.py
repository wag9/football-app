# main.py -- experiments and validation
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from data import load_matches
from elo import add_elo_features
from poisson import walk_forward_poisson

LABELS = ["A", "D", "H"]
FOLDS = [
    ("2023-08-01", "2024-08-01"),
    ("2024-08-01", "2025-08-01"),
    ("2025-08-01", "2026-08-01"),
]
K, HOME_ADV, NEW_START = 30, 75, 1400

# 1. Load data
df = load_matches()

# 2. Unified team logs
home = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].rename(
    columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA"}
)
home["Pts"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
home["Venue"] = "Home"

away = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].rename(
    columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA"}
)
away["Pts"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})
away["Venue"] = "Away"

team_df = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)

# 3. Rolling-5 form (shift(1) = no leakage)
for col in ["Pts", "GF", "GA"]:
    team_df[f"roll_{col.lower()}"] = team_df.groupby("Team")[col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

# 4. Venue-wise rolling-10 PPG
home_only = team_df[team_df["Venue"] == "Home"].copy()
home_only["home_ppg"] = home_only.groupby("Team")["Pts"].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
)
away_only = team_df[team_df["Venue"] == "Away"].copy()
away_only["away_ppg"] = away_only.groupby("Team")["Pts"].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
)

# 5. Merge rolling stats back to match schedule
roll_cols = ["Date", "Team", "roll_pts", "roll_gf", "roll_ga"]

df = df.merge(team_df[roll_cols], left_on=["Date", "HomeTeam"],
              right_on=["Date", "Team"], how="left"
              ).rename(columns={"roll_pts": "home_roll_pts",
                                "roll_gf": "home_roll_gf",
                                "roll_ga": "home_roll_ga"}).drop(columns=["Team"])

df = df.merge(team_df[roll_cols], left_on=["Date", "AwayTeam"],
              right_on=["Date", "Team"], how="left"
              ).rename(columns={"roll_pts": "away_roll_pts",
                                "roll_gf": "away_roll_gf",
                                "roll_ga": "away_roll_ga"}).drop(columns=["Team"])

df = df.merge(home_only[["Date", "Team", "home_ppg"]],
              left_on=["Date", "HomeTeam"], right_on=["Date", "Team"],
              how="left").drop(columns=["Team"])

df = df.merge(away_only[["Date", "Team", "away_ppg"]],
              left_on=["Date", "AwayTeam"], right_on=["Date", "Team"],
              how="left").drop(columns=["Team"])

# 6. Differential features
df["pts_diff"] = df["home_roll_pts"] - df["away_roll_pts"]
df["gf_diff"] = df["home_roll_gf"] - df["away_roll_gf"]
df["ga_diff"] = df["home_roll_ga"] - df["away_roll_ga"]
df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

# 7. Base table (same rows every run, so dropna uses the 12 old features)
old_feature_cols = [
    "home_roll_pts", "away_roll_pts",
    "home_ppg", "away_ppg",
    "pts_diff", "gf_diff", "ga_diff", "ppg_diff",
    "home_roll_gf", "home_roll_ga",
    "away_roll_gf", "away_roll_ga",
]

base_df = (
    df[["Date", "HomeTeam", "AwayTeam", "season", "FTR", "AvgH", "AvgD", "AvgA"] + old_feature_cols]
    .rename(columns={"FTR": "target"})
    .dropna(subset=old_feature_cols)
    .reset_index(drop=True)
)


# 8. Helpers (all functions are defined BEFORE they are called below)
def add_elo(base, **elo_kwargs):
    """Add elo_diff to base_df. Elo runs on the full df (it needs every match)."""
    elo = add_elo_features(df, **elo_kwargs)[["Date", "HomeTeam", "AwayTeam", "elo_diff"]]
    return base.merge(elo, on=["Date", "HomeTeam", "AwayTeam"], how="left")


def walk_forward(data, cols):
    scores = []
    for start, end in FOLDS:
        tr = data["Date"] < pd.Timestamp(start)
        te = (data["Date"] >= pd.Timestamp(start)) & (data["Date"] < pd.Timestamp(end))
        m = LogisticRegression(max_iter=1000)
        m.fit(data.loc[tr, cols], data.loc[tr, "target"])
        scores.append(log_loss(data.loc[te, "target"],
                               m.predict_proba(data.loc[te, cols]), labels=LABELS))
    return scores


def calibration(data, cols):
    parts = []
    for start, end in FOLDS:
        tr = data["Date"] < pd.Timestamp(start)
        te = (data["Date"] >= pd.Timestamp(start)) & (data["Date"] < pd.Timestamp(end))
        m = LogisticRegression(max_iter=1000)
        m.fit(data.loc[tr, cols], data.loc[tr, "target"])
        p = pd.DataFrame(m.predict_proba(data.loc[te, cols]), columns=m.classes_)
        p["actual"] = data.loc[te, "target"].values
        parts.append(p)
    preds = pd.concat(parts, ignore_index=True)
    preds["bin"] = pd.cut(preds["H"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    table = preds.groupby("bin", observed=True).agg(
        n=("H", "size"), predicted=("H", "mean"),
        actual=("actual", lambda s: (s == "H").mean()),
    )
    print("max P(H):", round(preds["H"].max(), 3))
    print("draw min/mean/max:", preds["D"].min().round(3), preds["D"].mean().round(3), preds["D"].max().round(3))
    print(table.round(3))


def benchmark(data):
    """Per-fold log loss of the class-frequency baseline and the bookmaker."""
    base, book = [], []
    for start, end in FOLDS:
        tr = data["Date"] < pd.Timestamp(start)
        te = (data["Date"] >= pd.Timestamp(start)) & (data["Date"] < pd.Timestamp(end))
        y_tr, y_te = data.loc[tr, "target"], data.loc[te, "target"]
        props = y_tr.value_counts(normalize=True).reindex(LABELS).values
        base.append(log_loss(y_te, np.tile(props, (len(y_te), 1)), labels=LABELS))
        raw = 1 / data.loc[te, ["AvgA", "AvgD", "AvgH"]].values
        book.append(log_loss(y_te, raw / raw.sum(axis=1, keepdims=True), labels=LABELS))
    return base, book


# 9. Final model validation
print("VERSION: FINAL elo_only, K=30, home_adv=75, nts=1400, full history")
print("rows:", len(base_df))
data = add_elo(base_df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START)

sc = walk_forward(data, ["elo_diff"])
print(f"folds: {[round(s, 4) for s in sc]} | avg: {np.mean(sc):.4f}")

base, book = benchmark(data)
print(f"baseline folds : {[round(s, 4) for s in base]} | avg: {np.mean(base):.4f}")
print(f"bookmaker folds: {[round(s, 4) for s in book]} | avg: {np.mean(book):.4f}")

# 10. Calibration (set to True to run)
RUN_CALIBRATION = False
if RUN_CALIBRATION:
    print("\n" + "=" * 40)
    calibration(data, ["elo_diff"])

# 11. Poisson attack/defence model vs Elo
print("\n" + "=" * 40)
print("VERSION: Poisson attack/defence, independent goals, halflife 730d, refit every 28d")
for alpha in [0.003, 0.01, 0.03]:
    sc_p, probs = walk_forward_poisson(df, base_df, FOLDS, alpha=alpha)
    print(f"alpha={alpha} | folds: {[round(s, 4) for s in sc_p]} | avg: {np.mean(sc_p):.4f}"
          f" | draw min/mean/max: {probs[:, 1].min():.3f}/{probs[:, 1].mean():.3f}/{probs[:, 1].max():.3f}")