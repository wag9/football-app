# laliga_check.py -- Elo vs Elo+SoT vs bookmaker on La Liga (same 3 folds as PL)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from data import load_matches
from elo import add_elo_features
from shots import add_sot_feature

LABELS = ["A", "D", "H"]
FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]
K, HOME_ADV, NEW_START = 30, 75, 1400

df = load_matches("data/laliga_matches.csv")
df = df.dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(drop=True)
print("VERSION: La Liga, Elo K30 HA75 NEW1400 (PL config, no tuning), + SoT10")
print("rows:", len(df), "| teams:", df["HomeTeam"].nunique())
print("class freq (all):", df["FTR"].value_counts(normalize=True).reindex(LABELS).round(3).to_dict())

d = add_elo_features(df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START)
d["sot_diff"] = add_sot_feature(df)
d["target"] = df["FTR"]
d[["AvgH", "AvgD", "AvgA"]] = df[["AvgH", "AvgD", "AvgA"]]

res = {"Elo": [], "Elo + SoT": [], "Baseline": [], "Bookmaker": []}
for s, e in FOLDS:
    tr = d["Date"] < pd.Timestamp(s)
    te = (d["Date"] >= pd.Timestamp(s)) & (d["Date"] < pd.Timestamp(e))
    y_te = d.loc[te, "target"]
    print(f"fold {s}: test rows {te.sum()} | odds missing: {d.loc[te, 'AvgH'].isna().sum()}")

    for name, cols in [("Elo", ["elo_diff"]), ("Elo + SoT", ["elo_diff", "sot_diff"])]:
        m = LogisticRegression(max_iter=1000).fit(d.loc[tr, cols], d.loc[tr, "target"])
        res[name].append(log_loss(y_te, m.predict_proba(d.loc[te, cols]), labels=LABELS))

    props = d.loc[tr, "target"].value_counts(normalize=True).reindex(LABELS).values
    res["Baseline"].append(log_loss(y_te, np.tile(props, (len(y_te), 1)), labels=LABELS))

    ok = te & d["AvgH"].notna()
    raw = 1 / d.loc[ok, ["AvgA", "AvgD", "AvgH"]].values
    res["Bookmaker"].append(log_loss(d.loc[ok, "target"], raw / raw.sum(axis=1, keepdims=True), labels=LABELS))

print()
for k, v in res.items():
    print(f"{k:10s}: {[round(x, 4) for x in v]} | avg: {np.mean(v):.4f}")
print("Elo+SoT vs Elo:", round(np.mean(res["Elo + SoT"]) - np.mean(res["Elo"]), 4))
print("Elo+SoT vs bookmaker gap:", round(np.mean(res["Elo + SoT"]) - np.mean(res["Bookmaker"]), 4))