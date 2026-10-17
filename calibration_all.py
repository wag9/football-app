# calibration_all.py -- pooled walk-forward calibration, all leagues (Elo + SoT10)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from data import load_matches
from elo import add_elo_features
from leagues import LEAGUES
from shots import add_sot_feature

LABELS = ["A", "D", "H"]
FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]
K, HOME_ADV, NEW_START = 30, 75, 1400
COLS = ["elo_diff", "sot_diff"]

parts = []
for code, cfg in LEAGUES.items():
    df = load_matches(cfg["matches"]).dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(drop=True)
    d = add_elo_features(df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START)
    d["sot_diff"] = add_sot_feature(df)
    d["target"] = df["FTR"]
    for s, e in FOLDS:
        tr = d["Date"] < pd.Timestamp(s)
        te = (d["Date"] >= pd.Timestamp(s)) & (d["Date"] < pd.Timestamp(e))
        m = LogisticRegression(max_iter=1000).fit(d.loc[tr, COLS], d.loc[tr, "target"])
        p = pd.DataFrame(m.predict_proba(d.loc[te, COLS]), columns=m.classes_)
        p["actual"] = d.loc[te, "target"].values
        p["league"] = code
        parts.append(p)

preds = pd.concat(parts, ignore_index=True)
print("VERSION: pooled walk-forward calibration, 5 leagues, Elo + SoT10, no fix applied")
print("test rows:", len(preds))
print("overall log loss:", round(log_loss(preds["actual"], preds[LABELS].values, labels=LABELS), 4))
print("per league:", {c: round(log_loss(g["actual"], g[LABELS].values, labels=LABELS), 4)
                      for c, g in preds.groupby("league")})

BINS = {
    "H": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "A": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
    "D": [0, 0.2, 0.24, 0.26, 0.28, 0.30, 1.0],
}
for cls, bins in BINS.items():
    t = preds.copy()
    t["bin"] = pd.cut(t[cls], bins=bins)
    t["hit"] = (t["actual"] == cls).astype(float)
    tab = t.groupby("bin", observed=True).agg(n=(cls, "size"), predicted=(cls, "mean"), actual=("hit", "mean"))
    tab["gap"] = tab["actual"] - tab["predicted"]
    print(f"\nClass {cls}: predicted vs actual frequency")
    print(tab.round(3))

print("\nmean predicted vs actual:",
      {c: (round(preds[c].mean(), 3), round((preds["actual"] == c).mean(), 3)) for c in LABELS})