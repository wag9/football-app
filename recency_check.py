# recency_check.py -- does down-weighting old seasons fix the home/away bias? (Elo + SoT10)
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
HALFLIVES = {"equal": None, "3y": 3, "6y": 6}     # years; "equal" = current production

print("VERSION: recency-weighted logistic fit (halflife 3y / 6y vs equal weights), Elo + SoT10")
rows = []
for code, cfg in LEAGUES.items():
    df = load_matches(cfg["matches"]).dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(drop=True)
    d = add_elo_features(df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START)
    d["sot_diff"] = add_sot_feature(df)
    d["target"] = df["FTR"]
    for fi, (s, e) in enumerate(FOLDS):
        start = pd.Timestamp(s)
        tr = d["Date"] < start
        te = (d["Date"] >= start) & (d["Date"] < pd.Timestamp(e))
        age = (start - d.loc[tr, "Date"]).dt.days.values
        y = d.loc[te, "target"]
        for name, hl in HALFLIVES.items():
            if hl is None:
                w = None
            else:
                w = 0.5 ** (age / (hl * 365.25))
                w = w / w.mean()                     # keep effective regularization comparable
            m = LogisticRegression(max_iter=1000).fit(d.loc[tr, COLS], d.loc[tr, "target"], sample_weight=w)
            P = m.predict_proba(d.loc[te, COLS])     # columns A, D, H
            rows.append({"league": code, "fold": fi, "hl": name, "n": int(te.sum()),
                         "ll": log_loss(y, P, labels=LABELS),
                         "pH": P[:, 2].mean(), "pA": P[:, 0].mean(),
                         "aH": (y == "H").mean(), "aA": (y == "A").mean()})

res = pd.DataFrame(rows)
print("test rows:", int(res[res["hl"] == "equal"]["n"].sum()))

piv = res.pivot_table(index=["league", "fold"], columns="hl", values="ll")
print("\nAvg log loss per league:")
print(piv.groupby(level="league").mean().round(4))
print("\nPooled avg over 15 folds:", piv.mean().round(4).to_dict())
for h in ["3y", "6y"]:
    delta = piv[h] - piv["equal"]
    print(f"{h} vs equal: delta {delta.mean():+.4f} | folds improved {int((delta < 0).sum())} of {len(delta)}")

print("\nMean predicted vs actual (pooled, weighted by rows):")
for h, g in res.groupby("hl"):
    n = g["n"].sum()
    print(f"{h:6s} H {np.average(g['pH'], weights=g['n']):.3f} vs {np.average(g['aH'], weights=g['n']):.3f}"
          f" | A {np.average(g['pA'], weights=g['n']):.3f} vs {np.average(g['aA'], weights=g['n']):.3f}")