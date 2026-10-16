# validate_prod.py -- checks the production feature path with walk-forward
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from data import load_matches
from elo import add_elo_features
from shots import add_sot_feature
from model import K, HOME_ADV, NEW_START, train, predict_match

LABELS = ["A", "D", "H"]
FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]

df = load_matches()
d = add_elo_features(df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START)
d["sot_diff"] = add_sot_feature(df)
d["target"] = df["FTR"]


def wf(cols):
    out = []
    for s, e in FOLDS:
        tr = d["Date"] < pd.Timestamp(s)
        te = (d["Date"] >= pd.Timestamp(s)) & (d["Date"] < pd.Timestamp(e))
        m = LogisticRegression(max_iter=1000).fit(d.loc[tr, cols], d.loc[tr, "target"])
        out.append(log_loss(d.loc[te, "target"], m.predict_proba(d.loc[te, cols]), labels=LABELS))
    return out


a, b = wf(["elo_diff"]), wf(["elo_diff", "sot_diff"])
print("Elo only      :", [round(x, 4) for x in a], "| avg:", round(np.mean(a), 4))
print("Elo + SoT diff:", [round(x, 4) for x in b], "| avg:", round(np.mean(b), 4))
print("delta:", round(np.mean(b) - np.mean(a), 4), "(expect about -0.005)")

s = train()
print("coefs (elo_diff, sot_diff):", s["model"].coef_.round(5).tolist())
print("Arsenal vs Liverpool:", {k: round(v, 3) for k, v in predict_match(s, "Arsenal", "Liverpool").items()})