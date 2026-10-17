# check.py -- python check.py [E0|SP1]
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from leagues import LEAGUES, DEFAULT

LABELS = ["A", "D", "H"]
league = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
cfg = LEAGUES[league]
print("league:", league, cfg["name"])

log = pd.read_csv(cfg["log"], parse_dates=["Date"])
res = pd.read_csv(cfg["matches"], parse_dates=["Date"])

# Sirf wahi predictions jinke result aa chuke
m = log.merge(res[["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]],
              on=["Date", "HomeTeam", "AwayTeam"], how="inner")
print("predictions with results:", len(m), "of", len(log))
if len(m) == 0:
      print("No results yet for logged predictions.")
      sys.exit(0)
# Model
P = m[["pA", "pD", "pH"]].values
P = P / P.sum(axis=1, keepdims=True)   # rounding ka sum fix
model_ll = log_loss(m["FTR"], P, labels=LABELS)

# Bookmaker (1/odds, normalize)
raw = 1 / m[["AvgA", "AvgD", "AvgH"]].values
book = raw / raw.sum(axis=1, keepdims=True)
book_ll = log_loss(m["FTR"], book, labels=LABELS)

# Top-pick accuracy
top = np.array(LABELS)[P.argmax(axis=1)]
acc = (top == m["FTR"].values).mean()

print(f"model log loss    : {model_ll:.4f}")
print(f"bookmaker log loss: {book_ll:.4f}")
print(f"top-pick accuracy : {acc:.0%} ({(top == m['FTR'].values).sum()}/{len(m)})")
print(m[["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA", "FTR"]])