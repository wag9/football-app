# check.py
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

LABELS = ["A", "D", "H"]

log = pd.read_csv("data/predictions_log.csv", parse_dates=["Date"])
res = pd.read_csv("data/all_matches.csv", parse_dates=["Date"])

# Sirf wahi predictions jinke result aa chuke
m = log.merge(res[["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]],
              on=["Date", "HomeTeam", "AwayTeam"], how="inner")
print("predictions with results:", len(m), "of", len(log))
if len(m) == 0:
    raise SystemExit("Abhi koi result nahi aaya.")

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