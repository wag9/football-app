# combine_league.py -- python combine_league.py laliga
import csv
import sys
import pandas as pd

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "laliga"
seasons = [f"{a:02d}{a+1:02d}" for a in range(3, 27)]

base_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
shot_cols = ["HS", "AS", "HST", "AST", "HC", "AC"]
odds_options = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]


def read_ragged(path):
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    data = [r[:n] + [""] * (n - len(r)) for r in rows[1:]]
    d = pd.DataFrame(data, columns=header)
    return d.replace("", pd.NA)


frames = []
for s in seasons:
    print("reading", s, "...", end=" ")
    d = read_ragged(f"data/{PREFIX}{s}.csv")
    d = d.dropna(subset=["Date", "HomeTeam"])

    odds_used = None
    for h, dr, a in odds_options:
        if h in d.columns and dr in d.columns and a in d.columns:
            d = d.rename(columns={h: "AvgH", dr: "AvgD", a: "AvgA"})
            odds_used = h
            break
    if odds_used is None:
        d["AvgH"] = d["AvgD"] = d["AvgA"] = float("nan")

    for c in shot_cols:
        if c not in d.columns:
            d[c] = float("nan")

    d = d[base_cols + ["AvgH", "AvgD", "AvgA"] + shot_cols].copy()
    for c in ["FTHG", "FTAG", "AvgH", "AvgD", "AvgA"] + shot_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
    d["season"] = s
    print(d.shape, "NaT:", d["Date"].isna().sum(), "| odds:", odds_used,
          "| HST missing:", d["HST"].isna().sum())
    frames.append(d)

all_matches = pd.concat(frames, ignore_index=True)
all_matches = all_matches.sort_values("Date").reset_index(drop=True)
print(all_matches.shape)
print("teams:", all_matches["HomeTeam"].nunique())
all_matches.to_csv(f"data/{PREFIX}_matches.csv", index=False)