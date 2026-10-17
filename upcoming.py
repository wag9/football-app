# upcoming.py -- python upcoming.py [E0|SP1]
import os
import sys

import pandas as pd

from leagues import LEAGUES, DEFAULT
from model import train, predict_match

MODEL_VERSION = "elo_sot10 K30 HA75 NEW1400"

league = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
cfg = LEAGUES[league]
LOG_PATH = cfg["log"]
state = train(league)
current = state["current"]
print("league:", league, cfg["name"])

# 1. Fixtures
fx = pd.read_csv("data/fixtures.csv")
if "Div" in fx.columns:
    fx = fx[fx["Div"] == league]
fx = fx.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True)

# 2. Jo matches already khele ja chuke, unhe hata do
played = pd.read_csv(cfg["matches"], parse_dates=["Date"])
played_keys = set(zip(played["Date"], played["HomeTeam"], played["AwayTeam"]))
fx = fx[[(d, h, a) not in played_keys for d, h, a in zip(fx["Date"], fx["HomeTeam"], fx["AwayTeam"])]]

# 2b. Guard: sirf aaj ya aage ke matches (beete hue matches pre-match prediction nahi ho sakte)
today = pd.Timestamp.today().normalize()
stale = fx[fx["Date"] < today]
if len(stale):
    print(f"skip {len(stale)} past fixtures not yet in results:",
          list(zip(stale["HomeTeam"], stale["AwayTeam"])))
fx = fx[fx["Date"] >= today]

# 3. Predict
rows = []
for r in fx.itertuples():
    if r.HomeTeam not in current or r.AwayTeam not in current:
        print("skip (naam nahi mila):", r.HomeTeam, "vs", r.AwayTeam)
        continue
    p = predict_match(state, r.HomeTeam, r.AwayTeam)
    rows.append({
        "Date": r.Date.date(), "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
        "pH": round(p["H"], 4), "pD": round(p["D"], 4), "pA": round(p["A"], 4),
        "model": MODEL_VERSION,
        "logged_on": pd.Timestamp.today().date(),
    })

if not rows:
    raise SystemExit("No new fixtures to log for this league.")
new = pd.DataFrame(rows)

# 4. Purane log ke saath jodo, purani prediction KABHI overwrite nahi (keep="first")
if os.path.exists(LOG_PATH):
    old = pd.read_csv(LOG_PATH)
    new = pd.concat([old, new], ignore_index=True)
new["Date"] = new["Date"].astype(str)
new = new.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
new.to_csv(LOG_PATH, index=False)

print(new.tail(15))
print("total logged:", len(new))