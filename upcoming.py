# upcoming.py
import os
import pandas as pd
from predict import predict, current

MODEL_VERSION = "elo_only K30 HA75 NEW1400 hist0304"
LOG_PATH = "data/predictions_log.csv"

# 1. Fixtures
fx = pd.read_csv("data/fixtures.csv")
if "Div" in fx.columns:
    fx = fx[fx["Div"] == "E0"]
fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True)

# 2. Jo matches already khele ja chuke, unhe hata do
played = pd.read_csv("data/all_matches.csv", parse_dates=["Date"])
played_keys = set(zip(played["Date"], played["HomeTeam"], played["AwayTeam"]))
fx = fx[[(d, h, a) not in played_keys for d, h, a in zip(fx["Date"], fx["HomeTeam"], fx["AwayTeam"])]]

# 3. Predict
rows = []
for r in fx.itertuples():
    if r.HomeTeam not in current or r.AwayTeam not in current:
        print("skip (naam nahi mila):", r.HomeTeam, "vs", r.AwayTeam)
        continue
    p = predict(r.HomeTeam, r.AwayTeam, verbose=False)
    rows.append({
        "Date": r.Date.date(), "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
        "pH": round(p["H"], 4), "pD": round(p["D"], 4), "pA": round(p["A"], 4),
        "model": MODEL_VERSION,
        "logged_on": pd.Timestamp.today().date(),
    })
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