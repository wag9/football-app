# upcoming.py -- python upcoming.py [E0|SP1|D1|I1|F1] [--dry-run]
#
# Logs pre-match predictions for fixtures that have not kicked off yet.
# --dry-run shows exactly what would be logged and writes nothing.
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

import prediction_log as pl
from leagues import LEAGUES, DEFAULT
from model import train, predict_match, MODEL_VERSION

FIXTURES_PATH = "data/fixtures.csv"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dry_run = "--dry-run" in sys.argv
league = args[0] if args else DEFAULT
if league not in LEAGUES:
    sys.exit(f"Unknown league '{league}'. Choose one of: {', '.join(LEAGUES)}")
cfg = LEAGUES[league]
log_path = cfg["log"]

now_utc = datetime.now(timezone.utc)
print(f"league: {league} {cfg['name']} | now (UTC): {now_utc:%Y-%m-%d %H:%M} "
      f"| mode: {'DRY RUN (nothing is written)' if dry_run else 'write'}")

# 1. Fixtures for this league, kept as text. Parsing happens in prediction_log.
fx = pd.read_csv(FIXTURES_PATH, dtype=str, keep_default_na=False)
if "Div" in fx.columns:
    fx = fx[fx["Div"] == league]
if "Time" not in fx.columns:
    fx["Time"] = ""
fixtures = fx[["Date", "Time", "HomeTeam", "AwayTeam"]].to_dict("records")

# 2. Matches that already have a result
res = pd.read_csv(cfg["matches"], usecols=["Date", "HomeTeam", "AwayTeam", "FTR"],
                  dtype=str, keep_default_na=False)
res = res[res["FTR"].isin(["H", "D", "A"])]
played_keys = {(d[:10], h, a) for d, h, a in zip(res["Date"], res["HomeTeam"], res["AwayTeam"])}

# 3. Existing log (must already be on the new schema)
header, existing = pl.read_log(log_path)
if existing and header != pl.COLUMNS:
    sys.exit("The prediction log still has the old columns. Run once: python migrate_log.py")

# 4. Model and plan
state = train(league)
plan = pl.plan_predictions(
    fixtures, existing, played_keys, state["current"],
    lambda home, away: predict_match(state, home, away),
    now_utc, MODEL_VERSION, local_date=datetime.now().date().isoformat(),
)

# 5. Report
counts = Counter(reason.split(":")[0] for _, reason in plan.skipped)
print(f"\nfixtures read: {len(fixtures)} | to log: {len(plan.new_rows)} | skipped: {dict(counts)}")
for label, reason in plan.skipped:
    if reason.split(":")[0] in ("too_late", "unknown_team", "malformed", "bad_date",
                                "model_error", "bad_probs"):
        print(f"  skip [{reason}] {label}")
for w in plan.warnings:
    print("  WARNING:", w)

if plan.new_rows:
    show = pd.DataFrame(plan.new_rows)[["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA", "kickoff_utc"]]
    print("\n" + show.to_string(index=False))

# 6. Write (append only)
if dry_run:
    print("\nDRY RUN: nothing written.")
elif plan.new_rows:
    pl.append_rows(log_path, plan.new_rows)
    print(f"\nappended {len(plan.new_rows)} rows to {log_path}")
else:
    print("\nNo new predictions to log.")
print("total logged:", len(pl.read_log(log_path)[1]))