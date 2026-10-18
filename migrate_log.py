# migrate_log.py -- python migrate_log.py
# One-time, additive: adds two blank columns (kickoff_utc, predicted_at_utc) to each
# existing prediction log. Existing values are not changed. A backup copy is saved in
# data/backups/ first, and the new file is verified before it replaces the old one.
from pathlib import Path

import prediction_log as pl
from leagues import LEAGUES

for code, cfg in LEAGUES.items():
    path = Path(cfg["log"])
    status, backup = pl.migrate_log(path, Path("data/backups"))
    rows = len(pl.read_log(path)[1]) if status != "missing" else 0
    extra = f" | backup: {backup}" if backup else ""
    print(f"{code:4s} {cfg['name']:15s} {status:16s} rows: {rows}{extra}")