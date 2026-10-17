# download_league.py -- python download_league.py
import time
from pathlib import Path

import requests

LEAGUE, PREFIX = "SP1", "laliga"          # SP1 = La Liga
seasons = [f"{a:02d}{a+1:02d}" for a in range(3, 27)]
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for s in seasons:
    out = Path(f"data/{PREFIX}{s}.csv")
    url = f"https://www.football-data.co.uk/mmz4281/{s}/{LEAGUE}.csv"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        print("ok    ", s, len(r.content), "bytes")
    except Exception as e:
        print("FAILED", s, repr(e))
    time.sleep(1)