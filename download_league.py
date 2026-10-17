# download_league.py -- python download_league.py <code> <prefix> [all|current]
import sys
import time
from pathlib import Path

import requests

LEAGUE, PREFIX = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "all"
seasons = [f"{a:02d}{a+1:02d}" for a in range(3, 27)]
if mode == "current":
    seasons = ["2627"]
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
Path("data").mkdir(exist_ok=True)

ok, failed = 0, []
for s in seasons:
    out = Path(f"data/{PREFIX}{s}.csv")
    url = f"http://www.football-data.co.uk/mmz4281/{s}/{LEAGUE}.csv"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        if not r.content.lstrip(b"\xef\xbb\xbf \r\n").startswith(b"Div"):
            raise ValueError("not a CSV, first bytes: %r" % r.content[:40])
        out.write_bytes(r.content)
        print("ok    ", s, len(r.content), "bytes ->", out.resolve())
        ok += 1
    except Exception as e:
        print("FAILED", s, repr(e)[:150])
        failed.append(s)
    time.sleep(1)

print(f"\nDONE: {ok} ok, {len(failed)} failed", failed)