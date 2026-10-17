# weekly.py -- python weekly.py [E0 SP1 D1 I1 F1] [--no-download]
import subprocess
import sys
import time
from pathlib import Path

import requests

from leagues import LEAGUES

CURRENT_SEASON = "2627"          # change every August
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "http://www.football-data.co.uk"

codes = [a for a in sys.argv[1:] if not a.startswith("--")] or list(LEAGUES)
download = "--no-download" not in sys.argv
Path("data").mkdir(exist_ok=True)


def fetch(url, out):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if not r.content.lstrip(b"\xef\xbb\xbf \r\n").startswith(b"Div"):
        raise ValueError("not a CSV, first bytes: %r" % r.content[:40])
    Path(out).write_bytes(r.content)


def run(*cmd):
    print("\n$ python", " ".join(cmd))
    return subprocess.run([sys.executable, *cmd]).returncode == 0


if download:
    try:
        fetch(f"{BASE}/fixtures.csv", "data/fixtures.csv")
        print("fixtures.csv updated")
    except Exception as e:
        print("FAILED fixtures.csv:", repr(e)[:150])

for code in codes:
    cfg = LEAGUES[code]
    print("\n" + "=" * 50)
    print(code, cfg["name"])
    print("=" * 50)

    if download:
        try:
            fetch(f"{BASE}/mmz4281/{CURRENT_SEASON}/{code}.csv",
                  f"data/{cfg['prefix']}{CURRENT_SEASON}.csv")
            print("results file updated")
        except Exception as e:
            print("FAILED results download:", repr(e)[:150], "-> skipping this league")
            continue
        time.sleep(1)

    combine = ["combine.py"] if code == "E0" else ["combine_league.py", cfg["prefix"]]
    if not run(*combine):
        print("combine failed -> skipping this league")
        continue
    if Path(cfg["log"]).exists():
        run("check.py", code)
    run("upcoming.py", code)

print("\nDone. Review the output above (check lines, 'skip' lines), then:")
print("  git add data/ && git commit -m \"Weekly update\" && git push")