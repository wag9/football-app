# weekly.py -- python weekly.py [E0 SP1 D1 I1 F1] [--no-download]
#
# For each league: download the current-season results, rebuild the matches table, audit it,
# score earlier predictions (check.py) and log predictions for upcoming fixtures (upcoming.py).
# It never commits anything. At the end it says plainly whether it is safe to commit.
#
# Every August, for a new season: change CURRENT_SEASON below AND the season range
# (range(3, 27)) in combine.py and combine_league.py.
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import requests

from leagues import LEAGUES

CURRENT_SEASON = "2627"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = os.environ.get("FOOTBALL_DATA_BASE", "http://www.football-data.co.uk").rstrip("/")
RETRIES = 3

# order of the steps shown in the final table
STEPS = ["download", "combine", "audit", "check", "log"]


# ------------------------------------------------------------------ downloading
def count_rows(content):
    """Data rows in a football-data CSV (bytes): lines whose first field is not empty, minus the header."""
    lines = [ln for ln in content.splitlines() if ln.split(b",")[0].strip(b"\xef\xbb\xbf \r\n\t")]
    return max(len(lines) - 1, 0)


def validate_download(content, existing_path=None, allow_shrink=True):
    """Raise ValueError if the downloaded bytes should not replace the file on disk."""
    if not content.lstrip(b"\xef\xbb\xbf \r\n").startswith(b"Div"):
        raise ValueError("not a CSV, first bytes: %r" % content[:40])
    if not allow_shrink and existing_path and Path(existing_path).exists():
        new_rows = count_rows(content)
        old_rows = count_rows(Path(existing_path).read_bytes())
        if new_rows < old_rows:
            raise ValueError(f"download has fewer rows than the file on disk ({new_rows} < {old_rows}); "
                             "refusing to overwrite")
    return count_rows(content)


def fetch(url, out, allow_shrink=True):
    """Download url to out. Retries connection problems and 5xx; a 4xx is final. Returns the row count."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} server error for url: {url}", response=r)
            r.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise                                   # 403/404: trying again will not help
            last = e
        if attempt < RETRIES:
            time.sleep(2 * attempt)
    else:
        raise last
    rows = validate_download(r.content, out, allow_shrink)
    tmp = Path(str(out) + ".part")
    tmp.write_bytes(r.content)
    os.replace(tmp, out)                                # the old file stays intact until the new one is complete
    return rows


def explain(e):
    msg = repr(e)[:150]
    resp = getattr(e, "response", None)
    if resp is not None and resp.status_code == 403:
        msg += " (the site refused this connection; if every league fails, try another network, e.g. a mobile hotspot)"
    return msg


# ------------------------------------------------------------------ running the other scripts
def run(*cmd):
    """Run one of the project scripts, show its output live, return (ok, full output)."""
    print("\n$ python", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable, "-u", *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    enc = sys.stdout.encoding or "utf-8"
    lines = []
    for line in p.stdout:
        print(line.encode(enc, "replace").decode(enc), end="", flush=True)
        lines.append(line)
    p.wait()
    return p.returncode == 0, "".join(lines)


# ------------------------------------------------------------------ the final verdict
def verdict(status, fixtures_ok, download):
    """status: {league: {step: 'ok' | 'FAILED' | 'skipped' | text}}. Returns (lines, exit_code)."""
    def done(s):                                         # league data rebuilt successfully
        return s.get("combine") == "ok"

    def clean(s):
        return all(v in ("ok", "skipped") or v.startswith("ok") for v in s.values()) and len(s) == len(STEPS)

    updated = [c for c, s in status.items() if done(s)]
    problems = [c for c, s in status.items() if not clean(s)]
    lines = []
    if download and not fixtures_ok:
        lines.append("WARNING: fixtures.csv was NOT updated, so no predictions for new fixtures can be logged.")
    if not updated:
        lines.append("RESULT: NOTHING WAS UPDATED. Do NOT commit. Fix the problem above (or try another network) and run again.")
        return lines, 1
    if problems:
        lines.append(f"RESULT: PARTIAL. Problems in: {', '.join(problems)}. Read the lines above for those leagues.")
        lines.append("        You may commit the leagues that are fine, but only after reading 'git diff --stat'.")
        return lines, 1
    if download and not fixtures_ok:
        lines.append("RESULT: results updated, fixtures not. Safe to commit what changed (read 'git diff --stat' first).")
        return lines, 1
    lines.append("RESULT: ALL OK. Next: read what changed, then commit.")
    return lines, 0


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    codes = [a for a in args if not a.startswith("--")] or list(LEAGUES)
    unknown = [c for c in codes if c not in LEAGUES]
    if unknown:
        print(f"Unknown league(s): {unknown}. Choose from: {list(LEAGUES)}")
        return 2
    download = "--no-download" not in args
    Path("data").mkdir(exist_ok=True)

    status = {c: {} for c in codes}
    fixtures_ok = False
    if download:
        try:
            n = fetch(f"{BASE}/fixtures.csv", "data/fixtures.csv")     # may legitimately shrink as matches are played
            fixtures_ok = True
            print(f"fixtures.csv updated ({n} rows)")
        except Exception as e:
            print("FAILED fixtures.csv:", explain(e))

    for code in codes:
        cfg, st = LEAGUES[code], status[code]
        print("\n" + "=" * 50)
        print(code, cfg["name"])
        print("=" * 50)

        if download:
            try:
                n = fetch(f"{BASE}/mmz4281/{CURRENT_SEASON}/{code}.csv",
                          f"data/{cfg['prefix']}{CURRENT_SEASON}.csv", allow_shrink=False)
                st["download"] = "ok"
                print(f"results file updated ({n} rows)")
            except Exception as e:
                st["download"] = "FAILED"
                print("FAILED results download:", explain(e), "-> skipping this league")
                continue
            time.sleep(1)
        else:
            st["download"] = "skipped"

        combine = ["combine.py"] if code == "E0" else ["combine_league.py", cfg["prefix"]]
        st["combine"] = "ok" if run(*combine)[0] else "FAILED"
        if st["combine"] != "ok":
            print("combine failed -> skipping this league")
            continue

        # Never log predictions on data that failed the integrity checks.
        st["audit"] = "ok" if run("audit_data.py", code)[0] else "ERRORS"
        if st["audit"] != "ok":
            print(f"audit_data.py found errors in {code} -> NOT logging predictions for this league."
                  " Fix the data problem shown above first.")
            continue

        if Path(cfg["log"]).exists():
            st["check"] = "ok" if run("check.py", code)[0] else "FAILED"
        else:
            st["check"] = "skipped"

        ok, out = run("upcoming.py", code)
        m = re.search(r"appended (\d+) rows", out)
        st["log"] = ("ok" if not m else f"ok (+{m.group(1)})") if ok else "FAILED"

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'league':8s}" + "".join(f"{s:>12s}" for s in STEPS))
    for code in codes:
        print(f"{code:8s}" + "".join(f"{status[code].get(s, '-'):>12s}" for s in STEPS))
    lines, exit_code = verdict(status, fixtures_ok, download)
    print()
    for ln in lines:
        print(ln)
    unsafe = [c for c in codes if status[c].get("audit") == "ERRORS"]
    if unsafe:
        # these leagues' files were rebuilt from data that failed the checks: do not commit them
        files = []
        for c in unsafe:
            files += [f"data/{LEAGUES[c]['prefix']}{CURRENT_SEASON}.csv", LEAGUES[c]["matches"]]
        print("\n  Do NOT run 'git add data/': the files of " + ", ".join(unsafe) + " failed the data checks.")
        print("  To undo them:  git checkout -- " + " ".join(files))
    elif exit_code == 0 or any(status[c].get("combine") == "ok" for c in codes):
        print("\n  git status")
        print("  git diff --stat        # expect: fixtures.csv, data/*2627.csv, the *_matches.csv files, the *_predictions_log.csv files")
        print("  git add data/")
        print(f'  git commit -m "Weekly update {date.today().isoformat()}"')
        print("  git push")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
