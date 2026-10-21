# audit_data.py -- READ-ONLY integrity audit of the five league match files and fixtures.csv.
# Writes nothing. Exit code: 1 if a matches file has an ERROR-level problem (or cannot be read), else 0.
# Problems found only in fixtures.csv are shown but do not set the exit code (upcoming.py skips such fixtures).
# Run from the repo root:
#   python audit_data.py                 # all five leagues
#   python audit_data.py E0 SP1          # selected leagues
#   python audit_data.py --today 2026-10-20
import datetime
import sys

import pandas as pd

from data_checks import run_matches_checks, check_fixtures
from leagues import LEAGUES

FIXTURES = "data/fixtures.csv"


def print_issues(issues):
    if not issues:
        print("  no issues found")
        return
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for i in sorted(issues, key=lambda x: order[x.level]):
        print(f"  [{i.level}] {i.check} (n={i.n}): {i.detail}")


def main():
    args = sys.argv[1:]
    today = datetime.date.today().isoformat()
    if "--today" in args:
        k = args.index("--today")
        today = args[k + 1]
        del args[k:k + 2]
    leagues = args or list(LEAGUES)

    try:
        fx_all = pd.read_csv(FIXTURES, encoding="utf-8-sig")
    except Exception as e:                      # fixtures are optional for the matches audit
        fx_all = None
        print(f"(could not read {FIXTURES}: {e})")

    totals = {"ERROR": 0, "WARN": 0, "INFO": 0}
    data_errors = 0
    print(f"audit_data.py | today = {today} | read-only\n")
    for code in leagues:
        cfg = LEAGUES[code]
        print(f"=== {code} {cfg['name']} | {cfg['matches']}")
        try:
            df = pd.read_csv(cfg["matches"], dtype={"season": str})   # str: keep "0304" (default read gives int 304)
        except Exception as e:
            print(f"  [ERROR] cannot read file: {e}\n")
            totals["ERROR"] += 1
            data_errors += 1
            continue
        print(f"  rows: {len(df)} | seasons: {df['season'].nunique() if 'season' in df.columns else '?'}"
              f" | dates: {df['Date'].min()} .. {df['Date'].max()}" if "Date" in df.columns else f"  rows: {len(df)}")
        issues = run_matches_checks(df, today=today)
        data_errors += sum(1 for i in issues if i.level == "ERROR")
        if fx_all is not None and "Div" in fx_all.columns and not issues_have_error(issues, "columns"):
            fx = fx_all[fx_all["Div"] == code]
            print(f"  fixtures rows for {code}: {len(fx)}")
            if len(fx):
                issues += check_fixtures(fx, df)
        print_issues(issues)
        for i in issues:
            totals[i.level] += 1
        print()
    print(f"SUMMARY: {totals['ERROR']} error checks, {totals['WARN']} warnings, {totals['INFO']} info. Nothing was written to disk.")
    return 1 if data_errors else 0


def issues_have_error(issues, check):
    return any(i.check == check and i.level == "ERROR" for i in issues)


if __name__ == "__main__":
    sys.exit(main())
