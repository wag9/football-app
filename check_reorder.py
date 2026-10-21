# check_reorder.py -- READ-ONLY. Compares each matches file on disk with the version in the last commit (HEAD).
# Use it after python weekly.py (or combine.py) and BEFORE git add, to see whether the files only
# changed order and gained new rows, or whether old rows were changed or removed.
#
#   python check_reorder.py                      # all five matches files
#   python check_reorder.py data/all_matches.csv
import io
import subprocess
import sys
from collections import Counter

import pandas as pd

from leagues import LEAGUES


def read_head(path):
    """The committed version of a file, as text-only DataFrame (None if it is not in HEAD)."""
    p = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True)
    if p.returncode != 0:
        return None
    return pd.read_csv(io.BytesIO(p.stdout), dtype=str, keep_default_na=False)


def compare(old, new):
    """Return (removed rows, added rows, number of rows that moved) as lists of tuples / an int."""
    o = Counter(map(tuple, old.values.tolist()))
    n = Counter(map(tuple, new.values.tolist()))
    removed = list((o - n).elements())
    added = list((n - o).elements())
    common_old = [r for r in map(tuple, old.values.tolist()) if r not in set(removed)]
    common_new = [r for r in map(tuple, new.values.tolist()) if r not in set(added)]
    moved = sum(1 for a, b in zip(common_old, common_new) if a != b)
    return removed, added, moved


def main():
    paths = sys.argv[1:] or [cfg["matches"] for cfg in LEAGUES.values()]
    bad = False
    for path in paths:
        old = read_head(path)
        if old is None:
            print(f"{path}: not found in HEAD (new file?) - nothing to compare")
            continue
        new = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(old.columns) != list(new.columns):
            print(f"{path}: COLUMNS CHANGED {list(old.columns)} -> {list(new.columns)}")
            bad = True
            continue
        removed, added, moved = compare(old, new)
        verdict = "OK (only new rows and/or a different order)" if not removed else "CHECK: old rows were changed or removed"
        print(f"{path}: rows {len(old)} -> {len(new)} | new rows {len(added)} | old rows removed/changed {len(removed)}"
              f" | old rows that moved position {moved} | {verdict}")
        for r in removed[:5]:
            print("   removed/changed:", ",".join(r[:8]))
        for r in added[:3]:
            print("   added:", ",".join(r[:8]))
        bad = bad or bool(removed)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
