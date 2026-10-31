# selection_bias.py -- how optimistic are the reported scores?
# Tune (K, home advantage, SoT window) on folds 1-2 only, then test ONCE on fold 3.
# Read-only: changes no data, no model, no log. Run from the repo root: python selection_bias.py
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import shots
from data import load_matches
from elo import add_elo_features
from leagues import LEAGUES

FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]
NEW_START = 1400
KS = [15, 20, 25, 30, 35, 40, 50]
HAS = [50, 60, 75, 90, 100]
WINDOWS = [5, 10, 15, 20]
ORIG = (30, 75, 10)                       # the config (K, home adv, window)
LABELS = np.array(["A", "D", "H"])
GRID = [(k, ha, w) for k in KS for ha in HAS for w in WINDOWS]


def sot_feature(df, window):
    """Same as shots.add_sot_feature, but with a chosen window."""
    st = shots._team_log(df)
    st["roll"] = st.groupby("Team")["net"].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=shots.MIN_PERIODS).mean())
    hr = st[st["side"] == "h"].set_index("idx")["roll"].reindex(df.index)
    ar = st[st["side"] == "a"].set_index("idx")["roll"].reindex(df.index)
    return ((hr - ar).fillna(0) * shots.SCALE).to_numpy()


def fold_losses(X, y, dates):
    """Per-match log loss on each fold; training = all rows before the fold start."""
    out = []
    for start, end in FOLDS:
        s, e = np.datetime64(start), np.datetime64(end)
        tr, te = dates < s, (dates >= s) & (dates < e)
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        assert list(m.classes_) == list(LABELS), m.classes_
        p = m.predict_proba(X[te])
        idx = np.searchsorted(LABELS, y[te])
        out.append(-np.log(np.clip(p[np.arange(len(idx)), idx], 1e-15, 1.0)))
    return out


def se(v):
    v = np.asarray(v)
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    t0 = time.time()
    codes = list(LEAGUES)
    dfs = {c: load_matches(LEAGUES[c]["matches"]) for c in codes}
    dates = {c: dfs[c]["Date"].to_numpy() for c in codes}
    ys = {c: dfs[c]["FTR"].to_numpy(dtype=str) for c in codes}

    # self-check: my window-10 feature must equal the production one
    for c in codes:
        assert np.allclose(sot_feature(dfs[c], 10), shots.add_sot_feature(dfs[c]).to_numpy()), c
    print("self-check OK: window-10 feature equals shots.add_sot_feature in all 5 leagues")

    sot = {(c, w): sot_feature(dfs[c], w) for c in codes for w in WINDOWS}

    sot_res, elo_res = {}, {}             # key -> {league: [fold1, fold2, fold3 per-match losses]}
    for k in KS:
        for ha in HAS:
            elo_res[(k, ha)] = {}
            elo = {}
            for c in codes:
                elo[c] = add_elo_features(dfs[c], k=k, home_adv=ha, new_team_start=NEW_START)["elo_diff"].to_numpy()
                elo_res[(k, ha)][c] = fold_losses(elo[c].reshape(-1, 1), ys[c], dates[c])
            for w in WINDOWS:
                sot_res[(k, ha, w)] = {
                    c: fold_losses(np.column_stack([elo[c], sot[(c, w)]]), ys[c], dates[c]) for c in codes}
            print(f"  done K={k} HA={ha}  ({time.time() - t0:.0f}s)", flush=True)

    def pool(res, key, folds, leagues=codes):
        return np.concatenate([res[key][c][f] for c in leagues for f in folds])

    n = [sum(len(sot_res[ORIG][c][f]) for c in codes) for f in range(3)]
    print(f"\ntest matches per fold (5 leagues pooled): {n}")

    # ---- 1. tune on folds 1-2 ----
    s12 = {g: pool(sot_res, g, [0, 1]).mean() for g in GRID}
    ranked = sorted(GRID, key=lambda g: s12[g])
    tuned = ranked[0]
    print("\n== Tuning on folds 1-2 (2023/24, 2024/25), pooled log loss ==")
    for g in ranked[:5]:
        print(f"  K={g[0]:>2} HA={g[1]:>3} window={g[2]:>2}: {s12[g]:.4f}")
    print(f"  original K={ORIG[0]} HA={ORIG[1]} window={ORIG[2]}: {s12[ORIG]:.4f} "
          f"(rank {ranked.index(ORIG) + 1} of {len(GRID)})")
    print(f"  TUNED = K={tuned[0]} HA={tuned[1]} window={tuned[2]}")

    # ---- 2. fold 3, tested once ----
    s3 = {g: pool(sot_res, g, [2]).mean() for g in GRID}
    r3 = sorted(GRID, key=lambda g: s3[g])
    oracle = r3[0]
    print("\n== Fold 3 (2025/26): pooled log loss, Elo + SoT ==")
    print(f"  tuned on folds 1-2 : {s3[tuned]:.4f}  (rank {r3.index(tuned) + 1} of {len(GRID)} on fold 3)")
    print(f"  original reported  : {s3[ORIG]:.4f}  (rank {r3.index(ORIG) + 1} of {len(GRID)} on fold 3)")
    print(f"  best on fold 3     : {s3[oracle]:.4f}  (K={oracle[0]} HA={oracle[1]} window={oracle[2]}; "
          "hindsight, not a valid result)")
    print(f"  median of all {len(GRID)}   : {np.median(list(s3.values())):.4f}")
    d = pool(sot_res, tuned, [2]) - pool(sot_res, ORIG, [2])
    print(f"  tuned minus original: {d.mean():+.4f} (paired se {se(d):.4f}, n={len(d)}; "
          "positive = original did better on fold 3)")

    # ---- 3. SoT gain on fold 3 ----
    print("\n== SoT gain on fold 3 (Elo + SoT minus Elo only, same K and HA; negative = SoT helps) ==")
    for name, g in (("original", ORIG), ("tuned", tuned)):
        diff = pool(sot_res, g, [2]) - pool(elo_res, g[:2], [2])
        print(f"  {name:8s} K={g[0]} HA={g[1]} w={g[2]}: pooled {diff.mean():+.4f} (se {se(diff):.4f}, "
              f"z {diff.mean() / se(diff):+.2f})")
        per = []
        for c in codes:
            dc = sot_res[g][c][2] - elo_res[g[:2]][c][2]
            per.append(f"{c} {dc.mean():+.4f}")
        print("           per league: " + " | ".join(per))
    print("  window sensitivity at the tuned K, HA (fold 3 gain vs Elo only):")
    for w in WINDOWS:
        diff = pool(sot_res, tuned[:2] + (w,), [2]) - pool(elo_res, tuned[:2], [2])
        print(f"    window {w:>2}: {diff.mean():+.4f} (se {se(diff):.4f})")

    # ---- 4. per league, fold 3 ----
    print("\n== Fold 3 per league, Elo + SoT: original vs tuned ==")
    for c in codes:
        print(f"  {c}: original {sot_res[ORIG][c][2].mean():.4f} | tuned {sot_res[tuned][c][2].mean():.4f}")
    print(f"\nfinished in {time.time() - t0:.0f}s. Nothing was written to disk.")


if __name__ == "__main__":
    main()