# recalibration.py -- does a recent-window bias correction fix the home-win over-prediction?
# Read-only: changes no data, no model, no log. Run from the repo root: python recalibration.py
import numpy as np
from sklearn.linear_model import LogisticRegression

import shots
from data import load_matches
from elo import add_elo_features
from leagues import LEAGUES

FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]
K, HA, NEW_START = 30, 75, 1400                      # production settings
LABELS = np.array(["A", "D", "H"])
WINDOWS = [1, 2, 3]                                  # years of recent data used to estimate the shift


def solve_shift(P, Y):
    """Log-probability shift b so that mean(softmax(log P + b)) equals the actual class frequencies."""
    target = np.bincount(Y, minlength=3) / len(Y)
    b = np.zeros(3)
    for _ in range(500):
        q = P * np.exp(b)
        q /= q.sum(axis=1, keepdims=True)
        b += np.log(target / q.mean(axis=0))
    return b


def losses(p, yi):
    return -np.log(np.clip(p[np.arange(len(yi)), yi], 1e-15, 1.0))


def se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def main():
    codes = list(LEAGUES)
    data = {}
    for c in codes:
        df = load_matches(LEAGUES[c]["matches"])
        elo = add_elo_features(df, k=K, home_adv=HA, new_team_start=NEW_START)["elo_diff"].to_numpy()
        X = np.column_stack([elo, shots.add_sot_feature(df).to_numpy()])
        data[c] = (X, df["FTR"].to_numpy(dtype=str), df["Date"].to_numpy())

    base = {c: [None] * 3 for c in codes}
    recal = {W: {c: [None] * 3 for c in codes} for W in WINDOWS}
    shifts = {}
    probs3 = {"current": [], "recal": {W: [] for W in WINDOWS}}
    y3 = []

    for f, (start, end) in enumerate(FOLDS):
        s, e = np.datetime64(start), np.datetime64(end)
        fitted = {}
        for c in codes:
            X, y, d = data[c]
            tr, te = d < s, (d >= s) & (d < e)
            m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            assert list(m.classes_) == list(LABELS)
            fitted[c] = (m, X, y, d, te)

        for W in WINDOWS:
            lo = s - np.timedelta64(365 * W, "D")
            Ps, Ys = [], []
            for c in codes:
                m, X, y, d, _ = fitted[c]
                rec = (d >= lo) & (d < s)
                Ps.append(m.predict_proba(X[rec]))
                Ys.append(np.searchsorted(LABELS, y[rec]))
            b = solve_shift(np.vstack(Ps), np.concatenate(Ys))
            shifts[(f, W)] = b
            for c in codes:
                m, X, y, d, te = fitted[c]
                yi = np.searchsorted(LABELS, y[te])
                p = m.predict_proba(X[te]) * np.exp(b)
                p /= p.sum(axis=1, keepdims=True)
                recal[W][c][f] = losses(p, yi)
                if f == 2:
                    probs3["recal"][W].append(p)

        for c in codes:
            m, X, y, d, te = fitted[c]
            yi = np.searchsorted(LABELS, y[te])
            p = m.predict_proba(X[te])
            base[c][f] = losses(p, yi)
            if f == 2:
                probs3["current"].append(p)
                y3.append(yi)

    def pool(model, folds):
        return np.concatenate([model[c][f] for c in codes for f in folds])

    print("== Self-check: current model, pooled (expect 0.9794 / 0.9936) ==")
    print(f"  folds 1-2: {pool(base, [0, 1]).mean():.4f} | fold 3: {pool(base, [2]).mean():.4f}")

    print("\n== Shift estimated from the last W years (log-prob shift for A / D / H) ==")
    for f in range(3):
        for W in WINDOWS:
            b = shifts[(f, W)]
            print(f"  fold {f + 1}, W={W}: {b[0]:+.3f} / {b[1]:+.3f} / {b[2]:+.3f}")

    print("\n== Choose W on folds 1-2 only (pooled log loss) ==")
    s12 = {W: pool(recal[W], [0, 1]).mean() for W in WINDOWS}
    for W in WINDOWS:
        print(f"  W={W}: {s12[W]:.4f}")
    best = min(s12, key=s12.get)
    print(f"  chosen W = {best}")

    print("\n== Fold 3, recalibrated minus current (negative = better) ==")
    for W in WINDOWS:
        d3 = pool(recal[W], [2]) - pool(base, [2])
        tag = "  <- chosen" if W == best else ""
        print(f"  W={W}: {d3.mean():+.4f} (se {se(d3):.4f}, z {d3.mean() / se(d3):+.2f}){tag}")
    d3 = pool(recal[best], [2]) - pool(base, [2])
    d12 = pool(recal[best], [0, 1]) - pool(base, [0, 1])
    z3 = d3.mean() / se(d3)
    print(f"  chosen W={best}: folds 1-2 {d12.mean():+.4f}")
    print("  per league, fold 3: " + " | ".join(
        f"{c} {(recal[best][c][2] - base[c][2]).mean():+.4f}" for c in codes))

    print("\n== Fold 3 calibration (mean predicted vs actual, A / D / H) ==")
    Y = np.concatenate(y3)
    act = np.bincount(Y, minlength=3) / len(Y)
    for name, plist in (("current", probs3["current"]), (f"recal W={best}", probs3["recal"][best])):
        pm = np.vstack(plist).mean(axis=0)
        print(f"  {name:14s} {pm[0]:.3f} / {pm[1]:.3f} / {pm[2]:.3f}   actual {act[0]:.3f} / {act[1]:.3f} / {act[2]:.3f}")

    ok = z3 <= -2 and d3.mean() <= -0.003 and d12.mean() < 0
    print("\n== Pre-registered rule: fold-3 z <= -2, fold-3 gain >= 0.003, folds 1-2 also negative ==")
    print(f"  recalibration: {'PASSES the rule' if ok else 'does not pass'}")
    print("\nNothing was written to disk.")


if __name__ == "__main__":
    main()