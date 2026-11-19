# model_comparison.py -- current logistic vs ordered logit vs gradient boosting.
# Read-only: changes no data, no model, no log. Run from the repo root: python model_comparison.py
import itertools
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import shots
from data import load_matches
from elo import add_elo_features
from leagues import LEAGUES

FOLDS = [("2023-08-01", "2024-08-01"), ("2024-08-01", "2025-08-01"), ("2025-08-01", "2026-08-01")]
K, HA, NEW_START = 30, 75, 1400                      # production settings
LABELS = np.array(["A", "D", "H"])
GB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=n, min_samples_leaf=leaf)
           for d, lr, n, leaf in itertools.product([2, 3], [0.03, 0.1], [50, 150, 300], [100, 300])]


class OrderedLogit:
    """P(A)=s(t1-eta), P(A or D)=s(t2-eta), eta = Z.b. Classes in the order A < D < H."""
    classes_ = LABELS

    def fit(self, X, y):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        yi = np.searchsorted(LABELS, y)
        fa, fd = (yi == 0).mean(), (yi == 1).mean()
        t1 = np.log(fa / (1 - fa))
        t2 = np.log((fa + fd) / (1 - fa - fd))
        start = np.concatenate([np.zeros(X.shape[1]), [t1, np.log(t2 - t1)]])

        def nll(p):
            pr = self._probs(p, Z)
            return -np.log(np.clip(pr[np.arange(len(yi)), yi], 1e-12, 1.0)).mean()

        res = minimize(nll, start, method="L-BFGS-B")
        assert res.success, res.message
        self.params = res.x
        return self

    @staticmethod
    def _probs(p, Z):
        eta = Z @ p[:-2]
        t1 = p[-2]
        t2 = t1 + np.exp(p[-1])
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        return np.column_stack([c1, c2 - c1, 1 - c2])

    def predict_proba(self, X):
        return self._probs(self.params, (X - self.mu) / self.sd)


def make_logit():
    return LogisticRegression(max_iter=1000)


def make_gb(cfg):
    return lambda: HistGradientBoostingClassifier(early_stopping=False, random_state=0, **cfg)


def run(make, X, y, dates):
    """Return per fold: (per-match log loss, probability matrix)."""
    out = []
    for start, end in FOLDS:
        s, e = np.datetime64(start), np.datetime64(end)
        tr, te = dates < s, (dates >= s) & (dates < e)
        m = make().fit(X[tr], y[tr])
        assert list(m.classes_) == list(LABELS), m.classes_
        p = m.predict_proba(X[te])
        yi = np.searchsorted(LABELS, y[te])
        out.append((-np.log(np.clip(p[np.arange(len(yi)), yi], 1e-15, 1.0)), p, yi))
    return out


def se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def main():
    t0 = time.time()
    codes = list(LEAGUES)
    data = {}
    for c in codes:
        df = load_matches(LEAGUES[c]["matches"])
        elo = add_elo_features(df, k=K, home_adv=HA, new_team_start=NEW_START)["elo_diff"].to_numpy()
        X = np.column_stack([elo, shots.add_sot_feature(df).to_numpy()])
        data[c] = (X, df["FTR"].to_numpy(dtype=str), df["Date"].to_numpy())

    res = {"logit": {}, "ordered": {}}
    for c in codes:
        res["logit"][c] = run(make_logit, *data[c])
        res["ordered"][c] = run(lambda: OrderedLogit(), *data[c])
    print(f"fitted logit and ordered logit ({time.time() - t0:.0f}s)", flush=True)

    gb = {}
    for i, cfg in enumerate(GB_GRID):
        key = tuple(cfg.values())
        gb[key] = {c: run(make_gb(cfg), *data[c]) for c in codes}
        print(f"  gb config {i + 1}/{len(GB_GRID)} done ({time.time() - t0:.0f}s)", flush=True)

    def pool(model, folds):
        return np.concatenate([model[c][f][0] for c in codes for f in folds])

    base = res["logit"]
    print("\n== Self-check: current model, pooled (selection_bias.py gave 0.9794 / 0.9936) ==")
    print(f"  folds 1-2: {pool(base, [0, 1]).mean():.4f} | fold 3: {pool(base, [2]).mean():.4f}")

    # ---- gradient boosting: tune on folds 1-2 only ----
    s12 = {k: pool(v, [0, 1]).mean() for k, v in gb.items()}
    best = min(s12, key=s12.get)
    names = list(GB_GRID[0].keys())
    print("\n== Gradient boosting: tuned on folds 1-2 ==")
    print("  chosen: " + ", ".join(f"{n}={v}" for n, v in zip(names, best)) + f" | folds 1-2: {s12[best]:.4f}")
    res["gb"] = gb[best]
    s3 = {k: pool(v, [2]).mean() for k, v in gb.items()}
    rank = sorted(s3, key=s3.get).index(best) + 1
    print(f"  fold 3 of the chosen config: {s3[best]:.4f} (rank {rank} of {len(GB_GRID)} on fold 3; "
          f"median config {np.median(list(s3.values())):.4f})")

    # ---- comparison with the current model ----
    print("\n== Candidate minus current (negative = candidate better), pooled over 5 leagues ==")
    verdict = {}
    for name in ("ordered", "gb"):
        d3 = pool(res[name], [2]) - pool(base, [2])
        z3 = d3.mean() / se(d3)
        line = f"  {name:8s} fold 3: {d3.mean():+.4f} (se {se(d3):.4f}, z {z3:+.2f}, n={len(d3)})"
        d12 = pool(res[name], [0, 1]) - pool(base, [0, 1])
        tag = "" if name == "ordered" else "  [tuning data, not a test]"
        line += f" | folds 1-2: {d12.mean():+.4f}{tag}"
        print(line)
        per = " | ".join(f"{c} {(res[name][c][2][0] - base[c][2][0]).mean():+.4f}" for c in codes)
        print(f"           per league, fold 3: {per}")
        ok = z3 <= -2 and (name == "gb" or d12.mean() < 0)
        verdict[name] = ok

    # ---- diagnostics on fold 3 ----
    print("\n== Fold 3 diagnostics (5 leagues pooled) ==")
    for name, model in (("current", base), ("ordered", res["ordered"]), ("gb", res["gb"])):
        P = np.vstack([model[c][2][1] for c in codes])
        Y = np.concatenate([model[c][2][2] for c in codes])
        top = P.argmax(axis=1)
        pred = P.mean(axis=0)
        act = np.bincount(Y, minlength=3) / len(Y)
        print(f"  {name:8s} mean pred A/D/H {pred[0]:.3f}/{pred[1]:.3f}/{pred[2]:.3f} "
              f"| actual {act[0]:.3f}/{act[1]:.3f}/{act[2]:.3f} "
              f"| max P(draw) {P[:, 1].max():.3f} | draw is top pick {(top == 1).sum()} of {len(Y)}")

    print("\n== Pre-registered rule: fold-3 z <= -2 (and folds 1-2 negative for ordered logit) ==")
    for name, ok in verdict.items():
        print(f"  {name:8s}: {'PASSES the rule' if ok else 'does not pass'}")
    print(f"\nfinished in {time.time() - t0:.0f}s. Nothing was written to disk.")


if __name__ == "__main__":
    main()