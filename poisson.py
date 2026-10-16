# poisson.py -- attack/defence Poisson model with time-decay weights
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss

LABELS = ["A", "D", "H"]
MAX_GOALS = 10


def build_X(home_teams, away_teams, teams):
    """2 rows per match: [home team scoring] stacked on [away team scoring].
    Columns: attack (T) | defence (T) | home flag (1)."""
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    h = np.array([idx[t] for t in home_teams])
    a = np.array([idx[t] for t in away_teams])
    n = len(h)
    rows = np.arange(n)
    r = np.concatenate([rows, rows, rows, rows + n, rows + n])
    c = np.concatenate([h, T + a, np.full(n, 2 * T), a, T + h])
    return sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(2 * n, 2 * T + 1)).tocsr()


def outcome_probs(lam_h, lam_a):
    """Expected goals -> [P(A), P(D), P(H)] for each match."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lam_h[:, None])
    pa = poisson.pmf(g[None, :], lam_a[:, None])
    joint = ph[:, :, None] * pa[:, None, :]          # (match, home goals, away goals)
    hg, ag = g[:, None], g[None, :]
    pH = (joint * (hg > ag)).sum(axis=(1, 2))
    pD = (joint * (hg == ag)).sum(axis=(1, 2))
    pA = (joint * (hg < ag)).sum(axis=(1, 2))
    P = np.column_stack([pA, pD, pH])
    return P / P.sum(axis=1, keepdims=True)           # fix tiny truncation error


def walk_forward_poisson(matches, base, folds, alpha=0.01, halflife_days=730, refit_days=28):
    matches = matches.dropna(subset=["FTHG", "FTAG"]).reset_index(drop=True)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    X = build_X(matches["HomeTeam"], matches["AwayTeam"], teams)
    y = np.concatenate([matches["FTHG"].values, matches["FTAG"].values])
    dates = np.concatenate([matches["Date"].values, matches["Date"].values])

    scores, all_probs = [], []
    for start, end in folds:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        test = base[(base["Date"] >= s) & (base["Date"] < e)].reset_index(drop=True)
        probs = np.zeros((len(test), 3))

        cuts = list(pd.date_range(s, e, freq=f"{refit_days}D"))
        if cuts[-1] != e:
            cuts.append(e)

        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            pos = np.where((test["Date"] >= c0) & (test["Date"] < c1))[0]
            if len(pos) == 0:
                continue
            mask = dates < c0.to_datetime64()                      # only matches before c0
            age = (c0.to_datetime64() - dates[mask]) / np.timedelta64(1, "D")
            w = 0.5 ** (age / halflife_days)                       # time decay
            m = PoissonRegressor(alpha=alpha, max_iter=500)
            m.fit(X[mask], y[mask], sample_weight=w)

            chunk = test.iloc[pos]
            mu = m.predict(build_X(chunk["HomeTeam"], chunk["AwayTeam"], teams))
            n = len(chunk)
            probs[pos] = outcome_probs(mu[:n], mu[n:])             # first n = home goals, last n = away goals

        scores.append(log_loss(test["target"], probs, labels=LABELS))
        all_probs.append(probs)
        print(f"  fold {start}: test rows {len(test)}")
    return scores, np.vstack(all_probs)