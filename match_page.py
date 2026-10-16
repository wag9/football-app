import numpy as np
import pandas as pd

from model import predict_match
from team_page import _match_row, LOG_PATH


def _clean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 2)


def _form(df, team, n=5):
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    played = df[mask].dropna(subset=["FTHG", "FTAG"]).sort_values("Date")
    return [_match_row(r, team) for _, r in played.tail(n).iterrows()][::-1]


def _h2h(df, home, away, n=5):
    m = (((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
         ((df["HomeTeam"] == away) & (df["AwayTeam"] == home)))
    played = df[m].dropna(subset=["FTHG", "FTAG"]).sort_values("Date").tail(n)
    rows, hw, dr, aw = [], 0, 0, 0
    for _, r in played.iterrows():
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        if hg == ag:
            dr += 1
        else:
            winner = r["HomeTeam"] if hg > ag else r["AwayTeam"]
            if winner == home:
                hw += 1
            else:
                aw += 1
        rows.append({"date": r["Date"].strftime("%Y-%m-%d"),
                     "home": r["HomeTeam"], "away": r["AwayTeam"],
                     "score": f"{hg}-{ag}"})
    return {"matches": rows[::-1], "home_team_wins": hw, "draws": dr, "away_team_wins": aw}


def match_summary(state, home, away, date=None):
    p = predict_match(state, home, away)          # raises ValueError for bad/equal teams
    df, ratings = state["df"], state["ratings"]
    order = sorted(state["current"], key=lambda t: ratings[t], reverse=True)

    logged, result = None, None
    if date:
        d = pd.Timestamp(date)                    # bad date -> ValueError -> 400
        if LOG_PATH.exists():
            log = pd.read_csv(LOG_PATH)
            hit = log[(log["Date"].astype(str).str[:10] == date) &
                      (log["HomeTeam"] == home) & (log["AwayTeam"] == away)]
            if len(hit):
                r = hit.iloc[0]
                logged = {"prob": {"home": float(r["pH"]), "draw": float(r["pD"]), "away": float(r["pA"])},
                          "model": str(r["model"]) if "model" in log.columns else None,
                          "logged_on": str(r["logged_on"]) if "logged_on" in log.columns else None}
        res = df[(df["Date"] == d) & (df["HomeTeam"] == home) &
                 (df["AwayTeam"] == away)].dropna(subset=["FTHG", "FTAG"])
        if len(res):
            result = f"{int(res.iloc[0]['FTHG'])}-{int(res.iloc[0]['FTAG'])}"

    form = state["shots_form"]
    return {
        "home": home, "away": away, "date": date,
        "elo_home": round(float(ratings[home])), "elo_away": round(float(ratings[away])),
        "rank_home": order.index(home) + 1, "rank_away": order.index(away) + 1,
        "n_teams": len(order),
        "prob": {"home": float(p["H"]), "draw": float(p["D"]), "away": float(p["A"])},
        "logged": logged,
        "result": result,
        "form_home": _form(df, home), "form_away": _form(df, away),
        "h2h": _h2h(df, home, away),
        "sot_home": _clean(form.get(home)), "sot_away": _clean(form.get(away)),
    }