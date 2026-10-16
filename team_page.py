from pathlib import Path

import pandas as pd

LOG_PATH = Path("data/predictions_log.csv")


def _match_row(r, team):
    home = r["HomeTeam"] == team
    gf, ga = (r["FTHG"], r["FTAG"]) if home else (r["FTAG"], r["FTHG"])
    result = "W" if gf > ga else ("D" if gf == ga else "L")
    return {
        "date": r["Date"].strftime("%Y-%m-%d"),
        "opponent": r["AwayTeam"] if home else r["HomeTeam"],
        "venue": "H" if home else "A",
        "gf": int(gf),
        "ga": int(ga),
        "result": result,
    }


def team_summary(state, name):
    df = state["df"]
    ratings = state["ratings"]
    current = list(state["current"])

    if name not in current:
        raise ValueError(f"Unknown team: {name}")

    season = int(df["season"].max())

    # --- Elo rank among current teams ---
    order = sorted(current, key=lambda t: ratings[t], reverse=True)
    elo_rank = order.index(name) + 1

    # --- all played matches of this team (oldest -> newest) ---
    mask = (df["HomeTeam"] == name) | (df["AwayTeam"] == name)
    played = df[mask].dropna(subset=["FTHG", "FTAG"]).sort_values("Date")

    # --- this season ---
    season_rows = [_match_row(r, name) for _, r in played[played["season"] == season].iterrows()]
    w = sum(m["result"] == "W" for m in season_rows)
    d = sum(m["result"] == "D" for m in season_rows)
    l = sum(m["result"] == "L" for m in season_rows)
    gf = sum(m["gf"] for m in season_rows)
    ga = sum(m["ga"] for m in season_rows)
    record = {"played": len(season_rows), "W": w, "D": d, "L": l,
              "GF": gf, "GA": ga, "Pts": 3 * w + d}

    # --- last 5 (across seasons) ---
    last5 = [_match_row(r, name) for _, r in played.tail(5).iterrows()][::-1]

    # --- predictions log for this team (with result if already played) ---
    predictions = []
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        log = log[(log["HomeTeam"] == name) | (log["AwayTeam"] == name)]
        results = {}
        for _, r in df.dropna(subset=["FTHG", "FTAG"]).iterrows():
            if r["HomeTeam"] == name or r["AwayTeam"] == name:
                key = (r["Date"].strftime("%Y-%m-%d"), r["HomeTeam"], r["AwayTeam"])
                results[key] = f"{int(r['FTHG'])}-{int(r['FTAG'])}"
        for _, r in log.sort_values("Date").iterrows():
            key = (str(r["Date"])[:10], r["HomeTeam"], r["AwayTeam"])
            predictions.append({
                "date": key[0],
                "home": r["HomeTeam"],
                "away": r["AwayTeam"],
                "pH": float(r["pH"]),
                "pD": float(r["pD"]),
                "pA": float(r["pA"]),
                "score": results.get(key),  # None if not played yet
            })

    return {
        "team": name,
        "season": season,
        "elo": round(float(ratings[name]), 1),
        "elo_rank": elo_rank,
        "n_teams": len(current),
        "record": record,
        "last5": last5,
        "season_results": season_rows[::-1],
        "predictions": predictions,
    }