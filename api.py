# api.py
import os
import threading
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from leagues import LEAGUES, DEFAULT
from model import train, predict_match
from table import league_table
from team_page import team_summary
from match_page import match_summary


app = FastAPI(title="Football Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wag9.github.io", "http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_states = {}
_lock = threading.Lock()


def get_state(league):
    """Train a league on first use, then keep it in memory."""
    if league not in LEAGUES:
        raise HTTPException(status_code=404,
                            detail=f"Unknown league '{league}'. Available: {sorted(LEAGUES)}")
    if league not in _states:
        with _lock:
            if league not in _states:
                try:
                    _states[league] = train(league)
                except FileNotFoundError:
                    raise HTTPException(status_code=503,
                                        detail=f"Data for {LEAGUES[league]['name']} is not available yet.")
    return _states[league]


get_state(DEFAULT)      # warm up only the default league at startup


@app.get("/")
def root(league: str = DEFAULT):
    st = get_state(league)
    return {
        "status": "ok",
        "teams": sorted(st["current"]),
        "leagues": [{"code": c, "name": cfg["name"]}
                    for c, cfg in LEAGUES.items() if os.path.exists(cfg["matches"])],
    }


@app.get("/predict")
def predict(home: str, away: str, league: str = DEFAULT):
    st = get_state(league)
    try:
        p = predict_match(st, home, away)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "home": home, "away": away,
        "elo_home": round(st["ratings"][home]),
        "elo_away": round(st["ratings"][away]),
        "prob": {"home": round(p["H"], 4), "draw": round(p["D"], 4), "away": round(p["A"], 4)},
    }


@app.get("/table")
def table(season: Optional[int] = None, league: str = DEFAULT):
    st = get_state(league)
    if season is None:
        season = int(st["df"]["season"].max())
    t = league_table(st["df"], season).reset_index()
    t = t.rename(columns={"index": "Pos"})
    return t.to_dict(orient="records")


@app.get("/fixtures")
def fixtures(status: str = "all", league: str = DEFAULT):
    """status: all | upcoming | played"""
    st = get_state(league)
    log_path = st["log_path"]
    if not os.path.exists(log_path):
        return []

    log = pd.read_csv(log_path, parse_dates=["Date"])

    res = st["df"][["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]]
    m = log.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    m["played"] = m["FTR"].notna()

    if status == "upcoming":
        m = m[~m["played"]]
    elif status == "played":
        m = m[m["played"]]

    m = m.sort_values("Date")
    m["Date"] = m["Date"].dt.strftime("%Y-%m-%d")
    m = m.astype(object).where(m.notna(), None)   # NaN -> None (JSON ke liye)

    return [
        {
            "date": r["Date"], "home": r["HomeTeam"], "away": r["AwayTeam"],
            "prob": {"home": r["pH"], "draw": r["pD"], "away": r["pA"]},
            "played": bool(r["played"]),
            "result": r["FTR"],
            "score": f"{int(r['FTHG'])}-{int(r['FTAG'])}" if r["played"] else None,
        }
        for _, r in m.iterrows()
    ]


@app.get("/record")
def record(league: str = DEFAULT):
    rows = fixtures(status="played", league=league)
    keyOf = {"H": "home", "D": "draw", "A": "away"}
    hits = sum(1 for r in rows if max(r["prob"], key=r["prob"].get) == keyOf[r["result"]])
    return {"played": len(rows), "top_pick_correct": hits}


@app.get("/team")
def team(name: str, league: str = DEFAULT):
    try:
        return team_summary(get_state(league), name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/match")
def match(home: str, away: str, date: Optional[str] = None, league: str = DEFAULT):
    try:
        return match_summary(get_state(league), home, away, date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))