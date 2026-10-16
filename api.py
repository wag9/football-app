# api.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from model import train, predict_match
from table import league_table
from fastapi.middleware.cors import CORSMiddleware
from team_page import team_summary
from match_page import match_summary


app = FastAPI(title="Football Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wag9.github.io", "http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
state = train()          # server start hote hi ek baar train hota hai


@app.get("/")
def root():
    return {"status": "ok", "teams": sorted(state["current"])}


@app.get("/predict")
def predict(home: str, away: str):
    try:
        p = predict_match(state, home, away)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "home": home, "away": away,
        "elo_home": round(state["ratings"][home]),
        "elo_away": round(state["ratings"][away]),
        "prob": {"home": round(p["H"], 4), "draw": round(p["D"], 4), "away": round(p["A"], 4)},
    }


@app.get("/table")
def table(season: int = 2627):
    t = league_table(state["df"], season).reset_index()
    t = t.rename(columns={"index": "Pos"})
    return t.to_dict(orient="records")

LOG_PATH = "data/predictions_log.csv"


@app.get("/fixtures")
def fixtures(status: str = "all"):
    """status: all | upcoming | played"""
    if not os.path.exists(LOG_PATH):
        return []

    log = pd.read_csv(LOG_PATH, parse_dates=["Date"])

    # result jodo: jo match all_matches mein hai wo played hai
    res = state["df"][["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]]
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
def record():
    rows = fixtures(status="played")
    keyOf = {"H": "home", "D": "draw", "A": "away"}
    hits = sum(1 for r in rows if max(r["prob"], key=r["prob"].get) == keyOf[r["result"]])
    return {"played": len(rows), "top_pick_correct": hits}

@app.get("/team")
def team(name: str):
    try:
        return team_summary(state, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/match")
def match(home: str, away: str, date: str = None):
    try:
        return match_summary(state, home, away, date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))