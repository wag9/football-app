# model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_matches, current_teams
from elo import add_elo_features
from shots import add_sot_feature, team_shots_form, SCALE, WINDOW
from leagues import LEAGUES, DEFAULT

# Final config: Elo (K=30, HA=75, NEW=1400) + rolling-10 shots-on-target net diff
K, HOME_ADV, NEW_START = 30, 75, 1400
FEATURES = ["elo_diff", "sot_diff"]

# Label written to the prediction log. Built from the settings above so it cannot drift from the model.
MODEL_VERSION = f"elo_sot{WINDOW} K{K} HA{HOME_ADV} NEW{NEW_START}"


def train(league=DEFAULT, df=None):
    """Ek league ke poore data pe Elo + shots feature + logistic regression train karo."""
    cfg = LEAGUES[league]
    if df is None:
        df = load_matches(cfg["matches"])

    df_elo, ratings = add_elo_features(
        df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START, return_ratings=True
    )
    X = df_elo[["elo_diff"]].copy()
    X["sot_diff"] = add_sot_feature(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X[FEATURES], df["FTR"])

    return {"df": df, "elo_df": df_elo, "ratings": ratings, "model": model,
            "current": current_teams(df), "shots_form": team_shots_form(df),
            "league": league, "log_path": cfg["log"]}


def predict_match(state, home, away):
    """Return {'H': .., 'D': .., 'A': ..} (floats, sum = 1)."""
    if home == away:
        raise ValueError("Home and away teams must be different.")
    for team in (home, away):
        if team not in state["current"]:
            raise ValueError(f"'{team}' not found. Available: {sorted(state['current'])}")

    elo_diff = state["ratings"][home] - state["ratings"][away]
    form = state["shots_form"]
    sot = (form.get(home, np.nan) - form.get(away, np.nan)) * SCALE
    if np.isnan(sot):
        sot = 0.0                                   # same "neutral" rule as training

    X = pd.DataFrame({"elo_diff": [elo_diff], "sot_diff": [sot]})[FEATURES]
    probs = state["model"].predict_proba(X)[0]
    return dict(zip(state["model"].classes_, probs))   # classes_ = A, D, H