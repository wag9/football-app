# model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data import load_matches, current_teams
from elo import add_elo_features

# Final config (walk-forward avg ~0.983)
K, HOME_ADV, NEW_START = 30, 75, 1400


def train(df=None):
    """Poore data pe Elo + logistic regression train karo. State dict return karta hai."""
    if df is None:
        df = load_matches()

    df_elo, ratings = add_elo_features(
        df, k=K, home_adv=HOME_ADV, new_team_start=NEW_START, return_ratings=True
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(df_elo[["elo_diff"]], df["FTR"])

    return {"df": df, "ratings": ratings, "model": model, "current": current_teams(df)}


def predict_match(state, home, away):
    """Return {'H': .., 'D': .., 'A': ..} (floats, sum = 1)."""
    if home == away:
        raise ValueError("Home and away teams must be different.")
    for team in (home, away):
        if team not in state["current"]:
            raise ValueError(f"'{team}' not found. Available: {sorted(state['current'])}")

    diff = state["ratings"][home] - state["ratings"][away]
    probs = state["model"].predict_proba(pd.DataFrame({"elo_diff": [diff]}))[0]
    return dict(zip(state["model"].classes_, probs))   # classes_ = A, D, H