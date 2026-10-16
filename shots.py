# shots.py -- rolling shots-on-target feature (shared by training and prediction)
import numpy as np
import pandas as pd

WINDOW = 10
MIN_PERIODS = 3
SCALE = 100          # brings sot_diff to a scale comparable with elo_diff


def _team_log(df):
    """Two rows per match (home view, away view) with net = SoT for - SoT against."""
    h = pd.DataFrame({"idx": df.index, "Date": df["Date"], "Team": df["HomeTeam"],
                      "net": df["HST"] - df["AST"], "side": "h"})
    a = pd.DataFrame({"idx": df.index, "Date": df["Date"], "Team": df["AwayTeam"],
                      "net": df["AST"] - df["HST"], "side": "a"})
    return pd.concat([h, a]).sort_values(["Team", "Date"], kind="stable")


def add_sot_feature(df):
    """sot_diff per match row (aligned to df.index). shift(1) = no leakage."""
    st = _team_log(df)
    st["roll"] = st.groupby("Team")["net"].transform(
        lambda x: x.shift(1).rolling(window=WINDOW, min_periods=MIN_PERIODS).mean()
    )
    hr = st[st["side"] == "h"].set_index("idx")["roll"].reindex(df.index)
    ar = st[st["side"] == "a"].set_index("idx")["roll"].reindex(df.index)
    return ((hr - ar).fillna(0) * SCALE).rename("sot_diff")


def team_shots_form(df):
    """Latest rolling-10 net SoT per team (includes the most recent match) for upcoming fixtures."""
    st = _team_log(df)
    form = {}
    for team, g in st.groupby("Team"):
        last = g["net"].dropna().tail(WINDOW)
        form[team] = float(last.mean()) if len(last) >= MIN_PERIODS else np.nan
    return form