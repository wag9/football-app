# data.py
import pandas as pd

DATA_PATH = "data/all_matches.csv"


def load_matches(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def current_teams(df):
    latest = df["season"].max()
    s = df[df["season"] == latest]
    return set(s["HomeTeam"]) | set(s["AwayTeam"])