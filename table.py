import pandas as pd


def league_table(df, season):
    s = df[df["season"] == season]

    # Har match ki 2 rows: ek home team ki, ek away team ki
    home = pd.DataFrame({
        "Team": s["HomeTeam"], "GF": s["FTHG"], "GA": s["FTAG"],
        "Pts": s["FTR"].map({"H": 3, "D": 1, "A": 0}),
    })
    away = pd.DataFrame({
        "Team": s["AwayTeam"], "GF": s["FTAG"], "GA": s["FTHG"],
        "Pts": s["FTR"].map({"A": 3, "D": 1, "H": 0}),
    })
    both = pd.concat([home, away])

    # Pts 3/1/0 se W/D/L flags
    both["W"] = (both["Pts"] == 3).astype(int)
    both["D"] = (both["Pts"] == 1).astype(int)
    both["L"] = (both["Pts"] == 0).astype(int)

    table = both.groupby("Team").agg(
        P=("Pts", "size"), W=("W", "sum"), D=("D", "sum"), L=("L", "sum"),
        GF=("GF", "sum"), GA=("GA", "sum"), Pts=("Pts", "sum"),
    )
    table["GD"] = table["GF"] - table["GA"]

    table = table.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index()
    table.index = table.index + 1          # rank 1 se shuru
    return table[["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


if __name__ == "__main__":
    df = pd.read_csv("data/all_matches.csv")
    print(league_table(df, 2526))