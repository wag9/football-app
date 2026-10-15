import pandas as pd
print ("Football Project Started!")

df = pd.read_csv("prem25-26.csv")

teams = df["HomeTeam"].unique()
team_data = {}

for x in teams:
  
 home = df[df["HomeTeam"] == x]
 away = df[df["AwayTeam"] == x]

 total_goals = home["FTHG"].sum() + away["FTAG"].sum()
 total_conceded = home["FTAG"].sum() + away["FTHG"].sum()
 gd = total_goals - total_conceded
 
 home_wins = home[home["FTR"] == "H"]
 away_wins = away[away["FTR"] == "A"]
 total_wins = len(home_wins) + len(away_wins)

 home_draws = home[home["FTR"] == "D"]
 away_draws = away[away["FTR"] == "D"]
 total_draws = len(home_draws) + len(away_draws)

 total_losses = 38 - total_wins - total_draws

 points = total_wins * 3 + total_draws

 team_data[x] = {
        "gf": total_goals,
        "ga": total_conceded,
        "gd" : gd,
        "wins": total_wins,
        "draws": total_draws,
        "losses": total_losses,
        "points": points
        }

table = pd.DataFrame(team_data).T

table = table.sort_values(
    ["points", "gd", "gf"],
    ascending=[False, False, False]
)

table["position"] = range(1, len(table) + 1)

print(table)

match_features = {
    "home_points_before": home_pts,
    "away_points_before": away_pts,
    "home_gf_before": home_gf,
    "away_gf_before": away_gf,
    "home_ga_before": home_ga,
    "away_ga_before": away_ga,
    "target": row["FTR"]  # Our target y!
}