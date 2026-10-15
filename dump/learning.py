import pandas as pd
print ("Football Project Started!")

df = pd.read_csv("prem25-26.csv")
print(df)
print(df.head())        #first 5 rows
print(df.shape)         #gives us (rows, columns)
print(df.columns)
print(df["HomeTeam"].unique())
print(df["HomeTeam"].value_counts())
print(df["AwayTeam"].value_counts())

liverpool_home = df[df["HomeTeam"] == "Liverpool"]
print(liverpool_home)

arsenal_home = df[df["HomeTeam"] == "Arsenal"]
print(arsenal_home)
print(arsenal_home["FTHG"].sum())
arsenal_away = df[df["AwayTeam"] == "Arsenal"]
print(arsenal_away["FTAG"].sum())
arsenal_total_goals = arsenal_home["FTHG"].sum() + arsenal_away["FTAG"].sum()
print(arsenal_total_goals)

team_data = {}
print(team_data)
print(team_data["Arsenal"])
print(team_data["Arsenal"]["goals"])

table = pd.DataFrame(team_data).T
table["gd"] = table["goals"] - table["conceded"]
print(df["FTR"].value_counts())
print(table)

arsenal_home = df[df["HomeTeam"] == "Arsenal"]
arsenal_away = df[df["AwayTeam"] == "Arsenal"]  

arsenal_home_wins = arsenal_home[arsenal_home["FTR"] == "H"]
print(arsenal_home_wins)

arsenal_away_wins = arsenal_away[arsenal_away["FTR"] == "A"]
print(arsenal_away_wins)

arsenal_home_draws = arsenal_home[arsenal_home["FTR"] == "D"]
arsenal_away_draws = arsenal_away[arsenal_away["FTR"] == "D"]
print(len(arsenal_home_draws))
print(len(arsenal_away_draws))

arsenal_points = 26 * 3 + 7
print(arsenal_points)

past_matches = df[df["Date"] < "2025-10-25"]        # ensures no data-leakage 
past_arsenal_home = past_matches[past_matches["HomeTeam"] == "Arsenal"]
past_arsenal_away = past_matches[past_matches["AwayTeam"] == "Arsenal"]
past_goals_scored = (
    past_arsenal_home["FTHG"].sum() + past_arsenal_away["FTAG"].sum()
)
past_goals_conceded = (
    past_arsenal_home["FTAG"].sum() + past_arsenal_away["FTHG"].sum()
)