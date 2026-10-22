from table import league_table
from predict import ratings, df

t = league_table(df, df["season"].max())
t["Elo"] = t["Team"].map(ratings).round().astype(int)
t["Elo_rank"] = t["Elo"].rank(ascending=False, method="min").astype(int)
t["Gap"] = t["Elo_rank"] - t.index      # + matlab Elo us team ko table se neeche rakhta hai 

print(t[["Team", "Pts", "Elo", "Elo_rank", "Gap"]])