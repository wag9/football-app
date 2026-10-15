import pandas as pd

files = ["prem1920", "prem2021", "prem2122", "prem2223", "prem2324", "prem2425"]

for f in files:
    d = pd.read_csv(f"data/{f}.csv")
    print(f, d.shape, "AvgH" in d.columns)