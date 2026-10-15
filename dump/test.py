import pandas as pd
x = pd.Series([3, 0, 1, 3, 3, 0])
print(x.shift(1).rolling(window=3, min_periods=1).mean())