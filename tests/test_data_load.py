"""load_matches must keep the season label as text ("0304", not 304). Uses a tiny temp CSV."""
import os
import tempfile
import unittest

import pandas as pd

from data import load_matches

CSV = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,season
2003-08-16,A,B,1,0,H,0304
2003-08-17,C,D,0,0,D,0304
2009-08-22,A,C,2,1,H,0910
"""


class TestLoadMatches(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write(CSV)

    def tearDown(self):
        os.remove(self.path)

    def test_season_stays_text_with_leading_zero(self):
        df = load_matches(self.path)
        self.assertEqual(sorted(df["season"].unique()), ["0304", "0910"])
        self.assertTrue(all(isinstance(s, str) for s in df["season"]))

    def test_dates_parsed_and_sorted(self):
        df = load_matches(self.path)
        self.assertTrue(df["Date"].is_monotonic_increasing)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))


if __name__ == "__main__":
    unittest.main()
