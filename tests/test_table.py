"""league_table must not silently return an empty table when the season is passed as int or text.
Regression test: the season column is text ("2627"); an int 2627 used to match nothing, so the /table page was blank."""
import unittest

import pandas as pd

from table import league_table
from tests.test_data_checks import valid_league


def two_seasons():
    a = valid_league(season="2324")
    b = valid_league(season="0304", start="2003-08-16")
    return pd.concat([a, b], ignore_index=True)


class TestLeagueTable(unittest.TestCase):
    def test_text_season_gives_a_full_table(self):
        t = league_table(two_seasons(), "2324")
        self.assertEqual(len(t), 4)                       # 4 teams
        self.assertEqual(int(t["P"].sum()), 24)           # 12 matches, 2 rows each

    def test_text_season_with_leading_zero(self):
        t = league_table(two_seasons(), "0304")
        self.assertEqual(int(t["P"].sum()), 24)

    def test_int_argument_still_matches_a_text_season_with_no_leading_zero(self):
        # Callers should pass text, but an int like 2324 must not silently give an empty table.
        t = league_table(two_seasons(), 2324)
        self.assertEqual(int(t["P"].sum()), 24)

    def test_unknown_season_is_empty_not_an_error(self):
        t = league_table(two_seasons(), "9999")
        self.assertEqual(len(t), 0)

    def test_max_of_text_seasons_is_the_latest(self):
        # The default season in /table is df["season"].max(). It must be the latest, which text compare gives
        # for 4-digit labels of the same width.
        df = two_seasons()
        self.assertEqual(df["season"].max(), "2324")


if __name__ == "__main__":
    unittest.main()