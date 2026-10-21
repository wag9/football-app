"""Tests for data_checks on small synthetic tables. No real data, no files, no network.

Each check gets: a clean table that must pass, and a table with one deliberate fault that must be caught.
"""
import unittest

import numpy as np
import pandas as pd

from data_checks import (check_dates, check_duplicates, check_fixtures, check_odds,
                         check_results, check_season_labels, check_season_shape, check_shots,
                         check_team_names, run_matches_checks)

TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]


def valid_league(season="2324", start="2023-08-05"):
    """Full round-robin for 4 teams: 4*3 = 12 matches, home+away each once, dates in the season window."""
    rows = []
    day = pd.Timestamp(start)
    fixtures = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    for i, (h, a) in enumerate(fixtures):
        hg, ag = [(2, 1), (1, 1), (0, 2)][i % 3]
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        rows.append({
            "Date": (day + pd.Timedelta(days=i * 7)).strftime("%Y-%m-%d"),
            "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag, "FTR": ftr,
            "AvgH": 2.10, "AvgD": 3.40, "AvgA": 3.60,
            "HS": 12, "AS": 9, "HST": 5, "AST": 3, "HC": 4, "AC": 3,
            "season": season,
        })
    return pd.DataFrame(rows)


def levels(issues):
    return {i.check for i in issues if i.level == "ERROR"}


class TestCleanTablePasses(unittest.TestCase):
    def test_valid_league_has_no_errors(self):
        issues = run_matches_checks(valid_league(), today="2026-10-21")
        self.assertEqual([i for i in issues if i.level == "ERROR"], [])


class TestColumnsAndDates(unittest.TestCase):
    def test_missing_column_stops_early(self):
        df = valid_league().drop(columns=["AvgH"])
        self.assertEqual(levels(run_matches_checks(df)), {"columns"})

    def test_unparseable_date_is_error(self):
        df = valid_league()
        df.loc[3, "Date"] = "not-a-date"
        self.assertIn("date_parse", levels(check_dates(df)))

    def test_result_in_the_future_is_error(self):
        df = valid_league()
        df["Date"] = pd.to_datetime(df["Date"]) + pd.Timedelta(days=2000)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        self.assertIn("future_result", levels(check_dates(df, today="2026-10-21")))

    def test_future_row_without_result_is_fine(self):
        df = valid_league()
        df.loc[0, "Date"] = "2099-01-01"
        df.loc[0, "FTR"] = np.nan
        self.assertNotIn("future_result", levels(check_dates(df, today="2026-10-21")))


class TestSeasonLabels(unittest.TestCase):
    def test_correct_label_passes(self):
        self.assertEqual(levels(check_season_labels(valid_league())), set())

    def test_label_a_year_off_is_error(self):
        df = valid_league(season="2223")          # dates are 2023/24
        self.assertEqual(levels(check_season_labels(df)), {"season_vs_date"})

    def test_bad_format_is_error(self):
        df = valid_league(season="23-24")
        self.assertEqual(levels(check_season_labels(df)), {"season_format"})

    def test_leading_zero_season_survives_as_string(self):
        # "0304" must stay a string; read as a number it becomes 304 and fails the format check.
        df = valid_league(season="0304", start="2003-08-16")
        self.assertEqual(levels(check_season_labels(df)), set())

    def test_leading_zero_season_read_as_int_is_caught(self):
        df = valid_league(season="0304", start="2003-08-16")
        df["season"] = df["season"].astype(int)   # what a careless read_csv would do
        self.assertIn("season_format", levels(check_season_labels(df)))

    def test_covid_season_ending_in_august_is_allowed(self):
        df = valid_league(season="1920", start="2019-09-01")
        df.loc[df.index[-1], "Date"] = "2020-08-02"   # last round played in August 2020, as in the real COVID season
        self.assertEqual(levels(check_season_labels(df)), set())


class TestDuplicatesAndTeams(unittest.TestCase):
    def test_duplicate_fixture_is_error(self):
        df = pd.concat([valid_league(), valid_league().iloc[[0]]], ignore_index=True)
        self.assertIn("duplicate_match", levels(check_duplicates(df)))

    def test_home_equals_away_is_error(self):
        df = valid_league()
        df.loc[0, "AwayTeam"] = df.loc[0, "HomeTeam"]
        self.assertIn("home_equals_away", levels(check_duplicates(df)))

    def test_team_twice_same_day_is_error(self):
        df = valid_league()
        df.loc[1, "Date"] = df.loc[0, "Date"]    # Alpha plays two matches that day
        self.assertIn("team_twice_same_day", levels(check_duplicates(df)))

    def test_whitespace_in_name_is_error(self):
        df = valid_league()
        df.loc[0, "HomeTeam"] = "Alpha "
        self.assertIn("team_whitespace", levels(check_team_names(df)))

    def test_case_and_punctuation_variant_is_error(self):
        df = valid_league()
        df.loc[0, "HomeTeam"] = "alpha."
        self.assertIn("team_name_variants", levels(check_team_names(df)))

    def test_one_season_team_is_info_not_error(self):
        df = valid_league()
        df.loc[0, "HomeTeam"] = "Epsilon"
        issues = check_team_names(df)
        self.assertEqual(levels(issues), set())
        self.assertTrue(any(i.check == "teams_seen_one_season" for i in issues))


class TestResults(unittest.TestCase):
    def test_fractional_goals_are_error(self):
        df = valid_league()
        df["FTHG"] = df["FTHG"].astype(float)
        df.loc[0, "FTHG"] = 1.5
        self.assertIn("FTHG_invalid", levels(check_results(df)))

    def test_negative_goals_are_error(self):
        df = valid_league()
        df.loc[0, "FTAG"] = -1
        self.assertIn("FTAG_invalid", levels(check_results(df)))

    def test_ftr_contradicting_goals_is_error(self):
        df = valid_league()
        df.loc[0, "FTR"] = "A"                    # goals say H (2-1)
        self.assertIn("FTR_vs_goals", levels(check_results(df)))

    def test_bad_ftr_value_is_error(self):
        df = valid_league()
        df.loc[0, "FTR"] = "X"
        self.assertIn("FTR_values", levels(check_results(df)))


class TestOdds(unittest.TestCase):
    def test_odds_at_or_below_one_is_error(self):
        df = valid_league()
        df.loc[0, "AvgH"] = 1.0
        self.assertIn("odds_le_1", levels(check_odds(df)))

    def test_normal_overround_passes(self):
        df = valid_league()                       # 1/2.10 + 1/3.40 + 1/3.60 ~ 1.06
        self.assertEqual(check_odds(df), [])

    def test_extreme_overround_is_warn(self):
        df = valid_league()
        df.loc[0, ["AvgH", "AvgD", "AvgA"]] = [1.5, 1.5, 1.5]   # sum 2.0
        issues = check_odds(df)
        self.assertEqual(levels(issues), set())
        self.assertIn("overround_range", {i.check for i in issues if i.level == "WARN"})

    def test_missing_odds_row_is_warn_and_not_zero(self):
        df = valid_league()
        df.loc[0, "AvgD"] = np.nan
        issues = check_odds(df)
        self.assertIn("odds_missing", {i.check for i in issues})
        self.assertNotIn("overround_range", {i.check for i in issues})   # NaN row must not become book = 0


class TestShots(unittest.TestCase):
    def test_shots_on_target_above_shots_is_warn_not_error(self):
        df = valid_league()
        df.loc[0, "HST"] = 15                     # more on target than HS=12
        issues = check_shots(df)
        self.assertEqual(levels(issues), set())
        self.assertIn("HST_gt_HS", {i.check for i in issues})

    def test_missing_shots_is_info(self):
        df = valid_league()
        df.loc[0, ["HST", "AST"]] = np.nan
        issues = check_shots(df)
        self.assertEqual(levels(issues), set())
        self.assertIn("shots_missing", {i.check for i in issues})

    def test_only_one_of_hst_ast_is_warn(self):
        df = valid_league()
        df.loc[0, "AST"] = np.nan
        self.assertIn("HST_AST_partial", {i.check for i in check_shots(df)})

    def test_negative_shots_is_error(self):
        df = valid_league()
        df.loc[0, "HS"] = -2
        self.assertIn("home_shots_negative", levels(check_shots(df)))


class TestSeasonShape(unittest.TestCase):
    def test_complete_round_robin_passes(self):
        self.assertEqual(check_season_shape(valid_league()), [])

    def test_missing_match_is_info_incomplete(self):
        df = valid_league().iloc[:-1]
        issues = check_season_shape(df)
        self.assertEqual(levels(issues), set())
        self.assertIn("season_incomplete", {i.check for i in issues})

    def test_too_many_matches_is_error(self):
        df = pd.concat([valid_league(), valid_league().iloc[[0]]], ignore_index=True)
        self.assertIn("season_too_many_matches", levels(check_season_shape(df)))

    def test_right_count_but_unbalanced_is_error(self):
        df = valid_league()
        # Alpha at home 3 times and never away; Beta takes that away slot instead. Still 12 matches.
        df.loc[(df.HomeTeam == "Beta") & (df.AwayTeam == "Alpha"), "AwayTeam"] = "Beta"
        df.loc[(df.HomeTeam == "Alpha") & (df.AwayTeam == "Beta"), "HomeTeam"] = "Beta"
        df.loc[(df.HomeTeam == "Beta") & (df.AwayTeam == "Alpha"), "HomeTeam"] = "Alpha"
        self.assertIn("season_unbalanced", levels(check_season_shape(df)))


class TestFixtures(unittest.TestCase):
    def fx(self, **over):
        row = {"Div": "E0", "Date": "21/10/2026", "Time": "18:00",
               "HomeTeam": "Alpha", "AwayTeam": "Beta"}
        row.update(over)
        return pd.DataFrame([row])

    def test_known_teams_pass(self):
        self.assertEqual(levels(check_fixtures(self.fx(), valid_league())), set())

    def test_unknown_team_is_error(self):
        self.assertIn("fixtures_unknown_HomeTeam",
                      levels(check_fixtures(self.fx(HomeTeam="Alfa"), valid_league())))

    def test_odd_time_is_warn(self):
        issues = check_fixtures(self.fx(Time="TBC"), valid_league())
        self.assertIn("fixtures_time", {i.check for i in issues if i.level == "WARN"})

    def test_bad_fixture_date_is_error(self):
        self.assertIn("fixtures_date_parse",
                      levels(check_fixtures(self.fx(Date="someday"), valid_league())))


if __name__ == "__main__":
    unittest.main()