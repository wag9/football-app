"""Synthetic tests for shots.py and elo.py. No real data."""
import unittest

import numpy as np
import pandas as pd

import elo
import shots


def matches(rows, season="2324"):
    """rows: (Home, Away, FTHG, FTAG, HST, AST). Dates one day apart, in order."""
    out = []
    day = pd.Timestamp("2023-08-01")
    for i, (h, a, hg, ag, hst, ast) in enumerate(rows):
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")
        out.append({"Date": day + pd.Timedelta(days=i), "HomeTeam": h, "AwayTeam": a,
                    "FTHG": hg, "FTAG": ag, "FTR": ftr, "HST": hst, "AST": ast, "season": season})
    return pd.DataFrame(out)


def team_run(team, n, hst=5, ast=3, opp="Opp"):
    """n matches where `team` is home and always has hst shots on target vs ast conceded."""
    return [(team, f"{opp}{i}", 1, 0, hst, ast) for i in range(n)]


class TestSotLog(unittest.TestCase):
    def test_two_rows_per_match_with_opposite_net(self):
        df = matches([("A", "B", 1, 0, 6, 2)])
        log = shots._team_log(df)
        self.assertEqual(len(log), 2)
        net = dict(zip(log["side"], log["net"]))
        self.assertEqual(net["h"], 4)      # A: 6 on target, 2 conceded
        self.assertEqual(net["a"], -4)     # B: the mirror


class TestSotFeature(unittest.TestCase):
    def test_scale_and_sign(self):
        # A's net is +2 every match, B's is -2 (mirror). From the 4th match on: home roll +2, away roll -2,
        # so sot_diff = (2 - (-2)) * SCALE.
        rows = [("A", "B", 1, 0, 5, 3)] * 4
        f = shots.add_sot_feature(matches(rows))
        self.assertAlmostEqual(f.iloc[3], (2 - (-2)) * shots.SCALE)

    def test_first_matches_below_min_periods_give_zero(self):
        f = shots.add_sot_feature(matches([("A", "B", 1, 0, 5, 3)] * 4))
        self.assertEqual(f.iloc[0], 0)
        self.assertEqual(f.iloc[1], 0)
        self.assertEqual(f.iloc[2], 0)     # only 2 prior matches, MIN_PERIODS is 3

    def test_no_leakage_current_match_does_not_affect_own_feature(self):
        # A and B both have 5 earlier matches, so the feature at row 5 is genuinely non-zero
        base = matches([("A", "C", 1, 0, 5, 3), ("B", "D", 0, 1, 3, 5)] * 5 + [("A", "B", 1, 0, 5, 3)])  # A net +2, B net -2
        before = shots.add_sot_feature(base)
        self.assertNotEqual(before.iloc[-1], 0)            # sanity: the feature is not trivially zero
        changed = base.copy()
        changed.loc[len(base) - 1, ["HST", "AST"]] = [0, 9]  # change only the last match's shots
        after = shots.add_sot_feature(changed)
        self.assertEqual(before.iloc[-1], after.iloc[-1])   # its own feature is unchanged

    def test_window_is_exactly_ten_matches(self):
        # 10 matches net +10, then 1 match net -10. Last 10 = nine +10s and one -10: mean 8.0.
        # A window of 9 or 11 would give a different number, so this pins the size.
        history = [("A", "B", 1, 0, 10, 0)] * 10 + [("A", "C", 0, 1, 0, 10)]
        form = shots.team_shots_form(matches(history))
        self.assertAlmostEqual(form["A"], 8.0)

    def test_feature_uses_exact_window_of_ten(self):
        # Home A's rolling SoT before the last row is 8.0 (see test_window_is_exactly_ten_matches).
        # Away E has three matches with net 0, so its roll is 0. Expected sot_diff = (8.0 - 0) * SCALE = 800.
        history = ([("E", "F", 1, 0, 5, 5)] * 3 + [("A", "B", 1, 0, 10, 0)] * 10
                   + [("A", "G", 0, 1, 0, 10)] + [("A", "E", 0, 0, np.nan, np.nan)])
        f = shots.add_sot_feature(matches(history))
        self.assertAlmostEqual(f.iloc[-1], 800.0)

    def test_nan_shots_are_skipped_not_zero(self):
        rows = [("A", "B", 1, 0, 5, 3)] * 3 + [("A", "C", 1, 0, np.nan, np.nan)] + [("A", "D", 1, 0, 5, 3)]
        f = shots.add_sot_feature(matches(rows))
        self.assertTrue(np.isfinite(f.iloc[-1]))


class TestFormMatchesFeature(unittest.TestCase):
    """Serving parity: the sot_diff used at prediction time (form diff, NaN -> 0) equals what training computes
    for a new row appended to the history."""

    def check_consistency(self, rows):
        hist = matches(rows)
        new = pd.DataFrame([{"Date": hist["Date"].max() + pd.Timedelta(days=1), "HomeTeam": "A", "AwayTeam": "B",
                             "FTHG": np.nan, "FTAG": np.nan, "FTR": np.nan,
                             "HST": np.nan, "AST": np.nan, "season": "2324"}])
        full = pd.concat([hist, new], ignore_index=True)
        feat_new = shots.add_sot_feature(full).iloc[-1]
        form = shots.team_shots_form(hist)
        serve = (form.get("A", np.nan) - form.get("B", np.nan)) * shots.SCALE
        if np.isnan(serve):                       # same neutral rule as model.predict_match and training
            serve = 0.0
        self.assertAlmostEqual(feat_new, serve, places=9)

    def test_consistent_with_plain_history(self):
        self.check_consistency([("A", "B", 1, 0, 5, 3), ("C", "A", 0, 2, 4, 6), ("B", "C", 1, 1, 7, 2)] * 3)

    def test_consistent_with_gaps_and_more_than_window(self):
        rows = [("A", "X", 1, 0, 3, 1), ("B", "A", 0, 1, 2, np.nan), ("A", "Y", 2, 0, 9, 4)] * 6
        rows += [("A", "Z", 1, 0, np.nan, np.nan)] * 2
        self.check_consistency(rows)

    def test_too_few_matches_gives_nan_form(self):
        form = shots.team_shots_form(matches([("A", "B", 1, 0, 5, 3)]))
        self.assertTrue(np.isnan(form["A"]))


class TestEloCore(unittest.TestCase):
    def test_expected_score(self):
        self.assertAlmostEqual(elo.expected_score(1500, 1500), 0.5)
        self.assertAlmostEqual(elo.expected_score(1900, 1500), 10 / 11)

    def test_update_is_zero_sum(self):
        na, nb = elo.update(1600, 1450, 0.5, k=30)
        self.assertAlmostEqual(na + nb, 1600 + 1450)

    def test_equal_teams_win_moves_half_k(self):
        na, nb = elo.update(1500, 1500, 1.0, k=20)
        self.assertAlmostEqual(na, 1510)
        self.assertAlmostEqual(nb, 1490)

    def test_upset_moves_more_than_favourite_win(self):
        upset = elo.update(1400, 1600, 1.0, k=20)[0] - 1400
        expected = elo.update(1600, 1400, 1.0, k=20)[0] - 1600
        self.assertGreater(upset, expected)

    def test_goal_multiplier_steps(self):
        self.assertEqual(elo.goal_multiplier(0), 1.0)
        self.assertEqual(elo.goal_multiplier(-1), 1.0)
        self.assertEqual(elo.goal_multiplier(2), 1.5)
        self.assertEqual(elo.goal_multiplier(-2), 1.5)
        self.assertAlmostEqual(elo.goal_multiplier(3), 1.75)
        self.assertAlmostEqual(elo.goal_multiplier(4), 15 / 8)


class TestEloFeatures(unittest.TestCase):
    def league(self):
        return matches([("A", "B", 2, 0, 0, 0), ("B", "C", 0, 1, 0, 0), ("C", "A", 1, 1, 0, 0),
                        ("A", "C", 3, 0, 0, 0)])

    def test_first_match_both_at_start_rating(self):
        out = elo.add_elo_features(self.league(), start=1500)
        self.assertEqual(out["elo_home"].iloc[0], 1500)
        self.assertEqual(out["elo_away"].iloc[0], 1500)

    def test_home_advantage_not_stored_in_elo_home(self):
        out = elo.add_elo_features(self.league(), start=1500, home_adv=75)
        self.assertEqual(out["elo_home"].iloc[0], 1500)     # stored rating excludes home advantage

    def test_elo_diff_is_home_minus_away(self):
        out = elo.add_elo_features(self.league())
        np.testing.assert_allclose(out["elo_diff"], out["elo_home"] - out["elo_away"])

    def test_ratings_sum_is_conserved(self):
        _, ratings = elo.add_elo_features(self.league(), start=1500, return_ratings=True)
        self.assertAlmostEqual(sum(ratings.values()), 1500 * 3)

    def test_winner_rating_rises(self):
        out, ratings = elo.add_elo_features(self.league(), start=1500, return_ratings=True)
        self.assertGreater(ratings["A"], 1500 - 1)          # A won 2 and 3 goal games
        self.assertLess(ratings["B"], 1500 + 1)

    def test_new_team_gets_new_team_start_outside_first_season(self):
        df = pd.concat([matches([("A", "B", 1, 0, 0, 0)], season="2324"),
                        matches([("A", "Newbie", 1, 0, 0, 0)], season="2425")], ignore_index=True)
        out = elo.add_elo_features(df, start=1500, new_team_start=1300)
        self.assertEqual(out["elo_away"].iloc[1], 1300)

    def test_reversion_pulls_ratings_toward_start_at_season_change(self):
        s1 = matches([("A", "B", 3, 0, 0, 0)] * 3, season="2324")
        s2 = matches([("C", "D", 0, 0, 0, 0)], season="2425")
        s2["Date"] = s2["Date"] + pd.Timedelta(days=400)
        df = pd.concat([s1, s2], ignore_index=True)
        _, r_none = elo.add_elo_features(df, start=1500, reversion=1.0, return_ratings=True)
        _, r_half = elo.add_elo_features(df, start=1500, reversion=0.5, return_ratings=True)
        self.assertAlmostEqual(r_half["A"], 1500 + 0.5 * (r_none["A"] - 1500))

    def test_use_mov_gives_bigger_move_for_bigger_win(self):
        one = matches([("A", "B", 1, 0, 0, 0)])
        big = matches([("A", "B", 4, 0, 0, 0)])
        d_one = elo.add_elo_features(one, use_mov=True, return_ratings=True)[1]["A"] - 1500
        d_big = elo.add_elo_features(big, use_mov=True, return_ratings=True)[1]["A"] - 1500
        d_plain = elo.add_elo_features(big, use_mov=False, return_ratings=True)[1]["A"] - 1500
        self.assertGreater(d_big, d_one)
        self.assertGreater(d_big, d_plain)


if __name__ == "__main__":
    unittest.main()
