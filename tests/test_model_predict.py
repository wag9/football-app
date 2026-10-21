"""predict_match on a small synthetic league. Trains with train(df=...), so no real data file is read."""
import unittest

import numpy as np
import pandas as pd

import model

TEAMS = ["Strong", "Mid", "Weak", "Other"]


def synthetic_league(seed=0, rounds=20):
    """Round-robin repeated `rounds` times with explicit outcome probabilities.
    'Strong' is clearly better than 'Weak'; there is a small home edge. 20 rounds = 240 matches, so the signal
    is well above sampling noise."""
    rng = np.random.default_rng(seed)
    strength = {"Strong": 0.8, "Mid": 0.5, "Weak": 0.2, "Other": 0.5}
    rows, day = [], pd.Timestamp("2023-08-01")
    for _ in range(rounds):
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                edge = strength[h] - strength[a] + 0.1          # 0.1 = home advantage
                p_h = float(np.clip(0.30 + 0.35 * edge, 0.05, 0.85))
                p_a = float(np.clip(0.30 - 0.35 * edge, 0.05, 0.85))
                r = rng.random()
                if r < p_h:
                    hg, ag, ftr = 2, 0, "H"
                elif r < p_h + p_a:
                    hg, ag, ftr = 0, 2, "A"
                else:
                    hg, ag, ftr = 1, 1, "D"
                hst, ast = int(rng.integers(2, 8)), int(rng.integers(2, 8))
                rows.append({"Date": day, "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag, "FTR": ftr,
                             "AvgH": 2.0, "AvgD": 3.4, "AvgA": 3.8, "HS": 10, "AS": 10,
                             "HST": hst, "AST": ast, "HC": 4, "AC": 4, "season": "2324"})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


class TestPredictMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = model.train(league="E0", df=synthetic_league())

    def test_probabilities_are_a_valid_distribution(self):
        p = model.predict_match(self.state, "Strong", "Weak")
        self.assertEqual(set(p), {"H", "D", "A"})
        self.assertAlmostEqual(sum(p.values()), 1.0, places=9)
        self.assertTrue(all(0 <= v <= 1 for v in p.values()))

    def test_stronger_home_team_is_more_likely_to_win(self):
        p_fav_home = model.predict_match(self.state, "Strong", "Weak")
        p_fav_away = model.predict_match(self.state, "Weak", "Strong")
        self.assertGreater(p_fav_home["H"], p_fav_away["H"])
        self.assertGreater(p_fav_away["A"], p_fav_home["A"])

    def test_same_team_raises(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            model.predict_match(self.state, "Strong", "Strong")

    def test_unknown_team_raises_and_lists_available(self):
        with self.assertRaisesRegex(ValueError, "'Ghost' not found"):
            model.predict_match(self.state, "Strong", "Ghost")

    def test_missing_shots_form_uses_neutral_zero(self):
        # Same as training's rule: no shots form (NaN) means sot_diff = 0.
        st = dict(self.state, shots_form={})
        got = model.predict_match(st, "Strong", "Weak")
        elo_diff = self.state["ratings"]["Strong"] - self.state["ratings"]["Weak"]
        X = pd.DataFrame({"elo_diff": [elo_diff], "sot_diff": [0.0]})[model.FEATURES]
        want = dict(zip(self.state["model"].classes_, self.state["model"].predict_proba(X)[0]))
        for k in want:
            self.assertAlmostEqual(got[k], want[k], places=12)

    def test_shots_form_changes_the_prediction(self):
        # With a non-zero sot coefficient, a big shots advantage must move the probabilities.
        self.assertNotEqual(self.state["model"].coef_[0][1], 0)
        neutral = model.predict_match(dict(self.state, shots_form={}), "Strong", "Weak")
        strong_form = dict(self.state, shots_form={"Strong": 20.0, "Weak": -20.0, "Mid": 0, "Other": 0})
        boosted = model.predict_match(strong_form, "Strong", "Weak")
        self.assertNotAlmostEqual(neutral["H"], boosted["H"], places=6)

    def test_sot_sign_and_scale_match_the_training_formula(self):
        # Home form minus away form, times SCALE: the same formula as shots.add_sot_feature.
        form = {"Strong": 3.0, "Weak": -1.0}
        st = dict(self.state, shots_form=form)
        got = model.predict_match(st, "Strong", "Weak")
        elo_diff = self.state["ratings"]["Strong"] - self.state["ratings"]["Weak"]
        sot = (form["Strong"] - form["Weak"]) * model.SCALE
        X = pd.DataFrame({"elo_diff": [elo_diff], "sot_diff": [sot]})[model.FEATURES]
        want = dict(zip(self.state["model"].classes_, self.state["model"].predict_proba(X)[0]))
        for k in want:
            self.assertAlmostEqual(got[k], want[k], places=12)

    def test_feature_order_matches_training(self):
        self.assertEqual(model.FEATURES, ["elo_diff", "sot_diff"])
        self.assertEqual(list(self.state["model"].feature_names_in_), model.FEATURES)

    def test_model_version_label_is_built_from_settings(self):
        self.assertIn(f"K{model.K}", model.MODEL_VERSION)
        self.assertIn(f"HA{model.HOME_ADV}", model.MODEL_VERSION)
        self.assertIn(f"NEW{model.NEW_START}", model.MODEL_VERSION)
        self.assertTrue(model.MODEL_VERSION.startswith(f"elo_sot{model.WINDOW} "))


if __name__ == "__main__":
    unittest.main()
