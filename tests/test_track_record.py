# tests/test_track_record.py -- run from the repo root: python -m unittest tests.test_track_record
import math
import unittest
from datetime import date
from datetime import date, timedelta
import track_record as tr

MODEL = "elo_sot10 K30 HA75 NEW1400"
OLD = "elo_only K30 HA75 NEW1400"
TODAY = date(2026, 10, 25)


def prospective(d, home, away, probs, model=MODEL):
    pH, pD, pA = probs
    day_before = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
    return {"Date": d, "HomeTeam": home, "AwayTeam": away,
            "pH": str(pH), "pD": str(pD), "pA": str(pA), "model": model,
            "logged_on": day_before, "kickoff_utc": f"{d}T14:00:00+00:00",
            "predicted_at_utc": f"{day_before}T18:00:00+00:00"}


def legacy(d, home, away, probs, model=OLD):
    pH, pD, pA = probs
    return {"Date": d, "HomeTeam": home, "AwayTeam": away,
            "pH": str(pH), "pD": str(pD), "pA": str(pA), "model": model,
            "logged_on": "2026-10-01", "kickoff_utc": "", "predicted_at_utc": ""}


class TrackRecordTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            prospective("2026-10-20", "Arsenal", "Leeds", (0.6, 0.2, 0.2)),    # H, top pick right
            prospective("2026-10-24", "Chelsea", "Spurs", (0.5, 0.25, 0.25)),  # A, top pick wrong
            prospective("2026-10-26", "Coventry", "Fulham", (0.3, 0.3, 0.4)),  # not played yet
            prospective("2026-10-20", "Man City", "Ipswich", (0.7, 0.2, 0.1), model="other label"),
            legacy("2026-10-20", "Everton", "Hull", (0.4, 0.3, 0.3)),          # legacy, not headline
        ]
        self.outcomes = {("2026-10-20", "Arsenal", "Leeds"): "H",
                         ("2026-10-24", "Chelsea", "Spurs"): "A",
                         ("2026-10-20", "Man City", "Ipswich"): "H",
                         ("2026-10-20", "Everton", "Hull"): "D"}

    def build(self, odds=None):
        return tr.build(self.rows, self.outcomes, odds or {}, set(), TODAY, MODEL)

    def test_only_prospective_rows_of_this_model_are_counted(self):
        out = self.build()
        self.assertEqual(out["logged"], 3)
        self.assertEqual(out["completed"], 2)
        self.assertEqual(out["waiting"], 1)
        self.assertEqual(out["metrics"]["n"], 2)

    def test_other_model_labels_are_listed_but_not_scored(self):
        self.assertEqual(sorted(self.build()["other_models"]), sorted([OLD, "other label"]))

    def test_top_pick_and_log_loss(self):
        m = self.build()["metrics"]
        self.assertEqual(m["top_pick_hits"], 1)
        self.assertAlmostEqual(m["top_pick_accuracy"], 0.5)
        self.assertAlmostEqual(m["logloss"], (-math.log(0.6) - math.log(0.25)) / 2, places=4)

    def test_matches_are_newest_first_with_top_pick_flag(self):
        matches = self.build()["matches"]
        self.assertEqual([m["date"] for m in matches], ["2026-10-24", "2026-10-20"])
        self.assertFalse(matches[0]["top_pick_correct"])
        self.assertTrue(matches[1]["top_pick_correct"])

    def test_small_sample_is_flagged(self):
        out = self.build()
        self.assertTrue(out["low_sample"])
        self.assertEqual(out["min_matches"], 30)

    def test_no_bookmaker_block_without_odds(self):
        self.assertIsNone(self.build()["bookmaker"])

    def test_bookmaker_comparison_on_matches_with_odds(self):
        odds = {("2026-10-20", "Arsenal", "Leeds"): {"H": 0.5, "D": 0.25, "A": 0.25},
                ("2026-10-24", "Chelsea", "Spurs"): {"H": 0.4, "D": 0.3, "A": 0.3}}
        b = self.build(odds=odds)["bookmaker"]
        self.assertEqual(b["n"], 2)
        self.assertAlmostEqual(b["bookmaker_logloss"], (-math.log(0.5) - math.log(0.3)) / 2, places=4)
        self.assertIsNotNone(b["diff"])
        self.assertIsNotNone(b["diff_se"])

    def test_empty_log(self):
        out = tr.build([], {}, {}, set(), TODAY, MODEL)
        self.assertEqual(out["completed"], 0)
        self.assertIsNone(out["metrics"])
        self.assertIsNone(out["bookmaker"])
        self.assertEqual(out["matches"], [])
        self.assertTrue(out["low_sample"])

    def test_prediction_logged_after_kickoff_is_never_scored(self):
        late = prospective("2026-10-20", "Late", "Row", (0.5, 0.25, 0.25))
        late["predicted_at_utc"] = "2026-10-21T10:00:00+00:00"   # after kickoff
        outcomes = dict(self.outcomes)
        outcomes[("2026-10-20", "Late", "Row")] = "H"
        out = tr.build(self.rows + [late], outcomes, {}, set(), TODAY, MODEL)
        self.assertEqual(out["logged"], 3)      
        self.assertEqual(out["completed"], 2)


if __name__ == "__main__":
    unittest.main()