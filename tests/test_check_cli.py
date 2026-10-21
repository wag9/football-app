"""check.py: formatting helpers, the report text for each situation, and the command-line entry point.
Uses temporary files and patched league paths, so no real log or results file is read."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

import check
import evaluation as ev


def summary(**over):
    s = {"logged": 0, "eligible": 0, "completed": 0, "upcoming": 0, "awaiting": 0, "overdue": 0,
         "invalid_probs": 0, "result_conflict": 0, "scored": False, "metrics": None, "bookmaker": None}
    s.update(over)
    return s


class TestFormatting(unittest.TestCase):
    def test_none_is_n_a(self):
        self.assertEqual(check.f4(None), "n/a")
        self.assertEqual(check.signed(None), "n/a")

    def test_four_decimals_and_sign(self):
        self.assertEqual(check.f4(1.23456), "1.2346")
        self.assertEqual(check.signed(-0.0116), "-0.0116")
        self.assertEqual(check.signed(0.5), "+0.5000")


class TestDescribe(unittest.TestCase):
    def test_nothing_completed_says_not_scored(self):
        lines = check.describe(summary(upcoming=3, logged=3, eligible=3))
        self.assertTrue(any("waiting for result 3" in l for l in lines))
        self.assertIn("not scored", lines)

    def test_scored_but_no_completed(self):
        lines = check.describe(summary(scored=True))
        self.assertIn("no completed matches yet", lines)

    def test_exclusions_and_overdue_are_reported(self):
        lines = check.describe(summary(overdue=2, invalid_probs=1, result_conflict=1, scored=True))
        self.assertTrue(any("invalid probabilities 1" in l for l in lines))
        self.assertTrue(any("conflicting results 1" in l for l in lines))
        self.assertTrue(any(f"more than {ev.OVERDUE_DAYS} days" in l for l in lines))

    def test_small_sample_gets_a_warning(self):
        m = {"hits": 3, "n": 5, "logloss": 1.0, "logloss_se": 0.1, "brier": 0.6}
        lines = check.describe(summary(scored=True, completed=5, metrics=m))
        self.assertTrue(any(f"< {ev.MIN_N}" in l for l in lines))

    def test_no_odds_says_no_comparison(self):
        m = {"hits": 6, "n": 10, "logloss": 0.8, "logloss_se": 0.1, "brier": 0.5}
        lines = check.describe(summary(scored=True, completed=10, metrics=m, bookmaker=None))
        self.assertIn("bookmaker: no completed match has valid odds, so no comparison", lines)

    def test_top_pick_line_shows_uniform_reference(self):
        m = {"hits": 6, "n": 10, "logloss": 0.8, "logloss_se": 0.1, "brier": 0.5}
        text = "\n".join(check.describe(summary(scored=True, completed=10, metrics=m)))
        self.assertIn("top-pick accuracy 6/10 = 60%", text)
        self.assertIn(f"{ev.UNIFORM_LOGLOSS:.4f}", text)


class TestCommandLine(unittest.TestCase):
    def test_unknown_league_exits_with_message(self):
        with self.assertRaises(SystemExit) as cm:
            check.main(["XX9"])
        self.assertIn("Unknown league 'XX9'", str(cm.exception))

    def test_missing_log_is_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as d:
            fake = {"E0": {"name": "Test League", "prefix": "t",
                           "matches": os.path.join(d, "m.csv"), "log": os.path.join(d, "missing.csv")}}
            buf = io.StringIO()
            with mock.patch.dict(check.LEAGUES, fake, clear=True), redirect_stdout(buf):
                rc = check.main(["E0", "--today", "2026-10-21"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("evaluation date: 2026-10-21", out)
            self.assertIn("no prediction log for this league yet", out)

    def test_bad_today_value_is_an_error(self):
        with self.assertRaises(ValueError):
            with redirect_stderr(io.StringIO()):
                check.main(["E0", "--today", "21-10-2026"])


if __name__ == "__main__":
    unittest.main()
