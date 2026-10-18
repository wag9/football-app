"""Tests for evaluation.py and check.py. Run from the project root:

    python -m unittest discover -s tests -v

Synthetic data only, in temporary folders. The real data/ folder is hashed before and
after to prove it is never touched.
"""
import hashlib
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation as ev  # noqa: E402

TODAY = date(2026, 10, 20)
_BEFORE = {}


def _hash_data_dir():
    d = ROOT / "data"
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def setUpModule():
    global _BEFORE
    _BEFORE = _hash_data_dir()


def tearDownModule():
    assert _hash_data_dir() == _BEFORE, "tests modified the real data/ folder"


def log_row(d, home, away, p, model="m1", logged_on=None, kickoff="", predicted=""):
    return {"Date": d, "HomeTeam": home, "AwayTeam": away,
            "pH": str(p[0]), "pD": str(p[1]), "pA": str(p[2]),
            "model": model, "logged_on": logged_on or d,
            "kickoff_utc": kickoff, "predicted_at_utc": predicted}


def run(rows, outcomes, odds=None, conflicts=(), today=TODAY, aliases=None):
    entries, stats = ev.prepare(rows, outcomes, odds or {}, set(conflicts), today, aliases)
    return entries, stats, ev.evaluate(entries)


def stratum_summary(result, model, stratum):
    for m, parts, _ in result:
        if m == model:
            for s, summary in parts:
                if s == stratum:
                    return summary
    raise KeyError((model, stratum))


class Metrics(unittest.TestCase):
    def test_log_loss_and_brier_by_hand(self):
        ll, br, hit = ev.per_match({"H": 0.5, "D": 0.3, "A": 0.2}, "H")
        self.assertAlmostEqual(ll, -math.log(0.5))
        self.assertAlmostEqual(br, 0.25 + 0.09 + 0.04)
        self.assertTrue(hit)

    def test_uniform_guess_matches_the_reference_values(self):
        ll, br, _ = ev.per_match({"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}, "D")
        self.assertAlmostEqual(ll, ev.UNIFORM_LOGLOSS)
        self.assertAlmostEqual(br, ev.UNIFORM_BRIER)

    def test_perfect_and_terrible_predictions(self):
        ll, br, hit = ev.per_match({"H": 1.0, "D": 0.0, "A": 0.0}, "H")
        self.assertAlmostEqual((ll, br, hit), (0.0, 0.0, True))
        ll, br, hit = ev.per_match({"H": 1.0, "D": 0.0, "A": 0.0}, "A")
        self.assertTrue(math.isfinite(ll))            # clipped, never infinity
        self.assertAlmostEqual(br, 2.0)
        self.assertFalse(hit)

    def test_top_pick_and_draw_never_chosen_when_lower(self):
        self.assertFalse(ev.per_match({"H": 0.4, "D": 0.3, "A": 0.3}, "D")[2])
        self.assertTrue(ev.per_match({"H": 0.2, "D": 0.5, "A": 0.3}, "D")[2])

    def test_standard_error(self):
        self.assertIsNone(ev.standard_error([1.0]))
        self.assertAlmostEqual(ev.standard_error([1.0, 3.0]), 1.0)

    def test_rounding_drift_is_renormalised(self):
        p = ev.normalise({"H": 0.4254, "D": 0.2764, "A": 0.2983})
        self.assertAlmostEqual(sum(p.values()), 1.0)


class Classification(unittest.TestCase):
    PRE = {"kickoff": "2026-10-24T14:00:00+00:00", "predicted": "2026-10-20T09:00:00+00:00"}

    def stratum_of(self, **kw):
        row = log_row("2026-10-24", "A", "B", (0.4, 0.3, 0.3), **kw)
        return ev.stratum(row, "2026-10-24")

    def test_prospective_verified(self):
        self.assertEqual(self.stratum_of(**self.PRE), "prospective_verified")

    def test_timestamp_after_kickoff_is_a_violation(self):
        self.assertEqual(self.stratum_of(kickoff="2026-10-24T14:00:00+00:00",
                                         predicted="2026-10-24T15:00:00+00:00"), "timing_violation")

    def test_garbage_timestamp(self):
        self.assertEqual(self.stratum_of(kickoff="soon", predicted="now"), "timing_invalid")

    def test_legacy_strata_from_logged_on(self):
        self.assertEqual(self.stratum_of(logged_on="2026-10-23"), "legacy_prior_day")
        self.assertEqual(self.stratum_of(logged_on="2026-10-24"), "legacy_same_day")
        self.assertEqual(self.stratum_of(logged_on="2026-10-25"), "timing_violation")
        self.assertEqual(self.stratum_of(logged_on="garbage"), "timing_invalid")


class StatusAndMatching(unittest.TestCase):
    def test_status_for_each_situation(self):
        rows = [log_row("2026-10-18", "A", "B", (0.5, 0.3, 0.2)),    # has result
                log_row("2026-10-19", "C", "D", (0.5, 0.3, 0.2)),    # yesterday, no result yet
                log_row("2026-10-25", "E", "F", (0.5, 0.3, 0.2)),    # future
                log_row("2026-09-01", "G", "H", (0.5, 0.3, 0.2))]    # long ago, no result: postponed?
        entries, _, _ = run(rows, {("2026-10-18", "A", "B"): "H"})
        self.assertEqual([e["status"] for e in entries], ["completed", "awaiting", "upcoming", "overdue"])

    def test_postponed_match_is_never_scored_and_a_rescheduled_row_is_separate(self):
        rows = [log_row("2026-10-10", "A", "B", (0.5, 0.3, 0.2)),    # postponed, no result ever
                log_row("2026-11-10", "A", "B", (0.5, 0.3, 0.2))]    # new date
        _, _, result = run(rows, {})
        s = stratum_summary(result, "m1", "legacy_same_day")
        self.assertEqual((s["completed"], s["overdue"] + s["awaiting"], s["upcoming"]), (0, 1, 1))
        self.assertIsNone(s["metrics"])

    def test_duplicate_fixture_keeps_the_first_row(self):
        rows = [log_row("2026-10-18", "A", "B", (0.7, 0.2, 0.1)),
                log_row("2026-10-18", "A", "B", (0.1, 0.2, 0.7))]
        entries, stats, _ = run(rows, {("2026-10-18", "A", "B"): "H"})
        self.assertEqual((len(entries), stats["duplicates_dropped"]), (1, 1))
        self.assertAlmostEqual(entries[0]["probs"]["H"], 0.7)

    def test_invalid_probabilities_are_counted_not_scored(self):
        rows = [log_row("2026-10-18", "A", "B", (0.9, 0.9, 0.9)),
                log_row("2026-10-18", "C", "D", ("x", 0.3, 0.3)),
                log_row("2026-10-18", "E", "F", (0.5, 0.3, 0.2))]
        outcomes = {("2026-10-18", t[0], t[1]): "H" for t in (("A", "B"), ("C", "D"), ("E", "F"))}
        _, _, result = run(rows, outcomes)
        s = stratum_summary(result, "m1", "legacy_same_day")
        self.assertEqual((s["logged"], s["invalid_probs"], s["eligible"], s["completed"]), (3, 2, 1, 1))
        self.assertEqual(s["metrics"]["n"], 1)

    def test_conflicting_results_are_excluded(self):
        rows = [log_row("2026-10-18", "A", "B", (0.5, 0.3, 0.2))]
        _, _, result = run(rows, {("2026-10-18", "A", "B"): "H"}, conflicts=[("2026-10-18", "A", "B")])
        s = stratum_summary(result, "m1", "legacy_same_day")
        self.assertEqual((s["result_conflict"], s["completed"]), (1, 0))

    def test_unreadable_rows_are_counted(self):
        rows = [log_row("not a date", "A", "B", (0.5, 0.3, 0.2)), log_row("2026-10-18", "", "B", (0.5, 0.3, 0.2))]
        entries, stats, _ = run(rows, {})
        self.assertEqual((len(entries), stats["unreadable_rows"]), (0, 2))

    def test_old_schema_rows_without_timestamp_columns_work(self):
        row = {"Date": "2026-10-10", "HomeTeam": "A", "AwayTeam": "B", "pH": "0.5", "pD": "0.3", "pA": "0.2",
               "model": "m1", "logged_on": "2026-10-09"}
        entries, _, _ = run([row], {("2026-10-10", "A", "B"): "H"})
        self.assertEqual((entries[0]["stratum"], entries[0]["status"]), ("legacy_prior_day", "completed"))


class Grouping(unittest.TestCase):
    def test_model_versions_are_never_merged(self):
        rows = [log_row("2026-10-18", "A", "B", (0.9, 0.05, 0.05), model="v1"),
                log_row("2026-10-18", "C", "D", (0.1, 0.1, 0.8), model="v2")]
        outcomes = {("2026-10-18", "A", "B"): "H", ("2026-10-18", "C", "D"): "H"}
        _, _, result = run(rows, outcomes)
        self.assertEqual([m for m, _, _ in result], ["v1", "v2"])
        v1 = stratum_summary(result, "v1", "legacy_same_day")["metrics"]
        v2 = stratum_summary(result, "v2", "legacy_same_day")["metrics"]
        self.assertEqual((v1["n"], v2["n"]), (1, 1))
        self.assertLess(v1["logloss"], 0.2)
        self.assertGreater(v2["logloss"], 2.0)

    def test_timing_strata_are_separate_and_violations_are_not_scored(self):
        rows = [log_row("2026-10-18", "A", "B", (0.5, 0.3, 0.2), logged_on="2026-10-17"),
                log_row("2026-10-18", "C", "D", (0.5, 0.3, 0.2), logged_on="2026-10-18"),
                log_row("2026-10-18", "E", "F", (0.99, 0.005, 0.005), logged_on="2026-10-19")]
        outcomes = {("2026-10-18", t[0], t[1]): "H" for t in (("A", "B"), ("C", "D"), ("E", "F"))}
        _, _, result = run(rows, outcomes)
        self.assertEqual(stratum_summary(result, "m1", "legacy_prior_day")["completed"], 1)
        self.assertEqual(stratum_summary(result, "m1", "legacy_same_day")["completed"], 1)
        bad = stratum_summary(result, "m1", "timing_violation")
        self.assertEqual((bad["logged"], bad["completed"], bad["metrics"]), (1, 0, None))
        combined = result[0][2]
        self.assertEqual(combined["metrics"]["n"], 2)       # violation row not in the combined line either

    def test_aliases_merge_only_when_explicitly_given(self):
        rows = [log_row("2026-10-18", "A", "B", (0.5, 0.3, 0.2), model="v1"),
                log_row("2026-10-18", "C", "D", (0.5, 0.3, 0.2), model="v1 hist")]
        outcomes = {("2026-10-18", "A", "B"): "H", ("2026-10-18", "C", "D"): "H"}
        self.assertEqual(len(run(rows, outcomes)[2]), 2)
        self.assertEqual(len(run(rows, outcomes, aliases={"v1 hist": "v1"})[2]), 1)


class BookmakerComparison(unittest.TestCase):
    def test_only_matches_with_odds_are_compared_and_the_model_is_rescored_on_them(self):
        rows = [log_row("2026-10-18", "A", "B", (0.9, 0.05, 0.05)),      # has odds
                log_row("2026-10-18", "C", "D", (0.1, 0.1, 0.8))]        # no odds
        outcomes = {("2026-10-18", "A", "B"): "H", ("2026-10-18", "C", "D"): "H"}
        odds = {("2026-10-18", "A", "B"): {"H": 0.5, "D": 0.3, "A": 0.2}, ("2026-10-18", "C", "D"): None}
        _, _, result = run(rows, outcomes, odds)
        s = stratum_summary(result, "m1", "legacy_same_day")
        self.assertEqual(s["metrics"]["n"], 2)
        b = s["bookmaker"]
        self.assertEqual(b["n"], 1)
        self.assertAlmostEqual(b["model_logloss"], -math.log(0.9))      # not the 2-match average
        self.assertAlmostEqual(b["book_logloss"], -math.log(0.5))
        self.assertAlmostEqual(b["diff"], -math.log(0.9) + math.log(0.5))

    def test_no_odds_means_no_comparison(self):
        rows = [log_row("2026-10-18", "A", "B", (0.5, 0.3, 0.2))]
        _, _, result = run(rows, {("2026-10-18", "A", "B"): "H"}, {("2026-10-18", "A", "B"): None})
        self.assertIsNone(stratum_summary(result, "m1", "legacy_same_day")["bookmaker"])

    def test_implied_probabilities_are_normalised(self):
        p = ev._bookmaker({"AvgH": "2.0", "AvgD": "4.0", "AvgA": "4.0"})
        self.assertAlmostEqual(p["H"], 0.5)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        for bad in ({"AvgH": "", "AvgD": "3", "AvgA": "3"}, {"AvgH": "0.9", "AvgD": "3", "AvgA": "3"},
                    {"AvgH": "nan", "AvgD": "3", "AvgA": "3"}, {}):
            self.assertIsNone(ev._bookmaker(bad))


class LoadResults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "r.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_loading_duplicates_conflicts_and_bad_rows(self):
        self.path.write_text(
            "Date,HomeTeam,AwayTeam,FTR,AvgH,AvgD,AvgA\n"
            "2026-10-18,A,B,H,2.0,4.0,4.0\n"
            "2026-10-18,A,B,H,2.0,4.0,4.0\n"          # identical duplicate: fine
            "2026-10-19,C,D,H,2,3,4\n"
            "2026-10-19,C,D,A,2,3,4\n"                # conflicting duplicate
            "2026-10-20,E,F,,2,3,4\n"                  # no result (postponed)
            "garbage,G,H,H,2,3,4\n",
            encoding="utf-8")
        outcomes, odds, conflicts = ev.load_results(self.path)
        self.assertEqual(set(outcomes), {("2026-10-18", "A", "B"), ("2026-10-19", "C", "D")})
        self.assertEqual(conflicts, {("2026-10-19", "C", "D")})
        self.assertAlmostEqual(odds[("2026-10-18", "A", "B")]["H"], 0.5)


class CommandLine(unittest.TestCase):
    """Runs the real check.py in a temporary folder that holds a stub leagues.py."""

    def test_end_to_end_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("check.py", "evaluation.py", "prediction_log.py"):
                shutil.copy(ROOT / name, tmp / name)
            (tmp / "data").mkdir()
            (tmp / "leagues.py").write_text(
                'DEFAULT = "T1"\nLEAGUES = {"T1": {"name": "Test League", '
                '"matches": "data/m.csv", "log": "data/l.csv"}, '
                '"T2": {"name": "Empty League", "matches": "data/m2.csv", "log": "data/none.csv"}}\n')
            (tmp / "data" / "m.csv").write_text(
                "Date,HomeTeam,AwayTeam,FTR,AvgH,AvgD,AvgA\n"
                "2026-10-10,A,B,H,2.0,4.0,4.0\n2026-10-11,C,D,D,2.0,3.5,4.0\n2026-10-12,E,F,A,,,\n")
            (tmp / "data" / "l.csv").write_text(
                "Date,HomeTeam,AwayTeam,pH,pD,pA,model,logged_on,kickoff_utc,predicted_at_utc\n"
                "2026-10-10,A,B,0.5,0.3,0.2,old model,2026-10-09,,\n"
                "2026-10-11,C,D,0.4,0.3,0.3,old model,2026-10-11,,\n"
                "2026-10-12,E,F,0.5,0.3,0.2,new model,2026-10-08,2026-10-12T14:00:00+00:00,2026-10-08T09:00:00+00:00\n"
                "2026-10-12,E,F,0.1,0.1,0.8,new model,2026-10-08,2026-10-12T14:00:00+00:00,2026-10-08T09:00:00+00:00\n"
                "2026-10-30,G,H,0.5,0.3,0.2,new model,2026-10-19,2026-10-30T14:00:00+00:00,2026-10-19T09:00:00+00:00\n")
            done = subprocess.run([sys.executable, "check.py", "--today", "2026-10-20", "--matches"],
                                  cwd=tmp, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            out = done.stdout
            self.assertIn("model: old model", out)
            self.assertIn("model: new model", out)
            self.assertIn("duplicate fixtures dropped (first kept): 1", out)
            self.assertIn("PROSPECTIVE, verified", out)
            self.assertIn("LEGACY, logged on an earlier day", out)
            self.assertIn("LEGACY, logged on the match day", out)
            self.assertIn("reconciliation only", out)
            self.assertIn("no prediction log for this league yet", out)
            self.assertIn("Too few to conclude", out)
            one = subprocess.run([sys.executable, "check.py", "T1"], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertNotIn("Empty League", one.stdout)
            bad = subprocess.run([sys.executable, "check.py", "XX"], cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()