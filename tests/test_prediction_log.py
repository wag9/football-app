"""Tests for prediction_log.py. Run from the project root:

    python -m unittest discover -s tests -v

Everything uses synthetic data in temporary folders. The real data/ folder is
hashed before and after the run to prove the tests never touch it.
"""
import hashlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prediction_log as pl  # noqa: E402

UTC = timezone.utc


def _hash_data_dir():
    d = ROOT / "data"
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


_BEFORE = {}


def setUpModule():
    global _BEFORE
    _BEFORE = _hash_data_dir()


def tearDownModule():
    assert _hash_data_dir() == _BEFORE, "tests modified the real data/ folder"


def fake_predict(home, away):
    return {"H": 0.45, "D": 0.27, "A": 0.28}


def fx(date, time_, home="Alpha", away="Beta"):
    return {"Date": date, "Time": time_, "HomeTeam": home, "AwayTeam": away}


TEAMS = {"Alpha", "Beta", "Gamma", "Delta"}
NOW = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)   # a Tuesday, 09:00 UTC


def plan(fixtures, existing=(), played=frozenset(), now=NOW, predict=fake_predict, teams=TEAMS):
    return pl.plan_predictions(list(fixtures), list(existing), set(played), teams, predict,
                               now, "test-model", local_date="2026-10-20")


class KickoffParsing(unittest.TestCase):
    def test_british_summer_time(self):
        ko, status = pl.kickoff_utc("24/10/2026", "15:00")
        self.assertEqual(ko, datetime(2026, 10, 24, 14, 0, tzinfo=UTC))
        self.assertEqual(status, "ok")

    def test_greenwich_mean_time_after_clocks_change(self):
        ko, _ = pl.kickoff_utc("31/10/2026", "15:00")      # clocks went back on 25 Oct 2026
        self.assertEqual(ko, datetime(2026, 10, 31, 15, 0, tzinfo=UTC))

    def test_two_digit_year_and_iso_date(self):
        self.assertEqual(pl.parse_date("17/10/26"), "2026-10-17")
        self.assertEqual(pl.parse_date("2026-10-17"), "2026-10-17")
        self.assertEqual(pl.parse_date("2026-10-17 00:00:00"), "2026-10-17")

    def test_invalid_dates(self):
        for bad in ["31/02/2026", "tbc", "", None, float("nan")]:
            self.assertIsNone(pl.parse_date(bad), bad)
        ko, status = pl.kickoff_utc("31/02/2026", "15:00")
        self.assertIsNone(ko)
        self.assertEqual(status, "bad_date")

    def test_missing_time_is_conservative(self):
        ko, status = pl.kickoff_utc("24/10/2026", "")
        self.assertEqual(status, "no_time")
        self.assertEqual(ko, datetime(2026, 10, 23, 23, 0, tzinfo=UTC))   # 00:00 BST = 23:00 UTC
        _, status_nan = pl.kickoff_utc("24/10/2026", float("nan"))
        self.assertEqual(status_nan, "no_time")

    def test_garbage_time_is_conservative(self):
        ko, status = pl.kickoff_utc("24/10/2026", "TBC")
        self.assertEqual(status, "bad_time")
        self.assertEqual(ko, datetime(2026, 10, 23, 23, 0, tzinfo=UTC))

    def test_seconds_in_time_are_accepted(self):
        ko, status = pl.kickoff_utc("31/10/2026", "15:00:00")
        self.assertEqual((ko.hour, status), (15, "ok"))


class Guard(unittest.TestCase):
    KO_TEXT = ("24/10/2026", "15:00")          # kickoff 14:00 UTC

    def decide(self, now):
        return pl.can_predict(*self.KO_TEXT, now)

    def test_well_before_kickoff(self):
        self.assertTrue(self.decide(datetime(2026, 10, 20, 9, 0, tzinfo=UTC)).ok)

    def test_exactly_at_the_lead_boundary_is_allowed(self):
        self.assertTrue(self.decide(datetime(2026, 10, 24, 12, 0, tzinfo=UTC)).ok)

    def test_one_second_inside_the_lead_is_rejected(self):
        d = self.decide(datetime(2026, 10, 24, 12, 0, 1, tzinfo=UTC))
        self.assertFalse(d.ok)
        self.assertEqual(d.reason, "too_late")

    def test_same_day_after_kickoff_is_rejected(self):
        """The original bug: a date-only check let this through."""
        d = self.decide(datetime(2026, 10, 24, 18, 0, tzinfo=UTC))
        self.assertFalse(d.ok)

    def test_naive_now_is_refused(self):
        with self.assertRaises(ValueError):
            pl.can_predict("24/10/2026", "15:00", datetime(2026, 10, 20, 9, 0))

    def test_missing_time_same_day_is_rejected_but_future_day_is_fine(self):
        self.assertFalse(pl.can_predict("20/10/2026", "", NOW).ok)
        self.assertTrue(pl.can_predict("24/10/2026", "", NOW).ok)


class ProbabilityValidity(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(pl.check_probs(0.4254, 0.2764, 0.2982))
        self.assertIsNone(pl.check_probs(0.9999, 0.0, 0.0001))

    def test_invalid(self):
        bad = [(0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), (1.2, -0.1, -0.1),
               (float("nan"), 0.5, 0.5), (float("inf"), 0, 0), ("x", 0.5, 0.5), (None, 0.5, 0.5)]
        for args in bad:
            self.assertIsNotNone(pl.check_probs(*args), args)


class Planning(unittest.TestCase):
    def reasons(self, p):
        return [r for _, r in p.skipped]

    def test_logs_a_valid_future_fixture(self):
        p = plan([fx("24/10/2026", "15:00")])
        self.assertEqual(len(p.new_rows), 1)
        row = p.new_rows[0]
        self.assertEqual(row["Date"], "2026-10-24")
        self.assertEqual(row["kickoff_utc"], "2026-10-24T14:00:00+00:00")
        self.assertEqual(row["predicted_at_utc"], "2026-10-20T09:00:00+00:00")
        self.assertEqual((row["pH"], row["pD"], row["pA"]), ("0.4500", "0.2700", "0.2800"))
        self.assertEqual(row["model"], "test-model")
        self.assertEqual(set(row), set(pl.COLUMNS))

    def test_never_logs_after_kickoff(self):
        late_now = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)
        p = plan([fx("24/10/2026", "15:00")], now=late_now)
        self.assertEqual(p.new_rows, [])
        self.assertEqual(self.reasons(p), ["too_late"])

    def test_already_played_is_skipped(self):
        p = plan([fx("24/10/2026", "15:00")], played={("2026-10-24", "Alpha", "Beta")})
        self.assertEqual(self.reasons(p), ["already_played"])

    def test_unknown_team(self):
        p = plan([fx("24/10/2026", "15:00", away="Nowhere FC")])
        self.assertEqual(self.reasons(p), ["unknown_team"])

    def test_malformed_rows(self):
        p = plan([fx("24/10/2026", "15:00", home=""),
                  fx("24/10/2026", "15:00", home="Alpha", away="Alpha"),
                  fx("not a date", "15:00")])
        self.assertEqual(self.reasons(p), ["malformed", "malformed", "bad_date"])

    def test_duplicate_rows_in_the_fixture_file(self):
        p = plan([fx("24/10/2026", "15:00"), fx("24/10/2026", "15:00")])
        self.assertEqual(len(p.new_rows), 1)
        self.assertEqual(self.reasons(p), ["duplicate_in_fixtures"])

    def test_existing_log_row_is_never_replaced(self):
        existing = [{"Date": "2026-10-24", "HomeTeam": "Alpha", "AwayTeam": "Beta"}]
        p = plan([fx("24/10/2026", "15:00")], existing=existing)
        self.assertEqual(p.new_rows, [])
        self.assertEqual(self.reasons(p), ["already_logged"])

    def test_already_logged_is_reported_even_if_kickoff_has_passed(self):
        existing = [{"Date": "2026-10-24", "HomeTeam": "Alpha", "AwayTeam": "Beta"}]
        p = plan([fx("24/10/2026", "15:00")], existing=existing,
                 now=datetime(2026, 10, 24, 20, 0, tzinfo=UTC))
        self.assertEqual(self.reasons(p), ["already_logged"])

    def test_model_failure_is_skipped_not_fatal(self):
        def boom(h, a):
            raise ValueError("no rating")
        p = plan([fx("24/10/2026", "15:00")], predict=boom)
        self.assertTrue(self.reasons(p)[0].startswith("model_error"))

    def test_invalid_probabilities_are_not_logged(self):
        p = plan([fx("24/10/2026", "15:00")], predict=lambda h, a: {"H": 0.9, "D": 0.9, "A": 0.9})
        self.assertTrue(self.reasons(p)[0].startswith("bad_probs"))

    def test_missing_time_is_logged_with_a_warning_when_far_ahead(self):
        p = plan([fx("24/10/2026", "")])
        self.assertEqual(len(p.new_rows), 1)
        self.assertEqual(p.new_rows[0]["kickoff_utc"], "2026-10-23T23:00:00+00:00")
        self.assertTrue(any("assumed 00:00" in w for w in p.warnings))

    def test_reschedule_warning(self):
        existing = [{"Date": "2026-10-24", "HomeTeam": "Alpha", "AwayTeam": "Beta"}]
        p = plan([fx("31/10/2026", "15:00")], existing=existing)
        self.assertEqual(len(p.new_rows), 1)
        self.assertTrue(any("possible reschedule" in w for w in p.warnings))

    def test_no_reschedule_warning_when_the_old_row_was_played(self):
        existing = [{"Date": "2025-05-01", "HomeTeam": "Alpha", "AwayTeam": "Beta"}]
        p = plan([fx("31/10/2026", "15:00")], existing=existing, played={("2025-05-01", "Alpha", "Beta")})
        self.assertEqual(p.warnings, [])


class FileHandling(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.log = self.dir / "log.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def write_legacy(self, newline="\n", trailing_newline=True):
        lines = ["Date,HomeTeam,AwayTeam,pH,pD,pA,model,logged_on",
                 "2026-10-10,Man United,Tottenham,0.6721,0.1989,0.1290,elo_only K30 HA75 NEW1400,2026-10-10",
                 "2026-10-11,Crystal Palace,Nott'm Forest,0.3944,0.2620,0.3436,elo_only K30 HA75 NEW1400,2026-10-10"]
        text = newline.join(lines) + (newline if trailing_newline else "")
        self.log.write_bytes(text.encode())

    def new_row(self, **kw):
        base = dict(Date="2026-10-24", HomeTeam="Alpha", AwayTeam="Beta", pH="0.4500", pD="0.2700",
                    pA="0.2800", model="m", logged_on="2026-10-20",
                    kickoff_utc="2026-10-24T14:00:00+00:00", predicted_at_utc="2026-10-20T09:00:00+00:00")
        base.update(kw)
        return base

    def test_migration_keeps_every_value_and_makes_a_backup(self):
        self.write_legacy()
        _, old_rows = pl.read_log(self.log)
        status, backup = pl.migrate_log(self.log, self.dir / "backups")
        self.assertEqual(status, "migrated")
        self.assertTrue(backup.exists())
        header, rows = pl.read_log(self.log)
        self.assertEqual(header, pl.COLUMNS)
        self.assertEqual(len(rows), len(old_rows))
        for a, b in zip(old_rows, rows):
            for c in pl.LEGACY_COLUMNS:
                self.assertEqual(a[c], b[c])
            self.assertEqual((b["kickoff_utc"], b["predicted_at_utc"]), ("", ""))
        self.assertEqual(pl.migrate_log(self.log, self.dir / "backups")[0], "already_current")

    def test_migration_refuses_unknown_columns(self):
        self.log.write_text("a,b,c\n1,2,3\n")
        before = self.log.read_bytes()
        with self.assertRaises(pl.SchemaError):
            pl.migrate_log(self.log, self.dir / "backups")
        self.assertEqual(self.log.read_bytes(), before)

    def test_append_before_migration_is_refused_and_leaves_file_alone(self):
        self.write_legacy()
        before = self.log.read_bytes()
        with self.assertRaises(pl.SchemaError):
            pl.append_rows(self.log, [self.new_row()])
        self.assertEqual(self.log.read_bytes(), before)

    def test_append_never_changes_existing_bytes_lf_and_crlf(self):
        for nl in ("\n", "\r\n"):
            with self.subTest(newline=repr(nl)):
                self.write_legacy(newline=nl)
                pl.migrate_log(self.log, self.dir / "backups")
                before = self.log.read_bytes()
                pl.append_rows(self.log, [self.new_row()])
                after = self.log.read_bytes()
                self.assertTrue(after.startswith(before))
                self.assertIn(nl.encode(), after[len(before):])
                self.assertEqual(after.count(b"\r\n" if nl == "\r\n" else b"\n"),
                                 after.count(b"\n"))   # one style only

    def test_append_handles_a_file_without_trailing_newline(self):
        self.write_legacy(trailing_newline=False)
        pl.migrate_log(self.log, self.dir / "backups")
        self.log.write_bytes(self.log.read_bytes().rstrip())
        before = self.log.read_bytes()
        pl.append_rows(self.log, [self.new_row()])
        self.assertTrue(self.log.read_bytes().startswith(before))
        _, rows = pl.read_log(self.log)
        self.assertEqual(len(rows), 3)

    def test_append_creates_a_new_log_with_header(self):
        pl.append_rows(self.log, [self.new_row()])
        header, rows = pl.read_log(self.log)
        self.assertEqual(header, pl.COLUMNS)
        self.assertEqual(len(rows), 1)

    def test_running_the_pipeline_twice_adds_nothing_the_second_time(self):
        self.write_legacy()
        pl.migrate_log(self.log, self.dir / "backups")
        fixtures = [fx("24/10/2026", "15:00"), fx("25/10/2026", "16:30", home="Gamma", away="Delta")]
        for expected in (2, 0):
            _, existing = pl.read_log(self.log)
            p = plan(fixtures, existing=existing)
            self.assertEqual(len(p.new_rows), expected)
            pl.append_rows(self.log, p.new_rows)
        _, rows = pl.read_log(self.log)
        self.assertEqual(len(rows), 4)          # 2 legacy + 2 new, no duplicates
        keys = [pl.row_key(r) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_old_rows_are_reported_as_unverifiable_and_new_rows_as_verified(self):
        self.write_legacy()
        pl.migrate_log(self.log, self.dir / "backups")
        p = plan([fx("24/10/2026", "15:00")])
        pl.append_rows(self.log, p.new_rows)
        _, rows = pl.read_log(self.log)
        self.assertEqual([pl.timing_status(r) for r in rows],
                         ["legacy_unverified", "legacy_unverified", "verified_pre_kickoff"])

    def test_timing_status_detects_violations_and_bad_values(self):
        late = {"kickoff_utc": "2026-10-24T14:00:00+00:00", "predicted_at_utc": "2026-10-24T14:00:01+00:00"}
        junk = {"kickoff_utc": "soon", "predicted_at_utc": "now"}
        naive = {"kickoff_utc": "2026-10-24T14:00:00", "predicted_at_utc": "2026-10-20T09:00:00"}
        self.assertEqual(pl.timing_status(late), "violation")
        self.assertEqual(pl.timing_status(junk), "invalid_timestamp")
        self.assertEqual(pl.timing_status(naive), "invalid_timestamp")

    def test_pandas_can_still_read_a_migrated_log_with_the_old_column_selection(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        self.write_legacy()
        pl.migrate_log(self.log, self.dir / "backups")
        df = pd.read_csv(self.log, parse_dates=["Date"])
        sub = df[["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA", "model", "logged_on"]]
        self.assertEqual(len(sub), 2)
        self.assertAlmostEqual(float(df.loc[0, "pH"]), 0.6721)


if __name__ == "__main__":
    unittest.main()