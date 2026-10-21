"""Tests for weekly.py helpers: download validation, retry rules, and the final verdict.
Synthetic data and a fake requests.get only; no network, no real data files."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import weekly

HEADER = b"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
ROW = b"E0,01/08/2026,15:00,A,B,1,0,H\n"


def csv_bytes(n):
    return HEADER + ROW * n


class FakeResponse:
    def __init__(self, status=200, content=b""):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class TestCountRows(unittest.TestCase):
    def test_counts_data_rows_only(self):
        self.assertEqual(weekly.count_rows(csv_bytes(3)), 3)

    def test_ignores_blank_and_comma_only_lines(self):
        self.assertEqual(weekly.count_rows(csv_bytes(2) + b"\n,,,,,,,\n   \n"), 2)

    def test_handles_bom_and_crlf(self):
        data = b"\xef\xbb\xbfDiv,Date\r\nE0,x\r\nE0,y\r\n"
        self.assertEqual(weekly.count_rows(data), 2)

    def test_empty_content(self):
        self.assertEqual(weekly.count_rows(b""), 0)


class TestValidateDownload(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "season.csv"
        self.path.write_bytes(csv_bytes(5))

    def tearDown(self):
        self.dir.cleanup()

    def test_rejects_html(self):
        with self.assertRaises(ValueError):
            weekly.validate_download(b"<html>blocked</html>")

    def test_accepts_bom_header(self):
        self.assertEqual(weekly.validate_download(b"\xef\xbb\xbf" + csv_bytes(2)), 2)

    def test_refuses_shrink_when_not_allowed(self):
        with self.assertRaises(ValueError) as cm:
            weekly.validate_download(csv_bytes(4), self.path, allow_shrink=False)
        self.assertIn("fewer rows", str(cm.exception))

    def test_allows_same_or_more_rows(self):
        self.assertEqual(weekly.validate_download(csv_bytes(5), self.path, allow_shrink=False), 5)
        self.assertEqual(weekly.validate_download(csv_bytes(6), self.path, allow_shrink=False), 6)

    def test_shrink_allowed_for_fixtures(self):
        self.assertEqual(weekly.validate_download(csv_bytes(1), self.path, allow_shrink=True), 1)

    def test_no_existing_file_is_fine(self):
        self.assertEqual(weekly.validate_download(csv_bytes(2), self.path.with_name("none.csv"), allow_shrink=False), 2)


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = str(Path(self.dir.name) / "out.csv")
        sleeper = mock.patch.object(weekly.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def tearDown(self):
        self.dir.cleanup()

    def test_success_writes_file_and_returns_rows(self):
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(200, csv_bytes(3))):
            self.assertEqual(weekly.fetch("http://x", self.out), 3)
        self.assertEqual(Path(self.out).read_bytes(), csv_bytes(3))
        self.assertFalse(Path(self.out + ".part").exists())

    def test_403_is_final_no_retry(self):
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(403)) as g:
            with self.assertRaises(requests.HTTPError):
                weekly.fetch("http://x", self.out)
        self.assertEqual(g.call_count, 1)
        self.assertFalse(Path(self.out).exists())

    def test_404_is_final_no_retry(self):
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(404)) as g:
            with self.assertRaises(requests.HTTPError):
                weekly.fetch("http://x", self.out)
        self.assertEqual(g.call_count, 1)

    def test_500_is_retried_until_success(self):
        seq = [FakeResponse(500), FakeResponse(503), FakeResponse(200, csv_bytes(2))]
        with mock.patch.object(weekly.requests, "get", side_effect=seq) as g:
            self.assertEqual(weekly.fetch("http://x", self.out), 2)
        self.assertEqual(g.call_count, 3)

    def test_connection_error_gives_up_after_retries(self):
        with mock.patch.object(weekly.requests, "get", side_effect=requests.ConnectionError("down")) as g:
            with self.assertRaises(requests.ConnectionError):
                weekly.fetch("http://x", self.out)
        self.assertEqual(g.call_count, weekly.RETRIES)

    def test_failed_download_leaves_existing_file_untouched(self):
        Path(self.out).write_bytes(csv_bytes(5))
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(200, b"<html>no</html>")):
            with self.assertRaises(ValueError):
                weekly.fetch("http://x", self.out, allow_shrink=False)
        self.assertEqual(Path(self.out).read_bytes(), csv_bytes(5))

    def test_old_file_survives_if_the_final_replace_fails(self):
        Path(self.out).write_bytes(csv_bytes(5))
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(200, csv_bytes(6))):
            with mock.patch.object(weekly.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    weekly.fetch("http://x", self.out)
        self.assertEqual(Path(self.out).read_bytes(), csv_bytes(5))

    def test_shrunk_download_does_not_replace_file(self):
        Path(self.out).write_bytes(csv_bytes(5))
        with mock.patch.object(weekly.requests, "get", return_value=FakeResponse(200, csv_bytes(2))):
            with self.assertRaises(ValueError):
                weekly.fetch("http://x", self.out, allow_shrink=False)
        self.assertEqual(Path(self.out).read_bytes(), csv_bytes(5))


class TestVerdict(unittest.TestCase):
    ALL_OK = {s: "ok" for s in weekly.STEPS}

    def test_all_ok(self):
        lines, code = weekly.verdict({"E0": dict(self.ALL_OK), "SP1": dict(self.ALL_OK)}, True, True)
        self.assertEqual(code, 0)
        self.assertIn("ALL OK", lines[-1])

    def test_all_downloads_failed_means_do_not_commit(self):
        st = {"E0": {"download": "FAILED"}, "SP1": {"download": "FAILED"}}
        lines, code = weekly.verdict(st, False, True)
        self.assertEqual(code, 1)
        self.assertTrue(any("NOTHING WAS UPDATED" in ln for ln in lines))

    def test_one_league_failed_is_partial(self):
        st = {"E0": dict(self.ALL_OK), "SP1": {"download": "FAILED"}}
        lines, code = weekly.verdict(st, True, True)
        self.assertEqual(code, 1)
        self.assertTrue(any("PARTIAL" in ln and "SP1" in ln for ln in lines))

    def test_audit_errors_count_as_problem(self):
        bad = dict(self.ALL_OK, audit="ERRORS")
        bad.pop("check"); bad.pop("log")
        lines, code = weekly.verdict({"E0": bad}, True, True)
        self.assertEqual(code, 1)
        self.assertTrue(any("PARTIAL" in ln for ln in lines))

    def test_failed_fixtures_download_is_warned_and_not_all_ok(self):
        lines, code = weekly.verdict({"E0": dict(self.ALL_OK)}, False, True)
        self.assertEqual(code, 1)
        self.assertTrue(any("fixtures.csv was NOT updated" in ln for ln in lines))

    def test_no_download_mode_ignores_fixtures_flag(self):
        st = {"E0": dict(self.ALL_OK, download="skipped")}
        _, code = weekly.verdict(st, False, False)
        self.assertEqual(code, 0)

    def test_check_or_log_failure_is_a_problem(self):
        _, code = weekly.verdict({"E0": dict(self.ALL_OK, log="FAILED")}, True, True)
        self.assertEqual(code, 1)
        _, code = weekly.verdict({"E0": dict(self.ALL_OK, log="ok (+3)")}, True, True)
        self.assertEqual(code, 0)


class TestMain(unittest.TestCase):
    def test_unknown_league_is_rejected_before_anything_runs(self):
        with mock.patch.object(weekly, "fetch") as f:
            self.assertEqual(weekly.main(["XX9"]), 2)
        f.assert_not_called()


if __name__ == "__main__":
    unittest.main()