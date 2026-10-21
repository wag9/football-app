"""Tests for check_reorder.compare on synthetic tables (no git, no real data)."""
import unittest

import pandas as pd

from check_reorder import compare


def table(rows):
    return pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTR"])


BASE = [["2026-08-01", "A", "B", "H"], ["2026-08-01", "C", "D", "D"], ["2026-08-02", "E", "F", "A"]]


class TestCompare(unittest.TestCase):
    def test_identical(self):
        removed, added, moved = compare(table(BASE), table(BASE))
        self.assertEqual((removed, added, moved), ([], [], 0))

    def test_pure_reorder_is_not_a_removal(self):
        new = [BASE[1], BASE[0], BASE[2]]
        removed, added, moved = compare(table(BASE), table(new))
        self.assertEqual((removed, added), ([], []))
        self.assertEqual(moved, 2)

    def test_new_rows_are_reported_as_added(self):
        new = BASE + [["2026-08-03", "G", "H", "H"]]
        removed, added, moved = compare(table(BASE), table(new))
        self.assertEqual(removed, [])
        self.assertEqual(len(added), 1)
        self.assertEqual(moved, 0)

    def test_changed_result_shows_up_as_removed_and_added(self):
        new = [BASE[0], ["2026-08-01", "C", "D", "H"], BASE[2]]
        removed, added, _ = compare(table(BASE), table(new))
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(added), 1)
        self.assertEqual(removed[0][3], "D")

    def test_deleted_row_is_reported(self):
        removed, added, _ = compare(table(BASE), table(BASE[:2]))
        self.assertEqual(len(removed), 1)
        self.assertEqual(added, [])

    def test_duplicate_rows_are_counted_not_collapsed(self):
        removed, _, _ = compare(table(BASE + [BASE[0]]), table(BASE))
        self.assertEqual(len(removed), 1)


if __name__ == "__main__":
    unittest.main()