"""Locks the model label written to the prediction log, so a config change can't silently split history."""
import unittest
import model


class ModelLabel(unittest.TestCase):
    def test_label_matches_logged_history(self):
        # This exact string is already in data/*_predictions logs. If this fails, K/HA/NEW/WINDOW changed:
        # that is a NEW model, and check.py will (correctly) score it separately.
        self.assertEqual(model.MODEL_VERSION, "elo_sot10 K30 HA75 NEW1400")


if __name__ == "__main__":
    unittest.main()