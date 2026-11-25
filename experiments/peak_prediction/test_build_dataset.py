"""Check that the peak label and the baseline use the intended dates."""

import unittest

import pandas as pd

from build_dataset import peak_windows


class PeakWindowTests(unittest.TestCase):
    def setUp(self):
        self.visits = pd.DataFrame({
            "page_id": 1,
            "date": pd.date_range("2023-11-01", "2025-07-08"),
            "views": 10,
        })

    def set_views(self, date, visits):
        self.visits.loc[self.visits.date.eq(date), "views"] = visits

    def test_exact_horizon_and_inclusive_threshold(self):
        self.set_views("2024-03-08", 30)
        self.set_views("2024-03-09", 100)
        row = peak_windows(self.visits).set_index("date").loc["2024-03-01"]
        self.assertEqual(row.window_start, pd.Timestamp("2024-03-02"))
        self.assertEqual(row.window_end, pd.Timestamp("2024-03-08"))
        self.assertEqual(row.reference_median_30, 10)
        self.assertEqual(row.future_max_7, 30)
        self.assertEqual(row.recent_peak_score, 1)
        self.assertEqual(row.peak_next_7, 1)

        self.set_views("2024-03-08", 29)
        updated = peak_windows(self.visits).set_index("date").loc["2024-03-01"]
        self.assertEqual(updated.peak_next_7, 0)

    def test_recent_peak_uses_earlier_reference(self):
        self.set_views("2024-03-01", 35)
        row = peak_windows(self.visits).set_index("date").loc["2024-03-01"]
        self.assertEqual(row.recent_peak_score, 3.5)
        self.assertEqual(row.peak_next_7, 0)

    def test_windows_stay_in_their_splits(self):
        self.set_views("2025-07-01", 10_000)
        windows = peak_windows(self.visits)
        train = windows[windows.split.eq("train")]
        validation = windows[windows.split.eq("validation")]
        self.assertEqual(train.window_end.max(), pd.Timestamp("2024-12-31"))
        self.assertEqual(validation.window_start.min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(validation.window_end.max(), pd.Timestamp("2025-06-30"))
        self.assertEqual(len(train), 360)
        self.assertEqual(len(validation), 175)
        self.assertFalse(validation.peak_next_7.any())


if __name__ == "__main__":
    unittest.main()
