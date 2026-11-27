"""Check feature windows, calculations and absence of future information."""

import unittest
from datetime import date, timedelta
from math import exp

from scripts.build_peak_prediction_dataset import dataset_rows, historical_features


class PredictionDatasetTest(unittest.TestCase):
    def setUp(self):
        self.start = date(2024, 11, 25)
        self.daily = {self.start + timedelta(days=i): i + 1 for i in range(45)}
        self.day = self.start + timedelta(days=36)

    def test_exact_windows_and_intensity_baseline(self):
        result = historical_features(self.daily, self.day)
        self.assertEqual([result[f"lag_{i}"] for i in range(1, 8)], list(range(37, 30, -1)))
        self.assertEqual(result["mean_3"], 36)
        self.assertEqual(result["mean_7"], 34)
        self.assertEqual(result["mean_30"], 22.5)
        self.assertAlmostEqual(result["peak_intensity_7"], 38 / 16.5)
        self.assertEqual(result["forecast_day_of_week"], 1)
        self.assertEqual(result["target_week_of_year"], 1)

    def test_logarithmic_trend_and_iqr(self):
        daily = {self.start + timedelta(days=i): exp(i / 10) - 1 for i in range(37)}
        result = historical_features(daily, self.day)
        self.assertAlmostEqual(result["trend_30"], 0.1)
        self.assertAlmostEqual(result["volatility_iqr_30"], 1.45)

    def test_future_changes_cannot_change_features(self):
        before = historical_features(self.daily, self.day)
        for day in self.daily:
            if day > self.day:
                self.daily[day] = 10**9
        self.assertEqual(before, historical_features(self.daily, self.day))

    def test_missing_calendar_days_are_not_bridged(self):
        del self.daily[self.day - timedelta(days=10)]
        result = historical_features(self.daily, self.day)
        self.assertEqual(result["mean_7"], 34)
        for field in ("mean_30", "trend_30", "volatility_iqr_30", "peak_intensity_7"):
            self.assertIsNone(result[field])
        self.assertIsNone(historical_features(self.daily, self.start)["lag_2"])

    def test_metadata_join_preserves_unknown_labels_and_rejects_duplicates(self):
        label = {"page_id": "a", "date": str(self.day),
                 "target_start": str(self.day + timedelta(days=1)),
                 "target_end": str(self.day + timedelta(days=7)), "peak_next_7d": ""}
        articles = {"a": dict(qid="Q1", title="Example", domain="deporte",
                              topic="fútbol", topic_type="evento")}
        result = list(dataset_rows([label], {"a": self.daily}, articles))[0]
        self.assertEqual(result["topic"], "fútbol")
        self.assertEqual(result["peak_next_7d"], "")
        self.assertNotIn("peak_start", result)
        with self.assertRaises(ValueError):
            list(dataset_rows([label, label], {"a": self.daily}, articles))
        with self.assertRaises(ValueError):
            list(dataset_rows([label], {"a": self.daily}, {}))


if __name__ == "__main__":
    unittest.main()
