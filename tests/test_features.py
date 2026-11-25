"""Check window boundaries and invariance to future observations."""
import unittest
import numpy as np
import pandas as pd
from src.train_validation import build_dataset, FEATURES


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(dict(page_id=1, date=pd.date_range("2023-01-01", periods=70),
                                     views=np.arange(70), domain="sport", topic="tennis",
                                     topic_type="event"))

    def test_exact_windows(self):
        row = build_dataset(self.raw).iloc[0]
        self.assertEqual(row.date, pd.Timestamp("2023-02-06"))
        self.assertEqual(row.target_views, 37)
        self.assertEqual(row.lag_1, 36)
        self.assertEqual(row.lag_7, 30)
        self.assertEqual(row.mean_30, np.mean(np.arange(7, 37)))
        self.assertAlmostEqual(row.peak_intensity_7, 37 / (1 + np.median(np.arange(30))))
        logs = np.log1p(np.arange(7, 37))
        self.assertAlmostEqual(row.trend_30, np.polyfit(np.arange(30), logs, 1)[0])
        self.assertAlmostEqual(row.volatility_iqr_30, np.quantile(logs, .75) - np.quantile(logs, .25))

    def test_future_cannot_change_features(self):
        original = build_dataset(self.raw).iloc[0]
        self.raw.loc[self.raw.date > original.date, "views"] = 999999
        changed = build_dataset(self.raw).iloc[0]
        pd.testing.assert_series_equal(original[FEATURES], changed[FEATURES])
        self.assertEqual(changed.target_views, 999999)

    def test_missing_day_rejected(self):
        with self.assertRaises(ValueError):
            build_dataset(self.raw.drop(index=10))


if __name__ == "__main__":
    unittest.main()
