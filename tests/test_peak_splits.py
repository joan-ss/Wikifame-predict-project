"""Verify temporal boundaries and episode eligibility."""

import unittest
from datetime import date, timedelta

from scripts.build_peak_prediction_dataset import FIELDS
from scripts.split_peak_dataset import split_rows, summarize


def prediction(day):
    row = dict.fromkeys(FIELDS, "1")
    row.update(page_id="a", domain="deporte", date=str(day),
               target_start=str(day + timedelta(days=1)),
               target_end=str(day + timedelta(days=7)))
    return row


class PeakSplitsTest(unittest.TestCase):
    def test_first_and_last_dates_and_crossing_horizons(self):
        cases = {
            "2023-12-30": None, "2023-12-31": "train",
            "2024-12-24": "train", "2024-12-25": None,
            "2024-12-30": None, "2024-12-31": "validation",
            "2025-06-23": "validation", "2025-06-24": None,
            "2025-06-29": None, "2025-06-30": "test",
            "2025-12-24": "test", "2025-12-25": None,
        }
        for value, expected in cases.items():
            with self.subTest(day=value):
                splits, excluded = split_rows([prediction(date.fromisoformat(value))])
                actual = next((k for k, rows in splits.items() if rows), None)
                self.assertEqual(actual, expected)
                self.assertEqual(len(excluded), int(expected is None))

    def test_incomplete_history_and_duplicate_handling(self):
        row = prediction(date(2024, 5, 1))
        row["mean_30"] = ""
        splits, excluded = split_rows([row, prediction(date(2023, 5, 1))])
        self.assertFalse(any(splits.values()))
        self.assertEqual([r["exclusion_reason"] for r in excluded],
                         ["incomplete_data", "historical_only"])
        with self.assertRaises(ValueError):
            split_rows([row, row])
        invalid = prediction(date(2024, 5, 1))
        invalid["target_end"] = "2024-05-09"
        with self.assertRaises(ValueError):
            split_rows([invalid])

    def test_episode_needs_all_seven_prior_predictions_of_same_article(self):
        rows = [prediction(date(2023, 12, 31) + timedelta(days=i)) for i in range(8)]
        splits, _ = split_rows(rows)
        episodes = [{"page_id": "a", "start_date": value} for value in
                    ["2024-01-01", "2024-01-07", "2024-01-08"]]
        articles = {"a": {"domain": "deporte"}}
        summary = list(summarize(splits, episodes, articles))[0]
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["evaluable_episodes"], 2)
        splits["train"] = [r for r in splits["train"] if r["date"] != "2024-01-03"]
        self.assertEqual(list(summarize(splits, episodes, articles))[0]["evaluable_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
