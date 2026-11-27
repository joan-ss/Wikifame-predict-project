import unittest
import pandas as pd

from scripts.evaluate_peak_test import episode_results, episode_summary


class EpisodeEvaluationTest(unittest.TestCase):
    def test_earliest_alert_and_article_isolation(self):
        frame = pd.DataFrame({"page_id": ["a"]*7 + ["b"]*7,
                              "date": list(pd.date_range("2025-07-01", periods=7))*2})
        episodes = pd.DataFrame({"page_id": ["a", "b", "a"],
                                 "start_date": ["2025-07-08", "2025-07-08", "2025-07-01"]})
        records = episode_results(frame, episodes, [False, True, False, False, True, False, False]+[False]*7)
        self.assertEqual(records[0]["lead_days"], 6)
        self.assertEqual(records[1]["covered"], 0)
        self.assertEqual(records[2]["evaluable"], 0)
        summary = episode_summary(records)
        self.assertEqual(summary["episode_coverage"], 0.5)
        self.assertEqual(summary["median_lead_days"], 6)

    def test_gap_excludes_episode_and_no_alert_lead_is_undefined(self):
        frame = pd.DataFrame({"page_id": ["a"]*7,
                              "date": pd.date_range("2025-07-01", periods=7)})
        episodes = pd.DataFrame({"page_id": ["a"], "start_date": ["2025-07-08"]})
        result = episode_summary(episode_results(frame, episodes, [False]*7))
        self.assertEqual(result["episode_coverage"], 0)
        self.assertIsNone(result["median_lead_days"])
        records = episode_results(frame.drop(index=3), episodes, [True]*6)
        self.assertIsNone(episode_summary(records)["episode_coverage"])


if __name__ == "__main__":
    unittest.main()
