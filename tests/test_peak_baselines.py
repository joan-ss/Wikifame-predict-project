import unittest
from datetime import date, timedelta
import pandas as pd
from scripts.evaluate_peak_baselines import recent_rise, training_frequencies, score_frame


class BaselinesTest(unittest.TestCase):
    def test_recent_rise_exact_offsets_and_future_independence(self):
        day = date(2025, 1, 1)
        daily = {day - timedelta(days=i): 10 for i in range(33)}
        for i in range(3):
            daily[day-timedelta(days=i)] = 21+i
        self.assertEqual(recent_rise(daily, day), 23/11)
        daily[day+timedelta(days=1)] = 10**9
        self.assertEqual(recent_rise(daily, day), 23/11)
        del daily[day-timedelta(days=32)]
        with self.assertRaises(ValueError):
            recent_rise(daily, day)

    def test_frequency_is_article_specific_and_rejects_unknown_articles(self):
        frame = pd.DataFrame({'page_id':['a','a','b','b'], 'peak_next_7d':[0,1,0,0]})
        self.assertEqual(training_frequencies(frame), {'a':0.5, 'b':0.0})
        with self.assertRaises(ValueError):
            score_frame(pd.DataFrame({'page_id':['unknown'], 'date':['2025-01-01']}),
                        training_frequencies(frame), {})


if __name__ == '__main__':
    unittest.main()
