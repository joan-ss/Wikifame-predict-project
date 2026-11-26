"""Check forecast boundaries and unknown observations."""

import unittest
from datetime import date, timedelta

from scripts.build_peak_labels import label_rows


def rows(values, page_id="a", start=date(2024, 12, 28)):
    return [{"page_id": page_id, "date": str(start + timedelta(days=i)),
             "peak_start": "" if value is None else str(value)}
            for i, value in enumerate(values)]


class PeakLabelsTest(unittest.TestCase):
    def test_horizon_includes_days_one_and_seven_but_not_today_or_eight(self):
        for position, expected in [(0, 0), (1, 1), (7, 1), (8, 0)]:
            with self.subTest(position=position):
                values = [0] * 9
                values[position] = 1
                result = list(label_rows(rows(values)))[0]
                self.assertEqual(result[-1], expected)
                self.assertEqual(result[2:4], (date(2024, 12, 29), date(2025, 1, 4)))

    def test_missing_unknown_and_incomplete_horizons(self):
        source = rows([0] * 10)
        del source[3]
        self.assertIsNone(list(label_rows(source))[0][-1])
        self.assertIsNone(list(label_rows(rows([0, 1, None] + [0] * 5)))[0][-1])
        result = list(label_rows(rows([None] + [0] * 7)))
        self.assertEqual(result[0][-1], 0)
        self.assertTrue(all(row[-1] is None for row in result[-7:]))

    def test_articles_are_isolated_and_input_can_be_unsorted(self):
        source = rows([0, 1] + [0] * 6) + rows([0] * 8, page_id="b")
        result = list(label_rows(reversed(source)))
        self.assertEqual(result[0][-1], 1)
        self.assertEqual(result[8][-1], 0)

    def test_duplicate_and_invalid_values_are_rejected(self):
        source = rows([0] * 8)
        with self.assertRaises(ValueError):
            list(label_rows(source + source[:1]))
        with self.assertRaises(ValueError):
            list(label_rows(rows([2])))


if __name__ == "__main__":
    unittest.main()
