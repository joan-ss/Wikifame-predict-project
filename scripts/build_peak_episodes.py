"""Detect high-traffic days and episode starts from the 30-day reference."""

import csv
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
DAILY_FIELDS = ["page_id", "date", "views", "threshold", "is_high", "peak_start"]
EPISODE_FIELDS = ["page_id", "start_date", "end_date"]
ONE_DAY = timedelta(days=1)


def classify_days(rows):
    previous_page = previous_day = previous_high = None
    for row in rows:
        page_id = row["page_id"]
        day = date.fromisoformat(row["date"])
        views = int(row["views"])
        threshold = is_high = peak_start = None
        if row["median_30"] and row["iqr_30"]:
            median = float(row["median_30"])
            iqr = float(row["iqr_30"])
            threshold = max(2 * median, median + 3 * iqr)
            is_high = int(views > threshold)
            if not is_high:
                peak_start = 0
            elif page_id == previous_page and previous_day + ONE_DAY == day:
                if previous_high is not None:
                    peak_start = 1 - previous_high
        yield page_id, day, views, threshold, is_high, peak_start
        previous_page, previous_day, previous_high = page_id, day, is_high


def collect_episodes(daily_rows):
    for page_id, rows in groupby(daily_rows, key=lambda row: row[0]):
        start = end = None
        for _, day, _, _, is_high, peak_start in rows:
            if start is not None and (not is_high or day != end + ONE_DAY):
                yield page_id, start, end
                start = None
            if peak_start == 1:
                start = day
            if start is not None:
                end = day
        if start is not None:
            yield page_id, start, end


def main():
    directory = ROOT / "data/processed"
    with (directory / "peak_reference.csv").open(encoding="utf-8", newline="") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: (row["page_id"], row["date"]))
    daily_rows = list(classify_days(rows))
    outputs = [
        ("peak_days.csv", DAILY_FIELDS, daily_rows),
        ("peak_episodes.csv", EPISODE_FIELDS, collect_episodes(daily_rows)),
    ]
    with TemporaryDirectory(dir=directory) as temporary:
        for name, fields, records in outputs:
            with (Path(temporary) / name).open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(records)
        for name, _, _ in outputs:
            (Path(temporary) / name).replace(directory / name)


if __name__ == "__main__":
    main()
