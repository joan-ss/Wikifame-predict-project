"""Calculate each article's trailing 30-day reference from raw pageviews."""

import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import quantiles
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
WINDOW_DAYS = 30


def read_pageviews(path):
    series = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            page_id = row["page_id"]
            day = date.fromisoformat(row["date"])
            views = int(row["views"])
            if day in series[page_id]:
                raise ValueError(f"Duplicate pageview for article {page_id} on {day}")
            if views < 0:
                raise ValueError(f"Negative pageviews for article {page_id} on {day}")
            series[page_id][day] = views
    return series


def reference_rows(series):
    lookback = timedelta(days=WINDOW_DAYS)
    for page_id, daily_views in series.items():
        observations = sorted(daily_views.items())
        for index, (day, views) in enumerate(observations):
            median_30 = iqr_30 = None
            start = index - WINDOW_DAYS
            # With unique dates, this check guarantees 30 consecutive days.
            if start >= 0 and observations[start][0] == day - lookback:
                history = [count for _, count in observations[start:index]]
                q1, median_30, q3 = quantiles(history, n=4, method="inclusive")
                iqr_30 = q3 - q1
            yield page_id, day, views, median_30, iqr_30


def main():
    series = read_pageviews(ROOT / "data/raw/pageviews.csv")
    output = ROOT / "data/processed/peak_reference.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary_output = Path(directory) / output.name
        with temporary_output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["page_id", "date", "views", "median_30", "iqr_30"])
            writer.writerows(reference_rows(series))
        temporary_output.replace(output)


if __name__ == "__main__":
    main()
