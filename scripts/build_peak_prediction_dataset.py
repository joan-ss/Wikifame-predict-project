"""Join historical predictors, calendar and metadata to seven-day labels."""

import csv
from datetime import date, timedelta
from math import log1p
from pathlib import Path
from statistics import mean, median, quantiles
from tempfile import NamedTemporaryFile

try:
    from .build_peak_reference import read_pageviews
except ImportError:
    from build_peak_reference import read_pageviews


ROOT = Path(__file__).resolve().parents[1]
METADATA = ["qid", "title", "domain", "topic", "topic_type"]
FEATURES = [f"lag_{i}" for i in range(1, 8)] + [
    "mean_3", "mean_7", "mean_30", "trend_30", "volatility_iqr_30",
    "peak_intensity_7", "forecast_day_of_week", "target_week_of_year",
]
FIELDS = ["page_id", "date", "target_start", "target_end", "peak_next_7d"] + METADATA + FEATURES


def historical_features(daily, day):
    """Use calendar offsets, never later observations or adjacent row positions."""
    history = [daily.get(day - timedelta(days=i)) for i in range(37)]
    result = {f"lag_{i + 1}": history[i] for i in range(7)}
    for window in (3, 7, 30):
        values = history[:window]
        result[f"mean_{window}"] = None if None in values else mean(values)
    result["trend_30"] = result["volatility_iqr_30"] = None
    if None not in history[:30]:
        logs = [log1p(value) for value in reversed(history[:30])]
        # OLS slope on chronological days 0..29; centered x sums to zero.
        result["trend_30"] = sum((i - 14.5) * y for i, y in enumerate(logs)) / 2247.5
        q1, _, q3 = quantiles(logs, n=4, method="inclusive")
        result["volatility_iqr_30"] = q3 - q1
    result["peak_intensity_7"] = None
    if None not in history:
        result["peak_intensity_7"] = (1 + max(history[:7])) / (1 + median(history[7:37]))
    result["forecast_day_of_week"] = day.weekday()
    result["target_week_of_year"] = (day + timedelta(days=1)).isocalendar().week
    return result


def dataset_rows(labels, series, articles):
    seen = set()
    for label in labels:
        page_id = label["page_id"]
        day = date.fromisoformat(label["date"])
        key = page_id, day
        if key in seen:
            raise ValueError(f"Duplicate label for article {page_id} on {day}")
        seen.add(key)
        if page_id not in articles or page_id not in series or day not in series[page_id]:
            raise ValueError(f"Missing article or pageviews for {page_id} on {day}")
        if label["peak_next_7d"] not in ("", "0", "1"):
            raise ValueError(f"Invalid label for {page_id} on {day}")
        if (label["target_start"] != str(day + timedelta(days=1)) or
                label["target_end"] != str(day + timedelta(days=7))):
            raise ValueError(f"Invalid horizon for {page_id} on {day}")
        metadata = {field: articles[page_id][field] for field in METADATA}
        if not all(metadata.values()):
            raise ValueError(f"Incomplete metadata for article {page_id}")
        yield {**label, **metadata, **historical_features(series[page_id], day)}


def main():
    series = read_pageviews(ROOT / "data/raw/pageviews.csv")
    with (ROOT / "data/raw/articles.csv").open(encoding="utf-8", newline="") as file:
        articles = {}
        for row in csv.DictReader(file):
            if row["page_id"] in articles:
                raise ValueError(f"Duplicate article {row['page_id']}")
            articles[row["page_id"]] = row
    output = ROOT / "data/processed/peak_prediction_dataset.csv"
    temporary = None
    try:
        with (output.parent / "peak_labels.csv").open(encoding="utf-8", newline="") as source:
            # A sibling file inherits the processed directory's user permissions.
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                    dir=output.parent, delete=False) as file:
                temporary = Path(file.name)
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(dataset_rows(csv.DictReader(source), series, articles))
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
