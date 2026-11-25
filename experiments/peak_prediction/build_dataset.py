"""Build the seven-day peak classification dataset for train and validation."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "peak_dataset.csv"
HORIZON = 7
REFERENCE_DAYS = 30
PEAK_MULTIPLIER = 3
SPLITS = {
    "train": ("2024-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-06-30"),
}
NUMERIC = [
    *[f"lag_{day}" for day in range(1, 8)],
    "mean_3", "mean_7", "mean_30", "trend_30", "volatility_iqr_30",
    "peak_intensity_7", "target_day_of_week",
]
CATEGORICAL = ["domain", "topic", "topic_type"]
FEATURES = NUMERIC + CATEGORICAL
KEYS = ["page_id", "date"]


def peak_windows(pageviews):
    if pageviews.date.isna().any():
        raise ValueError("Daily visits need a date")
    cutoff = pd.Timestamp(SPLITS["validation"][1])
    pageviews = pageviews.loc[pageviews.date <= cutoff].copy()
    if pageviews[KEYS + ["views"]].isna().any().any() or pageviews.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate daily visits")
    if not np.isfinite(pageviews.views).all() or (pageviews.views < 0).any():
        raise ValueError("Visit counts must be finite and nonnegative")

    one_day = timedelta(days=1)
    windows = []
    for page_id, article in pageviews.groupby("page_id"):
        article = article.sort_values("date")
        if not article.date.diff().dropna().eq(one_day).all():
            raise ValueError(f"Non-consecutive dates for article {page_id}")

        visits = article.views
        reference = visits.rolling(REFERENCE_DAYS).median()
        peak_threshold = PEAK_MULTIPLIER * reference.clip(lower=1)
        future_maximum = visits.rolling(HORIZON).max().shift(-HORIZON)

        # The baseline uses only the last seven days and their own prior reference.
        recent_maximum = visits.rolling(HORIZON).max()
        recent_reference = visits.shift(HORIZON).rolling(REFERENCE_DAYS).median()
        baseline_score = recent_maximum / recent_reference.clip(lower=1)

        windows.append(article[KEYS].assign(
            window_start=article.date + one_day,
            window_end=article.date + timedelta(days=HORIZON),
            reference_median_30=reference,
            peak_threshold=peak_threshold,
            future_max_7=future_maximum,
            recent_peak_score=baseline_score,
        ))

    windows = pd.concat(windows, ignore_index=True)
    splits = []
    for split, (start, end) in SPLITS.items():
        mask = windows.window_start.ge(start) & windows.window_end.le(end)
        rows = windows.loc[mask].copy()
        if rows[["peak_threshold", "future_max_7", "recent_peak_score"]].isna().any().any():
            raise ValueError(f"Incomplete peak windows for {split}")
        rows["peak_next_7"] = rows.future_max_7.ge(rows.peak_threshold).astype(int)
        rows["split"] = split
        splits.append(rows)
    return pd.concat(splits, ignore_index=True)


def build_dataset(pageviews, features):
    windows = peak_windows(pageviews)
    feature_columns = KEYS + FEATURES
    dataset = windows.merge(features[feature_columns], on=KEYS, how="left", validate="one_to_one")
    if dataset[FEATURES].isna().any().any():
        raise ValueError("Each forecast window needs a complete feature row")
    return dataset.sort_values(["date", "page_id"]).reset_index(drop=True)


def main():
    pageviews = pd.read_csv(DATA_DIR / "pageviews.csv", parse_dates=["date"])
    features = pd.read_csv(
        DATA_DIR / "prediction_dataset.csv", usecols=KEYS + FEATURES, parse_dates=["date"],
    )
    dataset = build_dataset(pageviews, features)
    dataset.to_csv(DATASET_PATH, index=False)
    print(dataset.groupby("split").peak_next_7.agg(windows="size", peaks="sum").to_string())


if __name__ == "__main__":
    main()
