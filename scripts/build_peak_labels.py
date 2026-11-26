"""Label whether an episode starts in the next seven calendar days."""

import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HORIZON_DAYS = 7
FIELDS = ["page_id", "date", "target_start", "target_end", "peak_next_7d"]


def label_rows(rows):
    series = defaultdict(dict)
    for row in rows:
        page_id = row["page_id"]
        day = date.fromisoformat(row["date"])
        value = row["peak_start"]
        if value not in ("", "0", "1"):
            raise ValueError(f"Invalid peak_start for article {page_id} on {day}")
        if day in series[page_id]:
            raise ValueError(f"Duplicate date for article {page_id} on {day}")
        series[page_id][day] = int(value) if value else None

    for page_id in sorted(series):
        daily = series[page_id]
        for day in sorted(daily):
            future = [daily.get(day + timedelta(days=offset))
                      for offset in range(1, HORIZON_DAYS + 1)]
            # Require all seven days, including at the end of the series.
            label = None if None in future else max(future)
            yield (page_id, day, day + timedelta(days=1),
                   day + timedelta(days=HORIZON_DAYS), label)


def main():
    directory = ROOT / "data/processed"
    with (directory / "peak_days.csv").open(encoding="utf-8", newline="") as file:
        labels = list(label_rows(csv.DictReader(file)))
    output = directory / "peak_labels.csv"
    with TemporaryDirectory(dir=directory) as temporary:
        path = Path(temporary) / output.name
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(FIELDS)
            writer.writerows(labels)
        path.replace(output)


if __name__ == "__main__":
    main()
