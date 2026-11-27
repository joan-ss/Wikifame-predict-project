"""Create chronological splits and audit row and episode counts."""

import csv
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    from .build_peak_prediction_dataset import FEATURES, FIELDS, METADATA
except ImportError:
    from build_peak_prediction_dataset import FEATURES, FIELDS, METADATA


ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    "train": (date(2024, 1, 1), date(2024, 12, 31)),
    "validation": (date(2025, 1, 1), date(2025, 6, 30)),
    "test": (date(2025, 7, 1), date(2025, 12, 31)),
}


def split_rows(rows):
    splits = {name: [] for name in PERIODS}
    excluded = []
    seen = set()
    for row in rows:
        day = date.fromisoformat(row["date"])
        start = date.fromisoformat(row["target_start"])
        end = date.fromisoformat(row["target_end"])
        key = row["page_id"], day
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if start != day + timedelta(days=1) or end != day + timedelta(days=7):
            raise ValueError(f"Invalid horizon: {key}")
        if row["peak_next_7d"] not in ("", "0", "1"):
            raise ValueError(f"Invalid label: {key}")
        name = next((name for name, (lo, hi) in PERIODS.items()
                     if lo <= start and end <= hi), None)
        # Mutually exclusive audit reasons, in this documented order.
        if end < PERIODS["train"][0]:
            reason = "historical_only"
        elif name is None:
            reason = "horizon_outside_period"
        elif not all(row.get(field, "") != "" for field in FEATURES + METADATA + ["peak_next_7d"]):
            reason = "incomplete_data"
        else:
            splits[name].append(row)
            continue
        excluded.append({**row, "exclusion_reason": reason})
    for rows in splits.values():
        rows.sort(key=lambda row: (row["date"], row["page_id"]))
    return splits, excluded


def summarize(splits, episodes, articles):
    domains = sorted({article["domain"] for article in articles.values()})
    for name, rows in splits.items():
        lo, hi = PERIODS[name]
        for domain in ["all"] + domains:
            selected = [r for r in rows if domain == "all" or r["domain"] == domain]
            keys = {(r["page_id"], date.fromisoformat(r["date"])) for r in selected}
            starts = [(e["page_id"], date.fromisoformat(e["start_date"]))
                      for e in episodes
                      if lo <= date.fromisoformat(e["start_date"]) <= hi
                      and (domain == "all" or articles[e["page_id"]]["domain"] == domain)]
            evaluable = sum(all((pid, day - timedelta(days=i)) in keys
                                for i in range(1, 8)) for pid, day in starts)
            positives = sum(r["peak_next_7d"] == "1" for r in selected)
            yield dict(split=name, domain=domain, target_period_start=lo,
                       target_period_end=hi, rows=len(selected), positive_rows=positives,
                       negative_rows=len(selected) - positives,
                       articles=len({r["page_id"] for r in selected}),
                       first_prediction=min((r["date"] for r in selected), default=""),
                       last_prediction=max((r["date"] for r in selected), default=""),
                       episodes=len(starts), evaluable_episodes=evaluable,
                       excluded_episodes=len(starts) - evaluable)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    processed = ROOT / "data/processed"
    splits, excluded = split_rows(read_csv(processed / "peak_prediction_dataset.csv"))
    articles = {r["page_id"]: r for r in read_csv(ROOT / "data/raw/articles.csv")}
    summary = list(summarize(splits, read_csv(processed / "peak_episodes.csv"), articles))
    for name, rows in splits.items():
        write_csv(processed / f"peak_{name}.csv", FIELDS, rows)
    write_csv(processed / "peak_split_exclusions.csv", FIELDS + ["exclusion_reason"], excluded)
    write_csv(ROOT / "reports/peak_splits/summary.csv", list(summary[0]), summary)
    print("Excluded rows:", dict(Counter(r["exclusion_reason"] for r in excluded)))
    for row in summary:
        if row["domain"] == "all":
            print(row)


if __name__ == "__main__":
    main()
