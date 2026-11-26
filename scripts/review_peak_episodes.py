"""Plot selected 2024 series and count episode starts by article."""

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median

import matplotlib.dates as mdates
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
YEAR = "2024"
EXAMPLES = [
    ("227983", "37837437"),  # Davis Cup, Mars 2020
    ("1401596", "5024887"),  # 24 Hours of Le Mans, Solar eclipse
    ("2150841", "736"),      # Lionel Messi, Albert Einstein
]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    processed = ROOT / "data/processed"
    articles = {
        row["page_id"]: row for row in read_csv(ROOT / "data/raw/articles.csv")
    }
    daily = defaultdict(list)
    for row in read_csv(processed / "peak_days.csv"):
        if row["date"][:4] == YEAR:
            daily[row["page_id"]].append(row)
    counts = Counter(
        row["page_id"]
        for row in read_csv(processed / "peak_episodes.csv")
        if row["start_date"][:4] == YEAR
    )

    report = ROOT / "reports/peak_review_2024"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "episodes_by_article.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["page_id", "title", "domain", "median_daily_views", "episodes"])
        for page_id, article in articles.items():
            traffic = median(int(row["views"]) for row in daily[page_id])
            writer.writerow(
                [page_id, article["title"], article["domain"], traffic, counts[page_id]]
            )

    figure = Figure(figsize=(12, 8), layout="constrained")
    axes = figure.subplots(3, 2, sharex=True)
    for panel, ids in zip(axes, EXAMPLES):
        for axis, page_id in zip(panel, ids):
            rows = daily[page_id]
            dates = [date.fromisoformat(row["date"]) for row in rows]
            views = [int(row["views"]) for row in rows]
            thresholds = [float(row["threshold"]) if row["threshold"] else None for row in rows]
            starts = [i for i, row in enumerate(rows) if row["peak_start"] == "1"]
            axis.plot(dates, views, color="#205976", linewidth=1.2, label="Visitas")
            axis.plot(dates, thresholds, color="#a55a1a", linestyle="--", label="Umbral")
            axis.scatter(
                [dates[i] for i in starts], [views[i] for i in starts],
                s=25, color="#9c2c42", zorder=3, label="Inicio",
            )
            axis.set_title(articles[page_id]["title"])
            axis.set_yscale("log")
            axis.grid(axis="y", alpha=0.2)
    axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    figure.supxlabel("Mes de 2024")
    figure.supylabel("Visitas diarias (escala logarítmica)")
    figure.savefig(report / "series.png", dpi=160)


if __name__ == "__main__":
    main()
