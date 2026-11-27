"""Build a standalone, offline report from existing validation results."""

import csv
import json
import math
from html import escape
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def case_plots(articles, predictions, daily):
    output = []
    keys = {(r["page_id"], r["date"]): r for r in predictions}
    for page_id in ("1401596", "5024887", "2150841", "37837437"):
        rows = [r for r in daily if r["page_id"] == page_id and "2025-06-30" <= r["date"] <= "2025-12-31"]
        maximum = max(math.log1p(max(float(r["views"]), float(r["threshold"] or 0))) for r in rows)
        def x(i):
            return 45 + i * 810 / (len(rows)-1)
        def y(value):
            return 220 - math.log1p(float(value)) / maximum * 190
        svg = '<svg viewBox="0 0 880 280" role="img" aria-label="Visites, llindar, inicis i alertes" style="width:100%;min-width:540px">'
        for field, color, dash in [("views", "#245bd6", ""), ("threshold", "#b66818", 'stroke-dasharray="5 4"')]:
            points = " ".join(f'{x(i):.1f},{y(r[field]):.1f}' for i, r in enumerate(rows) if r[field])
            svg += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6" {dash}/>'
        for i, row in enumerate(rows):
            if row["peak_start"] == "1":
                svg += f'<circle cx="{x(i):.1f}" cy="{y(row["views"]):.1f}" r="4" fill="#a52c4b"><title>Inici {row["date"]}: {row["views"]} visites</title></circle>'
            prediction = keys.get((page_id, row["date"]))
            if prediction and prediction["random_forest_global_alert"] == "1":
                svg += f'<path d="M{x(i):.1f} 237v10" stroke="#087f80" stroke-width="2"><title>Alerta emesa {row["date"]}</title></path>'
            if row["date"].endswith("-01"):
                svg += f'<text x="{x(i):.1f}" y="270" text-anchor="middle" font-size="13">{row["date"][5:7]}/2025</text>'
        svg += '</svg>'
        output.append(f'<div class="card" style="margin-top:16px"><h3>{escape(articles[page_id]["title"])}</h3><div class="scroll">{svg}</div></div>')
    return "".join(output)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    reports = ROOT / "reports"
    validation = reports / "peak_validation"
    selection = json.loads((ROOT / "models/peak/selection.json").read_text(encoding="utf-8"))
    articles = {r["page_id"]: r for r in read_csv(ROOT / "data/raw/articles.csv")}
    grouped = defaultdict(list)
    for row in read_csv(validation / "predictions.csv"):
        grouped[row["page_id"]].append(row)
    article_results = []
    for page_id, rows in grouped.items():
        for strategy, model in selection["winners"].items():
            positives = sum(r["peak_next_7d"] == "1" for r in rows)
            alerts = sum(r[model + "_alert"] == "1" for r in rows)
            tp = sum(r["peak_next_7d"] == "1" and r[model + "_alert"] == "1" for r in rows)
            article_results.append(dict(**articles[page_id], strategy=strategy, rows=len(rows),
                                        positive_rows=positives, alerts=alerts, tp=tp,
                                        fp=alerts-tp, fn=positives-tp,
                                        precision=tp/alerts if alerts else None,
                                        recall=tp/positives if positives else None))
    test_report = reports / "peak_test"
    data = dict(baselines=read_csv(reports / "peak_baselines/summary.csv"),
                test=read_csv(test_report / "summary.csv"),
                test_articles=read_csv(test_report / "by_article.csv"),
                selected=read_csv(validation / "selected.csv"),
                candidates=read_csv(validation / "candidates.csv"),
                splits=read_csv(reports / "peak_splits/summary.csv"),
                selection=selection, articles=article_results)
    template = (ROOT / "scripts/templates/results.html").read_text(encoding="utf-8")
    cases = case_plots(articles, read_csv(ROOT / "data/processed/test_predictions.csv"),
                       read_csv(ROOT / "data/processed/peak_days.csv"))
    template = template.replace("__TEST_CASES__", cases)
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    output = reports / "resultats.html"
    output.write_text(template.replace("__RESULTS_DATA__", payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
