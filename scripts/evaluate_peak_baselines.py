"""Compare the two planned baselines; choose thresholds on validation only."""

import hashlib
import json
from datetime import timedelta
from statistics import median

import numpy as np
import pandas as pd

try:
    from .train_peak_models import ROOT, load_data, best_threshold, metrics
    from .build_peak_reference import read_pageviews
    from .evaluate_peak_test import episode_results, episode_summary
    from .split_peak_dataset import write_csv
except ImportError:
    from train_peak_models import ROOT, load_data, best_threshold, metrics
    from build_peak_reference import read_pageviews
    from evaluate_peak_test import episode_results, episode_summary
    from split_peak_dataset import write_csv


def recent_rise(daily, day):
    recent = [daily.get(day - timedelta(days=i)) for i in range(3)]
    previous = [daily.get(day - timedelta(days=i)) for i in range(3, 33)]
    if None in recent or None in previous:
        raise ValueError(f"Incomplete baseline history on {day}")
    return (1 + sum(recent)/3) / (1 + median(previous))


def training_frequencies(train):
    return train.groupby("page_id")["peak_next_7d"].mean().to_dict()


def score_frame(frame, frequencies, series):
    historical = frame.page_id.map(frequencies)
    if historical.isna().any():
        raise ValueError("Article without training frequency")
    rise = [recent_rise(series[r.page_id], pd.Timestamp(r.date).date())
            for r in frame.itertuples(index=False)]
    return {"historical_frequency": historical.to_numpy(), "recent_rise": np.array(rise)}


def main():
    processed = ROOT / "data/processed"
    paths = [processed / f"peak_{name}.csv" for name in ("train", "validation", "test")]
    train = load_data(paths[0], "2024-01-01", "2024-12-31")
    validation = load_data(paths[1], "2025-01-01", "2025-06-30")
    test = load_data(paths[2], "2025-07-01", "2025-12-31")
    series = read_pageviews(ROOT / "data/raw/pageviews.csv")
    frequencies = training_frequencies(train)
    validation_scores = score_frame(validation, frequencies, series)
    # Freeze both thresholds before scoring or evaluating test.
    thresholds = {name: best_threshold(validation.peak_next_7d, scores)
                  for name, scores in validation_scores.items()}
    test_scores = score_frame(test, frequencies, series)
    episodes = pd.read_csv(processed / "peak_episodes.csv", dtype={"page_id": str})
    summary, details, article_results = [], [], []
    report = ROOT / "reports/peak_baselines"
    for split, frame, scores_by_name in [("validation", validation, validation_scores), ("test", test, test_scores)]:
        predictions = frame[["page_id", "date", "target_start", "target_end", "domain", "peak_next_7d"]].copy()
        for name, scores in scores_by_name.items():
            threshold = thresholds[name]
            alerts = scores >= threshold
            predictions[name + "_score"] = scores
            predictions[name + "_alert"] = alerts.astype(int)
            records = episode_results(frame, episodes, alerts) if split == "test" else []
            details.extend(dict(baseline=name, **r) for r in records)
            for domain in ["all"] + sorted(frame.domain.unique()):
                mask = np.ones(len(frame), dtype=bool) if domain == "all" else frame.domain.eq(domain).to_numpy()
                subset = frame.loc[mask]
                ids = set(subset.page_id)
                ep = episode_summary([r for r in records if r["page_id"] in ids]) if split == "test" else dict.fromkeys(episode_summary([]))
                summary.append(dict(split=split, model=name, domain=domain,
                                    **metrics(subset.peak_next_7d.to_numpy(), scores[mask], threshold),
                                    **ep, alerts_per_week=int(alerts[mask].sum())/(subset.date.nunique()/7)))
            if split == "test":
                for pid in sorted(frame.page_id.unique()):
                    mask = frame.page_id.eq(pid).to_numpy()
                    subset = frame.loc[mask]
                    # AP/recall undefined for an article without positive windows.
                    y = subset.peak_next_7d.to_numpy()
                    if y.sum():
                        stats = metrics(y, scores[mask], threshold)
                    else:
                        stats = dict(rows=len(y), positive_rows=0, prevalence=0, average_precision=None,
                                     threshold=threshold, precision=0.0, recall=None, f1=0.0,
                                     alerts=int(alerts[mask].sum()), tn=int((~alerts[mask]).sum()),
                                     fp=int(alerts[mask].sum()), fn=0, tp=0)
                    article_results.append(dict(model=name, page_id=pid, title=subset.title.iloc[0],
                                                domain=subset.domain.iloc[0], **stats,
                                                **episode_summary([r for r in records if r["page_id"] == pid])))
        write_csv(report / f"{split}_predictions.csv", list(predictions.columns), predictions.to_dict("records"))
    for name, rows in [("summary", summary), ("episodes", details), ("by_article", article_results)]:
        write_csv(report / f"{name}.csv", list(rows[0]), rows)
    audit = dict(comparison="complementary_after_test_observed", thresholds=thresholds,
                 threshold_selection="pooled validation F1; higher threshold on ties", frequencies=frequencies,
                 recent_rise="(1 + mean(t-2..t)) / (1 + median(t-32..t-3))",
                 inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths + [ROOT / "data/raw/pageviews.csv"]})
    (report / "selection.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for row in summary:
        if row["domain"] == "all":
            print(row)


if __name__ == "__main__":
    main()
