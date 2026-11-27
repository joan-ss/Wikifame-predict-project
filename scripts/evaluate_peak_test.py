"""Predict the held-out test and evaluate fixed validation selections."""

import hashlib
import json
from datetime import timedelta
from statistics import median

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from .train_peak_models import ROOT, load_data, metrics
    from .split_peak_dataset import write_csv
except ImportError:
    from train_peak_models import ROOT, load_data, metrics
    from split_peak_dataset import write_csv


def episode_results(frame, episodes, alerts):
    keys = {(r.page_id, pd.Timestamp(r.date).date()): bool(alert)
            for r, alert in zip(frame.itertuples(index=False), alerts)}
    records = []
    for episode in episodes.itertuples(index=False):
        day = pd.Timestamp(episode.start_date).date()
        if not pd.Timestamp("2025-07-01").date() <= day <= pd.Timestamp("2025-12-31").date():
            continue
        prior = [(episode.page_id, day - timedelta(days=i)) for i in range(7, 0, -1)]
        evaluable = all(key in keys for key in prior)
        leads = [(day - key[1]).days for key in prior if keys.get(key, False)] if evaluable else []
        records.append(dict(page_id=episode.page_id, start_date=str(day),
                            evaluable=int(evaluable), covered=int(bool(leads)) if evaluable else None,
                            lead_days=max(leads) if leads else None))
    return records


def episode_summary(records):
    eligible = [r for r in records if r["evaluable"]]
    covered = [r for r in eligible if r["covered"]]
    return dict(episodes=len(records), evaluable_episodes=len(eligible),
                excluded_episodes=len(records)-len(eligible), covered_episodes=len(covered),
                episode_coverage=len(covered)/len(eligible) if eligible else None,
                median_lead_days=median(r["lead_days"] for r in covered) if covered else None)


def main():
    processed = ROOT / "data/processed"
    path = processed / "peak_test.csv"
    test = load_data(path, "2025-07-01", "2025-12-31")
    episodes = pd.read_csv(processed / "peak_episodes.csv", dtype={"page_id": str})
    manifest_path = ROOT / "models/peak/selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Refuse changed training/validation inputs; no fitting or threshold search occurs here.
    for name, digest in manifest["inputs"].items():
        if hashlib.sha256((processed / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Selection input has changed: {name}")
    predictions = test[["page_id", "date", "target_start", "target_end", "domain", "peak_next_7d"]].copy()
    summary, by_article, all_episodes, hashes = [], [], [], {}
    with threadpool_limits(limits=2):
        for key, config in manifest["selected"].items():
            model_path = ROOT / f"models/peak/{key}.joblib"
            hashes[key] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            bundle = joblib.load(model_path)
            if bundle["candidate_id"] != config["candidate_id"] or bundle["metrics"]["threshold"] != config["metrics"]["threshold"]:
                raise ValueError(f"Model/selection mismatch: {key}")
            probabilities = np.full(len(test), np.nan)
            for domain, model in bundle["models"].items():
                mask = np.ones(len(test), dtype=bool) if config["strategy"] == "global" else test.domain.eq(domain).to_numpy()
                if mask.any():
                    probabilities[mask] = model.predict_proba(test.loc[mask])[:, 1]
            if not np.isfinite(probabilities).all():
                raise ValueError(f"Missing predictions: {key}")
            threshold = config["metrics"]["threshold"]
            alerts = probabilities >= threshold
            predictions[key + "_probability"] = probabilities
            predictions[key + "_alert"] = alerts.astype(int)
            records = episode_results(test, episodes, alerts)
            all_episodes.extend(dict(model=key, **r) for r in records)
            for domain in ["all"] + sorted(test.domain.unique()):
                mask = np.ones(len(test), dtype=bool) if domain == "all" else test.domain.eq(domain).to_numpy()
                subset = test.loc[mask]
                ids = set(subset.page_id)
                row = dict(model=key, algorithm=config["algorithm"], strategy=config["strategy"],
                           calendar=config["calendar"], domain=domain,
                           **metrics(subset.peak_next_7d.to_numpy(), probabilities[mask], threshold),
                           **episode_summary([r for r in records if r["page_id"] in ids]))
                row["alerts_per_week"] = int(alerts[mask].sum()) / (subset.date.nunique()/7)
                summary.append(row)
            for page_id in sorted(test.page_id.unique()):
                mask = test.page_id.eq(page_id).to_numpy()
                subset = test.loc[mask]
                y = subset.peak_next_7d.to_numpy()
                # AP and recall have no meaning when the article has no positive windows.
                if y.sum():
                    scores = metrics(y, probabilities[mask], threshold)
                else:
                    scores = dict(rows=len(y), positive_rows=0, prevalence=0,
                                  average_precision=None, threshold=threshold,
                                  precision=0.0, recall=None, f1=0.0,
                                  alerts=int(alerts[mask].sum()), tn=int((~alerts[mask]).sum()),
                                  fp=int(alerts[mask].sum()), fn=0, tp=0)
                by_article.append(dict(model=key, page_id=page_id, title=subset.title.iloc[0],
                                       domain=subset.domain.iloc[0], **scores,
                                       **episode_summary([r for r in records if r["page_id"] == page_id]),
                                       alerts_per_week=int(alerts[mask].sum())/(subset.date.nunique()/7)))
            assert hashlib.sha256(model_path.read_bytes()).hexdigest() == hashes[key]
    report = ROOT / "reports/peak_test"
    write_csv(processed / "test_predictions.csv", list(predictions.columns), predictions.to_dict("records"))
    for name, rows in [("summary", summary), ("by_article", by_article), ("episodes", all_episodes)]:
        write_csv(report / f"{name}.csv", list(rows[0]), rows)
    audit = dict(test_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_sha256=hashes,
                 selection_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 refit=False, threshold_retuned=False,
                 alerts_per_week_definition="alerts / (number of distinct prediction dates / 7)",
                 inference="fixed models on daily historical features; vectorized, no feedback or refitting")
    (report / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for row in summary:
        if row["domain"] == "all":
            print(row, flush=True)


if __name__ == "__main__":
    main()
