"""Select global and domain classifiers using training and validation only."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

try:
    from .build_peak_prediction_dataset import FEATURES
    from .split_peak_dataset import write_csv
except ImportError:
    from build_peak_prediction_dataset import FEATURES
    from split_peak_dataset import write_csv


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CALENDAR = ["forecast_day_of_week", "target_week_of_year"]
CATEGORICAL = ["domain", "topic", "topic_type"]
GRIDS = {
    "random_forest": [
        {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 5,
         "max_features": "sqrt", "class_weight": "balanced_subsample"},
        {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 20,
         "max_features": "sqrt", "class_weight": "balanced_subsample"},
    ],
    "hist_gradient_boosting": [
        {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 15,
         "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False},
        {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 31,
         "min_samples_leaf": 40, "l2_regularization": 3.0, "early_stopping": False},
    ],
}


def predictor_columns(calendar, strategy):
    numeric = [field for field in FEATURES if calendar or field not in CALENDAR]
    categorical = [field for field in CATEGORICAL if strategy == "global" or field != "domain"]
    return numeric, categorical


def build_pipeline(algorithm, params, calendar, strategy):
    numeric, categorical = predictor_columns(calendar, strategy)
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")
    if algorithm == "random_forest":
        model = RandomForestClassifier(**params, random_state=SEED, n_jobs=2)
    else:
        model = HistGradientBoostingClassifier(**params, random_state=SEED)
    return Pipeline([("preprocessing", preprocessing), ("classifier", model)])


def best_threshold(y, probabilities):
    """Maximize validation F1; prefer the higher threshold on ties."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                       out=np.zeros_like(thresholds), where=(precision[:-1] + recall[:-1]) != 0)
    return float(thresholds[np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))[-1]])


def metrics(y, probabilities, threshold):
    alerts = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, alerts, labels=[0, 1]).ravel()
    return dict(rows=len(y), positive_rows=int(np.sum(y)), prevalence=float(np.mean(y)),
                average_precision=float(average_precision_score(y, probabilities)),
                threshold=threshold, precision=float(precision_score(y, alerts, zero_division=0)),
                recall=float(recall_score(y, alerts, zero_division=0)),
                f1=float(f1_score(y, alerts, zero_division=0)), alerts=int(np.sum(alerts)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def fit_candidate(train, validation, algorithm, params, calendar, strategy):
    models = {}
    probabilities = np.empty(len(validation), dtype=float)
    domains = ["all"] if strategy == "global" else sorted(train["domain"].unique())
    if strategy == "local" and not set(validation["domain"]).issubset(domains):
        raise ValueError("Validation contains a domain without a local training model")
    for domain in domains:
        training_mask = np.ones(len(train), dtype=bool) if strategy == "global" else train["domain"].eq(domain).to_numpy()
        validation_mask = np.ones(len(validation), dtype=bool) if strategy == "global" else validation["domain"].eq(domain).to_numpy()
        subset = train.loc[training_mask]
        if set(subset["peak_next_7d"]) != {0, 1}:
            raise ValueError(f"Both classes required in training: {domain}")
        model = build_pipeline(algorithm, params, calendar, strategy)
        model.fit(subset, subset["peak_next_7d"])
        if validation_mask.any():
            probabilities[validation_mask] = model.predict_proba(validation.loc[validation_mask])[:, 1]
        models[domain] = model
    return models, probabilities


def load_data(path, lo, hi):
    frame = pd.read_csv(path, dtype={"page_id": str, "qid": str})
    required = FEATURES + CATEGORICAL + ["page_id", "date", "target_start", "target_end", "peak_next_7d"]
    if frame[required].isna().any().any() or frame.duplicated(["page_id", "date"]).any():
        raise ValueError(f"Incomplete or duplicate input: {path}")
    dates = pd.to_datetime(frame["date"])
    starts, ends = pd.to_datetime(frame["target_start"]), pd.to_datetime(frame["target_end"])
    if not ((starts == dates + pd.Timedelta(days=1)) &
            (ends == dates + pd.Timedelta(days=7)) &
            (starts >= pd.Timestamp(lo)) & (ends <= pd.Timestamp(hi))).all():
        raise ValueError(f"Invalid chronological split: {path}")
    if set(frame["peak_next_7d"]) != {0, 1}:
        raise ValueError(f"Both binary classes required: {path}")
    return frame


def main():
    processed = ROOT / "data/processed"
    train_path, validation_path = processed / "peak_train.csv", processed / "peak_validation.csv"
    train = load_data(train_path, "2024-01-01", "2024-12-31")
    validation = load_data(validation_path, "2025-01-01", "2025-06-30")
    y = validation["peak_next_7d"].to_numpy()
    report = ROOT / "reports/peak_validation"
    model_directory = ROOT / "models/peak"
    report.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    results, selected, winners = [], {}, {}
    predictions = validation[["page_id", "date", "target_start", "target_end", "domain", "peak_next_7d"]].copy()
    # Disable HGB's internal random holdout; all model fitting uses training only.
    with threadpool_limits(limits=2):
        for algorithm, grid in GRIDS.items():
            for strategy in ("global", "local"):
                best = None
                for calendar in (False, True):
                    for index, params in enumerate(grid):
                        candidate_id = f"{algorithm}_{strategy}_calendar{int(calendar)}_config{index + 1}"
                        models, probabilities = fit_candidate(train, validation, algorithm, params, calendar, strategy)
                        threshold = best_threshold(y, probabilities)
                        score = metrics(y, probabilities, threshold)
                        config = dict(candidate_id=candidate_id, algorithm=algorithm, strategy=strategy,
                                      calendar=calendar, params=params)
                        for domain in ["all"] + sorted(validation["domain"].unique()):
                            mask = np.ones(len(validation), dtype=bool) if domain == "all" else validation["domain"].eq(domain).to_numpy()
                            results.append({**config, "params": json.dumps(params, sort_keys=True),
                                            "domain": domain, **metrics(y[mask], probabilities[mask], threshold)})
                        print(f"{candidate_id}: AP={score['average_precision']:.4f}, F1={score['f1']:.4f}", flush=True)
                        # Deterministic AP tie-break: first candidate (no calendar first).
                        if best is None or score["average_precision"] > best["metrics"]["average_precision"]:
                            best = dict(**config, metrics=score, models=models, probabilities=probabilities)
                key = f"{algorithm}_{strategy}"
                bundle = {k: v for k, v in best.items() if k != "probabilities"}
                bundle["numeric_features"], bundle["categorical_features"] = predictor_columns(best["calendar"], strategy)
                joblib.dump(bundle, model_directory / f"{key}.joblib", compress=3)
                selected[key] = {k: v for k, v in bundle.items() if k != "models"}
                predictions[f"{key}_probability"] = best["probabilities"]
                predictions[f"{key}_alert"] = (best["probabilities"] >= best["metrics"]["threshold"]).astype(int)
    for strategy in ("global", "local"):
        candidates = [k for k, v in selected.items() if v["strategy"] == strategy]
        winners[strategy] = max(candidates, key=lambda k: selected[k]["metrics"]["average_precision"])
    manifest = dict(seed=SEED, selection_metric="average_precision", threshold_metric="validation_f1",
                    threshold_tie_break="highest", candidate_tie_break="first_in_grid",
                    local_selection="pooled validation AP, shared hyperparameters and threshold across domains",
                    refit_on_validation=False, versions=dict(python=platform.python_version(),
                    sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, joblib=joblib.__version__),
                    inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, validation_path)},
                    selected=selected, winners=winners)
    (model_directory / "selection.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report / "candidates.csv", list(results[0]), results)
    write_csv(report / "selected.csv", list(results[0]),
              [r for r in results if r["candidate_id"] in {v["candidate_id"] for v in selected.values()}])
    write_csv(report / "predictions.csv", list(predictions.columns), predictions.to_dict("records"))
    print("Selected strategies:", winners, flush=True)


if __name__ == "__main__":
    main()
