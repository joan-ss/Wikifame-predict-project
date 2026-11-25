"""Build features from past visits, then train and validate the stage 4 models."""

import json
import platform
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "stage4"
MODELS_DIR = ROOT / "models" / "stage4"
LAG_COLUMNS = [f"lag_{i}" for i in range(1, 8)]
NUMERIC = [
    *LAG_COLUMNS,
    "mean_3",
    "mean_7",
    "mean_30",
    "trend_30",
    "volatility_iqr_30",
    "peak_intensity_7",
    "target_day_of_week",
]
CATEGORICAL = ["domain", "topic", "topic_type"]
FEATURES = NUMERIC + CATEGORICAL


@dataclass
class ValidationRun:
    parameters: dict
    model: Pipeline
    predictions: np.ndarray
    scores: dict


def build_dataset(raw):
    required = {"page_id", "date", "views", *CATEGORICAL}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {required - set(raw.columns)}")

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw[list(required)].isna().any().any() or raw.duplicated(["page_id", "date"]).any():
        raise ValueError("Missing values or duplicate article dates")
    if not np.isfinite(raw.views).all() or (raw.views < 0).any():
        raise ValueError("Invalid visit counts")

    one_day = pd.Timedelta(days=1)
    days = np.arange(30, dtype=float)
    centered_days = days - days.mean()
    trend_denominator = np.dot(centered_days, centered_days)
    article_datasets = []

    for page_id, article in raw.groupby("page_id", sort=True):
        article = article.sort_values("date").copy()
        if not article.date.diff().dropna().eq(one_day).all():
            raise ValueError(f"Non-consecutive dates for {page_id}")
        if any(article[column].nunique() != 1 for column in CATEGORICAL):
            raise ValueError(f"Changing metadata for {page_id}")

        views = article.views.astype(float)
        # lag_1 is the last observed day, immediately before the target.
        for days_back, column in enumerate(LAG_COLUMNS):
            article[column] = views.shift(days_back)
        for window in (3, 7, 30):
            article[f"mean_{window}"] = views.rolling(window).mean()

        log_window = np.log1p(views).rolling(30)
        article["trend_30"] = log_window.apply(
            lambda values: np.dot(centered_days, values) / trend_denominator,
            raw=True,
        )
        article["volatility_iqr_30"] = log_window.quantile(0.75) - log_window.quantile(0.25)

        recent_peak = views.rolling(7).max()
        # Use the 30 days before the recent window as the reference level.
        reference_median = views.shift(7).rolling(30).median()
        article["peak_intensity_7"] = (1 + recent_peak) / (1 + reference_median)

        article["target_date"] = article.date + one_day
        article["target_day_of_week"] = article.target_date.dt.dayofweek
        article["target_views"] = views.shift(-1)
        article_datasets.append(article.dropna(subset=[*NUMERIC, "target_views"]))

    dataset = pd.concat(article_datasets, ignore_index=True)
    return dataset.drop(columns=["views"]).sort_values(["target_date", "page_id"])


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def candidates():
    for leaf, max_features in product((1, 5, 20), (0.7, 1.0)):
        parameters = dict(
            n_estimators=200, min_samples_leaf=leaf, max_features=max_features,
            random_state=42, n_jobs=2,
        )
        yield "random_forest", parameters, RandomForestRegressor(**parameters)

    for leaves, rate in product((15, 31), (0.05, 0.1)):
        parameters = dict(
            max_iter=200, max_leaf_nodes=leaves, learning_rate=rate,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        )
        yield "hist_gradient_boosting", parameters, HistGradientBoostingRegressor(**parameters)


def load_dataset():
    dataset_path = DATA_DIR / "prediction_dataset.csv"
    if dataset_path.exists():
        print(f"Loading existing dataset: {dataset_path.name}", flush=True)
        return pd.read_csv(dataset_path, parse_dates=["date", "target_date"])

    print(f"Building dataset: {dataset_path.name}", flush=True)
    raw = pd.read_csv(DATA_DIR / "pageviews_with_metadata.csv")
    dataset = build_dataset(raw)
    dataset.to_csv(dataset_path, index=False)
    return dataset


def selection_key(scores):
    return scores["mae"], scores["rmse"]


def fit_candidates(scope, train, validation):
    score_rows = []
    best_runs = {}
    train_features = train[FEATURES]
    validation_features = validation[FEATURES]

    for algorithm, parameters, estimator in candidates():
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        preprocessing = ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("categorical", encoder, CATEGORICAL),
        ])
        model = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        model.fit(train_features, train.target_views)
        predictions = model.predict(validation_features)
        scores = metrics(validation.target_views, predictions)

        score_rows.append(dict(
            scope=scope, algorithm=algorithm,
            parameters=json.dumps(parameters, sort_keys=True), **scores,
        ))
        print(scope, algorithm, parameters, scores, flush=True)

        current_best = best_runs.get(algorithm)
        if current_best is None or selection_key(scores) < selection_key(current_best.scores):
            best_runs[algorithm] = ValidationRun(parameters, model, predictions, scores)

    return score_rows, best_runs


def save_best_models(scope, validation, best_runs):
    predictions = []
    breakdown = []
    prediction_columns = ["page_id", "target_date", *CATEGORICAL, "target_views", "lag_1"]

    for algorithm, run in best_runs.items():
        joblib.dump(run.model, MODELS_DIR / f"{scope}_{algorithm}.joblib", compress=3)
        model_predictions = validation[prediction_columns].assign(
            scope=scope, algorithm=algorithm, prediction=run.predictions,
        )
        predictions.append(model_predictions)

        for column in ("domain", "page_id", "topic", "topic_type"):
            for value, group in model_predictions.groupby(column):
                breakdown.append(dict(
                    scope=scope, algorithm=algorithm, group_by=column, group=str(value),
                    n=len(group), **metrics(group.target_views, group.prediction),
                    baseline_mae=metrics(group.target_views, group.lag_1)["mae"],
                ))

    return predictions, breakdown


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    train = dataset[dataset.target_date.between("2024-01-01", "2024-12-31")]
    validation = dataset[dataset.target_date.between("2025-01-01", "2025-06-30")]
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation split")

    score_rows = []
    selected_models = {}
    predictions = []
    breakdown = []
    scopes = ["global", *sorted(train.domain.unique())]

    for scope in scopes:
        scope_train = train if scope == "global" else train[train.domain == scope]
        scope_validation = validation if scope == "global" else validation[validation.domain == scope]
        baseline = metrics(scope_validation.target_views, scope_validation.lag_1)
        score_rows.append(dict(scope=scope, algorithm="last_day", parameters="{}", **baseline))

        scores, best_runs = fit_candidates(scope, scope_train, scope_validation)
        score_rows.extend(scores)
        winner = min(best_runs, key=lambda algorithm: selection_key(best_runs[algorithm].scores))
        selected_run = best_runs[winner]
        selected_models[scope] = dict(
            algorithm=winner, parameters=selected_run.parameters,
            **selected_run.scores, baseline=baseline,
        )

        scope_predictions, scope_breakdown = save_best_models(scope, scope_validation, best_runs)
        predictions.extend(scope_predictions)
        breakdown.extend(scope_breakdown)

    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "validation_scores.csv", index=False)
    pd.concat(predictions).to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame(breakdown).to_csv(RESULTS_DIR / "validation_breakdown.csv", index=False)

    manifest = dict(
        selection_metric="MAE; RMSE breaks ties",
        seed=42,
        train_rows=len(train),
        validation_rows=len(validation),
        features=FEATURES,
        versions=dict(
            python=platform.python_version(), sklearn=sklearn.__version__,
            pandas=pd.__version__, numpy=np.__version__,
        ),
        selected=selected_models,
    )
    manifest_json = json.dumps(manifest, indent=2)
    (RESULTS_DIR / "selected_models.json").write_text(manifest_json, encoding="utf-8")
    print(manifest_json, flush=True)


if __name__ == "__main__":
    main()
