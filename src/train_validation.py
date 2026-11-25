"""Build causal features and run stage 4. Never fit or evaluate on test data."""
from pathlib import Path
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "stage4"
MODELS_DIR = ROOT / "models" / "stage4"
NUMERIC = ([f"lag_{i}" for i in range(1, 8)] +
           ["mean_3", "mean_7", "mean_30", "trend_30", "volatility_iqr_30",
            "peak_intensity_7", "target_day_of_week"])
CATEGORICAL = ["domain", "topic", "topic_type"]
FEATURES = NUMERIC + CATEGORICAL


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
    frames = []
    days = np.arange(30, dtype=float)
    centered = days - days.mean()
    for page_id, article in raw.groupby("page_id", sort=True):
        article = article.sort_values("date").copy()
        if not article.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Non-consecutive dates for {page_id}")
        if any(article[c].nunique() != 1 for c in CATEGORICAL):
            raise ValueError(f"Changing metadata for {page_id}")
        views = article.views.astype(float)
        for i in range(1, 8):
            article[f"lag_{i}"] = views.shift(i - 1)
        for window in (3, 7, 30):
            article[f"mean_{window}"] = views.rolling(window).mean()
        log_views = np.log1p(views)
        article["trend_30"] = log_views.rolling(30).apply(
            lambda values: np.dot(centered, values) / np.dot(centered, centered), raw=True)
        article["volatility_iqr_30"] = (log_views.rolling(30).quantile(.75)
                                         - log_views.rolling(30).quantile(.25))
        article["peak_intensity_7"] = ((1 + views.rolling(7).max()) /
                                        (1 + views.shift(7).rolling(30).median()))
        article["target_date"] = article.date + pd.Timedelta(days=1)
        article["target_day_of_week"] = article.target_date.dt.dayofweek
        article["target_views"] = views.shift(-1)
        frames.append(article.dropna(subset=NUMERIC + ["target_views"]))
    result = pd.concat(frames, ignore_index=True)
    return result.drop(columns=["views"]).sort_values(["target_date", "page_id"])


def metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}


def candidates():
    # Small, predefined grid; all candidates use the same features and dates.
    for leaf in (1, 5, 20):
        for max_features in (0.7, 1.0):
            params = dict(n_estimators=200, min_samples_leaf=leaf,
                          max_features=max_features, random_state=42, n_jobs=2)
            yield "random_forest", params, RandomForestRegressor(**params)
    for leaves in (15, 31):
        for rate in (.05, .1):
            params = dict(max_iter=200, max_leaf_nodes=leaves, learning_rate=rate,
                          l2_regularization=1.0, early_stopping=False, random_state=42)
            yield "hist_gradient_boosting", params, HistGradientBoostingRegressor(**params)


def main():
    output = RESULTS_DIR
    output.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_DIR / "prediction_dataset.csv"
    if dataset_path.exists():
        print(f"Loading existing dataset: {dataset_path.name}", flush=True)
        dataset = pd.read_csv(dataset_path, parse_dates=["date", "target_date"])
    else:
        print(f"Building dataset: {dataset_path.name}", flush=True)
        dataset = build_dataset(pd.read_csv(DATA_DIR / "pageviews_with_metadata.csv"))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(dataset_path, index=False)
    train = dataset[dataset.target_date.between("2024-01-01", "2024-12-31")]
    validation = dataset[dataset.target_date.between("2025-01-01", "2025-06-30")]
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation split")
    rows, selection, predictions, breakdown = [], {}, [], []
    scopes = ["global"] + sorted(train.domain.unique())
    for scope in scopes:
        tr = train if scope == "global" else train[train.domain == scope]
        va = validation if scope == "global" else validation[validation.domain == scope]
        baseline = metrics(va.target_views, va.lag_1)
        rows.append(dict(scope=scope, algorithm="last_day", parameters="{}", **baseline))
        best_by_algorithm = {}
        for algorithm, params, estimator in candidates():
            preprocessing = ColumnTransformer([
                ("numeric", "passthrough", NUMERIC),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
            model = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
            model.fit(tr[FEATURES], tr.target_views)
            predicted = model.predict(va[FEATURES])
            scores = metrics(va.target_views, predicted)
            rows.append(dict(scope=scope, algorithm=algorithm,
                             parameters=json.dumps(params, sort_keys=True), **scores))
            print(scope, algorithm, params, scores, flush=True)
            best = best_by_algorithm.get(algorithm)
            if best is None or (scores["mae"], scores["rmse"]) < (best[0]["mae"], best[0]["rmse"]):
                best_by_algorithm[algorithm] = (scores, params, model, predicted)
        winner = min(best_by_algorithm, key=lambda a: (
            best_by_algorithm[a][0]["mae"], best_by_algorithm[a][0]["rmse"]))
        selection[scope] = dict(algorithm=winner, parameters=best_by_algorithm[winner][1],
                               **best_by_algorithm[winner][0], baseline=baseline)
        for algorithm, (scores, params, model, predicted) in best_by_algorithm.items():
            joblib.dump(model, MODELS_DIR / f"{scope}_{algorithm}.joblib", compress=3)
            pred = va[["page_id", "target_date", *CATEGORICAL, "target_views", "lag_1"]].copy()
            pred["scope"], pred["algorithm"], pred["prediction"] = scope, algorithm, predicted
            predictions.append(pred)
            for column in ("domain", "page_id", "topic", "topic_type"):
                for value, group in pred.groupby(column):
                    breakdown.append(dict(scope=scope, algorithm=algorithm, group_by=column,
                                          group=str(value), n=len(group),
                                          **metrics(group.target_views, group.prediction),
                                          baseline_mae=metrics(group.target_views, group.lag_1)["mae"]))
    pd.DataFrame(rows).to_csv(output / "validation_scores.csv", index=False)
    pd.concat(predictions).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(breakdown).to_csv(output / "validation_breakdown.csv", index=False)
    manifest = dict(selection_metric="MAE; RMSE breaks ties", seed=42,
                    train_rows=len(train), validation_rows=len(validation), features=FEATURES,
                    versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                  pandas=pd.__version__, numpy=np.__version__), selected=selection)
    (output / "selected_models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
