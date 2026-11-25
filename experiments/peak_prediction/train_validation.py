"""Train and compare global and domain-specific seven-day peak classifiers."""

import json
import platform
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from build_dataset import CATEGORICAL, DATASET_PATH, FEATURES, HORIZON, KEYS, NUMERIC, PEAK_MULTIPLIER, ROOT, SPLITS

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET = "peak_next_7"
SEED = 42
BASELINE = "recent_peak"


@dataclass
class ValidationResult:
    parameters: dict
    model: Pipeline
    scores: np.ndarray
    threshold: float
    metrics: dict


def candidates():
    for leaf_size in (5, 20):
        parameters = dict(
            n_estimators=200, min_samples_leaf=leaf_size, max_features=0.7,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=2,
        )
        yield "random_forest", parameters, RandomForestClassifier(**parameters)

    for leaves in (15, 31):
        parameters = dict(
            max_iter=200, max_leaf_nodes=leaves, learning_rate=0.05, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=SEED,
        )
        yield "hist_gradient_boosting", parameters, HistGradientBoostingClassifier(**parameters)


def alert_threshold(actual, scores):
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(actual, scores, alerts):
    return dict(
        average_precision=float(average_precision_score(actual, scores)),
        precision=float(precision_score(actual, alerts, zero_division=0)),
        recall=float(recall_score(actual, alerts, zero_division=0)),
        f1=float(f1_score(actual, alerts, zero_division=0)),
    )


def fit_candidates(scope, train, validation):
    if set(train[TARGET]) != {0, 1} or set(validation[TARGET]) != {0, 1}:
        raise ValueError(f"Both classes are required in train and validation for {scope}")

    runs = []
    best = {}
    actual = validation[TARGET]
    for algorithm, parameters, estimator in candidates():
        preprocessing = ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        model = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        model.fit(train[FEATURES], train[TARGET])
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        threshold = alert_threshold(actual, scores)
        metrics = evaluate(actual, scores, scores >= threshold)
        runs.append(dict(scope=scope, algorithm=algorithm, threshold=threshold, **metrics,
                         parameters=json.dumps(parameters, sort_keys=True)))
        print(scope, algorithm, parameters, metrics, flush=True)

        previous = best.get(algorithm)
        if previous is None or metrics["average_precision"] > previous.metrics["average_precision"]:
            best[algorithm] = ValidationResult(parameters, model, scores, threshold, metrics)

    return runs, best


def prediction_rows(validation, scope, algorithm, scores, threshold):
    columns = [*KEYS, "window_start", "window_end", *CATEGORICAL, TARGET]
    return validation[columns].assign(
        scope=scope, algorithm=algorithm, score=scores, threshold=threshold,
        alert=(scores >= threshold).astype(int),
    )


def compare_strategies(predictions):
    baseline = predictions[predictions.algorithm.eq(BASELINE)]
    global_model = predictions[predictions.scope.eq("global") & predictions.algorithm.ne(BASELINE)]
    local_models = predictions[predictions.scope.ne("global")]
    expected = baseline.set_index(KEYS)[TARGET].sort_index()
    rows = []
    for strategy, forecasts in (("baseline", baseline), ("global", global_model), ("local", local_models)):
        for algorithm, group in forecasts.groupby("algorithm"):
            pd.testing.assert_series_equal(expected, group.set_index(KEYS)[TARGET].sort_index())
            for domain, subset in [("all", group), *group.groupby("domain")]:
                rows.append(dict(
                    strategy=strategy, algorithm=algorithm, domain=domain,
                    windows=len(subset), peaks=int(subset[TARGET].sum()), alerts=int(subset.alert.sum()),
                    **evaluate(subset[TARGET], subset.score, subset.alert),
                ))
    return pd.DataFrame(rows)


def write_report(comparison):
    lines = [
        "# Predicción de picos de visitas a siete días",
        "",
        "Los modelos se entrenaron con 2024 y se seleccionaron usando enero a junio de 2025.",
        "",
        "| Estrategia | Algoritmo | Precisión media (AP) | Precisión | Recall | F1 | Alertas |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison[comparison.domain.eq("all")].itertuples():
        lines.append(
            f"| {row.strategy} | {row.algorithm} | {row.average_precision:.3f} | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.alerts} |"
        )
    lines.extend([
        "",
        "La precisión mide la proporción de alertas acertadas y el recall la proporción de ventanas "
        "con pico que reciben una alerta. AP resume el rendimiento para distintos umbrales.",
        "",
        "Los umbrales de los modelos se eligen por F1 en validación. La regla de referencia emite una "
        "alerta cuando detecta un pico reciente con el factor fijo de tres.",
        "",
        "Las ventanas se solapan. Un mismo pico puede aparecer en varias ventanas.",
        "",
        "Julio a diciembre de 2025 queda reservado para la evaluación final.",
        "",
    ])
    (RESULTS_DIR / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    dataset = pd.read_csv(DATASET_PATH)
    train = dataset[dataset.split.eq("train")]
    validation = dataset[dataset.split.eq("validation")]
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    predictions = [
        prediction_rows(validation, "global", BASELINE, validation.recent_peak_score, PEAK_MULTIPLIER),
    ]
    score_rows = []
    selected_models = {}
    for scope in ["global", *sorted(train.domain.unique())]:
        scope_train = train if scope == "global" else train[train.domain.eq(scope)]
        scope_validation = validation if scope == "global" else validation[validation.domain.eq(scope)]
        runs, best = fit_candidates(scope, scope_train, scope_validation)
        score_rows.extend(runs)

        winner = max(best, key=lambda algorithm: best[algorithm].metrics["average_precision"])
        selected_models[scope] = dict(winner=winner, models={})
        for algorithm, result in best.items():
            filename = f"{scope}_{algorithm}.joblib"
            joblib.dump(result.model, MODELS_DIR / filename, compress=3)
            predictions.append(prediction_rows(
                scope_validation, scope, algorithm, result.scores, result.threshold,
            ))
            selected_models[scope]["models"][algorithm] = dict(
                file=filename, parameters=result.parameters, threshold=result.threshold, **result.metrics,
            )

    predictions = pd.concat(predictions, ignore_index=True)
    comparison = compare_strategies(predictions)
    predictions.to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "validation_scores.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "strategy_comparison.csv", index=False)
    manifest = dict(
        horizon_days=HORIZON, peak_multiplier=PEAK_MULTIPLIER, splits=SPLITS,
        train_windows=len(train), validation_windows=len(validation), features=FEATURES,
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                      pandas=pd.__version__, numpy=np.__version__),
        selected=selected_models,
    )
    (RESULTS_DIR / "selected_models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(comparison)
    print(comparison[comparison.domain.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
