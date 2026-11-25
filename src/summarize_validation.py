"""Summarize stage 4 using validation predictions only."""

import json

import pandas as pd

from src.train_validation import RESULTS_DIR, metrics


def check_matching_targets(global_rows, local_rows):
    keys = ["page_id", "target_date"]
    columns = [*keys, "target_views", "lag_1"]
    for rows in (global_rows, local_rows):
        if rows.duplicated(keys).any():
            raise ValueError("Duplicate validation predictions")

    pd.testing.assert_frame_equal(
        global_rows[columns].sort_values(keys).reset_index(drop=True),
        local_rows[columns].sort_values(keys).reset_index(drop=True),
    )


def compare_strategies(predictions):
    comparisons = []
    for algorithm in sorted(predictions.algorithm.unique()):
        algorithm_rows = predictions[predictions.algorithm == algorithm]
        global_rows = algorithm_rows[algorithm_rows.scope == "global"]
        local_rows = algorithm_rows[algorithm_rows.scope != "global"]
        check_matching_targets(global_rows, local_rows)

        for strategy, rows in (("global", global_rows), ("local", local_rows)):
            for domain in ["all", *sorted(rows.domain.unique())]:
                group = rows if domain == "all" else rows[rows.domain == domain]
                baseline = metrics(group.target_views, group.lag_1)
                scores = metrics(group.target_views, group.prediction)
                improvement = 100 * (1 - scores["mae"] / baseline["mae"])
                comparisons.append(dict(
                    algorithm=algorithm, strategy=strategy, domain=domain, n=len(group), **scores,
                    baseline_mae=baseline["mae"], baseline_rmse=baseline["rmse"],
                    mae_improvement_percent=improvement,
                ))

    return pd.DataFrame(comparisons)


def format_report(overall, selected_models):
    lines = [
        "# Resultados de validación · Etapa 4", "",
        "Entrenamiento: 2024. Validación: enero–junio de 2025. Test sin evaluar.", "",
        "Selección por MAE mínimo; RMSE para desempatar. Valores en visitas.", "",
        "| Ámbito | Algoritmo elegido | MAE | RMSE | MAE baseline |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for scope, result in selected_models.items():
        lines.append(
            f"| {scope} | {result['algorithm']} | {result['mae']:.2f} | "
            f"{result['rmse']:.2f} | {result['baseline']['mae']:.2f} |"
        )

    lines += [
        "", "## Comparación con el mismo algoritmo", "",
        "Los modelos locales se evalúan conjuntamente sobre los mismos 7.240 objetivos que el global.", "",
        "| Algoritmo | Estrategia | MAE | RMSE | Mejora MAE frente al baseline |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in overall.itertuples():
        lines.append(
            f"| {row.algorithm} | {row.strategy} | {row.mae:.2f} | "
            f"{row.rmse:.2f} | {row.mae_improvement_percent:.2f}% |"
        )

    lines += [
        "",
        "Una mejora positiva indica menor MAE que el baseline. Estos resultados se han "
        "utilizado para seleccionar configuraciones; no sustituyen la evaluación final en test.",
        "",
        "La muestra fue elegida manualmente según HISTORICO.md. Las conclusiones corresponden "
        "a estos artículos y no garantizan generalización a toda Wikipedia.",
        "",
    ]
    return "\n".join(lines)


def main():
    predictions = pd.read_csv(RESULTS_DIR / "validation_predictions.csv")
    manifest = json.loads((RESULTS_DIR / "selected_models.json").read_text(encoding="utf-8"))

    comparison = compare_strategies(predictions)
    comparison.to_csv(RESULTS_DIR / "strategy_comparison.csv", index=False)

    overall = comparison[comparison.domain == "all"]
    report = format_report(overall, manifest["selected"])
    (RESULTS_DIR / "RESULTADOS.md").write_text(report, encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
