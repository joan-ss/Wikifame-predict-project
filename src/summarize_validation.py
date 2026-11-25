"""Summarize stage 4 using validation predictions only."""
import json
import pandas as pd
from src.train_validation import RESULTS_DIR, metrics


def main():
    output = RESULTS_DIR
    predictions = pd.read_csv(output / "validation_predictions.csv")
    manifest = json.loads((output / "selected_models.json").read_text(encoding="utf-8"))
    rows = []
    for algorithm in sorted(predictions.algorithm.unique()):
        global_rows = predictions[(predictions.algorithm == algorithm) & (predictions.scope == "global")]
        local_rows = predictions[(predictions.algorithm == algorithm) & (predictions.scope != "global")]
        keys = ["page_id", "target_date"]
        for subset in (global_rows, local_rows):
            if subset.duplicated(keys).any():
                raise ValueError("Duplicate validation predictions")
        pd.testing.assert_frame_equal(
            global_rows[keys + ["target_views", "lag_1"]].sort_values(keys).reset_index(drop=True),
            local_rows[keys + ["target_views", "lag_1"]].sort_values(keys).reset_index(drop=True))
        for strategy in ("global", "local"):
            subset = predictions[(predictions.algorithm == algorithm) &
                                 ((predictions.scope == "global") if strategy == "global"
                                  else (predictions.scope != "global"))]
            for domain in ["all"] + sorted(subset.domain.unique()):
                group = subset if domain == "all" else subset[subset.domain == domain]
                baseline = metrics(group.target_views, group.lag_1)
                scores = metrics(group.target_views, group.prediction)
                rows.append(dict(algorithm=algorithm, strategy=strategy, domain=domain,
                                 n=len(group), **scores, baseline_mae=baseline["mae"],
                                 baseline_rmse=baseline["rmse"],
                                 mae_improvement_percent=100 * (1 - scores["mae"] / baseline["mae"])))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "strategy_comparison.csv", index=False)
    lines = ["# Resultados de validación · Etapa 4", "",
             "Entrenamiento: 2024. Validación: enero–junio de 2025. Test sin evaluar.", "",
             "Selección por MAE mínimo; RMSE para desempatar. Valores en visitas.", "",
             "| Ámbito | Algoritmo elegido | MAE | RMSE | MAE baseline |",
             "| --- | --- | ---: | ---: | ---: |"]
    for scope, result in manifest["selected"].items():
        lines.append(f"| {scope} | {result['algorithm']} | {result['mae']:.2f} | "
                     f"{result['rmse']:.2f} | {result['baseline']['mae']:.2f} |")
    lines += ["", "## Comparación con el mismo algoritmo", "",
              "Los modelos locales se evalúan conjuntamente sobre los mismos 7.240 objetivos que el global.", "",
              "| Algoritmo | Estrategia | MAE | RMSE | Mejora MAE frente al baseline |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in comparison[comparison.domain == "all"].itertuples():
        lines.append(f"| {row.algorithm} | {row.strategy} | {row.mae:.2f} | "
                     f"{row.rmse:.2f} | {row.mae_improvement_percent:.2f}% |")
    lines += ["", "Una mejora positiva indica menor MAE que el baseline. Estos resultados se han utilizado para seleccionar configuraciones; no sustituyen la evaluación final en test.",
              "", "La muestra fue elegida manualmente según HISTORICO.md. Las conclusiones corresponden a estos artículos y no garantizan generalización a toda Wikipedia.", ""]
    (output / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(comparison[comparison.domain == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
