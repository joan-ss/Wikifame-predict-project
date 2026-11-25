# Etapa 4: entrenamiento y validación

## Ejecución

Desde la raíz del proyecto (`Wikifame-predict-project`), ejecutar:

```powershell
& 'C:/Users/joans/anaconda3/python.exe' -m src.train_validation
& 'C:/Users/joans/anaconda3/python.exe' -m src.summarize_validation
& 'C:/Users/joans/anaconda3/python.exe' -m unittest discover -s tests -v
```

Para otro entorno, instalar `requirements.txt` desde la raíz del proyecto y utilizar su intérprete de Python.

## Procedimiento

Si `data/processed/prediction_dataset.csv` ya existe, `train_validation.py` lo lee sin reconstruirlo ni sobrescribirlo. Solo lo construye a partir de `data/processed/pageviews_with_metadata.csv` cuando no existe. Si cambian los datos originales o las fórmulas de las variables, hay que renombrar o eliminar el CSV preparado para regenerarlo. El entrenamiento y la validación sí se ejecutan cada vez.

`train_validation.py` construye `data/processed/prediction_dataset.csv` a partir de las visitas con metadatos. Todas las ventanas se calculan por artículo, incluyen únicamente días observados y respetan las fórmulas de `resum.md`. La intensidad de picos necesita 37 días de histórico. El último día sin objetivo conocido se descarta. `target_day_of_week` usa lunes=0 y domingo=6.

Entrena con objetivos de 2024 y valida con objetivos de enero a junio de 2025. Las fechas de test no intervienen en entrenamiento, selección ni evaluación. Los datos anteriores a cada predicción permiten actualizar las características durante validación, manteniendo fijo el modelo.

Se prueban 10 configuraciones en cada ámbito: global, deporte y ciencia y tecnología. Random Forest combina `min_samples_leaf` de 1, 5 o 20 con `max_features` de 0.7 o 1.0, usando 200 árboles. HistGradientBoosting combina 15 o 31 hojas con tasas de aprendizaje de 0.05 o 0.1, usando 200 iteraciones y regularización L2 de 1. Se desactiva su parada temprana para evitar una partición interna aleatoria. No es una búsqueda exhaustiva.

La selección utiliza **MAE mínimo**, con RMSE para desempatar. Se conserva también el mejor candidato de cada algoritmo para comparar global y local usando el mismo algoritmo. Todos reciben las mismas variables; `page_id`, `qid`, título y fechas no son predictores. La codificación one-hot se ajusta exclusivamente con entrenamiento. No se transforman logarítmicamente ni el objetivo ni las visitas recientes.

El baseline predice `lag_1`. Una selección entre los dos algoritmos de ML no implica que el ganador supere ese baseline: debe comprobarse en los resultados.

## Archivos de salida

- `results/stage4/validation_scores.csv`: MAE y RMSE de todos los candidatos y baseline.
- `results/stage4/RESULTADOS.md`: resumen legible, generado por `summarize_validation.py`.
- `results/stage4/strategy_comparison.csv`: comparación global/local por algoritmo, total y por dominio.
- `results/stage4/selected_models.json`: configuración elegida por ámbito, variables y versiones.
- `results/stage4/validation_predictions.csv`: predicciones del mejor candidato de cada algoritmo por ámbito.
- `results/stage4/validation_breakdown.csv`: errores por dominio, artículo, subtema y tipo de entidad, junto al MAE del baseline.
- `models/stage4/*.joblib`: seis pipelines entrenados únicamente con 2024, incluyendo codificación y modelo. Cargar solo archivos de confianza.

Para comparar la estrategia local con la global en todos los artículos, concatenar las predicciones de los dos ámbitos locales para el mismo algoritmo. Comparar sobre las mismas filas; el RMSE conjunto se calcula desde los errores, no promediando los dos RMSE.

Las métricas de validación sirven para seleccionar configuraciones y son exploratorias. La evaluación final, el posible reentrenamiento con train+validation y `test_predictions.csv` corresponden a la etapa 5.
