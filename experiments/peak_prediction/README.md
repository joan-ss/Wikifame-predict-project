# Predicción de picos de visitas a siete días

Para cada artículo y fecha, estimamos si alguno de los próximos siete días tendrá un pico de visitas.

## Datos y objetivo

- [pageviews.csv](data/pageviews.csv) es una copia de las visitas diarias originales.
- [prediction_dataset.csv](data/prediction_dataset.csv) es una copia del dataset de regresión. Reutilizamos sus variables temporales y sus metadatos.
- [peak_dataset.csv](data/peak_dataset.csv) es el dataset de clasificación. Cada fila contiene la fecha de emisión, la ventana futura, las variables conocidas al emitir la predicción y la etiqueta `peak_next_7`.

La etiqueta vale 1 si al menos un día de la ventana futura alcanza tres veces la mediana de visitas de los últimos 30 días, incluida la fecha de emisión. La mediana tiene un mínimo de una visita para calcular el umbral en series con ceros. Usamos 2023 como historial, 2024 para entrenar y enero a junio de 2025 para validar. Cada ventana objetivo queda completa dentro de su periodo. Julio a diciembre de 2025 queda reservado para el test.

## Entrenamiento y comparación

La regla de referencia da una alerta si, en los últimos siete días observados, hubo un día con al menos tres veces la mediana de los 30 días que preceden a esos siete. Entrenamos Random Forest e HistGradientBoosting sobre todos los artículos y por separado para Deporte y Ciencia y tecnología.

Seleccionamos configuraciones por precisión media (AP) y elegimos en validación el umbral de alerta que maximiza F1 para cada modelo. Informamos también de precisión y recall. Un mismo pico puede contar en varias ventanas semanales solapadas.

## Ejecución

Desde esta carpeta, con Python 3.14, ejecuta:

```bash
python -m pip install -r requirements.txt
python build_dataset.py
python train_validation.py
python -m unittest -v
```

Las dependencias de esta carpeta corresponden al entorno que generó los modelos guardados. El entrenamiento produce los seis pipelines en `models/`, las configuraciones seleccionadas, las métricas y las predicciones en `results/`.

- [Resumen de validación](results/RESULTADOS.md)
- [Métricas conjuntas y por dominio](results/strategy_comparison.csv)
- [Configuraciones seleccionadas](results/selected_models.json)
