# Progreso del proyecto

Este registro se actualiza a medida que avanzan los pasos de implementación del [plan](PLAN.md).

## Paso 1. Revisión de los datos del piloto

**Completado.** Revisé los dos CSV originales con una comprobación puntual en Python.

- [articles.csv](data/raw/articles.csv) contiene 40 artículos, 20 de deporte y 20 de ciencia y tecnología. Comprobé que los identificadores de página y de Wikidata son únicos y que todos los artículos tienen metadatos completos.
- [pageviews.csv](data/raw/pageviews.csv) contiene 43.840 registros, con 1.096 días por artículo del 1 de enero de 2023 al 31 de diciembre de 2025. Las series están ordenadas por fecha, todos los recuentos de visitas son enteros positivos y hay cero duplicados, días ausentes y fechas fuera del periodo. Ambas tablas contienen el mismo conjunto de artículos.

## Paso 2. Referencia histórica diaria

**Completado.** Implementé [build_peak_reference.py](scripts/build_peak_reference.py) y generé [peak_reference.csv](data/processed/peak_reference.csv). El CSV contiene una fila por artículo y fecha con las columnas `page_id`, `date`, `views`, `median_30` e `iqr_30`.

Para cada fecha, calculé la mediana y el IQR sobre las visitas originales de los 30 días naturales anteriores. Por ejemplo, para el 31 de enero se usan las visitas del 1 al 30 de enero. El IQR es la diferencia entre los percentiles 75 y 25, calculados mediante interpolación lineal. Estas dos cifras servirán para calcular el umbral diario de cada artículo en el paso 3. El horizonte de predicción abarca los siete días siguientes a la fecha de emisión.

Verifiqué las 43.840 filas. Hay 42.640 referencias completas y 1.200 con ambos campos vacíos, correspondientes a los primeros 30 días de cada artículo. También comprobé con una serie de prueba que ambos campos quedan vacíos cuando falta algún día en la ventana.

Este comando regenera la referencia desde la raíz del proyecto.

```bash
python scripts/build_peak_reference.py
```

## Paso 3. Días altos e inicios de episodios

**Completado.** Para el piloto queda fijado el umbral diario del plan, `max(2 × mediana, mediana + 3 × IQR)`. Un día es alto si sus visitas superan estrictamente su umbral. Agrupé los días altos consecutivos de cada artículo sin cortes por semana, mes o año. En la tabla diaria, `peak_start` vale `1` si el día es alto y el día natural anterior tiene un estado conocido y no fue alto. Vale `0` en los días normales y en las continuaciones. Queda vacío cuando falta la referencia del día o cuando el día es alto y se desconoce el estado del día anterior.

- [peak_days.csv](data/processed/peak_days.csv) contiene 43.840 filas con las visitas, el umbral, `is_high` y `peak_start`. Hay 1.691 días altos y 1.201 valores desconocidos de `peak_start`. Estos corresponden a los primeros 30 días por artículo y al 31 de enero de 2023 de *Heysel Stadium disaster*, que fue alto con el día anterior aún sin umbral.
- [peak_episodes.csv](data/processed/peak_episodes.csv) contiene 801 episodios con inicio confirmado y el último día alto observado. Por año de inicio hay 265 en 2023, 266 en 2024 y 270 en 2025.

### Revisión de 2024

Revisé seis artículos de ambos dominios con distintos niveles habituales de tráfico. Las gráficas muestran las visitas, el umbral diario y los inicios detectados. [Gráficas de 2024](reports/peak_review_2024/series.png)

- En *24 Hours of Le Mans*, el episodio del 12 al 17 de junio empieza con 10.358 visitas frente a un umbral de 8.842. En *Solar eclipse*, la subida principal de abril forma un episodio del 4 al 9.
- En *Mars 2020*, el artículo de menor tráfico habitual en 2024, la mediana diaria fue de 198 visitas y la menor observación fue de 128. El episodio del 26 al 31 de enero empezó con 1.498 visitas frente a un umbral de 404. En *Solar eclipse* también se detectó un caso ajustado el 28 de julio, con 5.307 visitas frente a un umbral de 5.302,25. La revisión respalda conservar los multiplicadores 2 y 3 sin añadir un mínimo absoluto para este piloto.
- De los 266 episodios de 2024, 180 corresponden a deporte y 86 a ciencia y tecnología. Los 40 artículos tuvieron al menos un episodio. Los cinco con más episodios sumaron 69, el 26 % del total. [Recuento por artículo](reports/peak_review_2024/episodes_by_article.csv)

Estos comandos regeneran los CSV y las gráficas. El segundo requiere Matplotlib.

```bash
python scripts/build_peak_episodes.py
python scripts/review_peak_episodes.py
```

## Paso 4. Etiqueta de inicio en los próximos siete días

**Completado.** Implementé [build_peak_labels.py](scripts/build_peak_labels.py) y generé [peak_labels.csv](data/processed/peak_labels.csv) a partir de `peak_days.csv`. Esta tabla intermedia contiene `page_id`, `date`, `target_start`, `target_end` y `peak_next_7d`. El dataset con predictores se construirá en el paso 5.

Para cada fecha de predicción `t`, la etiqueta vale `1` si empieza al menos un episodio entre `t+1` y `t+7`, ambos incluidos, y `0` si los siete días tienen un inicio conocido y ninguno es positivo. El día `t` queda fuera del horizonte. Se usan días naturales del mismo artículo, sin cortar las ventanas al cambiar de año.

Exigimos conocer los siete indicadores futuros: si falta alguna fecha o algún `peak_start`, la etiqueta queda vacía, incluso si otro día de la ventana contiene un inicio confirmado. Así mantenemos únicamente horizontes completos para el futuro entrenamiento y evitamos confundir datos ausentes con negativos. No se eliminan filas ni se asignan todavía a entrenamiento, validación o test.

- Filas totales: **43.840**.
- Etiquetas positivas: **5.271**.
- Etiquetas negativas: **37.128**.
- Etiquetas desconocidas: **1.441**, de las cuales 1.161 corresponden al inicio de 2023 sin referencia suficiente y 280 a los últimos siete días de 2025 de los 40 artículos.

Las cuatro pruebas de [test_peak_labels.py](tests/test_peak_labels.py) comprueban los límites del horizonte, los cambios de año, las fechas ausentes, los estados desconocidos, el final de las series, la separación entre artículos y el rechazo de duplicados o indicadores inválidos. Todas pasaron. Además, contrasté las 42.399 etiquetas conocidas con los inicios de `peak_episodes.csv`, y verifiqué la unicidad de las filas y las fechas de sus horizontes.

Estos comandos regeneran las etiquetas y ejecutan las pruebas desde la raíz del proyecto:

```bash
python scripts/build_peak_labels.py
python -m unittest discover -s tests -v
```

## Paso 5. Dataset con predictores históricos, calendario y metadatos

**Completado.** Implementé [build_peak_prediction_dataset.py](scripts/build_peak_prediction_dataset.py) y generé [peak_prediction_dataset.csv](data/processed/peak_prediction_dataset.csv), uniendo las etiquetas del paso 4 con las visitas originales y los metadatos.

El archivo contiene 43.840 filas y 25 columnas:

- Identificación, horizonte y etiqueta: `page_id`, `date`, `target_start`, `target_end`, `peak_next_7d`.
- Metadatos: `qid`, `title`, `domain`, `topic`, `topic_type`. Los tres primeros identificadores (`page_id`, `qid`, `title`) se reservan para identificar y unir artículos, no como predictores del modelo.
- Historial: `lag_1` a `lag_7`, donde `lag_1` es el día de predicción y `lag_7` seis días antes; `mean_3`, `mean_7` y `mean_30`, incluyendo el día de predicción.
- `trend_30`: pendiente por mínimos cuadrados sobre los últimos 30 valores de `ln(1 + visitas)`, ordenados cronológicamente.
- `volatility_iqr_30`: IQR de esos mismos valores logarítmicos, con percentiles de interpolación lineal.
- `peak_intensity_7`: `(1 + máximo entre t-6 y t) / (1 + mediana entre t-36 y t-7)`. Necesita 37 días consecutivos contando el día de predicción.
- Calendario: `forecast_day_of_week` utiliza lunes = 0 hasta domingo = 6; `target_week_of_year` es la semana ISO de `t+1`.

Cada variable usa días naturales del mismo artículo y únicamente visitas hasta `t`. Si falta alguna observación de su ventana, esa variable queda vacía. Se conservan todas las filas para auditar ausencias; en el paso 6 se seleccionarán las completas y se asignarán a los periodos temporales del plan. No se incorporan inicios futuros ni visitas futuras como predictores, y las categorías aún no se codifican.

Hay **42.120 filas con predictores y etiqueta completos**: 5.224 positivas y 36.896 negativas. Las 1.720 restantes no son utilizables tal como están: las primeras 36 fechas de cada artículo carecen de algún predictor (1.440 filas) y las últimas siete no tienen etiqueta completa (280 filas). Las 1.441 etiquetas desconocidas del paso 4 se solapan con esas ausencias. Estas cifras incluyen 2023; los tamaños definitivos de entrenamiento, validación y test se calcularán en el paso 6.

Pasaron las nueve pruebas de ambos pasos. Las nuevas pruebas comprueban las ventanas exactas, la pendiente y el IQR logarítmicos, el calendario al cambiar de año, las ausencias, la unión de metadatos y que modificar visitas futuras no altera los predictores. Además, verifiqué la unicidad y el total de filas y contrasté la pendiente y el IQR de las 42.120 filas completas con cálculos independientes. El CSV generado hereda los permisos de su directorio y tiene acceso para el usuario local.

```bash
python scripts/build_peak_prediction_dataset.py
python -m unittest discover -s tests -v
```

## Paso 6. Partición temporal y recuentos

**Completado.** Implementé [split_peak_dataset.py](scripts/split_peak_dataset.py). Selecciona filas con predictores, metadatos y etiqueta completos, y exige que todo el horizonte `t+1` a `t+7` esté dentro de un único periodo. Conserva las columnas y los valores originales, y ordena cada conjunto por fecha de predicción y artículo.

| Conjunto | Periodo objetivo | Fechas de predicción | Filas | Positivas | Negativas | Episodios | Episodios evaluables |
|---|---|---|---:|---:|---:|---:|---:|
| Entrenamiento | 2024 | 31/12/2023–24/12/2024 | 14.400 | 1.740 | 12.660 | 266 | 257 |
| Validación | Enero–junio de 2025 | 31/12/2024–23/06/2025 | 7.000 | 908 | 6.092 | 143 | 135 |
| Test | Julio–diciembre de 2025 | 30/06/2025–24/12/2025 | 7.120 | 821 | 6.299 | 127 | 117 |

Los tres conjuntos contienen los 40 artículos y alcanzan los tamaños máximos del plan. Un episodio se cuenta por su fecha de inicio dentro del periodo objetivo. Para considerarlo evaluable en las métricas de cobertura y antelación, las siete fechas de predicción entre `d-7` y `d-1` deben estar presentes para ese mismo artículo en el conjunto. Quedan fuera de esa evaluación 9, 8 y 10 episodios, respectivamente, por los bordes temporales.

Archivos generados:

- [peak_train.csv](data/processed/peak_train.csv), [peak_validation.csv](data/processed/peak_validation.csv) y [peak_test.csv](data/processed/peak_test.csv).
- [summary.csv](reports/peak_splits/summary.csv): recuentos globales y por dominio, fechas, filas positivas y negativas, y episodios totales, evaluables y excluidos.
- [peak_split_exclusions.csv](data/processed/peak_split_exclusions.csv): las 15.320 filas no asignadas, con un motivo de exclusión.

Los motivos se asignan de forma excluyente en este orden: `historical_only` para horizontes que terminan antes de 2024 (14.320 filas); `horizon_outside_period` para horizontes que cruzan un límite o salen de los periodos (1.000 filas); `incomplete_data` para filas dentro de un periodo que carezcan de variables, metadatos o etiqueta (ninguna en este piloto). Por esta prioridad, las filas incompletas de 2023 y del final de 2025 quedan contabilizadas en los dos primeros motivos.

Pasaron las 12 pruebas acumuladas, incluidas las nuevas comprobaciones de límites exactos, horizontes que cruzan periodos, filas incompletas, duplicados y elegibilidad de episodios. También comprobé que las 43.840 filas originales están contabilizadas una sola vez, que los valores se conservan y que no hay fechas objetivo compartidas entre conjuntos. Los CSV generados heredan los permisos del directorio.

```bash
python scripts/split_peak_dataset.py
python -m unittest discover -s tests -v
```

## Paso 7. Entrenamiento y selección en validación

**Completado.** Implementé [train_peak_models.py](scripts/train_peak_models.py). Entrené con las 14.400 filas de entrenamiento y comparé las configuraciones sobre las 7.000 filas de validación. El script solo lee `peak_train.csv` y `peak_validation.csv`; no utiliza el conjunto de test.

### Comparación realizada

Comparé 16 configuraciones: dos algoritmos, dos estrategias (global y local), dos opciones de variables (con y sin calendario) y dos configuraciones de hiperparámetros por algoritmo. Esto implica 24 clasificadores ajustados, porque cada configuración local entrena uno por dominio.

- Random Forest: 160 árboles, profundidad máxima 12, `max_features="sqrt"`, `class_weight="balanced_subsample"` y hojas de tamaño mínimo 5 o 20.
- HistGradientBoosting: 150 iteraciones, tasa de aprendizaje 0,08, y dos configuraciones: 15 hojas / mínimo 20 muestras / regularización L2 de 1; o 31 hojas / mínimo 40 muestras / regularización L2 de 3.
- Semilla 42. HistGradientBoosting utiliza `early_stopping=False` para evitar una partición interna aleatoria de la serie temporal.
- Las categorías se codifican con OneHotEncoder ajustado solo en entrenamiento y `handle_unknown="ignore"`. Los identificadores, fechas, horizontes y etiquetas quedan fuera de los predictores. En los modelos locales también se omite `domain`, que se usa para seleccionar el clasificador correspondiente.

Dentro de cada combinación de algoritmo y estrategia, seleccioné la configuración con mayor Average Precision (AP) conjunta de validación. Para la estrategia local se utiliza una misma configuración de hiperparámetros y calendario en ambos dominios, con clasificadores ajustados por separado. El umbral de alerta también es único para la estrategia local, elegido sobre las probabilidades reunidas de ambos dominios. Esto mantiene el mismo criterio de selección y alerta para las estrategias global y local.

El umbral maximiza F1 de validación usando la regla `probabilidad >= umbral`; en empates de F1 se prefiere el umbral mayor. Los empates de AP conservan el primer candidato de la rejilla, que prueba primero la opción sin calendario.

### Resultados de validación

| Algoritmo | Estrategia | Calendario | AP | Precisión | Recall | F1 | Umbral |
|---|---|---|---:|---:|---:|---:|---:|
| Random Forest | Global | No | 0,2782 | 0,2678 | 0,5011 | 0,3491 | 0,439011 |
| Random Forest | Local | No | 0,2555 | 0,2658 | 0,5000 | 0,3471 | 0,399665 |
| HistGradientBoosting | Global | No | 0,2459 | 0,2435 | 0,4956 | 0,3266 | 0,138418 |
| HistGradientBoosting | Local | Sí | 0,2517 | 0,2453 | 0,4934 | 0,3277 | 0,120819 |

Random Forest sin calendario queda seleccionado para ambas estrategias, con hojas de tamaño mínimo 5. En esta validación, el modelo global tiene mayor AP conjunta. Dentro de HistGradientBoosting, la estrategia local mejora ligeramente la global. La proporción de ventanas positivas de validación es del 12,97 %.

Los resultados varían por dominio. Para Random Forest, la AP global/local es 0,3369/0,3272 en deporte y 0,1212/0,1244 en ciencia y tecnología. Con el umbral global fijado, el recall es 0,6808 en deporte y 0,1259 en ciencia y tecnología. Por tanto, la ventaja conjunta del modelo global no implica una mejora en todos los dominios. Estos resultados han servido para seleccionar configuraciones: la comparación final debe hacerse en test, sin volver a ajustar los umbrales. Las salidas no se han calibrado como probabilidades.

### Archivos y verificación

- [candidates.csv](reports/peak_validation/candidates.csv): las 16 configuraciones con métricas conjuntas y por dominio (48 filas), alertas y matriz de confusión.
- [selected.csv](reports/peak_validation/selected.csv): el mejor candidato de cada algoritmo y estrategia, con su desglose por dominio.
- [predictions.csv](reports/peak_validation/predictions.csv): las 7.000 predicciones de validación de los cuatro candidatos seleccionados y sus alertas.
- [selection.json](models/peak/selection.json): configuraciones, variables, métricas, umbrales exactos, ganadores global/local, versiones y hashes SHA-256 de los dos conjuntos utilizados.
- Cuatro archivos `.joblib` en `models/peak/`, uno por combinación de algoritmo y estrategia. Cada archivo local incluye los dos clasificadores y sus codificadores. Se conservan los modelos ajustados solo con entrenamiento, sin reajustarlos con validación.
- [requirements-training.txt](requirements-training.txt): versiones de las dependencias utilizadas.

Pasaron las 15 pruebas acumuladas. Las nuevas pruebas contrastan el umbral con una búsqueda exhaustiva, los empates, las categorías desconocidas, la exclusión de identificadores y etiquetas de las entradas, la opción sin calendario y la asignación al clasificador local. Además, recargué los cuatro archivos de modelos y verifiqué que reproducen las probabilidades y alertas guardadas, que la selección maximiza AP dentro de cada algoritmo y estrategia y que los transformadores excluyen las columnas no predictoras. Los archivos tienen permisos de acceso para el usuario local.

```bash
python -m pip install -r requirements-training.txt
python scripts/train_peak_models.py
python -m unittest discover -s tests -v
```

En este equipo, ejecuté el entrenamiento con `C:\Users\joans\anaconda3\python.exe`, que ya tenía instaladas estas dependencias.

## Paso 8. Predicciones de test con configuración fijada

**Completado.** Implementé [evaluate_peak_test.py](scripts/evaluate_peak_test.py) y generé [test_predictions.csv](data/processed/test_predictions.csv): 7.120 filas, del 30 de junio al 24 de diciembre de 2025 como fechas de emisión, con etiquetas, probabilidades y alertas de los cuatro modelos seleccionados en validación.

No entrené de nuevo ni ajusté variables, hiperparámetros o umbrales con test. Cada modelo usa los predictores disponibles hasta el día de emisión, incluidos los valores reales de ese día. La inferencia se ejecuta de forma vectorizada sobre las filas diarias: equivale al avance diario con modelos fijos, sin retroalimentar predicciones ni incorporar visitas futuras. La regla de alerta sigue siendo `probabilidad >= umbral`.

El script comprueba los hashes de entrenamiento y validación contra el manifiesto de selección, verifica la correspondencia de los modelos y umbrales, y guarda los hashes del test, del manifiesto y de los cuatro modelos en [audit.json](reports/peak_test/audit.json). Verifiqué que los archivos de modelos no cambian durante la evaluación.

## Paso 9. Evaluación final y conclusiones del piloto

**Completado.** El test contiene 821 ventanas positivas (11,53 %), 127 episodios y 117 episodios evaluables. Los diez restantes no tienen sus siete fechas previas de predicción dentro del conjunto y quedan fuera de las métricas por episodio.

| Modelo | AP | Precisión | Recall por ventana | F1 | Episodios cubiertos | Cobertura | Alertas/semana |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest global | 0,2512 | 0,2475 | 0,4495 | 0,3192 | 79/117 | 67,52 % | 58,63 |
| Random Forest local | 0,2241 | 0,2219 | 0,4275 | 0,2921 | 74/117 | 63,25 % | 62,21 |
| HistGradientBoosting global | 0,2168 | 0,2137 | 0,4336 | 0,2863 | 85/117 | 72,65 % | 65,52 |
| HistGradientBoosting local | 0,2022 | 0,2021 | 0,4251 | 0,2739 | 70/117 | 59,83 % | 67,92 |

La mediana de antelación es de siete días para los cuatro modelos, calculada únicamente entre los episodios cubiertos y usando la primera alerta de `d-7` a `d-1`. No implica que todas las alertas sean útiles ni que el modelo localice el día exacto del inicio. El volumen semanal es el total de alertas dividido por los 178 días distintos de predicción / 7, sobre el conjunto de artículos del grupo; las alertas repetidas se cuentan por separado.

### Conclusiones

- Random Forest global conserva la mejor AP y F1 en test. Su AP baja de 0,2782 en validación a 0,2512 en test. La selección realizada en validación se mantiene como referencia; no se reajusta a partir del test.
- La especialización por dominio no mejora la AP conjunta en ninguno de los dos algoritmos en test. La ligera ventaja local de HistGradientBoosting en validación no se mantiene.
- HistGradientBoosting global cubre más episodios, pero genera 1.666 alertas y 1.310 falsos positivos, frente a 1.491 alertas y 1.122 falsos positivos de Random Forest global. La comparación debe considerar tanto cobertura como carga de avisos.
- Los resultados son muy desiguales por dominio. Random Forest global obtiene AP de 0,2943 en deporte y 0,0941 en ciencia y tecnología. Cubre 73/88 episodios deportivos evaluables (82,95 %) y 6/29 de ciencia y tecnología (20,69 %). HistGradientBoosting global cubre 10/29 de ciencia y tecnología (34,48 %), con más falsas alertas en ese dominio (362 frente a 155).
- Hay capacidad para ordenar ventanas por riesgo, pero la precisión de las alertas es limitada: para Random Forest global solo el 24,75 % son correctas. El piloto no justifica presentar el sistema como una herramienta fiable de aviso sin más investigación.
- La evaluación se limita a 40 artículos y un semestre de test, con ventanas solapadas y episodios relacionados. No se han calculado intervalos de incertidumbre ni pruebas de significación; las diferencias son descriptivas. Los resultados de test ya se han observado y no deben reutilizarse para ajustar configuraciones y seguir denominándolo test independiente.

### Archivos y comprobaciones

- [summary.csv](reports/peak_test/summary.csv): métricas por ventana y episodio, globales y por dominio, para los cuatro modelos (12 filas).
- [by_article.csv](reports/peak_test/by_article.csv): métricas por artículo y modelo (160 filas). AP y recall quedan vacíos si no hay ventanas positivas; cobertura y antelación quedan sin definir cuando no hay episodios evaluables o cubiertos, respectivamente.
- [episodes.csv](reports/peak_test/episodes.csv): elegibilidad, cobertura y antelación de cada episodio para cada modelo (508 filas).
- [resultats.html](reports/resultats.html): informe autónomo actualizado, con secciones separadas de test y validación, filtros, detalle por artículo y cuatro casos con visitas, umbral, inicios reales y alertas de Random Forest global. Se regenera con [build_results_html.py](scripts/build_results_html.py).

Pasaron las 17 pruebas acumuladas. Contrasté de forma independiente las matrices de confusión, los episodios evaluables, su cobertura y la antelación para los 12 grupos; verifiqué las 7.120 claves únicas de predicción. Comprobé los datos incrustados del HTML, la ejecución de JavaScript, los filtros y la búsqueda, y la presencia de los cuatro SVG. No realicé una inspección visual en navegador.

```bash
python scripts/evaluate_peak_test.py
python scripts/build_results_html.py
python -m unittest discover -s tests -v
```

## Comparación complementaria con baselines

**Completada por petición expresa del equipo.** Implementé [evaluate_peak_baselines.py](scripts/evaluate_peak_baselines.py) con las dos reglas previstas en el apartado 7.4 del plan. Esta comparación se incorpora después de observar el test y debe interpretarse como complementaria, sin ajustes posteriores de las reglas según sus resultados.

- **Frecuencia histórica:** proporción de ventanas positivas de cada artículo en entrenamiento. No usa las etiquetas de validación o test para estimar esa frecuencia.
- **Subida reciente:** `(1 + media de t-2 a t) / (1 + mediana de t-32 a t-3)`, con días naturales del mismo artículo y sin visitas futuras. Es una puntuación, no una probabilidad.
- Para ambos baselines fijé un único umbral conjunto maximizando F1 de validación, con la misma regla y desempate utilizados para los modelos. Los umbrales son 0,1722222222 y 1,0730696798, respectivamente. Se fijan antes de puntuar test y se aplican sin reajuste.
- Se usan exactamente las mismas 7.000 filas de validación y 7.120 de test, con los mismos criterios de elegibilidad de episodios y volumen semanal.

| Método, test | AP | Precisión | Recall | F1 | Episodios cubiertos | Alertas/semana |
|---|---:|---:|---:|---:|---:|---:|
| Frecuencia histórica | 0,2104 | 0,2067 | 0,4482 | 0,2830 | 53/117 (45,30 %) | 70,00 |
| Subida reciente | 0,1895 | 0,1840 | 0,5420 | 0,2748 | 96/117 (82,05 %) | 95,09 |
| Random Forest global | 0,2512 | 0,2475 | 0,4495 | 0,3192 | 79/117 (67,52 %) | 58,63 |

Random Forest global supera ambas reglas en AP, F1 y precisión, y emite menos alertas. Frente a frecuencia histórica, mejora AP aproximadamente un 19,3 % relativo y cubre más episodios con menos avisos. Sin embargo, la regla de subida reciente cubre más episodios y tiene mayor recall por ventana, a costa de 2.418 alertas y 1.973 falsos positivos, frente a 1.491 y 1.122 de Random Forest global. Esto apoya el valor del modelo para priorizar riesgo y reducir falsas alertas, pero no una superioridad en todos los criterios. No se ha realizado una comparación a igual volumen de alertas ni una prueba de significación.

La frecuencia histórica es constante dentro de cada artículo: no localiza cambios temporales y puede anticipar episodios avisando todos los días de determinados artículos. La mediana de antelación es de siete días en ambos baselines y debe interpretarse junto con el volumen de alertas.

Los resultados globales y por dominio están en [summary.csv](reports/peak_baselines/summary.csv); las predicciones de ambos periodos, los resultados por artículo y el registro de episodios están en `reports/peak_baselines/`. [selection.json](reports/peak_baselines/selection.json) registra frecuencias, umbrales, fórmula y hashes de las entradas. El [HTML](reports/resultats.html) incorpora una sección comparativa con filtros por dominio y mantiene separadas validación y test.

Pasaron las 19 pruebas acumuladas. Las nuevas pruebas comprueban los límites exactos de la ventana, ausencia de visitas futuras, días ausentes y frecuencias por artículo. Se corrigieron dos comprobaciones anteriores de probabilidades para tolerar errores numéricos de hasta `1e-14` por sumas paralelas de árboles; no se modificaron modelos ni sus predicciones. Contrasté claves, frecuencias, umbrales fijos, AP y F1 de los CSV y comprobé la ejecución del HTML y su filtro de dominio. Para reconstruir decisiones exactamente desde CSV, deben leerse los números con precisión de ida y vuelta (`float_precision="round_trip"` en pandas).

```bash
python scripts/evaluate_peak_baselines.py
python scripts/build_results_html.py
python -m unittest discover -s tests -v
```

## Punto de continuación

El piloto de 40 artículos queda completado hasta el **paso 9**. El **paso 10** está pendiente: ampliar a 150 artículos y cinco dominios con el muestreo previsto en el plan. Cualquier cambio metodológico posterior deberá distinguirse de esta evaluación ya completada y validarse con datos reservados para ese nuevo experimento.
