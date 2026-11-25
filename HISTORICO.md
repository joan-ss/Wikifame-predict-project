# Histórico del proyecto

## 2026-11-25 · Organización del proyecto en carpetas

Reorganizamos los archivos dentro de `Wikifame-predict-project`, conservando la carpeta del repositorio y `.git`. Los datos originales están en `data/raw`, los preparados en `data/processed`, el código en `src`, las pruebas en `tests`, los pipelines en `models/stage4`, las métricas y predicciones en `results/stage4`, la documentación en `docs` y el log en `logs`. Este histórico y `requirements.txt` permanecen en la raíz. Las rutas antiguas mencionadas en entradas anteriores describen el estado de aquel momento; los enlaces a archivos trasladados se han actualizado.

Verificamos mediante SHA-256 que los 22 archivos trasladados conservaban su contenido antes de adaptar código y documentación. Actualizamos las rutas de entrada y salida, los imports de las pruebas y del resumen, y los comandos de ejecución. Añadimos `src/__init__.py`, `tests/__init__.py` y un `README.md` con el mapa del proyecto. Se ejecutan los programas desde la raíz mediante `python -m src.train_validation` y `python -m src.summarize_validation`; las pruebas mediante `python -m unittest discover -s tests -v`. `.gitignore` excluye ahora `logs/`, además de las cachés de Python.

La reorganización conserva el comportamiento de reutilizar el dataset existente. No implica reentrenar modelos ni evaluar el test. Los CSV de datos y los seis modelos se han trasladado sin cambios. Las comprobaciones posteriores se registran a continuación.

Comprobaciones completadas: pasaron las tres pruebas de variables desde `tests`; regeneramos el resumen desde las nuevas rutas y mantuvo las mismas métricas; cargamos el Random Forest global desde `models/stage4` y reprodujo las predicciones guardadas. Verificamos que siguen presentes los seis modelos. Movimos la antigua caché de la raíz a `logs/legacy_python_cache`; las nuevas cachés quedan junto a sus módulos y están excluidas de Git. No modificamos el índice de Git ni creamos commits.

## 2026-11-25 · Reutilización del dataset de predicción

Modificamos `train_validation.py` para leer `prediction_dataset.csv` si ya existe y construirlo solo cuando falta. El dataset existente no se sobrescribe. El entrenamiento y la validación se siguen ejecutando en cada invocación. Para regenerar las variables tras modificar datos o fórmulas, hay que renombrar o eliminar el CSV preparado.

Al cargar el CSV, convertimos `date` y `target_date` a fechas mediante `parse_dates`, para mantener los filtros temporales de entrenamiento y validación. Añadimos mensajes de consola que indican si se carga el dataset existente o se construye uno nuevo. Actualizamos `ETAPA4.md` con este comportamiento.

Comprobamos la rama de carga del CSV existente: no llamó a la construcción del dataset, no cambió la fecha de modificación del archivo y conservó las 14.640 filas de entrenamiento al aplicar el filtro de 2024. Esta comprobación no volvió a entrenar los modelos. La rama de creación cuando falta el archivo no se ejecutó de nuevo en esta comprobación; conserva la construcción utilizada anteriormente.

Por petición del usuario, seguiremos registrando en este histórico los cambios realizados, las decisiones y las comprobaciones, indicando qué se ha ejecutado y qué queda pendiente.

Añadimos una entrada fechada cada vez que completamos un paso o tomamos una decisión sobre el proyecto.

## 2026-11-24 · Etapa 1 · Selección de artículos del piloto

Elegimos manualmente 40 artículos de Wikipedia en inglés para el piloto, con 20 de Deporte y 20 de Ciencia y tecnología. Respetamos las cuotas por subtema de [resum.md](docs/resum.md) y combinamos personas, organizaciones, productos, conceptos, competiciones y misiones para cubrir distintos patrones de visitas esperados.

| Dominio | Subtema | Cuota | Artículos elegidos |
| --- | --- | ---: | --- |
| Deporte | Fútbol | 6 | Lionel Messi, Boca Juniors, UEFA Champions League, FIFA Women's World Cup, Heysel Stadium disaster, Offside (association football) |
| Deporte | Tenis | 4 | Serena Williams, Australian Open, Davis Cup, Tennis scoring system |
| Deporte | Baloncesto | 4 | Michael Jordan, Golden State Warriors, NBA Finals, FIBA Basketball World Cup |
| Deporte | Automovilismo | 3 | Lewis Hamilton, Scuderia Ferrari, 24 Hours of Le Mans |
| Deporte | Otros deportes | 3 | Tour de France, Rugby World Cup, Simone Biles |
| Ciencia y tecnología | Científicos | 4 | Albert Einstein, Marie Curie, Alan Turing, Jennifer Doudna |
| Ciencia y tecnología | Empresas tecnológicas | 4 | Apple Inc., ASML Holding, Nintendo, SpaceX |
| Ciencia y tecnología | Tecnologías y productos | 5 | iPhone, Linux, Bitcoin, PlayStation 5, CRISPR gene editing |
| Ciencia y tecnología | Descubrimientos y conceptos | 4 | Black hole, Solar eclipse, Plate tectonics, Higgs boson |
| Ciencia y tecnología | Misiones y proyectos científicos | 3 | Apollo 11, James Webb Space Telescope, Mars 2020 |

La elección manual sustituye el sorteo estratificado previsto en el resumen para esta selección. Queda pendiente comprobar los títulos canónicos, los identificadores y la fecha de creación de cada artículo antes de generar `articles.csv`.

## 2026-11-24 · Etapas 1 y 2 · Validación y descarga de datos

1. Consultamos la API de Wikipedia en inglés para resolver redirecciones y obtener los títulos canónicos, los identificadores de página y los QID de Wikidata. Confirmamos que las 40 páginas son distintas, que ninguna es una página de desambiguación y que todas tienen una primera revisión anterior al 1 de enero de 2023. Guardamos los metadatos y la clasificación temática en [articles.csv](data/raw/articles.csv). [API de Wikipedia](https://www.mediawiki.org/wiki/API:Query)
2. Descargamos las visitas diarias del 1 de enero de 2023 al 31 de diciembre de 2025 con los parámetros `en.wikipedia.org`, `all-access`, `user` y `daily`. Guardamos los recuentos originales en [pageviews.csv](data/raw/pageviews.csv). [API de Wikimedia Pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/examples/page-metrics.html)
3. Verificamos las 43.840 observaciones contra las respuestas de Wikimedia. Hay 1.096 fechas por artículo, sin fechas ausentes ni pares `page_id`–`date` duplicados. Conservamos los picos observados. Por ejemplo, «FIFA Women's World Cup» registró 307.754 visitas el 20 de agosto de 2023, día de la final. [Fecha de la final según la FIFA](https://www.fifa.com/es/articles/a-que-hora-es-la-final-copa-mundial-femenina-fifa-2023)

### Problema de descarga · Cambio de título de ASML

Wikipedia trasladó «ASML Holding» a «ASML» el 28 de marzo de 2026. Al consultar el título actual para 2023–2025, la API devolvía recuentos distintos a los del artículo bajo el título que tenía en esos años. [Registro de traslados de Wikipedia](https://en.wikipedia.org/w/index.php?title=Special:Log&type=move&page=ASML+Holding)

Recuperamos las visitas de 2023–2025 con «ASML Holding» y conservamos «ASML» como título canónico en los metadatos. Vinculamos ambos por el mismo `page_id` (2875651) y comprobamos que la serie recuperada cubría los 1.096 días.

## 2026-11-25 · Criterio de evaluación del profesor

El profesor indicó que el origen del dataset queda fuera de la evaluación y que valorará el trabajo posterior con los datos.

Seguimos con los 40 artículos del piloto y centramos el trabajo en construir el dataset de predicción, entrenar y comparar los modelos y analizar los resultados.

## 2026-11-25 · Etapa 3 · Unión de visitas y metadatos

Unimos [pageviews.csv](data/raw/pageviews.csv) con [articles.csv](data/raw/articles.csv) por `page_id`. A cada fila de visitas le añadimos el QID, el título, el dominio, el subtema y el tipo de entidad del artículo. Guardamos el resultado intermedio en [pageviews_with_metadata.csv](data/processed/pageviews_with_metadata.csv).

Verificamos que las 43.840 filas conservan las fechas y los recuentos originales, que todas tienen los metadatos correspondientes y que cada artículo conserva sus 1.096 observaciones.

## 2026-11-25 · Etapas 3 y 4 · Variables, entrenamiento y validación

Completamos las variables temporales pendientes de la etapa 3 mediante [train_validation.py](src/train_validation.py), siguiendo las fórmulas de `resum.md`. Generamos [prediction_dataset.csv](data/processed/prediction_dataset.csv), con 42.360 filas utilizables, sin valores ausentes ni objetivos duplicados. Las ventanas requieren 37 días de histórico y descartamos el último día de cada artículo porque no tiene objetivo conocido.

Entrenamos exclusivamente con los 14.640 objetivos de 2024 y validamos con los 7.240 de enero–junio de 2025. Probamos seis configuraciones de Random Forest y cuatro de HistGradientBoosting en cada ámbito: global, deporte y ciencia y tecnología. Los codificadores se ajustan con entrenamiento; `page_id` no se utiliza como predictor. Conservamos todas las variables iniciales del resumen y no incorporamos las opcionales.

Fijamos MAE como criterio de selección y RMSE como desempate. Random Forest obtuvo el menor MAE entre los candidatos en los tres ámbitos. Los modelos locales combinados obtuvieron MAE 1.019,47 y RMSE 6.709,82; el global, MAE 1.041,42 y RMSE 6.832,46; el baseline del último día, MAE 1.086,64 y RMSE 7.342,10. La mejora de MAE frente al baseline fue 6,18 % para los locales y 4,16 % para el global. La ventaja local sobre el global fue aproximadamente 2,11 % en MAE.

Guardamos las configuraciones, seis pipelines, predicciones y métricas de validación en `stage4_results`. El [resumen de resultados](results/stage4/RESULTADOS.md) se genera con [summarize_validation.py](src/summarize_validation.py). Las instrucciones y el alcance están en [ETAPA4.md](docs/ETAPA4.md).

Verificamos mediante tres pruebas automáticas las fórmulas y los límites de las ventanas, la independencia de las características respecto a visitas futuras y el rechazo de fechas ausentes. También comprobamos que global y locales se comparan sobre exactamente los mismos objetivos y baseline.

No evaluamos el test ni reentrenamos con validación. Eso corresponde a la etapa 5. Los resultados actuales son exploratorios y se han utilizado para seleccionar configuraciones.

## 2026-11-25 · Detalles de implementación y archivos de la sesión

Revisamos `resum.md` y este histórico antes de implementar la etapa 4. Conservamos sin cambios `articles.csv`, `pageviews.csv`, `pageviews_with_metadata.csv` y `resum.md`. Detectamos que la etapa 3 documentada cubría la unión de tablas, pero aún faltaban las variables de predicción.

### Dataset y prevención de información futura

`build_dataset()` comprueba columnas obligatorias, valores ausentes, duplicados por artículo y fecha, visitas finitas y no negativas, días consecutivos y metadatos constantes por artículo. Ordena cada serie y calcula sus variables por separado para no mezclar artículos.

Las variables son `lag_1`–`lag_7`, `mean_3`, `mean_7`, `mean_30`, `trend_30`, `volatility_iqr_30`, `peak_intensity_7`, `target_day_of_week`, `domain`, `topic` y `topic_type`. Solo tendencia e IQR utilizan `ln(1 + views)`; el objetivo conserva las visitas originales. `date` identifica el último día observado y `target_date` el día siguiente. `target_views` contiene sus visitas reales, pero no se introduce como predictor. Se excluyen también identificadores, títulos y fechas como entradas directas.

Se descartan los primeros 36 días de cada artículo por falta de histórico y el último por falta de objetivo: `43.840 - 40 × 37 = 42.360` filas. El CSV preparado incluye fechas de test, pero esas filas no se usan para entrenar, seleccionar ni evaluar en esta etapa. Durante validación, las variables usan visitas reales anteriores al objetivo; los parámetros de los modelos permanecen fijos.

### Configuraciones probadas y seleccionadas

Random Forest usa 200 árboles y combina `min_samples_leaf` de 1, 5 o 20 con `max_features` de 0.7 o 1.0. Se fija `random_state=42` y `n_jobs=2`. HistGradientBoosting combina `max_leaf_nodes` de 15 o 31 con `learning_rate` de 0.05 o 0.1; utiliza 200 iteraciones, regularización L2 de 1, semilla 42 y `early_stopping=False` para evitar una partición interna aleatoria. Son 30 candidatos en total, más tres evaluaciones del baseline, no una búsqueda exhaustiva.

La codificación `OneHotEncoder` y el estimador se guardan juntos en un `Pipeline`. Se conserva el mejor candidato de cada algoritmo y ámbito, además de identificar el ganador por MAE. Los Random Forest seleccionados utilizan 200 árboles y `min_samples_leaf=5`; `max_features=0.7` en global y ciencia y tecnología, y `max_features=1.0` en deporte.

HistGradientBoosting obtuvo MAE 1.156,87 y RMSE 7.298,93 en global; los locales combinados obtuvieron MAE 1.284,33 y RMSE 9.596,38. La selección se basa en MAE, aunque las métricas no ordenen siempre los candidatos igual. Los errores en visitas originales pueden estar muy influidos por artículos de gran volumen; queda pendiente interpretar el desglose y los picos en detalle.

### Salidas y herramientas auxiliares

- `validation_scores.csv`: 33 filas con las 30 configuraciones y los tres baselines.
- `validation_predictions.csv`: 28.960 predicciones, cuatro por cada uno de los 7.240 objetivos: dos algoritmos globales y los dos algoritmos del dominio correspondiente.
- `validation_breakdown.csv`: métricas por dominio, artículo, subtema y tipo de entidad, con MAE del baseline.
- `strategy_comparison.csv`: comparación global/local para el mismo algoritmo, conjunta y por dominio. El RMSE conjunto se calcula desde los errores individuales.
- `selected_models.json`: ganadores, hiperparámetros, métricas, variables, tamaño de las particiones y versiones.
- `RESULTADOS.md`: resumen generado por `summarize_validation.py`, que verifica la igualdad de los objetivos y baselines entre estrategias antes de comparar.
- Seis archivos `.joblib`: mejores pipelines por algoritmo y ámbito, entrenados solo con 2024. Recargamos el Random Forest global y verificamos que reproduce las predicciones guardadas.
- `test_features.py`: tres pruebas automáticas de ventanas, independencia del futuro y rechazo de días ausentes; todas pasaron.
- `requirements.txt`: NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2 y joblib compatible (`>=1.2,<2`).
- `.gitignore`: excluye `__pycache__/`, `*.pyc` y `stage4_run.log`.
- `ETAPA4.md`: instrucciones de ejecución, metodología, archivos generados y comportamiento de reutilización del dataset.

Utilizamos Python 3.12.4 de `C:/Users/joans/anaconda3/python.exe`, porque el comando Python predeterminado no era accesible. La ejecución dejó un aviso de joblib sobre detección de núcleos; los modelos y resultados se generaron y se comprobaron. El log de ejecución queda en `stage4_run.log`.

Explicamos al usuario el contenido de los archivos, la diferencia entre construir el dataset y entrenar, la interpretación de las métricas y el cambio para reutilizar el CSV existente. El siguiente trabajo pendiente es la etapa 5: decidir el reentrenamiento con train+validation, evaluar el test reservado y profundizar en el análisis de errores.

## 2026-11-25 · Análisis exploratorio de la validación

Revisamos las 7.240 predicciones de validación de Random Forest. Los modelos por dominio tienen menor MAE que el global en 24 de los 40 artículos. Cinco artículos concentran el 51,6 % del error absoluto de los modelos por dominio: UEFA Champions League, Serena Williams, Lionel Messi, Lewis Hamilton y Michael Jordan.

Promediamos al 50 % las predicciones guardadas del Random Forest global y del modelo por dominio correspondiente. La combinación obtuvo MAE 1.009,99 y RMSE 6.674,52, frente a MAE 1.019,47 y RMSE 6.709,82 de los modelos por dominio. La reducción adicional de MAE es del 0,93 %. Es una comprobación exploratoria con las predicciones de enero a junio de 2025. El test de julio a diciembre sigue reservado. [Predicciones de validación](results/stage4/validation_predictions.csv)

## 2026-11-25 · Simplificación del código de la etapa 4

En [train_validation.py](src/train_validation.py) hicimos explícito el flujo de carga del dataset, entrenamiento, selección y guardado de los modelos. En [summarize_validation.py](src/summarize_validation.py) separamos el cálculo de las métricas del formato del informe.

Pasaron las tres pruebas de variables. En el mismo entorno local ejecutamos ambas versiones en carpetas temporales. Comprobamos las 42.360 filas del dataset regenerado y su reutilización, los seis modelos guardados, las 28.960 predicciones de validación y las métricas. El dataset, los modelos y el informe coincidieron exactamente. Las predicciones y métricas coincidieron dentro de la tolerancia numérica.

## 2026-11-25 · Sugerencia del profesor sobre la predicción de picos

El profesor consideró que las métricas actuales eran intermedias y sugirió explorar si podemos anticipar picos de visitas, con un horizonte de un día o de una semana.

Como primera comprobación, usamos una definición provisional por artículo. Marcamos una ventana como positiva si al menos uno de sus días alcanza tres veces la mediana de visitas de los 30 días disponibles al emitir la predicción. En 2024 contamos 362 positivos de 14.640 a un día y 1.190 de 14.400 a siete días. En enero–junio de 2025 contamos 178 de 7.240 y 599 de 7.000, respectivamente. Para cada periodo, todos los días de la ventana objetivo quedan dentro de sus fechas.

Las ventanas semanales se solapan y un mismo pico puede aparecer en varias ventanas positivas. El recuento sirve para estimar la frecuencia de la clase positiva antes de entrenar un clasificador. El test sigue reservado.

## 2026-11-25 · Experimento de clasificación de picos a siete días

Creamos [experiments/peak_prediction](experiments/peak_prediction/README.md) y copiamos `pageviews.csv` y `prediction_dataset.csv` dentro de la carpeta. Reutilizamos las variables temporales y construimos una etiqueta que indica si algún día de la ventana futura alcanza tres veces la mediana de los últimos 30 días. El umbral usa un mínimo de una visita para la mediana de referencia. El dataset contiene 14.400 ventanas de entrenamiento y 7.000 de validación.

Entrenamos Random Forest e HistGradientBoosting en los ámbitos global, Deporte y Ciencia y tecnología. Probamos 12 configuraciones y guardamos los seis mejores pipelines por algoritmo y ámbito. La regla de referencia alerta cuando detecta un pico en los últimos siete días observados. Seleccionamos modelos por precisión media (AP) y elegimos el umbral de cada modelo por F1 en validación.

Random Forest global obtuvo el mejor AP conjunto, 0,429 frente a 0,265 de la regla de referencia. Acertó 242 de 473 alertas y detectó 242 de las 599 ventanas con pico. Con el umbral global, el recall fue del 51,7 % en Deporte y del 5,5 % en Ciencia y tecnología. Julio a diciembre de 2025 queda reservado para la evaluación final. [Resultados de validación](experiments/peak_prediction/results/RESULTADOS.md)

Pasaron las tres pruebas de ventanas. Recalculamos las 21.400 etiquetas desde las visitas originales y comprobamos que los seis modelos guardados reproducen las predicciones. La carpeta incluye las versiones de Python y las dependencias empleadas para este experimento.
