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

## Punto de continuación

El **paso 4** está pendiente. Usará los inicios observados para construir `peak_next_7d` en cada fecha de predicción.
