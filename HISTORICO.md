# Histórico del proyecto

Añadimos una entrada fechada cada vez que completamos un paso o tomamos una decisión sobre el proyecto.

## 2026-11-24 · Etapa 1 · Selección de artículos del piloto

Elegimos manualmente 40 artículos de Wikipedia en inglés para el piloto, con 20 de Deporte y 20 de Ciencia y tecnología. Respetamos las cuotas por subtema de [resum.md](resum.md) y combinamos personas, organizaciones, productos, conceptos, competiciones y misiones para cubrir distintos patrones de visitas esperados.

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

1. Consultamos la API de Wikipedia en inglés para resolver redirecciones y obtener los títulos canónicos, los identificadores de página y los QID de Wikidata. Confirmamos que las 40 páginas son distintas, que ninguna es una página de desambiguación y que todas tienen una primera revisión anterior al 1 de enero de 2023. Guardamos los metadatos y la clasificación temática en [articles.csv](articles.csv). [API de Wikipedia](https://www.mediawiki.org/wiki/API:Query)
2. Descargamos las visitas diarias del 1 de enero de 2023 al 31 de diciembre de 2025 con los parámetros `en.wikipedia.org`, `all-access`, `user` y `daily`. Guardamos los recuentos originales en [pageviews.csv](pageviews.csv). [API de Wikimedia Pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/examples/page-metrics.html)
3. Verificamos las 43.840 observaciones contra las respuestas de Wikimedia. Hay 1.096 fechas por artículo, sin fechas ausentes ni pares `page_id`–`date` duplicados. Conservamos los picos observados. Por ejemplo, «FIFA Women's World Cup» registró 307.754 visitas el 20 de agosto de 2023, día de la final. [Fecha de la final según la FIFA](https://www.fifa.com/es/articles/a-que-hora-es-la-final-copa-mundial-femenina-fifa-2023)

### Problema de descarga · Cambio de título de ASML

Wikipedia trasladó «ASML Holding» a «ASML» el 28 de marzo de 2026. Al consultar el título actual para 2023–2025, la API devolvía recuentos distintos a los del artículo bajo el título que tenía en esos años. [Registro de traslados de Wikipedia](https://en.wikipedia.org/w/index.php?title=Special:Log&type=move&page=ASML+Holding)

Recuperamos las visitas de 2023–2025 con «ASML Holding» y conservamos «ASML» como título canónico en los metadatos. Vinculamos ambos por el mismo `page_id` (2875651) y comprobamos que la serie recuperada cubría los 1.096 días.

## 2026-11-25 · Criterio de evaluación del profesor

El profesor indicó que el origen del dataset queda fuera de la evaluación y que valorará el trabajo posterior con los datos.

Seguimos con los 40 artículos del piloto y centramos el trabajo en construir el dataset de predicción, entrenar y comparar los modelos y analizar los resultados.

## 2026-11-25 · Etapa 3 · Unión de visitas y metadatos

Unimos [pageviews.csv](pageviews.csv) con [articles.csv](articles.csv) por `page_id`. A cada fila de visitas le añadimos el QID, el título, el dominio, el subtema y el tipo de entidad del artículo. Guardamos el resultado intermedio en [pageviews_with_metadata.csv](pageviews_with_metadata.csv).

Verificamos que las 43.840 filas conservan las fechas y los recuentos originales, que todas tienen los metadatos correspondientes y que cada artículo conserva sus 1.096 observaciones.
