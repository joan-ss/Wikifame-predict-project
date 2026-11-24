# The Dynamics of Fame — Proyecto de Machine Learning

## 1. Idea y objetivo

Queremos predecir la popularidad de un tema a partir de las visitas diarias a su artículo de Wikipedia.

La medida de popularidad será el número de visitas diarias al artículo. Para cada artículo tendremos una serie temporal y utilizaremos su historial para predecir las visitas del día siguiente.

La pregunta principal del proyecto es:

**¿Entrenar un predictor especializado por dominio mejora la predicción de las visitas del día siguiente respecto a un único predictor global?**

Compararemos:

1. Un modelo de referencia que predice las mismas visitas que el día anterior.
2. Un modelo global de Random Forest o Gradient Boosting entrenado con artículos de todos los dominios.
3. Modelos locales de Random Forest o Gradient Boosting entrenados únicamente con artículos de un mismo dominio.

El proyecto utilizará machine learning clásico, sin deep learning.

Los algoritmos candidatos serán **Random Forest y Gradient Boosting**.

**No elegiremos el algoritmo de antemano. Durante la validación entrenaremos y compararemos ambos algoritmos para decidir cuál utilizar en los modelos definitivos.**

La especialización será por categorías semánticas previamente definidas, no mediante K-Means.

---

## 2. Diseño general del proyecto

### 2.1. Experimento piloto

El proyecto comenzará con un piloto de **40 artículos de Wikipedia inglesa**, distribuidos entre dos dominios:

| Dominio              | Artículos |
| -------------------- | --------: |
| Deporte              |        20 |
| Ciencia y tecnología |        20 |
| **Total**            |    **40** |

Entrenaremos:

- Un modelo global con los 40 artículos.
- Un modelo local especializado en Deporte con 20 artículos.
- Un modelo local especializado en Ciencia y tecnología con 20 artículos.

Para cada enfoque probaremos Random Forest y Gradient Boosting durante la validación.

También utilizaremos un predictor de referencia:

- Predicción basada en las visitas del día anterior.

### 2.2. Objetivo principal del piloto

El piloto no será simplemente una prueba de que el código funciona.

Su objetivo será **comprobar la viabilidad de la metodología y obtener una primera evaluación experimental de la especialización por dominios**, antes de ampliar el proyecto.

Queremos responder a las siguientes preguntas:

**1. ¿Podemos construir correctamente el dataset?**

Comprobaremos que podemos:

- Obtener las visitas diarias mediante Wikimedia Pageviews.
- Extraer metadatos mediante Wikidata.
- Clasificar los artículos en dominios, subtemas y tipos de entidad.
- Construir variables temporales sin utilizar información futura.
- Generar automáticamente las observaciones para el predictor.

**2. ¿Los modelos de Machine Learning mejoran las predicciones de referencia?**

Compararemos Random Forest y Gradient Boosting con un predictor sencillo:

- Visitas del día anterior.

Esto nos permitirá comprobar si los modelos realmente aprenden relaciones útiles.

**3. ¿La especialización por dominio mejora la predicción?**

Compararemos el modelo global con los dos modelos locales.

Queremos investigar si las diferencias entre las dinámicas de popularidad de Deporte y Ciencia y tecnología justifican entrenar modelos independientes.

No asumimos que la especialización será mejor.

Si el modelo global obtiene mejores resultados, también será una conclusión relevante.

**4. ¿Las mejoras son consistentes entre artículos?**

Analizaremos si las diferencias entre modelos:

- Se mantienen en ambos dominios.
- Dependen de unos pocos artículos.
- Son diferentes según el subtema.
- Son diferentes según el tipo de entidad.
- Cambian entre periodos temporales.

**5. ¿Qué variables resultan útiles para predecir?**

Estudiaremos si las características temporales y semánticas aportan información relevante.

Podremos comparar distintas combinaciones de variables durante la validación.

**6. ¿Qué algoritmo funciona mejor?**

Durante la validación compararemos Random Forest y Gradient Boosting.

Queremos comprobar:

- Cuál obtiene menor error de predicción.
- Si el mejor algoritmo es el mismo para los modelos globales y locales.
- Si las diferencias entre algoritmos son consistentes en ambos dominios.
- Si alguno presenta mayor tendencia al sobreajuste.

**7. ¿Qué problemas debemos solucionar antes de escalar?**

Identificaremos:

- Artículos con datos incompletos.
- Problemas en la clasificación semántica.
- Variables poco informativas.
- Errores de implementación.
- Dificultades computacionales.
- Posibles problemas de generalización.

### 2.3. Qué esperamos obtener del piloto

Al terminar el piloto deberíamos tener:

1. Un procedimiento reproducible para descargar y procesar artículos.
2. Un dataset correctamente construido.
3. Un sistema de entrenamiento y evaluación temporal.
4. Una comparación inicial entre modelo global y modelos locales.
5. Una comparación entre Random Forest y Gradient Boosting.
6. Un análisis de errores por artículo y dominio.
7. Una metodología preparada para incorporar nuevos dominios.

**Importante:** el piloto permitirá obtener resultados exploratorios, pero no demostrar definitivamente que un enfoque es superior.

---

## 3. Selección de artículos y recogida de datos

### 3.1. Selección de los 40 artículos del piloto

No seleccionaremos simplemente los artículos más famosos ni aquellos que presenten curvas especialmente interesantes.

Utilizaremos un **muestreo aleatorio estratificado por dominio y subtema**, buscando diversidad de tipos de entidad.

#### Deporte — 20 artículos

| Subtema (`topic`) | Artículos |
| ----------------- | --------: |
| Fútbol            |         6 |
| Tenis             |         4 |
| Baloncesto        |         4 |
| Automovilismo     |         3 |
| Otros deportes    |         3 |
| **Total**         |    **20** |

Incluiremos diferentes tipos de entidad:

- Personas.
- Equipos y organizaciones.
- Competiciones.
- Acontecimientos deportivos.

#### Ciencia y tecnología — 20 artículos

| Subtema (`topic`)                | Artículos |
| -------------------------------- | --------: |
| Científicos                      |         4 |
| Empresas tecnológicas            |         4 |
| Tecnologías y productos          |         5 |
| Descubrimientos y conceptos      |         4 |
| Misiones y proyectos científicos |         3 |
| **Total**                        |    **20** |

Incluiremos:

- Personas.
- Empresas y organizaciones.
- Tecnologías y productos.
- Conceptos científicos.
- Proyectos y misiones.

Estas cantidades son cuotas de muestreo definidas para el experimento, no proporciones representativas de Wikipedia.

### 3.2. Criterios de selección

Los artículos deberán cumplir los siguientes criterios:

| Criterio           | Decisión                                                |
| ------------------ | ------------------------------------------------------- |
| Idioma             | Wikipedia inglesa                                       |
| Periodo de visitas | 01/01/2023–31/12/2025                                   |
| Existencia         | Artículo creado antes del 01/01/2023                    |
| Dominio            | Deporte o Ciencia y tecnología                          |
| Subtema            | Una categoría principal por artículo                    |
| Tipo de entidad    | Persona, organización, obra/producto, evento o concepto |
| Desambiguaciones   | Excluir                                                 |
| Redirecciones      | Resolver al artículo canónico                           |
| Selección          | Aleatoria estratificada con semilla fija                |
| Datos ausentes     | Detectar y revisar antes del modelado                   |

No seleccionaremos artículos según su popularidad futura ni según el rendimiento obtenido por los modelos.

### 3.3. Procedimiento de muestreo

1. Obtener un conjunto amplio de artículos candidatos mediante Wikidata.
2. Clasificarlos por dominio y subtema.
3. Comprobar que cumplen los criterios de elegibilidad.
4. Resolver títulos y redirecciones de Wikipedia.
5. Seleccionar aleatoriamente los artículos dentro de cada subtema.
6. Utilizar una semilla fija, por ejemplo `random_state = 42`.
7. Guardar la lista definitiva y documentar el procedimiento.

Procuraremos incluir varios tipos de entidad dentro de cada dominio, sin imponer proporciones artificiales cuando no sean adecuadas.

La muestra será reproducible, pero no necesariamente representativa de toda Wikipedia, debido a las cuotas utilizadas y a la cobertura de Wikidata.

### 3.4. Metadatos — `articles.csv`

**Una fila por artículo.**

En el piloto tendremos 40 filas y, en la versión completa, hasta 150.

| Columna      | Qué contiene                            | Ejemplo        |
| ------------ | --------------------------------------- | -------------- |
| `page_id`    | Identificador de la página de Wikipedia | ID numérico    |
| `qid`        | Identificador del elemento de Wikidata  | `Q615`         |
| `title`      | Título del artículo                     | `Lionel Messi` |
| `domain`     | Dominio general                         | `deporte`      |
| `topic`      | Subtema                                 | `fútbol`       |
| `topic_type` | Tipo de entidad                         | `persona`      |

Los metadatos se obtendrán utilizando Wikidata y las reglas de clasificación definidas para el proyecto.

`domain`, `topic` y `topic_type` son categorías de nuestro dataset, no necesariamente propiedades que Wikimedia proporcione directamente con esos nombres.

### 3.5. Visitas diarias — `pageviews.csv`

Descargaremos las visitas mediante Wikimedia Pageviews API.

Configuración:

- Proyecto: `en.wikipedia.org`.
- Acceso: `all-access`.
- Agente: `user`.
- Granularidad: `daily`.
- Periodo: 01/01/2023–31/12/2025.

El periodo contiene **1.096 días**.

Con 40 artículos tendremos un máximo de:

\[
40 \times 1096 = 43\,840
\]

observaciones diarias.

El archivo tendrá:

| Columna   | Qué contiene               |
| --------- | -------------------------- |
| `page_id` | Identificador del artículo |
| `date`    | Fecha                      |
| `views`   | Número de visitas          |

Definimos:

\[
x_{i,t}=\text{visitas del artículo }i\text{ en el día }t
\]

También definimos la transformación logarítmica:

\[
z_{i,t}=\ln(1+x_{i,t})
\]

Donde:

- \(x_{i,t}\): número original de visitas.
- \(z_{i,t}\): logaritmo natural de las visitas más uno.

**Siempre que aparezca \(z_{i,t}\) o \(z_{i,s}\) en este proyecto, estaremos trabajando con las visitas transformadas mediante el logaritmo natural.**

El índice \(i\) identifica el artículo y los índices \(t\) y \(s\) representan días.

**¿Por qué utilizamos logaritmos?**

Porque los artículos pueden tener niveles de popularidad muy diferentes.

Por ejemplo:

- Un artículo pasa de 100 a 200 visitas.
- Otro pasa de 100.000 a 200.000 visitas.

En ambos casos las visitas se duplican, pero las diferencias absolutas son muy distintas.

El logaritmo permite trabajar con cambios relativos de popularidad y reduce la influencia de los valores extremadamente grandes.

Sumamos uno para poder aplicar la transformación cuando un artículo tiene cero visitas, ya que \(\ln(0)\) no está definido.

**No transformaremos automáticamente todas las variables.** En la primera versión utilizaremos el logaritmo específicamente para calcular la tendencia y la dispersión, como se explica en el apartado 5.

Conservaremos siempre las visitas originales.

Los días ausentes se revisarán antes de decidir cómo tratarlos. No asumiremos automáticamente que una observación ausente equivale a cero visitas.

---

## 4. Segmentación por dominio

### 4.1. Primer nivel — `domain`

En el piloto tendremos dos segmentos:

- `deporte`
- `ciencia_tecnologia`

Cada artículo pertenece a uno de ellos.

La segmentación viene determinada por la clasificación semántica del artículo.

No utilizaremos K-Means en esta primera versión.

### 4.2. Selección del modelo local

En el experimento principal:

\[
\text{segment}_i=\text{domain}_i
\]

Por tanto, no necesitamos crear un archivo `segments.csv` independiente.

La variable `domain` determinará qué modelo local recibe cada observación.

Ejemplo:

| Artículo        | domain             | Predictor local             |
| --------------- | ------------------ | --------------------------- |
| Lionel Messi    | deporte            | Modelo Deporte              |
| Wimbledon       | deporte            | Modelo Deporte              |
| Albert Einstein | ciencia_tecnologia | Modelo Ciencia y tecnología |
| Microsoft       | ciencia_tecnologia | Modelo Ciencia y tecnología |

El predictor global utilizará las observaciones de ambos dominios.

### 4.3. Segundo nivel opcional — `domain` y `topic`

Más adelante podremos probar una jerarquía más específica.

Por ejemplo:

- `deporte → fútbol`
- `deporte → tenis`
- `ciencia_tecnologia → científicos`
- `ciencia_tecnologia → tecnologías`

Antes de hacerlo deberemos comprobar que cada subgrupo contiene suficientes artículos distintos.

No implementaremos este segundo nivel en el piloto inicial.

---

## 5. Predicción de visitas del día siguiente

### 5.1. Unidad de observación y objetivo

Cada fila del dataset de predicción representa un artículo en una fecha \(t\).

Utilizando las visitas conocidas hasta ese día, queremos predecir las visitas del día siguiente:

\[
\hat{x}_{i,t+1}=f(X_{i,t})
\]

Donde \(X_{i,t}\) contiene las características disponibles del artículo \(i\) hasta el día \(t\).

Definimos:

- `date`: último día observado.
- `target_date`: día que queremos predecir.
- `target_views`: visitas reales del día objetivo.

El objetivo será:

\[
y_{i,t}=x_{i,t+1}
\]

**La variable objetivo son las visitas originales, sin aplicar logaritmos.**

Por tanto, Random Forest y Gradient Boosting predecirán directamente el número de visitas del día siguiente.

### 5.2. Variables del predictor — primera versión

| Variable             | Cómo se calcula                                              |
| -------------------- | ------------------------------------------------------------ |
| `lag_1` a `lag_7`    | Visitas de los siete días anteriores a `target_date`         |
| `mean_3`             | Media de visitas de los últimos 3 días                       |
| `mean_7`             | Media de visitas de los últimos 7 días                       |
| `mean_30`            | Media de visitas de los últimos 30 días                      |
| `trend_30`           | Pendiente de la regresión lineal sobre `ln(1 + views)` de los últimos 30 días |
| `volatility_iqr_30`  | Rango intercuartílico de los logaritmos de las visitas de los últimos 30 días |
| `peak_intensity_7`   | Intensidad del máximo reciente respecto al nivel habitual anterior |
| `target_day_of_week` | Día de la semana que queremos predecir                       |
| `domain`             | Dominio del artículo                                         |
| `topic`              | Subtema del artículo                                         |
| `topic_type`         | Tipo de entidad                                              |

Para predecir las visitas del día \(t+1\):

\[
\text{lag}_1=x_{i,t}
\]

\[
\text{lag}_7=x_{i,t-6}
\]

Las medias móviles incluyen el día \(t\), pero nunca el día \(t+1\).

#### Tendencia — `trend_30`

La tendencia se calculará ajustando una regresión lineal:

\[
z_{i,s}=\beta_0+\beta_1s+\varepsilon_s
\]

sobre los últimos 30 días.

Recordemos que:

\[
z_{i,s}=\ln(1+x_{i,s})
\]

Por tanto, **la regresión se ajusta sobre los logaritmos de las visitas, no sobre las visitas originales**.

La característica que guardaremos será la pendiente estimada:

\[
\text{trend}_{30}=\hat{\beta}_1
\]

**¿Por qué utilizamos logaritmos en la tendencia?**

Porque queremos medir el crecimiento relativo de la popularidad.

Por ejemplo, pasar de 100 a 200 visitas representa el mismo crecimiento relativo que pasar de 10.000 a 20.000 visitas.

Si utilizáramos directamente las visitas originales, el segundo artículo tendría una pendiente mucho mayor, aunque ambos se hubieran duplicado.

Con logaritmos, la pendiente refleja aproximadamente la tasa de crecimiento relativa diaria.

Por ejemplo:

\[
\hat{\beta}_1=0.02
\]

corresponde aproximadamente a un crecimiento del 2 % diario de \(1+x\), según la recta ajustada.

**Esta regresión lineal no es nuestro modelo principal de predicción.** Solo la utilizamos para calcular una variable que después recibirá Random Forest o Gradient Boosting.

#### Dispersión — `volatility_iqr_30`

La dispersión se calculará mediante:

\[
\text{volatility\_iqr}_{30}
=
Q_{0.75}(z)-Q_{0.25}(z)
\]

Donde \(z\) representa los valores:

\[
z_{i,s}=\ln(1+x_{i,s})
\]

de los últimos 30 días.

Es decir, **calculamos el rango intercuartílico sobre los logaritmos de las visitas, no sobre las visitas originales**.

**¿Por qué utilizamos logaritmos en el IQR?**

Porque queremos comparar la dispersión de artículos con niveles de popularidad muy diferentes.

Sin logaritmos, un artículo con 100.000 visitas diarias podría presentar un IQR mucho mayor que otro con 100 visitas, simplemente por su escala.

Con logaritmos, la medida está más relacionada con las diferencias relativas entre los niveles de visitas.

**Importante:** el IQR mide la dispersión de los valores durante los últimos 30 días, pero no tiene en cuenta su orden temporal.

Por tanto, esta variable mide dispersión de la popularidad, no necesariamente fluctuaciones entre días consecutivos.

#### Intensidad de picos — `peak_intensity_7`

La intensidad del pico será:

\[
\text{peak\_intensity}_7=
\frac{1+\max(x_{i,t-6},\ldots,x_{i,t})}
{1+\operatorname{median}(x_{i,t-36},\ldots,x_{i,t-7})}
\]

El máximo utiliza los últimos siete días y la mediana los treinta anteriores a esos siete.

**Aquí no utilizamos logaritmos.**

Trabajamos con las visitas originales porque la propia división ya permite comparar el pico reciente con el nivel habitual del artículo.

### 5.3. Resumen del uso de logaritmos

| Variable                      | ¿Utiliza logaritmos? | Motivo                           |
| ----------------------------- | -------------------- | -------------------------------- |
| `lag_1` a `lag_7`             | No                   | Conservar las visitas originales |
| `mean_3`, `mean_7`, `mean_30` | No                   | Medir niveles medios de visitas  |
| `trend_30`                    | **Sí**               | Medir crecimiento relativo       |
| `volatility_iqr_30`           | **Sí**               | Medir dispersión relativa        |
| `peak_intensity_7`            | No                   | Ya es una relación entre visitas |
| `target_views`                | No                   | Predecir visitas originales      |

En la primera versión, **el logaritmo se utilizará únicamente para calcular `trend_30` y `volatility_iqr_30`**.

No aplicaremos logaritmos al resto de variables ni a la variable objetivo.

### 5.4. Variables adicionales opcionales

Durante la validación podremos estudiar:

| Variable               | Descripción                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `peak_count_90`        | Número de episodios de popularidad en los últimos 90 días   |
| `mean_episode_days_90` | Duración media observada de esos episodios                  |
| `autocorr_7_90`        | Autocorrelación con retardo semanal tras eliminar tendencia |

Será necesario fijar criterios reproducibles para detectar los episodios de popularidad.

No añadiremos estas variables automáticamente: comprobaremos durante la validación si mejoran el rendimiento.

### 5.5. Forma de `prediction_dataset.csv`

El archivo tendrá una fila por artículo y fecha de predicción.

| Bloque            | Columnas                                                 |
| ----------------- | -------------------------------------------------------- |
| Identificación    | `page_id`, `date`, `target_date`                         |
| Visitas recientes | `lag_1` a `lag_7`, `mean_3`, `mean_7`, `mean_30`         |
| Tendencia y picos | `trend_30`, `volatility_iqr_30`, `peak_intensity_7`      |
| Calendario        | `target_day_of_week`                                     |
| Metadatos         | `domain`, `topic`, `topic_type`                          |
| Objetivo          | `target_views`                                           |
| Opcionales        | `peak_count_90`, `mean_episode_days_90`, `autocorr_7_90` |

Las variables categóricas se codificarán para poder utilizarlas en Random Forest y Gradient Boosting.

Los codificadores se ajustarán únicamente con los datos de entrenamiento.

`page_id` se utilizará para identificar y unir observaciones, pero no como variable predictiva.

---

## 6. Partición temporal

Mantendremos la misma división temporal tanto en el piloto como en el experimento completo.

La partición se realizará según `target_date`.

| Periodo                 | Uso                                       |
| ----------------------- | ----------------------------------------- |
| 2023                    | Histórico inicial para construir ventanas |
| 2024                    | Entrenamiento                             |
| Enero–junio de 2025     | Validación                                |
| Julio–diciembre de 2025 | Test final                                |

Para el piloto de 40 artículos, suponiendo datos completos:

| Periodo        | Días |  Filas |
| -------------- | ---: | -----: |
| Histórico 2023 |  365 | 14.600 |
| Train 2024     |  366 | 14.640 |
| Validation     |  181 |  7.240 |
| Test           |  184 |  7.360 |

El histórico de 2023 servirá para calcular las características de las primeras observaciones de entrenamiento.

Las ventanas temporales podrán cruzar las fronteras entre periodos.

Por ejemplo, para predecir el 1 de enero de 2025 podemos utilizar visitas de diciembre de 2024.

No utilizaremos particiones aleatorias de las observaciones temporales.

### 6.1. Validación temporal

La primera versión utilizará la partición temporal fija descrita anteriormente.

Durante la validación compararemos Random Forest y Gradient Boosting y ajustaremos sus hiperparámetros.

Si necesitamos una evaluación más robusta para seleccionar hiperparámetros, podremos implementar validación temporal con ventanas expansivas.

En ese caso, los folds respetarán siempre el orden cronológico.

El periodo de test permanecerá reservado para la evaluación final.

---

## 7. Entrenamiento y predicción walk-forward

### 7.1. Entrenamiento inicial y selección del algoritmo

Entrenaremos los modelos con las observaciones de 2024.

Para cada enfoque, probaremos dos algoritmos:

- Random Forest.
- Gradient Boosting (`HistGradientBoostingRegressor`).

Durante el periodo de validación (enero–junio de 2025), compararemos ambos algoritmos utilizando MAE y RMSE.

En el piloto entrenaremos:

1. Random Forest global.
2. Gradient Boosting global.
3. Random Forest local para Deporte.
4. Gradient Boosting local para Deporte.
5. Random Forest local para Ciencia y tecnología.
6. Gradient Boosting local para Ciencia y tecnología.

El predictor de referencia basado en las visitas del día anterior no requiere entrenamiento.

La validación tendrá dos objetivos:

**1. Seleccionar el algoritmo y sus hiperparámetros**

Compararemos Random Forest y Gradient Boosting para determinar cuál ofrece mejores predicciones.

También utilizaremos la validación para ajustar los hiperparámetros y seleccionar las variables predictivas.

**2. Comparar la estrategia global con la especializada**

Para estudiar el efecto de la segmentación, compararemos por separado:

- Random Forest global frente a Random Forest local.
- Gradient Boosting global frente a Gradient Boosting local.

De esta manera podremos distinguir si una mejora se debe al algoritmo utilizado o a la especialización por dominio.

La elección definitiva del algoritmo se realizará utilizando exclusivamente los resultados de validación.

Una vez seleccionada la configuración, podremos reentrenar los modelos con los datos de entrenamiento y validación conjuntamente.

Finalmente, evaluaremos los modelos seleccionados en el periodo de test (julio–diciembre de 2025), que no se habrá utilizado para tomar decisiones.

**No utilizaremos el conjunto de test para elegir entre Random Forest y Gradient Boosting.**

### 7.2. Predicciones durante el test

Utilizaremos **walk-forward con horizonte de un día y modelo fijo**.

El procedimiento será:

1. Predecir las visitas del 1 de julio de 2025 utilizando información disponible hasta el 30 de junio.
2. Incorporar las visitas reales del 1 de julio cuando estén disponibles.
3. Recalcular las variables temporales.
4. Predecir las visitas del 2 de julio.
5. Repetir hasta el 31 de diciembre de 2025.

**No reentrenaremos Random Forest ni Gradient Boosting para cada predicción.**

Los parámetros del modelo permanecerán fijos durante el test.

Lo que cambiará serán las variables de entrada, porque utilizaremos las visitas reales más recientes.

Tampoco incorporaremos las predicciones como si fueran observaciones reales.

### 7.3. Reentrenamiento periódico opcional

Como extensión podremos comparar el modelo fijo con otro que se reentrene periódicamente.

Por ejemplo, cada 30 días.

Esto permitiría estudiar si actualizar los parámetros con datos recientes mejora las predicciones.

No formará parte del experimento inicial.

---

## 8. Modelos y evaluación

### 8.1. Modelos del piloto

Compararemos tres enfoques:

**Modelo 1 — Referencia del último día**

\[
\hat{x}_{i,t+1}=x_{i,t}
\]

Este predictor simplemente supone que mañana un artículo tendrá las mismas visitas que hoy.

No requiere entrenamiento.

**Modelo 2 — Modelo global**

Un único modelo entrenado con los 40 artículos de Deporte y Ciencia y tecnología.

Durante la validación probaremos Random Forest y Gradient Boosting.

El algoritmo definitivo será el seleccionado mediante los resultados de validación.

**Modelo 3 — Modelos locales por dominio**

Dos modelos independientes:

- Modelo Deporte.
- Modelo Ciencia y tecnología.

Cada modelo local utilizará únicamente las observaciones de los artículos de su dominio.

Durante la validación probaremos Random Forest y Gradient Boosting para cada enfoque local.

El algoritmo definitivo se seleccionará mediante los resultados de validación.

El modelo global y los modelos locales recibirán las mismas variables predictivas.

De esta manera, podremos estudiar si entrenar por separado aporta valor adicional cuando el modelo global ya conoce el dominio y el subtema.

Para aislar el efecto de la especialización, también compararemos el modelo global y los locales utilizando el mismo algoritmo.

### 8.2. Métricas

Utilizaremos principalmente:

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{N}\sum_{j=1}^{N}|y_j-\hat{y}_j|
\]

**Root Mean Squared Error (RMSE)**

\[
RMSE=
\sqrt{\frac{1}{N}\sum_{j=1}^{N}(y_j-\hat{y}_j)^2}
\]

Las métricas se calcularán sobre las visitas originales, no sobre la escala logarítmica.

También podremos calcular \(R^2\) como métrica complementaria.

### 8.3. Análisis de resultados

No nos limitaremos a calcular un único MAE o RMSE global.

Analizaremos:

1. Error total de cada enfoque.
2. Error por dominio.
3. Error por artículo.
4. Error según `topic`.
5. Error según `topic_type`.
6. Comportamiento durante episodios con picos de visitas.
7. Diferencias entre periodos temporales.

Esto nos permitirá identificar si la especialización mejora de forma consistente o si las diferencias se concentran en determinados artículos.

### 8.4. Dataset de resultados — `test_predictions.csv`

| Columna         | Qué contiene                              |
| --------------- | ----------------------------------------- |
| `page_id`       | Identificador del artículo                |
| `target_date`   | Fecha evaluada                            |
| `domain`        | Dominio del artículo                      |
| `views_actual`  | Visitas reales                            |
| `pred_global`   | Predicción del modelo global seleccionado |
| `pred_domain`   | Predicción del modelo local seleccionado  |
| `pred_last_day` | Referencia del último día                 |

En el piloto tendremos hasta 7.360 filas de test.

Todos los enfoques se evaluarán sobre las mismas observaciones.

---

## 9. Ampliación del proyecto

El proyecto tendrá **únicamente dos etapas principales**:

### Fase 1 — Piloto inicial

**40 artículos y 2 dominios**

- Deporte: 20.
- Ciencia y tecnología: 20.

Entrenaremos:

- Un predictor de referencia basado en las visitas del día anterior.
- Un modelo global, seleccionando entre Random Forest y Gradient Boosting durante la validación.
- Dos modelos locales, seleccionando entre Random Forest y Gradient Boosting durante la validación.

Objetivo: validar la metodología, seleccionar los algoritmos y realizar la primera comparación experimental.

### Fase 2 — Experimento completo

**150 artículos y 5 dominios**

| Dominio              | Artículos |
| -------------------- | --------: |
| Deporte              |        30 |
| Ciencia y tecnología |        30 |
| Cine y televisión    |        30 |
| Música               |        30 |
| Política             |        30 |
| **Total**            |   **150** |

Ampliaremos directamente del piloto de 40 artículos al experimento de 150.

Entrenaremos:

- Un predictor de referencia basado en las visitas del día anterior.
- Un modelo global entrenado con los 150 artículos.
- Cinco modelos locales, uno por dominio.

Durante la validación del experimento completo volveremos a comparar Random Forest y Gradient Boosting para seleccionar los algoritmos definitivos.

### Consideraciones para el escalado

Al ampliar el dataset:

- Mantendremos los artículos ya seleccionados.
- Conservaremos las mismas reglas de clasificación.
- Mantendremos las mismas variables y metodología de evaluación.
- Reentrenaremos el modelo global con los 150 artículos.
- Entrenaremos un predictor local para cada uno de los cinco dominios.
- Repetiremos la comparación entre Random Forest y Gradient Boosting utilizando validación.

Utilizaremos la validación para tomar decisiones metodológicas.

No utilizaremos repetidamente el test para escoger la configuración que produzca mejores resultados.

---

## 10. Orden de implementación

### Etapa 1 — Selección de artículos

1. Definir los conjuntos de candidatos de Deporte y Ciencia y tecnología.
2. Obtener candidatos mediante Wikidata.
3. Aplicar los criterios de elegibilidad.
4. Seleccionar 20 artículos por dominio mediante muestreo estratificado.
5. Crear `articles.csv`.

### Etapa 2 — Descarga de datos

1. Descargar las visitas diarias de 2023 a 2025.
2. Guardar `pageviews.csv`.
3. Comprobar títulos, identificadores y fechas.
4. Revisar datos ausentes y posibles anomalías.

### Etapa 3 — Construcción del dataset

1. Unir visitas y metadatos mediante `page_id`.
2. Calcular las variables temporales.
3. Construir las observaciones de predicción del día siguiente.
4. Comprobar que ninguna variable utiliza información posterior a la fecha de predicción.
5. Guardar `prediction_dataset.csv`.

### Etapa 4 — Entrenamiento y validación

1. Implementar el predictor de referencia del último día.
2. Entrenar Random Forest y Gradient Boosting globales.
3. Entrenar Random Forest y Gradient Boosting para Deporte.
4. Entrenar Random Forest y Gradient Boosting para Ciencia y tecnología.
5. Ajustar hiperparámetros utilizando el conjunto de validación.
6. Comparar los resultados de ambos algoritmos.
7. Seleccionar las configuraciones definitivas utilizando exclusivamente validación.

### Etapa 5 — Evaluación

1. Reentrenar, si procede, los modelos seleccionados con train y validation.
2. Realizar las predicciones walk-forward en el test.
3. Generar `test_predictions.csv`.
4. Calcular MAE y RMSE.
5. Analizar resultados por dominio, artículo y subtema.
6. Comparar los enfoques.

### Etapa 6 — Conclusiones del piloto

Responderemos:

- ¿El procedimiento de extracción y preparación de datos es fiable?
- ¿Random Forest o Gradient Boosting mejoran el predictor de referencia?
- ¿Qué algoritmo funciona mejor durante la validación?
- ¿Los modelos locales superan al global?
- ¿Las diferencias son consistentes en los dos dominios?
- ¿Qué tipos de artículos son más difíciles de predecir?
- ¿Qué variables parecen aportar más información?
- ¿Qué cambios necesitamos antes de escalar?

### Etapa 7 — Experimento completo

1. Ampliar directamente de 40 a 150 artículos.
2. Incorporar los tres dominios restantes.
3. Ampliar Deporte y Ciencia y tecnología a 30 artículos cada uno.
4. Entrenar Random Forest y Gradient Boosting globales y locales.
5. Seleccionar los algoritmos y sus hiperparámetros mediante validación.
6. Evaluar los modelos definitivos en el test.
7. Comparar los resultados de los cinco dominios.
8. Estudiar opcionalmente el segundo nivel de segmentación por `topic`.

---

## 11. Resultado esperado del proyecto

El resultado final será un estudio experimental sobre la predicción de la popularidad de artículos de Wikipedia.

Queremos determinar si las características semánticas de un artículo justifican entrenar modelos especializados por dominio o si un único modelo global es suficiente.

También estudiaremos si Random Forest o Gradient Boosting ofrecen mejores resultados para este problema, seleccionando el algoritmo mediante validación.

El piloto de 40 artículos permitirá desarrollar y comprobar la metodología, así como obtener resultados exploratorios.

El experimento completo de 150 artículos permitirá estudiar si las conclusiones se mantienen al aumentar la diversidad del dataset.

La contribución principal del proyecto no será únicamente construir un predictor, sino **comparar de manera sistemática la predicción global y la predicción especializada por dominios, utilizando información temporal y semántica de Wikipedia**.
