# The Dynamics of Fame — Predicción de picos de popularidad en Wikipedia

## 1. Idea y objetivo

Queremos predecir si un artículo de Wikipedia experimentará un nuevo pico de visitas durante los próximos siete días. Utilizaremos el historial de visitas del artículo y sus metadatos para entrenar modelos de clasificación binaria.

La pregunta principal del proyecto es:

**¿Podemos anticipar el inicio de un pico de popularidad durante los próximos siete días? ¿Mejora la predicción si entrenamos un modelo especializado por dominio en lugar de un único modelo global?**

Haremos **una predicción diaria por artículo**. Al terminar el día de hoy, el modelo dará una probabilidad de que empiece al menos un pico entre mañana y los siete días siguientes a hoy, ambos extremos incluidos.

| Momento de la predicción | Información disponible | Ventana que queremos predecir |
|---|---|---|
| Domingo por la noche | Visitas hasta el domingo incluido | Del lunes al domingo siguiente |
| Lunes por la noche | Visitas hasta el lunes incluido | Del martes al lunes siguiente |
| Martes por la noche | Visitas hasta el martes incluido | Del miércoles al martes siguiente |

Por ejemplo, si el domingo el modelo devuelve `0.80` para el artículo de las 24 Horas de Le Mans, estima una probabilidad del 80 % de que **empiece un pico en algún momento de los siete días siguientes**. El lunes actualizaremos la predicción con las visitas del lunes.

Con 40 artículos obtendremos **40 predicciones al día**. Cada predicción tiene una única salida y un horizonte de siete días.

---

## 2. Diseño general del proyecto

### 2.1. Experimento piloto

Reutilizaremos los **40 artículos de Wikipedia inglesa** que ya seleccionamos para el piloto inicial.

| Dominio | Artículos |
|---|---:|
| Deporte | 20 |
| Ciencia y tecnología | 20 |
| **Total** | **40** |

Compararemos dos estrategias de entrenamiento:

1. **Modelo global.** Entrenaremos un único clasificador con los 40 artículos.
2. **Modelos locales.** Entrenaremos un clasificador con los 20 artículos de Deporte y otro con los 20 de Ciencia y tecnología.

Para cada estrategia probaremos dos algoritmos de machine learning clásico:

- Random Forest (`RandomForestClassifier`).
- Gradient Boosting (`HistGradientBoostingClassifier`).

Elegiremos el algoritmo y sus hiperparámetros durante la validación.

### 2.2. Qué queremos comprobar en el piloto

1. ¿La definición de pico identifica episodios que reconocemos al observar las series?
2. ¿Cuántos episodios distintos aparecen en cada dominio y entre cuántos artículos se reparten?
3. ¿Podemos anticipar esos episodios utilizando únicamente información disponible antes de que empiecen?
4. ¿Cuántas falsas alertas generamos y con cuántos días de antelación avisamos?
5. ¿Un modelo especializado por dominio mejora al modelo global?

En el piloto anterior, el mejor Random Forest reducía aproximadamente un 6–7 % el MAE de la predicción de visitas del día siguiente respecto al predictor del último día. Conservaremos ese resultado como antecedente. El nuevo experimento tiene una variable objetivo binaria y se evaluará con métricas de clasificación y detección de episodios.

---

## 3. Datos necesarios

### 3.1. Reutilización de los dos CSV iniciales

**Para el piloto podemos reutilizar `articles.csv` y `pageviews.csv` tal como están.** La información necesaria para definir los picos ya se encuentra en las visitas diarias.

| Archivo | Una fila representa | Columnas que utilizaremos | Para qué sirve |
|---|---|---|---|
| `articles.csv` | Un artículo | `page_id`, `qid`, `title`, `domain`, `topic`, `topic_type` | Identificar el artículo, incorporar metadatos y seleccionar su modelo local |
| `pageviews.csv` | Un artículo en una fecha | `page_id`, `date`, `views` | Calcular el historial de visitas, detectar episodios y construir las etiquetas |

Mantendremos el periodo del proyecto original, **del 1 de enero de 2023 al 31 de diciembre de 2025**, y la misma configuración de visitas de Wikipedia inglesa (`all-access`, agente `user`, granularidad diaria).

El archivo que construiremos de nuevo será el **dataset de predicción**. Lo llamaremos `peak_prediction_dataset.csv` para distinguirlo del dataset de regresión anterior.

### 3.2. Comprobaciones iniciales

Antes de calcular ventanas temporales:

1. Verificaremos que cada combinación de `page_id` y `date` aparece una sola vez.
2. Ordenaremos las fechas dentro de cada artículo y comprobaremos que las visitas son cantidades no negativas.
3. Identificaremos los días ausentes. Un dato ausente y un día con cero visitas tienen significados diferentes.
4. Comprobaremos que cada artículo tiene asignados su dominio, subtema y tipo de entidad.

Todas las ventanas se calcularán sobre **días naturales consecutivos del mismo artículo**. Si faltan datos necesarios para una variable o para saber si comenzó un pico, marcaremos el resultado como desconocido. En el piloto usaremos las filas con variables y etiqueta completas y documentaremos cuántas se excluyen.

---

## 4. Definición de un pico de popularidad

### 4.1. Nivel habitual de cada artículo

Sea \(x_{i,d}\) el número de visitas del artículo \(i\) en el día \(d\).

Para decidir si las visitas de ese día son excepcionalmente altas, utilizaremos como referencia **los 30 días inmediatamente anteriores**, desde \(d-30\) hasta \(d-1\). El día que estamos evaluando queda fuera de su propia referencia.

Calcularemos dos cantidades sobre las **visitas originales**:

- \(m_{i,d}\): mediana de las visitas de esos 30 días. Representa el nivel habitual del artículo.
- \(IQR_{i,d}\): diferencia entre el percentil 75 y el percentil 25 de esas visitas. Mide cuánto varía la mitad central de sus valores.

Cada artículo tendrá su propio nivel habitual, que se actualizará cada día.

### 4.2. Cuándo consideramos que un día tiene visitas excepcionalmente altas

Como regla inicial del piloto utilizaremos:

\[
U_{i,d}=\max\left(2m_{i,d},\;m_{i,d}+3IQR_{i,d}\right).
\]

Marcaremos el día como alto si:

\[
x_{i,d}>U_{i,d}.
\]

Tomar el máximo significa que las visitas deben superar **ambos criterios**: el doble del nivel habitual y un incremento de tres IQR por encima de la mediana.

| Artículo | Mediana previa | IQR previo | Umbral | Visitas de hoy | ¿Día alto? |
|---|---:|---:|---:|---:|---|
| A | 1.000 | 100 | 2.000 | 2.200 | Sí |
| B | 1.000 | 500 | 2.500 | 2.200 | No |

Los dos artículos reciben hoy las mismas visitas y tienen la misma mediana. En B, las visitas habituales presentan más dispersión, por lo que exigimos una subida mayor.

### 4.3. Cómo pasamos de días altos a episodios

Un **episodio** es una secuencia de uno o más días altos consecutivos. Su fecha de inicio es el primer día de la secuencia.

| Día | Jueves | Viernes | Sábado | Domingo | Lunes |
|---|---:|---:|---:|---:|---:|
| Día alto | 0 | 1 | 1 | 1 | 0 |
| Inicio de episodio | 0 | 1 | 0 | 0 | 0 |

Aquí hay **un pico que empieza el viernes** y dura tres días. Aunque el máximo de visitas ocurra el sábado, la fecha que queremos anticipar es el viernes.

En el código, `peak_start` valdrá `1` cuando el día actual sea alto y el día anterior no lo fuera. Calcularemos los episodios sobre la serie continua de cada artículo para que un cambio de semana, mes o año no divida artificialmente un episodio.

### 4.4. Cómo fijaremos la definición definitiva

Los multiplicadores 2 y 3 son una **propuesta inicial**. Antes de entrenar los clasificadores:

1. Dibujaremos series de artículos de 2024 con sus visitas, el umbral diario y los inicios detectados.
2. Revisaremos ejemplos de ambos dominios y de artículos con distintos niveles de tráfico.
3. Contaremos los episodios por artículo y dominio. Comprobaremos si unos pocos artículos concentran la mayoría.
4. Revisaremos especialmente los artículos con muy pocas visitas. Si pasar de 1 a 3 visitas genera episodios irrelevantes para el proyecto, estudiaremos exigir también un aumento mínimo absoluto.
5. Documentaremos la regla elegida y la fijaremos **antes de comparar los modelos**. Aplicaremos la misma definición en entrenamiento, validación y test.

---

## 5. Construcción del dataset de predicción

### 5.1. Una fila por artículo y fecha de predicción

Cada fila de `peak_prediction_dataset.csv` responderá a la pregunta:

**«Al terminar el día \(t\), ¿empezará al menos un episodio de este artículo entre \(t+1\) y \(t+7\)?»**

| Columna | Significado |
|---|---|
| `page_id` | Artículo que estamos evaluando |
| `date` | Último día observado, \(t\) |
| `target_start` | Primer día del horizonte, \(t+1\) |
| `target_end` | Último día del horizonte, \(t+7\) |
| `peak_next_7d` | Vale 1 si empieza al menos un episodio dentro del horizonte y 0 si no empieza ninguno |

Si \(s_{i,d}\) es el indicador de inicio de episodio del artículo \(i\) en el día \(d\), la etiqueta será:

\[
y_{i,t}=\max(s_{i,t+1},s_{i,t+2},\ldots,s_{i,t+7}).
\]

Tomamos el máximo porque basta con que **uno** de esos siete valores sea 1. Si empiezan dos episodios dentro de la ventana, la etiqueta también vale 1.

### 5.2. Ejemplo de ventanas solapadas

Supongamos que un pico empieza el viernes 7 de marzo, continúa el sábado y termina el domingo. En este ejemplo no comienza ningún otro episodio.

| Día al terminar el cual predecimos | Ventana futura | `peak_next_7d` | Motivo |
|---|---|---:|---|
| Jueves 27 de febrero | 28 de febrero–6 de marzo | 0 | El episodio empieza después de la ventana |
| Viernes 28 de febrero | 1–7 de marzo | 1 | El viernes 7 entra como último día del horizonte |
| Jueves 6 de marzo | 7–13 de marzo | 1 | El episodio empieza al día siguiente |
| Viernes 7 de marzo | 8–14 de marzo | 0 | El episodio ya ha empezado y durante esta ventana no empieza otro |

Por tanto, **un episodio puede generar hasta siete filas positivas**, aunque siga siendo un único episodio real. Tendremos en cuenta esta diferencia al evaluar los resultados.

### 5.3. Variables que reutilizaremos

Para construir los predictores de la fila del día \(t\), podemos utilizar las visitas hasta \(t\) incluido. Las visitas de \(t+1\) a \(t+7\) sirven para construir la etiqueta histórica.

| Variable | Cálculo al terminar el día \(t\) | Qué aporta |
|---|---|---|
| `lag_1` a `lag_7` | `lag_1` son las visitas de hoy, `lag_2` las de ayer y `lag_7` las de hace seis días | Forma reciente de la serie |
| `mean_3`, `mean_7`, `mean_30` | Media de visitas de los últimos 3, 7 y 30 días, incluyendo hoy | Nivel de atención a distintas escalas |
| `trend_30` | Pendiente de una recta ajustada a los últimos 30 valores de \(\ln(1+x)\) frente al tiempo | Tendencia reciente de crecimiento o descenso |
| `volatility_iqr_30` | IQR de los últimos 30 valores de \(\ln(1+x)\) | Dispersión relativa de los niveles recientes |
| `peak_intensity_7` | Máximo de visitas de los últimos 7 días comparado con la mediana de los 30 anteriores a esos 7 | Intensidad del máximo reciente respecto al nivel anterior |
| `domain`, `topic`, `topic_type` | Metadatos de `articles.csv` | Dominio, subtema y tipo de entidad |

Conservaremos la fórmula del proyecto original para la intensidad reciente:

\[
\text{peak\_intensity\_7}=
\frac{1+\max(x_{i,t-6},\ldots,x_{i,t})}
     {1+\operatorname{mediana}(x_{i,t-36},\ldots,x_{i,t-7})}.
\]

Conviene distinguir dos cálculos que aparecen en el proyecto:

- **Para detectar y etiquetar picos**, la mediana y el IQR se calculan en visitas originales y excluyen el día evaluado.
- **Como entrada del modelo**, `volatility_iqr_30` utiliza visitas logarítmicas e incluye el día de predicción. También utilizamos logaritmos en `trend_30`.

### 5.4. Variables de calendario

En el proyecto anterior, `target_day_of_week` indicaba el día cuyas visitas queríamos predecir. Ahora cada horizonte contiene los siete días de la semana. Para este proyecto probaremos:

| Variable | Definición | Qué queremos comprobar |
|---|---|---|
| `forecast_day_of_week` | Día de la semana en que emitimos la predicción | Si el momento en que empieza la ventana aporta información |
| `target_week_of_year` | Semana ISO del año a la que pertenece el primer día del horizonte | Si la época del año ayuda a anticipar picos estacionales |

Por ejemplo, `target_week_of_year` puede aportar información sobre competiciones que se celebran en fechas parecidas cada año. Con un único año de entrenamiento, veremos cada edición anual como mucho una vez por artículo. Compararemos en validación el rendimiento con y sin calendario para comprobar cuánto aporta.

Ambas variables se obtienen de `date`. Incorporar en el futuro el calendario oficial de competiciones sería una extensión distinta que requeriría fechas de eventos conocidas al emitir la predicción.

### 5.5. Comprobaciones antes de entrenar

- `page_id`, `qid` y `title` se utilizarán para identificar artículos y unir archivos.
- Ajustaremos los codificadores de variables categóricas con el conjunto de entrenamiento y contemplaremos categorías nuevas en validación o test.
- Guardaremos `peak_start` en la tabla diaria de episodios para poder evaluar los avisos. Los inicios **futuros** y cualquier resumen de las visitas **futuras** quedarán fuera de las entradas del clasificador.
- En cada periodo registraremos tanto el número de **filas positivas** como el número de **episodios distintos**.

---

## 6. Partición temporal

Conservaremos la división temporal del proyecto original. Para asignar una fila a un conjunto, sus **siete días objetivo completos** deberán estar dentro de ese periodo.

| Periodo | Uso | Fechas de predicción admisibles \(t\) | Máximo de filas con 40 artículos |
|---|---|---|---:|
| 2023 | Histórico para calcular ventanas previas | — | — |
| 2024 | Entrenamiento | 31/12/2023–24/12/2024 | 14.400 |
| Enero–junio de 2025 | Validación | 31/12/2024–23/06/2025 | 7.000 |
| Julio–diciembre de 2025 | Test final | 30/06/2025–24/12/2025 | 7.120 |

Por ejemplo, la predicción del **23 de junio** cubre del 24 al 30 de junio y entra en validación. La del **24 de junio** cubre del 25 de junio al 1 de julio y cruza la frontera, por lo que se excluye. La del **30 de junio** cubre del 1 al 7 de julio y es la primera del test.

Los predictores sí pueden utilizar visitas anteriores al comienzo de un periodo. Al predecir el 30 de junio ya conocemos el historial de junio. Esta es la información que también tendríamos en una aplicación real.

Evitaremos dividir filas al azar. Dos filas consecutivas del mismo artículo comparten seis de sus siete días objetivo.

---

## 7. Entrenamiento y predicción diaria

### 7.1. Modelos del piloto

| Estrategia | Random Forest | Gradient Boosting |
|---|---|---|
| Global | Un clasificador con los 40 artículos | Un clasificador con los 40 artículos |
| Local por dominio | Un clasificador para cada dominio | Un clasificador para cada dominio |

El modelo global recibirá `domain` y podrá aprender diferencias entre dominios. En la estrategia local, `domain` determinará qué clasificador se aplica a cada artículo. Utilizaremos las mismas fechas y variables para comparar ambas estrategias.

Primero compararemos **global frente a local dentro de cada algoritmo**. Así podremos estudiar por separado el efecto de especializar por dominio y el efecto de cambiar de algoritmo.

### 7.2. Selección en validación

1. Entrenaremos los candidatos con las filas de 2024.
2. Obtendremos para cada fila de validación una puntuación de riesgo mediante la probabilidad estimada de la clase positiva.
3. Compararemos los modelos con **Average Precision**, definida en el apartado 8, y revisaremos los resultados de cada dominio.
4. Elegiremos el algoritmo, los hiperparámetros y las variables utilizando validación.
5. Elegiremos también cómo convertir la probabilidad en alerta. Como criterio inicial, usaremos el umbral que maximice el F1 de validación. Guardaremos ese valor para aplicarlo en test.

Hay **dos umbrales diferentes** en este proyecto. El umbral de **visitas** del apartado 4 define lo que cuenta como episodio real. El umbral de **probabilidad** de este apartado decide cuándo el modelo lanza una alerta.

### 7.3. Evaluación walk-forward

Para la primera evaluación mantendremos el modelo entrenado y el umbral seleccionados. Durante el test avanzaremos un día cada vez:

1. Al terminar el día \(t\), actualizaremos los predictores con las visitas reales disponibles hasta ese día.
2. El modelo calculará la probabilidad de que empiece un pico entre \(t+1\) y \(t+7\).
3. Guardaremos la probabilidad y la decisión de alerta.
4. Al día siguiente incorporaremos las nuevas visitas reales y repetiremos el proceso para el nuevo horizonte.

No hace falta esperar siete días para **emitir** la siguiente predicción. Para **evaluar completamente** la predicción anterior sí hace falta conocer lo ocurrido durante su horizonte.

### 7.4. Baselines opcionales — solo por decisión expresa del equipo

**El plan principal compara los clasificadores globales y locales. Implementaremos los siguientes baselines únicamente si el equipo decide solicitarlos expresamente.**

| Baseline opcional | Cómo funcionaría |
|---|---|
| Frecuencia histórica por artículo | Asignaría al artículo la proporción de sus filas de entrenamiento con `peak_next_7d = 1`. Por ejemplo, si fueron positivas el 12 % de sus ventanas, recibiría una puntuación constante de 0,12. |
| Subida reciente | Dividiría \(1+\) la media de visitas de los últimos 3 días entre \(1+\) la mediana de los 30 días anteriores a esos 3. Una puntuación alta indicaría una subida reciente. |

Si se activara esta comparación, ambos se calcularían con los dos CSV existentes y se evaluarían sobre las mismas filas que los clasificadores. Para el segundo baseline, el umbral de alerta se elegiría en validación.

---

## 8. Evaluación de resultados

### 8.1. Métricas por predicción diaria

Una predicción diaria es positiva cuando se emite una alerta. Es un acierto si empieza al menos un episodio durante sus siete días objetivo.

| Métrica | Pregunta que responde |
|---|---|
| **Average Precision (AP)** | Al ordenar las ventanas por riesgo, ¿concentra el modelo las ventanas realmente positivas en las primeras posiciones? Resume la relación entre precisión y recall al variar el umbral. |
| **Precisión** | De todas las alertas emitidas, ¿qué proporción tuvo un inicio de episodio en los siete días siguientes? |
| **Recall o sensibilidad** | De todas las ventanas realmente positivas, ¿qué proporción recibió una alerta? |
| **F1** | ¿Qué equilibrio entre precisión y recall obtenemos con el umbral elegido? |

Acompañaremos las métricas con el número de alertas, la matriz de confusión y la proporción de ventanas realmente positivas. Por ejemplo, si solo el 5 % de las ventanas es positivo, acertar muchos días tranquilos podría producir una exactitud alta incluso sin anticipar ningún pico.

### 8.2. Métricas por episodio

Además de evaluar las filas diarias, evaluaremos cada **inicio real** una sola vez. Para un pico que empieza el día \(d\), buscaremos alertas del mismo artículo emitidas desde \(d-7\) hasta \(d-1\).

- **Cobertura de episodios.** Número de episodios que recibieron al menos una alerta previa dividido por el número total de episodios evaluables.
- **Antelación.** Para cada episodio cubierto, calcularemos cuántos días separan la primera alerta válida del inicio real. Resumiremos esos valores con la mediana.
- **Volumen de alertas.** Indicaremos cuántas alertas emite cada modelo por semana. Esto ayuda a interpretar si una mayor cobertura se consigue a costa de avisar con más frecuencia.

Por ejemplo, si un pico empieza el viernes y el modelo avisó el lunes, miércoles y jueves, contabilizaremos **un episodio cubierto** con **cuatro días de antelación**. Las tres alertas sí cuentan al calcular el volumen de avisos.

Para comparar la cobertura y la antelación, utilizaremos episodios cuyas siete fechas previas de predicción sean evaluables dentro del conjunto correspondiente. Indicaremos cuántos quedan fuera por estar en los bordes del periodo o por datos ausentes.

### 8.3. Análisis por dominio y artículo

Presentaremos los resultados:

1. Sobre el conjunto de artículos.
2. Por separado para Deporte y Ciencia y tecnología.
3. Por artículo, acompañados del número de episodios disponibles.
4. Mediante gráficos de casos concretos con visitas observadas, umbral de pico, inicios reales y alertas emitidas.

Si un artículo no presenta episodios en test, podremos estudiar sus falsas alertas. Su cobertura de episodios y su antelación quedarán sin definir.

Como análisis complementario, podremos utilizar SHAP para estudiar qué variables influyen en predicciones concretas. Por ejemplo, podremos examinar si una alerta temprana se relaciona con `trend_30`, con una subida reciente de `mean_3` o con el calendario.

### 8.4. Archivos de resultados

| Archivo | Contenido principal |
|---|---|
| `peak_episodes.csv` | Una fila por episodio con `page_id`, fecha de inicio y último día alto consecutivo |
| `peak_prediction_dataset.csv` | Una fila por artículo y fecha de predicción con variables históricas, horizonte y etiqueta |
| `test_predictions.csv` | Artículo, fecha, horizonte, etiqueta real, probabilidad global y local, y decisión de alerta de cada estrategia |

Estos archivos se construirán a partir de `articles.csv` y `pageviews.csv`.

---

## 9. Ampliación del proyecto

El proyecto tendrá dos etapas principales. Pasaremos directamente del piloto de 40 artículos al experimento completo de 150.

| Dominio | Piloto | Experimento completo |
|---|---:|---:|
| Deporte | 20 | 30 |
| Ciencia y tecnología | 20 | 30 |
| Cine y televisión | — | 30 |
| Música | — | 30 |
| Política | — | 30 |
| **Total** | **40** | **150** |

Mantendremos los artículos del piloto y completaremos las cuotas siguiendo el muestreo estratificado por dominio y subtema del proyecto original. Para los nuevos artículos ampliaremos los mismos dos CSV con metadatos y visitas del mismo periodo, sin seleccionar artículos por lo llamativos que sean sus picos.

En el experimento completo compararemos **un modelo global y cinco modelos locales**, utilizando la misma regla de episodios, horizonte temporal y procedimiento de evaluación. Si en algún dominio hay pocos episodios, lo tendremos en cuenta al interpretar su resultado. La segmentación adicional por `topic` quedará como extensión si hay suficientes artículos y episodios por subgrupo.

---

## 10. Orden de implementación

| Paso | Qué haremos | Resultado |
|---|---|---|
| 1 | Revisar los dos CSV del piloto y comprobar artículos, fechas, duplicados y ausencias | Series diarias utilizables |
| 2 | Calcular para cada día la mediana y el IQR de sus 30 días anteriores | Referencia histórica por artículo y día |
| 3 | Aplicar la regla inicial, agrupar días altos consecutivos y revisar ejemplos de 2024 | Definición de pico documentada y tabla de episodios |
| 4 | Para cada artículo y día \(t\), comprobar si empieza algún episodio entre \(t+1\) y \(t+7\) | Etiqueta `peak_next_7d` |
| 5 | Incorporar los predictores históricos, el calendario y los metadatos | `peak_prediction_dataset.csv` |
| 6 | Separar entrenamiento, validación y test usando el horizonte completo de cada fila | Conjuntos cronológicos y recuentos de filas y episodios |
| 7 | Entrenar y validar Random Forest y Gradient Boosting globales y locales | Algoritmos, variables y umbrales seleccionados |
| 8 | Emitir predicciones diarias sobre test con la configuración fijada | `test_predictions.csv` |
| 9 | Calcular métricas por ventana y episodio, comparar dominios y revisar ejemplos | Conclusiones del piloto |
| 10 | Ampliar directamente a 150 artículos y repetir la comparación en cinco dominios | Resultado del experimento completo |

---

## 11. Resultado esperado del proyecto

Al finalizar podremos responder a tres preguntas distintas:

1. **Capacidad predictiva.** ¿Con qué precisión podemos avisar de que empezará un pico durante los próximos siete días?
2. **Antelación práctica.** ¿Qué proporción de los episodios detectamos y cuántos días antes conseguimos avisar?
3. **Especialización.** ¿Para qué dominios mejora el resultado al entrenar un modelo propio, y en cuáles funciona mejor compartir información mediante el modelo global?

La conclusión principal combinará esas tres perspectivas. Por ejemplo, un modelo puede encontrar más episodios y, al mismo tiempo, emitir más falsas alertas o avisar con menos antelación. Mostraremos esas diferencias para explicar **cuándo y para qué artículos resulta útil la predicción**.
