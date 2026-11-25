# Resultados de validación · Etapa 4

Entrenamiento: 2024. Validación: enero–junio de 2025. Test sin evaluar.

Selección por MAE mínimo; RMSE para desempatar. Valores en visitas.

| Ámbito | Algoritmo elegido | MAE | RMSE | MAE baseline |
| --- | --- | ---: | ---: | ---: |
| global | random_forest | 1041.42 | 6832.46 | 1086.64 |
| ciencia_tecnologia | random_forest | 531.26 | 1513.18 | 564.67 |
| deporte | random_forest | 1507.68 | 9367.70 | 1608.61 |

## Comparación con el mismo algoritmo

Los modelos locales se evalúan conjuntamente sobre los mismos 7.240 objetivos que el global.

| Algoritmo | Estrategia | MAE | RMSE | Mejora MAE frente al baseline |
| --- | --- | ---: | ---: | ---: |
| hist_gradient_boosting | global | 1156.87 | 7298.93 | -6.46% |
| hist_gradient_boosting | local | 1284.33 | 9596.38 | -18.19% |
| random_forest | global | 1041.42 | 6832.46 | 4.16% |
| random_forest | local | 1019.47 | 6709.82 | 6.18% |

Una mejora positiva indica menor MAE que el baseline. Estos resultados se han utilizado para seleccionar configuraciones; no sustituyen la evaluación final en test.

La muestra fue elegida manualmente según HISTORICO.md. Las conclusiones corresponden a estos artículos y no garantizan generalización a toda Wikipedia.
