# Predicción de picos de visitas a siete días

Los modelos se entrenaron con 2024 y se seleccionaron usando enero a junio de 2025.

| Estrategia | Algoritmo | Precisión media (AP) | Precisión | Recall | F1 | Alertas |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | recent_peak | 0.265 | 0.329 | 0.324 | 0.326 | 590 |
| global | hist_gradient_boosting | 0.413 | 0.431 | 0.431 | 0.431 | 598 |
| global | random_forest | 0.429 | 0.512 | 0.404 | 0.451 | 473 |
| local | hist_gradient_boosting | 0.375 | 0.343 | 0.439 | 0.385 | 766 |
| local | random_forest | 0.420 | 0.414 | 0.412 | 0.413 | 597 |

La precisión mide la proporción de alertas acertadas y el recall la proporción de ventanas con pico que reciben una alerta. AP resume el rendimiento para distintos umbrales.

Los umbrales de los modelos se eligen por F1 en validación. La regla de referencia emite una alerta cuando detecta un pico reciente con el factor fijo de tres.

Las ventanas se solapan. Un mismo pico puede aparecer en varias ventanas.

Julio a diciembre de 2025 queda reservado para la evaluación final.
