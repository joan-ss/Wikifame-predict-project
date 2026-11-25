# Wikifame Predict

Predicción de visitas diarias a Wikipedia y comparación de modelos globales y especializados por dominio.

## Organización

```text
Wikifame-predict-project/
├── data/
│   ├── raw/                 # Artículos y visitas originales
│   └── processed/           # Unión con metadatos y dataset de predicción
├── src/                     # Entrenamiento y resumen de validación
├── tests/                   # Comprobaciones de las variables
├── models/stage4/           # Seis pipelines entrenados
├── results/stage4/          # Predicciones, métricas y resumen
├── docs/                    # Definición del proyecto y guía de la etapa 4
├── logs/                    # Registro de ejecución, excluido de Git
├── HISTORICO.md             # Registro de decisiones y cambios
└── requirements.txt         # Dependencias
```

## Ejecución

Abrir una terminal en esta carpeta, no dentro de `src`, y ejecutar:

```powershell
& 'C:/Users/joans/anaconda3/python.exe' -m src.train_validation
& 'C:/Users/joans/anaconda3/python.exe' -m src.summarize_validation
& 'C:/Users/joans/anaconda3/python.exe' -m unittest discover -s tests -v
```

En otro entorno, instalar `requirements.txt` y sustituir la ruta del intérprete por la correspondiente.

El entrenamiento reutiliza `data/processed/prediction_dataset.csv` si existe. Solo construye ese archivo cuando falta; para regenerarlo tras cambiar los datos o las fórmulas, renombrarlo o eliminarlo. Cada ejecución del entrenamiento vuelve a entrenar los modelos y guarda sus resultados. El resumen se genera por separado.

Los modelos actuales se entrenaron con 2024 y se seleccionaron con enero–junio de 2025. El test de julio–diciembre de 2025 sigue reservado.

## Documentación

- [Definición del proyecto](docs/resum.md)
- [Guía de entrenamiento y validación](docs/ETAPA4.md)
- [Resultados de validación](results/stage4/RESULTADOS.md)
- [Histórico de trabajo](HISTORICO.md)
