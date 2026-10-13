# Validación técnica del localhost

Fecha de validación del paquete: 2026-10-11.

## Comprobaciones ejecutadas

```text
PYTHONPATH=src python -m compileall -q src app scripts run_streamlit.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python run_experiment.py --config configs/experiments/smoke.yaml --force --fast
PYTHONPATH=src python scripts/13_validate_localhost.py --experiment-root results/smoke --config configs/experiments/smoke.yaml --processing
```

## Resultados

- compilación de módulos Python sin errores;
- 9 pruebas automatizadas superadas;
- ejecución completa de las ocho etapas del orquestador sobre datos sintéticos;
- generación de manifest, features jerárquicas, análisis de indicadores, Quality Gate, Good References, diagnóstico, procesamiento y reporte;
- carga de un caso ED/ES desde manifest;
- inferencia jerárquica Global/Anatómica/Textural;
- fusión tardía y decisión operacional;
- diagnóstico contra Good References;
- evaluación de candidatos de procesamiento adaptativo;
- serialización de resultados del caso.

## Alcance de esta validación

La prueba sintética confirma integración e ingeniería, pero no demuestra desempeño científico ni validez clínica. La aplicación debe probarse en la computadora de trabajo con los NIfTI CAMUS y con artefactos entrenados sobre el experimento elegido.

El servidor HTTP de Streamlit no se inició dentro del entorno de construcción porque la instalación del paquete Streamlit no estaba disponible allí. El módulo de la aplicación sí fue compilado y sus servicios de inferencia fueron probados directamente. En la computadora del usuario, ejecutar `python -m pip install -e ".[app]"` y luego `python run_streamlit.py` completa la prueba de interfaz.
