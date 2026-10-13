# EchoQuality 2.0 — Resumen de implementación

## Entrega

Este repositorio es un proyecto nuevo e independiente de EchoQuality 1.x. No reemplaza el baseline histórico; lo conserva como referencia para comparaciones metodológicas.

## Solicitud → implementación

| Solicitud | Implementación 2.0 |
|---|---|
| Tres niveles de representación | `analysis/global_features.py`, `morphology_features.py`, `border_features.py`, `texture_features.py` |
| Global, anatomía y textura por separado | prefijos `global__`, `anatomical__`, `textural__` y `HierarchicalFeatureSet` |
| Textura sólo en miocardio | ROI MYO, cuantización fija y matrices enmascaradas |
| Catálogo científico | 144 definiciones versionadas (23 globales, 82 anatómicas y 39 texturales en la configuración completa) |
| GLCM, GLRLM, GLSZM, GLDM, NGTDM y LBP | `analysis/radiomics_matrices.py` y `texture_features.py` |
| Tres clasificadores y fusión tardía | `recognition/hierarchical_gate.py` y `fusion.py` |
| Mantener Logistic Regression y Random Forest | ambos están disponibles por nivel y en el meta-modelo |
| Mantener bootstrap | análisis exploratorio por paciente en `recognition/associations.py` |
| Mantener nested CV | nested patient-group CV en `recognition/hierarchical_gate.py` |
| Evitar leakage | selección supervisada dentro del fold; targets/rutas/IDs excluidos |
| Diagnóstico por nivel | `recognition/diagnostics.py` |
| Good Reference por nivel | `recognition/good_reference.py` con mediana, MAD, cuantiles y Ledoit–Wolf |
| Procesamiento condicionado | `processing/adaptive_processing.py` |
| Objetivo `J_G + J_M + J_T` | `optimization/objective.py` y `filter_optimizer.py` |
| Orquestador | `experiments/orchestrator.py`, CLI y `run_experiment.py` |
| Ejecutar todo sin pasos manuales | `python run_experiment.py --config ...` |
| Configuración de experimentos | YAML para demo y CAMUS10/50/100/250/500 |
| Reanudación | `experiment_state.json`; etapas completadas se omiten |
| Interfaz jerárquica | `app/streamlit_app.py` muestra adecuación G/M/T, decisión y diagnóstico |
| Flujo VS Code + GitHub Desktop | `docs/VSCODE_GITHUB_WORKFLOW.md` |
| Propuesta experimental | `docs/EXPERIMENT_PLAN.md` |
| Auditoría del ZIP y resultados 1.x | `docs/AUDIT_V1_CODE_AND_RESULTS.md` |

## Correcciones derivadas de la auditoría

1. `view_features` ya no es una media ED/ES con nombre opaco. Se generan al menos `__mean` y `__abs_delta`; también se admiten ED, ES, mínimo y máximo.
2. El análisis exploratorio de indicadores no preselecciona el universo que entra a nested CV.
3. El fallback de estabilidad queda identificado con `fallback_used` y `selection_reason`.
4. `dynamic_range_p95_p05` se renombra `robust_intensity_range_p95_p05` para no confundirlo con rango dinámico físico.
5. GLCM usa `asm_uniformity` y `energy_sqrt_asm` para hacer explícita la relación algebraica.
6. Las matrices texturales se calculan dentro de MYO; en GLCM ambos píxeles del par deben pertenecer a la ROI.
7. Los subconjuntos CAMUS son aleatorios, reproducibles, anidados y usan un archivo de orden distinto al de los demos sintéticos.
8. Los artefactos quedan aislados en la carpeta de cada experimento.
9. La selección del número de folds cuenta pacientes representados en cada clase y no asigna una etiqueta arbitraria al paciente.
10. Se registran dominio de intensidad, soporte, cuantización, GLCM, LBP, conectividad, perímetro, CNR y agregación temporal en configuración/catálogo.

## Ejecución completa

```bat
conda activate echoquality-v2
set CAMUS_ROOT=C:\ruta\a\CAMUS\database_nifti
python run_experiment.py --config configs\experiments\camus10.yaml
```

El mismo comando con `camus50.yaml`, `camus100.yaml`, `camus250.yaml` o `camus500.yaml` ejecuta:

```text
manifest → features → indicators → gate → references → diagnostics → processing → report
```

## Validación realizada en esta entrega

- compilación de todos los módulos y scripts;
- ocho pruebas unitarias (configuración, manifiestos, agregación, features, gate, referencias y procesamiento);
- ejecución end-to-end sintética de ocho etapas con 18 pacientes, 72 frames y 36 paciente-vista;
- auditoría reproducida de CAMUS10/50/100/250/500 a partir de los CSV y artefactos adjuntos.

La extracción jerárquica CAMUS 2.0 completa no se ejecutó en el entorno de construcción porque el ZIP no contiene las imágenes NIfTI originales. Debe ejecutarse en la computadora Windows que tiene `database_nifti`.

## Estado científico

La versión 2.0 está diseñada para probar la hipótesis central de la tesis: la calidad ecocardiográfica puede representarse mediante evidencias globales, anatómicas y texturales independientes, combinadas por fusión tardía y conectadas a un procesamiento condicionado por diagnóstico.

Las implementaciones radiomics propias son auditables, pero no se declaran compatibles con IBSI/PyRadiomics hasta completar una validación numérica contra una implementación de referencia.
