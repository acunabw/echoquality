# Auditoría de EchoQuality 1.x: código y resultados CAMUS10–500

## Alcance

Esta auditoría se calculó directamente sobre el ZIP del repositorio y los CSV/artefactos de los cinco experimentos. Los resultados 1.x se preservan como **baseline histórico**, pero no deben presentarse como evaluación final de EchoQuality 2.0.

## Resultados reproducidos

| experiment   |   patients |   frames |   patient_views |   good |   medium |   poor |   reported_indicators |   indicators_meeting_config_threshold | silent_fallback_likely   | best_model         |   roc_auc |   balanced_accuracy |   sensitivity_not_apt |   specificity_apt |   brier |
|:-------------|-----------:|---------:|----------------:|-------:|---------:|-------:|----------------------:|--------------------------------------:|:-------------------------|:-------------------|----------:|--------------------:|----------------------:|------------------:|--------:|
| CAMUS10      |         10 |       40 |              20 |      8 |        6 |      6 |                    13 |                                     0 | True                     | RandomForest       |    0.8611 |              0.6944 |                1.0000 |            0.3889 |  0.1324 |
| CAMUS50      |         50 |      200 |             100 |     54 |       32 |     14 |                    12 |                                     0 | True                     | RandomForest       |    0.6558 |              0.5631 |                0.9333 |            0.1929 |  0.1452 |
| CAMUS100     |        100 |      400 |             200 |    106 |       66 |     28 |                     2 |                                     2 | False                    | LogisticRegression |    0.7127 |              0.5664 |                0.9200 |            0.2128 |  0.2260 |
| CAMUS250     |        250 |     1000 |             500 |    269 |      177 |     54 |                     9 |                                     9 | False                    | RandomForest       |    0.7399 |              0.6429 |                0.8909 |            0.3948 |  0.1204 |
| CAMUS500     |        500 |     2000 |            1000 |    505 |      379 |    116 |                     5 |                                     5 | False                    | RandomForest       |    0.6763 |              0.5751 |                0.9308 |            0.2194 |  0.1716 |

## Hallazgos de código de prioridad alta

1. **Preselección supervisada antes de la validación anidada.** `03_analyze_indicators.py` usa las etiquetas de todo el conjunto para producir `stable_nonredundant_indicators.csv`; luego `04_train_quality_gate.py` suministra esa lista a la validación anidada. Aunque `SelectKBest` se ajusta dentro de cada fold, el universo de candidatos ya fue filtrado con etiquetas externas al fold. Esto puede optimizar de forma sesgada la evaluación. En 2.0, el análisis de indicadores es exploratorio y la selección usada por el modelo ocurre únicamente dentro del entrenamiento de cada fold.
2. **Fallback silencioso.** Si ninguna feature alcanza frecuencia ≥0.65 y consistencia ≥0.80, el script usa automáticamente las primeras `top_k`. En CAMUS10 y CAMUS50 ninguna feature cumplió la regla configurada, pero se reportaron 13 y 12 “indicadores finales”. En 2.0 el fallback se etiqueta explícitamente y nunca se usa para medir rendimiento.
3. **Agregación ED/ES opaca.** `view_features.csv` es exactamente la media de ED y ES. Con dos fases, la mediana y la media coinciden; un defecto exclusivo de una fase puede quedar oculto. En 2.0 las columnas quedan nombradas como `__mean` y `__abs_delta`, y opcionalmente se conservan `__ed`/`__es`.
4. **Mezcla temprana de representaciones.** Variables globales, anatómicas y texturales se concatenan antes de aprender. 2.0 entrena tres modelos independientes y una fusión tardía.
5. **Textura fuera de la ROI y coocurrencias artificiales.** El diseño 1.x mezcla soporte global y ROI en varias texturas. 2.0 calcula GLCM/GLRLM/GLSZM/GLDM/NGTDM únicamente dentro del miocardio y cuenta pares GLCM sólo cuando ambos píxeles pertenecen a la ROI.
6. **Terminología de rango dinámico.** `dynamic_range_p95_p05` es un rango robusto de percentiles, no el rango dinámico físico del sistema. 2.0 lo renombra `robust_intensity_range_p95_p05` y conserva el nombre anterior únicamente como alias documental.
7. **Redundancia ASM/Energy.** En los CSV se cumple `glcm_energy²≈glcm_asm`. 2.0 usa nombres explícitos: `glcm_asm_uniformity` y `glcm_energy_sqrt_asm`.
8. **Subconjuntos por orden de archivo.** CAMUS10–250 se construyeron con los primeros N pacientes. 2.0 genera un orden aleatorio reproducible, guardado en disco, para que 10⊂50⊂100⊂250⊂500 sin depender del orden del dataset.
9. **Artefacto global sobrescrito.** Todos los experimentos podían escribir `artifacts/quality_gate.joblib`. 2.0 guarda artefactos dentro de cada experimento.
10. **Etiquetado de grupo mixto.** La función 1.x de número seguro de splits usaba `drop_duplicates(patient_id)`, asignando arbitrariamente una sola etiqueta a pacientes cuyas vistas pueden tener calidades distintas. 2.0 cuenta los grupos presentes en cada clase sin colapsar las vistas.

## Auditoría numérica de identidades

```json
{
  "view_features_max_abs_error_vs_ED_ES_mean": 1.4210854715202004e-14,
  "dynamic_range_identity_max_abs_error": 2.220446049250313e-16,
  "speckle_identity_max_abs_error": 4.216036819659763e-08,
  "mask_fraction_identity_max_abs_error": 2.7755575615628914e-16,
  "glcm_energy_squared_asm_max_abs_error": 7.974456136095043e-05
}
```

## Interpretación de los resultados 1.x

- CAMUS10 sobreestima el rendimiento por tamaño reducido y alta varianza.
- CAMUS50–500 muestran sensibilidad alta y especificidad baja bajo la política de umbral ≥0.90 de sensibilidad.
- La variación del número de “indicadores finales” (13, 12, 2, 9, 5) combina inestabilidad muestral, fallback y preselección global; no debe interpretarse aún como convergencia biológica.
- El valor principal de estos experimentos es establecer la línea base y revelar qué decisiones metodológicas deben aislarse en 2.0.

## Decisión de migración

Random Forest, Logistic Regression, bootstrap y nested patient-group cross-validation se conservan. Cambia la representación: global, anatómica y textural se aprenden de forma independiente y se fusionan mediante un meta-clasificador.
