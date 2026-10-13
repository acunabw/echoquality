# Plan de experimentos para la tesis

## Principio general

Toda comparación usa partición por paciente, semillas registradas y selección/ajuste exclusivamente dentro del entrenamiento. Se reportan media, desviación e intervalos por bootstrap de paciente cuando corresponda.

## E0. Reproducción del baseline 1.x

Objetivo: fijar la referencia early-fusion.

- mismos 500 pacientes;
- mismas etiquetas;
- Logistic Regression y Random Forest;
- documentar que 1.x usó preselección global;
- no usar esas métricas como evaluación final de 2.0.

## E1. Curva de aprendizaje jerárquica

```text
10 → 50 → 100 → 250 → 500 pacientes
```

Usar un único orden aleatorio guardado, de modo que los subconjuntos sean anidados. Métricas: ROC AUC, PR AUC, balanced accuracy, sensibilidad, especificidad, Brier y estabilidad de features por nivel.

## E2. Early fusion vs late fusion

- early fusion con todas las features y selección dentro del fold;
- late fusion con tres scores;
- mismas particiones y modelos.

Hipótesis: late fusion mejora explicabilidad y puede mejorar generalización/calibración aunque no maximice AUC.

## E3. Ablación de niveles

Comparar:

```text
G
M
T
G+M
G+T
M+T
G+M+T
```

Esto responde cuánto aporta cada representación.

## E4. Agregación temporal

- media ED/ES;
- media + `abs_delta`;
- ED y ES explícitos;
- peor fase según score;
- modelos separados por fase.

Hipótesis: `abs_delta` recupera degradaciones ocultadas por la media.

## E5. Política de etiquetas

Sin cambiar los clasificadores:

1. Good+Medium = APTA; Poor = NO APTA.
2. Good = APTA; Medium+Poor = NO APTA.
3. Gate binario Poor vs resto + diagnóstico explícito de Medium.

Reportar el intercambio sensibilidad/especificidad y no escoger la política usando el test final.

## E6. Política de umbral

Evaluar sensibilidad mínima de NO APTA:

```text
0.85, 0.90, 0.95
```

Construir curvas de decisión y costo. La especificidad baja de 1.x probablemente está determinada tanto por la política de umbral como por el modelo.

## E7. Robustez de textura

Factorial controlado:

- niveles GLCM: 16, 32, 64;
- distancia: 1, 2, 3;
- direcciones: una vs cuatro;
- bilinear vs bicubic cuando exista resampling;
- ROI original, erosión 1–2 píxeles, dilatación 1–2 píxeles.

Métricas: ICC, CV, rank stability, AUC y variación del Good Reference.

## E8. Robustez anatómica y de máscara

Perturbar máscaras de manera controlada y medir:

- área/perímetro/circularity/thickness;
- boundary strength;
- deriva de textura;
- cambio de score anatómico y textural.

Esto cuantifica la propagación del error de segmentación.

## E9. Procesamiento condicionado

Tres brazos:

1. sin procesamiento;
2. transformación aplicada indiscriminadamente;
3. transformación condicionada al diagnóstico.

Evaluación en imagen, anatomía y tarea final:

- SSIM y edge F1;
- Dice/IoU/Hausdorff con segmentador validado;
- deriva de morfología;
- ICC/CV de textura;
- distancia a Good Reference;
- desempeño downstream.

## E10. Grillas de parámetros

- gamma: 0.7, 1.0, 1.3;
- CLAHE clip: 1, 2, 4, 8 o equivalentes normalizados;
- Gaussian sigma: 0, 0.5, 1, 1.5;
- mediana: 3×3, 5×5;
- unsharp: radio y amount conservadores;
- elemento morfológico: 0, 3×3, 5×5 sólo en la rama de segmentación.

La selección se hace en validación. Se busca una región de estabilidad, no un máximo visual aislado.

## E11. Good Reference

Comparar:

- referencia conjunta;
- por vista;
- por vista y fase;
- robusta mediana/MAD vs media/covarianza;
- estabilidad por bootstrap.

## E12. Calibración y utilidad de scores

Comparar Brier, curvas de calibración y scores por nivel. Analizar si el meta-modelo usa los tres niveles o si uno domina sistemáticamente.

## Resultados mínimos para la tesis

- tabla baseline 1.x vs 2.0;
- curva de aprendizaje;
- ablation por nivel;
- política de umbral;
- estabilidad de texturas;
- procesamiento condicionado vs controles;
- ejemplos explicables por nivel;
- análisis de fallos y casos que requieren repetición de adquisición.
