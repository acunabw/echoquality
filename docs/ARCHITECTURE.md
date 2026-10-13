# Arquitectura de EchoQuality 2.0

## Separación de responsabilidades

```text
processing   → modifica píxeles
analysis     → transforma imagen/máscara en mediciones
recognition  → aprende patrones y toma decisiones
optimization → selecciona la transformación bajo restricciones
reporting    → conserva trazabilidad y resultados
experiments  → orquesta las etapas
```

## Dos ramas de imagen

La rama de segmentación y la rama de medición textural deben permanecer diferenciadas:

```text
Rama A: imagen → realce opcional → segmentación → máscara
Rama B: imagen controlada/original + máscara de A → textura y radiomics
```

Una transformación diseñada para localizar contornos no se considera neutra para textura.

## Nivel global

Pregunta: **¿La adquisición general es adecuada?**

Incluye brillo, distribución de intensidad, rangos robustos de percentiles, contraste local, nitidez, bordes, ruido, speckle global, dropout, saturación y soporte.

`global__robust_intensity_range_p95_p05` se define como `p95-p05`. Es un descriptor robusto del histograma, no una medición física del rango dinámico del sistema.

## Nivel anatómico

Pregunta: **¿LV, MYO y LA están representados de forma utilizable?**

Incluye área, perímetro, circularidad, compactness, aspecto, eccentricity, solidity, componentes, contacto con borde, grosor aproximado, intensidad por ROI y evidencia de borde. Cuando NIfTI aporta spacing, se añaden mediciones físicas aproximadas claramente etiquetadas.

Las formas esperadas dependen de estructura y fase; por ejemplo, no se impone a MYO el mismo rango de solidity que a LV.

## Nivel textural

Pregunta: **¿La ROI miocárdica conserva información suficiente y reproducible para radiomics?**

Incluye:

- primer orden dentro de MYO;
- GLCM enmascarada;
- LBP;
- GLRLM;
- GLSZM;
- GLDM;
- NGTDM.

Parámetros de cuantización, distancias, ángulos, simetría, método LBP y vecindarios quedan en YAML y en el catálogo de features.

## Fusión tardía

Cada nivel genera una probabilidad de `NO APTA`:

```text
score_global
score_anatomical
score_textural
```

El meta-clasificador aprende la decisión final a partir de esos tres scores. Logistic Regression y Random Forest se comparan con la misma partición de pacientes.

## Prevención de leakage

1. La partición externa es por `patient_id`.
2. La búsqueda de hiperparámetros ocurre dentro del entrenamiento externo.
3. `SelectKBest` vive dentro de cada pipeline.
4. Las listas de indicadores exploratorios nunca reducen el espacio antes de nested CV.
5. `quality`, `apt_target`, `not_apt`, rutas e identificadores no entran a `X`.

## Agregación temporal

La versión 1.x producía una media ED/ES con nombres sin operación. La versión 2.0 genera explícitamente:

```text
feature__mean
feature__abs_delta
```

También admite `__ed`, `__es`, `__min` y `__max`. De esta forma, una degradación presente en una sola fase no queda necesariamente cancelada.

## Good Reference

Cada nivel y vista se modela con:

- mediana;
- MAD escalado;
- percentiles 10–90;
- covarianza Ledoit–Wolf;
- umbral empírico de distancia de Mahalanobis.

Los modelos de referencia de frames se estratifican por vista/fase para la optimización; los de paciente-vista se usan para diagnóstico.

## Procesamiento condicionado

El diagnóstico determina la familia candidata:

| Diagnóstico | Acción candidata |
|---|---|
| brillo/contraste | gamma, stretch, CLAHE controlado |
| speckle/ruido | bilateral, mediana, difusión anisotrópica |
| borde débil | unsharp conservador |
| dropout severo | revisión/repetición, no “reconstrucción” |
| anatomía truncada | revisión/repetición |

## Optimización

```math
J = w_G J_G + w_M J_M + w_T J_T + w_S SSIM - penalizaciones
```

- `J_G`: reducción de distancia global al Good Reference;
- `J_M`: preservación anatómica;
- `J_T`: acercamiento textural menos deriva;
- restricciones: SSIM, edge F1, drift anatómico, drift textural y clipping.
