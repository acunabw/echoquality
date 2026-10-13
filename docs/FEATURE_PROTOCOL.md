# Protocolo reproducible de features

## Reglas generales

Cada feature debe responder cinco preguntas:

1. ¿Qué operación matemática la produce?
2. ¿Sobre qué píxeles se calcula?
3. ¿Qué parámetros usa?
4. ¿Qué hace que aumente o disminuya?
5. ¿Qué artefactos o decisiones del pipeline pueden alterarla?

El archivo generado `feature_catalog.csv` registra nombre, nivel, familia, fórmula, soporte, unidades, fuente, parámetros, confusores y aliases.

## Sampling, cuantización e interpolación

- El tamaño de matriz no se interpreta como resolución física sin `mm/píxel`.
- La reducción a 8 bits no es obligatoria para mediciones; la conversión a `uint8` se restringe a LBP cuando su definición lo requiere.
- Las máscaras usan nearest-neighbor si se re-muestrean.
- La textura debe utilizar el mismo protocolo de interpolación en todos los casos.

## Histograma y entropía

- La entropía describe incertidumbre/diversidad estadística; no equivale a calidad clínica.
- El ruido puede aumentar entropía.
- Las estadísticas globales no contienen ubicación espacial y se complementan con anatomía y textura.

## GLCM

Configuración declarada en YAML:

```yaml
levels: 32
distances: [1, 2]
angles_degrees: [0, 45, 90, 135]
```

La GLCM se construye sólo con pares donde **origen y vecino** están dentro de MYO.

Nomenclatura:

```text
textural__glcm_asm_uniformity = Σ P(i,j)^2
textural__glcm_energy_sqrt_asm = sqrt(ASM)
```

No se interpretan como descriptores independientes sin reconocer su relación algebraica.

## Matrices radiomics

GLRLM, GLSZM, GLDM y NGTDM se implementan en `analysis/radiomics_matrices.py`. Son extensiones radiomics y no deben atribuirse directamente a Gonzalez & Woods. Los nombres y fórmulas exactas quedan versionados con el código.

Estas implementaciones son deliberadamente transparentes para investigación y auditoría, pero **no se declaran compatibles con IBSI ni intercambiables con PyRadiomics** hasta completar una validación numérica contra una implementación de referencia. Esa validación forma parte del plan experimental antes de interpretar estas variables como biomarcadores radiomics estandarizados.

## Geometría

- área en píxeles: número de píxeles de la región;
- perímetro: longitud digital del contorno según scikit-image;
- circularity: `4πA/P²`;
- compactness: `P²/(4πA)`;
- eccentricity: elipse de segundos momentos;
- solidity: `A/A_convex_hull`;
- thickness: dos veces la distancia al borde sobre skeleton; es una aproximación computacional.

Las features de forma se interpretan por estructura, vista y fase.

## Auditorías automáticas

La etapa de features comprueba:

```text
robust_range = p95 - p05
speckle_index = std / mean
mask_fraction = lv_fraction + myo_fraction + la_fraction
energy_sqrt_asm^2 ≈ asm_uniformity
view mean y abs_delta reproducen ED/ES
```

La ejecución se detiene si una identidad crítica falla.
