# Sistema de dos etapas para calidad ecocardiografica

## Objetivo

Desarrollar un pipeline que:

1. determine automaticamente si una imagen ecocardiografica es **APTA**, **APTA CONDICIONADA** o **NO APTA** para analisis computacional;
2. para las imagenes aptas, identifique los indicadores que se apartan del patron de referencia **Good**;
3. proponga y evalúe filtros, transformaciones y parametros —incluido el tamano de kernel— para acercar la imagen al comportamiento Good sin deteriorar morfologia ni textura.

> Prototipo academico. No es un dispositivo medico y no debe utilizarse para decisiones clinicas.

## Estructura cientifica

```text
Imagen original
  -> caracteristicas de intensidad, textura, nitidez, ruido y morfologia
  -> Etapa 1: SRP APTA / NO APTA
  -> patron Good aprendido solo con entrenamiento
  -> diagnostico de indicadores desviados
  -> candidatos de procesamiento
  -> optimizacion multiobjetivo
  -> imagen recomendada + trazabilidad
  -> ROI, morfologia, textura y radiomics de la tesis
```

## Estandar de referencia

CAMUS proporciona la variable `ImageQuality` en los archivos `Info_2CH.cfg` y `Info_4CH.cfg` con las categorias `Good`, `Medium` y `Poor`. La politica primaria del prototipo es:

- `Good`: APTA directa;
- `Medium`: APTA condicionada y candidata a mejora controlada;
- `Poor`: NO APTA; revisar o repetir adquisicion.

El modelo binario se entrena con `Poor=1` para priorizar la sensibilidad a imagenes que no deben pasar a radiomics.

## Instalacion en Windows / Visual Studio

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Demo completamente reproducible

```powershell
python scripts\09_run_all_demo.py
python app\tk_demo.py
```

El demo genera imagenes sinteticas tipo ultrasonido, entrena la compuerta, construye el patron Good y ejecuta la segunda etapa. Los resultados sinteticos sirven para mostrar el funcionamiento; no son resultados de CAMUS.

## Ejecucion con CAMUS

```powershell
python scripts\01_build_camus_manifest.py --camus-root C:\Datos\CAMUS
python scripts\02_extract_features.py --manifest data\camus_manifest.csv
python scripts\03_analyze_indicators.py
python scripts\04_train_quality_gate.py
python scripts\05_build_good_reference.py
python scripts\07_evaluate_end_to_end.py --manifest data\camus_manifest.csv
```

## Etapa 1: reconocimiento de patrones

- unidad de particion: paciente;
- unidad primaria de calidad: paciente-vista;
- ED y ES se agregan por mediana para no duplicar la etiqueta;
- normalizacion, seleccion de variables e hiperparametros se ajustan dentro de validacion;
- modelos: regresion logistica regularizada y Random Forest como sensibilidad;
- salida: probabilidad de `NO APTA`, umbral y estado operativo.

## Etapa 2: diagnostico y procesamiento adaptativo

El patron Good se representa mediante mediana, MAD, intervalos 10-90 % y covarianza regularizada. Para cada imagen apta se calculan:

- z robusto por indicador;
- distancia de Mahalanobis al patron Good;
- problemas dominantes: contraste, ruido/speckle, nitidez, rango dinamico, dropout o morfologia.

A partir del diagnostico se prueban candidatos como CLAHE, gamma, estiramiento de histograma, mediana, bilateral, difusion anisotropica y unsharp. La seleccion maximiza la cercania al patron Good y mejoras de contraste/ruido/Dice, pero exige limites de SSIM, retencion de bordes y deriva textural.

## Interpretacion de Dice

El demo incluye un segmentador proxy para ilustrar la medicion de Dice antes y despues. En el experimento de tesis debe conectarse la red de segmentacion validada. Dice es una metrica de utilidad downstream, no la etiqueta de calidad.

## Salidas principales

- `results/indicator_analysis/stable_nonredundant_indicators.csv`
- `results/stage1_quality_gate/model_summary.csv`
- `artifacts/quality_gate.joblib`
- `artifacts/good_reference.joblib`
- `results/stage2_processing/candidate_transformations.csv`
- `results/end_to_end/end_to_end_results.csv`
- `results/figures/pipeline_two_stage.png`

## Buenas practicas

- no dividir fases o vistas del mismo paciente entre entrenamiento y prueba;
- no seleccionar caracteristicas con todo el conjunto antes de validar;
- no optimizar filtros utilizando el holdout final;
- no interpretar una imagen visualmente mas suave como necesariamente mejor para textura;
- no intentar recuperar anatomia ausente, foreshortening o dropout severo mediante realce digital.
