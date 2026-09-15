# EchoQuality

> **A Two-Stage Pattern Recognition Framework for Echocardiographic Image Quality Assessment and Adaptive Image Processing**

## Objetivo

EchoQuality es un proyecto de investigación orientado a la evaluación automática de la calidad de imágenes ecocardiográficas y a la recomendación de estrategias de procesamiento adaptativo.

El pipeline tiene tres objetivos principales:

1. Determinar si una imagen supera o no el **Quality Gate** para análisis computacional.
2. Para las imágenes aceptadas, identificar qué indicadores se desvían del patrón de referencia **Good**.
3. Recomendar y evaluar filtros, transformaciones y parámetros de procesamiento para acercar la imagen al comportamiento Good sin deteriorar la morfología cardíaca ni la textura miocárdica.

> Prototipo académico. No es un dispositivo médico y no debe utilizarse para decisiones clínicas.

---

## Arquitectura científica

```text
Imagen ecocardiográfica original

        ↓

Extracción de características
Intensidad · Textura · Nitidez · Ruido · Morfología

        ↓

ETAPA 1
Quality Gate binario

        ↓

APTA / NO APTA

        ↓

Para imágenes aceptadas:
estado operativo APTA o APTA CONDICIONADA

        ↓

ETAPA 2
Good Reference

        ↓

Análisis de desviaciones

        ↓

Diagnóstico de indicadores

        ↓

Procesamiento adaptativo

        ↓

Evaluación multiobjetivo

        ↓

Imagen recomendada

        ↓

Radiomics / análisis downstream
```

---

## Política de calidad

CAMUS proporciona la variable `ImageQuality` en los archivos:

```text
Info_2CH.cfg
Info_4CH.cfg
```

con tres categorías:

```text
Good
Medium
Poor
```

La política operativa actual de EchoQuality es:

| CAMUS | Estado operativo |
|---|---|
| Good | APTA |
| Medium | APTA CONDICIONADA |
| Poor | NO APTA |

El modelo de la Etapa 1 es actualmente binario. Se utiliza `Poor = 1` como clase positiva para priorizar la detección de imágenes que no deberían continuar hacia análisis posteriores.

`APTA CONDICIONADA` no representa una tercera clase aprendida por el clasificador, sino un estado operativo asignado a imágenes aceptadas que presentan desviaciones respecto al patrón Good y que pueden ser candidatas a procesamiento controlado.

---

## Estructura del repositorio

```text
echoquality/
│
├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── results/
├── scripts/
├── src/
│   └── echo_quality_pipeline/
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
├── pipeline_config.yaml
├── pyproject.toml
├── requirements.txt
├── run_streamlit.py
├── streamlit_app.py
├── test_pipeline.py
└── tk_demo.py
```

---

## Instalación

El proyecto se utiliza actualmente con Conda.

Activar el entorno:

```powershell
conda activate pattern-recognition
```

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

Instalar EchoQuality en modo editable:

```powershell
python -m pip install -e .
```

Verificar la instalación:

```powershell
python -c "import echo_quality_pipeline; print(echo_quality_pipeline.__file__)"
```

---

## Demo reproducible

El proyecto incluye un flujo de demostración con imágenes sintéticas.

```powershell
python scripts\09_run_all_demo.py
```

Interfaces disponibles:

```powershell
python tk_demo.py
```

o:

```powershell
python run_streamlit.py
```

Los datos sintéticos sirven únicamente para validar el funcionamiento técnico del pipeline y no deben interpretarse como resultados científicos de CAMUS.

---

## Ejecución con CAMUS

### 1. Construir el manifiesto

```powershell
python scripts\01_build_camus_manifest.py --camus-root "C:\Ruta\CAMUS\database_nifti" --output data\camus_manifest_all.csv
```

El manifiesto funciona como índice estructurado del dataset y contiene información como:

```text
patient_id
view
phase
quality
apt_target
image_path
mask_path
cfg_path
```

---

### 2. Extraer características

```powershell
python scripts\02_extract_features.py --manifest data\camus_manifest_all.csv
```

El script extrae indicadores relacionados con:

```text
intensidad
textura
nitidez
ruido
morfología
```

---

### 3. Analizar indicadores

```powershell
python scripts\03_analyze_indicators.py
```

---

### 4. Entrenar el Quality Gate

```powershell
python scripts\04_train_quality_gate.py
```

---

### 5. Construir el Good Reference

```powershell
python scripts\05_build_good_reference.py
```

---

### 6. Optimizar el procesamiento

```powershell
python scripts\06_optimize_processing.py
```

---

### 7. Evaluar el pipeline completo

```powershell
python scripts\07_evaluate_end_to_end.py
```

---

## Etapa 1: Quality Gate

La primera etapa utiliza reconocimiento de patrones para determinar si una imagen es apta para continuar hacia análisis computacional.

Principios metodológicos:

- división de datos a nivel de paciente;
- evitar que imágenes del mismo paciente aparezcan simultáneamente en entrenamiento y prueba;
- agregación ED/ES cuando corresponda;
- selección de características dentro del proceso de validación;
- evaluación con métricas como ROC AUC, Balanced Accuracy, sensibilidad y especificidad.

Los modelos considerados incluyen:

```text
Logistic Regression
Random Forest
```

La salida principal es una probabilidad de `NO APTA` y una decisión asociada a un umbral operativo.

---

## Etapa 2: Good Reference y procesamiento adaptativo

El patrón Good se construye a partir de imágenes clasificadas como `Good` dentro del conjunto de entrenamiento.

La referencia puede incluir:

- mediana;
- MAD;
- intervalos robustos;
- covarianza regularizada;
- distribución de indicadores seleccionados.

Para cada imagen aceptada se calculan desviaciones respecto al patrón Good.

Entre los análisis posibles se incluyen:

```text
z robusto
distancia de Mahalanobis
desviaciones por indicador
problemas dominantes
```

Los problemas pueden agruparse en categorías como:

```text
contraste
ruido / speckle
nitidez
rango dinámico
dropout
morfología
```

---

## Procesamiento adaptativo

Según el diagnóstico de calidad, EchoQuality puede evaluar transformaciones como:

- CLAHE;
- corrección gamma;
- estiramiento de histograma;
- filtro de mediana;
- filtro bilateral;
- difusión anisotrópica;
- unsharp masking.

La selección del procesamiento se plantea como un problema multiobjetivo.

La mejora debe acercar los indicadores al patrón Good sin degradar:

- morfología;
- bordes;
- textura;
- similitud estructural.

---

## Interpretación de Dice

El demo puede utilizar un segmentador proxy para ilustrar la medición de Dice antes y después del procesamiento.

En la investigación de tesis, Dice debe calcularse utilizando una red de segmentación validada.

Dice representa una métrica de utilidad downstream y no constituye la etiqueta de calidad del Quality Gate.

---

## Resultados experimentales

Los resultados se organizan por tamaño del experimento:

```text
results/
├── camus10/
├── camus50/
├── camus100/
├── camus250/
└── camus500/
```

Esta organización permite evaluar la estabilidad del pipeline al aumentar progresivamente el número de pacientes.

---

## Salidas principales

Dependiendo del experimento, el pipeline puede generar:

```text
frame_features.csv
view_features.csv
indicator_deviations.csv
issue_summary.csv
candidate_scores.png
before_after.png
original.png
recommended.png
optimization_summary.json
```

Además de:

```text
modelos entrenados
Good Reference
métricas del Quality Gate
reportes de optimización
figuras
resultados end-to-end
```

---

## Buenas prácticas experimentales

- No dividir imágenes del mismo paciente entre entrenamiento y prueba.
- No seleccionar características utilizando todo el conjunto antes de validar.
- No ajustar hiperparámetros utilizando el holdout final.
- No optimizar filtros sobre el conjunto final de evaluación.
- No interpretar una imagen visualmente más suave como necesariamente mejor para radiomics.
- Preservar morfología y textura durante el procesamiento.
- No intentar reconstruir digitalmente anatomía ausente.
- No considerar el realce como solución para foreshortening o dropout severo.
- Mantener separados los datos sintéticos de los resultados científicos obtenidos con CAMUS.

---

## Dataset

La implementación actual utiliza el dataset público CAMUS como base para los experimentos.

Los datos originales no se almacenan en este repositorio.

GitHub contiene únicamente:

```text
código
configuración
scripts
documentación
estructura experimental
```

---

## Estado del proyecto

Actualmente EchoQuality dispone de:

- estructura Python basada en `src/`;
- instalación mediante `pyproject.toml`;
- integración con CAMUS;
- lectura de imágenes NIfTI;
- extracción de características;
- análisis de indicadores;
- Quality Gate;
- Good Reference;
- procesamiento adaptativo;
- evaluación end-to-end;
- estructura para experimentos CAMUS10, CAMUS50, CAMUS100, CAMUS250 y CAMUS500.

---

## Licencia

MIT License