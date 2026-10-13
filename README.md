# EchoQuality 2.0

> **Arquitectura jerárquica de reconocimiento de patrones para evaluación de calidad ecocardiográfica y procesamiento adaptativo.**

EchoQuality 2.0 es una reconstrucción independiente del prototipo 1.x. Separa la representación de cada imagen en tres niveles —**global, anatómico y textural**—, aprende una evidencia de calidad en cada nivel y combina esas evidencias mediante **fusión tardía**.

> Prototipo académico. No es un dispositivo médico y no debe utilizarse para decisiones clínicas.

## Arquitectura

```text
                              Imagen
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      Representación       Representación       Representación
         global              anatómica             textural
             │                   │                   │
             ▼                   ▼                   ▼
        Clasificador G      Clasificador M      Clasificador T
             │                   │                   │
             └──────────────┬────┴────┬──────────────┘
                            ▼         
                    Fusión tardía / meta-modelo
                            │
                    APTA / NO APTA
                            │
                    Diagnóstico por nivel
                            │
                  Procesamiento condicionado
                            │
             J = J_G + J_M + J_T con restricciones
```

El proyecto conserva los métodos solicitados:

- Logistic Regression;
- Random Forest;
- bootstrap por paciente para análisis exploratorio;
- nested cross-validation agrupada por `patient_id`.

La diferencia principal es que la selección supervisada usada para estimar rendimiento se ajusta **dentro de cada fold**, evitando que el análisis exploratorio global contamine la validación.

## Cambios metodológicos principales

1. **Representaciones independientes:** `global__*`, `anatomical__*` y `textural__*`.
2. **Textura sólo en miocardio:** GLCM, GLRLM, GLSZM, GLDM, NGTDM y LBP usan la ROI MYO.
3. **GLCM enmascarada:** sólo se cuentan pares en los que ambos píxeles pertenecen a la ROI.
4. **Agregación temporal explícita:** `__mean` y `__abs_delta`; ya no se oculta que la antigua tabla por vista era la media ED/ES.
5. **Terminología corregida:** `robust_intensity_range_p95_p05` no se presenta como rango dinámico físico.
6. **ASM/Energy documentados:** `glcm_asm_uniformity = ΣP²`; `glcm_energy_sqrt_asm = √ASM`.
7. **Good Reference por nivel y vista:** mediana, MAD, percentiles y covarianza regularizada.
8. **Procesamiento condicionado:** no se aplica CLAHE, denoising o sharpening indiscriminadamente.
9. **Subconjuntos reproducibles y anidados:** orden CAMUS aleatorio con semilla y archivo exclusivo del dataset; los demos sintéticos usan archivos de orden separados.
10. **Artefactos por experimento:** ningún modelo de CAMUS10 sobrescribe el de CAMUS500.

## Estructura

```text
echoquality-v2/
├── app/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── demo.yaml
│       ├── camus10.yaml
│       ├── camus50.yaml
│       ├── camus100.yaml
│       ├── camus250.yaml
│       └── camus500.yaml
├── data/
├── docs/
├── notebooks/
├── results/
├── scripts/
├── src/echoquality/
│   ├── analysis/
│   ├── data/
│   ├── experiments/
│   ├── optimization/
│   ├── processing/
│   ├── recognition/
│   └── reporting/
├── tests/
├── environment.yml
├── pyproject.toml
├── requirements.txt
└── run_experiment.py
```

## Instalación en Windows con Conda

Desde Anaconda Prompt:

```bat
cd "C:\Users\TU_USUARIO\Documents\Github desktop\echoquality-v2"
conda env create -f environment.yml
conda activate echoquality-v2
python -m pip install -e .
```

Para conservar tu entorno existente también puedes usar:

```bat
conda activate pattern-recognition
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verificación:

```bat
python -c "import echoquality; print(echoquality.__version__)"
```

Para usar la interfaz Streamlit:

```bat
python -m pip install -e ".[app]"
python run_streamlit.py
```

## Configurar CAMUS

En Anaconda Prompt (`cmd.exe`):

```bat
set CAMUS_ROOT=C:\Users\acunabw\CAMUS_Texture_Project\data\raw\CAMUS\database_nifti
```

En PowerShell:

```powershell
$env:CAMUS_ROOT = "C:\Users\acunabw\CAMUS_Texture_Project\data\raw\CAMUS\database_nifti"
```

El dataset no se copia al repositorio.

## Ejecutar el pipeline completo

### Prueba sintética rápida

```bat
python run_experiment.py --config configs\experiments\demo.yaml --fast
```

### CAMUS10

```bat
python run_experiment.py --config configs\experiments\camus10.yaml
```

### CAMUS500

```bat
python run_experiment.py --config configs\experiments\camus500.yaml
```

El orquestador ejecuta:

```text
manifest
→ features jerárquicas
→ auditoría
→ indicadores exploratorios
→ Quality Gate jerárquico con nested CV
→ Good References
→ diagnóstico por nivel
→ procesamiento adaptativo
→ reporte
```

Los estados se guardan en `results/<experimento>/experiment_state.json`. Si la ejecución se interrumpe, vuelve a lanzar el mismo comando y continuará desde el último stage completado.

Para rehacer todo:

```bat
python run_experiment.py --config configs\experiments\camus100.yaml --force
```

Para ejecutar únicamente una etapa y sus dependencias:

```bat
python -m echoquality run --config configs\experiments\camus100.yaml --stages gate
```

## Salidas por experimento

```text
results/camus100/
├── manifest.csv
├── features/
│   ├── frame_features.csv
│   ├── view_features.csv
│   └── feature_catalog.csv
├── indicator_analysis/
│   ├── global/
│   ├── anatomical/
│   └── textural/
├── quality_gate/
│   ├── hierarchical_quality_gate.joblib
│   ├── model_summary.csv
│   ├── fold_metrics.csv
│   ├── oof_predictions.csv
│   ├── selected_features_by_fold.csv
│   ├── roc_quality_gate.png
│   ├── confusion_quality_gate.png
│   └── level_scores.png
├── references/
├── diagnostics/
├── adaptive_processing/
└── reports/
```

## Nota sobre indicadores exploratorios

Los CSV de `indicator_analysis` sirven para interpretar asociaciones, estabilidad y redundancia. **No se suministran como lista fija a la validación del clasificador.** Cada modelo selecciona sus variables dentro del entrenamiento de cada fold de nested CV.

## Comparar experimentos

```bat
python compare_experiments.py ^
  results\camus10\reports\experiment_summary.json ^
  results\camus50\reports\experiment_summary.json ^
  results\camus100\reports\experiment_summary.json ^
  results\camus250\reports\experiment_summary.json ^
  results\camus500\reports\experiment_summary.json ^
  --output results\learning_curve_summary.csv
```

## Documentación

- `docs/AUDIT_V1_CODE_AND_RESULTS.md`: auditoría del código y resultados adjuntos.
- `docs/IMPLEMENTATION_SUMMARY.md`: correspondencia entre requisitos y módulos implementados.
- `docs/ARCHITECTURE.md`: decisiones técnicas de la versión 2.0.
- `docs/FEATURE_PROTOCOL.md`: definiciones, parámetros y nomenclatura.
- `docs/EXPERIMENT_PLAN.md`: experimentos propuestos para la tesis.
- `docs/VSCODE_GITHUB_WORKFLOW.md`: flujo exacto de implementación y publicación.
- `docs/MIGRATION_V1_TO_V2.md`: cómo conservar 1.x y crear el repositorio nuevo.


## Localhost para analizar casos

La interfaz local usa Streamlit y se ejecuta solamente en:

```text
http://127.0.0.1:8501
```

El localhost separa dos procesos:

1. **Entrenamiento offline:** el orquestador genera el Quality Gate jerarquico y los Good References a partir de CAMUS.
2. **Inferencia local:** la app carga esos artefactos y analiza un caso sin reentrenar el modelo por cada imagen.

### Requisito para el algoritmo completo

El Quality Gate actual opera a nivel paciente-vista y fue entrenado con agregacion temporal `mean + abs_delta`. Por ello, el analisis completo necesita:

```text
ED + mascara ED
ES + mascara ES
```

Las mascaras deben seguir la codificacion CAMUS:

```text
1 = LV
2 = MYO
3 = LA
```

La app tambien ofrece un modo de una sola imagen. Ese modo extrae patrones y compara el frame con el Good Reference, pero no emite la decision final del Quality Gate porque falta la segunda fase.

### Preparar un runtime de demostracion

```bat
conda activate echoquality-v2
python run_experiment.py --config configs\experiments\smoke.yaml --force --fast
python scripts\13_validate_localhost.py --experiment-root results\smoke --config configs\experiments\smoke.yaml
```

### Iniciar el servidor local

```bat
python run_streamlit.py
```

Alternativamente:

```bat
start_localhost.bat
```

La interfaz permite:

- seleccionar un paciente-vista desde el manifest CAMUS;
- reemplazar una fase de CAMUS por una imagen cargada;
- cargar manualmente ED, ES y sus mascaras;
- ejecutar un analisis exploratorio de una sola imagen;
- visualizar Score G, Score M y Score T;
- obtener el estado APTA, APTA CONDICIONADA o NO APTA;
- revisar desviaciones frente al Good Reference;
- evaluar procesamiento adaptativo condicionado al diagnostico;
- descargar y guardar el reporte de cada caso;
- ejecutar el orquestador desde un YAML.

Consulta `docs/LOCALHOST_SETUP.md` y `docs/LOCALHOST_THEORY.pdf` para el procedimiento completo y la defensa teorica.

## Licencia

MIT.
