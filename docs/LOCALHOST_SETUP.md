# EchoQuality 2.0 Localhost - guia paso a paso

## 1. Objetivo

El localhost permite analizar un caso mediante un navegador sin exponer la aplicacion a Internet. La interfaz carga un experimento ya entrenado, extrae las representaciones Global, Anatomica y Textural, ejecuta la fusion tardia, compara con los Good References y, cuando corresponde, evalua procesamiento adaptativo.

> Prototipo academico. No es un dispositivo medico.

## 2. Que necesita el algoritmo completo

El modelo actual fue entrenado a nivel paciente-vista. Para reproducir la misma representacion se necesitan dos fases:

```text
ED + mascara ED
ES + mascara ES
```

Las mascaras CAMUS utilizan:

```text
1 = LV
2 = MYO
3 = LA
```

Una imagen sin mascara permite solamente un analisis parcial. Una sola fase permite comparacion con Good Reference de frame, pero no la decision final del Quality Gate entrenado con ED y ES.

## 3. Archivos agregados

```text
app/streamlit_app.py
src/echoquality/application/inference.py
src/echoquality/application/__init__.py
.streamlit/config.toml
scripts/13_validate_localhost.py
start_localhost.bat
start_localhost.ps1
docs/LOCALHOST_SETUP.md
```

## 4. Crear la rama de trabajo

En GitHub Desktop:

1. Abrir `echoquality-v2`.
2. Seleccionar `Current branch`.
3. Elegir `New branch`.
4. Nombre recomendado: `feature/localhost-streamlit`.
5. Abrir `Repository -> Open in Visual Studio Code`.

## 5. Crear o actualizar el entorno Conda

Desde Anaconda Prompt, en la raiz del repositorio:

```bat
cd "C:\Users\TU_USUARIO\OneDrive - Boston Scientific\Documents\Maestria\Github desktop\echoquality-v2"
conda env create -f environment.yml
conda activate echoquality-v2
python -m pip install -e ".[app]"
```

Si el entorno ya existe:

```bat
conda activate echoquality-v2
conda env update -f environment.yml --prune
python -m pip install -e ".[app]"
```

Verificar:

```bat
python -c "import echoquality; print(echoquality.__version__)"
python -m streamlit version
pytest
```

## 6. Seleccionar el interprete en VS Code

1. Abrir cualquier archivo `.py`.
2. Presionar `Ctrl+Shift+P`.
3. Ejecutar `Python: Select Interpreter`.
4. Seleccionar el interprete del entorno `echoquality-v2`.
5. Confirmar en la barra inferior de VS Code.

## 7. Configurar CAMUS

Anaconda Prompt:

```bat
set CAMUS_ROOT=C:\Users\acunabw\CAMUS_Texture_Project\data\raw\CAMUS\database_nifti
```

PowerShell:

```powershell
$env:CAMUS_ROOT = "C:\Users\acunabw\CAMUS_Texture_Project\data\raw\CAMUS\database_nifti"
```

El dataset permanece fuera de GitHub.

## 8. Generar artefactos de prueba

Primero ejecutar el smoke test sintetico:

```bat
python run_experiment.py --config configs\experiments\smoke.yaml --force --fast
```

Validar un caso completo:

```bat
python scripts\13_validate_localhost.py ^
  --experiment-root results\smoke ^
  --config configs\experiments\smoke.yaml ^
  --processing
```

## 9. Generar artefactos CAMUS

Prueba inicial:

```bat
python run_experiment.py --config configs\experiments\camus10.yaml --fast
```

Ejecucion cientifica sin modo fast:

```bat
python run_experiment.py --config configs\experiments\camus10.yaml --force
```

Para usar CAMUS500 en la interfaz:

```bat
python run_experiment.py --config configs\experiments\camus500.yaml
```

La app espera, dentro de `results/<experimento>`:

```text
quality_gate/hierarchical_quality_gate.joblib
references/good_reference_view.joblib
references/good_reference_frame.joblib
manifest.csv
reports/resolved_config.yaml
```

## 10. Iniciar el localhost

```bat
python run_streamlit.py
```

O:

```bat
start_localhost.bat
```

Abrir:

```text
http://127.0.0.1:8501
```

Para detenerlo, regresar a Anaconda Prompt y presionar `Ctrl+C`.

## 11. Configurar el runtime en la interfaz

En la barra lateral:

```text
Carpeta del experimento entrenado: results/camus10
YAML: configs/experiments/camus10.yaml
```

Luego presionar `Recargar modelos y referencias`.

## 12. Modos de analisis

### A. Seleccionar desde CAMUS

- Seleccionar paciente y vista.
- La app carga ED, ES y sus mascaras desde el manifest.
- Opcionalmente reemplazar una fase con una imagen cargada.
- Si la imagen de reemplazo es diferente, debe cargarse su mascara correspondiente.
- La mascara CAMUS original solo puede reutilizarse cuando la imagen es una transformacion geometricamente alineada del mismo frame.

### B. Cargar ED y ES

- Cargar imagen ED y mascara ED.
- Cargar imagen ES y mascara ES.
- Seleccionar 2CH o 4CH.
- Ejecutar el algoritmo completo.

### C. Una sola imagen

- Cargar una imagen y, preferiblemente, su mascara.
- Seleccionar vista y fase.
- La app calcula representaciones y distancias a Good Reference.
- No emite el Quality Gate final por falta de la otra fase.

## 13. Salidas

La app muestra:

```text
P(NO APTA)
Score Global
Score Anatomico
Score Textural
estado operacional
nivel dominante
problema dominante
distancias Mahalanobis
desviaciones robustas
familias de procesamiento
candidatos de transformacion
imagen recomendada
```

Los resultados pueden descargarse como ZIP o guardarse en:

```text
results/localhost_cases/
```

## 14. Ejecutar el orquestador desde la interfaz

En la pestana `Orquestador`:

1. Seleccionar el YAML.
2. Seleccionar etapas.
3. Usar `Fast` para pruebas de ingenieria.
4. Mantener `Reanudar` para omitir etapas completadas.
5. Usar `Forzar` solamente cuando se desea repetir.
6. Presionar `Ejecutar orquestador`.

## 15. Flujo GitHub Desktop

Despues de probar:

1. Revisar `Changes`.
2. Confirmar que no se incluyen CAMUS, resultados ni modelos.
3. Ejecutar `pytest` y el smoke test.
4. Commit sugerido:

```text
Add local Streamlit inference application
```

5. Description sugerida:

```text
- Add CAMUS and uploaded-case analysis modes
- Add hierarchical inference and Good Reference diagnostics
- Add adaptive-processing visualization and downloads
- Add YAML orchestrator control
- Add localhost setup and validation documentation
```

6. `Commit to feature/localhost-streamlit`.
7. `Push origin`.
8. Crear Pull Request o integrar a `main` despues de revisar.

## 16. Problemas frecuentes

### No aparecen modelos

Ejecutar el experimento hasta `gate`, `references` y `diagnostics`.

### El navegador no abre

Abrir manualmente `http://127.0.0.1:8501`.

### Puerto ocupado

```bat
python -m streamlit run app\streamlit_app.py --server.port 8502
```

### La mascara no coincide

Imagen y mascara deben tener exactamente la misma forma.

### Faltan ED o ES

El Quality Gate completo requiere ambas fases. Usar el modo de una sola imagen solo como analisis exploratorio.

### NIfTI no abre

```bat
python -m pip install nibabel
```

### Streamlit no esta instalado

```bat
python -m pip install -e ".[app]"
```

## 17. Seguridad y privacidad

La configuracion enlaza el servidor a `127.0.0.1`, por lo que no queda disponible para otros equipos de la red. No se debe cambiar a `0.0.0.0` sin revision institucional de seguridad. Los archivos cargados se materializan temporalmente para leer NIfTI/NPY y se eliminan al finalizar el analisis; los resultados solo se guardan cuando el usuario lo solicita.
