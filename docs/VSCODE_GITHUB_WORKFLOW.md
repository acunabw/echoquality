# Flujo Visual Studio Code + GitHub Desktop

## 1. Crear la carpeta local

Descomprime el proyecto en:

```text
C:\Users\acunabw\OneDrive - Boston Scientific\Documents\Maestria\Github desktop\echoquality-v2
```

No lo mezcles con `echoquality` 1.x.

## 1.1 Compatibilidad Windows/macOS

El repositorio incluye `.gitattributes` y `.editorconfig` para normalizar UTF-8, finales de línea y sangría entre Windows y macOS. No deben eliminarse al crear el repositorio.

## 2. Abrir en Visual Studio Code

En GitHub Desktop:

```text
File → Add Local Repository
```

Selecciona `echoquality-v2`. Si indica que aún no es un repositorio Git, elige **Create a Repository** o ejecuta desde Anaconda Prompt:

```bat
cd "C:\...\Github desktop\echoquality-v2"
git init
git add .
git commit -m "Initialize EchoQuality 2.0 hierarchical architecture"
```

Luego en GitHub Desktop:

```text
Repository → Open in Visual Studio Code
```

## 3. Crear el entorno

```bat
conda env create -f environment.yml
conda activate echoquality-v2
python -m pip install -e .
pytest
```

## 4. Configurar CAMUS

```bat
set CAMUS_ROOT=C:\Users\acunabw\CAMUS_Texture_Project\data\raw\CAMUS\database_nifti
```

Comprueba:

```bat
python -m echoquality run --config configs\experiments\camus10.yaml --stages manifest
```

## 5. Ejecutar una prueba completa

```bat
python run_experiment.py --config configs\experiments\camus10.yaml --fast
```

Después ejecuta sin `--fast` para resultados científicos.

## 6. Secuencia de commits recomendada

Aunque el paquete se entrega completo, al integrarlo puedes usar esta secuencia:

1. `Initialize EchoQuality 2.0 project structure`
2. `Add hierarchical feature representations`
3. `Add late-fusion nested quality gate`
4. `Add Good References and hierarchical diagnostics`
5. `Add diagnosis-conditioned adaptive processing`
6. `Add experiment orchestrator, tests, and documentation`

## 7. Publicar

En GitHub Desktop:

```text
Commit to main → Publish repository
```

Nombre recomendado:

```text
echoquality-v2
```

Puedes mantenerlo privado durante el desarrollo y hacerlo público cuando hayas revisado rutas, datos y documentación.

## 8. Trabajo entre Windows y Mac

Windows:

```text
editar → ejecutar → commit → push
```

Mac:

```text
fetch → pull → abrir carpeta local existente
```

No vuelvas a clonar cada vez. En VS Code usa `File → Open Recent`.

## 9. No subir

- CAMUS;
- manifests con rutas locales;
- resultados completos;
- modelos `.joblib`;
- secretos o rutas personales.

El `.gitignore` ya cubre estos elementos.
