# Validación técnica del paquete

La reconstrucción base de EchoQuality 2.0 fue validada mediante compilación, pruebas unitarias y un experimento sintético end-to-end. La extensión localhost fue validada nuevamente después de incorporar la aplicación de inferencia.

## Estado actual

- 9 pruebas automatizadas superadas;
- ocho etapas del orquestador completadas sobre datos sintéticos;
- extracción de representaciones Global, Anatómica y Textural;
- análisis bootstrap por nivel;
- fusión tardía con nested patient-group cross-validation;
- Good References de vista y frame;
- diagnóstico jerárquico;
- procesamiento adaptativo condicionado;
- carga e inferencia de un caso ED/ES desde manifest;
- resultados descargables y persistencia opcional en disco.

La prueba sintética valida ingeniería, no desempeño científico ni uso clínico. La extracción CAMUS y la ejecución del servidor deben realizarse en la computadora que contiene los NIfTI originales y el entorno `echoquality-v2`.

Para los comandos y las limitaciones de la validación de la interfaz, consultar `docs/LOCALHOST_VALIDATION.md`.
