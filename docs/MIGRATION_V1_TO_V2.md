# Migración de EchoQuality 1.x a 2.0

## Decisión recomendada

Conservar el repositorio `echoquality` 1.x como baseline histórico y publicar este proyecto como un repositorio nuevo:

```text
echoquality-v2
```

No reemplazar archivos dentro del repositorio 1.x. Esto mantiene reproducibles los resultados CAMUS10–500 originales y permite comparar early fusion contra late fusion.

## Qué se reutiliza

- lectura de CAMUS y política Good/Medium/Poor;
- Logistic Regression;
- Random Forest;
- bootstrap por paciente;
- nested group CV;
- conceptos de Good Reference;
- transformaciones y restricciones de preservación.

## Qué cambia

- paquete `echo_quality_pipeline` → `echoquality`;
- vector único → tres representaciones;
- preselección global → selección dentro del fold;
- media ED/ES implícita → agregadores nombrados;
- Good Reference único → nivel × vista (y vista/fase para frames);
- procesamiento general → procesamiento condicionado;
- scripts manuales → orquestador YAML reanudable.

## Compatibilidad de resultados

Las métricas 1.x no se mezclan en una misma tabla como si fueran obtenidas por el mismo método. Deben rotularse:

```text
EchoQuality 1.x baseline: early fusion + preselección global
EchoQuality 2.0: hierarchical late fusion + fold-local selection
```

La comparación válida exige usar el mismo orden de pacientes, las mismas etiquetas y las mismas particiones o semillas.
