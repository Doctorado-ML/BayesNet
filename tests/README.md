# Tests de BayesNet

Suite de tests con [Catch2](https://github.com/catchorg/Catch2). Ejecutable:
`TestBayesNet` en `build_Debug/tests/`.

```bash
make debug buildd   # configurar y compilar
make test           # ejecutar toda la suite
make test opt="-s"  # salida verbosa
build_Debug/tests/TestBayesNet "[Models]"   # una categoría concreta
```

Categorías: `[A2DE] [BoostA2DE] [BoostAODE] [XSPODE] [XSPnDE] [XBAODE]
[XBA2DE] [Classifier] [Ensemble] [FeatureSelection] [Metrics] [Models]
[Modules] [Network] [Node] [MST] [Golden]`.

## Golden tests (`[Golden]`, `TestGolden.cc`)

Red de seguridad de la refactorización 2.0 (ver `plan_2_0.md`, Fase 0). Para
cada modelo de la biblioteca se fija el comportamiento observable entrenando
con los datasets de referencia (iris, glass, ecoli, diabetes):

- `score` sobre el conjunto de entrenamiento.
- Primeras 20 predicciones (`predict`) y primeras 10 filas de
  `predict_proba` (tolerancia 1e-6).
- Contadores del grafo (`nodes`, `edges`, `states`, `class_states`).
- `notes` y `status` del entrenamiento.

Para los ensembles de boosting (BoostAODE, XBAODE, BoostA2DE, XBA2DE) se
fijan además 10 combinaciones de hiperparámetros (`select_features` con
CFS/IWSS/FCBF, `block_update`, `alpha_block`, `weightless`,
`convergence_best`, `bisection`, `order`).

Los valores de referencia viven en `tests/data/golden/golden_<modelo>.json`.

### Regenerar los golden

```bash
make golden
# equivalente a:
cd build_Debug/tests && GOLDEN_GENERATE=1 ./TestBayesNet "[Golden]"
```

**Solo deben regenerarse cuando un cambio de comportamiento es
intencionado**, en un commit separado que justifique el cambio. Ningún PR de
la serie 2.x debe integrarse con los golden en rojo.

## Sanitizers

```bash
make asan        # configura build_Asan con -fsanitize=address,undefined
make test-asan   # compila y ejecuta la suite con sanitizers
```

Útil para validar los refactors de ownership (Fase 1 y siguientes). En macOS
ASan puede reportar avisos conocidos procedentes de libtorch; evaluar cada
aviso antes de atribuirlo a la biblioteca.

## Datos

Los datasets ARFF están en `tests/data/` y se cargan a través de
`RawDatasets` (`TestUtils.h`), que discretiza con mdlp cuando
`discretize=true` y mantiene los valores continuos para los modelos Ld.
El catálogo `all.txt` indica qué features son numéricas en cada dataset.
