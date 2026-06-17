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

## Arquitectura de dos niveles

La suite tiene dos niveles con propósitos distintos. La razón es que las
diferencias de coma flotante de libtorch entre plataformas cambian los
**desempates** en los rankings por información mutua, y eso altera la selección
de features (FCBF/CFS) → nº de modelos, nodos, aristas, scores y `predict_proba`
de los ensembles de boosting. Los valores exactos, por tanto, **no son
portables** entre Linux y macOS (ver `analisis_tests_fallidos.md`).

1. **Suite cotidiana (`make test`) — portable.** Verifica contratos (notas,
   status, estructura), scores con **tolerancia** e invariantes/**rangos** para
   lo discreto. Debe quedar verde en cualquier máquina de cualquier
   contribuidor. Responde: *¿la biblioteca está funcionalmente sana?*
2. **Golden (`[Golden]`) — exacto y anclado al entorno.** Comportamiento
   bit-a-bit, regenerado en el **entorno de referencia (Linux)**. Responde:
   *¿mi refactor cambió el comportamiento respecto a antes?* No se sustituye por
   invariantes: ambos coexisten, cada uno en su papel.

### Convenciones del nivel 1 (portable)

Definidas en `tests/TestUtils.h`:

- **Scores**: `Catch::Approx(x).margin(PORTABLE_SCORE_MARGIN)` (0.08 absoluto,
  cubre la mayor divergencia Linux↔macOS observada ~0.067 con holgura).
- **Conteos** (`getNumberOfNodes/Edges/States`, `dump_cpt().size()`,
  `graph().size()`): rangos `>=`/`<=` que bracketean ambas plataformas.
- **Notas** con números (features/modelos/pares): `anyNoteContains(notes, frase)`
  casa la frase estable e ignora el conteo y el orden/tamaño de la lista.
- **`predict_proba`/voting con vuelco estructural**: solo se exige validez
  (`[0,1]`) y coherencia (`predict == argmax`); el valor exacto vive en golden.

**Coste honesto**: los invariantes detectan peor las regresiones sutiles (un
`score` de 0.804→0.812 no rompería un margen de 0.08). Por eso el nivel golden
sigue siendo exacto.

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

Los valores de referencia viven en `tests/data/golden/golden_<modelo>.json` y
están anclados al **entorno de referencia: Linux** (la generación es
determinista; dos ejecuciones producen ficheros idénticos byte a byte).
Cualquier comparación golden debe hacerse en ese entorno.

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
