# Plan de ejecución — BayesNet 2.0

Fecha: 2026-06-12
Documento de referencia: `diagnostico_previo_2_0.md`

Principio rector: **cada fase deja la biblioteca compilando, con todos los
tests en verde y con resultados numéricamente idénticos a la 1.2.3** (salvo
donde se indique lo contrario de forma explícita). Cada fase se desarrolla en
su propia rama y se integra con un PR independiente.

## Decisiones de diseño previas (a confirmar antes de la Fase 1)

| # | Decisión | Recomendación |
|---|----------|---------------|
| D1 | Convención de nombres de la API 2.0 | snake_case (coherente con `fit`, `predict_proba`, estilo scikit-learn) manteniendo alias deprecados de los nombres camelCase |
| D2 | Estándar de C++ | Subir a C++20 (designated initializers para `FitOptions`, `std::span`, conceptos para los learners del boosting). Si algún consumidor lo impide, quedarse en C++17 con builder pattern |
| D3 | Política de compatibilidad | Mantener las firmas v1 como adaptadores `[[deprecated]]` durante toda la serie 2.x; eliminarlas en 3.0 |
| D4 | Alcance de `Smoothing_t` | Pasa a hiperparámetro (`"smoothing": "ORIGINAL"|"LAPLACE"|"CESTNIK"|"NONE"`); el parámetro de `fit` v1 sigue funcionando vía adaptador |

---

## Fase 0 — Red de seguridad (sin tocar código de producción)

Objetivo: poder afirmar con evidencia que los refactors no cambian el
comportamiento.

1. **Inventario de cobertura actual**: revisar qué comprueban hoy los tests
   por categoría (scores esperados, nº de modelos, notas, grafos) y detectar
   huecos — en particular `predict_proba`, `dump_cpt`, `graph`, los
   hiperparámetros de Boost (`block_update`, `alpha_block`, `weightless`,
   `select_features` con CFS/IWSS/FCBF) y las rutas Ld.
2. **Tests golden de regresión numérica**: para cada modelo y cada dataset de
   `tests/data`, fijar como referencia:
   - `score` en train (y en un split fijo).
   - Primeras N filas de `predict_proba` con tolerancia 1e-6.
   - `getNumberOfNodes/Edges/States`, `getNotes`, nº de modelos del ensemble.
   Generar los valores de referencia con la 1.2.3 y guardarlos como datos del
   test (JSON en `tests/data/golden/`).
3. **Test de memoria**: una pasada de la suite con sanitizers (ASan/UBSan)
   como target opcional del Makefile (`make test-asan`), para cazar la fuga
   del feature selector y validar los refactors de ownership.
4. **CI**: asegurar que la suite completa (incluidos los golden) corre en
   cada PR de las fases siguientes.

Criterio de salida: suite golden en verde sobre `main` (1.2.3), documentada
en `tests/README.md`.

Tamaño estimado: pequeño-medio. Riesgo: bajo.

---

## Fase 1 — Nueva API pública (P1 + P6), sin tocar internals

Objetivo: que un usuario nuevo pueda usar la biblioteca con 3 líneas, sin
romper a ningún usuario actual.

1. **`bayesnet::Dataset`** (nuevo, en `bayesnet/data/`):
   - Construcción desde `torch::Tensor` (X, y), `std::vector<std::vector<int>>`,
     y desde el tensor combinado (n+1)×m actual.
   - Inferencia automática de `states` a partir de los datos; posibilidad de
     pasarlos explícitos (caso de estados no observados en train).
   - Acceso: `features()`, `class_name()`, `states()`, `X()`, `y()`,
     `n_samples()`, `n_features()`; nombres de features autogenerados
     (`f0..fn`) si no se proporcionan.
   - Soporte de datos continuos (para los modelos con discretización local):
     el `Dataset` sabe si cada columna es numérica o ya discreta (sustituye a
     la convención de "estado vacío" actual).
   - Helper de carga ARFF en `sample/` o como utilidad opcional (sin añadir
     dependencia obligatoria de arff-files a la biblioteca).
2. **`fit` nuevo**: `fit(const Dataset&)` y `fit(const Dataset&, const FitOptions&)`
   (weights; smoothing mientras dure la transición). Implementación inicial:
   *adaptador interno hacia el flujo actual* — no se toca `build()` todavía.
3. **`smoothing` como hiperparámetro** (D4) en `Classifier::setHyperparameters`.
4. **Deprecación**: marcar las 4 sobrecargas v1 de `fit` con `[[deprecated]]`
   (y documentar la equivalencia en la guía de migración).
5. **Factoría**: `bayesnet::ModelFactory::create("TANLd")` con registro de
   todos los modelos y `available_models()`. Reescribir `sample.cc` sobre la
   factoría y el `Dataset` (pasa a ser la demo de la API 2.0).
6. **Higiene (P6)**:
   - Excepciones propias en `bayesnet/Exceptions.h`
     (`not_fitted_error`, `invalid_data`, `invalid_hyperparameter`),
     heredando de las std actuales para no romper a quien capture
     `std::exception`.
   - `featureSelection` con `std::unique_ptr` (cierra la fuga).
   - `update_weights` devuelve valores, no `torch::Tensor&` en tuple.
   - Eliminar `XBAODE::getVersion()` fantasma.
   - `Classifier(Network)` por const ref / move.
   - Pasada de const-correctness en `predict/predict_proba/score`.

Criterio de salida: golden tests idénticos; `sample.cc` funciona solo con la
API nueva; compilación con `-Wdeprecated-declarations` muestra avisos solo en
los tests v1 que se conservan a propósito.

Tamaño estimado: medio. Riesgo: bajo (la API nueva es aditiva).

---

## Fase 2 — Pipeline de entrenamiento y jerarquía (P2 + P5)

Objetivo: que el flujo de entrenamiento sea legible y que cada clase use solo
lo que necesita. Es la fase más delicada.

1. **`FitContext` inmutable**: empaquetar dataset (n+1)×m, features,
   className, states, weights y smoothing. Sustituye el quinteto de miembros
   que hoy se asignan por mutación dispersa.
2. **Pipeline explícito en `Classifier::fit`**:
   ```
   validate(ctx) → build_structure(ctx) [virtual] → estimate_parameters(ctx) → finalize(ctx)
   ```
   - `buildModel/trainModel` actuales pasan a ser `build_structure` /
     `estimate_parameters` (alias protegidos deprecados mientras existan
     subclases externas).
   - El estado del clasificador tras `fit` queda confinado a: el modelo
     entrenado + un `FitResult { notes, status }` consultable.
3. **Boost sin efectos colaterales**: el split train/validación de
   `Boost::buildModel` deja de reescribir `dataset`; el contexto ofrece
   `train_view()` / `validation_view()`. Mismo fold, misma semilla (271),
   resultados idénticos.
4. **`BaseClassifier` mínimo (P5)**:
   - Núcleo: `fit / predict / predict_proba / score / setHyperparameters /
     getStatus / getNotes / getVersion`.
   - Nueva interfaz `GraphModel` (`graph`, `show`, `topological_order`,
     `dump_cpt`, contadores de red) implementada por los modelos basados en
     `Network`. Las variantes X dejan de implementar métodos vacíos.
   - Los métodos retirados del base se mantienen en los modelos concretos,
     de modo que el código que usa tipos concretos no se rompe; el código que
     usa `BaseClassifier*` para introspección migra con `dynamic_cast<GraphModel*>`
     (documentado en la guía de migración).
5. **`Ensemble` por composición**: contiene `vector<unique_ptr<BaseClassifier>>`
   y `significances`; implementa `BaseClassifier` directamente en vez de
   heredar de `Classifier` con una `Network` muerta. `XBAODE` deja de
   necesitar el `fitted = true` prematuro: el ensemble puede predecir con los
   modelos que tenga durante el entrenamiento (método interno
   `predict_partial`).

Criterio de salida: golden tests idénticos (este es el control crítico de la
fase); ASan limpio; ninguna subclase asigna miembros heredados fuera del
pipeline.

Tamaño estimado: grande. Riesgo: medio-alto → mitigación: PRs parciales (2.1
pipeline, 2.2 jerarquía, 2.3 ensemble) cada uno con los golden en verde.

---

## Fase 3 — Discretización como decorador y motor único de boosting (P3 + P4)

Objetivo: eliminar las dos fuentes principales de duplicación.

1. **`DiscretizingClassifier`** (P3):
   - Wrapper que posee un `unique_ptr<Classifier>` y los discretizadores
     fimdlp; en `fit` ejecuta la discretización local iterativa (lógica
     migrada desde `Proposal`), en `predict/predict_proba` transforma la
     entrada.
   - `TANLd`, `KDBLd`, `SPODELd`, `AODELd` pasan a ser alias/factorías finas
     (`DiscretizingClassifier(make_unique<TAN>())`) — los nombres públicos y
     su comportamiento se conservan.
   - Borrar `Proposal` y la herencia múltiple una vez migrados los 4 modelos.
   - Los hiperparámetros Ld (`ld_algorithm`, `mdlp_*`, ...) se reenvían al
     wrapper; el resto al clasificador interno.
2. **Motor de boosting único** (P4):
   - Extraer a `Boost::trainModel` el bucle AdaBoost completo (ranking
     SelectKBestWeighted, orden asc/desc/rand con semilla 173, bisección,
     `block_update`, `alpha_block`, `weightless`, convergencia con tolerancia
     y poda final), parametrizado por una **factoría de learners**:
     `std::function<std::unique_ptr<BaseClassifier>(const std::vector<int>& parents)>`.
   - `BoostAODE` = factoría SPODE(feature); `XBAODE` = factoría XSpode(feature);
     `BoostA2DE`/`XBA2DE` = factorías de pares (SPnDE/XSp2de). Cada clase
     queda reducida a su factoría + hiperparámetros propios.
   - Atención a las divergencias actuales entre las 4 copias: inventariarlas
     antes (diff dirigido) y decidir para cada una si es un bug en una copia
     (unificar) o comportamiento intencionado (parametrizar). Documentar cada
     caso en el PR.
3. **Revisión de `XSpode`/`XSp2de`**: con la interfaz mínima de la Fase 2 ya
   no necesitan heredar maquinaria de `Network`; implementan `BaseClassifier`
   directamente y desaparecen `fitx` y los overrides triviales.

Criterio de salida: golden tests idénticos para los 8 modelos afectados
(4 Ld + 4 boosting); `cloc` de `ensembles/` y `classifiers/*Ld*` reducido de
forma sustancial; ningún algoritmo duplicado.

Tamaño estimado: grande. Riesgo: medio (mitigado por los golden y por hacer
P3 y P4 en PRs separados).

---

## Fase 4 — Limpieza, documentación y publicación

1. Pasada final de naming (D1) con alias deprecados donde aplique.
2. Eliminar código muerto (comentarios VLOG, debug comentado, miembros sin
   uso) y unificar guards (`#pragma once` vs `#ifndef`).
3. **Documentación**:
   - Doxygen al día para la API 2.0 (`make doc`).
   - `docs/MIGRATION_2.0.md`: tabla v1 → v2 de cada firma deprecada, con
     ejemplos antes/después.
   - README y CLAUDE.md actualizados (ejemplos con `Dataset` + factoría).
   - Regenerar diagramas UML (`make diagrams`).
4. **Empaquetado**: bump a 2.0.0 en CMakeLists/vcpkg.json/conanfile.py;
   probar `make conan-create` y el test_package; CHANGELOG con el resumen de
   rupturas y deprecaciones.
5. Tag `v2.0.0` y publicación en los registros (vcpkg-stash y remote Conan).

Criterio de salida: paquete instalable por ambas vías; un proyecto consumidor
de prueba compila contra 2.0.0 usando solo la API nueva y, por separado,
usando la API v1 deprecada (con warnings).

Tamaño estimado: medio. Riesgo: bajo.

---

## Resumen de secuencia y dependencias

```
Fase 0 (golden tests)
  └→ Fase 1 (Dataset + factoría + higiene)   ← API aditiva, sin riesgo
       └→ Fase 2 (pipeline + jerarquía)       ← la más delicada, 3 PRs
            └→ Fase 3 (Ld decorador + boosting único)  ← 2 PRs
                 └→ Fase 4 (docs + release 2.0.0)
```

Reglas para todo el proceso:

- Ningún PR se integra con golden tests en rojo; si un cambio numérico es
  intencionado, se actualiza el golden en commit separado con justificación.
- Las firmas v1 no se eliminan en 2.x (D3); solo se marcan deprecadas.
- Cada fase actualiza la guía de migración en el mismo PR que introduce la
  deprecación.
