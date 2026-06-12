# Diagnóstico previo — BayesNet 2.0

Fecha: 2026-06-12
Versión analizada: 1.2.3 (rama `main`)

Este documento recoge el diagnóstico del estado actual de la biblioteca y las
propuestas de refactorización para la versión 2.0. El objetivo de la 2.0 no es
añadir clasificadores nuevos, sino simplificar la estructura interna y la
interfaz pública para que la biblioteca sea más ágil de usar y de extender en
la investigación futura.

## 1. Diagnóstico de la interfaz pública

### 1.1 El `fit` es el principal punto de fricción

`BaseClassifier` (bayesnet/BaseClassifier.h) expone 4 sobrecargas de `fit`,
cada una con 5–6 parámetros (`X, y, features, className, states, weights,
smoothing`). Problemas concretos:

- **El usuario debe construir `states` a mano**, con convenciones no obvias:
  en `sample/sample.cc` hay que crear entradas *vacías* por cada feature
  numérica para los modelos Ld y rellenar los estados de la clase con `iota`.
  Es conocimiento interno filtrado al exterior; los estados se pueden inferir
  de los datos en la inmensa mayoría de los casos.
- **Parámetros por referencia no-const** (`X&`, `y&`, `states&`) que obligan
  al llamador a mantener variables mutables y sugieren que `fit` puede
  modificar los datos (internamente `dataset` se comparte y se muta).
- **`smoothing` viaja en `fit`** cuando conceptualmente es un hiperparámetro:
  es el único parámetro de configuración que no entra por
  `setHyperparameters` ni por constructor.
- **Dos vías de configuración conviven**: parámetros de constructor
  (`KDB(k, theta)`, `SPODE(0)`) frente a `setHyperparameters(json)`.
- **No existe la llamada simple** `fit(X, y)` que es lo que se necesita el
  90% de las veces.

### 1.2 La interfaz base mezcla responsabilidades

`BaseClassifier` tiene más de 20 métodos virtuales puros que mezclan:

- Predicción: `predict`, `predict_proba`, `score`.
- Introspección del grafo: `graph`, `show`, `topological_order`, `dump_cpt`.
- Diagnóstico del entrenamiento: `getNotes`, `getStatus`.
- Contadores: `getNumberOfNodes`, `getNumberOfEdges`, `getNumberOfStates`,
  `getClassNumStates`.

Las variantes X implementan parte de esa interfaz vacía o con valores
triviales porque no les aplica. Además:

- Inconsistencia de naming: `predict_proba`, `dump_cpt`, `topological_order`
  (snake_case) junto a `getNumberOfStates`, `setHyperparameters` (camelCase).
- Falta de const-correctness: `predict(torch::Tensor& X)` debería tomar
  `const&` y ser un método `const`.
- `trainModel` (detalle de implementación) está declarado en la interfaz
  pública abstracta.

### 1.3 No hay factoría de modelos

Cada consumidor (el propio `sample.cc` y los proyectos de experimentación)
reimplementa su mapa `nombre → modelo`. Una factoría integrada
(`bayesnet::create("TANLd")`) eliminaría esa duplicación en todos los
proyectos dependientes.

## 2. Diagnóstico del flujo de entrenamiento (`Classifier`)

El patrón actual es un *template method* implícito y frágil:

```
fit(...) → buildDataset(y) → build(...) → checkFitParameters()
        → buildModel(weights) [virtual] → trainModel(weights, smoothing) [virtual] → fitted=true
```

Problemas observados:

- **Estado construido por mutación dispersa de miembros**: `dataset` primero
  es X, luego se le concatena y (`buildDataset`); `m`, `n`, `metrics` se
  asignan en `build`. Peor aún, `Boost::buildModel` (Boost.cc) **sustituye
  `dataset` por el fold de entrenamiento** como efecto colateral: el
  significado del miembro `dataset` cambia según el punto del flujo y la
  subclase.
- **Las variantes Ld se saltan el flujo entero**: `TANLd::fit` reimplementa
  el fit, llama a `TAN::fit` y pone `fitted = true` a mano. `Proposal` es un
  mixin que recibe *referencias a los miembros de la clase hermana* en el
  constructor (`Proposal(dataset, features, className, notes)`), un
  acoplamiento frágil que depende del orden de inicialización y de la
  herencia múltiple.
- **Las variantes X rompen el contrato**: `XSpode` hereda de `Classifier`
  pero ignora por completo el miembro `Network model` (usa sus propias tablas
  de probabilidades) y añade métodos paralelos (`fitx`,
  `predict(instance)`). `XBAODE` pone `fitted = true` al *principio* de
  `trainModel` para poder llamar a `predict` durante el boosting: un
  workaround del propio diseño.
- **`Ensemble` hereda de `Classifier`** pero el `Network model` heredado no
  se usa; anula media interfaz para redirigir a `models[]`. Es composición
  disfrazada de herencia.
- **Duplicación masiva en boosting**: `BoostAODE.cc`, `XBAODE.cc`,
  `BoostA2DE.cc` y `XBA2DE.cc` repiten ~150 líneas del mismo algoritmo
  AdaBoost (ranking por información mutua, bisección, convergencia, poda de
  modelos) cambiando solo el learner base. Cualquier corrección hay que
  aplicarla en 4 sitios (p. ej. el fix de significances con select_features,
  commit 6562a49).

### 2.1 Detalles de robustez

- `Boost::featureSelection` usa `new`/`delete` desnudos para
  `featureSelector`: fuga de memoria si se lanza una excepción intermedia.
- `update_weights` devuelve `std::tuple<torch::Tensor&, double, bool>`
  (referencia dentro de un tuple, propenso a errores de ciclo de vida).
- `XBAODE::getVersion()` devuelve una versión propia ("0.9.7") tapando la de
  la biblioteca.
- Excepciones estándar mezcladas (`invalid_argument`, `logic_error`,
  `runtime_error`) sin tipos propios de la biblioteca.
- `Classifier(Network model)` recibe la red por valor (copia innecesaria).

## 3. Propuestas para la 2.0

En orden de impacto:

### P1 — Tipo `Dataset` como pieza central de la nueva API

Un value-type que encapsule X, y, `features`, `className` y `states`, con
inferencia automática de estados y constructores desde tensores, vectores y
ARFF. Las 4 sobrecargas de `fit` colapsan en:

```cpp
clf.fit(dataset);                          // todo por defecto
clf.fit(dataset, options);                 // weights, etc.
```

`smoothing` pasa a ser hiperparámetro. Las firmas v1 se mantienen una versión
como adaptadores `[[deprecated]]` para permitir la migración gradual de los
proyectos dependientes.

### P2 — Pipeline de entrenamiento explícito en `Classifier`

Sustituir la mutación dispersa por fases con contrato claro:
`validate → buildStructure (virtual) → estimateParameters → finalize`, donde
los datos de entrenamiento viajan como parámetro inmutable y el resultado del
fit (notas, estado, métricas) se devuelve en un `FitResult` en lugar de
acumularse en miembros mutables. Boost obtiene su split train/validación sin
secuestrar `dataset`.

### P3 — Discretización local como decorador, no como mixin

Un wrapper `DiscretizingClassifier(unique_ptr<Classifier>)` (o plantilla
`LocalDiscretizer<TAN>`) que envuelve cualquier clasificador discreto:
discretiza en `fit`, transforma en `predict`. Desaparecen `Proposal` con sus
referencias cruzadas y las clases `*Ld` casi duplicadas; además cualquier
clasificador (presente o futuro) gana soporte de datos continuos sin escribir
una variante Ld.

### P4 — Un único motor de boosting

Extraer el bucle AdaBoost (ranking, bisección, block_update, convergencia,
poda) a `Boost`, parametrizado por una factoría de learners. `BoostAODE`,
`XBAODE`, `BoostA2DE` y `XBA2DE` quedan en ~20 líneas cada uno. Un
experimento nuevo de boosting pasa a ser solo una factoría nueva.

### P5 — Interfaz base mínima + capas opcionales

`BaseClassifier` reducido a
`fit / predict / predict_proba / score / setHyperparameters / getStatus`. La
introspección del grafo (`graph`, `show`, `dump_cpt`, `topological_order`)
pasa a una interfaz aparte que solo implementan los modelos basados en
`Network`. `Ensemble` pasa a componer en vez de heredar.

### P6 — Higiene general

- Factoría/registro de modelos por nombre.
- Const-correctness completa.
- Naming unificado (elegir una convención y aplicarla, con alias deprecados).
- Excepciones propias (`bayesnet::not_fitted_error`,
  `bayesnet::invalid_data`, ...).
- `unique_ptr` para el feature selector; eliminar referencias dentro de
  tuples.
- Eliminar la versión fantasma de XBAODE.
- Evitar copias innecesarias de `Network`.

## 4. Puntos a favor

- Suite Catch2 con categorías por modelo: red de seguridad para refactorizar.
- Separación clara red / clasificadores / ensembles / selección de features.
- Doble empaquetado vcpkg/Conan ya operativo.
- Tamaño manejable (~6.400 líneas de biblioteca).

## 5. Riesgo principal

El riesgo es de **comportamiento, no de compilación**: los refactors P2–P4
deben producir resultados *numéricamente idénticos* (mismas semillas, mismos
scores, mismos modelos generados). Antes de tocar nada hay que fijar tests de
regresión numérica con valores de referencia por modelo y dataset.

## 6. Fases propuestas

1. **Red de seguridad**: tests de regresión numérica con scores de referencia
   por modelo/dataset.
2. **P1 + P6**: API nueva sin tocar internals; adaptadores deprecados.
3. **P2 + P5**: pipeline de entrenamiento y jerarquía.
4. **P3 + P4**: Ld como decorador y motor de boosting único.
5. **Limpieza, documentación y publicación de 2.0.0** con guía de migración.

El plan detallado de ejecución se encuentra en `plan_2_0.md`.
