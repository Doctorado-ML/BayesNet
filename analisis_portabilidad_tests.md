# ¿Se pueden unificar los tests entre plataformas?

Fecha: 2026-08-15
Rama: `feat/fimdlp-3.0.0`
Contexto: `analisis_tests_fallidos.md`, `tests/README.md` (rama `v2/phase-0-golden-tests`)

## 0. Pregunta

Con `folding` 2.0.0 los folds ya no dependen de la plataforma. ¿Permite eso
retirar el `SKIP` de los golden fuera de Linux y tener una única suite exacta
que dé los mismos resultados en todas partes?

**Respuesta corta: sí, pero folding no era la causa principal.** El obstáculo
real es otro, está en nuestro código, y es reparable. La divergencia de coma
flotante — la explicación registrada hasta ahora — es **diez órdenes de
magnitud demasiado pequeña** para justificar lo que se observaba.

## 1. Lo que decía el diagnóstico anterior

`tests/README.md` y el commit `377587e0` atribuyen la divergencia a:

> las diferencias de coma flotante de libtorch entre plataformas cambian los
> desempates en los rankings por información mutua

Esa frase mezcla dos mecanismos distintos que conviene separar, porque tienen
soluciones opuestas:

1. **Desempate exacto**: dos features con puntuación *idénticamente igual*.
   Quién va primero lo decide `std::sort`, cuyo comportamiento con elementos
   equivalentes **no está especificado por el estándar**. libstdc++ y libc++
   dan órdenes distintos. No interviene la coma flotante.
2. **Casi-empate**: dos puntuaciones que difieren en los últimos bits, y el
   ruido numérico de la plataforma invierte la comparación.

El estudio mide ambos. El primero es masivo y reparable; el segundo es
irrelevante en este código.

## 2. Medición 1 — cuánto margen tienen las decisiones

Se calcularon las puntuaciones que ordenan `SelectKBestWeighted` (features) y
`SelectKPairs` (pares), y para cada ranking se contaron los empates exactos y
el menor hueco relativo entre puntuaciones consecutivas.

| dataset | ranking | n | empates exactos | menor hueco relativo |
|---|---|---:|---:|---:|
| iris | kbest | 4 | 0 | 2.2e-02 |
| iris | pairs | 6 | 0 | 1.1e-01 |
| glass | kbest | 9 | 0 | 6.9e-03 |
| glass | pairs | 36 | **15** | 1.7e-03 |
| ecoli | kbest | 7 | 0 | 1.5e-01 |
| ecoli | pairs | 21 | **7** | 1.2e-02 |
| diabetes | kbest | 8 | 0 | 4.2e-02 |
| diabetes | pairs | 28 | **7** | 1.1e-02 |
| heart-statlog | kbest | 13 | **1** | 9.0e-03 |
| heart-statlog | pairs | 78 | **51** | 1.4e-03 |
| liver-disorders | kbest | 6 | **3** | 8.4e-01 |
| liver-disorders | pairs | 15 | **14** | — (todas iguales) |
| kdd_JapaneseVowels | kbest | 14 | 0 | 1.7e-02 |
| kdd_JapaneseVowels | pairs | 91 | **10** | 3.4e-04 |

Dos lecturas:

- **Los empates exactos son masivos** en los rankings de pares: 15 de 36 en
  glass, 51 de 78 en heart-statlog, y en liver-disorders las 15 puntuaciones
  son iguales entre sí. Son igualdades exactas, no aproximadas: la CMI de
  muchos pares vale exactamente 0. Su orden lo decide íntegramente la
  implementación de `std::sort`.
- **Los huecos no empatados son enormes**: el más estrecho de todos es
  3.4e-04, y la mayoría están entre 1e-2 y 1e-1.

## 3. Medición 2 — cuánto ruido numérico hay realmente

`Metrics::conditionalEntropy` (versión de 2 features) acumula la entropía
recorriendo un `std::unordered_map`, cuyo orden de iteración es dependiente de
la implementación. Como la suma en coma flotante no es asociativa, ese orden
cambia los últimos bits del resultado — y `mutualInformation` y
`conditionalMutualInformation` pasan por ahí.

Se sustituyó `unordered_map` por `std::map` (orden determinista) y se
compararon las 336 puntuaciones con 17 dígitos:

| | |
|---|---|
| puntuaciones que cambian | 57 de 336 |
| diferencia relativa máxima | **6.4e-14** |
| diferencia relativa mínima | 1.2e-16 |
| menor hueco de decisión (medición 1) | 3.4e-04 |
| **razón hueco / ruido** | **≈ 5.4 × 10⁹** |

El ruido numérico es unos **nueve órdenes de magnitud menor** que el margen de
decisión más ajustado. No puede invertir ninguna comparación que no sea un
empate exacto.

## 4. Medición 3 — confirmación extremo a extremo

Se inyectó una perturbación relativa determinista en *todas* las puntuaciones
de *todas* las llamadas (también dentro del bucle de boosting, donde los pesos
cambian en cada iteración) y se ejecutó la suite completa:

| perturbación relativa | tests fallidos |
|---|---:|
| ninguna | 1 |
| orden de empates invertido | 1 |
| 1e-15 | 1 |
| 1e-12 | 1 |
| 1e-9 | 1 |
| 1e-6 | 1 |
| **1e-3** | **6** |

La suite es insensible hasta 1e-6 y solo se rompe en 1e-3. Frente al ruido real
medido de 6.4e-14, el margen de seguridad es de **más de siete órdenes de
magnitud**.

El "1 fallido" de la línea base no es ruido: aparece al cambiar `std::sort` por
`std::stable_sort` en los selectores, sin tocar ningún número. El test que cae
es `BoostA2DE / "Order asc, desc & random"`, que ordena pares sobre **glass**,
el dataset con 15 empates exactos de 36. Es exactamente el mecanismo (1) de la
sección 1, reproducido en local sin necesidad de otra plataforma.

## 5. Inventario de fuentes de divergencia

| # | Fuente | Estado | ¿Reparable aquí? |
|---|---|---|---|
| 1 | `folding` 1.1.x usaba `std::shuffle`, no especificado por el estándar → folds distintos por plataforma | **Eliminada** en folding 2.0.0 (Fisher-Yates propio + `bounded_rand` de Lemire sobre `mt19937`, ambos deterministas) | Ya resuelto |
| 2 | `std::sort` con puntuaciones empatadas en `argsort`, `SelectKBestWeighted` y `SelectKPairs` | **Activa y dominante** | **Sí** |
| 3 | Orden de iteración de `unordered_map` en `conditionalEntropy` → suma en distinto orden | Activa, magnitud 6.4e-14 | **Sí** |
| 4 | `entropy()` calcula en **float32** (`counts.to(torch::kFloat)`), mientras el resto va en double | Activa, reduce la precisión útil a ~1e-7 | **Sí** |
| 5 | `conditionalEntropy` lee el mismo peso como `double` (l. 275) y como `float` (l. 276) en dos líneas contiguas | Activa, inconsistencia numérica | **Sí** |
| 6 | Kernels de libtorch y aritmética arm64 vs x86_64 | Inherente | No, pero irrelevante: queda a 1e-14 |

Comprobados y **descartados** como fuentes: `Node::minFill` usa un
`unordered_set` pero solo consume su `size()`, que no depende del orden;
`Network::isCyclic` usa los `unordered_set` solo para pertenencia;
`MST::reorder` ya sustituyó deliberadamente `unordered_set` por `list`;
`Mst::kruskal_algorithm` ya usa `stable_sort`.

## 6. Propuesta

Es viable una suite única y exacta en todas las plataformas. Por orden de
impacto:

1. **Desempate determinista y explícito** (resuelve #2, el que realmente
   rompe). Que el criterio de orden sea total: a igualdad de puntuación,
   ordenar por índice de feature — o por el par `(i, j)` en los rankings de
   pares. Con eso el resultado deja de depender del algoritmo de ordenación y
   pasa a estar definido por nosotros. Afecta a `argsort`,
   `SelectKBestWeighted` y `SelectKPairs`.
2. **Orden de suma determinista** (resuelve #3): `std::map` en lugar de
   `std::unordered_map` en `conditionalEntropy`. Coste despreciable con las
   cardinalidades de estos datasets.
3. **Calcular `entropy()` en double** (resuelve #4) y unificar la lectura de
   pesos (#5). Son defectos numéricos por sí mismos, al margen de la
   portabilidad.

Con 1 y 2, dos plataformas que ejecuten el mismo binario lógico producen el
mismo ranking siempre que sus puntuaciones coincidan hasta ~1e-6, y la
medición 3 muestra que coinciden hasta 1e-14.

**Coste**: los pasos 1–3 **cambian los valores actuales** (el paso 1 ya rompe
un test hoy). Habría que regenerar los golden y los valores heredados una vez
más, en un commit separado, y esa regeneración pasaría a ser válida en
cualquier plataforma.

## 6.bis. Aplicado

Los tres pasos están implementados:

| Cambio | Fichero |
|---|---|
| Desempate por índice en `argsort` | `bayesnet/utils/bayesnetUtils.cc` |
| Desempate por `(i, j)` en `SelectKPairs`, por índice en `SelectKBestWeighted` (ambas ramas) | `bayesnet/utils/BayesMetrics.cc` |
| Desempate por índice al ordenar por importancia | `bayesnet/feature_selection/L1FS.cc` |
| `unordered_map` → `std::map` en `conditionalEntropy` | `bayesnet/utils/BayesMetrics.cc` |
| `entropy()` en double en vez de float32 | `bayesnet/utils/BayesMetrics.cc` |
| Peso leído como `double` en las dos líneas contiguas | `bayesnet/utils/BayesMetrics.cc` |

`CFS` no necesita cambios: su bucle usa `>` estricto, así que gana el primero
según `featureOrder`, que ya viene de `argsort`. `Mst::kruskal_algorithm` usa
`stable_sort` sobre una entrada construida en orden de índices, que es
determinista por construcción.

**Impacto en los valores esperados**: solo se movieron dos, ambos en
`BoostA2DE / "Order asc, desc & random"` sobre glass — el dataset con 15
empates exactos de 36, que es justamente el caso que el estudio predijo como
único sensible. `asc` 0.789720 → 0.799065 y `rand` 0.845794 → 0.855140; `desc`
no cambia. El resto de la suite (2002 aserciones, 135 casos) pasa sin tocar
nada más, lo que confirma que los otros valores no dependían de desempates.

## 7. Verificación pendiente

Todo lo anterior se midió en macOS arm64. Falta el contraste directo en Linux,
que es la única prueba definitiva de que la unificación funciona. El plan sería:

1. Aplicar los pasos 1–3.
2. Generar los golden en Linux y en macOS por separado.
3. Comprobar que los ficheros son idénticos byte a byte.

Si lo son, el `SKIP` de `TestGolden.cc` y las 66 aserciones relajadas al margen
portable de 0.08 (repartidas en 7 ficheros) dejan de tener sentido, y la
arquitectura de dos niveles puede colapsarse en una sola suite exacta.

Sin ese contraste, la conclusión de este estudio es que **no queda ninguna
fuente de divergencia conocida por encima de 1e-13**, lo que hace la
unificación muy probable pero no demostrada.

## 8. Reproducir las mediciones

Las mediciones 1 y 2 usan un programa aislado que llama a `Metrics`
directamente; la 3 es la suite con `BayesMetrics.cc` instrumentado. Ninguno de
los dos se ha commiteado: son instrumentación temporal, descrita aquí con el
detalle suficiente para rehacerla.
