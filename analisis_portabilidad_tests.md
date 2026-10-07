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
| 3.bis | El mismo `conditionalEntropy` recorría los tensores con `.item()` por elemento: ~50x más lento que su hermana de 3 argumentos, y dominaba el ranking de pares de XBA2DE | Activa (rendimiento, no divergencia) | **Sí** |
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
| `unordered_map` → tabla densa en `conditionalEntropy` | `bayesnet/utils/BayesMetrics.cc` |
| `entropy()` en double en vez de float32 | `bayesnet/utils/BayesMetrics.cc` |
| Peso leído como `double` en las dos líneas contiguas | `bayesnet/utils/BayesMetrics.cc` |

`CFS` no necesita cambios: su bucle usa `>` estricto, así que gana el primero
según `featureOrder`, que ya viene de `argsort`. `Mst::kruskal_algorithm` usa
`stable_sort` sobre una entrada construida en orden de índices, que es
determinista por construcción.

El paso 2 se implementó primero como `std::map` y después como una **tabla densa
`(X, Y)`** indexada por valor, que la sustituye por dos motivos a la vez:

- **Determinismo**: la entropía se acumula recorriendo las celdas en orden
  ascendente de índice, que es exactamente el orden que daba el `std::map`. El
  resultado es bit a bit idéntico al de esa versión, y sigue sin depender de la
  implementación de la biblioteca estándar. Las 2002 aserciones de la suite,
  regeneradas contra la versión con `std::map`, pasan sin tocar ningún valor.
- **Coste**: el `std::map` conservaba el recorrido con `firstFeature[i].item<int>()`,
  cuatro despachos de ATen por muestra. `SelectKPairs` llama a esta función
  O(n²) veces por ronda de boosting, así que era el ~98 % del entrenamiento de
  XBA2DE. Con accessors sobre la tabla, un ranking de pares baja de 72 s a
  0,32 s en n=40 / m=20 000 (~200x). Ver `benchmark/xba2de_profile/`.

Es decir, el paso 2 no tenía por qué costar nada: la nota original («coste
despreciable con las cardinalidades de estos datasets») era correcta sobre el
`std::map` en sí, pero la implementación que lo rodeaba sí costaba, y mucho.

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

El contraste en Linux ya está hecho: ver §7.bis para la medida y §7.ter
para las causas y el arreglo. Sale, salvo un valor.

## 7.bis. Contraste en Linux — primera medida

Fecha: 2026-10-07
Plataforma: Linux x86_64 (Fedora 43), g++ 15.3.1 / libstdc++, libtorch 2.7.1,
build Debug vía `make buildd`.
Ramas medidas: `main` en `d64e64a` y `feat/xba2de-memory-budget` en `1483664`,
en la misma máquina y con el mismo toolchain.

Los valores esperados que hay hoy en la suite se regeneraron en **macOS arm64**
(§6.bis), así que «falla en Linux» significa aquí **«diverge de macOS»**, sin
pronunciarse sobre qué plataforma tiene razón.

| | casos | aserciones | fallos |
|---|---:|---:|---:|
| `main` (d64e64a) | 135 | 1962 | **9** |
| `feat/xba2de-memory-budget` (1483664) | 137 | 1983 | **9** |

Los 9 fallos son **los mismos** en las dos ramas, con las mismas expansiones.

**Respuesta a la pregunta del §7: la unificación todavía no se cumple.** Los
pasos 1–3 eliminaron las fuentes que atacaban, pero queda al menos una
divergencia por encima del umbral de decisión. El paso 2 del plan («generar los
golden en las dos plataformas y comprobar que son idénticos byte a byte») no se
puede cerrar aún, y el `SKIP` de `TestGolden.cc` sigue siendo necesario.

### Los 9 fallos

| Test | Dataset | Linux | macOS (esperado) |
|---|---|---|---|
| `Metrics / Test Maximum Spanning Tree` | glass, raíz 0 | arista `(0, 4)` | arista `(3, 4)` |
| `XBA2DE / Bisection Best` | kdd_JapaneseVowels | 135 nodos | 240 |
| `XBA2DE / Bisection Best vs Last` | kdd_JapaneseVowels | 0.990000 | 0.983333 |
| `XBAODE / Bisection Best` | kdd_JapaneseVowels | 30 nodos | 75 |
| `XBAODE / Bisection Best vs Last` | kdd_JapaneseVowels | 0.990000 | 0.980000 |
| `BoostAODE / Bisection Best` | kdd_JapaneseVowels | 30 nodos | 75 |
| `BoostAODE / Bisection Best vs Last` | kdd_JapaneseVowels | 0.986667 | 0.990000 |
| `BoostA2DE / Bisection Best` | kdd_JapaneseVowels | 60 nodos | 465 |
| `Models / KDBLd` | glass | 0.864486 | 0.869159 |

Nueve es una **cota inferior**: `REQUIRE` aborta la sección, así que el MST de
glass con raíz 1 y los tres datasets restantes del bucle de `KDBLd` no llegaron
a evaluarse.

### Lecturas

**1. Los fallos se concentran exactamente donde el §2 predijo.** Solo aparecen
dos datasets: kdd_JapaneseVowels, que tiene el menor hueco de decisión de toda
la tabla (3.4e-04 en el ranking de pares) y 10 empates exactos de 91; y glass,
con 15 empates de 36. Ningún fallo en iris, ecoli, diabetes, heart-statlog ni
liver-disorders. El mecanismo sigue siendo el (1) de la sección 1 —
desempates—, no el (2).

**2. Las divergencias de nodos son divergencias en el número de modelos**, no
en la estructura de un modelo. Con 14 features cada modelo aporta 15 nodos, así
que: XBA2DE 9 modelos vs 16; XBAODE y BoostAODE 2 vs 5; BoostA2DE 4 vs 31. Una
sola decisión que se bifurca al principio la amplifica el bucle de boosting
hasta parar en un punto completamente distinto. Esto explica por qué un fallo
tan grande no contradice el margen de 1e-14 medido en el §3: no hace falta
mucho para cambiar el primer desempate.

**3. El MST de glass es la pista más valiosa, y contradice el inventario del
§5.** Ahí se dio por descartado `Mst::kruskal_algorithm` porque usa
`stable_sort` sobre una entrada construida en orden de índices. Si eso fuera
suficiente, un empate exacto se resolvería igual en las dos plataformas. Que
`(0, 4)` y `(3, 4)` se intercambien implica una de dos cosas, y conviene
averiguar cuál:

- los pesos de `conditionalEdge` para esas dos aristas **no** son exactamente
  iguales, y difieren entre plataformas lo bastante para invertir la
  comparación — lo que situaría el ruido muy por encima de los 6.4e-14 del §3; o
- la entrada de `kruskal_algorithm` no es tan determinista como se supuso.

Es un caso mínimo y aislado (una llamada, sin boosting, sin folds), así que es
el sitio por donde empezar.

**4. `KDBLd` sobre glass es un mecanismo independiente.** 0.864486 = 185/214 y
0.869159 = 186/214: **una sola muestra** clasificada distinto. `KDBLd` es
discretización local (fimdlp), un camino que no pasa por los rankings ni por el
bucle de boosting. Que caiga `KDBLd` y no `KDB`, `TANLd` ni `AODELd` apunta a la
discretización iterativa, no a los selectores. Es una segunda fuente, fuera del
inventario del §5.

### Contraejemplo útil: un valor que sí es portable

La rama `feat/xba2de-memory-budget` añade un test que imprime una línea
`GOLDEN[Memory-limited]` con nodos, aristas, estados, notas y la huella de
memoria acumulada. Esa línea sale **byte a byte idéntica** en macOS arm64 y en
Linux x86_64:

```
GOLDEN[Memory-limited] nodes=90 edges=216 states=2187 notes=3 || Memory limit reached: 9 models built, 0.10 MiB used of 0.12 MiB budget || Pairs not used in train: 27 || Number of models: 9
```

Interesa porque es sobre glass, con sus 15 empates, y porque la contabilidad de
memoria depende de `capacity()` de vectores de libstdc++ y de `sizeof` de las
estructuras: ni el presupuesto ni el punto de corte se mueven. Es decir, la
divergencia no está en todo el pipeline, está en decisiones concretas.

### Reproducir

```bash
find . -name "*.gcda" -delete        # los .gcda viejos ensucian la salida
make buildd
cd build_Debug/tests && ./TestBayesNet
```

Para el contraste con la línea base, lo mismo tras `git checkout d64e64a`.

## 7.ter. Causas y arreglo

Fecha: 2026-10-07

El §7.bis dejó nueve fallos sin explicar. Ocho tienen causa identificada y
arreglada; el noveno sigue abierto. La suite pasa en Linux: **2021 aserciones en
137 casos, 0 fallos**.

### Causa 1 — `std::shuffle` en el helper de tests (7 de los 9)

`ShuffleArffFiles` (`tests/TestUtils.cc`) elegía el submuestreo así:

```cpp
std::mt19937 g{ 173 };
std::shuffle(indices.begin(), indices.end(), g);
```

**Es el mismo defecto que folding 1.1.x** (fila #1 del inventario del §5, que se
dio por «eliminada» cuando se arregló en la dependencia): el estándar no
especifica el algoritmo de `std::shuffle`, solo que el resultado sea uniforme.
Compilando el mismo programa con las dos bibliotecas estándar en esta máquina:

| `std::shuffle(mt19937{173})`, n=1200 | primeros índices | checksum FNV |
|---|---|---|
| libstdc++ 20260722 | 609 547 894 399 905 487 440 78 35 95 | `8ea4c51c676bfcd5` |
| libc++ 210108 | 229 782 610 828 605 664 327 698 884 154 | `41655d5f0b1df965` |
| Fisher-Yates especificado | 183 179 412 875 128 387 222 265 1192 528 | `be0c5d85347ca05d` en las dos |

Los siete tests que submuestrean con `shuffle=true` **no entrenaban sobre
casi-empates: entrenaban sobre conjuntos de filas distintos**. Eso explica por
qué las divergencias eran tan grandes (2 modelos frente a 5, 4 frente a 31) y
por qué los valores que Linux producía eran exactamente los que `bf4b0cf`
sustituyó: esa regeneración no arregló nada, cambió la plataforma de referencia
de esos nueve de Linux a macOS.

La correlación es exacta: los 7 call sites con `shuffle=true` son los 7 tests de
bisección que fallaban, y los 3 que usan `num_samples` sin shuffle
(`mfeat-factors`, `spambase`) pasaban.

El mismo `std::shuffle` estaba **en la librería**, detrás de `order = "rand"`, en
`BoostAODE`, `BoostA2DE`, `XBAODE` y `XBA2DE`. Los cuatro usan ahora
`bayesnet::deterministicShuffle` (`bayesnet/utils/bayesnetUtils.h`),
deliberadamente el mismo Fisher-Yates + `bounded_rand` de Lemire que
`folding::detail::shuffle`.

### Causa 2 — el MST de glass: un empate a ocho, no ruido numérico

La sospecha del §7.bis («los pesos difieren, el ruido está por encima de
6.4e-14») era falsa. Medido: en glass, **las ocho aristas de la feature 4 (`Si`)
valen exactamente `0x00000000`**, porque MDLP deja `Si` con un solo estado y
`mutualInformation` de una variable constante es cero exacto. El empate es
exacto, no aproximado.

Comprobaciones que descartan las alternativas:

- `Si` es constante también en fimdlp 2.1.3, 3.0.0 y 3.0.1 (mismos cortes
  frontera `{69.81, 75.41}`, cero cortes internos), así que no es un cambio de
  versión de la dependencia.
- El margen de la decisión MDLP que rechaza cortar `Si` es `ig = 0.1192` frente a
  `term = 0.1254`, **5e-2 relativo**. No es frágil: el mínimo sobre las 36
  decisiones de glass es 5e-2.
- `CPPFImdlp::sortIndices` usa `stable_sort` con orden total (empate en X roto
  por y, y el resto por índice), así que la discretización es determinista.

> **Corrección (§7.quater)**: el empate es exacto en Linux, pero *no* en macOS.
> Hacer total el comparador era necesario y no suficiente; faltaba garantizar que
> los pesos empatados valen cero exacto en las dos plataformas. Ver §7.quater.

Con ocho aristas exactamente empatadas, la que entra en el árbol la decidía el
orden en que `addEdge` se llamó, que `stable_sort` preservaba. **El §5 descartó
`Mst::kruskal_algorithm` por usar `stable_sort`, y eso era insuficiente**:
estable no es lo mismo que total. El comparador es ahora un orden total (peso
descendente, luego los extremos `(u, v)`), así que el resultado queda definido
por los datos. `{0, 4}` y `{3, 4}` son los dos árboles de expansión máxima
válidos; `{0, 4}` es el canónico bajo ese orden.

Por la misma razón se hicieron totales dos desempates más que el §6.bis no
cubrió:

| Sitio | Qué decidía el empate | Arreglo |
|---|---|---|
| `TAN::buildModel` | la raíz, con un `sort` que solo comparaba la MI | desempate por índice de feature |
| `KDB::add_m_edges` | el siguiente padre, con `torch::argmax` sobre filas empatadas a 0 | barrido explícito al primer máximo |

`torch::argmax` documenta devolver el primer máximo, pero lo decide su estrategia
de reducción. En Linux ya devolvía el primero (el arreglo no mueve ningún valor),
de modo que es defensivo.

### Lo que queda abierto — `KDBLd` sobre glass

Linux da 0.864486 (185 de 214); el valor regenerado en macOS era 0.869159 (186).
**Una sola muestra**, y no he localizado el mecanismo. Descartado:

- No hay empates en el argmax de la predicción (0 de 214 muestras).
- No es sensibilidad numérica: el score es 185/214 con perturbaciones relativas
  de la entrada de 0, 1e-7, 1e-6, 1e-5, 1e-4 y 1e-3. No está en el filo.
- No son los márgenes MDLP de la discretización local: 320 decisiones, margen
  relativo mínimo **9.9e-05**, tres órdenes por encima del ruido de float32
  (6e-8). Ojo: `precision_t` de fimdlp es `float`, así que ese es el umbral
  relevante, no el 1e-14 del §3.
- No es fimdlp 3.0.0 vs 3.0.1 (cortes idénticos en los diez datasets).
- No es el criterio de convergencia del bucle iterativo: compara estructuras
  (`previousModel == classifier->getModel()`), no números.
- `factorize` numera con `std::map` en orden de inserción y `topological_sort` no
  usa contenedores desordenados, así que ninguno de los dos aporta orden
  arbitrario.

Lo que queda son mecanismos dentro de libtorch que no se pueden probar sin la
otra plataforma: orden de reducción en float32 sobre arm64 frente a x86_64. El
valor del test es ahora el de Linux, coherente con el resto de la regeneración.
**Pendiente: correr la suite en macOS.** Si este único valor vuelve a divergir,
lo honesto es no fijarlo con `epsilon(1e-5)`.

### Inventario del §5, actualizado

| # | Fuente | Estado |
|---|---|---|
| 1 | `std::shuffle` en `folding` 1.1.x | Resuelta en folding 2.0.0 |
| 1.bis | **`std::shuffle` en `ShuffleArffFiles` y en los cuatro Boost (`order = "rand"`)** | **Resuelta aquí** — era la dominante |
| 2 | `std::sort` con empates en `argsort`, `SelectKBestWeighted`, `SelectKPairs` | Resuelta en §6.bis |
| 2.bis | **Empates en `kruskal_algorithm`, `TAN::buildModel` y `KDB::add_m_edges`** | **Resuelta aquí** — el §5 los había descartado |
| 3 | Orden de iteración de `unordered_map` en `conditionalEntropy` | Resuelta en §6.bis |
| 4 | `entropy()` en float32 | Resuelta en §6.bis |
| 5 | Peso leído como `double` y como `float` | Resuelta en §6.bis |
| 6 | Kernels de libtorch, arm64 vs x86_64 | Inherente; es la sospecha que queda para `KDBLd`/glass |

Revisados y **descartados** en esta pasada: `Node::minFill` construye un
`unordered_set` pero solo consume el tamaño de las combinaciones, que no depende
del orden; `featureIndexMap` en `Node::computeCPT` solo se consulta por clave;
`Network::isCyclic` usa sus `unordered_set` solo para pertenencia; los `sort` de
`Network::operator==` ordenan pares completos; los de `BayesMetrics` líneas 147 y
160 ordenan `double` sueltos.

## 7.quater. La vuelta de macOS: el empate no era exacto en las dos plataformas

Fecha: 2026-10-07

Corrida la suite en macOS con los arreglos del §7.ter: **135 de 137 casos pasan**.

- Los **siete casos de bisección pasan**, y las líneas GOLDEN coinciden byte a byte
  con Linux (`nodes=195 edges=507 states=18382`, 13 modelos, `score=0.9875`,
  `order-rand 0.827103`). El arreglo de `std::shuffle` está verificado en las dos
  plataformas; esa causa queda cerrada.
- El **MST de glass seguía divergiendo**, ya con el comparador total. Y eso es una
  deducción forzada: si el orden es total y macOS elige `(3, 4)` en vez de `(0, 4)`,
  entonces en macOS `peso(3,4) > peso(0,4)`. **Las aristas de `Si` no valían cero
  exacto allí.** La afirmación del §7.ter de que no podían diferir era errónea.

### El mecanismo real

Las ocho aristas de `Si` se calculan en dos direcciones distintas, según de qué
lado caiga `Si` en `doCombinations`:

| Par | Llamada | Vale cero porque |
|---|---|---|
| `(0,4) (1,4) (2,4) (3,4)` | `mutualInformation(X, Si)` | `H(X) - H(X\|Si)` y `H(X\|Si)` debe valer **exactamente** `H(X)` |
| `(4,5) (4,6) (4,7) (4,8)` | `mutualInformation(Si, X)` | `H(Si) - H(Si\|X)` y ambos son cero |

La primera es la frágil: `H(X)` lo calcula `entropy()` con `bincount` y operaciones
de ATen, y `H(X|Si)` lo calcula `conditionalEntropy()` con una tabla densa y un
bucle secuencial. **Son dos implementaciones de la misma cantidad**, y
`mutualInformation` las resta. Que coincidieran bit a bit era un accidente del
build: en x86_64/libstdc++ sí, en arm64 no, y el residuo de ~1e-18 sobrevive al
`std::max(..., 0.0)` cuando cae del lado positivo.

Medido en Linux: `bincount` de ATen y la suma secuencial coinciden bit a bit en
las 145 celdas de glass, que es justo por lo que aquí salía cero exacto.

### El arreglo

Dos identidades declaradas en `conditionalEntropy`, en vez de dejarlas salir de la
aritmética:

```cpp
if (firstMax == first.min().item<int>())  return 0;                       // X constante: H(X|Y) = 0
if (second.max() == second.min())         return entropy(firstFeature, weights); // Y constante: H(X|Y) = H(X)
```

La segunda devuelve **la misma llamada** que `mutualInformation` va a restar, así
que la diferencia es cero exacto por construcción, en cualquier plataforma y con
cualquier orden de reducción. No mueve ningún valor en Linux y cuesta tres
reducciones de ATen más por llamada, sin efecto medible: `[XBA2DE]` tarda 31,45 s
con el arreglo y 31,46 s sin él.

Un intento intermedio **que no sirvió** y conviene no repetir: derivar el marginal
de la propia tabla conjunta en vez de `bincount`. Hace `conditionalEntropy`
autoconsistente, pero rompe la coincidencia con `entropy()`, que es la que de
verdad importa porque es la que se resta — y con eso aparecía un residuo de
6.74e-18 en las aristas `(4,0)`, `(4,2)` y `(4,3)` **en Linux**, donde antes no
había ninguno. La lección: lo que tiene que coincidir no es cada función consigo
misma, sino las dos que se restan entre sí.

### Tests que fijan la invariante, no el golden

Estos dos habrían atrapado el fallo en macOS sin necesidad de un valor esperado:

- `[Metrics]` «A constant feature has exactly zero mutual information»: comprueba
  primero que `Si` es constante y luego que `entropy`, `mutualInformation` en las
  dos direcciones, `conditionalMutualInformation` y las entradas de
  `conditionalEdge` valen **cero exacto** (comparado con `==`, no con `Approx`), y
  también por clase, que es lo que `conditionalEdge` acumula.
- `[MST]` «The maximum spanning tree breaks weight ties by endpoints»: con todos
  los pesos iguales el árbol tiene que ser `{0,1} {0,2} {0,3}`, y el resultado no
  puede depender del orden en que se añaden las aristas.

### Lo que sigue abierto

`KDBLd` sobre glass: macOS 186/214, Linux 185/214. El residuo que acabamos de
eliminar alimentaba también el `argmax` de `KDB::add_m_edges` a través de
`conditionalEdge`, así que es plausible que este arreglo lo cierre, pero no está
comprobado: hay que volver a correr la suite en macOS. Si persiste, lo descartado
en el §7.ter sigue descartado y la sospecha vuelve a ser el orden de reducción en
float32 de libtorch sobre arm64, esta vez en un sitio sin identidad exacta que
imponer.

### Inventario, corregido

La fila 6 del §5 («kernels de libtorch y aritmética arm64 vs x86_64 — inherente,
pero irrelevante: queda a 1e-14») era demasiado optimista. No es irrelevante: un
residuo de 1e-18 es decisivo en cuanto alimenta un desempate exacto. Las dos
cosas tienen que ir juntas — desempates totales **y** ceros exactos donde la
matemática dice cero.

## 8. Reproducir las mediciones

Las mediciones 1 y 2 usan un programa aislado que llama a `Metrics`
directamente; la 3 es la suite con `BayesMetrics.cc` instrumentado. Ninguno de
los dos se ha commiteado: son instrumentación temporal, descrita aquí con el
detalle suficiente para rehacerla.
