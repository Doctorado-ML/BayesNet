# Handoff: `max_memory_gb` en XBA2DE

Fecha: 2026-09-11
Rama: `feat/xba2de-memory-budget` (sale de `main` en `d64e64a`)
Commit del trabajo: `090a285 feat(XBA2DE): add max_memory_gb working-memory budget`
Plataforma donde se desarrolló y verificó: macOS arm64 (libc++), build Debug vía
`make buildd`. Suite completa: **2023 assertions en 137 test cases, todo pasa**.

El objetivo de este documento es que se pueda continuar en Linux (libstdc++),
donde hay tests que fallan, sin tener que reconstruir el contexto de diseño.

---

## 1. Qué se ha hecho

Nuevo hiperparámetro `max_memory_gb` en `XBA2DE`: presupuesto de memoria de
trabajo para la ensemble, en gigabytes (1 GB = 2^30 bytes). `0.0` (default) =
sin límite. Negativo → `std::invalid_argument`.

El bucle de boosting comprueba antes de construir cada `XSp2de` si cabe; si no
cabe, para. Es una condición de salida más, junto a agotamiento de pares y
convergencia. Al salir por memoria se emite la nota
`Memory limit reached: N models built, X MiB used of Y MiB budget` y
`status = WARNING` (mismo tratamiento que `Pairs not used in train`).

### Decisiones de diseño (todas confirmadas con el autor)

| # | Decisión | Elección |
|---|----------|----------|
| Q1 | Nombre, tipo, default | `max_memory_gb`, `double`, `0.0` = ilimitado, GiB (2^30) |
| Q2 | Qué cuenta en el presupuesto | **Solo** las tablas de los `XSp2de` acumulados. Ni `dataset`, ni `metrics`, ni el ranking de pares |
| Q3 | Cómo se obtiene el nº máximo de modelos | Contabilidad incremental exacta: se acumula la huella real de cada modelo tras `fit`; se sale cuando el siguiente no cabe. No hay un `N_max` precalculado |
| Q4 | Dónde vive la fórmula | En `XSp2de`: `static estimateFootprint(...)` decide si se construye; `memoryFootprint()` (no estático) es lo que se acumula |
| Q5 | Liberar `childCounts_` tras `computeProbabilities()` | Sí. Es el bloque dominante y nada aguas abajo lo lee. `to_string()` imprime `childProbs_` en su lugar |
| Q6 | Granularidad respecto al pack de bisección | Cortar en seco (pack parcial), no esperar a que quepa un pack entero |
| Q7 | No cabe ni un modelo | `std::runtime_error`, perezoso (cuando el primer par candidato no pasa el chequeo) |
| Q8 | Nota y status | Texto arriba + `WARNING`; convive con `Pairs not used in train` y `Number of models` |
| Q9 | Pico vs residente | El pre-chequeo acota el **pico** del candidato (counts + probs vivos durante `fit`); lo acumulado es el **residente** (sin `childCounts_`). Así nunca se supera el presupuesto en ningún instante |
| Q10 | Poda de `numItemsPack` al salir por memoria | Sin trato especial. Nota: en la práctica no puede coincidir, porque `finished = true` salta el bloque de convergencia y `tolerance` no se incrementa. Aun así se resta la huella de los modelos podados para que `memoryUsed` sea exacto si alguien cambia el flujo |
| Q12 | ¿Sube a `Boost`? | No. Solo `XBA2DE`. La estimación depende del layout de tablas de cada modelo (`XSpode` tiene otro) |
| Q14 | Qué mide `memoryFootprint()` | `capacity()*sizeof(T)` de los 6 vectores `double` + los 2 de `int` + `sizeof(*this)` |

### Fórmula de la estimación (pico)

```
S = Σ_{f ≠ sp1,sp2} s_f          (suma de cardinalidades de los hijos)

peak = sizeof(XSp2de)
     + 2·n·sizeof(int)                         # states_, childOffsets_
     + 8·[ 2C + 2·s1·C + 2·s2·C + 2·s1·s2·C    # class, sp1, sp2, pair (counts+probs)
           + 2·s1·s2·C·S ]                     # childCounts_ + childProbs_
```

`estimateFootprint` se alimenta con el mapa `states` de la ensemble. Es cota
superior de la huella real porque `XSp2de::buildModel` deriva `states_[f] =
dataset[f].max()+1` sobre el **fold de entrenamiento** (`Boost::buildModel`
hace `dataset = X_train` cuando `convergence=true`), cuyos máximos por feature
solo pueden ser menores.

### Desviación respecto a lo aprobado

El formato de la nota se aprobó como MiB enteros (`bytes >> 20`). En `glass`
daba `0 MiB used of 0 MiB budget` porque los modelos son de KiB. Se cambió a
MiB con dos decimales (`asMiB()` en `XBA2DE.cc`): `0.10 MiB used of 0.12 MiB budget`.

---

## 2. Archivos tocados

| Archivo | Cambio |
|---------|--------|
| `bayesnet/classifiers/XSP2DE.h` | Declara `memoryFootprint()` y `static estimateFootprint(states, statesClass, sp1, sp2)` |
| `bayesnet/classifiers/XSP2DE.cc` | Implementa ambos; `std::vector<double>().swap(childCounts_)` al final de `computeProbabilities()`; `to_string()` vuelca `childProbs_`; `#include <type_traits>` |
| `bayesnet/ensembles/XBA2DE.h` | Miembro `max_memory_gb_ = 0.0`; `version` `0.9.7` → `0.9.8` |
| `bayesnet/ensembles/XBA2DE.cc` | Registro y parseo del hiperparámetro; `asMiB()`; presupuesto, `stateCounts`, `memoryUsed`, `memoryLimited`; chequeo en el bucle interno; acumulación tras `fit`; resta en la poda; nota final |
| `tests/TestXBA2DE.cc` | Versión `0.9.8`; nuevo `TEST_CASE("Working memory budget", "[XBA2DE]")` |
| `tests/TestXSPnDE.cc` | Nuevo `TEST_CASE("Memory footprint estimate is an upper bound", "[XSP2DE]")` |
| `CHANGELOG.md` | Entradas en Added (XBA2DE, XSP2DE) y Changed (XSP2DE); lista de hiperparámetros válidos actualizada |
| `README.md` | Párrafo bajo `#### - XBA2DE` |

---

## 3. Los tests nuevos y de qué dependen

### `Working memory budget` (`tests/TestXBA2DE.cc`, dataset `glass`)

1. `max_memory_gb = -1.0` → `invalid_argument`. *Determinista, no depende de plataforma.*
2. `unlimited` (sin hiperparámetro) vs `zero` (`max_memory_gb = 0.0`): nodos,
   aristas, notas y score **idénticos**. *Ambos corren en el mismo binario, así
   que cualquier divergencia entre plataformas afecta a los dos por igual; no
   debería fallar en Linux.*
3. `max_memory_gb = 1/2^30` (1 byte) → `runtime_error` en `fit`. *Determinista.*
4. Presupuesto = 3 × pico del par más caro de `glass` →
   `limited.getNumberOfNodes() < unlimited.getNumberOfNodes()`,
   `status == WARNING`, alguna nota empieza por `Memory limit reached:`.
   *En macOS construye 9 modelos de 36. La comparación `<` es robusta salvo
   que la run sin límite converja con ≤ 9 modelos, cosa que no ocurre
   (36 pares, para por agotamiento).*

### `Memory footprint estimate is an upper bound` (`tests/TestXSPnDE.cc`, `iris`)

Para los 6 pares de `iris`: `estimateFootprint(...) >= clf.memoryFootprint()`
y `memoryFootprint() > 0`.

**Posible fuente de divergencia en Linux**: `memoryFootprint()` usa
`capacity()`. La estimación asume `capacity() == size()` tras `resize()` sobre
un vector vacío. libc++ y libstdc++ lo cumplen ambos para `resize` desde vacío,
pero si en libstdc++ alguna `capacity()` sale mayor que la estimada (por
ejemplo si algún vector se redimensiona dos veces), la desigualdad puede
romperse por unos bytes. Si es el caso, la solución correcta es medir con
`size()` en vez de `capacity()` — lo que el modelo *usa*, no lo que el
asignador *retiene* — y anotarlo en el comentario del método en `XSP2DE.h`.

Apunte: en `iris` este test discrimina poco; la holgura viene casi toda del
`childCounts_` liberado, porque el fold de entrenamiento contiene todos los
valores de cada feature.

---

## 4. Tests preexistentes que podrían haberse movido

Ninguno debería, en teoría: con `max_memory_gb = 0` toda la contabilidad
está detrás de `if (memoryBudget > 0)` y el bucle es byte a byte el de antes.
La liberación de `childCounts_` tampoco cambia ninguna predicción.

Lo único observable que cambia sin activar el hiperparámetro:

- `XSp2de::to_string()` imprime `childProbs_` en vez de `childCounts_`. Los
  dos tests que lo usan (`TestXSPnDE.cc`: `Check hyperparameters` compara dos
  `to_string()` iguales; `Joint vs independent` compara dos distintos) pasan
  en macOS. No hay ningún test que fije el tamaño del dump de `XSp2de`
  (`TestXSPODE.cc:126` lo hace para `XSpode`, que no se ha tocado).
- `XBA2DE::getVersion()` devuelve `0.9.8`.

Si en Linux fallan tests **fuera** de `[XBA2DE]` y `[XSP2DE]`, casi seguro no
es de esta rama: comprobar primero con `git stash` / checkout de `main`
(`d64e64a`) en la misma máquina. Recordar que `analisis_portabilidad_tests.md`
documenta que los desempates de `std::sort` entre libc++ y libstdc++ ya
movían golden values entre plataformas antes de este trabajo.

---

## 5. Cómo reproducir

```bash
# activar conda antes (ver memoria del proyecto: conda-for-conan-builds)
make buildd
cd build_Debug/tests
./TestBayesNet "[XBA2DE]"
./TestBayesNet "[XSP2DE]"
./TestBayesNet                       # suite completa
./TestBayesNet "Working memory budget" 2>&1 | grep -a GOLDEN   # imprime nodos/notas de la run limitada
```

Salida esperada del `GOLDEN` en macOS:

```
GOLDEN[Memory-limited] nodes=90 edges=216 states=2187 notes=3 || Memory limit reached: 9 models built, 0.10 MiB used of 0.12 MiB budget || Pairs not used in train: 27 || Number of models: 9
```

**Trampa de coverage**: tras recompilar, los `.gcda` viejos del build Debug
corrompen la ejecución con cientos de líneas `profiling: ... cannot merge
previous GCDA file`. Es ruido, no un fallo de test, pero oculta la salida.
Limpiar antes de correr:

```bash
find build_Debug -name "*.gcda" -delete
```

---

## 6. Pendiente / ideas fuera del alcance de esta rama

- Dataset donde algún valor falte en el fold de entrenamiento, para que el
  test de la cota superior verifique de verdad la parte "states del fold ≤
  states de la ensemble".
- Generalizar a `Boost` (`XBAODE` tiene el mismo problema con `XSpode`)
  cuando haya un segundo consumidor; exigiría un hook virtual "estima el
  siguiente modelo".
- El profiler `benchmark/xba2de_profile` acepta `--hyper JSON`, así que
  `--hyper '{"max_memory_gb": 1.0}'` sirve para medir en datasets grandes sin
  tocarlo. Sería el sitio para comprobar que el presupuesto se respeta de
  verdad (RSS del proceso vs `Y MiB budget` de la nota).
