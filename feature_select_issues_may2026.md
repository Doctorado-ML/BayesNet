# Informe de Revisión: Algoritmos de `bayesnet/feature_selection`

**Fecha:** 2026-05-08
**Alcance:** `FeatureSelect`, `CFS`, `FCBF`, `IWSS`, `L1FS`

A continuación se detallan las deficiencias detectadas, ordenadas por severidad.

---

## 1. `FeatureSelect.h` / `FeatureSelect.cc` (clase base)

### Bugs / Deficiencias graves
- **Orden de inicialización de miembros incorrecto** (`FeatureSelect.h:27-28` / `FeatureSelect.cc:23-25`): en el header `weights` se declara *antes* que `maxFeatures`, pero en la lista de inicialización del constructor se inicializa `maxFeatures` primero. C++ inicializa en orden de declaración, no de la lista. Genera warning `-Wreorder` y, lo más importante, hace pensar que `weights` puede usarse antes de inicializarse.
- **`weights` almacenado como `const torch::Tensor&`** (`FeatureSelect.h:27`): mantener referencia a un tensor externo es peligroso. Si quien construye la instancia destruye o modifica el tensor original, queda colgada. Debería almacenarse por valor (`torch::Tensor`, las copias son baratas porque `Tensor` ya es un wrapper con shared ownership).

### Deficiencias menores
- **Sin validación de `maxFeatures`**: valores negativos o mayores que `samples.size(0)-1` no se rechazan.
- **`symmetricalUncertainty`** (`FeatureSelect.cc:45`) acepta `b = -1` como "última fila = etiquetas", pero `computeSuFeatures` no lo controla y, si por error se llama con `-1`, se mete en la caché con clave negativa.
- **`computeMeritCFS`** (`FeatureSelect.cc:122`): la fórmula puede generar NaN si `k + k*(k-1)*rff_avg < 0` (en teoría no ocurre con SU ∈ [0,1] pero no hay defensa). Tampoco hay protección si el denominador resulta 0.
- **No se chequea `suLabels` constante / NaN** que se propagaría a la métrica.
- **`getFeatures()`/`getScores()` lanzan `runtime_error` por strings sin contexto**: añadir nombre de clase ayudaría a depurar.

---

## 2. `CFS.cc`

- **Violación potencial de `maxFeatures`** (`CFS.cc:18-22`): la primera *feature* se inserta antes del bucle sin comprobar `maxFeatures`. Si el usuario pide `maxFeatures = 1`, el bucle aún se ejecuta una vez y añade un segundo atributo. Conviene añadir una condición `selectedFeatures.size() < maxFeatures` antes del primer push, y/o evaluar `computeContinueCondition` antes de iterar.
- **Mezcla de escalas en `selectedScores`** (`CFS.cc:20,40`): el primer score guardado es la SU del mejor atributo individual; los siguientes son méritos CFS. La lógica de `computeContinueCondition` (que usa `selectedScores` para ver mejora en las últimas 5) compara magnitudes heterogéneas en las primeras iteraciones.
- **`continueCondition = true;` redundante** (`CFS.cc:17`): el `while` siempre entra; eliminar.
- **No se controla `featureOrder.empty()` antes del bucle interno**: si por algún motivo todas las *features* se han eliminado, el bucle no se ejecuta y se sale por `bestFeature == -1`, lo cual es correcto pero confuso.
- **`bestFeature` se localiza con `meritNew > merit`**: ante NaN (típico cuando una *feature* es constante) la comparación devuelve false; el algoritmo se rompe silenciosamente saliendo por el `break`. No se notifica el motivo.

---

## 3. `FCBF.cc`

- **Validación de `threshold` confusa** (`FCBF.cc:14-16`): el mensaje dice "Threshold cannot be less than 1e-7", pero técnicamente no rechaza `threshold == 1e-7`. Si la intención es exigir un valor estrictamente positivo, mejor `if (threshold <= 0)` con un mensaje claro.
- **`maxFeatures` no validado**.
- **Falta documentación del invariante**: el algoritmo asume `featureOrder` ordenado descendentemente por SU; si la implementación de `argsort` cambiase, el criterio de dominancia (`SU(fi,fj) >= SU(fj,class)`) deja de ser correcto. Podría blindarse con un `assert`.
- **Modifica `suLabels` para "marcar" *features* eliminadas** (`FCBF.cc:39`): es un efecto colateral sobre estado heredado. Más limpio: usar un `std::vector<bool>` local de descartados.
- **Bucle no permite reordenar tras descartes**: ortodoxo del algoritmo, pero conviene anotarlo.

---

## 4. `IWSS.cc`

### Bug serio
- **Cuando `secondFeature == -1`** (todas las parejas dan mérito 0): se entra al bucle principal con `merit = 0.0`. La condición `delta = merit != 0.0 ? ... : 0.0` deja `delta = 0`, por lo que `meritNew > merit || delta < threshold` se vuelve `0 < threshold` (verdadero para cualquier `threshold > 0`). El bucle **acepta todas las features restantes en orden hasta agotar `maxFeatures`**, lo que **no es el comportamiento esperado** (debería detenerse). Hace falta un caso especial cuando no se pudo seleccionar la segunda *feature*.

### Otras deficiencias
- **`maxMerit = 0.0` como inicial** (`IWSS.cc:30`): debería ser `std::numeric_limits<double>::lowest()` para no excluir candidatos legítimos con mérito 0.
- **`selectedScores` puede contener méritos menores que el máximo histórico**: línea 58 hace `push_back(meritNew)` aunque `meritNew <= merit`. No queda claro si la intención es que `selectedScores[i]` sea el mérito acumulado en el momento i o el "mejor hasta ahora".
- **`maxFeatures < 2` no controlado**: el algoritmo siempre intenta añadir 1ª y 2ª *feature*.
- **Sin documentación de la fórmula de `delta`**: se calcula como cambio relativo respecto a `merit`, no respecto a `meritNew`; conviene comentarlo.

---

## 5. `L1FS.cc` / `L1FS.h` (el más problemático)

### Bugs graves
- **Detección de tarea regresión vs. clasificación incorrecta** (`L1FS.cc:42`):
  ```cpp
  isRegression = (classNumStates > 2 || classNumStates == 0);
  ```
  Esto trata cualquier problema **multiclase (>2 clases) como regresión**, ejecutando Lasso sobre las etiquetas enteras. Es un error semántico mayor: las etiquetas no son ordinales en general. La interfaz tendría que admitir un `bool isRegression` explícito o, como mínimo, `classNumStates > 0` ⇒ clasificación (multinomial), `0` ⇒ regresión.
- **`logisticLoss` declarado en el header pero sin implementación** (`L1FS.h:79-80`): no se llama, pero su mera presencia es deuda técnica; cualquier referencia futura provoca error de enlazado.
- **Sombra de `weights`** (`L1FS.cc:134`, `L1FS.cc:201`):
  ```cpp
  torch::Tensor weights = sampleWeights.to(torch::kFloat32);
  ```
  Crea una variable local que oculta el miembro `weights` de la clase base. Confuso y propenso a errores en mantenimiento.
- **División por cero en Lasso** (`L1FS.cc:166`): si una feature queda completamente constante tras estandarizar (caso real cuando `X_std == 0` y se sustituye por 1), `featureNorms[j] = 0` y `coefficients[j] = softThreshold(...) / 0 = NaN/Inf`. Hay que detectar `featureNorms[j] == 0` y dejar el coeficiente en 0.
- **Inestabilidad en `sigmoid`** (`L1FS.cc:266-269`): `exp(-z)` desborda con `z` muy negativos. Debería usarse una versión estable.

### Problemas algorítmicos
- **Escala de regularización inconsistente**:
  - En Lasso (`L1FS.cc:166`): `softThreshold(rho, alpha)`, donde `rho` se calcula como suma ponderada (no media). Por tanto `alpha` está en unidades dependientes de `n_samples` y de la suma de pesos. No es lo que un usuario espera de "alpha".
  - En logística (`L1FS.cc:217, 224`): el gradiente sí se divide por `n_samples`, pero el threshold usa `learningRate * alpha`. Las dos rutas usan `alpha` con escalas distintas.
- **Tasa de aprendizaje fija + decaimiento ad-hoc** (`L1FS.cc:204, 243-245`): `learningRate = 0.01` y se reduce un 10% cada 100 iteraciones. Sin búsqueda de Lipschitz ni line search; convergencia frágil para problemas mal condicionados.
- **No se valida que las etiquetas para logística sean `{0,1}`**: si vienen como `{1,2}` (típico tras discretizaciones), el modelo se entrena con objetivos erróneos.
- **`weights.sum()` puede ser 0** y se usa como denominador en Lasso (`L1FS.cc:139, 178`).
- **Selección final usa `maxFeatures` por defecto (= todas las *features*)** (`L1FS.cc:111-112`): si el usuario deja el valor por defecto, devuelve todas las *features* con coeficiente no nulo, lo que puede ser muchas.
- **Fallback "mágico" a 3 features cuando todos los coef son 0** (`L1FS.cc:99`): el `min(..., 3)` arbitrario. Mejor usar `maxFeatures` directamente.

### Deficiencias menores
- **Variables no usadas**: `n_samples` en `fitLasso` (`L1FS.cc:126`).
- **`X = X.to(torch::kFloat32)`** se aplica siempre; afortunadamente solo se modifica una vista local, pero conviene revisar el flujo de tipos respecto a la base.
- **No se libera la caché `suFeatures` heredada** entre llamadas distintas. Sí lo hace `initialize()`, OK, pero conviene documentarlo.
- **No hay `numToSelect` cap por `maxFeatures > 0`** en la rama no-fallback: si `maxFeatures == 0`, el `std::min(..., maxFeatures)` da 0 y no devuelve nada.

---

## 6. Problemas transversales

- **Falta de validaciones de entrada**: ninguna clase verifica que `samples` no esté vacío, que `features.size() == samples.size(0) - 1`, ni que `weights.size(0) == samples.size(1)`.
- **Coste de memoria de `suFeatures`**: el caché crece a O(n²) entradas para conjuntos densos. En datasets de alta dimensionalidad puede ser problemático; convendría un `std::unordered_map` con hash de pareja, o computar bajo demanda sin caché en la primera pasada.
- **Sin paralelización**: bucles secuenciales sobre features incluso cuando son trivialmente paralelos (por ejemplo `computeSuLabels`).
- **Mensajes de excepción sin contexto** (qué clase, qué parámetro, qué valor recibió).
- **Los `.h` no incluyen `<map>`/`<string>` aunque sus tipos públicos los referencian** indirectamente vía la base; portabilidad frágil si cambian las cabeceras.
- **Documentación Doxygen ausente** salvo en L1FS.

---

## 7. Recomendaciones priorizadas

1. Corregir el **bug de IWSS** con `secondFeature == -1`.
2. Corregir el **criterio de regresión vs clasificación en L1FS** y eliminar `logisticLoss` o implementarlo.
3. Reordenar inicialización de miembros en `FeatureSelect` y/o cambiar `weights` a almacenamiento por valor.
4. Proteger división por cero en `fitLasso` y `weights.sum()` nulo.
5. Validar `maxFeatures`, dimensiones de `samples`/`features`/`weights` en todos los constructores.
6. Reescribir las llamadas `selectedFeatures.push_back(...)` previas al bucle de CFS para respetar `maxFeatures`.
7. Renombrar la variable local `weights` en L1FS para evitar la sombra del miembro.
8. Estandarizar la semántica de `selectedScores` (¿mérito en t o mejor mérito histórico?) y documentarla.

---

## Apéndice A: Justificación matemática de la fórmula CFS

En esta revisión **NO** se ha marcado como incorrecta la fórmula

```
Merit_S = (k * r_cf_avg) / sqrt(k + k * (k - 1) * r_ff_avg)
```

implementada en `FeatureSelect::computeMeritCFS` (`FeatureSelect.cc:122`). Si en otra revisión se sugirió cambiarla por

```
Merit_S = sum(r_cf) / sqrt(k + 2 * sum(r_ff))
```

ambas expresiones son **matemáticamente equivalentes**. Demostración:

Sea `k` el número de *features* seleccionadas. Se definen:

- `sum(r_cf) = Σ_i r_cf(i)` con `i = 1..k` ⇒ promedio: `r_cf_avg = sum(r_cf) / k`, por tanto **`sum(r_cf) = k * r_cf_avg`**.
- `sum(r_ff) = Σ_{i<j} r_ff(i,j)`, donde el número de parejas es `C(k,2) = k(k-1)/2` ⇒ promedio: `r_ff_avg = sum(r_ff) / [k(k-1)/2]`, por tanto **`sum(r_ff) = (k(k-1)/2) * r_ff_avg`**.

Sustituyendo en la versión "sumas":

```
sum(r_cf) / sqrt(k + 2 * sum(r_ff))
  = (k * r_cf_avg) / sqrt(k + 2 * (k(k-1)/2) * r_ff_avg)
  = (k * r_cf_avg) / sqrt(k + k(k-1) * r_ff_avg)
```

que es exactamente la versión "promedios" presente en el código. ∎

### Conclusión

- La fórmula original de Mark A. Hall (*"Correlation-based Feature Selection for Machine Learning"*, 1999, ec. 4.2) está expresada en términos de **promedios** (`r_cf_avg`, `r_ff_avg`) y es la que utiliza el código actual.
- La variante con **sumas** (`sum(r_cf)`, `sum(r_ff)`) es la misma cantidad reescrita; aparece a menudo en implementaciones que prefieren no calcular promedios explícitamente.
- Por tanto, la "corrección" propuesta en aquella otra revisión no era realmente una corrección, sino una **reformulación equivalente**. No hay diferencia numérica entre ambas.

Lo único que podría discutirse es la **estabilidad numérica** o la **eficiencia** (acumular sumas evita una división), pero en términos de corrección matemática las dos formas dan el mismo `Merit_S`.
