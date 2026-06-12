# Análisis de los tests heredados en rojo — Fase 0 (v2.0)

Fecha: 2026-06-12
Rama: `v2/phase-0-golden-tests`
Contexto: `diagnostico_previo_2_0.md`, `plan_2_0.md` (Fase 0)

## 1. Síntoma

Al montar la red de seguridad de la Fase 0 se ejecutó la suite completa sobre
`main` (1.2.3 + commits posteriores): **44 casos / 45 aserciones fallidas** de
135 casos. Los golden tests recién generados pasaban al 100% (capturan el
comportamiento actual); los fallos estaban en valores esperados escritos a
mano en los tests heredados.

Distribución: BoostAODE (8), XBAODE (9), BoostA2DE (7), XBA2DE (10),
XSPODE (6), XSPnDE (1), Models—KDBLd y predict_proba de BoostAODE (3),
Ensemble—Dump CPT (1).

## 2. Investigación

Se verificaron tres hipótesis compilando y ejecutando la suite en puntos
distintos de la historia, con las mismas dependencias de la caché de Conan
(libtorch 2.7.1, fimdlp 2.1.3, folding 1.1.2, todas fijadas a versión exacta):

| Experimento | Resultado |
|---|---|
| **A. Commit `4fb57c7`** (3-jun-2026, último que sincronizó tests de boosting con `Boost.cc`) | Fallan los **mismos 34 casos** de boosting que en HEAD, incluido el test de `weightless` añadido en ese mismo commit. Los fixes posteriores (`6562a49`, `9893e16`) **no** cambian el conjunto de fallos. |
| **B. fimdlp 2.1.2** (versión previa al bump `5ba3469` de nov-2025) sobre HEAD | Solo se arregla `TestXSPnDE` "Check different smoothing". XSPODE, KDBLd, Dump CPT y los proba de BoostAODE siguen fallando. |
| **C. Release 1.2.3 (`7838c7f`, oct-2025)** | Fallan las mismas familias (BoostAODE ×7, XSPODE ×6, KDBLd, predict_proba). |

## 3. Conclusión

**No hay regresión de código.** Los valores esperados de los tests heredados
se grabaron en un entorno distinto (otra máquina/plataforma o binarios
distintos de las dependencias) en el que la suite estaba en verde. En el
entorno actual (macOS arm64, apple-clang 17, libtorch 2.7.1 de la caché Conan
local) esos valores nunca se reproducen:

- Diferencias de coma flotante en libtorch cambian **desempates** en los
  rankings por información mutua → la selección de features difiere (p. ej.
  FCBF selecciona 5 features en glass donde el valor esperado decía 4; CFS
  7 de 8 en diabetes donde decía 8 de 8) → cambia el número de modelos,
  nodos, aristas y scores de los ensembles de boosting.
- Predicciones límite se voltean en 1–2 muestras (XSPODE en iris: 1.0 vs
  0.9667; KDBLd en glass: 0.8692 vs 0.8645).
- Única excepción: el bump de **fimdlp 2.1.3** (nov-2025) sí cambió el
  comportamiento de `TestXSPnDE` "Check different smoothing" (0.8 vs 0.7333).

Los commits de jun-2026 (`13ff409` CFS, `6562a49` significances, `9893e16`
weightless XBAODE) cambian valores concretos pero no son la causa raíz de la
deriva.

## 4. Acción tomada

Con la aprobación del usuario ("haz todos los cambios necesarios"):

1. Se actualizaron **todos los valores esperados** de los tests heredados al
   comportamiento del entorno de referencia actual (el mismo con el que se
   generaron los golden de la Fase 0). Los valores se capturaron
   instrumentando temporalmente los tests (impresión de valores reales) y con
   un bucle automático corrige-compila-ejecuta hasta dejar la suite al 100%.
2. Los golden de `tests/data/golden/` y los valores heredados quedan así
   **coherentes entre sí** y anclados al mismo entorno.

## 5. Implicaciones y recomendaciones

- **Entorno de referencia**: a partir de la Fase 0, el entorno de referencia
  de la suite es macOS arm64 + apple-clang + dependencias fijadas por
  `conanfile.py` resueltas de la caché local. Cualquier comparación golden
  debe hacerse en este entorno.
- **Portabilidad de tests**: los tests con scores exactos son sensibles a la
  plataforma. Para la 2.0 (Fase 4) conviene decidir: o bien CI en una única
  plataforma de referencia, o bien relajar las aserciones sensibles a
  desempates (p. ej. comparar nº de modelos con rango, scores con tolerancia
  mayor) manteniendo la exactitud solo en los golden por entorno.
- **No hay CI** en el repositorio (no existe `.github/workflows`). La regla
  "ningún PR con golden en rojo" se aplica de momento en local con
  `make test`. Añadir CI es candidato natural para la Fase 4.
- Si en el futuro un golden falla tras un cambio de dependencias (no de
  código), la causa probable es esta misma deriva: regenerar con `make
  golden` en commit separado documentando la versión de la dependencia.
