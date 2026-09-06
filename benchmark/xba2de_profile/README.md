# XBA2DE profiler

Per-stage timing breakdown of XBA2DE training, used to measure optimizations
against a recorded baseline instead of guessing.

## Build

```bash
cmake -S ../.. -B ../../build_Release -DENABLE_BENCHMARK=ON
cmake --build ../../build_Release --target xba2de_profile --parallel
```

(Conan lives in miniconda here, so prefix with
`source ~/miniconda3/etc/profile.d/conda.sh && conda activate`.)

## Use

```bash
./run.sh baseline            # record results/baseline.json + baseline-scaling.json
# ... change the code, rebuild ...
./run.sh h1-accessor         # record the new run
./compare.py results/baseline.json results/h1-accessor.json
```

`compare.py` prints a per-stage speedup **and** checks the invariants: accuracy
and model count must be identical for any change that claims to preserve
semantics. It exits non-zero and prints `SEMANTIC CHANGES DETECTED` otherwise.

Individual runs:

```bash
xba2de_profile --synthetic 40,5000,4 --reps 3
xba2de_profile --synthetic 100,50000,4 --skip-full   # ranking cost only
xba2de_profile --hyper '{"maxTolerance":1}' --synthetic 20,2000,4
```

## What is measured

| stage | what it is |
|---|---|
| `primitives_ms` | the `Metrics` kernels every pair score is built from |
| `select_k_pairs` | one full pair ranking — the boosting loop calls this **once per round** |
| `xsp2de` | fitting and predicting a single pair model |
| `xba2de` | the complete ensemble fit, plus accuracy and model count |

The line `~N pair rankings' worth of time` is the headline number: it says how
much of a full fit is spent ranking pairs rather than training models.

## Data

Synthetic only, deliberately. Each feature agrees with the class 30 % of the
time and is uniform noise otherwise; the seed is fixed, so runs are
byte-identical and accuracy is a usable invariant.

Real ARFF cases are not wired up because the test-suite loader
(`tests/TestUtils.cc`) is written against the arff-files 1.x API while Conan
resolves 2.0.0, which renamed the header and moved everything into an
`ArffFiles` namespace. That breaks `make debug` / `make test` too, and is
tracked separately from this profiler.

## Baseline (2026-09-06, 1.3.0, M-series, 1 torch thread)

| case | SelectKPairs | XSp2de fit | full fit | ranking share |
|---|---:|---:|---:|---:|
| n=20 m=2000 | 1.58 s | 0.022 s | 40.0 s | ~25 rankings |
| n=20 m=5000 | 4.28 s | 0.054 s | 114.4 s | ~27 rankings |
| n=40 m=5000 | 17.7 s | 0.104 s | — | — |
| n=80 m=5000 | 72.5 s | 0.245 s | — | — |
| n=40 m=20000 | 72.0 s | 0.436 s | — | — |

Ranking cost grows as O(n²·m) and is ~98 % of training time. Within it,
`mutualInformation` (5.1–5.7 ms) is ~50× the 3-argument `conditionalEntropy`
(0.07–0.11 ms) despite doing less work — the 2-argument `conditionalEntropy`
indexes tensors element-by-element with `.item()`.
