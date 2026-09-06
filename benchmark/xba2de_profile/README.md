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

Two sources, both deterministic.

**Synthetic** (`--synthetic N,M,CARD`): each feature agrees with the class 30 %
of the time and is uniform noise otherwise, fixed seed. Useful because n and m
move independently, which is what exposes the O(n^2 * m) growth of the ranking.

**Real** (`--arff NAME[,MAX]`): a dataset from `tests/data`, MDLP discretized.
The class column comes from `tests/data/all.txt`, not from position — see the
`kdd_JapaneseVowels` note in `../mdlp_compare/README.md`: it declares `speaker`
first, and loading it class-last picks a REAL feature as the class and yields
hundreds of "classes". The catalog also supplies the numeric-feature mask, so
nominal columns are cast rather than discretized.

MDLP is fitted on all the data: the profiler wants one fixed discrete dataset
to time, so the accuracy it reports is a resubstitution figure, used as a change
detector and not as a quality measure. `MAX` subsamples with a fixed seed over
the whole file rather than taking a prefix, since several of these datasets are
sorted by class.

## Results

Measured on `feat/fimdlp-3.0.0` with and without the optimization commits, so
both columns share the same dependency set and the same deterministic
tie-breaking. macOS arm64, Release, 1 torch thread. Accuracy and model count
were identical in every case.

### Real datasets — one pair ranking

| dataset | features | samples | pairs | before | after | gain |
|---|---:|---:|---:|---:|---:|---:|
| iris | 4 | 150 | 6 | 0.004 s | 0.0001 s | 51x |
| liver-disorders | 6 | 345 | 15 | 0.022 s | 0.0002 s | 140x |
| ecoli | 7 | 336 | 21 | 0.031 s | 0.0002 s | 129x |
| diabetes | 8 | 768 | 28 | 0.104 s | 0.0007 s | 141x |
| glass | 9 | 214 | 36 | 0.034 s | 0.0003 s | 106x |
| heart-statlog | 13 | 270 | 78 | 0.094 s | 0.0006 s | 152x |
| kdd_JapaneseVowels | 14 | 9 961 | 91 | 4.227 s | 0.031 s | 138x |
| letter | 16 | 20 000 | 120 | 10.788 s | 0.133 s | 81x |
| spambase | 57 | 4 601 | 1 596 | 31.984 s | 0.104 s | 307x |
| **mfeat-factors** | **216** | **2 000** | **23 220** | **209.3 s** | **2.4 s** | **87x** |

`mfeat-factors` is the case the algorithm was meant for and the one it could
not serve: 216 features, and the boosting loop pays a full ranking every round.

### Real datasets — complete fit

| dataset | before | after | gain | models |
|---|---:|---:|---:|---:|
| iris | 0.021 s | 0.003 s | 6x | 6 |
| glass | 0.186 s | 0.007 s | 29x | 17 |
| ecoli | 0.266 s | 0.013 s | 20x | 21 |
| liver-disorders | 0.111 s | 0.008 s | 13x | 1 |
| diabetes | 0.474 s | 0.015 s | 32x | 1 |
| heart-statlog | 0.422 s | 0.010 s | 41x | 1 |
| kdd_JapaneseVowels | 186.8 s | 6.6 s | 28x | 91 |

### Synthetic

| case | SelectKPairs before | after | gain | full fit before | after | gain |
|---|---:|---:|---:|---:|---:|---:|
| n=20 m=2 000 | 1.724 s | 0.011 s | 163x | 45.06 s | 0.59 s | 76x |
| n=20 m=5 000 | 4.456 s | 0.023 s | 192x | 123.89 s | 2.16 s | 57x |

The line `~N pair rankings' worth of time` in the output is the headline: it
says how much of a fit is spent ranking pairs rather than training models. It
goes *up* after the optimization, because what is left is dominated by the
ranking the boosting loop still repeats every round.
