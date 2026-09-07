# fimdlp 2.1.3 vs 3.0.0 comparison

Harness to check whether upgrading the discretization library (`fimdlp` /
`mdlp`) from **2.1.3** (the version BayesNet currently pins in `conanfile.py`)
to **3.0.0** changes any result, and what it costs or saves in time.

## What it does

`mdlp_bench.cc` is compiled **twice from the very same source**, once against
each `fimdlp` version. BayesNet itself is compiled from source into each
binary, so the whole library — not just the benchmark — is built against the
version under test. Each binary then runs the same experiment over the
datasets in `tests/data` and writes a JSON report; `compare.py` diffs the two.

Two models are exercised, because they use mdlp in different ways:

| Model | How mdlp is used | What the timing isolates |
|---|---|---|
| `TAN` | Global discretization done in the harness with `mdlp::CPPFImdlp`, fitted on the train fold only and applied to both folds | `discretize_ms` is **pure mdlp**; `fit_ms` is pure BayesNet |
| `TANLd` | Continuous data handed to `TANLd`, which discretizes internally (iterative local discretization, `Proposal`) | `fit_ms` mixes mdlp and BayesNet, which is the realistic end-to-end cost |

Evaluation is a stratified k-fold cross validation (5 folds, seed 271 by
default) — the same `folding::StratifiedKFold` the test suite uses. The
discretizers are fitted on the train fold only, so the reported accuracies are
honest and every fold exercises `n_features` MDLP fits.

Besides accuracy and timings, each run records:

- the **MDLP cut points of fold 0** for every numeric feature, which is the
  direct check of whether the two versions discretize identically;
- the number of **nodes, edges and states** of the network learnt on every
  fold, which catches structural differences that accuracy might hide.

## Running it

Requires the two versions in the local Conan cache
(`conan list "fimdlp/*"` should show `2.1.3` and `3.0.0`). Remember to activate
conda first, as with the rest of the project's Conan builds.

```bash
cd benchmark/mdlp_compare
./run_comparison.sh                       # build both, run both, compare
./run_comparison.sh --reps 3              # 3 timing repetitions, fastest kept
./run_comparison.sh --datasets iris,glass # a subset
./run_comparison.sh --max-samples 2000    # cap huge datasets
```

Outputs, all in this directory:

- `results_2.1.3.json`, `results_3.0.0.json` — raw per-dataset results
- `REPORT.md` — the comparison tables and the verdict

The comparison can also be re-run on its own:

```bash
python3 compare.py results_2.1.3.json results_3.0.0.json --markdown REPORT.md
```

## Benchmark options

```
--data <path>       directory holding the .arff files
--output <file>     json file to write the results to
--datasets a,b,c    comma separated list (default: every .arff in all.txt found on disk)
--models TAN,TANLd  comma separated list
--folds <n>         number of folds (default 5)
--seed <n>          seed for the stratified folds (default 271)
--reps <n>          timing repetitions, the fastest one is kept (default 1)
--max-samples <n>   seeded subsample cap per dataset (default 0 = all)
--threads <n>       torch threads (default 1, for reproducible timings)
```

Timings use the fastest of `--reps` repetitions, which is the measure least
polluted by scheduler noise. Accuracies always come from the first repetition
and are deterministic, so any difference between the two versions is a real
behavioural difference, not run-to-run variance.

## Side finding: `Proposal::prepareX` used the training data at predict time

`heart-statlog` is the only dataset here whose features are not all numeric
(`all.txt` marks `[0,3,4,7,9,11]`), and every `TANLd` fold on it used to throw:

```
The expanded size of the tensor (54) must match the existing size (216)
```

`Proposal::prepareX` copied the **training** matrix for the features it does
not have to discretize:

```cpp
Xtd.index_put_({ i }, Xf[i].to(torch::kInt32));   // Xf is the train data
```

while the numeric branch two lines above correctly used the `X` being
predicted. Predicting on a set of a different size threw; predicting on a
same-sized set silently used the wrong values. It affected every `Proposal`
user (`TANLd`, `KDBLd`, `SPODELd`, `AODELd`) on mixed numeric/nominal datasets,
and reproduced identically on both mdlp versions, so it had nothing to do with
the upgrade.

Fixed to read from `X`, with a regression test in `tests/TestBayesClassifier.cc`
(`"Ld models predict on datasets with categorical features"`) that covers both
failure modes. The commented-out `heart-statlog` cases in
`tests/TestBayesModels.cc:523-540` are the reason it went unnoticed.

## Side finding: the class column is not always the last one

`kdd_JapaneseVowels.arff` declares `speaker` as its **first** attribute. Loading
it with `class_last = true` — the default, and what `tests/TestUtils.h`
(`RawDatasets`) still does — takes `coefficient12`, a REAL, as the class:
hundreds of "classes", enormous CPTs (a 200 s fit on 2000 samples instead of
0.5 s) and an accuracy of exactly 0. This benchmark takes the class name from
the third column of `all.txt` instead, via `ArffFiles::load(file, className)`.

## Notes

- `mfeat-factors` (216 features) and `letter` (20 000 samples) dominate the
  wall clock, especially with `TANLd`, whose iterative local discretization
  refits the discretizers on every iteration. Use `--max-samples` or
  `--datasets` while iterating.
- The benchmark's own mdlp calls compile unchanged against both versions:
  3.0.0 keeps the 2.1.3 API (`fit(samples_t&, labels_t&)`, `transform`,
  `getCutPoints`) and only adds to it (config structs, move overloads,
  `discretize` helpers, typed exceptions).
- `arff-files` is the one dependency that is *not* common to both: 2.1.3 pins
  `arff-files/1.2.1` and 3.0.0 pins `arff-files/2.0.0`, and the two conflict in
  a Conan graph. So `conanfile.py` derives the pin from `mdlp_version`, and
  `mdlp_bench.cc` supports both APIs behind `ARFF_V2` (2.0.0 moved the header
  to `ArffFiles/ArffFiles.hpp` and wrapped everything in `namespace ArffFiles`).
  `CMakeLists.txt` defines `ARFF_V2` from the version `find_package` resolved,
  so nothing has to be passed by hand.
