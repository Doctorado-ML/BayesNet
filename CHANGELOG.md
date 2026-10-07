# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `mutualInformation` returned a tiny non-zero value on arm64 for a feature that carries no information. `MI(X, Y) = H(X) - H(X|Y)`, and when `Y` has a single state `H(X|Y)` has to come out as *exactly* `H(X)` — but `entropy()` sums through ATen's `bincount` while `conditionalEntropy()` sums a dense table in a sequential loop, two implementations of the same quantity that are subtracted from one another. Their agreement was an accident of the build: bit-identical on x86_64/libstdc++, not on arm64, where a ~1e-18 residue survived the `std::max(..., 0.0)` clamp. That is decisive rather than negligible, because every edge of a single-state feature ties at exactly zero and the tie-break then defines the result: it is why glass's maximum spanning tree still differed between platforms after the tie-break was made total. `conditionalEntropy` now states the two degenerate identities outright — `H(X|Y) = 0` for constant `X`, `H(X|Y) = H(X)` for constant `Y`, the latter returning the very call that will be subtracted — so the difference is zero by construction on any platform. No value moves on Linux, and the cost is not measurable (`[XBA2DE]` runs in 31.45 s with the change against 31.46 s without).
- Make the test suite platform independent. Verified: 2120 assertions in 139 test cases, green on both Linux x86_64/libstdc++ and macOS arm64/libc++, with the same assertion count on each. Nine assertions failed on Linux/libstdc++ with the values the suite carries, which were generated on macOS/libc++, and the cause was `std::shuffle` in the test helper `ShuffleArffFiles`: the standard does not specify the algorithm, so libstdc++ and libc++ build different permutations from the same `mt19937{173}` seed (verified side by side on one machine — first indices 609 547 894 399 … against 229 782 610 828 …). Every test that subsamples with `shuffle=true` was therefore training on a *different set of rows* per platform, which is all seven `Bisection Best` / `Bisection Best vs Last` cases of `BoostAODE`, `XBAODE`, `BoostA2DE` and `XBA2DE`. It now uses the same specified Fisher-Yates that `folding` 2.0.0 adopted to fix the identical problem for the folds. The same `std::shuffle` was in the library itself, behind `order = "rand"` in all four Boost ensembles, and is replaced by the new `bayesnet::deterministicShuffle`.
- `Mst::kruskal_algorithm` ordered equal-weight edges by `stable_sort` alone, which made the spanning tree depend on the order `addEdge` happened to be called in rather than on the data. On glass, feature `Si` is left with a single state by MDLP, so its eight edges all score exactly 0 and whichever came first entered the tree. The comparator is now a total order (weight descending, then the `(u, v)` endpoints).
- `TAN::buildModel` picked the root with a `sort` that compared mutual information only, so a tie at the maximum was resolved by the standard library. Equal values are now broken by feature index.
- `KDB::add_m_edges` chose the next parent with `torch::argmax`, which is documented to return the first maximal index but decides that through its reduction strategy; rows of the conditional-mutual-information matrix tie in whole blocks at 0 for the same single-state reason. Replaced by an explicit first-maximum scan.
- `Proposal::prepareX` read the **training** matrix instead of the samples being predicted for the features it does not discretize. Only datasets with mixed numeric/nominal features reach that branch, so it affected every `Proposal` user (`TANLd`, `KDBLd`, `SPODELd`, `AODELd`) on such data, in `predict`, `predict_proba` and `score`. Predicting on a set whose sample count differs from the training one threw from torch; predicting on a same-sized set silently substituted the training values into the categorical columns and returned wrong answers. `heart-statlog` is the only dataset in `tests/data` that exercises this, and its cases in `tests/TestBayesModels.cc` are commented out, which is why it went unnoticed.

### Changed

- `KDBLd` on glass is the one score in the suite that is not reproducible across platforms to 1e-5 (x86_64 gets 185 of 214 samples right, arm64 186), so it is asserted within one sample instead of to five decimals. Nothing on that path is near a boundary: the MDLP margins of the local discretization (320 decisions, minimum 9.9e-05 relative against float32's 6e-8), the mutual-information ranking that orders KDB's nodes (minimum relative gap 1.2e-03), the 357 comparisons against `theta` (minimum 5.9e-03), and the score does not move under input perturbations up to 1e-3. Unlike the spanning-tree tie there is no canonical value to impose — local discretization is a fixed-point iteration and two models differing by one sample are equally valid — so the assertion now states what can actually be claimed. It is the only special-cased value in the suite.
- Two new cases pin the invariants the platform divergences broke, instead of pinning a value: `[Metrics]` "A constant feature has exactly zero mutual information" checks that `entropy`, `mutualInformation` in both argument orders, `conditionalMutualInformation` and the `conditionalEdge` entries of a single-state feature are exactly zero — compared with `==`, not `Approx` — and `[MST]` "The maximum spanning tree breaks weight ties by endpoints" checks that an all-equal weight matrix yields the canonical tree and that the result does not depend on the order edges are added in. Either would have caught the arm64 divergence without a golden value.
- Regenerate the expected values the determinism fixes above move, on Linux. The seven subsampling cases change because the subsample is now a different (and platform independent) set of rows: `BoostAODE` and `XBAODE` bisection reach 14 models instead of 5 (210 nodes), `BoostA2DE` 38 instead of 31 (570 nodes), `XBA2DE` 13 instead of 16 (195 nodes). The `rand` order of all four ensembles moves because the shuffle changed, and glass's maximum spanning tree now contains `{0, 4}` instead of `{3, 4}` — both are valid maximum spanning trees, the new one is the canonical pick under the total order. `KDBLd` on glass goes back to 0.864486 (185 of 214), the value the suite carried before it was regenerated on macOS; this is the one divergence left unexplained, see `analisis_portabilidad_tests.md` §7.ter.
- `BoostAODE`'s "weightless vs weighted" test now compares the two ensembles' predictions instead of their accuracies. On the new folds both reach 0.727 on diabetes, so the old `score_weightless != score_weighted` assertion no longer held even though the models genuinely differ — they disagree on 12 of the test samples, with a max probability difference of 0.31.
- `BoostA2DE`'s FCBF case no longer eliminates models on convergence, so it emits 3 notes instead of 4; the case asserts the note count explicitly now.
- Regenerate the nine expected values the `folding` 2.0.0 pass left behind, all in tests that ride on the new folds: the "Bisection Best" and "Bisection Best vs Last" cases of `BoostAODE`, `XBAODE`, `BoostA2DE` and `XBA2DE`, the `asc`/`rand` orders of `BoostA2DE`, and `KDBLd` on glass. The boosting loop now keeps more models before converging (`BoostAODE` 2 -> 5, `XBA2DE` 9 -> 16, `BoostA2DE` 4 -> 31), which is what moves the node and edge counts; the scores move within the same range and mostly up (`BoostAODE` bisection 0.9875 -> 0.9958, `BoostA2DE` bisection 0.9583 -> 0.9833, `BoostA2DE` asc 0.7523 -> 0.7897). These nine failed identically before and after the `arff-files` upgrade, so they are fold churn, not a regression from it.

### Known issues

- `Metrics::conditionalMutualInformation` is not symmetric, and `I(X;Y|C)` is symmetric by definition: all 6 pairs of iris and all 36 of glass disagree when the arguments are swapped, by up to 1.32. The four-argument `conditionalEntropy` indexes `keyJoint = (first, labels, second)` and `keyMarginal = (first, labels)`, so it accumulates `H(second | first, labels)` rather than the `H(first | second, labels)` its own comment describes, and `conditionalMutualInformation` subtracts that from `H(first | labels)`. The two valid forms of the identity are `H(X|C) - H(X|Y,C)` and `H(Y|C) - H(Y|X,C)`; this mixes one of each. On glass it is plain to see: `I(0;4|C)` comes out as 0.81, which is just `H(X0|C)`, because feature 4 is constant and `H(Si|X0,C) = 0`. Not fixed here — it is outside the portability work and correcting it moves `SelectKPairs`' pair ranking and with it the expected values of `BoostA2DE` and `XBA2DE`, and would invalidate the tie counts section 2 of `analisis_portabilidad_tests.md` measured on this function.

### Build

- Bump `fimdlp` to 3.0.1 and track it in `tests/TestModulesVersions.cc`. Discretization is unchanged: cut points are identical to 3.0.0 on all ten datasets in `tests/data`, so no expected value moves.
- Upgrade `arff-files` from 1.2.1 to 2.0.0, which `fimdlp/3.0.0` now requires — the two cannot coexist in the graph, so this was forced rather than optional. 2.0.0 moves the header to `ArffFiles/ArffFiles.hpp`, puts everything in `namespace ArffFiles` (`ArffFiles::ArffFiles` is the reader), makes the class move-only and keeps all its storage private. The last point is what actually needed work: `ShuffleArffFiles` in `tests/TestUtils.cc` used to subset the data by assigning to the protected `X`, `y` and `attributes` members, and now goes through the non-const `getX()`/`getY()`/`getAttributes()` getters. Parsing is unaffected: the feature matrix and labels 2.0.0 produces are byte-identical to 1.2.1 on all ten datasets in `tests/data`, so no expected value in the suite moves because of this.
- Bring `sample/` onto the same dependency set: `fimdlp/3.0.0`, `folding/2.0.0`, `arff-files/2.0.0`, plus `libtorch/2.7.1` and `bayesnet/1.3.0`. It had been left on `fimdlp/2.1.0`, `folding/1.1.1`, `arff-files/1.2.0`, `libtorch/2.7.0` and `bayesnet/1.2.0`, which no longer resolve together now that `fimdlp/3.0.0` requires `arff-files/2.0.0`. `sample.cc` and `sample_xspode.cc` move to the 2.0 header and namespace. Note that consuming the new set needs a `bayesnet/1.3.0` package built from this branch (`make conan-create`), since the previously cached one still declares the old dependencies.
- `benchmark/mdlp_compare` now selects its `arff-files` pin from the `mdlp_version` option, because each `fimdlp` version pins its own and they conflict. `mdlp_bench.cc` supports both APIs behind `ARFF_V2`, which `CMakeLists.txt` defines from the resolved package version, so the harness still compiles the same source against both `fimdlp` versions.
- Upgrade `folding` from 1.1.2 to 2.0.0. The API is source compatible — BayesNet compiles unchanged — but the folds a seed produces are different, which is the whole point of the release: 1.1.x shuffled with `std::shuffle`, which the standard only requires to be uniform, not to pick a particular permutation, so libstdc++ and libc++ built different folds out of the same seed and a cross-validation result was only reproducible on the platform that generated it. 2.0.0 does its own Fisher-Yates, so a seed now identifies the same fold everywhere. Two consequences for us: `folding.hpp` no longer includes `<torch/torch.h>` (its tensor constructor is a template constrained on `numel()`/`data_ptr<int>()`, and `libtorch` moved to `test_requires` in the recipe — harmless here, BayesNet requires `libtorch` directly); and `Boost::buildModel`'s internal validation split plus `RawDatasets`' train/test split both changed, so every expected value in the test suite that rides on those folds was regenerated. Scores move in both directions and stay in the same range — this is a different split, not a worse one. Pin `folding/1.1.3` to reproduce results generated with the 1.1.x folds.
- Upgrade `fimdlp` from 2.1.3 to 3.0.0. 3.0.0 keeps the 2.1.3 API (`fit(samples_t&, labels_t&)`, `transform`, `getCutPoints`) and only adds to it — config structs, move overloads, static `discretize` helpers, typed exceptions — so BayesNet compiles unchanged. Verified with `benchmark/mdlp_compare` over the 10 datasets in `tests/data` with TAN and TANLd: accuracies, MDLP cut points (342 features, max abs diff 0.0) and learnt networks are identical, while discretization time drops 89.5% overall (3516 ms to 368 ms), from -51% on iris to -93% on kdd_JapaneseVowels. End to end the gain is ~2%, since discretization is a small fraction of the total next to `fit`.
- Fix `conandata.yml`. Every entry was fictional: the `sha256` fields were the literal string `placeholder_sha256`, and the `github.com/rmontanana/BayesNet` archive URLs all return 404 (including `1.1.2`, a version that was never tagged). Entries now point at the Gitea origin and carry real, verified hashes for 1.0.7, 1.1.0, 1.2.1, 1.2.2, 1.2.3 and 1.3.0. Note that the file is still reference metadata only: `conanfile.py` packages from `exports_sources` and has no `source()` method, so nothing here is fetched during `conan create`.

## [1.3.0] - 2026-08-02

### Added

- **XBA2DE**: `max_memory_gb` hyperparameter, a working-memory budget for the ensemble in gigabytes (1 GB = 2^30 bytes; default 0.0 = unlimited). The boosting loop now estimates what the next `XSp2de` will hold before building it and stops when it no longer fits, which makes memory a first-class exit condition alongside pair exhaustion and convergence. The budget covers the tables of the accumulated `XSp2de` models only — not the dataset, the metrics or the pair ranking — and the check bounds the *peak* of the candidate (counts and probabilities both alive during `fit`) against the *resident* total of the models already kept, so the ensemble never exceeds the budget at any instant. Stopping this way pushes a `Memory limit reached: N models built, X MiB used of Y MiB budget` note and sets `status = WARNING`, mirroring how `Pairs not used in train` reports an ensemble that did not use every model available to it. A budget too small for even the first model throws `std::runtime_error` rather than returning an empty ensemble.
- **XSP2DE**: `memoryFootprint()` reports the bytes a fitted model holds, and the static `estimateFootprint(states, statesClass, sp1, sp2)` predicts the peak for a pair without building it. Fed the ensemble's `states` map, the estimate is a guaranteed upper bound: the model derives its own cardinalities from the training fold, whose per-feature maxima can only be smaller.
- **XA1DE**: optimized AODE ensemble built on the new header-only `Xaode` engine, which materializes every SPODE submodel as flat count arrays filled in a single pass instead of going through the `Network` representation.
- **XA2DE**: optimized A2DE ensemble built on the new header-only `Xaode2de` engine. It materializes all C(n,2) pair-superparent SP2DE submodels as flat count arrays and averages their per-pair posteriors in log-space with uniform significance. Its single-pair posterior matches `XSp2de` exactly, and the full model matches the average of `XSp2de` over all pairs.
- `ExpClf`, a shared base for the flat-table experimental ensembles.
- **XSP2DE**: joint superparent model `P(sp1,sp2|c)` (classic A2DE) as the default, behind the new `joint_parents` hyperparameter. The legacy independent factorization `P(sp1|c)*P(sp2|c)` remains available for ablation. `getNumberOfEdges` is mode-aware: 3n-4 independent, 3n-3 joint.
- **XBA2DE**: joint-relevance pair-selection criterion `I(Xi,Xj;C) = I(Xi;C) + I(Xj;C) + beta * (I(Xi;Xj|C) - I(Xi;Xj))`, exposed through the new `beta` hyperparameter (default 1.0). `beta` is the ablation axis: 0 = marginal-relevance sum, 1 = joint relevance, large = pure synergy.
- `BayesMetrics::SelectKPairs` gains a `beta` parameter. A negative `beta` (the default) keeps the legacy CMI-only ranking, so BoostA2DE is untouched. All four mutual-information terms come from the same 3-way (Xi,Xj,C) table, so the new criterion costs nothing extra.
- Add the `weightless` hyperparameter to BoostAODE and XBAODE. When enabled, the Boost ensemble never updates instance weights and every SPODE votes with the same significance (1.0), effectively disabling the AdaBoost reweighting.
- Local discretization implementation review reports.
- Conda and Conan setup in the devcontainer Dockerfile.

### Changed

- **XSP2DE**: `childCounts_` is released at the end of `computeProbabilities()`. It is the dominant block of the model (`states[sp1] * states[sp2] * statesClass * sum of the children cardinalities`) and nothing downstream reads it — every consumer goes through `childProbs_` — so a fitted model now holds roughly half of what it did, which is what lets `max_memory_gb` fit about twice as many models in the same budget. `to_string()` dumps `childProbs_` where it used to dump the counts; that also gives `TestXSPnDE`'s joint-vs-independent comparison real content, since the count tables are identical under both settings and the assertion rested on the printed `jointParents_` flag alone.
- **XSPODE** and **XSP2DE**: `predict_proba` now works in log-space, summing log-probabilities and normalizing with log-sum-exp instead of multiplying probabilities and rescaling by a `DBL_MAX / nFeatures^2` constant. This removes the underflow that the old `initializer_` member papered over — relevant because SP2DE's 4-way child tables `p(x|c,sp1,sp2)` are very sparse — and makes both flat-count base classifiers numerically identical. ORIGINAL, LAPLACE and NONE are numerically unchanged.
- **XBA2DE**: feature-selection seeding (CFS/IWSS/FCBF) removed; the ensemble always starts empty and grows only through the boosting loop. This was the sole source of the C(k,2) base-model explosion on high-dimensional datasets. `select_features` and `threshold` are now rejected by `setHyperparameters`.
- **XBA2DE**: bisection is always on, and `block_update` / `alpha_block` are gone. `bisection`, `block_update` and `alpha_block` are now rejected by `setHyperparameters`. The remaining valid hyperparameters are `order`, `convergence`, `convergence_best`, `maxTolerance`, `predict_voting`, `weightless`, `beta` and `max_memory_gb`.

### Fixed

- **XSPODE** and **XSP2DE**: `CESTNIK` smoothing was a silent no-op — neither class had a `CESTNIK` case in its smoothing switch, so selecting it fell through to no smoothing at all. It is now a proper m-estimate (m=1, uniform prior) with a per-cell pseudocount of 1/K, where K is the cardinality of the distributed variable, matching the `Network` path. `ORIGINAL` and `LAPLACE` are byte-for-byte unchanged.
- **XBA2DE**: boost without replacement of pairs. The boosting unit in XBA2DE is the pair, not a single superparent, so used *pairs* are now tracked and excluded from future selection (a used pair (A,B) is excluded, but A and B may still pair with other variables). Previously no pairs were tracked at all, so a pair could be re-selected across packs and `SelectKPairs` never emptied, leaving convergence as the only stop condition. Pair exhaustion is now a natural termination criterion.
- Correct the model significance update in BoostAODE when feature selection was used.
- Improve the stopping criterion in the CFS feature selection algorithm.

### Build

- Install `*.hpp` headers as well as `*.h`. `XA1DE.h` and `XA2DE.h` include the header-only flat engines (`Xaode.hpp`, `Xaode2de.hpp`), and the install rule only matched `*.h`, so consumers of the packaged library could not find them.

### Internal

- Add AI agent definitions.
- Add a v2.0 diagnostic report and execution plan.
- Ignore Node/npm artifacts, coverage artifacts and local editor state; stop tracking `.claude/settings.local.json`.

## [1.2.3] - 2025-10-20

### Fixed

- Fix issue in joining the local discretization values of the label and the fathers of a node, not using a separator caused issues when the states of the features contained the same values as the label. i.e. (1,23) and (12,3) both resulted in (123) when joined without a separator.

## [1.2.2] - 2025-08-29

### Fixed

- Fixed an issue with local discretization that was discretizing all features wether they were numeric or categorical.
- Fix testutils to return states for all features:
  - An empty vector is now returned for numeric features.
  - Categorical features now return their unique states.

## [1.2.1] - 2025-07-19

### Internal

- Update Libtorch to version 2.7.1
- Update libraries versions:
  - mdlp: 2.1.1
  - Folding: 1.1.2
  - ArffFiles: 1.2.1

## [1.2.0] - 2025-07-08

### Internal

- Add docs generation to CMakeLists.txt.
- Add new hyperparameters to the Ld classifiers:
  - *ld_algorithm*: algorithm to use for local discretization, with the following options: "MDLP", "BINQ", "BINU".
  - *ld_proposed_cuts*: number of cut points to return.
  - *mdlp_min_length*: minimum length of a partition in MDLP algorithm to be evaluated for partition.
  - *mdlp_max_depth*: maximum level of recursion in MDLP algorithm.
  - *max_iterations*: maximum number of iterations of discretization-build model loop.
  - *verbose_convergence*: display status messages during the convergence process.
- Remove vcpkg as a dependency manager, now the library is built with Conan package manager and CMake.
- Add `build_type` option to the sample target in the Makefile to allow building in *Debug* or *Release* mode. Default is *Debug*.

## [1.1.1] - 2025-05-20

### Internal

- Fix CFS metric expression in the FeatureSelection class.
- Fix the vcpkg configuration in building the library.
- Fix the sample app to use the vcpkg configuration.
- Refactor the computeCPT method in the Node class with libtorch vectorized operations.
- Refactor the sample to use local discretization models.

### Added

- Add predict_proba method to all Ld classifiers.
- Add L1FS feature selection methods to the FeatureSelection class.

## [1.1.0] - 2025-04-27

### Internal

- Add changes to .clang-format to adjust to vscode format style thanks to <https://clang-format-configurator.site/>
- Remove all the dependencies as git submodules and add them as vcpkg dependencies.
- Fix the dependencies versions for this specific BayesNet version.

## [1.0.7] 2025-03-16

### Added

- A new hyperparameter to the BoostAODE class, *alphablock*, to control the way &alpha; is computed, with the last model or with the ensmble built so far. Default value is *false*.
- A new hyperparameter to the SPODE class, *parent*, to set the root node of the model. If no value is set the root parameter of the constructor is used.
- A new hyperparameter to the TAN class, *parent*, to set the root node of the model. If not set the first feature is used as root.
- A new model named XSPODE, an optimized for speed averaged one dependence estimator.
- A new model named XSP2DE, an optimized for speed averaged two dependence estimator.
- A new model named XBAODE, an optimized for speed BoostAODE model.
- A new model named XBA2DE, an optimized for speed BoostA2DE model.

### Internal

- Optimize ComputeCPT method in the Node class.
- Add methods getCount and getMaxCount to the CountingSemaphore class, returning the current count and the maximum count of threads respectively.

### Changed

- Hyperparameter *maxTolerance* in the BoostAODE class is now in [1, 6] range (it was in [1, 4] range before).

## [1.0.6] 2024-11-23

### Fixed

- Prevent existing edges to be added to the network in the `add_edge` method.
- Don't allow to add nodes or edges on already fiited networks.
- Number of threads spawned
- Network class tests

### Added

- Library logo generated with <https://openart.ai> to README.md
- Link to the coverage report in the README.md coverage label.
- *convergence_best* hyperparameter to the BoostAODE class, to control the way the prior accuracy is computed if convergence is set. Default value is *false*.
- SPnDE model.
- A2DE model.
- BoostA2DE model.
- A2DE & SPnDE tests.
- Add tests to reach 99% of coverage.
- Add tests to check the correct version of the mdlp, folding and json libraries.
- Library documentation generated with Doxygen.
- Link to documentation in the README.md.
- Three types of smoothing the Bayesian Network ORIGINAL, LAPLACE and CESTNIK.

### Internal

- Fixed doxygen optional dependency
- Add env parallel variable to Makefile
- Add CountingSemaphore class to manage the number of threads spawned.
- Ignore CUDA language in CMake CodeCoverage module.
- Update mdlp library as a git submodule.
- Create library ShuffleArffFile to limit the number of samples with a parameter and shuffle them.
- Refactor catch2 library location to test/lib
- Refactor loadDataset function in tests.
- Remove conditionalEdgeWeights method in BayesMetrics.
- Refactor Coverage Report generation.
- Add devcontainer to work on apple silicon.
- Change build cmake folder names to Debug & Release.
- Add a Makefile target (doc) to generate the documentation.
- Add a Makefile target (doc-install) to install the documentation.

### Libraries versions

- mdlp: 2.0.1
- Folding: 1.1.0
- json: 3.11
- ArffFiles: 1.1.0

## [1.0.5] 2024-04-20

### Added

- Install command and instructions in README.md
- Prefix to install command to install the package in the any location.
- The 'block_update' hyperparameter to the BoostAODE class, to control the way weights/significances are updated. Default value is false.
- Html report of coverage in the coverage folder. It is created with *make viewcoverage*
- Badges of coverage and code quality (codacy) in README.md. Coverage badge is updated with *make viewcoverage*
- Tests to reach 97% of coverage.
- Copyright header to source files.
- Diagrams to README.md: UML class diagram & dependency diagram
- Action to create diagrams to Makefile: *make diagrams*

### Changed

- Sample app now is a separate target in the Makefile and shows how to use the library with a sample dataset
- The worse model count in BoostAODE is reset to 0 every time a new model produces better accuracy, so the tolerance of the model is meant to be the number of **consecutive** models that produce worse accuracy.
- Default hyperparameter values in BoostAODE: bisection is true, maxTolerance is 3, convergence is true

### Removed

- The 'predict_single' hyperparameter from the BoostAODE class.
- The 'repeatSparent' hyperparameter from the BoostAODE class.

## [1.0.4] 2024-03-06

### Added

- Change *ascending* hyperparameter to *order* with these possible values *{"asc", "desc", "rand"}*, Default is *"desc"*.
- Add the *predict_single* hyperparameter to control if only the last model created is used to predict in boost training or the whole ensemble (all the models built so far). Default is true.
- sample app to show how to use the library (make sample)

### Changed

- Change the library structure adding folders for each group of classes (classifiers, ensembles, etc).
- The significances of the models generated under the feature selection algorithm are now computed after all the models have been generated and an &alpha;<sub>t</sub> value is computed and assigned to each model.

## [1.0.3] 2024-02-25

### Added

- Voting / probability aggregation in Ensemble classes
- predict_proba method in Classifier
- predict_proba method in BoostAODE
- predict_voting parameter in BoostAODE constructor to use voting or probability to predict (default is voting)
- hyperparameter predict_voting to AODE, AODELd and BoostAODE (Ensemble child classes)
- tests to check predict & predict_proba coherence

## [1.0.2] - 2024-02-20

### Fixed

- Fix bug in BoostAODE: do not include the model if epsilon sub t is greater than 0.5
- Fix bug in BoostAODE: compare accuracy with previous accuracy instead of the first of the ensemble if convergence true

## [1.0.1] - 2024-02-12

### Added

- Notes in Classifier class
- BoostAODE: Add note with used features in initialization with feature selection
- BoostAODE: Add note with the number of models
- BoostAODE: Add note with the number of features used to create models if not all features are used
- Test version number in TestBayesModels
- Add tests with feature_select and notes on BoostAODE

### Fixed

- Network predict test
- Network predict_proba test
- Network score test
