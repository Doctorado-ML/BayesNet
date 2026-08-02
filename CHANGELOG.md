# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Build

- Fix `conandata.yml`. Every entry was fictional: the `sha256` fields were the literal string `placeholder_sha256`, and the `github.com/rmontanana/BayesNet` archive URLs all return 404 (including `1.1.2`, a version that was never tagged). Entries now point at the Gitea origin and carry real, verified hashes for 1.0.7, 1.1.0, 1.2.1, 1.2.2, 1.2.3 and 1.3.0. Note that the file is still reference metadata only: `conanfile.py` packages from `exports_sources` and has no `source()` method, so nothing here is fetched during `conan create`.

## [1.3.0] - 2026-08-02

### Added

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

- **XSPODE** and **XSP2DE**: `predict_proba` now works in log-space, summing log-probabilities and normalizing with log-sum-exp instead of multiplying probabilities and rescaling by a `DBL_MAX / nFeatures^2` constant. This removes the underflow that the old `initializer_` member papered over — relevant because SP2DE's 4-way child tables `p(x|c,sp1,sp2)` are very sparse — and makes both flat-count base classifiers numerically identical. ORIGINAL, LAPLACE and NONE are numerically unchanged.
- **XBA2DE**: feature-selection seeding (CFS/IWSS/FCBF) removed; the ensemble always starts empty and grows only through the boosting loop. This was the sole source of the C(k,2) base-model explosion on high-dimensional datasets. `select_features` and `threshold` are now rejected by `setHyperparameters`.
- **XBA2DE**: bisection is always on, and `block_update` / `alpha_block` are gone. `bisection`, `block_update` and `alpha_block` are now rejected by `setHyperparameters`. The remaining valid hyperparameters are `order`, `convergence`, `convergence_best`, `maxTolerance`, `predict_voting`, `weightless` and `beta`.

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
