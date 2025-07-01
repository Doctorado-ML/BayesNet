# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2025-06-30

### Internal

- Add docs generation to CMakeLists.txt.
- Add new hyperparameters to the Ld classifiers:
  - *ld_algorithm*: algorithm to use for local discretization, with the following options: "MDLP", "BINQ", "BINU".
  - *ld_proposed_cuts*: number of cut points to return.
  - *mdlp_min_length*: minimum length of a partition in MDLP algorithm to be evaluated for partition.
  - *mdlp_max_depth*: maximum level of recursion in MDLP algorithm.
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
