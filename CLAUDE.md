# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BayesNet is a C++ library implementing Bayesian Network Classifiers. It provides various algorithms for machine learning classification including TAN, KDB, SPODE, SPnDE, AODE, A2DE, and their ensemble variants (Boost, XB). The library also includes local discretization variants (Ld) and feature selection algorithms.

## Build System & Dependencies

### Dependency Management
- Uses **vcpkg** for package management with private registry at https://github.com/rmontanana/vcpkg-stash
- Core dependencies: libtorch, nlohmann-json, folding, fimdlp, arff-files, catch2
- All dependencies defined in `vcpkg.json` with version overrides

### Build Commands
```bash
# Initialize dependencies
make init

# Build debug version (with tests and coverage)
make debug
make buildd

# Build release version  
make release
make buildr

# Run tests
make test

# Generate coverage report
make coverage
make viewcoverage

# Clean project
make clean
```

### CMake Configuration
- Uses CMake 3.27+ with C++17 standard
- Debug builds automatically enable testing and coverage
- Release builds optimize with `-Ofast`
- Supports both static library and vcpkg package installation

## Testing Framework

- **Catch2** testing framework (version 3.8.1)
- Test executable: `TestBayesNet` in `build_Debug/tests/`
- Individual test categories can be run: `./TestBayesNet "[CategoryName]"`
- Coverage reporting with lcov/genhtml

### Test Categories
- A2DE, BoostA2DE, BoostAODE, XSPODE, XSPnDE, XBAODE, XBA2DE
- Classifier, Ensemble, FeatureSelection, Metrics, Models
- Network, Node, MST, Modules

## Code Architecture

### Core Structure
```
bayesnet/
├── BaseClassifier.h          # Abstract base for all classifiers
├── classifiers/             # Basic Bayesian classifiers (TAN, KDB, SPODE, etc.)
├── ensembles/              # Ensemble methods (AODE, A2DE, Boost variants)
├── feature_selection/      # Feature selection algorithms (CFS, FCBF, IWSS, L1FS)
├── network/               # Bayesian network structure (Network, Node)
└── utils/                 # Utilities (metrics, MST, tensor operations)
```

### Key Design Patterns
- **BaseClassifier** abstract interface for all algorithms
- Template-based design with both std::vector and torch::Tensor support
- Network/Node abstraction for Bayesian network representation
- Feature selection as separate, composable modules

### Data Handling
- Supports both discrete integer data and continuous data with discretization
- ARFF file format support through arff-files library
- Tensor operations via PyTorch C++ (libtorch)
- Local discretization variants use fimdlp library

## Documentation & Tools

- **Doxygen** for API documentation: `make doc`
- **lcov** for coverage reports: `make coverage`
- **plantuml + clang-uml** for UML diagrams: `make diagrams`
- Man pages available in `docs/man3/`

## Sample Applications

Sample code in `sample/` directory demonstrates library usage:
```bash
make sample fname=tests/data/iris.arff model=TANLd
```

## Common Development Tasks

- **Add new classifier**: Extend BaseClassifier, implement in appropriate subdirectory
- **Add new test**: Update `tests/CMakeLists.txt` and create test in `tests/`
- **Modify build**: Edit main `CMakeLists.txt` or use Makefile targets
- **Update dependencies**: Modify `vcpkg.json` and run `make init`