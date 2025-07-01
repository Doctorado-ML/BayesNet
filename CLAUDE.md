# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BayesNet is a C++ library implementing Bayesian Network Classifiers. It provides various algorithms for machine learning classification including TAN, KDB, SPODE, SPnDE, AODE, A2DE, and their ensemble variants (Boost, XB). The library also includes local discretization variants (Ld) and feature selection algorithms.

## Build System & Dependencies

### Dependency Management

The project supports **two package managers**:

#### vcpkg (Default)

- Uses vcpkg with private registry at <https://github.com/rmontanana/vcpkg-stash>
- Core dependencies: libtorch, nlohmann-json, folding, fimdlp, arff-files, catch2
- All dependencies defined in `vcpkg.json` with version overrides

#### Conan (Alternative)

- Modern C++ package manager with better dependency resolution
- Configured via `conanfile.py` for packaging and distribution
- Supports subset of dependencies (libtorch, nlohmann-json, catch2)
- Custom dependencies (folding, fimdlp, arff-files) need custom Conan recipes

### Build Commands

#### Using vcpkg (Default)

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

#### Using Conan

```bash
# Install Conan first: pip install conan

# Initialize dependencies
make conan-init

# Build debug version (with tests and coverage)
make conan-debug
make buildd

# Build release version
make conan-release
make buildr

# Create and test Conan package
make conan-create

# Upload to Conan remote
make conan-upload remote=myremote

# Clean Conan cache and builds
make conan-clean
```

### CMake Configuration

- Uses CMake 3.27+ with C++17 standard
- Debug builds automatically enable testing and coverage
- Release builds optimize with `-Ofast`
- **Automatic package manager detection**: CMake detects whether Conan or vcpkg is being used
- Supports both static library and package manager installation
- Conditional dependency linking based on availability

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

## Package Distribution

### Creating Conan Packages

```bash
# Create package locally
make conan-create

# Test package installation
cd test_package
conan create ..

# Upload to remote repository
make conan-upload remote=myremote profile=myprofile
```

### Using the Library

With Conan:

```python
# conanfile.txt or conanfile.py
[requires]
bayesnet/1.1.2@user/channel

[generators]
cmake
```

With vcpkg:

```json
{
  "dependencies": ["bayesnet"]
}
```

## Common Development Tasks

- **Add new classifier**: Extend BaseClassifier, implement in appropriate subdirectory
- **Add new test**: Update `tests/CMakeLists.txt` and create test in `tests/`
- **Modify build**: Edit main `CMakeLists.txt` or use Makefile targets
- **Update dependencies**:
  - vcpkg: Modify `vcpkg.json` and run `make init`
  - Conan: Modify `conanfile.py` and run `make conan-init`
- **Package for distribution**: Use `make conan-create` for Conan packaging
