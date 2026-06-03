# <img src="logo.png" alt="logo" width="50"/>  BayesNet

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat&logo=c%2B%2B&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)
![Gitea Release](https://img.shields.io/gitea/v/release/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cf3e0ac71d764650b1bf4d8d00d303b1)](https://app.codacy.com/gh/Doctorado-ML/BayesNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Doctorado-ML/BayesNet)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Doctorado-ML/BayesNet)
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es&logo=gitea)
[![Coverage Badge](https://img.shields.io/badge/Coverage-99,0%25-green)](https://gitea.rmontanana.es/rmontanana/BayesNet)
[![DOI](https://zenodo.org/badge/667782806.svg)](https://doi.org/10.5281/zenodo.14210344)

Bayesian Network Classifiers library

## Using the Library

### Using Conan Package Manager

You can use the library with the [Conan](https://conan.io/) package manager. In your project you need to add the following files:

#### conanfile.txt

```txt
[requires]
bayesnet/1.1.2

[generators]
CMakeDeps
CMakeToolchain
```

#### CMakeLists.txt

Include the following lines in your `CMakeLists.txt` file:

```cmake
find_package(bayesnet REQUIRED)

add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE bayesnet::bayesnet)
```

Then install the dependencies and build your project:

```bash
conan install . --output-folder=build --build=missing
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build
```

**Note: In the `sample` folder you can find a sample application that uses the library. You can use it as a reference to create your own application.**

## Building and Testing

The project uses [Conan](https://conan.io/) for dependency management and provides convenient Makefile targets for common tasks.

### Prerequisites

- [Conan](https://conan.io/) package manager (`pip install conan`)
- CMake 3.27+
- C++17 compatible compiler

### Getting the code

```bash
git clone https://github.com/doctorado-ml/bayesnet
cd bayesnet
```

### Build Commands

#### Release Build

```bash
make release        # Configure release build with Conan
make buildr         # Build the release version
```

#### Debug Build & Tests

```bash
make debug          # Configure debug build with Conan
make buildd         # Build the debug version
make test           # Run the tests
```

#### Coverage Analysis

```bash
make coverage       # Run tests with coverage analysis
make viewcoverage   # View coverage report in browser
```

#### Sample Application

Run the sample application with different datasets and models:

```bash
make sample                                    # Run with default settings
make sample fname=tests/data/glass.arff       # Use glass dataset
make sample fname=tests/data/iris.arff model=AODE  # Use specific model
```

### Available Makefile Targets

- `debug` - Configure debug build using Conan
- `release` - Configure release build using Conan  
- `buildd` - Build debug targets
- `buildr` - Build release targets
- `test` - Run all tests (use `opt="-s"` for verbose output)
- `coverage` - Generate test coverage report
- `viewcoverage` - Open coverage report in browser
- `sample` - Build and run sample application
- `conan-create` - Create Conan package
- `conan-upload` - Upload package to Conan remote
- `conan-clean` - Clean Conan cache and build folders
- `clean` - Clean all build artifacts
- `doc` - Generate documentation
- `diagrams` - Generate UML diagrams
- `help` - Show all available targets

## Models

#### - TAN

#### - KDB

#### - SPODE

#### - SPnDE

#### - AODE

#### - A2DE

#### - [BoostAODE](docs/BoostAODE.md)

#### - XBAODE

#### - BoostA2DE

#### - XBA2DE

### With Local Discretization

#### - TANLd

#### - KDBLd

#### - SPODELd

#### - AODELd

## Documentation

### [Manual](https://rmontanana.github.io/bayesnet/)

### [Coverage report](https://rmontanana.github.io/bayesnet/coverage/index.html)

## Diagrams

### UML Class Diagram

![BayesNet UML Class Diagram](diagrams/BayesNet.svg)

### Dependency Diagram

![BayesNet Dependency Diagram](diagrams/dependency.svg)
