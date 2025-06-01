# <img src="logo.png" alt="logo" width="50"/>  BayesNet

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat&logo=c%2B%2B&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)
![Gitea Release](https://img.shields.io/gitea/v/release/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cf3e0ac71d764650b1bf4d8d00d303b1)](https://app.codacy.com/gh/Doctorado-ML/BayesNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Doctorado-ML/BayesNet)
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es&logo=gitea)
[![Coverage Badge](https://img.shields.io/badge/Coverage-99,2%25-green)](https://gitea.rmontanana.es/rmontanana/BayesNet)
[![DOI](https://zenodo.org/badge/667782806.svg)](https://doi.org/10.5281/zenodo.14210344)

Bayesian Network Classifiers library

## Setup

### Using the vcpkg library

You can use the library with the vcpkg library manager. In your project you have to add the following files:

#### vcpkg.json

```json
{
  "name": "sample-project",
  "version-string": "0.1.0",
  "dependencies": [
    "bayesnet"
  ]
}
```

#### vcpkg-configuration.json

```json
{
  "registries": [
    {
      "kind": "git",
      "repository": "https://github.com/rmontanana/vcpkg-stash",
      "baseline": "393efa4e74e053b6f02c4ab03738c8fe796b28e5",
      "packages": [
        "folding",
        "bayesnet",
        "arff-files",
        "fimdlp",
        "libtorch-bin"
      ]
    }
  ],
  "default-registry": {
    "kind": "git",
    "repository": "https://github.com/microsoft/vcpkg",
    "baseline": "760bfd0c8d7c89ec640aec4df89418b7c2745605"
  }
}
```

#### CMakeLists.txt

You have to include the following lines in your `CMakeLists.txt` file:

```cmake
find_package(bayesnet CONFIG REQUIRED)

add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE bayesnet::bayesnet)
```

After that, you can use the `vcpkg` command to install the dependencies:

```bash
vcpkg install
```

**Note: In the `sample` folder you can find a sample application that uses the library. You can use it as a reference to create your own application.**

## Playing with the library

The dependencies are managed with [vcpkg](https://vcpkg.io/) and supported by a private vcpkg repository in [https://github.com/rmontanana/vcpkg-stash](https://github.com/rmontanana/vcpkg-stash).

### Getting the code

```bash
git clone https://github.com/doctorado-ml/bayesnet
```

Once you have the code, you can use the `make` command to build the project. The `Makefile` is used to manage the build process and it will automatically download and install the dependencies.

### Release

```bash
make init # Install dependencies
make release # Build the release version
make buildr # compile and link the release version
```

### Debug & Tests

```bash
make init # Install dependencies
make debug # Build the debug version
make test # Run the tests
```

### Coverage

```bash
make coverage # Run the tests with coverage
make viewcoverage # View the coverage report in the browser
```

### Sample app

After building and installing the release version, you can run the sample app with the following commands:

```bash
make sample
make sample fname=tests/data/glass.arff
```

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
