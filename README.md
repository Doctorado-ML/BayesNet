# <img src="logo.png" alt="logo" width="50"/>  BayesNet

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=flat&logo=c%2B%2B&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)
![Gitea Release](https://img.shields.io/gitea/v/release/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cf3e0ac71d764650b1bf4d8d00d303b1)](https://app.codacy.com/gh/Doctorado-ML/BayesNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=rmontanana_BayesNet&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rmontanana_BayesNet)
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/rmontanana/bayesnet?gitea_url=https://gitea.rmontanana.es&logo=gitea)
[![Coverage Badge](https://img.shields.io/badge/Coverage-99,1%25-green)](https://gitea.rmontanana.es/rmontanana/BayesNet)
[![DOI](https://zenodo.org/badge/667782806.svg)](https://doi.org/10.5281/zenodo.14210344)

Bayesian Network Classifiers library

## Dependencies

The only external dependency is [libtorch](https://pytorch.org/cppdocs/installing.html) which can be installed with the following commands:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

## Setup

### Getting the code

```bash
git clone --recurse-submodules https://github.com/doctorado-ml/bayesnet
```

### Release

```bash
make release
make buildr
sudo make install
```

### Debug & Tests

```bash
make debug
make test
```

### Coverage

```bash
make coverage
make viewcoverage
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
