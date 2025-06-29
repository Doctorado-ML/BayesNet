# Using BayesNet with Conan

This document explains how to use Conan as an alternative package manager for BayesNet.

## Prerequisites

```bash
pip install conan
```

## Quick Start

### As a Consumer

1. Create a `conanfile.txt` in your project:
```ini
[requires]
bayesnet/1.1.2@user/channel

[generators]
CMakeDeps
CMakeToolchain

[options]

[imports]
```

2. Install dependencies:
```bash
conan install . --build=missing
```

3. In your CMakeLists.txt:
```cmake
find_package(bayesnet REQUIRED)
target_link_libraries(your_target bayesnet::bayesnet)
```

### Building BayesNet with Conan

```bash
# Install dependencies
make conan-init

# Build debug version
make conan-debug
make buildd

# Build release version
make conan-release
make buildr

# Create package
make conan-create
```

## Current Limitations

- Custom dependencies (folding, fimdlp, arff-files) are not available in ConanCenter
- These need to be built as custom Conan packages or replaced with alternatives
- The conanfile.py currently comments out these dependencies

## Creating Custom Dependency Packages

For the custom dependencies, you'll need to create Conan recipes:

1. **folding**: Cross-validation library
2. **fimdlp**: Discretization library
3. **arff-files**: ARFF file format parser

Contact the maintainer or create custom recipes for these packages.

## Package Distribution

Once custom dependencies are resolved:

```bash
# Create and test package
make conan-create

# Upload to your remote
conan upload bayesnet/1.1.2@user/channel -r myremote
```