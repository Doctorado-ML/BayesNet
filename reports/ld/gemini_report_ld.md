# Professional Review of Local Discretization Classifiers

This report provides a detailed analysis of the C++ implementation of the local discretization proposal in the `BayesNet` library, focusing on the `TANLd`, `KDBLd`, `SPODELd`, and `AODELd` classifiers.

## 1. Overall Summary

The implementation introduces an advanced, network-aware discretization strategy that iteratively refines feature discretization based on the learned model structure. This is a sophisticated approach that goes beyond standard global discretization.

The implementation for single classifiers (`TANLd`, `KDBLd`, `SPODELd`) is functional and follows a consistent pattern. However, it suffers from significant code duplication.

The ensemble implementation (`AODELd`) appears to be incomplete, incorrect, and highly inefficient. It does not correctly apply the proposed discretization logic at the ensemble level.

This report details these findings and provides recommendations for improvement.

## 2. Analysis of the Local Discretization Algorithm

The core logic resides in the `Proposal` class and is executed in two main phases within the `iterativeLocalDiscretization` function.

### Phase 1: Initial Global Discretization (`fit_local_discretization`)
- **Process**: Each continuous feature is discretized independently using the class label as the target variable. It employs methods from the `fimdlp` library (MDLP, Binning).
- **Assessment**: This is a standard and sound approach for an initial discretization.

### Phase 2: Iterative Local Refinement (`iterativeLocalDiscretization` & `localDiscretizationProposal`)
- **Process**: This is the novel part of the proposal.
    1. A base classifier model (`TAN`, `KDB`, etc.) is trained on the currently discretized data.
    2. The algorithm inspects the resulting network structure.
    3. For each feature, it re-discretizes it, but this time conditioned on both the class and the feature's parents in the learned network. This is the "local" aspect.
    4. The process repeats until the network structure converges (i.e., doesn't change between iterations).
- **Assessment**: This is a powerful concept. By making discretization network-aware, the process can capture more complex feature interactions than a simple global discretization. However, there are implementation concerns.

### Issues in the Algorithm Implementation
- **Inefficiency in `localDiscretizationProposal`**:
    - To create a joint variable for local discretization, the code concatenates strings in a loop: `yJoinParents[i] += to_string(...)`. This is very inefficient. A better approach would be to use a mathematical formula to combine the discrete parent values into a single unique ID for each instance.
    - The model is refit (`model.fit(...)`) inside `localDiscretizationProposal`, which is then called from the main loop in `iterativeLocalDiscretization`. This seems redundant as the model is already fit at the beginning of the main loop. This leads to unnecessary computational cost.

## 3. Review of Classifier Implementations

### `TANLd`, `KDBLd`, `SPODELd`
These classes wrap their respective base classifiers to apply the discretization logic.

- **Strengths**:
    - They correctly use the `iterativeLocalDiscretization` function to perform the full discretization process.
    - The pattern is consistent across all three classifiers.

- **Errors and Mistakes**:
    - **Major Code Duplication**: The source and header files for these three classes are nearly identical, differing only in the base class name. This is a significant maintenance issue.
    - **Recommendation**: This pattern is a prime candidate for a C++ template. A single template class, e.g., `DiscretizingClassifier<BaseClassifier>`, could replace all three, eliminating redundant code and reducing the chance of bugs when making changes.

- **Style and Best Practices**:
    - **Const Correctness**: `predict` and `predict_proba` methods should take `const torch::Tensor&` as input, as they do not modify the input tensor `X`.
    - **Tight Coupling**: The `Proposal` class constructor takes non-const references to members of its parent class (e.g., `dataset`, `features`). This creates a fragile, tight coupling. It would be cleaner to pass necessary data as arguments to the methods that need them (e.g., `iterativeLocalDiscretization`).

### `AODELd` (Ensemble Classifier)
The implementation of `AODELd` is seriously flawed and appears incomplete.

- **Errors and Mistakes**:
    - **Incorrect `fit` Logic**: The `AODELd::fit` method is a mix of commented-out code and logic that seems partially copied from `Ensemble::fit`. It does **not** call the `iterativeLocalDiscretization` function, which is the core of the proposal. It seems to attempt a separate, incomplete logic path.
    - **Extreme Inefficiency**: `AODELd` creates an ensemble of `SPODELd` models. Its `trainModel` method calls `fit` on each of these `SPODELd` models. This means that each model in the ensemble will independently execute the entire, expensive `iterativeLocalDiscretization` process. For an ensemble of *N* models, the discretization is run *N* times.
    - **Recommendation**: The discretization should be performed **once** before training the ensemble members. The `AODELd::fit` method should be responsible for running `iterativeLocalDiscretization` once on the data. Then, when training the individual `SPODE` models of the ensemble, they should be fitted on this single, pre-discretized dataset. The base models should be `SPODE`, not `SPODELd`, as the discretization is handled at the ensemble level.

## 4. General Design and Style Review

- **SPDX Headers**: The use of SPDX license headers is excellent practice.
- **Multiple Inheritance**: The use of multiple inheritance (`public TAN, public Proposal`) works here but is not ideal. It leads to the tight coupling issues mentioned earlier. A design based on composition (where the classifier "has a" discretizer) or using templates would be more robust and flexible.
- **Clarity**: The code, especially in `Proposal.cc`, could benefit from more comments explaining the "why" behind the algorithm's steps, particularly in the `localDiscretizationProposal` function.

## 5. Conclusion and Recommendations

The local discretization proposal is a promising and powerful feature for the `BayesNet` library. However, the implementation requires significant refactoring to be robust, efficient, and maintainable.

**High-Priority Recommendations:**

1.  **Fix `AODELd`**: The `AODELd::fit` logic must be completely rewritten. It should perform the `iterativeLocalDiscretization` once and then train its base `SPODE` models on the resulting discretized data.
2.  **Eliminate Code Duplication**: Refactor `TANLd`, `KDBLd`, and `SPODELd` into a single template class to improve maintainability.

**Medium-Priority Recommendations:**

3.  **Improve `localDiscretizationProposal` Efficiency**: Replace the string-based joint variable creation with a more performant mathematical approach.
4.  **Review Model Fitting Loop**: Analyze the necessity of re-fitting the model inside `localDiscretizationProposal` as it appears redundant.
5.  **Improve Encapsulation**: Decouple the `Proposal` class from the classifiers by passing data via method arguments instead of relying on member references passed during construction.
6.  **Enforce Const Correctness**: Apply `const` to function parameters where the data is not modified to improve API clarity and safety.
