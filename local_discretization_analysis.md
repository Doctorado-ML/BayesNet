# Local Discretization Analysis - BayesNet Library

## Overview

This document analyzes the local discretization implementation in the BayesNet library, specifically focusing on the `Proposal.cc` implementation, and evaluates the feasibility of implementing an iterative discretization approach.

## Current Local Discretization Implementation

### Core Architecture

The local discretization functionality is implemented through a **Proposal class** (`bayesnet/classifiers/Proposal.h`) that serves as a mixin/base class for creating "Ld" (Local Discretization) variants of existing classifiers.

### Key Components

#### 1. The Proposal Class
- **Purpose**: Handles continuous data by applying local discretization using discretization algorithms
- **Dependencies**: Uses the `fimdlp` library for discretization algorithms
- **Supported Algorithms**:
  - **MDLP** (Minimum Description Length Principle) - Default
  - **BINQ** - Quantile-based binning
  - **BINU** - Uniform binning

#### 2. Local Discretization Variants

The codebase implements Ld variants using multiple inheritance:

**Individual Classifiers:**
- `TANLd` - Tree Augmented Naive Bayes with Local Discretization
- `KDBLd` - K-Dependence Bayesian with Local Discretization  
- `SPODELd` - Super-Parent One-Dependence Estimator with Local Discretization

**Ensemble Classifiers:**
- `AODELd` - Averaged One-Dependence Estimator with Local Discretization

### Implementation Pattern

All Ld variants follow a consistent pattern using **multiple inheritance**:

```cpp
class TANLd : public TAN, public Proposal {
    // Inherits from both the base classifier and Proposal
};
```

### Two-Phase Discretization Process

#### Phase 1: Initial Discretization (`fit_local_discretization`)
- Each continuous feature is discretized independently using the chosen algorithm
- Creates initial discrete dataset
- Uses only class labels for discretization decisions

#### Phase 2: Network-Aware Refinement (`localDiscretizationProposal`)
- After building the initial Bayesian network structure
- Features are re-discretized considering their parent nodes in the network
- Uses topological ordering to ensure proper dependency handling
- Creates more informed discretization boundaries based on network relationships

### Hyperparameter Support

The Proposal class supports several configurable hyperparameters:
- `ld_algorithm`: Choice of discretization algorithm (MDLP, BINQ, BINU)
- `ld_proposed_cuts`: Number of proposed cuts for discretization
- `mdlp_min_length`: Minimum interval length for MDLP
- `mdlp_max_depth`: Maximum depth for MDLP tree

## Current Implementation Strengths

1. **Sophisticated Approach**: Considers network structure in discretization decisions
2. **Modular Design**: Clean separation through Proposal class mixin
3. **Multiple Algorithm Support**: Flexible discretization strategies
4. **Proper Dependency Handling**: Topological ordering ensures correct processing
5. **Well-Integrated**: Seamless integration with existing classifier architecture

## Areas for Improvement

### Code Quality Issues

1. **Dead Code**: Line 161 in `Proposal.cc` contains unused variable `allDigits`
2. **Performance Issues**: 
   - String concatenation in tight loop (lines 82-84) using `+=` operator
   - Memory allocations could be optimized
   - Tensor operations could be batched better
3. **Error Handling**: Could be more robust with better exception handling

### Algorithm Clarity

1. **Logic Clarity**: The `upgrade` flag logic could be more descriptive
2. **Variable Naming**: Some variables need more descriptive names
3. **Documentation**: Better inline documentation of the two-phase process
4. **Method Complexity**: `localDiscretizationProposal` method is quite long and complex

### Suggested Code Improvements

```cpp
// Instead of string concatenation in loop:
for (auto idx : indices) {
    for (int i = 0; i < Xf.size(1); ++i) {
        yJoinParents[i] += to_string(pDataset.index({ idx, i }).item<int>());
    }
}

// Consider using stringstream or pre-allocation:
std::stringstream ss;
for (auto idx : indices) {
    for (int i = 0; i < Xf.size(1); ++i) {
        ss << pDataset.index({ idx, i }).item<int>();
        yJoinParents[i] = ss.str();
        ss.str("");
    }
}
```

## Iterative Discretization Proposal

### Concept

Implement an iterative process: discretize → build model → re-discretize → rebuild model → repeat until convergence.

### Feasibility Assessment

**Highly Feasible** - The current implementation already provides a solid foundation with its two-phase approach, making extension straightforward.

### Proposed Implementation Strategy

```cpp
class IterativeProposal : public Proposal {
public:
    struct ConvergenceParams {
        int max_iterations = 10;
        double tolerance = 1e-6;
        bool check_network_structure = true;
        bool check_discretization_stability = true;
    };

private:
    map<string, vector<int>> iterativeLocalDiscretization(const torch::Tensor& y) {
        auto states = fit_local_discretization(y);  // Initial discretization
        Network previousModel, currentModel;
        int iteration = 0;
        
        do {
            previousModel = currentModel;
            
            // Build model with current discretization
            const torch::Tensor weights = torch::full({ pDataset.size(1) }, 1.0 / pDataset.size(1), torch::kDouble);
            currentModel.fit(pDataset, weights, pFeatures, pClassName, states, Smoothing_t::ORIGINAL);
            
            // Apply local discretization based on current model
            auto newStates = localDiscretizationProposal(states, currentModel);
            
            // Check for convergence
            if (hasConverged(previousModel, currentModel, states, newStates)) {
                break;
            }
            
            states = newStates;
            iteration++;
            
        } while (iteration < convergenceParams.max_iterations);
        
        return states;
    }
    
    bool hasConverged(const Network& prev, const Network& curr, 
                     const map<string, vector<int>>& oldStates,
                     const map<string, vector<int>>& newStates) {
        // Implementation of convergence criteria
        return checkNetworkStructureConvergence(prev, curr) && 
               checkDiscretizationStability(oldStates, newStates);
    }
};
```

### Convergence Criteria Options

1. **Network Structure Comparison**: Compare edge sets between iterations
   ```cpp
   bool checkNetworkStructureConvergence(const Network& prev, const Network& curr) {
       // Compare adjacency matrices or edge lists
       return prev.getEdges() == curr.getEdges();
   }
   ```

2. **Discretization Stability**: Check if cut points change significantly
   ```cpp
   bool checkDiscretizationStability(const map<string, vector<int>>& oldStates,
                                    const map<string, vector<int>>& newStates) {
       for (const auto& [feature, states] : oldStates) {
           if (states != newStates.at(feature)) {
               return false;
           }
       }
       return true;
   }
   ```

3. **Performance Metrics**: Monitor accuracy/likelihood convergence
4. **Maximum Iterations**: Prevent infinite loops

### Expected Benefits

1. **Better Discretization Quality**: Each iteration refines boundaries based on learned dependencies
2. **Improved Model Accuracy**: More informed discretization leads to better classification
3. **Adaptive Process**: Automatically finds optimal discretization-model combination
4. **Principled Approach**: Theoretically sound iterative refinement
5. **Reduced Manual Tuning**: Less need for hyperparameter optimization

### Implementation Considerations

1. **Convergence Detection**: Need robust criteria to detect when to stop
2. **Performance Impact**: Multiple iterations increase computational cost
3. **Overfitting Prevention**: May need regularization to avoid over-discretization
4. **Stability Guarantees**: Ensure the process doesn't oscillate indefinitely
5. **Memory Management**: Handle multiple model instances efficiently

### Integration Strategy

1. **Backward Compatibility**: Keep existing two-phase approach as default
2. **Optional Feature**: Add iterative mode as configurable option
3. **Hyperparameter Extension**: Add convergence-related parameters
4. **Testing Framework**: Comprehensive testing on standard datasets

## Conclusion

The current local discretization implementation in BayesNet is well-designed and functional, providing a solid foundation for the proposed iterative enhancement. The iterative approach would significantly improve the quality of discretization by creating a feedback loop between model structure and discretization decisions.

The implementation is highly feasible given the existing architecture, and the expected benefits justify the additional computational complexity. The key to success will be implementing robust convergence criteria and maintaining the modularity of the current design.

## Recommendations

1. **Immediate Improvements**: Fix code quality issues and optimize performance bottlenecks
2. **Iterative Implementation**: Develop the iterative approach as an optional enhancement
3. **Comprehensive Testing**: Validate improvements on standard benchmark datasets
4. **Documentation**: Enhance inline documentation and user guides
5. **Performance Profiling**: Monitor computational overhead and optimize where necessary