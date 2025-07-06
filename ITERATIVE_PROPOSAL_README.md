# Iterative Proposal Implementation

This implementation extends the existing local discretization framework with iterative convergence capabilities, following the analysis from `local_discretization_analysis.md`.

## Key Components

### 1. IterativeProposal Class
- **File**: `bayesnet/classifiers/IterativeProposal.h|cc`
- **Purpose**: Extends the base `Proposal` class with iterative convergence logic
- **Key Method**: `iterativeLocalDiscretization()` - performs iterative refinement until convergence

### 2. TANLdIterative Example
- **File**: `bayesnet/classifiers/TANLdIterative.h|cc` 
- **Purpose**: Demonstrates how to adapt existing Ld classifiers to use iterative discretization
- **Pattern**: Inherits from both `TAN` and `IterativeProposal`

## Architecture

The implementation follows the established dual inheritance pattern:

```cpp
class TANLdIterative : public TAN, public IterativeProposal
```

This maintains the same interface as existing Ld classifiers while adding convergence capabilities.

## Convergence Algorithm

The iterative process works as follows:

1. **Initial Discretization**: Use class-only discretization (`fit_local_discretization()`)
2. **Iterative Refinement Loop**:
   - Build model with current discretization (call parent `fit()`)
   - Refine discretization using network structure (`localDiscretizationProposal()`)
   - Compute convergence metric (likelihood or accuracy)
   - Check for convergence based on tolerance
   - Repeat until convergence or max iterations reached

## Configuration Parameters

- `max_iterations`: Maximum number of iterations (default: 10)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `convergence_metric`: "likelihood" or "accuracy" (default: "likelihood")
- `verbose_convergence`: Enable verbose logging (default: false)

## Usage Example

```cpp
#include "bayesnet/classifiers/TANLdIterative.h"

// Create classifier
bayesnet::TANLdIterative classifier;

// Set convergence parameters
nlohmann::json hyperparams;
hyperparams["max_iterations"] = 5;
hyperparams["tolerance"] = 1e-4;
hyperparams["convergence_metric"] = "likelihood";
hyperparams["verbose_convergence"] = true;

classifier.setHyperparameters(hyperparams);

// Fit and use normally
classifier.fit(X, y, features, className, states, smoothing);
auto predictions = classifier.predict(X_test);
```

## Testing

Run the test with:
```bash
make -f Makefile.iterative test-iterative
```

## Integration with Existing Code

To convert existing Ld classifiers to use iterative discretization:

1. Change inheritance from `Proposal` to `IterativeProposal`
2. Replace the discretization logic in `fit()` method:
   ```cpp
   // Old approach:
   states = fit_local_discretization(y);
   TAN::fit(dataset, features, className, states, smoothing);
   states = localDiscretizationProposal(states, model);
   
   // New approach:
   states = iterativeLocalDiscretization(y, this, dataset, features, className, states_, smoothing);
   TAN::fit(dataset, features, className, states, smoothing);
   ```

## Benefits

1. **Convergence**: Iterative refinement until stable discretization
2. **Flexibility**: Configurable convergence criteria and limits
3. **Compatibility**: Maintains existing interface and patterns
4. **Monitoring**: Optional verbose logging for convergence tracking
5. **Extensibility**: Easy to add new convergence metrics or stopping criteria

## Performance Considerations

- Iterative approach will be slower than the original two-phase method
- Convergence monitoring adds computational overhead
- Consider setting appropriate `max_iterations` to prevent infinite loops
- The `tolerance` parameter should be tuned based on your specific use case

## Future Enhancements

Potential improvements:
1. Add more convergence metrics (e.g., AIC, BIC, cross-validation score)
2. Implement early stopping based on validation performance
3. Add support for different discretization schedules
4. Optimize likelihood computation for better performance
5. Add convergence visualization and reporting tools