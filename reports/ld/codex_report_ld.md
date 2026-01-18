# Local Discretization Review

## Critical Issues
- Training-time values leak into inference for categorical attributes. `Proposal::prepareX` copies rows from the training tensor (`Xf`) for every non-numeric feature instead of using the caller-provided batch, so inference ignores the new categorical evidence and reuses the training set values (`bayesnet/classifiers/Proposal.cc:175`). This makes the ensemble/classifiers return the same predictions regardless of input for categorical features.
- All histogram/discretization routines assume `float32` feature tensors and `int32` targets, but none of the `fit` overloads normalise the dtypes. The code later reads raw data via `data_ptr<float>()` and `data_ptr<int>()` (`bayesnet/classifiers/Proposal.cc:134`, `bayesnet/classifiers/Proposal.cc:169`, `bayesnet/classifiers/Proposal.cc:129`), while `TANLd`, `KDBLd`, `SPODELd`, and `AODELd` merely copy the incoming tensors without casting (`bayesnet/classifiers/TANLd.cc:18`, `bayesnet/classifiers/KDBLd.cc:20`, `bayesnet/classifiers/SPODELd.cc:18`, `bayesnet/ensembles/AODELd.cc:19`). With PyTorch defaults (`float64` / `int64`), these `data_ptr` calls exhibit undefined behaviour and will corrupt or crash during discretization.
- The parent/context key used to refit discretizers concatenates integer states without a delimiter (`bayesnet/classifiers/Proposal.cc:95`). Different parent combinations (e.g., `(1, 23)` vs. `(12, 3)`) collide to the same string, leading to incorrect conditional discretization.

## High-Impact Behavioural Gaps
- Local refinement always retrains the network with `Smoothing_t::ORIGINAL`, ignoring the smoothing argument propagated by the caller (`bayesnet/classifiers/Proposal.cc:118`). Any custom smoothing requested via hyperparameters is silently discarded after the first refinement pass.
- `AODELd::fit` never updates the ensemble-level `states` map after sub-models discretize and expand the class/cardinality information (`bayesnet/ensembles/AODELd.cc:21`). Downstream ensemble utilities (e.g., voting normalisation in `Ensemble::voting`) rely on `states.at(className)` and can throw or behave inconsistently unless the caller pre-populates the map with the final state counts.

## Maintainability & Style Observations
- `Proposal::factorize` still carries vestigial logic (`allDigits`) that is never used (`bayesnet/classifiers/Proposal.cc:188`), hinting at incomplete validation.
- Tensor construction frequently omits explicit dtype declarations (e.g., `torch::tensor(...)` defaults to `int64`) while the storage tensor is `int32` (`bayesnet/classifiers/Proposal.cc:145`, `bayesnet/classifiers/Proposal.cc:173`). Being explicit would prevent inadvertent promotions.
- The `alreadyDiscretized` branch in `iterativeLocalDiscretization` is dead code; no caller sets the flag to `true`. This complicates understanding how AODE local discretization is meant to be reused.

## Recommendations
1. Normalise every incoming `X`/`y` to `float32`/`int32` (with `.contiguous()`), and guard `prepareX` against non-contiguous inputs before taking raw pointers.
2. Fix `prepareX` so categorical features read from the prediction tensor, and consider re-using the discretized training data via `wasNumeric` to avoid unnecessary copies.
3. Build parent keys with a delimiter (e.g., `std::ostringstream` with separators or structured tuples) to avoid collisions during conditional MDLP fitting.
4. Thread the requested smoothing value through `localDiscretizationProposal`, and reconcile ensemble-level `states` with the discretized outputs from its members.
5. Trim dead code and add targeted tests (unit or property-based) around the iterative discretization loop to cover convergence, dtype handling, and inference on purely categorical datasets.
