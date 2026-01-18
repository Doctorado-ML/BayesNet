# Local Discretization Review (TANLd, KDBLd, SPODELd, AODELd)

## Overview
- Reviewed the iterative local discretization flow shared by `TANLd`, `KDBLd`, `SPODELd`, and the ensemble `AODELd`.
- Focused on correctness of the tensor preprocessing, convergence logic, and how the proposal integrates into base learners.
- Consulted Context7 for external references; no relevant library guidance was available for the in-house MDLP implementation.

## Critical Bugs
- **Unsafe tensor dtype assumptions** – `checkInput` only verifies that `X` is “floating” and `y` is “integer`, yet downstream code reinterprets tensors as `float32`/`int32` via raw `data_ptr` calls (`bayesnet/classifiers/Proposal.cc:60-170`). With PyTorch’s default `int64` targets (and even `float64` feature tensors), this yields undefined behaviour and sporadic crashes during the very first discretization pass (`y.data_ptr<int>()` at line 129, `Xf.index(...).data_ptr<float>()` at lines 134, 169). Every `*Ld::fit` simply copies the incoming tensors without casting (`bayesnet/classifiers/TANLd.cc:16-19`, `KDBLd.cc:17-21`, `SPODELd.cc:15-19`, `bayesnet/ensembles/AODELd.cc:14-21`), so the bug is systemic across all classifiers.
- **Categorical features ignored at prediction time** – when a feature was already discrete, `prepareX` writes back the training row `Xf[i]` instead of the caller-supplied data (`bayesnet/classifiers/Proposal.cc:165-176`). As a result, `predict`/`predict_proba` for `TANLd`, `KDBLd`, and `SPODELd` disregard every categorical input column and reuse the values seen during fit.
- **Parent joint keys collide during refinement** – the network-aware refinement concatenates parent states without delimiters (`bayesnet/classifiers/Proposal.cc:95-101`), so tuples like `(1, 23)` and `(12, 3)` both become `"123"`. This silently breaks the conditional histograms that drive MDLP splits, producing wrong cut points whenever any parent has more than one digit in its state label.

## High-Impact Issues
- **AODELd re-discretizes N times** – `AODELd::fit` never calls `iterativeLocalDiscretization`; instead, `trainModel` invokes `model->fit(Xf, y, …)` for each base `SPODELd` (`bayesnet/ensembles/AODELd.cc:22-53`). Every base model reruns the full iterative discretizer on identical data, turning an `O(N)` ensemble into `O(N²)` work and multiplying convergence instability. Discretizing once before spawning the ensemble (and feeding plain `SPODE` members) would remove the redundant cost.
- **Missing hyperparameter plumbing** – `AODELd::setHyperparameters` just caches the JSON blob and never forwards it to `Proposal::setHyperparameters` (`bayesnet/ensembles/AODELd.h:20-23`). Consequently, ensemble-level validation is skipped and any convergence/MDLP overrides intended for shared preprocessing are ignored unless each child happens to reapply them explicitly.

## Maintainability & Style
- The three single-model classifiers duplicate identical `fit`, `commonFit`, and `predict` glue (`bayesnet/classifiers/TANLd.cc`, `KDBLd.cc`, `SPODELd.cc`). A traits-based helper (e.g., CRTP) would centralise the Proposal/Torch wiring.
- `Proposal::factorize` keeps an unused `allDigits` flag (`bayesnet/classifiers/Proposal.cc:186-190`), and several comments still read like TODOs (e.g., “no good 0, 1, 2…” at line 72) which could be clarified or removed for production readiness.

## Suggested Next Steps
1. Normalise tensor dtypes at the `fit` entry point (clone + `to(torch::kFloat32)` / `to(torch::kInt32)`), then assert contiguity before taking raw pointers.
2. Fix `prepareX` so it respects incoming categorical rows, and add regression tests exercising mixed numeric/categorical inputs.
3. Replace the string concatenation in `localDiscretizationProposal` with a delimiter-safe encoding (e.g., `std::vector<int>` hash or use of tuples).
4. Refactor `AODELd` to discretize once up front (or pass a pre-discretized dataset) and reintroduce `Proposal::setHyperparameters` so shared knobs apply consistently.
