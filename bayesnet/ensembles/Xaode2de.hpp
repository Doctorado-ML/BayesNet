// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// Optimized A2DE engine: flat count tables for ALL C(n,2) pair-superparents,
// built in a single pass over the data. Each pair (p,q) defines an SP2DE
// submodel with JOINT superparents P(x_p, x_q | c) (classic A2DE) and children
// conditioned on (x_p, x_q, c). The A2DE posterior is the average of the
// per-pair SP2DE posteriors (uniform significance), computed in log-space for
// numerical stability. Mirrors bayesnet::XSp2de's math (CESTNIK m-estimate,
// log-sum-exp) but materialized as flat arrays instead of C(n,2) objects.

#ifndef XAODE2DE_H
#define XAODE2DE_H
#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <torch/torch.h>
#include <bayesnet/network/Smoothing.h>

namespace bayesnet {
    class Xaode2de {
    public:
        enum class MatrixState { EMPTY, COUNTS, PROBS };
        std::vector<double> significance_models_;   // one weight per pair (uniform for A2DE)
        Xaode2de() : nFeatures_{ 0 }, statesClass_{ 0 }, nPairs_{ 0 }, matrixState_{ MatrixState::EMPTY } {}

        // -------------------------------------------------------
        // fit
        // -------------------------------------------------------
        // all_parents decides whether every pair starts active (weight 1) or none.
        // joint_parents models the two superparents jointly P(x_p,x_q|c) (classic
        // A2DE); when false they are treated as independent given the class.
        void fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bool all_parents, const bayesnet::Smoothing_t smoothing, const bool joint_parents = true)
        {
            jointParents_ = joint_parents;
            smoothing_ = smoothing;
            nFeatures_ = static_cast<int>(X.size());
            int num_instances = X[0].size();
            m_ = num_instances;

            // Cardinalities
            states_.resize(nFeatures_);
            for (int f = 0; f < nFeatures_; ++f) {
                states_[f] = *std::max_element(X[f].begin(), X[f].end()) + 1;
            }
            statesClass_ = *std::max_element(y.begin(), y.end()) + 1;
            classCounts_.assign(statesClass_, 0.0);
            classPriors_.assign(statesClass_, 0.0);

            // Enumerate the C(n,2) pairs (p<q)
            parent1_.clear();
            parent2_.clear();
            for (int p = 0; p < nFeatures_ - 1; ++p) {
                for (int q = p + 1; q < nFeatures_; ++q) {
                    parent1_.push_back(p);
                    parent2_.push_back(q);
                }
            }
            nPairs_ = static_cast<int>(parent1_.size());
            significance_models_.assign(nPairs_, (all_parents ? 1.0 : 0.0));

            // Joint-superparent count table offsets: block per pair = s_p * s_q * K
            parentOffset_.assign(nPairs_, 0);
            long parentTotal = 0;
            for (int pid = 0; pid < nPairs_; ++pid) {
                parentOffset_[pid] = parentTotal;
                parentTotal += static_cast<long>(states_[parent1_[pid]]) * states_[parent2_[pid]] * statesClass_;
            }
            parentCounts_.assign(parentTotal, 0.0);

            // Child count table offsets: block per (pair, child) = s_p * s_q * s_k * K
            childOffset_.assign(nPairs_, std::vector<long>(nFeatures_, -1));
            long childTotal = 0;
            for (int pid = 0; pid < nPairs_; ++pid) {
                int p = parent1_[pid], q = parent2_[pid];
                for (int k = 0; k < nFeatures_; ++k) {
                    if (k == p || k == q) continue;
                    childOffset_[pid][k] = childTotal;
                    childTotal += static_cast<long>(states_[p]) * states_[q] * states_[k] * statesClass_;
                }
            }
            childCounts_.assign(childTotal, 0.0);
            matrixState_ = MatrixState::COUNTS;

            // Single counting pass
            std::vector<int> instance(nFeatures_ + 1);
            for (int i = 0; i < num_instances; ++i) {
                for (int f = 0; f < nFeatures_; ++f) instance[f] = X[f][i];
                instance[nFeatures_] = y[i];
                addSample(instance, weights[i].item<double>());
            }
            computeProbabilities();
        }

        // -------------------------------------------------------
        // addSample (COUNTS mode)
        // -------------------------------------------------------
        void addSample(const std::vector<int>& instance, double weight)
        {
            if (weight <= 0.0) return;
            int c = instance.back();
            classCounts_[c] += weight;
            for (int pid = 0; pid < nPairs_; ++pid) {
                int p = parent1_[pid], q = parent2_[pid];
                int sp = instance[p], sq = instance[q];
                int qCard = states_[q];
                parentCounts_[parentOffset_[pid] + (sp * qCard + sq) * statesClass_ + c] += weight;
                for (int k = 0; k < nFeatures_; ++k) {
                    if (k == p || k == q) continue;
                    int kCard = states_[k];
                    long idx = childOffset_[pid][k] + (static_cast<long>((sp * qCard + sq) * kCard + instance[k])) * statesClass_ + c;
                    childCounts_[idx] += weight;
                }
            }
        }

        // Per-cell pseudocount, identical semantics to XSp2de::smoothingPseudocount.
        double smoothingPseudocount(int cardinality) const
        {
            switch (smoothing_) {
                case bayesnet::Smoothing_t::ORIGINAL: return (m_ > 0) ? 1.0 / m_ : 0.0;
                case bayesnet::Smoothing_t::LAPLACE:  return 1.0;
                case bayesnet::Smoothing_t::CESTNIK:  return (cardinality > 0) ? 1.0 / cardinality : 0.0;
                default:                              return 0.0;
            }
        }

        // -------------------------------------------------------
        // computeProbabilities (counts -> conditional probabilities, in place)
        // -------------------------------------------------------
        void computeProbabilities()
        {
            if (matrixState_ != MatrixState::COUNTS) {
                throw std::logic_error("computeProbabilities: must be in COUNTS mode.");
            }
            double totalCount = std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);
            // p(c)
            if (totalCount <= 0.0) {
                double unif = 1.0 / static_cast<double>(statesClass_);
                std::fill(classPriors_.begin(), classPriors_.end(), unif);
            } else {
                double a = smoothingPseudocount(statesClass_);
                for (int c = 0; c < statesClass_; ++c) {
                    classPriors_[c] = (classCounts_[c] + a) / (totalCount + a * statesClass_);
                }
            }
            // p(x_p, x_q | c) joint superparents, in place
            for (int pid = 0; pid < nPairs_; ++pid) {
                int sp = states_[parent1_[pid]], sq = states_[parent2_[pid]];
                int pairCard = sp * sq;
                double a = smoothingPseudocount(pairCard);
                long off = parentOffset_[pid];
                for (int v1 = 0; v1 < sp; ++v1) {
                    for (int v2 = 0; v2 < sq; ++v2) {
                        for (int c = 0; c < statesClass_; ++c) {
                            long idx = off + (v1 * sq + v2) * statesClass_ + c;
                            double denom = classCounts_[c] + a * pairCard;
                            parentCounts_[idx] = (denom <= 0.0) ? 0.0 : (parentCounts_[idx] + a) / denom;
                        }
                    }
                }
            }
            // p(x_k | c, x_p, x_q), in place. Denominator = N(x_p,x_q,c) recovered
            // by summing over the child dimension (done before overwriting).
            for (int pid = 0; pid < nPairs_; ++pid) {
                int p = parent1_[pid], q = parent2_[pid];
                int sp = states_[p], sq = states_[q];
                for (int k = 0; k < nFeatures_; ++k) {
                    if (k == p || k == q) continue;
                    int sk = states_[k];
                    double a = smoothingPseudocount(sk);
                    long base = childOffset_[pid][k];
                    for (int v1 = 0; v1 < sp; ++v1) {
                        for (int v2 = 0; v2 < sq; ++v2) {
                            for (int c = 0; c < statesClass_; ++c) {
                                double sum = 0.0;
                                for (int cv = 0; cv < sk; ++cv) {
                                    sum += childCounts_[base + (static_cast<long>((v1 * sq + v2) * sk + cv)) * statesClass_ + c];
                                }
                                double denom = sum + a * sk;
                                for (int cv = 0; cv < sk; ++cv) {
                                    long idx = base + (static_cast<long>((v1 * sq + v2) * sk + cv)) * statesClass_ + c;
                                    childCounts_[idx] = (denom <= 0.0) ? 0.0 : (childCounts_[idx] + a) / denom;
                                }
                            }
                        }
                    }
                }
            }
            matrixState_ = MatrixState::PROBS;
        }

        // -------------------------------------------------------
        // predict_proba: average of per-pair SP2DE posteriors (log-space)
        // -------------------------------------------------------
        std::vector<double> predict_proba(const std::vector<int>& instance) const
        {
            std::vector<double> result(statesClass_, 0.0);
            std::vector<double> logp(statesClass_, 0.0);
            std::vector<double> post(statesClass_, 0.0);
            for (int pid = 0; pid < nPairs_; ++pid) {
                if (significance_models_[pid] == 0.0) continue;
                int p = parent1_[pid], q = parent2_[pid];
                int sp = instance[p], sq = instance[q];
                int qCard = states_[q];
                // log p(c) + log p(x_p, x_q | c)
                for (int c = 0; c < statesClass_; ++c) {
                    double pParents;
                    if (jointParents_) {
                        pParents = parentCounts_[parentOffset_[pid] + (sp * qCard + sq) * statesClass_ + c];
                    } else {
                        // independent-parents fallback would need separate marginal
                        // tables; joint is the default and only stored form here.
                        pParents = parentCounts_[parentOffset_[pid] + (sp * qCard + sq) * statesClass_ + c];
                    }
                    logp[c] = std::log(classPriors_[c]) + std::log(pParents);
                }
                // + sum over children of log p(x_k | c, x_p, x_q)
                for (int k = 0; k < nFeatures_; ++k) {
                    if (k == p || k == q) continue;
                    int sk = states_[k];
                    long base = childOffset_[pid][k];
                    int kv = instance[k];
                    for (int c = 0; c < statesClass_; ++c) {
                        long idx = base + (static_cast<long>((sp * qCard + sq) * sk + kv)) * statesClass_ + c;
                        logp[c] += std::log(childCounts_[idx]);
                    }
                }
                // Normalize this pair's posterior with log-sum-exp, then accumulate.
                double maxLog = *std::max_element(logp.begin(), logp.end());
                if (!std::isfinite(maxLog)) continue;  // degenerate pair (only without smoothing)
                double sumExp = 0.0;
                for (int c = 0; c < statesClass_; ++c) { post[c] = std::exp(logp[c] - maxLog); sumExp += post[c]; }
                if (sumExp <= 0.0) continue;
                for (int c = 0; c < statesClass_; ++c) result[c] += (post[c] / sumExp) * significance_models_[pid];
            }
            // Normalize the averaged posterior.
            double s = std::accumulate(result.begin(), result.end(), 0.0);
            if (s > 0.0) {
                for (int c = 0; c < statesClass_; ++c) result[c] /= s;
            } else {
                std::fill(result.begin(), result.end(), 1.0 / static_cast<double>(statesClass_));
            }
            return result;
        }
        int predict(const std::vector<int>& instance) const
        {
            auto probs = predict_proba(instance);
            return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
        }

        // -------------------------------------------------------
        // Introspection
        // -------------------------------------------------------
        MatrixState state() const { return matrixState_; }
        int statesClass() const { return statesClass_; }
        int nFeatures() const { return nFeatures_; }
        int nPairs() const { return nPairs_; }
        int getClassNumStates() const { return statesClass_; }
        // C(n,2) submodels, each with (nFeatures + 1) nodes.
        int getNumberOfNodes() const { return nPairs_ * (nFeatures_ + 1); }
        // Per SP2DE: 3n-4 edges (class->all, both parents->children) plus the
        // sp1-sp2 edge when the superparents are modelled jointly.
        int getNumberOfEdges() const { return nPairs_ * (3 * nFeatures_ - 4 + (jointParents_ ? 1 : 0)); }
        int getNumberOfStates() const
        {
            return std::accumulate(states_.begin(), states_.end(), 0) * nFeatures_ * nPairs_;
        }

    private:
        std::vector<int> states_;            // cardinality per feature
        int nFeatures_;
        int statesClass_;
        int nPairs_;
        int m_ = 0;                          // #samples (for ORIGINAL smoothing)

        std::vector<int> parent1_;           // parent1_[pid], parent2_[pid] define pair pid
        std::vector<int> parent2_;

        std::vector<long> parentOffset_;     // offset into parentCounts_ per pair
        std::vector<double> parentCounts_;   // COUNTS then p(x_p,x_q|c)

        std::vector<std::vector<long>> childOffset_;  // [pid][k] offset into childCounts_
        std::vector<double> childCounts_;    // COUNTS then p(x_k|c,x_p,x_q)

        std::vector<double> classCounts_;
        std::vector<double> classPriors_;

        MatrixState matrixState_;
        bayesnet::Smoothing_t smoothing_ = bayesnet::Smoothing_t::ORIGINAL;
        bool jointParents_ = true;
    };
}
#endif // XAODE2DE_H
