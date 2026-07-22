// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XSPODE_H
#define XSPODE_H

#include <vector>
#include <torch/torch.h>
#include "Classifier.h"
#include "bayesnet/utils/CountingSemaphore.h"

namespace bayesnet {

    class XSpode : public Classifier {
    public:
        explicit XSpode(int spIndex);
        std::vector<double> predict_proba(const std::vector<int>& instance) const;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        int predict(const std::vector<int>& instance) const;
        void normalize(std::vector<double>& v) const;
        std::string to_string() const;
        int getNFeatures() const;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        int getClassNumStates() const override;
        std::vector<int>& getStates();
        std::vector<std::string> graph(const std::string& title) const override { return std::vector<std::string>({ title }); }
        void fitx(torch::Tensor& X, torch::Tensor& y, torch::Tensor& weights_, const Smoothing_t smoothing);
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;

        //
        // Classifier interface
        //
        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
    private:
        void addSample(const std::vector<int>& instance, double weight);
        void computeProbabilities();
        // Per-cell smoothing pseudocount for an additive-smoothed table whose
        // distributed variable has `cardinality` states. ORIGINAL (1/m) and
        // LAPLACE (1) ignore cardinality; CESTNIK is the m-estimate (m=1, uniform
        // prior) => 1/cardinality.
        double smoothingPseudocount(int cardinality) const;
        int superParent_;
        int nFeatures_;
        int statesClass_;
        std::vector<int> states_;          // [states_feat0, ..., states_feat(N-1)] (class not included in this array)

        // Class counts
        std::vector<double> classCounts_;  // [c], accumulative
        std::vector<double> classPriors_;  // [c], after normalization

        // For p(x_sp = spVal | c)
        std::vector<double> spFeatureCounts_; // [spVal * statesClass_ + c]
        std::vector<double> spFeatureProbs_;  // same shape, after normalization

        // For p(x_child = childVal | x_sp = spVal, c)
        // childCounts_ is big enough to hold all child features except sp:
        //   For each child f, we store childOffsets_[f] as the start index, then
        //   childVal, spVal, c => the data.
        std::vector<double> childCounts_;
        std::vector<double> childProbs_;
        std::vector<int>    childOffsets_;

        // Smoothing strategy chosen at fit time; the actual per-cell pseudocount
        // is derived from it and the table cardinality (see smoothingPseudocount).
        bayesnet::Smoothing_t smoothing_ = bayesnet::Smoothing_t::ORIGINAL;
        CountingSemaphore& semaphore_;
    };
}

#endif // XSPODE_H
