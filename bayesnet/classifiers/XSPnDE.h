// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XSPNDE_H
#define XSPNDE_H

#include "Classifier.h"
#include "bayesnet/utils/CountingSemaphore.h"
#include <torch/torch.h>
#include <vector>

namespace bayesnet {

class XSpnde : public Classifier {
  public:
    XSpnde(int spIndex1, int spIndex2);
    void setHyperparameters(const nlohmann::json &hyperparameters_) override;
    void fitx(torch::Tensor &X, torch::Tensor &y, torch::Tensor &weights_, const Smoothing_t smoothing);
    std::vector<double> predict_proba(const std::vector<int> &instance) const;
    std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>> &test_data) override;
    int predict(const std::vector<int> &instance) const;
    std::vector<int> predict(std::vector<std::vector<int>> &test_data) override;
    torch::Tensor predict(torch::Tensor &X) override;
    torch::Tensor predict_proba(torch::Tensor &X) override;

    float score(torch::Tensor &X, torch::Tensor &y) override;
    float score(std::vector<std::vector<int>> &X, std::vector<int> &y) override;
    std::string to_string() const;
    std::vector<std::string> graph(const std::string &title) const override {
        return std::vector<std::string>({title});
    }

    int getNumberOfNodes() const override;
    int getNumberOfEdges() const override;
    int getNFeatures() const;
    int getClassNumStates() const override;
    int getNumberOfStates() const override;

  protected:
    void buildModel(const torch::Tensor &weights) override;
    void trainModel(const torch::Tensor &weights, const bayesnet::Smoothing_t smoothing) override;

  private:
    void addSample(const std::vector<int> &instance, double weight);
    void normalize(std::vector<double> &v) const;
    void computeProbabilities();

    int superParent1_;
    int superParent2_;
    int nFeatures_;
    int statesClass_;
    double alpha_;
    double initializer_;

    std::vector<int> states_;
    std::vector<double> classCounts_;
    std::vector<double> classPriors_;
    std::vector<double> sp1FeatureCounts_, sp1FeatureProbs_;
    std::vector<double> sp2FeatureCounts_, sp2FeatureProbs_;
    // childOffsets_[f] will be the offset into childCounts_ for feature f.
    // If f is either superParent1 or superParent2, childOffsets_[f] = -1
    std::vector<int> childOffsets_;
    // For each child f, we store p(x_f | c, sp1Val, sp2Val).  We'll store the raw
    // counts in childCounts_, and the probabilities in childProbs_, with a
    // dimension block of size: states_[f]* statesClass_* states_[sp1]* states_[sp2].
    std::vector<double> childCounts_;
    std::vector<double> childProbs_;
    CountingSemaphore &semaphore_;
};

} // namespace bayesnet
#endif // XSPNDE_H
