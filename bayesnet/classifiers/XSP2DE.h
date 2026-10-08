// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XSP2DE_H
#define XSP2DE_H

#include "Classifier.h"
#include "bayesnet/utils/CountingSemaphore.h"
#include <torch/torch.h>
#include <vector>

namespace bayesnet {

class XSp2de : public Classifier {
  public:
    XSp2de(int spIndex1, int spIndex2);
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

    // Bytes held by the probability/count tables of a *fitted* model, measured
    // from the vectors' capacity. This is the RESIDENT footprint: computeProbabilities
    // releases childCounts_, so it excludes that block. It deliberately ignores the
    // `dataset` tensor (storage shared with the ensemble, does not scale with the
    // number of models) and `metrics`: the budget is for the ensemble, not the process.
    size_t memoryFootprint() const;

    // Upper bound on the PEAK bytes a model for the pair (sp1, sp2) will hold while
    // fitting, i.e. with both childCounts_ and childProbs_ alive. `states` is the
    // per-feature cardinality vector and `statesClass` the number of class states.
    // Feeding it the ensemble's `states` map yields an upper bound of the real
    // footprint, because buildModel derives states_ from the training fold, whose
    // per-feature maxima can only be smaller.
    static size_t estimateFootprint(const std::vector<int> &states, int statesClass, int sp1, int sp2);

  protected:
    void buildModel(const torch::Tensor &weights) override;
    void trainModel(const torch::Tensor &weights, const bayesnet::Smoothing_t smoothing) override;

  private:
    void addSample(const std::vector<int> &instance, double weight);
    void normalize(std::vector<double> &v) const;
    void computeProbabilities();
    // Per-cell smoothing pseudocount for a table whose distributed variable has
    // `cardinality` states. ORIGINAL (1/m) and LAPLACE (1) ignore cardinality;
    // CESTNIK is the m-estimate (m=1, uniform prior) => 1/cardinality.
    double smoothingPseudocount(int cardinality) const;

    int superParent1_;
    int superParent2_;
    int nFeatures_;
    int statesClass_;
    // Smoothing strategy chosen at fit time; the actual per-cell pseudocount is
    // derived from it and the table cardinality (see smoothingPseudocount).
    bayesnet::Smoothing_t smoothing_;
    // If true, model the two superparents jointly P(sp1,sp2|c) (classic A2DE),
    // instead of as independent factors P(sp1|c)*P(sp2|c). This is the parent
    // layer where BoostA2DE departs from BoostAODE.
    bool jointParents_;

    std::vector<int> states_;
    std::vector<double> classCounts_;
    std::vector<double> classPriors_;
    std::vector<double> sp1FeatureCounts_, sp1FeatureProbs_;
    std::vector<double> sp2FeatureCounts_, sp2FeatureProbs_;
    // Joint superparents: p(sp1Val, sp2Val | c). Block layout
    // (sp1Val*states_[sp2]+sp2Val)*statesClass_ + c.
    std::vector<double> spPairCounts_, spPairProbs_;
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
#endif // XSP2DE_H
