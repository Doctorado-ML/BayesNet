// **
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// **

#include "bayesnet/utils/bayesnetUtils.h"
#include "FeatureSelect.h"

namespace bayesnet {

    using namespace torch::indexing;          // for Ellipsis constant

    //---------------------------------------------------------------------
    // ctor
    //---------------------------------------------------------------------
    FeatureSelect::FeatureSelect(const torch::Tensor& samples,
        const std::vector<std::string>& features,
        const std::string& className,
        int maxFeatures,
        int classNumStates,
        const torch::Tensor& weights)
        : Metrics(samples, features, className, classNumStates),
        maxFeatures(maxFeatures == 0 ? samples.size(0) - 1 : maxFeatures),
        weights(weights)
    {
    }

    //---------------------------------------------------------------------
    // public helpers
    //---------------------------------------------------------------------
    void FeatureSelect::initialize()
    {
        selectedFeatures.clear();
        selectedScores.clear();
        suLabels.clear();
        suFeatures.clear();

        fitted = false;
    }

    //---------------------------------------------------------------------
    // Symmetrical Uncertainty (SU)
    //---------------------------------------------------------------------
    double FeatureSelect::symmetricalUncertainty(int a, int b)
    {
        /*
         * Compute symmetrical uncertainty. Normalises the information gain
         * (mutual information) with the entropies of the variables to compensate
         * the bias due to high‑cardinality features. Range: [0, 1]
         * See: https://www.sciencedirect.com/science/article/pii/S0020025519303603
         */

        auto x = samples.index({ a, Ellipsis });             // row a => feature a
        auto y = (b >= 0) ? samples.index({ b, Ellipsis })    // row b (>=0) => feature b
            : samples.index({ -1, Ellipsis }); // ‑1 treated as last row = labels

        double mu = mutualInformation(x, y, weights);
        double hx = entropy(x, weights);
        double hy = entropy(y, weights);

        const double denom = hx + hy;
        if (denom == 0.0) return 0.0;   // perfectly pure variables

        return 2.0 * mu / denom;
    }

    //---------------------------------------------------------------------
    // SU feature–class
    //---------------------------------------------------------------------
    void FeatureSelect::computeSuLabels()
    {
        // Compute Symmetrical Uncertainty between each feature and the class labels
        // https://en.wikipedia.org/wiki/Symmetric_uncertainty
        const int classIdx = static_cast<int>(samples.size(0)) - 1; // labels in last row
        suLabels.reserve(features.size());
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            suLabels.emplace_back(symmetricalUncertainty(i, classIdx));
        }
    }

    //---------------------------------------------------------------------
    // SU feature–feature with cache
    //---------------------------------------------------------------------
    double FeatureSelect::computeSuFeatures(int firstFeature, int secondFeature)
    {
        // Order the pair to exploit symmetry => only one entry in the map
        auto ordered = std::minmax(firstFeature, secondFeature);
        const std::pair<int, int> key{ ordered.first, ordered.second };

        auto it = suFeatures.find(key);
        if (it != suFeatures.end()) return it->second;

        double result = symmetricalUncertainty(key.first, key.second);
        suFeatures[key] = result;  // store once (symmetry handled by ordering)
        return result;
    }

    //---------------------------------------------------------------------
    // Correlation‑based Feature Selection (CFS) merit
    //---------------------------------------------------------------------
    double FeatureSelect::computeMeritCFS()
    {
        const int n = static_cast<int>(selectedFeatures.size());
        if (n == 0) return 0.0;

        // average r_cf (feature–class)
        double rcf_sum = 0.0;
        for (int f : selectedFeatures) rcf_sum += suLabels[f];
        const double rcf_avg = rcf_sum / n;

        // average r_ff (feature–feature)
        double rff_sum = 0.0;
        const auto& pairs = doCombinations(selectedFeatures); // generates each unordered pair once
        for (const auto& p : pairs) rff_sum += computeSuFeatures(p.first, p.second);

        const double numPairs = n * (n - 1) * 0.5;
        const double rff_avg = (numPairs > 0) ? rff_sum / numPairs : 0.0;

        // Merit_S = k * r_cf / sqrt( k + k*(k‑1) * r_ff )      (Hall, 1999)
        const double k = static_cast<double>(n);
        return (k * rcf_avg) / std::sqrt(k + k * (k - 1) * rff_avg);
    }

    //---------------------------------------------------------------------
    // getters
    //---------------------------------------------------------------------
    std::vector<int> FeatureSelect::getFeatures() const
    {
        if (!fitted) throw std::runtime_error("FeatureSelect not fitted");
        return selectedFeatures;
    }

    std::vector<double> FeatureSelect::getScores() const
    {
        if (!fitted) throw std::runtime_error("FeatureSelect not fitted");
        return selectedScores;
    }

} // namespace bayesnet
 