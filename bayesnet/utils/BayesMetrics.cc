// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <map>
#include <tuple>
#include "Mst.h"
#include "BayesMetrics.h"
namespace bayesnet {
    //samples is n+1xm tensor used to fit the model
    Metrics::Metrics(const torch::Tensor& samples, const std::vector<std::string>& features, const std::string& className, const int classNumStates)
        : samples(samples)
        , className(className)
        , features(features)
        , classNumStates(classNumStates)
    {
    }
    //samples is n+1xm std::vector used to fit the model
    Metrics::Metrics(const std::vector<std::vector<int>>& vsamples, const std::vector<int>& labels, const std::vector<std::string>& features, const std::string& className, const int classNumStates)
        : samples(torch::zeros({ static_cast<int>(vsamples.size() + 1), static_cast<int>(vsamples[0].size()) }, torch::kInt32))
        , className(className)
        , features(features)
        , classNumStates(classNumStates)
    {
        for (int i = 0; i < vsamples.size(); ++i) {
            samples.index_put_({ i,  "..." }, torch::tensor(vsamples[i], torch::kInt32));
        }
        samples.index_put_({ -1, "..." }, torch::tensor(labels, torch::kInt32));
    }
    std::vector<std::pair<int, int>> Metrics::SelectKPairs(const torch::Tensor& weights, std::vector<int>& featuresExcluded, bool ascending, unsigned k, double beta)
    {
        // Return the K Best features 
        auto n = features.size();
        // compute scores
        scoresKPairs.clear();
        pairsKBest.clear();
        auto labels = samples.index({ -1, "..." });
        for (int i = 0; i < n - 1; ++i) {
            if (std::find(featuresExcluded.begin(), featuresExcluded.end(), i) != featuresExcluded.end()) {
                continue;
            }
            for (int j = i + 1; j < n; ++j) {
                if (std::find(featuresExcluded.begin(), featuresExcluded.end(), j) != featuresExcluded.end()) {
                    continue;
                }
                auto key = std::make_pair(i, j);
                auto xi = samples.index({ i, "..." });
                auto xj = samples.index({ j, "..." });
                double value;
                if (beta < 0.0) {
                    // Legacy ranking: conditional mutual information I(Xi;Xj|C) alone.
                    value = conditionalMutualInformation(xi, xj, labels, weights);
                } else {
                    // Joint relevance I(Xi,Xj;C) = I(Xi;C) + I(Xj;C) + beta * S(i,j),
                    // with interaction information S(i,j) = I(Xi;Xj|C) - I(Xi;Xj).
                    // beta=0 -> marginal-relevance sum, beta=1 -> joint relevance,
                    // beta large -> pure synergy. Same 3-way (Xi,Xj,C) table cost.
                    double miIC = mutualInformation(labels, xi, weights);
                    double miJC = mutualInformation(labels, xj, weights);
                    double cmiIJ = conditionalMutualInformation(xi, xj, labels, weights);
                    double miIJ = mutualInformation(xi, xj, weights);
                    value = miIC + miJC + beta * (cmiIJ - miIJ);
                }
                scoresKPairs.push_back({ key, value });
            }
        }
        // sort scores; pairs with an equal score are ordered by (i, j) so that
        // the ranking is fully defined here and not by std::sort's handling of
        // equivalent elements, which differs between standard libraries
        if (ascending) {
            sort(scoresKPairs.begin(), scoresKPairs.end(), [](const auto& a, const auto& b)
                {
                    if (a.second != b.second) return a.second < b.second;
                    return a.first < b.first;
                });

        } else {
            sort(scoresKPairs.begin(), scoresKPairs.end(), [](const auto& a, const auto& b)
                {
                    if (a.second != b.second) return a.second > b.second;
                    return a.first < b.first;
                });
        }
        for (auto& [pairs, score] : scoresKPairs) {
            pairsKBest.push_back(pairs);
        }
        if (k != 0 && k < pairsKBest.size()) {
            if (ascending) {
                int limit = pairsKBest.size() - k;
                for (int i = 0; i < limit; i++) {
                    pairsKBest.erase(pairsKBest.begin());
                    scoresKPairs.erase(scoresKPairs.begin());
                }
            } else {
                pairsKBest.resize(k);
                scoresKPairs.resize(k);
            }
        }
        return pairsKBest;
    }
    std::vector<int> Metrics::SelectKBestWeighted(const torch::Tensor& weights, bool ascending, unsigned k)
    {
        // Return the K Best features 
        auto n = features.size();
        if (k == 0) {
            k = n;
        }
        // compute scores
        scoresKBest.clear();
        featuresKBest.clear();
        auto label = samples.index({ -1, "..." });
        for (int i = 0; i < n; ++i) {
            scoresKBest.push_back(mutualInformation(label, samples.index({ i, "..." }), weights));
            featuresKBest.push_back(i);
        }
        // sort & reduce scores and features
        if (ascending) {
            sort(featuresKBest.begin(), featuresKBest.end(), [&](int i, int j)
                {
                    if (scoresKBest[i] != scoresKBest[j]) return scoresKBest[i] < scoresKBest[j];
                    return i < j;
                });
            sort(scoresKBest.begin(), scoresKBest.end(), std::less<double>());
            if (k < n) {
                for (int i = 0; i < n - k; ++i) {
                    featuresKBest.erase(featuresKBest.begin());
                    scoresKBest.erase(scoresKBest.begin());
                }
            }
        } else {
            sort(featuresKBest.begin(), featuresKBest.end(), [&](int i, int j)
                {
                    if (scoresKBest[i] != scoresKBest[j]) return scoresKBest[i] > scoresKBest[j];
                    return i < j;
                });
            sort(scoresKBest.begin(), scoresKBest.end(), std::greater<double>());
            featuresKBest.resize(k);
            scoresKBest.resize(k);
        }
        return featuresKBest;
    }
    std::vector<double> Metrics::getScoresKBest() const
    {
        return scoresKBest;
    }
    std::vector<std::pair<std::pair<int, int>, double>> Metrics::getScoresKPairs() const
    {
        return scoresKPairs;
    }
    torch::Tensor Metrics::conditionalEdge(const torch::Tensor& weights)
    {
        auto result = std::vector<double>();
        auto source = std::vector<std::string>(features);
        source.push_back(className);
        auto combinations = doCombinations(source);
        // Compute class prior
        auto margin = torch::zeros({ classNumStates }, torch::kFloat);
        for (int value = 0; value < classNumStates; ++value) {
            auto mask = samples.index({ -1,  "..." }) == value;
            margin[value] = mask.sum().item<double>() / samples.size(1);
        }
        for (auto [first, second] : combinations) {
            int index_first = find(features.begin(), features.end(), first) - features.begin();
            int index_second = find(features.begin(), features.end(), second) - features.begin();
            double accumulated = 0;
            for (int value = 0; value < classNumStates; ++value) {
                auto mask = samples.index({ -1, "..." }) == value;
                auto first_dataset = samples.index({ index_first, mask });
                auto second_dataset = samples.index({ index_second, mask });
                auto weights_dataset = weights.index({ mask });
                auto mi = mutualInformation(first_dataset, second_dataset, weights_dataset);
                auto pb = margin[value].item<double>();
                accumulated += pb * mi;
            }
            result.push_back(accumulated);
        }
        long n_vars = source.size();
        auto matrix = torch::zeros({ n_vars, n_vars });
        auto indices = torch::triu_indices(n_vars, n_vars, 1);
        for (auto i = 0; i < result.size(); ++i) {
            auto x = indices[0][i];
            auto y = indices[1][i];
            matrix[x][y] = result[i];
            matrix[y][x] = result[i];
        }
        return matrix;
    }
    // Measured in nats (natural logarithm (log) base e)
    // Elements of Information Theory, 2nd Edition, Thomas M. Cover, Joy A. Thomas p. 14
    double Metrics::entropy(const torch::Tensor& feature, const torch::Tensor& weights)
    {
        torch::Tensor counts = feature.bincount(weights);
        double totalWeight = counts.sum().item<double>();
        // In double: the rest of the computation, and conditionalEntropy which
        // this is subtracted from, are double. Going through float32 here left
        // the mutual information with ~7 significant digits.
        torch::Tensor probs = counts.to(torch::kDouble) / totalWeight;
        torch::Tensor logProbs = torch::log(probs);
        torch::Tensor entropy = -probs * logProbs;
        return entropy.nansum().item<double>();
    }
    // H(Y|X) = sum_{x in X} p(x) H(Y|X=x)
    double Metrics::conditionalEntropy(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& weights)
    {
        int numSamples = firstFeature.sizes()[0];
        torch::Tensor featureCounts = secondFeature.bincount(weights);
        // Ordered: the entropy below is accumulated by iterating this, and
        // floating point addition is not associative, so an implementation
        // defined iteration order would put the result's last bits at the mercy
        // of the standard library in use.
        std::map<int, std::map<int, double>> jointCounts;
        double totalWeight = 0;
        for (auto i = 0; i < numSamples; i++) {
            jointCounts[secondFeature[i].item<int>()][firstFeature[i].item<int>()] += weights[i].item<double>();
            totalWeight += weights[i].item<double>(); // was item<float>(), inconsistent with the line above
        }
        if (totalWeight == 0)
            return 0;
        double entropyValue = 0;
        for (int value = 0; value < featureCounts.sizes()[0]; ++value) {
            double p_f = featureCounts[value].item<double>() / totalWeight;
            double entropy_f = 0;
            for (auto& [label, jointCount] : jointCounts[value]) {
                double p_l_f = jointCount / featureCounts[value].item<double>();
                if (p_l_f > 0) {
                    entropy_f -= p_l_f * log(p_l_f);
                } else {
                    entropy_f = 0;
                }
            }
            entropyValue += p_f * entropy_f;
        }
        return entropyValue;
    }
    // H(X|Y,C) = sum_{y in Y, c in C} p(x,c) H(X|Y=y,C=c)
    double Metrics::conditionalEntropy(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& labels, const torch::Tensor& weights)
    {
        // Ensure the tensors are of the same length
        assert(firstFeature.size(0) == secondFeature.size(0) && firstFeature.size(0) == labels.size(0) && firstFeature.size(0) == weights.size(0));
        // Convert tensors to vectors for easier processing
        auto firstFeatureData = firstFeature.accessor<int, 1>();
        auto secondFeatureData = secondFeature.accessor<int, 1>();
        auto labelsData = labels.accessor<int, 1>();
        auto weightsData = weights.accessor<double, 1>();
        int numSamples = firstFeature.size(0);
        // Maps for joint and marginal probabilities
        std::map<std::tuple<int, int, int>, double> jointCount;
        std::map<std::tuple<int, int>, double> marginalCount;
        // Compute joint and marginal counts
        for (int i = 0; i < numSamples; ++i) {
            auto keyJoint = std::make_tuple(firstFeatureData[i], labelsData[i], secondFeatureData[i]);
            auto keyMarginal = std::make_tuple(firstFeatureData[i], labelsData[i]);

            jointCount[keyJoint] += weightsData[i];
            marginalCount[keyMarginal] += weightsData[i];
        }
        // Total weight sum
        double totalWeight = torch::sum(weights).item<double>();
        if (totalWeight == 0)
            return 0;
        // Compute the conditional entropy
        double conditionalEntropy = 0.0;
        for (const auto& [keyJoint, jointFreq] : jointCount) {
            auto [x, c, y] = keyJoint;
            auto keyMarginal = std::make_tuple(x, c);
            //double p_xc = marginalCount[keyMarginal] / totalWeight;
            double p_y_given_xc = jointFreq / marginalCount[keyMarginal];
            if (p_y_given_xc > 0) {
                conditionalEntropy -= (jointFreq / totalWeight) * std::log(p_y_given_xc);
            }
        }
        return conditionalEntropy;
    }
    // I(X;Y) = H(Y) - H(Y|X) ; I(X;Y) >= 0
    double Metrics::mutualInformation(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& weights)
    {
        return std::max(entropy(firstFeature, weights) - conditionalEntropy(firstFeature, secondFeature, weights), 0.0);
    }
    // I(X;Y|C) = H(X|C) - H(X|Y,C) >= 0
    double Metrics::conditionalMutualInformation(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& labels, const torch::Tensor& weights)
    {
        return std::max(conditionalEntropy(firstFeature, labels, weights) - conditionalEntropy(firstFeature, secondFeature, labels, weights), 0.0);
    }
    /*
    Compute the maximum spanning tree considering the weights as distances
    and the indices of the weights as nodes of this square matrix using
    Kruskal algorithm
    */
    std::vector<std::pair<int, int>> Metrics::maximumSpanningTree(const std::vector<std::string>& features, const torch::Tensor& weights, const int root)
    {
        auto mst = MST(features, weights, root);
        return mst.maximumSpanningTree();
    }
}