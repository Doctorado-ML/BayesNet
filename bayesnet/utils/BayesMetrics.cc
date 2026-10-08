// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <algorithm>
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

        // Three of the four quantities every pair score is built from depend on
        // a single variable: I(Xi;C), H(Xi|C) and H(Xi). They used to be
        // recomputed inside the pair loop, i.e. n-1 times each. Hoisting them
        // turns that part of the ranking from O(n^2*m) into O(n*m) and leaves
        // only the genuinely pairwise H(Xi|Xj,C) and H(Xi|Xj) in the inner loop.
        // The arithmetic is unchanged, so the scores are bit-for-bit the same.
        std::vector<torch::Tensor> rows(n);
        std::vector<double> miWithClass(n, 0.0);           // I(Xi;C)
        std::vector<double> condEntropyGivenClass(n, 0.0); // H(Xi|C)
        std::vector<double> featureEntropy(n, 0.0);        // H(Xi)
        for (int i = 0; i < n; ++i) {
            if (std::find(featuresExcluded.begin(), featuresExcluded.end(), i) != featuresExcluded.end()) {
                continue;
            }
            rows[i] = samples.index({ i, "..." });
            miWithClass[i] = mutualInformation(labels, rows[i], weights);
            condEntropyGivenClass[i] = conditionalEntropy(rows[i], labels, weights);
            featureEntropy[i] = entropy(rows[i], weights);
        }

        for (int i = 0; i < n - 1; ++i) {
            if (std::find(featuresExcluded.begin(), featuresExcluded.end(), i) != featuresExcluded.end()) {
                continue;
            }
            const auto& xi = rows[i];
            for (int j = i + 1; j < n; ++j) {
                if (std::find(featuresExcluded.begin(), featuresExcluded.end(), j) != featuresExcluded.end()) {
                    continue;
                }
                auto key = std::make_pair(i, j);
                const auto& xj = rows[j];
                // I(Xi;Xj|C) = H(Xi|C) - H(Xi|Xj,C), with H(Xi|C) hoisted above.
                double cmiIJ = std::max(condEntropyGivenClass[i] - conditionalEntropy(xi, xj, labels, weights), 0.0);
                double value;
                if (beta < 0.0) {
                    // Legacy ranking: conditional mutual information I(Xi;Xj|C) alone.
                    value = cmiIJ;
                } else {
                    // Joint relevance I(Xi,Xj;C) = I(Xi;C) + I(Xj;C) + beta * S(i,j),
                    // with interaction information S(i,j) = I(Xi;Xj|C) - I(Xi;Xj).
                    // beta=0 -> marginal-relevance sum, beta=1 -> joint relevance,
                    // beta large -> pure synergy. Same 3-way (Xi,Xj,C) table cost.
                    // I(Xi;Xj) = H(Xi) - H(Xi|Xj), with H(Xi) hoisted above.
                    double miIJ = std::max(featureEntropy[i] - conditionalEntropy(xi, xj, weights), 0.0);
                    value = miWithClass[i] + miWithClass[j] + beta * (cmiIJ - miIJ);
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
                // Drop the head in one shot; erasing element by element made
                // this O(P^2) in the number of pairs.
                int limit = pairsKBest.size() - k;
                pairsKBest.erase(pairsKBest.begin(), pairsKBest.begin() + limit);
                scoresKPairs.erase(scoresKPairs.begin(), scoresKPairs.begin() + limit);
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
    // Counts live in a dense (X,Y) table read through raw accessors. The
    // previous version walked the tensors with firstFeature[i].item<int>() and
    // friends, four ATen dispatches per sample, which made this ~50x slower
    // than the three-argument overload below despite doing less work. It is
    // called O(n^2) times per boosting round by SelectKPairs, so it dominated
    // XBA2DE training.
    //
    // The table also settles the iteration order the std::map here used to
    // provide: the entropy is accumulated over cells in ascending (X,Y) index
    // order, which is the order the map gave, so the result stays independent
    // of the standard library and bit-identical to that version.
    double Metrics::conditionalEntropy(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& weights)
    {
        // to() is a no-op when the dtype already matches, so the common
        // int32/float64 path costs nothing; other integral dtypes keep working
        // as they did when this walked the tensors with item<int>().
        auto first = firstFeature.to(torch::kInt32).contiguous();
        auto second = secondFeature.to(torch::kInt32).contiguous();
        auto weights_ = weights.to(torch::kFloat64).contiguous();
        auto firstData = first.accessor<int, 1>();
        auto secondData = second.accessor<int, 1>();
        auto weightsData = weights_.accessor<double, 1>();
        int numSamples = first.size(0);
        if (numSamples == 0)
            return 0;

        // Two degenerate cases, stated as the identities they are instead of left to
        // fall out of the arithmetic. Both produce whole blocks of edges that tie at
        // exactly zero, and the tie-break in kruskal_algorithm is what then defines
        // the spanning tree, so "near zero" is not good enough: any residue turns the
        // tie into an ordering. H(X|Y) cannot be left to agree with H(X) by accident
        // either -- entropy() sums through ATen and the table below sums
        // sequentially, two implementations of the same quantity that mutualInformation
        // subtracts from one another. They matched on x86_64 and did not on arm64,
        // which is what made glass's spanning tree platform dependent.
        const int firstMax = first.max().item<int>();
        if (firstMax == first.min().item<int>())
            return 0;                                    // X constant: H(X|Y) = 0
        if (second.max().item<int>() == second.min().item<int>())
            return entropy(firstFeature, weights);       // Y constant: H(X|Y) = H(X)

        auto featureCounts = second.bincount(weights_).to(torch::kFloat64).contiguous();
        auto featureCountsData = featureCounts.accessor<double, 1>();
        int numSecondStates = static_cast<int>(featureCounts.size(0));
        int numFirstStates = firstMax + 1;

        // jointWeight accumulates the weight of every (second, first) cell;
        // observed marks the cells that actually occur, which is what the
        // nested maps this replaced used to express implicitly.
        std::vector<double> jointWeight(static_cast<size_t>(numSecondStates) * numFirstStates, 0.0);
        std::vector<char> observed(jointWeight.size(), 0);
        double totalWeight = 0;
        for (int i = 0; i < numSamples; ++i) {
            auto cell = static_cast<size_t>(secondData[i]) * numFirstStates + firstData[i];
            jointWeight[cell] += weightsData[i];
            observed[cell] = 1;
            // Accumulated in double; the old code narrowed each weight to float
            // here while using double for the joint counts just above.
            totalWeight += weightsData[i];
        }
        if (totalWeight == 0)
            return 0;
        double entropyValue = 0;
        for (int value = 0; value < numSecondStates; ++value) {
            double countValue = featureCountsData[value];
            double p_f = countValue / totalWeight;
            double entropy_f = 0;
            auto base = static_cast<size_t>(value) * numFirstStates;
            for (int label = 0; label < numFirstStates; ++label) {
                if (!observed[base + label])
                    continue;
                double p_l_f = jointWeight[base + label] / countValue;
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
    // H(X|Y,C) = -sum_{x,y,c} p(x,y,c) log p(x|y,c)
    //
    // The conditioning set is (Y,C) and the variable whose uncertainty is left is X,
    // so the marginal has to be keyed on (y,c). It used to be keyed on (x,c), which
    // made this return H(Y|X,C) instead -- the roles of the two features swapped.
    // conditionalMutualInformation and SelectKPairs both subtract this from H(X|C),
    // and the two valid forms of the identity are H(X|C) - H(X|Y,C) and
    // H(Y|C) - H(Y|X,C); pairing H(X|C) with H(Y|X,C) mixes one of each. The visible
    // symptom was that I(X;Y|C) came out asymmetric, which it cannot be: on glass all
    // 36 pairs disagreed when the arguments were swapped, by up to 0.81.
    double Metrics::conditionalEntropy(const torch::Tensor& firstFeature, const torch::Tensor& secondFeature, const torch::Tensor& labels, const torch::Tensor& weights)
    {
        // Ensure the tensors are of the same length
        assert(firstFeature.size(0) == secondFeature.size(0) && firstFeature.size(0) == labels.size(0) && firstFeature.size(0) == weights.size(0));
        int numSamples = firstFeature.size(0);
        if (numSamples == 0)
            return 0;

        // The same two degenerate identities the two-argument overload states, and for
        // the same reason: conditioning on a constant is vacuous, and the callers
        // subtract this from H(X|C) computed by that *other* overload. Returning the
        // very call they subtract makes I(X;Y|C) come out as exactly zero rather than
        // as whatever ~1e-16 two different implementations of H(X|C) happen to differ
        // by. Blocks of pairs tie at exactly zero that way -- 15 of the 36 on glass --
        // and SelectKPairs' ranking is decided by the tie-break, so the zeros have to
        // be exact on every platform.
        if (firstFeature.max().item<int>() == firstFeature.min().item<int>())
            return 0;                                              // X constant: H(X|Y,C) = 0
        if (secondFeature.max().item<int>() == secondFeature.min().item<int>())
            return conditionalEntropy(firstFeature, labels, weights); // Y constant: H(X|Y,C) = H(X|C)

        // Convert tensors to vectors for easier processing
        auto firstFeatureData = firstFeature.accessor<int, 1>();
        auto secondFeatureData = secondFeature.accessor<int, 1>();
        auto labelsData = labels.accessor<int, 1>();
        auto weightsData = weights.accessor<double, 1>();
        // Maps for joint and marginal probabilities. std::map rather than
        // unordered_map so the accumulation order, and therefore the result, does not
        // depend on the standard library.
        std::map<std::tuple<int, int, int>, double> jointCount;
        std::map<std::tuple<int, int>, double> marginalCount;
        // Compute joint and marginal counts
        for (int i = 0; i < numSamples; ++i) {
            auto keyJoint = std::make_tuple(firstFeatureData[i], secondFeatureData[i], labelsData[i]);
            auto keyMarginal = std::make_tuple(secondFeatureData[i], labelsData[i]);

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
            auto [x, y, c] = keyJoint;
            double p_x_given_yc = jointFreq / marginalCount[std::make_tuple(y, c)];
            if (p_x_given_yc > 0) {
                conditionalEntropy -= (jointFreq / totalWeight) * std::log(p_x_given_yc);
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