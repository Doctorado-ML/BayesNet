// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include "bayesnet/utils/bayesnetUtils.h"
#include "KDB.h"

namespace bayesnet {
    KDB::KDB(int k, float theta) : Classifier(Network()), k(k), theta(theta)
    {
        validHyperparameters = { "k", "theta" };

    }
    void KDB::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("k")) {
            k = hyperparameters["k"];
            hyperparameters.erase("k");
        }
        if (hyperparameters.contains("theta")) {
            theta = hyperparameters["theta"];
            hyperparameters.erase("theta");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void KDB::buildModel(const torch::Tensor& weights)
    {
        /*
        1. For each feature Xi, compute mutual information, I(X;C),
        where C is the class.
        2. Compute class conditional mutual information I(Xi;XjIC), f or each
        pair of features Xi and Xj, where i#j.
        3. Let the used variable list, S, be empty.
        4. Let the DAG network being constructed, BN, begin with a single
        class node, C.
        5. Repeat until S includes all domain features
        5.1. Select feature Xmax which is not in S and has the largest value
        I(Xmax;C).
        5.2. Add a node to BN representing Xmax.
        5.3. Add an arc from C to Xmax in BN.
        5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
        the highest value for I(Xmax;X,jC).
        5.5. Add Xmax to S.
        Compute the conditional probabilility infered by the structure of BN by
        using counts from DB, and output BN.
        */
        // 1. For each feature Xi, compute mutual information, I(X;C),
        // where C is the class.
        addNodes();
        const torch::Tensor& y = dataset.index({ -1, "..." });
        std::vector<double> mi;
        for (auto i = 0; i < features.size(); i++) {
            torch::Tensor firstFeature = dataset.index({ i, "..." });
            mi.push_back(metrics.mutualInformation(firstFeature, y, weights));
        }
        // 2. Compute class conditional mutual information I(Xi;XjIC), f or each
        auto conditionalEdgeWeights = metrics.conditionalEdge(weights);
        // 3. Let the used variable list, S, be empty.
        std::vector<int> S;
        // 4. Let the DAG network being constructed, BN, begin with a single
        // class node, C.
        // 5. Repeat until S includes all domain features
        // 5.1. Select feature Xmax which is not in S and has the largest value
        // I(Xmax;C).
        auto order = argsort(mi);
        for (auto idx : order) {
            // 5.2. Add a node to BN representing Xmax.
            // 5.3. Add an arc from C to Xmax in BN.
            model.addEdge(className, features[idx]);
            // 5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
            // the highest value for I(Xmax;X,jC).
            add_m_edges(idx, S, conditionalEdgeWeights);
            // 5.5. Add Xmax to S.
            S.push_back(idx);
        }
    }
    void KDB::add_m_edges(int idx, std::vector<int>& S, torch::Tensor& weights)
    {
        auto n_edges = std::min(k, static_cast<int>(S.size()));
        auto cond_w = clone(weights);
        bool exit_cond = k == 0;
        int num = 0;
        while (!exit_cond) {
            // First maximal value, broken by the lowest index. torch::argmax is
            // documented to return the first one, but whole blocks of this row tie at
            // 0 (a feature left with a single state after discretization has zero
            // conditional mutual information with every other), and which index comes
            // back then depends on how the reduction is split rather than on the data.
            auto row = cond_w.index({ idx, "..." }).contiguous();
            auto rowData = row.accessor<float, 1>();
            int max_minfo = 0;
            for (int j = 1; j < row.size(0); ++j) {
                if (rowData[j] > rowData[max_minfo]) max_minfo = j;
            }
            auto belongs = find(S.begin(), S.end(), max_minfo) != S.end();
            if (belongs && cond_w.index({ idx, max_minfo }).item<float>() > theta) {
                try {
                    model.addEdge(features[max_minfo], features[idx]);
                    num++;
                }
                catch (const std::invalid_argument& e) {
                    // Loops are not allowed
                }
            }
            cond_w.index_put_({ idx, max_minfo }, -1);
            auto candidates_mask = cond_w.index({ idx, "..." }).gt(theta);
            auto candidates = candidates_mask.nonzero();
            exit_cond = num == n_edges || candidates.size(0) == 0;
        }
    }
    std::vector<std::string> KDB::graph(const std::string& title) const
    {
        std::string header{ title };
        if (title == "KDB") {
            header += " (k=" + std::to_string(k) + ", theta=" + std::to_string(theta) + ")";
        }
        return model.graph(header);
    }
}