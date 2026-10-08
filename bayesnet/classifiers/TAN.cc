// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "TAN.h"

namespace bayesnet {
    TAN::TAN() : Classifier(Network())
    {
        validHyperparameters = { "parent" };
    }

    void TAN::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("parent")) {
            parent = hyperparameters["parent"];
            hyperparameters.erase("parent");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void TAN::buildModel(const torch::Tensor& weights)
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Compute mutual information between each feature and the class and set the root node
        // as the highest mutual information with the class
        auto mi = std::vector <std::pair<int, float >>();
        torch::Tensor class_dataset = dataset.index({ -1, "..." });
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            torch::Tensor feature_dataset = dataset.index({ i, "..." });
            auto mi_value = metrics.mutualInformation(class_dataset, feature_dataset, weights);
            mi.push_back({ i, mi_value });
        }
        // Equal mutual informations are broken by feature index, so the root picked
        // below does not depend on std::sort's handling of equivalent elements
        std::sort(mi.begin(), mi.end(), [](const auto& left, const auto& right) {
            if (left.second != right.second) return left.second < right.second;
            return left.first < right.first;
            });
        auto root = parent == -1 ? mi[mi.size() - 1].first : parent;
        if (root >= static_cast<int>(features.size())) {
            throw std::invalid_argument("The parent node is not in the dataset");
        }
        // 2. Compute mutual information between each feature and the class
        auto weights_matrix = metrics.conditionalEdge(weights);
        // 3. Compute the maximum spanning tree
        auto mst = metrics.maximumSpanningTree(features, weights_matrix, root);
        // 4. Add edges from the maximum spanning tree to the model
        for (auto i = 0; i < mst.size(); ++i) {
            auto [from, to] = mst[i];
            model.addEdge(features[from], features[to]);
        }
        // 5. Add edges from the class to all features
        for (auto feature : features) {
            model.addEdge(className, feature);
        }
    }
    std::vector<std::string> TAN::graph(const std::string& title) const
    {
        return model.graph(title);
    }
}