// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "Proposal.h"

namespace bayesnet {
    Proposal::Proposal(torch::Tensor& dataset_, std::vector<std::string>& features_, std::string& className_) : pDataset(dataset_), pFeatures(features_), pClassName(className_) {}
    Proposal::~Proposal()
    {
        for (auto& [key, value] : discretizers) {
            delete value;
        }
    }
    void Proposal::checkInput(const torch::Tensor& X, const torch::Tensor& y)
    {
        if (!torch::is_floating_point(X)) {
            throw std::invalid_argument("X must be a floating point tensor");
        }
        if (torch::is_floating_point(y)) {
            throw std::invalid_argument("y must be an integer tensor");
        }
    }
    // Fit method for single classifier
    map<std::string, std::vector<int>> Proposal::localDiscretizationProposal(const map<std::string, std::vector<int>>& oldStates, Network& model)
    {
        // order of local discretization is important. no good 0, 1, 2...
        // although we rediscretize features after the local discretization of every feature
        auto order = model.topological_sort();
        auto& nodes = model.getNodes();
        map<std::string, std::vector<int>> states = oldStates;
        std::vector<int> indicesToReDiscretize;
        bool upgrade = false; // Flag to check if we need to upgrade the model
        for (auto feature : order) {
            auto nodeParents = nodes[feature]->getParents();
            if (nodeParents.size() < 2) continue; // Only has class as parent
            upgrade = true;
            int index = find(pFeatures.begin(), pFeatures.end(), feature) - pFeatures.begin();
            indicesToReDiscretize.push_back(index); // We need to re-discretize this feature
            std::vector<std::string> parents;
            transform(nodeParents.begin(), nodeParents.end(), back_inserter(parents), [](const auto& p) { return p->getName(); });
            // Remove class as parent as it will be added later
            parents.erase(remove(parents.begin(), parents.end(), pClassName), parents.end());
            // Get the indices of the parents
            std::vector<int> indices;
            indices.push_back(-1); // Add class index
            transform(parents.begin(), parents.end(), back_inserter(indices), [&](const auto& p) {return find(pFeatures.begin(), pFeatures.end(), p) - pFeatures.begin(); });
            // Now we fit the discretizer of the feature, conditioned on its parents and the class i.e. discretizer.fit(X[index], X[indices] + y)
            std::vector<std::string> yJoinParents(Xf.size(1));
            for (auto idx : indices) {
                for (int i = 0; i < Xf.size(1); ++i) {
                    yJoinParents[i] += to_string(pDataset.index({ idx, i }).item<int>());
                }
            }
            auto yxv = factorize(yJoinParents);
            auto xvf_ptr = Xf.index({ index }).data_ptr<float>();
            auto xvf = std::vector<mdlp::precision_t>(xvf_ptr, xvf_ptr + Xf.size(1));
            discretizers[feature]->fit(xvf, yxv);
        }
        if (upgrade) {
            // Discretize again X (only the affected indices) with the new fitted discretizers
            for (auto index : indicesToReDiscretize) {
                auto Xt_ptr = Xf.index({ index }).data_ptr<float>();
                auto Xt = std::vector<float>(Xt_ptr, Xt_ptr + Xf.size(1));
                pDataset.index_put_({ index, "..." }, torch::tensor(discretizers[pFeatures[index]]->transform(Xt)));
                auto xStates = std::vector<int>(discretizers[pFeatures[index]]->getCutPoints().size() + 1);
                iota(xStates.begin(), xStates.end(), 0);
                //Update new states of the feature/node
                states[pFeatures[index]] = xStates;
            }
            const torch::Tensor weights = torch::full({ pDataset.size(1) }, 1.0 / pDataset.size(1), torch::kDouble);
            model.fit(pDataset, weights, pFeatures, pClassName, states, Smoothing_t::ORIGINAL);
        }
        return states;
    }
    map<std::string, std::vector<int>> Proposal::fit_local_discretization(const torch::Tensor& y)
    {
        // Discretize the continuous input data and build pDataset (Classifier::dataset)
        int m = Xf.size(1);
        int n = Xf.size(0);
        map<std::string, std::vector<int>> states;
        pDataset = torch::zeros({ n + 1, m }, torch::kInt32);
        auto yv = std::vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        // discretize input data by feature(row)
        for (auto i = 0; i < pFeatures.size(); ++i) {
            auto* discretizer = new mdlp::CPPFImdlp();
            auto Xt_ptr = Xf.index({ i }).data_ptr<float>();
            auto Xt = std::vector<float>(Xt_ptr, Xt_ptr + Xf.size(1));
            discretizer->fit(Xt, yv);
            pDataset.index_put_({ i, "..." }, torch::tensor(discretizer->transform(Xt)));
            auto xStates = std::vector<int>(discretizer->getCutPoints().size() + 1);
            iota(xStates.begin(), xStates.end(), 0);
            states[pFeatures[i]] = xStates;
            discretizers[pFeatures[i]] = discretizer;
        }
        int n_classes = torch::max(y).item<int>() + 1;
        auto yStates = std::vector<int>(n_classes);
        iota(yStates.begin(), yStates.end(), 0);
        states[pClassName] = yStates;
        pDataset.index_put_({ n, "..." }, y);
        return states;
    }
    torch::Tensor Proposal::prepareX(torch::Tensor& X)
    {
        auto Xtd = torch::zeros_like(X, torch::kInt32);
        for (int i = 0; i < X.size(0); ++i) {
            auto Xt = std::vector<float>(X[i].data_ptr<float>(), X[i].data_ptr<float>() + X.size(1));
            auto Xd = discretizers[pFeatures[i]]->transform(Xt);
            Xtd.index_put_({ i }, torch::tensor(Xd, torch::kInt32));
        }
        return Xtd;
    }
    std::vector<int> Proposal::factorize(const std::vector<std::string>& labels_t)
    {
        std::vector<int> yy;
        yy.reserve(labels_t.size());
        std::map<std::string, int> labelMap;
        int i = 0;
        for (const std::string& label : labels_t) {
            if (labelMap.find(label) == labelMap.end()) {
                labelMap[label] = i++;
                bool allDigits = std::all_of(label.begin(), label.end(), ::isdigit);
            }
            yy.push_back(labelMap[label]);
        }
        return yy;
    }
}
