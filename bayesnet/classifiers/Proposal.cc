// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "Proposal.h"
#include <iostream>
#include <cmath>
#include <limits>
#include "Classifier.h"
#include "KDB.h"
#include "TAN.h"
#include "SPODE.h"
#include "KDBLd.h"
#include "TANLd.h"

namespace bayesnet {
    Proposal::Proposal(torch::Tensor& dataset_, std::vector<std::string>& features_, std::string& className_, std::vector<std::string>& notes_) : pDataset(dataset_), pFeatures(features_), pClassName(className_), notes(notes_)
    {
    }
    void Proposal::setHyperparameters(nlohmann::json& hyperparameters)
    {
        if (hyperparameters.contains("ld_proposed_cuts")) {
            ld_params.proposed_cuts = hyperparameters["ld_proposed_cuts"];
            hyperparameters.erase("ld_proposed_cuts");
        }
        if (hyperparameters.contains("mdlp_max_depth")) {
            ld_params.max_depth = hyperparameters["mdlp_max_depth"];
            hyperparameters.erase("mdlp_max_depth");
        }
        if (hyperparameters.contains("mdlp_min_length")) {
            ld_params.min_length = hyperparameters["mdlp_min_length"];
            hyperparameters.erase("mdlp_min_length");
        }
        if (hyperparameters.contains("ld_algorithm")) {
            auto algorithm = hyperparameters["ld_algorithm"];
            hyperparameters.erase("ld_algorithm");
            if (algorithm == "MDLP") {
                discretizationType = discretization_t::MDLP;
            } else if (algorithm == "BINQ") {
                discretizationType = discretization_t::BINQ;
            } else if (algorithm == "BINU") {
                discretizationType = discretization_t::BINU;
            } else {
                throw std::invalid_argument("Invalid discretization algorithm: " + algorithm.get<std::string>());
            }
        }
        // Convergence parameters
        if (hyperparameters.contains("max_iterations")) {
            convergence_params.maxIterations = hyperparameters["max_iterations"];
            hyperparameters.erase("max_iterations");
        }
        if (hyperparameters.contains("verbose_convergence")) {
            convergence_params.verbose = hyperparameters["verbose_convergence"];
            hyperparameters.erase("verbose_convergence");
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
    map<std::string, std::vector<int>> Proposal::fit_local_discretization(const torch::Tensor& y, map<std::string, std::vector<int>> states)
    {
        // Discretize the continuous input data and build pDataset (Classifier::dataset)
        // We expect to have in states for numeric features an empty vector and for discretized features a vector of states
        int m = Xf.size(1);
        int n = Xf.size(0);
        pDataset = torch::zeros({ n + 1, m }, torch::kInt32);
        auto yv = std::vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
        // discretize input data by feature(row)
        std::unique_ptr<mdlp::Discretizer> discretizer;
        for (auto i = 0; i < pFeatures.size(); ++i) {
            auto Xt_ptr = Xf.index({ i }).data_ptr<float>();
            auto Xt = std::vector<float>(Xt_ptr, Xt_ptr + Xf.size(1));
            if (states[pFeatures[i]].empty()) {
                // If the feature is numeric, we discretize it
                if (discretizationType == discretization_t::BINQ) {
                    discretizer = std::make_unique<mdlp::BinDisc>(ld_params.proposed_cuts, mdlp::strategy_t::QUANTILE);
                } else if (discretizationType == discretization_t::BINU) {
                    discretizer = std::make_unique<mdlp::BinDisc>(ld_params.proposed_cuts, mdlp::strategy_t::UNIFORM);
                } else { // Default is MDLP
                    discretizer = std::make_unique<mdlp::CPPFImdlp>(ld_params.min_length, ld_params.max_depth, ld_params.proposed_cuts);
                }
                pDataset.index_put_({ i, "..." }, torch::tensor(discretizer->fit_transform(Xt, yv)));
                int n_states = discretizer->getCutPoints().size() + 1;
                auto xStates = std::vector<int>(n_states);
                iota(xStates.begin(), xStates.end(), 0);
                states[pFeatures[i]] = xStates;
            } else {
                // If the feature is categorical, we just copy it
                pDataset.index_put_({ i, "..." }, Xf[i].to(torch::kInt32));
            }
            discretizers[pFeatures[i]] = std::move(discretizer);
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

    template<typename Classifier>
    map<std::string, std::vector<int>> Proposal::iterativeLocalDiscretization(
        const torch::Tensor& y,
        Classifier* classifier,
        torch::Tensor& dataset,
        const std::vector<std::string>& features,
        const std::string& className,
        const map<std::string, std::vector<int>>& initialStates,
        Smoothing_t smoothing
    )
    {
        // Phase 1: Initial discretization (same as original)
        auto currentStates = fit_local_discretization(y, initialStates);
        auto previousModel = Network();

        if (convergence_params.verbose) {
            std::cout << "Starting iterative local discretization with "
                << convergence_params.maxIterations << " max iterations" << std::endl;
        }

        const torch::Tensor weights = torch::full({ pDataset.size(1) }, 1.0 / pDataset.size(1), torch::kDouble);
        for (int iteration = 0; iteration < convergence_params.maxIterations; ++iteration) {
            if (convergence_params.verbose) {
                std::cout << "Iteration " << (iteration + 1) << "/" << convergence_params.maxIterations << std::endl;
            }

            // Phase 2: Build model with current discretization
            classifier->fit(dataset, features, className, currentStates, weights, smoothing);

            // Phase 3: Network-aware discretization refinement
            currentStates = localDiscretizationProposal(currentStates, classifier->getModel());

            // Check convergence
            if (iteration > 0 && previousModel == classifier->getModel()) {
                if (convergence_params.verbose) {
                    std::cout << "Converged after " << (iteration + 1) << " iterations" << std::endl;
                }
                notes.push_back("Converged after " + std::to_string(iteration + 1) + " of "
                    + std::to_string(convergence_params.maxIterations) + " iterations");
                break;
            }

            // Update for next iteration
            previousModel = classifier->getModel();
        }

        return currentStates;
    }

    // Explicit template instantiation for common classifier types
    template map<std::string, std::vector<int>> Proposal::iterativeLocalDiscretization<KDB>(
        const torch::Tensor&, KDB*, torch::Tensor&, const std::vector<std::string>&,
        const std::string&, const map<std::string, std::vector<int>>&, Smoothing_t);

    template map<std::string, std::vector<int>> Proposal::iterativeLocalDiscretization<TAN>(
        const torch::Tensor&, TAN*, torch::Tensor&, const std::vector<std::string>&,
        const std::string&, const map<std::string, std::vector<int>>&, Smoothing_t);
    template map<std::string, std::vector<int>> Proposal::iterativeLocalDiscretization<SPODE>(
        const torch::Tensor&, SPODE*, torch::Tensor&, const std::vector<std::string>&,
        const std::string&, const map<std::string, std::vector<int>>&, Smoothing_t);
}
