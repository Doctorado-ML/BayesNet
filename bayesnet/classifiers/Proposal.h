// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef PROPOSAL_H
#define PROPOSAL_H
#include <string>
#include <map>
#include <torch/torch.h>
#include <fimdlp/CPPFImdlp.h>
#include <fimdlp/BinDisc.h>
#include "bayesnet/network/Network.h"
#include <nlohmann/json.hpp>
#include "Classifier.h"

namespace bayesnet {
    class Proposal {
    public:
        Proposal(torch::Tensor& pDataset, std::vector<std::string>& features_, std::string& className_, std::vector<std::string>& notes);
        void setHyperparameters(nlohmann::json& hyperparameters_);
    protected:
        void checkInput(const torch::Tensor& X, const torch::Tensor& y);
        torch::Tensor prepareX(torch::Tensor& X);
        map<std::string, std::vector<int>> localDiscretizationProposal(const map<std::string, std::vector<int>>& states, Network& model);
        map<std::string, std::vector<int>> fit_local_discretization(const torch::Tensor& y);

        // Iterative discretization method
        template<typename Classifier>
        map<std::string, std::vector<int>> iterativeLocalDiscretization(
            const torch::Tensor& y,
            Classifier* classifier,
            torch::Tensor& dataset,
            const std::vector<std::string>& features,
            const std::string& className,
            const map<std::string, std::vector<int>>& initialStates,
            const Smoothing_t smoothing
        );

        torch::Tensor Xf; // X continuous nxm tensor
        torch::Tensor y; // y discrete nx1 tensor
        map<std::string, std::unique_ptr<mdlp::Discretizer>> discretizers;

        // MDLP parameters
        struct {
            size_t min_length = 3; // Minimum length of the interval to consider it in mdlp
            float proposed_cuts = 0.0; // Proposed cuts for the Discretization algorithm
            int max_depth = std::numeric_limits<int>::max(); // Maximum depth of the MDLP tree
        } ld_params;

        // Convergence parameters
        struct {
            int maxIterations = 10;
            bool verbose = false;
        } convergence_params;

        nlohmann::json validHyperparameters_ld = {
            "ld_algorithm", "ld_proposed_cuts", "mdlp_min_length", "mdlp_max_depth",
            "max_iterations", "verbose_convergence"
        };
    private:
        std::vector<int> factorize(const std::vector<std::string>& labels_t);
        std::vector<std::string>& notes; // Notes during fit from BaseClassifier
        torch::Tensor& pDataset; // (n+1)xm tensor
        std::vector<std::string>& pFeatures;
        std::string& pClassName;
        enum class discretization_t {
            MDLP,
            BINQ,
            BINU
        } discretizationType = discretization_t::MDLP; // Default discretization type
    };
}

#endif  