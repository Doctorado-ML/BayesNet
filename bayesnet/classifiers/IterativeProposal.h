// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef ITERATIVE_PROPOSAL_H
#define ITERATIVE_PROPOSAL_H

#include "Proposal.h"
#include "bayesnet/network/Network.h"
#include <nlohmann/json.hpp>

namespace bayesnet {
    class IterativeProposal : public Proposal {
    public:
        IterativeProposal(torch::Tensor& pDataset, std::vector<std::string>& features_, std::string& className_);
        void setHyperparameters(const nlohmann::json& hyperparameters_);
        
    protected:
        template<typename Classifier>
        map<std::string, std::vector<int>> iterativeLocalDiscretization(
            const torch::Tensor& y, 
            Classifier* classifier,
            const torch::Tensor& dataset,
            const std::vector<std::string>& features,
            const std::string& className,
            const map<std::string, std::vector<int>>& initialStates,
            double smoothing = 1.0
        );
        
        // Convergence parameters
        struct {
            int maxIterations = 10;
            double tolerance = 1e-6;
            std::string convergenceMetric = "likelihood"; // "likelihood" or "accuracy"
            bool verbose = false;
        } convergence_params;
        
        nlohmann::json validHyperparameters_iter = { 
            "max_iterations", "tolerance", "convergence_metric", "verbose_convergence" 
        };
        
    private:
        double computeLogLikelihood(const Network& model, const torch::Tensor& dataset);
        bool hasConverged(double currentValue, double previousValue, const std::string& metric);
    };
}

#endif