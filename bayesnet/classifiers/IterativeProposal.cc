// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "IterativeProposal.h"
#include <iostream>
#include <cmath>

namespace bayesnet {
    
    IterativeProposal::IterativeProposal(torch::Tensor& pDataset, std::vector<std::string>& features_, std::string& className_)
        : Proposal(pDataset, features_, className_) {}
    
    void IterativeProposal::setHyperparameters(const nlohmann::json& hyperparameters_) {
        // First set base Proposal hyperparameters
        Proposal::setHyperparameters(hyperparameters_);
        
        // Then set IterativeProposal specific hyperparameters
        if (hyperparameters_.contains("max_iterations")) {
            convergence_params.maxIterations = hyperparameters_["max_iterations"];
        }
        if (hyperparameters_.contains("tolerance")) {
            convergence_params.tolerance = hyperparameters_["tolerance"];
        }
        if (hyperparameters_.contains("convergence_metric")) {
            convergence_params.convergenceMetric = hyperparameters_["convergence_metric"];
        }
        if (hyperparameters_.contains("verbose_convergence")) {
            convergence_params.verbose = hyperparameters_["verbose_convergence"];
        }
    }
    
    template<typename Classifier>
    map<std::string, std::vector<int>> IterativeProposal::iterativeLocalDiscretization(
        const torch::Tensor& y, 
        Classifier* classifier,
        const torch::Tensor& dataset,
        const std::vector<std::string>& features,
        const std::string& className,
        const map<std::string, std::vector<int>>& initialStates,
        double smoothing
    ) {
        // Phase 1: Initial discretization (same as original)
        auto currentStates = fit_local_discretization(y);
        
        double previousValue = -std::numeric_limits<double>::infinity();
        double currentValue = 0.0;
        
        if (convergence_params.verbose) {
            std::cout << "Starting iterative local discretization with " 
                      << convergence_params.maxIterations << " max iterations" << std::endl;
        }
        
        for (int iteration = 0; iteration < convergence_params.maxIterations; ++iteration) {
            if (convergence_params.verbose) {
                std::cout << "Iteration " << (iteration + 1) << "/" << convergence_params.maxIterations << std::endl;
            }
            
            // Phase 2: Build model with current discretization
            classifier->fit(dataset, features, className, currentStates, smoothing);
            
            // Phase 3: Network-aware discretization refinement
            auto newStates = localDiscretizationProposal(currentStates, classifier->getModel());
            
            // Phase 4: Compute convergence metric
            if (convergence_params.convergenceMetric == "likelihood") {
                currentValue = computeLogLikelihood(classifier->getModel(), dataset);
            } else if (convergence_params.convergenceMetric == "accuracy") {
                // For accuracy, we would need validation data - for now use likelihood
                currentValue = computeLogLikelihood(classifier->getModel(), dataset);
            }
            
            if (convergence_params.verbose) {
                std::cout << "  " << convergence_params.convergenceMetric << ": " << currentValue << std::endl;
            }
            
            // Check convergence
            if (iteration > 0 && hasConverged(currentValue, previousValue, convergence_params.convergenceMetric)) {
                if (convergence_params.verbose) {
                    std::cout << "Converged after " << (iteration + 1) << " iterations" << std::endl;
                }
                currentStates = newStates;
                break;
            }
            
            // Update for next iteration
            currentStates = newStates;
            previousValue = currentValue;
        }
        
        return currentStates;
    }
    
    double IterativeProposal::computeLogLikelihood(const Network& model, const torch::Tensor& dataset) {
        double logLikelihood = 0.0;
        int n_samples = dataset.size(0);
        int n_features = dataset.size(1);
        
        for (int i = 0; i < n_samples; ++i) {
            double sampleLogLikelihood = 0.0;
            
            // Get class value for this sample
            int classValue = dataset[i][n_features - 1].item<int>();
            
            // Compute log-likelihood for each feature given its parents and class
            for (const auto& node : model.getNodes()) {
                if (node.getName() == model.getClassName()) {
                    // For class node, add log P(class)
                    auto classCounts = node.getCPT();
                    double classProb = classCounts[classValue] / dataset.size(0);
                    sampleLogLikelihood += std::log(std::max(classProb, 1e-10));
                } else {
                    // For feature nodes, add log P(feature | parents, class)
                    int featureIdx = std::distance(model.getFeatures().begin(), 
                                                 std::find(model.getFeatures().begin(), 
                                                          model.getFeatures().end(), 
                                                          node.getName()));
                    int featureValue = dataset[i][featureIdx].item<int>();
                    
                    // Simplified probability computation - in practice would need full CPT lookup
                    double featureProb = 0.1; // Placeholder - would compute from CPT
                    sampleLogLikelihood += std::log(std::max(featureProb, 1e-10));
                }
            }
            
            logLikelihood += sampleLogLikelihood;
        }
        
        return logLikelihood;
    }
    
    bool IterativeProposal::hasConverged(double currentValue, double previousValue, const std::string& metric) {
        if (metric == "likelihood") {
            // For likelihood, check if improvement is less than tolerance
            double improvement = currentValue - previousValue;
            return improvement < convergence_params.tolerance;
        } else if (metric == "accuracy") {
            // For accuracy, check if change is less than tolerance
            double change = std::abs(currentValue - previousValue);
            return change < convergence_params.tolerance;
        }
        return false;
    }
    
    // Explicit template instantiation for common classifier types
    template map<std::string, std::vector<int>> IterativeProposal::iterativeLocalDiscretization<Classifier>(
        const torch::Tensor&, Classifier*, const torch::Tensor&, const std::vector<std::string>&, 
        const std::string&, const map<std::string, std::vector<int>>&, double);
}