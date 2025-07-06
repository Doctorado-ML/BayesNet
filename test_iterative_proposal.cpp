// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <iostream>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include "bayesnet/classifiers/TANLdIterative.h"

using json = nlohmann::json;

int main() {
    std::cout << "Testing Iterative Proposal Implementation" << std::endl;
    
    // Create synthetic continuous data
    torch::Tensor X = torch::rand({100, 3}); // 100 samples, 3 features
    torch::Tensor y = torch::randint(0, 2, {100}); // Binary classification
    
    // Create feature names
    std::vector<std::string> features = {"feature1", "feature2", "feature3"};
    std::string className = "class";
    
    // Create initial states (will be updated by discretization)
    std::map<std::string, std::vector<int>> states;
    states[className] = {0, 1};
    
    // Create classifier
    bayesnet::TANLdIterative classifier;
    
    // Set convergence hyperparameters
    json hyperparams;
    hyperparams["max_iterations"] = 5;
    hyperparams["tolerance"] = 1e-4;
    hyperparams["convergence_metric"] = "likelihood";
    hyperparams["verbose_convergence"] = true;
    
    classifier.setHyperparameters(hyperparams);
    
    try {
        // Fit the model
        std::cout << "Fitting TANLdIterative classifier..." << std::endl;
        classifier.fit(X, y, features, className, states, bayesnet::Smoothing_t::LAPLACE);
        
        // Make predictions
        torch::Tensor X_test = torch::rand({10, 3});
        torch::Tensor predictions = classifier.predict(X_test);
        torch::Tensor probabilities = classifier.predict_proba(X_test);
        
        std::cout << "Predictions: " << predictions << std::endl;
        std::cout << "Probabilities shape: " << probabilities.sizes() << std::endl;
        
        // Generate graph
        auto graph = classifier.graph();
        std::cout << "Graph nodes: " << graph.size() << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}