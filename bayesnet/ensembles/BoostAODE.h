// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BOOSTAODE_H
#define BOOSTAODE_H
#include <map>
#include "bayesnet/classifiers/SPODE.h"
#include "bayesnet/feature_selection/FeatureSelect.h"
#include "boost.h"
#include "Ensemble.h"
namespace bayesnet {
    class BoostAODE : public Ensemble {
    public:
        explicit BoostAODE(bool predict_voting = false);
        virtual ~BoostAODE() = default;
        std::vector<std::string> graph(const std::string& title = "BoostAODE") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights) override;
    private:
        std::tuple<torch::Tensor&, double, bool> update_weights_block(int k, torch::Tensor& ytrain, torch::Tensor& weights);
        std::vector<int> initializeModels();
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool bisection = true; // if true, use bisection stratety to add k models at once to the ensemble
        int maxTolerance = 3;
        std::string order_algorithm; // order to process the KBest features asc, desc, rand
        bool convergence = true; //if true, stop when the model does not improve
        bool convergence_best = false; // wether to keep the best accuracy to the moment or the last accuracy as prior accuracy
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm = Orders.DESC; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
        bool block_update = false;
    };
}
#endif