// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BOOST_H
#define BOOST_H
#include <string>
#include <tuple>
#include <vector>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include "Ensemble.h"
#include "bayesnet/feature_selection/FeatureSelect.h"
namespace bayesnet {
    const struct {
        std::string CFS = "CFS";
        std::string FCBF = "FCBF";
        std::string IWSS = "IWSS";
    }SelectFeatures;
    const struct {
        std::string ASC = "asc";
        std::string DESC = "desc";
        std::string RAND = "rand";
    }Orders;
    class Boost : public Ensemble {
    public:
        explicit Boost(bool predict_voting = false);
        virtual ~Boost() override = default;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
    protected:
        std::vector<int> featureSelection(torch::Tensor& weights_);
        void buildModel(const torch::Tensor& weights) override;
        std::tuple<torch::Tensor&, double, bool> update_weights(torch::Tensor& ytrain, torch::Tensor& ypred, torch::Tensor& weights);
        std::tuple<torch::Tensor&, double, bool> update_weights_block(int k, torch::Tensor& ytrain, torch::Tensor& weights);
        void add_model(std::unique_ptr<Classifier> model, double significance);
        void remove_last_model();
        //
        // Attributes
        //
        torch::Tensor X_train, y_train, X_test, y_test;
        // Hyperparameters
        bool bisection = true; // if true, use bisection stratety to add k models at once to the ensemble
        int maxTolerance = 3;
        bool weightless = false; // if true, don't use weights for samples or SPODEs, don't use weights at all
        std::string order_algorithm = Orders.DESC; // order to process the KBest features asc, desc, rand
        bool convergence = true; //if true, stop when the model does not improve
        bool convergence_best = false; // wether to keep the best accuracy to the moment or the last accuracy as prior accuracy
        bool selectFeatures = false; // if true, use feature selection
        std::string select_features_algorithm; // Selected feature selection algorithm
        FeatureSelect* featureSelector = nullptr;
        double threshold = -1;
        bool block_update = false; // if true, use block update algorithm, only meaningful if bisection is true
        bool alpha_block = false; // if true, the alpha is computed with the ensemble built so far and the new model
    };
}
#endif
