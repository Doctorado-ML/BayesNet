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
    // BayesNet#50. AdaBoost variant used by update_weights.
    const struct {
        std::string M1 = "M1";       // AdaBoost.M1: reject epsilon_t >= 0.5 (the published BoostAODE)
        std::string SAMME = "SAMME"; // Zhu et al. 2009: reject epsilon_t >= 1 - 1/K, alpha_t + ln(K - 1)
    }AdaBoostVariants;
    // BayesNet#51. Instance weight update after an accepted model.
    const struct {
        std::string BOOSTAODE = "BoostAODE"; // wrong w(1 + e^alpha), right w(1 + e^-alpha): ratio e^alpha (published)
        std::string ADABOOST = "AdaBoost";   // wrong w e^alpha, right w e^-alpha: ratio e^(2 alpha) (Zhou 2012)
    }WeightUpdates;
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
        // BayesNet#48. Called when update_weights() has rejected the model just
        // built (epsilon_t >= 0.5, alpha_t <= 0). AdaBoost leaves a rejected model out
        // of the ensemble, so it returns false and counts the rejection, unless the
        // ensemble is still empty: then the model is the only one there is, and it
        // is kept with alpha_t = 1 (any positive significance gives the same
        // predictions with a single model; 0 left the average as 0/0 = NaN).
        bool keepRejectedModel(double& alpha_t);
        // Note with the number of rejected models left out; call at the end of trainModel.
        void noteRejectedModels();
        int rejectedModels = 0;
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
        std::string adaboost = AdaBoostVariants.M1; // AdaBoost variant (M1, SAMME); both are the same with two classes
        std::string weight_update = WeightUpdates.BOOSTAODE; // instance weight update (BoostAODE, AdaBoost)
    };
}
#endif
