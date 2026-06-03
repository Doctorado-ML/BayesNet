// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include "Boost.h"
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include <folding.hpp>

namespace bayesnet {
    Boost::Boost(bool predict_voting) : Ensemble(predict_voting)
    {
        validHyperparameters = { "alpha_block", "order",        "convergence",    "convergence_best", "bisection",
                                "threshold",   "maxTolerance", "predict_voting", "select_features",  "block_update", "weightless" };
    }
    void Boost::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("order")) {
            std::vector<std::string> algos = { Orders.ASC, Orders.DESC, Orders.RAND };
            order_algorithm = hyperparameters["order"];
            if (std::find(algos.begin(), algos.end(), order_algorithm) == algos.end()) {
                throw std::invalid_argument("Invalid order algorithm, valid values [" + Orders.ASC + ", " + Orders.DESC +
                    ", " + Orders.RAND + "]");
            }
            hyperparameters.erase("order");
        }
        if (hyperparameters.contains("alpha_block")) {
            alpha_block = hyperparameters["alpha_block"];
            hyperparameters.erase("alpha_block");
        }
        if (hyperparameters.contains("convergence")) {
            convergence = hyperparameters["convergence"];
            hyperparameters.erase("convergence");
        }
        if (hyperparameters.contains("convergence_best")) {
            convergence_best = hyperparameters["convergence_best"];
            hyperparameters.erase("convergence_best");
        }
        if (hyperparameters.contains("bisection")) {
            bisection = hyperparameters["bisection"];
            hyperparameters.erase("bisection");
        }
        if (hyperparameters.contains("threshold")) {
            threshold = hyperparameters["threshold"];
            hyperparameters.erase("threshold");
        }
        if (hyperparameters.contains("maxTolerance")) {
            maxTolerance = hyperparameters["maxTolerance"];
            if (maxTolerance < 1 || maxTolerance > 6)
                throw std::invalid_argument("Invalid maxTolerance value, must be greater in [1, 6]");
            hyperparameters.erase("maxTolerance");
        }
        if (hyperparameters.contains("predict_voting")) {
            predict_voting = hyperparameters["predict_voting"];
            hyperparameters.erase("predict_voting");
        }
        if (hyperparameters.contains("weightless")) {
            weightless = hyperparameters["weightless"];
            hyperparameters.erase("weightless");
        }
        if (hyperparameters.contains("select_features")) {
            auto selectedAlgorithm = hyperparameters["select_features"];
            std::vector<std::string> algos = { SelectFeatures.IWSS, SelectFeatures.CFS, SelectFeatures.FCBF };
            selectFeatures = true;
            select_features_algorithm = selectedAlgorithm;
            if (std::find(algos.begin(), algos.end(), selectedAlgorithm) == algos.end()) {
                throw std::invalid_argument("Invalid selectFeatures value, valid values [" + SelectFeatures.IWSS + ", " +
                    SelectFeatures.CFS + ", " + SelectFeatures.FCBF + "]");
            }
            hyperparameters.erase("select_features");
        }
        if (hyperparameters.contains("block_update")) {
            block_update = hyperparameters["block_update"];
            hyperparameters.erase("block_update");
        }
        if (block_update && alpha_block) {
            throw std::invalid_argument("alpha_block and block_update cannot be true at the same time");
        }
        if (block_update && !bisection) {
            throw std::invalid_argument("block_update needs bisection to be true");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void Boost::add_model(std::unique_ptr<Classifier> model, double significance)
    {
        models.push_back(std::move(model));
        n_models++;
        significanceModels.push_back(significance);
    }
    void Boost::remove_last_model()
    {
        models.pop_back();
        significanceModels.pop_back();
        n_models--;
    }
    void Boost::buildModel(const torch::Tensor& weights)
    {
        // Models shall be built in trainModel
        models.clear();
        significanceModels.clear();
        n_models = 0;
        // Prepare the validation dataset
        auto y_ = dataset.index({ -1, "..." });
        if (convergence) {
            // Prepare train & validation sets from train data
            auto fold = folding::StratifiedKFold(5, y_, 271);
            auto [train, test] = fold.getFold(0);
            auto train_t = torch::tensor(train);
            auto test_t = torch::tensor(test);
            // Get train and validation sets
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), train_t });
            y_train = dataset.index({ -1, train_t });
            X_test = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), test_t });
            y_test = dataset.index({ -1, test_t });
            dataset = X_train;
            m = X_train.size(1);
            auto n_classes = states.at(className).size();
            // Build dataset with train data
            buildDataset(y_train);
            metrics = Metrics(dataset, features, className, n_classes);
        } else {
            // Use all data to train
            X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." });
            y_train = y_;
        }
    }
    std::vector<int> Boost::featureSelection(torch::Tensor& weights_)
    {
        int maxFeatures = 0;
        if (select_features_algorithm == SelectFeatures.CFS) {
            featureSelector = new CFS(dataset, features, className, maxFeatures, states.at(className).size(), weights_);
        } else if (select_features_algorithm == SelectFeatures.IWSS) {
            if (threshold < 0 || threshold > 0.5) {
                throw std::invalid_argument("Invalid threshold value for " + SelectFeatures.IWSS + " [0, 0.5]");
            }
            featureSelector =
                new IWSS(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        } else if (select_features_algorithm == SelectFeatures.FCBF) {
            if (threshold < 1e-7 || threshold > 1) {
                throw std::invalid_argument("Invalid threshold value for " + SelectFeatures.FCBF + " [1e-7, 1]");
            }
            featureSelector =
                new FCBF(dataset, features, className, maxFeatures, states.at(className).size(), weights_, threshold);
        }
        featureSelector->fit();
        auto featuresUsed = featureSelector->getFeatures();
        delete featureSelector;
        return featuresUsed;
    }
    std::tuple<torch::Tensor&, double, bool> Boost::update_weights(torch::Tensor& ytrain, torch::Tensor& ypred,
        torch::Tensor& weights)
    {
        bool terminate = false;
        double alpha_t = 0;
        auto mask_wrong = ypred != ytrain;
        auto mask_right = ypred == ytrain;
        auto masked_weights = weights * mask_wrong.to(weights.dtype());
        double epsilon_t = masked_weights.sum().item<double>();
        // std::cout << "epsilon_t: " << epsilon_t << " count wrong: " << mask_wrong.sum().item<int>() << " count right: "
        // << mask_right.sum().item<int>() << std::endl;
        if (epsilon_t > 0.5) {
            // Inverse the weights policy (plot ln(wt))
            // "In each round of AdaBoost, there is a sanity check to ensure that the current base
            // learner is better than random guess" (Zhi-Hua Zhou, 2012)
            terminate = true;
        } else {
            double wt = (1 - epsilon_t) / epsilon_t;
            alpha_t = epsilon_t == 0 ? 1 : 0.5 * log(wt);
            // Step 3.2: Update weights for next classifier
            // Step 3.2.1: Update weights of wrong samples
            weights += mask_wrong.to(weights.dtype()) * exp(alpha_t) * weights;
            // Step 3.2.2: Update weights of right samples
            weights += mask_right.to(weights.dtype()) * exp(-alpha_t) * weights;
            // Step 3.3: Normalise the weights
            double totalWeights = torch::sum(weights).item<double>();
            weights = weights / totalWeights;
        }
        return { weights, alpha_t, terminate };
    }
    std::tuple<torch::Tensor&, double, bool> Boost::update_weights_block(int k, torch::Tensor& ytrain,
        torch::Tensor& weights)
    {
        /* Update Block algorithm
            k = # of models in block
            n_models = # of models in ensemble to make predictions
            n_models_bak = # models saved
            models = vector of models to make predictions
            models_bak = models not used to make predictions
            significances_bak = backup of significances vector

            Case list
            A) k = 1, n_models = 1		=> n = 0 , n_models = n + k
            B) k = 1, n_models = n + 1	=> n_models = n + k
            C) k > 1, n_models = k + 1 	=> n= 1, n_models = n + k
            D) k > 1, n_models = k		=> n = 0, n_models = n + k
            E) k > 1, n_models = k + n	=> n_models = n + k

            A, D) n=0, k > 0, n_models == k
            1. n_models_bak <- n_models
            2. significances_bak <- significances
            3. significances = vector(k, 1)
            4. Don’t move any classifiers out of models
            5. n_models <- k
            6. Make prediction, compute alpha, update weights
            7. Don’t restore any classifiers to models
            8. significances <- significances_bak
            9. Update last k significances
            10. n_models <- n_models_bak

            B, C, E) n > 0, k > 0, n_models == n + k
            1. n_models_bak <- n_models
            2. significances_bak <- significances
            3. significances = vector(k, 1)
            4. Move first n classifiers to models_bak
            5. n_models <- k
            6. Make prediction, compute alpha, update weights
            7. Insert classifiers in models_bak to be the first n models
            8. significances <- significances_bak
            9. Update last k significances
            10. n_models <- n_models_bak
        */
        //
        // Make predict with only the last k models
        //
        std::unique_ptr<Classifier> model;
        std::vector<std::unique_ptr<Classifier>> models_bak;
        // 1. n_models_bak <- n_models 2. significances_bak <- significances
        auto significance_bak = significanceModels;
        auto n_models_bak = n_models;
        // 3. significances = vector(k, 1)
        significanceModels = std::vector<double>(k, 1.0);
        // 4. Move first n classifiers to models_bak
        // backup the first n_models - k models (if n_models == k, don't backup any)
        for (int i = 0; i < n_models - k; ++i) {
            model = std::move(models[0]);
            models.erase(models.begin());
            models_bak.push_back(std::move(model));
        }
        assert(models.size() == k);
        // 5. n_models <- k
        n_models = k;
        // 6. Make prediction, compute alpha, update weights
        auto ypred = predict(X_train);
        //
        // Update weights
        //
        double alpha_t;
        bool terminate;
        std::tie(weights, alpha_t, terminate) = update_weights(y_train, ypred, weights);
        //
        // Restore the models if needed
        //
        // 7. Insert classifiers in models_bak to be the first n models
        // if n_models_bak == k, don't restore any, because none of them were moved
        if (k != n_models_bak) {
            // Insert in the same order as they were extracted
            int bak_size = models_bak.size();
            for (int i = 0; i < bak_size; ++i) {
                model = std::move(models_bak[bak_size - 1 - i]);
                models_bak.erase(models_bak.end() - 1);
                models.insert(models.begin(), std::move(model));
            }
        }
        // 8. significances <- significances_bak
        significanceModels = significance_bak;
        //
        // Update the significance of the last k models
        //
        // 9. Update last k significances
        for (int i = 0; i < k; ++i) {
            significanceModels[n_models_bak - k + i] = alpha_t;
        }
        // 10. n_models <- n_models_bak
        n_models = n_models_bak;
        return { weights, alpha_t, terminate };
    }
} // namespace bayesnet
