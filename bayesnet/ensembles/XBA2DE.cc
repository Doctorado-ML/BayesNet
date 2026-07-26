// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <folding.hpp>
#include <algorithm>
#include <set>
#include <limits.h>
#include "XBA2DE.h"
#include "bayesnet/classifiers/XSP2DE.h"
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {

XBA2DE::XBA2DE(bool predict_voting) : Boost(predict_voting) {
    // XBA2DE fixes several knobs to keep the algorithm lean, mirroring how the
    // BoostAODE experiments were actually run: it always starts from an empty
    // ensemble (no CFS/IWSS/FCBF seeding) and always uses plain bisection (no
    // block_update, no alpha_block). Those hyperparameters are dropped here and
    // the beta pair-ranking criterion is added.
    auto& vh = validHyperparameters;
    vh.erase(std::remove_if(vh.begin(), vh.end(), [](const std::string& h) {
        return h == "select_features" || h == "threshold" || h == "bisection" ||
               h == "block_update" || h == "alpha_block";
    }), vh.end());
    vh.push_back("beta");
}
void XBA2DE::setHyperparameters(const nlohmann::json& hyperparameters_) {
    auto hyperparameters = hyperparameters_;
    // These knobs are fixed in XBA2DE (see the constructor) and therefore not
    // accepted: empty-start (no seeding) and plain bisection are always on.
    for (const auto& forbidden : { "select_features", "threshold", "bisection", "block_update", "alpha_block" }) {
        if (hyperparameters.contains(forbidden)) {
            throw std::invalid_argument(std::string("XBA2DE does not support the '") + forbidden + "' hyperparameter");
        }
    }
    if (hyperparameters.contains("beta")) {
        beta_ = hyperparameters["beta"];
        if (beta_ < 0.0) {
            throw std::invalid_argument("Invalid beta value, must be >= 0");
        }
        hyperparameters.erase("beta");
    }
    // Hand off the rest to the boosting base.
    Boost::setHyperparameters(hyperparameters);
}
void XBA2DE::trainModel(const torch::Tensor &weights, const Smoothing_t smoothing) {
    //
    // Logging setup
    //
    // loguru::set_thread_name("XBA2DE");
    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    // loguru::add_file("boostA2DE.log", loguru::Truncate, loguru::Verbosity_MAX);

    // Algorithm based on the adaboost algorithm for classification
    // as explained in Ensemble methods (Zhi-Hua Zhou, 2012)
    X_train_ = TensorUtils::to_matrix(X_train);
    y_train_ = TensorUtils::to_vector<int>(y_train);
    if (convergence) {
        X_test_ = TensorUtils::to_matrix(X_test);
        y_test_ = TensorUtils::to_vector<int>(y_test);
    }
    fitted = true;
    double alpha_t = 0;
    torch::Tensor weights_ = torch::full({m}, 1.0 / m, torch::kFloat64);
    bool finished = false;
    // XBA2DE always starts from an empty ensemble; no feature-selection seeding.
    // Pairs already added to the ensemble. The boosting unit here is the PAIR,
    // not a single superparent: a pair is used at most once, but its two
    // variables may still pair with OTHER variables (that is expected). Allowing
    // a pair to repeat with updated weights could be a future hyperparameter.
    std::set<std::pair<int, int>> pairsUsed;
    std::vector<int> featuresExcluded; // XBA2DE does no feature-level exclusion
    int numItemsPack = 0; // The counter of the models inserted in the current pack
    // Variables to control the accuracy finish condition
    double priorAccuracy = 0.0;
    double improvement = 1.0;
    double convergence_threshold = 1e-4;
    int tolerance = 0; // number of times the accuracy is lower than the convergence_threshold
    // Step 0: Set the finish condition
    // epsilon sub t > 0.5 => inverse the weights policy
    // validation error is not decreasing
    // run out of features
    bool ascending = order_algorithm == Orders.ASC;
    std::mt19937 g{173};
    std::vector<std::pair<int, int>> pairSelection;
    while (!finished) {
        // Step 1: Build ranking with mutual information
        pairSelection = metrics.SelectKPairs(weights_, featuresExcluded, ascending, 0, beta_); // Get all the pairs sorted by joint relevance
        if (order_algorithm == Orders.RAND) {
            std::shuffle(pairSelection.begin(), pairSelection.end(), g);
        }
        // Remove pairs already used (boosting without replacement of pairs).
        pairSelection.erase(std::remove_if(pairSelection.begin(), pairSelection.end(),
            [&](const std::pair<int, int>& p) { return pairsUsed.count(p) > 0; }), pairSelection.end());
        int k = pow(2, tolerance); // XBA2DE always uses bisection
        int counter = 0; // The model counter of the current pack
        // VLOG_SCOPE_F(1, "counter=%d k=%d featureSelection.size: %zu", counter, k, featureSelection.size());
        while (counter++ < k && pairSelection.size() > 0) {
            auto feature_pair = pairSelection[0];
            pairSelection.erase(pairSelection.begin());
            std::unique_ptr<Classifier> model;
            model = std::make_unique<XSp2de>(feature_pair.first, feature_pair.second);
            model->fit(dataset, features, className, states, weights_, smoothing);
            alpha_t = 0.0;
            auto ypred = model->predict(X_train);
            // Step 3.1: Compute the classifier amount of say
            std::tie(weights_, alpha_t, finished) = update_weights(y_train, ypred, weights_);
            // Step 3.4: Store classifier and its accuracy to weigh its future vote
            numItemsPack++;
            pairsUsed.insert(feature_pair);
            models.push_back(std::move(model));
            significanceModels.push_back(alpha_t);
            n_models++;
            // VLOG_SCOPE_F(2, "numItemsPack: %d n_models: %d featuresUsed: %zu", numItemsPack, n_models,
            // featuresUsed.size());
        }
        if (convergence && !finished) {
            auto y_val_predict = predict(X_test);
            double accuracy = (y_val_predict == y_test).sum().item<double>() / (double)y_test.size(0);
            if (priorAccuracy == 0) {
                priorAccuracy = accuracy;
            } else {
                improvement = accuracy - priorAccuracy;
            }
            if (improvement < convergence_threshold) {
                // VLOG_SCOPE_F(3, "  (improvement<threshold) tolerance: %d numItemsPack: %d improvement: %f prior: %f
                // current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                tolerance++;
            } else {
                // VLOG_SCOPE_F(3, "* (improvement>=threshold) Reset. tolerance: %d numItemsPack: %d improvement: %f
                // prior: %f current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                tolerance = 0; // Reset the counter if the model performs better
                numItemsPack = 0;
            }
            if (convergence_best) {
                // Keep the best accuracy until now as the prior accuracy
                priorAccuracy = std::max(accuracy, priorAccuracy);
            } else {
                // Keep the last accuray obtained as the prior accuracy
                priorAccuracy = accuracy;
            }
        }
        // VLOG_SCOPE_F(1, "tolerance: %d featuresUsed.size: %zu features.size: %zu", tolerance, featuresUsed.size(),
        // features.size());
        finished = finished || tolerance > maxTolerance || pairSelection.size() == 0;
    }
    if (tolerance > maxTolerance) {
        if (numItemsPack < n_models) {
            notes.push_back("Convergence threshold reached & " + std::to_string(numItemsPack) + " models eliminated");
            // VLOG_SCOPE_F(4, "Convergence threshold reached & %d models eliminated of %d", numItemsPack, n_models);
            for (int i = 0; i < numItemsPack; ++i) {
                significanceModels.pop_back();
                models.pop_back();
                n_models--;
            }
        } else {
            notes.push_back("Convergence threshold reached & 0 models eliminated");
            // VLOG_SCOPE_F(4, "Convergence threshold reached & 0 models eliminated n_models=%d numItemsPack=%d",
            // n_models, numItemsPack);
        }
    }
    if (pairSelection.size() > 0) {
        notes.push_back("Pairs not used in train: " + std::to_string(pairSelection.size()));
        status = WARNING;
    }
    notes.push_back("Number of models: " + std::to_string(n_models));
}
std::vector<std::string> XBA2DE::graph(const std::string &title) const { return Ensemble::graph(title); }
} // namespace bayesnet
