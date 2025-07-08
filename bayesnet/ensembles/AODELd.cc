// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "AODELd.h"

namespace bayesnet {
    AODELd::AODELd(bool predict_voting) : Ensemble(predict_voting), Proposal(dataset, features, className, Ensemble::notes)
    {
        validHyperparameters = validHyperparameters_ld; // Inherits the valid hyperparameters from Proposal
    }
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal AODE structure, Ensemble::fit  
        // calls buildModel to initialize the base models
        Ensemble::fit(dataset, features, className, states, smoothing);
        return *this;

    }
    void AODELd::buildModel(const torch::Tensor& weights)
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODELd>(i));
            models.back()->setHyperparameters(hyperparameters);
        }
        n_models = models.size();
        significanceModels = std::vector<double>(n_models, 1.0);
    }
    void AODELd::trainModel(const torch::Tensor& weights, const Smoothing_t smoothing)
    {
        for (const auto& model : models) {
            model->fit(Xf, y, features, className, states, smoothing);
        }
    }
    std::vector<std::string> AODELd::graph(const std::string& name) const
    {
        return Ensemble::graph(name);
    }
}