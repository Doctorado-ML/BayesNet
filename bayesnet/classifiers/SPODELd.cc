// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "SPODELd.h"

namespace bayesnet {
    SPODELd::SPODELd(int root) : SPODE(root), Proposal(dataset, features, className) {}
    SPODELd& SPODELd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        Xf = X_;
        y = y_;
        return commonFit(features_, className_, states_, smoothing);
    }

    SPODELd& SPODELd::fit(torch::Tensor& dataset, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        if (!torch::is_floating_point(dataset)) {
            throw std::runtime_error("Dataset must be a floating point tensor");
        }
        Xf = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." }).clone();
        y = dataset.index({ -1, "..." }).clone().to(torch::kInt32);
        return commonFit(features_, className_, states_, smoothing);
    }

    SPODELd& SPODELd::commonFit(const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        features = features_;
        className = className_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal SPODE structure, SPODE::fit initializes the base Bayesian network
        SPODE::fit(dataset, features, className, states, smoothing);
        states = localDiscretizationProposal(states, model);
        return *this;
    }
    torch::Tensor SPODELd::predict(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict(Xt);
    }
    torch::Tensor SPODELd::predict_proba(torch::Tensor& X)
    {
        std::cout << "Debug: SPODELd::predict_proba" << std::endl;
        auto Xt = prepareX(X);
        return SPODE::predict_proba(Xt);
    }
    std::vector<std::string> SPODELd::graph(const std::string& name) const
    {
        return SPODE::graph(name);
    }
}