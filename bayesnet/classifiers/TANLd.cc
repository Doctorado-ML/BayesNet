// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "TANLd.h"

namespace bayesnet {
    TANLd::TANLd() : TAN(), Proposal(dataset, features, className) {}
    TANLd& TANLd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills std::vectors Xv & yv with the data from tensors X_ (discretized) & y
        states = fit_local_discretization(y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal TAN structure, TAN::fit initializes the base Bayesian network
        TAN::fit(dataset, features, className, states, smoothing);
        states = localDiscretizationProposal(states, model);
        return *this;

    }
    torch::Tensor TANLd::predict(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return TAN::predict(Xt);
    }
    torch::Tensor TANLd::predict_proba(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return TAN::predict_proba(Xt);
    }
    std::vector<std::string> TANLd::graph(const std::string& name) const
    {
        return TAN::graph(name);
    }
}