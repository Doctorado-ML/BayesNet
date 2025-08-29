// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "TANLd.h"
#include <memory>

namespace bayesnet {
    TANLd::TANLd() : TAN(), Proposal(dataset, features, className, TAN::notes)
    {
        validHyperparameters = validHyperparameters_ld; // Inherits the valid hyperparameters from Proposal
    }
    TANLd& TANLd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        Xf = X_;
        y = y_;
        return commonFit(features_, className_, states_, smoothing);
    }
    TANLd& TANLd::fit(torch::Tensor& dataset, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        if (!torch::is_floating_point(dataset)) {
            throw std::runtime_error("Dataset must be a floating point tensor");
        }
        Xf = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." }).clone();
        y = dataset.index({ -1, "..." }).clone().to(torch::kInt32);
        return commonFit(features_, className_, states_, smoothing);
    }

    TANLd& TANLd::commonFit(const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        features = features_;
        className = className_;
        states = iterativeLocalDiscretization(y, static_cast<TAN*>(this), dataset, features, className, states_, smoothing);
        TAN::fit(dataset, features, className, states, smoothing);
        fitted = true;
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
