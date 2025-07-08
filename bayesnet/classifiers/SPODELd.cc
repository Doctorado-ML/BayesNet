// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "SPODELd.h"

namespace bayesnet {
    SPODELd::SPODELd(int root) : SPODE(root), Proposal(dataset, features, className)
    {
        validHyperparameters = validHyperparameters_ld; // Inherits the valid hyperparameters from Proposal
    }

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
        states = iterativeLocalDiscretization(y, static_cast<SPODE*>(this), dataset, features, className, states_, smoothing);
        SPODE::fit(dataset, features, className, states, smoothing);
        return *this;
    }
    torch::Tensor SPODELd::predict(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict(Xt);
    }
    torch::Tensor SPODELd::predict_proba(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return SPODE::predict_proba(Xt);
    }
    std::vector<std::string> SPODELd::graph(const std::string& name) const
    {
        return SPODE::graph(name);
    }
}