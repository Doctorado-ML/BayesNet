// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "KDBLd.h"
#include <memory>

namespace bayesnet {
    KDBLd::KDBLd(int k) : KDB(k), Proposal(dataset, features, className, KDB::notes)
    {
        validHyperparameters = validHyperparameters_ld;
        validHyperparameters.push_back("k");
        validHyperparameters.push_back("theta");
    }
    KDBLd& KDBLd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        Xf = X_;
        y = y_;
        return commonFit(features_, className_, states_, smoothing);
    }
    KDBLd& KDBLd::fit(torch::Tensor& dataset, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        if (!torch::is_floating_point(dataset)) {
            throw std::runtime_error("Dataset must be a floating point tensor");
        }
        Xf = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), "..." }).clone();
        y = dataset.index({ -1, "..." }).clone().to(torch::kInt32);
        return commonFit(features_, className_, states_, smoothing);
    }

    KDBLd& KDBLd::commonFit(const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        features = features_;
        className = className_;
        states = iterativeLocalDiscretization(y, static_cast<KDB*>(this), dataset, features, className, states_, smoothing);
        KDB::fit(dataset, features, className, states, smoothing);
        fitted = true;
        return *this;
    }
    torch::Tensor KDBLd::predict(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return KDB::predict(Xt);
    }
    torch::Tensor KDBLd::predict_proba(torch::Tensor& X)
    {
        auto Xt = prepareX(X);
        return KDB::predict_proba(Xt);
    }
    std::vector<std::string> KDBLd::graph(const std::string& name) const
    {
        return KDB::graph(name);
    }
}
