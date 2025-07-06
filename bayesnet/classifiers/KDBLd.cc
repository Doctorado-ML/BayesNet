// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "KDBLd.h"

namespace bayesnet {
    KDBLd::KDBLd(int k) : KDB(k), Proposal(dataset, features, className)
    {
        validHyperparameters = validHyperparameters_ld;
        validHyperparameters.push_back("k");
        validHyperparameters.push_back("theta");
    }
    void KDBLd::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("k")) {
            k = hyperparameters["k"];
            hyperparameters.erase("k");
        }
        if (hyperparameters.contains("theta")) {
            theta = hyperparameters["theta"];
            hyperparameters.erase("theta");
        }
        Proposal::setHyperparameters(hyperparameters);
    }
    KDBLd& KDBLd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        
        // Use iterative local discretization instead of the two-phase approach
        states = iterativeLocalDiscretization(y, this, dataset, features, className, states_, smoothing);
        
        // Final fit with converged discretization
        KDB::fit(dataset, features, className, states, smoothing);
        
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