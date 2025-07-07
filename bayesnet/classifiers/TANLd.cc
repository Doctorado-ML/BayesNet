// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "TANLd.h"
#include <memory>

namespace bayesnet {
    TANLd::TANLd() : TAN(), Proposal(dataset, features, className) {}
    TANLd& TANLd::fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing)
    {
        checkInput(X_, y_);
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        
        // Use iterative local discretization instead of the two-phase approach
        states = iterativeLocalDiscretization(y, static_cast<TAN*>(this), dataset, features, className, states_, smoothing);
        
        // Final fit with converged discretization
        TAN::fit(dataset, features, className, states, smoothing);
        
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
