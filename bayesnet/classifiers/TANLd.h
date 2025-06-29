// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef TANLD_H
#define TANLD_H
#include "TAN.h"
#include "Proposal.h"

namespace bayesnet {
    class TANLd : public TAN, public Proposal {
    private:
    public:
        TANLd();
        virtual ~TANLd() = default;
        TANLd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        std::vector<std::string> graph(const std::string& name = "TANLd") const override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
    };
}
#endif // !TANLD_H