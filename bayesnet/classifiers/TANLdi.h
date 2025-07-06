// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef TANLDI_H
#define TANLDI_H
#include "TAN.h"
#include "IterativeProposal.h"

namespace bayesnet {
    class TANLdi : public TAN, public IterativeProposal {
    private:
    public:
        TANLdi();
        virtual ~TANLdi() = default;
        TANLdi& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        std::vector<std::string> graph(const std::string& name = "TANLdi") const override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
    };
}
#endif // !TANLDI_H