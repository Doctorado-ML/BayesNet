// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef SPODELD_H
#define SPODELD_H
#include "SPODE.h"
#include "Proposal.h"

namespace bayesnet {
    class SPODELd : public SPODE, public Proposal {
    public:
        explicit SPODELd(int root);
        virtual ~SPODELd() = default;
        SPODELd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        SPODELd& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        SPODELd& fit_disc(torch::Tensor& Xf_, torch::Tensor& dataset_, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing, const std::vector<bool> wasNumeric_);
        SPODELd& commonFit(const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing);
        std::vector<std::string> graph(const std::string& name = "SPODELd") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override
        {
            auto hyperparameters = hyperparameters_;
            Proposal::setHyperparameters(hyperparameters);
            SPODE::setHyperparameters(hyperparameters);
        }
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !SPODELD_H