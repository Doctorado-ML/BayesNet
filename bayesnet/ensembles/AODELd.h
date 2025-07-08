// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef AODELD_H
#define AODELD_H
#include "bayesnet/classifiers/Proposal.h"
#include "bayesnet/classifiers/SPODELd.h"
#include "Ensemble.h"

namespace bayesnet {
    class AODELd : public Ensemble, public Proposal {
    public:
        AODELd(bool predict_voting = true);
        virtual ~AODELd() = default;
        AODELd& fit(torch::Tensor& X_, torch::Tensor& y_, const std::vector<std::string>& features_, const std::string& className_, map<std::string, std::vector<int>>& states_, const Smoothing_t smoothing) override;
        std::vector<std::string> graph(const std::string& name = "AODELd") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override
        {
            hyperparameters = hyperparameters_;
        }
    protected:
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
        void buildModel(const torch::Tensor& weights) override;
    private:
        nlohmann::json hyperparameters = {}; // Hyperparameters for the model
    };
}
#endif // !AODELD_H