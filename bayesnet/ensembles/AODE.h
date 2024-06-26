// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef AODE_H
#define AODE_H
#include "bayesnet/classifiers/SPODE.h"
#include "Ensemble.h"
namespace bayesnet {
    class AODE : public Ensemble {
    public:
        AODE(bool predict_voting = false);
        virtual ~AODE() {};
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
        std::vector<std::string> graph(const std::string& title = "AODE") const override;
    protected:
        void buildModel(const torch::Tensor& weights) override;
    };
}
#endif