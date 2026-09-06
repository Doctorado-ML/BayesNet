// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XBA2DE_H
#define XBA2DE_H
#include <string>
#include <vector>
#include "Boost.h"
namespace bayesnet {
    class XBA2DE : public Boost {
    public:
        explicit XBA2DE(bool predict_voting = false);
        virtual ~XBA2DE() = default;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
        std::vector<std::string> graph(const std::string& title = "XBA2DE") const override;
        std::string getVersion() override { return version; };
    protected:
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
    private:
        // Pair-ranking criterion knob (joint relevance). beta=1 joint relevance
        // (default), beta=0 marginal-relevance sum, beta large pure synergy.
        double beta_ = 1.0;
        std::string version = "0.9.7";
    };
}
#endif
