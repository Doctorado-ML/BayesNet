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
        // Working-memory budget for the ensemble, in gigabytes (1 GB = 2^30 bytes).
        // 0 means unlimited. It bounds the tables of the accumulated XSp2de models
        // only -- not the dataset, the metrics or the pair ranking.
        double max_memory_gb_ = 0.0;
        std::string version = "0.9.8";
    };
}
#endif
