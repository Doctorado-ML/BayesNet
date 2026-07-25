// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XA2DE_H
#define XA2DE_H
#include <vector>
#include <string>
#include <torch/torch.h>
#include <bayesnet/ensembles/Boost.h>
#include <bayesnet/network/Smoothing.h>
#include "bayesnet/utils/CountingSemaphore.h"
#include "Xaode2de.hpp"

namespace bayesnet {
    // Optimized A2DE: an ensemble of the C(n,2) SP2DE submodels evaluated through
    // the flat Xaode2de engine (no boosting, uniform significance). The A2DE
    // counterpart of XA1DE.
    class XA2DE : public Boost {
    public:
        XA2DE();
        virtual ~XA2DE() override = default;
        std::string getVersion() { return version; };
        // BaseClassifier interface (delegates to the flat engine, bypassing Network)
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<int>>& X);
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        int getClassNumStates() const override;
        std::vector<std::string> show() const override { return {}; }
        std::vector<std::string> topological_order()  override { return {}; }
        std::string dump_cpt() const override { return ""; }
        void setDebug(bool debug) { this->debug = debug; }
        bayesnet::status_t getStatus() const override { return status; }
        std::vector<std::string> getNotes() const override { return notes; }
        std::vector<std::string> graph(const std::string& title = "") const override { return {}; }
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
    protected:
        void buildModel(const torch::Tensor& weights) override {};
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
        bool debug = false;
        Xaode2de aode_;
        torch::Tensor weights_;
        bool joint_parents_ = true;
        const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";
    private:
        std::string version = "1.0.0";
        CountingSemaphore& semaphore_;
    };
}
#endif // XA2DE_H
