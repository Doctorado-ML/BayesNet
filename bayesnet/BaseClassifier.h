// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#pragma once
#include <vector>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include "bayesnet/network/Network.h"

namespace bayesnet {
    enum status_t { NORMAL, WARNING, ERROR };
    class BaseClassifier {
    public:
        virtual ~BaseClassifier() = default;
        // X is nxm std::vector, y is nx1 std::vector
        virtual BaseClassifier& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) = 0;
        // X is nxm tensor, y is nx1 tensor
        virtual BaseClassifier& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) = 0;
        virtual BaseClassifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) = 0;
        virtual BaseClassifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const Smoothing_t smoothing) = 0;
        torch::Tensor virtual predict(torch::Tensor& X) = 0;
        std::vector<int> virtual predict(std::vector<std::vector<int >>& X) = 0;
        torch::Tensor virtual predict_proba(torch::Tensor& X) = 0;
        std::vector<std::vector<double>> virtual predict_proba(std::vector<std::vector<int >>& X) = 0;
        status_t virtual getStatus() const = 0;
        float virtual score(std::vector<std::vector<int>>& X, std::vector<int>& y) = 0;
        float virtual score(torch::Tensor& X, torch::Tensor& y) = 0;
        int virtual getNumberOfNodes() const = 0;
        int virtual getNumberOfEdges() const = 0;
        // Size of the model. For the X engines (XSpode, XSp2de, Xaode, Xaode2de and
        // the ensembles built on them) it is the number of cells of the probability
        // tables, so 8 * getNumberOfStates() bytes is the size of the model as
        // doubles; for the Network based classifiers it is the sum of the node
        // cardinalities. It is a long because A2DE on a few hundred features
        // exceeds the range of an int.
        long virtual getNumberOfStates() const = 0;
        int virtual getClassNumStates() const = 0;
        std::vector<std::string> virtual show() const = 0;
        std::vector<std::string> virtual graph(const std::string& title = "") const = 0;
        virtual std::string getVersion() = 0;
        std::vector<std::string> virtual topological_order() = 0;
        std::vector<std::string> virtual getNotes() const = 0;
        std::string virtual dump_cpt() const = 0;
        virtual void setHyperparameters(const nlohmann::json& hyperparameters) = 0;
        std::vector<std::string>& getValidHyperparameters() { return validHyperparameters; }
    protected:
        virtual void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) = 0;
        std::vector<std::string> validHyperparameters;
        std::vector<std::string> notes; // Used to store messages occurred during the fit process
        status_t status = NORMAL;
    };
}