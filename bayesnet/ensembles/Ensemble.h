// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "bayesnet/utils/BayesMetrics.h"
#include "bayesnet/utils/bayesnetUtils.h"
#include "bayesnet/classifiers/Classifier.h"

namespace bayesnet {
    class Ensemble : public Classifier {
    public:
        Ensemble(bool predict_voting = true);
        virtual ~Ensemble() = default;
        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        long getNumberOfStates() const override;
        std::vector<std::string> show() const override;
        std::vector<std::string> graph(const std::string& title) const override;
        std::vector<std::string> topological_order()  override
        {
            return std::vector<std::string>();
        }
        std::string dump_cpt() const override
        {
            std::string output;
            for (auto& model : models) {
                output += model->dump_cpt();
                output += std::string(80, '-') + "\n";
            }
            return output;
        }
    protected:
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
        torch::Tensor predict_average_voting(torch::Tensor& X);
        std::vector<std::vector<double>> predict_average_voting(std::vector<std::vector<int>>& X);
        torch::Tensor predict_average_proba(torch::Tensor& X);
        std::vector<std::vector<double>> predict_average_proba(std::vector<std::vector<int>>& X);
        torch::Tensor compute_arg_max(torch::Tensor& X);
        std::vector<int> compute_arg_max(std::vector<std::vector<double>>& X);
        torch::Tensor voting(torch::Tensor& votes);
        // Sum of the significances the predictions are divided by. Throws
        // std::logic_error when the ensemble has no models or the sum is 0, which
        // would make every probability 0/0 = NaN (BayesNet#48).
        double significanceSum() const;
        // Attributes
        unsigned n_models;
        std::vector<std::unique_ptr<Classifier>> models;
        std::vector<double> significanceModels;
        bool predict_voting;
    };
}
#endif
