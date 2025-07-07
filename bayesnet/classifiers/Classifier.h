// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <torch/torch.h>
#include "bayesnet/utils/BayesMetrics.h"
#include "bayesnet/BaseClassifier.h"

namespace bayesnet {
    class Classifier : public BaseClassifier {
    public:
        Classifier(Network model);
        virtual ~Classifier() = default;
        Classifier& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        Classifier& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        Classifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        Classifier& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const Smoothing_t smoothing) override;
        void addNodes();
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        int getClassNumStates() const override;
        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        status_t getStatus() const override { return status; }
        std::string getVersion() override { return { project_version.begin(), project_version.end() }; };
        float score(torch::Tensor& X, torch::Tensor& y) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        std::vector<std::string> show() const override;
        std::vector<std::string> topological_order()  override;
        std::vector<std::string> getNotes() const override { return notes; }
        std::string dump_cpt() const override;
        void setHyperparameters(const nlohmann::json& hyperparameters) override; //For classifiers that don't have hyperparameters
        Network& getModel() { return model; }
    protected:
        bool fitted;
        unsigned int m, n; // m: number of samples, n: number of features
        Network model;
        Metrics metrics;
        std::vector<std::string> features;
        std::string className;
        std::map<std::string, std::vector<int>> states;
        torch::Tensor dataset; // (n+1)xm tensor
        void checkFitParameters();
        virtual void buildModel(const torch::Tensor& weights) = 0;
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;
        void buildDataset(torch::Tensor& y);
        const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";
    private:
        Classifier& build(const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const Smoothing_t smoothing);
    };
}
#endif





