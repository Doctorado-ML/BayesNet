// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef NETWORK_H
#define NETWORK_H
#include <map>
#include <vector>
#include "bayesnet/config.h"
#include "Node.h"
#include "Smoothing.h"

namespace bayesnet {

    class Network {
    public:
        Network();
        Network(const Network& other);
        Network& operator=(const Network& other);
        ~Network() = default;
        torch::Tensor& getSamples();
        void addNode(const std::string&);
        void addEdge(const std::string&, const std::string&);
        std::map<std::string, std::unique_ptr<Node>>& getNodes();
        std::vector<std::string> getFeatures() const;
        int getStates() const;
        std::vector<std::pair<std::string, std::string>> getEdges() const;
        int getNumEdges() const;
        int getClassNumStates() const;
        std::string getClassName() const;
        /*
        Notice: Nodes have to be inserted in the same order as they are in the dataset, i.e., first node is first column and so on.
        */
        void fit(const std::vector<std::vector<int>>& input_data, const std::vector<int>& labels, const std::vector<double>& weights, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing);
        void fit(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& weights, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing);
        void fit(const torch::Tensor& samples, const torch::Tensor& weights, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing);
        std::vector<int> predict(const std::vector<std::vector<int>>&); // Return mx1 std::vector of predictions
        torch::Tensor predict(const torch::Tensor&); // Return mx1 tensor of predictions
        torch::Tensor predict_tensor(const torch::Tensor& samples, const bool proba);
        std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<int>>&); // Return mxn std::vector of probabilities
        torch::Tensor predict_proba(const torch::Tensor&); // Return mxn tensor of probabilities
        double score(const std::vector<std::vector<int>>&, const std::vector<int>&);
        std::vector<std::string> topological_sort();
        std::vector<std::string> show() const;
        std::vector<std::string> graph(const std::string& title) const; // Returns a std::vector of std::strings representing the graph in graphviz format
        void initialize();
        std::string dump_cpt() const;
        inline std::string version() { return  { project_version.begin(), project_version.end() }; }
        bool operator==(const Network& other) const;
    private:
        std::map<std::string, std::unique_ptr<Node>> nodes;
        bool fitted;
        int classNumStates;
        std::vector<std::string> features; // Including classname
        std::string className;
        torch::Tensor samples; // n+1xm tensor used to fit the model
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
        std::vector<double> predict_sample(const std::vector<int>&);
        std::vector<double> predict_sample(const torch::Tensor&);
        std::vector<double> exactInference(std::map<std::string, int>&);
        void completeFit(const std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const Smoothing_t smoothing);
        void checkFitData(int n_samples, int n_features, int n_samples_y, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights);
        void setStates(const std::map<std::string, std::vector<int>>&);
    };
}
#endif