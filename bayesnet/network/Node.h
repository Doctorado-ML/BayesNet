// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef NODE_H
#define NODE_H
#include <unordered_set>
#include <vector>
#include <string>
#include <torch/torch.h>
namespace bayesnet {
    class Node {
    public:
        explicit Node(const std::string&);
        Node(const Node& other);
        Node& operator=(const Node& other);
        ~Node() = default;
        void clear();
        void addParent(Node*);
        void addChild(Node*);
        void removeParent(Node*);
        void removeChild(Node*);
        std::string getName() const;
        std::vector<Node*>& getParents();
        std::vector<Node*>& getChildren();
        torch::Tensor& getCPT();
        void computeCPT(const torch::Tensor& dataset, const std::vector<std::string>& features, const double smoothing, const torch::Tensor& weights);
        int getNumStates() const;
        void setNumStates(int);
        unsigned minFill();
        std::vector<std::string> graph(const std::string& clasName); // Returns a std::vector of std::strings representing the graph in graphviz format
        double getFactorValue(std::map<std::string, int>&);
    private:
        std::string name;
        std::vector<Node*> parents;
        std::vector<Node*> children;
        int numStates = 0; // number of states of the variable
        torch::Tensor cpTable; // Order of indices is 0-> node variable, 1-> 1st parent, 2-> 2nd parent, ...
        std::vector<int64_t> dimensions; // dimensions of the cpTable
        std::vector<std::pair<std::string, std::string>> combinations(const std::vector<std::string>&);
    };
}
#endif