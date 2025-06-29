// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "Node.h"
#include <iterator>

namespace bayesnet {

    Node::Node(const std::string& name)
        : name(name)
    {
    }
    void Node::clear()
    {
        parents.clear();
        children.clear();
        cpTable = torch::Tensor();
        dimensions.clear();
        numStates = 0;
    }
    std::string Node::getName() const
    {
        return name;
    }
    void Node::addParent(Node* parent)
    {
        parents.push_back(parent);
    }
    void Node::removeParent(Node* parent)
    {
        parents.erase(std::remove(parents.begin(), parents.end(), parent), parents.end());
    }
    void Node::removeChild(Node* child)
    {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }
    void Node::addChild(Node* child)
    {
        children.push_back(child);
    }
    std::vector<Node*>& Node::getParents()
    {
        return parents;
    }
    std::vector<Node*>& Node::getChildren()
    {
        return children;
    }
    int Node::getNumStates() const
    {
        return numStates;
    }
    void Node::setNumStates(int numStates)
    {
        this->numStates = numStates;
    }
    torch::Tensor& Node::getCPT()
    {
        return cpTable;
    }
    /*
     The MinFill criterion is a heuristic for variable elimination.
     The variable that minimizes the number of edges that need to be added to the graph to make it triangulated.
     This is done by counting the number of edges that need to be added to the graph if the variable is eliminated.
     The variable with the minimum number of edges is chosen.
     Here this is done computing the length of the combinations of the node neighbors taken 2 by 2.
    */
    unsigned Node::minFill()
    {
        std::unordered_set<std::string> neighbors;
        for (auto child : children) {
            neighbors.emplace(child->getName());
        }
        for (auto parent : parents) {
            neighbors.emplace(parent->getName());
        }
        auto source = std::vector<std::string>(neighbors.begin(), neighbors.end());
        return combinations(source).size();
    }
    std::vector<std::pair<std::string, std::string>> Node::combinations(const std::vector<std::string>& source)
    {
        std::vector<std::pair<std::string, std::string>> result;
        for (int i = 0; i < source.size(); ++i) {
            std::string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back({ temp, source[j] });
            }
        }
        return result;
    }
    void Node::computeCPT(const torch::Tensor& dataset, const std::vector<std::string>& features, const double smoothing, const torch::Tensor& weights)
    {
        dimensions.clear();
        dimensions.reserve(parents.size() + 1);
        dimensions.push_back(numStates);
        for (const auto& parent : parents) {
            dimensions.push_back(parent->getNumStates());
        }
        cpTable = torch::full(dimensions, smoothing, torch::kDouble);

        // Build feature index map
        std::unordered_map<std::string, int> featureIndexMap;
        for (size_t i = 0; i < features.size(); ++i) {
            featureIndexMap[features[i]] = i;
        }

        // Gather indices for node and parents
        std::vector<int64_t> all_indices;
        all_indices.push_back(featureIndexMap[name]);
        for (const auto& parent : parents) {
            all_indices.push_back(featureIndexMap[parent->getName()]);
        }

        // Extract relevant columns: shape (num_features, num_samples)
        auto indices_tensor = dataset.index_select(0, torch::tensor(all_indices, torch::kLong));
        indices_tensor = indices_tensor.transpose(0, 1).to(torch::kLong); // (num_samples, num_features)

        // Manual flattening of indices
        std::vector<int64_t> strides(all_indices.size(), 1);
        for (int i = strides.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * cpTable.size(i + 1);
        }
        auto indices_tensor_cpu = indices_tensor.cpu();
        auto indices_accessor = indices_tensor_cpu.accessor<int64_t, 2>();
        std::vector<int64_t> flat_indices(indices_tensor.size(0));
        for (int64_t i = 0; i < indices_tensor.size(0); ++i) {
            int64_t idx = 0;
            for (size_t j = 0; j < strides.size(); ++j) {
                idx += indices_accessor[i][j] * strides[j];
            }
            flat_indices[i] = idx;
        }

        // Accumulate weights into flat CPT
        auto flat_cpt = cpTable.flatten();
        auto flat_indices_tensor = torch::from_blob(flat_indices.data(), { (int64_t)flat_indices.size() }, torch::kLong).clone();
        flat_cpt.index_add_(0, flat_indices_tensor, weights.cpu());
        cpTable = flat_cpt.view(cpTable.sizes());

        // Normalize the counts (dividing each row by the sum of the row)
        cpTable /= cpTable.sum(0, true);
    }
    double Node::getFactorValue(std::map<std::string, int>& evidence)
    {
        c10::List<c10::optional<at::Tensor>> coordinates;
        // following predetermined order of indices in the cpTable (see Node.h)
        coordinates.push_back(at::tensor(evidence[name]));
        transform(parents.begin(), parents.end(), std::back_inserter(coordinates), [&evidence](const auto& parent) { return at::tensor(evidence[parent->getName()]); });
        return cpTable.index({ coordinates }).item<double>();
    }
    std::vector<std::string> Node::graph(const std::string& className)
    {
        auto output = std::vector<std::string>();
        auto suffix = name == className ? ", fontcolor=red, fillcolor=lightblue, style=filled " : "";
        output.push_back("\"" + name + "\" [shape=circle" + suffix + "] \n");
        transform(children.begin(), children.end(), back_inserter(output), [this](const auto& child) { return "\"" + name + "\" -> \"" + child->getName() + "\""; });
        return output;
    }
}