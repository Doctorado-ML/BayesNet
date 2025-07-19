// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <thread>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "Network.h"
#include "bayesnet/utils/bayesnetUtils.h"
#include "bayesnet/utils/CountingSemaphore.h"
#include <pthread.h>
#include <fstream>
namespace bayesnet {
    Network::Network() : fitted{ false }, classNumStates{ 0 }
    {
    }
    Network::Network(const Network& other) 
        : features(other.features), className(other.className), classNumStates(other.classNumStates),
          fitted(other.fitted)
    {
        // Deep copy the samples tensor
        if (other.samples.defined()) {
            samples = other.samples.clone();
        }
        
        // First, create all nodes (without relationships)
        for (const auto& node : other.nodes) {
            nodes[node.first] = std::make_unique<Node>(*node.second);
        }
        
        // Second, reconstruct the relationships between nodes
        for (const auto& node : other.nodes) {
            const std::string& nodeName = node.first;
            Node* originalNode = node.second.get();
            Node* newNode = nodes[nodeName].get();
            
            // Reconstruct parent relationships
            for (Node* parent : originalNode->getParents()) {
                const std::string& parentName = parent->getName();
                if (nodes.find(parentName) != nodes.end()) {
                    newNode->addParent(nodes[parentName].get());
                }
            }
            
            // Reconstruct child relationships
            for (Node* child : originalNode->getChildren()) {
                const std::string& childName = child->getName();
                if (nodes.find(childName) != nodes.end()) {
                    newNode->addChild(nodes[childName].get());
                }
            }
        }
    }
    
    Network& Network::operator=(const Network& other)
    {
        if (this != &other) {
            // Clear existing state
            nodes.clear();
            features = other.features;
            className = other.className;
            classNumStates = other.classNumStates;
            fitted = other.fitted;
            
            // Deep copy the samples tensor
            if (other.samples.defined()) {
                samples = other.samples.clone();
            } else {
                samples = torch::Tensor();
            }
            
            // First, create all nodes (without relationships)
            for (const auto& node : other.nodes) {
                nodes[node.first] = std::make_unique<Node>(*node.second);
            }
            
            // Second, reconstruct the relationships between nodes
            for (const auto& node : other.nodes) {
                const std::string& nodeName = node.first;
                Node* originalNode = node.second.get();
                Node* newNode = nodes[nodeName].get();
                
                // Reconstruct parent relationships
                for (Node* parent : originalNode->getParents()) {
                    const std::string& parentName = parent->getName();
                    if (nodes.find(parentName) != nodes.end()) {
                        newNode->addParent(nodes[parentName].get());
                    }
                }
                
                // Reconstruct child relationships
                for (Node* child : originalNode->getChildren()) {
                    const std::string& childName = child->getName();
                    if (nodes.find(childName) != nodes.end()) {
                        newNode->addChild(nodes[childName].get());
                    }
                }
            }
        }
        return *this;
    }
    void Network::initialize()
    {
        features.clear();
        className = "";
        classNumStates = 0;
        fitted = false;
        nodes.clear();
        samples = torch::Tensor();
    }
    torch::Tensor& Network::getSamples()
    {
        return samples;
    }
    void Network::addNode(const std::string& name)
    {
        if (fitted) {
            throw std::invalid_argument("Cannot add node to a fitted network. Initialize first.");
        }
        if (name == "") {
            throw std::invalid_argument("Node name cannot be empty");
        }
        if (nodes.find(name) != nodes.end()) {
            return;
        }
        if (find(features.begin(), features.end(), name) == features.end()) {
            features.push_back(name);
        }
        nodes[name] = std::make_unique<Node>(name);
    }
    std::vector<std::string> Network::getFeatures() const
    {
        return features;
    }
    int Network::getClassNumStates() const
    {
        return classNumStates;
    }
    int Network::getStates() const
    {
        int result = 0;
        for (auto& node : nodes) {
            result += node.second->getNumStates();
        }
        return result;
    }
    std::string Network::getClassName() const
    {
        return className;
    }
    bool Network::isCyclic(const std::string& nodeId, std::unordered_set<std::string>& visited, std::unordered_set<std::string>& recStack)
    {
        if (visited.find(nodeId) == visited.end()) // if node hasn't been visited yet
        {
            visited.insert(nodeId);
            recStack.insert(nodeId);
            for (Node* child : nodes[nodeId]->getChildren()) {
                if (visited.find(child->getName()) == visited.end() && isCyclic(child->getName(), visited, recStack))
                    return true;
                if (recStack.find(child->getName()) != recStack.end())
                    return true;
            }
        }
        recStack.erase(nodeId); // remove node from recursion stack before function ends
        return false;
    }
    void Network::addEdge(const std::string& parent, const std::string& child)
    {
        if (fitted) {
            throw std::invalid_argument("Cannot add edge to a fitted network. Initialize first.");
        }
        if (nodes.find(parent) == nodes.end()) {
            throw std::invalid_argument("Parent node " + parent + " does not exist");
        }
        if (nodes.find(child) == nodes.end()) {
            throw std::invalid_argument("Child node " + child + " does not exist");
        }
        // Check if the edge is already in the graph
        for (auto& node : nodes[parent]->getChildren()) {
            if (node->getName() == child) {
                throw std::invalid_argument("Edge " + parent + " -> " + child + " already exists");
            }
        }
        // Temporarily add edge to check for cycles
        nodes[parent]->addChild(nodes[child].get());
        nodes[child]->addParent(nodes[parent].get());
        std::unordered_set<std::string> visited;
        std::unordered_set<std::string> recStack;
        if (isCyclic(nodes[child]->getName(), visited, recStack)) // if adding this edge forms a cycle
        {
            // remove problematic edge
            nodes[parent]->removeChild(nodes[child].get());
            nodes[child]->removeParent(nodes[parent].get());
            throw std::invalid_argument("Adding this edge forms a cycle in the graph.");
        }
    }
    std::map<std::string, std::unique_ptr<Node>>& Network::getNodes()
    {
        return nodes;
    }
    void Network::checkFitData(int n_samples, int n_features, int n_samples_y, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights)
    {
        if (weights.size(0) != n_samples) {
            throw std::invalid_argument("Weights (" + std::to_string(weights.size(0)) + ") must have the same number of elements as samples (" + std::to_string(n_samples) + ") in Network::fit");
        }
        if (n_samples != n_samples_y) {
            throw std::invalid_argument("X and y must have the same number of samples in Network::fit (" + std::to_string(n_samples) + " != " + std::to_string(n_samples_y) + ")");
        }
        if (n_features != featureNames.size()) {
            throw std::invalid_argument("X and features must have the same number of features in Network::fit (" + std::to_string(n_features) + " != " + std::to_string(featureNames.size()) + ")");
        }
        if (features.size() == 0) {
            throw std::invalid_argument("The network has not been initialized. You must call addNode() before calling fit()");
        }
        if (n_features != features.size() - 1) {
            throw std::invalid_argument("X and local features must have the same number of features in Network::fit (" + std::to_string(n_features) + " != " + std::to_string(features.size() - 1) + ")");
        }
        if (find(features.begin(), features.end(), className) == features.end()) {
            throw std::invalid_argument("Class Name not found in Network::features");
        }
        for (auto& feature : featureNames) {
            if (find(features.begin(), features.end(), feature) == features.end()) {
                throw std::invalid_argument("Feature " + feature + " not found in Network::features");
            }
            if (states.find(feature) == states.end()) {
                throw std::invalid_argument("Feature " + feature + " not found in states");
            }
        }
    }
    void Network::setStates(const std::map<std::string, std::vector<int>>& states)
    {
        // Set states to every Node in the network
        for_each(features.begin(), features.end(), [this, &states](const std::string& feature) {
            nodes.at(feature)->setNumStates(states.at(feature).size());
            });
        classNumStates = nodes.at(className)->getNumStates();
    }
    // X comes in nxm, where n is the number of features and m the number of samples
    void Network::fit(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& weights, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing)
    {
        checkFitData(X.size(1), X.size(0), y.size(0), featureNames, className, states, weights);
        this->className = className;
        torch::Tensor ytmp = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
        samples = torch::cat({ X , ytmp }, 0);
        for (int i = 0; i < featureNames.size(); ++i) {
            auto row_feature = X.index({ i, "..." });
        }
        completeFit(states, weights, smoothing);
    }
    void Network::fit(const torch::Tensor& samples, const torch::Tensor& weights, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing)
    {
        checkFitData(samples.size(1), samples.size(0) - 1, samples.size(1), featureNames, className, states, weights);
        this->className = className;
        this->samples = samples;
        completeFit(states, weights, smoothing);
    }
    // input_data comes in nxm, where n is the number of features and m the number of samples
    void Network::fit(const std::vector<std::vector<int>>& input_data, const std::vector<int>& labels, const std::vector<double>& weights_, const std::vector<std::string>& featureNames, const std::string& className, const std::map<std::string, std::vector<int>>& states, const Smoothing_t smoothing)
    {
        const torch::Tensor weights = torch::tensor(weights_, torch::kFloat64);
        checkFitData(input_data[0].size(), input_data.size(), labels.size(), featureNames, className, states, weights);
        this->className = className;
        // Build tensor of samples (nxm) (n+1 because of the class)
        samples = torch::zeros({ static_cast<int>(input_data.size() + 1), static_cast<int>(input_data[0].size()) }, torch::kInt32);
        for (int i = 0; i < featureNames.size(); ++i) {
            samples.index_put_({ i, "..." }, torch::tensor(input_data[i], torch::kInt32));
        }
        samples.index_put_({ -1, "..." }, torch::tensor(labels, torch::kInt32));
        completeFit(states, weights, smoothing);
    }
    void Network::completeFit(const std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const Smoothing_t smoothing)
    {
        setStates(states);
        std::vector<std::thread> threads;
        auto& semaphore = CountingSemaphore::getInstance();
        const double n_samples = static_cast<double>(samples.size(1));
        auto worker = [&](std::pair<const std::string, std::unique_ptr<Node>>& node, int i) {
            std::string threadName = "FitWorker-" + std::to_string(i);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif
            double numStates = static_cast<double>(node.second->getNumStates());
            double smoothing_factor;
            switch (smoothing) {
                case Smoothing_t::ORIGINAL:
                    smoothing_factor = 1.0 / n_samples;
                    break;
                case Smoothing_t::LAPLACE:
                    smoothing_factor = 1.0;
                    break;
                case Smoothing_t::CESTNIK:
                    smoothing_factor = 1 / numStates;
                    break;
                default:
                    smoothing_factor = 0.0; // No smoothing 
            }
            node.second->computeCPT(samples, features, smoothing_factor, weights);
            semaphore.release();
            };
        int i = 0;
        for (auto& node : nodes) {
            semaphore.acquire();
            threads.emplace_back(worker, std::ref(node), i++);
        }
        for (auto& thread : threads) {
            thread.join();
        }
        fitted = true;
    }
    torch::Tensor Network::predict_tensor(const torch::Tensor& samples, const bool proba)
    {
        if (!fitted) {
            throw std::logic_error("You must call fit() before calling predict()");
        }
        // Ensure the sample size is equal to the number of features
        if (samples.size(0) != features.size() - 1) {
            throw std::invalid_argument("(T) Sample size (" + std::to_string(samples.size(0)) +
                ") does not match the number of features (" + std::to_string(features.size() - 1) + ")");
        }
        torch::Tensor result;
        std::vector<std::thread> threads;
        std::mutex mtx;
        auto& semaphore = CountingSemaphore::getInstance();
        result = torch::zeros({ samples.size(1), classNumStates }, torch::kFloat64);
        auto worker = [&](const torch::Tensor& sample, int i) {
            std::string threadName = "PredictWorker-" + std::to_string(i);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif
            auto psample = predict_sample(sample);
            auto temp = torch::tensor(psample, torch::kFloat64);
            {
                std::lock_guard<std::mutex> lock(mtx);
                result.index_put_({ i, "..." }, temp);
            }
            semaphore.release();
            };
        for (int i = 0; i < samples.size(1); ++i) {
            semaphore.acquire();
            const torch::Tensor sample = samples.index({ "...", i });
            threads.emplace_back(worker, sample, i);
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (proba)
            return result;
        return result.argmax(1);
    }
    // Return mxn tensor of probabilities
    torch::Tensor Network::predict_proba(const torch::Tensor& samples)
    {
        return predict_tensor(samples, true);
    }

    // Return mxn tensor of probabilities
    torch::Tensor Network::predict(const torch::Tensor& samples)
    {
        return predict_tensor(samples, false);
    }

    // Return mx1 std::vector of predictions
    // tsamples is nxm std::vector of samples
    std::vector<int> Network::predict(const std::vector<std::vector<int>>& tsamples)
    {
        if (!fitted) {
            throw std::logic_error("You must call fit() before calling predict()");
        }
        // Ensure the sample size is equal to the number of features
        if (tsamples.size() != features.size() - 1) {
            throw std::invalid_argument("(V) Sample size (" + std::to_string(tsamples.size()) +
                ") does not match the number of features (" + std::to_string(features.size() - 1) + ")");
        }
        std::vector<int> predictions(tsamples[0].size(), 0);
        std::vector<int> sample;
        std::vector<std::thread> threads;
        auto& semaphore = CountingSemaphore::getInstance();
        auto worker = [&](const std::vector<int>& sample, const int row, int& prediction) {
            std::string threadName = "(V)PWorker-" + std::to_string(row);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif
            auto classProbabilities = predict_sample(sample);
            auto maxElem = max_element(classProbabilities.begin(), classProbabilities.end());
            int predictedClass = distance(classProbabilities.begin(), maxElem);
            prediction = predictedClass;
            semaphore.release();
            };
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            semaphore.acquire();
            threads.emplace_back(worker, sample, row, std::ref(predictions[row]));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return predictions;
    }
    // Return mxn std::vector of probabilities
    // tsamples is nxm std::vector of samples
    std::vector<std::vector<double>> Network::predict_proba(const std::vector<std::vector<int>>& tsamples)
    {
        if (!fitted) {
            throw std::logic_error("You must call fit() before calling predict_proba()");
        }
        // Ensure the sample size is equal to the number of features
        if (tsamples.size() != features.size() - 1) {
            throw std::invalid_argument("(V) Sample size (" + std::to_string(tsamples.size()) +
                ") does not match the number of features (" + std::to_string(features.size() - 1) + ")");
        }
        std::vector<std::vector<double>> predictions(tsamples[0].size(), std::vector<double>(classNumStates, 0.0));
        std::vector<int> sample;
        std::vector<std::thread> threads;
        auto& semaphore = CountingSemaphore::getInstance();
        auto worker = [&](const std::vector<int>& sample, int row, std::vector<double>& predictions) {
            std::string threadName = "(V)PWorker-" + std::to_string(row);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif
            std::vector<double> classProbabilities = predict_sample(sample);
            predictions = classProbabilities;
            semaphore.release();
            };
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            semaphore.acquire();
            threads.emplace_back(worker, sample, row, std::ref(predictions[row]));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return predictions;
    }
    double Network::score(const std::vector<std::vector<int>>& tsamples, const std::vector<int>& labels)
    {
        std::vector<int> y_pred = predict(tsamples);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == labels[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();
    }
    // Return 1xn std::vector of probabilities
    std::vector<double> Network::predict_sample(const std::vector<int>& sample)
    {
        std::map<std::string, int> evidence;
        for (int i = 0; i < sample.size(); ++i) {
            evidence[features[i]] = sample[i];
        }
        return exactInference(evidence);
    }
    // Return 1xn std::vector of probabilities
    std::vector<double> Network::predict_sample(const torch::Tensor& sample)
    {
        std::map<std::string, int> evidence;
        for (int i = 0; i < sample.size(0); ++i) {
            evidence[features[i]] = sample[i].item<int>();
        }
        return exactInference(evidence);
    }
    std::vector<double> Network::exactInference(std::map<std::string, int>& evidence)
    {
        std::vector<double> result(classNumStates, 0.0);
        auto completeEvidence = std::map<std::string, int>(evidence);
        for (int i = 0; i < classNumStates; ++i) {
            completeEvidence[getClassName()] = i;
            double partial = 1.0;
            for (auto& node : getNodes()) {
                partial *= node.second->getFactorValue(completeEvidence);
            }
            result[i] = partial;
        }
        // Normalize result
        double sum = std::accumulate(result.begin(), result.end(), 0.0);
        transform(result.begin(), result.end(), result.begin(), [sum](const double& value) { return value / sum; });
        return result;
    }
    std::vector<std::string> Network::show() const
    {
        std::vector<std::string> result;
        // Draw the network
        for (auto& node : nodes) {
            std::string line = node.first + " -> ";
            for (auto child : node.second->getChildren()) {
                line += child->getName() + ", ";
            }
            result.push_back(line);
        }
        return result;
    }
    std::vector<std::string> Network::graph(const std::string& title) const
    {
        auto output = std::vector<std::string>();
        auto prefix = "digraph BayesNet {\nlabel=<BayesNet ";
        auto suffix = ">\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n";
        std::string header = prefix + title + suffix;
        output.push_back(header);
        for (auto& node : nodes) {
            auto result = node.second->graph(className);
            output.insert(output.end(), result.begin(), result.end());
        }
        output.push_back("}\n");
        return output;
    }
    std::vector<std::pair<std::string, std::string>> Network::getEdges() const
    {
        auto edges = std::vector<std::pair<std::string, std::string>>();
        for (const auto& node : nodes) {
            auto head = node.first;
            for (const auto& child : node.second->getChildren()) {
                auto tail = child->getName();
                edges.push_back({ head, tail });
            }
        }
        return edges;
    }
    int Network::getNumEdges() const
    {
        return getEdges().size();
    }
    std::vector<std::string> Network::topological_sort()
    {
        /* Check if al the fathers of every node are before the node */
        auto result = features;
        result.erase(remove(result.begin(), result.end(), className), result.end());
        bool ending{ false };
        while (!ending) {
            ending = true;
            for (auto feature : features) {
                auto fathers = nodes[feature]->getParents();
                for (const auto& father : fathers) {
                    auto fatherName = father->getName();
                    if (fatherName == className) {
                        continue;
                    }
                    // Check if father is placed before the actual feature
                    auto it = find(result.begin(), result.end(), fatherName);
                    if (it != result.end()) {
                        auto it2 = find(result.begin(), result.end(), feature);
                        if (it2 != result.end()) {
                            if (distance(it, it2) < 0) {
                                // if it is not, insert it before the feature
                                result.erase(remove(result.begin(), result.end(), fatherName), result.end());
                                result.insert(it2, fatherName);
                                ending = false;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
    std::string Network::dump_cpt() const
    {
        std::stringstream oss;
        for (auto& node : nodes) {
            oss << "* " << node.first << ": (" << node.second->getNumStates() << ") : " << node.second->getCPT().sizes() << std::endl;
            oss << node.second->getCPT() << std::endl;
        }
        return oss.str();
    }
    
    bool Network::operator==(const Network& other) const
    {
        // Compare number of nodes
        if (nodes.size() != other.nodes.size()) {
            return false;
        }
        
        // Compare if all node names exist in both networks
        for (const auto& node : nodes) {
            if (other.nodes.find(node.first) == other.nodes.end()) {
                return false;
            }
        }
        
        // Compare edges (topology)
        auto thisEdges = getEdges();
        auto otherEdges = other.getEdges();
        
        // Compare number of edges
        if (thisEdges.size() != otherEdges.size()) {
            return false;
        }
        
        // Sort both edge lists for comparison
        std::sort(thisEdges.begin(), thisEdges.end());
        std::sort(otherEdges.begin(), otherEdges.end());
        
        // Compare each edge
        for (size_t i = 0; i < thisEdges.size(); ++i) {
            if (thisEdges[i] != otherEdges[i]) {
                return false;
            }
        }
        
        return true;
    }
}
