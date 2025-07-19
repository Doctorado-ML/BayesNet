// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include "TestUtils.h"
#include "bayesnet/network/Network.h"
#include "bayesnet/network/Node.h"
#include "bayesnet/utils/bayesnetUtils.h"

const double threshold = 1e-4;
void buildModel(bayesnet::Network& net, const std::vector<std::string>& features, const std::string& className)
{
    std::vector<pair<int, int>> network = { {0, 1}, {0, 2}, {1, 3} };
    for (const auto& feature : features) {
        net.addNode(feature);
    }
    net.addNode(className);
    for (const auto& edge : network) {
        net.addEdge(features.at(edge.first), features.at(edge.second));
    }
    for (const auto& feature : features) {
        net.addEdge(className, feature);
    }
}
TEST_CASE("Test Bayesian Network", "[Network]")
{

    auto raw = RawDatasets("iris", true);
    auto net = bayesnet::Network();

    SECTION("Test get features")
    {
        net.addNode("A");
        net.addNode("B");
        REQUIRE(net.getFeatures() == std::vector<std::string>{"A", "B"});
        net.addNode("C");
        REQUIRE(net.getFeatures() == std::vector<std::string>{"A", "B", "C"});
    }
    SECTION("Test get edges")
    {
        net.addNode("A");
        net.addNode("B");
        net.addNode("C");
        net.addEdge("A", "B");
        net.addEdge("B", "C");
        REQUIRE(net.getEdges() == std::vector<pair<std::string, std::string>>{ {"A", "B"}, { "B", "C" } });
        REQUIRE(net.getNumEdges() == 2);
        net.addEdge("A", "C");
        REQUIRE(net.getEdges() == std::vector<pair<std::string, std::string>>{ {"A", "B"}, { "A", "C" }, { "B", "C" } });
        REQUIRE(net.getNumEdges() == 3);
    }
    SECTION("Test getNodes")
    {
        net.addNode("A");
        net.addNode("B");
        auto& nodes = net.getNodes();
        REQUIRE(nodes.count("A") == 1);
        REQUIRE(nodes.count("B") == 1);
    }

    SECTION("Test fit Network")
    {
        auto net2 = bayesnet::Network();
        auto net3 = bayesnet::Network();
        net3.initialize();
        net2.initialize();
        net.initialize();
        buildModel(net, raw.features, raw.className);
        buildModel(net2, raw.features, raw.className);
        buildModel(net3, raw.features, raw.className);
        std::vector<pair<std::string, std::string>> edges = {
            {"class", "sepallength"}, {"class", "sepalwidth"}, {"class", "petallength"},
            {"class", "petalwidth" }, {"sepallength", "sepalwidth"}, {"sepallength", "petallength"},
            {"sepalwidth", "petalwidth"}
        };
        REQUIRE(net.getEdges() == edges);
        REQUIRE(net2.getEdges() == edges);
        REQUIRE(net3.getEdges() == edges);
        std::vector<std::string> features = { "sepallength", "sepalwidth", "petallength", "petalwidth", "class" };
        REQUIRE(net.getFeatures() == features);
        REQUIRE(net2.getFeatures() == features);
        REQUIRE(net3.getFeatures() == features);
        auto& nodes = net.getNodes();
        auto& nodes2 = net2.getNodes();
        auto& nodes3 = net3.getNodes();
        // Check Nodes parents & children
        for (const auto& feature : features) {
            // Parents
            std::vector<std::string> parents, parents2, parents3, children, children2, children3;
            auto nodeParents = nodes[feature]->getParents();
            auto nodeParents2 = nodes2[feature]->getParents();
            auto nodeParents3 = nodes3[feature]->getParents();
            transform(nodeParents.begin(), nodeParents.end(), back_inserter(parents), [](const auto& p) { return p->getName(); });
            transform(nodeParents2.begin(), nodeParents2.end(), back_inserter(parents2), [](const auto& p) { return p->getName(); });
            transform(nodeParents3.begin(), nodeParents3.end(), back_inserter(parents3), [](const auto& p) { return p->getName(); });
            REQUIRE(parents == parents2);
            REQUIRE(parents == parents3);
            // Children
            auto nodeChildren = nodes[feature]->getChildren();
            auto nodeChildren2 = nodes2[feature]->getChildren();
            auto nodeChildren3 = nodes2[feature]->getChildren();
            transform(nodeChildren.begin(), nodeChildren.end(), back_inserter(children), [](const auto& p) { return p->getName(); });
            transform(nodeChildren2.begin(), nodeChildren2.end(), back_inserter(children2), [](const auto& p) { return p->getName(); });
            transform(nodeChildren3.begin(), nodeChildren3.end(), back_inserter(children3), [](const auto& p) { return p->getName(); });
            REQUIRE(children == children2);
            REQUIRE(children == children3);
        }
        // Fit networks
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        net2.fit(raw.dataset, raw.weights, raw.features, raw.className, raw.states, raw.smoothing);
        net3.fit(raw.Xt, raw.yt, raw.weights, raw.features, raw.className, raw.states, raw.smoothing);
        REQUIRE(net.getStates() == net2.getStates());
        REQUIRE(net.getStates() == net3.getStates());
        REQUIRE(net.getFeatures() == net2.getFeatures());
        REQUIRE(net.getFeatures() == net3.getFeatures());
        REQUIRE(net.getClassName() == net2.getClassName());
        REQUIRE(net.getClassName() == net3.getClassName());
        REQUIRE(net.getNodes().size() == net2.getNodes().size());
        REQUIRE(net.getNodes().size() == net3.getNodes().size());
        REQUIRE(net.getEdges() == net2.getEdges());
        REQUIRE(net.getEdges() == net3.getEdges());
        REQUIRE(net.getNumEdges() == net2.getNumEdges());
        REQUIRE(net.getNumEdges() == net3.getNumEdges());
        REQUIRE(net.getClassNumStates() == net2.getClassNumStates());
        REQUIRE(net.getClassNumStates() == net3.getClassNumStates());
        REQUIRE(net.getSamples().size(0) == net2.getSamples().size(0));
        REQUIRE(net.getSamples().size(0) == net3.getSamples().size(0));
        REQUIRE(net.getSamples().size(1) == net2.getSamples().size(1));
        REQUIRE(net.getSamples().size(1) == net3.getSamples().size(1));
        // Check Conditional Probabilities tables
        for (int i = 0; i < features.size(); ++i) {
            auto feature = features.at(i);
            for (const auto& feature : features) {
                auto cpt = nodes[feature]->getCPT();
                auto cpt2 = nodes2[feature]->getCPT();
                auto cpt3 = nodes3[feature]->getCPT();
                REQUIRE(cpt.equal(cpt2));
                REQUIRE(cpt.equal(cpt3));
            }
        }
    }
    SECTION("Test show")
    {
        INFO("Test show");
        net.addNode("A");
        net.addNode("B");
        net.addNode("C");
        net.addEdge("A", "B");
        net.addEdge("A", "C");
        auto str = net.show();
        REQUIRE(str.size() == 3);
        REQUIRE(str[0] == "A -> B, C, ");
        REQUIRE(str[1] == "B -> ");
        REQUIRE(str[2] == "C -> ");
    }
    SECTION("Test topological_sort")
    {
        INFO("Test topological sort");
        net.addNode("A");
        net.addNode("B");
        net.addNode("C");
        net.addEdge("A", "B");
        net.addEdge("A", "C");
        auto sorted = net.topological_sort();
        REQUIRE(sorted.size() == 3);
        REQUIRE(sorted[0] == "A");
        bool result = sorted[1] == "B" && sorted[2] == "C";
        REQUIRE(result);
    }
    SECTION("Test graph")
    {
        INFO("Test graph");
        net.addNode("A");
        net.addNode("B");
        net.addNode("C");
        net.addEdge("A", "B");
        net.addEdge("A", "C");
        auto str = net.graph("Test Graph");
        REQUIRE(str.size() == 7);
        REQUIRE(str[0] == "digraph BayesNet {\nlabel=<BayesNet Test Graph>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n");
        REQUIRE(str[1] == "\"A\" [shape=circle] \n");
        REQUIRE(str[2] == "\"A\" -> \"B\"");
        REQUIRE(str[3] == "\"A\" -> \"C\"");
        REQUIRE(str[4] == "\"B\" [shape=circle] \n");
        REQUIRE(str[5] == "\"C\" [shape=circle] \n");
        REQUIRE(str[6] == "}\n");
    }
    SECTION("Test predict")
    {
        INFO("Test predict");
        buildModel(net, raw.features, raw.className);
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        std::vector<std::vector<int>> test = { {1, 2, 0, 1, 1}, {0, 1, 2, 0, 1}, {0, 0, 0, 0, 1}, {2, 2, 2, 2, 1} };
        std::vector<int> y_test = { 2, 2, 0, 2, 1 };
        auto y_pred = net.predict(test);
        REQUIRE(y_pred == y_test);
    }
    SECTION("Test predict_proba")
    {
        INFO("Test predict_proba");
        buildModel(net, raw.features, raw.className);
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        std::vector<std::vector<int>> test = { {1, 2, 0, 1, 1}, {0, 1, 2, 0, 1}, {0, 0, 0, 0, 1}, {2, 2, 2, 2, 1} };
        std::vector<std::vector<double>> y_test = {
            {0.450237, 0.0866621, 0.463101},
            {0.244443, 0.0925922, 0.662964},
            {0.913441, 0.0125857, 0.0739732},
            {0.450237, 0.0866621, 0.463101},
            {0.0135226, 0.971726, 0.0147519}
        };
        auto y_pred = net.predict_proba(test);
        REQUIRE(y_pred.size() == 5);
        REQUIRE(y_pred[0].size() == 3);
        for (int i = 0; i < y_pred.size(); ++i) {
            for (int j = 0; j < y_pred[i].size(); ++j) {
                REQUIRE(y_pred[i][j] == Catch::Approx(y_test[i][j]).margin(threshold));
            }
        }
    }
    SECTION("Test score")
    {
        INFO("Test score");
        buildModel(net, raw.features, raw.className);
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        auto score = net.score(raw.Xv, raw.yv);
        REQUIRE(score == Catch::Approx(0.97333333).margin(threshold));
    }
    SECTION("Copy constructor")
    {
        INFO("Test copy constructor");
        buildModel(net, raw.features, raw.className);
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        auto net2 = bayesnet::Network(net);
        REQUIRE(net.getFeatures() == net2.getFeatures());
        REQUIRE(net.getEdges() == net2.getEdges());
        REQUIRE(net.getNumEdges() == net2.getNumEdges());
        REQUIRE(net.getStates() == net2.getStates());
        REQUIRE(net.getClassName() == net2.getClassName());
        REQUIRE(net.getClassNumStates() == net2.getClassNumStates());
        REQUIRE(net.getSamples().size(0) == net2.getSamples().size(0));
        REQUIRE(net.getSamples().size(1) == net2.getSamples().size(1));
        REQUIRE(net.getNodes().size() == net2.getNodes().size());
        for (const auto& feature : net.getFeatures()) {
            auto& node = net.getNodes().at(feature);
            auto& node2 = net2.getNodes().at(feature);
            REQUIRE(node->getName() == node2->getName());
            REQUIRE(node->getChildren().size() == node2->getChildren().size());
            REQUIRE(node->getParents().size() == node2->getParents().size());
            REQUIRE(node->getCPT().equal(node2->getCPT()));
        }
    }
    SECTION("Network oddities")
    {
        INFO("Network oddities");
        buildModel(net, raw.features, raw.className);
        // predict without fitting
        std::vector<std::vector<int>> test = { {1, 2, 0, 1, 1}, {0, 1, 2, 0, 1}, {0, 0, 0, 0, 1}, {2, 2, 2, 2, 1} };
        auto test_tensor = bayesnet::vectorToTensor(test);
        REQUIRE_THROWS_AS(net.predict(test), std::logic_error);
        REQUIRE_THROWS_WITH(net.predict(test), "You must call fit() before calling predict()");
        REQUIRE_THROWS_AS(net.predict(test_tensor), std::logic_error);
        REQUIRE_THROWS_WITH(net.predict(test_tensor), "You must call fit() before calling predict()");
        REQUIRE_THROWS_AS(net.predict_proba(test), std::logic_error);
        REQUIRE_THROWS_WITH(net.predict_proba(test), "You must call fit() before calling predict_proba()");
        REQUIRE_THROWS_AS(net.score(raw.Xv, raw.yv), std::logic_error);
        REQUIRE_THROWS_WITH(net.score(raw.Xv, raw.yv), "You must call fit() before calling predict()");
        // predict with wrong data
        auto netx = bayesnet::Network();
        buildModel(netx, raw.features, raw.className);
        netx.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        std::vector<std::vector<int>> test2 = { {1, 2, 0, 1, 1}, {0, 1, 2, 0, 1}, {0, 0, 0, 0, 1} };
        auto test_tensor2 = bayesnet::vectorToTensor(test2, false);
        REQUIRE_THROWS_AS(netx.predict(test2), std::invalid_argument);
        REQUIRE_THROWS_WITH(netx.predict(test2), "(V) Sample size (3) does not match the number of features (4)");
        REQUIRE_THROWS_AS(netx.predict(test_tensor2), std::invalid_argument);
        REQUIRE_THROWS_WITH(netx.predict(test_tensor2), "(T) Sample size (3) does not match the number of features (4)");
        // fit with wrong data
        // Weights
        auto net2 = bayesnet::Network();
        REQUIRE_THROWS_AS(net2.fit(raw.Xv, raw.yv, std::vector<double>(), raw.features, raw.className, raw.states, raw.smoothing), std::invalid_argument);
        std::string invalid_weights = "Weights (0) must have the same number of elements as samples (150) in Network::fit";
        REQUIRE_THROWS_WITH(net2.fit(raw.Xv, raw.yv, std::vector<double>(), raw.features, raw.className, raw.states, raw.smoothing), invalid_weights);
        // X & y
        std::string invalid_labels = "X and y must have the same number of samples in Network::fit (150 != 0)";
        REQUIRE_THROWS_AS(net2.fit(raw.Xv, std::vector<int>(), raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net2.fit(raw.Xv, std::vector<int>(), raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing), invalid_labels);
        // Features
        std::string invalid_features = "X and features must have the same number of features in Network::fit (4 != 0)";
        REQUIRE_THROWS_AS(net2.fit(raw.Xv, raw.yv, raw.weightsv, std::vector<std::string>(), raw.className, raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net2.fit(raw.Xv, raw.yv, raw.weightsv, std::vector<std::string>(), raw.className, raw.states, raw.smoothing), invalid_features);
        // Different number of features
        auto net3 = bayesnet::Network();
        auto test2y = { 1, 2, 3, 4, 5 };
        buildModel(net3, raw.features, raw.className);
        auto features3 = raw.features;
        features3.pop_back();
        std::string invalid_features2 = "X and local features must have the same number of features in Network::fit (3 != 4)";
        REQUIRE_THROWS_AS(net3.fit(test2, test2y, std::vector<double>(5, 0), features3, raw.className, raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net3.fit(test2, test2y, std::vector<double>(5, 0), features3, raw.className, raw.states, raw.smoothing), invalid_features2);
        // Uninitialized network
        std::string network_invalid = "The network has not been initialized. You must call addNode() before calling fit()";
        REQUIRE_THROWS_AS(net2.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, "duck", raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net2.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, "duck", raw.states, raw.smoothing), network_invalid);
        // Classname
        std::string invalid_classname = "Class Name not found in Network::features";
        REQUIRE_THROWS_AS(net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, "duck", raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, "duck", raw.states, raw.smoothing), invalid_classname);
        // Invalid feature
        auto features2 = raw.features;
        features2.pop_back();
        features2.push_back("duck");
        std::string invalid_feature = "Feature duck not found in Network::features";
        REQUIRE_THROWS_AS(net.fit(raw.Xv, raw.yv, raw.weightsv, features2, raw.className, raw.states, raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net.fit(raw.Xv, raw.yv, raw.weightsv, features2, raw.className, raw.states, raw.smoothing), invalid_feature);
        // Add twice the same node name to the network => Nothing should happen
        net.addNode("A");
        net.addNode("A");
        // invalid state in checkfit
        auto net4 = bayesnet::Network();
        buildModel(net4, raw.features, raw.className);
        std::string invalid_state = "Feature sepallength not found in states";
        REQUIRE_THROWS_AS(net4.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, std::map<std::string, std::vector<int>>(), raw.smoothing), std::invalid_argument);
        REQUIRE_THROWS_WITH(net4.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, std::map<std::string, std::vector<int>>(), raw.smoothing), invalid_state);
        // Try to add node or edge to a fitted network
        auto net5 = bayesnet::Network();
        buildModel(net5, raw.features, raw.className);
        net5.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
        REQUIRE_THROWS_AS(net5.addNode("A"), std::logic_error);
        REQUIRE_THROWS_WITH(net5.addNode("A"), "Cannot add node to a fitted network. Initialize first.");
        REQUIRE_THROWS_AS(net5.addEdge("A", "B"), std::logic_error);
        REQUIRE_THROWS_WITH(net5.addEdge("A", "B"), "Cannot add edge to a fitted network. Initialize first.");
    }
    SECTION("Test assignment operator")
    {
        INFO("Test assignment operator");
        // Create original network
        auto net1 = bayesnet::Network();
        buildModel(net1, raw.features, raw.className);
        net1.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);

        // Create empty network and assign
        auto net2 = bayesnet::Network();
        net2.addNode("TempNode"); // Add something to make sure it gets cleared
        net2 = net1;

        // Verify they are equal
        REQUIRE(net1.getFeatures() == net2.getFeatures());
        REQUIRE(net1.getEdges() == net2.getEdges());
        REQUIRE(net1.getNumEdges() == net2.getNumEdges());
        REQUIRE(net1.getStates() == net2.getStates());
        REQUIRE(net1.getClassName() == net2.getClassName());
        REQUIRE(net1.getClassNumStates() == net2.getClassNumStates());
        REQUIRE(net1.getSamples().size(0) == net2.getSamples().size(0));
        REQUIRE(net1.getSamples().size(1) == net2.getSamples().size(1));
        REQUIRE(net1.getNodes().size() == net2.getNodes().size());

        // Verify topology equality
        REQUIRE(net1 == net2);

        // Verify they are separate objects by modifying one
        net2.initialize();
        net2.addNode("OnlyInNet2");
        REQUIRE(net1.getNodes().size() != net2.getNodes().size());
        REQUIRE_FALSE(net1 == net2);
    }
    SECTION("Test self assignment")
    {
        INFO("Test self assignment");
        buildModel(net, raw.features, raw.className);
        net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);

        int original_edges = net.getNumEdges();
        int original_nodes = net.getNodes().size();

        // Self assignment should not corrupt the network
        net = net;
        auto all_features = raw.features;
        all_features.push_back(raw.className);
        REQUIRE(net.getNumEdges() == original_edges);
        REQUIRE(net.getNodes().size() == original_nodes);
        REQUIRE(net.getFeatures() == all_features);
        REQUIRE(net.getClassName() == raw.className);
    }
    SECTION("Test operator== topology comparison")
    {
        INFO("Test operator== topology comparison");

        // Test 1: Two identical networks
        auto net1 = bayesnet::Network();
        auto net2 = bayesnet::Network();

        net1.addNode("A");
        net1.addNode("B");
        net1.addNode("C");
        net1.addEdge("A", "B");
        net1.addEdge("B", "C");

        net2.addNode("A");
        net2.addNode("B");
        net2.addNode("C");
        net2.addEdge("A", "B");
        net2.addEdge("B", "C");

        REQUIRE(net1 == net2);

        // Test 2: Different nodes
        auto net3 = bayesnet::Network();
        net3.addNode("A");
        net3.addNode("D"); // Different node
        REQUIRE_FALSE(net1 == net3);

        // Test 3: Same nodes, different edges
        auto net4 = bayesnet::Network();
        net4.addNode("A");
        net4.addNode("B");
        net4.addNode("C");
        net4.addEdge("A", "C"); // Different topology
        net4.addEdge("B", "C");
        REQUIRE_FALSE(net1 == net4);

        // Test 4: Empty networks
        auto net5 = bayesnet::Network();
        auto net6 = bayesnet::Network();
        REQUIRE(net5 == net6);

        // Test 5: Same topology, different edge order
        auto net7 = bayesnet::Network();
        net7.addNode("A");
        net7.addNode("B");
        net7.addNode("C");
        net7.addEdge("B", "C"); // Add edges in different order
        net7.addEdge("A", "B");
        REQUIRE(net1 == net7); // Should still be equal
    }
    SECTION("Test RAII compliance with smart pointers")
    {
        INFO("Test RAII compliance with smart pointers");

        std::unique_ptr<bayesnet::Network> net1 = std::make_unique<bayesnet::Network>();
        buildModel(*net1, raw.features, raw.className);
        net1->fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);

        // Test that copy constructor works with smart pointers
        std::unique_ptr<bayesnet::Network> net2 = std::make_unique<bayesnet::Network>(*net1);

        REQUIRE(*net1 == *net2);
        REQUIRE(net1->getNumEdges() == net2->getNumEdges());
        REQUIRE(net1->getNodes().size() == net2->getNodes().size());

        // Destroy original
        net1.reset();

        // Test predictions still work
        std::vector<std::vector<int>> test = { {1}, {2}, {0}, {1} };
        REQUIRE_NOTHROW(net2->predict(test));

        // net2 should still be valid and functional
        net2->initialize();
        REQUIRE_NOTHROW(net2->addNode("NewNode"));
        REQUIRE(net2->getNodes().count("NewNode") == 1);
    }
    SECTION("Test complex topology copy")
    {
        INFO("Test complex topology copy");

        auto original = bayesnet::Network();

        // Create a more complex network
        original.addNode("Root");
        original.addNode("Child1");
        original.addNode("Child2");
        original.addNode("Grandchild1");
        original.addNode("Grandchild2");
        original.addNode("Grandchild3");

        original.addEdge("Root", "Child1");
        original.addEdge("Root", "Child2");
        original.addEdge("Child1", "Grandchild1");
        original.addEdge("Child1", "Grandchild2");
        original.addEdge("Child2", "Grandchild3");

        // Copy it
        auto copy = original;

        // Verify topology is identical
        REQUIRE(original == copy);
        REQUIRE(original.getNodes().size() == copy.getNodes().size());
        REQUIRE(original.getNumEdges() == copy.getNumEdges());

        // Verify edges are properly reconstructed
        auto originalEdges = original.getEdges();
        auto copyEdges = copy.getEdges();
        REQUIRE(originalEdges.size() == copyEdges.size());

        // Verify node relationships are properly copied
        for (const auto& nodePair : original.getNodes()) {
            const std::string& nodeName = nodePair.first;
            auto* originalNode = nodePair.second.get();
            auto* copyNode = copy.getNodes().at(nodeName).get();

            REQUIRE(originalNode->getParents().size() == copyNode->getParents().size());
            REQUIRE(originalNode->getChildren().size() == copyNode->getChildren().size());

            // Verify parent names match
            for (size_t i = 0; i < originalNode->getParents().size(); ++i) {
                REQUIRE(originalNode->getParents()[i]->getName() ==
                    copyNode->getParents()[i]->getName());
            }

            // Verify child names match
            for (size_t i = 0; i < originalNode->getChildren().size(); ++i) {
                REQUIRE(originalNode->getChildren()[i]->getName() ==
                    copyNode->getChildren()[i]->getName());
            }
        }
    }

}
TEST_CASE("Test and empty Node", "[Network]")
{
    auto net = bayesnet::Network();
    REQUIRE_THROWS_AS(net.addNode(""), std::invalid_argument);
    REQUIRE_THROWS_WITH(net.addNode(""), "Node name cannot be empty");
}
TEST_CASE("Cicle in Network", "[Network]")
{
    auto net = bayesnet::Network();
    net.addNode("A");
    net.addNode("B");
    net.addNode("C");
    net.addEdge("A", "B");
    net.addEdge("B", "C");
    REQUIRE_THROWS_AS(net.addEdge("C", "A"), std::invalid_argument);
    REQUIRE_THROWS_WITH(net.addEdge("C", "A"), "Adding this edge forms a cycle in the graph.");
}
TEST_CASE("Edges troubles", "[Network]")
{
    auto net = bayesnet::Network();
    net.addNode("A");
    net.addNode("B");
    REQUIRE_THROWS_AS(net.addEdge("A", "C"), std::invalid_argument);
    REQUIRE_THROWS_WITH(net.addEdge("A", "C"), "Child node C does not exist");
    REQUIRE_THROWS_AS(net.addEdge("C", "A"), std::invalid_argument);
    REQUIRE_THROWS_WITH(net.addEdge("C", "A"), "Parent node C does not exist");
    net.addEdge("A", "B");
    REQUIRE_THROWS_AS(net.addEdge("A", "B"), std::invalid_argument);
    REQUIRE_THROWS_WITH(net.addEdge("A", "B"), "Edge A -> B already exists");
}
TEST_CASE("Dump CPT", "[Network]")
{
    auto net = bayesnet::Network();
    auto raw = RawDatasets("iris", true);
    buildModel(net, raw.features, raw.className);
    net.fit(raw.Xv, raw.yv, raw.weightsv, raw.features, raw.className, raw.states, raw.smoothing);
    auto res = net.dump_cpt();
    std::string expected = R"(* class: (3) : [3]
 0.3333
 0.3333
 0.3333
[ CPUDoubleType{3} ]
* petallength: (4) : [4, 3, 3]
(1,.,.) = 
  0.9388  0.1000  0.2000
  0.6250  0.0526  0.1667
  0.4000  0.0303  0.0196

(2,.,.) = 
  0.0204  0.7000  0.4000
  0.1250  0.8421  0.1667
  0.2000  0.7273  0.0196

(3,.,.) = 
  0.0204  0.1000  0.2000
  0.1250  0.0526  0.5000
  0.2000  0.1818  0.1373

(4,.,.) = 
  0.0204  0.1000  0.2000
  0.1250  0.0526  0.1667
  0.2000  0.0606  0.8235
[ CPUDoubleType{4,3,3} ]
* petalwidth: (3) : [3, 6, 3]
(1,.,.) = 
  0.5000  0.0417  0.0714
  0.3333  0.1111  0.0909
  0.5000  0.1000  0.2000
  0.7778  0.0909  0.0667
  0.8667  0.1000  0.0667
  0.9394  0.2500  0.1250

(2,.,.) = 
  0.2500  0.9167  0.2857
  0.3333  0.7778  0.1818
  0.2500  0.8000  0.2000
  0.1111  0.8182  0.1333
  0.0667  0.7000  0.0667
  0.0303  0.5000  0.1250

(3,.,.) = 
  0.2500  0.0417  0.6429
  0.3333  0.1111  0.7273
  0.2500  0.1000  0.6000
  0.1111  0.0909  0.8000
  0.0667  0.2000  0.8667
  0.0303  0.2500  0.7500
[ CPUDoubleType{3,6,3} ]
* sepallength: (3) : [3, 3]
 0.8679  0.1321  0.0377
 0.0943  0.3019  0.0566
 0.0377  0.5660  0.9057
[ CPUDoubleType{3,3} ]
* sepalwidth: (6) : [6, 3, 3]
(1,.,.) = 
  0.0392  0.5000  0.2857
  0.1000  0.4286  0.2500
  0.1429  0.2571  0.1887

(2,.,.) = 
  0.0196  0.0833  0.1429
  0.1000  0.1429  0.2500
  0.1429  0.1429  0.1509

(3,.,.) = 
  0.0392  0.0833  0.1429
  0.1000  0.1429  0.1250
  0.1429  0.1714  0.0566

(4,.,.) = 
  0.1373  0.1667  0.1429
  0.1000  0.1905  0.1250
  0.1429  0.1429  0.2453

(5,.,.) = 
  0.2549  0.0833  0.1429
  0.1000  0.0476  0.1250
  0.1429  0.2286  0.2453

(6,.,.) = 
  0.5098  0.0833  0.1429
  0.5000  0.0476  0.1250
  0.2857  0.0571  0.1132
[ CPUDoubleType{6,3,3} ]
)";
    REQUIRE(res == expected);
}

TEST_CASE("Test Smoothing A", "[Network]")
{
    /*
    Tomando m = 1 Pa = 0.5
    Si estoy calculando P(A | C), con C en{ 0,1,2 } y tengo :
    AC = { 11, 12, 11, 10, 10, 12, 10, 01, 00, 02 }
    Entonces:
    P(A = 1 | C = 0) = (3 + 1 / 2 * 1) / (4 + 1) = 3.5 / 5
    P(A = 0 | C = 0) = (1 + 1 / 2 * 1) / (4 + 1) = 1.5 / 5
    Donde m aquí es el número de veces de C = 0 que es la que condiciona y la a priori vuelve a ser sobre A que es sobre las que estaríamos calculando esas marginales.
    P(A = 1 | C = 1) = (2 + 1 / 2 * 1) / (3 + 1) = 2.5 / 4
    P(A = 0 | C = 1) = (1 + 1 / 2 * 1) / (3 + 1) = 1.5 / 4
    P(A = 1 | C = 2) = (2 + 1 / 2 * 1) / (3 + 1) = 2.5 / 5
    P(A = 0 | C = 2) = (1 + 1 / 2 * 1) / (3 + 1) = 1.5 / 5
    En realidad es parecido a Laplace, que en este caso p.e.con C = 0 sería
    P(A = 1 | C = 0) = (3 + 1) / (4 + 2) = 4 / 6
    P(A = 0 | C = 0) = (1 + 1) / (4 + 2) = 2 / 6
    */
    auto net = bayesnet::Network();
    net.addNode("A");
    net.addNode("C");
    net.addEdge("C", "A");
    std::vector<int> C = { 1, 2, 1, 0, 0, 2, 0, 1, 0, 2 };
    std::vector<std::vector<int>> A = { { 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 } };
    std::map<std::string, std::vector<int>> states = { { "A", {0, 1} }, { "C", {0, 1, 2} } };
    auto weights = std::vector<double>(C.size(), 1);
    //
    // Laplace
    //
    net.fit(A, C, weights, { "A" }, "C", states, bayesnet::Smoothing_t::LAPLACE);
    auto cpt_c_laplace = net.getNodes().at("C")->getCPT();
    REQUIRE(cpt_c_laplace.size(0) == 3);
    auto laplace_c = std::vector<float>({ 0.3846, 0.3077, 0.3077 });
    for (int i = 0; i < laplace_c.size(); ++i) {
        REQUIRE(cpt_c_laplace.index({ i }).item<float>() == Catch::Approx(laplace_c[i]).margin(threshold));
    }
    auto cpt_a_laplace = net.getNodes().at("A")->getCPT();
    REQUIRE(cpt_a_laplace.size(0) == 2);
    REQUIRE(cpt_a_laplace.size(1) == 3);
    auto laplace_a = std::vector<std::vector<float>>({ {0.3333, 0.4000,0.4000}, {0.6667,  0.6000,  0.6000} });
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(cpt_a_laplace.index({ i, j }).item<float>() == Catch::Approx(laplace_a[i][j]).margin(threshold));
        }
    }
    //
    // Cestnik
    //
    net.fit(A, C, weights, { "A" }, "C", states, bayesnet::Smoothing_t::CESTNIK);
    auto cpt_c_cestnik = net.getNodes().at("C")->getCPT();
    REQUIRE(cpt_c_cestnik.size(0) == 3);
    auto cestnik_c = std::vector<float>({ 0.3939, 0.3030, 0.3030 });
    for (int i = 0; i < laplace_c.size(); ++i) {
        REQUIRE(cpt_c_cestnik.index({ i }).item<float>() == Catch::Approx(cestnik_c[i]).margin(threshold));
    }
    auto cpt_a_cestnik = net.getNodes().at("A")->getCPT();
    REQUIRE(cpt_a_cestnik.size(0) == 2);
    REQUIRE(cpt_a_cestnik.size(1) == 3);
    auto cestnik_a = std::vector<std::vector<float>>({ {0.3000, 0.3750, 0.3750}, {0.7000, 0.6250, 0.6250} });
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(cpt_a_cestnik.index({ i, j }).item<float>() == Catch::Approx(cestnik_a[i][j]).margin(threshold));
        }
    }
}

TEST_CASE("Test Smoothing B", "[Network]")
{
    auto net = bayesnet::Network();
    net.addNode("X");
    net.addNode("Y");
    net.addNode("Z");
    net.addNode("C");
    net.addEdge("C", "X");
    net.addEdge("C", "Y");
    net.addEdge("C", "Z");
    net.addEdge("Y", "Z");
    std::vector<int> C = { 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1 };
    std::vector<std::vector<int>> Data = {
        { 0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0},
        { 1,2,0,2,2,2,1,0,0,1,1,1,0,1,2,1,0,2},
        { 2,1,3,3,2,0,0,1,3,2,1,2,2,3,0,0,1,2}
    };
    std::map<std::string, std::vector<int>> states = {
        { "X", {0, 1} },
        { "Y", {0, 1, 2} },
        { "Z", {0, 1, 2, 3} },
        { "C", {0, 1} }
    };
    auto weights = std::vector<double>(C.size(), 1);
    // See https://www.overleaf.com/read/tfnhpfysfkfx#2d576c example for calculations
    INFO("Test Smoothing B - Laplace");
    net.fit(Data, C, weights, { "X", "Y", "Z" }, "C", states, bayesnet::Smoothing_t::LAPLACE);
    auto laplace_values = std::vector<std::vector<float>>({ {0.377418, 0.622582}, {0.217821, 0.782179} });
    auto laplace_score = net.predict_proba({ {0, 1}, {1, 2}, {2, 3} });
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            REQUIRE(laplace_score.at(i).at(j) == Catch::Approx(laplace_values.at(i).at(j)).margin(threshold));
        }
    }
    INFO("Test Smoothing B - Original");
    net.fit(Data, C, weights, { "X", "Y", "Z" }, "C", states, bayesnet::Smoothing_t::ORIGINAL);
    auto original_values = std::vector<std::vector<float>>({ {0.344769, 0.655231}, {0.0421263, 0.957874} });
    auto original_score = net.predict_proba({ {0, 1}, {1, 2}, {2, 3} });
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            REQUIRE(original_score.at(i).at(j) == Catch::Approx(original_values.at(i).at(j)).margin(threshold));
        }
    }
    INFO("Test Smoothing B - Cestnik");
    net.fit(Data, C, weights, { "X", "Y", "Z" }, "C", states, bayesnet::Smoothing_t::CESTNIK);
    auto cestnik_values = std::vector<std::vector<float>>({ {0.353422, 0.646578}, {0.12364, 0.87636} });
    auto cestnik_score = net.predict_proba({ {0, 1}, {1, 2}, {2, 3} });
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            REQUIRE(cestnik_score.at(i).at(j) == Catch::Approx(cestnik_values.at(i).at(j)).margin(threshold));
        }
    }
    INFO("Test Smoothing B - No smoothing");
    net.fit(Data, C, weights, { "X", "Y", "Z" }, "C", states, bayesnet::Smoothing_t::NONE);
    auto nosmooth_values = std::vector<std::vector<float>>({ {0.342465753, 0.65753424}, {0.0, 1.0} });
    auto nosmooth_score = net.predict_proba({ {0, 1}, {1, 2}, {2, 3} });
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 2; ++j) {
            REQUIRE(nosmooth_score.at(i).at(j) == Catch::Approx(nosmooth_values.at(i).at(j)).margin(threshold));
        }
    }
}