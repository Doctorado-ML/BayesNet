// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include <vector>
#include "TestUtils.h"
#include "bayesnet/network/Network.h"



TEST_CASE("Test Node children and parents", "[Node]")
{
    auto node = bayesnet::Node("Node");
    REQUIRE(node.getName() == "Node");
    auto parent_1 = bayesnet::Node("P1");
    auto parent_2 = bayesnet::Node("P2");
    auto child_1 = bayesnet::Node("H1");
    auto child_2 = bayesnet::Node("H2");
    auto child_3 = bayesnet::Node("H3");
    node.addParent(&parent_1);
    node.addParent(&parent_2);
    node.addChild(&child_1);
    node.addChild(&child_2);
    node.addChild(&child_3);
    auto parents = node.getParents();
    auto children = node.getChildren();
    REQUIRE(parents.size() == 2);
    REQUIRE(children.size() == 3);
    REQUIRE(parents[0]->getName() == "P1");
    REQUIRE(parents[1]->getName() == "P2");
    REQUIRE(children[0]->getName() == "H1");
    REQUIRE(children[1]->getName() == "H2");
    REQUIRE(children[2]->getName() == "H3");
    node.removeParent(&parent_1);
    node.removeChild(&child_1);
    parents = node.getParents();
    children = node.getChildren();
    REQUIRE(parents.size() == 1);
    REQUIRE(children.size() == 2);
    node.clear();
    parents = node.getParents();
    children = node.getChildren();
    REQUIRE(parents.size() == 0);
    REQUIRE(children.size() == 0);
}
TEST_CASE("Test Node computeCPT", "[Node]")
{
    // Generate a test to test the computeCPT method of the Node class
    // Create a dataset with 3 features and 4 samples
    // The dataset is a 2D tensor with 4 rows and 4 columns
    auto dataset = torch::tensor({ {1, 0, 0, 1}, {1, 1, 2, 0}, {0, 1, 2, 1}, {0, 1, 0, 1} });
    auto states = std::vector<int>({ 2, 3, 3 });
    // Create a vector with the names of the features
    auto features = std::vector<std::string>{ "F1", "F2", "F3" };
    // Create a vector with the names of the classes
    auto className = std::string("Class");
    // weights
    auto weights = torch::tensor({ 1.0, 1.0, 1.0, 1.0 }, torch::kDouble);
    std::vector<bayesnet::Node> nodes;
    for (int i = 0; i < features.size(); i++) {
        auto node = bayesnet::Node(features[i]);
        node.setNumStates(states[i]);
        nodes.push_back(node);
    }
    // Create node class with 2 states
    nodes.push_back(bayesnet::Node(className));
    nodes[features.size()].setNumStates(2);
    // The network is c->f1, f2, f3 y f1->f2, f3 
    for (int i = 0; i < features.size(); i++) {
        // Add class node as parent of all feature nodes
        nodes[i].addParent(&nodes[features.size()]);
        // Node[0] -> Node[1], Node[2]
        if (i > 0)
            nodes[i].addParent(&nodes[0]);
    }
    features.push_back(className);
    // Compute the conditional probability table
    nodes[1].computeCPT(dataset, features, 0.0, weights);
    // Get the conditional probability table
    auto cpTable = nodes[1].getCPT();
    // Get the dimensions of the conditional probability table
    auto dimensions = cpTable.sizes();
    // Check the dimensions of the conditional probability table
    REQUIRE(dimensions.size() == 3);
    REQUIRE(dimensions[0] == 3);
    REQUIRE(dimensions[1] == 2);
    REQUIRE(dimensions[2] == 2);
    // Check the values of the conditional probability table
    REQUIRE(cpTable[0][0][0].item<float>() == Catch::Approx(0));
    REQUIRE(cpTable[0][0][1].item<float>() == Catch::Approx(0));
    REQUIRE(cpTable[0][1][0].item<float>() == Catch::Approx(0));
    REQUIRE(cpTable[0][1][1].item<float>() == Catch::Approx(1));
    REQUIRE(cpTable[1][0][0].item<float>() == Catch::Approx(0));
    REQUIRE(cpTable[1][0][1].item<float>() == Catch::Approx(1));
    REQUIRE(cpTable[1][1][0].item<float>() == Catch::Approx(1));
    REQUIRE(cpTable[1][1][1].item<float>() == Catch::Approx(0));
    // Compute evidence
    for (auto& node : nodes) {
        node.computeCPT(dataset, features, 0.0, weights);
    }
    auto evidence = std::map<std::string, int>{ { "F1", 0 },  { "F2", 1 }, { "F3", 1 } };
    REQUIRE(nodes[3].getFactorValue(evidence) == 0.5);
    // Oddities
    auto features_back = features;
    // Remove a parent from features
    // features.pop_back();
    // REQUIRE_THROWS_AS(nodes[0].computeCPT(dataset, features, 0.0, weights), std::logic_error);
    // REQUIRE_THROWS_WITH(nodes[0].computeCPT(dataset, features, 0.0, weights), "Feature parent Class not found in dataset");
    // Remove a feature from features
    // features = features_back;
    // features.erase(features.begin());
    // REQUIRE_THROWS_AS(nodes[0].computeCPT(dataset, features, 0.0, weights), std::logic_error);
    // REQUIRE_THROWS_WITH(nodes[0].computeCPT(dataset, features, 0.0, weights), "Feature F1 not found in dataset");
}
TEST_CASE("TEST MinFill method", "[Node]")
{
    // Generate a test to test the minFill method of the Node class
    // Create a graph with 5 nodes
    // The graph is a chain with some additional edges
    // 0 -> 1,2,3
    // 1 -> 2,4
    // 2 -> 3
    // 3 -> 4
    auto node_0 = bayesnet::Node("0");
    auto node_1 = bayesnet::Node("1");
    auto node_2 = bayesnet::Node("2");
    auto node_3 = bayesnet::Node("3");
    auto node_4 = bayesnet::Node("4");
    // node 0
    node_0.addChild(&node_1);
    node_0.addChild(&node_2);
    node_0.addChild(&node_3);
    // node 1
    node_1.addChild(&node_2);
    node_1.addChild(&node_4);
    node_1.addParent(&node_0);
    // node 2
    node_2.addChild(&node_3);
    node_2.addChild(&node_4);
    node_2.addParent(&node_0);
    node_2.addParent(&node_1);
    // node 3
    node_3.addChild(&node_4);
    node_3.addParent(&node_0);
    node_3.addParent(&node_2);
    // node 4
    node_4.addParent(&node_1);
    node_4.addParent(&node_3);
    REQUIRE(node_0.minFill() == 3);
    REQUIRE(node_1.minFill() == 3);
    REQUIRE(node_2.minFill() == 6);
    REQUIRE(node_3.minFill() == 3);
    REQUIRE(node_4.minFill() == 1);
}
TEST_CASE("Test operator =", "[Node]")
{
    // Generate a test to test the operator = of the Node class
    // Create a node with 3 parents and 2 children
    auto node = bayesnet::Node("N1");
    auto parent_1 = bayesnet::Node("P1");
    parent_1.setNumStates(3);
    auto child_1 = bayesnet::Node("H1");
    child_1.setNumStates(2);
    node.addParent(&parent_1);
    node.addChild(&child_1);
    // Create a cpt in the node using computeCPT
    auto dataset = torch::tensor({ {1, 0, 0, 1}, {0, 1, 2, 1}, {0, 1, 1, 0} });
    auto states = std::vector<int>({ 2, 3, 3 });
    auto features = std::vector<std::string>{ "N1", "P1", "H1" };
    auto className = std::string("Class");
    auto weights = torch::tensor({ 1.0, 1.0, 1.0, 1.0 }, torch::kDouble);
    node.setNumStates(2);
    node.computeCPT(dataset, features, 0.0, weights);
    // Get the cpt of the node
    auto cpt = node.getCPT();
    // Check that the cpt is not empty
    REQUIRE(cpt.numel() > 0);
    // Check that the cpt has the correct dimensions
    auto dimensions = cpt.sizes();
    REQUIRE(dimensions.size() == 2);
    REQUIRE(dimensions[0] == 2); // Number of states of the node
    REQUIRE(dimensions[1] == 3); // Number of states of the first parent
    // Create a copy of the node
    auto node_copy = node;
    // Check that the copy has not any parents or children
    auto parents = node_copy.getParents();
    auto children = node_copy.getChildren();
    REQUIRE(parents.size() == 0);
    REQUIRE(children.size() == 0);
    // Check that the copy has the same name
    REQUIRE(node_copy.getName() == "N1");
    // Check that the copy has the same cpt
    auto cpt_copy = node_copy.getCPT();
    REQUIRE(cpt_copy.equal(cpt));
    // Check that the copy has the same number of states
    REQUIRE(node_copy.getNumStates() == node.getNumStates());
}