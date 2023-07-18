#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <string>
#include "../src/KDB.h"
#include "utils.h"

TEST_CASE("Test Bayesian Network")
{
    auto[Xd, y, features, className, states] = loadFile("iris");

    SECTION("Test Update Nodes")
    {
        auto net = bayesnet::Network();
        net.addNode("A", 3);
        REQUIRE(net.getStates() == 3);
        net.addNode("A", 5);
        REQUIRE(net.getStates() == 5);
    }
    SECTION("Test get features")
    {
        auto net = bayesnet::Network();
        net.addNode("A", 3);
        net.addNode("B", 5);
        REQUIRE(net.getFeatures() == vector<string>{"A", "B"});
        net.addNode("C", 2);
        REQUIRE(net.getFeatures() == vector<string>{"A", "B", "C"});
    }
}
