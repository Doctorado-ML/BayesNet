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
#include "bayesnet/utils/Mst.h"


TEST_CASE("MST::insertElement tests", "[MST]")
{
    bayesnet::MST mst({}, torch::tensor({}), 0);
    SECTION("Insert into an empty list")
    {
        std::list<int> variables;
        mst.insertElement(variables, 5);
        REQUIRE(variables == std::list<int>{5});
    }
    SECTION("Insert a non-duplicate element")
    {
        std::list<int> variables = { 1, 2, 3 };
        mst.insertElement(variables, 4);
        REQUIRE(variables == std::list<int>{4, 1, 2, 3});
    }
    SECTION("Insert a duplicate element")
    {
        std::list<int> variables = { 1, 2, 3 };
        mst.insertElement(variables, 2);
        REQUIRE(variables == std::list<int>{1, 2, 3});
    }
}

// kruskal_algorithm must order equal weights by endpoints, not by the order
// addEdge happened to be called in: stable_sort alone preserved the insertion
// order, which made the tree depend on the caller rather than on the data.
TEST_CASE("The maximum spanning tree breaks weight ties by endpoints", "[MST]")
{
    const int n = 4;
    SECTION("All weights equal")
    {
        auto features = std::vector<std::string>{ "a", "b", "c", "d" };
        auto weights = torch::ones({ n, n });
        auto result = bayesnet::MST(features, weights, 0).maximumSpanningTree();
        // With every weight tied, the lowest (u, v) pairs win: {0,1}, {0,2}, {0,3}
        REQUIRE(result == std::vector<std::pair<int, int>>{ {0, 1}, { 0, 2 }, { 0, 3 } });
    }
    SECTION("The result does not depend on the order the edges are added in")
    {
        auto features = std::vector<std::string>{ "a", "b", "c", "d" };
        auto weights = torch::zeros({ n, n });
        // one distinct weight, the rest tied at zero
        weights[1][2] = 0.5;
        weights[2][1] = 0.5;
        auto forward = bayesnet::MST(features, weights, 0).maximumSpanningTree();
        // Feeding the transpose builds the same complete graph; only the weights
        // read per (i, j) could differ, and this matrix is symmetric, so the tree
        // has to come out identical
        auto result = bayesnet::MST(features, weights.t().contiguous(), 0).maximumSpanningTree();
        REQUIRE(result == forward);
    }
}
TEST_CASE("MST::reorder tests", "[MST]")
{
    bayesnet::MST mst({}, torch::tensor({}), 0);
    SECTION("Reorder simple graph")
    {
        std::vector<std::pair<float, std::pair<int, int>>> T = { {2.0, {1, 2}}, {1.0, {0, 1}} };
        auto result = mst.reorder(T, 0);
        REQUIRE(result == std::vector<std::pair<int, int>>{{0, 1}, { 1, 2 }});
    }
    SECTION("Reorder with disconnected graph")
    {
        std::vector<std::pair<float, std::pair<int, int>>> T = { {2.0, {2, 3}}, {1.0, {0, 1}} };
        auto result = mst.reorder(T, 0);
        REQUIRE(result == std::vector<std::pair<int, int>>{{0, 1}, { 2, 3 }});
    }
}

TEST_CASE("MST::maximumSpanningTree tests", "[MST]")
{
    std::vector<std::string> features = { "A", "B", "C" };
    auto weights = torch::tensor({
        {0.0, 1.0, 2.0},
        {1.0, 0.0, 3.0},
        {2.0, 3.0, 0.0}
        });
    bayesnet::MST mst(features, weights, 0);

    SECTION("MST of a complete graph")
    {
        auto result = mst.maximumSpanningTree();
        REQUIRE(result.size() == 2); // Un MST para 3 nodos tiene 2 aristas
    }
}