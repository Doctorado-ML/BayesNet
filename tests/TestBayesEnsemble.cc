// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/BoostAODE.h"
#include "bayesnet/ensembles/AODE.h"
#include "bayesnet/ensembles/AODELd.h"
#include "TestUtils.h"


TEST_CASE("Topological Order", "[Ensemble]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto order = clf.topological_order();
    REQUIRE(order.size() == 0);
}
TEST_CASE("Dump CPT", "[Ensemble]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto dump = clf.dump_cpt();
    REQUIRE(dump.size() == 39903);
}
TEST_CASE("Number of States", "[Ensemble]")
{
    auto clf = bayesnet::BoostAODE();
    auto raw = RawDatasets("iris", true);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfStates() == 76);
}
TEST_CASE("Show", "[Ensemble]")
{
    auto clf = bayesnet::BoostAODE();
    auto raw = RawDatasets("iris", true);
    clf.setHyperparameters({
            {"bisection", false},
            {"maxTolerance", 1},
            {"convergence", false},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    std::vector<std::string> expected = {
        "class -> sepallength, sepalwidth, petallength, petalwidth, ",
        "petallength -> sepallength, sepalwidth, petalwidth, ",
        "petalwidth -> ",
        "sepallength -> ",
        "sepalwidth -> ",
        "class -> sepallength, sepalwidth, petallength, petalwidth, ",
        "petallength -> ",
        "petalwidth -> sepallength, sepalwidth, petallength, ",
        "sepallength -> ",
        "sepalwidth -> ",
        "class -> sepallength, sepalwidth, petallength, petalwidth, ",
        "petallength -> ",
        "petalwidth -> ",
        "sepallength -> sepalwidth, petallength, petalwidth, ",
        "sepalwidth -> ",
        "class -> sepallength, sepalwidth, petallength, petalwidth, ",
        "petallength -> ",
        "petalwidth -> ",
        "sepallength -> ",
        "sepalwidth -> sepallength, petallength, petalwidth, ",
    };
    auto show = clf.show();
    REQUIRE(show.size() == expected.size());
    for (size_t i = 0; i < show.size(); i++)
        REQUIRE(show[i] == expected[i]);
}
TEST_CASE("Graph", "[Ensemble]")
{
    auto clf = bayesnet::BoostAODE();
    auto raw = RawDatasets("iris", true);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = clf.graph();
    REQUIRE(graph.size() == 56);
    auto clf2 = bayesnet::AODE();
    clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    graph = clf2.graph();
    REQUIRE(graph.size() == 56);
    raw = RawDatasets("glass", false);
    auto clf3 = bayesnet::AODELd();
    clf3.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    graph = clf3.graph();
    REQUIRE(graph.size() == 261);
}
TEST_CASE("Compute ArgMax", "[Ensemble]")
{
    class TestEnsemble : public bayesnet::BoostAODE {
    public:
        TestEnsemble() : bayesnet::BoostAODE() {}
        torch::Tensor compute_arg_max(torch::Tensor& X) { return Ensemble::compute_arg_max(X); }
        std::vector<int> compute_arg_max(std::vector<std::vector<double>>& X) { return Ensemble::compute_arg_max(X); }
    };
    TestEnsemble clf;
    std::vector<std::vector<double>> X = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.9f, 0.6f},
        {0.7f, 0.8f, 0.9f},
        {0.5f, 0.2f, 0.1f},
        {0.3f, 0.7f, 0.2f},
        {0.5f, 0.5f, 0.2f}
    };
    std::vector<int> expected = { 2, 1, 2, 0, 1, 0 };
    auto argmax = clf.compute_arg_max(X);
    REQUIRE(argmax.size() == expected.size());
    REQUIRE(argmax == expected);
    auto Xt = torch::zeros({ 6, 3 }, torch::kFloat32);
    Xt[0][0] = 0.1f; Xt[0][1] = 0.2f; Xt[0][2] = 0.3f;
    Xt[1][0] = 0.4f; Xt[1][1] = 0.9f; Xt[1][2] = 0.6f;
    Xt[2][0] = 0.7f; Xt[2][1] = 0.8f; Xt[2][2] = 0.9f;
    Xt[3][0] = 0.5f; Xt[3][1] = 0.2f; Xt[3][2] = 0.1f;
    Xt[4][0] = 0.3f; Xt[4][1] = 0.7f; Xt[4][2] = 0.2f;
    Xt[5][0] = 0.5f; Xt[5][1] = 0.5f; Xt[5][2] = 0.2f;
    auto argmaxt = clf.compute_arg_max(Xt);
    REQUIRE(argmaxt.size(0) == expected.size());
    for (int i = 0; i < argmaxt.size(0); i++)
        REQUIRE(argmaxt[i].item<int>() == expected[i]);
}