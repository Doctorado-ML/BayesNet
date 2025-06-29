// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/utils/BayesMetrics.h"
#include "bayesnet/ensembles/BoostA2DE.h"
#include "TestUtils.h"


TEST_CASE("Build basic model", "[BoostA2DE]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::BoostA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 342);
    REQUIRE(clf.getNumberOfEdges() == 684);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Pairs not used in train: 20");
    REQUIRE(clf.getNotes()[2] == "Number of models: 38");
    auto score = clf.score(raw.Xv, raw.yv);
    REQUIRE(score == Catch::Approx(0.919271).epsilon(raw.epsilon));
}
TEST_CASE("Feature_select IWSS", "[BoostA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostA2DE();
    clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5 } });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 756);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with IWSS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
}
TEST_CASE("Feature_select FCBF", "[BoostA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostA2DE();
    clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7 } });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 110);
    REQUIRE(clf.getNumberOfEdges() == 231);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 4 of 9 with FCBF");
    REQUIRE(clf.getNotes()[1] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[2] == "Pairs not used in train: 2");
    REQUIRE(clf.getNotes()[3] == "Number of models: 11");
}
TEST_CASE("Test used features in train note and score", "[BoostA2DE]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::BoostA2DE(true);
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"select_features","CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 189);
    REQUIRE(clf.getNumberOfEdges() == 378);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 7 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 21");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.85546875f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.85546875f).epsilon(raw.epsilon));
}
TEST_CASE("Voting vs proba", "[BoostA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostA2DE(false);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_proba = clf.score(raw.Xv, raw.yv);
    auto pred_proba = clf.predict_proba(raw.Xv);
    clf.setHyperparameters({
        {"predict_voting",true},
        });
    auto score_voting = clf.score(raw.Xv, raw.yv);
    auto pred_voting = clf.predict_proba(raw.Xv);
    REQUIRE(score_proba == Catch::Approx(0.98).epsilon(raw.epsilon));
    REQUIRE(score_voting == Catch::Approx(0.946667).epsilon(raw.epsilon));
    REQUIRE(pred_voting[83][2] == Catch::Approx(0.53508).epsilon(raw.epsilon));
    REQUIRE(pred_proba[83][2] == Catch::Approx(0.48394).epsilon(raw.epsilon));
    REQUIRE(clf.dump_cpt().size() == 7742);
    REQUIRE(clf.topological_order() == std::vector<std::string>());
}
TEST_CASE("Order asc, desc & random", "[BoostA2DE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{
        {"asc", 0.752336f }, { "desc", 0.813084f }, { "rand", 0.850467 }
    };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::BoostA2DE();
        clf.setHyperparameters({
            {"order", order},
            {"bisection", false},
            {"maxTolerance", 1},
            {"convergence", false},
            });
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        INFO("BoostA2DE order: " + order);
        REQUIRE(score == Catch::Approx(scores[order]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[order]).epsilon(raw.epsilon));
    }
}
TEST_CASE("Oddities2", "[BoostA2DE]")
{
    auto clf = bayesnet::BoostA2DE();
    auto raw = RawDatasets("iris", true);
    auto bad_hyper = nlohmann::json{
        { { "order", "duck" } },
        { { "select_features", "duck" } },
        { { "maxTolerance", 0 } },
        { { "maxTolerance", 7 } },
    };
    for (const auto& hyper : bad_hyper.items()) {
        INFO("BoostA2DE hyper: " + hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"maxTolerance", 0 } }), std::invalid_argument);
    auto bad_hyper_fit = nlohmann::json{
        { { "select_features","IWSS" }, { "threshold", -0.01 } },
        { { "select_features","IWSS" }, { "threshold", 0.51 } },
        { { "select_features","FCBF" }, { "threshold", 1e-8 } },
        { { "select_features","FCBF" }, { "threshold", 1.01 } },
    };
    for (const auto& hyper : bad_hyper_fit.items()) {
        INFO("BoostA2DE hyper: " + hyper.value().dump());
        clf.setHyperparameters(hyper.value());
        REQUIRE_THROWS_AS(clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing), std::invalid_argument);
    }
}
TEST_CASE("No features selected", "[BoostA2DE]")
{
    // Check that the note "No features selected in initialization" is added
    //
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostA2DE();
    clf.setHyperparameters({ {"select_features","FCBF"}, {"threshold", 1 } });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNotes().size() == 1);
    REQUIRE(clf.getNotes()[0] == "No features selected in initialization");
}
TEST_CASE("Bisection Best", "[BoostA2DE]")
{
    auto clf = bayesnet::BoostA2DE();
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1200, true, false);
    clf.setHyperparameters({
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        {"block_update", false},
        {"convergence_best", true},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 480);
    REQUIRE(clf.getNumberOfEdges() == 1152);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes().at(0) == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes().at(1) == "Pairs not used in train: 83");
    REQUIRE(clf.getNotes().at(2) == "Number of models: 32");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.966667f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.966667f).epsilon(raw.epsilon));
}
TEST_CASE("Block Update", "[BoostA2DE]")
{
    auto clf = bayesnet::BoostA2DE();
    auto raw = RawDatasets("spambase", true, 500);
    clf.setHyperparameters({
        {"bisection", true},
        {"block_update", true},
        {"maxTolerance", 3},
        {"convergence", true},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 58);
    REQUIRE(clf.getNumberOfEdges() == 165);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Pairs not used in train: 1588");
    REQUIRE(clf.getNotes()[2] == "Number of models: 1");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(1.0f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(1.0f).epsilon(raw.epsilon));
    //
    // std::cout << "Number of nodes " << clf.getNumberOfNodes() << std::endl;
    // std::cout << "Number of edges " << clf.getNumberOfEdges() << std::endl;
    // std::cout << "Notes size " << clf.getNotes().size() << std::endl;
    // for (auto note : clf.getNotes()) {
    //     std::cout << note << std::endl;
    // }
    // std::cout << "Score " << score << std::endl;
}
TEST_CASE("Test graph b2a2de", "[BoostA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto graph = clf.graph();
    REQUIRE(graph.size() == 26);
    REQUIRE(graph[0] == "digraph BayesNet {\nlabel=<BayesNet BoostA2DE_0>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n");
    REQUIRE(graph[1] == "\"class\" [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n");
}
