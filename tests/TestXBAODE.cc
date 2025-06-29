// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "TestUtils.h"
#include "bayesnet/ensembles/XBAODE.h"

TEST_CASE("Normal test", "[XBAODE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::XBAODE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 20);
    REQUIRE(clf.getNumberOfEdges() == 36);
    REQUIRE(clf.getNotes().size() == 1);
    REQUIRE(clf.getVersion() == "0.9.7");
    REQUIRE(clf.getNotes()[0] == "Number of models: 4");
    REQUIRE(clf.getNumberOfStates() == 256);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.933333));
}
TEST_CASE("Feature_select CFS", "[XBAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBAODE();
    clf.setHyperparameters({ {"select_features", "CFS"} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 171);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.720930219));
}
TEST_CASE("Feature_select IWSS", "[XBAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBAODE();
    clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 171);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with IWSS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.720930219f));
}
TEST_CASE("Feature_select FCBF", "[XBAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBAODE();
    clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 171);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 4 of 9 with FCBF");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.720930219));
}
TEST_CASE("Test used features in train note and score", "[XBAODE]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::XBAODE();
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"select_features", "CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 72);
    REQUIRE(clf.getNumberOfEdges() == 136);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 7 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 8");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.82421875f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.82421875f).epsilon(raw.epsilon));
}
TEST_CASE("Order asc, desc & random", "[XBAODE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{ {"asc", 0.83645f}, {"desc", 0.84579f}, {"rand", 0.84112} };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::XBAODE();
        clf.setHyperparameters({
            {"order", order},
            {"bisection", false},
            {"maxTolerance", 1},
            {"convergence", false},
            });
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        INFO("XBAODE order: " << order);
        REQUIRE(score == Catch::Approx(scores[order]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[order]).epsilon(raw.epsilon));
    }
}
TEST_CASE("Oddities", "[XBAODE]")
{
    auto clf = bayesnet::XBAODE();
    auto raw = RawDatasets("iris", true);
    auto bad_hyper = nlohmann::json{
        {{"order", "duck"}},
        {{"select_features", "duck"}},
        {{"maxTolerance", 0}},
        {{"maxTolerance", 7}},
    };
    for (const auto& hyper : bad_hyper.items()) {
        INFO("XBAODE hyper: " << hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"maxTolerance", 0} }), std::invalid_argument);
    auto bad_hyper_fit = nlohmann::json{
        {{"select_features", "IWSS"}, {"threshold", -0.01}},
        {{"select_features", "IWSS"}, {"threshold", 0.51}},
        {{"select_features", "FCBF"}, {"threshold", 1e-8}},
        {{"select_features", "FCBF"}, {"threshold", 1.01}},
    };
    for (const auto& hyper : bad_hyper_fit.items()) {
        INFO("XBAODE hyper: " << hyper.value().dump());
        clf.setHyperparameters(hyper.value());
        REQUIRE_THROWS_AS(clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing),
            std::invalid_argument);
    }
    auto bad_hyper_fit2 = nlohmann::json{
        {{"alpha_block", true}, {"block_update", true}},
        {{"bisection", false}, {"block_update", true}},
    };
    for (const auto& hyper : bad_hyper_fit2.items()) {
        INFO("XBAODE hyper: " << hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
}
TEST_CASE("Bisection Best", "[XBAODE]")
{
    auto clf = bayesnet::XBAODE();
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1200, true, false);
    clf.setHyperparameters({
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", false},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 210);
    REQUIRE(clf.getNumberOfEdges() == 406);
    REQUIRE(clf.getNotes().size() == 1);
    REQUIRE(clf.getNotes().at(0) == "Number of models: 14");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.991666675f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.991666675f).epsilon(raw.epsilon));
}
TEST_CASE("Bisection Best vs Last", "[XBAODE]")
{
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1500, true, false);
    auto clf = bayesnet::XBAODE();
    auto hyperparameters = nlohmann::json{
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", true},
    };
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_best = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score_best == Catch::Approx(0.973333359f).epsilon(raw.epsilon));
    // Now we will set the hyperparameter to use the last accuracy
    hyperparameters["convergence_best"] = false;
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_last = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score_last == Catch::Approx(0.976666689f).epsilon(raw.epsilon));
}
TEST_CASE("Block Update", "[XBAODE]")
{
    auto clf = bayesnet::XBAODE();
    auto raw = RawDatasets("mfeat-factors", true, 500);
    clf.setHyperparameters({
        {"bisection", true},
        {"block_update", true},
        {"maxTolerance", 3},
        {"convergence", true},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 1085);
    REQUIRE(clf.getNumberOfEdges() == 2165);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Used features in train: 20 of 216");
    REQUIRE(clf.getNotes()[2] == "Number of models: 5");
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
TEST_CASE("Alphablock", "[XBAODE]")
{
    auto clf_alpha = bayesnet::XBAODE();
    auto clf_no_alpha = bayesnet::XBAODE();
    auto raw = RawDatasets("diabetes", true);
    clf_alpha.setHyperparameters({
        {"alpha_block", true},
        });
    clf_alpha.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    clf_no_alpha.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_alpha = clf_alpha.score(raw.X_test, raw.y_test);
    auto score_no_alpha = clf_no_alpha.score(raw.X_test, raw.y_test);
    REQUIRE(score_alpha == Catch::Approx(0.720779f).epsilon(raw.epsilon));
    REQUIRE(score_no_alpha == Catch::Approx(0.733766f).epsilon(raw.epsilon));
}
