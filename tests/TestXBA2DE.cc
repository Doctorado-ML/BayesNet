// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "TestUtils.h"
#include "bayesnet/ensembles/XBA2DE.h"
#include <iostream>
#define DUMP(tag, clf) do { std::cerr << "GOLDEN[" << tag << "] nodes=" << (clf).getNumberOfNodes() \
    << " edges=" << (clf).getNumberOfEdges() << " states=" << (clf).getNumberOfStates() \
    << " notes=" << (clf).getNotes().size(); for (auto& _n : (clf).getNotes()) std::cerr << " || " << _n; \
    std::cerr << std::endl; } while(0)

TEST_CASE("Normal test", "[XBA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::XBA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("Normal", clf); std::cerr << "GOLDEN[Normal] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 40);
    REQUIRE(clf.getNumberOfEdges() == 72);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getVersion() == "0.9.7");
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 13 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Number of models: 8");
    REQUIRE(clf.getNumberOfStates() == 512);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.933333f));
    REQUIRE(clf.graph().size() == 8);
}
TEST_CASE("Feature_select CFS", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "CFS"} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("CFS", clf); std::cerr << "GOLDEN[CFS] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 864);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.697674));
}
TEST_CASE("Feature_select IWSS", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("IWSS", clf); std::cerr << "GOLDEN[IWSS] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 864);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with IWSS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    REQUIRE(clf.getNumberOfStates() == 8748);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.697674));
}
TEST_CASE("Feature_select FCBF", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("FCBF", clf); std::cerr << "GOLDEN[FCBF] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 140);
    REQUIRE(clf.getNumberOfEdges() == 336);
    REQUIRE(clf.getNumberOfStates() == 3402);
    REQUIRE(clf.getNotes().size() == 4);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 4 of 9 with FCBF");
    REQUIRE(clf.getNotes()[1] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[2] == "Pairs not used in train: 2");
    REQUIRE(clf.getNotes()[3] == "Number of models: 14");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.744186));
}
TEST_CASE("Test used features in train note and score", "[XBA2DE]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"select_features", "CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    DUMP("diabetes-CFS", clf); std::cerr << "GOLDEN[diabetes-CFS] score=" << score << " scoret=" << scoret << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 252);
    REQUIRE(clf.getNumberOfEdges() == 588);
    REQUIRE(clf.getNumberOfStates() == 9632);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 8 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 28");
    REQUIRE(score == Catch::Approx(0.876302f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.876302f).epsilon(raw.epsilon));
}
TEST_CASE("Test used features in train note and score with glass", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"select_features", "CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    DUMP("glass-CFS", clf); std::cerr << "GOLDEN[glass-CFS] score=" << score << " scoret=" << scoret << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 864);
    REQUIRE(clf.getNumberOfStates() == 8748);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    REQUIRE(score == Catch::Approx(0.813084).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.813084).epsilon(raw.epsilon));
}
TEST_CASE("Order asc, desc & random", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{ {"asc", 0.771028}, {"desc", 0.845794}, {"rand", 0.766355} };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::XBA2DE();
        clf.setHyperparameters({
            {"order", order},
            {"bisection", false},
            {"maxTolerance", 1},
            {"convergence", true},
            });
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        std::cerr << "GOLDEN[order-" << order << "] score=" << score << " scoret=" << scoret << std::endl;
        INFO("XBA2DE order: " << order);
        REQUIRE(score == Catch::Approx(scores[order]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[order]).epsilon(raw.epsilon));
    }
}
TEST_CASE("Oddities", "[XBA2DE]")
{
    auto clf = bayesnet::XBA2DE();
    auto raw = RawDatasets("iris", true);
    auto bad_hyper = nlohmann::json{
        {{"order", "duck"}},
        {{"select_features", "duck"}},
        {{"maxTolerance", 0}},
        {{"maxTolerance", 7}},
    };
    for (const auto& hyper : bad_hyper.items()) {
        INFO("XBA2DE hyper: " << hyper.value().dump());
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
        INFO("XBA2DE hyper: " << hyper.value().dump());
        clf.setHyperparameters(hyper.value());
        REQUIRE_THROWS_AS(clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing),
            std::invalid_argument);
    }
    auto bad_hyper_fit2 = nlohmann::json{
        {{"alpha_block", true}, {"block_update", true}},
        {{"bisection", false}, {"block_update", true}},
    };
    for (const auto& hyper : bad_hyper_fit2.items()) {
        INFO("XBA2DE hyper: " << hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
    // Check not enough selected features
    raw.Xv.pop_back();
    raw.Xv.pop_back();
    raw.Xv.pop_back();
    raw.features.pop_back();
    raw.features.pop_back();
    raw.features.pop_back();
    clf.setHyperparameters({ {"select_features", "CFS"}, {"alpha_block", false}, {"block_update", false} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNotes().size() == 1);
    REQUIRE(clf.getNotes()[0] == "No features selected in initialization");
}
TEST_CASE("Bisection Best", "[XBA2DE]")
{
    auto clf = bayesnet::XBA2DE();
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1200, true, false);
    clf.setHyperparameters({
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", false},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("Bisection", clf); std::cerr << "GOLDEN[Bisection] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 435);
    REQUIRE(clf.getNumberOfEdges() == 1131);
    REQUIRE(clf.getNumberOfStates() == 41006);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes().at(0) == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes().at(1) == "Pairs not used in train: 83");
    REQUIRE(clf.getNotes().at(2) == "Number of models: 29");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.9625).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.9625).epsilon(raw.epsilon));
}
TEST_CASE("Bisection Best vs Last", "[XBA2DE]")
{
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1500, true, false);
    auto clf = bayesnet::XBA2DE();
    auto hyperparameters = nlohmann::json{
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", true},
    };
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_best = clf.score(raw.X_test, raw.y_test);
    std::cerr << "GOLDEN[Bisection-best] score_best=" << score_best << std::endl;
    REQUIRE(score_best == Catch::Approx(0.98).epsilon(raw.epsilon));
    // Now we will set the hyperparameter to use the last accuracy
    hyperparameters["convergence_best"] = false;
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_last = clf.score(raw.X_test, raw.y_test);
    std::cerr << "GOLDEN[Bisection-last] score_last=" << score_last << std::endl;
    REQUIRE(score_last == Catch::Approx(0.983333).epsilon(raw.epsilon));
}
TEST_CASE("Block Update", "[XBA2DE]")
{
    auto clf = bayesnet::XBA2DE();
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1500, true, false);
    clf.setHyperparameters({
        {"bisection", true},
        {"block_update", true},
        {"maxTolerance", 3},
        {"convergence", true},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("BlockUpdate", clf); std::cerr << "GOLDEN[BlockUpdate] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 180);
    REQUIRE(clf.getNumberOfEdges() == 468);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Pairs not used in train: 83");
    REQUIRE(clf.getNotes()[2] == "Number of models: 12");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.966667).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.966667).epsilon(raw.epsilon));
    /*std::cout << "Number of nodes " << clf.getNumberOfNodes() << std::endl;*/
    /*std::cout << "Number of edges " << clf.getNumberOfEdges() << std::endl;*/
    /*std::cout << "Notes size " << clf.getNotes().size() << std::endl;*/
    /*for (auto note : clf.getNotes()) {*/
    /*    std::cout << note << std::endl;*/
    /*}*/
    /*std::cout << "Score " << score << std::endl;*/
}
TEST_CASE("Alphablock", "[XBA2DE]")
{
    auto clf_alpha = bayesnet::XBA2DE();
    auto clf_no_alpha = bayesnet::XBA2DE();
    auto raw = RawDatasets("diabetes", true);
    clf_alpha.setHyperparameters({
        {"alpha_block", true},
        });
    clf_alpha.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    clf_no_alpha.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_alpha = clf_alpha.score(raw.X_test, raw.y_test);
    auto score_no_alpha = clf_no_alpha.score(raw.X_test, raw.y_test);
    std::cerr << "GOLDEN[Alphablock] score_alpha=" << score_alpha << " score_no_alpha=" << score_no_alpha << std::endl;
    REQUIRE(score_alpha == Catch::Approx(0.688312).epsilon(raw.epsilon));
    REQUIRE(score_no_alpha == Catch::Approx(0.688312).epsilon(raw.epsilon));
}
TEST_CASE("Beta joint-relevance criterion", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    // beta (ablation knob for the pair-ranking criterion) must be non-negative.
    auto bad = bayesnet::XBA2DE();
    REQUIRE_THROWS_AS(bad.setHyperparameters({ {"beta", -0.5} }), std::invalid_argument);

    // Observable signature of the fitted ensemble for a given beta.
    auto signature = [&](bool set, double beta) {
        bayesnet::XBA2DE c;
        nlohmann::json hyper = { {"order", "asc"}, {"convergence", true} };
        if (set) hyper["beta"] = beta;
        c.setHyperparameters(hyper);
        c.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        return std::make_tuple(c.getNumberOfNodes(), c.getNumberOfEdges(), c.score(raw.Xt, raw.yt));
    };
    // Explicit beta=1 (joint relevance) reproduces the default exactly.
    REQUIRE(signature(false, 0.0) == signature(true, 1.0));
    // beta=0 (marginal-relevance sum) is a different ranking -> different ensemble.
    REQUIRE(signature(true, 0.0) != signature(true, 1.0));
}
