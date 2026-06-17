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

TEST_CASE("Normal test", "[XBA2DE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::XBA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getVersion() == "0.9.7");
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 13 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Number of models: 1");
    REQUIRE(clf.getNumberOfStates() == 64);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(1.0f));
    REQUIRE(clf.graph().size() == 1);
}
TEST_CASE("Feature_select CFS", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "CFS"} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 828);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.738095224).margin(PORTABLE_SCORE_MARGIN));
}
TEST_CASE("Feature_select IWSS", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 828);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with IWSS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    REQUIRE(clf.getNumberOfStates() == 8748);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.738095224).margin(PORTABLE_SCORE_MARGIN));
}
TEST_CASE("Feature_select FCBF", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XBA2DE();
    clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7} });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    // Counts depend on platform-sensitive feature selection (Linux 290 nodes, macOS
    // 180); exact structure lives in golden. Keep portable ranges + note phrases.
    REQUIRE(clf.getNumberOfNodes() >= 150);
    REQUIRE(clf.getNumberOfNodes() <= 330);
    REQUIRE(clf.getNumberOfEdges() >= 350);
    REQUIRE(clf.getNumberOfEdges() <= 800);
    REQUIRE(clf.getNumberOfStates() >= 1500);
    REQUIRE(clf.getNumberOfStates() <= 13000);
    REQUIRE(anyNoteContains(clf.getNotes(), "with FCBF"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Number of models"));
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.738095224).margin(PORTABLE_SCORE_MARGIN));
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
    // Counts depend on platform-sensitive feature selection (Linux 252 nodes, macOS
    // 189); exact structure lives in golden. Keep portable ranges + note phrases.
    REQUIRE(clf.getNumberOfNodes() >= 150);
    REQUIRE(clf.getNumberOfNodes() <= 300);
    REQUIRE(clf.getNumberOfEdges() >= 350);
    REQUIRE(clf.getNumberOfEdges() <= 640);
    REQUIRE(clf.getNumberOfStates() >= 2500);
    REQUIRE(clf.getNumberOfStates() <= 22000);
    REQUIRE(anyNoteContains(clf.getNotes(), "with CFS"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Number of models"));
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.85546875).margin(PORTABLE_SCORE_MARGIN));
    REQUIRE(scoret == Catch::Approx(0.85546875).margin(PORTABLE_SCORE_MARGIN));
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
    REQUIRE(clf.getNumberOfNodes() == 360);
    REQUIRE(clf.getNumberOfEdges() == 828);
    REQUIRE(clf.getNumberOfStates() == 8748);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 9 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 36");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.813084126).margin(PORTABLE_SCORE_MARGIN));
    REQUIRE(scoret == Catch::Approx(0.813084126).margin(PORTABLE_SCORE_MARGIN));
}
TEST_CASE("Order asc, desc & random", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{ {"asc", 0.7710280418}, {"desc", 0.803738296}, {"rand", 0.7897196412} };
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
        INFO("XBA2DE order: " << order);
        REQUIRE(score == Catch::Approx(scores[order]).margin(PORTABLE_SCORE_MARGIN));
        REQUIRE(scoret == Catch::Approx(scores[order]).margin(PORTABLE_SCORE_MARGIN));
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
    // Counts depend on platform-sensitive feature selection (Linux 330 nodes, macOS
    // 435); exact structure lives in golden. Keep portable ranges + note phrases.
    REQUIRE(clf.getNumberOfNodes() >= 280);
    REQUIRE(clf.getNumberOfNodes() <= 500);
    REQUIRE(clf.getNumberOfEdges() >= 700);
    REQUIRE(clf.getNumberOfEdges() <= 1400);
    REQUIRE(clf.getNumberOfStates() >= 14000);
    REQUIRE(clf.getNumberOfStates() <= 120000);
    REQUIRE(anyNoteContains(clf.getNotes(), "models eliminated"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Pairs not used in train"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Number of models"));
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.987447679).margin(PORTABLE_SCORE_MARGIN));
    REQUIRE(scoret == Catch::Approx(0.987447679).margin(PORTABLE_SCORE_MARGIN));
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
    REQUIRE(score_best == Catch::Approx(0.936454833).margin(PORTABLE_SCORE_MARGIN));
    // Now we will set the hyperparameter to use the last accuracy
    hyperparameters["convergence_best"] = false;
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_last = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score_last == Catch::Approx(0.963210702).margin(PORTABLE_SCORE_MARGIN));
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
    // Counts depend on platform-sensitive feature selection (Linux 120 nodes, macOS
    // 60); exact structure lives in golden. Keep portable ranges + note phrases.
    REQUIRE(clf.getNumberOfNodes() >= 50);
    REQUIRE(clf.getNumberOfNodes() <= 160);
    REQUIRE(clf.getNumberOfEdges() >= 120);
    REQUIRE(clf.getNumberOfEdges() <= 380);
    REQUIRE(anyNoteContains(clf.getNotes(), "models eliminated"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Pairs not used in train"));
    REQUIRE(anyNoteContains(clf.getNotes(), "Number of models"));
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.936454833).margin(PORTABLE_SCORE_MARGIN));
    REQUIRE(scoret == Catch::Approx(0.936454833).margin(PORTABLE_SCORE_MARGIN));
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
    REQUIRE(score_alpha == Catch::Approx(0.666666687).margin(PORTABLE_SCORE_MARGIN));
    REQUIRE(score_no_alpha == Catch::Approx(0.666666687).margin(PORTABLE_SCORE_MARGIN));
}
