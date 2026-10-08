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
#include "bayesnet/classifiers/XSP2DE.h"
#include <algorithm>
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
    REQUIRE(clf.getNumberOfNodes() == 30);
    REQUIRE(clf.getNumberOfEdges() == 54);
    REQUIRE(clf.getNotes().size() == 1);
    REQUIRE(clf.getVersion() == "0.9.8");
    // iris has 4 features -> C(4,2)=6 distinct pairs; each used once (no repeat),
    // so training stops on pair exhaustion, not convergence.
    REQUIRE(clf.getNotes()[0] == "Number of models: 6");
    REQUIRE(clf.getNumberOfStates() == 384);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(1.0f));
    REQUIRE(clf.graph().size() == 6);
}
TEST_CASE("Feature selection seeding is rejected", "[XBA2DE]")
{
    // XBA2DE always starts from an empty ensemble: CFS/IWSS/FCBF seeding and
    // its companion 'threshold' are deliberately not valid hyperparameters here
    // (mirrors how the BoostAODE experiments were run).
    auto clf = bayesnet::XBA2DE();
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"select_features", "CFS"} }), std::invalid_argument);
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5} }), std::invalid_argument);
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7} }), std::invalid_argument);
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"threshold", 0.5} }), std::invalid_argument);
}
TEST_CASE("Order asc, desc & random", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{ {"asc", 0.81775701}, {"desc", 0.822429895}, {"rand", 0.831775725} };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::XBA2DE();
        clf.setHyperparameters({
            {"order", order},
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
        {{"maxTolerance", 0}},
        {{"maxTolerance", 7}},
        {{"beta", -1.0}},
        // Knobs fixed in XBA2DE -> not accepted as hyperparameters (even with
        // their fixed value, e.g. bisection=true).
        {{"select_features", "CFS"}},
        {{"select_features", "duck"}},
        {{"threshold", 0.1}},
        {{"bisection", false}},
        {{"bisection", true}},
        {{"block_update", true}},
        {{"alpha_block", true}},
    };
    for (const auto& hyper : bad_hyper.items()) {
        INFO("XBA2DE hyper: " << hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
}
TEST_CASE("Bisection Best", "[XBA2DE]")
{
    auto clf = bayesnet::XBA2DE();
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1200, true, false);
    clf.setHyperparameters({
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", false},
        });
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("Bisection", clf); std::cerr << "GOLDEN[Bisection] score=" << clf.score(raw.X_test, raw.y_test) << std::endl;
    REQUIRE(clf.getNumberOfNodes() == 180);
    REQUIRE(clf.getNumberOfEdges() == 468);
    REQUIRE(clf.getNumberOfStates() == 16968);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes().at(0) == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes().at(1) == "Pairs not used in train: 64");
    REQUIRE(clf.getNotes().at(2) == "Number of models: 12");
    auto score = clf.score(raw.X_test, raw.y_test);
    auto scoret = clf.score(raw.X_test, raw.y_test);
    REQUIRE(score == Catch::Approx(0.995833337).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.995833337).epsilon(raw.epsilon));
}
TEST_CASE("Bisection Best vs Last", "[XBA2DE]")
{
    auto raw = RawDatasets("kdd_JapaneseVowels", true, 1500, true, false);
    auto clf = bayesnet::XBA2DE();
    auto hyperparameters = nlohmann::json{
        {"maxTolerance", 3},
        {"convergence", true},
        {"convergence_best", true},
    };
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_best = clf.score(raw.X_test, raw.y_test);
    std::cerr << "GOLDEN[Bisection-best] score_best=" << score_best << std::endl;
    REQUIRE(score_best == Catch::Approx(0.983333349).epsilon(raw.epsilon));
    // Now we will set the hyperparameter to use the last accuracy
    hyperparameters["convergence_best"] = false;
    clf.setHyperparameters(hyperparameters);
    clf.fit(raw.X_train, raw.y_train, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_last = clf.score(raw.X_test, raw.y_test);
    std::cerr << "GOLDEN[Bisection-last] score_last=" << score_last << std::endl;
    REQUIRE(score_last == Catch::Approx(0.99000001).epsilon(raw.epsilon));
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
TEST_CASE("Working memory budget", "[XBA2DE]")
{
    auto raw = RawDatasets("glass", true);

    // The budget is a size, so it must be non-negative.
    auto bad = bayesnet::XBA2DE();
    REQUIRE_THROWS_AS(bad.setHyperparameters({ {"max_memory_gb", -1.0} }), std::invalid_argument);

    // Reference run: no budget at all.
    bayesnet::XBA2DE unlimited;
    unlimited.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);

    // max_memory_gb = 0 means unlimited, so it must reproduce the reference run
    // exactly: the accounting is inert and no note mentions memory.
    bayesnet::XBA2DE zero;
    zero.setHyperparameters({ {"max_memory_gb", 0.0} });
    zero.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(zero.getNumberOfNodes() == unlimited.getNumberOfNodes());
    REQUIRE(zero.getNumberOfEdges() == unlimited.getNumberOfEdges());
    REQUIRE(zero.getNotes() == unlimited.getNotes());
    REQUIRE(zero.score(raw.Xt, raw.yt) == Catch::Approx(unlimited.score(raw.Xt, raw.yt)));

    // A budget too small for even the first model is an error, not a silent
    // empty ensemble: one byte cannot hold any XSp2de.
    bayesnet::XBA2DE tiny;
    tiny.setHyperparameters({ {"max_memory_gb", 1.0 / (1 << 30)} });
    REQUIRE_THROWS_AS(tiny.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing),
                      std::runtime_error);

    // A budget that fits some models but not all. Size it from the models
    // themselves -- the peak of the most expensive pair, times three -- so the
    // test does not hardcode the absolute footprint of a glass model: at least
    // one model always fits (no throw) and far fewer than the 36 pairs of glass do.
    std::vector<int> stateCounts;
    for (const auto& feature : raw.features) stateCounts.push_back((int)raw.states.at(feature).size());
    size_t worstPair = 0;
    for (size_t i = 0; i < stateCounts.size(); ++i)
        for (size_t j = i + 1; j < stateCounts.size(); ++j)
            worstPair = std::max(worstPair,
                bayesnet::XSp2de::estimateFootprint(stateCounts, (int)raw.states.at(raw.className).size(), (int)i, (int)j));

    bayesnet::XBA2DE limited;
    limited.setHyperparameters({ {"max_memory_gb", 3.0 * (double)worstPair / (double)(1ULL << 30)} });
    limited.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    DUMP("Memory-limited", limited);
    REQUIRE(limited.getNumberOfNodes() < unlimited.getNumberOfNodes());
    REQUIRE(limited.getStatus() == bayesnet::WARNING);
    auto notes = limited.getNotes();
    REQUIRE(std::any_of(notes.begin(), notes.end(), [](const std::string& n) {
        return n.rfind("Memory limit reached:", 0) == 0; }));
}
