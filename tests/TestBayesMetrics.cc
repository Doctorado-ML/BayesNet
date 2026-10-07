// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/utils/BayesMetrics.h"
#include "TestUtils.h"
#include "Timer.h"

TEST_CASE("Metrics Test", "[Metrics]")
{
    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    std::map<std::string, std::pair<int, std::vector<int>>> resultsKBest = {
        {"glass", {7, { 0, 1, 7, 6, 3, 5, 2 }}},
        {"iris", {3, { 0, 3, 2 }} },
        {"ecoli", {6, { 2, 4, 1, 0, 6, 5 }}},
        {"diabetes", {2, { 7, 1 }}}
    };
    std::map<std::string, double> resultsMI = {
        {"glass", 0.12805398},
        {"iris", 0.3158139948},
        {"ecoli", 0.0089431099},
        {"diabetes", 0.0345470614}
    };
    std::map<std::pair<std::string, int>, std::vector<std::pair<int, int>>> resultsMST = {
        { {"glass", 0}, { {0, 6}, {0, 5}, {0, 3}, {0, 4}, {5, 1}, {5, 8}, {6, 2}, {6, 7} } },
        { {"glass", 1}, { {1, 5}, {5, 0}, {5, 8}, {0, 6}, {0, 3}, {0, 4}, {6, 2}, {6, 7} } },
        { {"iris", 0}, { {0, 1}, {0, 2}, {1, 3} } },
        { {"iris", 1}, { {1, 0}, {1, 3}, {0, 2} } },
        { {"ecoli", 0}, { {0, 1}, {0, 2}, {1, 5}, {1, 3}, {5, 6}, {5, 4} } },
        { {"ecoli", 1}, { {1, 0}, {1, 5}, {1, 3}, {5, 6}, {5, 4}, {0, 2} } },
        { {"diabetes", 0}, { {0, 7}, {0, 2}, {0, 6}, {2, 3}, {3, 4}, {3, 5}, {4, 1} } },
        { {"diabetes", 1}, { {1, 4}, {4, 3}, {3, 2}, {3, 5}, {2, 0}, {0, 7}, {0, 6} } }
    };
    auto raw = RawDatasets(file_name, true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    bayesnet::Metrics metricsv(raw.Xv, raw.yv, raw.features, raw.className, raw.classNumStates);

    SECTION("Test Constructor")
    {
        REQUIRE(metrics.getScoresKBest().size() == 0);
        REQUIRE(metricsv.getScoresKBest().size() == 0);
    }

    SECTION("Test SelectKBestWeighted")
    {
        std::vector<int> kBest = metrics.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
        std::vector<int> kBestv = metricsv.SelectKBestWeighted(raw.weights, true, resultsKBest.at(file_name).first);
        REQUIRE(kBest.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBestv.size() == resultsKBest.at(file_name).first);
        REQUIRE(kBest == resultsKBest.at(file_name).second);
        REQUIRE(kBestv == resultsKBest.at(file_name).second);
    }

    SECTION("Test Mutual Information")
    {
        auto result = metrics.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
        auto resultv = metricsv.mutualInformation(raw.dataset.index({ 1, "..." }), raw.dataset.index({ 2, "..." }), raw.weights);
        REQUIRE(result == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
        REQUIRE(resultv == Catch::Approx(resultsMI.at(file_name)).epsilon(raw.epsilon));
    }

    SECTION("Test Maximum Spanning Tree")
    {
        auto weights_matrix = metrics.conditionalEdge(raw.weights);
        auto weights_matrixv = metricsv.conditionalEdge(raw.weights);
        for (int i = 0; i < 2; ++i) {
            auto result = metrics.maximumSpanningTree(raw.features, weights_matrix, i);
            auto resultv = metricsv.maximumSpanningTree(raw.features, weights_matrixv, i);
            REQUIRE(result == resultsMST.at({ file_name, i }));
            REQUIRE(resultv == resultsMST.at({ file_name, i }));
        }
    }
}
TEST_CASE("Select all features ordered by Mutual Information", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto kBest = metrics.SelectKBestWeighted(raw.weights, true, 0);
    REQUIRE(kBest.size() == raw.features.size());
    REQUIRE(kBest == std::vector<int>({ 1, 0, 3, 2 }));
}
TEST_CASE("Entropy Test", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto result = metrics.entropy(raw.dataset.index({ 0, "..." }), raw.weights);
    REQUIRE(result == Catch::Approx(0.9848175048828125).epsilon(raw.epsilon));
    auto data = torch::tensor({ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 }, torch::kInt32);
    auto weights = torch::tensor({ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, torch::kFloat32);
    result = metrics.entropy(data, weights);
    REQUIRE(result == Catch::Approx(0.61086434125900269).epsilon(raw.epsilon));
    data = torch::tensor({ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, torch::kInt32);
    result = metrics.entropy(data, weights);
    REQUIRE(result == Catch::Approx(0.693147180559945).epsilon(raw.epsilon));
}
TEST_CASE("Conditional Entropy", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto expected = std::map<std::pair<int, int>, double>{
        { { 0, 1 }, 1.32674 },
        { { 0, 2 }, 0.236253 },
        { { 0, 3 }, 0.1202 },
        { { 1, 2 }, 0.252551 },
        { { 1, 3 }, 0.10515 },
        { { 2, 3 }, 0.108323 },
    };
    for (int i = 0; i < raw.features.size() - 1; ++i) {
        for (int j = i + 1; j < raw.features.size(); ++j) {
            double result = metrics.conditionalEntropy(raw.dataset.index({ i, "..." }), raw.dataset.index({ j, "..." }), raw.yt, raw.weights);
            REQUIRE(result == Catch::Approx(expected.at({ i, j })).epsilon(raw.epsilon));
        }
    }
}
TEST_CASE("Conditional Mutual Information", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    auto expected = std::map<std::pair<int, int>, double>{
        { { 0, 1 }, 0.0 },
        { { 0, 2 }, 0.287696 },
        { { 0, 3 }, 0.403749 },
        { { 1, 2 }, 1.17112 },
        { { 1, 3 }, 1.31852 },
        { { 2, 3 }, 0.210068 },
    };
    for (int i = 0; i < raw.features.size() - 1; ++i) {
        for (int j = i + 1; j < raw.features.size(); ++j) {
            double result = metrics.conditionalMutualInformation(raw.dataset.index({ i, "..." }), raw.dataset.index({ j, "..." }), raw.yt, raw.weights);
            REQUIRE(result == Catch::Approx(expected.at({ i, j })).epsilon(raw.epsilon));
        }
    }
}
TEST_CASE("Select K Pairs descending", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    std::vector<int> empty;
    auto results = metrics.SelectKPairs(raw.weights, empty, false);
    auto expected = std::vector<std::pair<std::pair<int, int>, double>>{
        { { 1, 3 }, 1.31852 },
        { { 1, 2 }, 1.17112 },
        { { 0, 3 }, 0.403749 },
        { { 0, 2 }, 0.287696 },
        { { 2, 3 }, 0.210068 },
        { { 0, 1 }, 0.0 },
    };
    auto scores = metrics.getScoresKPairs();
    for (int i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto expect = expected[i];
        auto score = scores[i];
        REQUIRE(result.first == expect.first.first);
        REQUIRE(result.second == expect.first.second);
        REQUIRE(score.first.first == expect.first.first);
        REQUIRE(score.first.second == expect.first.second);
        REQUIRE(score.second == Catch::Approx(expect.second).epsilon(raw.epsilon));
    }
    REQUIRE(results.size() == 6);
    REQUIRE(scores.size() == 6);
}
TEST_CASE("Select K Pairs ascending", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    std::vector<int> empty;
    auto results = metrics.SelectKPairs(raw.weights, empty, true);
    auto expected = std::vector<std::pair<std::pair<int, int>, double>>{
        { { 0, 1 }, 0.0 },
        { { 2, 3 }, 0.210068 },
        { { 0, 2 }, 0.287696 },
        { { 0, 3 }, 0.403749 },
        { { 1, 2 }, 1.17112 },
        { { 1, 3 }, 1.31852 },
    };
    auto scores = metrics.getScoresKPairs();
    for (int i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto expect = expected[i];
        auto score = scores[i];
        REQUIRE(result.first == expect.first.first);
        REQUIRE(result.second == expect.first.second);
        REQUIRE(score.first.first == expect.first.first);
        REQUIRE(score.first.second == expect.first.second);
        REQUIRE(score.second == Catch::Approx(expect.second).epsilon(raw.epsilon));
    }
    REQUIRE(results.size() == 6);
    REQUIRE(scores.size() == 6);
}
TEST_CASE("Select K Pairs with features excluded", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    std::vector<int> excluded = { 0, 3 };
    auto results = metrics.SelectKPairs(raw.weights, excluded, true);
    auto expected = std::vector<std::pair<std::pair<int, int>, double>>{
        { { 1, 2 }, 1.17112 },
    };
    auto scores = metrics.getScoresKPairs();
    for (int i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto expect = expected[i];
        auto score = scores[i];
        REQUIRE(result.first == expect.first.first);
        REQUIRE(result.second == expect.first.second);
        REQUIRE(score.first.first == expect.first.first);
        REQUIRE(score.first.second == expect.first.second);
        REQUIRE(score.second == Catch::Approx(expect.second).epsilon(raw.epsilon));
    }
    REQUIRE(results.size() == 1);
    REQUIRE(scores.size() == 1);
}
TEST_CASE("Select K Pairs with number of pairs descending", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    std::vector<int> empty;
    auto results = metrics.SelectKPairs(raw.weights, empty, false, 3);
    auto expected = std::vector<std::pair<std::pair<int, int>, double>>{
        { { 1, 3 }, 1.31852 },
        { { 1, 2 }, 1.17112 },
        { { 0, 3 }, 0.403749 }
    };
    auto scores = metrics.getScoresKPairs();
    REQUIRE(results.size() == 3);
    REQUIRE(scores.size() == 3);
    for (int i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto expect = expected[i];
        auto score = scores[i];
        REQUIRE(result.first == expect.first.first);
        REQUIRE(result.second == expect.first.second);
        REQUIRE(score.first.first == expect.first.first);
        REQUIRE(score.first.second == expect.first.second);
        REQUIRE(score.second == Catch::Approx(expect.second).epsilon(raw.epsilon));
    }
}
TEST_CASE("Select K Pairs with number of pairs ascending", "[Metrics]")
{
    auto raw = RawDatasets("iris", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    std::vector<int> empty;
    auto results = metrics.SelectKPairs(raw.weights, empty, true, 3);
    auto expected = std::vector<std::pair<std::pair<int, int>, double>>{
        { { 0, 3 }, 0.403749 },
        { { 1, 2 }, 1.17112 },
        { { 1, 3 }, 1.31852 }
    };
    auto scores = metrics.getScoresKPairs();
    REQUIRE(results.size() == 3);
    REQUIRE(scores.size() == 3);
    for (int i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto expect = expected[i];
        auto score = scores[i];
        REQUIRE(result.first == expect.first.first);
        REQUIRE(result.second == expect.first.second);
        REQUIRE(score.first.first == expect.first.first);
        REQUIRE(score.first.second == expect.first.second);
        REQUIRE(score.second == Catch::Approx(expect.second).epsilon(raw.epsilon));
    }
}