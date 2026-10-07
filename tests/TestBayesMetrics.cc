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
// A feature left with a single state by discretization carries no information, so
// its mutual information must be *exactly* zero, not merely close to it: whole
// blocks of edges then tie at zero and the tie-break in kruskal_algorithm is what
// defines the spanning tree. Any noise here turns the tie into an ordering, and a
// platform dependent one -- which is how glass's tree came out differently on
// arm64 than on x86_64. The marginal in the two-argument conditionalEntropy used
// to come from second.bincount(weights) while the joint came from a sequential
// loop, two different summation orders over the same weights, and one ulp of
// disagreement left ~1e-16 behind. Compared with == on purpose.
TEST_CASE("A constant feature has exactly zero mutual information", "[Metrics]")
{
    auto raw = RawDatasets("glass", true);
    bayesnet::Metrics metrics(raw.dataset, raw.features, raw.className, raw.classNumStates);
    // Si is the constant one on glass; assert that rather than assume it
    const int constantFeature = 4;
    auto column = raw.dataset.index({ constantFeature, "..." });
    REQUIRE(column.max().item<int>() == column.min().item<int>());

    auto classes = raw.dataset.index({ -1, "..." });
    for (int i = 0; i < static_cast<int>(raw.features.size()); ++i) {
        if (i == constantFeature) continue;
        auto other = raw.dataset.index({ i, "..." });
        REQUIRE(metrics.entropy(column, raw.weights) == 0.0);
        REQUIRE(metrics.mutualInformation(column, other, raw.weights) == 0.0);
        REQUIRE(metrics.mutualInformation(other, column, raw.weights) == 0.0);
        REQUIRE(metrics.conditionalMutualInformation(column, other, classes, raw.weights) == 0.0);
        // and per class, which is what conditionalEdge accumulates
        for (int value = 0; value < raw.classNumStates; ++value) {
            auto mask = classes == value;
            REQUIRE(metrics.mutualInformation(column.index({ mask }), other.index({ mask }),
                raw.weights.index({ mask })) == 0.0);
        }
    }
    // so every edge of that feature is exactly zero in the conditionalEdge matrix
    auto weights_matrix = metrics.conditionalEdge(raw.weights);
    for (int i = 0; i < static_cast<int>(raw.features.size()); ++i) {
        if (i == constantFeature) continue;
        REQUIRE(weights_matrix[constantFeature][i].item<float>() == 0.0f);
        REQUIRE(weights_matrix[i][constantFeature].item<float>() == 0.0f);
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
        { { 0, 1 }, 0.427020291 },
        { { 0, 2 }, 0.44181006 },
        { { 0, 3 }, 0.503108294 },
        { { 1, 2 }, 1.35782975 },
        { { 1, 3 }, 1.38778032 },
        { { 2, 3 }, 0.285674619 },
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
        { { 0, 1 }, 0.096928648 },
        { { 0, 2 }, 0.0821388783 },
        { { 0, 3 }, 0.0208406446 },
        { { 1, 2 }, 0.0658413624 },
        { { 1, 3 }, 0.0358907881 },
        { { 2, 3 }, 0.0327172475 },
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
        { { 0, 1 }, 0.096928648 },
        { { 0, 2 }, 0.0821388783 },
        { { 1, 2 }, 0.0658413624 },
        { { 1, 3 }, 0.0358907881 },
        { { 2, 3 }, 0.0327172475 },
        { { 0, 3 }, 0.0208406446 },
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
        { { 0, 3 }, 0.0208406446 },
        { { 2, 3 }, 0.0327172475 },
        { { 1, 3 }, 0.0358907881 },
        { { 1, 2 }, 0.0658413624 },
        { { 0, 2 }, 0.0821388783 },
        { { 0, 1 }, 0.096928648 },
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
        { { 1, 2 }, 0.0658413624 },
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
        { { 0, 1 }, 0.096928648 },
        { { 0, 2 }, 0.0821388783 },
        { { 1, 2 }, 0.0658413624 }
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
        { { 1, 2 }, 0.0658413624 },
        { { 0, 2 }, 0.0821388783 },
        { { 0, 1 }, 0.096928648 }
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