// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "bayesnet/utils/BayesMetrics.h"
#include "bayesnet/feature_selection/CFS.h"
#include "bayesnet/feature_selection/FCBF.h"
#include "bayesnet/feature_selection/IWSS.h"
#include "TestUtils.h"

bayesnet::FeatureSelect* build_selector(RawDatasets& raw, std::string selector, double threshold, int max_features = 0)
{
    max_features = max_features == 0 ? raw.features.size() : max_features;
    if (selector == "CFS") {
        return new bayesnet::CFS(raw.dataset, raw.features, raw.className, max_features, raw.classNumStates, raw.weights);
    } else if (selector == "FCBF") {
        return new bayesnet::FCBF(raw.dataset, raw.features, raw.className, max_features, raw.classNumStates, raw.weights, threshold);
    } else if (selector == "IWSS") {
        return new bayesnet::IWSS(raw.dataset, raw.features, raw.className, max_features, raw.classNumStates, raw.weights, threshold);
    }
    return nullptr;
}

TEST_CASE("Features Selected", "[FeatureSelection]")
{
    // std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    std::string file_name = GENERATE("ecoli");

    auto raw = RawDatasets(file_name, true);

    SECTION("Test features selected, scores and sizes")
    {
        map<pair<std::string, std::string>, pair<std::vector<int>, std::vector<double>>> results = {
            { {"glass", "CFS"}, { { 2, 3, 5, 6, 7, 1, 0, 8, 4 }, {0.365513, 0.42895, 0.46186, 0.481897, 0.500943, 0.504027, 0.505625, 0.493256, 0.478226} } },
            { {"iris", "CFS"}, { { 3, 2, 0, 1 }, {0.870521, 0.890375, 0.84104719, 0.799310961} } },
            { {"ecoli", "CFS"}, { { 5, 0, 6, 1, 4, 2, 3 }, {0.512319, 0.565381, 0.61824, 0.637094, 0.637759, 0.633802, 0.598266} } },
            { {"diabetes", "CFS"}, { { 1, 5, 7, 4, 6, 0 }, {0.132858, 0.151209, 0.148887, 0.14862, 0.142902, 0.137233} } },
            { {"glass", "IWSS" }, { { 2, 3, 5, 7, 6, 1, 0, 8, 4 }, {0.365513, 0.42895, 0.46186, 0.479866, 0.500943, 0.504027, 0.505625, 0.493256, 0.478226} } },
            { {"iris", "IWSS"}, { { 3, 2, 0  }, {0.870521, 0.890375, 0.841047} }},
            { {"ecoli", "IWSS"}, { { 5, 0, 6, 1, 4, 2, 3}, {0.512319, 0.565381, 0.61824, 0.637094, 0.637759, 0.633802, 0.598266} } },
            { {"diabetes", "IWSS"}, { { 1, 5, 4, 7, 3 }, {0.132858, 0.151209, 0.146771, 0.14862, 0.136493,} } },
            { {"glass", "FCBF" }, { { 2, 3, 5, 7, 6 }, {0.365513, 0.304911, 0.302109, 0.281621, 0.253297} } },
            { {"iris", "FCBF"}, {{ 3, 2 }, {0.870521, 0.816401} }},
            { {"ecoli", "FCBF"}, {{ 5, 0, 1, 4, 2 }, {0.512319, 0.350406, 0.260905, 0.203132, 0.11229} }},
            { {"diabetes", "FCBF"}, {{ 1, 5, 7, 6 }, {0.132858, 0.083191, 0.0480135, 0.0224186} }}
        };
        double threshold;
        std::string selector;
        std::vector<std::pair<std::string, double>> selectors = {
            { "CFS", 0.0 },
            { "IWSS", 0.1 },
            { "FCBF", 1e-7 }
        };
        for (const auto item : selectors) {
            selector = item.first; threshold = item.second;
            bayesnet::FeatureSelect* featureSelector = build_selector(raw, selector, threshold);
            featureSelector->fit();
            INFO("file_name: " << file_name << ", selector: " << selector);
            // Features
            auto expected_features = results.at({ file_name, selector }).first;
            std::vector<int> selected_features = featureSelector->getFeatures();
            REQUIRE(selected_features.size() == expected_features.size());
            REQUIRE(selected_features == expected_features);
            // Scores
            auto expected_scores = results.at({ file_name, selector }).second;
            std::vector<double> selected_scores = featureSelector->getScores();
            REQUIRE(selected_scores.size() == selected_features.size());
            for (int i = 0; i < selected_scores.size(); i++) {
                REQUIRE(selected_scores[i] == Catch::Approx(expected_scores[i]).epsilon(raw.epsilon));
            }
            delete featureSelector;
        }
    }
}
TEST_CASE("Oddities", "[FeatureSelection]")
{
    auto raw = RawDatasets("iris", true);
    // FCBF Limits
    REQUIRE_THROWS_AS(bayesnet::FCBF(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1e-8), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::FCBF(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1e-8), "Threshold cannot be less than 1e-7");
    REQUIRE_THROWS_AS(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -1e4), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -1e4), "Threshold has to be in [0, 0.5]");
    REQUIRE_THROWS_AS(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 0.501), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 0.501), "Threshold has to be in [0, 0.5]");
    // Not fitted error
    auto selector = build_selector(raw, "CFS", 0);
    const std::string message = "FeatureSelect not fitted";
    REQUIRE_THROWS_AS(selector->getFeatures(), std::runtime_error);
    REQUIRE_THROWS_AS(selector->getScores(), std::runtime_error);
    REQUIRE_THROWS_WITH(selector->getFeatures(), message);
    REQUIRE_THROWS_WITH(selector->getScores(), message);
    delete selector;
}
TEST_CASE("Test threshold limits", "[FeatureSelection]")
{
    auto raw = RawDatasets("diabetes", true);
    // FCBF Limits
    auto selector = build_selector(raw, "FCBF", 0.051);
    selector->fit();
    REQUIRE(selector->getFeatures().size() == 2);
    delete selector;
    selector = build_selector(raw, "FCBF", 1e-7, 3);
    selector->fit();
    REQUIRE(selector->getFeatures().size() == 3);
    delete selector;
    selector = build_selector(raw, "IWSS", 0.5, 5);
    selector->fit();
    REQUIRE(selector->getFeatures().size() == 5);
    delete selector;
}