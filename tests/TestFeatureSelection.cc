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
#include "bayesnet/feature_selection/L1FS.h"
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
    } else if (selector == "L1FS") {
        // For L1FS, threshold is used as alpha parameter
        return new bayesnet::L1FS(raw.dataset, raw.features, raw.className, max_features, raw.classNumStates, raw.weights, threshold);
    }
    return nullptr;
}

TEST_CASE("Features Selected", "[FeatureSelection]")
{
    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");

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
            { {"diabetes", "FCBF"}, {{ 1, 5, 7, 6 }, {0.132858, 0.083191, 0.0480135, 0.0224186} }},
            { {"glass", "L1FS" }, { { 2, 3, 5}, { 0.365513, 0.304911, 0.302109 } } },
            { {"iris", "L1FS"}, {{ 3, 2, 1, 0 }, { 0.570928, 0.37569, 0.0774792, 0.00835904 }}},
            { {"ecoli", "L1FS"}, {{ 0, 1, 6, 5, 2, 3 }, {0.490179, 0.365944, 0.291177, 0.199171, 0.0400928, 0.0192575} }},
            { {"diabetes", "L1FS"}, {{ 1, 5, 4 }, {0.132858, 0.083191, 0.0486187} }}
        };
        double threshold;
        std::string selector;
        std::vector<std::pair<std::string, double>> selectors = {
            { "CFS", 0.0 },
            { "IWSS", 0.1 },
            { "FCBF", 1e-7 },
            { "L1FS", 0.01 }
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
    SECTION("Test L1FS")
    {
        bayesnet::L1FS* featureSelector = new bayesnet::L1FS(
            raw.dataset, raw.features, raw.className,
            raw.features.size(), raw.classNumStates, raw.weights,
            0.01, 1000, 1e-4, true
        );
        featureSelector->fit();

        std::vector<int> selected_features = featureSelector->getFeatures();
        std::vector<double> selected_scores = featureSelector->getScores();

        // Check if features are selected
        REQUIRE(selected_features.size() > 0);
        REQUIRE(selected_scores.size() == selected_features.size());

        // Scores should be non-negative (absolute coefficient values)
        for (double score : selected_scores) {
            REQUIRE(score >= 0.0);
        }

        // Scores should be in descending order
        // std::cout << file_name << " " << selected_features << std::endl << "{";
        for (size_t i = 1; i < selected_scores.size(); i++) {
            // std::cout << selected_scores[i - 1] << ", ";
            REQUIRE(selected_scores[i - 1] >= selected_scores[i]);
        }
        // std::cout << selected_scores[selected_scores.size() - 1];
        // std::cout << "}" << std::endl;
        delete featureSelector;
    }
}

TEST_CASE("L1FS Features Selected", "[FeatureSelection]")
{
    auto raw = RawDatasets("ecoli", true);

    SECTION("Test L1FS with different alpha values")
    {
        std::vector<double> alphas = { 0.01, 0.1, 0.5 };

        for (double alpha : alphas) {
            bayesnet::L1FS* featureSelector = new bayesnet::L1FS(
                raw.dataset, raw.features, raw.className,
                raw.features.size(), raw.classNumStates, raw.weights,
                alpha, 1000, 1e-4, true
            );
            featureSelector->fit();

            INFO("Alpha: " << alpha);

            std::vector<int> selected_features = featureSelector->getFeatures();
            std::vector<double> selected_scores = featureSelector->getScores();

            // Higher alpha should lead to fewer features
            REQUIRE(selected_features.size() > 0);
            REQUIRE(selected_features.size() <= raw.features.size());
            REQUIRE(selected_scores.size() == selected_features.size());

            // Scores should be non-negative (absolute coefficient values)
            for (double score : selected_scores) {
                REQUIRE(score >= 0.0);
            }

            // Scores should be in descending order
            for (size_t i = 1; i < selected_scores.size(); i++) {
                REQUIRE(selected_scores[i - 1] >= selected_scores[i]);
            }

            delete featureSelector;
        }
    }

    SECTION("Test L1FS with max features limit")
    {
        int max_features = 2;
        bayesnet::L1FS* featureSelector = new bayesnet::L1FS(
            raw.dataset, raw.features, raw.className,
            max_features, raw.classNumStates, raw.weights,
            0.1, 1000, 1e-4, true
        );
        featureSelector->fit();

        std::vector<int> selected_features = featureSelector->getFeatures();
        REQUIRE(selected_features.size() <= max_features);

        delete featureSelector;
    }

    SECTION("Test L1FS getCoefficients method")
    {
        bayesnet::L1FS* featureSelector = new bayesnet::L1FS(
            raw.dataset, raw.features, raw.className,
            raw.features.size(), raw.classNumStates, raw.weights,
            0.1, 1000, 1e-4, true
        );

        // Should throw before fitting
        REQUIRE_THROWS_AS(featureSelector->getCoefficients(), std::runtime_error);
        REQUIRE_THROWS_WITH(featureSelector->getCoefficients(), "L1FS not fitted");

        featureSelector->fit();

        // Should work after fitting
        auto coefficients = featureSelector->getCoefficients();
        REQUIRE(coefficients.size() == raw.features.size());

        delete featureSelector;
    }
}

TEST_CASE("Oddities", "[FeatureSelection]")
{
    auto raw = RawDatasets("iris", true);

    // FCBF Limits
    REQUIRE_THROWS_AS(bayesnet::FCBF(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1e-8), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::FCBF(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1e-8), "Threshold cannot be less than 1e-7");

    // IWSS Limits
    REQUIRE_THROWS_AS(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -1e4), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -1e4), "Threshold has to be in [0, 0.5]");
    REQUIRE_THROWS_AS(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 0.501), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::IWSS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 0.501), "Threshold has to be in [0, 0.5]");

    // L1FS Limits
    REQUIRE_THROWS_AS(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -0.1), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, -0.1), "Alpha (regularization strength) must be non-negative");

    REQUIRE_THROWS_AS(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 0), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 0), "Maximum iterations must be positive");

    REQUIRE_THROWS_AS(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 1000, 0.0), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 1000, 0.0), "Tolerance must be positive");

    REQUIRE_THROWS_AS(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 1000, -1e-4), std::invalid_argument);
    REQUIRE_THROWS_WITH(bayesnet::L1FS(raw.dataset, raw.features, raw.className, raw.features.size(), raw.classNumStates, raw.weights, 1.0, 1000, -1e-4), "Tolerance must be positive");

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

    // L1FS with different alpha values
    selector = build_selector(raw, "L1FS", 0.01);  // Low alpha - more features
    selector->fit();
    int num_features_low_alpha = selector->getFeatures().size();
    delete selector;

    selector = build_selector(raw, "L1FS", 0.9);   // High alpha - fewer features
    selector->fit();
    int num_features_high_alpha = selector->getFeatures().size();
    REQUIRE(num_features_high_alpha <= num_features_low_alpha);
    delete selector;

    // L1FS with max features limit
    selector = build_selector(raw, "L1FS", 0.01, 4);
    selector->fit();
    REQUIRE(selector->getFeatures().size() <= 4);
    delete selector;
}

TEST_CASE("L1FS Regression vs Classification", "[FeatureSelection]")
{
    SECTION("Regression Task")
    {
        auto raw = RawDatasets("diabetes", true);
        // diabetes dataset should be treated as regression (classNumStates > 2)
        bayesnet::L1FS* l1fs = new bayesnet::L1FS(
            raw.dataset, raw.features, raw.className,
            raw.features.size(), raw.classNumStates, raw.weights,
            0.1, 1000, 1e-4, true
        );
        l1fs->fit();

        auto features = l1fs->getFeatures();
        REQUIRE(features.size() > 0);

        delete l1fs;
    }

    SECTION("Binary Classification Task")
    {
        // Create a simple binary classification dataset
        int n_samples = 100;
        int n_features = 5;

        torch::Tensor X = torch::randn({ n_features, n_samples });
        torch::Tensor y = (X[0] + X[2] > 0).to(torch::kFloat32);
        torch::Tensor samples = torch::cat({ X, y.unsqueeze(0) }, 0);

        std::vector<std::string> features;
        for (int i = 0; i < n_features; ++i) {
            features.push_back("feature_" + std::to_string(i));
        }

        torch::Tensor weights = torch::ones({ n_samples });

        bayesnet::L1FS* l1fs = new bayesnet::L1FS(
            samples, features, "target",
            n_features, 2, weights,  // 2 states = binary classification
            0.1, 1000, 1e-4, true
        );
        l1fs->fit();

        auto selected_features = l1fs->getFeatures();
        REQUIRE(selected_features.size() > 0);

        // Features 0 and 2 should be among the top selected
        bool has_feature_0 = std::find(selected_features.begin(), selected_features.end(), 0) != selected_features.end();
        bool has_feature_2 = std::find(selected_features.begin(), selected_features.end(), 2) != selected_features.end();
        REQUIRE((has_feature_0 || has_feature_2));

        delete l1fs;
    }
}