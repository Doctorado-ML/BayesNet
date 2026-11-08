// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// BayesNet#50: the "adaboost" hyperparameter of the boosted ensembles.
//   * "M1" (default): AdaBoost.M1, a model is rejected when epsilon_t >= 0.5.
//   * "SAMME" (Zhu, Zou, Rosset & Hastie 2009): rejected when
//     epsilon_t >= 1 - 1/K, and alpha_t = 1/2 (ln((1 - epsilon_t) / epsilon_t) + ln(K - 1)).
// With two classes both are the same algorithm.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "TestUtils.h"
#include "bayesnet/ensembles/BoostAODE.h"
#include "bayesnet/ensembles/BoostA2DE.h"
#include "bayesnet/ensembles/XBAODE.h"
#include "bayesnet/ensembles/XBA2DE.h"

namespace {
    template <class Base>
    class SammeProbe : public Base {
    public:
        using Base::Base;
        const std::vector<double>& significances() const { return this->significanceModels; }
        unsigned nModels() const { return this->n_models; }
        void setClasses(int k)
        {
            this->className = "class";
            this->states[this->className] = std::vector<int>(k);
        }
        std::tuple<torch::Tensor&, double, bool> updateWeights(torch::Tensor& y, torch::Tensor& ypred, torch::Tensor& weights)
        {
            return this->update_weights(y, ypred, weights);
        }
    };
    // Ten instances, one per class; the first `wrong` are misclassified.
    struct Round {
        torch::Tensor y, ypred, weights;
        explicit Round(int wrong)
        {
            y = torch::arange(10, torch::kInt32);
            ypred = y.clone();
            for (int i = 0; i < wrong; ++i) ypred[i] = (i + 1) % 10;
            weights = torch::full({ 10 }, 0.1, torch::kFloat64);
        }
    };
}

TEMPLATE_TEST_CASE("adaboost hyperparameter", "[Boost50][Ensemble]",
    bayesnet::BoostAODE, bayesnet::BoostA2DE, bayesnet::XBAODE, bayesnet::XBA2DE)
{
    TestType clf;
    REQUIRE_NOTHROW(clf.setHyperparameters({ { "adaboost", "M1" } }));
    REQUIRE_NOTHROW(clf.setHyperparameters({ { "adaboost", "SAMME" } }));
    REQUIRE_THROWS_AS(clf.setHyperparameters({ { "adaboost", "SAMME.R" } }), std::invalid_argument);
}

TEST_CASE("SAMME threshold and amount of say", "[Boost50][Ensemble]")
{
    const int K = 10;
    SECTION("epsilon_t = 0.6 with ten classes: M1 rejects, SAMME accepts")
    {
        SammeProbe<bayesnet::XBAODE> m1;
        m1.setClasses(K);
        Round r1(6);
        auto [w1, alpha1, terminate1] = m1.updateWeights(r1.y, r1.ypred, r1.weights);
        REQUIRE(terminate1);
        REQUIRE(alpha1 == 0.0);

        SammeProbe<bayesnet::XBAODE> samme;
        samme.setHyperparameters({ { "adaboost", "SAMME" } });
        samme.setClasses(K);
        Round r2(6);
        auto [w2, alpha2, terminate2] = samme.updateWeights(r2.y, r2.ypred, r2.weights);
        REQUIRE_FALSE(terminate2);
        const double expected = 0.5 * (std::log(0.4 / 0.6) + std::log(K - 1.0));
        REQUIRE(alpha2 == Catch::Approx(expected).epsilon(1e-12));
        // Same reweighting as M1: a misclassified instance ends e^alpha times heavier.
        REQUIRE(w2.sum().item<double>() == Catch::Approx(1.0));
        REQUIRE(w2[0].item<double>() / w2[9].item<double>() == Catch::Approx(std::exp(alpha2)).epsilon(1e-12));
    }
    SECTION("SAMME rejects at 1 - 1/K and above")
    {
        SammeProbe<bayesnet::XBAODE> samme;
        samme.setHyperparameters({ { "adaboost", "SAMME" } });
        samme.setClasses(K);
        Round at(9); // epsilon_t = 0.9 = 1 - 1/10
        auto [w, alpha, terminate] = samme.updateWeights(at.y, at.ypred, at.weights);
        REQUIRE(terminate);
        REQUIRE(alpha == 0.0);
    }
    SECTION("with two classes SAMME is M1")
    {
        for (const std::string variant : { "M1", "SAMME" }) {
            SammeProbe<bayesnet::XBAODE> clf;
            clf.setHyperparameters({ { "adaboost", variant } });
            clf.setClasses(2);
            Round low(3), half(5);
            auto [w, alpha, terminate] = clf.updateWeights(low.y, low.ypred, low.weights);
            INFO(variant);
            REQUIRE_FALSE(terminate);
            REQUIRE(alpha == Catch::Approx(0.5 * std::log(0.7 / 0.3)).epsilon(1e-12));
            auto [w2, alpha2, terminate2] = clf.updateWeights(half.y, half.ypred, half.weights);
            REQUIRE(terminate2);
        }
    }
}

TEMPLATE_TEST_CASE("SAMME on a binary dataset gives the M1 model", "[Boost50][Ensemble]",
    bayesnet::BoostAODE, bayesnet::BoostA2DE, bayesnet::XBAODE, bayesnet::XBA2DE)
{
    auto raw = RawDatasets("diabetes", true);
    SammeProbe<TestType> m1, samme;
    samme.setHyperparameters({ { "adaboost", "SAMME" } });
    m1.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    samme.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(samme.nModels() == m1.nModels());
    REQUIRE(samme.significances() == m1.significances());
    REQUIRE(samme.predict_proba(raw.Xv) == m1.predict_proba(raw.Xv));
}

TEMPLATE_TEST_CASE("SAMME accepts the first model where M1 rejects it", "[Boost50][Ensemble]",
    bayesnet::XBAODE, bayesnet::XBA2DE)
{
    // glass has six classes. Every significance is positive in both variants, but
    // SAMME adds ln(5)/2 to each model's amount of say and does not reject models
    // with 0.5 <= epsilon_t < 5/6, so it never reports rejections here.
    auto raw = RawDatasets("glass", true);
    SammeProbe<TestType> samme;
    samme.setHyperparameters({ { "adaboost", "SAMME" } });
    samme.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(samme.nModels() > 0);
    for (double s : samme.significances()) {
        REQUIRE(s > 0.0);
    }
    for (const auto& note : samme.getNotes()) {
        INFO(note);
        REQUIRE(note.find("rejected") == std::string::npos);
    }
}
