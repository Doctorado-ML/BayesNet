// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// Regression tests for BayesNet#48: the four boosted ensembles (BoostAODE,
// BoostA2DE, XBAODE, XBA2DE) added the model that update_weights rejects
// (epsilon_t > 0.5) to the ensemble with significance 0.
//   * When it was the first model, every significance was 0 and the averaged
//     prediction divided 0 by 0: predict_proba returned NaN and predict always
//     class 0, with no error.
//   * Otherwise the model stayed in the ensemble with no say in the prediction
//     but counted in the number of models, nodes and cells.
// Now the first rejected model is kept with significance 1 (with a single model
// any positive significance gives the same predictions), any later one is not
// added, and an ensemble whose significances sum to 0 throws instead of
// returning NaN.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "TestUtils.h"
#include "bayesnet/ensembles/BoostAODE.h"
#include "bayesnet/ensembles/BoostA2DE.h"
#include "bayesnet/ensembles/XBAODE.h"
#include "bayesnet/ensembles/XBA2DE.h"

namespace {
    const std::string kFirstKept = "First model rejected (epsilon_t > 0.5): kept with significance 1 as the only model";
    const std::string kFirstBlockKept = "First models rejected (epsilon_t > 0.5): kept with significance 1";
    std::string rejectedNote(int n) { return "Models rejected (epsilon_t > 0.5) and not added: " + std::to_string(n); }

    // Exposes the protected internals the checks need.
    template <class Base>
    class Probe : public Base {
    public:
        using Base::Base;
        const std::vector<double>& significances() const { return this->significanceModels; }
        unsigned nModels() const { return this->n_models; }
        std::vector<int> predictWithModel(int i, std::vector<std::vector<int>>& X) { return this->models.at(i)->predict(X); }
        void zeroSignificances() { std::fill(this->significanceModels.begin(), this->significanceModels.end(), 0.0); }
        void dropModels() { this->models.clear(); this->significanceModels.clear(); this->n_models = 0; }
    };

    // Ten classes at random and six features at random with four values: no model
    // gets below epsilon_t = 0.5, so the very first one is rejected. This is the
    // shape of volkert (10 classes) in the BoostA2DE experiments, where the first
    // SPODE had epsilon_t = 0.51.
    struct Random10 {
        std::vector<std::vector<int>> X;
        std::vector<int> y;
        std::vector<std::string> features;
        std::string className{ "class" };
        std::map<std::string, std::vector<int>> states;
        Random10()
        {
            std::mt19937 g(1);
            const int n = 6, m = 3000;
            X.assign(n, std::vector<int>(m));
            y.resize(m);
            for (auto& column : X)
                for (auto& v : column) v = static_cast<int>(g() % 4);
            for (auto& v : y) v = static_cast<int>(g() % 10);
            for (int i = 0; i < n; ++i) {
                features.push_back("f" + std::to_string(i));
                states[features.back()] = { 0, 1, 2, 3 };
            }
            states[className] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        }
    };

    bool contains(const std::vector<std::string>& notes, const std::string& note)
    {
        return std::find(notes.begin(), notes.end(), note) != notes.end();
    }

    template <class Clf>
    void requireProperProbabilities(Clf& clf, std::vector<std::vector<int>>& X)
    {
        auto proba = clf.predict_proba(X);
        for (const auto& row : proba) {
            double sum = 0.0;
            for (double p : row) {
                REQUIRE(std::isfinite(p));
                sum += p;
            }
            REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("A rejected first model is kept with significance 1", "[Boost48][Ensemble]",
    bayesnet::BoostAODE, bayesnet::BoostA2DE, bayesnet::XBAODE, bayesnet::XBA2DE)
{
    Random10 data;
    Probe<TestType> clf;
    clf.fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(clf.nModels() == 1);
    REQUIRE(clf.significances() == std::vector<double>{ 1.0 });
    REQUIRE(contains(clf.getNotes(), kFirstKept));
    requireProperProbabilities(clf, data.X);
    // The ensemble predicts exactly what its only model predicts (1.4.2 returned
    // class 0 for every instance).
    REQUIRE(clf.predict(data.X) == clf.predictWithModel(0, data.X));
}

TEMPLATE_TEST_CASE("A rejected first block is kept with significance 1", "[Boost48][Ensemble]",
    bayesnet::BoostAODE, bayesnet::BoostA2DE)
{
    // block_update weighs a whole pack at once (update_weights_block); its first
    // pack is a single model.
    Random10 data;
    Probe<TestType> clf;
    clf.setHyperparameters({ { "block_update", true } });
    clf.fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(clf.nModels() == 1);
    REQUIRE(clf.significances() == std::vector<double>{ 1.0 });
    REQUIRE(contains(clf.getNotes(), kFirstBlockKept));
    requireProperProbabilities(clf, data.X);
}

TEMPLATE_TEST_CASE("A rejected seeding ensemble is kept with significance 1", "[Boost48][Ensemble]",
    bayesnet::BoostAODE, bayesnet::XBAODE)
{
    // select_features seeds the ensemble with the selected SPODEs and weighs them
    // together; when that seed is rejected it is the whole ensemble.
    Random10 data;
    Probe<TestType> clf;
    clf.setHyperparameters({ { "select_features", "IWSS" }, { "threshold", 0.5 } });
    clf.fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(clf.nModels() >= 1);
    REQUIRE(contains(clf.getNotes(), kFirstBlockKept));
    for (double s : clf.significances()) {
        REQUIRE(s == 1.0);
    }
    requireProperProbabilities(clf, data.X);
}

namespace {
    // Reference values computed with 1.4.2 on the training data. The rejected
    // models had significance 0, so dropping them must leave the probabilities
    // untouched: only the number of models and nodes go down.
    struct Golden {
        int okBefore;                 // correct predictions on the training data (1.4.2)
        std::vector<double> colSums;  // sum over instances of each class probability (1.4.2)
        unsigned modelsBefore;
        int rejected;
        int nodesAfter;
    };
    template <class Clf>
    void checkGolden(const std::string& dataset, const nlohmann::json& hyper, const Golden& golden)
    {
        auto raw = RawDatasets(dataset, true);
        Probe<Clf> clf;
        clf.setHyperparameters(hyper);
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        INFO(dataset << " " << hyper.dump());
        REQUIRE(clf.nModels() == golden.modelsBefore - golden.rejected);
        for (double s : clf.significances()) {
            REQUIRE(s > 0.0);
        }
        REQUIRE(contains(clf.getNotes(), rejectedNote(golden.rejected)));
        REQUIRE(clf.getNumberOfNodes() == golden.nodesAfter);
        auto proba = clf.predict_proba(raw.Xv);
        std::vector<double> cols(proba[0].size(), 0.0);
        for (const auto& row : proba)
            for (size_t c = 0; c < row.size(); ++c) cols[c] += row[c];
        REQUIRE(cols.size() == golden.colSums.size());
        for (size_t c = 0; c < cols.size(); ++c) {
            REQUIRE(cols[c] == Catch::Approx(golden.colSums[c]).epsilon(1e-9));
        }
        auto pred = clf.predict(raw.Xv);
        int ok = 0;
        for (size_t i = 0; i < pred.size(); ++i) ok += pred[i] == raw.yv[i];
        REQUIRE(ok == golden.okBefore);
    }
}

TEST_CASE("Rejected models after the first are not added", "[Boost48][Ensemble]")
{
    SECTION("XBA2DE glass: seven rejections, six of them inside a bisection pack")
    {
        // With bisection the pack goes on after a rejection, and a later model of
        // the pack resets the stop flag; 1.4.2 kept all seven with significance 0.
        checkGolden<bayesnet::XBA2DE>("glass", nlohmann::json::object(),
            { 181, { 65.120701445070424, 27.103271779878163, 7.818455577327315, 75.267754519309463, 28.960982454876305, 9.7288342235383158 }, 25, 7, 180 });
    }
    SECTION("XBA2DE ecoli without convergence")
    {
        checkGolden<bayesnet::XBA2DE>("ecoli", { { "convergence", false } },
            { 302, { 131.82775298786007, 75.03721159943828, 3.2156535817646259, 1.5287784010097059, 39.164206351359809, 20.118833054206597, 3.7141696232453114, 61.393394401114939 }, 12, 1, 88 });
    }
    SECTION("BoostA2DE ecoli: rejection in the middle of a pack")
    {
        checkGolden<bayesnet::BoostA2DE>("ecoli", nlohmann::json::object(),
            { 300, { 122.25759559933158, 71.844801972769375, 3.3540935814704356, 1.9775940214776058, 41.894960010731083, 16.676454520278085, 3.4221862685132836, 74.572314025428327 }, 21, 1, 160 });
    }
    SECTION("BoostA2DE glass without convergence")
    {
        checkGolden<bayesnet::BoostA2DE>("glass", { { "convergence", false } },
            { 177, { 68.933312472193776, 22.044501530839121, 6.6986764448423859, 79.509360911849242, 27.390421083405165, 9.4237275568703378 }, 10, 1, 90 });
    }
    SECTION("BoostAODE glass")
    {
        checkGolden<bayesnet::BoostAODE>("glass", nlohmann::json::object(),
            { 177, { 66.400982386726028, 25.467030163663225, 9.0231730864023696, 71.38674420405367, 29.589445005525608, 12.132625153629082 }, 9, 1, 80 });
    }
    SECTION("XBAODE glass")
    {
        checkGolden<bayesnet::XBAODE>("glass", nlohmann::json::object(),
            { 177, { 66.400982386726028, 25.467030163663221, 9.0231730864023678, 71.38674420405367, 29.589445005525604, 12.132625153629082 }, 9, 1, 80 });
    }
}

TEST_CASE("An ensemble whose significances sum to 0 throws", "[Boost48][Ensemble]")
{
    auto raw = RawDatasets("iris", true);
    SECTION("average of probabilities")
    {
        Probe<bayesnet::XBAODE> clf;
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        clf.zeroSignificances();
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xv), std::logic_error);
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xt), std::logic_error);
        REQUIRE_THROWS_AS(clf.predict(raw.Xv), std::logic_error);
    }
    SECTION("voting")
    {
        Probe<bayesnet::BoostAODE> clf(true);
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        clf.zeroSignificances();
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xv), std::logic_error);
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xt), std::logic_error);
    }
    SECTION("no models at all")
    {
        Probe<bayesnet::XBAODE> clf;
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        clf.dropModels();
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xv), std::logic_error);
        REQUIRE_THROWS_AS(clf.predict_proba(raw.Xt), std::logic_error);
    }
}
