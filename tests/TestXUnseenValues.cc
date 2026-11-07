// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// Regression tests for BayesNet#47: the X family (Xaode, Xaode2de, XSpode, XSp2de
// and the XA1DE, XA2DE, XBAODE, XBA2DE classifiers built on them) sized their
// probability tables with the largest training value + 1 of each feature instead
// of the `states` map handed to fit. A value present only at prediction time, with
// a code above every training value, was then read out of range (a segfault in
// b_grid, silent garbage in b_main).
//
// The tests rely on an invariance that only holds when the tables are sized with
// the states map: relabelling the codes of a feature (a permutation of its values,
// the map unchanged) cannot change the model. Dataset A leaves the top code of
// feature 0 out of the training data and uses it at prediction time; dataset B is
// the same data with the codes of feature 0 rotated, so the unseen value gets code
// 0, which the old sizing (max + 1) already covered. Both must give the same
// posteriors and the same number of cells.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <stdexcept>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "bayesnet/ensembles/Xaode.hpp"
#include "bayesnet/ensembles/Xaode2de.hpp"
#include "bayesnet/ensembles/XA1DE.h"
#include "bayesnet/ensembles/XA2DE.h"
#include "bayesnet/ensembles/XBAODE.h"
#include "bayesnet/ensembles/XBA2DE.h"
#include "bayesnet/classifiers/XSPODE.h"
#include "bayesnet/classifiers/XSP2DE.h"

namespace {
    constexpr int kTrain = 120;
    const std::vector<int> kTrainCard = { 2, 3, 2, 3, 2 }; // values used in training
    const std::vector<int> kMapCard = { 3, 3, 2, 3, 2 };   // states map: feature 0 has one more

    struct Data {
        std::vector<std::vector<int>> X;     // features x samples (training)
        std::vector<int> y;
        std::vector<std::vector<int>> Xtest; // features x samples (prediction)
        std::vector<std::string> features{ "f0", "f1", "f2", "f3", "f4" };
        std::string className{ "class" };
        std::map<std::string, std::vector<int>> states;
    };
    // rotate = false: feature 0 trains on {0,1}, the unseen value is 2 (top code).
    // rotate = true: codes of feature 0 rotated (v -> (v+1) % 3), so it trains on
    // {1,2} and the unseen value is 0.
    Data makeData(bool rotate)
    {
        Data d;
        int nf = static_cast<int>(kTrainCard.size());
        d.X.assign(nf, std::vector<int>(kTrain));
        d.y.resize(kTrain);
        for (int i = 0; i < kTrain; ++i) {
            int c = (i % 3 == 0) ? 1 : 0;
            d.y[i] = c;
            for (int f = 0; f < nf; ++f) {
                // class related with some noise, deterministic
                d.X[f][i] = (c * (f + 1) + i / (f + 2) + (i * (f + 3)) % 5) % kTrainCard[f];
            }
        }
        // prediction instances: every one carries the unseen value of feature 0
        // and the remaining features sweep their values
        d.Xtest.assign(nf, std::vector<int>());
        for (int j = 0; j < 6; ++j) {
            d.Xtest[0].push_back(2);
            for (int f = 1; f < nf; ++f) d.Xtest[f].push_back((j + f) % kTrainCard[f]);
        }
        // and two with seen values, which must be unaffected
        for (int j = 0; j < 2; ++j) {
            d.Xtest[0].push_back(j);
            for (int f = 1; f < nf; ++f) d.Xtest[f].push_back((j * 2 + f) % kTrainCard[f]);
        }
        if (rotate) {
            for (auto& v : d.X[0]) v = (v + 1) % 3;
            for (auto& v : d.Xtest[0]) v = (v + 1) % 3;
        }
        for (int f = 0; f < nf; ++f) {
            d.states[d.features[f]] = std::vector<int>(kMapCard[f]);
            for (int s = 0; s < kMapCard[f]; ++s) d.states[d.features[f]][s] = s;
        }
        d.states[d.className] = { 0, 1 };
        return d;
    }
    std::vector<int> instance(const std::vector<std::vector<int>>& X, int j)
    {
        std::vector<int> v;
        for (const auto& column : X) v.push_back(column[j]);
        return v;
    }
    torch::Tensor uniformWeights()
    {
        return torch::full({ kTrain }, 1.0 / kTrain, torch::kFloat64);
    }
    void requireSamePosteriors(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
    {
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            REQUIRE(a[i].size() == b[i].size());
            double sum = 0.0;
            for (size_t c = 0; c < a[i].size(); ++c) {
                INFO("instance " << i << " class " << c);
                REQUIRE(std::isfinite(a[i][c]));
                REQUIRE(a[i][c] == Catch::Approx(b[i][c]).margin(1e-12));
                sum += a[i][c];
            }
            REQUIRE(sum == Catch::Approx(1.0).margin(1e-9));
        }
    }
    template <typename Model>
    std::vector<std::vector<double>> posteriors(Model& model, const std::vector<std::vector<int>>& Xtest)
    {
        std::vector<std::vector<double>> out;
        for (size_t j = 0; j < Xtest[0].size(); ++j) out.push_back(model.predict_proba(instance(Xtest, static_cast<int>(j))));
        return out;
    }
}

TEST_CASE("Xaode: a value unseen in training has its own cell", "[XA1DE]")
{
    auto a = makeData(false), b = makeData(true);
    bayesnet::Xaode ma, mb;
    ma.fit(a.X, a.y, a.features, a.className, a.states, uniformWeights(), true, bayesnet::Smoothing_t::ORIGINAL);
    mb.fit(b.X, b.y, b.features, b.className, b.states, uniformWeights(), true, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
    requireSamePosteriors(posteriors(ma, a.Xtest), posteriors(mb, b.Xtest));
    // a value outside the states map is rejected instead of read out of range
    auto bad = instance(a.Xtest, 0);
    bad[0] = 3;
    REQUIRE_THROWS_AS(ma.predict_proba(bad), std::out_of_range);
    REQUIRE_THROWS_WITH(ma.predict_proba(bad), Catch::Matchers::ContainsSubstring("'f0'"));
}

TEST_CASE("Xaode2de: a value unseen in training has its own cell", "[XA2DE]")
{
    auto a = makeData(false), b = makeData(true);
    bayesnet::Xaode2de ma, mb;
    ma.fit(a.X, a.y, a.features, a.className, a.states, uniformWeights(), true, bayesnet::Smoothing_t::ORIGINAL);
    mb.fit(b.X, b.y, b.features, b.className, b.states, uniformWeights(), true, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
    requireSamePosteriors(posteriors(ma, a.Xtest), posteriors(mb, b.Xtest));
    auto bad = instance(a.Xtest, 0);
    bad[0] = 3;
    REQUIRE_THROWS_AS(ma.predict_proba(bad), std::out_of_range);
}

TEST_CASE("XSpode: a value unseen in training has its own cell", "[XSPODE]")
{
    auto a = makeData(false), b = makeData(true);
    // superparent 0 (unseen value in the superparent) and 1 (unseen value in a child)
    for (int sp : { 0, 1 }) {
        INFO("superparent " << sp);
        bayesnet::XSpode ma(sp), mb(sp);
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        REQUIRE(ma.getNumberOfStates() == bayesnet::XSpode::cellsOfSpode(kMapCard, 2, sp));
        requireSamePosteriors(posteriors(ma, a.Xtest), posteriors(mb, b.Xtest));
        auto bad = instance(a.Xtest, 0);
        bad[0] = 3;
        REQUIRE_THROWS_AS(ma.predict_proba(bad), std::out_of_range);
    }
}

TEST_CASE("XSp2de: a value unseen in training has its own cell", "[XSP2DE]")
{
    auto a = makeData(false), b = makeData(true);
    for (auto [p, q] : std::vector<std::pair<int, int>>{ { 0, 1 }, { 1, 2 } }) {
        INFO("superparents " << p << ", " << q);
        bayesnet::XSp2de ma(p, q), mb(p, q);
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        REQUIRE(ma.getNumberOfStates() == bayesnet::XSp2de::cellsOfSp2de(kMapCard, 2, p, q));
        requireSamePosteriors(posteriors(ma, a.Xtest), posteriors(mb, b.Xtest));
        auto bad = instance(a.Xtest, 0);
        bad[0] = 3;
        REQUIRE_THROWS_AS(ma.predict_proba(bad), std::out_of_range);
    }
}

TEST_CASE("XA1DE and XA2DE: a value unseen in training has its own cell", "[XA1DE][XA2DE]")
{
    auto a = makeData(false), b = makeData(true);
    {
        bayesnet::XA1DE ma, mb;
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        requireSamePosteriors(ma.predict_proba(a.Xtest), mb.predict_proba(b.Xtest));
    }
    {
        bayesnet::XA2DE ma, mb;
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        requireSamePosteriors(ma.predict_proba(a.Xtest), mb.predict_proba(b.Xtest));
    }
}

TEST_CASE("XBAODE and XBA2DE: a value unseen in training has its own cell", "[XBAODE][XBA2DE]")
{
    // Boost trains every model on 80 % of the data (the rest validates), so the
    // unseen value also reaches the validation predictions.
    auto a = makeData(false), b = makeData(true);
    {
        bayesnet::XBAODE ma, mb;
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfNodes() == mb.getNumberOfNodes());
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        requireSamePosteriors(ma.predict_proba(a.Xtest), mb.predict_proba(b.Xtest));
    }
    {
        bayesnet::XBA2DE ma, mb;
        ma.fit(a.X, a.y, a.features, a.className, a.states, bayesnet::Smoothing_t::ORIGINAL);
        mb.fit(b.X, b.y, b.features, b.className, b.states, bayesnet::Smoothing_t::ORIGINAL);
        REQUIRE(ma.getNumberOfNodes() == mb.getNumberOfNodes());
        REQUIRE(ma.getNumberOfStates() == mb.getNumberOfStates());
        requireSamePosteriors(ma.predict_proba(a.Xtest), mb.predict_proba(b.Xtest));
    }
}
