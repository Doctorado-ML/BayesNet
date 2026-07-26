// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/XA2DE.h"
#include "bayesnet/classifiers/XSP2DE.h"
#include "TestUtils.h"

TEST_CASE("Fit and Score", "[XA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 360);   // C(9,2)=36 pairs * (9+1) nodes
    REQUIRE(clf.getNumberOfEdges() == 864);   // 36 pairs * (3*9-4+1) edges (joint superparents)
    REQUIRE(clf.getClassNumStates() == 6);
    REQUIRE(clf.getVersion() == "1.0.0");
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.827103).epsilon(raw.epsilon));
}

// The flat engine's single-pair posterior must equal the trusted XSp2de exactly
// (same joint-parents + CESTNIK + log-space math, no averaging involved).
TEST_CASE("Single pair equals XSp2de", "[XA2DE]")
{
    auto raw = RawDatasets("iris", true);
    std::vector<std::vector<int>> X2 = { raw.Xv[0], raw.Xv[1] };
    std::vector<std::string> f2 = { raw.features[0], raw.features[1] };

    auto clf = bayesnet::XA2DE();
    clf.fit(X2, raw.yv, f2, raw.className, raw.states, raw.smoothing);
    auto xa = clf.predict_proba(X2);

    auto sp = bayesnet::XSp2de(0, 1);
    sp.fit(X2, raw.yv, f2, raw.className, raw.states, raw.smoothing);

    int nSamples = raw.yv.size();
    int mism = 0;
    for (int s = 0; s < nSamples; ++s) {
        std::vector<int> inst = { X2[0][s], X2[1][s] };
        auto p = sp.predict_proba(inst);
        for (int c = 0; c < (int)p.size(); ++c) {
            if (std::abs(p[c] - xa[s][c]) > 1e-9) { mism++; break; }
        }
    }
    REQUIRE(mism == 0);
}

// A2DE is the average of the C(n,2) SP2DE submodel posteriors. Build that average
// explicitly from XSp2de and require XA2DE to reproduce the predictions exactly.
TEST_CASE("Matches averaged XSp2de over all pairs", "[XA2DE]")
{
    auto raw = RawDatasets("iris", true);
    int n = static_cast<int>(raw.features.size());
    int nSamples = static_cast<int>(raw.yv.size());

    auto clf = bayesnet::XA2DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto xa2de_pred = clf.predict(raw.Xv);

    std::vector<std::unique_ptr<bayesnet::XSp2de>> pairs;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto sp = std::make_unique<bayesnet::XSp2de>(i, j);
            sp->fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
            pairs.push_back(std::move(sp));
        }
    }
    int nClasses = pairs[0]->getClassNumStates();
    std::vector<int> oracle_pred(nSamples, 0);
    for (int s = 0; s < nSamples; ++s) {
        std::vector<int> instance(n);
        for (int f = 0; f < n; ++f) instance[f] = raw.Xv[f][s];
        std::vector<double> acc(nClasses, 0.0);
        for (auto& sp : pairs) {
            auto p = sp->predict_proba(instance);
            for (int c = 0; c < nClasses; ++c) acc[c] += p[c];
        }
        oracle_pred[s] = std::distance(acc.begin(), std::max_element(acc.begin(), acc.end()));
    }
    REQUIRE(xa2de_pred == oracle_pred);
}

TEST_CASE("CESTNIK smoothing runs and differs from ORIGINAL", "[XA2DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf1 = bayesnet::XA2DE();
    clf1.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::ORIGINAL);
    auto clf2 = bayesnet::XA2DE();
    clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::CESTNIK);
    REQUIRE(clf1.score(raw.Xv, raw.yv) > 0.5);
    REQUIRE(clf2.score(raw.Xv, raw.yv) > 0.5);
    REQUIRE(clf1.score(raw.Xv, raw.yv) != clf2.score(raw.Xv, raw.yv));
}
