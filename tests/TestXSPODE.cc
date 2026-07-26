// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <stdexcept>
#include "bayesnet/classifiers/XSPODE.h"
#include "TestUtils.h"

TEST_CASE("fit vector test", "[XSPODE]") {
  auto raw = RawDatasets("iris", true);
  auto scores = std::vector<float>({0.966667, 0.9333333, 0.966667, 0.966667});
  for (int i = 0; i < 4; ++i) {
    auto clf = bayesnet::XSpode(i);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states,
            raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 9);
    REQUIRE(clf.getNotes().size() == 0);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(scores.at(i)));
  }
}
TEST_CASE("fit dataset test", "[XSPODE]") {
  auto raw = RawDatasets("iris", true);
  auto scores = std::vector<float>({0.966667, 0.9333333, 0.966667, 0.966667});
  for (int i = 0; i < 4; ++i) {
    auto clf = bayesnet::XSpode(i);
    clf.fit(raw.dataset, raw.features, raw.className, raw.states,
            raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 9);
    REQUIRE(clf.getNotes().size() == 0);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(scores.at(i)));
  }
}
TEST_CASE("tensors dataset predict & predict_proba", "[XSPODE]") {
  auto raw = RawDatasets("iris", true);
  auto scores = std::vector<float>({0.966667, 0.9333333, 0.966667, 0.966667});
  auto probs_expected = std::vector<std::vector<float>>({ 
      {0.999017, 0.000306908, 0.000676449}, 
      {0.99831, 0.00119304, 0.000497099}, 
      {0.998432, 0.00078416, 0.00078416}, 
      {0.998801, 0.000599438, 0.000599438}
  });
  for (int i = 0; i < 4; ++i) {
    auto clf = bayesnet::XSpode(i);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states,
            raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 9);
    REQUIRE(clf.getNotes().size() == 0);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(scores.at(i)));
    // Get the first 4 lines of X_test to do predict_proba
    auto X_reduced = raw.X_test.slice(1, 0, 4);
    auto proba = clf.predict_proba(X_reduced);
    for (int p = 0; p < 3; ++p) {
      REQUIRE(proba[0][p].item<double>() == Catch::Approx(probs_expected.at(i).at(p)));
    }
  }
}

TEST_CASE("mfeat-factors dataset test", "[XSPODE]") {
  auto raw = RawDatasets("mfeat-factors", true);
  auto scores = std::vector<float>({0.9825, 0.9775, 0.9775, 0.99});
  for (int i = 0; i < 4; ++i) {
    auto clf = bayesnet::XSpode(i);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 217);
    REQUIRE(clf.getNumberOfEdges() == 433);
    REQUIRE(clf.getNotes().size() == 0);
    REQUIRE(clf.getNumberOfStates() == 652320);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(scores.at(i)));
  }
}
TEST_CASE("Laplace predict", "[XSPODE]") {
  auto raw = RawDatasets("iris", true);
  auto scores = std::vector<float>({0.966666639, 1.0f, 0.933333337, 1.0f});
  for (int i = 0; i < 4; ++i) {
    auto clf = bayesnet::XSpode(0);
    clf.setHyperparameters({ {"parent", i} });
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::LAPLACE);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 9);
    REQUIRE(clf.getNotes().size() == 0);
    REQUIRE(clf.getNumberOfStates() == 64);
    REQUIRE(clf.getNFeatures() == 4);
    REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(scores.at(i)));
  }
}
TEST_CASE("Not fitted model predict", "[XSPODE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::XSpode(0);
    REQUIRE_THROWS_AS(clf.predict(std::vector<int>({1,2,3})), std::logic_error);
}
TEST_CASE("Test instance predict", "[XSPODE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::XSpode(0);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::ORIGINAL);
    REQUIRE(clf.predict(std::vector<int>({1,2,3,4})) == 1);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.973333359f));
    // CESTNIK is now a real m-estimate (m=1, uniform prior), not a silent no-op:
    // it smooths like the other strategies, so this instance is classified as 1
    // (as ORIGINAL), unlike the old unsmoothed behaviour that predicted 0.
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::CESTNIK);
    REQUIRE(clf.predict(std::vector<int>({1,2,3,4})) == 1);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.966666639f));
}
TEST_CASE("Test to_string and fitx", "[XSPODE]")
{
  auto raw = RawDatasets("iris", true);
  auto clf = bayesnet::XSpode(0);
  auto weights = torch::full({raw.Xt.size(1)}, 1.0 / raw.Xt.size(1), torch::kFloat64);
  clf.fitx(raw.Xt, raw.yt, weights, bayesnet::Smoothing_t::ORIGINAL);
  REQUIRE(clf.getNumberOfNodes() == 5);
  REQUIRE(clf.getNumberOfEdges() == 9);
  REQUIRE(clf.getNotes().size() == 0);
  REQUIRE(clf.getNumberOfStates() == 64);
  REQUIRE(clf.getNFeatures() == 4);
  REQUIRE(clf.score(raw.X_test, raw.y_test) == Catch::Approx(0.966666639f));
  REQUIRE(clf.to_string().size() == 1966);
  REQUIRE(clf.graph("Not yet implemented") == std::vector<std::string>({"Not yet implemented"}));
}
