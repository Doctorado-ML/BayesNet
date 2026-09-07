// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "bayesnet/classifiers/XSP2DE.h"  // <-- your new 2-superparent classifier
#include "TestUtils.h"                   // for RawDatasets, etc.
#include <iostream>

// Helper function to handle each (sp1, sp2) pair in tests
static void check_spnde_pair(
    int sp1, 
    int sp2, 
    RawDatasets &raw, 
    bool fitVector, 
    bool fitTensor)
{
  // Create our classifier
  bayesnet::XSp2de clf(sp1, sp2);

  // Option A: fit with vector-based data
  if (fitVector) {
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  }
  // Option B: fit with the whole dataset in torch::Tensor form
  else if (fitTensor) {
    // your “tensor” version of fit
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
  }
  // Option C: or you might do the “dataset” version:
  else {
    clf.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
  }

  // Basic checks
  REQUIRE(clf.getNumberOfNodes() == 5);  // for iris: 4 features + 1 class
  REQUIRE(clf.getNumberOfEdges() == 9);  // joint superparents: sp1->sp2 edge (default joint_parents=true)
  REQUIRE(clf.getNotes().size() == 0);

  // Evaluate on test set
  float sc = clf.score(raw.X_test, raw.y_test);
  REQUIRE(sc >= 0.93f);  
}

// ------------------------------------------------------------
// 1) Fit vector test
// ------------------------------------------------------------
TEST_CASE("fit vector test (XSP2DE)", "[XSP2DE]") {
  auto raw = RawDatasets("iris", true);

  std::vector<std::pair<int,int>> parentPairs = {
    {0,1}, {2,3}
  };
  for (auto &p : parentPairs) {
    check_spnde_pair(p.first, p.second, raw, /*fitVector=*/true, /*fitTensor=*/false);
  }
}

// ------------------------------------------------------------
// 2) Fit dataset test
// ------------------------------------------------------------
TEST_CASE("fit dataset test (XSP2DE)", "[XSP2DE]") {
  auto raw = RawDatasets("iris", true);

  // Again test multiple pairs:
  std::vector<std::pair<int,int>> parentPairs = {
    {0,2}, {1,3}
  };
  for (auto &p : parentPairs) {
    check_spnde_pair(p.first, p.second, raw, /*fitVector=*/false, /*fitTensor=*/false);
  }
}

// ------------------------------------------------------------
// 3) Tensors dataset predict & predict_proba
// ------------------------------------------------------------
TEST_CASE("tensors dataset predict & predict_proba (XSP2DE)", "[XSP2DE]") {
  auto raw = RawDatasets("iris", true);

  std::vector<std::pair<int,int>> parentPairs = {
    {0,3}, {1,2}
  };

  for (auto &p : parentPairs) {
    bayesnet::XSp2de clf(p.first, p.second);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);

    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 9);  // joint superparents: sp1->sp2 edge (default joint_parents=true)
    REQUIRE(clf.getNotes().size() == 0);

    // Check the score
    float sc = clf.score(raw.X_test, raw.y_test);
    REQUIRE(sc >= 0.90f);

    auto X_reduced = raw.X_test.slice(1, 0, 3); 
    auto proba = clf.predict_proba(X_reduced);
  }
}
TEST_CASE("Check hyperparameters", "[XSP2DE]")
{
  auto raw = RawDatasets("iris", true);

  auto clf = bayesnet::XSp2de(0, 1);
  clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  auto clf2 = bayesnet::XSp2de(2, 3);
  clf2.setHyperparameters({{"parent1", 0}, {"parent2", 1}});
  clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  REQUIRE(clf.to_string() == clf2.to_string());
}
TEST_CASE("Check different smoothing", "[XSP2DE]")
{
  auto raw = RawDatasets("iris", true);

  auto clf = bayesnet::XSp2de(0, 1);
  clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::ORIGINAL);
  auto clf2 = bayesnet::XSp2de(0, 1);
  clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::LAPLACE);
  auto clf3 = bayesnet::XSp2de(0, 1);
  clf3.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::NONE);
  auto clf4 = bayesnet::XSp2de(0, 1);
  clf4.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, bayesnet::Smoothing_t::CESTNIK);
  auto score = clf.score(raw.X_test, raw.y_test);
  auto score2 = clf2.score(raw.X_test, raw.y_test);
  auto score3 = clf3.score(raw.X_test, raw.y_test);
  auto score4 = clf4.score(raw.X_test, raw.y_test);
  std::cerr << "GOLDEN[smoothing] ORIGINAL=" << score << " LAPLACE=" << score2
            << " NONE=" << score3 << " CESTNIK=" << score4 << std::endl;
  REQUIRE(score == Catch::Approx(1.0).epsilon(raw.epsilon));
  REQUIRE(score2 == Catch::Approx(1.0).epsilon(raw.epsilon));
  REQUIRE(score3 == Catch::Approx(1.0).epsilon(raw.epsilon));
  // CESTNIK is now a real m-estimate (not a silent no-op that falls back to no
  // smoothing): it produces a genuinely smoothed model, so its score differs
  // from the unsmoothed NONE run.
  REQUIRE(score4 == Catch::Approx(0.866667).epsilon(raw.epsilon));
  REQUIRE(score4 != Catch::Approx(score3).epsilon(raw.epsilon));
}
TEST_CASE("Check rest", "[XSP2DE]")
{
  auto raw = RawDatasets("iris", true);
  auto clf = bayesnet::XSp2de(0, 1);
  REQUIRE_THROWS_AS(clf.predict_proba(std::vector<int>({1,2,3,4})), std::logic_error);
  clf.fitx(raw.Xt, raw.yt, raw.weights, bayesnet::Smoothing_t::ORIGINAL);
  std::cerr << "GOLDEN[rest] score=" << clf.score(raw.Xv, raw.yv) << " predict=" << clf.predict({1,2,3,4}) << std::endl;
  REQUIRE(clf.getNFeatures() == 4);
  REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.973333359f).epsilon(raw.epsilon));
  REQUIRE(clf.predict({1,2,3,4}) == 1);

}
TEST_CASE("Joint superparents model (XSP2DE)", "[XSP2DE]")
{
  auto raw = RawDatasets("iris", true);

  // Joint parent model P(sp1,sp2|c): the two superparents become dependent,
  // adding the sp1->sp2 edge (8 -> 9 for iris) w.r.t. the independent model.
  bayesnet::XSp2de clfJoint(0, 1);
  clfJoint.setHyperparameters({{"joint_parents", true}});
  clfJoint.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  REQUIRE(clfJoint.getNumberOfNodes() == 5);
  REQUIRE(clfJoint.getNumberOfEdges() == 9);
  REQUIRE(clfJoint.score(raw.X_test, raw.y_test) >= 0.90f);

  // Independent model keeps 8 edges, and the flag really changes the model.
  bayesnet::XSp2de clfInd(0, 1);
  clfInd.setHyperparameters({{"joint_parents", false}});
  clfInd.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
  REQUIRE(clfInd.getNumberOfEdges() == 8);
  REQUIRE(clfJoint.to_string() != clfInd.to_string());
}
