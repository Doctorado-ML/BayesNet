// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/XA1DE.h"
#include "TestUtils.h"

TEST_CASE("Fit and Score", "[XA1DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XA1DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    REQUIRE(clf.getNumberOfStates() == 297);
    REQUIRE(clf.getClassNumStates() == 6);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.82243).epsilon(raw.epsilon));
    REQUIRE(clf.getVersion() == "1.0.0");
}
