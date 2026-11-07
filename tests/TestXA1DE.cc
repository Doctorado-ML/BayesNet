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
#include "bayesnet/classifiers/XSPODE.h"
#include "TestUtils.h"
#include <algorithm>

// Cardinalities as the X engines derive them from the data (max value + 1),
// features first and the class apart, to compute the expected number of cells.
static std::pair<std::vector<int>, int> cardinalities(const RawDatasets& raw)
{
    std::vector<int> k;
    for (const auto& column : raw.Xv) k.push_back(*std::max_element(column.begin(), column.end()) + 1);
    return { k, *std::max_element(raw.yv.begin(), raw.yv.end()) + 1 };
}

TEST_CASE("Fit and Score", "[XA1DE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::XA1DE();
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    // Cells of the probability tables: the sum over the 9 SPODEs of glass.
    auto [k, C] = cardinalities(raw);
    long cells = 0;
    for (int sp = 0; sp < 9; ++sp) cells += bayesnet::XSpode::cellsOfSpode(k, C, sp);
    REQUIRE(clf.getNumberOfStates() == cells);
    REQUIRE(clf.getNumberOfStates() == 4008);
    REQUIRE(clf.getClassNumStates() == 6);
    REQUIRE(clf.score(raw.Xv, raw.yv) == Catch::Approx(0.82243).epsilon(raw.epsilon));
    REQUIRE(clf.getVersion() == "1.0.1");
}
