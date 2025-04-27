// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <string>
#include <fimdlp/CPPFImdlp.h>
#include <folding.hpp>
#include <nlohmann/json.hpp>
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define JSON_VERSION (TO_STR(NLOHMANN_JSON_VERSION_MAJOR)  "."  TO_STR(NLOHMANN_JSON_VERSION_MINOR))
#include "TestUtils.h"

std::map<std::string, std::string> modules = {
    { "mdlp", "2.0.1" },
    { "Folding", "1.1.1" },
    { "json", "3.11" },
    { "ArffFiles", "1.1.0" }
};

TEST_CASE("MDLP", "[Modules]")
{
    auto fimdlp = mdlp::CPPFImdlp();
    REQUIRE(fimdlp.version() == modules["mdlp"]);
}
TEST_CASE("Folding", "[Modules]")
{
    auto folding = folding::KFold(5, 200);
    REQUIRE(folding.version() == modules["Folding"]);
}
TEST_CASE("NLOHMANN_JSON", "[Modules]")
{
    REQUIRE(JSON_VERSION == modules["json"]);
}
TEST_CASE("ArffFiles", "[Modules]")
{
    auto handler = ArffFiles();
    REQUIRE(handler.version() == modules["ArffFiles"]);
}
