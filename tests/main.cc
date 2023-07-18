#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "../src/KDB.h"
#include "../src/TAN.h"
#include "../src/SPODE.h"
#include "../src/AODE.h"
#include "utils.h"

TEST_CASE("Test Bayesian Classifiers score", "[BayesNet]")
{
    map <pair<string, string>, float> scores = {
        {{"diabetes", "AODE"}, 0.811198}, {{"diabetes", "KDB"}, 0.852865}, {{"diabetes", "SPODE"}, 0.802083}, {{"diabetes", "TAN"}, 0.821615},
        {{"ecoli", "AODE"}, 0.889881}, {{"ecoli", "KDB"}, 0.889881}, {{"ecoli", "SPODE"}, 0.880952}, {{"ecoli", "TAN"}, 0.892857},
        {{"glass", "AODE"}, 0.78972}, {{"glass", "KDB"}, 0.827103}, {{"glass", "SPODE"}, 0.775701}, {{"glass", "TAN"}, 0.827103},
        {{"iris", "AODE"}, 0.973333}, {{"iris", "KDB"}, 0.973333}, {{"iris", "SPODE"}, 0.973333}, {{"iris", "TAN"}, 0.973333}
    };

    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto[Xd, y, features, className, states] = loadFile(file_name);

    SECTION("Test TAN classifier (" + file_name + ")")
    {
        auto clf = bayesnet::TAN();
        clf.fit(Xd, y, features, className, states);
        auto score = clf.score(Xd, y);
        //scores[{file_name, "TAN"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "TAN"}]).epsilon(1e-6));
    }
    SECTION("Test KDB classifier (" + file_name + ")")
    {
        auto clf = bayesnet::KDB(2);
        clf.fit(Xd, y, features, className, states);
        auto score = clf.score(Xd, y);
        //scores[{file_name, "KDB"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "KDB"
        }]).epsilon(1e-6));
    }
    SECTION("Test SPODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::SPODE(1);
        clf.fit(Xd, y, features, className, states);
        auto score = clf.score(Xd, y);
        // scores[{file_name, "SPODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "SPODE"}]).epsilon(1e-6));
    }
    SECTION("Test AODE classifier (" + file_name + ")")
    {
        auto clf = bayesnet::AODE();
        clf.fit(Xd, y, features, className, states);
        auto score = clf.score(Xd, y);
        // scores[{file_name, "AODE"}] = score;
        REQUIRE(score == Catch::Approx(scores[{file_name, "AODE"}]).epsilon(1e-6));
    }
    // for (auto scores : scores) {
    //     cout << "{{\"" << scores.first.first << "\", \"" << scores.first.second << "\"}, " << scores.second << "}, ";
    // }
}