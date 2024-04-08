#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "bayesnet/ensembles/BoostAODE.h"
#include "TestUtils.h"


TEST_CASE("Feature_select CFS", "[BoostAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.setHyperparameters({ {"select_features", "CFS"} });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 9 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
}
TEST_CASE("Feature_select IWSS", "[BoostAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.setHyperparameters({ {"select_features", "IWSS"}, {"threshold", 0.5 } });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 5 of 9 with IWSS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
}
TEST_CASE("Feature_select FCBF", "[BoostAODE]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::BoostAODE();
    clf.setHyperparameters({ {"select_features", "FCBF"}, {"threshold", 1e-7 } });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 90);
    REQUIRE(clf.getNumberOfEdges() == 153);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 5 of 9 with FCBF");
    REQUIRE(clf.getNotes()[1] == "Number of models: 9");
}
TEST_CASE("Test used features in train note and score", "[BoostAODE]")
{
    auto raw = RawDatasets("diabetes", true);
    auto clf = bayesnet::BoostAODE(true);
    clf.setHyperparameters({
        {"order", "asc"},
        {"convergence", true},
        {"select_features","CFS"},
        });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 72);
    REQUIRE(clf.getNumberOfEdges() == 120);
    REQUIRE(clf.getNotes().size() == 2);
    REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 8 with CFS");
    REQUIRE(clf.getNotes()[1] == "Number of models: 8");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.80078).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.80078).epsilon(raw.epsilon));
}
TEST_CASE("Voting vs proba", "[BoostAODE]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostAODE(false);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    auto score_proba = clf.score(raw.Xv, raw.yv);
    auto pred_proba = clf.predict_proba(raw.Xv);
    clf.setHyperparameters({
        {"predict_voting",true},
        });
    auto score_voting = clf.score(raw.Xv, raw.yv);
    auto pred_voting = clf.predict_proba(raw.Xv);
    REQUIRE(score_proba == Catch::Approx(0.97333).epsilon(raw.epsilon));
    REQUIRE(score_voting == Catch::Approx(0.98).epsilon(raw.epsilon));
    REQUIRE(pred_voting[83][2] == Catch::Approx(0.552091).epsilon(raw.epsilon));
    REQUIRE(pred_proba[83][2] == Catch::Approx(0.546017).epsilon(raw.epsilon));
    REQUIRE(clf.dump_cpt() == "");
    REQUIRE(clf.topological_order() == std::vector<std::string>());
}
TEST_CASE("Order asc, desc & random", "[BoostAODE]")
{
    auto raw = RawDatasets("glass", true);
    std::map<std::string, double> scores{
        {"asc", 0.83645f }, { "desc", 0.84579f }, { "rand", 0.84112 }
    };
    for (const std::string& order : { "asc", "desc", "rand" }) {
        auto clf = bayesnet::BoostAODE();
        clf.setHyperparameters({
            {"order", order},
            });
        clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto score = clf.score(raw.Xv, raw.yv);
        auto scoret = clf.score(raw.Xt, raw.yt);
        INFO("BoostAODE order: " + order);
        REQUIRE(score == Catch::Approx(scores[order]).epsilon(raw.epsilon));
        REQUIRE(scoret == Catch::Approx(scores[order]).epsilon(raw.epsilon));
    }
}
TEST_CASE("Oddities", "[BoostAODE]")
{
    auto clf = bayesnet::BoostAODE();
    auto raw = RawDatasets("iris", true);
    auto bad_hyper = nlohmann::json{
        { { "order", "duck" } },
        { { "select_features", "duck" } },
        { { "maxTolerance", 0 } },
        { { "maxTolerance", 5 } },
    };
    for (const auto& hyper : bad_hyper.items()) {
        INFO("BoostAODE hyper: " + hyper.value().dump());
        REQUIRE_THROWS_AS(clf.setHyperparameters(hyper.value()), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(clf.setHyperparameters({ {"maxTolerance", 0 } }), std::invalid_argument);
    auto bad_hyper_fit = nlohmann::json{
        { { "select_features","IWSS" }, { "threshold", -0.01 } },
        { { "select_features","IWSS" }, { "threshold", 0.51 } },
        { { "select_features","FCBF" }, { "threshold", 1e-8 } },
        { { "select_features","FCBF" }, { "threshold", 1.01 } },
    };
    for (const auto& hyper : bad_hyper_fit.items()) {
        INFO("BoostAODE hyper: " + hyper.value().dump());
        clf.setHyperparameters(hyper.value());
        REQUIRE_THROWS_AS(clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv), std::invalid_argument);
    }
}

TEST_CASE("Bisection", "[BoostAODE]")
{
    auto clf = bayesnet::BoostAODE();
    auto raw = RawDatasets("mfeat-factors", true);
    clf.setHyperparameters({
        {"bisection", true},
        {"maxTolerance", 3},
        {"convergence", true},
        });
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 217);
    REQUIRE(clf.getNumberOfEdges() == 431);
    REQUIRE(clf.getNotes().size() == 3);
    REQUIRE(clf.getNotes()[0] == "Convergence threshold reached & 15 models eliminated");
    REQUIRE(clf.getNotes()[1] == "Used features in train: 16 of 216");
    REQUIRE(clf.getNotes()[2] == "Number of models: 1");
    auto score = clf.score(raw.Xv, raw.yv);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(1.0f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(1.0f).epsilon(raw.epsilon));
}