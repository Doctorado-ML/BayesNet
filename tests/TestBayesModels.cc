// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "TestUtils.h"
#include "bayesnet/classifiers/KDB.h"
#include "bayesnet/classifiers/KDBLd.h"
#include "bayesnet/classifiers/SPODE.h"
#include "bayesnet/classifiers/SPODELd.h"
#include "bayesnet/classifiers/TAN.h"
#include "bayesnet/classifiers/TANLd.h"
#include "bayesnet/classifiers/XSPODE.h"
#include "bayesnet/ensembles/AODE.h"
#include "bayesnet/ensembles/AODELd.h"
#include "bayesnet/ensembles/BoostAODE.h"

const std::string ACTUAL_VERSION = "1.2.1";

TEST_CASE("Test Bayesian Classifiers score & version", "[Models]")
{
    map<pair<std::string, std::string>, float> scores{// Diabetes
                                                      {{"diabetes", "AODE"}, 0.82161},
                                                      {{"diabetes", "KDB"}, 0.852865},
                                                      {{"diabetes", "XSPODE"}, 0.631510437f},
                                                      {{"diabetes", "SPODE"}, 0.802083},
                                                      {{"diabetes", "TAN"}, 0.821615},
                                                      {{"diabetes", "AODELd"}, 0.8125f},
                                                      {{"diabetes", "KDBLd"}, 0.804688f},
                                                      {{"diabetes", "SPODELd"}, 0.7890625f},
                                                      {{"diabetes", "TANLd"}, 0.8125f},
                                                      {{"diabetes", "BoostAODE"}, 0.83984f},
                                                      // Ecoli
                                                      {{"ecoli", "AODE"}, 0.889881},
                                                      {{"ecoli", "KDB"}, 0.889881},
                                                      {{"ecoli", "XSPODE"}, 0.696428597f},
                                                      {{"ecoli", "SPODE"}, 0.880952},
                                                      {{"ecoli", "TAN"}, 0.892857},
                                                      {{"ecoli", "AODELd"}, 0.875f},
                                                      {{"ecoli", "KDBLd"}, 0.872024f},
                                                      {{"ecoli", "SPODELd"}, 0.839285731f},
                                                      {{"ecoli", "TANLd"}, 0.869047642f},
                                                      {{"ecoli", "BoostAODE"}, 0.89583f},
                                                      // Glass
                                                      {{"glass", "AODE"}, 0.79439},
                                                      {{"glass", "KDB"}, 0.827103},
                                                      {{"glass", "XSPODE"}, 0.775701},
                                                      {{"glass", "SPODE"}, 0.775701},
                                                      {{"glass", "TAN"}, 0.827103},
                                                      {{"glass", "AODELd"}, 0.799065411f},
                                                      {{"glass", "KDBLd"}, 0.864485979f},
                                                      {{"glass", "SPODELd"}, 0.780373812f},
                                                      {{"glass", "TANLd"}, 0.831775725f},
                                                      {{"glass", "BoostAODE"}, 0.84579f},
                                                      // Iris
                                                      {{"iris", "AODE"}, 0.973333},
                                                      {{"iris", "KDB"}, 0.973333},
                                                      {{"iris", "XSPODE"}, 0.853333354f},
                                                      {{"iris", "SPODE"}, 0.973333},
                                                      {{"iris", "TAN"}, 0.973333},
                                                      {{"iris", "AODELd"}, 0.973333},
                                                      {{"iris", "KDBLd"}, 0.973333},
                                                      {{"iris", "SPODELd"}, 0.96f},
                                                      {{"iris", "TANLd"}, 0.97333f},
                                                      {{"iris", "BoostAODE"}, 0.98f} };
    std::map<std::string, std::unique_ptr<bayesnet::BaseClassifier>> models;
    models["AODE"] = std::make_unique<bayesnet::AODE>();
    models["AODELd"] = std::make_unique<bayesnet::AODELd>();
    models["BoostAODE"] = std::make_unique<bayesnet::BoostAODE>();
    models["KDB"] = std::make_unique<bayesnet::KDB>(2);
    models["KDBLd"] = std::make_unique<bayesnet::KDBLd>(2);
    models["XSPODE"] = std::make_unique<bayesnet::XSpode>(1);
    models["SPODE"] = std::make_unique<bayesnet::SPODE>(1);
    models["SPODELd"] = std::make_unique<bayesnet::SPODELd>(1);
    models["TAN"] = std::make_unique<bayesnet::TAN>();
    models["TANLd"] = std::make_unique<bayesnet::TANLd>();
    std::string name = GENERATE("AODE", "AODELd", "KDB", "KDBLd", "SPODE", "XSPODE", "SPODELd", "TAN", "TANLd");
    auto clf = std::move(models[name]);

    SECTION("Test " + name + " classifier")
    {
        for (const std::string& file_name : { "glass", "iris", "ecoli", "diabetes" }) {
            auto discretize = name.substr(name.length() - 2) != "Ld";
            auto raw = RawDatasets(file_name, discretize);
            clf->fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
            auto score = clf->score(raw.Xt, raw.yt);
            // std::cout << "Classifier: " << name << " File: " << file_name << " Score: " << score << " expected = " <<
            //     scores[{file_name, name}] << std::endl;
            INFO("Classifier: " << name << " File: " << file_name);
            REQUIRE(score == Catch::Approx(scores[{file_name, name}]).epsilon(raw.epsilon));
            REQUIRE(clf->getStatus() == bayesnet::NORMAL);
        }
    }
    SECTION("Library check version")
    {
        INFO("Checking version of " << name << " classifier");
        REQUIRE(clf->getVersion() == ACTUAL_VERSION);
    }
}
TEST_CASE("Models features & Graph", "[Models]")
{
    auto graph = std::vector<std::string>(
        { "digraph BayesNet {\nlabel=<BayesNet Test>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n",
         "\"class\" [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n",
         "\"class\" -> \"sepallength\"", "\"class\" -> \"sepalwidth\"", "\"class\" -> \"petallength\"",
         "\"class\" -> \"petalwidth\"", "\"petallength\" [shape=circle] \n", "\"petallength\" -> \"sepallength\"",
         "\"petalwidth\" [shape=circle] \n", "\"sepallength\" [shape=circle] \n", "\"sepallength\" -> \"sepalwidth\"",
         "\"sepalwidth\" [shape=circle] \n", "\"sepalwidth\" -> \"petalwidth\"", "}\n" });
    SECTION("Test TAN")
    {
        auto raw = RawDatasets("iris", true);
        auto clf = bayesnet::TAN();
        clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        REQUIRE(clf.getNumberOfNodes() == 5);
        REQUIRE(clf.getNumberOfEdges() == 7);
        REQUIRE(clf.getNumberOfStates() == 19);
        REQUIRE(clf.getClassNumStates() == 3);
        REQUIRE(clf.show() == std::vector<std::string>{"class -> sepallength, sepalwidth, petallength, petalwidth, ",
            "petallength -> sepallength, ", "petalwidth -> ",
            "sepallength -> sepalwidth, ", "sepalwidth -> petalwidth, "});
        REQUIRE(clf.graph("Test") == graph);
    }
    SECTION("Test TANLd")
    {
        auto clf = bayesnet::TANLd();
        auto raw = RawDatasets("iris", false);
        clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
        REQUIRE(clf.getNumberOfNodes() == 5);
        REQUIRE(clf.getNumberOfEdges() == 7);
        REQUIRE(clf.getNumberOfStates() == 26);
        REQUIRE(clf.getClassNumStates() == 3);
        REQUIRE(clf.show() == std::vector<std::string>{"class -> sepallength, sepalwidth, petallength, petalwidth, ",
            "petallength -> sepallength, ", "petalwidth -> ",
            "sepallength -> sepalwidth, ", "sepalwidth -> petalwidth, "});
        REQUIRE(clf.graph("Test") == graph);
    }
}
TEST_CASE("Get num features & num edges", "[Models]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::KDB(2);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
}
TEST_CASE("Model predict_proba", "[Models]")
{
    std::string model = GENERATE("TAN", "SPODE", "BoostAODEproba", "BoostAODEvoting", "TANLd", "SPODELd", "KDBLd");
    auto res_prob_tan = std::vector<std::vector<double>>({ {0.00375671, 0.994457, 0.00178621},
                                                          {0.00137462, 0.992734, 0.00589123},
                                                          {0.00137462, 0.992734, 0.00589123},
                                                          {0.00137462, 0.992734, 0.00589123},
                                                          {0.00218225, 0.992877, 0.00494094},
                                                          {0.00494209, 0.0978534, 0.897205},
                                                          {0.0054192, 0.974275, 0.0203054},
                                                          {0.00433012, 0.985054, 0.0106159},
                                                          {0.000860806, 0.996922, 0.00221698} });
    auto res_prob_spode = std::vector<std::vector<double>>({ {0.00419032, 0.994247, 0.00156265},
                                                            {0.00172808, 0.993433, 0.00483862},
                                                            {0.00172808, 0.993433, 0.00483862},
                                                            {0.00172808, 0.993433, 0.00483862},
                                                            {0.00279211, 0.993737, 0.00347077},
                                                            {0.0120674, 0.357909, 0.630024},
                                                            {0.00386239, 0.913919, 0.0822185},
                                                            {0.0244389, 0.966447, 0.00911374},
                                                            {0.003135, 0.991799, 0.0050661} });
    auto res_prob_baode = std::vector<std::vector<double>>({ {0.0112349, 0.962274, 0.0264907},
                                                            {0.00371025, 0.950592, 0.0456973},
                                                            {0.00371025, 0.950592, 0.0456973},
                                                            {0.00371025, 0.950592, 0.0456973},
                                                            {0.00369275, 0.84967, 0.146637},
                                                            {0.0252205, 0.113564, 0.861215},
                                                            {0.0284828, 0.770524, 0.200993},
                                                            {0.0213182, 0.857189, 0.121493},
                                                            {0.00868436, 0.949494, 0.0418215} });
    auto res_prob_tanld = std::vector<std::vector<double>>({ {0.000597557, 0.9957, 0.00370254},
                                                            {0.000731377, 0.997914, 0.0013544},
                                                            {0.000731377, 0.997914, 0.0013544},
                                                            {0.000731377, 0.997914, 0.0013544},
                                                            {0.000838614, 0.998122, 0.00103923},
                                                            {0.00130852, 0.0659492, 0.932742},
                                                            {0.00365946, 0.979412, 0.0169281},
                                                            {0.00435035, 0.986248, 0.00940212},
                                                            {0.000583815, 0.997746, 0.00167066} });
    auto res_prob_spodeld = std::vector<std::vector<double>>({ {0.000908024, 0.993742, 0.00535024 },
                                                            {0.00187726, 0.99167, 0.00645308 },
                                                            {0.00187726, 0.99167, 0.00645308 },
                                                            {0.00187726, 0.99167, 0.00645308 },
                                                            {0.00287539, 0.993736, 0.00338846 },
                                                            {0.00294402, 0.268495, 0.728561 },
                                                            {0.0132381, 0.873282, 0.113479 },
                                                            {0.0159412, 0.969228, 0.0148308 },
                                                            {0.00203487, 0.989762, 0.00820356 } });
    auto res_prob_kdbld = std::vector<std::vector<double>>({ {0.000738981, 0.997208, 0.00205272 },
                                                            {0.00087708, 0.996687, 0.00243633 },
                                                            {0.00087708, 0.996687, 0.00243633 },
                                                            {0.00087708, 0.996687, 0.00243633 },
                                                            {0.000738981, 0.997208, 0.00205272 },
                                                            {0.00512442, 0.0455504, 0.949325 },
                                                            {0.0023632, 0.976631, 0.0210063 },
                                                            {0.00189194, 0.992853, 0.00525538 },
                                                            {0.00189194, 0.992853, 0.00525538, } });
    auto res_prob_voting = std::vector<std::vector<double>>(
        { {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0} });
    std::map<std::string, std::vector<std::vector<double>>> res_prob{ {"TAN", res_prob_tan},
                                                                     {"SPODE", res_prob_spode},
                                                                     {"BoostAODEproba", res_prob_baode},
                                                                     {"BoostAODEvoting", res_prob_voting},
                                                                     {"TANLd", res_prob_tanld},
                                                                     {"SPODELd", res_prob_spodeld},
                                                                     {"KDBLd", res_prob_kdbld} };

    std::map<std::string, std::unique_ptr<bayesnet::BaseClassifier>> models;
    models["TAN"] = std::make_unique<bayesnet::TAN>();
    models["SPODE"] = std::make_unique<bayesnet::SPODE>(0);
    models["BoostAODEproba"] = std::make_unique<bayesnet::BoostAODE>(false);
    models["BoostAODEvoting"] = std::make_unique<bayesnet::BoostAODE>(true);
    models["TANLd"] = std::make_unique<bayesnet::TANLd>();
    models["SPODELd"] = std::make_unique<bayesnet::SPODELd>(0);
    models["KDBLd"] = std::make_unique<bayesnet::KDBLd>(2);

    int init_index = 78;

    SECTION("Test " + model + " predict_proba")
    {
        INFO("Testing " << model << " predict_proba");
        auto ld_model = model.substr(model.length() - 2) == "Ld";
        auto discretize = !ld_model;
        auto raw = RawDatasets("iris", discretize);
        auto& clf = *models[model];
        clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
        auto yt_pred_proba = clf.predict_proba(raw.Xt);
        auto yt_pred = clf.predict(raw.Xt);
        std::vector<int> y_pred;
        std::vector<std::vector<double>> y_pred_proba;
        if (!ld_model) {
            y_pred = clf.predict(raw.Xv);
            y_pred_proba = clf.predict_proba(raw.Xv);
            REQUIRE(y_pred.size() == y_pred_proba.size());
            REQUIRE(y_pred.size() == yt_pred.size(0));
            REQUIRE(y_pred.size() == yt_pred_proba.size(0));
            REQUIRE(y_pred_proba[0].size() == 3);
            REQUIRE(y_pred.size() == raw.yv.size());
            REQUIRE(yt_pred_proba.size(1) == y_pred_proba[0].size());
            for (int i = 0; i < 9; ++i) {
                auto maxElem = max_element(y_pred_proba[i].begin(), y_pred_proba[i].end());
                int predictedClass = distance(y_pred_proba[i].begin(), maxElem);
                REQUIRE(predictedClass == y_pred[i]);
                // Check predict is coherent with predict_proba
                REQUIRE(yt_pred_proba[i].argmax().item<int>() == y_pred[i]);
                for (int j = 0; j < yt_pred_proba.size(1); j++) {
                    REQUIRE(yt_pred_proba[i][j].item<double>() == Catch::Approx(y_pred_proba[i][j]).epsilon(raw.epsilon));
                }
            }
            // Check predict_proba values for vectors and tensors
            for (int i = 0; i < 9; i++) {
                REQUIRE(y_pred[i] == yt_pred[i].item<int>());
                for (int j = 0; j < 3; j++) {
                    REQUIRE(res_prob[model][i][j] == Catch::Approx(y_pred_proba[i + init_index][j]).epsilon(raw.epsilon));
                    REQUIRE(res_prob[model][i][j] ==
                        Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
                }
            }
        } else {
            // Check predict_proba values for vectors and tensors
            auto predictedClasses = yt_pred_proba.argmax(1);
            // std::cout << model << std::endl;
            for (int i = 0; i < 9; i++) {
                REQUIRE(predictedClasses[i].item<int>() == yt_pred[i].item<int>());
                // std::cout << "{";
                for (int j = 0; j < 3; j++) {
                    // std::cout << yt_pred_proba[i + init_index][j].item<double>() << ", ";
                    REQUIRE(res_prob[model][i][j] ==
                        Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
                }
                // std::cout << "\b\b}," << std::endl;
            }
        }
    }
}
TEST_CASE("AODE voting-proba", "[Models]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::AODE(false);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score_proba = clf.score(raw.Xv, raw.yv);
    auto pred_proba = clf.predict_proba(raw.Xv);
    clf.setHyperparameters({
        {"predict_voting", true},
        });
    auto score_voting = clf.score(raw.Xv, raw.yv);
    auto pred_voting = clf.predict_proba(raw.Xv);
    REQUIRE(score_proba == Catch::Approx(0.79439f).epsilon(raw.epsilon));
    REQUIRE(score_voting == Catch::Approx(0.78972f).epsilon(raw.epsilon));
    REQUIRE(pred_voting[67][0] == Catch::Approx(0.888889).epsilon(raw.epsilon));
    REQUIRE(pred_proba[67][0] == Catch::Approx(0.702184).epsilon(raw.epsilon));
    REQUIRE(clf.topological_order() == std::vector<std::string>());
}
TEST_CASE("Ld models with dataset", "[Models]")
{
    auto raw = RawDatasets("iris", false);
    auto clf = bayesnet::SPODELd(0);
    clf.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xt, raw.yt);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    auto clf2 = bayesnet::TANLd();
    clf2.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
    auto score2 = clf2.score(raw.Xt, raw.yt);
    clf2.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto score2t = clf2.score(raw.Xt, raw.yt);
    REQUIRE(score2 == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    REQUIRE(score2t == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    auto clf3 = bayesnet::KDBLd(2);
    clf3.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
    auto score3 = clf3.score(raw.Xt, raw.yt);
    clf3.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto score3t = clf3.score(raw.Xt, raw.yt);
    REQUIRE(score3 == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    REQUIRE(score3t == Catch::Approx(0.97333f).epsilon(raw.epsilon));
}
TEST_CASE("KDB with hyperparameters", "[Models]")
{
    auto raw = RawDatasets("glass", true);
    auto clf = bayesnet::KDB(2);
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xv, raw.yv);
    clf.setHyperparameters({
        {"k", 3},
        {"theta", 0.7},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto scoret = clf.score(raw.Xv, raw.yv);
    REQUIRE(score == Catch::Approx(0.827103).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.761682).epsilon(raw.epsilon));
}
TEST_CASE("Incorrect type of data for Ld models", "[Models]")
{
    auto raw = RawDatasets("iris", true);
    auto clfs = bayesnet::SPODELd(0);
    REQUIRE_THROWS_AS(clfs.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
    auto clft = bayesnet::TANLd();
    REQUIRE_THROWS_AS(clft.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
    auto clfk = bayesnet::KDBLd(0);
    REQUIRE_THROWS_AS(clfk.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
}
TEST_CASE("Predict, predict_proba & score without fitting", "[Models]")
{
    auto clf = bayesnet::AODE();
    auto raw = RawDatasets("iris", true);
    std::string message = "Ensemble has not been fitted";
    REQUIRE_THROWS_AS(clf.predict(raw.Xv), std::logic_error);
    REQUIRE_THROWS_AS(clf.predict_proba(raw.Xv), std::logic_error);
    REQUIRE_THROWS_AS(clf.predict(raw.Xt), std::logic_error);
    REQUIRE_THROWS_AS(clf.predict_proba(raw.Xt), std::logic_error);
    REQUIRE_THROWS_AS(clf.score(raw.Xv, raw.yv), std::logic_error);
    REQUIRE_THROWS_AS(clf.score(raw.Xt, raw.yt), std::logic_error);
    REQUIRE_THROWS_WITH(clf.predict(raw.Xv), message);
    REQUIRE_THROWS_WITH(clf.predict_proba(raw.Xv), message);
    REQUIRE_THROWS_WITH(clf.predict(raw.Xt), message);
    REQUIRE_THROWS_WITH(clf.predict_proba(raw.Xt), message);
    REQUIRE_THROWS_WITH(clf.score(raw.Xv, raw.yv), message);
    REQUIRE_THROWS_WITH(clf.score(raw.Xt, raw.yt), message);
}
TEST_CASE("TAN & SPODE with hyperparameters", "[Models]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::TAN();
    clf.setHyperparameters({
        {"parent", 1},
        });
    clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xv, raw.yv);
    REQUIRE(score == Catch::Approx(0.973333).epsilon(raw.epsilon));
    auto clf2 = bayesnet::SPODE(0);
    clf2.setHyperparameters({
        {"parent", 1},
        });
    clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
    auto score2 = clf2.score(raw.Xv, raw.yv);
    REQUIRE(score2 == Catch::Approx(0.973333).epsilon(raw.epsilon));
}
TEST_CASE("TAN & SPODE with invalid hyperparameters", "[Models]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::TAN();
    clf.setHyperparameters({
        {"parent", 5},
        });
    REQUIRE_THROWS_AS(clf.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing),
        std::invalid_argument);
    auto clf2 = bayesnet::SPODE(0);
    clf2.setHyperparameters({
        {"parent", 5},
        });
    REQUIRE_THROWS_AS(clf2.fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing),
        std::invalid_argument);
}
TEST_CASE("Check proposal checkInput", "[Models]")
{
    class testProposal : public bayesnet::Proposal {
    public:
        testProposal(torch::Tensor& dataset_, std::vector<std::string>& features_, std::string& className_, std::vector<std::string>& notes_)
            : Proposal(dataset_, features_, className_, notes_)
        {
        }
        void test_X_y(const torch::Tensor& X, const torch::Tensor& y) { checkInput(X, y); }
    };
    auto raw = RawDatasets("iris", true);
    std::vector<std::string> notes;
    auto clf = testProposal(raw.dataset, raw.features, raw.className, notes);
    torch::Tensor X = torch::randint(0, 3, { 10, 4 });
    torch::Tensor y = torch::rand({ 10 });
    INFO("Check X is not float");
    REQUIRE_THROWS_AS(clf.test_X_y(X, y), std::invalid_argument);
    X = torch::rand({ 10, 4 });
    INFO("Check y is not integer");
    REQUIRE_THROWS_AS(clf.test_X_y(X, y), std::invalid_argument);
    y = torch::randint(0, 3, { 10 });
    INFO("X and y are correct");
    REQUIRE_NOTHROW(clf.test_X_y(X, y));
}
TEST_CASE("Check KDB loop detection", "[Models]")
{
    class testKDB : public bayesnet::KDB {
    public:
        testKDB() : KDB(2, 0) {}
        void test_add_m_edges(std::vector<std::string> features_, int idx, std::vector<int>& S, torch::Tensor& weights)
        {
            features = features_;
            add_m_edges(idx, S, weights);
        }
    };
    auto clf = testKDB();
    auto features = std::vector<std::string>{ "A", "B", "C" };
    int idx = 0;
    std::vector<int> S = { 0 };
    torch::Tensor weights = torch::tensor({
        {  1.0, 10.0,  0.0 },   // row0 -> picks col1
        {  0.0,  1.0, 10.0 },   // row1 -> picks col2
        { 10.0,  0.0,  1.0 },   // row2 -> picks col0
        });
    REQUIRE_NOTHROW(clf.test_add_m_edges(features, 0, S, weights));
    REQUIRE_NOTHROW(clf.test_add_m_edges(features, 1, S, weights));
}
TEST_CASE("Local discretization hyperparameters", "[Models]")
{
    auto raw = RawDatasets("iris", false);
    auto clfs = bayesnet::SPODELd(0);
    clfs.setHyperparameters({
        {"max_iterations", 7},
        {"verbose_convergence", true},
        });
    REQUIRE_NOTHROW(clfs.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clfs.getStatus() == bayesnet::NORMAL);
    auto clfk = bayesnet::KDBLd(0);
    clfk.setHyperparameters({
        {"k", 3},
        {"theta", 1e-4},
        });
    REQUIRE_NOTHROW(clfk.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clfk.getStatus() == bayesnet::NORMAL);
    auto clfa = bayesnet::AODELd();
    clfa.setHyperparameters({
        {"ld_proposed_cuts", 9},
        {"ld_algorithm", "BINQ"},
        });
    REQUIRE_NOTHROW(clfa.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clfa.getStatus() == bayesnet::NORMAL);
    auto clft = bayesnet::TANLd();
    clft.setHyperparameters({
        {"ld_proposed_cuts", 7},
        {"mdlp_max_depth", 5},
        {"mdlp_min_length", 3},
        {"ld_algorithm", "MDLP"},
        });
    REQUIRE_NOTHROW(clft.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clft.getStatus() == bayesnet::NORMAL);
    clft.setHyperparameters({
        {"ld_proposed_cuts", 9},
        {"ld_algorithm", "BINQ"},
        });
    REQUIRE_NOTHROW(clft.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clft.getStatus() == bayesnet::NORMAL);
    clft.setHyperparameters({
        {"ld_proposed_cuts", 5},
        {"ld_algorithm", "BINU"},
        });
    REQUIRE_NOTHROW(clft.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing));
    REQUIRE(clft.getStatus() == bayesnet::NORMAL);
}
