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

const std::string ACTUAL_VERSION = "1.1.1";

TEST_CASE("Test Bayesian Classifiers score & version", "[Models]")
{
    map<pair<std::string, std::string>, float> scores{// Diabetes
                                                      {{"diabetes", "AODE"}, 0.82161},
                                                      {{"diabetes", "KDB"}, 0.852865},
                                                      {{"diabetes", "XSPODE"}, 0.631510437f},
                                                      {{"diabetes", "SPODE"}, 0.802083},
                                                      {{"diabetes", "TAN"}, 0.821615},
                                                      {{"diabetes", "AODELd"}, 0.8125f},
                                                      {{"diabetes", "KDBLd"}, 0.80208f},
                                                      {{"diabetes", "SPODELd"}, 0.7890625f},
                                                      {{"diabetes", "TANLd"}, 0.803385437f},
                                                      {{"diabetes", "BoostAODE"}, 0.83984f},
                                                      // Ecoli
                                                      {{"ecoli", "AODE"}, 0.889881},
                                                      {{"ecoli", "KDB"}, 0.889881},
                                                      {{"ecoli", "XSPODE"}, 0.696428597f},
                                                      {{"ecoli", "SPODE"}, 0.880952},
                                                      {{"ecoli", "TAN"}, 0.892857},
                                                      {{"ecoli", "AODELd"}, 0.875f},
                                                      {{"ecoli", "KDBLd"}, 0.880952358f},
                                                      {{"ecoli", "SPODELd"}, 0.839285731f},
                                                      {{"ecoli", "TANLd"}, 0.848214269f},
                                                      {{"ecoli", "BoostAODE"}, 0.89583f},
                                                      // Glass
                                                      {{"glass", "AODE"}, 0.79439},
                                                      {{"glass", "KDB"}, 0.827103},
                                                      {{"glass", "XSPODE"}, 0.775701},
                                                      {{"glass", "SPODE"}, 0.775701},
                                                      {{"glass", "TAN"}, 0.827103},
                                                      {{"glass", "AODELd"}, 0.799065411f},
                                                      {{"glass", "KDBLd"}, 0.82710278f},
                                                      {{"glass", "SPODELd"}, 0.780373812f},
                                                      {{"glass", "TANLd"}, 0.869158864f},
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
    std::map<std::string, bayesnet::BaseClassifier*> models{ {"AODE", new bayesnet::AODE()},
                                                             {"AODELd", new bayesnet::AODELd()},
                                                             {"BoostAODE", new bayesnet::BoostAODE()},
                                                             {"KDB", new bayesnet::KDB(2)},
                                                             {"KDBLd", new bayesnet::KDBLd(2)},
                                                             {"XSPODE", new bayesnet::XSpode(1)},
                                                             {"SPODE", new bayesnet::SPODE(1)},
                                                             {"SPODELd", new bayesnet::SPODELd(1)},
                                                             {"TAN", new bayesnet::TAN()},
                                                             {"TANLd", new bayesnet::TANLd()} };
    std::string name = GENERATE("AODE", "AODELd", "KDB", "KDBLd", "SPODE", "XSPODE", "SPODELd", "TAN", "TANLd");
    auto clf = models[name];

    SECTION("Test " + name + " classifier")
    {
        for (const std::string& file_name : { "glass", "iris", "ecoli", "diabetes" }) {
            auto clf = models[name];
            auto discretize = name.substr(name.length() - 2) != "Ld";
            auto raw = RawDatasets(file_name, discretize);
            clf->fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
            auto score = clf->score(raw.Xt, raw.yt);
            // std::cout << "Classifier: " << name << " File: " << file_name << " Score: " << score << " expected = " <<
            // scores[{file_name, name}] << std::endl;
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
    delete clf;
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
        REQUIRE(clf.getNumberOfStates() == 27);
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
    std::string model = GENERATE("TAN", "SPODE", "BoostAODEproba", "BoostAODEvoting");
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
    auto res_prob_voting = std::vector<std::vector<double>>(
        { {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0} });
    std::map<std::string, std::vector<std::vector<double>>> res_prob{ {"TAN", res_prob_tan},
                                                                     {"SPODE", res_prob_spode},
                                                                     {"BoostAODEproba", res_prob_baode},
                                                                     {"BoostAODEvoting", res_prob_voting} };
    std::map<std::string, bayesnet::BaseClassifier*> models{ {"TAN", new bayesnet::TAN()},
                                                             {"SPODE", new bayesnet::SPODE(0)},
                                                             {"BoostAODEproba", new bayesnet::BoostAODE(false)},
                                                             {"BoostAODEvoting", new bayesnet::BoostAODE(true)} };
    int init_index = 78;
    auto raw = RawDatasets("iris", true);

    SECTION("Test " + model + " predict_proba")
    {
        auto clf = models[model];
        clf->fit(raw.Xv, raw.yv, raw.features, raw.className, raw.states, raw.smoothing);
        auto y_pred_proba = clf->predict_proba(raw.Xv);
        auto yt_pred_proba = clf->predict_proba(raw.Xt);
        auto y_pred = clf->predict(raw.Xv);
        auto yt_pred = clf->predict(raw.Xt);
        REQUIRE(y_pred.size() == yt_pred.size(0));
        REQUIRE(y_pred.size() == y_pred_proba.size());
        REQUIRE(y_pred.size() == yt_pred_proba.size(0));
        REQUIRE(y_pred.size() == raw.yv.size());
        REQUIRE(y_pred_proba[0].size() == 3);
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
        delete clf;
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
TEST_CASE("SPODELd dataset", "[Models]")
{
    auto raw = RawDatasets("iris", false);
    auto clf = bayesnet::SPODELd(0);
    // raw.dataset.to(torch::kFloat32);
    clf.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing);
    auto score = clf.score(raw.Xt, raw.yt);
    clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
    auto scoret = clf.score(raw.Xt, raw.yt);
    REQUIRE(score == Catch::Approx(0.97333f).epsilon(raw.epsilon));
    REQUIRE(scoret == Catch::Approx(0.97333f).epsilon(raw.epsilon));
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
TEST_CASE("Incorrect type of data for SPODELd", "[Models]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::SPODELd(0);
    REQUIRE_THROWS_AS(clf.fit(raw.dataset, raw.features, raw.className, raw.states, raw.smoothing), std::runtime_error);
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
        testProposal(torch::Tensor& dataset_, std::vector<std::string>& features_, std::string& className_)
            : Proposal(dataset_, features_, className_)
        {
        }
        void test_X_y(const torch::Tensor& X, const torch::Tensor& y) { checkInput(X, y); }
    };
    auto raw = RawDatasets("iris", true);
    auto clf = testProposal(raw.dataset, raw.features, raw.className);
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
