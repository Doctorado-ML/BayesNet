#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "KDB.h"
#include "TAN.h"
#include "SPODE.h"
#include "AODE.h"
#include "BoostAODE.h"
#include "TANLd.h"
#include "KDBLd.h"
#include "SPODELd.h"
#include "AODELd.h"
#include "TestUtils.h"

TEST_CASE("Library check version", "[BayesNet]")
{
    auto clf = bayesnet::KDB(2);
    REQUIRE(clf.getVersion() == "1.0.2");
}
// TEST_CASE("Test Bayesian Classifiers score", "[BayesNet]")
// {
//     map <pair<std::string, std::string>, float> scores = {
//         // Diabetes
//         {{"diabetes", "AODE"}, 0.811198}, {{"diabetes", "KDB"}, 0.852865}, {{"diabetes", "SPODE"}, 0.802083}, {{"diabetes", "TAN"}, 0.821615},
//         {{"diabetes", "AODELd"}, 0.8138f}, {{"diabetes", "KDBLd"}, 0.80208f}, {{"diabetes", "SPODELd"}, 0.78646f}, {{"diabetes", "TANLd"}, 0.8099f},  {{"diabetes", "BoostAODE"}, 0.83984f},
//         // Ecoli
//         {{"ecoli", "AODE"}, 0.889881}, {{"ecoli", "KDB"}, 0.889881}, {{"ecoli", "SPODE"}, 0.880952}, {{"ecoli", "TAN"}, 0.892857},
//         {{"ecoli", "AODELd"}, 0.8869f}, {{"ecoli", "KDBLd"}, 0.875f}, {{"ecoli", "SPODELd"}, 0.84226f}, {{"ecoli", "TANLd"}, 0.86905f}, {{"ecoli", "BoostAODE"}, 0.89583f},
//         // Glass
//         {{"glass", "AODE"}, 0.78972}, {{"glass", "KDB"}, 0.827103}, {{"glass", "SPODE"}, 0.775701}, {{"glass", "TAN"}, 0.827103},
//         {{"glass", "AODELd"}, 0.79439f}, {{"glass", "KDBLd"}, 0.85047f}, {{"glass", "SPODELd"}, 0.79439f}, {{"glass", "TANLd"}, 0.86449f}, {{"glass", "BoostAODE"}, 0.84579f},
//         // Iris
//         {{"iris", "AODE"}, 0.973333}, {{"iris", "KDB"}, 0.973333}, {{"iris", "SPODE"}, 0.973333}, {{"iris", "TAN"}, 0.973333},
//         {{"iris", "AODELd"}, 0.973333}, {{"iris", "KDBLd"}, 0.973333}, {{"iris", "SPODELd"}, 0.96f}, {{"iris", "TANLd"}, 0.97333f}, {{"iris", "BoostAODE"}, 0.98f}
//     };

//     std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
//     auto raw = RawDatasets(file_name, false);

//     SECTION("Test TAN classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::TAN();
//         clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//         auto score = clf.score(raw.Xv, raw.yv);
//         //scores[{file_name, "TAN"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "TAN"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test TANLd classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::TANLd();
//         clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
//         auto score = clf.score(raw.Xt, raw.yt);
//         //scores[{file_name, "TANLd"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "TANLd"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test KDB classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::KDB(2);
//         clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//         auto score = clf.score(raw.Xv, raw.yv);
//         //scores[{file_name, "KDB"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "KDB"
//         }]).epsilon(raw.epsilon));
//     }
//     SECTION("Test KDBLd classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::KDBLd(2);
//         clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
//         auto score = clf.score(raw.Xt, raw.yt);
//         //scores[{file_name, "KDBLd"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "KDBLd"
//         }]).epsilon(raw.epsilon));
//     }
//     SECTION("Test SPODE classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::SPODE(1);
//         clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//         auto score = clf.score(raw.Xv, raw.yv);
//         // scores[{file_name, "SPODE"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "SPODE"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test SPODELd classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::SPODELd(1);
//         clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
//         auto score = clf.score(raw.Xt, raw.yt);
//         // scores[{file_name, "SPODELd"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "SPODELd"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test AODE classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::AODE();
//         clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//         auto score = clf.score(raw.Xv, raw.yv);
//         // scores[{file_name, "AODE"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "AODE"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test AODELd classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::AODELd();
//         clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
//         auto score = clf.score(raw.Xt, raw.yt);
//         // scores[{file_name, "AODELd"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "AODELd"}]).epsilon(raw.epsilon));
//     }
//     SECTION("Test BoostAODE classifier (" + file_name + ")")
//     {
//         auto clf = bayesnet::BoostAODE();
//         clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//         auto score = clf.score(raw.Xv, raw.yv);
//         // scores[{file_name, "BoostAODE"}] = score;
//         REQUIRE(score == Catch::Approx(scores[{file_name, "BoostAODE"}]).epsilon(raw.epsilon));
//     }
//     // for (auto scores : scores) {
//     //     std::cout << "{{\"" << scores.first.first << "\", \"" << scores.first.second << "\"}, " << scores.second << "}, ";
//     // }
// }
TEST_CASE("Models features", "[BayesNet]")
{
    auto graph = std::vector<std::string>({ "digraph BayesNet {\nlabel=<BayesNet Test>\nfontsize=30\nfontcolor=blue\nlabelloc=t\nlayout=circo\n",
        "class [shape=circle, fontcolor=red, fillcolor=lightblue, style=filled ] \n",
        "class -> sepallength", "class -> sepalwidth", "class -> petallength", "class -> petalwidth", "petallength [shape=circle] \n",
        "petallength -> sepallength", "petalwidth [shape=circle] \n", "sepallength [shape=circle] \n",
        "sepallength -> sepalwidth", "sepalwidth [shape=circle] \n", "sepalwidth -> petalwidth", "}\n"
        }
    );
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::TAN();
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 7);
    REQUIRE(clf.getNumberOfStates() == 19);
    REQUIRE(clf.getClassNumStates() == 3);
    REQUIRE(clf.show() == std::vector<std::string>{"class -> sepallength, sepalwidth, petallength, petalwidth, ", "petallength -> sepallength, ", "petalwidth -> ", "sepallength -> sepalwidth, ", "sepalwidth -> petalwidth, "});
    REQUIRE(clf.graph("Test") == graph);
}
TEST_CASE("Get num features & num edges", "[BayesNet]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::KDB(2);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    REQUIRE(clf.getNumberOfNodes() == 5);
    REQUIRE(clf.getNumberOfEdges() == 8);
}
TEST_CASE("BoostAODE feature_select CFS", "[BayesNet]")
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
// TEST_CASE("BoostAODE test used features in train note and score", "[BayesNet]")
// {
//     auto raw = RawDatasets("diabetes", true);
//     auto clf = bayesnet::BoostAODE();
//     clf.setHyperparameters({
//         {"ascending",true},
//         {"convergence", true},
//         {"repeatSparent",true},
//         {"select_features","CFS"},
//         });
//     clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
//     REQUIRE(clf.getNumberOfNodes() == 72);
//     REQUIRE(clf.getNumberOfEdges() == 120);
//     REQUIRE(clf.getNotes().size() == 3);
//     REQUIRE(clf.getNotes()[0] == "Used features in initialization: 6 of 8 with CFS");
//     REQUIRE(clf.getNotes()[1] == "Used features in train: 7 of 8");
//     REQUIRE(clf.getNotes()[2] == "Number of models: 8");
//     auto score = clf.score(raw.Xv, raw.yv);
//     auto scoret = clf.score(raw.Xt, raw.yt);
//     REQUIRE(score == Catch::Approx(0.8138).epsilon(raw.epsilon));
//     REQUIRE(scoret == Catch::Approx(0.8138).epsilon(raw.epsilon));
// }
TEST_CASE("Model predict_proba", "[BayesNet]")
{
    // std::string model = GENERATE("TAN", "SPODE", "BoostAODEprobabilities", "BoostAODEvoting");
    std::string model = GENERATE("TAN", "SPODE");
    std::cout << string(100, '*') << std::endl;
    std::cout << "************************************* CHANGE MODEL GENERATE ****************************************" << std::endl;
    std::cout << string(100, '*') << std::endl;
    auto res_prob_tan = std::vector<std::vector<double>>({
    { 0.00375671, 0.994457, 0.00178621 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00137462, 0.992734, 0.00589123 },
    { 0.00218225, 0.992877, 0.00494094 },
    { 0.00494209, 0.0978534, 0.897205 },
    { 0.0054192, 0.974275, 0.0203054 },
    { 0.00433012, 0.985054, 0.0106159 },
    { 0.000860806, 0.996922, 0.00221698 }
        });
    auto res_prob_spode = std::vector<std::vector<double>>({
     {0.00419032, 0.994247, 0.00156265},
     {0.00172808, 0.993433, 0.00483862},
     {0.00172808, 0.993433, 0.00483862},
     {0.00172808, 0.993433, 0.00483862},
     {0.00279211, 0.993737, 0.00347077},
     {0.0120674, 0.357909, 0.630024},
     {0.00386239, 0.913919, 0.0822185},
     {0.0244389, 0.966447, 0.00911374},
     {0.003135, 0.991799, 0.0050661}
        });
    auto res_prob_baode = std::vector<std::vector<double>>({
        {0.00803291, 0.9676, 0.0243672},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00189227, 0.859575, 0.138533},
        {0.0118341, 0.442149, 0.546017},
        {0.0216135, 0.785781, 0.192605},
        {0.0204803, 0.844276, 0.135244},
        {0.00576313, 0.961665, 0.0325716},
        });
    std::map<std::string, std::vector<std::vector<double>>> res_prob = { {"TAN", res_prob_tan}, {"SPODE", res_prob_spode} , {"BoostAODEproba", res_prob_baode }, {"BoostAODEvoting", res_prob_baode } };
    std::map<std::string, bayesnet::BaseClassifier*> models = { {"TAN", new bayesnet::TAN()}, {"SPODE", new bayesnet::SPODE(0)}, {"BoostAODEproba", new bayesnet::BoostAODE(false)}, {"BoostAODEvoting", new bayesnet::BoostAODE(true)} };
    int init_index = 78;
    auto raw = RawDatasets("iris", true);

    SECTION("Test " + model + " predict_proba")
    {
        auto clf = models[model];
        clf->fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
        auto y_pred_proba = clf->predict_proba(raw.Xv);
        auto y_pred = clf->predict(raw.Xv);
        auto yt_pred = clf->predict(raw.Xt);
        auto yt_pred_proba = clf->predict_proba(raw.Xt);
        REQUIRE(y_pred.size() == yt_pred.size(0));
        REQUIRE(y_pred.size() == y_pred_proba.size());
        REQUIRE(y_pred.size() == yt_pred_proba.size(0));
        REQUIRE(y_pred.size() == raw.yv.size());
        REQUIRE(y_pred_proba[0].size() == 3);
        REQUIRE(yt_pred_proba.size(1) == y_pred_proba[0].size());
        for (int i = 0; i < y_pred_proba.size(); ++i) {
            auto maxElem = max_element(y_pred_proba[i].begin(), y_pred_proba[i].end());
            int predictedClass = distance(y_pred_proba[i].begin(), maxElem);
            REQUIRE(predictedClass == y_pred[i]);
            // Check predict is coherent with predict_proba
            REQUIRE(yt_pred_proba[i].argmax().item<int>() == y_pred[i]);
        }
        // Check predict_proba values for vectors and tensors
        for (int i = 0; i < res_prob.size(); i++) {
            REQUIRE(y_pred[i] == yt_pred[i].item<int>());
            for (int j = 0; j < 3; j++) {
                REQUIRE(res_prob[model][i][j] == Catch::Approx(y_pred_proba[i + init_index][j]).epsilon(raw.epsilon));
                REQUIRE(res_prob[model][i][j] == Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
            }
        }
        delete clf;
    }
}
TEST_CASE("BoostAODE predict_proba proba", "[BayesNet]")
{
    auto res_prob = std::vector<std::vector<double>>({
        {0.00803291, 0.9676, 0.0243672},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00189227, 0.859575, 0.138533},
        {0.0118341, 0.442149, 0.546017},
        {0.0216135, 0.785781, 0.192605},
        {0.0204803, 0.844276, 0.135244},
        {0.00576313, 0.961665, 0.0325716},
        });
    int init_index = 78;
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostAODE(false);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    auto y_pred_proba = clf.predict_proba(raw.Xv);
    auto y_pred = clf.predict(raw.Xv);
    auto yt_pred = clf.predict(raw.Xt);
    auto yt_pred_proba = clf.predict_proba(raw.Xt);
    std::cout << "yt_pred_proba proba sizes " << yt_pred_proba.sizes() << std::endl;
    REQUIRE(y_pred.size() == yt_pred.size(0));
    REQUIRE(y_pred.size() == y_pred_proba.size());
    REQUIRE(y_pred.size() == yt_pred_proba.size(0));
    REQUIRE(y_pred.size() == raw.yv.size());
    REQUIRE(y_pred_proba[0].size() == 3);
    REQUIRE(yt_pred_proba.size(1) == y_pred_proba[0].size());
    for (int i = 0; i < y_pred_proba.size(); ++i) {
        // Check predict is coherent with predict_proba
        auto maxElem = max_element(y_pred_proba[i].begin(), y_pred_proba[i].end());
        int predictedClass = distance(y_pred_proba[i].begin(), maxElem);
        REQUIRE(predictedClass == y_pred[i]);
        REQUIRE(yt_pred_proba[i].argmax().item<int>() == y_pred[i]);
    }
    // Check predict_proba values for vectors and tensors
    for (int i = 0; i < res_prob.size(); i++) {
        REQUIRE(y_pred[i] == yt_pred[i].item<int>());
        for (int j = 0; j < 3; j++) {
            REQUIRE(res_prob[i][j] == Catch::Approx(y_pred_proba[i + init_index][j]).epsilon(raw.epsilon));
            REQUIRE(res_prob[i][j] == Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
        }
    }
    // for (int i = 0; i < res_prob.size(); i++) {
    //     for (int j = 0; j < 3; j++) {
    //         std::cout << y_pred_proba[i + init_index][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
TEST_CASE("BoostAODE predict_proba voting", "[BayesNet]")
{
    auto res_prob = std::vector<std::vector<double>>({
        {0.00803291, 0.9676, 0.0243672},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00398714, 0.945126, 0.050887},
        {0.00189227, 0.859575, 0.138533},
        {0.0118341, 0.442149, 0.546017},
        {0.0216135, 0.785781, 0.192605},
        {0.0204803, 0.844276, 0.135244},
        {0.00576313, 0.961665, 0.0325716},
        });
    int init_index = 78;
    auto raw = RawDatasets("iris", true);
    auto clf = bayesnet::BoostAODE(true);
    clf.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv);
    auto y_pred_proba = clf.predict_proba(raw.Xv);
    auto y_pred = clf.predict(raw.Xv);
    auto yt_pred = clf.predict(raw.Xt);
    auto yt_pred_proba = clf.predict_proba(raw.Xt);
    std::cout << "yt_pred_proba proba sizes " << yt_pred_proba.sizes() << std::endl;
    REQUIRE(y_pred.size() == yt_pred.size(0));
    REQUIRE(y_pred.size() == y_pred_proba.size());
    REQUIRE(y_pred.size() == yt_pred_proba.size(0));
    REQUIRE(y_pred.size() == raw.yv.size());
    REQUIRE(y_pred_proba[0].size() == 3);
    REQUIRE(yt_pred_proba.size(1) == y_pred_proba[0].size());
    for (int i = 0; i < y_pred_proba.size(); ++i) {
        auto maxElem = max_element(y_pred_proba[i].begin(), y_pred_proba[i].end());
        int predictedClass = distance(y_pred_proba[i].begin(), maxElem);
        REQUIRE(predictedClass == y_pred[i]);
        // Check predict is coherent with predict_proba
        for (int k = 0; k < yt_pred_proba[i].size(0); k++) {
            std::cout << yt_pred_proba[i][k].item<double>() << " ";
        }
        std::cout << "-> " << y_pred[i] << std::endl;
        REQUIRE(yt_pred_proba[i].argmax().item<int>() == y_pred[i]);
    }
    // Check predict_proba values for vectors and tensors
    for (int i = 0; i < res_prob.size(); i++) {
        REQUIRE(y_pred[i] == yt_pred[i].item<int>());
        for (int j = 0; j < 3; j++) {
            REQUIRE(res_prob[i][j] == Catch::Approx(y_pred_proba[i + init_index][j]).epsilon(raw.epsilon));
            REQUIRE(res_prob[i][j] == Catch::Approx(yt_pred_proba[i + init_index][j].item<double>()).epsilon(raw.epsilon));
        }
    }
    // for (int i = 0; i < res_prob.size(); i++) {
    //     for (int j = 0; j < 3; j++) {
    //         std::cout << y_pred_proba[i + init_index][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
