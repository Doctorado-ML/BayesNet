// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

// Golden regression tests for the 2.0 refactoring (see plan_2_0.md, Fase 0).
// Every model is fitted with default settings on the reference datasets and
// its observable behaviour (score, predictions, probabilities, graph counters
// and notes) is compared against reference values stored in
// tests/data/golden/. The reference files are generated with:
//
//   GOLDEN_GENERATE=1 ./TestBayesNet "[Golden]"
//
// A golden file must only be regenerated when a behaviour change is
// intentional, in a separate commit that justifies the change.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>
#include "TestUtils.h"
#include "bayesnet/config.h"
#include "bayesnet/classifiers/TAN.h"
#include "bayesnet/classifiers/TANLd.h"
#include "bayesnet/classifiers/KDB.h"
#include "bayesnet/classifiers/KDBLd.h"
#include "bayesnet/classifiers/SPODE.h"
#include "bayesnet/classifiers/SPODELd.h"
#include "bayesnet/classifiers/SPnDE.h"
#include "bayesnet/classifiers/XSPODE.h"
#include "bayesnet/classifiers/XSP2DE.h"
#include "bayesnet/ensembles/AODE.h"
#include "bayesnet/ensembles/AODELd.h"
#include "bayesnet/ensembles/A2DE.h"
#include "bayesnet/ensembles/BoostAODE.h"
#include "bayesnet/ensembles/BoostA2DE.h"
#include "bayesnet/ensembles/XBAODE.h"
#include "bayesnet/ensembles/XBA2DE.h"

using json = nlohmann::json;

namespace {

    const int N_PREDICTIONS = 20; // predictions stored per dataset
    const int N_PROBA_ROWS = 10;  // predict_proba rows stored per dataset
    const double TOLERANCE = 1e-6;

    std::string golden_dir()
    {
        return std::string(data_path.begin(), data_path.end()) + "golden/";
    }
    std::string golden_file(const std::string& model)
    {
        return golden_dir() + "golden_" + model + ".json";
    }
    bool generate_mode()
    {
        return std::getenv("GOLDEN_GENERATE") != nullptr;
    }

    struct GoldenModel {
        std::function<std::unique_ptr<bayesnet::BaseClassifier>()> make;
        bool continuous; // true => fit with continuous data (Ld models)
    };
    const std::map<std::string, GoldenModel>& golden_models()
    {
        static const std::map<std::string, GoldenModel> models{
            {"TAN",       {[]() { return std::make_unique<bayesnet::TAN>(); }, false}},
            {"KDB",       {[]() { return std::make_unique<bayesnet::KDB>(2); }, false}},
            {"SPODE",     {[]() { return std::make_unique<bayesnet::SPODE>(1); }, false}},
            {"SPnDE",     {[]() { return std::make_unique<bayesnet::SPnDE>(std::vector<int>{0, 1}); }, false}},
            {"XSPODE",    {[]() { return std::make_unique<bayesnet::XSpode>(1); }, false}},
            {"XSP2DE",    {[]() { return std::make_unique<bayesnet::XSp2de>(0, 1); }, false}},
            {"AODE",      {[]() { return std::make_unique<bayesnet::AODE>(); }, false}},
            {"A2DE",      {[]() { return std::make_unique<bayesnet::A2DE>(); }, false}},
            {"BoostAODE", {[]() { return std::make_unique<bayesnet::BoostAODE>(); }, false}},
            {"BoostA2DE", {[]() { return std::make_unique<bayesnet::BoostA2DE>(); }, false}},
            {"XBAODE",    {[]() { return std::make_unique<bayesnet::XBAODE>(); }, false}},
            {"XBA2DE",    {[]() { return std::make_unique<bayesnet::XBA2DE>(); }, false}},
            {"TANLd",     {[]() { return std::make_unique<bayesnet::TANLd>(); }, true}},
            {"KDBLd",     {[]() { return std::make_unique<bayesnet::KDBLd>(2); }, true}},
            {"SPODELd",   {[]() { return std::make_unique<bayesnet::SPODELd>(1); }, true}},
            {"AODELd",    {[]() { return std::make_unique<bayesnet::AODELd>(); }, true}},
        };
        return models;
    }

    // Fit the classifier and capture every externally observable result
    json capture(bayesnet::BaseClassifier& clf, RawDatasets& raw)
    {
        json entry;
        clf.fit(raw.Xt, raw.yt, raw.features, raw.className, raw.states, raw.smoothing);
        entry["score"] = clf.score(raw.Xt, raw.yt);
        auto y_pred = clf.predict(raw.Xt).to(torch::kInt32);
        auto predictions = json::array();
        for (int i = 0; i < std::min<int64_t>(N_PREDICTIONS, y_pred.size(0)); ++i) {
            predictions.push_back(y_pred[i].item<int>());
        }
        entry["predict"] = predictions;
        auto y_proba = clf.predict_proba(raw.Xt).to(torch::kDouble);
        auto proba = json::array();
        for (int i = 0; i < std::min<int64_t>(N_PROBA_ROWS, y_proba.size(0)); ++i) {
            auto row = json::array();
            for (int j = 0; j < y_proba.size(1); ++j) {
                row.push_back(y_proba[i][j].item<double>());
            }
            proba.push_back(row);
        }
        entry["predict_proba"] = proba;
        entry["nodes"] = clf.getNumberOfNodes();
        entry["edges"] = clf.getNumberOfEdges();
        entry["states"] = clf.getNumberOfStates();
        entry["class_states"] = clf.getClassNumStates();
        entry["notes"] = clf.getNotes();
        entry["status"] = static_cast<int>(clf.getStatus());
        return entry;
    }

    void check_entry(const json& expected, const json& actual)
    {
        REQUIRE(actual["nodes"] == expected["nodes"]);
        REQUIRE(actual["edges"] == expected["edges"]);
        REQUIRE(actual["states"] == expected["states"]);
        REQUIRE(actual["class_states"] == expected["class_states"]);
        REQUIRE(actual["status"] == expected["status"]);
        REQUIRE(actual["notes"] == expected["notes"]);
        REQUIRE(actual["predict"] == expected["predict"]);
        REQUIRE(actual["score"].get<double>() ==
            Catch::Approx(expected["score"].get<double>()).margin(TOLERANCE));
        REQUIRE(actual["predict_proba"].size() == expected["predict_proba"].size());
        for (size_t i = 0; i < expected["predict_proba"].size(); ++i) {
            REQUIRE(actual["predict_proba"][i].size() == expected["predict_proba"][i].size());
            for (size_t j = 0; j < expected["predict_proba"][i].size(); ++j) {
                INFO("predict_proba[" << i << "][" << j << "]");
                REQUIRE(actual["predict_proba"][i][j].get<double>() ==
                    Catch::Approx(expected["predict_proba"][i][j].get<double>()).margin(TOLERANCE));
            }
        }
    }

    json load_golden(const std::string& model)
    {
        auto file_name = golden_file(model);
        std::ifstream file(file_name);
        if (!file.is_open()) {
            FAIL("Golden file not found: " << file_name
                << ". Generate it with GOLDEN_GENERATE=1 ./TestBayesNet \"[Golden]\"");
        }
        return json::parse(file);
    }

    void save_golden(const std::string& model, const json& golden)
    {
        std::filesystem::create_directories(golden_dir());
        std::ofstream file(golden_file(model));
        file << golden.dump(2) << std::endl;
    }
}

TEST_CASE("Golden regression with default hyperparameters", "[Golden]")
{
    std::string name = GENERATE("TAN", "KDB", "SPODE", "SPnDE", "XSPODE", "XSP2DE",
        "AODE", "A2DE", "BoostAODE", "BoostA2DE", "XBAODE", "XBA2DE",
        "TANLd", "KDBLd", "SPODELd", "AODELd");
    const auto& model = golden_models().at(name);
    json golden;
    if (!generate_mode()) {
        golden = load_golden(name);
    }
    json results;
    results["model"] = name;
    for (const std::string& dataset : { "iris", "glass", "ecoli", "diabetes" }) {
        INFO("Model: " << name << " Dataset: " << dataset);
        auto raw = RawDatasets(dataset, !model.continuous);
        auto clf = model.make();
        auto entry = capture(*clf, raw);
        if (generate_mode()) {
            results["datasets"][dataset] = entry;
        } else {
            check_entry(golden["datasets"][dataset], entry);
        }
    }
    if (generate_mode()) {
        // Preserve variants section if it was already generated
        std::ifstream existing(golden_file(name));
        if (existing.is_open()) {
            auto previous = json::parse(existing);
            if (previous.contains("variants")) {
                results["variants"] = previous["variants"];
            }
        }
        save_golden(name, results);
        SUCCEED("Golden file generated for " << name);
    }
}

TEST_CASE("Golden regression of boosting hyperparameters", "[Golden]")
{
    // AODE-family ensembles use diabetes (8 features), A2DE-family use glass
    // (9 features, fewer samples) to keep the runtime reasonable.
    std::string name = GENERATE("BoostAODE", "XBAODE", "BoostA2DE", "XBA2DE");
    const std::string dataset = (name == "BoostAODE" || name == "XBAODE") ? "diabetes" : "glass";
    const std::vector<std::pair<std::string, json>> variants = {
        {"select_cfs", {{"select_features", "CFS"}}},
        {"select_iwss", {{"select_features", "IWSS"}, {"threshold", 0.5}}},
        {"select_fcbf", {{"select_features", "FCBF"}, {"threshold", 1e-7}}},
        {"block_update", {{"block_update", true}}},
        {"alpha_block", {{"alpha_block", true}}},
        {"weightless", {{"weightless", true}}},
        {"convergence_best", {{"convergence_best", true}}},
        {"no_bisection", {{"bisection", false}}},
        {"order_asc", {{"order", "asc"}}},
        {"order_rand", {{"order", "rand"}}},
    };
    json golden;
    if (!generate_mode()) {
        golden = load_golden(name);
    }
    json results;
    for (const auto& [variant_name, hyperparameters] : variants) {
        INFO("Model: " << name << " Dataset: " << dataset << " Variant: " << variant_name);
        auto raw = RawDatasets(dataset, true);
        auto clf = golden_models().at(name).make();
        clf->setHyperparameters(hyperparameters);
        auto entry = capture(*clf, raw);
        entry["hyperparameters"] = hyperparameters;
        if (generate_mode()) {
            results[variant_name] = entry;
        } else {
            check_entry(golden["variants"][variant_name], entry);
        }
    }
    if (generate_mode()) {
        // Merge into the model golden file generated by the previous test case
        json full;
        std::ifstream existing(golden_file(name));
        if (existing.is_open()) {
            full = json::parse(existing);
        } else {
            full["model"] = name;
        }
        full["variants"] = results;
        save_golden(name, full);
        SUCCEED("Golden variants generated for " << name);
    }
}
