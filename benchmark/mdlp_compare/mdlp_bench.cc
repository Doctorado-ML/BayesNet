// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
//
// Benchmark harness to compare two versions of the fimdlp (mdlp) library.
//
// The very same source is compiled twice, once against each fimdlp version,
// and run over every .arff file in tests/data. It reports, per dataset and
// model, the accuracy obtained with a stratified k-fold cross validation plus
// the time spent discretizing, fitting and scoring. It also dumps the cut
// points of the first fold so that any behavioural difference between the two
// versions can be located down to the individual feature.
//
// Models:
//   TAN   -> global discretization done here with mdlp::CPPFImdlp (fitted on
//            the train fold only) and then a plain TAN over the discrete data.
//   TANLd -> continuous data handed to TANLd, which discretizes internally
//            (iterative local discretization, also mdlp).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ArffFiles 2.0 moved the header and put everything in a namespace. Which one
// we get is decided by the fimdlp version under test, so support both.
#if defined(ARFF_V2)
#include <ArffFiles/ArffFiles.hpp>
using ArffReader = ArffFiles::ArffFiles;
#else
#include <ArffFiles.hpp>
using ArffReader = ArffFiles;
#endif
#include <fimdlp/CPPFImdlp.h>
#include <folding.hpp>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <bayesnet/classifiers/TAN.h>
#include <bayesnet/classifiers/TANLd.h>

using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static double ms_since(const clk::time_point& start)
{
    return std::chrono::duration<double, std::milli>(clk::now() - start).count();
}

// ---------------------------------------------------------------- catalog --

// tests/data/all.txt holds "name;className;numericFeatures" where the third
// field is "all", "none" or a json list of the numeric feature indices.
struct CatalogEntry {
    std::string className;
    std::vector<int> numericFeaturesIdx;   // {-1} means "all of them"
};

static std::string trim(const std::string& str)
{
    auto result = str;
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](int ch) { return !std::isspace(ch); }));
    result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch) { return !std::isspace(ch); }).base(), result.end());
    return result;
}

static std::vector<std::string> split(const std::string& text, char delimiter)
{
    std::vector<std::string> result;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        result.push_back(trim(token));
    }
    return result;
}

static std::map<std::string, CatalogEntry> loadCatalog(const std::string& data_path)
{
    std::map<std::string, CatalogEntry> catalog;
    std::ifstream file(data_path + "all.txt");
    if (!file.is_open()) {
        throw std::invalid_argument("Unable to open catalog file [" + data_path + "all.txt]");
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tokens = split(line, ';');
        CatalogEntry entry;
        entry.className = tokens.size() > 1 ? tokens[1] : "";
        if (tokens.size() < 3 || tokens[2] == "all") {
            entry.numericFeaturesIdx.push_back(-1);       // every feature is numeric
        } else if (tokens[2] != "none") {
            for (auto& f : nlohmann::json::parse(tokens[2])) {
                entry.numericFeaturesIdx.push_back(f);
            }
        }                                                 // "none" -> empty vector
        catalog[tokens[0]] = entry;
    }
    return catalog;
}

// ---------------------------------------------------------------- dataset --

struct Dataset {
    std::string name;
    std::vector<mdlp::samples_t> X;   // [feature][sample], continuous
    std::vector<int> y;
    std::vector<std::string> features;
    std::string className;
    std::vector<bool> is_numeric;
    int classNumStates = 0;
};

// Keeps a random (but seeded, hence identical across versions) subset of the
// samples. Only used to keep the huge datasets within a sensible runtime.
static void subsample(Dataset& ds, int max_samples, int seed)
{
    const auto n = static_cast<int>(ds.y.size());
    if (max_samples <= 0 || n <= max_samples) return;
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(seed);
    std::shuffle(indices.begin(), indices.end(), g);
    indices.resize(max_samples);
    std::sort(indices.begin(), indices.end());

    std::vector<int> y;
    y.reserve(max_samples);
    for (auto i : indices) y.push_back(ds.y[i]);
    for (auto& column : ds.X) {
        mdlp::samples_t col;
        col.reserve(max_samples);
        for (auto i : indices) col.push_back(column[i]);
        column = std::move(col);
    }
    ds.y = std::move(y);
    ds.classNumStates = *std::max_element(ds.y.begin(), ds.y.end()) + 1;
}

static Dataset loadDataset(const std::string& data_path, const std::string& name,
    const std::map<std::string, CatalogEntry>& catalog)
{
    auto it = catalog.find(name);
    if (it == catalog.end()) {
        throw std::invalid_argument("Dataset " + name + " is not present in all.txt");
    }
    ArffReader handler;
    // The class is not always the last attribute (kdd_JapaneseVowels has it
    // first), so take the name from the catalog rather than assuming.
    if (it->second.className.empty()) {
        handler.load(data_path + name + ".arff", true);
    } else {
        handler.load(data_path + name + ".arff", it->second.className);
    }
    Dataset ds;
    ds.name = name;
    ds.X = handler.getX();
    ds.y = handler.getY();
    ds.className = handler.getClassName();
    for (const auto& attribute : handler.getAttributes()) {
        ds.features.push_back(attribute.first);
    }
    ds.classNumStates = *std::max_element(ds.y.begin(), ds.y.end()) + 1;

    const auto& numericIdx = it->second.numericFeaturesIdx;
    if (numericIdx.empty()) {
        ds.is_numeric.assign(ds.features.size(), false);
    } else if (numericIdx[0] == -1) {
        ds.is_numeric.assign(ds.features.size(), true);
    } else {
        ds.is_numeric.assign(ds.features.size(), false);
        for (auto idx : numericIdx) {
            if (idx >= 0 && idx < static_cast<int>(ds.features.size())) ds.is_numeric[idx] = true;
        }
    }
    return ds;
}

// ------------------------------------------------------- global TAN model --

struct FoldDiscretization {
    std::vector<std::vector<int>> Xtrain, Xtest;          // [feature][sample]
    std::map<std::string, std::vector<int>> states;
    std::vector<mdlp::cutPoints_t> cutPoints;             // per feature, empty if not numeric
    double ms = 0.0;
};

// Fits the discretizer on the train fold only and applies it to both folds,
// which is the methodologically correct way and the one that actually stresses
// mdlp: n_features fits per fold.
static FoldDiscretization discretizeFold(const Dataset& ds, const std::vector<int>& train, const std::vector<int>& test)
{
    FoldDiscretization out;
    const auto n_features = ds.features.size();
    out.Xtrain.resize(n_features);
    out.Xtest.resize(n_features);
    out.cutPoints.resize(n_features);

    mdlp::labels_t ytrain;
    ytrain.reserve(train.size());
    for (auto idx : train) ytrain.push_back(ds.y[idx]);

    for (size_t f = 0; f < n_features; ++f) {
        mdlp::samples_t col_train, col_test;
        col_train.reserve(train.size());
        col_test.reserve(test.size());
        for (auto idx : train) col_train.push_back(ds.X[f][idx]);
        for (auto idx : test) col_test.push_back(ds.X[f][idx]);

        if (ds.is_numeric[f]) {
            auto start = clk::now();
            mdlp::CPPFImdlp discretizer;
            discretizer.fit(col_train, ytrain);
            out.Xtrain[f] = discretizer.transform(col_train);
            out.Xtest[f] = discretizer.transform(col_test);
            out.ms += ms_since(start);
            out.cutPoints[f] = discretizer.getCutPoints();
        } else {
            for (auto v : col_train) out.Xtrain[f].push_back(static_cast<int>(v));
            for (auto v : col_test) out.Xtest[f].push_back(static_cast<int>(v));
        }
        int max_state = *std::max_element(out.Xtrain[f].begin(), out.Xtrain[f].end());
        max_state = std::max(max_state, *std::max_element(out.Xtest[f].begin(), out.Xtest[f].end()));
        out.states[ds.features[f]] = std::vector<int>(max_state + 1);
        std::iota(out.states[ds.features[f]].begin(), out.states[ds.features[f]].end(), 0);
    }
    out.states[ds.className] = std::vector<int>(ds.classNumStates);
    std::iota(out.states[ds.className].begin(), out.states[ds.className].end(), 0);
    return out;
}

static std::vector<std::vector<int>> selectColumns(const std::vector<std::vector<int>>& X, const std::vector<int>& idx)
{
    std::vector<std::vector<int>> out(X.size());
    for (size_t f = 0; f < X.size(); ++f) {
        out[f].reserve(idx.size());
        for (auto i : idx) out[f].push_back(X[f][i]);
    }
    return out;
}

// ------------------------------------------------------------- statistics --

static double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stdev(const std::vector<double>& v)
{
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double acc = 0.0;
    for (auto x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / (v.size() - 1));
}

// --------------------------------------------------------------- run TAN ---

static json runTAN(const Dataset& ds, int folds, int seed, int reps)
{
    json result;
    std::vector<double> accuracies;
    std::vector<int> nodes, edges, states;
    double best_disc = 0.0, best_fit = 0.0, best_score = 0.0;
    json cutpoints_fold0 = json::object();

    for (int rep = 0; rep < reps; ++rep) {
        double disc_ms = 0.0, fit_ms = 0.0, score_ms = 0.0;
        auto y_copy = ds.y;
        folding::StratifiedKFold fold(folds, y_copy, seed);
        std::vector<double> rep_acc;
        for (int k = 0; k < folds; ++k) {
            auto [train, test] = fold.getFold(k);
            auto disc = discretizeFold(ds, train, test);
            disc_ms += disc.ms;

            std::vector<int> ytrain, ytest;
            for (auto i : train) ytrain.push_back(ds.y[i]);
            for (auto i : test) ytest.push_back(ds.y[i]);

            bayesnet::TAN clf;
            auto start = clk::now();
            clf.fit(disc.Xtrain, ytrain, ds.features, ds.className, disc.states, bayesnet::Smoothing_t::ORIGINAL);
            fit_ms += ms_since(start);

            start = clk::now();
            rep_acc.push_back(clf.score(disc.Xtest, ytest));
            score_ms += ms_since(start);

            if (rep == 0) {
                nodes.push_back(clf.getNumberOfNodes());
                edges.push_back(clf.getNumberOfEdges());
                states.push_back(clf.getNumberOfStates());
                if (k == 0) {
                    for (size_t f = 0; f < ds.features.size(); ++f) {
                        if (ds.is_numeric[f]) cutpoints_fold0[ds.features[f]] = disc.cutPoints[f];
                    }
                }
            }
        }
        if (rep == 0) accuracies = rep_acc;
        // Keep the fastest repetition: least polluted by scheduling noise.
        if (rep == 0 || disc_ms + fit_ms + score_ms < best_disc + best_fit + best_score) {
            best_disc = disc_ms;
            best_fit = fit_ms;
            best_score = score_ms;
        }
    }
    result["accuracy_folds"] = accuracies;
    result["accuracy_mean"] = mean(accuracies);
    result["accuracy_std"] = stdev(accuracies);
    result["discretize_ms"] = best_disc;
    result["fit_ms"] = best_fit;
    result["score_ms"] = best_score;
    result["total_ms"] = best_disc + best_fit + best_score;
    result["nodes"] = nodes;
    result["edges"] = edges;
    result["states"] = states;
    result["cutpoints_fold0"] = cutpoints_fold0;
    return result;
}

// ------------------------------------------------------------- run TANLd ---

static json runTANLd(const Dataset& ds, int folds, int seed, int reps)
{
    json result;
    const auto n_features = static_cast<long>(ds.features.size());
    const auto n_samples = static_cast<long>(ds.y.size());
    auto Xt = torch::empty({ n_features, n_samples }, torch::kFloat32);
    for (long f = 0; f < n_features; ++f) {
        Xt[f] = torch::tensor(ds.X[f], torch::kFloat32);
    }
    auto yt = torch::tensor(ds.y, torch::kInt32);

    // Ld models expect an empty state vector for every feature they have to
    // discretize themselves, and the class states pre-filled.
    std::map<std::string, std::vector<int>> base_states;
    for (size_t f = 0; f < ds.features.size(); ++f) {
        if (ds.is_numeric[f]) {
            base_states[ds.features[f]] = std::vector<int>();
        } else {
            int max_state = 0;
            for (auto v : ds.X[f]) max_state = std::max(max_state, static_cast<int>(v));
            base_states[ds.features[f]] = std::vector<int>(max_state + 1);
            std::iota(base_states[ds.features[f]].begin(), base_states[ds.features[f]].end(), 0);
        }
    }
    base_states[ds.className] = std::vector<int>(ds.classNumStates);
    std::iota(base_states[ds.className].begin(), base_states[ds.className].end(), 0);

    std::vector<double> accuracies;
    std::vector<int> nodes, edges, states;
    double best_fit = 0.0, best_score = 0.0;

    for (int rep = 0; rep < reps; ++rep) {
        double fit_ms = 0.0, score_ms = 0.0;
        auto y_copy = ds.y;
        folding::StratifiedKFold fold(folds, y_copy, seed);
        std::vector<double> rep_acc;
        for (int k = 0; k < folds; ++k) {
            auto [train, test] = fold.getFold(k);
            auto train_t = torch::tensor(train);
            auto test_t = torch::tensor(test);
            auto X_train = Xt.index({ torch::indexing::Slice(), train_t });
            auto y_train = yt.index({ train_t });
            auto X_test = Xt.index({ torch::indexing::Slice(), test_t });
            auto y_test = yt.index({ test_t });

            auto fold_states = base_states;
            bayesnet::TANLd clf;
            auto start = clk::now();
            clf.fit(X_train, y_train, ds.features, ds.className, fold_states, bayesnet::Smoothing_t::ORIGINAL);
            fit_ms += ms_since(start);

            start = clk::now();
            rep_acc.push_back(clf.score(X_test, y_test));
            score_ms += ms_since(start);

            if (rep == 0) {
                nodes.push_back(clf.getNumberOfNodes());
                edges.push_back(clf.getNumberOfEdges());
                states.push_back(clf.getNumberOfStates());
            }
        }
        if (rep == 0) accuracies = rep_acc;
        if (rep == 0 || fit_ms + score_ms < best_fit + best_score) {
            best_fit = fit_ms;
            best_score = score_ms;
        }
    }
    result["accuracy_folds"] = accuracies;
    result["accuracy_mean"] = mean(accuracies);
    result["accuracy_std"] = stdev(accuracies);
    result["discretize_ms"] = 0.0;                 // done inside fit()
    result["fit_ms"] = best_fit;
    result["score_ms"] = best_score;
    result["total_ms"] = best_fit + best_score;
    result["nodes"] = nodes;
    result["edges"] = edges;
    result["states"] = states;
    return result;
}

// ------------------------------------------------------------------ main ---

static void usage(const char* program)
{
    std::cerr << "Usage: " << program << " [options]\n"
        << "  --data <path>      directory holding the .arff files (default: tests/data of the build)\n"
        << "  --output <file>    json file to write the results to (default: results.json)\n"
        << "  --datasets a,b,c   comma separated list of datasets (default: every .arff found)\n"
        << "  --models TAN,TANLd comma separated list of models (default: TAN,TANLd)\n"
        << "  --folds <n>        number of folds (default: 5)\n"
        << "  --seed <n>         seed for the stratified folds (default: 271)\n"
        << "  --reps <n>         timing repetitions, fastest one is kept (default: 1)\n"
        << "  --max-samples <n>  cap the samples per dataset, seeded subsample (default: 0 = all)\n"
        << "  --append           merge into an existing --output file instead of overwriting it\n"
        << "  --threads <n>      torch threads (default: 1, for reproducible timings)\n";
}

int main(int argc, char* argv[])
{
    std::string data_path = "../../tests/data/";
    std::string output = "results.json";
    std::vector<std::string> dataset_names;
    std::vector<std::string> model_names{ "TAN", "TANLd" };
    int folds = 5, seed = 271, reps = 1, threads = 1, max_samples = 0;
    bool append = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
            };
        if (arg == "--data") data_path = next();
        else if (arg == "--output") output = next();
        else if (arg == "--datasets") dataset_names = split(next(), ',');
        else if (arg == "--models") model_names = split(next(), ',');
        else if (arg == "--folds") folds = std::stoi(next());
        else if (arg == "--seed") seed = std::stoi(next());
        else if (arg == "--reps") reps = std::stoi(next());
        else if (arg == "--max-samples") max_samples = std::stoi(next());
        else if (arg == "--append") append = true;
        else if (arg == "--threads") threads = std::stoi(next());
        else { usage(argv[0]); return 1; }
    }
    // ArffFiles rejects paths containing "..", so hand it an absolute one.
    data_path = std::filesystem::weakly_canonical(std::filesystem::path(data_path)).string();
    if (!data_path.empty() && data_path.back() != '/') data_path += '/';
    torch::set_num_threads(threads);

    auto catalog = loadCatalog(data_path);
    if (dataset_names.empty()) {
        // Every dataset of the catalog that actually has its .arff on disk.
        for (const auto& [name, _] : catalog) {
            std::ifstream probe(data_path + name + ".arff");
            if (probe.good()) dataset_names.push_back(name);
        }
    }

    const auto mdlp_version = mdlp::Discretizer::version();
    json report;
    report["mdlp_version"] = mdlp_version;
    report["bayesnet_version"] = bayesnet::TAN().getVersion();
    report["folds"] = folds;
    report["seed"] = seed;
    report["reps"] = reps;
    report["threads"] = threads;
    report["max_samples"] = max_samples;
    report["datasets"] = json::object();
    if (append) {
        // Lets a heavy dataset be measured in its own run and still end up in
        // the same report; the settings of the last run win.
        std::ifstream previous(output);
        if (previous.good()) {
            json old_report;
            previous >> old_report;
            report["datasets"] = old_report.value("datasets", json::object());
        }
    }

    std::cout << "mdlp version    : " << mdlp_version << "\n";
    std::cout << "bayesnet version: " << report["bayesnet_version"].get<std::string>() << "\n";
    std::cout << "datasets        : " << dataset_names.size() << " | folds: " << folds
        << " | seed: " << seed << " | reps: " << reps << "\n\n";
    std::cout << std::left << std::setw(22) << "dataset" << std::setw(8) << "model"
        << std::right << std::setw(10) << "accuracy" << std::setw(12) << "disc(ms)"
        << std::setw(12) << "fit(ms)" << std::setw(12) << "score(ms)" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (const auto& name : dataset_names) {
        auto ds = loadDataset(data_path, name, catalog);
        subsample(ds, max_samples, seed);
        json entry;
        entry["samples"] = ds.y.size();
        entry["features"] = ds.features.size();
        entry["classes"] = ds.classNumStates;
        entry["numeric_features"] = std::count(ds.is_numeric.begin(), ds.is_numeric.end(), true);
        // Per dataset, so that a heavy one measured in its own --append run does
        // not misrepresent the settings of the datasets already in the report.
        entry["reps"] = reps;
        entry["max_samples"] = max_samples;
        entry["models"] = json::object();
        for (const auto& model : model_names) {
            json r;
            // A model that throws on one dataset must not take the whole
            // comparison down; record it and carry on.
            try {
                if (model == "TAN") r = runTAN(ds, folds, seed, reps);
                else if (model == "TANLd") r = runTANLd(ds, folds, seed, reps);
                else throw std::invalid_argument("Unknown model " + model);
            }
            catch (const std::exception& e) {
                entry["models"][model] = { {"error", e.what()} };
                std::cout << std::left << std::setw(22) << name << std::setw(8) << model
                    << std::right << std::setw(10) << "ERROR" << "  " << e.what() << std::endl;
                continue;
            }
            entry["models"][model] = r;
            std::cout << std::left << std::setw(22) << name << std::setw(8) << model
                << std::right << std::fixed << std::setprecision(5) << std::setw(10) << r["accuracy_mean"].get<double>()
                << std::setprecision(2) << std::setw(12) << r["discretize_ms"].get<double>()
                << std::setw(12) << r["fit_ms"].get<double>()
                << std::setw(12) << r["score_ms"].get<double>() << std::endl;
        }
        report["datasets"][name] = entry;
    }

    std::ofstream out(output);
    out << report.dump(2) << std::endl;
    std::cout << "\nResults written to " << output << std::endl;
    return 0;
}
