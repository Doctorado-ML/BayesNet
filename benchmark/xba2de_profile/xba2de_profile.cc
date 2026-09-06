// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2026 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
//
// XBA2DE performance profiler.
//
// Produces a per-stage timing breakdown of XBA2DE training so that an
// optimization can be measured against a recorded baseline instead of a
// guess. Every stage is timed in isolation and the whole fit is timed on
// top, so the numbers are directly attributable:
//
//   primitives      the information-theoretic kernels in Metrics
//   select_k_pairs  one full pair ranking (the boosting inner loop calls
//                   this once per round)
//   xsp2de          fitting and predicting a single pair model
//   xba2de          the complete ensemble fit
//
// Accuracy and the model count are reported alongside the timings: an
// optimization that is meant to preserve semantics must leave them
// untouched, and the JSON output makes that diffable.
//
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <bayesnet/classifiers/XSP2DE.h>
#include <bayesnet/ensembles/XBA2DE.h>
#include <bayesnet/utils/BayesMetrics.h>

using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

// A dataset ready to be handed to any BayesNet classifier.
struct Case {
    std::string label;
    torch::Tensor dataset; // (n+1) x m, last row is the class
    std::vector<std::string> features;
    std::string className = "class";
    std::map<std::string, std::vector<int>> states;
    int n = 0, m = 0, nClasses = 0;
};

// --------------------------------------------------------------------------
// Synthetic data: every feature agrees with the class `signal`% of the time
// and is uniform noise otherwise. Fixed seed => byte-identical across runs,
// which is what makes before/after comparisons meaningful.
// --------------------------------------------------------------------------
static Case makeSynthetic(int n, int m, int card, int nClasses, int signal, unsigned seed)
{
    Case c;
    std::ostringstream oss;
    oss << "synth n=" << n << " m=" << m << " card=" << card;
    c.label = oss.str();
    c.n = n; c.m = m; c.nClasses = nClasses;

    std::mt19937 g(seed);
    std::uniform_int_distribution<int> noise(0, card - 1);
    std::uniform_int_distribution<int> cls(0, nClasses - 1);
    std::uniform_int_distribution<int> pct(0, 99);

    c.dataset = torch::zeros({ n + 1, m }, torch::kInt32);
    std::vector<int> y(m);
    for (int i = 0; i < m; ++i) y[i] = cls(g);
    std::vector<int> col(m);
    for (int f = 0; f < n; ++f) {
        for (int i = 0; i < m; ++i) {
            col[i] = (pct(g) < signal) ? (y[i] % card) : noise(g);
        }
        c.dataset.index_put_({ f, "..." }, torch::tensor(col, torch::kInt32));
    }
    c.dataset.index_put_({ -1, "..." }, torch::tensor(y, torch::kInt32));

    std::vector<int> cardStates(card);
    std::iota(cardStates.begin(), cardStates.end(), 0);
    for (int f = 0; f < n; ++f) {
        auto name = "f" + std::to_string(f);
        c.features.push_back(name);
        c.states[name] = cardStates;
    }
    std::vector<int> classStates(nClasses);
    std::iota(classStates.begin(), classStates.end(), 0);
    c.states[c.className] = classStates;
    return c;
}

// Median is more robust than the mean for a handful of reps on a laptop.
static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t k = v.size() / 2;
    return v.size() % 2 ? v[k] : 0.5 * (v[k - 1] + v[k]);
}

template <typename F>
static double timeIt(int reps, F&& f)
{
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        f();
        samples.push_back(secs(t0, clk::now()));
    }
    return median(std::move(samples));
}

// --------------------------------------------------------------------------
// The four Metrics kernels every pair score is built from. Timed on one
// representative pair; SelectKPairs calls them O(n^2) times per round.
// --------------------------------------------------------------------------
static json profilePrimitives(const Case& c, int reps)
{
    bayesnet::Metrics metrics(c.dataset, c.features, c.className, c.nClasses);
    auto weights = torch::full({ c.m }, 1.0 / c.m, torch::kFloat64);
    auto xi = c.dataset.index({ 0, "..." });
    auto xj = c.dataset.index({ 1, "..." });
    auto labels = c.dataset.index({ -1, "..." });

    volatile double sink = 0.0;
    json out;
    out["entropy"] = 1e3 * timeIt(reps, [&] { sink = metrics.entropy(xi, weights); });
    // mutualInformation == entropy + the 2-argument conditionalEntropy, which
    // is the kernel that dominates the pair ranking.
    out["mutual_information"] = 1e3 * timeIt(reps, [&] { sink = metrics.mutualInformation(xi, xj, weights); });
    out["conditional_entropy3"] = 1e3 * timeIt(reps, [&] { sink = metrics.conditionalEntropy(xi, xj, labels, weights); });
    out["conditional_mutual_information"] =
        1e3 * timeIt(reps, [&] { sink = metrics.conditionalMutualInformation(xi, xj, labels, weights); });
    (void)sink;
    return out;
}

// One full pair ranking, in both the joint-relevance (XBA2DE default) and the
// legacy conditional-MI criteria.
static json profileSelectKPairs(const Case& c, int reps, double beta)
{
    bayesnet::Metrics metrics(c.dataset, c.features, c.className, c.nClasses);
    auto weights = torch::full({ c.m }, 1.0 / c.m, torch::kFloat64);
    std::vector<int> excluded;

    size_t nPairs = 0;
    json out;
    out["beta"] = beta;
    out["joint_relevance_s"] = timeIt(reps, [&] {
        nPairs = metrics.SelectKPairs(weights, excluded, false, 0, beta).size();
    });
    out["legacy_cmi_s"] = timeIt(reps, [&] {
        metrics.SelectKPairs(weights, excluded, false, 0, -1.0);
    });
    out["n_pairs"] = nPairs;
    return out;
}

// A single pair model: the unit of work the boosting loop actually adds.
static json profileXSp2de(const Case& c, int reps)
{
    auto weights = torch::full({ c.m }, 1.0 / c.m, torch::kFloat64);
    auto X = c.dataset.index({ torch::indexing::Slice(0, c.n), "..." }).contiguous();

    json out;
    out["fit_s"] = timeIt(reps, [&] {
        bayesnet::XSp2de model(0, 1);
        auto ds = c.dataset.clone();
        auto states = c.states;
        model.fit(ds, c.features, c.className, states, weights, bayesnet::Smoothing_t::ORIGINAL);
    });

    bayesnet::XSp2de fitted(0, 1);
    {
        auto ds = c.dataset.clone();
        auto states = c.states;
        fitted.fit(ds, c.features, c.className, states, weights, bayesnet::Smoothing_t::ORIGINAL);
    }
    out["predict_s"] = timeIt(reps, [&] {
        auto Xc = X;
        fitted.predict(Xc);
    });
    return out;
}

// The whole ensemble. Accuracy and model count travel with the timing so a
// refactor that silently changes behaviour cannot hide behind a speedup.
static json profileXba2de(const Case& c, const json& hyper)
{
    auto ds = c.dataset.clone();
    auto states = c.states;
    bayesnet::XBA2DE clf;
    if (!hyper.empty()) clf.setHyperparameters(hyper);

    auto t0 = clk::now();
    clf.fit(ds, c.features, c.className, states, bayesnet::Smoothing_t::ORIGINAL);
    double elapsed = secs(t0, clk::now());

    // Score on the full original dataset: deterministic, and sensitive to any
    // change in the models that were kept.
    auto X = c.dataset.index({ torch::indexing::Slice(0, c.n), "..." }).contiguous();
    auto y = c.dataset.index({ -1, "..." }).contiguous();
    double accuracy = clf.score(X, y);

    int nModels = -1;
    auto notes = clf.getNotes();
    for (const auto& note : notes) {
        const std::string prefix = "Number of models: ";
        if (note.rfind(prefix, 0) == 0) nModels = std::stoi(note.substr(prefix.size()));
    }

    json out;
    out["fit_s"] = elapsed;
    out["accuracy"] = accuracy;
    out["n_models"] = nModels;
    out["nodes"] = clf.getNumberOfNodes();
    out["edges"] = clf.getNumberOfEdges();
    out["notes"] = notes;
    return out;
}

static void usage(const char* prog)
{
    std::cout <<
        "usage: " << prog << " [options]\n"
        "  --synthetic N,M,CARD   add a synthetic case (repeatable)\n"
        "                         default cases if none given and no --arff\n"
        "  --reps K               repetitions per micro-benchmark, median kept (default 5)\n"
        "  --threads K            torch intra-op threads (default 1)\n"
        "  --beta V               pair-ranking beta for SelectKPairs (default 1.0)\n"
        "  --hyper JSON           hyperparameters for the XBA2DE fit\n"
        "  --skip-full            skip the whole-ensemble fit (it can take hours)\n"
        "  --json FILE            write results as JSON\n"
        "  --label NAME           tag the run (e.g. baseline, h1-fix)\n"
        "  -h, --help             this message\n";
}

int main(int argc, char** argv)
{
    int reps = 5, threads = 1;
    double beta = 1.0;
    bool skipFull = false;
    std::string jsonPath, runLabel = "run";
    json hyper = json::object();
    std::vector<Case> cases;
    std::vector<std::string> pendingSynthetic;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--synthetic") pendingSynthetic.push_back(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--beta") beta = std::stod(next());
        else if (a == "--hyper") hyper = json::parse(next());
        else if (a == "--skip-full") skipFull = true;
        else if (a == "--json") jsonPath = next();
        else if (a == "--label") runLabel = next();
        else { std::cerr << "unknown option: " << a << "\n"; usage(argv[0]); return 2; }
    }

    torch::set_num_threads(threads);

    // Default sweep: holds n and m fixed in turn so the n^2 and m growth of
    // the pair ranking are both visible.
    if (pendingSynthetic.empty()) {
        pendingSynthetic = { "20,2000,4", "20,5000,4", "40,5000,4", "40,20000,4" };
    }

    for (const auto& spec : pendingSynthetic) {
        int n = 20, m = 5000, card = 4;
        if (std::sscanf(spec.c_str(), "%d,%d,%d", &n, &m, &card) < 2) {
            std::cerr << "bad --synthetic spec: " << spec << " (expected N,M,CARD)\n";
            return 2;
        }
        cases.push_back(makeSynthetic(n, m, card, /*nClasses=*/3, /*signal=*/30, /*seed=*/42));
    }

    json report;
    report["label"] = runLabel;
    report["reps"] = reps;
    report["threads"] = threads;
    report["beta"] = beta;
    report["skip_full"] = skipFull;
    report["cases"] = json::array();

    std::cout << std::fixed;
    for (const auto& c : cases) {
        std::cout << "\n=== " << c.label << "  (n=" << c.n << " m=" << c.m
                  << " classes=" << c.nClasses << ") ===\n";

        json entry;
        entry["label"] = c.label;
        entry["n_features"] = c.n;
        entry["n_samples"] = c.m;
        entry["n_classes"] = c.nClasses;

        auto prim = profilePrimitives(c, reps);
        entry["primitives_ms"] = prim;
        std::cout << std::setprecision(4)
                  << "  entropy                        " << prim["entropy"].get<double>() << " ms\n"
                  << "  mutualInformation              " << prim["mutual_information"].get<double>() << " ms\n"
                  << "  conditionalEntropy (3-arg)     " << prim["conditional_entropy3"].get<double>() << " ms\n"
                  << "  conditionalMutualInformation   " << prim["conditional_mutual_information"].get<double>() << " ms\n";

        auto skp = profileSelectKPairs(c, reps, beta);
        entry["select_k_pairs"] = skp;
        std::cout << std::setprecision(3)
                  << "  SelectKPairs joint (beta=" << beta << ")   " << skp["joint_relevance_s"].get<double>()
                  << " s  (" << skp["n_pairs"].get<size_t>() << " pairs)\n"
                  << "  SelectKPairs legacy cMI        " << skp["legacy_cmi_s"].get<double>() << " s\n";

        auto sp = profileXSp2de(c, reps);
        entry["xsp2de"] = sp;
        std::cout << "  XSp2de fit                     " << sp["fit_s"].get<double>() << " s\n"
                  << "  XSp2de predict                 " << sp["predict_s"].get<double>() << " s\n";

        if (!skipFull) {
            auto full = profileXba2de(c, hyper);
            entry["xba2de"] = full;
            double fit = full["fit_s"].get<double>();
            double ranking = skp["joint_relevance_s"].get<double>();
            std::cout << "  XBA2DE fit TOTAL               " << fit << " s\n"
                      << "    models=" << full["n_models"].get<int>()
                      << "  accuracy=" << std::setprecision(5) << full["accuracy"].get<double>() << "\n"
                      << std::setprecision(1)
                      << "    ~" << (ranking > 0 ? fit / ranking : 0.0) << " pair rankings' worth of time\n";
        }
        report["cases"].push_back(entry);
    }

    if (!jsonPath.empty()) {
        std::ofstream f(jsonPath);
        f << report.dump(2) << "\n";
        std::cout << "\nwrote " << jsonPath << "\n";
    }
    return 0;
}
