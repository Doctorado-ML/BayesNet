// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>
#include <torch/torch.h>

#include <bayesnet/BaseClassifier.h>
#include <bayesnet/classifiers/TAN.h>
#include <bayesnet/classifiers/TANLd.h>
#include <bayesnet/classifiers/KDB.h>
#include <bayesnet/classifiers/KDBLd.h>
#include <bayesnet/classifiers/SPODE.h>
#include <bayesnet/classifiers/SPODELd.h>
#include <bayesnet/ensembles/AODE.h>
#include <bayesnet/ensembles/AODELd.h>
#include <bayesnet/ensembles/BoostAODE.h>

using ModelFactory = std::function<std::unique_ptr<bayesnet::BaseClassifier>()>;

static const std::map<std::string, ModelFactory>& available_models()
{
    static const std::map<std::string, ModelFactory> models{
        {"TAN",       [] { return std::make_unique<bayesnet::TAN>(); }},
        {"KDB",       [] { return std::make_unique<bayesnet::KDB>(2); }},
        {"SPODE",     [] { return std::make_unique<bayesnet::SPODE>(0); }},
        {"AODE",      [] { return std::make_unique<bayesnet::AODE>(); }},
        {"BoostAODE", [] { return std::make_unique<bayesnet::BoostAODE>(); }},
        {"TANLd",     [] { return std::make_unique<bayesnet::TANLd>(); }},
        {"KDBLd",     [] { return std::make_unique<bayesnet::KDBLd>(2); }},
        {"SPODELd",   [] { return std::make_unique<bayesnet::SPODELd>(0); }},
        {"AODELd",    [] { return std::make_unique<bayesnet::AODELd>(); }},
    };
    return models;
}

static bool is_local_discretization_model(const std::string& name)
{
    return name.size() >= 2 && name.compare(name.size() - 2, 2, "Ld") == 0;
}

static torch::Tensor matrix_to_tensor(const std::vector<std::vector<float>>& matrix)
{
    const auto rows = static_cast<long>(matrix.size());
    const auto cols = static_cast<long>(matrix.front().size());
    auto tensor = torch::empty({ rows, cols }, torch::kFloat32);
    for (long i = 0; i < rows; ++i) {
        tensor[i] = torch::tensor(matrix[i], torch::kFloat32);
    }
    return tensor;
}

struct ContinuousDataset {
    torch::Tensor X;                          // [n_features, n_samples] float
    torch::Tensor y;                          // [n_samples] int32
    std::vector<std::string> features;
    std::string className;
    std::map<std::string, std::vector<int>> states;
};

static ContinuousDataset load_arff(const std::string& path, bool class_last = true)
{
    ArffFiles handler;
    handler.load(path, class_last);
    auto X = handler.getX();
    auto y = handler.getY();
    std::vector<std::string> features;
    for (const auto& attribute : handler.getAttributes()) {
        features.push_back(attribute.first);
    }
    // Ld classifiers expect: an empty entry per (numeric) feature and the class states pre-filled
    std::map<std::string, std::vector<int>> states;
    for (const auto& feature : features) {
        states[feature] = std::vector<int>{};
    }
    const auto n_classes = *std::max_element(y.begin(), y.end()) + 1;
    states[handler.getClassName()] = std::vector<int>(n_classes);
    std::iota(states[handler.getClassName()].begin(), states[handler.getClassName()].end(), 0);
    return { matrix_to_tensor(X), torch::tensor(y, torch::kInt32), std::move(features), handler.getClassName(), std::move(states) };
}

struct DiscreteDataset {
    std::vector<std::vector<int>> X;          // [n_features][n_samples]
    std::vector<int> y;
    std::vector<std::string> features;
    std::string className;
    std::map<std::string, std::vector<int>> states;
};

static DiscreteDataset discretize_arff(const std::string& path, bool class_last = true)
{
    ArffFiles handler;
    handler.load(path, class_last);
    auto X = handler.getX();
    auto y = handler.getY();

    std::vector<std::string> features;
    for (const auto& attribute : handler.getAttributes()) {
        features.push_back(attribute.first);
    }

    std::vector<std::vector<int>> Xd;
    Xd.reserve(X.size());
    mdlp::CPPFImdlp discretizer;
    for (auto& column : X) {
        discretizer.fit(column, y);
        Xd.push_back(discretizer.transform(column));
    }

    std::map<std::string, std::vector<int>> states;
    for (size_t i = 0; i < features.size(); ++i) {
        const auto n_states = *std::max_element(Xd[i].begin(), Xd[i].end()) + 1;
        states[features[i]] = std::vector<int>(n_states);
        std::iota(states[features[i]].begin(), states[features[i]].end(), 0);
    }
    const auto n_classes = *std::max_element(y.begin(), y.end()) + 1;
    states[handler.getClassName()] = std::vector<int>(n_classes);
    std::iota(states[handler.getClassName()].begin(), states[handler.getClassName()].end(), 0);

    return { std::move(Xd), std::move(y), std::move(features), handler.getClassName(), std::move(states) };
}

static void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <arff_file> <model>\n\n";
    std::cerr << "Available models:\n";
    for (const auto& entry : available_models()) {
        std::cerr << "  - " << entry.first << "\n";
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    const std::string file_name = argv[1];
    const std::string model_name = argv[2];

    const auto& models = available_models();
    auto it = models.find(model_name);
    if (it == models.end()) {
        std::cerr << "Model not found: " << model_name << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    auto clf = it->second();
    std::cout << "BayesNet library version: " << clf->getVersion() << "\n";
    std::cout << "Dataset: " << file_name << "\n";
    std::cout << "Model:   " << model_name << "\n";

    float score = 0.0f;
    if (is_local_discretization_model(model_name)) {
        // Ld models take continuous tensors directly and discretize internally
        auto data = load_arff(file_name);
        clf->fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
        score = clf->score(data.X, data.y);
    } else {
        // Discrete classifiers need the dataset to be discretized beforehand
        auto data = discretize_arff(file_name);
        clf->fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
        score = clf->score(data.X, data.y);
    }

    std::cout << "Score:   " << score << std::endl;
    return 0;
}
