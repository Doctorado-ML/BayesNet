// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>
#include <torch/torch.h>

#include <bayesnet/classifiers/XSPODE.h>

static std::vector<mdlp::labels_t> discretize_columns(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y)
{
    std::vector<mdlp::labels_t> Xd;
    Xd.reserve(X.size());
    mdlp::CPPFImdlp discretizer;
    for (auto& column : X) {
        discretizer.fit(column, y);
        Xd.push_back(discretizer.transform(column));
    }
    return Xd;
}

struct DiscreteDataset {
    std::vector<std::vector<int>> X;
    std::vector<int> y;
    std::vector<std::string> features;
    std::string className;
    std::map<std::string, std::vector<int>> states;
};

static DiscreteDataset load_dataset(const std::string& name, bool class_last = true)
{
    ArffFiles handler;
    handler.load(name, class_last);
    auto X = handler.getX();
    auto y = handler.getY();

    std::vector<std::string> features;
    for (const auto& attribute : handler.getAttributes()) {
        features.push_back(attribute.first);
    }

    auto Xd = discretize_columns(X, y);

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

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <arff_file> [super_parent_index]" << std::endl;
        return 1;
    }
    const std::string file_name = argv[1];
    const int super_parent = (argc >= 3) ? std::stoi(argv[2]) : 0;

    auto clf = std::make_unique<bayesnet::XSpode>(super_parent);
    std::cout << "BayesNet library version: " << clf->getVersion() << "\n";

    auto data = load_dataset(file_name);
    clf->fit(data.X, data.y, data.features, data.className, data.states, bayesnet::Smoothing_t::ORIGINAL);
    auto score = clf->score(data.X, data.y);
    std::cout << "Dataset: " << file_name << " Model: XSpode(" << super_parent << ") score: " << score << std::endl;
    return 0;
}
