// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <ArffFiles/ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>
#include <bayesnet/ensembles/BoostAODE.h>
#include <bayesnet/classifiers/XSPODE.h>

std::vector<mdlp::labels_t> discretizeDataset(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y)
{
    std::vector<mdlp::labels_t> Xd;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        Xd.push_back(xd);
    }
    return Xd;
}
tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<std::string>, std::string, map<std::string, std::vector<int>>> loadDataset(const std::string& name, bool class_last)
{
    auto handler = ArffFiles();
    handler.load(name, class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    torch::Tensor Xd;
    auto states = map<std::string, std::vector<int>>();
    auto Xr = discretizeDataset(X, y);
    for (int i = 0; i < features.size(); ++i) {
        states[features[i]] = std::vector<int>(*max_element(Xr[i].begin(), Xr[i].end()) + 1);
        auto item = states.at(features[i]);
        iota(begin(item), end(item), 0);
    }
    states[className] = std::vector<int>(*max_element(y.begin(), y.end()) + 1);
    iota(begin(states.at(className)), end(states.at(className)), 0);
    return { Xr, y, features, className, states };
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name>" << std::endl;
        return 1;
    }
    std::string file_name = argv[1];
    bayesnet::BaseClassifier* clf = new bayesnet::XSpode(0);
    std::cout << "Library version: " << clf->getVersion() << std::endl;
    auto [X, y, features, className, states] = loadDataset(file_name, true);
    torch::Tensor weights = torch::full({ static_cast<long>(X[0].size()) }, 1.0 / X[0].size(), torch::kDouble);
    clf->fit(X, y, features, className, states, bayesnet::Smoothing_t::ORIGINAL);
    auto score = clf->score(X, y);
    std::cout << "File: " << file_name << " Model: XSpode(0) score: " << score << std::endl;
    delete clf;
    return 0;
}

