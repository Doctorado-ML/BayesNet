// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <ArffFiles.hpp>
#include <CPPFImdlp.h>
#include <bayesnet/ensembles/BoostAODE.h>
#include <torch/torch.h>

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

tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::string, map<std::string, std::vector<int>>> loadDataset(const std::string& name, bool class_last, torch::Device device)
{
    auto handler = ArffFiles();
    handler.load(name, class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    torch::Tensor Xd;
    auto states = map<std::string, std::vector<int>>();
    auto Xr = discretizeDataset(X, y);
    Xd = torch::zeros({ static_cast<int>(Xr.size()), static_cast<int>(Xr[0].size()) }, torch::kInt32).to(device);
    for (int i = 0; i < features.size(); ++i) {
        states[features[i]] = std::vector<int>(*max_element(Xr[i].begin(), Xr[i].end()) + 1);
        auto item = states.at(features[i]);
        iota(begin(item), end(item), 0);
        Xd.index_put_({ i, "..." }, torch::tensor(Xr[i], torch::kInt32).to(device));
    }
    states[className] = std::vector<int>(*max_element(y.begin(), y.end()) + 1);
    iota(begin(states.at(className)), end(states.at(className)), 0);
    return { Xd, torch::tensor(y, torch::kInt32).to(device), features, className, states };
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_name>" << std::endl;
        return 1;
    }
    std::string file_name = argv[1];
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }
    torch::Tensor X, y;
    std::vector<std::string> features;
    std::string className;
    map<std::string, std::vector<int>> states;
    auto clf = bayesnet::BoostAODE(false); // false for not using voting in predict
    std::cout << "Library version: " << clf.getVersion() << std::endl;
    tie(X, y, features, className, states) = loadDataset(file_name, true, device);
    clf.fit(X, y, features, className, states, bayesnet::Smoothing_t::LAPLACE);
    auto score = clf.score(X, y);
    std::cout << "File: " << file_name << " Model: BoostAODE score: " << score << std::endl;
    return 0;
}