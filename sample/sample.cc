// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <map>
#include <string>
#include <ArffFiles/ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>
#include <bayesnet/classifiers/TANLd.h>
#include <bayesnet/classifiers/KDBLd.h>
#include <bayesnet/ensembles/AODELd.h>

torch::Tensor matrix2tensor(const std::vector<std::vector<float>>& matrix)
{
    auto tensor = torch::empty({ static_cast<int>(matrix.size()), static_cast<int>(matrix[0].size()) }, torch::kFloat32);
    for (int i = 0; i < matrix.size(); ++i) {
        tensor.index_put_({ i, "..." }, torch::tensor(matrix[i], torch::kFloat32));
    }
    return tensor;
}

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
std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::string> loadArff(const std::string& name, bool class_last)
{
    auto handler = ArffFiles();
    handler.load(name, class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t> X = handler.getX();
    mdlp::labels_t y = handler.getY();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    auto Xt = matrix2tensor(X);
    auto yt = torch::tensor(y, torch::kInt32);
    return { Xt, yt, features, handler.getClassName() };
}
// tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::string, map<std::string, std::vector<int>>> loadDataset(const std::string& name, bool class_last)
// {
//     auto [X, y, features, className] = loadArff(name, class_last);
//     // Discretize the dataset
//     torch::Tensor Xd;
//     auto states = map<std::string, std::vector<int>>();
//     // Fill the class states
//     states[className] = std::vector<int>(*max_element(y.begin(), y.end()) + 1);
//     iota(begin(states.at(className)), end(states.at(className)), 0);
//     auto Xr = discretizeDataset(X, y);
//     Xd = torch::zeros({ static_cast<int>(Xr.size()), static_cast<int>(Xr[0].size()) }, torch::kInt32);
//     for (int i = 0; i < features.size(); ++i) {
//         states[features[i]] = std::vector<int>(*max_element(Xr[i].begin(), Xr[i].end()) + 1);
//         auto item = states.at(features[i]);
//         iota(begin(item), end(item), 0);
//         Xd.index_put_({ i, "..." }, torch::tensor(Xr[i], torch::kInt32));
//     }
//     auto yt = torch::tensor(y, torch::kInt32);
//     return { Xd, yt, features, className, states };
// }

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <arff_file_name> <model>" << std::endl;
        return 1;
    }
    std::string file_name = argv[1];
    std::string model_name = argv[2];
    std::map<std::string, bayesnet::Classifier*> models{ {"TANLd", new bayesnet::TANLd()}, {"KDBLd", new bayesnet::KDBLd(2)}, {"AODELd", new bayesnet::AODELd() }
    };
    if (models.find(model_name) == models.end()) {
        std::cerr << "Model not found: " << model_name << std::endl;
        std::cerr << "Available models: ";
        for (const auto& model : models) {
            std::cerr << model.first << " ";
        }
        std::cerr << std::endl;
        return 1;
    }
    auto clf = models[model_name];
    std::cout << "Library version: " << clf->getVersion() << std::endl;
    // auto [X, y, features, className, states] = loadDataset(file_name, true);
    auto [Xt, yt, features, className] = loadArff(file_name, true);
    std::map<std::string, std::vector<int>> states;
    // int m = Xt.size(1);
    // auto weights = torch::full({ m }, 1 / m, torch::kDouble);
    // auto dataset = buildDataset(Xv, yv);
    // try {
    //     auto yresized = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
    //     dataset = torch::cat({ X, yresized }, 0);
    // }
    // catch (const std::exception& e) {
    //     std::stringstream oss;
    //     oss << "* Error in X and y dimensions *\n";
    //     oss << "X dimensions: " << dataset.sizes() << "\n";
    //     oss << "y dimensions: " << y.sizes();
    //     throw std::runtime_error(oss.str());
    // }
    clf->fit(Xt, yt, features, className, states, bayesnet::Smoothing_t::ORIGINAL);
    auto total = yt.size(0);
    auto y_proba = clf->predict_proba(Xt);
    auto y_pred = y_proba.argmax(1);
    auto accuracy_value = (y_pred == yt).sum().item<float>() / total;
    auto score = clf->score(Xt, yt);
    std::cout << "File: " << file_name << " Model: " << model_name << " score: " << score << " Computed accuracy: " << accuracy_value << std::endl;
    for (const auto clf : models) {
        delete clf.second;
    }
    return 0;
}

