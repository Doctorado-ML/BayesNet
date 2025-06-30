// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>
#include <folding.hpp>
#include <bayesnet/network/Network.h>


class RawDatasets {
public:
    RawDatasets(const std::string& file_name, bool discretize_, int num_samples_ = 0, bool shuffle_ = false, bool class_last = true, bool debug = false);
    torch::Tensor Xt, yt, dataset, weights;
    torch::Tensor X_train, y_train, X_test, y_test;
    std::vector<vector<int>> Xv;
    std::vector<int> yv;
    std::vector<double> weightsv;
    std::vector<string> features;
    std::string className;
    map<std::string, std::vector<int>> states;
    int nSamples, classNumStates;
    double epsilon = 1e-5;
    bool discretize;
    int num_samples = 0;
    bool shuffle = false;
    bayesnet::Smoothing_t smoothing = bayesnet::Smoothing_t::ORIGINAL;
private:
    std::string to_string()
    {
        std::string features_ = "";
        for (auto& f : features) {
            features_ += f + " ";
        }
        std::string states_ = "";
        for (auto& s : states) {
            states_ += s.first + " ";
            for (auto& v : s.second) {
                states_ += std::to_string(v) + " ";
            }
            states_ += "\n";
        }
        return "Xt dimensions: " + std::to_string(Xt.size(0)) + " " + std::to_string(Xt.size(1)) + "\n"
            "Xv dimensions: " + std::to_string(Xv.size()) + " " + std::to_string(Xv[0].size()) + "\n"
            + "yt dimensions: " + std::to_string(yt.size(0)) + "\n"
            + "yv dimensions: " + std::to_string(yv.size()) + "\n"
            + "X_train dimensions: " + std::to_string(X_train.size(0)) + " " + std::to_string(X_train.size(1)) + "\n"
            + "X_test dimensions: " + std::to_string(X_test.size(0)) + " " + std::to_string(X_test.size(1)) + "\n"
            + "y_train dimensions: " + std::to_string(y_train.size(0)) + "\n"
            + "y_test dimensions: " + std::to_string(y_test.size(0)) + "\n"
            + "features: " + std::to_string(features.size()) + "\n"
            + features_ + "\n"
            + "className: " + className + "\n"
            + "states: " + std::to_string(states.size()) + "\n"
            + "nSamples: " + std::to_string(nSamples) + "\n"
            + "classNumStates: " + std::to_string(classNumStates) + "\n"
            + "states: " + states_ + "\n";
    }
    map<std::string, int> discretizeDataset(std::vector<mdlp::samples_t>& X);
    void loadDataset(const std::string& name, bool class_last);
};

#endif //TEST_UTILS_H