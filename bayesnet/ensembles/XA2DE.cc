// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <pthread.h>
#include <functional>  // std::cref
#include <thread>
#include "XA2DE.h"
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {
    XA2DE::XA2DE() : semaphore_{ CountingSemaphore::getInstance() }, Boost(false)
    {
        validHyperparameters = { "joint_parents" };
    }
    void XA2DE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("joint_parents")) {
            joint_parents_ = hyperparameters["joint_parents"];
            hyperparameters.erase("joint_parents");
        }
        Classifier::setHyperparameters(hyperparameters);
    }
    void XA2DE::trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        auto X = TensorUtils::to_matrix(dataset.slice(0, 0, dataset.size(0) - 1));
        auto y = TensorUtils::to_vector<int>(dataset.index({ -1, "..." }));
        int num_instances = X[0].size();
        // Normalized weights (sum = 1), matching Classifier::fit. The additive
        // smoothing pseudocount is calibrated to this scale, so training the flat
        // engine with weight 1.0 would silently under-smooth and diverge from an
        // equivalent ensemble of XSp2de submodels.
        weights_ = torch::full({ num_instances }, 1.0 / num_instances, torch::kDouble);
        aode_.fit(X, y, features, className, states, weights_, true, smoothing, joint_parents_);
    }
    //
    // Predict
    //
    torch::Tensor XA2DE::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        torch::Tensor y = torch::tensor(predict(X_));
        return y;
    }
    torch::Tensor XA2DE::predict_proba(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto probabilities = predict_proba(X_);
        auto n_samples = X.size(1);
        int n_classes = probabilities[0].size();
        auto y = torch::zeros({ n_samples, n_classes });
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                y[i][j] = probabilities[i][j];
            }
        }
        return y;
    }
    std::vector<std::vector<double>> XA2DE::predict_proba(const std::vector<std::vector<int>>& test_data)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto probabilities = std::vector<std::vector<double>>(test_size, std::vector<double>(aode_.statesClass()));

        int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
        std::vector<std::thread> threads;
        auto worker = [&](const std::vector<std::vector<int>>& samples, int begin, int chunk, int sample_size, std::vector<std::vector<double>>& predictions) {
            std::string threadName = "XA2DE-" + std::to_string(begin) + "-" + std::to_string(chunk);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif
            std::vector<int> instance(sample_size);
            for (int sample = begin; sample < begin + chunk; ++sample) {
                for (int feature = 0; feature < sample_size; ++feature) {
                    instance[feature] = samples[feature][sample];
                }
                predictions[sample] = aode_.predict_proba(instance);
            }
            semaphore_.release();
            };
        for (int begin = 0; begin < test_size; begin += chunk_size) {
            int chunk = std::min(chunk_size, test_size - begin);
            semaphore_.acquire();
            // std::cref: without it std::thread copies the whole test set into
            // every chunk's argument tuple, i.e. O(m^2 * n) bytes over the m/150 chunks.
            threads.emplace_back(worker, std::cref(test_data), begin, chunk, sample_size, std::ref(probabilities));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return probabilities;
    }
    std::vector<int> XA2DE::predict(std::vector<std::vector<int>>& test_data)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        auto probabilities = predict_proba(test_data);
        std::vector<int> predictions(probabilities.size(), 0);
        for (size_t i = 0; i < probabilities.size(); i++) {
            predictions[i] = std::distance(probabilities[i].begin(), std::max_element(probabilities[i].begin(), probabilities[i].end()));
        }
        return predictions;
    }
    float XA2DE::score(torch::Tensor& X, torch::Tensor& y)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return score(X_, y_);
    }
    float XA2DE::score(std::vector<std::vector<int>>& test_data, std::vector<int>& labels)
    {
        std::vector<int> predictions = predict(test_data);
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == labels[i]) {
                correct++;
            }
        }
        return static_cast<float>(correct) / predictions.size();
    }
    //
    // statistics
    //
    int XA2DE::getNumberOfNodes() const { return aode_.getNumberOfNodes(); }
    int XA2DE::getNumberOfEdges() const { return aode_.getNumberOfEdges(); }
    int XA2DE::getNumberOfStates() const { return aode_.getNumberOfStates(); }
    int XA2DE::getClassNumStates() const { return aode_.getClassNumStates(); }
}
