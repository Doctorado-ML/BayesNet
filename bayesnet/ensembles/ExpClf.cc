// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "ExpClf.h"
#include <functional>  // std::cref
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {
    ExpClf::ExpClf() : semaphore_{ CountingSemaphore::getInstance() }, Boost(false)
    {
        validHyperparameters = {};
    }
    //
    // Parents
    //
    void ExpClf::add_active_parents(const std::vector<int>& active_parents)
    {
        for (const auto& parent : active_parents)
            aode_.add_active_parent(parent);
    }
    void ExpClf::add_active_parent(int parent)
    {
        aode_.add_active_parent(parent);
    }
    void ExpClf::remove_last_parent()
    {
        aode_.remove_last_parent();
    }
    //
    // Predict
    //
    std::vector<int> ExpClf::predict_spode(std::vector<std::vector<int>>& test_data, int parent)
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto predictions = std::vector<int>(test_size);

        int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
        std::vector<std::thread> threads;
        auto worker = [&](const std::vector<std::vector<int>>& samples, int begin, int chunk, int sample_size, std::vector<int>& predictions) {
            std::string threadName = "(V)PWorker-" + std::to_string(begin) + "-" + std::to_string(chunk);
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
                predictions[sample] = aode_.predict_spode(instance, parent);
            }
            semaphore_.release();
            };
        for (int begin = 0; begin < test_size; begin += chunk_size) {
            int chunk = std::min(chunk_size, test_size - begin);
            semaphore_.acquire();
            // std::cref: without it std::thread copies the whole test set into
            // every chunk's argument tuple, i.e. O(m^2 * n) bytes over the m/150 chunks.
            threads.emplace_back(worker, std::cref(test_data), begin, chunk, sample_size, std::ref(predictions));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return predictions;
    }
    torch::Tensor ExpClf::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        torch::Tensor y = torch::tensor(predict(X_));
        return y;
    }
    torch::Tensor ExpClf::predict_proba(torch::Tensor& X)
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
    float ExpClf::score(torch::Tensor& X, torch::Tensor& y)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return score(X_, y_);
    }
    std::vector<std::vector<double>> ExpClf::predict_proba(const std::vector<std::vector<int>>& test_data)
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto probabilities = std::vector<std::vector<double>>(test_size, std::vector<double>(aode_.statesClass()));

        int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
        std::vector<std::thread> threads;
        auto worker = [&](const std::vector<std::vector<int>>& samples, int begin, int chunk, int sample_size, std::vector<std::vector<double>>& predictions) {
            std::string threadName = "(V)PWorker-" + std::to_string(begin) + "-" + std::to_string(chunk);
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
    std::vector<int> ExpClf::predict(std::vector<std::vector<int>>& test_data)
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
    float ExpClf::score(std::vector<std::vector<int>>& test_data, std::vector<int>& labels)
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
    int ExpClf::getNumberOfNodes() const
    {
        return aode_.getNumberOfNodes();
    }
    int ExpClf::getNumberOfEdges() const
    {
        return aode_.getNumberOfEdges();
    }
    int ExpClf::getNumberOfStates() const
    {
        return aode_.getNumberOfStates();
    }
    int ExpClf::getClassNumStates() const
    {
        return aode_.statesClass();
    }


}