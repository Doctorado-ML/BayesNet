// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include "XSPODE.h"
#include "bayesnet/utils/TensorUtils.h"


namespace bayesnet {

    // --------------------------------------
    // Constructor
    // --------------------------------------
    XSpode::XSpode(int spIndex)
        : superParent_{ spIndex },
        nFeatures_{ 0 },
        statesClass_{ 0 },
        alpha_{ 1.0 },
        initializer_{ 1.0 },
        semaphore_{ CountingSemaphore::getInstance() }, Classifier(Network())
    {
        validHyperparameters = { "parent" };
    }

    void XSpode::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("parent")) {
            superParent_ = hyperparameters["parent"];
            hyperparameters.erase("parent");
        }
        Classifier::setHyperparameters(hyperparameters);
    }

    void XSpode::fit(std::vector<std::vector<int>>& X, std::vector<int>& y, torch::Tensor& weights_, const Smoothing_t smoothing)
    {
        m = X[0].size();
        n = X.size();
        buildModel(weights_);
        trainModel(weights_, smoothing);
        fitted=true;
    }

    // --------------------------------------
    // trainModel
    // --------------------------------------
    // Initialize storage needed for the super-parent and child features counts and probs.
    // --------------------------------------
    void XSpode::buildModel(const torch::Tensor& weights)
    {
        int numInstances = m;
        nFeatures_ = n;

        // Derive the number of states for each feature and for the class.
        // (This is just one approach; adapt to match your environment.)
        // Here, we assume the user also gave us the total #states per feature in e.g. statesMap.
        // We'll simply reconstruct the integer states_ array. The last entry is statesClass_.
        states_.resize(nFeatures_);
        for (int f = 0; f < nFeatures_; f++) {
            // Suppose you look up in “statesMap” by the feature name, or read directly from X.
            // We'll assume states_[f] = max value in X[f] + 1.
            states_[f] = dataset[f].max().item<int>() + 1;
        }
        // For the class: states_.back() = max(y)+1
        statesClass_ = dataset[-1].max().item<int>() + 1;

        // Initialize counts
        classCounts_.resize(statesClass_, 0.0);
        // p(x_sp = spVal | c)
        // We'll store these counts in spFeatureCounts_[spVal * statesClass_ + c].
        spFeatureCounts_.resize(states_[superParent_] * statesClass_, 0.0);

        // For each child ≠ sp, we store p(childVal| c, spVal) in a separate block of childCounts_.
        // childCounts_ will be sized as sum_{child≠sp} (states_[child] * statesClass_ * states_[sp]).
        // We also need an offset for each child to index into childCounts_.
        childOffsets_.resize(nFeatures_, -1);
        int totalSize = 0;
        for (int f = 0; f < nFeatures_; f++) {
            if (f == superParent_) continue; // skip sp
            childOffsets_[f] = totalSize;
            // block size for this child's counts: states_[f] * statesClass_ * states_[superParent_]
            totalSize += (states_[f] * statesClass_ * states_[superParent_]);
        }
        childCounts_.resize(totalSize, 0.0);
    }
    // --------------------------------------
    // buildModel
    // --------------------------------------
    //
    // We only store conditional probabilities for:
    //   p(x_sp| c)   (the super-parent feature)
    //   p(x_child| c, x_sp)  for all child ≠ sp
    //
    // --------------------------------------
    void XSpode::trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        // Accumulate raw counts
        for (int i = 0; i < m; i++) {
            std::vector<int> instance(nFeatures_ + 1);
            for (int f = 0; f < nFeatures_; f++) {
                instance[f] = dataset[f][i].item<int>();
            }
            instance[nFeatures_] = dataset[-1][i].item<int>();
            addSample(instance, weights[i].item<double>());
        }

        switch (smoothing) {
            case bayesnet::Smoothing_t::ORIGINAL:
                alpha_ = 1.0 / m;
                break;
            case bayesnet::Smoothing_t::LAPLACE:
                alpha_ = 1.0;
                break;
            default:
                alpha_ = 0.0; // No smoothing 
        }
        initializer_ = std::numeric_limits<double>::max() / (nFeatures_ * nFeatures_); // for numerical stability
        // Convert raw counts to probabilities
        computeProbabilities();
    }

    // --------------------------------------
    // addSample
    // --------------------------------------
    //
    // instance has size nFeatures_ + 1, with the class at the end.
    // We add 1 to the appropriate counters for each (c, superParentVal, childVal).
    //
    void XSpode::addSample(const std::vector<int>& instance, double weight)
    {
        if (weight <= 0.0) return;

        int c = instance.back();
        // (A) increment classCounts
        classCounts_[c] += weight;

        // (B) increment super-parent counts => p(x_sp | c)
        int spVal = instance[superParent_];
        spFeatureCounts_[spVal * statesClass_ + c] += weight;

        // (C) increment child counts => p(childVal | c, x_sp)
        for (int f = 0; f < nFeatures_; f++) {
            if (f == superParent_) continue;
            int childVal = instance[f];
            int offset = childOffsets_[f];
            // Compute index in childCounts_.
            // Layout: [ offset + (spVal * states_[f] + childVal) * statesClass_ + c ]
            int blockSize = states_[f] * statesClass_;
            int idx = offset + spVal * blockSize + childVal * statesClass_ + c;
            childCounts_[idx] += weight;
        }
    }

    // --------------------------------------
    // computeProbabilities
    // --------------------------------------
    //
    // Once all samples are added in COUNTS mode, call this to:
    //    p(c)
    //    p(x_sp = spVal | c)
    //    p(x_child = v | c, x_sp = s_sp)
    //
    // --------------------------------------
    void XSpode::computeProbabilities()
    {
        double totalCount = std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);

        // p(c) => classPriors_
        classPriors_.resize(statesClass_, 0.0);
        if (totalCount <= 0.0) {
            // fallback => uniform
            double unif = 1.0 / static_cast<double>(statesClass_);
            for (int c = 0; c < statesClass_; c++) {
                classPriors_[c] = unif;
            }
        } else {
            for (int c = 0; c < statesClass_; c++) {
                classPriors_[c] = (classCounts_[c] + alpha_)
                    / (totalCount + alpha_ * statesClass_);
            }
        }

        // p(x_sp | c)
        spFeatureProbs_.resize(spFeatureCounts_.size());
        // denominator for spVal * statesClass_ + c is just classCounts_[c] + alpha_ * (#states of sp)
        int spCard = states_[superParent_];
        for (int spVal = 0; spVal < spCard; spVal++) {
            for (int c = 0; c < statesClass_; c++) {
                double denom = classCounts_[c] + alpha_ * spCard;
                double num = spFeatureCounts_[spVal * statesClass_ + c] + alpha_;
                spFeatureProbs_[spVal * statesClass_ + c] = (denom <= 0.0 ? 0.0 : num / denom);
            }
        }

        // p(x_child | c, x_sp)
        childProbs_.resize(childCounts_.size());
        for (int f = 0; f < nFeatures_; f++) {
            if (f == superParent_) continue;
            int offset = childOffsets_[f];
            int childCard = states_[f];

            // For each spVal, c, childVal in childCounts_:
            for (int spVal = 0; spVal < spCard; spVal++) {
                for (int childVal = 0; childVal < childCard; childVal++) {
                    for (int c = 0; c < statesClass_; c++) {
                        int idx = offset + spVal * (childCard * statesClass_)
                            + childVal * statesClass_
                            + c;

                        double num = childCounts_[idx] + alpha_;
                        // denominator = spFeatureCounts_[spVal * statesClass_ + c] + alpha_ * (#states of child)
                        double denom = spFeatureCounts_[spVal * statesClass_ + c]
                            + alpha_ * childCard;
                        childProbs_[idx] = (denom <= 0.0 ? 0.0 : num / denom);
                    }
                }
            }
        }
    }

    // --------------------------------------
    // predict_proba
    // --------------------------------------
    //
    // For a single instance x of dimension nFeatures_:
    //  P(c | x) ∝ p(c) × p(x_sp | c) × ∏(child ≠ sp) p(x_child | c, x_sp).
    //
    // --------------------------------------
    std::vector<double> XSpode::predict_proba(const std::vector<int>& instance) const
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        std::vector<double> probs(statesClass_, 0.0);
        // Multiply p(c) × p(x_sp | c)
        int spVal = instance[superParent_];
        for (int c = 0; c < statesClass_; c++) {
            double pc = classPriors_[c];
            double pSpC = spFeatureProbs_[spVal * statesClass_ + c];
            probs[c] = pc * pSpC * initializer_;
        }

        // Multiply by each child’s probability p(x_child | c, x_sp)
        for (int feature = 0; feature < nFeatures_; feature++) {
            if (feature == superParent_) continue;  // skip sp
            int sf = instance[feature];
            int offset = childOffsets_[feature];
            int childCard = states_[feature]; // not used directly, but for clarity
            // Index into childProbs_ = offset + spVal*(childCard*statesClass_) + childVal*statesClass_ + c
            int base = offset + spVal * (childCard * statesClass_) + sf * statesClass_;
            for (int c = 0; c < statesClass_; c++) {
                probs[c] *= childProbs_[base + c];
            }
        }

        // Normalize
        normalize(probs);
        return probs;
    }
    std::vector<std::vector<double>> XSpode::predict_proba(std::vector<std::vector<int>>& test_data) 
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto probabilities = std::vector<std::vector<double>>(test_size, std::vector<double>(statesClass_));

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
                predictions[sample] = predict_proba(instance);
            }
            semaphore_.release();
            };
        for (int begin = 0; begin < test_size; begin += chunk_size) {
            int chunk = std::min(chunk_size, test_size - begin);
            semaphore_.acquire();
            threads.emplace_back(worker, test_data, begin, chunk, sample_size, std::ref(probabilities));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return probabilities;
    }

    // --------------------------------------
    // Utility: normalize
    // --------------------------------------
    void XSpode::normalize(std::vector<double>& v) const
    {
        double sum = 0.0;
        for (auto val : v) { sum += val; }
        if (sum <= 0.0) {
            return;
        }
        for (auto& val : v) {
            val /= sum;
        }
    }

    // --------------------------------------
    // representation of the model
    // --------------------------------------
    std::string XSpode::to_string() const
    {
        std::ostringstream oss;
        oss << "---- SPODE Model ----" << std::endl
            << "nFeatures_  = " << nFeatures_ << std::endl
            << "superParent_ = " << superParent_ << std::endl
            << "statesClass_ = " << statesClass_ << std::endl
            << std::endl;

        oss << "States: [";
        for (int s : states_) oss << s << " ";
        oss << "]" << std::endl;
        oss << "classCounts_: [";
        for (double c : classCounts_) oss << c << " ";
        oss << "]" << std::endl;
        oss << "classPriors_: [";
        for (double c : classPriors_) oss << c << " ";
        oss << "]" << std::endl;
        oss << "spFeatureCounts_: size = " << spFeatureCounts_.size() << std::endl << "[";
        for (double c : spFeatureCounts_) oss << c << " ";
        oss << "]" << std::endl;
        oss << "spFeatureProbs_: size = " << spFeatureProbs_.size() << std::endl << "[";
        for (double c : spFeatureProbs_) oss << c << " ";
        oss << "]" << std::endl;
        oss << "childCounts_: size = " << childCounts_.size() << std::endl << "[";
        for (double cc : childCounts_) oss << cc << " ";
        oss << "]" << std::endl;

        for (double cp : childProbs_) oss << cp << " ";
        oss << "]" << std::endl;
        oss << "childOffsets_: [";
        for (int co : childOffsets_) oss << co << " ";
        oss << "]" << std::endl;
        oss << "---------------------" << std::endl;
        return oss.str();
    }
    int XSpode::getNumberOfNodes() const { return nFeatures_ + 1; }
    int XSpode::getClassNumStates() const { return statesClass_; }
    int XSpode::getNFeatures() const { return nFeatures_; }
    int XSpode::getNumberOfStates() const
    {
        return std::accumulate(states_.begin(), states_.end(), 0) * nFeatures_;
    }
    int XSpode::getNumberOfEdges() const
    {
        return nFeatures_ * (2 * nFeatures_ - 1);
    }
    std::vector<int>& XSpode::getStates() { return states_; }

    // ------------------------------------------------------
    // Predict overrides (classifier interface)
    // ------------------------------------------------------
    int XSpode::predict(const std::vector<int>& instance) const
    {
        auto p = predict_proba(instance);
        return static_cast<int>(std::distance(p.begin(),
            std::max_element(p.begin(), p.end())));
    }
    std::vector<int> XSpode::predict(std::vector<std::vector<int>>& test_data)
    {
        auto probabilities = predict_proba(test_data);
        std::vector<int> predictions(probabilities.size(), 0);

        for (size_t i = 0; i < probabilities.size(); i++) {
            predictions[i] = std::distance(probabilities[i].begin(), std::max_element(probabilities[i].begin(), probabilities[i].end()));
        }

        return predictions;
    }
    torch::Tensor XSpode::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto result_v = predict(X_);
        torch::Tensor result;
        for (int i = 0; i < result_v.size(); ++i) {
            result.index_put_({ i, "..." }, torch::tensor(result_v[i], torch::kInt32));
        }
        return result;
    }
    torch::Tensor XSpode::predict_proba(torch::Tensor& X) 
    {
       auto X_ = TensorUtils::to_matrix(X);
       auto result_v = predict_proba(X_);
       torch::Tensor result;
       for (int i = 0; i < result_v.size(); ++i) {
           result.index_put_({ i, "..." }, torch::tensor(result_v[i], torch::kDouble));
       }
       return result;
    }
    torch::Tensor XSpode::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto predict = predict(X_);
        return TensorUtils::to_tensor(predict);
    }

}

