// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include <algorithm>
#include <functional>  // std::cref
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "XSPODE.h"
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {

  // --------------------------------------
  // Constructor
  // --------------------------------------
  XSpode::XSpode(int spIndex)
    : superParent_{ spIndex }, nFeatures_{ 0 }, statesClass_{ 0 },
    semaphore_{ CountingSemaphore::getInstance() },
    Classifier(Network())
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

  void XSpode::fitx(torch::Tensor & X, torch::Tensor& y, torch::Tensor& weights_, const Smoothing_t smoothing)
  {
    m = X.size(1);
    n = X.size(0);
    dataset = X;
    buildDataset(y);
    buildModel(weights_);
    trainModel(weights_, smoothing);
    fitted = true;
  }

  // --------------------------------------
  // trainModel
  // --------------------------------------
  // Initialize storage needed for the super-parent and child features counts and
  // probs.
  // --------------------------------------
  void XSpode::buildModel(const torch::Tensor& weights)
  {
    int numInstances = m;
    nFeatures_ = n;

    // Derive the number of states for each feature and for the class.
    // (This is just one approach; adapt to match your environment.)
    // Here, we assume the user also gave us the total #states per feature in e.g.
    // statesMap. We'll simply reconstruct the integer states_ array. The last
    // entry is statesClass_.
    states_.resize(nFeatures_);
    for (int f = 0; f < nFeatures_; f++) {
      // Suppose you look up in “statesMap” by the feature name, or read directly
      // from X. We'll assume states_[f] = max value in X[f] + 1.
      states_[f] = dataset[f].max().item<int>() + 1;
    }
    // For the class: states_.back() = max(y)+1
    statesClass_ = dataset[-1].max().item<int>() + 1;

    // Initialize counts
    classCounts_.resize(statesClass_, 0.0);
    // p(x_sp = spVal | c)
    // We'll store these counts in spFeatureCounts_[spVal * statesClass_ + c].
    spFeatureCounts_.resize(states_[superParent_] * statesClass_, 0.0);

    // For each child ≠ sp, we store p(childVal| c, spVal) in a separate block of
    // childCounts_. childCounts_ will be sized as sum_{child≠sp} (states_[child]
    // * statesClass_ * states_[sp]). We also need an offset for each child to
    // index into childCounts_.
    childOffsets_.resize(nFeatures_, -1);
    int totalSize = 0;
    for (int f = 0; f < nFeatures_; f++) {
      if (f == superParent_)
        continue; // skip sp
      childOffsets_[f] = totalSize;
      // block size for this child's counts: states_[f] * statesClass_ *
      // states_[superParent_]
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
  void XSpode::trainModel(const torch::Tensor& weights,
    const bayesnet::Smoothing_t smoothing)
  {
    // Accumulate raw counts. Read through raw accessors: indexing the tensor
    // per value (dataset[f][i].item<int>()) costs an ATen dispatch each time,
    // m*n of them per fit, and the boosting loop refits a model every round.
    auto data = dataset.to(torch::kInt32).contiguous();
    auto dataData = data.accessor<int, 2>();
    auto weights_ = weights.to(torch::kFloat64).contiguous();
    auto weightsData = weights_.accessor<double, 1>();
    std::vector<int> instance(nFeatures_ + 1);
    for (int i = 0; i < m; i++) {
      for (int f = 0; f < nFeatures_; f++) {
        instance[f] = dataData[f][i];
      }
      instance[nFeatures_] = dataData[nFeatures_][i];
      addSample(instance, weightsData[i]);
    }
    // Store the smoothing strategy; the per-cell pseudocount is derived per table
    // from its cardinality (see smoothingPseudocount), which is what makes CESTNIK
    // a proper m-estimate rather than a single constant (previously CESTNIK fell
    // through to no smoothing).
    smoothing_ = smoothing;
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
    if (weight <= 0.0)
      return;

    int c = instance.back();
    // (A) increment classCounts
    classCounts_[c] += weight;

    // (B) increment super-parent counts => p(x_sp | c)
    int spVal = instance[superParent_];
    spFeatureCounts_[spVal * statesClass_ + c] += weight;

    // (C) increment child counts => p(childVal | c, x_sp)
    for (int f = 0; f < nFeatures_; f++) {
      if (f == superParent_)
        continue;
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
  // smoothingPseudocount
  // --------------------------------------
  // Per-cell pseudocount for an additive-smoothed table  P = (count + a)/(N + a*K),
  // with K = `cardinality` states of the distributed variable.
  //   ORIGINAL : a = 1/m       LAPLACE : a = 1
  //   CESTNIK  : a = 1/K       (m-estimate, m=1, uniform prior)
  //   otherwise: a = 0         (no smoothing)
  double XSpode::smoothingPseudocount(int cardinality) const
  {
    switch (smoothing_) {
      case bayesnet::Smoothing_t::ORIGINAL:
        return (m > 0) ? 1.0 / m : 0.0;
      case bayesnet::Smoothing_t::LAPLACE:
        return 1.0;
      case bayesnet::Smoothing_t::CESTNIK:
        return (cardinality > 0) ? 1.0 / cardinality : 0.0;
      default:
        return 0.0; // no smoothing
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
    double totalCount =
      std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);

    // p(c) => classPriors_
    classPriors_.resize(statesClass_, 0.0);
    if (totalCount <= 0.0) {
      // fallback => uniform
      double unif = 1.0 / static_cast<double>(statesClass_);
      for (int c = 0; c < statesClass_; c++) {
        classPriors_[c] = unif;
      }
    } else {
      double a = smoothingPseudocount(statesClass_);
      for (int c = 0; c < statesClass_; c++) {
        classPriors_[c] =
          (classCounts_[c] + a) / (totalCount + a * statesClass_);
      }
    }

    // p(x_sp | c)
    spFeatureProbs_.resize(spFeatureCounts_.size());
    // denominator for spVal * statesClass_ + c is just classCounts_[c] + a *
    // (#states of sp)
    int spCard = states_[superParent_];
    double aSp = smoothingPseudocount(spCard);
    for (int spVal = 0; spVal < spCard; spVal++) {
      for (int c = 0; c < statesClass_; c++) {
        double denom = classCounts_[c] + aSp * spCard;
        double num = spFeatureCounts_[spVal * statesClass_ + c] + aSp;
        spFeatureProbs_[spVal * statesClass_ + c] = (denom <= 0.0 ? 0.0 : num / denom);
      }
    }

    // p(x_child | c, x_sp)
    childProbs_.resize(childCounts_.size());
    for (int f = 0; f < nFeatures_; f++) {
      if (f == superParent_)
        continue;
      int offset = childOffsets_[f];
      int childCard = states_[f];
      double aChild = smoothingPseudocount(childCard);

      // For each spVal, c, childVal in childCounts_:
      for (int spVal = 0; spVal < spCard; spVal++) {
        for (int childVal = 0; childVal < childCard; childVal++) {
          for (int c = 0; c < statesClass_; c++) {
            int idx = offset + spVal * (childCard * statesClass_) +
              childVal * statesClass_ + c;

            double num = childCounts_[idx] + aChild;
            // denominator = spFeatureCounts_[spVal * statesClass_ + c] + aChild *
            // (#states of child)
            double denom =
              spFeatureCounts_[spVal * statesClass_ + c] + aChild * childCard;
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
    // Work in log-space (sum of log-probabilities + log-sum-exp normalization):
    // numerically stable, needs no magic scaling constant, and matches XSp2de.
    std::vector<double> logProbs(statesClass_, 0.0);
    // log p(c) + log p(x_sp | c)
    int spVal = instance[superParent_];
    for (int c = 0; c < statesClass_; c++) {
      double pc = classPriors_[c];
      double pSpC = spFeatureProbs_[spVal * statesClass_ + c];
      logProbs[c] = std::log(pc) + std::log(pSpC);
    }

    // + sum over child features of log p(x_child | c, x_sp)
    for (int feature = 0; feature < nFeatures_; feature++) {
      if (feature == superParent_)
        continue; // skip sp
      int sf = instance[feature];
      int offset = childOffsets_[feature];
      int childCard = states_[feature];
      // Index into childProbs_ = offset + spVal*(childCard*statesClass_) +
      // childVal*statesClass_ + c
      int base = offset + spVal * (childCard * statesClass_) + sf * statesClass_;
      for (int c = 0; c < statesClass_; c++) {
        logProbs[c] += std::log(childProbs_[base + c]);
      }
    }

    // Normalize with log-sum-exp.
    std::vector<double> probs(statesClass_, 0.0);
    double maxLog = *std::max_element(logProbs.begin(), logProbs.end());
    if (!std::isfinite(maxLog)) {
      // Every class has zero likelihood (only possible without smoothing) -> uniform.
      std::fill(probs.begin(), probs.end(), 1.0 / static_cast<double>(statesClass_));
      return probs;
    }
    double sumExp = 0.0;
    for (int c = 0; c < statesClass_; c++) {
      probs[c] = std::exp(logProbs[c] - maxLog);
      sumExp += probs[c];
    }
    for (int c = 0; c < statesClass_; c++) {
      probs[c] /= sumExp;
    }
    return probs;
  }
  std::vector<std::vector<double>> XSpode::predict_proba(std::vector<std::vector<int>>& test_data)
  {
    int test_size = test_data[0].size();
    int sample_size = test_data.size();
    auto probabilities = std::vector<std::vector<double>>(
      test_size, std::vector<double>(statesClass_));

    int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
    std::vector<std::thread> threads;
    auto worker = [&](const std::vector<std::vector<int>>& samples, int begin,
      int chunk, int sample_size,
      std::vector<std::vector<double>>& predictions) {
        std::string threadName =
          "(V)PWorker-" + std::to_string(begin) + "-" + std::to_string(chunk);
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
      // std::cref: without it std::thread copies the whole test set into every
      // chunk's argument tuple, i.e. O(m^2 * n) bytes over the m/150 chunks.
      threads.emplace_back(worker, std::cref(test_data), begin, chunk, sample_size, std::ref(probabilities));
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
    for (auto val : v) {
      sum += val;
    }
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
    oss << "----- XSpode Model -----" << std::endl
      << "nFeatures_  = " << nFeatures_ << std::endl
      << "superParent_ = " << superParent_ << std::endl
      << "statesClass_ = " << statesClass_ << std::endl
      << std::endl;

    oss << "States: [";
    for (int s : states_)
      oss << s << " ";
    oss << "]" << std::endl;
    oss << "classCounts_: [";
    for (double c : classCounts_)
      oss << c << " ";
    oss << "]" << std::endl;
    oss << "classPriors_: [";
    for (double c : classPriors_)
      oss << c << " ";
    oss << "]" << std::endl;
    oss << "spFeatureCounts_: size = " << spFeatureCounts_.size() << std::endl
      << "[";
    for (double c : spFeatureCounts_)
      oss << c << " ";
    oss << "]" << std::endl;
    oss << "spFeatureProbs_: size = " << spFeatureProbs_.size() << std::endl
      << "[";
    for (double c : spFeatureProbs_)
      oss << c << " ";
    oss << "]" << std::endl;
    oss << "childCounts_: size = " << childCounts_.size() << std::endl << "[";
    for (double cc : childCounts_)
      oss << cc << " ";
    oss << "]" << std::endl;

    for (double cp : childProbs_)
      oss << cp << " ";
    oss << "]" << std::endl;
    oss << "childOffsets_: [";
    for (int co : childOffsets_)
      oss << co << " ";
    oss << "]" << std::endl;
    oss << std::string(40,'-') << std::endl;
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
    return 2 * nFeatures_ + 1;
  }

  // ------------------------------------------------------
  // Predict overrides (classifier interface)
  // ------------------------------------------------------
  int XSpode::predict(const std::vector<int>& instance) const
  {
    auto p = predict_proba(instance);
    return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
  }
  std::vector<int> XSpode::predict(std::vector<std::vector<int>>& test_data)
  {
    auto probabilities = predict_proba(test_data);
    std::vector<int> predictions(probabilities.size(), 0);

    for (size_t i = 0; i < probabilities.size(); i++) {
      predictions[i] = std::distance(
        probabilities[i].begin(),
        std::max_element(probabilities[i].begin(), probabilities[i].end()));
    }
    return predictions;
  }
  torch::Tensor XSpode::predict(torch::Tensor& X)
  {
    auto X_ = TensorUtils::to_matrix(X);
    auto result_v = predict(X_);
    return torch::tensor(result_v, torch::kInt32);
  }
  torch::Tensor XSpode::predict_proba(torch::Tensor& X)
  {
    auto X_ = TensorUtils::to_matrix(X);
    auto result_v = predict_proba(X_);
    int n_samples = X.size(1);
    torch::Tensor result =
      torch::zeros({ n_samples, statesClass_ }, torch::kDouble);
    for (int i = 0; i < result_v.size(); ++i) {
      result.index_put_({ i, "..." }, torch::tensor(result_v[i]));
    }
    return result;
  }
  float XSpode::score(torch::Tensor& X, torch::Tensor& y)
  {
    torch::Tensor y_pred = predict(X);
    return (y_pred == y).sum().item<float>() / y.size(0);
  }
  float XSpode::score(std::vector<std::vector<int>>& X, std::vector<int>& y)
  {
    auto y_pred = this->predict(X);
    int correct = 0;
    for (int i = 0; i < y_pred.size(); ++i) {
      if (y_pred[i] == y[i]) {
        correct++;
      }
    }
    return (double)correct / y_pred.size();
  }
} // namespace bayesnet
