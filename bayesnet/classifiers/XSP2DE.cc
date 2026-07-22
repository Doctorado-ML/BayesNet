// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XSP2DE.h"
#include <pthread.h>   // for pthread_setname_np on linux
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {

// --------------------------------------
// Constructor
// --------------------------------------
XSp2de::XSp2de(int spIndex1, int spIndex2)
  : superParent1_{ spIndex1 }
  , superParent2_{ spIndex2 }
  , nFeatures_{0}
  , statesClass_{0}
  , alpha_{1.0}
  , initializer_{1.0}
  , jointParents_{true}
  , semaphore_{ CountingSemaphore::getInstance() }
  , Classifier(Network())
{
  validHyperparameters = { "parent1", "parent2", "joint_parents" };
}

// --------------------------------------
// setHyperparameters
// --------------------------------------
void XSp2de::setHyperparameters(const nlohmann::json &hyperparameters_)
{
  auto hyperparameters = hyperparameters_;
  if (hyperparameters.contains("parent1")) {
    superParent1_ = hyperparameters["parent1"];
    hyperparameters.erase("parent1");
  }
  if (hyperparameters.contains("parent2")) {
    superParent2_ = hyperparameters["parent2"];
    hyperparameters.erase("parent2");
  }
  if (hyperparameters.contains("joint_parents")) {
    jointParents_ = hyperparameters["joint_parents"];
    hyperparameters.erase("joint_parents");
  }
  // Hand off anything else to base Classifier
  Classifier::setHyperparameters(hyperparameters);
}

// --------------------------------------
// fitx
// --------------------------------------
void XSp2de::fitx(torch::Tensor & X, torch::Tensor & y, 
                  torch::Tensor & weights_, const Smoothing_t smoothing)
{
  m = X.size(1);  // number of samples
  n = X.size(0);  // number of features
  dataset = X;

  // Build the dataset in your environment if needed:
  buildDataset(y);

  // Construct the data structures needed for counting
  buildModel(weights_);

  // Accumulate counts & convert to probabilities
  trainModel(weights_, smoothing);
  fitted = true;
}

// --------------------------------------
// buildModel
// --------------------------------------
void XSp2de::buildModel(const torch::Tensor &weights)
{
  nFeatures_ = n;

  // Derive the number of states for each feature from the dataset
  // states_[f] = max value in dataset[f] + 1.
  states_.resize(nFeatures_);
  for (int f = 0; f < nFeatures_; f++) {
    // This is naive: we take max in feature f. You might adapt for real data.
    states_[f] = dataset[f].max().item<int>() + 1;
  }
  // Class states:
  statesClass_ = dataset[-1].max().item<int>() + 1;

  // Initialize the class counts
  classCounts_.resize(statesClass_, 0.0);

  // For sp1 -> p(sp1Val| c)
  sp1FeatureCounts_.resize(states_[superParent1_] * statesClass_, 0.0);

  // For sp2 -> p(sp2Val| c)
  sp2FeatureCounts_.resize(states_[superParent2_] * statesClass_, 0.0);

  // Joint superparents -> p(sp1Val, sp2Val | c)
  spPairCounts_.resize(states_[superParent1_] * states_[superParent2_] * statesClass_, 0.0);

  // For child features, we store p(childVal | c, sp1Val, sp2Val).
  // childCounts_ will hold raw counts. We’ll gather them in one big vector.
  // We need an offset for each feature.
  childOffsets_.resize(nFeatures_, -1);

  int totalSize = 0;
  for (int f = 0; f < nFeatures_; f++) {
    if (f == superParent1_ || f == superParent2_) {
      // skip the superparents
      childOffsets_[f] = -1;
      continue;
    }
    childOffsets_[f] = totalSize;
    // block size for a single child f: states_[f] * statesClass_ 
    //                               * states_[superParent1_] 
    //                               * states_[superParent2_].
    totalSize += (states_[f] * statesClass_ 
                  * states_[superParent1_] 
                  * states_[superParent2_]);
  }
  childCounts_.resize(totalSize, 0.0);
}

// --------------------------------------
// trainModel
// --------------------------------------
void XSp2de::trainModel(const torch::Tensor &weights, 
                        const bayesnet::Smoothing_t smoothing)
{
  // Accumulate raw counts
  for (int i = 0; i < m; i++) {
    std::vector<int> instance(nFeatures_ + 1);
    for (int f = 0; f < nFeatures_; f++) {
      instance[f] = dataset[f][i].item<int>();
    }
    instance[nFeatures_] = dataset[-1][i].item<int>();  // class
    double w = weights[i].item<double>();
    addSample(instance, w);
  }

  // Choose alpha based on smoothing:
  switch (smoothing) {
    case bayesnet::Smoothing_t::ORIGINAL:
      alpha_ = 1.0 / m;
      break;
    case bayesnet::Smoothing_t::LAPLACE:
      alpha_ = 1.0;
      break;
    default:
      alpha_ = 0.0; // no smoothing
  }

  // Large initializer factor for numerical stability
  initializer_ = std::numeric_limits<double>::max() / (nFeatures_ * nFeatures_);

  // Convert raw counts to probabilities
  computeProbabilities();
}

// --------------------------------------
// addSample
// --------------------------------------
void XSp2de::addSample(const std::vector<int> &instance, double weight)
{
  if (weight <= 0.0)
    return;

  int c = instance.back();
  // increment classCounts
  classCounts_[c] += weight;

  int sp1Val = instance[superParent1_];
  int sp2Val = instance[superParent2_];

  // p(sp1|c)
  sp1FeatureCounts_[sp1Val * statesClass_ + c] += weight;

  // p(sp2|c)
  sp2FeatureCounts_[sp2Val * statesClass_ + c] += weight;

  // p(sp1,sp2|c) joint superparents
  spPairCounts_[(sp1Val * states_[superParent2_] + sp2Val) * statesClass_ + c] += weight;

  // p(childVal| c, sp1Val, sp2Val)
  for (int f = 0; f < nFeatures_; f++) {
    if (f == superParent1_ || f == superParent2_)
      continue;

    int childVal = instance[f];
    int offset = childOffsets_[f];
    // block layout: 
    //    offset + (sp1Val*(states_[sp2_]* states_[f]* statesClass_)) 
    //            + (sp2Val*(states_[f]* statesClass_)) 
    //            + childVal*(statesClass_) 
    //            + c
    int blockSizeSp2 = states_[superParent2_] 
                       * states_[f] 
                       * statesClass_;
    int blockSizeChild = states_[f] * statesClass_;

    int idx = offset 
            + sp1Val*blockSizeSp2 
            + sp2Val*blockSizeChild 
            + childVal*statesClass_ 
            + c;
    childCounts_[idx] += weight;
  }
}

// --------------------------------------
// computeProbabilities
// --------------------------------------
void XSp2de::computeProbabilities()
{
  double totalCount = std::accumulate(classCounts_.begin(), 
                                      classCounts_.end(), 0.0);

  // classPriors_
  classPriors_.resize(statesClass_, 0.0);
  if (totalCount <= 0.0) {
    // fallback => uniform
    double unif = 1.0 / static_cast<double>(statesClass_);
    for (int c = 0; c < statesClass_; c++) {
      classPriors_[c] = unif;
    }
  } else {
    for (int c = 0; c < statesClass_; c++) {
      classPriors_[c] = 
        (classCounts_[c] + alpha_) 
        / (totalCount + alpha_ * statesClass_);
    }
  }

  // p(sp1Val| c)
  sp1FeatureProbs_.resize(sp1FeatureCounts_.size());
  int sp1Card = states_[superParent1_];
  for (int spVal = 0; spVal < sp1Card; spVal++) {
    for (int c = 0; c < statesClass_; c++) {
      double denom = classCounts_[c] + alpha_ * sp1Card;
      double num = sp1FeatureCounts_[spVal * statesClass_ + c] + alpha_;
      sp1FeatureProbs_[spVal * statesClass_ + c] = 
         (denom <= 0.0 ? 0.0 : num / denom);
    }
  }

  // p(sp2Val| c)
  sp2FeatureProbs_.resize(sp2FeatureCounts_.size());
  int sp2Card = states_[superParent2_];
  for (int spVal = 0; spVal < sp2Card; spVal++) {
    for (int c = 0; c < statesClass_; c++) {
      double denom = classCounts_[c] + alpha_ * sp2Card;
      double num = sp2FeatureCounts_[spVal * statesClass_ + c] + alpha_;
      sp2FeatureProbs_[spVal * statesClass_ + c] =
         (denom <= 0.0 ? 0.0 : num / denom);
    }
  }

  // p(sp1Val, sp2Val | c) joint superparents
  spPairProbs_.resize(spPairCounts_.size());
  int pairCard = sp1Card * sp2Card;
  for (int v1 = 0; v1 < sp1Card; v1++) {
    for (int v2 = 0; v2 < sp2Card; v2++) {
      for (int c = 0; c < statesClass_; c++) {
        int idx = (v1 * sp2Card + v2) * statesClass_ + c;
        double num = spPairCounts_[idx] + alpha_;
        double denom = classCounts_[c] + alpha_ * pairCard;
        spPairProbs_[idx] = (denom <= 0.0 ? 0.0 : num / denom);
      }
    }
  }

  // p(childVal| c, sp1Val, sp2Val)
  childProbs_.resize(childCounts_.size());
  int offset = 0;
  for (int f = 0; f < nFeatures_; f++) {
    if (f == superParent1_ || f == superParent2_) 
      continue;

    int fCard = states_[f];
    int sp1Card_ = states_[superParent1_];
    int sp2Card_ = states_[superParent2_];
    int childBlockSizeSp2 = sp2Card_ * fCard * statesClass_;
    int childBlockSizeF   = fCard * statesClass_;

    int blockSize = fCard * sp1Card_ * sp2Card_ * statesClass_;
    for (int sp1Val = 0; sp1Val < sp1Card_; sp1Val++) {
      for (int sp2Val = 0; sp2Val < sp2Card_; sp2Val++) {
        for (int childVal = 0; childVal < fCard; childVal++) {
          for (int c = 0; c < statesClass_; c++) {
            // index in childCounts_ 
            int idx = offset 
                    + sp1Val*childBlockSizeSp2 
                    + sp2Val*childBlockSizeF 
                    + childVal*statesClass_ 
                    + c;
            double num = childCounts_[idx] + alpha_;
            // denominator is the count of (sp1Val,sp2Val,c) plus alpha * fCard
            // We can find that by summing childVal dimension, but we already
            // have it in childCounts_[...] or we can re-check the superparent 
            // counts if your approach is purely hierarchical. 
            // Here we'll do it like the XSpode approach: sp1&sp2 are 
            // conditionally independent given c, so denominators come from 
            // summing the relevant block or we treat sp1,sp2 as "parents."
            // A simpler approach: 
            double sumSp1Sp2C = 0.0;
            // sum over all childVal:
            for (int cv = 0; cv < fCard; cv++) {
              int idx2 = offset
                       + sp1Val*childBlockSizeSp2
                       + sp2Val*childBlockSizeF
                       + cv*statesClass_ + c;
              sumSp1Sp2C += childCounts_[idx2];
            }
            double denom = sumSp1Sp2C + alpha_ * fCard;
            childProbs_[idx] = (denom <= 0.0 ? 0.0 : num / denom);
          }
        }
      }
    }
    offset += blockSize;
  }
}

// --------------------------------------
// predict_proba (single instance)
// --------------------------------------
std::vector<double> XSp2de::predict_proba(const std::vector<int> &instance) const
{
  if (!fitted) {
    throw std::logic_error(CLASSIFIER_NOT_FITTED);
  }
  std::vector<double> probs(statesClass_, 0.0);

  int sp1Val = instance[superParent1_];
  int sp2Val = instance[superParent2_];

  // Start with p(c) * p(sp1Val, sp2Val | c).
  // jointParents_: model the two superparents jointly (classic A2DE).
  // else: legacy behaviour, treat them as independent given c.
  for (int c = 0; c < statesClass_; c++) {
    double pC = classPriors_[c];
    double pParents;
    if (jointParents_) {
      pParents = spPairProbs_[(sp1Val * states_[superParent2_] + sp2Val) * statesClass_ + c];
    } else {
      double pSp1C = sp1FeatureProbs_[sp1Val * statesClass_ + c];
      double pSp2C = sp2FeatureProbs_[sp2Val * statesClass_ + c];
      pParents = pSp1C * pSp2C;
    }
    probs[c] = pC * pParents * initializer_;
  }

  // Multiply by each child feature f
  int offset = 0;
  for (int f = 0; f < nFeatures_; f++) {
    if (f == superParent1_ || f == superParent2_) 
      continue;

    int valF = instance[f];
    int fCard = states_[f];
    int sp1Card = states_[superParent1_];
    int sp2Card = states_[superParent2_];
    int blockSizeSp2 = sp2Card * fCard * statesClass_;
    int blockSizeF   = fCard * statesClass_;

    // base index for childProbs_ for this child and sp1Val, sp2Val
    int base = offset 
             + sp1Val*blockSizeSp2 
             + sp2Val*blockSizeF 
             + valF*statesClass_;
    for (int c = 0; c < statesClass_; c++) {
      probs[c] *= childProbs_[base + c];
    }
    offset += (fCard * sp1Card * sp2Card * statesClass_);
  }

  // Normalize
  normalize(probs);
  return probs;
}

// --------------------------------------
// predict_proba (batch)
// --------------------------------------
std::vector<std::vector<double>> XSp2de::predict_proba(std::vector<std::vector<int>> &test_data)
{
  int test_size = test_data[0].size();  // each feature is test_data[f], size = #samples
  int sample_size = test_data.size();   // = nFeatures_
  std::vector<std::vector<double>> probabilities(
      test_size, std::vector<double>(statesClass_, 0.0));

  // same concurrency approach
  int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
  std::vector<std::thread> threads;

  auto worker = [&](const std::vector<std::vector<int>> &samples, 
                    int begin, 
                    int chunk, 
                    int sample_size, 
                    std::vector<std::vector<double>> &predictions) {
    std::string threadName =
      "XSp2de-" + std::to_string(begin) + "-" + std::to_string(chunk);
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
    threads.emplace_back(worker, test_data, begin, chunk, sample_size, 
                         std::ref(probabilities));
  }
  for (auto &th : threads) {
    th.join();
  }
  return probabilities;
}

// --------------------------------------
// predict (single instance)
// --------------------------------------
int XSp2de::predict(const std::vector<int> &instance) const
{
  auto p = predict_proba(instance);
  return static_cast<int>(
    std::distance(p.begin(), std::max_element(p.begin(), p.end()))
  );
}

// --------------------------------------
// predict (batch of data)
// --------------------------------------
std::vector<int> XSp2de::predict(std::vector<std::vector<int>> &test_data)
{
  auto probabilities = predict_proba(test_data);
  std::vector<int> predictions(probabilities.size(), 0);

  for (size_t i = 0; i < probabilities.size(); i++) {
    predictions[i] = static_cast<int>(
      std::distance(probabilities[i].begin(), 
                    std::max_element(probabilities[i].begin(), 
                                     probabilities[i].end()))
    );
  }
  return predictions;
}

// --------------------------------------
// predict (torch::Tensor version)
// --------------------------------------
torch::Tensor XSp2de::predict(torch::Tensor &X)
{
  auto X_ = TensorUtils::to_matrix(X);
  auto result_v = predict(X_);
  return torch::tensor(result_v, torch::kInt32);
}

// --------------------------------------
// predict_proba (torch::Tensor version)
// --------------------------------------
torch::Tensor XSp2de::predict_proba(torch::Tensor &X)
{
  auto X_ = TensorUtils::to_matrix(X);
  auto result_v = predict_proba(X_);
  int n_samples = X.size(1);
  torch::Tensor result =
    torch::zeros({ n_samples, statesClass_ }, torch::kDouble);
  for (int i = 0; i < (int)result_v.size(); ++i) {
    result.index_put_({ i, "..." }, torch::tensor(result_v[i]));
  }
  return result;
}

// --------------------------------------
// score (torch::Tensor version)
// --------------------------------------
float XSp2de::score(torch::Tensor &X, torch::Tensor &y)
{
  torch::Tensor y_pred = predict(X);
  return (y_pred == y).sum().item<float>() / y.size(0);
}

// --------------------------------------
// score (vector version)
// --------------------------------------
float XSp2de::score(std::vector<std::vector<int>> &X, std::vector<int> &y)
{
  auto y_pred = predict(X);
  int correct = 0;
  for (size_t i = 0; i < y_pred.size(); ++i) {
    if (y_pred[i] == y[i]) {
      correct++;
    }
  }
  return static_cast<float>(correct) / static_cast<float>(y_pred.size());
}

// --------------------------------------
// Utility: normalize
// --------------------------------------
void XSp2de::normalize(std::vector<double> &v) const
{
  double sum = 0.0;
  for (auto &val : v) {
    sum += val;
  }
  if (sum > 0.0) {
    for (auto &val : v) {
      val /= sum;
    }
  }
}

// --------------------------------------
// to_string
// --------------------------------------
std::string XSp2de::to_string() const
{
  std::ostringstream oss;
  oss << "----- XSp2de Model -----\n"
      << "nFeatures_    = " << nFeatures_    << "\n"
      << "superParent1_ = " << superParent1_ << "\n"
      << "superParent2_ = " << superParent2_ << "\n"
      << "jointParents_ = " << jointParents_ << "\n"
      << "statesClass_  = " << statesClass_  << "\n\n";

  oss << "States: [";
  for (auto s : states_) oss << s << " ";
  oss << "]\n";

  oss << "classCounts_:\n";
  for (auto v : classCounts_) oss << v << " ";
  oss << "\nclassPriors_:\n";
  for (auto v : classPriors_) oss << v << " ";
  oss << "\nsp1FeatureCounts_ (size=" << sp1FeatureCounts_.size() << ")\n";
  for (auto v : sp1FeatureCounts_) oss << v << " ";
  oss << "\nsp2FeatureCounts_ (size=" << sp2FeatureCounts_.size() << ")\n";
  for (auto v : sp2FeatureCounts_) oss << v << " ";
  oss << "\nchildCounts_ (size=" << childCounts_.size() << ")\n";
  for (auto v : childCounts_) oss << v << " ";

  oss << "\nchildOffsets_:\n";
  for (auto c : childOffsets_) oss << c << " ";

  oss << "\n----------------------------------------\n";
  return oss.str();
}

// --------------------------------------
// Some introspection about the graph
// --------------------------------------
int XSp2de::getNumberOfNodes() const 
{
  // nFeatures + 1 class node
  return nFeatures_ + 1;
}

int XSp2de::getClassNumStates() const 
{ 
  return statesClass_; 
}

int XSp2de::getNFeatures() const 
{ 
  return nFeatures_; 
}

int XSp2de::getNumberOfStates() const
{
  // purely an example. Possibly you want to sum up actual 
  // cardinalities or something else. 
  return std::accumulate(states_.begin(), states_.end(), 0) * nFeatures_;
}

int XSp2de::getNumberOfEdges() const
{
  //   - class -> each of the nFeatures nodes            => nFeatures edges
  //   - sp1 -> child, sp2 -> child (nFeatures-2 childs) => 2*(nFeatures-2) edges
  //   => nFeatures + 2*(nFeatures-2) = 3*nFeatures - 4
  //   - jointParents_: one extra edge sp1 -> sp2 (the two superparents are
  //     modelled jointly, so they are dependent given the class).
  return 3 * nFeatures_ - 4 + (jointParents_ ? 1 : 0);
}

} // namespace bayesnet

