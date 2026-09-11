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
#include <algorithm>
#include <numeric>
#include <functional>  // std::cref
#include <stdexcept>
#include <type_traits>
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
  , smoothing_{bayesnet::Smoothing_t::ORIGINAL}
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
  // Accumulate raw counts. Read through raw accessors: indexing the tensor per
  // value (dataset[f][i].item<int>()) costs an ATen dispatch each time, m*n of
  // them per fit, and the boosting loop refits a model every round.
  auto data = dataset.to(torch::kInt32).contiguous();
  auto dataData = data.accessor<int, 2>();
  auto weights_ = weights.to(torch::kFloat64).contiguous();
  auto weightsData = weights_.accessor<double, 1>();
  std::vector<int> instance(nFeatures_ + 1);
  for (int i = 0; i < m; i++) {
    for (int f = 0; f < nFeatures_; f++) {
      instance[f] = dataData[f][i];
    }
    instance[nFeatures_] = dataData[nFeatures_][i];  // class
    addSample(instance, weightsData[i]);
  }

  // Store the smoothing strategy; the per-cell pseudocount is derived per table
  // from its cardinality (see smoothingPseudocount), which is what makes CESTNIK
  // a proper m-estimate rather than a single constant.
  smoothing_ = smoothing;

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
// smoothingPseudocount
// --------------------------------------
// Per-cell pseudocount for an additive-smoothed table  P = (count + a) / (N + a*K),
// where K = `cardinality` is the number of states of the distributed variable.
//   ORIGINAL : a = 1/m           (Laplace-like, m = #samples)
//   LAPLACE  : a = 1
//   CESTNIK  : a = 1/K           (m-estimate, m=1, uniform prior: the K
//                                 pseudocounts sum to 1, so denom adds exactly 1)
//   otherwise: a = 0             (no smoothing)
double XSp2de::smoothingPseudocount(int cardinality) const
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
    double a = smoothingPseudocount(statesClass_);
    for (int c = 0; c < statesClass_; c++) {
      classPriors_[c] =
        (classCounts_[c] + a)
        / (totalCount + a * statesClass_);
    }
  }

  // p(sp1Val| c)
  sp1FeatureProbs_.resize(sp1FeatureCounts_.size());
  int sp1Card = states_[superParent1_];
  double aSp1 = smoothingPseudocount(sp1Card);
  for (int spVal = 0; spVal < sp1Card; spVal++) {
    for (int c = 0; c < statesClass_; c++) {
      double denom = classCounts_[c] + aSp1 * sp1Card;
      double num = sp1FeatureCounts_[spVal * statesClass_ + c] + aSp1;
      sp1FeatureProbs_[spVal * statesClass_ + c] =
         (denom <= 0.0 ? 0.0 : num / denom);
    }
  }

  // p(sp2Val| c)
  sp2FeatureProbs_.resize(sp2FeatureCounts_.size());
  int sp2Card = states_[superParent2_];
  double aSp2 = smoothingPseudocount(sp2Card);
  for (int spVal = 0; spVal < sp2Card; spVal++) {
    for (int c = 0; c < statesClass_; c++) {
      double denom = classCounts_[c] + aSp2 * sp2Card;
      double num = sp2FeatureCounts_[spVal * statesClass_ + c] + aSp2;
      sp2FeatureProbs_[spVal * statesClass_ + c] =
         (denom <= 0.0 ? 0.0 : num / denom);
    }
  }

  // p(sp1Val, sp2Val | c) joint superparents
  spPairProbs_.resize(spPairCounts_.size());
  int pairCard = sp1Card * sp2Card;
  double aPair = smoothingPseudocount(pairCard);
  for (int v1 = 0; v1 < sp1Card; v1++) {
    for (int v2 = 0; v2 < sp2Card; v2++) {
      for (int c = 0; c < statesClass_; c++) {
        int idx = (v1 * sp2Card + v2) * statesClass_ + c;
        double num = spPairCounts_[idx] + aPair;
        double denom = classCounts_[c] + aPair * pairCard;
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
    double aChild = smoothingPseudocount(fCard);
    int childBlockSizeSp2 = sp2Card_ * fCard * statesClass_;
    int childBlockSizeF   = fCard * statesClass_;

    int blockSize = fCard * sp1Card_ * sp2Card_ * statesClass_;
    std::vector<double> sumSp1Sp2C(statesClass_);
    for (int sp1Val = 0; sp1Val < sp1Card_; sp1Val++) {
      for (int sp2Val = 0; sp2Val < sp2Card_; sp2Val++) {
        int base = offset
                 + sp1Val*childBlockSizeSp2
                 + sp2Val*childBlockSizeF;
        // The denominator is the count of (sp1Val,sp2Val,c), i.e. the child
        // block summed over childVal. It does not depend on childVal, so it is
        // computed once per (sp1Val,sp2Val) instead of once per cell -- that
        // inner re-summation made this loop O(fCard^2).
        std::fill(sumSp1Sp2C.begin(), sumSp1Sp2C.end(), 0.0);
        for (int cv = 0; cv < fCard; cv++) {
          for (int c = 0; c < statesClass_; c++) {
            sumSp1Sp2C[c] += childCounts_[base + cv*statesClass_ + c];
          }
        }
        for (int childVal = 0; childVal < fCard; childVal++) {
          for (int c = 0; c < statesClass_; c++) {
            // index in childCounts_
            int idx = base + childVal*statesClass_ + c;
            double num = childCounts_[idx] + aChild;
            double denom = sumSp1Sp2C[c] + aChild * fCard;
            childProbs_[idx] = (denom <= 0.0 ? 0.0 : num / denom);
          }
        }
      }
    }
    offset += blockSize;
  }

  // childCounts_ is dead from here on: every consumer downstream reads
  // childProbs_. It is the dominant block of the model (states_[sp1] *
  // states_[sp2] * statesClass_ * sum of the children cardinalities), so
  // releasing it roughly halves what a fitted model holds -- which is what
  // lets XBA2DE's memory budget fit about twice as many models.
  std::vector<double>().swap(childCounts_);
}

// --------------------------------------
// Memory accounting
// --------------------------------------
size_t XSp2de::memoryFootprint() const
{
  auto bytes = [](const auto &v) {
    return v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type);
  };
  return sizeof(*this)
       + bytes(states_) + bytes(childOffsets_)
       + bytes(classCounts_) + bytes(classPriors_)
       + bytes(sp1FeatureCounts_) + bytes(sp1FeatureProbs_)
       + bytes(sp2FeatureCounts_) + bytes(sp2FeatureProbs_)
       + bytes(spPairCounts_) + bytes(spPairProbs_)
       + bytes(childCounts_) + bytes(childProbs_);
}

size_t XSp2de::estimateFootprint(const std::vector<int> &states, int statesClass, int sp1, int sp2)
{
  const size_t nFeatures = states.size();
  if (sp1 < 0 || sp2 < 0 || static_cast<size_t>(sp1) >= nFeatures || static_cast<size_t>(sp2) >= nFeatures) {
    throw std::invalid_argument("XSp2de::estimateFootprint: superparent index out of range");
  }
  const size_t c = static_cast<size_t>(statesClass);
  const size_t s1 = static_cast<size_t>(states[sp1]);
  const size_t s2 = static_cast<size_t>(states[sp2]);

  // Sum of the cardinalities of the children, i.e. every feature but the two
  // superparents. The child tables are blocked per child, so their total size
  // is (s1 * s2 * c) times this sum.
  size_t childCardinalities = 0;
  for (size_t f = 0; f < nFeatures; ++f) {
    if (static_cast<int>(f) == sp1 || static_cast<int>(f) == sp2) continue;
    childCardinalities += static_cast<size_t>(states[f]);
  }

  const size_t d = sizeof(double);
  size_t total = sizeof(XSp2de);
  total += 2 * nFeatures * sizeof(int);        // states_, childOffsets_
  total += 2 * c * d;                          // classCounts_, classPriors_
  total += 2 * s1 * c * d;                     // sp1FeatureCounts_/Probs_
  total += 2 * s2 * c * d;                     // sp2FeatureCounts_/Probs_
  total += 2 * s1 * s2 * c * d;                // spPairCounts_/Probs_
  // Peak: childCounts_ and childProbs_ are both alive inside computeProbabilities.
  total += 2 * s1 * s2 * c * childCardinalities * d;
  return total;
}

// --------------------------------------
// predict_proba (single instance)
// --------------------------------------
std::vector<double> XSp2de::predict_proba(const std::vector<int> &instance) const
{
  if (!fitted) {
    throw std::logic_error(CLASSIFIER_NOT_FITTED);
  }
  // Work in log-space: the naive-Bayes product of many probabilities in [0,1]
  // underflows for high-dimensional instances (SP2DE's 4-way child tables are
  // very sparse). Summing log-probabilities and normalizing with log-sum-exp is
  // numerically stable and needs no magic scaling constant.
  std::vector<double> logProbs(statesClass_, 0.0);

  int sp1Val = instance[superParent1_];
  int sp2Val = instance[superParent2_];

  // log p(c) + log p(sp1Val, sp2Val | c).
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
    logProbs[c] = std::log(pC) + std::log(pParents);
  }

  // + sum over child features f of log p(x_f | c, sp1Val, sp2Val)
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
      logProbs[c] += std::log(childProbs_[base + c]);
    }
    offset += (fCard * sp1Card * sp2Card * statesClass_);
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
    // std::cref: without it std::thread copies the whole test set into every
    // chunk's argument tuple, i.e. O(m^2 * n) bytes over the m/150 chunks.
    threads.emplace_back(worker, std::cref(test_data), begin, chunk, sample_size,
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
  oss << "\nchildProbs_ (size=" << childProbs_.size() << ")\n";
  for (auto v : childProbs_) oss << v << " ";

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

