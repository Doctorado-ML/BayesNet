// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// Based on the Geoff. I. Webb A1DE java algorithm
// https://weka.sourceforge.io/packageMetaData/AnDE/Latest.html

#ifndef XAODE_H
#define XAODE_H
#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <torch/torch.h>
#include <bayesnet/network/Smoothing.h>


namespace bayesnet {
    class Xaode {
    public:
        // -------------------------------------------------------
        // The Xaode can be EMPTY (just created), in COUNTS mode (accumulating raw counts)
        // or PROBS mode (storing conditional probabilities).
        enum class MatrixState {
            EMPTY,
            COUNTS,
            PROBS
        };
        std::vector<double> significance_models_;
        Xaode() : nFeatures_{ 0 }, statesClass_{ 0 }, matrixState_{ MatrixState::EMPTY } {}
        // -------------------------------------------------------
        // fit
        // -------------------------------------------------------
        //
        // Classifiers interface
        // all parameter decide if the model is initialized with all the parents active or none of them
        //
        // states.size() = nFeatures + 1,
        //   where states.back() = number of class states.
        //
        // We'll store:
        //  1) p(x_i=si | c) in classFeatureProbs_
        //  2) p(x_j=sj | c, x_i=si) in data_, with i<j => i is "superparent," j is "child."
        //
        // Internally, in COUNTS mode, data_ accumulates raw counts, then
        // computeProbabilities(...) normalizes them into conditionals.
        void fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bool all_parents, const bayesnet::Smoothing_t smoothing)
        {
            int num_instances = X[0].size();
            nFeatures_ = X.size();

            significance_models_.resize(nFeatures_, (all_parents ? 1.0 : 0.0));
            for (int i = 0; i < nFeatures_; i++) {
                if (all_parents) active_parents.push_back(i);
                states_.push_back(*max_element(X[i].begin(), X[i].end()) + 1);
            }
            states_.push_back(*max_element(y.begin(), y.end()) + 1);
            //
            statesClass_ = states_.back();
            classCounts_.resize(statesClass_, 0.0);
            classPriors_.resize(statesClass_, 0.0);
            //
            // Initialize data structures
            //
            active_parents.resize(nFeatures_);
            int totalStates = std::accumulate(states_.begin(), states_.end(), 0) - statesClass_;

            // For p(x_i=si | c), we store them in a 1D array classFeatureProbs_ after we compute.
            // We'll need the offsets for each feature i in featureClassOffset_.
            featureClassOffset_.resize(nFeatures_);
            // We'll store p(x_child=sj | c, x_sp=si) for each pair (i<j).
            // So data_(i, si, j, sj, c) indexes into a big 1D array with an offset.
            // For p(x_i=si | c), we store them in a 1D array classFeatureProbs_ after we compute.
            // We'll need the offsets for each feature i in featureClassOffset_.
            featureClassOffset_.resize(nFeatures_);
            pairOffset_.resize(totalStates);
            int feature_offset = 0;
            int runningOffset = 0;
            int feature = 0, index = 0;
            for (int i = 0; i < nFeatures_; ++i) {
                featureClassOffset_[i] = feature_offset;
                feature_offset += states_[i];
                for (int j = 0; j < states_[i]; ++j) {
                    pairOffset_[feature++] = index;
                    index += runningOffset;
                }
                runningOffset += states_[i];
            }
            int totalSize = index * statesClass_;
            data_.resize(totalSize);
            dataOpp_.resize(totalSize);

            classFeatureCounts_.resize(feature_offset * statesClass_);
            classFeatureProbs_.resize(feature_offset * statesClass_);

            matrixState_ = MatrixState::COUNTS;
            //
            // Add samples
            //
            std::vector<int> instance(nFeatures_ + 1);
            for (int n_instance = 0; n_instance < num_instances; n_instance++) {
                for (int feature = 0; feature < nFeatures_; feature++) {
                    instance[feature] = X[feature][n_instance];
                }
                instance[nFeatures_] = y[n_instance];
                addSample(instance, weights[n_instance].item<double>());
            }
            switch (smoothing) {
                case bayesnet::Smoothing_t::ORIGINAL:
                    alpha_ = 1.0 / num_instances;
                    break;
                case bayesnet::Smoothing_t::LAPLACE:
                    alpha_ = 1.0;
                    break;
                default:
                    alpha_ = 0.0; // No smoothing 
            }
            initializer_ = std::numeric_limits<double>::max() / (nFeatures_ * nFeatures_);
            computeProbabilities();
        }
        std::string to_string() const
        {
            std::ostringstream ostream;
            ostream << "-------- Xaode.status --------" << std::endl
                << "- nFeatures = " << nFeatures_ << std::endl
                << "- statesClass = " << statesClass_ << std::endl
                << "- matrixState = " << (matrixState_ == MatrixState::COUNTS ? "COUNTS" : "PROBS") << std::endl;
            ostream << "- states: size: " << states_.size() << std::endl;
            for (int s : states_) ostream << s << " "; ostream << std::endl;
            ostream << "- classCounts: size: " << classCounts_.size() << std::endl;
            for (double cc : classCounts_) ostream << cc << " "; ostream << std::endl;
            ostream << "- classPriors: size: " << classPriors_.size() << std::endl;
            for (double cp : classPriors_) ostream << cp << " "; ostream << std::endl;
            ostream << "- classFeatureCounts: size: " << classFeatureCounts_.size() << std::endl;
            for (double cfc : classFeatureCounts_) ostream << cfc << " "; ostream << std::endl;
            ostream << "- classFeatureProbs: size: " << classFeatureProbs_.size() << std::endl;
            for (double cfp : classFeatureProbs_) ostream << cfp << " "; ostream << std::endl;
            ostream << "- featureClassOffset: size: " << featureClassOffset_.size() << std::endl;
            for (int f : featureClassOffset_) ostream << f << " "; ostream << std::endl;
            ostream << "- pairOffset_: size: " << pairOffset_.size() << std::endl;
            for (int p : pairOffset_) ostream << p << " "; ostream << std::endl;
            ostream << "- data: size: " << data_.size() << std::endl;
            for (double d : data_) ostream << d << " "; ostream << std::endl;
            ostream << "- dataOpp: size: " << dataOpp_.size() << std::endl;
            for (double d : dataOpp_) ostream << d << " "; ostream << std::endl;
            ostream << "--------------------------------" << std::endl;
            std::string output = ostream.str();
            return output;
        }
        // -------------------------------------------------------
        // addSample (only in COUNTS mode)
        // -------------------------------------------------------
        // 
        // instance should have the class at the end.
        // 
        void addSample(const std::vector<int>& instance, double weight)
        {
            //
            // (A) increment classCounts_
            // (B) increment feature–class counts => for p(x_i|c)
            // (C) increment pair (superparent= i, child= j) counts => data_ 
            //
            int c = instance.back();
            if (weight <= 0.0) {
                return;
            }
            // (A) increment classCounts_
            classCounts_[c] += weight;

            // (B,C)
            // We'll store raw counts now and turn them into p(child| c, superparent) later.
            int idx, fcIndex, sp, sc, i_offset;
            for (int parent = 0; parent < nFeatures_; ++parent) {
                sp = instance[parent];
                // (B) increment feature–class counts => for p(x_i|c)
                fcIndex = (featureClassOffset_[parent] + sp) * statesClass_ + c;
                classFeatureCounts_[fcIndex] += weight;
                // (C) increment pair (superparent= i, child= j) counts => data_
                i_offset = pairOffset_[featureClassOffset_[parent] + sp];
                for (int child = 0; child < parent; ++child) {
                    sc = instance[child];
                    idx = (i_offset + featureClassOffset_[child] + sc) * statesClass_ + c;
                    data_[idx] += weight;
                }
            }
        }
        // -------------------------------------------------------
        // computeProbabilities
        // -------------------------------------------------------
        //
        // Once all samples are added in COUNTS mode, call this to:
        //  1) compute p(c) => classPriors_
        //  2) compute p(x_i=si | c) => classFeatureProbs_
        //  3) compute p(x_j=sj | c, x_i=si) => data_ (for i<j) dataOpp_ (for i>j)
        //
        void computeProbabilities()
        {
            if (matrixState_ != MatrixState::COUNTS) {
                throw std::logic_error("computeProbabilities: must be in COUNTS mode.");
            }
            double totalCount = std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);
            // (1) p(c)
            if (totalCount <= 0.0) {
                // fallback => uniform
                double unif = 1.0 / statesClass_;
                for (int c = 0; c < statesClass_; ++c) {
                    classPriors_[c] = unif;
                }
            } else {
                for (int c = 0; c < statesClass_; ++c) {
                    classPriors_[c] = (classCounts_[c] + alpha_) / (totalCount + alpha_ * statesClass_);
                }
            }
            // (2) p(x_i=si | c) => classFeatureProbs_
            int idx, sf;
            double denom;
            for (int feature = 0; feature < nFeatures_; ++feature) {
                sf = states_[feature];
                for (int c = 0; c < statesClass_; ++c) {
                    denom = classCounts_[c] + alpha_ * sf;
                    for (int sf_value = 0; sf_value < sf; ++sf_value) {
                        idx = (featureClassOffset_[feature] + sf_value) * statesClass_ + c;
                        classFeatureProbs_[idx] = (classFeatureCounts_[idx] + alpha_) / denom;
                    }
                }
            }
            // getCountFromTable(int classVal, int pIndex, int childIndex)
            // (3) p(x_c=sc | c, x_p=sp) => data_(parent,sp,child,sc,c)
            // (3) p(x_p=sp | c, x_c=sc) => dataOpp_(child,sc,parent,sp,c)
            //                    C(x_c, x_p, c) + alpha_
            // P(x_p | x_c, c) = -----------------------------------
            //                           C(x_c, c) + alpha_
            double pcc_count, pc_count, cc_count;
            double conditionalProb, oppositeCondProb;
            int part1, part2, p1, part2_class, p1_class;
            for (int parent = 1; parent < nFeatures_; ++parent) {
                for (int sp = 0; sp < states_[parent]; ++sp) {
                    p1 = featureClassOffset_[parent] + sp;
                    part1 = pairOffset_[p1];
                    p1_class = p1 * statesClass_;
                    for (int child = 0; child < parent; ++child) {
                        for (int sc = 0; sc < states_[child]; ++sc) {
                            part2 = featureClassOffset_[child] + sc;
                            part2_class = part2 * statesClass_;
                            for (int c = 0; c < statesClass_; c++) {
                                idx = (part1 + part2) * statesClass_ + c;
                                // Parent, Child, Class Count
                                pcc_count = data_[idx];
                                // Parent, Class count
                                pc_count = classFeatureCounts_[p1_class + c];
                                // Child, Class count
                                cc_count = classFeatureCounts_[part2_class + c];
                                // p(x_c=sc | c, x_p=sp)
                                conditionalProb = (pcc_count + alpha_) / (pc_count + alpha_ * states_[child]);
                                data_[idx] = conditionalProb;
                                // p(x_p=sp | c, x_c=sc)
                                oppositeCondProb = (pcc_count + alpha_) / (cc_count + alpha_ * states_[parent]);
                                dataOpp_[idx] = oppositeCondProb;
                            }
                        }
                    }
                }
            }
            matrixState_ = MatrixState::PROBS;
        }
        // -------------------------------------------------------
        // predict_proba_spode
        // -------------------------------------------------------
        //
        // Single-superparent approach:
        // P(c | x) ∝ p(c) * p(x_sp| c) * ∏_{i≠sp} p(x_i | c, x_sp)
        //
        // 'instance' should have size == nFeatures_ (no class).
        // sp in [0..nFeatures_).
        // We multiply p(c) * p(x_sp| c) * p(x_i| c, x_sp).
        // Then normalize the distribution.
        //
        std::vector<double> predict_proba_spode(const std::vector<int>& instance, int parent)
        {
            // accumulates posterior probabilities for each class
            auto probs = std::vector<double>(statesClass_);
            auto spodeProbs = std::vector<double>(statesClass_, 0.0);
            if (std::find(active_parents.begin(), active_parents.end(), parent) == active_parents.end()) {
                return spodeProbs;
            }
            // Initialize the probabilities with the feature|class probabilities x class priors
            int localOffset;
            int sp = instance[parent];
            localOffset = (featureClassOffset_[parent] + sp) * statesClass_;
            for (int c = 0; c < statesClass_; ++c) {
                spodeProbs[c] = classFeatureProbs_[localOffset + c] * classPriors_[c] * initializer_;
            }
            int idx, base, sc, parent_offset;
            for (int child = 0; child < nFeatures_; ++child) {
                if (child == parent) {
                    continue;
                }
                sc = instance[child];
                if (child > parent) {
                    parent_offset = pairOffset_[featureClassOffset_[child] + sc];
                    base = (parent_offset + featureClassOffset_[parent] + sp) * statesClass_;
                } else {
                    parent_offset = pairOffset_[featureClassOffset_[parent] + sp];
                    base = (parent_offset + featureClassOffset_[child] + sc) * statesClass_;
                }
                for (int c = 0; c < statesClass_; ++c) {
                    /*
                    * The probability P(xc|xp,c) is stored in dataOpp_, and
                    * the probability P(xp|xc,c) is stored in data_
                    */
                    idx = base + c;
                    double factor = child > parent ? dataOpp_[idx] : data_[idx];
                    // double factor = data_[idx];
                    spodeProbs[c] *= factor;
                }
            }
            // Normalize the probabilities
            normalize(spodeProbs);
            return spodeProbs;
        }
        int predict_spode(const std::vector<int>& instance, int parent)
        {
            auto probs = predict_proba_spode(instance, parent);
            return (int)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        }
        // -------------------------------------------------------
        // predict_proba
        // -------------------------------------------------------
        //
        // P(c | x) ∝ p(c) * ∏_{i} p(x_i | c) * ∏_{i<j} p(x_j | c, x_i) * p(x_i | c, x_j)
        //
        // 'instance' should have size == nFeatures_ (no class).
        // We multiply p(c) * p(x_i| c) * p(x_j| c, x_i) for all i, j.
        // Then normalize the distribution.
        //
        std::vector<double> predict_proba(const std::vector<int>& instance)
        {
            // accumulates posterior probabilities for each class
            auto probs = std::vector<double>(statesClass_);
            auto spodeProbs = std::vector<std::vector<double>>(nFeatures_, std::vector<double>(statesClass_));
            // Initialize the probabilities with the feature|class probabilities
            int localOffset;
            for (int feature = 0; feature < nFeatures_; ++feature) {
                // if feature is not in the active_parents, skip it
                if (std::find(active_parents.begin(), active_parents.end(), feature) == active_parents.end()) {
                    continue;
                }
                localOffset = (featureClassOffset_[feature] + instance[feature]) * statesClass_;
                for (int c = 0; c < statesClass_; ++c) {
                    spodeProbs[feature][c] = classFeatureProbs_[localOffset + c] * classPriors_[c] * initializer_;
                }
            }
            int idx, base, sp, sc, parent_offset;
            for (int parent = 1; parent < nFeatures_; ++parent) {
                // if parent is not in the active_parents, skip it
                if (std::find(active_parents.begin(), active_parents.end(), parent) == active_parents.end()) {
                    continue;
                }
                sp = instance[parent];
                parent_offset = pairOffset_[featureClassOffset_[parent] + sp];
                for (int child = 0; child < parent; ++child) {
                    sc = instance[child];
                    if (child > parent) {
                        parent_offset = pairOffset_[featureClassOffset_[child] + sc];
                        base = (parent_offset + featureClassOffset_[parent] + sp) * statesClass_;
                    } else {
                        parent_offset = pairOffset_[featureClassOffset_[parent] + sp];
                        base = (parent_offset + featureClassOffset_[child] + sc) * statesClass_;
                    }
                    for (int c = 0; c < statesClass_; ++c) {
                        /*
                         * The probability P(xc|xp,c) is stored in dataOpp_, and
                         * the probability P(xp|xc,c) is stored in data_
                         */
                        idx = base + c;
                        double factor_child = child > parent ? data_[idx] : dataOpp_[idx];
                        double factor_parent = child > parent ? dataOpp_[idx] : data_[idx];
                        spodeProbs[child][c] *= factor_child;
                        spodeProbs[parent][c] *= factor_parent;
                    }
                }
            }
            /* add all the probabilities for each class */
            for (int c = 0; c < statesClass_; ++c) {
                for (int i = 0; i < nFeatures_; ++i) {
                    probs[c] += spodeProbs[i][c] * significance_models_[i];
                }
            }
            // Normalize the probabilities
            normalize(probs);
            return probs;
        }
        void normalize(std::vector<double>& probs) const
        {
            double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
            if (std::isnan(sum)) {
                throw std::runtime_error("Can't normalize array. Sum is NaN.");
            }
            if (sum == 0) {
                return;
            }
            for (int i = 0; i < (int)probs.size(); i++) {
                probs[i] /= sum;
            }
        }
        // Returns current mode: INIT, COUNTS or PROBS
        MatrixState state() const
        {
            return matrixState_;
        }
        int statesClass() const
        {
            return statesClass_;
        }
        int nFeatures() const
        {
            return nFeatures_;
        }
        int getNumberOfStates() const
        {
            return std::accumulate(states_.begin(), states_.end(), 0) * nFeatures_;
        }
        int getNumberOfEdges() const
        {
            return nFeatures_ * (2 * nFeatures_ - 1);
        }
        int getNumberOfNodes() const
        {
            return (nFeatures_ + 1) * nFeatures_;
        }
        void add_active_parent(int active_parent)
        {
            active_parents.push_back(active_parent);
        }
        void remove_last_parent()
        {
            active_parents.pop_back();
        }

    private:
        // -----------
        // MEMBER DATA
        // -----------
        std::vector<int> states_;            // [states_feat0, ..., states_feat(n-1), statesClass_]
        int nFeatures_;
        int statesClass_;

        // data_ means p(child=sj | c, superparent= si) after normalization.
        // But in COUNTS mode, it accumulates raw counts.
        std::vector<int> pairOffset_;
        // data_ stores p(child=sj | c, superparent=si) for each pair (i<j).
        std::vector<double> data_;
        // dataOpp_ stores p(superparent=si | c, child=sj) for each pair (i<j).
        std::vector<double> dataOpp_;

        // classCounts_[c]
        std::vector<double> classCounts_;
        std::vector<double> classPriors_;       // => p(c)

        // For p(x_i=si| c), we store counts in classFeatureCounts_ => offset by featureClassOffset_[i]
        std::vector<int> featureClassOffset_;
        std::vector<double> classFeatureCounts_;
        std::vector<double> classFeatureProbs_;  // => p(x_i=si | c) after normalization

        MatrixState matrixState_;

        double alpha_ = 1.0; // Laplace smoothing
        double initializer_ = 1.0;
        std::vector<int> active_parents;
    };
}
#endif // XAODE_H