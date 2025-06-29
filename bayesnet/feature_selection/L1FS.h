// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef L1FS_H
#define L1FS_H
#include <torch/torch.h>
#include <vector>
#include "bayesnet/feature_selection/FeatureSelect.h"

namespace bayesnet {
    /**
     * L1-Regularized Feature Selection (L1FS)
     *
     * This class implements feature selection using L1-regularized linear models.
     * For classification tasks, it uses one-vs-rest logistic regression with L1 penalty.
     * For regression tasks, it uses Lasso regression.
     *
     * The L1 penalty induces sparsity in the model coefficients, effectively
     * performing feature selection by setting irrelevant feature weights to zero.
     */
    class L1FS : public FeatureSelect {
    public:
        /**
         * Constructor for L1FS
         * @param samples n+1xm tensor where samples[-1] is the target variable
         * @param features vector of feature names
         * @param className name of the class/target variable
         * @param maxFeatures maximum number of features to select (0 = all)
         * @param classNumStates number of states for classification (ignored for regression)
         * @param weights sample weights
         * @param alpha L1 regularization strength (higher = more sparsity)
         * @param maxIter maximum iterations for optimization
         * @param tolerance convergence tolerance
         * @param fitIntercept whether to fit an intercept term
         */
        L1FS(const torch::Tensor& samples,
            const std::vector<std::string>& features,
            const std::string& className,
            const int maxFeatures,
            const int classNumStates,
            const torch::Tensor& weights,
            const double alpha = 1.0,
            const int maxIter = 1000,
            const double tolerance = 1e-4,
            const bool fitIntercept = true);

        virtual ~L1FS() {};

        void fit() override;

        // Get the learned coefficients for each feature
        std::vector<double> getCoefficients() const;

    private:
        double alpha;        // L1 regularization strength
        int maxIter;         // Maximum iterations for optimization
        double tolerance;    // Convergence tolerance
        bool fitIntercept;   // Whether to fit intercept
        bool isRegression;   // Task type (regression vs classification)

        std::vector<double> coefficients;  // Learned coefficients

        // Coordinate descent for Lasso regression
        void fitLasso(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& sampleWeights);

        // Proximal gradient descent for L1-regularized logistic regression
        void fitL1Logistic(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& sampleWeights);

        // Soft thresholding operator for L1 regularization
        double softThreshold(double x, double lambda) const;

        // Logistic function
        torch::Tensor sigmoid(const torch::Tensor& z) const;

        // Compute logistic loss
        double logisticLoss(const torch::Tensor& X, const torch::Tensor& y,
            const torch::Tensor& coef, const torch::Tensor& sampleWeights) const;
    };
}
#endif