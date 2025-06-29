// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <algorithm>
#include <cmath>
#include <numeric>
#include "bayesnet/utils/bayesnetUtils.h"
#include "L1FS.h"

namespace bayesnet {
    using namespace torch::indexing;

    L1FS::L1FS(const torch::Tensor& samples,
        const std::vector<std::string>& features,
        const std::string& className,
        const int maxFeatures,
        const int classNumStates,
        const torch::Tensor& weights,
        const double alpha,
        const int maxIter,
        const double tolerance,
        const bool fitIntercept)
        : FeatureSelect(samples, features, className, maxFeatures, classNumStates, weights),
        alpha(alpha), maxIter(maxIter), tolerance(tolerance), fitIntercept(fitIntercept)
    {
        if (alpha < 0) {
            throw std::invalid_argument("Alpha (regularization strength) must be non-negative");
        }
        if (maxIter < 1) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
        if (tolerance <= 0) {
            throw std::invalid_argument("Tolerance must be positive");
        }

        // Determine if this is a regression or classification task
        // For simplicity, assume binary classification if classNumStates == 2
        // and regression otherwise (this can be refined based on your needs)
        isRegression = (classNumStates > 2 || classNumStates == 0);
    }

    void L1FS::fit()
    {
        initialize();

        // Prepare data
        int n_samples = samples.size(1);
        int n_features = features.size();

        // Extract features (all rows except last)
        auto X = samples.index({ Slice(0, n_features), Slice() }).t().contiguous();

        // Extract labels (last row)
        auto y = samples.index({ -1, Slice() }).contiguous();

        // Convert to float for numerical operations
        X = X.to(torch::kFloat32);
        y = y.to(torch::kFloat32);

        // Normalize features for better convergence
        auto X_mean = X.mean(0);
        auto X_std = X.std(0);
        X_std = torch::where(X_std == 0, torch::ones_like(X_std), X_std);
        X = (X - X_mean) / X_std;

        if (isRegression) {
            // Normalize y for regression
            auto y_mean = y.mean();
            auto y_std = y.std();
            if (y_std.item<double>() > 0) {
                y = (y - y_mean) / y_std;
            }
            fitLasso(X, y, weights);
        } else {
            // For binary classification
            fitL1Logistic(X, y, weights);
        }

        // Select features based on non-zero coefficients
        std::vector<std::pair<int, double>> featureImportance;
        for (int i = 0; i < n_features; ++i) {
            double coef_magnitude = std::abs(coefficients[i]);
            if (coef_magnitude > 1e-10) {  // Threshold for numerical zero
                featureImportance.push_back({ i, coef_magnitude });
            }
        }

        // If all coefficients are zero (high regularization), select based on original feature-class correlation
        if (featureImportance.empty() && maxFeatures > 0) {
            // Compute SU with labels as fallback
            computeSuLabels();
            auto featureOrder = argsort(suLabels);

            // Select top features by SU score
            int numToSelect = std::min(static_cast<int>(featureOrder.size()),
                std::min(maxFeatures, 3)); // At most 3 features as fallback

            for (int i = 0; i < numToSelect; ++i) {
                selectedFeatures.push_back(featureOrder[i]);
                selectedScores.push_back(suLabels[featureOrder[i]]);
            }
        } else {
            // Sort by importance (absolute coefficient value)
            std::sort(featureImportance.begin(), featureImportance.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });

            // Select top features up to maxFeatures
            int numToSelect = std::min(static_cast<int>(featureImportance.size()),
                maxFeatures);

            for (int i = 0; i < numToSelect; ++i) {
                selectedFeatures.push_back(featureImportance[i].first);
                selectedScores.push_back(featureImportance[i].second);
            }
        }

        fitted = true;
    }

    void L1FS::fitLasso(const torch::Tensor& X, const torch::Tensor& y,
        const torch::Tensor& sampleWeights)
    {
        int n_samples = X.size(0);
        int n_features = X.size(1);

        // Initialize coefficients
        coefficients.resize(n_features, 0.0);
        double intercept = 0.0;

        // Ensure consistent types
        torch::Tensor weights = sampleWeights.to(torch::kFloat32);

        // Coordinate descent for Lasso
        torch::Tensor residuals = y.clone();
        if (fitIntercept) {
            intercept = (y * weights).sum().item<float>() / weights.sum().item<float>();
            residuals = y - intercept;
        }

        // Precompute feature norms
        std::vector<double> featureNorms(n_features);
        for (int j = 0; j < n_features; ++j) {
            auto Xj = X.index({ Slice(), j });
            featureNorms[j] = (Xj * Xj * weights).sum().item<float>();
        }

        // Coordinate descent iterations
        for (int iter = 0; iter < maxIter; ++iter) {
            double maxChange = 0.0;

            // Update each coordinate
            for (int j = 0; j < n_features; ++j) {
                auto Xj = X.index({ Slice(), j });

                // Compute partial residuals (excluding feature j)
                torch::Tensor partialResiduals = residuals + coefficients[j] * Xj;

                // Compute rho (correlation with residuals)
                double rho = (Xj * partialResiduals * weights).sum().item<float>();

                // Soft thresholding
                double oldCoef = coefficients[j];
                coefficients[j] = softThreshold(rho, alpha) / featureNorms[j];

                // Update residuals
                residuals = partialResiduals - coefficients[j] * Xj;

                maxChange = std::max(maxChange, std::abs(coefficients[j] - oldCoef));
            }

            // Update intercept if needed
            if (fitIntercept) {
                double oldIntercept = intercept;
                intercept = (residuals * weights).sum().item<float>() /
                    weights.sum().item<float>();
                residuals = residuals - (intercept - oldIntercept);
                maxChange = std::max(maxChange, std::abs(intercept - oldIntercept));
            }

            // Check convergence
            if (maxChange < tolerance) {
                break;
            }
        }
    }

    void L1FS::fitL1Logistic(const torch::Tensor& X, const torch::Tensor& y,
        const torch::Tensor& sampleWeights)
    {
        int n_samples = X.size(0);
        int n_features = X.size(1);

        // Initialize coefficients
        torch::Tensor coef = torch::zeros({ n_features }, torch::kFloat32);
        double intercept = 0.0;

        // Ensure consistent types
        torch::Tensor weights = sampleWeights.to(torch::kFloat32);

        // Learning rate (can be adaptive)
        double learningRate = 0.01;

        // Proximal gradient descent
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute predictions
            torch::Tensor linearPred = X.matmul(coef);
            if (fitIntercept) {
                linearPred = linearPred + intercept;
            }
            torch::Tensor pred = sigmoid(linearPred);

            // Compute gradient
            torch::Tensor diff = pred - y;
            torch::Tensor grad = X.t().matmul(diff * weights) / n_samples;

            // Gradient descent step
            torch::Tensor coef_new = coef - learningRate * grad;

            // Proximal step (soft thresholding)
            for (int j = 0; j < n_features; ++j) {
                coef_new[j] = softThreshold(coef_new[j].item<float>(),
                    learningRate * alpha);
            }

            // Update intercept if needed
            if (fitIntercept) {
                double grad_intercept = (diff * weights).sum().item<float>() / n_samples;
                intercept -= learningRate * grad_intercept;
            }

            // Check convergence
            double change = (coef_new - coef).abs().max().item<float>();
            coef = coef_new;

            if (change < tolerance) {
                break;
            }

            // Adaptive learning rate (optional)
            if (iter % 100 == 0) {
                learningRate *= 0.9;
            }
        }

        // Store final coefficients
        coefficients.resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            coefficients[j] = coef[j].item<float>();
        }
    }

    double L1FS::softThreshold(double x, double lambda) const
    {
        if (x > lambda) {
            return x - lambda;
        } else if (x < -lambda) {
            return x + lambda;
        } else {
            return 0.0;
        }
    }

    torch::Tensor L1FS::sigmoid(const torch::Tensor& z) const
    {
        return 1.0 / (1.0 + torch::exp(-z));
    }

    std::vector<double> L1FS::getCoefficients() const
    {
        if (!fitted) {
            throw std::runtime_error("L1FS not fitted");
        }
        return coefficients;
    }

} // namespace bayesnet