// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XA1DE.h"
#include "bayesnet/utils/TensorUtils.h"

namespace bayesnet {
    void XA1DE::trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        auto X = TensorUtils::to_matrix(dataset.slice(0, 0, dataset.size(0) - 1));
        auto y = TensorUtils::to_vector<int>(dataset.index({ -1, "..." }));
        int num_instances = X[0].size();
        weights_ = torch::full({ num_instances }, 1.0);
        //normalize_weights(num_instances);
        aode_.fit(X, y, features, className, states, weights_, true, smoothing);
    }
}
