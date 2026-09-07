// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************


#include "bayesnetUtils.h"
namespace bayesnet {
    // Return the indices in descending order.
    // Equal scores are ordered by index. Without that the comparator is not a
    // total order, and the standard leaves the relative order of equivalent
    // elements to std::sort's implementation: libstdc++ and libc++ then rank
    // tied features differently and the selection diverges by platform. Exact
    // ties are common here (many pairs have a mutual information of exactly 0).
    std::vector<int> argsort(std::vector<double>& nums)
    {
        int n = nums.size();
        std::vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {
            if (nums[i] != nums[j]) return nums[i] > nums[j];
            return i < j;
            });
        return indices;
    }
    std::vector<std::vector<double>> tensorToVectorDouble(torch::Tensor& dtensor)
    {
        // convert mxn tensor to mxn std::vector
        std::vector<std::vector<double>> result;
        // Iterate over cols
        for (int i = 0; i < dtensor.size(0); ++i) {
            auto col_tensor = dtensor.index({ i, "..." });
            auto col = std::vector<double>(col_tensor.data_ptr<float>(), col_tensor.data_ptr<float>() + dtensor.size(1));
            result.push_back(col);
        }
        return result;
    }
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector, bool transpose)
    {
        // convert nxm std::vector to mxn tensor if transpose
        long int m = transpose ? vector[0].size() : vector.size();
        long int n = transpose ? vector.size() : vector[0].size();
        auto tensor = torch::zeros({ m, n }, torch::kInt32);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                tensor[i][j] = transpose ? vector[j][i] : vector[i][j];
            }
        }
        return tensor;
    }
}