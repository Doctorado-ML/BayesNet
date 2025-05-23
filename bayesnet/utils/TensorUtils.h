#ifndef TENSORUTILS_H
#define TENSORUTILS_H
#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    class TensorUtils {
    public:
        static std::vector<std::vector<int>> to_matrix(const torch::Tensor& X)
        {
            // Ensure tensor is contiguous in memory
            auto X_contig = X.contiguous();

            // Access tensor data pointer directly
            auto data_ptr = X_contig.data_ptr<int>();

            // IF you are using int64_t as the data type, use the following line
            //auto data_ptr = X_contig.data_ptr<int64_t>();
            //std::vector<std::vector<int64_t>> data(X.size(0), std::vector<int64_t>(X.size(1)));

            // Prepare output container
            std::vector<std::vector<int>> data(X.size(0), std::vector<int>(X.size(1)));

            // Fill the 2D vector in a single loop using pointer arithmetic
            int rows = X.size(0);
            int cols = X.size(1);
            for (int i = 0; i < rows; ++i) {
                std::copy(data_ptr + i * cols, data_ptr + (i + 1) * cols, data[i].begin());
            }
            return data;
        }
        template <typename T>
        static std::vector<T> to_vector(const torch::Tensor& y)
        {
            // Ensure the tensor is contiguous in memory
            auto y_contig = y.contiguous();

            // Access data pointer
            auto data_ptr = y_contig.data_ptr<T>();

            // Prepare output container
            std::vector<T> data(y.size(0));

            // Copy data efficiently
            std::copy(data_ptr, data_ptr + y.size(0), data.begin());

            return data;
        }
    };
}

#endif // TENSORUTILS_H