// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <cstdint>
#include <utility>
#include <vector>
#include <torch/torch.h>
namespace bayesnet {
    namespace detail {
        // Draws a uniform value in [0, range) from a 32 bit generator with Lemire's
        // nearly-divisionless method. std::uniform_int_distribution is only required
        // to be uniform, not to pick any particular value, so two standard libraries
        // give different draws from the same generator state.
        template <typename Generator>
        inline uint32_t bounded_rand(Generator& g, uint32_t range)
        {
            uint32_t x = static_cast<uint32_t>(g());
            uint64_t m = static_cast<uint64_t>(x) * static_cast<uint64_t>(range);
            uint32_t low = static_cast<uint32_t>(m);
            if (low < range) {
                uint32_t threshold = (0u - range) % range; // 2^32 % range
                while (low < threshold) {
                    x = static_cast<uint32_t>(g());
                    m = static_cast<uint64_t>(x) * static_cast<uint64_t>(range);
                    low = static_cast<uint32_t>(m);
                }
            }
            return static_cast<uint32_t>(m >> 32);
        }
    }
    // Fisher-Yates. std::shuffle cannot be used here: the standard leaves the
    // algorithm unspecified, so libstdc++ and libc++ build different permutations
    // out of the same seed, which made the "rand" feature and pair orders depend on
    // the platform. Doing it here means a seed selects the same order everywhere.
    // Deliberately the same algorithm as folding::detail::shuffle, which fixed the
    // identical problem for the folds.
    template <typename Iterator, typename Generator>
    inline void deterministicShuffle(Iterator first, Iterator last, Generator& g)
    {
        for (auto i = last - first; i > 1; --i) {
            auto j = static_cast<decltype(i)>(detail::bounded_rand(g, static_cast<uint32_t>(i)));
            if (j != i - 1) {
                std::swap(first[i - 1], first[j]);
            }
        }
    }
    std::vector<int> argsort(std::vector<double>& nums);
    std::vector<std::vector<double>> tensorToVectorDouble(torch::Tensor& dtensor);
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector, bool transpose = true);
}
#endif //BAYESNET_UTILS_H