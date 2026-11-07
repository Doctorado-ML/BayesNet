// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
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
    // Number of states of a variable in the probability tables of the X family
    // (Xaode, Xaode2de, XSpode, XSp2de). The `states` map handed to fit lists every
    // value the variable can take, in training and in test (Platform builds it from
    // both partitions), so its size is the cardinality to use: a value that never
    // occurs in the training data still needs its own, smoothed, cell when it shows
    // up at prediction time. Sizing the tables with the largest training value + 1
    // instead made predict_proba read out of range for such values (BayesNet#47).
    // `maxSeen` (largest training value) is the lower bound, used as such when the
    // map lacks the variable or lists fewer values (e.g. an empty entry for a
    // continuous feature).
    inline int cardinality(const std::map<std::string, std::vector<int>>& states, const std::string& name, int maxSeen)
    {
        auto it = states.find(name);
        int fromMap = it == states.end() ? 0 : static_cast<int>(it->second.size());
        return std::max(fromMap, maxSeen + 1);
    }
    // Throws std::out_of_range if a value of the instance has no cell in the tables
    // (value >= cardinality), instead of reading out of range (BayesNet#47).
    inline void checkInstanceRange(const std::vector<int>& instance, const std::vector<int>& cardinalities, int nFeatures, const std::vector<std::string>& names, const char* where)
    {
        for (int f = 0; f < nFeatures; ++f) {
            if (instance[f] < 0 || instance[f] >= cardinalities[f]) {
                std::string name = f < static_cast<int>(names.size()) ? names[f] : std::to_string(f);
                throw std::out_of_range(std::string(where) + ": value " + std::to_string(instance[f]) + " of feature '" + name
                    + "' is outside [0, " + std::to_string(cardinalities[f]) + ")");
            }
        }
    }
    std::vector<int> argsort(std::vector<double>& nums);
    std::vector<std::vector<double>> tensorToVectorDouble(torch::Tensor& dtensor);
    torch::Tensor vectorToTensor(std::vector<std::vector<int>>& vector, bool transpose = true);
}
#endif //BAYESNET_UTILS_H