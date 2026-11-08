// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XBAODE_H
#define XBAODE_H
#include <vector>
#include <cmath>
#include "Boost.h"

namespace bayesnet {
    class XBAODE : public Boost {
    public:
        XBAODE();
        std::string getVersion() override { return version; };
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
    private:
        std::vector<int> initializeModels(const Smoothing_t smoothing);
        std::vector<std::vector<int>> X_train_, X_test_;
        std::vector<int> y_train_, y_test_;
        std::string version = "0.9.9";
    };
}
#endif // XBAODE_H
