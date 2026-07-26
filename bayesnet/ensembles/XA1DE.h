// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XA1DE_H
#define XA1DE_H
#include "Xaode.hpp"
#include "ExpClf.h"
#include <bayesnet/network/Smoothing.h>

namespace bayesnet {
    class XA1DE : public ExpClf {
    public:
        XA1DE() = default;
        virtual ~XA1DE() override = default;
        std::string getVersion() override { return version; };
    protected:
        void buildModel(const torch::Tensor& weights) override {};
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
    private:
        std::string version = "1.0.0";
    };
}
#endif // XA1DE_H