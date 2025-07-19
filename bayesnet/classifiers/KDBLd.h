// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef KDBLD_H
#define KDBLD_H
#include "Proposal.h"
#include "KDB.h"

namespace bayesnet {
    class KDBLd : public KDB, public Proposal {
    public:
        explicit KDBLd(int k);
        virtual ~KDBLd() = default;
        KDBLd& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        KDBLd& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing) override;
        KDBLd& commonFit(const std::vector<std::string>& features, const std::string& className, map<std::string, std::vector<int>>& states, const Smoothing_t smoothing);
        std::vector<std::string> graph(const std::string& name = "KDB") const override;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override
        {
            auto hyperparameters = hyperparameters_;
            Proposal::setHyperparameters(hyperparameters);
            KDB::setHyperparameters(hyperparameters);
        }
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        static inline std::string version() { return "0.0.1"; };
    };
}
#endif // !KDBLD_H