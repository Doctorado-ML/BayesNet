// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef PROPOSAL_H
#define PROPOSAL_H
#include <string>
#include <map>
#include <torch/torch.h>
#include <fimdlp/CPPFImdlp.h>
#include "bayesnet/network/Network.h"
#include "Classifier.h"

namespace bayesnet {
    class Proposal {
    public:
        Proposal(torch::Tensor& pDataset, std::vector<std::string>& features_, std::string& className_);
        virtual ~Proposal();
    protected:
        void checkInput(const torch::Tensor& X, const torch::Tensor& y);
        torch::Tensor prepareX(torch::Tensor& X);
        map<std::string, std::vector<int>> localDiscretizationProposal(const map<std::string, std::vector<int>>& states, Network& model);
        map<std::string, std::vector<int>> fit_local_discretization(const torch::Tensor& y);
        torch::Tensor Xf; // X continuous nxm tensor
        torch::Tensor y; // y discrete nx1 tensor
        map<std::string, mdlp::CPPFImdlp*> discretizers;
    private:
        std::vector<int> factorize(const std::vector<std::string>& labels_t);
        torch::Tensor& pDataset; // (n+1)xm tensor
        std::vector<std::string>& pFeatures;
        std::string& pClassName;
    };
}

#endif  