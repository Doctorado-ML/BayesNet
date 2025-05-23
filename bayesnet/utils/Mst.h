// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef MST_H
#define MST_H
#include <vector>
#include <string>
#include <torch/torch.h>
namespace bayesnet {
    class MST {
    public:
        MST() = default;
        MST(const std::vector<std::string>& features, const torch::Tensor& weights, const int root);
        void insertElement(std::list<int>& variables, int variable);
        std::vector<std::pair<int, int>> reorder(std::vector<std::pair<float, std::pair<int, int>>> T, int root_original);
        std::vector<std::pair<int, int>> maximumSpanningTree();
    private:
        torch::Tensor weights;
        std::vector<std::string> features;
        int root = 0;
    };
    class Graph {
    public:
        explicit Graph(int V);
        void addEdge(int u, int v, float wt);
        int find_set(int i);
        void union_set(int u, int v);
        void kruskal_algorithm();
        std::vector <std::pair<float, std::pair<int, int>>> get_mst() { return T; }
    private:
        int V;      // number of nodes in graph
        std::vector <std::pair<float, std::pair<int, int>>> G; // std::vector for graph
        std::vector <std::pair<float, std::pair<int, int>>> T; // std::vector for mst
        std::vector<int> parent;
    };
}
#endif