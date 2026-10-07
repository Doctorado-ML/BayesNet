// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <sstream>
#include <vector>
#include <list>
#include "Mst.h"
/*
    Based on the code from https://www.softwaretestinghelp.com/minimum-spanning-tree-tutorial/

*/

namespace bayesnet {
    Graph::Graph(int V) : V(V), parent(std::vector<int>(V))
    {
        for (int i = 0; i < V; i++)
            parent[i] = i;
        G.clear();
        T.clear();
    }
    void Graph::addEdge(int u, int v, float wt)
    {
        G.push_back({ wt, { u, v } });
    }
    int Graph::find_set(int i)
    {
        // If i is the parent of itself
        if (i == parent[i])
            return i;
        else
            //else recursively find the parent of i
            return find_set(parent[i]);
    }
    void Graph::union_set(int u, int v)
    {
        parent[u] = parent[v];
    }
    void Graph::kruskal_algorithm()
    {
        // Sort the edges on decreasing weight. Equal weights are broken by the
        // (u, v) endpoints so that the order is a total one: whole blocks of edges
        // tie here (every edge of a feature left with a single state after
        // discretization scores exactly 0), and which of them enters the tree
        // decided the result. Leaving that to stable_sort made it depend on the
        // order addEdge happened to be called in.
        std::stable_sort(G.begin(), G.end(), [](const auto& left, const auto& right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second < right.second;
            });
        for (int i = 0; i < G.size(); i++) {
            int uSt, vEd;
            uSt = find_set(G[i].second.first);
            vEd = find_set(G[i].second.second);
            if (uSt != vEd) {
                T.push_back(G[i]); // add to mst std::vector
                union_set(uSt, vEd);
            }
        }
    }

    void MST::insertElement(std::list<int>& variables, int variable)
    {
        if (std::find(variables.begin(), variables.end(), variable) == variables.end()) {
            variables.push_front(variable);
        }
    }

    std::vector<std::pair<int, int>> MST::reorder(std::vector<std::pair<float, std::pair<int, int>>> T, int root_original)
    {
        // Create the edges of a DAG from the MST
        // replacing unordered_set with list because unordered_set cannot guarantee the order of the elements inserted
        auto result = std::vector<std::pair<int, int>>();
        auto visited = std::vector<int>();
        auto nextVariables = std::list<int>();
        nextVariables.push_front(root_original);
        while (nextVariables.size() > 0) {
            int root = nextVariables.front();
            nextVariables.pop_front();
            for (int i = 0; i < T.size(); ++i) {
                auto [weight, edge] = T[i];
                auto [from, to] = edge;
                if (from == root || to == root) {
                    visited.insert(visited.begin(), i);
                    if (from == root) {
                        result.push_back({ from, to });
                        insertElement(nextVariables, to);
                    } else {
                        result.push_back({ to, from });
                        insertElement(nextVariables, from);
                    }
                }
            }
            // Remove visited
            for (int i = 0; i < visited.size(); ++i) {
                T.erase(T.begin() + visited[i]);
            }
            visited.clear();
        }
        if (T.size() > 0) {
            for (int i = 0; i < T.size(); ++i) {
                auto [weight, edge] = T[i];
                auto [from, to] = edge;
                result.push_back({ from, to });
            }
        }
        return result;
    }

    MST::MST(const std::vector<std::string>& features, const torch::Tensor& weights, const int root) : features(features), weights(weights), root(root) {}
    std::vector<std::pair<int, int>> MST::maximumSpanningTree()
    {
        auto num_features = features.size();
        Graph g(num_features);
        // Make a complete graph
        for (int i = 0; i < num_features - 1; ++i) {
            for (int j = i + 1; j < num_features; ++j) {
                g.addEdge(i, j, weights[i][j].item<float>());
            }
        }
        g.kruskal_algorithm();
        auto mst = g.get_mst();
        return reorder(mst, root);
    }

}