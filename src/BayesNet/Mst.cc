#include "Mst.h"
#include <vector>
#include <list>
/*
    Based on the code from https://www.softwaretestinghelp.com/minimum-spanning-tree-tutorial/

*/

namespace bayesnet {
    using namespace std;
    Graph::Graph(int V) : V(V), parent(vector<int>(V))
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
        // sort the edges ordered on decreasing weight
        stable_sort(G.begin(), G.end(), [](const auto& left, const auto& right) {return left.first > right.first;});
        for (int i = 0; i < G.size(); i++) {
            int uSt, vEd;
            uSt = find_set(G[i].second.first);
            vEd = find_set(G[i].second.second);
            if (uSt != vEd) {
                T.push_back(G[i]); // add to mst vector
                union_set(uSt, vEd);
            }
        }
    }
    void Graph::display_mst()
    {
        cout << "Edge :" << " Weight" << endl;
        for (int i = 0; i < T.size(); i++) {
            cout << T[i].second.first << " - " << T[i].second.second << " : "
                << T[i].first;
            cout << endl;
        }
    }

    void insertElement(list<int>& variables, int variable)
    {
        if (find(variables.begin(), variables.end(), variable) == variables.end()) {
            variables.push_front(variable);
        }
    }

    vector<pair<int, int>> reorder(vector<pair<float, pair<int, int>>> T, int root_original)
    {
        // Create the edges of a DAG from the MST
        // replacing unordered_set with list because unordered_set cannot guarantee the order of the elements inserted
        auto result = vector<pair<int, int>>();
        auto visited = vector<int>();
        auto nextVariables = list<int>();
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

    MST::MST(const vector<string>& features, const Tensor& weights, const int root) : features(features), weights(weights), root(root) {}
    vector<pair<int, int>> MST::maximumSpanningTree()
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