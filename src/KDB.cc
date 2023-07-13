#include "KDB.h"
#include "Metrics.hpp"

namespace bayesnet {
    using namespace std;
    using namespace torch;
    vector<int> argsort(vector<float>& nums)
    {
        int n = nums.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {return nums[i] > nums[j];});
        return indices;
    }
    KDB::KDB(int k, float theta = 0.03) : BaseClassifier(Network()), k(k), theta(theta) {}
    void KDB::train()
    {
        /*
        1. For each feature Xi, compute mutual information, I(X;C),
        where C is the class.
        2. Compute class conditional mutual information I(Xi;XjIC), f or each
        pair of features Xi and Xj, where i#j.
        3. Let the used variable list, S, be empty.
        4. Let the DAG network being constructed, BN, begin with a single
        class node, C.
        5. Repeat until S includes all domain features
        5.1. Select feature Xmax which is not in S and has the largest value
        I(Xmax;C).
        5.2. Add a node to BN representing Xmax.
        5.3. Add an arc from C to Xmax in BN.
        5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
        the highest value for I(Xmax;X,jC).
        5.5. Add Xmax to S.
        Compute the conditional probabilility infered by the structure of BN by
        using counts from DB, and output BN.
        */
        // 1. For each feature Xi, compute mutual information, I(X;C),
        // where C is the class.
        cout << "Computing mutual information between features and class" << endl;
        auto n_classes = states[className].size();
        auto metrics = Metrics(dataset, features, className, n_classes);
        vector <float> mi;
        for (auto i = 0; i < features.size(); i++) {
            Tensor firstFeature = X.index({ "...", i });
            mi.push_back(metrics.mutualInformation(firstFeature, y));
            cout << "Mutual information between " << features[i] << " and " << className << " is " << mi[i] << endl;
        }
        // 2. Compute class conditional mutual information I(Xi;XjIC), f or each
        auto conditionalEdgeWeights = metrics.conditionalEdge();
        cout << "Conditional edge weights" << endl;
        cout << conditionalEdgeWeights << endl;
        // 3. Let the used variable list, S, be empty.
        vector<int> S;
        // 4. Let the DAG network being constructed, BN, begin with a single
        // class node, C.
        model.addNode(className, states[className].size());
        cout << "Adding node " << className << " to the network" << endl;
        // 5. Repeat until S includes all domain features
        // 5.1. Select feature Xmax which is not in S and has the largest value
        // I(Xmax;C).
        auto order = argsort(mi);
        for (auto idx : order) {
            cout << idx << " " << mi[idx] << endl;
            // 5.2. Add a node to BN representing Xmax.
            model.addNode(features[idx], states[features[idx]].size());
            // 5.3. Add an arc from C to Xmax in BN.
            model.addEdge(className, features[idx]);
            // 5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
            // the highest value for I(Xmax;X,jC).
            add_m_edges(idx, S, conditionalEdgeWeights);
            // 5.5. Add Xmax to S.
            S.push_back(idx);
        }
    }
    void KDB::add_m_edges(int idx, vector<int>& S, Tensor& weights)
    {
        auto n_edges = min(k, static_cast<int>(S.size()));
        auto cond_w = clone(weights);
        bool exit_cond = k == 0;
        int num = 0;
        while (!exit_cond) {
            auto max_minfo = argmax(cond_w.index({ "...", idx })).item<int>();
            auto belongs = find(S.begin(), S.end(), max_minfo) != S.end();
            if (belongs && cond_w.index({ idx, max_minfo }).item<float>() > theta) {
                try {
                    model.addEdge(features[idx], features[max_minfo]);
                    num++;
                }
                catch (const invalid_argument& e) {
                    // Loops are not allowed
                }
            }
            cond_w.index_put_({ "...", max_minfo }, -1);
            auto candidates = cond_w.gt(theta);
            exit_cond = num == n_edges || candidates.size(0) == 0;
        }
    }
}