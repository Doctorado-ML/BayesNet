#include "BaseClassifier.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    BaseClassifier::BaseClassifier(Network model) : model(model), m(0), n(0), metrics(Metrics()) {}
    BaseClassifier& BaseClassifier::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {

        dataset = torch::cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        checkFitParameters();
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        train();
        return *this;
    }
    BaseClassifier& BaseClassifier::fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = X;
        this->y = y;
        return build(features, className, states);
    }
    BaseClassifier& BaseClassifier::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, kInt64);
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt64));
        }
        this->y = torch::tensor(y, kInt64);
        return build(features, className, states);
    }
    void BaseClassifier::checkFitParameters()
    {
        auto sizes = X.sizes();
        m = sizes[0];
        n = sizes[1];
        if (m != y.size(0)) {
            throw invalid_argument("X and y must have the same number of samples");
        }
        if (n != features.size()) {
            throw invalid_argument("X and features must have the same number of features");
        }
        if (states.find(className) == states.end()) {
            throw invalid_argument("className not found in states");
        }
        for (auto feature : features) {
            if (states.find(feature) == states.end()) {
                throw invalid_argument("feature [" + feature + "] not found in states");
            }
        }
    }
    vector<int> BaseClassifier::argsort(vector<float>& nums)
    {
        int n = nums.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {return nums[i] > nums[j];});
        return indices;
    }
    vector<vector<int>> tensorToVector(const torch::Tensor& tensor)
    {
        // convert mxn tensor to nxm vector
        vector<vector<int>> result;
        auto tensor_accessor = tensor.accessor<int, 2>();

        // Iterate over columns and rows of the tensor
        for (int j = 0; j < tensor.size(1); ++j) {
            vector<int> column;
            for (int i = 0; i < tensor.size(0); ++i) {
                column.push_back(tensor_accessor[i][j]);
            }
            result.push_back(column);
        }

        return result;
    }
    Tensor BaseClassifier::predict(Tensor& X)
    {
        auto n_models = models.size();
        Tensor y_pred = torch::zeros({ X.size(0), n_models }, torch::kInt64);
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, models[i].predict(X));
        }
        auto y_pred_ = y_pred.accessor<int64_t, 2>();
        vector<int> y_pred_final;
        for (int i = 0; i < y_pred.size(0); ++i) {
            vector<float> votes(states[className].size(), 0);
            for (int j = 0; j < y_pred.size(1); ++j) {
                votes[y_pred_[i][j]] += 1;
            }
            auto indices = argsort(votes);
            y_pred_final.push_back(indices[0]);
        }
        return torch::tensor(y_pred_final, torch::kInt64);
    }
    float BaseClassifier::score(Tensor& X, Tensor& y)
    {
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    vector<string> BaseClassifier::show()
    {
        return model.show();
    }
    void BaseClassifier::addNodes()
    {
        // Add all nodes to the network
        for (auto feature : features) {
            model.addNode(feature, states[feature].size());
        }
        model.addNode(className, states[className].size());
    }
}