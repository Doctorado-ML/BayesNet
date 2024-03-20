#include "bayesnet/utils/bayesnetUtils.h"
#include "Classifier.h"

namespace bayesnet {
    Classifier::Classifier(Network model) : model(model), m(0), n(0), metrics(Metrics()), fitted(false) {}
    const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";
    Classifier& Classifier::build(const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights)
    {
        this->features = features;
        this->className = className;
        this->states = states;
        m = dataset.size(1);
        n = features.size();
        checkFitParameters();
        auto n_classes = states.at(className).size();
        metrics = Metrics(dataset, features, className, n_classes);
        model.initialize();
        buildModel(weights);
        trainModel(weights);
        fitted = true;
        return *this;
    }
    void Classifier::buildDataset(torch::Tensor& ytmp)
    {
        try {
            auto yresized = torch::transpose(ytmp.view({ ytmp.size(0), 1 }), 0, 1);
            dataset = torch::cat({ dataset, yresized }, 0);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            std::cout << "X dimensions: " << dataset.sizes() << "\n";
            std::cout << "y dimensions: " << ytmp.sizes() << "\n";
            exit(1);
        }
    }
    void Classifier::trainModel(const torch::Tensor& weights)
    {
        model.fit(dataset, weights, features, className, states);
    }
    // X is nxm where n is the number of features and m the number of samples
    Classifier& Classifier::fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states)
    {
        dataset = X;
        buildDataset(y);
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    // X is nxm where n is the number of features and m the number of samples
    Classifier& Classifier::fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states)
    {
        dataset = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, torch::kInt32);
        for (int i = 0; i < X.size(); ++i) {
            dataset.index_put_({ i, "..." }, torch::tensor(X[i], torch::kInt32));
        }
        auto ytmp = torch::tensor(y, torch::kInt32);
        buildDataset(ytmp);
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    Classifier& Classifier::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states)
    {
        this->dataset = dataset;
        const torch::Tensor weights = torch::full({ dataset.size(1) }, 1.0 / dataset.size(1), torch::kDouble);
        return build(features, className, states, weights);
    }
    Classifier& Classifier::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights)
    {
        this->dataset = dataset;
        return build(features, className, states, weights);
    }
    void Classifier::checkFitParameters()
    {
        if (torch::is_floating_point(dataset)) {
            throw std::invalid_argument("dataset (X, y) must be of type Integer");
        }
        if (n != features.size()) {
            throw std::invalid_argument("Classifier: X " + std::to_string(n) + " and features " + std::to_string(features.size()) + " must have the same number of features");
        }
        if (states.find(className) == states.end()) {
            throw std::invalid_argument("className not found in states");
        }
        for (auto feature : features) {
            if (states.find(feature) == states.end()) {
                throw std::invalid_argument("feature [" + feature + "] not found in states");
            }
        }
    }
    torch::Tensor Classifier::predict(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        return model.predict(X);
    }
    std::vector<int> Classifier::predict(std::vector<std::vector<int>>& X)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        std::vector<std::vector<int>> Xd(n_, std::vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = std::vector<int>(X[i].begin(), X[i].end());
        }
        auto yp = model.predict(Xd);
        return yp;
    }
    torch::Tensor Classifier::predict_proba(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        return model.predict_proba(X);
    }
    std::vector<std::vector<double>> Classifier::predict_proba(std::vector<std::vector<int>>& X)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        std::vector<std::vector<int>> Xd(n_, std::vector<int>(m_, 0));
        // Convert to nxm vector
        for (auto i = 0; i < n_; i++) {
            Xd[i] = std::vector<int>(X[i].begin(), X[i].end());
        }
        auto yp = model.predict_proba(Xd);
        return yp;
    }
    float Classifier::score(torch::Tensor& X, torch::Tensor& y)
    {
        torch::Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    float Classifier::score(std::vector<std::vector<int>>& X, std::vector<int>& y)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        return model.score(X, y);
    }
    std::vector<std::string> Classifier::show() const
    {
        return model.show();
    }
    void Classifier::addNodes()
    {
        // Add all nodes to the network
        for (const auto& feature : features) {
            model.addNode(feature);
        }
        model.addNode(className);
    }
    int Classifier::getNumberOfNodes() const
    {
        // Features does not include class
        return fitted ? model.getFeatures().size() : 0;
    }
    int Classifier::getNumberOfEdges() const
    {
        return fitted ? model.getNumEdges() : 0;
    }
    int Classifier::getNumberOfStates() const
    {
        return fitted ? model.getStates() : 0;
    }
    int Classifier::getClassNumStates() const
    {
        return fitted ? model.getClassNumStates() : 0;
    }
    std::vector<std::string> Classifier::topological_order()
    {
        return model.topological_sort();
    }
    void Classifier::dump_cpt() const
    {
        model.dump_cpt();
    }
    void Classifier::setHyperparameters(const nlohmann::json& hyperparameters)
    {
        //For classifiers that don't have hyperparameters
    }
}