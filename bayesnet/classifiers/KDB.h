#ifndef KDB_H
#define KDB_H
#include <torch/torch.h>
#include "bayesnet/utils/bayesnetUtils.h"
#include "Classifier.h"
namespace bayesnet {
    class KDB : public Classifier {
    private:
        int k;
        float theta;
        void add_m_edges(int idx, std::vector<int>& S, torch::Tensor& weights);
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        explicit KDB(int k, float theta = 0.03);
        virtual ~KDB() = default;
        void setHyperparameters(const nlohmann::json& hyperparameters_) override;
        std::vector<std::string> graph(const std::string& name = "KDB") const override;
    };
}
#endif