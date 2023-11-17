#ifndef SVC_H
#define SVC_H
#include "PyClassifier.h"

namespace pywrap {
    class SVC : public PyClassifier {
    public:
        SVC() : PyClassifier("sklearn.svm", "SVC", true) {};
        ~SVC() = default;
        void setHyperparameters(nlohmann::json& hyperparameters) override;
    };

} /* namespace pywrap */
#endif /* STREE_H */