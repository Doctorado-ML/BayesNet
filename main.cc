#include <iostream>
#include <string>
#include <torch/torch.h>
#include "ArffFiles.h"
#include "Network.h"
#include "CPPFImdlp.h"


using namespace std;

vector<mdlp::labels_t> discretize(vector<mdlp::samples_t>& X, mdlp::labels_t& y)
{
    vector<mdlp::labels_t>Xd;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        Xd.push_back(fimdlp.transform(X[i]));
    }
    return Xd;
}

int main()
{
    auto handler = ArffFiles();
    handler.load("iris.arff");
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    for (auto feature : handler.getAttributes()) {
        features.push_back(feature.first);
    }
    // Discretize Dataset
    vector<mdlp::labels_t> Xd = discretize(X, y);;
    // Build Network    
    auto network = bayesnet::Network();
    network.fit(Xd, y, features, className);
    cout << "Hello, Bayesian Networks!" << endl;
    cout << "Nodes:" << endl;
    for (auto [name, item] : network.getNodes()) {
        cout << "*" << item->getName() << " -> " << item->getNumStates() << endl;
        cout << "-Parents:" << endl;
        for (auto parent : item->getParents()) {
            cout << " " << parent->getName() << endl;
        }
        cout << "-Children:" << endl;
        for (auto child : item->getChildren()) {
            cout << " " << child->getName() << endl;
        }
    }
    cout << "Root: " << network.getRoot()->getName() << endl;
    network.setRoot(className);
    cout << "Now Root should be class: " << network.getRoot()->getName() << endl;
    cout << "PyTorch version: " << TORCH_VERSION << endl;
    return 0;
}