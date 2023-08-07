#ifndef NODE_H
#define NODE_H
#include <torch/torch.h>
#include <unordered_set>
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    class Node {
    private:
        string name;
        vector<Node*> parents;
        vector<Node*> children;
        int numStates; // number of states of the variable
        torch::Tensor cpTable; // Order of indices is 0-> node variable, 1-> 1st parent, 2-> 2nd parent, ...
        vector<int64_t> dimensions; // dimensions of the cpTable
    public:
        vector<pair<string, string>> combinations(const vector<string>&);
        explicit Node(const string&);
        void clear();
        void addParent(Node*);
        void addChild(Node*);
        void removeParent(Node*);
        void removeChild(Node*);
        string getName() const;
        vector<Node*>& getParents();
        vector<Node*>& getChildren();
        torch::Tensor& getCPT();
        void computeCPT(const torch::Tensor&, const vector<string>&, const int);
        int getNumStates() const;
        void setNumStates(int);
        unsigned minFill();
        vector<string> graph(const string& clasName); // Returns a vector of strings representing the graph in graphviz format
        float getFactorValue(map<string, int>&);
    };
}
#endif