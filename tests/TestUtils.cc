// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <random>
#include <nlohmann/json.hpp>
#include "TestUtils.h"
#include "bayesnet/config.h"

class Paths {
public:
    static std::string datasets()
    {
        return { data_path.begin(), data_path.end() };
    }
};

class ShuffleArffFiles : public ArffFiles {
public:
    ShuffleArffFiles(int num_samples = 0, bool shuffle = false) : ArffFiles(), num_samples(num_samples), shuffle(shuffle) {}
    void load(const std::string& file_name, bool class_last = true)
    {
        ArffFiles::load(file_name, class_last);
        if (num_samples > 0) {
            if (num_samples > getY().size()) {
                throw std::invalid_argument("num_lines must be less than the number of lines in the file");
            }
            auto indices = std::vector<int>(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            if (shuffle) {
                std::mt19937 g{ 173 };
                std::shuffle(indices.begin(), indices.end(), g);
            }
            auto XX = std::vector<std::vector<float>>(attributes.size(), std::vector<float>(num_samples));
            auto yy = std::vector<int>(num_samples);
            for (int i = 0; i < num_samples; i++) {
                yy[i] = getY()[indices[i]];
                for (int j = 0; j < attributes.size(); j++) {
                    XX[j][i] = X[j][indices[i]];
                }
            }
            X = XX;
            y = yy;
        }
    }
private:
    int num_samples;
    bool shuffle;
};

RawDatasets::RawDatasets(const std::string& file_name, bool discretize_, int num_samples_, bool shuffle_, bool class_last, bool debug)
{
    catalog = loadCatalog();
    num_samples = num_samples_;
    shuffle = shuffle_;
    discretize = discretize_;
    // Xt can be either discretized or not
    // Xv is always discretized
    loadDataset(file_name, class_last);
    auto yresized = torch::transpose(yt.view({ yt.size(0), 1 }), 0, 1);
    dataset = torch::cat({ Xt, yresized }, 0);
    nSamples = dataset.size(1);
    weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
    weightsv = std::vector<double>(nSamples, 1.0 / nSamples);
    classNumStates = states.at(className).size();
    auto fold = folding::StratifiedKFold(5, yt, 271);
    auto [train, test] = fold.getFold(0);
    auto train_t = torch::tensor(train);
    auto test_t = torch::tensor(test);
    // Get train and validation sets
    X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), train_t });
    y_train = dataset.index({ -1, train_t });
    X_test = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), test_t });
    y_test = dataset.index({ -1, test_t });
    if (debug)
        std::cout << to_string();
}

map<std::string, int> RawDatasets::discretizeDataset(std::vector<mdlp::samples_t>& X)
{
    map<std::string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        mdlp::labels_t xd;
        if (is_numeric.at(i)) {
            fimdlp.fit(X[i], yv);
            xd = fimdlp.transform(X[i]);
        } else {
            std::transform(X[i].begin(), X[i].end(), back_inserter(xd), [](const auto& val) {
                return static_cast<int>(val);
                });
        }
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xv.push_back(xd);
    }
    return maxes;
}

map<std::string, std::vector<int>> RawDatasets::loadCatalog()
{
    map<std::string, std::vector<int>> catalogNames;
    ifstream catalog(Paths::datasets() + "all.txt");
    std::vector<int> numericFeaturesIdx;
    if (!catalog.is_open()) {
        throw std::invalid_argument("Unable to open catalog file. [" + Paths::datasets() + +"all.txt" + "]");
    }
    std::string line;
    std::vector<std::string> sorted_lines;
    while (getline(catalog, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        sorted_lines.push_back(line);
    }
    sort(sorted_lines.begin(), sorted_lines.end(), [](const auto& lhs, const auto& rhs) {
        const auto result = mismatch(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(), [](const auto& lhs, const auto& rhs) {return tolower(lhs) == tolower(rhs);});

        return result.second != rhs.cend() && (result.first == lhs.cend() || tolower(*result.first) < tolower(*result.second));
        });

    for (const auto& line : sorted_lines) {
        std::vector<std::string> tokens = split(line, ';');
        std::string name = tokens[0];
        std::string className;
        numericFeaturesIdx.clear();
        int size = tokens.size();
        switch (size) {
            case 1:
                className = "-1";
                numericFeaturesIdx.push_back(-1);
                break;
            case 2:
                className = tokens[1];
                numericFeaturesIdx.push_back(-1);
                break;
            case 3:
                {
                    className = tokens[1];
                    auto numericFeatures = tokens[2];
                    if (numericFeatures == "all") {
                        numericFeaturesIdx.push_back(-1);
                    } else {
                        if (numericFeatures != "none") {
                            auto features = nlohmann::json::parse(numericFeatures);
                            for (auto& f : features) {
                                numericFeaturesIdx.push_back(f);
                            }
                        }
                    }
                }
                break;
            default:
                throw std::invalid_argument("Invalid catalog file format.");

        }
        catalogNames[name] = numericFeaturesIdx;
    }
    catalog.close();
    if (catalogNames.empty()) {
        throw std::invalid_argument("Catalog is empty. Please check the catalog file.");
    }
    return catalogNames;
}

void RawDatasets::loadDataset(const std::string& name, bool class_last)
{
    auto handler = ShuffleArffFiles(num_samples, shuffle);
    handler.load(Paths::datasets() + static_cast<std::string>(name) + ".arff", class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    yv = handler.getY();
    // Get className & Features
    className = handler.getClassName();
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    is_numeric.clear();
    is_numeric.reserve(features.size());
    auto numericFeaturesIdx = catalog.at(name);
    if (numericFeaturesIdx.empty()) {
        // no numeric features
        is_numeric.assign(features.size(), false);
    } else {
        if (numericFeaturesIdx[0] == -1) {
            // all features are numeric
            is_numeric.assign(features.size(), true);
        } else {
            // some features are numeric
            is_numeric.assign(features.size(), false);
            for (const auto& idx : numericFeaturesIdx) {
                if (idx >= 0 && idx < features.size()) {
                    is_numeric[idx] = true;
                }
            }
        }
    }
    // Discretize Dataset
    auto maxValues = discretizeDataset(X);
    maxValues[className] = *max_element(yv.begin(), yv.end()) + 1;
    if (discretize) {
        // discretize the tensor as well
        Xt = torch::zeros({ static_cast<int>(Xv.size()), static_cast<int>(Xv[0].size()) }, torch::kInt32);
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = std::vector<int>(maxValues[features[i]]);
            iota(begin(states.at(features[i])), end(states.at(features[i])), 0);
            Xt.index_put_({ i, "..." }, torch::tensor(Xv[i], torch::kInt32));
        }
        states[className] = std::vector<int>(maxValues[className]);
    } else {
        Xt = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, torch::kFloat32);
        for (int i = 0; i < features.size(); ++i) {
            Xt.index_put_({ i, "..." }, torch::tensor(X[i]));
            if (!is_numeric.at(i)) {
                states[features[i]] = std::vector<int>(maxValues[features[i]]);
                iota(begin(states.at(features[i])), end(states.at(features[i])), 0);
            } else {
                states[features[i]] = std::vector<int>();
            }
        }
        yt = torch::tensor(yv, torch::kInt32);
        int maxy = *max_element(yv.begin(), yv.end()) + 1;
        states[className] = std::vector<int>(maxy);
    }
    iota(begin(states.at(className)), end(states.at(className)), 0);
    yt = torch::tensor(yv, torch::kInt32);

}

