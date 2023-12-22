#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "Datasets.h"
#include "HyperParameters.h"
#include "GridData.h"
#include "Timer.h"

namespace platform {
    using json = nlohmann::json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string continue_from;
        std::string platform;
        bool quiet;
        bool only; // used with continue_from to only compute that dataset
        bool discretize;
        bool stratified;
        int nested;
        int n_folds;
        json excluded;
        std::vector<int> seeds;
    };
    struct ConfigMPI {
        int rank;
        int n_procs;
        int manager;
    };
    typedef struct {
        uint idx_dataset;
        uint idx_combination;
        double score;
    } Task_Result;
    const TAG_QUERY = 1;
    const TAG_RESULT = 2;
    const TAG_TASK = 3;
    const TAG_END = 4;
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go();
        void go_mpi(struct ConfigMPI& config_mpi);
        ~GridSearch() = default;
        json getResults();
        static inline std::string NO_CONTINUE() { return "NO_CONTINUE"; }
    private:
        void save(json& results);
        json initializeResults();
        vector<std::string> processDatasets(Datasets& datasets);
        pair<double, json> processFileSingle(std::string fileName, Datasets& datasets, std::vector<json>& combinations);
        pair<double, json> processFileNested(std::string fileName, Datasets& datasets, std::vector<json>& combinations);
        struct ConfigGrid config;
        pair<int, int> part_range_mpi(int n_tasks, int nprocs, int rank);
        json build_tasks_mpi();
        void process_task_mpi(struct ConfigMPI& config_mpi, json& task, Datasets& datasets, json& results);
        Timer timer; // used to measure the time of the whole process
    };
} /* namespace platform */
#endif /* GRIDSEARCH_H */