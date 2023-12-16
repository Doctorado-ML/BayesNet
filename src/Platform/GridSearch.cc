#include <iostream>
#include <torch/torch.h>
#include "GridSearch.h"
#include "Models.h"
#include "Paths.h"
#include "Folding.h"
#include "Colors.h"

namespace platform {
    std::string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    std::string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    GridSearch::GridSearch(struct ConfigGrid& config) : config(config)
    {
    }
    json GridSearch::getResults()
    {
        std::ifstream file(Paths::grid_output(config.model));
        if (file.is_open()) {
            return json::parse(file);
        }
        return json();
    }
    vector<std::string> GridSearch::processDatasets(Datasets& datasets)
    {
        // Load datasets
        auto datasets_names = datasets.getNames();
        if (config.continue_from != NO_CONTINUE()) {
            // Continue previous execution:
            if (std::find(datasets_names.begin(), datasets_names.end(), config.continue_from) == datasets_names.end()) {
                throw std::invalid_argument("Dataset " + config.continue_from + " not found");
            }
            // Remove datasets already processed
            vector< string >::iterator it = datasets_names.begin();
            while (it != datasets_names.end()) {
                if (*it != config.continue_from) {
                    it = datasets_names.erase(it);
                } else {
                    if (config.only)
                        ++it;
                    else
                        break;
                }
            }
        }
        // Exclude datasets
        for (const auto& name : config.excluded) {
            auto dataset = name.get<std::string>();
            auto it = std::find(datasets_names.begin(), datasets_names.end(), dataset);
            if (it == datasets_names.end()) {
                throw std::invalid_argument("Dataset " + dataset + " already excluded or doesn't exist!");
            }
            datasets_names.erase(it);
        }
        return datasets_names;
    }
    void showProgressComb(const int num, const int n_folds, const int total, const std::string& color)
    {
        int spaces = int(log(total) / log(10)) + 1;
        int magic = n_folds * 3 + 22 + 2 * spaces;
        std::string prefix = num == 1 ? "" : string(magic, '\b') + string(magic + 1, ' ') + string(magic + 1, '\b');
        std::cout << prefix << color << "(" << setw(spaces) << num << "/" << setw(spaces) << total << ") " << Colors::RESET() << flush;
    }
    void showProgressFold(int fold, const std::string& color, const std::string& phase)
    {
        std::string prefix = phase == "a" ? "" : "\b\b\b\b";
        std::cout << prefix << color << fold << Colors::RESET() << "(" << color << phase << Colors::RESET() << ")" << flush;
    }
    std::string getColor(bayesnet::status_t status)
    {
        switch (status) {
            case bayesnet::NORMAL:
                return Colors::GREEN();
            case bayesnet::WARNING:
                return Colors::YELLOW();
            case bayesnet::ERROR:
                return Colors::RED();
            default:
                return Colors::RESET();
        }
    }
    json GridSearch::build_tasks_mpi()
    {
        auto tasks = json::array();
        auto grid = GridData(Paths::grid_input(config.model));
        auto datasets = Datasets(false, Paths::datasets());
        auto datasets_names = processDatasets(datasets);
        for (const auto& dataset : datasets_names) {
            for (const auto& seed : config.seeds) {
                auto combinations = grid.getGrid(dataset);
                for (int n_fold = 0; n_fold < config.n_folds; n_fold++) {
                    json task = {
                        { "dataset", dataset },
                        { "seed", seed },
                        { "fold", n_fold}
                    };
                    tasks.push_back(task);
                }
            }
        }
        // It's important to shuffle the array so heavy datasets are spread across the Workers
        std::mt19937 g{ 271 }; // Use fixed seed to obtain the same shuffle
        std::shuffle(tasks.begin(), tasks.end(), g);
        std::cout << "Tasks size: " << tasks.size() << std::endl;
        std::cout << "|";
        for (int i = 0; i < tasks.size(); ++i) {
            std::cout << (i + 1) % 10;
        }
        std::cout << "|" << std::endl << "|" << std::flush;
        return tasks;
    }
    std::pair<int, int> GridSearch::part_range_mpi(int n_tasks, int nprocs, int rank)
    {
        int assigned = 0;
        int remainder = n_tasks % nprocs;
        int start = 0;
        if (rank < remainder) {
            assigned = n_tasks / nprocs + 1;
        } else {
            assigned = n_tasks / nprocs;
            start = remainder;
        }
        start += rank * assigned;
        int end = start + assigned;
        if (rank == nprocs - 1) {
            end = n_tasks;
        }
        return { start, end };
    }
    std::string get_color_rank(int rank)
    {
        auto colors = { Colors::RED(), Colors::GREEN(),  Colors::BLUE(), Colors::MAGENTA(), Colors::CYAN() };
        return *(colors.begin() + rank % colors.size());
    }
    void GridSearch::process_task_mpi(struct ConfigMPI& config_mpi, json& task, Datasets& datasets, json& results)
    {
        // Process the task and store the result in the results json
        auto grid = GridData(Paths::grid_input(config.model));
        auto dataset = task["dataset"].get<std::string>();
        auto seed = task["seed"].get<int>();
        auto n_fold = task["fold"].get<int>();
        // Generate the hyperparamters combinations
        auto combinations = grid.getGrid(dataset);
        auto [X, y] = datasets.getTensors(dataset);
        auto states = datasets.getStates(dataset);
        auto features = datasets.getFeatures(dataset);
        auto className = datasets.getClassName(dataset);
        //
        // Start working on task
        //
        Fold* fold;
        if (config.stratified)
            fold = new StratifiedKFold(config.n_folds, y, seed);
        else
            fold = new KFold(config.n_folds, y.size(0), seed);
        auto [train, test] = fold->getFold(n_fold);
        auto train_t = torch::tensor(train);
        auto test_t = torch::tensor(test);
        auto X_train = X.index({ "...", train_t });
        auto y_train = y.index({ train_t });
        auto X_test = X.index({ "...", test_t });
        auto y_test = y.index({ test_t });
        auto num = 0;
        double best_fold_score = 0.0;
        json best_fold_hyper;
        for (const auto& hyperparam_line : combinations) {
            auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
            Fold* nested_fold;
            if (config.stratified)
                nested_fold = new StratifiedKFold(config.nested, y_train, seed);
            else
                nested_fold = new KFold(config.nested, y_train.size(0), seed);
            double score = 0.0;
            for (int n_nested_fold = 0; n_nested_fold < config.nested; n_nested_fold++) {
                // Nested level fold
                auto [train_nested, test_nested] = nested_fold->getFold(n_nested_fold);
                auto train_nested_t = torch::tensor(train_nested);
                auto test_nested_t = torch::tensor(test_nested);
                auto X_nested_train = X_train.index({ "...", train_nested_t });
                auto y_nested_train = y_train.index({ train_nested_t });
                auto X_nested_test = X_train.index({ "...", test_nested_t });
                auto y_nested_test = y_train.index({ test_nested_t });
                // Build Classifier with selected hyperparameters
                auto clf = Models::instance()->create(config.model);
                auto valid = clf->getValidHyperparameters();
                hyperparameters.check(valid, dataset);
                clf->setHyperparameters(hyperparameters.get(dataset));
                // Train model
                clf->fit(X_nested_train, y_nested_train, features, className, states);
                // Test model
                score += clf->score(X_nested_test, y_nested_test);
            }
            delete nested_fold;
            score /= config.nested;
            if (score > best_fold_score) {
                best_fold_score = score;
                best_fold_hyper = hyperparam_line;
            }
        }
        delete fold;
        // Build Classifier with the best hyperparameters to obtain the best score
        auto hyperparameters = platform::HyperParameters(datasets.getNames(), best_fold_hyper);
        auto clf = Models::instance()->create(config.model);
        auto valid = clf->getValidHyperparameters();
        hyperparameters.check(valid, dataset);
        clf->setHyperparameters(best_fold_hyper);
        clf->fit(X_train, y_train, features, className, states);
        best_fold_score = clf->score(X_test, y_test);
        // Save results
        results[dataset][std::to_string(n_fold)]["score"] = best_fold_score;
        results[dataset][std::to_string(n_fold)]["hyperparameters"] = best_fold_hyper;
        results[dataset][std::to_string(n_fold)]["seed"] = seed;
        std::cout << get_color_rank(config_mpi.rank) << "*" << std::flush;
    }
    void GridSearch::go_mpi(struct ConfigMPI& config_mpi)
    {
        /*
         * Each task is a json object with the following structure:
         * {
         *   "dataset": "dataset_name",
         *   "seed": # of seed to use,
         *   "model": "model_name",
         *   "Fold": # of fold to process
         * }
         *
         * The overall process consists in these steps:
            * 1. Manager will broadcast the tasks to all the processes
            * 1.1 Broadcast the number of tasks
            * 1.2 Broadcast the length of the following string
            * 1.2 Broadcast the tasks as a char* string
            * 2. Workers will receive the tasks and start the process
            * 2.1 A method will tell each worker the range of tasks to process
            * 2.2 Each worker will process the tasks and generate the best score for each task
            * 3. Manager gather the scores from all the workers and find out the best hyperparameters for each dataset
            * 3.1 Obtain the maximum size of the results message of all the workers
            * 3.2 Gather all the results from the workers into the manager
            * 3.3 Compile the results from all the workers
            * 3.4 Filter the best hyperparameters for each dataset
         */
        char* msg;
        int tasks_size;
        if (config_mpi.rank == config_mpi.manager) {
            timer.start();
            auto tasks = build_tasks_mpi();
            auto tasks_str = tasks.dump();
            tasks_size = tasks_str.size();
            msg = new char[tasks_size + 1];
            strcpy(msg, tasks_str.c_str());
        }
        //
        // 1. Manager will broadcast the tasks to all the processes
        //
        MPI_Bcast(&tasks_size, 1, MPI_INT, config_mpi.manager, MPI_COMM_WORLD);
        if (config_mpi.rank != config_mpi.manager) {
            msg = new char[tasks_size + 1];
        }
        MPI_Bcast(msg, tasks_size + 1, MPI_CHAR, config_mpi.manager, MPI_COMM_WORLD);
        json tasks = json::parse(msg);
        delete[] msg;
        //
        // 2. All Workers will receive the tasks and start the process
        //
        int num_tasks = tasks.size();
        // 2.1 A method will tell each worker the range of tasks to process
        auto [start, end] = part_range_mpi(num_tasks, config_mpi.n_procs, config_mpi.rank);
        // 2.2 Each worker will process the tasks and return the best scores obtained
        auto datasets = Datasets(config.discretize, Paths::datasets());
        json results;
        for (int i = start; i < end; ++i) {
            // Process task
            process_task_mpi(config_mpi, tasks[i], datasets, results);
        }
        int size = results.dump().size() + 1;
        int max_size = 0;
        //
        // 3. Manager gather the scores from all the workers and find out the best hyperparameters for each dataset
        //
        //3.1 Obtain the maximum size of the results message of all the workers
        MPI_Allreduce(&size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // Assign the memory to the message and initialize it to 0s
        char* total = NULL;
        msg = new char[max_size];
        strncpy(msg, results.dump().c_str(), size);
        if (config_mpi.rank == config_mpi.manager) {
            total = new char[max_size * config_mpi.n_procs];
        }
        // 3.2 Gather all the results from the workers into the manager
        MPI_Gather(msg, max_size, MPI_CHAR, total, max_size, MPI_CHAR, config_mpi.manager, MPI_COMM_WORLD);
        delete[] msg;
        if (config_mpi.rank == config_mpi.manager) {
            std::cout << Colors::RESET() << "|" << std::endl;
            json total_results;
            json best_results;
            // 3.3 Compile the results from all the workers
            for (int i = 0; i < config_mpi.n_procs; ++i) {
                json partial_results = json::parse(total + i * max_size);
                for (auto& [dataset, folds] : partial_results.items()) {
                    for (auto& [fold, result] : folds.items()) {
                        total_results[dataset][fold] = result;
                    }
                }
            }
            delete[] total;
            // 3.4 Filter the best hyperparameters for each dataset
            auto grid = GridData(Paths::grid_input(config.model));
            for (auto& [dataset, folds] : total_results.items()) {
                double best_score = 0.0;
                json best_hyper;
                for (auto& [fold, result] : folds.items()) {
                    if (result["score"] > best_score) {
                        best_score = result["score"];
                        best_hyper = result["hyperparameters"];
                    }
                }
                json result = {
                    { "score", best_score },
                    { "hyperparameters", best_hyper },
                    { "date", get_date() + " " + get_time() },
                    { "grid", grid.getInputGrid(dataset) },
                    { "duration", 0 }
                };
                best_results[dataset] = result;
            }
            save(best_results);
        }
    }
    void GridSearch::go()
    {
        timer.start();
        auto grid_type = config.nested == 0 ? "Single" : "Nested";
        auto datasets = Datasets(config.discretize, Paths::datasets());
        auto datasets_names = processDatasets(datasets);
        json results = initializeResults();
        std::cout << "***************** Starting " << grid_type << " Gridsearch *****************" << std::endl;
        std::cout << "input file=" << Paths::grid_input(config.model) << std::endl;
        auto grid = GridData(Paths::grid_input(config.model));
        Timer timer_dataset;
        double bestScore = 0;
        json bestHyperparameters;
        for (const auto& dataset : datasets_names) {
            if (!config.quiet)
                std::cout << "- " << setw(20) << left << dataset << " " << right << flush;
            auto combinations = grid.getGrid(dataset);
            timer_dataset.start();
            if (config.nested == 0)
                // for dataset // for hyperparameters // for seed // for fold
                tie(bestScore, bestHyperparameters) = processFileSingle(dataset, datasets, combinations);
            else
                // for dataset // for seed // for fold // for hyperparameters // for nested fold
                tie(bestScore, bestHyperparameters) = processFileNested(dataset, datasets, combinations);
            if (!config.quiet) {
                std::cout << "end." << " Score: " << Colors::IBLUE() << setw(9) << setprecision(7) << fixed
                    << bestScore << Colors::BLUE() << " [" << bestHyperparameters.dump() << "]"
                    << Colors::RESET() << ::endl;
            }
            json result = {
                { "score", bestScore },
                { "hyperparameters", bestHyperparameters },
                { "date", get_date() + " " + get_time() },
                { "grid", grid.getInputGrid(dataset) },
                { "duration", timer_dataset.getDurationString() }
            };
            results[dataset] = result;
            // Save partial results
            save(results);
        }
        // Save final results
        save(results);
        std::cout << "***************** Ending " << grid_type << " Gridsearch *******************" << std::endl;
    }
    pair<double, json> GridSearch::processFileSingle(std::string fileName, Datasets& datasets, vector<json>& combinations)
    {
        int num = 0;
        double bestScore = 0.0;
        json bestHyperparameters;
        auto totalComb = combinations.size();
        for (const auto& hyperparam_line : combinations) {
            if (!config.quiet)
                showProgressComb(++num, config.n_folds, totalComb, Colors::CYAN());
            auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
            // Get dataset
            auto [X, y] = datasets.getTensors(fileName);
            auto states = datasets.getStates(fileName);
            auto features = datasets.getFeatures(fileName);
            auto className = datasets.getClassName(fileName);
            double totalScore = 0.0;
            int numItems = 0;
            for (const auto& seed : config.seeds) {
                if (!config.quiet)
                    std::cout << "(" << seed << ") doing Fold: " << flush;
                Fold* fold;
                if (config.stratified)
                    fold = new StratifiedKFold(config.n_folds, y, seed);
                else
                    fold = new KFold(config.n_folds, y.size(0), seed);
                for (int nfold = 0; nfold < config.n_folds; nfold++) {
                    auto clf = Models::instance()->create(config.model);
                    auto valid = clf->getValidHyperparameters();
                    hyperparameters.check(valid, fileName);
                    clf->setHyperparameters(hyperparameters.get(fileName));
                    auto [train, test] = fold->getFold(nfold);
                    auto train_t = torch::tensor(train);
                    auto test_t = torch::tensor(test);
                    auto X_train = X.index({ "...", train_t });
                    auto y_train = y.index({ train_t });
                    auto X_test = X.index({ "...", test_t });
                    auto y_test = y.index({ test_t });
                    // Train model
                    if (!config.quiet)
                        showProgressFold(nfold + 1, getColor(clf->getStatus()), "a");
                    clf->fit(X_train, y_train, features, className, states);
                    // Test model
                    if (!config.quiet)
                        showProgressFold(nfold + 1, getColor(clf->getStatus()), "b");
                    totalScore += clf->score(X_test, y_test);
                    numItems++;
                    if (!config.quiet)
                        std::cout << "\b\b\b,  \b" << flush;
                }
                delete fold;
            }
            double score = numItems == 0 ? 0.0 : totalScore / numItems;
            if (score > bestScore) {
                bestScore = score;
                bestHyperparameters = hyperparam_line;
            }
        }
        return { bestScore, bestHyperparameters };
    }
    pair<double, json> GridSearch::processFileNested(std::string fileName, Datasets& datasets, vector<json>& combinations)
    {
        // Get dataset
        auto [X, y] = datasets.getTensors(fileName);
        auto states = datasets.getStates(fileName);
        auto features = datasets.getFeatures(fileName);
        auto className = datasets.getClassName(fileName);
        int spcs_combinations = int(log(combinations.size()) / log(10)) + 1;
        double goatScore = 0.0;
        json goatHyperparameters;
        // for dataset // for seed // for fold // for hyperparameters // for nested fold
        for (const auto& seed : config.seeds) {
            Fold* fold;
            if (config.stratified)
                fold = new StratifiedKFold(config.n_folds, y, seed);
            else
                fold = new KFold(config.n_folds, y.size(0), seed);
            double bestScore = 0.0;
            json bestHyperparameters;
            std::cout << "(" << seed << ") doing Fold: " << flush;
            for (int nfold = 0; nfold < config.n_folds; nfold++) {
                if (!config.quiet)
                    std::cout << Colors::GREEN() << nfold + 1 << " " << flush;
                // First level fold
                auto [train, test] = fold->getFold(nfold);
                auto train_t = torch::tensor(train);
                auto test_t = torch::tensor(test);
                auto X_train = X.index({ "...", train_t });
                auto y_train = y.index({ train_t });
                auto X_test = X.index({ "...", test_t });
                auto y_test = y.index({ test_t });
                auto num = 0;
                json result_fold;
                double hypScore = 0.0;
                double bestHypScore = 0.0;
                json bestHypHyperparameters;
                for (const auto& hyperparam_line : combinations) {
                    std::cout << "[" << setw(spcs_combinations) << ++num << "/" << setw(spcs_combinations)
                        << combinations.size() << "] " << std::flush;
                    Fold* nested_fold;
                    if (config.stratified)
                        nested_fold = new StratifiedKFold(config.nested, y_train, seed);
                    else
                        nested_fold = new KFold(config.nested, y_train.size(0), seed);
                    for (int n_nested_fold = 0; n_nested_fold < config.nested; n_nested_fold++) {
                        // Nested level fold
                        auto [train_nested, test_nested] = nested_fold->getFold(n_nested_fold);
                        auto train_nested_t = torch::tensor(train_nested);
                        auto test_nested_t = torch::tensor(test_nested);
                        auto X_nexted_train = X_train.index({ "...", train_nested_t });
                        auto y_nested_train = y_train.index({ train_nested_t });
                        auto X_nested_test = X_train.index({ "...", test_nested_t });
                        auto y_nested_test = y_train.index({ test_nested_t });
                        // Build Classifier with selected hyperparameters
                        auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
                        auto clf = Models::instance()->create(config.model);
                        auto valid = clf->getValidHyperparameters();
                        hyperparameters.check(valid, fileName);
                        clf->setHyperparameters(hyperparameters.get(fileName));
                        // Train model
                        if (!config.quiet)
                            showProgressFold(n_nested_fold + 1, getColor(clf->getStatus()), "a");
                        clf->fit(X_nexted_train, y_nested_train, features, className, states);
                        // Test model
                        if (!config.quiet)
                            showProgressFold(n_nested_fold + 1, getColor(clf->getStatus()), "b");
                        hypScore += clf->score(X_nested_test, y_nested_test);
                        if (!config.quiet)
                            std::cout << "\b\b\b,  \b" << flush;
                    }
                    int magic = 3 * config.nested + 2 * spcs_combinations + 4;
                    std::cout << string(magic, '\b') << string(magic, ' ') << string(magic, '\b') << flush;
                    delete nested_fold;
                    hypScore /= config.nested;
                    if (hypScore > bestHypScore) {
                        bestHypScore = hypScore;
                        bestHypHyperparameters = hyperparam_line;
                    }
                }
                // Build Classifier with selected hyperparameters
                auto clf = Models::instance()->create(config.model);
                clf->setHyperparameters(bestHypHyperparameters);
                // Train model
                if (!config.quiet)
                    showProgressFold(nfold + 1, getColor(clf->getStatus()), "a");
                clf->fit(X_train, y_train, features, className, states);
                // Test model
                if (!config.quiet)
                    showProgressFold(nfold + 1, getColor(clf->getStatus()), "b");
                double score = clf->score(X_test, y_test);
                if (!config.quiet)
                    std::cout << string(2 * config.nested - 1, '\b') << "," << string(2 * config.nested, ' ') << string(2 * config.nested - 1, '\b') << flush;
                if (score > bestScore) {
                    bestScore = score;
                    bestHyperparameters = bestHypHyperparameters;
                }
            }
            if (bestScore > goatScore) {
                goatScore = bestScore;
                goatHyperparameters = bestHyperparameters;
            }
            delete fold;
        }
        return { goatScore, goatHyperparameters };
    }
    json GridSearch::initializeResults()
    {
        // Load previous results
        json results;
        if (config.continue_from != NO_CONTINUE()) {
            if (!config.quiet)
                std::cout << "* Loading previous results" << std::endl;
            try {
                std::ifstream file(Paths::grid_output(config.model));
                if (file.is_open()) {
                    results = json::parse(file);
                    results = results["results"];
                }
            }
            catch (const std::exception& e) {
                std::cerr << "* There were no previous results" << std::endl;
                std::cerr << "* Initizalizing new results" << std::endl;
                results = json();
            }
        }
        return results;
    }
    void GridSearch::save(json& results)
    {
        std::ofstream file(Paths::grid_output(config.model));
        json output = {
            { "model", config.model },
            { "score", config.score },
            { "discretize", config.discretize },
            { "stratified", config.stratified },
            { "n_folds", config.n_folds },
            { "seeds", config.seeds },
            { "date", get_date() + " " + get_time()},
            { "nested", config.nested},
            { "platform", config.platform },
            { "duration", timer.getDurationString(true)},
            { "results", results }

        };
        file << output.dump(4);
    }
} /* namespace platform */