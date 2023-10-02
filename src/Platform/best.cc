#include <iostream>
#include <argparse/argparse.hpp>
#include "Paths.h"
#include "BestResults.h"
#include "Colors.h"

using namespace std;

argparse::ArgumentParser manageArguments(int argc, char** argv)
{
    argparse::ArgumentParser program("best");
    program.add_argument("-m", "--model").default_value("").help("Filter results of the selected model) (any for all models)");
    program.add_argument("-s", "--score").default_value("").help("Filter results of the score name supplied");
    program.add_argument("--build").help("build best score results file").default_value(false).implicit_value(true);
    program.add_argument("--report").help("report of best score results file").default_value(false).implicit_value(true);
    program.add_argument("--friedman").help("Friedman test").default_value(false).implicit_value(true);
    program.add_argument("--excel").help("Output to excel").default_value(false).implicit_value(true);
    program.add_argument("--level").help("significance level").default_value(0.05).scan<'g', double>().action([](const string& value) {
        try {
            auto k = stod(value);
            if (k < 0.01 || k > 0.15) {
                throw runtime_error("Significance level hast to be a number in [0.01, 0.15]");
            }
            return k;
        }
        catch (const runtime_error& err) {
            throw runtime_error(err.what());
        }
        catch (...) {
            throw runtime_error("Number of folds must be an decimal number");
        }});
    try {
        program.parse_args(argc, argv);
        auto model = program.get<string>("model");
        auto score = program.get<string>("score");
        auto build = program.get<bool>("build");
        auto report = program.get<bool>("report");
        auto friedman = program.get<bool>("friedman");
        auto excel = program.get<bool>("excel");
        auto level = program.get<double>("level");
        if (model == "" || score == "") {
            throw runtime_error("Model and score name must be supplied");
        }
        if (friedman && model != "any") {
            cerr << "Friedman test can only be used with all models" << endl;
            cerr << program;
            exit(1);
        }
        if (excel && model != "any") {
            cerr << "Excel ourput can only be used with all models" << endl;
            cerr << program;
            exit(1);
        }
        if (!report && !build) {
            cerr << "Either build, report or both, have to be selected to do anything!" << endl;
            cerr << program;
            exit(1);
        }
    }
    catch (const exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        exit(1);
    }
    return program;
}

int main(int argc, char** argv)
{
    auto program = manageArguments(argc, argv);
    auto model = program.get<string>("model");
    auto score = program.get<string>("score");
    auto build = program.get<bool>("build");
    auto report = program.get<bool>("report");
    auto friedman = program.get<bool>("friedman");
    auto excel = program.get<bool>("excel");
    auto level = program.get<double>("level");
    auto results = platform::BestResults(platform::Paths::results(), score, model, friedman, level);
    if (build) {
        if (model == "any") {
            results.buildAll();
        } else {
            string fileName = results.build();
            cout << Colors::GREEN() << fileName << " created!" << Colors::RESET() << endl;
        }
    }
    if (report) {
        if (model == "any") {
            results.reportAll(excel);
        } else {
            results.reportSingle();
        }
    }
    return 0;
}
