#include <filesystem>
#include "platformUtils.h"
#include "Results.h"
#include "ReportConsole.h"
#include "ReportExcel.h"
#include "BestResult.h"
#include "Colors.h"
namespace platform {
    Result::Result(const string& path, const string& filename)
        : path(path)
        , filename(filename)
    {
        auto data = load();
        date = data["date"];
        score = 0;
        for (const auto& result : data["results"]) {
            score += result["score"].get<double>();
        }
        scoreName = data["score_name"];
        if (scoreName == BestResult::scoreName()) {
            score /= BestResult::score();
        }
        title = data["title"];
        duration = data["duration"];
        model = data["model"];
        complete = data["results"].size() > 1;
    }
    json Result::load() const
    {
        ifstream resultData(path + "/" + filename);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw invalid_argument("Unable to open result file. [" + path + "/" + filename + "]");
    }
    void Results::load()
    {
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != string::npos && filename.find("results_") == 0) {
                auto result = Result(path, filename);
                bool addResult = true;
                if (model != "any" && result.getModel() != model || scoreName != "any" && scoreName != result.getScoreName() || complete && !result.isComplete() || partial && result.isComplete())
                    addResult = false;
                if (addResult)
                    files.push_back(result);
            }
        }
    }
    string Result::to_string() const
    {
        stringstream oss;
        oss << date << " ";
        oss << setw(12) << left << model << " ";
        oss << setw(11) << left << scoreName << " ";
        oss << right << setw(11) << setprecision(7) << fixed << score << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << setw(1) << " " << completeString << "  ";
        oss << setw(9) << setprecision(3) << fixed << duration << " ";
        oss << setw(50) << left << title << " ";
        return  oss.str();
    }
    void Results::show() const
    {
        cout << Colors::GREEN() << "Results found: " << files.size() << endl;
        cout << "-------------------" << endl;
        if (complete) {
            cout << Colors::MAGENTA() << "Only listing complete results" << endl;
        }
        if (partial) {
            cout << Colors::MAGENTA() << "Only listing partial results" << endl;
        }
        auto i = 0;
        cout << Colors::GREEN() << " #  Date       Model        Score Name  Score       C/P Duration  Title" << endl;
        cout << "=== ========== ============ =========== =========== === ========= =============================================================" << endl;
        bool odd = true;
        for (const auto& result : files) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << result.to_string() << endl;
            if (i == max && max != 0) {
                break;
            }
            odd = !odd;
        }
    }
    int Results::getIndex(const string& intent) const
    {
        string color;
        if (intent == "delete") {
            color = Colors::RED();
        } else {
            color = Colors::YELLOW();
        }
        cout << color << "Choose result to " << intent << " (cancel=-1): ";
        string line;
        getline(cin, line);
        int index = stoi(line);
        if (index >= -1 && index < static_cast<int>(files.size())) {
            return index;
        }
        cout << "Invalid index" << endl;
        return -1;
    }
    void Results::report(const int index, const bool excelReport)
    {
        cout << Colors::YELLOW() << "Reporting " << files.at(index).getFilename() << endl;
        auto data = files.at(index).load();
        if (excelReport) {
            ReportExcel reporter(data, compare, workbook);
            reporter.show();
            openExcel = true;
            workbook = reporter.getWorkbook();
        } else {
            ReportConsole reporter(data, compare);
            reporter.show();
        }
    }
    void Results::showIndex(const int index, const int idx) const
    {
        auto data = files.at(index).load();
        if (idx < 0 or idx >= static_cast<int>(data["results"].size())) {
            cout << "Invalid index" << endl;
            return;
        }
        cout << Colors::YELLOW() << "Showing " << files.at(index).getFilename() << endl;
        ReportConsole reporter(data, idx);
        reporter.show();
    }
    void Results::menu()
    {
        char option;
        int index;
        bool finished = false;
        string filename, line, options = "qldhsre";
        while (!finished) {
            cout << Colors::RESET() << "Choose option (quit='q', list='l', delete='d', hide='h', sort='s', report='r', excel='e'): ";
            getline(cin, line);
            if (line.size() == 0)
                continue;
            if (options.find(line[0]) != string::npos) {
                if (line.size() > 1) {
                    cout << "Invalid option" << endl;
                    continue;
                }
                option = line[0];
            } else {
                if (all_of(line.begin(), line.end(), ::isdigit)) {
                    int idx = stoi(line);
                    if (indexList) {
                        // The value is about the files list
                        index = idx;
                        if (index >= 0 && index < files.size()) {
                            report(index, false);
                            indexList = false;
                            continue;
                        }
                    } else {
                        // The value is about the result showed on screen
                        showIndex(index, idx);
                        continue;
                    }
                }
                cout << "Invalid option" << endl;
                continue;
            }
            switch (option) {
                case 'q':
                    finished = true;
                    break;
                case 'l':
                    show();
                    indexList = true;
                    break;
                case 'd':
                    index = getIndex("delete");
                    if (index == -1)
                        break;
                    filename = files[index].getFilename();
                    cout << "Deleting " << filename << endl;
                    remove((path + "/" + filename).c_str());
                    files.erase(files.begin() + index);
                    cout << "File: " + filename + " deleted!" << endl;
                    show();
                    indexList = true;
                    break;
                case 'h':
                    index = getIndex("hide");
                    if (index == -1)
                        break;
                    filename = files[index].getFilename();
                    cout << "Hiding " << filename << endl;
                    rename((path + "/" + filename).c_str(), (path + "/." + filename).c_str());
                    files.erase(files.begin() + index);
                    show();
                    menu();
                    indexList = true;
                    break;
                case 's':
                    sortList();
                    indexList = true;
                    show();
                    break;
                case 'r':
                    index = getIndex("report");
                    if (index == -1)
                        break;
                    indexList = false;
                    report(index, false);
                    break;
                case 'e':
                    index = getIndex("excel");
                    if (index == -1)
                        break;
                    indexList = true;
                    report(index, true);
                    break;
                default:
                    cout << "Invalid option" << endl;
            }
        }
    }
    void Results::sortList()
    {
        cout << Colors::YELLOW() << "Choose sorting field (date='d', score='s', duration='u', model='m'): ";
        string line;
        char option;
        getline(cin, line);
        if (line.size() == 0)
            return;
        if (line.size() > 1) {
            cout << "Invalid option" << endl;
            return;
        }
        option = line[0];
        switch (option) {
            case 'd':
                sortDate();
                break;
            case 's':
                sortScore();
                break;
            case 'u':
                sortDuration();
                break;
            case 'm':
                sortModel();
                break;
            default:
                cout << "Invalid option" << endl;
        }
    }
    void Results::sortDate()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDate() > b.getDate();
            });
    }
    void Results::sortModel()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getModel() > b.getModel();
            });
    }
    void Results::sortDuration()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDuration() > b.getDuration();
            });
    }
    void Results::sortScore()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getScore() > b.getScore();
            });
    }
    void Results::manage()
    {
        if (files.size() == 0) {
            cout << "No results found!" << endl;
            exit(0);
        }
        sortDate();
        show();
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        cout << "Done!" << endl;
    }

}