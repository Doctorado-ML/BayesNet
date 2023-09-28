#ifndef BESTRESULTS_EXCEL_H
#define BESTRESULTS_EXCEL_H
#include "ExcelFile.h"
#include <vector>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace platform {
    class BestResultsExcel : ExcelFile {
    public:
        BestResultsExcel(vector<string> models, vector<string> datasets, json table, bool friedman);
        ~BestResultsExcel();
        void build();
    private:
        void header();
        void body();
        void footer();
        void formatColumns();
        const string fileName = "BestResults.xlsx";
        vector<string> models;
        vector<string> datasets;
        json table;
        bool friedman;

    };
}
#endif //BESTRESULTS_EXCEL_H