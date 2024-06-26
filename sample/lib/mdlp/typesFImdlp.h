// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <map>
#include <stdexcept>

using namespace std;
namespace mdlp {
    typedef float precision_t;
    typedef vector<precision_t> samples_t;
    typedef vector<int> labels_t;
    typedef vector<size_t> indices_t;
    typedef vector<precision_t> cutPoints_t;
    typedef map<pair<int, int>, precision_t> cacheEnt_t;
    typedef map<tuple<int, int, int>, precision_t> cacheIg_t;
}
#endif
