// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef CPPFIMDLP_H
#define CPPFIMDLP_H

#include "typesFImdlp.h"
#include "Metrics.h"
#include <limits>
#include <utility>
#include <string>

namespace mdlp {
    class CPPFImdlp {
    protected:
        size_t min_length = 3;
        int depth = 0;
        int max_depth = numeric_limits<int>::max();
        float proposed_cuts = 0;
        indices_t indices = indices_t();
        samples_t X = samples_t();
        labels_t y = labels_t();
        Metrics metrics = Metrics(y, indices);
        cutPoints_t cutPoints;
        size_t num_cut_points = numeric_limits<size_t>::max();
        labels_t discretizedData = labels_t();

        static indices_t sortIndices(samples_t&, labels_t&);

        void computeCutPoints(size_t, size_t, int);
        void resizeCutPoints();
        bool mdlp(size_t, size_t, size_t);
        size_t getCandidate(size_t, size_t);
        size_t compute_max_num_cut_points() const;
        pair<precision_t, size_t> valueCutPoint(size_t, size_t, size_t);

    public:
        CPPFImdlp();
        CPPFImdlp(size_t, int, float);
        ~CPPFImdlp();
        void fit(samples_t&, labels_t&);
        inline cutPoints_t getCutPoints() const { return cutPoints; };
        labels_t& transform(const samples_t&);
        inline int get_depth() const { return depth; };
        static inline string version() { return "1.1.2"; };
    };
}
#endif
