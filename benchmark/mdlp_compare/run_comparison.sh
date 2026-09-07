#!/usr/bin/env bash
# Builds and runs the benchmark once per fimdlp version and compares results.
#
#   ./run_comparison.sh [--reps N] [--folds N] [--datasets a,b] [--models TAN,TANLd]
#
# Everything else (build dirs, json outputs, report) lands in this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${HERE}/../../tests/data"
VERSIONS=("2.1.3" "3.0.0")
BENCH_ARGS=("$@")

CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
JOBS=$(( CPUS > 7 ? CPUS - 7 : 1 ))

for v in "${VERSIONS[@]}"; do
    BUILD="${HERE}/build_${v}"
    echo ">>> [fimdlp ${v}] installing dependencies"
    conan install "${HERE}" -o mdlp_version="${v}" --build=missing -of "${BUILD}" -s build_type=Release
    echo ">>> [fimdlp ${v}] configuring"
    cmake -S "${HERE}" -B "${BUILD}" \
        -DCMAKE_TOOLCHAIN_FILE="${BUILD}/conan_toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release
    echo ">>> [fimdlp ${v}] building with ${JOBS} jobs"
    cmake --build "${BUILD}" --config Release --parallel "${JOBS}"
done

for v in "${VERSIONS[@]}"; do
    BUILD="${HERE}/build_${v}"
    echo
    echo ">>> [fimdlp ${v}] running benchmark"
    "${BUILD}/mdlp_bench" --data "${DATA}" --output "${HERE}/results_${v}.json" "${BENCH_ARGS[@]}"
done

echo
python3 "${HERE}/compare.py" "${HERE}/results_${VERSIONS[0]}.json" "${HERE}/results_${VERSIONS[1]}.json" \
    --markdown "${HERE}/REPORT.md"
