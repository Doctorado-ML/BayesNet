#!/usr/bin/env bash
# Capture a labelled XBA2DE performance baseline.
#
#   ./run.sh baseline          -> results/baseline.json + results/baseline-scaling.json
#   ./run.sh h1-accessor       -> the same two files under a different label
#
# Two sweeps, because they answer different questions:
#   * "fit"     small cases with the whole ensemble trained, so accuracy and
#               model count can be diffed against the baseline. An optimization
#               that claims to preserve semantics must not move them.
#   * "scaling" larger cases with --skip-full, to watch the n^2 and m growth of
#               the pair ranking without waiting for a full fit.
set -euo pipefail

LABEL="${1:-baseline}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/build_Release/benchmark/xba2de_profile/xba2de_profile"
OUT="$HERE/results"

if [[ ! -x "$BIN" ]]; then
    echo "profiler not built. Run:" >&2
    echo "  cmake -S $ROOT -B $ROOT/build_Release -DENABLE_BENCHMARK=ON" >&2
    echo "  cmake --build $ROOT/build_Release --target xba2de_profile" >&2
    exit 1
fi
mkdir -p "$OUT"

echo "### full-fit sweep (label: $LABEL)"
"$BIN" --label "$LABEL" --reps 3 \
    --synthetic 20,2000,4 \
    --synthetic 20,5000,4 \
    --json "$OUT/$LABEL.json"

echo
echo "### scaling sweep, ranking only (label: $LABEL)"
"$BIN" --label "$LABEL-scaling" --reps 1 --skip-full \
    --synthetic 40,5000,4 \
    --synthetic 80,5000,4 \
    --synthetic 40,20000,4 \
    --json "$OUT/$LABEL-scaling.json"
