#!/usr/bin/env python3
"""Diff two xba2de_profile JSON runs.

    ./compare.py results/baseline.json results/h1-accessor.json

Speedups are reported per stage. Accuracy and model count are checked, not
timed: an optimization that is meant to preserve semantics must leave them
identical, and anything that moves is called out as a REGRESSION.
"""
import json
import sys

TIMINGS = [
    ("primitives_ms", "entropy", "ms"),
    ("primitives_ms", "mutual_information", "ms"),
    ("primitives_ms", "conditional_entropy3", "ms"),
    ("primitives_ms", "conditional_mutual_information", "ms"),
    ("select_k_pairs", "joint_relevance_s", "s"),
    ("select_k_pairs", "legacy_cmi_s", "s"),
    ("xsp2de", "fit_s", "s"),
    ("xsp2de", "predict_s", "s"),
    ("xba2de", "fit_s", "s"),
]
INVARIANTS = [("xba2de", "accuracy"), ("xba2de", "n_models")]


def speedup(before, after):
    if after <= 0:
        return "n/a"
    return f"{before / after:6.1f}x"


def main(path_a, path_b):
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    print(f"{a['label']}  ->  {b['label']}\n")

    by_label = {c["label"]: c for c in b["cases"]}
    problems = []

    for ca in a["cases"]:
        cb = by_label.get(ca["label"])
        if cb is None:
            print(f"=== {ca['label']}: missing in {path_b}, skipped\n")
            continue
        print(f"=== {ca['label']}")
        print(f"  {'stage':<34}{'before':>12}{'after':>12}{'speedup':>10}")
        for section, key, unit in TIMINGS:
            if section not in ca or section not in cb:
                continue
            if key not in ca[section] or key not in cb[section]:
                continue
            va, vb = ca[section][key], cb[section][key]
            name = f"{key} ({unit})"
            print(f"  {name:<34}{va:>12.4f}{vb:>12.4f}{speedup(va, vb):>10}")

        for section, key in INVARIANTS:
            if section not in ca or section not in cb:
                continue
            va, vb = ca[section][key], cb[section][key]
            same = abs(va - vb) < 1e-9 if isinstance(va, float) else va == vb
            mark = "ok" if same else "REGRESSION"
            print(f"  {key:<34}{va:>12}{vb:>12}{mark:>10}")
            if not same:
                problems.append(f"{ca['label']}: {key} {va} -> {vb}")
        print()

    if problems:
        print("SEMANTIC CHANGES DETECTED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("no semantic changes detected")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
