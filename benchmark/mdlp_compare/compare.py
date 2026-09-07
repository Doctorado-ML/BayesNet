#!/usr/bin/env python3
"""Compare two mdlp_bench result files (one per fimdlp version).

    python3 compare.py results_2.1.3.json results_3.0.0.json [--markdown REPORT.md]

Prints, per dataset and model, the accuracy and timings of both versions plus
the deltas, and checks whether the MDLP cut points are identical.
"""
import argparse
import json
import math
import sys

ACC_TOL = 1e-9      # accuracies are computed the same way, expect bit equality
CUT_TOL = 1e-6      # cut points are floats; tolerate representation noise


def load(path):
    with open(path) as fh:
        return json.load(fh)


def pct(new, old):
    """Relative change of `new` over `old`, in percent."""
    if old == 0:
        return float("nan")
    return (new - old) / old * 100.0


def compare_cutpoints(a, b):
    """Return (n_features, n_differing, max_abs_diff, structural_mismatches)."""
    features = sorted(set(a) | set(b))
    differing, max_diff, structural = 0, 0.0, []
    for f in features:
        ca, cb = a.get(f), b.get(f)
        if ca is None or cb is None:
            structural.append(f"{f}: missing in one version")
            differing += 1
            continue
        if len(ca) != len(cb):
            structural.append(f"{f}: {len(ca)} vs {len(cb)} cut points")
            differing += 1
            continue
        d = max((abs(x - y) for x, y in zip(ca, cb)), default=0.0)
        if d > CUT_TOL:
            differing += 1
        max_diff = max(max_diff, d)
    return len(features), differing, max_diff, structural


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--markdown", help="also write a markdown report to this file")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    vold, vnew = old["mdlp_version"], new["mdlp_version"]

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit(f"# mdlp {vold} vs {vnew}")
    emit()
    emit(f"- BayesNet: {old['bayesnet_version']}")
    emit(f"- Folds: {old['folds']} (seed {old['seed']}), torch threads: {old['threads']}")

    datasets = [d for d in old["datasets"] if d in new["datasets"]]

    # Settings can differ per dataset when a heavy one was measured in its own
    # --append run, so group the datasets by the settings they actually ran with.
    groups = {}
    for name in datasets:
        d = old["datasets"][name]
        key = (d.get("reps", old["reps"]), d.get("max_samples", old.get("max_samples", 0)))
        groups.setdefault(key, []).append(name)
    for (reps, cap), names in sorted(groups.items()):
        capped = f", subsampled to {cap}" if cap else ", all samples"
        who = "all datasets" if len(groups) == 1 else ", ".join(f"`{n}`" for n in names)
        emit(f"- Timing reps: {reps}{capped} — {who}")
    emit("- A dataset appears in a model's table only if both versions measured it "
         "successfully; anything left out is listed in the verdict.")
    emit()
    models = sorted({m for d in datasets for m in old["datasets"][d]["models"]})

    acc_mismatches = []
    cut_mismatches = []
    errors = []
    totals = {m: {"old": 0.0, "new": 0.0} for m in models}

    def usable(mo, mn, model, name):
        """Both sides must have run for the entry to be comparable."""
        for r, version in ((mo, vold), (mn, vnew)):
            if r is not None and "error" in r:
                entry = (model, name, version, r["error"].splitlines()[0])
                if entry not in errors:
                    errors.append(entry)
        return mo is not None and mn is not None and "error" not in mo and "error" not in mn

    for model in models:
        emit(f"## {model}")
        emit()
        header = (f"| dataset | n | feat | acc {vold} | acc {vnew} | Δacc | "
                  f"t {vold} (ms) | t {vnew} (ms) | Δt |")
        emit(header)
        emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in datasets:
            do, dn = old["datasets"][name], new["datasets"][name]
            mo, mn = do["models"].get(model), dn["models"].get(model)
            if not usable(mo, mn, model, name):
                continue
            ao, an = mo["accuracy_mean"], mn["accuracy_mean"]
            to, tn = mo["total_ms"], mn["total_ms"]
            totals[model]["old"] += to
            totals[model]["new"] += tn
            if abs(ao - an) > ACC_TOL:
                acc_mismatches.append((model, name, ao, an))
            emit(f"| {name} | {do['samples']} | {do['features']} | "
                 f"{ao:.5f} | {an:.5f} | {an - ao:+.5f} | "
                 f"{to:.1f} | {tn:.1f} | {pct(tn, to):+.1f}% |")
        to, tn = totals[model]["old"], totals[model]["new"]
        emit(f"| **total** | | | | | | **{to:.1f}** | **{tn:.1f}** | **{pct(tn, to):+.1f}%** |")
        emit()

    # Timing breakdown for the globally discretized model: the discretization
    # column is the one that isolates mdlp from the rest of BayesNet.
    disc_totals = None
    if "TAN" in models:
        emit("## TAN timing breakdown (discretization is pure mdlp)")
        emit()
        emit(f"| dataset | disc {vold} | disc {vnew} | Δdisc | fit {vold} | fit {vnew} | Δfit |")
        emit("|---|---:|---:|---:|---:|---:|---:|")
        sdo = sdn = sfo = sfn = 0.0
        for name in datasets:
            mo = old["datasets"][name]["models"].get("TAN")
            mn = new["datasets"][name]["models"].get("TAN")
            if not usable(mo, mn, "TAN", name):
                continue
            sdo += mo["discretize_ms"]
            sdn += mn["discretize_ms"]
            sfo += mo["fit_ms"]
            sfn += mn["fit_ms"]
            emit(f"| {name} | {mo['discretize_ms']:.1f} | {mn['discretize_ms']:.1f} | "
                 f"{pct(mn['discretize_ms'], mo['discretize_ms']):+.1f}% | "
                 f"{mo['fit_ms']:.1f} | {mn['fit_ms']:.1f} | "
                 f"{pct(mn['fit_ms'], mo['fit_ms']):+.1f}% |")
        emit(f"| **total** | **{sdo:.1f}** | **{sdn:.1f}** | **{pct(sdn, sdo):+.1f}%** | "
             f"**{sfo:.1f}** | **{sfn:.1f}** | **{pct(sfn, sfo):+.1f}%** |")
        emit()
        emit("`fit` is pure BayesNet and should be unaffected by the mdlp version; "
             "any difference there is measurement noise, and gives the scale of the "
             "noise floor for the other columns.")
        emit()
        disc_totals = (sdo, sdn)

    # Cut points: the direct check that both versions discretize identically.
    emit("## MDLP cut points (fold 0, global discretization)")
    emit()
    emit("| dataset | features | differing | max abs diff |")
    emit("|---|---:|---:|---:|")
    for name in datasets:
        mo = old["datasets"][name]["models"].get("TAN")
        mn = new["datasets"][name]["models"].get("TAN")
        if not usable(mo, mn, "TAN", name):
            continue
        n, diff, maxd, structural = compare_cutpoints(
            mo.get("cutpoints_fold0", {}), mn.get("cutpoints_fold0", {}))
        if diff:
            cut_mismatches.append((name, diff, maxd, structural))
        emit(f"| {name} | {n} | {diff} | {maxd:.3e} |")
    emit()

    # Structural comparison of the learnt networks.
    struct_mismatches = []
    for name in datasets:
        for model in models:
            mo = old["datasets"][name]["models"].get(model)
            mn = new["datasets"][name]["models"].get(model)
            if not usable(mo, mn, model, name):
                continue
            for key in ("nodes", "edges", "states"):
                if mo[key] != mn[key]:
                    struct_mismatches.append((name, model, key, mo[key], mn[key]))

    emit("## Verdict")
    emit()
    if errors:
        emit(f"- Excluded from the comparison, they failed on both versions alike "
             f"({len(errors)} case(s)):")
        for model, name, version, msg in errors:
            emit(f"  - `{model}` / `{name}` / mdlp {version}: {msg}")
    if acc_mismatches:
        emit(f"**{len(acc_mismatches)} accuracy differences:**")
        emit()
        for model, name, ao, an in acc_mismatches:
            emit(f"- `{model}` / `{name}`: {ao:.6f} -> {an:.6f} ({an - ao:+.6f})")
    else:
        emit("- Accuracy: **identical** on every dataset and model.")
    if cut_mismatches:
        emit(f"- Cut points: **differ** on {len(cut_mismatches)} dataset(s).")
        for name, diff, maxd, structural in cut_mismatches:
            emit(f"  - `{name}`: {diff} feature(s), max diff {maxd:.3e}")
            for s in structural[:10]:
                emit(f"    - {s}")
    else:
        emit("- Cut points: **identical** on every dataset.")
    if struct_mismatches:
        emit(f"- Learnt networks: **differ** in {len(struct_mismatches)} case(s).")
        for name, model, key, a, b in struct_mismatches[:20]:
            emit(f"  - `{model}` / `{name}` / {key}: {a} vs {b}")
    else:
        emit("- Learnt networks (nodes/edges/states): **identical**.")
    emit()
    if disc_totals:
        do_, dn_ = disc_totals
        verb = "faster" if dn_ < do_ else "slower"
        emit(f"- **Discretization only** (the part that is actually mdlp): "
             f"{do_:.1f} ms -> {dn_:.1f} ms ({abs(pct(dn_, do_)):.1f}% {verb} with {vnew}).")
    for model in models:
        to, tn = totals[model]["old"], totals[model]["new"]
        verb = "faster" if tn < to else "slower"
        emit(f"- `{model}` end-to-end time: {to:.1f} ms -> {tn:.1f} ms "
             f"({abs(pct(tn, to)):.1f}% {verb} with {vnew}).")

    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nMarkdown report written to {args.markdown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
