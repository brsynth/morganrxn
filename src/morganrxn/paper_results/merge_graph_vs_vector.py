#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the per-target CSVs of graph_vs_vector_timing.py into one row per
(database, fpSize, radius): totals, medians and quartiles of each route, speedups
(ratio of totals and median of per-target ratios), Table 2 accuracy, and checks.

Example:
    python merge_graph_vs_vector.py --inputs results/graph_vs_vector/*.csv \
        --out-csv results/graph_vs_vector/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = [
    "t_vector_s", "t_graph_s", "t_vector_then_graph_s",
    "t_prefilter_s", "t_subgraph_all_s", "t_prefilter_then_subgraph_s",
]


def summarize(g: pd.DataFrame) -> pd.Series:
    out = {
        "n_targets": g["target_idx"].nunique(),
        "n_rules": int(g["n_rules"].iloc[0]),
        "mean_candidates_per_target": g["n_candidates_unique"].mean(),
        "accuracy": g["n_correct"].sum() / g["n_cases"].sum() if g["n_cases"].sum() else np.nan,
        "n_cases": int(g["n_cases"].sum()),
        "missed_applied_by_prefilter": int(g["missed_applied_by_prefilter"].sum()),
        "missed_subgraph_by_prefilter": int(g["missed_subgraph_by_prefilter"].sum()),
        # graph times of censored targets are lower bounds, so graph totals and
        # graph-vs-vector speedups are lower bounds when n_graph_censored > 0
        "n_graph_censored": int(g["graph_censored"].sum()),
    }
    for col in ROUTES:
        s = g[col]
        name = col[2:-2]
        out[f"{name}_total_h"] = s.sum() / 3600
        out[f"{name}_median_s"] = s.median()
        out[f"{name}_q1_s"] = s.quantile(0.25)
        out[f"{name}_q3_s"] = s.quantile(0.75)
        out[f"{name}_max_s"] = s.max()
    for num, den, name in [
        ("t_graph_s", "t_vector_then_graph_s", "speedup_vector_then_graph_vs_graph"),
        ("t_graph_s", "t_vector_s", "speedup_vector_vs_graph"),
        ("t_subgraph_all_s", "t_prefilter_s", "speedup_prefilter_vs_subgraph"),
    ]:
        out[f"{name}_of_totals"] = g[num].sum() / g[den].sum()
        out[f"{name}_median_per_target"] = (g[num] / g[den]).median()
    return pd.Series(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--graph-budget-s", type=float, default=1800.0,
                   help="Same per-target cap as the runs; applied to every row so that "
                        "rows computed without a cap follow the same censoring rule.")
    p.add_argument("--exclude-targets", default="",
                   help="Comma-separated target indices dropped from every group.")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.inputs], ignore_index=True)
    n_dup = df.duplicated(["database", "fpSize", "radius", "target_idx"]).sum()
    if n_dup:
        print(f"WARNING: {n_dup} duplicated (radius, target) rows; keeping the first.")
        df = df.drop_duplicates(["database", "fpSize", "radius", "target_idx"])

    excluded = {int(x) for x in args.exclude_targets.split(",") if x.strip()}
    df = df[~df["target_idx"].isin(excluded)].copy()
    if "graph_censored" not in df:
        df["graph_censored"] = False
    df["graph_censored"] = df["graph_censored"].fillna(False).astype(bool) | (
        df["t_graph_s"] >= args.graph_budget_s
    )
    df["t_graph_s"] = df["t_graph_s"].clip(upper=args.graph_budget_s)

    summary = df.groupby(["database", "fpSize", "radius"]).apply(summarize).reset_index()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(summary.round(4).to_string(index=False))
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
