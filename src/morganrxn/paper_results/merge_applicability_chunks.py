#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge applicability-benchmark chunks (``--target-start/--target-end``) or radius
groups into one table: additive counters are summed and rates recomputed.

Diversity ratios are set to NaN for chunked runs (unique sets are not additive).

Example:
    python merge_applicability_chunks.py \
        --inputs results/one_step_accuracy_split/fp1024/applicability_r0_mnx_c*.xlsx \
        --out-xlsx results/one_step_accuracy_split/fp1024/applicability_r0_mnx_merged.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = [
    "benchmark_name", "database_name", "applicability_mode", "radius",
    "fpSize", "folded", "custom", "min_smi_sub_atoms",
]
SUM_COLS = [
    "n_total_cases", "n_succeeded", "n_failed", "n_ecfp_applies",
    "n_fail_graph_level_false_positive", "n_fail_fingerprint_translation_mismatch",
    "n_fail_product_ecfp_computation_error",
    "n_targets_total", "n_targets_valid", "n_targets_with_ecfp_apply",
    "n_invalid_targets", "n_target_ecfp_errors", "n_one_step_errors",
    "n_apply_errors", "n_product_ecfp_errors", "elapsed_s",
]


def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    sum_cols = [c for c in SUM_COLS if c in df.columns]
    keys = [k for k in KEYS if k in df.columns]
    out = df.groupby(keys, dropna=False)[sum_cols].sum().reset_index()

    n = out["n_total_cases"].replace(0, np.nan)
    nf = out["n_failed"].replace(0, np.nan)
    out["accuracy"] = out["n_succeeded"] / n
    out["failure_rate"] = out["n_failed"] / n
    out["target_coverage"] = out["n_targets_with_ecfp_apply"] / out["n_targets_valid"].replace(0, np.nan)
    if "n_fail_graph_level_false_positive" in out:
        out["graph_level_false_positive_rate"] = out["n_fail_graph_level_false_positive"] / nf
        out["fingerprint_translation_mismatch_rate"] = out["n_fail_fingerprint_translation_mismatch"] / nf
    out["mean_candidates_per_target"] = out["n_total_cases"] / out["n_targets_valid"].replace(0, np.nan)
    out["n_chunks"] = df.groupby(keys, dropna=False).size().values
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="xlsx files (globs expanded by the shell)")
    p.add_argument("--out-xlsx", type=Path, required=True)
    args = p.parse_args()

    frames = [pd.read_excel(f, sheet_name="details") for f in args.inputs]
    merged = merge(frames)
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(args.out_xlsx, sheet_name="details", index=False)
    print(merged.to_string(index=False))
    print(f"Saved: {args.out_xlsx}")


if __name__ == "__main__":
    main()
