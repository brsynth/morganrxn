#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute overlap statistics between MetaNetX and USPTO reaction representations.

For each ECFP radius h, the script loads ReactionRules for two databases and computes:
    - number of unique reaction ECFP vectors in MetaNetX
    - number of unique reaction ECFP vectors in USPTO
    - number of shared reaction ECFP vectors
    - number of unique reaction-center ECFP vectors in MetaNetX
    - number of unique reaction-center ECFP vectors in USPTO
    - number of shared reaction-center ECFP vectors

Outputs:
    1. CSV file with one row per radius and representation
    2. LaTeX table printed to stdout

Examples
--------
Default run, radii 0 to 5:

    python compute_reaction_vector_overlap.py

Specify database names:

    python compute_reaction_vector_overlap.py \
        --metanetx-database-name metanetx \
        --uspto-database-name uspto

Radius 2 only:

    python compute_reaction_vector_overlap.py --radii 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from morganrxn.core.paths import RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules


# ======================================================================================
# Helpers
# ======================================================================================

def parse_radii(value: str):
    radii = []

    for x in str(value).split(","):
        x = x.strip()
        if x:
            radii.append(int(x))

    radii = sorted(set(radii))

    if not radii:
        raise ValueError("No valid radius found.")

    return radii


def make_ecfp_params(radius: int, fp_size: int, folded: bool, custom: bool) -> dict:
    return {
        "radius": int(radius),
        "fpSize": int(fp_size),
        "folded": bool(folded),
        "custom": bool(custom),
    }


def vector_to_key(vector):
    """
    Convert a vector-like object into a hashable key.

    Works for lists, tuples, numpy arrays, and integer/float counted vectors.
    Values are cast to int because reaction ECFPs are counted fingerprints.
    """
    arr = np.asarray(vector).ravel()
    return tuple(arr.astype(int).tolist())


def unique_vector_set(vectors):
    """
    Return the set of unique vector keys.
    """
    return {vector_to_key(v) for v in vectors}


def compute_overlap_stats(metanetx_rules, uspto_rules, radius: int):
    """
    Compute overlap statistics for one radius.
    """
    mnx_reaction = unique_vector_set(metanetx_rules.ecfp_reaction)
    usp_reaction = unique_vector_set(uspto_rules.ecfp_reaction)

    mnx_center = unique_vector_set(metanetx_rules.ecfp_reaction_center)
    usp_center = unique_vector_set(uspto_rules.ecfp_reaction_center)

    rows = [
        {
            "radius": int(radius),
            "representation": "Reaction ECFP",
            "metanetx_unique_vectors": len(mnx_reaction),
            "uspto_unique_vectors": len(usp_reaction),
            "shared_vectors": len(mnx_reaction & usp_reaction),
        },
        {
            "radius": int(radius),
            "representation": "Reaction-center ECFP",
            "metanetx_unique_vectors": len(mnx_center),
            "uspto_unique_vectors": len(usp_center),
            "shared_vectors": len(mnx_center & usp_center),
        },
    ]

    return rows


def format_int(x):
    return f"{int(x):,}"


def print_latex_table(df: pd.DataFrame, fp_size: int):
    """
    Print one compact LaTeX table containing all radii.
    """
    print()
    print(r"\begin{table}[ht]")
    print(r"\footnotesize")
    print(r"\centering")
    print(
        rf"\caption{{Overlap between MetaNetX and USPTO-50k reaction representations "
        rf"across ECFP radii. Both datasets were computed with folded counted ECFPs "
        rf"of dimension $d={fp_size}$.}}"
    )
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(
        r"Radius & Representation "
        r"& MetaNetX unique vectors "
        r"& USPTO-50k unique vectors "
        r"& Shared vectors \\")
    print(r"\midrule")

    for radius in sorted(df["radius"].unique()):
        sub = df[df["radius"] == radius]

        for j, (_, row) in enumerate(sub.iterrows()):
            radius_label = str(radius) if j == 0 else ""
            print(
                f"{radius_label} & {row['representation']} "
                f"& {format_int(row['metanetx_unique_vectors'])} "
                f"& {format_int(row['uspto_unique_vectors'])} "
                f"& {format_int(row['shared_vectors'])} \\\\" 
            )

        if radius != sorted(df["radius"].unique())[-1]:
            print(r"\addlinespace")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:reaction-vector-overlap-all-radii}")
    print(r"\end{table}")


# ======================================================================================
# Main
# ======================================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute MetaNetX/USPTO overlap statistics for reaction ECFPs."
    )

    parser.add_argument(
        "--radii",
        default="0,1,2,3,4,5",
        help="Comma-separated radii. Default: 0,1,2,3,4,5.",
    )

    parser.add_argument(
        "--fp-size",
        type=int,
        default=1024,
        help="Folded ECFP dimension. Default: 1024.",
    )

    parser.add_argument(
        "--metanetx-database-name",
        default="metanetx",
        help="ReactionRules database name for MetaNetX. Default: metanetx.",
    )

    parser.add_argument(
        "--uspto-database-name",
        default="uspto",
        help="ReactionRules database name for USPTO. Default: uspto.",
    )

    parser.add_argument(
        "--unfolded",
        action="store_true",
        help="Load unfolded ReactionRules instead of folded ones.",
    )

    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom ReactionRules loading parameters.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "RESULTS_DIR / reaction_vector_overlap."
        ),
    )

    parser.add_argument(
        "--output-name",
        default="reaction_vector_overlap_by_radius.csv",
        help="Output CSV filename.",
    )

    return parser


def main():
    args = build_parser().parse_args()

    radii = parse_radii(args.radii)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else RESULTS_DIR / "reaction_vector_overlap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    all_rows = []

    print("Reaction-vector overlap benchmark")
    print("=================================")
    print("radii:", radii)
    print("fp_size:", args.fp_size)
    print("folded:", not args.unfolded)
    print("custom:", args.custom)
    print("metanetx_database_name:", args.metanetx_database_name)
    print("uspto_database_name:", args.uspto_database_name)
    print("output_path:", output_path)

    for radius in radii:
        ecfp_params = make_ecfp_params(
            radius=radius,
            fp_size=args.fp_size,
            folded=not args.unfolded,
            custom=args.custom,
        )

        print()
        print("=" * 80)
        print(f"Radius {radius}")
        print("=" * 80)
        print("ecfp_params:", ecfp_params)

        print("Loading MetaNetX ReactionRules...")
        metanetx_rules = ReactionRules.load(
            database_name=args.metanetx_database_name,
            ecfp_params=ecfp_params,
        )
        metanetx_rules.filter_by_smi_sub_atoms(min_atoms=5)
        metanetx_rules.drop_duplicates()

        print("Loading USPTO ReactionRules...")
        uspto_rules = ReactionRules.load(
            database_name=args.uspto_database_name,
            ecfp_params=ecfp_params,
        )
        uspto_rules.filter_by_smi_sub_atoms(min_atoms=5)
        uspto_rules.drop_duplicates()

        rows = compute_overlap_stats(
            metanetx_rules=metanetx_rules,
            uspto_rules=uspto_rules,
            radius=radius,
        )

        for row in rows:
            print(
                f"{row['representation']}: "
                f"MetaNetX={format_int(row['metanetx_unique_vectors'])}, "
                f"USPTO={format_int(row['uspto_unique_vectors'])}, "
                f"shared={format_int(row['shared_vectors'])}"
            )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    print()
    print("Saved CSV:", output_path)

    print_latex_table(df=df, fp_size=args.fp_size)


if __name__ == "__main__":
    main()
