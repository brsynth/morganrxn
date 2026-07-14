#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 of the USPTO pipeline: sanitize reaction SMILES.

This step does *not* map and does *not* deduplicate. It only:
    - drops the agent side of `reactants>agents>products`,
    - sanitizes both sides (canonical SMILES, atom maps and stereo removed),
    - keeps the L2R direction only (substrates >> products),

and writes a TSV with two columns:

    id    reaction

The mapping is applied afterwards by `map_reactions.py` (stage 2), and the
deduplication / open-matter-loss filtering happens in
`create_reactionrules_from_mapped_rules.py` (stage 3).

Usage:

    python uspto.py --max-rows 1000
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit import rdBase

from morganrxn.core.paths import USPTO_DIR
from morganrxn.core.reaction_utils import sanitize_reaction, suppress_agent


# =================================================================================================
# Global RDKit log suppression.
# =================================================================================================

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


# =================================================================================================
# Reaction sanitation.
# =================================================================================================

def sanitize_forward_reaction(rxn: str) -> str | None:
    """
    Sanitize one raw USPTO reaction into an L2R reaction SMILES.

    Returns None when the reaction is empty, malformed, or collapses to an
    empty side after sanitation.
    """
    if not isinstance(rxn, str) or not rxn.strip():
        return None

    try:
        reaction = suppress_agent(rxn.strip())
        reaction = sanitize_reaction(reaction)
    except Exception:
        return None

    if ">>" not in reaction:
        return None

    left, right = reaction.split(">>", 1)

    if left.strip() == "" or right.strip() == "":
        return None

    return reaction


# =================================================================================================
# IDs.
# =================================================================================================

def make_reaction_id(
    row: pd.Series,
    row_index: int,
    id_col: str | None = None,
) -> str:
    """
    Build a reaction ID.

    Priority:
        1. If id_col is provided, use this column.
        2. Else, use patentID__rowIndex if patentID exists.
        3. Else, use rowIndex.
    """
    if id_col is not None:
        return str(row[id_col])

    if "patentID" in row.index and pd.notna(row["patentID"]):
        return f"{row['patentID']}__{row_index}"

    return str(row_index)


# =================================================================================================
# Processing.
# =================================================================================================

def process_datasetB(
    input_csv: str | Path,
    output_tsv: str | Path,
    reaction_col: str = "rxnSmiles_Mapping_NameRxn",
    id_col: str | None = None,
    max_rows: int | None = None,
) -> None:
    """
    Read datasetB.csv, sanitize each reaction (L2R only), and write id/reaction.
    """
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")

    input_csv = Path(input_csv)
    output_tsv = Path(output_tsv)

    df = pd.read_csv(input_csv)

    if reaction_col not in df.columns:
        raise ValueError(
            f"Column '{reaction_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if id_col is not None and id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    n_processed_rows = 0
    n_failed_reactions = 0

    for i, row in df.iterrows():
        if max_rows is not None and n_processed_rows >= max_rows:
            break

        n_processed_rows += 1

        reaction = sanitize_forward_reaction(row[reaction_col])

        if reaction is None:
            n_failed_reactions += 1
            continue

        rows.append(
            {
                "id": make_reaction_id(row=row, row_index=i, id_col=id_col),
                "reaction": reaction,
            }
        )

    out = pd.DataFrame(rows, columns=["id", "reaction"])

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_tsv, sep="\t", index=False)

    print(f"Input CSV reactions:             {len(df)}")
    print(f"Processed rows:                  {n_processed_rows}")
    print(f"Sanitized reactions written:     {len(out)}")
    print(f"Failed reactions:                {n_failed_reactions}")
    print(f"Reaction column:                 {reaction_col}")
    print(f"Output:                          {output_tsv}")


# =================================================================================================
# CLI.
# =================================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load USPTO datasetB.csv, sanitize each reaction SMILES (L2R only, "
            "no mapping, no deduplication), and save a TSV file with two "
            "columns: id, reaction."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        default=USPTO_DIR / "datasetB.csv",
        help="Input CSV file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=USPTO_DIR / "processed" / "uspto_reactions.tsv",
        help="Output TSV file.",
    )

    parser.add_argument(
        "--reaction-col",
        default="rxnSmiles_Mapping_NameRxn",
        help="Column containing reaction SMILES.",
    )

    parser.add_argument(
        "--id-col",
        default=None,
        help=(
            "Optional ID column. If not given, uses patentID__rowIndex "
            "when patentID exists, otherwise rowIndex."
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    process_datasetB(
        input_csv=args.input,
        output_tsv=args.output,
        reaction_col=args.reaction_col,
        id_col=args.id_col,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
