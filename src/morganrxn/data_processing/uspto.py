#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare USPTO reaction rules from datasetB.csv.

This version silences RDKit / rulesmith floods robustly by running the actual
processing in a child Python process with stdout/stderr redirected to DEVNULL.

Normal usage:

    python uspto.py --max-rows 1000

To debug RDKit logs:

    python uspto.py --max-rows 1000 --show-rdkit-logs
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit import rdBase

from morganrxn.core.paths import USPTO_DIR


# =================================================================================================
# Global RDKit log suppression.
# =================================================================================================

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


# =================================================================================================
# Reaction preparation.
# =================================================================================================

def prepare_reaction_rules(
    rxn: str,
    direction: str = "forward",
    verbose: bool = False,
) -> list[str]:
    """
    Prepare one reaction into cleaned mapped reaction rules.
    """
    if not isinstance(rxn, str) or not rxn.strip():
        return []

    try:
        from morganrxn.core.reaction_utils import prepare_and_clean_reaction_pipeline

        reaction_rules = prepare_and_clean_reaction_pipeline(
            reaction_rule=rxn.strip(),
            df_cofactors=None,
            direction=direction,
            verbose=verbose,
        )

        return sorted(set(reaction_rules))

    except Exception as e:
        if verbose:
            print("FAILED REACTION")
            print(rxn)
            print("ERROR")
            print(e)
        return []


# =================================================================================================
# IDs.
# =================================================================================================

def make_parent_reaction_id(
    row: pd.Series,
    row_index: int,
    id_col: str | None = None,
) -> str:
    """
    Build parent reaction ID.

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
# Worker processing.
# =================================================================================================

def process_datasetB_worker(
    input_csv: str | Path,
    output_tsv: str | Path,
    summary_json: str | Path,
    reaction_col: str = "rxnSmiles_Mapping_NameRxn",
    id_col: str | None = None,
    direction: str = "forward",
    max_rows: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Actual processing function.

    This function is intended to run inside a silent child process.
    """
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")

    input_csv = Path(input_csv)
    output_tsv = Path(output_tsv)
    summary_json = Path(summary_json)

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
    n_input_reactions_with_rules = 0
    n_failed_reactions = 0

    for i, row in df.iterrows():
        if max_rows is not None and n_processed_rows >= max_rows:
            break

        n_processed_rows += 1

        rxn = row[reaction_col]

        reaction_rules = prepare_reaction_rules(
            rxn=rxn,
            direction=direction,
            verbose=verbose,
        )

        if len(reaction_rules) == 0:
            n_failed_reactions += 1
            continue

        n_input_reactions_with_rules += 1

        parent_id = make_parent_reaction_id(
            row=row,
            row_index=i,
            id_col=id_col,
        )

        for split_idx, reaction_rule in enumerate(reaction_rules):
            rxn_id = f"{parent_id}__split{split_idx}"

            rows.append(
                {
                    "id": rxn_id,
                    "reaction": reaction_rule,
                }
            )

    out = pd.DataFrame(rows, columns=["id", "reaction"])

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_tsv, sep="\t", index=False)

    summary = {
        "input_csv_reactions": int(len(df)),
        "processed_rows": int(n_processed_rows),
        "input_reactions_with_rules": int(n_input_reactions_with_rules),
        "output_reaction_rules": int(len(out)),
        "failed_reactions": int(n_failed_reactions),
        "direction": direction,
        "reaction_col": reaction_col,
        "output": str(output_tsv),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# =================================================================================================
# Silent parent launcher.
# =================================================================================================

def run_silent_child(args) -> None:
    """
    Relaunch this script as a worker process.

    The worker stdout/stderr are redirected to DEVNULL, so RDKit floods cannot
    reach the console. The worker writes a JSON summary, and the parent prints it.
    """
    output_tsv = Path(args.output)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        summary_json = Path(tmp.name)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--reaction-col",
        str(args.reaction_col),
        "--direction",
        str(args.direction),
        "--summary-json",
        str(summary_json),
    ]

    if args.id_col is not None:
        cmd.extend(["--id-col", str(args.id_col)])

    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])

    if args.verbose:
        cmd.append("--verbose")

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        text=True,
    )

    if result.returncode != 0:
        print("Worker failed.")
        print(f"Return code: {result.returncode}")
        print("Run again with --show-rdkit-logs to debug.")
        return

    if not summary_json.exists():
        print("Worker finished, but no summary file was produced.")
        return

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    try:
        summary_json.unlink()
    except Exception:
        pass

    print(f"Input CSV reactions:             {summary['input_csv_reactions']}")
    print(f"Processed rows:                  {summary['processed_rows']}")
    print(f"Input reactions with rules:      {summary['input_reactions_with_rules']}")
    print(f"Output reaction rules:           {summary['output_reaction_rules']}")
    print(f"Failed reactions:                {summary['failed_reactions']}")
    print(f"Direction:                       {summary['direction']}")
    print(f"Reaction column:                 {summary['reaction_col']}")
    print(f"Output:                          {summary['output']}")


# =================================================================================================
# CLI.
# =================================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load USPTO datasetB.csv, prepare cleaned mapped reaction rules "
            "with prepare_and_clean_reaction_pipeline, and save a TSV file "
            "with two columns: id, reaction."
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
        default=USPTO_DIR / "processed" / "uspto_rules.tsv",
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
        "--direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help="Direction used to generate reaction rules.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed pipeline information. Suppressed unless --show-rdkit-logs is used.",
    )

    parser.add_argument(
        "--show-rdkit-logs",
        action="store_true",
        help="Do not use silent subprocess mode. Useful for debugging.",
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--summary-json",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.worker:
        if args.summary_json is None:
            raise ValueError("--summary-json is required in worker mode.")

        process_datasetB_worker(
            input_csv=args.input,
            output_tsv=args.output,
            summary_json=args.summary_json,
            reaction_col=args.reaction_col,
            id_col=args.id_col,
            direction=args.direction,
            max_rows=args.max_rows,
            verbose=args.verbose,
        )
        return

    if args.show_rdkit_logs:
        summary_json = Path(args.output).with_suffix(".summary.json")

        process_datasetB_worker(
            input_csv=args.input,
            output_tsv=args.output,
            summary_json=summary_json,
            reaction_col=args.reaction_col,
            id_col=args.id_col,
            direction=args.direction,
            max_rows=args.max_rows,
            verbose=args.verbose,
        )

        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"Input CSV reactions:             {summary['input_csv_reactions']}")
        print(f"Processed rows:                  {summary['processed_rows']}")
        print(f"Input reactions with rules:      {summary['input_reactions_with_rules']}")
        print(f"Output reaction rules:           {summary['output_reaction_rules']}")
        print(f"Failed reactions:                {summary['failed_reactions']}")
        print(f"Direction:                       {summary['direction']}")
        print(f"Reaction column:                 {summary['reaction_col']}")
        print(f"Output:                          {summary['output']}")

        return

    run_silent_child(args)


if __name__ == "__main__":
    main()