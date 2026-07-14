#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 of the pipeline: apply atom mapping to sanitized reaction SMILES.

Input
-----
A TSV file with at least:

    id    reaction

where ``reaction`` is a sanitized (unmapped) reaction SMILES, as produced by
stage 1 (``uspto.py`` / ``metanetx.py``).

Output
------
A TSV file with the same ``id`` column and the ``reaction`` column replaced by
its atom-mapped version (RXNMapper, with missing atoms completed):

    id    reaction

Reactions that fail to map are dropped, unless ``--keep-failed-rows`` is given
(in which case their ``reaction`` is left empty).

RXNMapper (transformers / torch) floods stdout and stderr. As in the other
scripts, the real work runs in a silent child process whose output is
redirected to DEVNULL unless ``--show-logs`` is used.

Examples
--------
    python map_reactions.py --data uspto
    python map_reactions.py --data metanetx
    python map_reactions.py --input path/to/reactions.tsv --output path/to/mapped.tsv
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit import rdBase

from morganrxn.core.paths import DATA_DIR
from morganrxn.core.mapping import map_reactions_with_rxnmapper


# =================================================================================================
# Global RDKit log suppression.
# =================================================================================================

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


# =================================================================================================
# Paths.
# =================================================================================================

def default_reactions_path(data_name: str) -> Path:
    """Default stage-1 (sanitized, unmapped) reactions path."""
    return DATA_DIR / data_name / "processed" / f"{data_name}_reactions.tsv"


def default_mapped_path(data_name: str) -> Path:
    """Default stage-2 (mapped) reactions path."""
    return DATA_DIR / data_name / "processed" / f"{data_name}_mapped.tsv"


# =================================================================================================
# Worker processing.
# =================================================================================================

def process_worker(
    input_tsv: str | Path,
    output_tsv: str | Path,
    summary_json: str | Path,
    id_col: str = "id",
    reaction_col: str = "reaction",
    batch_size: int = 32,
    chunksize: int = 5000,
    max_rows: int | None = None,
    keep_failed_rows: bool = False,
) -> None:
    """
    Map every reaction in the input TSV. Intended to run in a silent child
    process.
    """
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")

    input_tsv = Path(input_tsv)
    output_tsv = Path(output_tsv)
    summary_json = Path(summary_json)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(input_tsv, sep="\t", chunksize=chunksize, dtype=str)

    n_rows_read = 0
    n_mapped = 0
    n_failed = 0
    header_written = False

    for chunk in reader:
        required = {id_col, reaction_col}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(
                f"Missing columns in input file: {missing}. "
                f"Available columns: {list(chunk.columns)}"
            )

        if max_rows is not None:
            remaining = max_rows - n_rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining).copy()

        n_rows_read += len(chunk)

        ids = chunk[id_col].tolist()
        reactions = [
            r if isinstance(r, str) else ""
            for r in chunk[reaction_col].tolist()
        ]

        results = map_reactions_with_rxnmapper(
            reactions,
            batch_size=batch_size,
        )

        rows = []
        for reaction_id, result in zip(ids, results):
            mapped = result.get("mapped_rxn") if isinstance(result, dict) else None

            if mapped and ">>" in mapped:
                n_mapped += 1
                rows.append({"id": reaction_id, "reaction": mapped})
            else:
                n_failed += 1
                if keep_failed_rows:
                    rows.append({"id": reaction_id, "reaction": ""})

        out = pd.DataFrame(rows, columns=["id", "reaction"])
        out.to_csv(
            output_tsv,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

        print(
            f"Processed {n_rows_read} rows: "
            f"{n_mapped} mapped, {n_failed} failed."
        )

        if max_rows is not None and n_rows_read >= max_rows:
            break

    if not header_written:
        # Empty input: still write a valid header-only file.
        pd.DataFrame(columns=["id", "reaction"]).to_csv(
            output_tsv, sep="\t", index=False
        )

    summary = {
        "input": str(input_tsv),
        "output": str(output_tsv),
        "rows_read": int(n_rows_read),
        "mapped_reactions": int(n_mapped),
        "failed_reactions": int(n_failed),
        "batch_size": int(batch_size),
        "keep_failed_rows": bool(keep_failed_rows),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# =================================================================================================
# Silent parent launcher.
# =================================================================================================

def run_silent_child(args) -> None:
    """
    Relaunch this script as a worker process with stdout/stderr redirected to
    DEVNULL, so RXNMapper / torch floods cannot reach the console.
    """
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
        "--id-col",
        str(args.id_col),
        "--reaction-col",
        str(args.reaction_col),
        "--batch-size",
        str(args.batch_size),
        "--chunksize",
        str(args.chunksize),
        "--summary-json",
        str(summary_json),
    ]

    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])

    if args.keep_failed_rows:
        cmd.append("--keep-failed-rows")

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
        print("Run again with --show-logs to debug.")
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

    print(f"Rows read:                       {summary['rows_read']}")
    print(f"Mapped reactions:                {summary['mapped_reactions']}")
    print(f"Failed reactions:                {summary['failed_reactions']}")
    print(f"Output:                          {summary['output']}")


# =================================================================================================
# CLI.
# =================================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply RXNMapper atom mapping to a TSV of sanitized reactions "
            "(id, reaction), and save a TSV of mapped reactions (id, reaction)."
        )
    )

    parser.add_argument(
        "--data",
        choices=["metanetx", "uspto"],
        default="uspto",
        help=(
            "Dataset to process. When given, --input and --output default to "
            "data/<data>/processed/<data>_reactions.tsv and "
            "data/<data>/processed/<data>_mapped.tsv respectively."
        ),
    )

    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Input TSV file (id, reaction). Overrides --data.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output TSV file (id, mapped reaction). Overrides --data.",
    )

    parser.add_argument(
        "--id-col",
        default="id",
        help="Column containing reaction IDs.",
    )

    parser.add_argument(
        "--reaction-col",
        default="reaction",
        help="Column containing reaction SMILES to map.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="RXNMapper batch size.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of TSV rows read (and written) at a time.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )

    parser.add_argument(
        "--keep-failed-rows",
        action="store_true",
        help="Keep rows whose reaction failed to map, with an empty reaction.",
    )

    parser.add_argument(
        "--show-logs",
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


def resolve_paths(args) -> None:
    """Fill in --input / --output from --data when they are not given."""
    if args.input is None:
        if args.data is None:
            raise ValueError("Provide either --data or --input.")
        args.input = default_reactions_path(args.data)

    if args.output is None:
        if args.data is None:
            raise ValueError("Provide either --data or --output.")
        args.output = default_mapped_path(args.data)


def main() -> None:
    args = build_parser().parse_args()

    if args.worker:
        if args.summary_json is None:
            raise ValueError("--summary-json is required in worker mode.")

        process_worker(
            input_tsv=args.input,
            output_tsv=args.output,
            summary_json=args.summary_json,
            id_col=args.id_col,
            reaction_col=args.reaction_col,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            keep_failed_rows=args.keep_failed_rows,
        )
        return

    resolve_paths(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.show_logs:
        summary_json = Path(args.output).with_suffix(".summary.json")

        process_worker(
            input_tsv=args.input,
            output_tsv=args.output,
            summary_json=summary_json,
            id_col=args.id_col,
            reaction_col=args.reaction_col,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            keep_failed_rows=args.keep_failed_rows,
        )

        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print(f"Rows read:                       {summary['rows_read']}")
        print(f"Mapped reactions:                {summary['mapped_reactions']}")
        print(f"Failed reactions:                {summary['failed_reactions']}")
        print(f"Output:                          {summary['output']}")
        return

    run_silent_child(args)


if __name__ == "__main__":
    main()
