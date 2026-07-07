#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create ReactionRules from mapped reaction rules for multiple ECFP radii.

Input
-----
A TSV file with at least:

    id    reaction

where `reaction` is a mapped reaction SMILES/SMARTS-like rule:

    mapped_substrate>>mapped_product

Default input paths
-------------------
    data/uspto/processed/uspto_rules.tsv
    data/metanetx/processed/metanetx_rules.tsv

Output
------
One ReactionRules .npz file per radius, saved by ReactionRules.save(), e.g.

    data/processed/reaction_rules/uspto/ecfp_r0_fp1024_folded_uncustom/rules.npz
    data/processed/reaction_rules/uspto/ecfp_r1_fp1024_folded_uncustom/rules.npz
    data/processed/reaction_rules/uspto/ecfp_r2_fp1024_folded_uncustom/rules.npz

Examples
--------
USPTO radii 0 to 5:

    python create_reactionrules_from_mapped_rules.py --data uspto --radii 0,1,2,3,4,5

MetaNetX radii 0 to 5:

    python create_reactionrules_from_mapped_rules.py --data metanetx --radii 0,1,2,3,4,5

Test on first 1000 rows:

    python create_reactionrules_from_mapped_rules.py --data uspto --radii 0,1,2 --max-rows 1000

Single radius still works:

    python create_reactionrules_from_mapped_rules.py --data uspto --radius 2
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from rdkit import RDLogger
from rdkit import rdBase

from rulesmith import SmartsFeatures, SmartsParams

from morganrxn.core.molecule_utils import sanitize_smiles
from morganrxn.core.paths import DATA_DIR
from morganrxn.core.reaction_rules import ReactionRules
from morganrxn.core.reaction_utils import (
    process_a_reaction,
    has_open_matter_loss,
)


# ======================================================================================
# Silence RDKit / C++ stderr floods
# ======================================================================================

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


@contextmanager
def suppress_stderr_fd():
    """
    Suppress low-level C/C++ stderr messages.

    RDKit sometimes writes directly to stderr, so RDLogger.DisableLog is not
    always sufficient.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass

        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


# ======================================================================================
# Helpers
# ======================================================================================

def default_rules_path(data_name: str) -> Path:
    """
    Default mapped-rules path.

    Example
    -------
    data/uspto/processed/uspto_rules.tsv
    """
    return DATA_DIR / data_name / "processed" / f"{data_name}_rules.tsv"


def parse_radii(radii_value, fallback_radius):
    """
    Parse radii from CLI.

    Priority:
        1. --radii if provided
        2. --radius otherwise

    Examples
    --------
    "0,1,2,3,4,5" -> [0, 1, 2, 3, 4, 5]
    """
    if radii_value is None or str(radii_value).strip() == "":
        return [int(fallback_radius)]

    radii = []

    for x in str(radii_value).split(","):
        x = x.strip()

        if x == "":
            continue

        radii.append(int(x))

    radii = sorted(set(radii))

    if not radii:
        raise ValueError("No valid radius found in --radii.")

    return radii


def make_ecfp_params(radius: int, fp_size: int, folded: bool, custom: bool) -> dict:
    return {
        "radius": int(radius),
        "fpSize": int(fp_size),
        "folded": bool(folded),
        "custom": bool(custom),
    }


def make_smarts_params(custom: bool):
    """
    SMARTS parameters used for template extraction.

    For the non-custom case, this follows the richer atom description used in
    the previous scripts.
    """
    if custom:
        return SmartsParams(
            SmartsFeatures.SYMBOL
            | SmartsFeatures.DEGREE
            | SmartsFeatures.AROMATICITY
            | SmartsFeatures.CHARGE
            | SmartsFeatures.MAP_NUMBER
            | SmartsFeatures.KEEP_H_ON_EDGE_CASES
        )

    return SmartsParams(
        SmartsFeatures.SYMBOL
        | SmartsFeatures.MAP_NUMBER
        | SmartsFeatures.H_COUNT
        | SmartsFeatures.AROMATICITY
        | SmartsFeatures.ALIPHATICITY
        | SmartsFeatures.DEGREE
        | SmartsFeatures.VALENCE
        | SmartsFeatures.CONNECTIVITY
        | SmartsFeatures.RING
        | SmartsFeatures.CHARGE
        | SmartsFeatures.KEEP_H_ON_EDGE_CASES
    )


def is_valid_reaction_string(reaction: str) -> bool:
    if not isinstance(reaction, str):
        return False

    reaction = reaction.strip()

    if ">>" not in reaction:
        return False

    left, right = reaction.split(">>", maxsplit=1)

    if left.strip() == "" or right.strip() == "":
        return False

    return True


def is_valid_template(template_reaction: str) -> bool:
    if template_reaction is None:
        return False

    if ">>" not in template_reaction:
        return False

    left, right = template_reaction.split(">>", maxsplit=1)

    if left.strip() == "" or right.strip() == "":
        return False

    if left.strip() == right.strip():
        return False

    return True


def count_products(reaction: str) -> int:
    """
    Count products from right side of reaction.
    """
    _, right = reaction.split(">>", maxsplit=1)

    return len(
        [
            x
            for x in right.split(".")
            if x.strip()
        ]
    )


def get_substrate_side(reaction: str) -> str:
    return reaction.split(">>", maxsplit=1)[0].strip()


def process_rule_row(
    row: dict,
    id_col: str,
    reaction_col: str,
    sp_min,
    ecfp_params: dict,
    template_radius: int,
    filter_open_matter_loss: bool,
) -> tuple[bool, dict]:
    """
    Process one mapped rule row.

    Returns
    -------
    ok, payload

    If ok:
        payload contains fields for ReactionRules.add()

    If failed:
        payload is a debug row.
    """
    reaction_id = str(row[id_col])
    reaction = row[reaction_col]

    if not is_valid_reaction_string(reaction):
        return False, {
            "id": reaction_id,
            "reaction": reaction,
            "stage": "invalid_reaction_string",
            "error": "",
        }

    reaction = str(reaction).strip()

    try:
        with suppress_stderr_fd(), rdBase.BlockLogs():
            (
                template_reaction,
                ecfp_reaction_center,
                ecfp_reaction,
            ) = process_a_reaction(
                reaction_smiles=reaction,
                sp_min=sp_min,
                ecfp_params=ecfp_params,
                template_radius=template_radius,
                verbose=False,
            )

    except Exception as exc:
        return False, {
            "id": reaction_id,
            "reaction": reaction,
            "stage": "process_a_reaction",
            "error": repr(exc),
        }

    if not is_valid_template(template_reaction):
        return False, {
            "id": reaction_id,
            "reaction": reaction,
            "stage": "invalid_template",
            "error": str(template_reaction),
        }

    if filter_open_matter_loss:
        try:
            with suppress_stderr_fd(), rdBase.BlockLogs():
                if has_open_matter_loss(template_reaction):
                    return False, {
                        "id": reaction_id,
                        "reaction": reaction,
                        "stage": "open_matter_loss",
                        "error": template_reaction,
                    }
        except Exception as exc:
            return False, {
                "id": reaction_id,
                "reaction": reaction,
                "stage": "open_matter_loss_error",
                "error": repr(exc),
            }

    return True, {
        "template_reaction": template_reaction,
        "ecfp_reaction": tuple(int(x) for x in ecfp_reaction),
        "ecfp_reaction_center": tuple(int(x) for x in ecfp_reaction_center),
        "smi_sub": sanitize_smiles(get_substrate_side(reaction)),
        "nb_prod": count_products(reaction),
        "reaction_monocomp_id": reaction_id,
        "reaction_id": reaction_id,
    }


def iter_rule_chunks(
    input_path: Path,
    id_col: str,
    reaction_col: str,
    chunksize: int,
    max_rows: int | None,
):
    """
    Stream the input TSV once.

    This avoids reading the file separately for each radius.
    """
    reader = pd.read_csv(
        input_path,
        sep="\t",
        chunksize=chunksize,
    )

    n_rows_read = 0

    for chunk_index, chunk in enumerate(reader, start=1):
        required_columns = {id_col, reaction_col}
        missing_columns = required_columns - set(chunk.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in input file: {missing_columns}. "
                f"Available columns: {list(chunk.columns)}"
            )

        if max_rows is not None:
            remaining = max_rows - n_rows_read

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.head(remaining).copy()

        n_rows_read += len(chunk)

        yield chunk_index, chunk

        if max_rows is not None and n_rows_read >= max_rows:
            break


def save_debug_rows(debug_rows, debug_output: Path):
    if not debug_rows:
        return

    debug_output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        debug_rows,
        columns=["id", "reaction", "stage", "error"],
    ).to_csv(
        debug_output,
        sep="\t",
        index=False,
    )

    print()
    print("Debug rows:", len(debug_rows))
    print("Saved debug to:", debug_output)


def process_one_radius(
    radius: int,
    args,
    input_path: Path,
    database_name: str,
    sp_min,
):
    """
    Compute and save ReactionRules for one ECFP radius.
    """
    debug_output = (
        Path(args.debug_output)
        if args.debug_output is not None and len(args.radii_list) == 1
        else input_path.with_name(f"{input_path.stem}_ecfp_r{radius}_debug.tsv")
    )

    ecfp_params = make_ecfp_params(
        radius=radius,
        fp_size=args.fp_size,
        folded=not args.unfolded,
        custom=args.custom,
    )

    template_radius = (
        args.template_radius
        if args.template_radius is not None
        else 2 * radius
    )

    print()
    print("=" * 100)
    print(f"Creating ReactionRules for radius {radius}")
    print("=" * 100)
    print("data:", args.data)
    print("input:", input_path)
    print("database_name:", database_name)
    print("ecfp_params:", ecfp_params)
    print("template_radius:", template_radius)
    print("max_rows:", args.max_rows)
    print("chunksize:", args.chunksize)
    print("filter_open_matter_loss:", args.filter_open_matter_loss)
    print("debug_output:", debug_output)

    reaction_rules = ReactionRules(
        database_name=database_name,
        ecfp_params=ecfp_params,
    )

    n_rows_read = 0
    n_rows_ok = 0
    debug_rows = []
    stage_counts = {}

    for chunk_index, chunk in iter_rule_chunks(
        input_path=input_path,
        id_col=args.id_col,
        reaction_col=args.reaction_col,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    ):
        if chunk_index == 1:
            print()
            print("First chunk:")
            print(chunk.head())

        for row in chunk.to_dict(orient="records"):
            n_rows_read += 1

            ok, payload = process_rule_row(
                row=row,
                id_col=args.id_col,
                reaction_col=args.reaction_col,
                sp_min=sp_min,
                ecfp_params=ecfp_params,
                template_radius=template_radius,
                filter_open_matter_loss=args.filter_open_matter_loss,
            )

            if ok:
                reaction_rules.add(
                    rule=payload["template_reaction"],
                    ecfp_reaction=payload["ecfp_reaction"],
                    ecfp_reaction_center=payload["ecfp_reaction_center"],
                    smi_sub=payload["smi_sub"],
                    nb_prod=payload["nb_prod"],
                    reaction_monocomp_id=payload["reaction_monocomp_id"],
                    reaction_id=payload["reaction_id"],
                )

                n_rows_ok += 1
                stage_counts["ok"] = stage_counts.get("ok", 0) + 1

            else:
                debug_rows.append(payload)
                stage = payload["stage"]
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        print(
            f"[radius {radius}] Processed chunk {chunk_index}: "
            f"{n_rows_read} rows read, "
            f"{n_rows_ok} rules kept, "
            f"{len(debug_rows)} failed."
        )

    print()
    print(f"Before filters | radius {radius}")
    print("=" * (24 + len(str(radius))))
    print("Rows read:", n_rows_read)
    print("Rules kept:", n_rows_ok)
    print("ReactionRules length:", len(reaction_rules))
    print("Status counts:")
    print(pd.Series(stage_counts).sort_values(ascending=False))

    print()
    print("Computing scores...")
    reaction_rules.compute_score()

    if not args.keep_zero_ecfps:
        print()
        print("Dropping zero ECFPs...")
        reaction_rules.drop_zero_ECFPs(verbose=True)

    if not args.keep_duplicates:
        print()
        print("Dropping duplicates...")
        reaction_rules.drop_duplicates(verbose=True)

    print()
    print(f"After filters | radius {radius}")
    print("=" * (23 + len(str(radius))))
    print("ReactionRules length:", len(reaction_rules))

    save_debug_rows(
        debug_rows=debug_rows,
        debug_output=debug_output,
    )

    print()
    print("Saving ReactionRules...")
    reaction_rules.save()

    print()
    print("Testing reload...")
    loaded = ReactionRules.load(
        database_name=database_name,
        ecfp_params=ecfp_params,
    )
    print("Reloaded ReactionRules length:", len(loaded))

    summary = {
        "radius": int(radius),
        "database_name": database_name,
        "input_path": str(input_path),
        "ecfp_params": ecfp_params,
        "template_radius": int(template_radius),
        "rows_read": int(n_rows_read),
        "rows_ok": int(n_rows_ok),
        "rules_after_filters": int(len(reaction_rules)),
        "debug_rows": int(len(debug_rows)),
        "stage_counts": {
            str(k): int(v)
            for k, v in stage_counts.items()
        },
    }

    return summary


# ======================================================================================
# CLI
# ======================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load mapped reaction rules for MetaNetX or USPTO, compute reaction "
            "ECFPs and reaction-center ECFPs for one or many radii, and save "
            "ReactionRules objects."
        )
    )

    parser.add_argument(
        "--data",
        choices=["metanetx", "uspto"],
        default="metanetx",
        help="Dataset to process. Default: uspto.",
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input TSV path. If omitted, uses "
            "data/<data>/processed/<data>_rules.tsv."
        ),
    )

    parser.add_argument(
        "--database-name",
        default=None,
        help=(
            "Name used by ReactionRules.save(). "
            "Default: same as --data."
        ),
    )

    parser.add_argument(
        "--id-col",
        default="id",
        help="Column containing rule IDs.",
    )

    parser.add_argument(
        "--reaction-col",
        default="reaction",
        help="Column containing mapped reaction rules.",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help=(
            "Single Morgan/ECFP radius. Used only if --radii is not provided."
        ),
    )

    parser.add_argument(
        "--radii",
        default="0,1,2,3,4,5",
        help=(
            "Comma-separated list of radii. "
            "Example: --radii 0,1,2,3,4,5"
        ),
    )

    parser.add_argument(
        "--fp-size",
        type=int,
        default=1024,
        help="Folded fingerprint size.",
    )

    parser.add_argument(
        "--unfolded",
        action="store_true",
        help="Use unfolded Morgan fingerprints instead of folded vectors.",
    )

    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom ECFP atom/bond invariants.",
    )

    parser.add_argument(
        "--template-radius",
        type=int,
        default=None,
        help=(
            "Template extraction radius. "
            "Default: 2 * radius for each radius."
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process first N rows, useful for tests.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of TSV rows read at a time.",
    )

    parser.add_argument(
        "--keep-zero-ecfps",
        action="store_true",
        help="Do not remove zero reaction ECFPs.",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not collapse duplicate rules.",
    )

    parser.add_argument(
        "--filter-open-matter-loss",
        action="store_true",
        help="Remove rules with open matter loss.",
    )

    parser.add_argument(
        "--debug-output",
        default=None,
        help=(
            "Debug TSV path. Only used as-is when a single radius is processed. "
            "For multiple radii, debug files are named automatically."
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional summary JSON path. "
            "Default: next to input file, named <stem>_multi_radius_summary.json."
        ),
    )

    return parser


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = build_parser().parse_args()

    input_path = Path(args.input) if args.input is not None else default_rules_path(args.data)

    if args.database_name is None:
        database_name = args.data
    else:
        database_name = args.database_name

    radii = parse_radii(
        radii_value=args.radii,
        fallback_radius=args.radius,
    )

    args.radii_list = radii

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else input_path.with_name(f"{input_path.stem}_multi_radius_summary.json")
    )

    sp_min = make_smarts_params(custom=args.custom)

    print("Creating ReactionRules for multiple radii")
    print("=========================================")
    print("data:", args.data)
    print("input:", input_path)
    print("database_name:", database_name)
    print("radii:", radii)
    print("fp_size:", args.fp_size)
    print("folded:", not args.unfolded)
    print("custom:", args.custom)
    print("max_rows:", args.max_rows)
    print("chunksize:", args.chunksize)
    print("summary_output:", summary_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_summaries = []

    for radius in radii:
        summary = process_one_radius(
            radius=radius,
            args=args,
            input_path=input_path,
            database_name=database_name,
            sp_min=sp_min,
        )

        all_summaries.append(summary)

    summary_output.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": args.data,
                "database_name": database_name,
                "input_path": str(input_path),
                "radii": radii,
                "summaries": all_summaries,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("All radii done")
    print("=" * 100)
    print("summary_output:", summary_output)

    print()
    print(pd.DataFrame(all_summaries))


if __name__ == "__main__":
    main()