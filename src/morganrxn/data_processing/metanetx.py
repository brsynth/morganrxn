"""
Prepare MetaNetX reaction rules from chem_prop.tsv and reac_prop.tsv.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from io import StringIO
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase

from morganrxn.core.paths import METANETX_DIR
from morganrxn.core.molecule_utils import sanitize_smiles
from morganrxn.core.reaction_utils import (
    map_reaction,
    add_missing_mappings_both_sides,
    deduplicate_reaction,
    valid_reaction,
)


# ======================================================================================
# Global RDKit log suppression
# ======================================================================================

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")


# ======================================================================================
# Constants
# ======================================================================================

CHEM_PROP_COLUMNS = [
    "ID",
    "name",
    "reference",
    "formula",
    "charge",
    "mass",
    "InChI",
    "InChIKey",
    "SMILES",
]

REAC_PROP_COLUMNS = [
    "ID",
    "mnx_equation",
    "reference",
    "classifs",
    "is_balanced",
    "is_transport",
]

OUTPUT_COLUMNS = [
    "id",
    "reaction",
    "split_reaction_ID",
    "ec_numbers",
    "substrate",
    "products",
    "sub_excluded",
    "no_struct",
    "reaction_smiles_for_mapping",
    "mapped_full_reaction",
    "n_deduplicated_reactions",
    "mapping_status",
]

EC_PATTERN = re.compile(
    r"\b\d+\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)\b"
)

MNX_TERM_PATTERN = re.compile(
    r"^(?:(?P<coef>[0-9.]+)\s+)?"
    r"(?P<compound>MNXM[0-9]+|BIOMASS)"
    r"(?:@[A-Za-z0-9_]+)?$"
)


# ======================================================================================
# Generic helpers
# ======================================================================================

def read_metanetx_tsv_skip_header_comments(path, names):
    """
    Read a MetaNetX TSV file while skipping only lines that start with '#'.

    Important:
    We do not use pandas comment="#", because SMILES may contain '#'
    for triple bonds, e.g. N#C.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        lines = [
            line
            for line in file
            if not line.startswith("#")
        ]

    return pd.read_csv(
        StringIO("".join(lines)),
        sep="\t",
        header=None,
        names=names,
        index_col=False,
    )


def normalize_smiles_value(value):
    """
    Normalize missing SMILES values.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "-"}:
        return None

    return value


def sanitize_smiles_safe(smiles):
    """
    Safely sanitize a SMILES.

    Returns
    -------
    str or None
    """
    smiles = normalize_smiles_value(smiles)

    if smiles is None:
        return None

    try:
        smiles_san = sanitize_smiles(smiles)
    except Exception:
        return None

    smiles_san = normalize_smiles_value(smiles_san)

    if smiles_san is None:
        return None

    return smiles_san


def ids_to_string(ids):
    """
    Join MNX IDs as:
        MNXM1.MNXM2.MNXM3

    Duplicates are removed while preserving first-seen order.
    """
    out = []
    seen = set()

    for x in ids:
        if x is None or pd.isna(x):
            continue

        x = str(x).strip()

        if x == "" or x in seen:
            continue

        seen.add(x)
        out.append(x)

    if len(out) == 0:
        return ""

    return ".".join(out)


def split_ids(value):
    """
    Split dot-separated MNX IDs.

    Empty / NaN values return an empty list.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    ids = [
        x.strip()
        for x in value.split(".")
        if x.strip()
    ]

    out = []
    seen = set()

    for mnx_id in ids:
        if mnx_id in seen:
            continue

        seen.add(mnx_id)
        out.append(mnx_id)

    return out


def smiles_side(smiles_list):
    """
    Join molecule SMILES as dot-separated side.
    """
    out = []

    for smiles in smiles_list:
        smiles = normalize_smiles_value(smiles)

        if smiles is None:
            continue

        out.append(smiles)

    if len(out) == 0:
        return ""

    return ".".join(out)


def print_counter(counter, title):
    print()
    print(title)
    print("=" * len(title))

    if len(counter) == 0:
        print("No rows.")
    else:
        print(pd.Series(counter).sort_values(ascending=False))


# ======================================================================================
# Load chem_prop
# ======================================================================================

def load_mnx_to_smiles_from_chem_prop(
    chem_prop_path: str | Path,
    max_compounds: int | None = None,
):
    """
    Load chem_prop.tsv and return:
        MNX ID -> sanitized SMILES or None

    This replaces the older source_chem_prop_*.tsv step.
    """
    chem_prop_path = Path(chem_prop_path)

    print("Loading MetaNetX chem_prop.tsv...")
    print("chem_prop path:", chem_prop_path)

    df_chem_prop = read_metanetx_tsv_skip_header_comments(
        chem_prop_path,
        names=CHEM_PROP_COLUMNS,
    )

    if max_compounds is not None:
        df_chem_prop = df_chem_prop.head(max_compounds).copy()

    print("chem_prop shape:", df_chem_prop.shape)
    print(df_chem_prop.head())

    mnx_to_smiles = {}
    n_with_original_smiles = 0
    n_with_sanitized_smiles = 0

    for row in df_chem_prop.itertuples(index=False):
        mnx_id = str(row.ID)

        smiles = normalize_smiles_value(row.SMILES)

        if smiles is not None:
            n_with_original_smiles += 1

        smiles_san = sanitize_smiles_safe(smiles)

        if smiles_san is not None:
            n_with_sanitized_smiles += 1

        mnx_to_smiles[mnx_id] = smiles_san

    print()
    print("chem_prop diagnostics")
    print("=====================")
    print("Compounds:", len(mnx_to_smiles))
    print("With original SMILES:", n_with_original_smiles)
    print("With sanitized SMILES:", n_with_sanitized_smiles)
    print("Without sanitized SMILES:", len(mnx_to_smiles) - n_with_sanitized_smiles)

    return mnx_to_smiles


# ======================================================================================
# Parse reac_prop
# ======================================================================================

def unique_items_by_id(items):
    """
    Keep one parsed item per compound ID, preserving first-seen order.
    """
    out = []
    seen = set()

    for item in items:
        compound_id = str(item["id"])

        if compound_id in seen:
            continue

        seen.add(compound_id)
        out.append(item)

    return out


def parse_mnx_side(side):
    """
    Parse one side of a MetaNetX equation.

    Examples
    --------
        MNXM1@MNXD1
        2 MNXM1@MNXD1
        0.5 MNXM1@MNXD1

    Returns
    -------
    list[dict]
        Each item has:
        - id
        - position
    """
    if pd.isna(side):
        return []

    terms = str(side).split("+")
    parsed = []
    position = 0

    for term in terms:
        term = term.strip()

        if not term:
            continue

        match = MNX_TERM_PATTERN.match(term)

        if match is None:
            continue

        position += 1
        compound_id = match.group("compound")

        parsed.append(
            {
                "id": compound_id,
                "position": position,
            }
        )

    return unique_items_by_id(parsed)


def parse_mnx_equation(mnx_equation):
    """
    Split a MetaNetX reaction equation into left and right sides.
    """
    if pd.isna(mnx_equation):
        return [], []

    mnx_equation = str(mnx_equation)

    if "=" not in mnx_equation:
        return [], []

    left, right = mnx_equation.split("=", maxsplit=1)

    left_side = parse_mnx_side(left)
    right_side = parse_mnx_side(right)

    return left_side, right_side


def extract_ec_numbers_from_values(reference, classifs):
    """
    Extract EC numbers from reac_prop metadata.

    Returns
    -------
    str
        EC numbers joined with ';', or 'NOEC'.
    """
    text_parts = [reference, classifs]

    text = " ".join(
        str(x)
        for x in text_parts
        if pd.notna(x)
    )

    ec_numbers = sorted(set(EC_PATTERN.findall(text)))

    if len(ec_numbers) == 0:
        return "NOEC"

    return ";".join(ec_numbers)


def iter_reac_prop_chunks(
    reac_prop_path: str | Path,
    chunksize: int,
    max_rows: int | None = None,
):
    """
    Yield chunks of MetaNetX reac_prop.tsv.

    We skip only lines starting with '#', not arbitrary '#'.
    For reac_prop this is less critical than chem_prop, but consistent.
    """
    reac_prop_path = Path(reac_prop_path)

    rows = []
    rows_seen = 0
    chunk_id = 0

    with open(reac_prop_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            rows.append(line)

            if len(rows) >= chunksize:
                chunk_id += 1

                chunk = pd.read_csv(
                    StringIO("".join(rows)),
                    sep="\t",
                    header=None,
                    names=REAC_PROP_COLUMNS,
                    index_col=False,
                )

                rows = []

                if max_rows is not None:
                    remaining = max_rows - rows_seen

                    if remaining <= 0:
                        break

                    if len(chunk) > remaining:
                        chunk = chunk.head(remaining).copy()

                rows_seen += len(chunk)

                yield chunk_id, chunk

                if max_rows is not None and rows_seen >= max_rows:
                    break

        if rows and (max_rows is None or rows_seen < max_rows):
            chunk_id += 1

            chunk = pd.read_csv(
                StringIO("".join(rows)),
                sep="\t",
                header=None,
                names=REAC_PROP_COLUMNS,
                index_col=False,
            )

            if max_rows is not None:
                remaining = max_rows - rows_seen

                if remaining > 0 and len(chunk) > remaining:
                    chunk = chunk.head(remaining).copy()

            yield chunk_id, chunk


# ======================================================================================
# Split reactions, no cofactors
# ======================================================================================

def get_no_struct_ids(mnx_to_smiles, *id_lists):
    """
    Return sorted unique IDs that have no usable structure.
    """
    out = []

    for ids in id_lists:
        for mnx_id in ids:
            mnx_id = str(mnx_id)

            smiles = mnx_to_smiles.get(mnx_id)

            if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
                out.append(mnx_id)

    return sorted(set(out))


def iter_split_rows_for_direction(
    mnx_reaction_id: str,
    ec_numbers: str,
    substrate_side: list[dict],
    product_side: list[dict],
    direction_tag: str,
    mnx_to_smiles: dict,
):
    """
    Yield one split row per unique substrate-side species.

    No cofactor filtering:
        every compound on the substrate side can become a selected substrate.

    products:
        all product-side IDs

    sub_excluded:
        all other substrate-side IDs
    """
    substrate_side = unique_items_by_id(substrate_side)
    product_side = unique_items_by_id(product_side)

    product_ids = [x["id"] for x in product_side]

    for substrate_item in substrate_side:
        substrate_id = substrate_item["id"]
        substrate_position = substrate_item["position"]

        sub_excluded = []

        for item in substrate_side:
            current_id = item["id"]

            if current_id == substrate_id:
                continue

            sub_excluded.append(current_id)

        no_struct = get_no_struct_ids(
            mnx_to_smiles,
            [substrate_id],
            product_ids,
            sub_excluded,
        )

        split_reaction_id = (
            f"{mnx_reaction_id}_{direction_tag}_r{substrate_position}"
        )

        yield {
            "split_reaction_ID": split_reaction_id,
            "ec_numbers": ec_numbers,
            "substrate": str(substrate_id),
            "products": ids_to_string(product_ids),
            "sub_excluded": ids_to_string(sub_excluded),
            "no_struct": ids_to_string(no_struct),
        }


def iter_split_rows_for_reac_prop_row(
    reac_prop_row,
    direction: str,
    mnx_to_smiles: dict,
):
    """
    Yield split rows for one reac_prop row.
    """
    mnx_reaction_id = str(reac_prop_row.ID)
    mnx_equation = reac_prop_row.mnx_equation

    ec_numbers = extract_ec_numbers_from_values(
        reference=reac_prop_row.reference,
        classifs=reac_prop_row.classifs,
    )

    left_side, right_side = parse_mnx_equation(mnx_equation)

    if len(left_side) == 0 or len(right_side) == 0:
        return

    if direction in {"L2R", "both"}:
        yield from iter_split_rows_for_direction(
            mnx_reaction_id=mnx_reaction_id,
            ec_numbers=ec_numbers,
            substrate_side=left_side,
            product_side=right_side,
            direction_tag="L2R",
            mnx_to_smiles=mnx_to_smiles,
        )

    if direction in {"R2L", "both"}:
        yield from iter_split_rows_for_direction(
            mnx_reaction_id=mnx_reaction_id,
            ec_numbers=ec_numbers,
            substrate_side=right_side,
            product_side=left_side,
            direction_tag="R2L",
            mnx_to_smiles=mnx_to_smiles,
        )


# ======================================================================================
# Mapping
# ======================================================================================

def get_smiles_for_ids(ids, mnx_to_smiles):
    """
    Convert MNX IDs to sanitized SMILES.

    Returns
    -------
    smiles_list : list[str]
    ids_without_structure : list[str]
    """
    smiles_list = []
    ids_without_structure = []

    for mnx_id in ids:
        mnx_id = str(mnx_id)

        smiles = mnx_to_smiles.get(mnx_id)

        if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
            ids_without_structure.append(mnx_id)
            continue

        smiles_list.append(str(smiles))

    return smiles_list, ids_without_structure


def remove_atom_maps_and_canonicalize(smiles):
    """
    Remove atom mapping numbers and return canonical SMILES.

    Used to select the deduplicated reaction corresponding to the selected
    monosubstrate.
    """
    smiles = normalize_smiles_value(smiles)

    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def select_deduplicated_reaction_for_substrate(
    deduplicated_reactions,
    substrate_smiles,
):
    """
    Select the deduplicated reaction whose left side corresponds to the
    selected monosubstrate.

    Fallback:
        return the first sorted deduplicated reaction.
    """
    substrate_key = remove_atom_maps_and_canonicalize(substrate_smiles)

    if substrate_key is not None:
        for reaction in deduplicated_reactions:
            if ">>" not in reaction:
                continue

            left, _right = reaction.split(">>", maxsplit=1)
            left_key = remove_atom_maps_and_canonicalize(left)

            if left_key == substrate_key:
                return reaction

    if len(deduplicated_reactions) == 0:
        return None

    return sorted(deduplicated_reactions)[0]


def build_mapping_input_from_split_row(split_row, mnx_to_smiles):
    """
    Reconstruct the full reaction for mapping from a split row.

    Reaction used for mapping:
        selected_substrate + sub_excluded >> products

    IDs without structure are dropped from the mapping reaction, but tracked.
    """
    substrate_id = str(split_row["substrate"])
    product_ids = split_ids(split_row["products"])
    sub_excluded_ids = split_ids(split_row["sub_excluded"])

    substrate_smiles_list, no_struct_substrate = get_smiles_for_ids(
        [substrate_id],
        mnx_to_smiles=mnx_to_smiles,
    )

    sub_excluded_smiles, no_struct_sub_excluded = get_smiles_for_ids(
        sub_excluded_ids,
        mnx_to_smiles=mnx_to_smiles,
    )

    product_smiles, no_struct_products = get_smiles_for_ids(
        product_ids,
        mnx_to_smiles=mnx_to_smiles,
    )

    if len(substrate_smiles_list) == 0:
        return {
            "reaction_smiles": None,
            "substrate_smiles": None,
            "no_struct_substrate": no_struct_substrate,
            "no_struct_sub_excluded": no_struct_sub_excluded,
            "no_struct_products": no_struct_products,
            "status": "missing_substrate_structure",
        }

    if len(product_smiles) == 0:
        return {
            "reaction_smiles": None,
            "substrate_smiles": substrate_smiles_list[0],
            "no_struct_substrate": no_struct_substrate,
            "no_struct_sub_excluded": no_struct_sub_excluded,
            "no_struct_products": no_struct_products,
            "status": "missing_all_product_structures",
        }

    left_smiles = smiles_side(
        substrate_smiles_list + sub_excluded_smiles
    )

    right_smiles = smiles_side(product_smiles)

    if left_smiles == "" or right_smiles == "":
        return {
            "reaction_smiles": None,
            "substrate_smiles": substrate_smiles_list[0],
            "no_struct_substrate": no_struct_substrate,
            "no_struct_sub_excluded": no_struct_sub_excluded,
            "no_struct_products": no_struct_products,
            "status": "empty_side_after_structure_filter",
        }

    reaction_smiles = f"{left_smiles}>>{right_smiles}"

    return {
        "reaction_smiles": reaction_smiles,
        "substrate_smiles": substrate_smiles_list[0],
        "no_struct_substrate": no_struct_substrate,
        "no_struct_sub_excluded": no_struct_sub_excluded,
        "no_struct_products": no_struct_products,
        "status": "ok",
    }


def map_then_deduplicate_for_split_row(split_row, mnx_to_smiles, verbose=False):
    """
    Map full reconstructed reaction, deduplicate, and keep the reaction
    corresponding to the selected monosubstrate.
    """
    prepared = build_mapping_input_from_split_row(
        split_row=split_row,
        mnx_to_smiles=mnx_to_smiles,
    )

    if prepared["status"] != "ok":
        return {
            **prepared,
            "mapped_full_reaction": None,
            "mapped_reaction": None,
            "n_deduplicated_reactions": 0,
            "mapping_status": prepared["status"],
        }

    reaction_smiles = prepared["reaction_smiles"]
    substrate_smiles = prepared["substrate_smiles"]

    if not valid_reaction(reaction_smiles):
        return {
            **prepared,
            "mapped_full_reaction": None,
            "mapped_reaction": None,
            "n_deduplicated_reactions": 0,
            "mapping_status": "invalid_reaction_before_mapping",
        }

    try:
        mapped_full_reaction, _ = map_reaction(reaction_smiles)
        mapped_full_reaction = add_missing_mappings_both_sides(
            mapped_full_reaction
        )
    except Exception as error:
        if verbose:
            print("MAPPING FAILED")
            print("reaction_smiles:", reaction_smiles)
            print("error:", repr(error))

        return {
            **prepared,
            "mapped_full_reaction": None,
            "mapped_reaction": None,
            "n_deduplicated_reactions": 0,
            "mapping_status": f"mapping_failed:{type(error).__name__}",
        }

    try:
        deduplicated_reactions = deduplicate_reaction(mapped_full_reaction)
    except Exception as error:
        if verbose:
            print("DEDUPLICATION FAILED")
            print("mapped_full_reaction:", mapped_full_reaction)
            print("error:", repr(error))

        return {
            **prepared,
            "mapped_full_reaction": mapped_full_reaction,
            "mapped_reaction": None,
            "n_deduplicated_reactions": 0,
            "mapping_status": f"deduplication_failed:{type(error).__name__}",
        }

    if len(deduplicated_reactions) == 0:
        return {
            **prepared,
            "mapped_full_reaction": mapped_full_reaction,
            "mapped_reaction": None,
            "n_deduplicated_reactions": 0,
            "mapping_status": "no_deduplicated_reaction",
        }

    selected_reaction = select_deduplicated_reaction_for_substrate(
        deduplicated_reactions=deduplicated_reactions,
        substrate_smiles=substrate_smiles,
    )

    if selected_reaction is None or ">>" not in selected_reaction:
        return {
            **prepared,
            "mapped_full_reaction": mapped_full_reaction,
            "mapped_reaction": None,
            "n_deduplicated_reactions": len(deduplicated_reactions),
            "mapping_status": "no_selected_monosubstrate_reaction",
        }

    return {
        **prepared,
        "mapped_full_reaction": mapped_full_reaction,
        "mapped_reaction": selected_reaction,
        "n_deduplicated_reactions": len(deduplicated_reactions),
        "mapping_status": "ok",
    }


def build_output_row(split_row, mapping_result):
    """
    Build output row.

    The two important columns for later ReactionRules creation are:
        id
        reaction
    """
    return {
        "id": split_row["split_reaction_ID"],
        "reaction": mapping_result["mapped_reaction"],
        "split_reaction_ID": split_row["split_reaction_ID"],
        "ec_numbers": split_row["ec_numbers"],
        "substrate": split_row["substrate"],
        "products": split_row["products"],
        "sub_excluded": split_row["sub_excluded"],
        "no_struct": split_row["no_struct"],
        "reaction_smiles_for_mapping": mapping_result["reaction_smiles"],
        "mapped_full_reaction": mapping_result["mapped_full_reaction"],
        "n_deduplicated_reactions": mapping_result["n_deduplicated_reactions"],
        "mapping_status": mapping_result["mapping_status"],
    }


# ======================================================================================
# Worker
# ======================================================================================

def process_metanetx_worker(
    chem_prop_path: str | Path,
    reac_prop_path: str | Path,
    output_tsv: str | Path,
    summary_json: str | Path,
    direction: str = "both",
    max_rows: int | None = None,
    max_compounds: int | None = None,
    chunksize: int = 10_000,
    keep_failed_rows: bool = False,
    verbose: bool = False,
):
    """
    Actual processing function.

    Intended to run in a silent child process, as in uspto.py.
    """
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")

    chem_prop_path = Path(chem_prop_path)
    reac_prop_path = Path(reac_prop_path)
    output_tsv = Path(output_tsv)
    summary_json = Path(summary_json)

    mnx_to_smiles = load_mnx_to_smiles_from_chem_prop(
        chem_prop_path=chem_prop_path,
        max_compounds=max_compounds,
    )

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    n_reac_prop_read = 0
    n_split_rows = 0
    n_output_rows = 0

    direction_counts = Counter()
    mapping_status_counts_all = Counter()
    mapping_status_counts_output = Counter()

    n_rows_with_no_struct = 0
    n_multi_deduplicated = 0

    examples = []

    with open(output_tsv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for chunk_index, chunk in iter_reac_prop_chunks(
            reac_prop_path=reac_prop_path,
            chunksize=chunksize,
            max_rows=max_rows,
        ):
            if chunk_index == 1:
                print()
                print("First reac_prop chunk:")
                print(chunk.head())

            n_reac_prop_read += len(chunk)

            for reac_prop_row in chunk.itertuples(index=False):
                for split_row in iter_split_rows_for_reac_prop_row(
                    reac_prop_row=reac_prop_row,
                    direction=direction,
                    mnx_to_smiles=mnx_to_smiles,
                ):
                    n_split_rows += 1

                    split_reaction_id = split_row["split_reaction_ID"]

                    if "_L2R_" in split_reaction_id:
                        direction_counts["L2R"] += 1
                    elif "_R2L_" in split_reaction_id:
                        direction_counts["R2L"] += 1
                    else:
                        direction_counts["unknown"] += 1

                    if split_row["no_struct"] != "":
                        n_rows_with_no_struct += 1

                    mapping_result = map_then_deduplicate_for_split_row(
                        split_row=split_row,
                        mnx_to_smiles=mnx_to_smiles,
                        verbose=verbose,
                    )

                    mapping_status = mapping_result["mapping_status"]
                    mapping_status_counts_all[mapping_status] += 1

                    if mapping_result["n_deduplicated_reactions"] > 1:
                        n_multi_deduplicated += 1

                    if mapping_status != "ok" and not keep_failed_rows:
                        continue

                    output_row = build_output_row(
                        split_row=split_row,
                        mapping_result=mapping_result,
                    )

                    writer.writerow(output_row)

                    n_output_rows += 1
                    mapping_status_counts_output[mapping_status] += 1

                    if len(examples) < 20:
                        examples.append(output_row.copy())

            print(
                f"Processed chunk {chunk_index}: "
                f"{n_reac_prop_read} MetaNetX reactions read, "
                f"{n_split_rows} split rows generated, "
                f"{n_output_rows} output rows written."
            )

    summary = {
        "chem_prop_path": str(chem_prop_path),
        "reac_prop_path": str(reac_prop_path),
        "output": str(output_tsv),
        "direction": direction,
        "max_rows": max_rows,
        "max_compounds": max_compounds,
        "chunksize": chunksize,
        "keep_failed_rows": keep_failed_rows,
        "input_reac_prop_reactions": int(n_reac_prop_read),
        "split_rows": int(n_split_rows),
        "output_rows": int(n_output_rows),
        "rows_with_no_struct": int(n_rows_with_no_struct),
        "rows_with_more_than_one_deduplicated_reaction": int(n_multi_deduplicated),
        "direction_counts": {
            str(k): int(v)
            for k, v in direction_counts.items()
        },
        "mapping_status_counts_all": {
            str(k): int(v)
            for k, v in mapping_status_counts_all.items()
        },
        "mapping_status_counts_output": {
            str(k): int(v)
            for k, v in mapping_status_counts_output.items()
        },
        "examples": examples,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Diagnostics")
    print("===========")
    print("MetaNetX reactions read:", n_reac_prop_read)
    print("Split rows generated:", n_split_rows)
    print("Output rows written:", n_output_rows)
    print("Rows with no_struct:", n_rows_with_no_struct)
    print("Rows with >1 deduplicated reaction:", n_multi_deduplicated)

    print_counter(direction_counts, "Direction counts")
    print_counter(mapping_status_counts_all, "Mapping status counts, all rows")
    print_counter(mapping_status_counts_output, "Mapping status counts, output rows")

    print()
    print("Output:", output_tsv)
    print("Summary:", summary_json)


# ======================================================================================
# Silent parent launcher
# ======================================================================================

def run_silent_child(args):
    """
    Relaunch this script as a worker process.

    The worker stdout/stderr are redirected to DEVNULL unless --show-rdkit-logs
    is used. This avoids RDKit / mapper floods in the console.
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
        "--chem-prop",
        str(args.chem_prop),
        "--reac-prop",
        str(args.reac_prop),
        "--output",
        str(args.output),
        "--direction",
        str(args.direction),
        "--chunksize",
        str(args.chunksize),
        "--summary-json",
        str(summary_json),
    ]

    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])

    if args.max_compounds is not None:
        cmd.extend(["--max-compounds", str(args.max_compounds)])

    if args.keep_failed_rows:
        cmd.append("--keep-failed-rows")

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

    final_summary_json = Path(args.output).with_suffix(".summary.json")
    final_summary_json.parent.mkdir(parents=True, exist_ok=True)

    with open(final_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        summary_json.unlink()
    except Exception:
        pass

    print(f"MetaNetX reactions read:          {summary['input_reac_prop_reactions']}")
    print(f"Split rows generated:            {summary['split_rows']}")
    print(f"Output mapped rules:             {summary['output_rows']}")
    print(f"Rows with no_struct:             {summary['rows_with_no_struct']}")
    print(f"Rows with >1 deduplicated rxns:  {summary['rows_with_more_than_one_deduplicated_reaction']}")
    print(f"Direction:                       {summary['direction']}")
    print(f"Output:                          {summary['output']}")
    print(f"Summary:                         {final_summary_json}")

    print()
    print("Mapping status counts:")
    print(pd.Series(summary["mapping_status_counts_all"]).sort_values(ascending=False))


# ======================================================================================
# CLI
# ======================================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load MetaNetX chem_prop.tsv and reac_prop.tsv, reconstruct "
            "monosubstrate mapped reaction rules, and save a TSV with "
            "columns id and reaction."
        )
    )

    parser.add_argument(
        "--chem-prop",
        default=METANETX_DIR / "chem_prop.tsv",
        help="Input MetaNetX chem_prop.tsv.",
    )

    parser.add_argument(
        "--reac-prop",
        default=METANETX_DIR / "reac_prop.tsv",
        help="Input MetaNetX reac_prop.tsv.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=METANETX_DIR / "processed" / "metanetx_rules.tsv",
        help="Output TSV file.",
    )

    parser.add_argument(
        "--direction",
        choices=["L2R", "R2L", "both"],
        default="both",
        help="Reaction direction(s) to generate.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of reac_prop rows to process.",
    )

    parser.add_argument(
        "--max-compounds",
        type=int,
        default=None,
        help="Optional maximum number of chem_prop rows to load, useful for tests.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Number of reac_prop rows read at once.",
    )

    parser.add_argument(
        "--keep-failed-rows",
        action="store_true",
        help="Keep failed mapping rows in output with empty reaction.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed mapping errors. Suppressed unless --show-rdkit-logs is used.",
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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.worker:
        if args.summary_json is None:
            raise ValueError("--summary-json is required in worker mode.")

        process_metanetx_worker(
            chem_prop_path=args.chem_prop,
            reac_prop_path=args.reac_prop,
            output_tsv=args.output,
            summary_json=args.summary_json,
            direction=args.direction,
            max_rows=args.max_rows,
            max_compounds=args.max_compounds,
            chunksize=args.chunksize,
            keep_failed_rows=args.keep_failed_rows,
            verbose=args.verbose,
        )
        return

    if args.show_rdkit_logs:
        summary_json = Path(args.output).with_suffix(".summary.json")

        process_metanetx_worker(
            chem_prop_path=args.chem_prop,
            reac_prop_path=args.reac_prop,
            output_tsv=args.output,
            summary_json=summary_json,
            direction=args.direction,
            max_rows=args.max_rows,
            max_compounds=args.max_compounds,
            chunksize=args.chunksize,
            keep_failed_rows=args.keep_failed_rows,
            verbose=args.verbose,
        )
        return

    run_silent_child(args)


if __name__ == "__main__":
    main()