"""
Stage 1 of the MetaNetX pipeline: sanitize reaction SMILES.

Reads MetaNetX ``chem_prop.tsv`` and ``reac_prop.tsv`` and, for every reaction
equation, builds the full sanitized reaction SMILES in both directions:

    L2R : substrates >> products
    R2L : products   >> substrates

This step does *not* map and does *not* deduplicate. It only sanitizes each
side (canonical SMILES, atom maps and stereo removed) and drops compounds
without a usable structure. The output is a TSV with columns:

    id    reaction    ec_numbers    mnxr_id    direction

``id`` embeds the MetaNetX reaction id (e.g. ``MNXR12345_L2R``) so that later
stages, and in particular ``metanetx_ec_prediction.py``, can still recover the
MNXR id (and hence the EC annotation) from a rule id.

The mapping is applied afterwards by ``map_reactions.py`` (stage 2), and the
deduplication / open-matter-loss filtering happens in
``create_reactionrules_from_mapped_rules.py`` (stage 3).
"""

import argparse
import csv
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from rdkit import RDLogger
from rdkit import rdBase

from morganrxn.core.paths import METANETX_DIR
from morganrxn.core.molecule_utils import sanitize_smiles


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
    "ec_numbers",
    "mnxr_id",
    "direction",
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

    mnx_to_smiles = {}
    n_with_sanitized_smiles = 0

    for row in df_chem_prop.itertuples(index=False):
        mnx_id = str(row.ID)
        smiles_san = sanitize_smiles_safe(row.SMILES)

        if smiles_san is not None:
            n_with_sanitized_smiles += 1

        mnx_to_smiles[mnx_id] = smiles_san

    print("Compounds:", len(mnx_to_smiles))
    print("With sanitized SMILES:", n_with_sanitized_smiles)

    return mnx_to_smiles


# ======================================================================================
# Parse reac_prop
# ======================================================================================

def parse_mnx_side(side):
    """
    Parse one side of a MetaNetX equation into an ordered list of unique
    compound IDs.

    Examples
    --------
        MNXM1@MNXD1
        2 MNXM1@MNXD1
        0.5 MNXM1@MNXD1
    """
    if pd.isna(side):
        return []

    ids = []
    seen = set()

    for term in str(side).split("+"):
        term = term.strip()

        if not term:
            continue

        match = MNX_TERM_PATTERN.match(term)

        if match is None:
            continue

        compound_id = match.group("compound")

        if compound_id in seen:
            continue

        seen.add(compound_id)
        ids.append(compound_id)

    return ids


def parse_mnx_equation(mnx_equation):
    """
    Split a MetaNetX reaction equation into left and right lists of IDs.
    """
    if pd.isna(mnx_equation):
        return [], []

    mnx_equation = str(mnx_equation)

    if "=" not in mnx_equation:
        return [], []

    left, right = mnx_equation.split("=", maxsplit=1)

    return parse_mnx_side(left), parse_mnx_side(right)


def extract_ec_numbers_from_values(reference, classifs):
    """
    Extract EC numbers from reac_prop metadata.

    Returns
    -------
    str
        EC numbers joined with ';', or 'NOEC'.
    """
    text = " ".join(
        str(x)
        for x in (reference, classifs)
        if pd.notna(x)
    )

    ec_numbers = sorted(set(EC_PATTERN.findall(text)))

    if len(ec_numbers) == 0:
        return "NOEC"

    return ";".join(ec_numbers)


# ======================================================================================
# Build sanitized reaction SMILES
# ======================================================================================

def side_smiles(ids, mnx_to_smiles):
    """
    Convert a list of MNX IDs to a sanitized dot-joined SMILES side.

    Compounds without a usable structure are dropped. Returns an empty string
    if no compound on the side has a structure.
    """
    components = []

    for mnx_id in ids:
        smiles = mnx_to_smiles.get(str(mnx_id))

        if smiles is None or str(smiles).strip() == "":
            continue

        components.append(str(smiles))

    if not components:
        return ""

    return sanitize_smiles(".".join(components))


def iter_reaction_rows(reac_prop_row, direction, mnx_to_smiles):
    """
    Yield sanitized reaction rows for one reac_prop row, one per requested
    direction. Reactions with an empty side (after dropping structureless
    compounds) are skipped.
    """
    mnxr_id = str(reac_prop_row.ID)

    ec_numbers = extract_ec_numbers_from_values(
        reference=reac_prop_row.reference,
        classifs=reac_prop_row.classifs,
    )

    left_ids, right_ids = parse_mnx_equation(reac_prop_row.mnx_equation)

    if not left_ids or not right_ids:
        return

    left_smiles = side_smiles(left_ids, mnx_to_smiles)
    right_smiles = side_smiles(right_ids, mnx_to_smiles)

    if left_smiles == "" or right_smiles == "":
        return

    if direction in {"L2R", "both"}:
        yield {
            "id": f"{mnxr_id}_L2R",
            "reaction": f"{left_smiles}>>{right_smiles}",
            "ec_numbers": ec_numbers,
            "mnxr_id": mnxr_id,
            "direction": "L2R",
        }

    if direction in {"R2L", "both"}:
        yield {
            "id": f"{mnxr_id}_R2L",
            "reaction": f"{right_smiles}>>{left_smiles}",
            "ec_numbers": ec_numbers,
            "mnxr_id": mnxr_id,
            "direction": "R2L",
        }


# ======================================================================================
# Processing
# ======================================================================================

def process_metanetx(
    chem_prop_path: str | Path,
    reac_prop_path: str | Path,
    output_tsv: str | Path,
    direction: str = "both",
    max_rows: int | None = None,
    max_compounds: int | None = None,
):
    """
    Build the sanitized MetaNetX reactions TSV (id, reaction, ...).
    """
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")

    chem_prop_path = Path(chem_prop_path)
    reac_prop_path = Path(reac_prop_path)
    output_tsv = Path(output_tsv)

    mnx_to_smiles = load_mnx_to_smiles_from_chem_prop(
        chem_prop_path=chem_prop_path,
        max_compounds=max_compounds,
    )

    df_reac_prop = read_metanetx_tsv_skip_header_comments(
        reac_prop_path,
        names=REAC_PROP_COLUMNS,
    )

    if max_rows is not None:
        df_reac_prop = df_reac_prop.head(max_rows).copy()

    print()
    print("reac_prop shape:", df_reac_prop.shape)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    n_reactions_read = 0
    n_output_rows = 0
    direction_counts = {"L2R": 0, "R2L": 0}

    with open(output_tsv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for reac_prop_row in df_reac_prop.itertuples(index=False):
            n_reactions_read += 1

            for output_row in iter_reaction_rows(
                reac_prop_row=reac_prop_row,
                direction=direction,
                mnx_to_smiles=mnx_to_smiles,
            ):
                writer.writerow(output_row)
                n_output_rows += 1
                direction_counts[output_row["direction"]] += 1

    print()
    print("Diagnostics")
    print("===========")
    print("MetaNetX reactions read:", n_reactions_read)
    print("Sanitized reactions written:", n_output_rows)
    print("L2R:", direction_counts["L2R"])
    print("R2L:", direction_counts["R2L"])
    print("Output:", output_tsv)


# ======================================================================================
# CLI
# ======================================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load MetaNetX chem_prop.tsv and reac_prop.tsv, build sanitized "
            "reaction SMILES (L2R and R2L, no mapping, no deduplication), and "
            "save a TSV with columns id, reaction, ec_numbers, mnxr_id, "
            "direction."
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
        default=METANETX_DIR / "processed" / "metanetx_reactions.tsv",
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

    return parser


def main():
    args = build_parser().parse_args()

    process_metanetx(
        chem_prop_path=args.chem_prop,
        reac_prop_path=args.reac_prop,
        output_tsv=args.output,
        direction=args.direction,
        max_rows=args.max_rows,
        max_compounds=args.max_compounds,
    )


if __name__ == "__main__":
    main()
