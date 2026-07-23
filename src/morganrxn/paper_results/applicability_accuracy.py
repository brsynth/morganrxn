#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired one-step applicability/accuracy benchmark for morganrxn.

Evaluates how well the ECFP-based one-step prediction matches the actual
template application for each (target molecule, reaction rule) pair where
the rule is ECFP-applicable to the target.

The evaluated criterion is:

    For each target molecule S and each ECFP-applicable rule R,
    predicted_ecfp = ECFP(S) + reaction_ecfp(R).

The case is correct if applying the corresponding template to S produces at
least one product P such that ECFP(P) == predicted_ecfp.

By default the benchmark is run for two applicability criteria so their numbers
can be compared side by side (see ``--applicability-modes``):

    - ``reaction_center``: a rule is applicable to S when S contains the rule's
      reaction-centre ECFP. This is the intended morganrxn criterion.
    - ``reaction``: a rule is applicable to S when S contains the (removed bits
      of the) reaction ECFP itself. This coarser baseline quantifies what the
      reaction-centre ECFP buys us — it typically applies to more (target, rule)
      pairs but at a lower accuracy.

Running the script with no arguments reproduces the full benchmark:
    python applicability_accuracy.py

is equivalent to:
    python applicability_accuracy.py \\
        --radii 0,1,2,3,4,5 \\
        --n-samples 1000 \\
        --applicability-modes reaction_center,reaction \\
        --benchmark-dataset metanetx=metanetx \\
        --benchmark-dataset uspto=uspto

This is meant for a cluster job. For a quick local sanity check:
    python applicability_accuracy.py --n-samples 100 --radii 0,2
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

from morganrxn.core.cli_utils import make_ecfp_params, parse_radii, timer
from morganrxn.core.molecule_utils import (
    get_mol_ecfp,
    sanitize_list_of_smiles,
)
from morganrxn.core.paths import RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules
from morganrxn.core.reaction_utils import apply_reaction, one_step


DEFAULT_RADII = [0, 1, 2, 3, 4, 5]
DEFAULT_FP_SIZE = 1024
DEFAULT_N_SAMPLES = 20
DEFAULT_MIN_HEAVY_ATOMS = 5
DEFAULT_MIN_SMI_SUB_ATOMS = 5
DEFAULT_MAX_MOL_WT = 1000.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUT_XLSX = (
    RESULTS_DIR / "one_step_accuracy" / "applicability_accuracy_morganrxn_formats.xlsx"
)
DEFAULT_BENCHMARK_DATASETS = {
    "metanetx": ["metanetx"],
    "uspto": ["uspto"],
}
DEFAULT_PAIRED_RULES = {
    "metanetx": ["metanetx"],
    "uspto": ["uspto"],
}

DEBUG_MAX_PRODUCTS_PER_ROW = 25


# ======================================================================================
# Small utilities
# ======================================================================================


def parse_dataset_specs(
    specs: Optional[Sequence[str]], default: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Parse repeated specs of the form ``dataset_name=database_a,database_b``."""
    if not specs:
        return {k: list(v) for k, v in default.items()}

    parsed: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid dataset spec {spec!r}. Expected dataset=database_a,database_b."
            )
        dataset_name, databases_raw = spec.split("=", 1)
        dataset_name = dataset_name.strip()
        databases = [x.strip() for x in databases_raw.split(",") if x.strip()]
        if not dataset_name or not databases:
            raise ValueError(
                f"Invalid dataset spec {spec!r}: empty dataset or database list."
            )
        parsed[dataset_name] = databases
    return parsed


def valid_smiles(smi: str) -> bool:
    if smi is None:
        return False
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def mol_from_smiles(smi: str):
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def ecfp_to_key(ecfp) -> Tuple[int, ...]:
    return tuple(np.asarray(ecfp, dtype=np.int32).tolist())


def safe_list_get(values: Any, idx: int, default: Any = "") -> Any:
    try:
        return values[idx]
    except Exception:
        return default


def ecfp_nonzero_bits(ecfp) -> str:
    arr = np.asarray(ecfp)
    return ";".join(map(str, np.flatnonzero(arr).tolist()))


# ======================================================================================
# Benchmark creation
# ======================================================================================


def collect_unique_smi_sub(
    database_names: Iterable[str],
    ecfp_params: Dict[str, Any],
) -> Set[str]:
    smiles_unique: Set[str] = set()
    for database_name in database_names:
        print("=" * 80)
        print(f"Loading ReactionRules: {database_name}")
        rr = ReactionRules.load(database_name=database_name, ecfp_params=ecfp_params)
        rr.filter_by_smi_sub_atoms(min_atoms=5)
        before = len(smiles_unique)
        for smi in rr.smi_sub:
            if smi:
                smiles_unique.add(smi)
        print(f"Loaded smi_sub entries: {len(list(rr.smi_sub))}")
        print(f"Added unique SMILES: {len(smiles_unique) - before}")
    print(f"Total unique sanitized SMILES: {len(smiles_unique)}")
    return smiles_unique


def filter_smiles_dataframe(
    smiles: Iterable[str],
    dataset_name: str,
    min_heavy_atoms: int,
    max_mol_wt: float,
) -> pd.DataFrame:
    rows = []
    n_total = n_invalid = n_too_small = n_too_heavy = 0

    for smi in smiles:
        n_total += 1
        mol = mol_from_smiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        heavy_atoms = mol.GetNumHeavyAtoms()
        mol_wt = Descriptors.MolWt(mol)
        if heavy_atoms < min_heavy_atoms:
            n_too_small += 1
            continue
        if mol_wt > max_mol_wt:
            n_too_heavy += 1
            continue
        rows.append({"dataset": dataset_name, "smiles": smi, "heavy_atoms": heavy_atoms, "mol_wt": mol_wt})

    df = pd.DataFrame(rows)
    print("-" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Total unique sanitized SMILES: {n_total}")
    print(f"Invalid SMILES removed: {n_invalid}")
    print(f"Removed with < {min_heavy_atoms} heavy atoms: {n_too_small}")
    print(f"Removed with mol_wt > {max_mol_wt}: {n_too_heavy}")
    print(f"Remaining after filters: {len(df)}")
    return df


def create_benchmark_sets_in_memory(
    benchmark_datasets: Dict[str, List[str]],
    ecfp_params: Dict[str, Any],
    n_samples: int,
    random_seed: int,
    min_heavy_atoms: int,
    max_mol_wt: float,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    random.seed(random_seed)
    print("=" * 80)
    print("Creating paired benchmark molecules in memory")
    print(f"Requested sample size per dataset: {n_samples}")
    print(f"Random seed: {random_seed}")

    benchmark_smiles: Dict[str, List[str]] = {}
    summary_rows = []

    for dataset_name, database_names in benchmark_datasets.items():
        print("=" * 80)
        print(f"Creating benchmark dataset: {dataset_name}")
        print(f"Source ReactionRules database(s): {', '.join(database_names)}")

        smiles_unique = collect_unique_smi_sub(
            database_names=database_names,
            ecfp_params=ecfp_params,
        )
        df_filtered = filter_smiles_dataframe(
            smiles=smiles_unique,
            dataset_name=dataset_name,
            min_heavy_atoms=min_heavy_atoms,
            max_mol_wt=max_mol_wt,
        )

        if len(df_filtered) <= n_samples:
            print(f"Only {len(df_filtered)} molecules available; keeping all.")
            df_sampled = df_filtered.reset_index(drop=True)
        else:
            df_sampled = df_filtered.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
            print(f"Randomly sampled molecules: {len(df_sampled)}")

        benchmark_smiles[dataset_name] = df_sampled["smiles"].dropna().astype(str).tolist()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "source_databases": "|".join(database_names),
                "n_unique_sanitized_before_filter": len(smiles_unique),
                "n_after_filter": len(df_filtered),
                "n_sampled": len(df_sampled),
                "min_heavy_atoms": min_heavy_atoms,
                "max_mol_wt": max_mol_wt,
                "random_seed": random_seed,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("=" * 80)
    print("Benchmark creation summary")
    print(summary_df.to_string(index=False))
    return benchmark_smiles, summary_df


# ======================================================================================
# Accuracy computation
# ======================================================================================


def export_debug_tables(
    debug_dir: Path,
    benchmark_name: str,
    database_name: str,
    ecfp_params: Dict[str, Any],
    failed_cases: List[Dict[str, Any]],
    error_cases: List[Dict[str, Any]],
    applicability_mode: str = "reaction_center",
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(
        debug_dir
        / f"{benchmark_name}__{database_name}__{applicability_mode}"
        f"__r{ecfp_params['radius']}__fp{ecfp_params['fpSize']}"
    )

    pd.DataFrame(failed_cases).to_csv(f"{prefix}__failed_cases.tsv", sep="\t", index=False)
    pd.DataFrame(error_cases).to_csv(f"{prefix}__error_cases.tsv", sep="\t", index=False)

    failed_df = pd.DataFrame(failed_cases)
    if len(failed_df) > 0:
        (
            failed_df.groupby(
                ["rule_idx", "reaction_id", "reaction_monocomp_id", "rule_smi_sub", "template_reaction"],
                dropna=False,
            )
            .agg(
                n_failures=("rule_idx", "size"),
                n_unique_targets=("target_smiles", "nunique"),
                example_target=("target_smiles", "first"),
                example_products=("sanitized_products", "first"),
                example_failure_reason=("failure_reason", "first"),
            )
            .reset_index()
            .sort_values(["n_failures", "n_unique_targets"], ascending=False)
            .to_csv(f"{prefix}__bad_reactions.tsv", sep="\t", index=False)
        )
        (
            failed_df.groupby(["target_idx", "target_smiles"], dropna=False)
            .agg(
                n_failures=("target_smiles", "size"),
                n_unique_failed_rules=("rule_idx", "nunique"),
                example_rule_idx=("rule_idx", "first"),
                example_reaction_id=("reaction_id", "first"),
                example_template=("template_reaction", "first"),
                example_failure_reason=("failure_reason", "first"),
            )
            .reset_index()
            .sort_values(["n_failures", "n_unique_failed_rules"], ascending=False)
            .to_csv(f"{prefix}__target_failures.tsv", sep="\t", index=False)
        )
    else:
        pd.DataFrame().to_csv(f"{prefix}__bad_reactions.tsv", sep="\t", index=False)
        pd.DataFrame().to_csv(f"{prefix}__target_failures.tsv", sep="\t", index=False)

    print(f"[debug] saved debug tables to: {debug_dir}")


def compute_ecfp_applies_accuracy(
    benchmark_name: str,
    database_name: str,
    smi_targets: Iterable[str],
    ecfp_params: Dict[str, Any],
    min_smi_sub_atoms: int,
    applicability_mode: str = "reaction_center",
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute the paired applicability/accuracy metrics for one configuration.

    ``applicability_mode`` selects the fingerprint used to decide whether a rule
    is applicable to a target (the child ECFP prediction is always the reaction
    ECFP added to the target):

    - ``"reaction_center"`` (default): a rule applies when the target contains the
      reaction-centre ECFP of the rule. This is the intended morganrxn criterion.
    - ``"reaction"``: a rule applies when the target contains the (negative bits
      of the) reaction ECFP itself. This coarser criterion is provided as a
      baseline to quantify what the reaction-centre ECFP buys us.
    """
    if applicability_mode not in ("reaction_center", "reaction"):
        raise ValueError(
            f"Invalid applicability_mode {applicability_mode!r}; "
            "expected 'reaction_center' or 'reaction'."
        )

    t0 = time.perf_counter()

    print("=" * 80)
    print(
        f"Loading ReactionRules: {database_name} | ecfp_params: {ecfp_params} | "
        f"applicability_mode: {applicability_mode}"
    )
    reaction_rules = ReactionRules.load(database_name=database_name, ecfp_params=ecfp_params)

    reaction_rules.filter_by_smi_sub_atoms(min_atoms=min_smi_sub_atoms, verbose=True)

    ecfp_reaction_np = np.asarray(reaction_rules.ecfp_reaction, dtype=np.int32)
    ecfp_reaction_center_np = np.asarray(reaction_rules.ecfp_reaction_center, dtype=np.int32)

    # The applicability filter uses either the reaction-centre ECFP (the intended
    # criterion) or the reaction ECFP itself (baseline). Either way the predicted
    # child ECFP is target + reaction ECFP, so only the mask differs.
    applicability_ecfp_np = (
        ecfp_reaction_center_np if applicability_mode == "reaction_center" else ecfp_reaction_np
    )

    n_targets_total = n_targets_valid = n_targets_with_ecfp_apply = 0
    n_ecfp_applies = n_correct = 0
    n_invalid_targets = n_target_ecfp_errors = n_one_step_errors = 0
    n_apply_errors = n_product_ecfp_errors = 0

    unique_template_smiles: Set[str] = set()
    unique_predicted_ecfps: Set[Tuple[int, ...]] = set()
    failed_cases: List[Dict[str, Any]] = []
    error_cases: List[Dict[str, Any]] = []

    for target_idx, smi_sub_raw in enumerate(smi_targets):
        n_targets_total += 1

        smi_sub = smi_sub_raw

        if not smi_sub or not valid_smiles(smi_sub):
            n_invalid_targets += 1
            continue

        n_targets_valid += 1

        try:
            smi_ecfp = np.asarray(get_mol_ecfp(smi_sub, ecfp_params), dtype=np.int32)
        except Exception as exc:
            n_target_ecfp_errors += 1
            if debug:
                error_cases.append({
                    "stage": "target_ecfp", "benchmark_name": benchmark_name,
                    "database_name": database_name, "radius": ecfp_params["radius"],
                    "target_idx": target_idx, "target_smiles": smi_sub, "error": repr(exc),
                })
            continue

        try:
            smi_ecfp_childs, rxn_idxs_unique = one_step(
                smi_ecfp, ecfp_reaction_np, applicability_ecfp_np
            )
        except Exception as exc:
            n_one_step_errors += 1
            if debug:
                error_cases.append({
                    "stage": "one_step", "benchmark_name": benchmark_name,
                    "database_name": database_name, "radius": ecfp_params["radius"],
                    "target_idx": target_idx, "target_smiles": smi_sub, "error": repr(exc),
                })
            continue

        if len(rxn_idxs_unique) > 0:
            n_targets_with_ecfp_apply += 1

        for child_idx, rxn_idx in enumerate(rxn_idxs_unique):
            n_ecfp_applies += 1
            rxn_idx_int = int(rxn_idx)

            predicted_child_ecfp = np.asarray(smi_ecfp_childs[child_idx], dtype=np.int32)
            unique_predicted_ecfps.add(ecfp_to_key(predicted_child_ecfp))

            template_reaction = reaction_rules.template_reaction[rxn_idx_int]
            rule_smi_sub = safe_list_get(reaction_rules.smi_sub, rxn_idx_int)
            reaction_id = safe_list_get(reaction_rules.reaction_id, rxn_idx_int)
            reaction_monocomp_id = safe_list_get(reaction_rules.reaction_monocomp_id, rxn_idx_int)

            local_accuracy = False
            apply_error = ""
            product_ecfp_errors_for_case = 0

            try:
                smi_prods = apply_reaction(template_reaction, smi_sub)
            except Exception as exc:
                n_apply_errors += 1
                smi_prods = []
                apply_error = repr(exc)
                if debug:
                    error_cases.append({
                        "stage": "apply_reaction", "benchmark_name": benchmark_name,
                        "database_name": database_name, "radius": ecfp_params["radius"],
                        "target_idx": target_idx, "target_smiles": smi_sub,
                        "rule_idx": rxn_idx_int, "reaction_id": reaction_id,
                        "reaction_monocomp_id": reaction_monocomp_id,
                        "template_reaction": template_reaction, "error": apply_error,
                    })

            smi_prods_sanitized = sanitize_list_of_smiles(smi_prods)
            unique_template_smiles.update(smi_prods_sanitized)

            for smi_prod in smi_prods_sanitized:
                try:
                    smi_prod_ecfp = np.asarray(get_mol_ecfp(smi_prod, ecfp_params), dtype=np.int32)
                except Exception as exc:
                    n_product_ecfp_errors += 1
                    product_ecfp_errors_for_case += 1
                    if debug:
                        error_cases.append({
                            "stage": "product_ecfp", "benchmark_name": benchmark_name,
                            "database_name": database_name, "radius": ecfp_params["radius"],
                            "target_idx": target_idx, "target_smiles": smi_sub,
                            "rule_idx": rxn_idx_int, "reaction_id": reaction_id,
                            "product_smiles": smi_prod, "error": repr(exc),
                        })
                    continue

                if np.array_equal(smi_prod_ecfp, predicted_child_ecfp):
                    local_accuracy = True
                    break

            if local_accuracy:
                n_correct += 1
            elif debug:
                if apply_error:
                    failure_reason = "apply_reaction_error"
                elif len(smi_prods) == 0:
                    failure_reason = "no_raw_products"
                elif len(smi_prods_sanitized) == 0:
                    failure_reason = "no_sanitized_products"
                elif product_ecfp_errors_for_case == len(smi_prods_sanitized):
                    failure_reason = "all_product_ecfp_errors"
                else:
                    failure_reason = "predicted_ecfp_not_found_in_products"

                failed_cases.append({
                    "benchmark_name": benchmark_name, "database_name": database_name,
                    "radius": ecfp_params["radius"], "fpSize": ecfp_params["fpSize"],
                    "folded": ecfp_params["folded"], "custom": ecfp_params["custom"],
                    "target_idx": target_idx, "target_smiles": smi_sub,
                    "rule_idx": rxn_idx_int, "reaction_id": reaction_id,
                    "reaction_monocomp_id": reaction_monocomp_id,
                    "rule_smi_sub": rule_smi_sub, "template_reaction": template_reaction,
                    "predicted_child_ecfp_nonzero_bits": ecfp_nonzero_bits(predicted_child_ecfp),
                    "n_raw_products": len(smi_prods),
                    "n_sanitized_products": len(smi_prods_sanitized),
                    "n_product_ecfp_errors_for_case": product_ecfp_errors_for_case,
                    "sanitized_products": ";".join(smi_prods_sanitized[:DEBUG_MAX_PRODUCTS_PER_ROW]),
                    "products_truncated": len(smi_prods_sanitized) > DEBUG_MAX_PRODUCTS_PER_ROW,
                    "apply_error": apply_error, "failure_reason": failure_reason,
                })

        if (target_idx + 1) % 100 == 0:
            current_accuracy = n_correct / n_ecfp_applies if n_ecfp_applies else float("nan")
            print(
                f"{benchmark_name} | {database_name} | radius={ecfp_params['radius']} | "
                f"target {target_idx + 1}/{n_targets_total} | accuracy={current_accuracy:.4f} | "
                f"succeeded={n_correct} | failed={n_ecfp_applies - n_correct} | total={n_ecfp_applies}"
            )

    n_total_cases = n_ecfp_applies
    n_failed = n_total_cases - n_correct
    accuracy = n_correct / n_total_cases if n_total_cases > 0 else float("nan")
    elapsed_s = time.perf_counter() - t0

    print(
        f"Final | benchmark={benchmark_name} | database={database_name} | "
        f"radius={ecfp_params['radius']} | accuracy={accuracy:.6f} | "
        f"succeeded={n_correct} | failed={n_failed} | total={n_total_cases} | "
        f"time={elapsed_s:.1f}s"
    )

    if debug:
        if debug_dir is None:
            debug_dir = DEFAULT_OUT_XLSX.parent / "debug"
        export_debug_tables(
            debug_dir=debug_dir,
            benchmark_name=benchmark_name,
            database_name=database_name,
            ecfp_params=ecfp_params,
            failed_cases=failed_cases,
            error_cases=error_cases,
            applicability_mode=applicability_mode,
        )

    return {
        "benchmark_name": benchmark_name,
        "database_name": database_name,
        "applicability_mode": applicability_mode,
        "radius": ecfp_params["radius"],
        "fpSize": ecfp_params["fpSize"],
        "folded": ecfp_params["folded"],
        "custom": ecfp_params["custom"],
        "min_smi_sub_atoms": min_smi_sub_atoms,
        "accuracy": accuracy,
        "failure_rate": n_failed / n_total_cases if n_total_cases > 0 else float("nan"),
        "target_coverage": (
            n_targets_with_ecfp_apply / n_targets_valid if n_targets_valid > 0 else float("nan")
        ),
        "diversity_ratio_templates": (
            len(unique_template_smiles) / n_total_cases if n_total_cases > 0 else float("nan")
        ),
        "diversity_ratio_predicted_ecfps": (
            len(unique_predicted_ecfps) / n_total_cases if n_total_cases > 0 else float("nan")
        ),
        "n_total_cases": n_total_cases,
        "n_succeeded": n_correct,
        "n_failed": n_failed,
        "n_ecfp_applies": n_ecfp_applies,
        "n_unique_template_smiles": len(unique_template_smiles),
        "n_unique_predicted_ecfps": len(unique_predicted_ecfps),
        "n_targets_total": n_targets_total,
        "n_targets_valid": n_targets_valid,
        "n_targets_with_ecfp_apply": n_targets_with_ecfp_apply,
        "n_invalid_targets": n_invalid_targets,
        "n_target_ecfp_errors": n_target_ecfp_errors,
        "n_one_step_errors": n_one_step_errors,
        "n_apply_errors": n_apply_errors,
        "n_product_ecfp_errors": n_product_ecfp_errors,
        "elapsed_s": elapsed_s,
    }


# ======================================================================================
# Output helpers
# ======================================================================================


def make_matrix(results_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    index_cols = ["benchmark_name", "database_name"]
    if "applicability_mode" in results_df.columns:
        index_cols.append("applicability_mode")
    matrix = results_df.pivot_table(
        index=index_cols,
        columns="radius",
        values=value_col,
        aggfunc="first",
    )
    matrix = matrix.rename(columns={r: f"r{r}" for r in matrix.columns})
    return matrix.reset_index()


def export_results_to_excel(
    results_df: pd.DataFrame,
    benchmark_summary_df: pd.DataFrame,
    out_xlsx: Path,
) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "accuracy", "failure_rate", "target_coverage",
        "n_total_cases", "n_succeeded", "n_failed", "n_ecfp_applies",
        "n_unique_template_smiles", "n_unique_predicted_ecfps",
        "diversity_ratio_templates", "diversity_ratio_predicted_ecfps",
        "n_targets_total", "n_targets_valid", "n_targets_with_ecfp_apply",
        "n_invalid_targets", "n_target_ecfp_errors", "n_one_step_errors",
        "n_apply_errors", "n_product_ecfp_errors", "elapsed_s",
    ]

    with pd.ExcelWriter(out_xlsx) as writer:
        results_df.to_excel(writer, sheet_name="details", index=False)
        benchmark_summary_df.to_excel(writer, sheet_name="benchmark_summary", index=False)
        for col in metric_columns:
            make_matrix(results_df, col).to_excel(writer, sheet_name=col[:31], index=False)

    print(f"Saved Excel file to: {out_xlsx}")


# ======================================================================================
# CLI
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired one-step applicability/accuracy benchmark for morganrxn.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radii", default=",".join(map(str, DEFAULT_RADII)))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--fp-size", type=int, default=DEFAULT_FP_SIZE)
    parser.add_argument("--unfolded", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--min-heavy-atoms", type=int, default=DEFAULT_MIN_HEAVY_ATOMS)
    parser.add_argument("--min-smi-sub-atoms", type=int, default=DEFAULT_MIN_SMI_SUB_ATOMS)
    parser.add_argument("--max-mol-wt", type=float, default=DEFAULT_MAX_MOL_WT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--limit-targets", type=int, default=None)
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX)
    parser.add_argument(
        "--applicability-modes",
        default="reaction_center,reaction",
        help=(
            "Comma-separated applicability criteria to evaluate. "
            "'reaction_center' uses the reaction-centre ECFP (intended morganrxn "
            "criterion); 'reaction' uses the reaction ECFP as a baseline to show "
            "the value of the reaction centre. Default: both."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--benchmark-dataset",
        action="append",
        default=None,
        help="Repeatable: dataset=database_a,database_b. Default: metanetx=metanetx and uspto=uspto.",
    )
    parser.add_argument(
        "--paired-rules",
        action="append",
        default=None,
        help="Repeatable: dataset=database_a,database_b. Default: no cross-tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    radii = parse_radii(args.radii)
    folded = not args.unfolded
    benchmark_datasets = parse_dataset_specs(args.benchmark_dataset, DEFAULT_BENCHMARK_DATASETS)
    paired_rule_names = parse_dataset_specs(args.paired_rules, DEFAULT_PAIRED_RULES)

    applicability_modes = [m.strip() for m in args.applicability_modes.split(",") if m.strip()]
    invalid_modes = [m for m in applicability_modes if m not in ("reaction_center", "reaction")]
    if not applicability_modes or invalid_modes:
        raise ValueError(
            f"Invalid --applicability-modes {args.applicability_modes!r}; "
            "expected a comma-separated subset of 'reaction_center,reaction'."
        )

    print("Paired one-step applicability/accuracy benchmark")
    print("=" * 80)
    print("radii:", radii)
    print("applicability_modes:", applicability_modes)
    print("n_samples:", args.n_samples)
    print("fp_size:", args.fp_size)
    print("folded:", folded)
    print("min_heavy_atoms:", args.min_heavy_atoms)
    print("min_smi_sub_atoms:", args.min_smi_sub_atoms)
    print("max_mol_wt:", args.max_mol_wt)
    print("random_seed:", args.random_seed)
    print("out_xlsx:", args.out_xlsx)
    print("debug:", args.debug)

    ecfp_params_for_benchmark = make_ecfp_params(
        radius=2, fp_size=args.fp_size, folded=folded, custom=args.custom
    )

    with timer("Creating benchmark molecule sets"):
        benchmark_smiles, benchmark_summary_df = create_benchmark_sets_in_memory(
            benchmark_datasets=benchmark_datasets,
            ecfp_params=ecfp_params_for_benchmark,
            n_samples=args.n_samples,
            random_seed=args.random_seed,
            min_heavy_atoms=args.min_heavy_atoms,
            max_mol_wt=args.max_mol_wt,
        )

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = out_xlsx.parent / "debug"

    results: List[Dict[str, Any]] = []

    for benchmark_name, targets in benchmark_smiles.items():
        rule_names = paired_rule_names.get(benchmark_name)
        if not rule_names:
            print(f"[skip] No paired rules configured for benchmark: {benchmark_name}")
            continue

        smi_targets = targets[: args.limit_targets] if args.limit_targets is not None else targets
        print("=" * 80)
        print(f"Benchmark: {benchmark_name}")
        print(f"Paired ReactionRules database(s): {', '.join(rule_names)}")
        print(f"Target molecules: {len(smi_targets)}")

        for database_name in rule_names:
            for radius in radii:
                ecfp_params = make_ecfp_params(
                    radius=radius, fp_size=args.fp_size, folded=folded, custom=args.custom
                )
                for applicability_mode in applicability_modes:
                    with timer(
                        f"Accuracy | {benchmark_name} | {database_name} | "
                        f"radius={radius} | mode={applicability_mode}"
                    ):
                        result = compute_ecfp_applies_accuracy(
                            benchmark_name=benchmark_name,
                            database_name=database_name,
                            smi_targets=smi_targets,
                            ecfp_params=ecfp_params,
                            min_smi_sub_atoms=args.min_smi_sub_atoms,
                            applicability_mode=applicability_mode,
                            debug=args.debug,
                            debug_dir=debug_dir if args.debug else None,
                        )
                    results.append(result)

                    # Save incrementally so a partial run still produces output.
                    results_df = pd.DataFrame(results)
                    export_results_to_excel(
                        results_df=results_df,
                        benchmark_summary_df=benchmark_summary_df,
                        out_xlsx=out_xlsx,
                    )

    print("=" * 80)
    print("All benchmarks done.")
    print("out_xlsx:", out_xlsx)


if __name__ == "__main__":
    main()
