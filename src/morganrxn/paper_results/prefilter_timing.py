#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime comparison: O(d) reaction-centre ECFP prefilter vs graph-level matching.

For the target molecules of the applicability benchmark (Table 2) and every rule of a ReactionRules database, times

  1. ECFP computation of the target (needed once per target),
  2. the vectorised reaction-centre ECFP prefilter over ALL rules (one_step mask),
  3. a graph-level applicability test over all rules: RDKit subgraph isomorphism
     (``HasSubstructMatch``) of the template reactant pattern against the target,
  4. the prefilter followed by subgraph matching on the surviving candidates only
     (the intended pipeline),
  5. direct comparison on the first ``--n-full-targets`` targets: apply EVERY
     template at the graph level vs prefilter + apply on candidates, with a check
     that both give the same applicable rules,
  6. full template application (``apply_reaction``): per random (target, rule)
     pair (cost without prefilter, extrapolated to all rules) and per pair that
     passes the prefilter (cost inside the vector-then-graph pipeline).

It also counts (target, rule) pairs where the target contains the template
pattern but the prefilter rejects it; the prefilter is a necessary condition, so
this must be 0.

Pre-compilation of the rule patterns (SMARTS parsing) is done once and excluded
from the per-target timings; it is reported separately.

Example:
    python prefilter_timing.py --database-name uspto --radii 0,1,2,3,4,5 \
        --n-targets 1000 --n-graph-targets 100 --n-apply-pairs 500
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog("rdApp.*")

from morganrxn.core.cli_utils import make_ecfp_params, parse_radii
from morganrxn.core.molecule_utils import get_mol_ecfp
from morganrxn.core.paths import RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules
from morganrxn.core.reaction_utils import apply_reaction, one_step
from morganrxn.paper_results.applicability_accuracy import create_benchmark_sets_in_memory

DEFAULT_OUT_CSV = RESULTS_DIR / "prefilter_timing" / "prefilter_timing.csv"


def compile_patterns(templates: List[str]):
    """Reactant query pattern of each single-reactant template (None otherwise)."""
    patterns = []
    for tpl in templates:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(tpl)
            # copy: the template is a view on `rxn`, which is freed at loop exit
            patterns.append(
                Chem.Mol(rxn.GetReactantTemplate(0)) if rxn.GetNumReactantTemplates() == 1 else None
            )
        except Exception:
            patterns.append(None)
    return patterns


def time_config(
    database_name: str,
    ecfp_params: dict,
    targets: List[str],
    n_graph_targets: int,
    n_apply_pairs: int,
    seed: int,
    min_smi_sub_atoms: int,
    n_repeats: int,
    all_candidate_pairs: bool = False,
    n_full_targets: int = 0,
) -> dict:
    rules = ReactionRules.load(database_name=database_name, ecfp_params=ecfp_params)
    rules.filter_by_smi_sub_atoms(min_atoms=min_smi_sub_atoms, verbose=False)

    centers = np.asarray(rules.ecfp_reaction_center, dtype=np.int32)
    reactions = np.asarray(rules.ecfp_reaction, dtype=np.int32)
    templates = list(rules.template_reaction)
    n_rules, d = centers.shape

    t0 = time.perf_counter()
    patterns = compile_patterns(templates)
    t_compile = time.perf_counter() - t0
    valid = np.array([p is not None for p in patterns])

    graph_targets = targets[:n_graph_targets]

    rng = random.Random(seed)
    t_ecfp, t_prefilter, t_one_step = [], [], []
    n_candidates, n_unique_candidates, cand_pairs = [], [], []
    for smi in targets:
        t = time.perf_counter()
        v = np.asarray(get_mol_ecfp(smi, ecfp_params), dtype=np.int32)
        t_ecfp.append(time.perf_counter() - t)

        best = np.inf
        for _ in range(n_repeats):
            t = time.perf_counter()
            mask = np.all((v[None, :] + centers) >= 0, axis=1)
            idxs = np.flatnonzero(mask)
            best = min(best, time.perf_counter() - t)
        t_prefilter.append(best)
        n_candidates.append(len(idxs))

        # one_step = mask + child vectors + de-duplication of child vectors: the exact
        # candidate pairs of the applicability benchmark (Table 2)
        best = np.inf
        for _ in range(n_repeats):
            t = time.perf_counter()
            _, rxn_unique = one_step(v, reactions, centers)
            best = min(best, time.perf_counter() - t)
        t_one_step.append(best)
        n_unique_candidates.append(len(rxn_unique))
        if len(rxn_unique):
            if all_candidate_pairs:
                cand_pairs.extend((smi, int(i)) for i in rxn_unique)
            else:
                cand_pairs.append((smi, int(rxn_unique[rng.randrange(len(rxn_unique))])))

    t_graph_all, t_pipeline, missed = [], [], 0
    n_graph_true_in_cands = n_cands_checked = 0
    for smi in graph_targets:
        mol = Chem.MolFromSmiles(smi)
        v = np.asarray(get_mol_ecfp(smi, ecfp_params), dtype=np.int32)

        t = time.perf_counter()
        hit = np.zeros(n_rules, dtype=bool)
        for i in np.flatnonzero(valid):
            hit[i] = mol.HasSubstructMatch(patterns[i])
        t_graph_all.append(time.perf_counter() - t)

        t = time.perf_counter()
        mask = np.all((v[None, :] + centers) >= 0, axis=1)
        cands = np.flatnonzero(mask & valid)
        n_true = sum(1 for i in cands if mol.HasSubstructMatch(patterns[i]))
        t_pipeline.append(time.perf_counter() - t)

        missed += int(np.sum(hit & ~mask))
        n_graph_true_in_cands += n_true
        n_cands_checked += len(cands)

    # Direct comparison on the same targets: (A) apply EVERY template at the graph
    # level vs (B) vector prefilter, then apply only the surviving templates.
    tA, tB, tBu, missed_apply, n_applied_A, n_applied_B = [], [], [], 0, 0, 0
    for smi in targets[:n_full_targets]:
        v = np.asarray(get_mol_ecfp(smi, ecfp_params), dtype=np.int32)

        t = time.perf_counter()
        applied_A = set()
        for i in range(n_rules):
            try:
                if apply_reaction(templates[i], smi):
                    applied_A.add(i)
            except Exception:
                pass
        tA.append(time.perf_counter() - t)

        t = time.perf_counter()
        cands = np.flatnonzero(np.all((v[None, :] + centers) >= 0, axis=1))
        applied_B = set()
        for i in cands:
            try:
                if apply_reaction(templates[int(i)], smi):
                    applied_B.add(int(i))
            except Exception:
                pass
        tB.append(time.perf_counter() - t)

        t = time.perf_counter()
        _, rxn_unique = one_step(v, reactions, centers)
        for i in rxn_unique:
            try:
                apply_reaction(templates[int(i)], smi)
            except Exception:
                pass
        tBu.append(time.perf_counter() - t)

        missed_apply += len(applied_A - applied_B)
        n_applied_A += len(applied_A)
        n_applied_B += len(applied_B)

    def time_apply(pairs):
        out = []
        for smi, tpl in pairs:
            t = time.perf_counter()
            try:
                apply_reaction(tpl, smi)
            except Exception:
                pass
            out.append(time.perf_counter() - t)
        return out

    # (a) random pairs: cost of trying every rule without any prefilter
    random_pairs = [
        (targets[rng.randrange(len(targets))], templates[rng.randrange(n_rules)])
        for _ in range(n_apply_pairs if targets else 0)
    ]
    t_apply = time_apply(random_pairs)
    # (b) pairs that pass the prefilter: cost per candidate in the pipeline
    cand_sample = (
        cand_pairs if all_candidate_pairs
        else rng.sample(cand_pairs, min(n_apply_pairs, len(cand_pairs)))
    )
    t_apply_cand = time_apply([(smi, templates[i]) for smi, i in cand_sample])

    def mean_ms(xs):
        return float(np.mean(xs) * 1e3) if len(xs) else float("nan")

    prefilter_ms = mean_ms(t_prefilter)
    graph_all_ms = mean_ms(t_graph_all)
    pipeline_ms = mean_ms(t_pipeline)
    apply_ms_pair = mean_ms(t_apply)
    apply_ms_cand = mean_ms(t_apply_cand)
    mean_cands = float(np.mean(n_candidates)) if n_candidates else float("nan")
    return {
        "database": database_name,
        "radius": ecfp_params["radius"],
        "fpSize": ecfp_params["fpSize"],
        "n_rules": n_rules,
        "n_targets_prefilter": len(targets),
        "n_targets_graph": len(graph_targets),
        "n_apply_pairs": len(t_apply),
        "mean_candidates_per_target": float(np.mean(n_candidates)) if n_candidates else float("nan"),
        "ecfp_target_ms": mean_ms(t_ecfp),
        "prefilter_ms_per_target": prefilter_ms,
        "prefilter_us_per_pair": prefilter_ms * 1e3 / n_rules,
        "subgraph_all_rules_ms_per_target": graph_all_ms,
        "subgraph_us_per_pair": graph_all_ms * 1e3 / max(int(valid.sum()), 1),
        "speedup_prefilter_vs_subgraph": graph_all_ms / prefilter_ms if prefilter_ms else float("nan"),
        "pipeline_ms_per_target": pipeline_ms,
        "speedup_pipeline_vs_subgraph": graph_all_ms / pipeline_ms if pipeline_ms else float("nan"),
        "apply_reaction_ms_per_pair": apply_ms_pair,
        "apply_reaction_s_per_target_extrapolated": apply_ms_pair * n_rules / 1e3,
        "n_full_targets": len(tA),
        "graph_all_templates_s_per_target_measured": float(np.mean(tA)) if tA else float("nan"),
        "vector_then_graph_s_per_target_measured": float(np.mean(tB)) if tB else float("nan"),
        "speedup_measured": float(np.mean(tA) / np.mean(tB)) if tA else float("nan"),
        "vector_then_graph_unique_s_per_target_measured": float(np.mean(tBu)) if tBu else float("nan"),
        "speedup_measured_unique": float(np.mean(tA) / np.mean(tBu)) if tA else float("nan"),
        "mean_unique_candidates_per_target": float(np.mean(n_unique_candidates)) if n_unique_candidates else float("nan"),
        "one_step_ms_per_target": mean_ms(t_one_step),
        "rules_applied_graph_only": n_applied_A,
        "rules_applied_vector_then_graph": n_applied_B,
        "rules_missed_by_vector_filter": missed_apply,
        "apply_reaction_ms_per_candidate_pair": apply_ms_cand,
        "vector_then_apply_s_per_target": (
            mean_ms(t_one_step) + float(np.mean(n_unique_candidates)) * apply_ms_cand
        ) / 1e3,
        "speedup_vector_then_apply_vs_apply_all": (
            apply_ms_pair * n_rules
            / (mean_ms(t_one_step) + float(np.mean(n_unique_candidates)) * apply_ms_cand)
        ),
        "missed_by_prefilter": int(missed),
        "subgraph_true_among_candidates": int(n_graph_true_in_cands),
        "candidates_checked_with_subgraph": int(n_cands_checked),
        "compile_patterns_s": t_compile,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Runtime of the ECFP prefilter vs graph-level matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database-name", default="uspto")
    p.add_argument("--radii", default="0,1,2,3,4,5")
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--unfolded", action="store_true")
    p.add_argument("--custom", action="store_true")
    p.add_argument("--n-targets", type=int, default=1000,
                   help="Benchmark targets (1000 = Table 2 sample).")
    p.add_argument("--n-graph-targets", type=int, default=100,
                   help="First N benchmark targets for the all-rules subgraph baseline.")
    p.add_argument("--n-apply-pairs", type=int, default=500,
                   help="Random (target, rule) pairs for full template application.")
    p.add_argument("--all-candidate-pairs", action="store_true",
                   help="Time apply_reaction on EVERY pair passing the prefilter "
                        "(the exact pairs of the applicability benchmark).")
    p.add_argument("--n-full-targets", type=int, default=20,
                   help="Targets on which EVERY template is applied at graph level "
                        "(direct graph-only vs vector+graph comparison).")
    p.add_argument("--n-repeats", type=int, default=3,
                   help="Repeats of the prefilter per target (best time kept).")
    p.add_argument("--min-heavy-atoms", type=int, default=5)
    p.add_argument("--min-smi-sub-atoms", type=int, default=5)
    p.add_argument("--max-mol-wt", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return p


def main() -> None:
    args = build_parser().parse_args()
    radii = parse_radii(args.radii)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Same target molecules as the applicability benchmark (Table 2): pool built
    # from the radius-2 rules, same filters, same seed.
    benchmark_smiles, _ = create_benchmark_sets_in_memory(
        benchmark_datasets={args.database_name: [args.database_name]},
        ecfp_params=make_ecfp_params(
            radius=2, fp_size=args.fp_size, folded=not args.unfolded, custom=args.custom
        ),
        n_samples=args.n_targets,
        random_seed=args.seed,
        min_heavy_atoms=args.min_heavy_atoms,
        max_mol_wt=args.max_mol_wt,
    )
    targets = benchmark_smiles[args.database_name]

    rows = []
    for radius in radii:
        ecfp_params = make_ecfp_params(
            radius=radius, fp_size=args.fp_size, folded=not args.unfolded, custom=args.custom
        )
        print(f"[start] {args.database_name} | radius={radius} | fp={args.fp_size}", flush=True)
        row = time_config(
            database_name=args.database_name,
            ecfp_params=ecfp_params,
            targets=targets,
            n_graph_targets=args.n_graph_targets,
            n_apply_pairs=args.n_apply_pairs,
            seed=args.seed,
            min_smi_sub_atoms=args.min_smi_sub_atoms,
            n_repeats=args.n_repeats,
            all_candidate_pairs=args.all_candidate_pairs,
            n_full_targets=args.n_full_targets,
        )
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
