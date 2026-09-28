#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-target runtime comparison on the applicability benchmark (Table 2) targets.

For each target molecule (same deterministic 1000-molecule sample, same rules and
filters as ``applicability_accuracy.py``) and each radius, three routes are timed:

  graph          apply EVERY template at the graph level and compute the ECFP of
                 every product (graph-only way of obtaining the children),
  vector         ``one_step``: reaction-centre prefilter + child ECFP vectors
                 (target + reaction ECFP), de-duplicated,
  vector+graph   ``one_step``, then graph-level application (and product ECFPs)
                 on the de-duplicated candidates only: the pairs of Table 2.

and the filtering step alone:

  subgraph       RDKit subgraph isomorphism of every template reactant pattern,
  prefilter      coordinate-wise reaction-centre mask over all rules.

Each template is applied once; the per-template times are reused for the
vector+graph route, which therefore costs exactly the same calls as Table 2.
Templates are compiled once beforehand (compile time reported per job).

Checks written per target: rules that apply at the graph level or match as a
subgraph but are rejected by the prefilter (must be 0), and Table 2 counters
(cases, correct) so the accuracy can be compared with Table 2.

One CSV row per (radius, target); aggregate with merge_graph_vs_vector.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog("rdApp.*")

from morganrxn.core.cli_utils import make_ecfp_params, parse_radii
from morganrxn.core.molecule_utils import get_mol_ecfp, sanitize_list_of_smiles
from morganrxn.core.paths import RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules
from morganrxn.core.reaction_utils import apply_reaction, one_step
from morganrxn.paper_results.applicability_accuracy import (
    create_benchmark_sets_in_memory,
    ecfp_to_key,
)

DEFAULT_OUT_DIR = RESULTS_DIR / "graph_vs_vector"
N_REPEATS = 3


def compile_rules(templates):
    rxns, patterns = [], []
    for tpl in templates:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(tpl)
            rxn.Initialize()
        except Exception:
            rxns.append(None)
            patterns.append(None)
            continue
        rxns.append(rxn)
        # copy: the template is a view on `rxn`
        patterns.append(
            Chem.Mol(rxn.GetReactantTemplate(0)) if rxn.GetNumReactantTemplates() == 1 else None
        )
    return rxns, patterns


def product_keys(rxn, smi, ecfp_params):
    """Apply one template and return the ECFP keys of its sanitized products."""
    try:
        prods = apply_reaction(rxn, smi)
    except Exception:
        return None
    if not prods:
        return None
    keys = set()
    for p in sanitize_list_of_smiles(prods):
        try:
            keys.add(ecfp_to_key(np.asarray(get_mol_ecfp(p, ecfp_params), dtype=np.int32)))
        except Exception:
            pass
    return keys


def time_target(smi, ecfp_params, centers, reactions, rxns, patterns, valid, graph_budget_s):
    n_rules = len(rxns)
    mol = Chem.MolFromSmiles(smi)

    t = time.perf_counter()
    v = np.asarray(get_mol_ecfp(smi, ecfp_params), dtype=np.int32)
    t_ecfp = time.perf_counter() - t

    # warm-up (first large allocation pays page faults), then best of N_REPEATS
    one_step(v, reactions, centers)
    t_mask = t_one_step = np.inf
    for _ in range(N_REPEATS):
        t = time.perf_counter()
        mask = np.all((v[None, :] + centers) >= 0, axis=1)
        t_mask = min(t_mask, time.perf_counter() - t)
        t = time.perf_counter()
        child_vecs, rxn_unique = one_step(v, reactions, centers)
        t_one_step = min(t_one_step, time.perf_counter() - t)

    # graph filter: subgraph isomorphism of every template pattern
    t_sub = np.zeros(n_rules)
    sub_hit = np.zeros(n_rules, dtype=bool)
    t0 = time.perf_counter()
    for i in np.flatnonzero(valid):
        t = time.perf_counter()
        sub_hit[i] = mol.HasSubstructMatch(patterns[i])
        t_sub[i] = time.perf_counter() - t
    t_subgraph_all = time.perf_counter() - t0

    # graph route: apply every template, ECFP of every product
    unique_set = set(int(i) for i in rxn_unique)
    t_app = np.zeros(n_rules)
    applied = np.zeros(n_rules, dtype=bool)
    keys_unique = {}
    # Templates are applied in rule order; past graph_budget_s the graph route is
    # stopped (censored: t_graph is then a lower bound) and the candidates not yet
    # reached are applied on their own, so the vector+graph route stays complete.
    censored = False
    n_reached = n_rules
    t0 = time.perf_counter()
    for i in range(n_rules):
        if time.perf_counter() - t0 > graph_budget_s:
            censored = True
            n_reached = i
            break
        if rxns[i] is None:
            continue
        t = time.perf_counter()
        keys = product_keys(rxns[i], smi, ecfp_params)
        t_app[i] = time.perf_counter() - t
        if keys is not None:
            applied[i] = True
        if i in unique_set:
            keys_unique[i] = keys or set()
    t_graph = time.perf_counter() - t0

    for i in sorted(unique_set):
        if i < n_reached or rxns[i] is None:
            continue
        t = time.perf_counter()
        keys = product_keys(rxns[i], smi, ecfp_params)
        t_app[i] = time.perf_counter() - t
        keys_unique[i] = keys or set()

    # Table 2 counters on the de-duplicated candidates
    n_correct = sum(
        1 for k, i in enumerate(rxn_unique)
        if ecfp_to_key(np.asarray(child_vecs[k], dtype=np.int32)) in keys_unique.get(int(i), set())
    )

    mask_idx = np.flatnonzero(mask)
    return {
        "smiles": smi,
        "n_rules": n_rules,
        "n_candidates_mask": len(mask_idx),
        "n_candidates_unique": len(rxn_unique),
        "n_subgraph_hits": int(sub_hit.sum()),
        "n_rules_applied": int(applied.sum()),
        "graph_censored": censored,
        "n_templates_reached": int(n_reached),
        "missed_subgraph_by_prefilter": int(np.sum(sub_hit & ~mask)),
        "missed_applied_by_prefilter": int(np.sum(applied & ~mask)),
        "n_cases": len(rxn_unique),
        "n_correct": int(n_correct),
        "t_ecfp_target_s": t_ecfp,
        "t_prefilter_s": t_mask,
        "t_subgraph_all_s": t_subgraph_all,
        "t_prefilter_then_subgraph_s": t_mask + float(t_sub[mask_idx].sum()),
        "t_vector_s": t_one_step,
        "t_graph_s": t_graph,
        "t_vector_then_graph_s": t_one_step + float(t_app[list(unique_set)].sum()) if unique_set else t_one_step,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-target graph vs vector runtime on the Table 2 targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database-name", default="uspto")
    p.add_argument("--radii", default="0,1,2,3,4,5")
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--unfolded", action="store_true")
    p.add_argument("--custom", action="store_true")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--target-start", type=int, default=0)
    p.add_argument("--target-end", type=int, default=None)
    p.add_argument("--min-heavy-atoms", type=int, default=5)
    p.add_argument("--min-smi-sub-atoms", type=int, default=5)
    p.add_argument("--max-mol-wt", type=float, default=1000.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--graph-budget-s", type=float, default=1800.0,
                   help="Per-target budget of the graph route; beyond it the target is "
                        "censored (graph time = lower bound).")
    p.add_argument("--exclude-targets", default="",
                   help="Comma-separated target indices to skip (e.g. 99,912).")
    p.add_argument("--out-csv", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    folded = not args.unfolded
    radii = parse_radii(args.radii)

    benchmark_smiles, _ = create_benchmark_sets_in_memory(
        benchmark_datasets={args.database_name: [args.database_name]},
        ecfp_params=make_ecfp_params(radius=2, fp_size=args.fp_size, folded=folded, custom=args.custom),
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        min_heavy_atoms=args.min_heavy_atoms,
        max_mol_wt=args.max_mol_wt,
    )
    targets = benchmark_smiles[args.database_name]
    end = args.target_end if args.target_end is not None else len(targets)
    excluded = {int(x) for x in args.exclude_targets.split(",") if x.strip()}
    chunk = [(i, t) for i, t in list(enumerate(targets))[args.target_start:end] if i not in excluded]
    print(f"Targets {args.target_start}..{end - 1} ({len(chunk)})", flush=True)

    out_dir = DEFAULT_OUT_DIR if args.max_mol_wt == 1000 else (
        DEFAULT_OUT_DIR.parent / f"{DEFAULT_OUT_DIR.name}_mw{args.max_mol_wt:g}"
    )
    out_csv = args.out_csv or (
        out_dir
        / f"{args.database_name}_fp{args.fp_size}_r{''.join(map(str, radii))}"
          f"_t{args.target_start:04d}-{end:04d}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for radius in radii:
        ecfp_params = make_ecfp_params(radius=radius, fp_size=args.fp_size, folded=folded, custom=args.custom)
        rules = ReactionRules.load(database_name=args.database_name, ecfp_params=ecfp_params)
        rules.filter_by_smi_sub_atoms(min_atoms=args.min_smi_sub_atoms, verbose=False)
        centers = np.asarray(rules.ecfp_reaction_center, dtype=np.int32)
        reactions = np.asarray(rules.ecfp_reaction, dtype=np.int32)

        t = time.perf_counter()
        rxns, patterns = compile_rules(list(rules.template_reaction))
        t_compile = time.perf_counter() - t
        valid = np.array([p is not None for p in patterns])
        print(f"[radius {radius}] {len(rxns)} rules, compiled in {t_compile:.0f}s", flush=True)

        for target_idx, smi in chunk:
            row = time_target(smi, ecfp_params, centers, reactions, rxns, patterns, valid, args.graph_budget_s)
            row.update({
                "database": args.database_name, "fpSize": args.fp_size, "radius": radius,
                "target_idx": target_idx, "t_compile_rules_s": t_compile,
            })
            rows.append(row)
            print(
                f"[r{radius}] target {target_idx} | graph={row['t_graph_s']:.1f}s "
                f"vector={row['t_vector_s'] * 1e3:.0f}ms "
                f"vector+graph={row['t_vector_then_graph_s']:.1f}s "
                f"cands={row['n_candidates_unique']} "
                f"missed={row['missed_applied_by_prefilter']}"
                f"{' CENSORED' if row['graph_censored'] else ''}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
