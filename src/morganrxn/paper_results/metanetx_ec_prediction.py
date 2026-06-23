#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-safe MetaNetX EC-number prediction benchmark from reaction ECFP features.

This script loads MetaNetX EC annotations from reac_prop.tsv, matches them to
ReactionRules entries, and compares reaction ECFP, reaction-center ECFP, and
optionally both concatenated as inputs for EC-number prediction.

Changes compared to the previous version
------------------------------------------
- Feature matrices are built directly in sparse form (row by row, then
  ``scipy.sparse.vstack``ed once), instead of materializing a dense
  (n_samples, d) array before converting it to sparse. This removes the main
  source of the MemoryError / out-of-memory crashes observed on large runs
  (in particular with ``--sample-mode occurrences``).
- The SGD (linear, one-vs-rest) classifier now uses class-balanced weights by
  default. Without it, on highly imbalanced, high-cardinality multi-label
  settings (e.g. EC truncation levels 3-4, 250-300 labels), the classifier was
  observed to collapse below the majority-class baseline: pass
  ``--sgd-unbalanced`` to reproduce the old (not recommended) behaviour.
- The "no prediction" fallback now defaults to the most frequent *training*
  label (matching the majority-class baseline) instead of an arbitrary fixed
  column, when no usable score is available.
- EC-label vocabulary selection (``--min-label-count``, ``--max-labels``) is
  now computed from the training split only, not from the full dataset, to
  avoid a (mild) form of label leakage from the test set into the choice of
  which EC labels are considered at all.
- Train/test stratification groups rare "primary label" classes into a single
  bucket instead of disabling stratification entirely as soon as one class is
  too rare.
- ``--n-jobs`` defaults to 4, matching the full-benchmark defaults below;
  lower it (e.g. ``--n-jobs 1``) on memory-constrained machines, since
  joblib's loky backend duplicates data per worker.
- The script now loops over EC truncation levels itself, the same way it
  already loops over radii: ``--ec-levels 1,2,3,4`` (the default) runs all
  four levels and writes a *single* combined CSV (one row per EC level x
  radius x feature set x classifier) to ``--summary-output``, instead of
  requiring one job per EC level. ``--ec-level`` (singular) still works as a
  deprecated single-value alias.

Running the script with no arguments at all now reproduces the full
benchmark:
    python metanetx_ec_prediction.py
is equivalent to:
    python metanetx_ec_prediction.py \\
        --ec-levels 1,2,3,4 --radii 0,1,2,3,4,5 --max-rules -1 --models sgd,et

This is meant for a cluster job (e.g. the IFB SLURM cluster), not a laptop.

Lighter, explicit local sanity check (single EC level, single radius, capped
rules, sgd only, n_jobs=1):
    python metanetx_ec_prediction.py \\
        --ec-levels 3 \\
        --radii 2 \\
        --max-labels 100 \\
        --min-label-count 10 \\
        --max-rules 5000 \\
        --sample-mode unique_rules \\
        --models sgd \\
        --n-jobs 1
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from morganrxn.core.paths import DATA_DIR, RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

METANETX_DIR = DATA_DIR / "metanetx"
DEFAULT_REAC_PROP = METANETX_DIR / "reac_prop.tsv"

RARE_PRIMARY_BUCKET = "__rare_primary__"


# ======================================================================================
# Small utilities
# ======================================================================================


class timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[start] {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        status = "failed" if exc_type is not None else "done"
        print(f"[{status}] {self.label}: {dt:.2f} s")
        return False


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_radii(radii_value: Optional[str], fallback_radius: int) -> List[int]:
    if radii_value is None or str(radii_value).strip() == "":
        return [int(fallback_radius)]
    radii = sorted({int(x.strip()) for x in str(radii_value).split(",") if x.strip()})
    if not radii:
        raise ValueError("No valid radius found.")
    return radii


def parse_ec_levels(ec_levels_value: Optional[str], fallback_ec_level: int) -> List[int]:
    if ec_levels_value is None or str(ec_levels_value).strip() == "":
        return [int(fallback_ec_level)]
    ec_levels = sorted({int(x.strip()) for x in str(ec_levels_value).split(",") if x.strip()})
    if not ec_levels:
        raise ValueError("No valid EC level found.")
    for lvl in ec_levels:
        if lvl not in (1, 2, 3, 4):
            raise ValueError(f"Invalid EC level {lvl}: must be one of 1, 2, 3, 4.")
    return ec_levels


def make_ecfp_params(radius: int, fp_size: int, folded: bool, custom: bool) -> dict:
    return {
        "radius": int(radius),
        "fpSize": int(fp_size),
        "folded": bool(folded),
        "custom": bool(custom),
    }


def split_merged_ids(value) -> List[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [x.strip() for x in value.split("|") if x.strip()]


def extract_mnxr_id(value) -> Optional[str]:
    if value is None:
        return None
    match = re.search(r"(MNXR[0-9]+)", str(value))
    return match.group(1) if match else None


def get_rule_source_ids(reaction_rules: ReactionRules, i: int) -> List[str]:
    ids = []
    if hasattr(reaction_rules, "reaction_monocomp_id"):
        ids.extend(split_merged_ids(reaction_rules.reaction_monocomp_id[i]))
    if hasattr(reaction_rules, "reaction_id"):
        ids.extend(split_merged_ids(reaction_rules.reaction_id[i]))

    if not ids:
        ids = [str(i)]

    out = []
    seen = set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def to_csr_float32(X) -> sparse.csr_matrix:
    """Defensive conversion to a float32 CSR matrix. A no-op (besides dtype
    enforcement) when ``X`` is already sparse, which is the expected case
    everywhere in this script after the rewrite."""
    if sparse.issparse(X):
        return X.astype(np.float32).tocsr()
    return sparse.csr_matrix(np.asarray(X, dtype=np.float32))


def row_to_sparse(vec) -> sparse.csr_matrix:
    """Convert a single feature vector to a 1 x d sparse row.

    Building feature matrices row-by-row in sparse form, then vstacking once,
    avoids ever materializing a dense (n_samples, d) array. This is the main
    fix for the MemoryError / OOM crashes seen on large runs.
    """
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    return sparse.csr_matrix(arr)


def stack_sparse_rows(rows: List[sparse.csr_matrix], n_features: int) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix((0, n_features), dtype=np.float32)
    return sparse.vstack(rows, format="csr", dtype=np.float32)


def subset_X(X, indices):
    return X[indices]


# ======================================================================================
# EC labels
# ======================================================================================


def parse_ec_list(value, ec_level: Optional[int] = None) -> List[str]:
    if value is None or pd.isna(value):
        return []

    tokens = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token or token.upper() in {"B", "T", "NOEC"}:
            continue
        if not re.match(r"^[0-9]+(\.[0-9-]+){0,3}$", token):
            continue
        if ec_level is not None:
            parts = token.split(".")
            if len(parts) < int(ec_level):
                continue
            token = ".".join(parts[: int(ec_level)])
        tokens.append(token)
    return sorted(set(tokens))


def load_metanetx_ec_labels(reac_prop_path: Path, ec_level: Optional[int] = None) -> Dict[str, List[str]]:
    print("Loading MetaNetX EC annotations...")
    print("reac_prop_path:", reac_prop_path)

    df = pd.read_csv(
        reac_prop_path,
        sep="\t",
        comment="#",
        header=None,
        names=["mnxr_id", "mnx_equation", "reference", "classifs", "is_balanced", "is_transport"],
        dtype=str,
        keep_default_na=False,
    )

    df["ec_numbers"] = df["classifs"].apply(lambda x: parse_ec_list(x, ec_level=ec_level))

    ec_by_mnxr = {
        row.mnxr_id: row.ec_numbers
        for row in df.itertuples(index=False)
        if row.ec_numbers
    }

    all_ec = [ec for ecs in ec_by_mnxr.values() for ec in ecs]
    print("reac_prop rows:", len(df))
    print("reactions with EC annotation:", len(ec_by_mnxr))
    print("unique EC labels:", len(set(all_ec)))
    print("top EC labels:")
    print(pd.Series(all_ec).value_counts().head(20))

    return ec_by_mnxr


# ======================================================================================
# ReactionRules -> ML samples (built directly as sparse rows)
# ======================================================================================


def expand_unique_rules(reaction_rules, ec_by_mnxr, max_rules=None):
    X_reaction_rows, X_center_rows, y_labels, meta_rows = [], [], [], []
    n_features = None

    n_rules = len(reaction_rules)
    if max_rules is not None and int(max_rules) > 0:
        n_rules = min(n_rules, int(max_rules))

    n_source_ids = 0
    n_bad_ids = 0
    n_rules_without_ec = 0
    n_labelled_source_ids = 0

    for i in range(n_rules):
        source_ids = get_rule_source_ids(reaction_rules, i)
        ec_set = set()
        mnxr_ids = []
        labelled_source_ids = []

        for source_id in source_ids:
            n_source_ids += 1
            mnxr_id = extract_mnxr_id(source_id)
            if mnxr_id is None:
                n_bad_ids += 1
                continue
            mnxr_ids.append(mnxr_id)

            ec_numbers = ec_by_mnxr.get(mnxr_id, [])
            if not ec_numbers:
                continue

            labelled_source_ids.append(source_id)
            n_labelled_source_ids += 1
            ec_set.update(ec_numbers)

        if not ec_set:
            n_rules_without_ec += 1
            continue

        ec_numbers = sorted(ec_set)
        unique_mnxr_ids = sorted(set(mnxr_ids))

        reaction_vec = reaction_rules.ecfp_reaction[i]
        center_vec = reaction_rules.ecfp_reaction_center[i]
        if n_features is None:
            n_features = int(np.asarray(reaction_vec).shape[-1])

        X_reaction_rows.append(row_to_sparse(reaction_vec))
        X_center_rows.append(row_to_sparse(center_vec))
        y_labels.append(tuple(ec_numbers))
        meta_rows.append(
            {
                "rule_index": i,
                "source_id": "|".join(labelled_source_ids),
                "source_ids": "|".join(source_ids),
                "labelled_source_ids": "|".join(labelled_source_ids),
                "mnxr_id": "|".join(unique_mnxr_ids),
                "mnxr_ids": "|".join(unique_mnxr_ids),
                "ec_numbers": ";".join(ec_numbers),
                "n_ec_numbers": len(ec_numbers),
                "n_source_ids": len(source_ids),
                "n_labelled_source_ids": len(labelled_source_ids),
            }
        )

    print("\nLabel matching summary")
    print("======================")
    print("Sample mode: unique_rules")
    print("ReactionRules entries read:", n_rules)
    print("Source IDs seen:", n_source_ids)
    print("Bad MNXR IDs:", n_bad_ids)
    print("Labelled source IDs:", n_labelled_source_ids)
    print("Rules without EC annotation:", n_rules_without_ec)
    print("Final labelled unique-rule samples before label filtering:", len(y_labels))

    if not y_labels:
        raise ValueError("No labelled samples recovered.")

    X_reaction_sp = stack_sparse_rows(X_reaction_rows, n_features)
    X_center_sp = stack_sparse_rows(X_center_rows, n_features)
    del X_reaction_rows, X_center_rows

    return X_reaction_sp, X_center_sp, y_labels, pd.DataFrame(meta_rows)


def expand_occurrences(reaction_rules, ec_by_mnxr, max_rules=None):
    X_reaction_rows, X_center_rows, y_labels, meta_rows = [], [], [], []
    n_features = None

    n_rules = len(reaction_rules)
    if max_rules is not None and int(max_rules) > 0:
        n_rules = min(n_rules, int(max_rules))

    n_source_ids = 0
    n_bad_ids = 0
    n_no_ec = 0

    for i in range(n_rules):
        reaction_vec = None
        center_vec = None

        for source_id in get_rule_source_ids(reaction_rules, i):
            n_source_ids += 1
            mnxr_id = extract_mnxr_id(source_id)
            if mnxr_id is None:
                n_bad_ids += 1
                continue

            ec_numbers = ec_by_mnxr.get(mnxr_id, [])
            if not ec_numbers:
                n_no_ec += 1
                continue

            if reaction_vec is None:
                reaction_vec = reaction_rules.ecfp_reaction[i]
                center_vec = reaction_rules.ecfp_reaction_center[i]
                if n_features is None:
                    n_features = int(np.asarray(reaction_vec).shape[-1])

            X_reaction_rows.append(row_to_sparse(reaction_vec))
            X_center_rows.append(row_to_sparse(center_vec))
            y_labels.append(tuple(ec_numbers))
            meta_rows.append(
                {
                    "rule_index": i,
                    "source_id": source_id,
                    "source_ids": source_id,
                    "labelled_source_ids": source_id,
                    "mnxr_id": mnxr_id,
                    "mnxr_ids": mnxr_id,
                    "ec_numbers": ";".join(ec_numbers),
                    "n_ec_numbers": len(ec_numbers),
                    "n_source_ids": 1,
                    "n_labelled_source_ids": 1,
                }
            )

    print("\nLabel matching summary")
    print("======================")
    print("Sample mode: occurrences")
    print("ReactionRules entries read:", n_rules)
    print("Source IDs seen:", n_source_ids)
    print("Bad MNXR IDs:", n_bad_ids)
    print("Source IDs without EC annotation:", n_no_ec)
    print("Final labelled occurrence samples before label filtering:", len(y_labels))
    print(
        "NOTE: --sample-mode occurrences can produce far more rows than "
        "unique_rules (one row per supporting source reaction). This is the "
        "single biggest driver of memory usage in this script: prefer "
        "unique_rules unless you specifically need occurrence weighting."
    )

    if not y_labels:
        raise ValueError("No labelled samples recovered.")

    X_reaction_sp = stack_sparse_rows(X_reaction_rows, n_features)
    X_center_sp = stack_sparse_rows(X_center_rows, n_features)
    del X_reaction_rows, X_center_rows

    return X_reaction_sp, X_center_sp, y_labels, pd.DataFrame(meta_rows)


def expand_reactionrules_with_ec_labels(reaction_rules, ec_by_mnxr, sample_mode, max_rules=None):
    if sample_mode == "unique_rules":
        return expand_unique_rules(reaction_rules, ec_by_mnxr, max_rules=max_rules)
    if sample_mode == "occurrences":
        return expand_occurrences(reaction_rules, ec_by_mnxr, max_rules=max_rules)
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


# ======================================================================================
# Labels, features and split (vocabulary selected from the training split only)
# ======================================================================================


def grouped_stratify_labels(primary: np.ndarray, min_count_for_stratify: int) -> Optional[np.ndarray]:
    """Group classes with fewer than ``min_count_for_stratify`` occurrences
    into a single bucket, instead of disabling stratification outright as
    soon as a single class is too rare. Returns ``None`` only if even the
    grouped distribution cannot be stratified."""
    counts = pd.Series(primary).value_counts()
    rare = set(counts[counts < min_count_for_stratify].index)
    if rare:
        grouped = np.array([RARE_PRIMARY_BUCKET if p in rare else p for p in primary], dtype=object)
    else:
        grouped = primary
    grouped_counts = pd.Series(grouped).value_counts()
    if grouped_counts.min() < 2:
        return None
    return grouped


def split_raw_indices(
    y_labels: Sequence[Tuple[str, ...]],
    test_size: float,
    random_state: int,
    min_count_for_stratify: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sample indices into train/test *before* any EC-label vocabulary
    filtering. Stratification uses the first (sorted) raw EC label per
    sample as a surrogate, grouping rare ones to keep stratification usable.

    Performing this split first, then choosing the kept EC-label vocabulary
    from the training half only, avoids leaking test-set label-frequency
    information into the choice of which EC labels are even considered.
    """
    n = len(y_labels)
    indices = np.arange(n)
    primary_raw = np.array([ecs[0] if ecs else "__none__" for ecs in y_labels], dtype=object)

    stratify = grouped_stratify_labels(primary_raw, min_count_for_stratify)
    if stratify is None:
        print("WARNING: non-stratified raw split used (too few examples per primary EC label, even after grouping).")

    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def binarize_labels_from_train_vocabulary(
    y_labels: Sequence[Tuple[str, ...]],
    train_idx: np.ndarray,
    min_label_count: int = 5,
    max_labels: Optional[int] = 300,
):
    """Choose the kept EC-label vocabulary using training-set frequencies
    only, then binarize all samples (train and test) against that fixed
    vocabulary. Samples left with zero labels after filtering are flagged in
    ``keep_sample`` for removal from both train and test.
    """
    train_set = set(int(i) for i in train_idx)
    train_label_counts = pd.Series(
        [ec for i in train_idx for ec in y_labels[int(i)]]
    ).value_counts()

    kept = train_label_counts[train_label_counts >= int(min_label_count)]
    if max_labels is not None:
        kept = kept.head(int(max_labels))

    kept_labels = sorted(kept.index.tolist())
    kept_set = set(kept_labels)

    filtered_y = [tuple(ec for ec in ecs if ec in kept_set) for ecs in y_labels]
    keep_sample = np.asarray([len(ecs) > 0 for ecs in filtered_y], dtype=bool)

    if len(kept_labels) == 0:
        raise ValueError("No EC labels kept from the training split. Lower --min-label-count or increase --max-labels.")
    if not keep_sample.any():
        raise ValueError("No samples left after EC label filtering.")

    mlb = MultiLabelBinarizer(classes=kept_labels)
    # Binarize every sample (train and test) against the train-chosen vocabulary.
    # Samples with no kept label become all-zero rows here and are dropped
    # afterwards via keep_sample.
    Y = mlb.fit_transform([ecs if ecs else ("__placeholder__",) for ecs in filtered_y]).astype(np.int8)
    # The placeholder above guarantees fit_transform never errors on an empty
    # tuple; it never produces a column (it is not in kept_labels), so rows
    # without a kept label are correctly all-zero.
    Y[~keep_sample, :] = 0

    print("\nLabel filtering (vocabulary chosen from the training split only)")
    print("===================================================================")
    print("Unique EC labels seen in training split:", len(train_label_counts))
    print("Kept EC labels:", len(kept_labels))
    print("min_label_count:", min_label_count)
    print("max_labels:", max_labels)
    print("Samples kept after EC label filtering:", int(keep_sample.sum()), "/", len(filtered_y))
    print("Top kept labels (training-split counts):")
    print(train_label_counts.loc[kept_labels].sort_values(ascending=False).head(20))

    return Y, mlb, keep_sample, train_label_counts


def remap_split_after_filtering(
    train_idx: np.ndarray, test_idx: np.ndarray, keep_sample: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate the raw train/test indices (computed before label
    filtering) into indices over the filtered sample population."""
    old_to_new = -np.ones(len(keep_sample), dtype=int)
    kept_positions = np.where(keep_sample)[0]
    old_to_new[kept_positions] = np.arange(len(kept_positions))

    new_train_idx = old_to_new[train_idx[keep_sample[train_idx]]]
    new_test_idx = old_to_new[test_idx[keep_sample[test_idx]]]

    n_dropped_train = int(len(train_idx) - len(new_train_idx))
    n_dropped_test = int(len(test_idx) - len(new_test_idx))
    if n_dropped_train or n_dropped_test:
        print(
            f"Dropped {n_dropped_train} train and {n_dropped_test} test samples "
            "that lost all their EC labels after vocabulary filtering."
        )

    return new_train_idx, new_test_idx


def make_feature_sets(X_reaction, X_center, feature_names):
    X_reaction_sp = to_csr_float32(X_reaction)
    X_center_sp = to_csr_float32(X_center)

    out = {}
    for name in feature_names:
        if name == "reaction_ecfp":
            out[name] = X_reaction_sp
        elif name == "reaction_center_ecfp":
            out[name] = X_center_sp
        elif name in {"both", "reaction_ecfp_plus_reaction_center_ecfp"}:
            out["reaction_ecfp_plus_reaction_center_ecfp"] = sparse.hstack(
                [X_reaction_sp, X_center_sp], format="csr", dtype=np.float32
            )
        else:
            raise ValueError(
                f"Unknown feature set {name!r}. Use reaction_ecfp,reaction_center_ecfp,both."
            )
    return out


# ======================================================================================
# Prediction helpers
# ======================================================================================


def ensure_at_least_one_label(Y_pred, Y_train=None, scores=None):
    """If a sample receives no predicted label, assign one.

    Preference order: the class with the highest score (decision_function or
    predict_proba), if available, since with class-balanced classifiers this
    is informative. If no score is available, fall back to the single most
    frequent label in the training set, matching the majority-class
    baseline, rather than an arbitrary fixed column index (the previous
    behaviour, which could silently make a classifier perform *worse* than
    the baseline on the no-prediction subset).
    """
    Y_pred = np.asarray(Y_pred, dtype=np.int8).copy()
    empty = np.where(Y_pred.sum(axis=1) == 0)[0]
    if len(empty) == 0:
        return Y_pred

    if scores is not None:
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        best = np.argmax(scores[empty], axis=1)
        Y_pred[empty, best] = 1
        return Y_pred

    if Y_train is not None:
        fallback_label = int(np.asarray(Y_train).sum(axis=0).argmax())
    else:
        fallback_label = 0
    Y_pred[empty, fallback_label] = 1
    return Y_pred


def get_scores(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            return np.vstack([p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0]) for p in proba]).T
        return np.asarray(proba)
    return None


def get_multioutput_proba_scores(model, X_test, n_labels):
    """Return an n_samples x n_labels probability-like score matrix if possible."""
    try:
        proba = model.predict_proba(X_test)
    except Exception:
        return None

    if not isinstance(proba, list):
        return np.asarray(proba)

    scores = []
    for p in proba:
        if p.shape[1] > 1:
            scores.append(p[:, 1])
        else:
            scores.append(np.zeros(p.shape[0], dtype=np.float32))

    if len(scores) != n_labels:
        return None

    return np.vstack(scores).T


# ======================================================================================
# Models
# ======================================================================================


def predict_sgd(X, Y, train_idx, test_idx, args):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    Y_train = Y[train_idx]

    # Class-balanced weights are the default: without them, this one-vs-rest
    # linear classifier was observed to collapse below the majority-class
    # baseline on highly imbalanced, high-cardinality multi-label targets
    # (e.g. EC truncation levels 3-4, 250-300 labels). Pass --sgd-unbalanced
    # to reproduce the old (not recommended) behaviour.
    class_weight = None if args.sgd_unbalanced else "balanced"

    base = SGDClassifier(
        loss=args.sgd_loss,
        penalty="l2",
        alpha=args.sgd_alpha,
        max_iter=args.sgd_max_iter,
        tol=args.sgd_tol,
        class_weight=class_weight,
        n_jobs=1,
        random_state=args.random_state,
    )

    model = make_pipeline(
        StandardScaler(with_mean=False),
        OneVsRestClassifier(base, n_jobs=args.n_jobs),
    )

    model.fit(X_train, Y_train)

    # Use decision_function scores with a configurable threshold instead of
    # predict() (which thresholds probabilities at 0.5).  With log_loss and
    # balanced class weights on high-cardinality multi-label targets (EC levels
    # 3-4, 235-300 labels), calibrated probabilities fall below 0.5 for all
    # classes, causing predict() to return all-negative rows.  The raw margin
    # (decision_function) with threshold 0.0 — the natural SVM boundary — is
    # more robust in this regime.  Lower --sgd-threshold further (e.g. -1.0)
    # to increase recall at the cost of precision.
    scores = np.asarray(model.decision_function(X_test))
    Y_pred = (scores >= args.sgd_threshold).astype(np.int8)
    return ensure_at_least_one_label(Y_pred, Y_train=Y_train, scores=scores)


def predict_extratrees_direct(X, Y, train_idx, test_idx, args):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    Y_train = Y[train_idx]

    model = ExtraTreesClassifier(
        n_estimators=args.et_n_estimators,
        max_depth=args.et_max_depth,
        min_samples_leaf=args.et_min_samples_leaf,
        max_features=args.et_max_features,
        bootstrap=False,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=0,
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_multioutput_proba_scores(model, X_test, Y.shape[1])
    return ensure_at_least_one_label(Y_pred, Y_train=Y_train, scores=scores)


def predict_randomforest_direct(X, Y, train_idx, test_idx, args):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    Y_train = Y[train_idx]

    model = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features=args.rf_max_features,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=0,
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_multioutput_proba_scores(model, X_test, Y.shape[1])
    return ensure_at_least_one_label(Y_pred, Y_train=Y_train, scores=scores)


def evaluate_dummy(Y, train_idx, test_idx):
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    label_counts = Y_train.sum(axis=0)
    best_label = int(np.argmax(label_counts))
    Y_pred = np.zeros_like(Y_test, dtype=np.int8)
    Y_pred[:, best_label] = 1
    return Y_pred


# ======================================================================================
# Evaluation and saving
# ======================================================================================


def compute_multilabel_metrics(
    feature_name,
    classifier_name,
    X,
    Y_test,
    Y_pred,
    train_idx,
    test_idx,
    args,
    radius,
    ec_level,
    n_labels,
    status="ok",
    error_message="",
):
    base = {
        "ec_level": int(ec_level),
        "radius": int(radius),
        "sample_mode": args.sample_mode,
        "result_name": f"ec{ec_level}__r{radius}__{feature_name}__{classifier_name}",
        "model": feature_name,
        "classifier": classifier_name,
        "status": status,
        "error_message": error_message,
        "n_samples": int(len(train_idx) + len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]),
        "n_labels": int(n_labels),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
    }

    if status != "ok":
        for key in [
            "subset_accuracy",
            "hamming_loss",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
            "samples_f1",
            "samples_jaccard",
        ]:
            base[key] = np.nan
        print("status:", status)
        print("error_message:", error_message)
        return base

    base.update(
        {
            "subset_accuracy": float(accuracy_score(Y_test, Y_pred)),
            "hamming_loss": float(hamming_loss(Y_test, Y_pred)),
            "micro_precision": float(precision_score(Y_test, Y_pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(Y_test, Y_pred, average="micro", zero_division=0)),
            "micro_f1": float(f1_score(Y_test, Y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(Y_test, Y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(Y_test, Y_pred, average="weighted", zero_division=0)),
            "samples_f1": float(f1_score(Y_test, Y_pred, average="samples", zero_division=0)),
            "samples_jaccard": float(jaccard_score(Y_test, Y_pred, average="samples", zero_division=0)),
        }
    )

    print("subset_accuracy:", base["subset_accuracy"])
    print("micro_f1:", base["micro_f1"])
    print("macro_f1:", base["macro_f1"])
    print("samples_jaccard:", base["samples_jaccard"])
    return base


def failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, n_labels, exc):
    error_message = f"{type(exc).__name__}: {exc}"
    return compute_multilabel_metrics(
        feature_name=feature_name,
        classifier_name=classifier_name,
        X=X,
        Y_test=Y_test,
        Y_pred=None,
        train_idx=train_idx,
        test_idx=test_idx,
        args=args,
        radius=radius,
        ec_level=ec_level,
        n_labels=n_labels,
        status="failed",
        error_message=error_message,
    )


def ordered_metrics_df(metrics_rows):
    metrics_df = pd.DataFrame(metrics_rows)
    preferred_cols = [
        "ec_level",
        "radius",
        "sample_mode",
        "result_name",
        "model",
        "classifier",
        "status",
        "n_samples",
        "n_train",
        "n_test",
        "n_features",
        "n_labels",
        "subset_accuracy",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "samples_f1",
        "samples_jaccard",
        "hamming_loss",
        "error_message",
    ]
    cols = [c for c in preferred_cols if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in cols]
    return metrics_df[cols + other_cols]


def save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    ordered_metrics_df(metrics_rows).to_csv(metrics_path, index=False)

    label_counts_path = output_dir / "ec_label_counts.csv"
    label_counts.rename("count").to_csv(label_counts_path, header=True)

    labels_path = output_dir / "ec_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(list(mlb.classes_), f, indent=2)

    split_path = output_dir / "split_indices.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_idx": [int(x) for x in train_idx],
                "test_idx": [int(x) for x in test_idx],
            },
            f,
            indent=2,
        )

    if args.save_meta:
        meta_path = output_dir / "samples_metadata.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)
        print("sample metadata:", meta_path)

    if args.save_predictions and results_by_name:
        prediction_rows = []
        for name, result in results_by_name.items():
            true_labels = mlb.inverse_transform(result["Y_test"])
            pred_labels = mlb.inverse_transform(result["Y_pred"])
            for local_i, global_i in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "result_name": name,
                        "sample_index": int(global_i),
                        "mnxr_id": meta.iloc[global_i]["mnxr_id"],
                        "true_ec_numbers": ";".join(true_labels[local_i]),
                        "pred_ec_numbers": ";".join(pred_labels[local_i]),
                    }
                )
        predictions_path = output_dir / "test_predictions.tsv"
        pd.DataFrame(prediction_rows).to_csv(predictions_path, sep="\t", index=False)
        print("predictions:", predictions_path)

    print("\nSaved outputs")
    print("=============")
    print("metrics:", metrics_path)
    print("label counts:", label_counts_path)
    print("labels:", labels_path)
    print("split:", split_path)
    print("output_dir:", output_dir)


# ======================================================================================
# One radius
# ======================================================================================


def run_one_radius(radius, ec_level, args, ec_by_mnxr):
    ecfp_params = make_ecfp_params(
        radius=radius,
        fp_size=args.fp_size,
        folded=not args.unfolded,
        custom=args.custom,
    )

    base_output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "metanetx_ec_prediction_fast_safe"
    output_dir = base_output_dir / f"ec_level_{ec_level}" / args.sample_mode / f"radius_{radius}"

    print("\n" + "=" * 100)
    print(f"Memory-safe MetaNetX EC prediction benchmark | EC level {ec_level} | radius {radius}")
    print("=" * 100)
    print("database_name:", args.database_name)
    print("ecfp_params:", ecfp_params)
    print("output_dir:", output_dir)
    print("sample_mode:", args.sample_mode)
    print("models:", args.models)
    print("feature_sets:", args.feature_sets)

    with timer("Loading ReactionRules"):
        reaction_rules = ReactionRules.load(
            database_name=args.database_name,
            ecfp_params=ecfp_params,
        )

    with timer("Building ML samples (sparse)"):
        X_reaction, X_center, y_labels, meta = expand_reactionrules_with_ec_labels(
            reaction_rules=reaction_rules,
            ec_by_mnxr=ec_by_mnxr,
            sample_mode=args.sample_mode,
            max_rules=args.max_rules,
        )

    # Free ReactionRules as soon as vectors are extracted.
    del reaction_rules
    gc.collect()

    with timer("Splitting raw indices (before label-vocabulary filtering)"):
        raw_train_idx, raw_test_idx = split_raw_indices(
            y_labels=y_labels,
            test_size=args.test_size,
            random_state=args.random_state,
            min_count_for_stratify=args.min_count_for_stratify,
        )

    with timer("Choosing EC-label vocabulary from the training split and binarizing"):
        Y_all, mlb, keep_sample, label_counts = binarize_labels_from_train_vocabulary(
            y_labels=y_labels,
            train_idx=raw_train_idx,
            min_label_count=args.min_label_count,
            max_labels=args.max_labels,
        )
        train_idx, test_idx = remap_split_after_filtering(raw_train_idx, raw_test_idx, keep_sample)

        X_reaction = X_reaction[keep_sample]
        X_center = X_center[keep_sample]
        Y = Y_all[keep_sample]
        meta = meta.loc[keep_sample].reset_index(drop=True)

    print("\nFinal dataset")
    print("=============")
    print("X_reaction:", X_reaction.shape)
    print("X_center:", X_center.shape)
    print("Y:", Y.shape)
    print("Mean labels per sample:", float(Y.sum(axis=1).mean()))
    print("n_train:", len(train_idx), "n_test:", len(test_idx))

    feature_sets = make_feature_sets(
        X_reaction=X_reaction,
        X_center=X_center,
        feature_names=parse_csv_list(args.feature_sets),
    )

    del X_reaction, X_center
    gc.collect()

    models = parse_csv_list(args.models)
    metrics_rows = []
    results_by_name = {}
    Y_test = Y[test_idx]

    with timer("Dummy baseline"):
        pred = evaluate_dummy(Y, train_idx, test_idx)
        X_ref = next(iter(feature_sets.values()))
        metrics = compute_multilabel_metrics(
            "dummy_most_frequent_ec",
            "dummy",
            X_ref,
            Y_test,
            pred,
            train_idx,
            test_idx,
            args,
            radius,
            ec_level,
            Y.shape[1],
        )
        metrics_rows.append(metrics)
        if args.save_predictions:
            results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}

    # Save baseline immediately.
    save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)

    for feature_name, X in feature_sets.items():
        print("\n" + "-" * 100)
        print("Feature set:", feature_name)
        print("X:", X.shape)
        print("-" * 100)

        if "sgd" in models:
            classifier_name = "sgd"
            try:
                with timer(f"Training {classifier_name} | {feature_name}"):
                    pred = predict_sgd(X, Y, train_idx, test_idx, args)
                metrics = compute_multilabel_metrics(
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, ec_level, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

        if "et" in models or "extratrees" in models:
            classifier_name = "extratrees_direct"
            try:
                with timer(f"Training {classifier_name} | {feature_name}"):
                    pred = predict_extratrees_direct(X, Y, train_idx, test_idx, args)
                metrics = compute_multilabel_metrics(
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, ec_level, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except MemoryError as exc:
                print("MemoryError caught. Skipping this ExtraTrees run and continuing.")
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, Y.shape[1], exc)
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

        if "rf" in models or "randomforest" in models:
            classifier_name = "randomforest_direct"
            try:
                with timer(f"Training {classifier_name} | {feature_name}"):
                    pred = predict_randomforest_direct(X, Y, train_idx, test_idx, args)
                metrics = compute_multilabel_metrics(
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, ec_level, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except MemoryError as exc:
                print("MemoryError caught. Skipping this RandomForest run and continuing.")
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, Y.shape[1], exc)
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, ec_level, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

    metrics_df = ordered_metrics_df(metrics_rows)

    print("\n" + f"Comparison | EC level {ec_level} | radius {radius}")
    print("=" * (35 + len(str(ec_level)) + len(str(radius))))
    display_cols = [
        c
        for c in [
            "ec_level",
            "radius",
            "model",
            "classifier",
            "status",
            "n_samples",
            "n_features",
            "n_labels",
            "subset_accuracy",
            "micro_f1",
            "macro_f1",
            "samples_jaccard",
            "error_message",
        ]
        if c in metrics_df.columns
    ]
    print(metrics_df[display_cols])

    return metrics_df


# ======================================================================================
# CLI
# ======================================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Memory-safe EC-number prediction from reaction ECFP and reaction-center ECFP features."
    )

    # NOTE on defaults: this script now runs the FULL benchmark with NO
    # arguments at all (all radii, no rule cap, sgd+et). This is meant for a
    # cluster job, not a laptop -- see the module docstring for a lighter,
    # explicit "local sanity check" invocation if you need one.
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--radii", default="0,1,2,3,4,5")
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--database-name", default="metanetx")
    parser.add_argument("--reac-prop-path", default=DEFAULT_REAC_PROP)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--max-rules",
        type=int,
        default=-1,
        help="Cap the number of ReactionRules entries read. Default is -1 "
        "(no cap, full dataset). Pass a positive value (e.g. 5000) for a "
        "lighter, faster local sanity check.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for OvR/forest training. joblib's loky backend "
        "duplicates data per worker, so memory usage scales with --n-jobs, "
        "not just speed. Lower to 1 on memory-constrained machines.",
    )

    parser.add_argument(
        "--sample-mode",
        choices=["unique_rules", "occurrences"],
        default="unique_rules",
    )
    parser.add_argument("--unfolded", action="store_true")
    parser.add_argument("--custom", action="store_true")

    parser.add_argument(
        "--ec-levels",
        default="1,2,3,4",
        help="Comma-separated list of EC truncation levels to loop over "
        "(each in {1,2,3,4}), e.g. '1,2,3,4' or '3'. Level 3 alone is a good "
        "compromise for a quick check; the default loops over all four "
        "levels and writes a single combined summary CSV, mirroring how "
        "--radii is handled.",
    )
    parser.add_argument(
        "--ec-level",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Deprecated single-value alias for --ec-levels (kept for "
        "backward compatibility). If set, it overrides --ec-levels.",
    )
    parser.add_argument("--min-label-count", type=int, default=5)
    parser.add_argument(
        "--max-labels",
        type=int,
        default=300,
        help="Keep only the most frequent labels (by training-split count) after min-label-count filtering.",
    )
    parser.add_argument(
        "--min-count-for-stratify",
        type=int,
        default=2,
        help="Primary EC labels (for stratification purposes only) with fewer "
        "occurrences than this are grouped into a single bucket instead of "
        "disabling stratification entirely.",
    )

    parser.add_argument(
        "--feature-sets",
        default="reaction_ecfp,reaction_center_ecfp,both",
        help="Comma-separated subset of: reaction_ecfp,reaction_center_ecfp,both.",
    )
    parser.add_argument(
        "--models",
        default="sgd,et",
        help="Comma-separated subset of: sgd, et, rf.",
    )

    parser.add_argument("--sgd-loss", default="log_loss", choices=["log_loss", "modified_huber", "hinge"])
    parser.add_argument("--sgd-alpha", type=float, default=1e-4)
    parser.add_argument("--sgd-max-iter", type=int, default=300)
    parser.add_argument("--sgd-tol", type=float, default=1e-3)
    parser.add_argument(
        "--sgd-threshold",
        type=float,
        default=0.0,
        help="Decision-function threshold for SGD label prediction. "
        "Predictions use decision_function() >= threshold instead of "
        "predict() (which applies 0.5 on calibrated probabilities and "
        "collapses to all-negative on rare, high-cardinality EC levels). "
        "0.0 is the natural SVM margin boundary. Lower values (e.g. -1.0) "
        "increase recall at the cost of precision.",
    )
    parser.add_argument(
        "--sgd-unbalanced",
        action="store_true",
        help="Disable class-balanced weights for SGD. NOT recommended for "
        "high-cardinality multi-label EC prediction: without balancing, the "
        "linear one-vs-rest classifier can become unstable and underperform "
        "the majority-class baseline at fine EC granularity (levels 3-4).",
    )

    parser.add_argument("--et-n-estimators", type=int, default=100)
    parser.add_argument("--et-max-depth", type=int, default=25)
    parser.add_argument("--et-min-samples-leaf", type=int, default=2)
    parser.add_argument("--et-max-features", default="sqrt")

    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=25)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", default="sqrt")

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    radii = parse_radii(args.radii, args.radius)
    # --ec-level (singular) is a deprecated alias kept for backward
    # compatibility: if set, it overrides --ec-levels entirely.
    if args.ec_level is not None:
        ec_levels = [int(args.ec_level)]
    else:
        ec_levels = parse_ec_levels(args.ec_levels, fallback_ec_level=3)

    base_output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "metanetx_ec_prediction_fast_safe"
    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else base_output_dir / args.sample_mode / "metrics_all_ec_levels_all_radii.csv"
    )

    print("Memory-safe MetaNetX EC prediction benchmark across EC levels and radii")
    print("========================================================================")
    print("ec_levels:", ec_levels)
    print("radii:", radii)
    print("database_name:", args.database_name)
    print("reac_prop_path:", args.reac_prop_path)
    print("min_label_count:", args.min_label_count)
    print("max_labels:", args.max_labels)
    print("sample_mode:", args.sample_mode)
    print("feature_sets:", args.feature_sets)
    print("models:", args.models)
    print("n_jobs:", args.n_jobs)
    print("sgd_unbalanced:", args.sgd_unbalanced)
    print("base_output_dir:", base_output_dir)
    print("summary_output (single global file):", summary_output)

    all_metrics = []
    for ec_level in ec_levels:
        with timer(f"Loading EC labels | EC level {ec_level}"):
            ec_by_mnxr = load_metanetx_ec_labels(
                reac_prop_path=Path(args.reac_prop_path),
                ec_level=ec_level,
            )

        for radius in radii:
            try:
                metrics_df = run_one_radius(radius=radius, ec_level=ec_level, args=args, ec_by_mnxr=ec_by_mnxr)
                all_metrics.append(metrics_df)
            except Exception as exc:
                print("\n" + "!" * 100)
                print(f"EC level {ec_level} / radius {radius} failed completely: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
                print("Continuing with next (EC level, radius) combination.")
                print("!" * 100)

            # Write the combined summary after every (EC level, radius), so a
            # partial run (e.g. killed by a time/memory limit) still leaves a
            # single, up-to-date global CSV rather than nothing at all.
            if all_metrics:
                summary_output.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(all_metrics, ignore_index=True).to_csv(summary_output, index=False)

        del ec_by_mnxr
        gc.collect()

    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(summary_output, index=False)

        print("\n" + "=" * 100)
        print("All available (EC level, radius) combinations done")
        print("=" * 100)
        print("summary_output (single global file):", summary_output)

        cols = [
            c
            for c in [
                "ec_level",
                "radius",
                "sample_mode",
                "result_name",
                "model",
                "classifier",
                "status",
                "n_samples",
                "n_features",
                "n_labels",
                "subset_accuracy",
                "micro_f1",
                "macro_f1",
                "samples_jaccard",
                "error_message",
            ]
            if c in all_metrics_df.columns
        ]
        print(all_metrics_df[cols])
    else:
        raise RuntimeError("No (EC level, radius) combination completed successfully.")


if __name__ == "__main__":
    main()
