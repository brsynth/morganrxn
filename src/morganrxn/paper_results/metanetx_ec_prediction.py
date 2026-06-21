#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-safe MetaNetX EC-number prediction benchmark from reaction ECFP features.

This script loads MetaNetX EC annotations from reac_prop.tsv, matches them to
ReactionRules entries, and compares reaction ECFP, reaction-center ECFP, and
optionally both concatenated as inputs for EC-number prediction.

Compared with the previous fast version, this version is safer on RAM:
    - EC labels are coarsened to level 3 by default;
    - the number of labels is limited by default;
    - ExtraTrees has conservative defaults: fewer trees, limited depth, larger leaves;
    - sparse matrices are used whenever possible;
    - MemoryError during model fitting does not kill the whole benchmark;
    - metrics are saved after each radius;
    - models can be selected independently.

Recommended run:
    python metanetx_ec_prediction_fast_safe.py \
        --radii 0,1,2,3,4,5 \
        --ec-level 3 \
        --max-labels 300 \
        --models sgd,et \
        --feature-sets reaction_ecfp,reaction_center_ecfp,both

Very safe first run:
    python metanetx_ec_prediction_fast_safe.py \
        --radii 0,1,2,3,4,5 \
        --ec-level 3 \
        --max-labels 200 \
        --models sgd \
        --feature-sets reaction_ecfp,reaction_center_ecfp

Fuller, but heavier run:
    python metanetx_ec_prediction_fast_safe.py \
        --radii 0,1,2,3,4,5 \
        --ec-level 4 \
        --max-labels 1000 \
        --models sgd,et
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
    if sparse.issparse(X):
        return X.astype(np.float32).tocsr()
    return sparse.csr_matrix(np.asarray(X, dtype=np.float32))


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
# ReactionRules -> ML samples
# ======================================================================================


def expand_unique_rules(reaction_rules, ec_by_mnxr, max_rules=None):
    X_reaction, X_center, y_labels, meta_rows = [], [], [], []

    n_rules = len(reaction_rules)
    if max_rules is not None:
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

        X_reaction.append(reaction_rules.ecfp_reaction[i])
        X_center.append(reaction_rules.ecfp_reaction_center[i])
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

    return (
        np.asarray(X_reaction, dtype=np.float32),
        np.asarray(X_center, dtype=np.float32),
        y_labels,
        pd.DataFrame(meta_rows),
    )


def expand_occurrences(reaction_rules, ec_by_mnxr, max_rules=None):
    X_reaction, X_center, y_labels, meta_rows = [], [], [], []

    n_rules = len(reaction_rules)
    if max_rules is not None:
        n_rules = min(n_rules, int(max_rules))

    n_source_ids = 0
    n_bad_ids = 0
    n_no_ec = 0

    for i in range(n_rules):
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

            X_reaction.append(reaction_rules.ecfp_reaction[i])
            X_center.append(reaction_rules.ecfp_reaction_center[i])
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

    if not y_labels:
        raise ValueError("No labelled samples recovered.")

    return (
        np.asarray(X_reaction, dtype=np.float32),
        np.asarray(X_center, dtype=np.float32),
        y_labels,
        pd.DataFrame(meta_rows),
    )


def expand_reactionrules_with_ec_labels(reaction_rules, ec_by_mnxr, sample_mode, max_rules=None):
    if sample_mode == "unique_rules":
        return expand_unique_rules(reaction_rules, ec_by_mnxr, max_rules=max_rules)
    if sample_mode == "occurrences":
        return expand_occurrences(reaction_rules, ec_by_mnxr, max_rules=max_rules)
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


# ======================================================================================
# Labels, features and split
# ======================================================================================


def binarize_and_filter_labels(y_labels, min_label_count=5, max_labels=300):
    counts = pd.Series([ec for ecs in y_labels for ec in ecs]).value_counts()

    kept = counts[counts >= int(min_label_count)]
    if max_labels is not None:
        kept = kept.head(int(max_labels))

    kept_labels = sorted(kept.index.tolist())
    kept_set = set(kept_labels)

    filtered_y = [tuple(ec for ec in ecs if ec in kept_set) for ecs in y_labels]
    keep_sample = np.asarray([len(ecs) > 0 for ecs in filtered_y], dtype=bool)
    filtered_y = [ecs for ecs in filtered_y if len(ecs) > 0]

    if len(kept_labels) == 0:
        raise ValueError("No EC labels kept. Lower --min-label-count or increase --max-labels.")
    if len(filtered_y) == 0:
        raise ValueError("No samples left after EC label filtering.")

    mlb = MultiLabelBinarizer(classes=kept_labels)
    Y = mlb.fit_transform(filtered_y).astype(np.int8)

    print("\nLabel filtering")
    print("===============")
    print("Original unique EC labels:", len(counts))
    print("Kept EC labels:", len(kept_labels))
    print("min_label_count:", min_label_count)
    print("max_labels:", max_labels)
    print("Samples after label filtering:", len(filtered_y))
    print("Top kept labels:")
    print(counts.loc[kept_labels].sort_values(ascending=False).head(20))

    return Y, mlb, keep_sample, counts


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


def make_shared_split_indices(Y, test_size, random_state):
    indices = np.arange(Y.shape[0])

    # Stratification is only approximate for multilabel data.
    # We stratify on the first active label if possible.
    primary = np.argmax(Y, axis=1)
    primary_counts = pd.Series(primary).value_counts()
    stratify = primary if primary_counts.min() >= 2 else None

    if stratify is None:
        print("WARNING: non-stratified split used because some primary labels are rare.")

    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


# ======================================================================================
# Prediction helpers
# ======================================================================================


def ensure_at_least_one_label(Y_pred, scores=None):
    Y_pred = np.asarray(Y_pred, dtype=np.int8)
    empty = np.where(Y_pred.sum(axis=1) == 0)[0]
    if len(empty) == 0:
        return Y_pred

    if scores is None:
        Y_pred[empty, 0] = 1
        return Y_pred

    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    best = np.argmax(scores[empty], axis=1)
    Y_pred[empty, best] = 1
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

    base = SGDClassifier(
        loss=args.sgd_loss,
        penalty="l2",
        alpha=args.sgd_alpha,
        max_iter=args.sgd_max_iter,
        tol=args.sgd_tol,
        class_weight="balanced" if args.sgd_balanced else None,
        n_jobs=1,
        random_state=args.random_state,
    )

    model = make_pipeline(
        StandardScaler(with_mean=False),
        OneVsRestClassifier(base, n_jobs=args.n_jobs),
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    return ensure_at_least_one_label(Y_pred, scores)


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
    return ensure_at_least_one_label(Y_pred, scores)


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
    return ensure_at_least_one_label(Y_pred, scores)


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
    n_labels,
    status="ok",
    error_message="",
):
    base = {
        "radius": int(radius),
        "sample_mode": args.sample_mode,
        "result_name": f"r{radius}__{feature_name}__{classifier_name}",
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


def failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, n_labels, exc):
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
        n_labels=n_labels,
        status="failed",
        error_message=error_message,
    )


def ordered_metrics_df(metrics_rows):
    metrics_df = pd.DataFrame(metrics_rows)
    preferred_cols = [
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


def run_one_radius(radius, args, ec_by_mnxr):
    ecfp_params = make_ecfp_params(
        radius=radius,
        fp_size=args.fp_size,
        folded=not args.unfolded,
        custom=args.custom,
    )

    base_output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "metanetx_ec_prediction_fast_safe"
    output_dir = base_output_dir / args.sample_mode / f"radius_{radius}"

    print("\n" + "=" * 100)
    print(f"Memory-safe MetaNetX EC prediction benchmark | radius {radius}")
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

    with timer("Building ML samples"):
        X_reaction, X_center, y_labels, meta = expand_reactionrules_with_ec_labels(
            reaction_rules=reaction_rules,
            ec_by_mnxr=ec_by_mnxr,
            sample_mode=args.sample_mode,
            max_rules=args.max_rules,
        )

    # Free ReactionRules as soon as vectors are extracted.
    del reaction_rules
    gc.collect()

    with timer("Binarizing labels"):
        Y, mlb, keep_sample, label_counts = binarize_and_filter_labels(
            y_labels=y_labels,
            min_label_count=args.min_label_count,
            max_labels=args.max_labels,
        )
        X_reaction = X_reaction[keep_sample]
        X_center = X_center[keep_sample]
        meta = meta.loc[keep_sample].reset_index(drop=True)

    print("\nFinal dataset")
    print("=============")
    print("X_reaction:", X_reaction.shape)
    print("X_center:", X_center.shape)
    print("Y:", Y.shape)
    print("Mean labels per sample:", float(Y.sum(axis=1).mean()))

    feature_sets = make_feature_sets(
        X_reaction=X_reaction,
        X_center=X_center,
        feature_names=parse_csv_list(args.feature_sets),
    )

    del X_reaction, X_center
    gc.collect()

    train_idx, test_idx = make_shared_split_indices(
        Y=Y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nShared train/test split")
    print("=======================")
    print("n_train:", len(train_idx))
    print("n_test:", len(test_idx))

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
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

        if "et" in models or "extratrees" in models:
            classifier_name = "extratrees_direct"
            try:
                with timer(f"Training {classifier_name} | {feature_name}"):
                    pred = predict_extratrees_direct(X, Y, train_idx, test_idx, args)
                metrics = compute_multilabel_metrics(
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except MemoryError as exc:
                print("MemoryError caught. Skipping this ExtraTrees run and continuing.")
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, Y.shape[1], exc)
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

        if "rf" in models or "randomforest" in models:
            classifier_name = "randomforest_direct"
            try:
                with timer(f"Training {classifier_name} | {feature_name}"):
                    pred = predict_randomforest_direct(X, Y, train_idx, test_idx, args)
                metrics = compute_multilabel_metrics(
                    feature_name, classifier_name, X, Y_test, pred, train_idx, test_idx, args, radius, Y.shape[1]
                )
                if args.save_predictions:
                    results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}
            except MemoryError as exc:
                print("MemoryError caught. Skipping this RandomForest run and continuing.")
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, Y.shape[1], exc)
            except Exception as exc:
                traceback.print_exc(limit=2)
                metrics = failed_metrics(feature_name, classifier_name, X, Y_test, train_idx, test_idx, args, radius, Y.shape[1], exc)
            metrics_rows.append(metrics)
            save_outputs(output_dir, metrics_rows, meta, label_counts, mlb, train_idx, test_idx, results_by_name, args)
            gc.collect()

    metrics_df = ordered_metrics_df(metrics_rows)

    print("\n" + f"Comparison | radius {radius}")
    print("=" * (20 + len(str(radius))))
    display_cols = [
        c
        for c in [
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

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--radii", default="0,1,2,3,4,5")
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--database-name", default="metanetx")
    parser.add_argument("--reac-prop-path", default=DEFAULT_REAC_PROP)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)

    parser.add_argument(
        "--sample-mode",
        choices=["unique_rules", "occurrences"],
        default="unique_rules",
    )
    parser.add_argument("--unfolded", action="store_true")
    parser.add_argument("--custom", action="store_true")

    parser.add_argument(
        "--ec-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="EC truncation level. Level 3 is a good compromise for speed and biological detail.",
    )
    parser.add_argument("--min-label-count", type=int, default=5)
    parser.add_argument(
        "--max-labels",
        type=int,
        default=300,
        help="Keep only the most frequent labels after min-label-count filtering. Use None only for very large RAM.",
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
    parser.add_argument("--sgd-balanced", action="store_true", help="Use class_weight='balanced' for SGD.")

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
    base_output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "metanetx_ec_prediction_fast_safe"
    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else base_output_dir / args.sample_mode / "metrics_all_radii.csv"
    )

    print("Memory-safe MetaNetX EC prediction benchmark across radii")
    print("========================================================")
    print("radii:", radii)
    print("database_name:", args.database_name)
    print("reac_prop_path:", args.reac_prop_path)
    print("ec_level:", args.ec_level)
    print("min_label_count:", args.min_label_count)
    print("max_labels:", args.max_labels)
    print("sample_mode:", args.sample_mode)
    print("feature_sets:", args.feature_sets)
    print("models:", args.models)
    print("n_jobs:", args.n_jobs)
    print("base_output_dir:", base_output_dir)
    print("summary_output:", summary_output)

    with timer("Loading EC labels"):
        ec_by_mnxr = load_metanetx_ec_labels(
            reac_prop_path=Path(args.reac_prop_path),
            ec_level=args.ec_level,
        )

    all_metrics = []
    for radius in radii:
        try:
            metrics_df = run_one_radius(radius=radius, args=args, ec_by_mnxr=ec_by_mnxr)
            all_metrics.append(metrics_df)
        except Exception as exc:
            print("\n" + "!" * 100)
            print(f"Radius {radius} failed completely: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            print("Continuing with next radius.")
            print("!" * 100)

    if all_metrics:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(summary_output, index=False)

        print("\n" + "=" * 100)
        print("All available radii done")
        print("=" * 100)
        print("summary_output:", summary_output)

        cols = [
            c
            for c in [
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
        raise RuntimeError("No radius completed successfully.")


if __name__ == "__main__":
    main()
