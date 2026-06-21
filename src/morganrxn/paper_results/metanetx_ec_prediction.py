#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict MetaNetX EC numbers from reaction ECFP representations.

The script:
    - loads MetaNetX ReactionRules for one or several ECFP radii,
    - reads EC numbers from DATA_DIR/metanetx/reac_prop.tsv,
    - associates MNXR reaction IDs to split ReactionRules entries,
    - builds three feature sets:
        1. reaction ECFP
        2. reaction-center ECFP
        3. reaction ECFP + reaction-center ECFP
    - trains multi-label classifiers to predict EC numbers,
    - saves metrics and predictions.

Expected reac_prop.tsv columns:
    #ID    mnx_equation    reference    classifs    is_balanced    is_transport

Notes
-----
MetaNetX `classifs` can contain zero, one, or several EC numbers separated by ';'.
This is therefore treated as a multi-label prediction problem.

By default, labels occurring fewer than --min-label-count times are removed.
Samples with no remaining EC label are removed from the ML dataset.

Examples
--------
Run radii 0 to 5:

    python metanetx_ec_prediction.py --radii 0,1,2,3,4,5

Run radius 2 only:

    python metanetx_ec_prediction.py --radius 2

Use only frequent EC labels:

    python metanetx_ec_prediction.py --radii 0,1,2,3,4,5 --min-label-count 20

Limit to the 100 most frequent EC labels:

    python metanetx_ec_prediction.py --max-labels 100

Run only logistic regression and random forest:

    python metanetx_ec_prediction.py --models lr,rf
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from morganrxn.core.paths import DATA_DIR, RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules


warnings.filterwarnings("ignore", category=ConvergenceWarning)

METANETX_DIR = DATA_DIR / "metanetx"
DEFAULT_REAC_PROP = METANETX_DIR / "reac_prop.tsv"


# ======================================================================================
# General helpers
# ======================================================================================


def parse_radii(radii_value, fallback_radius):
    if radii_value is None or str(radii_value).strip() == "":
        return [int(fallback_radius)]

    radii = [int(x.strip()) for x in str(radii_value).split(",") if x.strip()]
    radii = sorted(set(radii))

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


def parse_models(models_value):
    aliases = {
        "lr": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "logreg": "logistic_regression",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "gb": "gradient_boosting",
        "gradient_boosting": "gradient_boosting",
        "gradientboosting": "gradient_boosting",
        "mlp": "mlp",
    }

    raw = [x.strip().lower() for x in str(models_value).split(",") if x.strip()]
    unknown = [x for x in raw if x not in aliases]

    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Allowed models: {sorted(aliases)}"
        )

    out = []
    for x in raw:
        canonical = aliases[x]
        if canonical not in out:
            out.append(canonical)

    return out


def parse_mlp_hidden_layers(value):
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def split_merged_ids(value):
    if value is None:
        return []

    value = str(value).strip()
    if value == "":
        return []

    return [x.strip() for x in value.split("|") if x.strip()]


def extract_mnxr_id(value):
    """
    Extract the core MetaNetX reaction ID from a ReactionRules ID.

    Examples
    --------
    MNXR100024              -> MNXR100024
    MNXR100024_L2R_r1       -> MNXR100024
    MNXR100024_R2L_r2       -> MNXR100024
    MNXR100024__split0      -> MNXR100024
    metanetx:MNXR100024...  -> MNXR100024
    """
    if value is None:
        return None

    value = str(value).strip()

    match = re.search(r"(MNXR[0-9]+)", value)

    if match is None:
        return None

    return match.group(1)


# ======================================================================================
# EC loading
# ======================================================================================


def parse_ec_list(value, ec_level=None):
    """
    Parse MetaNetX classifs field.

    Examples:
        "6.3.1.2" -> ["6.3.1.2"]
        "1.4.1.13;1.4.1.14" -> ["1.4.1.13", "1.4.1.14"]

    ec_level can be:
        None -> keep exact EC strings as provided
        1    -> keep first level, e.g. 1
        2    -> keep first two levels, e.g. 1.4
        3    -> keep first three levels, e.g. 1.4.1
        4    -> keep full EC numbers when available
    """
    if value is None or pd.isna(value):
        return []

    tokens = []
    for token in str(value).split(";"):
        token = token.strip()
        if token == "" or token.upper() in {"B", "T"}:
            continue

        # Accept partial EC numbers such as 1.2.1 and full EC numbers such as 1.2.1.3.
        if not re.match(r"^[0-9]+(\.[0-9-]+){0,3}$", token):
            continue

        if ec_level is not None:
            parts = token.split(".")
            if len(parts) < int(ec_level):
                continue
            token = ".".join(parts[: int(ec_level)])

        tokens.append(token)

    return sorted(set(tokens))


def load_metanetx_ec_labels(reac_prop_path: Path, ec_level=None):
    print("Loading MetaNetX EC annotations...")
    print("reac_prop_path:", reac_prop_path)

    df = pd.read_csv(
        reac_prop_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "mnxr_id",
            "mnx_equation",
            "reference",
            "classifs",
            "is_balanced",
            "is_transport",
        ],
        dtype=str,
        keep_default_na=False,
    )

    df["ec_numbers"] = df["classifs"].apply(
        lambda x: parse_ec_list(x, ec_level=ec_level)
    )

    ec_by_mnxr = {
        row.mnxr_id: row.ec_numbers
        for row in df.itertuples(index=False)
        if row.ec_numbers
    }

    print("reac_prop rows:", len(df))
    print("reactions with EC annotation:", len(ec_by_mnxr))

    all_ec = [ec for ecs in ec_by_mnxr.values() for ec in ecs]
    print("unique EC labels:", len(set(all_ec)))
    print("top EC labels:")
    print(pd.Series(all_ec).value_counts().head(20))

    return df, ec_by_mnxr


# ======================================================================================
# ReactionRules expansion
# ======================================================================================


def get_rule_source_ids(reaction_rules, i):
    ids = []

    if hasattr(reaction_rules, "reaction_monocomp_id"):
        ids.extend(split_merged_ids(reaction_rules.reaction_monocomp_id[i]))

    if hasattr(reaction_rules, "reaction_id"):
        ids.extend(split_merged_ids(reaction_rules.reaction_id[i]))

    if not ids:
        ids = [str(i)]

    # Preserve order while removing duplicates.
    out = []
    for x in ids:
        if x not in out:
            out.append(x)

    return out


def expand_reactionrules_with_ec_labels_occurrences(
    reaction_rules,
    ec_by_mnxr,
    max_rules=None,
):
    """
    Build one ML sample per original labelled source ID.

    This is the occurrence-weighted mode. If a deduplicated ReactionRules entry
    contains several IDs separated by ``|``, each source ID is treated as one
    distinct sample and receives the EC labels of its parent MNXR reaction.

    This mode can be useful as a robustness / supplementary analysis, but it
    re-weights frequent duplicated rules.
    """
    X_reaction = []
    X_center = []
    y_labels = []
    meta_rows = []

    n_rules = len(reaction_rules)
    if max_rules is not None:
        n_rules = min(n_rules, int(max_rules))

    n_source_ids = 0
    n_bad_ids = 0
    n_no_ec = 0

    for i in range(n_rules):
        source_ids = get_rule_source_ids(reaction_rules, i)

        for source_id in source_ids:
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
                    "mnxr_id": mnxr_id,
                    "mnxr_ids": mnxr_id,
                    "ec_numbers": ";".join(ec_numbers),
                    "n_ec_numbers": len(ec_numbers),
                    "n_source_ids": 1,
                    "n_labelled_source_ids": 1,
                    "template_reaction": reaction_rules.template_reaction[i],
                }
            )

    print()
    print("Label matching summary")
    print("======================")
    print("Sample mode: occurrences")
    print("ReactionRules entries read:", n_rules)
    print("Source IDs seen:", n_source_ids)
    print("Bad MNXR IDs:", n_bad_ids)
    print("Source IDs without EC annotation:", n_no_ec)
    print("Final labelled occurrence samples before label filtering:", len(y_labels))

    if len(y_labels) == 0:
        raise ValueError(
            "No labelled samples recovered. Check that ReactionRules IDs contain "
            "MNXR identifiers matching reac_prop.tsv."
        )

    return (
        np.asarray(X_reaction, dtype=np.float32),
        np.asarray(X_center, dtype=np.float32),
        y_labels,
        pd.DataFrame(meta_rows),
    )


def expand_reactionrules_with_ec_labels_unique_rules(
    reaction_rules,
    ec_by_mnxr,
    max_rules=None,
):
    """
    Build one ML sample per unique ReactionRules entry.

    Duplicated source reactions have already been collapsed by ReactionRules.
    Their source IDs are kept as ``|``-separated values. Here, all EC numbers
    associated with these source IDs are aggregated into a single multi-label
    target for the unique rule.

    This is the recommended mode for representation benchmarking because each
    unique reaction vector contributes only once.
    """
    X_reaction = []
    X_center = []
    y_labels = []
    meta_rows = []

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
                "template_reaction": reaction_rules.template_reaction[i],
            }
        )

    print()
    print("Label matching summary")
    print("======================")
    print("Sample mode: unique_rules")
    print("ReactionRules entries read:", n_rules)
    print("Source IDs seen:", n_source_ids)
    print("Bad MNXR IDs:", n_bad_ids)
    print("Labelled source IDs:", n_labelled_source_ids)
    print("Rules without EC annotation:", n_rules_without_ec)
    print("Final labelled unique-rule samples before label filtering:", len(y_labels))

    if len(y_labels) == 0:
        raise ValueError(
            "No labelled samples recovered. Check that ReactionRules IDs contain "
            "MNXR identifiers matching reac_prop.tsv."
        )

    return (
        np.asarray(X_reaction, dtype=np.float32),
        np.asarray(X_center, dtype=np.float32),
        y_labels,
        pd.DataFrame(meta_rows),
    )


def expand_reactionrules_with_ec_labels(
    reaction_rules,
    ec_by_mnxr,
    max_rules=None,
    sample_mode="unique_rules",
):
    """
    Dispatch ReactionRules-to-ML-samples expansion.

    Parameters
    ----------
    sample_mode : {"unique_rules", "occurrences"}
        - unique_rules: one sample per unique ReactionRules entry.
        - occurrences: one sample per labelled source ID.
    """
    if sample_mode == "unique_rules":
        return expand_reactionrules_with_ec_labels_unique_rules(
            reaction_rules=reaction_rules,
            ec_by_mnxr=ec_by_mnxr,
            max_rules=max_rules,
        )

    if sample_mode == "occurrences":
        return expand_reactionrules_with_ec_labels_occurrences(
            reaction_rules=reaction_rules,
            ec_by_mnxr=ec_by_mnxr,
            max_rules=max_rules,
        )

    raise ValueError(
        f"Unknown sample_mode={sample_mode!r}. "
        "Expected 'unique_rules' or 'occurrences'."
    )


def binarize_and_filter_labels(y_labels, min_label_count=5, max_labels=None):
    """
    Keep labels frequent enough, then remove samples with no remaining labels.
    """
    counts = pd.Series([ec for ecs in y_labels for ec in ecs]).value_counts()

    kept = counts[counts >= int(min_label_count)]

    if max_labels is not None:
        kept = kept.head(int(max_labels))

    kept_labels = sorted(kept.index.tolist())
    kept_set = set(kept_labels)

    filtered_y_labels = [tuple(ec for ec in ecs if ec in kept_set) for ecs in y_labels]
    keep_sample = np.asarray([len(ecs) > 0 for ecs in filtered_y_labels], dtype=bool)
    filtered_y_labels = [ecs for ecs in filtered_y_labels if len(ecs) > 0]

    mlb = MultiLabelBinarizer(classes=kept_labels)
    Y = mlb.fit_transform(filtered_y_labels).astype(np.int32)

    print()
    print("Label filtering")
    print("===============")
    print("Original unique EC labels:", len(counts))
    print("Kept EC labels:", len(kept_labels))
    print("min_label_count:", min_label_count)
    print("max_labels:", max_labels)
    print("Samples after label filtering:", len(filtered_y_labels))
    print("Top kept labels:")
    print(counts.loc[kept_labels].sort_values(ascending=False).head(20))

    if len(kept_labels) == 0:
        raise ValueError(
            "No EC labels kept. Lower --min-label-count or increase --max-labels."
        )

    if len(filtered_y_labels) == 0:
        raise ValueError(
            "No samples left after EC label filtering. Lower --min-label-count."
        )

    return Y, mlb, keep_sample, counts


# ======================================================================================
# Feature sets and splits
# ======================================================================================


def make_feature_sets(X_reaction, X_center, use_sparse=True):
    if use_sparse:
        X_reaction_sp = sparse.csr_matrix(X_reaction)
        X_center_sp = sparse.csr_matrix(X_center)
        X_both_sp = sparse.hstack([X_reaction_sp, X_center_sp], format="csr")

        return {
            "reaction_ecfp": X_reaction_sp,
            "reaction_center_ecfp": X_center_sp,
            "reaction_ecfp_plus_reaction_center_ecfp": X_both_sp,
        }

    X_both = np.hstack([X_reaction, X_center]).astype(np.float32)
    return {
        "reaction_ecfp": X_reaction,
        "reaction_center_ecfp": X_center,
        "reaction_ecfp_plus_reaction_center_ecfp": X_both,
    }


def make_shared_split_indices(Y, test_size, random_state):
    indices = np.arange(Y.shape[0])

    # Stratify on the primary EC label when possible.
    primary = np.argmax(Y, axis=1)
    primary_counts = pd.Series(primary).value_counts()
    stratify = primary if primary_counts.min() >= 2 else None

    if stratify is None:
        print("WARNING: non-stratified split used because some primary labels are rare.")

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return train_idx, test_idx


def subset_X(X, indices):
    return X[indices]


def to_dense_float32(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# ======================================================================================
# Metrics and thresholding
# ======================================================================================


def ensure_at_least_one_label(Y_pred, scores):
    """
    Multi-label models may predict no label for a sample.
    In that case, set the highest-scoring label to 1.
    """
    Y_pred = np.asarray(Y_pred, dtype=np.int32)

    empty = np.where(Y_pred.sum(axis=1) == 0)[0]
    if len(empty) == 0:
        return Y_pred

    best = np.argmax(scores[empty], axis=1)
    Y_pred[empty, best] = 1
    return Y_pred


def get_scores(model, X):
    """
    Return label scores for fallback top-label assignment.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # OneVsRest returns an array for multilabel; pipelines usually forward it.
        if isinstance(proba, list):
            proba = np.vstack([p[:, 1] for p in proba]).T

        return np.asarray(proba)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))

    pred = np.asarray(model.predict(X))
    return pred.astype(float)


def compute_multilabel_metrics(
    feature_name,
    classifier_name,
    X,
    Y_test,
    Y_pred,
    train_idx,
    test_idx,
    test_size,
    random_state,
    radius,
    n_labels,
):
    metrics = {
        "radius": int(radius),
        "result_name": f"r{radius}__{feature_name}__{classifier_name}",
        "model": feature_name,
        "classifier": classifier_name,
        "n_samples": int(len(train_idx) + len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]),
        "n_labels": int(n_labels),
        "test_size": float(test_size),
        "random_state": int(random_state),
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

    print("subset_accuracy:", metrics["subset_accuracy"])
    print("micro_f1:", metrics["micro_f1"])
    print("macro_f1:", metrics["macro_f1"])
    print("samples_jaccard:", metrics["samples_jaccard"])

    return metrics


# ======================================================================================
# Models
# ======================================================================================


def train_predict_logistic_regression(X, Y, train_idx, test_idx, max_iter, random_state):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    Y_train = Y[train_idx]

    base = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state,
    )

    model = make_pipeline(
        StandardScaler(with_mean=False),
        OneVsRestClassifier(base, n_jobs=-1),
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    Y_pred = ensure_at_least_one_label(Y_pred, scores)

    return Y_pred


def train_predict_random_forest(
    X,
    Y,
    train_idx,
    test_idx,
    n_estimators,
    max_depth,
    min_samples_leaf,
    random_state,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    Y_train = Y[train_idx]

    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    model = OneVsRestClassifier(base, n_jobs=-1)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    Y_pred = ensure_at_least_one_label(Y_pred, scores)

    return Y_pred


def train_predict_gradient_boosting(
    X,
    Y,
    train_idx,
    test_idx,
    n_estimators,
    learning_rate,
    max_depth,
    random_state,
):
    X_train = to_dense_float32(subset_X(X, train_idx))
    X_test = to_dense_float32(subset_X(X, test_idx))
    Y_train = Y[train_idx]

    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    model = OneVsRestClassifier(base, n_jobs=-1)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    Y_pred = ensure_at_least_one_label(Y_pred, scores)

    return Y_pred


def train_predict_mlp(X, Y, train_idx, test_idx, hidden_layer_sizes, max_iter, random_state):
    X_train = to_dense_float32(subset_X(X, train_idx))
    X_test = to_dense_float32(subset_X(X, test_idx))
    Y_train = Y[train_idx]

    model = make_pipeline(
        StandardScaler(with_mean=True),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False,
        ),
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    Y_pred = ensure_at_least_one_label(Y_pred, scores)

    return Y_pred


def evaluate_dummy(Y, train_idx, test_idx):
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # Predict the most frequent single EC label for all samples.
    label_counts = Y_train.sum(axis=0)
    best_label = int(np.argmax(label_counts))

    Y_pred = np.zeros_like(Y_test)
    Y_pred[:, best_label] = 1

    return Y_pred


# ======================================================================================
# Saving
# ======================================================================================


def save_outputs(
    output_dir,
    metrics_rows,
    results_by_name,
    meta,
    label_counts,
    mlb,
    train_idx,
    test_idx,
    save_meta,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

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

    prediction_rows = []
    for name, result in results_by_name.items():
        Y_test = result["Y_test"]
        Y_pred = result["Y_pred"]

        true_labels = mlb.inverse_transform(Y_test)
        pred_labels = mlb.inverse_transform(Y_pred)

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

    if save_meta:
        meta_path = output_dir / "samples_metadata.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)
        print("sample metadata:", meta_path)

    print()
    print("Saved outputs")
    print("=============")
    print("metrics:", metrics_path)
    print("label counts:", label_counts_path)
    print("labels:", labels_path)
    print("predictions:", predictions_path)
    print("output_dir:", output_dir)


# ======================================================================================
# One radius
# ======================================================================================


def run_one_radius(radius, args, ec_by_mnxr, requested_models, mlp_hidden_layer_sizes):
    ecfp_params = make_ecfp_params(
        radius=radius,
        fp_size=args.fp_size,
        folded=not args.unfolded,
        custom=args.custom,
    )

    output_dir = (
        Path(args.output_dir) / args.sample_mode / f"radius_{radius}"
        if args.output_dir is not None
        else RESULTS_DIR / "metanetx_ec_prediction" / args.sample_mode / f"radius_{radius}"
    )

    print()
    print("=" * 100)
    print(f"MetaNetX EC prediction benchmark | radius {radius}")
    print("=" * 100)
    print("database_name:", args.database_name)
    print("ecfp_params:", ecfp_params)
    print("output_dir:", output_dir)
    print("sample_mode:", args.sample_mode)
    print("requested_models:", requested_models)

    print()
    print("Loading ReactionRules...")
    reaction_rules = ReactionRules.load(
        database_name=args.database_name,
        ecfp_params=ecfp_params,
    )

    X_reaction, X_center, y_labels, meta = expand_reactionrules_with_ec_labels(
        reaction_rules=reaction_rules,
        ec_by_mnxr=ec_by_mnxr,
        max_rules=args.max_rules,
        sample_mode=args.sample_mode,
    )

    Y, mlb, keep_sample, label_counts = binarize_and_filter_labels(
        y_labels=y_labels,
        min_label_count=args.min_label_count,
        max_labels=args.max_labels,
    )

    X_reaction = X_reaction[keep_sample]
    X_center = X_center[keep_sample]
    meta = meta.loc[keep_sample].reset_index(drop=True)

    print()
    print("Final dataset")
    print("=============")
    print("X_reaction:", X_reaction.shape)
    print("X_center:", X_center.shape)
    print("Y:", Y.shape)
    print("Mean labels per sample:", float(Y.sum(axis=1).mean()))

    feature_sets = make_feature_sets(
        X_reaction=X_reaction,
        X_center=X_center,
        use_sparse=not args.dense,
    )

    train_idx, test_idx = make_shared_split_indices(
        Y=Y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print()
    print("Shared train/test split")
    print("=======================")
    print("n_train:", len(train_idx))
    print("n_test:", len(test_idx))

    metrics_rows = []
    results_by_name = {}
    Y_test = Y[test_idx]

    # Dummy baseline.
    dummy_pred = evaluate_dummy(Y, train_idx, test_idx)
    dummy_metrics = compute_multilabel_metrics(
        feature_name="dummy_most_frequent_ec",
        classifier_name="dummy",
        X=feature_sets["reaction_ecfp"],
        Y_test=Y_test,
        Y_pred=dummy_pred,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=args.test_size,
        random_state=args.random_state,
        radius=radius,
        n_labels=Y.shape[1],
    )
    metrics_rows.append(dummy_metrics)
    results_by_name[dummy_metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": dummy_pred}

    for feature_name, X in feature_sets.items():
        print()
        print("-" * 100)
        print("Feature set:", feature_name)
        print("X:", X.shape)
        print("-" * 100)

        if "logistic_regression" in requested_models:
            print("Training logistic_regression...")
            pred = train_predict_logistic_regression(
                X=X,
                Y=Y,
                train_idx=train_idx,
                test_idx=test_idx,
                max_iter=args.max_iter,
                random_state=args.random_state,
            )
            metrics = compute_multilabel_metrics(
                feature_name=feature_name,
                classifier_name="logistic_regression",
                X=X,
                Y_test=Y_test,
                Y_pred=pred,
                train_idx=train_idx,
                test_idx=test_idx,
                test_size=args.test_size,
                random_state=args.random_state,
                radius=radius,
                n_labels=Y.shape[1],
            )
            metrics_rows.append(metrics)
            results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}

        if "random_forest" in requested_models:
            print("Training random_forest...")
            pred = train_predict_random_forest(
                X=X,
                Y=Y,
                train_idx=train_idx,
                test_idx=test_idx,
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.random_state,
            )
            metrics = compute_multilabel_metrics(
                feature_name=feature_name,
                classifier_name="random_forest",
                X=X,
                Y_test=Y_test,
                Y_pred=pred,
                train_idx=train_idx,
                test_idx=test_idx,
                test_size=args.test_size,
                random_state=args.random_state,
                radius=radius,
                n_labels=Y.shape[1],
            )
            metrics_rows.append(metrics)
            results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}

        if "gradient_boosting" in requested_models:
            print("Training gradient_boosting...")
            pred = train_predict_gradient_boosting(
                X=X,
                Y=Y,
                train_idx=train_idx,
                test_idx=test_idx,
                n_estimators=args.gb_n_estimators,
                learning_rate=args.gb_learning_rate,
                max_depth=args.gb_max_depth,
                random_state=args.random_state,
            )
            metrics = compute_multilabel_metrics(
                feature_name=feature_name,
                classifier_name="gradient_boosting",
                X=X,
                Y_test=Y_test,
                Y_pred=pred,
                train_idx=train_idx,
                test_idx=test_idx,
                test_size=args.test_size,
                random_state=args.random_state,
                radius=radius,
                n_labels=Y.shape[1],
            )
            # Fix typo-safe values in case above argument was accidentally changed.
            metrics["n_train"] = int(len(train_idx))
            metrics["n_test"] = int(len(test_idx))
            metrics_rows.append(metrics)
            results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}

        if "mlp" in requested_models:
            print("Training mlp...")
            pred = train_predict_mlp(
                X=X,
                Y=Y,
                train_idx=train_idx,
                test_idx=test_idx,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=args.mlp_max_iter,
                random_state=args.random_state,
            )
            metrics = compute_multilabel_metrics(
                feature_name=feature_name,
                classifier_name="mlp",
                X=X,
                Y_test=Y_test,
                Y_pred=pred,
                train_idx=train_idx,
                test_idx=test_idx,
                test_size=args.test_size,
                random_state=args.random_state,
                radius=radius,
                n_labels=Y.shape[1],
            )
            metrics_rows.append(metrics)
            results_by_name[metrics["result_name"]] = {"Y_test": Y_test, "Y_pred": pred}

    for row in metrics_rows:
        row["sample_mode"] = args.sample_mode

    metrics_df = pd.DataFrame(metrics_rows)

    preferred_cols = [
        "radius",
        "sample_mode",
        "result_name",
        "model",
        "classifier",
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
    ]

    cols = [c for c in preferred_cols if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in cols]
    metrics_df = metrics_df[cols + other_cols]

    print()
    print(f"Comparison | radius {radius}")
    print("============================")
    print(metrics_df)

    save_outputs(
        output_dir=output_dir,
        metrics_rows=metrics_rows,
        results_by_name=results_by_name,
        meta=meta,
        label_counts=label_counts,
        mlb=mlb,
        train_idx=train_idx,
        test_idx=test_idx,
        save_meta=args.save_meta,
    )

    return metrics_df


# ======================================================================================
# CLI
# ======================================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Predict MetaNetX EC numbers from reaction ECFP, reaction-center ECFP, "
            "and their concatenation."
        )
    )

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--radii", default="0,1,2,3,4,5")
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--database-name", default="metanetx")
    parser.add_argument("--reac-prop-path", default=DEFAULT_REAC_PROP)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        choices=["unique_rules", "occurrences"],
        default="unique_rules",
        help=(
            "How to build ML samples from deduplicated ReactionRules. "
            "unique_rules uses one sample per unique rule and aggregates EC labels; "
            "occurrences re-expands source IDs and reproduces the older occurrence-weighted dataset."
        ),
    )
    parser.add_argument("--unfolded", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--dense", action="store_true")

    parser.add_argument(
        "--ec-level",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help=(
            "Optional EC truncation level. "
            "Example: --ec-level 3 turns 1.2.1.19 into 1.2.1. "
            "Default keeps exact EC strings as present in reac_prop.tsv."
        ),
    )

    parser.add_argument(
        "--min-label-count",
        type=int,
        default=5,
        help="Remove EC labels occurring fewer than this many times.",
    )

    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Optional number of most frequent EC labels to keep.",
    )

    parser.add_argument(
        "--models",
        default="lr,rf,gb,mlp",
        help="Models to run: lr, rf, gb, mlp.",
    )

    parser.add_argument("--max-iter", type=int, default=2000)

    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)

    parser.add_argument("--gb-n-estimators", type=int, default=100)
    parser.add_argument("--gb-learning-rate", type=float, default=0.1)
    parser.add_argument("--gb-max-depth", type=int, default=3)

    parser.add_argument("--mlp-hidden-layers", default="512,128")
    parser.add_argument("--mlp-max-iter", type=int, default=100)

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--save-meta", action="store_true")

    return parser


# ======================================================================================
# Main
# ======================================================================================


def main():
    args = build_parser().parse_args()

    radii = parse_radii(args.radii, args.radius)
    requested_models = parse_models(args.models)
    mlp_hidden_layer_sizes = parse_mlp_hidden_layers(args.mlp_hidden_layers)

    base_output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else RESULTS_DIR / "metanetx_ec_prediction"
    )

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else base_output_dir / args.sample_mode / "metrics_all_radii.csv"
    )

    print("MetaNetX EC prediction benchmark across radii")
    print("=============================================")
    print("radii:", radii)
    print("database_name:", args.database_name)
    print("reac_prop_path:", args.reac_prop_path)
    print("ec_level:", args.ec_level)
    print("min_label_count:", args.min_label_count)
    print("max_labels:", args.max_labels)
    print("sample_mode:", args.sample_mode)
    print("base_output_dir:", base_output_dir)
    print("summary_output:", summary_output)
    print("requested_models:", requested_models)

    _, ec_by_mnxr = load_metanetx_ec_labels(
        reac_prop_path=Path(args.reac_prop_path),
        ec_level=args.ec_level,
    )

    all_metrics = []

    for radius in radii:
        metrics_df = run_one_radius(
            radius=radius,
            args=args,
            ec_by_mnxr=ec_by_mnxr,
            requested_models=requested_models,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
        )
        all_metrics.append(metrics_df)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(summary_output, index=False)

    print()
    print("=" * 100)
    print("All radii done")
    print("=" * 100)
    print("summary_output:", summary_output)

    preferred_cols = [
        "radius",
        "sample_mode",
        "result_name",
        "model",
        "classifier",
        "n_samples",
        "n_features",
        "n_labels",
        "subset_accuracy",
        "micro_f1",
        "macro_f1",
        "samples_jaccard",
    ]

    cols = [c for c in preferred_cols if c in all_metrics_df.columns]
    print(all_metrics_df[cols])


if __name__ == "__main__":
    main()
