#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark USPTO reaction-class prediction from reaction ECFPs across multiple radii.

Compares three feature sets:

    1. reaction ECFP
    2. reaction-center ECFP
    3. reaction ECFP + reaction-center ECFP

With four ML models:

    1. Logistic Regression
    2. Random Forest
    3. Gradient Boosting
    4. MLP

Target:
    USPTO datasetB.csv column `rxn_Class`.

Expected ReactionRules IDs:
    patentID__rowIndex__splitIndex

Example:
    US05849732__0__split0

Examples
--------
Run radii 0 to 5:

    python prediction.py --radii 0,1,2,3,4,5

Run radius 2 only:

    python prediction.py --radius 2

Run only LR/RF for radii 0 to 5:

    python prediction.py --radii 0,1,2,3,4,5 --models logistic_regression,random_forest

Test quickly:

    python prediction.py --radii 0,1,2 --max-rules 5000 --models logistic_regression,random_forest

Spyder/IPython:

    %runfile path/to/prediction.py --args "--radii 0,1,2,3,4,5 --models logistic_regression,random_forest"
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from morganrxn.core.paths import USPTO_DIR, RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules


# ======================================================================================
# Warnings
# ======================================================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ======================================================================================
# General helpers
# ======================================================================================

def parse_radii(radii_value, fallback_radius):
    """
    Parse radii from CLI.

    Priority:
        1. --radii if provided
        2. --radius otherwise

    Example
    -------
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

    if len(radii) == 0:
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
    """
    Parse requested models.

    Canonical model names:
        - logistic_regression
        - random_forest
        - gradient_boosting
        - mlp

    Short aliases are also accepted:
        lr, rf, gb
    """
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

    raw_models = [
        x.strip().lower()
        for x in str(models_value).split(",")
        if x.strip()
    ]

    unknown_models = [
        x for x in raw_models
        if x not in aliases
    ]

    if unknown_models:
        allowed = sorted(set(aliases.keys()))
        raise ValueError(
            f"Unknown models: {unknown_models}. "
            f"Allowed models: {allowed}"
        )

    # Keep user order while removing duplicates.
    requested_models = []
    for x in raw_models:
        canonical = aliases[x]
        if canonical not in requested_models:
            requested_models.append(canonical)

    return requested_models

def parse_mlp_hidden_layers(value):
    return tuple(
        int(x.strip())
        for x in str(value).split(",")
        if x.strip()
    )


# ======================================================================================
# ID parsing
# ======================================================================================

def split_merged_ids(value):
    """
    Split IDs that may have been merged by ReactionRules.drop_duplicates().

    Example
    -------
    "US1__0__split0|US2__7__split0"
    ->
    ["US1__0__split0", "US2__7__split0"]
    """
    if value is None:
        return []

    value = str(value).strip()

    if value == "":
        return []

    return [
        x.strip()
        for x in value.split("|")
        if x.strip()
    ]


def parse_uspto_rule_id(rule_id: str):
    """
    Parse USPTO rule ID.

    Expected format:
        patentID__rowIndex__splitIndex

    Example:
        US05849732__0__split0
    """
    if rule_id is None:
        return None

    rule_id = str(rule_id).strip()

    match = re.match(
        r"^(?P<patent_id>.+)__+(?P<row_index>[0-9]+)__split(?P<split_index>[0-9]+)$",
        rule_id,
    )

    if match is None:
        return None

    return {
        "patent_id": match.group("patent_id"),
        "row_index": int(match.group("row_index")),
        "split_index": int(match.group("split_index")),
    }


# ======================================================================================
# Data loading
# ======================================================================================

def load_dataset_classes(dataset_path: Path, class_col: str) -> pd.Series:
    """
    Load datasetB.csv and return class labels indexed by original row number.
    """
    print("Loading USPTO dataset classes...")
    print("dataset path:", dataset_path)

    df = pd.read_csv(dataset_path)

    if class_col not in df.columns:
        raise ValueError(
            f"Column '{class_col}' not found in datasetB.\n"
            f"Available columns: {list(df.columns)}"
        )

    y_by_row_index = df[class_col].copy()

    print("dataset shape:", df.shape)
    print("class counts in original dataset:")
    print(y_by_row_index.value_counts().sort_index())

    return y_by_row_index


def expand_reactionrules_with_labels(
    reaction_rules,
    y_by_row_index,
    max_rules=None,
):
    """
    Expand ReactionRules entries into labelled ML samples.

    If ReactionRules were deduplicated, one entry may contain several original
    IDs in reaction_monocomp_id separated by '|'. We expand them so each
    original USPTO source row contributes one sample.

    Returns
    -------
    X_reaction : np.ndarray
    X_center : np.ndarray
    y : np.ndarray
    meta : pd.DataFrame
    """
    X_reaction = []
    X_center = []
    y = []
    meta_rows = []

    n_rules_seen = 0
    n_ids_seen = 0
    n_bad_ids = 0
    n_missing_labels = 0

    n_total_rules = len(reaction_rules)

    if max_rules is not None:
        n_total_rules = min(n_total_rules, max_rules)

    for i in range(n_total_rules):
        n_rules_seen += 1

        merged_ids = reaction_rules.reaction_monocomp_id[i]
        source_ids = split_merged_ids(merged_ids)

        if not source_ids:
            source_ids = [reaction_rules.reaction_id[i]]

        for source_id in source_ids:
            n_ids_seen += 1

            parsed = parse_uspto_rule_id(source_id)

            if parsed is None:
                n_bad_ids += 1
                continue

            row_index = parsed["row_index"]

            if row_index < 0 or row_index >= len(y_by_row_index):
                n_missing_labels += 1
                continue

            label = y_by_row_index.iloc[row_index]

            if pd.isna(label):
                n_missing_labels += 1
                continue

            X_reaction.append(reaction_rules.ecfp_reaction[i])
            X_center.append(reaction_rules.ecfp_reaction_center[i])
            y.append(int(label))

            meta_rows.append(
                {
                    "rule_index": i,
                    "source_id": source_id,
                    "patent_id": parsed["patent_id"],
                    "datasetB_row_index": row_index,
                    "split_index": parsed["split_index"],
                    "rxn_Class": int(label),
                    "template_reaction": reaction_rules.template_reaction[i],
                }
            )

    print()
    print("Label matching summary")
    print("======================")
    print("ReactionRules entries read:", n_rules_seen)
    print("Source IDs seen:", n_ids_seen)
    print("Bad source IDs:", n_bad_ids)
    print("Missing labels:", n_missing_labels)
    print("Final ML samples:", len(y))

    if len(y) == 0:
        raise ValueError(
            "No labelled samples were recovered. Check that reaction_monocomp_id "
            "contains IDs like patentID__rowIndex__splitX."
        )

    X_reaction = np.asarray(X_reaction, dtype=np.float32)
    X_center = np.asarray(X_center, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    meta = pd.DataFrame(meta_rows)

    return X_reaction, X_center, y, meta


# ======================================================================================
# Feature sets and split
# ======================================================================================

def make_feature_sets(
    X_reaction: np.ndarray,
    X_center: np.ndarray,
    use_sparse: bool,
):
    """
    Build all feature sets.

    Returns
    -------
    dict
        feature_set_name -> X
    """
    if use_sparse:
        X_reaction_sp = sparse.csr_matrix(X_reaction)
        X_center_sp = sparse.csr_matrix(X_center)
        X_both_sp = sparse.hstack(
            [X_reaction_sp, X_center_sp],
            format="csr",
        )

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


def make_shared_split_indices(
    y,
    test_size: float,
    random_state: int,
):
    """
    Create one shared train/test split for every model and every feature set.
    This makes the comparison fair.
    """
    indices = np.arange(len(y))

    stratify = y
    class_counts = pd.Series(y).value_counts()

    if class_counts.min() < 2:
        print(
            "WARNING: at least one class has fewer than 2 samples; "
            "using non-stratified split."
        )
        stratify = None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return train_idx, test_idx


def subset_X(X, indices):
    """
    Works for dense numpy arrays and scipy sparse matrices.
    """
    return X[indices]


def to_dense_float32(X):
    """
    Convert sparse/dense input to dense float32 array.
    Used for MLP.
    """
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)

    return np.asarray(X, dtype=np.float32)


# ======================================================================================
# Metrics
# ======================================================================================

def compute_metrics_and_outputs(
    model_name: str,
    classifier_name: str,
    X,
    y,
    y_test,
    y_pred,
    train_idx,
    test_idx,
    test_size: float,
    random_state: int,
    class_weight,
    radius: int,
):
    labels = sorted(int(x) for x in np.unique(y))

    metrics = {
        "radius": int(radius),
        "result_name": f"r{radius}__{model_name}__{classifier_name}",
        "model": model_name,
        "classifier": classifier_name,
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "class_weight": str(class_weight),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    print("accuracy:", metrics["accuracy"])
    print("balanced_accuracy:", metrics["balanced_accuracy"])
    print("macro_f1:", metrics["macro_f1"])
    print("weighted_f1:", metrics["weighted_f1"])

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ======================================================================================
# Models
# ======================================================================================

def train_and_evaluate_logistic_regression(
    X,
    y,
    train_idx,
    test_idx,
    model_name: str,
    test_size: float,
    random_state: int,
    max_iter: int,
    class_weight,
    radius: int,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        class_weight=class_weight,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )

    model = make_pipeline(
        StandardScaler(with_mean=False),
        clf,
    )

    full_model_name = f"r{radius}__{model_name}__logistic_regression"

    print()
    print(f"Training model: {full_model_name}")
    print("=" * (16 + len(full_model_name)))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("class_weight:", class_weight)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return compute_metrics_and_outputs(
        model_name=model_name,
        classifier_name="logistic_regression",
        X=X,
        y=y,
        y_test=y_test,
        y_pred=y_pred,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=test_size,
        random_state=random_state,
        class_weight=class_weight,
        radius=radius,
    )


def train_and_evaluate_random_forest(
    X,
    y,
    train_idx,
    test_idx,
    model_name: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth,
    min_samples_leaf: int,
    class_weight,
    radius: int,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )

    full_model_name = f"r{radius}__{model_name}__random_forest"

    print()
    print(f"Training model: {full_model_name}")
    print("=" * (16 + len(full_model_name)))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("n_estimators:", n_estimators)
    print("max_depth:", max_depth)
    print("min_samples_leaf:", min_samples_leaf)
    print("class_weight:", class_weight)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return compute_metrics_and_outputs(
        model_name=model_name,
        classifier_name="random_forest",
        X=X,
        y=y,
        y_test=y_test,
        y_pred=y_pred,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=test_size,
        random_state=random_state,
        class_weight=class_weight,
        radius=radius,
    )


def train_and_evaluate_gradient_boosting(
    X,
    y,
    train_idx,
    test_idx,
    model_name: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    class_weight,
    radius: int,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # GradientBoostingClassifier does not support sparse matrices directly.
    X_train = to_dense_float32(X_train)
    X_test = to_dense_float32(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        verbose=0,
    )

    sample_weight = None
    if class_weight == "balanced":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

    full_model_name = f"r{radius}__{model_name}__gradient_boosting"

    print()
    print(f"Training model: {full_model_name}")
    print("=" * (16 + len(full_model_name)))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("n_estimators:", n_estimators)
    print("learning_rate:", learning_rate)
    print("max_depth:", max_depth)
    print("min_samples_leaf:", min_samples_leaf)
    print("class_weight:", class_weight)

    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test)

    return compute_metrics_and_outputs(
        model_name=model_name,
        classifier_name="gradient_boosting",
        X=X,
        y=y,
        y_test=y_test,
        y_pred=y_pred,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=test_size,
        random_state=random_state,
        class_weight=class_weight,
        radius=radius,
    )

def train_and_evaluate_mlp(
    X,
    y,
    train_idx,
    test_idx,
    model_name: str,
    test_size: float,
    random_state: int,
    hidden_layer_sizes,
    max_iter: int,
    radius: int,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train = to_dense_float32(X_train)
    X_test = to_dense_float32(X_test)

    clf = MLPClassifier(
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
    )

    model = make_pipeline(
        StandardScaler(with_mean=True),
        clf,
    )

    full_model_name = f"r{radius}__{model_name}__mlp"

    print()
    print(f"Training model: {full_model_name}")
    print("=" * (16 + len(full_model_name)))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("hidden_layer_sizes:", hidden_layer_sizes)
    print("max_iter:", max_iter)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return compute_metrics_and_outputs(
        model_name=model_name,
        classifier_name="mlp",
        X=X,
        y=y,
        y_test=y_test,
        y_pred=y_pred,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=test_size,
        random_state=random_state,
        class_weight=None,
        radius=radius,
    )


def evaluate_dummy_baseline(
    X,
    y,
    train_idx,
    test_idx,
    test_size: float,
    random_state: int,
    radius: int,
):
    X_train = subset_X(X, train_idx)
    X_test = subset_X(X, test_idx)
    y_train = y[train_idx]
    y_test = y[test_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)

    metrics = {
        "radius": int(radius),
        "result_name": f"r{radius}__dummy_most_frequent",
        "model": "dummy_most_frequent",
        "classifier": "dummy",
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "class_weight": "None",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    print()
    print(f"Dummy baseline | radius {radius}")
    print("==============================")
    print("accuracy:", metrics["accuracy"])
    print("balanced_accuracy:", metrics["balanced_accuracy"])
    print("macro_f1:", metrics["macro_f1"])
    print("weighted_f1:", metrics["weighted_f1"])

    return metrics


# ======================================================================================
# Saving
# ======================================================================================

def save_class_counts(
    output_dir: Path,
    y_original: pd.Series,
    y_final: np.ndarray,
):
    original_counts = y_original.value_counts().sort_index()
    final_counts = pd.Series(y_final).value_counts().sort_index()

    class_counts = pd.DataFrame(
        {
            "datasetB_original": original_counts,
            "reactionrules_labelled_samples": final_counts,
        }
    ).fillna(0).astype(int)

    class_counts.index.name = "rxn_Class"

    path = output_dir / "class_counts.csv"
    class_counts.to_csv(path)

    print("class counts:", path)


def save_outputs(
    output_dir: Path,
    metrics_rows,
    results_by_model,
    meta: pd.DataFrame,
    y_original: pd.Series,
    y_final: np.ndarray,
    save_meta: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    reports_path = output_dir / "classification_reports.json"
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                name: result["classification_report"]
                for name, result in results_by_model.items()
                if "classification_report" in result
            },
            f,
            indent=2,
        )

    predictions_rows = []

    for name, result in results_by_model.items():
        if "confusion_matrix" not in result:
            continue

        labels = result["labels"]
        cm = result["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        )

        cm_path = output_dir / f"confusion_matrix_{name}.csv"
        cm_df.to_csv(cm_path)

        if "y_test" in result and "y_pred" in result:
            y_test = result["y_test"]
            y_pred = result["y_pred"]

            for yt, yp in zip(y_test, y_pred):
                predictions_rows.append(
                    {
                        "model": name,
                        "y_true": int(yt),
                        "y_pred": int(yp),
                    }
                )

    if predictions_rows:
        predictions_path = output_dir / "test_predictions.tsv"
        pd.DataFrame(predictions_rows).to_csv(
            predictions_path,
            sep="\t",
            index=False,
        )

    save_class_counts(
        output_dir=output_dir,
        y_original=y_original,
        y_final=y_final,
    )

    if save_meta:
        meta_path = output_dir / "samples_metadata.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)
        print("sample metadata:", meta_path)

    print()
    print("Saved outputs")
    print("=============")
    print("metrics:", metrics_path)
    print("reports:", reports_path)
    print("output_dir:", output_dir)


# ======================================================================================
# One-radius benchmark
# ======================================================================================

def run_one_radius(
    radius: int,
    args,
    y_by_row_index: pd.Series,
    requested_models,
    lr_class_weight,
    rf_class_weight,
    gb_class_weight,
    mlp_hidden_layer_sizes,
):
    ecfp_params = make_ecfp_params(
        radius=radius,
        fp_size=args.fp_size,
        folded=not args.unfolded,
        custom=args.custom,
    )

    output_dir = (
        Path(args.output_dir) / f"radius_{radius}"
        if args.output_dir is not None
        else RESULTS_DIR / "uspto_reaction_classification" / f"radius_{radius}"
    )

    print()
    print("=" * 100)
    print(f"USPTO reaction-class benchmark | radius {radius}")
    print("=" * 100)
    print("database_name:", args.database_name)
    print("ecfp_params:", ecfp_params)
    print("output_dir:", output_dir)
    print("requested_models:", requested_models)

    print()
    print("Loading ReactionRules...")
    reaction_rules = ReactionRules.load(
        database_name=args.database_name,
        ecfp_params=ecfp_params,
    )

    X_reaction, X_center, y, meta = expand_reactionrules_with_labels(
        reaction_rules=reaction_rules,
        y_by_row_index=y_by_row_index,
        max_rules=args.max_rules,
    )

    print()
    print("Final dataset")
    print("=============")
    print("X_reaction:", X_reaction.shape)
    print("X_center:", X_center.shape)
    print("y:", y.shape)
    print("class counts after rule filtering / matching:")
    print(pd.Series(y).value_counts().sort_index())

    use_sparse = not args.dense

    feature_sets = make_feature_sets(
        X_reaction=X_reaction,
        X_center=X_center,
        use_sparse=use_sparse,
    )

    train_idx, test_idx = make_shared_split_indices(
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print()
    print("Shared train/test split")
    print("=======================")
    print("n_train:", len(train_idx))
    print("n_test:", len(test_idx))

    metrics_rows = []
    results_by_model = {}

    dummy_metrics = evaluate_dummy_baseline(
        X=feature_sets["reaction_ecfp"],
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        test_size=args.test_size,
        random_state=args.random_state,
        radius=radius,
    )

    metrics_rows.append(dummy_metrics)

    for feature_name, X in feature_sets.items():
        if "logistic_regression" in requested_models:
            result_lr = train_and_evaluate_logistic_regression(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                model_name=feature_name,
                test_size=args.test_size,
                random_state=args.random_state,
                max_iter=args.max_iter,
                class_weight=lr_class_weight,
                radius=radius,
            )

            result_name = result_lr["metrics"]["result_name"]
            results_by_model[result_name] = result_lr
            metrics_rows.append(result_lr["metrics"])

        if "random_forest" in requested_models:
            result_rf = train_and_evaluate_random_forest(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                model_name=feature_name,
                test_size=args.test_size,
                random_state=args.random_state,
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                class_weight=rf_class_weight,
                radius=radius,
            )

            result_name = result_rf["metrics"]["result_name"]
            results_by_model[result_name] = result_rf
            metrics_rows.append(result_rf["metrics"])

        if "gradient_boosting" in requested_models:
            result_gb = train_and_evaluate_gradient_boosting(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                model_name=feature_name,
                test_size=args.test_size,
                random_state=args.random_state,
                n_estimators=args.gb_n_estimators,
                learning_rate=args.gb_learning_rate,
                max_depth=args.gb_max_depth,
                min_samples_leaf=args.gb_min_samples_leaf,
                class_weight=gb_class_weight,
                radius=radius,
            )

            result_name = result_gb["metrics"]["result_name"]
            results_by_model[result_name] = result_gb
            metrics_rows.append(result_gb["metrics"])

        if "mlp" in requested_models:
            result_mlp = train_and_evaluate_mlp(
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                model_name=feature_name,
                test_size=args.test_size,
                random_state=args.random_state,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=args.mlp_max_iter,
                radius=radius,
            )

            result_name = result_mlp["metrics"]["result_name"]
            results_by_model[result_name] = result_mlp
            metrics_rows.append(result_mlp["metrics"])

    print()
    print(f"Comparison | radius {radius}")
    print("============================")
    metrics_df = pd.DataFrame(metrics_rows)

    preferred_cols = [
        "radius",
        "result_name",
        "model",
        "classifier",
        "n_samples",
        "n_train",
        "n_test",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "class_weight",
    ]

    cols = [c for c in preferred_cols if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in cols]
    metrics_df = metrics_df[cols + other_cols]

    print(metrics_df)

    save_outputs(
        output_dir=output_dir,
        metrics_rows=metrics_rows,
        results_by_model=results_by_model,
        meta=meta,
        y_original=y_by_row_index,
        y_final=y,
        save_meta=args.save_meta,
    )

    return metrics_df


# ======================================================================================
# CLI
# ======================================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare USPTO rxn_Class prediction using reaction ECFP, "
            "reaction-center ECFP, and their concatenation across many radii."
        )
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Single ECFP radius. Used only if --radii is not provided.",
    )

    parser.add_argument(
        "--radii",
        default="0,1,2,3,4,5",
        help="Comma-separated radii. Example: --radii 0,1,2,3,4,5",
    )

    parser.add_argument(
        "--fp-size",
        type=int,
        default=1024,
        help="Folded ECFP size used to load ReactionRules.",
    )

    parser.add_argument(
        "--database-name",
        default="uspto",
        help="ReactionRules database name.",
    )

    parser.add_argument(
        "--dataset-path",
        default=USPTO_DIR / "datasetB.csv",
        help="Path to original USPTO datasetB.csv.",
    )

    parser.add_argument(
        "--class-col",
        default="rxn_Class",
        help="Class column in datasetB.csv.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test-set fraction.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--max-rules",
        type=int,
        default=None,
        help="Optional maximum number of ReactionRules entries to use.",
    )

    parser.add_argument(
        "--unfolded",
        action="store_true",
        help="Use unfolded ReactionRules loading parameters.",
    )

    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom ReactionRules loading parameters.",
    )

    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense arrays instead of scipy sparse matrices.",
    )

    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "balanced_subsample", "none"],
        help=(
            "Class weighting. "
            "For LogisticRegression and GradientBoosting, balanced_subsample is treated as balanced."
        ),
    )

    parser.add_argument(
        "--models",
        default="logistic_regression,random_forest,gradient_boosting,mlp",
        help=(
            "Models to run: logistic_regression, random_forest, gradient_boosting, mlp. Short aliases lr, rf, gb are accepted. "
            "Default: logistic_regression,random_forest,gradient_boosting,mlp."
        ),
    )

    # Logistic regression
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Max iterations for LogisticRegression.",
    )

    # Random forest
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Number of trees for RandomForestClassifier.",
    )

    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Maximum depth for RandomForestClassifier.",
    )

    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf for RandomForestClassifier.",
    )

    # Gradient Boosting
    parser.add_argument(
        "--gb-n-estimators",
        type=int,
        default=100,
        help="Number of boosting stages for GradientBoostingClassifier.",
    )

    parser.add_argument(
        "--gb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for GradientBoostingClassifier.",
    )

    parser.add_argument(
        "--gb-max-depth",
        type=int,
        default=3,
        help="Maximum depth for GradientBoostingClassifier base trees.",
    )

    parser.add_argument(
        "--gb-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf for GradientBoostingClassifier.",
    )

    # MLP
    parser.add_argument(
        "--mlp-hidden-layers",
        default="512,128",
        help=(
            "Comma-separated hidden layer sizes for MLPClassifier. "
            "Default: 512,128."
        ),
    )

    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=100,
        help="Max iterations for MLPClassifier.",
    )

    # Outputs
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base output directory. "
            "Default: results/uspto_reaction_classification. "
            "Each radius is saved in radius_<r>/."
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Path to global metrics CSV. "
            "Default: <base_output_dir>/metrics_all_radii.csv."
        ),
    )

    parser.add_argument(
        "--save-meta",
        action="store_true",
        help="Save per-sample metadata TSV for each radius.",
    )

    return parser


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = build_parser().parse_args()

    radii = parse_radii(
        radii_value=args.radii,
        fallback_radius=args.radius,
    )

    requested_models = parse_models(args.models)
    mlp_hidden_layer_sizes = parse_mlp_hidden_layers(args.mlp_hidden_layers)

    if args.class_weight == "none":
        lr_class_weight = None
        rf_class_weight = None
        gb_class_weight = None
    elif args.class_weight == "balanced_subsample":
        lr_class_weight = "balanced"
        rf_class_weight = "balanced_subsample"
        gb_class_weight = "balanced"
    else:
        lr_class_weight = "balanced"
        rf_class_weight = "balanced"
        gb_class_weight = "balanced"

    base_output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else RESULTS_DIR / "uspto_reaction_classification"
    )

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else base_output_dir / "metrics_all_radii.csv"
    )

    dataset_path = Path(args.dataset_path)

    print("USPTO reaction-class benchmark across radii")
    print("==========================================")
    print("radii:", radii)
    print("database_name:", args.database_name)
    print("dataset_path:", dataset_path)
    print("class_col:", args.class_col)
    print("base_output_dir:", base_output_dir)
    print("summary_output:", summary_output)
    print("requested_models:", requested_models)
    print("lr_class_weight:", lr_class_weight)
    print("rf_class_weight:", rf_class_weight)
    print("gb_class_weight:", gb_class_weight)
    print("mlp_hidden_layer_sizes:", mlp_hidden_layer_sizes)

    y_by_row_index = load_dataset_classes(
        dataset_path=dataset_path,
        class_col=args.class_col,
    )

    all_metrics = []

    for radius in radii:
        metrics_df = run_one_radius(
            radius=radius,
            args=args,
            y_by_row_index=y_by_row_index,
            requested_models=requested_models,
            lr_class_weight=lr_class_weight,
            rf_class_weight=rf_class_weight,
            gb_class_weight=gb_class_weight,
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
        "result_name",
        "model",
        "classifier",
        "n_samples",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    cols = [c for c in preferred_cols if c in all_metrics_df.columns]
    print()
    print(all_metrics_df[cols])


if __name__ == "__main__":
    main()