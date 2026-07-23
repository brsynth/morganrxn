# -*- coding: utf-8 -*-
"""
t_sne.py

Run t-SNE on morganrxn reaction-rule ECFP vectors.

The script outputs t-SNE figures for radii 0 to 5 by default.

For each radius, it saves:

1. t-SNE of reaction ECFPs
   reaction_rules.ecfp_reaction

2. t-SNE of reaction-center ECFPs
   reaction_rules.ecfp_reaction_center

Example
-------
python t_sne.py

Compare MetaNetX and USPTO with signed split encoding and Jaccard distance:
python t_sne.py \
    --datasets metanetx uspto \
    --encoding signed_split \
    --metric jaccard \
    --init random

Save coordinates as .npz files:
python t_sne.py --save-coords
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from morganrxn.core.paths import RESULTS_DIR
from morganrxn.core.reaction_rules import ReactionRules


# =============================================================================
# Default user parameters
# =============================================================================

DEFAULT_ECFP_PARAMS = {
    "radius": 2,
    "fpSize": 1024,
    "folded": True,
    "custom": False,
}

DEFAULT_RADII = list(range(6))

DEFAULT_MIN_SMI_SUB_ATOMS = 5

DEFAULT_DATASETS = ["metanetx", "uspto"]

DEFAULT_LABELS = {
    "metanetx": "MetaNetX",
    "uspto": "USPTO",
}

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "t_sne"

DEFAULT_REACTION_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "tsne_reaction_ecfp.png"
DEFAULT_REACTION_CENTER_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "tsne_reaction_center_ecfp.png"

DEFAULT_REACTION_OUTPUT_COORDS = DEFAULT_OUTPUT_DIR / "tsne_reaction_ecfp_coords.npz"
DEFAULT_REACTION_CENTER_OUTPUT_COORDS = DEFAULT_OUTPUT_DIR / "tsne_reaction_center_ecfp_coords.npz"


# =============================================================================
# Data loading and preprocessing
# =============================================================================


def load_ecfps(
    dataset_names: Iterable[str],
    ecfp_params: Dict,
    vector_type: str,
    min_smi_sub_atoms: int = DEFAULT_MIN_SMI_SUB_ATOMS,
) -> Dict[str, np.ndarray]:
    """
    Load ECFP vectors for each reaction-rule dataset.

    Parameters
    ----------
    dataset_names:
        Names of the reaction-rule datasets to load.

    ecfp_params:
        ECFP parameters used by ReactionRules.load.

    vector_type:
        - "reaction": loads reaction_rules.ecfp_reaction
        - "reaction_center": loads reaction_rules.ecfp_reaction_center

    min_smi_sub_atoms:
        Minimum number of atoms required in smi_sub for a rule to be kept,
        same filtering as applied in the other paper_results scripts.
    """
    ecfps_by_dataset: Dict[str, np.ndarray] = {}

    for dataset_name in dataset_names:
        print(f"Loading {dataset_name} ...")
        reaction_rules = ReactionRules.load(database_name=dataset_name, ecfp_params=ecfp_params)
        reaction_rules.filter_by_smi_sub_atoms(min_atoms=min_smi_sub_atoms, verbose=True)

        if vector_type == "reaction":
            ecfps = np.asarray(reaction_rules.ecfp_reaction)
        elif vector_type == "reaction_center":
            ecfps = np.asarray(reaction_rules.ecfp_reaction_center)
        else:
            raise ValueError(f"Unknown vector_type: {vector_type}")

        if ecfps.ndim != 2:
            raise ValueError(
                f"{dataset_name} returned an invalid ECFP matrix "
                f"for vector_type={vector_type}: shape={ecfps.shape}"
            )

        ecfps_by_dataset[dataset_name] = ecfps

        print(
            f"  {dataset_name}: {ecfps.shape[0]} rules, "
            f"vector size {ecfps.shape[1]}"
        )

    return ecfps_by_dataset


def ecfp_rows_as_set(X: np.ndarray) -> set[tuple]:
    """Convert an ECFP matrix to a set of hashable row tuples."""
    return {tuple(row.tolist()) for row in X}


def print_ecfp_intersections(
    ecfps_by_dataset: Dict[str, np.ndarray],
    dataset_names: List[str],
    vector_label: str,
) -> None:
    """Print the number of unique ECFP vectors shared by datasets."""
    ecfp_sets = {
        dataset_name: ecfp_rows_as_set(ecfps_by_dataset[dataset_name])
        for dataset_name in dataset_names
    }

    print(f"\n{vector_label} intersections:")

    if len(dataset_names) < 2:
        print("  Only one dataset provided; no intersection to compute.")
        return

    for i, dataset_a in enumerate(dataset_names):
        for dataset_b in dataset_names[i + 1 :]:
            intersection_size = len(ecfp_sets[dataset_a] & ecfp_sets[dataset_b])
            print(f"  {dataset_a} ∩ {dataset_b}: {intersection_size} unique vectors")

    if len(dataset_names) > 2:
        common_to_all = set.intersection(*(ecfp_sets[name] for name in dataset_names))
        print(f"  intersection across all datasets: {len(common_to_all)} unique vectors")


def encode_ecfps(X: np.ndarray, encoding: str) -> np.ndarray:
    """
    Encode ECFP vectors before t-SNE.

    Parameters
    ----------
    X:
        Raw ECFP matrix, shape (n_samples, n_features).

    encoding:
        - "raw": keep raw values as float32. Good with metric="cosine".
        - "binary": convert nonzero entries to 1. Good with metric="jaccard".
        - "signed_split": split signed vectors into positive and negative binary halves.
          Useful when reaction ECFPs contain positive and negative signed counts.
    """
    if encoding == "raw":
        return X.astype(np.float32)

    if encoding == "binary":
        return (X != 0).astype(np.uint8)

    if encoding == "signed_split":
        X_pos = (X > 0).astype(np.uint8)
        X_neg = (X < 0).astype(np.uint8)
        return np.concatenate([X_pos, X_neg], axis=1)

    raise ValueError(f"Unknown encoding: {encoding}")


def build_combined_matrix(
    ecfps_by_dataset: Dict[str, np.ndarray],
    dataset_names: List[str],
    encoding: str,
    max_per_dataset: int | None = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one combined matrix X and one label vector."""
    rng = np.random.default_rng(random_state)

    X_parts: List[np.ndarray] = []
    labels: List[str] = []

    for dataset_name in dataset_names:
        X_dataset = ecfps_by_dataset[dataset_name]

        if max_per_dataset is not None and len(X_dataset) > max_per_dataset:
            idx = rng.choice(len(X_dataset), size=max_per_dataset, replace=False)
            X_dataset = X_dataset[idx]
            print(f"Subsampled {dataset_name} to {max_per_dataset} rules")

        X_dataset = encode_ecfps(X_dataset, encoding=encoding)
        label = DEFAULT_LABELS.get(dataset_name, dataset_name)

        X_parts.append(X_dataset)
        labels.extend([label] * len(X_dataset))

    X = np.vstack(X_parts)
    y = np.asarray(labels)

    print("Combined matrix:", X.shape)
    print("Labels:", y.shape)

    return X, y


# =============================================================================
# t-SNE and plotting
# =============================================================================


def run_tsne(
    X: np.ndarray,
    metric: str = "cosine",
    perplexity: float = 50.0,
    learning_rate: str | float = "auto",
    init: str = "random",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> np.ndarray:
    """Run scikit-learn t-SNE and return 2D coordinates."""
    if metric == "jaccard" and not np.issubdtype(X.dtype, np.integer):
        print("Warning: Jaccard is usually intended for binary/integer vectors.")

    if metric == "jaccard" and init == "pca":
        raise ValueError(
            "init='pca' is not compatible with metric='jaccard'. "
            "Use --init random instead."
        )

    tsne = TSNE(
        n_components=2,
        metric=metric,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return tsne.fit_transform(X)


def robust_limits(
    coords: np.ndarray,
    quantile: float = 0.999,
    margin: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute axis limits that ignore extreme outliers.

    A handful of stray points can sit far outside the main cloud and squash
    the rest of the figure. Limits come from a symmetric quantile range
    instead of the raw min/max, padded by `margin`.
    """
    lo = np.quantile(coords, 1.0 - quantile, axis=0)
    hi = np.quantile(coords, quantile, axis=0)

    span = hi - lo
    pad = margin * np.where(span > 0, span, 1.0)

    lo = lo - pad
    hi = hi + pad

    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    output_png: Path,
    title: str,
    point_size: float = 2.0,
    alpha: float = 0.5,
    dpi: int = 300,
    clip_quantile: float | None = 0.999,
    formats: Iterable[str] = ("png",),
) -> None:
    """
    Plot and save a t-SNE scatter plot.

    clip_quantile:
        If set, the view is cropped to this symmetric quantile range so that
        stray outliers do not compress the main cloud. Clipped points are
        still part of the embedding, they are simply outside the view.
        Set to None to show the full extent.

    formats:
        Output file formats to write, e.g. ("png",), ("pdf",) or ("png", "pdf").
        The base name comes from `output_png`; the suffix is replaced per format.
        PDF is vector-based, so its quality is independent of `dpi`.
    """
    plt.figure(figsize=(9, 9))

    for label in sorted(np.unique(labels)):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=label,
        )

    if clip_quantile is not None:
        (x_lo, x_hi), (y_lo, y_hi) = robust_limits(coords, quantile=clip_quantile)
        n_clipped = int(
            np.sum(
                (coords[:, 0] < x_lo)
                | (coords[:, 0] > x_hi)
                | (coords[:, 1] < y_lo)
                | (coords[:, 1] > y_hi)
            )
        )
        plt.xlim(x_lo, x_hi)
        plt.ylim(y_lo, y_hi)
        if n_clipped:
            print(f"  {n_clipped} outlier point(s) outside the plotted view")

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.legend(markerscale=5)
    plt.axis("off")
    plt.title(title)
    for fmt in formats:
        output_path = output_png.with_suffix(f".{fmt}")
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")
    plt.close()


# =============================================================================
# Workflow
# =============================================================================


def run_tsne_workflow(
    dataset_names: List[str],
    ecfp_params: Dict,
    radius: int,
    vector_type: str,
    vector_label: str,
    encoding: str,
    metric: str,
    perplexity: float,
    learning_rate: str | float,
    init: str,
    random_state: int,
    n_jobs: int,
    max_per_dataset: int | None,
    output_png: Path,
    output_coords: Path | None,
    point_size: float,
    alpha: float,
    dpi: int,
    clip_quantile: float | None = 0.999,
    min_smi_sub_atoms: int = DEFAULT_MIN_SMI_SUB_ATOMS,
    formats: Iterable[str] = ("png",),
) -> None:
    """Run the full t-SNE workflow for one vector type."""
    print("\n" + "=" * 80)
    print(f"Running t-SNE for {vector_label} at radius h={radius}")
    print("=" * 80)

    ecfps_by_dataset = load_ecfps(
        dataset_names=dataset_names,
        ecfp_params=ecfp_params,
        vector_type=vector_type,
        min_smi_sub_atoms=min_smi_sub_atoms,
    )

    print_ecfp_intersections(
        ecfps_by_dataset=ecfps_by_dataset,
        dataset_names=dataset_names,
        vector_label=vector_label,
    )

    X, labels = build_combined_matrix(
        ecfps_by_dataset=ecfps_by_dataset,
        dataset_names=dataset_names,
        encoding=encoding,
        max_per_dataset=max_per_dataset,
        random_state=random_state,
    )

    coords = run_tsne(
        X,
        metric=metric,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )

    plot_tsne(
        coords=coords,
        labels=labels,
        output_png=output_png,
        title=f"t-SNE of {vector_label} (h={radius})",
        point_size=point_size,
        alpha=alpha,
        dpi=dpi,
        clip_quantile=clip_quantile,
        formats=formats,
    )

    if output_coords is not None:
        output_coords.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_coords, coords=coords, labels=labels)
        print(f"Saved coordinates to: {output_coords}")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run t-SNE on morganrxn reaction ECFPs and "
            "reaction-center ECFPs."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Reaction-rule database names to load. Example: "
            "metanetx uspto"
        ),
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=DEFAULT_RADII,
        help="ECFP radii to compute. Default: 0 1 2 3 4 5.",
    )
    parser.add_argument(
        "--encoding",
        choices=["raw", "binary", "signed_split"],
        default="raw",
        help="How to encode ECFP vectors before t-SNE.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="t-SNE metric. Typical values: cosine, euclidean, jaccard.",
    )
    parser.add_argument("--perplexity", type=float, default=50.0)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument(
        "--init",
        default="pca",
        choices=["random", "pca"],
        help=(
            "t-SNE initialization. 'pca' preserves global structure and avoids "
            "stray points left behind by early exaggeration. Use 'random' with "
            "metric='jaccard'."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional random subsample size per dataset, useful for quick tests.",
    )

    parser.add_argument(
        "--min-smi-sub-atoms",
        type=int,
        default=DEFAULT_MIN_SMI_SUB_ATOMS,
        help="Minimum number of heavy atoms in smi_sub for a rule to be kept.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and optional coordinates are saved.",
    )

    parser.add_argument(
        "--save-coords",
        action="store_true",
        help="Save t-SNE coordinates and labels as compressed .npz files.",
    )

    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.999,
        help=(
            "Crop the plotted view to this symmetric quantile range so stray "
            "outliers do not squash the main cloud. Use 1.0 to disable."
        ),
    )

    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=["png", "pdf"],
        default=["png"],
        help=(
            "Figure output format(s). Use 'pdf' for lossless vector figures "
            "(quality independent of --dpi), or 'png pdf' to save both."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Convert learning rate to float when a numeric value is passed as a string.
    if args.learning_rate == "auto":
        learning_rate: str | float = "auto"
    else:
        learning_rate = float(args.learning_rate)

    clip_quantile: float | None = (
        None if args.clip_quantile >= 1.0 else args.clip_quantile
    )

    output_dir: Path = args.output_dir

    for radius in args.radii:
        ecfp_params = dict(DEFAULT_ECFP_PARAMS)
        ecfp_params["radius"] = radius

        radius_output_dir = output_dir / f"radius_{radius}"

        reaction_output_png = radius_output_dir / f"tsne_reaction_ecfp_h{radius}.png"
        reaction_center_output_png = (
            radius_output_dir / f"tsne_reaction_center_ecfp_h{radius}.png"
        )

        if args.save_coords:
            reaction_output_coords = (
                radius_output_dir / f"tsne_reaction_ecfp_h{radius}_coords.npz"
            )
            reaction_center_output_coords = (
                radius_output_dir / f"tsne_reaction_center_ecfp_h{radius}_coords.npz"
            )
        else:
            reaction_output_coords = None
            reaction_center_output_coords = None

        # ---------------------------------------------------------------------
        # 1. t-SNE of reaction ECFPs
        # ---------------------------------------------------------------------
        run_tsne_workflow(
            dataset_names=args.datasets,
            ecfp_params=ecfp_params,
            radius=radius,
            vector_type="reaction",
            vector_label="reaction ECFPs",
            encoding=args.encoding,
            metric=args.metric,
            perplexity=args.perplexity,
            learning_rate=learning_rate,
            init=args.init,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            max_per_dataset=args.max_per_dataset,
            output_png=reaction_output_png,
            output_coords=reaction_output_coords,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
            clip_quantile=clip_quantile,
            min_smi_sub_atoms=args.min_smi_sub_atoms,
            formats=args.formats,
        )

        # ---------------------------------------------------------------------
        # 2. t-SNE of reaction-center ECFPs
        # ---------------------------------------------------------------------
        run_tsne_workflow(
            dataset_names=args.datasets,
            ecfp_params=ecfp_params,
            radius=radius,
            vector_type="reaction_center",
            vector_label="reaction-center ECFPs",
            encoding=args.encoding,
            metric=args.metric,
            perplexity=args.perplexity,
            learning_rate=learning_rate,
            init=args.init,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            max_per_dataset=args.max_per_dataset,
            output_png=reaction_center_output_png,
            output_coords=reaction_center_output_coords,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
            clip_quantile=clip_quantile,
            min_smi_sub_atoms=args.min_smi_sub_atoms,
            formats=args.formats,
        )


if __name__ == "__main__":
    main()