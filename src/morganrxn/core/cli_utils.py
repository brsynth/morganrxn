"""
Small shared helpers for the command-line scripts (data_processing and
paper_results).

These were previously copy-pasted, identically, across several scripts
(`make_ecfp_params`, `parse_radii`, `split_merged_ids`, the `timer` context
manager, ...). Keeping a single definition here avoids the copies drifting
apart. This module only depends on the standard library, so it stays a leaf
import with no risk of circular dependencies.
"""

import time
from typing import List, Optional


# =================================================================================================
# ECFP parameters
# =================================================================================================

def make_ecfp_params(radius: int, fp_size: int, folded: bool, custom: bool) -> dict:
    """Build the ``ecfp_params`` dict understood by molecule/ReactionRules code."""
    return {
        "radius": int(radius),
        "fpSize": int(fp_size),
        "folded": bool(folded),
        "custom": bool(custom),
    }


# =================================================================================================
# CLI list parsing
# =================================================================================================

def parse_csv_list(value: Optional[str]) -> List[str]:
    """Split a comma-separated string into a list of trimmed, non-empty tokens."""
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_radii(radii_value, fallback_radius=None) -> List[int]:
    """
    Parse a comma-separated list of radii into a sorted list of unique ints.

    If ``radii_value`` is empty/None, fall back to ``[fallback_radius]`` when a
    fallback is given, otherwise raise. Example: ``"0,1,2,3,4,5" -> [0, 1, ..., 5]``.
    """
    if radii_value is None or str(radii_value).strip() == "":
        if fallback_radius is None:
            raise ValueError("No valid radius found.")
        return [int(fallback_radius)]

    radii = sorted({int(x.strip()) for x in str(radii_value).split(",") if x.strip()})

    if not radii:
        raise ValueError("No valid radius found.")

    return radii


# =================================================================================================
# ReactionRules ID handling
# =================================================================================================

def split_merged_ids(value) -> List[str]:
    """
    Split IDs merged by ``ReactionRules.drop_duplicates`` on ``'|'``.

    Example: ``"US1__0__split0|US2__7__split0" -> ["US1__0__split0", "US2__7__split0"]``.
    """
    if value is None:
        return []

    value = str(value).strip()

    if value == "":
        return []

    return [x.strip() for x in value.split("|") if x.strip()]


# =================================================================================================
# Array subsetting
# =================================================================================================

def subset_X(X, indices):
    """Row-subset a dense numpy array or a scipy sparse matrix."""
    return X[indices]


# =================================================================================================
# Timing
# =================================================================================================

class timer:
    """Context manager printing the wall-clock duration of a labelled block."""

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
