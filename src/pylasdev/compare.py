"""LAS data comparison utilities.

Replaces las_compare.py with Python 3 syntax and proper logging
instead of print statements.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────


def _scalars_equal(a: Any, b: Any) -> bool:
    """Compare two scalars, treating NaN == NaN as equal."""
    # Guard against empty or multi-element ndarrays that would raise
    # ValueError ("ambiguous truth value") on bool(a == b).
    if isinstance(a, np.ndarray):
        if a.size == 0:
            logger.warning(
                "Empty ndarray in scalar comparison; treating as unequal."
            )
            return False
        if a.size > 1:
            logger.warning(
                "Multi-element ndarray in scalar comparison; treating as unequal."
            )
            return False
        a = a.item()
    if isinstance(b, np.ndarray):
        if b.size == 0:
            logger.warning(
                "Empty ndarray in scalar comparison; treating as unequal."
            )
            return False
        if b.size > 1:
            logger.warning(
                "Multi-element ndarray in scalar comparison; treating as unequal."
            )
            return False
        b = b.item()
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if math.isnan(a) and math.isnan(b):
            return True
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        # numpy arrays that pass the size guards can still raise
        # ValueError ("ambiguous truth value") or TypeError on
        # incompatible type comparisons.  Do not swallow other
        # exceptions (KeyboardInterrupt, MemoryError, etc.).
        return False


def _compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    label: str,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two numpy arrays with tolerance."""
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(arr1).__name__,
            type(arr2).__name__,
        )
        return False

    if arr1.shape != arr2.shape:
        logger.warning(
            "Array shape mismatch at '%s': %s vs %s",
            label,
            arr1.shape,
            arr2.shape,
        )
        return False

    # Use np.array_equal for non-numeric dtypes (string, object, void, datetime,
    # timedelta, boolean, complex) where np.allclose would fail or be inappropriate.
    if arr1.dtype.kind in ("U", "S", "O", "V", "M", "m", "b", "c") or arr2.dtype.kind in (
        "U",
        "S",
        "O",
        "V",
        "M",
        "m",
        "b",
        "c",
    ):
        if not np.array_equal(arr1, arr2):
            logger.warning("Array values mismatch at '%s'", label)
            return False
    else:
        # MaskedArray IS-A ndarray — convert before np.allclose so masked
        # values are filled with NaN (not compared at face value).
        if isinstance(arr1, np.ma.MaskedArray):
            arr1 = arr1.astype(np.float64).filled(np.nan)  # type: ignore[attr-defined]  # MaskedArray.astype preserves mask
        if isinstance(arr2, np.ma.MaskedArray):
            arr2 = arr2.astype(np.float64).filled(np.nan)  # type: ignore[attr-defined]  # MaskedArray.astype preserves mask
        try:
            if not np.allclose(
                arr1, arr2, rtol=rtol, atol=atol, equal_nan=True
            ):
                logger.warning("Array values mismatch at '%s'", label)
                return False
        except (ValueError, TypeError):
            # Defense-in-depth for incompatible broadcast shapes that
            # survive the shape guard above.
            logger.warning("Array values mismatch at '%s'", label)
            return False

    return True


# ──────────────────────────────────────────────────────────────
# Single type-dispatch hub: replaces _compare_values + all
# duplicated isinstance cascades.
# ──────────────────────────────────────────────────────────────


def _coerce_and_compare(
    a: Any, b: Any, label: str, rtol: float, atol: float
) -> bool:
    """Compare two values with tolerance-aware dispatch.

    Handles numpy arrays, lists, dicts, and scalars at any nesting depth.
    All isinstance checks are **symmetric** — both operands are checked.
    """
    # ── Phase 1: Normalize 0-d ndarrays to scalars ──
    # 0-d ndarrays are genuine arrays but represent scalar values.
    # Coerce them BEFORE type-dispatch so they follow the scalar path.
    # F-42: .item() ignores the mask on MaskedArray/MaskedConstant,
    # returning the raw underlying data even when the value is marked
    # as invalid.  Check the mask first and preserve masked semantics.
    if isinstance(a, np.ndarray) and a.ndim == 0:
        if np.ma.is_masked(a):
            a = np.ma.masked
        else:
            a = a.item()
    if isinstance(b, np.ndarray) and b.ndim == 0:
        if np.ma.is_masked(b):
            b = np.ma.masked
        else:
            b = b.item()

    # ── Phase 2: Symmetric type-dispatch ──
    # AND-checks (both operands match a type) come before OR-checks
    # (one operand matches) so valid pairs aren't flagged as mismatches.

    # ndarray x ndarray (after 0-d coercion — both are true arrays)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return _compare_arrays(a, b, label, rtol, atol)

    # Single-side ndarray (type mismatch after 0-d coercion)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False

    # dict x dict — compare key-by-key recursively
    # MUST come before list check: dict passes isinstance(x, Iterable)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            only_a = set(a.keys()) - set(b.keys())
            only_b = set(b.keys()) - set(a.keys())
            if only_a:
                logger.warning(
                    "Keys only in first dict at '%s': %s", label, only_a
                )
            if only_b:
                logger.warning(
                    "Keys only in second dict at '%s': %s", label, only_b
                )
            return False
        for k in a:
            if not _coerce_and_compare(
                a[k], b[k], f"{label}.{k}", rtol, atol
            ):
                return False
        return True

    # Single-side dict (type mismatch)
    if isinstance(a, dict) or isinstance(b, dict):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False

    # list x list — delegate to _compare_lists
    if isinstance(a, list) and isinstance(b, list):
        return _compare_lists(a, b, label, rtol, atol)

    # Single-side list (type mismatch)
    if isinstance(a, list) or isinstance(b, list):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False

    # scalar x scalar — all remaining types
    return _scalars_equal(a, b)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def compare_las_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> bool:
    """Compare two LAS data dictionaries for equality.

    Each key-value pair is delegated to ``_coerce_and_compare``, which
    handles all type combinations (ndarray, list, dict, scalar) at any
    nesting depth.  ``data_sections`` receives a dedicated structural
    validator that preserves the richer labeling from the original
    implementation.

    Args:
        dict1: First LAS data dictionary.
        dict2: Second LAS data dictionary.
        rtol: Relative tolerance for numpy array comparison.
        atol: Absolute tolerance for numpy array comparison.

    Returns:
        True if the dictionaries are equivalent, False otherwise.
    """
    if set(dict1.keys()) != set(dict2.keys()):
        only_in_first = set(dict1.keys()) - set(dict2.keys())
        only_in_second = set(dict2.keys()) - set(dict1.keys())
        if only_in_first:
            logger.warning("Keys only in first dict: %s", only_in_first)
        if only_in_second:
            logger.warning("Keys only in second dict: %s", only_in_second)
        return False

    for key in dict2:
        val1, val2 = dict1[key], dict2[key]

        # data_sections gets a dedicated structural validator that handles
        # per-section dict comparisons with richer labeling.  All other
        # list-valued keys are handled by _coerce_and_compare's list branch.
        if key == "data_sections":
            if not _compare_data_sections(val1, val2, rtol, atol):
                return False
        elif not _coerce_and_compare(val1, val2, key, rtol, atol):
            return False

    return True


# ──────────────────────────────────────────────────────────────
# List & data_sections helpers
# ──────────────────────────────────────────────────────────────


def _compare_lists(
    l1: Any,
    l2: Any,
    label: str,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two lists that may contain numpy arrays.

    Tries direct list equality first.  When numpy arrays inside the list
    cause ValueError/TypeError (ambiguous truth value), falls back to
    per-element comparison using ``_coerce_and_compare``.
    """
    try:
        # Check for NaN values before shortcut comparison — NaN != NaN
        # evaluates True without raising ValueError/TypeError, bypassing
        # per-element comparison that correctly handles NaN==NaN.
        def _has_nan(obj: Any) -> bool:
            """Check if obj or any nested element is NaN."""
            if isinstance(obj, (float, np.floating)):
                return obj != obj
            if isinstance(obj, list):
                return any(_has_nan(x) for x in obj)
            if isinstance(obj, dict):
                return any(_has_nan(v) for v in obj.values())
            return False

        if _has_nan(l1) or _has_nan(l2):
            raise ValueError  # Route to per-element comparison
        if l1 != l2:
            logger.warning(
                "List mismatch at '%s': %r vs %r", label, l1, l2
            )
            return False
    except (ValueError, TypeError):
        if not isinstance(l1, list):
            logger.warning(
                "Type mismatch at '%s': %s vs list",
                label,
                type(l1).__name__,
            )
            return False
        if len(l1) != len(l2):
            logger.warning(
                "List length mismatch at '%s': %d vs %d",
                label,
                len(l1),
                len(l2),
            )
            return False
        for idx, (a, b) in enumerate(zip(l1, l2, strict=False)):
            try:
                if not _coerce_and_compare(
                    a, b, f"{label}[{idx}]", rtol, atol
                ):
                    return False
            except (ValueError, TypeError):
                return False
    return True


def _compare_data_sections(
    sections1: list[dict[str, Any]],
    sections2: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> bool:
    """Compare data_sections lists element-wise.

    Validates list structure (length, key sets) then delegates every
    per-key comparison to ``_coerce_and_compare``.  Kept as a thin
    wrapper to preserve data_sections-level labeling and structural
    guards that differ from generic list comparison.
    """
    if not isinstance(sections1, list):
        logger.warning(
            "Type mismatch in data_sections: %s vs list",
            type(sections1).__name__,
        )
        return False
    if len(sections1) != len(sections2):
        logger.warning(
            "data_sections length mismatch: %d vs %d",
            len(sections1),
            len(sections2),
        )
        return False

    for i, (ds1, ds2) in enumerate(
        zip(sections1, sections2, strict=False)
    ):
        if set(ds1.keys()) != set(ds2.keys()):
            only_in_first = set(ds1.keys()) - set(ds2.keys())
            only_in_second = set(ds2.keys()) - set(ds1.keys())
            if only_in_first:
                logger.warning(
                    "Keys 'data_sections[%d]' only in first: %s",
                    i,
                    only_in_first,
                )
            if only_in_second:
                logger.warning(
                    "Keys 'data_sections[%d]' only in second: %s",
                    i,
                    only_in_second,
                )
            return False

        for k in ds2:
            if not _coerce_and_compare(
                ds1[k], ds2[k], f"data_sections[{i}].{k}", rtol, atol
            ):
                return False

    return True
