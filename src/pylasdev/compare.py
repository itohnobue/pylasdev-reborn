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


def _allclose_symmetric(
    arr1: np.ndarray,
    arr2: np.ndarray,
    rtol: float,
    atol: float,
) -> bool:
    """Symmetric tolerance comparison of two numeric arrays.

    ``np.allclose`` references the *second* operand for the relative
    tolerance (``|a - b| <= atol + rtol * |b|``), which makes
    ``compare_las_dicts(d1, d2)`` disagree with
    ``compare_las_dicts(d2, d1)`` at the tolerance boundary.  Reference
    the larger magnitude instead so the result is order-independent:

        |a - b| <= atol + rtol * max(|a|, |b|)

    NaN is treated as equal to NaN and ``inf`` to ``inf`` (with the same
    sign) — matching ``np.allclose(..., equal_nan=True)`` semantics.

    Integer operands are promoted to float64 before the diff/abs/tol
    computation (mirroring ``np.isclose``'s own promotion via
    ``np.result_type(y, 1.)``): native-dtype int64 subtraction/abs wraps
    in two's complement for large magnitudes (``abs(-2**63)`` stays
    negative), which would otherwise produce wrong "equal" verdicts for
    far-apart integer values (F-17).
    """
    if arr1.dtype.kind in "iu":
        arr1 = arr1.astype(np.float64)
    if arr2.dtype.kind in "iu":
        arr2 = arr2.astype(np.float64)
    with np.errstate(invalid="ignore"):
        # Only positions where BOTH operands are finite participate in
        # the tolerance check; non-finite positions are matched below.
        both_finite = np.isfinite(arr1) & np.isfinite(arr2)
        diff = np.abs(arr1 - arr2)
        tol = atol + rtol * np.maximum(np.abs(arr1), np.abs(arr2))
        close = (diff <= tol) & both_finite
    equal_nan = np.isnan(arr1) & np.isnan(arr2)
    equal_pos_inf = np.isposinf(arr1) & np.isposinf(arr2)
    equal_neg_inf = np.isneginf(arr1) & np.isneginf(arr2)
    return bool(np.all(close | equal_nan | equal_pos_inf | equal_neg_inf))


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
            if not _allclose_symmetric(arr1, arr2, rtol, atol):
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


def _has_nan(obj: Any) -> bool:
    """Check if obj or any nested element is NaN or masked.

    Masks are treated as NaN-equivalent because masked values are
    filled with NaN before comparison (``_compare_arrays``) and then
    matched via equal_nan=True.
    """
    if isinstance(obj, (float, np.floating)):
        return obj != obj
    if isinstance(obj, np.ndarray):
        # Descend into array elements: a size-1 NaN array would otherwise
        # slip past the guard and be compared by direct equality, where
        # NaN != NaN makes two identical arrays look UNEQUAL (M-30).
        if isinstance(obj, np.ma.MaskedArray):
            mask = obj.mask
            if mask is not np.ma.nomask and np.any(mask):
                return True
        if obj.dtype.kind in "fiu":
            try:
                return bool(np.isnan(obj).any())
            except TypeError:
                return False
        return False
    if isinstance(obj, list):
        return any(_has_nan(x) for x in obj)
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    return False


def _list_to_numeric_array(lst: list[Any]) -> np.ndarray | None:
    """Convert a homogeneous numeric list to a float64 ndarray.

    All elements must be real numbers (int/float/np.number/np.bool_),
    numeric ndarrays, or masked values (filled with NaN).  Returns None
    for empty lists and lists containing non-numeric elements (strings,
    dicts, tuples, None, ragged list-of-arrays, ...) so callers fall
    back to element-wise comparison.
    """
    if not lst:
        return None
    for item in lst:
        if isinstance(item, np.ndarray):
            # MaskedArray IS-A ndarray (including the masked scalar
            # singleton): masked elements are filled with NaN below;
            # numeric dtype required either way.
            if item.dtype.kind not in "fiu":
                return None
        elif isinstance(item, (int, float, np.integer, np.floating, np.bool_)):
            continue
        else:
            return None
    # Fill masked elements with NaN explicitly (avoids numpy's
    # "converting a masked element to nan" warning and mirrors the
    # MaskedArray handling in _compare_arrays).
    cleaned = [
        np.ma.filled(item, np.nan) if isinstance(item, np.ma.MaskedArray) else item
        for item in lst
    ]
    try:
        return np.asarray(cleaned, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _compare_lists(
    l1: Any,
    l2: Any,
    label: str,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two lists that may contain numpy arrays.

    Homogeneous numeric lists are compared with ``np.allclose`` via
    ``_compare_arrays`` so rtol/atol apply consistently with the ndarray
    path (M-58) — including size-1 ndarray elements and NaN/masked
    elements (M-30, M-33).  Other lists fall back to direct equality,
    routing to per-element comparison when numpy ambiguity or NaN
    requires it.
    """
    arr1 = _list_to_numeric_array(l1)
    arr2 = _list_to_numeric_array(l2)
    if arr1 is not None and arr2 is not None:
        return _compare_arrays(arr1, arr2, label, rtol, atol)

    try:
        # Check for NaN values before shortcut comparison — NaN != NaN
        # evaluates True without raising ValueError/TypeError, bypassing
        # per-element comparison that correctly handles NaN==NaN.
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
    # E-08: Symmetric type check.  Previously only sections1 was checked —
    # sections2=None raised a bare TypeError on len(), and sections2={}
    # (a non-list empty container) silently compared equal to [].
    if not isinstance(sections2, list):
        logger.warning(
            "Type mismatch in data_sections: %s vs %s",
            type(sections1).__name__,
            type(sections2).__name__,
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
        # E-08: Per-element type checks.  Previously a non-dict element
        # (e.g. a str) raised a bare AttributeError on .keys().
        if not isinstance(ds1, dict) or not isinstance(ds2, dict):
            logger.warning(
                "Type mismatch at 'data_sections[%d]': %s vs %s",
                i,
                type(ds1).__name__,
                type(ds2).__name__,
            )
            return False
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
