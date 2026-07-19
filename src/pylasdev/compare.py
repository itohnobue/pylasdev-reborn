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
    except Exception:
        return False


def compare_las_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> bool:
    """Compare two LAS data dictionaries for equality.

    The high branch count (23) is inherent to the function's purpose:
    it must handle every type that appears in LAS dicts — numpy arrays
    (with tolerance comparison), nested dicts, lists (with special
    handling for data_sections containing numpy arrays), and scalars —
    each requiring distinct comparison logic.  Breaking this into
    separate functions for each type would scatter the comparison
    protocol and make it harder to follow the control flow.

    Args:
        dict1: First LAS data dictionary.
        dict2: Second LAS data dictionary.
        rtol: Relative tolerance for numpy array comparison.
        atol: Absolute tolerance for numpy array comparison.

    Returns:
        True if the dictionaries are equivalent, False otherwise.
    """
    # Check for key equality using set operations
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

        if isinstance(val2, dict):
            # Check for key equality in nested dicts using set operations
            if isinstance(val1, dict):
                if set(val1.keys()) != set(val2.keys()):
                    only_in_first = set(val1.keys()) - set(val2.keys())
                    only_in_second = set(val2.keys()) - set(val1.keys())
                    if only_in_first:
                        logger.warning("Keys '%s'.%s only in first dict", key, only_in_first)
                    if only_in_second:
                        logger.warning("Keys '%s'.%s only in second dict", key, only_in_second)
                    return False

            # F-27: When val2 is a dict but val1 is not, detect the type
            # mismatch before the inner loop begins.  Without this guard,
            # an empty dict val2 produces zero inner-loop iterations and
            # the mismatch is silently missed.
            if not isinstance(val1, dict):
                logger.warning(
                    "Type mismatch at '%s': %s vs dict",
                    key,
                    type(val1).__name__,
                )
                return False

            for in_key in val2:
                if isinstance(val1, dict) and in_key not in val1:
                    logger.warning("Key '%s.%s' not found in first dict", key, in_key)
                    return False

                # Guard: val1 may not be subscriptable with val2's keys
                # (e.g., val1 is str/int/float but val2 is a dict with ndarray children).
                try:
                    if isinstance(val2[in_key], np.ndarray):
                        if not _compare_arrays(
                            val1[in_key],
                            val2[in_key],
                            key,
                            in_key,
                            rtol,
                            atol,
                        ):
                            return False
                    elif isinstance(val1[in_key], np.ndarray):
                        # val1 has array but val2 doesn't — type mismatch
                        logger.warning(
                            "Type mismatch at '%s.%s': %s vs %s",
                            key,
                            in_key,
                            type(val1[in_key]).__name__,
                            type(val2[in_key]).__name__,
                        )
                        return False
                    elif not _scalars_equal(val1[in_key], val2[in_key]):
                        logger.warning(
                            "Mismatch at '%s.%s': %r vs %r",
                            key,
                            in_key,
                            val1[in_key],
                            val2[in_key],
                        )
                        return False
                except (TypeError, IndexError, KeyError):
                    logger.warning(
                        "Type mismatch at '%s.%s': cannot compare %s with %s",
                        key,
                        in_key,
                        type(val1).__name__,
                        type(val2).__name__,
                    )
                    return False

        elif isinstance(val2, np.ndarray):
            if not _compare_arrays(val1, val2, key, None, rtol, atol):
                return False

        elif isinstance(val2, list):
            # F-10: data_sections contains numpy arrays that can't
            # be compared with generic list equality (ambiguity error).
            if key == "data_sections":
                if not _compare_data_sections(val1, val2, rtol, atol):
                    return False
            else:
                if not _compare_lists(val1, val2, key, rtol, atol):
                    return False
        else:
            # F-I2-M24: When val2 is a scalar type but val1 is a multi-element
            # ndarray, _scalars_equal would raise ValueError ("ambiguous truth
            # value"). Guard against this asymmetric dispatch.
            if isinstance(val1, np.ndarray) and val1.size > 1:
                logger.warning(
                    "Type mismatch at '%s': ndarray vs %s",
                    key,
                    type(val2).__name__,
                )
                return False
            if not _scalars_equal(val1, val2):
                logger.warning("Mismatch at '%s': %r vs %r", key, val1, val2)
                return False

    return True


def _compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    key: str,
    in_key: str | None,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two numpy arrays with tolerance."""
    label = f"{key}.{in_key}" if in_key else key

    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(arr1).__name__,
            type(arr2).__name__,
        )
        return False

    if arr1.shape != arr2.shape:
        logger.warning("Array shape mismatch at '%s': %s vs %s", label, arr1.shape, arr2.shape)
        return False

    # F-4: np.allclose() fails on string/object arrays (e.g. string_data).
    # Use np.array_equal for non-numeric dtypes.
    if arr1.dtype.kind in ("U", "S", "O", "V", "M", "m", "b") or arr2.dtype.kind in ("U", "S", "O", "V", "M", "m", "b"):
        if not np.array_equal(arr1, arr2):
            logger.warning("Array values mismatch at '%s'", label)
            return False
    else:
        try:
            if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
                logger.warning("Array values mismatch at '%s'", label)
                return False
        except (ValueError, TypeError):
            # F2-009: np.allclose raises ValueError when arrays have
            # incompatible broadcast shapes (e.g., same .size but
            # different .shape that cannot broadcast).  The .shape
            # guard above catches most cases; this try/except is
            # defense-in-depth for any remaining edge cases.
            logger.warning("Array values mismatch at '%s'", label)
            return False

    return True


def _compare_values(
    a: Any,
    b: Any,
    label: str,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two values, dispatching by type.

    Handles numpy arrays, lists, dicts, and scalars recursively.
    Calls to this function should be wrapped in try/except (ValueError,
    TypeError) for defence-in-depth against numpy ambiguity.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return _compare_arrays(a, b, label, None, rtol, atol)
    if isinstance(a, list) and isinstance(b, list):
        return _compare_lists(a, b, label, rtol, atol)
    if isinstance(a, dict) and isinstance(b, dict):
        # F-M30: _compare_lists per-element fallback previously
        # treated dict elements as scalars, delegating to
        # _scalars_equal which returned False for equal dicts
        # containing numpy arrays (ValueError caught by bare
        # except).  Compare dicts key-by-key recursively.
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
            if not _compare_values(a[k], b[k], f"{label}.{k}", rtol, atol):
                return False
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False
    if isinstance(a, list) or isinstance(b, list):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(a).__name__,
            type(b).__name__,
        )
        return False
    return _scalars_equal(a, b)


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
    per-element comparison using _compare_values which dispatches by
    type — handling ndarrays, nested lists, dicts (F-M30), and scalars.
    """
    try:
        if l1 != l2:
            logger.warning("List mismatch at '%s': %r vs %r", label, l1, l2)
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
                if not _compare_values(
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

    Handles numpy arrays inside nested dicts where generic list
    equality (val1 != val2) would raise "truth value of an array
    is ambiguous" (F-10).
    """
    if len(sections1) != len(sections2):
        logger.warning(
            "data_sections length mismatch: %d vs %d",
            len(sections1),
            len(sections2),
        )
        return False

    for i, (ds1, ds2) in enumerate(zip(sections1, sections2, strict=False)):
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
            v1, v2 = ds1[k], ds2[k]
            if isinstance(v2, np.ndarray):
                if not _compare_arrays(
                    v1,
                    v2,
                    f"data_sections[{i}].{k}",
                    None,
                    rtol,
                    atol,
                ):
                    return False
            elif isinstance(v2, dict):
                # F31: Handle dict values (LAS 3.0 data_sections entries
                # where values are dicts mapping curve names to ndarrays).
                if not isinstance(v1, dict):
                    logger.warning(
                        "Type mismatch at 'data_sections[%d].%s': expected dict, got %s",
                        i,
                        k,
                        type(v1).__name__,
                    )
                    return False
                if set(v1.keys()) != set(v2.keys()):
                    only_in_first = set(v1.keys()) - set(v2.keys())
                    only_in_second = set(v2.keys()) - set(v1.keys())
                    if only_in_first:
                        logger.warning(
                            "Keys 'data_sections[%d].%s' only in first: %s",
                            i,
                            k,
                            only_in_first,
                        )
                    if only_in_second:
                        logger.warning(
                            "Keys 'data_sections[%d].%s' only in second: %s",
                            i,
                            k,
                            only_in_second,
                        )
                    return False
                for in_key in v2:
                    if in_key not in v1:
                        logger.warning(
                            "Key 'data_sections[%d].%s.%s' not found in first dict",
                            i,
                            k,
                            in_key,
                        )
                        return False
                    if isinstance(v2[in_key], np.ndarray):
                        if not _compare_arrays(
                            v1[in_key],
                            v2[in_key],
                            f"data_sections[{i}].{k}.{in_key}",
                            None,
                            rtol,
                            atol,
                        ):
                            return False
                    elif isinstance(v1[in_key], np.ndarray):
                        # v1 has array but v2 doesn't — type mismatch
                        logger.warning(
                            "Type mismatch at 'data_sections[%d].%s.%s': %s vs %s",
                            i,
                            k,
                            in_key,
                            type(v1[in_key]).__name__,
                            type(v2[in_key]).__name__,
                        )
                        return False
                    elif not _scalars_equal(v1[in_key], v2[in_key]):
                        logger.warning(
                            "Mismatch at 'data_sections[%d].%s.%s': %r vs %r",
                            i,
                            k,
                            in_key,
                            v1[in_key],
                            v2[in_key],
                        )
                        return False
            elif isinstance(v2, list):
                if not _compare_lists(
                    v1, v2, f"data_sections[{i}].{k}", rtol, atol
                ):
                    return False
            else:
                # F-I2-M25: When v2 is a scalar but v1 is a multi-element
                # ndarray, raw comparison raises ValueError ("ambiguous truth
                # value"). Guard against this asymmetric dispatch.
                if isinstance(v1, np.ndarray) and v1.size > 1:
                    logger.warning(
                        "Type mismatch at 'data_sections[%d].%s': ndarray vs %s",
                        i,
                        k,
                        type(v2).__name__,
                    )
                    return False
                if not _scalars_equal(v1, v2):
                    logger.warning(
                        "Mismatch at 'data_sections[%d].%s': %r vs %r",
                        i,
                        k,
                        v1,
                        v2,
                    )
                    return False

    return True
