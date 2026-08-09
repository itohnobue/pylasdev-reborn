"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from .exceptions import LASParseError
from .models import LASFile, WellSection, _GuardedDict

logger = logging.getLogger(__name__)

# Maximum bounds for array allocations to prevent memory exhaustion
# from malformed or malicious files. Overridable by setting the module constant.
MAX_CURVES = 100_000
MAX_DATA_LINES = 10_000_000
# Combined allocation guard: curve_count x data_line_count must not exceed this.
# Individual MAX_CURVES and MAX_DATA_LINES checks alone are insufficient — a file
# with 1K curves x 1M lines (1B elements ≈ 8 GB) passes both guards independently
# but OOMs during np.zeros pre-allocation. Overridable by setting module constant.
MAX_TOTAL_ELEMENTS = 1_000_000_000
# DR-05: Python-object accumulation cap for string data on the LAS 1.2/2.0
# paths.  MAX_TOTAL_ELEMENTS guards numpy array allocation at ~8 B/element
# (~8 GB intent), but each Python str object stored in an object-dtype array
# or list costs ~50-100 B (measured) — a crafted file that passes the element
# guard can amplify to ~50-100 GB.  The LAS 3.0 path already caps string
# values at MAX_TOTAL_ELEMENTS // 12 (_las30_data._MAX_STRING_VALUES); this
# is the equivalent cap for _read_normal / _read_wrapped string accumulation.
# When the cap is exceeded the file is malformed/crafted — reject like the
# other MAX_* guards.  Overridable by setting the module constant.
MAX_STRING_VALUES = MAX_TOTAL_ELEMENTS // 12
# G-18: Per-line token limit to prevent single-line memory DoS via unbounded
# split().  Existing guards (MAX_DATA_LINES, MAX_CURVES, MAX_TOTAL_ELEMENTS)
# are product-based and do not protect against a single line with millions
# of space-separated tokens.  Legitimate LAS files never exceed MAX_CURVES
# tokens per line — this cap matches the curve limit.  Overridable.
# F-MDR-03: Sentinel — resolves to MAX_CURVES at call time.  Setting this
# to an explicit int overrides the default curve-count-based limit.
# Previously was ``= MAX_CURVES`` (import-time snapshot), which caused the
# documented "Overridable" behavior to break when MAX_CURVES is overridden.
MAX_TOKENS_PER_LINE: int | None = None

# E-20: Wrapped-mode pending-buffer trim cadence (in tokens).  The
# ``_read_wrapped`` accumulation protocol keeps consumed tokens in the
# ``pending`` list (an O(1) ``read_idx`` frontier avoids per-flush list
# copies).  Without a trim, every token string stays alive until EOF — a
# crafted WRAP=YES file can retain ~58 GB before the MAX_TOTAL_ELEMENTS
# guard raises.  When ``read_idx`` crosses this threshold the consumed
# prefix is deleted in one O(n) slice (amortized O(1) per token) and the
# frontier resets; the buffer then holds only the unconsumed tail.  The
# read_idx-only bound keeps the O(1)-per-step extraction intact (the
# window between ``read_idx`` and ``len(pending)`` never exceeds a partial
# step plus one line).  Overridable; tests shrink it to exercise trims.
_PENDING_TRIM_THRESHOLD = 1_000_000

# M-06/M-62: int64 representability bounds for {I} integer curves.  The
# int64 storage branch must reject values (data values OR the null
# sentinel) outside this range — numpy int64 array assignment of a larger
# Python int raises OverflowError, which would escape the reader's
# LASParseError-only boundary.  Huge integral sentinels (>= 2^63) route
# to the object-dtype path instead; huge data values are replaced with
# the null sentinel and counted as conversion failures.
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)

# M-10 test seam: optional callback invoked from ``_read_wrapped`` once per
# line (after the periodic trim) while the ``pending`` buffer is live, so
# tests can measure token retention DURING parse.  ``None`` in production —
# the hot loop pays one None-check per line and nothing else.  Tests
# monkeypatch it (same established pattern as _PENDING_TRIM_THRESHOLD).
_read_wrapped_trace_hook: Callable[[list[str]], None] | None = None


def _resolve_max_tokens_per_line() -> int:
    """Return the per-line token limit, resolving from MAX_CURVES if not
    explicitly overridden via MAX_TOKENS_PER_LINE.

    The indirection allows MAX_CURVES to be overridden at runtime
    (documented behavior) and have MAX_TOKENS_PER_LINE follow automatically.
    Users who need a different per-line cap set MAX_TOKENS_PER_LINE directly.
    """
    return MAX_TOKENS_PER_LINE if MAX_TOKENS_PER_LINE is not None else MAX_CURVES


from ._data_section_reader import (  # noqa: E402
    _detect_string_curves,
    _is_ascii_section,
    _is_section_header,
    _iter_ascii_data_lines,
    _log_conversion_failures,
    _split_data_line,
    _split_header_row,
    detect_actual_wrap_from_window,
    is_mnemonic_header_row,
    is_units_header_row,
)
from ._sanitize import (  # noqa: E402
    _is_desanitize_enabled,
    _set_desanitize_enabled,
    desanitize_las_value,
)


def _parse_float_with_d_notation(value_str: str) -> float:
    """Convert a string to float, handling Fortran D-notation.

    Some scientific software writes numbers in D-exponent format
    (e.g., ``"-999.25D0"``) instead of standard E-notation.  Python's
    ``float()`` only understands E/e exponents; this helper replaces
    D/d with E/e before conversion.

    This is a shared helper used by both :func:`_get_null_value` (for
    the NULL sentinel value in the well section) and :func:`_to_finite_float`
    (for data values in the ASCII section).

    Args:
        value_str: String to convert (e.g. ``"-999.25D0"``).

    Returns:
        Float value.

    Raises:
        ValueError: If the string cannot be parsed as a float after
            D→E conversion.
        OverflowError: If the string represents a number whose absolute
            value exceeds the maximum representable finite float.
    """
    return float(value_str.replace("D", "E").replace("d", "e"))


def _desanitize_las_value(
    value: str,
    _enabled: bool | None = None,
    *,
    restore_tilde: bool = False,
) -> str:
    """Reverse the writer's ``_``-prefix escapes (shared helper).

    Thin wrapper over :func:`pylasdev._sanitize.desanitize_las_value` with
    the positional ``(value, _enabled)`` form the IT3-F-01 tests pin
    (II-12).

    E-19: *restore_tilde* is forwarded position-scoped by the data-row
    callers.  The LAS 1.2/2.0 writer DOES emit the M-85 ``_~`` escape for
    a first-column string value starting ``~``+non-letter (it keeps the
    row from being misread as a section header), so the 1.2/2.0 data
    path passes ``restore_tilde=(i == 0)`` exactly like the LAS 3.0 path
    (the wrapper's default stays the fail-safe ``False`` for the header
    and numeric call sites — a genuine external ``_~`` value in any
    other position is preserved; II-13).

    Two cases (matching writer's ``_sanitize_las_value``):

    1. ``value.startswith("#")`` → writer prepends ``_`` → ``"_#..."``
       → reverse: strip the leading ``_``.
    2. ``value.lstrip().startswith("#")`` → writer inserts ``_`` after
       leading whitespace → ``" _#..."`` → reverse: remove the ``_``
       between whitespace and ``#``.

    M11 (mirrors the parser's F-25 fix): Case 2 applies ONLY when the
    ``_#`` is the first non-whitespace content (preceded exclusively by
    leading whitespace) — the writer's actual escape scope.  Internal
    ``" _#"`` content the writer never escapes (e.g. ``"ACME _#Oil Corp"``)
    is preserved unchanged.

    IT3-F-01 (perf): *enabled* hoists the ``_DESANITIZE_ENABLED`` flag
    lookup.  ``read_ascii_data`` caches the thread-local flag once per
    read and passes the cached value through ``_read_normal`` /
    ``_read_wrapped``.  When *enabled* is None (any external caller), fall
    back to the thread-local lookup to preserve the previous behavior.
    """
    return desanitize_las_value(value, restore_tilde=restore_tilde, _enabled=_enabled)


def _get_well_entry_ci(
    well: WellSection | dict[str, str],
    key: str,
    default: str,
) -> str:
    """Case-insensitive lookup in a well section / dict (IT3-THR-01).

    from_dict stores well keys verbatim when ``mnem_base`` is None
    (``_norm_mnem`` identity), so a case-variant key like ``"null"``
    survives construction.  The parser, by contrast, uppercases well
    mnemonics on read (parser.py:1937 area).  A case-sensitive lookup
    for ``"NULL"`` misses the lowercase key, so the writer declares
    NULL=-1 in ~W but fills data rows with the -999.25 default — the
    re-read file's declared NULL then disagrees with its fill cells and
    downstream consumers read fill cells as real data.  This helper makes
    all ``_get_null_value`` call sites (reader, LAS 3.0 reader, both
    writers) agree on the declared sentinel regardless of key case.
    """
    entries = well.entries if isinstance(well, WellSection) else well
    key_upper = key.upper()
    for k, v in entries.items():
        if str(k).upper() == key_upper:
            return v
    return default


def _get_null_value(
    well: WellSection | dict[str, str],
    default: str = "-999.25",
    default_float: float = -999.25,
) -> float:
    """Extract null value from well section or return default.

    Shared utility used by parser, data_reader, and writer to avoid
    duplicated try/except blocks across four call sites.

    Args:
        well: WellSection or dict-like object with a .get() method.
        default: String default to look up (e.g. '-999.25').
        default_float: Float fallback when conversion fails.

    Returns:
        Null value as a float.
    """
    try:
        # IT3-THR-01: case-insensitive NULL lookup (see _get_well_entry_ci)
        # so declared NULL and the fill sentinel always agree.
        null_value = _parse_float_with_d_notation(_get_well_entry_ci(well, "NULL", default))
        # F-04: Reject non-finite sentinel values (NaN, Inf, -Inf) which
        # float() accepts without error.  These propagate through numpy
        # arrays → corrupted statistics → writer outputs "nan" (invalid LAS).
        if not np.isfinite(null_value):
            raise LASParseError(f"NULL value must be a finite number, got {null_value!r}")
        return null_value
    except OverflowError:
        # DR-02: Python <=3.12 float('1e400') raises OverflowError instead
        # of returning inf.  Match the F-04 finite-guard behavior below (the
        # 3.13+ path) so the documented LASParseError boundary holds on all
        # supported Pythons; falling back to the default here would diverge
        # per-version (3.12 falls back to the sentinel, 3.13 raises).
        raise LASParseError(
            "NULL value must be a finite number, got a value that overflows the float range"
        ) from None
    except (ValueError, TypeError, AttributeError):
        # F-I2-XPD-05: Log a warning when falling back to the default
        # null value.  Silent fallback to -999.25 with zero diagnostics
        # makes it impossible to distinguish a genuine -999.25 null
        # sentinel from a failed parse.
        warnings.warn(
            f"Could not parse NULL value from well section; "
            f"falling back to default value {default_float:.2f}.",
            UserWarning,
            stacklevel=2,
        )
        return default_float


def _detect_integer_curves(las_file: LASFile) -> set[int]:
    """Return set of curve indices whose CurveDefinition has integer format.

    LAS 3.0's ``{I}`` format specifier declares integer-typed data.  Storing
    such values as float64 silently loses precision above 2^53
    (L-03): ``float("9007199254740993")`` rounds to 9007199254740992.0.
    Parsing via ``int()`` and storing in an int64 array preserves the exact
    value.  This mirrors ``_detect_string_curves`` for the {S} format.

    Args:
        las_file: LASFile with curves and curves_order populated.

    Returns:
        Set of integer indices into curves_order for integer-format curves.
    """
    _integer_curve_indices: set[int] = set()
    for _idx in range(len(las_file.curves_order)):
        if _idx < len(las_file.curves):
            cd = las_file.curves[_idx]
            # Normalize to uppercase for consistent detection (same as
            # _detect_string_curves F-MDR-04).
            if cd.data_format.upper() == "I":
                _integer_curve_indices.add(_idx)
    return _integer_curve_indices


def _to_integer_value(
    value_str: str,
    null_value: float,
    _failure_counter: list[int] | None = None,
    _null_as_float: bool = False,
) -> int | float:
    """Convert string to exact integer for {I}-format curves (L-03).

    The {I} (integer) data format can carry values above 2^53 where float64
    cannot represent every integer exactly (e.g. 9007199254740993).  Parsing
    via ``int()`` preserves the exact value; parsing via ``float()`` rounds
    it to the nearest representable double.  This helper is used only for
    integer-format curves that were allocated with int64 dtype (the caller
    guarantees *null_value* is integral in that case).

    Args:
        value_str: String to convert.  May be empty.
        null_value: Integral null sentinel to return on failure.
        _failure_counter: Optional mutable list; ``_failure_counter[0]`` is
            incremented on each non-trivial conversion failure.
        _null_as_float: When True (EXT-04: {I} curves stored as object
            dtype because the declared NULL is fractional), return the
            fractional *null_value* itself on failure instead of
            ``int(null_value)`` — ``int(-999.25)`` would truncate the
            sentinel, corrupting null cells.

    Returns:
        Exact integer, or ``int(null_value)`` (or *null_value* when
        ``_null_as_float``) on failure.
    """
    if not value_str:
        return null_value if _null_as_float else int(null_value)
    try:
        # int() preserves exactness for plain integer tokens.  This is the
        # common {I} case and the whole point of the dtype branch.
        result = int(value_str)
    except ValueError:
        result = None
    if result is not None:
        # M-06: On the int64 storage path (integral NULL), a parsed value
        # beyond int64 range cannot be assigned to the int64 array —
        # numpy raises OverflowError, which would escape the reader's
        # LASParseError-only boundary.  Route to the null sentinel and
        # count as a conversion failure (the existing summary warning
        # fires) instead of letting the raw exception escape.  The object
        # dtype path (fractional NULL) holds arbitrary Python ints, so no
        # bound applies there.
        if not _null_as_float and not (_INT64_MIN <= result <= _INT64_MAX):
            if _failure_counter is not None:
                _failure_counter[0] += 1
            return int(null_value)
        return result
    # Fall back to D-notation / float parsing for non-plain tokens.  Only
    # accept the result when it is exactly integral — fractional values in
    # an integer column are conversion failures (null), never truncated.
    try:
        val = _parse_float_with_d_notation(value_str)
    except (ValueError, OverflowError):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    if not math.isfinite(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    if val != int(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    # M-06: bound the float-fallback result on the int64 storage path too
    # (e.g. "9.2e18" parsed via float() then int() can exceed int64 max).
    result = int(val)
    if not _null_as_float and not (_INT64_MIN <= result <= _INT64_MAX):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return int(null_value)
    return result


def _to_finite_float(
    value_str: str,
    null_value: float,
    _failure_counter: list[int] | None = None,
) -> float:
    """Convert string to float, replacing non-finite values with null_value.

    Python's ``float()`` accepts ``\"nan\"``, ``\"inf\"``, ``\"-inf\"`` and
    overflow exponents (e.g. ``\"1e309\"``) without error.  These non-finite
    values corrupt downstream numpy computations (NaN propagation, Inf
    making statistics invalid).  This helper catches them and returns
    *null_value* instead.

    Also handles empty strings and non-numeric strings gracefully.

    Args:
        value_str: String to convert.  May be empty.
        null_value: Value to return when conversion fails or result is
            non-finite.
        _failure_counter: Optional mutable list; ``_failure_counter[0]`` is
            incremented on each non-trivial conversion failure (non-empty
            input that could not be parsed as a finite float).  Used by
            callers to surface diagnostic counts.

    Returns:
        A finite float, or *null_value*.
    """
    if not value_str:
        return null_value
    try:
        val = _parse_float_with_d_notation(value_str)
    except (ValueError, OverflowError):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value
    # IT3-F-02: math.isfinite is ~15x faster than np.isfinite for Python
    # float scalars and semantically identical for scalars (verified: no
    # NaN-propagation divergence).  This helper is called once per data
    # value on every read path (data_reader, _las30_data, dev_reader via
    # _to_finite_float), so the swap has measurable end-to-end impact.
    if not math.isfinite(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value
    return val


def read_ascii_data(
    lines: list[str], las_file: LASFile, data_line_count: int, desanitize: bool = True
) -> None:
    """Read the ~A (ASCII data) section and populate las_file.logs.

    Args:
        lines: File content split into lines (pre-split by reader.py
            for efficiency — eliminates redundant content.splitlines()).
        las_file: LASFile object with curves_order already populated.
        data_line_count: Number of data lines (from pre-scan).
        desanitize: If True (default), strip writer escape prefixes
            (_# → #) for roundtrip correctness.  Set to False when
            reading files NOT produced by pylasdev's writer to avoid
            corrupting genuine _#-prefixed data.
    """
    if len(las_file.curves_order) == 0:
        # F32: Warn when data is present but no curves are defined,
        # so malformed files don't silently lose data.
        warnings.warn(
            "ASCII data section found but no curves are defined. Data has been discarded.",
            UserWarning,
            stacklevel=2,
        )
        return

    # F-ITER2-SEC-M02 + F-ITER2-SEC-M03: Deduplicate curves BEFORE bounds
    # checks and wrap detection.  Previously, _detect_actual_wrap and the
    # outer MAX_TOTAL_ELEMENTS check used pre-dedup curve_count, causing
    # false wrap detection and false rejection for files with duplicate
    # curves near the boundary.  _read_normal and _read_wrapped also call
    # _deduplicate_curves internally; the second call is a no-op when
    # curves are already unique.
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # F-M01: Use ``>=`` (not ``>``) for consistency with parser.py and models.py
    # which already use ``>=`` for all MAX_CURVES guards.  Using ``>`` here
    # allows exactly MAX_CURVES curves while from_dict rejects at the limit —
    # a roundtrip divergence where a data_reader-accepted file is rejected by
    # from_dict.  Synchronizing on ``>=`` closes the gap.
    if curve_count >= MAX_CURVES:
        raise LASParseError(
            f"Curve count ({curve_count}) exceeds maximum allowed ({MAX_CURVES}). "
            f"The file may be malformed or corrupt."
        )

    # F-D2-M03: Removed the outer MAX_TOTAL_ELEMENTS check that ran before
    # wrap detection.  For WRAP=YES files, data_line_count counts individual
    # lines (not depth steps), so a wrapped file with 100 curves and 10M
    # lines (100K depth steps) would falsely be rejected as 1B elements
    # when it's actually ~10M.  Both _read_normal and _read_wrapped
    # have their own wrapped-aware bounds checks.  For WRAP=NO, this
    # simply eliminates a redundant check (the same bounds guard runs
    # inside _read_normal).

    # F-212: Route the desanitize parameter to the unified thread-local
    # flag in _sanitize.py so _desanitize_las_value and
    # parser._desanitize_las_value share the same module-level state.
    # E-04: Save the prior thread-local value and restore it in a finally
    # block.  The flag is thread-local (F-21/F-088); an unconditional
    # reset-to-True would clobber another caller's value on the same
    # thread.  Without the restore, a desanitize=False read left the flag
    # False for the whole thread, silently changing the behavior of
    # subsequent direct LASParser.parse()/read_ascii_data() users.
    _prev_desanitize = _is_desanitize_enabled()
    _set_desanitize_enabled(desanitize)
    try:
        delimiter = las_file.version.delimiter_char

        # F-H01: Unconditionally detect actual wrap mode from data content
        # regardless of the WRAP header flag.  The header may be wrong
        # (e.g. mislabeled Petrel exports claim WRAP=YES but data is
        # non-wrapped, or files with WRAP=NO actually contain wrapped data).
        # Using the wrong reader corrupts all non-depth curve values:
        #   - _read_normal on wrapped data  → extra columns discarded,
        #     only first value per line used, rest become null_value
        #   - _read_wrapped on non-wrapped  → depth/C1 swap via the
        #     depth-line flag protocol treating full lines as single values
        actual_wrap = _detect_actual_wrap(
            lines,
            curve_count,
            delimiter,
            declared_wrap=las_file.version.wrap,
            las_file=las_file,
        )
        # IT3-F-01: The desanitize flag is constant for this read (E-04
        # sets it above and restores it in finally).  Cache it once here
        # and thread it through the per-value loops instead of re-running
        # the thread-local lookup per value.
        _desanitize_enabled = _is_desanitize_enabled()
        if actual_wrap:
            _read_wrapped(
                lines,
                las_file,
                curve_count,
                delimiter,
                _desanitize_enabled=_desanitize_enabled,
            )
        else:
            _read_normal(
                lines,
                las_file,
                curve_count,
                data_line_count,
                delimiter,
                _desanitize_enabled=_desanitize_enabled,
            )
    finally:
        _set_desanitize_enabled(_prev_desanitize)


def _detect_actual_wrap(
    lines: list[str],
    curve_count: int,
    delimiter: str = " ",
    declared_wrap: str | None = None,
    las_file: LASFile | None = None,
) -> bool:
    """Detect if data is actually wrapped by checking the first data lines.

    Thin per-caller wrapper: keeps the LAS 1.2/2.0 ``~A`` section scan
    (window collection stays caller-side per INT-02 — the LAS 3.0 twin
    collects from pre-scoped section lines instead) and delegates the
    decision to the shared core
    :func:`pylasdev._data_section_reader.detect_actual_wrap_from_window`.
    ``empty_window_default=True`` (conservative — test_detect_wrap_no_data
    locks empty-window → wrapped on this path; W-4/INT-01).

    Args:
        lines: File content split into lines.
        curve_count: Number of curves declared in ~C section.
        delimiter: Data column delimiter character (default space).
            Uses DLM-aware splitting when delimiter is not a space.
        declared_wrap: Optional declared WRAP header value ("YES"/"NO").
            Used as a tiebreak when the data window is genuinely ambiguous
            (2-2 or 1-1 split).  None means the header is unavailable —
            default to wrapped (conservative).
        las_file: Optional LASFile for the mnemonic-header match set
            (E-41).  When given, standalone mnemonic header rows are
            excluded from the window so a full-width header cannot flip a
            genuinely wrapped file to non-wrapped.  When None (direct
            detector callers/tests) no header filtering is applied.

    Returns:
        True if data is actually wrapped, False if non-wrapped despite header.
    """
    in_ascii = False
    window: list[int] = []  # value counts of up to 4 data lines
    # E-41: hoist the mnemonic-header match set + all-string clause once
    # (only when a LASFile is available).  The header-skip applies ONLY
    # while the window is empty — the first line(s) of the section, the
    # only position where a standalone mnemonic header can appear (DR-M3).
    _mnemonic_declared = _mnemonic_header_declared(las_file) if las_file is not None else None
    _all_string = (
        len(_detect_string_curves(las_file)) == curve_count if las_file is not None else None
    )
    for line in lines:
        stripped = line.strip()

        if _is_section_header(stripped):
            # P-16: Treat every genuine ~-prefixed section header (recognized
            # OR unrecognized) as a section boundary, matching the parser's
            # section-boundary classification.  F-I2-XPD-01 retained: only
            # genuine ~-prefixed section-like lines (checked by
            # _is_section_header) terminate the block; control-char noise
            # (~3D, ~., ~#) fails that check and is skipped below.
            if _is_ascii_section(stripped):
                in_ascii = True
            elif in_ascii:
                # F-048: Standardize section-detection guard with
                # _iter_ascii_data_lines — when we encounter a genuine
                # non-~A section header while inside an ~A block, exit the
                # loop (we've left the data section).
                break
            continue

        # P-16: ~-prefixed lines that are NOT section headers (e.g. ~3D,
        # ~., ~#, control-character replacement artifacts) are not data
        # lines — the parser routes them to other_lines.  Skip them.
        if stripped.startswith("~"):
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # E-41: a standalone mnemonic header row is a column header, not a
        # data line — counting it as a full-width data line lets the header
        # flip genuinely wrapped ≥3-curve data to non-wrapped (silent column
        # shift of the whole depth log).  Skip it while the window is empty
        # so the window counts DATA lines only (mirrors _read_normal's
        # current_line == 0 header gate; the LAS 3.0 twin never sees header
        # rows — the parser pre-filters them).
        if (
            not window
            and _mnemonic_declared is not None
            and is_mnemonic_header_row(
                _split_header_row(stripped),
                declared=_mnemonic_declared,
                curve_count=curve_count,
                all_string=bool(_all_string),
            )
        ):
            continue

        # Data line found — split using DLM-aware split (shared utility).
        values = _split_data_line(stripped, delimiter)

        # F-M20: When curve_count is 1, wrapped and non-wrapped modes are
        # equivalent — every line holds exactly one value regardless of
        # mode.  Return False at the first data line (before the window
        # fills), matching the pre-refactor in-loop guard position.
        if curve_count <= 1:
            return False

        window.append(len(values))
        if len(window) >= 4:
            break

    return detect_actual_wrap_from_window(
        window,
        curve_count,
        declared_wrap,
        empty_window_default=True,
    )


def _deduped_name_order(names: list[str]) -> tuple[list[str], list[tuple[int, str, str]]]:
    """Pure name-rename core of :func:`_deduplicate_curves` (F-M19).

    Computes the post-dedup curve-name order for *names* — appending
    ``_2``, ``_3``... to duplicate names (and F-22 cross-base collisions
    where an original name matches a previously generated ``_N`` suffix)
    exactly like the reader's dedup pass.  Returns the renamed order plus
    the ``(idx, old_name, new_name)`` rename triples so callers can
    either replay the model mutations (:func:`_deduplicate_curves`) or
    build the post-dedup declared set from PRE-dedup state without
    mutating anything (parser._finalize_pre_scan, E-43).

    No model mutation and no warnings — pure name computation.
    """
    seen: dict[str, int] = {}
    new_order: list[str] = []
    renames: list[tuple[int, str, str]] = []
    output_names: set[str] = set()
    for idx, name in enumerate(names):
        if name in seen:
            seen[name] += 1
            new_name = _resolve_unique_curve_name(name, seen[name], output_names)
            # Update the seen counter to match the actual suffix used
            seen[name] = _suffix_from_name(new_name, name)
            renames.append((idx, name, new_name))
            new_order.append(new_name)
            output_names.add(new_name)
        elif name in output_names:
            # F-22: cross-base collision — an original name matches a
            # previously generated _N suffix.  Input ["DEPT","DEPT","DEPT_2"]
            # should produce ["DEPT","DEPT_2","DEPT_2_2"], not
            # ["DEPT","DEPT_2","DEPT_2"] with duplicate keys.
            new_name = _resolve_unique_curve_name(name, 2, output_names)
            seen[name] = _suffix_from_name(new_name, name)
            renames.append((idx, name, new_name))
            new_order.append(new_name)
            output_names.add(new_name)
        else:
            seen[name] = 1
            new_order.append(name)
            output_names.add(name)
    return new_order, renames


def _deduplicate_curves(las_file: LASFile, _stacklevel: int = 2) -> None:
    """Detect and rename duplicate curve names with warning.

    Appends _2, _3, etc. to duplicate mnemonics so each curve gets
    its own array in las_file.logs. Also updates the corresponding
    CurveDefinition objects to keep curves_order and curves in sync.

    The rename DECISIONS come from the shared pure core
    :func:`_deduped_name_order` (E-17/E-43 — the parser's pre-scan
    finalize simulates the same algorithm on pre-dedup state); this
    function replays the renames over the model with a warning per
    renamed curve.

    Args:
        las_file: LASFile to deduplicate curves in.
        _stacklevel: Stacklevel for warnings.warn (default 2 points
            to the immediate caller; pass 3 when called from deeper
            call chains such as parser._process_ascii_data).
    """
    new_order, renames = _deduped_name_order(las_file.curves_order)
    if not renames:
        return
    # Replay the renames: each renamed curve gets its pre-dedup name as
    # original_mnemonic (when empty) and one dedup warning.  The scratch
    # accumulators are irrelevant post-decision — _rename_duplicate_curve
    # requires them by signature only.
    _scratch_order: list[str] = []
    _scratch_names: set[str] = set()
    for idx, old_name, new_name in renames:
        _rename_duplicate_curve(
            las_file,
            idx,
            old_name,
            new_name,
            _scratch_order,
            _scratch_names,
            _stacklevel,
        )
    las_file.curves_order = new_order


def _resolve_unique_curve_name(
    base_name: str,
    start_suffix: int,
    output_names: set[str],
) -> str:
    """Resolve a unique curve name by appending incrementing _N suffix.

    F-M19: Extracted helper for the deduplication collision-resolution
    algorithm shared across parser.py, data_reader.py, and dev_reader.py.

    Args:
        base_name: The original curve mnemonic.
        start_suffix: Starting integer for the suffix (e.g. 2 for first
            duplicate, or higher for already-seen names).
        output_names: Set of names already in the output (for collision
            detection against previously-generated _N suffixes).

    Returns:
        A unique name of the form ``f"{base_name}_{N}"`` that does not
        collide with any name in *output_names*.
    """
    suffix = start_suffix
    new_name = f"{base_name}_{suffix}"
    while new_name in output_names:
        suffix += 1
        new_name = f"{base_name}_{suffix}"
    return new_name


def _suffix_from_name(name: str, base_name: str) -> int:
    """Extract the _N suffix from a resolved unique curve name.

    Given ``name = "DEPT_3"`` and ``base_name = "DEPT"``, returns 3.
    Used to keep the ``seen`` counter in sync with the actual suffix
    assigned by :func:`_resolve_unique_curve_name`.
    """
    try:
        return int(name[len(base_name) + 1 :])
    except (ValueError, IndexError):
        return 0


def _rename_duplicate_curve(
    las_file: LASFile,
    idx: int,
    name: str,
    new_name: str,
    new_order: list[str],
    output_names: set[str],
    _stacklevel: int = 2,
) -> None:
    """Rename a duplicate curve with warning and keep CurveDefinition in sync."""
    # N-I-30: The old warning claimed the FILE contained repeated curve
    # names, which is misleading when the duplicates were produced by
    # mnem_base resolving DISTINCT vendor mnemonics (e.g. "LLD" and "LLS"
    # both → "BK" → "BFV") to the same canonical name.  Mention both
    # causes so the message is accurate for mnem_base-active reads.
    warnings.warn(
        f"Duplicate curve mnemonic '{name}' renamed to '{new_name}'. "
        "This may be caused by mnem_base resolving distinct curve names "
        "to the same canonical name, or by repeated curve names in the "
        "source file.",
        stacklevel=_stacklevel,
    )
    new_order.append(new_name)
    output_names.add(new_name)
    if idx < len(las_file.curves):
        if not las_file.curves[idx].original_mnemonic:
            las_file.curves[idx].original_mnemonic = name
        las_file.curves[idx].mnemonic = new_name


def _declared_mnemonic_set(
    names: Sequence[str], originals: Sequence[str] | None = None
) -> set[str]:
    """Build the mnemonic-header match set for a curve declaration list.

    Single source of truth for the mnemonic-header declared set
    (E-17/E-43 — the reader and the parser's pre-scan finalize share ONE
    code path instead of the historical parallel raw-text mirrors):

    - the reader calls this through :func:`_mnemonic_header_declared`
      with POST-dedup model state (the dedup branch is a no-op on unique
      names — outputs identical to the pre-refactor comprehension);
    - the parser's ``_finalize_pre_scan`` calls it with PRE-dedup model
      state (``curves_order`` + parallel ``original_mnemonic`` values),
      where the dedup branch simulates exactly what the reader's own
      ``_deduplicate_curves`` pass will do before it consumes data.

    The set contains the RESOLVED curve mnemonics (post-dedup order) plus
    each curve's ``original_mnemonic`` — mnem_base-normalized curves keep
    their vendor name (e.g. ``LLD``→``BFV`` with ``original_mnemonic="LLD"``)
    and renamed duplicates keep their pre-dedup name (dedup sets
    ``original_mnemonic`` to the old name only when it was empty —
    :func:`_rename_duplicate_curve`), so a header row written in raw
    vendor or pre-dedup mnemonics is still recognized.

    Args:
        names: Curve names in declaration order (pre- or post-dedup;
            duplicates are resolved exactly like ``_deduplicate_curves``).
        originals: Parallel ``original_mnemonic`` values (empty strings
            when absent).  When a curve is renamed and its original is
            empty, the pre-dedup name is used — reproducing the reader's
            post-dedup ``curves[idx].original_mnemonic`` state.
    """
    renamed, _renames = _deduped_name_order(list(names))
    declared = {name.upper() for name in renamed}
    for idx, name in enumerate(renamed):
        _orig = originals[idx] if originals is not None and idx < len(originals) else ""
        if _orig:
            declared.add(_orig.upper())
        elif name != names[idx]:
            declared.add(names[idx].upper())
    return declared


def _mnemonic_header_declared(las_file: LASFile) -> set[str]:
    """Build the mnemonic-header match set once per read.

    The set contains the RESOLVED curve mnemonics (``curves_order``) plus
    each curve's ``original_mnemonic`` — mnem_base-normalized curves keep
    their vendor name (e.g. ``LLD``→``BFV`` with ``original_mnemonic="LLD"``),
    so a header row written in raw vendor mnemonics is still recognized.
    Mirrors ``parser._is_standalone_mnemonic_header`` (parser.py:3004-3008).

    Must be called AFTER ``_deduplicate_curves`` so ``_2``-suffix renames
    and their ``original_mnemonic`` values are in place.

    E-17/E-43: thin wrapper over the shared :func:`_declared_mnemonic_set`
    with POST-dedup model state (dedup branch no-op on unique names).
    """
    return _declared_mnemonic_set(
        las_file.curves_order,
        [c.original_mnemonic for c in las_file.curves],
    )


def _is_mnemonic_header_row(
    values: list[str],
    las_file: LASFile,
    curve_count: int,
    string_curve_indices: set[int],
    *,
    declared: set[str] | None = None,
) -> bool:
    """M-37/FIX-CONV-2: True when *values* are declared curve mnemonics.

    Thin per-caller wrapper over the shared pure predicate
    :func:`pylasdev._data_section_reader.is_mnemonic_header_row`,
    preserving the current signature/name for the 7 direct test call sites
    (II-8).  Computes the ``all_string`` clause from *string_curve_indices*
    and passes the match set (resolved + original mnemonics, H-02) as
    *declared*.

    LAS 2.0 places curve mnemonics ON the ~A line, but some real-world
    files emit them as a standalone header row immediately after ~A
    (e.g. ``~A\\nDEPT GR\\n1000.0 50.0\\n...``).  Such a row is a column
    header, not a data row: consuming it creates a phantom all-null first
    row and shifts every subsequent value by one column.

    **DR-M3 (first-line-of-section contract):** the predicate is a HEADER
    detector, so callers MUST gate it on the section's first line(s) — the
    only position where a standalone mnemonic header can legitimately
    appear.  ``_read_normal`` calls it only when ``current_line == 0`` and
    ``_read_wrapped`` only when ``total_elements == 0``.

    *declared* is the precomputed match set from :func:`_mnemonic_header_declared`;
    when omitted it is built on demand.  Callers on the hot path hoist it.
    """
    if declared is None:
        declared = _mnemonic_header_declared(las_file)
    return is_mnemonic_header_row(
        values,
        declared=declared,
        curve_count=curve_count,
        all_string=(len(string_curve_indices) == curve_count),
    )


def _read_normal(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    data_line_count: int,
    delimiter: str = " ",
    _desanitize_enabled: bool | None = None,
) -> None:
    """Read non-wrapped ASCII data. One depth step per line."""
    # Deduplicate curve names before allocating arrays
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # F-MDR-01: Use ``>`` for consistency with models.py which uses ``>`` at
    # all 8 MAX_DATA_LINES guard sites (accepts at exactly MAX_DATA_LINES).
    # The F-M01 fix comment previously claimed models.py uses ``>=``, which
    # was incorrect — the divergence was introduced by the F-M01 change.
    if data_line_count > MAX_DATA_LINES:
        raise LASParseError(
            f"Data line count ({data_line_count}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks.
    if curve_count * data_line_count > MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Total allocation ({curve_count} curves x {data_line_count} lines = "
            f"{curve_count * data_line_count} elements) exceeds maximum allowed "
            f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    # F-WXP-01: Detect string curves from CurveDefinition data_format.
    # See _data_section_reader._detect_string_curves for full details.
    _string_curve_indices: set[int] = _detect_string_curves(las_file)
    # L-03: Detect {I} integer-format curves.  These are stored as int64
    # (when the null sentinel is integral) and parsed via int() so values
    # above 2^53 are not silently rounded by float().
    _integer_curve_indices: set[int] = _detect_integer_curves(las_file)
    # FIX-CONV-2: Hoist the mnemonic-header match set (resolved + original
    # mnemonics) once per read — building it per data row would make the
    # hot loop O(rows x curves) for set construction.
    _mnemonic_declared = _mnemonic_header_declared(las_file)
    # DR-05: Count string-curve values stored.  MAX_TOTAL_ELEMENTS accounts
    # numpy allocation at ~8 B/element, but each Python str object costs
    # ~50-100 B — a crafted file can pass the element guard and still
    # allocate ~50-100 GB.  Mirror the LAS 3.0 _MAX_STRING_VALUES cap.
    _string_value_count = 0

    # L-03: The null sentinel must be known BEFORE allocation to decide the
    # dtype.  int64 allocation would truncate a fractional null sentinel
    # (e.g. -999.25 → -999), silently corrupting every null cell — so the
    # int64 branch is only taken when the declared NULL is integral.
    # M-62: the int64 branch must ALSO check int64 representability.  A
    # huge integral sentinel (>= 2^63, e.g. NULL 1e19) passes
    # ``float(x).is_integer()`` but int64 assignment of ``int(null_value)``
    # (failure fills at :900) raises OverflowError that escapes the
    # LASParseError-only boundary.  Route such sentinels to the object
    # dtype path (EXT-04) which holds arbitrary Python ints exactly.
    null_value = _get_null_value(las_file.well)
    _null_is_integral = (
        float(null_value).is_integer() and _INT64_MIN <= int(null_value) <= _INT64_MAX
    )

    # Pre-allocate arrays — string curves go to string_data, numeric to logs.
    # C-505: Ensure curvenames that appear in both string_curve_indices and
    # are allocated mixed-ly do not cause double-entry issues in logs.
    _numeric_curve_indices: list[int] = []
    _string_curve_map: dict[int, str] = {}  # index → curve_name for fast lookup
    for i, curve_name in enumerate(las_file.curves_order):
        if i in _string_curve_indices:
            las_file.string_data[curve_name] = np.full(data_line_count, "", dtype=object)
            _string_curve_map[i] = curve_name
        elif i in _integer_curve_indices:
            if _null_is_integral:
                # L-03: int64 storage for {I} curves — preserves exact
                # integer values above 2^53 that float64 cannot represent.
                las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.int64)
            else:
                # EXT-04: fractional declared NULL — object dtype preserves
                # exact integer values above 2^53 while null cells keep the
                # fractional sentinel (int64 would truncate -999.25 → -999).
                las_file.logs[curve_name] = np.zeros(data_line_count, dtype=object)
            _numeric_curve_indices.append(i)
        else:
            las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)
            _numeric_curve_indices.append(i)

    # FIX-CONV-2: data_line_count is a pre-allocation hint, never a
    # correctness bound.  _allocated tracks the actual array capacity
    # (starts at the pre-scan estimate, grows geometrically on undercount);
    # data_line_count keeps its original meaning for the overcount warning.
    _allocated = data_line_count

    # Pre-extract numeric arrays for fast inner-loop access.
    # String array access is direct via las_file.string_data since we
    # cannot mix dtypes in a homogeneous list.
    curve_arrays = [las_file.logs[name] for name in las_file.curves_order if name in las_file.logs]
    # Build a mapping from logical curve index → numeric array index
    # for the inner loop (string curves are excluded from curve_arrays).
    _logical_to_numeric: dict[int, int] = {}
    for ni, li in enumerate(_numeric_curve_indices):
        _logical_to_numeric[li] = ni

    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    current_line = 0
    # F-I2-XPD-03: Replace boolean-once flags with counters so automated
    # validation tools can enumerate the total number of affected rows.
    # The first occurrence logs full context; subsequent occurrences are
    # counted silently; a summary is logged at the end of the section.
    extra_col_count: int | None = None  # Track extra-column count for summary
    short_row_count: int | None = None  # F-11: Track short-row count for summary
    # I2-02: Track embedded-delimiter string truncation for summary.
    embedded_delim_count: int | None = None
    # M-13: Track whether the standalone mnemonic header row was skipped so
    # an optional units row directly after it can also be skipped (first
    # data line only).
    _mnemonic_header_skipped = False

    for stripped in _iter_ascii_data_lines(lines):
        # Split using DLM-aware split (shared utility).
        # M-30: pass the declared curve count so the comma branch can
        # recombine thousands-separated fragments ("1,234.5") with a loud
        # warning instead of silently mis-assigning columns.
        values = _split_data_line(stripped, delimiter, expected=curve_count)

        # M-37: Skip a standalone mnemonic header row (e.g. "~A\nDEPT GR\n"
        # before the numeric rows).  LAS 2.0 places mnemonics on the ~A
        # line, but a common real-world variant emits them as a separate
        # first row.  Consuming it as a data row produced a phantom
        # all-null first row and shifted every value by one column.  Skip
        # it so it is not counted as a data row.
        # DR-M3: the header check applies ONLY to the first line(s) of the
        # section (current_line == 0).  A standalone mnemonic header can
        # only legitimately appear at the top of the ~A section; applying
        # the predicate to every line made the M12 partial-header
        # relaxation misclassify mid-section all-mnemonic rows (e.g. a
        # ragged "GR ZONE" row) as headers, silently dropping real data.
        if current_line == 0 and _is_mnemonic_header_row(
            _split_header_row(stripped),
            las_file,
            curve_count,
            _string_curve_indices,
            declared=_mnemonic_declared,
        ):
            _mnemonic_header_skipped = True
            continue

        # M-13: Some real-world files emit an optional UNITS row directly
        # after the mnemonic header row (e.g. "~A\nDEPT GR\nM GAPI\n...");
        # consuming it as a data row produced a phantom all-null first row
        # and a one-row shift of the whole depth log.  Skip it — but only
        # when (a) a mnemonic header row was just skipped (the units row
        # can only appear in that position) and (b) we are still on the
        # first data line (current_line == 0).  The shared
        # is_units_header_row predicate is deliberately narrow (letters
        # only) so a genuine first data row is never misclassified; the
        # parser pre-scan (parser._finalize_pre_scan) subtracts the units
        # row itself using the same predicate, so both sides agree
        # on which rows are header/units vs data (M-13 shared contract).
        if (
            current_line == 0
            and _mnemonic_header_skipped
            and is_units_header_row(_split_header_row(stripped))
        ):
            # M-13 (fix4-P1): one-shot — the units row has been consumed.
            # Close the position gate NOW (parser fix3-P1 parity,
            # parser.py:4498): a genuine letters-only first data row that
            # follows within this same section must not be dropped as a
            # units row.
            _mnemonic_header_skipped = False
            continue

        # M-13 (fix4-P1): a data row has been consumed — the units-row
        # position gate is closed (parser fix3-P1 parity, parser.py:4503).
        # The current_line == 0 position gate below already makes the flag
        # inert after the first data row; the explicit reset is parity
        # hygiene against future refactors that relax the position gate.
        if current_line == 0 and _mnemonic_header_skipped:
            _mnemonic_header_skipped = False

        # Warn about extra columns being silently discarded.
        # F-I2-XPD-03: First occurrence logs full context; subsequent
        # occurrences are counted silently; summary logged at end.
        if len(values) > curve_count:
            if extra_col_count is None:
                warnings.warn(
                    f"Data line has {len(values)} values but only {curve_count} curves declared "
                    f"in ~C section. Extra columns are discarded. "
                    f"(Further occurrences will be counted.)",
                    UserWarning,
                    stacklevel=2,
                )
                extra_col_count = 1
            else:
                extra_col_count += 1
            # I2-02: DLM=COMMA + embedded comma in a {S} string value
            # truncates the string AND shifts every following column —
            # the next column's genuine value is destroyed (verified:
            # `100.0,WELL, TX,10` with DEPT/WELL{S}/GR → WELL truncated
            # to "WELL", GR receives " TX" → null, genuine 10 lost as a
            # discarded extra column).  csv.reader quote-awareness is
            # deliberately NOT used (F2-015: the writer emits raw
            # delimiter.join() with no CSV quotes, so adding quote
            # parsing would break the writer's own roundtrips) — the
            # correct response for external files is a LOUD warning that
            # the value was truncated/lost, not a silent shift.
            if delimiter != " " and _string_curve_indices:
                if embedded_delim_count is None:
                    warnings.warn(
                        f"Data line has {len(values)} values but only {curve_count} "
                        f"curves declared with DLM={delimiter!r}. A string curve "
                        f"value may contain the delimiter character, truncating "
                        f"the string and shifting the following columns (genuine "
                        f"values may be lost). Consider quoting string values or "
                        f"using a delimiter not present in the data. "
                        f"(Further occurrences will be counted.)",
                        UserWarning,
                        stacklevel=2,
                    )
                    embedded_delim_count = 1
                else:
                    embedded_delim_count += 1

        # F-11: Warn when non-wrapped data lines have fewer values than
        # declared curves. Short rows in WRAP=YES mode are expected;
        # this warning only fires in non-wrapped (WRAP=NO) mode,
        # which we know because _read_normal is only called for non-wrapped.
        # F-I2-XPD-03: First occurrence logs full context; subsequent
        # occurrences are counted silently; summary logged at end.
        if len(values) < curve_count:
            if short_row_count is None:
                warnings.warn(
                    f"Data line has {len(values)} values but {curve_count} curves declared in ~C section. "
                    f"Missing values are filled with the null value ({null_value:.2f}). "
                    f"(Further occurrences will be counted.)",
                    UserWarning,
                    stacklevel=2,
                )
                short_row_count = 1
            else:
                short_row_count += 1

        # G-04 (FIX-CONV-2): data_line_count is a pre-allocation hint, never
        # a correctness bound.  When the pre-scan undercounts data lines
        # (rows the pre-scan skips but the reader keeps — short all-mnemonic
        # rows, all-string coincident rows), grow the arrays geometrically
        # (mirroring _read_wrapped._append_value) instead of silently
        # dropping the last data row.  A bounded re-allocation preserves the
        # MAX_TOTAL_ELEMENTS guard's memory-exhaustion intent.
        if current_line >= _allocated:
            _new_capacity = max(_allocated * 2, current_line + 1)
            if curve_count * _new_capacity > MAX_TOTAL_ELEMENTS:
                raise LASParseError(
                    f"Total allocation ({curve_count} curves x {_new_capacity} "
                    f"lines = {curve_count * _new_capacity} elements) exceeds "
                    f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                    f"The file may be malformed or corrupt."
                )
            for curve_name in las_file.curves_order:
                if curve_name in las_file.logs:
                    _old_arr = las_file.logs[curve_name]
                    _new_arr = np.zeros(_new_capacity, dtype=_old_arr.dtype)
                    _new_arr[: _old_arr.shape[0]] = _old_arr
                    # Whole-container growth: EVERY curve array is resized to
                    # the same _new_capacity, so the M-43 equal-length
                    # invariant holds after the pass.  Bypass the per-key
                    # __setitem__ guard during the transition, mirroring
                    # _GuardedDict.trim_all's own dict.__setitem__ pattern
                    # (models.py:302-303) for whole-container reconciliation.
                    dict.__setitem__(las_file.logs, curve_name, _new_arr)
                if curve_name in las_file.string_data:
                    _old_str = las_file.string_data[curve_name]
                    _new_str = np.full(_new_capacity, "", dtype=object)
                    _new_str[: _old_str.shape[0]] = _old_str
                    dict.__setitem__(las_file.string_data, curve_name, _new_str)
            _allocated = _new_capacity
            # Rebuild the pre-extracted numeric references — they now point
            # at the old (too-small) arrays.
            curve_arrays = [
                las_file.logs[name] for name in las_file.curves_order if name in las_file.logs
            ]

        for i in range(min(len(values), curve_count)):
            if i in _string_curve_indices:
                # F-WXP-01: String curve — store raw value verbatim.
                # DR-05: Cap Python-object accumulation (string values
                # amplify ~6-12x beyond the element guard's 8 B/element).
                # Mirrors _las30_data._check_string_cap semantics: raise
                # once the count reaches the cap, before storing.
                curve_name = _string_curve_map[i]
                if _string_value_count >= MAX_STRING_VALUES:
                    raise LASParseError(
                        f"String curve values exceed maximum allowed "
                        f"({MAX_STRING_VALUES}). The file may be malformed "
                        f"or corrupt."
                    )
                _string_value_count += 1
                # E-19: restore the writer's M-85 '_~' escape for the
                # FIRST-column token only (the escape fires only for a
                # first-column string value starting '~'+non-letter), so a
                # genuine '_~' value in any other column is preserved.
                # The 1.2/2.0 writer emits the same escape as LAS 3.0 and
                # the restore is position-scoped exactly like the 3.0 path
                # (_las30_data.py:1195).
                las_file.string_data[curve_name][current_line] = _desanitize_las_value(
                    values[i], _desanitize_enabled, restore_tilde=(i == 0)
                )
            elif i in _integer_curve_indices:
                # L-03/EXT-04: {I} curve — parse via int() to preserve
                # exactness above 2^53 (float() would round).  With a
                # fractional NULL the array is object dtype and failures
                # return the float sentinel (not int(null_value)).
                ni = _logical_to_numeric[i]
                curve_arrays[ni][current_line] = _to_integer_value(
                    _desanitize_las_value(values[i], _desanitize_enabled),
                    null_value,
                    _fc,
                    _null_as_float=not _null_is_integral,
                )
            else:
                ni = _logical_to_numeric[i]
                curve_arrays[ni][current_line] = _to_finite_float(
                    _desanitize_las_value(values[i], _desanitize_enabled),
                    null_value,
                    _fc,
                )

        # Fill remaining curves with null_value when line has fewer values
        for i in range(len(values), curve_count):
            if i in _string_curve_indices:
                # Fill missing string curve values with empty string.
                curve_name = _string_curve_map[i]
                if current_line < len(las_file.string_data[curve_name]):
                    las_file.string_data[curve_name][current_line] = ""
            else:
                ni = _logical_to_numeric[i]
                if current_line < len(curve_arrays[ni]):
                    # L-03: int64 arrays need an integral fill; the int64
                    # branch is only active when null_value is integral.
                    if i in _integer_curve_indices and _null_is_integral:
                        curve_arrays[ni][current_line] = int(null_value)
                    else:
                        curve_arrays[ni][current_line] = null_value

        current_line += 1

    # F-024: Warn when pre-scan overcounted data lines (fewer actual data
    # lines in the ~A section than declared).  Unlike an undercount (which
    # the G-04 grow branch now absorbs without data loss), this preserves
    # data but indicates a pre-scan discrepancy — e.g. a multi-section file
    # where _pre_scan counts lines across all sections but _read_normal
    # only consumes those in the first ~A section.  Uses the ORIGINAL
    # pre-scan count, never the grown capacity.
    if current_line < data_line_count:
        warnings.warn(
            f"Pre-scan overcount: declared {data_line_count} data lines but only {current_line} actual "
            f"data lines found in ~A section. Arrays will be trimmed to "
            f"actual line count.",
            UserWarning,
            stacklevel=2,
        )

    # F-PXR-03: Warn when non-trivial conversion failures occurred.
    _log_conversion_failures(_fc, null_value)

    # F-I2-XPD-03: Summary of extra-column and short-row occurrences,
    # replacing the previous boolean-once pattern that suppressed all
    # diagnostics after the first row.  This allows automated data
    # quality tools to enumerate affected rows.
    if extra_col_count is not None and extra_col_count > 1:
        warnings.warn(
            f"{extra_col_count} data line(s) had more values than the {curve_count} declared curves. "
            f"Extra columns were discarded.",
            UserWarning,
            stacklevel=2,
        )
    if embedded_delim_count is not None and embedded_delim_count > 1:
        warnings.warn(
            f"{embedded_delim_count} data line(s) had more values than the {curve_count} "
            f"declared curves with DLM={delimiter!r}. String curve values may contain "
            f"the delimiter character; those strings were truncated and following "
            f"columns' genuine values may have been lost.",
            UserWarning,
            stacklevel=2,
        )
    if short_row_count is not None and short_row_count > 1:
        warnings.warn(
            f"{short_row_count} data line(s) had fewer values than the {curve_count} declared curves. "
            f"Missing values were filled with the null value ({null_value:.2f}).",
            UserWarning,
            stacklevel=2,
        )

    # F36: Trim arrays when ~A section ended early (fewer data rows than
    # allocated). Pre-allocated np.zeros tail would otherwise expose 0.0
    # values that differ from null_value, corrupting downstream analysis.
    # Fill the tail with null_value before slicing to ensure consistency
    # even when pre-scan over-counts relative to _read_normal's actual
    # line consumption.  FIX-CONV-2: the condition is the ACTUAL allocated
    # capacity (_allocated), which covers both the pre-scan overcount case
    # and the slack left by geometric growth on undercount.
    # F-WXP-01: Also trim string_data arrays for string-curve columns.
    # FIX-CONV-1 (F-01): route the trim through _GuardedDict.trim_all —
    # per-key ``las_file.logs[curve_name] = arr[:current_line]``
    # reassignment tripped the M-43 length guard: the FIRST key is
    # compared against still-untrimmed siblings and rejected even though
    # the final trimmed state is fully consistent.  ``trim_all`` performs
    # the whole-container trim and validates the invariant once.
    if current_line < _allocated:
        # Fill numeric tails with null_value (float64) — trim_all then
        # slices every value to current_line rows.
        for curve_name in las_file.curves_order:
            if curve_name in las_file.logs:
                arr = las_file.logs[curve_name]
                if current_line < len(arr):
                    arr[current_line:] = null_value
        cast(_GuardedDict, las_file.logs).trim_all(current_line)
        # Fill string tails with "" (str_) before the whole-container trim.
        for curve_name in las_file.curves_order:
            if curve_name in las_file.string_data:
                arr_str = las_file.string_data[curve_name]
                if current_line < len(arr_str):
                    arr_str[current_line:] = ""
        cast(_GuardedDict, las_file.string_data).trim_all(current_line)


def _read_wrapped(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    delimiter: str = " ",
    _desanitize_enabled: bool | None = None,
) -> None:
    """Read wrapped ASCII data using n_curves value accumulation.

    A depth step is complete when ``curve_count`` values have accumulated,
    whether they arrive on one line (flowing layout — depth + curve values
    packed per line, W-A) or spread across depth/continuation lines
    (classic wrapped layout — depth on its own line).  Values are buffered
    in ``pending`` and flushed to the curve columns one complete step at a
    time, so both layouts parse identically (the pre-fix depth-line flag
    protocol could only handle depth-on-its-own-line and silently dropped
    flowing data — the reader had NO working path for flowing input).

    The M-38 mixed-wrap first-row case, W-B's mid-file depth lines, and
    F-06's pathologically-malformed shapes all reduce to plain value
    accumulation and parse cleanly (the F-06 hard-fail contract was
    explicitly accepted for removal — II-4).

    Keeps: the mnemonic-header skip (II-16), string/int/float dispatch,
    ``_append_value`` geometric growth, the N-I-08 trailing-partial-step
    diagnostic, the M-72 depth-monotonicity advisory, per-curve padding,
    and ``_log_conversion_failures``.
    """
    # Deduplicate curve names before reading
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # Count actual data lines for MAX_DATA_LINES bound check.
    # II-14: use _iter_ascii_data_lines — the SAME section scan as the
    # read loop — instead of a hand-rolled copy (the standalone mnemonic
    # header row is not skipped here, matching the pre-fix count loop's
    # ≤1 overcount slack; bounds-guard heuristic only).
    _count = sum(1 for _ in _iter_ascii_data_lines(lines, mode_suffix=" (wrapped mode)"))

    if _count > MAX_DATA_LINES:
        raise LASParseError(
            f"Data line count ({_count}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks.
    # In wrapped mode each depth step consumes curve_count values
    # (whether they span lines or arrive packed).  Estimate depth steps
    # from total line count using the known curve_count.  This is a
    # bounds-guard heuristic only — flowing files carry more values per
    # line than the line-based estimate (II-15, LOW/non-blocking); the
    # dynamic total_elements counter below is the real backstop.
    if curve_count > 0:
        # F-54-upgrade: Use math.ceil instead of integer division to avoid
        # undercounting depth steps in wrapped mode.  Integer division
        # _count // curve_count can undercount by up to curve_count-1
        # steps, allowing malicious files to bypass the resource guard.
        # F-37: Removed overly aggressive ceil(_count/2) which over-estimates
        # by up to curve_count/2x for files with many curves, falsely
        # rejecting valid wrapped files.  ceil(_count/curve_count) is the
        # accurate estimate per depth step in wrapped mode.
        depth_steps = max(1, math.ceil(_count / curve_count))
        if curve_count * depth_steps > MAX_TOTAL_ELEMENTS:
            raise LASParseError(
                f"Total allocation ({curve_count} curves x ~{depth_steps} depth steps ≈ "
                f"{curve_count * depth_steps} elements) exceeds maximum allowed "
                f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
            )

    # F-R-03: Detect string curves from CurveDefinition data_format.
    # See _data_section_reader._detect_string_curves for full details.
    _string_curve_indices: set[int] = _detect_string_curves(las_file)
    _string_curve_map: dict[int, str] = {
        _idx: las_file.curves_order[_idx] for _idx in _string_curve_indices
    }
    # FIX-CONV-2: Hoist the mnemonic-header match set once per read.
    _mnemonic_declared = _mnemonic_header_declared(las_file)
    # F-R-03: Accumulate string values into lists (same pattern as
    # data_lists).  Pad/trim and convert to np.array at the end.
    _string_lists: dict[str, list[str]] = {_name: [] for _name in _string_curve_map.values()}
    # DR-05: Cap Python-object accumulation for string values in wrapped
    # mode (mirrors _las30_data._MAX_STRING_VALUES; string objects amplify
    # ~6-12x beyond the element guard's 8 B/element intent).
    _string_value_count = 0

    def _check_string_cap() -> None:
        nonlocal _string_value_count
        if _string_value_count >= MAX_STRING_VALUES:
            raise LASParseError(
                f"String curve values exceed maximum allowed "
                f"({MAX_STRING_VALUES}). The file may be malformed or corrupt."
            )
        _string_value_count += 1

    # L-03: Detect {I} integer-format curves and pre-compute the null
    # sentinel BEFORE allocation — the int64 dtype branch is only taken
    # when the declared NULL is integral (int64 would truncate a
    # fractional sentinel like -999.25, corrupting every null cell).
    # M-62: the int64 branch must ALSO check int64 representability — a
    # huge integral sentinel (>= 2^63) passes is_integer() but int64
    # assignment of int(null_value) raises OverflowError.  Route to the
    # object dtype path instead.
    _integer_curve_indices: set[int] = _detect_integer_curves(las_file)
    null_value = _get_null_value(las_file.well)
    _null_is_integral = (
        float(null_value).is_integer() and _INT64_MIN <= int(null_value) <= _INT64_MAX
    )
    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    # IT3-F-03: Pre-allocate numeric columns instead of accumulating every
    # value as a Python float object (~32 B/value) then converting to numpy
    # at the end (~8 B/value).  The list phase previously amplified peak
    # memory ~4x and bypassed the MAX_TOTAL_ELEMENTS guard's OOM intent
    # (the guard counts final arrays only).  Capacity starts at the
    # depth-step estimate and grows geometrically; the dynamic
    # total_elements counter below still bounds total accumulation.
    _numeric_indices = [i for i in range(curve_count) if i not in _string_curve_indices]
    _est_steps = max(1, math.ceil(_count / curve_count)) if curve_count > 0 else 0
    data_arrays: dict[int, np.ndarray] = {}
    data_fill: dict[int, int] = {}
    for _i in _numeric_indices:
        if _i in _integer_curve_indices and _null_is_integral:
            data_arrays[_i] = np.zeros(_est_steps, dtype=np.int64)
        elif _i in _integer_curve_indices:
            # EXT-04: fractional declared NULL — object dtype preserves
            # exact {I} integers above 2^53 without truncating the
            # fractional null sentinel.
            data_arrays[_i] = np.zeros(_est_steps, dtype=object)
        else:
            data_arrays[_i] = np.zeros(_est_steps, dtype=np.float64)
        data_fill[_i] = 0

    def _append_value(_i: int, value: float | int) -> None:
        """Append one value to curve *_i*'s pre-allocated column, growing
        geometrically when the depth-step estimate is exceeded."""
        _arr = data_arrays[_i]
        _idx = data_fill[_i]
        if _idx >= _arr.shape[0]:
            _new = np.zeros(max(_arr.shape[0] * 2, 1), dtype=_arr.dtype)
            _new[:_idx] = _arr[:_idx]
            data_arrays[_i] = _new
            _arr = _new
        _arr[_idx] = value
        data_fill[_i] = _idx + 1

    # R-2: pending-value-buffer accumulation protocol.  Every flushed step
    # carries exactly curve_count values, so a complete step is consumed
    # per flush regardless of line structure (depth-on-own-line, flowing
    # [6,6], M-38 mixed-wrap, W-B mid-file depth lines all reduce to the
    # same protocol).
    # X-2: consumed values stay buffered until EOF — a `read_idx` pointer
    # tracks the frontier so a wide continuation line is NOT re-copied on
    # every flush (`pending = pending[curve_count:]` was O(n) per flush →
    # O(n²) per line on crafted wide-line wrapped files).  E-20: the
    # buffer is trimmed periodically (module constant
    # _PENDING_TRIM_THRESHOLD) so consumed tokens do not stay alive until
    # EOF (~58 GB retention on crafted files) — each trim is one O(n)
    # slice, amortized O(1) per token, and the O(1)-per-step extraction is
    # preserved (the unconsumed tail is always < one step plus one line).
    pending: list[str] = []
    read_idx = 0
    total_elements = 0  # F-54-upgrade: dynamic element counter for wrapped mode
    # E-42: physical data-line counter.  The header-skip gate below must
    # close after the FIRST data line (like _read_normal's current_line ==
    # 0) — gating on step completion (total_elements == 0) kept the gate
    # open across multiple lines in depth-first wrapped layouts, silently
    # dropping a string continuation value that coincidentally matched a
    # mnemonic on line 2+ as a "header".
    current_line = 0
    # M-13 (wrapped): track whether the standalone mnemonic header row was
    # skipped so an optional units row directly after it can also be
    # skipped (first data line only — same position gate as _read_normal).
    _mnemonic_header_skipped = False
    # M-07: I2-02 mirror on the wrapped twin.  Track embedded-delimiter
    # occurrences for the end-of-section summary (first occurrence logs
    # full context; subsequent occurrences are counted silently).
    embedded_delim_count: int | None = None

    for stripped in _iter_ascii_data_lines(lines, mode_suffix=" (wrapped mode)"):
        # Split using DLM-aware split (shared utility).
        # M-30: pass the declared curve count so the comma branch
        # recombines thousands-separated fragments ("1,234.5") with a loud
        # warning before they enter the step-accumulation buffer.
        values = _split_data_line(stripped, delimiter, expected=curve_count)

        # F-03 (FIX-CONV-1)/II-16: M-37's standalone-mnemonic-header skip
        # must apply in wrapped mode too, gated on the first data line only
        # (DR-M3 — a standalone mnemonic header can only legitimately
        # appear at the top of the section).  H-1/II-11: the predicate
        # consumes the SUPERSET tokenization so a space-separated header
        # row in a DLM=COMMA file is recognized (the DLM-aware values
        # would see one token "DEPT GR" and consume the header as data).
        # E-42: the gate is keyed to the LINE POSITION (current_line == 0,
        # first physical line only) — a step-completion key stayed open
        # across depth-first continuation lines.
        if current_line == 0 and _is_mnemonic_header_row(
            _split_header_row(stripped),
            las_file,
            curve_count,
            _string_curve_indices,
            declared=_mnemonic_declared,
        ):
            _mnemonic_header_skipped = True
            continue

        # M-13 (wrapped — M-03): some real-world files emit an optional
        # UNITS row directly after the mnemonic header row (e.g. "~A\nDEPT
        # GR\nM GAPI\n..."); consuming it as a data step produced a phantom
        # all-null first step and a one-step shift of the whole depth log.
        # Skip it — but only when (a) a mnemonic header row was just
        # skipped (the units row can only appear in that position) and (b)
        # we are still on the first data line (current_line == 0).  The
        # shared is_units_header_row predicate is deliberately narrow
        # (letters only) so a genuine first data row is never
        # misclassified; the parser pre-scan (parser._finalize_pre_scan)
        # subtracts the units row itself using the same predicate (M-13
        # shared contract).
        if (
            current_line == 0
            and _mnemonic_header_skipped
            and is_units_header_row(_split_header_row(stripped))
        ):
            # M-13 (fix4-P1): one-shot — the units row has been consumed.
            # Close the position gate NOW (parser fix3-P1 parity,
            # parser.py:4498): a genuine letters-only first data row that
            # follows within this same section must not be dropped as a
            # units row.  This path has no F-024 overcount diagnostic —
            # the drop was fully silent before this fix.
            _mnemonic_header_skipped = False
            continue

        # M-13 (fix4-P1): a data line has been consumed — the units-row
        # position gate is closed (parser fix3-P1 parity, parser.py:4503).
        # The current_line == 0 position gate below already makes the flag
        # inert after the first physical data line; the explicit reset is
        # parity hygiene against future refactors that relax the gate.
        if current_line == 0 and _mnemonic_header_skipped:
            _mnemonic_header_skipped = False

        # M-07: I2-02 mirror on the wrapped twin.  _read_normal warns when
        # DLM=COMMA + embedded comma in a {S} string value produces an
        # extra token (data_reader.py:1100-1126); the wrapped path had no
        # equivalent — the extra token silently shifted columns and only
        # the misleading N-I-08 fired at EOF (verified: wrapped
        # DEPT/NAME{S}/GR with NAME value 'WELL, INC,50.0' → DEPT=
        # [1000.0,50.0], NAME=['WELL','1001.0'], GR lost).
        #
        # The wrapped signal differs from _read_normal's extra-column
        # trigger: in wrapped mode a single line may legitimately carry
        # MORE than curve_count values (flowing layout packs multiple
        # depth steps per line).  The reliable signal is a step-boundary
        # overshoot: how many values does the current line add beyond
        # what completes the partially-buffered step?  A clean flowing
        # line adds exactly whole steps (a multiple of curve_count) — no
        # warning.  An embedded delimiter adds exactly ONE spurious token,
        # so the overshoot is a non-multiple of curve_count (e.g. the
        # depth-first continuation 'WELL, INC,50.0' carries curve_count
        # values where curve_count-1 are expected → overshoot of 1).
        _step_remainder = (len(pending) - read_idx) % curve_count
        # Values needed to complete the current partially-buffered step
        # (== curve_count for a fresh step with an empty buffer).
        _needed = curve_count - _step_remainder
        _overshoot = len(values) - _needed
        if (
            delimiter != " "
            and _string_curve_indices
            and len(values) > _needed
            and _overshoot > 0
            and _overshoot % curve_count != 0
        ):
            if embedded_delim_count is None:
                warnings.warn(
                    f"Wrapped data line has {len(values)} values but the "
                    f"current depth step needs {_needed} more value(s) for "
                    f"{curve_count} curves declared with DLM={delimiter!r}. "
                    f"A string curve value may contain the delimiter "
                    f"character, truncating the string and shifting the "
                    f"following columns (genuine values may be lost). "
                    f"Consider quoting string values or using a delimiter "
                    f"not present in the data. "
                    f"(Further occurrences will be counted.)",
                    UserWarning,
                    stacklevel=2,
                )
                embedded_delim_count = 1
            else:
                embedded_delim_count += 1

        pending.extend(values)
        while len(pending) - read_idx >= curve_count:
            step = pending[read_idx : read_idx + curve_count]
            read_idx += curve_count
            for _ci, _v in enumerate(step):
                # F-R-03: Dispatch string curves to _string_lists,
                # {I} curves via _to_integer_value, else _to_finite_float.
                if _ci in _string_curve_indices:
                    _check_string_cap()
                    # E-19: restore the writer's M-85 '_~' escape for the
                    # FIRST-column token of each step only (mirrors
                    # _read_normal and the LAS 3.0 path) — a genuine
                    # '_~' value in any other column is preserved.
                    _string_lists[_string_curve_map[_ci]].append(
                        _desanitize_las_value(_v, _desanitize_enabled, restore_tilde=(_ci == 0))
                    )
                elif _ci in _integer_curve_indices:
                    _append_value(
                        _ci,
                        _to_integer_value(
                            _desanitize_las_value(_v, _desanitize_enabled),
                            null_value,
                            _fc,
                            _null_as_float=not _null_is_integral,
                        ),
                    )
                else:
                    _append_value(
                        _ci,
                        _to_finite_float(
                            _desanitize_las_value(_v, _desanitize_enabled),
                            null_value,
                            _fc,
                        ),
                    )
            total_elements += curve_count
            if total_elements > MAX_TOTAL_ELEMENTS:
                raise LASParseError(
                    f"Total elements ({total_elements}) exceeds maximum allowed "
                    f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                    f"The file may be malformed or corrupt."
                )

        current_line += 1

        # E-20: bounded retention — once the consumed frontier crosses the
        # threshold, drop the consumed prefix in one slice and reset the
        # frontier.  The unconsumed tail (partial step + current line) is
        # unaffected, so the N-I-08 trailing-step diagnostic below still
        # sees exactly the leftover values.
        if read_idx >= _PENDING_TRIM_THRESHOLD:
            del pending[:read_idx]
            read_idx = 0

        # M-10 test seam: invoke the optional hook while `pending` is live
        # (after the trim, so the unconsumed-tail state is observable).
        # None in production — one None-check per line, no other cost.
        if _read_wrapped_trace_hook is not None:
            _read_wrapped_trace_hook(pending)

    # X-2: single O(n) trim at EOF — the read_idx pointer above avoided a
    # per-flush list copy; this final slice leaves only the trailing
    # partial step for the N-I-08 diagnostic below.  (E-20: with periodic
    # trims the buffer already holds only the unconsumed tail; when the
    # threshold was never crossed this slice still applies.)
    pending = pending[read_idx:]

    # N-I-08: Detect a trailing incomplete step.  In valid wrapped mode
    # every depth step consumes exactly curve_count values; a leftover
    # partial buffer at EOF cannot form a complete step.  Warn loudly and
    # DISCARD the orphan values — the accepted accumulation contract
    # (pre-fix audit R-6: test_wrapped_depth_line_extra_values → DT=[50,99]
    # not [50,51]; the depth-line protocol previously mis-assigned them).
    # The per-curve padding block below still pads genuine cross-curve
    # length mismatches (e.g. string curves shorter than numeric curves).
    if pending:
        warnings.warn(
            f"Wrapped mode: data section ended with {len(pending)} value(s) "
            f"not accounted for by the curve count {curve_count}. "
            f"A depth step appears to be under-filled at end-of-file; DEPTH "
            f"and curve values may be misaligned.  Check the source file for "
            f"a step missing one or more values.",
            stacklevel=2,
        )
    # M-72: The N-I-08 check above only fires on a trailing partial buffer;
    # it is blind to a mid-file under-fill whose total stays a multiple of
    # curve_count (aligned total).  In that case every depth step
    # "completes" from the reader's perspective, but the missing value
    # caused the reader to consume the next depth line as a data value —
    # permanently shifting a data value into the depth column (and a depth
    # value into the last curve column) with ZERO diagnostics.  Verify the
    # depth column is actually depth-aligned: a genuine depth log is
    # monotonic (non-decreasing OR non-increasing).  A non-monotonic depth
    # column signals the shift.  Advisory warning — not an error.
    if (
        curve_count > 0
        and 0 in data_arrays
        and data_fill[0] >= 2
        and np.issubdtype(data_arrays[0].dtype, np.number)
    ):
        _depth_vals = data_arrays[0][: data_fill[0]]
        _diffs = np.diff(_depth_vals)
        if not (np.all(_diffs >= 0) or np.all(_diffs <= 0)):
            warnings.warn(
                f"Wrapped mode: data section consumed {total_elements} values "
                f"(a multiple of the curve count {curve_count}) but the depth "
                f"column is not monotonic — a depth step appears to be "
                f"under-filled mid-file; DEPTH and curve values may be "
                f"misaligned.  Check the source file for a step missing one "
                f"or more values.",
                stacklevel=2,
            )

    # Compute actual number of depth steps from float curves only.
    # String curve indices have empty data_lists entries — skip them.
    _float_indices = [i for i in range(curve_count) if i not in _string_curve_indices]
    _max_len = max((data_fill[i] for i in _float_indices), default=0)
    # F-007-fix: Also consider string curve lengths so _max_len is not
    # zero when all curves are string-formatted (which would truncate
    # all accumulated string data to empty arrays).
    if _string_lists:
        _max_len = max(_max_len, *(len(sl) for sl in _string_lists.values()))

    # Pad incomplete last depth step for float curves
    for i in _float_indices:
        fill = data_fill[i]
        if fill < _max_len:
            warnings.warn(
                f"Wrapped mode: curve '{las_file.curves_order[i]}' has {fill} values "
                f"but expected {_max_len}. Padding with null value ({null_value}).",
                stacklevel=2,
            )
            _arr = data_arrays[i]
            if _arr.shape[0] < _max_len:
                _new = np.zeros(_max_len, dtype=_arr.dtype)
                _new[:fill] = _arr[:fill]
                data_arrays[i] = _new
                _arr = _new
            if i in _integer_curve_indices and _null_is_integral:
                _arr[fill:_max_len] = int(null_value)
            else:
                _arr[fill:_max_len] = null_value
            data_fill[i] = _max_len

    # F-PXR-03: Warn when non-trivial conversion failures occurred.
    _log_conversion_failures(_fc, null_value)

    # M-07: F-I2-XPD-03-style summary of embedded-delimiter occurrences
    # (I2-02 mirror on the wrapped twin — same contract as _read_normal's
    # summary at :1281-1289, so automated tools can enumerate affected
    # rows).
    if embedded_delim_count is not None and embedded_delim_count > 1:
        warnings.warn(
            f"Wrapped mode: {embedded_delim_count} data line(s) carried more "
            f"values than the {curve_count} curves declared with "
            f"DLM={delimiter!r}. String curve values may contain the "
            f"delimiter character; those strings were truncated and "
            f"following columns' genuine values may have been lost.",
            UserWarning,
            stacklevel=2,
        )

    # Convert pre-allocated columns to final (trimmed) numpy arrays
    for i in _float_indices:
        las_file.logs[las_file.curves_order[i]] = data_arrays[i][: data_fill[i]]

    # F-R-03: Pad/trim and convert string lists to numpy arrays.
    # String curve values are accumulated into _string_lists using the
    # same row-counting protocol as data_lists (one value per depth step).
    for _name, _sl in _string_lists.items():
        if len(_sl) < _max_len:
            warnings.warn(
                f"Wrapped mode: string curve '{_name}' has {len(_sl)} values "
                f"but expected {_max_len}. Padding with empty strings.",
                stacklevel=2,
            )
            _sl = _sl + [""] * (_max_len - len(_sl))
        elif len(_sl) > _max_len:
            _sl = _sl[:_max_len]
        las_file.string_data[_name] = np.array(_sl, dtype=object)
