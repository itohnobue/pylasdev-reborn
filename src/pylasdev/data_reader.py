"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import logging
import math
import warnings
from types import ModuleType

import numpy as np

from .exceptions import LASParseError
from .models import LASFile, WellSection

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
    _get_section_word,
    _is_ascii_section,
    _is_recognized_section_word,
    _is_section_header,
    _iter_ascii_data_lines,
    _log_conversion_failures,
    _split_data_line,
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


def _get_parser_module() -> ModuleType:
    """Lazy-import the parser module to break circular import.

    data_reader is imported by parser, but data_reader needs access
    to parser's module-level attributes (e.g., _DESANITIZE_ENABLED)
    for unified desanitize state.  Lazy import defers the import to
    call time, avoiding the cycle.
    """
    from . import parser as _p
    return _p


def _desanitize_las_value(value: str) -> str:
    """Reverse the writer's ``_``-prefix-on-``#`` escape (local copy).

    The writer prefixes ``#``-starting values with ``_`` to prevent the
    parser from interpreting them as comment lines.  This function strips
    that prefix, restoring the original ``#``-prefixed value.

    Defined locally in data_reader to avoid circular import from parser.
    """
    if not _get_parser_module()._DESANITIZE_ENABLED:
        return value
    if value.startswith("_#"):
        return value[1:]
    idx = value.find("_#")
    if idx > 0 and value[idx - 1].isspace():
        return value[:idx] + value[idx + 1:]
    return value


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
        null_value = _parse_float_with_d_notation(well.get("NULL", default))
        # F-04: Reject non-finite sentinel values (NaN, Inf, -Inf) which
        # float() accepts without error.  These propagate through numpy
        # arrays → corrupted statistics → writer outputs "nan" (invalid LAS).
        if not np.isfinite(null_value):
            raise LASParseError(
                f"NULL value must be a finite number, got {null_value!r}"
            )
        return null_value
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
    if not np.isfinite(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value
    return val


def read_ascii_data(lines: list[str], las_file: LASFile, data_line_count: int,
                    desanitize: bool = True) -> None:
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

    # F-212: Route the desanitize parameter to the unified parser flag
    # so _desanitize_las_value and parser._desanitize_las_value share the
    # same module-level state.
    _get_parser_module()._DESANITIZE_ENABLED = desanitize  # type: ignore[attr-defined]

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
    actual_wrap = _detect_actual_wrap(lines, curve_count, delimiter)
    if actual_wrap:
        _read_wrapped(lines, las_file, curve_count, delimiter)
    else:
        _read_normal(lines, las_file, curve_count, data_line_count, delimiter)


def _detect_actual_wrap(lines: list[str], curve_count: int, delimiter: str = " ") -> bool:
    """Detect if data is actually wrapped by checking the first data line(s).

    In true wrapped mode, the first data line has only 1 value (the depth).
    In non-wrapped mode (even if WRAP=YES header), each line has >= curve_count values.

    For space-delimited files, the check is: first line value count < curve_count.
    For non-space delimiters (COMMA, TAB), a different heuristic is needed because
    trailing empty values can be omitted (per CSV convention), making non-wrapped
    lines appear shorter than curve_count.  In that case, we check whether the
    first data line has >1 values (non-wrapped) or exactly 1 value (wrapped depth).
    When curve_count is 1, both wrapped and non-wrapped modes are equivalent.

    Args:
        lines: File content split into lines.
        curve_count: Number of curves declared in ~C section.
        delimiter: Data column delimiter character (default space).
            Uses DLM-aware splitting when delimiter is not a space.

    Returns:
        True if data is actually wrapped, False if non-wrapped despite header.
    """
    in_ascii = False
    _first_wrap: bool | None = None  # None = not yet found
    for line in lines:
        stripped = line.strip()

        if _is_section_header(stripped):
            # F-I2-XPD-01: Validate section word before toggling
            # in_ascii — unrecognized patterns may be artifacts from
            # control-character replacement (SPLITLINES_CHARS_RE).
            section_word = _get_section_word(stripped)
            if not _is_recognized_section_word(section_word):
                continue
            if _is_ascii_section(stripped):
                in_ascii = True
            elif in_ascii:
                # F-048: Standardize section-detection guard with
                # _iter_ascii_data_lines (uses return/break instead of
                # just resetting in_ascii).  When we encounter a recognized
                # non-~A section header while inside an ~A block, exit the
                # loop — we've left the data section and no more data lines
                # should be used for wrap detection.
                break
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # Data line found — split using DLM-aware split (shared utility).
        # See _data_section_reader._split_data_line for full rationale.
        values = _split_data_line(stripped, delimiter)

        # F-M20: When curve_count is 1, wrapped and non-wrapped modes are
        # equivalent — every line holds exactly one value regardless of
        # mode.  The space-delimiter heuristic ``len(values) < curve_count``
        # is degenerate for curve_count=1 (always False), and forcing
        # wrapped mode on a single-curve file triggers unnecessary overflow
        # warnings and counter logic in _read_wrapped.  Use non-wrapped
        # (the simpler path) when there is nothing to distinguish.
        if curve_count <= 1:
            return False

        if _first_wrap is None:
            # First data line — determine initial wrap heuristic.
            if delimiter == " ":
                # Space delimiter: wrapped lines are shorter than curve_count.
                # Use second-line corroboration (F-M16) to catch sparse
                # first-line false positives.
                _first_wrap = len(values) < curve_count
            else:
                # COMMA/TAB delimiter: trailing empty values are stripped by
                # _split_data_line per CSV convention, making len(values)
                # legitimately < curve_count even for non-wrapped files.
                # Use the depth-line heuristic: a wrapped depth line always
                # has exactly 1 value; if the first data line has >1 values,
                # it's non-wrapped regardless of curve_count (F-023).
                _first_wrap = len(values) <= 1
            if not _first_wrap:
                return False  # Not wrapped — no need for second peek
            # For COMMA/TAB: depth-line heuristic is definitive — no
            # second-line corroboration needed.  Non-wrapped files always
            # have >1 values per line (curve_count > 1 guaranteed above).
            if delimiter != " ":
                return True
            # Space delimiter: continue to second data line for
            # corroboration (F-M16: secondary peek prevents sparse
            # first-line false positives).
            continue

        # Second data line — corroborate wrap detection for space
        # delimiter (F-M16).
        _corroborates = len(values) < curve_count
        if not _corroborates:
            # Second line has full values — first line was sparse,
            # not wrapped.
            return False
        return True  # Second line corroborates wrap

    return True  # No data found, default to wrapped


def _deduplicate_curves(las_file: LASFile, _stacklevel: int = 2) -> None:
    """Detect and rename duplicate curve names with warning.

    Appends _2, _3, etc. to duplicate mnemonics so each curve gets
    its own array in las_file.logs. Also updates the corresponding
    CurveDefinition objects to keep curves_order and curves in sync.

    Args:
        las_file: LASFile to deduplicate curves in.
        _stacklevel: Stacklevel for warnings.warn (default 2 points
            to the immediate caller; pass 3 when called from deeper
            call chains such as parser._process_ascii_data).
    """
    seen: dict[str, int] = {}
    new_order: list[str] = []
    # Track all names in the output order for collision detection (F-22)
    output_names: set[str] = set()
    for idx, name in enumerate(las_file.curves_order):
        if name in seen:
            seen[name] += 1
            new_name = _resolve_unique_curve_name(
                name, seen[name], output_names
            )
            # Update the seen counter to match the actual suffix used
            seen[name] = _suffix_from_name(new_name, name)
            _rename_duplicate_curve(
                las_file,
                idx,
                name,
                new_name,
                new_order,
                output_names,
                _stacklevel,
            )
        else:
            # F-22: Check for cross-base collisions where an
            # original name matches a previously generated _N suffix.
            # Input ["DEPT","DEPT","DEPT_2"] should produce
            # ["DEPT","DEPT_2","DEPT_2_2"], not
            # ["DEPT","DEPT_2","DEPT_2"] with duplicate keys.
            if name in output_names:
                new_name = _resolve_unique_curve_name(
                    name, 2, output_names
                )
                seen[name] = _suffix_from_name(new_name, name)
                _rename_duplicate_curve(
                    las_file,
                    idx,
                    name,
                    new_name,
                    new_order,
                    output_names,
                    _stacklevel,
                )
            else:
                seen[name] = 1
                new_order.append(name)
                output_names.add(name)
    if new_order != las_file.curves_order:
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
        return int(name[len(base_name) + 1:])
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
    warnings.warn(
        f"Duplicate curve mnemonic '{name}' renamed to '{new_name}'. "
        "Data may come from a file with repeated curve names.",
        stacklevel=_stacklevel,
    )
    new_order.append(new_name)
    output_names.add(new_name)
    if idx < len(las_file.curves):
        if not las_file.curves[idx].original_mnemonic:
            las_file.curves[idx].original_mnemonic = name
        las_file.curves[idx].mnemonic = new_name


def _read_normal(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    data_line_count: int,
    delimiter: str = " ",
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

    # Pre-allocate arrays — string curves go to string_data, numeric to logs.
    # C-505: Ensure curvenames that appear in both string_curve_indices and
    # are allocated mixed-ly do not cause double-entry issues in logs.
    _numeric_curve_indices: list[int] = []
    _string_curve_map: dict[int, str] = {}  # index → curve_name for fast lookup
    for i, curve_name in enumerate(las_file.curves_order):
        if i in _string_curve_indices:
            las_file.string_data[curve_name] = np.full(
                data_line_count, "", dtype=object
            )
            _string_curve_map[i] = curve_name
        else:
            las_file.logs[curve_name] = np.zeros(
                data_line_count, dtype=np.float64
            )
            _numeric_curve_indices.append(i)

    # Pre-extract numeric arrays for fast inner-loop access.
    # String array access is direct via las_file.string_data since we
    # cannot mix dtypes in a homogeneous list.
    curve_arrays = [las_file.logs[name] for name in las_file.curves_order
                    if name in las_file.logs]
    # Build a mapping from logical curve index → numeric array index
    # for the inner loop (string curves are excluded from curve_arrays).
    _logical_to_numeric: dict[int, int] = {}
    for ni, li in enumerate(_numeric_curve_indices):
        _logical_to_numeric[li] = ni

    null_value = _get_null_value(las_file.well)
    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    current_line = 0
    # F-I2-XPD-03: Replace boolean-once flags with counters so automated
    # validation tools can enumerate the total number of affected rows.
    # The first occurrence logs full context; subsequent occurrences are
    # counted silently; a summary is logged at the end of the section.
    extra_col_count: int | None = None  # Track extra-column count for summary
    short_row_count: int | None = None  # F-11: Track short-row count for summary
    discarded_lines = 0  # Track silently-discarded lines from pre-scan undercount

    for stripped in _iter_ascii_data_lines(lines):
        # Split using DLM-aware split (shared utility).
        values = _split_data_line(stripped, delimiter)

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

        # G-04: Bounds guard — skip writes when current_line exceeds
        # pre-allocated array size.  This can happen when _pre_scan
        # undercounts data lines (e.g., due to section-header detection
        # mismatch — G-05).  Mirroring _read_wrapped guards at lines ~490.
        if current_line >= data_line_count:
            discarded_lines += 1
            continue

        for i in range(min(len(values), curve_count)):
            if i in _string_curve_indices:
                # F-WXP-01: String curve — store raw value verbatim.
                curve_name = _string_curve_map[i]
                las_file.string_data[curve_name][current_line] = _desanitize_las_value(values[i])
            else:
                ni = _logical_to_numeric[i]
                curve_arrays[ni][current_line] = _to_finite_float(
                    _desanitize_las_value(values[i]), null_value, _fc
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
                    curve_arrays[ni][current_line] = null_value

        current_line += 1

    # Warn when pre-scan undercounted data lines, causing data discard.
    if discarded_lines > 0:
        warnings.warn(
            f"Pre-scan undercount: {discarded_lines} data line(s) discarded because the "
            f"actual data exceeds the {data_line_count} lines declared by the pre-scan. "
            f"Las file data may be truncated.",
            UserWarning,
            stacklevel=2,
        )

    # F-024: Warn when pre-scan overcounted data lines (fewer actual data
    # lines in the ~A section than declared).  Unlike the undercount case
    # (data loss), this preserves data but indicates a pre-scan discrepancy
    # — e.g. a multi-section file where _pre_scan counts lines across all
    # sections but _read_normal only consumes those in the first ~A section.
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
    if short_row_count is not None and short_row_count > 1:
        warnings.warn(
            f"{short_row_count} data line(s) had fewer values than the {curve_count} declared curves. "
            f"Missing values were filled with the null value ({null_value:.2f}).",
            UserWarning,
            stacklevel=2,
        )

    # F36: Trim arrays when ~A section ended early (fewer data lines than
    # declared). Pre-allocated np.zeros tail would otherwise expose 0.0
    # values that differ from null_value, corrupting downstream analysis.
    # Fill the tail with null_value before slicing to ensure consistency
    # even when pre-scan over-counts relative to _read_normal's actual
    # line consumption.
    # F-WXP-01: Also trim string_data arrays for string-curve columns.
    if current_line < data_line_count:
        # Trim numeric arrays (float64)
        for curve_name in las_file.curves_order:
            if curve_name in las_file.logs:
                arr = las_file.logs[curve_name]
                if current_line < len(arr):
                    arr[current_line:] = null_value
                las_file.logs[curve_name] = arr[:current_line]
        # Trim string arrays (str_)
        for curve_name in las_file.curves_order:
            if curve_name in las_file.string_data:
                arr_str = las_file.string_data[curve_name]
                if current_line < len(arr_str):
                    arr_str[current_line:] = ""
                las_file.string_data[curve_name] = arr_str[:current_line]


def _read_wrapped(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    delimiter: str = " ",
) -> None:
    """Read wrapped ASCII data using depth_line flag protocol.

    In wrapped mode:
    - The DEPTH value appears ALONE on its own line
    - Subsequent lines contain the remaining curve values
    - Once all curves for a depth step are read, the next depth line follows

    Uses list accumulation then np.array() at end to avoid the O(n^2)
    numpy.append bug in the original code.
    """
    # Deduplicate curve names before reading
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # Count actual data lines for MAX_DATA_LINES bound check.
    # All other reading paths (_read_normal, parser, dev_reader)
    # already have this guard.
    _count_in_ascii = False
    _count = 0
    for line in lines:
        stripped = line.strip()
        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                _count_in_ascii = True
            elif _count_in_ascii:
                # F-I2-XPD-01: Only break counting for recognized LAS
                # section headers — unrecognized patterns may be artifacts
                # from control-character replacement (SPLITLINES_CHARS_RE).
                section_word = _get_section_word(stripped)
                if _is_recognized_section_word(section_word):
                    break  # End of ~A section
            continue
        if _count_in_ascii and stripped and not stripped.startswith("#"):
            _count += 1

    if _count > MAX_DATA_LINES:
        raise LASParseError(
            f"Data line count ({_count}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks.
    # In wrapped mode each depth step spans ~curve_count lines
    # (1 depth + curve_count-1 data values).  Estimate depth steps
    # from total line count using the known curve_count.  The
    # alternative heuristic of counting single-value lines (or using
    # ceil(_count/2)) overcounts depth steps when curve values
    # legitimately appear one per line, causing false rejection.
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
    # F-R-03: Accumulate string values into lists (same pattern as
    # data_lists).  Pad/trim and convert to np.array at the end.
    _string_lists: dict[str, list[str]] = {
        _name: [] for _name in _string_curve_map.values()
    }

    # Accumulate into lists, convert to numpy at end
    data_lists: list[list[float]] = [[] for _ in range(curve_count)]

    null_value = _get_null_value(las_file.well)
    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves
    depth_had_extra = False  # F-06: track pathologically-malformed depth lines
    total_elements = 0  # F-54-upgrade: dynamic element counter for wrapped mode

    for stripped in _iter_ascii_data_lines(lines, mode_suffix=" (wrapped mode)"):
        # Split using DLM-aware split (shared utility).
        values = _split_data_line(stripped, delimiter)

        if depth_line:
            # Depth line: single value = depth for this step.
            # Reset the extra-values flag at each depth step boundary
            # so stale flags from a previous step never persist into
            # the pathological-misalignment check for this step.
            depth_had_extra = False
            if not values:
                continue
            if len(values) > 1:
                warnings.warn(
                    f"Wrapped mode: depth line has {len(values)} values, expected 1. "
                    f"Extra values discarded. Line content: '{stripped[:80]}'",
                    stacklevel=2,
                )
                # F-06: Mark this depth line as malformed.  If the subsequent
                # data line cannot fill all non-depth curves, the file is
                # pathologically misaligned and will be rejected.
                depth_had_extra = True
            # F-R-03: If the depth curve is a string curve, accumulate
            # into _string_lists and use null_value as a data_lists[0]
            # placeholder for row-count alignment.
            if 0 in _string_curve_indices:
                _string_lists[_string_curve_map[0]].append(_desanitize_las_value(values[0]))
                data_lists[0].append(null_value)
            else:
                data_lists[0].append(_to_finite_float(_desanitize_las_value(values[0]), null_value, _fc))
            total_elements += 1
            if total_elements > MAX_TOTAL_ELEMENTS:
                raise LASParseError(
                    f"Total elements ({total_elements}) exceeds maximum allowed "
                    f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                    f"The file may be malformed or corrupt."
                )
            depth_line = False
            counter = 0
        else:
            # Data lines: values for remaining curves
            # F-06: Pathological misalignment detection.  If the previous
            # depth line had extra values AND this data line has fewer
            # values than the number of unfilled non-depth curves
            # (curve_count - 1 - counter), the file is likely misaligned.
            # A truly pathological case (depth line has extra data, next
            # line has very few values, and the gap exceeds 2 curves)
            # raises an error because data corruption is certain.
            if depth_had_extra and curve_count >= 3:
                remaining_curves = curve_count - 1 - counter
                if len(values) < remaining_curves:
                    if len(values) <= 2 and remaining_curves - len(values) >= 2:
                        # Truly pathological: data line provides ≤2 values
                        # but ≥2 more curves need filling — data WILL be
                        # irrecoverably misaligned across all curves.
                        raise LASParseError(
                            f"Wrapped mode: pathologically malformed data — the "
                            f"previous depth line had extra values, and this data "
                            f"line has only {len(values)} values but "
                            f"{remaining_curves} non-depth curves still need "
                            f"values (total curves={curve_count}). File is "
                            f"irrecoverably misaligned."
                        )
                    # Non-truly-pathological: data line has fewer values
                    # than needed but the shift is ≤2 curves — recovery
                    # would silently produce corrupt data (curve values
                    # shifted by 1 depth step).  Hard-fail with a clear
                    # diagnostic instead of producing corrupt output.
                    raise LASParseError(
                        f"Wrapped mode: unrecoverable data misalignment — "
                        f"the previous depth line had extra values, and "
                        f"this data line has only {len(values)} values "
                        f"but {remaining_curves} non-depth curves still "
                        f"need values (total curves={curve_count}). "
                        f"Continuing would produce corrupt data.  Consider "
                        f"reloading with wrapped=False if the original "
                        f"unwrapped file is available."
                    )

            for i, val_str in enumerate(values):
                counter += 1

                if counter >= curve_count:
                    # F2: Overflow — more values on this line than non-depth
                    # curves remaining in the step.  Extra values would shift
                    # subsequent lines if consumed silently; warn and discard
                    # the overflow portion.
                    warnings.warn(
                        f"Wrapped mode: overflow on data line — {len(values)} "
                        f"values but only {curve_count - 1} non-depth curves. "
                        f"Extra value(s) discarded. Line content: '{stripped[:80]}'",
                        stacklevel=2,
                    )
                    # F-I2-M15: If the previous depth line had extra values
                    # (suggesting a multi-value non-wrapped row was parsed as
                    # a depth line), and this data line overflows (has enough
                    # values to exceed all remaining non-depth curves), then
                    # the file is likely non-wrapped being read as wrapped.
                    # The first values from each line are misrouted: what
                    # should be the next depth step goes to C1 instead,
                    # causing a permanent DEPTH↔C1 swap.
                    if depth_had_extra:
                        warnings.warn(
                            f"Wrapped mode: suspected non-wrapped data — "
                            f"a previous depth line had extra values and "
                            f"this data line overflowed. DEPTH and curve "
                            f"values are likely swapped. "
                            f"Line content: '{stripped[:80]}'",
                            stacklevel=2,
                        )
                    break

                # F-R-03: Dispatch string curves to _string_lists,
                # float curves to data_lists via _to_finite_float.
                if counter in _string_curve_indices:
                    _string_lists[_string_curve_map[counter]].append(_desanitize_las_value(val_str))
                else:
                    data_lists[counter].append(_to_finite_float(_desanitize_las_value(val_str), null_value, _fc))
                total_elements += 1
                if total_elements > MAX_TOTAL_ELEMENTS:
                    raise LASParseError(
                        f"Total elements ({total_elements}) exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                        f"The file may be malformed or corrupt."
                    )

                if counter == curve_count - 1:
                    # All curves for this depth step are complete.
                    # Break to discard any extra values on this line
                    # (prevents silent misalignment if a line has
                    # more values than expected).
                    # F-D2-M01: Warn when extra values remain on this
                    # line after step completion (previously silent).
                    if i + 1 < len(values):
                        warnings.warn(
                            f"Wrapped mode: step complete with "
                            f"{len(values) - i - 1} extra value(s) "
                            f"discarded on this line. Line content: "
                            f"'{stripped[:80]}'",
                            stacklevel=2,
                        )
                        # F-I2-M15: If the previous depth line also had
                        # extra values (depth_had_extra is True) AND this
                        # data line completes the step with leftover values,
                        # the file is likely non-wrapped data misread as
                        # wrapped.  Each multi-value line contributes only
                        # its first value where one curve value is expected,
                        # permanently swapping DEPTH with C1 (and C1 with C2,
                        # etc.) for all subsequent depth steps.
                        if depth_had_extra:
                            warnings.warn(
                                f"Wrapped mode: suspected non-wrapped data — "
                                f"previous depth line had extra values and "
                                f"this data line completed the step with "
                                f"{len(values) - i - 1} leftover value(s). "
                                f"DEPTH and curve values may be swapped. "
                                f"Line content: '{stripped[:80]}'",
                                stacklevel=2,
                            )
                    counter = 0
                    depth_line = True
                    depth_had_extra = False
                    break

            # F11: After overflow (extra values on data line were
            # discarded), reset for the next depth line.  The break in
            # the overflow branch leaves counter at >= curve_count and
            # depth_line=False; without this reset the next line would
            # be incorrectly treated as a continuation of this depth step.
            # Partial under-fill (fewer values than needed) is NOT reset
            # here — in valid wrapped files data values can legitimately
            # span multiple lines.
            if counter >= curve_count and not depth_line:
                depth_line = True
                counter = 0
                depth_had_extra = False

    # Compute actual number of depth steps from float curves only.
    # String curve indices have empty data_lists entries — skip them.
    _float_indices = [i for i in range(curve_count) if i not in _string_curve_indices]
    _max_len = max((len(data_lists[i]) for i in _float_indices), default=0)
    # F-007-fix: Also consider string curve lengths so _max_len is not
    # zero when all curves are string-formatted (which would truncate
    # all accumulated string data to empty arrays).
    if _string_lists:
        _max_len = max(_max_len, *(len(sl) for sl in _string_lists.values()))

    # Pad incomplete last depth step for float curves
    for i in _float_indices:
        dl = data_lists[i]
        if len(dl) < _max_len:
            warnings.warn(
                f"Wrapped mode: curve '{las_file.curves_order[i]}' has {len(dl)} values "
                f"but expected {_max_len}. Padding with null value ({null_value}).",
                stacklevel=2,
            )
            dl.extend([null_value] * (_max_len - len(dl)))

    # F-PXR-03: Warn when non-trivial conversion failures occurred.
    _log_conversion_failures(_fc, null_value)

    # Convert float lists to numpy arrays
    for i in _float_indices:
        las_file.logs[las_file.curves_order[i]] = np.array(data_lists[i], dtype=np.float64)

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
