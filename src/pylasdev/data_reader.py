"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import csv
import logging
import math
import re
import warnings

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
MAX_TOKENS_PER_LINE = MAX_CURVES


# F-ITER2-D2-M04: Regex to extract the full section word from a header line
# (matching parser.SECTION_PATTERN semantics).  Used to compare against
# exact section words {"A", "ASCII"} instead of checking only the first
# character after ~, which caused divergence when headers like ~A_DEFINITION
# entered ASCII mode in the reader but were excluded by parser._pre_scan.
_SECTION_WORD_RE = re.compile(r"^~([A-Za-z]\S*)")


def _get_section_word(stripped: str) -> str:
    """Extract the section word from a section header line.

    Returns the uppercased section word (e.g. "A", "ASCII", "A_DEFINITION")
    or an empty string if the line is not a valid section header.
    """
    match = _SECTION_WORD_RE.match(stripped)
    return match.group(1).upper() if match else ""


def _is_ascii_section(stripped: str) -> bool:
    """Check if a section header targets the ASCII data (~A / ~ASCII) section.

    Aligned with parser._pre_scan which uses ``section_word in {"A", "ASCII"}``.
    """
    return _get_section_word(stripped) in {"A", "ASCII"}


def _is_section_header(stripped: str) -> bool:
    """Check if a stripped line is a section header (~[A-Za-z]).

    Uses ASCII-only character check to match the parser's SECTION_PATTERN,
    which is limited to [A-Za-z].  The parser upper-cases section letters,
    so matching must stay within ASCII.
    """
    return (
        stripped.startswith("~")
        and len(stripped) > 1
        and stripped[1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
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
    """
    return float(value_str.replace("D", "E").replace("d", "e"))


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
    except (ValueError, TypeError):
        return default_float


def _to_finite_float(value_str: str, null_value: float) -> float:
    """Convert string to float, replacing non-finite values with null_value.

    Python's ``float()`` accepts ``"nan"``, ``"inf"``, ``"-inf"`` and
    overflow exponents (e.g. ``"1e309"``) without error.  These non-finite
    values corrupt downstream numpy computations (NaN propagation, Inf
    making statistics invalid).  This helper catches them and returns
    *null_value* instead.

    Also handles empty strings and non-numeric strings gracefully.

    Args:
        value_str: String to convert.  May be empty.
        null_value: Value to return when conversion fails or result is
            non-finite.

    Returns:
        A finite float, or *null_value*.
    """
    if not value_str:
        return null_value
    try:
        val = _parse_float_with_d_notation(value_str)
    except ValueError:
        return null_value
    if not np.isfinite(val):
        return null_value
    return val


def read_ascii_data(lines: list[str], las_file: LASFile, data_line_count: int) -> None:
    """Read the ~A (ASCII data) section and populate las_file.logs.

    Args:
        lines: File content split into lines (pre-split by reader.py
            for efficiency — eliminates redundant content.splitlines()).
        las_file: LASFile object with curves_order already populated.
        data_line_count: Number of data lines (from pre-scan).
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

    if curve_count > MAX_CURVES:
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
    for line in lines:
        stripped = line.strip()

        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                in_ascii = True
            else:
                # F-ITER2-D2-M05: Reset in_ascii on non-A section headers.
                # Without this, _detect_actual_wrap would never exit an empty
                # ~A section and would scan lines from subsequent sections
                # (e.g. ~O) as wrap-detection input, producing false results.
                in_ascii = False
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # Data line found — split using DLM-aware approach.
        # F2-015: Use csv.reader for COMMA/TAB delimiters so values containing
        # the delimiter inside quotes are NOT incorrectly split.
        if delimiter == " ":
            values = stripped.split(maxsplit=MAX_TOKENS_PER_LINE)
        else:
            # F-I2-M12: Wrap csv.reader in try/except — csv.Error
            # (e.g. unclosed quotes or field-size overflow) is NOT a
            # LASParseError subclass and would escape the public API
            # (reader.py only catches LASParseError).
            try:
                reader = csv.reader(
                    [stripped], delimiter=delimiter, quoting=csv.QUOTE_MINIMAL
                )
                row = next(reader)
            except csv.Error as e:
                raise LASParseError(
                    f"Failed to parse delimited data line: {e}"
                ) from e
            # Safety cap: prevent unbounded token count from malformed input,
            # matching the maxsplit behavior of str.split.
            values = row[: MAX_TOKENS_PER_LINE + 1]

        # F-M20: When curve_count is 1, wrapped and non-wrapped modes are
        # equivalent — every line holds exactly one value regardless of
        # mode.  The space-delimiter heuristic ``len(values) < curve_count``
        # is degenerate for curve_count=1 (always False), and forcing
        # wrapped mode on a single-curve file triggers unnecessary overflow
        # warnings and counter logic in _read_wrapped.  Use non-wrapped
        # (the simpler path) when there is nothing to distinguish.
        if curve_count <= 1:
            return False
        # F-023: For non-space delimiters (COMMA, TAB), trailing empty values
        # can be omitted per CSV convention, making the first line appear
        # shorter than curve_count.  Use a different heuristic: a wrapped
        # depth line always has exactly 1 value; if the first data line has
        # >1 values, it's non-wrapped regardless of curve_count.
        if delimiter != " ":
            # A depth-only line in wrapped mode has exactly 1 value.
            # Multiple values on the first data line → non-wrapped.
            return len(values) <= 1
        else:
            # Space delimiter: traditional heuristic.
            return len(values) < curve_count

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

    # Pre-allocate arrays
    for curve_name in las_file.curves_order:
        las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)

    # Pre-extract arrays to avoid O(rows x curves) dict lookups in inner loop
    curve_arrays = [las_file.logs[name] for name in las_file.curves_order]

    null_value = _get_null_value(las_file.well)

    in_ascii = False
    current_line = 0
    warned_extra = False  # Track extra-column warning per file
    warned_short = False  # F-11: Track short-row warning per file
    discarded_lines = 0  # Track silently-discarded lines from pre-scan undercount

    for line in lines:
        stripped = line.strip()

        # F-20: Align section detection with parser.py's SECTION_PATTERN
        # (~[A-Za-z]).  Lines starting with ~ but without an alphabetic
        # section letter (e.g. bare ~, ~~~, etc.) are ignored.
        # F-ITER2-D2-M04: Use _is_ascii_section to check for exact ~A/~ASCII
        # match (aligned with parser._pre_scan), not just the first character.
        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                in_ascii = True
            else:
                if in_ascii:
                    # Stop reading at the first non-~A section header.
                    # _pre_scan counts data lines for the FIRST contiguous
                    # ~A block only (per_block_counts[0]), matching this
                    # break-at-first-non-~A behavior.  Both sides agree:
                    # only the first ~A block is ingested.
                    break
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # F2-015: Use csv.reader for COMMA/TAB delimiters so values
        # containing the delimiter inside double-quotes are NOT incorrectly
        # split (e.g., "Run 1, Tool A" stays as one token with COMMA
        # delimiter).  csv.QUOTE_MINIMAL handles CSV-style quoting.
        if delimiter == " ":
            values = stripped.split(maxsplit=MAX_TOKENS_PER_LINE)
        else:
            # F-I2-M12: Wrap csv.reader in try/except — csv.Error
            # (e.g. unclosed quotes or field-size overflow) is NOT a
            # LASParseError subclass and would escape the public API
            # (reader.py only catches LASParseError).
            try:
                reader = csv.reader(
                    [stripped], delimiter=delimiter, quoting=csv.QUOTE_MINIMAL
                )
                row = next(reader)
            except csv.Error as e:
                raise LASParseError(
                    f"Failed to parse delimited data line: {e}"
                ) from e
            # Safety cap: prevent unbounded token count from malformed input,
            # matching the maxsplit behavior of str.split.
            values = row[: MAX_TOKENS_PER_LINE + 1]

        # Warn about extra columns being silently discarded
        if len(values) > curve_count and not warned_extra:
            warned_extra = True
            logger.warning(
                "Data line has %d values but only %d curves declared "
                "in ~C section. Extra columns are discarded.",
                len(values),
                curve_count,
            )

        # F-11: Warn when non-wrapped data lines have fewer values than
        # declared curves. Short rows in WRAP=YES mode are expected;
        # this warning only fires in non-wrapped (WRAP=NO) mode,
        # which we know because _read_normal is only called for non-wrapped.
        if len(values) < curve_count and not warned_short:
            warned_short = True
            logger.warning(
                "Data line has %d values but %d curves declared in ~C section. "
                "Missing values are filled with the null value (%.2f).",
                len(values),
                curve_count,
                null_value,
            )

        # G-04: Bounds guard — skip writes when current_line exceeds
        # pre-allocated array size.  This can happen when _pre_scan
        # undercounts data lines (e.g., due to section-header detection
        # mismatch — G-05).  Mirroring _read_wrapped guards at lines ~490.
        if current_line >= data_line_count:
            discarded_lines += 1
            continue

        for i in range(min(len(values), curve_count)):
            curve_arrays[i][current_line] = _to_finite_float(values[i], null_value)

        # Fill remaining curves with null_value when line has fewer values
        for i in range(len(values), curve_count):
            if current_line < len(curve_arrays[i]):
                curve_arrays[i][current_line] = null_value

        current_line += 1

    # Warn when pre-scan undercounted data lines, causing data discard.
    if discarded_lines > 0:
        logger.warning(
            "Pre-scan undercount: %d data line(s) discarded because the "
            "actual data exceeds the %d lines declared by the pre-scan. "
            "Las file data may be truncated.",
            discarded_lines,
            data_line_count,
        )

    # F-024: Warn when pre-scan overcounted data lines (fewer actual data
    # lines in the ~A section than declared).  Unlike the undercount case
    # (data loss), this preserves data but indicates a pre-scan discrepancy
    # — e.g. a multi-section file where _pre_scan counts lines across all
    # sections but _read_normal only consumes those in the first ~A section.
    if current_line < data_line_count:
        logger.warning(
            "Pre-scan overcount: declared %d data lines but only %d actual "
            "data lines found in ~A section. Arrays will be trimmed to "
            "actual line count.",
            data_line_count,
            current_line,
        )

    # F36: Trim arrays when ~A section ended early (fewer data lines than
    # declared). Pre-allocated np.zeros tail would otherwise expose 0.0
    # values that differ from null_value, corrupting downstream analysis.
    # Fill the tail with null_value before slicing to ensure consistency
    # even when pre-scan over-counts relative to _read_normal's actual
    # line consumption.
    if current_line < data_line_count:
        for curve_name in las_file.curves_order:
            arr = las_file.logs[curve_name]
            if current_line < len(arr):
                arr[current_line:] = null_value
            las_file.logs[curve_name] = arr[:current_line]


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
    # conservatively from total line count — the alternative heuristic
    # of counting single-value lines overcounts depth steps when curve
    # values legitimately appear one per line, causing false rejection.
    if curve_count > 0:
        # F-54-upgrade: Use math.ceil instead of integer division to avoid
        # undercounting depth steps in wrapped mode.  Integer division
        # _count // curve_count can undercount by up to curve_count-1
        # steps, allowing malicious files to bypass the resource guard.
        depth_steps = max(1, math.ceil(_count / curve_count))
        if curve_count * depth_steps > MAX_TOTAL_ELEMENTS:
            raise LASParseError(
                f"Total allocation ({curve_count} curves x ~{depth_steps} depth steps ≈ "
                f"{curve_count * depth_steps} elements) exceeds maximum allowed "
                f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
            )

    # Accumulate into lists, convert to numpy at end
    data_lists: list[list[float]] = [[] for _ in range(curve_count)]

    null_value = _get_null_value(las_file.well)

    in_ascii = False
    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves
    depth_had_extra = False  # F-06: track pathologically-malformed depth lines
    total_elements = 0  # F-54-upgrade: dynamic element counter for wrapped mode

    for line in lines:
        stripped = line.strip()

        # F-20: Align section detection with parser.py's SECTION_PATTERN
        # (~[A-Za-z]).  Only treat lines as section headers when the
        # character after ~ is alphabetic.
        # F-ITER2-D2-M04: Use _is_ascii_section for exact match.
        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                in_ascii = True
            else:
                if in_ascii:
                    # Stop reading at the first non-~A section header.
                    # _pre_scan counts data lines for the FIRST contiguous
                    # ~A block only (per_block_counts[0]), matching this
                    # break-at-first-non-~A behavior.  Both sides agree:
                    # only the first ~A block is ingested.
                    break
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # F2-015: Use csv.reader for COMMA/TAB delimiters so values
        # containing the delimiter inside double-quotes are NOT incorrectly
        # split (e.g., "Run 1, Tool A" stays as one token with COMMA
        # delimiter).  csv.QUOTE_MINIMAL handles CSV-style quoting.
        if delimiter == " ":
            values = stripped.split(maxsplit=MAX_TOKENS_PER_LINE)
        else:
            # F-I2-M12: Wrap csv.reader in try/except — csv.Error
            # (e.g. unclosed quotes or field-size overflow) is NOT a
            # LASParseError subclass and would escape the public API
            # (reader.py only catches LASParseError).
            try:
                reader = csv.reader(
                    [stripped], delimiter=delimiter, quoting=csv.QUOTE_MINIMAL
                )
                row = next(reader)
            except csv.Error as e:
                raise LASParseError(
                    f"Failed to parse delimited data line: {e}"
                ) from e
            # Safety cap: prevent unbounded token count from malformed input,
            # matching the maxsplit behavior of str.split.
            values = row[: MAX_TOKENS_PER_LINE + 1]

        if depth_line:
            # Depth line: single value = depth for this step.
            # Reset the extra-values flag at each depth step boundary
            # so stale flags from a previous step never persist into
            # the pathological-misalignment check for this step.
            depth_had_extra = False
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
            try:
                data_lists[0].append(_to_finite_float(values[0], null_value))
            except IndexError:
                data_lists[0].append(null_value)
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
                    # Warn but continue: the padding logic at the end of
                    # _read_wrapped will fill gaps with null_value.
                    warnings.warn(
                        f"Wrapped mode: previous depth line had extra values, "
                        f"and this data line has only {len(values)} values but "
                        f"{remaining_curves} non-depth curves still need values "
                        f"(total curves={curve_count}). Data may be misaligned.",
                        stacklevel=2,
                    )
                depth_had_extra = False  # This line handled the extra-values case

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

                data_lists[counter].append(_to_finite_float(val_str, null_value))
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

    # Validate array lengths — pad incomplete last depth step
    max_len = max((len(dl) for dl in data_lists), default=0)
    for i, dl in enumerate(data_lists):
        if len(dl) < max_len:
            warnings.warn(
                f"Wrapped mode: curve '{las_file.curves_order[i]}' has {len(dl)} values "
                f"but expected {max_len}. Padding with null value ({null_value}).",
                stacklevel=2,
            )
            dl.extend([null_value] * (max_len - len(dl)))

    # Convert lists to numpy arrays
    for i, curve_name in enumerate(las_file.curves_order):
        las_file.logs[curve_name] = np.array(data_lists[i], dtype=np.float64)
