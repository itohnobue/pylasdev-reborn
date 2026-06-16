"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import warnings

import numpy as np

from .exceptions import LASParseError
from .models import LASFile, WellSection

# Maximum bounds for array allocations to prevent memory exhaustion
# from malformed or malicious files. Overridable by setting the module constant.
MAX_CURVES = 100_000
MAX_DATA_LINES = 10_000_000
# Combined allocation guard: curve_count × data_line_count must not exceed this.
# Individual MAX_CURVES and MAX_DATA_LINES checks alone are insufficient — a file
# with 1K curves × 1M lines (1B elements ≈ 8 GB) passes both guards independently
# but OOMs during np.zeros pre-allocation. Overridable by setting module constant.
MAX_TOTAL_ELEMENTS = 1_000_000_000


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
        return float(well.get("NULL", default))
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
        val = float(value_str)
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
    curve_count = len(las_file.curves_order)
    if curve_count == 0:
        return

    if curve_count > MAX_CURVES:
        raise LASParseError(
            f"Curve count ({curve_count}) exceeds maximum allowed ({MAX_CURVES}). "
            f"The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks where individual
    # curve_count and data_line_count checks pass but product exhausts memory.
    if curve_count * data_line_count > MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Total allocation ({curve_count} curves × {data_line_count} lines = "
            f"{curve_count * data_line_count} elements) exceeds maximum allowed "
            f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    wrap_mode = las_file.version.wrap.upper() == "YES"

    if wrap_mode:
        # Auto-detect wrap mismatch: if the first data line has >= curve_count
        # values, the data is actually non-wrapped despite WRAP=YES header.
        # This handles mislabeled files (e.g., Petrel exports).
        actual_wrap = _detect_actual_wrap(lines, curve_count)
        if actual_wrap:
            _read_wrapped(lines, las_file, curve_count)
        else:
            _read_normal(lines, las_file, curve_count, data_line_count)
    else:
        _read_normal(lines, las_file, curve_count, data_line_count)


def _detect_actual_wrap(lines: list[str], curve_count: int) -> bool:
    """Detect if data is actually wrapped by checking the first data line.

    In true wrapped mode, the first data line has only 1 value (the depth).
    In non-wrapped mode (even if WRAP=YES header), each line has >= curve_count values.

    Returns:
        True if data is actually wrapped, False if non-wrapped despite header.
    """
    in_ascii = False
    for line in lines:
        stripped = line.strip()

        if _is_section_header(stripped):
            if stripped[1].upper() == "A":
                in_ascii = True
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # First data line found — check value count
        values = stripped.split()
        # In proper wrapped mode, first line has only the depth value (1 value).
        # If it has as many or more values as curves, it's non-wrapped.
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
            # F-22: Ensure the generated name doesn't collide with
            # any name already in the output (including original names
            # that match the _N suffix pattern).
            suffix = seen[name]
            new_name = f"{name}_{suffix}"
            while new_name in output_names:
                suffix += 1
                new_name = f"{name}_{suffix}"
            # Update the seen counter to match the actual suffix used
            seen[name] = suffix
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
                suffix = 2
                new_name = f"{name}_{suffix}"
                while new_name in output_names:
                    suffix += 1
                    new_name = f"{name}_{suffix}"
                seen[name] = suffix
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
            f"Total allocation ({curve_count} curves × {data_line_count} lines = "
            f"{curve_count * data_line_count} elements) exceeds maximum allowed "
            f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    # Pre-allocate arrays
    for curve_name in las_file.curves_order:
        las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)

    null_value = _get_null_value(las_file.well)

    in_ascii = False
    current_line = 0

    for line in lines:
        stripped = line.strip()

        # F-20: Align section detection with parser.py's SECTION_PATTERN
        # (~[A-Za-z]).  Lines starting with ~ but without an alphabetic
        # section letter (e.g. bare ~, ~~~, etc.) are ignored.
        if _is_section_header(stripped):
            if stripped[1].upper() == "A":
                in_ascii = True
            else:
                if in_ascii:
                    break  # End of ASCII section — new section started
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = stripped.split()

        for i in range(min(len(values), curve_count)):
            try:
                las_file.logs[las_file.curves_order[i]][current_line] = _to_finite_float(
                    values[i], null_value
                )
            except IndexError:
                # IndexError can occur when curve_count was reduced after deduplication
                # (the pre-allocated arrays are sized for the deduplicated curve_count
                # which may be smaller than the original data column count)
                if (
                    i < curve_count
                    and current_line < las_file.logs[las_file.curves_order[i]].shape[0]
                ):
                    las_file.logs[las_file.curves_order[i]][current_line] = null_value

        # Fill remaining curves with null_value when line has fewer values
        for i in range(len(values), curve_count):
            if current_line < len(las_file.logs[las_file.curves_order[i]]):
                las_file.logs[las_file.curves_order[i]][current_line] = null_value

        current_line += 1

    # Trim arrays when ~A section ended early (fewer data lines than declared).
    # Pre-allocated np.zeros tail would otherwise expose 0.0 values that differ
    # from null_value, corrupting downstream analysis.
    if current_line < data_line_count:
        for curve_name in las_file.curves_order:
            las_file.logs[curve_name] = las_file.logs[curve_name][:current_line]


def _read_wrapped(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
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
            if stripped[1].upper() == "A":
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
    if curve_count * _count > MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Total allocation ({curve_count} curves × {_count} lines = "
            f"{curve_count * _count} elements) exceeds maximum allowed "
            f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    # Accumulate into lists, convert to numpy at end
    data_lists: list[list[float]] = [[] for _ in range(curve_count)]

    null_value = _get_null_value(las_file.well)

    in_ascii = False
    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves

    for line in lines:
        stripped = line.strip()

        # F-20: Align section detection with parser.py's SECTION_PATTERN
        # (~[A-Za-z]).  Only treat lines as section headers when the
        # character after ~ is alphabetic.
        if _is_section_header(stripped):
            if stripped[1].upper() == "A":
                in_ascii = True
            else:
                if in_ascii:
                    break  # End of ASCII section — new section started
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = stripped.split()

        if depth_line:
            # Depth line: single value = depth for this step
            if len(values) > 1:
                warnings.warn(
                    f"Wrapped mode: depth line has {len(values)} values, expected 1. "
                    f"Extra values discarded. Line content: '{stripped[:80]}'",
                    stacklevel=2,
                )
            try:
                data_lists[0].append(_to_finite_float(values[0], null_value))
            except IndexError:
                data_lists[0].append(null_value)
            depth_line = False
            counter = 0
        else:
            # Data lines: values for remaining curves
            for val_str in values:
                counter += 1
                try:
                    data_lists[counter].append(_to_finite_float(val_str, null_value))
                except IndexError:
                    if counter < curve_count:
                        data_lists[counter].append(null_value)

                if counter >= curve_count - 1:
                    # All curves for this depth step are complete.
                    # Break to discard any extra values on this line
                    # (prevents silent misalignment if a line has
                    # more values than expected).
                    counter = 0
                    depth_line = True
                    break

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
