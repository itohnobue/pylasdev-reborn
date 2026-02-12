"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import re

import numpy as np

from .models import LASFile


def read_ascii_data(content: str, las_file: LASFile, data_line_count: int) -> None:
    """Read the ~A (ASCII data) section and populate las_file.logs.

    Args:
        content: Full file content string.
        las_file: LASFile object with curves_order already populated.
        data_line_count: Number of data lines (from pre-scan).
    """
    curve_count = len(las_file.curves_order)
    if curve_count == 0:
        return

    lines = content.splitlines()
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

        if stripped.startswith("~A"):
            in_ascii = True
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # First data line found — check value count
        values = re.split(r"[\s\t]+", stripped)
        # In proper wrapped mode, first line has only the depth value (1 value).
        # If it has as many or more values as curves, it's non-wrapped.
        return len(values) < curve_count

    return True  # No data found, default to wrapped


def _read_normal(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    data_line_count: int,
) -> None:
    """Read non-wrapped ASCII data. One depth step per line."""
    # Pre-allocate arrays
    for curve_name in las_file.curves_order:
        las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)

    in_ascii = False
    current_line = 0
    null_value = float(las_file.well.get("NULL", "-999.25"))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("~"):
            if stripped.startswith("~A"):
                in_ascii = True
            else:
                if in_ascii:
                    break  # End of ASCII section — new section started
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        for i in range(min(len(values), curve_count)):
            try:
                las_file.logs[las_file.curves_order[i]][current_line] = float(values[i])
            except (ValueError, IndexError):
                las_file.logs[las_file.curves_order[i]][current_line] = null_value

        current_line += 1


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
    # Accumulate into lists, convert to numpy at end
    data_lists: list[list[float]] = [[] for _ in range(curve_count)]

    in_ascii = False
    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves
    null_value = float(las_file.well.get("NULL", "-999.25"))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("~"):
            if stripped.startswith("~A"):
                in_ascii = True
            else:
                if in_ascii:
                    break  # End of ASCII section — new section started
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        if depth_line:
            # Depth line: single value = depth for this step
            try:
                data_lists[0].append(float(values[0]))
            except (ValueError, IndexError):
                data_lists[0].append(null_value)
            depth_line = False
            counter = 0
        else:
            # Data lines: values for remaining curves
            for val_str in values:
                counter += 1
                try:
                    data_lists[counter].append(float(val_str))
                except (ValueError, IndexError):
                    if counter < curve_count:
                        data_lists[counter].append(null_value)

                if counter >= curve_count - 1:
                    # All curves for this depth step are complete
                    counter = 0
                    depth_line = True

    # Convert lists to numpy arrays
    for i, curve_name in enumerate(las_file.curves_order):
        las_file.logs[curve_name] = np.array(data_lists[i], dtype=np.float64)
