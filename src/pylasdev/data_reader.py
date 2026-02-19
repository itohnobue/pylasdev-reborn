"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import warnings

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
        values = stripped.split()
        # In proper wrapped mode, first line has only the depth value (1 value).
        # If it has as many or more values as curves, it's non-wrapped.
        return len(values) < curve_count

    return True  # No data found, default to wrapped


def _deduplicate_curves(las_file: LASFile) -> None:
    """Detect and rename duplicate curve names with warning.

    Appends _2, _3, etc. to duplicate mnemonics so each curve gets
    its own array in las_file.logs. Also updates the corresponding
    CurveDefinition objects to keep curves_order and curves in sync.
    """
    seen: dict[str, int] = {}
    new_order: list[str] = []
    for idx, name in enumerate(las_file.curves_order):
        if name in seen:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"
            warnings.warn(
                f"Duplicate curve mnemonic '{name}' renamed to '{new_name}'. "
                "Data may come from a file with repeated curve names.",
                stacklevel=3,
            )
            new_order.append(new_name)
            # Keep CurveDefinition in sync with renamed curves_order
            if idx < len(las_file.curves):
                if not las_file.curves[idx].original_mnemonic:
                    las_file.curves[idx].original_mnemonic = name
                las_file.curves[idx].mnemonic = new_name
        else:
            seen[name] = 1
            new_order.append(name)
    if new_order != las_file.curves_order:
        las_file.curves_order = new_order


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

        values = stripped.split()

        for i in range(min(len(values), curve_count)):
            try:
                las_file.logs[las_file.curves_order[i]][current_line] = float(values[i])
            except (ValueError, IndexError):
                las_file.logs[las_file.curves_order[i]][current_line] = null_value

        # Fill remaining curves with null_value when line has fewer values
        for i in range(len(values), curve_count):
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
    # Deduplicate curve names before reading
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

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
