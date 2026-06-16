"""DEV (deviation survey) file reader.

Replaces dev_reader.py with modern Python 3, context managers,
and proper encoding handling.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .data_reader import MAX_CURVES, MAX_DATA_LINES, MAX_TOTAL_ELEMENTS, _to_finite_float
from .encoding import read_with_encoding
from .exceptions import DEVReadError, LASEncodingError  # noqa: F401
from .models import DevFile

logger = logging.getLogger(__name__)


def read_dev_file(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> dict[str, Any]:
    """Read a DEV (deviation survey) file and return data dictionary.

    Maintains backward compatibility — returns dict of numpy arrays.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        DEVReadError: If file cannot be read or parsed.
        ValueError: If file exceeds max_file_size.
        LASEncodingError: If the explicit encoding parameter fails to decode
            the file.
    """
    dev = read_dev_file_as_object(file_path, encoding=encoding, max_file_size=max_file_size)
    return dev.to_dict()


def read_dev_file_as_object(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> DevFile:
    """Read a DEV file and return DevFile dataclass (new API).

    Same as read_dev_file() but returns the DevFile object directly
    instead of converting to dict.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

    Returns:
        DevFile dataclass with full parsed data.

    Raises:
        DEVReadError: If file cannot be read or parsed.
        ValueError: If file exceeds max_file_size.
        LASEncodingError: If the explicit encoding parameter fails to decode
            the file.

    Example:
        >>> from pylasdev import read_dev_file_as_object
        >>> dev = read_dev_file_as_object("survey.dev")
        >>> dev.column_order
        ['MD', 'TVD', 'X', 'Y']
        >>> dev.columns["MD"][:3]
        array([0., 100., 200.])
        >>> dev.to_dict()["MD"][:3]
        array([0., 100., 200.])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DEVReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise DEVReadError(f"Not a file: {file_path}")

    try:
        detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)
    except OSError as e:
        raise DEVReadError(f"Cannot read file: {file_path}") from e

    lines = content.splitlines()

    # Two-pass processing: first pass counts data lines to pre-allocate
    # numpy arrays at the correct size, second pass parses the data.
    # This avoids dynamic array resizing (O(n^2) behavior) and ensures
    # all columns have consistent lengths even with ragged input.

    # Pass 1: Count data lines (excluding comments, empty lines, and header)
    data_lines = 0
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not header_found:
            header_found = True  # First non-comment line is the header
        else:
            data_lines += 1

    if data_lines > MAX_DATA_LINES:
        raise DEVReadError(
            f"Data line count ({data_lines}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Pass 2: Parse header and data
    dev = DevFile()
    dev.source_file = str(file_path)
    dev.encoding = detected_encoding
    names: list[str] = []
    header_found = False
    current_line = 0
    warned_extra = False  # Track extra-column warning per file

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        values = stripped.split()

        if not header_found:
            # First non-comment line = column names
            names = values
            # Deduplicate column names with cross-base collision detection.
            # Ported from _deduplicate_curves in data_reader.py: uses an
            # output_names set + while-loop to ensure generated _N suffixes
            # don't collide with any name already in the output.
            seen: dict[str, int] = {}
            deduped_names: list[str] = []
            output_names: set[str] = set()
            for name in names:
                if name in seen:
                    seen[name] += 1
                    # Ensure the generated name doesn't collide with
                    # any name already in the output (including original names
                    # that match the _N suffix pattern).
                    suffix = seen[name]
                    new_name = f"{name}_{suffix}"
                    while new_name in output_names:
                        suffix += 1
                        new_name = f"{name}_{suffix}"
                    seen[name] = suffix
                    warnings.warn(
                        f"Duplicate DEV column name '{name}' renamed to "
                        f"'{new_name}'. Data may come from a file with "
                        f"repeated column names.",
                        stacklevel=2,
                    )
                    deduped_names.append(new_name)
                    output_names.add(new_name)
                else:
                    # Check for cross-base collisions where an
                    # original name matches a previously generated _N suffix.
                    # Input ["A","A","A_2"] should produce
                    # ["A","A_2","A_2_2"], not ["A","A_2","A_2"].
                    if name in output_names:
                        suffix = 2
                        new_name = f"{name}_{suffix}"
                        while new_name in output_names:
                            suffix += 1
                            new_name = f"{name}_{suffix}"
                        seen[name] = suffix
                        warnings.warn(
                            f"Duplicate DEV column name '{name}' renamed to "
                            f"'{new_name}'. Data may come from a file with "
                            f"repeated column names.",
                            stacklevel=2,
                        )
                        deduped_names.append(new_name)
                        output_names.add(new_name)
                    else:
                        seen[name] = 1
                        deduped_names.append(name)
                        output_names.add(name)
            names = deduped_names
            if len(names) > MAX_CURVES:
                raise DEVReadError(
                    f"Column count ({len(names)}) exceeds maximum allowed "
                    f"({MAX_CURVES}). The file may be malformed or corrupt."
                )
            if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                raise DEVReadError(
                    f"Total allocation ({len(names)} columns x {data_lines} lines = "
                    f"{len(names) * data_lines} elements) exceeds maximum allowed "
                    f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
                )
            for name in names:
                dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
            dev.column_order = list(names)
            header_found = True
        else:
            # Data lines
            # Warn about extra columns being silently discarded
            if len(values) > len(names) and not warned_extra:
                warned_extra = True
                logger.warning(
                    "Data line has %d values but only %d columns declared "
                    "in the header. Extra columns are discarded.",
                    len(values),
                    len(names),
                )
            for k in range(min(len(values), len(names))):
                try:
                    dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                except IndexError:
                    dev.columns[names[k]][current_line] = np.nan
            current_line += 1

    return dev
