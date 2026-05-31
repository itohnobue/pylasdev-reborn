"""DEV (deviation survey) file reader.

Replaces dev_reader.py with modern Python 3, context managers,
and proper encoding handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .encoding import read_with_encoding
from .exceptions import DEVReadError
from .models import DevFile


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

    detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)

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

    # Pass 2: Parse header and data
    dev = DevFile()
    dev.source_file = str(file_path)
    dev.encoding = detected_encoding
    names: list[str] = []
    header_found = False
    current_line = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        values = stripped.split()

        if not header_found:
            # First non-comment line = column names
            names = values
            for name in names:
                dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
            dev.column_order = list(names)
            header_found = True
        else:
            # Data lines
            for k in range(min(len(values), len(names))):
                try:
                    dev.columns[names[k]][current_line] = float(values[k])
                except (ValueError, IndexError):
                    dev.columns[names[k]][current_line] = np.nan
            current_line += 1

    return dev
