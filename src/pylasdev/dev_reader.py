"""DEV (deviation survey) file reader.

Replaces dev_reader.py with modern Python 3, context managers,
and proper encoding handling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import read_with_encoding
from .exceptions import DEVReadError


def read_dev_file(
    file_path: str | Path,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Read a DEV (deviation survey) file and return data dictionary.

    Maintains backward compatibility — returns dict of numpy arrays.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        DEVReadError: If file cannot be read or parsed.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DEVReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise DEVReadError(f"Not a file: {file_path}")

    _detected_encoding, content = read_with_encoding(file_path, encoding)

    lines = content.splitlines()

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
    dev_dict: dict[str, Any] = {}
    names: list[str] = []
    header_found = False
    current_line = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        values = re.split(r"[\s\t]+", stripped)

        if not header_found:
            # First non-comment line = column names
            names = values
            for name in names:
                dev_dict[name] = np.zeros(data_lines, dtype=np.float64)
            header_found = True
        else:
            # Data lines
            for k in range(min(len(values), len(names))):
                try:
                    dev_dict[names[k]][current_line] = float(values[k])
                except (ValueError, IndexError):
                    dev_dict[names[k]][current_line] = np.nan
            current_line += 1

    return dev_dict
