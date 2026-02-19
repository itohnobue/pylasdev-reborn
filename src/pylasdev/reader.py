"""LAS file reader — main entry point.

Replaces las_reader.py with modern Python 3, proper encoding handling,
context managers, and no global state.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .data_reader import read_ascii_data
from .encoding import read_with_encoding
from .exceptions import LASReadError
from .models import LASFile
from .parser import LASParser


def read_las_file(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> dict[str, Any]:
    """Read a LAS file and return data dictionary.

    This is the main entry point, maintaining backward compatibility
    with the original pylasdev API (returns dict, not LASFile).

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override. If None, auto-detected.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

    Returns:
        Dictionary with keys: version, well, parameters, logs, curves_order.
        Well values are strings. Log values are numpy arrays.

    Raises:
        LASReadError: If file cannot be read.
        LASParseError: If file content cannot be parsed.
        ValueError: If file exceeds max_file_size.

    Warns:
        UserWarning: If LAS version is > 3.0 (unsupported but attempted).

    Example:
        >>> data = read_las_file("sample.las")
        >>> print(data['well']['WELL'])
        >>> print(data['logs']['DEPT'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LASReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise LASReadError(f"Not a file: {file_path}")

    # Read with encoding detection
    detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)

    # Parse header sections
    parser = LASParser(mnem_base)
    las_file = parser.parse(content)
    las_file.source_file = str(file_path)
    las_file.encoding = detected_encoding

    # Check version - warn on unsupported versions but try to read anyway
    try:
        vers = float(las_file.version.vers)
        if vers > 3.0:
            warnings.warn(
                f"LAS version {las_file.version.vers} is not officially supported. "
                "Only LAS 1.2, 2.0, and 3.0 are supported. "
                "Attempting to read anyway.",
                stacklevel=2,
            )
    except ValueError:
        pass  # Non-numeric version string — let it through

    # Read ASCII data section
    # For LAS 3.0, the parser already handles this
    # For LAS 1.2/2.0, use the dedicated data reader
    if not las_file.is_las30:
        read_ascii_data(content, las_file, parser._data_line_count)

    # Return legacy dict format for backward compatibility
    return las_file.to_dict()


def read_las_file_as_object(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
    max_file_size: int | None = None,
) -> LASFile:
    """Read a LAS file and return LASFile dataclass (new API).

    Same as read_las_file() but returns the LASFile object directly
    instead of converting to dict. Use this for richer metadata access.

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.

    Returns:
        LASFile dataclass with full parsed data.

    Raises:
        LASReadError: If file cannot be read.
        ValueError: If file exceeds max_file_size.

    Warns:
        UserWarning: If LAS version is > 3.0 (unsupported but attempted).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LASReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise LASReadError(f"Not a file: {file_path}")

    detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)

    parser = LASParser(mnem_base)
    las_file = parser.parse(content)
    las_file.source_file = str(file_path)
    las_file.encoding = detected_encoding

    # Check version - warn on unsupported versions but try to read anyway
    try:
        vers = float(las_file.version.vers)
        if vers > 3.0:
            warnings.warn(
                f"LAS version {las_file.version.vers} is not officially supported. "
                "Only LAS 1.2, 2.0, and 3.0 are supported. "
                "Attempting to read anyway.",
                stacklevel=2,
            )
    except ValueError:
        pass  # Non-numeric version string — let it through

    # Read ASCII data section
    # For LAS 3.0, the parser already handles this
    # For LAS 1.2/2.0, use the dedicated data reader
    if not las_file.is_las30:
        read_ascii_data(content, las_file, parser._data_line_count)

    return las_file
