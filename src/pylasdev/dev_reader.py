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

from .data_reader import (
    MAX_CURVES,
    MAX_DATA_LINES,
    MAX_TOKENS_PER_LINE,
    MAX_TOTAL_ELEMENTS,
    _to_finite_float,
)
from .encoding import read_with_encoding
from .exceptions import DEVReadError, LASEncodingError  # noqa: F401
from .models import DevFile

logger = logging.getLogger(__name__)

# Lightweight DEV column name alias mapping for common variants found in
# deviation survey files.  Unlike the full mnem_base.py database (2000+
# LAS curve aliases), this mapping is DEV-specific and covers the most
# common column name variants encountered in the wild.
_DEV_ALIASES: dict[str, str] = {
    # Measured depth variants
    "MDKB": "MD",
    "MDSS": "MD",
    "MDRKB": "MD",
    # True vertical depth variants
    "TVDKB": "TVD",
    "TVDSS": "TVD",
    "TVDBML": "TVD",
    # Inclination variants
    "INCL": "INC",
    # Azimuth variants
    "AZIM": "AZI",
    # Easting (X) variants
    "UTMX": "X",
    # Northing (Y) variants
    "UTMY": "Y",
}


def _normalize_dev_column(name: str) -> str:
    """Normalize a DEV column name through the alias mapping.

    Performs case-insensitive lookup: the input name is uppercased,
    matched against the alias table, and the canonical name is returned.
    Names not in the alias table are returned unchanged.

    Args:
        name: Raw column name from the header line.

    Returns:
        Canonical column name if an alias exists, otherwise the original.
    """
    return _DEV_ALIASES.get(name.upper(), name)


def _deduplicate_dev_columns(names: list[str]) -> list[str]:
    """Deduplicate DEV column names with cross-base collision detection.

    Ported from ``_deduplicate_curves`` in ``data_reader.py``.  Uses an
    ``output_names`` set + while-loop to ensure generated ``_N`` suffixes
    don't collide with any name already in the output.

    Args:
        names: Raw column names from the header line.

    Returns:
        Deduplicated column names.  A warning is emitted for each
        duplicate.
    """
    seen: dict[str, int] = {}
    deduped: list[str] = []
    output_names: set[str] = set()
    for name in names:
        if name in seen:
            seen[name] += 1
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
            deduped.append(new_name)
            output_names.add(new_name)
        else:
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
                deduped.append(new_name)
                output_names.add(new_name)
            else:
                seen[name] = 1
                deduped.append(name)
                output_names.add(name)
    return deduped


def _is_float_token(token: str) -> bool:
    """Check if a token can be parsed as a float.

    Handles standard scientific notation (e/E) and Fortran D-exponent
    notation (d/D) used by some scientific software.

    Used by ``_detect_dev_format`` to distinguish numeric data lines
    from column-name header lines.
    """
    # Reject special float strings (nan, inf, -inf, infinity) that
    # float() can parse but are not meaningful numeric well-log data.
    lower = token.replace("D", "E").replace("d", "e")
    if lower.lower() in ("nan", "inf", "-inf", "infinity", "+inf", "-infinity", "+infinity"):
        return False
    try:
        float(lower)
        return True
    except ValueError:
        return False


def _detect_dev_format(content_entries: list[tuple[int, str]]) -> tuple[str, int]:
    """Detect DEV file format from first few content lines.

    Inspects the first 1-3 non-comment, non-empty lines to determine
    which of the three supported DEV format variants the file uses.

    Args:
        content_entries: List of ``(line_index, stripped_content)``
            for the first non-comment, non-empty lines.

    Returns:
        Tuple of ``(format_type, skip_content_lines)`` where *format_type*
        is ``"simple"``, ``"dug"``, or ``"headerless"``, and
        *skip_content_lines* is the number of content-bearing lines to
        skip before data starts (used by Pass 1 counting).
    """
    if not content_entries:
        return ("simple", 1)

    first_tokens = content_entries[0][1].split()

    # DUG Insight format detection — two patterns.
    #
    # Pattern A: single-token first line = column count (no separate title).
    #   "4"              ← integer column count (also serves as title)
    #   "MD TVD X Y"     ← header with non-numeric tokens
    #   2 content lines before data: count + header.
    if len(first_tokens) == 1:
        try:
            int(first_tokens[0])
        except ValueError:
            pass
        else:
            if len(content_entries) >= 2:
                second_tokens = content_entries[1][1].split()
                if any(not _is_float_token(t) for t in second_tokens):
                    return ("dug", 2)

    # Pattern B: multi-word title, integer column count, header.
    #   "Deviation survey for Well-1"   ← descriptive title
    #   "4"                              ← integer column count
    #   "MDKB TVDSS X Y"                ← header with non-numeric tokens
    #   3 content lines before data: title + count + header.
    if len(content_entries) >= 3:
        second_tokens = content_entries[1][1].split()
        if len(second_tokens) == 1:
            try:
                int(second_tokens[0])
            except ValueError:
                pass
            else:
                third_tokens = content_entries[2][1].split()
                if third_tokens and any(not _is_float_token(t) for t in third_tokens):
                    return ("dug", 3)

    # Headerless format: every token on the first content line parses as
    # a float (no column names present).  No content lines to skip.
    if first_tokens and all(_is_float_token(t) for t in first_tokens):
        return ("headerless", 0)

    return ("simple", 1)


def read_dev_file(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
    delimiter: str | None = None,
    normalize_aliases: bool = True,
) -> dict[str, Any]:
    """Read a DEV (deviation survey) file and return data dictionary.

    Maintains backward compatibility — returns dict of numpy arrays.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.
        delimiter: Column delimiter.  ``None`` (default) auto-detects
            comma vs whitespace from the header line.  Pass ``" "`` for
            whitespace-only, ``","`` for comma-delimited files.
        normalize_aliases: If ``True`` (default), apply DEV-specific alias
            normalization to column names (e.g. ``MDKB`` → ``MD``).

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        DEVReadError: If file cannot be read or parsed.
        ValueError: If file exceeds max_file_size.
        LASEncodingError: If the explicit encoding parameter fails to decode
            the file.
    """
    dev = read_dev_file_as_object(
        file_path,
        encoding=encoding,
        max_file_size=max_file_size,
        delimiter=delimiter,
        normalize_aliases=normalize_aliases,
    )
    return dev.to_dict()


def read_dev_file_as_object(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
    delimiter: str | None = None,
    normalize_aliases: bool = True,
) -> DevFile:
    """Read a DEV file and return DevFile dataclass (new API).

    Same as read_dev_file() but returns the DevFile object directly
    instead of converting to dict.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a ValueError is raised.
        delimiter: Column delimiter.  ``None`` (default) auto-detects
            comma vs whitespace from the header line.  Pass ``" "`` for
            whitespace-only, ``","`` for comma-delimited files.
        normalize_aliases: If ``True`` (default), apply DEV-specific alias
            normalization to column names (e.g. ``MDKB`` → ``MD``).

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

    # --- Gather first few content lines for format detection ---
    # Scan past comments and empty lines to collect up to 3 content-bearing
    # lines.  These are used for both format detection and delimiter
    # auto-detection.
    content_entries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            content_entries.append((i, stripped))
            if len(content_entries) >= 3:
                break

    format_type, skip_content_lines = _detect_dev_format(content_entries)

    # --- Auto-detect delimiter ---
    if delimiter is None:
        # Use the actual header line for delimiter detection:
        # - simple:   first content line
        # - dug:      third content line (the header — skip title+count)
        # - headerless: first content line (the data)
        if format_type == "dug" and len(content_entries) > skip_content_lines - 1:
            hdr = content_entries[skip_content_lines - 1][1]
        elif content_entries:
            hdr = content_entries[0][1]
        else:
            hdr = ""

        if hdr:
            # If the header contains commas and splitting on comma
            # yields at least as many tokens as splitting on whitespace,
            # treat the file as comma-delimited.  Using >= (not >) so
            # that "MD, TVD, INC, AZI" (commas with trailing spaces)
            # correctly detects comma delimiter.
            comma_tokens = [t for t in hdr.split(",") if t.strip()]
            space_tokens = hdr.split()
            if len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                delimiter = ","
            else:
                delimiter = " "
        else:
            delimiter = " "

    # --- Pass 1: Count data lines ---
    # skip_content_lines is returned by _detect_dev_format:
    #   dug:         2 (pattern A) or 3 (pattern B)
    #   headerless:  0 (no header — first content line IS data)
    #   simple:      1 (header only)
    content_seen = 0
    data_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content_seen += 1
        if content_seen > skip_content_lines:
            data_lines += 1

    if data_lines > MAX_DATA_LINES:
        raise DEVReadError(
            f"Data line count ({data_lines}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # --- Pass 2: Parse header and data ---
    dev = DevFile()
    dev.source_file = str(file_path)
    dev.encoding = detected_encoding
    names: list[str] = []
    content_seen = 0
    current_line = 0
    warned_extra = False  # Track extra-column warning per file

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        content_seen += 1

        if delimiter == " ":
            values = stripped.split(maxsplit=MAX_TOKENS_PER_LINE)
        else:
            values = [v.strip() for v in stripped.split(delimiter, maxsplit=MAX_TOKENS_PER_LINE)]

        if format_type == "headerless":
            if content_seen == 1:
                # Auto-generate column names from the first data row.
                names = [f"col_{i}" for i in range(len(values))]
                if len(names) > MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                dev.column_order = list(names)
                # Store first data row.
                for k in range(len(names)):
                    try:
                        dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                    except IndexError:
                        dev.columns[names[k]][current_line] = np.nan
                current_line += 1
            else:
                # Remaining data rows.
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

        elif format_type == "dug":
            if content_seen < skip_content_lines:
                # Skip title line(s) and column-count line.
                continue
            elif content_seen == skip_content_lines:
                # Header line — parse column names.
                names = values
                if not names:
                    raise DEVReadError(
                        "Empty header line in DUG-format DEV file. "
                        "Expected column names on the third content line."
                    )
                # Apply alias normalization before deduplication so that
                # variants like MDKB and MD are recognized as duplicates.
                if normalize_aliases:
                    names = [_normalize_dev_column(n) for n in names]
                # Deduplicate column names.
                names = _deduplicate_dev_columns(names)
                if len(names) > MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                dev.column_order = list(names)
            else:
                # Data lines.
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

        else:  # simple header format
            if content_seen == 1:
                # First non-comment line = column names.
                names = values
                if not names:
                    raise DEVReadError(
                        "Empty header line in DEV file. "
                        "Expected column names on the first content line."
                    )
                # Apply alias normalization before deduplication so that
                # variants like MDKB and MD are recognized as duplicates.
                if normalize_aliases:
                    names = [_normalize_dev_column(n) for n in names]
                # Deduplicate column names.
                names = _deduplicate_dev_columns(names)
                if len(names) > MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                dev.column_order = list(names)
            else:
                # Data lines.
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
