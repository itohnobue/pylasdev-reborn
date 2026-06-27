"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data_reader import _get_null_value
from .exceptions import LASWriteError
from .models import LASFile

# Control characters except space and tab (which are valid LAS whitespace).
# Matches \x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F (DEL), \x85 (NEL),
# \u2028 (LINE SEPARATOR), and \u2029 (PARAGRAPH SEPARATOR).
# The Unicode line break characters are treated as line breaks by Python's
# splitlines() but are not caught by \n/\r replacement.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85\u2028\u2029]")

# Pattern to detect section-header-like leading content after sanitization.
# A value starting with ~ followed by an alphabetic character mimics a
# LAS section header and could split the file if injected verbatim.
_LEADING_SECTION_RE = re.compile(r"^~([A-Za-z])")


def _sanitize_las_value(value: str) -> str:
    """Sanitize a string for safe inclusion in LAS output.

    Removes newlines, control characters, and leading ~A-Z patterns
    that could be interpreted as section headers when injected into
    LAS output text.

    Args:
        value: Raw string value.

    Returns:
        Sanitized string safe for LAS output.
    """
    # Strip all newline characters (prevents section injection).
    # Unicode line separators (\u2028, \u2029) and NEL (\x85) are also
    # treated as line breaks by Python's splitlines().
    value = (
        value.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
        .replace("\x85", " ")
    )
    # Strip control characters
    value = _CONTROL_CHARS_RE.sub("", value)
    # If the value now starts with ~ followed by a letter, remove the leading ~
    # to prevent it from being interpreted as a section header
    value = _LEADING_SECTION_RE.sub(r"\1", value, count=1)
    return value


def write_las_file(
    file_path: str | Path,
    las_data: dict[str, Any] | LASFile,
    encoding: str = "utf-8",
    precision: str = ".8g",
) -> None:
    """Write LAS data to file.

    Args:
        file_path: Output file path.
        las_data: LAS data as dict (legacy format) or LASFile object.
        encoding: Output file encoding (default: utf-8). Always writes
            using this encoding regardless of the input's original encoding.
            UTF-8 is recommended for maximum compatibility and reliable
            re-reading of files containing non-ASCII characters (e.g.
            Cyrillic curve mnemonics in Russian LAS files).
        precision: Format specifier for numeric data values (default: '.8g').
            Pass a Python format spec like '.6g' or '.10e' for more precision.

    Raises:
        LASWriteError: If file cannot be written.
    """
    file_path = Path(file_path)

    if isinstance(las_data, dict):
        # F34: Wrap from_dict in try/except so that malformed input
        # (e.g., non-numeric log values that fail np.array(dtype=np.float64)
        # in models.py) raises LASWriteError instead of raw ValueError.
        try:
            las_file = LASFile.from_dict(las_data)
        except (ValueError, TypeError) as e:
            raise LASWriteError(f"Cannot create LASFile from dict: {e}") from e
    elif isinstance(las_data, LASFile):
        las_file = las_data
    else:
        raise LASWriteError(
            f"write_las_file expects a dict or LASFile, got {type(las_data).__name__}"
        )

    # Always write with the specified encoding (default: utf-8).
    try:
        content = _generate_las_content(las_file, precision)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

    try:
        file_path.write_text(content, encoding=encoding)
    except OSError as e:
        raise LASWriteError(f"Cannot write to {file_path}: {e}") from e


def _generate_las_content(las_file: LASFile, precision: str = ".8g") -> str:
    """Generate LAS file content string with metadata preservation."""
    lines: list[str] = []
    lines.extend(_write_version_section(las_file))
    lines.extend(_write_well_section(las_file))
    lines.extend(_write_curve_section(las_file))
    lines.extend(_write_parameter_section(las_file))
    lines.extend(_write_other_section(las_file))
    lines.extend(_write_ascii_sections(las_file, precision))
    return "\n".join(lines) + "\n"


def _write_version_section(las_file: LASFile) -> list[str]:
    """Write ~V Version section."""
    lines: list[str] = []
    is_las30 = las_file.is_las30
    lines.append("~VERSION INFORMATION")
    vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0" if is_las30 else "CWLS LOG ASCII STANDARD"
    lines.append(f" VERS.   {las_file.version.vers}  : {vers_desc}")
    # Always write WRAP=NO since we write one line per depth step (non-wrapped)
    lines.append(" WRAP.   NO  : ONE LINE PER DEPTH STEP")
    if is_las30:
        dlm_desc = "DELIMITING CHARACTER BETWEEN DATA COLUMNS"
        lines.append(f" DLM .                        {las_file.version.dlm} : {dlm_desc}")
    lines.append("")
    return lines


def _write_well_section(las_file: LASFile) -> list[str]:
    """Write ~W Well section."""
    lines: list[str] = []
    lines.append("~WELL INFORMATION")
    for key, value in las_file.well.entries.items():
        lines.append(f" {key}.   {_sanitize_las_value(value)}  :")
    lines.append("")
    return lines


def _write_curve_section(las_file: LASFile) -> list[str]:
    """Write ~C Curve section — preserve units and descriptions."""
    lines: list[str] = []
    is_las30 = las_file.is_las30
    lines.append("~CURVE INFORMATION")
    for curve in las_file.curves:
        unit = curve.unit if curve.unit else ""
        desc = curve.description if curve.description else ""

        if is_las30 and curve.data_format:
            format_str = f"{{{curve.data_format}"
            if curve.array_info and curve.array_info.time_offset is not None:
                # Format time_offset as int if it's a whole number, float otherwise
                offset = curve.array_info.time_offset
                if offset == int(offset):
                    format_str += f":{int(offset)}"
                else:
                    format_str += f":{offset}"
            format_str += "}"
            desc = f"{desc}  {format_str}"

        api = f"  {curve.api_code}" if curve.api_code else ""
        lines.append(
            f" {_sanitize_las_value(curve.mnemonic)}.{unit}{api}  : {_sanitize_las_value(desc)}"
        )
    lines.append("")
    return lines


def _write_parameter_section(las_file: LASFile) -> list[str]:
    """Write ~P Parameter section."""
    lines: list[str] = []
    if not las_file.parameters:
        return lines
    is_las30 = las_file.is_las30
    lines.append("~PARAMETER INFORMATION")
    for param in las_file.parameters:
        unit = param.unit if param.unit else ""
        desc = param.description if param.description else ""

        if is_las30 and param.zone:
            zone_str = f" | {param.zone.zone_name}"
            if param.zone.zone_index is not None:
                zone_str += f"[{param.zone.zone_index}]"
            desc = f"{desc}{zone_str}"

        lines.append(
            f" {param.mnemonic}.{unit}  {_sanitize_las_value(param.value)}  : {_sanitize_las_value(desc)}"
        )
    lines.append("")
    return lines


def _write_other_section(las_file: LASFile) -> list[str]:
    """Write ~O Other section."""
    lines: list[str] = []
    if las_file.other and las_file.other.strip():
        lines.append("~OTHER")
        # Sanitize each line of free-form other content against section injection
        for line in las_file.other.splitlines():
            sanitized = _sanitize_las_value(line)
            if sanitized.strip():
                lines.append(sanitized)
        lines.append("")
    return lines


def _write_ascii_sections(las_file: LASFile, precision: str = ".8g") -> list[str]:
    """Write ~A ASCII data section(s)."""
    lines: list[str] = []
    null_value = _get_null_value(las_file.well)
    delimiter = las_file.version.delimiter_char

    if las_file.data_sections:
        # LAS 3.0: Multiple data sections
        for section in las_file.data_sections:
            section_name = f" {section.name}" if section.name else ""
            lines.append(f"~A{section_name}")
            lines.extend(
                _format_data_rows(
                    section.curves_order,
                    section.data,
                    section.string_data,
                    null_value,
                    delimiter,
                    precision,
                )
            )
    else:
        # Legacy single data section
        curve_names = las_file.curves_order
        if any(name in las_file.logs for name in curve_names):
            lines.append("~A  " + "  ".join(curve_names))
            lines.extend(
                _format_data_rows(
                    curve_names,
                    las_file.logs,
                    las_file.string_data,
                    null_value,
                    delimiter,
                    precision,
                )
            )
    return lines


def _format_data_rows(
    curve_names: list[str],
    data: dict[str, NDArray[np.float64]],
    string_data: dict[str, NDArray[np.str_]],
    null_value: float,
    delimiter: str,
    precision: str = ".8g",
) -> list[str]:
    """Format data rows for a section — handles both legacy and LAS 3.0 sections.

    Builds one line per depth step with delimiter-separated values.
    String curves are emitted as-is; numeric curves use configurable formatting.
    Missing values are filled with the null_value. NaN values are output as null.
    """
    lines: list[str] = []
    if not curve_names:
        return lines

    # Pre-extract curve data arrays to avoid O(rows x curves) dict lookups
    # inside the inner loop (F-23 performance optimization).
    curve_arrays: list[tuple[NDArray[np.float64] | NDArray[np.str_] | None, bool]] = []
    for name in curve_names:
        if name in string_data:
            curve_arrays.append((string_data[name], True))
        elif name in data:
            curve_arrays.append((data[name], False))
        else:
            curve_arrays.append((None, False))

    # Derive row count from max length across all curves, not just the first.
    # This handles per-curve variable-length data (e.g. curves populated
    # from different data sections in LAS 3.0).  Shorter curves are padded
    # with null_value in the inner loop.
    num_rows = max(
        (len(arr) for arr, _ in curve_arrays if arr is not None),
        default=0,
    )
    if num_rows == 0:
        return lines

    for i in range(num_rows):
        row_values: list[str] = []
        for arr, is_string in curve_arrays:
            if arr is None or i >= len(arr):
                row_values.append(_format_number(null_value, precision, null_value))
            elif is_string:
                row_values.append(str(arr[i]))
            else:
                val = arr[i]
                if np.isnan(val) or np.isinf(val):
                    row_values.append(_format_number(null_value, precision, null_value))
                else:
                    row_values.append(_format_number(val, precision, null_value))
        lines.append(delimiter.join(row_values))
    return lines


def _format_number(value: float, precision: str = ".8g", null_value: float | None = None) -> str:
    """Format a numeric value with configurable precision.

    Handles whole numbers as integers to avoid unnecessary decimal noise.
    Self-protecting against NaN and Inf — caller guards these already,
    but a NaN slipping through would otherwise crash on ``int(np.nan)``.
    If a ``null_value`` is provided and a NaN/Inf value slips past the
    primary guard, the null value is output instead of formatting the
    NaN/Inf directly (which would produce invalid LAS values like
    ``"nan"`` / ``"inf"``).
    """
    if np.isnan(value) or np.isinf(value):
        if null_value is not None:
            return format(null_value, precision)
        return format(float(value), precision)
    if value == int(value):
        return format(int(value), precision)
    return format(float(value), precision)
