"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import LASWriteError
from .models import LASFile


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
        encoding: Output file encoding (default: utf-8). If las_data is a
            dict, this parameter is used directly (dicts from to_dict()
            do not carry an encoding key). If las_data is a LASFile object,
            the object's .encoding attribute is used instead of this default.
        precision: Format specifier for numeric data values (default: '.8g').
            Pass a Python format spec like '.6g' or '.10e' for more precision.

    Raises:
        LASWriteError: If file cannot be written.
    """
    file_path = Path(file_path)

    if isinstance(las_data, dict):
        las_file = LASFile.from_dict(las_data)
        # Use detected encoding from the data dict if available,
        # otherwise fall back to the explicit parameter.
        file_encoding = las_data.get("encoding", encoding)
    else:
        las_file = las_data
        file_encoding = las_file.encoding or encoding

    content = _generate_las_content(las_file, precision)

    try:
        file_path.write_text(content, encoding=file_encoding)
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
        lines.append(f" {key}.   {value}  :")
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
        lines.append(f" {curve.mnemonic}.{unit}{api}  : {desc}")
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

        lines.append(f" {param.mnemonic}.{unit}  {param.value}  : {desc}")
    lines.append("")
    return lines


def _write_other_section(las_file: LASFile) -> list[str]:
    """Write ~O Other section."""
    lines: list[str] = []
    if las_file.other and las_file.other.strip():
        lines.append("~OTHER")
        lines.append(las_file.other.rstrip())
        lines.append("")
    return lines


def _write_ascii_sections(las_file: LASFile, precision: str = ".8g") -> list[str]:
    """Write ~A ASCII data section(s)."""
    lines: list[str] = []
    try:
        null_value = float(las_file.well.get("NULL", "-999.25"))
    except (ValueError, TypeError):
        null_value = -999.25
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
                    las_file.string_data,
                    null_value,
                    delimiter,
                    precision,
                )
            )
    else:
        # Legacy single data section
        curve_names = las_file.curves_order
        if curve_names and curve_names[0] in las_file.logs:
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
    if not curve_names or curve_names[0] not in data:
        return lines
    num_rows = len(data[curve_names[0]])
    for i in range(num_rows):
        row_values: list[str] = []
        for name in curve_names:
            if name in string_data and i < len(string_data[name]):
                row_values.append(str(string_data[name][i]))
            elif name in data and i < len(data[name]):
                val = data[name][i]
                if np.isnan(val):
                    row_values.append(_format_number(null_value, precision))
                else:
                    row_values.append(_format_number(val, precision))
            else:
                row_values.append(_format_number(null_value, precision))
        lines.append(delimiter.join(row_values))
    return lines


def _format_number(value: float, precision: str = ".8g") -> str:
    """Format a numeric value with configurable precision.

    Handles whole numbers as integers to avoid unnecessary decimal noise.
    """
    if np.isinf(value):
        return format(float(value), precision)
    if value == int(value):
        return format(int(value), precision)
    return format(float(value), precision)
