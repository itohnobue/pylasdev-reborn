"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import LASWriteError
from .models import LASFile


def write_las_file(
    file_path: str | Path,
    las_data: dict[str, Any] | LASFile,
    encoding: str = "utf-8",
) -> None:
    """Write LAS data to file.

    Args:
        file_path: Output file path.
        las_data: LAS data as dict (legacy format) or LASFile object.
        encoding: Output file encoding (default: utf-8).

    Raises:
        LASWriteError: If file cannot be written.
    """
    file_path = Path(file_path)

    if isinstance(las_data, dict):
        las_file = LASFile.from_dict(las_data)
    else:
        las_file = las_data

    content = _generate_las_content(las_file)

    try:
        file_path.write_text(content, encoding=encoding)
    except OSError as e:
        raise LASWriteError(f"Cannot write to {file_path}: {e}") from e


def _generate_las_content(las_file: LASFile) -> str:
    """Generate LAS file content string with metadata preservation.

    Supports LAS 3.0 features including:
    - DLM field in version section
    - Array notation in curves
    - Format specifiers ({F}, {E}, {S}, {A:x})
    - Zone associations in parameters
    """
    lines: list[str] = []
    is_las30 = las_file.is_las30

    # ~V Version section
    lines.append("~VERSION INFORMATION")
    vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0" if is_las30 else "CWLS LOG ASCII STANDARD"
    lines.append(f" VERS.   {las_file.version.vers}  : {vers_desc}")
    # Always write WRAP=NO since we write one line per depth step (non-wrapped)
    # Even if the original file had WRAP=YES, our output is always non-wrapped
    lines.append(" WRAP.   NO  : ONE LINE PER DEPTH STEP")

    # LAS 3.0: Add DLM field
    if is_las30:
        dlm_desc = "DELIMITING CHARACTER BETWEEN DATA COLUMNS"
        lines.append(f" DLM .                        {las_file.version.dlm} : {dlm_desc}")
    lines.append("")

    # ~W Well section
    lines.append("~WELL INFORMATION")
    for key, value in las_file.well.entries.items():
        lines.append(f" {key}.   {value}  :")
    lines.append("")

    # ~C Curve section — preserve units and descriptions
    lines.append("~CURVE INFORMATION")
    for curve in las_file.curves:
        unit = curve.unit if curve.unit else ""
        desc = curve.description if curve.description else ""

        # LAS 3.0: Add format specifier
        if is_las30 and curve.data_format:
            format_str = f"{{{curve.data_format}"
            if curve.array_info and curve.array_info.time_offset is not None:
                format_str += f":{curve.array_info.time_offset}"
            format_str += "}"
            desc = f"{desc}  {format_str}"

        api = f"  {curve.api_code}" if curve.api_code else ""
        lines.append(f" {curve.mnemonic}.{unit}{api}  : {desc}")
    lines.append("")

    # ~P Parameter section
    if las_file.parameters:
        lines.append("~PARAMETER INFORMATION")
        for param in las_file.parameters:
            unit = param.unit if param.unit else ""
            desc = param.description if param.description else ""

            # LAS 3.0: Add zone association
            if is_las30 and param.zone:
                zone_str = f" | {param.zone.zone_name}"
                if param.zone.zone_index is not None:
                    zone_str += f"[{param.zone.zone_index}]"
                desc = f"{desc}{zone_str}"

            lines.append(f" {param.mnemonic}.{unit}  {param.value}  : {desc}")
        lines.append("")

    # ~O Other section
    if las_file.other and las_file.other.strip():
        lines.append("~OTHER")
        lines.append(las_file.other.rstrip())
        lines.append("")

    # ~A ASCII data section(s)
    if las_file.data_sections:
        # LAS 3.0: Multiple data sections
        for section in las_file.data_sections:
            section_name = f" {section.name}" if section.name else ""
            lines.append(f"~A{section_name}")

            curve_names = section.curves_order
            if curve_names and curve_names[0] in section.data:
                num_rows = len(section.data[curve_names[0]])
                null_value = float(las_file.well.get("NULL", "-999.25"))
                delimiter = las_file.version.delimiter_char

                for i in range(num_rows):
                    row_values: list[str] = []
                    for name in curve_names:
                        if name in las_file.string_data and i < len(las_file.string_data[name]):
                            row_values.append(str(las_file.string_data[name][i]))
                        elif name in section.data and i < len(section.data[name]):
                            row_values.append(f"{section.data[name][i]:.4f}")
                        else:
                            row_values.append(f"{null_value:.4f}")
                    lines.append(delimiter.join(row_values))
    else:
        # Legacy single data section
        curve_names = las_file.curves_order
        if curve_names and curve_names[0] in las_file.logs:
            lines.append("~A  " + "  ".join(curve_names))
            num_rows = len(las_file.logs[curve_names[0]])
            null_value = float(las_file.well.get("NULL", "-999.25"))
            delimiter = las_file.version.delimiter_char

            for i in range(num_rows):
                vals: list[str] = []
                for name in curve_names:
                    if name in las_file.string_data and i < len(las_file.string_data[name]):
                        vals.append(str(las_file.string_data[name][i]))
                    elif name in las_file.logs and i < len(las_file.logs[name]):
                        vals.append(f"{las_file.logs[name][i]:.4f}")
                    else:
                        vals.append(f"{null_value:.4f}")
                lines.append(delimiter.join(vals))

    return "\n".join(lines) + "\n"
