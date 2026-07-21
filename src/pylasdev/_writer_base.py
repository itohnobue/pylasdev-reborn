"""Base writer infrastructure — constants, utilities, and shared writer class.

Contains all version-independent code shared by version-specific writer modules:
- Module-level constants and compiled regexes
- Module-level utility functions (sanitization, formatting)
- ``_WriterMutationGuard`` context manager
- ``_WriterBase`` abstract base class with template method
- ``write_las_file`` public API with version dispatch
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._version_spec import _LASVersionSpec
from .data_reader import _get_null_value
from .exceptions import LASDataError, LASWriteError, PylasdevError
from .models import CurveDefinition, LASFile, ParameterEntry

# ── Module-level constants & compiled regexes ────────────────────────────

# Control characters except space and tab (which are valid LAS whitespace).
# Tab (\x09) is handled separately in _sanitize_las_value — it is replaced
# with a space to prevent mis-tokenization on re-read.  A tab inside an
# identifier acts as a field separator for str.split(), corrupting the
# parsed structure.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029"
    r"\u00A0\u2000-\u200A\u202F\u205F\u3000]"
)

# Previous pattern ^~([A-Za-z]) only matched a leading tilde.
# Values like "\t~Version" or "  ~Curve" bypassed section-header
# sanitization because leading whitespace prevented the regex match.
_LEADING_SECTION_RE = re.compile(r"^\s*~([A-Za-z])")

# The 256-character line-length limit applies to:
#   - LAS 1.2 (all modes) per the LAS 1.2 specification.
#   - LAS 2.0 WRAP=NO per the CWLS specification.
MAX_LINE_LENGTH_LAS12: int = 256

# Pattern matching whitespace-before-colon (\s+:).
_COLON_PRECEDED_BY_WS_RE = re.compile(r"(\s+):")

# Pattern matching colon-followed-by-whitespace-or-end (:\s|\s*$).
_COLON_FOLLOWED_BY_WS_OR_END_RE = re.compile(r":(?=\s|$)")

# Map DataSection.section_type values to the LAS 3.0 section header prefix.
_SECTION_TYPE_TO_PREFIX: dict[str, str] = {
    "LOG_DATA": "A",
    "CORE_DATA": "CORE_DATA",
    "DRILLING_DATA": "DRILLING_DATA",
    "INCLINOMETRY_DATA": "INCLINOMETRY_DATA",
    "TOPS_DATA": "TOPS_DATA",
    "TEST_DATA": "TEST_DATA",
    "PERFORATIONS_DATA": "PERFORATIONS_DATA",
}

# Map DataSection.section_type values to the Definition section header prefix.
_SECTION_TYPE_TO_DEFINITION_PREFIX: dict[str, str] = {
    "CORE_DATA": "Core",
    "DRILLING_DATA": "Drilling",
    "INCLINOMETRY_DATA": "Inclinometry",
    "TOPS_DATA": "Tops",
    "TEST_DATA": "Test",
    "PERFORATIONS_DATA": "Perforations",
}


# ── Module-level utility functions ──────────────────────────────────────

def _sanitize_las_value(value: str) -> str:
    """Sanitize a string for safe inclusion in LAS output."""
    value = (
        value.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
        .replace("\x85", " ")
    )
    value = value.replace("\t", " ")
    value = _CONTROL_CHARS_RE.sub("", value)
    value = _LEADING_SECTION_RE.sub(r"\1", value, count=1)
    if value.startswith("#"):
        value = "_" + value
    elif value and value.lstrip().startswith("#"):
        stripped = value.lstrip()
        leading = value[:len(value) - len(stripped)]
        value = leading + "_" + stripped
    return value


def _escape_colons_for_las_value(value: str) -> str:
    """Escape colons in a LAS value to prevent parser misinterpretation."""
    value = _COLON_PRECEDED_BY_WS_RE.sub(r"\1_:", value)
    value = _COLON_FOLLOWED_BY_WS_OR_END_RE.sub(r"\g<0>_", value)
    return value


def _validate_precision(precision: str) -> None:
    """Validate the precision format specifier for numeric output."""
    if not re.match(r"^\.\d+([eEfFgGn%])?$", precision):
        raise ValueError(
            f"Invalid precision format specifier: '{precision}'. "
            f"Expected a format like '.8g', '.6f', or '.10e'. "
            f"Non-numeric format codes (x, o, b, c, d) are not supported "
            f"for LAS numeric data output."
        )
    if precision[-1] in ("n", "%"):
        import warnings

        warnings.warn(
            f"Precision format code '{precision[-1]}' in '{precision}' "
            f"is not safe for LAS output.  The 'n' format code produces "
            f"locale-dependent output (unparseable comma/grouping). "
            f"The '%' format code multiplies by 100 and appends '%' "
            f"(unparseable suffix).  Consider using 'g', 'f', or 'e' instead.",
            UserWarning,
            stacklevel=2,
        )


def _section_type_to_prefix(section_type: str) -> str:
    """Convert a DataSection.section_type to the LAS header prefix."""
    section_type = section_type.upper()
    known = _SECTION_TYPE_TO_PREFIX.get(section_type)
    if known is not None:
        return known
    if section_type.endswith("_DATA"):
        return _sanitize_las_value(section_type).replace("|", "")
    import warnings

    warnings.warn(
        f"Unknown section type '{section_type}'. "
        f"Falling back to ASCII data section header 'A'. "
        f"Known types: {', '.join(sorted(_SECTION_TYPE_TO_PREFIX.keys()))}. "
        f"Custom types must end with '_DATA'.",
        stacklevel=3,
    )
    return "A"


def _format_curve_line(curve: CurveDefinition, is_las30: bool) -> str:
    """Format a single CurveDefinition as a LAS curve line."""
    unit = _sanitize_las_value(curve.unit) if curve.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = curve.description if curve.description else ""

    if is_las30 and curve.data_format:
        format_str = f"{{{curve.data_format}"
        if curve.data_format == "A" and curve.array_info and curve.array_info.time_offset is not None:
            offset = curve.array_info.time_offset
            if math.isfinite(offset):
                if offset == int(offset):
                    format_str += f":{int(offset)}"
                else:
                    format_str += f":{offset}"
        format_str += "}"
        desc = f"{desc}  {format_str}"

    api_code = _sanitize_las_value(curve.api_code) if curve.api_code else ""
    api_code = _escape_colons_for_las_value(api_code)
    api = f"  {api_code}" if api_code else ""
    desc = _sanitize_las_value(desc)
    desc = _escape_colons_for_las_value(desc)
    return f" {_sanitize_las_value(curve.mnemonic)}.{unit}{api}  : {desc}"


def _format_parameter_line(param: ParameterEntry, is_las30: bool) -> str:
    """Format a single ParameterEntry as a LAS parameter line."""
    unit = _sanitize_las_value(param.unit) if param.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = param.description if param.description else ""

    if is_las30 and param.data_format:
        desc = f"{desc}  {{{param.data_format}}}"

    if is_las30 and param.zone:
        zone_str = f" | {param.zone.zone_name}"
        if param.zone.zone_index is not None:
            zone_str += f"[{param.zone.zone_index}]"
        desc = f"{desc}{zone_str}"

    value = _sanitize_las_value(param.value)
    desc = _sanitize_las_value(desc)
    value = _escape_colons_for_las_value(value)
    desc = _escape_colons_for_las_value(desc)

    return f" {_sanitize_las_value(param.mnemonic)}.{unit}  {value}  : {desc}"


def _format_data_rows(
    curve_names: list[str],
    data: dict[str, NDArray[np.float64]],
    string_data: dict[str, NDArray[np.object_]],
    null_value: float,
    delimiter: str,
    precision: str = ".8g",
    is_las12: bool = False,
) -> list[str]:
    """Format data rows for a section — handles both legacy and LAS 3.0 sections."""
    lines: list[str] = []
    if not curve_names:
        return lines

    curve_arrays: list[tuple[NDArray[np.float64] | NDArray[np.object_] | None, bool]] = []
    for name in curve_names:
        if name in string_data:
            curve_arrays.append((string_data[name], True))
        elif name in data:
            curve_arrays.append((data[name], False))
        else:
            curve_arrays.append((None, False))

    num_rows = max(
        (len(arr) for arr, _ in curve_arrays if arr is not None),
        default=0,
    )
    if num_rows == 0:
        return lines

    warned_long = False
    warned_delim_str = False
    warned_empty_str = False
    for i in range(num_rows):
        row_values: list[str] = []
        for arr, is_string in curve_arrays:
            if arr is None or i >= len(arr):
                row_values.append(_format_number(null_value, precision, null_value))
            elif is_string:
                raw_val = str(arr[i])
                raw_has_delim = delimiter in raw_val
                val = _sanitize_las_value(raw_val)
                if delimiter == " ":
                    if re.search(r"\s", val):
                        if not warned_delim_str:
                            import warnings

                            warnings.warn(
                                "String curve data contains whitespace "
                                "characters while using SPACE delimiter. "
                                "Internal whitespace (including Unicode "
                                "whitespace such as non-breaking spaces) "
                                "will be replaced with underscores to "
                                "prevent data corruption on re-read. "
                                "Consider switching to COMMA or TAB "
                                "delimiter for files with string curves.",
                                stacklevel=4,
                            )
                        warned_delim_str = True
                        val = re.sub(r"\s", "_", val)
                elif raw_has_delim:
                    if not warned_delim_str:
                        import warnings

                        delim_name = "COMMA" if delimiter == "," else "TAB"
                        replacement = ";" if delimiter == "," else " "
                        warnings.warn(
                            f"String curve data contains the active "
                            f"delimiter character ({delim_name}). The "
                            f"delimiter will be replaced with "
                            f"{'semicolons' if delimiter == ',' else 'spaces'} "
                            f"to prevent data corruption on re-read.",
                            stacklevel=4,
                        )
                        warned_delim_str = True
                    replacement = ";" if delimiter == "," else " "
                    val = val.replace(delimiter, replacement)
                if not val and delimiter == " ":
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Empty string curve value replaced by "
                            "'-' sentinel — roundtrip fidelity is "
                            "lost: parser cannot distinguish original "
                            "'-' from originally-empty value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    val = "-"
                row_values.append(val)
            else:
                val = arr[i]
                if np.isnan(val) or np.isinf(val):
                    row_values.append(_format_number(null_value, precision, null_value))
                else:
                    row_values.append(_format_number(val, precision, null_value))
        line = delimiter.join(row_values)
        if is_las12 and len(line) > MAX_LINE_LENGTH_LAS12:
            if not warned_long:
                import warnings

                warnings.warn(
                    f"Data line exceeds 256-character limit "
                    f"(length: {len(line)}).  Lines are NOT truncated "
                    f"to avoid data loss.  Subsequent violations in this "
                    f"section will not be reported.",
                    stacklevel=4,
                )
                warned_long = True
        lines.append(line)
    return lines


def _format_number(value: float, precision: str = ".8g", null_value: float | None = None) -> str:
    """Format a numeric value with configurable precision."""
    if np.isnan(value) or np.isinf(value):
        if null_value is not None:
            return _format_null_sentinel(null_value, precision)
        return format(float(value), precision)
    if null_value is not None and value == null_value:
        return _format_null_sentinel(null_value, precision)
    if value == int(value):
        result = format(int(value), precision)
    else:
        result = format(float(value), precision)
    if "e" in result.lower():
        result = _format_fixed_precision(value, precision)
    return result


def _format_null_sentinel(null_value: float, user_precision: str) -> str:
    """Format a null-value sentinel preserving its exact float identity."""
    result = repr(null_value)
    if "e" in result.lower():
        result = _format_fixed_precision(null_value, user_precision)
    return result


def _format_fixed_precision(value: float, precision: str) -> str:
    """Convert a value to fixed-point notation with magnitude-aware precision."""
    m = re.match(r"\.(\d+)", precision)
    sig_digits = min(int(m.group(1)), 100) if m else 8

    if value == 0:
        return format(value, f".{sig_digits}f")

    magnitude = math.floor(math.log10(abs(value)))
    decimal_places = sig_digits + max(0, (-magnitude) - 1)
    decimal_places = min(decimal_places, 100)
    decimal_places = max(decimal_places, sig_digits)

    result = format(value, f".{decimal_places}f")
    return result


# ── _WriterMutationGuard ────────────────────────────────────────────────

class _WriterMutationGuard:
    """Context manager that runs deferred validation after write."""

    def __init__(self, las_file: LASFile) -> None:
        self._las_file = las_file
        self._saved_wrap: str = las_file.version.wrap
        self._saved_dlm: str = las_file.version.dlm
        self._saved_logs = dict(las_file.logs)
        self._saved_string_data = dict(las_file.string_data)
        self._saved_curves_order = list(las_file.curves_order) if las_file.curves_order is not None else []
        self._saved_curves = list(las_file.curves) if las_file.curves is not None else []

    def __enter__(self) -> _WriterMutationGuard:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        import warnings

        try:
            issues = self._las_file.validate(complete=True)
            for msg in issues:
                warnings.warn(msg, UserWarning, stacklevel=2)
        except Exception:
            pass

        return False  # type: ignore[return-value]


# ── _WriterBase ─────────────────────────────────────────────────────────

class _WriterBase:
    """Abstract base class for version-specific LAS writers.

    Provides the template method ``write()`` that calls section writers in
    ordered sequence, plus shared utility methods and default implementations
    for sections common to some versions.

    Subclasses override version-specific section writers.
    """

    def __init__(self, las_file: LASFile, precision: str) -> None:
        self._las_file = las_file
        self._precision = precision
        self._spec = _LASVersionSpec(las_file.version.vers)

    def write(self) -> str:
        """Generate complete LAS content string (template method)."""
        import warnings

        for issue in self._las_file.validate(complete=True):
            warnings.warn(issue, stacklevel=2)
        lines: list[str] = []
        lines.extend(self._write_version_section())
        lines.extend(self._write_well_section())
        lines.extend(self._write_curve_section())
        lines.extend(self._write_parameter_section())
        lines.extend(self._write_other_section())
        lines.extend(self._write_ascii_sections())
        return "\n".join(lines) + "\n"

    # ── Section writers (overridable) ───────────────────────────────

    def _write_version_section(self) -> list[str]:
        raise NotImplementedError

    def _write_well_section(self) -> list[str]:
        """Write ~W Well section — LAS 2.0/3.0 format (VALUE before colon, desc after)."""
        lines: list[str] = []
        lines.append("~WELL INFORMATION")

        for key in self._las_file.well.entries:
            if not isinstance(key, str):
                raise TypeError(
                    f"WellSection entry key must be str, got {type(key).__name__}: {key!r}"
                )

        mandatory_order = ["STRT", "STOP", "STEP", "NULL"]
        ordered_keys: list[str] = []
        for mandatory in mandatory_order:
            for key in self._las_file.well.entries:
                if key.upper() == mandatory and key not in ordered_keys:
                    ordered_keys.append(key)
                    break
        for key in self._las_file.well.entries:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            value = self._las_file.well.entries[key]
            unit = _sanitize_las_value(self._las_file.well.units.get(key, ""))
            unit_dot = f".{unit}" if unit else "."
            val = _sanitize_las_value(value)
            desc = _sanitize_las_value(self._las_file.well.descriptions.get(key, ""))
            val = _escape_colons_for_las_value(val)
            desc = _escape_colons_for_las_value(desc)
            desc_str = f"  {desc}" if desc else ""
            # LAS 2.0+: MNEM.UNIT VALUE  :
            lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {val}  :{desc_str}")
        lines.append("")
        return lines

    def _write_curve_section(self) -> list[str]:
        """Write ~C Curve section — LAS 1.2/2.0 simple loop."""
        lines: list[str] = []
        lines.append("~CURVE INFORMATION")

        for curve in self._las_file.curves:
            lines.append(_format_curve_line(curve, self._spec.is_las30))

        lines.append("")
        return lines

    def _write_parameter_section(self) -> list[str]:
        """Write ~P Parameter section — LAS 1.2/2.0 flat format."""
        if not self._las_file.parameters:
            return []

        lines: list[str] = []
        lines.append("~PARAMETER INFORMATION")
        for param in self._las_file.parameters:
            lines.append(_format_parameter_line(param, self._spec.is_las30))
        lines.append("")
        return lines

    def _write_other_section(self) -> list[str]:
        """Write ~O Other section — LAS 1.2/2.0 (emit)."""
        lines: list[str] = []
        if not self._las_file.other or not self._las_file.other.strip():
            return lines
        lines.append("~OTHER")
        for line in self._las_file.other.splitlines():
            sanitized = _sanitize_las_value(line)
            if sanitized.strip():
                lines.append(sanitized)
        lines.append("")
        return lines

    def _write_ascii_sections(self) -> list[str]:
        raise NotImplementedError

    # ── Shared helpers for version-specific writers ──────────────────

    def _write_ascii_legacy(self, delimiter: str, check_line_limit: bool) -> list[str]:
        """Legacy ~A data path for LAS 1.2/2.0.

        Handles data_sections copy-back (Path A) and legacy single ~A
        section output (Path C) from the original ``_write_ascii_sections``.
        """
        lines: list[str] = []
        import warnings

        # Path A: non-LAS-3.0 data_sections copy-back
        if self._las_file.data_sections:
            if len(self._las_file.data_sections) > 1:
                raise LASWriteError(
                    f"Multiple data_sections ({len(self._las_file.data_sections)}) "
                    f"are only supported for LAS 3.0 files, but version is "
                    f"{self._las_file.version.vers!r}. Cannot safely write multi-section "
                    f"data for non-LAS-3.0 format."
                )
            warnings.warn(
                "data_sections are only supported for LAS 3.0 files. "
                "Falling back to single-section ~A format. "
                "Single-section data will be preserved.",
                stacklevel=3,
            )

            _ds = self._las_file.data_sections[0]
            if not self._las_file.logs and _ds.data:
                self._las_file.logs.update(_ds.data)
            if not self._las_file.string_data and _ds.string_data:
                self._las_file.string_data.update(_ds.string_data)
            if not self._las_file.curves_order and _ds.curves_order:
                self._las_file.curves_order = list(_ds.curves_order)
            if not self._las_file.curves and _ds.section_curves:
                self._las_file.curves = list(_ds.section_curves)

            if (self._las_file.curves_order and _ds.curves_order
                    and self._las_file.logs):
                existing = set(self._las_file.curves_order)
                for k in _ds.curves_order:
                    if k in self._las_file.logs and k not in existing:
                        self._las_file.curves_order.append(k)

            if (
                self._las_file.curves
                and self._las_file.curves_order
                and len(self._las_file.curves) != len(self._las_file.curves_order)
            ):
                raise LASDataError(
                    f"curves count ({len(self._las_file.curves)}) does not match "
                    f"curves_order count ({len(self._las_file.curves_order)}) "
                    f"after copy-back. This indicates inconsistent "
                    f"LASFile construction."
                )

            if self._las_file.curves_order and (self._las_file.logs or self._las_file.string_data):
                _log_keys = set(self._las_file.logs.keys()) if self._las_file.logs else set()
                _str_keys = (
                    set(self._las_file.string_data.keys()) if self._las_file.string_data
                    else set()
                )
                _order_set = set(self._las_file.curves_order)
                _uncovered = _order_set - _log_keys - _str_keys
                if _uncovered:
                    warnings.warn(
                        f"Curve(s) {sorted(_uncovered)} appear in "
                        f"curves_order but have no data in 'logs' or "
                        f"'string_data' after copy-back.  The writer will "
                        f"pad these curves with null_value.",
                        stacklevel=3,
                    )

        # Path C: Legacy single data section (~A)
        curve_names = self._las_file.curves_order
        if curve_names and not any(
            name in self._las_file.logs or name in self._las_file.string_data for name in curve_names
        ):
            warnings.warn(
                f"curves_order contains {len(curve_names)} curve(s) "
                f"but none have data in logs or string_data. "
                f"No data will be emitted.",
                stacklevel=3,
            )
        if any(name in self._las_file.logs or name in self._las_file.string_data for name in curve_names):
            lines.append("~A  " + "  ".join(_sanitize_las_value(name) for name in curve_names))
            lines.extend(
                _format_data_rows(
                    curve_names,
                    self._las_file.logs,
                    self._las_file.string_data,
                    _get_null_value(self._las_file.well),
                    delimiter,
                    self._precision,
                    is_las12=check_line_limit,
                )
            )

        return lines


# ── Public API: write_las_file ──────────────────────────────────────────

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
        encoding: Output file encoding (default: utf-8).
        precision: Format specifier for numeric data values (default: '.8g').

    Raises:
        LASWriteError: If file cannot be written.
    """
    try:
        _validate_precision(precision)
    except ValueError as e:
        raise LASWriteError(f"Invalid precision format: {e}") from e

    if precision and precision[-1] in ("n", "%"):
        raise LASWriteError(
            f"Precision format code '{precision[-1]}' in '{precision}' "
            f"is not supported for LAS output.  The 'n' format code "
            f"produces locale-dependent decimal separators and grouping. "
            f"The '%' format code multiplies values by 100 and appends "
            f"'%'.  Both corrupt numeric data on re-read.  Use 'g', 'f', "
            f"or 'e' instead."
        )

    file_path = Path(file_path)

    if isinstance(las_data, dict):
        try:
            las_file = LASFile.from_dict(las_data)
        except (ValueError, TypeError, AttributeError, PylasdevError) as e:
            raise LASWriteError(f"Cannot create LASFile from dict: {e}") from e
    elif isinstance(las_data, LASFile):
        las_file = las_data
    else:
        raise LASWriteError(
            f"write_las_file expects a dict or LASFile, got {type(las_data).__name__}"
        )

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LASWriteError(
            f"Cannot create output directory {file_path.parent}: {e}"
        ) from e

    # Version dispatch: choose the correct writer class.
    spec = _LASVersionSpec(las_file.version.vers)
    if spec.is_las12:
        from ._writer_las12 import _Las12Writer
        writer: _WriterBase = _Las12Writer(las_file, precision)
    elif spec.is_las20:
        from ._writer_las20 import _Las20Writer
        writer = _Las20Writer(las_file, precision)
    else:
        from ._writer_las30 import _Las30Writer
        writer = _Las30Writer(las_file, precision)

    with _WriterMutationGuard(las_file):
        try:
            content = writer.write()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, PylasdevError) as e:
            raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

        try:
            target_dir = str(file_path.parent)
            fd, tmp_path = tempfile.mkstemp(
                dir=target_dir, prefix=".tmp_", suffix=file_path.name
            )
            try:
                with os.fdopen(fd, 'w', encoding=encoding, newline='') as f:
                    f.write(content)
                os.replace(tmp_path, file_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, UnicodeError, LookupError) as e:
            raise LASWriteError(f"Cannot write to {file_path}: {e}") from e
