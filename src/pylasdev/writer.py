"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.

Supports LAS 1.2, 2.0, and 3.0 formats.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data_reader import _get_null_value
from .exceptions import LASWriteError, PylasdevError
from .models import CurveDefinition, LASFile, ParameterEntry

# Control characters except space and tab (which are valid LAS whitespace).
# Tab (\x09) is handled separately in _sanitize_las_value — it is replaced
# with a space to prevent mis-tokenization on re-read.  A tab inside an
# identifier acts as a field separator for str.split(), corrupting the
# parsed structure.
# Matches \x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F (DEL), \x85 (NEL),
# \u2028 (LINE SEPARATOR), and \u2029 (PARAGRAPH SEPARATOR).
# The Unicode line break characters are treated as line breaks by Python's
# splitlines() but are not caught by \n/\r replacement.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85\u2028\u2029]")

# F-86: Previous pattern ^~([A-Za-z]) only matched a leading tilde.
# Values like "\t~Version" or "  ~Curve" bypassed section-header
# sanitization because leading whitespace prevented the regex match.
# Extended to ^\s*~([A-Za-z]) — the \s* consumes leading whitespace
# and the replacement \1 strips both the whitespace and tilde.
_LEADING_SECTION_RE = re.compile(r"^\s*~([A-Za-z])")

# LAS 1.2 spec mandates a maximum line length of 256 characters.
# LAS 2.0+ has no line-length limit.  We warn when LAS 1.2 data rows
# exceed this limit but do NOT truncate (truncation would cause data loss).
MAX_LINE_LENGTH_LAS12: int = 256


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
    # F2-005: Replace tab characters with spaces.  While tab is valid
    # LAS whitespace between fields, a tab inside an identifier or
    # value causes mis-tokenization on re-read (str.split() treats
    # tab as a field separator, corrupting the parsed structure).
    value = value.replace("\t", " ")
    # Strip control characters
    value = _CONTROL_CHARS_RE.sub("", value)
    # If the value now starts with ~ followed by a letter, remove the leading ~
    # to prevent it from being interpreted as a section header
    value = _LEADING_SECTION_RE.sub(r"\1", value, count=1)
    # F-87: The parser skips #-prefixed lines as comments in data sections.
    # A value starting with # would be silently dropped on re-read, creating
    # data loss.  Prefix with _ to preserve it while preventing comment
    # injection — the parser treats _# as a normal value character.
    if value.startswith("#"):
        value = "_" + value
    # F-I2-M12: After newline-to-space conversion above, the "#" character
    # may be preceded by whitespace (e.g., "\n#comment" → " #comment"),
    # bypassing the startswith("#") guard.  For COMMA/TAB-delimited output
    # with single-curve data, this enables comment injection causing data
    # loss on re-read.  Check for "#" after any leading whitespace and
    # prefix with "_" to prevent this (SPACE delimiter is mitigated by
    # the whitespace-to-underscore conversion in _format_data_rows).
    elif value and value.lstrip().startswith("#"):
        stripped = value.lstrip()
        leading = value[:len(value) - len(stripped)]
        value = leading + "_" + stripped
    return value


# Pattern matching whitespace-before-colon (\s+:).
# Used in _escape_colons_for_las_value to insert an underscore between
# the whitespace and colon, preventing the parser regex alternative
# \s+:\s* from matching at embedded colon positions.
_COLON_PRECEDED_BY_WS_RE = re.compile(r"(\s+):")

# Pattern matching colon-followed-by-whitespace-or-end (:\s|\s*$).
# Used in _escape_colons_for_las_value to insert an underscore after
# the colon, preventing the parser regex alternatives \s*:\s+ and
# :\s*$ from matching at embedded colon positions.
_COLON_FOLLOWED_BY_WS_OR_END_RE = re.compile(r":(?=\s|$)")


def _escape_colons_for_las_value(value: str) -> str:
    """Escape colons in a LAS value to prevent parser misinterpretation.

    The parser's DATA_LINE_PATTERN uses colon as the structural separator
    between value and description fields.  The colon-separator regex::

        (\\s+:\\s*|\\s*:\\s+|:\\s*$)

    has three alternatives, all requiring whitespace on at least one side
    of the colon (or end-of-string).  This helper inserts ``_`` to break
    every whitespace-colon adjacency, ensuring no embedded colon can be
    mistaken for the structural separator:

    * ``" :"`` (space-colon)  → ``" _:"``  (``_`` between ws and colon)
    * ``": "`` (colon-space)  → ``":_ "``  (``_`` between colon and ws)
    * trailing ``":"``        → ``":_"``   (``_`` after colon at EOS)

    When both sides have whitespace (``" : "``) both fixes apply,
    producing ``" _:_ "`` which is safe against all three alternatives.

    This helper is intentionally NOT part of ``_sanitize_las_value``
    because that function also handles description text where ``": "``
    is a legitimate LAS spec formatting convention
    (e.g., ``"LAS 2.0 : CWLS LOG ASCII STANDARD"``).

    .. important::

        The underscore (``_``) characters inserted by this function are
        **permanent** — they are NOT reversed during parsing.  Values
        containing colons adjacent to whitespace (e.g., ``"Oil : Gas"``
        → ``"Oil _:_ Gas"``) will NOT roundtrip to their original form.

        This is a deliberate trade-off.  Without escaping, embedded colons
        adjacent to whitespace would be misinterpreted as structural
        separators, causing data truncation on re-read.  The underscore
        artifacts are the lesser evil compared to data loss.

        If roundtrip fidelity for colon-containing values is required,
        consider adding an unescape step in the parser that reverses
        this transformation (``_:_`` → `` : `` etc.).
    """
    # Step 1: Insert _ between whitespace and colon.
    # Prevents \\s+:\\s* from matching at embedded colons.
    value = _COLON_PRECEDED_BY_WS_RE.sub(r"\1_:", value)

    # Step 2: Insert _ after colon when followed by whitespace or end.
    # Prevents \\s*:\\s+ and :\\s*$ from matching.
    value = _COLON_FOLLOWED_BY_WS_OR_END_RE.sub(r"\g<0>_", value)

    return value


def _validate_precision(precision: str) -> None:
    """Validate the precision format specifier for numeric output.

    Rejects format codes that produce non-decimal output (x, o, b, c, d)
    which would silently corrupt LAS data with hex/octal/binary/character
    representations, or crash when applied to floating-point values.

    Accepted formats: '.N', '.Ng', '.Nf', '.Ne', '.NE', '.NF', '.NG',
    '.Nn', '.N%' where N is one or more digits.

    Raises:
        ValueError: If the precision string is not a valid float-compatible
            format specifier.
    """
    # Must start with '.' followed by digits, optionally ending with a
    # float-compatible presentation type (e, E, f, F, g, G).
    # Integer-specific codes (b, c, d, o, x, X) are rejected.
    # The 'n' and '%' format codes are accepted by the regex for backward
    # compatibility (existing tests call _validate_precision with these)
    # but produce a warning at write time — see _check_precision_for_write.
    # No type code at all defaults to g-type for floats (safe).
    if not re.match(r"^\.\d+([eEfFgGn%])?$", precision):
        raise ValueError(
            f"Invalid precision format specifier: '{precision}'. "
            f"Expected a format like '.8g', '.6f', or '.10e'. "
            f"Non-numeric format codes (x, o, b, c, d) are not supported "
            f"for LAS numeric data output."
        )
    # F-IF016/F-IF017: Warn about 'n' and '%' format codes.  'n' is
    # locale-dependent (produces comma decimals and grouping characters
    # that are unparseable in LAS format).  '%' multiplies values by 100
    # and appends '%' (e.g., format(-999.25, ".8%") → "-99925.00000000%"),
    # producing values that cannot be re-parsed as floating-point numbers.
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
            Pass a Python format spec like '.6g' or '.8f' for more precision.
            e-format specifiers (e.g. '.10e') produce exponent notation
            forbidden by the LAS spec and are automatically converted to
            fixed-point format.

    Raises:
        LASWriteError: If file cannot be written.
    """
    # F2-004: Validate the precision format spec before any processing.
    # Non-numeric format codes (x, o, b, c, d) produce hex/octal/binary/
    # character output or crash when applied to floating-point values.
    try:
        _validate_precision(precision)
    except ValueError as e:
        raise LASWriteError(f"Invalid precision format: {e}") from e

    # F-IF016/F-IF017: Reject 'n' and '%' format codes at write time.
    # The _validate_precision function warns about these but accepts them
    # for backward compatibility; the actual write path must reject them
    # to prevent silent data corruption.
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
        # F34: Wrap from_dict in try/except so that malformed input
        # (e.g., non-numeric log values that fail np.array(dtype=np.float64)
        # in models.py) raises LASWriteError instead of raw ValueError.
        # F-I2-M04: Also catch PylasdevError — from_dict may raise
        # LASDataError (a PylasdevError subclass) for data validation
        # failures instead of raw ValueError/TypeError.
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

    # Always write with the specified encoding (default: utf-8).
    # F-I2-M04: Also catch PylasdevError — content generation may raise
    # PylasdevError subclasses from models.py validation.
    try:
        content = _generate_las_content(las_file, precision)
    except (ValueError, TypeError, KeyError, AttributeError, PylasdevError) as e:
        raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

    try:
        file_path.write_text(content, encoding=encoding)
    except (OSError, UnicodeError, LookupError) as e:
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
    # F-014: Detect LAS 1.2 to guard DLM emission (DLM is LAS 2.0/3.0 only).
    is_las12 = las_file.version.vers.startswith("1.")
    lines.append("~VERSION INFORMATION")
    vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0" if is_las30 else "CWLS LOG ASCII STANDARD"
    # Guard against empty vers field (analogous to wrap guard below).
    # An empty vers produces a malformed version line with an empty value.
    vers = las_file.version.vers or "2.0"
    lines.append(f" VERS.   {_sanitize_las_value(vers)}  : {vers_desc}")
    # F-05 / F-01: The writer cannot produce wrapped output (we always write
    # one line per depth step).  If the source has WRAP=YES, override it to
    # NO so the header declaration matches the actual data layout.  Emit a
    # warning to inform the user of the override.
    actual_wrap = las_file.version.wrap.upper() if las_file.version.wrap else "NO"
    if actual_wrap == "YES":
        import warnings

        warnings.warn(
            "WRAP=YES overridden to WRAP=NO because the writer "
            "always produces ONE LINE PER DEPTH STEP (non-wrapped) output. "
            "The data WILL be non-wrapped regardless of the original declaration.",
            stacklevel=3,
        )
        actual_wrap = "NO"
    wrap_desc = (
        "ONE LINE PER DEPTH STEP" if actual_wrap == "NO" else "MULTIPLE LINES PER DEPTH STEP"
    )
    lines.append(f" WRAP.   {_sanitize_las_value(actual_wrap)}  : {wrap_desc}")
    # DLM is defined in LAS 2.0 and 3.0 specs.  LAS 1.2 does not use
    # DLM and always defaults to SPACE.  When DLM is not SPACE (e.g.,
    # COMMA or TAB), emit the DLM line so the file declares the correct
    # delimiter — otherwise a re-read would default to SPACE, corrupting
    # comma- or tab-delimited data.
    if las_file.version.dlm.upper() != "SPACE" and not is_las12:
        dlm_desc = "DELIMITING CHARACTER BETWEEN DATA COLUMNS"
        lines.append(
            f" DLM .                        {_sanitize_las_value(las_file.version.dlm)} : {dlm_desc}"
        )
    lines.append("")
    return lines


def _write_well_section(las_file: LASFile) -> list[str]:
    """Write ~W Well section.

    LAS 1.2 uses format ``MNEM.UNIT    :VALUE`` (value after colon,
    no description before colon).  LAS 2.0+ uses ``MNEM.UNIT VALUE  :``
    (value before colon, description after).
    """
    lines: list[str] = []
    is_las12 = las_file.version.vers.startswith("1.")
    lines.append("~WELL INFORMATION")

    # Validate mandatory well fields — emit warnings for missing fields
    # so that programmatically-constructed LAS files don't silently produce
    # semantically non-compliant output.  No exception raised; consistent
    # with the parser's read-time validation pattern.
    mandatory_fields = {"STRT", "STOP", "STEP", "NULL"}
    present_fields = {k.upper() for k in las_file.well.entries}
    for field in mandatory_fields:
        if field not in present_fields:
            import warnings

            warnings.warn(
                f"Missing mandatory well field: {field}. "
                "The output file may be semantically non-compliant.",
                stacklevel=4,
            )

    # F-26: Reorder mandatory well fields (STRT, STOP, STEP, NULL)
    # to appear first per CWLS spec, followed by remaining fields in
    # original dict insertion order.  Do not drop any fields.
    mandatory_order = ["STRT", "STOP", "STEP", "NULL"]
    ordered_keys: list[str] = []
    # First pass: collect mandatory fields that are present, in spec order.
    for mandatory in mandatory_order:
        for key in las_file.well.entries:
            if key.upper() == mandatory and key not in ordered_keys:
                ordered_keys.append(key)
                break
    # Second pass: remaining fields in original insertion order.
    for key in las_file.well.entries:
        if key not in ordered_keys:
            ordered_keys.append(key)

    for key in ordered_keys:
        value = las_file.well.entries[key]
        unit = _sanitize_las_value(las_file.well.units.get(key, ""))
        unit_dot = f".{unit}" if unit else "."
        val = _sanitize_las_value(value)
        # Emit CWLS well descriptions if present (F-D3-H01 fix).
        desc = _sanitize_las_value(las_file.well.descriptions.get(key, ""))
        # F-H04: Escape colons in well values/descriptions after general
        # sanitization.  Embedded colons with adjacent whitespace (": ")
        # or trailing colons get confused with the structural colon
        # separator in the parser's DATA_LINE_PATTERN regex.
        val = _escape_colons_for_las_value(val)
        desc = _escape_colons_for_las_value(desc)
        desc_str = f"  {desc}" if desc else ""
        if is_las12:
            # F-03: LAS 1.2 CWLS spec places numeric well fields (STRT,
            # STOP, STEP, NULL) BEFORE the colon.  Non-numeric fields
            # keep the lasio convention (value AFTER colon) for backward
            # compatibility with files that use that convention.
            # F-ITER2-D3-M06: case-insensitive check for well field names
            # because from_dict() stores keys as-is without uppercasing.
            if key.upper() in {"STRT", "STOP", "STEP", "NULL"}:
                lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {val}  :{desc_str}")
            else:
                # F-03: Non-mandatory LAS 1.2 well fields use CWLS convention:
                # DESCRIPTION before colon, VALUE after colon.
                # The parser's _store_well_entry (parser.py:934-942) expects
                #   MNEM.UNIT  DESC : VALUE
                # for non-mandatory fields across all well_format modes
                # (cwls, lasio, auto).  Placing value before colon with
                # description appended after (the previous behaviour) causes
                # roundtrip data corruption — description is lost and value
                # accumulates description text on re-read.
                if desc:
                    lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {desc}  : {val}")
                else:
                    lines.append(f" {_sanitize_las_value(key)}{unit_dot}    : {val}")
        else:
            # LAS 2.0+: MNEM.UNIT VALUE  :
            lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {val}  :{desc_str}")
    lines.append("")
    return lines


def _write_curve_section(las_file: LASFile) -> list[str]:
    """Write ~C Curve section — preserve units and descriptions.

    For LAS 3.0 files with structured data sections, only the main
    (LOG_DATA) curves are emitted in ~CURVE.  Per-section curves are
    emitted in their own Definition sections by _write_ascii_sections.
    """
    lines: list[str] = []
    is_las30 = las_file.is_las30
    lines.append("~CURVE INFORMATION")

    if is_las30 and las_file.data_sections:
        # Use the LOG_DATA section's curve definitions for the main ~C block.
        # F2-006: Normalize section_type to uppercase — from_dict does not
        # normalize, so programmatically constructed files may have lowercase.
        log_section = next(
            (ds for ds in las_file.data_sections if ds.section_type.upper() == "LOG_DATA"),
            None,
        )
        curves_to_emit = (
            log_section.section_curves
            if log_section and log_section.section_curves
            else las_file.curves
        )
        for curve in curves_to_emit:
            lines.append(_format_curve_line(curve, is_las30))
    else:
        for curve in las_file.curves:
            lines.append(_format_curve_line(curve, is_las30))

    lines.append("")
    return lines


def _write_parameter_section(las_file: LASFile) -> list[str]:
    """Write ~P Parameter section.

    For LAS 3.0 files with per-section parameters (F-053), groups
    parameters by section_type and emits separate typed parameter
    sections (e.g., ~Core_Parameter, ~Drilling_Parameter).  Parameters
    without a section_type go into the standard ~PARAMETER INFORMATION
    section.  LAS 1.2/2.0 files always emit a single flat section.
    """
    if not las_file.parameters:
        return []

    lines: list[str] = []
    is_las30 = las_file.is_las30

    def _format_one_param(param: ParameterEntry) -> str:
        """Format a single parameter line (common to all section types)."""
        unit = _sanitize_las_value(param.unit) if param.unit else ""
        desc = param.description if param.description else ""

        # F-M05: Emit data_format specifier in LAS 3.0 parameter lines.
        if is_las30 and param.data_format:
            desc = f"{desc}  {{{param.data_format}}}"

        if is_las30 and param.zone:
            zone_str = f" | {param.zone.zone_name}"
            if param.zone.zone_index is not None:
                zone_str += f"[{param.zone.zone_index}]"
            desc = f"{desc}{zone_str}"

        # F-I2-M30: Escape colons in parameter values and descriptions.
        value = _sanitize_las_value(param.value)
        desc = _sanitize_las_value(desc)
        value = _escape_colons_for_las_value(value)
        desc = _escape_colons_for_las_value(desc)

        return f" {_sanitize_las_value(param.mnemonic)}.{unit}  {value}  : {desc}"

    # F-053: For LAS 3.0, group parameters by section_type for per-section
    # parameter roundtrip.  Standard ~P section parameters have section_type=None.
    if is_las30:
        # Partition: parameters with section_type (per-section) vs without (standard).
        # dict preserves insertion order in Python 3.8+.
        sections: dict[str | None, list[ParameterEntry]] = {}
        for param in las_file.parameters:
            sections.setdefault(param.section_type, []).append(param)

        # Emit standard ~PARAMETER INFORMATION first (section_type=None).
        std_params = sections.pop(None, [])
        if std_params:
            lines.append("~PARAMETER INFORMATION")
            for param in std_params:
                lines.append(_format_one_param(param))
            lines.append("")

        # Emit per-section typed parameter sections.
        # Order is preserved from dict (first-seen order).
        # Section header format: ~{SectionType}_Parameter (e.g., ~Core_Parameter).
        for section_type, params in sections.items():
            if not section_type:
                continue
            lines.append(f"~{_sanitize_las_value(section_type)}_Parameter")
            for param in params:
                lines.append(_format_one_param(param))
            lines.append("")

        return lines

    # LAS 1.2 / 2.0: single flat ~PARAMETER INFORMATION section.
    lines.append("~PARAMETER INFORMATION")
    for param in las_file.parameters:
        lines.append(_format_one_param(param))
    lines.append("")
    return lines


def _write_other_section(las_file: LASFile) -> list[str]:
    """Write ~O Other section.

    LAS 3.0 deprecates the ~Other section — content must go into
    user-defined Parameter or Column Data sections instead.
    """
    lines: list[str] = []
    if not las_file.other or not las_file.other.strip():
        return lines
    # F-05: ~Other is NOT ALLOWED in LAS 3.0 per spec.  Skip emission
    # and warn so the caller knows the content was not written.
    if las_file.is_las30:
        import warnings

        warnings.warn(
            "~Other section content was NOT written because LAS 3.0 "
            "deprecates the ~Other section.  Other content should be "
            "migrated to user-defined Parameter or Column Data sections.",
            stacklevel=3,
        )
        return lines
    lines.append("~OTHER")
    # Sanitize each line of free-form other content against section injection
    for line in las_file.other.splitlines():
        sanitized = _sanitize_las_value(line)
        if sanitized.strip():
            lines.append(sanitized)
    lines.append("")
    return lines


# Map DataSection.section_type values to the LAS 3.0 section header prefix.
# The default "LOG_DATA" maps to "A" for backward compatibility.
_SECTION_TYPE_TO_PREFIX: dict[str, str] = {
    "LOG_DATA": "A",
    "CORE_DATA": "CORE_DATA",
    "DRILLING_DATA": "DRILLING_DATA",
    "INCLINOMETRY_DATA": "INCLINOMETRY_DATA",
    "TOPS_DATA": "TOPS_DATA",
    "TEST_DATA": "TEST_DATA",
    "PERFORATIONS_DATA": "PERFORATIONS_DATA",
}

# Map DataSection.section_type values to the Definition section header prefix
# (the root name without the _DATA suffix).  e.g., CORE_DATA → "Core_Definition".
_SECTION_TYPE_TO_DEFINITION_PREFIX: dict[str, str] = {
    "CORE_DATA": "Core",
    "DRILLING_DATA": "Drilling",
    "INCLINOMETRY_DATA": "Inclinometry",
    "TOPS_DATA": "Tops",
    "TEST_DATA": "Test",
    "PERFORATIONS_DATA": "Perforations",
}


def _section_type_to_prefix(section_type: str) -> str:
    """Convert a DataSection.section_type to the LAS header prefix.

    Known types (LOG_DATA → "A", CORE_DATA → "CORE_DATA", etc.) use the
    standard mapped prefix.  User-defined types ending with ``_DATA``
    (e.g., ``"CUSTOM_DATA"`` from a ``~Custom_Data`` section) use the
    type name itself as the prefix, preserving the original section
    identity on roundtrip.
    """
    # F2-006: Normalize to uppercase for case-insensitive matching.
    # from_dict does not normalize section_type, so programmatically
    # constructed LASFile objects may have lowercase section_type values.
    section_type = section_type.upper()
    known = _SECTION_TYPE_TO_PREFIX.get(section_type)
    if known is not None:
        return known
    # User-defined section types following the _DATA convention
    # use their own name as the header prefix for roundtrip fidelity.
    # F-ITER2-SEC-M04: Sanitize the user-provided type name to prevent
    # section-header injection via from_dict() with malicious section_type
    # values containing newlines or control characters.
    if section_type.endswith("_DATA"):
        return _sanitize_las_value(section_type)
    # F-008: Warn about unknown section types — they fall back to the
    # ASCII data section header "A" for backward compatibility, but the
    # caller should be informed that the section type is not recognized.
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
    desc = curve.description if curve.description else ""

    if is_las30 and curve.data_format:
        format_str = f"{{{curve.data_format}"
        if curve.array_info and curve.array_info.time_offset is not None:
            offset = curve.array_info.time_offset
            if offset == int(offset):
                format_str += f":{int(offset)}"
            else:
                format_str += f":{offset}"
        format_str += "}"
        desc = f"{desc}  {format_str}"

    api_code = _sanitize_las_value(curve.api_code) if curve.api_code else ""
    # F-IF024: Escape colons in the API code after general sanitization.
    # The api_code appears BEFORE the structural colon in the output:
    # MNEM.UNIT{api_code}  : DESC.  An embedded colon in api_code
    # (e.g., "42 : injected") creates a spurious structural colon match
    # in the parser's DATA_LINE_PATTERN, causing value/description
    # misrouting on re-read.
    api_code = _escape_colons_for_las_value(api_code)
    api = f"  {api_code}" if api_code else ""
    # F-M10: Escape colons in the curve description after general
    # sanitization.  Embedded colons with adjacent whitespace (e.g.,
    # "Oil : Gas") can be mistaken for the structural colon separator
    # in the parser's DATA_LINE_PATTERN, causing the description to
    # be truncated on re-read (parser reads api_code="Oil", desc="Gas"
    # instead of desc="Oil : Gas").  Well section handles this
    # (_write_well_section lines 345-346); curve section must do the
    # same.
    desc = _sanitize_las_value(desc)
    desc = _escape_colons_for_las_value(desc)
    return f" {_sanitize_las_value(curve.mnemonic)}.{unit}{api}  : {desc}"


def _write_ascii_sections(las_file: LASFile, precision: str = ".8g") -> list[str]:
    """Write data sections — ~A for LAS 1.2/2.0, typed sections for LAS 3.0."""
    lines: list[str] = []
    null_value = _get_null_value(las_file.well)
    delimiter = las_file.version.delimiter_char
    is_las30 = las_file.is_las30
    is_las12 = las_file.version.vers.startswith("1.")

    # F-04: LAS 1.2 only supports SPACE delimiter per spec.  The header
    # correctly suppresses non-SPACE DLM emission for LAS 1.2 (see
    # _write_version_section L172-176) but data rows used the non-SPACE
    # delimiter from delimiter_char, creating a header/data mismatch
    # that would corrupt roundtrip re-reads.  Force SPACE and warn.
    if is_las12 and delimiter != " ":
        import warnings

        warnings.warn(
            f"LAS 1.2 does not support the '{las_file.version.dlm}' delimiter. "
            "Forcing SPACE delimiter for data rows to match the header section.",
            stacklevel=3,
        )
        delimiter = " "

    # F-I2-M17 / F-019: Guard against data_sections in non-LAS-3.0 files.
    # from_dict populates data_sections from the input dict unconditionally
    # (regardless of LAS version).  Multiple sections on a non-LAS-3.0 file
    # cause guaranteed roundtrip data loss (parser skips all data for
    # non-LAS-3.0 settings).  A single section can safely fall back to
    # legacy ~A format.
    if las_file.data_sections and not is_las30:
        if len(las_file.data_sections) > 1:
            raise LASWriteError(
                f"Multiple data_sections ({len(las_file.data_sections)}) "
                f"are only supported for LAS 3.0 files, but version is "
                f"{las_file.version.vers!r}. Cannot safely write multi-section "
                f"data for non-LAS-3.0 format."
            )
        import warnings

        warnings.warn(
            "data_sections are only supported for LAS 3.0 files. "
            "Falling back to single-section ~A format. "
            "Single-section data will be preserved.",
            stacklevel=3,
        )

    if las_file.data_sections and is_las30:
        # LAS 3.0: Multiple data sections with typed headers.
        emitted_defs: set[str] = set()
        for section in las_file.data_sections:
            # F2-006: Normalize section_type to uppercase — from_dict does
            # not normalize, so programmatically constructed LASFile objects
            # may have lowercase section_type values (e.g., "log_data").
            # Normalize once at the top of the loop for all comparisons.
            sec_type = section.section_type.upper()
            section_prefix = _section_type_to_prefix(sec_type)
            section_name = f" {_sanitize_las_value(section.name)}" if section.name else ""

            # F-16: Compute definition prefix for non-LOG_DATA LAS 3.0
            # sections.  Used for both the _Definition section header and
            # the pipe notation on the data section header so the parser
            # can re-associate per-section curves on re-read.
            def_prefix: str | None = None
            if is_las30 and sec_type != "LOG_DATA":
                def_prefix = _SECTION_TYPE_TO_DEFINITION_PREFIX.get(sec_type)
                # F-D3-M01: Auto-derive Definition prefix for user-defined _DATA
                # section types not in the hardcoded mapping.  Strip _DATA suffix
                # and title-case the root (e.g., "CUSTOM_DATA" → "Custom").
                if def_prefix is None and sec_type.endswith("_DATA"):
                    root = sec_type[: -len("_DATA")]
                    root = _sanitize_las_value(root)
                    def_prefix = root.title().replace("_", "")

            # For non-LOG_DATA sections: emit per-section Definition section
            # so that the parser can correctly re-associate per-section curve
            # names on re-read.  Without this, all data sections get the
            # global curve set on roundtrip.  Only emit once per section type
            # (e.g., one Core_Definition for both Core[1] and Core[2]).
            if is_las30 and sec_type != "LOG_DATA" and section.section_curves:
                if def_prefix and def_prefix not in emitted_defs:
                    emitted_defs.add(def_prefix)
                    lines.append(f"~{def_prefix}_Definition")
                    for curve in section.section_curves:
                        lines.append(_format_curve_line(curve, is_las30))
                    lines.append("")  # blank line after definition

            # Data section header with pipe notation for curve association.
            # LOG_DATA sections reference "| CURVE" so the parser scopes
            # curves to only the global ~CURVE set on re-read.  Non-LOG_DATA
            # sections reference "| {Name}_Definition" for per-section curve
            # reassociation (F-16).
            # F-010: Only emit the pipe reference when the corresponding
            # Definition section was actually written above (tracked via
            # emitted_defs).  Without this guard, a section with a valid
            # def_prefix but empty section_curves would emit a phantom
            # pipe reference to a non-existent Definition section.
            if sec_type == "LOG_DATA" and is_las30:
                lines.append(f"~{section_prefix}{section_name} | CURVE")
            elif is_las30 and sec_type != "LOG_DATA" and def_prefix and def_prefix in emitted_defs:
                lines.append(
                    f"~{section_prefix}{section_name} | {def_prefix}_Definition"
                )
            else:
                lines.append(f"~{section_prefix}{section_name}")
            lines.extend(
                _format_data_rows(
                    section.curves_order,
                    section.data,
                    section.string_data,
                    null_value,
                    delimiter,
                    precision,
                    is_las12=is_las12,
                )
            )
    else:
        # Legacy single data section (~A).
        curve_names = las_file.curves_order
        if any(name in las_file.logs or name in las_file.string_data for name in curve_names):
            lines.append("~A  " + "  ".join(_sanitize_las_value(name) for name in curve_names))
            lines.extend(
                _format_data_rows(
                    curve_names,
                    las_file.logs,
                    las_file.string_data,
                    null_value,
                    delimiter,
                    precision,
                    is_las12=is_las12,
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
    is_las12: bool = False,
) -> list[str]:
    """Format data rows for a section — handles both legacy and LAS 3.0 sections.

    Builds one line per depth step with delimiter-separated values.
    String curves are emitted as-is; numeric curves use configurable formatting.
    Missing values are filled with the null_value. NaN values are output as null.

    When *is_las12* is True, warns if any data row exceeds the LAS 1.2
    256-character line limit.  Lines are NOT truncated — truncation would
    cause data loss.
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

    warned_long = False  # Deduplicate long-line warnings per section
    warned_delim_str = False  # Deduplicate delimiter-in-string warnings per section
    for i in range(num_rows):
        row_values: list[str] = []
        for arr, is_string in curve_arrays:
            if arr is None or i >= len(arr):
                row_values.append(_format_number(null_value, precision, null_value))
            elif is_string:
                raw_val = str(arr[i])
                # F2-005: Check the raw value for the delimiter character
                # BEFORE calling _sanitize_las_value.  _sanitize_las_value
                # now replaces tabs with spaces, so the original tab would
                # be masked from the delimiter-aware check below.
                raw_has_delim = delimiter in raw_val
                val = _sanitize_las_value(raw_val)
                # F2-29/F2-30/F2-31 + F-W05: Delimiter-aware string data
                # sanitization.  When the active delimiter character (or
                # any whitespace for SPACE delimiter) appears in a string
                # value, it must be replaced to prevent roundtrip corruption.
                # The reader splits on the delimiter, so embedded delimiter
                # characters create phantom field boundaries on re-read.
                if delimiter == " ":
                    # SPACE delimiter: the reader (str.split()) splits on
                    # any Unicode whitespace (including \xa0, \u2000-\u200a,
                    # \u202f, \u205f, \u3000), not just ASCII space and tab.
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
                    # COMMA or TAB delimiter: the delimiter character itself
                    # in the value creates a phantom field boundary on re-read.
                    # F2-005: raw_has_delim is checked on the raw value
                    # because _sanitize_las_value may have already handled
                    # the delimiter character (e.g., tab → space).  The
                    # replacement is applied to val (which already has the
                    # tab replaced by space, so the str.replace is a no-op
                    # for tab; for comma it works as before).
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
                # F-21: Empty string after sanitization causes column misalignment
                # with SPACE delimiter.  " ".join(["a", "", "b"]) produces "a  b"
                # (two consecutive spaces).  On re-read, str.split() collapses to
                # ["a", "b"] — one column permanently lost.  Replace with "-"
                # (traditional LAS placeholder for missing data).  COMMA/TAB
                # delimiters are unaffected — empty fields survive split(",").
                if not val and delimiter == " ":
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
                    f"Data line exceeds LAS 1.2 256-character limit "
                    f"(length: {len(line)}).  Lines are NOT truncated "
                    f"to avoid data loss.  Subsequent violations in this "
                    f"section will not be reported.",
                    stacklevel=4,
                )
                warned_long = True
        lines.append(line)
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
            return _format_null_sentinel(null_value, precision)
        return format(float(value), precision)
    # F-22: Null sentinel formatted with user-specified g-format precision
    # loses identity (e.g., format(-999.25, ".4g") → "-999.2").  On re-read,
    # -999.2 ≠ -999.25 → treated as real data instead of null.
    # Use repr() which produces the shortest identity-preserving string.
    if null_value is not None and value == null_value:
        return _format_null_sentinel(null_value, precision)
    if value == int(value):
        result = format(int(value), precision)
    else:
        result = format(float(value), precision)
    # The LAS spec forbids exponent notation in data sections.
    # Detect exponent output and reformat using magnitude-aware
    # fixed-point precision that preserves significant digits.
    if "e" in result.lower():
        result = _format_fixed_precision(value, precision)
    return result


def _format_null_sentinel(null_value: float, user_precision: str) -> str:
    """Format a null-value sentinel preserving its exact float identity.

    User-specified g-format precision (e.g., ``".4g"``) can truncate the
    last significant digits of the null sentinel (e.g., -999.25 → "-999.2").
    On re-read, the truncated value no longer matches the NULL declaration
    in the well section and is treated as real data.

    Uses ``repr()`` for an exact decimal representation.  Falls back to
    ``_format_fixed_precision()`` when ``repr()`` produces exponent
    notation (extremely large or small null sentinel values, unlikely
    in practice).
    """
    result = repr(null_value)
    if "e" in result.lower():
        result = _format_fixed_precision(null_value, user_precision)
    return result


def _format_fixed_precision(value: float, precision: str) -> str:
    """Convert a value to fixed-point notation with magnitude-aware precision.

    ``.8f`` would lose significant digits for values < 1e-4 (e.g.,
    ``0.0000123`` → ``"0.00001230"`` with ``.8f`` loses 4 digits).
    This helper computes the number of decimal places needed to preserve
    the significant-digit count implied by the original precision spec.

    Also handles e-format precision strings (e.g., ``".10e"``) that
    would otherwise pass through ``.replace("g", "f")`` unchanged.
    """
    # Extract the significant-digit count from the precision spec.
    # ".8g" → 8, ".10e" → 10, ".6f" → 6
    m = re.match(r"\.(\d+)", precision)
    sig_digits = int(m.group(1)) if m else 8

    if value == 0:
        return format(value, f".{sig_digits}f")

    magnitude = math.floor(math.log10(abs(value)))
    # Values >= 1e8 hit exponent with .8g but .8f is fine.
    # Values < 1: need sig_digits - magnitude - 1 decimal places
    #   e.g., 1.2345678e-05 (mag=-5): need 8 - (-5) - 1 = 12 → ".12f"
    decimal_places = sig_digits + max(0, (-magnitude) - 1)
    # Cap at a practical maximum to avoid excessively long output but
    # high enough to represent values down to ~1e-92 without precision loss
    # (8 sig_digits + 92 minus-1 = 99 decimal places).
    # F-016: Raised from 30 to 100; removed the exponent-notation fallback
    # below — the LAS spec forbids exponent notation in data sections.
    decimal_places = min(decimal_places, 100)
    # Ensure at least sig_digits places (for values > 1)
    decimal_places = max(decimal_places, sig_digits)

    result = format(value, f".{decimal_places}f")
    return result
