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
from .models import (
    _MNEMONIC_PATTERN,
    CurveDefinition,
    LASFile,
    ParameterEntry,
    _GuardedDict,
    _GuardedList,
)

# ── Module-level constants & compiled regexes ────────────────────────────

# Control characters except space and tab (which are valid LAS whitespace).
# Tab (\x09) is handled separately in _sanitize_las_value — it is replaced
# with a space to prevent mis-tokenization on re-read.  A tab inside an
# identifier acts as a field separator for str.split(), corrupting the
# parsed structure.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029]"
)

# Unicode whitespace characters that should be replaced with an ASCII
# space, not silently deleted.  These are layout/presentation characters
# (non-breaking space, en/em quads, thin spaces, ideographic space) that
# act as visual word separators.
_UNICODE_WS_RE = re.compile(
    r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]"
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


def _sanitize_las_value(value: str, *, preserve_leading_tilde: bool = False) -> str:
    """Sanitize a string for safe inclusion in LAS output.

    Args:
        preserve_leading_tilde: If True, a leading ``~`` (and any preceding
            whitespace) is NOT stripped.  The default strips a line-start
            ``~[A-Za-z]`` pattern so a value never mimics a LAS section
            header (``~CURVE``, ``~WELL``...).  That strip is only required
            for text emitted at the START of an output line.  Values emitted
            mid-line (well values, parameter values, descriptions,
            non-first-column data cells) can never be confused with a section
            header, and stripping them silently corrupts the model value on
            write→read (M-28).  Pass True for such mid-line content so the
            value survives roundtrip unchanged.
    """
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
    value = _UNICODE_WS_RE.sub(" ", value)
    if not preserve_leading_tilde:
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


def _escape_pipes_for_las_value(value: str) -> str:
    """Escape literal pipes in a LAS description (``|`` → ``\\|``).

    N-I-02: The parser treats a pipe at the END of a parameter description
    as a LAS 3.0 zone association (``| Zone``) and strips it.  Genuine
    description text that happens to contain a pipe would therefore be
    truncated and misinterpreted on re-read.  Escaping literal pipes keeps
    them out of ZONE_ASSOC_PATTERN's reach while real zone associations
    (appended separately by the writer, unescaped) still round-trip.
    The parser reverses this with ``_unescape_pipes_for_las_value``.
    """
    return value.replace("|", "\\|")


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
    # M-84: Bare known section types (e.g. "CORE") are accepted by the
    # model and by the parser's _SECTION_TYPE_MAP (which maps both "CORE"
    # and "CORE_DATA" → "CORE_DATA"), but the header prefix map only knows
    # the *_DATA forms.  Without this fallback a bare "CORE" falls back to
    # "A" + a misdiagnosing warning, and the re-read section_type silently
    # becomes LOG_DATA.  Try the canonical *_DATA form so the header stays
    # roundtrippable (defense-in-depth; the model also normalizes at
    # construction).
    _data_form = section_type + "_DATA"
    if _data_form in _SECTION_TYPE_TO_PREFIX:
        return _SECTION_TYPE_TO_PREFIX[_data_form]
    import warnings

    warnings.warn(
        f"Unknown section type '{section_type}'. "
        f"Falling back to ASCII data section header 'A'. "
        f"Known types: {', '.join(sorted(_SECTION_TYPE_TO_PREFIX.keys()))}. "
        f"Custom types must end with '_DATA'.",
        stacklevel=3,
    )
    return "A"


def _format_curve_line(
    curve: CurveDefinition,
    is_las30: bool,
    string_mnemonics: frozenset[str] | None = None,
) -> str:
    """Format a single CurveDefinition as a LAS curve line.

    Args:
        string_mnemonics: Mnemonics of curves whose DATA lives in a
            string_data container for the emitted scope.  M-77: a LAS 3.0
            string curve with an empty (or non-'S') data_format would be
            emitted WITHOUT the {S} marker — the parser's ONLY string
            signal — and re-read as numeric, silently destroying the
            values.  Callers with string_data context pass this set so the
            writer forces the {S} marker.
    """
    unit = _sanitize_las_value(curve.unit) if curve.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = curve.description if curve.description else ""

    is_string_curve = string_mnemonics is not None and curve.mnemonic in string_mnemonics
    if is_las30 and is_string_curve and (curve.data_format or "").upper() != "S":
        # M-77: a string-data curve without data_format='S' would be
        # emitted markerless; the parser classifies columns by the {S}
        # marker ONLY, so the values are re-read as numeric nulls.  Force
        # the {S} marker.  When a conflicting non-empty format is declared
        # the marker swap is a real contract change — warn loudly.
        if curve.data_format:
            import warnings

            warnings.warn(
                f"Curve '{curve.mnemonic}' is placed in string_data but "
                f"declares data_format={curve.data_format!r} (not 'S').  "
                f"Emitting the {{S}} marker so the parser recognizes the "
                f"values as strings; the declared format is not "
                f"representable for string data.",
                UserWarning,
                stacklevel=3,
            )
        format_str = "{S}"
        desc = f"{desc}  {format_str}"
    elif curve.data_format and (is_las30 or curve.data_format == "I"):
        # EXT-04: the braced {I} marker is emitted for integer-format
        # curves on ALL versions.  LAS 1.2/2.0 have no format-specifier
        # convention, but without the marker a >2^53 {I} value (e.g.
        # 9007199254740993) is re-read as float64 and silently rounded —
        # the marker is the only way the data reader restores integer
        # parsing on write→read roundtrip.  Other formats (F/E/S/A) remain
        # unmarked on LAS 1.2/2.0 to preserve existing output (string
        # curves are lossy on LAS 2.0 by design — see M-29).
        format_str = f"{{{curve.data_format}"
        if curve.data_format == "A" and curve.array_info and curve.array_info.time_offset is not None:
            offset = curve.array_info.time_offset
            if math.isfinite(offset):
                if offset == int(offset):
                    # IEEE 754 negative zero: int(-0.0) == 0 loses the sign.
                    # Use float formatting to preserve "-0" in the output,
                    # matching the copysign guard in _format_number.
                    if offset == 0 and math.copysign(1.0, offset) < 0:
                        format_str += f":{offset}"
                    else:
                        format_str += f":{int(offset)}"
                else:
                    format_str += f":{_format_offset_plain(offset)}"
        format_str += "}"
        desc = f"{desc}  {format_str}"
    elif curve.data_format:
        # M-27: non-LAS-3.0 output cannot represent a non-{I} format
        # specifier — the metadata is silently dropped on write→read.
        import warnings

        warnings.warn(
            f"Curve '{curve.mnemonic}' data_format "
            f"{curve.data_format!r} cannot be represented in LAS "
            f"1.2/2.0 output — it is dropped on write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )
    if curve.array_info is not None and not is_las30:
        # M-27: array_info (and the bracket mnemonic that carries it) is
        # only emitted for LAS 3.0; on 1.2/2.0 the metadata is silently
        # dropped on write→read.
        import warnings

        warnings.warn(
            f"Curve '{curve.mnemonic}' array_info is not representable "
            f"in LAS 1.2/2.0 output — it is dropped on write→read "
            f"roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    api_code = (
        _sanitize_las_value(curve.api_code, preserve_leading_tilde=True) if curve.api_code else ""
    )
    api_code = _escape_colons_for_las_value(api_code)
    api = f"  {api_code}" if api_code else ""
    desc = _sanitize_las_value(desc, preserve_leading_tilde=True)
    desc = _escape_colons_for_las_value(desc)

    # M-59: Emit the vendor-standard mnemonic when the model records one.
    # The reader's mnem_base rename (e.g. LLD→BFV) preserves the original
    # name in CurveDefinition.original_mnemonic, but the writer previously
    # emitted curve.mnemonic ONLY — so a read→write roundtrip
    # permanently canonicalized the colliding vendor name in the output
    # file (re-read without mnem_base never recovers it).  When
    # original_mnemonic is set and differs, emit it so the write
    # reconstructs the vendor-standard name.  Data columns stay
    # positional, so values remain aligned (verified: the parser routes
    # data by ~C order, not by the ~A header).
    _emit_mnem = (
        curve.original_mnemonic
        if curve.original_mnemonic and curve.original_mnemonic != curve.mnemonic
        else curve.mnemonic
    )
    mnemonic = _sanitize_las_value(_emit_mnem)
    if (
        is_las30
        and curve.array_info is not None
        and "[" not in mnemonic
    ):
        # W-09: The parser reconstructs CurveDefinition.array_info ONLY
        # from bracket mnemonics (ARRAY_MNEMONIC_PATTERN).  A curve whose
        # mnemonic lacks "[N]" loses its array_info on roundtrip — an
        # {A:N} format curve with array_info but no bracket is treated as
        # string-format on re-read and its numeric data is reclassified
        # into string_data.  Emit the bracket form so array_info survives
        # and the data stays numeric.  The "[" guard avoids
        # double-bracketing when the mnemonic already carries "[N]".
        mnemonic = f"{mnemonic}[{curve.array_info.index}]"

    return f" {mnemonic}.{unit}{api}  : {desc}"


def _format_parameter_line(param: ParameterEntry, is_las30: bool) -> str:
    """Format a single ParameterEntry as a LAS parameter line."""
    unit = _sanitize_las_value(param.unit) if param.unit else ""
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = param.description if param.description else ""
    # N-I-02(b): Escape literal pipes in the description BEFORE any zone
    # association is appended.  A genuine pipe in user text would otherwise
    # be misparsed as a LAS 3.0 zone association (| Zone) on re-read —
    # truncating the description and attaching a bogus ParameterZone.  Real
    # zone associations appended below remain unescaped so the parser's
    # ZONE_ASSOC_PATTERN still recognizes them.
    desc = _escape_pipes_for_las_value(desc)

    if is_las30 and param.data_format:
        # N-I-21: Always emit the braced {…} form.  Previously
        # multi-character values (e.g. "DD/MM/YYYY") were emitted
        # UNBRACED, which on re-read merged the format text into the
        # description and lost the data_format field entirely.  The
        # braced form is the valid LAS 3.0 construct; the parser
        # recognizes it (extracting or clearing the format while
        # keeping the description stable), so the roundtrip is
        # deterministic across construction paths.
        desc = f"{desc}  {{{param.data_format}}}"
    elif param.data_format:
        # M-27: LAS 1.2/2.0 output cannot represent a braced format
        # specifier — the metadata is silently dropped on write→read.
        import warnings

        warnings.warn(
            f"Parameter '{param.mnemonic}' data_format "
            f"{param.data_format!r} cannot be represented in LAS "
            f"1.2/2.0 output — it is dropped on write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    if is_las30 and param.zone:
        # M-12: escape literal pipes in the ZONE NAME itself.  A genuine
        # pipe inside zone_name (e.g. "Zone|X") would otherwise be
        # re-parsed as the LAST '|' fragment of the zone association —
        # ZONE_ASSOC_PATTERN's (?<!\\)\| matches any non-escaped pipe, so
        # "| Zone|X[2]" re-reads zone="X" and leaks "| Zone" into the
        # description.  Escaping the zone_name pipe keeps it inside the
        # zone text; the parser unescapes it after extraction.
        zone_name = _escape_pipes_for_las_value(param.zone.zone_name)
        zone_str = f" | {zone_name}"
        if param.zone.zone_index is not None:
            zone_str += f"[{param.zone.zone_index}]"
        desc = f"{desc}{zone_str}"
    elif param.zone:
        # M-27: LAS 1.2/2.0 output cannot represent a zone association —
        # the metadata is silently dropped on write→read.
        import warnings

        warnings.warn(
            f"Parameter '{param.mnemonic}' zone association is not "
            f"representable in LAS 1.2/2.0 output — it is dropped on "
            f"write→read roundtrip.",
            UserWarning,
            stacklevel=3,
        )

    value = _sanitize_las_value(param.value, preserve_leading_tilde=True)
    desc = _sanitize_las_value(desc, preserve_leading_tilde=True)
    value = _escape_colons_for_las_value(value)
    desc = _escape_colons_for_las_value(desc)

    mnemonic = _sanitize_las_value(param.mnemonic)
    if (
        is_las30
        and param.array_index is not None
        and "[" not in mnemonic
    ):
        # W-08: The parser reconstructs ParameterEntry.array_index ONLY
        # from bracket mnemonics (ARRAY_MNEMONIC_PATTERN).  A parameter
        # whose mnemonic lacks "[N]" loses its array_index on roundtrip
        # (RUN with array_index=1 → array_index=None).  Emit the bracket
        # form so the index survives.  The "[" guard avoids
        # double-bracketing when the mnemonic already carries "[N]".
        mnemonic = f"{mnemonic}[{param.array_index}]"

    return f" {mnemonic}.{unit}  {value}  : {desc}"


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
    warned_tilde_str = False
    for i in range(num_rows):
        row_values: list[str] = []
        for arr, is_string in curve_arrays:
            if arr is None or i >= len(arr):
                if is_string:
                    # M-78: a short (ragged) string curve was padded with
                    # the NUMERIC null sentinel, which on re-read becomes
                    # a fabricated "-999.25" STRING value.  Route missing
                    # string values through the string-branch missing-value
                    # routing (the '-' sentinel) instead.
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Missing string curve value (short array) "
                            "padded with '-' sentinel — roundtrip "
                            "fidelity is lost: parser cannot distinguish "
                            "original '-' from the missing value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    row_values.append("-")
                else:
                    row_values.append(_format_number(null_value, precision, null_value))
            elif is_string:
                _raw = arr[i]
                # N-I-17: A None/NaN/Inf value in a string-data array was
                # written as the literal string "None"/"nan" (via str()),
                # fabricating data on re-read — the numeric branch routes
                # non-finite values to the null sentinel, but the string
                # branch had no guard.  Route missing values to the same
                # '-' sentinel used for empty strings.
                if _raw is None or (
                    isinstance(_raw, (float, np.floating))
                    and not math.isfinite(_raw)
                ):
                    if not warned_empty_str:
                        import warnings

                        warnings.warn(
                            "Missing string curve value (None/NaN/Inf) "
                            "replaced by '-' sentinel — roundtrip "
                            "fidelity is lost: parser cannot distinguish "
                            "original '-' from the missing value.",
                            stacklevel=4,
                        )
                        warned_empty_str = True
                    row_values.append("-")
                    continue
                raw_val = str(_raw)
                raw_has_delim = delimiter in raw_val
                # M-28: preserve a leading '~' in string DATA values —
                # stripping it silently corrupts the model value.  Mid-row
                # cells can never be confused with a section header.
                val = _sanitize_las_value(raw_val, preserve_leading_tilde=True)
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
                if not row_values and re.match(r"\s*~[A-Za-z]", val):
                    # M-28: this value lands in the FIRST column, so the
                    # emitted data row would start with '~'+letter and the
                    # reader would treat it as a LAS section header (the
                    # writer's own reader skips such lines) — dropping the
                    # row and breaking the file structure.  The value is
                    # preserved everywhere else, but here the leading '~'
                    # MUST be stripped for the file to stay valid — warn
                    # loudly that the value was altered.
                    val = _LEADING_SECTION_RE.sub(r"\1", val, count=1)
                    if not warned_tilde_str:
                        import warnings

                        warnings.warn(
                            "String curve value in the first data column "
                            "begins with '~' followed by a letter; the "
                            "emitted row would be misread as a LAS section "
                            "header on re-read.  The leading '~' was "
                            "removed, so the written value differs from "
                            "the model.",
                            stacklevel=4,
                        )
                        warned_tilde_str = True
                elif not row_values and re.match(r"\s*~", val):
                    # M-85: value in the FIRST data column starts with '~'
                    # + NON-letter (e.g. '~3D', '~.', bare '~').  The
                    # M-28 guard above only matches '~'+letter; these
                    # survive _sanitize_las_value, so the emitted data row
                    # would BEGIN with '~' — and the parser/reader skip
                    # '~'-prefixed lines as section-header noise
                    # (parser.py F-83 / _data_section_reader.py), silently
                    # dropping the ENTIRE row to `other` with zero
                    # warnings.  Escape the leading '~' as '_~' (mirroring
                    # the existing '#'-prefix escape) so the line never
                    # starts with '~'; the parser's
                    # _desanitize_las_value restores '_~' → '~' on re-read.
                    val = "_" + val.lstrip()
                    if not warned_tilde_str:
                        import warnings

                        warnings.warn(
                            "String curve value in the first data column "
                            "begins with '~' followed by a non-letter; the "
                            "emitted row would be misread as a LAS section "
                            "header and silently dropped on re-read.  The "
                            "leading '~' was escaped as '_~' (restored on "
                            "re-read) to keep the row in the file.",
                            stacklevel=4,
                        )
                        warned_tilde_str = True
                row_values.append(val)
            else:
                val = arr[i]
                # IT3-F-02: math.isfinite is ~15x faster than the numpy
                # scalar isfinite/isnan/isinf chain for per-value checks and
                # is semantically identical for Python/numpy float scalars
                # (verified: no NaN-propagation divergence).  Array-vectorized
                # numpy uses elsewhere are untouched.
                if not math.isfinite(val):
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
    # IT3-F-02: math.isfinite (~15x faster than np.isnan/np.isinf for
    # Python float scalars, semantically identical for scalars).
    if not math.isfinite(value):
        if null_value is not None:
            return _format_null_sentinel(null_value, precision)
        return format(float(value), precision)
    if null_value is not None and value == null_value:
        return _format_null_sentinel(null_value, precision)
    if isinstance(value, (int, np.integer)):
        # EXT-04: integer-typed values (exact Python ints from object-dtype
        # {I} arrays, np.int64 from int64 arrays) must be formatted via
        # integer formatting.  `format(int(value), precision)` converts
        # through float64 internally whenever the .Ng result needs
        # scientific notation, silently rounding values above 2^53
        # (9007199254740993 → '9007199254740992.00000000').  str(int())
        # preserves the exact decimal.
        return str(int(value))
    if value == int(value):
        # IEEE 754 negative zero (-0.0): int(-0.0) == 0 loses the sign,
        # producing "0" instead of "-0".  Use float formatting for the
        # negative-zero case so that Python's native format(-0.0, ...)
        # correctly emits "-0".
        if value == 0 and math.copysign(1.0, value) < 0:
            result = format(float(value), precision)
        else:
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


# F-15: The parser's FORMAT_SPEC_PATTERN offset group is bounded at 64
# characters ([-\d.]{0,64}, parser.py) to keep the ReDoS-bounded quantifier
# linear.  The writer must never emit an offset field longer than that or
# the whole {A:N} spec fails to parse (the M-56 defect class).  '0.' +
# decimal_places fits in 64 chars up to 62 decimals; use 61 to leave room
# for a leading '-' sign on negative offsets.
_MAX_OFFSET_FIXED_DECIMALS: int = 61


def _format_offset_plain(offset: float) -> str:
    """Format a float WITHOUT scientific notation for ``{A:N}`` offsets.

    M-56: Python's default ``str()`` formats values in (0, 1e-4) as
    scientific notation (``9e-05``), which the parser's FORMAT_SPEC_PATTERN
    offset group ``[-\\d.]*`` cannot parse — the entire ``{A:N}`` spec is
    then treated as description text, losing data_format and time_offset.
    Values >= 1e-4 already format as fixed-point (str() has no exponent),
    so only the scientific case is rewritten.

    F-08: the fixed-point rewrite preserves ALL significant digits of the
    offset (counted from the shortest-repr mantissa), using the same
    magnitude-aware formula as ``_format_fixed_precision`` — the earlier
    ``-magnitude + 2`` heuristic produced only ~2-3 significant digits for
    offsets in (0, 1e-4) (1.2345e-05 → '0.0000123', 0.36% error).

    F-15: the emitted field is CLAMPED to ``_MAX_OFFSET_FIXED_DECIMALS`` so
    it never exceeds the parser's 64-character offset-group cap.  Offsets
    too small to represent exactly within the cap are rounded and a LOUD
    warning is emitted — data_format and the {A:N} spec survive the
    roundtrip (unlike the pre-fix >32-char fields which failed to parse and
    leaked the literal spec into the description).
    """
    s = str(offset)
    if "e" not in s.lower():
        return s
    if offset == 0:
        return "0"
    # Significant digits in the shortest repr — the minimum needed to
    # round-trip the float64 exactly.
    mantissa = s.split("e")[0]
    sig_digits = len(mantissa.replace(".", "").replace("-", ""))
    magnitude = math.floor(math.log10(abs(offset)))
    decimal_places = sig_digits + max(0, (-magnitude) - 1)
    if decimal_places > _MAX_OFFSET_FIXED_DECIMALS:
        import warnings

        warnings.warn(
            f"time_offset {offset!r} cannot be represented exactly in the "
            f"{{A:N}} offset field (needs {decimal_places} decimal places, "
            f"capped at {_MAX_OFFSET_FIXED_DECIMALS} to stay within the "
            f"parser's 64-character offset group).  The emitted value is "
            f"rounded and may not round-trip exactly.",
            UserWarning,
            stacklevel=3,
        )
        decimal_places = _MAX_OFFSET_FIXED_DECIMALS
    return format(offset, f".{decimal_places}f")


def _warn_long_header_lines(lines: list[str], max_length: int) -> None:
    """Warn if any header section line exceeds the LAS length limit.

    The LAS 1.2 CWLS specification limits ALL lines (including header
    lines) to 256 characters; the LAS 2.0 specification applies the same
    256-char limit to WRAP=NO files.  This check covers header-section
    lines (version, well, curve, parameter, other) that were not
    previously validated — data rows are checked separately in
    ``_format_data_rows``.  N-I-16: previously gated on ``is_las12``
    only, so LAS 2.0 WRAP=NO header lines were never checked.
    """
    warned = False
    for line in lines:
        if len(line) > max_length:
            if not warned:
                import warnings

                warnings.warn(
                    f"LAS header line exceeds {max_length}-character "
                    f"limit: {line[:80]!r}... ({len(line)} chars). "
                    f"The LAS 1.2/2.0 (WRAP=NO) specification limits all "
                    f"lines (including header lines) to {max_length} "
                    f"characters.  Subsequent violations in this file "
                    f"will not be reported.",
                    stacklevel=4,
                )
                warned = True


# ── _WriterMutationGuard ────────────────────────────────────────────────

class _WriterMutationGuard:
    """Context manager that runs deferred validation after write.

    Saves a snapshot of the LASFile state that is affected by the write
    pass (wrap/dlm flags, logs/string_data/curves containers).  On the
    SUCCESS path the model is intentionally NOT restored — the model must
    honestly reflect what was written to disk (documented G-018 intent:
    e.g. ``WRAP=YES`` is written as ``NO``).  On the FAILURE path the
    saved state IS restored so the caller's model is not left partially
    mutated by an aborted write.

    The version-specific writers' ``finally`` blocks restore the data
    containers from plain ``dict``/``list`` snapshots, which permanently
    strips the ``_GuardedDict``/``_GuardedList`` mutation guards.  This
    guard re-wraps the containers so invalid mutations are still caught
    after a write (success or failure).
    """

    def __init__(self, las_file: LASFile) -> None:
        self._las_file = las_file
        self._saved_wrap: str = las_file.version.wrap
        self._saved_dlm: str = las_file.version.dlm
        self._saved_logs = dict(las_file.logs)
        self._saved_string_data = dict(las_file.string_data)
        self._saved_curves_order = (
            list(las_file.curves_order) if las_file.curves_order is not None else None
        )
        self._saved_curves = (
            list(las_file.curves) if las_file.curves is not None else None
        )

    def __enter__(self) -> _WriterMutationGuard:
        return self

    def _restore_saved_state(self) -> None:
        """Restore the pre-write snapshot on the failure path."""
        las_file = self._las_file
        las_file.version.wrap = self._saved_wrap
        las_file.version.dlm = self._saved_dlm
        las_file.logs = self._saved_logs
        las_file.string_data = self._saved_string_data
        las_file.curves_order = self._saved_curves_order
        las_file.curves = self._saved_curves

    def _rewrap_guards(self) -> None:
        """Re-wrap data containers in guarded dict/list after a write.

        W-06: the version-specific writers restore logs/string_data/
        curves from plain dict/list snapshots in their ``finally`` blocks,
        permanently stripping the mutation guards installed at LASFile
        construction.  Re-install the guards so subsequent invalid
        mutations are still rejected.  None containers (a valid state for
        directly-constructed files) are left untouched.
        """
        las_file = self._las_file
        if las_file.logs is not None and not isinstance(las_file.logs, _GuardedDict):
            las_file.logs = _GuardedDict(
                las_file.logs, _container_name="LASFile.logs"
            )
        if las_file.string_data is not None and not isinstance(
            las_file.string_data, _GuardedDict
        ):
            las_file.string_data = _GuardedDict(
                las_file.string_data, _container_name="LASFile.string_data"
            )
        if las_file.curves is not None and not isinstance(las_file.curves, _GuardedList):
            las_file.curves = _GuardedList(
                las_file.curves,
                _container_name="LASFile.curves",
                _expected_type=CurveDefinition,
            )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        import warnings

        if exc_type is not None:
            # W-07: failure path — restore the saved state so the caller's
            # model is not left partially mutated by the failed write.
            self._restore_saved_state()
        else:
            try:
                issues = self._las_file.validate(complete=True)
                for msg in issues:
                    warnings.warn(msg, UserWarning, stacklevel=2)
            except Exception:
                pass

        # W-06: re-install mutation guards stripped by the writers' finally
        # blocks (runs on both success and failure paths).
        self._rewrap_guards()

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
        if self._spec.is_las30:
            self._warn_string_curves_without_s_marker()
        lines: list[str] = []
        lines.extend(self._write_version_section())
        lines.extend(self._write_well_section())
        lines.extend(self._write_curve_section())
        lines.extend(self._write_parameter_section())
        lines.extend(self._write_other_section())
        # N-I-16: The header-line length check was gated on `is_las12` only,
        # so LAS 2.0 WRAP=NO header lines (also subject to the CWLS 256-char
        # limit per `line_length_limit_for_wrap`) were never checked.  Use
        # the effective wrap — the writers ALWAYS emit non-wrapped output
        # (WRAP=YES is overridden to NO), so the effective limit for LAS
        # 1.2/2.0 output is always 256, matching the data-row check.
        effective_wrap = (self._las_file.version.wrap or "NO").upper()
        if effective_wrap == "YES":
            effective_wrap = "NO"
        header_limit = self._spec.line_length_limit_for_wrap(effective_wrap)
        if header_limit is not None:
            _warn_long_header_lines(lines, header_limit)
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

        # N-I-19: Defensive well-key CONTENT validation.  WellSection.
        # __post_init__ rejects non-roundtrippable keys at construction,
        # but entries can still be mutated afterwards (well.entries is a
        # plain dict).  A key containing dots/spaces/colons is emitted and
        # then silently DROPPED on re-read — the parser's ~W regex
        # (DATA_LINE_PATTERN mnemonic group) cannot match it.  Reject here
        # rather than emit metadata that cannot survive a roundtrip.
        for key in self._las_file.well.entries:
            if not _MNEMONIC_PATTERN.fullmatch(key):
                raise ValueError(
                    f"WellSection entry key {key!r} contains characters "
                    f"the LAS parser cannot roundtrip.  Well keys must "
                    f"match {_MNEMONIC_PATTERN.pattern!r}."
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
            # M-28: well values/descriptions are emitted mid-line (never at
            # line start) so a leading '~' must be preserved, not stripped.
            val = _sanitize_las_value(value, preserve_leading_tilde=True)
            desc = _sanitize_las_value(
                self._las_file.well.descriptions.get(key, ""),
                preserve_leading_tilde=True,
            )
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

        curves = self._las_file.curves
        if (
            not curves
            and not self._spec.is_las30
            and len(self._las_file.data_sections) == 1
            and self._las_file.data_sections[0].section_curves
        ):
            # W-05: The single-section copy-back (Path A in
            # _write_ascii_legacy) runs during the ASCII data pass, which
            # happens AFTER this section is emitted.  Consult the section's
            # curve definitions directly so ~C is not emitted EMPTY while
            # ~A carries the data columns — otherwise curve metadata
            # (units, descriptions, API codes) is silently lost from the
            # output and the data is discarded on re-read.
            curves = self._las_file.data_sections[0].section_curves

        for curve in curves:
            # M-77: pass the string_data mnemonic set so a string curve
            # without data_format='S' still gets the {S} marker.  On
            # 1.2/2.0 the is_las30 gate inside _format_curve_line keeps
            # the marker off (string curves are lossy there by design and
            # validate(complete=True) already warns).
            lines.append(
                _format_curve_line(
                    curve,
                    self._spec.is_las30,
                    frozenset(self._las_file.string_data.keys()),
                )
            )

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

    def _warn_string_curves_without_s_marker(self) -> None:
        """M-77: warn when a LAS 3.0 string curve lacks the {S} marker.

        The parser classifies a column as string ONLY from the {S} marker
        in its ~C/Definition line.  A string-data curve with an empty (or
        non-'S') data_format is emitted markerless and its values are
        re-read as numeric nulls — silent destruction.  This check covers
        BOTH the top-level (no data_sections) and per-section paths from
        the base template ``write()``; callers that pass string_data
        context into ``_format_curve_line`` also emit {S} directly.

        F-09: the M-77 {S}-forcing branch in ``_format_curve_line`` emits
        the {S} marker for any string-data curve whose mnemonic is in the
        string_mnemonics set passed for its emitted scope — the main ~C
        block passes the UNION of every scope's string_data keys and
        per-section Definitions pass the section's own keys.  When the
        curve IS in that set the values round-trip intact, so warning
        here would misdiagnose the exact scenario the fix prevents.
        Only warn when the marker is genuinely absent for the emitted
        scope (the curve's mnemonic is NOT in the union string_mnemonics
        set) — i.e. string data would actually be lost.
        """
        import warnings

        las_file = self._las_file
        top_curves = {c.mnemonic: c for c in las_file.curves or []}
        warned: set[str] = set()
        # Union of EVERY string_data mnemonic across all scopes — the set
        # the main ~C block passes to _format_curve_line
        # (_Las30Writer._all_string_mnemonics), which is a superset of
        # every per-section Definition set.  Membership here means {S} is
        # forced at emission.
        emitted_str_mnems: set[str] = set(las_file.string_data or {})
        for ds in las_file.data_sections:
            emitted_str_mnems.update(ds.string_data or {})

        def _warn_for(curve: CurveDefinition, mnem: str) -> None:
            if (curve.data_format or "").upper() == "S" or mnem in warned:
                return
            if curve.mnemonic in emitted_str_mnems:
                # F-09: {S} is forced via string_mnemonics — no loss.
                return
            warnings.warn(
                f"LAS 3.0 string curve '{mnem}' has "
                f"data_format={(curve.data_format or '')!r} (not 'S').  "
                f"Without the {{S}} marker the parser reads this column "
                f"as numeric and the string values are lost on "
                f"write→read roundtrip.",
                UserWarning,
                stacklevel=2,
            )
            warned.add(mnem)

        # Top-level string_data scope (no-data_sections path).
        for mnem in las_file.string_data or {}:
            cd = top_curves.get(mnem)
            if cd is not None:
                _warn_for(cd, mnem)
        # Per-section scopes — definitions may live in the section itself
        # or fall back to the top-level curves list.
        for ds in las_file.data_sections:
            sec_curves = {c.mnemonic: c for c in ds.section_curves or []}
            for mnem in ds.string_data or {}:
                cd = sec_curves.get(mnem) or top_curves.get(mnem)
                if cd is not None:
                    _warn_for(cd, mnem)

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
            _ds = self._las_file.data_sections[0]
            # W-04: The copy-back below only fills EMPTY top-level
            # containers.  When a top-level container is already
            # populated, the corresponding section content is dropped.
            # Warn honestly about the actual copy-back outcome instead
            # of always claiming "Single-section data will be preserved."
            _dropped: list[str] = []
            if _ds.data and self._las_file.logs:
                _dropped.append("data")
            if _ds.string_data and self._las_file.string_data:
                _dropped.append("string data")
            if _ds.section_curves and self._las_file.curves:
                _dropped.append("curve definitions")
            if _dropped:
                warnings.warn(
                    "data_sections are only supported for LAS 3.0 files. "
                    "Falling back to single-section ~A format.  Section "
                    f"content will NOT be preserved because the "
                    f"corresponding top-level container is already "
                    f"populated: {', '.join(_dropped)}.",
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    "data_sections are only supported for LAS 3.0 files. "
                    "Falling back to single-section ~A format. "
                    "Single-section data will be preserved.",
                    stacklevel=3,
                )
            if not self._las_file.logs and _ds.data:
                self._las_file.logs.update(_ds.data)
            if not self._las_file.string_data and _ds.string_data:
                self._las_file.string_data.update(_ds.string_data)
            if not self._las_file.curves_order and _ds.curves_order:
                self._las_file.curves_order = list(_ds.curves_order)
            if not self._las_file.curves and _ds.section_curves:
                self._las_file.curves = list(_ds.section_curves)

            if (self._las_file.curves_order and _ds.curves_order
                    and (self._las_file.logs or self._las_file.string_data)):
                existing = set(self._las_file.curves_order)
                for k in _ds.curves_order:
                    if (k in self._las_file.logs or k in self._las_file.string_data) and k not in existing:
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
                    set(self._las_file.string_data.keys()) if self._las_file.string_data else set()
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
            name in self._las_file.logs or name in self._las_file.string_data
            for name in curve_names
        ):
            warnings.warn(
                f"curves_order contains {len(curve_names)} curve(s) "
                f"but none have data in logs or string_data. "
                f"No data will be emitted.",
                stacklevel=3,
            )
        if any(
            name in self._las_file.logs or name in self._las_file.string_data
            for name in curve_names
        ):
            # M-59: Keep the ~A column-header line consistent with the
            # ~C curve lines.  The ~C section now emits
            # CurveDefinition.original_mnemonic (the vendor-standard name)
            # when it differs from curve.mnemonic; the ~A header must use
            # the SAME emitted names or an external parser sees a
            # column-header/curve mismatch (pylasdev routes data
            # positionally by ~C order, so its own roundtrip is
            # unaffected — this is for file-level consistency).
            _by_mnem = {c.mnemonic: c for c in self._las_file.curves or []}

            def _header_name(name: str) -> str:
                c = _by_mnem.get(name)
                if c is not None and c.original_mnemonic and c.original_mnemonic != c.mnemonic:
                    return c.original_mnemonic
                return name

            header_line = "~A  " + "  ".join(
                _sanitize_las_value(_header_name(name)) for name in curve_names
            )
            # N-I-16: The ~A column-header line is appended AFTER the
            # header-section length check in `write()` (which runs before
            # `_write_ascii_sections`), so it was NEVER length-checked for
            # any version.  Data rows ARE checked (`_format_data_rows`), so
            # a long ~A header slipped through with 0 warnings while the
            # data rows below it warned.  Apply the same limit here when
            # `check_line_limit` is active (LAS 1.2 all modes, LAS 2.0
            # WRAP=NO).
            if check_line_limit and len(header_line) > MAX_LINE_LENGTH_LAS12:
                import warnings

                warnings.warn(
                    f"~A column-header line exceeds 256-character limit "
                    f"(length: {len(header_line)}).  The LAS 1.2/2.0 "
                    f"specification limits all lines (including column "
                    f"headers) to 256 characters.  Lines are NOT truncated "
                    f"to avoid data loss.",
                    stacklevel=4,
                )
            lines.append(header_line)
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

    # W-02: A bare precision specifier (e.g. ".5") is accepted by
    # _validate_precision but format(int(v), ".5") raises ValueError
    # ("Precision not allowed in integer format specifier") whenever a
    # numeric value is integral — depths are commonly integral, so the
    # write crashes mid-output exactly when real data exists.  Normalize
    # bare ".N" to ".Ng" so integral values format without crashing.
    precision = re.sub(r"^\.(\d+)$", r".\1g", precision)

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
        raise LASWriteError(f"Cannot create output directory {file_path.parent}: {e}") from e

    # Version dispatch: choose the correct writer class.
    spec = _LASVersionSpec(las_file.version.vers)
    if spec.is_las12:
        from ._writer_las12 import _Las12Writer

        writer: _WriterBase = _Las12Writer(las_file, precision)
    elif spec.is_las20:
        from ._writer_las20 import _Las20Writer

        writer = _Las20Writer(las_file, precision)
    elif spec.is_las30:
        from ._writer_las30 import _Las30Writer

        writer = _Las30Writer(las_file, precision)
    else:
        raise LASWriteError(
            f"Unsupported LAS version: {las_file.version.vers!r}. "
            f"Supported versions are LAS 1.2, 2.0, and 3.0."
        )

    with _WriterMutationGuard(las_file):
        try:
            content = writer.write()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, PylasdevError) as e:
            raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

        try:
            target_dir = str(file_path.parent)
            fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_", suffix=file_path.name)
            try:
                with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
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
