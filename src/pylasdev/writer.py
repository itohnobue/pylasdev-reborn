"""LAS file writer.

Replaces las_writer.py with proper metadata preservation.
The original writer destroyed units (wrote '.X') and descriptions (wrote 'X').
This version preserves the original metadata when available.

Supports LAS 1.2, 2.0, and 3.0 formats.
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

# Control characters except space and tab (which are valid LAS whitespace).
# Tab (\x09) is handled separately in _sanitize_las_value — it is replaced
# with a space to prevent mis-tokenization on re-read.  A tab inside an
# identifier acts as a field separator for str.split(), corrupting the
# parsed structure.
# Matches \x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F (DEL), \x85 (NEL),
# \u2028 (LINE SEPARATOR), and \u2029 (PARAGRAPH SEPARATOR).
# The Unicode line break characters are treated as line breaks by Python's
# splitlines() but are not caught by \n/\r replacement.
# Also matches Unicode whitespace characters that Python's str.split()
# treats as field separators but are not in the ASCII control range:
# \u00A0 (NO-BREAK SPACE), \u2000-\u200A (EN QUAD..HAIR SPACE),
# \u202F (NARROW NO-BREAK SPACE), \u205F (MEDIUM MATHEMATICAL SPACE),
# and \u3000 (IDEOGRAPHIC SPACE).  These cause phantom field splits on
# re-read via str.split(), corrupting parsed structure.
# NOTE: Pipe (|, 0x7C) is NOT included here because it is a legitimate
# structural character in LAS 3.0 zone notation ("| RUN[1]").  Pipe
# stripping for section-name injection prevention is handled at the
# point of use in _write_ascii_sections.
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029"
    r"\u00A0\u2000-\u200A\u202F\u205F\u3000]"
)

# F-86: Previous pattern ^~([A-Za-z]) only matched a leading tilde.
# Values like "\t~Version" or "  ~Curve" bypassed section-header
# sanitization because leading whitespace prevented the regex match.
# Extended to ^\s*~([A-Za-z]) — the \s* consumes leading whitespace
# and the replacement \1 strips both the whitespace and tilde.
_LEADING_SECTION_RE = re.compile(r"^\s*~([A-Za-z])")

# The 256-character line-length limit applies to:
#   - LAS 1.2 (all modes) per the LAS 1.2 specification.
#   - LAS 2.0 WRAP=NO per the CWLS specification.
# LAS 2.0 WRAP=YES has no line-length limit (lines are joined).
# LAS 3.0 mandates one data row per line with no explicit length limit.
# We warn when data rows exceed this limit but do NOT truncate
# (truncation would cause data loss).
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
    # F-40 (F2-005): Replace tab characters with spaces.  While tab is valid
    # LAS whitespace between fields, a tab inside an identifier or
    # value causes mis-tokenization on re-read (str.split() treats
    # tab as a field separator, corrupting the parsed structure).
    #
    # WRITER->PARSER CONTRACT: Tab->space is a permanently one-way
    # sanitization - the parser cannot distinguish a space that was
    # originally a tab from a space that was originally a space.
    # PARSER FIX NEEDED (in _desanitize_las_value, parser.py):
    #   Exact reversal is inherently ambiguous, but for roundtrip
    #   best-effort, add a sentinel-based encoding:
    #   - Writer: replace "\t" with "_TAB_" (reversible sentinel)
    #   - Parser: replace "_TAB_" back to "\t"
    #   OR: Accept this as a documented one-way transformation
    #   when using SPACE or TAB delimiter for string curve data.
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
    # F-I2-XWP-01 (coordinate with s6-fix-parser-b): The leading underscore
    # is a permanently one-way transformation — the parser has no
    # _desanitize function to strip it on re-read.  Parser-side reversal
    # is needed for full roundtrip fidelity of _# values.
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
        **bidirectional** — they are reversed during parsing by
        ``_unescape_colons_for_las_value`` (``parser.py``), which
        restores the original whitespace-colon adjacencies.  Values
        containing colons adjacent to whitespace (e.g., ``"Oil : Gas"``
        → ``"Oil _:_ Gas"`` on write, → ``"Oil : Gas"`` on read) WILL
        roundtrip to their original form.

        The roundtrip is reliable because the unescape step runs after
        the structural colon separator has already been consumed by the
        parser, so reversed colons cannot cause data truncation on re-read.
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


class _WriterMutationGuard:
    """Context manager that runs deferred validation after write.

    During write, several LASFile attributes may be mutated to ensure
    format compliance (WRAP→NO, DLM→SPACE for LAS 1.2, copy-back from
    data_sections for legacy paths).  This guard snapshots the pre-write
    state for comparison/detection purposes and runs
    ``las_file.validate(complete=True)`` in ``__exit__``, warning on any
    issues found.  State is intentionally NOT restored — the model must
    honestly reflect what was written to disk (G-018 "honest model"
    principle).  Individual write paths manage their own state restoration
    for non-semantic mutations (e.g., data_sections copy-back restore in
    ``_write_ascii_sections``'s ``finally`` block).
    """

    def __init__(self, las_file: LASFile) -> None:
        self._las_file = las_file
        # Snapshot mutable state at construction time for potential
        # future comparison/detection logging.  State is intentionally
        # NOT written back to las_file — the model must honestly reflect
        # what was written to disk (G-018).
        self._saved_wrap: str = las_file.version.wrap
        self._saved_dlm: str = las_file.version.dlm
        self._saved_logs = dict(las_file.logs)
        self._saved_string_data = dict(las_file.string_data)
        self._saved_curves_order = list(las_file.curves_order) if las_file.curves_order is not None else []
        self._saved_curves = list(las_file.curves) if las_file.curves is not None else []

    def __enter__(self) -> _WriterMutationGuard:
        """Enter the guarded context (state already snapped in __init__)."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Run deferred validation against the post-write model state.

        Does NOT restore pre-write state — the model must honestly
        reflect what was written to disk per G-018.  Runs
        ``las_file.validate(complete=True)`` and warns on issues.
        Does not suppress exceptions.
        """
        # Run deferred validation and warn on any issues found.
        import warnings

        try:
            issues = self._las_file.validate(complete=True)
            for msg in issues:
                warnings.warn(msg, UserWarning, stacklevel=2)
        except Exception:
            pass  # validate should not break the write

        # Do not suppress exceptions.
        return False  # type: ignore[return-value]


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

    # F-I2-W-02: Ensure the output directory exists BEFORE generating
    # content.  Content generation can be expensive (large files), so
    # failing fast on directory creation avoids wasted computation.
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LASWriteError(
            f"Cannot create output directory {file_path.parent}: {e}"
        ) from e

    # Always write with the specified encoding (default: utf-8).
    # F-I2-M04: Also catch PylasdevError — content generation may raise
    # PylasdevError subclasses from models.py validation.
    #
    # Wrap all write operations in _WriterMutationGuard: snapshots the
    # LASFile's mutable state before mutations begin, restores it after
    # the write completes (success or failure), and runs deferred
    # validate(complete=True) to catch any mutation-induced corruption.
    with _WriterMutationGuard(las_file):
        try:
            content = _generate_las_content(las_file, precision)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, PylasdevError) as e:
            raise LASWriteError(f"Failed to generate LAS file content: {e}") from e

        try:
            # Atomic write: write content to a temporary file in the same
            # directory as the target, then atomically replace the target.
            # This prevents partial/corrupt files on I/O interruption,
            # disk-full, or system crash mid-write.
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


def _generate_las_content(las_file: LASFile, precision: str = ".8g") -> str:
    """Generate LAS file content string with metadata preservation."""
    import warnings
    # Run deferred validation before generating content — warn only
    # (don't raise) to preserve existing writer behavior.
    for issue in las_file.validate(complete=True):
        warnings.warn(issue, stacklevel=2)
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
    is_las12 = _LASVersionSpec(las_file.version.vers).is_las12
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
        # F-03-H: WRAP mutation moved to _write_ascii_sections
        # (after save, inside try/finally) to prevent model corruption
        # if content generation fails between _write_version_section
        # and the restore point.  The header line above correctly emits
        # WRAP=NO via the local actual_wrap variable.
    # F-M-027: LAS 3.0 requires one data row per line (non-wrapped output).
    # Force WRAP=NO regardless of user-supplied wrap parameter to prevent
    # data rows from being joined inline, which would produce a single
    # giant line that downstream parsers reject.
    if is_las30 and actual_wrap != "NO":
        import warnings

        warnings.warn(
            f"LAS 3.0 WRAP={actual_wrap} overridden to WRAP=NO. "
            "LAS 3.0 requires one data row per line. "
            "The data will be written in non-wrapped format.",
            stacklevel=3,
        )
        actual_wrap = "NO"
        # F-03-H: WRAP mutation moved to _write_ascii_sections.
    wrap_desc = (
        "ONE LINE PER DEPTH STEP" if actual_wrap == "NO" else "MULTIPLE LINES PER DEPTH STEP"
    )
    lines.append(f" WRAP.   {_sanitize_las_value(actual_wrap)}  : {wrap_desc}")
    # DLM is defined in LAS 2.0 and 3.0 specs.  LAS 1.2 does not use
    # DLM and always defaults to SPACE.  When DLM is not SPACE (e.g.,
    # COMMA or TAB), emit the DLM line so the file declares the correct
    # delimiter — otherwise a re-read would default to SPACE, corrupting
    # comma- or tab-delimited data.
    if las_file.version.dlm and las_file.version.dlm.upper() != "SPACE" and not is_las12:
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
    is_las12 = _LASVersionSpec(las_file.version.vers).is_las12
    lines.append("~WELL INFORMATION")

    # Key-type validation — defense-in-depth (F-M-008 covers model layer).
    # This protects the writer when the model layer is bypassed.
    for key in las_file.well.entries:
        if not isinstance(key, str):
            raise TypeError(
                f"WellSection entry key must be str, got {type(key).__name__}: {key!r}"
            )

    # Mandatory well field checks (missing, STEP=0, NULL empty, STRT==STOP)
    # are now centralized in WellSection.validate(complete=True), called at
    # the start of _generate_las_content before any section writers run.

    # Collect STRT/STOP for later comparison.

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
        # Accumulate curve definitions from ALL LOG_DATA sections.
        # F-209: next() selected only the first LOG_DATA section, silently
        # dropping curves from subsequent LOG_DATA sections.  Iterate all
        # sections so multi-section files preserve their ~CURVE metadata.
        # F2-006: Normalize section_type to uppercase — from_dict does not
        # normalize, so programmatically constructed files may have lowercase.
        curves_to_emit: list[CurveDefinition] = []
        for ds in las_file.data_sections:
            if (ds.section_type or "LOG_DATA").upper() == "LOG_DATA":
                if ds.section_curves:
                    curves_to_emit.extend(ds.section_curves)
        # E-F-007: Secondary scan for LOG_DATA sections that have
        # curves_order but no section_curves (programmatic construction).
        # Collect their CurveDefinitions from las_file.curves so ~CURVE
        # declares the correct complete curve set.
        emitted_mnems = {c.mnemonic for c in curves_to_emit}
        curves_by_mnem = {c.mnemonic: c for c in las_file.curves}
        for ds in las_file.data_sections:
            if (ds.section_type or "LOG_DATA").upper() == "LOG_DATA":
                if not ds.section_curves and ds.curves_order:
                    for mnem in ds.curves_order:
                        if mnem not in emitted_mnems:
                            curve_def = curves_by_mnem.get(mnem)
                            if curve_def is not None:
                                curves_to_emit.append(curve_def)
                                emitted_mnems.add(mnem)

        if not curves_to_emit:
            curves_to_emit = list(las_file.curves)
        if not curves_to_emit:
            import warnings

            warnings.warn(
                "No curves to emit for ~C section — skipping",
                UserWarning,
                stacklevel=3,
            )
            lines.append("")
            return lines
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
        # I2F-19: Escape colons in the unit field.  The unit appears between
        # the mnemonic and the value in the output (MNEM.UNIT  VALUE  : DESC),
        # BEFORE the structural colon.  A unit containing " : " creates a
        # spurious structural separator that misroutes parser split results.
        unit = _escape_colons_for_las_value(unit) if unit else ""
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
            # F-047: Normalize section_type to uppercase for case-insensitive
            # grouping, matching how DataSection.section_type is handled at
            # lines 455, 616, and 734.  Without this, "CORE" and "core" become
            # separate dict entries producing duplicate parameter section blocks.
            # F-I2-XWM-01: Strip pipe character from section_type.  Pipe (|)
            # is a legitimate LAS 3.0 structural zone-notation character but
            # MUST NOT appear in section type names — an embedded pipe in a
            # section header creates ambiguous pipe targets that corrupt
            # parser interpretation on re-read.  Replace with underscore.
            st_key = (
                param.section_type.upper().replace("|", "_")
                if param.section_type and param.section_type.strip() else None
            )
            sections.setdefault(st_key, []).append(param)

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
        # Strip pipe characters after sanitization.  Pipe (|) is
        # intentionally excluded from _CONTROL_CHARS_RE because it is
        # legitimate LAS 3.0 zone notation ("| RUN[1]"), but it MUST
        # NOT appear in section type names — an embedded pipe in the
        # section header creates ambiguous pipe targets that corrupt
        # parser interpretation on re-read (e.g., "~MY|PIPE_DATA | CURVE"
        # is parsed as section ~MY with a bogus pipe target).
        return _sanitize_las_value(section_type).replace("|", "")
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
    # F-210: Escape colons in the unit field.  The unit appears between
    # the mnemonic and the description (MNEM.UNIT  : DESC), BEFORE the
    # structural colon.  A unit containing " : " creates a spurious
    # structural separator that misroutes parser split results.
    # Same pattern used in _write_parameter_section and _write_well_section.
    unit = _escape_colons_for_las_value(unit) if unit else ""
    desc = curve.description if curve.description else ""

    if is_las30 and curve.data_format:
        format_str = f"{{{curve.data_format}"
        if curve.data_format == "A" and curve.array_info and curve.array_info.time_offset is not None:
            offset = curve.array_info.time_offset
            # F-024: Guard against non-finite time_offsets (inf, nan) that
            # cause OverflowError in int(offset) or produce invalid LAS values.
            if math.isfinite(offset):
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
    spec = _LASVersionSpec(las_file.version.vers)
    import warnings  # consolidated — shared by all warning paths below

    # F-02-H: Save original DLM before the LAS 1.2 SPACE-override
    # mutation below.  Restored in the finally block so the caller's
    # LASFile DLM is unchanged after writing.
    _saved_dlm = las_file.version.dlm

    # F-04: LAS 1.2 only supports SPACE delimiter per spec.  The header
    # correctly suppresses non-SPACE DLM emission for LAS 1.2 (see
    # _write_version_section L172-176) but data rows used the non-SPACE
    # delimiter from delimiter_char, creating a header/data mismatch
    # that would corrupt roundtrip re-reads.  Force SPACE and warn.
    if spec.is_las12 and delimiter != " ":
        warnings.warn(
            f"LAS 1.2 does not support the '{las_file.version.dlm}' delimiter. "
            "Forcing SPACE delimiter for data rows to match the header section.",
            stacklevel=3,
        )
        delimiter = " "
        # M-11: Update the model's DLM to reflect what was actually
        # written to disk.  Without this, a subsequent to_dict() or
        # re-write would report the original (non-SPACE) delimiter,
        # creating a model-disk inconsistency.
        las_file.version.dlm = "SPACE"

    # R8-007: Save original values before data_sections copy-back mutates the model.
    # The legacy fallback path below (lines ~846-857) copies data_sections[0] contents
    # to las_file.logs, string_data, curves_order, and curves.  These are bare
    # mutations — restored here so the caller's LASFile is unchanged after writing.
    _saved_logs = dict(las_file.logs)
    _saved_string_data = dict(las_file.string_data)
    _saved_curves_order = list(las_file.curves_order)
    _saved_curves = list(las_file.curves)

    # F-03-H (G-018): Mutate WRAP for model-disk consistency.  The writer
    # always produces WRAP=NO output regardless of input wrap value.
    # _write_version_section already emitted the correct WRAP=NO header
    # line; this mutation syncs the in-memory model with what was written
    # to disk.  Unlike DLM (restored in finally because the SPACE override
    # is a writer-internal FORCE), WRAP is NOT restored — the model must
    # honestly reflect the actual disk state, per F2-26 / G-018.
    _actual_wrap = (las_file.version.wrap or "NO").upper()
    if _actual_wrap == "YES" or (spec.is_las30 and _actual_wrap != "NO"):
        las_file.version.wrap = "NO"

    # F-28 / M-06: Compute line-limit check AFTER WRAP mutation.
    # The writer always produces WRAP=NO output regardless of input
    # (LAS 2.0 WRAP=YES is overridden, LAS 3.0 requires WRAP=NO).
    # The 256-char line-limit warning must use the actual output state,
    # so compute check_line_limit from the post-mutation wrap value.
    check_line_limit = spec.line_length_limit_for_wrap(
        las_file.version.wrap
    ) is not None

    try:
        # F-I2-M17 / F-019: Guard against data_sections in non-LAS-3.0 files.
        # from_dict populates data_sections from the input dict unconditionally
        # (regardless of LAS version).  Multiple sections on a non-LAS-3.0 file
        # cause guaranteed roundtrip data loss (parser skips all data for
        # non-LAS-3.0 settings).  A single section can safely fall back to
        # legacy ~A format.
        if las_file.data_sections and not spec.is_las30:
            if len(las_file.data_sections) > 1:
                raise LASWriteError(
                    f"Multiple data_sections ({len(las_file.data_sections)}) "
                    f"are only supported for LAS 3.0 files, but version is "
                    f"{las_file.version.vers!r}. Cannot safely write multi-section "
                    f"data for non-LAS-3.0 format."
                )
            warnings.warn(
                "data_sections are only supported for LAS 3.0 files. "
                "Falling back to single-section ~A format. "
                "Single-section data will be preserved.",
                stacklevel=3,
            )

            # F-067/F-111: Copy data_sections[0] contents to legacy attributes so
            # the legacy writer path at line 868+ reads the correct data.  Without
            # this copy-back, the single-section warning's promise ("data will be
            # preserved") is broken — the legacy path sees empty logs/string_data/
            # curves_order.  Use .update() for logs/string_data to preserve any
            # metadata entries (STRT, STOP, STEP, etc.) the user may have set.
            # Only copy if legacy attributes are not already populated — respect
            # user data.  curves_order is a list, so assignment is correct.
            # Each attribute has its own destination guard (not just logs) so
            # that pre-populated attributes are not silently overwritten.
            _ds = las_file.data_sections[0]
            if not las_file.logs and _ds.data:
                las_file.logs.update(_ds.data)
            if not las_file.string_data and _ds.string_data:
                las_file.string_data.update(_ds.string_data)
            if not las_file.curves_order and _ds.curves_order:
                las_file.curves_order = list(_ds.curves_order)
            # F-ITER2-W-R07: Also copy per-section curve definitions
            # (units, descriptions, API codes, data_formats) to the
            # top-level curves list so the legacy path preserves metadata.
            if not las_file.curves and _ds.section_curves:
                las_file.curves = list(_ds.section_curves)

            # F-014: Sync curves_order when logs was populated from
            # data_sections with extra keys not in pre-populated curves_order.
            # Without this sync, extra log keys have no corresponding order
            # entry and are silently dropped during write.
            if (las_file.curves_order and _ds.curves_order
                    and las_file.logs):
                existing = set(las_file.curves_order)
                for k in _ds.curves_order:
                    if k in las_file.logs and k not in existing:
                        las_file.curves_order.append(k)

            # F-R-05: Check for curves/curves_order mismatch after copy-back.
            # Independent per-attribute guards can produce len(curves) !=
            # len(curves_order) when one is pre-populated but the other is
            # copied from data_sections.
            if (
                las_file.curves
                and las_file.curves_order
                and len(las_file.curves) != len(las_file.curves_order)
            ):
                raise LASDataError(
                    f"curves count ({len(las_file.curves)}) does not match "
                    f"curves_order count ({len(las_file.curves_order)}) "
                    f"after copy-back. This indicates inconsistent "
                    f"LASFile construction."
                )

            # F-063: Check for uncovered curves after legacy copy-back.
            # Independent per-attribute guards (lines 838-848) can produce
            # curves_order entries without corresponding data in logs or
            # string_data.  The writer pads uncovered curves with null_value
            # silently — warn so the caller knows data was not propagated.
            if las_file.curves_order and (las_file.logs or las_file.string_data):
                _log_keys = set(las_file.logs.keys()) if las_file.logs else set()
                _str_keys = (
                    set(las_file.string_data.keys()) if las_file.string_data
                    else set()
                )
                _order_set = set(las_file.curves_order)
                _uncovered = _order_set - _log_keys - _str_keys
                if _uncovered:
                    warnings.warn(
                        f"Curve(s) {sorted(_uncovered)} appear in "
                        f"curves_order but have no data in 'logs' or "
                        f"'string_data' after copy-back.  The writer will "
                        f"pad these curves with null_value.",
                        stacklevel=3,
                    )

        if las_file.data_sections and spec.is_las30:
            # F-ITER2-W-M01 / F-ITER2-W-R08: The LAS 3.0 writer path only
            # iterates data_sections and never reads las_file.logs.  If
            # las_file.logs contains curve data not covered by any
            # data_section's data or string_data, that data is silently
            # discarded on write.  Emit a warning so the caller knows.
            if las_file.logs or las_file.string_data:
                # Collect all curve names present in any data_section.
                _ds_covered: set[str] = set()
                for _ds in las_file.data_sections:
                    _ds_covered.update(_ds.data.keys())
                    _ds_covered.update(_ds.string_data.keys())
                _orphaned_logs = (
                    set(las_file.logs.keys()) - _ds_covered
                    if las_file.logs else set()
                )
                if _orphaned_logs:
                    warnings.warn(
                        f"Top-level logs contain curve(s) not present in any "
                        f"data_section: {sorted(_orphaned_logs)}.  The LAS 3.0 "
                        f"writer path only writes data from data_sections; "
                        f"these curves' data will NOT appear in the output file.",
                        stacklevel=3,
                    )
                _orphaned_string_data = (
                    set(las_file.string_data.keys()) - _ds_covered
                    if las_file.string_data else set()
                )
                if _orphaned_string_data:
                    warnings.warn(
                        f"Top-level string_data contains curve(s) not present in any "
                        f"data_section: {sorted(_orphaned_string_data)}.  The LAS 3.0 "
                        f"writer path only writes data from data_sections; "
                        f"these curves' data will NOT appear in the output file.",
                        stacklevel=3,
                    )
            # LAS 3.0: Multiple data sections with typed headers.
            # F-042: emitted_defs maps def_prefix -> {curve_signature -> def_section_name}.
            # Two sections of the same type with IDENTICAL curve definitions share
            # one Definition block (dedup).  When curve definitions differ between
            # same-type sections, a new numbered Definition block is emitted.
            emitted_defs: dict[str, dict[tuple[tuple[str, str, str, str, str, float | None], ...], str]] = {}
            for section in las_file.data_sections:
                # F2-006: Normalize section_type to uppercase — from_dict does
                # not normalize, so programmatically constructed LASFile objects
                # may have lowercase section_type values (e.g., "log_data").
                # Normalize once at the top of the loop for all comparisons.
                sec_type = (section.section_type or "LOG_DATA").upper()
                section_prefix = _section_type_to_prefix(sec_type)
                # I2F-18: Strip pipe characters from section names before
                # constructing the pipe-delimited header.  A pipe in the
                # section name (e.g., " | CURVE") would create a spurious
                # pipe target, causing the parser to locate the wrong
                # definition on re-read.  Pipe is NOT in _CONTROL_CHARS_RE
                # because it is a legitimate LAS 3.0 zone notation character
                # ("| RUN[1]") in value/description fields.
                raw_section_name = (
                    f" {_sanitize_las_value(section.name).replace('|', '')}"
                    if section.name else ""
                )
                section_name = raw_section_name

                # F-16: Compute definition prefix for non-LOG_DATA LAS 3.0
                # sections.  Used for both the _Definition section header and
                # the pipe notation on the data section header so the parser
                # can re-associate per-section curves on re-read.
                def_prefix: str | None = None
                if spec.is_las30 and sec_type != "LOG_DATA":
                    def_prefix = _SECTION_TYPE_TO_DEFINITION_PREFIX.get(sec_type)
                    if def_prefix is None:
                        if sec_type.endswith("_DATA"):
                            # F-D3-M01: Auto-derive Definition prefix for user-defined
                            # _DATA section types not in the hardcoded mapping.
                            # Strip _DATA suffix and title-case the root
                            # (e.g., "CUSTOM_DATA" → "Custom").
                            root = sec_type[: -len("_DATA")]
                            root = _sanitize_las_value(root)
                            def_prefix = root.title().replace("_", "")
                        elif section.section_curves:
                            # I2F-21: Unknown non-_DATA section type with per-section
                            # curves.  Derive a definition prefix from the section type
                            # name so that curve metadata is preserved on roundtrip.
                            # Without this, unknown section types silently lose their
                            # per-section curve definitions.
                            import warnings

                            warnings.warn(
                                f"Unknown section type '{sec_type}' has per-section "
                                f"curve definitions.  Deriving definition prefix from "
                                f"section type name to preserve curve metadata.",
                                stacklevel=3,
                            )
                            st = _sanitize_las_value(sec_type)
                            def_prefix = st.title().replace("_", "")

                # For non-LOG_DATA sections: emit per-section Definition section
                # so that the parser can correctly re-associate per-section curve
                # names on re-read.  Without this, all data sections get the
                # global curve set on roundtrip.
                # F-042: Dedup by curve identity, not just prefix.
                # Only skip definitions when another section of the same type
                # has IDENTICAL curve definitions (same mnemonics, units,
                # descriptions, data_formats).  Different curve sets on the
                # same section type get separate numbered Definition blocks
                # (e.g., Core_Definition, Core_Definition_2).
                pipe_def_name: str | None = None
                if spec.is_las30 and sec_type != "LOG_DATA" and section.section_curves:
                    # Build a curve identity signature from the visible fields
                    # that distinguish one curve set from another.
                    sig = tuple(
                        (curve.mnemonic, curve.unit or "", curve.description or "", curve.data_format or "",
                         curve.api_code or "",
                         curve.array_info.time_offset if curve.array_info else None)
                        for curve in section.section_curves
                    )
                    if def_prefix:
                        if def_prefix not in emitted_defs:
                            emitted_defs[def_prefix] = {}
                        sig_map = emitted_defs[def_prefix]
                        if sig not in sig_map:
                            emit_idx = len(sig_map) + 1
                            def_section_name = (
                                f"{def_prefix}_Definition"
                                if emit_idx == 1
                                else f"{def_prefix}_Definition_{emit_idx}"
                            )
                            sig_map[sig] = def_section_name
                            lines.append(f"~{def_section_name}")
                            for curve in section.section_curves:
                                lines.append(_format_curve_line(curve, spec.is_las30))
                            lines.append("")  # blank line after definition
                        pipe_def_name = sig_map[sig]

                # Data section header with pipe notation for curve association.
                # LOG_DATA sections reference "| CURVE" so the parser scopes
                # curves to only the global ~CURVE set on re-read.  Non-LOG_DATA
                # sections reference "| {Name}_Definition" for per-section curve
                # reassociation (F-16).
                # F-010/F-042: Only emit the pipe reference when the corresponding
                # Definition section was actually written above (tracked via
                # pipe_def_name).  Without this guard, a section with a valid
                # def_prefix but empty section_curves would emit a phantom
                # pipe reference to a non-existent Definition section.
                if sec_type == "LOG_DATA" and spec.is_las30:
                    lines.append(f"~{section_prefix}{section_name} | CURVE")
                elif spec.is_las30 and sec_type != "LOG_DATA" and pipe_def_name:
                    lines.append(
                        f"~{section_prefix}{section_name} | {pipe_def_name}"
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
                        is_las12=check_line_limit,
                    )
                )
        else:
            # Legacy single data section (~A).
            curve_names = las_file.curves_order
            if curve_names and not any(
                name in las_file.logs or name in las_file.string_data for name in curve_names
            ):
                import warnings

                warnings.warn(
                    f"curves_order contains {len(curve_names)} curve(s) "
                    f"but none have data in logs or string_data. "
                    f"No data will be emitted.",
                    stacklevel=3,
                )
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
                        is_las12=check_line_limit,
                    )
                )

    finally:
        # R8-007: Restore original model state after copy-back write completes.
        # The save above captured the pre-mutation values of las_file.logs,
        # string_data, curves_order, and curves.  Restoring them ensures the
        # caller's LASFile is unchanged regardless of which write path executed.
        # WRAP is intentionally NOT restored — the model must honestly reflect
        # the actual WRAP=NO state written to disk (F2-26 / G-018 / H-01).
        las_file.logs = _saved_logs
        las_file.string_data = _saved_string_data
        las_file.curves_order = _saved_curves_order
        las_file.curves = _saved_curves
        las_file.version.dlm = _saved_dlm

    return lines


def _format_data_rows(
    curve_names: list[str],
    data: dict[str, NDArray[np.float64]],
    string_data: dict[str, NDArray[np.object_]],
    null_value: float,
    delimiter: str,
    precision: str = ".8g",
    is_las12: bool = False,
) -> list[str]:
    """Format data rows for a section — handles both legacy and LAS 3.0 sections.

    Builds one line per depth step with delimiter-separated values.
    String curves are emitted as-is; numeric curves use configurable formatting.
    Missing values are filled with the null_value. NaN values are output as null.

    When *is_las12* is True, warns if any data row exceeds the 256-character
    line limit (applies to LAS 1.2 and LAS 2.0 WRAP=NO per CWLS spec).
    Lines are NOT truncated — truncation would cause data loss.
    """
    lines: list[str] = []
    if not curve_names:
        return lines

    # Pre-extract curve data arrays to avoid O(rows x curves) dict lookups
    # inside the inner loop (F-23 performance optimization).
    curve_arrays: list[tuple[NDArray[np.float64] | NDArray[np.object_] | None, bool]] = []
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
    warned_empty_str = False  # Deduplicate empty→"-" substitution warnings
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
                        # F-39 (F-I2-XWP-04): Whitespace->underscore replacement
                        # for SPACE-delimiter string data.  re.sub(r"\s", "_", val)
                        # replaces all Unicode whitespace with underscores to
                        # prevent column misalignment on re-read (str.split()
                        # would split on the whitespace, creating phantom fields).
                        #
                        # WRITER->PARSER CONTRACT: Whitespace->underscore is
                        # permanently one-way - the parser has no mechanism to
                        # recover the original whitespace characters from
                        # underscores.  An underscore in the output may be an
                        # original underscore or originally whitespace.
                        # PARSER FIX NEEDED (in _desanitize_las_value, parser.py):
                        #   Exact reversal is inherently ambiguous, but for
                        #   roundtrip best-effort, add a sentinel-based encoding:
                        #   - Writer: replace each whitespace char with a
                        #     reversible sentinel (e.g., spaces -> "_S_",
                        #     non-breaking spaces -> "_N_", tabs -> "_T_")
                        #     instead of the blanket re.sub(r"\s", "_", val).
                        #   - Parser: reverse each sentinel back to its
                        #     original whitespace character.
                        #   OR: Accept this as a documented one-way
                        #   transformation when using SPACE delimiter.
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
                # I2-F-03 (F-21/F-I2-XWP-03): Empty string after sanitization
                # causes column misalignment with SPACE delimiter.
                # " ".join(["a", "", "b"]) produces "a  b" (two consecutive
                # spaces).  On re-read, str.split() collapses to ["a", "b"] -
                # one column permanently lost.  Replace with "-" (traditional
                # LAS placeholder for missing data).  COMMA/TAB delimiters
                # are unaffected - empty fields survive split(",").
                #
                # WRITER->PARSER CONTRACT: The "-" sentinel is a permanently
                # one-way transformation - the parser cannot distinguish an
                # original "-" value from an originally-empty one.
                # PARSER FIX NEEDED (in _desanitize_las_value, parser.py):
                #   - When delimiter is SPACE and the parsed value is
                #     exactly "-", restore it to the empty string "".
                #   - This will incorrectly convert genuinely "-" values
                #     to empty strings (inherent ambiguity).
                #   - Alternative: use a more distinctive sentinel that
                #     is less likely to be real data, e.g., "_EMPTY_".
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
    sig_digits = min(int(m.group(1)), 100) if m else 8

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
