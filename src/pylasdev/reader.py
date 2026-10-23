"""LAS file reader — main entry point.

Replaces las_reader.py with modern Python 3, proper encoding handling,
context managers, and no global state.
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from ._sanitize import normalize_nbsp_family
from .data_reader import read_ascii_data
from .encoding import read_with_encoding
from .exceptions import LASParseError, LASReadError
from .models import LASFile
from .parser import LASParser

logger = logging.getLogger(__name__)

# F-32 + G-17 + F-I2-M04 + F-I2-M05 + F-225 + F-231 (unit U6) + LD-08:
# characters in file content that must be neutralized before splitlines() to
# prevent section-header injection and silent data corruption.  The pre-fix
# code used ONE class for two different intents and substituted a space for
# both, so a character inside a numeric token split the token silently:
# '10\x0000.0' parsed as 10.0 / 0.0 and '1\xa0000.5' as 1.0 / 0.0, with at
# best a generic "extra columns" warning.  The policy is now:
#   * the non-breaking space family (NBSP U+00A0, THIN SPACE U+2009, NARROW
#     NBSP U+202F) is normalized by _sanitize.normalize_nbsp_family with the
#     ONE rule every entry point shares (LD-08): a family character INSIDE
#     one numeric token — a token-leading 1-3 digit group, then groups of
#     exactly three digits ('1\xa0000.5' -> '1000.5') — is DELETED — a space
#     would split one value into two columns and shift every later value on
#     the row (F-225); every other occurrence becomes a SPACE — deleting it
#     would fuse two genuinely separate values ('1.0\xa02.0' -> '1.02.0',
#     F-231).  The delete side is position-scoped precisely because a blanket
#     "delete between two digits" fused values, and it additionally requires
#     the 1-3 digit token-leading left group so a genuine column separator is
#     never fused ('1000\xa0500.0' is 1000 and 500.0 — M9);
#   * every character str.splitlines() treats as a line break beyond LF/CR
#     (VT, FF, FS/GS/RS, NEL, LINE/PARAGRAPH SEPARATOR) becomes a SPACE — a
#     fake line boundary must never forge a section header;
#   * every other Unicode space the writer's _UNICODE_WS_RE normalizes
#     (U+2000-U+200A other than U+2009, U+205F, U+3000) becomes a SPACE —
#     deleting them would concatenate adjacent tokens (F-05:
#     'ABC\xa0DEF' -> 'ABCDEF');
#   * every other C0 control (including NUL) and DEL is DELETED, mirroring
#     the writer's _CONTROL_CHARS_RE, and a counted UserWarning names the
#     affected line — a padding character is not a delimiter, so it must not
#     split a token, and it must not vanish silently either (F-231).
# Tab (\x09) is preserved: it is legitimate LAS whitespace.
# The space substitution for non-numeric text is silent: NBSP is not a
# control character, and tests/test_regression.py:94-116 pins NBSP-as-space
# for non-numeric text (F-001).  A DELETION — an NBSP-used-as-padding inside
# one numeric token — is NOT silent: normalize_nbsp_family counts it and
# names the fused token (M9(b)).  The NBSP family members stay listed in
# _SPACE_CHARS_RE as a fallback; normalize_nbsp_family runs first, so they
# are already gone by the time this class is applied.
_SPACE_CHARS_RE = re.compile(
    r"[\x0B\x0C\x1C-\x1E\x85"
    r"\u2028\u2029"
    r"\u00A0\u2000-\u200A\u202F\u205F\u3000]"
)
_DELETE_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0E-\x1B\x1F\x7F]")

# Line boundaries under str.splitlines() semantics, used only to report the
# 1-based line of the first deleted control character.
_LINE_BREAK_RE = re.compile(r"\r\n|[\n\r\x0B\x0C\x1C\x1D\x1E\x85\u2028\u2029]")


def _sanitize_control_chars(content: str, file_path: Path) -> str:
    """Apply the control-character policy to raw file content.

    Returns the content with the non-breaking space family normalized by the
    shared rule (thousands grouping separator deleted, every other family
    character one space), line-break characters and Unicode spaces replaced
    by spaces, and non-line-break C0 controls / DEL deleted.  Two counted
    UserWarnings report the destructive steps: the NBSP-family deletion names
    the fused token(s) (M9(b)), and the control-character deletion reports
    its count, first code point and first affected line — so a padding
    character inside a token is never silent corruption (F-225/F-231).
    """
    # LD-08: ONE NBSP-family rule, shared with the writer, parser and DEV
    # reader — a separator INSIDE one numeric token is deleted, everything
    # else becomes a space.  M9(b): the deletion is counted and the fused
    # token is named, so a value the normalizer joined is never silent.
    content = normalize_nbsp_family(content, warn_context=str(file_path), stacklevel=3)
    content = _SPACE_CHARS_RE.sub(" ", content)
    first_control = _DELETE_CONTROL_CHARS_RE.search(content)
    if first_control is None:
        return content
    first_char = first_control.group()
    # The space substitution above is 1:1, and the NBSP-family rule only
    # deletes a grouping separator (a digit-adjacent character, never a
    # line-break character), so the line number is accurate.
    first_line = len(_LINE_BREAK_RE.findall(content[: first_control.start()])) + 1
    content, n_deleted = _DELETE_CONTROL_CHARS_RE.subn("", content)
    warnings.warn(
        f"Removed {n_deleted} control character(s) from {file_path} before "
        f"parsing (first: U+{ord(first_char):04X} at line {first_line}). "
        "Control characters that are not line breaks are deleted — the LAS "
        "writer deletes the same character class — so a padding character "
        "can never split a value into two columns. Verify the file is not "
        "corrupt.",
        stacklevel=3,
    )
    return content


def read_las_file(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
    max_file_size: int | None = None,
    well_format: str = "auto",
    desanitize: bool = True,
) -> dict[str, Any]:
    """Read a LAS file and return data dictionary.

    This is the main entry point, maintaining backward compatibility
    with the original pylasdev API (returns dict, not LASFile).

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override. If None, auto-detected.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a LASReadError is raised.
        well_format: LAS 1.2 well section format convention:
            ``"auto"`` (default) heuristically detects CWLS vs lasio
            convention per field; ``"cwls"`` forces CWLS convention;
            ``"lasio"`` forces lasio convention.
        desanitize: If True (default), strip writer escape prefixes
            (``_#`` → ``#``) for roundtrip correctness.  Set to False
            when reading files NOT produced by pylasdev's writer to
            avoid corrupting ``_#``-prefixed data.

    Returns:
        Dictionary with keys: version, well, parameters, parameter_details,
        curves, logs, curves_order, other, data_sections, string_data, encoding.
        Well values are strings. Log values are numpy arrays.

    Raises:
        LASReadError: If file cannot be found, is not a regular file, or
            exceeds max_file_size.
        LASEncodingError: If the file content cannot be decoded with the
            detected or explicitly requested encoding.
        LASParseError: If file content cannot be parsed (e.g. missing
            required ~V section).
        MemoryError: If the system runs out of memory during
            conversion to dict.

    Warns:
        UserWarning: If LAS version is > 3.0 (unsupported but attempted).
        UserWarning: If a non-breaking space used as thousands padding was
            removed from inside a numeric token (counted, naming the fused
            token) — see M9/F-225.
        UserWarning: If control characters that are not line breaks were
            removed from the file content (counted, with the first
            affected line) — see F-225/F-231.

    Example:
        >>> data = read_las_file("sample.las")
        >>> print(data["well"]["WELL"])
        >>> print(data["logs"]["DEPT"])
    """
    las_file = read_las_file_as_object(
        file_path,
        mnem_base=mnem_base,
        encoding=encoding,
        max_file_size=max_file_size,
        well_format=well_format,
        desanitize=desanitize,
    )
    return las_file.to_dict()


def read_las_file_as_object(
    file_path: str | Path,
    mnem_base: dict[str, str] | None = None,
    encoding: str | None = None,
    max_file_size: int | None = None,
    well_format: str = "auto",
    desanitize: bool = True,
) -> LASFile:
    """Read a LAS file and return LASFile dataclass (new API).

    Same as read_las_file() but returns the LASFile object directly
    instead of converting to dict. Use this for richer metadata access.

    Args:
        file_path: Path to LAS file.
        mnem_base: Optional dictionary for curve name normalization.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a LASReadError is raised.
        well_format: LAS 1.2 well section format convention:
            ``"auto"`` (default) heuristically detects CWLS vs lasio
            convention per field; ``"cwls"`` forces CWLS convention;
            ``"lasio"`` forces lasio convention.
        desanitize: If True (default), strip writer escape prefixes
            (``_#`` → ``#``) for roundtrip correctness.  Set to False
            when reading files NOT produced by pylasdev's writer to
            avoid corrupting ``_#``-prefixed data.

    Returns:
        LASFile dataclass with full parsed data.

    Raises:
        LASReadError: If file cannot be found, is not a regular file, or
            exceeds max_file_size.
        LASEncodingError: If the file content cannot be decoded with the
            detected or explicitly requested encoding.
        LASParseError: If file content cannot be parsed (e.g. missing
            required ~V section).

    Warns:
        UserWarning: If LAS version is > 3.0 (unsupported but attempted).
        UserWarning: If a non-breaking space used as thousands padding was
            removed from inside a numeric token (counted, naming the fused
            token) — see M9/F-225.
        UserWarning: If control characters that are not line breaks were
            removed from the file content (counted, with the first
            affected line) — see F-225/F-231.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LASReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise LASReadError(f"Not a file: {file_path}")

    try:
        detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)
    except OSError as e:
        raise LASReadError(f"Cannot read file (I/O error): {file_path}") from e
    except ValueError as e:
        # ENC-03: max_file_size violations surface as ValueError from
        # read_with_encoding (encoding.py:441-444); re-raise as LASReadError
        # per the documented reader contract (README error section).
        # LookupError is intentionally NOT in this tuple — encoding.py wraps
        # invalid codec names into LASEncodingError (F-94), so it is
        # unreachable here; LASEncodingError itself propagates untouched so
        # callers get the accurate decoding message instead of a misleading
        # "size exceeded" LASReadError (ENC-03).
        raise LASReadError(
            f"Cannot read file (size exceeded or invalid parameter): {file_path}"
        ) from e

    parser = LASParser(mnem_base, well_format=well_format)
    # PERF-01: Split content once, pass lines list to both parser and
    # data_reader to eliminate redundant content.splitlines() calls
    # (double splitlines doubles peak memory for large files).
    # F-32 + G-17 + F-225/F-231 (unit U6) + LD-08: normalize the NBSP family
    # (and sanitize line-break characters) BEFORE splitting (section-header
    # injection), and delete the remaining control characters (a padding
    # char must not split a token); see _sanitize_control_chars.
    lines = _sanitize_control_chars(content, file_path).splitlines()
    parser.source_file = str(file_path)
    # F-047: Thread the desanitize parameter through to header parsing.
    # data_reader sets this flag before processing data, but header
    # parsing (well, curve, parameter, version) runs earlier — without
    # this, all 8 header desanitization call sites always run with the
    # default True, ignoring the caller's desanitize parameter.
    parser_mod = sys.modules["pylasdev.parser"]
    # E-04: Save the prior thread-local value and restore it in a finally
    # block.  The flag is thread-local (parser.py F-21/F-088); an
    # unconditional reset-to-True would clobber another caller's value on
    # the same thread.  Without the restore, a desanitize=False read left
    # the flag False for the whole thread, silently changing the behavior
    # of subsequent direct LASParser.parse()/read_ascii_data() users.
    _prev_desanitize = parser_mod._DESANITIZE_ENABLED
    parser_mod._DESANITIZE_ENABLED = desanitize  # type: ignore[attr-defined]
    try:
        las_file = parser.parse(content, lines=lines)
    except LASParseError as e:
        raise LASParseError(f"Error reading {file_path}: {e}") from e
    finally:
        parser_mod._DESANITIZE_ENABLED = _prev_desanitize  # type: ignore[attr-defined]
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
        # Non-numeric version string (e.g. "CWLS LOG ASCII STANDARD").
        # This is common in LAS 1.2 files and files with plain-text version
        # identifiers. Silently accept these — they cannot be compared via
        # float() but the file is still parseable.
        logger.debug("Non-numeric version string encountered: %s", las_file.version.vers)

    # Read ASCII data section
    # For LAS 3.0, the parser already handles this
    # For LAS 1.2/2.0, use the dedicated data reader
    if not las_file.is_las30:
        try:
            read_ascii_data(lines, las_file, parser.data_line_count, desanitize=desanitize)
        except LASParseError as e:
            raise LASParseError(f"Error reading {file_path}: {e}") from e

    return las_file
