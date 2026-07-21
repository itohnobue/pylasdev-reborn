"""Shared data section reading utilities extracted from data_reader.py.

Contains section-detection primitives (moved from data_reader.py) and
shared reader logic used by _read_normal, _read_wrapped, and _detect_actual_wrap.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LASFile

# Late import to work with circular imports — data_reader imports from this
# module, and we need _resolve_max_tokens_per_line from data_reader.
# Both are defined before the cross-import, so this is safe at import time.
from .data_reader import _resolve_max_tokens_per_line

# ---------------------------------------------------------------------------
# Section-detection primitives (moved from data_reader.py)
# ---------------------------------------------------------------------------

# F-ITER2-D2-M04: Regex to extract the full section word from a header line
# (matching parser.SECTION_PATTERN semantics).  Used to compare against
# exact section words {"A", "ASCII"} instead of checking only the first
# character after ~, which caused divergence when headers like ~A_DEFINITION
# entered ASCII mode in the reader but were excluded by parser._pre_scan.
_SECTION_WORD_RE = re.compile(r"^~([A-Za-z]\S*)")


def _get_section_word(stripped: str) -> str:
    """Extract the section word from a section header line.

    Returns the uppercased section word (e.g. "A", "ASCII", "A_DEFINITION")
    or an empty string if the line is not a valid section header.
    """
    match = _SECTION_WORD_RE.match(stripped)
    return match.group(1).upper() if match else ""


def _is_ascii_section(stripped: str) -> bool:
    """Check if a section header targets the ASCII data (~A / ~ASCII) section.

    Aligned with parser._pre_scan which uses ``section_word in {"A", "ASCII"}``.
    """
    return _get_section_word(stripped) in {"A", "ASCII"}


def _is_section_header(stripped: str) -> bool:
    """Check if a stripped line is a section header (~[A-Za-z]).

    Uses ASCII-only character check to match the parser's SECTION_PATTERN,
    which is limited to [A-Za-z].  The parser upper-cases section letters,
    so matching must stay within ASCII.
    """
    return (
        stripped.startswith("~")
        and len(stripped) > 1
        and stripped[1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )


# F-I2-XPD-01: Known LAS section words for section-header injection defense.
# SPLITLINES_CHARS_RE in reader.py replaces null bytes and control characters
# with spaces before splitlines(), which can create spurious section headers
# (null byte followed by ~SectionWord becomes a standalone ~SectionWord line
# after splitting).  Validating section words against this known set before
# breaking data reading prevents premature termination caused by injected
# artifacts.  Aligned with parser.py's known section-type dispatch tables.
_KNOWN_SECTION_WORDS: frozenset[str] = frozenset({
    # Data section words (from parser._DATA_SECTION_WORDS)
    "A", "ASCII",
    "CORE", "CORE_DATA",
    "DRILLING", "DRILLING_DATA",
    "FORMATION", "FORMATION_DATA",
    "INCLINOMETRY", "INCLINOMETRY_DATA",
    "LOG", "LOG_DATA",
    "MUD", "MUD_DATA",
    "PERFORATIONS", "PERFORATIONS_DATA",
    "RISK", "RISK_DATA",
    "STRUCTURE", "STRUCTURE_DATA",
    "TEST", "TEST_DATA",
    "TOPS", "TOPS_DATA",
    # Non-data section words (from parser dispatch table)
    "C", "CURVE",
    "D", "DEFINITION",
    "O", "OTHER",
    "P", "PARAMETER", "PARAMETERS",
    "V", "VERSION",
    "W", "WELL",
})


def _is_recognized_section_word(word: str) -> bool:
    """Check if a section word is a recognized LAS section header type.

    Validates that a potential section header is a genuine LAS section
    keyword, not an artifact created by control-character replacement
    (SPLITLINES_CHARS_RE in reader.py).  Used by _read_normal and
    _read_wrapped as defense-in-depth against section-header injection.

    Args:
        word: Uppercased section word extracted by _get_section_word
            (e.g. "A", "CORE_DATA", "VERSION").

    Returns:
        True if the word is a recognized LAS section type.
    """
    if not word:
        return False
    # Strip index brackets (e.g., CORE[1] → CORE) before checking.
    base = word.split("[", 1)[0] if "[" in word else word
    if base in _KNOWN_SECTION_WORDS:
        return True
    # Recognized suffix patterns: _DEFINITION (including numbered variants like
    # _DEFINITION_2), _PARAMETER, _PARAMETERS, _DATA.
    # These expand the known set to cover all parser-dispatched section types
    # (e.g., CORE_DEFINITION, LOG_DEFINITION, CORE_PARAMETERS etc.).
    if re.search(r"_DEFINITION(_\d+)?$", base):
        return True
    for suffix in ("_PARAMETER", "_PARAMETERS", "_DATA"):
        if base.endswith(suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# Shared reader utilities (new — extracted from duplicated logic)
# ---------------------------------------------------------------------------


def _iter_ascii_data_lines(
    lines: list[str],
    *,
    mode_suffix: str = "",
) -> Generator[str, None, None]:
    """Generator: yield stripped lines for each ASCII data line.

    Handles:
    - Section header detection (~A entry, ~X exit)
    - Comment/empty line skipping
    - Recognition-based section-header injection defense (F-I2-XPD-01)

    Args:
        lines: File content split into lines.
        mode_suffix: Suffix appended to the "data section" string in warning
            messages (e.g. ``" (wrapped mode)"`` for _read_wrapped).

    Yields:
        Each stripped data line within the ~A section.

    Stops yielding when a recognized non-~A section header is encountered.
    """
    in_ascii = False

    for line in lines:
        stripped = line.strip()

        # F-20: Align section detection with parser.py's SECTION_PATTERN
        # (~[A-Za-z]).  Lines starting with ~ but without an alphabetic
        # section letter (e.g. bare ~, ~~~, etc.) are ignored.
        # F-ITER2-D2-M04: Use _is_ascii_section to check for exact ~A/~ASCII
        # match (aligned with parser._pre_scan), not just the first character.
        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                in_ascii = True
            else:
                if in_ascii:
                    # F-I2-XPD-01: Defense-in-depth against section-header
                    # injection via SPLITLINES_CHARS_RE.  Only break reading
                    # for recognized LAS section headers — unrecognized
                    # patterns (potential artifacts from control-character
                    # replacement) are warned and skipped.
                    section_word = _get_section_word(stripped)
                    if _is_recognized_section_word(section_word):
                        return  # break out of the generator
                    warnings.warn(
                        f"Unrecognized section header '~{section_word}' found in ASCII "
                        f"data section{mode_suffix}.  This may be an "
                        f"artifact of control-character replacement "
                        f"(SPLITLINES_CHARS_RE).  Skipping line.",
                        UserWarning,
                        stacklevel=2,
                    )
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        yield stripped


def _split_data_line(
    stripped: str,
    delimiter: str,
) -> list[str]:
    """Split a data line into tokens using delimiter-aware logic.

    Handles:
    - Space delimiter: str.split(maxsplit=MAX_TOKENS)
    - Non-space delimiter: str.split(delimiter, maxsplit=MAX_TOKENS)
    - Trailing empty string stripping for non-space delimiters
    - F-WXP-06: All types use str.split (no csv.reader)
    - F-DR-01: Bounded allocation via maxsplit parameter

    Also used by _detect_actual_wrap, _read_normal, and _read_wrapped.
    """
    if delimiter == " ":
        return stripped.split(maxsplit=_resolve_max_tokens_per_line())

    _max_tokens = _resolve_max_tokens_per_line()
    values = stripped.split(delimiter, maxsplit=_max_tokens)

    # Strip trailing empty strings from csv.reader output.
    # Trailing delimiters (e.g. "100.0,") produce empty fields that
    # inflate len(values).  Strip only TRAILING empties — middle empty
    # fields represent legitimate sparse data values that must be preserved.
    while values and values[-1] == "":
        values.pop()

    return values


def _detect_string_curves(las_file: LASFile) -> set[int]:
    """Return set of curve indices whose CurveDefinition has string format.

    String curves are those with data_format in ("S",) or ("A",)
    without array_info.

    Args:
        las_file: LASFile with curves and curves_order populated.

    Returns:
        Set of integer indices into curves_order for string-valued curves.
    """
    _string_curve_indices: set[int] = set()
    for _idx in range(len(las_file.curves_order)):
        if _idx < len(las_file.curves):
            cd = las_file.curves[_idx]
            # F-MDR-04: Normalize data_format to uppercase for consistent
            # string-curve detection.  _create_parameter_entry stores
            # data_format without uppercasing; uppercase normalization
            # here guards against lowercase-passed format codes.
            if cd.data_format.upper() in ("S",) or (
                cd.data_format.upper() in ("A",) and cd.array_info is None
            ):
                _string_curve_indices.add(_idx)
    return _string_curve_indices


def _log_conversion_failures(
    _fc: list[int],
    null_value: float,
    _stacklevel: int = 3,
) -> None:
    """Issue warning if non-trivial float conversion failures occurred.

    F-PXR-03: Warn when non-trivial conversion failures occurred
    (non-empty input values that could not be parsed as finite floats
    and were silently replaced with the null value).

    Args:
        _fc: Mutable list with [count] of conversion failures.
        null_value: The null value used as replacement.
        _stacklevel: Stacklevel for warnings.warn (default 3 accounts for
            the extra call frame through _read_normal/_read_wrapped).
    """
    if _fc[0] > 0:
        warnings.warn(
            f"{_fc[0]} value(s) could not be converted to finite float "
            f"and were replaced with the null value ({null_value:.2f}). "
            f"This may indicate string data, corrupt values, or "
            f"non-standard formatting.",
            UserWarning,
            stacklevel=_stacklevel,
        )
