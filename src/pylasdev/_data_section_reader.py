"""Shared data section reading utilities extracted from data_reader.py.

Contains section-detection primitives (moved from data_reader.py) and
shared reader logic used by _read_normal, _read_wrapped, and _detect_actual_wrap.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Generator, Sequence, Set
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

    P-05: A no-space pipe target (e.g. "~ASCII|CURVE" → "ASCII") is
    stripped from the word, matching parser._pre_scan / parser._parse_line
    pipe handling.  Without this, LAS 1.2/2.0 files using a pipe-delimited
    ASCII header were not recognized as data sections by the reader and
    silently produced ZERO data rows.
    """
    match = _SECTION_WORD_RE.match(stripped)
    if not match:
        return ""
    word = match.group(1)
    if "|" in word:
        word = word[: word.find("|")].strip()
    return word.upper()


def _is_ascii_section(stripped: str) -> bool:
    """Check if a section header targets the ASCII data (~A / ~ASCII) section.

    Aligned with parser._pre_scan which uses ``section_word in {"A", "ASCII"}``.
    P-05: ``_get_section_word`` strips a no-space pipe target, so
    ``~ASCII|CURVE`` is recognized as an ASCII section (matching the parser).
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
_KNOWN_SECTION_WORDS: frozenset[str] = frozenset(
    {
        # Data section words (from parser._DATA_SECTION_WORDS)
        "A",
        "ASCII",
        "CORE",
        "CORE_DATA",
        "DRILLING",
        "DRILLING_DATA",
        "FORMATION",
        "FORMATION_DATA",
        "INCLINOMETRY",
        "INCLINOMETRY_DATA",
        "LOG",
        "LOG_DATA",
        "MUD",
        "MUD_DATA",
        "PERFORATIONS",
        "PERFORATIONS_DATA",
        "RISK",
        "RISK_DATA",
        "STRUCTURE",
        "STRUCTURE_DATA",
        "TEST",
        "TEST_DATA",
        "TOPS",
        "TOPS_DATA",
        # Non-data section words (from parser dispatch table)
        "C",
        "CURVE",
        "D",
        "DEFINITION",
        "O",
        "OTHER",
        "P",
        "PARAMETER",
        "PARAMETERS",
        "V",
        "VERSION",
        "W",
        "WELL",
    }
)


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
                    # P-16: Treat EVERY genuine ~-prefixed section header
                    # (recognized OR unrecognized) as a section boundary,
                    # matching the parser's section-boundary classification.
                    # Unrecognized sections are routed to other_lines by the
                    # parser, so their body must NOT be consumed as data rows.
                    # Previously unrecognized words were warn+skip, leaving
                    # in_ascii=True and reading the section's body as data —
                    # the same lines landed in BOTH logs AND other (garbage
                    # rows + duplicated text).
                    # F-I2-XPD-01 retained: only break on a genuine
                    # ~-prefixed section-like line (checked by
                    # _is_section_header); control-char noise (~3D, ~., ~~,
                    # ~#) fails that check and is skipped below — it never
                    # breaks data reading.
                    section_word = _get_section_word(stripped)
                    if not _is_recognized_section_word(section_word):
                        warnings.warn(
                            f"Unrecognized section header '~{section_word}' found in ASCII "
                            f"data section{mode_suffix}.  Data reading stops at this "
                            f"section boundary.",
                            UserWarning,
                            stacklevel=2,
                        )
                    return  # break out of the generator
            continue

        # P-16: ~-prefixed lines that are NOT section headers (e.g. ~3D,
        # ~., ~#, bare ~, control-character replacement artifacts) are not
        # data rows — the parser routes them to other_lines
        # (parser.py:1277-1283).  Skip them instead of yielding as data rows.
        if stripped.startswith("~"):
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


# ---------------------------------------------------------------------------
# Shared wrap-detection decision core (W-13: single source of truth)
# ---------------------------------------------------------------------------


def detect_actual_wrap_from_window(
    window: list[int],
    curve_count: int,
    declared_wrap: str | None,
    empty_window_default: bool = True,
) -> bool:
    """Pure decision: is the first-4-data-lines value-count *window* wrapped?

    Single source of truth for the wrap gate, shared by the LAS 1.2/2.0
    path (``data_reader._detect_actual_wrap``) and the LAS 3.0 path
    (``_las30_data._detect_actual_wrap_las30``).  Callers build the window
    with their own section scan (the two twins collect windows differently —
    data_reader scans the whole file for ``~A``; las30 scans pre-collected
    section lines — INT-02) and handle the empty-window default via
    *empty_window_default* (INT-01/W-4: data_reader True, las30 False).

    Arms (byte-identical to the pre-refactor twins, plus the R-1 flowing
    rule and the R-3/II-5 discriminator):

    - empty window → *empty_window_default* (MUST precede the curve_count
      guard — a 1-curve empty file returns True on the data_reader path);
    - ``curve_count <= 1`` → False (wrapped/non-wrapped indistinguishable);
    - depth-later arm (F-07): any 1-value row in window[1:] is depth
      evidence → declared WRAP=YES → True; else a full first line AND >=2
      one-value rows → True, UNLESS a full row immediately follows a
      1-value row (R-3/II-5: a wrapped continuation line carries
      curve_count-1 values, never a full row — such a 1-value row was a
      ragged short row, so the file is NOT wrapped);
    - first-line-full arm (M-38) with the R-1 multiple-of flowing rule
      (``window[0] % curve_count == 0 and window[0] > curve_count`` → True,
      placed BEFORE the declared-WRAP=YES fall-through): a full first line
      carrying 2+ complete depth steps is a continuous-flow wrapped layout,
      declaration-INDEPENDENT (W-A);
    - H-02 uniform-short guard: uniform short rows (1 < L < curve_count,
      no 1-value depth line) are a column-count mismatch → False;
    - majority vote: >=2 full → False; >=3 partial → True;
    - tiebreak: declared header YES → True, else True (conservative).

    Args:
        window: Value counts of up to 4 data lines (caller-collected).
        curve_count: Declared curve count for the section.
        declared_wrap: Declared WRAP header value ("YES"/"NO") or None.
        empty_window_default: Returned when *window* is empty (no data
            lines found); True on the LAS 1.2/2.0 path (conservative),
            False on the LAS 3.0 path (blank-only section must not be
            rejected as WRAP=YES).

    Returns:
        True when the data is genuinely wrapped.
    """
    # INT-01: the empty-window check MUST precede the curve_count<=1 guard —
    # data_reader's (1-curve, no-data) case returns True, las30's False.
    if not window:
        return empty_window_default
    if curve_count <= 1:
        return False

    def _is_full(n: int) -> bool:
        # COMMA/TAB: trailing empties are stripped by _split_data_line, so
        # value counts reflect real tokens.  A data line is "full" only
        # when it carries the complete row (curve_count values) — a
        # wrapped continuation line carries curve_count-1 values and a
        # depth-only line carries exactly 1, so both are partial (wrapped)
        # evidence.  (W2-F3: the twin's delimiter if/else was dead — both
        # branches returned n >= curve_count — collapsed to one form.)
        return n >= curve_count

    # F-07 (DR-01/I2-04): depth-line evidence rule.  A genuine wrapped
    # file ALWAYS has depth lines (rows with exactly 1 value).  When any
    # later window line carries exactly 1 value:
    #   - declared WRAP=YES → wrapped (I2-04)
    #   - first line full AND at least TWO 1-value rows in the window →
    #     wrapped (DR-01: mixed-wrap / mnemonic-header masquerade
    #     [3,1,1] / [3,1,2,1] with WRAP=NO or absent — content outranks a
    #     NO header).
    # R-3/II-5 discriminator: a full row immediately after a 1-value row
    # is IMPOSSIBLE in genuine wrapped data (a wrapped continuation line
    # carries curve_count-1 values, never curve_count) — the 1-value row
    # was a ragged short row, so the >=2 arm must NOT fire ([3,1,3,1] /
    # [3,1,1,3] WRAP=NO stay non-wrapped — W-2/NEW-1).
    depth_later = len(window) > 1 and any(n == 1 for n in window[1:])
    if depth_later:
        if declared_wrap is not None and declared_wrap.upper() == "YES":
            return True
        if _is_full(window[0]) and sum(1 for n in window[1:] if n == 1) >= 2:
            if not any(
                window[i] == 1 and _is_full(window[i + 1]) for i in range(1, len(window) - 1)
            ):
                return True

    # First line full → non-wrapped (wrapped first line is always depth).
    # R-1/II-6 (W-A, lasio #583): a full first line carrying 2+ complete
    # depth steps (multiple of curve_count AND > curve_count) is a flowing
    # wrapped layout — declaration-INDEPENDENT.  Placed BEFORE the
    # declared-WRAP=YES fall-through so [6,6] nc=3 → True, while
    # [3,3,3,3] YES → False (3 > 3 fails; test_regression.py:4708 lock)
    # and [4,4,4,4] nc=3 extra-columns → False (4 % 3 != 0) are preserved.
    # curve_count >= 3 guard: for nc=2, ANY even line length > 2 is
    # multiple-of-aligned, so the rule cannot separate a flowing nc=2 file
    # from an extra-columns nc=2 file — the extra-columns reading is
    # test-locked on both paths (test_reader.py:1736, test_parser.py:1395),
    # so nc=2 stays non-wrapped (II-6 spirit: never misroute extra-columns).
    if _is_full(window[0]) and curve_count >= 3:
        if window[0] % curve_count == 0 and window[0] > curve_count:
            return True
        # M-38: a WRAP=YES header with a COMPLETE first row can still be a
        # genuine mixed-wrap file (per-row line-width wrapping): first row
        # complete, continuation lines wrapped.  When the header declares
        # WRAP=YES and later window lines are partial (continuation/depth
        # evidence), fall through to the majority vote instead of
        # short-circuiting to non-wrapped.
        if declared_wrap is not None and declared_wrap.upper() == "YES":
            if len(window) > 1 and any(not _is_full(n) for n in window[1:]):
                pass  # fall through to the majority vote below
            else:
                return False
        else:
            return False

    full_count = sum(1 for n in window if _is_full(n))
    partial_count = len(window) - full_count

    # H-02: curve-count mismatch guard.  When ~C declares MORE curves than
    # ~A rows contain (e.g. 3 curves declared but every row carries 2
    # values), every line is "partial" and the majority vote below would
    # classify the file as WRAPPED — routing it to _read_wrapped, which
    # shifts every non-depth column and silently drops ~half the rows.
    # A genuine wrapped file's line lengths VARY: depth lines carry exactly
    # 1 value and continuation lines carry curve_count-1.  A uniform short
    # length L (1 < L < curve_count) across the whole window, with NO
    # 1-value depth line, is a column-count mismatch — treat as
    # non-wrapped so the graceful short-row null-fill preserves the data.
    if (
        len(window) >= 2
        and full_count == 0
        and len(set(window)) == 1
        and 1 < window[0] < curve_count
    ):
        return False

    # Two full rows among the first 4 → definitively non-wrapped.
    if full_count >= 2:
        return False
    # At least 3 partial rows and fewer than 2 full → wrapped.
    if partial_count >= 3:
        return True
    # Ambiguous window (e.g. 2-2 or 1-1): use the declared header as the
    # tiebreak, else default to wrapped (conservative).
    if declared_wrap is not None:
        return declared_wrap.upper() == "YES"
    return True


# ---------------------------------------------------------------------------
# Shared mnemonic-header predicate (H-scope single source of truth)
# ---------------------------------------------------------------------------


def _split_header_row(stripped: str) -> list[str]:
    """Split a candidate mnemonic-header row on whitespace OR commas.

    Superset tokenizer used by ALL header-skip predicate sites (H-1/II-11):
    a DLM=COMMA file whose mnemonic row is space-separated (``~A\\nDEPT GR\\n``)
    is recognized, because the DLM-aware ``_split_data_line`` would see a
    single token ``"DEPT GR"`` and consume the header as data (phantom
    all-null first row + one-row shift).

    .. note::

       The superset split introduces a narrow, accepted data-loss class
       (II-11): a genuine first-row STRING value containing a space in a
       COMMA-DLM mixed section (e.g. ``GR RHOB`` as string data) is split
       into mnemonic tokens and skipped as a header.  The all-string
       exclusion and the min(2, count)..count token bound keep this
       confined to mixed sections whose first row is all-mnemonic — locked
       by a dedicated regression test.
    """
    tokens = re.split(r"[\s,]+", stripped)
    while tokens and tokens[-1] == "":
        tokens.pop()
    return tokens


def is_mnemonic_header_row(
    tokens: Sequence[str],
    *,
    declared: Set[str],
    curve_count: int,
    all_string: bool,
) -> bool:
    """Pure predicate: True when *tokens* are a standalone mnemonic header.

    Single source of truth for the header-skip decision (H-scope),
    consumed by ``data_reader._is_mnemonic_header_row`` (LAS 1.2/2.0),
    ``parser._pre_scan`` (inline mirror), and
    ``parser._is_standalone_mnemonic_header`` (LAS 3.0).

    Callers MUST (a) gate on first-line-of-section (no data row consumed
    yet — DR-M3: the only position where a standalone mnemonic header can
    legitimately appear) and (b) pass the DEDICATED distinct-curve count
    for the section scope (H-4) — never ``len(declared)``, which includes
    original_mnemonic aliases (parser.py:1272-1279 trap).

    Clauses (identical across the three pre-refactor copies):

    1. **Token-count bounds**: ``min(2, curve_count) <= len(tokens) <=
       curve_count``.  The 2-token minimum keeps a single-token
       short/ragged data row that happens to equal a mnemonic (e.g. a
       string value ``LITH``) from being wrongly skipped as a header
       (M-02); for a SINGLE-curve section the lower bound drops to 1
       (PSR-1).
    2. **All-string exclusion**: when every curve in the section is a
       string curve, every data value is a string, so a mnemonic-coincident
       value is indistinguishable from a header row by content alone (M-03,
       F-19).
    3. **Match-set membership**: every token (uppercased) must be in the
       caller-supplied *declared* match set (resolved + original mnemonics;
       the caller owns the match-set builder — H-02/II-10c per-site
       parameterization).
    """
    if not tokens:
        return False
    if len(tokens) < min(2, curve_count) or len(tokens) > curve_count:
        return False
    if all_string:
        return False
    return all(tok.upper() in declared for tok in tokens)
