"""DEV (deviation survey) file reader.

Replaces dev_reader.py with modern Python 3, context managers,
and proper encoding handling.
"""

from __future__ import annotations

import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .data_reader import _to_finite_float
from .encoding import read_with_encoding
from .exceptions import DEVReadError, LASDataError, LASEncodingError  # noqa: F401
from .models import DevFile

# Sentinel tokens recognized in DEV data lines — values that indicate
# missing/absent data rather than column names.  Shared by both the
# comma-delimited and whitespace-delimited detection paths.  This is
# the single source of truth; when new sentinels are added they are
# covered in both paths automatically (root cause fix for F-044).
_DEV_SENTINELS: frozenset[str] = frozenset({
    "na", "null", "err", "n/a", "nan", "+nan", "-nan",
    "none", "-", "null.", "n.a.", "nil", "nd", "missing",
    "inf", "-inf", "+inf",
    "infinity", "-infinity", "+infinity",
})

# Characters that Python's splitlines() treats as line breaks beyond \n and \r.
# When present in file content, they cause splitlines() to produce fake section
# headers and corrupt parsed data.  Matches the full character class used by
# reader.py:29 and parser.py:102 — covers all 33 C0 control chars (including
# \x00 (NUL), \x7F (DEL)) plus NEL (\x85) and Unicode line/paragraph separators.
# F-001: Also include the 13 Unicode whitespace characters that the writer's
# _CONTROL_CHARS_RE strips (\u00A0, \u2000-\u200A, \u202F, \u205F, \u3000)
# so the write→read roundtrip is symmetric.
_SPLITLINES_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85"
    r"\u2028\u2029"
    r"\u00A0\u2000-\u200A\u202F\u205F\u3000]"
)

logger = logging.getLogger(__name__)

# Lightweight DEV column name alias mapping for common variants found in
# deviation survey files.  Unlike the full mnem_base.py database (2000+
# LAS curve aliases), this mapping is DEV-specific and covers the most
# common column name variants encountered in the wild.
_DEV_ALIASES: dict[str, str] = {
    # Measured depth variants
    "MDKB": "MD",
    "MDSS": "MD",
    "MDRKB": "MD",
    "DEPTH": "MD",      # Petrel / common industry alias
    "DPT": "MD",         # Petrel depth
    # True vertical depth variants
    "TVDKB": "TVD",
    "TVDSS": "TVD",
    "TVDBML": "TVD",
    # Inclination variants
    "INCL": "INC",
    "DEVI": "INC",       # Petrel deviation
    "DIP": "INC",         # Petrel dip
    # Azimuth variants
    "AZIM": "AZI",
    "AZ": "AZI",          # Petrel azimuth
    "AZM": "AZI",         # Petrel azimuth (abbreviated)
    # Easting (X) variants
    "UTMX": "X",
    "EW": "X",            # Petrel east-west
    "DX": "X",            # Petrel X offset
    # Northing (Y) variants
    "UTMY": "Y",
    "NS": "Y",            # Petrel north-south
    "DY": "Y",            # Petrel Y offset
    # Self-mappings — canonical names must map to themselves
    # so that lowercased canonical names (e.g. "md", "azi") are
    # normalised to uppercase rather than silently preserving the
    # original case and bypassing _validate_dev_data checks.
    "MD": "MD",
    "TVD": "TVD",
    "INC": "INC",
    "AZI": "AZI",
    "X": "X",
    "Y": "Y",
}


def _normalize_dev_column(name: str) -> str:
    """Normalize a DEV column name through the alias mapping.

    Performs case-insensitive lookup: the input name is uppercased,
    matched against the alias table, and the canonical name is returned.
    Names not in the alias table are returned unchanged.

    Args:
        name: Raw column name from the header line.

    Returns:
        Canonical column name if an alias exists, otherwise the original.
    """
    return _DEV_ALIASES.get(name.strip().upper(), name.strip().upper())


def _deduplicate_string_list(
    items: list[str],
    *,
    name_label: str = "name",
    _stacklevel: int = 2,
) -> list[str]:
    """Deduplicate a list of strings with cross-base collision detection.

    Common deduplication algorithm shared across DEV column names and
    LAS curve/parameter mnemonics.  Uses an ``output_names`` set +
    while-loop to ensure generated ``_N`` suffixes don't collide with
    any name already in the output.

    Args:
        items:       List of strings to deduplicate.
        name_label:  Human-readable label used in warning messages
                     (e.g. ``"DEV column name"``).
        _stacklevel: Stacklevel for ``warnings.warn``.

    Returns:
        Deduplicated list of strings with ``_N`` suffixes on collisions.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    output_names: set[str] = set()
    # Track all natural input names so generated _N suffixes don't
    # collide with a name that appears later in the input list.
    # Without this, e.g. input ["MD","TVD","MD","TVD","TVD_2"]
    # produces "TVD_2" for the first duplicate, then misnames the
    # natural "TVD_2" as "TVD_2_2" with a false "duplicate" warning.
    natural_names: frozenset[str] = frozenset(items)
    for name in items:
        if name in seen:
            seen[name] += 1
            suffix = seen[name]
            new_name = f"{name}_{suffix}"
            # Bump suffix past any existing output name OR any natural
            # input name — prevents cross-base collision with names
            # that appear later in the list.
            while new_name in output_names or new_name in natural_names:
                suffix += 1
                new_name = f"{name}_{suffix}"
            seen[name] = suffix
            warnings.warn(
                f"Duplicate {name_label} '{name}' renamed to "
                f"'{new_name}'. Data may come from a file with "
                f"repeated {name_label}s.",
                stacklevel=_stacklevel,
            )
            result.append(new_name)
            output_names.add(new_name)
        else:
            if name in output_names:
                suffix = 2
                new_name = f"{name}_{suffix}"
                while new_name in output_names:
                    suffix += 1
                    new_name = f"{name}_{suffix}"
                seen[name] = suffix
                warnings.warn(
                    f"Duplicate {name_label} '{name}' renamed to "
                    f"'{new_name}'. Data may come from a file with "
                    f"repeated {name_label}s.",
                    stacklevel=_stacklevel,
                )
                result.append(new_name)
                output_names.add(new_name)
            else:
                seen[name] = 1
                result.append(name)
                output_names.add(name)
    return result


def _deduplicate_dev_columns(names: list[str]) -> list[str]:
    """Deduplicate DEV column names with cross-base collision detection.

    Thin wrapper around ``_deduplicate_string_list`` with DEV-specific
    label and adjusted stacklevel so warnings point to the caller of
    this function rather than the internal helper.

    Args:
        names: Raw column names from the header line.

    Returns:
        Deduplicated column names.  A warning is emitted for each
        duplicate.
    """
    return _deduplicate_string_list(
        names, name_label="DEV column name", _stacklevel=3
    )


def _is_float_token(token: str) -> bool:
    """Check if a token can be parsed as a float.

    Handles standard scientific notation (e/E) and Fortran D-exponent
    notation (d/D) used by some scientific software.

    Used by ``_detect_dev_format`` to distinguish numeric data lines
    from column-name header lines.
    """
    # Reject special float strings (nan, inf, -inf, infinity) that
    # float() can parse but are not meaningful numeric well-log data.
    lower = token.replace("D", "E").replace("d", "e")
    if lower.lower() in ("nan", "inf", "-inf", "infinity", "+inf", "-infinity", "+infinity", "-nan", "+nan"):
        return False
    try:
        float(lower)
        return True
    except ValueError:
        return False


# --- Locale-style number handling (V-07 / V-08) ---

# Comma-decimal locale token: digits with a single comma as the decimal
# separator and no other punctuation (e.g. "1,00", "-2,5").  Used by the
# Directional Drilling export variants that are semicolon-delimited but
# use a comma for the decimal point.
_COMMA_DECIMAL_RE = re.compile(r"^([+-]?\d+),(\d+)$")

# Suffix fragment of a thousands-separator number as it appears after
# comma-splitting a comma-delimited data row: exactly 3 digits followed by
# an optional decimal/exponent part (e.g. "234.5" from "1,234.5").
_THOUSANDS_FRAG_RE = re.compile(r"^\d{3}(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _convert_comma_decimal(value: str) -> str:
    """Convert a comma-decimal locale token (e.g. ``"1,00"``) to dot notation.

    Some Directional Drilling export variants use a comma as the decimal
    separator while the column delimiter is ``;`` (or space/tab) (V-07).
    Only whole tokens matching ``digits,digits`` with no other punctuation
    are converted — tokens that already contain a dot or exponent are left
    untouched (those use commas as thousands separators, handled separately
    by ``_recombine_thousands_separators``).

    Args:
        value: Raw data token from a data line.

    Returns:
        Token with the comma replaced by a dot when it matches the
        comma-decimal pattern, otherwise the original token.
    """
    m = _COMMA_DECIMAL_RE.match(value)
    if m is None:
        return value
    return f"{m.group(1)}.{m.group(2)}"


def _dev_to_finite_float(
    value_str: str,
    null_value: float,
    _failure_counter: list[int] | None = None,
) -> float:
    """Convert a DEV data token to a finite float.

    Thin wrapper around :func:`data_reader._to_finite_float` that first
    applies comma-decimal locale conversion (V-07) so values like
    ``"1,00"`` parse as ``1.00`` instead of NaN.
    """
    return _to_finite_float(
        _convert_comma_decimal(value_str),
        null_value,
        _failure_counter=_failure_counter,
    )


def _recombine_thousands_separators(
    values: list[str], expected: int
) -> tuple[list[str], str, str] | None:
    """Recombine comma-split thousands-separator fragments in a data row.

    In comma-delimited files a value like ``1,234.5`` is split by the
    delimiter into two tokens (``"1"`` and ``"234.5"``), silently shifting
    every subsequent column (V-08).  When a row has exactly one extra token
    beyond the declared column count, consecutive tokens matching the
    thousands pattern (a short integer prefix + a 3-digit suffix, optionally
    with a decimal part) are recombined into a single value with the comma
    removed (``"1,234.5"`` → ``"1234.5"``).

    Args:
        values:   Tokens from a comma-delimited data row.
        expected: Number of columns declared by the header.

    Returns:
        Tuple of ``(recombined_token_list, original_fragment, merged_value)``
        when a recombination applies, or ``None`` when none does (row already
        matches the expected count, has multiple extra tokens, or contains
        no thousands pattern).
    """
    if len(values) != expected + 1:
        return None
    for i in range(len(values) - 1):
        if values[i].isdigit() and _THOUSANDS_FRAG_RE.match(values[i + 1]):
            merged = values[i] + values[i + 1]
            return (
                [*values[:i], merged, *values[i + 2:]],
                f"{values[i]},{values[i + 1]}",
                merged,
            )
    return None


# --- *COLUMNS keyword helpers (F-012) ---


def _is_columns_header(tokens: list[str]) -> bool:
    """Check whether a list of tokens starts with the *COLUMNS keyword.

    Recognises the Petra / CPS *COLUMNS format where column names are
    prefixed with ``*`` and the first token is the literal keyword
    ``*COLUMNS`` (case-insensitive).
    """
    return bool(tokens) and tokens[0].upper().startswith("*COLUMNS")


def _filter_header_names(values: list[str]) -> list[str]:
    """Filter empty tokens from a header row, preserving positional alignment.

    Only TRAILING empty tokens (from trailing delimiters like ``"MD,TVD,"``)
    are dropped — the data-row ``min()`` guard already tolerates those.  An
    empty token in the MIDDLE of the header (``"MD,TVD,,X,Y"``) would
    silently shift every subsequent column because data rows keep the
    position while the dropped name shortens the list (V-18); reject such
    malformed headers loudly instead of corrupting the column mapping.

    Args:
        values: Raw header tokens (whitespace already stripped).

    Returns:
        Header names with trailing empty tokens removed.

    Raises:
        DEVReadError: When a non-trailing token is empty.
    """
    _end = len(values)
    while _end > 0 and not values[_end - 1].strip():
        _end -= 1
    _head = values[:_end]
    if any(not t.strip() for t in _head):
        raise DEVReadError(
            "Empty column name in the middle of the header line. "
            f"Header tokens: {values!r}. Column names must be non-empty; "
            "a trailing delimiter is allowed but an empty middle cell "
            "would shift every subsequent column."
        )
    return _head


def _parse_columns_tokens(tokens: list[str]) -> list[str]:
    """Extract column names from a *COLUMNS-format header line.

    Strips the ``*COLUMNS`` keyword token, removes the ``*`` prefix from
    every remaining token, and filters out trailing empty names that
    result (e.g. from a trailing ``*``).  Middle empty cells (V-18) are
    rejected by :func:`_filter_header_names`.
    """
    return _filter_header_names([t.lstrip("*") for t in tokens[1:]])


# --- Delimited-line splitting with quoting support (F2-015) ---


def _split_delimited_line(
    line: str,
    delimiter: str,
    max_tokens: int | None = None,
) -> list[str]:
    """Split a delimited line with CSV quoting / escaping support.

    Uses Python's :mod:`csv` module so that double-quoted fields
    containing the delimiter character are kept intact rather than
    being split into separate (wrong) columns.

    Args:
        line:       The stripped content line to split.
        delimiter:  Single-character delimiter (e.g. ``","``).
        max_tokens: Safety cap on the number of tokens returned.
            When None, fetched from data_reader.MAX_TOKENS_PER_LINE
            at call time so that runtime overrides take effect.

    Returns:
        List of individual field values with leading / trailing
        whitespace stripped from each.
    """
    if max_tokens is None:
        from .data_reader import _resolve_max_tokens_per_line

        max_tokens = _resolve_max_tokens_per_line()

    # Pre-validate line length to avoid unbounded CSV token allocation.
    # Naive delimiter count gives an upper bound on actual token count
    # (csv quoting can only reduce, never increase, the count).  A line
    # with 2x the safety cap in delimiters is pathologically malformed
    # and would cause csv.reader to allocate unbounded memory before the
    # truncation slice can take effect.
    if line.count(delimiter) + 1 > max_tokens * 2:
        raise DEVReadError(
            f"Line has approximately {line.count(delimiter) + 1} tokens "
            f"(delimiter {delimiter!r}), exceeding safety cap "
            f"({max_tokens}). The file may be malformed or corrupt."
        )

    try:
        reader = csv.reader([line], delimiter=delimiter, skipinitialspace=True)
        tokens: list[str] = next(reader, [])
    except csv.Error as e:
        raise DEVReadError(
            f"Failed to parse delimited line with delimiter "
            f"{delimiter!r}: {e}"
        ) from e
    return [v.strip() for v in tokens[:max_tokens]]


def _detect_dev_format(content_entries: list[tuple[int, str]]) -> tuple[str, int]:
    """Detect DEV file format from first few content lines.

    Inspects the first 1-3 non-comment, non-empty lines to determine
    which of the three supported DEV format variants the file uses.

    Args:
        content_entries: List of ``(line_index, stripped_content)``
            for the first non-comment, non-empty lines.

    Returns:
        Tuple of ``(format_type, skip_content_lines)`` where *format_type*
        is ``"simple"``, ``"dug"``, or ``"headerless"``, and
        *skip_content_lines* is the number of content-bearing lines to
        skip before data starts (used by Pass 1 counting).
    """
    from .data_reader import _resolve_max_tokens_per_line

    _max_tokens = _resolve_max_tokens_per_line()

    if not content_entries:
        return ("simple", 1)

    # F-012: *COLUMNS keyword format (Petra / CPS variant).
    # The first content line starts with the literal keyword *COLUMNS
    # followed by *-prefixed column names.  Treat this as a simple
    # header format — column-name stripping is done in Pass 2.
    if content_entries[0][1].upper().startswith("*COLUMNS"):
        return ("simple", 1)

    first_tokens = content_entries[0][1].split(maxsplit=_max_tokens)

    # F-019: Check for comma-delimited headerless data BEFORE any format
    # detection.  When any of the first few content lines contains commas
    # and all comma-separated tokens parse as floats, it is headerless
    # comma-delimited data — not a header row.  Without this pre-check,
    # whitespace-based split() produces a single token like "1.0,2.0,3.0"
    # that fails _is_float_token, causing the line to be consumed as a
    # header and silently losing the first data row.
    #
    # I2F-30: Inspect the first 3 content lines for comma presence, not
    # just the first.  A headerless comma-delimited file where the first
    # line is a column count (e.g. "4\\n1.0,2.0,3.0,4.0") has no comma on
    # line 1, causing the original check to miss the comma and fall through
    # to DUG Pattern A detection, consuming the first data row as a header.
    _comma_texts = [entry[1] for entry in content_entries[:3] if "," in entry[1]]
    if _comma_texts:
        # Use the first comma-containing line for token analysis.
        _comma_text = _comma_texts[0]
        comma_tokens = [t.strip() for t in _comma_text.split(",", maxsplit=_max_tokens) if t.strip()]
        if comma_tokens:
            float_tokens = [t for t in comma_tokens if _is_float_token(t)]
            all_float = len(float_tokens) == len(comma_tokens)
            mostly_float = (
                len(float_tokens) >= 2
                and len(float_tokens) >= len(comma_tokens) - 1
            )

            if all_float:
                if len(comma_tokens) >= 2:
                    # F-I2-M03: Port integer-like heuristic from whitespace
                    # path (lines 277-292).  When all comma-separated tokens
                    # are integer-like (no decimal point, no e/d notation),
                    # they may be numeric column names rather than data.
                    # Check the second line for corroboration.
                    first_ints = [
                        t
                        for t in comma_tokens
                        if t.replace("D", "E").replace("d", "e").count(".") == 0
                        and "e" not in t.lower().replace("d", "e")
                    ]
                    all_integer_like = len(first_ints) == len(comma_tokens)
                    # Real headers contain alphabetic characters.  Tokens
                    # like "MD", "TVD" are genuine headers, not data.
                    has_alpha = any(c.isalpha() for t in comma_tokens for c in t)

                    if has_alpha and not (all_float and not all_integer_like):
                        # At least one token has letters — genuine header.
                        # Fall through to whitespace-based format detection.
                        # Skip the pass when all tokens are floats that are
                        # NOT integer-like (decimal points or scientific
                        # notation chars like e/E/d/D which match isalpha()
                        # as false positives). In that case the tokens are
                        # legitimate numeric data, not text headers.
                        pass
                    elif all_integer_like:
                        # All integer-like, no alphabetic: could be numeric
                        # column names (e.g. "100,200,300"), headerless
                        # data, or DUG Pattern A where the first line is an
                        # integer column count.  Check DUG Pattern A first,
                        # then fall back to comma-count matching on the
                        # second line.
                        #
                        # R-F-M01: The DUG check MUST run BEFORE the
                        # second-line comma-matching early return at L386.
                        # When the first content line has no commas (count)
                        # and the second line has comma-delimited all-integer
                        # data, the comma check is self-referential (same
                        # line is both comma_tokens source and second-line
                        # comparison), always matching and returning
                        # ("simple", 1) before the DUG check can fire.
                        # E-F-001: DUG Pattern A check first — first line
                        # may be an integer column count.
                        # V-02 (G9): When the count-prefix DUG pattern is
                        # detected (single-integer first line whose count
                        # matches the comma token count), do NOT return DUG —
                        # the second line is all-integer DATA, not a header.
                        # Return headerless instead, mirroring the whitespace
                        # I2F-001 contract (test_dev_reader.py:2344-2379).
                        first_line_tokens = content_entries[0][1].split(maxsplit=_max_tokens)
                        if (
                            len(first_line_tokens) == 1
                            and len(content_entries) >= 3
                        ):
                            try:
                                dug_count = int(first_line_tokens[0])
                            except ValueError:
                                pass
                            else:
                                if dug_count == len(comma_tokens):
                                    # N-I-24: count-prefix comma file — skip
                                    # the count line (("headerless", 1)) so
                                    # Pass 2 derives columns from the first
                                    # real data line (matches the whitespace
                                    # I2F-001/N-I-24 contract).
                                    return ("headerless", 1)
                        # Not DUG Pattern A — check second line for matching
                        # column count (numeric column names like "100,200,300").
                        if len(content_entries) >= 2:
                            second_comma_tokens = [
                                t.strip()
                                for t in content_entries[1][1].split(",", maxsplit=_max_tokens)
                                if t.strip()
                            ]
                            if len(second_comma_tokens) == len(comma_tokens):
                                return ("simple", 1)
                        # No second line or mismatch — ambiguous.
                        # Treat as headerless data (safer default).
                        return ("headerless", 0)
                    else:
                        # Non-integer float tokens (e.g. "1.5,2.3,3.7")
                        # are definitely data, not column names.
                        # E-F-001: DUG Pattern A check before headerless
                        # return.  First line may be an integer column count.
                        # V-02 (G9): count-prefix DUG misdetection — when
                        # the single-integer first line's count matches the
                        # comma token count, the second line is DATA, not a
                        # numeric header.  Fall through to headerless like
                        # the whitespace I2F-001 path; both comma branches
                        # (integer above, float here) must not return DUG.
                        first_line_tokens = content_entries[0][1].split(maxsplit=_max_tokens)
                        if (
                            len(first_line_tokens) == 1
                            and len(content_entries) >= 3
                        ):
                            try:
                                dug_count = int(first_line_tokens[0])
                            except ValueError:
                                pass
                            else:
                                if dug_count == len(comma_tokens):
                                    # N-I-24: count-prefix comma file — skip
                                    # the count line (("headerless", 1)) so
                                    # Pass 2 derives columns from the first
                                    # real data line (matches the whitespace
                                    # I2F-001/N-I-24 contract).
                                    return ("headerless", 1)
                        return ("headerless", 0)
                else:
                    # Single-column all-float comma token — headerless data.
                    return ("headerless", 0)
            elif mostly_float:
                # F-23: Mostly numeric with at most 1 non-float token.
                # Distinguish between a genuine sentinel (e.g. "na", "NULL",
                # "err", "N/A") and a column name (e.g. "DEPTH", "X").
                # A sentinel in an otherwise numeric row is headerless data;
                # a column name means the first row is a header.
                _non_float_tokens = [
                    t for t in comma_tokens if not _is_float_token(t)
                ]
                # F-023: _is_float_token rejects special float strings
                # (inf, nan, etc.), so they become non-float tokens.
                # Without these sentinels, a row like "1.0, inf, 2.0"
                # is treated as a header instead of being correctly
                # recognised as headerless sentinel-bearing data.
                _is_sentinel = all(
                    t.lower().strip() in _DEV_SENTINELS
                    for t in _non_float_tokens
                )
                if not _is_sentinel:
                    # Non-float token looks like a column name, not a
                    # sentinel.  Fall through to whitespace-based format
                    # detection (which will treat this as a simple header).
                    pass
                else:
                    return ("headerless", 0)
            # Otherwise: too many non-float tokens for this to be headerless
            # data.  Fall through to whitespace-based format detection.

    # V-04 (G9): Semicolon-delimited headerless data detection.
    # The comma pre-check above handles comma-delimited data and the
    # whitespace path below handles space-delimited data, but a
    # semicolon-delimited all-numeric first line (e.g. "1.0;2.0;3.0")
    # previously fell through to simple-format detection, consuming the
    # first data row as column names.  Detect it here so the file is
    # treated as headerless.  Comma-decimal tokens (e.g. "1,00") are
    # normalised before the float check so locale-style data is detected
    # as well (V-07 family).
    _semi_texts = [entry[1] for entry in content_entries[:3] if ";" in entry[1]]
    if _semi_texts:
        _semi_text = _semi_texts[0]
        semi_tokens = [
            t.strip()
            for t in _semi_text.split(";", maxsplit=_max_tokens)
            if t.strip()
        ]
        if semi_tokens and all(
            _is_float_token(_convert_comma_decimal(t)) for t in semi_tokens
        ):
            return ("headerless", 0)

    # DUG Insight format detection — two patterns.
    #
    # Pattern A: single-token first line = column count (no separate title).
    #   "4"              ← integer column count (also serves as title)
    #   "MD TVD X Y"     ← header with non-numeric tokens
    #   2 content lines before data: count + header.
    if len(first_tokens) == 1:
        try:
            col_count = int(first_tokens[0])
        except ValueError:
            pass
        else:
            if len(content_entries) >= 2:
                second_tokens = content_entries[1][1].split(maxsplit=_max_tokens)
                # F-ITER2-D3-M05: Primary check — any non-float token in the
                # second line means it's a header (text column names).
                if any(not _is_float_token(t) for t in second_tokens):
                    return ("dug", 2)
                # Secondary heuristic — when ALL second-line tokens parse as
                # floats (e.g. numeric column names "100 200 300"), check if
                # the integer count from the first line matches the token count
                # on the second line.  Column-count == header-token-count is a
                # strong signal of DUG format even with all-numeric headers.
                # F-08: Require >= 3 content entries before activating the
                # count-match heuristic.  Without this guard a 2-line file
                # like "4\\n100.0 200.0 300.0 400.0\\n" is misdetected as
                # DUG, skip_content_lines=2, zero data lines → total data loss.
                #
                # I2F-001: When the second line is all-float AND the column
                # count matches exactly, the file is ambiguous — it could be
                # DUG with numeric column names OR headerless data with a
                # column-count prefix.  Prefer headerless (safer default).
                # Without this guard a file like "4\\n1.0 2.0 3.0 4.0\\n5.0..." is
                # falsely detected as DUG and the first data row is consumed
                # as a column-count header.
                if len(content_entries) >= 3 and col_count == len(second_tokens):
                    # N-I-24: Do NOT return DUG, and do NOT fall through to
                    # plain ("headerless", 0) either — that path derives the
                    # column count from the FIRST line ("4" -> 1 token -> 1
                    # column), silently losing 3 of 4 columns (the I2F-001
                    # regression test asserted exactly that wrong result).
                    # Return ("headerless", 1) so Pass 2 skips the count line
                    # and derives column names from the first REAL data line,
                    # mirroring the comma I2F-30 count-prefix handling.
                    return ("headerless", 1)
                # F-DV01: Count-mismatch fallback — when the second line is
                # all-float but the count doesn't match the first line's
                # integer, it's STILL DUG format (not headerless) as long as
                # there are 3+ content entries (i.e. data lines exist beyond
                # the header).  Without >=3 guard a 2-line file like
                # "100\\n50.0\\n" is too ambiguous — it stays headerless.
                # F-01: Require len(second_tokens) > 1 to prevent single-column
                # headerless files (e.g. "100\\n200\\n300\\n") from being
                # misdetected as DUG format.  A DUG file with only 1 column is
                # extremely unusual and would be caught by the col_count ==
                # len(second_tokens) check above anyway.
                # F2-018: Constrain fallback to col_count <= len(second_tokens) + 1
                # to prevent pathologically mismatched headerless data (e.g.,
                # first-line "100" with 3 actual columns) from being misdetected
                # as DUG.  Genuine DUG count mismatches are typically off by 1-2
                # tokens; a 33x mismatch means the first line is data, not a count.
                #
                # I2F-001: Only fire for genuine count mismatches (count !=
                # token count).  When count == token count the case above
                # covers it (fall-through to headerless for all-float).
                elif len(second_tokens) > 1 and len(content_entries) >= 3 and all(
                    _is_float_token(t) for t in second_tokens
                ) and col_count <= len(second_tokens) + 1:
                    return ("dug", 2)

    # Pattern B: multi-word title, integer column count, header.
    #   "Deviation survey for Well-1"   ← descriptive title
    #   "4"                              ← integer column count
    #   "MDKB TVDSS X Y"                ← header with non-numeric tokens
    #   3 content lines before data: title + count + header.
    if len(content_entries) >= 3:
        second_tokens = content_entries[1][1].split(maxsplit=_max_tokens)
        if len(second_tokens) == 1:
            try:
                col_count = int(second_tokens[0])
            except ValueError:
                pass
            else:
                third_tokens = content_entries[2][1].split(maxsplit=_max_tokens)
                # Primary check — any non-float token in the third line
                # means it's a header (text column names).
                if third_tokens and any(not _is_float_token(t) for t in third_tokens):
                    return ("dug", 3)
                # V-01 / V-03 (G9): An all-float third line is DATA, not a
                # DUG header.  The F-21 count-match / count-mismatch numeric
                # column-name heuristics misdetect a normal header file with
                # a ragged or all-zero FIRST data row (e.g. "MD TVD X Y",
                # then "0", then "100.0 1000.0 100.0 200.0") as DUG Pattern
                # B, consuming the real header as a title and the data rows
                # as a column count + numeric column names — total parse
                # corruption with zero warnings.  Mirror Pattern A's I2F-001
                # all-float guard: fall through to simple-format detection
                # (which treats content[0] as the column header) so ragged
                # rows are NaN-filled by Pass 2 instead of corrupting the
                # parse.  Genuine DUG Pattern B files always have a text
                # header line (handled by the primary check above).

    # Headerless format: every token on the first content line parses as
    # a float (no column names present).  No content lines to skip.
    # F-92: When ALL tokens on the first content line parse as floats,
    # check if they also parse as integers (indicating integer column
    # names like "100 200 300" rather than decimal data).  When the
    # second content line has matching column count AND the first line
    # has 2+ integer-like tokens, treat as simple format with numeric
    # column names.  Single-column and decimal-token cases remain
    # headerless (too ambiguous for heuristic disambiguation).
    if first_tokens and all(_is_float_token(t) for t in first_tokens):
        # Only apply heuristic for multi-column files with integer-like
        # column names.  Decimal tokens (0.00, -20.06, 1.0e2) remain
        # headerless — they look like data, not column names.
        if len(first_tokens) >= 2:
            first_ints = [
                t for t in first_tokens
                if t.replace("D", "E").replace("d", "e").count(".") == 0
                and "e" not in t.lower().replace("d", "e")
            ]
            if len(first_ints) == len(first_tokens):
                # V-13: ANY all-integer first row is headerless DATA, not
                # integer column headers.  The F-92 numeric-column-names
                # heuristic previously consumed realistic integer-valued
                # surface stations ("0 0 45" = MD0/INC0/AZI45) as column
                # names whenever the second line's token count matched —
                # fully integer-valued survey files lost their first
                # station and fabricated columns like "0"/"0_2"/"45".
                # F-213's all-zero guard covered only the all-zero row;
                # the second-line count-match branch is the same defect
                # for non-zero integer data (e.g. "100 200 300 400").
                # Prefer headerless (data-preserving default): every row
                # survives as data and no column is fabricated.
                return ("headerless", 0)
        return ("headerless", 0)

    # F-021: Mostly-float whitespace-delimited line with sentinel tokens.
    # Mirror the comma-path sentinel detection (lines ~397-422) for
    # whitespace split.  When the first content line has mostly numeric
    # tokens and the non-numeric tokens are known sentinels (e.g., "na",
    # "err", "NULL"), the line is headerless data — not a column header.
    # Without this check, a line like "100.0 na 200.0 300.0" falls
    # through to simple-format, consuming the first data row as a header.
    if first_tokens:
        float_tokens = [t for t in first_tokens if _is_float_token(t)]
        mostly_float = (
            len(float_tokens) >= 2
            and len(float_tokens) >= len(first_tokens) - 1
        )
        if mostly_float:
            _non_float_tokens = [t for t in first_tokens if not _is_float_token(t)]
            _is_sentinel = all(
                t.lower().strip() in _DEV_SENTINELS for t in _non_float_tokens
            )
            if _is_sentinel:
                return ("headerless", 0)

    # F-093/F-095: Petrel well-header detection.
    # Petrel exports DEV files with a well-header line preceding the column
    # names.  The well-header line has a non-numeric first token (well name)
    # followed by numeric parameters (e.g., depths, coordinates).  Without
    # detection, the well-header is consumed as column names and the real
    # header line becomes NaN data, shifting all columns.
    # V-05 (G9): The original check required >= 2 whitespace tokens, so a
    # comma-delimited well-header like "WELL-1,1000.0,2000.0,50.0" (a single
    # whitespace token) was not detected.  Use the comma-split tokens as the
    # candidate when the whitespace split yields a single token.
    _hdr_tokens = first_tokens
    if len(first_tokens) < 2:
        _cand_tokens = [
            t.strip()
            for t in content_entries[0][1].split(",", maxsplit=_max_tokens)
            if t.strip()
        ]
        if len(_cand_tokens) >= 2:
            _hdr_tokens = _cand_tokens
    if (
        len(content_entries) >= 2
        and len(_hdr_tokens) >= 2
        and not _is_float_token(_hdr_tokens[0])
        and any(_is_float_token(t) for t in _hdr_tokens[1:])
    ):
        second_line_tokens = content_entries[1][1].split(maxsplit=_max_tokens)
        # Real column header line has ALL non-numeric tokens (MD, TVD, etc.).
        if second_line_tokens and not any(
            _is_float_token(t) for t in second_line_tokens
        ):
            warnings.warn(
                f"Detected Petrel well-header line with "
                f"{len(_hdr_tokens)} mixed tokens before column header "
                f"(first token: {_hdr_tokens[0]!r}).  Skipping well-header; "
                f"using second content line as column names.",
                stacklevel=3,
            )
            return ("simple", 2)

    return ("simple", 1)


def _validate_dev_data(
    dev: DevFile, *, _stacklevel: int = 3
) -> None:
    """Validate parsed DEV data for common data-quality issues.

    Performs post-read validation checks and emits warnings for:

    * Non-monotonically increasing MD values (unsorted MD can cause
      inaccurate trajectory calculations per the 3dwellbore.com spec).
    * Azimuth values outside the expected ``[0, 360]`` range.
    * Repeated station MD values (may indicate merged multi-tool
      surveys).
    * MD column with high NaN density (> 50%), which often indicates
      a delimiter mismatch where data was parsed with the wrong
      separator.

    All violations emit :func:`warnings.warn` rather than raising
    exceptions so that users can inspect raw data even when it
    contains quality issues.
    """
    # --- Check MD column exists (case-insensitive; F-043) ---
    # Each validation block independently guards its prerequisite
    # columns; azimuth and inclination range checks still run when
    # MD is absent but those columns are present.
    # V-17: MD dedup survivors (MD_2, MD_3 from MD+MDKB/DEPTH alias
    # collisions) previously escaped ALL MD validation because only the
    # exact "MD" name was checked (a6096f4 added the TVD survivor, not
    # MD).  Validate every MD-family column: the primary exact match
    # plus any _N-suffixed survivor, using the same pattern as the
    # AZI/INC/TVD survivor blocks below.  The survivor scan runs even
    # when ``_md_col is None`` so a file whose depth column is naturally
    # named ``MD_2`` is validated instead of triggering the misleading
    # "MD column not found" warning.
    _md_col = None
    for _cn in dev.columns:
        if _cn.upper() == "MD":
            _md_col = _cn
            break

    _md_survivors: list[str] = []
    for _col_name in dev.column_order:
        _col_upper = _col_name.upper()
        _suffix = _col_name[len("MD"):]
        if (
            _col_upper.startswith("MD")
            and _suffix.startswith("_")
            and _suffix[1:].isdigit()
        ):
            _md_survivors.append(_col_name)

    def _check_md_column(_md_name: str) -> None:
        """Run the MD-family data-quality checks on one column."""
        md = dev.columns[_md_name]
        total = len(md)
        # --- Check for negative MD values (F-45: moved outside
        #     _md_check_ok gate — runs even with single finite value) ---
        md_finite_all = md[~np.isnan(md)]
        if len(md_finite_all) > 0:
            neg_md = md_finite_all < 0
            if np.any(neg_md):
                n_neg = int(np.sum(neg_md))
                neg_vals = md_finite_all[neg_md][:3]
                warnings.warn(
                    f"Found {n_neg} negative MD value(s) in column "
                    f"'{_md_name}': "
                    f"{', '.join(str(v) for v in neg_vals)}"
                    f"{'...' if n_neg > 3 else ''}. "
                    f"Negative measured depth values are physically "
                    f"impossible in normal well logging.",
                    stacklevel=_stacklevel,
                )
        # Gating flag: controls whether monotonicity & repeat checks
        # run.  Set to False when conditions (single-row, all-NaN,
        # fewer than 2 finite values) prevent those checks, but do
        # NOT return — azimuth and inclination checks at function
        # scope must still execute.
        _md_check_ok = True

        if total < 2:
            # Need at least 2 stations for monotonicity and repeat checks.
            # NaN-density check still applies for single-row files.
            if total == 1 and np.isnan(md[0]):
                warnings.warn(
                    f"{_md_name} column has 1/1 NaN value. "
                    "Possible delimiter mismatch: data may have been parsed "
                    "with the wrong separator. Specify the correct delimiter "
                    "explicitly.",
                    stacklevel=_stacklevel,
                )
            _md_check_ok = False

        if _md_check_ok:
            # --- 1. Check NaN density (> 50% NaN suggests delimiter mismatch) ---
            nan_count = int(np.isnan(md).sum())
            if nan_count / total > 0.5:
                warnings.warn(
                    f"{_md_name} column has {nan_count}/{total} "
                    f"({nan_count / total:.1%}) "
                    f"NaN values. Possible delimiter mismatch: data may have been "
                    f"parsed with the wrong separator. Specify the correct "
                    f"delimiter explicitly.",
                    stacklevel=_stacklevel,
                )

            # Filter to finite values for monotonicity and duplicate checks.
            finite_mask = ~np.isnan(md)
            if not np.any(finite_mask):
                _md_check_ok = False

        if _md_check_ok:
            finite_md = md[finite_mask]
            if len(finite_md) < 2:
                _md_check_ok = False

        if _md_check_ok:
            # --- 2. Check MD monotonicity (strictly non-decreasing) ---
            diffs = np.diff(finite_md)
            non_monotonic = diffs < 0
            if np.any(non_monotonic):
                n_violations = int(np.sum(non_monotonic))
                violations = np.where(non_monotonic)[0]
                example_lines = []
                for idx in violations[:3]:
                    example_lines.append(f"{finite_md[idx]} -> {finite_md[idx + 1]}")
                warnings.warn(
                    f"{_md_name} values are not monotonically increasing: "
                    f"{n_violations} decrease(s) found. "
                    f"First violations: {', '.join(example_lines)}. "
                    f"Unsorted MD values can cause inaccurate trajectory "
                    f"calculations.",
                    stacklevel=_stacklevel,
                )

            # --- 3. Check repeated station MD values ---
            unique_md, counts = np.unique(finite_md, return_counts=True)
            duplicates = unique_md[counts > 1]
            if len(duplicates) > 0:
                n_dup = len(duplicates)
                example_vals = sorted(duplicates)[:3]
                warnings.warn(
                    f"Found {n_dup} repeated MD station value(s) in column "
                    f"'{_md_name}': "
                    f"{', '.join(str(v) for v in example_vals)}"
                    f"{'...' if n_dup > 3 else ''}. "
                    f"Repeated stations may indicate merged multi-tool surveys.",
                    stacklevel=_stacklevel,
                )

    if _md_col is not None:
        _check_md_column(_md_col)
    for _surv in _md_survivors:
        _check_md_column(_surv)

    if _md_col is None and not _md_survivors:
        warnings.warn(
            "MD column not found in DEV data — validation of MD "
            "monotonicity, NaN density, and repeated stations will be "
            "skipped. Column names found: "
            f"{list(dev.columns.keys()) if dev.columns else '(none)'}.",
            stacklevel=_stacklevel,
        )

    # --- 4. Check azimuth range [0, 360] ---
    # F-043: Iterate over all columns and match case-insensitively
    # so lowercase headers (e.g. "azi", "azim") are validated.
    _azi_names_upper = {"AZI", "AZIM", "AZ", "AZM", "AZIMUTH"}
    for azi_name, azi_data in dev.columns.items():
        if azi_name.upper() not in _azi_names_upper:
            continue
        azi = azi_data
        azi_finite = azi[~np.isnan(azi)]
        if len(azi_finite) == 0:
            continue
        out_of_range = (azi_finite < 0) | (azi_finite > 360)
        if np.any(out_of_range):
            n_bad = int(np.sum(out_of_range))
            bad_vals = azi_finite[out_of_range][:3]
            warnings.warn(
                f"Azimuth column '{azi_name}' has {n_bad} value(s) "
                f"outside [0, 360]: "
                f"{', '.join(str(v) for v in bad_vals)}"
                f"{'...' if n_bad > 3 else ''}. "
                f"Azimuth values outside [0, 360] can cause inaccurate "
                f"trajectory calculations.",
                stacklevel=_stacklevel,
            )
        # I2F-004: Do NOT break — continue checking all matching
        # azimuth columns.  When multiple distinct base-name variants
        # co-exist (e.g., AZI + AZIM with normalize_aliases=False),
        # both must be validated.

    # F-025: When normalize_aliases=True deduplicates columns (e.g.,
    # AZIM + AZ1 both normalise to AZI, producing AZI and AZI_2),
    # the _N-suffixed survivor bypasses the exact-name check above.
    # Validate those deduplication survivors.
    # F-043: Use case-insensitive startswith for column name matching.
    _azi_names = ("AZI", "AZIM", "AZ", "AZM", "AZIMUTH")
    for azi_name in _azi_names:
        _azi_upper = azi_name.upper()
        for col_name in dev.column_order:
            _col_upper = col_name.upper()
            _suffix = col_name[len(azi_name):]
            if (
                _col_upper.startswith(_azi_upper)
                and _suffix.startswith("_")
                and _suffix[1:].isdigit()
            ):
                azi = dev.columns[col_name]
                azi_finite = azi[~np.isnan(azi)]
                if len(azi_finite) == 0:
                    continue
                out_of_range = (azi_finite < 0) | (azi_finite > 360)
                if np.any(out_of_range):
                    n_bad = int(np.sum(out_of_range))
                    bad_vals = azi_finite[out_of_range][:3]
                    warnings.warn(
                        f"Azimuth column '{col_name}' has {n_bad} value(s) "
                        f"outside [0, 360]: "
                        f"{', '.join(str(v) for v in bad_vals)}"
                        f"{'...' if n_bad > 3 else ''}. "
                        f"Azimuth values outside [0, 360] can cause "
                        f"inaccurate trajectory calculations.",
                        stacklevel=_stacklevel,
                    )

    # --- 5. Check inclination range [0, 180] ---
    # F-043: Iterate over all columns and match case-insensitively
    # so lowercase headers (e.g. "inc", "incl") are validated.
    _inc_names_upper = {"INC", "INCL", "DEVI", "DIP"}
    for inc_name, inc_data in dev.columns.items():
        if inc_name.upper() not in _inc_names_upper:
            continue
        inc = inc_data
        inc_finite = inc[~np.isnan(inc)]
        if len(inc_finite) == 0:
            continue
        out_of_range = (inc_finite < 0) | (inc_finite > 180)
        if np.any(out_of_range):
            n_bad = int(np.sum(out_of_range))
            bad_vals = inc_finite[out_of_range][:3]
            warnings.warn(
                f"Inclination column '{inc_name}' has {n_bad} value(s) "
                f"outside [0, 180]: "
                f"{', '.join(str(v) for v in bad_vals)}"
                f"{'...' if n_bad > 3 else ''}. "
                f"Inclination values outside [0, 180] can cause inaccurate "
                f"trajectory calculations.",
                stacklevel=_stacklevel,
            )
        # I2F-004: Do NOT break — continue checking all matching
        # inclination columns.

    # F-025: Deduplication survivors for inclination columns
    # (e.g., INCL + DIP both normalise to INC, producing INC and INC_2).
    # F-043: Use case-insensitive startswith for column name matching.
    _inc_names = ("INC", "INCL", "DEVI", "DIP")
    for inc_name in _inc_names:
        _inc_upper = inc_name.upper()
        for col_name in dev.column_order:
            _col_upper = col_name.upper()
            _suffix = col_name[len(inc_name):]
            if (
                _col_upper.startswith(_inc_upper)
                and _suffix.startswith("_")
                and _suffix[1:].isdigit()
            ):
                inc = dev.columns[col_name]
                inc_finite = inc[~np.isnan(inc)]
                if len(inc_finite) == 0:
                    continue
                out_of_range = (inc_finite < 0) | (inc_finite > 180)
                if np.any(out_of_range):
                    n_bad = int(np.sum(out_of_range))
                    bad_vals = inc_finite[out_of_range][:3]
                    warnings.warn(
                        f"Inclination column '{col_name}' has {n_bad} value(s) "
                        f"outside [0, 180]: "
                        f"{', '.join(str(v) for v in bad_vals)}"
                        f"{'...' if n_bad > 3 else ''}. "
                        f"Inclination values outside [0, 180] can cause "
                        f"inaccurate trajectory calculations.",
                        stacklevel=_stacklevel,
                    )

    # F-100: TVD validation — minimal sanity checks to detect misparsed
    # or corrupt TVD data.  TVD is a fundamental survey column; NaN density
    # and MD-consistency checks catch delimiter mismatches and data corruption.
    # F-043: Iterate over all columns and match case-insensitively.
    _tvd_names_upper = {"TVD", "TVDKB", "TVDSS", "TVDBML"}
    for tvd_name, tvd_data in dev.columns.items():
        if tvd_name.upper() not in _tvd_names_upper:
            continue
        tvd = tvd_data
        tvd_total = len(tvd)

        # NaN density: >50% NaN suggests delimiter mismatch or corrupt data.
        tvd_nan_count = int(np.isnan(tvd).sum())
        if tvd_total > 0 and tvd_nan_count / tvd_total > 0.5:
            warnings.warn(
                f"TVD column '{tvd_name}' has {tvd_nan_count}/{tvd_total} "
                f"({tvd_nan_count / tvd_total:.1%}) NaN values. "
                f"Possible delimiter mismatch: data may have been parsed "
                f"with the wrong separator.",
                stacklevel=_stacklevel,
            )

        # MD-consistency: when both MD and TVD are present, TVD should not
        # decrease where MD increases (in normal wells TVD increases with
        # depth).  This is a soft check — TVD can stay constant in horizontal
        # sections, but backward jumps signal data corruption.
        if _md_col is None:
            continue
        md = dev.columns[_md_col]
        both_finite = ~np.isnan(tvd) & ~np.isnan(md)
        if np.sum(both_finite) < 2:
            continue
        md_finite = md[both_finite]
        tvd_finite = tvd[both_finite]
        md_increasing = np.diff(md_finite) > 0
        tvd_decreasing_where_md_increases = np.diff(tvd_finite) < 0
        violations = np.logical_and(
            md_increasing, tvd_decreasing_where_md_increases
        )
        if np.any(violations):
            n_bad = int(np.sum(violations))
            violations_idx = np.where(violations)[0]
            example_pairs = []
            for idx in violations_idx[:3]:
                example_pairs.append(
                    f"MD {md_finite[idx]:.1f}->{md_finite[idx + 1]:.1f}, "
                    f"TVD {tvd_finite[idx]:.1f}->{tvd_finite[idx + 1]:.1f}"
                )
            warnings.warn(
                f"TVD column '{tvd_name}' decreases at {n_bad} station(s) "
                f"where MD increases. Examples: {'; '.join(example_pairs)}"
                f"{'...' if n_bad > 3 else ''}. "
                f"Unexpected TVD reversals may indicate data corruption.",
                stacklevel=_stacklevel,
            )
        # I2F-004: Do NOT break — continue checking all matching
        # TVD columns.

    # F-045: Deduplication survivors for TVD columns
    # (e.g., TVDSS + TVDKB both normalise to TVD, producing TVD and TVD_2).
    # Validate NaN density and MD-consistency for deduplicated survivors
    # matching the pattern used for AZI/INC dedup survivors (F-025).
    _tvd_names = ("TVD", "TVDKB", "TVDSS", "TVDBML")
    for tvd_name in _tvd_names:
        _tvd_upper = tvd_name.upper()
        for col_name in dev.column_order:
            _col_upper = col_name.upper()
            _suffix = col_name[len(tvd_name):]
            if (
                _col_upper.startswith(_tvd_upper)
                and _suffix.startswith("_")
                and _suffix[1:].isdigit()
            ):
                tvd = dev.columns[col_name]
                tvd_total = len(tvd)

                # NaN density: >50% NaN suggests delimiter mismatch.
                tvd_nan_count = int(np.isnan(tvd).sum())
                if tvd_total > 0 and tvd_nan_count / tvd_total > 0.5:
                    warnings.warn(
                        f"TVD column '{col_name}' has "
                        f"{tvd_nan_count}/{tvd_total} "
                        f"({tvd_nan_count / tvd_total:.1%}) NaN values. "
                        f"Possible delimiter mismatch: data may have been "
                        f"parsed with the wrong separator.",
                        stacklevel=_stacklevel,
                    )

                # MD-consistency: TVD should not decrease where MD increases.
                if _md_col is None:
                    continue
                md = dev.columns[_md_col]
                both_finite = ~np.isnan(tvd) & ~np.isnan(md)
                if np.sum(both_finite) < 2:
                    continue
                md_finite = md[both_finite]
                tvd_finite = tvd[both_finite]
                md_increasing = np.diff(md_finite) > 0
                tvd_decreasing_where_md_increases = np.diff(tvd_finite) < 0
                violations = np.logical_and(
                    md_increasing, tvd_decreasing_where_md_increases
                )
                if np.any(violations):
                    n_bad = int(np.sum(violations))
                    violations_idx = np.where(violations)[0]
                    example_pairs = []
                    for idx in violations_idx[:3]:
                        example_pairs.append(
                            f"MD {md_finite[idx]:.1f}->"
                            f"{md_finite[idx + 1]:.1f}, "
                            f"TVD {tvd_finite[idx]:.1f}->"
                            f"{tvd_finite[idx + 1]:.1f}"
                        )
                    warnings.warn(
                        f"TVD column '{col_name}' decreases at "
                        f"{n_bad} station(s) where MD increases. "
                        f"Examples: {'; '.join(example_pairs)}"
                        f"{'...' if n_bad > 3 else ''}. "
                        f"Unexpected TVD reversals may indicate "
                        f"data corruption.",
                        stacklevel=_stacklevel,
                    )

    # --- 6. Check MD NaN density (already checked above; note that
    #       the NaN-density check on the full md array covers all
    #       data including non-MD columns that may be NaN due to
    #       delimiter mismatch, but the MD column is the most
    #       reliable indicator — if MD is >50% NaN, the file is
    #       almost certainly misparsed.) ---


def read_dev_file(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
    delimiter: str | None = None,
    normalize_aliases: bool = True,
) -> dict[str, Any]:
    """Read a DEV (deviation survey) file and return data dictionary.

    Maintains backward compatibility — returns dict of numpy arrays.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a DEVReadError is raised.
        delimiter: Column delimiter.  ``None`` (default) auto-detects
            comma vs whitespace from the header line.  Pass ``" "`` for
            whitespace-only, ``","`` for comma-delimited files.
        normalize_aliases: If ``True`` (default), apply DEV-specific alias
            normalization to column names (e.g. ``MDKB`` → ``MD``).

    Returns:
        Dictionary mapping column names to numpy arrays.

    Raises:
        DEVReadError: If file cannot be read, parsed, or exceeds
            max_file_size.
    """
    dev = read_dev_file_as_object(
        file_path,
        encoding=encoding,
        max_file_size=max_file_size,
        delimiter=delimiter,
        normalize_aliases=normalize_aliases,
    )
    # to_dict() now includes metadata keys (source_file, encoding,
    # column_order) that are strings/lists, not numpy arrays.  Strip
    # them here to maintain the documented contract: "Dictionary mapping
    # column names to numpy arrays."
    # R7F-01: Also strip _meta_-prefixed keys — to_dict stores metadata
    # under _meta_ prefix when a column name collides with a metadata
    # key (I2F-28), and the returned dict must not leak those.
    # F-I2-DV-04: Only strip metadata keys (encoding, source_file,
    # column_order) when they are NOT column names — if a column
    # shares a metadata key name, the column data must be preserved.
    _metadata_keys = {"encoding", "source_file", "column_order"}
    result = dev.to_dict()
    return {k: v for k, v in result.items()
            if not k.startswith("_meta_")
            and not (k in _metadata_keys and k not in dev.columns)}


def read_dev_file_as_object(
    file_path: str | Path,
    encoding: str | None = None,
    max_file_size: int | None = None,
    delimiter: str | None = None,
    normalize_aliases: bool = True,
) -> DevFile:
    """Read a DEV file and return DevFile dataclass (new API).

    Same as read_dev_file() but returns the DevFile object directly
    instead of converting to dict.

    Args:
        file_path: Path to DEV file.
        encoding: Optional encoding override.
        max_file_size: Optional maximum file size in bytes. If the file
            exceeds this limit, a DEVReadError is raised.
        delimiter: Column delimiter.  ``None`` (default) auto-detects
            comma vs whitespace from the header line.  Pass ``" "`` for
            whitespace-only, ``","`` for comma-delimited files.
        normalize_aliases: If ``True`` (default), apply DEV-specific alias
            normalization to column names (e.g. ``MDKB`` → ``MD``).

    Returns:
        DevFile dataclass with full parsed data.

    Raises:
        DEVReadError: If file cannot be read, parsed, or exceeds
            max_file_size.

    Example:
        >>> from pylasdev import read_dev_file_as_object
        >>> dev = read_dev_file_as_object("survey.dev")
        >>> dev.column_order
        ['MD', 'TVD', 'X', 'Y']
        >>> dev.columns["MD"][:3]
        array([0., 100., 200.])
        >>> dev.to_dict()["MD"][:3]
        array([0., 100., 200.])
    """
    from .data_reader import (
        MAX_CURVES,
        MAX_DATA_LINES,
        MAX_TOTAL_ELEMENTS,
        _resolve_max_tokens_per_line,
    )

    _max_tokens = _resolve_max_tokens_per_line()

    file_path = Path(file_path)

    if not file_path.exists():
        raise DEVReadError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise DEVReadError(f"Not a file: {file_path}")

    try:
        detected_encoding, content = read_with_encoding(file_path, encoding, max_file_size)
    except OSError as e:
        raise DEVReadError(f"Cannot read file (I/O error): {file_path}") from e
    except (ValueError, LookupError, LASEncodingError) as e:
        raise DEVReadError(
            f"Cannot read file (size exceeded or invalid parameter): {file_path}"
        ) from e

    # Sanitize control characters that Python's splitlines() treats as line
    # breaks before splitting.  This is the same protection used by reader.py
    # and parser.py — without it, embedded control characters can produce fake
    # lines and corrupt data boundaries.
    lines = _SPLITLINES_CHARS_RE.sub(" ", content).splitlines()

    # --- Gather first few content lines for format detection ---
    # Scan past comments and empty lines to collect up to 3 content-bearing
    # lines.  These are used for both format detection and delimiter
    # auto-detection.
    content_entries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            content_entries.append((i, stripped))
            if len(content_entries) >= 4:
                break

    format_type, skip_content_lines = _detect_dev_format(content_entries)

    if format_type == "headerless":
        warnings.warn(
            f"Auto-detected headerless format for {file_path}. "
            f"The first data line has all-numeric tokens.  "
            f"A matching column count on the second line would cause "
            f"the file to be treated as simple format (numeric column "
            f"names as headers).  If the file has a header row, "
            f"consider explicit format specification.",
            stacklevel=2,
        )

    # --- Auto-detect delimiter ---
    # When auto-correction fires (comma-header + space-data), the original
    # comma-split header tokens must be cached before the delimiter switch
    # so Pass 2 uses correct column names instead of re-splitting the header
    # line with the wrong (space) delimiter.
    _cached_hdr_names: list[str] | None = None

    # Extract the header line for delimiter detection and validation.
    # - simple:   first content line
    # - dug:      third content line (the header — skip title+count)
    # - headerless: first content line (the data), but when the first
    #   line has no commas and a later line does (e.g. column-count
    #   prefix like "4\\n1.0,2.0,3.0,4.0"), use the first
    #   comma-containing line for delimiter detection.  (I2F-30)
    if format_type == "dug" and len(content_entries) > skip_content_lines - 1:
        hdr = content_entries[skip_content_lines - 1][1]
    elif content_entries:
        _first = content_entries[
            skip_content_lines - 1 if skip_content_lines > 1 else 0
        ][1]
        if format_type == "headerless" and "," not in _first:
            # N-I-25: Only prefer a later comma-containing line when the
            # first line is a single-token integer column-count prefix
            # (I2F-30, e.g. "4\\n1.0,2.0,3.0,4.0").  For a real
            # space-delimited data first line, a later comma line is a
            # MIXED-delimiter file — the first line's own delimiter must
            # govern, otherwise the comma search derives a 1-column
            # header from a comma line and the space-delimited data is
            # silently reduced to a single col_0 with NaN values.
            _first_tokens = _first.split(maxsplit=_max_tokens)
            _is_count_prefix = False
            if len(_first_tokens) == 1:
                try:
                    int(_first_tokens[0])
                except ValueError:
                    pass
                else:
                    _is_count_prefix = True
            if _is_count_prefix:
                _comma_hdr = next(
                    (entry[1] for entry in content_entries[1:3] if "," in entry[1]),
                    None,
                )
                hdr = _comma_hdr if _comma_hdr else _first
            else:
                hdr = _first
        else:
            hdr = _first
    else:
        hdr = ""

    # Remember whether delimiter was auto-detected so cross-validation
    # can auto-correct for auto-detected delimiters only. User-provided
    # delimiters get validated but not silently changed.
    _delimiter_was_auto = delimiter is None

    if delimiter is None:

        if hdr:
            # If the header contains commas and splitting on comma
            # yields at least as many tokens as splitting on whitespace,
            # treat the file as comma-delimited.  Using >= (not >) so
            # that "MD, TVD, INC, AZI" (commas with trailing spaces)
            # correctly detects comma delimiter.
            comma_tokens = [t for t in hdr.split(",", maxsplit=_max_tokens) if t.strip()]
            # N-I-26: bound the space/tab splits with maxsplit so the
            # delimiter-detection block cannot allocate unbounded token
            # lists before the G-18 token-cap guard (in Pass 2 /
            # _split_delimited_line) ever runs.
            space_tokens = hdr.split(maxsplit=_max_tokens)
            if "\t" in hdr:
                # Tab-delimited file: use str.split("\t") to preserve
                # empty fields between consecutive tabs.
                tab_tokens = hdr.split("\t", maxsplit=_max_tokens)
                # When both tab and semicolon are present, compare token
                # counts to pick the real delimiter.  A semicolon-delimited
                # header with a stray tab (e.g., extra whitespace) should
                # not be misdetected as tab-delimited.  (I2F-21)
                _has_semi = ";" in hdr
                if _has_semi:
                    _semi_tokens = [t for t in hdr.split(";", maxsplit=_max_tokens) if t.strip()]
                    if len(_semi_tokens) > len(tab_tokens) and len(_semi_tokens) >= 2:
                        delimiter = ";"
                    elif len(tab_tokens) >= 2:
                        delimiter = "\t"
                    elif len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                        delimiter = ","
                    else:
                        delimiter = " "
                elif len(tab_tokens) >= 2:
                    delimiter = "\t"
                elif len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                    delimiter = ","
                else:
                    delimiter = " "
            elif ";" in hdr:
                # Semicolon-delimited file: use semicolon if it yields
                # more tokens than comma splitting (F-30).
                semicolon_tokens = [t for t in hdr.split(";", maxsplit=_max_tokens) if t.strip()]
                if len(semicolon_tokens) > len(comma_tokens) and len(semicolon_tokens) >= 2:
                    delimiter = ";"
                elif len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                    delimiter = ","
                else:
                    delimiter = " "
            elif len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                delimiter = ","
            else:
                delimiter = " "
        else:
            delimiter = " "

    # Guard against empty delimiter — str.split("") raises ValueError.
    # The auto-detection above always produces "," or " " but a caller
    # may pass delimiter="" explicitly, bypassing the is-None check.
    # Must run BEFORE cross-validation (which calls str.split(delimiter)).
    if not delimiter:
        raise DEVReadError(
            "Delimiter must be a non-empty string (e.g., ' ' for "
            "whitespace, ',' for comma). Received an empty string."
        )

    # Guard against multi-character delimiter — Python's csv.reader
    # raises TypeError on multi-char delimiters at iteration time,
    # which is not caught by the csv.Error handler in
    # _split_delimited_line.
    if len(delimiter) != 1:
        raise DEVReadError(
            f"Delimiter must be a single character (e.g., ' ' for "
            f"whitespace, ',' for comma). Got {delimiter!r} "
            f"({len(delimiter)} characters)."
        )

    # Cross-validate delimiter against data lines (runs for BOTH
    # auto-detected and user-provided delimiters).  When the header
    # uses commas but data lines are space-delimited, the comma
    # delimiter produces a single token from the entire data line,
    # causing all values to become NaN.  Skip for headerless — the
    # header line IS the data.
    # Auto-correction (switching delimiters) only fires for
    # auto-detected delimiters; user-provided delimiters get a
    # clear error when mismatched.
    #
    # F-047: Validate multiple data lines (up to 5), not just the
    # first.  A wrong delimiter with a matching first-line field
    # count goes undetected when only one line is checked.  Multi-line
    # sampling catches inconsistent delimiters across lines.
    if hdr and format_type != "headerless":
        _content_seen = 0
        _data_lines: list[str] = []
        for _lin in lines:
            _s = _lin.strip()
            if not _s or _s.startswith("#"):
                continue
            _content_seen += 1
            if _content_seen > skip_content_lines:
                _data_lines.append(_s)
                if len(_data_lines) >= 5:
                    break
        if _data_lines:
            _first_data = _data_lines[0]
            # N-I-26: all split sites in this block are bounded with
            # maxsplit so the G-18 token cap applies before any unbounded
            # token-list allocation (the delimiter-detection block runs on
            # the full header string before Pass 2's guards).
            if delimiter == " ":
                _hdr_cols = len(hdr.split(maxsplit=_max_tokens))
                _data_cols = len(_first_data.split(maxsplit=_max_tokens))
            else:
                _hdr_cols = len(
                    [t for t in hdr.split(delimiter, maxsplit=_max_tokens) if t.strip()]
                )
                _data_cols = len(
                    [t for t in _first_data.split(delimiter, maxsplit=_max_tokens) if t.strip()]
                )
            # V-06 (G9): Count how many sampled data lines AFTER the first
            # have exactly the header's token count under the current
            # delimiter.  When a short FIRST data row (ragged format, e.g.
            # an MD-only surface station) is followed by full rows, the
            # delimiter is correct and the short row should be NaN-filled
            # by Pass 2 rather than treated as a delimiter mismatch.
            _later_full_rows = 0
            for _dline in _data_lines[1:]:
                if delimiter == " ":
                    _dcols = len(_dline.split(maxsplit=_max_tokens))
                else:
                    _dcols = len(
                        [t for t in _dline.split(delimiter, maxsplit=_max_tokens) if t.strip()]
                    )
                if _dcols == _hdr_cols:
                    _later_full_rows += 1
            if _hdr_cols >= 2 and _data_cols == 1:
                # Delimiter fails on first data line (produces only 1
                # token).  Try the alternative delimiter.
                _alt_delim = " " if delimiter == "," else ","
                if _alt_delim == " ":
                    _data_alt_cols = len(_first_data.split(maxsplit=_max_tokens))
                else:
                    _data_alt_cols = len(
                        [t for t in _first_data.split(_alt_delim, maxsplit=_max_tokens)
                         if t.strip()]
                    )
                # F-013: Auto-correct only for auto-detected delimiters.
                # User-provided delimiters get a clear error instead of
                # silent data corruption.
                if _delimiter_was_auto and _data_alt_cols >= 2 and _data_alt_cols == _hdr_cols:
                    warnings.warn(
                        f"Auto-corrected delimiter from {delimiter!r} "
                        f"to {_alt_delim!r}: header has {_hdr_cols} "
                        f"columns but first data line produced only "
                        f"{_data_cols} token(s) with {delimiter!r} "
                        f"delimiter.",
                        stacklevel=2,
                    )
                    # F-01: Cache header column names parsed with the
                    # ORIGINAL delimiter before switching.  Without this,
                    # Pass 2 re-parses the header line with the new
                    # (space) delimiter, collapsing a comma-delimited
                    # header like "MD,TVD,INC" into a single bogus
                    # column name.  The cached names are passed to both
                    # the simple-format and DUG-format header paths.
                    # N-I-26: bound the split; V-18: middle empty cells
                    # are rejected (trailing empties dropped) instead of
                    # silently shifting columns.
                    _cached_hdr_names = _filter_header_names(
                        [t.strip() for t in hdr.split(delimiter, maxsplit=_max_tokens)]
                    )
                    delimiter = _alt_delim
                elif _delimiter_was_auto and _later_full_rows > 0:
                    # V-06: The first data row is a single token under
                    # BOTH delimiters but later rows confirm the current
                    # delimiter — the row is genuinely ragged, not a
                    # delimiter mismatch.  Warn and let Pass 2 NaN-fill.
                    warnings.warn(
                        f"First data line has only {_data_cols} token(s) "
                        f"but header declares {_hdr_cols} column(s). "
                        f"Missing values will be filled with NaN because "
                        f"later data line(s) confirm the delimiter "
                        f"{delimiter!r} is correct. This may indicate a "
                        f"ragged first data row.",
                        stacklevel=2,
                    )
                elif _delimiter_was_auto:
                    raise DEVReadError(
                        f"Delimiter mismatch: auto-detected delimiter "
                        f"{delimiter!r} produces {_hdr_cols} columns "
                        f"from header but only {_data_cols} token(s) "
                        f"from first data line. Alternative delimiter "
                        f"{_alt_delim!r} does not match ({_data_alt_cols} "
                        f"tokens). Specify delimiter explicitly."
                    )
                else:
                    raise DEVReadError(
                        f"Delimiter mismatch: explicitly specified "
                        f"delimiter {delimiter!r} produces {_hdr_cols} "
                        f"columns from header but only {_data_cols} "
                        f"token(s) from first data line. The file "
                        f"appears to use {_alt_delim!r} as delimiter "
                        f"({_data_alt_cols} tokens). Either omit the "
                        f"`delimiter` parameter for auto-detection or "
                        f"specify {_alt_delim!r}."
                    )
            elif abs(_hdr_cols - _data_cols) >= 3:
                # V-06: A short first data row (>= 3 token deficit) with
                # corroborating full rows is a ragged row, not a delimiter
                # mismatch — NaN-fill instead of raising.  The long
                # direction and user-provided delimiters keep raising.
                if (
                    _delimiter_was_auto
                    and _data_cols < _hdr_cols
                    and _later_full_rows > 0
                ):
                    warnings.warn(
                        f"First data line has {_data_cols} token(s) but "
                        f"header declares {_hdr_cols} column(s) "
                        f"(difference: {abs(_hdr_cols - _data_cols)}). "
                        f"Missing values will be filled with NaN because "
                        f"later data line(s) confirm the delimiter "
                        f"{delimiter!r} is correct. This may indicate a "
                        f"ragged first data row.",
                        stacklevel=2,
                    )
                else:
                    _source = (
                        "auto-detected" if _delimiter_was_auto
                        else "explicitly specified"
                    )
                    raise DEVReadError(
                        f"Delimiter mismatch: {_source} delimiter "
                        f"{delimiter!r} produces {_hdr_cols} columns from "
                        f"header but {_data_cols} tokens from first data "
                        f"line (difference: {abs(_hdr_cols - _data_cols)}). "
                        f"Specify the correct delimiter explicitly."
                    )
            # F-047: Multi-line consistency check.  When the first
            # data line passed validation, verify that subsequent
            # data lines are consistent with the same delimiter.
            # A mismatch across lines (e.g., comma-delimited line
            # in a space-delimited file) indicates data corruption.
            elif len(_data_lines) >= 3:
                _mismatch_lines: list[int] = []
                for _idx, _dline in enumerate(_data_lines[1:], start=2):
                    if delimiter == " ":
                        _dl_cols = len(_dline.split(maxsplit=_max_tokens))
                    else:
                        _dl_cols = len(
                            [t for t in _dline.split(delimiter, maxsplit=_max_tokens) if t.strip()]
                        )
                    if _dl_cols != _data_cols:
                        _mismatch_lines.append(_idx)
                if _mismatch_lines:
                    _sample = _mismatch_lines[:3]
                    _delim_source = (
                        "auto-detected" if _delimiter_was_auto
                        else "user-specified"
                    )
                    warnings.warn(
                        f"Delimiter consistency warning: "
                        f"{len(_mismatch_lines)} of {len(_data_lines)} "
                        f"sampled data line(s) have a different number "
                        f"of tokens than the first data line "
                        f"({_data_cols} tokens). "
                        f"Affected line(s): {_sample}"
                        f"{'...' if len(_mismatch_lines) > 3 else ''}. "
                        f"The {_delim_source} delimiter {delimiter!r} may "
                        f"be incorrect for some lines. Consider specifying "
                        f"the delimiter explicitly.",
                        stacklevel=2,
                    )

    # --- Pass 1: Count data lines ---
    # skip_content_lines is returned by _detect_dev_format:
    #   dug:         2 (pattern A) or 3 (pattern B)
    #   headerless:  0 (no header — first content line IS data)
    #   simple:      1 (header only)
    content_seen = 0
    data_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content_seen += 1
        if content_seen > skip_content_lines:
            data_lines += 1

    if data_lines > MAX_DATA_LINES:
        raise DEVReadError(
            f"Data line count ({data_lines}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    if data_lines == 0:
        raise DEVReadError("No data lines found in DEV file")

    # --- Pass 2: Parse header and data ---
    dev = DevFile()
    dev.source_file = str(file_path)
    dev.encoding = detected_encoding
    names: list[str] = []
    content_seen = 0
    current_line = 0
    # F-I2-XPD-03: Replace boolean-once flags with counters so automated
    # validation tools can enumerate the total number of affected rows.
    # The first occurrence logs full context; subsequent occurrences are
    # counted silently; a summary is logged at the end of the file.
    extra_col_count: int | None = None  # Track extra-column count for summary
    short_row_count: int | None = None  # Track short-row count for summary
    discarded_lines = 0  # Track silently-discarded lines from pre-scan undercount
    _fc: list[int] = [0]  # Count non-trivial float conversion failures

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        content_seen += 1

        if delimiter == " ":
            values = stripped.split(maxsplit=_max_tokens)
        elif delimiter == "\t":
            # Preserve empty fields between consecutive tabs (str.split()
            # collapses them, causing column shift).  str.split("\t") keeps
            # every empty cell as ''.
            values = [v.strip() for v in stripped.split("\t", maxsplit=_max_tokens)]
        else:
            values = _split_delimited_line(stripped, delimiter)

        # V-08 (G9): Thousands-separator recombination for comma-delimited
        # data rows.  A value like "1,234.5" is split by the comma delimiter
        # into two tokens ("1" and "234.5"), silently shifting every
        # subsequent column.  When a row has one extra token beyond the
        # declared columns and consecutive tokens match the thousands
        # pattern, recombine them (with a warning) before assignment.
        # Headerless first rows are exempt — they define the column count.
        if delimiter == "," and names and len(values) > len(names):
            _recombined = _recombine_thousands_separators(values, len(names))
            if _recombined is not None:
                _merged_vals, _original_frag, _merged_value = _recombined
                warnings.warn(
                    f"Data value '{_original_frag}' contains a thousands "
                    f"separator, which is not natively supported in "
                    f"comma-delimited DEV files; recombined to "
                    f"'{_merged_value}' for column mapping. Review the "
                    f"data line if this row has genuine extra columns.",
                    stacklevel=2,
                )
                values = _merged_vals

        if format_type == "headerless":
            if content_seen <= skip_content_lines:
                # N-I-24: column-count prefix line(s) (e.g. a DUG-style
                # integer count "4" before headerless data).  Skip — it is
                # neither a data row nor a column name.
                continue
            if content_seen == skip_content_lines + 1:
                # Auto-generate column names from the first real data row.
                names = [f"col_{i}" for i in range(len(values))]
                if len(names) >= MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    try:
                        dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                    except MemoryError as e:
                        raise DEVReadError(
                            f"Cannot allocate {data_lines} elements for column "
                            f"'{name}' ({len(names)} columns total): out of memory"
                        ) from e
                dev.column_order = list(names)
                # Store first data row (G-04 bounds guard).
                if current_line < data_lines:
                    for k in range(len(names)):
                        dev.columns[names[k]][current_line] = _dev_to_finite_float(values[k], np.nan, _failure_counter=_fc)
                    current_line += 1
                else:
                    discarded_lines += 1
            else:
                # Remaining data rows (G-04 bounds guard).
                if current_line < data_lines:
                    if len(values) > len(names):
                        if extra_col_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but only "
                                f"{len(names)} columns declared in the header. "
                                f"Extra columns are discarded. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but "
                                f"{len(names)} columns declared in the header. "
                                f"Missing values are filled with NaN. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        dev.columns[names[k]][current_line] = _dev_to_finite_float(values[k], np.nan, _failure_counter=_fc)
                    current_line += 1
                else:
                    discarded_lines += 1

        elif format_type == "dug":
            if content_seen < skip_content_lines:
                # Skip title line(s) and column-count line.
                continue
            elif content_seen == skip_content_lines:
                # Header line — parse column names.
                # Use cached header names if auto-correction changed
                # the delimiter (F-01 fix — prevents comma-header misparse).
                if _cached_hdr_names is not None:
                    values = _cached_hdr_names
                # Handle *COLUMNS keyword format (Petra/CPS variant)
                if _is_columns_header(values):
                    names = _parse_columns_tokens(values)
                else:
                    # V-18: drop only TRAILING empty tokens (trailing
                    # delimiters like "MD,TVD,"); an empty MIDDLE cell
                    # (e.g. "MD,TVD,,X,Y") is rejected instead of
                    # silently shifting every subsequent column.
                    names = _filter_header_names(values)
                if not names:
                    raise DEVReadError(
                        "Empty header line in DUG-format DEV file. "
                        "Expected column names on the third content line."
                    )
                # Apply alias normalization before deduplication so that
                # variants like MDKB and MD are recognized as duplicates.
                if normalize_aliases:
                    names = [_normalize_dev_column(n) for n in names]
                # Deduplicate column names.
                names = _deduplicate_dev_columns(names)
                if len(names) >= MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    try:
                        dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                    except MemoryError as e:
                        raise DEVReadError(
                            f"Cannot allocate {data_lines} elements for column "
                            f"'{name}' ({len(names)} columns total): out of memory"
                        ) from e
                dev.column_order = list(names)
            else:
                # Data lines (G-04 bounds guard).
                if current_line < data_lines:
                    if len(values) > len(names):
                        if extra_col_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but only "
                                f"{len(names)} columns declared in the header. "
                                f"Extra columns are discarded. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but "
                                f"{len(names)} columns declared in the header. "
                                f"Missing values are filled with NaN. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        dev.columns[names[k]][current_line] = _dev_to_finite_float(values[k], np.nan, _failure_counter=_fc)
                    current_line += 1
                else:
                    discarded_lines += 1

        else:  # simple header format
            # F-093: Skip Petrel well-header and any extra pre-header
            # content lines.  Normal simple format has skip_content_lines=1;
            # Petrel detection returns skip_content_lines=2.
            if content_seen < skip_content_lines:
                continue
            if content_seen == skip_content_lines:
                # First non-comment line = column names.
                # Use cached header names if auto-correction changed
                # the delimiter (F-01 fix — prevents comma-header misparse).
                if _cached_hdr_names is not None:
                    values = _cached_hdr_names
                # Handle *COLUMNS keyword format (Petra/CPS variant)
                if _is_columns_header(values):
                    names = _parse_columns_tokens(values)
                else:
                    # V-18: drop only TRAILING empty tokens (trailing
                    # delimiters like "MD,TVD,"); an empty MIDDLE cell
                    # (e.g. "MD,TVD,,X,Y") is rejected instead of
                    # silently shifting every subsequent column.
                    names = _filter_header_names(values)
                if not names:
                    raise DEVReadError(
                        "Empty header line in DEV file. "
                        "Expected column names on the first content line."
                    )
                # Apply alias normalization before deduplication so that
                # variants like MDKB and MD are recognized as duplicates.
                if normalize_aliases:
                    names = [_normalize_dev_column(n) for n in names]
                # Deduplicate column names.
                names = _deduplicate_dev_columns(names)
                if len(names) >= MAX_CURVES:
                    raise DEVReadError(
                        f"Column count ({len(names)}) exceeds maximum allowed "
                        f"({MAX_CURVES}). The file may be malformed or corrupt."
                    )
                if len(names) * data_lines > MAX_TOTAL_ELEMENTS:
                    raise DEVReadError(
                        f"Total allocation ({len(names)} columns x "
                        f"{data_lines} lines = "
                        f"{len(names) * data_lines} elements) exceeds "
                        f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                        f"The file may be malformed or corrupt."
                    )
                for name in names:
                    try:
                        dev.columns[name] = np.full(data_lines, np.nan, dtype=np.float64)
                    except MemoryError as e:
                        raise DEVReadError(
                            f"Cannot allocate {data_lines} elements for column "
                            f"'{name}' ({len(names)} columns total): out of memory"
                        ) from e
                dev.column_order = list(names)
            else:
                # Data lines (G-04 bounds guard).
                if current_line < data_lines:
                    if len(values) > len(names):
                        if extra_col_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but only "
                                f"{len(names)} columns declared in the header. "
                                f"Extra columns are discarded. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            warnings.warn(
                                f"Data line has {len(values)} values but "
                                f"{len(names)} columns declared in the header. "
                                f"Missing values are filled with NaN. (Subsequent "
                                f"occurrences will be counted; see summary.)",
                                stacklevel=2,
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        dev.columns[names[k]][current_line] = _dev_to_finite_float(values[k], np.nan, _failure_counter=_fc)
                    current_line += 1
                else:
                    discarded_lines += 1

    # --- Post-scan diagnostics (F-I2-DV-07) ---
    if discarded_lines > 0:
        warnings.warn(
            f"Pre-scan undercount: {discarded_lines} data line(s) discarded "
            f"because the actual data exceeds the {data_lines} lines declared "
            f"by Pass 1. Dev file data may be truncated.",
            stacklevel=2,
        )
    if current_line < data_lines:
        warnings.warn(
            f"Pre-scan overcount: declared {data_lines} data lines but only "
            f"{current_line} actual data lines found. Pre-allocated array "
            f"tail contains NaN.",
            stacklevel=2,
        )

    # F-I2-XPD-03: Summary of extra-column and short-row occurrences,
    # replacing the previous boolean-once pattern that suppressed all
    # diagnostics after the first row.  This allows automated data
    # quality tools to enumerate affected rows.
    if extra_col_count is not None and extra_col_count > 1:
        warnings.warn(
            f"{extra_col_count} data line(s) had more values than expected. "
            f"Extra columns were discarded.",
            stacklevel=2,
        )
    if short_row_count is not None and short_row_count > 1:
        warnings.warn(
            f"{short_row_count} data line(s) had fewer values than expected. "
            f"Missing values were filled with NaN.",
            stacklevel=2,
        )

    # Warn when non-trivial float conversion failures occurred
    # (non-empty input values that could not be parsed as finite floats
    # and were silently replaced with NaN).  Mirrors the pattern in
    # data_reader.py (F-PXR-03).
    if _fc[0] > 0:
        warnings.warn(
            f"{_fc[0]} value(s) could not be converted to finite float "
            f"and were replaced with NaN. This may indicate string data, "
            f"corrupt values, or non-standard formatting.",
            stacklevel=2,
        )

    # F-041: Re-invoke structural invariants and validate(complete=True)
    # after populating all columns.  The initial __post_init__ call during
    # DevFile() construction early-returns because columns is empty.
    # Calling it here verifies column_order consistency, array length
    # uniformity, and runs data-quality validation (NaN/Inf, MD
    # monotonicity, AZI/INC range via validate(complete=True)).
    dev.__post_init__()
    _validate_dev_data(dev, _stacklevel=3)
    return dev
