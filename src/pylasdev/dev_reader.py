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
from .exceptions import DEVReadError, LASEncodingError  # noqa: F401
from .models import DevFile

# Characters that Python's splitlines() treats as line breaks beyond \n and \r.
# When present in file content, they cause splitlines() to produce fake section
# headers and corrupt parsed data.  Matches the full character class used by
# reader.py:29 and parser.py:102 — covers all 33 C0 control chars (including
# \x00 (NUL), \x7F (DEL)) plus NEL (\x85) and Unicode line/paragraph separators.
_SPLITLINES_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x85\u2028\u2029]")

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
    return _DEV_ALIASES.get(name.upper(), name)


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
    for name in items:
        if name in seen:
            seen[name] += 1
            suffix = seen[name]
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
    if lower.lower() in ("nan", "inf", "-inf", "infinity", "+inf", "-infinity", "+infinity"):
        return False
    try:
        float(lower)
        return True
    except ValueError:
        return False


# --- *COLUMNS keyword helpers (F-012) ---


def _is_columns_header(tokens: list[str]) -> bool:
    """Check whether a list of tokens starts with the *COLUMNS keyword.

    Recognises the Petra / CPS *COLUMNS format where column names are
    prefixed with ``*`` and the first token is the literal keyword
    ``*COLUMNS`` (case-insensitive).
    """
    return bool(tokens) and tokens[0].upper().startswith("*COLUMNS")


def _parse_columns_tokens(tokens: list[str]) -> list[str]:
    """Extract column names from a *COLUMNS-format header line.

    Strips the ``*COLUMNS`` keyword token, removes the ``*`` prefix from
    every remaining token, and filters out any empty names that result
    (e.g. from a trailing ``*``).
    """
    return [t.lstrip("*") for t in tokens[1:] if t.lstrip("*")]


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
        comma_tokens = [t.strip() for t in _comma_text.split(",") if t.strip()]
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
                        # column names (e.g. "100,200,300") or headerless
                        # data.  Check second line for matching column count.
                        if len(content_entries) >= 2:
                            second_comma_tokens = [
                                t.strip()
                                for t in content_entries[1][1].split(",")
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
                _SENTINELS = {
                    "na", "null", "err", "n/a", "nan", "none", "-",
                    "null.", "n.a.", "nil", "nd", "missing",
                }
                _is_sentinel = all(
                    t.lower().strip() in _SENTINELS
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
                if len(content_entries) >= 3 and col_count == len(second_tokens):
                    return ("dug", 2)
                # F-DV01: Count-mismatch fallback — when the second line is
                # all-float but the count doesn't match the first line's
                # integer, it's still DUG format (not headerless) as long as
                # there are 3+ content entries (i.e. data lines exist beyond
                # the header).  Without >=3 guard a 2-line file like
                # "100\\n50.0\\n" is too ambiguous — it stays headerless.
                # F-01: Require len(second_tokens) > 1 to prevent single-column
                # headerless files (e.g. "100\\n200\\n300\\n") from being
                # misdetected as DUG format.  A DUG file with only 1 column is
                # extremely unusual and would be caught by the col_count ==
                # len(second_tokens) check above anyway.
                if len(second_tokens) > 1 and len(content_entries) >= 3 and all(
                    _is_float_token(t) for t in second_tokens
                ):
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
                # F-21: Port Pattern A's count-match and count-mismatch
                # fallback heuristics to Pattern B.  When all third-line
                # tokens parse as floats (all-numeric column names), the
                # integer count from line 2 matching the token count on
                # line 3 is a strong signal of DUG format.  Requires 4+
                # content entries (1+ data lines) to avoid 3-line false
                # positives.
                if len(content_entries) >= 4 and col_count == len(third_tokens):
                    return ("dug", 3)
                # Count-mismatch fallback — when the third line is all-float
                # but the count doesn't match, it's still DUG format as long
                # as there are 4+ content entries (data lines beyond the
                # header) AND the third line has more than 1 token.
                if len(third_tokens) > 1 and len(content_entries) >= 4 and all(
                    _is_float_token(t) for t in third_tokens
                ):
                    return ("dug", 3)

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
                if len(content_entries) >= 2:
                    second_tokens = content_entries[1][1].split(maxsplit=_max_tokens)
                    if len(second_tokens) == len(first_tokens):
                        return ("simple", 1)
        return ("headerless", 0)

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
    # --- Check MD column exists ---
    if "MD" not in dev.columns:
        return

    md = dev.columns["MD"]
    total = len(md)
    if total < 2:
        # Need at least 2 stations for monotonicity and repeat checks.
        # NaN-density check still applies for single-row files.
        if total == 1 and np.isnan(md[0]):
            warnings.warn(
                "MD column has 1/1 NaN value. "
                "Possible delimiter mismatch: data may have been parsed "
                "with the wrong separator. Specify the correct delimiter "
                "explicitly.",
                stacklevel=_stacklevel,
            )
        return

    # --- 1. Check NaN density (> 50% NaN suggests delimiter mismatch) ---
    nan_count = int(np.isnan(md).sum())
    if nan_count / total > 0.5:
        warnings.warn(
            f"MD column has {nan_count}/{total} ({nan_count / total:.1%}) "
            f"NaN values. Possible delimiter mismatch: data may have been "
            f"parsed with the wrong separator. Specify the correct "
            f"delimiter explicitly.",
            stacklevel=_stacklevel,
        )

    # Filter to finite values for monotonicity and duplicate checks.
    finite_mask = ~np.isnan(md)
    if not np.any(finite_mask):
        return

    finite_md = md[finite_mask]
    if len(finite_md) < 2:
        return

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
            f"MD values are not monotonically increasing: "
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
            f"Found {n_dup} repeated MD station value(s): "
            f"{', '.join(str(v) for v in example_vals)}"
            f"{'...' if n_dup > 3 else ''}. "
            f"Repeated stations may indicate merged multi-tool surveys.",
            stacklevel=_stacklevel,
        )

    # --- 4. Check azimuth range [0, 360] ---
    _azi_names = ("AZI", "AZIM", "AZ", "AZM", "AZIMUTH")
    for azi_name in _azi_names:
        if azi_name not in dev.columns:
            continue
        azi = dev.columns[azi_name]
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
        break  # Found one recognised azimuth column — done checking.

    # --- 5. Check inclination range [0, 180] ---
    _inc_names = ("INC", "INCL", "DEVI", "DIP")
    for inc_name in _inc_names:
        if inc_name not in dev.columns:
            continue
        inc = dev.columns[inc_name]
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
        break  # Found one recognised inclination column — done checking.

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
        LASEncodingError: If the explicit encoding parameter fails to decode
            the file.
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
        LASEncodingError: If the explicit encoding parameter fails to decode
            the file.

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
        raise DEVReadError(f"Cannot read file: {file_path}") from e
    except (ValueError, LookupError) as e:
        raise DEVReadError(f"Cannot read file: {file_path}") from e

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
        logger.warning(
            "Auto-detected headerless format for %s. "
            "The first data line has all-numeric tokens.  "
            "A matching column count on the second line would cause "
            "the file to be treated as simple format (numeric column "
            "names as headers).  If the file has a header row, "
            "consider explicit format specification.",
            file_path,
        )

    # --- Auto-detect delimiter ---
    # When auto-correction fires (comma-header + space-data), the original
    # comma-split header tokens must be cached before the delimiter switch
    # so Pass 2 uses correct column names instead of re-splitting the header
    # line with the wrong (space) delimiter.
    _cached_hdr_names: list[str] | None = None

    if delimiter is None:
        # Use the actual header line for delimiter detection:
        # - simple:   first content line
        # - dug:      third content line (the header — skip title+count)
        # - headerless: first content line (the data), but when the first
        #   line has no commas and a later line does (e.g. column-count
        #   prefix like "4\\n1.0,2.0,3.0,4.0"), use the first
        #   comma-containing line for delimiter detection.  (I2F-30)
        if format_type == "dug" and len(content_entries) > skip_content_lines - 1:
            hdr = content_entries[skip_content_lines - 1][1]
        elif content_entries:
            _first = content_entries[0][1]
            if format_type == "headerless" and "," not in _first:
                # Headerless format where the first line is a column count
                # (no commas).  Check subsequent lines for comma presence.
                _comma_hdr = next(
                    (entry[1] for entry in content_entries[1:3] if "," in entry[1]),
                    None,
                )
                hdr = _comma_hdr if _comma_hdr else _first
            else:
                hdr = _first
        else:
            hdr = ""

        if hdr:
            # If the header contains commas and splitting on comma
            # yields at least as many tokens as splitting on whitespace,
            # treat the file as comma-delimited.  Using >= (not >) so
            # that "MD, TVD, INC, AZI" (commas with trailing spaces)
            # correctly detects comma delimiter.
            comma_tokens = [t for t in hdr.split(",") if t.strip()]
            space_tokens = hdr.split()
            if len(comma_tokens) >= len(space_tokens) and len(comma_tokens) >= 2:
                delimiter = ","
            else:
                delimiter = " "
        else:
            delimiter = " "

        # Cross-validate auto-detected delimiter against first data line
        # (F-M27).  When the header uses commas but data lines are
        # space-delimited, the auto-detected comma delimiter produces a
        # single token from the entire data line, causing all values to
        # become NaN.  Skip for headerless — the header line IS the data.
        if hdr and format_type != "headerless":
            _content_seen = 0
            _first_data: str | None = None
            for _lin in lines:
                _s = _lin.strip()
                if not _s or _s.startswith("#"):
                    continue
                _content_seen += 1
                if _content_seen > skip_content_lines:
                    _first_data = _s
                    break
            if _first_data is not None:
                if delimiter == " ":
                    _hdr_cols = len(hdr.split())
                    _data_cols = len(_first_data.split())
                else:
                    _hdr_cols = len(
                        [t for t in hdr.split(delimiter) if t.strip()]
                    )
                    _data_cols = len(
                        [t for t in _first_data.split(delimiter) if t.strip()]
                    )
                if _hdr_cols >= 2 and _data_cols == 1:
                    # Auto-detected delimiter fails on first data line
                    # (produces only 1 token).  Try the alternative delimiter
                    # for auto-correction before raising an error.
                    _alt_delim = " " if delimiter == "," else ","
                    if _alt_delim == " ":
                        _data_alt_cols = len(_first_data.split())
                    else:
                        _data_alt_cols = len(
                            [t for t in _first_data.split(_alt_delim)
                             if t.strip()]
                        )
                    if _data_alt_cols >= 2 and _data_alt_cols == _hdr_cols:
                        logger.warning(
                            "Auto-corrected delimiter from %r to %r: "
                            "header has %d columns but first data line "
                            "produced only %d token(s) with %r delimiter.",
                            delimiter, _alt_delim, _hdr_cols,
                            _data_cols, delimiter,
                        )
                        # F-01: Cache header column names parsed with the
                        # ORIGINAL delimiter before switching.  Without this,
                        # Pass 2 re-parses the header line with the new
                        # (space) delimiter, collapsing a comma-delimited
                        # header like "MD,TVD,INC" into a single bogus
                        # column name.  The cached names are passed to both
                        # the simple-format and DUG-format header paths.
                        _cached_hdr_names = [t.strip() for t in hdr.split(delimiter) if t.strip()]
                        delimiter = _alt_delim
                    else:
                        raise DEVReadError(
                            f"Delimiter mismatch: auto-detected delimiter "
                            f"{delimiter!r} produces {_hdr_cols} columns "
                            f"from header but only {_data_cols} token(s) "
                            f"from first data line. Alternative delimiter "
                            f"{_alt_delim!r} does not match ({_data_alt_cols} "
                            f"tokens). Specify delimiter explicitly."
                        )
                elif abs(_hdr_cols - _data_cols) >= 3:
                    raise DEVReadError(
                        f"Delimiter mismatch: auto-detected delimiter "
                        f"{delimiter!r} produces {_hdr_cols} columns from "
                        f"header but {_data_cols} tokens from first data "
                        f"line (difference: {abs(_hdr_cols - _data_cols)}). "
                        f"Specify delimiter explicitly."
                    )

    # Guard against empty delimiter — str.split("") raises ValueError.
    # The auto-detection above always produces "," or " " but a caller
    # may pass delimiter="" explicitly, bypassing the is-None check.
    if not delimiter:
        raise DEVReadError(
            "Delimiter must be a non-empty string (e.g., ' ' for "
            "whitespace, ',' for comma). Received an empty string."
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

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        content_seen += 1

        if delimiter == " ":
            values = stripped.split(maxsplit=_max_tokens)
        else:
            values = _split_delimited_line(stripped, delimiter)

        if format_type == "headerless":
            if content_seen == 1:
                # Auto-generate column names from the first data row.
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
                        try:
                            dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                        except IndexError:
                            pass
                    current_line += 1
                else:
                    discarded_lines += 1
            else:
                # Remaining data rows (G-04 bounds guard).
                if current_line < data_lines:
                    if len(values) > len(names):
                        if extra_col_count is None:
                            logger.warning(
                                "Data line has %d values but only %d columns declared "
                                "in the header. Extra columns are discarded. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            logger.warning(
                                "Data line has %d values but %d columns declared "
                                "in the header. Missing values are filled with NaN. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        try:
                            dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                        except IndexError:
                            pass
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
                    # Filter out empty strings from trailing delimiters (e.g.,
                    # "MD,TVD," → ["MD","TVD",""]) so empty column names are
                    # rejected instead of creating dev.columns[""].
                    names = [v for v in values if v]
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
                            logger.warning(
                                "Data line has %d values but only %d columns declared "
                                "in the header. Extra columns are discarded. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            logger.warning(
                                "Data line has %d values but %d columns declared "
                                "in the header. Missing values are filled with NaN. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        try:
                            dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                        except IndexError:
                            pass
                    current_line += 1
                else:
                    discarded_lines += 1

        else:  # simple header format
            if content_seen == 1:
                # First non-comment line = column names.
                # Use cached header names if auto-correction changed
                # the delimiter (F-01 fix — prevents comma-header misparse).
                if _cached_hdr_names is not None:
                    values = _cached_hdr_names
                # Handle *COLUMNS keyword format (Petra/CPS variant)
                if _is_columns_header(values):
                    names = _parse_columns_tokens(values)
                else:
                    # Filter out empty strings from trailing delimiters (e.g.,
                    # "MD,TVD," → ["MD","TVD",""]) so empty column names are
                    # rejected instead of creating dev.columns[""].
                    names = [v for v in values if v]
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
                            logger.warning(
                                "Data line has %d values but only %d columns declared "
                                "in the header. Extra columns are discarded. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            extra_col_count = 1
                        else:
                            extra_col_count += 1
                    if len(values) < len(names):
                        if short_row_count is None:
                            logger.warning(
                                "Data line has %d values but %d columns declared "
                                "in the header. Missing values are filled with NaN. "
                                "(Subsequent occurrences will be counted; see summary.)",
                                len(values),
                                len(names),
                            )
                            short_row_count = 1
                        else:
                            short_row_count += 1
                    for k in range(min(len(values), len(names))):
                        try:
                            dev.columns[names[k]][current_line] = _to_finite_float(values[k], np.nan)
                        except IndexError:
                            pass
                    current_line += 1
                else:
                    discarded_lines += 1

    # --- Post-scan diagnostics (F-I2-DV-07) ---
    if discarded_lines > 0:
        logger.warning(
            "Pre-scan undercount: %d data line(s) discarded because the "
            "actual data exceeds the %d lines declared by Pass 1. "
            "Dev file data may be truncated.",
            discarded_lines,
            data_lines,
        )
    if current_line < data_lines:
        logger.warning(
            "Pre-scan overcount: declared %d data lines but only %d actual "
            "data lines found. Pre-allocated array tail contains NaN.",
            data_lines,
            current_line,
        )

    # F-I2-XPD-03: Summary of extra-column and short-row occurrences,
    # replacing the previous boolean-once pattern that suppressed all
    # diagnostics after the first row.  This allows automated data
    # quality tools to enumerate affected rows.
    if extra_col_count is not None and extra_col_count > 1:
        logger.warning(
            "%d data line(s) had more values than expected. "
            "Extra columns were discarded.",
            extra_col_count,
        )
    if short_row_count is not None and short_row_count > 1:
        logger.warning(
            "%d data line(s) had fewer values than expected. "
            "Missing values were filled with NaN.",
            short_row_count,
        )

    _validate_dev_data(dev, _stacklevel=3)
    return dev
