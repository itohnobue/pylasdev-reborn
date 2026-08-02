"""ASCII data section reader for LAS files.

Handles both normal and wrapped modes.
Replaces las_line_reader.py with corrected wrapped-mode logic
and O(n) performance (vs O(n^2) numpy.append bug in original).
"""

from __future__ import annotations

import logging
import math
import warnings
from types import ModuleType
from typing import cast

import numpy as np

from .exceptions import LASParseError
from .models import LASFile, WellSection, _GuardedDict

logger = logging.getLogger(__name__)

# Maximum bounds for array allocations to prevent memory exhaustion
# from malformed or malicious files. Overridable by setting the module constant.
MAX_CURVES = 100_000
MAX_DATA_LINES = 10_000_000
# Combined allocation guard: curve_count x data_line_count must not exceed this.
# Individual MAX_CURVES and MAX_DATA_LINES checks alone are insufficient — a file
# with 1K curves x 1M lines (1B elements ≈ 8 GB) passes both guards independently
# but OOMs during np.zeros pre-allocation. Overridable by setting module constant.
MAX_TOTAL_ELEMENTS = 1_000_000_000
# DR-05: Python-object accumulation cap for string data on the LAS 1.2/2.0
# paths.  MAX_TOTAL_ELEMENTS guards numpy array allocation at ~8 B/element
# (~8 GB intent), but each Python str object stored in an object-dtype array
# or list costs ~50-100 B (measured) — a crafted file that passes the element
# guard can amplify to ~50-100 GB.  The LAS 3.0 path already caps string
# values at MAX_TOTAL_ELEMENTS // 12 (_las30_data._MAX_STRING_VALUES); this
# is the equivalent cap for _read_normal / _read_wrapped string accumulation.
# When the cap is exceeded the file is malformed/crafted — reject like the
# other MAX_* guards.  Overridable by setting the module constant.
MAX_STRING_VALUES = MAX_TOTAL_ELEMENTS // 12
# G-18: Per-line token limit to prevent single-line memory DoS via unbounded
# split().  Existing guards (MAX_DATA_LINES, MAX_CURVES, MAX_TOTAL_ELEMENTS)
# are product-based and do not protect against a single line with millions
# of space-separated tokens.  Legitimate LAS files never exceed MAX_CURVES
# tokens per line — this cap matches the curve limit.  Overridable.
# F-MDR-03: Sentinel — resolves to MAX_CURVES at call time.  Setting this
# to an explicit int overrides the default curve-count-based limit.
# Previously was ``= MAX_CURVES`` (import-time snapshot), which caused the
# documented "Overridable" behavior to break when MAX_CURVES is overridden.
MAX_TOKENS_PER_LINE: int | None = None

# M-06/M-62: int64 representability bounds for {I} integer curves.  The
# int64 storage branch must reject values (data values OR the null
# sentinel) outside this range — numpy int64 array assignment of a larger
# Python int raises OverflowError, which would escape the reader's
# LASParseError-only boundary.  Huge integral sentinels (>= 2^63) route
# to the object-dtype path instead; huge data values are replaced with
# the null sentinel and counted as conversion failures.
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _resolve_max_tokens_per_line() -> int:
    """Return the per-line token limit, resolving from MAX_CURVES if not
    explicitly overridden via MAX_TOKENS_PER_LINE.

    The indirection allows MAX_CURVES to be overridden at runtime
    (documented behavior) and have MAX_TOKENS_PER_LINE follow automatically.
    Users who need a different per-line cap set MAX_TOKENS_PER_LINE directly.
    """
    return MAX_TOKENS_PER_LINE if MAX_TOKENS_PER_LINE is not None else MAX_CURVES


from ._data_section_reader import (  # noqa: E402
    _detect_string_curves,
    _is_ascii_section,
    _is_section_header,
    _iter_ascii_data_lines,
    _log_conversion_failures,
    _split_data_line,
)


def _parse_float_with_d_notation(value_str: str) -> float:
    """Convert a string to float, handling Fortran D-notation.

    Some scientific software writes numbers in D-exponent format
    (e.g., ``"-999.25D0"``) instead of standard E-notation.  Python's
    ``float()`` only understands E/e exponents; this helper replaces
    D/d with E/e before conversion.

    This is a shared helper used by both :func:`_get_null_value` (for
    the NULL sentinel value in the well section) and :func:`_to_finite_float`
    (for data values in the ASCII section).

    Args:
        value_str: String to convert (e.g. ``"-999.25D0"``).

    Returns:
        Float value.

    Raises:
        ValueError: If the string cannot be parsed as a float after
            D→E conversion.
        OverflowError: If the string represents a number whose absolute
            value exceeds the maximum representable finite float.
    """
    return float(value_str.replace("D", "E").replace("d", "e"))


def _get_parser_module() -> ModuleType:
    """Lazy-import the parser module to break circular import.

    data_reader is imported by parser, but data_reader needs access
    to parser's module-level attributes (e.g., _DESANITIZE_ENABLED)
    for unified desanitize state.  Lazy import defers the import to
    call time, avoiding the cycle.
    """
    from . import parser as _p

    return _p


def _desanitize_las_value(value: str, _enabled: bool | None = None) -> str:
    """Reverse the writer's ``_``-prefix-on-``#`` escape (local copy).

    The writer prefixes ``#``-starting values with ``_`` to prevent the
    parser from interpreting them as comment lines.  This function strips
    that prefix, restoring the original ``#``-prefixed value.

    Defined locally in data_reader to avoid circular import from parser.

    IT3-F-01 (perf): *enabled* hoists the ``_DESANITIZE_ENABLED`` flag
    lookup.  ``read_ascii_data`` caches the thread-local flag once per
    read (it is constant for the duration of the read — E-04 sets it at
    the top of the try and restores it in finally) and passes the cached
    value through ``_read_normal``/``_read_wrapped``.  Without the hoist,
    every data value re-ran the full import-machinery path
    (``_get_parser_module`` → import → ``__getattr__`` →
    ``_is_desanitize_enabled``), measured at ~1.04 µs/value.  When
    *enabled* is None (any external caller), fall back to the machinery
    lookup to preserve the previous behavior.
    """
    if _enabled is None:
        _enabled = bool(_get_parser_module()._DESANITIZE_ENABLED)
    if not _enabled:
        return value
    if value.startswith("_#"):
        return value[1:]
    idx = value.find("_#")
    if idx > 0 and value[idx - 1].isspace():
        return value[:idx] + value[idx + 1 :]
    return value


def _get_well_entry_ci(
    well: WellSection | dict[str, str],
    key: str,
    default: str,
) -> str:
    """Case-insensitive lookup in a well section / dict (IT3-THR-01).

    from_dict stores well keys verbatim when ``mnem_base`` is None
    (``_norm_mnem`` identity), so a case-variant key like ``"null"``
    survives construction.  The parser, by contrast, uppercases well
    mnemonics on read (parser.py:1937 area).  A case-sensitive lookup
    for ``"NULL"`` misses the lowercase key, so the writer declares
    NULL=-1 in ~W but fills data rows with the -999.25 default — the
    re-read file's declared NULL then disagrees with its fill cells and
    downstream consumers read fill cells as real data.  This helper makes
    all ``_get_null_value`` call sites (reader, LAS 3.0 reader, both
    writers) agree on the declared sentinel regardless of key case.
    """
    entries = well.entries if isinstance(well, WellSection) else well
    key_upper = key.upper()
    for k, v in entries.items():
        if str(k).upper() == key_upper:
            return v
    return default


def _get_null_value(
    well: WellSection | dict[str, str],
    default: str = "-999.25",
    default_float: float = -999.25,
) -> float:
    """Extract null value from well section or return default.

    Shared utility used by parser, data_reader, and writer to avoid
    duplicated try/except blocks across four call sites.

    Args:
        well: WellSection or dict-like object with a .get() method.
        default: String default to look up (e.g. '-999.25').
        default_float: Float fallback when conversion fails.

    Returns:
        Null value as a float.
    """
    try:
        # IT3-THR-01: case-insensitive NULL lookup (see _get_well_entry_ci)
        # so declared NULL and the fill sentinel always agree.
        null_value = _parse_float_with_d_notation(_get_well_entry_ci(well, "NULL", default))
        # F-04: Reject non-finite sentinel values (NaN, Inf, -Inf) which
        # float() accepts without error.  These propagate through numpy
        # arrays → corrupted statistics → writer outputs "nan" (invalid LAS).
        if not np.isfinite(null_value):
            raise LASParseError(f"NULL value must be a finite number, got {null_value!r}")
        return null_value
    except OverflowError:
        # DR-02: Python <=3.12 float('1e400') raises OverflowError instead
        # of returning inf.  Match the F-04 finite-guard behavior below (the
        # 3.13+ path) so the documented LASParseError boundary holds on all
        # supported Pythons; falling back to the default here would diverge
        # per-version (3.12 falls back to the sentinel, 3.13 raises).
        raise LASParseError(
            "NULL value must be a finite number, got a value that overflows the float range"
        ) from None
    except (ValueError, TypeError, AttributeError):
        # F-I2-XPD-05: Log a warning when falling back to the default
        # null value.  Silent fallback to -999.25 with zero diagnostics
        # makes it impossible to distinguish a genuine -999.25 null
        # sentinel from a failed parse.
        warnings.warn(
            f"Could not parse NULL value from well section; "
            f"falling back to default value {default_float:.2f}.",
            UserWarning,
            stacklevel=2,
        )
        return default_float


def _detect_integer_curves(las_file: LASFile) -> set[int]:
    """Return set of curve indices whose CurveDefinition has integer format.

    LAS 3.0's ``{I}`` format specifier declares integer-typed data.  Storing
    such values as float64 silently loses precision above 2^53
    (L-03): ``float("9007199254740993")`` rounds to 9007199254740992.0.
    Parsing via ``int()`` and storing in an int64 array preserves the exact
    value.  This mirrors ``_detect_string_curves`` for the {S} format.

    Args:
        las_file: LASFile with curves and curves_order populated.

    Returns:
        Set of integer indices into curves_order for integer-format curves.
    """
    _integer_curve_indices: set[int] = set()
    for _idx in range(len(las_file.curves_order)):
        if _idx < len(las_file.curves):
            cd = las_file.curves[_idx]
            # Normalize to uppercase for consistent detection (same as
            # _detect_string_curves F-MDR-04).
            if cd.data_format.upper() == "I":
                _integer_curve_indices.add(_idx)
    return _integer_curve_indices


def _to_integer_value(
    value_str: str,
    null_value: float,
    _failure_counter: list[int] | None = None,
    _null_as_float: bool = False,
) -> int | float:
    """Convert string to exact integer for {I}-format curves (L-03).

    The {I} (integer) data format can carry values above 2^53 where float64
    cannot represent every integer exactly (e.g. 9007199254740993).  Parsing
    via ``int()`` preserves the exact value; parsing via ``float()`` rounds
    it to the nearest representable double.  This helper is used only for
    integer-format curves that were allocated with int64 dtype (the caller
    guarantees *null_value* is integral in that case).

    Args:
        value_str: String to convert.  May be empty.
        null_value: Integral null sentinel to return on failure.
        _failure_counter: Optional mutable list; ``_failure_counter[0]`` is
            incremented on each non-trivial conversion failure.
        _null_as_float: When True (EXT-04: {I} curves stored as object
            dtype because the declared NULL is fractional), return the
            fractional *null_value* itself on failure instead of
            ``int(null_value)`` — ``int(-999.25)`` would truncate the
            sentinel, corrupting null cells.

    Returns:
        Exact integer, or ``int(null_value)`` (or *null_value* when
        ``_null_as_float``) on failure.
    """
    if not value_str:
        return null_value if _null_as_float else int(null_value)
    try:
        # int() preserves exactness for plain integer tokens.  This is the
        # common {I} case and the whole point of the dtype branch.
        result = int(value_str)
    except ValueError:
        result = None
    if result is not None:
        # M-06: On the int64 storage path (integral NULL), a parsed value
        # beyond int64 range cannot be assigned to the int64 array —
        # numpy raises OverflowError, which would escape the reader's
        # LASParseError-only boundary.  Route to the null sentinel and
        # count as a conversion failure (the existing summary warning
        # fires) instead of letting the raw exception escape.  The object
        # dtype path (fractional NULL) holds arbitrary Python ints, so no
        # bound applies there.
        if not _null_as_float and not (_INT64_MIN <= result <= _INT64_MAX):
            if _failure_counter is not None:
                _failure_counter[0] += 1
            return int(null_value)
        return result
    # Fall back to D-notation / float parsing for non-plain tokens.  Only
    # accept the result when it is exactly integral — fractional values in
    # an integer column are conversion failures (null), never truncated.
    try:
        val = _parse_float_with_d_notation(value_str)
    except (ValueError, OverflowError):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    if not math.isfinite(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    if val != int(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value if _null_as_float else int(null_value)
    # M-06: bound the float-fallback result on the int64 storage path too
    # (e.g. "9.2e18" parsed via float() then int() can exceed int64 max).
    result = int(val)
    if not _null_as_float and not (_INT64_MIN <= result <= _INT64_MAX):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return int(null_value)
    return result


def _to_finite_float(
    value_str: str,
    null_value: float,
    _failure_counter: list[int] | None = None,
) -> float:
    """Convert string to float, replacing non-finite values with null_value.

    Python's ``float()`` accepts ``\"nan\"``, ``\"inf\"``, ``\"-inf\"`` and
    overflow exponents (e.g. ``\"1e309\"``) without error.  These non-finite
    values corrupt downstream numpy computations (NaN propagation, Inf
    making statistics invalid).  This helper catches them and returns
    *null_value* instead.

    Also handles empty strings and non-numeric strings gracefully.

    Args:
        value_str: String to convert.  May be empty.
        null_value: Value to return when conversion fails or result is
            non-finite.
        _failure_counter: Optional mutable list; ``_failure_counter[0]`` is
            incremented on each non-trivial conversion failure (non-empty
            input that could not be parsed as a finite float).  Used by
            callers to surface diagnostic counts.

    Returns:
        A finite float, or *null_value*.
    """
    if not value_str:
        return null_value
    try:
        val = _parse_float_with_d_notation(value_str)
    except (ValueError, OverflowError):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value
    # IT3-F-02: math.isfinite is ~15x faster than np.isfinite for Python
    # float scalars and semantically identical for scalars (verified: no
    # NaN-propagation divergence).  This helper is called once per data
    # value on every read path (data_reader, _las30_data, dev_reader via
    # _to_finite_float), so the swap has measurable end-to-end impact.
    if not math.isfinite(val):
        if _failure_counter is not None:
            _failure_counter[0] += 1
        return null_value
    return val


def read_ascii_data(
    lines: list[str], las_file: LASFile, data_line_count: int, desanitize: bool = True
) -> None:
    """Read the ~A (ASCII data) section and populate las_file.logs.

    Args:
        lines: File content split into lines (pre-split by reader.py
            for efficiency — eliminates redundant content.splitlines()).
        las_file: LASFile object with curves_order already populated.
        data_line_count: Number of data lines (from pre-scan).
        desanitize: If True (default), strip writer escape prefixes
            (_# → #) for roundtrip correctness.  Set to False when
            reading files NOT produced by pylasdev's writer to avoid
            corrupting genuine _#-prefixed data.
    """
    if len(las_file.curves_order) == 0:
        # F32: Warn when data is present but no curves are defined,
        # so malformed files don't silently lose data.
        warnings.warn(
            "ASCII data section found but no curves are defined. Data has been discarded.",
            UserWarning,
            stacklevel=2,
        )
        return

    # F-ITER2-SEC-M02 + F-ITER2-SEC-M03: Deduplicate curves BEFORE bounds
    # checks and wrap detection.  Previously, _detect_actual_wrap and the
    # outer MAX_TOTAL_ELEMENTS check used pre-dedup curve_count, causing
    # false wrap detection and false rejection for files with duplicate
    # curves near the boundary.  _read_normal and _read_wrapped also call
    # _deduplicate_curves internally; the second call is a no-op when
    # curves are already unique.
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # F-M01: Use ``>=`` (not ``>``) for consistency with parser.py and models.py
    # which already use ``>=`` for all MAX_CURVES guards.  Using ``>`` here
    # allows exactly MAX_CURVES curves while from_dict rejects at the limit —
    # a roundtrip divergence where a data_reader-accepted file is rejected by
    # from_dict.  Synchronizing on ``>=`` closes the gap.
    if curve_count >= MAX_CURVES:
        raise LASParseError(
            f"Curve count ({curve_count}) exceeds maximum allowed ({MAX_CURVES}). "
            f"The file may be malformed or corrupt."
        )

    # F-D2-M03: Removed the outer MAX_TOTAL_ELEMENTS check that ran before
    # wrap detection.  For WRAP=YES files, data_line_count counts individual
    # lines (not depth steps), so a wrapped file with 100 curves and 10M
    # lines (100K depth steps) would falsely be rejected as 1B elements
    # when it's actually ~10M.  Both _read_normal and _read_wrapped
    # have their own wrapped-aware bounds checks.  For WRAP=NO, this
    # simply eliminates a redundant check (the same bounds guard runs
    # inside _read_normal).

    # F-212: Route the desanitize parameter to the unified parser flag
    # so _desanitize_las_value and parser._desanitize_las_value share the
    # same module-level state.
    # E-04: Save the prior thread-local value and restore it in a finally
    # block.  The flag is thread-local (parser.py F-21/F-088); an
    # unconditional reset-to-True would clobber another caller's value on
    # the same thread.  Without the restore, a desanitize=False read left
    # the flag False for the whole thread, silently changing the behavior
    # of subsequent direct LASParser.parse()/read_ascii_data() users.
    _parser_mod = _get_parser_module()
    _prev_desanitize = _parser_mod._DESANITIZE_ENABLED
    _parser_mod._DESANITIZE_ENABLED = desanitize  # type: ignore[attr-defined]
    try:
        delimiter = las_file.version.delimiter_char

        # F-H01: Unconditionally detect actual wrap mode from data content
        # regardless of the WRAP header flag.  The header may be wrong
        # (e.g. mislabeled Petrel exports claim WRAP=YES but data is
        # non-wrapped, or files with WRAP=NO actually contain wrapped data).
        # Using the wrong reader corrupts all non-depth curve values:
        #   - _read_normal on wrapped data  → extra columns discarded,
        #     only first value per line used, rest become null_value
        #   - _read_wrapped on non-wrapped  → depth/C1 swap via the
        #     depth-line flag protocol treating full lines as single values
        actual_wrap = _detect_actual_wrap(
            lines, curve_count, delimiter, declared_wrap=las_file.version.wrap
        )
        # IT3-F-01: The desanitize flag is constant for this read (E-04
        # sets it above and restores it in finally).  Cache it once here
        # and thread it through the per-value loops instead of re-running
        # the full import-machinery lookup per value.
        _desanitize_enabled = bool(_parser_mod._DESANITIZE_ENABLED)
        if actual_wrap:
            _read_wrapped(
                lines,
                las_file,
                curve_count,
                delimiter,
                _desanitize_enabled=_desanitize_enabled,
            )
        else:
            _read_normal(
                lines,
                las_file,
                curve_count,
                data_line_count,
                delimiter,
                _desanitize_enabled=_desanitize_enabled,
            )
    finally:
        _parser_mod._DESANITIZE_ENABLED = _prev_desanitize  # type: ignore[attr-defined]


def _detect_actual_wrap(
    lines: list[str],
    curve_count: int,
    delimiter: str = " ",
    declared_wrap: str | None = None,
) -> bool:
    """Detect if data is actually wrapped by checking the first data lines.

    In true wrapped mode, the first data line has only 1 value (the depth).
    In non-wrapped mode (even if WRAP=YES header), each line has >= curve_count values.

    Protocol-based detection (D-01/D-02/D-03): instead of relying on a
    two-line heuristic, examine up to four data lines and take a
    curve_count-aware majority vote:

    - A line is "full" (non-wrapped evidence) when it carries the complete
      row: ``len >= curve_count`` for every delimiter (COMMA/TAB trailing
      empties are stripped by _split_data_line, so a wrapped depth line is
      exactly 1 value and a wrapped continuation line carries
      ``curve_count-1`` values — both are partial evidence — F-023,
      EXT-01).
    - A line is "partial" (wrapped evidence) otherwise.

    Decision:
    - First line full → non-wrapped immediately (a wrapped first line is
      always a depth line with exactly 1 value).
    - Otherwise, if >= 2 full lines appear among the first 4 data lines,
      the file is non-wrapped (sparse leading rows then full rows — D-01's
      trailing-comma first row and D-02's two sparse rows both produce
      full rows quickly).
    - If >= 3 partial lines appear (and < 2 full), the file is wrapped —
      this covers D-03's genuine WRAP=YES file whose second line is
      overfull (full_count stays 1 while depth lines recur).
    - Ties (e.g. exactly 2 data lines: one full, one partial) fall back to
      the declared WRAP header as the tiebreak, then to wrapped.

    Args:
        lines: File content split into lines.
        curve_count: Number of curves declared in ~C section.
        delimiter: Data column delimiter character (default space).
            Uses DLM-aware splitting when delimiter is not a space.
        declared_wrap: Optional declared WRAP header value ("YES"/"NO").
            Used only as a tiebreak when the data window is genuinely
            ambiguous (2-2 or 1-1 split).  None means the header is
            unavailable — default to wrapped (conservative).

    Returns:
        True if data is actually wrapped, False if non-wrapped despite header.
    """
    in_ascii = False
    window: list[int] = []  # value counts of up to 4 data lines
    for line in lines:
        stripped = line.strip()

        if _is_section_header(stripped):
            # P-16: Treat every genuine ~-prefixed section header (recognized
            # OR unrecognized) as a section boundary, matching the parser's
            # section-boundary classification.  Previously unrecognized words
            # were skipped (continue) while in_ascii stayed True, so the body
            # lines of an unrecognized section (e.g. ~CUSTOMSECT) were counted
            # as data for wrap detection.  F-I2-XPD-01 retained: only genuine
            # ~-prefixed section-like lines (checked by _is_section_header)
            # terminate the block; control-char noise (~3D, ~., ~#) fails
            # that check and is skipped below.
            if _is_ascii_section(stripped):
                in_ascii = True
            elif in_ascii:
                # F-048: Standardize section-detection guard with
                # _iter_ascii_data_lines (uses return/break instead of
                # just resetting in_ascii).  When we encounter a genuine
                # non-~A section header while inside an ~A block, exit the
                # loop — we've left the data section and no more data lines
                # should be used for wrap detection.
                break
            continue

        # P-16: ~-prefixed lines that are NOT section headers (e.g. ~3D,
        # ~., ~#, control-character replacement artifacts) are not data
        # lines — the parser routes them to other_lines.  Skip them.
        if stripped.startswith("~"):
            continue

        if not in_ascii or not stripped or stripped.startswith("#"):
            continue

        # Data line found — split using DLM-aware split (shared utility).
        # See _data_section_reader._split_data_line for full rationale.
        values = _split_data_line(stripped, delimiter)

        # F-M20: When curve_count is 1, wrapped and non-wrapped modes are
        # equivalent — every line holds exactly one value regardless of
        # mode.  The space-delimiter heuristic ``len(values) < curve_count``
        # is degenerate for curve_count=1 (always False), and forcing
        # wrapped mode on a single-curve file triggers unnecessary overflow
        # warnings and counter logic in _read_wrapped.  Use non-wrapped
        # (the simpler path) when there is nothing to distinguish.
        if curve_count <= 1:
            return False

        window.append(len(values))
        if len(window) >= 4:
            break

    if not window:
        return True  # No data found, default to wrapped

    def _is_full(n: int) -> bool:
        if delimiter == " ":
            return n >= curve_count
        # COMMA/TAB: trailing empties are stripped by _split_data_line, so
        # value counts reflect real tokens.  A data line is "full" only
        # when it carries the complete row (curve_count values) — a
        # wrapped continuation line carries curve_count-1 values and a
        # depth-only line carries exactly 1, so both are partial (wrapped)
        # evidence.  (EXT-01: the previous ``n > 1`` predicate treated
        # wrapped COMMA/TAB files with >=3 curves as non-wrapped because
        # their continuation lines carry >=2 values, silently corrupting
        # DEPT/curve alignment.)
        return n >= curve_count

    # F-07 (DR-01/I2-04): depth-line evidence rule.  A genuine wrapped
    # file ALWAYS has depth lines (rows with exactly 1 value); a
    # non-wrapped file essentially never has a mid-window 1-value row (a
    # mnemonic-header masquerade IS wrapped evidence, DR-01a).  When any
    # later window line carries exactly 1 value:
    #   - declared WRAP=YES → wrapped (I2-04: [3,3,1,1] — two full
    #     leading rows no longer beat the declaration + depth evidence)
    #   - first line full AND the depth evidence is unambiguous → wrapped
    #     (DR-01 both triggers: mixed-wrap / mnemonic-header masquerade
    #     [3,1,1] / [3,1,2,1] with WRAP=NO or absent — content outranks a
    #     NO header).  "Unambiguous" means a 1-value row immediately
    #     after the full first row, OR at least two 1-value rows in the
    #     window — a single trailing 1-value row after short rows
    #     ([3,2,1]) is a ragged non-wrapped row and must stay non-wrapped
    #     (graceful short-row null-fill).
    #     REFINEMENT (mirrors _las30_data.py, I2-03 contract divergence):
    #     for curve_count == 2 a 1-value row is AMBIGUOUS — a wrapped
    #     continuation line also carries curve_count-1 == 1 value, so a
    #     single 1-value row right after the full first row ([2,1,2], a
    #     string-padding file where the second row has one missing column)
    #     must NOT be treated as unambiguous depth evidence.  The
    #     ≥2-one-value-rows arm still catches genuine 2-curve wrapped
    #     files ([2,1,1,1] mixed-wrap: depth + continuation pairs).  For
    #     curve_count >= 3 a 1-value row can ONLY be a depth line
    #     (continuations carry >= 2 values) → the window[1]==1 arm is
    #     unambiguous there.
    # Otherwise fall through to the existing rules unchanged.
    depth_later = len(window) > 1 and any(n == 1 for n in window[1:])
    if depth_later:
        if declared_wrap is not None and declared_wrap.upper() == "YES":
            return True
        if _is_full(window[0]) and (
            (curve_count >= 3 and window[1] == 1) or sum(1 for n in window[1:] if n == 1) >= 2
        ):
            return True

    # First line full → non-wrapped (wrapped first line is always depth).
    # M-38: BUT a WRAP=YES header with a COMPLETE first row can still be a
    # genuine mixed-wrap file (per-row line-width wrapping): first row
    # complete, continuation lines wrapped.  The first-line-full rule
    # serves the COMMON mislabeled case (WRAP=YES but data is fully
    # non-wrapped — all lines full).  When the header declares WRAP=YES
    # and later window lines are partial (continuation/depth evidence),
    # fall through to the majority vote instead of short-circuiting to
    # non-wrapped.
    if _is_full(window[0]):
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
    # non-wrapped so _read_normal's graceful short-row null-fill preserves
    # the data.  (Requires >=2 lines: a single short row is too ambiguous
    # to override the declared header tiebreak.)
    if (
        len(window) >= 2
        and full_count == 0
        and len(set(window)) == 1
        and 1 < window[0] < curve_count
    ):
        return False

    # Two full rows among the first 4 → definitively non-wrapped (D-01
    # trailing-comma first row, D-02 two sparse rows, then full rows).
    if full_count >= 2:
        return False
    # At least 3 partial rows and fewer than 2 full → wrapped.  A genuine
    # WRAP=YES file with an overfull second line (D-03) has full_count 1
    # while depth lines recur as partial rows.
    if partial_count >= 3:
        return True
    # Ambiguous window (e.g. 2-2 or 1-1): use the declared header as the
    # tiebreak, else default to wrapped (conservative).
    if declared_wrap is not None:
        return declared_wrap.upper() == "YES"
    return True


def _deduplicate_curves(las_file: LASFile, _stacklevel: int = 2) -> None:
    """Detect and rename duplicate curve names with warning.

    Appends _2, _3, etc. to duplicate mnemonics so each curve gets
    its own array in las_file.logs. Also updates the corresponding
    CurveDefinition objects to keep curves_order and curves in sync.

    Args:
        las_file: LASFile to deduplicate curves in.
        _stacklevel: Stacklevel for warnings.warn (default 2 points
            to the immediate caller; pass 3 when called from deeper
            call chains such as parser._process_ascii_data).
    """
    seen: dict[str, int] = {}
    new_order: list[str] = []
    # Track all names in the output order for collision detection (F-22)
    output_names: set[str] = set()
    for idx, name in enumerate(las_file.curves_order):
        if name in seen:
            seen[name] += 1
            new_name = _resolve_unique_curve_name(name, seen[name], output_names)
            # Update the seen counter to match the actual suffix used
            seen[name] = _suffix_from_name(new_name, name)
            _rename_duplicate_curve(
                las_file,
                idx,
                name,
                new_name,
                new_order,
                output_names,
                _stacklevel,
            )
        else:
            # F-22: Check for cross-base collisions where an
            # original name matches a previously generated _N suffix.
            # Input ["DEPT","DEPT","DEPT_2"] should produce
            # ["DEPT","DEPT_2","DEPT_2_2"], not
            # ["DEPT","DEPT_2","DEPT_2"] with duplicate keys.
            if name in output_names:
                new_name = _resolve_unique_curve_name(name, 2, output_names)
                seen[name] = _suffix_from_name(new_name, name)
                _rename_duplicate_curve(
                    las_file,
                    idx,
                    name,
                    new_name,
                    new_order,
                    output_names,
                    _stacklevel,
                )
            else:
                seen[name] = 1
                new_order.append(name)
                output_names.add(name)
    if new_order != las_file.curves_order:
        las_file.curves_order = new_order


def _resolve_unique_curve_name(
    base_name: str,
    start_suffix: int,
    output_names: set[str],
) -> str:
    """Resolve a unique curve name by appending incrementing _N suffix.

    F-M19: Extracted helper for the deduplication collision-resolution
    algorithm shared across parser.py, data_reader.py, and dev_reader.py.

    Args:
        base_name: The original curve mnemonic.
        start_suffix: Starting integer for the suffix (e.g. 2 for first
            duplicate, or higher for already-seen names).
        output_names: Set of names already in the output (for collision
            detection against previously-generated _N suffixes).

    Returns:
        A unique name of the form ``f"{base_name}_{N}"`` that does not
        collide with any name in *output_names*.
    """
    suffix = start_suffix
    new_name = f"{base_name}_{suffix}"
    while new_name in output_names:
        suffix += 1
        new_name = f"{base_name}_{suffix}"
    return new_name


def _suffix_from_name(name: str, base_name: str) -> int:
    """Extract the _N suffix from a resolved unique curve name.

    Given ``name = "DEPT_3"`` and ``base_name = "DEPT"``, returns 3.
    Used to keep the ``seen`` counter in sync with the actual suffix
    assigned by :func:`_resolve_unique_curve_name`.
    """
    try:
        return int(name[len(base_name) + 1 :])
    except (ValueError, IndexError):
        return 0


def _rename_duplicate_curve(
    las_file: LASFile,
    idx: int,
    name: str,
    new_name: str,
    new_order: list[str],
    output_names: set[str],
    _stacklevel: int = 2,
) -> None:
    """Rename a duplicate curve with warning and keep CurveDefinition in sync."""
    # N-I-30: The old warning claimed the FILE contained repeated curve
    # names, which is misleading when the duplicates were produced by
    # mnem_base resolving DISTINCT vendor mnemonics (e.g. "LLD" and "LLS"
    # both → "BK" → "BFV") to the same canonical name.  Mention both
    # causes so the message is accurate for mnem_base-active reads.
    warnings.warn(
        f"Duplicate curve mnemonic '{name}' renamed to '{new_name}'. "
        "This may be caused by mnem_base resolving distinct curve names "
        "to the same canonical name, or by repeated curve names in the "
        "source file.",
        stacklevel=_stacklevel,
    )
    new_order.append(new_name)
    output_names.add(new_name)
    if idx < len(las_file.curves):
        if not las_file.curves[idx].original_mnemonic:
            las_file.curves[idx].original_mnemonic = name
        las_file.curves[idx].mnemonic = new_name


def _mnemonic_header_declared(las_file: LASFile) -> set[str]:
    """Build the mnemonic-header match set once per read.

    The set contains the RESOLVED curve mnemonics (``curves_order``) plus
    each curve's ``original_mnemonic`` — mnem_base-normalized curves keep
    their vendor name (e.g. ``LLD``→``BFV`` with ``original_mnemonic="LLD"``),
    so a header row written in raw vendor mnemonics is still recognized.
    Mirrors ``parser._is_standalone_mnemonic_header`` (parser.py:3004-3008).

    Must be called AFTER ``_deduplicate_curves`` so ``_2``-suffix renames
    and their ``original_mnemonic`` values are in place.
    """
    declared = {name.upper() for name in las_file.curves_order}
    for idx, curve in enumerate(las_file.curves):
        if idx >= len(las_file.curves_order):
            break
        if curve.original_mnemonic:
            declared.add(curve.original_mnemonic.upper())
    return declared


def _is_mnemonic_header_row(
    values: list[str],
    las_file: LASFile,
    curve_count: int,
    string_curve_indices: set[int],
    *,
    declared: set[str] | None = None,
) -> bool:
    """M-37/FIX-CONV-2: True when *values* are exactly the declared curve mnemonics.

    LAS 2.0 places curve mnemonics ON the ~A line, but some real-world
    files emit them as a standalone header row immediately after ~A
    (e.g. ``~A\\nDEPT GR\\n1000.0 50.0\\n...``).  Such a row is a column
    header, not a data row: consuming it creates a phantom all-null first
    row and shifts every subsequent value by one column.

    Detection mirrors the LAS 3.0 gold standard
    ``parser._is_standalone_mnemonic_header`` (parser.py:2991-3009) for the
    LAS 1.2/2.0 whole-file scope, with three clauses:

    1. **Token-count equality** (parser.py:2991 analog): ``len(values)``
       must equal ``curve_count``.  A wrapped-mode continuation row carries
       fewer tokens and can never be a full header signature — without this,
       a string value that coincides with a curve mnemonic (e.g. ``LITH``)
       would be wrongly skipped as a header (M-02).
    2. **All-string exclusion** (parser.py:3002 analog): when every curve in
       the section is a string curve, every data value is a string, so a
       mnemonic-coincident value is indistinguishable from a header row by
       content alone — string data rows are never dropped (M-03, F-19).
    3. **Match set = resolved + original mnemonics**: every token must match
       a resolved curve mnemonic or a curve's ``original_mnemonic``
       (parser.py:3004-3008 analog), so mnem_base-normalized header rows
       written in raw vendor mnemonics are recognized (M-01).

    *declared* is the precomputed match set from :func:`_mnemonic_header_declared`;
    when omitted it is built on demand.  Callers on the hot path hoist it.
    """
    if not values:
        return False
    if len(values) != curve_count:
        return False
    if len(string_curve_indices) == curve_count:
        return False
    if declared is None:
        declared = _mnemonic_header_declared(las_file)
    return all(token.upper() in declared for token in values)


def _read_normal(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    data_line_count: int,
    delimiter: str = " ",
    _desanitize_enabled: bool | None = None,
) -> None:
    """Read non-wrapped ASCII data. One depth step per line."""
    # Deduplicate curve names before allocating arrays
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # F-MDR-01: Use ``>`` for consistency with models.py which uses ``>`` at
    # all 8 MAX_DATA_LINES guard sites (accepts at exactly MAX_DATA_LINES).
    # The F-M01 fix comment previously claimed models.py uses ``>=``, which
    # was incorrect — the divergence was introduced by the F-M01 change.
    if data_line_count > MAX_DATA_LINES:
        raise LASParseError(
            f"Data line count ({data_line_count}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks.
    if curve_count * data_line_count > MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Total allocation ({curve_count} curves x {data_line_count} lines = "
            f"{curve_count * data_line_count} elements) exceeds maximum allowed "
            f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    # F-WXP-01: Detect string curves from CurveDefinition data_format.
    # See _data_section_reader._detect_string_curves for full details.
    _string_curve_indices: set[int] = _detect_string_curves(las_file)
    # L-03: Detect {I} integer-format curves.  These are stored as int64
    # (when the null sentinel is integral) and parsed via int() so values
    # above 2^53 are not silently rounded by float().
    _integer_curve_indices: set[int] = _detect_integer_curves(las_file)
    # FIX-CONV-2: Hoist the mnemonic-header match set (resolved + original
    # mnemonics) once per read — building it per data row would make the
    # hot loop O(rows x curves) for set construction.
    _mnemonic_declared = _mnemonic_header_declared(las_file)
    # DR-05: Count string-curve values stored.  MAX_TOTAL_ELEMENTS accounts
    # numpy allocation at ~8 B/element, but each Python str object costs
    # ~50-100 B — a crafted file can pass the element guard and still
    # allocate ~50-100 GB.  Mirror the LAS 3.0 _MAX_STRING_VALUES cap.
    _string_value_count = 0

    # L-03: The null sentinel must be known BEFORE allocation to decide the
    # dtype.  int64 allocation would truncate a fractional null sentinel
    # (e.g. -999.25 → -999), silently corrupting every null cell — so the
    # int64 branch is only taken when the declared NULL is integral.
    # M-62: the int64 branch must ALSO check int64 representability.  A
    # huge integral sentinel (>= 2^63, e.g. NULL 1e19) passes
    # ``float(x).is_integer()`` but int64 assignment of ``int(null_value)``
    # (failure fills at :900) raises OverflowError that escapes the
    # LASParseError-only boundary.  Route such sentinels to the object
    # dtype path (EXT-04) which holds arbitrary Python ints exactly.
    null_value = _get_null_value(las_file.well)
    _null_is_integral = (
        float(null_value).is_integer() and _INT64_MIN <= int(null_value) <= _INT64_MAX
    )

    # Pre-allocate arrays — string curves go to string_data, numeric to logs.
    # C-505: Ensure curvenames that appear in both string_curve_indices and
    # are allocated mixed-ly do not cause double-entry issues in logs.
    _numeric_curve_indices: list[int] = []
    _string_curve_map: dict[int, str] = {}  # index → curve_name for fast lookup
    for i, curve_name in enumerate(las_file.curves_order):
        if i in _string_curve_indices:
            las_file.string_data[curve_name] = np.full(data_line_count, "", dtype=object)
            _string_curve_map[i] = curve_name
        elif i in _integer_curve_indices:
            if _null_is_integral:
                # L-03: int64 storage for {I} curves — preserves exact
                # integer values above 2^53 that float64 cannot represent.
                las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.int64)
            else:
                # EXT-04: fractional declared NULL — object dtype preserves
                # exact integer values above 2^53 while null cells keep the
                # fractional sentinel (int64 would truncate -999.25 → -999).
                las_file.logs[curve_name] = np.zeros(data_line_count, dtype=object)
            _numeric_curve_indices.append(i)
        else:
            las_file.logs[curve_name] = np.zeros(data_line_count, dtype=np.float64)
            _numeric_curve_indices.append(i)

    # FIX-CONV-2: data_line_count is a pre-allocation hint, never a
    # correctness bound.  _allocated tracks the actual array capacity
    # (starts at the pre-scan estimate, grows geometrically on undercount);
    # data_line_count keeps its original meaning for the overcount warning.
    _allocated = data_line_count

    # Pre-extract numeric arrays for fast inner-loop access.
    # String array access is direct via las_file.string_data since we
    # cannot mix dtypes in a homogeneous list.
    curve_arrays = [las_file.logs[name] for name in las_file.curves_order if name in las_file.logs]
    # Build a mapping from logical curve index → numeric array index
    # for the inner loop (string curves are excluded from curve_arrays).
    _logical_to_numeric: dict[int, int] = {}
    for ni, li in enumerate(_numeric_curve_indices):
        _logical_to_numeric[li] = ni

    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    current_line = 0
    # F-I2-XPD-03: Replace boolean-once flags with counters so automated
    # validation tools can enumerate the total number of affected rows.
    # The first occurrence logs full context; subsequent occurrences are
    # counted silently; a summary is logged at the end of the section.
    extra_col_count: int | None = None  # Track extra-column count for summary
    short_row_count: int | None = None  # F-11: Track short-row count for summary
    # I2-02: Track embedded-delimiter string truncation for summary.
    embedded_delim_count: int | None = None

    for stripped in _iter_ascii_data_lines(lines):
        # Split using DLM-aware split (shared utility).
        values = _split_data_line(stripped, delimiter)

        # M-37: Skip a standalone mnemonic header row (e.g. "~A\nDEPT GR\n"
        # before the numeric rows).  LAS 2.0 places mnemonics on the ~A
        # line, but a common real-world variant emits them as a separate
        # first row.  Consuming it as a data row produced a phantom
        # all-null first row and shifted every value by one column.  Skip
        # it so it is not counted as a data row.
        if _is_mnemonic_header_row(
            values,
            las_file,
            curve_count,
            _string_curve_indices,
            declared=_mnemonic_declared,
        ):
            continue

        # Warn about extra columns being silently discarded.
        # F-I2-XPD-03: First occurrence logs full context; subsequent
        # occurrences are counted silently; summary logged at end.
        if len(values) > curve_count:
            if extra_col_count is None:
                warnings.warn(
                    f"Data line has {len(values)} values but only {curve_count} curves declared "
                    f"in ~C section. Extra columns are discarded. "
                    f"(Further occurrences will be counted.)",
                    UserWarning,
                    stacklevel=2,
                )
                extra_col_count = 1
            else:
                extra_col_count += 1
            # I2-02: DLM=COMMA + embedded comma in a {S} string value
            # truncates the string AND shifts every following column —
            # the next column's genuine value is destroyed (verified:
            # `100.0,WELL, TX,10` with DEPT/WELL{S}/GR → WELL truncated
            # to "WELL", GR receives " TX" → null, genuine 10 lost as a
            # discarded extra column).  csv.reader quote-awareness is
            # deliberately NOT used (F2-015: the writer emits raw
            # delimiter.join() with no CSV quotes, so adding quote
            # parsing would break the writer's own roundtrips) — the
            # correct response for external files is a LOUD warning that
            # the value was truncated/lost, not a silent shift.
            if delimiter != " " and _string_curve_indices:
                if embedded_delim_count is None:
                    warnings.warn(
                        f"Data line has {len(values)} values but only {curve_count} "
                        f"curves declared with DLM={delimiter!r}. A string curve "
                        f"value may contain the delimiter character, truncating "
                        f"the string and shifting the following columns (genuine "
                        f"values may be lost). Consider quoting string values or "
                        f"using a delimiter not present in the data. "
                        f"(Further occurrences will be counted.)",
                        UserWarning,
                        stacklevel=2,
                    )
                    embedded_delim_count = 1
                else:
                    embedded_delim_count += 1

        # F-11: Warn when non-wrapped data lines have fewer values than
        # declared curves. Short rows in WRAP=YES mode are expected;
        # this warning only fires in non-wrapped (WRAP=NO) mode,
        # which we know because _read_normal is only called for non-wrapped.
        # F-I2-XPD-03: First occurrence logs full context; subsequent
        # occurrences are counted silently; summary logged at end.
        if len(values) < curve_count:
            if short_row_count is None:
                warnings.warn(
                    f"Data line has {len(values)} values but {curve_count} curves declared in ~C section. "
                    f"Missing values are filled with the null value ({null_value:.2f}). "
                    f"(Further occurrences will be counted.)",
                    UserWarning,
                    stacklevel=2,
                )
                short_row_count = 1
            else:
                short_row_count += 1

        # G-04 (FIX-CONV-2): data_line_count is a pre-allocation hint, never
        # a correctness bound.  When the pre-scan undercounts data lines
        # (rows the pre-scan skips but the reader keeps — short all-mnemonic
        # rows, all-string coincident rows), grow the arrays geometrically
        # (mirroring _read_wrapped._append_value) instead of silently
        # dropping the last data row.  A bounded re-allocation preserves the
        # MAX_TOTAL_ELEMENTS guard's memory-exhaustion intent.
        if current_line >= _allocated:
            _new_capacity = max(_allocated * 2, current_line + 1)
            if curve_count * _new_capacity > MAX_TOTAL_ELEMENTS:
                raise LASParseError(
                    f"Total allocation ({curve_count} curves x {_new_capacity} "
                    f"lines = {curve_count * _new_capacity} elements) exceeds "
                    f"maximum allowed ({MAX_TOTAL_ELEMENTS}). "
                    f"The file may be malformed or corrupt."
                )
            for curve_name in las_file.curves_order:
                if curve_name in las_file.logs:
                    _old_arr = las_file.logs[curve_name]
                    _new_arr = np.zeros(_new_capacity, dtype=_old_arr.dtype)
                    _new_arr[: _old_arr.shape[0]] = _old_arr
                    # Whole-container growth: EVERY curve array is resized to
                    # the same _new_capacity, so the M-43 equal-length
                    # invariant holds after the pass.  Bypass the per-key
                    # __setitem__ guard during the transition, mirroring
                    # _GuardedDict.trim_all's own dict.__setitem__ pattern
                    # (models.py:302-303) for whole-container reconciliation.
                    dict.__setitem__(las_file.logs, curve_name, _new_arr)
                if curve_name in las_file.string_data:
                    _old_str = las_file.string_data[curve_name]
                    _new_str = np.full(_new_capacity, "", dtype=object)
                    _new_str[: _old_str.shape[0]] = _old_str
                    dict.__setitem__(las_file.string_data, curve_name, _new_str)
            _allocated = _new_capacity
            # Rebuild the pre-extracted numeric references — they now point
            # at the old (too-small) arrays.
            curve_arrays = [
                las_file.logs[name] for name in las_file.curves_order if name in las_file.logs
            ]

        for i in range(min(len(values), curve_count)):
            if i in _string_curve_indices:
                # F-WXP-01: String curve — store raw value verbatim.
                # DR-05: Cap Python-object accumulation (string values
                # amplify ~6-12x beyond the element guard's 8 B/element).
                # Mirrors _las30_data._check_string_cap semantics: raise
                # once the count reaches the cap, before storing.
                curve_name = _string_curve_map[i]
                if _string_value_count >= MAX_STRING_VALUES:
                    raise LASParseError(
                        f"String curve values exceed maximum allowed "
                        f"({MAX_STRING_VALUES}). The file may be malformed "
                        f"or corrupt."
                    )
                _string_value_count += 1
                las_file.string_data[curve_name][current_line] = _desanitize_las_value(
                    values[i], _desanitize_enabled
                )
            elif i in _integer_curve_indices:
                # L-03/EXT-04: {I} curve — parse via int() to preserve
                # exactness above 2^53 (float() would round).  With a
                # fractional NULL the array is object dtype and failures
                # return the float sentinel (not int(null_value)).
                ni = _logical_to_numeric[i]
                curve_arrays[ni][current_line] = _to_integer_value(
                    _desanitize_las_value(values[i], _desanitize_enabled),
                    null_value,
                    _fc,
                    _null_as_float=not _null_is_integral,
                )
            else:
                ni = _logical_to_numeric[i]
                curve_arrays[ni][current_line] = _to_finite_float(
                    _desanitize_las_value(values[i], _desanitize_enabled),
                    null_value,
                    _fc,
                )

        # Fill remaining curves with null_value when line has fewer values
        for i in range(len(values), curve_count):
            if i in _string_curve_indices:
                # Fill missing string curve values with empty string.
                curve_name = _string_curve_map[i]
                if current_line < len(las_file.string_data[curve_name]):
                    las_file.string_data[curve_name][current_line] = ""
            else:
                ni = _logical_to_numeric[i]
                if current_line < len(curve_arrays[ni]):
                    # L-03: int64 arrays need an integral fill; the int64
                    # branch is only active when null_value is integral.
                    if i in _integer_curve_indices and _null_is_integral:
                        curve_arrays[ni][current_line] = int(null_value)
                    else:
                        curve_arrays[ni][current_line] = null_value

        current_line += 1

    # F-024: Warn when pre-scan overcounted data lines (fewer actual data
    # lines in the ~A section than declared).  Unlike an undercount (which
    # the G-04 grow branch now absorbs without data loss), this preserves
    # data but indicates a pre-scan discrepancy — e.g. a multi-section file
    # where _pre_scan counts lines across all sections but _read_normal
    # only consumes those in the first ~A section.  Uses the ORIGINAL
    # pre-scan count, never the grown capacity.
    if current_line < data_line_count:
        warnings.warn(
            f"Pre-scan overcount: declared {data_line_count} data lines but only {current_line} actual "
            f"data lines found in ~A section. Arrays will be trimmed to "
            f"actual line count.",
            UserWarning,
            stacklevel=2,
        )

    # F-PXR-03: Warn when non-trivial conversion failures occurred.
    _log_conversion_failures(_fc, null_value)

    # F-I2-XPD-03: Summary of extra-column and short-row occurrences,
    # replacing the previous boolean-once pattern that suppressed all
    # diagnostics after the first row.  This allows automated data
    # quality tools to enumerate affected rows.
    if extra_col_count is not None and extra_col_count > 1:
        warnings.warn(
            f"{extra_col_count} data line(s) had more values than the {curve_count} declared curves. "
            f"Extra columns were discarded.",
            UserWarning,
            stacklevel=2,
        )
    if embedded_delim_count is not None and embedded_delim_count > 1:
        warnings.warn(
            f"{embedded_delim_count} data line(s) had more values than the {curve_count} "
            f"declared curves with DLM={delimiter!r}. String curve values may contain "
            f"the delimiter character; those strings were truncated and following "
            f"columns' genuine values may have been lost.",
            UserWarning,
            stacklevel=2,
        )
    if short_row_count is not None and short_row_count > 1:
        warnings.warn(
            f"{short_row_count} data line(s) had fewer values than the {curve_count} declared curves. "
            f"Missing values were filled with the null value ({null_value:.2f}).",
            UserWarning,
            stacklevel=2,
        )

    # F36: Trim arrays when ~A section ended early (fewer data rows than
    # allocated). Pre-allocated np.zeros tail would otherwise expose 0.0
    # values that differ from null_value, corrupting downstream analysis.
    # Fill the tail with null_value before slicing to ensure consistency
    # even when pre-scan over-counts relative to _read_normal's actual
    # line consumption.  FIX-CONV-2: the condition is the ACTUAL allocated
    # capacity (_allocated), which covers both the pre-scan overcount case
    # and the slack left by geometric growth on undercount.
    # F-WXP-01: Also trim string_data arrays for string-curve columns.
    # FIX-CONV-1 (F-01): route the trim through _GuardedDict.trim_all —
    # per-key ``las_file.logs[curve_name] = arr[:current_line]``
    # reassignment tripped the M-43 length guard: the FIRST key is
    # compared against still-untrimmed siblings and rejected even though
    # the final trimmed state is fully consistent.  ``trim_all`` performs
    # the whole-container trim and validates the invariant once.
    if current_line < _allocated:
        # Fill numeric tails with null_value (float64) — trim_all then
        # slices every value to current_line rows.
        for curve_name in las_file.curves_order:
            if curve_name in las_file.logs:
                arr = las_file.logs[curve_name]
                if current_line < len(arr):
                    arr[current_line:] = null_value
        cast(_GuardedDict, las_file.logs).trim_all(current_line)
        # Fill string tails with "" (str_) before the whole-container trim.
        for curve_name in las_file.curves_order:
            if curve_name in las_file.string_data:
                arr_str = las_file.string_data[curve_name]
                if current_line < len(arr_str):
                    arr_str[current_line:] = ""
        cast(_GuardedDict, las_file.string_data).trim_all(current_line)


def _read_wrapped(
    lines: list[str],
    las_file: LASFile,
    curve_count: int,
    delimiter: str = " ",
    _desanitize_enabled: bool | None = None,
) -> None:
    """Read wrapped ASCII data using depth_line flag protocol.

    In wrapped mode:
    - The DEPTH value appears ALONE on its own line
    - Subsequent lines contain the remaining curve values
    - Once all curves for a depth step are read, the next depth line follows

    Uses list accumulation then np.array() at end to avoid the O(n^2)
    numpy.append bug in the original code.
    """
    # Deduplicate curve names before reading
    _deduplicate_curves(las_file)
    curve_count = len(las_file.curves_order)

    # Count actual data lines for MAX_DATA_LINES bound check.
    # All other reading paths (_read_normal, parser, dev_reader)
    # already have this guard.
    _count_in_ascii = False
    _count = 0
    for line in lines:
        stripped = line.strip()
        if _is_section_header(stripped):
            if _is_ascii_section(stripped):
                _count_in_ascii = True
            elif _count_in_ascii:
                # P-16: Every genuine section header ends the ~A block
                # (recognized OR unrecognized) — the parser routes
                # unrecognized sections to other_lines, so their body is
                # not data.  F-I2-XPD-01 retained: only genuine
                # ~-prefixed section-like lines (checked by
                # _is_section_header) terminate counting; control-char
                # noise (~3D, ~., ~#) fails that check and is skipped
                # below.
                break  # End of ~A section
            continue
        # P-16: ~-prefixed non-section-header lines are not data rows.
        if stripped.startswith("~"):
            continue
        if _count_in_ascii and stripped and not stripped.startswith("#"):
            _count += 1

    if _count > MAX_DATA_LINES:
        raise LASParseError(
            f"Data line count ({_count}) exceeds maximum allowed "
            f"({MAX_DATA_LINES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks.
    # In wrapped mode each depth step spans ~curve_count lines
    # (1 depth + curve_count-1 data values).  Estimate depth steps
    # from total line count using the known curve_count.  The
    # alternative heuristic of counting single-value lines (or using
    # ceil(_count/2)) overcounts depth steps when curve values
    # legitimately appear one per line, causing false rejection.
    if curve_count > 0:
        # F-54-upgrade: Use math.ceil instead of integer division to avoid
        # undercounting depth steps in wrapped mode.  Integer division
        # _count // curve_count can undercount by up to curve_count-1
        # steps, allowing malicious files to bypass the resource guard.
        # F-37: Removed overly aggressive ceil(_count/2) which over-estimates
        # by up to curve_count/2x for files with many curves, falsely
        # rejecting valid wrapped files.  ceil(_count/curve_count) is the
        # accurate estimate per depth step in wrapped mode.
        depth_steps = max(1, math.ceil(_count / curve_count))
        if curve_count * depth_steps > MAX_TOTAL_ELEMENTS:
            raise LASParseError(
                f"Total allocation ({curve_count} curves x ~{depth_steps} depth steps ≈ "
                f"{curve_count * depth_steps} elements) exceeds maximum allowed "
                f"({MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
            )

    # F-R-03: Detect string curves from CurveDefinition data_format.
    # See _data_section_reader._detect_string_curves for full details.
    _string_curve_indices: set[int] = _detect_string_curves(las_file)
    _string_curve_map: dict[int, str] = {
        _idx: las_file.curves_order[_idx] for _idx in _string_curve_indices
    }
    # FIX-CONV-2: Hoist the mnemonic-header match set once per read.
    _mnemonic_declared = _mnemonic_header_declared(las_file)
    # F-R-03: Accumulate string values into lists (same pattern as
    # data_lists).  Pad/trim and convert to np.array at the end.
    _string_lists: dict[str, list[str]] = {_name: [] for _name in _string_curve_map.values()}
    # DR-05: Cap Python-object accumulation for string values in wrapped
    # mode (mirrors _las30_data._MAX_STRING_VALUES; string objects amplify
    # ~6-12x beyond the element guard's 8 B/element intent).
    _string_value_count = 0

    def _check_string_cap() -> None:
        nonlocal _string_value_count
        if _string_value_count >= MAX_STRING_VALUES:
            raise LASParseError(
                f"String curve values exceed maximum allowed "
                f"({MAX_STRING_VALUES}). The file may be malformed or corrupt."
            )
        _string_value_count += 1

    # L-03: Detect {I} integer-format curves and pre-compute the null
    # sentinel BEFORE allocation — the int64 dtype branch is only taken
    # when the declared NULL is integral (int64 would truncate a
    # fractional sentinel like -999.25, corrupting every null cell).
    # M-62: the int64 branch must ALSO check int64 representability — a
    # huge integral sentinel (>= 2^63) passes is_integer() but int64
    # assignment of int(null_value) (pad at :1382) raises OverflowError.
    # Route to the object dtype path instead.
    _integer_curve_indices: set[int] = _detect_integer_curves(las_file)
    null_value = _get_null_value(las_file.well)
    _null_is_integral = (
        float(null_value).is_integer() and _INT64_MIN <= int(null_value) <= _INT64_MAX
    )
    _fc: list[int] = [0]  # F-PXR-03: count non-trivial conversion failures

    # IT3-F-03: Pre-allocate numeric columns instead of accumulating every
    # value as a Python float object (~32 B/value) then converting to numpy
    # at the end (~8 B/value).  The list phase previously amplified peak
    # memory ~4x and bypassed the MAX_TOTAL_ELEMENTS guard's OOM intent
    # (the guard counts final arrays only).  Capacity starts at the
    # depth-step estimate and grows geometrically; the dynamic
    # total_elements counter below still bounds total accumulation.
    _numeric_indices = [i for i in range(curve_count) if i not in _string_curve_indices]
    _est_steps = max(1, math.ceil(_count / curve_count)) if curve_count > 0 else 0
    data_arrays: dict[int, np.ndarray] = {}
    data_fill: dict[int, int] = {}
    for _i in _numeric_indices:
        if _i in _integer_curve_indices and _null_is_integral:
            data_arrays[_i] = np.zeros(_est_steps, dtype=np.int64)
        elif _i in _integer_curve_indices:
            # EXT-04: fractional declared NULL — object dtype preserves
            # exact {I} integers above 2^53 without truncating the
            # fractional null sentinel.
            data_arrays[_i] = np.zeros(_est_steps, dtype=object)
        else:
            data_arrays[_i] = np.zeros(_est_steps, dtype=np.float64)
        data_fill[_i] = 0

    def _append_value(_i: int, value: float | int) -> None:
        """Append one value to curve *_i*'s pre-allocated column, growing
        geometrically when the depth-step estimate is exceeded."""
        _arr = data_arrays[_i]
        _idx = data_fill[_i]
        if _idx >= _arr.shape[0]:
            _new = np.zeros(max(_arr.shape[0] * 2, 1), dtype=_arr.dtype)
            _new[:_idx] = _arr[:_idx]
            data_arrays[_i] = _new
            _arr = _new
        _arr[_idx] = value
        data_fill[_i] = _idx + 1

    depth_line = True  # First data line is always a depth line
    counter = 0  # Tracks position within non-depth curves
    depth_had_extra = False  # F-06: track pathologically-malformed depth lines
    total_elements = 0  # F-54-upgrade: dynamic element counter for wrapped mode

    for stripped in _iter_ascii_data_lines(lines, mode_suffix=" (wrapped mode)"):
        # Split using DLM-aware split (shared utility).
        values = _split_data_line(stripped, delimiter)

        # F-03 (FIX-CONV-1): M-37's standalone-mnemonic-header skip must
        # apply in wrapped mode too.  A header row (e.g. "~A\nDEPT GR\n"
        # before the wrapped depth/continuation rows) is a column header,
        # not data — consuming it produced a phantom null first row and
        # shifted every value by one column.  The M-38 WRAP=YES
        # fall-through routes such files into the wrapped path, so the
        # skip is required here (not just in _read_normal).
        if _is_mnemonic_header_row(
            values,
            las_file,
            curve_count,
            _string_curve_indices,
            declared=_mnemonic_declared,
        ):
            continue

        if depth_line:
            # Depth line: single value = depth for this step.
            # Reset the extra-values flag at each depth step boundary
            # so stale flags from a previous step never persist into
            # the pathological-misalignment check for this step.
            depth_had_extra = False
            if not values:
                continue
            if len(values) == curve_count and total_elements == 0:
                # M-38: The FIRST data line carries EXACTLY curve_count
                # values — a COMPLETE row (a mixed-wrap file: first row
                # written unwrapped, continuation lines wrapped, e.g.
                # per-row line-width wrapping).  Consume it as a full step
                # instead of warning + discarding the "extra" values,
                # which previously shifted every later depth into a curve
                # column (silent column/depth corruption).  Restricted to
                # the first line: a complete-value depth line MID-file is
                # anomalous junk (documented warn+discard behavior —
                # test_wrapped_depth_line_extra_values).
                for _ci, _v in enumerate(values):
                    if _ci in _string_curve_indices:
                        _check_string_cap()
                        _string_lists[_string_curve_map[_ci]].append(
                            _desanitize_las_value(_v, _desanitize_enabled)
                        )
                    elif _ci in _integer_curve_indices:
                        _append_value(
                            _ci,
                            _to_integer_value(
                                _desanitize_las_value(_v, _desanitize_enabled),
                                null_value,
                                _fc,
                                _null_as_float=not _null_is_integral,
                            ),
                        )
                    else:
                        _append_value(
                            _ci,
                            _to_finite_float(
                                _desanitize_las_value(_v, _desanitize_enabled),
                                null_value,
                                _fc,
                            ),
                        )
                    total_elements += 1
                    if total_elements > MAX_TOTAL_ELEMENTS:
                        raise LASParseError(
                            f"Total elements ({total_elements}) exceeds maximum allowed "
                            f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                            f"The file may be malformed or corrupt."
                        )
                # Step complete — next line starts a fresh depth step.
                depth_line = True
                counter = 0
                continue
            if len(values) > 1:
                warnings.warn(
                    f"Wrapped mode: depth line has {len(values)} values, expected 1. "
                    f"Extra values discarded. Line content: '{stripped[:80]}'",
                    stacklevel=2,
                )
                # F-06: Mark this depth line as malformed.  If the subsequent
                # data line cannot fill all non-depth curves, the file is
                # pathologically misaligned and will be rejected.
                depth_had_extra = True
            # F-R-03: If the depth curve is a string curve, accumulate
            # into _string_lists.  (The previous code also appended a
            # null_value placeholder to data_lists[0], but curve 0 is a
            # string curve and is excluded from _float_indices — the
            # placeholder never contributed to _max_len or to the final
            # arrays.  IT3-F-03's pre-allocated numeric columns only
            # cover numeric curves, so the placeholder is dropped; string
            # lengths are already considered by _max_len via F-007-fix.)
            if 0 in _string_curve_indices:
                _check_string_cap()
                _string_lists[_string_curve_map[0]].append(
                    _desanitize_las_value(values[0], _desanitize_enabled)
                )
            elif 0 in _integer_curve_indices:
                _append_value(
                    0,
                    _to_integer_value(
                        _desanitize_las_value(values[0], _desanitize_enabled),
                        null_value,
                        _fc,
                        _null_as_float=not _null_is_integral,
                    ),
                )
            else:
                _append_value(
                    0,
                    _to_finite_float(
                        _desanitize_las_value(values[0], _desanitize_enabled),
                        null_value,
                        _fc,
                    ),
                )
            total_elements += 1
            if total_elements > MAX_TOTAL_ELEMENTS:
                raise LASParseError(
                    f"Total elements ({total_elements}) exceeds maximum allowed "
                    f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                    f"The file may be malformed or corrupt."
                )
            depth_line = False
            counter = 0
        else:
            # Data lines: values for remaining curves
            # F-06: Pathological misalignment detection.  If the previous
            # depth line had extra values AND this data line has fewer
            # values than the number of unfilled non-depth curves
            # (curve_count - 1 - counter), the file is likely misaligned.
            # A truly pathological case (depth line has extra data, next
            # line has very few values, and the gap exceeds 2 curves)
            # raises an error because data corruption is certain.
            if depth_had_extra and curve_count >= 3:
                remaining_curves = curve_count - 1 - counter
                if len(values) < remaining_curves:
                    if len(values) <= 2 and remaining_curves - len(values) >= 2:
                        # Truly pathological: data line provides ≤2 values
                        # but ≥2 more curves need filling — data WILL be
                        # irrecoverably misaligned across all curves.
                        raise LASParseError(
                            f"Wrapped mode: pathologically malformed data — the "
                            f"previous depth line had extra values, and this data "
                            f"line has only {len(values)} values but "
                            f"{remaining_curves} non-depth curves still need "
                            f"values (total curves={curve_count}). File is "
                            f"irrecoverably misaligned."
                        )
                    # Non-truly-pathological: data line has fewer values
                    # than needed but the shift is ≤2 curves — recovery
                    # would silently produce corrupt data (curve values
                    # shifted by 1 depth step).  Hard-fail with a clear
                    # diagnostic instead of producing corrupt output.
                    raise LASParseError(
                        f"Wrapped mode: unrecoverable data misalignment — "
                        f"the previous depth line had extra values, and "
                        f"this data line has only {len(values)} values "
                        f"but {remaining_curves} non-depth curves still "
                        f"need values (total curves={curve_count}). "
                        f"Continuing would produce corrupt data.  Consider "
                        f"reloading with wrapped=False if the original "
                        f"unwrapped file is available."
                    )

            for i, val_str in enumerate(values):
                counter += 1

                if counter >= curve_count:
                    # F2: Overflow — more values on this line than non-depth
                    # curves remaining in the step.  Extra values would shift
                    # subsequent lines if consumed silently; warn and discard
                    # the overflow portion.
                    warnings.warn(
                        f"Wrapped mode: overflow on data line — {len(values)} "
                        f"values but only {curve_count - 1} non-depth curves. "
                        f"Extra value(s) discarded. Line content: '{stripped[:80]}'",
                        stacklevel=2,
                    )
                    # F-I2-M15: If the previous depth line had extra values
                    # (suggesting a multi-value non-wrapped row was parsed as
                    # a depth line), and this data line overflows (has enough
                    # values to exceed all remaining non-depth curves), then
                    # the file is likely non-wrapped being read as wrapped.
                    # The first values from each line are misrouted: what
                    # should be the next depth step goes to C1 instead,
                    # causing a permanent DEPTH↔C1 swap.
                    if depth_had_extra:
                        warnings.warn(
                            f"Wrapped mode: suspected non-wrapped data — "
                            f"a previous depth line had extra values and "
                            f"this data line overflowed. DEPTH and curve "
                            f"values are likely swapped. "
                            f"Line content: '{stripped[:80]}'",
                            stacklevel=2,
                        )
                    break

                # F-R-03: Dispatch string curves to _string_lists,
                # float curves to data_lists via _to_finite_float.
                if counter in _string_curve_indices:
                    _check_string_cap()
                    _string_lists[_string_curve_map[counter]].append(
                        _desanitize_las_value(val_str, _desanitize_enabled)
                    )
                elif counter in _integer_curve_indices:
                    _append_value(
                        counter,
                        _to_integer_value(
                            _desanitize_las_value(val_str, _desanitize_enabled),
                            null_value,
                            _fc,
                            _null_as_float=not _null_is_integral,
                        ),
                    )
                else:
                    _append_value(
                        counter,
                        _to_finite_float(
                            _desanitize_las_value(val_str, _desanitize_enabled),
                            null_value,
                            _fc,
                        ),
                    )
                total_elements += 1
                if total_elements > MAX_TOTAL_ELEMENTS:
                    raise LASParseError(
                        f"Total elements ({total_elements}) exceeds maximum allowed "
                        f"({MAX_TOTAL_ELEMENTS}) in wrapped mode. "
                        f"The file may be malformed or corrupt."
                    )

                if counter == curve_count - 1:
                    # All curves for this depth step are complete.
                    # Break to discard any extra values on this line
                    # (prevents silent misalignment if a line has
                    # more values than expected).
                    # F-D2-M01: Warn when extra values remain on this
                    # line after step completion (previously silent).
                    if i + 1 < len(values):
                        warnings.warn(
                            f"Wrapped mode: step complete with "
                            f"{len(values) - i - 1} extra value(s) "
                            f"discarded on this line. Line content: "
                            f"'{stripped[:80]}'",
                            stacklevel=2,
                        )
                        # F-I2-M15: If the previous depth line also had
                        # extra values (depth_had_extra is True) AND this
                        # data line completes the step with leftover values,
                        # the file is likely non-wrapped data misread as
                        # wrapped.  Each multi-value line contributes only
                        # its first value where one curve value is expected,
                        # permanently swapping DEPTH with C1 (and C1 with C2,
                        # etc.) for all subsequent depth steps.
                        if depth_had_extra:
                            warnings.warn(
                                f"Wrapped mode: suspected non-wrapped data — "
                                f"previous depth line had extra values and "
                                f"this data line completed the step with "
                                f"{len(values) - i - 1} leftover value(s). "
                                f"DEPTH and curve values may be swapped. "
                                f"Line content: '{stripped[:80]}'",
                                stacklevel=2,
                            )
                    counter = 0
                    depth_line = True
                    depth_had_extra = False
                    break

            # F11: After overflow (extra values on data line were
            # discarded), reset for the next depth line.  The break in
            # the overflow branch leaves counter at >= curve_count and
            # depth_line=False; without this reset the next line would
            # be incorrectly treated as a continuation of this depth step.
            # Partial under-fill (fewer values than needed) is NOT reset
            # here — in valid wrapped files data values can legitimately
            # span multiple lines.
            if counter >= curve_count and not depth_line:
                depth_line = True
                counter = 0
                depth_had_extra = False

    # N-I-08: Detect mid-file under-filled steps.  In valid wrapped mode
    # every depth step consumes exactly curve_count values (1 depth line +
    # curve_count-1 data values).  If the total consumed is NOT a multiple
    # of curve_count, some step was under-filled mid-file — the reader
    # silently consumed the next depth line as a data value, shifting the
    # depth value into the last curve column (and the next data value into
    # the depth column).  The F-06 guard (above) only fires for the
    # OVER-filled depth-line variant (depth_had_extra); this catches the
    # clean under-filled variant that previously produced only generic
    # padding warnings while corrupting the depth column.  Trailing-EOF
    # incomplete steps also trip this and are additionally handled by the
    # padding warnings below (the message is accurate for both).
    if curve_count > 0 and total_elements % curve_count != 0:
        warnings.warn(
            f"Wrapped mode: data section consumed {total_elements} values "
            f"but curve count is {curve_count} "
            f"({total_elements % curve_count} value(s) not accounted for). "
            f"A depth step appears to be under-filled mid-file; DEPTH and "
            f"curve values may be misaligned.  Check the source file for a "
            f"step missing one or more values.",
            stacklevel=2,
        )
    # M-72: The modulo check above is blind to a mid-file under-fill whose
    # total stays a multiple of curve_count (aligned total).  In that case
    # every depth step "completes" from the reader's perspective, but the
    # missing value caused the reader to consume the next depth line as a
    # data value — permanently shifting a data value into the depth column
    # (and a depth value into the last curve column) with ZERO diagnostics.
    # Verify the depth column is actually depth-aligned: a genuine depth
    # log is monotonic (non-decreasing OR non-increasing).  A non-monotonic
    # depth column signals the shift.  Advisory warning — not an error.
    if (
        curve_count > 0
        and total_elements % curve_count == 0
        and 0 in data_arrays
        and data_fill[0] >= 2
        and np.issubdtype(data_arrays[0].dtype, np.number)
    ):
        _depth_vals = data_arrays[0][: data_fill[0]]
        _diffs = np.diff(_depth_vals)
        if not (np.all(_diffs >= 0) or np.all(_diffs <= 0)):
            warnings.warn(
                f"Wrapped mode: data section consumed {total_elements} values "
                f"(a multiple of the curve count {curve_count}) but the depth "
                f"column is not monotonic — a depth step appears to be "
                f"under-filled mid-file; DEPTH and curve values may be "
                f"misaligned.  Check the source file for a step missing one "
                f"or more values.",
                stacklevel=2,
            )

    # Compute actual number of depth steps from float curves only.
    # String curve indices have empty data_lists entries — skip them.
    _float_indices = [i for i in range(curve_count) if i not in _string_curve_indices]
    _max_len = max((data_fill[i] for i in _float_indices), default=0)
    # F-007-fix: Also consider string curve lengths so _max_len is not
    # zero when all curves are string-formatted (which would truncate
    # all accumulated string data to empty arrays).
    if _string_lists:
        _max_len = max(_max_len, *(len(sl) for sl in _string_lists.values()))

    # Pad incomplete last depth step for float curves
    for i in _float_indices:
        fill = data_fill[i]
        if fill < _max_len:
            warnings.warn(
                f"Wrapped mode: curve '{las_file.curves_order[i]}' has {fill} values "
                f"but expected {_max_len}. Padding with null value ({null_value}).",
                stacklevel=2,
            )
            _arr = data_arrays[i]
            if _arr.shape[0] < _max_len:
                _new = np.zeros(_max_len, dtype=_arr.dtype)
                _new[:fill] = _arr[:fill]
                data_arrays[i] = _new
                _arr = _new
            if i in _integer_curve_indices and _null_is_integral:
                _arr[fill:_max_len] = int(null_value)
            else:
                _arr[fill:_max_len] = null_value
            data_fill[i] = _max_len

    # F-PXR-03: Warn when non-trivial conversion failures occurred.
    _log_conversion_failures(_fc, null_value)

    # Convert pre-allocated columns to final (trimmed) numpy arrays
    for i in _float_indices:
        las_file.logs[las_file.curves_order[i]] = data_arrays[i][: data_fill[i]]

    # F-R-03: Pad/trim and convert string lists to numpy arrays.
    # String curve values are accumulated into _string_lists using the
    # same row-counting protocol as data_lists (one value per depth step).
    for _name, _sl in _string_lists.items():
        if len(_sl) < _max_len:
            warnings.warn(
                f"Wrapped mode: string curve '{_name}' has {len(_sl)} values "
                f"but expected {_max_len}. Padding with empty strings.",
                stacklevel=2,
            )
            _sl = _sl + [""] * (_max_len - len(_sl))
        elif len(_sl) > _max_len:
            _sl = _sl[:_max_len]
        las_file.string_data[_name] = np.array(_sl, dtype=object)
