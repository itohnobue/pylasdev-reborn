"""LAS 3.0 ASCII data processing — extracted from parser.py:LASParser.

Contains :func:`process_ascii_data` (formerly
``LASParser._process_ascii_data``) and :func:`_deduplicate_curves`
(formerly ``LASParser._deduplicate_curves``), moved verbatim from
parser.py as a pure mechanical extraction.

All state that was previously on ``self`` is passed through an
:class:`AsciiDataContext` dataclass.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import LASFile

from . import data_reader as _data_reader
from .data_reader import (
    _get_null_value,
    _resolve_max_tokens_per_line,
    _to_finite_float,
)
from .exceptions import LASDataError, LASParseError
from .models import CurveDefinition, DataSection

logger = logging.getLogger(__name__)


@dataclass
class AsciiDataContext:
    """Mutable context passed to :func:`process_ascii_data`.

    Replaces ``self.*`` references from the original
    ``LASParser._process_ascii_data`` method.  All mutations to
    ``las_file`` (``.logs``, ``.string_data``, ``.data_sections``,
    ``.curves``, ``.curves_order``) happen through this reference.
    ``cumulative_elements`` is read AND written — callers must
    seed it before the call and persist it afterward.
    """

    las_file: LASFile
    """The :class:`.LASFile` being built (mutated in-place)."""

    ascii_data_lines: list[str]
    """Collected ASCII data lines for the current section."""

    section_curve_start_idx: int
    """Start index into ``las_file.curves`` for this section."""

    section_curve_end_idx: int | None
    """End index (exclusive) into ``las_file.curves``, or ``None``."""

    current_section_name: str
    """Human-readable section name (e.g. ``"ASCII"``, ``"Main Log"``)."""

    current_data_section_type: str
    """LAS 3.0 data section type (e.g. ``"LOG_DATA"``)."""

    current_data_section_idx: int
    """Zero-based index of the current data section."""

    cumulative_elements: int = 0
    """Running total of ``num_curves * actual_count`` across sections.

    Read by the function to check cross-section allocation bounds;
    incremented inside the function.  Callers must write the final
    value back to the parser instance.
    """


def _deduplicate_curves(
    ctx: AsciiDataContext,
    section_curves: list[CurveDefinition],
    is_first_section: bool,
) -> list[str]:
    """Deduplicate curve names and return the deduplicated order.

    F-M19: Extracted from _process_ascii_data to share between parser.py,
    data_reader.py, and dev_reader.py.  Each domain agent refactors
    its own copy; the algorithmic core is identical across all three.

    When curve mnemonics collide (duplicates within the same section),
    suffixes ``_2``, ``_3``, etc. are appended.  ``original_mnemonic``
    is preserved on the renamed copy so writers can reconstruct the
    original name.

    For the first data section, renamed mnemonics are written back to
    the global ``curves``/``curves_order`` lists so ``to_dict()`` and
    the writer see consistent names.

    Args:
        ctx: The shared parser context.
        section_curves: Per-section curve definitions (modified in-place).
        is_first_section: True when this is the first data section
            processed for this file (triggers global writeback).

    Returns:
        Deduplicated curve mnemonic order list.
    """
    seen: dict[str, int] = {}
    deduped_order: list[str] = []
    output_names: set[str] = set()
    for i, curve in enumerate(section_curves):
        name = curve.mnemonic
        if name in seen:
            seen[name] += 1
            suffix = seen[name]
            new_name = f"{name}_{suffix}"
            while new_name in output_names:
                suffix += 1
                new_name = f"{name}_{suffix}"
            seen[name] = suffix
            section_curves[i] = CurveDefinition(
                mnemonic=new_name,
                unit=curve.unit,
                api_code=curve.api_code,
                description=curve.description,
                original_mnemonic=name
                if not curve.original_mnemonic
                else curve.original_mnemonic,
                data_format=curve.data_format,
                array_info=curve.array_info,
            )
            deduped_order.append(new_name)
            output_names.add(new_name)
            if is_first_section:
                global_idx = ctx.section_curve_start_idx + i
                if not ctx.las_file.curves[global_idx].original_mnemonic:
                    ctx.las_file.curves[global_idx].original_mnemonic = name
                ctx.las_file.curves[global_idx].mnemonic = new_name
                ctx.las_file.curves_order[global_idx] = new_name
        elif name in output_names:
            suffix = 2
            new_name = f"{name}_{suffix}"
            while new_name in output_names:
                suffix += 1
                new_name = f"{name}_{suffix}"
            seen[name] = suffix
            section_curves[i] = CurveDefinition(
                mnemonic=new_name,
                unit=curve.unit,
                api_code=curve.api_code,
                description=curve.description,
                original_mnemonic=name
                if not curve.original_mnemonic
                else curve.original_mnemonic,
                data_format=curve.data_format,
                array_info=curve.array_info,
            )
            deduped_order.append(new_name)
            output_names.add(new_name)
            if is_first_section:
                global_idx = ctx.section_curve_start_idx + i
                if not ctx.las_file.curves[global_idx].original_mnemonic:
                    ctx.las_file.curves[global_idx].original_mnemonic = name
                ctx.las_file.curves[global_idx].mnemonic = new_name
                ctx.las_file.curves_order[global_idx] = new_name
        else:
            seen[name] = 1
            deduped_order.append(name)
            output_names.add(name)
    return deduped_order


def process_ascii_data(ctx: AsciiDataContext) -> None:
    """Process collected ASCII data lines into numpy arrays.

    Handles LAS 3.0 delimiters and string data formats.
    Uses per-section curves (F1) to support LAS 3.0 files where
    different ~C blocks define different curve sets before each ~A.

    Pure moved code from parser.py:LASParser._process_ascii_data.
    All side effects go through ``ctx.las_file`` and
    ``ctx.cumulative_elements`` (mutated in-place).
    """
    # Lazy imports from parser.py to avoid circular import at module level.
    # parser.py imports from _las30_data; importing parser symbols here at
    # function-call time avoids the import cycle.
    from .parser import (
        COMMENT_PATTERN,
        EMPTY_PATTERN,
        MAX_DATA_SECTIONS,
        _desanitize_las_value,
        _validate_curve_data_format,
    )

    # F-26 / F-033: Global aggregate section count guard — placed
    # before the empty-data-lines early return so that a file with
    # MAX_DATA_SECTIONS+1 empty sections cannot bypass the limit.
    if ctx.current_data_section_idx >= MAX_DATA_SECTIONS:
        raise LASParseError(
            f"Data section count ({ctx.current_data_section_idx + 1}) exceeds "
            f"maximum allowed ({MAX_DATA_SECTIONS}). "
            f"The file may be malformed or corrupt."
        )

    if not ctx.ascii_data_lines:
        return

    # F-M-004: LAS 3.0 WRAP=YES check — detect actual wrap from data
    # lines before rejecting.  Files with WRAP=YES in the header but
    # non-wrapped data (one full row per line) should parse normally.
    # The heuristic mirrors _detect_actual_wrap in data_reader.py:
    # first data line with >= curve_count values → non-wrapped.
    # Single-curve files cannot distinguish wrap mode so keep the reject.
    if ctx.las_file.version.wrap.upper() == "YES":
        if ctx.section_curve_end_idx is not None:
            n_curves = len(ctx.las_file.curves[ctx.section_curve_start_idx : ctx.section_curve_end_idx])
        else:
            n_curves = len(ctx.las_file.curves[ctx.section_curve_start_idx :])
        actual_wrap = True
        if n_curves > 1:
            delimiter = ctx.las_file.version.delimiter_char
            for line in ctx.ascii_data_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if delimiter == " ":
                    actual_wrap = len(stripped.split(maxsplit=_resolve_max_tokens_per_line())) < n_curves
                else:
                    _tokens = stripped.split(delimiter, maxsplit=_resolve_max_tokens_per_line())
                    # F8M-02: Strip trailing empty strings before wrap check.
                    # Trailing delimiters (e.g. "100.0,") produce empty fields that
                    # inflate len(tokens), causing false-negative wrap detection:
                    # len(["100.0", ""]) = 2 → incorrectly detected as non-wrapped.
                    # Strip only TRAILING empties — middle empty fields represent
                    # legitimate sparse data values that must be preserved.
                    while _tokens and _tokens[-1] == "":
                        _tokens.pop()
                    actual_wrap = len(_tokens) == 1
                break
        if actual_wrap:
            raise LASParseError(
                "LAS 3.0 WRAP=YES is not supported by pylasdev.  "
                "Convert the file to unwrapped format (one line per "
                "depth step) before parsing, or set WRAP to NO."
            )

    # Get delimiter character
    delimiter = ctx.las_file.version.delimiter_char

    # Resolve MAX_TOKENS_PER_LINE — data_reader now uses a sentinel
    # (None → _data_reader.MAX_CURVES) so overrides propagate correctly at runtime.
    _max_tokens = _resolve_max_tokens_per_line()

    # F1: Get per-section curves — only curves defined since the
    # most recent ~C block.  For the first data section (no ~C
    # encountered or _section_curve_start_idx == 0), this is the
    # full curve list (backward-compatible).
    # When a pipe "| CURVE" association set _section_curve_end_idx,
    # cap to the main curve block so LOG_DATA sections don't pick up
    # per-section curves from later Definition sections.
    if ctx.section_curve_end_idx is not None:
        section_curves = list(
            ctx.las_file.curves[ctx.section_curve_start_idx : ctx.section_curve_end_idx]
        )
    else:
        section_curves = list(ctx.las_file.curves[ctx.section_curve_start_idx :])
    if not section_curves:
        # F32: Warn when data is present but no curves are defined
        # for this section, then return early.
        warnings.warn(
            "ASCII data present but no curves defined for this section. "
            "Data has been discarded.",
            UserWarning,
            stacklevel=2,
        )
        return

    # F1: Local dedup on per-section curves.  Global curve writeback
    # (lines 2363-2368, 2389-2394) is deferred until after the
    # actual_count > 0 check — see F2-07 writeback block below.
    is_first_section = not ctx.las_file.data_sections
    deduped_order = _deduplicate_curves(ctx, section_curves, is_first_section=False)

    # F-10: Validate cross-curve array continuity per LAS 3.0 spec
    # (section 4.3.1: curve array elements must use sequential [1]→[n]
    # indices with no gaps and consistent data formats across elements).
    # F-034: Extended to track position (index within section_curves),
    # enforce 1-based index start, and detect interleaving of array
    # and non-array curves.
    _array_groups: dict[str, list[tuple[int, str, str, int]]] = {}
    for _pos, curve in enumerate(section_curves):
        if curve.array_info is not None:
            _base = curve.array_info.base_name
            if _base not in _array_groups:
                _array_groups[_base] = []
            _array_groups[_base].append(
                (curve.array_info.index, curve.mnemonic,
                 curve.data_format or "", _pos)
            )
    for _base_name, _elements in _array_groups.items():
        if len(_elements) < 2:
            continue
        _elements.sort(key=lambda e: e[0])
        # F-034 (a): Arrays in LAS 3.0 must use 1-based indices.
        _first_idx = _elements[0][0]
        if _first_idx != 1:
            raise LASParseError(
                f"Array '{_base_name}' starts at index {_first_idx}; "
                f"LAS 3.0 requires 1-based array indices ([1]→[n])"
            )
        _expected = _first_idx
        _ref_fmt = _elements[0][2]
        _prev_pos: int | None = None
        for _idx, _mnem, _fmt, _pos in _elements:
            if _idx != _expected:
                raise LASParseError(
                    f"Non-contiguous array indices for '{_base_name}': "
                    f"index {_idx} at mnemonic '{_mnem}' follows index "
                    f"{_expected - 1} (expected {_expected})"
                )
            if _fmt and _ref_fmt and _fmt != _ref_fmt:
                raise LASParseError(
                    f"Inconsistent data_format for array '{_base_name}': "
                    f"mnemonic '{_mnem}' has '{_fmt}', expected '{_ref_fmt}'"
                )
            # F-034 (b): Positional contiguity — array elements must
            # be consecutive in section_curves.  No non-array curves
            # may interleave between array elements of the same group.
            if _prev_pos is not None and _pos != _prev_pos + 1:
                raise LASParseError(
                    f"Array '{_base_name}' elements are not contiguous "
                    f"in curve order: mnemonic '{_mnem}' at position "
                    f"{_pos} follows position {_prev_pos} (expected "
                    f"{_prev_pos + 1}). Non-array curves may be "
                    f"interleaved."
                )
            _prev_pos = _pos
            _expected += 1

    # Determine which curves are string type.
    # F-001: Previous one-liner (c.data_format in ("S", "A")) routed ALL
    # "A"-format curves as strings, including array elements with {A:N}
    # format specifiers where N is numeric.  Array-element curves (those
    # with array_info set) contain numeric data and must be routed as
    # numeric, not stored as np.str_.
    string_curves = {
        i: c.data_format in ("S",)
        or (c.data_format in ("A",) and c.array_info is None)
        for i, c in enumerate(section_curves)
    }

    # F2-05: Validate curve format types — unrecognized formats silently
    # produce null data when routed through _to_finite_float().  Known
    # numeric format types are F (float), E (exponential), D (Fortran
    # F-088: Validate curve data formats via shared helper (also called
    # from _parse_curve).  The helper accepts single-letter codes
    # (F, E, D, S, A) and extended Fortran-style specifiers (F8.3, etc.)
    # while rejecting non-numeric templates like {DEG} or {DD/MM/YYYY}.
    for curve in section_curves:
        _validate_curve_data_format(curve.data_format, curve.mnemonic)

    # Get null value (shared utility, used by parser, data_reader, writer)
    null_value = _get_null_value(ctx.las_file.well)

    # F-PXR-05: Warn when null value defaults to -999.25 because the
    # NULL key was not set in the well section.  This typically occurs
    # when the data section (~A/ASCII) precedes the ~Well section in
    # LAS 3.0 files, causing section-ordering-dependent null value.
    if not ctx.las_file.well.get("NULL"):
        logger.warning(
            "NULL value not found in well section for data section '%s'; "
            "using default null value (%.4g). This may indicate that the "
            "data section precedes the ~Well section in the file.",
            ctx.current_section_name or "ASCII",
            null_value,
        )

    # Create data section with per-section curves.
    # PXM-02: wrap LASDataError from DataSection.__post_init__ so
    # the parser boundary only raises LASParseError.
    try:
        data_section = DataSection(
            name=ctx.current_section_name or f"Section_{ctx.current_data_section_idx}",
            section_type=ctx.current_data_section_type,
            curves_order=deduped_order,
            section_curves=list(section_curves),
        )
    except LASDataError as exc:
        raise LASParseError(
            f"Data section '{ctx.current_section_name or ctx.current_data_section_idx}' "
            f"validation failed: {exc}"
        ) from exc

    num_curves = len(section_curves)

    # Count actual data lines (excluding comments and blank lines) for array sizing.
    actual_count = sum(
        1
        for line in ctx.ascii_data_lines
        if not COMMENT_PATTERN.match(line) and not EMPTY_PATTERN.match(line)
    )

    # F-26: Global aggregate limit across ALL data sections.
    # Each section passes per-section bounds (_data_reader.MAX_DATA_LINES,
    # _data_reader.MAX_CURVES, _data_reader.MAX_TOTAL_ELEMENTS) individually,
    # but an attacker can craft N sections (each just under the limits) to
    # cumulatively exhaust memory.  This caps the total number of data
    # sections processed.
    # F-033: MOVED above actual_count == 0 early return so empty sections
    # cannot bypass the global section count limit.
    if ctx.current_data_section_idx >= MAX_DATA_SECTIONS:
        raise LASParseError(
            f"Data section count ({ctx.current_data_section_idx + 1}) exceeds "
            f"maximum allowed ({MAX_DATA_SECTIONS}). "
            f"The file may be malformed or corrupt."
        )

    # F-I2-M09: Guard against zero data rows.  When actual_count is 0
    # (all lines are comments/blanks), num_curves * 0 = 0 always passes
    # _data_reader.MAX_TOTAL_ELEMENTS check, but np.zeros(0) still allocates an empty
    # ndarray per curve — up to _data_reader.MAX_CURVES per section and MAX_DATA_SECTIONS
    # sections can produce GB-scale allocations within guard limits.
    # A section with zero data rows has nothing to process; return early.
    if actual_count == 0:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}' has "
            f"{len(ctx.ascii_data_lines)} raw line(s) but 0 data rows "
            f"(all lines are comments or blanks). No data will be stored "
            f"for this section.",
            UserWarning,
            stacklevel=2,
        )
        return

    # F2-07: Deferred global curve writeback — only apply after
    # confirming actual_count > 0, preventing stale deduped names
    # in global curves/curves_order when a data section has no
    # actual data rows and is discarded.
    if is_first_section:
        for i, curve in enumerate(section_curves):
            global_idx = ctx.section_curve_start_idx + i
            if curve.original_mnemonic and not ctx.las_file.curves[global_idx].original_mnemonic:
                ctx.las_file.curves[global_idx].original_mnemonic = curve.original_mnemonic
            ctx.las_file.curves[global_idx].mnemonic = curve.mnemonic
            ctx.las_file.curves_order[global_idx] = deduped_order[i]

    # Use ``>`` for consistency with data_reader.py and models.py which use ``>``
    # throughout all MAX_DATA_LINES guards (accepts files at exactly MAX_DATA_LINES).
    if actual_count > _data_reader.MAX_DATA_LINES:
        raise LASParseError(
            f"ASCII data line count ({actual_count}) exceeds maximum allowed "
            f"({_data_reader.MAX_DATA_LINES}). The file may be malformed or corrupt."
        )
    if num_curves >= _data_reader.MAX_CURVES:
        raise LASParseError(
            f"Curve count ({num_curves}) exceeds maximum allowed "
            f"({_data_reader.MAX_CURVES}). The file may be malformed or corrupt."
        )

    # Combined bound: protect against combination attacks where individual
    # curve_count and data_line_count checks pass but product exhausts memory.
    if num_curves * actual_count > _data_reader.MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Total allocation ({num_curves} curves x {actual_count} lines = "
            f"{num_curves * actual_count} elements) exceeds maximum allowed "
            f"({_data_reader.MAX_TOTAL_ELEMENTS}). The file may be malformed or corrupt."
        )

    # F2-006: Cumulative cross-section allocation check.
    # Each section passes the per-section bounds above, but multiple
    # sections can collectively exceed MAX_TOTAL_ELEMENTS (matching
    # the from_dict pattern at models.py:2286-2310).
    ctx.cumulative_elements += num_curves * actual_count
    if ctx.cumulative_elements > _data_reader.MAX_TOTAL_ELEMENTS:
        raise LASParseError(
            f"Cumulative cross-section allocation "
            f"({ctx.cumulative_elements} elements across "
            f"{ctx.current_data_section_idx + 1} sections) exceeds "
            f"maximum allowed ({_data_reader.MAX_TOTAL_ELEMENTS}). "
            f"The file may be malformed or corrupt."
        )

    # PERF-03: Pre-allocate numpy arrays for numeric curves.
    string_data_lists: dict[int, list[str]] = {}
    for i, curve in enumerate(section_curves):
        if string_curves.get(i, False):
            string_data_lists[i] = []
        else:
            arr = np.zeros(actual_count, dtype=np.float64)
            data_section.data[curve.mnemonic] = arr
            # F2-014: las_file.logs assigned via defensive copy after data
            # fill to avoid shared mutable ndarray between LASFile.logs
            # and DataSection.data — in-place mutations were silently
            # corrupting both views.

    # Fill arrays by index (no list accumulation overhead for numerics)
    numeric_arrays = [
        data_section.data[c.mnemonic] if not string_curves.get(i, False) else None
        for i, c in enumerate(section_curves)
    ]
    idx = 0
    extra_count = 0  # I2-XPD-03: Row-count accumulator for extra columns
    short_count = 0  # I2-XPD-03: Row-count accumulator for short rows
    _fc = [0]  # F-002: Mutable counter for float conversion diagnostics
    # F-39, F-40, I2-F-03: Desanitize reversal warning dedup flags.
    _desan_warned_ws = False   # whitespace→underscore (F-39)
    _desan_warned_tab = False  # tab→space (F-40)
    _desan_warned_empty = False  # empty→"-" (I2-F-03)
    for line in ctx.ascii_data_lines:
        # Skip comment lines and blank/whitespace-only lines.
        # F-32: EMPTY_PATTERN was defined at module level but never
        # used in this loop — blank lines split to [''] and produce
        # a full row of null_value entries, silently inflating data.
        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            continue

        # Split by delimiter.
        # F-I2-M01: Strip the line before splitting with TAB/COMMA
        # delimiters to avoid leading whitespace producing an empty
        # first token and column shift.  SPACE mode is unaffected
        # (str.split(None) strips implicitly).  Consistent with
        # data_reader.py which strips before all delimiter splits.
        # F2-015: Use csv.reader for TAB/COMMA delimiters so values
        # containing the delimiter inside double-quotes are NOT
        # incorrectly split (e.g., "Run 1, Tool A" stays as one token
        # with COMMA delimiter).  csv.QUOTE_MINIMAL handles CSV-style
        # quoting: fields are quoted only when they contain the
        # delimiter, quotechar, or line terminator.
        if delimiter == " ":
            values = line.split(maxsplit=_max_tokens)
        else:
            # str.split(delimiter) avoids the csv.reader quoting
            # asymmetry with the writer (which uses raw
            # delimiter.join()).  csv.reader with QUOTE_MINIMAL
            # interprets " as CSV quoting; the writer does not emit
            # CSV quotes — causing roundtrip data corruption for
            # string values containing double-quote characters.
            values = line.strip().split(delimiter, maxsplit=_max_tokens)
            # Strip trailing empty strings from non-space delimiters
            # (e.g., trailing COMMA produces phantom empty column).
            # Space-delimited split handles this automatically.
            while values and values[-1] == "":
                values.pop()

        # Warn about extra columns being silently discarded
        if len(values) > num_curves:
            extra_count += 1

        # F-11: Warn when non-wrapped data lines have fewer values than
        # declared curves.  Short rows in wrapped mode are expected
        # (values span multiple lines), so this warning only fires in
        # non-wrapped (WRAP=NO) mode.
        if len(values) < num_curves:
            is_not_wrapped = ctx.las_file.version.wrap.upper() != "YES"
            if is_not_wrapped:
                short_count += 1

        # Pad with null values if needed.
        # String curves use "" (empty string) to avoid width-ambiguity
        # from str(null_value) padding (matching data_reader.py:836-840).
        while len(values) < num_curves:
            if string_curves.get(len(values), False):
                values.append("")
            else:
                values.append(str(null_value))

        for i in range(num_curves):
            val_str = values[i].strip()
            # F-007: Reverse the writer's _sanitize_las_value #-prefix escape
            val_str = _desanitize_las_value(val_str)
            if string_curves.get(i, False):
                # F-39, F-40, I2-F-03: Desanitize reversal warnings for
                # writer-side one-way sanitization transformations.  Each
                # transformation is lossy (original form cannot be
                # recovered), so we warn rather than blindly convert.
                # Warnings are per-section deduplicated to avoid log spam.
                if delimiter == " " and "_" in val_str and not _desan_warned_ws:
                    _desan_warned_ws = True
                    warnings.warn(
                        "String curve data contains underscore "
                        "characters with SPACE delimiter. Original "
                        "whitespace characters may have been replaced "
                        "with underscores by the writer. Roundtrip "
                        "fidelity may be lost.",
                        UserWarning,
                        stacklevel=2,
                    )
                if "  " in val_str and not _desan_warned_tab:
                    _desan_warned_tab = True
                    warnings.warn(
                        "String curve data contains consecutive spaces. "
                        "Original tab characters may have been replaced "
                        "with spaces by the writer. Roundtrip fidelity "
                        "may be lost.",
                        UserWarning,
                        stacklevel=2,
                    )
                if delimiter == " " and val_str == "-" and not _desan_warned_empty:
                    _desan_warned_empty = True
                    warnings.warn(
                        "String curve data contains '-' value with SPACE "
                        "delimiter. This may be an originally-empty "
                        "string value replaced by the writer sentinel. "
                        "Roundtrip fidelity may be lost.",
                        UserWarning,
                        stacklevel=2,
                    )
                string_data_lists[i].append(val_str)
            else:
                val = _to_finite_float(val_str, null_value, _failure_counter=_fc)
                arr = numeric_arrays[i]  # type: ignore[assignment]
                if arr is None:
                    raise LASParseError(
                        f"Internal error: numeric array '{i}' was not pre-allocated"
                    )
                arr[idx] = val

        idx += 1

    # I2-XPD-03: Emit per-section summary warnings with total affected
    # row counts, replacing the previous boolean-once pattern that
    # suppressed all diagnostics after the first occurrence.  This
    # allows automated data quality tools to enumerate affected rows.
    if extra_count > 0:
        logger.warning(
            "Data section '%s': %d row(s) had more values than the %d "
            "declared curves. Extra columns were silently discarded.",
            ctx.current_section_name or "ASCII",
            extra_count,
            num_curves,
        )
    if short_count > 0:
        logger.warning(
            "Data section '%s': %d row(s) had fewer values than the %d "
            "declared curves. Missing values are filled with the null "
            "value (%s).",
            ctx.current_section_name or "ASCII",
            short_count,
            num_curves,
            null_value,
        )

    # F-002: Emit diagnostic warning when float conversions fail in
    # the LAS 3.0 data path — data processing is correct (values fall
    # back to null_value), but users should know about unparseable data.
    if _fc[0] > 0:
        logger.warning(
            "Data section '%s': %d value(s) could not be converted "
            "to finite floats and were replaced with the null value "
            "(%s).",
            ctx.current_section_name or "ASCII",
            _fc[0],
            null_value,
        )

    # F2-014: Copy filled numeric arrays from data_section.data to
    # las_file.logs for independent views.  The allocation above only
    # assigned to data_section.data; the fill loop wrote values via
    # numeric_arrays (which references data_section.data).  Now copy
    # the fully-populated arrays so in-place mutations on one view do
    # not silently corrupt the other.
    if is_first_section:
        for curve in section_curves:
            if curve.mnemonic in data_section.data:
                ctx.las_file.logs[curve.mnemonic] = (
                    data_section.data[curve.mnemonic].copy()
                )

    # Convert string data lists to numpy arrays
    for i, curve in enumerate(section_curves):
        if i in string_data_lists:
            string_arr = np.array(string_data_lists[i], dtype=object)
            data_section.string_data[curve.mnemonic] = string_arr
            if is_first_section:
                # F2-014: Defensive copy — prevents shared-reference
                # mutation between LASFile.string_data and
                # DataSection.string_data.
                ctx.las_file.string_data[curve.mnemonic] = string_arr.copy()

    # Store data section (LAS 3.0)
    ctx.las_file.data_sections.append(data_section)
