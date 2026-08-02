"""LAS 3.0 ASCII data processing — extracted from parser.py:LASParser.

Contains :func:`process_ascii_data` (formerly
``LASParser._process_ascii_data``) and :func:`_deduplicate_curves`
(formerly ``LASParser._deduplicate_curves``), moved verbatim from
parser.py as a pure mechanical extraction.

All state that was previously on ``self`` is passed through an
:class:`AsciiDataContext` dataclass.
"""

from __future__ import annotations

import re
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
    _to_integer_value,
)
from .exceptions import LASDataError, LASParseError
from .models import ArrayElementInfo, CurveDefinition, DataSection


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


# N-I-31: Private attribute names used to record null-value fill cells on a
# DataSection when the ~Well section is not yet known at fill time.  The
# cells are re-filled with the declared NULL once the well section is parsed.
_NULL_FILL_CELLS_ATTR = "_pylasdev_null_fill_cells"
_NULL_FILL_SENTINEL_ATTR = "_pylasdev_null_fill_sentinel"
# M-04: Private attribute marking the DataSection whose arrays were copied
# into the top-level ``las_file.logs`` view (the first LOG_DATA section at
# fill time).  ``_reconcile_null_sentinels`` only writes fill cells back
# into the logs view of the SAME section — writing another section's
# fill-cell row into ``las_file.logs`` would clobber genuine values of a
# DIFFERENT section (e.g. a CORE_DATA section's null fill overwriting the
# LOG_DATA section's real GR value).
_NULL_LOGS_OWNER_ATTR = "_pylasdev_logs_owner"

# M-63: Python-object accumulation caps.  ``MAX_TOTAL_ELEMENTS`` guards
# numpy array allocation at ~8 B/element (~8 GB intent), but ``fill_cells``
# tuples (~100 B each) and per-row string objects (~50-100 B each) bypass
# that accounting — a file that passes the element guard can amplify to
# ~100 GB of Python objects (measured ~96-103 B/fill-cell tuple; the
# guard's own comment says ~8 B/element).  Cap the tracked counts at a
# fraction of the element budget so Python-object memory stays within the
# same order as the guard's intent.  When a cap is exceeded the file is
# malformed/crafted — reject like the other MAX_* guards.
_MAX_FILL_CELLS = _data_reader.MAX_TOTAL_ELEMENTS // 12
_MAX_STRING_VALUES = _data_reader.MAX_TOTAL_ELEMENTS // 12


def _reconcile_null_sentinels(las_file: LASFile) -> None:
    """Re-fill tracked null cells with the well's declared NULL (N-I-31).

    When a LAS 3.0 data section is processed before the ~Well section, its
    null-value fill cells (padding for short rows, conversion failures) are
    baked with the default sentinel (-999.25) because the declared NULL is
    not yet known.  Once the well declares a different NULL, later
    ``process_ascii_data`` calls invoke this helper to replace the tracked
    fill cells so the in-memory data agrees with the declared sentinel —
    downstream consumers using the declared NULL no longer misread fill
    cells as real data.

    Genuine data values that happen to equal the default sentinel are NOT
    touched: only positions recorded at fill time are re-filled.

    Args:
        las_file: The LASFile being built.  Prior data sections are
            mutated in place.
    """
    try:
        declared_null = _get_null_value(las_file.well)
    except LASParseError:
        # Non-finite declared NULL already raises during the fill path;
        # nothing safe to reconcile against.
        return
    for data_section in las_file.data_sections:
        fill_cells = getattr(data_section, _NULL_FILL_CELLS_ATTR, None)
        if not fill_cells:
            continue
        sentinel = getattr(data_section, _NULL_FILL_SENTINEL_ATTR, None)
        if sentinel is None or sentinel == declared_null:
            # M-71: No re-fill is needed (cells already carry the declared
            # NULL, or no sentinel was recorded), but the tracking list
            # MUST still be released — the previous ``continue`` bypassed
            # the clear below, permanently retaining the (row, col) list
            # on the returned LASFile.
            setattr(data_section, _NULL_FILL_CELLS_ATTR, [])
            continue
        for row, col in fill_cells:
            if col >= len(data_section.curves_order):
                continue
            mnemonic = data_section.curves_order[col]
            arr = data_section.data.get(mnemonic)
            if arr is not None and row < len(arr):
                arr[row] = declared_null
                # M-04: Keep the top-level logs convenience view in sync
                # ONLY when this section owns it.  ``las_file.logs`` is a
                # defensive copy of the FIRST LOG_DATA section's arrays
                # made at fill time; a DIFFERENT section's fill-cell row
                # must never be written into another section's logs array
                # (an earlier CORE_DATA section's null fill would clobber
                # the LOG_DATA section's genuine value).
                if getattr(data_section, _NULL_LOGS_OWNER_ATTR, False):
                    log_arr = las_file.logs.get(mnemonic)
                    if log_arr is not None and row < len(log_arr):
                        log_arr[row] = declared_null
        setattr(data_section, _NULL_FILL_CELLS_ATTR, [])
        warnings.warn(
            f"Data section '{data_section.name or 'ASCII'}': null-value "
            f"fill cells were baked with the default sentinel ({sentinel:g}) "
            f"before the ~Well section declared NULL={declared_null:g}. "
            f"Fill cells were updated to match the declared NULL.",
            UserWarning,
            stacklevel=3,
        )


def _detect_actual_wrap_las30(
    data_lines: list[str],
    n_curves: int,
    delimiter: str,
    declared_wrap: str | None,
) -> bool:
    """Detect actual wrap from the section's collected data lines.

    M-05/M-07: The LAS 3.0 path must apply the SAME content-based wrap
    detection as ``data_reader._detect_actual_wrap`` (F-H01) — running
    regardless of the declared WRAP header, and using the hardened 4-line
    majority vote instead of deciding on the first data line only.  The
    declared WRAP header may be wrong: mislabeled Petrel exports claim
    WRAP=YES with non-wrapped data, and WRAP=NO/absent files can contain
    genuinely wrapped data (which would otherwise be silently misparsed
    with a DEPT-shift).

    A line is "full" (non-wrapped evidence) when it carries the complete
    row (``len >= curve_count``) for every delimiter — trailing empties
    are stripped, so a wrapped depth line is exactly 1 value and a
    wrapped continuation line carries ``curve_count-1`` values, both
    partial evidence.  Decision (mirrors data_reader):

    - First line full → non-wrapped immediately (a wrapped first line is
      always a depth line with exactly 1 value).
    - Otherwise, if >= 2 full lines among the first 4 → non-wrapped
      (sparse leading rows then full rows).
    - If >= 3 partial lines among the first 4 → wrapped.
    - Ties fall back to the declared WRAP header, then to wrapped
      (conservative).

    Args:
        data_lines: Collected ASCII data lines for the current section
            (may include comment and blank lines, which are skipped).
        n_curves: Number of curves declared for the section.
        delimiter: Data column delimiter character.
        declared_wrap: Declared WRAP header value ("YES"/"NO"), or None.

    Returns:
        True when the data is genuinely wrapped.
    """
    window: list[int] = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Lazy import mirrors the existing `from .parser import ...`
        # pattern inside process_ascii_data: _data_section_reader imports
        # data_reader at module level, so a top-level import here would
        # create a cycle through data_reader.
        from ._data_section_reader import _split_data_line

        values = _split_data_line(stripped, delimiter)
        window.append(len(values))
        if len(window) >= 4:
            break

    if not window:
        # No data lines — nothing to classify as wrapped.
        return False

    def _is_full(n: int) -> bool:
        return n >= n_curves

    # First line full → non-wrapped (wrapped first line is always depth).
    # M-38 (las30 side): BUT a WRAP=YES header with a COMPLETE first row
    # can still be a genuine mixed-wrap file (per-row line-width wrapping):
    # first row complete, continuation lines wrapped.  The first-line-full
    # rule serves the COMMON mislabeled case (WRAP=YES but data is fully
    # non-wrapped — all lines full).  When the header declares WRAP=YES
    # and later window lines are partial (continuation/depth evidence),
    # fall through to the majority vote below instead of short-circuiting
    # to non-wrapped.  For LAS 3.0 (no wrapped reader) the majority vote
    # classifies such a file as wrapped → the caller raises the loud
    # "LAS 3.0 WRAP=YES is not supported" LASParseError instead of
    # silently misparsing.
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

    # H-02 (las30 side): curve-count mismatch guard.  When ~C declares
    # MORE curves than ~A rows contain (e.g. 3 curves declared but every
    # row carries 2 values), every line is "partial" and the majority vote
    # below would classify the file as WRAPPED — routing a WRAP=NO file to
    # the loud (but factually wrong) "LAS 3.0 WRAP=YES is not supported"
    # rejection.  A genuine wrapped file's line lengths VARY (depth lines
    # carry exactly 1 value and continuation lines curve_count-1); a
    # uniform short length L (1 < L < curve_count) across the whole
    # window, with NO 1-value depth line, is a column-count mismatch —
    # treat as non-wrapped so the graceful short-row null-fill preserves
    # the data (mirrors data_reader.py:622-628).
    if (
        len(window) >= 2
        and full_count == 0
        and len(set(window)) == 1
        and 1 < window[0] < n_curves
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


# M-08: LAS 3.0 spec-form array channels are written with REPEATED PLAIN
# mnemonics plus a per-element ``{A:N}`` format code, e.g.::
#
#     NMR .ms : NMR Echo Array {A:0}
#     NMR .ms : NMR Echo Array {A:5}
#
# The parser only builds ``array_info`` from BRACKET mnemonics
# (``NMR[1]``/``NMR[2]`` — see parser.py ARRAY_MNEMONIC_PATTERN); the
# spec-form above arrives as duplicate plain mnemonics with
# ``data_format="A"`` and ``array_info=None``.  The old code deduplicated
# them (NMR/NMR_2) and routed them as STRING curves, discarding the array
# structure and spacing metadata.  Detect the repeated-plain-mnemonic
# pattern and synthesize ``array_info`` so the channel routes to numeric
# arrays exactly like bracket-notation.
_SPEC_FORM_ARRAY_RE = re.compile(r"\{A:(?P<offset>[-\d.]*)\}")


def _spec_form_group_data_is_numeric(
    data_lines: list[str] | None,
    delimiter: str | None,
    indices: list[int],
) -> bool:
    """Return True when every value in the group's columns is numeric.

    F-06: Confirms a repeated-plain-mnemonic A-format group is a genuine
    spec-form array channel.  The parser strips the ``{A:N}`` marker from
    curve descriptions (parser.py:2502-2504) before this layer runs, so
    the marker is unobservable here — a repeated plain mnemonic with
    ``data_format="A"`` is ambiguous between a spec-form array channel and
    duplicate STRING curves.  The DATA is the only unambiguous
    discriminator: array channels carry numeric values, duplicate STRING
    curves carry non-numeric values (e.g. two ``LITH ... {A}`` entries
    holding "SAND SHALE").  A group whose columns contain ANY non-numeric
    value is a string curve and must NOT be reclassified (pre-fix
    behavior: deduplicate as STRING curves, values preserved).

    Numeric compatibility mirrors the fill loop's ``_to_finite_float``:
    empty/missing tokens are null-compatible (not string evidence).

    Args:
        data_lines: Collected ASCII data lines for the section, or None.
        delimiter: Data column delimiter character, or None.
        indices: Curve positions (columns) of the candidate group.

    Returns:
        True only when the group is unambiguously numeric.
    """
    if not data_lines or delimiter is None:
        return False
    from ._data_section_reader import _split_data_line

    _failure_counter = [0]
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = _split_data_line(stripped, delimiter)
        for col in indices:
            if col >= len(values):
                # Short row — the cell is null-filled downstream (null for
                # numeric curves), which is not string evidence.
                continue
            _before = _failure_counter[0]
            # null_value argument is irrelevant to the counter: only the
            # parse success/failure of the token is being probed.
            _to_finite_float(values[col], 0.0, _failure_counter=_failure_counter)
            if _failure_counter[0] > _before:
                return False
    return True


def _build_spec_form_array_info(
    section_curves: list[CurveDefinition],
    data_lines: list[str] | None = None,
    delimiter: str | None = None,
) -> list[CurveDefinition]:
    """Detect spec-form array channels and attach ``array_info`` (M-08).

    A group of 2+ curves with the SAME plain mnemonic, ``data_format``
    ``"A"``, and no ``array_info`` is a LAS 3.0 spec-form array channel.
    Each member is rewritten with a bracket mnemonic (``NMR[1]``,
    ``NMR[2]``, ...) and an ``ArrayElementInfo`` carrying the sequential
    index and the ``{A:N}`` time offset from the description (when the
    parser preserved it — see the parser-side coordination note).

    F-06: Synthesis only fires when the group is confirmed as a GENUINE
    array channel — its data must be unambiguously numeric (see
    :func:`_spec_form_group_data_is_numeric`).  Duplicate A-format STRING
    curves (e.g. two ``LITH ... {A}`` entries) are left untouched so the
    caller's dedup preserves them as STRING curves with values intact.

    Returns a NEW list; the caller's list is not mutated.
    """
    # Group consecutive runs of identical plain mnemonics.  LAS 3.0 array
    # elements are contiguous (array_info validation enforces positional
    # contiguity), so only a consecutive run is treated as a channel.
    groups: list[tuple[str, list[int]]] = []
    current_base: str | None = None
    current_indices: list[int] = []
    for i, curve in enumerate(section_curves):
        if (
            curve.array_info is not None
            or (curve.data_format or "").upper() != "A"
            or "[" in curve.mnemonic
        ):
            if current_base is not None:
                groups.append((current_base, current_indices))
                current_base = None
                current_indices = []
            continue
        base = curve.mnemonic.upper()
        if base == current_base:
            current_indices.append(i)
        else:
            if current_base is not None:
                groups.append((current_base, current_indices))
            current_base = base
            current_indices = [i]
    if current_base is not None:
        groups.append((current_base, current_indices))

    if not any(len(indices) >= 2 for _, indices in groups):
        return section_curves

    result = list(section_curves)
    for base, indices in groups:
        if len(indices) < 2:
            continue
        # F-06: Only synthesize GENUINE array channels.  A repeated plain
        # mnemonic with data_format "A" is ambiguous (marker stripped by
        # parser); numeric data confirms the array channel, non-numeric
        # data means duplicate STRING curves whose values must be
        # preserved (pre-fix behavior).
        if not _spec_form_group_data_is_numeric(data_lines, delimiter, indices):
            continue
        for pos, curve_idx in enumerate(indices, start=1):
            curve = result[curve_idx]
            offset: float | None = None
            match = _SPEC_FORM_ARRAY_RE.search(curve.description or "")
            if match and match.group("offset"):
                try:
                    offset = float(match.group("offset"))
                except ValueError:
                    offset = None
            result[curve_idx] = CurveDefinition(
                mnemonic=f"{base}[{pos}]",
                unit=curve.unit,
                api_code=curve.api_code,
                description=curve.description,
                original_mnemonic=curve.original_mnemonic or curve.mnemonic,
                data_format=curve.data_format,
                array_info=ArrayElementInfo(
                    base_name=base, index=pos, time_offset=offset,
                ),
            )
    return result


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

    # N-I-31: When the ~Well section is already known at this point, re-fill
    # null-value fill cells of EARLIER data sections that were processed
    # before ~Well declared its NULL (they were baked with the default
    # -999.25 sentinel).  Runs even when the current section is empty so a
    # trailing non-data section cannot skip reconciliation.
    well_known = bool(ctx.las_file.well.get("NULL"))
    if well_known:
        _reconcile_null_sentinels(ctx.las_file)

    if not ctx.ascii_data_lines:
        return

    # F-M-004 / M-05 / M-07: LAS 3.0 WRAP detection — content-based,
    # unconditional (mirroring data_reader F-H01).  The declared WRAP
    # header may be wrong (mislabeled Petrel exports claim WRAP=YES with
    # non-wrapped data, or WRAP=NO files contain genuinely wrapped data).
    # The previous code gated detection on the declared WRAP header and
    # decided COMMA/TAB on the FIRST data line only — WRAP=NO/absent
    # files with genuinely wrapped data were silently misparsed (DEPT
    # shift) and sparse-first-row files were falsely rejected.  Use the
    # same hardened 4-line majority vote as data_reader._detect_actual_wrap.
    # P-15: Single-curve files (n_curves <= 1) are exempt — like
    # data_reader's curve_count<=1 rule, they cannot distinguish wrap
    # mode and are treated as non-wrapped (previously always rejected).
    if ctx.section_curve_end_idx is not None:
        n_curves = len(ctx.las_file.curves[ctx.section_curve_start_idx : ctx.section_curve_end_idx])
    else:
        n_curves = len(ctx.las_file.curves[ctx.section_curve_start_idx :])
    actual_wrap = False
    if n_curves > 1:
        actual_wrap = _detect_actual_wrap_las30(
            ctx.ascii_data_lines,
            n_curves,
            ctx.las_file.version.delimiter_char,
            ctx.las_file.version.wrap,
        )
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
    # M-08: Detect LAS 3.0 spec-form array channels (repeated plain
    # mnemonics + {A:N} format codes) BEFORE deduplication.  These arrive
    # as duplicate plain-mnemonic A-format curves with array_info=None and
    # were previously deduplicated into STRING curves, discarding the array
    # structure and spacing metadata.  Synthesize array_info (bracket
    # mnemonics + base_name/index/offset) so the channel routes to numeric
    # arrays exactly like bracket-notation.
    # F-06: Pass the section's data lines so synthesis only fires when the
    # group's data is confirmed numeric — duplicate A-format STRING curves
    # (non-numeric data) are left as strings, preserving their values.
    section_curves = _build_spec_form_array_info(
        section_curves,
        data_lines=ctx.ascii_data_lines,
        delimiter=delimiter,
    )
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
    # (see F2-07 block below) is deferred until after the actual_count
    # > 0 check — see F2-07 writeback block below.
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

    # L-03: Detect {I} integer-format curves.  Stored as int64 (when the
    # null sentinel is integral) and parsed via int() so values above 2^53
    # are not silently rounded by float().
    integer_curves = {
        i: c.data_format == "I" for i, c in enumerate(section_curves)
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
    # L-01: Use warnings.warn (not logger.warning) for symmetry with the
    # LAS 1.2/2.0 data-quality diagnostics (data_reader.py) so LAS 3.0
    # corruption is visible to warnings-API monitoring.
    if not well_known:
        warnings.warn(
            f"NULL value not found in well section for data section "
            f"'{ctx.current_section_name or 'ASCII'}'; using default null "
            f"value ({null_value:.4g}). This may indicate that the data "
            f"section precedes the ~Well section in the file.",
            UserWarning,
            stacklevel=2,
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
    # L-02: Apply to EVERY data section, not just the first.  The
    # writeback is scoped to the section's own curve slice
    # (section_curve_start_idx + i), so dedup renames in non-first
    # sections propagate to the global curves/curves_order that the
    # writer and to_dict() consume — without them, a re-read of the
    # written file sees more data columns than ~C curve definitions.
    for i, curve in enumerate(section_curves):
        global_idx = ctx.section_curve_start_idx + i
        if curve.original_mnemonic and not ctx.las_file.curves[global_idx].original_mnemonic:
            ctx.las_file.curves[global_idx].original_mnemonic = curve.original_mnemonic
        ctx.las_file.curves[global_idx].mnemonic = curve.mnemonic
        ctx.las_file.curves_order[global_idx] = deduped_order[i]
        # M-08: Propagate array_info synthesized for spec-form array
        # channels to the GLOBAL curve definitions.  The parser builds
        # array_info for bracket mnemonics directly; spec-form channels
        # get array_info here, and the LASFile-level format-vs-placement
        # validation requires the global curve to be an array element
        # (data_format "A" + array_info) before its numeric data may live
        # in logs.
        if curve.array_info is not None:
            ctx.las_file.curves[global_idx].array_info = curve.array_info

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
    # L-03: {I} integer-format curves get int64 arrays when the null
    # sentinel is integral — int64 would truncate a fractional NULL (e.g.
    # -999.25 → -999), silently corrupting null cells, so the int64 branch
    # is gated on null integrality (same rule as data_reader).
    # M-62 (coordination): the int64 branch must ALSO check int64
    # representability.  A huge integral sentinel (>= 2^63, e.g. NULL 1e19)
    # passes ``float(x).is_integer()`` but int64 assignment of
    # ``int(null_value)`` (short-row/conversion-failure fills at line ~950)
    # raises OverflowError that escapes the LASParseError-only boundary.
    # Route such sentinels to the object-dtype path (EXT-04) which holds
    # arbitrary Python ints exactly — mirroring the data_reader agent's
    # M-62 fix (_INT64_MIN/_INT64_MAX).
    _null_is_integral = (
        float(null_value).is_integer()
        and _data_reader._INT64_MIN <= int(null_value) <= _data_reader._INT64_MAX
    )
    string_data_lists: dict[int, list[str]] = {}
    for i, curve in enumerate(section_curves):
        if string_curves.get(i, False):
            string_data_lists[i] = []
        elif integer_curves.get(i, False) and _null_is_integral:
            # L-03: int64 storage for {I} curves — preserves exact integer
            # values above 2^53 that float64 cannot represent.
            data_section.data[curve.mnemonic] = np.zeros(
                actual_count, dtype=np.int64
            )
        elif integer_curves.get(i, False):
            # EXT-04: fractional declared NULL — object dtype preserves
            # exact {I} integers above 2^53 while null cells keep the
            # fractional sentinel (int64 would truncate -999.25 → -999).
            data_section.data[curve.mnemonic] = np.zeros(
                actual_count, dtype=object
            )
        else:
            data_section.data[curve.mnemonic] = np.zeros(
                actual_count, dtype=np.float64
            )
            # F2-014: las_file.logs assigned via defensive copy after data
            # fill to avoid shared mutable ndarray between LASFile.logs
            # and DataSection.data — in-place mutations were silently
            # corrupting both views.

    # Fill arrays by index (no list accumulation overhead for numerics)
    # L-03: numeric arrays may be float64 or int64 (for {I} curves) —
    # annotate the union explicitly so mypy accepts both dtypes.
    numeric_arrays: list[np.ndarray | None] = [
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
    # N-I-31: When the ~Well section is not yet known, record every cell
    # that is filled with the default null sentinel (padding for short
    # rows, conversion failures) so a later pass can re-fill them with the
    # well's declared NULL.  Genuine data values are never tracked.
    track_fill = not well_known
    fill_cells: list[tuple[int, int]] = []  # (row_idx, col_idx) null fills

    # M-63: Bound the Python-object tracking/accumulation structures.
    # fill_cells tuples (~100 B each) and string values (~50-100 B each)
    # amplify beyond MAX_TOTAL_ELEMENTS' ~8 B/element accounting — a
    # crafted file can pass the element guard while allocating ~100 GB of
    # Python objects.  Reject once a cap is exceeded (same philosophy as
    # the other MAX_* guards).
    def _check_fill_cell_cap() -> None:
        if len(fill_cells) >= _MAX_FILL_CELLS:
            raise LASParseError(
                f"Data section '{ctx.current_section_name or 'ASCII'}': "
                f"tracked null-fill cells exceed maximum allowed "
                f"({_MAX_FILL_CELLS}). The file may be malformed or corrupt."
            )

    _string_value_count = 0

    def _check_string_cap() -> None:
        nonlocal _string_value_count
        if _string_value_count >= _MAX_STRING_VALUES:
            raise LASParseError(
                f"Data section '{ctx.current_section_name or 'ASCII'}': "
                f"string curve values exceed maximum allowed "
                f"({_MAX_STRING_VALUES}). The file may be malformed or corrupt."
            )
        _string_value_count += 1

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
            _pad_idx = len(values)
            if string_curves.get(_pad_idx, False):
                values.append("")
            else:
                values.append(str(null_value))
                if track_fill:
                    _check_fill_cell_cap()
                    fill_cells.append((idx, _pad_idx))

        for i in range(num_curves):
            # IT3-F1: Do NOT strip per-token whitespace.  The LAS 1.2/2.0
            # path (_split_data_line) keeps token-edge whitespace, so
            # stripping here silently destroyed leading/trailing spaces in
            # {S} string data on LAS 3.0 reads.  Numeric parsing is
            # whitespace-tolerant (float()) and _desanitize_las_value
            # handles whitespace-prefixed "_#" escapes (case 2), so the
            # token is passed through untouched.
            val_str = values[i]
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
                _check_string_cap()
            elif integer_curves.get(i, False):
                # L-03/EXT-04: {I} curve — parse via int() to preserve
                # exactness above 2^53 (float() would round).  With a
                # fractional NULL the array is object dtype and failures
                # return the float sentinel (not int(null_value)).
                _fc_before = _fc[0]
                int_val = _to_integer_value(
                    val_str, null_value, _failure_counter=_fc,
                    _null_as_float=not _null_is_integral,
                )
                if track_fill and (_fc[0] > _fc_before or not val_str):
                    _check_fill_cell_cap()
                    fill_cells.append((idx, i))
                arr = numeric_arrays[i]
                if arr is None:
                    raise LASParseError(
                        f"Internal error: numeric array '{i}' was not pre-allocated"
                    )
                # L-03: int64-allocated arrays accept int values directly.
                arr[idx] = int_val
            else:
                _fc_before = _fc[0]
                val = _to_finite_float(val_str, null_value, _failure_counter=_fc)
                if track_fill and (_fc[0] > _fc_before or not val_str):
                    # Conversion failure or empty token — the cell was
                    # replaced with the default null sentinel.  Empty
                    # tokens return null_value without incrementing the
                    # failure counter, so they need an explicit check.
                    _check_fill_cell_cap()
                    fill_cells.append((idx, i))
                arr = numeric_arrays[i]
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
    # L-01: Use warnings.warn (not logger.warning) for symmetry with the
    # LAS 1.2/2.0 data-quality diagnostics (data_reader.py) so LAS 3.0
    # data-quality issues are visible to warnings-API monitoring.
    if extra_count > 0:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"{extra_count} row(s) had more values than the {num_curves} "
            f"declared curves. Extra columns were silently discarded.",
            UserWarning,
            stacklevel=2,
        )
    if short_count > 0:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"{short_count} row(s) had fewer values than the {num_curves} "
            f"declared curves. Missing values are filled with the null "
            f"value ({null_value}).",
            UserWarning,
            stacklevel=2,
        )

    # F-002: Emit diagnostic warning when float conversions fail in
    # the LAS 3.0 data path — data processing is correct (values fall
    # back to null_value), but users should know about unparseable data.
    if _fc[0] > 0:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"{_fc[0]} value(s) could not be converted to finite floats "
            f"and were replaced with the null value ({null_value}).",
            UserWarning,
            stacklevel=2,
        )

    # F2-014: Copy filled numeric arrays from data_section.data to
    # las_file.logs for independent views.  The allocation above only
    # assigned to data_section.data; the fill loop wrote values via
    # numeric_arrays (which references data_section.data).  Now copy
    # the fully-populated arrays so in-place mutations on one view do
    # not silently corrupt the other.
    # N-I-32: Populate the top-level logs/string_data views from the
    # LOG_DATA section (the main log), not from the FIRST data section
    # regardless of section_type.  Typed-first files (e.g. ~Core_Data
    # before LOG_DATA) previously gave logs = CORE curves with the main
    # DEPT/GR absent from to_dict()["logs"].  Only the first LOG_DATA
    # section populates the view (matching the single-population
    # semantics of the previous order-based gate).
    is_log_data = ctx.current_data_section_type == "LOG_DATA"
    if is_log_data and not ctx.las_file.logs:
        for curve in section_curves:
            if curve.mnemonic in data_section.data:
                ctx.las_file.logs[curve.mnemonic] = (
                    data_section.data[curve.mnemonic].copy()
                )
        # M-04: Record which section's arrays were copied into the
        # top-level logs view so _reconcile_null_sentinels only writes
        # fill cells back into the SAME section's logs arrays (the gate
        # above ensures this is the first LOG_DATA section).
        setattr(data_section, _NULL_LOGS_OWNER_ATTR, True)

    # Convert string data lists to numpy arrays
    for i, curve in enumerate(section_curves):
        if i in string_data_lists:
            string_arr = np.array(string_data_lists[i], dtype=object)
            data_section.string_data[curve.mnemonic] = string_arr

    # M-09: Copy ALL string curves of the first LOG_DATA section to the
    # top-level string_data view.  The gate must be OUTSIDE the per-curve
    # loop (like the numeric twin at the logs population above): the
    # previous gate inside the loop stopped after the FIRST string curve
    # (``not ctx.las_file.string_data`` becomes False once any key exists),
    # silently dropping every later string curve from the top-level view.
    if is_log_data and not ctx.las_file.string_data:
        for i, curve in enumerate(section_curves):
            if i in string_data_lists:
                # F2-014: Defensive copy — prevents shared-reference
                # mutation between LASFile.string_data and
                # DataSection.string_data.
                ctx.las_file.string_data[curve.mnemonic] = (
                    data_section.string_data[curve.mnemonic].copy()
                )

    # N-I-31: Attach the tracked fill cells to the section so a later
    # process_ascii_data call (once ~Well is known) can re-fill them
    # with the declared NULL.  Only needed when the well was unknown.
    if track_fill and fill_cells:
        setattr(data_section, _NULL_FILL_CELLS_ATTR, fill_cells)
        setattr(data_section, _NULL_FILL_SENTINEL_ATTR, null_value)

    # Store data section (LAS 3.0)
    ctx.las_file.data_sections.append(data_section)
