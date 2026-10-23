"""LAS 3.0 ASCII data processing — extracted from parser.py:LASParser.

Contains :func:`process_ascii_data` (formerly
``LASParser._process_ascii_data``) and :func:`_deduplicate_curves`
(formerly ``LASParser._deduplicate_curves``), moved verbatim from
parser.py as a pure mechanical extraction.

All state that was previously on ``self`` is passed through an
:class:`AsciiDataContext` dataclass.
"""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import LASFile

from . import data_reader as _data_reader
from ._data_section_reader import _split_data_line
from .data_reader import (
    _get_null_value,
    _parse_float_with_d_notation,
    _resolve_max_tokens_per_line,
    _suffixed_curve_name,
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
    suffixes ``_2``, ``_3``, etc. are appended to the mnemonic STEM —
    ``NMR[1]`` becomes ``NMR_2[1]``, never ``NMR[1]_2``, which the
    ``CurveDefinition`` mnemonic grammar rejects (F-075/F-077; the LAS
    1.2/2.0 twin builds its names with the same
    ``data_reader._suffixed_curve_name``).  ``original_mnemonic``
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
            new_name = _suffixed_curve_name(name, suffix)
            while new_name in output_names:
                suffix += 1
                new_name = _suffixed_curve_name(name, suffix)
            seen[name] = suffix
            section_curves[i] = CurveDefinition(
                mnemonic=new_name,
                unit=curve.unit,
                api_code=curve.api_code,
                description=curve.description,
                original_mnemonic=name if not curve.original_mnemonic else curve.original_mnemonic,
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
            new_name = _suffixed_curve_name(name, suffix)
            while new_name in output_names:
                suffix += 1
                new_name = _suffixed_curve_name(name, suffix)
            seen[name] = suffix
            section_curves[i] = CurveDefinition(
                mnemonic=new_name,
                unit=curve.unit,
                api_code=curve.api_code,
                description=curve.description,
                original_mnemonic=name if not curve.original_mnemonic else curve.original_mnemonic,
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

# F-129 (reader half): how many data rows the ``{D}``-format channel
# fingerprint looks at.  A Fortran-D numeric column proves itself numeric on
# its FIRST value; a date column never does.  The bound keeps the probe O(1)
# per section regardless of the file size (it is a classification hint, not a
# full validation pass).
_D_FORMAT_PROBE_ROWS = 50


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

    Thin per-caller wrapper: keeps the LAS 3.0 section-line window
    collection (blank/comment skip only — the twin scans the whole file
    for ``~A``; INT-02) and delegates the decision to the shared core
    :func:`pylasdev._data_section_reader.detect_actual_wrap_from_window`
    with ``empty_window_default=False`` (W-4/INT-01 — a blank-only section
    must not be falsely rejected as WRAP=YES).

    M-05/M-07: The LAS 3.0 path must apply the SAME content-based wrap
    detection as the LAS 1.2/2.0 twin (F-H01) — running regardless of the
    declared WRAP header, and using the hardened 4-line majority vote.  The
    shared core's R-1 flowing rule makes an all-full flowing window
    (``[6,6]`` nc=3) return True so ``process_ascii_data``'s loud
    "LAS 3.0 WRAP=YES is not supported" LASParseError fires instead of the
    silent depth-step loss (R-5 — detection-only; NO accumulation in
    ``process_ascii_data``).  The R-3/II-5 discriminator keeps ragged
    WRAP=NO windows (``[3,1,3,1]``/``[3,1,1,3]``) non-wrapped so they parse
    instead of being falsely rejected (NEW-1).

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
    _degraded: list[bool] = []  # M10: per-line recombination-degraded flags
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Lazy import mirrors the existing `from .parser import ...`
        # pattern inside process_ascii_data: _data_section_reader imports
        # data_reader at module level, so a top-level import here would
        # create a cycle through data_reader.
        from ._data_section_reader import _split_data_line_traced

        # F-074: pass ``expected=n_curves`` so the wrap gate sees the SAME
        # column layout the fill loop sees (thousands-grouped COMMA rows are
        # 2N raw tokens but N columns).  Without it a valid WRAP=NO file
        # whose first-row values are comma-grouped counts 2N tokens, the R-1
        # flowing rule fires (2N is a multiple of N and > N) and the LAS 3.0
        # reader rejects the file as a wrapped layout.  warn=False: this is a
        # speculative 4-line window scan — the fill loop re-splits the same
        # lines and emits the recombination diagnostic there.
        # M10: the traced splitter also reports whether recombination merged
        # fragments on this row, so a shrunken (degraded) count is not read
        # as a depth line by the shared detector.
        values, _recombined = _split_data_line_traced(
            stripped, delimiter, expected=n_curves, warn=False
        )
        window.append(len(values))
        _degraded.append(_recombined)
        if len(window) >= 4:
            break

    # Lazy import mirrors the established pattern above (the module-level
    # chain _las30_data → _data_section_reader → data_reader would
    # ImportError when _data_section_reader's names are not yet defined).
    from ._data_section_reader import detect_actual_wrap_from_window

    return detect_actual_wrap_from_window(
        window,
        n_curves,
        declared_wrap,
        empty_window_default=False,
        recombination_degraded=_degraded,
    )


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
# F-11: tolerate whitespace before the closing brace ({A:0 } — the
# official CWLS sample form, e.g. 'NMR Echo Array {A:0 }').  The parser
# preserves the marker verbatim in the description (parser.py:3074-3080)
# and FORMAT_SPEC_PATTERN already accepts \s{0,64} before '}'; the
# synthesis regex must accept the same form or time_offset is silently
# lost and the marker stays in the description (doubled '{A:0 }  {A}'
# artifact on write).
# F-007: the CWLS 3.0 NORMATIVE array-element grammar is
# ``{A[<format>];<spacing><unit>}`` — ``{A;0ms}``, ``{AF;0ms}``,
# ``{AF10.4;5ms}`` (Release 3.00 pp.11/27).  The COLON form ``{A:<offset>}``
# exists only in the CWLS-distributed example file, and it is what
# pylasdev's own writer emits (_writer_base.py:466-499), so BOTH forms
# must be accepted.  Accepted set (unit U3, shared with the parser's
# _FORMAT_SPEC_RE):
#   {A[:<offset>]}                     colon form, offset [-\d.]*
#   {A;<spacing>[unit]}                normative form, spacing [-\d.]+,
#                                      unit optional letters
#   {AF<digits>[.<digits>]<sep>...}    same two forms with an F format body
# Trailing whitespace before '}' is tolerated (F-11).
# Plain markers ({A}, {A } with no colon/semicolon) and non-numeric
# semicolon groups ({A;X}, a string-format suffix group) intentionally do
# NOT match: they carry no spacing and are not array elements.
_SPEC_FORM_ARRAY_RE = re.compile(
    r"\{A(?:F\d*(?:\.\d+)?)?"
    r"(?::(?P<offset>[-\d.]*)|;(?P<spacing>[-\d.]+)[A-Za-z]*)"
    r"\s{0,64}\}"
)


def _spec_form_group_data_is_numeric(
    data_lines: list[str] | None,
    delimiter: str | None,
    indices: list[int],
) -> bool:
    """Return True when every value in the group's columns is numeric.

    F-06: Confirms a repeated-plain-mnemonic A-format group is a genuine
    spec-form array channel.  The parser strips plain ``{A}`` format
    markers from curve descriptions before this layer runs, so a repeated
    plain mnemonic with ``data_format="A"`` is ambiguous between a
    spec-form array channel and duplicate STRING curves (offset-bearing
    ``{A:N}`` markers ARE preserved — see L30-01).  The DATA is the
    unambiguous discriminator for marker-less groups: array channels
    carry numeric values, duplicate STRING curves carry non-numeric
    values (e.g. two ``LITH ... {A}`` entries holding "SAND SHALE").  A
    group whose columns contain ANY non-numeric value is a string curve
    and must NOT be reclassified (pre-fix behavior: deduplicate as
    STRING curves, values preserved).

    Numeric compatibility mirrors the fill loop's ``_to_finite_float``:
    empty/missing tokens are null-compatible (not string evidence).
    M-06: NaN/Inf tokens (``nan``, ``inf``, ``1e999``) are ALSO
    null-compatible — the fill loop parses them numerically and
    null-fills the cell for a numeric curve, so they must not count as
    string evidence (pre-fix: a NaN token in a >=2-element spec-form
    group misrouted the whole channel to duplicate STRING curves,
    contradicting the fill loop's NaN->null semantics).

    Args:
        data_lines: Collected ASCII data lines for the section, or None.
        delimiter: Data column delimiter character, or None.
        indices: Curve positions (columns) of the candidate group.

    Returns:
        True only when the group is unambiguously numeric.

    .. note::

       F-008: :func:`_build_spec_form_array_info` no longer calls this
       function per group — doing so re-scanned every data line once per
       group (G x R x C token work).  The reader computes the same
       per-column evidence in ONE pass and asks ``_group_is_numeric``.
       This function is kept as the reference predicate (direct callers
       and the pinned tests import it).  It is NOT a semantic oracle for
       that single-pass derivation: the pass splits each line with
       ``expected=len(section_curves)``, so it sees the columns the fill
       loop will actually parse, while this predicate sees the raw
       fragments (a thousands-grouped COMMA row reads as
       ``['1', '234.5']`` here and as ``['1234.5']`` there).  An EMPTY
       ``data_lines`` list answers False here but True in the single pass,
       which has no line to contradict it — unreachable in production,
       because :func:`process_ascii_data` returns before
       ``_build_spec_form_array_info`` when the section has no data lines.
       Token-level empty/short-row/NaN handling IS identical.
    """
    if not data_lines or delimiter is None:
        return False
    from ._data_section_reader import _split_data_line

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
            token = values[col]
            if not token:
                # Empty token — null-compatible, not string evidence.
                continue
            # M-06: mirror the fill loop's NaN->null semantics — a
            # parseable number (including nan/inf, which float() accepts)
            # is numeric evidence; only a genuine parse failure proves
            # string semantics.
            try:
                _parse_float_with_d_notation(token)
            except (ValueError, OverflowError):
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

    E-22: A SINGLE-element group whose member carries the preserved
    ``{A:N}`` spec-form element marker is a lone array channel (e.g. a
    one-element ``{A:0}`` array) — it synthesizes with index 1 and the
    marker's time offset instead of being misrouted to string_data.  A
    marker-less single A-format curve stays untouched (it is a plain
    ``{A}``-style string curve; test-pinned F-06 behavior).

    M-16: For 2+ member groups, the parser preserves ONLY offset-bearing
    ``{A:N}`` markers (L30-01, parser.py) — plain ``{A}`` markers are
    stripped, so a marker-less group is ambiguous between a spec-form
    array and duplicate STRING curves.  The numeric-data probe remains
    the discriminator for those (F-06, test-pinned), but the conversion
    is a DATA-TYPE MUTATION on external files (string_data -> logs) and
    must not be silent: a loud warning names the reclassification.

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

    result = list(section_curves)

    # F-008: compute the numeric evidence ONCE per section as a per-column
    # flag, instead of once per group over every data line.  The old shape
    # called _spec_form_group_data_is_numeric() inside the group loop, and
    # each call re-split EVERY data line while reading only that group's
    # columns — G x R x C token work against the fill loop's R x C, i.e. an
    # amplification of G on a public read API (measured 71 s vs 3.3 s on an
    # 8.1 MB file that passes every MAX_* guard).
    #
    # Only candidate columns are watched (2+ member groups, and
    # single-element groups carrying an {A:N}/{A;N} marker), so the pass
    # costs R x |watched| <= R x C — the fill loop's own order.
    #
    # The PREDICATE is unchanged (this is a cost fix, not a semantics fix):
    #   - empty/missing tokens are null-compatible, not string evidence;
    #   - NaN/Inf tokens are numeric evidence (float() accepts them, and the
    #     fill loop null-fills them) — M-06;
    #   - any genuine parse failure makes that COLUMN string evidence.
    # A marker-bearing duplicate group holding string data still must NOT be
    # synthesized (tests/test_coverage_gaps.py:431 pins it).
    #
    # The split now passes ``expected=len(section_curves)`` so the probe sees
    # the SAME column layout the fill loop sees (thousands recombination
    # included, warning suppressed — the fill loop re-splits the same lines
    # and emits the diagnostic there).
    _numeric_ok: list[bool] | None = None
    _candidate_groups = [
        _indices
        for _base, _indices in groups
        if len(_indices) >= 2 or _SPEC_FORM_ARRAY_RE.search(result[_indices[0]].description or "")
    ]
    _watched_cols = sorted({_col for _indices in _candidate_groups for _col in _indices})
    if data_lines is not None and delimiter is not None and _watched_cols:
        _numeric_ok = [True] * len(section_curves)
        _expected_cols = len(section_curves)
        for _line in data_lines:
            _stripped = _line.strip()
            if not _stripped or _stripped.startswith("#"):
                continue
            if not any(_numeric_ok[_col] for _col in _watched_cols):
                break  # every candidate group already proven string
            _values = _split_data_line(_stripped, delimiter, expected=_expected_cols, warn=False)
            for _col in _watched_cols:
                if not _numeric_ok[_col]:
                    continue
                if _col >= len(_values):
                    # Short row — the cell is null-filled downstream (null for
                    # numeric curves), which is not string evidence.
                    continue
                _token = _values[_col]
                if not _token:
                    # Empty token — null-compatible, not string evidence.
                    continue
                try:
                    _parse_float_with_d_notation(_token)
                except (ValueError, OverflowError):
                    _numeric_ok[_col] = False

    def _group_is_numeric(indices: list[int]) -> bool:
        """Group numeric verdict from the single-pass per-column evidence."""
        return _numeric_ok is not None and all(_numeric_ok[_col] for _col in indices)

    def _synthesize_members(indices: list[int]) -> None:
        """Rewrite the group members at *indices* as array elements.

        Shared by the multi-element (M-08) and single-element (E-22)
        paths.  Extracts each member's ``{A:N}`` time offset, strips the
        marker from the description (the writer re-emits it once from
        ``array_info.time_offset``), and attaches 1-based
        ``ArrayElementInfo``.
        """
        for pos, curve_idx in enumerate(indices, start=1):
            curve = result[curve_idx]
            offset: float | None = None
            description = curve.description or ""
            match = _SPEC_FORM_ARRAY_RE.search(description)
            if match:
                # F-007: the colon form carries the offset, the normative
                # semicolon form carries the spacing — same quantity
                # (ArrayElementInfo.time_offset).  Exactly one group can be
                # populated for a given match.
                _offset_text = match.group("offset") or match.group("spacing")
                if _offset_text:
                    try:
                        offset = float(_offset_text)
                    except ValueError:
                        offset = None
            # L30-01: When the {A:N} marker survived in the description
            # (the parser-side coordination change preserves it for LAS 3.0
            # A-format spec-form curves), strip it here so the WRITER
            # re-emits {A:N} exactly once — from array_info.time_offset
            # (_writer_base.py:350-374).  Keeping the marker would produce
            # a doubled ``NMR Echo Array {A:0}  {A:0}`` on write.
            if match:
                description = _SPEC_FORM_ARRAY_RE.sub("", description).strip()
            try:
                result[curve_idx] = CurveDefinition(
                    mnemonic=f"{base}[{pos}]",
                    unit=curve.unit,
                    api_code=curve.api_code,
                    description=description,
                    original_mnemonic=curve.original_mnemonic or curve.mnemonic,
                    data_format=curve.data_format,
                    array_info=ArrayElementInfo(
                        base_name=base,
                        index=pos,
                        time_offset=offset,
                    ),
                )
            except ValueError as exc:
                # P-11: the parser boundary may only raise LASParseError.
                # A negative {A:N} time offset (e.g. "{A:-5}") parses via
                # float() but is rejected by ArrayElementInfo's >= 0
                # validation — previously the bare ValueError escaped
                # parse() (W-5).
                raise LASParseError(
                    f"Invalid spec-form array element '{base}[{pos}]' ({description!r}): {exc}"
                ) from exc

    for base, indices in groups:
        if len(indices) < 2:
            # E-22: single-element spec-form groups synthesize ONLY when
            # the member carries the preserved '{A:N}' element marker
            # (offset-bearing markers survive the parser; a lone '{A:0}'
            # array channel is a genuine single-element array).  Marker-
            # less single A-format curves (plain '{A}' duplicates, no
            # marker) are string curves — untouched (test-pinned).
            member = result[indices[0]]
            if not _SPEC_FORM_ARRAY_RE.search(member.description or ""):
                continue
            if not _group_is_numeric(indices):
                continue
            _synthesize_members(indices)
            continue
        # F-06: Only synthesize GENUINE array channels.  A repeated plain
        # mnemonic with data_format "A" is ambiguous (plain markers
        # stripped by parser); numeric data confirms the array channel,
        # non-numeric data means duplicate STRING curves whose values must
        # be preserved (pre-fix behavior).
        if not _group_is_numeric(indices):
            continue
        # M-16: offset-bearing '{A:N}' markers ARE observable (L30-01
        # preserves them) — their presence confirms the spec-form channel
        # and the conversion is legitimate.  A marker-less group is
        # ambiguous: the numeric probe says array, but duplicate STRING
        # curves holding numeric-looking codes ('1','2') would be silently
        # reclassified (string_data -> logs data-type mutation).  Warn
        # loudly instead of converting silently.
        has_spec_form_marker = any(
            _SPEC_FORM_ARRAY_RE.search(result[i].description or "") for i in indices
        )
        if not has_spec_form_marker:
            warnings.warn(
                f"Duplicate A-format curve group '{base}' holds numeric-looking "
                f"data and was reclassified as a spec-form array channel "
                f"({', '.join(f'{base}[{pos}]' for pos in range(1, len(indices) + 1))}). "
                f"If these curves were intended as STRING data (e.g. duplicate "
                f"'{base} ... {{A}}' declarations), their values are now stored "
                f"as numeric arrays instead of string_data.",
                UserWarning,
                stacklevel=2,
            )
        _synthesize_members(indices)
    return result


def _split_quoted_data_line(
    stripped: str, delimiter: str, max_tokens: int
) -> list[str] | None:
    """Split a LAS 3.0 data line that carries CWLS double-quoted items (F-011).

    CWLS 3.0 (data records): an item that CONTAINS the delimiting character
    "must be entirely surrounded by a single pair of quotes".  The plain
    delimiter split cannot see that, so ``100,"WELL, A",50`` truncated the
    string at the comma, took ``A`` as the next column and discarded the
    rest as extra columns — every following column shifted.

    Quoting is spec'd for every delimiter (comma, tab and space), so this
    helper is delimiter-agnostic.  ``skipinitialspace`` is enabled only for
    the space delimiter, where a single space separates items; for
    comma/tab it stays off so token-edge whitespace is preserved exactly as
    the plain split preserves it (IT3-F1).

    Returns None when the line is not safely quotable — unbalanced quotes
    (an odd count, i.e. a stray quote character inside a value) or a csv
    parse failure.  The caller then falls back to the plain delimiter
    split, i.e. the pre-fix behaviour, so a quote character in an unquoted
    row cannot change how that row is read.
    """
    if '"' not in stripped or stripped.count('"') % 2:
        return None
    try:
        values = next(
            csv.reader([stripped], delimiter=delimiter, skipinitialspace=delimiter == " ")
        )
    except (csv.Error, StopIteration):
        return None
    if not values:
        return None
    if len(values) > max_tokens:
        # Bounded allocation, mirroring _split_data_line's maxsplit: overflow
        # folds into the last token instead of growing the row unboundedly.
        values = [*values[: max_tokens - 1], delimiter.join(values[max_tokens - 1 :])]
    while values and values[-1] == "":
        values.pop()
    return values


def _split_plain_las30_row(
    line: str,
    delimiter: str,
    max_tokens: int,
    num_curves: int,
    warn: bool = True,
) -> list[str]:
    """Plain (non-CSV) split policy for ONE LAS 3.0 data row.

    SPACE keeps ``str.split``; every other delimiter routes through the
    shared DLM-aware splitter so thousands-grouped fragments recombine
    ("1,234.5" → "1234.5") with a loud warning instead of shifting every
    later column (M-05/M-30).  *warn* is forwarded to that splitter.
    """
    if delimiter == " ":
        return line.split(maxsplit=max_tokens)
    return _split_data_line(line.strip(), delimiter, expected=num_curves, warn=warn)


def _split_las30_row(
    line: str,
    delimiter: str,
    max_tokens: int,
    num_curves: int,
    has_string_curve: bool,
) -> list[str]:
    """Split ONE LAS 3.0 data row into its items (the loop's split policy).

    ONE place for the row-splitting rules, shared by the read loop and the
    ``{D}``-format data probe (F-129): drifting twins of this logic are the
    repo's dominant defect class, so the probe must not carry its own copy.

    F-011: a line carrying a spec-conforming double-quoted item is split by
    ``csv`` only when ALL of these hold — the section declares a string
    curve (a numeric-only section has no item that may legally contain the
    delimiter), the line contains a quote, and the plain split of that line
    OVER-COUNTS the declared columns, i.e. quoting is genuinely needed to
    keep the row intact (``100,"WELL, A",50`` is 4 plain tokens for 3
    curves).  Otherwise the plain split is kept: the writer emits raw
    ``delimiter.join()`` with no CSV quoting, so parsing every even-quote
    row through ``csv`` silently stripped quote characters the library
    itself wrote (M1: ``"WELL"`` → ``WELL``, ``""`` → empty).  Stated
    trade: an external item quoted WITHOUT an embedded delimiter keeps its
    quote characters (the pre-F-011 reading) instead of having them
    stripped — the safe direction for the writer's own roundtrip.
    """
    if has_string_curve and '"' in line:
        # Counting probe only — recombination diagnostics are emitted by the
        # real split below, never twice.
        _plain_values = _split_plain_las30_row(line, delimiter, max_tokens, num_curves, warn=False)
        if len(_plain_values) > num_curves:
            _quoted_values = _split_quoted_data_line(line.strip(), delimiter, max_tokens)
            if _quoted_values is not None:
                return _quoted_values
    return _split_plain_las30_row(line, delimiter, max_tokens, num_curves)


def _detect_d_format_date_columns(
    lines: list[str],
    section_curves: list[CurveDefinition],
    delimiter: str,
    max_tokens: int,
) -> dict[int, str]:
    """Return ``{column index: first offending value}`` for ``{D}`` date columns.

    F-129 (reader half).  ``D`` is ambiguous in LAS 3.0: the CWLS 3.0
    "Data Formats" chapter defines no ``D`` code (dates are letter patterns
    such as ``{DD/MM/YYYY}``), the standard's OWN appendix uses ``{D}`` for a
    DATE channel ("C_DT … {D}", R-03 F-7), and the Fortran
    double-precision reading (``1.234D+05``) is legitimate for a numeric
    channel.  The models/parser grammar therefore keeps ``D`` in the
    accepted set, and the token alone cannot decide the channel type.

    The DATA decides it: a Fortran-D column's values parse as numbers, a
    date column's do not.  This bounded probe (up to
    ``_D_FORMAT_PROBE_ROWS`` data rows) reports a column as a date column
    when it produced at least one non-empty value and NOT ONE value parsed
    as a number.  A column with no usable evidence stays numeric, i.e. the
    pre-fix behaviour.

    Returns an empty mapping when the section declares no ``{D}`` curve.
    """
    candidates = [
        i for i, c in enumerate(section_curves) if (c.data_format or "").upper() == "D"
    ]
    if not candidates:
        return {}
    # Lazy import (same reason as process_ascii_data): parser.py imports this
    # module, so the patterns are fetched at call time to avoid the cycle.
    from .parser import COMMENT_PATTERN, EMPTY_PATTERN

    pending: set[int] = set(candidates)
    numeric: set[int] = set()
    evidence: dict[int, str] = {}
    scanned = 0
    for line in lines:
        if scanned >= _D_FORMAT_PROBE_ROWS:
            break
        if COMMENT_PATTERN.match(line) or EMPTY_PATTERN.match(line):
            continue
        scanned += 1
        # Quotes are honoured whenever present: a quoted item is a single
        # column even for the probe (F-011 split policy, shared helper).
        values = _split_las30_row(
            line, delimiter, max_tokens, len(section_curves), '"' in line
        )
        for idx in list(pending):
            if idx >= len(values):
                continue
            token = values[idx].strip()
            if not token:
                continue
            try:
                _parse_float_with_d_notation(token)
            except (ValueError, OverflowError):
                evidence.setdefault(idx, token)
            else:
                numeric.add(idx)
                pending.discard(idx)
        pending -= numeric
        if not pending:
            break
    return {idx: sample for idx, sample in evidence.items() if idx not in numeric}


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
    from ._sanitize import desanitize_las_value as _desanitize_las_value
    from .parser import (
        COMMENT_PATTERN,
        EMPTY_PATTERN,
        MAX_DATA_SECTIONS,
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
        # E-21: LAS 3.0 has no wrapped-data reader — this is a DELIBERATE
        # rejection (R-5), not an accidental gap.  The LAS 1.2/2.0
        # accumulator (_read_wrapped, data_reader.py:1142) operates on
        # LASFile.logs with a different row protocol and cannot be shared
        # with the 3.0 per-section DataSection fill loop without a
        # cross-module refactor; the shared detection core therefore feeds
        # a LOUD, declaration-independent error instead of the silent
        # DEPT-shift that would follow from misparsing wrapped rows as
        # non-wrapped.  The CWLS 3.0 spec's WRAP rule is unverified, so
        # NO new rejection rule is encoded here beyond the existing
        # content-based detection.  Because detection is content-based
        # (the declared WRAP header may be wrong), the message must not
        # instruct the user to "set WRAP to NO" — the file may already
        # declare NO (the pre-fix message misdiagnosed such files).
        if (ctx.las_file.version.wrap or "").upper() == "YES":
            raise LASParseError(
                "LAS 3.0 WRAP=YES is not supported by pylasdev.  "
                "Convert the file to unwrapped format (one line per "
                "depth step) before parsing."
            )
        raise LASParseError(
            "LAS 3.0 wrapped/flowing data layout (multiple depth steps "
            "per line) is not supported by pylasdev, even when WRAP=NO "
            "is declared — the wrapped layout was detected from the "
            "data content.  Convert the file to unwrapped format (one "
            "line per depth step) before parsing."
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
            "ASCII data present but no curves defined for this section. Data has been discarded.",
            UserWarning,
            stacklevel=2,
        )
        return

    # F1: Local dedup on per-section curves.  Global curve writeback
    # (see F2-07 block below) is deferred until after the actual_count
    # > 0 check — see F2-07 writeback block below.
    # P-11 (W-6): deduplicating a duplicate BRACKET-notation array mnemonic
    # (e.g. "NMR[1]") produces a suffixed name "NMR[1]_2" that fails
    # CurveDefinition's _MNEMONIC_PATTERN — the construction raises a bare
    # ValueError that would escape parse().  Wrap it as LASParseError.
    try:
        deduped_order = _deduplicate_curves(ctx, section_curves, is_first_section=False)
    except ValueError as exc:
        raise LASParseError(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"curve deduplication failed: {exc}"
        ) from exc

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
                (curve.array_info.index, curve.mnemonic, curve.data_format or "", _pos)
            )
    for _base_name, _elements in _array_groups.items():
        # M-15: validate SINGLE-element array groups too.  The previous
        # ``len(_elements) < 2: continue`` short-circuit skipped the
        # index rule for lone elements — a metadata-only or
        # single-element file with ``NMR[0]`` parsed silently (sibling of
        # the models-side N-01 fix: the read-side check here + the
        # model-side construction check cover the full chain).  A single
        # element is rejected only for a NON-POSITIVE index: a section
        # may legitimately carry a per-section continuation element
        # (section 1 ``NMR[1]``, section 2 ``NMR[2]`` — M-64 locks that
        # roundtrip), so ``index >= 1`` is the valid range.
        _elements.sort(key=lambda e: e[0])
        _first_idx = _elements[0][0]
        if len(_elements) == 1:
            if _first_idx < 1:
                raise LASParseError(
                    f"Array '{_base_name}' starts at index {_first_idx}; "
                    f"LAS 3.0 requires 1-based array indices ([1]→[n])"
                )
            continue
        # F-034 (a): Arrays in LAS 3.0 must use 1-based indices.
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
        i: c.data_format in ("S",) or (c.data_format in ("A",) and c.array_info is None)
        for i, c in enumerate(section_curves)
    }

    # F-129 (reader half): a `{D}` channel declared as a DATE column.
    # CWLS LAS 3.0 defines no `D` data-format code (its "Data Formats"
    # chapter lists F/I/S/E plus the date/time letter patterns and the DMS
    # symbol form), yet the standard's own appendix writes `{D}` for a DATE
    # channel ("C_DT … {D}", R-03 F-7), and the models layer keeps `D` in
    # `_VALID_DATA_FORMATS` because the Fortran double-precision reading is
    # legitimate.  The token alone therefore cannot classify the channel; the
    # DATA can — a Fortran-D column parses as a number ("1.234D+05"), a date
    # column does not.  Without this the whole date column failed float()
    # and became the null sentinel (probed: `C_DT = [-999.25, -999.25]`).
    # The routed column carries `data_format="S"`, the representation the
    # models/parser grammar uses for every non-numeric date channel
    # (`_normalize_curve_data_format`), which also keeps
    # `_check_df_vs_placement` / `LASFile.__post_init__` consistent.
    _d_date_columns = _detect_d_format_date_columns(
        ctx.ascii_data_lines, section_curves, delimiter, _max_tokens
    )
    for _i, _sample in sorted(_d_date_columns.items()):
        string_curves[_i] = True
        _curve = section_curves[_i]
        _curve.data_format = "S"
        # Propagate to the GLOBAL curve definition (the F2-07 writeback below
        # copies mnemonic/description/array_info, not data_format; for a
        # dedup-renamed curve the section entry is a COPY, so the global
        # definition would otherwise keep `D` and the format-vs-placement
        # check would reject the string column on to_dict → from_dict).
        _gidx = ctx.section_curve_start_idx + _i
        if _gidx < len(ctx.las_file.curves):
            ctx.las_file.curves[_gidx].data_format = "S"
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': curve "
            f"'{_curve.mnemonic}' declares the undefined '{{D}}' format and its "
            f"values are not numbers (first: {_sample!r}) — reading the column "
            f"as STRING data.  CWLS LAS 3.0 defines no 'D' code (dates are "
            f"letter patterns such as {{DD/MM/YYYY}}) while the standard's own "
            f"appendix uses '{{D}}' for a date channel; a Fortran "
            f"double-precision column would parse as a number.",
            UserWarning,
            stacklevel=2,
        )

    # L-03: Detect {I} integer-format curves.  Stored as int64 (when the
    # null sentinel is integral) and parsed via int() so values above 2^53
    # are not silently rounded by float().
    integer_curves = {i: c.data_format == "I" for i, c in enumerate(section_curves)}

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
        # PF-19 (L30-01 completion): Propagate the section-level description
        # to the GLOBAL curve definition, mirroring the mnemonic/array_info
        # propagation.  For spec-form array channels, _build_spec_form_array_info
        # stripped the {A:N} marker from the section curve's description (the
        # writer re-emits {A:N} from array_info.time_offset); without this copy
        # the global curve keeps the raw marker, which to_dict() leaks and the
        # no-data_sections write path double-emits as ``{A:0}  {A:0}``.
        # Non-array channels are the same objects (slice aliasing), so this is
        # a self-assignment no-op for them.
        ctx.las_file.curves[global_idx].description = curve.description
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
            data_section.data[curve.mnemonic] = np.zeros(actual_count, dtype=np.int64)
        elif integer_curves.get(i, False):
            # EXT-04: fractional declared NULL — object dtype preserves
            # exact {I} integers above 2^53 while null cells keep the
            # fractional sentinel (int64 would truncate -999.25 → -999).
            data_section.data[curve.mnemonic] = np.zeros(actual_count, dtype=object)
        else:
            data_section.data[curve.mnemonic] = np.zeros(actual_count, dtype=np.float64)
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
    # L30-02: Row-count accumulator for extra columns caused by an embedded
    # delimiter inside a {S} string value (DLM=COMMA/TAB).  The string is
    # truncated at the delimiter, every following column shifts, and the
    # genuine trailing values are lost as "extra columns" — the split cannot
    # distinguish an embedded delimiter from a column separator.  Mirrors the
    # LAS 1.2/2.0 twin (I2-02) fix so both paths emit the same loud warning.
    embedded_delim_count = 0  # I2-02 twin: delimiter-in-string truncation
    # F-012/F-013: Row-count accumulator for a BLANK first (index) item.
    # CWLS 3.0: when the first channel of a column-data line is the indexed
    # channel, that first item cannot be blank or null.  A blank item was
    # read as the null value with zero diagnostics.
    blank_index_count = 0
    _fc = [0]  # F-002: Mutable counter for float conversion diagnostics
    # F-39, F-40, I2-F-03: Desanitize reversal warning dedup flags.
    _desan_warned_ws = False  # whitespace→underscore (F-39)
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

    # F-011: quoting is only meaningful when the section declares a string
    # curve — a numeric-only section has no item that may legally contain the
    # delimiter — so hoist the test out of the per-line loop.
    _has_string_curve = any(string_curves.get(i, False) for i in range(num_curves))

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
        # F2-015/F-011/M-05: the split policy itself (quoting, thousands
        # recombination, delimiter branches) lives in _split_las30_row, shared
        # with the {D}-format data probe (F-129) so the two cannot drift.
        values = _split_las30_row(
            line, delimiter, _max_tokens, num_curves, _has_string_curve
        )

        # Warn about extra columns being silently discarded
        if len(values) > num_curves:
            extra_count += 1
            # L30-02: DLM=COMMA/TAB + embedded delimiter in a {S} string
            # value truncates the string AND shifts every following column
            # — the next column's genuine value is destroyed and trailing
            # values are lost as "extra columns" (verified: `100.0,WELL A,
            # B,10.0` with DEPT/WELLS{S}/GR → WELLS truncated to 'WELL A',
            # GR receives ' B' → conversion failure → -999.25, genuine 10.0
            # discarded).  F-011 handles the SPEC-CONFORMING case (the item
            # is quoted); this warning is the remaining case — an unquoted
            # value holding the delimiter — where the split genuinely cannot
            # tell an embedded delimiter from a column separator, so the
            # correct response is a LOUD warning naming the mechanism rather
            # than a silent shift.  Mirrors the LAS 1.2/2.0 I2-02 fix.
            if delimiter != " " and any(string_curves.get(i, False) for i in range(num_curves)):
                if embedded_delim_count == 0:
                    warnings.warn(
                        f"Data section '{ctx.current_section_name or 'ASCII'}': "
                        f"row has {len(values)} values but only {num_curves} "
                        f"declared curves with DLM={delimiter!r}. A string "
                        f"curve value may contain the delimiter character, "
                        f"truncating the string and shifting the following "
                        f"columns (genuine values may be lost). Consider "
                        f"quoting string values or using a delimiter not "
                        f"present in the data.",
                        UserWarning,
                        stacklevel=2,
                    )
                embedded_delim_count += 1

        # F-11: Warn when non-wrapped data lines have fewer values than
        # declared curves.  Short rows in wrapped mode are expected
        # (values span multiple lines), so this warning only fires in
        # non-wrapped mode.
        # M-21: key the gate on the CONTENT-DETECTED wrap state
        # (actual_wrap), matching the LAS 1.2/2.0 twin (_read_normal only
        # runs when not actually wrapped and warns on short rows
        # unconditionally, data_reader.py:946-963).  The previous gate
        # keyed on the DECLARED WRAP header — a declared-WRAP=YES file
        # whose data is actually non-wrapped silently null-filled short
        # rows with ZERO diagnostics.
        if len(values) < num_curves and not actual_wrap:
            short_count += 1

        # Pad with null values if needed.
        # String curves use "" (empty string) to avoid width-ambiguity
        # from str(null_value) padding (matching data_reader.py:836-840).
        # M-36: capture the pre-padding token count so the fill loop below
        # can recognize cells this pass already filled (direct assignment,
        # no re-parse — see the {I} branch).
        _padded_from = len(values)
        while len(values) < num_curves:
            _pad_idx = len(values)
            if string_curves.get(_pad_idx, False):
                values.append("")
            else:
                values.append(str(null_value))
                if track_fill:
                    _check_fill_cell_cap()
                    fill_cells.append((idx, _pad_idx))

        # F-012/F-013: CWLS 3.0 index-channel rule — when the first channel
        # of a column-data line is the indexed channel, that first data item
        # cannot be blank or null.  A blank first token was converted to the
        # null value by _to_finite_float("") without incrementing the failure
        # counter and without any diagnostic, so a corrupt/truncated row
        # silently produced a null index.  Counted per section and warned
        # once below.
        # Scope: only a NUMERIC first channel.  A {S} first curve may hold a
        # legitimately empty string (existing pinned behaviour), and the
        # library's own writer emits the NULL sentinel — not a blank — for a
        # padded index cell, so the sentinel form is not counted here.
        if not string_curves.get(0, False) and not values[0].strip():
            blank_index_count += 1

        for i in range(num_curves):
            # IT3-F1: Do NOT strip per-token whitespace.  The LAS 1.2/2.0
            # path (_split_data_line) keeps token-edge whitespace, so
            # stripping here silently destroyed leading/trailing spaces in
            # {S} string data on LAS 3.0 reads.  Numeric parsing is
            # whitespace-tolerant (float()) and _desanitize_las_value
            # handles whitespace-prefixed "_#" escapes (case 2), so the
            # token is passed through untouched.
            val_str = values[i]
            # F-007: Reverse the writer's _sanitize_las_value #-prefix escape.
            # E-01 (HIGH): the M-85 '_~' escape is emitted by the writer
            # ONLY for FIRST-column string values starting '~'+non-letter
            # (_writer_base.py:710-737); a genuine '_~'-prefixed value is
            # written VERBATIM (the escape predicate `\s*~` never matches
            # '_~...').  restore_tilde is therefore position-scoped to the
            # first column (i == 0) — the pre-fix code applied it to EVERY
            # token of every row, corrupting genuine '_~' values in ANY
            # column (e.g. a second-column {S} value '_~sandstone' was
            # silently read as '~sandstone').  _sanitize.desanitize_las_value
            # additionally applies the writer-escape predicate (only
            # '_~'+non-letter tokens restore), so a first-column genuine
            # '_~DEPT' — which the writer would never emit — survives too.
            # The escape-collision residual (genuine '~3D' escaped AND
            # genuine '_~3D' both write byte-identically as '_~3D') is a
            # documented irreducible trade (see _sanitize.py).
            val_str = _desanitize_las_value(val_str, restore_tilde=(i == 0))
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
                arr = numeric_arrays[i]
                if arr is None:
                    raise LASParseError(
                        f"Internal error: numeric array '{i}' was not pre-allocated"
                    )
                if i >= _padded_from:
                    # M-36: This cell was already filled by the padding
                    # pass above (str(null_value) appended + tracked in
                    # fill_cells when track_fill).  Assign the sentinel
                    # DIRECTLY, mirroring data_reader.py:1244-1247 — do
                    # NOT re-parse the sentinel text through
                    # _to_integer_value: a fractional NULL (e.g. -999.25)
                    # fails int() conversion, increments the failure
                    # counter (false "could not be converted" warning at
                    # the summary below) and re-tracks the SAME cell in
                    # fill_cells (2x fill-cell accounting, prematurely
                    # consuming _MAX_FILL_CELLS).
                    arr[idx] = int(null_value) if _null_is_integral else null_value
                else:
                    _fc_before = _fc[0]
                    int_val = _to_integer_value(
                        val_str,
                        null_value,
                        _failure_counter=_fc,
                        _null_as_float=not _null_is_integral,
                    )
                    if track_fill and (_fc[0] > _fc_before or not val_str):
                        _check_fill_cell_cap()
                        fill_cells.append((idx, i))
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
    # L30-02: Summary for embedded-delimiter string truncation.  The first
    # occurrence already warned with full context above; this counts the
    # total so automated tools can enumerate affected rows (mirrors the
    # LAS 1.2/2.0 I2-02 twin).
    if embedded_delim_count > 1:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"{embedded_delim_count} row(s) had more values than the "
            f"{num_curves} declared curves with DLM={delimiter!r}. String "
            f"curve values may contain the delimiter character; those "
            f"strings were truncated and following columns' genuine values "
            f"may have been lost.",
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

    # F-012/F-013: Report the CWLS 3.0 index-channel violation once per
    # section (the per-row detail is the count itself; the fill already
    # stored the null value, so there is nothing to reconcile here).
    if blank_index_count > 0:
        warnings.warn(
            f"Data section '{ctx.current_section_name or 'ASCII'}': "
            f"{blank_index_count} row(s) had a blank first (index) value.  "
            f"LAS 3.0 requires the first item of an indexed column-data line "
            f"to be a value; the blank item was read as the null value "
            f"({null_value}).  The file may be truncated or corrupt.",
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
    # F-014: ONE ownership decision for BOTH top-level views.  The two
    # independent emptiness gates (``not logs`` here, ``not string_data``
    # below) let a numeric-only first LOG_DATA section populate ``logs``
    # while a LATER section populated ``string_data``: the two views then
    # described different sections with different row counts, so the
    # library's own write→read output failed validate() and
    # ``LASFile.from_dict(las.to_dict())`` raised LASDataError.
    # The latch is the owner marker already used by
    # _reconcile_null_sentinels, so "the first LOG_DATA section owns both
    # views" needs no new state: it fires on that section whether or not it
    # carries string curves, and both copies below read the same section.
    # This subsumes the former ``not logs`` / ``not string_data`` clauses —
    # nothing else populates these views on the LAS 3.0 read path.
    _views_owned = any(
        getattr(_section, _NULL_LOGS_OWNER_ATTR, False) for _section in ctx.las_file.data_sections
    )
    _owns_views = is_log_data and not _views_owned
    if _owns_views:
        for curve in section_curves:
            if curve.mnemonic in data_section.data:
                ctx.las_file.logs[curve.mnemonic] = data_section.data[curve.mnemonic].copy()
        # M-04: Record which section's arrays were copied into the
        # top-level logs view so _reconcile_null_sentinels only writes
        # fill cells back into the SAME section's logs arrays (the latch
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
    # F-014: keyed on the SAME ``_owns_views`` decision as the numeric copy.
    if _owns_views:
        for i, curve in enumerate(section_curves):
            if i in string_data_lists:
                # F2-014: Defensive copy — prevents shared-reference
                # mutation between LASFile.string_data and
                # DataSection.string_data.
                ctx.las_file.string_data[curve.mnemonic] = data_section.string_data[
                    curve.mnemonic
                ].copy()

    # N-I-31: Attach the tracked fill cells to the section so a later
    # process_ascii_data call (once ~Well is known) can re-fill them
    # with the declared NULL.  Only needed when the well was unknown.
    if track_fill and fill_cells:
        setattr(data_section, _NULL_FILL_CELLS_ATTR, fill_cells)
        setattr(data_section, _NULL_FILL_SENTINEL_ATTR, null_value)

    # Store data section (LAS 3.0)
    ctx.las_file.data_sections.append(data_section)
