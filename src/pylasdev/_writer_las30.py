"""LAS 3.0 writer — typed sections, per-section parameters, zone notation."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np

from ._writer_base import (
    _SECTION_TYPE_TO_DEFINITION_PREFIX,
    _emission_plan,
    _emitted_mnemonic,
    _format_curve_line,
    _format_data_rows,
    _format_parameter_line,
    _lookup_data_array,
    _mnem_key,
    _sanitize_las_value,
    _section_type_to_prefix,
    _WriterBase,
)
from .data_reader import _get_null_value
from .exceptions import LASWriteError
from .models import CurveDefinition, DataSection, LASFile, ParameterEntry


def _curve_identity(curve: CurveDefinition, emitted_name: str | None = None) -> tuple[Any, ...]:
    """Composite identity used for cross-section dedup/scoping comparisons.

    Includes the EMITTED mnemonic, unit, data format, and array index so
    that:
    - a second section with the same mnemonic but a different
      unit/format is recognized as a DIFFERENT scope (M-26),
    - array curves NMR[1]/NMR[2] do not collide (M-64),
    - column ORDER is significant (M-66).
    description/api_code are deliberately EXCLUDED — differences there
    are preserved via the W-01 richer-definition merge, not by forcing a
    per-section Definition (M-83).

    W-10: ``emitted_name`` overrides the M-59 reconstruction for a
    collision-free emission (a reader-renamed duplicate like IK_2 with
    original_mnemonic='IK' is emitted as IK_2, and the scoping identity
    must use that same name or the section would mismatch the main ~C
    block).
    """
    return (
        emitted_name if emitted_name is not None else _emitted_mnemonic(curve),
        curve.unit or "",
        curve.data_format or "",
        curve.array_info.index if curve.array_info is not None else None,
    )


def _definition_signature(
    curve: CurveDefinition, emitted_name: str | None = None
) -> tuple[Any, ...]:
    """Full per-curve signature used to dedup per-section Definition blocks.

    Unlike ``_curve_identity`` this INCLUDES description/api_code (two
    sections with identical scoping but different metadata must get
    separate Definition blocks) and the array index + time_offset (M-64:
    the previous signature omitted ``array_info.index`` so NMR[1]/NMR[2]
    collapsed to one Definition).

    W-10: ``emitted_name`` overrides the M-59 reconstruction (see
    ``_curve_identity``).
    """
    return (
        emitted_name if emitted_name is not None else _emitted_mnemonic(curve),
        curve.unit or "",
        curve.description or "",
        curve.data_format or "",
        curve.api_code or "",
        curve.array_info.index if curve.array_info is not None else None,
        curve.array_info.time_offset if curve.array_info is not None else None,
    )


def _arrays_equal(a: Any, b: Any) -> bool:
    """Value equality for M-80 covered-key comparison.

    NaN is treated as equal (a parser roundtrip writes NaN positions
    through the null sentinel and reads them back identically, so NaN
    does not indicate a value conflict).
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    return bool(a == b)


class _Las30Writer(_WriterBase):
    """LAS 3.0 format writer.

    Overrides all section writers for LAS 3.0 typed-section behavior:
    - Version section: VERSION 3.0 desc, WRAP forced NO
    - Curve section: data_sections scan instead of simple loop
    - Parameter section: per-section grouping with data_format/zone
    - Other section: deprecated, skip with warning
    - ASCII sections: multi-section typed headers with pipe/definition logic
    """

    def __init__(self, las_file: LASFile, precision: str) -> None:
        super().__init__(las_file, precision)
        # N-I-20: Set by _write_curve_section — the main ~C block's curve
        # definitions, IN ORDER.  Used by _write_ascii_las30 to decide
        # whether a LOG_DATA section pipes ``| CURVE`` (matches the main
        # block, same definitions AND same column order) or gets a
        # per-section Definition (distinct curve set / identity / order —
        # M-26/M-64/M-66/M-68).
        self._main_curves: list[CurveDefinition] = []
        # W-10: collision-free emitted names for the main ~C block (set
        # by _write_curve_section), keyed by id(curve).  A reader-renamed
        # duplicate (IK_2 with original_mnemonic='IK') is emitted as its
        # own mnemonic; the scoping identity must use the same name or a
        # section would mismatch the main block.
        self._main_emitted_names: dict[int, str] = {}

    def _all_string_mnemonics(self) -> frozenset[str]:
        """Union of EVERY string_data mnemonic across all scopes.

        M-77: the parser classifies a column as string ONLY from the {S}
        marker in its ~C/Definition line.  A string curve with an empty
        (or non-'S') data_format is emitted markerless and its values are
        re-read as numeric nulls.  The main ~C block emits the curve
        definitions that data sections inherit (via ``| CURVE``), so a
        curve whose DATA lives in string_data in ANY scope must get the
        {S} marker here.  Per-section Definitions use the section's own
        string_data keys instead.
        """
        # N2b-1: the set is built UPPER-CASED (matching
        # _format_curve_line's _mnem_key membership) so a case-variant
        # string_data key ('dept_str' vs curve 'DEPT_STR') still forces
        # the {S} marker.
        mnems = (
            {_mnem_key(k) for k in self._las_file.string_data.keys()}
            if self._las_file.string_data
            else set()
        )
        for ds in self._las_file.data_sections:
            if ds.string_data:
                mnems.update(_mnem_key(k) for k in ds.string_data.keys())
        return frozenset(mnems)

    # ── Version section ──────────────────────────────────────────────

    def _write_version_section(self) -> list[str]:
        """Write ~V Version section — LAS 3.0 format."""
        lines: list[str] = []
        lines.append("~VERSION INFORMATION")
        vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0"
        vers = self._las_file.version.vers or "2.0"
        # F-35: normalize a VERS value the reader cannot round-trip to the
        # canonical "3.0".  The reader's VERS normalization (parser.py)
        # accepts "3.0", "3.x" drafts, and \d+\.\d+ — but a bare "3"
        # (accepted as LAS 3.0 by is_las30 = startswith("3")) matches none
        # and is downgraded to "2.0" on re-read, silently dropping every
        # typed data_section.  Draft forms like "3.1beta"/"3.0-draft"
        # (startswith "3.") are preserved so the documented I2F-02
        # draft-version roundtrip keeps working.
        if not (vers.startswith("3.") or re.match(r"^\d+\.\d+$", vers)):
            vers = "3.0"
        lines.append(f" VERS.   {_sanitize_las_value(vers)}  : {vers_desc}")

        actual_wrap = self._las_file.version.wrap.upper() if self._las_file.version.wrap else "NO"
        if actual_wrap == "YES":
            import warnings

            warnings.warn(
                "WRAP=YES overridden to WRAP=NO because the writer "
                "always produces ONE LINE PER DEPTH STEP (non-wrapped) output. "
                "The data WILL be non-wrapped regardless of the original declaration.",
                stacklevel=3,
            )
            actual_wrap = "NO"
        # LAS 3.0 forces WRAP=NO regardless of user input.
        if actual_wrap != "NO":
            import warnings

            warnings.warn(
                f"LAS 3.0 WRAP={actual_wrap} overridden to WRAP=NO. "
                "LAS 3.0 requires one data row per line. "
                "The data will be written in non-wrapped format.",
                stacklevel=3,
            )
            actual_wrap = "NO"
        wrap_desc = (
            "ONE LINE PER DEPTH STEP" if actual_wrap == "NO" else "MULTIPLE LINES PER DEPTH STEP"
        )
        lines.append(f" WRAP.   {_sanitize_las_value(actual_wrap)}  : {wrap_desc}")

        if self._las_file.version.dlm and self._las_file.version.dlm.upper() != "SPACE":
            dlm_desc = "DELIMITING CHARACTER BETWEEN DATA COLUMNS"
            lines.append(
                f" DLM .                        {_sanitize_las_value(self._las_file.version.dlm)} : {dlm_desc}"
            )
        lines.append("")
        return lines

    # ── Curve section ────────────────────────────────────────────────

    def _write_curve_section(self) -> list[str]:
        """Write ~C Curve section — LAS 3.0 data_sections scan."""
        lines: list[str] = []
        lines.append("~CURVE INFORMATION")

        if self._las_file.data_sections:
            curves_to_emit: list[CurveDefinition] = []
            emitted_mnems: set[str] = set()
            # W-10: per-curve mnemonic_override for M-59 collisions (a
            # reader-renamed duplicate like IK_2 with original_mnemonic
            # ='IK' falls back to its own mnemonic so the distinct column
            # survives in ~C).
            _emission_overrides: dict[int, str] = {}
            # W-01: Dedup during the FIRST extension loop.  Previously the
            # primary accumulation path (extend of each section's curves)
            # had NO dedup — `emitted_mnems` was computed AFTER the extend
            # and only guarded the fallback loop.  Two LOG_DATA sections
            # sharing a mnemonic (e.g. DEPT in section 1 and section 2)
            # therefore emitted DEPT twice in ~C, and on re-read the
            # parser inflated the curve count (2→4) and null-filled the
            # phantom columns.  Dedup by EMITTED mnemonic here so the
            # shared definition is emitted once — and so array curves
            # NMR[1]/NMR[2] (same base mnemonic, different array_info
            # index) do NOT collide (M-64).
            for ds in self._las_file.data_sections:
                if (ds.section_type or "LOG_DATA").upper() == "LOG_DATA":
                    if ds.section_curves:
                        for curve in ds.section_curves:
                            _emitted = _emitted_mnemonic(curve)
                            _override: str | None = None
                            # N2b-2: emitted_mnems stores UPPER-CASED keys —
                            # a case-variant duplicate pair ('DEPT' + 'dept')
                            # must be treated as the duplicate it is (re-read
                            # identity is case-insensitive and would rename
                            # the second to DEPT_2).
                            if _mnem_key(_emitted) in emitted_mnems:
                                # W-10: M-59 collision — a reader-renamed
                                # duplicate (IK_2 with original_mnemonic
                                # ='IK') or a mnem_base vendor-rename
                                # collision falls back to its OWN mnemonic
                                # when free, preserving the distinct column.
                                if _mnem_key(curve.mnemonic) not in emitted_mnems:
                                    _override = curve.mnemonic
                                else:
                                    # W-01: dedup-by-mnemonic silently drops a
                                    # SECOND section's DIFFERING definition
                                    # (e.g. DEPT.M in A1 vs DEPT.FT in A2) — the
                                    # ~C shows one unit for both sections.  Warn
                                    # on unit/format/index mismatch so the drop is
                                    # visible instead of silent (M-26/M-64), and
                                    # on desc/api_code mismatch, preserving the
                                    # richer definition (M-83).
                                    for _i, emitted in enumerate(curves_to_emit):
                                        if _mnem_key(_emitted_mnemonic(emitted)) == _mnem_key(
                                            _emitted
                                        ):
                                            _emitted_unit = emitted.unit or ""
                                            _curve_unit = curve.unit or ""
                                            _emitted_fmt = emitted.data_format or ""
                                            _curve_fmt = curve.data_format or ""
                                            _emitted_idx = (
                                                emitted.array_info.index
                                                if emitted.array_info
                                                else None
                                            )
                                            _curve_idx = (
                                                curve.array_info.index if curve.array_info else None
                                            )
                                            if (
                                                _emitted_unit != _curve_unit
                                                or _emitted_fmt != _curve_fmt
                                                or _emitted_idx != _curve_idx
                                            ):
                                                import warnings

                                                warnings.warn(
                                                    f"Duplicate curve mnemonic "
                                                    f"'{_emitted}' in LAS 3.0 "
                                                    f"data sections has a differing "
                                                    f"definition (unit "
                                                    f"{_emitted_unit!r} vs "
                                                    f"{_curve_unit!r}).  Keeping the "
                                                    f"first definition in ~C; the "
                                                    f"second section's curve "
                                                    f"metadata is not re-emitted.",
                                                    UserWarning,
                                                    stacklevel=3,
                                                )
                                            # M-83: desc/api_code differences are
                                            # metadata-only; keep the RICHER
                                            # definition (non-empty description /
                                            # api_code wins over empty) in ~C and
                                            # warn so the drop is visible.
                                            _desc_changed = (emitted.description or "") != (
                                                curve.description or ""
                                            )
                                            _api_changed = (emitted.api_code or "") != (
                                                curve.api_code or ""
                                            )
                                            if _desc_changed or _api_changed:
                                                import warnings

                                                warnings.warn(
                                                    f"Duplicate curve mnemonic "
                                                    f"'{_emitted}' in LAS 3.0 "
                                                    f"data sections has a differing "
                                                    f"description/api_code.  "
                                                    f"Keeping the richer definition "
                                                    f"in ~C; the second section's "
                                                    f"metadata is not re-emitted.",
                                                    UserWarning,
                                                    stacklevel=3,
                                                )
                                                if not emitted.description and curve.description:
                                                    curves_to_emit[_i] = replace(
                                                        emitted,
                                                        description=curve.description,
                                                    )
                                                if not emitted.api_code and curve.api_code:
                                                    curves_to_emit[_i] = replace(
                                                        curves_to_emit[_i],
                                                        api_code=curve.api_code,
                                                    )
                                            break
                                    continue
                            emitted_mnems.add(_mnem_key(_override or _emitted))
                            curves_to_emit.append(curve)
                            if _override is not None:
                                _emission_overrides[id(curve)] = _override
            curves_by_mnem: dict[str, CurveDefinition] = {}
            for _c in self._las_file.curves:
                curves_by_mnem.setdefault(_c.mnemonic, _c)
                curves_by_mnem.setdefault(_emitted_mnemonic(_c), _c)
                # I2-22 (PF-21): case-insensitive resolution — a lowercase
                # 'dept' in a section's curves_order must resolve to the
                # DEPT definition exactly like the ASCII/emission paths
                # (_effective_section_curves / _section_emission_pairs).
                # Without the upper-cased keys the fallback loop below
                # fails to resolve the entry, emits a false N-I-15 warning,
                # and the empty emission falls back to the full top-level
                # list WITHOUT updating emitted_mnems — so the M-79 loop
                # re-emits the same curves (duplicate ~C lines) and fires
                # a false "definition but no data" warning.
                curves_by_mnem.setdefault(_c.mnemonic.upper(), _c)
                curves_by_mnem.setdefault(_emitted_mnemonic(_c).upper(), _c)
            # N-I-15: the fallback loop was gated on LOG_DATA only, so a
            # non-LOG_DATA section (e.g. CORE_DATA) with curves_order+data
            # but empty section_curves never entered ~C at all — the
            # written file had more data columns than ~C definitions and
            # re-read silently discarded the extra columns' data.  Process
            # ANY section that lacks section_curves (its curves must come
            # from the top-level ~C block; sections WITH section_curves get
            # their definitions in ~X_Definition instead).
            for ds in self._las_file.data_sections:
                if not ds.section_curves and ds.curves_order:
                    for mnem in ds.curves_order:
                        # I2-22 (PF-21): case-insensitive lookup — try the
                        # entry as-is then upper-cased, mirroring
                        # _effective_section_curves' resolution so a
                        # lowercase 'dept' resolves to the DEPT definition.
                        curve_def = curves_by_mnem.get(mnem) or curves_by_mnem.get(mnem.upper())
                        _emitted = _emitted_mnemonic(curve_def) if curve_def is not None else mnem
                        if _mnem_key(_emitted) not in emitted_mnems:
                            if curve_def is not None:
                                emitted_mnems.add(_mnem_key(_emitted))
                                curves_to_emit.append(curve_def)
                            else:
                                # N-I-15: the section curve is absent from
                                # top-level curves and cannot be emitted —
                                # warn so the data loss on re-read is
                                # visible at write time instead of silent.
                                # W-11: the actual outcome is WORSE than
                                # "discarded" — the data rows for the
                                # unresolvable column are emitted anyway
                                # (from the full curves_order) and the
                                # parser RELABELS them onto the next
                                # declared curve, silently corrupting
                                # another curve's genuine values.  The
                                # ASCII path (W-11 fix) now refuses to
                                # write data-bearing unresolvable columns.
                                import warnings

                                warnings.warn(
                                    f"Curve '{mnem}' appears in a LAS 3.0 "
                                    f"data section's curves_order but has no "
                                    f"definition in the top-level curves or "
                                    f"any section's section_curves.  The "
                                    f"written file would have more data "
                                    f"columns than ~C curve definitions, "
                                    f"and on re-read this column's values "
                                    f"would be silently relabeled onto "
                                    f"another curve.  The ASCII data pass "
                                    f"will refuse to write if this curve "
                                    f"carries data.",
                                    UserWarning,
                                    stacklevel=3,
                                )
                        else:
                            # W-10: the emitted name is already taken.  If
                            # the curve's OWN mnemonic is free, fall back to
                            # it so the distinct column survives in ~C
                            # (reader-renamed IK_2 with original_mnemonic
                            # ='IK'); otherwise the definition is
                            # metadata-only — skip.
                            # I2-22 (PF-21): curve_def was already resolved
                            # case-insensitively above; the mnemonic-exact
                            # fallback compares the definition's OWN
                            # mnemonic, which is exact by construction.
                            if (
                                curve_def is not None
                                and _mnem_key(curve_def.mnemonic) not in emitted_mnems
                            ):
                                emitted_mnems.add(_mnem_key(curve_def.mnemonic))
                                curves_to_emit.append(curve_def)
                                _emission_overrides[id(curve_def)] = curve_def.mnemonic

            # W-01: When ALL data sections are non-LOG_DATA and carry their
            # own section_curves (e.g. CORE-only files), the curve
            # definitions belong in the typed ~X_Definition sections — the
            # top-level fallback would duplicate them in ~C AND the
            # Definition block, inflating the re-read curve count (2→4).
            curves_in_definitions = bool(self._las_file.data_sections) and all(
                (ds.section_type or "LOG_DATA").upper() != "LOG_DATA" and bool(ds.section_curves)
                for ds in self._las_file.data_sections
            )
            if not curves_to_emit and not curves_in_definitions:
                curves_to_emit = list(self._las_file.curves)
            # F-28: when curves_in_definitions (all sections non-LOG_DATA
            # with section_curves, e.g. CORE-only files), the empty
            # curves_to_emit is EXPECTED — the section curves live in the
            # typed ~X_Definition blocks (W-01), so do not warn-and-return;
            # fall through to the M-79 loop below which still preserves any
            # metadata-only top-level curves (e.g. a data-free TEMP).
            if not curves_to_emit and not curves_in_definitions:
                import warnings

                warnings.warn(
                    "No curves to emit for ~C section — skipping",
                    UserWarning,
                    stacklevel=3,
                )
                self._main_curves = []
                lines.append("")
                return lines

            # M-79: Preserve top-level curve DEFINITIONS that no data
            # section references (metadata-only curves — e.g. a TEMP
            # definition with unit/description but no data anywhere).
            # Previously these were silently dropped from ~C whenever any
            # data_section contributed curves, permanently losing their
            # metadata on re-read.  Add them to the main block with a
            # warning (they carry no data, but the definition survives).
            # F-28: the loop ALSO runs when curves_in_definitions (all
            # sections non-LOG_DATA with section_curves, e.g. CORE-only
            # files).  The section-covered curves are skipped below via
            # _section_mnems, so only genuinely metadata-only top-level
            # curves are added — the section definitions stay in the typed
            # ~X_Definition blocks (W-01), avoiding the 2→4 re-read
            # inflation the full-list fallback would cause.
            if self._las_file.curves:
                # I2-22 (PF-21): _section_mnems stores UPPER-CASED keys so a
                # lowercase curves_order entry ('dept') still marks the
                # top-level DEPT definition as section-referenced.  Without
                # this the M-79 loop treats DEPT as data-free and re-emits
                # it with a false "definition but no data" warning — and
                # when the fallback loop above fell back to the full
                # top-level list, the same curve got emitted TWICE.
                _section_mnems: set[str] = set()
                for ds in self._las_file.data_sections:
                    if ds.section_curves:
                        _section_mnems.update(
                            _emitted_mnemonic(c).upper() for c in ds.section_curves
                        )
                    _section_mnems.update(m.upper() for m in ds.curves_order)
                _top_data_mnems = set(self._las_file.logs.keys()) | set(
                    self._las_file.string_data.keys()
                )
                for curve in self._las_file.curves:
                    _emitted = _emitted_mnemonic(curve)
                    if _mnem_key(_emitted) in emitted_mnems:
                        continue
                    # I2-22 (PF-21): case-insensitive comparison — _section_mnems
                    # holds upper-cased keys, so compare the emitted/mnemonic
                    # uppercased to match a lowercase curves_order reference.
                    if (
                        _emitted.upper() in _section_mnems
                        or curve.mnemonic.upper() in _section_mnems
                    ):
                        continue
                    if _emitted in _top_data_mnems or curve.mnemonic in _top_data_mnems:
                        # Has data at the top level (e.g. orphaned logs) —
                        # handled by the covered/orphan warnings, not here.
                        continue
                    # Definition-only: no data in any section and no
                    # top-level data either.  Emit the definition so its
                    # metadata survives; warn that it has no data.
                    emitted_mnems.add(_mnem_key(_emitted))
                    curves_to_emit.append(curve)
                    import warnings

                    warnings.warn(
                        f"Top-level curve '{curve.mnemonic}' has a "
                        f"definition but no data in any data section "
                        f"or in top-level logs/string_data.  Its "
                        f"metadata is written to ~C so it survives "
                        f"re-read; no data rows are emitted for it.",
                        UserWarning,
                        stacklevel=3,
                    )

            # N-I-20: Record the main ~C block's curves (ordered) so the
            # ASCII writer can decide per-section pipe targets.  A LOG_DATA
            # section whose curves exactly match the main block (same
            # identities in the same ORDER) pipes ``| CURVE``; a section
            # with a DIFFERENT curve set, identity, or column order gets a
            # per-section Definition section + pipe so its own scope
            # survives re-read (M-26/M-64/M-66/M-68).
            self._main_curves = list(curves_to_emit)
            self._main_emitted_names = {
                id(c): (_emission_overrides.get(id(c)) or _emitted_mnemonic(c))
                for c in self._main_curves
            }
            _all_str = self._all_string_mnemonics()
            for curve in curves_to_emit:
                lines.append(
                    _format_curve_line(
                        curve,
                        self._spec.is_las30,
                        _all_str,
                        mnemonic_override=_emission_overrides.get(id(curve)),
                    )
                )
        else:
            # I2-20: the no-data_sections path must dedup by EMITTED
            # mnemonic exactly like the data_sections path above — a
            # post-construction model (or a vendor-rename collision) can
            # carry two CurveDefinitions that emit the same name (LLD +
            # BFV with original_mnemonic='LLD').  Without the dedup the
            # ~C block emitted duplicate lines and the re-read renamed
            # the second (LLD → LLD_2), silently altering the model
            # identity.  W-10: a curve whose M-59 reconstruction would
            # collide falls back to its OWN mnemonic (preserving distinct
            # columns like a reader-renamed IK_2); only a metadata-only
            # duplicate (own mnemonic also taken) is dropped — warn so
            # the drop is visible.
            _main_pairs, _main_dropped = _emission_plan(self._las_file.curves)
            for _curve in _main_dropped:
                import warnings

                warnings.warn(
                    f"Duplicate curve mnemonic "
                    f"'{_emitted_mnemonic(_curve)}' in LAS 3.0 "
                    f"~C.  Keeping the first definition; the "
                    f"second curve's metadata is not re-emitted "
                    f"(a re-read would rename it and silently "
                    f"alter the model identity).",
                    UserWarning,
                    stacklevel=3,
                )
            _main_curves = [c for c, _ in _main_pairs]
            self._main_curves = list(_main_curves)
            _main_overrides = {id(c): o for c, o in _main_pairs}
            self._main_emitted_names = {
                id(c): (_main_overrides.get(id(c)) or _emitted_mnemonic(c))
                for c in self._main_curves
            }
            _top_str = (
                frozenset(_mnem_key(k) for k in self._las_file.string_data.keys())
                if self._las_file.string_data
                else frozenset()
            )
            for curve in _main_curves:
                lines.append(
                    _format_curve_line(
                        curve,
                        self._spec.is_las30,
                        _top_str,
                        mnemonic_override=_main_overrides.get(id(curve)),
                    )
                )

        lines.append("")
        return lines

    # ── Parameter section ────────────────────────────────────────────

    def _write_parameter_section(self) -> list[str]:
        """Write ~P Parameter section — LAS 3.0 per-section grouping."""
        if not self._las_file.parameters:
            return []

        lines: list[str] = []

        # Group by section_type for per-section parameter roundtrip.
        sections: dict[str | None, list[ParameterEntry]] = {}
        for param in self._las_file.parameters:
            st_key = (
                param.section_type.upper().replace("|", "_")
                if param.section_type and param.section_type.strip()
                else None
            )
            sections.setdefault(st_key, []).append(param)

        # Standard ~PARAMETER INFORMATION first (section_type=None).
        std_params = sections.pop(None, [])
        if std_params:
            lines.append("~PARAMETER INFORMATION")
            for param in std_params:
                lines.append(_format_parameter_line(param, self._spec.is_las30))
            lines.append("")

        # Per-section typed parameter sections.
        for section_type, params in sections.items():
            if not section_type:
                continue
            lines.append(f"~{_sanitize_las_value(section_type)}_Parameter")
            for param in params:
                lines.append(_format_parameter_line(param, self._spec.is_las30))
            lines.append("")

        return lines

    # ── Other section ────────────────────────────────────────────────

    def _write_other_section(self) -> list[str]:
        """Write ~O Other section — LAS 3.0 skips (deprecated)."""
        lines: list[str] = []
        if not self._las_file.other or not self._las_file.other.strip():
            return lines
        import warnings

        warnings.warn(
            "~Other section content was NOT written because LAS 3.0 "
            "deprecates the ~Other section.  Other content should be "
            "migrated to user-defined Parameter or Column Data sections.",
            stacklevel=3,
        )
        return lines

    # ── ASCII data sections ──────────────────────────────────────────

    def _write_ascii_sections(self) -> list[str]:
        """Write data sections — LAS 3.0 multi-section typed headers."""
        lines: list[str] = []
        null_value = _get_null_value(self._las_file.well)
        delimiter = self._las_file.version.delimiter_char
        import warnings

        _saved_dlm = self._las_file.version.dlm

        _saved_logs = dict(self._las_file.logs)
        _saved_string_data = dict(self._las_file.string_data)
        _saved_curves_order = list(self._las_file.curves_order)
        _saved_curves = list(self._las_file.curves)

        _actual_wrap = (self._las_file.version.wrap or "NO").upper()
        if _actual_wrap == "YES" or _actual_wrap != "NO":
            self._las_file.version.wrap = "NO"

        check_line_limit = (
            self._spec.line_length_limit_for_wrap(self._las_file.version.wrap) is not None
        )

        try:
            if self._las_file.data_sections:
                lines.extend(
                    self._write_ascii_las30(null_value, delimiter, check_line_limit, warnings)
                )
            else:
                # Fallback to legacy ~A for LAS 3.0 without data_sections.
                lines.extend(self._write_ascii_legacy(delimiter, check_line_limit))
        finally:
            self._las_file.logs = _saved_logs
            self._las_file.string_data = _saved_string_data
            self._las_file.curves_order = _saved_curves_order
            self._las_file.curves = _saved_curves
            self._las_file.version.dlm = _saved_dlm

        return lines

    def _effective_section_curves(self, section: DataSection) -> list[CurveDefinition] | None:
        """Resolve the curves that define a section's column scope.

        Prefers ``section.section_curves`` (the authoritative per-section
        definitions) as the DEFINITION source.  When ``section_curves`` is
        empty but ``curves_order`` is set (a documented-valid LAS 3.0
        pattern — the section inherits definitions from the top-level
        curve list), resolves each mnemonic from the top-level curves so
        the scoping comparison and per-section Definition still run
        (M-68).

        I2-13: whenever ``curves_order`` is present (regardless of
        ``section_curves``), the returned ORDER follows the LIVE
        ``curves_order`` — not the cached ``section_curves`` list.  A
        post-construction reorder of ``curves_order`` is therefore
        reflected in BOTH the pipe target AND the emitted data rows, so
        scoping and emission always agree at write time (no silent column
        swap).

        I2-22: mnemonic resolution is case-insensitive — a lowercase
        ``'dept'`` in ``curves_order`` resolves to the ``DEPT``
        definition instead of being dropped and silently relabeling the
        data.

        Returns None when no curves can be resolved.
        """
        if section.curves_order:
            # I2-13: section_curves is the definition source; the live
            # curves_order drives the order.  Fall back to top-level
            # curves when the section carries no per-section definitions.
            if section.section_curves:
                _source = section.section_curves
            else:
                _source = self._las_file.curves
            by_mnem: dict[str, CurveDefinition] = {}
            for c in _source:
                by_mnem.setdefault(c.mnemonic, c)
                by_mnem.setdefault(c.mnemonic.upper(), c)
            resolved: list[CurveDefinition] = []
            for mnem in section.curves_order:
                cdef = by_mnem.get(mnem) or by_mnem.get(mnem.upper())
                if cdef is not None and cdef not in resolved:
                    resolved.append(cdef)
            if resolved:
                return resolved
            if section.section_curves:
                return list(section.section_curves)
        elif section.section_curves:
            return list(section.section_curves)
        return None

    def _section_emission_pairs(
        self, section: DataSection
    ) -> tuple[list[tuple[str, CurveDefinition]], list[str]]:
        """Resolve the section's live column scope for data emission.

        Returns ``(pairs, unresolved)`` where ``pairs`` is an ordered list
        of ``(data_key, definition)`` — one entry per column the data rows
        will emit, in LIVE ``curves_order`` order — and ``unresolved`` is
        the list of ``curves_order`` entries that could not be resolved to
        any definition.

        ``data_key`` is the original ``curves_order`` entry (the key under
        which the section's ``data``/``string_data`` lookups succeed); the
        ``definition`` provides the EMITTED mnemonic for the pipe target
        and Definition block.  When ``curves_order`` is empty the resolved
        definitions' own mnemonics are used (section_curves-only sections).

        W-11: unresolvable entries are surfaced so the caller can refuse
        (data-bearing) or warn (data-free) instead of silently relabeling.
        I2-13: the order follows the LIVE ``curves_order`` — not a cached
        ``section_curves`` list — so the scoping comparison and the
        emitted rows agree at write time.  I2-22: mnemonic resolution is
        case-insensitive.
        """
        _eff = self._effective_section_curves(section) or []
        pairs: list[tuple[str, CurveDefinition]] = []
        if section.curves_order:
            # F-31: FIRST-wins resolution — consistent with the sibling
            # _effective_section_curves (setdefault).  A LAST-wins dict
            # comprehension here made case-variant duplicate mnemonics
            # (DEPT + dept) resolve the FIRST curves_order entry to the
            # WRONG definition, silently re-attributing the data column
            # to the second curve's unit/identity.
            by_upper: dict[str, CurveDefinition] = {}
            for c in _eff:
                by_upper.setdefault(c.mnemonic.upper(), c)
            for entry in section.curves_order:
                cd = by_upper.get(entry.upper())
                if cd is None:
                    continue
                if not any(e == entry for e, _ in pairs):
                    pairs.append((entry, cd))
        if not pairs and _eff:
            for cd in _eff:
                if not any(c is cd for _, c in pairs):
                    pairs.append((cd.mnemonic, cd))
        eff_mnems = {c.mnemonic.upper() for c in _eff}
        unresolved = [entry for entry in section.curves_order if entry.upper() not in eff_mnems]
        return pairs, unresolved

    def _write_ascii_las30(
        self,
        null_value: float,
        delimiter: str,
        check_line_limit: bool,
        _warnings_module: Any,
    ) -> list[str]:
        """LAS 3.0 multi-section typed data path (Path B of original _write_ascii_sections)."""
        lines: list[str] = []

        # Covered / orphaned top-level logs/string_data warnings.
        if self._las_file.logs or self._las_file.string_data:
            _ds_covered: dict[str, list[Any]] = {}
            for _ds in self._las_file.data_sections:
                for _k, _v_num in _ds.data.items():
                    _ds_covered.setdefault(_k, []).append(_v_num)
                for _k, _v_str in _ds.string_data.items():
                    _ds_covered.setdefault(_k, []).append(_v_str)
            # M-80: Parity with LAS 2.0 W-04 — a top-level logs/string_data
            # value whose key IS covered by a data_section is silently
            # dropped (the section's value wins).  Warn when the dropped
            # top-level value matches NO data_section's value (actual data
            # loss — the value survives nowhere in the output).  Matching
            # values (e.g. a parser roundtrip where the top-level view and
            # a section hold the same data, or a section that re-states the
            # top-level value) produce no warning — nothing is lost.
            if self._las_file.logs:
                _covered_conflicts = [
                    _k
                    for _k, _top_v in self._las_file.logs.items()
                    if _k in _ds_covered
                    and not any(_arrays_equal(_top_v, _sec_v) for _sec_v in _ds_covered[_k])
                ]
                if _covered_conflicts:
                    import warnings as _w3

                    _w3.warn(
                        f"Top-level logs value(s) {sorted(_covered_conflicts)} "
                        f"are also present in a data_section with a DIFFERENT "
                        f"value.  The LAS 3.0 writer path only writes data "
                        f"from data_sections; the top-level values for these "
                        f"curves will NOT appear in the output file.",
                        stacklevel=3,
                    )
            if self._las_file.string_data:
                _str_conflicts = [
                    _k
                    for _k, _top_v in self._las_file.string_data.items()
                    if _k in _ds_covered
                    and not any(_arrays_equal(_top_v, _sec_v) for _sec_v in _ds_covered[_k])
                ]
                if _str_conflicts:
                    import warnings as _w4

                    _w4.warn(
                        f"Top-level string_data value(s) {sorted(_str_conflicts)} "
                        f"are also present in a data_section with a DIFFERENT "
                        f"value.  The LAS 3.0 writer path only writes data "
                        f"from data_sections; the top-level values for these "
                        f"curves will NOT appear in the output file.",
                        stacklevel=3,
                    )
            _orphaned_logs = (
                set(self._las_file.logs.keys()) - set(_ds_covered.keys())
                if self._las_file.logs
                else set()
            )
            if _orphaned_logs:
                import warnings as _w

                _w.warn(
                    f"Top-level logs contain curve(s) not present in any "
                    f"data_section: {sorted(_orphaned_logs)}.  The LAS 3.0 "
                    f"writer path only writes data from data_sections; "
                    f"these curves' data will NOT appear in the output file.",
                    stacklevel=3,
                )
            _orphaned_string_data = (
                set(self._las_file.string_data.keys()) - set(_ds_covered.keys())
                if self._las_file.string_data
                else set()
            )
            if _orphaned_string_data:
                import warnings as _w2

                _w2.warn(
                    f"Top-level string_data contains curve(s) not present in any "
                    f"data_section: {sorted(_orphaned_string_data)}.  The LAS 3.0 "
                    f"writer path only writes data from data_sections; "
                    f"these curves' data will NOT appear in the output file.",
                    stacklevel=3,
                )

        emitted_defs: dict[str, dict[tuple[Any, ...], str]] = {}
        for section in self._las_file.data_sections:
            sec_type = (section.section_type or "LOG_DATA").upper()
            section_prefix = _section_type_to_prefix(sec_type)
            raw_section_name = (
                f" {_sanitize_las_value(section.name).replace('|', '')}" if section.name else ""
            )
            section_name = raw_section_name

            # ── W-11/W-12/I2-13/I2-21: resolve the section's live column
            # scope ONCE.  The SAME set drives the pipe target, the
            # per-section Definition, and the data rows, so the written
            # file can never silently relabel or discard data.
            _emission_pairs, _unresolved = self._section_emission_pairs(section)

            # W-11: a curves_order entry with NO definition cannot be
            # represented — its column would be silently relabeled onto
            # another curve (or discarded) on re-read.  Drop it from the
            # emission and warn loudly; the data rows below are emitted
            # for the RESOLVED set only, so no column is relabeled.
            # N2b-3: the data-bearing detection compared EXACT-case — a
            # case-variant data key ('ghost' vs curves_order entry
            # 'GHOST') fell into the data-free branch and fired the false
            # "no values are lost" assurance while the column's data WAS
            # dropped.  Match via _mnem_key (the emission path
            # _lookup_data_array is already case-insensitive).
            for _entry in _unresolved:
                if _mnem_key(_entry) in {_mnem_key(k) for k in (section.data or {})} or _mnem_key(
                    _entry
                ) in {_mnem_key(k) for k in (section.string_data or {})}:
                    _warnings_module.warn(
                        f"Curve '{_entry}' appears in the curves_order of "
                        f"section {section.name!r} but has no definition "
                        f"in the top-level curves or the section's "
                        f"section_curves.  The curve's DATA is dropped "
                        f"from the output — it cannot be represented "
                        f"without a definition, and re-read would "
                        f"silently relabel its values onto another "
                        f"curve.  Add a curve definition for '{_entry}' "
                        f"or remove it from curves_order.",
                        stacklevel=4,
                    )
                else:
                    _warnings_module.warn(
                        f"Curve '{_entry}' appears in the curves_order of "
                        f"section {section.name!r} but has no definition in "
                        f"the top-level curves or the section's "
                        f"section_curves.  The curve is dropped from the "
                        f"output — it has no data, so no values are lost.",
                        stacklevel=4,
                    )

            # W-10/W-12: two curves emitting the same mnemonic cannot both
            # be declared under that name.  A curve whose M-59
            # reconstruction would collide falls back to its OWN mnemonic
            # when free (preserving the distinct column — e.g. a
            # reader-renamed PERFT_2 alongside PERFT); when the own
            # mnemonic is ALSO taken the curve is a metadata-only
            # duplicate and is dropped — a dropped curve that carries
            # data would be silently discarded on re-read, so refuse.
            # N2b-2/N2b-3: _emitted_seen stores UPPER-CASED keys (a
            # case-variant pair must be treated as the duplicate it is),
            # and the data-bearing detection is case-insensitive — but it
            # must only RAISE when the colliding entry's data is actually
            # LOST.  A case-variant duplicate whose data key aliases the
            # surviving curve's array (F-31: curves_order=['DEPT','dept']
            # with data only under 'DEPT') shares the array — dropping the
            # second entry loses nothing and the "no values are lost"
            # warning is accurate.  An entry with its OWN distinct array
            # under a case-variant key ('y' vs 'Y') loses that data, so the
            # should-be LASWriteError must fire (never the false assurance).
            _emitted_seen: set[str] = set()
            _emit_pairs: list[tuple[str, CurveDefinition, str | None]] = []
            for _entry, _cd in _emission_pairs:
                _em = _emitted_mnemonic(_cd)
                _override: str | None = None
                if _mnem_key(_em) in _emitted_seen:
                    if _mnem_key(_cd.mnemonic) not in _emitted_seen:
                        _override = _cd.mnemonic
                    else:
                        _lost_arr, _ = _lookup_data_array(
                            _entry, section.data or {}, section.string_data or {}
                        )
                        # Does a surviving pair already emit the SAME array?
                        _shared = False
                        if _lost_arr is not None:
                            for _kept_entry, _kept_cd, _kept_override in _emit_pairs:
                                _kept_arr, _ = _lookup_data_array(
                                    _kept_entry, section.data or {}, section.string_data or {}
                                )
                                if _kept_arr is _lost_arr:
                                    _shared = True
                                    break
                        if _lost_arr is not None and not _shared:
                            raise LASWriteError(
                                f"Curve '{_entry}' in section {section.name!r} "
                                f"emits the same mnemonic '{_em}' as another "
                                f"curve in the section, AND has data.  The "
                                f"writer cannot represent both columns — "
                                f"re-read would silently discard or relabel "
                                f"one column's values.  Rename one of the "
                                f"colliding curves."
                            )
                        _warnings_module.warn(
                            f"Curve '{_entry}' in section {section.name!r} "
                            f"emits the same mnemonic '{_em}' as another "
                            f"curve in the section.  The curve is dropped "
                            f"from the output — it has no data, so no values "
                            f"are lost.",
                            stacklevel=4,
                        )
                        continue
                _emitted_seen.add(_mnem_key(_override or _em))
                _emit_pairs.append((_entry, _cd, _override))

            # The resolved+deduped definitions in live order — the set the
            # pipe target declares and the data rows align to.
            _sec_curves = [cd for _, cd, _ in _emit_pairs]
            _emitted_by_id = {
                id(cd): (_override or _emitted_mnemonic(cd))
                for _entry, cd, _override in _emit_pairs
            }

            # Definition prefix for sections that need per-section curve
            # scoping.  Non-LOG_DATA sections always emit a Definition
            # (dedup by signature) so their curves survive re-read.
            # LOG_DATA sections get one ONLY when their curve set/identity/
            # ORDER differs from the main ~C block — otherwise ``| CURVE``
            # scopes them to the main block correctly (N-I-20).  The
            # comparison is ORDER-SENSITIVE and identity-based
            # (mnemonic+unit+format+index), NOT a frozenset of mnemonics —
            # a frozenset comparison silently swapped columns when a
            # section had the same curve-name set but a different column
            # order (M-66) and missed unit/format/index differences
            # (M-26/M-64).  When ``section_curves`` is empty but
            # ``curves_order`` is set, the effective curves are derived so
            # the comparison still runs (M-68).
            def_prefix: str | None = None
            if sec_type != "LOG_DATA":
                def_prefix = _SECTION_TYPE_TO_DEFINITION_PREFIX.get(sec_type)
                if def_prefix is None:
                    if sec_type.endswith("_DATA"):
                        root = sec_type[: -len("_DATA")]
                        root = _sanitize_las_value(root)
                        def_prefix = root.title().replace("_", "")
                    elif section.section_curves:
                        import warnings

                        warnings.warn(
                            f"Unknown section type '{sec_type}' has per-section "
                            f"curve definitions.  Deriving definition prefix from "
                            f"section type name to preserve curve metadata.",
                            stacklevel=3,
                        )
                        st = _sanitize_las_value(sec_type)
                        def_prefix = st.title().replace("_", "")
            elif sec_type == "LOG_DATA" and _sec_curves:
                # F-16: the scoping comparison uses the SAME
                # (EMITTED-mnemonic-deduped) curve set as the main ~C
                # block's dedup loop.  Two curves that emit the same name
                # (LLD + BFV with original_mnemonic='LLD') fall back to
                # their own mnemonics above, so the section pipes
                # ``| CURVE`` to the clean main block instead of forcing a
                # per-section Definition that re-emits the duplicate
                # (structurally invalid, silently renamed on re-read).
                # W-10: the identity uses the COLLISION-FREE emitted
                # names (a reader-renamed PERFT_2 compares as PERFT_2,
                # not as PERFT).
                _sec_identity = tuple(
                    _curve_identity(c, _emitted_by_id[id(c)]) for c in _sec_curves
                )
                _main_identity = tuple(
                    _curve_identity(c, self._main_emitted_names.get(id(c)))
                    for c in self._main_curves
                )
                if _sec_identity != _main_identity:
                    # N-I-20: distinct curve set/identity/order — emit a
                    # per-section Definition and pipe to it.  The
                    # hardcoded ``| CURVE`` re-scoped EVERY LOG_DATA
                    # section to the global union on re-read, silently
                    # relabeling columns (e.g. DT landing in GR) for
                    # sections with their own scope.
                    def_prefix = "Log"

            # Emit per-section Definition section (dedup by curve signature).
            # I2-21: the emitted curve set is ALREADY deduped by emitted
            # mnemonic above, so a section whose raw curves collide on the
            # emitted name (LLD + BFV with original_mnemonic='LLD') cannot
            # re-emit duplicate Definition lines (structurally invalid).
            pipe_def_name: str | None = None
            if def_prefix:
                sec_curves = _sec_curves
                if sec_curves:
                    sig = tuple(
                        _definition_signature(curve, _emitted_by_id[id(curve)])
                        for curve in sec_curves
                    )
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
                        _sec_str = (
                            frozenset(_mnem_key(k) for k in section.string_data.keys())
                            if section.string_data
                            else frozenset()
                        )
                        for curve in sec_curves:
                            lines.append(
                                _format_curve_line(
                                    curve,
                                    self._spec.is_las30,
                                    _sec_str,
                                    mnemonic_override=_emitted_by_id.get(id(curve)),
                                )
                            )
                        lines.append("")
                    pipe_def_name = sig_map[sig]

            # Data section header with pipe notation.
            if sec_type == "LOG_DATA":
                if pipe_def_name:
                    lines.append(f"~{section_prefix}{section_name} | {pipe_def_name}")
                else:
                    lines.append(f"~{section_prefix}{section_name} | CURVE")
            elif pipe_def_name:
                lines.append(f"~{section_prefix}{section_name} | {pipe_def_name}")
            else:
                lines.append(f"~{section_prefix}{section_name}")
            # W-11/W-12/I2-13: emit data rows aligned to the SAME
            # resolved+deduped curve set the pipe target declares — the
            # live curves_order entries in live order.
            lines.extend(
                _format_data_rows(
                    [_entry for _entry, _cd, _override in _emit_pairs],
                    section.data,
                    section.string_data,
                    null_value,
                    delimiter,
                    self._precision,
                    is_las12=check_line_limit,
                )
            )

            # Warn about curves in curves_order that have no data,
            # matching the legacy path at _writer_base.py:564-571.
            # M13 (F-32 twin): the DATA-key lookup (_format_data_rows →
            # _lookup_data_array) is case-insensitive, so a case-variant
            # curves_order entry ('dept' vs data key 'DEPT') IS emitted,
            # not null-padded.  Compare upper-cased so the warning does
            # not falsely claim padding (mirrors _writer_base.py:1565-1573).
            _log_upper = {k.upper() for k in section.data.keys()} if section.data else set()
            _str_upper = (
                {k.upper() for k in section.string_data.keys()} if section.string_data else set()
            )
            _uncovered = {
                _k
                for _k in section.curves_order
                if _k.upper() not in _log_upper and _k.upper() not in _str_upper
            }
            if _uncovered:
                _warnings_module.warn(
                    f"Curve(s) {sorted(_uncovered)} appear in "
                    f"curves_order for section {section.name!r} but "
                    f"have no data in 'data' or 'string_data'.  "
                    f"The writer will pad these curves with "
                    f"null_value.",
                    stacklevel=4,
                )

        return lines
