"""LAS 3.0 writer — typed sections, per-section parameters, zone notation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ._writer_base import (
    _SECTION_TYPE_TO_DEFINITION_PREFIX,
    _format_curve_line,
    _format_data_rows,
    _format_parameter_line,
    _sanitize_las_value,
    _section_type_to_prefix,
    _WriterBase,
)
from .data_reader import _get_null_value
from .models import CurveDefinition, DataSection, LASFile, ParameterEntry


def _emitted_mnemonic(curve: CurveDefinition) -> str:
    """The mnemonic as written to the ~C / Definition line.

    Mirrors the emission logic in ``_format_curve_line``:
    - M-59 (F-16): when ``curve.original_mnemonic`` is set and differs
      from ``curve.mnemonic``, ``_format_curve_line`` emits the
      VENDOR-standard original name (reconstructing e.g. ``LLD`` from the
      reader-renamed ``BFV``).  The dedup/identity keys MUST use the same
      emitted name, or two curves that collide in the output (LLD + BFV
      with original_mnemonic='LLD') are seen as DISTINCT by dedup and BOTH
      are emitted — duplicate LLD lines in ~C, structurally invalid file,
      silent BFV identity loss on re-read (M-64 dedup-key divergence).
    - W-09: a curve with ``array_info`` but no ``[N]`` in its mnemonic is
      emitted with the bracket form (``NMR`` + index=1 → ``NMR[1]``).
      Using the EMITTED mnemonic for dedup/scoping keeps directly-
      constructed models (base mnemonic + array_info) consistent with
      parsed models (bracket mnemonic) — without it, NMR[1]/NMR[2]
      collide (M-64).
    """
    mnemonic = (
        curve.original_mnemonic
        if curve.original_mnemonic and curve.original_mnemonic != curve.mnemonic
        else curve.mnemonic
    )
    if curve.array_info is not None and "[" not in mnemonic:
        mnemonic = f"{mnemonic}[{curve.array_info.index}]"
    return mnemonic


def _curve_identity(curve: CurveDefinition) -> tuple[Any, ...]:
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
    """
    return (
        _emitted_mnemonic(curve),
        curve.unit or "",
        curve.data_format or "",
        curve.array_info.index if curve.array_info is not None else None,
    )


def _definition_signature(curve: CurveDefinition) -> tuple[Any, ...]:
    """Full per-curve signature used to dedup per-section Definition blocks.

    Unlike ``_curve_identity`` this INCLUDES description/api_code (two
    sections with identical scoping but different metadata must get
    separate Definition blocks) and the array index + time_offset (M-64:
    the previous signature omitted ``array_info.index`` so NMR[1]/NMR[2]
    collapsed to one Definition).
    """
    return (
        _emitted_mnemonic(curve),
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


def _dedup_by_emitted_mnemonic(curves: list[CurveDefinition]) -> list[CurveDefinition]:
    """Dedup a curve list by the EMITTED mnemonic (first definition wins).

    F-16 (M-59 ↔ M-64): ``_format_curve_line`` emits
    ``original_mnemonic`` when it differs from ``curve.mnemonic``, so two
    distinct curves can emit the SAME name (e.g. a real ``LLD`` and a
    ``BFV`` with ``original_mnemonic='LLD'``).  The main ~C block's dedup
    loop keys on ``_emitted_mnemonic``; the LOG_DATA scoping comparison
    must use the SAME (deduped) curve set or a section whose curves
    collide on the emitted name would get a per-section Definition that
    re-emits the duplicate (structurally invalid file).  First definition
    wins, mirroring the ~C block's W-01 dedup.
    """
    seen: set[str] = set()
    deduped: list[CurveDefinition] = []
    for curve in curves:
        emitted = _emitted_mnemonic(curve)
        if emitted in seen:
            continue
        seen.add(emitted)
        deduped.append(curve)
    return deduped


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
        mnems = set(self._las_file.string_data.keys()) if self._las_file.string_data else set()
        for ds in self._las_file.data_sections:
            if ds.string_data:
                mnems.update(ds.string_data.keys())
        return frozenset(mnems)

    # ── Version section ──────────────────────────────────────────────

    def _write_version_section(self) -> list[str]:
        """Write ~V Version section — LAS 3.0 format."""
        lines: list[str] = []
        lines.append("~VERSION INFORMATION")
        vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0"
        vers = self._las_file.version.vers or "2.0"
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
                            if _emitted not in emitted_mnems:
                                emitted_mnems.add(_emitted)
                                curves_to_emit.append(curve)
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
                                    if _emitted_mnemonic(emitted) == _emitted:
                                        _emitted_unit = emitted.unit or ""
                                        _curve_unit = curve.unit or ""
                                        _emitted_fmt = emitted.data_format or ""
                                        _curve_fmt = curve.data_format or ""
                                        _emitted_idx = (
                                            emitted.array_info.index
                                            if emitted.array_info else None
                                        )
                                        _curve_idx = (
                                            curve.array_info.index
                                            if curve.array_info else None
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
                                        _desc_changed = (
                                            (emitted.description or "")
                                            != (curve.description or "")
                                        )
                                        _api_changed = (
                                            (emitted.api_code or "")
                                            != (curve.api_code or "")
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
                                            if (
                                                not emitted.description
                                                and curve.description
                                            ):
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
            curves_by_mnem: dict[str, CurveDefinition] = {}
            for _c in self._las_file.curves:
                curves_by_mnem.setdefault(_c.mnemonic, _c)
                curves_by_mnem.setdefault(_emitted_mnemonic(_c), _c)
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
                        _emitted = _emitted_mnemonic(
                            curves_by_mnem[mnem]
                        ) if mnem in curves_by_mnem else mnem
                        if _emitted not in emitted_mnems:
                            curve_def = curves_by_mnem.get(mnem)
                            if curve_def is not None:
                                emitted_mnems.add(_emitted)
                                curves_to_emit.append(curve_def)
                            else:
                                # N-I-15: the section curve is absent from
                                # top-level curves and cannot be emitted —
                                # warn so the data loss on re-read is
                                # visible at write time instead of silent.
                                import warnings

                                warnings.warn(
                                    f"Curve '{mnem}' appears in a LAS 3.0 "
                                    f"data section's curves_order but has no "
                                    f"definition in the top-level curves or "
                                    f"any section's section_curves.  The "
                                    f"written file will have more data "
                                    f"columns than ~C curve definitions; "
                                    f"this curve's data will be discarded "
                                    f"on re-read.",
                                    UserWarning,
                                    stacklevel=3,
                                )

            # W-01: When ALL data sections are non-LOG_DATA and carry their
            # own section_curves (e.g. CORE-only files), the curve
            # definitions belong in the typed ~X_Definition sections — the
            # top-level fallback would duplicate them in ~C AND the
            # Definition block, inflating the re-read curve count (2→4).
            curves_in_definitions = bool(
                self._las_file.data_sections
            ) and all(
                (ds.section_type or "LOG_DATA").upper() != "LOG_DATA"
                and bool(ds.section_curves)
                for ds in self._las_file.data_sections
            )
            if not curves_to_emit and not curves_in_definitions:
                curves_to_emit = list(self._las_file.curves)
            if not curves_to_emit:
                if not curves_in_definitions:
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
            # Skip when curves_in_definitions — the CORE-only fallback
            # already emits the full top-level list there.
            if not curves_in_definitions and self._las_file.curves:
                _section_mnems: set[str] = set()
                for ds in self._las_file.data_sections:
                    if ds.section_curves:
                        _section_mnems.update(
                            _emitted_mnemonic(c) for c in ds.section_curves
                        )
                    _section_mnems.update(ds.curves_order)
                _top_data_mnems = (
                    set(self._las_file.logs.keys())
                    | set(self._las_file.string_data.keys())
                )
                for curve in self._las_file.curves:
                    _emitted = _emitted_mnemonic(curve)
                    if _emitted in emitted_mnems:
                        continue
                    if _emitted in _section_mnems or curve.mnemonic in _section_mnems:
                        continue
                    if _emitted in _top_data_mnems or curve.mnemonic in _top_data_mnems:
                        # Has data at the top level (e.g. orphaned logs) —
                        # handled by the covered/orphan warnings, not here.
                        continue
                    # Definition-only: no data in any section and no
                    # top-level data either.  Emit the definition so its
                    # metadata survives; warn that it has no data.
                    emitted_mnems.add(_emitted)
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
            _all_str = self._all_string_mnemonics()
            for curve in curves_to_emit:
                lines.append(_format_curve_line(curve, self._spec.is_las30, _all_str))
        else:
            self._main_curves = list(self._las_file.curves)
            _top_str = (
                frozenset(self._las_file.string_data.keys())
                if self._las_file.string_data
                else frozenset()
            )
            for curve in self._las_file.curves:
                lines.append(_format_curve_line(curve, self._spec.is_las30, _top_str))

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
                if param.section_type and param.section_type.strip() else None
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

        check_line_limit = self._spec.line_length_limit_for_wrap(
            self._las_file.version.wrap
        ) is not None

        try:
            if self._las_file.data_sections:
                lines.extend(self._write_ascii_las30(
                    null_value, delimiter, check_line_limit, warnings
                ))
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

    def _effective_section_curves(
        self, section: DataSection
    ) -> list[CurveDefinition] | None:
        """Resolve the curves that define a section's column scope.

        Prefers ``section.section_curves`` (the authoritative per-section
        definitions).  When ``section_curves`` is empty but ``curves_order``
        is set (a documented-valid LAS 3.0 pattern — the section inherits
        definitions from the top-level curve list), resolves each mnemonic
        from the top-level curves so the scoping comparison and per-section
        Definition still run (M-68).  Returns None when no curves can be
        resolved.

        """
        if section.section_curves:
            return list(section.section_curves)
        if section.curves_order:
            by_mnem = {c.mnemonic: c for c in self._las_file.curves}
            resolved: list[CurveDefinition] = []
            for mnem in section.curves_order:
                cdef = by_mnem.get(mnem)
                if cdef is not None:
                    resolved.append(cdef)
            if resolved:
                return resolved
        return None

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
                    _k for _k, _top_v in self._las_file.logs.items()
                    if _k in _ds_covered
                    and not any(
                        _arrays_equal(_top_v, _sec_v)
                        for _sec_v in _ds_covered[_k]
                    )
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
                    _k for _k, _top_v in self._las_file.string_data.items()
                    if _k in _ds_covered
                    and not any(
                        _arrays_equal(_top_v, _sec_v)
                        for _sec_v in _ds_covered[_k]
                    )
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
                if self._las_file.logs else set()
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
                if self._las_file.string_data else set()
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
                f" {_sanitize_las_value(section.name).replace('|', '')}"
                if section.name else ""
            )
            section_name = raw_section_name

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
            elif sec_type == "LOG_DATA":
                _sec_curves = self._effective_section_curves(section)
                if _sec_curves is not None:
                    # F-16: the scoping comparison must use the same
                    # (EMITTED-mnemonic-deduped) curve set as the main ~C
                    # block's dedup loop.  Two curves that emit the same
                    # name (LLD + BFV with original_mnemonic='LLD') were
                    # already deduped in ~C — the section's raw set would
                    # differ from the main block and force a per-section
                    # Definition that re-emits the duplicate (structurally
                    # invalid, silently renamed on re-read).  Compare the
                    # deduped set so such sections pipe ``| CURVE`` to the
                    # clean main block instead.
                    _sec_deduped = _dedup_by_emitted_mnemonic(_sec_curves)
                    _sec_identity = tuple(_curve_identity(c) for c in _sec_deduped)
                    _main_identity = tuple(_curve_identity(c) for c in self._main_curves)
                    if _sec_identity != _main_identity:
                        # N-I-20: distinct curve set/identity/order — emit a
                        # per-section Definition and pipe to it.  The
                        # hardcoded ``| CURVE`` re-scoped EVERY LOG_DATA
                        # section to the global union on re-read, silently
                        # relabeling columns (e.g. DT landing in GR) for
                        # sections with their own scope.
                        def_prefix = "Log"

            # Emit per-section Definition section (dedup by curve signature).
            pipe_def_name: str | None = None
            if def_prefix:
                sec_curves = self._effective_section_curves(section)
                if sec_curves:
                    sig = tuple(
                        _definition_signature(curve) for curve in sec_curves
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
                            frozenset(section.string_data.keys())
                            if section.string_data
                            else frozenset()
                        )
                        for curve in sec_curves:
                            lines.append(_format_curve_line(curve, self._spec.is_las30, _sec_str))
                        lines.append("")
                    pipe_def_name = sig_map[sig]

            # Data section header with pipe notation.
            if sec_type == "LOG_DATA":
                if pipe_def_name:
                    lines.append(
                        f"~{section_prefix}{section_name} | {pipe_def_name}"
                    )
                else:
                    lines.append(f"~{section_prefix}{section_name} | CURVE")
            elif pipe_def_name:
                lines.append(
                    f"~{section_prefix}{section_name} | {pipe_def_name}"
                )
            else:
                lines.append(f"~{section_prefix}{section_name}")
            lines.extend(
                _format_data_rows(
                    section.curves_order,
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
            _log_keys = set(section.data.keys()) if section.data else set()
            _str_keys = set(section.string_data.keys()) if section.string_data else set()
            _order_set = set(section.curves_order)
            _uncovered = _order_set - _log_keys - _str_keys
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
