"""LAS 3.0 writer — typed sections, per-section parameters, zone notation."""

from __future__ import annotations

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
from .models import LASFile, ParameterEntry


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
            curves_to_emit = []
            for ds in self._las_file.data_sections:
                if (ds.section_type or "LOG_DATA").upper() == "LOG_DATA":
                    if ds.section_curves:
                        curves_to_emit.extend(ds.section_curves)
            emitted_mnems = {c.mnemonic for c in curves_to_emit}
            curves_by_mnem = {c.mnemonic: c for c in self._las_file.curves}
            for ds in self._las_file.data_sections:
                if (ds.section_type or "LOG_DATA").upper() == "LOG_DATA":
                    if not ds.section_curves and ds.curves_order:
                        for mnem in ds.curves_order:
                            if mnem not in emitted_mnems:
                                curve_def = curves_by_mnem.get(mnem)
                                if curve_def is not None:
                                    curves_to_emit.append(curve_def)
                                    emitted_mnems.add(mnem)

            if not curves_to_emit:
                curves_to_emit = list(self._las_file.curves)
            if not curves_to_emit:
                import warnings

                warnings.warn(
                    "No curves to emit for ~C section — skipping",
                    UserWarning,
                    stacklevel=3,
                )
                lines.append("")
                return lines
            for curve in curves_to_emit:
                lines.append(_format_curve_line(curve, self._spec.is_las30))
        else:
            for curve in self._las_file.curves:
                lines.append(_format_curve_line(curve, self._spec.is_las30))

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

    def _write_ascii_las30(
        self,
        null_value: float,
        delimiter: str,
        check_line_limit: bool,
        _warnings_module: object,
    ) -> list[str]:
        """LAS 3.0 multi-section typed data path (Path B of original _write_ascii_sections)."""
        lines: list[str] = []

        # Orphaned logs/string_data warning.
        if self._las_file.logs or self._las_file.string_data:
            _ds_covered: set[str] = set()
            for _ds in self._las_file.data_sections:
                _ds_covered.update(_ds.data.keys())
                _ds_covered.update(_ds.string_data.keys())
            _orphaned_logs = (
                set(self._las_file.logs.keys()) - _ds_covered
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
                set(self._las_file.string_data.keys()) - _ds_covered
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

        emitted_defs: dict[str, dict[tuple[tuple[str, str, str, str, str, float | None], ...], str]] = {}
        for section in self._las_file.data_sections:
            sec_type = (section.section_type or "LOG_DATA").upper()
            section_prefix = _section_type_to_prefix(sec_type)
            raw_section_name = (
                f" {_sanitize_las_value(section.name).replace('|', '')}"
                if section.name else ""
            )
            section_name = raw_section_name

            # Definition prefix for non-LOG_DATA sections.
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

            # Emit per-section Definition section (dedup by curve signature).
            pipe_def_name: str | None = None
            if sec_type != "LOG_DATA" and section.section_curves:
                sig = tuple(
                    (curve.mnemonic, curve.unit or "", curve.description or "", curve.data_format or "",
                     curve.api_code or "",
                     curve.array_info.time_offset if curve.array_info else None)
                    for curve in section.section_curves
                )
                if def_prefix:
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
                        for curve in section.section_curves:
                            lines.append(_format_curve_line(curve, self._spec.is_las30))
                        lines.append("")
                    pipe_def_name = sig_map[sig]

            # Data section header with pipe notation.
            if sec_type == "LOG_DATA":
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

        return lines
