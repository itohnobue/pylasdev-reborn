"""LAS 2.0 writer — standard format with DLM support."""

from __future__ import annotations

from ._writer_base import (
    _sanitize_las_value,
    _WriterBase,
)
from .models import LASFile


class _Las20Writer(_WriterBase):
    """LAS 2.0 format writer.

    Overrides section writers where LAS 2.0 diverges from base defaults:
    - Version section: standard vers_desc, DLM line, WRAP logic
    - ASCII sections: preserves user DLM choice (no SPACE force)
    """

    def __init__(self, las_file: LASFile, precision: str) -> None:
        super().__init__(las_file, precision)

    # ── Version section ──────────────────────────────────────────────

    def _write_version_section(self) -> list[str]:
        """Write ~V Version section — LAS 2.0 format."""
        lines: list[str] = []
        lines.append("~VERSION INFORMATION")
        vers_desc = "CWLS LOG ASCII STANDARD"
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

    # ── ASCII data sections ──────────────────────────────────────────

    def _write_ascii_sections(self) -> list[str]:
        """Write data sections — LAS 2.0 legacy ~A path (no SPACE force)."""
        lines: list[str] = []
        delimiter = self._las_file.version.delimiter_char

        _saved_dlm = self._las_file.version.dlm

        _saved_logs = dict(self._las_file.logs)
        _saved_string_data = dict(self._las_file.string_data)
        _saved_curves_order = list(self._las_file.curves_order)
        _saved_curves = list(self._las_file.curves)

        _actual_wrap = (self._las_file.version.wrap or "NO").upper()
        if _actual_wrap == "YES":
            self._las_file.version.wrap = "NO"

        check_line_limit = (
            self._spec.line_length_limit_for_wrap(self._las_file.version.wrap) is not None
        )

        try:
            lines.extend(self._write_ascii_legacy(delimiter, check_line_limit))
        finally:
            self._las_file.logs = _saved_logs
            self._las_file.string_data = _saved_string_data
            self._las_file.curves_order = _saved_curves_order
            self._las_file.curves = _saved_curves
            self._las_file.version.dlm = _saved_dlm

        return lines
