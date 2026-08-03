"""LAS 1.2 writer — CWLS format with numeric/non-numeric well field branching."""

from __future__ import annotations

from ._writer_base import (
    _escape_colons_for_las_value,
    _sanitize_las_value,
    _WriterBase,
)
from .data_reader import _get_well_entry_ci
from .models import _MNEMONIC_PATTERN, LASFile


class _Las12Writer(_WriterBase):
    """LAS 1.2 format writer.

    Overrides section writers where LAS 1.2 diverges from the base defaults:
    - Version section: no DLM line, CWLS desc
    - Well section: numeric vs non-numeric colon placement
    - ASCII sections: forces SPACE delimiter
    """

    def __init__(self, las_file: LASFile, precision: str) -> None:
        super().__init__(las_file, precision)

    # ── Version section ──────────────────────────────────────────────

    def _write_version_section(self) -> list[str]:
        """Write ~V Version section — LAS 1.2 format (no DLM line)."""
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

        # LAS 1.2 does not emit DLM line (DLM is LAS 2.0+).
        lines.append("")
        return lines

    # ── Well section ─────────────────────────────────────────────────

    def _write_well_section(self) -> list[str]:
        """Write ~W Well section — CWLS LAS 1.2 format.

        Numeric well fields (STRT, STOP, STEP, NULL) use VALUE before colon.
        Non-numeric fields use DESC before colon, VALUE after.
        """
        lines: list[str] = []
        lines.append("~WELL INFORMATION")

        for key in self._las_file.well.entries:
            if not isinstance(key, str):
                raise TypeError(
                    f"WellSection entry key must be str, got {type(key).__name__}: {key!r}"
                )
        # N-I-19: Defensive well-key CONTENT validation (mirrors the LAS
        # 2.0/3.0 writer).  A dot/space/colon key is emitted and then
        # silently dropped on re-read — the parser's ~W regex cannot match
        # it.  Reject here so metadata is not silently lost.
        for key in self._las_file.well.entries:
            if not _MNEMONIC_PATTERN.fullmatch(key):
                raise ValueError(
                    f"WellSection entry key {key!r} contains characters "
                    f"the LAS parser cannot roundtrip.  Well keys must "
                    f"match {_MNEMONIC_PATTERN.pattern!r}."
                )

        mandatory_order = ["STRT", "STOP", "STEP", "NULL"]
        ordered_keys: list[str] = []
        for mandatory in mandatory_order:
            for key in self._las_file.well.entries:
                if key.upper() == mandatory and key not in ordered_keys:
                    ordered_keys.append(key)
                    break
        for key in self._las_file.well.entries:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            value = self._las_file.well.entries[key]
            # II-20 (X-3): well.entries keys and units/descriptions keys can
            # differ in case (from_dict mnem_base=None / direct construction
            # store them verbatim), so an exact-case .get(key) silently
            # dropped the unit/description from the emitted ~W line.  Use the
            # codebase's CI well lookup (data_reader._get_well_entry_ci),
            # matching the base writer's fix (II-20 for LAS 2.0/3.0).
            unit = _sanitize_las_value(_get_well_entry_ci(self._las_file.well.units or {}, key, ""))
            unit_dot = f".{unit}" if unit else "."
            # W-07 (M-28 parity): well values/descriptions are emitted
            # mid-line (never at line start) so a leading '~' must be
            # preserved, not stripped — matching the LAS 2.0/3.0 base
            # writer.  Stripping it silently corrupted the model value on
            # write→read (WELL='~INCIDENTAL' → 'INCIDENTAL').
            val = _sanitize_las_value(value, preserve_leading_tilde=True)
            desc = _sanitize_las_value(
                _get_well_entry_ci(self._las_file.well.descriptions or {}, key, ""),
                preserve_leading_tilde=True,
            )
            val = _escape_colons_for_las_value(val)
            desc = _escape_colons_for_las_value(desc)
            desc_str = f"  {desc}" if desc else ""
            if key.upper() in {"STRT", "STOP", "STEP", "NULL"}:
                lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {val}  :{desc_str}")
            else:
                if desc:
                    lines.append(f" {_sanitize_las_value(key)}{unit_dot}   {desc}  : {val}")
                else:
                    lines.append(f" {_sanitize_las_value(key)}{unit_dot}    : {val}")
        lines.append("")
        return lines

    # ── ASCII data sections ──────────────────────────────────────────

    def _write_ascii_sections(self) -> list[str]:
        """Write data sections — LAS 1.2 forces SPACE delimiter."""
        lines: list[str] = []
        delimiter = self._las_file.version.delimiter_char
        import warnings

        _saved_dlm = self._las_file.version.dlm

        if delimiter != " ":
            warnings.warn(
                f"LAS 1.2 does not support the '{self._las_file.version.dlm}' delimiter. "
                "Forcing SPACE delimiter for data rows to match the header section.",
                stacklevel=3,
            )
            delimiter = " "
            self._las_file.version.dlm = "SPACE"

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
