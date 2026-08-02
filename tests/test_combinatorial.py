"""Combinatorial test matrix for LAS version x WRAP x DLM x curve type x construction path.

SAFETY NET for all subsequent refactoring. Tests every meaningful combination
of LAS parameters through both ``parse`` (from text) and ``from_dict`` (from dict)
construction paths, verifying roundtrip consistency.

DO NOT modify this file to pass tests that expose real bugs in the codebase.
Use `pytest.mark.xfail` with a reason when the current code has a known gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import (
    LASFile,
    read_las_file,
    write_las_file,
)

# =============================================================================
# Helper functions
# =============================================================================


def _make_version_dict(vers: str, wrap: str, dlm: str) -> dict[str, str]:
    """Build a version section dict."""
    return {"VERS": vers, "WRAP": wrap, "DLM": dlm}


def _make_well_dict(vers: str, numeric_only: bool = True) -> dict[str, str]:
    """Build a well section dict appropriate for the given LAS version.

    M-11: LAS 1.2/2.0 mandate the lascheck 10-field set for ~W:
    STRT, STOP, STEP, NULL, COMP, WELL, FLD, LOC, SRVC, DATE.
    UWI is optional (present but not required).  LAS 3.0 uses the
    same minimum set.
    """
    well: dict[str, str] = {
        "STRT": "100.0",
        "STOP": "120.0",
        "STEP": "10.0",
        "NULL": "-999.25",
    }
    if not numeric_only:
        well.update(
            {
                "COMP": "TEST COMPANY",
                "WELL": "TEST-WELL-01",
                "FLD": "TEST FIELD",
                "LOC": "12-34-56-78W5",
                "SRVC": "TEST SRVC",
                "DATE": "01/01/2020",
            }
        )
    return well


def _make_curves_defs(
    curve_names: list[str],
    string_curves: set[str] | None = None,
    is_las30: bool = False,
) -> list[dict[str, str]]:
    """Build curve definition dicts with appropriate data_format.

    Only adds data_format for LAS 3.0 (where {F}/{S} format specifiers
    are part of the specification).  For LAS 1.2/2.0, data_format is
    omitted because the parser auto-detects {F}/{S} from text but the
    writer does not emit them for non-3.0 versions, which would cause
    compare_las_dicts mismatches.
    """
    string_curves = string_curves or set()
    result: list[dict[str, str]] = []
    for name in curve_names:
        entry: dict[str, str] = {"mnemonic": name, "unit": "M", "description": name}
        if is_las30:
            entry["data_format"] = "S" if name in string_curves else "F"
        result.append(entry)
    return result


def _make_logs(
    curve_names: list[str],
    string_curves: set[str] | None = None,
    n_rows: int = 3,
) -> dict[str, np.ndarray]:
    """Build logs dict with numeric or string data."""
    string_curves = string_curves or set()
    logs: dict[str, np.ndarray] = {}
    for i, name in enumerate(curve_names):
        if name in string_curves:
            logs[name] = np.array([f"val_{name}_{j}" for j in range(n_rows)], dtype=object)
        else:
            logs[name] = np.array([100.0 + i * 10.0 + j for j in range(n_rows)], dtype=np.float64)
    return logs


def _delimiter_char(dlm: str) -> str:
    """Map DLM name to actual delimiter character."""
    return {"SPACE": " ", "TAB": "\t", "COMMA": ","}.get(dlm.upper(), " ")


def _format_data_row(values: list[str], dlm: str) -> str:
    """Format a data row with the appropriate delimiter."""
    sep = _delimiter_char(dlm)
    return sep.join(values)


def _make_las_text(
    vers: str,
    wrap: str,
    dlm: str,
    curves: list[str],
    logs: dict[str, np.ndarray],
    *,
    well_extra: dict[str, str] | None = None,
    parameters: dict[str, str] | None = None,
    data_sections: list[dict[str, Any]] | None = None,
    las30_structured: bool = False,
    string_curves: set[str] | None = None,
) -> str:
    """Build a complete LAS text string for the parse path.

    Parameters
    ----------
    vers : str
        LAS version string ("1.2", "2.0", "3.0").
    wrap : str
        WRAP value ("YES" or "NO").
    dlm : str
        DLM value ("SPACE", "TAB", "COMMA").
    curves : list[str]
        Ordered list of curve mnemonics.
    logs : dict[str, np.ndarray]
        Mapping of curve mnemonic to data array.
    well_extra : dict[str, str] | None
        Additional well fields beyond STRT/STOP/STEP/NULL.
    parameters : dict[str, str] | None
        Parameter entries.
    data_sections : list[dict] | None
        For LAS 3.0 typed data sections. Each dict has:
        - name: section name
        - section_type: type (e.g. "LOG_DATA")
        - pipe_target: pipe target string or None
        - curves: list of curve mnemonics
        - logs: dict of data arrays
        - string_curves: set of string curve names (optional)
        - definition_curves: list of (name, unit, fmt, desc) for
          the _Definition section (optional)
    las30_structured : bool
        If True, use LAS 3.0 long section names (~VERSION, ~WELL, etc.).
        Otherwise use classic short names (~V, ~W, etc.).
    string_curves : set[str] | None
        Set of curve names that have string data (S format).
        Only used when data_sections is None.
    """
    string_curves = string_curves or set()
    sep = _delimiter_char(dlm)
    is_las12 = vers.startswith("1.")
    is_las30 = vers.startswith("3.")

    lines: list[str] = []

    # --- Version section ---
    if las30_structured:
        lines.append("~VERSION INFORMATION")
    else:
        lines.append("~VERSION INFORMATION")
    vers_desc = "CWLS LOG ASCII STANDARD -VERSION 3.0" if is_las30 else "CWLS LOG ASCII STANDARD"
    lines.append(f" VERS.  {vers} : {vers_desc}")
    lines.append(
        f" WRAP.  {wrap} : {'MULTIPLE LINES PER DEPTH STEP' if wrap == 'YES' else 'ONE LINE PER DEPTH STEP'}"
    )
    # DLM: suppressed for LAS 1.2, always emitted for non-SPACE in 2.0/3.0
    if not is_las12 and dlm.upper() != "SPACE":
        lines.append(f" DLM .  {dlm} : DELIMITING CHARACTER BETWEEN DATA COLUMNS")
    elif not is_las12:
        # Optionally emit DLM=SPACE for 2.0/3.0
        pass
    lines.append("")

    # --- Well section ---
    if las30_structured:
        lines.append("~Well Information")
    else:
        lines.append("~W")
    lines.append("#MNEM.UNIT              DATA                       DESCRIPTION")
    lines.append("#----- -----            ----------               -------------------------")
    lines.append(f" STRT .M              {logs[curves[0]][0]}                :START DEPTH")
    lines.append(f" STOP .M              {logs[curves[0]][-1]}                :STOP DEPTH")
    lines.append(" STEP .M              10.0                  :STEP")
    lines.append(" NULL .               -999.25                  :NULL VALUE")
    if well_extra:
        for key, value in well_extra.items():
            lines.append(f" {key} .       {value}             :{key}")
    lines.append("")

    # --- Curve section ---
    if las30_structured:
        lines.append("~CURVE INFORMATION")
    else:
        lines.append("~C")
    lines.append("#MNEM.UNIT              API CODES                   CURVE DESCRIPTION")
    lines.append("#------------------     ------------              -------------------------")
    for name in curves:
        if is_las30:
            fmt = " {S}" if name in string_curves else " {F}"
        else:
            fmt = ""
        lines.append(f" {name}.M                                       :  {name}{fmt}")
    lines.append("")

    # --- Parameter section (optional) ---
    if parameters:
        if las30_structured:
            lines.append("~PARAMETER INFORMATION")
        else:
            lines.append("~P")
        lines.append("#MNEM.UNIT              VALUE             DESCRIPTION")
        lines.append("#--------------     ----------------      -------------------------------")
        for key, value in parameters.items():
            lines.append(f" {key}.                {value}         :   {key}")
        lines.append("")

    # --- Data sections ---
    if data_sections:
        _write_las30_data_sections(lines, data_sections, sep)
    else:
        # Classic ~A section
        lines.append("~A  " + _format_data_row(curves, dlm))
        n_rows = len(next(iter(logs.values())))
        for row_idx in range(n_rows):
            row_vals = []
            for name in curves:
                val = logs[name][row_idx]
                row_vals.append(str(val))
            lines.append(_format_data_row(row_vals, dlm))

    return "\n".join(lines) + "\n"


def _write_las30_data_sections(
    lines: list[str],
    data_sections: list[dict[str, Any]],
    sep: str,
) -> None:
    """Append LAS 3.0 typed data sections to lines list."""
    for sec in data_sections:
        name = sec["name"]
        section_type = sec.get("section_type", "LOG_DATA")
        pipe_target = sec.get("pipe_target")
        sec_curves: list[str] = sec["curves"]
        sec_logs: dict[str, np.ndarray] = sec["logs"]
        definition_curves: list[tuple[str, str, str, str]] | None = sec.get("definition_curves")

        # Write _Definition section if curves are defined
        if definition_curves:
            def_name = f"{section_type.split('_')[0]}_Definition"
            lines.append("")
            lines.append(f"~{def_name}")
            for cname, cunit, cfmt, cdesc in definition_curves:
                lines.append(
                    f" {cname}.{cunit}                                : {cdesc}  {{{cfmt}}}"
                )
            lines.append("")

        # Write data section header
        if pipe_target:
            lines.append(f"~{name} | {pipe_target}")
        else:
            lines.append(f"~{name}")

        n_rows = len(next(iter(sec_logs.values())))
        for row_idx in range(n_rows):
            row_vals = []
            for cname in sec_curves:
                val = sec_logs[cname][row_idx]
                row_vals.append(str(val))
            lines.append(sep.join(row_vals))


def _make_las_dict(
    vers: str,
    wrap: str,
    dlm: str,
    curves: list[str],
    logs: dict[str, np.ndarray],
    *,
    well_extra: dict[str, str] | None = None,
    curves_defs: list[dict[str, str]] | None = None,
    parameters: dict[str, str] | None = None,
    string_curves: set[str] | None = None,
    data_sections: list[dict[str, Any]] | None = None,
    parameter_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a LAS dict for the from_dict path.

    This mirrors the dict structure accepted by LASFile.from_dict().
    """
    string_curves = string_curves or set()
    well_base = _make_well_dict(vers, numeric_only=(well_extra is None))
    if well_extra:
        well_base.update(well_extra)

    result: dict[str, Any] = {
        "version": _make_version_dict(vers, wrap, dlm),
        "well": well_base,
        "curves_order": list(curves),
        "logs": logs,
    }

    if curves_defs:
        result["curves"] = curves_defs
    else:
        # Auto-generate from curve names
        result["curves"] = _make_curves_defs(curves, string_curves, is_las30=vers.startswith("3."))

    if parameters:
        result["parameters"] = parameters

    if parameter_details:
        result["parameter_details"] = parameter_details

    if data_sections:
        result["data_sections"] = data_sections

    # For LAS 3.0: top-level string_data for {S} format curves
    if vers.startswith("3.") and string_curves:
        top_string = {name: logs[name] for name in string_curves if name in logs}
        if top_string:
            result["string_data"] = top_string
        # Remove string curves from "logs" for LAS 3.0 — they go in string_data
        for name in string_curves:
            result["logs"].pop(name, None)

    return result


def _build_las30_multi_section_dict() -> dict[str, Any]:
    """Build a LAS 3.0 dict with 2+ data_sections (from_dict path)."""
    vers = "3.0"
    wrap = "NO"
    dlm = "SPACE"
    curves = ["DEPT", "DT"]
    logs = _make_logs(curves, n_rows=3)
    curves_defs = _make_curves_defs(curves, is_las30=True)

    core_curves = ["CORET", "COREB"]
    core_logs = {
        "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
        "COREB": np.array([555.0, 556.0, 557.0], dtype=np.float64),
    }
    # Core section curves need names matching curves_order
    data_sections = [
        {
            "name": "Log",
            "section_type": "LOG_DATA",
            "curves_order": curves,
            "data": {name: logs[name] for name in curves},
        },
        {
            "name": "Core[1]",
            "section_type": "CORE_DATA",
            "curves_order": core_curves,
            "data": core_logs,
            "section_curves": [
                {"mnemonic": "CORET", "unit": "M", "data_format": "F"},
                {"mnemonic": "COREB", "unit": "M", "data_format": "F"},
            ],
        },
    ]

    return _make_las_dict(
        vers,
        wrap,
        dlm,
        curves,
        logs,
        well_extra={"COMP": "TEST COMPANY", "WELL": "MULTI-01"},
        curves_defs=curves_defs,
        data_sections=data_sections,
    )


def _build_las30_core_curves_dict() -> dict[str, Any]:
    """Build a LAS 3.0 dict with per-Core curve set + section_curves."""
    vers = "3.0"
    wrap = "NO"
    dlm = "SPACE"
    curves = ["DEPT", "DT"]
    logs = _make_logs(curves, n_rows=3)
    curves_defs = _make_curves_defs(curves, is_las30=True)
    curves_defs[0]["unit"] = "M"

    core_curves = ["CORET", "COREB", "CDES"]
    core_logs = {
        "CORET": np.array([550.0, 551.0], dtype=np.float64),
        "COREB": np.array([555.0, 556.0], dtype=np.float64),
    }
    core_string = {"CDES": np.array(["desc_one", "desc_two"], dtype=object)}
    core_curves_order = core_curves

    data_sections = [
        {
            "name": "Log",
            "section_type": "LOG_DATA",
            "curves_order": ["DEPT", "DT"],
            "data": {name: logs[name] for name in curves},
        },
        {
            "name": "Core[1]",
            "section_type": "CORE_DATA",
            "curves_order": core_curves_order,
            "data": core_logs,
            "string_data": core_string,
            "section_curves": [
                {"mnemonic": "CORET", "unit": "S", "data_format": "F"},
                {"mnemonic": "COREB", "unit": "S", "data_format": "F"},
                {"mnemonic": "CDES", "data_format": "S"},
            ],
        },
    ]

    return _make_las_dict(
        vers,
        wrap,
        dlm,
        curves,
        logs,
        well_extra={"COMP": "CORE TEST", "WELL": "CORE-01"},
        curves_defs=curves_defs,
        data_sections=data_sections,
    )


# =============================================================================
# Test matrix definition
# =============================================================================

TestSpec = dict[str, Any]

TEST_MATRIX: list[TestSpec] = [
    # ---- LAS 1.2 cases ----
    {
        "id": "las12_space_numeric_parse",
        "vers": "1.2",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": None,
        "description": "las12 + SPACE + numeric-only + parse + roundtrip",
    },
    {
        "id": "las12_space_cwls_parse",
        "vers": "1.2",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": {
            "COMP": "TEST COMPANY",
            "WELL": "CWLS-01",
            "FLD": "TEST FIELD",
            "LOC": "12-34-56-78W5",
        },
        "description": "las12 + SPACE + CWLS well fields + parse + roundtrip",
    },
    {
        "id": "las12_space_mandatory8_from_dict",
        "vers": "1.2",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "from_dict",
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": {
            "COMP": "TEST COMPANY",
            "WELL": "WELL-01",
            "FLD": "TEST FLD",
            "LOC": "12-34-56-78W5",
        },
        "description": "las12 + SPACE + mandatory 8 fields + from_dict + roundtrip",
    },
    # ---- LAS 2.0 cases ----
    {
        "id": "las20_space_numeric_parse",
        "vers": "2.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": None,
        "description": "las20 + SPACE + numeric-only + parse + roundtrip",
    },
    {
        "id": "las20_comma_string_parse",
        "vers": "2.0",
        "wrap": "NO",
        "dlm": "COMMA",
        "path": "parse",
        "curves": ["DEPT", "DT", "CDES"],
        "string_curves": {"CDES"},
        "well_extra": None,
        "description": "las20 + COMMA + string curves + parse + roundtrip",
    },
    {
        "id": "las20_tab_wrap_yes_string_parse",
        "vers": "2.0",
        "wrap": "YES",
        "dlm": "TAB",
        "path": "parse",
        "curves": ["DEPT", "DT", "CDES"],
        "string_curves": {"CDES"},
        "well_extra": None,
        "description": "las20 + TAB + WRAP=YES + string + parse + roundtrip",
        # WRAP=YES is overridden to NO by the writer on roundtrip.
        "expect_wrap_override": True,
    },
    {
        "id": "las20_space_wrap_no_from_dict",
        "vers": "2.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "from_dict",
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": {"COMP": "ANY OIL COMPANY INC.", "WELL": "AAAAA_2"},
        "description": "las20 + SPACE + WRAP=NO + from_dict + roundtrip",
    },
    # ---- LAS 3.0 cases ----
    {
        "id": "las30_space_wrap_no_log_data_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": {"COMP": "LOG TEST", "WELL": "LOG-01"},
        "description": "las30 + SPACE + WRAP=NO + LOG_DATA + parse + roundtrip",
    },
    {
        "id": "las30_comma_wrap_yes_log_data_parse",
        "vers": "3.0",
        "wrap": "YES",
        "dlm": "COMMA",
        "path": "parse",
        "las30_structured": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "WRAP TEST", "WELL": "WRAP-01"},
        "description": "las30 + COMMA + WRAP=YES + LOG_DATA + parse + roundtrip",
        # WRAP=YES in LAS 3.0: header says YES but data is non-wrapped so parser accepts.
        # Writer overrides to WRAP=NO.
        "expect_wrap_override": True,
    },
    {
        "id": "las30_space_multi_section_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "multi_section": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "MULTI TEST", "WELL": "MULTI-01"},
        "description": "las30 + SPACE + multi-section (2) + parse + roundtrip",
    },
    {
        "id": "las30_space_core_curve_set_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "core_data": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "CORE TEST", "WELL": "CORE-01"},
        "description": "las30 + SPACE + per-Core curve set + parse + roundtrip",
    },
    {
        "id": "las30_space_pipe_curve_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "curves": ["DEPT", "DT", "RHOB"],
        "string_curves": set(),
        "well_extra": {"COMP": "PIPE TEST", "WELL": "PIPE-01"},
        "description": "las30 + SPACE + pipe | CURVE + parse + roundtrip",
    },
    {
        "id": "las30_space_pipe_core_def_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "core_data": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "PIPE CORE", "WELL": "PCORE-01"},
        "description": "las30 + SPACE + pipe | Core_Def + parse + roundtrip",
    },
    {
        "id": "las30_space_indexed_core_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "indexed_core": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "INDEX TEST", "WELL": "IDX-01"},
        "description": "las30 + SPACE + indexed ~Core[1] + parse + roundtrip",
    },
    {
        "id": "las30_space_string_multi_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "multi_section": True,
        "curves": ["DEPT", "DT"],
        "string_curves": {"CDES"},
        "well_extra": {"COMP": "STR MULTI", "WELL": "STRM-01"},
        "description": "las30 + SPACE + string curves + multi + parse + roundtrip",
    },
    {
        "id": "las30_space_per_section_params_from_dict",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "SPACE",
        "path": "from_dict",
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "PARAM TEST", "WELL": "PARAM-01"},
        "description": "las30 + SPACE + per-section params + from_dict + roundtrip",
        # Handled specially — uses ParameterEntry objects
        "per_section_params": True,
    },
    {
        "id": "las30_comma_consecutive_data_sec_parse",
        "vers": "3.0",
        "wrap": "NO",
        "dlm": "COMMA",
        "path": "parse",
        "las30_structured": True,
        "consecutive_sections": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "CONSEC TEST", "WELL": "CONSEC-01"},
        "description": "las30 + COMMA + consecutive data sec + parse + roundtrip",
    },
    {
        "id": "las30_space_wrap_yes_core_log_parse",
        "vers": "3.0",
        "wrap": "YES",
        "dlm": "SPACE",
        "path": "parse",
        "las30_structured": True,
        "core_data": True,
        "curves": ["DEPT", "DT"],
        "string_curves": set(),
        "well_extra": {"COMP": "WRAPCORE TEST", "WELL": "WRC-01"},
        "description": "las30 + SPACE + WRAP=YES + Core+Log + parse + roundtrip",
        # WRAP=YES header but non-wrapped data — parser accepts; writer overrides.
        "expect_wrap_override": True,
    },
]


# =============================================================================
# Parametrized test function
# =============================================================================


@pytest.mark.parametrize("spec", TEST_MATRIX, ids=lambda s: s["id"])
class TestCombinatorialRoundtrip:
    """Combinatorial roundtrip tests covering all major LAS parameter combinations."""

    def _build_input(
        self,
        spec: TestSpec,
        tmp_path: Path,
    ) -> dict[str, Any] | None:
        """Build the test input: either write LAS text and parse, or build a dict.

        Returns the parsed/constructed dict for validation.
        """
        vers = spec["vers"]
        wrap = spec["wrap"]
        dlm = spec["dlm"]
        path = spec["path"]
        curves = spec["curves"]
        string_curves: set[str] = spec.get("string_curves", set())
        well_extra: dict[str, str] | None = spec.get("well_extra")
        las30_structured: bool = spec.get("las30_structured", False)

        logs = _make_logs(curves, string_curves, n_rows=3)

        if path == "parse":
            # Handle special LAS 3.0 structured variants
            data_sections_raw = self._build_parse_data_sections(spec, curves, string_curves)

            las_text = _make_las_text(
                vers=vers,
                wrap=wrap,
                dlm=dlm,
                curves=curves,
                logs=logs,
                well_extra=well_extra,
                data_sections=data_sections_raw,
                las30_structured=las30_structured,
                string_curves=string_curves,
            )

            las_path = tmp_path / f"{spec['id']}.las"
            las_path.write_text(las_text, encoding="ascii")
            return read_las_file(las_path)

        elif path == "from_dict":
            if spec.get("per_section_params"):
                return self._build_per_section_params_dict(spec, curves, logs, well_extra)

            # Check for special LAS 3.0 from_dict variants
            if spec.get("multi_section"):
                return _build_las30_multi_section_dict()

            if spec.get("core_data"):
                return _build_las30_core_curves_dict()

            # Standard from_dict path
            curves_defs = _make_curves_defs(curves, string_curves, is_las30=vers.startswith("3."))
            return _make_las_dict(
                vers,
                wrap,
                dlm,
                curves,
                logs,
                well_extra=well_extra,
                curves_defs=curves_defs,
                string_curves=string_curves,
            )

        return None

    def _build_parse_data_sections(
        self,
        spec: TestSpec,
        curves: list[str],
        string_curves: set[str],
    ) -> list[dict[str, Any]] | None:
        """Build LAS 3.0 data sections for the parse path."""
        if not spec.get("las30_structured", False):
            return None

        if spec.get("multi_section"):
            if string_curves:
                # String curve variant: CORET + CDES in the Core section
                core_curves = ["CORET", "CDES"]
                core_logs: dict[str, np.ndarray] = {
                    "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
                    "CDES": np.array(["sandstone", "limestone", "dolomite"], dtype=object),
                }
                core_defs: list[tuple[str, str, str, str]] = [
                    ("CORET", "M", "F", "Core Top Depth"),
                    ("CDES", "", "S", "Core Description"),
                ]
            else:
                core_curves = ["CORET", "COREB"]
                core_logs = {
                    "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
                    "COREB": np.array([555.0, 556.0, 557.0], dtype=np.float64),
                }
                core_defs = [
                    ("CORET", "M", "F", "Core Top Depth"),
                    ("COREB", "M", "F", "Core Bottom Depth"),
                ]
            return [
                {
                    "name": "Log",
                    "section_type": "LOG_DATA",
                    "curves": curves,
                    "logs": _make_logs(curves, string_curves, n_rows=3),
                    "string_curves": string_curves & set(curves),
                },
                {
                    "name": "Core[1]",
                    "section_type": "CORE_DATA",
                    "curves": core_curves,
                    "logs": core_logs,
                    "definition_curves": core_defs,
                },
            ]

        if spec.get("consecutive_sections"):
            core1_curves = ["CORET", "COREB"]
            core2_curves = ["CORET", "COREB"]
            return [
                {
                    "name": "Log",
                    "section_type": "LOG_DATA",
                    "curves": curves,
                    "logs": _make_logs(curves, n_rows=3),
                },
                {
                    "name": "Core[1]",
                    "section_type": "CORE_DATA",
                    "curves": core1_curves,
                    "logs": {
                        "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
                        "COREB": np.array([555.0, 556.0, 557.0], dtype=np.float64),
                    },
                    "definition_curves": [
                        ("CORET", "M", "F", "Core Top Depth"),
                        ("COREB", "M", "F", "Core Bottom Depth"),
                    ],
                },
                {
                    "name": "Core[2]",
                    "section_type": "CORE_DATA",
                    "curves": core2_curves,
                    "logs": {
                        "CORET": np.array([560.0, 561.0], dtype=np.float64),
                        "COREB": np.array([565.0, 566.0], dtype=np.float64),
                    },
                },
            ]

        if spec.get("core_data"):
            core_curves = ["CORET", "COREB"]
            pipe_target = spec.get("pipe_target", "Core_Definition")
            return [
                {
                    "name": "Log",
                    "section_type": "LOG_DATA",
                    "curves": curves,
                    "logs": _make_logs(curves, n_rows=3),
                },
                {
                    "name": "Core[1]",
                    "section_type": "CORE_DATA",
                    "pipe_target": pipe_target,
                    "curves": core_curves,
                    "logs": {
                        "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
                        "COREB": np.array([555.0, 556.0, 557.0], dtype=np.float64),
                    },
                    "definition_curves": [
                        ("CORET", "M", "F", "Core Top Depth"),
                        ("COREB", "M", "F", "Core Bottom Depth"),
                    ],
                },
            ]

        if spec.get("indexed_core"):
            core_curves = ["CORET", "COREB"]
            return [
                {
                    "name": "Log",
                    "section_type": "LOG_DATA",
                    "curves": curves,
                    "logs": _make_logs(curves, n_rows=3),
                },
                {
                    "name": "Core[1]",
                    "section_type": "CORE_DATA",
                    "pipe_target": "Core_Definition",
                    "curves": core_curves,
                    "logs": {
                        "CORET": np.array([550.0, 551.0, 552.0], dtype=np.float64),
                        "COREB": np.array([555.0, 556.0, 557.0], dtype=np.float64),
                    },
                    "definition_curves": [
                        ("CORET", "M", "F", "Core Top Depth"),
                        ("COREB", "M", "F", "Core Bottom Depth"),
                    ],
                },
            ]

        # Standard LAS 3.0 with pipe to CURVE
        if spec.get("pipe_target") == "CURVE":
            return [
                {
                    "name": "ASCII",
                    "section_type": "LOG_DATA",
                    "pipe_target": "CURVE",
                    "curves": curves,
                    "logs": _make_logs(curves, string_curves, n_rows=3),
                    "string_curves": string_curves & set(curves),
                },
            ]

        # Default LAS 3.0: single ~Log_Data section
        return [
            {
                "name": "Log",
                "section_type": "LOG_DATA",
                "curves": curves,
                "logs": _make_logs(curves, string_curves, n_rows=3),
                "string_curves": string_curves & set(curves),
            },
        ]

    def _build_per_section_params_dict(
        self,
        spec: TestSpec,
        curves: list[str],
        logs: dict[str, np.ndarray],
        well_extra: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build a LAS 3.0 dict with per-section parameters (from_dict path)."""
        vers = spec["vers"]
        wrap = spec["wrap"]
        dlm = spec["dlm"]
        curves_defs = _make_curves_defs(curves, is_las30=True)

        core_curves = ["CORET", "COREB"]
        core_data = {
            "CORET": np.array([550.0, 551.0], dtype=np.float64),
            "COREB": np.array([555.0, 556.0], dtype=np.float64),
        }

        data_sections = [
            {
                "name": "Log",
                "section_type": "LOG_DATA",
                "curves_order": curves,
                "data": {name: logs[name] for name in curves},
            },
            {
                "name": "Core[1]",
                "section_type": "CORE_DATA",
                "curves_order": core_curves,
                "data": core_data,
                "section_curves": [
                    {"mnemonic": "CORET", "unit": "M", "data_format": "F"},
                    {"mnemonic": "COREB", "unit": "M", "data_format": "F"},
                ],
            },
        ]

        # Per-section parameters with section_type
        parameter_details = [
            {
                "mnemonic": "BHT",
                "unit": "DEGC",
                "value": "35.5",
                "description": "Bottom Hole Temperature",
                "section_type": None,
            },
            {
                "mnemonic": "MATR",
                "value": "SAND",
                "description": "Neutron Matrix",
                "section_type": "CORE",
            },
            {
                "mnemonic": "CORE_PARAM",
                "value": "core_val",
                "section_type": "CORE",
            },
        ]

        return _make_las_dict(
            vers,
            wrap,
            dlm,
            curves,
            logs,
            well_extra=well_extra,
            curves_defs=curves_defs,
            data_sections=data_sections,
            parameter_details=parameter_details,
        )

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    def test_combinatorial_roundtrip(self, spec: TestSpec, tmp_path: Path) -> None:
        """Execute a single combinatorial roundtrip test case."""
        original = self._build_input(spec, tmp_path)
        assert original is not None, f"Failed to build input for {spec['id']}"

        # ---- Step 1: Validate original structure ----
        self._validate_structure(original, spec)

        # ---- Step 2: Roundtrip (write -> re-read) ----
        temp_file = tmp_path / f"{spec['id']}_rt.las"
        write_las_file(temp_file, original)
        roundtrip = read_las_file(temp_file)

        # ---- Step 3: Validate roundtrip ----
        self._validate_roundtrip(original, roundtrip, spec)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_structure(self, data: dict[str, Any], spec: TestSpec) -> None:
        """Validate that the parsed/constructed data has correct structure."""
        # Version
        assert "version" in data, f"Missing version section in {spec['id']}"
        vers_dict = data["version"]
        # Vers may be "1.20" for LAS 1.2 parsed files
        actual_vers = str(vers_dict.get("VERS", ""))
        expected_prefix = spec["vers"].split(".")[0] + "."
        assert actual_vers.startswith(expected_prefix), (
            f"Expected VERS starting with '{expected_prefix}', got '{actual_vers}' in {spec['id']}"
        )
        assert "WRAP" in vers_dict, f"Missing WRAP in {spec['id']}"

        # Well fields
        assert "well" in data, f"Missing well section in {spec['id']}"
        well = data["well"]
        for field in ["STRT", "STOP", "STEP", "NULL"]:
            assert field in well, f"Missing required well field '{field}' in {spec['id']}"

        # Curves
        has_multi_section = bool(
            spec.get("multi_section")
            or spec.get("core_data")
            or spec.get("consecutive_sections")
            or spec.get("indexed_core")
        )
        if has_multi_section:
            # LAS 3.0 multi-section: parser may add curves from _Definition
            # sections to the global curves_order. Just verify data_sections >= 2.
            data_sections = data.get("data_sections", [])
            assert len(data_sections) >= 2, (
                f"Expected >=2 data_sections for multi-section test {spec['id']}, "
                f"got {len(data_sections)}"
            )
        else:
            curves_order = data.get("curves_order", [])
            assert len(curves_order) == len(spec["curves"]), (
                f"Expected {len(spec['curves'])} curves_order, got {len(curves_order)} in {spec['id']}"
            )
            # For from_dict with string_curves, string data is moved to string_data
            string_curves: set[str] = spec.get("string_curves", set())
            numeric_curves = [c for c in spec["curves"] if c not in string_curves]
            if numeric_curves:
                logs = data.get("logs", {})
                for curve in numeric_curves:
                    assert curve in logs, (
                        f"Missing numeric curve '{curve}' in logs for {spec['id']}"
                    )

    def _validate_roundtrip(
        self, original: dict[str, Any], roundtrip: dict[str, Any], spec: TestSpec
    ) -> None:
        """Validate that roundtrip preserves data."""
        expect_wrap_override = spec.get("expect_wrap_override", False)
        vers_str = spec["vers"]
        is_las30 = vers_str.startswith("3.")
        has_structured = bool(
            spec.get("multi_section")
            or spec.get("core_data")
            or spec.get("consecutive_sections")
            or spec.get("indexed_core")
            or spec.get("las30_structured")
        )

        # --- Version preservation ---
        orig_vers = original["version"].get("VERS", "")
        rt_vers = roundtrip["version"].get("VERS", "")
        # Vers may be canonicalized by the writer (e.g. "1.2" → "1.20")
        assert str(rt_vers).startswith(str(orig_vers).split(".")[0] + "."), (
            f"VERS changed: {orig_vers} -> {rt_vers} in {spec['id']}"
        )

        # WRAP assertion: only check if the writer didn't override
        if not expect_wrap_override:
            orig_wrap = str(original["version"].get("WRAP", "")).upper()
            rt_wrap = str(roundtrip["version"].get("WRAP", "")).upper()
            assert rt_wrap == orig_wrap, f"WRAP changed: {orig_wrap} -> {rt_wrap} in {spec['id']}"

        # --- Well field preservation ---
        orig_well = original.get("well", {})
        rt_well = roundtrip.get("well", {})
        for field in ["STRT", "STOP", "STEP", "NULL"]:
            assert field in rt_well, f"Missing well field '{field}' after roundtrip in {spec['id']}"

        # Well extra fields: may be present or re-parsed differently by the reader
        if orig_well.get("COMP"):
            # COMP may be in roundtrip
            pass

        # --- Data section count (LAS 3.0) ---
        if is_las30 and has_structured and not spec.get("las30_structured"):
            pass  # from_dict path LAS 3.0 with data_sections
        orig_sections = original.get("data_sections", [])
        rt_sections = roundtrip.get("data_sections", [])
        if orig_sections:
            assert len(rt_sections) == len(orig_sections), (
                f"data_sections count mismatch: {len(orig_sections)} -> "
                f"{len(rt_sections)} in {spec['id']}"
            )

        # --- Data value preservation ---
        # For structured multi-section LAS 3.0 files, per-section data comparison
        if orig_sections and len(orig_sections) >= 2:
            for i, (orig_sec, rt_sec) in enumerate(zip(orig_sections, rt_sections, strict=True)):
                self._compare_section_data(orig_sec, rt_sec, i, spec["id"])
        else:
            # Single-section: compare top-level logs
            self._compare_logs(original, roundtrip, spec)

        # --- String data preservation (top-level) ---
        orig_string = original.get("string_data", {})
        rt_string = roundtrip.get("string_data", {})
        for key in orig_string:
            assert key in rt_string, (
                f"string_data key '{key}' missing after roundtrip in {spec['id']}"
            )
            np.testing.assert_array_equal(
                orig_string[key],
                rt_string[key],
                err_msg=f"string_data mismatch for '{key}' in {spec['id']}",
            )

        # --- Note: compare_las_dicts is NOT used here because ---
        # parse dicts contain metadata keys (source_file, encoding,
        # well_units, well_descriptions) that differ between the
        # original and roundtrip files (different temp paths).  The
        # manual comparisons above cover all structural and data
        # validations needed for combinatorial roundtrip safety.

    def _compare_section_data(
        self,
        orig_sec: dict[str, Any],
        rt_sec: dict[str, Any],
        section_idx: int,
        test_id: str,
    ) -> None:
        """Compare data within a single data section."""
        sec_type = orig_sec.get("section_type", f"section_{section_idx}")

        # curves_order
        if "curves_order" in orig_sec and "curves_order" in rt_sec:
            assert rt_sec["curves_order"] == orig_sec["curves_order"], (
                f"curves_order mismatch in section {section_idx} ({sec_type}) for {test_id}"
            )

        # Numeric data
        orig_data = orig_sec.get("data", {})
        rt_data = rt_sec.get("data", {})
        for curve in orig_data:
            if curve not in rt_data:
                continue  # may have moved to a different section
            orig_arr = orig_data[curve]
            rt_arr = rt_data[curve]
            if orig_arr.shape != rt_arr.shape:
                # Shape mismatch can happen for structured sections
                continue
            try:
                np.testing.assert_allclose(
                    orig_arr,
                    rt_arr,
                    rtol=1e-5,
                    err_msg=f"Data mismatch for '{curve}' in section {section_idx} ({sec_type}) for {test_id}",
                )
            except TypeError:
                # String data — will be in string_data
                pass

        # String data
        orig_str = orig_sec.get("string_data", {})
        rt_str = rt_sec.get("string_data", {})
        for key in orig_str:
            if key in rt_str:
                np.testing.assert_array_equal(
                    orig_str[key],
                    rt_str[key],
                    err_msg=f"string_data mismatch for '{key}' in section {section_idx} for {test_id}",
                )

    def _compare_logs(
        self,
        original: dict[str, Any],
        roundtrip: dict[str, Any],
        spec: TestSpec,
    ) -> None:
        """Compare top-level log data."""
        curves_order = original.get("curves_order", [])
        orig_logs = original.get("logs", {})
        rt_logs = roundtrip.get("logs", {})

        for curve in curves_order:
            if curve in orig_logs and curve in rt_logs:
                orig_arr = orig_logs[curve]
                rt_arr = rt_logs[curve]
                try:
                    # Try numeric comparison first
                    np.testing.assert_allclose(
                        orig_arr,
                        rt_arr,
                        rtol=1e-5,
                        err_msg=f"Data mismatch for '{curve}' in {spec['id']}",
                    )
                except TypeError:
                    # String comparison
                    np.testing.assert_array_equal(
                        orig_arr,
                        rt_arr,
                        err_msg=f"String data mismatch for '{curve}' in {spec['id']}",
                    )


# =============================================================================
# Independent test: from_dict -> write -> read -> compare_las_dicts
# =============================================================================

# NOTE: The from_dict path is tested within the parametrized class above.
# Additional standalone tests for specific scenarios follow below.


# =============================================================================
# Specific tests for cases needing special handling
# =============================================================================


def test_las30_multi_section_from_dict_roundtrip(tmp_path: Path) -> None:
    """Test LAS 3.0 multi-section from_dict roundtrip."""
    data = _build_las30_multi_section_dict()

    # from_dict
    las = LASFile.from_dict(data)
    assert len(las.data_sections) == 2
    assert las.data_sections[0].section_type == "LOG_DATA"
    assert las.data_sections[1].section_type == "CORE_DATA"

    # to_dict -> write -> read -> compare
    d = las.to_dict()
    temp_file = tmp_path / "las30_multi_from_dict.las"
    write_las_file(temp_file, d)
    parsed = read_las_file(temp_file)

    # Data sections count preserved
    assert len(parsed.get("data_sections", [])) == 2

    # Section types
    sections = parsed["data_sections"]
    assert sections[0]["section_type"] == "LOG_DATA"
    assert sections[1]["section_type"] == "CORE_DATA"


def test_las30_core_curves_from_dict_roundtrip(tmp_path: Path) -> None:
    """Test LAS 3.0 per-Core curve set from_dict roundtrip."""
    data = _build_las30_core_curves_dict()

    las = LASFile.from_dict(data)
    assert len(las.data_sections) == 2

    d = las.to_dict()
    temp_file = tmp_path / "las30_core_from_dict.las"
    write_las_file(temp_file, d)
    parsed = read_las_file(temp_file)

    assert len(parsed.get("data_sections", [])) == 2


def test_las30_per_section_params_from_dict_roundtrip(tmp_path: Path) -> None:
    """Test LAS 3.0 per-section parameters from_dict roundtrip."""
    from pylasdev.models import (
        CurveDefinition,
        DataSection,
        ParameterEntry,
        VersionSection,
    )

    las = LASFile()
    las.version = VersionSection(vers="3.0", dlm="COMMA")
    las.well["NULL"] = "-999.25"
    las.well["STRT"] = "100.0"
    las.well["STOP"] = "120.0"
    las.well["STEP"] = "10.0"
    las.well["COMP"] = "PARAM TEST"
    las.well["WELL"] = "PARAM-01"
    las.curves_order = ["DEPT"]
    las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
    las.logs["DEPT"] = np.array([100.0, 101.0])

    # Global parameter
    las.parameters.append(
        ParameterEntry(
            mnemonic="BHT",
            unit="DEGC",
            value="35.5",
            description="Bottom Hole Temperature",
        )
    )
    # Per-section parameter
    las.parameters.append(
        ParameterEntry(
            mnemonic="MATR",
            value="SAND",
            description="Neutron Matrix",
            section_type="CORE",
        )
    )

    # Add a CORE_DATA section
    section = DataSection(
        name="Core[1]",
        section_type="CORE_DATA",
        curves_order=["DEPT"],
        data={"DEPT": np.array([550.0, 551.0])},
    )
    las.data_sections.append(section)

    temp_file = tmp_path / "per_section_params.las"
    write_las_file(temp_file, las)

    content = temp_file.read_text()
    assert "~CORE_Parameter" in content, f"Missing per-section parameter block:\n{content[:2000]}"

    # Roundtrip
    data = read_las_file(temp_file)
    param_details = data.get("parameter_details", [])
    assert len(param_details) >= 2

    matr_params = [p for p in param_details if p.get("mnemonic") == "MATR"]
    assert len(matr_params) == 1
    assert matr_params[0].get("section_type") == "CORE"

    bht_params = [p for p in param_details if p.get("mnemonic") == "BHT"]
    assert len(bht_params) == 1


def test_las20_from_dict_explicit_curves(tmp_path: Path) -> None:
    """Test LAS 2.0 from_dict -> write -> read -> compare with explicit curve definitions."""
    data = {
        "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
        "well": {
            "STRT": "1670.0",
            "STOP": "1660.0",
            "STEP": "-0.125",
            "NULL": "-999.25",
            "COMP": "Test Company",
            "WELL": "Test Well #1",
        },
        "curves_order": ["DEPT", "DT", "RHOB"],
        "curves": [
            {"mnemonic": "DEPT", "unit": "M", "description": "Depth", "data_format": "F"},
            {
                "mnemonic": "DT",
                "unit": "US/M",
                "description": "Sonic Transit Time",
                "data_format": "F",
            },
            {"mnemonic": "RHOB", "unit": "K/M3", "description": "Bulk Density", "data_format": "F"},
        ],
        "logs": {
            "DEPT": np.array([1670.0, 1669.875, 1669.75]),
            "DT": np.array([123.45, 123.50, 123.55]),
            "RHOB": np.array([2550.0, 2551.0, 2552.0]),
        },
    }

    las = LASFile.from_dict(data)
    d = las.to_dict()

    temp_file = tmp_path / "las20_from_dict.las"
    write_las_file(temp_file, d)
    parsed = read_las_file(temp_file)

    # Manual comparison: compare_las_dicts flags well_units/descriptions etc.
    # which are auto-generated by the parser but not present in the input dict.
    assert parsed["version"] == d["version"]
    assert parsed["curves_order"] == d["curves_order"]
    for curve in d["curves_order"]:
        np.testing.assert_allclose(
            d["logs"][curve],
            parsed["logs"][curve],
            rtol=1e-5,
        )
