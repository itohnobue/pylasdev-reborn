"""Tests for LAS format compliance of written output.

T10/G-13: Golden-file / format compliance tests verifying that written
output conforms to LAS specification requirements:
  - No exponent notation in data sections
  - Correct LAS 1.2 colon placement (numeric fields: value BEFORE colon)
  - Data values are space-delimited in non-LAS30 files
  - Version section has correct format
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from pylasdev import read_las_file, write_las_file
from pylasdev.models import (
    CurveDefinition,
    LASFile,
    VersionSection,
)


class TestFormatCompliance:
    """T10/G-13: LAS format compliance tests for written output."""

    # --- LAS 1.2 compliance ---
    def test_las12_no_exponents_in_data_section(self, tmp_path: Path) -> None:
        """Verify that LAS 1.2 written output has no exponent notation in data.

        CWLS LAS 1.2 §5.6: All data values must be plain decimal numbers.
        Exponent-formatted numbers (e.g., '1e+08') are explicitly forbidden.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "1.2", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": "100000.0",
                "STOP": "100002.0",
                "STEP": "1.0",
                "NULL": "-999.25",
                "COMP": "TestCo",
                "WELL": "Well1",
            },
            "parameters": {},
            "logs": {
                "DEPT": np.array([100000.0, 100001.0, 100002.0]),
                "DT": np.array([123.45, 123.50, 123.55]),
            },
            "curves_order": ["DEPT", "DT"],
        }
        temp_file = tmp_path / "las12_compliance.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        # Extract data section (after ~A header)
        data_section = content.split("~A")[-1]
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        # Data lines (after header) must not contain exponent notation
        for line in data_lines[1:]:  # Skip header line
            assert not re.search(r"[0-9][eE][+\-]?[0-9]", line), (
                f"Exponent notation found in data line: {line!r}"
            )

    def test_las12_well_section_colon_placement(self, tmp_path: Path) -> None:
        """Verify LAS 1.2 well section has correct colon placement.

        CWLS LAS 1.2 §3.2: Numeric well fields (STRT, STOP, STEP, NULL) have
        value BEFORE the colon.  Non-numeric fields use the lasio convention
        (value AFTER colon) for backward compatibility.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "1.2", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": "1670.0",
                "STOP": "1660.0",
                "STEP": "-0.125",
                "NULL": "-999.25",
                "COMP": "TestCo",
            },
            "parameters": {},
            "logs": {"DEPT": np.array([1670.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "las12_well_fmt.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        assert "~WELL" in content, "Missing ~WELL section header"

        # Numeric fields (STRT, STOP, STEP, NULL) must have value BEFORE colon.
        # Correct output:  STRT.M   1670.0  :
        # Wrong output:    STRT.M    : 1670.0
        # The regex requires: KEY.UNIT VALUECHARS SPACES COLON
        for numeric_key in ("STRT", "STOP", "STEP", "NULL"):
            assert re.search(
                rf"{numeric_key}\.\s+\S+\s+:",
                content,
            ), (
                f"LAS 1.2 numeric field {numeric_key}: value must appear "
                f"BEFORE the colon on the same line"
            )

        # Non-numeric field COMP uses lasio convention (value AFTER colon).
        # The regex requires: KEY.UNIT SPACES COLON SPACES VALUECHARS
        assert re.search(r"COMP\.\s+:.*TestCo", content), (
            "LAS 1.2 non-numeric field COMP: value must appear AFTER the colon"
        )

    # --- LAS 2.0 compliance ---
    def test_las20_version_section_format(self, tmp_path: Path) -> None:
        """Verify LAS 2.0 version section has correct format.

        CWLS LAS 2.0 §2.0: VERS value BEFORE colon. WRAP must be YES or NO.
        DLM line is optional (SPACE is default).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {"DEPT": np.array([100.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "las20_fmt.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()

        # 1. Section header ~VERSION must appear on its own line
        assert re.search(r"^~VERSION", content, re.MULTILINE), "Missing ~VERSION section header"

        # 2. VERS line: value ("2.0") must appear BEFORE the colon.
        #    Correct:  VERS.   2.0  : CWLS LOG ASCII STANDARD
        #    Malformed: VERS.   : 2.0
        vers_match = re.search(r"VERS\.\s+([\d.]+)\s+:", content)
        assert vers_match is not None, "VERS line malformed: value must appear BEFORE the colon"
        assert vers_match.group(1) == "2.0", (
            f"VERS value should be 2.0, got {vers_match.group(1)!r}"
        )

        # 3. WRAP line: value (YES/NO) must appear BEFORE the colon.
        #    Correct:  WRAP.   NO  : ONE LINE PER DEPTH STEP
        #    Malformed: WRAP.   : NO
        wrap_match = re.search(r"WRAP\.\s+(YES|NO)\s+:", content)
        assert wrap_match is not None, (
            "WRAP line malformed: value (YES/NO) must appear BEFORE the colon"
        )

        # 4. Description text still appears (after the colon)
        assert "CWLS LOG ASCII STANDARD" in content

    def test_las20_data_values_are_space_delimited(self, tmp_path: Path) -> None:
        """Verify LAS 2.0 data section uses space delimiter.

        CWLS LAS 2.0 §5.2: Data values are space-delimited; each depth
        step on a single line when WRAP=NO.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([50.0, 51.0]),
                "GR": np.array([75.0, 76.0]),
            },
            "curves_order": ["DEPT", "DT", "GR"],
        }
        temp_file = tmp_path / "las20_data_fmt.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        data_section = content.split("~A")[-1]
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        # Data lines (after curve header) should have 3 space-separated values
        for line in data_lines[1:]:  # Skip header
            parts = line.split()
            assert len(parts) >= 3, f"Expected 3 values in data line, got {len(parts)}: {line!r}"

    def test_las20_curve_section_has_units(self, tmp_path: Path) -> None:
        """Verify LAS 2.0 curve section preserves units."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        temp_file = tmp_path / "las20_units.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "DEPT.M" in content
        assert "DT.US/M" in content

    # --- LAS 3.0 compliance ---
    def test_las30_version_section_format(self, tmp_path: Path) -> None:
        """Verify LAS 3.0 version section has correct format with DLM field.

        CWLS LAS 3.0 §3.1: VERSION key is "3.0" with description "CWLS LOG
        ASCII STANDARD -VERSION 3.0". DLM field is required (SPACE / COMMA / TAB).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30_v_fmt.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "VERSION 3.0" in content
        assert "DLM" in content
        assert "COMMA" in content

    def test_las30_no_exponents_in_data(self, tmp_path: Path) -> None:
        """Verify LAS 3.0 output has no exponent notation in data sections."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"))
        las.logs["DEPT"] = np.array([100000.0, 100001.0])
        las.logs["DT"] = np.array([123.45, 123.55])

        temp_file = tmp_path / "las30_noexp.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Find data section and verify no exponent notation
        for section_header in ["~A", "~ASCII", "~LOG_DATA"]:
            if section_header in content:
                data_section = content.split(section_header)[-1]
                break
        else:
            data_section = content
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        for line in data_lines[1:]:
            assert not re.search(r"[0-9][eE][+\-]?[0-9]", line), (
                f"Exponent notation found in LAS 3.0 data: {line!r}"
            )

    def test_las20_no_exponents_in_data_section(self, tmp_path: Path) -> None:
        """Verify LAS 2.0 written output has no exponent notation in data."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {
                "DEPT": np.array([100000.0, 100001.0]),
                "DT": np.array([123.45, 123.55]),
            },
            "curves_order": ["DEPT", "DT"],
        }
        temp_file = tmp_path / "las20_noexp.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        data_section = content.split("~A")[-1]
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        for line in data_lines[1:]:
            assert not re.search(r"[0-9][eE][+\-]?[0-9]", line), (
                f"Exponent notation found in LAS 2.0 data: {line!r}"
            )

    # --- Roundtrip format preservation ---
    def test_roundtrip_preserves_section_order(self, tmp_path: Path) -> None:
        """Verify that roundtrip preserves section order: V→W→C→P→O→A."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25", "STRT": "100.0"},
            "parameters": {"BHT": "35.5"},
            "logs": {"DEPT": np.array([100.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "section_order.las"
        write_las_file(temp_file, data)
        content = temp_file.read_text()

        # Section headers should appear in correct order
        v_pos = content.find("~VERSION")
        w_pos = content.find("~WELL")
        c_pos = content.find("~CURVE")
        p_pos = content.find("~PARAMETER")
        a_pos = content.find("~A")

        assert v_pos >= 0, "Missing ~VERSION section"
        assert w_pos >= 0, "Missing ~WELL section"
        assert c_pos >= 0, "Missing ~CURVE section"
        assert p_pos >= 0, "Missing ~PARAMETER section"
        assert a_pos >= 0, "Missing ~A section"

        # Verify order: V < W < C < P < A
        assert v_pos < w_pos < c_pos < p_pos < a_pos, (
            f"Section order violated: V={v_pos} W={w_pos} C={c_pos} P={p_pos} A={a_pos}"
        )

    # --- F-M-25: LAS 1.2 256-char line-length compliance ---
    def test_las12_long_data_lines_do_not_crash(self, tmp_path: Path) -> None:
        """Test LAS 1.2 output with data lines exceeding 256 characters.

        The LAS 1.2 spec allows a maximum of 256 characters per line in
        unwrapped mode (each line = one depth step). This test verifies
        that the library does not crash or silently corrupt data when
        data lines exceed this limit.

        NOTE: The library currently does NOT enforce the 256-char limit.
        Long lines are written as-is and read back correctly. This test
        documents the current behavior (no enforcement).
        """
        # Create 30 curves — each formatted value is ~10 chars, so 30
        # columns produce lines well over 256 characters.
        curve_names = [f"C{i:02d}" for i in range(30)]
        logs: dict[str, np.ndarray] = {}
        for i, name in enumerate(curve_names):
            logs[name] = np.array([123456.789 + i * 10, 234567.890 + i * 10], dtype=np.float64)

        data: dict[str, Any] = {
            "version": {"VERS": "1.2", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": "0.0",
                "STOP": "1.0",
                "STEP": "1.0",
                "NULL": "-999.25",
                "COMP": "TestCo",
                "WELL": "Well1",
            },
            "parameters": {},
            "logs": logs,
            "curves_order": curve_names,
        }
        temp_file = tmp_path / "las12_long_lines.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        # Extract data section after ~A
        data_section = content.split("~A")[-1]
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        # Skip header line (curve names)
        data_lines = data_lines[1:]

        # All data lines should be present
        assert len(data_lines) == 2, f"Expected 2 data lines, got {len(data_lines)}"

        # Verify at least one line exceeds 256 characters
        long_lines = [line for line in data_lines if len(line) > 256]
        assert len(long_lines) > 0, (
            f"No data line exceeded 256 chars; max was {max(len(ln) for ln in data_lines)}"
        )

        # Verify roundtrip: re-read and check data is intact
        reread = read_las_file(temp_file)
        assert reread["curves_order"] == curve_names
        for name in curve_names:
            np.testing.assert_allclose(
                reread["logs"][name],
                logs[name],
                rtol=1e-4,
                err_msg=f"Data mismatch for {name} after long-line roundtrip",
            )

    # --- F-M-27: LAS 3.0 WRAP=NO enforcement ---
    def test_las30_wrap_yes_emits_warning(self, tmp_path: Path) -> None:
        """Test that writing LAS 3.0 with WRAP=YES emits a warning.

        The LAS 3.0 specification states that WRAP must be NO
        (non-wrapped mode is mandatory in LAS 3.0). This test verifies
        that the library warns when WRAP=YES is used with LAS 3.0.

        NOTE: The library currently emits a warning but does NOT raise
        LASWriteError. VersionSection accepts any wrap value silently.
        This test documents the current behavior (warning-only).
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="YES", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30_wrap_yes.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            wrap_warnings = [x for x in w if "WRAP=YES" in str(x.message)]
            assert len(wrap_warnings) >= 1, (
                "Expected warning about WRAP=YES in LAS 3.0, but none was emitted"
            )

        # Verify the file was written despite the warning
        content = temp_file.read_text()
        assert "~VERSION" in content
        assert "3.0" in content
        # The WRAP value in output must be overridden to NO (F-01 fix:
        # writer cannot produce wrapped output, so header must match data)
        assert "WRAP.   NO" in content
        assert "WRAP.   YES" not in content
        # Data should be written in non-wrapped format (one line per depth step)
        assert "100" in content

    def test_las30_wrap_no_silent(self, tmp_path: Path) -> None:
        """Test that LAS 3.0 with WRAP=NO is accepted silently (no warnings).

        This is the correct/expected LAS 3.0 configuration per spec.
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["DT"] = np.array([50.0, 51.0])

        temp_file = tmp_path / "las30_wrap_no.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            wrap_warnings = [x for x in w if "WRAP=YES" in str(x.message)]
            assert len(wrap_warnings) == 0, "Unexpected WRAP=YES warning for LAS 3.0 with WRAP=NO"

        content = temp_file.read_text()
        assert "WRAP.   NO" in content
        assert "~VERSION 3.0" in content or "VERSION 3.0" in content
