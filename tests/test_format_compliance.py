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

from pylasdev import write_las_file
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

        LAS spec explicitly forbids exponent-formatted numbers (e.g., '1e+08')
        in data sections. All data values must be plain decimal.
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
            line for line in data_section.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        # Data lines (after header) must not contain exponent notation
        for line in data_lines[1:]:  # Skip header line
            assert not re.search(r"[0-9][eE][+\-]?[0-9]", line), (
                f"Exponent notation found in data line: {line!r}"
            )

    def test_las12_well_section_colon_placement(self, tmp_path: Path) -> None:
        """Verify LAS 1.2 well section has correct colon placement.

        LAS 1.2 spec: numeric fields (STRT, STOP, STEP, NULL) have
        value BEFORE the colon.  Non-numeric fields (COMP) use the
        lasio convention: value AFTER the colon.
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
        """Verify LAS 2.0 version section has correct format."""
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
        assert re.search(r"^~VERSION", content, re.MULTILINE), (
            "Missing ~VERSION section header"
        )

        # 2. VERS line: value ("2.0") must appear BEFORE the colon.
        #    Correct:  VERS.   2.0  : CWLS LOG ASCII STANDARD
        #    Malformed: VERS.   : 2.0
        vers_match = re.search(r"VERS\.\s+([\d.]+)\s+:", content)
        assert vers_match is not None, (
            "VERS line malformed: value must appear BEFORE the colon"
        )
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
        """Verify LAS 2.0 data section uses space delimiter."""
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
            line for line in data_section.splitlines()
            if line.strip() and not line.startswith("#")
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
        """Verify LAS 3.0 version section has correct format with DLM field."""
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
            line for line in data_section.splitlines()
            if line.strip() and not line.startswith("#")
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
            line for line in data_section.splitlines()
            if line.strip() and not line.startswith("#")
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
