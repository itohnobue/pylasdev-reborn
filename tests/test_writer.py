"""Tests for LAS file writer."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import read_las_file, read_las_file_as_object, write_las_file
from pylasdev.exceptions import LASWriteError
from pylasdev.models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    LASFile,
    ParameterEntry,
    ParameterZone,
    VersionSection,
)
from pylasdev.writer import (
    _escape_colons_for_las_value,
    _format_data_rows,
    _format_fixed_precision,
    _format_number,
    _sanitize_las_value,
    _section_type_to_prefix,
    _validate_precision,
)


class TestWriteLASFile:
    """Tests for write_las_file function."""

    def test_write_from_dict(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test writing from a dictionary."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        assert temp_file.exists()
        content = temp_file.read_text()
        assert "~VERSION" in content
        assert "~WELL" in content
        assert "~CURVE" in content

    def test_write_from_las_file_object(self, tmp_path: Path) -> None:
        """Test writing from a LASFile dataclass."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "200.0"
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0, 101.0, 102.0])
        las.logs["DT"] = np.array([50.0, 51.0, 52.0])

        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "VERS" in content
        assert "DEPT" in content
        assert "DT" in content
        assert "100" in content

    def test_write_preserves_version(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that version info is preserved in output."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        content = temp_file.read_text()
        assert "2.0" in content

    def test_write_well_info(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that well info entries are written."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        content = temp_file.read_text()
        assert "STRT" in content
        assert "STOP" in content
        assert "COMP" in content
        assert "Test Company" in content

    def test_write_curve_names(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that curve names appear in curve section."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        content = temp_file.read_text()
        assert "DEPT" in content
        assert "DT" in content
        assert "RHOB" in content

    def test_write_parameters(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that parameters are written."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        content = temp_file.read_text()
        assert "~PARAMETER" in content
        assert "BHT" in content
        assert "35.5" in content

    def test_write_ascii_data(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that ASCII data section is present."""
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, sample_las_data)

        content = temp_file.read_text()
        assert "~A" in content
        # F-I2-M46: Verify data values appear in the data section specifically,
        # not just anywhere in the file.  The well section also contains "1670"
        # (STRT value), so a whole-file substring match creates false confidence
        # if the data section is empty.
        data_section = content.split("~A", 1)[1]
        data_lines = [ln for ln in data_section.splitlines() if ln.strip()]
        assert len(data_lines) >= 4  # 1 header + 3 data rows
        # Data section must contain the first DEPT value (1670.0 → "1670")
        assert "1670" in data_section
        # Verify the actual data-row values — the previous OR clause
        # ("1669.875" in data_section or "1670" in data_lines[1]) was
        # OR-masked: data_lines[1] is the FIRST data row, which ALWAYS
        # contains "1670", so corrupting any other value (1669.875 →
        # 1669.000) still passed.  Assert each DEPT value in its own row.
        assert "1670" in data_lines[1]
        assert "1669.875" in data_lines[2]

    def test_write_read_roundtrip(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that write then read produces equivalent data."""
        temp_file = tmp_path / "roundtrip.las"
        write_las_file(temp_file, sample_las_data)

        reread = read_las_file(temp_file)
        assert reread["version"]["VERS"] == "2.0"
        assert reread["curves_order"] == sample_las_data["curves_order"]
        for curve in sample_las_data["curves_order"]:
            np.testing.assert_array_almost_equal(
                reread["logs"][curve],
                sample_las_data["logs"][curve],
                decimal=6,
            )

    def test_write_empty_data(self, tmp_path: Path) -> None:
        """Test writing with no log data."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {},
            "parameters": {},
            "logs": {},
            "curves_order": [],
        }
        temp_file = tmp_path / "empty.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        assert "~VERSION" in content
        assert "~WELL" in content

    def test_write_error_on_bad_path(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test LASWriteError on invalid path.

        The parent directory is made un-creatable on ALL platforms by
        placing a REGULAR FILE where the parent directory would be —
        ``mkdir()`` then raises ``NotADirectoryError`` (an OSError)
        regardless of user or root privileges, so the test does not
        depend on ``/`` being read-only (Linux-root CI can create
        ``/nonexistent``).
        """
        blocker = tmp_path / "afile"
        blocker.write_text("not a directory")
        with pytest.raises(LASWriteError):
            write_las_file(blocker / "sub" / "out.las", sample_las_data)

    def test_write_preserves_curve_units(self, tmp_path: Path) -> None:
        """Test that curve units are preserved in output."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        temp_file = tmp_path / "units.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "DEPT.M" in content
        assert "DT.US/M" in content

    def test_write_other_section(self, tmp_path: Path) -> None:
        """Test that ~O (other) section is written when present."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.other = "Free form text line 1.\nFree form text line 2.\n"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "other.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "~OTHER" in content
        assert "Free form text line 1." in content
        assert "Free form text line 2." in content

    def test_write_las30_version(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 version with DLM field."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "3.0" in content
        assert "DLM" in content
        assert "COMMA" in content
        assert "VERSION 3.0" in content

    # --- F-M-26: DLM=TAB writer test ---
    def test_write_las30_with_dlm_tab(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 with DLM=TAB produces tab-delimited data.

        The writer supports three DLM values: SPACE, COMMA, and TAB.
        This test verifies that DLM=TAB produces actual tab characters
        ("\\t") as separators in the data section.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="TAB")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"))
        las.logs["DEPT"] = np.array([100.0, 101.0, 102.0])
        las.logs["DT"] = np.array([50.0, 51.0, 52.0])
        las.logs["GR"] = np.array([75.0, 76.0, 77.0])

        temp_file = tmp_path / "las30_tab.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # DLM should be present in version section
        assert "DLM" in content
        assert "TAB" in content

        # Find the data section after ~A (or ~ASCII/~LOG_DATA)
        data_section = ""
        for header in ("~ASCII", "~LOG_DATA", "~A"):
            if header in content:
                data_section = content.split(header, 1)[1]
                break
        assert data_section != "", "No data section header found"

        # Get data lines (skip header line with curve names)
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        # Skip the first line (curve names header)
        data_lines = data_lines[1:]

        assert len(data_lines) >= 3, f"Expected >=3 data lines, got {len(data_lines)}"

        # Each data line must use tab characters as separators
        for line in data_lines:
            assert "\t" in line, f"Expected tab-separated data line, got: {line!r}"
            # The tab separator must produce the correct number of columns
            parts = line.split("\t")
            assert len(parts) == 3, f"Expected 3 tab-separated values, got {len(parts)}: {line!r}"

        # Verify roundtrip: re-read and check data values
        reread = read_las_file(temp_file)
        np.testing.assert_allclose(reread["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(reread["logs"]["DT"], [50.0, 51.0, 52.0])
        np.testing.assert_allclose(reread["logs"]["GR"], [75.0, 76.0, 77.0])

    # --- F-057: DLM is NOT emitted for LAS 1.2 files ---

    def test_write_las12_with_dlm_comma_suppressed(self, tmp_path: Path) -> None:
        """DLM line is NOT emitted for LAS 1.2 even when DLM=COMMA (F-057 fix).

        LAS 1.2 does not use DLM; the delimiter is always SPACE.
        The writer.py:176 guard suppresses DLM output for LAS 1.2.
        """
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las12_dlm_comma.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # DLM line must NOT appear in the output for LAS 1.2
        assert " DLM " not in content, f"DLM line unexpectedly emitted for LAS 1.2:\n{content}"

    def test_write_las30_format_specifiers(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 curve format specifiers {F}, {S}."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH", data_format="F")
        )
        las.curves.append(
            CurveDefinition(mnemonic="CDES", unit="", description="CORE DESC", data_format="S")
        )
        las.logs["DEPT"] = np.array([100.0])
        las.string_data["CDES"] = np.array(["SANDSTONE"], dtype=np.str_)

        temp_file = tmp_path / "las30_fmt.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "{F}" in content
        assert "{S}" in content

    def test_write_las30_array_notation(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 array curves with time offsets."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "NMR[1]"]
        las.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH", data_format="F")
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR[1]",
                unit="ms",
                description="NMR Echo",
                data_format="A",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0),
            )
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["NMR[1]"] = np.array([10.0])

        temp_file = tmp_path / "las30_arr.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Time offset 0.0 is formatted as int 0 when it's a whole number
        assert "{A:0}" in content
        assert "NMR[1]" in content

    # --- T9: Fractional time_offset (writer.py:121) ---
    def test_write_las30_fractional_time_offset(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 array curve with fractional time_offset.

        Exercises writer.py:120-121 — the format string path for non-integer
        time_offset values (e.g., 5.5).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "NMR[1]"]
        las.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH", data_format="F")
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR[1]",
                unit="ms",
                description="NMR Echo",
                data_format="A",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=5.5),
            )
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["NMR[1]"] = np.array([10.0])

        temp_file = tmp_path / "las30_frac_offset.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Fractional time_offset 5.5 should appear as float in format
        assert "{A:5.5}" in content
        assert "NMR[1]" in content

    def test_write_las30_zone_association(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 parameter zone associations."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.parameters.append(
            ParameterEntry(
                mnemonic="MATR",
                unit="",
                value="SAND",
                description="Neutron Matrix",
                zone=ParameterZone(zone_name="RUN", zone_index=1),
            )
        )

        temp_file = tmp_path / "las30_zone.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "MATR" in content
        assert "SAND" in content
        assert "| RUN[1]" in content

    def test_write_las30_data_sections(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 with explicit data_sections."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "DT"],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([50.0, 51.0]),
            },
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "las30_data.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "~A CURVE" in content
        assert "100" in content
        assert "50" in content

    def test_write_las30_string_data(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 string data in data_sections."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="CDES", unit="", data_format="S"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "CDES"],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "CDES": np.array([0.0, 0.0]),
            },
        )
        las.data_sections.append(section)
        section.string_data["CDES"] = np.array(["LIMESTONE", "DOLOMITE"], dtype=np.str_)

        temp_file = tmp_path / "las30_str.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "LIMESTONE" in content
        assert "DOLOMITE" in content

    def test_write_non_numeric_null(self, tmp_path: Path) -> None:
        """Test that non-numeric NULL value falls back to -999.25."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "NONE"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        # F-I2-M47: Include a NaN value to verify that _get_null_value()'s
        # fallback to -999.25 is actually exercised.  Without actual null
        # sentinel values in the data, the output only contains clean floats
        # and the null-value path is never tested.
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["DT"] = np.array([50.0, np.nan])

        temp_file = tmp_path / "null_test.las"
        write_las_file(temp_file, las)  # Should not crash

        content = temp_file.read_text()
        assert "100" in content
        # Verify NaN was replaced with the fallback null value (-999.25)
        assert "-999.25" in content
        # Verify roundtrip: NaN position is restored as null_value
        reread = read_las_file(temp_file)
        assert reread["logs"]["DT"][1] == pytest.approx(-999.25)

    # --- TEST-17: LAS 3.0 data_sections with non-numeric NULL value ---
    def test_write_las30_non_numeric_null_in_data_sections(self, tmp_path: Path) -> None:
        """Test LAS 3.0 data_sections path with non-numeric NULL value fallback."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "NOT A NUMBER"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))

        # F-I2-M48: Include a NaN value to verify that the non-numeric NULL
        # fallback (-999.25) is exercised in the LAS 3.0 data_sections path.
        # Without actual null sentinel values, the test only verifies clean
        # floats pass through — the null-value path is never tested.
        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "DT"],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([np.nan, 51.0]),
            },
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "las30_null.las"
        write_las_file(temp_file, las)  # Should not crash

        content = temp_file.read_text()
        assert "100" in content
        assert "101" in content
        # Verify NaN was replaced with the fallback null value (-999.25)
        assert "-999.25" in content
        # Verify roundtrip: NaN position is restored as null_value
        reread = read_las_file(temp_file)
        assert reread["logs"]["DT"][0] == pytest.approx(-999.25)

    # --- F-I2-M49: roundtrip test for #-prefixed value ---
    def test_write_hash_prefix_value_preserved(self, tmp_path: Path) -> None:
        """Value starting with ``#`` is escaped with ``_`` prefix to
        prevent silent data loss on re-read.  The parser treats
        ``#``-prefixed lines as comments, so without the guard at
        writer.py:86-87 the value would be dropped entirely.
        F-007: The parser's _desanitize_las_value reverses this
        transformation, restoring the original ``#``-prefixed value."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "200.0"
        las.well["COMP"] = "#TestCompany"  # Value starting with #
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "hash_prefix.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # The value should appear as _#TestCompany (escaped) — never as a
        # bare # at the start of a line, which the parser would skip.
        assert "_#TestCompany" in content
        # F-007: Verify re-read restores original value
        reread = read_las_file(temp_file)
        assert "#TestCompany" == reread["well"]["COMP"]

    # --- TEST-06: zone_index=None branch (line 125->127) ---
    def test_write_zone_without_index(self, tmp_path: Path) -> None:
        """Test writing LAS 3.0 parameter with zone but no zone_index.

        When zone_index is None, the zone is written as "| ZONENAME"
        without the [N] suffix (exercises writer.py line 125->127).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.parameters.append(
            ParameterEntry(
                mnemonic="TEMP",
                unit="DEGC",
                value="75.0",
                description="Temperature",
                zone=ParameterZone(zone_name="RUN", zone_index=None),
            )
        )

        temp_file = tmp_path / "zone_no_index.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "TEMP" in content
        # Zone should be written without index: | RUN (not | RUN[1])
        assert "| RUN" in content
        assert "RUN[" not in content

    # --- TEST-06: curve_names[0] not in data (line 199/198 guard) ---
    def test_write_curve_names_not_in_data(self, tmp_path: Path) -> None:
        """Test that _format_data_rows returns early when curve_names[0]
        is not present in the data dict.

        This guards against misconfigured data sections where curves_order
        contains names that are not keys in the data dictionary.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))

        # DataSection has curves_order but no data for DEPT
        section = DataSection(
            name="BROKEN",
            curves_order=["DEPT", "DT"],
            data={},  # Empty data — curve_names[0] not in data
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "no_data_keys.las"
        write_las_file(temp_file, las)  # Should not crash

        content = temp_file.read_text()
        assert "~A BROKEN" in content
        # F-I2-M52: Verify that when curve_names[0] is not in the data dict,
        # no data rows are written after the section header.  The section
        # header is emitted unconditionally at writer.py:687, but
        # _format_data_rows returns [] when num_rows == 0 (writer.py:760-761).
        # Without this assertion, the test only proves the header exists —
        # it doesn't verify the guard actually prevented data rows.
        # The header line includes " | CURVE" for LAS 3.0 LOG_DATA sections.
        data_section_after = content.split("~A BROKEN", 1)[1]
        # Skip the first line (rest of the header: " | CURVE")
        remainder = data_section_after.split("\n", 1)[1] if "\n" in data_section_after else ""
        data_rows = [ln for ln in remainder.splitlines() if ln.strip()]
        assert data_rows == [], f"Expected no data rows after ~A BROKEN, got: {data_rows}"

    # --- F-25: Inf values in _format_data_rows ---
    def test_write_inf_values(self, tmp_path: Path) -> None:
        """Test that Inf/-Inf values are replaced with null_value in output.

        Inf values are not valid LAS data; they are serialized as the
        null value (like NaN) to avoid producing "inf" or "-inf" strings
        that parsers cannot handle.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0, 101.0, 102.0])
        # Use float dtype explicitly so inf can be stored
        dt_vals = np.array([50.0, np.inf, -np.inf], dtype=np.float64)
        las.logs["DT"] = dt_vals

        temp_file = tmp_path / "inf_test.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Extract data lines (after ~A header)
        data_lines = content.split("~A")[-1].splitlines()
        # Verify inf values are NOT present in data — replaced with null_value
        assert not any("inf" in line.lower() for line in data_lines)
        # Verify the null value appears in data lines where inf was
        assert any("-999.25" in line for line in data_lines[1:])  # skip header line
        # Verify the normal value (50) still appears
        assert "50" in data_lines[1]
        # Verify roundtrip: re-read and check inf values became null_value
        reread = read_las_file(temp_file)
        assert reread["logs"]["DT"][1] == pytest.approx(-999.25)
        assert reread["logs"]["DT"][2] == pytest.approx(-999.25)

    # --- F-26: Custom precision parameter ---
    def test_write_custom_precision(self, tmp_path: Path) -> None:
        """Test writing with a custom precision format specifier.

        Exercises writer.py:27 — the precision parameter.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([1234.56789, 1235.12345])
        las.logs["DT"] = np.array([123.45, 123.55])

        temp_file = tmp_path / "prec.las"
        write_las_file(temp_file, las, precision=".4g")

        content = temp_file.read_text()
        # With .4g format: format(1234.56789, '.4g') == '1235'
        # Extract data lines after ~A section
        data_section = content.split("~A")[-1]
        data_lines = [line for line in data_section.splitlines() if line.strip()]

        # First line after ~A is the curve names header: "  DEPT  DT"
        # Data lines follow after the header
        assert len(data_lines) >= 2  # header + at least 1 data line
        first_data_line = data_lines[1]  # skip header line

        # Verify full formatted data line contains both values with correct precision
        # For .4g: 1234.56789 → "1235", 123.45 → "123.5"
        parts = first_data_line.split()
        assert len(parts) == 2
        assert parts[0] == "1235"  # 1234.56789 formatted as .4g
        assert parts[1] == "123.5"  # 123.45 formatted as .4g

        # Verify multiple data points
        assert len(data_lines) >= 3
        second_data_line = data_lines[2]
        parts2 = second_data_line.split()
        assert len(parts2) == 2

        # Verify roundtrip preserves data values with the formatted precision
        reread = read_las_file(temp_file)
        assert reread["curves_order"] == ["DEPT", "DT"]
        # Values should roundtrip approximately (precision loss from .4g)
        np.testing.assert_allclose(reread["logs"]["DEPT"][0], 1235.0, rtol=1e-2)
        np.testing.assert_allclose(reread["logs"]["DEPT"][1], 1235.0, rtol=1e-2)

    # --- F-29: Dict encoding key ---
    def test_write_dict_with_encoding_key(self, tmp_path: Path) -> None:
        """Test writing from dict with an 'encoding' key.

        The encoding key is processed by LASFile.from_dict()
        (models.py:5979) and stored on the model.  The writers
        themselves never read las_file.encoding — write_las_file takes
        its own encoding parameter — so the from_dict processing is
        the only behavior this test can honestly pin.
        """
        data = {
            "encoding": "ascii",
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {"DEPT": np.array([1.0, 2.0])},
            "curves_order": ["DEPT"],
        }
        # The model must carry the declared encoding after from_dict.
        las = LASFile.from_dict(data)
        assert las.encoding == "ascii", f"encoding key lost: {las.encoding!r}"

        temp_file = tmp_path / "enc_key.las"
        write_las_file(temp_file, las)

        assert temp_file.exists()
        content = temp_file.read_text()
        assert "~VERSION" in content

    # --- Test: write_las_file rejects invalid input types ---
    def test_write_rejects_invalid_types(self, tmp_path: Path) -> None:
        """Test that write_las_file raises LASWriteError for invalid types.

        Only dict and LASFile are valid; int, str, None, and list should
        all raise LASWriteError with a descriptive message naming the
        received type.
        """
        temp_file = tmp_path / "output.las"

        # int
        with pytest.raises(LASWriteError, match="expects a dict or LASFile, got int"):
            write_las_file(temp_file, 42)

        # str
        with pytest.raises(LASWriteError, match="expects a dict or LASFile, got str"):
            write_las_file(temp_file, "not valid las data")

        # None
        with pytest.raises(LASWriteError, match="expects a dict or LASFile, got NoneType"):
            write_las_file(temp_file, None)

        # list
        with pytest.raises(LASWriteError, match="expects a dict or LASFile, got list"):
            write_las_file(temp_file, [1, 2, 3])

    # --- F25: dict conversion error wrapping (writer.py:86-87) ---
    def test_write_malformed_dict_raises_las_write_error(self, tmp_path: Path) -> None:
        """Test that malformed dict triggers LASWriteError in from_dict path.

        When a dict contains non-numeric values in logs that cannot be
        converted to float64, LASFile.from_dict() raises ValueError,
        which is wrapped in LASWriteError at writer.py:86-87.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO"},
            "well": {},
            "parameters": {},
            "logs": {"DEPT": ["not", "numeric"]},  # non-numeric strings cannot become float64
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "malformed.las"
        with pytest.raises(LASWriteError, match="Cannot create LASFile from dict"):
            write_las_file(temp_file, data)

    # --- F26: _generate_las_content error wrapping (writer.py:98-99) ---
    def test_generate_las_content_error_wrapped(self, tmp_path: Path) -> None:
        """Test that errors during _generate_las_content are wrapped.

        Setting curves to None causes TypeError when _write_curve_section
        tries to iterate, which is caught by the except block at
        writer.py:98-99 and re-raised as LASWriteError.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves = None  # type: ignore[assignment]  # triggers TypeError during iteration
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "broken_gen.las"
        with pytest.raises(LASWriteError, match="Failed to generate LAS file content"):
            write_las_file(temp_file, las)

    def test_generate_las_content_error_las30(self, tmp_path: Path) -> None:
        """Test LAS 3.0 generation error wrapping.

        Setting version.vers to "3.0" with broken curves exercises the
        LAS 3.0 write path through _generate_las_content error wrapping.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves = None  # type: ignore[assignment]
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "broken_las30.las"
        with pytest.raises(LASWriteError, match="Failed to generate LAS file content"):
            write_las_file(temp_file, las)

    # --- T4/G-06: _sanitize_las_value direct unit tests ---
    def test_sanitize_removes_newlines(self) -> None:
        """Test that _sanitize_las_value strips newline characters."""
        assert _sanitize_las_value("hello\nworld") == "hello world"
        assert _sanitize_las_value("hello\r\nworld") == "hello world"
        assert _sanitize_las_value("hello\rworld") == "hello world"

    def test_sanitize_removes_unicode_line_separators(self) -> None:
        """Test that _sanitize_las_value strips Unicode line break characters."""
        assert _sanitize_las_value("hello\u2028world") == "hello world"
        assert _sanitize_las_value("hello\u2029world") == "hello world"
        assert _sanitize_las_value("hello\x85world") == "hello world"

    def test_sanitize_removes_control_characters(self) -> None:
        """Test that _sanitize_las_value strips control characters via regex."""
        # NEL (\x85) is handled by replace above, but also in _CONTROL_CHARS_RE
        # Test other control chars: \x0b (VT), \x0c (FF), \x1c (FS), etc.
        assert _sanitize_las_value("da\x0bta") == "data"
        assert _sanitize_las_value("da\x0cta") == "data"
        assert _sanitize_las_value("da\x1cta") == "data"
        assert _sanitize_las_value("da\x1dta") == "data"
        assert _sanitize_las_value("da\x1eta") == "data"
        assert _sanitize_las_value("da\x7fta") == "data"  # DEL

    def test_sanitize_handles_leading_section_header(self) -> None:
        """Test that _sanitize_las_value removes leading ~[A-Za-z] pattern
        that would mimic a LAS section header."""
        assert _sanitize_las_value("~VERSION broken") == "VERSION broken"
        assert _sanitize_las_value("~A data") == "A data"
        assert _sanitize_las_value("~W text") == "W text"
        # Lowercase section letter also matched
        assert _sanitize_las_value("~a lowercase") == "a lowercase"

    # --- F-I2-M49: # prefix escaping ---
    def test_sanitize_escapes_hash_prefix(self) -> None:
        """_sanitize_las_value prefixes ``#`` with ``_`` to prevent
        comment injection.  The parser skips ``#``-prefixed lines as
        comments in data sections, so a value starting with ``#``
        would be silently dropped on re-read without this guard
        (writer.py:86-87)."""
        assert _sanitize_las_value("#comment") == "_#comment"
        assert _sanitize_las_value("#") == "_#"
        # Values not starting with # are unchanged
        assert _sanitize_las_value("comment #not_start") == "comment #not_start"
        # Leading whitespace before # also escaped (writer.py:95-98)
        assert _sanitize_las_value(" #comment") == " _#comment"
        assert _sanitize_las_value("  #value") == "  _#value"

    # --- F-I2-M51: edge-case tests for _sanitize_las_value ---
    def test_sanitize_empty_string(self) -> None:
        """_sanitize_las_value with empty string returns empty string."""
        assert _sanitize_las_value("") == ""

    def test_sanitize_control_char_only(self) -> None:
        """_sanitize_las_value strips all control characters,
        returning empty string for control-char-only input."""
        assert _sanitize_las_value("\x00\x01\x02") == ""
        assert _sanitize_las_value("\x0b\x0c\x1c") == ""

    def test_sanitize_standalone_section_header(self) -> None:
        """_sanitize_las_value strips leading ~ from standalone
        section-header-like values ("~A", "~V")."""
        assert _sanitize_las_value("~A") == "A"
        assert _sanitize_las_value("~V") == "V"
        assert _sanitize_las_value("~a") == "a"

    def test_sanitize_preserves_clean_text(self) -> None:
        """Test that _sanitize_las_value leaves clean text unchanged."""
        clean = "LAS 2.0 : CWLS LOG ASCII STANDARD"
        assert _sanitize_las_value(clean) == clean

    def test_sanitize_combined_attack_string(self) -> None:
        """Test _sanitize_las_value with combined attack characters."""
        # Newlines + control chars + leading section pattern.
        # Order: replace (\u2028, \n → space), strip ctrl chars (\x0b),
        # then strip leading ~[A-Za-z].
        attack = "~\x0bVERSION\ninfo"  # ~ + ctrl + TEXTVERSION + newline + info
        result = _sanitize_las_value(attack)
        assert "~" not in result  # leading ~ stripped
        assert "\x0b" not in result
        assert "\n" not in result
        # Should produce something like "VERSION info"
        assert "VERSION" in result

    # --- T7/G-10: Writer Definition block dedup ---
    def test_writer_dedups_definition_blocks(self, tmp_path: Path) -> None:
        """Test that two same-type DataSections produce a single Definition block.

        When writing LAS 3.0 with two CORE_DATA sections (e.g., Core[1], Core[2]),
        the writer emits ~Core_Definition only once (per-section curve definitions
        are the same for both sections).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"

        # Main curves for LOG_DATA
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        # Create core curves
        core_curves = [
            CurveDefinition(mnemonic="CORET", unit="M"),
            CurveDefinition(mnemonic="COREB", unit="M"),
            CurveDefinition(mnemonic="CDES", unit="", data_format="S"),
        ]

        # Two CORE_DATA sections with same section_curves
        for section_name in ("Core[1]", "Core[2]"):
            section = DataSection(
                name=section_name,
                section_type="CORE_DATA",
                curves_order=["CORET", "COREB", "CDES"],
                section_curves=list(core_curves),
                data={
                    "CORET": np.array([545.5]),
                    "COREB": np.array([550.6]),
                },
                string_data={
                    "CDES": np.array(["ROCK"]),
                },
            )
            las.data_sections.append(section)

        temp_file = tmp_path / "dedup_def.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # ~Core_Definition should appear exactly once
        assert content.count("~Core_Definition") == 1
        # Both CORE data sections should appear (written as ~CORE_DATA)
        assert content.count("~CORE_DATA Core[1]") >= 1
        assert content.count("~CORE_DATA Core[2]") >= 1

    # --- T11/G-14: WRAP=YES spec violation ---
    def test_wrap_yes_produces_correct_data_layout(self, tmp_path: Path) -> None:
        """Test that when WRAP=YES is in the header, the written data actually
        follows wrapped convention (multiple lines per depth step) OR the
        writer warns and corrects it.

        Currently the writer always produces non-wrapped output (one line per
        depth step). With >=2 curves, the test verifies that the data section
        contains exactly one line per depth step, NOT multiple lines. The
        warning is emitted during write.
        """
        import warnings

        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "YES", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {
                "DEPT": np.array([1.0, 2.0]),
                "DT": np.array([50.0, 51.0]),
                "GR": np.array([75.0, 76.0]),
            },
            "curves_order": ["DEPT", "DT", "GR"],
        }
        temp_file = tmp_path / "wrap_yes_output.las"
        # WRAP=YES should emit a warning since writer always produces non-wrapped
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, data)
            wrap_warnings = [x for x in w if "WRAP=YES" in str(x.message)]
            assert len(wrap_warnings) >= 1, "Expected warning about WRAP=YES"

        content = temp_file.read_text()
        # The ~A data section header should be present
        assert "~A" in content
        # Get data lines after ~A header
        data_section = content.split("~A")[-1]
        data_lines = [
            line for line in data_section.splitlines() if line.strip() and not line.startswith("#")
        ]
        # With 3 curves, non-wrapped mode: 1 header line + 2 data lines = 3 lines total
        # F-I2-M50: Assert exact count — >= 2 would pass even if one row is lost.
        assert len(data_lines) == 3, f"Expected 3 lines (1 header + 2 data), got {len(data_lines)}"
        # Verify WRAP is overridden to NO (F-01: header-data consistency fix)
        assert "WRAP.   NO" in content
        assert "WRAP.   YES" not in content

    # --- F-S6-M4: _format_fixed_precision unit tests ---

    def test_format_fixed_precision_normal_value(self) -> None:
        """Normal value (no exponent in .8g output): magnitude 2, 8 sig digits."""
        result = _format_fixed_precision(123.456, ".8g")
        # magnitude = floor(log10(123.456)) = 2
        # decimal_places = 8 + max(0, -2-1) = 8 → format(..., ".8f")
        assert result == "123.45600000"

    def test_format_fixed_precision_zero(self) -> None:
        """Zero special case: math.log10(0) is guarded by value==0 check."""
        result = _format_fixed_precision(0.0, ".8g")
        assert result == "0.00000000"

    def test_format_fixed_precision_small_value(self) -> None:
        """Small value: magnitude=-3, gets extra decimal places for sig digits."""
        result = _format_fixed_precision(0.001, ".8g")
        # magnitude = floor(log10(0.001)) = -3
        # decimal_places = 8 + max(0, 3-1) = 10 → format(..., ".10f")
        assert result == "0.0010000000"

    def test_format_fixed_precision_large_value(self) -> None:
        """Large value: magnitude=6, stays at sig_digits decimal places."""
        result = _format_fixed_precision(1234567.89, ".8g")
        # magnitude = floor(log10(1234567.89)) = 6
        # decimal_places = 8 + max(0, -6-1) = 8 → format(..., ".8f")
        assert result == "1234567.89000000"

    def test_format_fixed_precision_exponent_trigger_large(self) -> None:
        """Value >= 1e8 triggers exponent in .8g format → _format_fixed_precision.

        Verified through _format_number: format(1.23456789e8, ".8g") produces
        "1.2345679e+08", which contains "e", so _format_fixed_precision is called.
        """
        result = _format_number(1.23456789e8)
        # magnitude = floor(log10(1.23456789e8)) = 8
        # decimal_places = 8 → format(..., ".8f")
        assert "e" not in result.lower()
        assert result == "123456789.00000000"

    def test_format_fixed_precision_exponent_trigger_small(self) -> None:
        """Value < 1e-4 triggers exponent in .8g → _format_fixed_precision path."""
        result = _format_number(1e-5)
        # magnitude = -5, decimal_places = 8 + max(0, 5-1) = 12 → ".12f"
        assert "e" not in result.lower()
        assert result == "0.000010000000"

    def test_format_fixed_precision_negative_value(self) -> None:
        """Negative value: abs() makes magnitude positive, format uses minus sign."""
        result = _format_fixed_precision(-123.456, ".8g")
        # magnitude = floor(log10(123.456)) = 2 (abs used)
        # decimal_places = 8 → format(..., ".8f")
        assert result == "-123.45600000"

    def test_format_fixed_precision_custom_sig_digits(self) -> None:
        """Custom precision string: .5g → 5 significant digits."""
        result = _format_fixed_precision(12345.6789, ".5g")
        # sig_digits = 5, magnitude = 4
        # decimal_places = 5 + max(0, -4-1) = 5 → format(..., ".5f")
        assert result == "12345.67890"

    def test_format_fixed_precision_subnormal_value(self) -> None:
        """Very small value: magnitude=-12, decimal_places clamped to 30."""
        result = _format_fixed_precision(1e-12, ".8g")
        # magnitude = -12, decimal_places = 8 + max(0, 12-1) = 19
        # 19 < 30 → no clamp, stays at 19
        assert "e" not in result.lower()
        assert result.startswith("0.")
        assert len(result.split(".")[1]) == 19

    # --- F-C1: String data injection prevention ---
    def test_string_data_injection_sanitized(self, tmp_path: Path) -> None:
        """String data values containing section-header-like content are sanitized.

        A string curve value containing ``\\n~VERSION`` would be emitted as a
        raw line break causing the reader to detect a fake section header.
        After the fix, ``_sanitize_las_value()`` strips newlines from string
        data values before emission, preventing section injection.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="CDES", unit="", data_format="S"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "CDES"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)

        # String data with embedded newline and section-header-like pattern
        section.string_data["CDES"] = np.array(
            ["normal\n~VERSION INFORMATION\nhijacked", "clean"], dtype=np.str_
        )

        temp_file = tmp_path / "injection_test.las"
        write_las_file(temp_file, las)
        content = temp_file.read_text()

        # The injected raw newline must NOT appear in the data section.
        # The newline after "~A CURVE | CURVE" is the normal section header line
        # break.  Data lines should not contain raw newlines that would split
        # the file into fake section headers.
        data_section = content.split("~A CURVE | CURVE")[1]
        # Count newlines in the data section: one per data row + trailing newline
        data_lines = [line for line in data_section.split("\n") if line.strip()]
        # With 2 rows and no injection: 2 data lines
        # With injection: newline chars in string_data split into extra lines
        assert len(data_lines) == 2, f"Expected 2 data lines, got {len(data_lines)}: {data_lines}"
        # The sanitized string value should be on a single data line
        first_data_line = data_lines[0]
        assert "100" in first_data_line
        # Sanitization replaces newlines with spaces — the injected content
        # is preserved but made safe
        assert "normal" in first_data_line
        assert "hijacked" in first_data_line
        # The second data line is clean
        assert "101" in data_lines[1]

    # --- E-18: LAS 3.0 ~Other deprecation refusal test ---
    def test_las30_other_section_deprecation_warning(self, tmp_path: Path) -> None:
        """Test that LAS 3.0 files with ~Other content are REFUSED.

        LAS 3.0 deprecates the ~Other section — content must go into
        user-defined Parameter or Column Data sections instead.  The
        parser rejects ~O on LAS 3.0 reads; for a directly-constructed
        3.0 model the writer REFUSES (LASWriteError) instead of the old
        warn+drop, which silently discarded user content (E-18).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.other = "This content should trigger a deprecation refusal.\n"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30_other.las"

        with pytest.raises(LASWriteError, match="~Other content cannot be written"):
            write_las_file(temp_file, las)

        # The refused write must not have produced an output file.
        assert not temp_file.exists()

    # --- F-26: Well section mandatory field ordering ---
    def test_well_section_mandatory_field_ordering(self, tmp_path: Path) -> None:
        """Verify that STRT, STOP, STEP, NULL appear first in well section.

        The CWLS spec requires these four mandatory fields to appear
        before other well information fields. The writer must reorder
        the output regardless of dict insertion order.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        # Insert fields in non-spec order to verify reordering
        las.well["COMP"] = "TestCompany"
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["WELL"] = "Well-X"
        las.well["STOP"] = "200.0"
        las.well["FLD"] = "OilField"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "well_order.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Extract well section content between ~WELL and next section
        well_start = content.index("~WELL")
        well_end = content.index("~CURVE")
        well_section = content[well_start:well_end]

        # Find positions of mandatory fields — they must appear before COMP/WELL/FLD
        strt_pos = well_section.index("STRT")
        stop_pos = well_section.index("STOP")
        step_pos = well_section.index("STEP")
        null_pos = well_section.index("NULL")
        comp_pos = well_section.index("COMP")

        # All four mandatory fields must appear before COMP
        assert strt_pos < comp_pos, "STRT must appear before COMP"
        assert stop_pos < comp_pos, "STOP must appear before COMP"
        assert step_pos < comp_pos, "STEP must appear before COMP"
        assert null_pos < comp_pos, "NULL must appear before COMP"

    # --- F2-29: SPACE delimiter + tab in string data ---
    def test_string_data_space_delimiter_with_tab(self, tmp_path: Path) -> None:
        """Tab in string data with SPACE delimiter must be sanitized.

        The SPACE delimiter reader uses str.split() which treats tabs
        as whitespace separators. Embedded tabs must be replaced with
        underscores to prevent one value splitting into multiple tokens.
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="CDES", unit="", data_format="S"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "CDES"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)
        # String data with embedded tab character
        section.string_data["CDES"] = np.array(["LIMESTONE\tFRACTURED", "DOLOMITE"], dtype=np.str_)

        temp_file = tmp_path / "space_delim_tab.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            delim_warnings = [x for x in w if "whitespace" in str(x.message)]
            assert len(delim_warnings) >= 1, (
                "Expected warning about whitespace in string data with SPACE delimiter"
            )

        content = temp_file.read_text()
        # The tab must be replaced with underscore, not left as raw tab
        data_section = content.split("~A CURVE | CURVE")[1]
        assert "\t" not in data_section, "Raw tab must NOT appear in data section"
        assert "LIMESTONE_FRACTURED" in content, "Tab should be replaced with underscore"

        # Verify roundtrip: re-read does not corrupt
        reread = read_las_file(temp_file)
        assert "CDES" in reread.get("string_data", {})
        # The re-read value has the tab replaced by underscore
        assert "LIMESTONE_FRACTURED" in str(reread["string_data"]["CDES"][0])

    # --- F2-30: COMMA delimiter + comma in string data ---
    def test_string_data_comma_delimiter_with_comma(self, tmp_path: Path) -> None:
        """Comma in string data with COMMA delimiter must be sanitized.

        The COMMA delimiter reader splits on commas. Embedded commas
        must be replaced with semicolons to prevent one value
        fragmenting into multiple tokens.
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="CDES", unit="", data_format="S"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "CDES"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)
        # String data with embedded comma
        section.string_data["CDES"] = np.array(
            ["SANDSTONE, FINE GRAINED", "DOLOMITE"], dtype=np.str_
        )

        temp_file = tmp_path / "comma_delim_comma.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            delim_warnings = [x for x in w if "delimiter character (COMMA)" in str(x.message)]
            assert len(delim_warnings) >= 1, "Expected warning about COMMA delimiter in string data"

        content = temp_file.read_text()
        # Find the data section — for LAS 3.0 with data_sections
        data_section = content.split("~A CURVE | CURVE")[1]
        # The first data line should have the comma replaced with semicolon
        first_line = next(ln for ln in data_section.splitlines() if ln.strip())
        assert "SANDSTONE; FINE GRAINED" in first_line, (
            f"Comma should be replaced with semicolon, got: {first_line!r}"
        )
        assert "SANDSTONE, FINE GRAINED" not in first_line

        # Verify roundtrip
        reread = read_las_file(temp_file)
        assert "CDES" in reread.get("string_data", {})
        assert "SANDSTONE; FINE GRAINED" in str(reread["string_data"]["CDES"][0])

    # --- F2-31: TAB delimiter + tab in string data ---
    def test_string_data_tab_delimiter_with_tab(self, tmp_path: Path) -> None:
        """Tab in string data with TAB delimiter must be sanitized.

        The TAB delimiter reader splits on tab characters. Embedded tabs
        must be replaced with spaces to prevent one value fragmenting
        into multiple tokens.
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="TAB")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "CDES"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="CDES", unit="", data_format="S"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "CDES"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)
        # String data with embedded tab
        section.string_data["CDES"] = np.array(["SANDSTONE\tLAMINATED", "DOLOMITE"], dtype=np.str_)

        temp_file = tmp_path / "tab_delim_tab.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            delim_warnings = [x for x in w if "delimiter character (TAB)" in str(x.message)]
            assert len(delim_warnings) >= 1, "Expected warning about TAB delimiter in string data"

        content = temp_file.read_text()
        # The embedded tab should be replaced with space
        assert "SANDSTONE LAMINATED" in content, "Tab should be replaced with space"

        # Verify roundtrip does not corrupt
        reread = read_las_file(temp_file)
        assert "CDES" in reread.get("string_data", {})
        assert "SANDSTONE LAMINATED" in str(reread["string_data"]["CDES"][0])


class TestSectionTypeToPrefix:
    """F-T2-M02/F-ITER2-T2-M04: Tests for _section_type_to_prefix."""

    def test_known_types(self) -> None:
        """Known section types return their mapped prefix."""
        assert _section_type_to_prefix("LOG_DATA") == "A"
        assert _section_type_to_prefix("CORE_DATA") == "CORE_DATA"
        assert _section_type_to_prefix("DRILLING_DATA") == "DRILLING_DATA"
        assert _section_type_to_prefix("INCLINOMETRY_DATA") == "INCLINOMETRY_DATA"
        assert _section_type_to_prefix("TOPS_DATA") == "TOPS_DATA"
        assert _section_type_to_prefix("TEST_DATA") == "TEST_DATA"
        assert _section_type_to_prefix("PERFORATIONS_DATA") == "PERFORATIONS_DATA"

    def test_user_defined_data_type(self) -> None:
        """User-defined section types ending with ``_DATA`` return their
        own name as the prefix for roundtrip fidelity (F-T2-M02)."""
        assert _section_type_to_prefix("CUSTOM_DATA") == "CUSTOM_DATA"
        assert _section_type_to_prefix("MY_DATA") == "MY_DATA"
        assert _section_type_to_prefix("XYZ_DATA") == "XYZ_DATA"

    def test_unknown_type_falls_back_to_a(self) -> None:
        """Completely unknown section types (not ending with ``_DATA``
        and not in the known map) fall back to ``"A"`` (F-ITER2-T2-M04)."""
        assert _section_type_to_prefix("UNKNOWN") == "A"
        assert _section_type_to_prefix("SOMETHING") == "A"
        assert _section_type_to_prefix("") == "A"


class TestFormatNumberNaNInf:
    """F-T2-M06: Test _format_number NaN/Inf defensive path.

    The NaN/Inf guard at _format_number:486-489 catches values that
    slip past the primary guard in _format_data_rows. When null_value
    is provided, it outputs that value; otherwise it formats the
    NaN/Inf directly.
    """

    def test_nan_with_null_value(self) -> None:
        """NaN passed directly to _format_number outputs null_value."""
        result = _format_number(float("nan"), ".8g", null_value=-999.25)
        assert result == "-999.25"

    def test_inf_with_null_value(self) -> None:
        """Inf passed directly to _format_number outputs null_value."""
        result = _format_number(float("inf"), ".8g", null_value=-999.25)
        assert result == "-999.25"

    def test_nan_without_null_value(self) -> None:
        """NaN without null_value falls through to format(float('nan'), '.8g')."""
        result = _format_number(float("nan"), ".8g")
        # format(float('nan'), '.8g') produces 'nan'
        assert "nan" in result.lower()

    def test_inf_without_null_value(self) -> None:
        """Inf without null_value falls through to format(float('inf'), '.8g')."""
        result = _format_number(float("inf"), ".8g")
        assert "inf" in result.lower()

    def test_negative_inf_with_null_value(self) -> None:
        """-Inf with null_value outputs null_value."""
        result = _format_number(float("-inf"), ".8g", null_value=-999.25)
        assert result == "-999.25"


class TestFormatDataRowsVariableLength:
    """F-T2-M03: Test variable-length array padding in _format_data_rows.

    When curves in a data section have different lengths, the writer
    derives the row count from the longest curve and pads shorter
    curves with null_value. The per-curve variable-length support
    handles LAS 3.0 sections where curves are populated from
    different data sections.
    """

    def test_unequal_length_curves_padded_with_null(self) -> None:
        """Curve with 3 values, other with 2 — shorter gets padded."""
        data = {
            "DEPT": np.array([100.0, 101.0, 102.0]),
            "DT": np.array([50.0, 51.0]),
        }
        rows = _format_data_rows(
            ["DEPT", "DT"],
            data,
            {},
            null_value=-999.25,
            delimiter=" ",
            precision=".8g",
        )
        assert len(rows) == 3
        # Row 0: both values present
        assert "50" in rows[0]
        # Row 2: DT padded with null_value
        assert "-999.25" in rows[2]

    def test_single_long_curve_others_short(self) -> None:
        """One curve 5 values, others 1 — all padded to 5 rows."""
        data = {
            "DEPT": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            "DT": np.array([50.0]),
            "GR": np.array([75.0]),
        }
        rows = _format_data_rows(
            ["DEPT", "DT", "GR"],
            data,
            {},
            null_value=-999.25,
            delimiter=" ",
            precision=".8g",
        )
        assert len(rows) == 5
        # Row 0: all values present
        parts = rows[0].split()
        assert len(parts) == 3
        # Row 4: DEPT=104, DT=-999.25, GR=-999.25
        parts4 = rows[4].split()
        assert "-999.25" in parts4[1] or "-999.25000000" in parts4[1]
        assert "-999.25" in parts4[2] or "-999.25000000" in parts4[2]

    def test_empty_data_returns_empty_lines(self) -> None:
        """Empty data dict with 0 rows returns empty list."""
        rows = _format_data_rows(
            ["DEPT"],
            {},  # no data
            {},
            null_value=-999.25,
            delimiter=" ",
        )
        assert rows == []


class TestLOGDataWithSectionCurves:
    """F-ITER2-T2-M03: Test LOG_DATA DataSection with populated section_curves.

    When a LOG_DATA DataSection has truthy ``section_curves``, the
    writer's ``curves_to_emit`` branch (writer.py:222-226) uses them
    instead of the global ``las_file.curves``. This branch has zero
    test coverage — all existing tests use empty/falsy values.
    """

    def test_log_data_with_section_curves_written(self, tmp_path: Path) -> None:
        """LOG_DATA section with section_curves writes curve definitions
        from the section, not the global curve list."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"

        # Global curves (different from section_curves)
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        # Section-specific curves for LOG_DATA
        section_curves = [
            CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
            CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
        ]
        section = DataSection(
            name="CURVE",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=section_curves,
            data={
                "DEPT": np.array([100.0]),
                "GR": np.array([75.0]),
            },
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "log_section_curves.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # ~CURVE section should contain section-specific curves (DEPT, GR)
        # not the globals (DEPT, DT)
        assert "~CURVE INFORMATION" in content
        assert "GR.GAPI" in content
        assert "DT.US/M" not in content


class TestPrecisionRejection:
    """IF-016/IF-017: Write-time rejection of 'n' and '%' precision codes.

    _validate_precision accepts 'n' and '%' with a warning for backward
    compatibility, but write_las_file must REJECT them with LASWriteError
    to prevent silent data corruption (locale-dependent output or
    percentage-scaled values).
    """

    def test_precision_n_rejected_at_write(self, tmp_path: Path) -> None:
        """IF-016: write_las_file with precision='.5n' raises LASWriteError.

        The 'n' format code produces locale-dependent output (e.g., comma
        as decimal separator) that is unparseable in LAS format.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "prec_n.las"
        with pytest.raises(LASWriteError, match="Precision format code 'n'"):
            write_las_file(temp_file, las, precision=".5n")

    def test_precision_percent_rejected_at_write(self, tmp_path: Path) -> None:
        """IF-017: write_las_file with precision='.3%' raises LASWriteError.

        The '%' format code multiplies values by 100 and appends '%',
        producing values that cannot be re-parsed as floating-point numbers.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "prec_pct.las"
        with pytest.raises(LASWriteError, match="Precision format code '%'"):
            write_las_file(temp_file, las, precision=".3%")

    def test_precision_f_valid_succeeds(self, tmp_path: Path) -> None:
        """Valid precision '.5f' is accepted and writes successfully.

        This is the sanity check — standard format codes must work.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "prec_f.las"
        write_las_file(temp_file, las, precision=".5f")  # Should not raise
        assert temp_file.exists()
        content = temp_file.read_text()
        assert "100.00000" in content


class TestAPICodeColonEscaping:
    """IF-024: api_code colon escaping roundtrip.

    The parser's DATA_LINE_PATTERN uses colon as structural separator.
    A CurveEntry api_code with an embedded colon adjacent to whitespace
    would be split across fields on re-read, corrupting the curve
    definition.  The writer escapes api_code colons via
    _escape_colons_for_las_value.
    """

    def test_api_code_embedded_colon_roundtrip(self, tmp_path: Path) -> None:
        """IF-024: CurveEntry api_code='42 : injected' roundtrips correctly.

        Without escaping, the parser would split this at the colon,
        treating '42' as the api_code and 'injected' as a separate field.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(
            CurveDefinition(
                mnemonic="DEPT",
                unit="M",
                api_code="42 : injected",
                description="Depth",
            )
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="DT",
                unit="US/M",
                api_code="99",
                description="Sonic",
            )
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        temp_file = tmp_path / "api_colon.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # The api_code with colon must be escaped in the output
        # (e.g., "42 _:_ injected" or similar — _escape_colons_for_las_value
        # inserts _ at whitespace-colon adjacencies)
        # Verify it's readable
        assert "42" in content
        assert "injected" in content
        # The raw colon-space pattern should be escaped
        curve_section = content.split("~CURVE")[1].split("~A")[0]
        assert "42" in curve_section

        # Roundtrip: re-read and check api_code is preserved (escaped form)
        from pylasdev.parser import LASParser

        parser = LASParser()
        re_read = parser.parse(content)
        # DEPT has the colon-containing api_code
        dept_curve = re_read.curves[0]  # DEPT is first in curves_order
        # api_code is preserved (though in escaped form — _escape_colons
        # inserts _ between whitespace-colon adjacencies)
        assert "42" in dept_curve.api_code, (
            f"api_code should contain '42', got: {dept_curve.api_code!r}"
        )
        assert "injected" in dept_curve.api_code, (
            f"api_code should contain 'injected', got: {dept_curve.api_code!r}"
        )

    def test_api_code_no_colon_unchanged(self, tmp_path: Path) -> None:
        """IF-024: api_code without colon survives roundtrip unchanged."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", api_code="12345", description="Depth")
        )
        las.curves.append(
            CurveDefinition(mnemonic="DT", unit="US/M", api_code="99", description="Sonic")
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        temp_file = tmp_path / "api_no_colon.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "12345" in content

        from pylasdev.parser import LASParser

        parser = LASParser()
        re_read = parser.parse(content)
        assert re_read.curves[0].api_code == "12345"


class TestWriteMultiSectionRejection:
    """F-019/IF-007: Write-time rejection of multi-section non-3.0 files.

    The writer at writer.py:709-716 raises LASWriteError when a non-LAS-3.0
    file has multiple data_sections because the parser cannot correctly
    re-read multi-section data from LAS 1.2/2.0 files.
    """

    def test_write_non_3_0_multi_section_raises(self, tmp_path: Path) -> None:
        """F-019: write_las_file with version 2.0 and 2 data_sections
        raises LASWriteError.

        The writer guards against multiple data_sections in non-LAS-3.0
        files because the parser only reads the first ~A block for
        LAS 1.2/2.0.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        # Two data_sections on a LAS 2.0 file
        section1 = DataSection(
            name="CURVE",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0])},
        )
        section2 = DataSection(
            name="CURVE2",
            curves_order=["DEPT"],
            data={"DEPT": np.array([200.0])},
        )
        las.data_sections.append(section1)
        las.data_sections.append(section2)

        temp_file = tmp_path / "non3_multi.las"
        with pytest.raises(
            LASWriteError, match=r"Multiple data_sections.*only supported for LAS 3.0"
        ):
            write_las_file(temp_file, las)

    def test_write_las30_multi_section_succeeds(self, tmp_path: Path) -> None:
        """F-019: write_las_file with version 3.0 and 2 data_sections succeeds.

        LAS 3.0 natively supports multiple data sections.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0])

        section1 = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        section2 = DataSection(
            name="CORE",
            section_type="CORE_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([200.0])},
        )
        las.data_sections.append(section1)
        las.data_sections.append(section2)

        temp_file = tmp_path / "las30_multi.las"
        write_las_file(temp_file, las)  # Should not raise
        assert temp_file.exists()


class TestColonEscaping:
    """Tests for _escape_colons_for_las_value and colon roundtrip integrity.

    F-EX-03 / F-H04 follow-up: The parser's DATA_LINE_PATTERN uses colon as
    the structural separator between value and description, with a regex
    (\\s+:\\s*|\\s*:\\s+|:\\s*$) that requires whitespace on at least one
    side of the colon.  Well values containing embedded colons with adjacent
    whitespace would be truncated on re-read.  _escape_colons_for_las_value
    inserts ``_`` to break all whitespace-colon adjacencies.
    """

    # ── unit tests for the escaping function ────────────────────────

    def test_no_adjacent_whitespace_not_escaped(self) -> None:
        """Colon with no whitespace on either side is unchanged."""
        assert _escape_colons_for_las_value("Oil:Gas Corp") == "Oil:Gas Corp"

    def test_colon_space_escaped(self) -> None:
        """ ": " (colon-space) → ":_ " (underscore between colon and space)."""
        assert _escape_colons_for_las_value("Oil: Gas Corp") == "Oil:_ Gas Corp"

    def test_trailing_colon_escaped(self) -> None:
        """Trailing ":" → ":_" to prevent :\\s*$ match."""
        assert _escape_colons_for_las_value("Oil:") == "Oil:_"

    def test_space_colon_escaped(self) -> None:
        """ " :" (space-colon) → " _:" (underscore between space and colon)."""
        assert _escape_colons_for_las_value("Oil :Gas Corp") == "Oil _:Gas Corp"

    def test_space_colon_space_double_escaped(self) -> None:
        """ " : " (space-colon-space) → " _:_ " (underscore on both sides)."""
        assert _escape_colons_for_las_value("Oil : Gas Corp") == "Oil _:_ Gas Corp"

    def test_trailing_space_colon_escaped(self) -> None:
        """Trailing " :" → " _:_" (both sides: before-colon + end-of-string)."""
        assert _escape_colons_for_las_value("Oil :") == "Oil _:_"

    def test_leading_colon_not_escaped_when_no_ws(self) -> None:
        """Leading ":" with no following whitespace is unchanged."""
        assert _escape_colons_for_las_value(":Oil") == ":Oil"

    def test_leading_colon_space_escaped(self) -> None:
        """ " : " at start → ":_ " (colon-space escaping at position 0)."""
        assert _escape_colons_for_las_value(": value") == ":_ value"

    def test_leading_space_colon_escaped(self) -> None:
        """ " :" at start (space-colon-value) → " _:..." ."""
        assert _escape_colons_for_las_value(" :value") == " _:value"

    def test_multiple_spaces_before_colon_escaped(self) -> None:
        """Multiple spaces before colon: all whitespace preserved, _ inserted."""
        assert _escape_colons_for_las_value("  :value") == "  _:value"

    def test_multiple_spaces_before_colon_with_trailing_space(self) -> None:
        """ "  : value" (two spaces, colon, space) is fully escaped."""
        assert _escape_colons_for_las_value("  : value") == "  _:_ value"

    def test_multiple_embedded_colons(self) -> None:
        """Multiple embedded colons are all escaped independently."""
        assert _escape_colons_for_las_value("A: B : C") == "A:_ B _:_ C"

    # ── roundtrip tests (write → parse → verify value survived) ────

    def _roundtrip_well_value(self, tmp_path: Path, value: str) -> str:
        """Helper: write a LAS file with a single well entry, parse it back,
        and return the parsed well value."""
        from pylasdev.parser import LASParser

        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "200.0"
        las.well["COMP"] = value

        temp_file = tmp_path / "colon_roundtrip.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        parser = LASParser()
        re_read = parser.parse(content)
        return re_read.well["COMP"]

    def test_roundtrip_no_colon(self, tmp_path: Path) -> None:
        """Plain value without colon survives roundtrip unchanged."""
        result = self._roundtrip_well_value(tmp_path, "Oil")
        assert result == "Oil"

    def test_roundtrip_colon_space(self, tmp_path: Path) -> None:
        """ "Oil: Gas Corp" (colon-space) survives roundtrip — parser now un-escapes."""
        result = self._roundtrip_well_value(tmp_path, "Oil: Gas Corp")
        # F-022: The parser now un-escapes colon artifacts inserted by
        # _escape_colons_for_las_value, restoring the original value.
        assert result == "Oil: Gas Corp"
        assert "Oil" in result  # Not truncated to just "Oil"

    def test_roundtrip_space_colon_space(self, tmp_path: Path) -> None:
        """ "Oil : Gas Corp" (space-colon-space) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil : Gas Corp")
        # Must NOT be truncated to just "Oil" (the F-EX-03 corruption).
        assert result != "Oil"
        assert "Oil" in result
        assert "Gas Corp" in result

    def test_roundtrip_space_colon_no_trailing_space(self, tmp_path: Path) -> None:
        """ "Oil :Gas Corp" (space-colon, no space after) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil :Gas Corp")
        assert result != "Oil"
        assert "Oil" in result
        assert "Gas Corp" in result

    def test_roundtrip_trailing_colon(self, tmp_path: Path) -> None:
        """ "Oil:" (trailing colon) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil:")
        # Must NOT be truncated — colon-containing value preserved.
        assert result != "Oil"
        assert "Oil" in result

    def test_roundtrip_trailing_space_colon(self, tmp_path: Path) -> None:
        """ "Oil :" (trailing space-colon) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil :")
        # Must NOT be truncated to just "Oil" (the F-EX-03 corruption).
        assert result != "Oil"
        assert "Oil" in result

    def test_roundtrip_leading_colon_no_ws(self, tmp_path: Path) -> None:
        """Leading colon without whitespace is harmless."""
        result = self._roundtrip_well_value(tmp_path, ":Oil")
        # Leading colon is not a structural separator — preserved as-is.
        assert ":Oil" in result

    def test_roundtrip_multiple_embedded_colons(self, tmp_path: Path) -> None:
        """Multiple embedded colons all survive roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "A: B : C")
        # Must contain A, B, C — not truncated at any colon.
        assert "A" in result
        assert "B" in result
        assert "C" in result


class TestValidatePrecision:
    """F-I2-M70: Tests for _validate_precision with invalid format codes.

    All existing precision tests use valid values (.4g, .8g, .5g).
    The validation guard at writer.py:169 rejects non-numeric format
    codes (x, o, b, c, d) that would produce hex/octal/binary/character
    output when applied to floating-point values.
    """

    def test_valid_precision_formats_pass(self) -> None:
        """All float-compatible format specifiers pass validation."""
        _validate_precision(".8g")
        _validate_precision(".6f")
        _validate_precision(".10e")
        _validate_precision(".4E")
        _validate_precision(".12F")
        _validate_precision(".5n")
        _validate_precision(".3%")
        _validate_precision(".1G")
        # No type code defaults to g-type (safe for floats)
        _validate_precision(".8")

    def test_invalid_hex_format_raises(self) -> None:
        """Hex format code 'x' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".8x")

    def test_invalid_hex_uppercase_raises(self) -> None:
        """Uppercase hex format code 'X' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".4X")

    def test_invalid_decimal_integer_raises(self) -> None:
        """Decimal integer format code 'd' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".5d")

    def test_invalid_binary_format_raises(self) -> None:
        """Binary format code 'b' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".8b")

    def test_invalid_octal_format_raises(self) -> None:
        """Octal format code 'o' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".8o")

    def test_invalid_character_format_raises(self) -> None:
        """Character format code 'c' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".2c")

    def test_missing_dot_raises(self) -> None:
        """Format string without leading dot raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision("8g")

    def test_no_digits_raises(self) -> None:
        """Format string with dot but no digits raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision(".g")

    def test_arbitrary_string_raises(self) -> None:
        """Completely non-format string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision("abc")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision format"):
            _validate_precision("")


class TestLeadingSectionRegex:
    """F-I2-M71: Tests for \\s* portion of _LEADING_SECTION_RE.

    The regex ``^\\s*~([A-Za-z])`` (writer.py:40) strips leading
    whitespace before a section-header-like ``~[A-Za-z]`` pattern.
    Existing tests only verify ``~VERSION`` without leading whitespace.
    A regression to ``^~`` (without \\s*) would cause leading-whitespace
    values to bypass sanitization and go undetected.
    """

    def test_leading_tab_before_section_header(self) -> None:
        """Tab before ~VERSION: \t is stripped along with ~."""
        assert _sanitize_las_value("\t~VERSION broken") == "VERSION broken"

    def test_leading_spaces_before_section_header(self) -> None:
        """Two spaces before ~VERSION: spaces and ~ are stripped."""
        assert _sanitize_las_value("  ~VERSION info") == "VERSION info"

    def test_leading_spaces_before_A_section(self) -> None:
        """Three spaces before ~A data: all whitespace and ~ stripped."""
        assert _sanitize_las_value("   ~A data") == "A data"

    def test_leading_tab_before_lowercase_section(self) -> None:
        """Tab before lowercase ~a: whitespace + ~ stripped, letter preserved."""
        assert _sanitize_las_value("\t~a lowercase") == "a lowercase"

    def test_leading_mixed_whitespace_before_section(self) -> None:
        """Mix of tab and spaces before ~W text."""
        assert _sanitize_las_value("\t  ~W text") == "W text"

    def test_leading_whitespace_only_no_tilde(self) -> None:
        """Leading whitespace without tilde is left unchanged
        (not a section header pattern)."""
        result = _sanitize_las_value("  no-tilde here")
        assert result == "  no-tilde here"

    def test_leading_tilde_no_letter_unchanged(self) -> None:
        """~ followed by non-letter (digit) is NOT matched by
        [A-Za-z] — leading whitespace + tilde preserved."""
        result = _sanitize_las_value("  ~1 numeric")
        assert result == "  ~1 numeric"


class TestSingleDataSectionFallback:
    """Tests for F-067/F-111/F-208 — copy-back from data_sections to legacy attrs.

    When a non-LAS-3.0 file has exactly one DataSection and empty logs,
    the writer copies data_sections[0].data → las.logs (and similarly
    string_data and curves_order) so the legacy ~A write path can
    emit the data.

    Regression risk: a refactor that breaks the condition at
    writer.py:751 (``if not las_file.logs:``) would silently lose
    data on write.
    """

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_las2_with_single_section() -> LASFile:
        """Return a LASFile(vers='2.0') with one DataSection, no logs."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="API"))

        ds = DataSection(
            name="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={
                "DEPT": np.array([100.0, 200.0, 300.0]),
                "GR": np.array([50.0, 60.0, 70.0]),
            },
        )
        las.data_sections.append(ds)
        return las

    # ── tests ───────────────────────────────────────────────────────

    def test_copy_back_fires_and_data_is_preserved(self, tmp_path: Path) -> None:
        """LAS 2.0 with single DataSection + empty logs → copy-back fires,
        data appears in output file."""
        las = self._make_las2_with_single_section()

        output_file = tmp_path / "test.las"
        with pytest.warns(
            UserWarning,
            match="data_sections are only supported for LAS 3.0",
        ):
            write_las_file(str(output_file), las)

        content = output_file.read_text()
        # Data section values are present in the output
        assert "100" in content
        assert "50" in content

        # Re-parse and verify the data roundtrips
        reparsed = read_las_file(str(output_file))
        logs = reparsed.get("logs", {})
        assert "DEPT" in logs, f"Expected 'DEPT' in reparsed logs, got {list(logs.keys())}"
        np.testing.assert_array_almost_equal(logs["DEPT"], np.array([100.0, 200.0, 300.0]))
        assert "GR" in logs, f"Expected 'GR' in reparsed logs, got {list(logs.keys())}"
        np.testing.assert_array_almost_equal(logs["GR"], np.array([50.0, 60.0, 70.0]))

    def test_copy_back_not_overwrite_user_logs(self, tmp_path: Path) -> None:
        """If user pre-populates las.logs, copy-back does NOT fire
        — user data is preserved."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"

        # User-populated curves and logs (different from data_sections)
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([1000.0, 2000.0, 3000.0])
        las.logs["DT"] = np.array([400.0, 500.0, 600.0])

        # DataSection with conflicting data — should be ignored
        # because las.logs is already populated
        ds = DataSection(
            name="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={
                "DEPT": np.array([100.0, 200.0, 300.0]),
                "GR": np.array([50.0, 60.0, 70.0]),
            },
        )
        las.data_sections.append(ds)

        output_file = tmp_path / "user_logs.las"
        with pytest.warns(
            UserWarning,
            match="data_sections are only supported for LAS 3.0",
        ):
            write_las_file(str(output_file), las)

        # Verify user's DT data is in the output (from las.logs, not data_sections)
        content = output_file.read_text()
        assert "DT" in content

        # Re-parse: DEPT values should be user's (1000, 2000, 3000),
        # not data_sections' (100, 200, 300)
        reparsed = read_las_file(str(output_file))
        logs = reparsed.get("logs", {})
        assert "DEPT" in logs, "Expected DEPT in reparsed logs"
        assert logs["DEPT"][0] == 1000.0, f"Expected user DEPT value 1000.0, got {logs['DEPT'][0]}"
        # GR should NOT be present — it was in data_sections only,
        # and copy-back was blocked by pre-populated las.logs
        assert "GR" not in logs, (
            f"GR should NOT be in output (copy-back blocked), but found in {list(logs.keys())}"
        )

    def test_copy_back_preserves_curves_order(self, tmp_path: Path) -> None:
        """Copy-back sets las.curves_order from data_sections[0].curves_order."""
        las = self._make_las2_with_single_section()

        output_file = tmp_path / "order.las"
        with pytest.warns(
            UserWarning,
            match="data_sections are only supported for LAS 3.0",
        ):
            write_las_file(str(output_file), las)

        content = output_file.read_text()
        # Verify both curves appear in the output in expected order
        dept_pos = content.index("DEPT")
        gr_pos = content.index("GR")
        assert dept_pos < gr_pos, (
            f"Expected DEPT before GR in output, but DEPT at {dept_pos}, GR at {gr_pos}"
        )

    def test_copy_back_inconsistent_curves_count_raises(self, tmp_path: Path) -> None:
        """F-R-05: Pre-populated curves_order + empty curves but
        data_sections[0] has section_curves → independent guards
        produce mismatch → LASDataError raised."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"

        # Pre-populate curves_order but leave curves empty
        las.curves_order = ["DEPT", "GR", "LLS"]
        # No las.curves.append(...) — deliberately empty

        # DataSection with section_curves of a different count
        section_curves = [
            CurveDefinition(mnemonic="DEPT", unit="M"),
            CurveDefinition(mnemonic="GR", unit="API"),
        ]
        ds = DataSection(
            name="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=section_curves,
            data={
                "DEPT": np.array([100.0, 200.0, 300.0]),
                "GR": np.array([50.0, 60.0, 70.0]),
            },
        )
        las.data_sections.append(ds)

        output_file = tmp_path / "inconsistent.las"
        with (
            pytest.warns(
                UserWarning,
                match="data_sections are only supported for LAS 3.0",
            ),
            pytest.raises(LASWriteError, match=r"curves count.*does not match"),
        ):
            write_las_file(str(output_file), las)


class TestF010EmptyDlmWriterRegression:
    """F-010 regression: empty DLM string should not crash the writer.

    Before F-010, VersionSection(dlm="") passed validation but
    writer.py's ``if las_file.version.dlm`` was falsy for empty
    string, causing the DLM header line to be skipped with no
    substitute.  Downstream code expecting a DLM to be set then
    crashed.  After F-010, the writer guards with an explicit
    ``and las_file.version.dlm`` check.
    """

    def test_write_with_empty_dlm_succeeds(self, tmp_path: Path) -> None:
        """Writer with DLM='' should not crash — header line is skipped."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 200.0, 300.0])

        temp_file = tmp_path / "empty_dlm.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        assert "~V" in content
        assert "~W" in content
        assert "~C" in content
        assert "~A" in content
        # DLM line should NOT appear since DLM is empty
        assert "DLM ." not in content


class TestF016TimeOffsetNonAsciiRegression:
    """F-016 regression: time_offset only emitted for data_format='A'.

    Before F-016, the writer emitted time_offset for ALL array
    curves regardless of data_format.  This produced unparseable
    LAS when the curve used a non-ASCII format (e.g., 'F8.3') —
    the parser sees `{F8.3:5.5}` in the format string and chokes.
    After F-016, time_offset is restricted to data_format == "A".
    """

    def test_non_ascii_array_no_time_offset(self, tmp_path: Path) -> None:
        """Array curve with data_format='F' gets no time_offset in format."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "NMR[1]"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR[1]",
                unit="ms",
                data_format="F",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=5.5),
            )
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["NMR[1]"] = np.array([10.0])

        temp_file = tmp_path / "non_ascii_array.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # F-016: time_offset must NOT appear in format spec for non-'A'
        # data_format curves.  With data_format='F', the format is just
        # {F} with no colon or time_offset (contrast with 'A' curves
        # where {A:5.5} would appear).
        assert "{F}" in content  # format without time_offset
        assert "   {A}" not in content  # no A-format curves
        # Specifically: no time_offset 5.5 in the format specs
        assert ":5.5" not in content


# ============================================================
# Production Check Regression Tests
# ============================================================


class TestProductionCheckWriterFixes:
    """Regression tests for production check fixes in writer.py."""

    # --- F-209 (MEDIUM): multi-section LOG_DATA curves preserved ---

    def test_multi_log_data_section_all_curves_emitted(self, tmp_path: Path) -> None:
        """F-209: Write with 2 LOG_DATA sections emits curves from BOTH.

        Before the fix, next() selected only the first LOG_DATA section's
        curves. Now all LOG_DATA sections' section_curves are accumulated.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"

        # Section 1: DEPT + GR
        section1 = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
            ],
            data={
                "DEPT": np.array([100.0]),
                "GR": np.array([75.0]),
            },
        )
        # Section 2: DEPT + DT (different set)
        section2 = DataSection(
            name="LOG2",
            section_type="LOG_DATA",
            curves_order=["DEPT", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"),
            ],
            data={
                "DEPT": np.array([200.0]),
                "DT": np.array([50.0]),
            },
        )
        las.data_sections.append(section1)
        las.data_sections.append(section2)

        temp_file = tmp_path / "multi_log_data.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Both section curves must be represented in the ~CURVE block
        assert "GR.GAPI" in content, "GR from section1 should be in ~CURVE"
        assert "DT.US/M" in content, "DT from section2 should be in ~CURVE"

    # --- F-210 / M-04: colon-escaping for curve unit ---

    def test_curve_unit_colon_escaped(self, tmp_path: Path) -> None:
        """F-210/M-04: Curve unit with ' : ' (colon-space adjacency) is rejected.

        F-210 originally documented the writer escaping colon-space
        adjacencies in the unit (``g/cm3 : grain`` → ``g/cm3 _:_ grain``).
        M-04 (CONFIRMED) found that escape path insufficient: an escaped
        colon unit cannot be re-parsed by the parser's unit grammar, so
        the whole curve + data column is silently dropped on roundtrip.
        The fix rejects such units at the model layer instead —
        CurveDefinition construction now raises ValueError.
        """
        with pytest.raises(ValueError, match="invalid unit"):
            CurveDefinition(
                mnemonic="SPECIAL",
                unit="g/cm3 : grain",
                description="Special Density",
            )

    # --- F-209 / F-277: early-return path blank line + warning ---

    def test_empty_curves_section_has_blank_line(self, tmp_path: Path) -> None:
        """F-209: the no-curves early-return path emits a trailing blank line.

        The F-209 fix added an early-return when no curves are available
        (_writer_las30.py:454-464).  This path must emit a
        section-terminating blank line like the normal code path does,
        per CWLS LAS 2.0 Section 3.2.

        The early-return fires only with a TRUTHY ``data_sections`` that
        contributes no curves AND no top-level curves — ``data_sections
        = []`` (falsy) falls through to the else-branch and never reaches
        it.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        # TRUTHY data_sections contributing no curves — reaches the
        # F-209 early-return (a falsy [] would not).
        las.data_sections.append(
            DataSection(name="LOG", section_type="LOG_DATA", curves_order=[], data={})
        )

        temp_file = tmp_path / "empty_curves_bl.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text()
        # CURVE section should exist with proper structure.
        assert "~CURVE" in content
        # The F-209 early-return appends the section-terminating blank
        # line (lines.append("") at _writer_las30.py:463) right after
        # the ~CURVE INFORMATION header.
        after_header = content.split("~CURVE INFORMATION", 1)[1]
        assert after_header.startswith("\n\n"), (
            f"no blank line after ~CURVE INFORMATION: {after_header[:40]!r}"
        )

    def test_empty_curves_early_return_warns(self, tmp_path: Path) -> None:
        """F-277: the no-curves early-return emits a loud warning.

        When a LAS 3.0 file has no curves to emit for ~C (truthy
        data_sections contributing no curves and no top-level curves),
        the writer warns "No curves to emit for ~C section — skipping"
        instead of silently writing a header-only curve section.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.data_sections.append(
            DataSection(name="LOG", section_type="LOG_DATA", curves_order=[], data={})
        )

        temp_file = tmp_path / "empty_curves_warn.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
        assert any("No curves to emit" in str(w.message) for w in rec), (
            f"no early-return warning: {[str(w.message) for w in rec]}"
        )

    # --- F-001 (CRITICAL): np.str_ truncation — string curves roundtrip ---

    def test_string_curve_no_truncation(self, tmp_path: Path) -> None:
        """F-001: LAS 3.0 string curves roundtrip without truncation.

        Before the fix, ``dtype=np.str_`` created fixed-width string
        arrays that silently truncated values exceeding the maximum
        element length.  The fix uses ``dtype=object`` to preserve
        arbitrary-length Python strings.
        """
        long_strings = [
            "A" * 50,
            "B" * 100,
            "C" * 200,
            "D" * 512,  # Well beyond default np.str_ width of ~32
        ]
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["STRVAL"]
        las.curves.append(CurveDefinition(mnemonic="STRVAL", data_format="S"))
        las.string_data["STRVAL"] = np.array(long_strings, dtype=object)

        temp_file = tmp_path / "long_string_curves.las"
        write_las_file(temp_file, las)

        # Roundtrip: read it back via the object API
        las2 = read_las_file_as_object(temp_file)
        # String data must match exactly, no truncation
        assert "STRVAL" in las2.string_data, "STRVAL not found in string_data after roundtrip"
        result = las2.string_data["STRVAL"]
        assert result.tolist() == long_strings, f"Expected {long_strings}, got {result.tolist()}"

    # --- F-031: writer wrap/state exception safety ---

    def test_model_state_preserved_after_write(self, tmp_path: Path) -> None:
        """F-031: LASFile state (logs, string_data, curves) restored after write.

        Before the fix, ``_write_ascii_sections`` mutated
        ``las_file.logs``, ``las_file.string_data``,
        ``las_file.curves_order``, and ``las_file.curves`` during
        the legacy copy-back path without a finally block to restore
        the caller's state if an exception occurred downstream.

        The fix saves all four attributes before the try block and
        restores them in a finally block.  This test verifies that
        writing a file preserves the caller's pre-write state.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "200.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        las.logs["DEPT"] = np.array([100.0, 200.0])
        las.logs["GR"] = np.array([75.0, 85.0])

        saved_logs = dict(las.logs)
        saved_curves_order = list(las.curves_order)
        saved_curves = list(las.curves)

        temp_file = tmp_path / "state_preserved.las"
        write_las_file(temp_file, las)

        # After write, the caller's model MUST not have corrupted logs
        assert list(las.logs.keys()) == list(saved_logs.keys()), (
            f"Log keys changed: {list(las.logs.keys())} vs {list(saved_logs.keys())}"
        )
        assert np.array_equal(las.logs["DEPT"], saved_logs["DEPT"]), (
            "DEPT data mutated during write"
        )
        assert np.array_equal(las.logs["GR"], saved_logs["GR"]), "GR data mutated during write"
        assert las.curves_order == saved_curves_order, "curves_order changed during write"
        assert len(las.curves) == len(saved_curves), "curves count changed during write"

    # --- F2-016: writer precision cap ---

    def test_precision_capped_at_100(self) -> None:
        """F2-016: High-precision specifiers cap at 100 decimal places.

        Before the fix, ``max(decimal_places, sig_digits)`` undid the
        ``min(..., 100)`` cap when ``sig_digits`` was large.  The fix
        caps ``sig_digits`` at the point of extraction so that the
        subsequent ``min(..., 100)`` actually limits the output width.
        """
        # 1e-100 with sig_digits=200 would produce ~293 decimal places
        # without the cap.  With the fix, it caps at 100.
        result = _format_fixed_precision(1e-100, ".200g")
        # The formatted result should not be excessively long
        # (cap at 100 decimal places ≈ ~104 chars including leading zero)
        assert len(result) < 200, (
            f"Precision cap should limit output length, got {len(result)} chars: {result!r}"
        )

        # Also verify sensible inputs still work
        normal = _format_fixed_precision(3.14, ".4g")
        assert "3.14" in normal

    # --- I2F-007: string_data-only curves included in curves_order ---

    def test_string_data_only_curves_in_legacy_writer(self, tmp_path: Path) -> None:
        """I2F-007: string_data-only curves appear in legacy (~A) writer output.

        Before the fix, the merge guard at _writer_base.py:590 checked only
        ``k in self._las_file.logs``, silently skipping curves that exist
        only in ``string_data``.  After the fix, the condition also checks
        ``k in self._las_file.string_data``.

        The merge code (Path A in _write_ascii_legacy) handles data_sections
        for LAS 1.2/2.0 files.  LAS 3.0 files use a separate writer path
        (_write_ascii_las30) that is not affected by this fix.

        We set data_sections after construction because models.py blocks
        non-LAS-3.0 data_sections at init time (design choice).
        We also set curves/curves_order pre-write because the ~C section
        is written before the data_sections copy-back.
        """
        ds = DataSection(
            name="ASCII",
            section_type="LOG_DATA",
            curves_order=["DEPT", "CONTINENT"],
            data={"DEPT": np.array([100.0, 200.0], dtype=np.float64)},
            string_data={"CONTINENT": np.array(["NA", "NA"], dtype=object)},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="CONTINENT", unit=""),
            ],
        )
        las = LASFile(version=VersionSection(vers="2.0"))
        # Set data_sections post-construction to bypass __post_init__
        # validation (which requires LAS 3.0 for data_sections).
        las.data_sections = [ds]
        # Set curves pre-write — the ~C section is written before the
        # data_sections copy-back in _write_ascii_legacy.
        las.curves = list(ds.section_curves)
        las.curves_order = ["DEPT"]
        # NOTE: curves_order has only DEPT initially — the fix at
        # _writer_base.py:590 appends CONTINENT during the merge because
        # it now checks string_data (not just logs).

        temp_file = tmp_path / "string_data_only_output.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")

        # Both curves must appear in the written ~A header line.
        # LAS 2.0 legacy format doesn't support {S} string markers in
        # ~C section, so "NA" values for CONTINENT won't roundtrip
        # correctly on re-read.  The fix is verified by the presence
        # of CONTINENT in the output content.
        assert "DEPT" in content, "Numeric curve DEPT not found in writer output"
        assert "CONTINENT" in content, (
            "string_data-only curve CONTINENT not found in writer output — I2F-007 fix not applied"
        )
        # Verify the ~A header line includes both curves
        assert "~A" in content
        # The ~A line should list DEPT and CONTINENT (space-separated)
        a_line = [ln for ln in content.splitlines() if ln.startswith("~A")]
        assert len(a_line) == 1, f"Expected one ~A line, found {len(a_line)}"
        assert "CONTINENT" in a_line[0], f"CONTINENT missing from ~A header: {a_line[0]!r}"


# ============================================================
# G7 — Writer Base Fix Regression Tests (W-02, W-04..W-09)
# ============================================================


class TestG7WriterBaseFixes:
    """Regression tests for fix group G7 (_writer_base.py).

    Each test FAILS on the pre-fix code and PASSES after the fix.
    """

    # --- W-02: bare precision specifier must not crash on integral data ---

    def test_write_bare_precision_normalized(self, tmp_path: Path) -> None:
        """W-02: precision='.5' (bare, no code letter) must not crash.

        Before the fix, format(int(v), '.5') raised ValueError
        ("Precision not allowed in integer format specifier") for
        integer-valued data such as depths, so writing any file with a
        bare precision specifier failed with LASWriteError mid-write.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": "1000.0",
                "STOP": "1002.0",
                "STEP": "1.0",
                "NULL": "-999.25",
            },
            "logs": {"DEPT": np.array([1000.0, 1001.0, 1002.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "bare_precision.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, data, precision=".5")

        reread = read_las_file(str(temp_file))
        np.testing.assert_array_almost_equal(
            reread["logs"]["DEPT"], np.array([1000.0, 1001.0, 1002.0])
        )

    # --- W-04: copy-back warning reflects the actual outcome ---

    def _make_las2_single_section(self, with_top_level_logs: bool) -> LASFile:
        """Build a LAS 2.0 file with one DataSection (and optionally
        pre-populated top-level logs)."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"
        if with_top_level_logs:
            las.curves_order = ["DEPT"]
            las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
            las.logs["DEPT"] = np.array([1000.0, 2000.0])
        section = DataSection(
            name="CURVE",
            curves_order=["DEPT"],
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
            data={"DEPT": np.array([1.0, 2.0])},
        )
        las.data_sections.append(section)
        return las

    def test_copy_back_warning_when_content_dropped(self, tmp_path: Path) -> None:
        """W-04: when top-level logs already exist, the section data is
        dropped and the warning must say so — not claim it 'will be
        preserved'."""
        las = self._make_las2_single_section(with_top_level_logs=True)

        temp_file = tmp_path / "w04_dropped.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
        messages = [str(w.message) for w in caught]
        assert any("will NOT be preserved" in m for m in messages), (
            f"Expected 'will NOT be preserved' warning, got {messages}"
        )
        assert not any("Single-section data will be preserved" in m for m in messages)

    def test_copy_back_warning_when_content_preserved(self, tmp_path: Path) -> None:
        """W-04: with empty top-level containers the section data is
        copied back and the warning still claims preservation."""
        las = self._make_las2_single_section(with_top_level_logs=False)

        temp_file = tmp_path / "w04_preserved.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
        messages = [str(w.message) for w in caught]
        assert any("Single-section data will be preserved" in m for m in messages), (
            f"Expected preservation warning, got {messages}"
        )

        reread = read_las_file_as_object(temp_file)
        np.testing.assert_array_almost_equal(reread.logs["DEPT"], np.array([1.0, 2.0]))

    # --- W-05: ~C must not be empty when data_sections provides curves ---

    def test_single_data_section_emits_curve_definitions(self, tmp_path: Path) -> None:
        """W-05: LAS 2.0 with a single data_sections and empty top-level
        curves must emit curve definitions in ~C, not an empty section.

        Before the fix, ~C was emitted empty while ~A carried the data
        columns — curve metadata (units/descriptions) was silently lost
        and the data was discarded on re-read ('no curves are defined').
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", description="Depth"),
                CurveDefinition(mnemonic="DT", unit="US/M", description="Delta-T"),
            ],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([50.0, 51.0]),
            },
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "w05_curve_defs.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        curve_sec = content.split("~CURVE", 1)[1].split("~", 1)[0]
        assert "DEPT.M" in curve_sec, f"~C missing DEPT definition: {curve_sec!r}"
        assert "DT.US/M" in curve_sec, f"~C missing DT definition: {curve_sec!r}"

        reread = read_las_file_as_object(temp_file)
        assert [c.mnemonic for c in reread.curves] == ["DEPT", "DT"], (
            f"Unexpected curves after roundtrip: {[c.mnemonic for c in reread.curves]}"
        )
        np.testing.assert_array_almost_equal(reread.logs["DEPT"], np.array([100.0, 101.0]))
        np.testing.assert_array_almost_equal(reread.logs["DT"], np.array([50.0, 51.0]))

    # --- W-06: guarded containers survive a write ---

    def test_guards_survive_successful_write(self, tmp_path: Path) -> None:
        """W-06: after a successful write, logs/curves must remain
        guarded — invalid mutations are still rejected.

        Before the fix, the writers' finally blocks restored plain
        dict/list snapshots, permanently stripping the guards, so
        ``logs[123] = ...`` was silently accepted."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "w06_guards.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        with pytest.raises(TypeError):
            las.logs[123] = np.array([1.0])
        with pytest.raises(TypeError):
            las.curves.append("not a curve")

    def test_guards_survive_failed_write(self, tmp_path: Path) -> None:
        """W-06/W-07: after a FAILED write, the model is restored to its
        pre-write state and the containers remain guarded."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="YES", dlm="SPACE")
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        # Two data_sections force LASWriteError inside the write pass
        # (multiple data_sections are only supported for LAS 3.0).
        las.data_sections.append(
            DataSection(name="A", curves_order=["DEPT"], data={"DEPT": np.array([1.0])})
        )
        las.data_sections.append(
            DataSection(name="B", curves_order=["DEPT"], data={"DEPT": np.array([2.0])})
        )

        temp_file = tmp_path / "w06_failed.las"
        with pytest.raises(LASWriteError):
            write_las_file(temp_file, las)

        # W-07: wrap restored to the pre-write value on failure.
        assert las.version.wrap == "YES", f"wrap leaked to {las.version.wrap!r} after failed write"
        # W-06: guards intact after the failed write.
        with pytest.raises(TypeError):
            las.logs[123] = np.array([1.0])

    # --- W-08: parameter array_index roundtrip ---

    def test_parameter_array_index_roundtrip(self, tmp_path: Path) -> None:
        """W-08: a parameter with array_index but a non-bracket mnemonic
        (RUN, not RUN[1]) must roundtrip with array_index preserved.

        Before the fix, the writer emitted 'RUN' (no bracket) and the
        parser reconstructs array_index only from bracket mnemonics, so
        array_index was silently lost (1 → None)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.parameters.append(
            ParameterEntry(
                mnemonic="RUN",
                unit="",
                value="10",
                description="Run number",
                array_index=1,
            )
        )

        temp_file = tmp_path / "w08_param_idx.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        assert "RUN[1]" in content

        reread = read_las_file_as_object(temp_file)
        runs = [p for p in reread.parameters if p.mnemonic == "RUN[1]"]
        assert len(runs) == 1, (
            f"Expected RUN[1] parameter, got {[p.mnemonic for p in reread.parameters]}"
        )
        assert runs[0].array_index == 1

    # --- W-09: curve array_info roundtrip ---

    def test_curve_array_info_roundtrip_without_bracket_mnemonic(self, tmp_path: Path) -> None:
        """W-09: a curve with array_info but a non-bracket mnemonic
        (NMR, not NMR[5]) must roundtrip with array_info preserved and
        numeric data intact.

        Before the fix, the writer emitted 'NMR' with {A:5}; the parser
        reconstructs array_info only from bracket mnemonics, so the curve
        was treated as string-format and its numeric data was
        reclassified into string_data on re-read."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "NMR"]
        las.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH", data_format="F")
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR",
                unit="ms",
                description="NMR Echo",
                data_format="A",
                array_info=ArrayElementInfo(base_name="NMR", index=5, time_offset=0.0),
            )
        )
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["NMR"] = np.array([10.0, 11.0])

        temp_file = tmp_path / "w09_array_info.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        assert "NMR[5]" in content

        reread = read_las_file_as_object(temp_file)
        nmr = [c for c in reread.curves if c.mnemonic == "NMR[5]"]
        assert len(nmr) == 1, f"Expected NMR[5] curve, got {[c.mnemonic for c in reread.curves]}"
        assert nmr[0].is_array_element
        assert nmr[0].array_info is not None
        assert nmr[0].array_info.base_name == "NMR"
        assert nmr[0].array_info.index == 5
        # Numeric data must land in logs, NOT string_data.
        assert "NMR[5]" in reread.logs
        assert "NMR[5]" not in reread.string_data
        np.testing.assert_array_almost_equal(reread.logs["NMR[5]"], np.array([10.0, 11.0]))


class TestG8FixGroup:
    """Regression tests for fix group G8 (W-01, N-I-15, N-I-16, N-I-17,
    N-I-18, N-I-20) — LAS 3.0 writer duplicate-curve emission, per-section
    curve scoping, 256-char line-limit gaps, string null/NaN guard, and
    brace-token-in-description format extraction.
    """

    # --- W-01: duplicate curve emission in ~C ---

    def test_w01_no_duplicate_curve_emission_across_sections(self, tmp_path: Path) -> None:
        """W-01: two LOG_DATA sections sharing DEPT emit DEPT once in ~C.

        Before the fix, the first loop extended each section's curves with
        NO dedup, so DEPT appeared twice in ~C and re-read inflated the
        curve count (2→4) with phantom null-filled columns.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"

        section1 = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
            ],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        section2 = DataSection(
            name="LOG2",
            section_type="LOG_DATA",
            curves_order=["DEPT", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"),
            ],
            data={
                "DEPT": np.array([200.0, 210.0]),
                "DT": np.array([50.0, 55.0]),
            },
        )
        las.data_sections.append(section1)
        las.data_sections.append(section2)

        temp_file = tmp_path / "w01_dedup.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        # DEPT must appear exactly once in the ~C block.
        assert curve_block.count("DEPT.M") == 1, f"DEPT.M duplicated in ~C block: {curve_block!r}"

    def test_f28_core_only_metadata_only_curve_preserved(self, tmp_path: Path) -> None:
        """F-28: a metadata-only top-level curve survives in ~C for CORE-only files.

        When ALL data_sections are non-LOG_DATA with section_curves
        (curves_in_definitions=True), the section curves belong in the
        typed ~Core_Definition block — but a top-level curve with no data
        anywhere (e.g. TEMP) must still be written to ~C with a warning.
        Pre-fix the M-79 loop was gated on `not curves_in_definitions`
        and ~C was left empty, silently dropping the curve's metadata on
        re-read with zero warnings.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["CORE_DEPTH", "CORE_VAL"]
        las.curves.append(CurveDefinition(mnemonic="CORE_DEPTH", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="CORE_VAL", unit=""))
        las.curves.append(
            CurveDefinition(mnemonic="TEMP", unit="degC", description="metadata only")
        )

        section = DataSection(
            name="Core[1]",
            section_type="CORE_DATA",
            curves_order=["CORE_DEPTH", "CORE_VAL"],
            section_curves=[
                CurveDefinition(mnemonic="CORE_DEPTH", unit="M"),
                CurveDefinition(mnemonic="CORE_VAL", unit=""),
            ],
            data={
                "CORE_DEPTH": np.array([100.0]),
                "CORE_VAL": np.array([1.5]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "f28_core.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        m79 = [str(w.message) for w in rec if "definition but no data" in str(w.message)]
        assert len(m79) == 1, f"expected one M-79 warning for TEMP, got: {m79}"

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        # Section curves must NOT be duplicated into ~C (W-01).
        assert "CORE_DEPTH" not in curve_block, f"section curve duplicated in ~C: {curve_block!r}"
        # The metadata-only curve must survive in ~C.
        assert "TEMP.degC" in curve_block, f"metadata-only curve missing from ~C: {curve_block!r}"
        assert "~Core_Definition" in content

        back = read_las_file_as_object(out)
        back_curve = next((c for c in back.curves if c.mnemonic == "TEMP"), None)
        assert back_curve is not None, "TEMP metadata lost on re-read"
        assert back_curve.unit == "degC", f"TEMP unit lost: {back_curve.unit!r}"

    # --- N-I-16: 256-char line-limit gaps ---

    def test_ni16_las12_column_header_warns(self, tmp_path: Path) -> None:
        """N-I-16(a): the ~A column-header line is length-checked for LAS 1.2.

        Before the fix the ~A header was appended AFTER _warn_long_header_lines
        ran, so a 362-char column header produced 0 warnings while data rows
        warned.
        """
        las = LASFile()
        las.version = VersionSection(vers="1.2")
        las.well["NULL"] = "-999.25"
        names = [f"C{i:03d}" for i in range(60)]
        las.curves_order = names
        for n in names:
            las.curves.append(CurveDefinition(mnemonic=n, unit="M"))
            las.logs[n] = np.array([1.0])

        temp_file = tmp_path / "ni16_header.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            messages = [str(x.message) for x in w]
        assert any("column-header" in m for m in messages), (
            f"Expected ~A column-header warning, got: {messages[:5]}"
        )

    def test_ni16_las20_wrap_no_header_warns(self, tmp_path: Path) -> None:
        """N-I-16(b): LAS 2.0 WRAP=NO header lines are length-checked.

        Before the fix the header check was gated on is_las12 only, so a
        LAS 2.0 WRAP=NO file with a 312-char ~W description produced 0
        warnings despite the CWLS 256-char limit.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["COMP"] = "X" * 300
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "ni16_las20.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            messages = [str(x.message) for x in w]
        assert any("header line exceeds" in m for m in messages), (
            f"Expected LAS 2.0 WRAP=NO header warning, got: {messages[:5]}"
        )

    # --- N-I-17: string curve None/NaN guard ---

    def test_ni17_string_none_nan_written_as_sentinel(self, tmp_path: Path) -> None:
        """N-I-17: None/NaN in string data must not be written as literal
        "None"/"nan" strings — route to the '-' sentinel like the numeric
        branch routes non-finite to the null sentinel."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "WELLID"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="WELLID", unit="", data_format="S"))
        las.logs["DEPT"] = np.array([100.0, 101.0, 102.0])

        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "WELLID"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="WELLID", unit="", data_format="S"),
            ],
            data={"DEPT": np.array([100.0, 101.0, 102.0])},
            string_data={
                "WELLID": np.array([None, "A", np.nan], dtype=object),
            },
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "ni17_string_null.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        data_part = content.split("~A", 1)[1] if "~A" in content else ""
        # Literal "None"/"nan" must NOT leak into the data rows.
        for bad in ("None", "nan"):
            assert bad not in data_part, f"Literal {bad!r} leaked into data rows: {data_part!r}"

    # --- N-I-18: brace-token-in-description ---

    def test_ni18_user_brace_token_before_appended_format(self, tmp_path: Path) -> None:
        """N-I-18: a user description containing a format-brace token before
        the writer-appended format must not mis-extract data_format.

        Parser now prefers the TRAILING (writer-appended) format specifier;
        previously it took the FIRST match, so user text "{F}" with real
        data_format="E" roundtripped as data_format="F" (mis-extracted).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["GR"]
        las.curves.append(
            CurveDefinition(
                mnemonic="GR",
                unit="API",
                description="Gamma {F} log",
                data_format="E",
            )
        )
        las.logs["GR"] = np.array([75.0])

        temp_file = tmp_path / "ni18_brace.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        reread = read_las_file_as_object(temp_file)
        assert reread.curves[0].data_format == "E", (
            f"data_format mis-extracted from user brace token: {reread.curves[0].data_format!r}"
        )

    def test_ni18_non_format_brace_preserved_with_real_format(self, tmp_path: Path) -> None:
        """N-I-18: non-format brace text ({Density}) stays in the description
        while the writer-appended real format is extracted.  Previously the
        FIRST match ({Density}) failed validation and cleared data_format,
        losing the real format."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["RHOB"]
        las.curves.append(
            CurveDefinition(
                mnemonic="RHOB",
                unit="G/C3",
                description="Bulk {Density}",
                data_format="F",
            )
        )
        las.logs["RHOB"] = np.array([2550.0])

        temp_file = tmp_path / "ni18_density.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        reread = read_las_file_as_object(temp_file)
        assert reread.curves[0].data_format == "F", (
            f"real data_format lost: {reread.curves[0].data_format!r}"
        )
        assert "{Density}" in reread.curves[0].description, (
            f"user text {{Density}} lost: {reread.curves[0].description!r}"
        )

    # --- N-I-20: per-section curve-scope emission ---

    def test_ni20_per_section_scope_and_top_level_dedup(self, tmp_path: Path) -> None:
        """N-I-20: LOG_DATA sections with distinct curve sets keep their own
        scope on re-read AND their per-section Definitions do not inflate
        the top-level curve model.

        Before the fix every LOG_DATA section hardcoded ``| CURVE``, so
        both sections re-read scoped to the global union and section 2's
        columns were silently relabeled (DT → GR).  After the fix the
        writer emits per-section Definitions, but the parser registered
        every Definition curve into the global curves/curves_order with
        no cross-section dedup — the re-read top-level model inflated
        3→7 (['DEPT','GR','DT','DEPT','GR','DEPT','DT']).  The
        final-assembly dedup must collapse it back to the main ~C set
        while keeping per-section data scoping intact.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"

        section1 = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
            ],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        section2 = DataSection(
            name="LOG2",
            section_type="LOG_DATA",
            curves_order=["DEPT", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="DT", unit="US/M", data_format="F"),
            ],
            data={
                "DEPT": np.array([200.0, 210.0]),
                "DT": np.array([50.0, 55.0]),
            },
        )
        las.data_sections.append(section1)
        las.data_sections.append(section2)

        temp_file = tmp_path / "ni20_scope.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        content = temp_file.read_text(encoding="utf-8")
        # Each section must pipe to its own per-section Definition, not the
        # hardcoded ``| CURVE`` that destroyed per-section scope.
        assert "| Log_Definition" in content, "Section 1 missing per-section Definition pipe"
        assert "| Log_Definition_2" in content, "Section 2 missing per-section Definition pipe"

        reread = read_las_file_as_object(temp_file)
        # Top-level curves dedupe to the main ~C set.
        assert reread.curves_order == ["DEPT", "GR", "DT"], (
            f"Top-level curves_order inflated: {reread.curves_order}"
        )
        assert len(reread.curves) == 3, (
            f"Top-level curve definitions inflated: {len(reread.curves)}"
        )
        # Per-section scoping survives (the N-I-20 data-scoping guarantee).
        by_name = {ds.name: ds for ds in reread.data_sections}
        assert "LOG2" in by_name
        dt_vals = by_name["LOG2"].data.get("DT")
        assert dt_vals is not None
        np.testing.assert_array_almost_equal(
            np.asarray(dt_vals, dtype=float), np.array([50.0, 55.0])
        )
        gr_vals = by_name["LOG2"].data.get("GR")
        assert gr_vals is None, "LOG2 should not have a GR column after per-section scoping"


class TestExt04IntegerRoundtripWriter:
    """EXT-04 convergence: {I} curves with a fractional declared NULL must
    survive a write→read roundtrip exactly.

    Pre-fix, the writer rounded values above 2^53 through float64
    (9007199254740993 → '9007199254740992.00000000') and the LAS 1.2/2.0
    writer dropped the {I} marker, so re-read routed the column as float64
    and the exact value was permanently lost on both version paths.
    """

    _VALUE = 9007199254740993  # 2^53 + 1 — not representable in float64

    def _make_las(self, vers: str) -> LASFile:
        las = LASFile()
        las.version = VersionSection(vers=vers, wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT", "RUN_NO"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="RUN_NO", unit="", data_format="I"))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["RUN_NO"] = np.array([self._VALUE, -999.25], dtype=object)
        return las

    @pytest.mark.parametrize("vers", ["1.2", "2.0", "3.0"])
    def test_integer_precision_write_read_roundtrip_exact(self, tmp_path: Path, vers: str) -> None:
        """Write a {I} curve with fractional NULL, re-read, and assert the
        >2^53 value survives exactly and the null cell keeps -999.25.

        Parametrized over LAS 1.2/2.0/3.0 — the 1.2 writer dropped the
        {I} marker pre-fix just like 2.0 (class docstring), so the
        >2^53 precision-loss regression is pinned on every version path.
        """
        las = self._make_las(vers)
        out = tmp_path / f"ext04_rt_{vers}.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # The {I} marker must be emitted even on LAS 1.2/2.0 — without it
        # re-read routes the column as float64 and rounds >2^53 values.
        assert "{I}" in content, "'{I}' marker missing from " + vers + " output"
        # The >2^53 value must be written as its exact decimal.
        assert "9007199254740993" in content, f"written value was rounded: {content!r}"

        reread = read_las_file_as_object(out)
        if vers == "3.0":
            arr = reread.data_sections[0].data["RUN_NO"]
        else:
            arr = reread.logs["RUN_NO"]
        assert int(arr[0]) == self._VALUE, f"write→read rounded {self._VALUE}: {arr[0]}"
        assert arr[1] == -999.25, "null cell must keep the fractional sentinel"


class TestWriterFixBatchW010708101112:
    """Regression tests for the WRITER fix batch (Stage 8).

    Each test covers one CONFIRMED finding:
      W-01  _format_fixed_precision silently zeroes abs<1e-100
      W-07  LAS 1.2 writer strips leading ~ from well values
      W-08  ~O section sanitization is silent (leading ~, tabs)
      W-10  LAS 1.2/2.0 ~C emits duplicate mnemonics with no dedup
      W-11  undefined section curve silently relabels data
      W-12  emitted-mnemonic dedup collision silently discards data
      I2-13 post-construction curves_order mutation → silent column swap
      I2-20 LAS 3.0 no-data_sections path lacks ~C dedup
      I2-21 per-section Definition re-emits duplicate mnemonics
      I2-22 lowercase curves_order dropped by exact-case resolution
    """

    # ── W-01: tiny-value preservation ────────────────────────────────

    def test_w01_tiny_value_not_zeroed(self) -> None:
        """W-01: _format_fixed_precision must preserve abs<1e-100 values.

        Pre-fix, format(1e-150, '.100f') produced an all-zero string
        (100-char cap), silently zeroing the value on write→read.
        """
        result = _format_number(1e-150)
        assert "e" in result.lower(), f"expected scientific notation, got {result!r}"
        assert float(result) == 1e-150, f"value lost: {result!r}"

    def test_w01_tiny_value_write_read_roundtrip(self, tmp_path: Path) -> None:
        """W-01: a 1e-150 data cell survives write→read (not 0.0)."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT", "TINY"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="TINY", unit=""))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["TINY"] = np.array([1e-150, 2.5])

        out = tmp_path / "w01_tiny.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["TINY"], [1e-150, 2.5])

    def test_w01_null_tiny_value_semantics(self, tmp_path: Path) -> None:
        """W-01: a declared NULL of 1e-150 keeps null semantics.

        The null sentinel must be written as 1e-150 (not 0.0) so a data
        cell equal to the null reads back as the sentinel.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "1e-150"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT", "TINY"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="TINY", unit=""))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        # second cell equals the declared NULL
        las.logs["TINY"] = np.array([1e-150, 1e-150])

        out = tmp_path / "w01_null.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        assert back.well["NULL"] == "1e-150", back.well["NULL"]
        np.testing.assert_allclose(back.logs["TINY"], [1e-150, 1e-150])

    # ── W-07: LAS 1.2 leading-tilde preservation ─────────────────────

    def test_w07_las12_preserves_leading_tilde(self, tmp_path: Path) -> None:
        """W-07: LAS 1.2 well values keep a leading '~' on write→read.

        Pre-fix the LAS 1.2 well writer called _sanitize_las_value without
        preserve_leading_tilde, so WELL='~INCIDENTAL' was written as
        'INCIDENTAL' and the model value was silently corrupted.
        """
        las = LASFile()
        las.version = VersionSection(vers="1.2")
        las.well["NULL"] = "-999.25"
        las.well["WELL"] = "~INCIDENTAL"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        out = tmp_path / "w07_tilde.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        assert back.well["WELL"] == "~INCIDENTAL", f"leading tilde stripped: {back.well['WELL']!r}"

    # ── W-08: ~O sanitization warnings ───────────────────────────────

    def test_w08_other_section_tilde_strip_warns(self, tmp_path: Path) -> None:
        """W-08: ~O lines starting with ~[A-Za-z] warn when the leading ~
        is stripped (the parser would misread the line as a section
        header)."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.other = "~CURVEISH content\n"

        out = tmp_path / "w08_tilde.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("starts with '~' followed by a letter" in str(w.message) for w in rec), (
            f"no leading-tilde warning: {[str(w.message) for w in rec]}"
        )

    def test_w08_other_section_tab_transform_warns(self, tmp_path: Path) -> None:
        """W-08: ~O lines containing tabs warn that tabs become spaces."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.other = "plain\ttab here\n"

        out = tmp_path / "w08_tab.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("contains tab characters" in str(w.message) for w in rec), (
            f"no tab warning: {[str(w.message) for w in rec]}"
        )

    # ── W-10: 1.2/2.0 ~C duplicate-mnemonic dedup ────────────────────

    def test_w10_duplicate_curve_definition_deduped_with_warning(self, tmp_path: Path) -> None:
        """W-10: duplicate DEPT(M)/DEPT(FT) definitions emit DEPT once in
        ~C with a warning — re-read does not rename DEPT→DEPT_2."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="FT"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        out = tmp_path / "w10_dup.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("Duplicate curve mnemonic" in str(w.message) for w in rec), (
            f"no dedup warning: {[str(w.message) for w in rec]}"
        )

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert curve_block.count("DEPT.M") == 1, f"DEPT duplicated in ~C: {curve_block!r}"
        back = read_las_file_as_object(out)
        assert back.curves_order == ["DEPT"], f"re-read renamed a duplicate: {back.curves_order}"

    # ── W-11: undefined section curve must not relabel data ──────────

    def test_w11_undefined_curve_data_dropped_not_relabeled(self, tmp_path: Path) -> None:
        """E-30: section [DEPT, X, GR] with undefined X that CARRIES data
        is REFUSED with LASWriteError.

        Pre-fix the writer emitted X's column against a 2-curve scope, so
        X's values landed in GR and the genuine GR data was discarded; the
        warn-only interim fix dropped X's column, but the ~C-side promise
        is a REFUSAL (E-30) — a data-bearing column with no definition
        cannot be represented and the write must not proceed.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "X", "GR"],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "X": np.array([1.0, 2.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "w11_undef.las"
        with pytest.raises(LASWriteError, match=r"'X' .* carries data"):
            write_las_file(out, las)

    def test_w11_undefined_curve_no_data_dropped(self, tmp_path: Path) -> None:
        """W-11: an undefined curve WITHOUT data is dropped with a
        warning and no data is lost."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "X", "GR"],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "w11_undef2.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("'X'" in str(w.message) and "no definition" in str(w.message) for w in rec), (
            f"no drop warning: {[str(w.message) for w in rec]}"
        )

        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["GR"], [75.0, 80.0])
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 110.0])

    # ── W-12: emitted-mnemonic collision preserves both columns ──────

    def test_w12_collision_both_columns_preserved(self, tmp_path: Path) -> None:
        """W-12: LLD + BFV(original_mnemonic='LLD') both with data — the
        writer falls back to the colliding curve's own mnemonic so BOTH
        columns' values survive write→read (no silent discard).

        Pre-fix the pipe target deduped to one LLD while the data carried
        two columns, so BFV's values were silently discarded.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["LLD", "BFV"]
        las.curves.append(CurveDefinition(mnemonic="LLD", unit="OHMM"))
        las.curves.append(CurveDefinition(mnemonic="BFV", unit="OHMM", original_mnemonic="LLD"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["LLD", "BFV"],
            section_curves=[
                CurveDefinition(mnemonic="LLD", unit="OHMM"),
                CurveDefinition(mnemonic="BFV", unit="OHMM", original_mnemonic="LLD"),
            ],
            data={"LLD": np.array([10.0, 11.0]), "BFV": np.array([20.0, 21.0])},
        )
        las.data_sections.append(section)

        out = tmp_path / "w12_collision.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert curve_block.count("LLD.OHMM") == 1, f"duplicate LLD emitted in ~C: {curve_block!r}"

        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        values = {k: list(v) for k, v in ds.data.items()}
        flat = [round(x, 2) for arr in values.values() for x in arr]
        assert sorted(flat) == [10.0, 11.0, 20.0, 21.0], (
            f"colliding curve's data silently discarded: {values}"
        )

    # ── I2-13: post-construction curves_order mutation ───────────────

    def test_i213_curves_order_mutation_no_column_swap(self, tmp_path: Path) -> None:
        """I2-13: reordering curves_order after construction must not
        silently swap data columns — the writer emits in the LIVE order
        with a matching per-section Definition.

        Pre-fix the scoping used the cached section_curves order while the
        data rows followed the mutated curves_order, swapping GR/DEPT on
        re-read.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data={"DEPT": np.array([1.0, 2.0]), "GR": np.array([10.0, 11.0])},
        )
        las.data_sections.append(section)
        # POST-CONSTRUCTION mutation: swap the live column order.
        section.curves_order = ["GR", "DEPT"]

        out = tmp_path / "i213_swap.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [1.0, 2.0], err_msg="DEPT swapped")
        np.testing.assert_allclose(ds.data["GR"], [10.0, 11.0], err_msg="GR swapped")

    # ── PF-22: top-level legacy ~A path curves_order mutation ────────

    @pytest.mark.parametrize("vers", ["2.0", "3.0"])
    def test_pf22_top_level_reorder_write_read_preserves_columns(
        self, tmp_path: Path, vers: str
    ) -> None:
        """PF-22: I2-13 covered the data_sections path but NOT the
        top-level legacy ~A path.  Reordering curves_order AFTER
        construction must not silently swap columns on write→read — ~C
        must emit in the same LIVE order the data rows use.  Runs for
        LAS 2.0 and for LAS 3.0 without data_sections (the same legacy
        ~A fallback path).

        Pre-fix ~C emitted from the cached `curves` list (DEPT, GR) while
        the ~A rows followed the mutated curves_order (GR, DEPT); the
        re-read mapped data positionally per ~C and swapped the columns
        with no writer-side signal.
        """
        las = LASFile(
            version=VersionSection(vers=vers, wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="FT"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0]),
                "GR": np.array([50.0, 60.0]),
            },
        )
        las.well["NULL"] = "-999.25"
        # POST-CONSTRUCTION mutation: reverse the live column order.
        las.curves_order = ["GR", "DEPT"]

        out = tmp_path / f"pf22_{vers.replace('.', '')}_reorder.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        # ~C must agree with the data rows: GR emitted first, DEPT second.
        assert [c.mnemonic for c in back.curves] == ["GR", "DEPT"], (
            f"~C order changed: {[c.mnemonic for c in back.curves]}"
        )
        np.testing.assert_allclose(
            back.logs["DEPT"], [100.0, 101.0], err_msg="DEPT holds GR's values (swapped)"
        )
        np.testing.assert_allclose(
            back.logs["GR"], [50.0, 60.0], err_msg="GR holds DEPT's values (swapped)"
        )

    def test_pf22_unmutated_model_roundtrips_identically(self, tmp_path: Path) -> None:
        """PF-22 control: a model with curves_order aligned to curves
        (no post-construction mutation) must roundtrip identically — no
        reorder, no value shift, no warning — so the live-order fix does
        not regress the normal path."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="FT"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0]),
                "GR": np.array([50.0, 60.0]),
            },
        )
        las.well["NULL"] = "-999.25"

        out = tmp_path / "pf22_control.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        # No dedup/reorder/mismatch writer warnings for an aligned model.
        assert not any(
            "Duplicate curve mnemonic" in str(w.message) or "does not match" in str(w.message)
            for w in rec
        ), f"unexpected writer warnings: {[str(w.message) for w in rec]}"
        back = read_las_file_as_object(out)
        assert list(back.curves_order) == ["DEPT", "GR"]
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(back.logs["GR"], [50.0, 60.0])

    def test_pf22_single_section_copy_back_section_reorder_write_read(self, tmp_path: Path) -> None:
        """PF-22 (W-05 variant): LAS 2.0 with a single data_section and
        EMPTY top-level curves — the ~C definitions come from the
        section's section_curves via copy-back.  Reordering the SECTION's
        curves_order after construction must not silently swap columns:
        ~C must follow the same live order the ~A data rows use."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves=[],
        )
        las.well["NULL"] = "-999.25"
        section = DataSection(
            name="CURVE",
            curves_order=["DEPT", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", description="Depth"),
                CurveDefinition(mnemonic="DT", unit="US/M", description="Delta-T"),
            ],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([50.0, 51.0]),
            },
        )
        las.data_sections.append(section)
        # POST-CONSTRUCTION mutation: reverse the section's live order.
        section.curves_order = ["DT", "DEPT"]

        out = tmp_path / "pf22_w05_reorder.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        np.testing.assert_allclose(
            back.logs["DEPT"], [100.0, 101.0], err_msg="DEPT holds DT's values (swapped)"
        )
        np.testing.assert_allclose(
            back.logs["DT"], [50.0, 51.0], err_msg="DT holds DEPT's values (swapped)"
        )

    # ── I2-20: LAS 3.0 no-data_sections ~C dedup ─────────────────────

    def test_i220_no_data_sections_duplicate_deduped(self, tmp_path: Path) -> None:
        """I2-20: the LAS 3.0 no-data_sections path dedups duplicate
        curve mnemonics with a warning — no silent rename on re-read.

        Pre-fix the else branch emitted both duplicate lines and re-read
        renamed the second (LLD → LLD_2).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.curves_order = ["LLD"]
        las.curves.append(CurveDefinition(mnemonic="LLD", unit="OHMM"))
        # post-construction duplicate definition
        las.curves.append(CurveDefinition(mnemonic="LLD", unit="OHMM"))
        las.logs["LLD"] = np.array([10.0, 11.0])

        out = tmp_path / "i220_dup.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("Duplicate curve mnemonic" in str(w.message) for w in rec), (
            f"no dedup warning: {[str(w.message) for w in rec]}"
        )

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert curve_block.count("LLD.OHMM") == 1, f"LLD duplicated in ~C: {curve_block!r}"
        back = read_las_file_as_object(out)
        assert back.curves_order == ["LLD"], f"re-read renamed a duplicate: {back.curves_order}"

    # ── I2-21: Definition block dedup ────────────────────────────────

    def test_i221_definition_block_no_duplicate_mnemonics(self, tmp_path: Path) -> None:
        """I2-21: a section whose curves collide on the emitted mnemonic
        (LLD + BFV with original_mnemonic='LLD') must not emit duplicate
        mnemonic lines in its Definition block.

        Pre-fix the F-16 dedup was applied to the scoping identity but
        not the Definition emission, producing a structurally invalid
        duplicate-mnemonic Definition.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["LLD", "BFV"],
            section_curves=[
                CurveDefinition(mnemonic="LLD", unit="OHMM"),
                CurveDefinition(mnemonic="BFV", unit="OHMM", original_mnemonic="LLD"),
            ],
            data={"LLD": np.array([10.0])},
        )
        las.data_sections.append(section)

        out = tmp_path / "i221_def.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        idx = content.find("~Log_Definition")
        assert idx >= 0, "expected a per-section Definition block"
        end = content.find("~A", idx)
        block = content[idx:end]
        assert block.count("LLD.OHMM") == 1, f"duplicate mnemonic in Definition block: {block!r}"

    # ── I2-22: case-insensitive curve resolution ─────────────────────

    def test_i222_lowercase_curves_order_resolved(self, tmp_path: Path) -> None:
        """I2-22: a lowercase 'dept' in curves_order resolves
        case-insensitively — GR keeps its genuine values.

        Pre-fix the exact-case resolution dropped 'dept', so the section
        scoped to one curve (GR) and dept's data was relabeled into GR.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["dept", "GR"],
            data={
                "dept": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "i222_lower.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["GR"], [75.0, 80.0], err_msg="GR relabeled")
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 110.0])

    # ── PF-21: case-insensitive ~C fallback and M-79 loops ───────────

    def test_pf21_lowercase_curves_order_no_false_warnings(self, tmp_path: Path) -> None:
        """PF-21: a section WITH section_curves whose curves_order is
        lowercase ('dept') must not trigger the false N-I-15 ("has no
        definition") or false M-79 ("definition but no data") warnings,
        and must not re-emit DEPT a second time in ~C.

        Pre-fix the fallback/M-79 loops used exact-case lookups
        (curves_by_mnem / _section_mnems), so 'dept' did not resolve:
        the M-79 loop treated top-level DEPT as data-free and appended a
        duplicate DEPT.M line after the section's dept.M, with a false
        "definition but no data" warning.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["dept", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="dept", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data={
                "dept": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "pf21_with_sc.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        false_n115 = [str(w.message) for w in rec if "has no definition" in str(w.message)]
        false_m79 = [str(w.message) for w in rec if "definition but no data" in str(w.message)]
        assert false_n115 == [], f"false N-I-15 warnings: {false_n115}"
        assert false_m79 == [], f"false M-79 warnings: {false_m79}"

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        # DEPT must appear exactly once (case-insensitive) in ~C.
        dept_count = sum(1 for line in curve_block.splitlines() if "dept" in line.lower())
        assert dept_count == 1, f"DEPT emitted {dept_count} times: {curve_block!r}"
        # Order must match the model (dept/GR, not GR then DEPT).
        dept_idx = next(
            i for i, line in enumerate(curve_block.splitlines()) if "dept" in line.lower()
        )
        gr_idx = next(i for i, line in enumerate(curve_block.splitlines()) if "GR" in line)
        assert dept_idx < gr_idx, f"~C order swapped: {curve_block!r}"

        back = read_las_file_as_object(out)
        assert back.curves_order == ["DEPT", "GR"], (
            f"re-read top-level order changed: {back.curves_order}"
        )
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 110.0])
        np.testing.assert_allclose(ds.data["GR"], [75.0, 80.0])

    @pytest.mark.parametrize("case", ["lower", "upper"])
    def test_pf21_no_section_curves_single_emission(self, tmp_path: Path, case: str) -> None:
        """PF-21: a section WITHOUT section_curves emits each curve
        exactly once in ~C regardless of curves_order casing.

        Pre-fix the fallback loop's exact-case curves_by_mnem failed to
        resolve lowercase entries, so curves_to_emit fell back to the
        full top-level list without updating emitted_mnems, and the M-79
        loop then re-emitted the whole set (DEPT,GR,RHOB,DEPT,GR,RHOB) —
        structurally invalid LAS 3.0.  The 'upper' case is the control:
        uppercase curves_order is unaffected by the case-insensitive
        resolution — no new warnings, no duplicate emission, order
        preserved.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        if case == "lower":
            las.curves_order = ["DEPT", "GR", "RHOB"]
            las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
            las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
            las.curves.append(CurveDefinition(mnemonic="RHOB", unit="G/C3"))
            section = DataSection(
                name="CORE",
                section_type="CORE_DATA",
                curves_order=["dept", "rhob"],
                data={
                    "dept": np.array([100.0, 110.0]),
                    "rhob": np.array([2.3, 2.4]),
                },
            )
            # GR has no data anywhere → a legitimate M-79 warning fires;
            # only the false N-I-15 warnings must be absent.
            expect_m79_empty = False
            expected_order: list[str] | None = None
            expected_data = {"DEPT": [100.0, 110.0], "RHOB": [2.3, 2.4]}
        else:
            las.curves_order = ["DEPT", "GR"]
            las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
            las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
            section = DataSection(
                name="LOG",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={
                    "DEPT": np.array([100.0, 110.0]),
                    "GR": np.array([75.0, 80.0]),
                },
            )
            expect_m79_empty = True
            expected_order = ["DEPT", "GR"]
            expected_data = {"DEPT": [100.0, 110.0], "GR": [75.0, 80.0]}
        las.data_sections.append(section)

        out = tmp_path / f"pf21_{case}.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        false_n115 = [str(w.message) for w in rec if "has no definition" in str(w.message)]
        assert false_n115 == [], f"false N-I-15 warnings: {false_n115}"
        if expect_m79_empty:
            false_m79 = [str(w.message) for w in rec if "definition but no data" in str(w.message)]
            assert false_m79 == [], f"false M-79 warnings: {false_m79}"

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        for needle in expected_data:
            count = sum(1 for line in curve_block.splitlines() if needle.lower() in line.lower())
            assert count == 1, f"{needle} emitted {count} times: {curve_block!r}"

        back = read_las_file_as_object(out)
        if expected_order is not None:
            assert back.curves_order == expected_order, (
                f"re-read top-level order changed: {back.curves_order}"
            )
        ds = back.data_sections[0]
        for key, vals in expected_data.items():
            np.testing.assert_allclose(ds.data[key], vals)

    # ── F-31: case-variant duplicate resolution must be FIRST-wins ─────

    def test_f31_case_variant_duplicate_first_wins(self, tmp_path: Path) -> None:
        """F-31: case-variant duplicate mnemonics resolve FIRST-wins.

        _section_emission_pairs built by_upper as a LAST-wins dict
        comprehension while the sibling _effective_section_curves uses
        FIRST-wins setdefault.  With section_curves=[DEPT/M, dept/FT] and
        curves_order=['DEPT','dept'], the FIRST entry resolved to the WRONG
        definition (dept/FT) pre-fix — the written ~Log_Definition declared
        the first column dept.FT and re-read silently re-attributed the
        DEPT data column to the FT-unit curve.  Post-fix the first entry
        resolves to DEPT/M.

        N2b-2 update: the case-normalization fix upper-cases the
        ``emitted_mnems`` dedup in _write_curve_section, so the case-variant
        duplicate ('dept') is now deduped against 'DEPT' in the main ~C
        block with the W-01 differing-definition warning — the section pipes
        ``| CURVE`` (its effective curve set matches the main block) instead
        of getting a per-section ~Log_Definition.  The FIRST-wins contract
        is preserved: the surviving definition is DEPT/M, the roundtrip
        keeps unit M and the data, and re-read does NOT rename the second
        curve to DEPT_2 (the model-identity corruption N2b-2 prevents).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "dept"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="dept", unit="FT", data_format="F"),
            ],
            data={
                "DEPT": np.array([100.0, 110.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "f31_case.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        first_col = next((ln for ln in curve_block.splitlines() if ln.strip()), "")
        # The first column's definition must be DEPT.M — not dept.FT.
        assert "DEPT.M" in first_col, f"first column definition wrong: {first_col!r}"
        # N2b-2: the case-variant duplicate is deduped with the W-01
        # differing-definition warning, never emitted twice.
        dedup_warns = [str(w.message) for w in rec if "Duplicate curve mnemonic" in str(w.message)]
        assert dedup_warns, "expected the W-01 case-variant dedup warning"

        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 110.0], err_msg="DEPT data corrupted")
        # The surviving section curve keeps the FIRST definition's unit (M).
        assert ds.section_curves[0].unit == "M", (
            f"section curve unit corrupted: {ds.section_curves[0].unit!r}"
        )
        # N2b-2: re-read must NOT rename the duplicate to DEPT_2 — the
        # model identity is preserved.
        assert not any("_2" in c for c in back.curves_order), (
            f"re-read renamed a duplicate curve: {back.curves_order}"
        )


# ── F-26/F-27/F-32/F-36: writer-base findings ────────────────────────


class TestFixWriterBaseF26273236:
    """Regression tests for writer-base findings:
    F-26  metadata-only curve → fabricated null column on re-read
    F-27  dedup key diverges from emitted mnemonic on LAS 1.2/2.0
    F-32  case-variant curves_order data-key lookup (LAS 3.0 + legacy)
    F-36  empty curves_order + populated logs → silent full data loss
    """

    def test_f26_metadata_only_curve_divergence_warns(self, tmp_path: Path) -> None:
        """F-26: a metadata-only curve (in curves, absent from
        curves_order — legal per models.py:2858-2862) is emitted to ~C
        but gets no ~A column; the write must warn that the ~C/~A column
        counts diverge so re-read will fabricate a null-padded column.

        Pre-fix the write was silent; re-read fabricated
        METADATA=[-999.25,-999.25] from a file whose ~A never declared it.
        """
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
                CurveDefinition(mnemonic="METADATA", unit=""),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.well["NULL"] = "-999.25"

        out = tmp_path / "f26_metadata.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("have no data column" in str(w.message) for w in rec), (
            f"no ~C/~A divergence warning: {[str(w.message) for w in rec]}"
        )

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert curve_block.count("METADATA.") == 1, f"METADATA missing from ~C: {curve_block!r}"
        header_line = next(ln for ln in content.splitlines() if ln.startswith("~A"))
        # "~A DEPT GR" → 3 tokens, i.e. 2 data columns for 3 ~C curves.
        assert len(header_line.split()) == 3, f"~A should have 2 columns: {header_line!r}"

    def test_f27_legacy_array_curve_dedup_no_duplicate_lines(self, tmp_path: Path) -> None:
        """F-27: a reader-renamed array curve (IK_2 with
        original_mnemonic='IK' + array_info) on LAS 2.0 must NOT emit two
        identical IK lines in ~C.

        Pre-fix the dedup key appended [N] unconditionally (IK vs IK[1])
        while the emitter wrote IK without the bracket on LAS 1.2/2.0, so
        both curves emitted 'IK' — duplicate ~C lines, structurally
        invalid, silently written.
        """
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["IK", "IK_2"],
            curves=[
                CurveDefinition(mnemonic="IK", unit="OHMM"),
                CurveDefinition(
                    mnemonic="IK_2",
                    unit="OHMM",
                    original_mnemonic="IK",
                    data_format="A",
                    array_info=ArrayElementInfo(base_name="IK", index=1, time_offset=0.0),
                ),
            ],
            logs={
                "IK": np.array([1.0, 2.0]),
                "IK_2": np.array([10.0, 11.0]),
            },
        )
        las.well["NULL"] = "-999.25"

        out = tmp_path / "f27_array_dedup.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert curve_block.count("IK.OHMM") == 1, f"duplicate IK lines in ~C: {curve_block!r}"
        assert "IK_2.OHMM" in curve_block, f"IK_2 missing from ~C: {curve_block!r}"

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["IK"], [1.0, 2.0])
        np.testing.assert_allclose(back.logs["IK_2"], [10.0, 11.0])

    def test_wl_m1_las30_case_variant_section_validate_no_false_warnings(
        self, tmp_path: Path
    ) -> None:
        """WL-M1 (M13 residual, models side): DataSection.validate's
        uncovered + orphaned checks were exact-case, so on the M13 state
        (curves_order=['dept','GR'], data keyed DEPT/GR) write() emitted
        FALSE 'will pad with null_value' and 'will not emit these columns'
        diagnostics even though the data IS emitted and preserved.  No
        pad/orphaned warning of ANY variant (models.py has no 'for
        section' suffix — the M13 test's narrow discriminator missed it)
        may fire, and the data must survive write→re-read."""
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
        )
        las.well["NULL"] = "-999.25"
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)
        section.curves_order = ["dept", "GR"]

        out = tmp_path / "wl_m1_30_no_false_diags.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(out, las)

        false_diags = [
            str(w.message)
            for w in caught
            if "will pad" in str(w.message) or "will not emit" in str(w.message)
        ]
        assert false_diags == [], (
            f"false pad/orphaned diagnostics fired for case-variant entry: {false_diags}"
        )

        back = read_las_file_as_object(out)
        ds = back.data_sections[0]
        np.testing.assert_allclose(
            ds.data["DEPT"], [100.0, 110.0], err_msg="DEPT values null-filled"
        )
        np.testing.assert_allclose(ds.data["GR"], [75.0, 80.0])

    def test_wl_m1_las30_section_genuinely_uncovered_curve_still_warns(
        self, tmp_path: Path
    ) -> None:
        """WL-M1 true-positive preservation (WL-L2 concern): the
        case-insensitive comparison must NOT mask a genuinely-uncovered
        curve.  A section whose curves_order names a curve with no data
        anywhere must still emit the 'will pad with null_value' warning."""
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
        )
        las.well["NULL"] = "-999.25"
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([100.0, 110.0])},  # GR genuinely uncovered
        )
        las.data_sections.append(section)

        out = tmp_path / "wl_m1_30_true_positive.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(out, las)

        pads = [
            str(w.message)
            for w in caught
            if "will pad" in str(w.message) and "GR" in str(w.message)
        ]
        assert pads, "genuinely-uncovered curve GR produced no 'will pad' warning"

    def test_f32_legacy_lowercase_curves_order_emits_data(self, tmp_path: Path) -> None:
        """F-32 (legacy): a post-construction lowercase curves_order entry
        ('dept') whose data is stored under the uppercase key ('DEPT')
        must still emit the ~A section.  Pre-fix the exact-case gate
        concluded 'none have data', skipped ~A entirely, and the data was
        silently dropped.
        """
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
            logs={"DEPT": np.array([100.0, 101.0])},
        )
        las.well["NULL"] = "-999.25"
        # POST-CONSTRUCTION mutation: lowercase the order entry.
        las.curves_order = ["dept"]

        out = tmp_path / "f32_legacy_case.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        assert any(ln.startswith("~A") for ln in content.splitlines()), (
            "~A section skipped for case-variant curves_order entry"
        )
        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0], err_msg="DEPT data lost")

    def test_f36_empty_curves_order_with_logs_warns(self, tmp_path: Path) -> None:
        """F-36: an EMPTY curves_order with populated logs is a
        direct-construction-only inconsistent state — the write must warn
        that the data will not be emitted.  Pre-fix the empty-list
        short-circuit (`any([])` is False) suppressed every diagnostic and
        the data vanished silently with zero warnings.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "101.0"
        las.well["STEP"] = "1.0"
        las.logs["DEPT"] = np.array([100.0, 101.0])

        out = tmp_path / "f36_empty_order.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("curves_order is empty" in str(w.message) for w in rec), (
            f"no data-loss warning: {[str(w.message) for w in rec]}"
        )

        content = out.read_text(encoding="utf-8")
        assert not any(ln.startswith("~A") for ln in content.splitlines()), (
            "unexpected ~A section for empty curves_order"
        )
        back = read_las_file_as_object(out)
        assert len(back.logs) == 0, f"re-read fabricated data: {dict(back.logs)}"

    def test_mod2_top_level_case_variant_no_false_write_warnings(self, tmp_path: Path) -> None:
        """MOD-2 (WL-M1 class, LASFile top level): the validate() twins
        (:3756 desync, :3791 orphan) compared exact-case, so writing the
        post-construction case-variant state (curves_order=['dept','GR'],
        logs keyed DEPT/GR) emitted 4 false warnings (2 desync + 2 orphan)
        even though the writer resolves case-insensitively and the data is
        emitted byte-correct.  No desync/orphan warning of any variant may
        fire, and the data must survive write→re-read."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "102.0"
        las.well["STEP"] = "1.0"
        # POST-CONSTRUCTION mutation: lowercase the order entries.
        las.curves_order = ["dept", "GR"]

        out = tmp_path / "mod2_top_case.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(out, las)

        false_diags = [
            str(w.message)
            for w in caught
            if "desynced" in str(w.message) or "will not emit" in str(w.message)
        ]
        assert false_diags == [], f"false desync/orphan warnings: {false_diags}"

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(back.logs["GR"], [75.0, 76.0, 77.0])


# ──────────────────────────────────────────────────────────────
# MOD-3 (MEDIUM, F-04 residual): public dict-write API on the
# case-variant state.  The pass-3 fix blessed the state, but
# write_las_file(path, dict) → LASFile.from_dict hard-failed with
# LASWriteError ("Cannot create LASFile from dict: curves_order[0] =
# 'dept' does not match curves[0].mnemonic = 'DEPT'").
# ──────────────────────────────────────────────────────────────


class TestMOD3DictWriteCaseVariant:
    """F-04: write_las_file(path, dict) must accept the to_dict output of
    the supported case-variant state and emit byte-correct data (pre-fix:
    LASWriteError wrapping the from_dict rejection)."""

    def test_mod3_write_dict_case_variant_las20(self, tmp_path: Path) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "102.0"
        las.well["STEP"] = "1.0"

        out = tmp_path / "mod3_dict_case_variant.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las.to_dict())

        content = out.read_text(encoding="utf-8")
        assert "~A  DEPT  GR" in content, f"header not emitted: {content}"
        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(back.logs["GR"], [75.0, 76.0, 77.0])

    def test_mod3_write_dict_case_variant_las30(self, tmp_path: Path) -> None:
        ds = DataSection(
            name="Log1",
            curves_order=["dept", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "GR"],
            curves=[],
            data_sections=[ds],
        )
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "102.0"
        las.well["STEP"] = "1.0"

        out = tmp_path / "mod3_dict_case_variant_30.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las.to_dict())

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.data_sections[0].data["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(back.data_sections[0].data["GR"], [75.0, 76.0, 77.0])


# ── Case-normalization writer regression tests (N2b/II family) ─────────


class TestCaseNormalizationWriterRegression:
    """Writer-side regression tests for the case-normalization fixes.

    Each FAILS on the pre-fix code and PASSES after the ``_mnem_key``
    migration.  The writer's emission contract keeps original case
    (M-59 original_mnemonic reconstruction); matching is case-insensitive.
    """

    def test_n2b1_string_marker_preserved_case_variant_key(self, tmp_path: Path) -> None:
        """N2b-1: a LAS 3.0 string curve whose string_data key differs by
        case from the curve mnemonic must still get the {S} marker and
        round-trip its values.  Pre-fix the marker membership compared
        exact-case, so the emitted ~C line was markerless and the parser
        re-read the values as numeric nulls (silent destruction)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["DEPT_STR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT_STR", data_format=""))
        las.string_data["dept_str"] = np.array(["alpha", "beta", "gamma"], dtype=object)

        out = tmp_path / "n2b1_case.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert "{S}" in curve_block, f"{{S}} marker lost: {curve_block!r}"

        back = read_las_file_as_object(out)
        assert "DEPT_STR" in back.string_data, (
            f"string values destroyed on re-read: logs={dict(back.logs) if back.logs else None}"
        )
        assert list(back.string_data["DEPT_STR"]) == ["alpha", "beta", "gamma"]

    def test_n2b1_emitted_name_original_mnemonic_marker(self, tmp_path: Path) -> None:
        """N2b-1 (II-7): the {S} marker membership must test the EMITTED
        mnemonic (_emit_mnem / M-59 original_mnemonic reconstruction), NOT
        curve.mnemonic.  A curve BFV with original_mnemonic='LLD' emits
        'LLD' — with NO case variance — and pre-fix lost its marker and
        destroyed the string values."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["BFV"]
        las.curves.append(CurveDefinition(mnemonic="BFV", original_mnemonic="LLD", data_format=""))
        las.string_data["LLD"] = np.array(["x", "y"], dtype=object)
        las.curves_order = ["LLD"]  # M-59 reconstruction emits 'LLD'

        out = tmp_path / "n2b1_emit.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert "{S}" in curve_block, f"{{S}} marker lost for emitted name: {curve_block!r}"

        back = read_las_file_as_object(out)
        assert "LLD" in back.string_data, (
            f"emitted-name string values destroyed: logs={dict(back.logs) if back.logs else None}"
        )
        assert list(back.string_data["LLD"]) == ["x", "y"]

    def test_n2b2_case_variant_duplicate_distinct_data_refuses(self, tmp_path: Path) -> None:
        """N2b-2 (X-1): case-variant duplicate curves ('DEPT' + 'dept')
        with DISTINCT data cannot be represented in the legacy single-block
        format.  ~C dedups case-insensitively to one curve but ~A would
        still emit both columns, and re-read DISCARDS the second column's
        data ("Extra columns are discarded").  The write must REFUSE
        (LASWriteError) rather than silently lose the data — the pre-N2b-2
        outcome was rename-to-DEPT_2 with data preserved; the Stage-4
        CI-dedup changed it to silent discard, which this guard restores
        the no-loss guarantee for.

        Regression: pre-fix this test PASSED while the 'dept' values were
        discarded; it now asserts the refusal so the data is never lost."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        # Post-construction mutation — the construction-time duplicate
        # check (now case-insensitive) would reject the state; the writer
        # dedup must handle it anyway (N2b-2's reachability path).
        las.curves_order = ["DEPT", "dept"]
        las.curves.append(CurveDefinition(mnemonic="dept"))
        las.logs["DEPT"] = np.array([1.0, 2.0])
        las.logs["dept"] = np.array([10.0, 20.0])

        out = tmp_path / "n2b2_dup.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(LASWriteError) as excinfo:
                write_las_file(out, las)

        dedup_warns = [str(w.message) for w in rec if "Duplicate curve mnemonic" in str(w.message)]
        assert dedup_warns, "expected the W-01-class dedup warning"
        msg = str(excinfo.value)
        assert "same mnemonic" in msg and "has data" in msg, f"unclear refusal: {msg}"
        assert not out.exists(), "no file may be written when the write refuses"
        # No data may be silently lost: the write either refused (here) or
        # preserved the columns — it never writes a file that discards data.

    def test_x1_case_variant_duplicate_shared_data_warns_not_refuses(self, tmp_path: Path) -> None:
        """X-1: a case-variant duplicate whose data is SHARED with the
        surviving curve (data keyed only under 'DEPT', no 'dept' array)
        must NOT refuse — nothing is lost, so the write succeeds, the
        W-01 dedup warning fires, and the warning text must be ACCURATE
        (the stale 'a re-read would rename it' claim is gone post-fix)."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.curves_order = ["DEPT", "dept"]
        las.curves.append(CurveDefinition(mnemonic="dept"))
        las.logs["DEPT"] = np.array([1.0, 2.0])  # no 'dept' data — shared array

        out = tmp_path / "x1_shared.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)

        dedup_warns = [str(w.message) for w in rec if "Duplicate curve mnemonic" in str(w.message)]
        assert dedup_warns, "expected the W-01-class dedup warning"
        stale = [str(w.message) for w in rec if "a re-read would rename it" in str(w.message)]
        assert stale == [], f"stale warning text still present: {stale}"

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [1.0, 2.0], err_msg="shared data lost")

    def test_n2b3_w12_case_variant_shared_data_not_lost(self, tmp_path: Path) -> None:
        """N2b-3 (W-12): a colliding case-variant duplicate whose data is
        keyed under the SECOND curve's name ('y') is NOT lost — the
        FIRST-wins surviving pair emits it as its own column.  The write
        must succeed with the accurate 'no values are lost' warning (the
        data survives) and the re-read must preserve the values.

        Pre-fix the exact-case data-bearing check fired the false
        assurance OR over-strictly raised for the case-variant state; the
        fix resolves the colliding entry's data case-insensitively and
        only refuses (LASWriteError) when that data would actually be
        dropped (distinct array, see the W-11 sibling)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["Y", "y"],
            section_curves=[
                CurveDefinition(mnemonic="Y", unit="M", data_format="F"),
                CurveDefinition(mnemonic="y", unit="M", data_format="F"),
            ],
            data={
                "y": np.array([7.0, 8.0]),  # case-variant key of the colliding entry
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "n2b3_w12.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)

        false_assurance = [str(w.message) for w in rec if "no values are lost" in str(w.message)]
        # The message is ACCURATE here: the 'y' values survive as the 'Y'
        # column (first-wins resolution), so it must fire rather than a
        # LASWriteError or a silent drop.
        assert false_assurance, "expected the (accurate) no-values-lost warning"

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(
            back.data_sections[0].data["Y"], [7.0, 8.0], err_msg="data lost on roundtrip"
        )

    def test_n2b3_w11_case_variant_data_dropped_warns_data(self, tmp_path: Path) -> None:
        """E-30 (N2b-3 W-11): an unresolvable curve whose data sits under a
        case-variant key is REFUSED with LASWriteError — the data-bearing
        unresolved column can no longer be silently dropped (the pre-fix
        'DATA is dropped' warning left the data out of the file)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GHOST"],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "ghost": np.array([7.0, 8.0]),  # no definition; case-variant key
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "n2b3_w11.las"
        with pytest.raises(LASWriteError, match=r"'GHOST' .* carries data"):
            write_las_file(out, las)

    def test_n2b1_m77_warning_fires_for_emitted_name_loss(self, tmp_path: Path) -> None:
        """N2b-1 (II-7c) + M-35: a renamed curve (storage 'BFV', emitted
        'LLD') whose string_data is keyed 'BFV' MUST get the {S} marker.

        Pre-fix the M-77 membership test checked only the EMITTED name
        against string_data STORAGE keys — the renamed curve missed, the
        marker was NOT forced, and the string values were destroyed on
        write→read.  Post-M-35 the membership tests BOTH keys: {S} IS
        forced on the emitted 'LLD' line and the values round-trip.  The
        M-77 loss warning must NOT fire (no loss)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["BFV"]
        las.curves.append(CurveDefinition(mnemonic="BFV", original_mnemonic="LLD", data_format=""))
        las.string_data["BFV"] = np.array(["a", "b"], dtype=object)

        out = tmp_path / "n2b1_m77.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # The emitted 'LLD' line MUST carry the {S} marker — the M-35
        # both-key membership forces it (values are NOT lost).
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        lld_line = next((ln for ln in curve_block.splitlines() if "LLD" in ln), "")
        assert "{S}" in lld_line, f"{{S}} not forced for renamed curve: {lld_line!r}"
        m77 = [str(w.message) for w in rec if "Without the {S} marker" in str(w.message)]
        assert not m77, f"M-77 loss warning must NOT fire post-M-35: {m77}"
        # The string values must round-trip intact (pre-fix destroyed).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert list(back.string_data["LLD"]) == ["a", "b"], (
            f"renamed-curve string values destroyed: {back.string_data!r}"
        )

    @pytest.mark.parametrize("vers", ["2.0", "1.2"])
    def test_i20_well_units_descriptions_case_variant_preserved(self, tmp_path: Path, vers: str) -> None:
        """II-20: well units/descriptions lookups must match
        case-insensitively — a case-mismatched entries-vs-units pair must
        keep the unit in the emitted ~W line.  Pre-fix the exact-case
        .get(key) silently dropped the unit/description from the output.
        Runs for the base writer (2.0) and the LAS 1.2 well-section
        writer (X-3), which previously used its own exact-case lookup."""
        las = LASFile(
            version=VersionSection(vers=vers, wrap="NO", dlm="SPACE"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )
        las.well.entries["strt"] = "100"
        las.well.entries["stop"] = "102"
        las.well.entries["step"] = "1"
        las.well.entries["null"] = "-999.25"
        las.well.units["STRT"] = "m"  # case-variant vs entries key 'strt'
        las.well.descriptions["STRT"] = "START DEPTH"

        out = tmp_path / f"i20_well_{vers}.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        well_block = content.split("~WELL INFORMATION", 1)[1].split("~", 1)[0]
        strt_line = next((ln for ln in well_block.splitlines() if "strt" in ln.lower()), "")
        assert ".m" in strt_line, f"unit silently dropped: {strt_line!r}"
        assert "START DEPTH" in strt_line, f"description silently dropped: {strt_line!r}"


# ──────────────────────────────────────────────────────────────────────
# E-16 / E-19 / E-30 / E-31 / E-32 / E-36 / E-40 / E-44 / M-27 / M-29 /
# N-04 / N-09 — fix-writer regression tests (stage-7 verified findings)
# ──────────────────────────────────────────────────────────────────────


class TestFixWriterE16ZeroDimArrays:
    """E-16: a 0-d numpy array in logs/string_data must not crash the
    write — the M-18 convention treats it as a single-element value."""

    def test_e16_zero_dim_logs_array_writes_one_row(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array(100.0)  # 0-d array

        out = tmp_path / "e16_zerod.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        data_lines = [
            ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip() == "100"
        ]
        assert len(data_lines) == 1, f"0-d array must emit exactly 1 row, got: {data_lines}"

    def test_e16_zero_dim_string_data_writes_one_row(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["TAG"]
        las.curves.append(CurveDefinition(mnemonic="TAG", unit="", data_format="S"))
        las.string_data["TAG"] = np.array("sandstone")  # 0-d object array

        out = tmp_path / "e16_zerod_str.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        assert "sandstone" in content, "0-d string value not emitted"


class TestFixWriterE19TildeEscapeRestore:
    """E-19: the M-85 '_~' escape is emitted on ALL versions to keep the
    row parseable; the LAS 1.2/2.0 read path now restores it for the
    first-column string token (position-scoped, like the 3.0 path)."""

    def test_e19_las20_first_column_tilde_escape_keeps_row(self, tmp_path: Path) -> None:
        """A first-column string value '~3D' on LAS 2.0 is escaped to
        '_~3D' so the row is not misread as a section header; the row
        (and the numeric columns) survive the roundtrip, and the warning
        is ACCURATE (pre-fix it falsely promised a full restore — string
        values are lossy on 1.2/2.0 by design, M-29)."""
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["TAG", "DEPT"]
        las.curves.append(CurveDefinition(mnemonic="TAG", unit=""))
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.string_data["TAG"] = np.array(["~3D", "plain"], dtype=object)
        las.logs["DEPT"] = np.array([100.0, 101.0])

        out = tmp_path / "e19_las20.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # The escape is emitted (row preserved) — NOT the raw '~3D' line.
        assert "_~3D" in content, f"no '_~' escape emitted: {content!r}"
        # E-19: the warning must be accurate for 1.2/2.0 — the escape is
        # restored for a first-column string token, but string values do
        # not round-trip by design (M-29).
        assert any("do not round-trip by design" in str(w.message) for w in rec), (
            f"inaccurate tilde-escape warning: {[str(w.message) for w in rec]}"
        )
        # Re-read: the row is NOT dropped (pre-fix the reader skipped the
        # raw '~'-prefixed row entirely, losing the DEPT value).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0], err_msg="row dropped")

    def test_e19_las20_first_column_restore_position_scoped(self, tmp_path: Path) -> None:
        """A LAS 2.0 file with {S} markers: the writer's '_~' escape is
        restored ONLY for the first-column token ('_~3D' → '~3D'); a
        genuine '_~DEPT' first-column value (never writer-escaped —
        '~'+letter) and a genuine '_~3D' in a NON-first column survive
        verbatim."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " TAG.         : TAG  {S}\n"
            " LITHO.       : LITHOLOGY  {S}\n"
            " DEPT.M       : DEPTH\n"
            "~A  TAG  LITHO  DEPT\n"
            "_~3D,_~3D,100\n"
            "_~DEPT,plain,101\n"
        )
        f = tmp_path / "e19_restore.las"
        f.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(f)
        # First-column '_~3D' is the writer escape → restored to '~3D'.
        assert las.string_data["TAG"].tolist() == ["~3D", "_~DEPT"], (
            las.string_data["TAG"].tolist()
        )
        # Non-first-column '_~3D' is genuine content → preserved verbatim.
        assert las.string_data["LITHO"].tolist() == ["_~3D", "plain"], (
            las.string_data["LITHO"].tolist()
        )


class TestFixWriterE31WellKeyCIVariants:
    """E-31: case-variant duplicate well keys are deduped at emission —
    refused loudly when the values differ, warned when identical."""

    def test_e31_case_variant_well_keys_distinct_values_refused(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.well["null"] = "0"  # case-variant with a DIFFERENT value
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([1.0])

        out = tmp_path / "e31_refuse.las"
        with pytest.raises(LASWriteError, match="differ only in case"):
            write_las_file(out, las)

    def test_e31_case_variant_well_keys_identical_warn_single_emission(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.well["null"] = "-999.25"  # case-variant with the SAME value
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([1.0])

        out = tmp_path / "e31_warn.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("differ only in case" in str(w.message) for w in rec), (
            "no case-variant duplicate warning"
        )
        well_block = out.read_text(encoding="utf-8").split("~WELL INFORMATION", 1)[1].split("~", 1)[0]
        assert well_block.count("NULL.") == 1, f"duplicate ~W lines emitted: {well_block!r}"

    def test_e31_single_key_no_spurious_warning(self, tmp_path: Path) -> None:
        """A plain single-casing well entry must not trigger the E-31
        duplicate path (regression guard for the mandatory-order loop)."""
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([1.0])

        out = tmp_path / "e31_single.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert not any("differ only in case" in str(w.message) for w in rec), (
            f"spurious E-31 warning: {[str(w.message) for w in rec]}"
        )


class TestFixWriterE32DeletedKeyWarning:
    """E-32: a post-construction deletion of a logs/string_data key must
    warn loudly at emission instead of silently null-padding the orphaned
    curves_order column (re-read fabricates a -999.25 column)."""

    def test_e32_deleted_logs_key_warns(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        las.logs["DEPT"] = np.array([1.0])
        las.logs["GR"] = np.array([2.0])
        del las.logs["GR"]  # _GuardedDict has no deletion override

        out = tmp_path / "e32_deleted.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("'GR'" in str(w.message) and "no data in 'logs' or 'string_data'" in str(w.message)
                   for w in rec), f"no missing-data warning: {[str(w.message) for w in rec]}"


class TestFixWriterE36WellUnitValidation:
    """E-36: a well unit that cannot round-trip through the parser's ~W
    unit grammar (e.g. whitespace-colon 'kg : m') is refused at emission
    instead of truncating the unit and destroying the entry value."""

    def _model_with_unit(self, unit: str, vers: str = "2.0") -> LASFile:
        las = LASFile(version=VersionSection(vers=vers, wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.well.units["NULL"] = unit
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([1.0])
        return las

    @pytest.mark.parametrize("vers", ["2.0", "1.2"])
    def test_e36_whitespace_colon_unit_refused(self, tmp_path: Path, vers: str) -> None:
        with pytest.raises(LASWriteError, match="unit grammar"):
            write_las_file(
                tmp_path / f"e36_{vers.replace('.', '')}.las",
                self._model_with_unit("kg : m", vers=vers),
            )

    def test_e36_valid_colon_unit_still_writes(self, tmp_path: Path) -> None:
        """A colon-containing unit that IS in the parser's grammar
        ('kg:m') remains representable and writes fine."""
        las = self._model_with_unit("kg:m")
        out = tmp_path / "e36_ok.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        assert ".kg:m" in out.read_text(encoding="utf-8"), "valid colon unit not emitted"


class TestFixWriterE40OffsetCap:
    """E-40: {A:N} time_offset fields must fit the parser's 64-char
    offset group — offsets >= 1e64 (integral or non-integral) are
    refused instead of emitting a spec the parser drops (whole curve
    absent on re-read)."""

    def _model(self, offset: float) -> LASFile:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["NMR"]
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR",
                unit="",
                data_format="A",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=offset),
            )
        )
        las.logs["NMR"] = np.array([1.0])
        return las

    def test_e40_integral_offset_ge_1e64_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LASWriteError, match="cannot be represented in the \\{A:N\\}"):
            write_las_file(tmp_path / "e40_int.las", self._model(1e64))

    def test_e40_non_integral_huge_offset_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LASWriteError, match="cannot be represented in the \\{A:N\\}"):
            write_las_file(tmp_path / "e40_frac.las", self._model(1.5e100))

    def test_e40_representable_offset_still_emitted(self, tmp_path: Path) -> None:
        las = self._model(1e20)
        out = tmp_path / "e40_ok.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        assert "A:100000000000000000000" in out.read_text(encoding="utf-8"), (
            "representable {A:N} spec not emitted"
        )

    def test_e40_sub_1e61_offset_warns_and_clamps(self, tmp_path: Path) -> None:
        r"""E-40 (round+clamp branch): a sub-1e-61 time_offset cannot be
        represented exactly in the {A:N} offset field — the writer warns
        and CLAMPS the decimal places so the emitted spec stays within
        the parser's 64-char offset group.

        Pre-fix a 1e-70 offset would emit a 72-char fixed-point field;
        the parser's FORMAT_SPEC_PATTERN (``[-\d.]{0,64}``) fails to
        match and data_format + time_offset are silently lost on
        write→read (M-56 class).  The warning + clamp keeps the spec
        parseable (data_format survives) at the documented F-15/E-40
        trade-off (time_offset rounds to 0.0).
        """
        import re

        las = self._model(1e-70)
        out = tmp_path / "e40_clamp.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("cannot be represented exactly" in str(w.message) for w in rec), (
            f"no round+clamp warning: {[str(w.message) for w in rec]}"
        )

        # The clamped field must stay within the 64-char offset group.
        content = out.read_text(encoding="utf-8")
        specs = re.findall(r"\{A:([^}]*)\}", content)
        assert specs, f"no {{A:N}} spec emitted: {content!r}"
        assert all(len(spec) <= 64 for spec in specs), (
            f"offset field exceeds 64 chars: {[len(s) for s in specs]}"
        )

        # data_format must survive write→read (the M-56 silent-loss guard).
        back = read_las_file_as_object(out)
        assert back.curves[0].data_format == "A", (
            f"data_format lost after clamp: {back.curves[0].data_format!r}"
        )


class TestFixWriterE44EmptySectionWarning:
    """E-44: a LAS 3.0 data section with ZERO data rows is emitted
    header-only and its metadata is lost on re-read — warn loudly at
    write time."""

    def test_e44_zero_data_row_section_warns(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.data_sections.append(
            DataSection(
                name="LOG",
                section_type="LOG_DATA",
                curves_order=["DEPT"],
                data={"DEPT": np.array([])},
            )
        )
        out = tmp_path / "e44_empty.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        assert any("NO data rows" in str(w.message) for w in rec), (
            f"no empty-section warning: {[str(w.message) for w in rec]}"
        )


class TestFixWriterM27SingleValidation:
    """M-27: write() must run validate(complete=True) exactly ONCE —
    the _WriterMutationGuard re-validation double-warned every issue."""

    def test_m27_string_data_on_las20_warned_once(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "TAG"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="TAG", unit=""))
        las.logs["DEPT"] = np.array([1.0])
        las.string_data["TAG"] = np.array(["a"], dtype=object)

        out = tmp_path / "m27_once.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        m29 = [w for w in rec if "string_data is present" in str(w.message)]
        assert len(m29) == 1, f"expected exactly 1 M-29 validation warning, got {len(m29)}"


class TestFixWriterM29StringAwareDefinitionDedup:
    """M-29: the per-section Definition dedup signature must include
    string-ness — two sections with identical curve definitions but
    different string placement must get SEPARATE Definitions, or the
    shared {S} marker null-fills the other section's values (and the
    iter-3 variant produced a SELF-UNREADABLE file)."""

    def _two_section_model(self) -> LASFile:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format=""))
        las.data_sections.append(
            DataSection(
                name="LOG1",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 101.0])},
                string_data={"GR": np.array(["sand", "shale"], dtype=object)},
            )
        )
        las.data_sections.append(
            DataSection(
                name="LOG2",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([200.0, 201.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        return las

    def test_m29_string_and_numeric_sections_get_separate_definitions(
        self, tmp_path: Path,
    ) -> None:
        las = self._two_section_model()
        out = tmp_path / "m29_defs.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # H-01: the STRING LOG1 section gets its OWN ~Log_Definition; the
        # numeric LOG2 section scopes to the main ~C block (| CURVE).
        # Pre-fix the main ~C union-forced {S} from LOG1's string
        # placement, so the numeric section needed its own Definition to
        # avoid reclassifying GR as string.  Post-H-01 the main ~C no
        # longer carries the union-forced {S}, so the numeric section
        # pipes | CURVE and the string section gets the Definition.
        # Count the actual ~*_Definition HEADER, not total
        # "_Definition" occurrences — content.count("_Definition") == 2
        # was a coincidental 1 header + 1 pipe-target reference.
        assert content.count("~Log_Definition") == 1, (
            f"expected exactly 1 ~Log_Definition header (string LOG1 gets its own "
            f"Definition; numeric LOG2 pipes | CURVE), got "
            f"{content.count('~Log_Definition')}: "
            f"{[ln for ln in content.splitlines() if 'Definition' in ln]}"
        )
        assert "~A LOG1 | Log_Definition" in content, (
            f"string LOG1 must pipe to its own Definition, got: "
            f"{[ln for ln in content.splitlines() if ln.startswith('~A')]}"
        )
        assert "~A LOG2 | CURVE" in content, (
            f"numeric LOG2 must pipe | CURVE (main ~C is markerless), got: "
            f"{[ln for ln in content.splitlines() if ln.startswith('~A')]}"
        )
        # The write→read roundtrip MUST succeed (pre-fix it raised
        # LASParseError — self-unreadable file) and preserve each
        # section's placement.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert len(back.data_sections) == 2
        assert back.data_sections[0].string_data.get("GR", None) is not None, (
            "section 1 GR not string"
        )
        np.testing.assert_allclose(
            back.data_sections[1].data["GR"], [75.0, 80.0], err_msg="numeric GR null-filled"
        )


class TestFixWriterN04UnionForcedMarkerAware:
    """N-04: the main ~C union-forced {S} marker must not reclassify a
    NUMERIC section's column as string — a section whose curve is
    numeric here gets its own Definition instead of piping | CURVE."""

    def test_n04_numeric_section_not_reclassified_by_union_marker(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format=""))
        las.data_sections.append(
            DataSection(
                name="LOGA",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 101.0])},
                string_data={"GR": np.array(["sand", "shale"], dtype=object)},
            )
        )
        las.data_sections.append(
            DataSection(
                name="LOGB",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([200.0, 201.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        out = tmp_path / "n04_marker.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # H-01: the STRING LOGA section gets its OWN ~Log_Definition;
        # the numeric LOGB section pipes | CURVE (the main ~C no longer
        # carries the union-forced {S} that would reclassify its GR).
        assert "~A LOGA | Log_Definition" in content, content
        assert "~A LOGB | CURVE" in content, content
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        np.testing.assert_allclose(
            back.data_sections[1].data["GR"], [75.0, 80.0],
            err_msg="numeric GR re-read as string",
        )


# ──────────────────────────────────────────────────────────────
# Stage 12 fix regression pins — writer domain (H-01 numeric-first,
# F-01 explicit-format matrix, H-02 'S'-numeric markerless, M-28,
# M-29, M-30, M-38).  Each FAILS on pre-fix code and PASSES on
# post-fix.  Adversarial evidence: tmp/s11-adv-cross1-report.md,
# tmp/s11-adv-h1-report.md, tmp/s11-adv-m5-report.md.
# ──────────────────────────────────────────────────────────────


class TestH01NumericFirstMixedRoundTrip:
    """H-01 (HIGH, CONFIRMED): the writer emitted a SELF-UNREADABLE
    LAS 3.0 file when the same mnemonic was string in one data_section
    and numeric in another, in numeric-first order — the main ~C
    union-forced {S} and the parser's format-vs-placement check raised
    LASParseError on the writer's own output.  Post-fix the main ~C is
    markerless and both sections round-trip with correct types."""

    def _mixed_model(self) -> LASFile:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format=""))
        # numeric FIRST, string SECOND
        las.data_sections.append(
            DataSection(
                name="LOG_NUM",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 200.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        las.data_sections.append(
            DataSection(
                name="LOG_STR",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([300.0, 400.0])},
                string_data={"GR": np.array(["sand", "shale"], dtype=object)},
            )
        )
        return las

    def test_numeric_first_mixed_roundtrip(self, tmp_path: Path) -> None:
        las = self._mixed_model()
        out = tmp_path / "h01_numfirst.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert las.validate(complete=True) == [], las.validate(complete=True)
            write_las_file(out, las)
        # Main ~C must be markerless for the mixed mnemonic.
        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        gr_line = next((ln for ln in curve_block.splitlines() if ln.strip().startswith("GR")), "")
        assert "{S}" not in gr_line, f"main ~C carries union-forced {{S}}: {gr_line!r}"
        # Re-read must succeed (pre-fix: LASParseError) with correct types.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        by_name = {ds.name: ds for ds in back.data_sections}
        np.testing.assert_allclose(
            by_name["LOG_NUM"].data["GR"], [75.0, 80.0], err_msg="numeric GR lost"
        )
        assert by_name["LOG_STR"].string_data["GR"] is not None, "string GR lost"


class TestF01ExplicitFormatMatrixMixed:
    """F-01 (HIGH, pass-2): the H-01 explicit-format sub-variant — a
    curve with an explicit NON-'S' numeric format ('F'/'E'/'A') at top
    level, placed string in one data_section and numeric in another,
    made the main ~C carry the curve's own {F} token → parser rejected
    the writer's output.  _suppress_s_marker now extends to ANY explicit
    format for mixed placement; both direct-construction orders and the
    from_dict path round-trip."""

    def _mixed_f(self, numeric_first: bool) -> LASFile:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"))
        num_sec = DataSection(
            name="LOG_NUM",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
            ],
            data={"DEPT": np.array([100.0, 200.0]), "GR": np.array([75.0, 80.0])},
        )
        str_sec = DataSection(
            name="LOG_STR",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                CurveDefinition(mnemonic="GR", unit="GAPI", data_format=""),
            ],
            data={"DEPT": np.array([300.0, 400.0])},
            string_data={"GR": np.array(["sand", "shale"], dtype=object)},
        )
        if numeric_first:
            las.data_sections.append(num_sec)
            las.data_sections.append(str_sec)
        else:
            las.data_sections.append(str_sec)
            las.data_sections.append(num_sec)
        return las

    @pytest.mark.parametrize("numeric_first", [True, False])
    def test_explicit_f_mixed_both_orders_roundtrip(self, tmp_path: Path, numeric_first: bool) -> None:
        las = self._mixed_f(numeric_first)
        out = tmp_path / f"f01_f_{'num' if numeric_first else 'str'}first.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert las.validate(complete=True) == [], las.validate(complete=True)
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        by_name = {ds.name: ds for ds in back.data_sections}
        np.testing.assert_allclose(
            by_name["LOG_NUM"].data["GR"], [75.0, 80.0],
            err_msg=f"F-01 explicit-F numeric GR lost (order={numeric_first})",
        )
        assert by_name["LOG_STR"].string_data["GR"] is not None, "string GR lost"

    def test_explicit_e_mixed_numeric_first_roundtrip(self, tmp_path: Path) -> None:
        """'E' format: same mixed placement must round-trip."""
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="E"))
        las.data_sections.append(
            DataSection(
                name="LOG_NUM",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 200.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        las.data_sections.append(
            DataSection(
                name="LOG_STR",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([300.0, 400.0])},
                string_data={"GR": np.array(["sand", "shale"], dtype=object)},
            )
        )
        out = tmp_path / "f01_e_numfirst.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert las.validate(complete=True) == [], las.validate(complete=True)
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        by_name = {ds.name: ds for ds in back.data_sections}
        np.testing.assert_allclose(by_name["LOG_NUM"].data["GR"], [75.0, 80.0])

    def test_all_numeric_explicit_f_keeps_token(self, tmp_path: Path) -> None:
        """Control: a purely-numeric curve keeps its {F} token (no
        over-suppression)."""
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"))
        las.data_sections.append(
            DataSection(
                name="S1",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                section_curves=[
                    CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"),
                    CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F"),
                ],
                data={"DEPT": np.array([100.0, 200.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        out = tmp_path / "f01_control.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        gr_line = next((ln for ln in curve_block.splitlines() if ln.strip().startswith("GR")), "")
        assert "{F}" in gr_line, f"all-numeric explicit-F token suppressed: {gr_line!r}"

    def test_explicit_a_from_dict_sc0_roundtrip(self, tmp_path: Path) -> None:
        """The F-01 'A' from_dict sc=0 shape: a top-level 'A' format with
        mixed sections (section_curves stripped from the dict) — pre-fix
        the numeric GR re-read as strings (data loss); post-fix numeric."""
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="A"))
        las.data_sections.append(
            DataSection(
                name="LOG_NUM",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 200.0]), "GR": np.array([75.0, 80.0])},
            )
        )
        las.data_sections.append(
            DataSection(
                name="LOG_STR",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([300.0, 400.0])},
                string_data={"GR": np.array(["sand", "shale"], dtype=object)},
            )
        )
        d = las.to_dict()
        for ds in d.get("data_sections", []):
            ds.pop("section_curves", None)
        from_dict_las = LASFile.from_dict(d)
        out = tmp_path / "f01_a_sc0.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert from_dict_las.validate(complete=True) == [], from_dict_las.validate(complete=True)
            write_las_file(out, from_dict_las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        by_name = {ds.name: ds for ds in back.data_sections}
        np.testing.assert_allclose(
            by_name["LOG_NUM"].data["GR"], [75.0, 80.0],
            err_msg="F-01 'A' from_dict numeric GR re-read as strings/lost",
        )
        assert by_name["LOG_STR"].string_data["GR"] is not None, "string GR lost"


class TestH02SNumericMarkerless:
    """H-02 (HIGH, CONFIRMED): a curve with data_format='S' placed
    NUMERICALLY in a section forced the {S} marker → the parser re-read
    the numeric column as STRING (silent type corruption, 0 warnings).
    The marker is now suppressed when the emitted scope places the
    mnemonic numerically."""

    def test_s_format_numeric_section_roundtrip_numeric(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI", data_format="S"))
        las.data_sections.append(
            DataSection(
                name="S1",
                section_type="LOG_DATA",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([100.0, 101.0]), "GR": np.array([11.0, 12.0])},
            )
        )
        out = tmp_path / "h02_s_numeric.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        # Main ~C must NOT carry {S} for the numeric placement.
        content = out.read_text(encoding="utf-8")
        curve_block = content.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        gr_line = next((ln for ln in curve_block.splitlines() if ln.strip().startswith("GR")), "")
        assert "{S}" not in gr_line, f"H-02 spurious {{S}} in main ~C: {gr_line!r}"
        # Re-read must yield NUMERIC values (pre-fix: strings).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        np.testing.assert_allclose(
            back.data_sections[0].data["GR"], [11.0, 12.0],
            err_msg="H-02 numeric column re-read as string",
        )


class TestM28Las30TopLevelDuplicateRefusal:
    """M-28 (CONFIRMED MEDIUM): the LAS 3.0 top-level path (no
    data_sections) silently discarded a duplicate column's DISTINCT data
    on write→re-read while the legacy path refused loudly.  The X-1
    duplicate-distinct-data refusal now fires on ALL versions."""

    def test_las30_top_level_distinct_dup_refuses(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.curves_order = ["DEPT", "dept"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.curves.append(CurveDefinition(mnemonic="dept"))
        las.logs["DEPT"] = np.array([1000.0, 1001.0])
        las.logs["dept"] = np.array([50.0, 60.0])
        out = tmp_path / "m28_las30_dup.las"
        with pytest.raises(LASWriteError, match="same mnemonic"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_las_file(out, las)


class TestM29CommaTabEmptyRowPreserved:
    """M-29 (CONFIRMED MEDIUM): empty-string/whitespace-only data values
    with COMMA/TAB emitted a BLANK data line the reader silently skipped
    (row dropped 2→1).  The empty-value guard now routes blank rows
    through the '-' sentinel for COMMA/TAB too."""

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_value_comma_row_preserved(self, tmp_path: Path, value: str) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DESC"]
        las.curves.append(CurveDefinition(mnemonic="DESC", unit="", data_format="S"))
        las.string_data["DESC"] = np.array([value, "x"], dtype=object)
        out = tmp_path / "m29_comma.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        desc = back.string_data["DESC"]
        assert len(desc) == 2, f"M-29 row dropped ({len(desc)} rows): {desc}"

    def test_tab_empty_value_row_preserved(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="TAB"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DESC"]
        las.curves.append(CurveDefinition(mnemonic="DESC", unit="", data_format="S"))
        las.string_data["DESC"] = np.array(["", "x"], dtype=object)
        out = tmp_path / "m29_tab.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert len(back.string_data["DESC"]) == 2, (
            f"M-29 TAB row dropped: {back.string_data['DESC']}"
        )


class TestM30NoneContainersLAsWriteError:
    """M-30 (CONFIRMED MEDIUM): _WriterMutationGuard.__init__ ran
    dict(las_file.logs)/dict(las_file.string_data) BEFORE the try that
    wraps exceptions in LASWriteError — logs=None leaked a raw
    TypeError, violating the write_las_file "Raises: LASWriteError"
    contract.  None guards now mirror the curves/curves_order guards."""

    @pytest.mark.parametrize("attr", ["logs", "string_data"])
    def test_none_container_raises_las_write_error(self, tmp_path: Path, attr: str) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        setattr(las, attr, None)
        out = tmp_path / f"m30_{attr}_none.las"
        with pytest.raises(LASWriteError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_las_file(out, las)


class TestM38SanitizedNameWarnings:
    """M-38 (CONFIRMED MEDIUM): the S8-1 keyword-collision warning gate
    tested the RAW section name while the emitted name is sanitized —
    '~A' bypassed the warning yet the name was still lost to Section_N;
    pipes/tildes were silently stripped ('A|B'→'AB').  The gate now
    tests the SANITIZED name and warns on pipe/tilde stripping."""

    @pytest.mark.parametrize(
        "name",
        ["~A", "~ASCII", "A|B", "~CURVE"],
        ids=["tilde_A", "tilde_ASCII", "pipe", "tilde_CURVE"],
    )
    def test_sanitized_name_warns(self, tmp_path: Path, name: str) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.data_sections.append(
            DataSection(
                name=name,
                section_type="LOG_DATA",
                curves_order=["DEPT"],
                section_curves=[CurveDefinition(mnemonic="DEPT", unit="M", data_format="F")],
                data={"DEPT": np.array([100.0, 110.0])},
            )
        )
        out = tmp_path / "m38_sanitized.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        msgs = [str(w.message) for w in rec]
        assert any("altered when written" in m or "collides with the reserved standard-ASCII" in m for m in msgs), (
            f"M-38 {name!r}: no name-alteration warning: {msgs}"
        )

    def test_normal_name_no_warning(self, tmp_path: Path) -> None:
        """Control: a legit name is preserved exactly with no warning."""
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.data_sections.append(
            DataSection(
                name="Core 1",
                section_type="LOG_DATA",
                curves_order=["DEPT"],
                section_curves=[CurveDefinition(mnemonic="DEPT", unit="M", data_format="F")],
                data={"DEPT": np.array([100.0, 110.0])},
            )
        )
        out = tmp_path / "m38_normal.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        msgs = [str(w.message) for w in rec]
        assert not any("altered when written" in m or "collides with the reserved standard-ASCII" in m for m in msgs), (
            f"M-38 normal name warned: {msgs}"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert back.data_sections[0].name == "Core 1", back.data_sections[0].name


class TestFixWriterN09DescriptionBraceTokens:
    """N-09: a user description containing a valid format token in braces
    ("Gamma {S} ray") must round-trip unchanged — the writer escapes
    braces and the parser strips only the trailing writer-appended token."""

    def test_n09_curve_description_brace_token_roundtrips(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["GR"]
        las.curves.append(
            CurveDefinition(
                mnemonic="GR", unit="GAPI", data_format="S", description="Gamma {S} ray"
            )
        )
        las.string_data["GR"] = np.array(["a", "b"], dtype=object)

        out = tmp_path / "n09_curve.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        # The user's brace token is escaped in the output (the parser's
        # FORMAT_SPEC_PATTERN cannot strip it).
        assert r"Gamma \{S\} ray" in content, content
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        cd = next(c for c in back.curves if c.mnemonic == "GR")
        assert cd.description == "Gamma {S} ray", f"description destroyed: {cd.description!r}"
        assert cd.data_format == "S", f"data_format lost: {cd.data_format!r}"

    def test_n09_curve_description_no_fabricated_format(self, tmp_path: Path) -> None:
        """A numeric curve with a mid-description {S} token and EMPTY
        data_format must not be re-routed to string_data (the pre-fix
        amplification: fabricated 'S' reclassified the column)."""
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["GR"]
        las.curves.append(
            CurveDefinition(
                mnemonic="GR", unit="GAPI", data_format="", description="Gamma {S} ray"
            )
        )
        las.logs["GR"] = np.array([75.0, 80.0])

        out = tmp_path / "n09_nofmt.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        cd = next(c for c in back.curves if c.mnemonic == "GR")
        assert cd.description == "Gamma {S} ray", f"description destroyed: {cd.description!r}"
        assert cd.data_format == "", f"data_format fabricated: {cd.data_format!r}"
        np.testing.assert_allclose(back.logs["GR"], [75.0, 80.0], err_msg="column reclassified")

    def test_n09_parameter_description_brace_token_with_zone(self, tmp_path: Path) -> None:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.logs["DEPT"] = np.array([1.0])
        las.parameters.append(
            ParameterEntry(
                mnemonic="MUD",
                unit="",
                value="x",
                description="Mud {S} in hole",
                data_format="E",
                zone=ParameterZone(zone_name="MAIN", zone_index=1),
            )
        )
        out = tmp_path / "n09_param.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        pm = next(p for p in back.parameters if p.mnemonic == "MUD")
        assert pm.description == "Mud {S} in hole", f"description destroyed: {pm.description!r}"
        assert pm.data_format == "E", f"data_format lost: {pm.data_format!r}"
        assert pm.zone is not None and pm.zone.zone_name == "MAIN", f"zone lost: {pm.zone!r}"


class TestFixWriterS8AsciiKeywordNameCollision:
    """S8-1: a DataSection explicitly named 'A'/'ASCII' collides with the
    reserved standard-ASCII section keyword.  The LAS 3.0 writer must
    WARN loudly (pre-fix the name was silently lost on re-read — the
    parser treats ``~A A``/``~A ASCII`` as a bare keyword and auto-names
    the section ``Section_N``), and the write→read roundtrip returns the
    Section_N name.

    Discriminating: pre-fix no warning fires; post-fix the warning fires.
    The rename semantics (M-22 alignment with the deferred path) are NOT
    changed — the warning makes the name-loss non-silent.
    """

    def _make_las3_with_named_section(self, name: str) -> LASFile:
        las = LASFile(version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"))
        las.well["STRT"] = "100.0"
        las.well["STOP"] = "300.0"
        las.well["STEP"] = "100.0"
        las.well["NULL"] = "-999.25"
        section = DataSection(
            name=name,
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="M", data_format="F")],
            data={"DEPT": np.array([100.0, 110.0])},
        )
        las.data_sections.append(section)
        return las

    def test_explicit_a_name_warns_and_roundtrips_as_section_n(self, tmp_path: Path) -> None:
        """DataSection(name='A') → write warns loudly; re-read yields Section_0."""
        las = self._make_las3_with_named_section("A")
        out = tmp_path / "s8_a_name.las"
        with pytest.warns(
            UserWarning,
            match="collides with the reserved standard-ASCII section keyword",
        ):
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert [d.name for d in back.data_sections] == ["Section_0"], (
            f"expected auto-named Section_0, got {[d.name for d in back.data_sections]}"
        )

    def test_explicit_ascii_name_warns_and_roundtrips_as_section_n(self, tmp_path: Path) -> None:
        """DataSection(name='ASCII') → write warns loudly; re-read yields Section_0."""
        las = self._make_las3_with_named_section("ASCII")
        out = tmp_path / "s8_ascii_name.las"
        with pytest.warns(
            UserWarning,
            match="collides with the reserved standard-ASCII section keyword",
        ):
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert [d.name for d in back.data_sections] == ["Section_0"], (
            f"expected auto-named Section_0, got {[d.name for d in back.data_sections]}"
        )

    def test_normal_name_no_warning_roundtrips_exact(self, tmp_path: Path) -> None:
        """A non-colliding explicit name does NOT warn and is preserved exactly."""
        las = self._make_las3_with_named_section("FirstSection")
        out = tmp_path / "s8_normal_name.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(out, las)
        collision = [
            x for x in w if "collides with the reserved standard-ASCII" in str(x.message)
        ]
        assert collision == [], (
            f"unexpected collision warning for normal name: "
            f"{[str(x.message) for x in collision]}"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert [d.name for d in back.data_sections] == ["FirstSection"], (
            f"normal name not preserved: {[d.name for d in back.data_sections]}"
        )
