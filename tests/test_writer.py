"""Tests for LAS file writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import read_las_file, write_las_file
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

    def test_write_always_wrap_no(self, tmp_path: Path) -> None:
        """Test that WRAP is always written as NO (writer outputs non-wrapped)."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "YES", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {"DEPT": np.array([1.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "output.las"
        write_las_file(temp_file, data)

        content = temp_file.read_text()
        assert "WRAP.   NO" in content

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
        # Check numeric data is written
        assert "1670" in content

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

    def test_write_encoding(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test writing with explicit encoding."""
        temp_file = tmp_path / "utf8.las"
        write_las_file(temp_file, sample_las_data, encoding="utf-8")
        assert temp_file.exists()

    def test_write_error_on_bad_path(self, sample_las_data: dict) -> None:
        """Test LASWriteError on invalid path."""
        with pytest.raises(LASWriteError):
            write_las_file(Path("/nonexistent/dir/file.las"), sample_las_data)

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

    def test_write_real_files_roundtrip(self, all_las_files: list[Path], tmp_path: Path) -> None:
        """Test writing all real LAS files and reading back."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            temp_file = tmp_path / las_path.name
            write_las_file(temp_file, data)
            assert temp_file.exists()
            reread = read_las_file(temp_file)
            assert len(reread["curves_order"]) > 0

            # Verify version preserved
            assert reread["version"]["VERS"] == data["version"]["VERS"]

            # Verify curves_order preserved
            assert reread["curves_order"] == data["curves_order"], (
                f"curves_order mismatch in {las_path.name}: "
                f"{reread['curves_order']} vs {data['curves_order']}"
            )

            # Verify data shapes and values for non-empty log files
            if data.get("logs"):
                for curve in data["curves_order"]:
                    if curve in reread["logs"]:
                        assert data["logs"][curve].shape == reread["logs"][curve].shape, (
                            f"Shape mismatch for {curve} in {las_path.name}: "
                            f"{data['logs'][curve].shape} vs {reread['logs'][curve].shape}"
                        )
                        np.testing.assert_allclose(
                            data["logs"][curve],
                            reread["logs"][curve],
                            rtol=1e-5,
                            err_msg=f"Data mismatch for {curve} in {las_path.name}",
                        )

            # For LAS 3.0 files, verify data_sections and string_data if present
            if data.get("string_data"):
                for key in data["string_data"]:
                    assert key in reread.get("string_data", {}), (
                        f"string_data key {key} missing in roundtrip for {las_path.name}"
                    )
                    np.testing.assert_array_equal(
                        data["string_data"][key],
                        reread["string_data"][key],
                        err_msg=f"string_data mismatch for {key} in {las_path.name}",
                    )
            if data.get("data_sections"):
                assert len(reread.get("data_sections", [])) == len(data["data_sections"]), (
                    f"data_sections count mismatch in {las_path.name}"
                )

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
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "null_test.las"
        write_las_file(temp_file, las)  # Should not crash

        content = temp_file.read_text()
        assert "100" in content

    # --- TEST-17: LAS 3.0 data_sections with non-numeric NULL value ---
    def test_write_las30_non_numeric_null_in_data_sections(self, tmp_path: Path) -> None:
        """Test LAS 3.0 data_sections path with non-numeric NULL value fallback."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "NOT A NUMBER"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))

        section = DataSection(
            name="CURVE",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "las30_null.las"
        write_las_file(temp_file, las)  # Should not crash

        content = temp_file.read_text()
        assert "100" in content
        assert "101" in content

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
        # No data rows should be written (guard at line 198 returns [])
        # Content after ~A BROKEN should be empty or next section

    # --- F-24: NaN values in _format_data_rows ---
    def test_write_nan_values(self, tmp_path: Path) -> None:
        """Test that NaN values in data are written as null_value.

        Exercises writer.py:226-227 — the np.isnan check in _format_data_rows.
        """
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["DT"] = np.array([50.0, np.nan])

        temp_file = tmp_path / "nan_test.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # The NaN value should be output as null_value (-999.25)
        assert "-999.25" in content
        # Verify roundtrip: re-read and check the NaN position is null_value
        reread = read_las_file(temp_file)
        assert reread["logs"]["DT"][1] == -999.25

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

        The encoding key is processed by LASFile.from_dict() at models.py:333.
        """
        data = {
            "encoding": "ascii",
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "parameters": {},
            "logs": {"DEPT": np.array([1.0, 2.0])},
            "curves_order": ["DEPT"],
        }
        temp_file = tmp_path / "enc_key.las"
        write_las_file(temp_file, data)

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
