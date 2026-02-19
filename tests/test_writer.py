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
            # Skip LAS 3.0 files (different data handling)
            if data["version"]["VERS"].startswith("3"):
                continue
            temp_file = tmp_path / las_path.name
            write_las_file(temp_file, data)
            assert temp_file.exists()
            reread = read_las_file(temp_file)
            assert len(reread["curves_order"]) > 0

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
        assert "{A:0.0}" in content
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
        las.string_data["CDES"] = np.array(["LIMESTONE", "DOLOMITE"], dtype=np.str_)

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
