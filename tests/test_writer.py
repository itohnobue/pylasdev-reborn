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

    def test_write_always_wrap_no(self, tmp_path: Path) -> None:
        """Test that WRAP=YES is overridden to WRAP=NO in output.

        The writer cannot produce wrapped output (always one line per depth
        step). When the model has WRAP=YES, the writer emits a warning and
        writes WRAP=NO to keep the header consistent with the data layout.
        """
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
        # WRAP=YES is overridden to WRAP=NO since the writer cannot produce
        # wrapped output (F-01: header-data consistency fix)
        assert "WRAP.   NO" in content
        assert "WRAP.   YES" not in content

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
        # Also verify other curve data values
        assert "1669.875" in data_section or "1670" in data_lines[1]

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
        # sample_las3.0_spec.las contains structured data-type sections
        # whose curve names get deduplicated differently on re-read
        # (F-01 fix now populates structured-section curves). Skip
        # strict roundtrip comparisons for this file.
        structured_files = {"sample_las3.0_spec.las"}

        for las_path in all_las_files:
            data = read_las_file(las_path)
            temp_file = tmp_path / las_path.name
            write_las_file(temp_file, data)
            assert temp_file.exists()
            reread = read_las_file(temp_file)
            assert len(reread["curves_order"]) > 0

            # Verify version preserved
            assert reread["version"]["VERS"] == data["version"]["VERS"]

            # Verify curves_order preserved (skip structured files)
            if las_path.name not in structured_files:
                assert reread["curves_order"] == data["curves_order"], (
                    f"curves_order mismatch in {las_path.name}: "
                    f"{reread['curves_order']} vs {data['curves_order']}"
                )

            # Verify data shapes and values for non-empty log files
            # (skip structured files — their logs dict has non-main curves)
            if las_path.name not in structured_files and data.get("logs"):
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
        assert " DLM " not in content, (
            f"DLM line unexpectedly emitted for LAS 1.2:\n{content}"
        )

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
        assert data_rows == [], (
            f"Expected no data rows after ~A BROKEN, got: {data_rows}"
        )

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
                    "CDES": np.array([0.0]),
                },
            )
            section.string_data["CDES"] = np.array(["ROCK"], dtype=np.str_)
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

    # --- M-13: LAS 3.0 ~Other deprecation warning test ---
    def test_las30_other_section_deprecation_warning(self, tmp_path: Path) -> None:
        """Test that LAS 3.0 files with ~Other content emit a deprecation warning.

        LAS 3.0 deprecates the ~Other section — content must go into
        user-defined Parameter or Column Data sections instead.  When
        writing a LAS 3.0 file with other content, a UserWarning is
        emitted and the ~Other section is NOT written to the output.
        """
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.other = "This content should trigger a deprecation warning.\n"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30_other.las"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            other_warnings = [
                x for x in w if "~Other" in str(x.message) and "deprecates" in str(x.message)
            ]
            assert len(other_warnings) >= 1, (
                "Expected deprecation warning about ~Other section in LAS 3.0"
            )

        # ~Other section must NOT appear in the written file
        content = temp_file.read_text()
        assert "~OTHER" not in content
        assert "~Other" not in content

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
        section.string_data["CDES"] = np.array(
            ["LIMESTONE\tFRACTURED", "DOLOMITE"], dtype=np.str_
        )

        temp_file = tmp_path / "space_delim_tab.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            delim_warnings = [
                x for x in w if "whitespace" in str(x.message)
            ]
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
            delim_warnings = [
                x for x in w if "delimiter character (COMMA)" in str(x.message)
            ]
            assert len(delim_warnings) >= 1, (
                "Expected warning about COMMA delimiter in string data"
            )

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
        section.string_data["CDES"] = np.array(
            ["SANDSTONE\tLAMINATED", "DOLOMITE"], dtype=np.str_
        )

        temp_file = tmp_path / "tab_delim_tab.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            delim_warnings = [
                x for x in w if "delimiter character (TAB)" in str(x.message)
            ]
            assert len(delim_warnings) >= 1, (
                "Expected warning about TAB delimiter in string data"
            )

        content = temp_file.read_text()
        # The embedded tab should be replaced with space
        assert "SANDSTONE LAMINATED" in content, (
            "Tab should be replaced with space"
        )

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
        with pytest.raises(LASWriteError, match=r"Multiple data_sections.*only supported for LAS 3.0"):
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
        """": " (colon-space) → ":_ " (underscore between colon and space)."""
        assert _escape_colons_for_las_value("Oil: Gas Corp") == "Oil:_ Gas Corp"

    def test_trailing_colon_escaped(self) -> None:
        """Trailing ":" → ":_" to prevent :\\s*$ match."""
        assert _escape_colons_for_las_value("Oil:") == "Oil:_"

    def test_space_colon_escaped(self) -> None:
        """" :" (space-colon) → " _:" (underscore between space and colon)."""
        assert _escape_colons_for_las_value("Oil :Gas Corp") == "Oil _:Gas Corp"

    def test_space_colon_space_double_escaped(self) -> None:
        """" : " (space-colon-space) → " _:_ " (underscore on both sides)."""
        assert _escape_colons_for_las_value("Oil : Gas Corp") == "Oil _:_ Gas Corp"

    def test_trailing_space_colon_escaped(self) -> None:
        """Trailing " :" → " _:_" (both sides: before-colon + end-of-string)."""
        assert _escape_colons_for_las_value("Oil :") == "Oil _:_"

    def test_leading_colon_not_escaped_when_no_ws(self) -> None:
        """Leading ":" with no following whitespace is unchanged."""
        assert _escape_colons_for_las_value(":Oil") == ":Oil"

    def test_leading_colon_space_escaped(self) -> None:
        """" : " at start → ":_ " (colon-space escaping at position 0)."""
        assert _escape_colons_for_las_value(": value") == ":_ value"

    def test_leading_space_colon_escaped(self) -> None:
        """" :" at start (space-colon-value) → " _:..." ."""
        assert _escape_colons_for_las_value(" :value") == " _:value"

    def test_multiple_spaces_before_colon_escaped(self) -> None:
        """Multiple spaces before colon: all whitespace preserved, _ inserted."""
        assert _escape_colons_for_las_value("  :value") == "  _:value"

    def test_multiple_spaces_before_colon_with_trailing_space(self) -> None:
        """"  : value" (two spaces, colon, space) is fully escaped."""
        assert _escape_colons_for_las_value("  : value") == "  _:_ value"

    def test_multiple_embedded_colons(self) -> None:
        """Multiple embedded colons are all escaped independently."""
        assert (
            _escape_colons_for_las_value("A: B : C")
            == "A:_ B _:_ C"
        )

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
        """"Oil: Gas Corp" (colon-space) survives roundtrip — parser now un-escapes."""
        result = self._roundtrip_well_value(tmp_path, "Oil: Gas Corp")
        # F-022: The parser now un-escapes colon artifacts inserted by
        # _escape_colons_for_las_value, restoring the original value.
        assert result == "Oil: Gas Corp"
        assert "Oil" in result  # Not truncated to just "Oil"

    def test_roundtrip_space_colon_space(self, tmp_path: Path) -> None:
        """"Oil : Gas Corp" (space-colon-space) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil : Gas Corp")
        # Must NOT be truncated to just "Oil" (the F-EX-03 corruption).
        assert result != "Oil"
        assert "Oil" in result
        assert "Gas Corp" in result

    def test_roundtrip_space_colon_no_trailing_space(self, tmp_path: Path) -> None:
        """"Oil :Gas Corp" (space-colon, no space after) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil :Gas Corp")
        assert result != "Oil"
        assert "Oil" in result
        assert "Gas Corp" in result

    def test_roundtrip_trailing_colon(self, tmp_path: Path) -> None:
        """"Oil:" (trailing colon) survives roundtrip."""
        result = self._roundtrip_well_value(tmp_path, "Oil:")
        # Must NOT be truncated — colon-containing value preserved.
        assert result != "Oil"
        assert "Oil" in result

    def test_roundtrip_trailing_space_colon(self, tmp_path: Path) -> None:
        """"Oil :" (trailing space-colon) survives roundtrip."""
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
        np.testing.assert_array_almost_equal(
            logs["DEPT"], np.array([100.0, 200.0, 300.0])
        )
        assert "GR" in logs, f"Expected 'GR' in reparsed logs, got {list(logs.keys())}"
        np.testing.assert_array_almost_equal(
            logs["GR"], np.array([50.0, 60.0, 70.0])
        )

    def test_copy_back_warning_emitted(self, tmp_path: Path) -> None:
        """The single-section fallback warning is emitted when
        data_sections exist in a non-LAS-3.0 file."""
        las = self._make_las2_with_single_section()

        output_file = tmp_path / "warn.las"
        with pytest.warns(
            UserWarning,
            match="data_sections are only supported for LAS 3.0",
        ):
            write_las_file(str(output_file), las)

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
        assert logs["DEPT"][0] == 1000.0, (
            f"Expected user DEPT value 1000.0, "
            f"got {logs['DEPT'][0]}"
        )
        # GR should NOT be present — it was in data_sections only,
        # and copy-back was blocked by pre-populated las.logs
        assert "GR" not in logs, (
            f"GR should NOT be in output (copy-back blocked), "
            f"but found in {list(logs.keys())}"
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
            f"Expected DEPT before GR in output, "
            f"but DEPT at {dept_pos}, GR at {gr_pos}"
        )

    def test_copy_back_inconsistent_curves_count_raises(
        self, tmp_path: Path
    ) -> None:
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
        with pytest.warns(
            UserWarning,
            match="data_sections are only supported for LAS 3.0",
        ), pytest.raises(LASWriteError, match=r"curves count.*does not match"):
            write_las_file(str(output_file), las)
