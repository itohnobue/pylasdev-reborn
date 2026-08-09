"""Tests for LAS file reader."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import read_dev_file, read_las_file, read_las_file_as_object
from pylasdev.exceptions import LASParseError, LASReadError
from pylasdev.models import CurveDefinition, LASFile
from pylasdev.parser import LASParser


class TestReadLASFile:
    """Tests for read_las_file function."""

    def test_read_all_las_files(self, all_las_files: list[Path]) -> None:
        """Test reading every LAS file in test_data/."""
        assert len(all_las_files) > 0, "No LAS test files found"

        for las_path in all_las_files:
            data = read_las_file(las_path)

            assert "version" in data
            assert "well" in data
            assert "logs" in data
            assert "curves_order" in data
            assert isinstance(data, dict)

    def test_returns_numpy_arrays(self, all_las_files: list[Path]) -> None:
        """Test that log data is returned as numpy arrays."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            for curve_data in data["logs"].values():
                assert isinstance(curve_data, np.ndarray)

    def test_preserves_curve_order(self, all_las_files: list[Path]) -> None:
        """Test that curve order matches log keys for non-3.0 files.

        Files with duplicate curve mnemonics are skipped since dict keys
        are unique but curves_order preserves duplicates.
        """
        for las_path in all_las_files:
            data = read_las_file(las_path)
            if data["curves_order"] and not data["version"]["VERS"].startswith("3"):
                # Skip files with duplicate curve names (dict keys collapse duplicates)
                if len(data["curves_order"]) != len(set(data["curves_order"])):
                    continue
                assert list(data["logs"].keys()) == data["curves_order"], (
                    f"Curve order mismatch in {las_path.name}"
                )

    def test_well_values_are_strings(self, all_las_files: list[Path]) -> None:
        """Test that well section values are strings (backward compat)."""
        for las_path in all_las_files:
            data = read_las_file(las_path)
            for key, value in data["well"].items():
                assert isinstance(value, str), (
                    f"Well value for {key} is {type(value).__name__}, not str, in {las_path.name}"
                )

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error handling for missing file."""
        with pytest.raises(LASReadError):
            read_las_file(tmp_path / "nonexistent.las")

    def test_not_a_file(self, tmp_path: Path) -> None:
        """Test error handling for directory path."""
        with pytest.raises(LASReadError):
            read_las_file(tmp_path)

    def test_version_is_valid(self, all_las_files: list[Path]) -> None:
        """Test version section contains valid version string."""
        valid_versions = ["1.2", "1.20", "2.0", "3.0"]
        for las_path in all_las_files:
            data = read_las_file(las_path)
            assert data["version"]["VERS"] in valid_versions

    def test_sample_las_specific_values(self, test_data_dir: Path) -> None:
        """Test specific values from sample.las."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        data = read_las_file(sample)
        assert "DEPT" in data["logs"]
        assert len(data["logs"]["DEPT"]) == 3
        # Verify specific data values from sample.las
        np.testing.assert_array_almost_equal(
            data["logs"]["DEPT"], np.array([1670.0, 1669.875, 1669.75])
        )
        # Verify well section values (LAS 1.2: data is after colon, not before)
        # COMP.   COMPANY:   # ANY OIL COMPANY LTD. → value is "# ANY OIL COMPANY LTD."
        assert data["well"]["COMP"] == "# ANY OIL COMPANY LTD."
        # WELL.   WELL:   ANY ET AL OIL WELL #12 → value is "ANY ET AL OIL WELL #12"
        assert data["well"]["WELL"] == "ANY ET AL OIL WELL #12"
        assert data["well"]["STRT"] == "1670.000000"
        # Verify parameter values
        assert data["parameters"]["BHT"] == "35.5000"
        assert data["parameters"]["BS"] == "200.0000"

    def test_wrapped_file_correct_shape(self, test_data_dir: Path) -> None:
        """Test that wrapped files produce equal-length arrays."""
        wrapped_files = [
            test_data_dir / "sample_wrapped.las",
            test_data_dir / "sample_2.0_wrapped.las",
        ]
        for wf in wrapped_files:
            if not wf.exists():
                continue
            data = read_las_file(wf)
            if data["curves_order"]:
                sizes = [len(data["logs"][c]) for c in data["curves_order"] if c in data["logs"]]
                assert len(set(sizes)) == 1, f"Arrays have different sizes in {wf.name}: {sizes}"

    def test_mislabeled_wrap_handled(self, test_data_dir: Path) -> None:
        """Test that files with WRAP=YES but non-wrapped data are handled."""
        ct = test_data_dir / "comment_test.las"
        assert ct.exists(), f"Required test data missing: {ct}"
        data = read_las_file(ct)
        # All arrays should have equal length
        if data["curves_order"]:
            sizes = [len(data["logs"][c]) for c in data["curves_order"]]
            assert len(set(sizes)) == 1, f"Arrays have different sizes: {sizes}"

    def test_wrapped_string_curve_preserved(self, tmp_path: Path) -> None:
        """F-R-03: String curve values in wrapped LAS files are preserved.

        Before this fix, _read_wrapped converted all values through
        _to_finite_float(), silently converting string values to the
        null value (-999.25).  String curves ({S} format in ~C section)
        must be stored in string_data, not converted to float.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " GR .API  :  Gamma Ray {F}\n"
            " LITH .   :  Lithology {S}\n"
            "~A\n"
            "1000.0\n"
            "50.0  Sandstone\n"
            "1001.0\n"
            "51.0  Shale\n"
        )
        test_file = tmp_path / "wrapped_string.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # Float curves should be parsed normally
        assert "DEPT" in data["logs"]
        assert "GR" in data["logs"]
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

        # String curve must be in string_data, NOT silently converted
        assert "LITH" in data["string_data"], (
            "String curve 'LITH' missing from string_data — values "
            "were likely converted to null_value by _to_finite_float"
        )
        lith_values = data["string_data"]["LITH"]
        assert len(lith_values) == 2, f"Expected 2 LITH values, got {len(lith_values)}"
        np.testing.assert_array_equal(
            lith_values,
            np.array(["Sandstone", "Shale"], dtype=np.str_),
        )

        # String curve should NOT be in float logs
        assert "LITH" not in data["logs"], (
            "String curve 'LITH' found in logs (float) — should only be in string_data"
        )

    # --- F-H-006: string curve multi-char value preservation ---

    def test_string_curve_multi_char_values_preserved(self, tmp_path: Path) -> None:
        """F-H-006: String curve values > 1 char are preserved in string_data.

        Before the fix at _las30_data.py (dtype=object string arrays),
        string curve arrays were pre-allocated with dtype=np.str_ which
        defaults to a single-character fixed-width Unicode type (U1),
        truncating values to their first character.  After the fix,
        dtype=object preserves arbitrary-length strings.

        This test uses a LAS 3.0 file with {S}-format curve containing
        multi-character lithology names like "Sandstone" and "Limestone".
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITING CHARACTER\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " LITH .       :  Lithology {S}\n"
            "~ASCII\n"
            " 100.0,Sandstone\n"
            " 200.0,Limestone\n"
            " 300.0,Dolomite\n"
        )
        test_file = tmp_path / "string_curve_multi_char.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # String curve must be in string_data
        assert "LITH" in data["string_data"], "String curve 'LITH' missing from string_data"
        lith_values = data["string_data"]["LITH"]
        assert len(lith_values) == 3, f"Expected 3 LITH values, got {len(lith_values)}"
        np.testing.assert_array_equal(
            lith_values,
            np.array(["Sandstone", "Limestone", "Dolomite"]),
        )
        # Verify no single-char truncation (before the data_reader fix,
        # dtype=np.str_ pre-allocation at U1 would truncate multi-char values
        # to their first character — "Sandstone" → "S").
        # For LAS 3.0 parser path, np.str_ auto-sizes to fit longest value,
        # so dtype may be U9 or similar.  The critical check is that values
        # are full length, validated by assert_array_equal above.

        # Float curve should be correct
        assert "DEPT" in data["logs"]
        np.testing.assert_array_equal(
            data["logs"]["DEPT"],
            np.array([100.0, 200.0, 300.0]),
        )

    def test_encoding_parameter(self, test_data_dir: Path) -> None:
        """Test that explicit encoding parameter works."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        data = read_las_file(sample, encoding="utf-8")
        assert "logs" in data


class TestReadLASFileAsObject:
    """Tests for read_las_file_as_object function."""

    def test_returns_las_file_object(self, test_data_dir: Path) -> None:
        """Test that read_las_file_as_object returns LASFile."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        assert isinstance(las, LASFile)
        assert las.source_file != ""
        assert las.encoding != ""

    def test_object_has_curves(self, test_data_dir: Path) -> None:
        """Test LASFile object has curve definitions."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        assert len(las.curves) > 0
        assert len(las.curves_order) > 0
        assert len(las.logs) > 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error for missing file."""
        with pytest.raises(LASReadError):
            read_las_file_as_object(tmp_path / "missing.las")

    def test_not_a_file(self, tmp_path: Path) -> None:
        """Test error for directory path."""
        with pytest.raises(LASReadError, match="Not a file"):
            read_las_file_as_object(tmp_path)

    # --- F22: LASParseError from parser re-raised with file path ---
    def test_parse_error_re_raises_with_filename(self, tmp_path: Path) -> None:
        """Test that LASParseError from parser is re-raised with file path.

        Exercises reader.py:122-123 — when parser.parse() raises LASParseError
        (e.g., missing required ~V section), the reader catches it and re-raises
        a new LASParseError that includes the file path in the message.
        """
        content = "~W WELL.NAME: Test Well\n~A\n1.0 2.0"
        test_file = tmp_path / "no_version.las"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(LASParseError, match="Error reading"):
            read_las_file_as_object(test_file)

    def test_las30_object_has_version(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 file parsed as object has correct version."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        assert las.version.vers == "3.0"
        assert las.version.is_las30 is True
        assert las.version.dlm == "COMMA"

    def test_las30_curves_with_formats(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 curve format specifiers are parsed."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        # Check that format specifiers were extracted
        dept_curve = las.get_curve_by_mnemonic("DEPT")
        assert dept_curve is not None
        assert dept_curve.data_format == "F"
        # String format
        cdes_curve = las.get_curve_by_mnemonic("CDES")
        assert cdes_curve is not None
        assert cdes_curve.data_format == "S"

    def test_las30_array_curves(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 array notation curves are parsed."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        nmr_curves = las.get_array_curves("NMR")
        assert len(nmr_curves) == 5
        assert nmr_curves[0].array_info is not None
        assert nmr_curves[0].array_info.time_offset == 0.0
        assert nmr_curves[4].array_info is not None
        assert nmr_curves[4].array_info.time_offset == 20.0

    def test_las30_parameters_with_zones(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 parameter zone associations."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        # Check zone-associated parameters
        assert len(las.parameters) > 0
        zoned = [p for p in las.parameters if p.zone is not None]
        assert len(zoned) > 0

    def test_las30_scientific_notation_format(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 {E} (scientific notation) format specifier."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        yme_curve = las.get_curve_by_mnemonic("YME")
        assert yme_curve is not None
        assert yme_curve.data_format == "E"


class TestAPICodeParsing:
    """Tests for API code extraction from curve definitions."""

    def test_api_codes_parsed_from_las12(self, test_data_dir: Path) -> None:
        """Test API codes are extracted from LAS 1.2 curve_api file."""
        api_file = test_data_dir / "sample_curve_api.las"
        assert api_file.exists(), f"Required test data missing: {api_file}"
        las = read_las_file_as_object(api_file)
        rhob = las.get_curve_by_mnemonic("RHOB")
        assert rhob is not None
        assert rhob.api_code == "7 350 02 00"

    def test_api_codes_parsed_from_las20(self, test_data_dir: Path) -> None:
        """Test API codes are extracted from LAS 2.0 file."""
        las20 = test_data_dir / "sample_2.0.las"
        assert las20.exists(), f"Required test data missing: {las20}"
        las = read_las_file_as_object(las20)
        dt = las.get_curve_by_mnemonic("DT")
        assert dt is not None
        assert dt.api_code == "60 520 32 00"

    def test_empty_api_code_for_curves_without_it(self, test_data_dir: Path) -> None:
        """Test that curves without API codes have empty api_code."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        dept = las.get_curve_by_mnemonic("DEPT")
        assert dept is not None
        assert dept.api_code == ""

    def test_api_code_roundtrip(self, test_data_dir: Path, tmp_path: Path) -> None:
        """Test API codes survive a write-then-read roundtrip."""
        from pylasdev import write_las_file

        api_file = test_data_dir / "sample_curve_api.las"
        assert api_file.exists(), f"Required test data missing: {api_file}"
        las = read_las_file_as_object(api_file)
        temp_file = tmp_path / "api_roundtrip.las"
        write_las_file(temp_file, las)
        las2 = read_las_file_as_object(temp_file)
        rhob = las2.get_curve_by_mnemonic("RHOB")
        assert rhob is not None
        assert rhob.api_code == "7 350 02 00"


class TestDataReaderEdgeCases:
    """Tests for data reader edge cases and boundary conditions."""

    def test_missing_values_filled_with_null(self, tmp_path: Path) -> None:
        """Test that short data lines fill remaining curves with null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A  DEPT  DT  GR\n"
            "100.0  50.0  75.0\n"
            "101.0  51.0\n"
            "102.0\n"
        )
        test_file = tmp_path / "short_lines.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Line 1: all values present
        assert data["logs"]["GR"][0] == 75.0
        # Line 2: GR missing -> should be null_value, NOT 0.0
        assert data["logs"]["GR"][1] == -999.25
        # Line 3: DT and GR missing -> both should be null_value
        assert data["logs"]["DT"][2] == -999.25
        assert data["logs"]["GR"][2] == -999.25

    def test_section_after_ascii_not_parsed_as_data(self, tmp_path: Path) -> None:
        """Test that sections appearing after ~A don't corrupt data."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
            "~OTHER\n"
            "Some free text after data.\n"
        )
        test_file = tmp_path / "section_after_a.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Should have exactly 2 data points, not 3+
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])

    def test_duplicate_curve_names_renamed_with_warning(self, tmp_path: Path) -> None:
        """Test that duplicate curve mnemonics are renamed with a warning."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray 1\n"
            " GR.GAPI  :  Gamma Ray 2\n"
            "~A  DEPT  GR  GR\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "dup_curves.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Duplicate curve mnemonic" in str(x.message) for x in w)

        # Second GR should be renamed to GR_2
        assert "GR" in data["logs"]
        assert "GR_2" in data["logs"]
        np.testing.assert_array_equal(data["logs"]["GR"], [10.0, 11.0])
        np.testing.assert_array_equal(data["logs"]["GR_2"], [20.0, 21.0])

    def test_duplicate_curves_metadata_synced(self, tmp_path: Path) -> None:
        """Test that CurveDefinition objects are synced with renamed curves_order."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray 1\n"
            " GR.GAPI  :  Gamma Ray 2\n"
            "~A  DEPT  GR  GR\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "dup_meta.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)

        # curves_order and curves mnemonics must match
        assert las.curves_order == ["DEPT", "GR", "GR_2"]
        assert [c.mnemonic for c in las.curves] == ["DEPT", "GR", "GR_2"]

        # get_curve_by_mnemonic must find renamed curve
        gr2 = las.get_curve_by_mnemonic("GR_2")
        assert gr2 is not None
        assert gr2.unit == "GAPI"
        assert gr2.original_mnemonic == "GR"

        # to_dict should have consistent curves and curves_order
        d = las.to_dict()
        dict_mnemonics = [c["mnemonic"] for c in d["curves"]]
        assert dict_mnemonics == d["curves_order"]

    def test_unsupported_version_warns_but_reads(self, tmp_path: Path) -> None:
        """Test that unsupported LAS version emits warning but still reads."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   4.0  : FUTURE VERSION\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "v4.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("not officially supported" in str(x.message) for x in w)

        # Data should still be read successfully
        assert "DEPT" in data["logs"]
        assert data["logs"]["DEPT"][0] == 100.0

    def test_unsupported_version_warns_as_object(self, tmp_path: Path) -> None:
        """Test warning from read_las_file_as_object for unsupported version."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   5.0  : FUTURE VERSION\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "v5.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
            assert any("not officially supported" in str(x.message) for x in w)

        # Data should still be read
        assert "DEPT" in las.logs

    # --- T6: Non-numeric VERS branch (reader.py:129-134) ---
    def test_non_numeric_vers(self, tmp_path: Path) -> None:
        """Test that non-numeric VERS string (e.g. 'CWLS') reads without error.

        Exercises reader.py:129-134 — the ValueError handler for non-numeric
        version strings. Common in LAS 1.2 files with plain text identifiers.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   CWLS LOG ASCII STANDARD  : VERSION 1.2\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "non_numeric_vers.las"
        test_file.write_text(content, encoding="utf-8")

        # Should not crash — non-numeric VERS is logged and file reads anyway
        data = read_las_file(test_file)
        assert "DEPT" in data["logs"]
        assert data["logs"]["DEPT"][0] == 100.0

    def test_non_numeric_vers_as_object(self, tmp_path: Path) -> None:
        """Test non-numeric VERS with read_las_file_as_object."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   CWLS LOG ASCII STANDARD  : VERSION 1.2\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "non_numeric_vers_obj.las"
        test_file.write_text(content, encoding="utf-8")

        # Should not crash — non-numeric VERS just gets logged at debug
        las = read_las_file_as_object(test_file)
        assert las.version.vers == "CWLS LOG ASCII STANDARD"
        assert "DEPT" in las.logs

    def test_max_file_size_limit(self, tmp_path: Path) -> None:
        """Test that max_file_size parameter rejects oversized files."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "size_test.las"
        test_file.write_text(content, encoding="utf-8")

        # Should succeed with generous limit
        data = read_las_file(test_file, max_file_size=10_000_000)
        assert "DEPT" in data["logs"]

        # Should fail with tiny limit
        with pytest.raises(LASReadError, match="Cannot read file"):
            read_las_file(test_file, max_file_size=10)

    # --- TEST-02: Non-numeric data triggers ValueError handler in _read_normal ---
    def test_non_numeric_data_normal_mode(self, tmp_path: Path) -> None:
        """Test that non-numeric values in normal mode trigger null_value substitution."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  BAD\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "bad_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # BAD should be replaced with null_value (-999.25)
        assert data["logs"]["DT"][0] == -999.25
        # DEPT is fine
        assert data["logs"]["DEPT"][0] == 100.0
        # Next row should be fine
        assert data["logs"]["DT"][1] == 51.0

    # --- TEST-03: Wrapped-mode incomplete depth step padding ---
    def test_wrapped_incomplete_step_padding(self, tmp_path: Path) -> None:
        """Wrapped-mode incomplete final step (accumulation contract).

        II-4/R-6: under the n_curves-accumulation rewrite, a trailing
        partial buffer (fewer than curve_count values at EOF) cannot form
        a complete step — the orphan values are DISCARDED with the
        N-I-08-style "not accounted for" warning instead of being padded
        into a phantom step.  Only the complete first step survives; all
        arrays stay equal-length (the equal-length invariant is preserved
        for the steps that exist)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            # Only 2 values for the second step → incomplete trailing buffer
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_short.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
            # The N-I-08-style trailing-step warning replaces the old
            # per-curve "Padding" warnings (the partial step is discarded,
            # not padded).
            assert any("under-filled" in m or "not accounted for" in m for m in messages), (
                f"Expected trailing-step warning, got: {messages[-3:]}"
            )

        # All arrays should be same length after trimming
        sizes = [len(data["logs"][c]) for c in data["curves_order"]]
        assert len(set(sizes)) == 1
        # Only the complete first step survives
        assert len(data["logs"]["DEPT"]) == 1
        assert data["logs"]["DEPT"][0] == 100.0

    # --- TEST-04: Wrapped-mode depth line has >1 value ---
    def test_wrapped_depth_line_extra_values(self, tmp_path: Path) -> None:
        """Wrapped mode with a multi-value "depth" line (accumulation).

        II-4/R-6 (accepted accumulation contract): the depth-line flag
        protocol's "depth line has 2 values → warn+discard the extra"
        behavior is gone — _read_wrapped now uses an n_curves
        pending-value buffer, so `101.0 99.0` contributes BOTH values to
        the step stream: step 2 = [101.0, 99.0] (DT=99.0, not 51.0) and
        the trailing `51.0` is an incomplete final step that is discarded
        with the N-I-08-style "not accounted for" warning.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            # First depth line: 1 value (< curve_count) -> _detect_actual_wrap returns True
            "100.0\n"
            # DT for first step
            "50.0\n"
            # Second "depth" line: 2 values -> under accumulation both are data
            "101.0  99.0\n"
            # Trailing single value -> incomplete final step, discarded
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_extra_depth.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
            # The depth-line warn+discard is gone; the trailing incomplete
            # step emits the N-I-08-style warning instead.
            assert any("under-filled" in m or "not accounted for" in m for m in messages), (
                f"Expected trailing-step warning, got: {messages[-3:]}"
            )

        # DEPT gets the two flushed steps' depth values (100.0, 101.0)
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DEPT"][1] == 101.0
        # DT: step 2 carries 99.0 (the "extra" depth-line value is now data)
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DT"][1] == 99.0

    # --- TEST-11: Zero-curve early return ---
    def test_zero_curves_early_return(self, tmp_path: Path) -> None:
        """Test reading LAS file with empty curve section returns empty logs."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            "~A\n"
            "100.0\n"
        )
        test_file = tmp_path / "no_curves.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"] == {}
        assert data["curves_order"] == []

    # --- TEST-12: _detect_actual_wrap with no data after ~A ---
    def test_detect_wrap_no_data(self, tmp_path: Path) -> None:
        """Test _detect_actual_wrap returns True when no data lines exist.

        When no data follows ~A, _detect_actual_wrap defaults to True
        (data_reader.py:196). Verify that the returned dict has the
        expected structure: all required keys present, logs empty, and
        the version section correctly reflects WRAP=YES.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A\n"
            "# Just a comment, no data\n"
        )
        test_file = tmp_path / "wrap_no_data.las"
        test_file.write_text(content, encoding="utf-8")

        # Should not crash — _detect_actual_wrap defaults to True
        data = read_las_file(test_file)
        assert isinstance(data, dict)
        # F-M-21: Strengthen assertions — verify expected keys and structure.
        assert "version" in data
        assert "well" in data
        assert "logs" in data
        assert "curves_order" in data
        # No data lines were parsed — logs contain pre-allocated empty arrays.
        assert data["logs"] != {}
        assert len(data["logs"]["DEPT"]) == 0
        # curves_order should still list the declared curve
        assert data["curves_order"] == ["DEPT"]
        # Version section must reflect WRAP=YES
        assert data["version"]["VERS"] == "1.2"
        assert data["version"]["WRAP"] == "YES"
        # Well section should contain the NULL value
        assert data["well"]["NULL"] == "-999.25"

    # --- TEST-13: Wrapped-mode malformed data handling ---
    def test_wrapped_malformed_data_handlers(self, tmp_path: Path) -> None:
        """Non-numeric / non-finite values in wrapped-mode data are
        substituted with null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            # Depth line with non-numeric value (ValueError)
            "BAD\n"
            "50.0\n"
            "75.0\n"
        )
        test_file = tmp_path / "wrapped_bad.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # BAD in depth should become null_value
        assert data["logs"]["DEPT"][0] == -999.25
        # DT and GR should be fine
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["GR"][0] == 75.0

    # --- TEST-03: Non-numeric NULL value fallback in data_reader.py (lines 124-125, 189-190) ---
    def test_non_numeric_null_fallback_normal(self, tmp_path: Path) -> None:
        """Test that non-numeric NULL field falls back to -999.25 in normal mode.

        The NULL value in a LAS file should be numeric, but when it's not
        (e.g., "NONE"), the reader should fall back to -999.25.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   NONE  : NON-NUMERIC NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "non_numeric_null.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DEPT"][0] == 100.0
        # NULL value itself should be in the well section as string "NONE"
        assert data["well"]["NULL"] == "NONE"

    def test_non_numeric_null_fallback_wrapped(self, tmp_path: Path) -> None:
        """Test that non-numeric NULL field falls back to -999.25 in wrapped mode."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   BADNULL  : NON-NUMERIC NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_non_numeric_null.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Data should be read correctly despite bad NULL value
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DEPT"][1] == 101.0
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DT"][1] == 51.0
        assert data["well"]["NULL"] == "NON-NUMERIC NULL VALUE"

    # --- F-19: Wrapped-mode section transition break ---
    def test_wrapped_section_after_ascii_stops_data(self, tmp_path: Path) -> None:
        """Test that a new section after ~A in wrapped mode stops data collection.

        Exercises the break in _read_wrapped when a section
        other than ~A is encountered during ASCII data reading.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "101.0\n"
            "51.0\n"
            "~OTHER\n"
            "Some free text after data.\n"
        )
        test_file = tmp_path / "wrapped_section_break.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Should have 2 data points per curve (stopped before ~OTHER)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])

    # --- F-22: Curve deduplication name collision ---
    def test_curve_dedup_already_has_suffix(self, tmp_path: Path) -> None:
        """Test curve deduplication when curves already have _N suffix.

        Exercises data_reader.py:86 — name collision edge case where
        curves already use _2 suffix notation.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " GR_2.API  :  Pre-existing suffix 1\n"
            " GR_2.API  :  Pre-existing suffix 2\n"
            "~A  GR_2  GR_2\n"
            "100.0  10.0\n"
        )
        test_file = tmp_path / "already_suffixed.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Duplicate curve mnemonic" in str(x.message) for x in w)

        # First GR_2 keeps its name, second becomes GR_2_2
        assert "GR_2" in data["logs"]
        assert "GR_2_2" in data["logs"]
        np.testing.assert_array_equal(data["logs"]["GR_2"], [100.0])
        np.testing.assert_array_equal(data["logs"]["GR_2_2"], [10.0])

    # --- T7: Cross-base collision while-loop (data_reader.py:149-165) ---
    def test_cross_base_collision_dedup(self, tmp_path: Path) -> None:
        """Test dedup when original name matches a previously generated suffix.

        Input curves_order: ["DEPT", "DEPT", "DEPT_2"]
        Expected: ["DEPT", "DEPT_2", "DEPT_2_2"] — the second DEPT becomes
        DEPT_2 which collides with the third original name DEPT_2,
        so the third gets renamed to DEPT_2_2.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth 1\n"
            " DEPT.M   :  Depth 2\n"
            " DEPT_2.M :  Pre-named curve\n"
            "~A  DEPT  DEPT  DEPT_2\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "cross_base_collision.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            # Should have 2 warnings: one for duplicate DEPT, one for cross-base
            dup_warnings = [x for x in w if "Duplicate curve mnemonic" in str(x.message)]
            assert len(dup_warnings) >= 2

        # Expected order: DEPT, DEPT_2, DEPT_2_2
        assert "DEPT" in data["logs"]
        assert "DEPT_2" in data["logs"]
        assert "DEPT_2_2" in data["logs"]
        # All three must be distinct keys
        assert len(data["logs"]) == 3
        # Verify data values
        np.testing.assert_array_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(data["logs"]["DEPT_2"], [10.0, 11.0])
        np.testing.assert_array_equal(data["logs"]["DEPT_2_2"], [20.0, 21.0])


class TestMaxTotalElementsGuard:
    """F-023: MAX_TOTAL_ELEMENTS allocation guard tests."""

    def test_max_total_elements_normal_mode(self, tmp_path: Path) -> None:
        """Test that curve_count * data_line_count > MAX_TOTAL_ELEMENTS raises error."""
        from unittest import mock

        # Build a file with 5 curves and 5 data lines
        curves_block = "\n".join(f" C{i:02d}.UNIT  :  Curve {i}" for i in range(5))
        # 5 data lines to get 5*5=25 elements
        data_block = "\n".join(f"{100.0 + i}  1.0  2.0  3.0  4.0" for i in range(5))
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n" + curves_block + "\n~A" + "  C00" * 5 + "\n" + data_block + "\n"
        )
        test_file = tmp_path / "max_total.las"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_TOTAL_ELEMENTS to 20: 5 curves * 5 lines = 25 > 20
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 20):
            with pytest.raises(LASParseError, match="Total allocation"):
                read_las_file(test_file)

    def test_max_total_elements_wrapped_mode(self, tmp_path: Path) -> None:
        """Test MAX_TOTAL_ELEMENTS in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " C00.M  :  Curve 0\n"
            " C01.M  :  Curve 1\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "101.0\n"
            "51.0\n"
            "102.0\n"
            "52.0\n"
        )
        test_file = tmp_path / "max_total_wrap.las"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_TOTAL_ELEMENTS to a low value
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 5):
            with pytest.raises(LASParseError, match="Total allocation"):
                read_las_file(test_file)


class TestPermissionErrorHandler:
    """F-024: PermissionError handler tests."""

    def test_unreadable_file_raises_las_read_error(self, tmp_path: Path) -> None:
        """Test that an unreadable file raises LASReadError, not PermissionError."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "noperm.las"
        test_file.write_text(content, encoding="utf-8")

        try:
            # Make file unreadable
            os.chmod(test_file, 0o000)
            with pytest.raises(LASReadError, match="Cannot read"):
                read_las_file(test_file)
        finally:
            # Restore permissions so tmp_path cleanup works
            os.chmod(test_file, 0o644)

    def test_unreadable_file_as_object(self, tmp_path: Path) -> None:
        """Test read_las_file_as_object with unreadable file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "noperm_obj.las"
        test_file.write_text(content, encoding="utf-8")

        try:
            os.chmod(test_file, 0o000)
            with pytest.raises(LASReadError, match="Cannot read"):
                read_las_file_as_object(test_file)
        finally:
            os.chmod(test_file, 0o644)


class TestFortranDExponent:
    """T2/F-11: Fortran D-exponent format (e.g., '1.0D+03') in data."""

    def test_d_exponent_values_parsed_correctly(self, tmp_path: Path) -> None:
        """Test that Fortran D-exponent notation is converted via _to_finite_float.

        Some scientific software writes numbers in D-exponent format
        (e.g., '1.0D+03' instead of '1.0E+03').  _to_finite_float
        replaces D/d with E/e before calling float().
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma\n"
            "~A  DEPT  DT  GR\n"
            "100.0  1.0D+03  75.5\n"
            "101.0  2.5D-01  76.0\n"
            "102.0  3.14D0   77.0\n"
        )
        test_file = tmp_path / "d_exponent.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DT"], [1000.0, 0.25, 3.14], rtol=1e-10)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.5, 76.0, 77.0])

    def test_lowercase_d_exponent(self, tmp_path: Path) -> None:
        """Test that lowercase 'd' exponent is also handled."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  1.0d+03\n"
        )
        test_file = tmp_path / "d_lower.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == 1000.0

    def test_mixed_case_d_exponent_in_wrapped_mode(self, tmp_path: Path) -> None:
        """Test D-exponent in wrapped mode."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "1.0D+03\n"
            "101.0\n"
            "5.0d+02\n"
        )
        test_file = tmp_path / "d_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == 1000.0
        assert data["logs"]["DT"][1] == 500.0

    def test_d_exponent_null_value_different_value(self, tmp_path: Path) -> None:
        """F-06: Test that a non-default D-notation NULL value is used correctly.

        NULL. -99.99D0 with BAD data → DT should be -99.99, NOT the
        hardcoded default -999.25. This verifies the D-notation path
        actually produces the correct value.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -99.99D0       : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  BAD\n"
        )
        test_file = tmp_path / "d_null_99.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # -99.99D0 = -99.99, not -999.25
        assert data["logs"]["DT"][0] == -99.99


class TestWrappedPathologicalMisalignment:
    """T6/G-09: _read_wrapped pathological misalignment path."""

    def test_pathological_misalignment_parses_cleanly(self, tmp_path: Path) -> None:
        """II-4 (accepted): the F-06 pathological-misalignment hard-fail is
        removed by the n_curves-accumulation rewrite — the "extra" values
        on the second depth line are now data, and the whole section
        accumulates into 2 clean steps instead of raising LASParseError.

        The F-06 guard's premise ("depth line had extra values" = certain
        misalignment) only held under the depth-line flag protocol; under
        accumulation the same values align into complete steps."""
        # Curve count = 4 (DEPT + 3 non-depth curves)
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            " SP.MV    :  Spontaneous Potential\n"
            "~A\n"
            # Step 1: depth + 3 continuation values
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "10.0\n"
            # Step 2: "depth" line with 3 values + 1 continuation value —
            # previously depth_had_extra + short next line → LASParseError
            "101.0  99.0  88.0\n"
            "77.0\n"
        )
        test_file = tmp_path / "pathological.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # 2 clean steps: [100,50,75,10] and [101,99,88,77]
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 99.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 88.0])
        np.testing.assert_allclose(data["logs"]["SP"], [10.0, 77.0])


class TestDetectActualWrapDLM:
    """T8/G-11: _detect_actual_wrap DLM-aware branch."""

    def test_wrap_yes_with_tab_delimiter(self, tmp_path: Path) -> None:
        """Test truly-wrapped data with DLM=TAB and one value per line.

        When WRAP=YES and DLM=TAB, the DLM-aware branch at
        data_reader.py:190-191 uses strip-and-filter split to handle
        trailing delimiters. The first data line has 1 value (< curve_count)
        → _detect_actual_wrap returns True and parsing uses _read_wrapped.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.    YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM .   TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n50.0\n101.0\n51.0\n"
        )
        test_file = tmp_path / "wrapped_tab_dlm.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])

    def test_wrap_yes_with_mislabeled_data_tab_dlm(self, tmp_path: Path) -> None:
        """Test WRAP=YES with mislabeled non-wrapped data and TAB delimiter.

        The file claims WRAP=YES but each data line has full column set
        separated by TAB.  _detect_actual_wrap should detect non-wrapped
        and route to _read_normal.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            " DLM .   TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0\t50.0\n"
            "101.0\t51.0\n"
        )
        test_file = tmp_path / "mislabeled_tab_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])


class TestReaderLAS12WellExtraction:
    """T1/F-10: LAS 1.2 well section value extraction via reader."""

    def test_las12_well_values_read_correctly(self, tmp_path: Path) -> None:
        """End-to-end test of LAS 1.2 well value extraction through reader.

        Verifies that read_las_file correctly extracts both numeric and
        non-numeric well values from LAS 1.2 files using auto-detection.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            # Numeric fields in spec format (value before colon)
            " STRT.M   1670.0000 : START DEPTH\n"
            " STOP.M   1660.0000 : STOP DEPTH\n"
            " STEP.M   -0.1250    : STEP\n"
            " NULL.    -999.25    : NULL VALUE\n"
            # Non-numeric fields in lasio convention (value after colon)
            " COMP.    COMPANY : ANY OIL COMPANY LTD.\n"
            " WELL.    WELL : ANY ET AL OIL WELL #12\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "1670.0\n"
        )
        test_file = tmp_path / "las12_well.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["well"]["STRT"] == "1670.0000"
        assert data["well"]["STOP"] == "1660.0000"
        assert data["well"]["NULL"] == "-999.25"
        assert data["well"]["COMP"] == "ANY OIL COMPANY LTD."
        assert data["well"]["WELL"] == "ANY ET AL OIL WELL #12"


class TestMaxLimitsGuards:
    """F-026: MAX_DATA_LINES / MAX_CURVES guard tests."""

    def test_max_data_lines_normal_mode(self, tmp_path: Path) -> None:
        """Test MAX_DATA_LINES guard raises error."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "max_lines.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 0):
            with pytest.raises(LASParseError, match="Data line count"):
                read_las_file(test_file)

    def test_max_curves_normal_mode(self, tmp_path: Path) -> None:
        """Test MAX_CURVES guard raises error."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
        )
        test_file = tmp_path / "max_curves.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_CURVES", 1):
            with pytest.raises(LASParseError, match=r"(?i)curve count"):
                read_las_file(test_file)

    def test_max_curves_wrapped_mode(self, tmp_path: Path) -> None:
        """Test MAX_CURVES guard in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
        )
        test_file = tmp_path / "max_curves_wrap.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_CURVES", 1):
            with pytest.raises(LASParseError, match=r"(?i)curve count"):
                read_las_file(test_file)


class TestToFiniteFloat:
    """F-029: _to_finite_float non-finite value guard tests."""

    def test_nan_value_returns_null(self, tmp_path: Path) -> None:
        """Test that 'nan' in data returns null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  nan\n"
        )
        test_file = tmp_path / "nan_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == -999.25

    def test_inf_value_returns_null(self, tmp_path: Path) -> None:
        """Test that 'inf' in data returns null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  inf\n"
        )
        test_file = tmp_path / "inf_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == -999.25

    def test_neg_inf_value_returns_null(self, tmp_path: Path) -> None:
        """Test that '-inf' in data returns null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  -inf\n"
        )
        test_file = tmp_path / "neginf_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == -999.25

    def test_overflow_exponent_returns_null(self, tmp_path: Path) -> None:
        """Test that large overflow exponent '1e309' returns null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  1e309\n"
        )
        test_file = tmp_path / "overflow_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == -999.25


class TestMetadataOnlyLas:
    """CF-021: Metadata-only LAS file (no ~A section)."""

    def test_las_without_ascii_section_parses(self, tmp_path: Path) -> None:
        """Test that LAS file with ~V and ~C but no ~A parses correctly."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            " STOP.M   200.0 : STOP DEPTH\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
        )
        test_file = tmp_path / "no_ascii.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["curves_order"] == ["DEPT", "DT"]
        # logs contain zero-length arrays (pre-allocated with np.zeros(0))
        assert len(data["logs"]) == 2
        assert len(data["logs"]["DEPT"]) == 0
        assert len(data["logs"]["DT"]) == 0

    def test_las_without_ascii_section_as_object(self, tmp_path: Path) -> None:
        """Test LASFile object from metadata-only file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
        )
        test_file = tmp_path / "no_ascii_obj.las"
        test_file.write_text(content, encoding="utf-8")

        las = read_las_file_as_object(test_file)
        assert las.curves_order == ["DEPT"]
        # logs contains zero-length array (pre-allocated with np.zeros(0))
        assert "DEPT" in las.logs
        assert len(las.logs["DEPT"]) == 0
        assert las.well["STRT"] == "100.0"


class TestArrayTrimmingOvercount:
    """F24: Tests for array trimming when pre-scan over-counts data lines.

    The trimming branch (data_reader.py:354-359) runs when current_line <
    data_line_count. This happens when the parser's _pre_scan counts data
    lines across ALL ~A sections but _read_normal stops at the first
    non-A section header (~OTHER, ~P, etc.).
    """

    def test_multi_a_section_trimming(self, tmp_path: Path) -> None:
        """Dup-~A-drop: a second ~A block after a non-~A section must NOT
        be ingested; only the first ~A block's data survives.

        The parser's pre-scan counts only the FIRST contiguous ~A block
        (data_line_count == 2), so the trim branch never fires — the
        primary guard here is the first-block-only ingestion contract
        (F-EX-02 / I2-06): data in later ~A blocks after a non-~A
        section is DROPPED, and the reader emits the
        "Multiple ~A data sections encountered" warning.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
            "~OTHER\n"
            "Freeform text.\n"
            "~A  DEPT  DT\n"
            "200.0  60.0\n"
        )
        test_file = tmp_path / "multi_a_trim.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        # Only first section's 2 data lines should be present
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])
        # Dup-~A-drop guard: the second ~A block's values (200.0 / 60.0)
        # must never appear in the ingested arrays.
        assert 200.0 not in data["logs"]["DEPT"]
        assert 60.0 not in data["logs"]["DT"]
        # The reader must announce the drop via the warnings API.
        assert any("Multiple ~A data sections" in str(x.message) for x in w)


class TestExtraColumnWarning:
    """F34: Tests for extra-column warning in _read_normal.

    The warning at data_reader.py:330-331 triggers when a data line has
    MORE columns than curve_count. Exercises the `warned_extra` flag path.
    """

    def test_extra_columns_warning_normal(self, tmp_path: Path) -> None:
        """Test extra-column warning triggered in non-wrapped mode."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0  75.0  99.0\n"  # 4 values but only 2 curves
            "101.0  51.0\n"
        )
        test_file = tmp_path / "extra_cols.las"
        test_file.write_text(content, encoding="utf-8")

        # F-089: warnings now issued via warnings.warn, not logger.warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Extra columns are discarded" in str(x.message) for x in w)

        # Data should be read correctly: extra values are silently skipped
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DEPT"][1] == 101.0
        assert data["logs"]["DT"][1] == 51.0

    def test_extra_columns_only_first_row_warns(self, tmp_path: Path) -> None:
        """Test extra-column warning fires only once (warned_extra flag)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0  1.0  2.0\n"  # 3 values, 1 curve
            "101.0  3.0  4.0\n"  # 3 values, 1 curve — should NOT warn again
        )
        test_file = tmp_path / "extra_once.las"
        test_file.write_text(content, encoding="utf-8")

        # F-089: warnings now issued via warnings.warn, not logger.warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_las_file(test_file)
            # The warning text should appear exactly once
            warning_texts = [str(x.message) for x in w]
            assert sum("Extra columns are discarded" in t for t in warning_texts) == 1


class TestLAS30StructuredDataValues:
    """F-15: Value validation for structured data sections in sample_las3.0_spec.las.

    Tests that data_sections values from the spec-conformant LAS 3.0 file
    are parsed correctly on INITIAL READ (not roundtrip). The roundtrip
    path has a known key-name corruption issue (per-section curve names
    become global curve names on re-read), so only first-read validation
    is performed here.
    """

    def assert_section_values(
        self,
        sections: list[dict],
        expected_name: str,
        expected_type: str,
        expected_data: dict[str, list[float]],
    ) -> dict:
        """Find a section by name and assert its numeric data."""
        for section in sections:
            if section["name"] == expected_name:
                assert section["section_type"] == expected_type
                data = section["data"]
                for key, expected_values in expected_data.items():
                    assert key in data, (
                        f"Key '{key}' missing in {expected_name} section. "
                        f"Available keys: {list(data.keys())}"
                    )
                    np.testing.assert_array_almost_equal(
                        data[key],
                        np.array(expected_values),
                        err_msg=f"{expected_name}.{key} value mismatch",
                    )
                return section
        pytest.fail(f"Section '{expected_name}' not found in data_sections")

    def test_las30_spec_drilling_data(self, test_data_dir: Path) -> None:
        """Validate Drilling data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "DRILLING",
            "DRILLING_DATA",
            {
                "DEPT": [322.02, 323.05],
                "DIST": [1.02, 2.05],
                "HRS": [0.0, 0.1],
                "ROP": [24.0, 37.5],
                "WOB": [3.0, 2.0],
                "RPM": [59.0, 69.0],
                "TQ": [111.0, 118.0],
                "PUMP": [1199.0, 1182.0],
                "TSPM": [179.0, 175.0],
                "GPM": [879.0, 861.0],
                "ECD": [8.73, 8.73],
                "TBR": [39.0, 202.0],
            },
        )

    def test_las30_spec_core1_data(self, test_data_dir: Path) -> None:
        """Validate CORE[1] data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "CORE[1]",
            "CORE_DATA",
            {
                "CORET": [545.50, 551.20, 575.00],
                "COREB": [550.60, 554.90, 595.00],
            },
        )

    def test_las30_spec_core2_data(self, test_data_dir: Path) -> None:
        """Validate CORE[2] data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "CORE[2]",
            "CORE_DATA",
            {
                "CORET": [655.50, 661.20, 675.00],
                "COREB": [660.60, 664.90, 695.00],
            },
        )

    def test_las30_spec_inclinometry_data(self, test_data_dir: Path) -> None:
        """Validate Inclinometry data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "INCLINOMETRY",
            "INCLINOMETRY_DATA",
            {
                "MD": [0.00, 100.00, 200.00, 300.00, 400.00, 500.00, 600.00],
                "TVD": [0.00, 100.00, 198.34, 295.44, 390.71, 482.85, 571.90],
                "AZIM": [290.00, 234.00, 284.86, 234.21, 224.04, 224.64, 204.39],
                "DEVI": [0.00, 0.00, 1.43, 2.04, 3.93, 5.88, 7.41],
            },
        )

    def test_las30_spec_tops_data(self, test_data_dir: Path) -> None:
        """Validate Tops data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "TOPS",
            "TOPS_DATA",
            {
                "TOPT": [545.50, 602.00, 615.00],
                "TOPB": [602.00, 615.00, 655.00],
            },
        )

    def test_las30_spec_perforations_data(self, test_data_dir: Path) -> None:
        """Validate Perforations data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "PERFORATIONS",
            "PERFORATIONS_DATA",
            {
                "PERFT": [545.50, 551.20, 575.00],
                "PERFB": [550.60, 554.90, 595.00],
                "PERFD": [12.0, 12.0, 12.0],
            },
        )

    def test_las30_spec_string_data_in_sections(self, test_data_dir: Path) -> None:
        """Validate string_data within structured data sections.

        CORE[1] has CDES with string descriptions; Core[2] has same.
        Perforations has PERFT (charge type) as string.
        """
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)

        # Find CORE[1] section (parser uppercases section names)
        core1 = None
        for ds in las.data_sections:
            if ds.name == "CORE[1]":
                core1 = ds
                break
        assert core1 is not None, "CORE[1] section not found"
        assert "CDES" in core1.string_data, (
            f"CDES missing from CORE[1] string_data. Keys: {list(core1.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            core1.string_data["CDES"],
            np.array(["Long cylindrical hunk of rock", "Long broken hunk of rock", "Debris only"]),
        )

        # Verify section count and names (parser uppercases all section names)
        section_names = [ds.name for ds in las.data_sections]
        # Should include 8 sections: DRILLING, CORE[1], CORE[2], INCLINOMETRY, TEST, TOPS, PERFORATIONS, ASCII
        assert "DRILLING" in section_names
        assert "CORE[1]" in section_names
        assert "CORE[2]" in section_names
        assert "INCLINOMETRY" in section_names
        assert "TEST" in section_names
        assert "TOPS" in section_names
        assert "PERFORATIONS" in section_names
        assert len(las.data_sections) == 8

        # Validate CORE[2] CDES string_data
        core2 = None
        for ds in las.data_sections:
            if ds.name == "CORE[2]":
                core2 = ds
                break
        assert core2 is not None, "CORE[2] section not found"
        assert "CDES" in core2.string_data, (
            f"CDES missing from CORE[2] string_data. Keys: {list(core2.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            core2.string_data["CDES"],
            np.array(["Long cylindrical hunk of rock", "Long broken hunk of rock", "Debris only"]),
        )

        # Validate TEST BLOWD string_data
        test_section = None
        for ds in las.data_sections:
            if ds.name == "TEST":
                test_section = ds
                break
        assert test_section is not None, "TEST section not found"
        assert "BLOWD" in test_section.string_data, (
            f"BLOWD missing from TEST string_data. Keys: {list(test_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            test_section.string_data["BLOWD"],
            np.array(["Weak Blow", "Strong Blow", "Blow Out"]),
        )

        # Validate TOPS TOPN string_data
        tops_section = None
        for ds in las.data_sections:
            if ds.name == "TOPS":
                tops_section = ds
                break
        assert tops_section is not None, "TOPS section not found"
        assert "TOPN" in tops_section.string_data, (
            f"TOPN missing from TOPS string_data. Keys: {list(tops_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            tops_section.string_data["TOPN"],
            np.array(["Viking", "Colony", "Basal Quartz"]),
        )

        # Validate PERFORATIONS PERFT_2 string_data (PERFT appears twice:
        # once as float depth, once as string charge type — the string
        # one gets renamed to PERFT_2 by per-section dedup)
        perfs_section = None
        for ds in las.data_sections:
            if ds.name == "PERFORATIONS":
                perfs_section = ds
                break
        assert perfs_section is not None, "PERFORATIONS section not found"
        assert "PERFT_2" in perfs_section.string_data, (
            f"PERFT_2 missing from PERFORATIONS string_data. Keys: {list(perfs_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            perfs_section.string_data["PERFT_2"],
            np.array(["BIG HOLE", "BIG HOLE", "BIG HOLE"]),
        )


class TestGetNullValue:
    """F-06: Direct unit tests for _get_null_value Fortran D-notation handling."""

    def test_standard_null_value(self) -> None:
        """Standard -999.25 null value parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "-999.25"}
        result = _get_null_value(well)
        assert result == -999.25

    def test_d_notation_with_exponent(self) -> None:
        """Fortran D-notation with explicit exponent (+03) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "1.0D+03"}
        result = _get_null_value(well)
        assert result == 1000.0

    def test_d_notation_negative_exponent(self) -> None:
        """Fortran D-notation with negative exponent (D-02) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "3.14D-02"}
        result = _get_null_value(well)
        assert result == 0.0314

    def test_missing_null_key_returns_default(self) -> None:
        """Missing NULL key returns default_float."""
        from pylasdev.data_reader import _get_null_value

        well: dict[str, str] = {}
        result = _get_null_value(well)
        assert result == -999.25


class TestMaxTokensPerLineGuard:
    """M68: Tests for MAX_TOKENS_PER_LINE DoS guard.

    MAX_TOKENS_PER_LINE (data_reader.py:37, =MAX_CURVES=100_000)
    caps the number of tokens parsed from a single data line to
    prevent resource exhaustion from pathological input. Used at
    7 call sites across data_reader.py and dev_reader.py.
    Previously had zero test coverage (grep confirmed).
    """

    def test_token_count_capped_in_normal_mode(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE caps tokens in normal (non-wrapped) mode.

        Mock MAX_TOKENS_PER_LINE to a low value and verify that extra
        tokens on a data line are silently truncated instead of causing
        resource exhaustion.
        """
        from unittest import mock

        # Build a file with 3 declared curves but 4+ space-separated
        # tokens on a data line.
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A  DEPT  DT  GR\n"
            "100.0  50.0  75.0  99.0\n"
            "101.0  51.0  76.0  88.0\n"
        )
        test_file = tmp_path / "token_cap_normal.las"
        test_file.write_text(content, encoding="utf-8")

        # Patch MAX_TOKENS_PER_LINE to 2: split() produces at most 3 tokens.
        # Data line "100.0  50.0  75.0  99.0" → "100.0", "50.0", "75.0  99.0"
        # The 4th value is merged into the 3rd token, causing a float parse
        # failure → substituted with null_value.
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_las_file(test_file)
            # Extra 4th value on each line should be silently handled
            assert len(data["logs"]["DEPT"]) == 2
            assert data["logs"]["DEPT"][0] == 100.0
            # Discriminator: with the cap at 2, the line's 3rd token is
            # "75.0  99.0" (4th value merged into it) — the float parse
            # fails and GR is null-filled.  Removing the cap guard leaves
            # GR[0] == 75.0, so this assertion fails on cap deletion.
            assert data["logs"]["GR"][0] == -999.25

    def test_token_count_capped_in_wrapped_mode(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE caps tokens in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0  99.0  88.0\n"
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "token_cap_wrapped.las"
        test_file.write_text(content, encoding="utf-8")

        # Patch MAX_TOKENS_PER_LINE low: wrapped-mode split also capped
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 1):
            data = read_las_file(test_file)
            assert data["logs"]["DEPT"][0] == 100.0
            # Discriminator: with cap=1 the "50.0  99.0  88.0" line splits
            # to ["50.0"] (99.0/88.0 truncated away), so step 2's DEPT
            # value is lost to the null fill.  Removing the cap guard
            # leaves DEPT == [100.0, 99.0, 101.0], so this assertion
            # (DEPT[1] == -999.25) fails on cap deletion.
            assert data["logs"]["DEPT"][1] == -999.25

    def test_token_cap_dev_reader(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE capped in DEV reader (dev_reader.py:719)."""
        from unittest import mock

        content = "MD TVD X Y Z\n0.0 0.0 100.0 200.0 300.0\n100.0 99.0 101.0 201.0 301.0\n"
        test_file = tmp_path / "token_cap_dev.dev"
        test_file.write_text(content, encoding="utf-8")

        # Patch data_reader.MAX_TOKENS_PER_LINE: dev_reader fetches at
        # runtime via _resolve_max_tokens_per_line (F-DVR-01 fix).
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_dev_file(test_file)
            # split(maxsplit=2) produces at most 3 tokens from 5-space line
            assert "MD" in data
            assert "TVD" in data
            # Discriminator: with cap=2 the last three header words collapse
            # into a single "X Y Z" column; removing the cap guard splits
            # them back into X/Y/Z, so "X Y Z" disappears and this
            # assertion fails on cap deletion.
            assert "X Y Z" in data


class TestMaxLimitsAtLimit:
    """M69: At-limit tests for MAX_* reader guards.

    F-M01 changed data_reader.py operator from ``>`` to ``>=`` for
    MAX_CURVES and MAX_DATA_LINES (matching parser.py and models.py).
    At-limit cases now REJECT instead of passing — the limit is exclusive.
    MAX_TOTAL_ELEMENTS still uses ``>`` so at-limit still passes.
    """

    def test_max_curves_at_limit_rejected(self, tmp_path: Path) -> None:
        """MAX_CURVES set exactly to the file's curve count — must reject."""
        from unittest import mock

        # File has exactly 2 curves
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
        )
        test_file = tmp_path / "at_limit_curves.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_CURVES=2 matches actual curve count — F-M01 now rejects at-limit
        with mock.patch("pylasdev.data_reader.MAX_CURVES", 2):
            with pytest.raises(LASParseError, match="exceeds maximum"):
                read_las_file(test_file)

    def test_max_data_lines_at_limit_accepted(self, tmp_path: Path) -> None:
        """MAX_DATA_LINES set exactly to the file's data line count — must accept.

        F-MDR-01: Changed _read_normal from ``>=`` to ``>`` for consistency
        with models.py which uses ``>`` at all 8 MAX_DATA_LINES guard sites
        (accepts at exactly the limit).  The F-M01 fix comment that claimed
        models.py uses ``>=`` was incorrect.
        """
        from unittest import mock

        # File has exactly 1 data line
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "at_limit_lines.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_DATA_LINES=1 matches actual data line count — F-MDR-01 accepts at-limit
        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 1):
            data = read_las_file(test_file)
            assert len(data["logs"]["DEPT"]) == 1
            assert data["logs"]["DEPT"][0] == 100.0

    def test_max_total_elements_at_limit_passes(self, tmp_path: Path) -> None:
        """MAX_TOTAL_ELEMENTS set exactly to curves*lines — must pass."""
        from unittest import mock

        # File has 2 curves x 1 data line = 2 elements
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
        )
        test_file = tmp_path / "at_limit_total.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_TOTAL_ELEMENTS=2 matches curves*lines — should pass
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 2):
            data = read_las_file(test_file)
            assert len(data["logs"]["DEPT"]) == 1
            assert data["logs"]["DEPT"][0] == 100.0

    def test_max_curves_wrapped_at_limit_rejected(self, tmp_path: Path) -> None:
        """MAX_CURVES at limit in wrapped mode — must reject."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
        )
        test_file = tmp_path / "at_limit_wrap_curves.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_CURVES=2 matches actual — F-M01 now rejects at-limit
        with mock.patch("pylasdev.data_reader.MAX_CURVES", 2):
            with pytest.raises(LASParseError, match="exceeds maximum"):
                read_las_file(test_file)


# ============================================================
# Production Check Regression Tests
# ============================================================


class TestProductionCheckReaderFixes:
    """Regression tests for production check fixes in data_reader.py and reader.py."""

    # --- F-027 (MEDIUM): _get_null_value silences LASParseError ---

    def test_non_finite_null_value_raises_las_parse_error(self) -> None:
        """F-027: Non-finite NULL value ('nan') raises LASParseError.

        Before the fix, LASParseError was caught by the except clause
        (lines 240), falling back to -999.25 silently. Now the error
        propagates so callers know the NULL is corrupt.
        """
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "nan"}
        with pytest.raises(LASParseError, match="NULL value must be a finite number"):
            _get_null_value(well)

    def test_inf_null_value_raises_las_parse_error(self) -> None:
        """F-027: NULL value 'inf' raises LASParseError."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "inf"}
        with pytest.raises(LASParseError, match="NULL value must be a finite number"):
            _get_null_value(well)

    def test_non_numeric_null_still_falls_back_to_default(self) -> None:
        """F-027: Non-numeric NULL ('NOT_A_NUMBER') still falls back to default.

        Regression check: the except clause still catches ValueError from
        float conversion. Only LASParseError was removed.
        """
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "NONSENSE"}
        result = _get_null_value(well)
        assert result == -999.25

    # --- F-211 (MEDIUM): _read_wrapped depth-step underestimation ---

    def test_wrapped_compact_all_zero(self, tmp_path: Path) -> None:
        """F-211: Compact wrapped file (2 lines/step, all zero data).

        The depth-step estimate max(1, ceil(N/curve_count)) must not
        undercount steps in compact wrapped format.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            # Depth step 1: 2 lines
            "100.0\n"
            "50.0  75.0\n"
            # Depth step 2: 2 lines
            "101.0\n"
            "51.0  76.0\n"
            # Depth step 3: 2 lines
            "102.0\n"
            "52.0  77.0\n"
        )
        test_file = tmp_path / "compact_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # All 3 depth steps must be parsed correctly
        assert len(data["logs"]["DEPT"]) == 3
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0, 52.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 76.0, 77.0])

    # --- E-F-018 (HIGH): depth_line state machine corruption regression test ---

    def test_wrapped_depth_line_state_machine_non_pathological_ef018(self, tmp_path: Path) -> None:
        """E-F-018 — ACCEPTED CONTRACT CHANGE (II-4/R-6): the n_curves-
        accumulation rewrite removes the depth-line state machine's
        "unrecoverable data misalignment" hard-fail.  The file previously
        hard-failed; it now parses as 3 valid steps.  The second "depth"
        line's extra value (1050.5) is data under accumulation, not
        corrupting junk.

        Test scenario (curve_count=4, DEPT + DT, GR, SP):
          Step 1: depth=1000.0, data=200.0,300.0,400.0  (normal baseline)
          Step 2: depth=1010.0 + extra 1050.5, data=210.0,310.0
          Step 3: depth=1020.0, data=220.0,320.0,420.0
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            " SP.MV    :  Spontaneous Potential\n"
            "~A\n"
            # Step 1 (normal baseline): one value per line
            "1000.0\n"
            "200.0  300.0  400.0\n"
            # Step 2: previously triggered the F-032 hard-fail (depth line
            # with 2 values + short next line); under accumulation the
            # 5 values flow into step 2 = [1010.0, 1050.5, 210.0, 310.0]
            "1010.0  1050.5\n"
            "210.0  310.0\n"
            # Step 3
            "1020.0\n"
            "220.0  320.0  420.0\n"
        )
        test_file = tmp_path / "ef018_wrapped_state_machine.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # 3 clean steps under accumulation (previously LASParseError)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1010.0, 1020.0])
        np.testing.assert_allclose(data["logs"]["DT"], [200.0, 1050.5, 220.0])
        np.testing.assert_allclose(data["logs"]["GR"], [300.0, 210.0, 320.0])
        np.testing.assert_allclose(data["logs"]["SP"], [400.0, 310.0, 420.0])

    # --- F-212 (MEDIUM): _desanitize_las_value unconditional _# strip ---

    def test_desanitize_disabled_preserves_hash_prefix(self) -> None:
        """F-212: With _DESANITIZE_ENABLED=False, _# values are preserved.

        When reading non-pylasdev files that genuinely contain _# in
        their data, the desanitize function must not strip the underscore.
        """
        import pylasdev.data_reader as dr_mod
        import pylasdev.parser as _parser_mod

        prior = _parser_mod._DESANITIZE_ENABLED
        try:
            # F-088: _DESANITIZE_ENABLED is now unified in parser.py;
            # setting parser's flag affects data_reader._desanitize_las_value.
            _parser_mod._DESANITIZE_ENABLED = False
            # Value starts with _# — should be preserved as-is
            result = dr_mod._desanitize_las_value("_#original_value")
            assert result == "_#original_value", (
                f"Expected '_#original_value' preserved, got {result!r}"
            )
            # Non-_# value unchanged
            assert dr_mod._desanitize_las_value("normal_value") == "normal_value"
        finally:
            _parser_mod._DESANITIZE_ENABLED = prior

    def test_desanitize_enabled_strips_hash_prefix(self) -> None:
        """F-212: Default behavior (enabled) strips _# prefix for roundtrip.

        When _DESANITIZE_ENABLED=True (default), _# prefix is stripped
        to restore the original #-prefixed value from the writer escape.
        """
        import pylasdev.data_reader as dr_mod
        import pylasdev.parser as _parser_mod

        # F-088: _DESANITIZE_ENABLED is now unified in parser.py.
        assert _parser_mod._DESANITIZE_ENABLED is True
        result = dr_mod._desanitize_las_value("_#test_value")
        assert result == "#test_value", f"Expected '#test_value' after desanitize, got {result!r}"

    # --- E-04 (MEDIUM): thread-local _DESANITIZE_ENABLED never reset ---

    def test_desanitize_flag_restored_after_false_read(self, tmp_path: Path) -> None:
        """E-04: read_las_file_as_object(desanitize=False) restores the flag.

        The thread-local _DESANITIZE_ENABLED flag is set before parser
        header parsing and must be restored afterwards.  Previously a
        desanitize=False read left the flag False on the current thread,
        silently changing the behavior of subsequent direct
        LASParser.parse() users that never passed desanitize=False.
        """
        import pylasdev.parser as _parser_mod

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " STRT.M   100.0   : START DEPTH\n"
            " STOP.M   200.0   : STOP DEPTH\n"
            " STEP.M   0.5     : STEP DEPTH\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.API   :  Gamma\n"
            "~A  DEPT  GR\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "e04_desanitize_restore.las"
        test_file.write_text(content, encoding="utf-8")

        # Record the thread-local default before the read.
        prior = _parser_mod._DESANITIZE_ENABLED
        read_las_file_as_object(test_file, desanitize=False)
        # The flag must be restored — the read must not leak desanitize=False
        # to subsequent same-thread callers.
        assert _parser_mod._DESANITIZE_ENABLED is prior, (
            "E-04: _DESANITIZE_ENABLED not restored after desanitize=False "
            f"read (got {_parser_mod._DESANITIZE_ENABLED}, expected {prior})"
        )
        # And a subsequent read with default desanitize=True still works.
        read_las_file_as_object(test_file)

    def test_desanitize_flag_restored_on_parse_error(self, tmp_path: Path) -> None:
        """E-04: flag restored even when parsing fails (finally semantics)."""
        import pylasdev.parser as _parser_mod

        test_file = tmp_path / "e04_parse_error.las"
        test_file.write_text("not a las file at all\n", encoding="utf-8")

        prior = _parser_mod._DESANITIZE_ENABLED
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)
        assert _parser_mod._DESANITIZE_ENABLED is prior

    # --- F-219 / ENC-03: LASEncodingError propagation ---

    def test_encoding_error_propagates_as_las_encoding_error(self, tmp_path: Path) -> None:
        """F-219/ENC-03: a genuine decoding failure propagates as
        LASEncodingError with an accurate message.

        F-219 previously pinned LASEncodingError → LASReadError.  ENC-03
        corrected that contract: read_with_encoding raises LASEncodingError
        with the file path and codec cause, and reader.py must NOT relabel
        it as a misleading 'size exceeded' LASReadError.  Callers now
        receive LASEncodingError, matching read_with_encoding's contract.
        """
        from pylasdev.exceptions import LASEncodingError

        content = b"\xff\xfe\x00\x01"  # Invalid encoding bytes
        test_file = tmp_path / "bad_encoding.las"
        test_file.write_bytes(content)

        # Without chardet and empty fallback chain, encoding fails
        from unittest import mock

        with mock.patch("pylasdev.encoding.FALLBACK_ENCODINGS", []):
            with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
                with pytest.raises(LASEncodingError, match="Failed to decode"):
                    read_las_file(test_file)


class TestG6DataReaderFixes:
    """G6 regression tests: D-01/D-02/D-03 wrap detection, L-03 {I}
    precision, N-I-08 mid-file under-fill, IT3-F-01 desanitize hoist,
    IT3-F-02 math.isfinite swap, IT3-F-03 wrapped pre-allocation."""

    # --- D-01 (HIGH): COMMA/TAB wrap misdetection ---

    def test_d01_comma_sparse_first_row_not_wrapped(self, tmp_path: Path) -> None:
        """D-01: WRAP=NO comma file with a sparse first row (trailing comma)
        must NOT be misdetected as wrapped.

        Pre-fix, the comma/tab path returned wrapped on any 1-value first
        line without corroboration — 50% of rows were lost and the depth
        value landed in the GR column.  The protocol-based detector sees the
        full rows that follow and classifies the file non-wrapped.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A DEPT DT GR\n"
            "100.0,\n"
            "101.0,50.0,75.0\n"
            "102.0,51.0,76.0\n"
            "103.0,52.0,77.0\n"
        )
        test_file = tmp_path / "d01_comma.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        assert len(data["logs"]["DEPT"]) == 4, (
            f"Expected 4 rows, got {len(data['logs']['DEPT'])} — file was misdetected as wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0, 103.0])
        np.testing.assert_allclose(data["logs"]["GR"], [-999.25, 75.0, 76.0, 77.0])

    def test_d01_tab_sparse_first_row_not_wrapped(self, tmp_path: Path) -> None:
        """D-01: same trigger with TAB delimiter."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A DEPT DT\n"
            "100.0\n"
            "101.0\t50.0\n"
            "102.0\t51.0\n"
        )
        test_file = tmp_path / "d01_tab.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 3
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])

    # --- D-02 (MEDIUM): two consecutive sparse rows ---

    def test_d02_two_sparse_rows_not_wrapped(self, tmp_path: Path) -> None:
        """D-02: WRAP=NO space file with TWO consecutive sparse rows must
        not be misdetected as wrapped (pre-fix, the 2-line F-M16
        corroboration returned wrapped on two sparse rows)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A DEPT DT GR\n"
            "100.0\n"
            "101.0\n"
            "102.0 50.0 75.0\n"
            "103.0 51.0 76.0\n"
            "104.0 52.0 77.0\n"
        )
        test_file = tmp_path / "d02_two_sparse.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 5, (
            f"Expected 5 rows, got {len(data['logs']['DEPT'])} — file was misdetected as wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0, 103.0, 104.0])

    # --- D-03 (MEDIUM): genuine WRAP=YES with overfull second line ---

    def test_d03_genuine_wrap_overfull_second_line(self, tmp_path: Path) -> None:
        """D-03: a genuine WRAP=YES file whose second data line has
        >= curve_count values must stay wrapped (pre-fix, the second-line
        corroboration misdetected it as non-wrapped and the F2 overflow
        handler became unreachable)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0 75.0 99.0\n"
            "101.0\n"
            "51.0 76.0\n"
        )
        test_file = tmp_path / "d03_overfull.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Still WRAPPED (declared YES + depth evidence in window [1,3,1,2]) —
        # the D-03 detection outcome is preserved.  The READ semantics
        # changed under the accepted n_curves-accumulation rewrite: the
        # overfull continuation's extra value (99.0) "simply starts the next
        # step" (iter-2 design), so step 2 = [99.0, 101.0, 51.0] and the
        # trailing 76.0 is an incomplete final step (discarded with warning).
        assert len(data["logs"]["DEPT"]) == 2, (
            f"Expected 2 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 99.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 101.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 51.0])

    # --- X-2 (O(n²) pending slicing): wide-line wrapped files ---

    def _build_wide_wrapped_file(self, tmp_path: Path, n_tokens: int) -> Path:
        """Build a WRAP=YES file whose first 4 data lines are 1-value
        depth evidence (window [1,1,1,1] → depth-later arm → wrapped) and
        whose 6th data line is a single *n_tokens*-wide continuation line.

        The wide line is what the pre-fix pending-buffer rewrite handled
        quadratically (``pending = pending[curve_count:]`` per flush).
        """
        giant = " ".join(str(i) for i in range(n_tokens))
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n1.0\n101.0\n2.0\n"
            "102.0\n" + giant + "\n"
        )
        path = tmp_path / f"x2_wide_{n_tokens}.las"
        path.write_text(content, encoding="utf-8")
        return path

    def test_x2_wide_wrapped_line_values_correct_and_fast(self, tmp_path: Path) -> None:
        """X-2: a crafted wide-line wrapped file must parse with the SAME
        value-order accumulation the pending protocol defines, and must do
        so in near-linear time (the pre-fix per-flush list slicing was
        O(n²) — a 90K-token line cost ~5-6s CPU; with the index pointer it
        is ~0.1s).  A reintroduced quadratic path blows the wall-clock
        bound below, so the regression fails on the old implementation."""
        import time

        n_tokens = 90_000
        test_file = self._build_wide_wrapped_file(tmp_path, n_tokens)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            start = time.perf_counter()
            data = read_las_file(test_file)
            elapsed = time.perf_counter() - start

        # Value-order accumulation: every flushed step takes the next
        # curve_count buffered values in order (line boundaries are
        # irrelevant to the protocol).  values = [100,1,101,2,102] +
        # range(n_tokens); steps = len(values) // 2.
        values = [100.0, 1.0, 101.0, 2.0, 102.0] + [float(i) for i in range(n_tokens)]
        n_steps = len(values) // 2
        np.testing.assert_allclose(data["logs"]["DEPT"], values[0::2][:n_steps])
        np.testing.assert_allclose(data["logs"]["DT"], values[1::2][:n_steps])
        assert len(data["logs"]["DEPT"]) == n_steps == 45002
        trail = [str(w.message) for w in rec if "not accounted for" in str(w.message)]
        assert trail, "expected the N-I-08 trailing-partial warning (1 leftover value)"
        # Near-linear bound: quadratic at 90K tokens ≈ 5s; linear ≈ 0.1s.
        assert elapsed < 2.5, f"wide wrapped line took {elapsed:.2f}s — quadratic pending slicing?"

    def test_x2_wide_wrapped_line_scales_linearly(self, tmp_path: Path) -> None:
        """X-2: doubling the wide-line token count must roughly double the
        read time (linear) — NOT quadruple it (the pre-fix per-flush
        ``pending[curve_count:]`` slice was quadratic, ~4.0x per 2x).  The
        ratio uses the minimum of 3 timed reads per size so a transient CI
        spike inflates the denominator instead of failing the assert."""
        import time

        sizes = (30_000, 60_000)
        best: dict[int, float] = {}
        for n in sizes:
            test_file = self._build_wide_wrapped_file(tmp_path, n)
            timings: list[float] = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(4):  # 1 warm-up + 3 measured
                    start = time.perf_counter()
                    read_las_file(test_file)
                    timings.append(time.perf_counter() - start)
            best[n] = min(timings[1:])
        ratio = best[sizes[1]] / best[sizes[0]]
        assert ratio < 3.0, (
            f"wide wrapped line scaling ratio {ratio:.2f} for 2x tokens — "
            f"expected <3.0 (linear ~2.0, quadratic ~4.0)"
        )

    # --- EXT-01 (regression): wrapped COMMA/TAB >=3 curves ---

    def test_ext01_wrapped_comma_three_curves_parses_wrapped(self, tmp_path: Path) -> None:
        """EXT-01: a genuine WRAP=YES COMMA file with 3 curves (depth line
        alone, data lines carrying curve_count-1 = 2 values) must be
        detected as wrapped.

        Pre-fix, the D-01/D-02/D-03 protocol rewrite defined "full" for
        COMMA/TAB as ``n > 1``, so continuation lines carrying 2 values
        counted as full rows → full_count >= 2 → misdetected non-wrapped →
        DEPT received curve values, columns shifted, null padding.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.      :  Curve 1\n"
            " C2.      :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0,\n"
            "1.0,2.0,\n"
            "200.0,\n"
            "3.0,4.0,\n"
            "300.0,\n"
            "5.0,6.0,\n"
        )
        test_file = tmp_path / "ext01_comma_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Wrapped: 3 depth steps with DEPT holding depth values only.
        assert len(data["logs"]["DEPT"]) == 3, (
            f"Expected 3 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 200.0, 300.0])
        np.testing.assert_allclose(data["logs"]["C1"], [1.0, 3.0, 5.0])
        np.testing.assert_allclose(data["logs"]["C2"], [2.0, 4.0, 6.0])

    def test_ext01_wrapped_tab_four_curves_parses_wrapped(self, tmp_path: Path) -> None:
        """EXT-01: same misdetection with TAB delimiter and 4 curves
        (data lines carry curve_count-1 = 3 values)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.      :  Curve 1\n"
            " C2.      :  Curve 2\n"
            " C3.      :  Curve 3\n"
            "~A DEPT C1 C2 C3\n"
            "100.0\n"
            "1.0\t2.0\t3.0\n"
            "200.0\n"
            "4.0\t5.0\t6.0\n"
        )
        test_file = tmp_path / "ext01_tab_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2, (
            f"Expected 2 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 200.0])
        np.testing.assert_allclose(data["logs"]["C1"], [1.0, 4.0])
        np.testing.assert_allclose(data["logs"]["C2"], [2.0, 5.0])
        np.testing.assert_allclose(data["logs"]["C3"], [3.0, 6.0])

    def test_ext01_sparse_first_row_comma_still_non_wrapped(self, tmp_path: Path) -> None:
        """EXT-01 guard: the D-01 sparse-first-row COMMA file must STILL be
        detected non-wrapped after the curve_count-aware "full" predicate.

        The first row carries only the DEPT value (a sparse first row,
        not a wrapped continuation line), so the reader must not route the
        file to _read_wrapped — the full rows that follow (carrying exactly
        curve_count values) classify it non-wrapped."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.      :  Curve 1\n"
            " C2.      :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0,\n"
            "101.0,50.0,75.0\n"
            "102.0,51.0,76.0\n"
        )
        test_file = tmp_path / "ext01_sparse_first.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 3, (
            f"Expected 3 non-wrapped rows, got {len(data['logs']['DEPT'])}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])

    # --- L-03 (MEDIUM): {I} integer precision ---

    def test_l03_integer_curve_precision_las30(self, tmp_path: Path) -> None:
        """L-03: {I}-format curves must be parsed via int() and stored as
        int64, preserving values above 2^53 that float64 cannot represent.

        Pre-fix, float('9007199254740993') rounded to 9007199254740992.0.
        """
        content = (
            "~VERSION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL\n"
            " NULL.   -999 : NULL VALUE\n"
            "~CURVE\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 9007199254740994\n"
        )
        test_file = tmp_path / "l03_int.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["RUN_NO"].dtype == np.int64, (
            f"Expected int64 dtype, got {data['logs']['RUN_NO'].dtype}"
        )
        assert data["logs"]["RUN_NO"][0] == 9007199254740993, (
            f"Precision lost: {data['logs']['RUN_NO'][0]}"
        )
        assert data["logs"]["RUN_NO"][1] == 9007199254740994

    def test_l03_integer_curve_precision_las20(self, tmp_path: Path) -> None:
        """L-03: {I} precision also preserved on the LAS 1.2/2.0
        data_reader path (_read_normal)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 9007199254740994\n"
        )
        test_file = tmp_path / "l03_int20.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["RUN_NO"].dtype == np.int64
        assert data["logs"]["RUN_NO"][0] == 9007199254740993
        assert data["logs"]["RUN_NO"][1] == 9007199254740994

    def test_l03_fractional_null_uses_object_dtype(self, tmp_path: Path) -> None:
        """L-03/EXT-04 trap: when the declared NULL is non-integral
        (e.g. -999.25), int64 allocation would truncate it to -999 — the
        {I} branch stores an object array instead: data values stay exact
        Python ints, null cells keep the fractional sentinel."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 5\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "l03_frac_null.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # Fractional NULL → object dtype; the null cell keeps -999.25.
        assert data["logs"]["RUN_NO"].dtype == np.object_, (
            f"Expected object dtype, got {data['logs']['RUN_NO'].dtype}"
        )
        assert data["logs"]["RUN_NO"][0] == 5
        assert data["logs"]["RUN_NO"][1] == -999.25

    def test_l03_roundtrip_from_dict_preserves_int64(self) -> None:
        """L-03: from_dict must not coerce {I} log arrays back to float64
        (roundtrip precision preservation)."""
        from pylasdev.models import LASFile

        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO"},
                "well": {"NULL": "-999"},
                "curves": [
                    {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                    {"mnemonic": "RUN_NO", "data_format": "I"},
                ],
                "curves_order": ["DEPT", "RUN_NO"],
                "logs": {
                    "DEPT": np.array([100.0, 101.0]),
                    "RUN_NO": np.array([9007199254740993, 9007199254740994]),
                },
            }
        )
        assert las.logs["RUN_NO"].dtype == np.int64
        assert las.logs["RUN_NO"][0] == 9007199254740993

    def test_l03_roundtrip_from_dict_fractional_null_preserves_object(
        self,
    ) -> None:
        """L-03/EXT-04: from_dict must NOT coerce {I} arrays to int64 when
        the declared NULL is fractional (-999.25) — int64 would truncate
        the null sentinel.  The object dtype (exact ints + float sentinel)
        is preserved so to_dict → from_dict keeps >2^53 precision."""
        from pylasdev.models import LASFile

        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO"},
                "well": {"NULL": "-999.25"},
                "curves": [
                    {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                    {"mnemonic": "RUN_NO", "data_format": "I"},
                ],
                "curves_order": ["DEPT", "RUN_NO"],
                "logs": {
                    "DEPT": np.array([100.0, 101.0]),
                    "RUN_NO": np.array([5, -999.25]),
                },
            }
        )
        assert las.logs["RUN_NO"].dtype == np.object_, (
            f"Expected object dtype, got {las.logs['RUN_NO'].dtype}"
        )
        assert las.logs["RUN_NO"][0] == 5
        assert las.logs["RUN_NO"][1] == -999.25

    def test_ext04_integer_precision_fractional_null_las20(self, tmp_path: Path) -> None:
        """EXT-04: {I} curve value 9007199254740993 (2^53+1) survives
        exactly on the LAS 1.2/2.0 path with the default/fractional NULL
        -999.25 (pre-fix it rounded to 9007199254740992.0 float64)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_las20.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert int(data["logs"]["RUN_NO"][0]) == 9007199254740993, (
            f"Precision lost: {data['logs']['RUN_NO'][0]}"
        )
        assert data["logs"]["RUN_NO"][1] == -999.25, "null cell must keep the fractional sentinel"

    def test_ext04_integer_precision_fractional_null_las30(self, tmp_path: Path) -> None:
        """EXT-04: same exactness guarantee on the LAS 3.0 path with the
        default/fractional NULL -999.25."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_las30.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        run_no = las.data_sections[0].data["RUN_NO"]
        assert int(run_no[0]) == 9007199254740993, f"Precision lost: {run_no[0]}"
        assert run_no[1] == -999.25, "null cell must keep the fractional sentinel"

    def test_ext04_fractional_null_no_spurious_dtype_validate_issue(self, tmp_path: Path) -> None:
        """EXT-04 convergence: parsing a valid LAS 3.0 file with an {I}
        curve and a fractional declared NULL must NOT produce a spurious
        non-numeric-dtype validation issue.

        The object-dtype exemption must cover DataSection.validate (which
        LASFile.validate(complete=True) delegates to, and which the parser
        runs on every parse), not just the top-level logs loop — otherwise
        every valid {I} + fractional NULL file reports a false-positive
        "non-numeric dtype (object)" issue.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n"
            " STOP.M   101.0 : \n"
            " STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_validate.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        issues = las.validate(complete=True)
        dtype_issues = [i for i in issues if "non-numeric dtype" in i]
        assert dtype_issues == [], (
            "Spurious non-numeric dtype issue for a valid {I} + fractional "
            f"NULL file: {dtype_issues}"
        )

    # --- N-I-08 (MEDIUM): wrapped-mode trailing EOF under-fill ---

    def test_n08_trailing_eof_under_fill_warns(self, tmp_path: Path) -> None:
        """N-I-08: a wrapped file whose FINAL step is under-filled (the file
        ends with a partial step — leftover values not accounted for by the
        curve count) must emit the N-I-08 EOF warning instead of silently
        misaligning depth and curve columns.

        This fixture ends with 2 leftover values (8 total, 3 curves), so
        the N-I-08 trailing-partial warning fires.  The mid-file aligned
        under-fill case (total is a multiple of curve_count, depth column
        non-monotonic) is a DIFFERENT diagnostic — see
        test_m72_aligned_total_mid_file_advisory below."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "101.0\n"
            "51.0\n"
            "102.0\n"
            "52.0\n"
            "76.0\n"
        )
        test_file = tmp_path / "n08_underfill.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_las_file(test_file)
            messages = [str(x.message) for x in w]
        assert any("not accounted for by the curve count" in m for m in messages), (
            f"No N-I-08 trailing-partial warning emitted. Got: {messages[-3:]}"
        )

    def test_m72_aligned_total_mid_file_advisory(self, tmp_path: Path) -> None:
        """M-72: a wrapped file whose TOTAL is a multiple of curve_count
        (aligned total) but whose depth column is non-monotonic must emit
        the M-72 non-monotonic-depth advisory.

        N-I-08 only fires on a trailing partial buffer; it is blind to a
        non-monotonic depth column that leaves the total aligned.  In that
        case every step "completes" from the reader's perspective but a
        backward depth step produces a non-monotonic depth column — the
        ONLY diagnostic is the "depth column is not monotonic" advisory at
        data_reader.py:1660-1677.

        Fixture: 2 curves, 6 values (3 steps of 2) — total 6 is a multiple
        of 2, so N-I-08 does NOT fire; the depth column [1000, 1001, 999]
        is non-monotonic (backward step), so M-72 fires.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "51.0\n"
            "999.0\n"
            "52.0\n"
        )
        test_file = tmp_path / "m72_aligned.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]

        # The shifted depth column is observable: [1000, 1001, 999].
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 999.0])
        assert any("depth column is not monotonic" in m for m in messages), (
            f"M-72 advisory not emitted. Got: {messages[-3:]}"
        )
        # N-I-08 must NOT fire — the total is a multiple of curve_count.
        assert not any("not accounted for by the curve count" in m for m in messages), (
            f"N-I-08 fired on an aligned total — wrong diagnostic. Got: {messages[-3:]}"
        )

    # --- IT3-F-01 (MEDIUM): desanitize flag hoist ---

    def test_it3f01_desanitize_flag_hoist_semantics(self) -> None:
        """IT3-F-01: the hoisted desanitize flag must preserve the exact
        _# escape semantics (value-start and leading-whitespace unescape;
        M11: internal " _#" content is preserved, mirroring F-25)."""
        from pylasdev import parser as _parser_mod
        from pylasdev.data_reader import _desanitize_las_value

        prior = _parser_mod._DESANITIZE_ENABLED
        _parser_mod._DESANITIZE_ENABLED = True
        try:
            assert _desanitize_las_value("_#comment") == "#comment"
            assert _desanitize_las_value("_#comment", False) == "_#comment"
            # M11: the data_reader copy mirrors the parser's F-25 scope —
            # the writer escapes '#' ONLY at value start / after LEADING
            # whitespace, so an internal mid-value " _#" is preserved, while
            # a leading-whitespace writer escape is still restored.
            assert _desanitize_las_value("abc _#def", True) == "abc _#def"
            assert _desanitize_las_value("abc _#def", False) == "abc _#def"
            assert _desanitize_las_value(" _#comment", True) == " #comment"
            assert _desanitize_las_value("plain", True) == "plain"
            # None (no cache) falls back to the module flag
            assert _desanitize_las_value("_#x") == "#x"
        finally:
            _parser_mod._DESANITIZE_ENABLED = prior

    # --- IT3-F-02 (MEDIUM): math.isfinite swap ---

    def test_it3f02_to_finite_float_non_finite(self) -> None:
        """IT3-F-02: _to_finite_float must still null non-finite values
        after the np.isfinite → math.isfinite swap."""
        from pylasdev.data_reader import _to_finite_float

        for bad in ("nan", "inf", "-inf", "1e309", "1.0D309"):
            assert _to_finite_float(bad, -999.25) == -999.25, f"{bad} not nulled"
        assert _to_finite_float("50.5", -999.25) == 50.5
        assert _to_finite_float("", -999.25) == -999.25

    def test_it3f02_writer_math_isfinite(self) -> None:
        """IT3-F-02: writer _format_number must still route NaN/Inf to the
        null sentinel after the np.isnan/np.isinf → math.isfinite swap."""
        from pylasdev._writer_base import _format_number

        for bad in (float("nan"), float("inf"), float("-inf")):
            assert _format_number(bad, ".8g", -999.25) == "-999.25", (
                f"{bad} not routed to null sentinel"
            )
        assert _format_number(50.5, ".8g", -999.25) == "50.5"

    # --- IT3-F-03 (MEDIUM): wrapped pre-allocation ---

    def test_it3f03_wrapped_prealloc_correct(self, tmp_path: Path) -> None:
        """IT3-F-03: _read_wrapped pre-allocated numeric columns must
        produce identical results to the previous list-accumulation path."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "101.0\n"
            "51.0\n"
            "76.0\n"
        )
        test_file = tmp_path / "it3f03_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 76.0])

    def test_it3f03_wrapped_growth_capacity(self, tmp_path: Path) -> None:
        """IT3-F-03: pre-allocated wrapped columns must grow beyond the
        depth-step estimate (files denser than the ceil estimate)."""
        # 2 curves, 22 data lines: 10 single-value steps (20 lines) plus 2
        # two-value steps (2 lines).  The depth-step estimate
        # ceil(22/2)=11 undercounts the actual 12 steps — the two-value
        # lines make actual steps (12) exceed the line-count estimate (11),
        # forcing the pre-allocated column to grow (growth body fires 2x).
        lines: list[str] = []
        for step in range(10):
            lines.append(f"{100.0 + step}")
            lines.append(f"{10.0 + step}")
        # Two-value lines MUST go at the END: placing them first flips the
        # wrap detector to non-wrapped (the leading [2,2,2,2] window reads
        # as full rows) and misaligns the columns.  The leading single-value
        # window is what the test relies on for wrapped classification.
        lines.append("110.0 20.0")
        lines.append("111.0 21.0")
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n" + "\n".join(lines) + "\n"
        )
        test_file = tmp_path / "it3f03_growth.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0 + s for s in range(12)])
        np.testing.assert_allclose(data["logs"]["DT"], [10.0 + s for s in range(12)])

    def test_it3f03_wrapped_string_depth_curve(self, tmp_path: Path) -> None:
        """IT3-F-03: _read_wrapped must handle a wrapped file whose DEPTH
        curve (index 0) is a {S} string curve — the pre-allocated numeric
        columns path must not KeyError on a string depth line."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " HOLE.M   :  Hole {S}\n"
            " GR.GAPI  :  Gamma Ray {F}\n"
            "~A\n"
            "A\n50.0\nB\n51.0\n"
        )
        test_file = tmp_path / "it3f03_string_depth.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["string_data"]["HOLE"].tolist() == ["A", "B"]
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])


# ──────────────────────────────────────────────────────────────
# P-16 (reader-side, MEDIUM): unrecognized ~section header inside
# ~A must terminate data reading (matching the parser's
# section-boundary classification).
# ──────────────────────────────────────────────────────────────


class TestP16ReaderUnrecognizedSection:
    """P-16: `_iter_ascii_data_lines` must STOP reading data when it
    encounters an unrecognized ~section header (break, not
    skip-and-continue), matching the parser's section-boundary
    classification.  Previously the body of an unrecognized section
    (e.g. ~CUSTOMSECT) was consumed as data rows — the same physical
    lines landed in BOTH logs AND other_lines (garbage rows +
    duplicated text)."""

    def test_customsect_body_not_in_logs(self, tmp_path: Path) -> None:
        """~CUSTOMSECT after ~A: body lines must go to other_lines only."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0  : START DEPTH\n"
            " STOP.M   1660.0  : STOP DEPTH\n"
            " STEP.M   0.1     : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            " 10.0  50.0\n"
            " 11.0  51.0\n"
            "~CUSTOMSECT\n"
            " 30.0  70.0\n"
            " 40.0  80.0\n"
        )
        test_file = tmp_path / "p16_customsect.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # Only the ~A rows may land in logs.
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])
        # The ~CUSTOMSECT body goes to other_lines (parser-side routing).
        assert "~CUSTOMSECT" in data["other"]
        assert "30.0  70.0" in data["other"]
        assert "40.0  80.0" in data["other"]

    def test_unrecognized_section_with_trailing_data(self, tmp_path: Path) -> None:
        """Data after the unrecognized section header is not consumed as
        ~A rows either (the section boundary ends the ASCII block)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            " 10.0  50.0\n"
            "~Units\n"
            " 99.0  99.0\n"
        )
        test_file = tmp_path / "p16_units.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0])
        assert "~Units" in data["other"]

    def test_tilde_garbage_line_not_data_row(self, tmp_path: Path) -> None:
        """~-prefixed lines that are NOT section headers (e.g. ~., ~#)
        must not be consumed as data rows — the parser routes them to
        other_lines (N-2 adjacent defect, same divergence class)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A\n"
            " 10.0\n"
            "~.\n"
            " 11.0\n"
        )
        test_file = tmp_path / "p16_garbage.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        assert "~." in data["other"]


# ──────────────────────────────────────────────────────────────
# P-05 (reader-side, MEDIUM): LAS 1.2/2.0 no-space pipe ASCII
# header (~ASCII|CURVE) must be recognized by the reader.
# ──────────────────────────────────────────────────────────────


class TestP05ReaderNoSpacePipeAscii:
    """P-05: `_is_ascii_section` (and data_reader's section-word
    detection) must strip the `| <target>` pipe suffix so a LAS 1.2/2.0
    `~ASCII|CURVE` header is recognized as a data section.  Previously
    the parser recognized it but the reader silently read ZERO rows."""

    def test_ascii_pipe_curve_parses_data(self, tmp_path: Path) -> None:
        """LAS 1.2 `~ASCII|CURVE` header produces non-empty data rows."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~ASCII|CURVE\n"
            " 10.0  50.0\n"
            " 11.0  51.0\n"
        )
        test_file = tmp_path / "p05_ascii_pipe.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2, (
            f"~ASCII|CURVE header must produce data rows — got {len(data['logs']['DEPT'])}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

    def test_ascii_pipe_curve_las20(self, tmp_path: Path) -> None:
        """LAS 2.0 `~ASCII|CURVE` header also produces data rows."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~ASCII|CURVE\n"
            " 100.0\n"
            " 101.0\n"
        )
        test_file = tmp_path / "p05_ascii_pipe_20.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])


# ──────────────────────────────────────────────────────────────
# M-11 (MEDIUM): mandatory_well_fields aligned with lascheck —
# UWI removed (optional), COMP/FLD/DATE added (required).
# ──────────────────────────────────────────────────────────────


class TestM11MandatoryWellFields:
    """M-11: `_LASVersionSpec.mandatory_well_fields` must match lascheck's
    10-field ~W set for LAS 2.0 (STRT, STOP, STEP, NULL, COMP, WELL,
    FLD, LOC, SRVC, DATE).  UWI is optional — a common LAS file with
    no UWI must NOT warn.  LAS 1.2 is narrower (I2-07): COMP/FLD/DATE
    are 2.0-era requirements, NOT mandatory for 1.2, so a 1.2 file
    carrying only the four numeric fields must NOT warn either."""

    def test_las12_no_uwi_no_warning(self) -> None:
        """LAS 1.2 without UWI but with COMP/FLD/DATE produces NO
        'missing UWI' warning (M-11 false-positive fix)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.5 : START DEPTH\n"
            " STOP.M   500.0  : STOP DEPTH\n"
            " STEP.M   -0.125 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " COMP.    ACME : COMPANY\n"
            " WELL.    W-1 : WELL NAME\n"
            " FLD.     NORTH : FIELD\n"
            " LOC.     12-34 : LOCATION\n"
            " SRVC.    LOGCO : SERVICE COMPANY\n"
            " DATE.    15/01/2001 : LOG DATE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            "~A\n"
            " 100.0\n"
        )
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [x for x in w if "missing mandatory well field" in str(x.message)]
            assert len(mandatory_warnings) == 0, (
                f"Expected no mandatory well field warnings, got: "
                f"{[str(x.message) for x in mandatory_warnings]}"
            )
            uwi_warnings = [
                x
                for x in w
                if "UWI" in str(x.message) and "missing mandatory well field" in str(x.message)
            ]
            assert len(uwi_warnings) == 0, (
                "UWI is optional per lascheck — must not warn when absent"
            )

    def test_las12_missing_comp_fld_date_does_not_warn(self) -> None:
        """LAS 1.2 missing COMP/FLD/DATE does NOT warn (I2-07).

        I2-07 corrected the LAS 1.2 mandatory-field set: lascheck's
        10-field set is a LAS 2.0-era requirement (lascheck documents
        "supports checking against LAS 2.0 standard only"), and
        frackoptima/GERDA require only the four numeric fields
        (STRT, STOP, STEP, NULL) for LAS 1.2.  COMP/FLD/DATE are NOT
        mandatory for 1.2, so a 1.2 file carrying the 4 numeric fields
        (plus optional extras) must produce zero mandatory-field
        warnings — even when UWI is present.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.5 : START DEPTH\n"
            " STOP.M   500.0  : STOP DEPTH\n"
            " STEP.M   -0.125 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " WELL.    W-1 : WELL NAME\n"
            " LOC.     12-34 : LOCATION\n"
            " SRVC.    LOGCO : SERVICE COMPANY\n"
            " UWI.     10006170502W500 : UNIQUE WELL ID\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            "~A\n"
            " 100.0\n"
        )
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [
                str(x.message) for x in w if "missing mandatory well field" in str(x.message)
            ]
            # I2-07: COMP/FLD/DATE are 2.0-era mandatory fields, NOT 1.2.
            assert len(mandatory_warnings) == 0, (
                f"LAS 1.2 with the 4 numeric fields must not warn about "
                f"COMP/FLD/DATE (I2-07); got: {mandatory_warnings}"
            )
            assert not any("UWI" in t for t in mandatory_warnings), (
                f"UWI must not be reported as missing: {mandatory_warnings}"
            )


# ──────────────────────────────────────────────────────────────
# ENC-03 (reader, MEDIUM): max_file_size ValueError conversion +
# LASEncodingError mislabeling.  README.md:346-360 documented ValueError
# (catch separately) but code raises LASReadError — the README was the
# sole outlier (code+docstrings+tests agree on LASReadError) and has been
# corrected.  A genuine LASEncodingError must also NOT be swallowed under
# a misleading "size exceeded" message: it propagates as LASEncodingError.
# ──────────────────────────────────────────────────────────────


class TestENC03SizeLimitErrorContract:
    """ENC-03: size-limit failures raise LASReadError (documented
    contract); genuine encoding failures raise LASEncodingError with an
    accurate message — never a misleading 'size exceeded' LASReadError."""

    def test_max_file_size_raises_las_read_error(self, tmp_path: Path) -> None:
        """read_las_file(max_file_size=1) raises LASReadError (asserting
        the actual behavior — the README now documents LASReadError, not
        ValueError)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "enc03_size.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASReadError, match="Cannot read file"):
            read_las_file(test_file, max_file_size=1)

    def test_genuine_decode_failure_raises_las_encoding_error(self, tmp_path: Path) -> None:
        """A genuine decoding failure (explicit encoding that cannot decode
        the bytes) raises LASEncodingError with an accurate message — NOT a
        misleading 'size exceeded or invalid parameter' LASReadError."""
        from pylasdev.exceptions import LASEncodingError

        test_file = tmp_path / "enc03_bad_encoding.las"
        # CP1252 bytes are invalid UTF-8, so forcing utf-8 must fail to decode.
        test_file.write_bytes("Caf\u00e9 r\u00e9sum\u00e9".encode("cp1252"))
        with pytest.raises(LASEncodingError, match="Failed to decode"):
            read_las_file(test_file, encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# F-02 (data_reader, MEDIUM, REGRESSION): the F-07 depth-line
# evidence rule's ``window[1] == 1`` arm fired for ANY single
# 1-value row after a full first row when curve_count >= 3,
# misclassifying a ragged NON-wrapped nc>=3 file with a single
# 1-value middle row ([3,1,3] / [3,1,2]) as WRAPPED → silent
# column-shift corruption (the genuine depth value swallowed into
# curve 1, trailing values discarded).  A single 1-value row is
# ragged-row evidence (graceful short-row null-fill), not
# unambiguous depth evidence — only TWO+ 1-value rows trigger the
# wrapped arm.  Mirrored on the LAS 3.0 path (_las30_data.py) and
# covered there by TestF02Las30RaggedMiddleRowNotWrapped.
# ──────────────────────────────────────────────────────────────


class TestF02ThreeCurveShortMiddleRowNotWrapped:
    """F-02: 3-curve WRAP=NO files with a single 1-value middle row must
    NOT be classified wrapped (regression — pre-fix the nc>=3
    window[1]==1 arm silently column-shifted the file)."""

    def test_las20_three_curve_short_middle_row_not_wrapped(self, tmp_path: Path) -> None:
        """3-curve LAS 2.0 WRAP=NO `100.0 50.0 30.0 / 101.0 / 102.0 60.0 40.0`
        (window [3,1,3]): DEPT=[100,101,102], C1=[50,-999.25,60],
        C2=[30,-999.25,40].  Pre-fix: WRAPPED=True → DEPT=[100,101],
        C1=[50,102], C2=[30,60] — the depth value 102.0 swallowed into
        C1 and the genuine 40.0 discarded."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0 40.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_short_middle.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, -999.25, 40.0])

    def test_las20_three_curve_short_middle_two_value_row_not_wrapped(self, tmp_path: Path) -> None:
        """[3,1,2] shape: `100.0 50.0 30.0 / 101.0 / 102.0 60.0` — the
        single 1-value middle row must stay ragged (non-wrapped):
        DEPT=[100,101,102], C1=[50,-999.25,60], C2=[30,-999.25,-999.25].
        Pre-fix this silently produced DEPT=[100,101], C1=[50,102],
        C2=[30,60] with no warning at all."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_short_middle2.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, -999.25, -999.25])

    def test_las20_three_curve_wrap_yes_short_middle_row_still_wrapped(
        self, tmp_path: Path
    ) -> None:
        """Control: the [3,1,3] shape with WRAP=YES declared must STILL
        be classified wrapped (declared-YES + depth evidence → wrapped;
        the gate must not regress genuine wrapped detection)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0 40.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_wrap_yes.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Wrapped parse: DEPT holds the depth-line values only.
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, 60.0])

    def test_detector_single_one_value_row_never_wrapped(self) -> None:
        """Detector-level audit: a SINGLE 1-value row (in any window
        position) after a full first row is ragged — never wrapped.
        Two+ 1-value rows (masquerade/mixed-wrap) stay wrapped."""
        from pylasdev.data_reader import _detect_actual_wrap

        def lines_from_window(window: list[int]) -> list[str]:
            out = ["~A DEPT GR RHOB"]
            for n in window:
                out.append(" ".join(str(i) for i in range(1, n + 1)))
            return out

        # (window, curve_count, declared_wrap, expected_wrapped)
        shapes = [
            ([3, 1, 3], 3, "NO", False),  # F-02: ragged single middle 1-value row
            ([3, 1, 2], 3, "NO", False),  # F-02: ragged single middle 1-value row
            ([3, 1, 3], 3, None, False),  # absent declaration behaves like NO
            ([3, 1, 2], 3, None, False),  # absent declaration behaves like NO
            ([3, 1, 1], 3, "NO", True),  # mnemonic-header masquerade (2 one-value rows)
            ([3, 1, 2, 1], 3, "NO", True),  # I2-03 mixed-wrap (2 one-value rows)
            ([3, 1, 2, 1], 3, None, True),  # mixed-wrap without declaration
            ([3, 1, 3], 3, "YES", True),  # declared YES stays wrapped
            ([3, 2, 1], 3, "NO", False),  # ragged trailing 1-value row (unchanged)
        ]
        for window, nc, decl, expected in shapes:
            got = _detect_actual_wrap(lines_from_window(window), nc, " ", declared_wrap=decl)
            assert got is expected, (
                f"window={window} nc={nc} decl={decl}: expected wrapped={expected}, got {got}"
            )


class TestIsMnemonicHeaderRowSingleCurve:
    """PSR-1 (Stage 11): _is_mnemonic_header_row's token-count gate must
    use min(2, curve_count), not a flat 2.  A SINGLE-curve section must
    still recognize its 1-token standalone mnemonic header row
    ("~A\\nDEPT\\n1670.0\\n...") — the DR-M2 `< 2` gate consumed it as
    data, producing a phantom all-null first row + value shift.  The
    2-token minimum is preserved for multi-curve sections (M-02), and the
    all-string exclusion still protects single-curve STRING sections
    (M-03/F-19)."""

    @staticmethod
    def _las(mnemonics: list[str]) -> LASFile:
        las = LASFile()
        las.curves_order = list(mnemonics)
        for name in mnemonics:
            las.curves.append(CurveDefinition(mnemonic=name, unit="M"))
        return las

    def test_single_curve_one_token_mnemonic_is_header(self) -> None:
        """PSR-1: a 1-token row equal to the sole curve's mnemonic is a
        header in a single-curve section (pre-fix `< 2` returned False —
        the phantom-row defect)."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["DEPT"], las, 1, set()) is True

    def test_single_curve_one_token_non_mnemonic_is_data(self) -> None:
        """A 1-token numeric data row is not a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["1670.0"], las, 1, set()) is False

    def test_single_curve_two_tokens_not_header(self) -> None:
        """A 2-token row in a 1-curve section exceeds curve_count — it is
        an extra-column data row, never a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["DEPT", "GR"], las, 1, set()) is False

    def test_two_curve_one_token_not_header(self) -> None:
        """M-02: the 2-token minimum stays intact for multi-curve
        sections — a 1-token row cannot be a full header signature."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR"])
        assert _is_mnemonic_header_row(["DEPT"], las, 2, set()) is False

    def test_single_curve_string_section_one_token_not_header(self) -> None:
        """M-03/F-19: in a single-curve all-STRING section the all-string
        exclusion fires — a mnemonic-coincident value is data, not a
        header (min(2, 1) alone must NOT turn string sections into header
        droppers)."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["LITH"])
        assert _is_mnemonic_header_row(["LITH"], las, 1, {0}) is False

    def test_multi_curve_partial_header_still_header(self) -> None:
        """M12/DR-M2: the partial-header relaxation (2..curve_count) stays
        intact — "DEPT GR" with 3 declared curves is still a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR", "RHOB"])
        assert _is_mnemonic_header_row(["DEPT", "GR"], las, 3, set()) is True

    def test_multi_curve_full_header_still_header(self) -> None:
        """The full-width header remains a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR", "RHOB"])
        assert _is_mnemonic_header_row(["DEPT", "GR", "RHOB"], las, 3, set()) is True


# ──────────────────────────────────────────────────────────────
# W-A / W-2 / H-1 / F-02 regression tests (IMPLEMENT read-path pass)
# Each test FAILS on the pre-refactor library and PASSES on the new code.
# ──────────────────────────────────────────────────────────────


def _write_las(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _flowing_content(version: str, wrap: str, dlm: str = "SPACE") -> str:
    """Flowing [6,6] nc=3 content: 2 complete depth steps per line."""
    dlm_line = f" DLM.    {dlm} :\n" if dlm != "SPACE" else ""
    if dlm == "COMMA":
        rows = "1000.0,10.0,20.0,1001.0,11.0,21.0\n1002.0,12.0,22.0,1003.0,13.0,23.0\n"
    elif dlm == "TAB":
        rows = "1000.0\t10.0\t20.0\t1001.0\t11.0\t21.0\n1002.0\t12.0\t22.0\t1003.0\t13.0\t23.0\n"
    else:
        rows = "1000.0 10.0 20.0 1001.0 11.0 21.0\n1002.0 12.0 22.0 1003.0 13.0 23.0\n"
    return (
        "~VERSION INFORMATION\n"
        f" VERS.   {version}  : CWLS LOG ASCII STANDARD\n"
        f" WRAP.   {wrap}  :\n"
        f"{dlm_line}"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        " GR.GAPI  :  Gamma\n"
        " RHOB.K/M3:  Density\n"
        "~A\n"
        f"{rows}"
    )


class TestWrapFlowingAccumulation:
    """W-A (HIGH): the flowing layout (depth NOT on its own line — 2+
    complete depth steps per line) must parse with ALL steps preserved.

    Pre-fix the reader had NO working path for flowing data: the detector
    misrouted all-full windows to _read_normal (2 of 4 steps silently
    lost) and _read_wrapped's depth-line protocol corrupted it worse (3
    of 4 lost).  The n_curves-accumulation rewrite + the multiple-of
    detection rule fix both."""

    def test_flowing_wrap_yes_las12(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_yes.las", _flowing_content("1.2", "YES"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, 21.0, 22.0, 23.0])

    def test_flowing_wrap_yes_las20(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_yes20.las", _flowing_content("2.0", "YES"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, 21.0, 22.0, 23.0])

    def test_flowing_wrap_no_declaration_independent(self, tmp_path: Path) -> None:
        """W-A's trigger is declaration-INDEPENDENT: a WRAP=NO header must
        not hide the flowing signature (the multiple-of rule is content-
        based, placed before the declared-header fall-through)."""
        test_file = _write_las(tmp_path, "flow_no.las", _flowing_content("2.0", "NO"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_with_string_curve(self, tmp_path: Path) -> None:
        """Flowing with a {S} string curve: step-position dispatch must
        preserve string values in order (pre-fix S12: 2 of 4 lost)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "1000.0 SAND 10.0 1001.0 SHALE 11.0\n"
            "1002.0 LIME 12.0 1003.0 DOLO 13.0\n"
        )
        test_file = _write_las(tmp_path, "flow_str.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"],
            np.array(["SAND", "SHALE", "LIME", "DOLO"], dtype=object),
        )
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_comma_delimiter(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_comma.las", _flowing_content("2.0", "YES", "COMMA"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_tab_delimiter(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_tab.las", _flowing_content("1.2", "YES", "TAB"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_trailing_partial_step_warns(self, tmp_path: Path) -> None:
        """A flowing section ending with an incomplete step (2 trailing
        values, nc=3) emits the N-I-08-style "not accounted for" warning
        and the orphan values are discarded (R-6 contract)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3:  Density\n"
            "~A\n"
            "1000.0 10.0 20.0 1001.0 11.0 21.0\n"
            "1002.0 12.0\n"
        )
        test_file = _write_las(tmp_path, "flow_partial.las", content)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
        assert any("under-filled" in m or "not accounted for" in m for m in messages), (
            f"No trailing-step warning emitted. Got: {messages[-3:]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, 21.0])


class TestWrapDetectorFlowingRule:
    """Detector-unit locks for the R-1 multiple-of flowing rule and the
    R-3/II-5 discriminator (all FAIL pre-fix, PASS post-fix)."""

    def _detect(self, window: list[int], nc: int, decl: str | None) -> bool:
        from pylasdev.data_reader import _detect_actual_wrap

        lines = ["~A DEPT GR"]
        for n in window:
            lines.append(" ".join(str(i) for i in range(1, n + 1)))
        return _detect_actual_wrap(lines, nc, " ", declared_wrap=decl)

    def test_flowing_all_full_detected_wrapped(self) -> None:
        """[6,6] nc=3 → True (W-A): a full first line carrying 2 complete
        depth steps is flowing, regardless of the declared header."""
        assert self._detect([6, 6], 3, "YES") is True
        assert self._detect([6, 6], 3, "NO") is True

    def test_mislabeled_all_full_stays_unwrapped(self) -> None:
        """[3,3,3,3] nc=3 YES → False (test_regression.py:4708 lock): one
        complete step per line is NOT flowing."""
        assert self._detect([3, 3, 3, 3], 3, "YES") is False

    def test_extra_columns_aligned_stays_unwrapped(self) -> None:
        """[4,4] nc=3 → False (4 % 3 != 0): the extra-columns control —
        a non-aligned all-full window stays non-wrapped so _read_normal's
        extra-column discard is correct (QA N2 hazard)."""
        assert self._detect([4, 4], 3, "YES") is False

    def test_nc2_aligned_extra_columns_stays_unwrapped(self) -> None:
        """[4,4] nc=2 → False: for nc=2 the multiple-of rule is disabled
        (curve_count >= 3 guard) so test-locked extra-columns files are
        never misrouted to the wrapped reader."""
        assert self._detect([4, 4], 2, "NO") is False
        assert self._detect([4, 4], 2, "YES") is False

    def test_ragged_two_one_value_rows_stays_unwrapped(self) -> None:
        """W-2/II-5: [3,1,3,1] and [3,1,1,3] WRAP=NO → False — a full row
        immediately after a 1-value row is impossible in genuine wrapped
        data, so the ≥2-one-value-rows arm must not fire."""
        assert self._detect([3, 1, 3, 1], 3, "NO") is False
        assert self._detect([3, 1, 1, 3], 3, "NO") is False


class TestWrapRaggedNullFill:
    """W-2 (MEDIUM): ragged WRAP=NO files with ≥2 one-value rows must
    null-fill (not corrupt via the wrapped reader)."""

    _base = (
        "~VERSION INFORMATION\n"
        " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        " GR.GAPI  :  Gamma\n"
        " RHOB.K/M3:  Density\n"
        "~A\n"
    )

    def test_ragged_3131_wrap_no_null_fill(self, tmp_path: Path) -> None:
        content = self._base + ("1000.0 10.0 20.0\n1001.0\n1002.0 12.0 22.0\n1003.0\n")
        test_file = _write_las(tmp_path, "r3131.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, -999.25, 12.0, -999.25])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, -999.25, 22.0, -999.25])

    def test_ragged_3113_wrap_no_null_fill(self, tmp_path: Path) -> None:
        content = self._base + ("1000.0 10.0 20.0\n1001.0\n1002.0\n1003.0 13.0 23.0\n")
        test_file = _write_las(tmp_path, "r3113.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, -999.25, -999.25, 13.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, -999.25, -999.25, 23.0])


class TestWrapLas30FlowingRejection:
    """R-5: the LAS 3.0 fix is DETECTION-ONLY — the shared gate's True
    feeds the existing loud WRAP=YES rejection (no accumulation in
    process_ascii_data); ragged WRAP=NO files parse instead of being
    falsely rejected (NEW-1)."""

    def test_las30_flowing_six_six_raises_loudly(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " GR.GAPI  :  Gamma {F}\n"
            " RHOB.K/M3:  Density {F}\n"
            "~A\n"
            "1000.0 10.0 20.0 1001.0 11.0 21.0\n"
            "1002.0 12.0 22.0 1003.0 13.0 23.0\n"
        )
        test_file = _write_las(tmp_path, "l30_flow.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(LASParseError, match="WRAP=YES is not supported"):
                read_las_file(test_file)

    def test_las30_ragged_3131_wrap_no_parses(self, tmp_path: Path) -> None:
        """NEW-1: a ragged WRAP=NO file must NOT be falsely rejected as
        "LAS 3.0 WRAP=YES" — the II-5 discriminator keeps it non-wrapped
        and it parses with null-fill."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " GR.GAPI  :  Gamma {F}\n"
            " RHOB.K/M3:  Density {F}\n"
            "~A\n"
            "1000.0 10.0 20.0\n"
            "1001.0\n"
            "1002.0 12.0 22.0\n"
            "1003.0\n"
        )
        test_file = _write_las(tmp_path, "l30_ragged.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, -999.25, 12.0, -999.25])


class TestHeaderSkipSupersetSplit:
    """H-1 (MEDIUM): the DLM-aware reader must recognize a space-separated
    mnemonic header row in a DLM=COMMA file (superset split at all 3
    predicate sites), and the II-11 collision class (a genuine first-row
    string value containing a space in a mixed COMMA section) must NOT be
    skipped as a header."""

    def test_space_separated_header_in_comma_file_skipped(self, tmp_path: Path) -> None:
        """Pre-fix: the reader split the header row with the DLM-aware
        tokenizer → one token "DEPT GR" → consumed as data → phantom
        all-null first row + one-row shift."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "100.0,50.0\n"
            "101.0,51.0\n"
        )
        test_file = _write_las(tmp_path, "h1_comma_header.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

    def test_space_in_string_first_row_not_header(self, tmp_path: Path) -> None:
        """II-11 collision class: a genuine first-row {S} string value
        containing a space ("LITH SHALE") in a COMMA-DLM mixed section is
        split by the superset tokenizer into 4 tokens (> curve_count), so
        the count bound keeps it DATA — never skipped as a header."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " LITH.    :  Lithology {S}\n"
            "~A DEPT GR LITH\n"
            "100.0,10.0,LITH SHALE\n"
            "101.0,11.0,SAND STONE\n"
        )
        test_file = _write_las(tmp_path, "h1_collision.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"],
            np.array(["LITH SHALE", "SAND STONE"], dtype=object),
        )


class TestHeaderTildeRoundtrip:
    """F-02/II-26: header fields must NOT restore the writer-only '_~'
    data escape — a genuine '_~'-prefixed header value roundtrips
    unchanged (parser header call sites use restore_tilde=False, II-13)."""

    def test_well_value_underscore_tilde_roundtrips_unchanged(self, tmp_path: Path) -> None:
        from pylasdev.writer import write_las_file

        las = LASFile()
        las.version.wrap = "NO"
        las.well["NULL"] = "-999.25"
        las.well["WELL"] = "_~Acme"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        out = tmp_path / "tilde_well.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        # Pre-fix: parser._desanitize_las_value restored '_~' → '~' at the
        # header site, corrupting the genuine value to '~Acme'.
        assert back.well["WELL"] == "_~Acme", (
            f"F-02: '_~' header escape wrongly restored, got {back.well['WELL']!r}"
        )


class TestE20WrappedPendingBufferBounded:
    """E-20 (CONFIRMED MEDIUM): _read_wrapped's pending buffer retained
    every consumed token string until EOF (~58 GB on crafted WRAP=YES
    files before the MAX_TOTAL_ELEMENTS guard fires).  The fix trims the
    consumed prefix once read_idx crosses _PENDING_TRIM_THRESHOLD
    (amortized O(1) per token) while keeping the O(1)-per-step
    extraction and the N-I-08 trailing-step diagnostic."""

    @staticmethod
    def _wrapped_content(n_steps: int) -> str:
        rows = []
        for i in range(n_steps):
            rows.append(f"{1000.0 + i:.1f}")
            rows.append(f"{50.0 + 5 * i:.1f} {1.0 + i:.1f}")
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A\n"
            + "\n".join(rows)
            + "\n"
        )

    def test_trim_preserves_exact_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrapped file crossing the trim threshold many times must parse
        to exactly the same values as an untrimmed read (the trim must not
        shift, drop, or duplicate buffered values)."""
        from pylasdev import data_reader as dr

        # Force a trim every few tokens so the test exercises many trims.
        monkeypatch.setattr(dr, "_PENDING_TRIM_THRESHOLD", 8)
        n_steps = 40
        test_file = _write_las(tmp_path, "e20_trim.las", self._wrapped_content(n_steps))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(
            las.logs["DEPT"], [1000.0 + i for i in range(n_steps)]
        )
        np.testing.assert_allclose(
            las.logs["GR"], [50.0 + 5 * i for i in range(n_steps)]
        )
        np.testing.assert_allclose(
            las.logs["RHOB"], [1.0 + i for i in range(n_steps)]
        )

    def test_pending_retention_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """E-20/M-10: consumed tokens must not stay alive until EOF.  With
        the pre-fix code every token string from _split_data_line stays
        referenced by `pending` (a 12K-token file retains ~600 KB of token
        objects at peak); post-fix the buffer holds only the unconsumed
        tail.  Retention is measured DURING parse via the
        _read_wrapped_trace_hook seam (while `pending` is still live) — a
        snapshot AFTER _read_wrapped returns sees the freed list and
        cannot distinguish trimmed from untrimmed behavior (M-10: the old
        post-parse snapshot passed both ways, so removing the E-20 trim
        went undetected by CI)."""
        import tracemalloc

        from pylasdev import data_reader as dr

        monkeypatch.setattr(dr, "_PENDING_TRIM_THRESHOLD", 64)
        n_steps = 4000  # 12_000 tokens — pre-fix retains every one of them
        test_file = _write_las(tmp_path, "e20_retention.las", self._wrapped_content(n_steps))

        snapshots: list[Any] = []
        calls = 0

        def _trace_hook(_pending: list[str]) -> None:
            nonlocal calls
            calls += 1
            # Sample every 128th line (~62 snapshots for 8_000 lines) —
            # tracemalloc.take_snapshot is too expensive for every line.
            if calls % 128 == 1:
                snapshots.append(tracemalloc.take_snapshot())

        monkeypatch.setattr(dr, "_read_wrapped_trace_hook", _trace_hook)

        tracemalloc.start()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                las = read_las_file_as_object(test_file)
        finally:
            tracemalloc.stop()

        # Sanity: the file parsed completely.
        np.testing.assert_allclose(las.logs["DEPT"][-1], 1000.0 + n_steps - 1)

        def _retained(snapshot: Any) -> int:
            retained = 0
            for stat in snapshot.statistics("lineno"):
                fname = str(stat.traceback[0].filename)
                if "_data_section_reader" in fname or "data_reader" in fname:
                    retained += stat.size
            return retained

        assert snapshots, "M-10: trace hook never fired — the seam is not being invoked"
        peak = max(_retained(s) for s in snapshots)
        # Pre-fix (no trim): at peak `pending` holds all ~12_000 tokens
        # (~600 KB).  Post-fix: the buffer holds only the unconsumed tail
        # (< threshold + one line) at every sampled point of the parse.
        assert peak < 200_000, (
            f"E-20: {peak} bytes of token objects still retained during "
            f"parse — the pending buffer is not being trimmed"
        )


class TestE41WrapDetectorHeaderRowNotCounted:
    """E-41 (CONFIRMED MEDIUM): _detect_actual_wrap's 4-line window counts
    the standalone mnemonic header row as a full data line, flipping
    genuinely wrapped >=3-curve data to non-wrapped → silent column shift
    of the whole depth log.  The fix skips mnemonic-header rows while the
    window is empty.  MUST NOT regress test_regression.py:2377
    ([3,1,2,1] wrapped lock)."""

    @staticmethod
    def _content(with_header: bool, wrap: str = "NO") -> str:
        header_row = "DEPT  GR  RHOB\n" if with_header else ""
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            f" WRAP.   {wrap}  : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A\n"
            + header_row
            # Data: continuation lines of 2 values + depth lines of 1 value.
            # Window WITHOUT the header row: [2,2,1,2] → partial majority →
            # wrapped.  WITH the header row counted: [3,2,2,1] → first-line
            # full + WRAP=NO → non-wrapped (pre-fix silent shift).
            + "50.0 1.0\n"
            + "1000.0\n"
            + "55.0 2.0\n"
            + "1001.0\n"
            + "60.0 3.0\n"
            + "1002.0\n"
        )

    def test_wrapped_with_header_parses_identically_to_without(
        self, tmp_path: Path
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with_header = read_las_file_as_object(
                _write_las(tmp_path, "e41_with_header.las", self._content(True))
            )
            without_header = read_las_file_as_object(
                _write_las(tmp_path, "e41_without_header.las", self._content(False))
            )
        # Pre-fix: the header row (full width) + WRAP=NO → non-wrapped →
        # DEPT=[50,1000,55,1001,60,1002] (silent shift).  Post-fix both
        # variants are wrapped and identical.
        np.testing.assert_allclose(with_header.logs["DEPT"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(with_header.logs["GR"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(with_header.logs["RHOB"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(without_header.logs["DEPT"], with_header.logs["DEPT"])
        np.testing.assert_allclose(without_header.logs["GR"], with_header.logs["GR"])
        np.testing.assert_allclose(without_header.logs["RHOB"], with_header.logs["RHOB"])

    def test_comma_dlm_space_header_not_flipped_to_wrapped(self, tmp_path: Path) -> None:
        """E-41 iter-3 NEW TRIGGER VARIANT: DLM=COMMA + space-separated
        header row → the comma split sees ONE token ("DEPT GR RHOB NPHI")
        → the window starts [1,3,3,3] which is NOT uniform (the header's
        1-token entry breaks the H-02 shape) → partial majority → WRAPPED
        verdict (pre-fix) → silent column shift of every row.  Post-fix
        the header row is not counted → window [3,3,3] → H-02 uniform-short
        → non-wrapped → graceful short-row null-fill."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A\n"
            "DEPT  GR  RHOB  NPHI\n"
            "1000.0,50.0,1.0\n"
            "1001.0,55.0,2.0\n"
            "1002.0,60.0,3.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "e41_comma.las", content))
        # Pre-fix: wrapped verdict → 4-value steps from 3-token rows →
        # DEPT=[1000,50,1] (silent shift).  Post-fix: non-wrapped →
        # short-row null-fill preserves every genuine value.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(las.logs["RHOB"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(las.logs["NPHI"], [-999.25, -999.25, -999.25])


class TestE42WrappedHeaderGateClosesAfterFirstDataLine:
    """E-42 (CONFIRMED MEDIUM): _read_wrapped's header-skip gate was keyed
    to step completion (total_elements == 0), staying open across multiple
    lines in depth-first wrapped layouts — a string continuation value
    coinciding with a mnemonic on line 2+ was silently dropped as a
    "header" (data loss + column shift).  The gate is now keyed to the
    line position (current_line == 0), like _read_normal."""

    def test_string_continuation_on_line_two_not_dropped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT LITH GR\n"
            "1000.0\n"
            "LITH GR\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "e42_gate.las", content))
        # The second data line "LITH GR" is a genuine wrapped continuation
        # (LITH string value + GR value) that coincides with the curve
        # mnemonics.  Pre-fix: the gate (total_elements == 0) is still open
        # → the row is skipped as a "header" → the step never completes →
        # DEPT=[] (all values lost).  Post-fix: gate closed after the first
        # data line → the continuation is consumed → one complete step.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"], np.array(["LITH"], dtype=object)
        )


class TestM13UnitsRowAfterMnemonicHeaderSkipped:
    """M-13 (CONFIRMED MEDIUM): a units row emitted directly after the
    standalone mnemonic header row inside ~A ("~A\\nDEPT GR\\nM GAPI\\n...")
    was consumed as a DATA row → phantom all-null first row + one-row shift
    of the whole depth log.  The fix skips an optional units row (all
    letters-only tokens, shared is_units_header_row predicate) on the
    first data line only, gated on a mnemonic header row having been
    skipped."""

    def test_units_row_skipped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "1000.0 50.0\n"
            "1001.0 51.0\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "m13_units.las", content))
        # Pre-fix: "M GAPI" consumed as the first data row →
        # DEPT=[-999.25, 1000.0, 1001.0] (phantom all-null first row).
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 51.0])
        # M-02 pin: the parser pre-scan subtracts the units row itself via
        # the shared is_units_header_row predicate, so NO spurious
        # "Pre-scan overcount" warning may fire on an M-13 file (pre-fix
        # the pre-scan counted the units row → declared 3 data lines for 2
        # actual → spurious warning; the old simplefilter("ignore") mask
        # hid it from the test).
        pre_scan_msgs = [
            str(w.message) for w in caught if "Pre-scan overcount" in str(w.message)
        ]
        assert not pre_scan_msgs, (
            f"M-02: spurious pre-scan overcount warning on M-13 file: {pre_scan_msgs}"
        )

    def test_units_row_skipped_wrapped(self, tmp_path: Path) -> None:
        """M-03: the M-13 units-row skip must also apply in WRAP=YES
        (wrapped) mode.  Pre-fix the wrapped path skipped the mnemonic
        header row but consumed the units row as a data step → phantom
        all-null first step + one-step depth shift
        (DEPT=[-999.25, 1000.0, 1001.0, 1002.0])."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "55.0\n"
            "1002.0\n"
            "60.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(
                _write_las(tmp_path, "m13_units_wrapped.las", content)
            )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 55.0, 60.0])

    def test_units_like_first_data_row_not_skipped_without_header(
        self, tmp_path: Path
    ) -> None:
        """Control: without a preceding mnemonic header row, a first row of
        letters-only tokens is DATA (an all-string section), never a units
        row — the skip requires the mnemonic header to have been seen first
        (position gate).  The all-string exclusion already stops the header
        skip; the units skip must not fire either."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " LITH.    :  Lithology {S}\n"
            " FORM.    :  Formation {S}\n"
            "~A\n"
            "ACME SAND\n"
            "SHALE GRN\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m13_control.las", content))
        # Both letters-only rows are genuine data rows: no mnemonic header
        # row precedes them, so nothing may be skipped as a units row.
        np.testing.assert_array_equal(
            las.string_data["LITH"], np.array(["ACME", "SHALE"], dtype=object)
        )
        np.testing.assert_array_equal(
            las.string_data["FORM"], np.array(["SAND", "GRN"], dtype=object)
        )


class TestM30CommaThousandsRecombined:
    """M-30 (CONFIRMED MEDIUM): LAS DLM=COMMA data lines with
    comma-grouped thousands ("1,234.5") split into fragments that silently
    mis-assign every subsequent column when the token count matches
    curve_count (zero warnings).  The LAS path now ports the DEV reader's
    recombination with the same loud per-pair warnings, including the
    equal-token-count case (E-24's hole)."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A DEPT RHOB NPHI\n"
            + rows
        )

    def test_equal_token_count_recombined_with_warning(self, tmp_path: Path) -> None:
        """Equal-token-count case (the E-24 hole): 3 tokens == 3 curves with
        a comma-grouped DEPT.  Pre-fix: DEPT=[1.0,5.0], RHOB=[234.5,60.0],
        NPHI=[50.0,0.2] with ZERO warnings.  Post-fix: recombined with a
        loud warning → DEPT=[1234.5,5.0], RHOB=[50.0,60.0], NPHI null-filled
        (short row)."""
        test_file = _write_las(
            tmp_path, "m30_equal.las", self._content("1,234.5,50\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert thousands_msgs, "M-30: no thousands-separator recombination warning"
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [-999.25, 0.2])

    def test_surplus_tokens_recombined(self, tmp_path: Path) -> None:
        """Surplus case: 4 tokens vs 3 curves — the run merges when the
        recombined count exactly satisfies the declared columns.  Pre-fix:
        DEPT=[1.0,5.0] with the 60 discarded as an extra column."""
        test_file = _write_las(
            tmp_path, "m30_surplus.las", self._content("1,234.5,50,60\n5,60,0.2\n")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [60.0, 0.2])

    def test_bare_fragment_row_not_false_recombined(self, tmp_path: Path) -> None:
        """M-23 control: a row of bare 3-digit fragments ("100,450") is a
        genuine multi-column row, never thousands — no merge, no warning
        (protects the equal-count widening from over-merging)."""
        test_file = _write_las(
            tmp_path, "m30_bare.las", self._content("100,450,20\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert not thousands_msgs, "M-30: bare fragment row must not recombine"
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [450.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [20.0, 0.2])

    def test_three_digit_leading_equal_count_not_false_recombined(
        self, tmp_path: Path
    ) -> None:
        """Widening control: at equal token count a THREE-digit leading
        group ("100,234.5,50" for 3 curves) stays ambiguous with genuine
        columns (DEPT=100, GR=234.5, RHOB=50 is plausible) — the merge is
        limited to 1-2 digit leading groups.  No merge, no warning."""
        test_file = _write_las(
            tmp_path, "m30_leading3.las", self._content("100,234.5,50\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert not thousands_msgs, "M-30: 3-digit leading group must not recombine"
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [234.5, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [50.0, 0.2])

    def test_wrapped_comma_continuation_recombined(self, tmp_path: Path) -> None:
        """M-30 in wrapped mode: a comma-DLM WRAP=YES file whose
        continuation line carries a comma-grouped thousands value
        ("1,234.5,5.0" = GR 1234.5 + RHOB 5.0 for step 1).  Pre-fix the
        split fragments enter the step buffer as [1,234.5,5.0] → a full
        step of WRONG values (DEPT=1, GR=234.5, RHOB=5.0).  Post-fix the
        recombination restores the true step alignment."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT GR RHOB\n"
            "1000.0\n"
            "1,234.5,5.0\n"
            "1001.0\n"
            "55.0,2.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(
                _write_las(tmp_path, "m30_wrapped.las", content)
            )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [1234.5, 55.0])
        np.testing.assert_allclose(las.logs["RHOB"], [5.0, 2.0])


# ──────────────────────────────────────────────────────────────
# Stage 12 fix regression pins — LAS read paths (M-32 LAS port,
# M-33 LAS port, F-02 shape (b) LAS port, M-06, M-07).  Each FAILS
# on pre-fix code and PASSES on post-fix.  Adversarial evidence:
# tmp/s11-adv-m2-report.md, tmp/s11-adv-m5-report.md.
# ──────────────────────────────────────────────────────────────


class TestM32LasPortDecimalExponent:
    """M-32 LAS port (CONFIRMED MEDIUM): the LAS 1.2/2.0 port of
    _THOUSANDS_FRAG_RE had the same decimal-OR-exponent grammar drift —
    "1,234.5E3" was detected but its fragment could not merge → silent
    column shift.  The fragment grammar now accepts decimal+exponent."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT RHOB\n"
            + rows
        )

    def test_decimal_exponent_recombined(self, tmp_path: Path) -> None:
        test_file = _write_las(
            tmp_path, "m32_las_decexp.las", self._content("1,234.5E3,5.0\n2,345.6E3,6.0\n")
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234500.0, 2345600.0])
        np.testing.assert_allclose(las.logs["RHOB"], [5.0, 6.0])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 1, f"M-32 LAS no thousands warning: {[str(w.message) for w in rec]}"


class TestM33LasPortTwoSingleGroup:
    """M-33 LAS port (CONFIRMED MEDIUM): two single-comma-group
    thousands values in the LAS 1.2/2.0 DLM=COMMA path failed the
    per-run exact-fit gate individually — full-row recombination now
    drives the gate (same fix as the DEV twin)."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT RHOB\n"
            + rows
        )

    def test_two_single_group_recombined(self, tmp_path: Path) -> None:
        test_file = _write_las(
            tmp_path, "m33_las_two_single.las", self._content("1,234.5,2,345.6\n3,456.7,4,567.8\n")
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 3456.7])
        np.testing.assert_allclose(las.logs["RHOB"], [2345.6, 4567.8])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 2, f"M-33 LAS fewer than 2 warnings: {[str(w.message) for w in rec]}"


class TestF02LasPortShapeB:
    """F-02 shape (b) LAS port (pass-2, CONFIRMED): the hybrid gate
    restores the surplus-row mixed-run shape on the LAS port — the
    unambiguous run merges while the genuine 3-digit pair stays."""

    def test_surplus_mixed_run(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A DEPT RHOB NPHI\n"
            "1,234.5,100,250.5\n"
            "2.0,3.0,4.0\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "f02_las_surplus.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 2.0])
        np.testing.assert_allclose(las.logs["RHOB"], [100.0, 3.0])
        np.testing.assert_allclose(las.logs["NPHI"], [250.5, 4.0])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 1, f"F-02 LAS no thousands warning: {[str(w.message) for w in rec]}"


class TestM06LettersRowPreservedOnReadPaths:
    """M-06 (CONFIRMED MEDIUM): the shared is_units_header_row predicate
    classified a genuine letters-only first DATA row directly after the
    mnemonic header (no units row) as units and dropped it on all 4 read
    paths.  The predicate now uses a units-form token pattern (every
    token letters-only AND <=4 chars AND >=1 token <=2 chars) so genuine
    data words ("ACME SAND") are preserved."""

    def test_las20_normal_letters_row_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_normal.las", content))
        # The letters row is preserved as a null-filled first data row.
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [-999.25, 150.0, 155.0])

    def test_units_row_still_skipped(self, tmp_path: Path) -> None:
        """Control: the standard M-13 units row ("M GAPI") is still
        skipped."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_units.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [150.0, 155.0])

    def test_las30_letters_row_preserved(self, tmp_path: Path) -> None:
        """LAS 3.0 path: same letters-only-first-row shape preserved."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_las30.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])

    def test_las30_deferred_letters_row_preserved(self, tmp_path: Path) -> None:
        """LAS 3.0 DEFERRED path (data section before ~VERSION): the
        same letters-only-first-row shape is preserved on replay, with no
        spurious pre-scan overcount warning."""
        content = (
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_las30_def.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        assert not any("overcount" in str(w.message) for w in rec), (
            f"spurious pre-scan overcount: {[str(w.message) for w in rec]}"
        )


class TestM07WrappedEmbeddedDelimiterWarning:
    """M-07 (CONFIRMED MEDIUM): the wrapped LAS 1.2/2.0 path had NO
    I2-02 embedded-delimiter warning (present only in _read_normal) — an
    embedded comma in a {S} string value silently corrupted columns with
    only a misleading N-I-08 diagnostic.  _read_wrapped now mirrors the
    I2-02 detection (step-boundary-overshoot signal)."""

    def test_wrapped_embedded_comma_warns(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " NAME.    :  Well Name {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "1000.0\n"
            "WELL, INC,50.0\n"
            "1001.0\n"
            "WELL, INC2,51.0\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            read_las_file_as_object(_write_las(tmp_path, "m07_wrapped.las", content))
        # The warning must name the delimiter-in-string mechanism (mirror
        # of the I2-02 contract asserted at test_regression.py:2612-2613).
        assert any("delimiter" in str(w.message) and "string" in str(w.message) for w in rec), (
            f"M-07 no embedded-delimiter warning: {[str(w.message) for w in rec]}"
        )
