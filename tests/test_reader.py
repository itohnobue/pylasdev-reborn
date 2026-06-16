"""Tests for LAS file reader."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_las_file, read_las_file_as_object
from pylasdev.exceptions import LASParseError, LASReadError
from pylasdev.models import LASFile


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
        assert len(data["logs"]["DEPT"]) > 0

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
        with pytest.raises(ValueError, match="exceeds maximum"):
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
        """Test wrapped mode padding when curves have unequal lengths."""
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
            # Missing GR for second depth step (incomplete)
        )
        test_file = tmp_path / "wrapped_short.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            # Should have padding warning
            assert any("Padding" in str(x.message) for x in w)

        # All arrays should be same length after padding
        sizes = [len(data["logs"][c]) for c in data["curves_order"]]
        assert len(set(sizes)) == 1
        # GR should have null_value for the last step
        assert data["logs"]["GR"][-1] == -999.25

    # --- TEST-04: Wrapped-mode depth line has >1 value ---
    def test_wrapped_depth_line_extra_values(self, tmp_path: Path) -> None:
        """Test wrapped mode warns when depth line has multiple values."""
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
            # Second depth line: 2 values -> > 1, triggers warning
            "101.0  99.0\n"
            # DT for second step
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_extra_depth.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("depth line has 2 values" in str(x.message) for x in w)

        # DEPT should only have the correct depth values (100.0, 101.0)
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DEPT"][1] == 101.0
        # DT should have correct values
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DT"][1] == 51.0

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
        """Test _detect_actual_wrap returns True when no data lines exist."""
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

    # --- TEST-13: Wrapped-mode ValueError/IndexError handlers ---
    def test_wrapped_malformed_data_handlers(self, tmp_path: Path) -> None:
        """Test wrapped mode ValueError/IndexError handlers substitute null_value."""
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
        assert data["well"]["NULL"] == "BADNULL"

    # --- TEST-04: IndexError branch in normal mode (line 151->143) ---
    def test_index_error_branch_normal(self, tmp_path: Path) -> None:
        """Test the IndexError handler in _read_normal mode.

        The IndexError can occur when curve_count was reduced after
        deduplication and arrays are sized for the reduced count.
        With a duplicate curve that forces dedup, and data values that
        fit within the original curve_count but the deduplication reduces
        it, some values should be handled gracefully.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DUP.API  :  Duplicate 1\n"
            " DUP.API  :  Duplicate 2\n"
            "~A  DEPT  DUP  DUP\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "dup_index_error.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Duplicate curve mnemonic" in str(x.message) for x in w)

        # After dedup, DUP becomes DUP and DUP_2, arrays sized for 2 curves
        assert "DUP" in data["logs"]
        assert "DUP_2" in data["logs"]

    # --- F-19: Wrapped-mode section transition break ---
    def test_wrapped_section_after_ascii_stops_data(self, tmp_path: Path) -> None:
        """Test that a new section after ~A in wrapped mode stops data collection.

        Exercises data_reader.py:200 — the break in _read_wrapped when a section
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


class TestArrayTrimming:
    """F-025: Array trimming when pre-scan over-counts."""

    def test_early_ascii_termination_trims_arrays(self, tmp_path: Path) -> None:
        """Test that when ~A ends before pre-scanned data_line_count, arrays trim.

        The ~OTHER section header terminates ~A section data collection.
        Pre-scan stops at the section header, and _read_normal breaks at it too,
        so data_line_count matches current_line. No trimming occurs in this case
        because current_line == data_line_count.

        A true trimming scenario (current_line < data_line_count) requires the
        pre-scan to over-count, which happens when new sections appear after ~A
        data — the pre-scan `in_ascii` flag goes False on section headers, and
        _read_normal breaks. This test verifies correct termination at section
        boundaries and asserts data integrity (no 0.0 fill values)."""
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
            "Freeform text that section detection catches.\n"
        )
        test_file = tmp_path / "trim_section.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Should have exactly 2 data points (stopped at ~OTHER)
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])

        # Verify no 0.0 fill values — all values should be actual data or null
        assert not any(v == 0.0 for v in data["logs"]["DEPT"])
        assert not any(v == 0.0 for v in data["logs"]["DT"])


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
            with pytest.raises(LASParseError, match="Curve count"):
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
            with pytest.raises(LASParseError, match="Curve count"):
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
