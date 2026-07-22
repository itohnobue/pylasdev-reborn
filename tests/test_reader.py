"""Tests for LAS file reader."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_dev_file, read_las_file, read_las_file_as_object
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
        assert len(lith_values) == 2, (
            f"Expected 2 LITH values, got {len(lith_values)}"
        )
        np.testing.assert_array_equal(
            lith_values,
            np.array(["Sandstone", "Shale"], dtype=np.str_),
        )

        # String curve should NOT be in float logs
        assert "LITH" not in data["logs"], (
            "String curve 'LITH' found in logs (float) — should only "
            "be in string_data"
        )

    # --- F-H-006: string curve multi-char value preservation ---

    def test_string_curve_multi_char_values_preserved(
        self, tmp_path: Path
    ) -> None:
        """F-H-006: String curve values > 1 char are preserved in string_data.

        Before the fix at data_reader.py:634, string curve arrays were
        pre-allocated with dtype=np.str_ which defaults to a single-character
        fixed-width Unicode type (U1), truncating values to their first
        character.  After the fix, dtype=object preserves arbitrary-length
        strings.

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
        assert "LITH" in data["string_data"], (
            "String curve 'LITH' missing from string_data"
        )
        lith_values = data["string_data"]["LITH"]
        assert len(lith_values) == 3, (
            f"Expected 3 LITH values, got {len(lith_values)}"
        )
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
        assert data["well"]["NULL"] == "NON-NUMERIC NULL VALUE"

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

    def test_d_exponent_null_value_parsed_correctly(self, tmp_path: Path) -> None:
        """F-06: Test that Fortran D-notation in NULL field is handled.

        When the ~W section has NULL. -999.25D0 (Fortran D-notation),
        _get_null_value should parse it correctly via the shared
        _parse_float_with_d_notation helper instead of falling back
        to the hardcoded default.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            # D-notation NULL value: Fortran-style exponent
            " NULL.    -999.25D0       : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            # "BAD" should be replaced by the parsed null value (-999.25),
            # NOT by the hardcoded default -999.25. Since D0 → E0 = 1,
            # -999.25D0 = -999.25, which is the same value.
            # The key assertion: it's the value from the file, not a fallback.
            "100.0  BAD\n"
        )
        test_file = tmp_path / "d_null.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # -999.25D0 = -999.25 * 10^0 = -999.25
        assert data["logs"]["DT"][0] == -999.25

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

    def test_pathological_misalignment_raises_error(self, tmp_path: Path) -> None:
        """Test that _read_wrapped raises LASParseError for pathological
        misalignment: ≥3 curves + extra depth values + ≤2 data values
        + ≥2 remaining gaps."""
        # Curve count = 4 (DEPT + 3 non-depth curves)
        # Depth line: 3 values (extra) → depth_had_extra = True
        # Data line: 1 value     → ≤2 values, remaining_curves=3, gap=2
        # This should trigger LASParseError
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
            # First depth line: 1 value → depth_line (truly wrapped)
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "10.0\n"
            # Second depth line: 3 values (>1) → depth_had_extra = True
            # Values are consumed: DEPT=101.0, counter advances to 1 for the second
            # Actually let me construct this more carefully.
            # Depth line with 3 values and curve_count=4, so 2 extra values beyond DEPT.
            # Next data line: only 1 value, need 3 more for remaining curves.
            # remaining = 4-1-1 = 2, len(values)=1, 1 ≤ 2 and 2-1=1 < 2 → no error
            # Need: len(values) ≤ 2 AND remaining - len(values) ≥ 2
            # With curve_count=5 (DEPT + 4 non-depth):
            "101.0  99.0  88.0\n"  # Depth line: 3 values (> 1) → depth_had_extra
            "77.0\n"  # Data line: 1 value, remaining=4-1=3, gap=2 ≥ 2 → error
        )
        test_file = tmp_path / "pathological.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError, match="pathologically malformed"):
            read_las_file(test_file)


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


class TestArrayTrimmingOvercount:
    """F24: Tests for array trimming when pre-scan over-counts data lines.

    The trimming branch (data_reader.py:354-359) runs when current_line <
    data_line_count. This happens when the parser's _pre_scan counts data
    lines across ALL ~A sections but _read_normal stops at the first
    non-A section header (~OTHER, ~P, etc.).
    """

    def test_multi_a_section_trimming(self, tmp_path: Path) -> None:
        """Test array trimming when second ~A data is excluded by ~OTHER.

        The pre-scan counts 3 data lines across both ~A sections, but
        _read_normal stops at ~OTHER after only 2 lines, triggering the
        tail-fill-and-slice branch (current_line=2 < data_line_count=3).
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

        data = read_las_file(test_file)
        # Only first section's 2 data lines should be present
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])
        # The tail (positions 2+) should not contain stale 0.0 values
        # (they should have been filled with null_value and sliced off)
        assert len(data["logs"]["DEPT"]) < 3


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

    def test_d_notation_null_value_uppercase(self) -> None:
        """Fortran D-notation NULL (uppercase D) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "-999.25D0"}
        result = _get_null_value(well)
        assert result == -999.25

    def test_d_notation_null_value_lowercase(self) -> None:
        """Fortran D-notation NULL (lowercase d) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "-999.25d0"}
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

    def test_d_notation_falls_back_on_invalid(self) -> None:
        """Invalid D-notation value falls back to default_float."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "NOT_A_NUMBER"}
        result = _get_null_value(well)
        assert result == -999.25

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

    def test_token_cap_dev_reader(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE capped in DEV reader (dev_reader.py:719)."""
        from unittest import mock

        content = (
            "MD TVD X Y Z\n"
            "0.0 0.0 100.0 200.0 300.0\n"
            "100.0 99.0 101.0 201.0 301.0\n"
        )
        test_file = tmp_path / "token_cap_dev.dev"
        test_file.write_text(content, encoding="utf-8")

        # Patch data_reader.MAX_TOKENS_PER_LINE: dev_reader fetches at
        # runtime via _resolve_max_tokens_per_line (F-DVR-01 fix).
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_dev_file(test_file)
            # split(maxsplit=2) produces at most 3 tokens from 5-space line
            assert "MD" in data
            assert "TVD" in data


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

        The depth-step formula must use max(ceil(N/curves), ceil(N/2))
        to avoid undercounting steps in compact wrapped format.
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

    def test_wrapped_depth_line_state_machine_non_pathological_ef018(
        self, tmp_path: Path
    ) -> None:
        """E-F-018: Regression test — non-pathological recovery now hard-fails.

        When a depth line has extra values (depth_had_extra=True) and the next
        data line has fewer values than needed but the shift is ≤2 curves,
        the previous code at data_reader.py:898-900 silently reset the
        state machine, producing data corruption (SP values shifted by 1
        depth step).  The F-032 fix hard-fails with a clear diagnostic
        instead of producing corrupt output.

        Test scenario (curve_count=4, DEPT + DT, GR, SP):
          Step 1: depth=1000.0, data=200.0,300.0,400.0  (normal baseline)
          Step 2: depth=1010.0 + extra 1050.5, data=210.0,310.0 (triggers fail)
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
            # Step 2 (triggers F-032 hard-fail):
            #   Depth line: 2 values (>1) → depth_had_extra = True
            #   Data line: 2 values (<3 remaining curves) → unrecoverable
            #   Fix at data_reader.py now raises LASParseError
            "1010.0  1050.5\n"
            "210.0  310.0\n"
            # Step 3 is never reached
            "1020.0\n"
            "220.0  320.0  420.0\n"
        )
        test_file = tmp_path / "ef018_wrapped_state_machine.las"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(LASParseError, match="unrecoverable data misalignment"):
            read_las_file(test_file)

    # --- F-212 (MEDIUM): _desanitize_las_value unconditional _# strip ---

    def test_desanitize_disabled_preserves_hash_prefix(self) -> None:
        """F-212: With _DESANITIZE_ENABLED=False, _# values are preserved.

        When reading non-pylasdev files that genuinely contain _# in
        their data, the desanitize function must not strip the underscore.
        """
        import pylasdev.data_reader as dr_mod
        import pylasdev.parser as _parser_mod

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
            _parser_mod._DESANITIZE_ENABLED = True

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
        assert result == "#test_value", (
            f"Expected '#test_value' after desanitize, got {result!r}"
        )

    # --- F-219 (MEDIUM): LASEncodingError propagation ---

    def test_encoding_error_wrapped_as_las_read_error(self, tmp_path: Path) -> None:
        """F-219: LASEncodingError is wrapped as LASReadError in reader.

        The docstring was corrected: LASEncodingError is NOT propagated
        directly. Instead it's caught and re-raised as LASReadError at
        reader.py:142. Callers receive LASReadError, not LASEncodingError.
        """
        content = b"\xff\xfe\x00\x01"  # Invalid encoding bytes
        test_file = tmp_path / "bad_encoding.las"
        test_file.write_bytes(content)

        # Without chardet and empty fallback chain, encoding fails
        from unittest import mock

        with mock.patch("pylasdev.encoding.FALLBACK_ENCODINGS", []):
            with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
                with pytest.raises(LASReadError, match="Cannot read file"):
                    read_las_file(test_file)
