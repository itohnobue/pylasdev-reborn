"""Tests for LAS file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_las_file, read_las_file_as_object
from pylasdev.exceptions import LASReadError
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
        if not sample.exists():
            pytest.skip("sample.las not found")
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
        if not ct.exists():
            pytest.skip("comment_test.las not found")
        data = read_las_file(ct)
        # All arrays should have equal length
        if data["curves_order"]:
            sizes = [len(data["logs"][c]) for c in data["curves_order"]]
            assert len(set(sizes)) == 1, f"Arrays have different sizes: {sizes}"

    def test_encoding_parameter(self, test_data_dir: Path) -> None:
        """Test that explicit encoding parameter works."""
        sample = test_data_dir / "sample.las"
        if not sample.exists():
            pytest.skip("sample.las not found")
        data = read_las_file(sample, encoding="utf-8")
        assert "logs" in data


class TestReadLASFileAsObject:
    """Tests for read_las_file_as_object function."""

    def test_returns_las_file_object(self, test_data_dir: Path) -> None:
        """Test that read_las_file_as_object returns LASFile."""
        sample = test_data_dir / "sample.las"
        if not sample.exists():
            pytest.skip("sample.las not found")
        las = read_las_file_as_object(sample)
        assert isinstance(las, LASFile)
        assert las.source_file != ""
        assert las.encoding != ""

    def test_object_has_curves(self, test_data_dir: Path) -> None:
        """Test LASFile object has curve definitions."""
        sample = test_data_dir / "sample.las"
        if not sample.exists():
            pytest.skip("sample.las not found")
        las = read_las_file_as_object(sample)
        assert len(las.curves) > 0
        assert len(las.curves_order) > 0
        assert len(las.logs) > 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error for missing file."""
        with pytest.raises(LASReadError):
            read_las_file_as_object(tmp_path / "missing.las")

    def test_las30_object_has_version(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 file parsed as object has correct version."""
        las30 = test_data_dir / "sample_3.0.las"
        if not las30.exists():
            pytest.skip("sample_3.0.las not found")
        las = read_las_file_as_object(las30)
        assert las.version.vers == "3.0"
        assert las.version.is_las30 is True
        assert las.version.dlm == "COMMA"

    def test_las30_curves_with_formats(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 curve format specifiers are parsed."""
        las30 = test_data_dir / "sample_3.0.las"
        if not las30.exists():
            pytest.skip("sample_3.0.las not found")
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
        if not las30.exists():
            pytest.skip("sample_3.0.las not found")
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
        if not las30.exists():
            pytest.skip("sample_3.0.las not found")
        las = read_las_file_as_object(las30)
        # Check zone-associated parameters
        assert len(las.parameters) > 0
        zoned = [p for p in las.parameters if p.zone is not None]
        assert len(zoned) > 0

    def test_las30_scientific_notation_format(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 {E} (scientific notation) format specifier."""
        las30 = test_data_dir / "sample_3.0.las"
        if not las30.exists():
            pytest.skip("sample_3.0.las not found")
        las = read_las_file_as_object(las30)
        yme_curve = las.get_curve_by_mnemonic("YME")
        assert yme_curve is not None
        assert yme_curve.data_format == "E"


class TestDataReaderEdgeCases:
    """Tests for data reader edge cases and boundary conditions."""

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
