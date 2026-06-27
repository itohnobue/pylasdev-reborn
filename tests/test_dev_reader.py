"""Tests for DEV file reader."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_dev_file, read_dev_file_as_object
from pylasdev.exceptions import DEVReadError
from pylasdev.models import DevFile


class TestReadDEVFile:
    """Tests for read_dev_file function."""

    def test_read_all_dev_files(self, all_dev_files: list[Path]) -> None:
        """Test reading every DEV file in test_data/."""
        for dev_path in all_dev_files:
            data = read_dev_file(dev_path)

            assert isinstance(data, dict)
            assert len(data) > 0

            for col_data in data.values():
                assert isinstance(col_data, np.ndarray)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error handling for missing file."""
        with pytest.raises(DEVReadError, match="File not found"):
            read_dev_file(tmp_path / "nonexistent.dev")

    def test_sample_dev_columns(self, test_data_dir: Path) -> None:
        """Test that sample.dev has expected columns."""
        sample_dev = test_data_dir / "sample.dev"
        assert sample_dev.exists(), f"Required test data missing: {sample_dev}"
        data = read_dev_file(sample_dev)
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert "Y" in data

    def test_sample_dev_data_shape(self, test_data_dir: Path) -> None:
        """Test that all columns have the same length."""
        sample_dev = test_data_dir / "sample.dev"
        assert sample_dev.exists(), f"Required test data missing: {sample_dev}"
        data = read_dev_file(sample_dev)
        sizes = [len(arr) for arr in data.values()]
        assert len(set(sizes)) == 1, f"Column sizes differ: {sizes}"

    def test_sample_dev_md_starts_at_zero(self, test_data_dir: Path) -> None:
        """Test that MD column starts at 0."""
        sample_dev = test_data_dir / "sample.dev"
        assert sample_dev.exists(), f"Required test data missing: {sample_dev}"
        data = read_dev_file(sample_dev)
        assert data["MD"][0] == 0.0

    def test_sample_dev_has_multiple_rows(self, test_data_dir: Path) -> None:
        """Test that sample.dev has multiple data rows."""
        sample_dev = test_data_dir / "sample.dev"
        assert sample_dev.exists(), f"Required test data missing: {sample_dev}"
        data = read_dev_file(sample_dev)
        assert len(data["MD"]) > 1

    def test_dev_values_are_numeric(self, all_dev_files: list[Path]) -> None:
        """Test that all values are numeric (float64)."""
        for dev_path in all_dev_files:
            data = read_dev_file(dev_path)
            for name, arr in data.items():
                assert arr.dtype == np.float64, (
                    f"Column {name} in {dev_path.name} has dtype {arr.dtype}"
                )

    def test_dev_encoding_parameter(self, test_data_dir: Path) -> None:
        """Test that explicit encoding parameter works."""
        sample_dev = test_data_dir / "sample.dev"
        assert sample_dev.exists(), f"Required test data missing: {sample_dev}"
        data = read_dev_file(sample_dev, encoding="utf-8")
        assert len(data) > 0

    def test_ragged_columns_fill_nan(self, tmp_path: Path) -> None:
        """Test that missing values in ragged DEV data become NaN, not 0.0."""
        content = (
            "# DEV file with ragged data\n"
            "MD TVD X Y\n"
            "0.0 0.0 100.0 200.0\n"
            "100.0 99.0 101.0\n"
            "200.0 198.0\n"
        )
        test_file = tmp_path / "ragged.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Row 1 (index 1): Y is missing → should be NaN
        assert np.isnan(data["Y"][1])
        # Row 2 (index 2): X and Y are missing → both NaN
        assert np.isnan(data["X"][2])
        assert np.isnan(data["Y"][2])
        # Filled values should be correct
        assert data["MD"][1] == 100.0
        assert data["TVD"][2] == 198.0

    def test_dev_file_model(self) -> None:
        """Test DevFile model to_dict."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0, 200.0])
        dev.columns["TVD"] = np.array([0.0, 99.0, 198.0])
        d = dev.to_dict()
        assert "MD" in d
        assert "TVD" in d
        np.testing.assert_array_equal(d["MD"], np.array([0.0, 100.0, 200.0]))

    def test_directory_input_raises_error(self, tmp_path: Path) -> None:
        """Test that passing a directory path raises DEVReadError."""
        with pytest.raises(DEVReadError, match="Not a file"):
            read_dev_file(tmp_path)

    def test_non_numeric_values(self, tmp_path: Path) -> None:
        """Test DEV file with non-numeric values gets substituted with NaN."""
        content = "MD TVD X Y\n0.0 0.0 100.0 200.0\n100.0 BAD 101.0 201.0\n200.0 198.0 102.0 ERR\n"
        test_file = tmp_path / "nonnum.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file)
        # BAD in TVD column -> NaN
        assert np.isnan(data["TVD"][1])
        # ERR in Y column -> NaN
        assert np.isnan(data["Y"][2])
        # Other values should parse correctly
        assert data["MD"][1] == 100.0
        assert data["X"][2] == 102.0

    # --- F-47: Direct test of read_dev_file_as_object ---
    def test_read_dev_file_as_object_direct(self, tmp_path: Path) -> None:
        """Test read_dev_file_as_object directly with a simple DEV file.

        Exercises dev_reader.py:45-67 — the read_dev_file_as_object function.
        """
        content = (
            "MD TVD X Y\n0.0 0.0 100.0 200.0\n100.0 99.0 101.0 201.0\n200.0 198.0 102.0 202.0\n"
        )
        test_file = tmp_path / "direct.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert isinstance(dev, DevFile)
        assert dev.source_file != ""
        assert dev.encoding != ""
        assert "MD" in dev.columns
        assert "TVD" in dev.columns
        assert len(dev.columns["MD"]) == 3
        np.testing.assert_array_equal(dev.columns["MD"], np.array([0.0, 100.0, 200.0]))

    # --- F-48: max_file_size for DEV ---
    def test_max_file_size_rejects_oversized_dev(self, tmp_path: Path) -> None:
        """Test that max_file_size parameter rejects oversized DEV files.

        Exercises dev_reader.py:69-76 — the max_file_size guard via
        read_with_encoding.
        """
        content = "MD TVD\n0.0 0.0\n100.0 99.0\n"
        test_file = tmp_path / "size.dev"
        test_file.write_text(content, encoding="utf-8")

        # Should succeed with generous limit
        data = read_dev_file(test_file, max_file_size=10_000_000)
        assert "MD" in data

        # Should fail with tiny limit
        with pytest.raises(ValueError, match="exceeds maximum"):
            read_dev_file(test_file, max_file_size=10)

        # Also test read_dev_file_as_object with max_file_size
        dev = read_dev_file_as_object(test_file, max_file_size=10_000_000)
        assert isinstance(dev, DevFile)

        with pytest.raises(ValueError, match="exceeds maximum"):
            read_dev_file_as_object(test_file, max_file_size=10)

    # --- F-49: Header-only DEV file ---
    def test_header_only_dev_file(self, tmp_path: Path) -> None:
        """Test reading a DEV file with only a header line, no data.

        Exercises dev_reader.py:86-96 — the header-only path where
        data_lines is 0.
        """
        content = "MD TVD X Y\n"
        test_file = tmp_path / "header_only.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert isinstance(data, dict)
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert "Y" in data
        # All columns should be empty (0-length)
        assert len(data["MD"]) == 0
        assert len(data["TVD"]) == 0
        assert len(data["X"]) == 0
        assert len(data["Y"]) == 0

    # --- F-50: Extra columns in data row vs header ---
    def test_extra_columns_in_data_row(self, tmp_path: Path) -> None:
        """Test DEV file with data row having more values than header columns.

        Exercises dev_reader.py:119-126 — the min(len(values), len(names))
        guard that prevents IndexError on extra columns.
        """
        content = "MD TVD\n0.0 0.0 100.0 200.0\n100.0 99.0\n"
        test_file = tmp_path / "extra_cols.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Extra values in first row should be silently ignored
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["TVD"][0] == 0.0
        assert data["MD"][1] == 100.0


class TestDevIndexErrorHandler:
    """CF-022: DEV reader IndexError handler tests."""

    def test_normal_parsing_handled_gracefully(self, tmp_path: Path) -> None:
        """Test DEV reader handles normal data without errors.

        The IndexError handler (dev_reader.py:191-192) is a safety net for
        pass-1/pass-2 data-line count inconsistency. In practice, both
        passes count the same data, so the handler is unreachable under
        normal conditions. This test verifies that normal parsing works
        correctly.
        """
        content = "MD TVD X\n0.0 0.0 100.0\n100.0 99.0 101.0\n200.0 198.0 102.0\n"
        test_file = tmp_path / "normal.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert len(data["MD"]) == 3
        assert data["MD"][0] == 0.0
        assert data["MD"][2] == 200.0
        assert data["X"][0] == 100.0
        assert data["X"][2] == 102.0


class TestDevDedup:
    """F12: DEV deduplication tests.

    The DEV reader (dev_reader.py:146-194) has deduplication logic
    mirroring LAS _deduplicate_curves: duplicate column names get
    ``_N`` suffixes, and cross-base collisions (where an original
    name matches a previously generated ``_N`` suffix) are resolved
    correctly.
    """

    def test_duplicate_column_names_get_suffix(self, tmp_path: Path) -> None:
        """Duplicate DEV columns receive _N suffixes.

        Input columns: ["DEPTH", "GR", "GR"]
        Expected: ["DEPTH", "GR", "GR_2"] — first duplicate starts at _2
        (matching the LAS _deduplicate_curves convention).
        """
        content = "DEPTH GR GR\n100.0 10.0 20.0\n"
        test_file = tmp_path / "dup.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            # Should warn about the duplicate
            assert any("Duplicate DEV column name" in str(x.message) for x in w)

        assert list(data.keys()) == ["DEPTH", "GR", "GR_2"]
        np.testing.assert_array_equal(data["DEPTH"], [100.0])
        np.testing.assert_array_equal(data["GR"], [10.0])
        np.testing.assert_array_equal(data["GR_2"], [20.0])

    def test_cross_base_collision_dedup(self, tmp_path: Path) -> None:
        """Cross-base collision: original name matches prior _N suffix.

        Input columns: ["A", "A", "A_2"]
        Expected: ["A", "A_2", "A_2_2"] — second "A" → "A_2" which
        collides with the third original name "A_2", so the third
        becomes "A_2_2".
        """
        content = "A A A_2\n10.0 20.0 30.0\n"
        test_file = tmp_path / "cross_base.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            dup_warnings = [x for x in w if "Duplicate DEV column name" in str(x.message)]
            assert len(dup_warnings) >= 2

        assert list(data.keys()) == ["A", "A_2", "A_2_2"]
        np.testing.assert_array_equal(data["A"], [10.0])
        np.testing.assert_array_equal(data["A_2"], [20.0])
        np.testing.assert_array_equal(data["A_2_2"], [30.0])


class TestDevSafetyGuards:
    """F13/F37: DEV reader safety guard tests.

    Tests all six guards in dev_reader.py:
    - OSError handler (line 98)
    - MAX_DATA_LINES (line 119-123)
    - MAX_CURVES (line 196-200)
    - MAX_TOTAL_ELEMENTS (line 201-206)
    - IndexError handler (line 225-226)
    """

    def test_oserror_read_with_encoding_raises_dev_read_error(self, tmp_path: Path) -> None:
        """OSError from read_with_encoding is wrapped in DEVReadError.

        Exercises dev_reader.py:97-98.
        """
        from unittest import mock

        test_file = tmp_path / "denied.dev"
        test_file.write_text("MD TVD\n0.0 0.0\n", encoding="utf-8")

        with mock.patch(
            "pylasdev.dev_reader.read_with_encoding",
            side_effect=OSError("Permission denied"),
        ):
            with pytest.raises(DEVReadError, match="Cannot read file"):
                read_dev_file(test_file)

    def test_max_data_lines_guard(self, tmp_path: Path) -> None:
        """MAX_DATA_LINES guard raises DEVReadError for excess data lines.

        Exercises dev_reader.py:119-123.
        """
        from unittest import mock

        content = "MD TVD\n0.0 0.0\n100.0 99.0\n"
        test_file = tmp_path / "max_lines.dev"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_DATA_LINES to 0: any data line triggers guard
        with mock.patch("pylasdev.dev_reader.MAX_DATA_LINES", 0):
            with pytest.raises(DEVReadError, match="Data line count"):
                read_dev_file(test_file)

    def test_max_curves_guard(self, tmp_path: Path) -> None:
        """MAX_CURVES guard raises DEVReadError for excess columns.

        Exercises dev_reader.py:196-200.
        """
        from unittest import mock

        content = "MD TVD X\n0.0 0.0 100.0\n"
        test_file = tmp_path / "max_curves.dev"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_CURVES to 1: 3 columns > 1
        with mock.patch("pylasdev.dev_reader.MAX_CURVES", 1):
            with pytest.raises(DEVReadError, match="Column count"):
                read_dev_file(test_file)

    def test_max_total_elements_guard(self, tmp_path: Path) -> None:
        """MAX_TOTAL_ELEMENTS guard raises DEVReadError for excess allocation.

        Exercises dev_reader.py:201-206.
        """
        from unittest import mock

        content = "MD TVD X\n0.0 0.0 100.0\n"
        test_file = tmp_path / "max_total.dev"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_TOTAL_ELEMENTS to 1: 3 cols * 1 line = 3 > 1
        with mock.patch("pylasdev.dev_reader.MAX_TOTAL_ELEMENTS", 1):
            with pytest.raises(DEVReadError, match="Total allocation"):
                read_dev_file(test_file)

    def test_index_error_handler_caught_by_safety_net(self, tmp_path: Path) -> None:
        """IndexError in _to_finite_float is caught by safety net.

        Exercises dev_reader.py:225-226.  The IndexError handler is a
        safety net for pass-1/pass-2 line-count inconsistency.  Under
        normal conditions both passes process the same lines identically
        so the handler is unreachable.  We trigger it by mocking
        ``_to_finite_float`` to raise IndexError, which the handler
        catches and substitutes ``np.nan``.
        """
        from unittest import mock

        content = "MD TVD\n0.0 0.0\n"
        test_file = tmp_path / "index_err.dev"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch(
            "pylasdev.dev_reader._to_finite_float",
            side_effect=IndexError("simulated"),
        ):
            data = read_dev_file(test_file)
            # Values should be NaN (substituted by handler)
            assert np.isnan(data["MD"][0])
            assert np.isnan(data["TVD"][0])

    def test_index_error_handler_read_dev_file_as_object(self, tmp_path: Path) -> None:
        """IndexError safety net also covers read_dev_file_as_object path.

        Same as above but exercises the read_dev_file_as_object API.
        """
        from unittest import mock

        content = "X Y\n1.0 2.0\n"
        test_file = tmp_path / "index_err2.dev"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch(
            "pylasdev.dev_reader._to_finite_float",
            side_effect=IndexError("simulated"),
        ):
            dev = read_dev_file_as_object(test_file)
            assert np.isnan(dev.columns["X"][0])
            assert np.isnan(dev.columns["Y"][0])
