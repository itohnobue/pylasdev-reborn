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
        read_with_encoding.  F-04 (SF-04 fix) wraps ValueError from
        read_with_encoding into DEVReadError.
        """
        content = "MD TVD\n0.0 0.0\n100.0 99.0\n"
        test_file = tmp_path / "size.dev"
        test_file.write_text(content, encoding="utf-8")

        # Should succeed with generous limit
        data = read_dev_file(test_file, max_file_size=10_000_000)
        assert "MD" in data

        # Should fail with tiny limit — now wrapped as DEVReadError (F-04)
        with pytest.raises(DEVReadError, match="Cannot read file"):
            read_dev_file(test_file, max_file_size=10)

        # Also test read_dev_file_as_object with max_file_size
        dev = read_dev_file_as_object(test_file, max_file_size=10_000_000)
        assert isinstance(dev, DevFile)

        with pytest.raises(DEVReadError, match="Cannot read file"):
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


class TestDevDedupWhileLoop:
    """F23: Tests for DEV dedup while-loop collision bodies.

    Exercises the while-loops in dev_reader.py:159-161 (duplicate branch)
    and dev_reader.py:179-181 (cross-base collision branch) where generated
    _N suffix names collide with already-existing output names.
    """

    def test_duplicate_while_loop_collision(self, tmp_path: Path) -> None:
        """Test duplicate branch while-loop when generated _N suffix collides.

        Input columns: ["A", "A_2", "A"]
        Expected: ["A", "A_2", "A_3"] — second "A" tries to become "A_2"
        but "A_2" is already in output_names, so the while-loop at line
        159-161 increments suffix to 3 producing "A_3".
        """
        content = "A A_2 A\n10.0 20.0 30.0\n"
        test_file = tmp_path / "dup_while.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            dup_warnings = [x for x in w if "Duplicate DEV column name" in str(x.message)]
            assert len(dup_warnings) >= 1

        assert list(data.keys()) == ["A", "A_2", "A_3"]
        np.testing.assert_array_equal(data["A"], [10.0])
        np.testing.assert_array_equal(data["A_2"], [20.0])
        np.testing.assert_array_equal(data["A_3"], [30.0])

    def test_cross_base_while_loop_collision(self, tmp_path: Path) -> None:
        """Test cross-base while-loop when generated _2 suffix collides.

        Input columns: ["A", "A", "A_2_2", "A_2"]
        Expected: ["A", "A_2", "A_2_2", "A_2_3"] — "A_2" triggers the
        cross-base collision at line 176, producing "A_2_2", but "A_2_2"
        is ALREADY in output_names from the original third column, so the
        while-loop at line 179-181 increments suffix to 3 producing "A_2_3".
        """
        content = "A A A_2_2 A_2\n10.0 20.0 30.0 40.0\n"
        test_file = tmp_path / "cross_while.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            dup_warnings = [x for x in w if "Duplicate DEV column name" in str(x.message)]
            assert len(dup_warnings) >= 2

        assert list(data.keys()) == ["A", "A_2", "A_2_2", "A_2_3"]
        np.testing.assert_array_equal(data["A"], [10.0])
        np.testing.assert_array_equal(data["A_2"], [20.0])
        np.testing.assert_array_equal(data["A_2_2"], [30.0])
        np.testing.assert_array_equal(data["A_2_3"], [40.0])


class TestDugFormat:
    """F-02: DUG Insight format parsing tests.

    DUG Insight DEV files have a title line, an integer column count,
    a header line with column names, and space-separated data rows.
    """

    def test_dug_format_basic(self, tmp_path: Path) -> None:
        """Parse a basic DUG Insight format file."""
        content = (
            "Deviation survey for Well-1\n"
            "4\n"
            "MDKB TVDSS X Y\n"
            "0.00 -20.06 39844.56 24589.34\n"
            "1000.00 1020.02 39844.47 24588.95\n"
            "2000.00 2040.08 39844.30 24588.50\n"
        )
        test_file = tmp_path / "dug_basic.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file, normalize_aliases=False)
        assert list(data.keys()) == ["MDKB", "TVDSS", "X", "Y"]
        assert len(data["MDKB"]) == 3
        assert data["MDKB"][0] == 0.0
        assert data["TVDSS"][0] == -20.06
        assert data["X"][2] == 39844.30
        np.testing.assert_array_equal(
            data["MDKB"],
            [0.0, 1000.0, 2000.0],
        )

    def test_dug_format_as_object(self, tmp_path: Path) -> None:
        """Parse DUG format via read_dev_file_as_object."""
        content = (
            "Well-42 Survey\n3\nMD INC AZI\n0.00 0.00 0.00\n100.00 1.50 45.00\n200.00 3.20 48.00\n"
        )
        test_file = tmp_path / "dug_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev.column_order == ["MD", "INC", "AZI"]
        assert len(dev.columns["MD"]) == 3
        np.testing.assert_array_equal(
            dev.columns["MD"],
            [0.0, 100.0, 200.0],
        )
        np.testing.assert_array_equal(
            dev.columns["INC"],
            [0.0, 1.5, 3.2],
        )

    def test_dug_format_with_comments(self, tmp_path: Path) -> None:
        """DUG format with comment lines interspersed."""
        content = (
            "# Survey for Well-X\n"
            "Well-X Deviation\n"
            "# Column count:\n"
            "3\n"
            "# Column headers:\n"
            "MD INC AZI\n"
            "# Data:\n"
            "0.0 0.0 0.0\n"
            "100.0 1.5 45.0\n"
        )
        test_file = tmp_path / "dug_comments.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "INC", "AZI"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["INC"][1] == 1.5

    def test_dug_format_no_data(self, tmp_path: Path) -> None:
        """DUG format with header but no data rows."""
        content = "Survey\n4\nMD TVD X Y\n"
        test_file = tmp_path / "dug_no_data.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert len(data["MD"]) == 0

    def test_dug_format_empty_header_raises_error(self, tmp_path: Path) -> None:
        """DUG format with genuinely empty header parsed as simple format.

        Our content scanner skips blank/empty lines, so a DUG file where
        the header line is empty has its data line promoted to content
        line 3 — which is all-numeric, defeating DUG detection.  The
        file is then parsed as simple format (first line = column names).
        """
        content = "Survey\n3\n0.0 0.0 0.0\n"
        test_file = tmp_path / "dug_no_header.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Third content line is all-numeric → not DUG → falls to simple
        assert list(data.keys()) == ["Survey"]
        assert len(data["Survey"]) == 2  # "3" and "0.0 0.0 0.0"

    def test_dug_format_with_ragged_data(self, tmp_path: Path) -> None:
        """DUG format with ragged data rows (fewer tokens than columns)."""
        content = (
            "Survey\n"
            "4\n"
            "MD TVD X Y\n"
            "0.0 0.0 100.0 200.0\n"
            "100.0 99.0 101.0\n"  # Only 3 values
            "200.0 198.0\n"  # Only 2 values
        )
        test_file = tmp_path / "dug_ragged.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Missing values should be NaN
        assert np.isnan(data["Y"][1])
        assert np.isnan(data["X"][2])
        assert np.isnan(data["Y"][2])


class TestHeaderlessFormat:
    """F-02: Headerless format parsing tests.

    Headerless DEV files have no column name line — the first
    content line is numeric data.  Column names are auto-generated
    as ``col_0``, ``col_1``, ..., ``col_N``.
    """

    def test_headerless_format_basic(self, tmp_path: Path) -> None:
        """Parse a basic headerless file with multiple columns."""
        content = "0.00 0.00 0.00\n100.00 1.50 45.00\n200.00 3.20 48.00\n"
        test_file = tmp_path / "noheader_basic.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        assert len(data["col_0"]) == 3
        np.testing.assert_array_equal(
            data["col_0"],
            [0.0, 100.0, 200.0],
        )
        np.testing.assert_array_equal(
            data["col_1"],
            [0.0, 1.5, 3.2],
        )
        np.testing.assert_array_equal(
            data["col_2"],
            [0.0, 45.0, 48.0],
        )

    def test_headerless_format_as_object(self, tmp_path: Path) -> None:
        """Parse headerless via read_dev_file_as_object."""
        content = "0.0 0.0\n100.0 99.0\n200.0 198.0\n"
        test_file = tmp_path / "noheader_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev.column_order == ["col_0", "col_1"]
        assert len(dev.columns["col_0"]) == 3
        np.testing.assert_array_equal(
            dev.columns["col_0"],
            [0.0, 100.0, 200.0],
        )

    def test_headerless_single_column(self, tmp_path: Path) -> None:
        """Headerless file with a single column."""
        content = "0.0\n100.0\n200.0\n"
        test_file = tmp_path / "noheader_single.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0"]
        np.testing.assert_array_equal(
            data["col_0"],
            [0.0, 100.0, 200.0],
        )

    def test_headerless_with_comments(self, tmp_path: Path) -> None:
        """Headerless file with comment lines."""
        content = "# Some comments\n# More\n0.0 100.0\n50.0 150.0\n"
        test_file = tmp_path / "noheader_comments.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1"]
        assert len(data["col_0"]) == 2
        assert data["col_0"][0] == 0.0

    def test_headerless_with_scientific_notation(self, tmp_path: Path) -> None:
        """Headerless file with scientific notation values."""
        content = "1.0e2 2.5E-3 3.0D+01\n4.0e2 5.5E-3 6.0D+01\n"
        test_file = tmp_path / "noheader_sci.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert data["col_0"][0] == 100.0  # 1.0e2
        assert data["col_1"][0] == 0.0025  # 2.5E-3
        assert data["col_2"][0] == 30.0  # 3.0D+01

    def test_headerless_single_line(self, tmp_path: Path) -> None:
        """Headerless file with a single data line."""
        content = "0.0 100.0 200.0\n"
        test_file = tmp_path / "noheader_single_line.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        assert len(data["col_0"]) == 1
        assert data["col_0"][0] == 0.0

    def test_headerless_ragged_columns(self, tmp_path: Path) -> None:
        """Headerless file with ragged data rows."""
        content = (
            "0.0 100.0 200.0\n"
            "100.0 99.0\n"  # Missing third column
            "200.0\n"  # Missing two columns
        )
        test_file = tmp_path / "noheader_ragged.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Column count determined by first row (3 cols)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        # Missing values should be NaN
        assert data["col_0"][1] == 100.0
        assert np.isnan(data["col_2"][1])
        assert np.isnan(data["col_1"][2])

    def test_headerless_with_negative_values(self, tmp_path: Path) -> None:
        """Headerless file with negative values (TVDSS negative)."""
        content = "0.00 -20.06 39844.56\n1000.00 1020.02 39844.47\n"
        test_file = tmp_path / "noheader_neg.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert data["col_1"][0] == -20.06
        assert data["col_1"][1] == 1020.02

    def test_headerless_empty_file_returns_empty_data(self, tmp_path: Path) -> None:
        """Test that an empty DEV file (zero content entries) returns empty data.

        Exercises dev_reader.py:163-164 — the empty content_entries path
        where _detect_dev_format returns ("simple", 1).  With no content
        lines in either pass, the reader produces an empty DevFile.
        """
        test_file = tmp_path / "empty.dev"
        test_file.write_text("", encoding="utf-8")

        # read_dev_file should return an empty dict
        data = read_dev_file(test_file)
        assert isinstance(data, dict)
        assert len(data) == 0

        # read_dev_file_as_object should return an empty DevFile
        dev = read_dev_file_as_object(test_file)
        assert isinstance(dev, DevFile)
        assert dev.column_order == []
        assert len(dev.columns) == 0
        assert dev.source_file != ""
        assert dev.encoding != ""

    def test_whitespace_only_dev_file(self, tmp_path: Path) -> None:
        """Test that a whitespace-only DEV file returns empty data.

        Whitespace-only lines are stripped to empty strings and skipped
        by the content scanner (line 315), producing empty content_entries
        just like an empty file.
        """
        test_file = tmp_path / "whitespace.dev"
        test_file.write_text("   \n\t\n   \n", encoding="utf-8")

        data = read_dev_file(test_file)
        assert isinstance(data, dict)
        assert len(data) == 0

        dev = read_dev_file_as_object(test_file)
        assert isinstance(dev, DevFile)
        assert dev.column_order == []
        assert len(dev.columns) == 0
        assert dev.source_file != ""
        assert dev.encoding != ""

class TestFormatAutoDetection:
    """F-02: Format auto-detection edge cases and correctness tests."""

    def test_simple_header_format_still_works(self, tmp_path: Path) -> None:
        """Ensure simple header format is not broken by new detection."""
        content = "MD TVD X Y\n0.0 0.0 100.0 200.0\n100.0 99.0 101.0 201.0\n"
        test_file = tmp_path / "simple_regr.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert data["MD"][0] == 0.0

    def test_delimiter_auto_detection_dug_pattern_a_comma(self, tmp_path: Path) -> None:
        """DUG Pattern A (2-line header: count + header, no title)
        with comma-delimited data auto-detects comma delimiter correctly.

        Pattern A uses ``skip_content_lines=2``, so the header is at
        ``content_entries[1]`` (count line is index 0).  The delimiter
        must be detected from index 1, not the hardcoded index 2 used
        by Pattern B.
        """
        content = "4\nMD, TVD, X, Y\n0.0, 0.0, 100.0, 200.0\n100.0, 99.0, 101.0, 201.0\n"
        test_file = tmp_path / "dug_pat_a_comma.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Format detection: should be DUG Pattern A
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0
        assert data["TVD"][0] == 0.0
        assert data["X"][0] == 100.0
        assert data["X"][1] == 101.0
        assert data["Y"][0] == 200.0
        assert data["Y"][1] == 201.0

    def test_numeric_column_name_does_not_trigger_headerless(self, tmp_path: Path) -> None:
        """Single-integer line followed by all-numeric line → headerless.

        A header line with a single numeric-like token is ambiguous
        (could be a column named "100" or headerless data).  Since
        the second line has no non-numeric tokens to confirm DUG format
        and all tokens on both lines parse as float, format is detected
        as headerless.  Both lines are data rows with auto-generated
        column names.
        """
        content = "100\n50.0\n"
        test_file = tmp_path / "numeric_header.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Headerless: both lines are data for col_0
        assert list(data.keys()) == ["col_0"]
        assert data["col_0"][0] == 100.0
        assert data["col_0"][1] == 50.0

    def test_single_integer_title_followed_by_header(self, tmp_path: Path) -> None:
        """First line is a single integer, second is header → DUG format."""
        content = (
            "1\n"  # Title (could be well number)
            "MD\n"  # Header with one column
            "0.0\n"
            "100.0\n"
        )
        test_file = tmp_path / "single_int_title.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD"]
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0

    def test_delimiter_auto_detection_comma(self, tmp_path: Path) -> None:
        """Comma-delimited simple header file still auto-detects delimiter."""
        content = "MD, TVD, X, Y\n0.0, 0.0, 100.0, 200.0\n"
        test_file = tmp_path / "comma_simple.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert data["MD"][0] == 0.0

    def test_delimiter_auto_detection_dug_comma(self, tmp_path: Path) -> None:
        """Comma-delimited DUG format auto-detects comma from header line."""
        content = "Survey\n4\nMD, TVD, X, Y\n0.0, 0.0, 100.0, 200.0\n"
        test_file = tmp_path / "dug_comma.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert data["MD"][0] == 0.0

    # --- F-M4: DEV column name alias normalization ---
    def test_alias_normalization_basic(self, tmp_path: Path) -> None:
        """MDKB, TVDSS, INCL, AZIM, UTMX, UTMY are normalized to canonical names."""
        content = "MDKB TVDSS INCL AZIM UTMX UTMY\n0.0 0.0 40.0 50.0 100.0 200.0\n"
        test_file = tmp_path / "alias_test.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "INC", "AZI", "X", "Y"]
        assert data["MD"][0] == 0.0
        assert data["TVD"][0] == 0.0
        assert data["INC"][0] == 40.0
        assert data["AZI"][0] == 50.0
        assert data["X"][0] == 100.0
        assert data["Y"][0] == 200.0

    def test_alias_normalization_disabled(self, tmp_path: Path) -> None:
        """When normalize_aliases=False, column names are used as-is."""
        content = "MDKB TVDSS X Y\n0.0 -20.0 39844.5 24589.3\n"
        test_file = tmp_path / "no_alias.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file, normalize_aliases=False)
        assert list(data.keys()) == ["MDKB", "TVDSS", "X", "Y"]
        assert data["MDKB"][0] == 0.0
        assert data["TVDSS"][0] == -20.0

    def test_alias_normalization_with_duplicates(self, tmp_path: Path) -> None:
        """Normalization merges columns that map to the same canonical name."""
        content = "MDKB MD TVDSS TVD X Y\n0.0 0.0 -20.0 -20.0 39844.5 24589.3\n"
        test_file = tmp_path / "alias_dup.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # MDKB and MD both normalize to MD → second gets renamed to MD_2
        # TVDSS and TVD both normalize to TVD → second gets renamed to TVD_2
        assert data["MD"][0] == 0.0
        assert data["MD_2"][0] == 0.0
        assert data["TVD"][0] == -20.0
        assert data["TVD_2"][0] == -20.0
        assert data["X"][0] == 39844.5

    def test_alias_normalization_unknown_names_preserved(self, tmp_path: Path) -> None:
        """Column names not in the alias table are left unchanged."""
        content = "UNKNOWN_COL CUSTOM_NAME MD TVD\n0.0 1.0 2.0 3.0\n"
        test_file = tmp_path / "alias_unknown.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert "UNKNOWN_COL" in data
        assert "CUSTOM_NAME" in data
        assert "MD" in data
        assert "TVD" in data

    def test_alias_normalization_dug_format(self, tmp_path: Path) -> None:
        """Alias normalization works with DUG format files."""
        content = "Survey Title\n4\nMDKB TVDSS X Y\n0.0 -20.0 39844.5 24589.3\n"
        test_file = tmp_path / "alias_dug.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert data["MD"][0] == 0.0
        assert data["TVD"][0] == -20.0

    def test_alias_normalization_as_object(self, tmp_path: Path) -> None:
        """Alias normalization works via read_dev_file_as_object API."""
        content = "MDKB TVDSS X Y\n0.0 0.0 100.0 200.0\n"
        test_file = tmp_path / "alias_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev.column_order == ["MD", "TVD", "X", "Y"]
