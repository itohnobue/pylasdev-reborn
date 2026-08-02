"""Tests for DEV file reader."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_dev_file, read_dev_file_as_object
from pylasdev.exceptions import DEVReadError, LASEncodingError
from pylasdev.models import DevFile


class TestReadDEVFile:
    """Tests for read_dev_file function."""

    def test_read_all_dev_files(self, all_dev_files: list[Path]) -> None:
        """Test reading every DEV file in test_data/."""
        assert len(all_dev_files) > 0, "No DEV test files found"

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
        if not sample_dev.exists():
            pytest.skip(f"Required test data missing: {sample_dev}")
        data = read_dev_file(sample_dev)
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert "Y" in data

    def test_sample_dev_data_shape(self, test_data_dir: Path) -> None:
        """Test that all columns have the same length."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip(f"Required test data missing: {sample_dev}")
        data = read_dev_file(sample_dev)
        sizes = [len(arr) for arr in data.values()]
        assert len(set(sizes)) == 1, f"Column sizes differ: {sizes}"

    def test_sample_dev_md_starts_at_zero(self, test_data_dir: Path) -> None:
        """Test that MD column starts at 0."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip(f"Required test data missing: {sample_dev}")
        data = read_dev_file(sample_dev)
        assert data["MD"][0] == 0.0

    def test_sample_dev_has_multiple_rows(self, test_data_dir: Path) -> None:
        """Test that sample.dev has multiple data rows."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip(f"Required test data missing: {sample_dev}")
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
        if not sample_dev.exists():
            pytest.skip(f"Required test data missing: {sample_dev}")
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

        G-019: Empty/header-only files now raise DEVReadError instead of
        returning empty data.  See dev_reader.py:973.
        """
        content = "MD TVD X Y\n"
        test_file = tmp_path / "header_only.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

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

        Input columns: ["A", "GR", "GR"]
        Expected: ["A", "GR", "GR_2"] — first duplicate starts at _2
        (matching the LAS _deduplicate_curves convention).
        """
        content = "A GR GR\n100.0 10.0 20.0\n"
        test_file = tmp_path / "dup.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            # Should warn about the duplicate
            assert any("Duplicate DEV column name" in str(x.message) for x in w)

        assert list(data.keys()) == ["A", "GR", "GR_2"]
        np.testing.assert_array_equal(data["A"], [100.0])
        np.testing.assert_array_equal(data["GR"], [10.0])
        np.testing.assert_array_equal(data["GR_2"], [20.0])

    def test_cross_base_collision_dedup(self, tmp_path: Path) -> None:
        """Cross-base collision: original name matches prior _N suffix.

        Input columns: ["A", "A", "A_2"]
        Expected: ["A", "A_3", "A_2"] — second "A" tries to become
        "A_2" but "A_2" is a natural name (appears in input), so the
        auto-generated suffix is bumped to "A_3".  The natural "A_2"
        is preserved.
        F-033: natural names are now tracked to prevent false
        "duplicate" warnings and incorrect renaming.
        """
        content = "A A A_2\n10.0 20.0 30.0\n"
        test_file = tmp_path / "cross_base.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            dup_warnings = [x for x in w if "Duplicate DEV column name" in str(x.message)]
            assert len(dup_warnings) == 1

        assert list(data.keys()) == ["A", "A_3", "A_2"]
        np.testing.assert_array_equal(data["A"], [10.0])
        np.testing.assert_array_equal(data["A_3"], [20.0])
        np.testing.assert_array_equal(data["A_2"], [30.0])


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

    def test_memory_error_allocation_wrapped_in_dev_read_error(self, tmp_path: Path) -> None:
        """MemoryError from np.full() is wrapped in DEVReadError.

        Exercises the three np.full() try/except blocks in dev_reader.py.
        """
        from unittest import mock

        test_file = tmp_path / "oom.dev"
        test_file.write_text(
            "MD TVD X Y\n0.0 0.0 0.0 0.0\n1.0 1.0 1.0 1.0\n",
            encoding="utf-8",
        )

        with mock.patch(
            "pylasdev.dev_reader.np.full",
            side_effect=MemoryError("Cannot allocate memory"),
        ):
            with pytest.raises(DEVReadError, match="out of memory"):
                read_dev_file(test_file)

    def test_max_data_lines_guard(self, tmp_path: Path) -> None:
        """MAX_DATA_LINES guard raises DEVReadError for excess data lines.

        Exercises dev_reader.py:119-123.
        """
        from unittest import mock

        content = "MD TVD\n0.0 0.0\n100.0 99.0\n"
        test_file = tmp_path / "max_lines.dev"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_DATA_LINES to 0: any data line triggers guard.
        # Patched at data_reader source since F-DVR-01 moved imports to
        # function-local (dev_reader reads the constant at runtime).
        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 0):
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

        # Monkey-patch MAX_CURVES to 1: 3 columns > 1.
        # Patched at data_reader source since F-DVR-01 moved imports to
        # function-local (dev_reader reads the constant at runtime).
        with mock.patch("pylasdev.data_reader.MAX_CURVES", 1):
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

        # Monkey-patch MAX_TOTAL_ELEMENTS to 1: 3 cols * 1 line = 3 > 1.
        # Patched at data_reader source since F-DVR-01 moved imports to
        # function-local (dev_reader reads the constant at runtime).
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 1):
            with pytest.raises(DEVReadError, match="Total allocation"):
                read_dev_file(test_file)

    def test_index_error_handler_caught_by_safety_net(self, tmp_path: Path) -> None:
        """IndexError from _to_finite_float now propagates (handler removed).

        G-012: The dead ``try/except IndexError: pass`` wrappers were
        removed at 4 sites in dev_reader.py.  An IndexError from
        ``_to_finite_float`` is no longer silently swallowed — it
        propagates to the caller.
        """
        from unittest import mock

        content = "MD TVD\n0.0 0.0\n"
        test_file = tmp_path / "index_err.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-012: IndexError handler removed — error now propagates
        with (
            mock.patch(
                "pylasdev.dev_reader._to_finite_float",
                side_effect=IndexError("simulated"),
            ),
            pytest.raises(IndexError, match="simulated"),
        ):
            read_dev_file(test_file)

    def test_index_error_handler_read_dev_file_as_object(self, tmp_path: Path) -> None:
        """IndexError from _to_finite_float propagates via as_object path.

        G-012: Same as above but exercises the read_dev_file_as_object API.
        """
        from unittest import mock

        content = "X Y\n1.0 2.0\n"
        test_file = tmp_path / "index_err2.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-012: IndexError handler removed — error now propagates
        with (
            mock.patch(
                "pylasdev.dev_reader._to_finite_float",
                side_effect=IndexError("simulated"),
            ),
            pytest.raises(IndexError, match="simulated"),
        ):
            read_dev_file_as_object(test_file)

    # ── F-R-08: MAX_DATA_LINES at-limit acceptance ───────────────────

    def test_max_data_lines_at_limit_accepted(self, tmp_path: Path) -> None:
        """MAX_DATA_LINES set exactly to file's data line count — must accept.

        F-R-08: Changed dev_reader Pass 1 from ``>=`` to ``>`` to match
        data_reader.py and models.py convention (accept at-limit, reject above).
        """
        from unittest import mock

        # File has exactly 1 data line
        content = "MD TVD\n0.0 0.0\n"
        test_file = tmp_path / "at_limit_lines.dev"
        test_file.write_text(content, encoding="utf-8")

        # MAX_DATA_LINES=1 matches actual data line count — F-R-08 accepts at-limit
        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 1):
            data = read_dev_file(test_file)
            assert "MD" in data
            assert len(data["MD"]) == 1
            assert data["MD"][0] == 0.0

    # ── F-R-04: SPLITLINES_CHARS_RE full character class ─────────────

    def test_splitlines_chars_nul_byte_sanitized(self, tmp_path: Path) -> None:
        """NUL byte (\\x00) in DEV content is sanitized — no spurious splits.

        F-R-04: dev_reader._SPLITLINES_CHARS_RE previously covered only 8
        control chars, missing NUL and 24 others.  A NUL byte in a data
        value would cause splitlines() to produce a fake line break,
        corrupting parsed data.  The fix expands coverage to 33 chars
        matching parser.py:102 and reader.py:29.
        """
        # NUL byte embedded in data — without sanitization this would
        # produce spurious line breaks via splitlines()
        content = "MD TVD INC\n0.0\x000.0 0.0\n"
        test_file = tmp_path / "nul_byte.dev"
        test_file.write_text(content, encoding="utf-8")

        # Must not raise; must produce valid 1-row output
        data = read_dev_file(test_file)
        assert "MD" in data, "Expected MD column in output"
        assert len(data["MD"]) == 1

    def test_splitlines_chars_del_byte_sanitized(self, tmp_path: Path) -> None:
        """DEL byte (\\x7F) in DEV content is sanitized — no spurious splits.

        Part of F-R-04 character class expansion.
        """
        content = "MD TVD\n0.0\x7f0.0\n"
        test_file = tmp_path / "del_byte.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert "MD" in data, "Expected MD column in output"
        assert len(data["MD"]) == 1

    # ── F-R-07: Diagnostic counter pattern ───────────────────────────

    def test_extra_short_row_counter_summary(self, tmp_path: Path) -> None:
        """Multiple mismatched rows produce end-of-section counter summary.

        F-R-07: Replaces boolean-once warnings with counters — first
        occurrence logs full context, subsequent are counted silently,
        and a summary is warned at the end.

        Creates a DEV file with 2 extra-column rows and 2 short rows,
        verifies that all 4 rows produce warnings (not just the first
        of each type) and that the end-of-section summary fires.
        """
        import warnings

        content = (
            "MD TVD INC\n"  # 3 columns expected
            "0.0 0.0 0.0\n"  # matching row (row 1)
            "1.0 1.0 1.0 9.0\n"  # extra column (row 2)
            "2.0 2.0 2.0 9.0\n"  # extra column (row 3)
            "3.0 3.0\n"  # short row (row 4)
            "4.0 4.0\n"  # short row (row 5)
        )
        test_file = tmp_path / "counter.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Verify data was parsed correctly (discard works, NaN fill works)
        assert len(data["MD"]) == 5
        # Row 2 and 3 have extra column values discarded
        assert data["MD"][1] == 1.0
        assert data["MD"][2] == 2.0
        # Row 4 and 5 have short rows — INC filled with NaN
        assert np.isnan(data["INC"][3])
        assert np.isnan(data["INC"][4])

        # Find warning messages
        warning_texts: list[str] = [str(warn.message) for warn in w]

        summary_extra = [m for m in warning_texts if "data line(s) had more values" in m]
        summary_short = [m for m in warning_texts if "data line(s) had fewer values" in m]

        assert len(summary_extra) == 1, f"Expected 1 extra-col summary, got {warning_texts}"
        assert len(summary_short) == 1, f"Expected 1 short-row summary, got {warning_texts}"

        # The summary should show count 2 (rows 2 and 3 for extra, rows 4 and 5 for short)
        assert any("2 data line(s)" in m for m in summary_extra), (
            f"Extra-col summary should mention '2 data line(s)', got: {summary_extra}"
        )
        assert any("2 data line(s)" in m for m in summary_short), (
            f"Short-row summary should mention '2 data line(s)', got: {summary_short}"
        )


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
        Expected: ["A", "A_3", "A_2_2", "A_2"] — second "A" tries to
        become "A_2" but "A_2" is a natural name (the fourth column),
        so the while-loop bumps the suffix to "A_3".  The natural
        "A_2_2" and "A_2" are preserved.
        F-033: natural names are now tracked — auto-generated suffixes
        cannot collide with names that appear in the input list,
        preventing false "duplicate" warnings and misnaming.
        """
        content = "A A A_2_2 A_2\n10.0 20.0 30.0 40.0\n"
        test_file = tmp_path / "cross_while.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)
            dup_warnings = [x for x in w if "Duplicate DEV column name" in str(x.message)]
            assert len(dup_warnings) == 1

        assert list(data.keys()) == ["A", "A_3", "A_2_2", "A_2"]
        np.testing.assert_array_equal(data["A"], [10.0])
        np.testing.assert_array_equal(data["A_3"], [20.0])
        np.testing.assert_array_equal(data["A_2_2"], [30.0])
        np.testing.assert_array_equal(data["A_2"], [40.0])


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
        """DUG format with header but no data rows.

        G-019: No data lines — now raises DEVReadError instead of
        returning empty columns.
        """
        content = "Survey\n4\nMD TVD X Y\n"
        test_file = tmp_path / "dug_no_data.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

    def test_dug_format_falls_to_simple_when_header_is_numeric(self, tmp_path: Path) -> None:
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
        assert list(data.keys()) == ["SURVEY"]
        assert len(data["SURVEY"]) == 2  # "3" and "0.0 0.0 0.0"

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

    def test_headerless_comma_scientific_notation(self, tmp_path: Path) -> None:
        """F-PR-03: Comma-delimited headerless DEV with scientific notation.

        Regression test: scientific-notation characters (e/E/d/D) match
        str.isalpha(), causing has_alpha=True false positive. Without the
        fix, the comma-delimited path falls through to whitespace-based
        detection and the first data row is consumed as column names.
        """
        content = "1e5,2e6,3e7\n4.0,5.0,6.0\n7.0,8.0,9.0\n"
        test_file = tmp_path / "noheader_comma_sci.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        assert len(data["col_0"]) == 3
        # First row: 1e5=100000, 2e6=2000000, 3e7=30000000
        assert data["col_0"][0] == 100000.0
        assert data["col_1"][0] == 2000000.0
        assert data["col_2"][0] == 30000000.0
        # Second row
        assert data["col_0"][1] == 4.0
        assert data["col_1"][1] == 5.0
        assert data["col_2"][1] == 6.0
        # Third row
        assert data["col_0"][2] == 7.0
        assert data["col_1"][2] == 8.0
        assert data["col_2"][2] == 9.0

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
        """Test that an empty DEV file raises DEVReadError.

        G-019: Empty files (zero data lines) now raise DEVReadError
        instead of returning empty data.  See dev_reader.py:973.
        """
        test_file = tmp_path / "empty.dev"
        test_file.write_text("", encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file_as_object(test_file)

    def test_whitespace_only_dev_file(self, tmp_path: Path) -> None:
        """Test that a whitespace-only DEV file raises DEVReadError.

        G-019: Whitespace-only files (zero data lines after stripping)
        now raise DEVReadError instead of returning empty data.
        """
        test_file = tmp_path / "whitespace.dev"
        test_file.write_text("   \n\t\n   \n", encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file_as_object(test_file)


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

    def test_numeric_column_name_triggers_headerless(self, tmp_path: Path) -> None:
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

    # F-093: Petrel well-header regression tests
    def test_petrel_well_header_detected_and_skipped(self, tmp_path: Path) -> None:
        """F-093 HIGH: Petrel well-header line detected and data parsed correctly.

        Petrel exports DEV files with a well-header line preceding column
        names, e.g. "WELL-1 1000.0 2000.0 50.0".  Without F-093 fix, this
        line is consumed as column names (WELL-1, 1000.0, 2000.0, 50.0) and
        the real header "MD INC AZI TVD" becomes NaN data.
        """
        import warnings

        content = (
            "WELL-1 1000.0 2000.0 50.0\n"  # Petrel well-header
            "MD INC AZI TVD\n"  # real column names
            "0.0 0.0 90.0 0.0\n"  # data row 1
            "100.0 0.0 90.0 -100.0\n"  # data row 2
            "200.0 0.0 90.0 -200.0\n"  # data row 3
        )
        test_file = tmp_path / "petrel.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Verify Petrel well-header warning was emitted
        petrel_warnings = [str(x.message) for x in w if "Petrel well-header" in str(x.message)]
        assert len(petrel_warnings) == 1, (
            f"Expected Petrel well-header warning, got: {[str(x.message) for x in w]}"
        )

        # Verify correct column names (NOT "WELL-1", "1000.0", etc.)
        assert list(data.keys()) == ["MD", "INC", "AZI", "TVD"], (
            f"Expected ['MD','INC','AZI','TVD'], got {list(data.keys())}"
        )

        # Verify actual data (NOT NaN from consuming well-header as data)
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0
        assert data["MD"][2] == 200.0
        assert data["TVD"][0] == 0.0
        assert data["TVD"][1] == -100.0
        assert data["TVD"][2] == -200.0
        assert data["INC"][0] == 0.0
        assert data["AZI"][0] == 90.0

    def test_petrel_well_header_all_numeric_line_alone_not_misdetected(
        self, tmp_path: Path
    ) -> None:
        """F-093: A single numeric-token header line alone is not misdetected.

        When a line like "100 200 300" (all float) appears as the first
        content line, it should remain headerless (or simple with numeric
        column names) — NOT trigger Petrel detection.
        """
        content = "100 200 300\n1.0 2.0 3.0\n4.0 5.0 6.0\n"
        test_file = tmp_path / "numeric_header.dev"
        test_file.write_text(content, encoding="utf-8")

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = read_dev_file(test_file)

        # Should NOT have a Petrel warning
        petrel_warnings = [str(x.message) for x in w if "Petrel well-header" in str(x.message)]
        assert len(petrel_warnings) == 0, (
            "Should NOT detect Petrel well-header for all-numeric line"
        )


class TestExplicitDelimiterParameter:
    """F-ITER2-T2-M05: Test explicit delimiter parameter on read_dev_file.

    The ``delimiter`` parameter lets callers override auto-detection
    when the file uses a delimiter that auto-detection cannot infer
    (e.g., pipe-separated, semicolon).  It has zero test coverage
    across 9 existing read_dev_file_as_object test calls.
    """

    def test_explicit_space_delimiter(self, tmp_path: Path) -> None:
        """Explicit space delimiter on simple format file."""
        content = "MD TVD X\n0.0 0.0 100.0\n100.0 99.0 101.0\n"
        test_file = tmp_path / "space_delim.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file, delimiter=" ")
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0

    def test_explicit_comma_delimiter(self, tmp_path: Path) -> None:
        """Explicit comma delimiter on comma-separated file."""
        content = "MD,TVD,X\n0.0,0.0,100.0\n100.0,99.0,101.0\n"
        test_file = tmp_path / "comma_delim.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file, delimiter=",")
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0

    def test_explicit_delimiter_as_object(self, tmp_path: Path) -> None:
        """Explicit delimiter via read_dev_file_as_object API."""
        content = "MD,TVD,X\n0.0,0.0,100.0\n100.0,99.0,101.0\n"
        test_file = tmp_path / "delim_obj.dev"
        test_file.write_text(content, encoding="utf-8")
        dev = read_dev_file_as_object(test_file, delimiter=",")
        assert dev.column_order == ["MD", "TVD", "X"]
        assert len(dev.columns["MD"]) == 2
        assert dev.columns["MD"][0] == 0.0

    def test_explicit_delimiter_overrides_auto_detection(self, tmp_path: Path) -> None:
        """Explicit delimiter overrides auto-detection when both possible."""
        # File uses space-separated format but we force comma delimiter.
        # The first line "MD TVD X" has no commas, so comma split
        # produces a single token — one column named "MD TVD X".
        # The data line "0.0 0.0 100.0" also has no commas, so the
        # single token cannot be parsed as float → NaN.
        content = "MD TVD X\n0.0 0.0 100.0\n"
        test_file = tmp_path / "override_delim.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file, delimiter=",")
        # Comma-delimited: entire header is one column name
        assert "MD TVD X" in data
        # Data token "0.0 0.0 100.0" is not parseable as float → NaN
        assert np.isnan(data["MD TVD X"][0])

    # --- F-08: 2-line DUG count-match heuristic no longer fires ---
    def test_two_line_file_not_misdetected_as_dug(self, tmp_path: Path) -> None:
        """F-08: A 2-line file with integer count + all-numeric second line
        must NOT be detected as DUG via the count-match heuristic.

        Before the fix, "4\\n100.0 200.0 300.0 400.0\\n" triggered:
        col_count (4) == len(second_tokens) (4) → ("dug", 2) →
        skip_content_lines=2 → zero data lines → total data loss.

        After the fix, the count-match heuristic requires >= 3 content
        entries, so the 2-line file correctly falls through to headerless
        detection and produces data instead of an empty result.  The
        column count is 1 (first row "4" has one token), and the second
        row's extra tokens are discarded with a warning.
        """
        content = "4\n100.0 200.0 300.0 400.0\n"
        test_file = tmp_path / "f08_twoline.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Critical: should NOT be empty (was the bug — total data loss)
        assert len(data) > 0, "2-line file should NOT produce zero columns"
        # Detected as headerless: first row "4" → 1 column, col_0
        assert "col_0" in data
        assert len(data["col_0"]) == 2, "Should have 2 data rows"
        assert data["col_0"][0] == 4.0
        # Second row's extra values discarded (only 1 column declared)
        assert data["col_0"][1] == 100.0

    # --- F-10: F-DV01 count-mismatch fallback test coverage ---
    def test_dv01_count_mismatch_fallback(self, tmp_path: Path) -> None:
        """F-10/M-20: F-DV01 count-mismatch fallback (dev_reader.py).

        Trigger: first line is a single integer (col_count), second line
        has all-float tokens with a DIFFERENT count than col_count, and
        3+ content entries exist.

        M-20 fix: the count-mismatch fallback is now data-preserving.
        The count line is skipped and ALL data rows — including the first
        — are kept as data with generated col_N names.  The pre-fix
        behavior consumed the first data row as numeric column names,
        losing one station and fabricating numeric headers.
        """
        # col_count=3 but second line has 4 float tokens (mismatch)
        # 3+ content entries → fallback activates
        content = "3\n1.0 2.0 3.0 4.0\n100.0 200.0 300.0 400.0\n500.0 600.0 700.0 800.0\n"
        test_file = tmp_path / "dv01_fallback.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Data-preserving contract: no fabricated numeric column names —
        # the first data row is preserved as data under generated col_N
        # names (M-20).
        assert "1.0" not in data
        assert "col_0" in data
        assert "col_1" in data
        assert "col_2" in data
        assert "col_3" in data
        # col_0 = [first-row 1.0, 100.0, 500.0] — all 3 rows preserved
        assert len(data["col_0"]) == 3
        assert data["col_0"][0] == 1.0
        assert data["col_0"][1] == 100.0
        assert data["col_0"][2] == 500.0
        assert data["col_1"][0] == 2.0
        assert data["col_2"][0] == 3.0
        assert data["col_3"][0] == 4.0

    # --- F-10 variant: F-DV01 with 2.0e1-style float tokens ---
    def test_dv01_fallback_with_scientific_notation_headers(self, tmp_path: Path) -> None:
        """F-10/M-20: F-DV01 fallback with scientific-notation float tokens.

        M-20 fix: the count-mismatch fallback is data-preserving.  The
        first data row's scientific-notation tokens (1.0e2, 2.0E-1,
        3.14159) are parsed as DATA values, not fabricated numeric column
        names — no row is lost, no numeric header is created.
        """
        content = "2\n1.0e2 2.0E-1 3.14159\n100.0 200.0 300.0\n400.0 500.0 600.0\n"
        test_file = tmp_path / "dv01_sci.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # No fabricated numeric column names — first row is data.
        assert "1.0E2" not in data
        assert "col_0" in data
        assert "col_1" in data
        assert "col_2" in data
        # col_0 = [1.0e2, 100.0, 400.0] — all 3 rows preserved
        assert len(data["col_0"]) == 3
        assert data["col_0"][0] == 100.0  # 1.0e2 parsed as data
        assert data["col_0"][1] == 100.0
        assert data["col_0"][2] == 400.0
        assert data["col_1"][0] == 0.2  # 2.0E-1 parsed as data
        assert data["col_1"][1] == 200.0
        assert data["col_2"][0] == 3.14159


class TestEmptyColumnNames:
    """F2-11: Empty column names from trailing delimiters are rejected.

    Before the fix, "MD,TVD,".split(",") → ["MD","TVD",""] and the
    empty string passed through normalization, dedup, and allocation,
    creating dev.columns[""] = array.  After the fix, empty strings
    are filtered out so they are silently dropped.
    """

    def test_trailing_comma_no_data(self, tmp_path: Path) -> None:
        """Trailing comma in header with no data rows raises DEVReadError.

        G-019: No data lines — now raises DEVReadError instead of
        returning empty columns.
        """
        content = "MD,TVD,\n"
        test_file = tmp_path / "f2_11_trail_no_data.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

    def test_trailing_comma_with_data(self, tmp_path: Path) -> None:
        """Trailing comma in header with data rows works correctly."""
        content = "MD,TVD,\n0.0,0.0,\n100.0,99.0,\n200.0,198.0,\n"
        test_file = tmp_path / "f2_11_trail_data.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Only 2 valid columns
        assert list(data.keys()) == ["MD", "TVD"]
        assert len(data["MD"]) == 3
        assert data["MD"][0] == 0.0
        assert data["MD"][2] == 200.0
        assert data["TVD"][2] == 198.0

    def test_trailing_comma_dug_format(self, tmp_path: Path) -> None:
        """Trailing comma in DUG format header is filtered."""
        content = "Survey\n3\nMD,TVD,X,\n0.0,0.0,100.0,\n100.0,99.0,101.0,\n"
        test_file = tmp_path / "f2_11_dug_trail.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0

    def test_trailing_space_in_comma_header(self, tmp_path: Path) -> None:
        """Trailing space after a comma is stripped (v.strip() handles it)."""
        content = "MD, TVD, \n0.0, 0.0, \n"
        test_file = tmp_path / "f2_11_space_trail.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD"]
        assert len(data["MD"]) == 1

    def test_no_trailing_delimiter_still_works(self, tmp_path: Path) -> None:
        """Normal comma-delimited files without trailing delimiter still work."""
        content = "MD,TVD,X\n0.0,0.0,100.0\n100.0,99.0,101.0\n"
        test_file = tmp_path / "f2_11_normal.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0


class TestColumnsKeywordFormat:
    """M59: Tests for *COLUMNS keyword format (Petra/CPS variant).

    The *COLUMNS keyword format uses a header line starting with the
    literal keyword *COLUMNS followed by *-prefixed column names.
    Functions at dev_reader.py:195-213,272,781-783,844-846.
    """

    def test_columns_simple_format_basic(self, tmp_path: Path) -> None:
        """Parse a *COLUMNS header in simple format with data rows."""
        content = "*COLUMNS *MD *TVD *X *Y\n0.0 0.0 100.0 200.0\n100.0 99.0 101.0 201.0\n"
        test_file = tmp_path / "columns_simple.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0
        assert data["TVD"][0] == 0.0
        assert data["X"][0] == 100.0
        assert data["Y"][0] == 200.0

    def test_columns_simple_format_single_row(self, tmp_path: Path) -> None:
        """Parse *COLUMNS header with a single data row."""
        content = "*COLUMNS *MD *TVD *GR\n100.0 50.0 75.0\n"
        test_file = tmp_path / "columns_single.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "GR"]
        assert len(data["MD"]) == 1
        assert data["MD"][0] == 100.0

    def test_columns_lowercase_keyword(self, tmp_path: Path) -> None:
        """*columns (lowercase) still triggers the keyword format path.

        F-034: _normalize_dev_column now uppercases column names before
        alias lookup, so lowercase *columns header names like *md, *tvd,
        *x are normalized to canonical uppercase (MD, TVD, X).
        Previously they were returned in lowercase, which skipped
        validation and produced inconsistent column naming.
        """
        content = "*columns *md *tvd *x\n0.0 0.0 100.0\n"
        test_file = tmp_path / "columns_lower.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert data["MD"][0] == 0.0

    def test_columns_mixed_case_keyword(self, tmp_path: Path) -> None:
        """*Columns (mixed case) triggers keyword format via .upper() check."""
        content = "*Columns *MD *TVD *GR\n100.0 50.0 75.0\n"
        test_file = tmp_path / "columns_mixed.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "GR"]
        assert data["GR"][0] == 75.0

    def test_columns_header_only_no_data(self, tmp_path: Path) -> None:
        """*COLUMNS header with no data rows raises DEVReadError.

        G-019: No data lines — now raises DEVReadError instead of
        returning empty columns.
        """
        content = "*COLUMNS *MD *TVD *X *Y\n"
        test_file = tmp_path / "columns_no_data.dev"
        test_file.write_text(content, encoding="utf-8")

        # G-019: No data lines — must raise DEVReadError (F2-20 fix)
        with pytest.raises(DEVReadError, match="No data lines found"):
            read_dev_file(test_file)

    def test_columns_with_comments(self, tmp_path: Path) -> None:
        """*COLUMNS format with comment lines interspersed."""
        content = (
            "# Well-X survey\n"
            "*COLUMNS *MD *TVD *X\n"
            "# Data starts here\n"
            "0.0 0.0 100.0\n"
            "100.0 99.0 101.0\n"
        )
        test_file = tmp_path / "columns_comments.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0

    def test_columns_as_object(self, tmp_path: Path) -> None:
        """*COLUMNS format via read_dev_file_as_object API."""
        content = "*COLUMNS *MD *TVD *X\n0.0 0.0 100.0\n100.0 99.0 101.0\n"
        test_file = tmp_path / "columns_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev.column_order == ["MD", "TVD", "X"]
        assert len(dev.columns["MD"]) == 2
        np.testing.assert_array_equal(dev.columns["MD"], [0.0, 100.0])


class TestEmptyDelimiterGuard:
    """M60: Tests for empty delimiter guard at dev_reader.py:674-678.

    The guard rejects empty string delimiters that would cause
    str.split("") to raise ValueError.
    """

    def test_empty_delimiter_raises_error(self, tmp_path: Path) -> None:
        """Passing delimiter="" raises DEVReadError."""
        content = "MD TVD X\n0.0 0.0 100.0\n"
        test_file = tmp_path / "empty_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(DEVReadError, match="Delimiter must be a non-empty string"):
            read_dev_file(test_file, delimiter="")

    def test_empty_delimiter_as_object(self, tmp_path: Path) -> None:
        """Passing delimiter="" to read_dev_file_as_object raises DEVReadError."""
        content = "MD TVD\n0.0 0.0\n"
        test_file = tmp_path / "empty_delim_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(DEVReadError, match="Delimiter must be a non-empty string"):
            read_dev_file_as_object(test_file, delimiter="")

    def test_multi_char_delimiter_not_rejected_by_guard(self, tmp_path: Path) -> None:
        """Multi-char delimiter (e.g. "::") passes the empty guard.

        The guard at 674 only checks truthiness — not single-char validation.
        csv.reader(delimiter="::") raises TypeError which may or may not
        be caught. This test documents current behavior with a valid
        single-char delimiter.
        """
        content = "MD,TVD,X\n0.0,0.0,100.0\n"
        test_file = tmp_path / "good_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        # Valid single-char delimiter should work
        data = read_dev_file(test_file, delimiter=",")
        assert list(data.keys()) == ["MD", "TVD", "X"]


class TestMultiCharDelimiterGuard:
    """F-H-007: Tests for multi-character delimiter guard at dev_reader.py:938-947.

    Python's csv.reader raises TypeError on multi-character delimiters at
    iteration time, which is not caught by the csv.Error handler.  The guard
    explicitly validates len(delimiter) == 1 before reaching csv.reader.
    """

    def test_multi_char_delimiter_raises_error(self, tmp_path: Path) -> None:
        """Passing delimiter='::' raises DEVReadError."""
        content = "MD::TVD::X\n0.0::0.0::100.0\n"
        test_file = tmp_path / "multi_char_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(
            DEVReadError,
            match="Delimiter must be a single character",
        ):
            read_dev_file(test_file, delimiter="::")

    def test_multi_char_delimiter_as_object(self, tmp_path: Path) -> None:
        """Passing delimiter='::' to read_dev_file_as_object raises DEVReadError."""
        content = "MD::TVD\n0.0::0.0\n"
        test_file = tmp_path / "multi_char_delim_obj.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(
            DEVReadError,
            match="Delimiter must be a single character",
        ):
            read_dev_file_as_object(test_file, delimiter="::")

    def test_single_char_delimiter_still_works(self, tmp_path: Path) -> None:
        """Single-char delimiter (',') still works after adding the guard."""
        content = "MD,TVD,X\n0.0,0.0,100.0\n"
        test_file = tmp_path / "single_char_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file, delimiter=",")
        assert list(data.keys()) == ["MD", "TVD", "X"]


class TestDelimiterAutoCorrectionHeader:
    """F-01: Tests for delimiter auto-correction preserving header column names.

    When a DEV file has a comma-delimited header but space-delimited data,
    the auto-correction at dev_reader.py switches the delimiter from comma
    to space.  Before the F-01 fix, Pass 2 would re-split the comma header
    with the space delimiter, collapsing multi-column headers like
    "MD,TVD,INC" into a single bogus column name.  After the fix, the
    original comma-split header names are cached and used in Pass 2.
    """

    def test_auto_correction_simple_format(self, tmp_path: Path) -> None:
        """Comma-delimited header + space-delimited data: columns correct."""
        content = "MD,TVD,INC\n1000.0 2000.0 3000.0\n1100.0 2100.0 3100.0\n"
        test_file = tmp_path / "mix_comma_header_space_data.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Must have 3 columns, not 1 bogus column "MD,TVD,INC"
        assert list(data.keys()) == ["MD", "TVD", "INC"], (
            f"Expected ['MD','TVD','INC'], got {list(data.keys())}"
        )
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 1000.0
        assert data["MD"][1] == 1100.0
        assert data["TVD"][0] == 2000.0
        assert data["TVD"][1] == 2100.0
        assert data["INC"][0] == 3000.0
        assert data["INC"][1] == 3100.0

    def test_auto_correction_dug_format(self, tmp_path: Path) -> None:
        """DUG format: comma-header + space-data — columns correct."""
        content = "Well-Survey\n3\nMD,TVD,INC\n1000.0 2000.0 3000.0\n1100.0 2100.0 3100.0\n"
        test_file = tmp_path / "dug_mix_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "INC"], (
            f"Expected ['MD','TVD','INC'], got {list(data.keys())}"
        )
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 1000.0
        assert data["MD"][1] == 1100.0

    def test_auto_correction_disk_as_object(self, tmp_path: Path) -> None:
        """Read via read_dev_file_as_object API — columns correct."""
        content = "X,Y,Z\n100.0 200.0 300.0\n150.0 250.0 350.0\n"
        test_file = tmp_path / "obj_mix_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev.column_order == ["X", "Y", "Z"], (
            f"Expected ['X','Y','Z'], got {dev.column_order}"
        )
        assert len(dev.columns["X"]) == 2
        assert dev.columns["X"][0] == 100.0

    def test_auto_correction_with_columns_keyword(self, tmp_path: Path) -> None:
        """*COLUMNS header with space data — raises DEVReadError.

        The *COLUMNS keyword adds an extra token to the header, making
        the column counts mismatch (4 tokens in header, 3 in data).
        Auto-correction correctly rejects this as unresolvable.
        """
        content = "*COLUMNS,*MD,*TVD,*GR\n100.0 50.0 75.0\n"
        test_file = tmp_path / "columns_mix_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        # Auto-correction: header has 4 comma tokens, data has 3 space
        # tokens.  _data_alt_cols (3) != _hdr_cols (4) → DEVReadError.
        with pytest.raises(DEVReadError, match="Delimiter mismatch"):
            read_dev_file(test_file)

    def test_auto_correction_with_trailing_comma(self, tmp_path: Path) -> None:
        """Comma-header with trailing comma + space data — correct columns."""
        content = "MD,TVD,\n1000.0 2000.0\n1100.0 2100.0\n"
        test_file = tmp_path / "trail_mix_delim.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Trailing comma stripped by the filter logic
        assert list(data.keys()) == ["MD", "TVD"], f"Expected ['MD','TVD'], got {list(data.keys())}"
        assert len(data["MD"]) == 2

    def test_auto_correction_with_commas_and_spaces(self, tmp_path: Path) -> None:
        """Comma-header with spaces after commas + space data — correct."""
        content = "MD, TVD, INC, AZI\n1000.0 2000.0 3000.0 4000.0\n"
        test_file = tmp_path / "space_comma_mix.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "INC", "AZI"], (
            f"Expected ['MD','TVD','INC','AZI'], got {list(data.keys())}"
        )
        assert data["MD"][0] == 1000.0
        assert data["AZI"][0] == 4000.0

    def test_pure_space_delimiter_unaffected(self, tmp_path: Path) -> None:
        """Pure space-delimited file unaffected by the fix (no regression)."""
        content = "MD TVD X\n0.0 0.0 100.0\n100.0 99.0 101.0\n"
        test_file = tmp_path / "pure_space.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0

    def test_pure_comma_delimiter_unaffected(self, tmp_path: Path) -> None:
        """Pure comma-delimited file unaffected by the fix (no regression)."""
        content = "MD,TVD,X\n0.0,0.0,100.0\n100.0,99.0,101.0\n"
        test_file = tmp_path / "pure_comma.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2


class TestValidateDevData:
    """Tests for _validate_dev_data() — F-008/F-201 validation rules.

    Covers NaN-density, MD monotonicity, repeated stations, azimuth range,
    and edge cases (no MD column, single-row, all-NaN, no azimuth column).
    """

    # ── edge cases (no warning expected) ─────────────────────────────

    def test_no_md_column_no_warning(self) -> None:
        """No MD column → warns about skipped validation but does not raise."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["TVD"] = np.array([0.0, 100.0])
        dev.columns["X"] = np.array([100.0, 101.0])
        with pytest.warns(UserWarning, match="MD column not found"):
            _validate_dev_data(dev)

    def test_single_row_non_nan_no_warning(self) -> None:
        """Single-row file with valid MD → returns early, no warning."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_dev_data(dev)

    def test_single_row_nan_warns(self) -> None:
        """Single-row file with NaN MD → warns about NaN density."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([np.nan])
        with pytest.warns(UserWarning, match="1/1 NaN"):
            _validate_dev_data(dev)

    # ── NaN density (> 50%) ─────────────────────────────────────────

    def test_nan_density_above_50_percent_warns(self) -> None:
        """>50% NaN in MD column warns about delimiter mismatch."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        # 4 NaN out of 5 total = 80% → triggers warning
        dev.columns["MD"] = np.array([100.0, np.nan, np.nan, np.nan, np.nan])
        with pytest.warns(UserWarning, match="NaN values.*delimiter mismatch"):
            _validate_dev_data(dev)

    # ── MD monotonicity ─────────────────────────────────────────────

    def test_md_non_monotonic_warns(self) -> None:
        """MD going backwards (100→200→150) triggers non-monotonic warning."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 150.0, 300.0])
        with pytest.warns(UserWarning, match="not monotonically increasing"):
            _validate_dev_data(dev)

    # ── repeated station MD ─────────────────────────────────────────

    def test_repeated_station_md_warns(self) -> None:
        """Repeated MD values (200.0 appears twice) trigger duplicates warning."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 200.0, 300.0])
        with pytest.warns(UserWarning, match="repeated MD station"):
            _validate_dev_data(dev)

    # ── azimuth out of range ────────────────────────────────────────

    def test_azimuth_out_of_range_warns(self) -> None:
        """Azimuth outside [0, 360] (370, -5) triggers azimuth warning."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        dev.columns["AZIM"] = np.array([10.0, 370.0, -5.0])
        with pytest.warns(UserWarning, match="Azimuth.*outside.*0, 360"):
            _validate_dev_data(dev)

    # ── clean data (no warnings) ────────────────────────────────────

    def test_clean_data_no_warnings(self) -> None:
        """Clean monotonic MD with valid azimuth produces no warnings."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0, 400.0])
        dev.columns["AZIM"] = np.array([10.0, 90.0, 180.0, 350.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_dev_data(dev)

    # ── all-NaN MD ──────────────────────────────────────────────────

    def test_all_nan_md_warns_and_returns_early(self) -> None:
        """All-NaN MD → NaN-density warning fires, then returns early.

        No monotonicity/azimuth checks follow because there are no
        finite values to check.

        F-042: MD is now included in the NaN/Inf validation loop,
        so two warnings fire: NaN/Inf for MD and the >50% NaN density
        warning.  Both are expected; no subsequent checks follow.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([np.nan, np.nan, np.nan])
        dev.columns["AZIM"] = np.array([10.0, 20.0, 30.0])
        # Two warnings expected: NaN/Inf loop + NaN density check
        with pytest.warns(UserWarning) as w:
            _validate_dev_data(dev)
        # At least the NaN density warning fires; NaN/Inf warning
        # also fires for MD since F-042 removed the MD skip.
        assert len(w) >= 1
        assert any("NaN values" in str(msg.message) for msg in w)

    # ── no azimuth column ───────────────────────────────────────────

    def test_no_azimuth_column_no_warning(self) -> None:
        """No azimuth column → no azimuth range check (graceful skip)."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_dev_data(dev)


class TestF035FailureCounterRegression:
    """F-035 regression: _failure_counter passed to _to_finite_float.

    Before F-035, four non-MD column sites in dev_reader.py called
    ``_to_finite_float(token, ...)`` without passing a
    ``_failure_counter`` list.  When a non-numeric token hit the
    ``float(token)`` branch and failed, it was logged at errors=10+
    but the counter was None → silent NaN absorption with no
    end-of-section diagnostic summary for the corrupted column.
    After F-035, all call sites share a single counter list so the
    end-of-section report covers all columns.
    """

    def test_non_numeric_all_columns_counted(self, tmp_path: Path) -> None:
        """Non-numeric tokens in non-MD columns are counted in diagnostics."""
        # All columns have non-numeric data — this exercises the
        # _failure_counter path at all sites (TVD, X, Y, etc.)
        content = "MD TVD X Y\n100.0 BAD BAD BAD\n200.0 BAD BAD BAD\n"
        test_file = tmp_path / "f035_all_bad.dev"
        test_file.write_text(content, encoding="utf-8")

        # Should not crash; non-numeric values become NaN
        data = read_dev_file(test_file)
        assert np.isnan(data["TVD"][0])
        assert np.isnan(data["TVD"][1])
        assert np.isnan(data["X"][0])
        assert np.isnan(data["Y"][0])
        # MD should parse correctly
        assert data["MD"][0] == 100.0
        assert data["MD"][1] == 200.0


# ============================================================
# Production Check Regression Tests
# ============================================================


class TestProductionCheckDevReaderFixes:
    """Regression tests for production check fixes in dev_reader.py."""

    # --- F-023 (MEDIUM): _SENTINELS incomplete ---

    def test_inf_sentinel_recognized_as_headerless(self, tmp_path: Path) -> None:
        """F-023: Comma-delimited row with 'inf' is headerless, not a header.

        Before the fix, a row like "1.0, inf, 2.0" was treated as a header
        because 'inf' is not a float token (rejected by _is_float_token)
        and was not in _SENTINELS. Now it's recognized as a sentinel.
        """
        content = "1.0, inf, 2.0\n100.0, 50.0, 75.0\n"
        test_file = tmp_path / "inf_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Should be headerless (3 columns)
        assert len(data) == 3
        assert "col_0" in data
        assert data["col_0"][0] == 1.0
        assert np.isnan(data["col_1"][0])  # 'inf' → NaN
        assert data["col_2"][0] == 2.0

    def test_minus_inf_sentinel_recognized(self, tmp_path: Path) -> None:
        """F-023: '-inf' is recognized as a sentinel."""
        content = "-inf, 2.0, 3.0\n50.0, 75.0, 100.0\n"
        test_file = tmp_path / "minus_inf_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert "col_0" in data

    # --- F-021 (MEDIUM): Whitespace path sentinel detection ---

    def test_whitespace_na_sentinel_headerless(self, tmp_path: Path) -> None:
        """F-021: Whitespace line with 'na' sentinel is headerless.

        Before the fix, whitespace-delimited "100.0 na 200.0" on the
        first line fell through to "simple" format (treated as header).
        Now it's detected as headerless data with a sentinel value.
        """
        content = "100.0 na 200.0\n50.0 75.0 100.0\n"
        test_file = tmp_path / "ws_na_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Should produce 3 columns (headerless), not try to parse line as header
        assert len(data) == 3

    def test_whitespace_null_sentinel_headerless(self, tmp_path: Path) -> None:
        """F-021: Whitespace line with 'null' sentinel is headerless."""
        content = "100.0 null 200.0\n50.0 75.0 100.0\n"
        test_file = tmp_path / "ws_null_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert len(data) == 3

    # --- F-213 (MEDIUM): All-zero first row heuristic ---

    def test_all_zero_first_row_headerless(self, tmp_path: Path) -> None:
        """F-213: All-zero first row is headerless data, not integer headers.

        Column names like "0 0 0 0" are nonsensical; an all-zero row
        is a common first data row in deviation surveys (MD=0 at surface).
        """
        content = "0.0 0.0 0.0 0.0\n100.0 99.0 100.0 200.0\n200.0 198.0 101.0 201.0\n"
        test_file = tmp_path / "all_zero_headerless.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Should be headerless (not simple with column names "0", "0", "0", "0")
        assert "col_0" in data
        assert "col_1" in data
        assert len(data["col_0"]) == 3
        assert data["col_0"][0] == 0.0
        assert data["col_0"][1] == 100.0
        assert data["col_0"][2] == 200.0

    # --- F-025 (MEDIUM): Deduplicated azi/inc columns validation ---

    def test_deduplicated_azimuth_out_of_range_raises_warning(self) -> None:
        """F-025: _N-suffixed azimuth survivor still gets range validation.

        When normalize_aliases=True deduplicates columns (AZIM + AZ →
        AZI, AZI_2), the _2-suffixed survivor must still be validated
        for [0, 360] range.
        """
        from pylasdev.dev_reader import _validate_dev_data
        from pylasdev.models import DevFile

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        # AZI_2 is a deduplicated survivor that should be validated
        dev.columns["AZI_2"] = np.array([10.0, 400.0, -50.0])
        dev.column_order = ["MD", "AZI_2"]

        with pytest.warns(UserWarning, match="Azimuth.*outside.*0, 360"):
            _validate_dev_data(dev)

    def test_deduplicated_azimuth_valid_no_warning(self) -> None:
        """F-025: Valid _N-suffixed azimuth produces no warning."""
        from pylasdev.dev_reader import _validate_dev_data
        from pylasdev.models import DevFile

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        dev.columns["AZI_2"] = np.array([10.0, 90.0, 350.0])
        dev.column_order = ["MD", "AZI_2"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_dev_data(dev)  # Should not warn — all in [0, 360]

    # --- M-33 (MEDIUM): Tab token strip ---

    def test_tab_delimited_tokens_whitespace_stripped(self, tmp_path: Path) -> None:
        """M-33: Tab-delimited DEV values with whitespace are stripped.

        Before the fix, tab-delimited values with leading/trailing
        whitespace (e.g., ``" 100.0\t"``) could retain spaces/tabs
        in parsed values.  The fix ensures ``v.strip()`` is applied
        to every token after csv.reader splits the line.
        """
        from pylasdev.dev_reader import _split_delimited_line

        # Values with leading/trailing whitespace between tabs
        line = "  MD  \t\t  100.0  \t  200.0  "
        tokens = _split_delimited_line(line, "\t")
        assert tokens == ["MD", "", "100.0", "200.0"], f"Expected stripped tokens, got {tokens!r}"

    def test_tab_delimited_dev_file_whitespace_stripped(self, tmp_path: Path) -> None:
        """M-33: Full DEV file read with tab-delimited whitespace values.

        End-to-end: tab-delimited .dev file with values that have
        trailing whitespace should parse correctly with stripped values.
        """
        content = "MD\tTVD\n 100.0 \t 99.0 \n 200.0 \t 198.0 \n"
        test_file = tmp_path / "tab_ws.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert "MD" in data, f"Expected 'MD' column, got keys: {list(data.keys())}"
        assert data["MD"][0] == 100.0, f"Unexpected MD[0]: {data['MD'][0]!r}"
        assert data["MD"][1] == 200.0, f"Unexpected MD[1]: {data['MD'][1]!r}"
        assert data["TVD"][0] == 99.0, f"Unexpected TVD[0]: {data['TVD'][0]!r}"
        assert data["TVD"][1] == 198.0, f"Unexpected TVD[1]: {data['TVD'][1]!r}"

    # === F-044: -nan/+nan sentinel recognition ===

    def test_plus_nan_sentinel_headerless_comma(self, tmp_path: Path) -> None:
        """F-044: '+nan' in comma-delimited line is recognised as sentinel.

        Before the fix, '+nan' and '-nan' were missing from _SENTINELS
        sets.  A line like "1.0, +nan, 2.0" was treated as a header
        and the real data was consumed as column names.
        """
        content = "1.0, +nan, 2.0, 3.0\n10.0, 50.0, 75.0, 100.0\n"
        test_file = tmp_path / "plus_nan_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Should be headerless (4 columns, not a header line)
        assert len(data) == 4
        assert "col_0" in data
        assert data["col_0"][0] == 1.0
        assert np.isnan(data["col_1"][0])  # +nan → NaN
        assert data["col_2"][0] == 2.0
        assert data["col_3"][0] == 3.0

    def test_minus_nan_sentinel_headerless_whitespace(self, tmp_path: Path) -> None:
        """F-044: '-nan' in whitespace-delimited line is recognised as sentinel."""
        content = "100.0 -nan 200.0\n50.0 75.0 100.0\n"
        test_file = tmp_path / "minus_nan_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Should produce 3 columns (headerless), not treated as header
        assert len(data) == 3
        assert "col_0" in data
        assert data["col_0"][0] == 100.0
        assert np.isnan(data["col_1"][0])  # -nan → NaN

    # === F-042: MD NaN/Inf validation ===

    def test_single_nan_md_triggers_nan_inf_warning(self, tmp_path: Path) -> None:
        """F-042: Single NaN in MD triggers NaN/Inf warning via __post_init__.

        F-041/F-047: __post_init__ calls validate(complete=True) which
        checks NaN/Inf for all columns.  The duplicate NaN/Inf check in
        _validate_dev_data was removed (F-047).  A single NaN in MD
        (below the 50% NaN density threshold) is caught by the validate()
        NaN/Inf check in __post_init__.
        """
        content = "MD,AZIM\n100.0,10.0\nnan,20.0\n200.0,30.0\n300.0,40.0\n"
        test_file = tmp_path / "single_nan_md.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.warns(UserWarning, match="non-finite values"):
            read_dev_file_as_object(test_file)

    # === F-043: Case-insensitive column lookups ===

    def test_lowercase_md_header_triggers_validation(self, tmp_path: Path) -> None:
        """F-043: Lowercase 'md' header triggers MD validation.

        With normalize_aliases=False and lowercase column names,
        MD checks (negative, monotonicity, duplicates) were silently
        skipped.  Now column names are matched case-insensitively.
        """
        content = "md,tvd,azi,inc\n"
        content += "-10.0,0.0,45.0,30.0\n"
        content += "0.0,10.0,90.0,30.0\n"
        test_file = tmp_path / "lowercase_md.dev"
        test_file.write_text(content, encoding="utf-8")

        # normalize_aliases=False preserves lowercase column names
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file_as_object(test_file, normalize_aliases=False)
        # Negative MD should trigger a warning
        assert any("negative MD" in str(x.message) for x in w), (
            f"Expected 'negative MD' warning, got: {[str(x.message) for x in w]}"
        )
        # Columns should be lowercase as-is
        assert "md" in data.columns
        assert data.columns["md"][0] == -10.0

    def test_lowercase_azi_header_triggers_range_check(self) -> None:
        """F-043: Lowercase 'azi' header triggers azimuth range check.

        Direct _validate_dev_data call with lowercase column names
        should still validate azimuth range.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["md"] = np.array([100.0, 200.0, 300.0])
        dev.columns["azi"] = np.array([10.0, 400.0, 50.0])
        dev.column_order = ["md", "azi"]

        with pytest.warns(UserWarning, match="Azimuth.*outside.*0, 360"):
            _validate_dev_data(dev)

    # === F-045: TVD dedup survivor validation ===

    def test_tvd_dedup_survivor_nan_density_warns(self) -> None:
        """F-045: TVD_2 dedup survivor gets NaN density validation.

        Before the fix, TVD dedup survivors (e.g., TVD_2) bypassed all
        validation because only AZI/INC had dedup survivor blocks.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        # TVD_2 is a dedup survivor — 2/3 NaN = 66% > 50%
        dev.columns["TVD_2"] = np.array([np.nan, np.nan, 300.0])
        dev.column_order = ["MD", "TVD_2"]

        with pytest.warns(UserWarning, match="TVD.*NaN values"):
            _validate_dev_data(dev)

    def test_tvd_dedup_survivor_md_consistency_warns(self) -> None:
        """F-045: TVD_2 dedup survivor gets MD-consistency validation.

        TVD decreases where MD increases should trigger a warning.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0, 400.0])
        # TVD decreases from 500→400 while MD increases 200→300
        dev.columns["TVD_2"] = np.array([100.0, 500.0, 400.0, 450.0])
        dev.column_order = ["MD", "TVD_2"]

        with pytest.warns(UserWarning, match="TVD.*decreases"):
            _validate_dev_data(dev)

    # === F-041: DevFile __post_init__ called after reader construction ===

    def test_read_dev_file_as_object_runs_post_init_validation(self, tmp_path: Path) -> None:
        """F-041: read_dev_file_as_object calls __post_init__ after construction.

        __post_init__ runs validate(complete=True) which checks NaN/Inf
        for all columns.  A file with NaN values in non-MD columns
        should trigger the validate() NaN/Inf warning in addition to
        _validate_dev_data warnings.
        """
        content = "MD,AZI,INC\n100.0,45.0,nan\n200.0,90.0,30.0\n"
        test_file = tmp_path / "nan_post_init.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dev = read_dev_file_as_object(test_file)
        # __post_init__ runs validate(complete=True) which warns about
        # NaN in INC column
        inc_nan_warnings = [x for x in w if "DevFile: column 'INC'" in str(x.message)]
        assert len(inc_nan_warnings) >= 1, (
            f"Expected __post_init__ validate() warning for INC NaN, got {len(inc_nan_warnings)}"
        )
        # Data should be parsed correctly
        assert dev.columns["MD"][0] == 100.0

    # === F-046: Semicolon-only auto-detection ===

    def test_semicolon_delimited_basic(self, tmp_path: Path) -> None:
        """F-046: Pure semicolon-delimited DEV file without tabs.

        Auto-detection must correctly recognise semicolon as delimiter
        and parse the file with proper column names.
        """
        content = "MD;TVD;X;Y\n0.0;0.0;100.0;200.0\n100.0;99.0;101.0;201.0\n"
        test_file = tmp_path / "semicolon.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0
        assert data["TVD"][0] == 0.0
        assert data["TVD"][1] == 99.0
        assert data["X"][0] == 100.0
        assert data["Y"][0] == 200.0

    def test_semicolon_delimited_with_whitespace(self, tmp_path: Path) -> None:
        """F-046: Semicolon-delimited with whitespace around values."""
        content = "MD; TVD ; X ; Y\n0.0 ; 0.0 ; 100.0 ; 200.0\n"
        test_file = tmp_path / "semicolon_ws.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"]
        assert data["MD"][0] == 0.0
        assert data["TVD"][0] == 0.0
        assert data["X"][0] == 100.0
        assert data["Y"][0] == 200.0

    def test_semicolon_delimited_trailing_semicolons(self, tmp_path: Path) -> None:
        """F-046: Semicolon-delimited file with trailing semicolons."""
        content = "MD;TVD;X;\n0.0;0.0;100.0;\n100.0;99.0;101.0;\n"
        test_file = tmp_path / "semicolon_trail.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X"]
        assert len(data["MD"]) == 2
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0

    def test_semicolon_delimited_single_column(self, tmp_path: Path) -> None:
        """F-046: Semicolon-delimited file with a single column.

        Single-column files have only 1 semicolon token, so the
        auto-detection falls back to other delimiter checks.
        """
        content = "MD\n0.0\n100.0\n"
        test_file = tmp_path / "semicolon_single.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD"]
        assert data["MD"][0] == 0.0
        assert data["MD"][1] == 100.0

    # === F-047: Multi-line delimiter cross-validation ===

    def test_multi_line_delimiter_consistency_warns(self, tmp_path: Path) -> None:
        """F-047: Inconsistent delimiter across data lines triggers warning.

        A file where some lines have a different number of tokens
        than the first data line (but not enough to trigger the
        single-line >=3 difference error) should emit a consistency
        warning.
        """
        import warnings

        content = (
            "MD TVD INC\n"  # 3-column header
            "0.0 0.0 0.0\n"  # 3 tokens (consistent)
            "100.0 99.0\n"  # 2 tokens (mismatch)
            "200.0 198.0 30.0 40.0\n"  # 4 tokens (mismatch)
            "300.0 297.0 60.0\n"  # 3 tokens (back to consistent)
        )
        test_file = tmp_path / "multi_line_inconsist.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Verify data was parsed (4 rows, despite inconsistencies)
        assert len(data["MD"]) == 4

        # Check for delimiter consistency warning
        consistency_warnings = [
            str(x.message) for x in w if "Delimiter consistency warning" in str(x.message)
        ]
        assert len(consistency_warnings) >= 1, (
            f"Expected delimiter consistency warning, got: {[str(x.message) for x in w]}"
        )

    def test_multi_line_delimiter_consistent_no_warning(self, tmp_path: Path) -> None:
        """F-047: Consistent delimiter across data lines — no spurious warning."""
        import warnings

        content = (
            "MD TVD INC\n"
            "0.0 0.0 0.0\n"
            "100.0 99.0 30.0\n"
            "200.0 198.0 60.0\n"
            "300.0 297.0 90.0\n"
            "400.0 396.0 120.0\n"
        )
        test_file = tmp_path / "multi_line_consist.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = read_dev_file(test_file)

        consistency_warnings = [
            str(x.message) for x in w if "Delimiter consistency warning" in str(x.message)
        ]
        assert len(consistency_warnings) == 0, (
            f"Expected NO delimiter consistency warning, got: {consistency_warnings}"
        )

    # === I2F-001: DUG Pattern A all-float false positive ===

    def test_dug_pattern_a_all_float_count_match_is_headerless(self, tmp_path: Path) -> None:
        """I2F-001/N-I-24: All-float second line with count match → headerless.

        Reproducer: "4\\n1.0 2.0 3.0 4.0\\n5.0 6.0 7.0 8.0\\n9.0..."
        Before fix: col_count (4) == len(second_tokens) (4), >=3 content
        entries → detected as DUG, consuming first data row as header.
        After the I2F-001 fix it fell to ("headerless", 0), which derived
        the column count from the count line ("4" → 1 token → 1 column)
        — silently losing 3 of 4 columns (this test previously asserted
        that wrong result).  N-I-24: the count line is a column-count
        PREFIX, not data — skip it and derive 4 columns from the first
        real data line, preserving all 3 data rows.
        """
        content = "4\n1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n9.0 10.0 11.0 12.0\n"
        test_file = tmp_path / "dug_float_count_match.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Count-prefix line "4" is skipped; 4 columns from the first
        # real data line; all 3 data rows preserved (NOT consumed as header).
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(data["col_3"], [4.0, 8.0, 12.0])

    # === I2F-002: _DEV_SENTINELS shared constant ===

    def test_dev_sentinels_constant_shared(self) -> None:
        """I2F-002: Comma and whitespace paths use same _DEV_SENTINELS constant.

        Verifies the module-level constant exists and contains all
        expected sentinels (including +nan/-nan from F-044).
        """
        from pylasdev.dev_reader import _DEV_SENTINELS

        # Must be a frozenset (module-level constant)
        assert isinstance(_DEV_SENTINELS, frozenset)

        # Must contain standard sentinels
        assert "na" in _DEV_SENTINELS
        assert "null" in _DEV_SENTINELS
        assert "err" in _DEV_SENTINELS
        assert "nan" in _DEV_SENTINELS
        assert "inf" in _DEV_SENTINELS

        # Must contain +nan/-nan from F-044 fix
        assert "+nan" in _DEV_SENTINELS
        assert "-nan" in _DEV_SENTINELS

        # Must contain infinity variants
        assert "infinity" in _DEV_SENTINELS
        assert "+infinity" in _DEV_SENTINELS
        assert "-infinity" in _DEV_SENTINELS

    # === I2F-004: Co-existing variant column validation ===

    def test_coexisting_azi_variants_both_validated(self) -> None:
        """I2F-004: AZI + AZIM variants both get range validation.

        With normalize_aliases=False, multiple distinct base-name variants
        (AZI + AZIM) should both be validated.  Before fix, break after
        first match skipped AZIM.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        dev.columns["AZI"] = np.array([10.0, 90.0, 350.0])  # valid
        dev.columns["AZIM"] = np.array([400.0, 50.0, -10.0])  # out of range
        dev.column_order = ["MD", "AZI", "AZIM"]

        with pytest.warns(UserWarning) as w:
            _validate_dev_data(dev)

        # Both AZI and AZIM should be validated; AZIM has out-of-range values
        azi_warnings = [str(x.message) for x in w if "Azimuth column" in str(x.message)]
        assert len(azi_warnings) >= 1, (
            f"Expected at least 1 azimuth warning (AZIM out of range), got "
            f"{len(azi_warnings)}: {azi_warnings}"
        )
        # AZIM specifically should have an out-of-range warning
        assert any("AZIM" in msg for msg in azi_warnings), (
            f"Expected AZIM out-of-range warning, got: {azi_warnings}"
        )

    def test_coexisting_tvd_variants_both_validated(self) -> None:
        """I2F-004: TVDKB + TVDSS variants both get NaN density validated.

        With normalize_aliases=False, both TVD variants should get NaN
        density validation.  Before fix, break after first match skipped
        TVDSS.
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([100.0, 200.0, 300.0])
        dev.columns["TVDKB"] = np.array([100.0, 200.0, 300.0])  # all valid
        dev.columns["TVDSS"] = np.array([np.nan, np.nan, 100.0])  # 66% NaN
        dev.column_order = ["MD", "TVDKB", "TVDSS"]

        with pytest.warns(UserWarning) as w:
            _validate_dev_data(dev)

        # TVDSS has >50% NaN → NaN density warning expected
        tvd_nan_warnings = [
            str(x.message) for x in w if "TVD" in str(x.message) and "NaN" in str(x.message)
        ]
        assert len(tvd_nan_warnings) >= 1, (
            f"Expected at least 1 TVD NaN warning, got "
            f"{len(tvd_nan_warnings)}: {[str(x.message) for x in w]}"
        )
        # TVDSS should be specifically mentioned
        assert any("TVDSS" in msg for msg in tvd_nan_warnings), (
            f"Expected TVDSS NaN warning, got: {tvd_nan_warnings}"
        )


class TestFixGroupG9:
    """Regression tests for fix group G9 (dev_reader.py detection/parsing).

    Covers V-01..V-08 from the consolidated fix list.  Each test fails on
    pre-fix code and passes on post-fix code.
    """

    # === V-01 (HIGH): DUG Pattern B false positive ===

    def test_v01_dug_pattern_b_not_misdetected_on_ragged_first_row(self, tmp_path: Path) -> None:
        """V-01: Normal header + ragged single-integer first data row.

        DUG Pattern B's count-mismatch fallback consumed the real header
        ("MD TVD X Y") as a title, "0" as a column count, and the second
        data row as numeric column names — total parse corruption with
        zero warnings.  After the fix the all-float third line falls
        through to simple format and the ragged row is NaN-filled.
        """
        content = "MD TVD X Y\n0\n100.0 1000.0 100.0 200.0\n200.0 1100.0 150.0 250.0\n"
        test_file = tmp_path / "v01_ragged_first.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"], (
            f"Expected real column names, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(data["TVD"][1:], [1000.0, 1100.0])
        np.testing.assert_array_equal(data["X"][1:], [100.0, 150.0])
        np.testing.assert_array_equal(data["Y"][1:], [200.0, 250.0])
        # Ragged first row: only MD=0 was provided
        assert np.isnan(data["TVD"][0])
        assert np.isnan(data["X"][0])
        assert np.isnan(data["Y"][0])

    # === V-02 (HIGH): comma count-prefix DUG misdetection ===

    def test_v02_comma_count_prefix_not_misdetected_as_dug(self, tmp_path: Path) -> None:
        """V-02/N-I-24: "4\\n1.0,2.0,3.0,4.0\\n..." must be headerless, not DUG.

        The comma path returned ("dug", 2), consuming the first data row
        as a numeric header.  Matches the whitespace I2F-001 contract
        (test_dug_pattern_a_all_float_count_match_is_headerless), which
        N-I-24 corrected: the count line is a column-count PREFIX that is
        skipped, 4 columns are derived from the first real data line, and
        all 3 data rows are preserved.
        """
        content = "4\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n"
        test_file = tmp_path / "v02_comma_count_prefix.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(data["col_3"], [4.0, 8.0, 12.0])

    # === V-03 (MEDIUM): Pattern B all-float count-match returns DUG ===

    def test_v03_dug_pattern_b_all_float_count_match_not_dug(self, tmp_path: Path) -> None:
        """V-03: Pattern B all-float count-match must not return DUG.

        A normal header + single-integer count + all-float first data row
        was misdetected as DUG with numeric column names ("0.0","0.0_2"...).
        After the fix the all-float third line falls through to simple
        format, preserving real column names and data.
        """
        content = (
            "MD TVD X Y\n4\n0.0 0.0 0.0 0.0\n100.0 1000.0 100.0 200.0\n200.0 1100.0 150.0 250.0\n"
        )
        test_file = tmp_path / "v03_all_float_b.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"], (
            f"Expected real column names, got {list(data.keys())}"
        )
        # Count line "4" becomes a ragged data row; full rows are preserved
        np.testing.assert_array_equal(data["MD"], [4.0, 0.0, 100.0, 200.0])
        np.testing.assert_array_equal(data["TVD"][2:], [1000.0, 1100.0])
        np.testing.assert_array_equal(data["X"][2:], [100.0, 150.0])
        np.testing.assert_array_equal(data["Y"][2:], [200.0, 250.0])

    # === V-04 (MEDIUM): headerless semicolon first row consumed as names ===

    def test_v04_headerless_semicolon_first_row_not_names(self, tmp_path: Path) -> None:
        """V-04: Semicolon-delimited headerless file keeps first row as data.

        Before the fix there was no semicolon pre-check, so the first
        all-numeric row was consumed as column names ("1.00","2.00","3.00")
        and the real data shifted.
        """
        content = "1.00;2.00;3.00\n4.00;5.00;6.00\n7.00;8.00;9.00\n"
        test_file = tmp_path / "v04_semicolon.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 4.0, 7.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 5.0, 8.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 6.0, 9.0])

    # === V-05 (MEDIUM): Petrel well-header comma variant ===

    def test_v05_petrel_well_header_comma_variant(self, tmp_path: Path) -> None:
        """V-05: Comma-delimited Petrel well-header is detected and skipped.

        "WELL-1,1000.0,2000.0,50.0" is a single whitespace token, so the
        F-093 detection (>= 2 whitespace tokens) missed it; the well-header
        became column names and the real header became NaN data.
        """
        content = (
            "WELL-1,1000.0,2000.0,50.0\nMD,INC,AZI,TVD\n0.0,0.0,90.0,0.0\n100.0,0.0,90.0,-100.0\n"
        )
        test_file = tmp_path / "v05_petrel_comma.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        petrel_warnings = [str(x.message) for x in w if "Petrel well-header" in str(x.message)]
        assert len(petrel_warnings) == 1, (
            f"Expected Petrel well-header warning, got: {[str(x.message) for x in w]}"
        )
        assert list(data.keys()) == ["MD", "INC", "AZI", "TVD"], (
            f"Expected ['MD','INC','AZI','TVD'], got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0])
        np.testing.assert_array_equal(data["INC"], [0.0, 0.0])
        np.testing.assert_array_equal(data["AZI"], [90.0, 90.0])
        np.testing.assert_array_equal(data["TVD"], [0.0, -100.0])

    # === V-06 (MEDIUM): short FIRST data row raises instead of NaN-fill ===

    def test_v06_short_first_data_row_nan_filled_not_raise(self, tmp_path: Path) -> None:
        """V-06: Ragged first data row NaN-fills instead of DEVReadError.

        A short first row (only MD) was indistinguishable from a delimiter
        mismatch when the alternative delimiter also failed; the
        cross-validation raised.  With corroborating full rows later in the
        file, the short row is NaN-filled like any other ragged row.
        """
        content = "MD TVD X Y\n0.0\n100.0 1000.0 100.0 200.0\n200.0 1100.0 150.0 250.0\n"
        test_file = tmp_path / "v06_short_first.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0, 200.0])
        assert np.isnan(data["TVD"][0])
        assert np.isnan(data["X"][0])
        assert np.isnan(data["Y"][0])
        np.testing.assert_array_equal(data["TVD"][1:], [1000.0, 1100.0])

    # === V-07 (MEDIUM): comma-decimal locale values all-NaN ===

    def test_v07_comma_decimal_locale_values_converted(self, tmp_path: Path) -> None:
        """V-07: Comma-decimal locale values ("1,00") parse as 1.00.

        The documented Directional Drilling variant uses ``;`` as delimiter
        and ``,`` as the decimal separator.  Before the fix these values
        became NaN (no comma-to-dot conversion anywhere).
        """
        content = "MD;TVD;INC;AZI\n1,00;2,00;3,00;4,00\n5,00;6,00;7,00;8,00\n"
        test_file = tmp_path / "v07_locale.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "INC", "AZI"]
        np.testing.assert_array_equal(data["MD"], [1.0, 5.0])
        np.testing.assert_array_equal(data["TVD"], [2.0, 6.0])
        np.testing.assert_array_equal(data["INC"], [3.0, 7.0])
        np.testing.assert_array_equal(data["AZI"], [4.0, 8.0])

    # === V-08 (MEDIUM): thousands separator silently corrupts in comma mode ===

    def test_v08_thousands_separator_recombined_with_warning(self, tmp_path: Path) -> None:
        """V-08: "1,234.5" in comma mode is recombined, not column-shifted.

        Before the fix the comma delimiter split "1,234.5" into "1" and
        "234.5", shifting every subsequent column (TVD=1, X=234.5) with
        only a generic "extra columns" warning.  After the fix the
        thousands fragment is recombined to "1234.5" with a specific
        warning.
        """
        content = "MD,TVD,X,Y\n0.0,0.0,0.0,0.0\n1000.0,1,234.5,5000.0,6000.0\n"
        test_file = tmp_path / "v08_thousands.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["MD"], [0.0, 1000.0])
        np.testing.assert_array_equal(data["TVD"], [0.0, 1234.5])
        np.testing.assert_array_equal(data["X"], [0.0, 5000.0])
        np.testing.assert_array_equal(data["Y"], [0.0, 6000.0])

        thousands_warnings = [str(x.message) for x in w if "thousands separator" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Expected thousands-separator warning, got: {[str(x.message) for x in w]}"
        )


class TestFixGroupG10:
    """G10: dev_reader validation/columns fixes (V-13, V-17, V-18, E-01,
    N-I-24, N-I-25, N-I-26).

    Each test FAILS on pre-fix code and PASSES on post-fix code.
    """

    # === V-13 (MEDIUM): headerless all-integer first row consumed as names ===

    def test_v13_all_integer_first_row_preserved_as_data(self, tmp_path: Path) -> None:
        """V-13: `0 0 45` surface station is data, not column names.

        The F-92 integer heuristic consumed any all-integer first row as
        numeric column names whenever the second line's token count matched
        (only the all-zero row was protected by F-213).  `0 0 45` =
        MD0/INC0/AZI45 is a realistic surface station — it must be parsed
        as the first data row, not fabricated into columns "0"/"0_2"/"45".
        """
        content = "0 0 45\n100 5 90\n200 8 120\n"
        test_file = tmp_path / "v13_surface_station.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(data["col_1"], [0.0, 5.0, 8.0])
        np.testing.assert_array_equal(data["col_2"], [45.0, 90.0, 120.0])

    def test_v13_non_zero_integer_first_row_preserved(self, tmp_path: Path) -> None:
        """V-13: `100 200 300 400` all-integer first row is data, not names.

        EXT: the bug is broader than the all-zero F-213 gap — ANY
        all-integer headerless first row with a matching second-row count
        was consumed as column names.  All rows must survive as data.
        """
        content = "100 200 300 400\n1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n"
        test_file = tmp_path / "v13_integer_row.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [100.0, 1.0, 5.0])
        np.testing.assert_array_equal(data["col_1"], [200.0, 2.0, 6.0])
        np.testing.assert_array_equal(data["col_2"], [300.0, 3.0, 7.0])
        np.testing.assert_array_equal(data["col_3"], [400.0, 4.0, 8.0])

    # === V-17 (MEDIUM): MD dedup survivor escapes ALL MD validation ===

    def test_v17_md_survivor_non_monotonic_warns(self, tmp_path: Path) -> None:
        """V-17: MD_2 dedup survivor (MD+MDKB alias) gets MD validation.

        Before the fix, a non-monotonic MD_2 produced ZERO MD validation
        warnings (a6096f4 added the TVD survivor, not MD).  After the fix
        the _N-suffixed survivor is validated like the primary MD column.
        """
        content = (
            "MD MDKB TVD INC AZI\n"
            "0.0 150.0 0.0 0.0 90.0\n"
            "150.0 120.0 100.0 5.0 90.0\n"
            "250.0 130.0 200.0 10.0 90.0\n"
            "400.0 140.0 300.0 15.0 90.0\n"
        )
        test_file = tmp_path / "v17_md_survivor.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert "MD_2" in data, f"Expected MD_2 survivor column, got {list(data.keys())}"
        md2_warnings = [
            str(x.message)
            for x in w
            if "MD_2 values are not monotonically increasing" in str(x.message)
        ]
        assert len(md2_warnings) >= 1, (
            f"Expected MD_2 monotonicity warning, got: {[str(x.message) for x in w]}"
        )

    def test_v17_natural_md2_column_validated_not_md_not_found(self, tmp_path: Path) -> None:
        """V-17: a NATURAL MD_2 depth column is validated, not skipped.

        When the file has only MD_2 (no plain MD), the old exact-match
        lookup reported the misleading "MD column not found" warning and
        skipped ALL MD checks.  The survivor block must fire even when
        ``_md_col is None``.
        """
        content = (
            "MD_2 TVD X Y\n0.0 0.0 1.0 2.0\n-10.0 100.0 101.0 102.0\n-20.0 200.0 201.0 202.0\n"
        )
        test_file = tmp_path / "v17_natural_md2.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert "MD_2" in data
        not_found_warnings = [str(x.message) for x in w if "MD column not found" in str(x.message)]
        assert len(not_found_warnings) == 0, (
            f"Natural MD_2 column must not trigger 'MD column not found': {not_found_warnings}"
        )
        md2_warnings = [
            str(x.message)
            for x in w
            if "MD_2 values are not monotonically increasing" in str(x.message)
        ]
        assert len(md2_warnings) >= 1, (
            f"Expected MD_2 monotonicity warning, got: {[str(x.message) for x in w]}"
        )

    # === V-18 (MEDIUM): empty MIDDLE header cell → column shift ===

    def test_v18_empty_middle_header_cell_rejected(self, tmp_path: Path) -> None:
        """V-18: `MD,TVD,,X,Y` header is rejected, not silently shifted.

        The empty-token filter dropped the middle cell from the names
        while data rows kept the position → X received the empty column's
        value, Y received X's value, the last value was discarded.  A
        non-trailing empty cell is a malformed header: reject loudly.
        """
        content = "MD,TVD,,X,Y\n0,0,1,2,3\n100,50,30,40,20\n"
        test_file = tmp_path / "v18_empty_middle.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(DEVReadError, match="Empty column name in the middle"):
            read_dev_file(test_file)

    def test_v18_trailing_comma_header_still_parses(self, tmp_path: Path) -> None:
        """V-18: trailing empty cells are still dropped (no regression).

        `MD,TVD,X,Y,` (trailing delimiter) must keep parsing with 4
        columns — the fix must not regress the documented trailing-comma
        behavior.
        """
        content = "MD,TVD,X,Y,\n0,0,1,2\n100,50,30,40\n"
        test_file = tmp_path / "v18_trailing.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["MD", "TVD", "X", "Y"], (
            f"Expected 4 columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0])
        np.testing.assert_array_equal(data["TVD"], [0.0, 50.0])
        np.testing.assert_array_equal(data["X"], [1.0, 30.0])
        np.testing.assert_array_equal(data["Y"], [2.0, 40.0])

    def test_v18_dug_path_empty_middle_header_rejected(self, tmp_path: Path) -> None:
        """V-18: the DUG header path rejects middle empty cells too."""
        content = "Survey\n4\nMD,TVD,,X,Y\n0,0,1,2,3\n"
        test_file = tmp_path / "v18_dug_middle.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(DEVReadError, match="Empty column name in the middle"):
            read_dev_file(test_file)

    # === E-01 (MEDIUM) / ENC-03: DEV docstring error contract ===

    def test_e01_dev_docstrings_promise_lasencodingerror(self) -> None:
        """E-01/ENC-03: both DEV entry-point docstrings document the
        LASEncodingError propagation contract.  A genuine decoding failure
        raises LASEncodingError (NOT a misleading 'size exceeded'
        DEVReadError); the docstrings' Raises sections must list it.
        """
        import inspect

        from pylasdev.dev_reader import read_dev_file, read_dev_file_as_object

        for fn in (read_dev_file, read_dev_file_as_object):
            doc = inspect.getdoc(fn)
            assert doc is not None, f"Missing docstring on {fn.__name__}"
            assert "LASEncodingError" in doc, (
                f"{fn.__name__} docstring must document LASEncodingError propagation (ENC-03)"
            )
            assert "DEVReadError" in doc, f"{fn.__name__} docstring must document DEVReadError"

    # === N-I-24 (MEDIUM, PFA): whitespace count-prefix data loss ===

    def test_n_i24_whitespace_count_prefix_preserves_all_columns(self, tmp_path: Path) -> None:
        """N-I-24: whitespace count-prefix file keeps all columns/rows.

        `4\\n1.0 2.0 3.0 4.0\\n...` — the count line is a column-count
        PREFIX, not data.  Before the fix the headerless path derived 1
        column from the count line, silently losing 3 of 4 columns (the
        I2F-001 test asserted that wrong result).  After the fix the count
        line is skipped and all 3 data rows are preserved in 4 columns.
        """
        content = "4\n1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n9.0 10.0 11.0 12.0\n"
        test_file = tmp_path / "n_i24_count_prefix.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(data["col_3"], [4.0, 8.0, 12.0])

    def test_n_i24_comma_count_prefix_consistent_with_whitespace(self, tmp_path: Path) -> None:
        """N-I-24: comma count-prefix file matches the whitespace contract.

        The comma twin must produce the same 4-column / all-rows-preserved
        result as the whitespace side (the asymmetry was the finding's
        crux; G9's V-02 fix aligned it with the wrong I2F-001 result).
        """
        content = "4\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n"
        test_file = tmp_path / "n_i24_comma_prefix.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(data["col_3"], [4.0, 8.0, 12.0])

    # === N-I-25 (MEDIUM): mixed-delimiter headerless → single col_0 ===

    def test_n_i25_mixed_delimiter_first_line_delimiter_governs(self, tmp_path: Path) -> None:
        """N-I-25: a space first line is not re-interpreted via a later
        comma line.

        `1.0 2.0 3.0\\n4.0,5.0,6.0\\n...` — the comma search picked the
        comma line for delimiter detection, reducing the space-delimited
        first line to a single col_0 with NaN first value.  The first
        line's own delimiter must govern; the comma lines become ragged
        rows (NaN-filled) with warnings.
        """
        content = "1.0 2.0 3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n"
        test_file = tmp_path / "n_i25_mixed.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected 3 columns from the first line's delimiter, got {list(data.keys())}"
        )
        # First (space-delimited) line is parsed correctly
        assert data["col_0"][0] == 1.0
        assert data["col_1"][0] == 2.0
        assert data["col_2"][0] == 3.0

    # === N-I-26 (MEDIUM): unbounded str.split DoS bypass ===

    def test_n_i26_delimiter_detection_split_respects_token_cap(self, tmp_path: Path) -> None:
        """N-I-26: delimiter-detection splits are bounded by the G-18 cap.

        The delimiter-detection block ran unbounded ``str.split()`` on the
        full header string before Pass 2's token-cap guards applied (177x
        memory amplification on pathological single-line files).  With a
        low cap, the bounded space split changes the comma-vs-space count
        comparison and therefore the chosen delimiter — proving the cap
        applies inside the detection block.
        """
        from unittest import mock

        content = "A,B,C D E F G\n1,2,3 4 5 6 7\n8,9,10 11 12 13\n"
        test_file = tmp_path / "n_i26_capped.dev"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_dev_file(test_file)

        # maxsplit=2 caps the space split at 3 tokens; the comma split is
        # also capped at 3 → comma (3) >= space (3) → comma delimiter.
        # An UNBOUNDED space split (5 tokens) would pick space instead.
        assert list(data.keys()) == ["A", "B"], (
            f"Expected comma delimiter with capped token counts, got {list(data.keys())}"
        )


class TestFixDev01SemicolonCountPrefix:
    """DEV-01 (MEDIUM): semicolon count-prefix file silently destroyed.

    ``4\\n1.0;2.0;3.0;4.0\\n...`` — the comma pre-check and the whitespace
    DUG Pattern A return ("headerless", 1) for a single-integer column-count
    prefix first line, but the semicolon pre-check returned ("headerless", 0)
    unconditionally, so the count line was consumed as the first data row and
    3 of 4 columns were lost (col_0=[4.0,nan,nan,nan]).  The header
    re-derivation also only searched for comma-bearing lines, so the
    semicolon data line was never found and the delimiter fell back to space.

    Each test FAILS on pre-fix code and PASSES on post-fix code.
    """

    def test_semicolon_count_prefix_preserves_all_columns(self, tmp_path: Path) -> None:
        """The semicolon twin must match the comma/whitespace contract."""
        content = "4\n1.0;2.0;3.0;4.0\n5.0;6.0;7.0;8.0\n9.0;10.0;11.0;12.0\n"
        test_file = tmp_path / "dev01_semi_prefix.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 7.0, 11.0])
        np.testing.assert_array_equal(data["col_3"], [4.0, 8.0, 12.0])

    def test_semicolon_count_prefix_explicit_delimiter(self, tmp_path: Path) -> None:
        """Explicit delimiter=";" must not rescue differently (count line
        is still skipped)."""
        content = "4\n1.0;2.0;3.0;4.0\n5.0;6.0;7.0;8.0\n9.0;10.0;11.0;12.0\n"
        test_file = tmp_path / "dev01_semi_prefix_explicit.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file, delimiter=";")

        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3"], (
            f"Expected 4 headerless columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 6.0, 10.0])


class TestFixDev02RaggedRowNoFalseMerge:
    """DEV-02 (MEDIUM): thousands-recombination false-merge.

    M-76's relaxed gate (``len(values) > expected``) merged a genuinely
    ragged row's columns into one bogus value: ``MD,TVD,X,Y`` + row
    ``100,200,300.5,400,500`` → MD=100200300.5, Y=NaN, with a factually
    wrong "thousands separator" warning.  The recombine function must
    reject the run when the recombined row cannot satisfy the declared
    column count and leave the genuine columns intact.

    FAILS on pre-fix code (false merge), PASSES on post-fix code.
    """

    def test_headered_ragged_row_not_false_merged(self, tmp_path: Path) -> None:
        content = "MD,TVD,X,Y\n100,200,300.5,400,500\n101,201,301.5,401,501\n"
        test_file = tmp_path / "dev02_ragged.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Genuine values preserved — NO 100200300.5 bogus value.
        np.testing.assert_array_equal(data["MD"], [100.0, 101.0])
        np.testing.assert_array_equal(data["TVD"], [200.0, 201.0])
        np.testing.assert_array_equal(data["X"], [300.5, 301.5])
        np.testing.assert_array_equal(data["Y"], [400.0, 401.0])

        # The warning must NOT claim a thousands separator (factually wrong).
        thousands_warnings = [str(x.message) for x in w if "thousands separator" in str(x.message)]
        assert len(thousands_warnings) == 0, (
            f"Ragged row must not be reported as a thousands separator, got: {thousands_warnings}"
        )

    def test_headerless_ragged_row_not_false_merged(self, tmp_path: Path) -> None:
        content = "100,200,300.5,400,500\n101,201,301.5,401,501\n"
        test_file = tmp_path / "dev02_ragged_hdrless.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        # All five genuine columns survive (no 100200300.5 merge).
        assert list(data.keys()) == ["col_0", "col_1", "col_2", "col_3", "col_4"], (
            f"Expected 5 genuine columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [100.0, 101.0])
        np.testing.assert_array_equal(data["col_2"], [300.5, 301.5])
        np.testing.assert_array_equal(data["col_4"], [500.0, 501.0])


class TestFixDev03TwoThousandsValues:
    """DEV-03 (MEDIUM): only the first of two thousands values recombined.

    M-53's first-match early return recombined only the FIRST run per row;
    a second thousands value (``5,987,654.3``) was destroyed into fabricated
    columns with NO warning.  Every completed run must now recombine with
    its own warning.

    FAILS on pre-fix code (second value destroyed, single warning),
    PASSES on post-fix code.
    """

    def test_two_thousands_values_both_recombined(self, tmp_path: Path) -> None:
        content = "A,B,C,D,E,F\n1,2,4,123,456.7,5,987,654.3\n"
        test_file = tmp_path / "dev03_two_thousands.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["A"], [1.0])
        np.testing.assert_array_equal(data["B"], [2.0])
        np.testing.assert_array_equal(data["C"], [4123456.7])
        # Second thousands value recombined instead of D=5.0/E=987.0/F=654.3.
        np.testing.assert_array_equal(data["D"], [5987654.3])

        thousands_warnings = [str(x.message) for x in w if "thousands separator" in str(x.message)]
        assert len(thousands_warnings) >= 2, (
            f"Expected a warning for EVERY recombined value, got: {[str(x.message) for x in w]}"
        )
        assert any("5,987,654.3" in m for m in thousands_warnings), (
            f"Second thousands value must be warned, got: {thousands_warnings}"
        )


class TestFixDev04SemicolonCommaDecimalHeaderless:
    """DEV-04 (MEDIUM): semicolon+comma-decimal headerless mis-delimited.

    The format detector correctly reports headerless (semicolon path), but
    the delimiter picker counted comma FRAGMENTS ("1","00;2","00" = 3) vs
    semicolon tokens (2) and picked comma, so ``1,00;2,00`` became 3 columns
    with ``2,00`` destroyed.  For headerless files the delimiter must be
    chosen by how cleanly it splits the first data line into numeric values.

    FAILS on pre-fix code (3 columns), PASSES on post-fix code (2 columns).
    """

    def test_semicolon_comma_decimal_headerless_two_columns(self, tmp_path: Path) -> None:
        content = "1,00;2,00\n3,00;4,00\n5,00;6,00\n"
        test_file = tmp_path / "dev04_semi_locale.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1"], (
            f"Expected 2 semicolon-delimited columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 4.0, 6.0])

    def test_semicolon_comma_decimal_first_line_shape(self, tmp_path: Path) -> None:
        """Literal task shape: ``1,5;2,0;3,0`` first line must stay
        semicolon-delimited with all three comma-decimal values intact."""
        content = "1,5;2,0;3,0\n4,5;5,0;6,0\n"
        test_file = tmp_path / "dev04_semi_locale3.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected 3 semicolon-delimited columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.5, 4.5])
        np.testing.assert_array_equal(data["col_1"], [2.0, 5.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 6.0])


class TestFixDev05SpaceHeaderlessThousandsFirstRow:
    """DEV-05 (MEDIUM): space headerless + thousands first row.

    ``1,234 5,678\\n...`` — the whitespace headerless detection rejected the
    thousands tokens (M-25 gate: "1,234" is NOT a float), so the first data
    row was consumed as a fabricated comma-split header AND the delimiter
    switched to comma (3 comma fragments > 2 space tokens), destroying the
    whole file (cols ['1','234 5','678']).  Thousands-style tokens must be
    recognised as numeric data for detection/delimiter purposes (values
    still parse to NaN per the documented M-25 thousands behavior, with a
    warning — but the structure, delimiter, and row count are correct).

    FAILS on pre-fix code (fabricated 3-column header), PASSES on post-fix.
    """

    def test_space_thousands_first_row_not_consumed_as_header(self, tmp_path: Path) -> None:
        content = "1,234 5,678\n9,000 10,456\n11,000 12,789\n"
        test_file = tmp_path / "dev05_space_thousands.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Structure: 2 space-delimited columns, 3 data rows — the first
        # row is NOT consumed as a fabricated header ('1','234 5','678').
        assert list(data.keys()) == ["col_0", "col_1"], (
            f"Expected 2 columns from the space delimiter, got {list(data.keys())}"
        )
        assert len(data["col_0"]) == 3, f"Expected 3 data rows, got {len(data['col_0'])}"
        # M-25 documented behavior: thousands values in space-delimited
        # data are left unconverted (NaN) with an explicit warning.
        assert all(np.isnan(v) for v in data["col_0"]), (
            f"col_0 should be all-NaN (thousands unsupported), got {list(data['col_0'])}"
        )
        thousands_warnings = [str(x.message) for x in w if "thousands separators" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Expected a thousands-separator warning, got: {[str(x.message) for x in w]}"
        )


class TestFixI2_14SpaceCommaDecimalHeaderless:
    """I2-14 (MEDIUM): space-delimited comma-decimal headerless file.

    ``1,5 2,0 3,0`` headerless was mis-delimited as comma (the raw
    fragment-count heuristic always beat space: 4 comma fragments > 3 space
    tokens), destroying the file (M-54's own docstring example corrupts).
    The delimiter picker is now comma-decimal-aware (DEV-A's
    ``_pick_headerless_delimiter`` scores by numeric-token fraction, so
    space wins 3/3 vs comma 2/4), and the count-prefix variant
    (``3\\n1,5 2,0 3,0\\n...``) is no longer misdetected as DUG (the DUG
    Pattern A second-line check is comma-decimal aware).

    FAILS on pre-fix code (fabricated cols ['1','5 2','0 3','0']), PASSES
    on post-fix.
    """

    def test_space_comma_decimal_headerless_m_docstring_example(self, tmp_path: Path) -> None:
        """M-54's own docstring example ``1,5 2,0 3,0`` parses as space."""
        content = "1,5 2,0 3,0\n2,5 4,0 6,0\n3,5 6,0 9,0\n"
        test_file = tmp_path / "i214_space_locale.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected 3 space-delimited columns, got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.5, 2.5, 3.5])
        np.testing.assert_array_equal(data["col_1"], [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 6.0, 9.0])

    def test_count_prefix_variant_not_misdetected_as_dug(self, tmp_path: Path) -> None:
        """``3\\n1,5 2,0 3,0\\n...`` — count-prefix stays headerless,
        not DUG (pre-fix: ('dug', 2), fabricated comma-split columns)."""
        content = "3\n1,5 2,0 3,0\n4,5 5,0 6,0\n7,5 8,0 9,0\n"
        test_file = tmp_path / "i214_count_prefix.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1", "col_2"], (
            f"Expected 3 space-delimited columns (count-prefix), got {list(data.keys())}"
        )
        np.testing.assert_array_equal(data["col_0"], [1.5, 4.5, 7.5])
        np.testing.assert_array_equal(data["col_1"], [2.0, 5.0, 8.0])
        np.testing.assert_array_equal(data["col_2"], [3.0, 6.0, 9.0])


class TestFixI2_15NumericPetrelWellName:
    """I2-15 (MEDIUM): numeric Petrel well name defeats F-093.

    A Petrel well-header whose well NAME is numeric (``100 1000.0 2000.0
    50.0``) is all-numeric, so F-093's ``not _is_float_token(_hdr_tokens[0])``
    guard rejected it → headerless misdetection, fabricated col_0..col_3,
    real 10-column header row all-NaN.  The fix: when the SECOND content
    line is a genuine text column header, the all-float headerless paths
    fall through to F-093, and F-093 accepts a numeric well name.

    FAILS on pre-fix code (fabricated col_0..col_3), PASSES on post-fix.
    """

    def test_numeric_well_name_petrel_parses_correctly(self, tmp_path: Path) -> None:
        content = (
            "100 1000.0 2000.0 50.0\n"  # Petrel well-header, numeric name
            "MD TVD INC AZI\n"  # real column names
            "0 0 45 1.0\n"  # data row 1
            "100 100.0 45.5 2.0\n"  # data row 2
            "200 200.0 45.6 3.0\n"  # data row 3
        )
        test_file = tmp_path / "i215_numeric_petrel.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Correct column names from the real header (NOT col_0..col_3
        # fabricated from the well-header, and NOT the well-header tokens).
        assert list(data.keys()) == ["MD", "TVD", "INC", "AZI"], (
            f"Expected ['MD','TVD','INC','AZI'], got {list(data.keys())}"
        )
        # Real data rows (the header row must NOT become an all-NaN row).
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(data["TVD"], [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(data["INC"], [45.0, 45.5, 45.6])
        np.testing.assert_array_equal(data["AZI"], [1.0, 2.0, 3.0])
        # The well-header warning still fires (F-093 behavior preserved).
        petrel_warnings = [str(x.message) for x in w if "Petrel well-header" in str(x.message)]
        assert len(petrel_warnings) == 1, (
            f"Expected Petrel well-header warning, got: {[str(x.message) for x in w]}"
        )


class TestFixI2_17SemicolonUSThousands:
    """I2-17 (MEDIUM): semicolon US-locale files silently corrupted.

    ``MD;TVD\\n100.0;1,234\\n`` read TVD as [1.234] — a genuine US-locale
    thousands value 1234 silently converted to 1.234 (1000x corruption)
    with ZERO warnings (F-13 ungated the semicolon path, contradicting
    M-25's documented intent).  The fix reads 3-digit comma groups in
    semicolon files as thousands VALUES (1234) with a LOUD summary warning.

    FAILS on pre-fix code (TVD=[1.234,2.345], 0 warnings), PASSES on
    post-fix.
    """

    def test_semicolon_us_thousands_read_as_thousands_with_warning(self, tmp_path: Path) -> None:
        content = "MD;TVD\n100.0;1,234\n200.0;2,345\n"
        test_file = tmp_path / "i217_semi_us.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["MD", "TVD"]
        np.testing.assert_array_equal(data["MD"], [100.0, 200.0])
        # Thousands reading: 1,234 -> 1234 (NOT 1.234).
        np.testing.assert_array_equal(data["TVD"], [1234.0, 2345.0])
        # A LOUD warning must fire (pre-fix: zero warnings).
        thousands_warnings = [str(x.message) for x in w if "thousands separators" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Expected a thousands-separator warning, got: {[str(x.message) for x in w]}"
        )
        assert "read as thousands" in thousands_warnings[0], (
            f"Warning must state the thousands reading, got: {thousands_warnings[0]}"
        )

    def test_semicolon_v07_locale_decimals_still_convert(self, tmp_path: Path) -> None:
        """V-07 regression guard: 1-2 digit comma decimals still convert."""
        content = "MD;TVD;INC;AZI\n1,00;2,00;3,00;4,00\n5,00;6,00;7,00;8,00\n"
        test_file = tmp_path / "i217_v07_guard.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["MD", "TVD", "INC", "AZI"]
        np.testing.assert_array_equal(data["MD"], [1.0, 5.0])
        np.testing.assert_array_equal(data["TVD"], [2.0, 6.0])
        np.testing.assert_array_equal(data["INC"], [3.0, 7.0])
        np.testing.assert_array_equal(data["AZI"], [4.0, 8.0])


class TestFixI2_18LinearThousandsRecombine:
    """I2-18 (MEDIUM): O(n²) hang in _recombine_thousands_separators.

    A comma row of n bare 3-digit tokens (e.g. ``234,234,...``) triggered a
    quadratic scan — ~18min at 100K tokens.  The recombine is now a single
    left-to-right pass: a run without a closing decimal/exponent fragment
    is emitted and LOCKED (never re-scanned from inside).  The function is
    linear; a 20000-bare-token row completes in well under a second.

    FAILS on pre-fix code (44.5s at n=20000 end-to-end), PASSES on post-fix.
    """

    def test_large_bare_three_digit_row_is_linear(self, tmp_path: Path) -> None:
        import time

        from pylasdev.dev_reader import _recombine_thousands_separators

        n = 20000
        values = ["234"] * n
        t0 = time.perf_counter()
        _recombine_thousands_separators(values, n - 1)
        elapsed = time.perf_counter() - t0
        # Linear single-pass: ~4ms at n=20000 post-fix (was ~44s end-to-end
        # pre-fix).  A generous <2s bound rejects the quadratic pre-fix
        # behavior while leaving headroom for slow CI machines.
        assert elapsed < 2.0, (
            f"_recombine_thousands_separators took {elapsed:.2f}s on a "
            f"{n}-token bare row — must be linear, not O(n²) (I2-18)"
        )

    def test_large_bare_three_digit_row_end_to_end(self, tmp_path: Path) -> None:
        """End-to-end through the reader on a headerless comma file with 2
        rows of n bare 3-digit tokens (finding baseline: 29.28s at n=16000).
        """
        import time

        n = 8000
        row = ",".join(["234"] * n)
        content = f"{row}\n{row}\n"
        test_file = tmp_path / "i218_bare_row.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            data = read_dev_file(test_file)
            elapsed = time.perf_counter() - t0

        assert list(data.keys()) == [f"col_{i}" for i in range(n)], (
            f"Expected {n} columns, got {len(list(data.keys()))}"
        )
        assert elapsed < 5.0, (
            f"End-to-end read took {elapsed:.2f}s for n={n} bare tokens — "
            f"must be linear (pre-fix: 7.30s at n=8000, 29.28s at n=16000; "
            f"post-fix: ~0.5s at n=8000)"
        )


class TestFixENC03DevEncodingErrorPropagation:
    """ENC-03 (dev_reader side): genuine LASEncodingError is not relabeled.

    read_dev_file / read_dev_file_as_object previously caught LASEncodingError
    in the ``(ValueError, LookupError, LASEncodingError)`` tuple and re-raised
    a misleading ``DEVReadError('size exceeded or invalid parameter')``.  The
    tuple is now ``except ValueError`` (size-exceeded stays DEVReadError per
    README/code/tests consensus); LASEncodingError propagates with its
    accurate message.

    FAILS on pre-fix code (DEVReadError 'size exceeded'), PASSES on post-fix.
    """

    def test_genuine_decode_failure_propagates_las_encoding_error(self, tmp_path: Path) -> None:
        from unittest import mock

        content = b"\xff\xfe\x00\x01"  # Invalid encoding bytes
        test_file = tmp_path / "enc03_bad_encoding.dev"
        test_file.write_bytes(content)

        # Without chardet and empty fallback chain, encoding fails.
        with mock.patch("pylasdev.encoding.FALLBACK_ENCODINGS", []):
            with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
                with pytest.raises(LASEncodingError, match="Failed to decode"):
                    read_dev_file(test_file)

    def test_size_exceeded_stays_dev_read_error(self, tmp_path: Path) -> None:
        """Guard: the size-exceeded contract stays DEVReadError (ENC-03)."""
        content = "MD TVD\n0 0\n100 100\n"
        test_file = tmp_path / "enc03_size.dev"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(DEVReadError, match="size exceeded"):
            read_dev_file(test_file, max_file_size=1)


class TestFixMOD11DevMetaFilter:
    """MOD-11 (dev_reader side): legacy dict filter respects _DEV_META_KEYS.

    read_dev_file's legacy dict filter stripped EVERY ``_meta_``-prefixed key
    unconditionally, hijacking a user column literally named ``_meta_...``
    (e.g. ``_meta_source_file`` with an array value).  The filter now uses
    models' ``_is_encoded_dev_metadata_key`` (closed ``_DEV_META_KEYS`` +
    value-shape disambiguation): only ``_meta_<known>`` with a
    metadata-shaped value is metadata; everything else is a user column
    verbatim.

    FAILS on pre-fix code (column dropped), PASSES on post-fix.
    """

    def test_meta_source_file_user_column_preserved(self, tmp_path: Path) -> None:
        content = "_meta_source_file MD\n1 0\n2 100\n"
        test_file = tmp_path / "mod11_meta_col.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file, normalize_aliases=False)

        # The user column must survive the legacy dict filter.
        assert "_meta_source_file" in data, (
            f"User column '_meta_source_file' must be preserved, got {list(data.keys())}"
        )
        assert "MD" in data, f"Expected MD column, got {list(data.keys())}"
        # Real metadata keys still strip (they are not columns).
        assert "source_file" not in data, (
            f"Metadata key 'source_file' must be stripped, got {list(data.keys())}"
        )
        assert "encoding" not in data
        assert "column_order" not in data
        # Column data is intact.
        np.testing.assert_array_equal(data["_meta_source_file"], [1.0, 2.0])
        np.testing.assert_array_equal(data["MD"], [0.0, 100.0])

    def test_real_metadata_keys_still_strip(self, tmp_path: Path) -> None:
        """Guard: genuine metadata still removed; collision roundtrip works."""
        content = "source_file MD\n0 0\n100 100\n"
        test_file = tmp_path / "mod11_collision.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # normalize_aliases=False keeps the literal "source_file" column
            # name so the metadata collision path (to_dict emits
            # _meta_source_file) is exercised.
            data = read_dev_file(test_file, normalize_aliases=False)

        # Column named "source_file" is data, preserved verbatim.
        assert "source_file" in data, (
            f"Column 'source_file' must be preserved, got {list(data.keys())}"
        )
        # The METADATA emitted as _meta_source_file (collision path) strips.
        assert not any(k.startswith("_meta_") for k in data), (
            f"No _meta_ metadata may leak, got {list(data.keys())}"
        )
        assert "MD" in data


class TestFixI2_16HeaderedThousandsNoFalseMerge:
    """I2-16 (MEDIUM): F-07 4-digit-leading-group guard for ALL rows.

    The F-07 guard (a 4+ digit leading group like ``1000`` in
    ``1000,234.5`` is NOT a valid thousands grouping) was previously wired
    only to the headerless FIRST row; headered rows and headerless LATER
    rows false-merged ``1000,234.5`` -> ``1000234.5``, destroying the
    genuine 1000 and 234.5 values.  The recombine function now enforces
    the guard inside for EVERY row (DEV-A's rewrite, I2-16).

    FAILS on pre-fix code (MD=[1000234.5]), PASSES on post-fix.
    """

    def test_headered_row_four_digit_leading_group_not_merged(self, tmp_path: Path) -> None:
        """Finding shape: 2-column header + 3-token row (surplus) — the
        4+ digit leading group ``1000`` must NOT recombine into 1000234.5.
        """
        content = "MD,INC\n1000,234.5,999\n1001,235.5,998\n"
        test_file = tmp_path / "i216_headered.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["MD", "INC"], f"Expected ['MD','INC'], got {list(data.keys())}"
        # 1000 and 234.5 must stay separate columns — NOT 1000234.5.
        np.testing.assert_array_equal(data["MD"], [1000.0, 1001.0])
        np.testing.assert_array_equal(data["INC"], [234.5, 235.5])
        # No thousands warning — nothing was recombined.
        thousands_warnings = [str(x.message) for x in w if "thousands separator" in str(x.message)]
        assert len(thousands_warnings) == 0, (
            f"No thousands warning expected (no merge), got: {thousands_warnings}"
        )

    def test_headerless_later_row_four_digit_leading_group_not_merged(self, tmp_path: Path) -> None:
        """Finding shape (ADV-M2): headerless file whose FIRST row defines 2
        columns, and a LATER row carries ``1000,234.5,999`` (surplus).  The
        F-07 guard was wired only to the headerless FIRST row, so the later
        row false-merged 1000,234.5 -> 1000234.5.  The guard now applies
        inside the recombine function for EVERY row (I2-16)."""
        content = "1,2\n3,4\n1000,234.5,999\n"
        test_file = tmp_path / "i216_later.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert list(data.keys()) == ["col_0", "col_1"], (
            f"Expected 2 columns (first row defines count), got {list(data.keys())}"
        )
        # 1000 and 234.5 must stay separate — NOT 1000234.5 in col_0.
        np.testing.assert_array_equal(data["col_0"], [1.0, 3.0, 1000.0])
        np.testing.assert_array_equal(data["col_1"], [2.0, 4.0, 234.5])
        # The extra 999 is discarded with an extra-column warning (not a
        # thousands merge warning).
        thousands_warnings = [str(x.message) for x in w if "thousands separator" in str(x.message)]
        assert len(thousands_warnings) == 0, (
            f"No thousands warning expected (no merge), got: {thousands_warnings}"
        )
