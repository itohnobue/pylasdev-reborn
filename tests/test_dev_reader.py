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
        with mock.patch(
            "pylasdev.dev_reader._to_finite_float",
            side_effect=IndexError("simulated"),
        ), pytest.raises(IndexError, match="simulated"):
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
        with mock.patch(
            "pylasdev.dev_reader._to_finite_float",
            side_effect=IndexError("simulated"),
        ), pytest.raises(IndexError, match="simulated"):
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
        content = "MD TVD\n0.0\x7F0.0\n"
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
            "MD TVD INC\n"              # 3 columns expected
            "0.0 0.0 0.0\n"             # matching row (row 1)
            "1.0 1.0 1.0 9.0\n"         # extra column (row 2)
            "2.0 2.0 2.0 9.0\n"         # extra column (row 3)
            "3.0 3.0\n"                 # short row (row 4)
            "4.0 4.0\n"                 # short row (row 5)
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
        assert any("2 data line(s)" in m for m in summary_extra), \
            f"Extra-col summary should mention '2 data line(s)', got: {summary_extra}"
        assert any("2 data line(s)" in m for m in summary_short), \
            f"Short-row summary should mention '2 data line(s)', got: {summary_short}"


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
            "WELL-1 1000.0 2000.0 50.0\n"   # Petrel well-header
            "MD INC AZI TVD\n"                # real column names
            "0.0 0.0 90.0 0.0\n"             # data row 1
            "100.0 0.0 90.0 -100.0\n"         # data row 2
            "200.0 0.0 90.0 -200.0\n"         # data row 3
        )
        test_file = tmp_path / "petrel.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Verify Petrel well-header warning was emitted
        petrel_warnings = [
            str(x.message) for x in w
            if "Petrel well-header" in str(x.message)
        ]
        assert len(petrel_warnings) == 1, (
            f"Expected Petrel well-header warning, got: "
            f"{[str(x.message) for x in w]}"
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
        content = (
            "100 200 300\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
        )
        test_file = tmp_path / "numeric_header.dev"
        test_file.write_text(content, encoding="utf-8")

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = read_dev_file(test_file)

        # Should NOT have a Petrel warning
        petrel_warnings = [
            str(x.message) for x in w
            if "Petrel well-header" in str(x.message)
        ]
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
        """F-10: F-DV01 count-mismatch fallback (dev_reader.py:215-218).

        Trigger: first line is a single integer (col_count), second line
        has all-float tokens with a DIFFERENT count than col_count, and
        3+ content entries exist.  This activates the fallback which
        treats the second line as a DUG header with numeric column names.
        """
        # col_count=3 but second line has 4 float tokens (mismatch)
        # 3+ content entries → fallback activates, returns ("dug", 2)
        # Second line becomes header: columns ["1.0", "2.0", "3.0", "4.0"]
        content = (
            "3\n"
            "1.0 2.0 3.0 4.0\n"
            "100.0 200.0 300.0 400.0\n"
            "500.0 600.0 700.0 800.0\n"
        )
        test_file = tmp_path / "dv01_fallback.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # Detected as DUG with numeric column names
        assert "1.0" in data
        assert "2.0" in data
        assert "3.0" in data
        assert "4.0" in data
        assert len(data["1.0"]) == 2  # 2 data rows
        assert data["1.0"][0] == 100.0
        assert data["1.0"][1] == 500.0
        assert data["2.0"][0] == 200.0
        assert data["3.0"][0] == 300.0
        assert data["4.0"][0] == 400.0

    # --- F-10 variant: F-DV01 with 2.0e1-style float tokens ---
    def test_dv01_fallback_with_scientific_notation_headers(self, tmp_path: Path) -> None:
        """F-10: F-DV01 fallback with scientific-notation numeric header tokens.

        Verifies the fallback works when second-line tokens use scientific
        notation (e.g. 1.0e2).  The _is_float_token() function handles
        e/E/d/D notation.
        """
        content = (
            "2\n"
            "1.0e2 2.0E-1 3.14159\n"
            "100.0 200.0 300.0\n"
            "400.0 500.0 600.0\n"
        )
        test_file = tmp_path / "dv01_sci.dev"
        test_file.write_text(content, encoding="utf-8")

        data = read_dev_file(test_file)
        # col_count=2, 3 tokens on second line, all float → fallback
        # Column names are the float strings as-is
        assert "1.0e2" in data
        assert "2.0E-1" in data
        assert "3.14159" in data
        assert len(data["1.0e2"]) == 2
        assert data["1.0e2"][0] == 100.0
        assert data["2.0E-1"][0] == 200.0
        assert data["3.14159"][0] == 300.0


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
        content = (
            "MD,TVD,INC\n"
            "1000.0 2000.0 3000.0\n"
            "1100.0 2100.0 3100.0\n"
        )
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
        content = (
            "Well-Survey\n"
            "3\n"
            "MD,TVD,INC\n"
            "1000.0 2000.0 3000.0\n"
            "1100.0 2100.0 3100.0\n"
        )
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
        content = (
            "*COLUMNS,*MD,*TVD,*GR\n"
            "100.0 50.0 75.0\n"
        )
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
        assert list(data.keys()) == ["MD", "TVD"], (
            f"Expected ['MD','TVD'], got {list(data.keys())}"
        )
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
        """No MD column → validation is a no-op (returns immediately)."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["TVD"] = np.array([0.0, 100.0])
        dev.columns["X"] = np.array([100.0, 101.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
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
        """
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile()
        dev.columns["MD"] = np.array([np.nan, np.nan, np.nan])
        dev.columns["AZIM"] = np.array([10.0, 20.0, 30.0])
        # Only NaN-density warning fires; no subsequent warnings
        with pytest.warns(UserWarning, match="NaN values.*delimiter mismatch") as w:
            _validate_dev_data(dev)
        assert len(w) == 1

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
