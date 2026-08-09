"""Tests for LAS data comparison utilities."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pytest

from pylasdev.compare import _compare_lists, compare_las_dicts
from pylasdev.models import CurveDefinition


class TestCompareLasDicts:
    """Tests for compare_las_dicts function."""

    def test_identical_dicts(self) -> None:
        """Test comparing identical dicts returns True."""
        d1 = {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "100"},
            "logs": {"DEPT": np.array([1.0, 2.0, 3.0])},
            "curves_order": ["DEPT"],
        }
        d2 = {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "100"},
            "logs": {"DEPT": np.array([1.0, 2.0, 3.0])},
            "curves_order": ["DEPT"],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_different_values(self) -> None:
        """Test comparing dicts with different scalar values."""
        d1 = {"version": {"VERS": "2.0"}}
        d2 = {"version": {"VERS": "1.2"}}
        assert compare_las_dicts(d1, d2) is False

    def test_missing_key(self) -> None:
        """Test comparing dicts where key is missing in first."""
        d1 = {"version": {"VERS": "2.0"}}
        d2 = {"version": {"VERS": "2.0"}, "extra": "value"}
        assert compare_las_dicts(d1, d2) is False

    def test_extra_nested_key(self) -> None:
        """Test comparing dicts where first has extra nested key."""
        d1 = {"well": {"STRT": "100", "STOP": "200"}}
        d2 = {"well": {"STRT": "100"}}
        assert compare_las_dicts(d1, d2) is False

    def test_array_size_mismatch(self) -> None:
        """Test comparing dicts with different array sizes."""
        d1 = {"logs": {"DEPT": np.array([1.0, 2.0])}}
        d2 = {"logs": {"DEPT": np.array([1.0, 2.0, 3.0])}}
        assert compare_las_dicts(d1, d2) is False

    def test_array_value_mismatch(self) -> None:
        """Test comparing dicts with different array values."""
        d1 = {"logs": {"DEPT": np.array([1.0, 2.0, 3.0])}}
        d2 = {"logs": {"DEPT": np.array([1.0, 2.0, 4.0])}}
        assert compare_las_dicts(d1, d2) is False

    def test_array_within_tolerance(self) -> None:
        """Test comparing arrays within tolerance."""
        d1 = {"logs": {"DEPT": np.array([1.0, 2.0])}}
        d2 = {"logs": {"DEPT": np.array([1.0, 2.0 + 1e-8])}}
        assert compare_las_dicts(d1, d2) is True

    def test_list_comparison(self) -> None:
        """Test comparing lists."""
        d1 = {"curves_order": ["A", "B"]}
        d2 = {"curves_order": ["A", "B"]}
        assert compare_las_dicts(d1, d2) is True

    def test_list_mismatch(self) -> None:
        """Test comparing different lists."""
        d1 = {"curves_order": ["A", "B"]}
        d2 = {"curves_order": ["A", "C"]}
        assert compare_las_dicts(d1, d2) is False

    def test_nan_handling(self) -> None:
        """Test that NaN values are compared correctly."""
        d1 = {"logs": {"DEPT": np.array([1.0, np.nan, 3.0])}}
        d2 = {"logs": {"DEPT": np.array([1.0, np.nan, 3.0])}}
        assert compare_las_dicts(d1, d2) is True

    def test_key_in_first_not_in_second(self) -> None:
        """Test comparing dicts where dict1 has key not in dict2."""
        d1 = {"version": {"VERS": "2.0"}, "extra": "value"}
        d2 = {"version": {"VERS": "2.0"}}
        assert compare_las_dicts(d1, d2) is False

    # --- TEST-01: val2 is dict, val1 is NOT dict (line 49 uncovered branch) ---
    def test_val2_dict_val1_not_dict(self) -> None:
        """Test comparison when val2 is a dict but val1 is not (e.g., list).

        Exercises the branch where isinstance(val2, dict) is True but
        isinstance(val1, dict) is False at line 49 of compare.py.
        A list supports integer subscripting so the iteration works.
        """
        d1 = {"data": ["A", "B"]}
        d2 = {"data": {0: "A", 1: "B"}}
        assert compare_las_dicts(d1, d2) is False

        # Different values should return False
        d1["data"] = ["A", "C"]
        assert compare_las_dicts(d1, d2) is False

    # --- TEST-01: nested type mismatch (line 67-74) ---
    def test_nested_type_mismatch_array_vs_scalar(self) -> None:
        """Test detecting type mismatch in nested dict where val1 is ndarray
        but val2 is a scalar (line 67-74 of compare.py)."""
        d1 = {"logs": {"DEPT": np.array([1.0, 2.0])}}
        d2 = {"logs": {"DEPT": 5}}
        assert compare_las_dicts(d1, d2) is False

    # --- F-53: Non-standard value types at top level ---
    def test_scalar_values_comparison(self) -> None:
        """Test comparing dicts with scalar (int/float/str) top-level values.

        Exercises compare.py:89-92 — the else branch for values that are
        not dicts, ndarrays, or lists (scalar types).
        """
        d1 = {"a": 1, "b": 2.0, "c": "hello"}
        d2 = {"a": 1, "b": 2.0, "c": "hello"}
        assert compare_las_dicts(d1, d2) is True

    def test_scalar_values_mismatch(self) -> None:
        """Test scalar top-level value mismatch detected."""
        d1 = {"a": 1, "b": 2.0}
        d2 = {"a": 1, "b": 3.0}
        assert compare_las_dicts(d1, d2) is False

    def test_scalar_string_mismatch(self) -> None:
        """Test scalar string value mismatch detected."""
        d1 = {"name": "hello"}
        d2 = {"name": "world"}
        assert compare_las_dicts(d1, d2) is False

    # --- E-27: scalar leaf mismatches must log at WARNING (README contract) ---
    def test_scalar_mismatch_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """E-27: scalar leaf mismatch (units/version/well/descriptions) logs
        a WARNING with the offending key — README.md:166-168 contract that
        every mismatch path logs at WARNING level."""
        d1 = {"well": {"STRT": "100.0", "STOP": "200.0"}}
        d2 = {"well": {"STRT": "100.0", "STOP": "250.0"}}
        with caplog.at_level(logging.WARNING, logger="pylasdev.compare"):
            assert compare_las_dicts(d1, d2) is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "scalar mismatch must emit a WARNING record"
        assert any("well.STOP" in r.message for r in warnings), caplog.text

    def test_scalar_mismatch_nested_in_data_sections_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """E-27: scalar mismatch inside data_sections also warns."""
        d1 = {"data_sections": [{"name": "section_a"}]}
        d2 = {"data_sections": [{"name": "section_b"}]}
        with caplog.at_level(logging.WARNING, logger="pylasdev.compare"):
            assert compare_las_dicts(d1, d2) is False
        assert any(
            r.levelno == logging.WARNING and "section" in r.message
            for r in caplog.records
        ), caplog.text

    def test_scalar_match_emits_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """E-27: matching scalar leaves stay silent (no spurious warning)."""
        d1 = {"well": {"STRT": "100.0", "STOP": "200.0"}}
        d2 = {"well": {"STRT": "100.0", "STOP": "200.0"}}
        with caplog.at_level(logging.WARNING, logger="pylasdev.compare"):
            assert compare_las_dicts(d1, d2) is True
        assert not any(
            r.levelno == logging.WARNING for r in caplog.records
        ), caplog.text

    def test_scalar_mismatch_missing_key_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """E-27: missing scalar key warns (key-diff path) and is False.

        The nested-dict key-diff warning (compare.py:342, "Keys only in
        second dict") fires on the missing key and is asserted here — the
        E-27 contract that every mismatch path logs at WARNING level.
        """
        d1 = {"well": {"STRT": "100.0"}}
        d2 = {"well": {"STRT": "100.0", "STOP": "200.0"}}
        with caplog.at_level(logging.WARNING, logger="pylasdev.compare"):
            assert compare_las_dicts(d1, d2) is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "missing-key mismatch must emit a WARNING record"
        assert any("Keys only in second dict" in r.message for r in warnings), caplog.text

    # --- T1: _compare_data_sections coverage (compare.py:151-194) ---

    def test_compare_data_sections_match(self) -> None:
        """Test _compare_data_sections returns True for matching sections."""
        d1 = {
            "data_sections": [
                {"DEPT": np.array([1000.0, 1001.0]), "DT": np.array([50.0, 51.0])},
                {"GR": np.array([75.0, 76.0])},
            ],
        }
        d2 = {
            "data_sections": [
                {"DEPT": np.array([1000.0, 1001.0]), "DT": np.array([50.0, 51.0])},
                {"GR": np.array([75.0, 76.0])},
            ],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_compare_data_sections_length_mismatch(self) -> None:
        """Test _compare_data_sections returns False when section counts differ."""
        d1 = {"data_sections": [{"DEPT": np.array([1.0])}]}
        d2 = {"data_sections": [{"DEPT": np.array([1.0])}, {"DT": np.array([2.0])}]}
        assert compare_las_dicts(d1, d2) is False

    def test_compare_data_sections_key_mismatch(self) -> None:
        """Test _compare_data_sections returns False when dict keys differ."""
        d1 = {"data_sections": [{"DEPT": np.array([1.0]), "DT": np.array([2.0])}]}
        d2 = {"data_sections": [{"DEPT": np.array([1.0]), "GR": np.array([3.0])}]}
        assert compare_las_dicts(d1, d2) is False

    def test_compare_data_sections_ndarray_mismatch(self) -> None:
        """Test _compare_data_sections returns False when ndarray values differ."""
        d1 = {"data_sections": [{"DEPT": np.array([1.0, 2.0])}]}
        d2 = {"data_sections": [{"DEPT": np.array([1.0, 3.0])}]}
        assert compare_las_dicts(d1, d2) is False

    def test_compare_data_sections_list_mismatch(self) -> None:
        """Test _compare_data_sections returns False when list values differ."""
        d1 = {"data_sections": [{"order": ["A", "B"]}]}
        d2 = {"data_sections": [{"order": ["A", "C"]}]}
        assert compare_las_dicts(d1, d2) is False

    # --- T4: String/object array comparison (compare.py:129-131) ---

    def test_compare_string_arrays_match(self) -> None:
        """Test _compare_arrays handles Unicode string arrays correctly."""
        d1 = {"string_data": {"CDES": np.array(["SAND", "SHALE"], dtype="U")}}
        d2 = {"string_data": {"CDES": np.array(["SAND", "SHALE"], dtype="U")}}
        assert compare_las_dicts(d1, d2) is True

    def test_compare_string_arrays_mismatch(self) -> None:
        """Test _compare_arrays detects mismatch in string arrays."""
        d1 = {"string_data": {"CDES": np.array(["SAND", "SHALE"], dtype="U")}}
        d2 = {"string_data": {"CDES": np.array(["SAND", "LIMESTONE"], dtype="U")}}
        assert compare_las_dicts(d1, d2) is False

    def test_compare_object_arrays_match(self) -> None:
        """Test _compare_arrays handles object dtype arrays correctly."""
        d1 = {"meta": {"tags": np.array(["A", "B"], dtype="O")}}
        d2 = {"meta": {"tags": np.array(["A", "B"], dtype="O")}}
        assert compare_las_dicts(d1, d2) is True

    def test_compare_object_arrays_mismatch(self) -> None:
        """Test _compare_arrays detects mismatch in object arrays."""
        d1 = {"meta": {"tags": np.array(["A", "B"], dtype="O")}}
        d2 = {"meta": {"tags": np.array(["A", "C"], dtype="O")}}
        assert compare_las_dicts(d1, d2) is False

    # --- CF-025: dtype "S" (byte strings) branch test ---
    def test_compare_byte_string_arrays_match(self) -> None:
        """Test _compare_arrays handles byte string arrays (dtype 'S')."""
        d1 = {"logs": {"ID": np.array([b"DEPT", b"GR"], dtype="S")}}
        d2 = {"logs": {"ID": np.array([b"DEPT", b"GR"], dtype="S")}}
        assert compare_las_dicts(d1, d2) is True

    def test_compare_byte_string_arrays_mismatch(self) -> None:
        """Test _compare_arrays detects mismatch in byte string arrays."""
        d1 = {"logs": {"ID": np.array([b"DEPT", b"GR"], dtype="S")}}
        d2 = {"logs": {"ID": np.array([b"DEPT", b"DT"], dtype="S")}}
        assert compare_las_dicts(d1, d2) is False

    # --- R-005: Parametrized tolerance comparison ---
    @pytest.mark.parametrize(
        "d1,d2,atol,rtol,expected",
        [
            # Arrays within tolerance
            (
                {"logs": {"DEPT": np.array([1.0, 2.0])}},
                {"logs": {"DEPT": np.array([1.0, 2.0 + 1e-8])}},
                1e-7,
                1e-7,
                True,
            ),
            # Custom atol — passes with 0.02, fails with 0.001
            (
                {"logs": {"DEPT": np.array([1.0])}},
                {"logs": {"DEPT": np.array([1.01])}},
                0.02,
                1e-7,
                True,
            ),
            (
                {"logs": {"DEPT": np.array([1.0])}},
                {"logs": {"DEPT": np.array([1.01])}},
                0.001,
                1e-7,
                False,
            ),
            # rtol-based: ~1% relative difference
            (
                {"logs": {"VAL": np.array([100.0, 200.0])}},
                {"logs": {"VAL": np.array([101.0, 202.0])}},
                0.0,
                2e-2,
                True,
            ),
            (
                {"logs": {"VAL": np.array([100.0, 200.0])}},
                {"logs": {"VAL": np.array([101.0, 202.0])}},
                0.0,
                1e-3,
                False,
            ),
            # Both atol and rtol together
            (
                {"logs": {"V": np.array([1.0, 10.0])}},
                {"logs": {"V": np.array([1.01, 10.01])}},
                0.0,
                1e-7,
                False,
            ),
            # rtol=1e-2 gives ~1% tolerance (folded from the former
            # test_rtol_interaction_with_atol — the unique discriminating row)
            (
                {"logs": {"V": np.array([1.0, 10.0])}},
                {"logs": {"V": np.array([1.01, 10.01])}},
                0.0,
                1e-2,
                True,
            ),
            (
                {"logs": {"V": np.array([1.0, 10.0])}},
                {"logs": {"V": np.array([1.01, 10.01])}},
                0.02,
                1e-7,
                True,
            ),
        ],
    )
    def test_compare_tolerance_parametrized(
        self,
        d1: dict,
        d2: dict,
        atol: float,
        rtol: float,
        expected: bool,
    ) -> None:
        """Parametrized test for array comparison with tolerance values."""
        assert compare_las_dicts(d1, d2, atol=atol, rtol=rtol) is expected


class TestCompareDataSectionsNestedDict:
    """F7: Tests for _compare_data_sections isinstance(v2, dict) branch.

    When a data_sections entry has a dict value (e.g. LAS 3.0 DataSection.to_dict()
    structure where 'data' or 'string_data' maps curve names to ndarrays),
    the isinstance(v2, dict) branch at compare.py:222-290 is exercised.
    """

    def test_data_sections_nested_dict_match(self) -> None:
        """Test data_sections with nested dict values that match."""
        d1 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1000.0, 1001.0]), "DT": np.array([50.0, 51.0])},
                    "string_data": {},
                    "curves_order": ["DEPT", "DT"],
                    "name": "Section_0",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1000.0, 1001.0]), "DT": np.array([50.0, 51.0])},
                    "string_data": {},
                    "curves_order": ["DEPT", "DT"],
                    "name": "Section_0",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_data_sections_nested_dict_ndarray_mismatch(self) -> None:
        """Test data_sections nested dict ndarray value mismatch."""
        d1 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1000.0, 1001.0])},
                    "curves_order": ["DEPT"],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1000.0, 9999.0])},
                    "curves_order": ["DEPT"],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_key_mismatch(self) -> None:
        """Test data_sections nested dict with different keys."""
        d1 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0]), "DT": np.array([2.0])},
                    "curves_order": ["DEPT", "DT"],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0]), "GR": np.array([3.0])},
                    "curves_order": ["DEPT", "GR"],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_missing_inner_key(self) -> None:
        """Test data_sections nested dict where a key is missing in first."""
        d1 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0])},
                    "curves_order": ["DEPT"],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0]), "DT": np.array([2.0])},
                    "curves_order": ["DEPT", "DT"],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_type_mismatch(self) -> None:
        """Test data_sections nested dict where v1 is not a dict (type mismatch)."""
        d1 = {
            "data_sections": [
                {
                    "data": ["not", "a", "dict"],
                    "curves_order": [],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0])},
                    "curves_order": ["DEPT"],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_scalar_mismatch(self) -> None:
        """Test data_sections nested dict with scalar value mismatch."""
        d1 = {
            "data_sections": [
                {
                    "meta": {"version": "3.0", "source": "file1"},
                    "curves_order": [],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "meta": {"version": "3.0", "source": "file2"},
                    "curves_order": [],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_array_type_mismatch(self) -> None:
        """Test data_sections nested dict where v1 has array but v2 doesn't."""
        d1 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0]), "DT": np.array([2.0])},
                    "curves_order": ["DEPT", "DT"],
                    "name": "S1",
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {"DEPT": np.array([1.0]), "DT": 5},
                    "curves_order": ["DEPT", "DT"],
                    "name": "S1",
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False


class TestTypeErrorHandler:
    """Type-mismatch guards for scalar-vs-dict top-level values.

    All variants exercise the single-side-dict branch in _coerce_and_compare
    (compare.py:349-357): when one operand is a dict and the other is not,
    the comparison returns False with a "Type mismatch" warning.  The
    historical "TypeError handler at compare.py:104-112" mechanism no
    longer exists — the dispatch hub replaced the subscript-based
    fallthrough, so the operand types are all equivalent here.
    """

    @pytest.mark.parametrize(
        "scalar, mapping",
        [
            (42, {0: "value"}),
            (3.14, {0: "value"}),
            ("hello", {"key": "value"}),
            ("hello", {0: "hello"}),
            ("h", {0: "h"}),
            ("ab", {0: "a", 1: "c"}),
        ],
    )
    def test_scalar_vs_dict_mismatch(self, scalar: Any, mapping: dict) -> None:
        """Scalar vs dict at the same key returns False (single-side-dict)."""
        assert compare_las_dicts({"section": scalar}, {"section": mapping}) is False

    # --- F19: List with numpy arrays triggers ValueError/TypeError handler ---
    def test_numpy_array_in_list_comparison(self) -> None:
        """Test comparison when non-data_sections list contains numpy arrays.

        Exercises the numeric list path: both lists are homogeneous numeric
        (each converts to a 2x2 float64), so _compare_lists takes the
        _compare_arrays route and returns the correct verdict.  The
        historical "except (ValueError, TypeError)" fallback (compare.py:
        601-631) is NOT reached by this fixture — it only fires when
        _list_to_numeric_array returns None (ragged/non-numeric lists).
        """
        d1 = {"other_data": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        d2 = {"other_data": [np.array([1.0, 5.0]), np.array([3.0, 4.0])]}
        # Arrays at index 0 differ — should return False without crashing
        assert compare_las_dicts(d1, d2) is False

        # Matching arrays should return True
        d3 = {"other_data": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        assert compare_las_dicts(d1, d3) is True


class TestIntegrationCompareLasDicts:
    """F-T3-M05: Integration test using LASFile.to_dict() output.

    Verifies that compare_las_dicts works correctly with the actual
    dict produced by LASFile.to_dict() — the primary roundtrip format.
    """

    def test_compare_las_file_to_dict_self(self) -> None:
        """A LASFile compared to its own to_dict() output must match."""
        from pylasdev.models import LASFile, VersionSection

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.well["STRT"] = "100.0"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["DT"] = np.array([50.0, 51.0])

        d = las.to_dict()
        assert compare_las_dicts(d, d) is True, "Self-comparison must be True"

    def test_compare_las_file_to_dict_identical(self) -> None:
        """Two identical LASFile objects produce matching dicts."""
        from pylasdev.models import LASFile, VersionSection

        las1 = LASFile()
        las1.version = VersionSection(vers="2.0")
        las1.well["NULL"] = "-999.25"
        las1.curves_order = ["DEPT"]
        las1.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las1.logs["DEPT"] = np.array([1.0, 2.0])

        las2 = LASFile()
        las2.version = VersionSection(vers="2.0")
        las2.well["NULL"] = "-999.25"
        las2.curves_order = ["DEPT"]
        las2.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las2.logs["DEPT"] = np.array([1.0, 2.0])

        assert compare_las_dicts(las1.to_dict(), las2.to_dict()) is True

    def test_compare_las_file_to_dict_mismatch(self) -> None:
        """Different LASFile objects produce non-matching dicts."""
        from pylasdev.models import LASFile, VersionSection

        las1 = LASFile()
        las1.version = VersionSection(vers="2.0")
        las1.well["NULL"] = "-999.25"
        las1.curves_order = ["DEPT"]
        las1.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las1.logs["DEPT"] = np.array([1.0, 2.0])

        las2 = LASFile()
        las2.version = VersionSection(vers="1.2")
        las2.well["NULL"] = "-99.0"
        las2.curves_order = ["DEPT"]
        las2.curves.append(CurveDefinition(mnemonic="DEPT", unit="FT"))
        las2.logs["DEPT"] = np.array([1.0, 3.0])

        assert compare_las_dicts(las1.to_dict(), las2.to_dict()) is False


class TestProductionCheckCompareFix:
    """Regression test for F-005 fix in compare.py."""

    def test_nan_guard_covers_all_float_dtypes(self) -> None:
        """F-005: arrays containing NaN of mixed float dtypes compare equal.

        The np.float32/np.float16 elements are promoted to float64 when the
        array is constructed, so these fixtures exercise the ARRAY path
        (_compare_arrays -> _allclose_symmetric), where NaN matches NaN via
        the equal_nan clause (compare.py:186) — NOT the _scalars_equal
        np.floating guard (compare.py:43), which sees 0 calls here.  The
        genuine bare-scalar pin of the guard is
        test_nan_guard_bare_scalar_np_floating below.
        """
        d1 = {"logs": {"VAL": np.array([1.0, np.float32(np.nan), 3.0])}}
        d2 = {"logs": {"VAL": np.array([1.0, np.float32(np.nan), 3.0])}}
        assert compare_las_dicts(d1, d2) is True, "NaN np.float32 comparison must match"

    def test_nan_float16_guard(self) -> None:
        """F-005: arrays containing np.float16 NaN compare equal.

        Same float64-promotion + equal_nan array path as
        test_nan_guard_covers_all_float_dtypes: the np.float16 element is
        erased at array construction, so this guards NaN-in-array handling
        (compare.py:186), not the _scalars_equal np.floating guard
        (compare.py:43).
        """
        d1 = {"logs": {"VAL": np.array([1.0, np.float16(np.nan)])}}
        d2 = {"logs": {"VAL": np.array([1.0, np.float16(np.nan)])}}
        assert compare_las_dicts(d1, d2) is True, "NaN np.float16 comparison must match"

    def test_nan_different_float_dtypes_both_nan(self) -> None:
        """F-005: NaN values of different float dtypes still match as equal."""
        # np.float32(nan) and np.float64(nan) should both be treated as NaN=NaN
        d1 = {"logs": {"VAL": np.array([np.float32(np.nan)])}}
        d2 = {"logs": {"VAL": np.array([np.float64(np.nan)])}}
        # Both are NaN → comparison should return True (NaN==NaN via per-element path)
        assert compare_las_dicts(d1, d2) is True

    def test_nan_guard_bare_scalar_np_floating(self) -> None:
        """F-005/F-26: bare np.floating NaN scalars hit the guard directly.

        The array-form tests above route through _allclose_symmetric's
        equal_nan path (compare.py:186) and stay green even if the
        _scalars_equal np.floating guard at compare.py:43 is removed.  A
        bare np.float32/np.float16/np.longdouble NaN dict value exercises
        that guard directly: pre-fix (isinstance(x, float) only, which
        np.floating subtypes fail) it compared False; post-fix True.
        """
        assert compare_las_dicts({"x": np.float32(np.nan)}, {"x": np.float32(np.nan)}) is True
        assert compare_las_dicts({"x": np.float16(np.nan)}, {"x": np.float16(np.nan)}) is True
        assert compare_las_dicts({"x": np.longdouble(np.nan)}, {"x": np.longdouble(np.nan)}) is True


# ──────────────────────────────────────────────────────────────
# M-01 Regression: list-of-arrays dispatch branches
# ──────────────────────────────────────────────────────────────


class TestF01HListOfArraysNestedDict:
    """F-01-H: Regression tests for isinstance(val2[in_key], list) branch
    in compare_las_dicts (compare.py:168-189).

    Without this branch, nested dicts containing list-of-numpy-arrays
    values would fall through to _scalars_equal, which raises ValueError
    on list-equality-with-ndarray-values (ambiguous truth value), causing
    identical dicts to return False.
    """

    def test_identical_nested_dict_with_list_of_arrays(self) -> None:
        """Identical nested dict with list-of-arrays values returns True."""
        arr = np.array([1.0, 2.0, 3.0])
        d1 = {
            "sections": {
                "data": {"curves": [arr, arr]},
            },
        }
        d2 = {
            "sections": {
                "data": {"curves": [arr, arr]},
            },
        }
        assert compare_las_dicts(d1, d2) is True

    def test_nested_dict_list_of_arrays_different_values(self) -> None:
        """Nested dict with list-of-arrays where arrays differ returns False."""
        d1 = {
            "sections": {
                "data": {"curves": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]},
            },
        }
        d2 = {
            "sections": {
                "data": {"curves": [np.array([1.0, 2.0]), np.array([3.0, 5.0])]},
            },
        }
        assert compare_las_dicts(d1, d2) is False

    def test_nested_dict_list_of_arrays_single_element(self) -> None:
        """List-of-arrays with a single element matches."""
        d1 = {
            "data": {
                "channels": [np.array([1.0, 2.0, 3.0])],
            },
        }
        d2 = {
            "data": {
                "channels": [np.array([1.0, 2.0, 3.0])],
            },
        }
        assert compare_las_dicts(d1, d2) is True

    def test_nested_dict_list_of_arrays_empty(self) -> None:
        """Empty list-of-arrays values match."""
        d1 = {
            "data": {
                "channels": [],
            },
        }
        d2 = {
            "data": {
                "channels": [],
            },
        }
        assert compare_las_dicts(d1, d2) is True

    def test_nested_dict_list_of_arrays_length_mismatch(self) -> None:
        """Lists of different lengths inside nested dict return False."""
        d1 = {
            "data": {
                "curves": [np.array([1.0, 2.0])],
            },
        }
        d2 = {
            "data": {
                "curves": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            },
        }
        assert compare_las_dicts(d1, d2) is False

    def test_nested_dict_val1_not_list_type_mismatch(self) -> None:
        """When val1 is not a list but val2 is, returns False with warning."""
        d1 = {
            "data": {
                "channels": np.array([1.0, 2.0]),  # ndarray, not list
            },
        }
        d2 = {
            "data": {
                "channels": [np.array([1.0, 2.0])],  # list
            },
        }
        assert compare_las_dicts(d1, d2) is False

    def test_nested_dict_list_of_arrays_multiple_keys(self) -> None:
        """Nested dict with multiple keys, one containing list-of-arrays."""
        d1 = {
            "meta": {
                "curves": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                "name": "section_a",
            },
        }
        d2 = {
            "meta": {
                "curves": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                "name": "section_a",
            },
        }
        assert compare_las_dicts(d1, d2) is True

    def test_nested_dict_list_of_arrays_with_nan(self) -> None:
        """List-of-arrays containing NaN values match when identical."""
        d1 = {
            "data": {
                "signals": [np.array([1.0, np.nan, 3.0]), np.array([np.nan, 5.0])],
            },
        }
        d2 = {
            "data": {
                "signals": [np.array([1.0, np.nan, 3.0]), np.array([np.nan, 5.0])],
            },
        }
        assert compare_las_dicts(d1, d2) is True


class TestF43DataSectionsListOfArrays:
    """F-43: Regression tests for isinstance(v2[in_key], list) branch
    in _compare_data_sections (compare.py:573-597).

    When data_sections entries contain nested dicts with list values,
    the list dispatch branch handles list-of-ndarray values at depth 3+
    inside the data_sections comparison loop.
    """

    def test_data_sections_nested_dict_list_of_arrays_match(self) -> None:
        """Data section nested dict with list-of-arrays values matches."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "channels": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "channels": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_data_sections_nested_dict_list_different_arrays(self) -> None:
        """Data section list-of-arrays with different values returns False."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "signals": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "signals": [np.array([1.0, 2.0]), np.array([9.0, 10.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_list_single_element(self) -> None:
        """Data section list-of-arrays with single element matches."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "trace": [np.array([10.0, 20.0, 30.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "trace": [np.array([10.0, 20.0, 30.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_data_sections_nested_dict_list_empty(self) -> None:
        """Data section with empty list inside nested dict matches."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "channels": [],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "channels": [],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is True

    def test_data_sections_nested_dict_list_length_mismatch(self) -> None:
        """Data section list-of-arrays with different lengths returns False."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "curves": [np.array([1.0, 2.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "curves": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_val1_not_list(self) -> None:
        """Data section: val1 is ndarray but val2 is list → type mismatch."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "values": np.array([1.0, 2.0]),
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "values": [np.array([1.0, 2.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is False

    def test_data_sections_nested_dict_list_with_nan(self) -> None:
        """Data section list-of-arrays with NaN values matches."""
        d1 = {
            "data_sections": [
                {
                    "data": {
                        "readings": [np.array([np.nan, 2.0, 3.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        d2 = {
            "data_sections": [
                {
                    "data": {
                        "readings": [np.array([np.nan, 2.0, 3.0])],
                    },
                    "name": "S1",
                    "curves_order": [],
                },
            ],
        }
        assert compare_las_dicts(d1, d2) is True


class TestCompareListsDirect:
    """Direct tests for _compare_lists() (compare.py:374-432).

    _compare_lists had zero test coverage before M-01. These tests
    exercise it directly with edge cases including empty lists,
    single elements, nested lists, and numpy arrays in lists.
    """

    def test_empty_lists_match(self) -> None:
        """Two empty lists are equal."""
        assert _compare_lists([], [], "test", 1e-7, 0.0) is True

    def test_single_element_lists_match(self) -> None:
        """Single-element lists with identical scalars match."""
        assert _compare_lists([1], [1], "test", 1e-7, 0.0) is True

    def test_different_lengths(self) -> None:
        """Lists of different lengths return False."""
        assert _compare_lists([1, 2], [1], "test", 1e-7, 0.0) is False

    def test_identical_list_of_arrays(self) -> None:
        """Lists containing identical numpy arrays match."""
        assert (
            _compare_lists(
                [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                "test",
                1e-7,
                0.0,
            )
            is True
        )

    def test_different_arrays_in_list(self) -> None:
        """Lists containing different numpy arrays return False."""
        assert (
            _compare_lists(
                [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                [np.array([1.0, 2.0]), np.array([3.0, 5.0])],
                "test",
                1e-7,
                0.0,
            )
            is False
        )

    def test_list_of_arrays_within_tolerance(self) -> None:
        """List of arrays matches within floating-point tolerance."""
        assert (
            _compare_lists(
                [np.array([1.0, 2.0])],
                [np.array([1.0, 2.0 + 1e-12])],
                "test",
                1e-7,
                0.0,
            )
            is True
        )

    def test_list_scalar_elements(self) -> None:
        """Lists of scalars match via _compare_values fallthrough."""
        assert (
            _compare_lists(
                ["a", "b", "c"],
                ["a", "b", "c"],
                "test",
                1e-7,
                0.0,
            )
            is True
        )

    def test_list_scalar_elements_mismatch(self) -> None:
        """Lists of scalars with different values return False."""
        assert (
            _compare_lists(
                ["a", "b", "c"],
                ["a", "x", "c"],
                "test",
                1e-7,
                0.0,
            )
            is False
        )

    def test_list_with_nan_values(self) -> None:
        """Lists containing NaN float values match when identical."""
        assert (
            _compare_lists(
                [np.nan, 2.0],
                [np.nan, 2.0],
                "test",
                1e-7,
                0.0,
            )
            is True
        )

    def test_mixed_types_in_list(self) -> None:
        """Lists with mixed scalar types match."""
        assert (
            _compare_lists(
                [1, "hello", 3.14],
                [1, "hello", 3.14],
                "test",
                1e-7,
                0.0,
            )
            is True
        )

    def test_empty_vs_nonempty_list(self) -> None:
        """Empty list vs non-empty list returns False."""
        assert _compare_lists([], [np.array([1.0])], "test", 1e-7, 0.0) is False

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ([np.inf], [np.inf], True),
            ([-np.inf], [-np.inf], True),
            ([np.inf], [-np.inf], False),
            ([np.inf], [1.0], False),
        ],
    )
    def test_inf_handling(self, left: list, right: list, expected: bool) -> None:
        """+inf == +inf and -inf == -inf, but inf never matches the opposite
        sign or a finite value.

        Guards the equal_pos_inf/equal_neg_inf branches in _allclose_symmetric
        (compare.py:187-188): the non-finite exclusion mask excludes inf from
        the rtol/atol tolerance check, so without those branches inf operands
        would always compare unequal.
        """
        assert _compare_lists(left, right, "test", 1e-7, 0.0) is expected


class TestMaskedArrayComparison:
    """Tests for 0-d (scalar) MaskedArray comparison — F-035.

    The 0-d MaskedArray -> .item() coercion path in _coerce_and_compare
    (compare.py:147-156) had zero test coverage.  These tests exercise
    that path via _compare_lists (which calls _coerce_and_compare for
    each element).
    """

    def test_0d_masked_scalar_vs_regular_scalar(self) -> None:
        """0-d MaskedArray with mask=False vs regular scalar matches."""
        ma_val = np.ma.array(42.0)
        assert not np.ma.is_masked(ma_val), "Precondition: mask should be False"
        # Single-element list triggers _coerce_and_compare per element
        assert _compare_lists([ma_val], [42.0], "test", 1e-7, 0.0) is True

    def test_0d_masked_true_vs_regular_scalar(self) -> None:
        """0-d MaskedArray with mask=True vs regular scalar returns False.

        The mask check in compare.py:148 converts a masked 0-d array to
        np.ma.masked, which causes a type mismatch and returns False.
        """
        ma_val = np.ma.array(42.0, mask=True)
        assert np.ma.is_masked(ma_val), "Precondition: mask should be True"
        assert _compare_lists([ma_val], [42.0], "test", 1e-7, 0.0) is False

    def test_0d_masked_vs_same_masked(self) -> None:
        """0-d MaskedArray vs same 0-d MaskedArray matches."""
        ma_val = np.ma.array(3.14)
        assert _compare_lists([ma_val], [np.ma.array(3.14)], "test", 1e-7, 0.0) is True

    def test_0d_masked_vs_different_masked(self) -> None:
        """0-d MaskedArray vs different value 0-d MaskedArray returns False."""
        ma_val = np.ma.array(3.14)
        assert _compare_lists([ma_val], [np.ma.array(2.72)], "test", 1e-7, 0.0) is False

    def test_0d_masked_vs_regular_scalar_mask_preserved(self) -> None:
        """0-d MaskedArray mask is preserved through nesting in lists."""
        # Two-element list with masked and non-masked values
        ma_masked = np.ma.array(42.0, mask=True)
        ma_normal = np.ma.array(3.14)
        assert (
            _compare_lists(
                [ma_normal, ma_masked], [np.ma.array(3.14), ma_masked], "test", 1e-7, 0.0
            )
            is True
        )

    def test_0d_regular_ndarray_scalar_vs_scalar(self) -> None:
        """0-d regular ndarray vs Python scalar — .item() path still works."""
        arr = np.array(42.0)  # 0-d ndarray, NOT MaskedArray
        assert not isinstance(arr, np.ma.MaskedArray)
        assert _compare_lists([arr], [42.0], "test", 1e-7, 0.0) is True


class TestE08DataSectionsSymmetricGuards:
    """E-08: _compare_data_sections must handle malformed sections2 operands.

    Previously only sections1 was checked for list-ness, so sections2=None
    raised a bare TypeError on len(), a non-dict element raised a bare
    AttributeError on .keys(), and empty-container mismatches ([] == {},
    [] == (), [] == "") silently compared equal.  All cases must now
    return False (or True for genuinely equal operands) without raising.
    """

    @pytest.mark.parametrize(
        "wrong",
        [None, {}, (), ""],
    )
    def test_sections2_wrong_type_returns_false(self, wrong: Any) -> None:
        """E-08: malformed sections2 (non-list) returns False, not True/raise.

        The empty-container silent FALSE-EQUAL: [] == {} previously
        returned True because both had length 0 and the loop never ran;
        sections2=None raised a bare TypeError on len().  All four
        wrong-type variants hit the identical ``not isinstance(sections2,
        list)`` guard (compare.py:657-663).
        """
        assert compare_las_dicts({"data_sections": []}, {"data_sections": wrong}) is False

    @pytest.mark.parametrize(
        "sections1, sections2",
        [
            ({"data_sections": [{"a": 1}]}, {"data_sections": ["notadict"]}),
            ({"data_sections": ["notadict"]}, {"data_sections": [{"a": 1}]}),
        ],
    )
    def test_non_dict_element_returns_false(self, sections1: dict, sections2: dict) -> None:
        """E-08: non-dict element returns False instead of AttributeError.

        The per-element dict guard (compare.py:675-682) fires for either
        operand order.
        """
        assert compare_las_dicts(sections1, sections2) is False

    def test_matching_sections_still_equal(self) -> None:
        """E-08: well-formed matching data_sections still compare equal."""
        d1 = {"data_sections": [{"DEPT": np.array([1.0, 2.0])}]}
        d2 = {"data_sections": [{"DEPT": np.array([1.0, 2.0])}]}
        assert compare_las_dicts(d1, d2) is True


class TestListToNumericArrayIntDtype:
    """F-07: _list_to_numeric_array must preserve integer precision.

    Pre-fix every homogeneous numeric list was converted to float64,
    which cannot represent int64 values above 2^53.  All-integer lists
    now become int64 (exact); a float element forces float64.
    """

    def test_all_int_list_converts_to_int64(self) -> None:
        from pylasdev.compare import _list_to_numeric_array

        arr = _list_to_numeric_array([2**53, 2**53 + 1])
        assert arr is not None
        assert arr.dtype.kind == "i"
        assert arr.tolist() == [2**53, 2**53 + 1]

    def test_mixed_int_float_list_stays_float64(self) -> None:
        from pylasdev.compare import _list_to_numeric_array

        arr = _list_to_numeric_array([1, 2.5])
        assert arr is not None
        assert arr.dtype.kind == "f"

    def test_int_list_overflowing_int64_falls_back(self) -> None:
        """Python ints beyond int64 range can't be exact in int64; the
        list falls back to element-wise comparison instead of crashing."""
        from pylasdev.compare import _list_to_numeric_array

        assert _list_to_numeric_array([2**70]) is None
        assert compare_las_dicts({"x": [2**70]}, {"x": [2**70]}) is True
        assert compare_las_dicts({"x": [2**70]}, {"x": [2**70 + 1]}) is False


class TestListToNumericMasked:
    """M2: _list_to_numeric_masked preserves list-item masks so the
    list path compares masked positions by mask (like the array path)
    instead of NaN-filling them (which conflates masked with NaN)."""

    def test_no_masked_items_returns_none(self) -> None:
        from pylasdev.compare import _list_to_numeric_masked

        assert _list_to_numeric_masked([1.0, 2.0]) is None

    def test_masked_item_preserves_mask(self) -> None:
        from pylasdev.compare import _list_to_numeric_masked

        ma = _list_to_numeric_masked([1.0, np.ma.array(2.0, mask=True)])
        assert ma is not None
        assert np.ma.is_masked(ma[1])
        assert not np.ma.is_masked(ma[0])
        # The data value is preserved (not NaN-filled) so unmasked
        # positions keep full precision (F-07).
        assert ma[0] == 1.0

    def test_masked_int_item_keeps_integer_dtype(self) -> None:
        from pylasdev.compare import _list_to_numeric_masked

        ma = _list_to_numeric_masked([np.ma.array(5, mask=True), 3])
        assert ma is not None
        assert ma.dtype.kind == "i"
        assert np.ma.is_masked(ma[0])

    def test_non_numeric_item_returns_none(self) -> None:
        from pylasdev.compare import _list_to_numeric_masked

        assert _list_to_numeric_masked([np.ma.array(2.0, mask=True), "x"]) is None

    def test_compare_lists_masked_vs_nan_false(self) -> None:
        """The list path agrees with the array path: a masked position
        never matches a NaN (pre-fix this returned True via NaN-fill)."""
        assert _compare_lists([np.ma.array(2.0, mask=True)], [np.nan], "test", 1e-7, 0.0) is False


class TestMaskedNonNumericArrayComparison:
    """E-28: masked NON-NUMERIC arrays must follow the mask-unwrap contract.

    Pre-fix, the non-numeric branch of _compare_arrays compared with
    np.array_equal directly, whose internal asarray() strips masks — so
    masked positions were compared by their raw data, inverting the
    documented mask semantics (a masked position matches only another
    masked position; compare.py _allclose_symmetric docstring) in three
    directions: masked-vs-masked with different data compared UNEQUAL,
    masked-vs-unmasked with same data compared EQUAL, and differing mask
    patterns compared EQUAL.  The fix applies the mask-unwrap to the
    non-numeric branch too, covering all dtype classes.
    """

    _STRUCTURED_DTYPE: ClassVar[np.dtype] = np.dtype([("a", "f8"), ("b", "i4")])

    # (v0, v1, w0, w1): base pair + alternative pair per dtype class.
    _CASES: ClassVar[dict[str, tuple[Any, Any, Any, Any]]] = {
        "U": ("SAND", "SHALE", "SILT", "LIME"),
        "S": (b"SAND", b"SHALE", b"SILT", b"LIME"),
        "V": (
            np.array((1.0, 2), dtype=_STRUCTURED_DTYPE),
            np.array((3.0, 4), dtype=_STRUCTURED_DTYPE),
            np.array((9.0, 9), dtype=_STRUCTURED_DTYPE),
            np.array((5.0, 6), dtype=_STRUCTURED_DTYPE),
        ),
        "O": (
            np.array("SAND", dtype=object),
            np.array("SHALE", dtype=object),
            np.array("SILT", dtype=object),
            np.array("LIME", dtype=object),
        ),
        "b": (True, False, False, True),
        "M": (
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-04"),
        ),
        "m": (
            np.timedelta64(1, "s"),
            np.timedelta64(2, "s"),
            np.timedelta64(3, "s"),
            np.timedelta64(4, "s"),
        ),
        "c": (1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j),
    }

    _DTYPES: ClassVar[list[str]] = ["U", "S", "O", "V", "b", "M", "m", "c"]

    @staticmethod
    def _ma(dtype: str, v0: Any, v1: Any, mask: list[bool]) -> np.ma.MaskedArray:
        """Build a 2-element masked array of the given dtype class."""
        return np.ma.array([v0, v1], mask=mask)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_masked_position_matches_masked_position(self, dtype: str) -> None:
        """E-28 inversion 1: data differing ONLY at masked positions is EQUAL.

        Pre-fix np.array_equal compared the raw data and returned False."""
        v0, v1, _w0, w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [False, True])
        b = self._ma(dtype, v0, w1, [False, True])
        assert compare_las_dicts({"x": a}, {"x": b}) is True

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_mask_mismatch_unequal(self, dtype: str) -> None:
        """E-28 inversion 2: same data with different mask patterns is UNEQUAL.

        Pre-fix np.array_equal stripped the masks and returned True."""
        v0, v1, _w0, _w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [False, True])
        b = self._ma(dtype, v0, v1, [True, False])
        assert compare_las_dicts({"x": a}, {"x": b}) is False

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_masked_vs_unmasked_same_data_unequal(self, dtype: str) -> None:
        """E-28 inversion 3: masked position never matches an unmasked value.

        A MaskedArray with a masked position vs a plain ndarray with the
        same data is UNEQUAL (pre-fix: True)."""
        v0, v1, _w0, _w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [False, True])
        plain = np.array([v0, v1])
        assert compare_las_dicts({"x": a}, {"x": plain}) is False

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_unmasked_position_mismatch_still_unequal(self, dtype: str) -> None:
        """E-28 guard: data differing at an UNMASKED position stays unequal."""
        v0, v1, w0, _w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [False, True])
        b = self._ma(dtype, w0, v1, [False, True])
        assert compare_las_dicts({"x": a}, {"x": b}) is False

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_all_masked_equal_regardless_of_data(self, dtype: str) -> None:
        """E-28: fully-masked arrays are EQUAL even with different data."""
        v0, v1, w0, w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [True, True])
        b = self._ma(dtype, w0, w1, [True, True])
        assert compare_las_dicts({"x": a}, {"x": b}) is True

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_identical_masked_arrays_equal(self, dtype: str) -> None:
        """E-28 guard: identical masked arrays (mask + data) still equal."""
        v0, v1, _w0, _w1 = self._CASES[dtype]
        a = self._ma(dtype, v0, v1, [False, True])
        b = self._ma(dtype, v0, v1, [False, True])
        assert compare_las_dicts({"x": a}, {"x": b}) is True

    def test_mask_mismatch_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """E-28: mask mismatch logs the dedicated warning."""
        a = np.ma.array(["SAND", "SHALE"], mask=[False, True])
        b = np.ma.array(["SAND", "SHALE"], mask=[True, False])
        with caplog.at_level(logging.WARNING, logger="pylasdev.compare"):
            assert compare_las_dicts({"x": a}, {"x": b}) is False
        assert any("mask mismatch" in r.message for r in caplog.records), caplog.text

    def test_structured_masked_vs_masked_never_raises(self) -> None:
        """H-02: V-kind masked vs V-kind masked returns a verdict, never raises.

        Pre-fix, equal masks reached ``~mask`` on a structured boolean
        mask and raised ``TypeError: ufunc 'invert' not supported``.
        Data differing only at a masked position still compares EQUAL
        (mask contract); differing mask patterns are a structural
        mismatch, not a crash."""
        v0, v1, _w0, w1 = self._CASES["V"]
        a = self._ma("V", v0, v1, [False, True])
        b = self._ma("V", v0, w1, [False, True])
        assert compare_las_dicts({"x": a}, {"x": b}) is True
        c = self._ma("V", v0, v1, [True, False])
        assert compare_las_dicts({"x": a}, {"x": c}) is False

    def test_structured_masked_vs_plain_never_raises(self) -> None:
        """H-02: V-kind masked vs plain returns a verdict, never raises.

        Pre-fix, ``np.array_equal`` compared the structured mask against
        a plain boolean zeros array and raised ``TypeError: Cannot
        compare structured or void to non-void arrays``."""
        v0, v1, _w0, _w1 = self._CASES["V"]
        a = self._ma("V", v0, v1, [False, True])
        plain = np.array([v0, v1])
        assert compare_las_dicts({"x": a}, {"x": plain}) is False
        assert compare_las_dicts({"x": plain}, {"x": a}) is False

    def test_structured_plain_vs_plain_never_raises(self) -> None:
        """H-02: V-kind plain vs plain returns a verdict, never raises."""
        v0, v1, _w0, w1 = self._CASES["V"]
        same = np.array([v0, v1])
        other = np.array([v0, w1])
        assert compare_las_dicts({"x": same}, {"x": same}) is True
        assert compare_las_dicts({"x": same}, {"x": other}) is False


class TestZeroDimNumericTolerance:
    """N-11: 0-d numeric ndarrays must use rtol/atol like 1-d arrays.

    Pre-fix, Phase 1 coerced 0-d ndarrays to Python scalars before the
    type dispatch, so 0-d-vs-0-d fell into exact scalar equality and
    bypassed rtol/atol — 0-d 1.0 vs 1.0005 @ rtol=1e-3 was False while
    [1.0] vs [1.0005] was True.  Both operands being 0-d now stay on
    the array path, giving identical verdicts to the 1-d representation.
    """

    def test_0d_within_default_tolerance_matches_1d(self) -> None:
        """0-d and 1-d both True at default tolerances for a tiny diff."""
        d_0d = {"logs": {"DEPT": np.array(1.0)}}
        d_1d = {"logs": {"DEPT": np.array([1.0])}}
        o_0d = {"logs": {"DEPT": np.array(1.0 + 1e-9)}}
        o_1d = {"logs": {"DEPT": np.array([1.0 + 1e-9])}}
        assert compare_las_dicts(d_0d, o_0d) is True
        assert compare_las_dicts(d_1d, o_1d) is True

    def test_0d_beyond_default_tolerance_matches_1d(self) -> None:
        """0-d and 1-d both False at default tolerances for a 0.05% diff."""
        d_0d = {"logs": {"DEPT": np.array(1.0)}}
        d_1d = {"logs": {"DEPT": np.array([1.0])}}
        o_0d = {"logs": {"DEPT": np.array(1.0005)}}
        o_1d = {"logs": {"DEPT": np.array([1.0005])}}
        assert compare_las_dicts(d_0d, o_0d) is False
        assert compare_las_dicts(d_1d, o_1d) is False
        # Custom rtol widens both representations identically.
        assert compare_las_dicts(d_0d, o_0d, rtol=1e-3) is True
        assert compare_las_dicts(d_1d, o_1d, rtol=1e-3) is True

    def test_0d_integer_arrays_stay_exact(self) -> None:
        """N-11 guard: integer 0-d arrays keep exact comparison."""
        assert compare_las_dicts({"x": np.array(5)}, {"x": np.array(5)}) is True
        assert compare_las_dicts({"x": np.array(5)}, {"x": np.array(6)}) is False

    def test_0d_masked_vs_masked_equal(self) -> None:
        """N-11 guard: masked 0-d vs masked 0-d still equals (masked == masked)."""
        assert (
            compare_las_dicts(
                {"x": np.ma.array(1.0, mask=True)}, {"x": np.ma.array(2.0, mask=True)}
            )
            is True
        )

    def test_0d_vs_scalar_still_exact(self) -> None:
        """N-11 guard (F-42): 0-d vs plain scalar keeps the scalar path."""
        assert compare_las_dicts({"x": np.array(42.0)}, {"x": 42.0}) is True
        assert compare_las_dicts({"x": np.array(42.0)}, {"x": 42.5}) is False
