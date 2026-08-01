"""Tests for LAS data comparison utilities."""

from __future__ import annotations

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

    def test_missing_nested_key(self) -> None:
        """Test comparing dicts where nested key is missing."""
        d1 = {"well": {"STRT": "100"}}
        d2 = {"well": {"STRT": "100", "STOP": "200"}}
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

    def test_custom_tolerance(self) -> None:
        """Test comparing with custom tolerance."""
        d1 = {"logs": {"DEPT": np.array([1.0])}}
        d2 = {"logs": {"DEPT": np.array([1.01])}}
        assert compare_las_dicts(d1, d2, atol=0.02) is True
        assert compare_las_dicts(d1, d2, atol=0.001) is False

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

    def test_empty_dicts(self) -> None:
        """Test comparing empty dicts."""
        assert compare_las_dicts({}, {}) is True

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

    def test_val2_dict_val1_not_dict_mismatch(self) -> None:
        """Test val2-dict vs val1-non-dict mismatch detected as inequality."""
        d1 = {"data": ["A"]}
        d2 = {"data": {0: "B"}}
        assert compare_las_dicts(d1, d2) is False

    # --- TEST-01: nested type mismatch (line 67-74) ---
    def test_nested_type_mismatch_array_vs_scalar(self) -> None:
        """Test detecting type mismatch in nested dict where val1 is ndarray
        but val2 is a scalar (line 67-74 of compare.py)."""
        d1 = {"logs": {"DEPT": np.array([1.0, 2.0])}}
        d2 = {"logs": {"DEPT": 5}}
        assert compare_las_dicts(d1, d2) is False

    def test_nested_type_mismatch_scalar_vs_array(self) -> None:
        """Test detecting type mismatch where val2 value is ndarray
        but val1 value is not (line 64-65 of compare.py).

        The _compare_arrays guard now handles non-ndarray arguments
        gracefully by logging a type mismatch and returning False.
        """
        d1 = {"logs": {"DEPT": 5}}
        d2 = {"logs": {"DEPT": np.array([1.0, 2.0])}}
        assert compare_las_dicts(d1, d2) is False

    # --- F-52: Top-level ndarray comparison ---
    def test_top_level_ndarray_comparison(self) -> None:
        """Test comparing dicts containing ndarray values at the top level.

        Exercises compare.py:81-83 — the isinstance(val2, np.ndarray) branch
        for top-level (non-nested) ndarray values.
        """
        d1 = {"data": np.array([1.0, 2.0, 3.0])}
        d2 = {"data": np.array([1.0, 2.0, 3.0])}
        assert compare_las_dicts(d1, d2) is True

    def test_top_level_ndarray_different_values(self) -> None:
        """Test top-level ndarray comparison detects different values."""
        d1 = {"data": np.array([1.0, 2.0, 3.0])}
        d2 = {"data": np.array([1.0, 2.0, 4.0])}
        assert compare_las_dicts(d1, d2) is False

    def test_top_level_ndarray_size_mismatch(self) -> None:
        """Test top-level ndarray comparison detects size mismatch."""
        d1 = {"data": np.array([1.0, 2.0])}
        d2 = {"data": np.array([1.0, 2.0, 3.0])}
        assert compare_las_dicts(d1, d2) is False

    def test_top_level_ndarray_type_mismatch(self) -> None:
        """Test top-level ndarray vs scalar type mismatch."""
        d1 = {"data": np.array([1.0])}
        d2 = {"data": 5}
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

    def test_scalar_missing_key_in_first(self) -> None:
        """Test missing key when comparing scalar-valued dicts."""
        d1 = {"a": 1}
        d2 = {"a": 1, "b": 2}
        assert compare_las_dicts(d1, d2) is False

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

    def test_compare_data_sections_scalar_mismatch(self) -> None:
        """Test _compare_data_sections returns False when scalar values differ."""
        d1 = {"data_sections": [{"name": "section_a"}]}
        d2 = {"data_sections": [{"name": "section_b"}]}
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

    # --- CF-026: rtol parameter with custom value ---
    def test_rtol_custom_tolerance(self) -> None:
        """Test rtol parameter with a custom value different from atol."""
        # Values with ~1% relative difference
        d1 = {"logs": {"VAL": np.array([100.0, 200.0])}}
        d2 = {"logs": {"VAL": np.array([101.0, 202.0])}}
        # rtol=1e-7 (default) should fail — 1% difference > 1e-7
        assert compare_las_dicts(d1, d2, rtol=1e-7) is False
        # rtol=1e-3 should also fail — 101/100 - 1 = 0.01 > 1e-3
        assert compare_las_dicts(d1, d2, rtol=1e-3) is False
        # rtol=0.02 should pass — 0.01 < 0.02
        assert compare_las_dicts(d1, d2, rtol=2e-2) is True

    def test_rtol_interaction_with_atol(self) -> None:
        """Test that rtol and atol work together correctly."""
        d1 = {"logs": {"V": np.array([1.0, 10.0])}}
        d2 = {"logs": {"V": np.array([1.01, 10.01])}}
        # rtol=1e-7 too tight, atol=0.0 too tight
        assert compare_las_dicts(d1, d2, rtol=1e-7, atol=0.0) is False
        # rtol=1e-2 gives ~1% tolerance — 0.01/1.0=0.01 passes, 0.01/10.0=0.001 passes
        assert compare_las_dicts(d1, d2, rtol=1e-2, atol=0.0) is True
        # atol=0.02 alone passes since all absolute diffs are 0.01
        assert compare_las_dicts(d1, d2, rtol=0.0, atol=0.02) is True

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
    """F5: Tests for TypeError handler at compare.py:104-112.

    When val1 is a non-subscriptable type (e.g., int, float, str) and
    val2 is a dict, the subscript access `val1[in_key]` raises TypeError,
    which is caught at line 104 and returns False.
    """

    def test_val1_int_val2_dict(self) -> None:
        """Test TypeError when val1 is int and val2 is dict."""
        d1 = {"section": 42}
        d2 = {"section": {0: "value"}}
        assert compare_las_dicts(d1, d2) is False

    def test_val1_float_val2_dict(self) -> None:
        """Test TypeError when val1 is float and val2 is dict."""
        d1 = {"section": 3.14}
        d2 = {"section": {0: "value"}}
        assert compare_las_dicts(d1, d2) is False

    def test_val1_str_val2_dict(self) -> None:
        """Test TypeError when val1 is str and val2 is dict with non-int keys."""
        d1 = {"section": "hello"}
        d2 = {"section": {"key": "value"}}
        assert compare_las_dicts(d1, d2) is False

    def test_val1_str_val2_dict_with_int_keys(self) -> None:
        """Test TypeError when val1 is str and val2 dict has int keys.

        str supports integer subscript so val1[0] gives 'h', which falls
        through to scalar comparison with val2[0] == "hello".  Since 'h'
        != "hello", the comparison correctly returns False.

        This exercises the subscriptable-with-int-keys path WITHOUT
        triggering TypeError (str supports integer subscript).
        """
        d1 = {"section": "hello"}
        d2 = {"section": {0: "hello"}}
        assert compare_las_dicts(d1, d2) is False

        # Matching single-character str vs dict with int keys
        d1["section"] = "h"
        d2 = {"section": {0: "h"}}
        assert compare_las_dicts(d1, d2) is False

    def test_val1_str_val2_dict_multi_key(self) -> None:
        """Test TypeError when str val1 doesn't match dict val2 with multiple keys."""
        d1 = {"section": "ab"}
        d2 = {"section": {0: "a", 1: "c"}}
        assert compare_las_dicts(d1, d2) is False

    # --- F19: List with numpy arrays triggers ValueError/TypeError handler ---
    def test_numpy_array_in_list_comparison(self) -> None:
        """Test comparison when non-data_sections list contains numpy arrays.

        Exercises compare.py:129-159 — the except (ValueError, TypeError) block
        that handles numpy arrays in lists. Direct list equality comparison
        between lists containing numpy arrays raises ValueError because the
        element-wise array comparison produces an array of bools with ambiguous
        truth value. The exception handler falls through to per-element
        comparison using _compare_arrays for ndarray elements.
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
        """F-005: NaN guard catches np.float32, np.float16, np.longdouble.

        Before the fix, isinstance(x, float) missed numpy float subtypes.
        Now isinstance(x, (float, np.floating)) catches all.
        """
        d1 = {"logs": {"VAL": np.array([1.0, np.float32(np.nan), 3.0])}}
        d2 = {"logs": {"VAL": np.array([1.0, np.float32(np.nan), 3.0])}}
        assert compare_las_dicts(d1, d2) is True, "NaN np.float32 comparison must match"

    def test_nan_float16_guard(self) -> None:
        """F-005: NaN in np.float16 is handled correctly."""
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
        assert _compare_lists(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_different_arrays_in_list(self) -> None:
        """Lists containing different numpy arrays return False."""
        assert _compare_lists(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([1.0, 2.0]), np.array([3.0, 5.0])],
            "test",
            1e-7,
            0.0,
        ) is False

    def test_nested_list_of_arrays(self) -> None:
        """Nested lists containing numpy arrays match correctly."""
        # _compare_lists dispatches per-element via _compare_values
        # which handles ndarray elements via _compare_arrays
        assert _compare_lists(
            [np.array([1.0, 2.0])],
            [np.array([1.0, 2.0])],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_nested_list_of_arrays_mismatch(self) -> None:
        """Nested lists with different arrays return False."""
        assert _compare_lists(
            [np.array([1.0, 2.0])],
            [np.array([9.0, 10.0])],
            "test",
            1e-7,
            0.0,
        ) is False

    def test_list_of_arrays_within_tolerance(self) -> None:
        """List of arrays matches within floating-point tolerance."""
        assert _compare_lists(
            [np.array([1.0, 2.0])],
            [np.array([1.0, 2.0 + 1e-12])],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_list_scalar_elements(self) -> None:
        """Lists of scalars match via _compare_values fallthrough."""
        assert _compare_lists(
            ["a", "b", "c"],
            ["a", "b", "c"],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_list_scalar_elements_mismatch(self) -> None:
        """Lists of scalars with different values return False."""
        assert _compare_lists(
            ["a", "b", "c"],
            ["a", "x", "c"],
            "test",
            1e-7,
            0.0,
        ) is False

    def test_list_with_nan_values(self) -> None:
        """Lists containing NaN float values match when identical."""
        assert _compare_lists(
            [np.nan, 2.0],
            [np.nan, 2.0],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_mixed_types_in_list(self) -> None:
        """Lists with mixed scalar types match."""
        assert _compare_lists(
            [1, "hello", 3.14],
            [1, "hello", 3.14],
            "test",
            1e-7,
            0.0,
        ) is True

    def test_empty_vs_nonempty_list(self) -> None:
        """Empty list vs non-empty list returns False."""
        assert _compare_lists(
            [], [np.array([1.0])], "test", 1e-7, 0.0
        ) is False

    def test_large_list_of_arrays(self) -> None:
        """Larger lists of arrays with many elements match."""
        arrays1 = [np.array([float(i), float(i + 1)]) for i in range(10)]
        arrays2 = [np.array([float(i), float(i + 1)]) for i in range(10)]
        assert _compare_lists(arrays1, arrays2, "test", 1e-7, 0.0) is True


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
        assert _compare_lists(
            [ma_val], [42.0], "test", 1e-7, 0.0
        ) is True

    def test_0d_masked_true_vs_regular_scalar(self) -> None:
        """0-d MaskedArray with mask=True vs regular scalar returns False.

        The mask check in compare.py:148 converts a masked 0-d array to
        np.ma.masked, which causes a type mismatch and returns False.
        """
        ma_val = np.ma.array(42.0, mask=True)
        assert np.ma.is_masked(ma_val), "Precondition: mask should be True"
        assert _compare_lists(
            [ma_val], [42.0], "test", 1e-7, 0.0
        ) is False

    def test_0d_masked_vs_same_masked(self) -> None:
        """0-d MaskedArray vs same 0-d MaskedArray matches."""
        ma_val = np.ma.array(3.14)
        assert _compare_lists(
            [ma_val], [np.ma.array(3.14)], "test", 1e-7, 0.0
        ) is True

    def test_0d_masked_vs_different_masked(self) -> None:
        """0-d MaskedArray vs different value 0-d MaskedArray returns False."""
        ma_val = np.ma.array(3.14)
        assert _compare_lists(
            [ma_val], [np.ma.array(2.72)], "test", 1e-7, 0.0
        ) is False

    def test_0d_masked_true_vs_masked_true(self) -> None:
        """Two 0-d MaskedArrays both with mask=True are considered unequal.

        np.ma.masked == np.ma.masked returns np.ma.masked (not True/False),
        causing the scalar fallthrough to treat them as a mismatch.  This is
        the current behavior — the comparison layer does not treat two
        masked values as equal.
        """
        ma1 = np.ma.array(42.0, mask=True)
        ma2 = np.ma.array(99.0, mask=True)
        assert np.ma.is_masked(ma1), "Precondition: ma1 should be masked"
        assert np.ma.is_masked(ma2), "Precondition: ma2 should be masked"
        # Both become np.ma.masked → but masked == masked != True
        assert _compare_lists(
            [ma1], [ma2], "test", 1e-7, 0.0
        ) is False

    def test_0d_masked_vs_regular_scalar_mask_preserved(self) -> None:
        """0-d MaskedArray mask is preserved through nesting in lists."""
        # Two-element list with masked and non-masked values
        ma_masked = np.ma.array(42.0, mask=True)
        ma_normal = np.ma.array(3.14)
        assert _compare_lists(
            [ma_normal, ma_masked], [np.ma.array(3.14), ma_masked], "test", 1e-7, 0.0
        ) is True

    def test_0d_regular_ndarray_scalar_vs_scalar(self) -> None:
        """0-d regular ndarray vs Python scalar — .item() path still works."""
        arr = np.array(42.0)  # 0-d ndarray, NOT MaskedArray
        assert not isinstance(arr, np.ma.MaskedArray)
        assert _compare_lists(
            [arr], [42.0], "test", 1e-7, 0.0
        ) is True


class TestE08DataSectionsSymmetricGuards:
    """E-08: _compare_data_sections must handle malformed sections2 operands.

    Previously only sections1 was checked for list-ness, so sections2=None
    raised a bare TypeError on len(), a non-dict element raised a bare
    AttributeError on .keys(), and empty-container mismatches ([] == {},
    [] == (), [] == "") silently compared equal.  All cases must now
    return False (or True for genuinely equal operands) without raising.
    """

    def test_sections2_none_returns_false(self) -> None:
        """E-08: sections2=None returns False instead of raising TypeError."""
        assert compare_las_dicts(
            {"data_sections": []}, {"data_sections": None}
        ) is False

    def test_sections2_dict_vs_list_returns_false(self) -> None:
        """E-08: sections2={} vs sections1=[] returns False, not True.

        The empty-container silent FALSE-EQUAL: [] == {} previously
        returned True because both had length 0 and the loop never ran.
        """
        assert compare_las_dicts(
            {"data_sections": []}, {"data_sections": {}}
        ) is False

    def test_sections2_tuple_returns_false(self) -> None:
        """E-08: sections2=() vs sections1=[] returns False, not True."""
        assert compare_las_dicts(
            {"data_sections": []}, {"data_sections": ()}
        ) is False

    def test_sections2_string_returns_false(self) -> None:
        """E-08: sections2='' vs sections1=[] returns False, not True."""
        assert compare_las_dicts(
            {"data_sections": []}, {"data_sections": ""}
        ) is False

    def test_non_dict_element_returns_false(self) -> None:
        """E-08: non-dict element returns False instead of AttributeError."""
        assert compare_las_dicts(
            {"data_sections": [{"a": 1}]}, {"data_sections": ["notadict"]}
        ) is False

    def test_non_dict_first_element_returns_false(self) -> None:
        """E-08: non-dict element in sections1 also returns False."""
        assert compare_las_dicts(
            {"data_sections": ["notadict"]}, {"data_sections": [{"a": 1}]}
        ) is False

    def test_matching_sections_still_equal(self) -> None:
        """E-08: well-formed matching data_sections still compare equal."""
        d1 = {"data_sections": [{"DEPT": np.array([1.0, 2.0])}]}
        d2 = {"data_sections": [{"DEPT": np.array([1.0, 2.0])}]}
        assert compare_las_dicts(d1, d2) is True
