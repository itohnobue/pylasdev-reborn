"""Tests for LAS data comparison utilities."""

from __future__ import annotations

import numpy as np

from pylasdev.compare import compare_las_dicts


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
        assert compare_las_dicts(d1, d2) is True

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
