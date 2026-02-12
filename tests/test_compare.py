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
