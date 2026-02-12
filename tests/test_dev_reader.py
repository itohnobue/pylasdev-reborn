"""Tests for DEV file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_dev_file
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
        with pytest.raises(DEVReadError):
            read_dev_file(tmp_path / "nonexistent.dev")

    def test_sample_dev_columns(self, test_data_dir: Path) -> None:
        """Test that sample.dev has expected columns."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip("sample.dev not found")
        data = read_dev_file(sample_dev)
        assert "MD" in data
        assert "TVD" in data
        assert "X" in data
        assert "Y" in data

    def test_sample_dev_data_shape(self, test_data_dir: Path) -> None:
        """Test that all columns have the same length."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip("sample.dev not found")
        data = read_dev_file(sample_dev)
        sizes = [len(arr) for arr in data.values()]
        assert len(set(sizes)) == 1, f"Column sizes differ: {sizes}"

    def test_sample_dev_md_starts_at_zero(self, test_data_dir: Path) -> None:
        """Test that MD column starts at 0."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip("sample.dev not found")
        data = read_dev_file(sample_dev)
        assert data["MD"][0] == 0.0

    def test_sample_dev_has_multiple_rows(self, test_data_dir: Path) -> None:
        """Test that sample.dev has multiple data rows."""
        sample_dev = test_data_dir / "sample.dev"
        if not sample_dev.exists():
            pytest.skip("sample.dev not found")
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
            pytest.skip("sample.dev not found")
        data = read_dev_file(sample_dev, encoding="utf-8")
        assert len(data) > 0

    def test_dev_file_model(self) -> None:
        """Test DevFile model to_dict."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0, 200.0])
        dev.columns["TVD"] = np.array([0.0, 99.0, 198.0])
        d = dev.to_dict()
        assert "MD" in d
        assert "TVD" in d
        np.testing.assert_array_equal(d["MD"], np.array([0.0, 100.0, 200.0]))
