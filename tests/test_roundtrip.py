"""Tests for read/write round-trip consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pylasdev import read_las_file, write_las_file


class TestRoundTrip:
    """Tests for read-write-read consistency."""

    def test_roundtrip_from_dict(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that writing from dict and reading back preserves data."""
        temp_file = tmp_path / "roundtrip.las"
        write_las_file(temp_file, sample_las_data)
        roundtrip = read_las_file(temp_file)

        # Check structure
        assert set(roundtrip["curves_order"]) == set(sample_las_data["curves_order"])

        # Check data values
        for curve in sample_las_data["curves_order"]:
            np.testing.assert_array_almost_equal(
                sample_las_data["logs"][curve],
                roundtrip["logs"][curve],
                decimal=4,
            )

    def test_roundtrip_all_files(self, all_las_files: list[Path], tmp_path: Path) -> None:
        """Test round-trip on all test files."""
        for las_path in all_las_files:
            original = read_las_file(las_path)

            temp_file = tmp_path / las_path.name
            write_las_file(temp_file, original)
            roundtrip = read_las_file(temp_file)

            # Verify curve count preserved
            assert len(roundtrip["curves_order"]) == len(original["curves_order"])

            # Verify data shapes match (skip curves not in both logs, e.g. LAS 3.0 string curves)
            for curve in original["curves_order"]:
                if curve in original["logs"] and curve in roundtrip["logs"]:
                    assert original["logs"][curve].shape == roundtrip["logs"][curve].shape, (
                        f"Shape mismatch for {curve} in {las_path.name}: "
                        f"{original['logs'][curve].shape} vs {roundtrip['logs'][curve].shape}"
                    )
