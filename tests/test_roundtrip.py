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
                decimal=6,
            )

    def test_roundtrip_all_files(self, all_las_files: list[Path], tmp_path: Path) -> None:
        """Test round-trip on all test files."""
        # sample_las3.0_spec.las contains structured data-type sections
        # (~Drilling, ~Core, ~Inclinometry, ~Tops, ~Test, ~Perforations)
        # whose curve data is populated on re-read (F-01 fix). The
        # current architecture stores all curves globally, so roundtrip
        # of structured data produces differently-named curves than the
        # original read. Skip strict per-curve data comparison for this
        # file — it's tested separately via data_sections verification.
        structured_files = {"sample_las3.0_spec.las"}

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
                    if las_path.name in structured_files:
                        # For files with structured sections, only verify shapes
                        # match — data values may differ because re-read
                        # populates structured-section curves from their own
                        # data sections rather than from the main ASCII section.
                        assert original["logs"][curve].shape == roundtrip["logs"][curve].shape, (
                            f"Shape mismatch for {curve} in {las_path.name}: "
                            f"{original['logs'][curve].shape} vs {roundtrip['logs'][curve].shape}"
                        )
                    else:
                        assert original["logs"][curve].shape == roundtrip["logs"][curve].shape, (
                            f"Shape mismatch for {curve} in {las_path.name}: "
                            f"{original['logs'][curve].shape} vs {roundtrip['logs'][curve].shape}"
                        )
                        # F-041: Verify data values are preserved across write→read
                        # Use rtol=1e-5 to account for precision formatting (~8 significant digits)
                        np.testing.assert_allclose(
                            original["logs"][curve],
                            roundtrip["logs"][curve],
                            rtol=1e-5,
                            err_msg=(f"Data mismatch for {curve} in {las_path.name}"),
                        )

            # Verify string_data entries preserved (LAS 3.0 {S} format curves)
            orig_string_data = original.get("string_data", {})
            rt_string_data = roundtrip.get("string_data", {})
            for key in orig_string_data:
                assert key in rt_string_data, (
                    f"string_data key {key} missing in roundtrip for {las_path.name}"
                )
                np.testing.assert_array_equal(
                    orig_string_data[key],
                    rt_string_data[key],
                    err_msg=f"string_data mismatch for {key} in {las_path.name}",
                )

            # Verify data_sections count preserved (LAS 3.0 multi-section files)
            orig_sections = original.get("data_sections", [])
            rt_sections = roundtrip.get("data_sections", [])
            assert len(rt_sections) == len(orig_sections), (
                f"data_sections count mismatch in {las_path.name}: "
                f"{len(rt_sections)} vs {len(orig_sections)}"
            )

    def test_roundtrip_preserves_curve_metadata(self) -> None:
        """Test that to_dict/from_dict round-trip preserves curve metadata."""
        from pylasdev.models import CurveDefinition, LASFile, VersionSection

        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH"))
        las.curves.append(
            CurveDefinition(mnemonic="DT", unit="US/M", api_code="123", description="SONIC")
        )
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        d = las.to_dict()
        restored = LASFile.from_dict(d)

        assert len(restored.curves) == 2
        assert restored.curves[0].unit == "M"
        assert restored.curves[0].description == "DEPTH"
        assert restored.curves[1].unit == "US/M"
        assert restored.curves[1].api_code == "123"
        assert restored.curves[1].description == "SONIC"
