"""Tests for read/write round-trip consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
        # whose per-section curve data is populated on re-read. The
        # roundtrip fix (s7-fix-roundtrip) now preserves per-section
        # curve names — these are verified below via data_sections
        # curves_order comparison. However, the global curves_order list
        # and per-curve data values may differ on roundtrip because
        # re-read populates structured-section data from their own
        # sections rather than from the main ASCII section. Skip strict
        # per-curve data value comparison for this file — shapes only.
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

            # Verify per-section curve name preservation (MEDIUM-2)
            for i, (orig_sec, rt_sec) in enumerate(zip(orig_sections, rt_sections, strict=True)):
                assert orig_sec["section_type"] == rt_sec["section_type"], (
                    f"section_type mismatch for section {i} in {las_path.name}: "
                    f"{orig_sec['section_type']} vs {rt_sec['section_type']}"
                )
                assert orig_sec["curves_order"] == rt_sec["curves_order"], (
                    f"curves_order mismatch for section {i} "
                    f"({orig_sec['section_type']}) in {las_path.name}: "
                    f"{orig_sec['curves_order']} vs {rt_sec['curves_order']}"
                )

    # --- T9/G-12: LAS 3.0 structured sections roundtrip value verification ---
    def test_roundtrip_structured_sections_values(
        self, test_data_dir: Path, tmp_path: Path
    ) -> None:
        """Test that LAS 3.0 structured data sections roundtrip preserves
        data VALUES, not just shapes.

        Reads sample_las3.0_spec.las, writes, re-reads, and verifies that
        per-section data arrays and string_data arrays match in shape AND
        in actual values (within numeric tolerance).
        """
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"

        original = read_las_file(spec_file)
        temp_file = tmp_path / "roundtrip_spec.las"
        write_las_file(temp_file, original)
        roundtrip = read_las_file(temp_file)

        # Verify data_sections count matches
        orig_sections = original.get("data_sections", [])
        rt_sections = roundtrip.get("data_sections", [])
        assert len(rt_sections) == len(orig_sections)

        # Per-section value verification
        for i, (orig_sec, rt_sec) in enumerate(zip(orig_sections, rt_sections, strict=True)):
            section_type = orig_sec["section_type"]
            # Verify curves_order preserved
            assert orig_sec["curves_order"] == rt_sec["curves_order"], (
                f"curves_order mismatch for section {i} ({section_type}): "
                f"{orig_sec['curves_order']} vs {rt_sec['curves_order']}"
            )
            # Verify data arrays: shapes and values
            orig_data = orig_sec.get("data", {})
            rt_data = rt_sec.get("data", {})
            for curve in orig_sec["curves_order"]:
                if curve not in orig_data:
                    continue
                assert curve in rt_data, f"curve {curve} missing in roundtrip data for section {i}"
                assert orig_data[curve].shape == rt_data[curve].shape, (
                    f"Shape mismatch for {curve} in section {i}: "
                    f"{orig_data[curve].shape} vs {rt_data[curve].shape}"
                )
                np.testing.assert_allclose(
                    orig_data[curve],
                    rt_data[curve],
                    rtol=1e-5,
                    err_msg=f"Data mismatch for {curve} in section {i} ({section_type})",
                )
            # Verify string_data arrays if present
            orig_str = orig_sec.get("string_data", {})
            rt_str = rt_sec.get("string_data", {})
            for key in orig_str:
                assert key in rt_str, f"string_data key {key} missing in section {i}"
                np.testing.assert_array_equal(
                    orig_str[key],
                    rt_str[key],
                    err_msg=f"string_data mismatch for {key} in section {i}",
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


class TestDEVRoundtripSkipped:
    """DEV roundtrip tests — skipped until DEV writer is implemented.

    F-T2-M04: No DEV writer exists.  ``write_dev_file`` is needed before
    DEV roundtrip tests can be meaningful.
    F-T3-M02: DEV roundtrip is untested as a result.
    """

    @pytest.mark.skip(reason="F-T2-M04: DEV writer not implemented")
    def test_dev_roundtrip_skipped(self) -> None:
        """DEV read → write → read roundtrip — not yet testable."""
        pass
