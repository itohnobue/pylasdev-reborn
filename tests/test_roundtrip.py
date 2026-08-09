"""Tests for read/write round-trip consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev import (
    CurveDefinition,
    LASFile,
    ParameterEntry,
    ParameterZone,
    VersionSection,
    read_las_file,
    read_las_file_as_object,
    write_las_file,
)


class TestRoundTrip:
    """Tests for read-write-read consistency."""

    def test_roundtrip_from_dict(self, sample_las_data: dict, tmp_path: Path) -> None:
        """Test that writing from dict and reading back preserves data."""
        temp_file = tmp_path / "roundtrip.las"
        write_las_file(temp_file, sample_las_data)
        roundtrip = read_las_file(temp_file)

        # Check structure — strict list equality: DEPT/DT position swap must fail
        assert roundtrip["curves_order"] == sample_las_data["curves_order"]

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

            # F-112 fold (from test_writer.py test_write_real_files_roundtrip):
            # VERS must roundtrip exactly for every real file.
            assert roundtrip["version"]["VERS"] == original["version"]["VERS"], (
                f"VERS mismatch in {las_path.name}: "
                f"{roundtrip['version']['VERS']} vs {original['version']['VERS']}"
            )

            # F-112 fold: STRICT curves_order LIST equality for
            # non-structured files — len() alone cannot detect a mnemonic
            # rename that preserves count.
            if las_path.name not in structured_files:
                assert roundtrip["curves_order"] == original["curves_order"], (
                    f"curves_order mismatch in {las_path.name}: "
                    f"{roundtrip['curves_order']} vs {original['curves_order']}"
                )

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

    # --- F-053/IF-011: Per-section parameter roundtrip ---
    def test_per_section_parameter_roundtrip(self, tmp_path: Path) -> None:
        """F-053/IF-011 (R-003): Per-section parameters survive write/read roundtrip.

        Construct a LAS 3.0 LASFile with parameters having
        section_type='CORE' and section_type=None.  Write to file,
        re-read, and verify that section_type is preserved on each
        ParameterEntry.  Also verify the writer emits a separate
        ~Core_Parameter section for section-typed parameters.
        """
        from pylasdev.models import (
            CurveDefinition,
            DataSection,
            LASFile,
            ParameterEntry,
            VersionSection,
        )

        las = LASFile()
        las.version = VersionSection(vers="3.0", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        # Global parameter (section_type=None → standard ~P section)
        las.parameters.append(
            ParameterEntry(
                mnemonic="BHT",
                unit="DEGC",
                value="35.5",
                description="Bottom Hole Temperature",
            )
        )
        # Per-section parameter (section_type="CORE")
        las.parameters.append(
            ParameterEntry(
                mnemonic="MATR",
                value="SAND",
                description="Neutron Matrix",
                section_type="CORE",
            )
        )

        # Add a CORE_DATA section so the writer has something to pair
        section = DataSection(
            name="Core[1]",
            section_type="CORE_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([550.0, 551.0])},
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "per_section_param.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()
        # Write-time verification: ~CORE_Parameter section
        # should be present for the CORE-typed parameter
        assert "~CORE_Parameter" in content, (
            f"Expected per-section parameter block for CORE in output:\n{content[:2000]}"
        )
        # Standard ~P section should also be present for global parameter
        assert "~PARAMETER" in content or "~Parameter" in content, (
            "Expected standard ~P section for global parameters"
        )

        # Roundtrip: re-read and verify section_type preserved
        data = read_las_file(temp_file)
        param_details = data.get("parameter_details", [])
        assert len(param_details) >= 2, f"Expected at least 2 parameters, got {len(param_details)}"

        # Find BHT (section_type should be None → not in output or default)
        bht_params = [p for p in param_details if p.get("mnemonic") == "BHT"]
        assert len(bht_params) == 1
        # BHT should have no section_type (global parameter)
        assert bht_params[0].get("section_type") in (None, ""), (
            f"BHT should not have section_type, got: {bht_params[0].get('section_type')!r}"
        )

        # Find MATR (section_type should be 'CORE')
        matr_params = [p for p in param_details if p.get("mnemonic") == "MATR"]
        assert len(matr_params) == 1, "MATR parameter missing from roundtrip"
        assert matr_params[0].get("section_type") == "CORE", (
            f"MATR section_type should be 'CORE', got: {matr_params[0].get('section_type')!r}"
        )

    def test_per_section_parameter_multiple_section_types(self, tmp_path: Path) -> None:
        """F-053/IF-011: Multiple section types each get their own
        per-section parameter block."""
        from pylasdev.models import (
            CurveDefinition,
            DataSection,
            LASFile,
            ParameterEntry,
            VersionSection,
        )

        las = LASFile()
        las.version = VersionSection(vers="3.0", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", data_format="F"))
        las.logs["DEPT"] = np.array([100.0])

        # Parameters with different section types
        las.parameters.append(
            ParameterEntry(
                mnemonic="CORE_PARAM",
                value="core_val",
                section_type="CORE",
            )
        )
        las.parameters.append(
            ParameterEntry(
                mnemonic="DRILL_PARAM",
                value="drill_val",
                section_type="DRILLING",
            )
        )

        # Add corresponding data sections
        las.data_sections.append(
            DataSection(
                name="Core[1]",
                section_type="CORE_DATA",
                curves_order=["DEPT"],
                data={"DEPT": np.array([550.0])},
            )
        )
        las.data_sections.append(
            DataSection(
                name="Drill[1]",
                section_type="DRILLING_DATA",
                curves_order=["DEPT"],
                data={"DEPT": np.array([100.0])},
            )
        )

        temp_file = tmp_path / "multi_section_param.las"
        write_las_file(temp_file, las)

        content = temp_file.read_text()

        # Both per-section parameter blocks should be present
        assert "~CORE_Parameter" in content, f"Missing ~CORE_Parameter section:\n{content[:2000]}"
        assert "~DRILLING_Parameter" in content, (
            f"Missing ~DRILLING_Parameter section:\n{content[:2000]}"
        )
        assert "core_val" in content
        assert "drill_val" in content

        # Roundtrip: re-read and verify section_type preserved
        data = read_las_file(temp_file)
        param_details = data.get("parameter_details", [])

        core_params = [p for p in param_details if p.get("mnemonic") == "CORE_PARAM"]
        assert len(core_params) == 1
        assert core_params[0].get("section_type") == "CORE"

        drill_params = [p for p in param_details if p.get("mnemonic") == "DRILL_PARAM"]
        assert len(drill_params) == 1
        assert drill_params[0].get("section_type") == "DRILLING"

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


# ──────────────────────────────────────────────────────────────
# G2 (N-I-22 / N-I-02): parser/writer iter-2 new findings
# ──────────────────────────────────────────────────────────────


class TestNI22UnitRoundtrip:
    """N-I-22 (HIGH): units containing ``%`` / ``°C`` / ``ohm.m`` must
    survive the write→read roundtrip.  Previously the parser's unit
    character class rejected them and the whole curve + data column were
    silently dropped."""

    _LAS20 = (
        "~VERSION INFORMATION\n"
        " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
        "~WELL INFORMATION\n"
        " STRT.M   100.0 : \n"
        " STOP.M   200.0 : \n"
        " STEP.M   1.0 : \n"
        " NULL.    -999.25 : \n"
    )

    def _write_read(self, content: str, tmp_path: Path) -> object:
        import warnings

        src = tmp_path / "unit_src.las"
        src.write_text(content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(src)
            out = tmp_path / "unit_out.las"
            write_las_file(out, las)
            return read_las_file_as_object(out)

    def test_roundtrip_percent_and_ohm_m_units(self, tmp_path: Path) -> None:
        content = (
            self._LAS20
            + "~CURVE INFORMATION\n"
            + " DEPT.M      1000.0 : DEPTH\n"
            + " PHIT.%      25.5 : POROSITY\n"
            + " RT.ohm.m    15.5 : RESISTIVITY\n"
            + "~A DEPTH PHIT RT\n"
            + "1000.0 25.5 15.5\n"
            + "1001.0 26.5 16.5\n"
        )
        rt = self._write_read(content, tmp_path)
        units = {c.mnemonic: c.unit for c in rt.curves}
        assert units.get("PHIT") == "%", units
        assert units.get("RT") == "ohm.m", units
        # Data columns preserved (previously the curve was dropped entirely).
        assert "PHIT" in rt.logs and "RT" in rt.logs
        assert list(rt.logs["RT"]) == [15.5, 16.5]

    def test_roundtrip_degree_celsius_unit(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : LAS 3.0\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n"
            " STOP.M   200.0 : \n"
            " STEP.M   1.0 : \n"
            " NULL.    -999.25 : \n"
            "~CURVE INFORMATION\n"
            " DEPT.M      1000.0 : DEPTH\n"
            " TEMP.°C     23.5 : TEMPERATURE\n"
            "~A DEPTH TEMP\n"
            "1000.0 23.5\n"
            "1001.0 24.5\n"
        )
        rt = self._write_read(content, tmp_path)
        units = {c.mnemonic: c.unit for c in rt.curves}
        assert units.get("TEMP") == "°C", units
        assert list(rt.logs["TEMP"]) == [23.5, 24.5]


class TestNI02ParameterPipeRoundtrip:
    """N-I-02 (MEDIUM): ZONE_ASSOC_PATTERN ran unconditionally, so a
    LAS 1.2/2.0 description ending in a pipe (``Run number | Main Zone``)
    was truncated, a bogus ParameterZone attached, and the pipe text
    permanently lost on roundtrip (the writer never re-emits zones for
    non-3.0).  Zone extraction is now LAS 3.0-only, and the writer escapes
    literal pipes (``|`` → ``\\|``) which the parser unescapes."""

    _LAS20 = (
        "~VERSION INFORMATION\n"
        " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
        "~WELL INFORMATION\n"
        " STRT.M   100.0 : \n"
        " STOP.M   200.0 : \n"
        " STEP.M   1.0 : \n"
        " NULL.    -999.25 : \n"
    )

    def test_las20_pipe_description_roundtrips(self, tmp_path: Path) -> None:
        import warnings

        content = (
            self._LAS20
            + "~PARAMETER INFORMATION\n"
            + " RUN. 5 : Run number | Main Zone\n"
            + "~CURVE INFORMATION\n"
            + " DEPT.M 100.0 : DEPTH\n"
            + "~A DEPTH\n"
            + "100.0\n"
            + "200.0\n"
        )
        src = tmp_path / "pipe_src.las"
        src.write_text(content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(src)
            out = tmp_path / "pipe_out.las"
            write_las_file(out, las)
            rt = read_las_file_as_object(out)
        # Description preserved (pipe text NOT truncated), no bogus zone.
        assert rt.parameters[0].description == "Run number | Main Zone"
        assert rt.parameters[0].zone is None

    def test_las30_pipe_description_not_misparsed_as_zone(self, tmp_path: Path) -> None:
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        # Genuine description text with a pipe — NOT a zone association.
        las.parameters.append(
            ParameterEntry(mnemonic="NOTE", value="x", description="Note | Extra text")
        )
        out = tmp_path / "pipe_las30.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
            rt = read_las_file_as_object(out)
        assert rt.parameters[0].description == "Note | Extra text", rt.parameters[0].description
        assert rt.parameters[0].zone is None

    def test_las30_real_zone_still_roundtrips(self, tmp_path: Path) -> None:
        import warnings

        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        las.parameters.append(
            ParameterEntry(
                mnemonic="MATR",
                unit="",
                value="SAND",
                description="Neutron Matrix",
                zone=ParameterZone(zone_name="RUN", zone_index=1),
            )
        )
        out = tmp_path / "zone_las30.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
            rt = read_las_file_as_object(out)
        assert rt.parameters[0].zone is not None
        assert rt.parameters[0].zone.zone_name == "RUN"
        assert rt.parameters[0].zone.zone_index == 1
        assert rt.parameters[0].description == "Neutron Matrix"
