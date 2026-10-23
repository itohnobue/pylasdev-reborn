"""Tests for _section_transition.py — section transition lifecycle handler.

Covers:
- _process_consecutive_data (A→A swap state transitions)
- MAX_TOTAL_ELEMENTS cumulative guard in parser path
- verify_user_generated_files_reasonably_short
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pylasdev import read_las_file, read_las_file_as_object, write_las_file
from pylasdev.exceptions import LASParseError
from pylasdev.parser import LASParser


class TestProcessConsecutiveData:
    """Tests for _process_consecutive_data (I2F-17).

    The A→A consecutive data section swap path saves/restores curve
    indices and swaps the data section type to correctly process the
    previous section's data before proceeding to the new section.
    """

    def test_consecutive_a_to_a_stores_both_sections(self) -> None:
        """Two consecutive ~A sections produce two data sections in the LASFile.

        Also asserts the per-section section_type (LOG_DATA) survives the
        A→A swap (F-254 — folded from the former
        test_consecutive_a_to_a_section_type_preserved).
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A Section1
100.0
~A Section2
200.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[1].data["DEPT"][0] == 200.0
        assert las.data_sections[0].section_type == "LOG_DATA"
        assert las.data_sections[1].section_type == "LOG_DATA"

    def test_consecutive_a_to_a_with_different_data_counts(self) -> None:
        """A→A swap preserves per-section data integrity.

        Two consecutive data sections with different numbers of data
        rows.  The A→A swap must correctly save/restore curve indices
        so each section's data is processed independently with the
        correct row count.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 GR.API       : GAMMA RAY  {F}
~A Section1
100.0,50.0
200.0,60.0
~A Section2
300.0,70.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2
        sec1 = las.data_sections[0]
        sec2 = las.data_sections[1]
        # Section1: 2 rows, 2 curves (DEPT, GR)
        assert len(sec1.data["DEPT"]) == 2
        assert sec1.data["DEPT"][0] == 100.0
        assert sec1.data["DEPT"][1] == 200.0
        assert sec1.data["GR"][0] == 50.0
        assert sec1.data["GR"][1] == 60.0
        # Section2: 1 row, 2 curves (DEPT, GR)
        assert len(sec2.data["DEPT"]) == 1
        assert sec2.data["DEPT"][0] == 300.0
        assert sec2.data["GR"][0] == 70.0

    def test_consecutive_a_to_a_with_three_sections(self) -> None:
        """Three consecutive ~A sections produce three data sections."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A Section1
100.0
~A Section2
200.0
~A Section3
300.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 3
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[1].data["DEPT"][0] == 200.0
        assert las.data_sections[2].data["DEPT"][0] == 300.0


class TestMAXTOTAL_ELEMENTS:
    """Tests for cumulative MAX_TOTAL_ELEMENTS guard in parser path (I2F-18).

    The cumulative cross-section allocation guard at
    _las30_data.py:468-480 checks that the total elements across ALL
    data sections does not exceed MAX_TOTAL_ELEMENTS.  Each individual
    section may be well under the per-section limit, but the cumulative
    total can still exceed the hard bound.
    """

    def test_cumulative_max_total_elements_across_sections(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cumulative guard fires when multiple sections exceed total limit.

        Two sections, each with 1 curve x 3 lines = 3 elements.
        MAX_TOTAL_ELEMENTS=5: section1 (3) + section2 (3) = 6 > 5.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 5)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A Section1
100.0
200.0
300.0
~A Section2 | DEPT,GR
400.0
500.0
600.0
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Cumulative cross-section allocation"):
            parser.parse(content)

    def test_cumulative_max_total_elements_below_limit_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cumulative guard does NOT fire when total is at or below limit.

        Two sections: 3 + 1 = 4 elements.  MAX_TOTAL_ELEMENTS=5 passes.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 5)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A Section1
100.0
200.0
300.0
~A Section2
400.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2
        assert len(las.data_sections[0].data["DEPT"]) == 3
        assert len(las.data_sections[1].data["DEPT"]) == 1

    def test_cumulative_guard_not_triggered_by_single_section(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single section under per-section limit passes cumulative guard.

        MAX_TOTAL_ELEMENTS=100: 1 curve x 10 lines = 10, well under limit.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 100)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A Section1
"""
        for i in range(10):
            content += f"{100.0 + i}\n"
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        assert len(las.data_sections[0].data["DEPT"]) == 10

    def test_cumulative_guard_uses_live_state_after_replay(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies F-56/F-57 fix: cumulative counter after deferred replay.

        Data sections before ~V are deferred and replayed.  The
        cumulative counter must correctly include the replayed
        sections' elements, not just the post-~V sections.

        Pre-fix: only 1 section counted (3 elements), guard never fires.
        Post-fix: 2 sections counted (6 elements > 5), guard fires.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 5)
        # Data section before ~V — will be deferred and replayed.
        # Data section after ~V — processed normally.
        # Total: 3 + 3 = 6 elements > 5 → guard MUST fire.
        content = """~A PreVersion
100.0
200.0
300.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A PostVersion
400.0
500.0
600.0
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Cumulative cross-section allocation"):
            parser.parse(content)


class TestF185IndexedParameterSection:
    """F-185 / M6 regression pin: the ``[n]`` index suffix on a typed
    parameter section must not defeat the ``_PARAMETER`` dispatch.

    ``_section_transition.enter_new_section`` (``_section_transition.py:299-315``)
    tested the RAW section word, so a spec-legal ``~Log_Parameter[2]`` failed
    ``endswith("_PARAMETER")``; with an annotation the raw word was then
    replaced by the annotation text, ``_parse_parameter_entry`` had no
    ``_PARAMETER`` left to strip, ``section_type`` became ``None`` and the
    writer re-emitted ``~PARAMETER INFORMATION``.  The same bytes with ``~V``
    last parsed correctly, so the model was version-order dependent
    (ADV-2 matrix: V-first ``[None]`` / V-late ``['LOG']``).

    Pre-fix this test fails on the ``v-first`` row (``section_type`` None and
    a ``~PARAMETER INFORMATION`` header); post-fix both orders agree on
    ``'LOG'`` and the writer re-emits ``~LOG_Parameter``.
    """

    _HEAD = (
        "~VERSION INFORMATION\n"
        " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
        " WRAP.   NO  : ONE LINE PER DEPTH STEP\n"
        " DLM .   COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS\n"
        "~WELL INFORMATION\n"
        " STRT .M   100.0  : First Index Value\n"
        " STOP .M   102.0  : Last Index Value\n"
        " STEP .M   1.0  : STEP\n"
        " NULL .    -999.25  : NULL VALUE\n"
    )
    _PARAM_ANNOTATED = (
        "~Log_Parameter[2]  Parameter Information\n BHT.DEGC   35.5 : Bottom Hole Temperature\n"
    )
    _PARAM_BARE = "~Log_Parameter[2]\n BHT.DEGC   35.5 : Bottom Hole Temperature\n"
    _CURVES = "~CURVE INFORMATION\n DEPT .M  : DEPTH  {F}\n GR   .API  : GAMMA RAY  {F}\n"
    _DATA = "~Log_Data | CURVE\n100.0,10.0\n101.0,20.0\n102.0,30.0\n"

    @pytest.mark.parametrize("annotated", [True, False], ids=["annotated", "bare"])
    @pytest.mark.parametrize("order", ["v-first", "v-late"])
    def test_indexed_parameter_section_types_as_log(
        self, annotated: bool, order: str, tmp_path: Path
    ) -> None:
        """Both ~V orders must produce section_type 'LOG' and a ~LOG_Parameter header."""
        param_block = self._PARAM_ANNOTATED if annotated else self._PARAM_BARE
        if order == "v-first":
            content = self._HEAD + param_block + self._CURVES + self._DATA
        else:
            content = param_block + self._HEAD + self._CURVES + self._DATA
        src = tmp_path / f"param_{order}_{'ann' if annotated else 'bare'}.las"
        src.write_text(content, encoding="ascii")

        las = read_las_file_as_object(src)
        param_section_types = [entry.section_type for entry in las.parameters]
        assert param_section_types == ["LOG"], (
            f"{order}/{annotated}: indexed ~Log_Parameter must type as LOG, got "
            f"{param_section_types}"
        )

        details = read_las_file(src).get("parameter_details", [])
        assert [(d.get("mnemonic"), d.get("section_type")) for d in details] == [("BHT", "LOG")]

        out = tmp_path / f"param_{order}_{'ann' if annotated else 'bare'}_out.las"
        write_las_file(out, las)
        headers = [
            line
            for line in out.read_text().splitlines()
            if line.lower().startswith("~log_parameter")
        ]
        assert headers, "the writer must re-emit ~LOG_Parameter for a LOG-typed parameter section"
        assert "~PARAMETER INFORMATION" not in out.read_text(), (
            "the annotation text must not be re-emitted as a section header"
        )
