"""Tests for _section_transition.py — section transition lifecycle handler.

Covers:
- _process_consecutive_data (A→A swap state transitions)
- MAX_TOTAL_ELEMENTS cumulative guard in parser path
- verify_user_generated_files_reasonably_short
"""

from __future__ import annotations

import pytest

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
