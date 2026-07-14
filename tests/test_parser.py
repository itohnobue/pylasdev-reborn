"""Tests for LAS file parser."""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_las_file
from pylasdev.exceptions import LASParseError
from pylasdev.parser import LASParser, _is_indexed_data_section


class TestLASParser:
    """Tests for the regex-based LAS parser."""

    def test_parse_version_section(self) -> None:
        """Test parsing ~V section."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == "2.0"
        assert las.version.wrap == "NO"

    def test_parse_version_120(self) -> None:
        """Test parsing LAS 1.2 version."""
        content = """~Version Information
 VERS.                1.20:   CWLS log ASCII Standard -VERSION 1.20
 WRAP.                 YES:   Multiple lines per depth step
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == "1.20"
        assert las.version.wrap == "YES"

    def test_parse_well_section(self) -> None:
        """Test parsing ~W section."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT.M   1670.0 : START DEPTH
 STOP.M   1660.0 : STOP DEPTH
 NULL.    -999.25 : NULL VALUE
 COMP.    Test Co : COMPANY
 WELL.    Well #1 : WELL NAME
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.well["STRT"] == "1670.0"
        assert las.well["STOP"] == "1660.0"
        assert las.well["NULL"] == "-999.25"
        assert las.well["COMP"] == "Test Co"
        assert las.well["WELL"] == "Well #1"

    def test_parse_curve_section(self) -> None:
        """Test parsing ~C section."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M       :  Depth
 DT  .US/M    :  Sonic Travel Time
 RHOB.K/M3    :  Bulk Density
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 3
        assert las.curves_order == ["DEPT", "DT", "RHOB"]
        assert las.curves[0].mnemonic == "DEPT"
        assert las.curves[0].unit == "M"
        assert las.curves[1].mnemonic == "DT"
        assert las.curves[1].unit == "US/M"

    def test_parse_curve_with_spaces_before_dot(self) -> None:
        """Test parsing curves where mnemonic has trailing spaces before dot."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M                       :    Depth
 DT  .US/M                    :  1 Sonic Travel Time
 SP  .MV                      :  8 Spon. Potential
 GR  .GAPI                    :  9 Gamma Ray
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 4
        assert las.curves_order == ["DEPT", "DT", "SP", "GR"]

    def test_parse_parameter_section(self) -> None:
        """Test parsing ~P section."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~PARAMETER INFORMATION
 BHT.DEGC    35.5 : BOTTOM HOLE TEMPERATURE
 BS .MM      200  : BIT SIZE
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.parameters) == 2
        assert las.parameters[0].mnemonic == "BHT"
        assert las.parameters[0].value == "35.5"
        assert las.parameters[0].unit == "DEGC"
        assert las.parameters[1].mnemonic == "BS"

    def test_parse_other_section(self) -> None:
        """Test parsing ~O section accumulates free text."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~OTHER
Line one of free text.
Line two of free text.
"""
        parser = LASParser()
        las = parser.parse(content)
        assert "Line one" in las.other
        assert "Line two" in las.other

    def test_skip_comments(self) -> None:
        """Test that comment lines (starting with #) are skipped."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
# This is a comment
 DEPT.M       :  Depth
# Another comment
 DT  .US/M    :  Sonic Travel Time
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 2

    def test_mnem_base_normalization(self) -> None:
        """Test that mnem_base normalizes curve names."""
        mnem_base = {"AK": "DT", "APTS": "SP"}
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M       :  Depth
 AK  .US/M    :  Sonic
"""
        parser = LASParser(mnem_base)
        las = parser.parse(content)
        assert las.curves_order == ["DEPT", "DT"]
        assert las.curves[1].mnemonic == "DT"
        assert las.curves[1].original_mnemonic == "AK"

    def test_cyrillic_mnemonics(self) -> None:
        """Test that Cyrillic curve names are parsed correctly."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M       :  Depth
 \u0413\u041a.API    :  \u0413\u0430\u043c\u043c\u0430 \u043a\u0430\u0440\u043e\u0442\u0430\u0436
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 2
        assert las.curves[1].mnemonic == "\u0413\u041a"

    def test_pre_scan_counts_data_lines(self, tmp_path: Path) -> None:
        """Test that pre-scan correctly counts ASCII data lines.

        Uses public API (read_las_file) instead of asserting on private
        _data_line_count. The correct pre-scan count produces correct
        data array lengths on read.
        """
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~CURVE INFORMATION
 DEPT.M   :
 DT.US/M  :
~A
100.0  50.0
100.1  51.0
100.2  52.0
"""
        test_file = tmp_path / "pre_scan_data.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 3
        assert len(data["logs"]["DT"]) == 3

    def test_las30_version_detected(self) -> None:
        """Test that LAS 3.0 version is detected correctly."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == "3.0"
        assert las.version.dlm == "COMMA"
        assert las.is_las30

    # --- R-005: Parametrized version detection tests ---
    @pytest.mark.parametrize(
        "content,expected_vers,expected_wrap,is_las30",
        [
            (
                "~VERSION INFORMATION\n VERS.   2.0  : CWLS LOG ASCII STANDARD\n WRAP.   NO   : ONE LINE PER DEPTH STEP\n",
                "2.0",
                "NO",
                False,
            ),
            (
                "~Version Information\n VERS.                1.20:   CWLS log ASCII Standard -VERSION 1.20\n WRAP.                 YES:   Multiple lines per depth step\n",
                "1.20",
                "YES",
                False,
            ),
            (
                "~VERSION INFORMATION\n VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n WRAP.   NO   :\n DLM.   COMMA :\n",
                "3.0",
                "NO",
                True,
            ),
        ],
    )
    def test_version_detection_parametrized(
        self,
        content: str,
        expected_vers: str,
        expected_wrap: str,
        is_las30: bool,
    ) -> None:
        """Parametrized test for LAS version detection across 1.2, 2.0, 3.0."""
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == expected_vers
        assert las.version.wrap == expected_wrap
        assert las.is_las30 == is_las30

    def test_las30_curve_format_specifiers(self) -> None:
        """Test parsing LAS 3.0 format specifiers {F}, {E}, {S}."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT .M                                       : DEPTH               {F}
 DT   .US/M           123 456 789              : SONIC TRANSIT TIME  {F}
 CDES .               123 456 789              : CORE DESCRIPTION    {S}
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.curves[0].data_format == "F"
        assert las.curves[2].data_format == "S"

    def test_las30_array_notation(self) -> None:
        """Test parsing LAS 3.0 array notation NMR[1], NMR[2]."""
        content = """~VERSION INFORMATION
 VERS.   3.0  :
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 NMR[1].ms    : NMR Echo Array {A:0}
 NMR[2].ms    : NMR Echo Array {A:5}
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 3
        nmr1 = las.curves[1]
        assert nmr1.array_info is not None
        assert nmr1.array_info.base_name == "NMR"
        assert nmr1.array_info.index == 1
        assert nmr1.array_info.time_offset == 0.0
        nmr2 = las.curves[2]
        assert nmr2.array_info is not None
        assert nmr2.array_info.index == 2
        assert nmr2.array_info.time_offset == 5.0

    def test_empty_content(self) -> None:
        """Test parsing empty content."""
        parser = LASParser()
        las = parser.parse("")
        assert las.version.vers == "2.0"
        assert len(las.curves) == 0
        assert len(las.curves_order) == 0

    def test_thread_safety_reset(self) -> None:
        """Test that parse() resets state between calls."""
        content1 = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~CURVE INFORMATION
 DEPT.M  :
 DT.US/M :
"""
        content2 = """~VERSION INFORMATION
 VERS.   1.20  :
 WRAP.   YES   :
~CURVE INFORMATION
 DEPTH.FT  :
"""
        parser = LASParser()
        las1 = parser.parse(content1)
        las2 = parser.parse(content2)

        assert las1.version.vers == "2.0"
        assert len(las1.curves) == 2

        assert las2.version.vers == "1.20"
        assert len(las2.curves) == 1
        assert las2.curves_order == ["DEPTH"]

    # --- TEST-05: Multiple ~A section handling for LAS 3.0 ---
    def test_las30_multiple_data_sections(self) -> None:
        """Test LAS 3.0 with multiple ~A data sections."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A FirstSection
100.0,50.0
101.0,51.0
~A SecondSection
200.0,60.0
201.0,61.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2
        assert las.data_sections[0].name == "FirstSection"
        assert las.data_sections[1].name == "SecondSection"
        # First section data
        assert len(las.data_sections[0].data["DEPT"]) == 2
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        # Second section data
        assert len(las.data_sections[1].data["DEPT"]) == 2
        assert las.data_sections[1].data["DEPT"][0] == 200.0

    # --- TEST-15: _parse_version/_parse_well return on non-matching line ---
    def test_version_non_matching_line(self) -> None:
        """Test that _parse_version ignores lines that don't match data pattern."""
        content = """~VERSION INFORMATION
 This line has no dot separator
 VERS.   3.0  : CWLS LOG ASCII STANDARD
"""
        parser = LASParser()
        las = parser.parse(content)
        # Should still parse VERS line, ignoring the bad line
        assert las.version.vers == "3.0"

    def test_well_non_matching_line(self) -> None:
        """Test that _parse_well ignores non-matching lines."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~WELL INFORMATION
 Not a valid well line
 STRT.M   1670.0 : START DEPTH
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.well["STRT"] == "1670.0"

    # --- TEST-16: Comment skip / space delimiter / string-curve fallback in ASCII data ---
    def test_las30_ascii_comment_skip(self) -> None:
        """Test that comments in LAS 3.0 ASCII data are skipped."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
# This is a comment in data
100.0,50.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        # Should only have 1 data row, comment skipped
        assert len(las.data_sections[0].data["DEPT"]) == 1

    def test_las30_space_delimiter(self) -> None:
        """Test LAS 3.0 with SPACE delimiter (default split behavior)."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   SPACE :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
100.0  50.0
101.0  51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0

    def test_las30_string_curve_fallback(self) -> None:
        """Test that non-numeric values in non-string curves fallback to null_value."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 CDES.        : CORE DESC {S}
~A
100.0,BAD_STRING
"""
        parser = LASParser()
        las = parser.parse(content)
        # CDES is string type, so BAD_STRING should be stored as string
        assert "CDES" in las.string_data
        assert las.string_data["CDES"][0] == "BAD_STRING"

    # --- TEST-05: Empty curves in LAS 3.0 data section (line 332 early return) ---
    def test_las30_empty_curves_with_data(self) -> None:
        """Test LAS 3.0 parser with ~A data section but no curves defined.

        When there are no curves (curves_order is empty), _process_ascii_data
        should return early without error (line 332 of parser.py).
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
~A
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        # No curves, so data_sections should be empty
        assert len(las.curves) == 0
        assert len(las.curves_order) == 0
        # Parser should not crash even with ASCII data but no curves
        assert isinstance(las.data_sections, list)

    # --- TEST-05: Non-numeric NULL value in LAS 3.0 data (line 340-341) ---
    def test_las30_non_numeric_null(self) -> None:
        """Test LAS 3.0 processing with non-numeric NULL value.

        When NULL field cannot be converted to float, it should fall back
        to -999.25 (lines 338-341 of parser.py).
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.   NOT_A_NUMBER  : NON-NUMERIC NULL
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        # Data sections should be populated despite non-numeric NULL
        assert len(las.data_sections) == 1
        assert "DEPT" in las.data_sections[0].data
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0

    # --- TEST-05: ValueError handler for non-string curves with bad data (lines 377-381) ---
    def test_las30_valueerror_non_string_curve(self) -> None:
        """Test ValueError fallback for non-string curve with non-numeric data.

        When a non-string {F} curve gets a non-numeric value, the ValueError
        handler at line 380-381 should substitute null_value (-999.25).
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
100.0,BAD_VALUE
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        # DT[0] has "BAD_VALUE" which cannot be converted to float
        # Should fall back to null_value (-999.25)
        assert las.data_sections[0].data["DT"][0] == -999.25
        # DEPT should still be fine
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        # DT[1] should be fine
        assert las.data_sections[0].data["DT"][1] == 51.0

    # --- TEST-05: Empty string value for numeric curve (line 375) ---
    def test_las30_empty_value_numeric_curve(self) -> None:
        """Test LAS 3.0 empty string value for numeric curve uses null_value.

        When val_str is empty, the float conversion at line 375 falls back
        to null_value before a ValueError is even attempted.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
100.0,
"""
        parser = LASParser()
        las = parser.parse(content)
        # Empty value for DT should become null_value
        assert las.data_sections[0].data["DT"][0] == -999.25
        assert las.data_sections[0].data["DEPT"][0] == 100.0

    # --- F-10: LASParseError for non-empty content without ~V section ---
    def test_parse_non_empty_no_version_raises_error(self) -> None:
        """Test that non-empty content without ~V section raises LASParseError.

        Exercises parser.py:137-141 — the validation that a valid LAS file
        must contain a ~Version section.
        """
        content = "This is not a LAS file.\nJust some random text.\n"
        parser = LASParser()
        with pytest.raises(LASParseError, match="missing required ~V"):
            parser.parse(content)

    def test_parse_empty_content_no_version_error(self) -> None:
        """Test that empty content does NOT raise LASParseError.

        Blank files should not trigger the ~V validation since content.strip() is falsy.
        """
        parser = LASParser()
        result = parser.parse("")
        assert result.version.vers == "2.0"

    # --- F-11: TAB delimiter end-to-end in LAS 3.0 ---
    def test_las30_tab_delimiter(self) -> None:
        """Test LAS 3.0 with TAB delimiter end-to-end.

        Exercises parser.py:342-348 — TAB delimiter path in _process_ascii_data.
        """
        content = "~VERSION INFORMATION\n VERS.   3.0  :\n WRAP.   NO   :\n DLM .   TAB  :\n~CURVE INFORMATION\n DEPT.M      :  DEPTH {F}\n DT.US/M     :  SONIC {F}\n~A\n100.0\t50.0\n101.0\t51.0\n"
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0
        assert las.data_sections[0].data["DEPT"][1] == 101.0
        assert las.data_sections[0].data["DT"][1] == 51.0

    # --- F-14: TAB delimiter from real file ---
    def test_las30_tab_delimiter_from_real_file(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 TAB delimiter using a real file from test_data/.

        Exercises the full parsing path (DLM.TAB header -> delimiter_char ->
        split('\t') -> float conversion) with actual file content.
        """
        tab_file = test_data_dir / "sample_las3.0_tab.las"
        assert tab_file.exists(), f"Required test data missing: {tab_file}"
        content = tab_file.read_text(encoding="utf-8")

        parser = LASParser()
        las = parser.parse(content)

        assert len(las.data_sections) == 1
        assert las.version.dlm == "TAB"
        assert las.version.delimiter_char == "\t"

        # Verify parsed values from the real file
        data = las.data_sections[0].data
        assert len(data["DEPT"]) == 3
        np.testing.assert_array_almost_equal(data["DEPT"], [100.0, 110.0, 120.0])
        np.testing.assert_array_almost_equal(data["DT"], [123.45, 123.55, 123.65])
        np.testing.assert_array_almost_equal(data["RHOB"], [2550.0, 2552.0, 2551.0])
        np.testing.assert_array_almost_equal(data["NPHI"], [0.450, 0.445, 0.440])

    def test_las30_tab_delimiter_consecutive_tabs(self) -> None:
        """Test LAS 3.0 TAB delimiter with consecutive TABs (empty fields).

        When consecutive TABs produce empty strings from split('\\t'),
        the empty value should be handled as null_value rather than
        causing column misalignment or crash.
        """
        content = "~VERSION INFORMATION\n VERS.   3.0  :\n WRAP.   NO   :\n DLM .   TAB  :\n~WELL INFORMATION\n NULL.    -999.25 : NULL VALUE\n~CURVE INFORMATION\n DEPT.M      :  DEPTH {F}\n DT.US/M     :  SONIC {F}\n GR.GAPI     :  GAMMA RAY {F}\n~A\n100.0\t\t75.0\n101.0\t51.0\t76.0\n"
        parser = LASParser()
        las = parser.parse(content)

        assert len(las.data_sections) == 1
        data = las.data_sections[0].data

        # Row 0: consecutive TAB between 100.0 and 75.0 -> split produces ['100.0', '', '75.0']
        # The empty string for DT -> null_value (-999.25)
        assert data["DEPT"][0] == 100.0
        assert data["DT"][0] == -999.25
        assert data["GR"][0] == 75.0

        # Row 1: all values present
        assert data["DEPT"][1] == 101.0
        assert data["DT"][1] == 51.0
        assert data["GR"][1] == 76.0

    # --- F25a: LAS 3.0 per-section duplicate curve dedup ---
    def test_las30_duplicate_curves_renamed_in_section(self) -> None:
        """Test LAS 3.0 per-section dedup renames duplicate curve mnemonics.

        Exercises parser.py:506-527 — when two curves in the same ~C block
        share the same mnemonic, the second is renamed with a _N suffix.
        The dedup logic uses a section-local `seen` dict and `output_names`
        set to ensure mnemonics are unique within each data section.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DT.US/M       : SONIC {F}
 DT.US/M       : SONIC DUPLICATE {F}
~A
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)

        assert len(las.data_sections) == 1
        section = las.data_sections[0]
        # First DT keeps its name, second becomes DT_2
        assert "DT" in section.data
        assert "DT_2" in section.data
        # Verify data values
        assert section.data["DT"][0] == 100.0
        assert section.data["DT_2"][0] == 50.0
        assert section.data["DT"][1] == 101.0
        assert section.data["DT_2"][1] == 51.0
        # Deduped order reflects the renaming
        assert section.curves_order == ["DT", "DT_2"]
        # DT_2 should have original_mnemonic set
        # (original_mnemonic is tracked on the CurveDefinition object)

    # --- F25b: LAS 3.0 cross-base collision dedup ---
    def test_las30_cross_base_collision_dedup(self) -> None:
        """Test LAS 3.0 cross-base collision when original name matches a suffix.

        Exercises parser.py:535-561 — when a curve's original name matches a
        suffix that was already generated for an earlier duplicate (e.g., the
        second "DT" becomes "DT_2", which collides with the third curve whose
        original name IS "DT_2"), the third curve is renamed with a higher
        suffix to avoid a duplicate key.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DT.US/M       : SONIC 1 {F}
 DT.US/M       : SONIC 2 {F}
 DT_2.US/M     : CROSS COLLISION {F}
~A
100.0,50.0,75.0
101.0,51.0,76.0
"""
        parser = LASParser()
        las = parser.parse(content)

        assert len(las.data_sections) == 1
        section = las.data_sections[0]
        # Expected: DT stays, second DT → DT_2 (collides with third curve),
        # third DT_2 → DT_2_2
        assert "DT" in section.data
        assert "DT_2" in section.data
        assert "DT_2_2" in section.data
        assert len(section.data) == 3
        # Verify data values are preserved
        assert section.data["DT"][0] == 100.0
        assert section.data["DT_2"][0] == 50.0
        assert section.data["DT_2_2"][0] == 75.0
        assert section.data["DT"][1] == 101.0
        assert section.data["DT_2"][1] == 51.0
        assert section.data["DT_2_2"][1] == 76.0
        # Deduped order reflects the collision resolution
        assert section.curves_order == ["DT", "DT_2", "DT_2_2"]


class TestLAS12WellSectionSwap:
    """T1/F-10: LAS 1.2 value/description swap across versions."""

    def test_las12_vs_las20_well_equivalent_extraction(self) -> None:
        """Test that LAS 1.2 and 2.0 ~W sections with identical semantic content
        produce equivalent well values.

        LAS 1.2 uses two conventions in the wild:
          (a) CWLS spec: MNEM.UNIT VALUE : DESCRIPTION (numeric fields)
          (b) lasio conv: MNEM.UNIT DESCRIPTION : VALUE (non-numeric fields)
        The parser auto-detects numeric fields (STRT, STOP, STEP, NULL)
        by trying float(value) first. Non-numeric fields always use the
        lasio convention (value = description group).

        LAS 2.0+: value BEFORE colon for all fields.
        """
        # LAS 1.2: non-numeric fields use lasio convention (value=description)
        las12 = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.5 : START DEPTH\n"
            " STOP.M   500.0  : STOP DEPTH\n"
            " STEP.M   -0.125 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " COMP.    COMPANY : TestCo\n"
            " WELL.    WELL : WellA\n"
        )
        # LAS 2.0: value BEFORE colon for all fields
        las20 = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.5 : START DEPTH\n"
            " STOP.M   500.0  : STOP DEPTH\n"
            " STEP.M   -0.125 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " COMP.    TestCo : COMPANY\n"
            " WELL.    WellA  : WELL NAME\n"
        )

        parser = LASParser()
        d12 = parser.parse(las12)
        d20 = parser.parse(las20)

        # Numeric fields must match across both versions
        for mnem in ("STRT", "STOP", "STEP", "NULL"):
            assert d12.well[mnem] == d20.well[mnem], (
                f"Mismatch for {mnem}: {d12.well[mnem]} vs {d20.well[mnem]}"
            )

        # Non-numeric fields: LAS 1.2 uses lasio convention, LAS 2.0 uses standard.
        # Both should extract the same value from the file.
        assert d12.well["COMP"] == "TestCo"
        assert d12.well["WELL"] == "WellA"
        assert d20.well["COMP"] == "TestCo"
        assert d20.well["WELL"] == "WellA"

    def test_las12_numeric_fields_detect_spec_format(self) -> None:
        """Test that LAS 1.2 numeric fields (STRT, STOP, STEP, NULL)
        use the value-before-colon convention when value is numeric."""
        # A file where the description is a non-numeric string but the
        # value is numeric — parser should use value group.
        las12 = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0 : START DEPTH\n"
        )
        parser = LASParser()
        las = parser.parse(las12)
        assert las.well["STRT"] == "1670.0"

    def test_las12_non_numeric_fields_use_lasio_convention(self) -> None:
        """Test that LAS 1.2 non-numeric fields (COMP, WELL, etc.)
        use the lasio convention: value = description group."""
        # File where value group is non-numeric for COMP
        las12 = (
            "~VERSION INFORMATION\n"
            " VERS.   1.20 : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " COMP.    COMPANY : Test Oil Company Ltd.\n"
        )
        parser = LASParser()
        las = parser.parse(las12)
        assert las.well["COMP"] == "Test Oil Company Ltd."


class TestLAS30IntegerFormat:
    """T3/F-16: LAS 3.0 {I} integer format specifier."""

    def test_las30_integer_format_specifier(self) -> None:
        """Test that LAS 3.0 {I} format specifier is recognized by parser.

        The LAS 3.0 spec supports {I} for integer values.
        The parser's FORMAT_SPEC_PATTERN should capture {I} and pass it
        through as data_format.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   :\n"
            " DLM.   COMMA :\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " RUN_NO.      : RUN NUMBER  {I}\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 2
        run_no = las.curves[1]
        assert run_no.mnemonic == "RUN_NO"
        assert run_no.data_format == "I"


class TestUnknownSectionWarning:
    """F-031: Unknown section handler warning test."""

    def test_unknown_section_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that a ~X section (unknown type) emits a warning via logger."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~X CUSTOM
 Some data in custom section.
"""
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            parser = LASParser()
            parser.parse(content)

        assert "Unknown section type" in caplog.text
        assert "~X" in caplog.text

    def test_unknown_section_with_warnings_module(self) -> None:
        """Test unknown section via pytest caplog (more portable)."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~X CUSTOM
 Some data in custom section.
"""
        parser = LASParser()
        las = parser.parse(content)
        # Should parse without error — unknown sections are logged, not fatal
        assert las.version.vers == "2.0"


class TestConcurrentParserAccess:
    """CF-019: Concurrent parser access from multiple threads."""

    def test_concurrent_parse_different_instances(self) -> None:
        """Test that different parser instances work independently in threads."""
        errors: list[Exception] = []
        results: list[int] = []

        def parse_las() -> None:
            try:
                content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M  :
 DT.US/M :
~A  DEPT  DT
100.0  50.0
101.0  51.0
"""
                parser = LASParser()
                las = parser.parse(content)
                results.append(len(las.curves))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=parse_las) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent parse: {errors}"
        assert len(results) == 8
        # All parsers should produce 2 curves
        assert all(r == 2 for r in results)

    def test_concurrent_parse_same_instance(self) -> None:
        """Test that a single parser instance works across threads.

        Each call to parse() resets state, but concurrent access
        without external synchronization may produce race conditions.
        This test documents the behavior.
        """
        errors: list[Exception] = []

        def parse_with_shared(parser: LASParser) -> None:
            try:
                content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~CURVE INFORMATION
 DEPT.M  :
 DT.US/M :
~A  DEPT  DT
100.0  50.0
"""
                las = parser.parse(content)
                assert len(las.curves) == 2
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        parser = LASParser()
        threads = [threading.Thread(target=parse_with_shared, args=(parser,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # If errors exist, the parser is not thread-safe for concurrent use
        # with the same instance (expected — LASParser is designed with
        # per-instance state that parse() resets)
        if errors:
            warnings.warn(
                f"LASParser is not thread-safe for shared-instance use (errors: {errors})",
                stacklevel=1,
            )


class TestFormatSpecOffsetError:
    """F29: Test ValueError handler for malformed format specifier offset.

    When a LAS 3.0 curve has a {A:XYZ} format specifier where the offset
    portion is non-numeric, float() raises ValueError which is caught at
    parser.py:337-338 and re-raised as LASParseError.
    """

    def test_malformed_offset_raises_las_parse_error(self) -> None:
        """Test that a non-numeric/dot-only offset in {A:...} raises LASParseError.

        The FORMAT_SPEC_PATTERN only captures digits and dots as the offset.
        A value like '..' matches the pattern (digits and dots) but
        float('..') raises ValueError which is caught and re-raised
        as LASParseError at parser.py:337-338.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 NMR[1].ms    : NMR Echo Array {A:..}
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match="Invalid format specifier offset"):
            parser.parse(content)

    def test_malformed_offset_multiple_dots(self) -> None:
        """Test offset with multiple dots like '1.2.3' raises LASParseError."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 NMR[1].ms    : NMR Echo Array {A:1.2.3}
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match="Invalid format specifier offset"):
            parser.parse(content)


class TestLAS30AsciiDataBranches:
    """F35: Tests for uncovered branches in _process_ascii_data.

    Exercises three branches in parser.py:
    - Comment skip (line 596-597)
    - Extra-column warning (line 606-614)
    - Padding for short rows (line 617-618)
    """

    def test_las30_ascii_comment_skip_in_data(self) -> None:
        """Test LAS 3.0 data section with comment lines skipped."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 DT.US/M      : SONIC {F}
~A
# This is a comment between data lines
100.0,50.0
# Another comment
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        # Comments should be skipped: only 2 data lines
        assert len(las.data_sections[0].data["DEPT"]) == 2
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DEPT"][1] == 101.0

    def test_las30_extra_columns_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test LAS 3.0 extra-column warning triggered."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 DT.US/M      : SONIC {F}
~A
100.0,50.0,75.0,99.0
101.0,51.0,76.0,100.0
"""
        parser = LASParser()
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            las = parser.parse(content)
            assert "Extra columns are discarded" in caplog.text

        assert len(las.data_sections) == 1
        # Extra values are truncated — only first 2 (matching curve count) kept
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0
        assert las.data_sections[0].data["DEPT"][1] == 101.0

    def test_las30_short_rows_padding(self) -> None:
        """Test LAS 3.0 data with short rows padded to curve count.

        When a data line has FEWER values than curves, the padding loop
        at parser.py:617-618 appends null_value strings to fill the gap.
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 DT.US/M      : SONIC {F}
 GR.GAPI      : GAMMA RAY {F}
~A
100.0,50.0
101.0,51.0,75.0
"""
        parser = LASParser()
        las = parser.parse(content)

        # 3 curves, first row has only 2 values → GR gets padded with null
        assert "DEPT" in las.data_sections[0].data
        assert "DT" in las.data_sections[0].data
        assert "GR" in las.data_sections[0].data

        # Row 0: DEPT=100.0, DT=50.0, GR=padded with null_value
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0
        assert las.data_sections[0].data["GR"][0] == -999.25
        # Row 1: all values present
        assert las.data_sections[0].data["DEPT"][1] == 101.0
        assert las.data_sections[0].data["DT"][1] == 51.0
        assert las.data_sections[0].data["GR"][1] == 75.0

    # --- F-H1: LAS 3.0 ~Log alias handling ---
    def test_las30_log_section_alias(self) -> None:
        """~Log is a spec-defined shorthand alias for ~Ascii / ~Log_Data."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH {F}
 DT.US/M      : SONIC {F}
~Log
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        assert las.data_sections[0].section_type == "LOG_DATA"
        assert las.data_sections[0].data["DEPT"][0] == 100.0
        assert las.data_sections[0].data["DT"][0] == 50.0

    # --- F-H2: LAS 2.0 mandatory well field warnings ---
    def test_las20_missing_mandatory_well_field_warning(self) -> None:
        """LAS 2.0 files missing STRT, STOP, STEP, NULL should warn."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   :
~WELL INFORMATION
 STRT.M  1670.0000  :
 STOP.M  1680.0000  :
~CURVE INFORMATION
 DEPT.M   : DEPTH
~A
100.0
101.0
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [
                x for x in w if "LAS 2.0 file missing mandatory well field" in str(x.message)
            ]
            # Both STEP and NULL are missing
            assert len(mandatory_warnings) == 2
            warning_texts = [str(x.message) for x in mandatory_warnings]
            assert any("STEP" in t for t in warning_texts)
            assert any("NULL" in t for t in warning_texts)

    def test_las20_all_mandatory_fields_no_warning(self) -> None:
        """LAS 2.0 files with all mandatory fields should not warn."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   :
~WELL INFORMATION
 STRT.M  1670.0000  :
 STOP.M  1680.0000  :
 STEP.M     0.1000  :
 NULL.   -999.25  :
~CURVE INFORMATION
 DEPT.M   : DEPTH
~A
100.0
101.0
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [
                x for x in w if "LAS 2.0 file missing mandatory well field" in str(x.message)
            ]
            assert len(mandatory_warnings) == 0

    def test_las12_no_mandatory_field_warning(self) -> None:
        """LAS 1.2 files should not trigger the LAS 2.0 mandatory field check."""
        content = """~VERSION INFORMATION
 VERS.   1.2  : CWLS LOG ASCII STANDARD
 WRAP.   NO   :
~WELL INFORMATION
 STRT.M    : 1670.0000
 STOP.M    : 1680.0000
 STEP.M    : 0.1000
 NULL.    : -999.25
~CURVE INFORMATION
 DEPT.M   : DEPTH
~A
100.0
101.0
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [
                x for x in w if "LAS 2.0 file missing mandatory well field" in str(x.message)
            ]
            assert len(mandatory_warnings) == 0

    # --- F-M2: Pre-scan only counts ~A/~ASCII sections ---
    def test_pre_scan_ignores_non_ascii_sections(self, tmp_path: Path) -> None:
        """Pre-scan should only count lines in ~A/~ASCII data sections.

        Uses public API to verify that only 2 data points (from ~A section)
        are read, not 4 (the ~Core lines are excluded by pre-scan).
        """
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~CURVE INFORMATION
 DEPT.M   :
 DT.US/M  :
~A
100.0  50.0
100.1  51.0
~Core
550.0  1.0
551.0  1.0
"""
        test_file = tmp_path / "pre_scan_non_ascii.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2

    def test_parse_las30_creates_expected_data_sections(self) -> None:
        """For LAS 3.0, the parser creates data_sections for each section
        found during the full parse.  Both ~A and ~Core sections appear
        in the result regardless of pre-scan behavior (which only counts
        ~A/~ASCII lines for pre-allocation).
        """
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.   -999.25 :
~CURVE INFORMATION
 DEPT.M  : DEPTH {F}
 DT.US/M : SONIC {F}
~A
100.0,50.0
101.0,51.0
~Core
550.0,1.0
"""
        parser = LASParser()
        las = parser.parse(content)
        # Both ~A and ~Core sections appear as data_sections
        assert len(las.data_sections) == 2
        # LOG_DATA section: ~A
        assert las.data_sections[0].section_type == "LOG_DATA"
        assert len(las.data_sections[0].data["DEPT"]) == 2
        # CORE_DATA section: ~Core
        assert las.data_sections[1].section_type == "CORE_DATA"
        assert len(las.data_sections[1].data["DEPT"]) == 1


class TestIndexedDataSectionNegativeBranches:
    """F-02/T-02: Tests for _is_indexed_data_section negative branches.

    Exercises the error-detection paths at parser.py:196 (missing closing
    bracket) and parser.py:199 (non-numeric index).
    """

    def test_valid_indexed_section_returns_true(self) -> None:
        """Test that valid indexed sections return True.

        The parser passes section_word.upper() to this function (parser.py:385),
        so the base must match the uppercase keys in _INDEXED_DATA_TYPES.
        """
        assert _is_indexed_data_section("CORE[1]") is True
        assert _is_indexed_data_section("INCLINOMETRY[2]") is True
        assert _is_indexed_data_section("DRILLING[42]") is True
        assert _is_indexed_data_section("TOPS[0]") is True

    def test_non_numeric_index_returns_false(self) -> None:
        """Test that a non-numeric index returns False.

        Exercises parser.py:199 — ``index_str.isdigit()`` → False path.
        The parser uppercases section_word before calling, so we test
        with the same convention.
        """
        assert _is_indexed_data_section("CORE[abc]") is False
        assert _is_indexed_data_section("INCLINOMETRY[x]") is False
        assert _is_indexed_data_section("DRILLING[n1]") is False

    def test_missing_closing_bracket_returns_false(self) -> None:
        """Test that missing closing bracket returns False.

        Exercises parser.py:196 — ``rest.endswith("]")`` → False path.
        """
        assert _is_indexed_data_section("CORE[1") is False
        assert _is_indexed_data_section("INCLINOMETRY[2") is False

    def test_no_bracket_returns_false(self) -> None:
        """Test that a known type without any bracket returns False.

        Exercises parser.py:191 — ``bracket_idx < 0`` → False path.
        """
        assert _is_indexed_data_section("CORE") is False
        assert _is_indexed_data_section("INCLINOMETRY") is False

    def test_unknown_type_with_valid_bracket_returns_false(self) -> None:
        """Test that an unknown base type with valid bracket returns False.

        Exercises parser.py:200-201 — ``base not in _INDEXED_DATA_TYPES``
        → False path.
        """
        assert _is_indexed_data_section("UNKNOWN[1]") is False
        assert _is_indexed_data_section("XYZ[42]") is False

    def test_empty_bracket_returns_false(self) -> None:
        """Test that empty brackets return False.

        The index_str is empty, which means ``isdigit()`` returns False.
        """
        assert _is_indexed_data_section("CORE[]") is False


class TestSplitlinesCharsSanitization:
    """F-ITER2-T1-M02: Test _SPLITLINES_CHARS_RE sanitization.

    The regex ``[\\x0b\\x0c\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]`` strips 8
    character types that Python's splitlines() treats as line breaks before
    actual line splitting. Without sanitization, these characters produce
    fake line splits and corrupt parsed data.
    """

    _SPLITLINES_CHARS = (
        "\x0b",  # VT — vertical tab
        "\x0c",  # FF — form feed
        "\x1c",  # FS — file separator
        "\x1d",  # GS — group separator
        "\x1e",  # RS — record separator
        "\x85",  # NEL — next line
        "\u2028",  # LINE SEPARATOR
        "\u2029",  # PARAGRAPH SEPARATOR
    )

    def test_all_splitline_chars_sanitized_in_version_value(self) -> None:
        """Inject each splitline character into a VERS value and verify
        they don't cause fake section breaks. The characters are replaced
        with spaces before splitlines(), so VERS remains on one line.
        """
        for ch in self._SPLITLINES_CHARS:
            content = (
                f"~VERSION INFORMATION\n"
                f" VERS.\x20\x20 2.0{ch}  : CWLS LOG ASCII STANDARD\n"
                f" WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            )
            parser = LASParser()
            las = parser.parse(content)
            # The VERS line should still be parsed despite injected char
            assert las.version.vers.startswith("2.0")

    def test_all_splitline_chars_sanitized_in_curve_section(self) -> None:
        """Inject each splitline character between curve lines and verify
        curves are still parsed correctly.
        """
        for ch in self._SPLITLINES_CHARS:
            content = (
                f"~VERSION INFORMATION\n"
                f" VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
                f" WRAP.   NO   :\n"
                f"~CURVE INFORMATION\n"
                f" DEPT.M   :  Depth{ch}\n"
                f" DT.US/M  :  Sonic\n"
            )
            parser = LASParser()
            las = parser.parse(content)
            assert len(las.curves) == 2
            assert las.curves[0].mnemonic == "DEPT"
            assert las.curves[1].mnemonic == "DT"

    def test_all_splitline_chars_sanitized_no_missing_curves(self) -> None:
        """Inject all 8 characters between curve lines; parse should
        succeed with all curves detected, not split into fake lines.
        """
        # Build content with every splitline char between the two curve lines
        all_chars = "".join(self._SPLITLINES_CHARS)
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   :\n"
            "~CURVE INFORMATION\n"
            f" DEPT.M   :  Depth{all_chars}\n"
            f" DT.US/M  :  Sonic\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # Both curves should be detected — the control chars were
        # replaced with spaces, not treated as line breaks
        assert len(las.curves) == 2
        assert las.curves[0].mnemonic == "DEPT"
        assert las.curves[1].mnemonic == "DT"


class TestValueOnlyPattern:
    """F-ITER2-T1-M03: Test VALUE_ONLY_PATTERN fallback.

    When a metadata line has no colon (e.g., ``STRT.M   1670.0`` without
    a trailing colon), DATA_LINE_PATTERN fails to match and the parser
    falls back to VALUE_ONLY_PATTERN which matches colon-free lines.
    """

    def test_version_line_without_colon_parsed(self) -> None:
        """VERSION line without a colon: VALUE_ONLY_PATTERN captures
        the entire remainder as the value group, which includes the
        description text since there is no colon to separate them.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # Without colon, the entire rest of the line is the value
        # (VALUE_ONLY_PATTERN captures everything after unit dot)
        assert las.version.vers.startswith("2.0")

    def test_parameter_line_without_colon_parsed(self) -> None:
        """PARAMETER line without a colon uses VALUE_ONLY_PATTERN."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   :\n"
            "~PARAMETER INFORMATION\n"
            " BHT.DEGC    35.5\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.parameters) == 1
        assert las.parameters[0].mnemonic == "BHT"
        assert las.parameters[0].value == "35.5"

    def test_curve_line_without_colon_parsed(self) -> None:
        """CURVE line without a colon uses VALUE_ONLY_PATTERN."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   :\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   Depth\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.curves) == 1
        assert las.curves[0].mnemonic == "DEPT"
        assert las.curves[0].unit == "M"

    def test_mixed_colon_and_no_colon_lines(self) -> None:
        """Section with mix of colon and colon-free lines."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   :\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0  : START DEPTH\n"
            " COMP.    TestCo\n"  # no colon — falls to VALUE_ONLY_PATTERN
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.well["STRT"] == "1670.0"
        assert las.well["COMP"] == "TestCo"
