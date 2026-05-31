"""Tests for LAS file parser."""

from __future__ import annotations

import pytest

from pylasdev.exceptions import LASParseError
from pylasdev.parser import LASParser


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

    def test_pre_scan_counts_data_lines(self) -> None:
        """Test that pre-scan correctly counts ASCII data lines."""
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
        parser = LASParser()
        parser.parse(content)
        assert parser._data_line_count == 3

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

