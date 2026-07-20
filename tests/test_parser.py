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
        to -999.25 (parser.py:1879 via _get_null_value).  Data values that
        equal the fallback null sentinel (-999.25) must be stored as-is,
        and non-numeric data values must be replaced with the null_value.
        This exercises BOTH the _get_null_value ValueError except path AND
        the null-value replacement in _to_finite_float.
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
-999.25,BAD_VALUE
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        data = las.data_sections[0].data
        assert data["DEPT"][0] == 100.0
        assert data["DT"][0] == 50.0
        # Row 2: DEPT="-999.25" (the actual null sentinel from the fallback),
        #         DT="BAD_VALUE" (non-numeric → replaced with null_value).
        assert data["DEPT"][2] == -999.25, (
            "DEPTH null sentinel -999.25 must be stored as -999.25"
        )
        assert data["DT"][2] == -999.25, (
            "Non-numeric BAD_VALUE must be replaced with null_value (-999.25)"
        )

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

    # ── F-EX-01: Deferred data section boundary preservation ──────

    def test_las30_deferred_data_separate_sections(self) -> None:
        """F-EX-01: Deferred pre-~V data is NOT merged into post-~V DataSection.

        When a LAS 3.0 file has ~A sections both BEFORE and AFTER ~V,
        the deferred pre-~V data lines must produce a SEPARATE DataSection
        — not be merged into the post-~V section.  Merging distinct sections
        with different curve assignments and depth ranges produces corrupted
        data in reversed order.
        """
        content = """~A
100.0,50.0
101.0,51.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
200.0,60.0
201.0,61.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2, (
            f"Expected 2 DataSections (pre-~V and post-~V), got {len(las.data_sections)}"
        )
        # First section: pre-~V data (name auto-generated since ~V not yet parsed)
        sec0 = las.data_sections[0]
        assert len(sec0.data["DEPT"]) == 2
        assert sec0.data["DEPT"][0] == 100.0
        assert sec0.data["DEPT"][1] == 101.0
        assert sec0.data["DT"][0] == 50.0
        assert sec0.data["DT"][1] == 51.0
        # Second section: post-~V data
        sec1 = las.data_sections[1]
        assert len(sec1.data["DEPT"]) == 2
        assert sec1.data["DEPT"][0] == 200.0
        assert sec1.data["DEPT"][1] == 201.0
        assert sec1.data["DT"][0] == 60.0
        assert sec1.data["DT"][1] == 61.0

    def test_las30_deferred_data_single_section_no_separation(self) -> None:
        """F-EX-01: Single ~A section pre-~V with no post-~V data.

        When there is ONLY a pre-~V ~A section (no subsequent data section),
        the deferred data is appended to the empty _ascii_data_lines and
        processed as one DataSection — the normal single-section path.
        """
        content = """~A
100.0,50.0
101.0,51.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        sec0 = las.data_sections[0]
        assert len(sec0.data["DEPT"]) == 2
        assert sec0.data["DEPT"][0] == 100.0

    # ── F-EX-02: First-block-only semantics ──────────────────────

    def test_multi_a_first_block_only(self, tmp_path: Path) -> None:
        """F-EX-02: Multi-~A files ingest only the FIRST contiguous ~A block.

        When a LAS 2.0 file has ``~A(data1) ~OTHER ~A(data2)``, only data1
        is ingested.  The parser's _pre_scan counts per_block_counts[0] and
        the data reader breaks at the first non-~A section header — both
        sides agree on first-block-only semantics.

        Uses read_las_file() to exercise the full parser→data_reader pipeline.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
            "~OTHER\n"
            "Freeform text here.\n"
            "~A  DEPT  DT\n"
            "200.0  60.0\n"
        )
        test_file = tmp_path / "multi_a_first_block.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # Only the first ~A block's 2 data lines are ingested.
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2
        np.testing.assert_array_almost_equal(
            data["logs"]["DEPT"], [100.0, 101.0]
        )
        np.testing.assert_array_almost_equal(
            data["logs"]["DT"], [50.0, 51.0]
        )

    # ── F-S9-01: DataSection idx increment after save/swap ─────────

    def test_deferred_data_section_names_unique(self) -> None:
        """F-S9-01: DataSection names are unique after save/swap path.

        When ~A(pre-~V) ~V ~C ~A(post-~V) produces two DataSections via
        the save/swap path in _replay_deferred_well(), the idx increment
        ensures both sections get distinct names (Section_0, Section_1)
        rather than the same auto-generated name.
        """
        content = """~A
100.0,50.0
101.0,51.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A
200.0,60.0
201.0,61.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2
        assert las.data_sections[0].name != las.data_sections[1].name, (
            f"DataSection names must be unique: got '{las.data_sections[0].name}' "
            f"for both sections"
        )

    def test_deferred_data_pre_curve_scoping(self) -> None:
        """F-S9-02: Pre-~V DataSection has correct curve scoping.

        When a pre-~V ~C exists before the deferred ~A section, the
        pre-~V DataSection must be scoped to the full curve range
        (allowing pre-~V data values to map to their correct curves)
        rather than the post-~V curve range which would cross-assign
        values to wrong curves.

        Without this fix: pre-~V data (100.0, 50.0) with 2 columns
        gets scoped to post-~V curves (only GR — 1 curve), causing
        DEPT=100.0 to be stored as GR and DT=50.0 silently dropped.
        """
        content = """~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A PRE_SECTION
100.0,50.0
101.0,51.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 GR.API       : GAMMA  {F}
~A POST_SECTION
200.0
201.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2

        # Pre-~V section: should have data for DEPT and DT (pre-~V curves).
        # GR (post-~V curve) gets null — acceptable since pre-~V data has
        # no values for it.
        sec_pre = las.data_sections[0]
        assert "DEPT" in sec_pre.data, (
            f"Pre-~V section must contain DEPT data, got curves: {list(sec_pre.data.keys())}"
        )
        assert "DT" in sec_pre.data, (
            f"Pre-~V section must contain DT data, got curves: {list(sec_pre.data.keys())}"
        )
        assert sec_pre.data["DEPT"][0] == 100.0
        assert sec_pre.data["DEPT"][1] == 101.0
        assert sec_pre.data["DT"][0] == 50.0
        assert sec_pre.data["DT"][1] == 51.0

        # Post-~V section: only GR data.
        sec_post = las.data_sections[1]
        assert sec_post.data["GR"][0] == 200.0
        assert sec_post.data["GR"][1] == 201.0

    def test_deferred_data_section_name_not_corrupted(self) -> None:
        """F-S9-02: Pre-~V DataSection name is not corrupted by post-~V context.

        When the pre-~V ~A section has a different name than post-~V ~A,
        the pre-~V DataSection must NOT inherit the post-~V section's name.
        The pre-~V section falls back to auto-generated "Section_N" since
        its original name was overwritten by post-~V section handling
        (known limitation).
        """
        content = """~A PRE_V_NAME
100.0,50.0
101.0,51.0
~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A POST_V_NAME
200.0,60.0
201.0,61.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 2

        # Pre-~V section: should NOT be named 'POST_V_NAME'
        sec_pre = las.data_sections[0]
        assert sec_pre.name != "POST_V_NAME", (
            f"Pre-~V DataSection must not inherit post-~V section name: "
            f"got '{sec_pre.name}'"
        )

        # Post-~V section: should keep its name
        sec_post = las.data_sections[1]
        assert sec_post.name == "POST_V_NAME", (
            f"Post-~V DataSection name should be 'POST_V_NAME', got '{sec_post.name}'"
        )
        # Names must be different
        assert sec_pre.name != sec_post.name, (
            f"Section names must differ: '{sec_pre.name}' vs '{sec_post.name}'"
        )


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

    # --- R7F-05: Deferred CWLS bare-colon (parser.py:1339-1362) ---

    def test_las12_deferred_well_bare_colon_split(self) -> None:
        """R7F-05: When ~W appears before ~V in a LAS 1.2 CWLS file,
        bare-colon well entries (e.g., 'LOG DATE:15/01/2001') must be
        correctly split into description and value.

        Before the fix, bare-colon detection ran in _parse_well before
        the version was known, so is_las12 defaulted to False and no split
        occurred.  Deferred entries were replayed without splitting,
        storing the raw colon-bearing string as the value.

        The fix moves bare-colon detection into _store_well_entry which
        always receives the correct is_las12 flag.
        """
        content = (
            # ~W BEFORE ~V — triggers deferred well entry path
            "~WELL INFORMATION\n"
            " DATE.    LOG DATE:15/01/2001\n"
            " COMP.    COMPANY : Test Oil Company Ltd.\n"
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        )
        parser = LASParser()
        las = parser.parse(content)

        # R7F-05 fix: bare-colon entry "LOG DATE:15/01/2001" should be
        # split into description="LOG DATE" and value="15/01/2001".
        assert las.well["DATE"] == "15/01/2001", (
            f"Expected split value '15/01/2001', got {las.well['DATE']!r}"
        )
        # Non-bare-colon entries should still work
        assert las.well["COMP"] == "Test Oil Company Ltd."

    def test_las12_normal_order_bare_colon_split(self) -> None:
        """R7F-05: When ~V appears before ~W (normal order), bare-colon
        entries in LAS 1.2 are also correctly split — no regression."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " DATE.    LOG DATE:15/01/2001\n"
            " COMP.    COMPANY : Test Oil Company Ltd.\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.well["DATE"] == "15/01/2001"
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

    def test_las30_non_format_braces_rejected(self) -> None:
        """F-REV-01 + F-003 + G-003: Non-format brace text (e.g. {Density})
        is preserved in the description and does NOT cause a parse error.

        Previously _validate_curve_data_format rejected {Density} with
        LASParseError.  Now non-format braces pass through as metadata
        (matching the parameter handler behavior) and are kept in the
        description via targeted format-stripping.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   :\n"
            " DLM.   COMMA :\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " DENS.GCC3    : BULK DENSITY  {Density}\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # {F} was a valid format specifier — stripped from description
        assert las.curves[0].data_format == "F"
        # {Density} was non-format — preserved in description
        assert "{Density}" in las.curves[1].description
        assert las.curves[1].data_format == ""

    def test_las30_uppercase_non_format_braces_rejected(self) -> None:
        """F-REV-01 + F-003 + G-003: Uppercase non-format brace text is
        preserved in the description and does NOT cause a parse error.

        Single-word braces starting with F/E/D/S/A/I that are NOT valid
        format specifiers (e.g. {ENERGY}) are kept as metadata, matching
        the parameter handler behavior.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   :\n"
            " DLM.   COMMA :\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " ENRG.MEV     : ENERGY LEVEL  {ENERGY}\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # {F} was a valid format specifier
        assert las.curves[0].data_format == "F"
        # {ENERGY} was non-format — preserved in description
        assert "{ENERGY}" in las.curves[1].description
        assert las.curves[1].data_format == ""


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
        assert len(errors) == 0, (
            f"LASParser is not thread-safe for shared-instance use. "
            f"Errors from concurrent threads: {errors}"
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
            # I2-XPD-03: Summary warning with total row count —
            # the old boolean-once "Extra columns are discarded" message
            # is now a per-section summary with the affected row count.
            assert "Extra columns were silently discarded" in caplog.text

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


class TestVERSValidation:
    """F-005/IF-003: VERS validation — reject non-standard values.

    The parser validates VERS against known LAS versions (1.2, 2.0, 3.0).
    Non-standard values must warn and fall back to safe defaults.
    """

    def test_vers_comma_decimal_defaults_to_2_0(self) -> None:
        """F-005: VERS with comma decimal separator ('1,2') emits warning
        and defaults to '2.0'.

        Comma-separated version strings (e.g., '1,2' from European locale)
        silently fail all startswith() checks, causing the parser to use
        LAS 1.2 well-field conventions for LAS 2.0 files.  The fix warns
        and defaults to 2.0.
        """
        import warnings

        content = """~VERSION INFORMATION
 VERS . 1,2 : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las = parser.parse(content)
            vers_warnings = [
                x for x in w if "Unknown VERS value" in str(x.message)
            ]
            assert len(vers_warnings) >= 1, (
                f"Expected VERS warning for coma decimal, got: "
                f"{[str(x.message) for x in w]}"
            )
            assert "1,2" in str(vers_warnings[0].message)
            # Must default to "2.0" — not "1,2" (which would be unusable)
            assert las.version.vers == "2.0"

    def test_vers_non_standard_preserved_with_warning(self) -> None:
        """F-005: VERS with non-standard but version-like value ('1.20')
        emits warning and preserves the value.

        LAS 1.20 is a real format; the value is preserved so the reader
        can emit its own 'not officially supported' warning.
        """
        import warnings

        content = """~VERSION INFORMATION
 VERS . 1.20 : CWLS LOG ASCII STANDARD -VERSION 1.20
 WRAP.   NO   : ONE LINE PER DEPTH STEP
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las = parser.parse(content)
            vers_warnings = [
                x for x in w if "Non-standard VERS value" in str(x.message)
            ]
            assert len(vers_warnings) >= 1, (
                f"Expected warning for non-standard VERS 1.20, got: "
                f"{[str(x.message) for x in w]}"
            )
            assert "1.20" in str(vers_warnings[0].message)
            # Value must be preserved as-is for reader's own validation
            assert las.version.vers == "1.20"

    def test_vers_standard_no_warning(self) -> None:
        """F-005: Standard VERS value '2.0' produces no warning."""
        import warnings

        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
"""
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las = parser.parse(content)
            vers_warnings = [
                x for x in w
                if "VERS" in str(x.message) and (
                    "Non-standard" in str(x.message)
                    or "Unknown VERS" in str(x.message)
                )
            ]
            assert len(vers_warnings) == 0, (
                f"Unexpected VERS warning: {[str(x.message) for x in vers_warnings]}"
            )
            assert las.version.vers == "2.0"


class TestEmptyUnitWellEntry:
    """F-008: Empty-unit well entry parsing.

    When a well entry line has an empty unit (e.g., ``MNEM . : VALUE``),
    the unit should be stored as an empty string — not dropped entirely.
    """

    def test_empty_unit_well_entry_preserved(self) -> None:
        """F-008: Parse a well entry with empty unit -> unit stored as ''.

        The regex captures an empty string for the unit group (text between
        dot and whitespace).  The fix changed the truthiness
        check to ``is not None`` so that empty string units are stored
        rather than silently dropped.
        """
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~WELL INFORMATION
 MNEM. VALUE : DESCRIPTION
 STRT.M      1670.0 : START DEPTH
 STOP.M      1660.0 : STOP DEPTH
 STEP.M     -0.1250 : STEP
 NULL.    -999.25   : NULL VALUE
"""
        parser = LASParser()
        las = parser.parse(content)
        # The unit dict entry must exist with an empty string value
        assert "MNEM" in las.well.units, (
            f"Empty unit should be stored, got units: {las.well.units}"
        )
        assert las.well.units["MNEM"] == "", (
            f"Expected empty string unit, got: {las.well.units['MNEM']!r}"
        )
        assert las.well["MNEM"] == "VALUE"

    def test_empty_unit_well_entry_roundtrip(self, tmp_path: Path) -> None:
        """F-008: Empty-unit well entry survives write/read roundtrip."""
        content = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO   : ONE LINE PER DEPTH STEP
~WELL INFORMATION
 MNEM. VALUE : DESCRIPTION
 STRT.M      1670.0 : START DEPTH
 STOP.M      1660.0 : STOP DEPTH
 STEP.M     -0.1250 : STEP
 NULL.    -999.25   : NULL VALUE
~CURVE INFORMATION
 DEPT.M  :  Depth
~A  DEPT
100.0
"""
        temp_file = tmp_path / "empty_unit.las"
        temp_file.write_text(content, encoding="utf-8")
        data = read_las_file(temp_file)
        assert data["well"]["MNEM"] == "VALUE"


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

    # ── F-07: LAS 2.0+ well descriptions ──────────────────────────

    def test_well_descriptions_preserved_las20(self) -> None:
        """F-07: LAS 2.0+ well descriptions should be preserved."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0  : START DEPTH\n"
            " STOP.M   1660.0  : STOP DEPTH\n"
            " COMP.    OILCO   : Oil Company Name\n"
            " WELL.    Well#1  : Test Well Name\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.well.descriptions["STRT"] == "START DEPTH"
        assert las.well.descriptions["STOP"] == "STOP DEPTH"
        assert las.well.descriptions["COMP"] == "Oil Company Name"
        assert las.well.descriptions["WELL"] == "Test Well Name"

    def test_well_descriptions_roundtrip_las20(self) -> None:
        """F-07: LAS 2.0+ well descriptions survive roundtrip through writer."""
        from pylasdev import write_las_file

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0  : START DEPTH\n"
            " STOP.M   1660.0  : STOP DEPTH\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " COMP.    OILCO   : Oil Company Name\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.well.descriptions["COMP"] == "Oil Company Name"

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            suffix=".las", delete=False, mode="w"
        ) as tf:
            write_las_file(tf.name, las)
            out_path = Path(tf.name)

        re_read = parser.parse(out_path.read_text())
        assert re_read.well.descriptions["COMP"] == "Oil Company Name"
        out_path.unlink()

    def test_well_no_description_no_error(self) -> None:
        """F-07: LAS 2.0+ well entries without descriptions (no colon) should not crash."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  :\n"
            " WRAP.   NO   :\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0\n"  # no colon
            " STOP.M   1660.0\n"
            " COMP.    TestCo\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.well["STRT"] == "1670.0"
        assert "STRT" not in las.well.descriptions  # no description → not stored

    # ── F2-03: WRAP=YES warning in LAS 3.0 ──────────────────────

    def test_las30_wrap_yes_warns(self) -> None:
        """F-003: LAS 3.0 WRAP=YES should raise LASParseError.

        WRAP=YES data processing is not implemented — previously a
        logger.warning allowed corrupt parsing to continue; now a
        LASParseError prevents silent data corruption.
        """
        from pylasdev.exceptions import LASParseError

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth\n"
            "~ASCII\n"
            " 1670.0\n"  # at least one data line so _process_ascii_data runs
        )
        parser = LASParser()
        with pytest.raises(LASParseError, match="WRAP=YES"):
            parser.parse(content)

    def test_las30_wrap_no_silent(self, caplog) -> None:
        """F2-03: LAS 3.0 WRAP=NO should NOT produce a WRAP=YES warning."""
        import logging

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth\n"
            "~ASCII\n"
            " 1670.0\n"  # at least one data line so _process_ascii_data runs
        )
        parser = LASParser()
        with caplog.at_level(logging.WARNING):
            parser.parse(content)
        assert not any(
            "WRAP=YES" in record.message for record in caplog.records
        ), f"Unexpected WRAP=YES warning: {[r.message for r in caplog.records]}"

    # ── F-05: Indexed _DATA sections ────────────────────────────

    def test_indexed_data_section_with_data_suffix(self) -> None:
        """F-05: ~Core_Data[1] should be recognized as an indexed data section."""
        assert _is_indexed_data_section("CORE_DATA[1]") is True
        assert _is_indexed_data_section("DRILLING_DATA[2]") is True
        assert _is_indexed_data_section("INCLINOMETRY_DATA[1]") is True
        assert _is_indexed_data_section("TOPS_DATA[3]") is True
        assert _is_indexed_data_section("TEST_DATA[1]") is True
        assert _is_indexed_data_section("PERFORATIONS_DATA[2]") is True
        assert _is_indexed_data_section("LOG_DATA[1]") is True


    # --- F-8-003: WRAP=YES trailing-comma regression ---

    def test_las30_wrap_yes_multi_curve_trailing_comma_raises(self) -> None:
        """F-8-003: WRAP=YES single-value line with trailing comma correctly detected.

        When WRAP=YES with 2+ curves and COMMA delimiter, a data line containing
        a single value plus a trailing comma (e.g. "1670.0,") would produce
        ["1670.0", ""] → len=2 tokens.  Without the F-7-003 fix stripping
        trailing empty tokens, len(_tokens) == 1 is False, incorrectly classifying
        the wrapped data as non-wrapped.  After the fix, trailing empties are
        stripped and the line is correctly identified as wrapped.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITING CHARACTER\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " GR.API       :  Gamma Ray {F}\n"
            "~ASCII\n"
            " 1670.0,\n"  # single value + trailing comma → wrapped data
        )
        parser = LASParser()
        with pytest.raises(LASParseError, match="WRAP=YES"):
            parser.parse(content)

    def test_las30_wrap_yes_multi_curve_full_row_no_raise(self) -> None:
        """F-8-003: WRAP=YES with full multi-curve row does NOT raise.

        When WRAP=YES in header but data rows have >= curve_count values
        (non-wrapped data), the heuristic correctly detects the inconsistency
        and allows parsing to proceed.  This test verifies the heuristic's
        negative path — the trailing-empty-strip fix does NOT cause false
        positives on clean data.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITING CHARACTER\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " GR.API       :  Gamma Ray {F}\n"
            "~ASCII\n"
            " 1670.0,45.5\n"
            " 1670.5,46.0\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # Should parse without raising LASParseError(WRAP=YES)
        assert "DEPT" in las.logs
        assert "GR" in las.logs

    # --- F-7-004: deferred-population uncovered-curve guard ---

    def test_las30_no_uncovered_curve_warning_on_normal_parse(
        self, caplog
    ) -> None:
        """F-7-004: Normal LAS 3.0 parse does NOT emit uncovered-curve warnings.

        DataSection.__post_init__ (models.py:989-993) has a guard that skips
        the F-M-036 uncovered-curve warning when BOTH data and string_data are
        empty — the deferred-population case.  The parser constructs DataSection
        before populating data, so the warning must not fire during normal parse.
        Without this guard, every LAS 3.0 parse would emit spurious warnings
        about uncovered curves for every curve after the first data section.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITING CHARACTER\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " GR.API       :  Gamma Ray {F}\n"
            "~ASCII\n"
            " 1670.0,45.5\n"
        )
        parser = LASParser()
        with caplog.at_level(logging.WARNING):
            parser.parse(content)
        uncovered_warnings = [
            r.message
            for r in caplog.records
            if "uncovered" in r.message.lower()
        ]
        assert not uncovered_warnings, (
            f"Unexpected uncovered-curve warnings: {uncovered_warnings}"
        )


class TestMaxGuardLimits:
    """F-I2-M42: MAX_* guard tests for parser.py.

    All 7 parser guards raise LASParseError when exceeded.  Each test
    monkeypatches one constant to a tiny value and verifies the guard fires.
    """

    # ── MAX_CURVES (parser.py:1513) ──────────────────────────────

    def test_max_curves_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_CURVES guard raises LASParseError when curve count exceeds limit."""
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 1)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Curve count.*exceeds"):
            parser.parse(content)

    # ── MAX_PARAMETERS (parser.py:1625) ──────────────────────────

    def test_max_parameters_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_PARAMETERS guard raises LASParseError when parameter count exceeds limit."""
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 1)
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~PARAMETER INFORMATION
 BHT.DEGC    35.5 : BOTTOM HOLE TEMPERATURE
 BS .MM      200  : BIT SIZE
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Parameter count.*exceeds"):
            parser.parse(content)

    # ── MAX_OTHER_LINES (parser.py:985) ──────────────────────────

    def test_max_other_lines_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_OTHER_LINES guard raises LASParseError when other-line count exceeds limit."""
        monkeypatch.setattr("pylasdev.parser.MAX_OTHER_LINES", 1)
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~X CUSTOM
 Some body text here.
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Other section line count.*exceeds"):
            parser.parse(content)

    # ── MAX_SECTION_SEQUENCE (parser.py:950) ─────────────────────

    def test_max_section_sequence_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_SECTION_SEQUENCE guard raises LASParseError when section count exceeds limit."""
        monkeypatch.setattr("pylasdev.parser.MAX_SECTION_SEQUENCE", 1)
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~WELL INFORMATION
 STRT.M   1670.0 :
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Section sequence length.*exceeds"):
            parser.parse(content)

    # ── MAX_DEFERRED_WELL_ENTRIES (parser.py:1375) ───────────────

    def test_max_deferred_well_entries_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_DEFERRED_WELL_ENTRIES guard raises LASParseError for ~W before ~V."""
        monkeypatch.setattr("pylasdev.parser.MAX_DEFERRED_WELL_ENTRIES", 1)
        # ~W before ~V: both entries are deferred
        content = """~WELL INFORMATION
 STRT.M   1670.0 : START DEPTH
 STOP.M   1660.0 : STOP DEPTH
~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"Deferred well entry count.*exceeds"):
            parser.parse(content)

    # ── MAX_DATA_LINES (parser.py:1927) ──────────────────────────

    def test_max_data_lines_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_DATA_LINES guard raises LASParseError when data lines exceed limit."""
        monkeypatch.setattr("pylasdev.data_reader.MAX_DATA_LINES", 1)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A
100.0
101.0
"""
        parser = LASParser()
        with pytest.raises(LASParseError, match=r"ASCII data line count.*exceeds"):
            parser.parse(content)

    def test_max_data_lines_at_limit_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_DATA_LINES set exactly to the file's data line count — must accept.

        F-R-01: parser.py MAX_DATA_LINES guards now use ``>`` for consistency
        with data_reader.py and models.py (accepts at exactly MAX_DATA_LINES).
        Tests both the accumulation guard (line 2157) and the _process_ascii_data
        guard (line 2414) via a single parse call.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_DATA_LINES", 1)
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
~A
100.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.logs) == 1
        assert len(las.logs["DEPT"]) == 1

    # ── MAX_DATA_SECTIONS (parser.py:1920) ───────────────────────

    def test_max_data_sections_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_DATA_SECTIONS guard raises LASParseError when section count exceeds limit."""
        monkeypatch.setattr("pylasdev.parser.MAX_DATA_SECTIONS", 1)
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
        with pytest.raises(LASParseError, match=r"Data section count.*exceeds"):
            parser.parse(content)


class TestPipeDelimitedSectionHeaders:
    """F-I2-M43: Pipe-delimited section header parsing in LAS 3.0.

    Tests pipe-delimiter syntax on data section headers as parser INPUT
    (not just writer output).  Exercises the pipe-extraction branches
    at parser.py:551-563 and the pipe-resolution branches at lines
    726-779.
    """

    def test_pipe_to_definition_curves(self) -> None:
        """~Core[1] | Core_Definition routes data to the definition's curves."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CORE_DEFINITION
 RHOZ.OHMM       :  RESISTIVITY  {F}
 PHIZ.V/V        :  POROSITY  {F}
~Core[1] | Core_Definition
2.5,0.15
3.1,0.18
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        sec = las.data_sections[0]
        assert sec.section_type == "CORE_DATA"
        # Data must be scoped to Core_Definition curves (RHOZ, PHIZ)
        assert sec.data["RHOZ"][0] == 2.5
        assert sec.data["PHIZ"][0] == 0.15
        assert sec.data["RHOZ"][1] == 3.1
        assert sec.data["PHIZ"][1] == 0.18

    def test_pipe_to_curve_main_block(self) -> None:
        """~ASCII | CURVE routes data to the main (non-definition) curve block."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~ASCII | CURVE
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        sec = las.data_sections[0]
        assert sec.data["DEPT"][0] == 100.0
        assert sec.data["DT"][0] == 50.0

    def test_pipe_to_short_c_alias(self) -> None:
        """~A | C is accepted as equivalent to | CURVE (both uppercase match)."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A | C
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        las = parser.parse(content)
        assert len(las.data_sections) == 1
        sec = las.data_sections[0]
        assert sec.data["DEPT"][0] == 100.0

    def test_pipe_target_resets_on_unrecognized_target(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unrecognized pipe target logs warning and resets curve indices."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 DT.US/M      : SONIC  {F}
~A | UNKNOWN_TARGET
100.0,50.0
101.0,51.0
"""
        parser = LASParser()
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            las = parser.parse(content)
            assert "Unrecognized pipe target" in caplog.text
        # Data should still parse (uses default curve scope)
        assert len(las.data_sections) == 1
        assert las.data_sections[0].data["DEPT"][0] == 100.0


class TestIndexedSectionOrdering:
    """R7F-04: Indexed data sections (e.g., CORE[1]) resolve to their
    definition type in section ordering checks.

    Before the fix, CORE[1] fell through the _SECTION_TYPE_MAP resolution
    chain to __MAIN__ because only unindexed keys exist in the map.  The
    fix strips the bracket suffix to extract the base type before lookup.
    """

    def test_indexed_section_before_definition_emits_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CORE[1] data section before ~CORE_DEFINITION should emit a
        per-type ordering warning."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~Core[1]
 2.5,0.15
 3.1,0.18
~CORE_DEFINITION
 RHOZ.OHMM       :  RESISTIVITY  {F}
 PHIZ.V/V        :  POROSITY  {F}
~Core[2]
 4.0,0.20
"""
        parser = LASParser()
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            las = parser.parse(content)
            warning_text = caplog.text
            # Should warn about CORE[1] before ~CORE_DEFINITION
            assert "~CORE[1] before ~CORE_DEFINITION" in warning_text, (
                f"Expected ordering warning for CORE[1]; got: {warning_text}"
            )

        # CORE[1] data before definition is discarded (no curves available);
        # CORE[2] data after definition is properly stored.
        assert len(las.data_sections) == 1
        assert las.data_sections[0].name == "CORE[2]"
        assert las.data_sections[0].data["RHOZ"][0] == 4.0
        assert las.data_sections[0].data["PHIZ"][0] == 0.2

    def test_indexed_section_after_definition_no_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CORE[1] after ~CORE_DEFINITION (correct order) — no warning."""
        content = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO   :
 DLM.   COMMA :
~WELL INFORMATION
 NULL.    -999.25 : NULL VALUE
~CORE_DEFINITION
 RHOZ.OHMM       :  RESISTIVITY  {F}
 PHIZ.V/V        :  POROSITY  {F}
~Core[1]
 2.5,0.15
 3.1,0.18
"""
        parser = LASParser()
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            las = parser.parse(content)
            warning_text = caplog.text
            # No per-type ordering warning for CORE[1] (correct order)
            assert "before ~CORE_DEFINITION" not in warning_text, (
                f"Unexpected ordering warning: {warning_text}"
            )

        assert len(las.data_sections) == 1
        # Data is correctly stored when section follows definition
        assert las.data_sections[0].name == "CORE[1]"
        assert las.data_sections[0].data["RHOZ"][0] == 2.5
        assert las.data_sections[0].data["PHIZ"][0] == 0.15
        assert las.data_sections[0].data["RHOZ"][1] == 3.1
        assert las.data_sections[0].data["PHIZ"][1] == 0.18


class TestNonLetterTildeHeaders:
    """F-I2-M44: Non-letter tilde header handling.

    Lines starting with ~ followed by a non-letter character (~., ~#,
    ~/, bare ~) do NOT match SECTION_PATTERN but DO match the
    startswith("~") guard at parser.py:966.  They are routed to
    _other_lines rather than producing corrupt data rows or crashing.
    """

    def test_tilde_period_header_routed_to_other(self) -> None:
        """~. header is routed to _other_lines via startswith("~") guard.

        Non-letter tilde headers do NOT reset _current_section, so
        subsequent body lines are still processed by the active section
        handler.  This test verifies the ~. line itself reaches
        _other_lines and the parser does not crash.
        """
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~.
"""
        parser = LASParser()
        las = parser.parse(content)
        assert "~." in las.other

    def test_tilde_hash_header_routed_to_other(self) -> None:
        """~# header is routed to _other_lines."""
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~#
"""
        parser = LASParser()
        las = parser.parse(content)
        assert "~#" in las.other

    def test_tilde_slash_header_routed_to_other(self) -> None:
        """~/ header is routed to _other_lines."""
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~/
"""
        parser = LASParser()
        las = parser.parse(content)
        assert "~/" in las.other

    def test_bare_tilde_header_routed_to_other(self) -> None:
        """Bare ~ (tilde alone on a line) is routed to _other_lines."""
        content = """~VERSION INFORMATION
 VERS.   2.0  :
 WRAP.   NO   :
~
"""
        parser = LASParser()
        las = parser.parse(content)
        assert "~" in las.other


# ============================================================
# Production Check Regression Tests
# ============================================================

class TestProductionCheckParserFix:
    """Regression test for F-212 fix in parser.py."""

    def test_parser_desanitize_disabled_preserves_hash_prefix(self) -> None:
        """F-212 (parser.py side): With _DESANITIZE_ENABLED=False, _# preserved.

        The parser.py copy of _desanitize_las_value received the same
        _DESANITIZE_ENABLED guard as data_reader.py.
        """
        import pylasdev.parser as p_mod

        try:
            p_mod._DESANITIZE_ENABLED = False
            result = p_mod._desanitize_las_value("_#external_data")
            assert result == "_#external_data", (
                f"Expected '_#external_data' preserved, got {result!r}"
            )
            assert p_mod._desanitize_las_value("clean") == "clean"
        finally:
            p_mod._DESANITIZE_ENABLED = True

    def test_parser_desanitize_enabled_default_behavior(self) -> None:
        """F-212 (parser.py side): Default retains roundtrip-correct behavior."""
        import pylasdev.parser as p_mod

        assert p_mod._DESANITIZE_ENABLED is True
        result = p_mod._desanitize_las_value("_#hash_val")
        assert result == "#hash_val"
