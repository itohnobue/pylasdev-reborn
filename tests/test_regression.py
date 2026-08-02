"""Regression tests for verified findings from Stage 6 and Stage 9 fixes.

These tests exercise specific fixes identified by the adversarial verification
pipeline. Each test documents which finding it covers and tests the actual
fixed behaviour against the current (fixed) source.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from pylasdev import (
    read_dev_file,
    read_las_file,
    read_las_file_as_object,
    write_las_file,
)
from pylasdev.compare import _compare_lists, compare_las_dicts
from pylasdev.encoding import read_with_encoding
from pylasdev.exceptions import LASParseError
from pylasdev.mnem_base import MNEM_BASE
from pylasdev.models import (
    CurveDefinition,
    DataSection,
    DevFile,
    LASFile,
    ParameterEntry,
    VersionSection,
    _GuardedList,
)
from pylasdev.parser import LASParser

# ──────────────────────────────────────────────────────────────
# F-001 (parser, HIGH): Unicode whitespace / non-standard line
# endings handled correctly by the parser.
# ──────────────────────────────────────────────────────────────

class TestF001ParserLineEndings:
    """F-001: Parser handles non-standard line endings and Unicode whitespace
    that the writer strips (the regex in _SPLITLINES_CHARS_RE was updated
    to include 13 additional Unicode whitespace characters for symmetry)."""

    def test_parse_crlf_line_endings(self) -> None:
        """F-001: Parser handles \\r\\n (CRLF) line endings."""
        content = (
            "~VERSION INFORMATION\r\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\r\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\r\n"
            "~WELL INFORMATION\r\n"
            " STRT.M   1670.0 : START DEPTH\r\n"
            " STOP.M   1660.0 : STOP DEPTH\r\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == "2.0"
        assert las.well["STRT"] == "1670.0"
        assert las.well["STOP"] == "1660.0"

    def test_parse_cr_only_line_endings(self) -> None:
        """F-001: Parser handles \\r-only (classic Mac) line endings.

        \\r is covered by _SPLITLINES_CHARS_RE control-character range
        (\\x0D is within \\x00-\\x1F)."""
        content = (
            "~VERSION INFORMATION\r"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\r"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\r"
        )
        parser = LASParser()
        las = parser.parse(content)
        assert las.version.vers == "2.0"

    def test_parse_non_breaking_space_well_value(self) -> None:
        """F-001: Well value with NO-BREAK SPACE (\\u00A0) is preserved
        through the parser (the writer strips it; the parser now sympathetically
        handles it via its _SPLITLINES_CHARS_RE)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " COMP.    ACME\u00A0Corp : COMPANY\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # The NO-BREAK SPACE is inside the dot-delimited section value.
        # After _SPLITLINES_CHARS_RE substitution (\\u00A0 is now in the
        # regex), the raw access depends on whether the regex substitutes
        # with a space. Verify the company name is extracted.
        assert "COMP" in las.well
        # Verify \u00A0 was replaced by regular space (the F-001 fix behavior)
        assert "\u00A0" not in las.well["COMP"], (
            "NO-BREAK SPACE should be replaced by regular space"
        )
        assert " " in las.well["COMP"], (
            "Value should contain regular space after NBSP replacement"
        )


# ──────────────────────────────────────────────────────────────
# F-007 (parser, HIGH): #-prefixed value desanitize roundtrip
# via _desanitize_las_value in the parser.
# ──────────────────────────────────────────────────────────────

class TestF007Desanitize:
    """F-007: Parser reverses the writer's ``_#``-prefix escape via
    ``_desanitize_las_value`` so that values starting with ``#`` survive
    a write→read roundtrip (previously they were permanently escaped as
    ``_#...``)."""

    def test_desanitize_hash_prefix_well_value(self) -> None:
        """F-007: #-prefixed well value roundtrips correctly via parser."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " COMP.    _#TestCompany : COMPANY\n"  # writer-escaped form
        )
        parser = LASParser()
        las = parser.parse(content)
        # The parser's _desanitize_las_value strips the leading _
        assert las.well["COMP"] == "#TestCompany"

    def test_desanitize_hash_prefix_with_whitespace(self) -> None:
        """F-007: whitespace-padded _# value is desanitized correctly.

        The writer inserts _ between leading whitespace and #.  The parser
        strips the leading whitespace during dot-split → value is "_#Company"
        → _desanitize_las_value strips the _ via case-1 (starts with _#)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " COMP.     _#Company : COMPANY\n"  # space-padded writer output
        )
        parser = LASParser()
        las = parser.parse(content)
        # After dot-split and whitespace stripping, the value is "_#Company";
        # _desanitize strips the leading underscore → "#Company".
        assert las.well["COMP"] == "#Company"

    def test_desanitize_parameter_value(self) -> None:
        """F-007: #-prefixed parameter value is desanitized."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~PARAMETER INFORMATION\n"
            " FLAG.    _#ERROR : Status Flag\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        par = next((p for p in las.parameters if p.mnemonic == "FLAG"), None)
        assert par is not None
        assert par.value == "#ERROR"


# ──────────────────────────────────────────────────────────────
# F-063 (writer, MEDIUM): Uncovered curve warning after
# data_sections copy-back.
# ──────────────────────────────────────────────────────────────

class TestF063UncoveredCurveWarning:
    """F-063: After data_sections→legacy copy-back, curves_order entries
    without corresponding data in logs or string_data produce a UserWarning."""

    def test_uncovered_curve_emits_warning(self, tmp_path: Path) -> None:
        """F-063: Writing a LASFile with an uncovered curve in curves_order
        emits a UserWarning about padding with null_value.

        The copy-back path (non-LAS-3.0 fallback) can produce curves_order
        entries without corresponding data in logs or string_data.  This
        test triggers that path by using LAS 1.2 with a data_section where
        curves_order names a curve not present in the data dict.

        Note: The writer's copy-back check uses phrasing "no data in 'logs'"
        (distinct from DataSection's own "no data in 'data'" warning)."""
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO")  # NOT LAS 3.0
        las.well["NULL"] = "-999.25"
        # DO NOT pre-populate curves_order, curves, logs — let copy-back
        # populate them from the DataSection to trigger the uncovered
        # curve scenario.
        # las.curves_order and las.curves are intentionally left empty.

        # DataSection has a curve in curves_order that is NOT present in
        # data or string_data.  The copy-back will populate las_file's
        # attributes, and the F-063 check will detect the gap.
        section = DataSection(
            name="LOG",
            curves_order=["DEPT", "UNCOVERED"],
            data={"DEPT": np.array([100.0, 101.0])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="UNCOVERED", unit=""),
            ],
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "uncovered_curve.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_las_file(temp_file, las)
            writer_warnings = [
                x for x in w
                if "no data in 'logs'" in str(x.message)
            ]
            assert len(writer_warnings) >= 1, (
                "Expected uncovered curve warning from writer (matching "
                "'no data in logs'), got warnings: "
                f"{[str(x.message) for x in w]}"
            )
            assert "UNCOVERED" in str(writer_warnings[0].message)


# ──────────────────────────────────────────────────────────────
# G-001 (models, HIGH): mnem_base normalization applied to
# data/logs/string_data dict keys in from_dict.
# ──────────────────────────────────────────────────────────────

class TestG001MnemBaseDictKeys:
    """G-001: LASFile.from_dict with mnem_base normalises data/logs/
    string_data dict keys through the mnemonic lookup."""

    def test_mnem_base_normalizes_logs_keys(self) -> None:
        """G-001: from_dict with mnem_base normalises logs dict keys.

        When mnem_base maps 'AK' → 'DT', the logs key 'AK' should
        be stored as 'DT' so the writer can find it when iterating
        the normalised curves_order."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT", "AK"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "AK", "unit": "US/M"},
            ],
            "logs": {
                "DEPT": np.array([100.0, 101.0]),
                "AK": np.array([50.0, 51.0]),
            },
        }
        las = LASFile.from_dict(data, mnem_base={"AK": "DT"})
        assert "DT" in las.logs, (
            f"Expected 'DT' key in logs via mnem_base, got: {list(las.logs.keys())}"
        )

    def test_mnem_base_normalizes_string_data_keys(self) -> None:
        """G-001: from_dict with mnem_base normalises string_data dict keys
        within a data_section (the LAS 3.0 pattern).

        Uses a data_section because the top-level path without data_sections
        now has conflicting validations after F2-11 (string_data keys must
        be in curves_order) and F-011 (curves_order keys must be in logs).
        Per-section string_data avoids these conflicts."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
            ],
            "logs": {
                "DEPT": np.array([100.0]),
            },
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "AK"],
                    "data": {"DEPT": np.array([100.0])},
                    "string_data": {"AK": np.array(["SAND"], dtype=np.str_)},
                }
            ],
        }
        las = LASFile.from_dict(data, mnem_base={"AK": "DT"})
        assert len(las.data_sections) == 1
        assert "DT" in las.data_sections[0].string_data, (
            f"Expected 'DT' key in string_data via mnem_base, "
            f"got: {list(las.data_sections[0].string_data.keys())}"
        )


# ──────────────────────────────────────────────────────────────
# G-018 (writer, MEDIUM): version.wrap restored after write.
# ──────────────────────────────────────────────────────────────

class TestG018WrapPreservation:
    """G-018: Writer reflects actual disk state in version.wrap after write.
    The writer always produces WRAP=NO output (non-wrapped), and the model
    now honestly reflects what was written rather than restoring the pre-write
    value (F2-26 fix removed the ``finally:`` restore block)."""

    def test_wrap_yes_reflects_disk_after_write(self, tmp_path: Path) -> None:
        """G-018: Write with WRAP=YES — wrap reflects actual disk state.

        The writer overrides WRAP=YES to WRAP=NO during content generation
        (it cannot produce wrapped output).  After writing, the LASFile
        model's wrap attribute honestly reflects what was written to disk."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="YES")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        temp_file = tmp_path / "wrap_yes.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        # After writing, the model's wrap value reflects what was written
        assert las.version.wrap == "NO", (
            f"Expected wrap='NO' after write (disk state), got wrap={las.version.wrap!r}"
        )

    def test_wrap_no_preserved_after_write(self, tmp_path: Path) -> None:
        """G-018: Write with WRAP=NO stays WRAP=NO after write (no-op case)."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "wrap_no.las"
        write_las_file(temp_file, las)
        assert las.version.wrap == "NO"

    def test_wrap_non_default_reflects_disk_after_write(self, tmp_path: Path) -> None:
        """G-018: Write with non-default wrap=YES on LASFile.

        The writer always produces non-wrapped output. The model honestly
        reflects the disk state after writing rather than restoring the
        pre-write value (F2-26 fix)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="YES", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        temp_file = tmp_path / "wrap_yes_las30.las"
        write_las_file(temp_file, las)
        assert las.version.wrap == "NO", (
            f"Expected wrap='NO' after write (disk state), "
            f"got wrap={las.version.wrap!r}"
        )


# ──────────────────────────────────────────────────────────────
# R8-002 (compare, HIGH): Integer-dtype MaskedArray comparison
# does not crash.
# ──────────────────────────────────────────────────────────────

class TestR8_002IntegerMaskedArray:
    """R8-002: compare_las_dicts handles integer-dtype MaskedArrays
    without crashing on `.filled(np.nan)`.

    The fix adds `.astype(np.float64)` before `.filled(np.nan)`, which
    works for int8/16/32/64 and uint8/16/32/64 dtypes."""

    def test_int32_masked_array_comparison_match(self) -> None:
        """R8-002: Two identical int32 MaskedArrays are compared correctly."""
        a1 = np.ma.array([1, 2, 3], dtype=np.int32, mask=[0, 1, 0])
        a2 = np.ma.array([1, 2, 3], dtype=np.int32, mask=[0, 1, 0])
        d1 = {"logs": {"VAL": a1}}
        d2 = {"logs": {"VAL": a2}}
        # Should not crash; should return True (same data, same mask)
        result = compare_las_dicts(d1, d2)
        assert result is True

    def test_int64_masked_array_comparison(self) -> None:
        """R8-002: int64 MaskedArray comparison does not crash."""
        a1 = np.ma.array([10, 20, 30], dtype=np.int64, mask=[0, 0, 0])
        a2 = np.ma.array([10, 20, 30], dtype=np.int64, mask=[0, 0, 0])
        d1 = {"logs": {"V": a1}}
        d2 = {"logs": {"V": a2}}
        assert compare_las_dicts(d1, d2) is True

    def test_int32_masked_array_mismatch_detected(self) -> None:
        """R8-002: int32 MaskedArrays with different unmasked values
        return False and do not crash."""
        a1 = np.ma.array([1, 2, 3], dtype=np.int32, mask=[0, 1, 0])
        a2 = np.ma.array([1, 999, 3], dtype=np.int32, mask=[0, 0, 0])
        d1 = {"logs": {"VAL": a1}}
        d2 = {"logs": {"VAL": a2}}
        assert compare_las_dicts(d1, d2) is False

    def test_uint8_masked_array_comparison(self) -> None:
        """R8-002: uint8 MaskedArray comparison does not crash."""
        a1 = np.ma.array([1, 2, 3], dtype=np.uint8, mask=[0, 0, 0])
        a2 = np.ma.array([1, 2, 3], dtype=np.uint8, mask=[0, 0, 0])
        d1 = {"logs": {"V": a1}}
        d2 = {"logs": {"V": a2}}
        assert compare_las_dicts(d1, d2) is True


# ──────────────────────────────────────────────────────────────
# R8-005 (models, MEDIUM): Non-str section_type in ParameterEntry
# does not crash __post_init__.
# ──────────────────────────────────────────────────────────────

class TestR8_005NonStrSectionType:
    """R8-005: ParameterEntry with non-str section_type (e.g. integer)
    does not crash __post_init__.  The fix converts non-str values
    via _safe_str before calling .strip()."""

    def test_section_type_int_converted(self) -> None:
        """R8-005: section_type=42 is converted to '42' via _safe_str."""
        p = ParameterEntry(mnemonic="TEST", section_type=42)  # type: ignore[arg-type]
        assert p.section_type == "42"

    def test_section_type_int_descriptive_name(self) -> None:
        """R8-005: section_type=0 → '0' (via _safe_str).

        _safe_str converts int via str() then returns the original
        str result. 0 → '0' and survives .strip(). No crash."""
        p = ParameterEntry(mnemonic="TEST", section_type=0)  # type: ignore[arg-type]
        # 0 is falsy but `if self.section_type is not None` passes;
        # isinstance guard catches non-str, _safe_str(0) → "0".
        # Then .strip() on "0" → "0".
        assert p.section_type is not None
        assert isinstance(p.section_type, str)

    def test_section_type_str_unchanged(self) -> None:
        """R8-005: str section_type is unchanged (no regression)."""
        p = ParameterEntry(mnemonic="TEST", section_type="LOG_DATA")
        assert p.section_type == "LOG_DATA"

    def test_section_type_none_unchanged(self) -> None:
        """R8-005: section_type=None stays None (no crash)."""
        p = ParameterEntry(mnemonic="TEST", section_type=None)
        assert p.section_type is None


# ──────────────────────────────────────────────────────────────
# R8-007 (writer, MEDIUM): data_sections copy-back does NOT
# permanently mutate the LASFile model.
# ──────────────────────────────────────────────────────────────

class TestR8_007ModelStatePreserved:
    """R8-007: After writing, the LASFile model's logs, string_data,
    curves_order, and curves are restored to their pre-write state.
    The save/restore pattern in _write_ascii_sections ensures the
    copy-back mutation does not leak to the caller."""

    def test_logs_restored_after_write_with_data_sections(self, tmp_path: Path) -> None:
        """R8-007: logs are not mutated after writing with data_sections.

        When writing a non-LAS 3.0 file with data_sections (triggering
        legacy copy-back), the originally-empty logs should be restored."""
        las = LASFile()
        las.version = VersionSection(vers="2.0")  # NOT LAS 3.0 — triggers copy-back
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        # las.logs is empty — data lives only in data_sections
        section = DataSection(
            name="LOG",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)

        # Snapshot pre-write state
        pre_logs_keys = list(las.logs.keys())
        pre_string_data_keys = list(las.string_data.keys())
        pre_curves_order = list(las.curves_order)
        pre_curves_count = len(las.curves)

        temp_file = tmp_path / "r8_007_mutation.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        # After writing, all model state should be restored
        assert list(las.logs.keys()) == pre_logs_keys, (
            f"logs keys changed: {list(las.logs.keys())} vs {pre_logs_keys}"
        )
        assert list(las.string_data.keys()) == pre_string_data_keys
        assert list(las.curves_order) == pre_curves_order
        assert len(las.curves) == pre_curves_count

    def test_curves_order_restored_after_write(self, tmp_path: Path) -> None:
        """R8-007: curves_order restored when writing triggers copy-back."""
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "DT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="DT", unit="US/M"))
        las.logs["DEPT"] = np.array([100.0])
        las.logs["DT"] = np.array([50.0])

        # Also add a data_section that carries different curves_order
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "DT", "GR"],
            data={
                "DEPT": np.array([100.0]),
                "DT": np.array([50.0]),
                "GR": np.array([75.0]),
            },
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="DT", unit="US/M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
        )
        las.data_sections.append(section)

        original_curves_order = list(las.curves_order)

        temp_file = tmp_path / "r8_007_curves_order.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        assert list(las.curves_order) == original_curves_order, (
            f"curves_order mutated: {list(las.curves_order)} "
            f"vs {original_curves_order}"
        )


# ──────────────────────────────────────────────────────────────
# R8-009 (writer/models, MEDIUM): DataSection with section_type=None
# does NOT crash the writer.
# ──────────────────────────────────────────────────────────────

class TestR8_009SectionTypeNone:
    """R8-009 / G-021: DataSection.section_type=None does not crash
    the writer.  The writer guards ``.upper()`` calls with
    ``(section.section_type or "LOG_DATA").upper()``."""

    def test_data_section_none_section_type_writes(self, tmp_path: Path) -> None:
        """R8-009: Writing a LASFile whose DataSection has
        section_type=None does not crash.

        The writer uses ``(section.section_type or "LOG_DATA")`` as a
        fallback so the .upper() call succeeds on a valid string."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))

        section = DataSection(
            name="LOG",
            section_type=None,  # type: ignore[arg-type]  # This IS the test case
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0, 101.0])},
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "none_section_type.las"
        # Should not crash with AttributeError on .upper()
        write_las_file(temp_file, las)
        assert temp_file.exists()
        content = temp_file.read_text()
        assert "~LOG_DATA" in content or "~A" in content

    def test_data_section_none_section_type_no_crash(self, tmp_path: Path) -> None:
        """R8-009: Multiple DataSections with section_type=None
        triggered the crash at two sites (writer.py:548 and ~938).
        Both are now guarded."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))

        # Two sections both with section_type=None
        for name in ("Sec1", "Sec2"):
            section = DataSection(
                name=name,
                section_type=None,  # type: ignore[arg-type]
                curves_order=["DEPT"],
                data={"DEPT": np.array([100.0 + (idx * 100) for idx in range(2)])},
            )
            las.data_sections.append(section)

        temp_file = tmp_path / "two_none_sections.las"
        write_las_file(temp_file, las)  # Must not crash
        assert temp_file.exists()

    def test_data_section_empty_string_section_type(self, tmp_path: Path) -> None:
        """R8-009: DataSection with section_type='' (empty string)
        does not crash the writer.

        Models now pass empty string through validation (``is not None``
        instead of truthiness). The writer's ``or "LOG_DATA"`` handles
        empty string as a falsy fallback."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))

        section = DataSection(
            name="EMPTY_TYPE",
            section_type="",   # empty string — valid "no particular type"
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0])},
        )
        las.data_sections.append(section)

        temp_file = tmp_path / "empty_section_type.las"
        write_las_file(temp_file, las)  # Must not crash
        assert temp_file.exists()


# ──────────────────────────────────────────────────────────────
# M-10 (writer, MEDIUM): version.dlm preserved after write.
# ──────────────────────────────────────────────────────────────

class TestM10DlmPreservation:
    """M-10: Writer preserves version.dlm after write_las_file returns.

    The writer temporarily mutates DLM to SPACE for LAS 1.2 output
    (since LAS 1.2 only supports SPACE delimiter per spec), then
    restores the original DLM in the finally block (writer.py:893→1228).
    For LAS 2.0 and 3.0, no DLM mutation occurs — the original value
    is preserved unchanged.

    Equivalent WRAP preservation is tested in TestG018WrapPreservation."""

    def test_las12_dlm_comma_restored_after_write(self, tmp_path: Path) -> None:
        """M-10: LAS 1.2 with DLM=COMMA restored after write.

        The writer temporarily mutates DLM to SPACE for LAS 1.2 data
        output (the spec only supports SPACE), then restores the
        original value in the finally block."""
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        temp_file = tmp_path / "las12_dlm_comma.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        assert las.version.dlm == "COMMA", (
            f"Expected version.dlm='COMMA' after write (restored), "
            f"got dlm={las.version.dlm!r}"
        )

    def test_las12_dlm_space_unchanged_after_write(self, tmp_path: Path) -> None:
        """M-10: LAS 1.2 with DLM=SPACE unchanged after write.

        No temporary mutation needed — DLM is already SPACE to begin with."""
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO", dlm="SPACE")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las12_dlm_space.las"
        write_las_file(temp_file, las)
        assert las.version.dlm == "SPACE", (
            f"Expected version.dlm='SPACE' after write (unchanged), "
            f"got dlm={las.version.dlm!r}"
        )

    def test_las20_dlm_comma_preserved_after_write(self, tmp_path: Path) -> None:
        """M-10: LAS 2.0 with DLM=COMMA preserved after write.

        LAS 2.0 supports COMMA; no temporary DLM mutation occurs.
        WRAP may be overridden (YES→NO) but DLM remains COMMA."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las20_dlm_comma.las"
        write_las_file(temp_file, las)
        assert las.version.dlm == "COMMA", (
            f"Expected version.dlm='COMMA' after write, "
            f"got dlm={las.version.dlm!r}"
        )

    def test_las30_dlm_comma_preserved_after_write(self, tmp_path: Path) -> None:
        """M-10: LAS 3.0 with DLM=COMMA preserved after write.

        LAS 3.0 supports COMMA; no temporary DLM mutation occurs."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        temp_file = tmp_path / "las30_dlm_comma.las"
        write_las_file(temp_file, las)
        assert las.version.dlm == "COMMA", (
            f"Expected version.dlm='COMMA' after write, "
            f"got dlm={las.version.dlm!r}"
        )

    def test_dlm_not_permanently_mutated_to_space(self, tmp_path: Path) -> None:
        """M-10: DLM is NOT permanently mutated to SPACE after writing.

        LAS 1.2 with DLM=COMMA writes SPACE-delimited output (per spec)
        but restores the original DLM.  This test explicitly asserts
        that the model is not left with a mutated DLM value."""
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        temp_file = tmp_path / "las12_not_mutated.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(temp_file, las)

        assert las.version.dlm != "SPACE", (
            "Expected version.dlm to NOT be 'SPACE' after write — "
            "it was permanently mutated to SPACE"
        )
        assert las.version.dlm == "COMMA", (
            f"Expected version.dlm='COMMA' (restored), "
            f"got dlm={las.version.dlm!r}"
        )


# ──────────────────────────────────────────────────────────────
# H-01 (parser, HIGH): LAS 3.0 LOG_DATA curve-scoping via
# LOG_DEFINITION lookup.  _Definition-only files (no bare ~C, no
# __MAIN__ sentinel) previously fell to the else-reset (ALL curves)
# and the consumer sliced every curve — data columns mapped
# positionally into wrong curve names.
# ──────────────────────────────────────────────────────────────

class TestH01LogDataCurveScoping:
    """H-01: A bare ~LOG_DATA section in a _Definition-only LAS 3.0 file
    must consult the saved LOG_DEFINITION range (like every other typed
    section resolves its _DEFINITION) instead of resetting to ALL curves.

    Discriminating shape (adversarial F-02): a PRECEDING ~CORE_DEFINITION
    + ~CORE_DATA section makes the LOG_DEFINITION curve range ≠ [0, all),
    so a bare ~LOG_DATA (no pipe, no classic ~C, no __MAIN__) either lands
    on the LOG_DEFINITION curves (fixed) or on a positional ALL-curves
    slice (pre-fix).  A single LOG_DEFINITION + LOG_DATA alone does NOT
    discriminate — the range would equal [0, all) on both trees."""

    def test_log_data_scopes_to_log_definition(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~CORE_DEFINITION\n"
            " DEPTH.M   :  Depth\n"
            " CORE1.G   :  Core\n"
            "~CORE_DATA\n"
            "1000.0  1.0\n"
            "1001.0  2.0\n"
            "~LOG_DEFINITION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG_DATA\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "h01_scoping.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # The bare ~LOG_DATA section must scope to the LOG_DEFINITION curves
        # (DEPT, GR) — NOT the pre-fix ALL-curves slice ['DEPTH','CORE1',
        # 'DEPT','GR'] that mapped the GR data column under CORE1 and left
        # DEPT/GR null-filled.
        assert las.data_sections[1].curves_order == ["DEPT", "GR"], (
            f"H-01: LOG_DATA scoped to wrong curves: "
            f"{las.data_sections[1].curves_order}"
        )
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 51.0])


# ──────────────────────────────────────────────────────────────
# H-04 (parser, HIGH): FORMAT_SPEC_PATTERN quadratic ReDoS.
# The lazy [^}:\s]*? + anchored \s*\} backtracked quadratically on
# long unclosed-brace descriptions.  Bounded quantifiers make the
# pattern linear.  Timing guard with a generous threshold.
# ──────────────────────────────────────────────────────────────

class TestH04FormatSpecLinearTime:
    """H-04: Parsing a file with many long unclosed-brace descriptions
    must complete in linear time.  Pre-fix, ~100 lines of ~4000-byte text
    each containing 2000 unclosed braces took ~30s (quadratic — every
    '{' is a potential start that re-scans the remainder); post-fix it is
    ~0.5s.  The 10s threshold is ~20x the post-fix cost and still catches
    the pre-fix quadratic regression on this input (measured 30.9s
    pre-fix at HEAD)."""

    def test_long_unclosed_brace_descriptions_parse_quickly(self) -> None:
        lines = [
            "~VERSION INFORMATION",
            " VERS.   2.0  : CWLS LOG ASCII STANDARD",
            " WRAP.   NO   : ONE LINE PER DEPTH STEP",
            "~WELL INFORMATION",
            " STRT.M   100.0 : START DEPTH",
            " STOP.M   200.0 : STOP DEPTH",
            " STEP.M   1.0   : STEP",
            " NULL.    -999.25 : NULL VALUE",
            "~PARAMETER INFORMATION",
        ]
        # 100 parameter lines, each with 2000 unclosed braces on ONE line.
        # The quadratic path needs MANY unclosed braces per line (the lazy
        # [^}:\s]*? expands one char at a time per start '{'); the old
        # input (one '{' per line) was LINEAR on the pre-fix pattern.
        for i in range(100):
            lines.append(f" P{i:02d}.   1.0  : " + "{A" * 2000)
        lines.append("~CURVE INFORMATION")
        lines.append(" DEPT.M   :  Depth")
        lines.append("~A  DEPT")
        lines.append("100.0")
        content = "\n".join(lines) + "\n"

        start = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASParser().parse(content)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, (
            f"FORMAT_SPEC_PATTERN parse took {elapsed:.1f}s on 100 "
            f"unclosed-brace lines — quadratic backtracking regression (H-04)"
        )
        # The parameters themselves are still parsed correctly.
        assert len(las.parameters) == 100


# ──────────────────────────────────────────────────────────────
# H-02 (data_reader, HIGH): wrap-detection curve-count mismatch.
# ~C declares more curves than ~A rows → every line short → majority
# vote misclassified the file as WRAPPED and _read_wrapped dropped
# ~half the rows.  Fixed: uniformly short rows are parsed gracefully.
# ──────────────────────────────────────────────────────────────

class TestH02CurveCountMismatchNotWrapped:
    """H-02: A WRAP=NO file whose data rows consistently carry fewer
    values than declared curves (column-count mismatch) must NOT be
    misdetected as wrapped — every row must be preserved, with the
    missing curve null-filled."""

    def test_uniform_short_rows_not_wrapped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A  DEPT  GR  RHOB\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
            "1002.0  60.0\n"
        )
        test_file = tmp_path / "h02_short.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # All 3 rows preserved — not wrapped-and-dropped.
        assert len(data["logs"]["DEPT"]) == 3
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0, 60.0])
        # The undeclared-value curve is null-filled, not shifted.
        np.testing.assert_allclose(
            data["logs"]["RHOB"], [-999.25, -999.25, -999.25]
        )


# ──────────────────────────────────────────────────────────────
# M-38 (data_reader, MEDIUM): WRAP=YES with a complete first row.
# Mixed-wrap files (complete first line + wrapped continuation) were
# misdetected non-wrapped → DEPT pollution.  Content-based detection
# now handles the complete-first-row case.
# ──────────────────────────────────────────────────────────────

class TestM38WrapCompleteFirstRow:
    """M-38: A WRAP=YES file whose first line carries a full row and whose
    continuation lines wrap must parse correctly — the content-based
    detector must not classify it non-wrapped on the complete first row."""

    def test_wrap_yes_complete_first_row_parses(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : WRAPPED MODE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A  DEPT  GR\n"
            "1000.0  50.0\n"
            "1001.0\n"
            "55.0\n"
        )
        test_file = tmp_path / "m38_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0])


# ──────────────────────────────────────────────────────────────
# M-05/M-38 (LAS 3.0, MEDIUM): content-based wrap on the 3.0 path.
# Genuinely wrapped LAS 3.0 data (WRAP=YES declared) has no wrapped
# reader — the correct behavior is a LOUD rejection, not a silent
# DEPT-shift.
# ──────────────────────────────────────────────────────────────

class TestM05Las30WrappedRejected:
    """M-05/M-38-las30: LAS 3.0 has no wrapped reader.  A WRAP=YES
    LAS 3.0 file (or WRAP=NO file with genuinely wrapped continuation
    rows) must raise loudly — never silently misparse with a DEPT
    shift and garbage columns."""

    def test_las30_wrap_yes_raises(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : WRAPPED\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG\n"
            "DEPT  GR\n"
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "55.0\n"
        )
        test_file = tmp_path / "m05_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)


# ──────────────────────────────────────────────────────────────
# F-05 (LAS 3.0, MEDIUM): uniform-short-row guard.  WRAP=NO files
# with consistently short rows (column-count mismatch) must parse
# gracefully with null-fill — NOT be rejected with the factually
# wrong "WRAP=YES is not supported" error.
# ──────────────────────────────────────────────────────────────

class TestF05Las30UniformShortRows:
    """F-05: LAS 3.0 WRAP=NO + uniformly short rows (fewer values per
    row than declared curves) must null-fill gracefully, not raise
    "WRAP=YES is not supported" (which misdiagnoses a WRAP=NO file)."""

    def test_las30_uniform_short_rows_null_filled(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~LOG\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
        )
        test_file = tmp_path / "f05_short.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(sec.data["GR"], [50.0, 55.0])
        np.testing.assert_allclose(sec.data["RHOB"], [-999.25, -999.25])


# ──────────────────────────────────────────────────────────────
# M-10 (models, MEDIUM): one-shot iterables in container mutation.
# _GuardedList.extend/__iadd__/slice-__setitem__ consumed generators
# during validation and then mutated the exhausted iterator → silent
# data loss.  Each materializes before validating.
# ──────────────────────────────────────────────────────────────

class TestM10GuardedListOneShotIterables:
    """M-10: _GuardedList mutation methods must materialize one-shot
    iterables (generators) BEFORE validating so the items survive
    (pre-fix `las.curves.extend(gen)` → [])."""

    def test_extend_generator_preserves_items(self) -> None:
        gl = _GuardedList([], _expected_type=CurveDefinition)
        gen = (CurveDefinition(mnemonic=f"C{i}") for i in range(3))
        gl.extend(gen)
        assert [c.mnemonic for c in gl] == ["C0", "C1", "C2"]

    def test_iadd_generator_preserves_items(self) -> None:
        gl = _GuardedList([], _expected_type=CurveDefinition)
        gen = (CurveDefinition(mnemonic=f"D{i}") for i in range(2))
        gl += gen  # type: ignore[arg-type]
        assert [c.mnemonic for c in gl] == ["D0", "D1"]


# ──────────────────────────────────────────────────────────────
# M-13 (models, MEDIUM): _DevColumns.__ior__ bypassed all guards.
# dict.__ior__ (|=) skipped str-key validation, float64 coercion,
# length consistency, MAX_DATA_LINES, and column_order sync.
# Now routed through update() so every guard applies.
# ──────────────────────────────────────────────────────────────

class TestM13DevColumnsIor:
    """M-13: _DevColumns.__ior__ must validate like every other
    mutation route — wrong-length and non-str-key inserts raise,
    and valid inserts stay in sync with column_order."""

    def test_ior_wrong_length_raises(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        with pytest.raises(ValueError, match="length"):
            dev.columns |= {"TVD": np.array([3.0])}

    def test_ior_non_str_key_raises(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0])}, column_order=["MD"])
        with pytest.raises(TypeError, match="keys must be str"):
            dev.columns |= {7: np.array([3.0])}

    def test_ior_valid_roundtrips(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        dev.columns |= {"TVD": np.array([3.0, 4.0])}
        assert list(dev.columns.keys()) == ["MD", "TVD"]
        assert dev.column_order == ["MD", "TVD"]
        back = DevFile.from_dict(dev.to_dict())
        assert list(back.columns.keys()) == ["MD", "TVD"]


# ──────────────────────────────────────────────────────────────
# M-18 (models, MEDIUM): _DevColumns.popitem bypassed column_order
# sync (C-level dict.popitem skips __delitem__) → stale order →
# to_dict→from_dict LASDataError.  Now routes through __delitem__.
# ──────────────────────────────────────────────────────────────

class TestM18PopitemSyncsColumnOrder:
    """M-18: _DevColumns.popitem must remove the popped column from
    column_order (LIFO semantics preserved) so the to_dict→from_dict
    roundtrip stays consistent."""

    def test_popitem_syncs_column_order(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([1.0]),
                "TVD": np.array([2.0]),
                "INC": np.array([3.0]),
            },
            column_order=["MD", "TVD", "INC"],
        )
        key, _value = dev.columns.popitem()
        assert key == "INC"  # LIFO
        assert dev.column_order == ["MD", "TVD"]
        back = DevFile.from_dict(dev.to_dict())
        assert list(back.columns.keys()) == ["MD", "TVD"]

    def test_popitem_empty_raises_keyerror(self) -> None:
        dev = DevFile(columns={}, column_order=[])
        with pytest.raises(KeyError):
            dev.columns.popitem()


# ──────────────────────────────────────────────────────────────
# M-14/H-03 (models, HIGH): {I} int64 precision above 2^53 on the
# LAS 3.0 per-section from_dict path.  Pre-fix the per-section path
# coerced np.array(v, dtype=np.float64) unconditionally, rounding
# 9007199254740993 → 9007199254740992.0.  Full parse→to_dict→
# from_dict roundtrip preserves the exact int64 value.
# ──────────────────────────────────────────────────────────────

class TestM14Int64PrecisionRoundtrip:
    """M-14/H-03: {I} curve values above 2^53 survive the full
    LAS 3.0 parse → to_dict → from_dict roundtrip with exact integer
    precision (per-section path included)."""

    def test_int64_above_2_53_roundtrips(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " CNT.     :  Count {I}\n"
            "~LOG\n"
            "DEPT,CNT\n"
            "100.0,9007199254740993\n"
            "101.0,9007199254740995\n"
        )
        test_file = tmp_path / "m14_int64.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        cnt = las.logs["CNT"]
        assert int(cnt[0]) == 9007199254740993, (
            f"Precision lost at parse: {cnt[0]}"
        )
        assert int(cnt[1]) == 9007199254740995
        # to_dict → from_dict roundtrip preserves precision.
        las2 = LASFile.from_dict(las.to_dict())
        cnt2 = las2.logs["CNT"]
        assert int(cnt2[0]) == 9007199254740993
        assert int(cnt2[1]) == 9007199254740995


# ──────────────────────────────────────────────────────────────
# M-62 (data_reader, MEDIUM): huge integral NULL sentinel (>= 2^63)
# with a short row must not OverflowError.  The _null_is_integral
# gate checked integrality but never int64 representability.
# ──────────────────────────────────────────────────────────────

class TestM62HugeNullSentinel:
    """M-62: An integral NULL sentinel >= 2^63 (outside int64 range)
    with {I} curves must parse without raising OverflowError on the
    LAS 3.0 path."""

    def test_huge_null_sentinel_no_overflow(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA\n"
            "~WELL INFORMATION\n"
            " NULL.    9223372036854775808 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " CNT.     :  Count {I}\n"
            "~LOG\n"
            "DEPT,CNT\n"
            "100.0,9223372036854775808\n"
        )
        test_file = tmp_path / "m62_null.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # The huge sentinel value is preserved as data (no OverflowError).
        cnt = las.logs["CNT"]
        assert int(cnt[0]) == 9223372036854775808


# ──────────────────────────────────────────────────────────────
# M-04 (LAS 3.0, MEDIUM): reconcile of null fill cells must not
# clobber another section's logs.  Pre-fix, an earlier section's
# fill-cell row was written into a DIFFERENT section's top-level
# logs array (GR 30.0 clobbered → -999).
# ──────────────────────────────────────────────────────────────

class TestM04CrossSectionNullReconcile:
    """M-04: Null-fill reconciliation for one data section must never
    write into another section's top-level logs arrays.

    Discriminating shape (adversarial F-05): a PARSE-based repro — the
    write→read roundtrip of a constructed LASFile cannot catch this bug
    (the writer emits ~WELL before the data sections, so no fill cells
    are ever tracked).  This file puts ~CORE_DATA BEFORE ~WELL with a
    conversion-fail row whose column mnemonic matches a top-level log
    (GR), a ~A LOG_DATA with genuine GR values, and a ~WELL declaring a
    NULL DIFFERENT from the fill sentinel (-999 vs -999.25) so the
    reconcile write path executes.  Pre-fix the earlier section's
    fill-cell row was written into the LOG section's top-level logs
    array (GR 30.0 clobbered → -999); post-fix logs stay intact."""

    def test_second_section_data_not_clobbered(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  DEPTH {F}\n"
            " CORE1    :  CORE 1 {F}\n"
            " GR.API   :  GAMMA {F}\n"
            "~Core_Data\n"
            " 1000.0 1.5\n"
            " 1000.5 2.5\n"
            " 1001.0 na\n"
            " 1001.5 4.5\n"
            "~A\n"
            " 1000.0 10.0\n"
            " 1000.5 20.0\n"
            " 1001.0 30.0\n"
            " 1001.5 40.0\n"
            "~Well Information\n"
            " NULL .               -999 : NULL VALUE\n"
            " STRT .M             1000.0 : \n"
            " STOP .M             1001.5 : \n"
            " STEP .M               0.5 : \n"
        )
        test_file = tmp_path / "m04_cross.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(test_file)
        # The LOG section's top-level GR log must be preserved — pre-fix the
        # CORE_DATA fill-cell reconcile clobbered row index 2 (30.0 → -999).
        np.testing.assert_allclose(back.logs["GR"], [10.0, 20.0, 30.0, 40.0])


# ──────────────────────────────────────────────────────────────
# M-71 (LAS 3.0, MEDIUM): fill_cells tracking list must be released
# after reconcile even when declared NULL == the default sentinel.
# Pre-fix the `continue` skipped the clear → permanent memory
# retention on the returned LASFile.
# ──────────────────────────────────────────────────────────────

class TestM71FillCellsReleased:
    """M-71: After parsing, the (row, col) fill-cell tracking list must
    not remain attached to the returned LASFile's data sections — even
    when the declared NULL equals the default sentinel and no re-fill
    is needed (pre-fix the continue skipped the release)."""

    def test_fill_cells_cleared_when_null_matches(self, tmp_path: Path) -> None:
        from pylasdev._las30_data import _NULL_FILL_CELLS_ATTR

        # Data section BEFORE ~Well triggers fill-cell tracking with the
        # default sentinel; the well then declares the same NULL (-999.25).
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG\n"
            "DEPT,GR\n"
            "100.0\n"
            "101.0,30.0\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
        )
        test_file = tmp_path / "m71_fill.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        for section in las.data_sections:
            fill_cells = getattr(section, _NULL_FILL_CELLS_ATTR, None)
            assert not fill_cells, (
                "M-71: fill-cell tracking list retained on returned LASFile "
                f"({fill_cells!r})"
            )


# ──────────────────────────────────────────────────────────────
# M-26 (writer, MEDIUM): cross-section curve dedup preserved
# unit/format.  Pre-fix the second LOG_DATA section with the same
# mnemonics but different unit/format lost its definition → re-read
# re-labeled with the first unit.
# ──────────────────────────────────────────────────────────────

class TestM26SectionDedupPreservesUnit:
    """M-26: Two LOG_DATA sections using the same mnemonic with
    different units must each keep their own unit through write→read."""

    def test_same_mnemonic_different_units_preserved(self, tmp_path: Path) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
            logs={"DEPT": np.array([100.0, 101.0])},
        )
        las.well["NULL"] = "-999.25"
        s1 = DataSection(
            name="S1",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([1.0, 2.0])},
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="FT")],
        )
        s2 = DataSection(
            name="S2",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([3.0, 4.0])},
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
        )
        las.data_sections.append(s1)
        las.data_sections.append(s2)
        out = tmp_path / "m26_units.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        text = out.read_text(encoding="utf-8")
        assert "DEPT.FT" in text
        assert "DEPT.M" in text
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(str(out))
        assert back.data_sections[0].section_curves[0].unit == "FT"
        assert back.data_sections[1].section_curves[0].unit == "M"


# ──────────────────────────────────────────────────────────────
# M-64 (writer, MEDIUM): curve-dedup key must include array_info.index.
# Pre-fix NMR[1]/NMR[2] collided on mnemonic-only dedup → second
# section's definition dropped and data re-labeled NMR[1].
# ──────────────────────────────────────────────────────────────

class TestM64ArrayInfoIndexInIdentity:
    """M-64: Array curves NMR[1]/NMR[2] must not collide in section
    dedup — the identity includes array_info.index, so both sections
    keep their own array index through write→read."""

    def test_array_curves_not_collided(self, tmp_path: Path) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "NMR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="NMR", unit="", data_format="A"),
            ],
            logs={"DEPT": np.array([100.0, 101.0])},
            string_data={"NMR": np.array(["A", "B"])},
        )
        las.well["NULL"] = "-999.25"
        s1 = DataSection(
            name="S1",
            section_type="LOG_DATA",
            curves_order=["DEPT", "NMR[1]"],
            data={"DEPT": np.array([1.0, 2.0])},
            string_data={"NMR[1]": np.array(["X", "Y"])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="NMR[1]", unit=""),
            ],
        )
        s2 = DataSection(
            name="S2",
            section_type="LOG_DATA",
            curves_order=["DEPT", "NMR[2]"],
            data={"DEPT": np.array([3.0, 4.0])},
            string_data={"NMR[2]": np.array(["P", "Q"])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="NMR[2]", unit=""),
            ],
        )
        las.data_sections.append(s1)
        las.data_sections.append(s2)
        out = tmp_path / "m64_arrays.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(str(out))
        assert list(back.data_sections[0].string_data.keys()) == ["NMR[1]"]
        assert list(back.data_sections[1].string_data.keys()) == ["NMR[2]"]


# ──────────────────────────────────────────────────────────────
# M-66 (writer, MEDIUM): frozenset order-insensitive scoping → silent
# GR/DEPT column SWAP when a section has same curve-name set but
# different column order.  Fixed: per-section column order is
# preserved through write→read.
# ──────────────────────────────────────────────────────────────

class TestM66NoColumnSwapOnReorder:
    """M-66: A second section with the SAME curve names in a DIFFERENT
    column order must keep its own order — no silent column swap on
    roundtrip."""

    def test_reordered_section_values_not_swapped(self, tmp_path: Path) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={"DEPT": np.array([100.0]), "GR": np.array([30.0])},
        )
        las.well["NULL"] = "-999.25"
        s1 = DataSection(
            name="S1",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0, 2.0]), "GR": np.array([10.0, 11.0])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
        )
        s2 = DataSection(
            name="S2",
            section_type="LOG_DATA",
            curves_order=["GR", "DEPT"],  # REVERSED order
            data={"GR": np.array([20.0, 21.0]), "DEPT": np.array([200.0, 201.0])},
            section_curves=[
                CurveDefinition(mnemonic="GR", unit="GAPI"),
                CurveDefinition(mnemonic="DEPT", unit="M"),
            ],
        )
        las.data_sections.append(s1)
        las.data_sections.append(s2)
        out = tmp_path / "m66_reorder.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(str(out))
        s2b = back.data_sections[1]
        np.testing.assert_allclose(s2b.data["DEPT"], [200.0, 201.0])
        np.testing.assert_allclose(s2b.data["GR"], [20.0, 21.0])


# ──────────────────────────────────────────────────────────────
# M-83 (writer, MEDIUM): desc/api_code dedup warning.  Cross-section
# dedup was mnemonic-only and silently dropped the second section's
# desc/api_code.  Now the differing desc/api_code triggers a warning.
# ──────────────────────────────────────────────────────────────

class TestM83DescApiCodeDedupWarning:
    """M-83: Two sections with the same mnemonic+unit+format but
    DIFFERENT description must emit a dedup warning (not silently
    drop the second section's metadata)."""

    def test_differing_description_warns(self, tmp_path: Path) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT", unit="M", description="Depth")],
            logs={"DEPT": np.array([100.0])},
        )
        las.well["NULL"] = "-999.25"
        s1 = DataSection(
            name="S1",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([1.0])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", description="Depth A")
            ],
        )
        s2 = DataSection(
            name="S2",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([2.0])},
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M", description="Depth B")
            ],
        )
        las.data_sections.append(s1)
        las.data_sections.append(s2)
        out = tmp_path / "m83_desc.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(str(out), las)
        desc_warns = [
            str(w.message)
            for w in rec
            if "differing description" in str(w.message)
        ]
        assert len(desc_warns) >= 1, (
            f"Expected differing-description dedup warning, got: "
            f"{[str(w.message) for w in rec]}"
        )


# ──────────────────────────────────────────────────────────────
# M-50 (dev_reader, MEDIUM): comma-path all-integer first row must
# be headerless DATA (not fabricated numeric column names).  The
# whitespace path already returned headerless; the comma path
# returned ("simple",1) consuming the first station.
# ──────────────────────────────────────────────────────────────

class TestM50CommaAllIntegerHeaderless:
    """M-50: A comma file whose first row is all-integer data must be
    headerless — the first row is DATA, not numeric column names.
    No station is lost and no fabricated numeric headers appear."""

    def test_comma_all_integer_first_row_is_data(self, tmp_path: Path) -> None:
        content = "0,0,45\n100,250,55\n200,300,65\n"
        test_file = tmp_path / "m50_comma.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        assert data["col_0"][0] == 0.0  # first station preserved
        np.testing.assert_allclose(data["col_0"], [0.0, 100.0, 200.0])
        np.testing.assert_allclose(data["col_1"], [0.0, 250.0, 300.0])
        np.testing.assert_allclose(data["col_2"], [45.0, 55.0, 65.0])


class TestM51CommaCountMismatch:
    """M-51: A comma count-prefix file whose declared count disagrees
    with the data token count must be headerless — the count line is
    skipped and every data row is preserved (pre-fix the count became
    a data row, collapsing 3 of 4 columns)."""

    def test_count_mismatch_count_line_not_data(self, tmp_path: Path) -> None:
        # Count declares 4 but each data row has 3 tokens (mismatch).
        content = "4\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
        test_file = tmp_path / "m51_count.dev"
        test_file.write_text(content, encoding="utf-8")
        data = read_dev_file(test_file)
        assert list(data.keys()) == ["col_0", "col_1", "col_2"]
        np.testing.assert_allclose(data["col_0"], [1.0, 4.0])
        np.testing.assert_allclose(data["col_1"], [2.0, 5.0])
        np.testing.assert_allclose(data["col_2"], [3.0, 6.0])


# ──────────────────────────────────────────────────────────────
# M-76 (dev_reader, MEDIUM): multi-thousands-separator recombination.
# Values >= 1e6 with 2+ separators were only partially recombined —
# the true value was destroyed.  Now all consecutive pairs recombine.
# ──────────────────────────────────────────────────────────────

class TestM76MultiThousandsSeparator:
    """M-76: A value with MULTIPLE thousands separators (>= 1e6, e.g.
    1,234,567.8) must recombine into the true numeric value — not just
    the first pair."""

    def test_multi_separator_value_recombined(self, tmp_path: Path) -> None:
        content = "MD,TVD,X\n1,234,567.8,90\n"
        test_file = tmp_path / "m76_multi.dev"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)
        assert data["MD"][0] == 1234567.8, (
            f"Multi-separator value not fully recombined: {data['MD'][0]}"
        )
        assert data["TVD"][0] == 90.0

    def test_space_delimited_basic_parse(self, tmp_path: Path) -> None:
        """M-76 companion: a plain SPACE-delimited file parses correctly.

        This is deliberately NOT a recombination test — the multi-thousands-
        separator recombination is comma-delimiter-gated (dev_reader.py), so
        the space-delimited path can never recombine values; the comma
        variant above (test_multi_separator_value_recombined) is the M-76
        discriminator.  This sanity check only asserts the plain space-parse
        behavior (no separators present in the input)."""
        content = "MD TVD X\n1234567.8 90 5\n"
        test_file = tmp_path / "m76_space.dev"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)
        assert data["MD"][0] == 1234567.8


# ──────────────────────────────────────────────────────────────
# M-31 (encoding, MEDIUM): Cyrillic detection scans the WHOLE file.
# A fixed window (64K→1MiB) left files whose Cyrillic content starts
# beyond the window misdecoded as cp1252 → silent mojibake.
# ──────────────────────────────────────────────────────────────

class TestM31CyrillicBeyond1MiB:
    """M-31: cp1251 Cyrillic content starting BEYOND 1 MiB of ASCII
    preamble must still be detected (whole-file scan, no fixed window)."""

    def test_cyrillic_after_1mib_preamble(self, tmp_path: Path) -> None:
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u0422\u0415\u0421\u0422 "
        preamble = ("# " + "X" * 77 + "\n") * 14000  # ~1.1 MB ASCII
        raw = preamble.encode("ascii") + (russian * 300).encode("cp1251") + b"\n"
        test_file = tmp_path / "m31_1mib.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, text = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"M-31: Cyrillic beyond 1MiB misdecoded as {enc!r}"
        )
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in text


# ──────────────────────────────────────────────────────────────
# M-57 (encoding, MEDIUM): Cyrillic byte-frequency false-positive on
# Spanish/Portuguese.  n-tilde-dense cp1252 was decoded as cp1251
# (n-tilde -> Cyrillic es, o-acute -> Cyrillic u).  Run-length
# confirmation subsumes the frequency detector.
# ──────────────────────────────────────────────────────────────

class TestM57SpanishNotMisdetectedAsCp1251:
    """M-57: A genuine cp1252 Spanish file with accented letters must
    decode as cp1252 — NOT be false-positived as cp1251 (mojibake)."""

    def test_spanish_cp1252_stays_cp1252(self, tmp_path: Path) -> None:
        text = "CA\u00d1\u00d3N DE PERFORACI\u00d3N \u00d1\u00e1\u00f1ez \u00e9\u00e8\u00ea"
        test_file = tmp_path / "m57_spanish.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-57: Spanish cp1252 misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00f1" in content  # ñ preserved


# ──────────────────────────────────────────────────────────────
# M-81 (encoding, MEDIUM): №-dense cp1251.  E-06's near-tie rescue
# failed at №-density > ~4% because each № inflated the cp1252
# ratio.  The №-artifact is now subtracted from the ratio gap.
# ──────────────────────────────────────────────────────────────

class TestM81NumeroDenseCp1251:
    """M-81: A №-dense genuine cp1251 file (many '№' markers) must
    decode as cp1251 — not flip to cp1252 (mojibake)."""

    def test_numero_dense_cp1251_stays_cp1251(self, tmp_path: Path) -> None:
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u2116 123 \u041c\u0415\u0421\u0422\u041e "
        test_file = tmp_path / "m81_numero.las"
        test_file.write_bytes((russian * 100).encode("cp1251"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, _content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"M-81: №-dense cp1251 misdecoded as {enc!r}"
        )


# ──────────────────────────────────────────────────────────────
# M-82/F-18 (encoding, MEDIUM): run-length detector false-positive
# on Western cp1252 with consecutive accented bytes, and the №-rule
# must be ADJACENCY-scoped (a far-away footnote ¹ must not flip a
# genuine Western file to cp1251).
# ──────────────────────────────────────────────────────────────

class TestM82AdjacencyScopedNumeroRule:
    """M-82/F-18: A genuine cp1252 Western file with accented letters
    and a single '¹' far from the accented run must stay cp1252 — the
    №-confirmation rule is adjacency-scoped, not whole-file membership."""

    def test_western_with_far_numero_stays_cp1252(self, tmp_path: Path) -> None:
        # 'Ñáñez' (3 accented bytes) at the start; '¹' in a footnote
        # ~1KB away — pre-fix the whole-file №-rule flipped to cp1251.
        text = "WELL \u00d1\u00e1\u00f1ez " + ("x" * 1000) + " TVD \u00b9"
        test_file = tmp_path / "m82_western.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-82: Western cp1252 + far-away ¹ misdecoded as {enc!r}"
        )
        # No Cyrillic mojibake.
        assert not any(0x0400 <= ord(c) <= 0x04FF for c in content)


# ──────────────────────────────────────────────────────────────
# M-32 (compare, MEDIUM): compare_las_dicts NOT symmetric —
# np.allclose uses the SECOND operand as rtol reference.  Fixed:
# symmetric tolerance comparison.
# ──────────────────────────────────────────────────────────────

class TestM32CompareSymmetry:
    """M-32: compare_las_dicts must be symmetric under argument swap —
    comparing (a, b) gives the same answer as comparing (b, a).

    Discriminating shape (adversarial F-03): ndarray operands at the
    tolerance boundary.  np.allclose uses the SECOND operand as the
    rtol reference: |1000-999| = 1 vs rtol*999 = 0.999 → False, but vs
    rtol*1000 = 1.0 → True.  Pre-fix fwd=False/bwd=True (asymmetric);
    post-fix both True.  Python-list operands do NOT discriminate
    (pre-fix exact-compared lists are symmetric False/False)."""

    def test_symmetric_under_arg_swap(self) -> None:
        d1 = {"x": np.array([1000.0])}
        d2 = {"x": np.array([999.0])}
        fwd = compare_las_dicts(d1, d2, rtol=1e-3, atol=0.0)
        bwd = compare_las_dicts(d2, d1, rtol=1e-3, atol=0.0)
        assert fwd == bwd, (
            f"M-32: comparison not symmetric: fwd={fwd} bwd={bwd}"
        )
        assert fwd is True, (
            f"M-32: values within rtol=1e-3 should compare equal: fwd={fwd}"
        )


# ──────────────────────────────────────────────────────────────
# M-33 (compare, MEDIUM): masked invalid values EQUAL via dict path,
# UNEQUAL in lists — 3 paths gave 2 answers.  Masked→NaN unification
# now applies consistently in the list path too.
# ──────────────────────────────────────────────────────────────

class TestM33MaskedEqualityUnified:
    """M-33: Two masked values compare EQUAL in the list path (unified
    with the dict/data_sections/nested paths via masked→NaN handling)."""

    def test_masked_values_equal_in_list_path(self) -> None:
        ma1 = np.ma.array(42.0, mask=True)
        ma2 = np.ma.array(99.0, mask=True)
        assert _compare_lists([ma1], [ma2], "test", 1e-7, 0.0) is True

    def test_masked_equal_in_dict_path(self) -> None:
        ma1 = np.ma.array(42.0, mask=True)
        ma2 = np.ma.array(99.0, mask=True)
        assert compare_las_dicts(
            {"logs": {"V": [ma1]}}, {"logs": {"V": [ma2]}}
        ) is True


# ──────────────────────────────────────────────────────────────
# M-58 (compare, MEDIUM): rtol/atol silently ignored for Python-list
# data.  _compare_lists now routes homogeneous numeric lists to
# np.allclose so tolerance applies.
# ──────────────────────────────────────────────────────────────

class TestM58ListTolerance:
    """M-58: Python-list data must honor rtol/atol — values within
    tolerance compare equal (pre-fix lists ignored tolerance entirely)."""

    def test_list_within_tolerance_equal(self) -> None:
        assert compare_las_dicts(
            {"x": [1.0]}, {"x": [1.005]}, rtol=1e-2, atol=0.0
        ) is True

    def test_list_beyond_tolerance_unequal(self) -> None:
        assert compare_las_dicts(
            {"x": [1.0]}, {"x": [2.0]}, rtol=1e-2, atol=0.0
        ) is False


# ──────────────────────────────────────────────────────────────
# F-17 (compare, MEDIUM): int64 subtraction/abs overflow in
# _allclose_symmetric.  Pre-fix -2^63 vs 0 (two's-complement wrap)
# compared equal.  Now int64 is promoted before diff/abs.
# ──────────────────────────────────────────────────────────────

class TestF17Int64OverflowCompare:
    """F-17: int64 comparison must not wrap on subtraction/abs —
    -2^63 vs 0 must NOT compare equal (both are common int64
    missing-sentinel patterns)."""

    def test_int64_sentinel_not_equal_to_zero(self) -> None:
        r = compare_las_dicts(
            {"logs": {"V": np.array([-2**63], dtype=np.int64)}},
            {"logs": {"V": np.array([0], dtype=np.int64)}},
        )
        assert r is False

    def test_equal_int64_still_equal(self) -> None:
        r = compare_las_dicts(
            {"logs": {"V": np.array([5], dtype=np.int64)}},
            {"logs": {"V": np.array([5], dtype=np.int64)}},
        )
        assert r is True


# ──────────────────────────────────────────────────────────────
# M-37/M-40 (data_reader/parser, MEDIUM): standalone mnemonic header
# row after ~A consumed as data → phantom all-null first row + shift.
# The header row must be SKIPPED on both the LAS 1.2/2.0 reader path
# and the LAS 3.0 parser path — no phantom row, no crash.
# ──────────────────────────────────────────────────────────────

class TestM37MnemonicHeaderSkipped:
    """M-37/M-40: A standalone mnemonic header row directly below ~A
    (e.g. 'DEPT  GR') is a header, not data — it must be skipped so no
    phantom all-null first row is created and values don't shift."""

    def test_las20_mnemonic_header_skipped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT  GR\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
        )
        test_file = tmp_path / "m37_header.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        # No phantom row, no shift, no crash.
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0])
        # No inconsistent-length crash escapes.
        assert not any(
            "inconsistent lengths" in str(w.message) for w in rec
        )

    def test_las30_mnemonic_header_skipped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG\n"
            "DEPT  GR\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
        )
        test_file = tmp_path / "m40_header.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0] if las.data_sections else las
        data = sec.data if hasattr(sec, "data") else las.logs
        np.testing.assert_allclose(data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["GR"], [50.0, 55.0])


# ──────────────────────────────────────────────────────────────
# F-19/M-03 (parser/data_reader, MEDIUM): all-string sections must
# preserve mnemonic-coincident rows.  A string value that equals a
# curve mnemonic (LITH=['LITH','SHALE']) was dropped as a "header".
# String rows are never dropped.
# ──────────────────────────────────────────────────────────────

class TestF19AllStringSectionsPreserveRows:
    """F-19/M-03: In an all-string section, a data row whose value
    coincides with a curve mnemonic is legitimate string data — it
    must NOT be dropped as a mnemonic header."""

    def test_las20_all_string_rows_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            "~A  DEPT  LITH\n"
            "1000.0  LITH\n"
            "1001.0  SHALE\n"
            "1002.0  SAND\n"
        )
        test_file = tmp_path / "f19_allstring.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_array_equal(
            data["string_data"]["LITH"], np.array(["LITH", "SHALE", "SAND"])
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])

    def test_las30_all_string_rows_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            "~LOG\n"
            "DEPT  LITH\n"
            "1000.0  LITH\n"
            "1001.0  SHALE\n"
            "1002.0  SAND\n"
        )
        test_file = tmp_path / "f19_las30_allstring.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_array_equal(
            sec.string_data["LITH"], np.array(["LITH", "SHALE", "SAND"])
        )


# ──────────────────────────────────────────────────────────────
# M-01/M-02 (data_reader, MEDIUM): the regressing-function family
# (_is_mnemonic_header_row) — mnem_base header rows and WRAP=YES
# mixed sections.  mnem_base {LLD→BFV, LLS→BFV} + a raw-vendor
# standalone header must not produce a phantom row; WRAP=YES mixed
# sections must not drop mnemonic-coincident string continuation
# rows.
# ──────────────────────────────────────────────────────────────

class TestM01MnemBaseHeader:
    """M-01: A mnem_base header row (DEPT LLD LLS, where LLD/LLS both
    resolve to BFV) must be recognized as a header and skipped — no
    phantom first row, no overcount warning."""

    def test_mnem_base_header_no_phantom_row(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LLD.OHMM :  Laterolog deep\n"
            " LLS.OHMM :  Laterolog shallow\n"
            "~A\n"
            "DEPT  LLD  LLS\n"
            "1000.0  10.0  20.0\n"
            "1001.0  11.0  21.0\n"
            "1002.0  12.0  22.0\n"
        )
        test_file = tmp_path / "m01_mnem.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            data = read_las_file(test_file, mnem_base=MNEM_BASE)
        # No phantom row — 3 real rows, starting at 1000.0.
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        assert data["curves_order"] == ["DEPT", "BFV", "LLS"]
        # No overcount warning (header recognized by both pre-scan and reader).
        assert not any(
            "overcount" in str(w.message).lower() for w in rec
        )


class TestM02WrapYesMixedSection:
    """M-02: WRAP=YES mixed section (numeric + {S} string curves) whose
    string continuation value coincides with a curve mnemonic must not
    drop the row (the token-count clause rejects the 1-token continuation
    row before any mnemonic match)."""

    def test_wrap_yes_mixed_section_string_row_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            "~A\n"
            "1000.0\n"
            "LITH\n"
            "1001.0\n"
            "SHALE\n"
            "1002.0\n"
            "SAND\n"
        )
        test_file = tmp_path / "m02_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_array_equal(
            data["string_data"]["LITH"], np.array(["LITH", "SHALE", "SAND"])
        )
