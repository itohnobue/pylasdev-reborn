"""Regression tests for verified findings from Stage 6 and Stage 9 fixes.

These tests exercise specific fixes identified by the adversarial verification
pipeline. Each test documents which finding it covers and tests the actual
fixed behaviour against the current (fixed) source.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from pylasdev import (
    read_dev_file,
    read_dev_file_as_object,
    read_las_file,
    read_las_file_as_object,
    write_las_file,
)
from pylasdev._las30_data import _build_spec_form_array_info
from pylasdev.compare import _compare_lists, compare_las_dicts
from pylasdev.data_reader import (
    _declared_mnemonic_set,
    _deduplicate_curves,
    _mnemonic_header_declared,
)
from pylasdev.encoding import read_with_encoding
from pylasdev.exceptions import LASDataError, LASParseError
from pylasdev.mnem_base import MNEM_BASE, build_mnemonic_lookup, resolve_mnemonic
from pylasdev.models import (
    CurveDefinition,
    DataSection,
    DevFile,
    LASFile,
    ParameterEntry,
    VersionSection,
    _GuardedDict,
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
            " COMP.    ACME\u00a0Corp : COMPANY\n"
        )
        parser = LASParser()
        las = parser.parse(content)
        # The NO-BREAK SPACE is inside the dot-delimited section value.
        # After _SPLITLINES_CHARS_RE substitution (\\u00A0 is now in the
        # regex), the raw access depends on whether the regex substitutes
        # with a space. Verify the company name is extracted.
        assert "COMP" in las.well
        # Verify \u00A0 was replaced by regular space (the F-001 fix behavior)
        assert "\u00a0" not in las.well["COMP"], (
            "NO-BREAK SPACE should be replaced by regular space"
        )
        assert " " in las.well["COMP"], "Value should contain regular space after NBSP replacement"


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
            writer_warnings = [x for x in w if "no data in 'logs'" in str(x.message)]
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
            f"Expected wrap='NO' after write (disk state), got wrap={las.version.wrap!r}"
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
            f"curves_order mutated: {list(las.curves_order)} vs {original_curves_order}"
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
            section_type="",  # empty string — valid "no particular type"
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
            f"Expected version.dlm='COMMA' after write (restored), got dlm={las.version.dlm!r}"
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
            f"Expected version.dlm='SPACE' after write (unchanged), got dlm={las.version.dlm!r}"
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
            f"Expected version.dlm='COMMA' after write, got dlm={las.version.dlm!r}"
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
            f"Expected version.dlm='COMMA' after write, got dlm={las.version.dlm!r}"
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
            f"Expected version.dlm='COMMA' (restored), got dlm={las.version.dlm!r}"
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
            f"H-01: LOG_DATA scoped to wrong curves: {las.data_sections[1].curves_order}"
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
        np.testing.assert_allclose(data["logs"]["RHOB"], [-999.25, -999.25, -999.25])


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
        assert int(cnt[0]) == 9007199254740993, f"Precision lost at parse: {cnt[0]}"
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
                f"M-71: fill-cell tracking list retained on returned LASFile ({fill_cells!r})"
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
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="M", description="Depth A")],
        )
        s2 = DataSection(
            name="S2",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([2.0])},
            section_curves=[CurveDefinition(mnemonic="DEPT", unit="M", description="Depth B")],
        )
        las.data_sections.append(s1)
        las.data_sections.append(s2)
        out = tmp_path / "m83_desc.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(str(out), las)
        desc_warns = [str(w.message) for w in rec if "differing description" in str(w.message)]
        assert len(desc_warns) >= 1, (
            f"Expected differing-description dedup warning, got: {[str(w.message) for w in rec]}"
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
# F-06 (dev_reader, MEDIUM): thousands-DECIMAL conversion.
# Detection recognized "1,234.5"/"1,234,567.8"/"1,234E3" as numeric
# data but conversion silently dropped them to NaN (grammar drift:
# _THOUSANDS_GROUP_RE was bare-integer while _THOUSANDS_NUMBER_RE
# accepted decimal/exponent forms).  The shared normalization
# grammar now converts unambiguous decimal/exponent forms in every
# delimiter context with the loud thousands warning (never silent
# NaN); the ambiguous bare "1,234" keeps the documented M-25/I2-17
# delimiter-aware policy.
# ──────────────────────────────────────────────────────────────


class TestF06ThousandsDecimalConversion:
    """F-06: thousands-DECIMAL variants convert to real values (not NaN).

    Each test FAILS on pre-fix code (NaN + generic failure counter only,
    no thousands warning) and PASSES on post-fix (real value + thousands
    summary warning).
    """

    def test_space_decimal_thousands_converts_to_value(self, tmp_path: Path) -> None:
        """F-06 core regression: ``1,234.5`` in a space-delimited
        headerless file reads as 1234.5, NOT NaN, with the thousands
        warning (pre-fix: all-NaN column with only the generic
        conversion-failure warning)."""
        content = "1,234.5 5,678.9\n9,000.5 10,456.7\n"
        test_file = tmp_path / "f06_space_decimal.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["col_0"], [1234.5, 9000.5])
        np.testing.assert_array_equal(data["col_1"], [5678.9, 10456.7])
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Decimal thousands must fire the thousands warning, got: {[str(x.message) for x in w]}"
        )

    def test_space_multi_group_decimal_thousands_converts(self, tmp_path: Path) -> None:
        """``1,234,567.8`` (M-76 form) converts in space mode too."""
        content = "1,234,567.8 90.0\n2,345,678.9 91.0\n"
        test_file = tmp_path / "f06_space_multi.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["col_0"], [1234567.8, 2345678.9])
        np.testing.assert_array_equal(data["col_1"], [90.0, 91.0])
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Multi-group decimal thousands must fire the thousands warning, "
            f"got: {[str(x.message) for x in w]}"
        )

    def test_space_exponent_thousands_converts(self, tmp_path: Path) -> None:
        """``1,234E3`` converts to 1234000.0 (exponent-bearing form)."""
        content = "1,234E3 1.0\n2,345E2 2.0\n"
        test_file = tmp_path / "f06_space_exp.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["col_0"], [1234000.0, 234500.0])

    def test_semicolon_decimal_thousands_converts(self, tmp_path: Path) -> None:
        """``1,234.5`` / ``2,345,678.9`` in a semicolon file convert to
        real values (pre-fix: NaN in the semicolon context too)."""
        content = "MD;TVD;X;Y\n100.0;90.0;1,234.5;2,345,678.9\n200.0;190.0;3,456.7;4,567.8\n"
        test_file = tmp_path / "f06_semi_decimal.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        np.testing.assert_array_equal(data["MD"], [100.0, 200.0])
        np.testing.assert_array_equal(data["X"], [1234.5, 3456.7])
        np.testing.assert_array_equal(data["Y"], [2345678.9, 4567.8])
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Semicolon decimal thousands must fire the thousands warning, "
            f"got: {[str(x.message) for x in w]}"
        )

    def test_detection_conversion_agreement(self) -> None:
        """F-06 property (pre-fix audit recommendation): every token that
        DETECTION recognizes as a thousands number must either convert to
        a finite value or be counted in ``_thousands_counter`` — never a
        silent NaN with only the generic failure counter."""
        from pylasdev.dev_reader import _dev_to_finite_float, _is_thousands_number

        corpus = [
            "1,234",
            "-12,345",
            "1,234.5",
            "-1,234.5",
            "1,234,567.8",
            "12,345.6",
            "1,234E3",
            "1,234e-3",
            "-1,234,567.8E2",
            "+1,234",
            "999,999,999",
        ]
        for token in corpus:
            assert _is_thousands_number(token), (
                f"Corpus token {token!r} must be recognized as a thousands number"
            )
            fc: list[int] = [0]
            tc: list[int] = [0]
            result = _dev_to_finite_float(
                token,
                np.nan,
                _failure_counter=fc,
                _thousands_counter=tc,
                _comma_as_thousands=True,
            )
            # Never a silent NaN: either converted to a finite value or
            # counted in _thousands_counter (loud summary warning).
            if np.isnan(result):
                assert tc[0] >= 1, (
                    f"Token {token!r} NaN without the thousands counter — silent data loss (F-06)"
                )
            else:
                assert np.isfinite(result)
                assert tc[0] >= 1, (
                    f"Token {token!r} converted but not counted in _thousands_counter"
                )

    def test_bare_thousands_space_still_nan_with_warning(self, tmp_path: Path) -> None:
        """M-25 control: the ambiguous bare form ``1,234`` in space mode
        STILL reads as NaN with the loud thousands warning (documented
        delimiter-aware policy must not regress)."""
        content = "1,234 5,678\n9,000 10,456\n"
        test_file = tmp_path / "f06_bare_control.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert all(np.isnan(v) for v in data["col_0"]), (
            f"Bare thousands in space must stay NaN, got {list(data['col_0'])}"
        )
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1, (
            f"Bare thousands must fire the thousands warning, got: {[str(x.message) for x in w]}"
        )


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
        assert enc == "cp1251", f"M-31: Cyrillic beyond 1MiB misdecoded as {enc!r}"
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
        assert enc == "cp1252", f"M-57: Spanish cp1252 misdecoded as {enc!r} (mojibake)"
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
        assert enc == "cp1251", f"M-81: №-dense cp1251 misdecoded as {enc!r}"


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
        assert enc == "cp1252", f"M-82: Western cp1252 + far-away ¹ misdecoded as {enc!r}"
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
        assert fwd == bwd, f"M-32: comparison not symmetric: fwd={fwd} bwd={bwd}"
        assert fwd is True, f"M-32: values within rtol=1e-3 should compare equal: fwd={fwd}"


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
        assert compare_las_dicts({"logs": {"V": [ma1]}}, {"logs": {"V": [ma2]}}) is True


# ──────────────────────────────────────────────────────────────
# M2 (compare, MEDIUM): the F-07 fix reintroduced the M-33
# masked-vs-NaN divergence — the array path unwraps masked arrays by
# mask (masked vs unmasked-NaN → False) while the list path NaN-filled
# masked items (masked vs unmasked-NaN → True).  Both paths must
# agree: a masked position matches only another masked position, never
# an unmasked value or a NaN.
# ──────────────────────────────────────────────────────────────


class TestM2MaskedVsNanPathConsistent:
    """M2: masked-vs-NaN must return the SAME verdict on the array and
    list paths.  Pre-fix (F-07): array path False, list path True —
    the exact "3 paths gave 2 answers" inconsistency M-33 unified."""

    def test_array_path_masked_vs_nan_not_equal(self) -> None:
        """Array path: a masked position never matches a NaN."""
        d1 = {"x": np.ma.array([1.0, 2.0], mask=[0, 1])}
        d2 = {"x": np.array([1.0, np.nan])}
        assert compare_las_dicts(d1, d2) is False

    def test_array_path_nan_vs_masked_not_equal(self) -> None:
        d1 = {"x": np.array([1.0, np.nan])}
        d2 = {"x": np.ma.array([1.0, 2.0], mask=[0, 1])}
        assert compare_las_dicts(d1, d2) is False

    def test_list_path_masked_vs_nan_not_equal(self) -> None:
        """List path must agree with the array path (pre-fix returned
        True — the F-07 NaN-fill conflated masked with genuine NaN)."""
        d1 = {"x": [np.ma.array(2.0, mask=True)]}
        d2 = {"x": [np.nan]}
        assert compare_las_dicts(d1, d2) is False

    def test_list_path_nan_vs_masked_not_equal(self) -> None:
        d1 = {"x": [np.nan]}
        d2 = {"x": [np.ma.array(2.0, mask=True)]}
        assert compare_las_dicts(d1, d2) is False

    def test_mixed_list_masked_item_vs_nan_not_equal(self) -> None:
        """A masked item inside a multi-element list never matches NaN
        (pre-fix: NaN-fill made [1.0, masked] == [1.0, NaN] True)."""
        d1 = {"x": [1.0, np.ma.array(2.0, mask=True)]}
        d2 = {"x": [1.0, np.nan]}
        assert compare_las_dicts(d1, d2) is False

    def test_list_path_same_verdict_as_array_path(self) -> None:
        """Cross-path consistency: the same logical data (a masked value
        vs NaN) must produce the same verdict whether expressed as
        ndarrays or as Python lists."""
        array_verdict = compare_las_dicts(
            {"x": np.ma.array([1.0, 2.0], mask=[0, 1])},
            {"x": np.array([1.0, np.nan])},
        )
        list_verdict = compare_las_dicts(
            {"x": [1.0, np.ma.array(2.0, mask=True)]},
            {"x": [1.0, np.nan]},
        )
        assert array_verdict is False
        assert list_verdict is False

    def test_masked_vs_masked_still_equal(self) -> None:
        """M-33 preserved: two masked values still compare equal."""
        d1 = {"x": [np.ma.array(2.0, mask=True)]}
        d2 = {"x": [np.ma.array(99.0, mask=True)]}
        assert compare_las_dicts(d1, d2) is True

    def test_mixed_list_masked_vs_masked_equal(self) -> None:
        d1 = {"x": [1.0, np.ma.array(2.0, mask=True)]}
        d2 = {"x": [1.0, np.ma.array(7.0, mask=True)]}
        assert compare_las_dicts(d1, d2) is True

    def test_plain_nan_vs_nan_still_equal(self) -> None:
        """No overcorrection: genuine NaN vs genuine NaN stays equal."""
        assert compare_las_dicts({"x": [np.nan]}, {"x": [np.nan]}) is True

    def test_masked_int_list_vs_nan_not_equal(self) -> None:
        """Masked INTEGER items also never match NaN."""
        d1 = {"x": [np.ma.array(5, mask=True)]}
        d2 = {"x": [np.nan]}
        assert compare_las_dicts(d1, d2) is False

    def test_masked_int_list_vs_unmasked_int_not_equal(self) -> None:
        d1 = {"x": [np.ma.array(5, mask=True)]}
        d2 = {"x": [5]}
        assert compare_las_dicts(d1, d2) is False


# ──────────────────────────────────────────────────────────────
# M-58 (compare, MEDIUM): rtol/atol silently ignored for Python-list
# data.  _compare_lists now routes homogeneous numeric lists to
# np.allclose so tolerance applies.
# ──────────────────────────────────────────────────────────────


class TestM58ListTolerance:
    """M-58: Python-list data must honor rtol/atol — values within
    tolerance compare equal (pre-fix lists ignored tolerance entirely)."""

    def test_list_within_tolerance_equal(self) -> None:
        assert compare_las_dicts({"x": [1.0]}, {"x": [1.005]}, rtol=1e-2, atol=0.0) is True

    def test_list_beyond_tolerance_unequal(self) -> None:
        assert compare_las_dicts({"x": [1.0]}, {"x": [2.0]}, rtol=1e-2, atol=0.0) is False


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
            {"logs": {"V": np.array([-(2**63)], dtype=np.int64)}},
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
# F-07 (compare, MEDIUM): int64/uint64 -> float64 promotion in
# _allclose_symmetric collapsed integer values above 2^53, making
# genuinely different integer arrays compare EQUAL.  Integer-dtype
# operands are now compared exactly (no float64 promotion); the list
# path (_list_to_numeric_array) converts all-integer lists to int64
# instead of float64.  MaskedArray operands are unwrapped by mask so
# unmasked int64 data keeps full precision.
# ──────────────────────────────────────────────────────────────


class TestF07Int64PrecisionCompare:
    """F-07: int64/uint64 values above 2^53 must not collapse to
    float64 during comparison — genuinely different integer arrays
    must compare unequal (pre-fix they compared EQUAL)."""

    def test_int64_above_2_53_different_values_not_equal(self) -> None:
        """The core F-07 defect: [2^53] vs [2^53+1] compared True pre-fix
        (both collapse to the same float64).  Now must be False."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([2**53], dtype=np.int64)}},
            {"logs": {"V": np.array([2**53 + 1], dtype=np.int64)}},
        )
        assert r is False

    def test_int64_above_2_53_two_apart_not_equal(self) -> None:
        """Even 2 apart (2^53 vs 2^53+2) collapsed pre-fix."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([2**53], dtype=np.int64)}},
            {"logs": {"V": np.array([2**53 + 2], dtype=np.int64)}},
        )
        assert r is False

    def test_int64_above_2_53_same_values_equal(self) -> None:
        """Identical int64 values above 2^53 still compare equal."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([2**53 + 1], dtype=np.int64)}},
            {"logs": {"V": np.array([2**53 + 1], dtype=np.int64)}},
        )
        assert r is True

    def test_list_int64_above_2_53_different_values_not_equal(self) -> None:
        """The list path collapsed too: _list_to_numeric_array converted
        all-integer lists to float64.  [2^53] vs [2^53+1] must be False."""
        assert compare_las_dicts({"logs": {"V": [2**53]}}, {"logs": {"V": [2**53 + 1]}}) is False

    def test_list_int64_above_2_53_same_values_equal(self) -> None:
        assert compare_las_dicts({"logs": {"V": [2**53 + 1]}}, {"logs": {"V": [2**53 + 1]}}) is True

    def test_uint64_above_2_53_different_values_not_equal(self) -> None:
        """uint64 operands (kind 'u') are in scope for the exact path too."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([2**63], dtype=np.uint64)}},
            {"logs": {"V": np.array([2**63 + 1], dtype=np.uint64)}},
        )
        assert r is False

    def test_masked_int64_above_2_53_different_values_not_equal(self) -> None:
        """MaskedArray operands were pre-converted to float64-filled-NaN,
        which collapsed unmasked int64 values above 2^53 as well."""
        a1 = np.ma.array([2**53, 5], dtype=np.int64, mask=[0, 1])
        a2 = np.ma.array([2**53 + 1, 5], dtype=np.int64, mask=[0, 1])
        assert compare_las_dicts({"logs": {"V": a1}}, {"logs": {"V": a2}}) is False

    def test_masked_int64_above_2_53_same_values_equal(self) -> None:
        a1 = np.ma.array([2**53 + 1, 5], dtype=np.int64, mask=[0, 1])
        a2 = np.ma.array([2**53 + 1, 5], dtype=np.int64, mask=[0, 1])
        assert compare_las_dicts({"logs": {"V": a1}}, {"logs": {"V": a2}}) is True

    def test_masked_int64_masked_vs_unmasked_not_equal(self) -> None:
        """A masked position must never match an unmasked value."""
        a1 = np.ma.array([2**53, 5], dtype=np.int64, mask=[0, 1])
        a2 = np.ma.array([2**53, 5], dtype=np.int64, mask=[0, 0])
        assert compare_las_dicts({"logs": {"V": a1}}, {"logs": {"V": a2}}) is False

    def test_f17_wrap_sentinel_still_not_equal(self) -> None:
        """The F-17 fix is preserved: far-apart int64 sentinels must not
        wrap in two's complement (native int64 diff/abs would overflow)."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([-(2**63)], dtype=np.int64)}},
            {"logs": {"V": np.array([0], dtype=np.int64)}},
        )
        assert r is False

    def test_mixed_int_float_uses_tolerance_promotion(self) -> None:
        """Mixed int/float pairs keep float64 promotion (the float operand
        bounds representable precision); equal values still compare equal."""
        r = compare_las_dicts(
            {"logs": {"V": np.array([2**53 + 1], dtype=np.int64)}},
            {"logs": {"V": np.array([2**53 + 1.0])}},
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
        assert not any("inconsistent lengths" in str(w.message) for w in rec)

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
        np.testing.assert_array_equal(sec.string_data["LITH"], np.array(["LITH", "SHALE", "SAND"]))


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
        assert not any("overcount" in str(w.message).lower() for w in rec)


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


# ──────────────────────────────────────────────────────────────
# DR-01 (data_reader, MEDIUM): wrap misdetection — first-line-full
# rule short-circuits to non-wrapped when declared WRAP≠YES, but
# genuinely wrapped data (full first row OR mnemonic-header row
# masquerading as full) read via _read_normal → silent column shift.
# F-07 depth-line evidence rule: a later 1-value row is wrapped
# evidence that outranks a NO/absent header.
# ──────────────────────────────────────────────────────────────


class TestDR01WrapMisdetection:
    """DR-01: A WRAP=NO/absent file whose data is genuinely wrapped
    (first row complete, then depth/continuation lines) must be read as
    wrapped — never silently column-shifted by the first-line-full
    short-circuit."""

    def test_wrap_no_full_first_row_parses_wrapped(self, tmp_path: Path) -> None:
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
            "1000.0  50.0  1.0\n"
            "1001.0\n"
            "55.0  2.0\n"
            "1002.0\n"
            "60.0  3.0\n"
        )
        test_file = tmp_path / "dr01_full_first.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Pre-fix: window [3,1,2,1] → first-line-full → non-wrapped →
        # DEPT=[1000,1001,55,1002,60] (silent shift). Post-fix: wrapped.
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [1.0, 2.0, 3.0])

    def test_wrap_no_mnemonic_header_masquerade_parses_wrapped(self, tmp_path: Path) -> None:
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
            "~A\n"
            "DEPT  GR  RHOB\n"
            "1000.0\n"
            "50.0  1.0\n"
            "1001.0\n"
            "55.0  2.0\n"
            "1002.0\n"
            "60.0  3.0\n"
        )
        test_file = tmp_path / "dr01_mnemonic.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Pre-fix: mnemonic header row (full width) + WRAP=NO → first-line
        # full → non-wrapped → DEPT polluted with GR values. Post-fix:
        # depth-line evidence (1-value rows after the header) → wrapped.
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [1.0, 2.0, 3.0])


# ──────────────────────────────────────────────────────────────
# I2-04 (data_reader, MEDIUM): majority-vote misfire — 2 full rows in
# the window (mnemonic header + complete row, or two complete rows) +
# genuinely wrapped trailing data → non-wrapped even with WRAP=YES
# declared → silent column-identity shift.  F-07: WRAP=YES + later
# depth-line evidence → wrapped.
# ──────────────────────────────────────────────────────────────


class TestI204WrapYesTwoFullRows:
    """I2-04: WRAP=YES + 2 full leading rows + genuinely wrapped trailing
    data must be detected as wrapped (the majority vote's unconditional
    full_count>=2 must not beat the declaration + depth-line evidence)."""

    def test_wrap_yes_two_full_rows_then_wrapped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A\n"
            "DEPT  GR  RHOB\n"
            "1000.0  50.0  1.0\n"
            "1001.0\n"
            "55.0  2.0\n"
            "1002.0\n"
            "60.0  3.0\n"
        )
        test_file = tmp_path / "i204_two_full.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Pre-fix: window [3,3,1,2] → full_count>=2 → non-wrapped → silent
        # shift despite WRAP=YES. Post-fix: depth-line evidence → wrapped.
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [1.0, 2.0, 3.0])


# ──────────────────────────────────────────────────────────────
# DR-02 (data_reader, MEDIUM): _get_null_value except clause omits
# OverflowError — the documented raise of _parse_float_with_d_notation
# escapes as a raw OverflowError on Python <=3.12.  Dedicated
# except OverflowError → LASParseError (F-03, NOT a tuple append).
# ──────────────────────────────────────────────────────────────


class TestDR02NullOverflowError:
    """DR-02: A NULL well value whose float() conversion raises
    OverflowError (Python <=3.12 '1e400') must surface as the documented
    LASParseError boundary, not a raw OverflowError."""

    def test_get_null_value_overflow_raises_las_parse_error(self) -> None:
        from pylasdev.data_reader import _get_null_value

        ws = {"NULL": "1e400"}
        with mock.patch(
            "pylasdev.data_reader._parse_float_with_d_notation",
            side_effect=OverflowError("simulated <=3.12 float overflow"),
        ):
            with pytest.raises(LASParseError):
                _get_null_value(ws)

    def test_read_file_null_overflow_raises_las_parse_error(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    1e400 : OVERFLOW\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "dr02_null.las"
        test_file.write_text(content, encoding="utf-8")
        with mock.patch(
            "pylasdev.data_reader._parse_float_with_d_notation",
            side_effect=OverflowError("simulated <=3.12 float overflow"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(LASParseError):
                    read_las_file(test_file)


# ──────────────────────────────────────────────────────────────
# DR-05 (data_reader, MEDIUM): no Python-object cap on LAS 1.2/2.0
# string-data accumulation.  MAX_TOTAL_ELEMENTS counts elements (~8B)
# but Python str/object values amplify ~4-12x (up to ~50-100 GB).
# Mirror the LAS 3.0 _MAX_STRING_VALUES cap on both reader paths.
# ──────────────────────────────────────────────────────────────


class TestDR05StringObjectCap:
    """DR-05: The 1.2/2.0 string-accumulation paths must reject files
    whose string-curve values exceed the Python-object cap."""

    def test_normal_path_exceeds_string_cap(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " LITH.    :  Lith {S}\n"
            "~A  DEPT  LITH\n"
            "1000.0  SAND\n"
            "1001.0  SHALE\n"
            "1002.0  LIME\n"
        )
        test_file = tmp_path / "dr05_normal.las"
        test_file.write_text(content, encoding="utf-8")
        with mock.patch("pylasdev.data_reader.MAX_STRING_VALUES", 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(LASParseError):
                    read_las_file(test_file)

    def test_normal_path_under_cap_parses(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " LITH.    :  Lith {S}\n"
            "~A  DEPT  LITH\n"
            "1000.0  SAND\n"
            "1001.0  SHALE\n"
        )
        test_file = tmp_path / "dr05_under.las"
        test_file.write_text(content, encoding="utf-8")
        with mock.patch("pylasdev.data_reader.MAX_STRING_VALUES", 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = read_las_file(test_file)
        np.testing.assert_array_equal(data["string_data"]["LITH"], np.array(["SAND", "SHALE"]))

    def test_wrapped_path_exceeds_string_cap(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " LITH.    :  Lith {S}\n"
            "~A\n"
            "1000.0\nSAND\n1001.0\nSHALE\n1002.0\nLIME\n"
        )
        test_file = tmp_path / "dr05_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with mock.patch("pylasdev.data_reader.MAX_STRING_VALUES", 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(LASParseError):
                    read_las_file(test_file)


# ──────────────────────────────────────────────────────────────
# I2-02 (data_reader, MEDIUM): DLM=COMMA + embedded comma in a {S}
# string value truncates the string AND destroys the following
# column's genuine value (100.0,WELL, TX,10 → WELL truncated, GR
# destroyed).  csv.reader quote-awareness is deliberately NOT used
# (F2-015: the writer emits raw delimiter.join()); the fix is a LOUD
# warning naming the truncation/loss.
# ──────────────────────────────────────────────────────────────


class TestI202CommaEmbeddedDelimiterWarning:
    """I2-02: A DLM=COMMA file whose {S} string value contains an
    embedded comma must emit a clear warning that the value was
    truncated/lost — never a silent column shift."""

    def test_embedded_comma_in_string_warns(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " WELL .   :  Well Name {S}\n"
            " GR.GAPI  :  Gamma {F}\n"
            "~A  DEPT  WELL  GR\n"
            "100.0,WELL, TX,10\n"
            "101.0,WELL B,20\n"
        )
        test_file = tmp_path / "i202_comma.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        # The truncation/loss must be loud — a warning that names the
        # delimiter-in-string mechanism.
        assert any("delimiter" in str(w.message) and "string" in str(w.message) for w in rec), (
            f"Expected embedded-delimiter warning, got: {[str(w.message) for w in rec]}"
        )
        # The genuine GR value for the non-corrupt second row is preserved.
        assert data["logs"]["GR"][1] == 20.0


# ──────────────────────────────────────────────────────────────
# Stage 8 MODELS-A fixes (s8-fix-models-a):
# MOD-11, MOD-14, MOD-17, MOD-23, MOD-01, I2-01, MOD-22, PXM-06
# Each test FAILS on the pre-fix tree and PASSES post-fix.
# ──────────────────────────────────────────────────────────────


class TestMod11MetaPrefixHijack:
    """MOD-11: DevFile.from_dict must not hijack user columns literally
    named ``_meta_*``.  Only the CLOSED set of well-known metadata keys
    (encoding/source_file/column_order) carrying a metadata-shaped value
    (str/list-of-str — the shapes to_dict emits) are treated as metadata;
    array-valued ``_meta_<known>`` keys and unknown ``_meta_*`` keys are
    user columns stored verbatim."""

    def test_meta_source_file_array_column_preserved(self) -> None:
        """A user column literally named ``_meta_source_file`` (array value)
        is preserved as data — no silent drop, no source_file hijack."""
        dev = DevFile.from_dict(
            {
                "_meta_source_file": np.array([1.0, 2.0]),
                "DEPT": np.array([0.0, 1.0]),
            }
        )
        assert "_meta_source_file" in dev.columns
        np.testing.assert_allclose(dev.columns["_meta_source_file"], [1.0, 2.0])
        # Metadata NOT overwritten with the array's string repr.
        assert dev.source_file == ""
        np.testing.assert_allclose(dev.columns["DEPT"], [0.0, 1.0])

    def test_meta_encoding_array_column_preserved(self) -> None:
        dev = DevFile.from_dict(
            {
                "_meta_encoding": np.array([5.0, 6.0]),
                "MD": np.array([0.0, 1.0]),
            }
        )
        assert "_meta_encoding" in dev.columns
        np.testing.assert_allclose(dev.columns["_meta_encoding"], [5.0, 6.0])
        assert dev.encoding == "utf-8"

    def test_unknown_meta_suffix_is_column_verbatim(self) -> None:
        """An unknown ``_meta_*`` suffix is a user column under its literal
        name (no strip, no warning about a 'column' being dropped).  With
        normalize_aliases=False the literal name is preserved verbatim."""
        dev = DevFile.from_dict(
            {
                "_meta_custom": np.array([9.0, 9.0]),
                "MD": np.array([0.0, 1.0]),
            },
            normalize_aliases=False,
        )
        assert "_meta_custom" in dev.columns
        np.testing.assert_allclose(dev.columns["_meta_custom"], [9.0, 9.0])
        assert dev.source_file == ""
        assert dev.encoding == "utf-8"

    def test_unknown_meta_suffix_normalize_aliases_is_column(self) -> None:
        """With the default normalize_aliases=True an unknown ``_meta_*``
        column is still a COLUMN (data preserved, no crash) — the name is
        normalized like any other column name."""
        dev = DevFile.from_dict(
            {
                "_meta_custom": np.array([9.0, 9.0]),
                "MD": np.array([0.0, 1.0]),
            }
        )
        assert "_META_CUSTOM" in dev.columns
        np.testing.assert_allclose(dev.columns["_META_CUSTOM"], [9.0, 9.0])

    def test_real_metadata_keys_still_roundtrip(self) -> None:
        """The real metadata keys (bare and collision-emitted) still
        roundtrip through to_dict/from_dict."""
        dev = DevFile(columns={"MD": np.array([0.0, 1.0])}, column_order=["MD"])
        dev.source_file = "well.dev"
        dev.encoding = "cp1251"
        d = dev.to_dict()
        assert d["source_file"] == "well.dev"
        assert d["encoding"] == "cp1251"
        dev2 = DevFile.from_dict(d)
        assert dev2.source_file == "well.dev"
        assert dev2.encoding == "cp1251"
        assert list(dev2.columns.keys()) == ["MD"]

    def test_meta_collision_roundtrip_source_file(self) -> None:
        """Column named 'source_file' (collision) — to_dict emits
        ``_meta_source_file`` (str metadata) + bare 'source_file' (array
        column); from_dict restores both."""
        dev = DevFile(
            columns={"MD": np.array([0.0, 1.0]), "source_file": np.array([7.0, 8.0])},
            column_order=["MD", "source_file"],
        )
        dev.source_file = "the-meta"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = dev.to_dict()
        assert d["_meta_source_file"] == "the-meta"
        dev2 = DevFile.from_dict(d)
        assert dev2.source_file == "the-meta"
        assert "source_file" in dev2.columns
        np.testing.assert_allclose(dev2.columns["source_file"], [7.0, 8.0])


class TestMod14EmptyContainerUpdate:
    """MOD-14: _GuardedDict.update/|= on an EMPTY container must validate
    the incoming batch (the first key defines the batch length) instead of
    silently accepting inconsistent lengths (writer then null-padded
    fabricated -999.25 rows)."""

    def test_empty_update_inconsistent_lengths_raises(self) -> None:
        g = _GuardedDict()
        with pytest.raises(ValueError, match="inconsistent"):
            g.update({"A": [1.0] * 5, "B": [1.0] * 3})

    def test_empty_ior_inconsistent_lengths_raises(self) -> None:
        g = _GuardedDict()
        with pytest.raises(ValueError, match="inconsistent"):
            g |= {"A": np.arange(5, dtype=float), "B": np.arange(3, dtype=float)}

    def test_empty_update_consistent_succeeds(self) -> None:
        g = _GuardedDict()
        g.update({"A": np.arange(5, dtype=float), "B": np.arange(5, dtype=float)})
        assert list(g.keys()) == ["A", "B"]
        assert len(g["A"]) == 5 and len(g["B"]) == 5

    def test_empty_setitem_single_key_succeeds(self) -> None:
        g = _GuardedDict()
        g["A"] = np.arange(3, dtype=float)
        assert len(g["A"]) == 3


class TestMod17NoNdimGuard:
    """MOD-17: 2-D arrays must be rejected with a clear error at EVERY
    construction/mutation entry point (from_dict, __setitem__, update,
    DataSection, DevFile).  Previously numeric paths crashed with a
    misleading LASWriteError and the string_data path SILENTLY corrupted
    data on write→read."""

    def _lasfile_2d(self) -> LASFile:
        las = LASFile(version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"))
        las.logs["DEPT"] = np.array([1.0, 2.0])
        return las

    def test_from_dict_2d_log_raises_lasdataerror(self) -> None:
        data = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([[1.0, 2.0], [3.0, 4.0]])},
        }
        with pytest.raises(LASDataError, match="1-D"):
            LASFile.from_dict(data)

    def test_setitem_2d_raises(self) -> None:
        las = self._lasfile_2d()
        with pytest.raises(ValueError, match="1-D"):
            las.logs["GR"] = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_update_2d_raises(self) -> None:
        las = self._lasfile_2d()
        with pytest.raises(ValueError, match="1-D"):
            las.logs.update({"GR": np.array([[1.0, 2.0], [3.0, 4.0]])})

    def test_string_data_2d_raises(self) -> None:
        las = self._lasfile_2d()
        with pytest.raises(ValueError, match="1-D"):
            las.string_data["LITH"] = np.array([["a", "b"], ["c", "d"]])

    def test_datasection_2d_data_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            DataSection(
                curves_order=["GR"],
                data={"GR": np.array([[1.0, 2.0], [3.0, 4.0]])},
            )

    def test_devfile_2d_column_raises(self) -> None:
        dev = DevFile(columns={"MD": np.array([0.0, 1.0])}, column_order=["MD"])
        with pytest.raises(ValueError, match="1-D"):
            dev.columns["AZI"] = np.array([[90.0, 90.0], [91.0, 91.0]])

    def test_1d_arrays_still_accepted(self) -> None:
        las = self._lasfile_2d()
        las.logs["GR"] = np.array([10.0, 20.0])
        assert list(las.logs["GR"]) == [10.0, 20.0]


class TestMod23ScalarStringContract:
    """MOD-23: _GuardedDict must reject non-array-like values (scalars,
    str/bytes) whose len() coincidentally matches the row count.  The old
    length-only guard let scalar 3.14 into a 1-row file (validate()=0
    issues, writer crashed with 'len() of unsized object') and 'abc' into
    a 3-row file."""

    def test_scalar_into_1row_raises(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["DEPT"] = np.array([1.0])
        with pytest.raises(ValueError, match="array-like"):
            las.logs["GR"] = 3.14

    def test_string_into_3row_raises(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["DEPT"] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="array-like"):
            las.logs["GR"] = "abc"

    def test_valid_arrays_pass(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["DEPT"] = np.array([1.0, 2.0, 3.0])
        las.logs["GR"] = np.array([10.0, 20.0, 30.0])
        assert len(las.logs["GR"]) == 3


class TestMod01GuardedGrowth:
    """MOD-01: guarded containers must GROW post-construction when the
    resulting state is consistent (the reader bypasses internally via
    dict.__setitem__; the public API now supports the same via update/|=)."""

    def test_consistent_growth_via_update_succeeds(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["A"] = np.arange(5, dtype=float)
        las.logs["B"] = np.arange(5, dtype=float)
        las.logs.update({"A": np.arange(10, dtype=float), "B": np.arange(10, dtype=float)})
        assert len(las.logs["A"]) == 10 and len(las.logs["B"]) == 10

    def test_inconsistent_growth_raises(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["A"] = np.arange(5, dtype=float)
        las.logs["B"] = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="inconsistent"):
            las.logs.update({"A": np.arange(10, dtype=float), "B": np.arange(7, dtype=float)})

    def test_partial_growth_raises(self) -> None:
        """Growing only ONE key of a multi-key container is inconsistent."""
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["A"] = np.arange(5, dtype=float)
        las.logs["B"] = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="inconsistent"):
            las.logs["A"] = np.arange(10, dtype=float)

    def test_single_key_container_growth_succeeds(self) -> None:
        """Replacing the ONLY key with a different length is consistent
        growth (trivially consistent resulting state)."""
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs["A"] = np.arange(5, dtype=float)
        las.logs["A"] = np.arange(10, dtype=float)
        assert len(las.logs["A"]) == 10


class TestI2_01ParameterDetailsPriority:
    """I2-01: parameter_details must not be silently dropped when the
    ``parameters`` key is absent or list-form.  When present it is the
    first-class, metadata-rich key and takes priority over both forms."""

    def _base(self) -> dict[str, Any]:
        return {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }

    def test_parameter_details_only_preserved(self) -> None:
        las = LASFile.from_dict(
            {
                **self._base(),
                "parameter_details": [
                    {
                        "mnemonic": "BHT",
                        "unit": "DEGC",
                        "value": "35.5",
                        "description": "Bottom Hole Temp",
                    }
                ],
            }
        )
        assert len(las.parameters) == 1
        assert las.parameters[0].mnemonic == "BHT"
        assert las.parameters[0].unit == "DEGC"
        assert las.parameters[0].value == "35.5"
        assert las.parameters[0].description == "Bottom Hole Temp"

    def test_parameter_details_with_list_form_parameters(self) -> None:
        """List-form ``parameters`` + ``parameter_details``: details wins
        (metadata preserved) instead of details being silently dropped."""
        las = LASFile.from_dict(
            {
                **self._base(),
                "parameters": [{"mnemonic": "BHT", "value": "99.9"}],
                "parameter_details": [{"mnemonic": "BS", "unit": "MM", "value": "200"}],
            }
        )
        assert len(las.parameters) == 1
        assert las.parameters[0].mnemonic == "BS"
        assert las.parameters[0].unit == "MM"
        assert las.parameters[0].value == "200"


class TestMod22WholesaleReassignment:
    """MOD-22: wholesale reassignment (lf.logs = {...} / ds.data = {...})
    must re-wrap into _GuardedDict and validate — previously it replaced
    the guard with a plain dict, validate() reported 0 issues, and the
    writer silently fabricated -999.25 rows."""

    def test_lasfile_logs_reassignment_inconsistent_raises(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        with pytest.raises(ValueError, match="inconsistent"):
            las.logs = {
                "DEPT": np.array([1.0, 2.0, 3.0]),
                "GR": np.array([10.0, 20.0]),
            }

    def test_lasfile_logs_reassignment_stays_guarded(self) -> None:
        las = LASFile(version=VersionSection(vers="2.0"))
        las.logs = {
            "DEPT": np.array([1.0, 2.0, 3.0]),
            "GR": np.array([10.0, 20.0, 30.0]),
        }
        assert isinstance(las.logs, _GuardedDict)
        # Guard still works after reassignment.
        with pytest.raises(ValueError, match="inconsistent"):
            las.logs["RHOB"] = np.array([1.0, 2.0])

    def test_lasfile_string_data_reassignment_stays_guarded(self) -> None:
        las = LASFile(version=VersionSection(vers="3.0"))
        las.string_data = {"WELL": np.array(["a", "b"])}
        assert isinstance(las.string_data, _GuardedDict)

    def test_datasection_data_reassignment_inconsistent_raises(self) -> None:
        ds = DataSection(curves_order=["DEPT", "GR"], _from_dict=True)
        ds.data = {"DEPT": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="inconsistent"):
            ds.data = {
                "DEPT": np.array([1.0, 2.0, 3.0]),
                "GR": np.array([10.0, 20.0]),
            }

    def test_datasection_data_reassignment_stays_guarded(self) -> None:
        ds = DataSection(curves_order=["DEPT", "GR"], _from_dict=True)
        ds.data = {
            "DEPT": np.array([1.0, 2.0]),
            "GR": np.array([10.0, 20.0]),
        }
        assert isinstance(ds.data, _GuardedDict)
        with pytest.raises(ValueError, match="inconsistent"):
            ds.data["RHOB"] = np.array([1.0, 2.0, 3.0])


class TestPXM06CurveCollisionCanonicalPriority:
    """PXM-06 (models side): from_dict curve normalization must accept the
    alias-before-canonical collision state (e.g. ['DEPT','LLD','BFV'] with
    MNEM_BASE) without LASDataError, preserving BOTH curves' identity.
    The canonical name wins its own slot; the earlier alias is re-keyed to
    its original mnemonic (mirroring the well path's raw==resolved branch)."""

    def test_from_dict_alias_before_canonical_succeeds(self) -> None:
        data = {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT", "LLD", "BFV"],
            "curves": [
                {"mnemonic": "DEPT"},
                {"mnemonic": "LLD"},
                {"mnemonic": "BFV"},
            ],
            "logs": {
                "DEPT": np.array([1.0, 2.0]),
                "LLD": np.array([3.0, 4.0]),
                "BFV": np.array([5.0, 6.0]),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data, mnem_base=MNEM_BASE)
        # No LASDataError.  Both distinct curves survive under distinct
        # names; no duplicate in curves_order.
        assert las.curves_order == ["DEPT", "LLD", "BFV"]
        assert [c.mnemonic for c in las.curves] == ["DEPT", "LLD", "BFV"]
        assert list(las.logs.keys()) == ["DEPT", "LLD", "BFV"]
        # Identity is NOT swapped: LLD data stays under LLD, BFV under BFV.
        np.testing.assert_allclose(las.logs["LLD"], [3.0, 4.0])
        np.testing.assert_allclose(las.logs["BFV"], [5.0, 6.0])

    def test_from_dict_legacy_alias_before_canonical_succeeds(self) -> None:
        """Legacy path (no 'curves' key — created from curves_order)."""
        data = {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT", "AK", "DT"],
            "logs": {
                "DEPT": np.array([1.0, 2.0]),
                "AK": np.array([11.0, 12.0]),
                "DT": np.array([21.0, 22.0]),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data, mnem_base=MNEM_BASE)
        assert las.curves_order == ["DEPT", "AK", "DT"]
        np.testing.assert_allclose(las.logs["AK"], [11.0, 12.0])
        np.testing.assert_allclose(las.logs["DT"], [21.0, 22.0])

    def test_per_section_alias_before_canonical_data_preserved(self) -> None:
        """Per-section data keys are normalized BEFORE the section's
        curves_order, so the canonical-priority branch must re-key the
        alias's stored data (dict-dest) — otherwise LLD's values are
        silently overwritten by BFV's under the 'BFV' key."""
        data = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([1.0, 2.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "LLD", "BFV"],
                    "data": {
                        "DEPT": np.array([1.0, 2.0]),
                        "LLD": np.array([3.0, 4.0]),
                        "BFV": np.array([5.0, 6.0]),
                    },
                }
            ],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data, mnem_base=MNEM_BASE)
        ds = las.data_sections[0]
        assert ds.curves_order == ["DEPT", "LLD", "BFV"]
        assert list(ds.data.keys()) == ["DEPT", "LLD", "BFV"]
        np.testing.assert_allclose(ds.data["LLD"], [3.0, 4.0])
        np.testing.assert_allclose(ds.data["BFV"], [5.0, 6.0])

    def test_canonical_first_control_still_works(self) -> None:
        """Canonical-first order must still produce the N-I-30 result."""
        data = {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT", "BFV", "LLD"],
            "curves": [
                {"mnemonic": "DEPT"},
                {"mnemonic": "BFV"},
                {"mnemonic": "LLD"},
            ],
            "logs": {
                "DEPT": np.array([1.0, 2.0]),
                "BFV": np.array([5.0, 6.0]),
                "LLD": np.array([3.0, 4.0]),
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data, mnem_base=MNEM_BASE)
        assert las.curves_order == ["DEPT", "BFV", "LLD"]
        np.testing.assert_allclose(las.logs["BFV"], [5.0, 6.0])
        np.testing.assert_allclose(las.logs["LLD"], [3.0, 4.0])


# ──────────────────────────────────────────────────────────────
# MOD-02 (MEDIUM): parameter data_format preprocessing asymmetric
# ──────────────────────────────────────────────────────────────


class TestMOD02ParameterDataFormatNormalization:
    """MOD-02: from_dict's parameter data_format path RAISED on lowercase
    'f' and invalid single-char 'X' while the curve path normalized and the
    parser warn-and-cleared — three construction paths, three outcomes on
    identical input.  from_dict now uppercases the parameter data_format
    (mirroring the curve path) and warn-and-clears invalid single-char
    codes (mirroring the parser)."""

    def _base_dict(self) -> dict[str, Any]:
        return {
            "version": {"VERS": "3.0"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([1.0])},
            "parameters": {"MUD": "x"},
        }

    def test_from_dict_lowercase_f_normalizes_to_F(self) -> None:
        """Pre-fix: from_dict RAISED LASDataError for 'f'.  Post-fix:
        the value is uppercased to 'F' exactly like the curve path."""
        data = self._base_dict()
        data["parameter_details"] = [
            {"mnemonic": "MUD", "value": "x", "description": "mud", "data_format": "f"}
        ]
        las = LASFile.from_dict(data)
        assert las.parameters[0].data_format == "F"

    def test_from_dict_invalid_X_warns_and_clears(self) -> None:
        """Pre-fix: from_dict RAISED LASDataError for 'X'.  Post-fix:
        a warning is emitted and data_format is cleared to '' (parser-
        aligned tolerance)."""
        data = self._base_dict()
        data["parameter_details"] = [
            {"mnemonic": "MUD", "value": "x", "description": "mud", "data_format": "X"}
        ]
        with pytest.warns(UserWarning, match="Clearing to empty string"):
            las = LASFile.from_dict(data)
        assert las.parameters[0].data_format == ""

    def test_direct_parameter_entry_lowercase_f_normalizes(self) -> None:
        """MOD-02 direct-construction twin: ParameterEntry('f') normalizes
        to 'F' (pre-fix it raised)."""
        pe = ParameterEntry(mnemonic="MUD", value="x", data_format="f")
        assert pe.data_format == "F"

    def test_direct_parameter_entry_invalid_X_warns_and_clears(self) -> None:
        """MOD-02 direct-construction twin: ParameterEntry('X') warns and
        clears (pre-fix it raised)."""
        with pytest.warns(UserWarning, match="Clearing to empty string"):
            pe = ParameterEntry(mnemonic="MUD", value="x", data_format="X")
        assert pe.data_format == ""


# ──────────────────────────────────────────────────────────────
# MOD-12 (MEDIUM): from_dict rejects extra-curves state that
# __post_init__ explicitly allows
# ──────────────────────────────────────────────────────────────


class TestMOD12FromDictExtraCurveDefinitions:
    """MOD-12: __post_init__ documents that extra curve definitions
    (curves beyond curves_order length — LAS 3.0 per-section definitions
    also registered at the top level) are tolerated, but from_dict raised
    LASDataError on them, so the library's OWN to_dict→from_dict roundtrip
    of the documented-valid state failed.  from_dict now mirrors
    __post_init__: extra definitions are tolerated, only a curves_order
    LONGER than curves is rejected."""

    def test_roundtrip_with_extra_curve_definitions(self) -> None:
        """Pre-fix: to_dict()→from_dict() raised LASDataError
        ('curves_order length (2) does not match curves length (3)').
        Post-fix: the roundtrip succeeds and preserves both curves."""
        las = LASFile(
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
                CurveDefinition(mnemonic="DT"),  # extra (per-section)
            ],
            logs={"DEPT": np.array([1.0, 2.0]), "GR": np.array([3.0, 4.0])},
        )
        data = las.to_dict()
        back = LASFile.from_dict(data)
        assert back.curves_order == ["DEPT", "GR"]
        assert [c.mnemonic for c in back.curves] == ["DEPT", "GR", "DT"]
        np.testing.assert_allclose(back.logs["GR"], [3.0, 4.0])

    def test_curves_order_longer_than_curves_still_rejected(self) -> None:
        """An order entry with no matching definition is still invalid —
        the tolerance is one-directional (extra definitions OK, missing
        definitions not)."""
        with pytest.raises(ValueError, match="greater than curves length"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "2.0"},
                    "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
                    "curves_order": ["DEPT", "GR"],
                    "curves": [{"mnemonic": "DEPT"}],
                    "logs": {"DEPT": np.array([1.0])},
                }
            )


# ──────────────────────────────────────────────────────────────
# I2-12 (MEDIUM): curves_order plain unguarded list post-construction
# ──────────────────────────────────────────────────────────────


class TestI212CurvesOrderMutationGuarded:
    """I2-12: curves_order was a plain list whose __post_init__ element-type
    guard was bypassable via the public list API (``append(42)``) — the
    per-section path wrote a column-count-corrupted file and the top-level
    path crashed the writer with a raw AttributeError.  curves_order is now
    wrapped in a _GuardedList (``_expected_type=str``) so every mutation
    entry point validates element types."""

    def test_datasection_append_int_raises(self) -> None:
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0, 2.0]), "GR": np.array([3.0, 4.0])},
        )
        with pytest.raises(TypeError, match="items must be str"):
            ds.curves_order.append(42)  # type: ignore[arg-type]  # I2-12: guard must reject

    def test_lasfile_append_int_raises(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
            ],
            logs={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        with pytest.raises(TypeError, match="items must be str"):
            las.curves_order.append(42)  # type: ignore[arg-type]  # I2-12: guard must reject

    def test_datasection_insert_and_setitem_guarded(self) -> None:
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        with pytest.raises(TypeError, match="items must be str"):
            ds.curves_order.insert(1, 42)  # type: ignore[arg-type]  # I2-12: guard must reject
        with pytest.raises(TypeError, match="items must be str"):
            ds.curves_order[0] = 42  # type: ignore[call-overload]  # I2-12: guard must reject

    def test_wholesale_assignment_rewraps_guard(self) -> None:
        """Wholesale ``ds.curves_order = [...]`` must re-wrap through the
        guarded list (I2-12 self-healing via __setattr__), so a subsequent
        append(42) still raises."""
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        ds.curves_order = ["DEPT", "GR", "DT"]
        assert isinstance(ds.curves_order, _GuardedList)
        with pytest.raises(TypeError, match="items must be str"):
            ds.curves_order.append(42)


# ──────────────────────────────────────────────────────────────
# I2-13 (MEDIUM, models side): post-construction curves_order mutation
# → silent column swap; validate() must detect the desync
# ──────────────────────────────────────────────────────────────


class TestI213CurvesOrderMutationValidation:
    """I2-13 (models side): post-construction curves_order mutation
    (reverse/insert/reorder) desynced the order from section_curves/curves,
    and validate() reported 0 issues — the writer silently swapped columns.
    validate(complete=True) now detects order/data mismatches so the writer
    never silently emits swapped columns."""

    def test_datasection_reorder_detected_by_validate(self) -> None:
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR", "DT"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
                CurveDefinition(mnemonic="DT"),
            ],
            data={
                "DEPT": np.array([1.0, 2.0]),
                "GR": np.array([3.0, 4.0]),
                "DT": np.array([5.0, 6.0]),
            },
        )
        assert ds.validate(complete=True) == []
        # POST-CONSTRUCTION mutation: reverse the live order.
        ds.curves_order.reverse()
        issues = ds.validate(complete=True)
        assert any("does not match" in issue and "section_curves" in issue for issue in issues), (
            f"no desync issue after reorder: {issues}"
        )

    def test_lasfile_top_level_reorder_detected_by_validate(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT", "GR", "DT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
                CurveDefinition(mnemonic="DT"),
            ],
            logs={
                "DEPT": np.array([1.0, 2.0]),
                "GR": np.array([3.0, 4.0]),
                "DT": np.array([5.0, 6.0]),
            },
        )
        las.curves_order.reverse()
        issues = las.validate(complete=True)
        assert any("does not match" in issue and "curves[0]" in issue for issue in issues), (
            f"no desync issue after top-level reorder: {issues}"
        )

    def test_datasection_orphaned_data_detected_by_validate(self) -> None:
        """Removing an order entry whose data still exists must be flagged
        (the writer would silently drop that column)."""
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
            ],
            data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        del ds.curves_order[0]
        issues = ds.validate(complete=True)
        assert any("NOT in curves_order" in issue for issue in issues), (
            f"no orphaned-data issue after deletion: {issues}"
        )

    def test_wl_m1_datasection_case_variant_no_false_uncovered_orphaned(self) -> None:
        """WL-M1 (M13 residual, models side): DataSection.validate's
        uncovered + orphaned checks were exact-case.  On the M13 state
        (curves_order=['dept','GR'], data keyed DEPT/GR) they emitted
        FALSE 'will pad' and 'will not emit' diagnostics although the
        writer's case-insensitive lookup emits the data.  The checks must
        compare upper-cased so the supported case-variant state reports no
        issues — while a genuinely uncovered curve still warns (asserted
        in the same section)."""
        ds = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        ds.curves_order = ["dept", "GR"]  # case-variant order entries
        issues = ds.validate(complete=True)
        assert not any("will pad" in issue for issue in issues), issues
        assert not any("will not emit" in issue for issue in issues), issues

        # True positive preserved: a genuinely uncovered curve still warns.
        ds2 = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0])},  # GR genuinely uncovered
        )
        issues2 = ds2.validate(complete=True)
        assert any("will pad" in issue for issue in issues2), issues2


# ──────────────────────────────────────────────────────────────
# MOD-1 / MOD-2 (MEDIUM, models side): case-insensitive positional +
# orphan checks (WL-M1 class, DataSection + LASFile top level)
# ──────────────────────────────────────────────────────────────


class TestMOD12CaseInsensitiveCaseVariant:
    """MOD-1 / MOD-2 (WL-M1 class): the writer resolves curves_order ↔
    section_curves/curves/logs/string_data keys case-insensitively, so
    validate() must NOT emit false desync/orphan diagnostics on the
    supported case-variant state (curves_order=['dept','GR'] with
    definitions/data keyed DEPT/GR) — while a genuine reorder (a
    different curve name in the position) must still be detected."""

    def test_mod1_datasection_case_variant_no_false_desync(self) -> None:
        """MOD-1: DataSection.validate's positional check (:2602) compared
        exact-case, so a populated section_curves + case-variant
        curves_order emitted a FALSE 'has desynced the column order' issue
        although the writer resolves the per-section order
        case-insensitively and emits the data in the right position."""
        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        ds.section_curves = [
            CurveDefinition(mnemonic="DEPT", unit="M"),
            CurveDefinition(mnemonic="GR", unit="GAPI"),
        ]
        # POST-CONSTRUCTION mutation: lowercase the first order entry.
        ds.curves_order = ["dept", "GR"]
        issues = ds.validate(complete=True)
        assert not any("desynced the column order" in i for i in issues), issues
        assert not any("does not match" in i for i in issues), issues

        # True positive preserved: a genuine reorder (different curve in the
        # position, same casing) still desyncs.
        ds2 = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        ds2.section_curves = [
            CurveDefinition(mnemonic="DEPT", unit="M"),
            CurveDefinition(mnemonic="GR", unit="GAPI"),
        ]
        ds2.curves_order = ["GR", "DEPT"]
        issues2 = ds2.validate(complete=True)
        assert any("desynced the column order" in i for i in issues2), issues2

    def test_mod2_lasfile_top_level_case_variant_no_false_diags(self) -> None:
        """MOD-2: LASFile.validate's top-level twins (:3756 positional
        desync, :3791 orphan) compared exact-case, so the post-construction
        case-variant state (curves_order=['dept','GR'], logs keyed DEPT/GR)
        emitted 2 false issues although the writer emits the data
        byte-correct."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        # POST-CONSTRUCTION mutation: lowercase the order entries.
        las.curves_order = ["dept", "GR"]
        issues = las.validate(complete=True)
        assert not any("desynced" in i for i in issues), issues
        assert not any("will not emit" in i for i in issues), issues
        assert not any("NOT in curves_order" in i for i in issues), issues

        # True positive preserved: a genuine top-level reorder (different
        # curve in the position, same casing) still desyncs.
        las2 = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        las2.curves_order = ["GR", "DEPT"]
        issues2 = las2.validate(complete=True)
        assert any("desynced" in i for i in issues2), issues2

    def test_mod2_lasfile_direct_construction_case_variant_accepted(self) -> None:
        """MOD-2: the construction twins (:3196 positional, :3248 orphan
        logs) and the sibling missing/orphan checks (:3263 logs, :3288/:3300
        string_data) compared exact-case, so DIRECT construction of the
        supported case-variant state raised a false LASDataError.  The
        writer accepts the state case-insensitively — construction must
        too, and the model must preserve the data."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            logs={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        assert las.curves_order == ["dept", "GR"], "case-variant construction rejected"
        issues = las.validate(complete=True)
        assert not any("desynced" in i for i in issues), issues
        assert not any("will not emit" in i for i in issues), issues

        # string_data case-variant keys exercise the :3288/:3300 twins.
        las_s = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "TDEP"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="TDEP", unit="US/M"),
            ],
            logs={"DEPT": np.array([100.0, 101.0, 102.0])},
            string_data={"TDEP": np.array(["a", "b", "c"])},
        )
        assert las_s.curves_order == ["dept", "TDEP"], (
            "case-variant string_data construction rejected"
        )

        # True positives preserved: genuine orphans/missing still raise.
        with pytest.raises(LASDataError, match="logs contain keys not in curves_order"):
            LASFile(
                version=VersionSection(vers="2.0"),
                curves_order=["DEPT"],
                curves=[CurveDefinition(mnemonic="DEPT")],
                logs={"DEPT": np.array([1.0]), "RHOB": np.array([2.0])},
            )
        with pytest.raises(LASDataError, match="curves_order has keys not found in logs"):
            LASFile(
                version=VersionSection(vers="2.0"),
                curves_order=["DEPT", "RHOB"],
                curves=[CurveDefinition(mnemonic="DEPT"), CurveDefinition(mnemonic="RHOB")],
                logs={"DEPT": np.array([1.0])},
            )
        with pytest.raises(LASDataError, match="string_data contain keys not in curves_order"):
            LASFile(
                version=VersionSection(vers="2.0"),
                curves_order=["DEPT"],
                curves=[CurveDefinition(mnemonic="DEPT")],
                logs={"DEPT": np.array([1.0])},
                string_data={"RHOB": np.array(["x"])},
            )


# ──────────────────────────────────────────────────────────────
# MOD-3 (MEDIUM, F-04 residual): from_dict / DataSection twins of the
# case-variant state.  The pass-3 MOD-1/MOD-2 fix blessed the state but
# left DataSection.__post_init__ (orphan + positional) and the LASFile
# per-section orphan twins exact-case — direct construction with
# case-variant section_curves and the roundtrip hard-failed.
# ──────────────────────────────────────────────────────────────


class TestMOD3CaseVariantDataSectionConstruction:
    """F-04: DataSection direct construction with case-variant
    section_curves (curves_order ['dept','GR'], section_curves/data keyed
    DEPT/GR) must be accepted — the writer resolves the per-section order
    case-insensitively (pre-fix: LASDataError 'data keys not in
    curves_order: [DEPT]' at models.py:2803)."""

    def test_mod3_datasection_direct_construction_case_variant_accepted(self) -> None:
        ds = DataSection(
            name="Log1",
            curves_order=["dept", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data={
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "GR": np.array([75.0, 76.0, 77.0]),
            },
        )
        assert ds.curves_order == ["dept", "GR"], "case-variant DataSection construction rejected"
        issues = ds.validate(complete=True)
        assert not any("will pad" in issue for issue in issues), issues
        assert not any("will not emit" in issue for issue in issues), issues
        assert not any("NOT in curves_order" in issue for issue in issues), issues

        # True positive preserved: a genuinely distinct section_curves
        # mnemonic in the position still raises.
        with pytest.raises(LASDataError, match="does not match"):
            DataSection(
                name="Log1",
                curves_order=["DEPT", "GR"],
                section_curves=[
                    CurveDefinition(mnemonic="DEPT"),
                    CurveDefinition(mnemonic="RHOB"),
                ],
                data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
            )

        # True positive preserved: a genuinely orphaned data key still raises.
        with pytest.raises(LASDataError, match="data keys not in curves_order"):
            DataSection(
                name="Log1",
                curves_order=["DEPT", "GR"],
                data={"DEPT": np.array([1.0]), "XYZ": np.array([2.0])},
            )

    def test_mod3_lasfile_section_orphan_twins_case_insensitive(self) -> None:
        """F-04: the LASFile.__post_init__ per-section orphan twins must
        accept the case-variant state (pre-fix: LASDataError 'data in
        section contains keys not in curves_order') while a genuine orphan
        still raises."""
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data_sections=[
                DataSection(
                    name="Log1",
                    curves_order=["dept", "GR"],
                    data={
                        "DEPT": np.array([100.0, 101.0, 102.0]),
                        "GR": np.array([75.0, 76.0, 77.0]),
                    },
                )
            ],
        )
        assert las.curves_order == ["dept", "GR"]

        # True positive preserved: a genuinely orphaned data key added
        # POST-construction (bypassing DataSection.__post_init__) is
        # caught by the LASFile per-section orphan twin.
        ds_orphan = DataSection(
            name="Log1",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        ds_orphan.data["XYZ"] = np.array([3.0])
        with pytest.raises(LASDataError, match="contains keys not in curves_order"):
            LASFile(
                version=VersionSection(vers="3.0", wrap="NO", dlm="SPACE"),
                curves_order=["DEPT", "GR"],
                curves=[
                    CurveDefinition(mnemonic="DEPT", unit="M"),
                    CurveDefinition(mnemonic="GR", unit="GAPI"),
                ],
                data_sections=[ds_orphan],
            )

    def test_mod3_datasection_non_str_data_key_clean_error(self) -> None:
        """F-04 defensive (F-06 class): the case-variant DataSection orphan
        check's ``.upper()`` comparison must never surface a raw
        AttributeError on corrupt input — a non-str data key is rejected
        with a clean TypeError by the _GuardedDict key guard (pre-existing
        contract, unchanged by the MOD-3 case-insensitivity fix)."""
        with pytest.raises(TypeError, match="keys must be str"):
            DataSection(
                name="Log1",
                curves_order=["DEPT"],
                data={None: np.array([1.0])},  # type: ignore[dict-item]
            )


# ──────────────────────────────────────────────────────────────
# PXM-01 (MEDIUM, models side): well collision preserved via M-44
# ──────────────────────────────────────────────────────────────


class TestPXM01WellCollisionPreservedModels:
    """PXM-01 (models side): from_dict's M-44 collision-aware re-keying
    preserves BOTH well entries when two raw mnemonics resolve to the same
    canonical (the parser side, which last-wins, is fixed by the PARSER
    agent).  Regression test pins the preserved-both outcome for parser
    parity."""

    def test_from_dict_well_collision_preserves_both(self) -> None:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = LASFile.from_dict(
                {
                    "version": {"VERS": "2.0"},
                    "well": {
                        "STRT": "1",
                        "STOP": "2",
                        "STEP": "0.5",
                        "NULL": "-999",
                        "DEPT": "5000.0",
                        "DEPTH": "4999.0",
                    },
                    "curves_order": ["DEPT"],
                    "curves": [{"mnemonic": "DEPT"}],
                    "logs": {"DEPT": np.array([1.0])},
                },
                mnem_base={"DEPTH": "DEPT"},
            )
        entries = dict(las.well.entries)
        # Both values survive: DEPT keeps its own, DEPTH is re-keyed to
        # its original mnemonic instead of silently overwriting DEPT.
        assert entries["DEPT"] == "5000.0"
        assert entries["DEPTH"] == "4999.0"
        # A collision warning documents the preservation.
        assert any(
            "Preserving both" in str(w.message) or "Keeping original mnemonic" in str(w.message)
            for w in rec
        ), f"no preservation warning: {[str(w.message) for w in rec]}"


# ──────────────────────────────────────────────────────────────
# PXM-03 (MEDIUM, models side): VERS normalization in from_dict
# ──────────────────────────────────────────────────────────────


class TestPXM03VersNormalizationModels:
    """PXM-03 (models side): from_dict kept VERS verbatim ('1,2', '1.2.0')
    so the model built but write_las_file raised LASWriteError — while the
    parser normalized the same input and wrote fine.  from_dict now applies
    parser-equivalent VERS normalization (strip 3-segment → 2-segment,
    default unknown comma values to '2.0' with a warning)."""

    def _dict_for(self, vers: str) -> dict[str, Any]:
        return {
            "version": {"VERS": vers},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([1.0, 2.0])},
        }

    def test_vers_comma_defaults_to_20_and_writes(self, tmp_path: Path) -> None:
        """Pre-fix: VERS='1,2' built a model but write_las_file RAISED
        LASWriteError ('Unsupported LAS version: 1,2').  Post-fix: the
        model normalizes to '2.0' with a warning and writes, emitting
        'VERS. 2.0'."""
        with pytest.warns(UserWarning, match="Defaulting to 2.0"):
            las = LASFile.from_dict(self._dict_for("1,2"))
        assert las.version.vers == "2.0"
        out = tmp_path / "vers_comma.las"
        write_las_file(out, las)
        text = out.read_text(encoding="utf-8")
        assert "VERS.   2.0" in text
        assert "VERS.   1,2" not in text

    def test_vers_three_segment_normalizes_to_two(self, tmp_path: Path) -> None:
        """Pre-fix: VERS='1.2.0' was kept verbatim and the writer emitted
        'VERS. 1.2.0' (non-canonical).  Post-fix: '1.2.0' normalizes to
        '1.2' (parser-equivalent) and writes."""
        las = LASFile.from_dict(self._dict_for("1.2.0"))
        assert las.version.vers == "1.2"
        out = tmp_path / "vers_three.las"
        write_las_file(out, las)
        text = out.read_text(encoding="utf-8")
        assert "VERS.   1.2  " in text


# ──────────────────────────────────────────────────────────────
# MN-01 (MEDIUM): mnem_base chain VALUES not uppercased
# ──────────────────────────────────────────────────────────────


class TestMN01MnemBaseMixedCaseChainValues:
    """MN-01: resolve_mnemonic/build_mnemonic_lookup did not uppercase chain
    VALUES — a mixed-case user mnem_base ({'AK':'Dt','DT':'X'}) stopped the
    chain early and silently returned the non-canonical lowercase name.
    Values are now uppercased on storage (build_mnemonic_lookup) and during
    the chain walk (resolve_mnemonic)."""

    def test_mixed_case_chain_value_resolves_through(self) -> None:
        """Pre-fix: resolve(build({'AK':'Dt','DT':'X'}), 'AK') → 'Dt'
        (chain terminated).  Post-fix: → 'X'."""
        lk = build_mnemonic_lookup({"AK": "Dt", "DT": "X"})
        assert resolve_mnemonic(lk, "AK") == "X"

    def test_mixed_case_value_not_a_key_uppercased_terminal(self) -> None:
        """A mixed-case terminal value must be returned uppercased, not
        verbatim ('dt' → 'DT')."""
        lk = build_mnemonic_lookup({"AK": "dt", "DT": "DEPTH"})
        assert resolve_mnemonic(lk, "AK") == "DEPTH"

    def test_direct_resolve_uppercases_value_during_walk(self) -> None:
        """resolve_mnemonic itself uppercases values during the chain walk —
        even when called directly with a raw mixed-case mapping."""
        assert resolve_mnemonic({"AK": "Dt", "DT": "X"}, "AK") == "X"

    def test_shipped_mnem_base_unaffected(self) -> None:
        """Shipped MNEM_BASE has 0 mixed-case values — uppercasing is a
        no-op; canonical resolutions are unchanged."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "BK") == "BFV"
        assert resolve_mnemonic(lk, "LLD") == "BFV"


# ──────────────────────────────────────────────────────────────
# I2-19 (MEDIUM): GZ-family chain — GZ1 restored as terminal canonical
# ──────────────────────────────────────────────────────────────


class TestI219GzFamilyGZ1Terminal:
    """I2-19 + F-12: the 'GZ1':'PZ' entry re-routed 24 GZ1-targeting keys (GZ11,
    GZ110, GZ1A, ...) through GZ1 to PZ, and the former GZ2-GZ5 → PZ entries
    re-routed another 92 keys (GZ21, ГЗ2, ...) to PZ — silently renaming distinct
    gradient-probe curves to the potential-probe canonical on read with
    mnem_base=MNEM_BASE.  GZ1-GZ5 are now terminal canonicals (like GKST):
    GZ21 → GZ2, ГЗ3 → GZ3, and GZ2/GZ3/GZ4/GZ5 resolve to themselves.  The
    R-variant / deep-spacing keys (GZ1R, GZ3R1, GZ6, GZ8, ...) still route to
    OGZ; PZ keeps its own direct aliases (PZ1..PZ25, ПЗ*, OPZ, PROX*)."""  # noqa: RUF002

    def test_gz11_resolves_to_gz1(self) -> None:
        """Pre-fix: GZ11 → PZ (silent rename).  Post-fix: GZ11 → GZ1."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ11") == "GZ1"

    def test_gz1_is_terminal(self) -> None:
        """GZ1 resolves to itself — it is no longer an alias key."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ1") == "GZ1"

    def test_gz2_is_terminal_canonical(self) -> None:
        """F-12: GZ2 resolves to itself — the former GZ2 → PZ alias silently
        re-routed 25 GZ2-targeting keys (GZ21, GZ210, ...) to PZ."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ2") == "GZ2"

    def test_gz_family_intended_mapping(self) -> None:
        """The intended family mapping: GZ1-GZ5 terminal; R-variant/deep keys
        → OGZ; PZ aliases → PZ."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ1") == "GZ1"
        assert resolve_mnemonic(lk, "GZ2") == "GZ2"
        assert resolve_mnemonic(lk, "GZ3") == "GZ3"
        assert resolve_mnemonic(lk, "GZ4") == "GZ4"
        assert resolve_mnemonic(lk, "GZ5") == "GZ5"
        assert resolve_mnemonic(lk, "GZ6") == "OGZ"
        assert resolve_mnemonic(lk, "GZ8") == "OGZ"
        assert resolve_mnemonic(lk, "GZ1R") == "OGZ"
        assert resolve_mnemonic(lk, "PZ1") == "PZ"


# ──────────────────────────────────────────────────────────────
# F-12 (MEDIUM): GZ2-GZ5 chain silently renamed 92 keys to PZ —
# the I2-19 fix was incomplete (comment/test claimed "no other
# mnemonic targets them", which was false).  GZ2-GZ5 restored as
# terminal canonicals like GZ1.
# ──────────────────────────────────────────────────────────────


class TestF12Gz2Gz5TerminalCanonicals:
    """F-12: 92 keys target GZ2(25)/GZ3(31)/GZ4(24)/GZ5(12), and the former
    GZ2-GZ5 → PZ entries silently renamed all of them (GZ21, GZ210, ГЗ2, ...)
    to PZ on read with mnem_base=MNEM_BASE when no PZ curve coexisted.  The
    chain entries are removed; GZ2-GZ5 are terminal canonicals: GZ21 → GZ2,
    ГЗ3 → GZ3, GZ5 → GZ5 — no silent rename to PZ."""  # noqa: RUF002

    def test_gz21_family_resolves_to_gz2_not_pz(self) -> None:
        """Pre-fix: GZ21/GZ210 → PZ (silent rename).  Post-fix: → GZ2."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ21") == "GZ2"
        assert resolve_mnemonic(lk, "GZ210") == "GZ2"
        assert resolve_mnemonic(lk, "GZ2A") == "GZ2"

    def test_gz31_family_resolves_to_gz3_not_pz(self) -> None:
        """Pre-fix: GZ31/GZ310 → PZ.  Post-fix: → GZ3."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ31") == "GZ3"
        assert resolve_mnemonic(lk, "GZ310") == "GZ3"

    def test_gz4_gz5_families_resolve_to_own_canonical(self) -> None:
        """Pre-fix: GZ41/GZ51 → PZ.  Post-fix: → GZ4/GZ5."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "GZ41") == "GZ4"
        assert resolve_mnemonic(lk, "GZ410") == "GZ4"
        assert resolve_mnemonic(lk, "GZ51") == "GZ5"
        assert resolve_mnemonic(lk, "GZ510") == "GZ5"

    def test_cyrillic_gz_family_resolves_to_own_canonical(self) -> None:
        """Pre-fix: ГЗ2/ГЗ3вм/ГЗ4 → PZ (silent rename).  Post-fix: → GZ2/GZ3/GZ4."""  # noqa: RUF002
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "ГЗ2") == "GZ2"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗ2вм") == "GZ2"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗ3") == "GZ3"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗ3вм") == "GZ3"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗ4") == "GZ4"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗ5") == "GZ5"  # noqa: RUF001
        assert resolve_mnemonic(lk, "ГЗК3") == "GZ3"  # noqa: RUF001

    def test_no_gz_family_key_resolves_to_pz(self) -> None:
        """No GZ-family key may resolve to PZ — the gradient-probe family is
        distinct from the potential-probe canonical."""
        lk = build_mnemonic_lookup(MNEM_BASE)
        gz_keys = [k for k in lk if k.startswith("GZ") or k.startswith("ГЗ")]
        gz_to_pz = [k for k in gz_keys if lk[k] == "PZ"]
        assert gz_to_pz == [], f"GZ-family keys still resolving to PZ: {gz_to_pz}"

    def test_gz3_curve_preserved_end_to_end_without_pz(self, tmp_path: Path) -> None:
        """End-to-end: a file with a GZ3 curve and NO PZ curve must keep GZ3
        when read with mnem_base=MNEM_BASE.  Pre-fix: GZ3 → PZ silently."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS. 2.0 : CWLS LOG ASCII STANDARD\n"
            " WRAP. NO : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL. -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M : Depth\n"
            " GZ3.OHMM : Gradient probe 3\n"
            "~A DEPT GZ3\n"
            "1000.0 10.0\n"
            "1001.0 11.0\n"
        )
        path = tmp_path / "f12_gz3_no_pz.las"
        path.write_text(content, encoding="utf-8")
        data = read_las_file(str(path), mnem_base=MNEM_BASE)
        assert data["curves_order"] == ["DEPT", "GZ3"], data["curves_order"]


# ──────────────────────────────────────────────────────────────
# F-13 (MEDIUM): Cyrillic РС→SP mapped the Russian resistivity  # noqa: RUF003
# abbreviation to spontaneous potential (Р/П keyboard-adjacent  # noqa: RUF003
# typo); every other Cyrillic R-* entry maps to its R-* canonical.
# ──────────────────────────────────────────────────────────────


class TestF13CyrillicRsResistivity:
    """F-13: 'РС':'SP' was the only Cyrillic R-* entry breaking the
    R-*→R-* pattern (РД→RD, РЕЗ/РЕЗ1→RS, РП→RP, РПЗ→RZP).  РС is the
    Russian resistivity abbreviation — corrected to RS (consistent with
    РЕЗ→RS)."""  # noqa: RUF002

    def test_rs_resolves_to_resistivity_not_sp(self) -> None:
        """Pre-fix: РС → SP (silent relabel to spontaneous potential).
        Post-fix: РС → RS."""  # noqa: RUF002
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "РС") == "RS"  # noqa: RUF001

    def test_rs_consistent_with_rez_family(self) -> None:
        """РС resolves to the same canonical as the РЕЗ resistivity family."""  # noqa: RUF002
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "РС") == resolve_mnemonic(lk, "РЕЗ") == "RS"  # noqa: RUF001

    def test_no_cyrillic_r_key_resolves_to_sp(self) -> None:
        """No Cyrillic Р-* key may resolve to SP — SP is covered by the
        ПС*/СП family."""  # noqa: RUF002
        lk = build_mnemonic_lookup(MNEM_BASE)
        cyr_r = [k for k in lk if k.startswith("Р")]  # noqa: RUF001
        r_to_sp = [k for k in cyr_r if lk[k] == "SP"]
        assert r_to_sp == [], f"Cyrillic Р-* keys still resolving to SP: {r_to_sp}"  # noqa: RUF001

    def test_sp_family_untouched(self) -> None:
        """The SP family (ПС*, СП) is unaffected by the РС correction."""  # noqa: RUF002
        lk = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lk, "ПС") == "SP"
        assert resolve_mnemonic(lk, "ПСк1") == "SP"  # noqa: RUF001
        assert resolve_mnemonic(lk, "СП") == "SP"


# ──────────────────────────────────────────────────────────────
# L30-01 (LAS 3.0, MEDIUM): spec-form array {A:x} time-offset is
# dead code in the real parse flow — the parser strips {A:N} from
# descriptions (parser.py:2694-2696) before _build_spec_form_array_info
# runs, so time_offset was always None for spec-form arrays and the
# writer re-emitted plain {A} (offset lost).  The las30 side now strips
# the {A:N} marker from the synthesized description after extracting the
# offset so the writer re-emits it exactly once (from
# array_info.time_offset).  Parser-side preservation is coordinated via
# tmp/s8-las30-parser-coordination.md.
# ──────────────────────────────────────────────────────────────


class TestL3001SpecFormTimeOffset:
    """L30-01: _build_spec_form_array_info must extract the {A:N}
    offset from the (pre-strip) description AND clean the marker from
    the synthesized description so the writer does not double-emit."""

    def _pair(self) -> list[CurveDefinition]:
        return [
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:0}"),
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:5}"),
        ]

    def test_offset_extracted_and_marker_cleaned(self) -> None:
        """Pre-fix: description kept '{A:0}' → writer double-emitted
        '{A:0}  {A:0}'.  Post-fix: offset 0.0/5.0 extracted AND the
        marker is stripped so the writer emits {A:N} exactly once."""
        out = _build_spec_form_array_info(self._pair(), ["1 2"], " ")
        assert [c.mnemonic for c in out] == ["NMR[1]", "NMR[2]"]
        assert out[0].array_info is not None
        assert out[0].array_info.time_offset == 0.0
        assert out[1].array_info.time_offset == 5.0
        # Marker must be cleaned from the description (pre-fix it stayed).
        assert out[0].description == "Echo", out[0].description
        assert out[1].description == "Echo", out[1].description

    def test_markerless_description_offset_none(self) -> None:
        """A markerless description (the current parser output) yields
        offset None — the las30 side is ready for the parser-side
        coordination change to preserve the marker."""
        sc = [
            CurveDefinition(mnemonic="NMR", data_format="A", description="NMR Echo Array"),
            CurveDefinition(mnemonic="NMR", data_format="A", description="NMR Echo Array"),
        ]
        out = _build_spec_form_array_info(sc, ["1 2"], " ")
        assert out[0].array_info is not None
        assert out[0].array_info.time_offset is None
        assert out[0].description == "NMR Echo Array"

    def test_writer_emits_offset_once_from_array_info(self, tmp_path: Path) -> None:
        """End-to-end writer check: a spec-form array with time_offset in
        array_info is emitted as '{A:0}' once — not '{A:0}  {A:0}'."""
        from pylasdev.models import ArrayElementInfo

        sc = [
            CurveDefinition(
                mnemonic="NMR[1]",
                unit="ms",
                data_format="A",
                description="Echo",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0),
            ),
            CurveDefinition(
                mnemonic="NMR[2]",
                unit="ms",
                data_format="A",
                description="Echo",
                array_info=ArrayElementInfo(base_name="NMR", index=2, time_offset=5.0),
            ),
        ]
        las = LASFile()
        las.version.vers = "3.0"
        las.curves = sc
        las.curves_order = ["NMR[1]", "NMR[2]"]
        out_path = tmp_path / "l3001_out.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out_path, las)
        content = out_path.read_text(encoding="utf-8")
        # Each curve emits the marker exactly once.
        assert content.count("{A:0}") == 1, content
        assert content.count("{A:5}") == 1, content
        assert "{A:0}  {A:0}" not in content


# ──────────────────────────────────────────────────────────────
# PF-19 (L30-01 completion): F2-07 writeback must propagate the
# STRIPPED description (without the {A:N} marker) to the top-level
# las.curves entry.  Pre-fix the writeback copied mnemonic/array_info
# only, so the global curve kept the raw marker → to_dict() leaked it
# and the no-data_sections write path double-emitted
# ``NMR Echo Array {A:0}  {A:0}`` (spurious "Multiple format
# specifiers" warnings on re-read).
# ──────────────────────────────────────────────────────────────

# LAS 3.0 spec-form array fixture (repeated plain NMR mnemonics with
# {A:N} offset markers in their descriptions, followed by numeric data).
_SPEC_FORM_ARRAY_CONTENT = """~VERSION
 3.0    : LAS 3.0
 VERS.   3.0 : CWLS log ASCII Standard - VERSION 3.0
 WRAP.   NO : One line per depth step
 DLM.   COMMA : Column delimiter
~WELL
 WELL.   W1 :
 STRT.   100.0 :
 STOP.   101.0 :
 STEP.   1.0 :
 NULL.   -999.25 :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 NMR.ms       : NMR Echo Array {A:0}
 NMR.ms       : NMR Echo Array {A:5}
~A
 100,10,11
 101,12,13
"""


class TestPF19F207WritebackStrippedDescription:
    """PF-19: the F2-07 writeback must copy the stripped (marker-free)
    description from section_curves to the top-level las.curves entry,
    mirroring the existing mnemonic/array_info propagation."""

    def _parse(self) -> LASFile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LASParser().parse(_SPEC_FORM_ARRAY_CONTENT)

    def test_parse_top_level_descriptions_have_no_marker(self) -> None:
        """Pre-fix: top-level las.curves kept 'NMR Echo Array {A:0}' /
        'NMR Echo Array {A:5}' (marker NOT propagated to the writeback).
        Post-fix: descriptions are the stripped 'NMR Echo Array'."""
        las = self._parse()
        leaked = [c.description for c in las.curves if "{A:" in (c.description or "")]
        assert leaked == [], f"top-level curve descriptions leak {{A:N}}: {leaked}"
        by_mnemonic = {c.mnemonic: c.description for c in las.curves}
        assert by_mnemonic["NMR[1]"] == "NMR Echo Array", by_mnemonic
        assert by_mnemonic["NMR[2]"] == "NMR Echo Array", by_mnemonic

    def test_to_dict_no_marker_leak(self) -> None:
        """Pre-fix: to_dict()['curves'][].description contained the raw
        '{A:0}' marker.  Post-fix: no marker leaks into the dict."""
        las = self._parse()
        data = las.to_dict()
        descs = [c["description"] for c in data["curves"]]
        assert "{A:" not in "\n".join(descs), f"to_dict leaks {{A:N}}: {descs}"
        assert "NMR Echo Array" in descs, descs

    def test_no_data_sections_write_single_emission(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pre-fix: the no-data_sections write path (programmatic build
        from parsed curves) double-emitted 'NMR Echo Array {A:0}  {A:0}'
        → re-read logged 'Multiple format specifiers'.  Post-fix: each
        {A:N} marker is emitted exactly once and re-read is clean of the
        spurious marker warnings."""
        las = self._parse()
        rebuilt = LASFile()
        rebuilt.version.vers = "3.0"
        rebuilt.version.dlm = "COMMA"
        rebuilt.version.wrap = "NO"
        rebuilt.curves = list(las.curves)
        rebuilt.curves_order = list(las.curves_order)
        for name in las.logs:
            rebuilt.logs[name] = las.logs[name]
        out_path = tmp_path / "pf19_nodata.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out_path, rebuilt)
        content = out_path.read_text(encoding="utf-8")
        # Each marker emitted exactly once per array element.
        assert content.count("{A:0}") == 1, content
        assert content.count("{A:5}") == 1, content
        assert "{A:0}  {A:0}" not in content, content
        assert "{A:5}  {A:5}" not in content, content
        # Re-read must not fire the spurious marker warnings.
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            LASParser().parse(content)
        assert "Multiple format specifiers" not in caplog.text, caplog.text

    def test_standard_data_sections_write_path_unaffected(self, tmp_path: Path) -> None:
        """The standard data_sections write path (parse → write) must
        remain single-emission — the fix must not change it."""
        las = self._parse()
        out_path = tmp_path / "pf19_ds.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out_path, las)
        content = out_path.read_text(encoding="utf-8")
        assert content.count("{A:0}") == 1, content
        assert content.count("{A:5}") == 1, content
        assert "{A:0}  {A:0}" not in content, content


# ──────────────────────────────────────────────────────────────
# F-11 (LAS 3.0, MEDIUM, REGRESSION): the spec-form synthesis regex
# _SPEC_FORM_ARRAY_RE (r"\{A:(?P<offset>[-\d.]*)\}") failed on the
# official CWLS trailing-space form '{A:0 }' (whitespace before the
# closing brace — see the shipped sample_las3.0_spec.las).  The parser
# preserves the marker verbatim (parser.py L30-01) and its own
# FORMAT_SPEC_PATTERN accepts \s{0,64} before '}', but the synthesis
# regex did not → time_offset silently lost (None), the marker stayed
# in the description, and the writer emitted the doubled artifact
# 'NMR Echo Array {A:0 }  {A}' whose re-read fired the spurious
# "Multiple format specifiers" warning.  The regex now mirrors
# FORMAT_SPEC_PATTERN's whitespace tolerance.
# ──────────────────────────────────────────────────────────────


class TestF11SpecFormTrailingSpaceOffset:
    """F-11: _SPEC_FORM_ARRAY_RE must accept the whitespace-before-
    closing-brace form '{A:0 }' so the offset survives parse → write →
    re-parse with a single {A:N} emission."""

    def _pair_space(self) -> list[CurveDefinition]:
        return [
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:0 }"),
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:5 }"),
        ]

    def test_offset_extracted_and_marker_cleaned_trailing_space(self) -> None:
        """Pre-fix: '{A:0 }' failed the regex → time_offset None and the
        marker stayed in the description.  Post-fix: offset 0.0/5.0
        extracted AND the marker (with its whitespace) stripped."""
        out = _build_spec_form_array_info(self._pair_space(), ["1 2"], " ")
        assert [c.mnemonic for c in out] == ["NMR[1]", "NMR[2]"]
        assert out[0].array_info is not None
        assert out[0].array_info.time_offset == 0.0
        assert out[1].array_info.time_offset == 5.0
        assert out[0].description == "Echo", out[0].description
        assert out[1].description == "Echo", out[1].description

    def test_no_trailing_space_form_unchanged(self) -> None:
        """The existing '{A:0}' (no whitespace) form keeps working."""
        sc = [
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:0}"),
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:5}"),
        ]
        out = _build_spec_form_array_info(sc, ["1 2"], " ")
        assert out[0].array_info is not None
        assert out[0].array_info.time_offset == 0.0
        assert out[0].description == "Echo"


# LAS 3.0 spec-form array fixture using the official trailing-space
# '{A:0 }' marker form (plain repeated mnemonics — the spec-form path).
_SPEC_FORM_ARRAY_SPACE_CONTENT = """~VERSION
 3.0    : LAS 3.0
 VERS.   3.0 : CWLS log ASCII Standard - VERSION 3.0
 WRAP.   NO : One line per depth step
 DLM.   COMMA : Column delimiter
~WELL
 WELL.   W1 :
 STRT.   100.0 :
 STOP.   101.0 :
 STEP.   1.0 :
 NULL.   -999.25 :
~CURVE INFORMATION
 DEPT.M       : DEPTH  {F}
 NMR.ms       : NMR Echo Array {A:0 }
 NMR.ms       : NMR Echo Array {A:5 }
~A
 100,10,11
 101,12,13
"""


class TestF11SpecFormTrailingSpaceRoundtrip:
    """F-11 end-to-end: parse (trailing-space markers) → write → re-read
    must preserve time_offset and emit {A:N} exactly once with no
    "Multiple format specifiers" warning."""

    def _parse(self) -> LASFile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LASParser().parse(_SPEC_FORM_ARRAY_SPACE_CONTENT)

    def test_parse_extracts_time_offset_and_strips_marker(self) -> None:
        """Pre-fix: time_offset silently lost (None) and the raw marker
        stayed in the description.  Post-fix: offset 0.0/5.0 present and
        descriptions are the stripped 'NMR Echo Array'."""
        las = self._parse()
        by_mnemonic = {c.mnemonic: c for c in las.curves}
        assert by_mnemonic["NMR[1]"].array_info is not None
        assert by_mnemonic["NMR[1]"].array_info.time_offset == 0.0
        assert by_mnemonic["NMR[2]"].array_info.time_offset == 5.0
        leaked = [c.description for c in las.curves if "{A:" in (c.description or "")]
        assert leaked == [], f"descriptions leak {{A:N}}: {leaked}"
        assert by_mnemonic["NMR[1]"].description == "NMR Echo Array"

    def test_write_emits_single_marker_and_clean_reread(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pre-fix: the writer emitted 'NMR Echo Array {A:0 }  {A}' (doubled
        artifact) and re-read fired 'Multiple format specifiers'.  Post-fix:
        '{A:0}' emitted exactly once per element and re-read is clean."""
        las = self._parse()
        out_path = tmp_path / "f11_space_roundtrip.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out_path, las)
        content = out_path.read_text(encoding="utf-8")
        assert content.count("{A:0}") == 1, content
        assert content.count("{A:5}") == 1, content
        assert "{A:0 }" not in content, content
        assert "{A:0}  {A}" not in content, content
        # Re-read must not fire the spurious marker warnings.
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            LASParser().parse(content)
        assert "Multiple format specifiers" not in caplog.text, caplog.text


# ──────────────────────────────────────────────────────────────
# E-01 (LAS 3.0, HIGH): restore_tilde over-restored '_~'→'~' on EVERY
# token of every data row while the writer's M-85 '_~' escape fires only
# for FIRST-column string values starting '~'+non-letter.  Genuine
# '_~'-prefixed values in any column were silently corrupted.  Fix:
# position/type-aware restore (first column only) + writer-predicate
# check (only '_~'+non-letter restores; '_~'+letter is never writer
# output).
# ──────────────────────────────────────────────────────────────


class TestE01Las30TildeRestoreScoped:
    """E-01: the '_~' restore must fire only where the writer could have
    emitted the M-85 escape — first-column string values starting
    '~'+non-letter.  Genuine '_~' values elsewhere survive verbatim."""

    _CURVES = (
        " DEPT.M       : DEPTH  {F}\n"
        " TAG.         : TAG  {S}\n"
    )

    def _las30(self, tmp_path: Path, curves: str, data: str, name: str) -> Path:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n" + curves + "~A LOG | CURVE\n" + data
        )
        test_file = tmp_path / name
        test_file.write_text(content, encoding="utf-8")
        return test_file

    def test_non_first_column_genuine_tilde_underscore_preserved(self, tmp_path: Path) -> None:
        """A genuine '_~3D' in a NON-first column (the writer never
        escapes it there) must survive — pre-fix it was read as '~3D'."""
        test_file = self._las30(
            tmp_path,
            self._CURVES,
            "1000,_~3D\n1001,~3D\n",
            "e01_nonfirst.las",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert las.string_data["TAG"].tolist() == ["_~3D", "~3D"], (
            f"E-01: genuine '_~3D' in non-first column corrupted: "
            f"{las.string_data['TAG'].tolist()!r}"
        )

    def test_first_column_genuine_tilde_underscore_preserved(self, tmp_path: Path) -> None:
        """A first-column genuine '_~DEPT' — which the writer would NEVER
        escape (M-85 fires only for '~'+non-letter; M-28 strips the tilde
        of '~'+letter) — must survive.  Pre-fix: read as '~DEPT'."""
        curves = (
            " TAG.         : TAG  {S}\n"
            " DEPT.M       : DEPTH  {F}\n"
        )
        test_file = self._las30(
            tmp_path,
            curves,
            "_~DEPT,100\n_~3D,200\n",
            "e01_firstcol.las",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert las.string_data["TAG"].tolist() == ["_~DEPT", "~3D"], (
            f"E-01: first-column genuine '_~' values corrupted: "
            f"{las.string_data['TAG'].tolist()!r}"
        )

    def test_escape_collision_writer_roundtrip(self, tmp_path: Path) -> None:
        """ESCAPE-COLLISION residual: a section holding BOTH a genuine
        '~3D' (writer-escaped to '_~3D') and a genuine '_~3D' writes two
        byte-identical '_~3D' rows.  The escaped value MUST roundtrip
        ('~3D' → '_~3D' → '~3D'); the genuine '_~3D' is byte-identical
        and falls in the documented irreducible trade (both restore) —
        the corruption surface is limited to exactly the writer's escape
        class instead of every token of every row."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["TAG", "GR"]
        las.curves = [
            CurveDefinition(mnemonic="TAG", unit="", data_format="S", description="Tag"),
            CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F", description="Gamma"),
        ]
        las.string_data["TAG"] = np.array(["~3D", "_~3D", "plain"], dtype=object)
        las.logs["GR"] = np.array([100.0, 200.0, 300.0])
        out = tmp_path / "e01_collision.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        written = out.read_text(encoding="utf-8")
        # Both values emitted as identical '_~3D' rows (escape + verbatim).
        assert written.count("_~3D") == 2, written
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        # The writer-escaped '~3D' roundtrips; the byte-identical genuine
        # '_~3D' restores to '~3D' — the documented irreducible trade.
        assert back.string_data["TAG"].tolist() == ["~3D", "~3D", "plain"], (
            back.string_data["TAG"].tolist()
        )
        np.testing.assert_allclose(back.logs["GR"], [100.0, 200.0, 300.0])


# ──────────────────────────────────────────────────────────────
# E-21 (LAS 3.0, MEDIUM): the wrapped-layout rejection message was
# misleading — it claimed "WRAP=YES" and told users to "set WRAP to NO"
# even when the file already declared WRAP=NO (detection is content-
# based, declaration-independent).  R-5 keeps the DELIBERATE rejection
# (no LAS 3.0 wrapped reader) but the message must reflect reality.
# ──────────────────────────────────────────────────────────────


class TestE21Las30FlowingRejectionMessage:
    """E-21: a WRAP=NO-declared LAS 3.0 file with flowing layout (2+
    depth steps per line) raises the content-detected wrapped-layout
    rejection WITHOUT claiming the file declares WRAP=YES or instructing
    "set WRAP to NO"."""

    _FLOWING = (
        "~VERSION INFORMATION\n"
        " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        " DLM.    SPACE :\n"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth {F}\n"
        " GR.GAPI  :  Gamma {F}\n"
        " RHOB.K/M3:  Density {F}\n"
        "~A\n"
        "1000.0 10.0 20.0 1001.0 11.0 21.0\n"
        "1002.0 12.0 22.0 1003.0 13.0 23.0\n"
    )

    def test_wrap_no_flowing_message_is_declaration_independent(self, tmp_path: Path) -> None:
        test_file = tmp_path / "e21_flowing_no.las"
        test_file.write_text(self._FLOWING, encoding="utf-8")
        with pytest.raises(LASParseError) as exc_info:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                read_las_file_as_object(test_file)
        msg = str(exc_info.value)
        # No misleading claim about the declared WRAP value, and no
        # impossible instruction (the file ALREADY declares WRAP=NO).
        assert "WRAP=YES" not in msg, msg
        assert "set WRAP to NO" not in msg, msg
        assert "not supported by pylasdev" in msg, msg

    def test_wrap_yes_flowing_keeps_pinned_message(self, tmp_path: Path) -> None:
        """Control: when the file DOES declare WRAP=YES the pinned
        'WRAP=YES is not supported' phrasing is retained (locked by
        tests/test_reader.py:4082)."""
        test_file = tmp_path / "e21_flowing_yes.las"
        test_file.write_text(self._FLOWING.replace(" WRAP.   NO", " WRAP.   YES"), encoding="utf-8")
        with pytest.raises(LASParseError, match="WRAP=YES is not supported"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                read_las_file_as_object(test_file)


# ──────────────────────────────────────────────────────────────
# M-21 (LAS 3.0, MEDIUM): the short-row warning gate keyed on the
# DECLARED WRAP header while the LAS 1.2/2.0 twin keys on the
# CONTENT-DETECTED wrap state — a declared-WRAP=YES file whose data is
# actually non-wrapped silently null-filled short rows with ZERO
# diagnostics.
# ──────────────────────────────────────────────────────────────


class TestM21Las30ShortRowGateOnActualWrap:
    """M-21: declared WRAP=YES + actually-non-wrapped (uniform short
    rows — H-02 column-mismatch class) must emit the short-row
    diagnostic, matching the 1.2/2.0 twin."""

    def test_declared_yes_actually_nonwrapped_short_row_warns(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   YES  : WRAPPED\n"
            " DLM.    SPACE:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " GR  .GAPI :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A LOG | CURVE\n"
            "1000.0 10.0\n"
            "1001.0 12.0\n"
            "1002.0 13.0\n"
        )
        test_file = tmp_path / "m21_declared_yes.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        # Pre-fix: declared WRAP=YES suppressed the short-row count → the
        # missing RHOB values were null-filled with ZERO diagnostics.
        assert any("fewer values" in str(w.message) for w in rec), (
            f"M-21: no short-row warning, got: {[str(w.message) for w in rec]}"
        )
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(sec.data["RHOB"], [-999.25, -999.25, -999.25])


# ──────────────────────────────────────────────────────────────
# M-06 (LAS 3.0, MEDIUM): NaN/Inf tokens in a >=2-element spec-form
# group counted as "string evidence" → the channel misrouted to
# duplicate STRING curves.  The probe now mirrors the fill loop's
# NaN→null semantics: NaN/Inf are numeric-parseable.
# ──────────────────────────────────────────────────────────────


class TestM06Las30SpecFormNanNotStringEvidence:
    """M-06: NaN/Inf tokens must not flip the spec-form numeric probe —
    the fill loop null-fills them for numeric curves."""

    def test_probe_accepts_nan_inf(self) -> None:
        from pylasdev._las30_data import _spec_form_group_data_is_numeric

        assert _spec_form_group_data_is_numeric(["1.0 nan inf"], " ", [0, 1, 2]) is True
        assert _spec_form_group_data_is_numeric(["1.0 NaN -1e999"], " ", [0, 1, 2]) is True
        assert _spec_form_group_data_is_numeric(["SAND nan"], " ", [0, 1]) is False

    def test_spec_form_channel_with_nan_routes_numeric(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " NMR.ms       : NMR Echo Array {A:0}\n"
            " NMR.ms       : NMR Echo Array {A:5}\n"
            "~A LOG | CURVE\n"
            "100,nan,11\n"
            "101,12,13\n"
        )
        test_file = tmp_path / "m06_nan.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # Pre-fix: the 'nan' token flipped the probe → duplicate STRING
        # curves 'NMR'/'NMR_2'.  Post-fix: numeric array channel with the
        # NaN cell null-filled.
        assert "NMR[1]" in las.logs, list(las.logs.keys())
        assert "NMR[1]" not in las.string_data, list(las.string_data.keys())
        np.testing.assert_allclose(las.logs["NMR[1]"], [-999.25, 12.0])
        np.testing.assert_allclose(las.logs["NMR[2]"], [11.0, 13.0])


# ──────────────────────────────────────────────────────────────
# E-22 (LAS 3.0, MEDIUM): a LONE '{A:0}' spec-form array element was
# never synthesized (synthesis required >=2 members) → numeric values
# misrouted to string_data and the '{A:0} {S}' marker doubled on write.
# Single-element groups now synthesize when the member carries the
# preserved '{A:N}' marker and the data is numeric.
# ──────────────────────────────────────────────────────────────


class TestE22Las30SingleElementSpecForm:
    """E-22: lone spec-form array elements synthesize (index 1, marker
    time_offset) instead of being misrouted to string_data."""

    def test_lone_a0_channel_routes_numeric(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " NMR.ms       : NMR Echo Array {A:0}\n"
            "~A LOG | CURVE\n"
            "100,10\n"
            "101,11\n"
        )
        test_file = tmp_path / "e22_lone_a0.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        nmr = next(c for c in las.curves if c.mnemonic == "NMR[1]")
        assert nmr.array_info is not None
        assert nmr.array_info.index == 1
        assert nmr.array_info.time_offset == 0.0
        assert nmr.description == "NMR Echo Array", nmr.description
        assert "NMR[1]" in las.logs, list(las.logs.keys())
        assert "NMR[1]" not in las.string_data, list(las.string_data.keys())
        np.testing.assert_allclose(las.logs["NMR[1]"], [10.0, 11.0])
        # Writer emits the marker exactly once — no '{A:0} {S}' doubling.
        out = tmp_path / "e22_lone_a0_out.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        written = out.read_text(encoding="utf-8")
        assert written.count("{A:0}") == 1, written
        assert "{A:0}  {A" not in written, written

    def test_lone_markerless_a_curve_stays_string(self, tmp_path: Path) -> None:
        """Control: a lone plain-{A} (marker stripped by the parser) or
        marker-less A-format curve with string data stays a STRING curve
        (test-pinned F-06 behavior — no synthesis without the preserved
        '{A:N}' element marker)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " LITH.        : Lithology {A}\n"
            "~A LOG | CURVE\n"
            "100,SAND\n"
            "101,CLAY\n"
        )
        test_file = tmp_path / "e22_lone_plain_a.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert [c.mnemonic for c in las.curves] == ["DEPT", "LITH"], [
            c.mnemonic for c in las.curves
        ]
        assert las.string_data["LITH"].tolist() == ["SAND", "CLAY"]


# ──────────────────────────────────────────────────────────────
# M-16 (LAS 3.0, MEDIUM): duplicate plain-mnemonic {A} groups with
# numeric-looking data were SILENTLY reclassified to float array
# channels — a data-type mutation on external files (string_data →
# logs).  Offset-bearing '{A:N}' markers confirm genuine spec-form
# channels; a marker-less conversion (plain '{A}' markers are stripped
# by the parser) now warns loudly.
# ──────────────────────────────────────────────────────────────


class TestM16Las30MarkerlessConversionWarns:
    """M-16: marker-less numeric-looking duplicate-A conversion must not
    be silent; marker-bearing spec-form channels convert cleanly."""

    def test_markerless_numeric_duplicate_warns_and_synthesizes(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " LITH.        : Lithology Code A {A}\n"
            " LITH.        : Lithology Code A {A}\n"
            "~A LOG | CURVE\n"
            "100,1,2\n"
            "101,3,4\n"
        )
        test_file = tmp_path / "m16_lith.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        assert any("reclassified as a spec-form array channel" in str(w.message) for w in rec), (
            f"M-16: markerless conversion silent: {[str(w.message) for w in rec]}"
        )
        # The conversion itself is test-pinned (F-06 numeric probe) — the
        # fix makes it LOUD, not silent.
        assert any(c.mnemonic == "LITH[1]" for c in las.curves)
        assert "LITH[1]" in las.logs
        assert "LITH[1]" not in las.string_data

    def test_marker_bearing_spec_form_converts_without_warning(self, tmp_path: Path) -> None:
        """Control: a genuine '{A:N}'-marked spec-form channel converts
        with no M-16 warning (the preserved marker confirms it)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " NMR.ms       : NMR Echo Array {A:0}\n"
            " NMR.ms       : NMR Echo Array {A:5}\n"
            "~A LOG | CURVE\n"
            "100,10,11\n"
            "101,12,13\n"
        )
        test_file = tmp_path / "m16_nmr.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        assert not any("reclassified as a spec-form array channel" in str(w.message) for w in rec), (
            f"M-16: marker-bearing channel warned: {[str(w.message) for w in rec]}"
        )
        assert "NMR[1]" in las.logs

    def test_nonnumeric_duplicate_stays_string_no_warning(self, tmp_path: Path) -> None:
        """Control: non-numeric data never converts and never warns."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " LITH.        : Lithology Code A {A}\n"
            " LITH.        : Lithology Code A {A}\n"
            "~A LOG | CURVE\n"
            "100,SAND,SHALE\n"
            "101,CLAY,SLT\n"
        )
        test_file = tmp_path / "m16_lith_str.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        assert not any("reclassified as a spec-form array channel" in str(w.message) for w in rec), (
            f"M-16: string group warned: {[str(w.message) for w in rec]}"
        )
        assert [c.mnemonic for c in las.curves] == ["DEPT", "LITH", "LITH_2"], [
            c.mnemonic for c in las.curves
        ]
        assert las.string_data["LITH"].tolist() == ["SAND", "CLAY"]
        assert las.string_data["LITH_2"].tolist() == ["SHALE", "SLT"]


# ──────────────────────────────────────────────────────────────
# M-15 (LAS 3.0, MEDIUM): the 1-based index validation was skipped for
# SINGLE-element array groups (`len(_elements) < 2: continue`) — a lone
# 'NMR[0]' parsed silently.  Single-element groups now validate the
# index (>= 1); per-section continuation elements (section 1 NMR[1],
# section 2 NMR[2]) remain valid (M-64 lock).
# ──────────────────────────────────────────────────────────────


class TestM15Las30SingleElementIndexValidation:
    """M-15: single-element array groups must enforce the positive-index
    rule (sibling of the models-side N-01 fix)."""

    def test_lone_zero_index_array_raises(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " NMR[0].ms    : NMR Echo Array {A:5}\n"
            "~A LOG | CURVE\n"
            "100,10\n"
            "101,11\n"
        )
        test_file = tmp_path / "m15_zero.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError, match="starts at index 0"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                read_las_file_as_object(test_file)

    def test_lone_index_one_array_parses(self, tmp_path: Path) -> None:
        """Control: a lone '[1]' element is a valid single-element
        array — parses normally."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       : DEPTH  {F}\n"
            " NMR[1].ms    : NMR Echo Array {A:5}\n"
            "~A LOG | CURVE\n"
            "100,10\n"
            "101,11\n"
        )
        test_file = tmp_path / "m15_one.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert "NMR[1]" in las.logs


# ──────────────────────────────────────────────────────────────
# L30-02 (LAS 3.0, MEDIUM): DLM=COMMA + embedded comma in a {S}
# string value truncates the string AND shifts the following columns,
# silently losing genuine values as "extra columns".  csv.reader
# quote-awareness is deliberately NOT used (F2-015: the writer emits
# raw delimiter.join()); the fix is a LOUD warning naming the
# delimiter-in-string truncation mechanism — mirroring the LAS 1.2/2.0
# I2-02 fix so both paths are consistent.
# ──────────────────────────────────────────────────────────────


class TestL3002CommaEmbeddedDelimiterWarning:
    """L30-02: A DLM=COMMA LAS 3.0 file whose {S} string value contains
    an embedded comma must emit a clear warning that the value was
    truncated/lost — never a silent column shift."""

    def test_embedded_comma_in_string_warns(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " WELLS.    :  Well Name {S}\n"
            " GR  .GAPI :  Gamma {F}\n"
            "~ASCII\n"
            "1000.0,WELL A, B,10.0\n"
            "1001.0,WELL C,20.0\n"
        )
        test_file = tmp_path / "l3002_comma.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        # The truncation/loss must be loud — a warning naming the
        # delimiter-in-string mechanism (mirrors I2-02).
        assert any("delimiter" in str(w.message) and "string" in str(w.message) for w in rec), (
            f"Expected embedded-delimiter warning, got: {[str(w.message) for w in rec]}"
        )
        # The non-corrupt second row's genuine GR value is preserved.
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["GR"], [-999.25, 20.0])


# ──────────────────────────────────────────────────────────────
# M-05 (LAS 3.0, MEDIUM): DLM=COMMA grouped-thousands values were split
# by the inline comma split and NEVER recombined — "1,234.5" silently
# mis-assigned every subsequent column with ZERO diagnostics, while the
# LAS 1.2/2.0 path recombined with a loud warning (M-30).  The 3.0 fill
# loop now routes the non-space split through the shared _split_data_line
# (expected=curve_count) so both paths deliver the same loud-warning
# parity.
# ──────────────────────────────────────────────────────────────


class TestM05Las30CommaGroupedThousands:
    """M-05: LAS 3.0 DLM=COMMA files with comma-grouped thousands must
    recombine with a loud warning (parity with LAS 1.2/2.0 M-30) — never a
    silent column mis-assignment."""

    def test_equal_token_count_recombined_with_warning(self, tmp_path: Path) -> None:
        """Equal-token-count case: 3 tokens == 3 curves with a
        comma-grouped DEPT.  Pre-fix: DEPT=[1.0,5.0], GR=[234.5,60.0],
        RHOB=[50.0,0.2] with ZERO warnings.  Post-fix: recombined with a
        loud warning → DEPT=[1234.5,5.0], GR=[50.0,60.0], RHOB null-filled
        (short row)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " GR  .GAPI :  Gamma {F}\n"
            " RHOB.K/M3 :  Density\n"
            "~ASCII\n"
            "1,234.5,50\n"
            "5,60,0.2\n"
        )
        test_file = tmp_path / "m05_30_comma.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert thousands_msgs, (
            f"M-05: no thousands-separator recombination warning on LAS 3.0: "
            f"{[str(w.message) for w in caught]}"
        )
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [1234.5, 5.0])
        np.testing.assert_allclose(sec.data["GR"], [50.0, 60.0])
        np.testing.assert_allclose(sec.data["RHOB"], [-999.25, 0.2])


# ──────────────────────────────────────────────────────────────
# I2-03 (LAS 3.0, MEDIUM): WRAP=NO-declared genuinely-wrapped file
# silently misparses (column shift) instead of loud rejection.  The
# F-07 depth-line evidence rule (mirrored from data_reader DR-01) is
# inserted before the first-line-full short-circuit so a complete
# first row + depth-line evidence classifies the file as wrapped →
# the caller raises the loud "LAS 3.0 WRAP=YES is not supported"
# LASParseError per TestM05Las30WrappedRejected's documented intent.
# REFINEMENT: for n_curves==2 a 1-value row is ambiguous (a wrapped
# continuation line also carries curve_count-1 == 1 value), so the
# single-window[1]==1 arm is gated on n_curves >= 3 — a 2-curve
# string-padding file ([2,1,2]) must stay non-wrapped.
# ──────────────────────────────────────────────────────────────


class TestI203Las30WrapNoMixedWrapRejected:
    """I2-03: WRAP=NO + genuinely wrapped data (complete first row then
    depth/continuation rows) must raise loudly — never silently misparse
    with a DEPT shift.  Mirrors TestM05Las30WrappedRejected intent."""

    def test_wrap_no_complete_first_row_wrapped_raises(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " GR  .GAPI :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~LOG\n"
            "1000.0  50.0  1.0\n"  # complete first row
            "1001.0\n"  # depth line (1 value)
            "60.0  2.0\n"  # continuation (curve_count-1 values)
            "1002.0\n"  # depth line
            "70.0  3.0\n"  # continuation
        )
        test_file = tmp_path / "i203_wrap_no.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)

    def test_wrap_no_two_curve_string_padding_stays_non_wrapped(self, tmp_path: Path) -> None:
        """REFINEMENT guard: a 2-curve file with a ragged string-padding
        row ([2,1,2] window) must NOT be classified wrapped — a 1-value
        row is ambiguous when n_curves==2 (continuation lines also carry
        1 value).  Pre-fix (unconditional window[1]==1 arm) this raised
        the wrong "WRAP=YES" rejection (regression found on
        test_string_curve_null_sentinel_no_padding)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA:\n"
            "~WELL INFORMATION\n"
            " STRT.   100.0  :\n"
            " STOP.   200.0  :\n"
            " STEP.   1.0  :\n"
            " NULL.   -999.25  :\n"
            "~CURVE INFORMATION\n"
            " STR1.  :   {S}\n"
            " DEPT.  :   {F}\n"
            "~A LOG | CURVE\n"
            "hello,100\n"
            "world\n"
            ",300\n"
        )
        test_file = tmp_path / "i203_two_curve_pad.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        assert "STR1" in sec.string_data
        assert sec.string_data["STR1"].tolist() == ["hello", "world", ""]
        np.testing.assert_allclose(sec.data["DEPT"], [100.0, -999.25, 300.0])


# ──────────────────────────────────────────────────────────────
# F-02 (LAS 3.0 mirror, MEDIUM, REGRESSION): the n_curves>=3
# ``window[1] == 1`` arm of the depth-evidence gate fired for ANY
# single 1-value row after a full first row, misclassifying a ragged
# NON-wrapped nc>=3 file ([3,1,3] / [3,1,2]) as wrapped → the LAS 3.0
# caller raised the loud "LAS 3.0 WRAP=YES is not supported"
# LASParseError for a file that is NOT wrapped.  A single 1-value row
# is ragged-row evidence (graceful short-row null-fill), not
# unambiguous depth evidence — only TWO+ 1-value rows trigger the
# wrapped arm.  Byte-identical to the data_reader twin (two-path wrap
# contract), covered there by TestF02ThreeCurveShortMiddleRowNotWrapped.
# ──────────────────────────────────────────────────────────────


class TestF02Las30RaggedMiddleRowNotWrapped:
    """F-02 mirror on LAS 3.0: a ragged non-wrapped nc>=3 file with a
    single 1-value middle row must parse with graceful short-row
    null-fill (pre-fix it was wrongly rejected as WRAP=YES), while the
    genuine [3,1,2,1] wrapped shape still raises loudly."""

    _CURVES = " DEPT .M   :  Depth\n C1  .GAPI :  Curve 1\n C2  .K/M3 :  Curve 2\n"

    def _write(self, tmp_path: Path, data_lines: str, name: str) -> Path:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n" + self._CURVES + "~A LOG | CURVE\n" + data_lines
        )
        test_file = tmp_path / name
        test_file.write_text(content, encoding="utf-8")
        return test_file

    def test_las30_three_curve_short_middle_row_parses_with_null_fill(self, tmp_path: Path) -> None:
        """[3,1,3] WRAP=NO: must parse with DEPT=[100,101,102],
        C1=[50,-999.25,60], C2=[30,-999.25,40].  Pre-fix: raised the
        wrong 'LAS 3.0 WRAP=YES is not supported' LASParseError."""
        test_file = self._write(
            tmp_path,
            "100.0 50.0 30.0\n101.0\n102.0 60.0 40.0\n",
            "f02_las30_short_middle.las",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(sec.data["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(sec.data["C2"], [30.0, -999.25, 40.0])

    def test_las30_three_curve_short_middle_two_value_row_parses(self, tmp_path: Path) -> None:
        """[3,1,2] WRAP=NO: the single 1-value middle row is ragged, not
        wrapped — parses with null-fill (pre-fix: wrongly rejected)."""
        test_file = self._write(
            tmp_path,
            "100.0 50.0 30.0\n101.0\n102.0 60.0\n",
            "f02_las30_short_middle2.las",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(sec.data["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(sec.data["C2"], [30.0, -999.25, -999.25])

    def test_las30_genuine_mixed_wrap_still_raises(self, tmp_path: Path) -> None:
        """Control: the genuine [3,1,2,1] mixed-wrap shape (two 1-value
        rows) must STILL be classified wrapped → the loud rejection.  The
        gate must not regress genuine wrapped detection."""
        test_file = self._write(
            tmp_path,
            "1000.0  50.0  1.0\n1001.0\n60.0  2.0\n1002.0\n",
            "f02_las30_genuine_wrapped.las",
        )
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)

    def test_las30_mnemonic_masquerade_still_raises(self, tmp_path: Path) -> None:
        """Control: the [3,1,1] mnemonic-header masquerade (two 1-value
        rows) must STILL be classified wrapped → loud rejection."""
        test_file = self._write(
            tmp_path,
            "1000.0  50.0  1.0\n1001.0\n1002.0\n",
            "f02_las30_masquerade.las",
        )
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)


# ──────────────────────────────────────────────────────────────
# PARS-05 (LAS 3.0 path coordination, MEDIUM): non-deferred
# data-before-curves silently discarded data (0 data sections) while
# the deferred path worked — the "data-before-curves supported" claim
# (M-67/M-69) held only via deferral.  PARSER-B fixed the parser side
# (re-queue data-before-curves into the deferred buffer so it attaches
# once ~C is parsed).  This test locks the LAS 3.0 data path: the data
# must be attached, and the _las30_data.py:657 "no curves defined"
# discard must NOT be the primary data-loss path for this shape.
# ──────────────────────────────────────────────────────────────


class TestPars05Las30DataBeforeCurves:
    """PARS-05: a LAS 3.0 file whose ~A data section precedes ~CURVE
    must attach the data (matching the deferred-path behavior) — never
    discard it with 'no curves defined'."""

    def test_data_before_curves_attaches(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~A LOG | CURVE\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " GR  .GAPI :  Gamma\n"
        )
        test_file = tmp_path / "pars05_before_curves.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # Data must be attached (pre-fix: data_sections == 0, discarded).
        assert len(las.data_sections) == 1
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(sec.data["GR"], [50.0, 55.0])

    def test_no_discard_warning_for_data_before_curves(self, tmp_path: Path) -> None:
        """The discard warning must NOT fire for the data-before-curves
        shape (it is buffered and attached).  The _las30_data.py:657
        'no curves defined' discard remains only for genuinely-orphaned
        data (no curves at all)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE:\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~A LOG | CURVE\n"
            "1000.0  50.0\n"
            "1001.0  55.0\n"
            "~CURVE INFORMATION\n"
            " DEPT .M   :  Depth\n"
            " GR  .GAPI :  Gamma\n"
        )
        test_file = tmp_path / "pars05_before_curves2.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            read_las_file_as_object(test_file)
        assert not any("no curves defined" in str(w.message) for w in rec), (
            f"Unexpected discard warning: {[str(w.message) for w in rec]}"
        )


# ──────────────────────────────────────────────────────────────
# PF-18 (data_reader, HIGH, REGRESSION): the F-07 depth-line
# evidence rule's `window[1] == 1` arm was UNCONDITIONAL on the
# data_reader side, misclassifying a 2-curve LAS 1.2/2.0 WRAP=NO
# file with a short middle row (window [2,1,2]) as WRAPPED →
# silent data corruption (genuine values discarded/misaligned).
# The las30 twin gates this arm on n_curves >= 3; the data_reader
# side now mirrors it (curve_count >= 3).  For curve_count == 2 a
# 1-value row is AMBIGUOUS (a wrapped continuation line also
# carries curve_count-1 == 1 value); the >=2-one-value-rows arm
# still catches genuine 2-curve wrapped files.
# ──────────────────────────────────────────────────────────────


class TestPF18DataReaderTwoCurveWrapGate:
    """PF-18: 2-curve WRAP=NO files with a short middle row must NOT be
    classified wrapped (regression — pre-fix the unconditional
    window[1]==1 arm discarded genuine values)."""

    def test_las20_two_curve_short_middle_row_not_wrapped(self, tmp_path: Path) -> None:
        """2-curve LAS 2.0 WRAP=NO `100.0 50.0 / 101.0 / 102.0 60.0`:
        DEPT=[100,101,102], GR=[50,-999.25,60].  Pre-fix: WRAPPED=True →
        DEPT=[100,101], GR=[50,102] — the genuine 60.0 discarded and the
        depth value 102.0 swallowed into GR."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
            "100.0 50.0\n"
            "101.0\n"
            "102.0 60.0\n"
        )
        test_file = tmp_path / "pf18_las20_short_middle.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, -999.25, 60.0])

    def test_las12_two_curve_short_middle_row_not_wrapped(self, tmp_path: Path) -> None:
        """Same [2,1,2] shape on LAS 1.2 (SPACE delimiter — the LAS 1.2
        default): DEPT=[100,101,102], GR=[50,-999.25,60].  Pre-fix the
        same silent corruption occurred."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
            "100.0 50.0\n"
            "101.0\n"
            "102.0 60.0\n"
        )
        test_file = tmp_path / "pf18_las12_short_middle.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, -999.25, 60.0])

    def test_las20_two_curve_string_padding_not_wrapped(self, tmp_path: Path) -> None:
        """String-padding [2,1,2] shape (COMMA, 2 curves STR1{S}/DEPT{F},
        WRAP=NO) on LAS 2.0: STR1=['hello','world',''],
        DEPT=[100,-999.25,300].  Pre-fix: WRAPPED=True → only 2 rows,
        the genuine 300.0 DEPT value discarded."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITING CHARACTER\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            " STOP.M   300.0 : STOP DEPTH\n"
            " STEP.M   1.0 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " STR1 .   :  String {S}\n"
            " DEPT .   :  Depth {F}\n"
            "~A STR1 DEPT\n"
            "hello,100\n"
            "world\n"
            ",300\n"
        )
        test_file = tmp_path / "pf18_las20_string_padding.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert "STR1" in las.string_data
        assert las.string_data["STR1"].tolist() == ["hello", "world", ""]
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, -999.25, 300.0])

    def test_las12_two_curve_string_padding_not_wrapped(self, tmp_path: Path) -> None:
        """String-padding [2,1,2] analog on LAS 1.2.  LAS 1.2 does not
        support a DLM line (parser resets non-SPACE DLM to SPACE), so the
        string-padding shape uses SPACE-delimited rows: STR1=['hello',
        'world','bye'], DEPT=[100,-999.25,300].  Pre-fix the third row
        was lost."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            " STOP.M   300.0 : STOP DEPTH\n"
            " STEP.M   1.0 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " STR1 .   :  String {S}\n"
            " DEPT .   :  Depth {F}\n"
            "~A STR1 DEPT\n"
            "hello 100\n"
            "world\n"
            "bye 300\n"
        )
        test_file = tmp_path / "pf18_las12_string_padding.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert "STR1" in las.string_data
        assert las.string_data["STR1"].tolist() == ["hello", "world", "bye"]
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, -999.25, 300.0])

    def test_las20_wrap_yes_short_middle_row_still_wrapped(self, tmp_path: Path) -> None:
        """Control: the [2,1,2] shape with WRAP=YES declared must STILL
        be classified wrapped (declared-YES + depth evidence → wrapped;
        the gate must not regress genuine 2-curve wrapped detection)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
            "100.0 50.0\n"
            "101.0\n"
            "102.0 60.0\n"
        )
        test_file = tmp_path / "pf18_las20_wrap_yes.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Wrapped parse: DEPT holds the depth-line values only.
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 102.0])

    def test_detector_audit_windows_unchanged(self) -> None:
        """Control: the gate must not regress ANY of the 9 audit-validated
        windows (coordination note tmp/s8-las30-datareader-coordination.md).
        Only the [2,1,2] nc=2 NO case changes (that IS the fix)."""
        from pylasdev.data_reader import _detect_actual_wrap

        def lines_from_window(window: list[int]) -> list[str]:
            out = ["~A DEPT GR"]
            for n in window:
                out.append(" ".join(str(i) for i in range(1, n + 1)))
            return out

        # (window, curve_count, declared_wrap, expected_wrapped)
        shapes = [
            ([2, 1, 2], 2, "NO", False),  # FIXED: string-padding stays non-wrapped
            ([3, 1, 2, 1], 3, "NO", True),  # I2-03 mixed-wrap
            ([3, 3, 1, 1], 3, "YES", True),  # I2-04
            ([3, 3, 3, 3], 3, "YES", False),  # mislabeled WRAP=YES all-full
            ([2, 2, 2], 3, "NO", False),  # H-02 uniform short rows
            ([1, 1, 3, 3], 3, "NO", False),  # D-02 two sparse then full
            ([3, 2, 1], 3, "NO", False),  # ragged trailing 1-value row
            ([2, 1, 1, 1], 2, "YES", True),  # genuine 2-curve mixed-wrap
            ([3, 1, 1], 3, "NO", True),  # mnemonic-header masquerade
        ]
        for window, nc, decl, expected in shapes:
            got = _detect_actual_wrap(lines_from_window(window), nc, " ", declared_wrap=decl)
            assert got is expected, (
                f"window={window} nc={nc} decl={decl}: expected wrapped={expected}, got {got}"
            )
        # And the [2,1,2] shape with declared YES remains wrapped.
        assert (
            _detect_actual_wrap(lines_from_window([2, 1, 2]), 2, " ", declared_wrap="YES") is True
        )


class TestF35BareVers3NormalizedOnWrite:
    """F-35: a bare VERS '3' must be normalized to '3.0' at write time.

    The writer dispatches to the LAS 3.0 writer via is_las30
    (startswith('3')) but the reader's VERS normalization requires a
    '3.' prefix or \\d+\\.\\d+ — a bare '3' fell through to the else
    branch and downgraded the file to 2.0 on re-read, silently dropping
    every typed data_section.  Pre-fix write→read returned vers='2.0'
    with data_sections=0; post-fix the roundtrip is stable."""

    def test_bare_vers_3_roundtrips_as_3_0(self, tmp_path: Path) -> None:
        """write_las_file with version.vers='3' emits VERS. 3.0 and the
        typed data_section survives re-read."""
        las = LASFile()
        las.version = VersionSection(vers="3", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT", "GR"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.curves.append(CurveDefinition(mnemonic="GR", unit="GAPI"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "GR"],
            data={
                "DEPT": np.array([100.0, 110.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)

        out = tmp_path / "f35_vers3.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        vers_line = next((ln for ln in content.splitlines() if ln.strip().startswith("VERS.")), "")
        assert "3.0" in vers_line, f"VERS not normalized in output: {vers_line!r}"

        back = read_las_file_as_object(out)
        assert back.version.vers == "3.0", f"re-read vers: {back.version.vers!r}"
        assert len(back.data_sections) == 1, "typed data_sections dropped on re-read"
        ds = back.data_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 110.0])
        np.testing.assert_allclose(ds.data["GR"], [75.0, 80.0])

    def test_draft_version_not_normalized(self, tmp_path: Path) -> None:
        """Control: a '3.x' draft VERS (e.g. '3.1beta') is preserved
        verbatim — the documented I2F-02 draft-version roundtrip must not
        regress."""
        las = LASFile()
        las.version = VersionSection(vers="3.1beta", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0])},
        )
        las.data_sections.append(section)

        out = tmp_path / "f35_draft.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)

        content = out.read_text(encoding="utf-8")
        vers_line = next((ln for ln in content.splitlines() if ln.strip().startswith("VERS.")), "")
        assert "3.1beta" in vers_line, f"draft VERS not preserved: {vers_line!r}"


# ──────────────────────────────────────────────────────────────
# F-32 (writer): copy-back uncovered check must be
# case-insensitive, matching the case-insensitive definition and
# data-key resolution.
# ──────────────────────────────────────────────────────────────


class TestF32CopyBackCaseInsensitiveUncovered:
    """F-32: after data_sections→legacy copy-back, a case-variant
    curves_order entry ('dept') whose data is stored under the uppercase
    key ('DEPT') must NOT be reported as uncovered ("no data in 'logs'")
    — it IS emitted, and the ~A section must not be skipped."""

    def test_case_variant_order_entry_not_falsely_uncovered(self, tmp_path: Path) -> None:
        """F-32 (Path A copy-back): LAS 1.2 single data_section with a
        post-construction lowercase curves_order entry and uppercase data
        keys.  Pre-fix the exact-case uncovered check warned "no data in
        'logs'" AND the exact-case Path C gate skipped ~A — data lost.
        """
        las = LASFile()
        las.version = VersionSection(vers="1.2", wrap="NO")
        las.well["NULL"] = "-999.25"
        section = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="GAPI"),
            ],
            data={
                "DEPT": np.array([100.0, 101.0]),
                "GR": np.array([75.0, 80.0]),
            },
        )
        las.data_sections.append(section)
        # POST-CONSTRUCTION mutation: lowercase the first order entry.
        section.curves_order = ["dept", "GR"]

        out = tmp_path / "f32_copyback_case.las"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            write_las_file(out, las)
        false_uncovered = [str(w.message) for w in rec if "no data in 'logs'" in str(w.message)]
        assert false_uncovered == [], f"false uncovered warnings: {false_uncovered}"

        back = read_las_file_as_object(out)
        np.testing.assert_allclose(back.logs["DEPT"], [100.0, 101.0], err_msg="DEPT data lost")
        np.testing.assert_allclose(back.logs["GR"], [75.0, 80.0])


# ──────────────────────────────────────────────────────────────
# F-20 (parser, MEDIUM): cross-section consistency checker fired
# two spurious warnings on valid LAS 3.0 bare-~A + ~C files —
# "~A before ~LOG_DEFINITION" and "main curve definition has no
# corresponding data section".  The checker derived LOG_DEFINITION
# from _SECTION_TYPE_MAP while the resolver falls back to __MAIN__
# (H-01); the two disagreed on the same bare section.  The checker
# now mirrors the resolver's __MAIN__ fallback.
# ──────────────────────────────────────────────────────────────


class TestF20CrossSectionConsistencyNoFalsePositives:
    """F-20: A bare ~A scoped to the main ~C block must NOT produce
    data-before-definition warnings — the resolver binds it to __MAIN__."""

    def test_bare_a_after_main_c_no_spurious_warnings(self, tmp_path: Path, caplog) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " STRT.M  100.0  : START DEPTH\n"
            " STOP.M  120.0  : STOP DEPTH\n"
            " STEP.M  10.0   : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M      : DEPTH  {F}\n"
            " DT.US/M     : SONIC  {F}\n"
            "~A\n"
            "100.0 123.45\n"
            "110.0 123.55\n"
        )
        test_file = tmp_path / "f20_bare_a.las"
        test_file.write_text(content, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            las = read_las_file_as_object(test_file)
        spurious = [
            r.message
            for r in caplog.records
            if "before ~LOG_DEFINITION" in r.message
            or (
                "main curve definition" in r.message
                and "no corresponding data section" in r.message
            )
        ]
        assert spurious == [], f"F-20 spurious warnings: {spurious}"
        # The data still binds to the main curves (4 values intact).
        assert "DEPT" in las.data_sections[0].data

    def test_data_before_definition_still_warns(self, tmp_path: Path, caplog) -> None:
        """Control: a CORE_DATA section with NO CORE_DEFINITION and NO main
        ~C before it still fires the data-before-definition warning."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~CORE_DATA\n"
            " 2.5\n"
            "~CURVE INFORMATION\n"
            " RHOZ.OHMM : RESISTIVITY\n"
        )
        test_file = tmp_path / "f20_control.las"
        test_file.write_text(content, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
            LASParser().parse(content)
        data_before = [r.message for r in caplog.records if "before" in r.message]
        assert data_before, "F-20 control: expected data-before-definition warning, got none"


# ──────────────────────────────────────────────────────────────
# F-21 (parser, MEDIUM): a deferred (pre-~V) bare LOG_DATA section
# was mis-scoped when a typed data section flushed BEFORE the main
# ~C block — the replay bound the 2-column ~A to the partial curve
# list ([RHOZ]) and silently discarded the second column (GR).
# The replay now re-queues unresolved main-scope groups for the
# final replay, which binds them to the complete curve list.
# ──────────────────────────────────────────────────────────────


class TestF21DeferredLogDataRescoped:
    """F-21: pre-~V ~A data must keep ALL its columns when a typed
    data section flushes before the main ~C block."""

    def test_typed_section_flush_before_main_c_preserves_columns(self, tmp_path: Path) -> None:
        content = (
            "~A DEPT GR\n"
            "100 10\n"
            "101 11\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            "~CORE_DEFINITION\n"
            " RHOZ.OHMM : RESISTIVITY\n"
            "~CORE_DATA\n"
            " 2.5\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : 1  : DEPTH\n"
            " GR.GAPI : 2  : GAMMA\n"
        )
        test_file = tmp_path / "f21_deferred.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # The LOG_DATA section must bind to BOTH main curves — GR is NOT
        # silently discarded.
        log_sections = [ds for ds in las.data_sections if ds.section_type == "LOG_DATA"]
        assert log_sections, "expected a LOG_DATA section"
        ds = log_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 101.0], err_msg="DEPT lost")
        np.testing.assert_allclose(ds.data["GR"], [10.0, 11.0], err_msg="GR column discarded")


# ──────────────────────────────────────────────────────────────
# F-22 (parser, MEDIUM): _is_standalone_mnemonic_header sliced the
# full section curve list (up to 100K refs) BEFORE the token-count
# check, so every data line paid O(curves) allocation before being
# rejected (CPU-exhaustion DoS on attacker-controlled files).  The
# count check now runs against the range size first.
# ──────────────────────────────────────────────────────────────


class TestF22MnemonicHeaderCountCheckFirst:
    """F-22: The standalone-mnemonic-header detector must reject rows by
    token count WITHOUT slicing the curve list — and keep correct
    count semantics at scale."""

    def test_bounded_scope_count_semantics(self, tmp_path: Path) -> None:
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(
                "~VERSION INFORMATION\n"
                " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
                "~CURVE INFORMATION\n"
            )
        parser.las_file.curves = [
            CurveDefinition(mnemonic="C0", unit="M"),
            CurveDefinition(mnemonic="C1", unit="M"),
            CurveDefinition(mnemonic="C2", unit="M"),
            CurveDefinition(mnemonic="C3", unit="M"),
        ]
        parser._state.section_curve_start_idx = 1
        parser._state.section_curve_end_idx = 3  # scope = [C1, C2]
        assert parser._is_standalone_mnemonic_header("C1 C2") is True
        assert parser._is_standalone_mnemonic_header("C1") is False  # short row
        assert parser._is_standalone_mnemonic_header("C1 C2 C3") is False  # long row
        assert parser._is_standalone_mnemonic_header("C1 X2") is False  # non-mnemonic

    def test_100k_curve_scope_short_row_rejected_fast(self, tmp_path: Path) -> None:
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(
                "~VERSION INFORMATION\n"
                " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
                "~CURVE INFORMATION\n"
            )
        parser.las_file.curves = [
            CurveDefinition(mnemonic=f"C{i}", unit="M") for i in range(100_000)
        ]
        parser._state.section_curve_start_idx = 0
        parser._state.section_curve_end_idx = None  # unbounded -> all 100K curves
        line = " ".join(f"D{i}" for i in range(10))
        # 100 calls must complete quickly — pre-fix this sliced 100K
        # CurveDefinition refs per call (~29ms total).
        start = time.perf_counter()
        for _ in range(100):
            assert parser._is_standalone_mnemonic_header(line) is False
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"F-22: 100 reject calls took {elapsed:.2f}s"


# ──────────────────────────────────────────────────────────────
# F-23 (parser, MEDIUM): lasio-convention header line shapes were
# silently dropped — (a) no-period "MNEM : VALUE" lines (lasio
# name_missing_period) and (b) colon-in-unit lines ("TIME.hh:mm").
# Both now parse; a multi-word no-period mnemonic ("HOLE DIA") stays
# dropped because the CurveDefinition/ParameterEntry models reject
# spaces.  M10 extended the same drop-with-warning to the WELL path:
# a multi-word no-period well key is also dropped (storing it made
# read→write raise LASWriteError — the writer's N-I-19 validation
# cannot roundtrip a space-containing well key).
# ──────────────────────────────────────────────────────────────


class TestF23LasioLineShapes:
    """F-23: lasio-convention line shapes parse instead of being dropped."""

    def test_no_period_well_line_parsed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """F-23 (a) + M10: a single-word no-period well line parses
        (lasio missing-period convention); a MULTI-WORD no-period mnemonic
        ("HOLE DIA") cannot be represented by WellSection — the writer's
        N-I-19 key validation rejects embedded spaces — so it is dropped
        with a warning instead of stored (storing it made read→write raise
        LASWriteError, M10)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " STRT.M  100.0  : START DEPTH\n"
            " LOC : ACME:OIL\n"
            " HOLE DIA : 85.7\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "f23_well.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with caplog.at_level(logging.WARNING, logger="pylasdev.parser"):
                las = read_las_file_as_object(test_file)
        # Single-word no-period mnemonic parses (F-23 (a) intent).
        assert las.well["LOC"] == "ACME:OIL", "F-23: no-period well line dropped"
        # Multi-word no-period mnemonic is dropped with a warning (M10 —
        # the writer cannot roundtrip a space-containing well key).
        assert "HOLE DIA" not in las.well.entries, "M10: multi-word well key stored"
        assert any("Non-matching ~W line" in r.message for r in caplog.records), (
            "M10: multi-word no-period well line should be dropped with a warning"
        )

    def test_colon_in_unit_parameter_parsed(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~PARAMETER INFORMATION\n"
            " TIME.hh:mm 23:15 21-JAN-2001 : Time Logger\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "f23_param.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        params = {p.mnemonic: p for p in las.parameters}
        assert "TIME" in params, "F-23: colon-in-unit parameter line dropped"
        assert params["TIME"].value == "23:15 21-JAN-2001", (
            f"F-23: parameter value corrupted: {params['TIME'].value!r}"
        )

    def test_no_period_single_word_curve_parsed(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR : GAMMA\n"
            "~A DEPT GR\n"
            "1000.0  50.0\n"
        )
        test_file = tmp_path / "f23_curve.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        mnems = [c.mnemonic for c in las.curves]
        assert "GR" in mnems, f"F-23: no-period single-word curve dropped: {mnems}"

    def test_no_period_multi_word_curve_dropped_not_crash(self, tmp_path: Path) -> None:
        """A multi-word missing-period mnemonic ("HOLE DIA") cannot be a
        CurveDefinition (models reject embedded spaces); the line must be
        dropped with the file still parsing — not crash the whole file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " HOLE DIA : 85.7\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "f23_curve_multi.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert [c.mnemonic for c in las.curves] == ["DEPT"]


# ──────────────────────────────────────────────────────────────
# F-24 (parser, MEDIUM): the pre-scan mnemonic-header skip had NO
# token-count guard, so a PARTIAL header row ("DEPT GR" with 3
# declared curves) was skipped by the pre-scan but consumed by the
# reader — data_line_count diverged from reader consumption.  The
# pre-scan now mirrors the reader's clauses verbatim (count
# equality + all-string exclusion) using a dedicated distinct-curve
# counter (NOT len(curve_mnems), which over-counts mnem_base
# aliases).
#
# M12 (data_reader, MEDIUM — Stage 9): the F-24 headline defect
# (phantom all-null first row + data shift) was UNFIXED: the reader
# side still treated the partial mnemonic header as a DATA row
# (_is_mnemonic_header_row required len(values) == curve_count).
# The reader now recognizes a partial all-mnemonic row (2..curve_count
# tokens, every token a declared mnemonic) as a header, so the row is
# skipped and no phantom row / shift occurs.
# ──────────────────────────────────────────────────────────────


class TestF24PreScanPartialHeaderParity:
    """F-24/M12: a PARTIAL mnemonic header ("DEPT GR" with 3 declared
    curves) is a column header, not data — the reader must skip it so
    no phantom all-null first row is created and no value shifts.  A
    FULL header is skipped by both pre-scan and reader, and mnem_base
    aliases do not over-count."""

    def test_partial_mnemonic_header_no_phantom_row(self, tmp_path: Path) -> None:
        """M12: full reader path — DEPT reads [1000,1001,1002], NOT
        [-999.25,1000,1001,1002].  Pre-fix, the partial header "DEPT GR"
        was consumed as a data row (DEPT failed to convert → null),
        producing the phantom all-null first row and shifting every
        value by one column."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " GR.GAPI  : GAMMA\n"
            " RHOB.K/M3 : DENSITY\n"
            "~A DEPT GR DT\n"
            "DEPT GR\n"
            "1000 10 50\n"
            "1001 11 51\n"
            "1002 12 52\n"
        )
        test_file = tmp_path / "m12_partial_header.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # No phantom all-null first row, no shift — values start at 1000.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0, 12.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 51.0, 52.0])

    def test_partial_mnemonic_header_no_phantom_row_wrapped(self, tmp_path: Path) -> None:
        """M12: the partial-mnemonic-header skip also applies in WRAP=YES
        mode (the _read_wrapped call site) — the "DEPT GR" row below ~A is
        a header, not a depth line, so the depth steps start at 1000.0
        with no phantom all-null first step and no shift."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " GR.GAPI  : GAMMA\n"
            " RHOB.K/M3 : DENSITY\n"
            "~A\n"
            "DEPT GR\n"
            "1000.0\n"
            "10.0\n"
            "50.0\n"
            "1001.0\n"
            "11.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "m12_partial_header_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 51.0])

    def test_full_mnemonic_header_still_skipped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " GR.GAPI  : GAMMA\n"
            "~A\n"
            "DEPT GR\n"
            "1000.0 50.0\n"
            "1001.0 55.0\n"
        )
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(content)
        # Full-width header is a header on BOTH sides — still skipped.
        assert parser.data_line_count == 2, (
            f"F-24: full header should be skipped, data_line_count={parser.data_line_count}"
        )

    def test_mnem_base_alias_header_does_not_overcount(self, tmp_path: Path) -> None:
        """The count guard must use the distinct-curve counter, not
        len(curve_mnems) — LLD/LLS aliases resolve to BFV and inflate the
        alias set (4 entries for 3 curves), which would mis-count the
        FULL raw-vendor header row and produce a pre-scan/reader
        divergence (spurious overcount)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " LLD.OHMM : LLD\n"
            " LLS.OHMM : LLS\n"
            "~A DEPT LLD LLS\n"
            "DEPT LLD LLS\n"
            "1000.0 15.0 16.0\n"
            "1001.0 15.5 16.5\n"
        )
        parser = LASParser()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            parser.parse(content)
        # "DEPT LLD LLS" is a FULL raw-vendor header (3 tokens == 3 curves)
        # — skipped by BOTH the pre-scan and the reader, so the pre-scan
        # counts exactly the 2 numeric rows (no overcount divergence).
        assert parser.data_line_count == 2, (
            f"F-24: mnem_base header mis-counted, data_line_count={parser.data_line_count}"
        )
        overcount = [str(w.message) for w in rec if "Pre-scan overcount" in str(w.message)]
        assert overcount == [], f"F-24: spurious pre-scan overcount: {overcount}"


# ──────────────────────────────────────────────────────────────
# DR-M1 (parser, MEDIUM, Stage 10): M12 residual — the reader skips a
# PARTIAL mnemonic header (2..curve_count, data_reader.py:929) but the
# pre-scan mirror used strict len(_tokens) == curve_def_count
# (parser.py:1483).  A valid LAS 1.2/2.0 file whose ~A section begins
# with a partial all-mnemonic header row ("DEPT GR" with 3 declared
# curves) was counted as a data line by the pre-scan but skipped by the
# reader → spurious "Pre-scan overcount" warning + F36 trim.  The mirror
# now uses the reader's 2..curve_def_count clause restricted to the
# section's first line(s) (count == 0 — the per-block analog of the
# reader's current_line == 0 gate, DR-M3 coordination).
# ──────────────────────────────────────────────────────────────


class TestS10DRM1PreScanPartialHeaderNoOvercount:
    """DR-M1: the pre-scan mirror must skip the section's FIRST-line
    partial mnemonic header exactly like the reader (2..curve_count
    clause + first-line-of-section restriction) — no spurious
    'Pre-scan overcount' warning on a valid file."""

    def test_partial_header_no_overcount_warning(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1002.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            "DEPT : DEPTH : m\n"
            "GR : GAMMA RAY : API\n"
            "RHOB : BULK DENSITY : g/cm3\n"
            "~A\n"
            "DEPT GR\n"
            "1000.0 10.0 2.50\n"
            "1001.0 11.0 2.51\n"
            "1002.0 12.0 2.52\n"
        )
        test_file = tmp_path / "drm1_partial_header.las"
        test_file.write_text(content, encoding="utf-8")
        # Pre-fix: pre-scan counted the partial header as a 4th data line
        # (strict equality failed to skip it) → data_line_count=4 vs the
        # reader's 3 consumed rows → "Pre-scan overcount" warning + F36 trim.
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(content)
        assert parser.data_line_count == 3, (
            f"DR-M1: pre-scan must skip the partial header, "
            f"data_line_count={parser.data_line_count}"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"DR-M1: spurious pre-scan overcount: {[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0, 12.0])
        np.testing.assert_allclose(las.logs["RHOB"], [2.5, 2.51, 2.52])

    def test_mid_section_all_mnemonic_row_still_counted(self, tmp_path: Path) -> None:
        """DR-M1/DR-M3 coordination: the first-line-of-section restriction
        means a MID-section all-mnemonic row ('GR ZONE') is DATA, not a
        header — the pre-scan counts it and the reader keeps it.  An
        over-relaxed pre-scan (partial clause WITHOUT the first-line gate)
        would undercount (2 vs 3) and diverge from the reader."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " ZONE.    :  Zone {S}\n"
            "~A\n"
            "1000.0 10.0 SHALE\n"
            "GR ZONE\n"
            "1002.0 12.0 SAND\n"
        )
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(content)
        assert parser.data_line_count == 3, (
            f"DR-M1: mid-section all-mnemonic row must be counted as data, "
            f"data_line_count={parser.data_line_count}"
        )


# ──────────────────────────────────────────────────────────────
# DR-M2 (parser, MEDIUM, Stage 10): M12 asymmetric — the LAS 3.0 twin
# of the M12 partial-header defect.  _is_standalone_mnemonic_header
# still required len(tokens) == section_count, so a LAS 3.0 file whose
# ~A section begins with a PARTIAL all-mnemonic header ("DEPT,GR" with
# 3 declared curves) consumed the header as data → phantom all-null
# first row + data shift (DEPT=[-999.25,1000,1001,1002]).  The detector
# now accepts 2..section_count tokens and is restricted to the section's
# first line(s) at both call sites (accumulation + deferred replay),
# mirroring the LAS 1.2/2.0 reader (DR-M1/DR-M3 coordination).
# ──────────────────────────────────────────────────────────────


class TestS10DRM2Las30PartialHeaderNoPhantom:
    """DR-M2: the LAS 3.0 standalone-mnemonic-header detector must accept
    partial headers (2..section_count) and apply them only to the section's
    first line(s) — no phantom all-null first row, no value shift."""

    def test_partial_header_no_phantom_row(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITER\n"
            "~WELL INFORMATION\n"
            " NULL    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            "DEPT : DEPTH : m\n"
            "GR : GAMMA RAY : API\n"
            "RHOB : BULK DENSITY : g/cm3\n"
            "~A LOG\n"
            "DEPT,GR\n"
            "1000.0,10,50\n"
            "1001.0,11,51\n"
            "1002.0,12,52\n"
        )
        test_file = tmp_path / "drm2_las30_partial.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # No phantom all-null first row, no shift — values start at 1000.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0, 12.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 51.0, 52.0])

    def test_mid_section_all_mnemonic_row_not_skipped(self, tmp_path: Path) -> None:
        """DR-M2/DR-M3 coordination: the header-skip applies only to the
        section's FIRST line(s) — a MID-section all-mnemonic row is DATA.
        Skipping it would silently drop a row and shift the remaining
        values (the exact DR-M3 false-positive class)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA : DELIMITER\n"
            "~WELL INFORMATION\n"
            " NULL    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            "DEPT : DEPTH : m\n"
            "GR : GAMMA RAY : API\n"
            "RHOB : BULK DENSITY : g/cm3\n"
            "~A LOG\n"
            "1000.0,10,50\n"
            "GR,RHOB\n"
            "1002.0,12,52\n"
        )
        test_file = tmp_path / "drm2_las30_midsection.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # All 3 rows kept (the mid-section row is data, not a header):
        # its mnemonic tokens fail float conversion → null fill.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, -999.25, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, -999.25, 12.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, -999.25, 52.0])


# ──────────────────────────────────────────────────────────────
# PSR-1 (parser, MEDIUM, Stage 11): DR-M2 regression-on-fix — the
# partial-header relaxation lowered the LAS 3.0 gate from strict
# equality to ``len(tokens) < 2 or len(tokens) > section_count``,
# which BROKE a section with EXACTLY ONE curve: a standalone 1-token
# mnemonic header row ('DEPT') fails the ``< 2`` gate and is consumed
# as data → phantom all-null -999.25 first row + value shift.  HEAD's
# strict equality (1==1) handled it correctly.  The lower bound is now
# ``min(2, section_count)`` so a single-curve section still recognizes
# its 1-token header, while multi-curve sections keep the 2-token
# minimum (M-02); the F-19 all-string exclusion still protects
# single-curve STRING sections from mnemonic-coincident data rows.
# The data_reader.py:939 twin (LAS 1.2/2.0) is fixed in the same pass.
# ──────────────────────────────────────────────────────────────


class TestS11PSR1SingleCurveHeaderNoPhantom:
    """PSR-1: a single-curve LAS 3.0 section with a standalone 1-token
    mnemonic header row directly below ~A must parse the header as a
    header — no phantom all-null first row, no value shift."""

    def test_las30_single_curve_standalone_header_no_phantom_row(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~LOG\n"
            "DEPT\n"
            "1670.0\n"
            "1669.0\n"
            "1668.0\n"
        )
        test_file = tmp_path / "psr1_las30_single_curve.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # No phantom all-null first row, no shift — values start at 1670.
        np.testing.assert_allclose(las.logs["DEPT"], [1670.0, 1669.0, 1668.0])

    def test_las30_single_curve_all_string_mnemonic_row_preserved(self, tmp_path: Path) -> None:
        """F-19 interaction pin: a single-curve STRING section whose data
        row coincides with the curve mnemonic must NOT be skipped as a
        header — the min(2, section_count) lower bound lets the 1-token row
        through the count gate, and the all-string exclusion must still
        protect it."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " LITH.    :  Lithology {S}\n"
            "~LOG\n"
            "LITH\n"
            "SHALE\n"
            "SAND\n"
        )
        test_file = tmp_path / "psr1_las30_single_curve_string.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_array_equal(sec.string_data["LITH"], np.array(["LITH", "SHALE", "SAND"]))


# ──────────────────────────────────────────────────────────────
# DR-M3 (data_reader, MEDIUM — Stage 10): the M12 partial-header
# relaxation applied the header predicate to EVERY data line, so
# mid-section all-mnemonic rows were misclassified as headers:
#   - non-wrapped: a ragged "GR ZONE" row was silently dropped
#     (real data rows lost),
#   - wrapped: a packed continuation row "LITH ZONE" was skipped
#     without resetting the depth-line state machine (silent
#     column shift — depth leaked into the string column).
# The header check is now restricted to the FIRST line(s) of the
# section (current_line == 0 / total_elements == 0).  These tests
# pin: mid-section mnemonic rows are DATA, while the section-first
# partial header is STILL skipped (no phantom row).
# ──────────────────────────────────────────────────────────────


class TestDRM3MidSectionMnemonicRows:
    """DR-M3: mid-section all-mnemonic rows are data, not headers.

    A standalone mnemonic header can only legitimately appear at the
    top of the ~A section.  Applying the M12 partial-header predicate
    (2..curve_count tokens, every token declared) to every line caused
    silent data loss (non-wrapped) and column shift (wrapped)."""

    def test_mid_section_mnemonic_row_not_dropped_nonwrapped(self, tmp_path: Path) -> None:
        """Non-wrapped: the mid-section ragged "GR ZONE" row must NOT be
        dropped as a header.  Pre-fix it was silently ``continue``d —
        DEPT lost the 1001.0 step ([1000,1002]) and the ZONE string
        values were all lost.  Post-fix all three rows are consumed and
        the final row's values land in the correct slots (no shift)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1002.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " ZONE.    :  Zone {S}\n"
            "~A\n"
            "1000.0 10.0 SHALE\n"
            "GR ZONE\n"
            "1002.0 12.0 SAND\n"
        )
        test_file = tmp_path / "drm3_midsection_nonwrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        # All 3 rows consumed — the mid-section row is NOT silently dropped
        # (pre-fix DEPT/GR shrank to 2 entries and ZONE lost every value).
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, -999.25, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, -999.25, 12.0])
        np.testing.assert_array_equal(las.string_data["ZONE"], np.array(["SHALE", "", "SAND"]))
        # The reader consumes exactly as many rows as the pre-scan counts
        # (the ragged row is data on BOTH sides) — no spurious divergence.
        assert not any("overcount" in str(w.message).lower() for w in rec), [
            str(w.message) for w in rec
        ]

    def test_mid_section_mnemonic_row_not_dropped_wrapped(self, tmp_path: Path) -> None:
        """Wrapped: the mid-section packed continuation row "LITH ZONE"
        (both tokens declared string-curve mnemonics) must NOT be treated
        as a header.  Pre-fix it was skipped WITHOUT resetting the
        depth_line/counter state machine — DEPT lost the 1002.0 step and
        the 1002.0 depth value leaked into the LITH string column.
        Post-fix all three depth steps are present and no depth value
        leaks into a string curve."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1002.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            " ZONE.    :  Zone {S}\n"
            "~A\n"
            "1000.0\n"
            "SHALE SAND\n"
            "1001.0\n"
            "LITH ZONE\n"
            "1002.0\n"
            "CLAY SILT\n"
        )
        test_file = tmp_path / "drm3_midsection_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # All three depth steps present — no step lost, no depth leak.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_array_equal(las.string_data["LITH"], np.array(["SHALE", "LITH", "CLAY"]))
        np.testing.assert_array_equal(las.string_data["ZONE"], np.array(["SAND", "ZONE", "SILT"]))

    def test_first_line_header_skipped_mid_section_row_kept(self, tmp_path: Path) -> None:
        """Single-file discriminator: the section-FIRST partial header
        ("DEPT GR") is still skipped (M12 phantom-row fix intact), while
        a MID-SECTION all-mnemonic row ("GR ZONE") is treated as DATA
        (DR-M3).  One predicate, two positions, opposite outcomes — this
        fails if the first-row header skip regresses OR if a mid-section
        mnemonic row is (re)misclassified as a header."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1002.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " ZONE.    :  Zone {S}\n"
            "~A\n"
            "DEPT GR\n"
            "1000.0 10.0 SHALE\n"
            "GR ZONE\n"
            "1002.0 12.0 SAND\n"
        )
        test_file = tmp_path / "drm3_discriminator.las"
        test_file.write_text(content, encoding="utf-8")
        # Warnings suppressed: the pre-scan mirror (parser.py) counts the
        # first-line partial header differently across DR-M1 fix states;
        # the data outcome depends only on data_reader.py.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # First-line "DEPT GR" skipped (no phantom all-null first row);
        # mid-section "GR ZONE" kept as data (3 rows, final row unshifted).
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, -999.25, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, -999.25, 12.0])
        np.testing.assert_array_equal(las.string_data["ZONE"], np.array(["SHALE", "", "SAND"]))


# ──────────────────────────────────────────────────────────────
# PSR-1 (data_reader, MEDIUM, Stage 11): DR-M2 regression-on-fix — the
# `< 2` token-count gate rejected a SINGLE-curve section's 1-token
# standalone mnemonic header row ("~A\nDEPT\n1670.0\n1669.0\n1668.0"),
# consuming it as data → phantom all-null -999.25 first row + value
# shift (adversarially verified vs HEAD 82cadce on both sides).  The
# lower bound is now min(2, curve_count): a single-curve NUMERIC section
# still recognizes its 1-token header; the all-string exclusion keeps
# single-curve STRING sections safe (M-03/F-19).
# ──────────────────────────────────────────────────────────────


class TestS11SingleCurveStandaloneHeader:
    """PSR-1: a single-curve section with a standalone mnemonic header
    row directly below ~A is a header, not data — no phantom all-null
    first row, no value shift (DR-M2 regression-on-fix).

    F-01 (Stage 12): the LAS 1.2/2.0 pre-scan mirror (parser.py:1507)
    now carries the same min(2, curve_def_count) lower bound as the
    reader (data_reader.py:958, min(2, curve_count)) — the 1-token
    header is skipped by BOTH sides, so these valid single-curve files
    parse with NO spurious "Pre-scan overcount" warning.  Asserting
    ``assert not overcount`` below pins the verbatim-mirror contract
    (do NOT suppress warnings in these tests)."""

    _LAS20_CONTENT = (
        "~VERSION INFORMATION\n"
        " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   {wrap}  : {wrap_desc}\n"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        "~A\n"
        "DEPT\n"
        "1670.0\n"
        "1669.0\n"
        "1668.0\n"
    )

    def test_las20_single_curve_standalone_header_no_phantom_row(self, tmp_path: Path) -> None:
        """PSR-1: non-wrapped single-curve LAS 2.0 — the "DEPT" header
        row is skipped; DEPT reads [1670.0, 1669.0, 1668.0], NOT
        [-999.25, 1670.0, 1669.0, 1668.0] (pre-fix phantom + shift)."""
        content = self._LAS20_CONTENT.format(wrap="NO", wrap_desc="ONE LINE PER DEPTH STEP")
        test_file = tmp_path / "psr1_single_curve.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"F-01: spurious pre-scan overcount on single-curve header: "
            f"{[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1670.0, 1669.0, 1668.0])

    def test_las12_single_curve_standalone_header_no_overcount(self, tmp_path: Path) -> None:
        """F-01: LAS 1.2 twin of the single-curve standalone-header case.
        The pre-scan mirror must skip the 1-token "DEPT" header for
        LAS 1.2 files too (same reader gate, data_reader.py:958) — no
        spurious 'Pre-scan overcount' warning on a valid file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A\n"
            "DEPT\n"
            "1670.0\n"
            "1669.0\n"
            "1668.0\n"
        )
        test_file = tmp_path / "f01_single_curve_las12.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"F-01: spurious pre-scan overcount on LAS 1.2 single-curve header: "
            f"{[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1670.0, 1669.0, 1668.0])

    def test_las20_single_curve_standalone_header_wrapped_no_phantom_row(
        self, tmp_path: Path
    ) -> None:
        """PSR-1: WRAP=YES single-curve LAS 2.0 — the _read_wrapped call
        site (:1504) must skip the same 1-token header; no phantom row."""
        content = self._LAS20_CONTENT.format(wrap="YES", wrap_desc="MULTIPLE LINES PER DEPTH STEP")
        test_file = tmp_path / "psr1_single_curve_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"F-01: spurious pre-scan overcount on WRAP=YES single-curve header: "
            f"{[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1670.0, 1669.0, 1668.0])

    def test_las20_single_curve_string_section_mnemonic_row_preserved(self, tmp_path: Path) -> None:
        """PSR-1/M-03: a single-curve all-STRING section keeps a
        mnemonic-coincident VALUE as data — min(2, 1) must not turn
        string sections into header droppers (the all-string exclusion
        handles them)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " ZONE.    :  Zone {S}\n"
            "~A\n"
            "ZONE\n"
            "SAND\n"
            "SHALE\n"
        )
        test_file = tmp_path / "psr1_single_curve_string.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"F-01: pre-scan must not overcount an all-string section: "
            f"{[str(w.message) for w in overcount]}"
        )
        # The first "ZONE" row is string data, NOT a dropped header.
        np.testing.assert_array_equal(
            data["string_data"]["ZONE"], np.array(["ZONE", "SAND", "SHALE"])
        )


# ──────────────────────────────────────────────────────────────
# F-25 (parser, MEDIUM): _desanitize_las_value unescaped "_#" ANYWHERE
# whitespace-preceded, but the writer only escapes value-start "#".
# Internal " _#" content ("ACME _#Oil Corp") was corrupted to
# "ACME #Oil Corp" on write→read.  The unescape is now scoped to
# leading-whitespace positions only.
# ──────────────────────────────────────────────────────────────


class TestF25InternalHashPreserved:
    """F-25: internal " _#" content in header values survives write→read;
    the leading-whitespace writer escape is still restored."""

    def test_internal_hash_roundtrips_unchanged(self, tmp_path: Path) -> None:
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["NULL"] = "-999.25"
        las.well["WELL"] = "ACME _#Oil Corp"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        out = tmp_path / "f25_hash.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        assert back.well["WELL"] == "ACME _#Oil Corp", (
            f"F-25: internal _# corrupted to {back.well['WELL']!r}"
        )

    def test_leading_whitespace_escape_still_restored(self) -> None:
        """Direct pin of the _desanitize_las_value scope: the writer's
        leading-whitespace "_#" escape is still restored, while internal
        " _#" content the writer never escapes is preserved."""
        from pylasdev.parser import _desanitize_las_value

        assert _desanitize_las_value(" _#comment") == " #comment", (
            "F-25: leading-whitespace writer escape not restored"
        )
        assert _desanitize_las_value(" \t _#comment") == " \t #comment", (
            "F-25: leading-whitespace writer escape not restored (tab)"
        )
        assert _desanitize_las_value("ACME _#Oil Corp") == "ACME _#Oil Corp", (
            "F-25: internal _# content corrupted"
        )
        assert _desanitize_las_value("_#comment") == "#comment", (
            "F-25: value-start escape not restored"
        )

    def test_internal_hash_preserved_dlm_comma_string_data(self, tmp_path: Path) -> None:
        """M11: F-25's root-cause fix (line-start-scoped "_#" unescape)
        must apply to the data_reader COPY of _desanitize_las_value —
        the flagship LAS 1.2/2.0 string-data path.  An EXTERNAL LAS 2.0
        DLM=COMMA file whose {S} string curve value contains an internal
        " _#" (the writer never escapes mid-value "_#") must read back
        unchanged via read_las_file_as_object.  Pre-fix, the data_reader
        copy blanket-unescaped any whitespace-preceded "_#" and corrupted
        "ACME _#Oil Corp" → "ACME #Oil Corp"."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " NAME.    : WELL NAME {S}\n"
            "~A DEPT NAME\n"
            "100.0,ACME _#Oil Corp\n"
            "101.0,Other Well\n"
        )
        test_file = tmp_path / "m11_internal_hash.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(
            las.string_data["NAME"],
            np.array(["ACME _#Oil Corp", "Other Well"], dtype=object),
        )


# ──────────────────────────────────────────────────────────────
# F-34 (parser, MEDIUM): original_mnemonic was cleared for CASE-ONLY
# differences ('dept' → normalized 'DEPT'), so the writer re-emitted
# the canonical casing instead of the file's.  The parser now keeps
# original_mnemonic whenever the file casing differs from the
# canonical mnemonic in ANY way (case or mnem_base), clearing it
# only when byte-identical.
# ──────────────────────────────────────────────────────────────


class TestF34OriginalMnemonicCasePreserved:
    """F-34: case-only original mnemonics are preserved and re-emitted."""

    def test_lowercase_curve_original_mnemonic_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " dept.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
            "1000.0  50.0\n"
        )
        test_file = tmp_path / "f34_case.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        by_mnem = {c.mnemonic: c for c in las.curves}
        assert by_mnem["DEPT"].original_mnemonic == "dept", (
            f"F-34: case-only original_mnemonic dropped: {by_mnem['DEPT'].original_mnemonic!r}"
        )
        # Canonical-case curves still get no original_mnemonic.
        assert by_mnem["GR"].original_mnemonic == ""

    def test_write_emits_original_casing(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " dept.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
            "1000.0  50.0\n"
        )
        test_file = tmp_path / "f34_case_src.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        out = tmp_path / "f34_case_out.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        written = Path(out).read_text()
        curve_block = written.split("~CURVE INFORMATION", 1)[1].split("~", 1)[0]
        assert any("dept.M" in line for line in curve_block.splitlines()), (
            f"F-34: writer did not emit original casing 'dept': {curve_block!r}"
        )


# ──────────────────────────────────────────────────────────────
# M7 (parser, MEDIUM, Stage 9): F-21 re-queued ONLY the bare deferred
# LOG_DATA variant (curve_end is None).  The "~A | CURVE" piped variant
# (stored as the _DEFERRED_MAIN_CURVE_SCOPE sentinel) resolved its scope
# IMMEDIATELY at a mid-parse flush, binding to the partial curve list
# (only the CORE_DEFINITION curves) — DEPT/GR columns silently lost.
# Replay now re-queues the piped variant too, so the final replay binds
# it to the complete main curve block.
# ──────────────────────────────────────────────────────────────


class TestM7DeferredPipedCurveRescoped:
    """M7: a deferred pre-~V "~A | CURVE" section flushed before the main
    ~C must keep ALL its columns (bound to the complete curve list)."""

    def test_piped_curve_section_flush_before_main_c_preserves_columns(
        self, tmp_path: Path
    ) -> None:
        content = (
            "~A | CURVE\n"
            "100 10\n"
            "101 11\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            "~CORE_DEFINITION\n"
            " RHOZ.OHMM : RESISTIVITY\n"
            "~CORE_DATA\n"
            " 2.5\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : 1  : DEPTH\n"
            " GR.GAPI : 2  : GAMMA\n"
        )
        test_file = tmp_path / "m7_piped_deferred.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        log_sections = [ds for ds in las.data_sections if ds.section_type == "LOG_DATA"]
        assert log_sections, "expected a LOG_DATA section"
        ds = log_sections[0]
        np.testing.assert_allclose(ds.data["DEPT"], [100.0, 101.0], err_msg="DEPT lost")
        np.testing.assert_allclose(ds.data["GR"], [10.0, 11.0], err_msg="GR column discarded")


# ──────────────────────────────────────────────────────────────
# M9 (parser, MEDIUM, Stage 9): F-24's pre-scan curve counter counted
# ~C lines the parser DROPS (multi-word no-period mnemonics like
# "HOLE DIA : hole diameter" fail _MNEMONIC_LINE_RE in _parse_curve),
# inflating curve_def_count.  The F-24 count-equality header-skip
# predicate then failed to skip a genuine full header row → spurious
# "Pre-scan overcount" warning + F36 trim on valid files.  The pre-scan
# now counts only curves that survive parsing.
# ──────────────────────────────────────────────────────────────


class TestM9PreScanDroppedCurveLine:
    """M9: the pre-scan must not count dropped ~C lines — a valid file
    with a dropped "HOLE DIA" line + mnemonic header row warns no more."""

    def test_dropped_curve_line_no_overcount_warning(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            " HOLE DIA : hole diameter\n"
            " GR.GAPI  : GAMMA\n"
            "~A DEPT GR\n"
            "DEPT GR\n"
            "1000 10\n"
            "1001 11\n"
        )
        test_file = tmp_path / "m9_dropped_curve.las"
        test_file.write_text(content, encoding="utf-8")
        # The pre-scan's data_line_count must equal the reader's actual
        # consumption: the header row "DEPT GR" IS skipped (2 tokens == 2
        # surviving curves), leaving exactly 2 data rows.
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(content)
        assert parser.data_line_count == 2, (
            f"M9: pre-scan counted {parser.data_line_count} data lines, expected 2 "
            "(dropped 'HOLE DIA' must not inflate the curve count)"
        )
        # Full read: no spurious overcount warning, data intact.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"M9: spurious overcount warning: {[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0])


# ──────────────────────────────────────────────────────────────
# M10 (parser x writer, MEDIUM, Stage 9): F-23(a) made the WELL parser
# store multi-word no-period keys ("HOLE DIA : 85.7" → well.entries),
# but the writer's N-I-19 key validation rejects space-containing keys
# → read→write raised LASWriteError.  The well path now guards with
# _MNEMONIC_LINE_RE like the curve/param paths (drop with warning), and
# the no-period value is not re-split by the LAS 1.2 bare-colon CWLS
# logic (L24 — "LOC : ACME:OIL" must keep value "ACME:OIL").
# ──────────────────────────────────────────────────────────────


class TestM10WellNoPeriodRoundtrip:
    """M10: a multi-word no-period well line must not crash read→write;
    the single-word no-period value with an embedded colon is preserved."""

    def test_multi_word_well_line_read_write_roundtrip(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " STRT.M  100.0  : START DEPTH\n"
            " HOLE DIA : 85.7\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "m10_well.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        # The multi-word key is dropped (not stored) — see TestF23LasioLineShapes.
        assert "HOLE DIA" not in las.well.entries
        out = tmp_path / "m10_out.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)  # must NOT raise LASWriteError
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las2 = read_las_file_as_object(out)
        assert las2.well["STRT"] == "100.0"

    def test_las12_no_period_colon_in_value_preserved(self, tmp_path: Path) -> None:
        """L24: the LAS 1.2 bare-colon CWLS split must NOT re-split a
        no-period value — the first colon already separated MNEM from VALUE."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            "~WELL INFORMATION\n"
            " LOC : ACME:OIL\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "m10_l24.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert las.well["LOC"] == "ACME:OIL", (
            f"L24: colon-in-value corrupted: {las.well.get('LOC')!r} (expected 'ACME:OIL')"
        )

    def test_las12_deferred_no_period_colon_in_value_preserved(self, tmp_path: Path) -> None:
        """L24 deferred path: ~W before ~V — the no_period flag must
        survive the deferred replay so the value is not re-split."""
        content = (
            "~WELL INFORMATION\n"
            " LOC : ACME:OIL\n"
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A DEPT\n"
            "1000.0\n"
        )
        test_file = tmp_path / "m10_l24_deferred.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        assert las.well["LOC"] == "ACME:OIL", (
            f"L24 (deferred): colon-in-value corrupted: {las.well.get('LOC')!r}"
        )


# ──────────────────────────────────────────────────────────────
# E-33 (dev_reader, MEDIUM): thousands summary warning text was
# factually wrong for decimal/exponent variants.
# ──────────────────────────────────────────────────────────────


class TestE33ThousandsWarningTextAccurate:
    """E-33: the non-semicolon thousands summary warning claimed the
    values "were not converted and may be NaN" — false for the
    decimal/exponent variants ("1,234.5", "1,234E3") which F-06 ALWAYS
    converts (comma-stripped).  Only the ambiguous bare form ("1,234")
    stays unconverted → NaN.  The message now states both behaviors
    accurately; conversion VALUES are unchanged (pinned by
    TestF06ThousandsDecimalConversion).  FAILS on pre-fix code (message
    claims "were not converted"), PASSES on post-fix code."""

    def test_decimal_thousands_warning_text_is_accurate(self, tmp_path: Path) -> None:
        content = "1,234.5 5,678.9\n9,000.5 10,456.7\n"
        test_file = tmp_path / "e33_decimal.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        # Values still convert (F-06 pinned behavior).
        np.testing.assert_array_equal(data["col_0"], [1234.5, 9000.5])
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1
        # E-33: the message must not claim the values were not converted.
        assert all("were not converted" not in m for m in thousands_warnings), (
            f"Decimal thousands are converted; message must not deny it: "
            f"{thousands_warnings}"
        )
        # And must state the converted reading explicitly.
        assert any("commas stripped" in m for m in thousands_warnings), (
            f"Message must state decimal thousands are comma-stripped: "
            f"{thousands_warnings}"
        )

    def test_bare_thousands_warning_mentions_nan(self, tmp_path: Path) -> None:
        """Bare "1,234" in space mode IS unconverted → NaN; the message
        must still say so (M-25 policy, unchanged)."""
        content = "1,234 5,678\n9,000 10,456\n"
        test_file = tmp_path / "e33_bare.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_dev_file(test_file)

        assert all(np.isnan(v) for v in data["col_0"])
        thousands_warnings = [str(x.message) for x in w if "thousands" in str(x.message)]
        assert len(thousands_warnings) >= 1
        assert any("not converted and becomes NaN" in m for m in thousands_warnings), (
            f"Bare thousands stay NaN; message must state it: {thousands_warnings}"
        )


# ──────────────────────────────────────────────────────────────
# N-14 (models, MEDIUM): DevFile.from_dict ALSO double-warned.
# ──────────────────────────────────────────────────────────────


class TestN14FromDictSingleValidationPass:
    """N-14 (shared root with E-34): DevFile.from_dict ran
    validate(complete=True) AND _validate_dev_data, double-warning every
    overlapping issue (4 warnings for 2 distinct issues).  from_dict now
    runs _validate_dev_data as the SINGLE data-quality pass.
    FAILS on pre-fix code (2 warnings per issue), PASSES on post-fix code
    (1 warning per issue)."""

    def test_from_dict_warns_once_per_issue(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([100.0, 50.0]),
                "AZI": np.array([450.0, 45.0]),
                "INC": np.array([30.0, 30.0]),
            },
            column_order=["MD", "AZI", "INC"],
            _from_dict=True,
        )
        d = dev.to_dict()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DevFile.from_dict(d)

        mono = [x for x in w if "not monotonically increasing" in str(x.message)]
        azi = [x for x in w if "Azimuth column" in str(x.message)]
        assert len(mono) == 1, (
            f"MD monotonicity must warn exactly once, got {len(mono)}: "
            f"{[str(x.message) for x in w]}"
        )
        assert len(azi) == 1, (
            f"AZI range must warn exactly once, got {len(azi)}: "
            f"{[str(x.message) for x in w]}"
        )

    def test_from_dict_nan_inf_warning_survives(self) -> None:
        """M-25 coordination: the NaN/Inf check must survive the from_dict
        dedup (it now lives in _validate_dev_data)."""
        dev = DevFile(
            columns={
                "MD": np.array([100.0, np.nan]),
                "AZIM": np.array([10.0, 20.0]),
            },
            column_order=["MD", "AZIM"],
            _from_dict=True,
        )
        d = dev.to_dict()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DevFile.from_dict(d)

        non_finite = [x for x in w if "non-finite values" in str(x.message)]
        assert len(non_finite) == 1, (
            f"NaN/Inf warning must fire exactly once, got {len(non_finite)}: "
            f"{[str(x.message) for x in w]}"
        )


# ──────────────────────────────────────────────────────────────
# M-01 (MEDIUM, CONFIRMED): `_designed_nan` was never set on the DEV
# read path — the read-path _validate_dev_data NaN block warned
# "column 'X' contains non-finite values (NaN/Inf)" as CORRUPTION on
# every reader-designed NaN (short-row fills, sentinel fills).  The
# reader now marks its DevFile with ``_designed_nan=True`` and the
# NaN half of the non-finite check is gated on the marker (Inf is
# always flagged).  Genuine non-finite user data (marker unset, e.g.
# direct construction or DevFile.from_dict) still warns.
# FAILS on pre-fix code (false positive fired), PASSES on post-fix code.
# ──────────────────────────────────────────────────────────────


class TestM01DesignedNanReadPathSuppression:
    """M-01: reader-designed NaN must not fire the false "non-finite"
    corruption warning on the DEV read path."""

    def test_short_row_fill_no_false_warning(self, tmp_path: Path) -> None:
        """A short data row (fewer values than declared columns) is filled
        with the reader's designed NaN — no non-finite corruption warning."""
        content = (
            "MD,TVD,X,Y\n"
            "100.0,100.0,10.0,20.0\n"
            "200.0,200.0,30.0\n"
        )
        test_file = tmp_path / "m01_short_row.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dev = read_dev_file_as_object(test_file)

        non_finite = [x for x in w if "non-finite values" in str(x.message)]
        assert len(non_finite) == 0, (
            f"Short-row designed NaN must not fire the corruption warning, "
            f"got {len(non_finite)}: {[str(x.message) for x in w]}"
        )
        assert np.isnan(dev.columns["Y"][1])

    def test_sentinel_fill_no_false_warning(self, tmp_path: Path) -> None:
        """Text (``na``) and numeric (``-999.25``) sentinels map to the
        reader's designed NaN — no non-finite corruption warning."""
        content = (
            "MD,TVD,AZIM,INC\n"
            "100.0,-999.25,45.0,30.0\n"
            "200.0,na,45.0,30.0\n"
            "300.0,300.0,45.0,30.0\n"
        )
        test_file = tmp_path / "m01_sentinel.dev"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dev = read_dev_file_as_object(test_file)

        non_finite = [x for x in w if "non-finite values" in str(x.message)]
        assert len(non_finite) == 0, (
            f"Sentinel designed NaN must not fire the corruption warning, "
            f"got {len(non_finite)}: {[str(x.message) for x in w]}"
        )
        assert np.isnan(dev.columns["TVD"][0])
        assert np.isnan(dev.columns["TVD"][1])

    def test_read_devfile_validate_clean_with_designed_nan(self, tmp_path: Path) -> None:
        """The DevFile the reader returns carries ``_designed_nan=True``:
        validate(complete=True) on it reports no non-finite issue."""
        content = "MD,AZIM\n100.0,10.0\nnan,20.0\n200.0,30.0\n"
        test_file = tmp_path / "m01_read_validate.dev"
        test_file.write_text(content, encoding="utf-8")

        dev = read_dev_file_as_object(test_file)
        assert dev._designed_nan is True
        issues = dev.validate(complete=True)
        assert not any("non-finite" in i for i in issues), issues

    def test_direct_corrupt_devfile_still_warns(self) -> None:
        """M-01 control: genuinely corrupt non-finite USER data (DevFile
        constructed directly, marker unset) still fires the warning."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile(
            columns={"MD": np.array([0.0, np.nan])},
            column_order=["MD"],
            _from_dict=True,
        )
        assert dev._designed_nan is False
        with pytest.warns(UserWarning, match="non-finite values"):
            _validate_dev_data(dev)

    def test_inf_always_warns_even_designed(self) -> None:
        """M-01: Inf is never a designed sentinel (the reader collapses all
        non-finite conversions to NaN), so it is flagged even when the
        marker is set."""
        from pylasdev.dev_reader import _validate_dev_data

        dev = DevFile(
            columns={"MD": np.array([0.0, np.inf])},
            column_order=["MD"],
            _from_dict=True,
            _designed_nan=True,
        )
        with pytest.warns(UserWarning, match="non-finite values"):
            _validate_dev_data(dev)


# ──────────────────────────────────────────────────────────────
# M-06 (MEDIUM, CONFIRMED): the top-level E-10 re-check in
# LASFile.validate(complete=True) passed no data_formats, so a
# post-construction top-level data_format mutation to mixed codes
# passed validate() with [] and the writer emitted a file the
# library's own reader rejects ("Inconsistent data_format") with
# zero warnings.  validate() now passes the top-level curve
# data_formats (mirroring construction-time __post_init__).
# FAILS on pre-fix code (validate == []), PASSES on post-fix code.
# ──────────────────────────────────────────────────────────────


class TestM06TopLevelDataFormatMutation:
    """M-06: validate(complete=True) flags a post-construction top-level
    data_format mutation to mixed codes."""

    @staticmethod
    def _las() -> LASFile:
        return LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "NMR[1]", "NMR[2]", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT", data_format="F"),
                CurveDefinition(mnemonic="NMR[1]", data_format="F"),
                CurveDefinition(mnemonic="NMR[2]", data_format="F"),
                CurveDefinition(mnemonic="GR", data_format="F"),
            ],
            logs={
                m: np.array([1.0, 2.0]) for m in ["DEPT", "NMR[1]", "NMR[2]", "GR"]
            },
        )

    def test_top_level_mixed_format_mutation_flagged(self) -> None:
        """Mutating one channel of a top-level array to a different
        data_format (NMR[2] F→I, NMR[1] stays F) must be flagged by
        validate(complete=True) — pre-fix validate() returned [] and the
        writer emitted a self-unreadable file with zero warnings."""
        las = self._las()
        assert las.validate(complete=True) == []

        las.curves[2].data_format = "I"
        issues = las.validate(complete=True)
        assert any("inconsistent data_format" in i and "NMR" in i for i in issues), issues

    def test_uniform_format_mutation_stays_clean(self) -> None:
        """M-06 control: mutating BOTH channels to the same format keeps
        validate(complete=True) clean — only MIXED formats are the
        reader-rejected state."""
        las = self._las()
        las.curves[1].data_format = "E"  # NMR[1]
        las.curves[2].data_format = "E"  # NMR[2]
        assert las.validate(complete=True) == []


# ──────────────────────────────────────────────────────────────
# E-17 + E-43 (parser, MEDIUM, CONFIRMED — regressing function
# `_pre_scan`, fix-audit-prescan): the pre-scan estimate must equal the
# reader's actual data-line consumption on every historical divergence
# trigger — no spurious "Pre-scan overcount" warning, no G-04 growth on
# valid files.  The phase-2 finalize re-uses the reader's own primitives
# (shared dedup-aware declared-set builder + _detect_string_curves +
# is_mnemonic_header_row), so these fixtures can never diverge again.
# FAILS on pre-fix code, PASSES on post-fix code.
# ──────────────────────────────────────────────────────────────


class TestE17E43PreScanReaderParity:
    """E-17/E-43 parity lock: ``parser.data_line_count`` equals the
    reader's consumption for every divergence trigger.  Pre-fix behavior
    per trigger: multi-marker ``{S}..{I}`` → spurious overcount; plain
    ``{A:5}`` single curve → silent undercount (all-string mismatch);
    ``NMR[1]`` bracket strip → overcount; raw ``GR GR`` + ``GR_2`` header
    → overcount; F-22 cross-base ``DEPT_2_2`` → overcount;
    ``~LOG_DEFINITION`` on 1.2/2.0 → overcount."""

    @staticmethod
    def _head() -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
        )

    def _assert_parity(self, tmp_path: Path, name: str, content: str, expected: int) -> None:
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(content)
        assert parser.data_line_count == expected, (
            f"{name}: pre-scan counted {parser.data_line_count} data lines, "
            f"expected {expected} (reader consumption)"
        )
        test_file = tmp_path / f"{name}.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_las_file_as_object(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"{name}: spurious pre-scan overcount: {[str(w.message) for w in overcount]}"
        )

    def test_multi_marker_s_i_description_no_overcount(self, tmp_path: Path) -> None:
        """E-17 (iter-3 trigger): a multi-marker description '{S}..{I}' on a
        SINGLE curve.  The reader uses the TRAILING marker ({I} → numeric);
        the old pre-scan used FIRST-match ({S} → all_string=True), so the
        standalone 'DEPT' header was counted as data → spurious overcount."""
        content = (
            self._head()
            + " DEPT.M : Depth {S} ray {I}\n"
            "~A\n"
            "DEPT\n"
            "1000.0\n"
            "1001.0\n"
        )
        self._assert_parity(tmp_path, "e17_multi_marker", content, expected=2)

    def test_plain_a_offset_single_curve_is_all_string(self, tmp_path: Path) -> None:
        """E-17 ({A:N} trigger): a PLAIN '{A:5}' single curve classifies as
        STRING on the reader side (data_format='A' with no bracket mnemonic
        keeps array_info=None), so 'LITH' IS data (all-string exclusion).
        The old pre-scan's `not _off` truthiness missed {A:5} and skipped
        the header → silent undercount + G-04 growth."""
        content = (
            self._head()
            + " LITH. : Lithology {A:5}\n"
            "~A\n"
            "LITH\n"
            "SAND\n"
            "CLAY\n"
        )
        self._assert_parity(tmp_path, "e17_plain_a5", content, expected=3)

    def test_bracketed_array_mnemonic_header_no_overcount(self, tmp_path: Path) -> None:
        """E-43 (bracket axis): header tokens 'DEPT NMR[1] NMR[2]' are all
        declared curves on the reader side (curves_order keeps brackets).
        The old pre-scan stripped brackets ('NMR[1]'→'NMR') so its match
        set could never contain 'NMR[1]' → header counted → overcount."""
        content = (
            self._head()
            + " DEPT.M : Depth\n"
            " NMR[1]. : NMR {A:1}\n"
            " NMR[2]. : NMR {A:2}\n"
            "~A\n"
            "DEPT NMR[1] NMR[2]\n"
            "1000.0 10.0 20.0\n"
            "1001.0 11.0 21.0\n"
        )
        self._assert_parity(tmp_path, "e43_bracket", content, expected=2)

    def test_raw_duplicate_curves_gr_2_header_no_overcount(self, tmp_path: Path) -> None:
        """E-43 (dedup axis): the ~C block declares GR twice; the reader
        renames the second to GR_2 BEFORE building its declared set, so the
        'DEPT GR GR_2' header IS skipped.  The old pre-scan built its match
        set from raw pre-dedup names (never GR_2) → header counted →
        overcount."""
        content = (
            self._head()
            + " DEPT.M : Depth\n"
            " GR.GAPI : Gamma\n"
            " GR.API : Gamma2\n"
            "~A\n"
            "DEPT GR GR_2\n"
            "1000.0 10.0 20.0\n"
            "1001.0 11.0 21.0\n"
        )
        self._assert_parity(tmp_path, "e43_dedup", content, expected=2)

    def test_f22_cross_base_collision_header_no_overcount(self, tmp_path: Path) -> None:
        """E-43 (F-22 axis): 'DEPT DEPT DEPT_2' dedups to
        'DEPT DEPT_2 DEPT_2_2'; the header uses the post-dedup names.
        The old pre-scan's set ({DEPT, DEPT_2}) lacked DEPT_2_2 → counted
        → overcount."""
        content = (
            self._head()
            + " DEPT.M : Depth\n"
            " DEPT.M : Depth2\n"
            " DEPT_2.M : Depth3\n"
            "~A\n"
            "DEPT DEPT_2 DEPT_2_2\n"
            "1000.0 10.0 20.0\n"
            "1001.0 11.0 21.0\n"
        )
        self._assert_parity(tmp_path, "e43_f22_crossbase", content, expected=2)

    def test_definition_section_curves_in_declared_set(self, tmp_path: Path) -> None:
        """E-17 (_DEFINITION axis): a ~LOG_DEFINITION section declares
        DEPT/GR (the parser routes _DEFINITION to the curve handler and
        the reader's declared set includes them), but the old pre-scan
        only entered curve mode for {'C','CURVE'} → its set was empty →
        the 'DEPT GR' header was counted → overcount.

        N-08 (fix-parser-C): the _DEFINITION dispatch is now gated on
        is_las30 — on a KNOWN non-3.0 file ~{Name}_DEFINITION is a
        customer section (other_lines) and must NOT inject curves.  This
        parity fixture therefore uses a LAS 3.0 file, where _DEFINITION
        legitimately declares curves and the pre-scan/reader lock still
        applies."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~LOG_DEFINITION\n"
            " DEPT.M : Depth\n"
            " GR.GAPI : Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "1000.0 10.0\n"
            "1001.0 11.0\n"
        )
        self._assert_parity(tmp_path, "e17_definition", content, expected=2)

    def test_shared_declared_set_pre_dedup_equals_reader_post_dedup(self) -> None:
        """Unit-level lock (fix-audit-prescan §b.4): the phase-2 finalize's
        declared set — built on PRE-dedup state via _declared_mnemonic_set —
        must equal _mnemonic_header_declared on the reader's POST-dedup
        model.  One code path, one algorithm, deterministic on the same
        declaration order."""
        names = ["DEPT", "GR", "GR", "GR_2"]  # raw dup + F-22 collision input
        declared_phase2 = _declared_mnemonic_set(names, ["", "", "", ""])
        las = LASFile()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las.curves_order = list(names)
            for n in names:
                las.curves.append(CurveDefinition(mnemonic=n))
            _deduplicate_curves(las)
        declared_reader = _mnemonic_header_declared(las)
        assert declared_phase2 == declared_reader, (
            f"phase-2 declared set {sorted(declared_phase2)} != "
            f"reader post-dedup set {sorted(declared_reader)}"
        )
        assert "GR_2" in declared_phase2 and "GR_2_2" in declared_phase2, (
            f"post-dedup names missing from phase-2 declared set: "
            f"{sorted(declared_phase2)}"
        )


# ──────────────────────────────────────────────────────────────
# E-18 (parser, MEDIUM, CONFIRMED): the LAS 3.0 ~Other rejection was
# version-ORDER-dependent — ~O BEFORE ~V was silently accepted (is_las30
# still False at the default 2.0), its content landed in `other`, and the
# LAS 3.0 writer DROPPED it on write (read-OK → write-drops asymmetry).
# parse() now re-checks once the version is final (keyed on the recorded
# section type "O") and raises LASParseError mirroring the in-section
# rejection.  FAILS on pre-fix code (parse succeeds), PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestE18OtherOrderIndependentLas30Rejection:
    """E-18: an ~Other section in a LAS 3.0 file raises LASParseError
    regardless of whether ~O appears before or after ~V; LAS 1.2/2.0 keeps
    accepting ~O in any position; ~-noise lines (~., ~#) routed to `other`
    without an ~O section must NOT raise."""

    _V3_BODY = (
        "~VERSION INFORMATION\n"
        " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        "~WELL INFORMATION\n"
        " NULL.   -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   : DEPTH\n"
        "~A\n"
        "1000.0\n"
    )

    def test_other_before_version_raises(self) -> None:
        """~O before ~V in a LAS 3.0 file: pre-fix silently accepted (is_las30
        still False at the default 2.0), content landed in `other`, then the
        3.0 writer dropped it — read-OK → write-drops asymmetry.  Post-fix
        the version-final re-check raises."""
        content = "~OTHER\nfree text\n" + self._V3_BODY
        with pytest.raises(LASParseError, match="~Other"):
            LASParser().parse(content)

    def test_other_before_version_raises_on_read_path(self, tmp_path: Path) -> None:
        """The full read path (read_las_file_as_object) surfaces the same
        rejection — the parser-side re-check must be version-order-independent
        for library consumers too."""
        content = "~OTHER\nfree text\n" + self._V3_BODY
        test_file = tmp_path / "e18_other_before_v.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError, match="~Other"):
            read_las_file_as_object(test_file)

    def test_other_after_version_still_raises(self) -> None:
        """Control: ~O AFTER ~V on LAS 3.0 was already rejected in-section —
        the post-parse re-check must not change that behavior."""
        content = self._V3_BODY + "~OTHER\nmore text\n"
        with pytest.raises(LASParseError, match="~Other"):
            LASParser().parse(content)

    def test_other_before_version_las20_accepted(self) -> None:
        """Control: LAS 1.2/2.0 allows ~Other — an ~O section before ~V must
        still parse (the E-18 rejection is LAS 3.0-only)."""
        content = (
            "~OTHER\nfree text\n"
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASParser().parse(content)
        assert "free text" in las.other

    def test_tilde_noise_lines_las30_not_rejected(self) -> None:
        """Control: ~-prefixed non-section lines (~., ~#) are routed to
        `other` by F-83 WITHOUT an ~O section — the re-check keys on the
        recorded section type "O", so these must NOT raise on LAS 3.0."""
        content = self._V3_BODY.replace("~A\n", "~A\n~.\n~#\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASParser().parse(content)
        assert las.other == "~.\n~#\n"


# ──────────────────────────────────────────────────────────────
# E-38 (parser, MEDIUM, CONFIRMED): parse() routed LASFile.validate
# (complete=True) + _ParserState.validate() data-integrity issues
# EXCLUSIVELY to logger.warning — zero warnings-API visibility.
# parse() now ALSO emits warnings.warn (L-01 convention) for every issue.
# FAILS on pre-fix code (catch_warnings sees nothing), PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestE38ParseValidationIssuesWarningsApi:
    """E-38: data-integrity issues found by parse() must be visible through
    the warnings API (catch_warnings / warnings-as-errors suites), not just
    the logging API."""

    def test_validate_complete_issue_emitted_via_warnings(self) -> None:
        """STEP=0 is a validate(complete=True) issue — pre-fix it was
        logger-only; post-fix a 'LASFile validation issue' UserWarning is
        emitted."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1002.0 : STOP DEPTH\n"
            " STEP.M   0.0    : STEP\n"
            " NULL.   -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            "~A DEPT\n"
            "1000.0\n"
            "1001.0\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            LASParser().parse(content)
        issues = [str(w.message) for w in rec if "LASFile validation issue" in str(w.message)]
        assert any("STEP is zero" in m for m in issues), (
            f"STEP=0 validate issue not visible via warnings API: {issues}"
        )


# ──────────────────────────────────────────────────────────────
# H-01 (parser, WEAKENED HIGH→MEDIUM, CONFIRMED regression facet):
# the N-09 curve-side brace unescape ran ONLY inside
# ``if format_matches:`` — brace descriptions that produce no format
# match (digit-led "2{3}4", unbalanced "Depth {" on 3.0; non-{I,S,A}
# brace text on 1.2/2.0 where the M-35 filter empties the match list)
# kept the writer's ``\{`` escapes → silent backslash corruption that
# accumulated on every write→read roundtrip (HEAD roundtripped EXACT).
# The unescape now runs unconditionally on the curve path, mirroring
# the parameter path.  FAILS on pre-fix code, PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestH01CurveBraceUnescapeUnconditional:
    """H-01: write→read roundtrip of brace descriptions that do NOT
    produce a format match must be EXACT — the curve-side brace unescape
    runs unconditionally (mirror of the parameter path), so the writer's
    ``\\{`` escapes are always reversed."""

    @staticmethod
    def _roundtrip(
        tmp_path: Path,
        vers: str,
        data_format: str,
        description: str,
        dlm: str = "COMMA",
    ) -> tuple[str, str]:
        las = LASFile(version=VersionSection(vers=vers, wrap="NO", dlm=dlm))
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(
            CurveDefinition(
                mnemonic="DEPT", unit="M", data_format=data_format, description=description
            )
        )
        las.logs["DEPT"] = np.array([1000.0, 1001.0])
        out = tmp_path / "h01_roundtrip.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        cd = next(c for c in back.curves if c.mnemonic == "DEPT")
        return cd.description, cd.data_format

    def test_las30_digit_led_braces_roundtrip_exact(self, tmp_path: Path) -> None:
        """3.0, data_format='' — "2{3}4" has NO format match (digit-led);
        pre-fix the description came back as '2\\{3\\}4' (literal
        backslashes, worsening per roundtrip); post-fix EXACT."""
        desc, fmt = self._roundtrip(tmp_path, "3.0", "", "2{3}4")
        assert desc == "2{3}4", f"H-01: digit-led braces corrupted: {desc!r}"
        assert fmt == "", f"H-01: data_format fabricated: {fmt!r}"

    def test_las30_unbalanced_brace_roundtrip_exact(self, tmp_path: Path) -> None:
        """3.0 — "Depth {" has no closing brace, so no format match;
        pre-fix 'Depth \\{' leaked literal backslashes; post-fix EXACT."""
        desc, fmt = self._roundtrip(tmp_path, "3.0", "", "Depth {")
        assert desc == "Depth {", f"H-01: unbalanced brace corrupted: {desc!r}"
        assert fmt == "", f"H-01: data_format fabricated: {fmt!r}"

    def test_las30_non_format_brace_text_roundtrip_exact(self, tmp_path: Path) -> None:
        """3.0 — "Bulk {Density}" + real format F: the {Density} token is
        user text and must survive; the trailing {F} is stripped; EXACT."""
        desc, fmt = self._roundtrip(tmp_path, "3.0", "F", "Bulk {Density}")
        assert desc == "Bulk {Density}", f"H-01: brace text corrupted: {desc!r}"
        assert fmt == "F", f"H-01: data_format lost: {fmt!r}"

    def test_las20_non_isa_brace_description_roundtrip_exact(self, tmp_path: Path) -> None:
        """2.0 — the M-35 filter drops {Density} (non-{I,S,A}) from the
        match list → format_matches empty → pre-fix the unescape never ran
        and 'Bulk \\{Density\\}' survived; post-fix EXACT."""
        desc, fmt = self._roundtrip(tmp_path, "2.0", "", "Bulk {Density}")
        assert desc == "Bulk {Density}", f"H-01: 2.0 brace text corrupted: {desc!r}"
        assert fmt == "", f"H-01: 2.0 data_format fabricated: {fmt!r}"

    def test_las20_n09_mid_description_s_token_preserved(self, tmp_path: Path) -> None:
        """N-09 target preserved on 2.0: 'Gamma {S} ray' roundtrips EXACT
        and no data_format is fabricated from the mid-description token
        (the M-35 filter empties the match list; pre-fix the escaped
        '\\{S\\}' text survived)."""
        desc, fmt = self._roundtrip(tmp_path, "2.0", "", "Gamma {S} ray")
        assert desc == "Gamma {S} ray", f"H-01: N-09 text corrupted: {desc!r}"
        assert fmt == "", f"H-01: 2.0 data_format fabricated: {fmt!r}"

    def test_las30_n09_mid_description_s_token_roundtrip_exact(self, tmp_path: Path) -> None:
        """N-09 target preserved on 3.0: 'Gamma {S} ray' with real
        data_format 'S' roundtrips EXACT with the format intact."""
        desc, fmt = self._roundtrip(tmp_path, "3.0", "S", "Gamma {S} ray")
        assert desc == "Gamma {S} ray", f"H-01: N-09 text corrupted: {desc!r}"
        assert fmt == "S", f"H-01: 3.0 data_format lost: {fmt!r}"


# ──────────────────────────────────────────────────────────────
# M-02 (parser, MEDIUM, CONFIRMED): _finalize_pre_scan subtracted only
# the standalone mnemonic header row, never the optional units row the
# reader also skips (data_reader.py:1056-1061) → spurious "Pre-scan
# overcount" warning + data_line_count off-by-one on M-13-shaped files
# (mnemonic header + units row).  The pre-scan now subtracts the units
# row with the SAME is_units_header_row predicate the reader consumes
# (shared M-13 contract).  FAILS on pre-fix code (overcount warning
# fires), PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestM02PreScanUnitsRowSubtracted:
    """M-02: a units row directly below the mnemonic header row is
    skipped by the reader AND subtracted by the parser pre-scan — no
    spurious 'Pre-scan overcount' warning, data_line_count matches the
    reader's actual consumption."""

    _CONTENT_TEMPLATE = (
        "~VERSION INFORMATION\n"
        " VERS.   {vers}  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        " GR.GAPI  :  Gamma\n"
        "~A\n"
        "DEPT GR\n"
        "M GAPI\n"
        "1000.0 50.0\n"
        "1001.0 51.0\n"
    )

    def test_las20_no_pre_scan_overcount_on_m13_file(self, tmp_path: Path) -> None:
        """End-to-end: the M-13 file parses with NO 'Pre-scan overcount'
        warning and correct data (pre-fix: phase-1 counted 4 lines, the
        header-only subtraction left 3, the reader consumed 2 → spurious
        warning + F36 trim path)."""
        content = self._CONTENT_TEMPLATE.format(vers="2.0")
        test_file = tmp_path / "m02_units.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        overcount = [w for w in caught if "Pre-scan overcount" in str(w.message)]
        assert not overcount, (
            f"M-02: spurious pre-scan overcount on M-13 file: "
            f"{[str(w.message) for w in overcount]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

    def test_las20_data_line_count_matches_reader_consumption(self) -> None:
        """Parser-level: phase-1 counts 4 non-comment lines; the header
        AND the units row are subtracted → data_line_count == 2 (the
        reader's actual consumption)."""
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(self._CONTENT_TEMPLATE.format(vers="2.0"))
        assert parser.data_line_count == 2, (
            f"M-02: pre-scan must subtract header + units row, "
            f"data_line_count={parser.data_line_count}"
        )

    def test_las12_data_line_count_matches_reader_consumption(self) -> None:
        """LAS 1.2 twin of the M-13 subtraction contract."""
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(self._CONTENT_TEMPLATE.format(vers="1.2"))
        assert parser.data_line_count == 2, (
            f"M-02: LAS 1.2 pre-scan must subtract header + units row, "
            f"data_line_count={parser.data_line_count}"
        )


# ──────────────────────────────────────────────────────────────
# M-04 (parser, MEDIUM, CONFIRMED): the LAS 3.0 accumulation skipped
# the standalone mnemonic header row (M-40) but NOT a following units
# row → phantom all-null first row + one-row shift (DEPT=
# [-999.25, 1000, 1001]).  The accumulation now skips the units row on
# the first data line only, gated on a mnemonic header row having just
# been skipped — same is_units_header_row predicate and same gate as
# the reader's M-13 skip (data_reader.py:1056-1061).  Also mirrored on
# the deferred (pre-~V) replay path.  FAILS on pre-fix code, PASSES on
# post-fix.
# ──────────────────────────────────────────────────────────────


class TestM04Las30HeaderUnitsAccumulation:
    """M-04: a units row directly below the mnemonic header row inside a
    LAS 3.0 data section is a header, not data — no phantom all-null
    first row, no one-row shift."""

    def test_las30_header_units_row_no_phantom_row(self, tmp_path: Path) -> None:
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
            "DEPT GR\n"
            "M GAPI\n"
            "1000.0 50.0\n"
            "1001.0 55.0\n"
        )
        test_file = tmp_path / "m04_units.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        # Pre-fix: "M GAPI" consumed as the first data row →
        # DEPT=[-999.25, 1000.0, 1001.0] (phantom all-null first row).
        np.testing.assert_allclose(sec.data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(sec.data["GR"], [50.0, 55.0])

    def test_las30_pre_v_deferred_header_units_row_no_phantom_row(
        self, tmp_path: Path
    ) -> None:
        """Deferred (pre-~V) twin: the PARS-04 replay filter must apply
        the same header + units skip — no phantom row on the deferred
        path either."""
        content = (
            "~ASCII\n"
            "DEPT,GR\n"
            "M,GAPI\n"
            "1000.0,50.0\n"
            "1001.0,55.0\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
        )
        test_file = tmp_path / "m04_units_deferred.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(sec.data["GR"], [50.0, 55.0])

    def test_las30_letters_only_first_row_not_skipped_without_header(
        self, tmp_path: Path
    ) -> None:
        """Control: without a preceding mnemonic header row, a first row
        of letters-only tokens is DATA (all-string section), never a
        units row — the units skip is gated on the header having been
        skipped first (position gate), mirroring the reader."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " LITH.    :  Lithology {S}\n"
            " FORM.    :  Formation {S}\n"
            "~LOG\n"
            "ACME SAND\n"
            "SHALE GRN\n"
        )
        test_file = tmp_path / "m04_control.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        # Both letters-only rows are genuine data rows: no mnemonic
        # header row precedes them, so nothing may be skipped as units.
        np.testing.assert_array_equal(
            sec.string_data["LITH"], np.array(["ACME", "SHALE"], dtype=object)
        )
        np.testing.assert_array_equal(
            sec.string_data["FORM"], np.array(["SAND", "GRN"], dtype=object)
        )


# ──────────────────────────────────────────────────────────────
# M-04 fix3 (parser, MEDIUM, CONFIRMED — P1): the
# _skipped_mnemonic_header units-row position gate leaked across
# section transitions when the prior A section accumulated ZERO data
# rows (skipped mnemonic header + optional units row only).  The next
# section's genuine letters-only first data row was then silently
# dropped (repro: DEPT=[2000.0, 2001.0] with 3 rows in the file —
# "ACME SAND" lost).  The gate is now closed one-shot at the units row
# AND at every section boundary (flush entry + A→unknown empty path),
# so the flag never survives into a subsequent section (invariant at
# parser.py:1202-1205).  FAILS on pre-fix code, PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestM04Fix3StaleHeaderFlagSectionTransition:
    """fix3-P1: _skipped_mnemonic_header must not leak across section
    transitions after an A section that accumulated no data rows — the
    next section's letters-only first data row must be consumed."""

    # LAS 3.0 template: the first ~LOG is header+units-only (ZERO data
    # rows); the second ~LOG carries a letters-only first row followed
    # by numeric rows scoped to the DEPT/GR curves from the initial ~C.
    # Post-fix the letters-only row is consumed as data (null-filled
    # DEPT/GR) → DEPT=[-999.25, 2000.0, 2001.0], 3 rows total.
    @staticmethod
    def _numeric_body(transition: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   2000.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG\n"
            "DEPT GR\n"
            "M GAPI\n"
            + transition
            + "~LOG\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )

    @staticmethod
    def _assert_three_rows(las: Any) -> None:
        sec = las.data_sections[1] if len(las.data_sections) > 1 else las.data_sections[0]
        # Pre-fix: "ACME SAND" dropped by the stale units gate →
        # DEPT=[2000.0, 2001.0] (2 rows).  Post-fix: consumed as data →
        # null-filled first row, 3 rows.
        np.testing.assert_allclose(sec.data["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(sec.data["GR"], [-999.25, 150.0, 155.0])

    def test_las30_a_to_unknown_empty_section_keeps_next_first_row(
        self, tmp_path: Path
    ) -> None:
        """A→unknown with a header+units-only empty prior section: the
        next section's letters-only first data row must be consumed."""
        content = self._numeric_body("  ~CUSTOMSECT\ncustomer\n")
        test_file = tmp_path / "m04fix3_a_unknown.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        self._assert_three_rows(las)

    def test_las30_a_to_unknown_empty_header_only_section_keeps_next_first_row(
        self, tmp_path: Path
    ) -> None:
        """A→unknown with a header-ONLY empty prior section (no units
        row): the units-gate one-shot clear never fires, so the leak can
        only be closed by the A→unknown boundary clear — the next
        section's letters-only first data row must still be consumed."""
        content = self._numeric_body("  ~CUSTOMSECT\ncustomer\n").replace(
            "DEPT GR\nM GAPI\n", "DEPT GR\n"
        )
        test_file = tmp_path / "m04fix3_a_unknown_no_units.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        self._assert_three_rows(las)

    def test_las30_a_to_a_consecutive_empty_section_keeps_next_first_row(
        self, tmp_path: Path
    ) -> None:
        """A→A consecutive data sections with a header+units-only empty
        first section: the second section's letters-only first data row
        must be consumed (flush entry clear)."""
        content = self._numeric_body("")
        test_file = tmp_path / "m04fix3_a_a.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        self._assert_three_rows(las)

    def test_las30_a_to_w_empty_section_keeps_next_first_row(self, tmp_path: Path) -> None:
        """A→W (known non-data section) with a header+units-only empty
        prior A section: the next A section's letters-only first data row
        must be consumed (flush entry clear)."""
        content = self._numeric_body(
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   2000.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
        )
        test_file = tmp_path / "m04fix3_a_w.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        self._assert_three_rows(las)

    def test_las30_a_to_c_empty_section_keeps_next_string_first_row(
        self, tmp_path: Path
    ) -> None:
        """A→C (known section re-scoping curves) with a header+units-only
        empty prior A section: the next A section's letters-only first
        data row (string curves) must be consumed."""
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
            "DEPT GR\n"
            "M GAPI\n"
            "~CURVE INFORMATION\n"
            " LITH.    :  Lithology {S}\n"
            " FORM.    :  Formation {S}\n"
            "~LOG\n"
            "ACME SAND\n"
            "SHALE GRN\n"
        )
        test_file = tmp_path / "m04fix3_a_c.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[-1]
        # Pre-fix: "ACME SAND" dropped by the stale units gate →
        # LITH=["SHALE"] only.  Post-fix: both rows consumed.
        np.testing.assert_array_equal(
            sec.string_data["LITH"], np.array(["ACME", "SHALE"], dtype=object)
        )
        np.testing.assert_array_equal(
            sec.string_data["FORM"], np.array(["SAND", "GRN"], dtype=object)
        )

    def test_las30_units_row_then_letters_only_data_row_same_section(
        self, tmp_path: Path
    ) -> None:
        """Within-section variant: a letters-only data row that follows
        the units row in the SAME section must be consumed — the units
        gate is one-shot (closes once the units row is skipped)."""
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
            "DEPT GR\n"
            "M GAPI\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
        )
        test_file = tmp_path / "m04fix3_within_section.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        # Pre-fix: "ACME SAND" dropped by the stale flag → DEPT=[2000.0].
        np.testing.assert_allclose(sec.data["DEPT"], [-999.25, 2000.0])
        np.testing.assert_allclose(sec.data["GR"], [-999.25, 150.0])

    def test_las30_a_to_eof_empty_section_no_leak_into_next_parse(self) -> None:
        """A→EOF control: a trailing header+units-only A section leaves
        the flag set at parse end (the EOF flush only clears it when
        data lines exist), but the next parse() on the SAME parser
        resets it (_reset) — no cross-file leak."""
        empty_tail = (
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
            "DEPT GR\n"
            "M GAPI\n"
        )
        parser = LASParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The header+units-only section accumulates no rows → no
            # DataSection is emitted; the flag stays set at parse end.
            las1 = parser.parse(empty_tail)
            assert las1.data_sections == []
            las2 = parser.parse(empty_tail + "1000.0 50.0\n1001.0 55.0\n")
        np.testing.assert_allclose(las2.data_sections[0].data["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las2.data_sections[0].data["GR"], [50.0, 55.0])


# ──────────────────────────────────────────────────────────────
# M-13 fix4 (reader, MEDIUM, CONFIRMED — F1): the reader's
# _mnemonic_header_skipped units-row position gate was SET when the
# standalone mnemonic header row was skipped but NEVER consumed
# one-shot — the gate stayed live at current_line == 0, silently
# dropping a genuine letters-only first DATA row that followed the
# header/units pair ("~A\nDEPT GR\nM GAPI\nACME SAND\n..." → DEPT=
# [2000.0, 2001.0] instead of [-999.25, 2000.0, 2001.0]; wrapped
# mode FULLY SILENT — no F-024 analog).  The gate is now closed
# one-shot at the units row AND when the first data row is consumed
# (parser fix3-P1 parity, parser.py:4498/:4503), in both
# _read_normal and _read_wrapped.  FAILS on pre-fix code, PASSES on
# post-fix.
# ──────────────────────────────────────────────────────────────


class TestM13Fix4ReaderStickyUnitsFlagOneShot:
    """fix4-F1: the reader's _mnemonic_header_skipped flag must be
    consumed one-shot — a letters-only first data row after the
    header/units pair is genuine data and must be preserved (LAS 1.2/
    2.0 reader paths, _read_normal and _read_wrapped)."""

    @staticmethod
    def _normal_content() -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )

    def test_normal_letters_only_first_data_row_preserved(self, tmp_path: Path) -> None:
        """WRAP=NO (_read_normal): a letters-only first data row after the
        header/units pair must be consumed (null-filled), not dropped as a
        units row — the units gate is one-shot.  Pre-fix DEPT=[2000.0,
        2001.0] ("ACME SAND" lost) and F-024 "Pre-scan overcount"
        fires; post-fix DEPT=[-999.25, 2000.0, 2001.0] with no F-024."""
        test_file = tmp_path / "m13fix4_normal.las"
        test_file.write_text(self._normal_content(), encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [-999.25, 150.0, 155.0])
        # M-02 pin: the parser pre-scan counts the letters row as a data
        # line (declared 3), so once the reader consumes all 3 no spurious
        # "Pre-scan overcount" (F-024) may fire (pre-fix it fired:
        # declared 3, actual 2).
        pre_scan_msgs = [
            str(w.message) for w in caught if "Pre-scan overcount" in str(w.message)
        ]
        assert not pre_scan_msgs, (
            f"F-024: spurious pre-scan overcount warning after fix4: {pre_scan_msgs}"
        )

    def test_wrapped_letters_only_first_data_row_preserved(self, tmp_path: Path) -> None:
        """WRAP=YES (_read_wrapped): identical defect — pre-fix the row
        was dropped with ZERO diagnostics (fully silent, no F-024 analog);
        post-fix it is consumed as a null-filled first step."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "ACME SAND\n"
            "2000.0\n"
            "150.0\n"
            "2001.0\n"
            "155.0\n"
        )
        test_file = tmp_path / "m13fix4_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [-999.25, 150.0, 155.0])


# ──────────────────────────────────────────────────────────────
# M-04 fix5 (parser deferred replay, MEDIUM, CONFIRMED — F-01):
# the deferred (pre-~V) replay filter's LOOP-LOCAL
# _mnemonic_header_skipped flag was SET when the standalone
# mnemonic header row was skipped but NEVER consumed one-shot —
# the gate stayed live, silently dropping a genuine letters-only
# first DATA row that followed the header/units pair
# ("~ASCII\nDEPT,GR\nM,GAPI\nACME,SAND\n2000.0,150.0\n..." →
# DEPT=[2000.0, 2001.0] instead of [-999.25, 2000.0, 2001.0];
# two letters rows → BOTH dropped) while the non-deferred twin
# preserved them.  The gate is now closed one-shot at the units
# row AND when the first data row is consumed
# (parser.py:3284-3290), mirroring parser.py:4498/:4503
# (fix3-P1) and data_reader.py:1074/:1082-1083, :1555/:1563-1564
# (fix4-F1).  FAILS on pre-fix code, PASSES on post-fix.
# ──────────────────────────────────────────────────────────────


class TestM04Fix5DeferredStickyUnitsFlagOneShot:
    """fix5-F-01: the deferred replay filter's _mnemonic_header_skipped
    local must be consumed one-shot — a letters-only first data row after
    the header/units pair is genuine data and must be preserved (LAS 3.0
    data-before-~V deferred path, PARS-04 replay filter)."""

    @staticmethod
    def _deferred_content() -> str:
        return (
            "~ASCII\n"
            "DEPT,GR\n"
            "M,GAPI\n"
            "ACME,SAND\n"
            "2000.0,150.0\n"
            "2001.0,155.0\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
        )

    @staticmethod
    def _non_deferred_content() -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~LOG\n"
            "DEPT,GR\n"
            "M,GAPI\n"
            "ACME,SAND\n"
            "2000.0,150.0\n"
            "2001.0,155.0\n"
        )

    def test_las30_pre_v_deferred_letters_only_first_data_row_preserved(
        self, tmp_path: Path
    ) -> None:
        """Deferred path (data before ~V): a letters-only first data row
        after the header/units pair must be consumed (null-filled), not
        dropped as a units row — the units gate is one-shot.  Pre-fix
        DEPT=[2000.0, 2001.0] ("ACME,SAND" lost); post-fix DEPT=
        [-999.25, 2000.0, 2001.0].  Parity: the non-deferred twin on
        identical content preserves the same rows."""
        test_file = tmp_path / "m04fix5_deferred.las"
        test_file.write_text(self._deferred_content(), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(sec.data["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(sec.data["GR"], [-999.25, 150.0, 155.0])
        # Parity: identical content with ~VERSION before the data section
        # (non-deferred twin) must preserve the same rows.
        twin_file = tmp_path / "m04fix5_non_deferred.las"
        twin_file.write_text(self._non_deferred_content(), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twin = read_las_file_as_object(twin_file)
        twin_sec = twin.data_sections[0]
        np.testing.assert_allclose(twin_sec.data["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(twin_sec.data["GR"], [-999.25, 150.0, 155.0])

    def test_las30_pre_v_deferred_two_letters_rows_preserved(
        self, tmp_path: Path
    ) -> None:
        """Deferred path: TWO letters-only first data rows after the
        header/units pair must BOTH be consumed (null-filled).  Pre-fix
        both rows were dropped (sticky gate — every subsequent
        letters-only row while _filtered_lines stays empty); post-fix
        DEPT=[-999.25, -999.25, 2000.0, 2001.0]."""
        content = self._deferred_content().replace(
            "ACME,SAND\n", "ACME,SAND\nSHALE,GRN\n", 1
        )
        test_file = tmp_path / "m04fix5_deferred_2l.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        sec = las.data_sections[0]
        np.testing.assert_allclose(
            sec.data["DEPT"], [-999.25, -999.25, 2000.0, 2001.0]
        )
        np.testing.assert_allclose(
            sec.data["GR"], [-999.25, -999.25, 150.0, 155.0]
        )

