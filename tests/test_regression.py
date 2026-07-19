"""Regression tests for verified findings from Stage 6 and Stage 9 fixes.

These tests exercise specific fixes identified by the adversarial verification
pipeline. Each test documents which finding it covers and tests the actual
fixed behaviour against the current (fixed) source.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pylasdev import write_las_file
from pylasdev.compare import compare_las_dicts
from pylasdev.models import (
    CurveDefinition,
    DataSection,
    LASFile,
    ParameterEntry,
    VersionSection,
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
        """G-001: from_dict with mnem_base normalises string_data dict keys.

        Provides log data for BOTH curves so from_dict's cross-validation
        of log keys against curves_order passes."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT", "AK"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "AK", "unit": "US/M"},
            ],
            "logs": {
                "DEPT": np.array([100.0]),
                "AK": np.array([50.0]),
            },
            "string_data": {"AK": np.array(["SAND"], dtype=np.str_)},
        }
        las = LASFile.from_dict(data, mnem_base={"AK": "DT"})
        assert "DT" in las.string_data, (
            f"Expected 'DT' key in string_data via mnem_base, "
            f"got: {list(las.string_data.keys())}"
        )


# ──────────────────────────────────────────────────────────────
# G-018 (writer, MEDIUM): version.wrap restored after write.
# ──────────────────────────────────────────────────────────────

class TestG018WrapPreservation:
    """G-018: Writer restores las_file.version.wrap to its pre-write value
    after writing completes, matching the G-018 save/restore pattern."""

    def test_wrap_yes_restored_after_write(self, tmp_path: Path) -> None:
        """G-018: Write with WRAP=YES — wrap restored after write.

        The writer overrides WRAP=YES to WRAP=NO during content generation
        (it cannot produce wrapped output).  After writing, the LASFile
        model's wrap attribute must be restored to the original value."""
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

        # After writing, the model's wrap value should be restored
        assert las.version.wrap == "YES", (
            f"Expected wrap='YES' after write, got wrap={las.version.wrap!r}"
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

    def test_wrap_non_default_restored(self, tmp_path: Path) -> None:
        """G-018: Write with non-default wrap=YES on LASFile.

        When using an LASFile with wrap=YES (non-default), writing
        should not permanently change the wrap attribute."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="YES", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])
        temp_file = tmp_path / "wrap_yes_las30.las"
        write_las_file(temp_file, las)
        assert las.version.wrap == "YES", (
            f"Expected wrap='YES' (restored) after write, "
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
