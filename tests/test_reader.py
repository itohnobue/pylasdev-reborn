"""Tests for LAS file reader."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import read_dev_file, read_las_file, read_las_file_as_object
from pylasdev.exceptions import LASParseError, LASReadError
from pylasdev.models import CurveDefinition, LASFile
from pylasdev.parser import LASParser


def _declared_vers(name: str) -> str:
    """Return the ``VERS.`` token literally declared in ``test_data/<name>``.

    Independent oracle for the corpus contract: the parsed version must equal
    what the file says, not merely be one of a few allowed strings (F-05 — the
    old allow-list ``["1.2", "1.20", "2.0", "3.0"]`` accepted any of them for
    every file, so a mis-parsed or silently normalized version stayed green).
    """
    text = (Path(__file__).resolve().parent.parent / "test_data" / name).read_text(
        encoding="utf-8", errors="replace"
    )
    for line in text.splitlines():
        match = re.match(r"\s*VERS\W*([^:\s]+)", line, re.IGNORECASE)
        if match:
            return match.group(1)
    raise AssertionError(f"{name}: no VERS line found in the fixture")


@pytest.fixture(scope="module")
def parsed_corpus() -> dict[str, dict[str, Any]]:
    """Parse every ``test_data/*.las`` fixture ONCE for the corpus contracts.

    Five separate tests used to re-read the whole corpus (5 x 18 parses); the
    merged contract test below shares one parse per fixture.
    """
    test_data = Path(__file__).resolve().parent.parent / "test_data"
    return {path.name: read_las_file(path) for path in sorted(test_data.glob("*.las"))}


class TestReadLASFile:
    """Tests for read_las_file function."""

    def test_corpus_read_contract(self, parsed_corpus: dict[str, dict[str, Any]]) -> None:
        """Every fixture reads into the documented dict contract.

        MERGED from the five former corpus fan-out tests (read-all-las-files,
        returns-numpy-arrays, preserves-curve-order, well-values-are-strings,
        version-is-valid): one parse per fixture, every contract asserted.
        """
        assert parsed_corpus, "No LAS test files found"

        for name, data in parsed_corpus.items():
            assert isinstance(data, dict), f"{name}: not a dict"
            assert set(data) >= {"version", "well", "logs", "curves_order"}, name

            # log data is returned as numpy arrays
            for curve, values in data["logs"].items():
                assert isinstance(values, np.ndarray), f"{name}: {curve} is not an ndarray"

            # equal-length invariant wherever the file actually has data
            sizes = {curve: len(values) for curve, values in data["logs"].items() if len(values)}
            assert len(set(sizes.values())) <= 1, f"{name}: unequal curve lengths {sizes}"

            # curve order matches log keys for non-3.0 files; duplicates in
            # curves_order collapse as dict keys, so those are skipped
            order = data["curves_order"]
            if order and not str(data["version"]["VERS"]).startswith("3"):
                if len(order) == len(set(order)):
                    assert list(data["logs"]) == order, f"{name}: curve order mismatch"

            # well values are strings (backward-compat contract)
            for key, value in data["well"].items():
                assert isinstance(value, str), (
                    f"{name}: well[{key}] is {type(value).__name__}, not str"
                )

            # declared VERS ↔ parsed VERS (F-05 independent oracle)
            declared = _declared_vers(name)
            assert data["version"]["VERS"] == declared, (
                f"{name}: parsed VERS {data['version']['VERS']!r} != declared {declared!r}"
            )

    # MERGED (register: test_file_not_found/test_not_a_file, DELETE): the dict
    # API is a thin wrapper — reader.py:280-288 calls read_las_file_as_object
    # and only converts the result with to_dict() — so the error contract is
    # covered once, on the object API, where the exact message is also pinned
    # (TestReadLASFileAsObject::test_file_not_found / test_not_a_file).

    def test_sample_las_specific_values(self, test_data_dir: Path) -> None:
        """Test specific values from sample.las."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        data = read_las_file(sample)
        assert "DEPT" in data["logs"]
        assert len(data["logs"]["DEPT"]) == 3
        # Verify specific data values from sample.las
        np.testing.assert_array_almost_equal(
            data["logs"]["DEPT"], np.array([1670.0, 1669.875, 1669.75])
        )
        # Verify well section values (LAS 1.2: data is after colon, not before)
        # COMP.   COMPANY:   # ANY OIL COMPANY LTD. → value is "# ANY OIL COMPANY LTD."
        assert data["well"]["COMP"] == "# ANY OIL COMPANY LTD."
        # WELL.   WELL:   ANY ET AL OIL WELL #12 → value is "ANY ET AL OIL WELL #12"
        assert data["well"]["WELL"] == "ANY ET AL OIL WELL #12"
        assert data["well"]["STRT"] == "1670.000000"
        # Verify parameter values
        assert data["parameters"]["BHT"] == "35.5000"
        assert data["parameters"]["BS"] == "200.0000"

    def test_wrapped_file_correct_shape(self, test_data_dir: Path) -> None:
        """Wrapped fixtures parse into equal-length arrays with real values.

        HARDENED (register F-01): the former assertion was equal-length only,
        so a wrapped file that produced 30 all-null rows passed.  Each fixture
        now pins its own depth values and asserts the depth column carries no
        null sentinel — a null-filled parse fails here.
        """
        expected = {
            "sample_wrapped.las": [910.0, 909.875, 909.75, 909.625, 909.5],
            "sample_2.0_wrapped.las": [910.0, 909.875],
        }
        for name, expected_dept in expected.items():
            wf = test_data_dir / name
            assert wf.exists(), f"Required test data missing: {wf}"
            data = read_las_file(wf)
            sizes = [len(data["logs"][c]) for c in data["curves_order"] if c in data["logs"]]
            assert len(set(sizes)) == 1, f"Arrays have different sizes in {wf.name}: {sizes}"
            np.testing.assert_allclose(data["logs"]["DEPT"], expected_dept, err_msg=wf.name)
            assert not np.any(np.asarray(data["logs"]["DEPT"]) == -999.25), (
                f"{wf.name}: depth column null-filled — the wrapped rows were not parsed"
            )

    def test_mislabeled_wrap_handled(self, test_data_dir: Path) -> None:
        """A WRAP=YES file whose rows are already full is read non-wrapped.

        HARDENED (register F-02): the former test only asserted equal lengths,
        which comment_test.las satisfies under BOTH routings (its DEPT is
        identical either way), so it could not see a routing regression.  The
        detector verdict is now asserted directly plus the first depth rows.
        """
        from pylasdev.data_reader import _detect_actual_wrap

        ct = test_data_dir / "comment_test.las"
        assert ct.exists(), f"Required test data missing: {ct}"
        lines = ct.read_text(encoding="utf-8", errors="replace").splitlines()
        ascii_start = next(
            i for i, line in enumerate(lines) if line.strip().upper().startswith("~A")
        )
        data = read_las_file(ct)
        n_curves = len(data["curves_order"])
        assert n_curves == 10, f"unexpected curve count in comment_test.las: {n_curves}"
        # WRAP=YES is mislabeled: every data line already carries all columns.
        assert _detect_actual_wrap(lines[ascii_start:], n_curves, " ") is False, (
            "comment_test.las must be detected as NON-wrapped despite WRAP=YES"
        )
        sizes = [len(data["logs"][c]) for c in data["curves_order"]]
        assert len(set(sizes)) == 1, f"Arrays have different sizes: {sizes}"
        # comment_test.las names its index curve DEPTXXXX (not DEPT) — assert the
        # first depth-like curve's own values, which only align when the file is
        # read non-wrapped.
        np.testing.assert_allclose(
            data["logs"][data["curves_order"][0]][:3], [2736.0, 2736.2, 2736.4]
        )

    def test_wrapped_string_curve_preserved(self, tmp_path: Path) -> None:
        """F-R-03: String curve values in wrapped LAS files are preserved.

        Before this fix, _read_wrapped converted all values through
        _to_finite_float(), silently converting string values to the
        null value (-999.25).  String curves ({S} format in ~C section)
        must be stored in string_data, not converted to float.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth {F}\n"
            " GR .API  :  Gamma Ray {F}\n"
            " LITH .   :  Lithology {S}\n"
            "~A\n"
            "1000.0\n"
            "50.0  Sandstone\n"
            "1001.0\n"
            "51.0  Shale\n"
        )
        test_file = tmp_path / "wrapped_string.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # Float curves should be parsed normally
        assert "DEPT" in data["logs"]
        assert "GR" in data["logs"]
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

        # String curve must be in string_data, NOT silently converted
        assert "LITH" in data["string_data"], (
            "String curve 'LITH' missing from string_data — values "
            "were likely converted to null_value by _to_finite_float"
        )
        lith_values = data["string_data"]["LITH"]
        assert len(lith_values) == 2, f"Expected 2 LITH values, got {len(lith_values)}"
        np.testing.assert_array_equal(
            lith_values,
            np.array(["Sandstone", "Shale"], dtype=np.str_),
        )

        # String curve should NOT be in float logs
        assert "LITH" not in data["logs"], (
            "String curve 'LITH' found in logs (float) — should only be in string_data"
        )

    # --- F-H-006: string curve multi-char value preservation ---

    def test_string_curve_multi_char_values_preserved(self, tmp_path: Path) -> None:
        """F-H-006: String curve values > 1 char are preserved in string_data.

        Before the fix at _las30_data.py (dtype=object string arrays),
        string curve arrays were pre-allocated with dtype=np.str_ which
        defaults to a single-character fixed-width Unicode type (U1),
        truncating values to their first character.  After the fix,
        dtype=object preserves arbitrary-length strings.

        This test uses a LAS 3.0 file with {S}-format curve containing
        multi-character lithology names like "Sandstone" and "Limestone".
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
            " LITH .       :  Lithology {S}\n"
            "~ASCII\n"
            " 100.0,Sandstone\n"
            " 200.0,Limestone\n"
            " 300.0,Dolomite\n"
        )
        test_file = tmp_path / "string_curve_multi_char.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        # String curve must be in string_data
        assert "LITH" in data["string_data"], "String curve 'LITH' missing from string_data"
        lith_values = data["string_data"]["LITH"]
        assert len(lith_values) == 3, f"Expected 3 LITH values, got {len(lith_values)}"
        np.testing.assert_array_equal(
            lith_values,
            np.array(["Sandstone", "Limestone", "Dolomite"]),
        )
        # Verify no single-char truncation (before the data_reader fix,
        # dtype=np.str_ pre-allocation at U1 would truncate multi-char values
        # to their first character — "Sandstone" → "S").
        # For LAS 3.0 parser path, np.str_ auto-sizes to fit longest value,
        # so dtype may be U9 or similar.  The critical check is that values
        # are full length, validated by assert_array_equal above.

        # Float curve should be correct
        assert "DEPT" in data["logs"]
        np.testing.assert_array_equal(
            data["logs"]["DEPT"],
            np.array([100.0, 200.0, 300.0]),
        )

    def test_encoding_parameter(self, test_data_dir: Path, tmp_path: Path) -> None:
        """An explicit ``encoding=`` really governs decoding.

        KEPT + HARDENED (register DISPUTED-DELETE resolved conservatively): the
        former body asserted only ``"logs" in data``, which passes with the
        parameter ignored entirely.  The explicit encoding is now shown to
        change the decoded value, and a wrong explicit encoding raises
        LASEncodingError instead of falling back to auto-detection.
        """
        from pylasdev.exceptions import LASEncodingError

        # Baseline: an explicit encoding on an ASCII fixture still reads.
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        assert "logs" in read_las_file(sample, encoding="utf-8")

        # A cp1252-only well value decodes to the accented text with the right
        # explicit encoding ...
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " WELL.    Caf\u00e9 Well : WELL NAME\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        cp1252_file = tmp_path / "explicit_encoding.las"
        cp1252_file.write_bytes(content.encode("cp1252"))
        assert read_las_file(cp1252_file, encoding="cp1252")["well"]["WELL"] == "Caf\u00e9 Well"

        # ... and a wrong explicit encoding must fail loudly rather than be
        # silently replaced by auto-detection.
        with pytest.raises(LASEncodingError, match="Failed to decode"):
            read_las_file(cp1252_file, encoding="utf-8")


class TestReadLASFileAsObject:
    """Tests for read_las_file_as_object function."""

    def test_returns_las_file_object(self, test_data_dir: Path) -> None:
        """Test that read_las_file_as_object returns LASFile."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        assert isinstance(las, LASFile)
        assert las.source_file != ""
        assert las.encoding != ""

    def test_object_has_curves(self, test_data_dir: Path) -> None:
        """Test LASFile object has curve definitions."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        assert len(las.curves) > 0
        assert len(las.curves_order) > 0
        assert len(las.logs) > 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error for missing file."""
        with pytest.raises(LASReadError):
            read_las_file_as_object(tmp_path / "missing.las")

    def test_not_a_file(self, tmp_path: Path) -> None:
        """Test error for directory path."""
        with pytest.raises(LASReadError, match="Not a file"):
            read_las_file_as_object(tmp_path)

    # --- F22: LASParseError from parser re-raised with file path ---
    def test_parse_error_re_raises_with_filename(self, tmp_path: Path) -> None:
        """Test that LASParseError from parser is re-raised with file path.

        Exercises reader.py:122-123 — when parser.parse() raises LASParseError
        (e.g., missing required ~V section), the reader catches it and re-raises
        a new LASParseError that includes the file path in the message.
        """
        content = "~W WELL.NAME: Test Well\n~A\n1.0 2.0"
        test_file = tmp_path / "no_version.las"
        test_file.write_text(content, encoding="utf-8")

        with pytest.raises(LASParseError, match="Error reading"):
            read_las_file_as_object(test_file)

    def test_las30_object_has_version(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 file parsed as object has correct version."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        assert las.version.vers == "3.0"
        assert las.version.is_las30 is True
        assert las.version.dlm == "COMMA"

    def test_las30_curves_with_formats(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 curve format specifiers are parsed."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        # Check that format specifiers were extracted
        dept_curve = las.get_curve_by_mnemonic("DEPT")
        assert dept_curve is not None
        assert dept_curve.data_format == "F"
        # String format
        cdes_curve = las.get_curve_by_mnemonic("CDES")
        assert cdes_curve is not None
        assert cdes_curve.data_format == "S"

    def test_las30_array_curves(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 array notation curves are parsed."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        nmr_curves = las.get_array_curves("NMR")
        assert len(nmr_curves) == 5
        assert nmr_curves[0].array_info is not None
        assert nmr_curves[0].array_info.time_offset == 0.0
        assert nmr_curves[4].array_info is not None
        assert nmr_curves[4].array_info.time_offset == 20.0

    def test_las30_parameters_with_zones(self, test_data_dir: Path) -> None:
        """LAS 3.0 parameter zone associations are resolved per parameter.

        HARDENED (register F-10): the former body asserted only ``len(zoned) >
        0``.  The exact zoned population and three specific
        (mnemonic, zone name, zone index) triples are now pinned, so a parser
        that attaches every parameter to the wrong zone fails.
        """
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        zoned = [p for p in las.parameters if p.zone is not None]
        assert len(las.parameters) == 71
        assert len(zoned) == 64, f"expected 64 zone-associated parameters, got {len(zoned)}"
        assert {p.zone.zone_name for p in zoned} == {"RUN", "NMAT_DEPTH", "DMAT_DEPTH"}
        triples = {(p.mnemonic, p.zone.zone_name, p.zone.zone_index) for p in zoned}
        for expected in (
            ("MATR", "NMAT_DEPTH", 1),
            ("MATR", "NMAT_DEPTH", 2),
            ("MDEN", "DMAT_DEPTH", 1),
            ("RUN_DEPTH", "RUN", 2),
        ):
            assert expected in triples, f"{expected} missing from {sorted(triples)[:5]}…"

    def test_las30_scientific_notation_format(self, test_data_dir: Path) -> None:
        """Test LAS 3.0 {E} (scientific notation) format specifier."""
        las30 = test_data_dir / "sample_3.0.las"
        assert las30.exists(), f"Required test data missing: {las30}"
        las = read_las_file_as_object(las30)
        yme_curve = las.get_curve_by_mnemonic("YME")
        assert yme_curve is not None
        assert yme_curve.data_format == "E"


class TestAPICodeParsing:
    """Tests for API code extraction from curve definitions."""

    def test_api_codes_parsed_from_las12(self, test_data_dir: Path) -> None:
        """Test API codes are extracted from LAS 1.2 curve_api file."""
        api_file = test_data_dir / "sample_curve_api.las"
        assert api_file.exists(), f"Required test data missing: {api_file}"
        las = read_las_file_as_object(api_file)
        rhob = las.get_curve_by_mnemonic("RHOB")
        assert rhob is not None
        assert rhob.api_code == "7 350 02 00"

    def test_api_codes_parsed_from_las20(self, test_data_dir: Path) -> None:
        """Test API codes are extracted from LAS 2.0 file."""
        las20 = test_data_dir / "sample_2.0.las"
        assert las20.exists(), f"Required test data missing: {las20}"
        las = read_las_file_as_object(las20)
        dt = las.get_curve_by_mnemonic("DT")
        assert dt is not None
        assert dt.api_code == "60 520 32 00"

    def test_empty_api_code_for_curves_without_it(self, test_data_dir: Path) -> None:
        """Test that curves without API codes have empty api_code."""
        sample = test_data_dir / "sample.las"
        assert sample.exists(), f"Required test data missing: {sample}"
        las = read_las_file_as_object(sample)
        dept = las.get_curve_by_mnemonic("DEPT")
        assert dept is not None
        assert dept.api_code == ""

    def test_api_code_roundtrip(self, test_data_dir: Path, tmp_path: Path) -> None:
        """Test API codes survive a write-then-read roundtrip."""
        from pylasdev import write_las_file

        api_file = test_data_dir / "sample_curve_api.las"
        assert api_file.exists(), f"Required test data missing: {api_file}"
        las = read_las_file_as_object(api_file)
        temp_file = tmp_path / "api_roundtrip.las"
        write_las_file(temp_file, las)
        las2 = read_las_file_as_object(temp_file)
        rhob = las2.get_curve_by_mnemonic("RHOB")
        assert rhob is not None
        assert rhob.api_code == "7 350 02 00"


class TestDataReaderEdgeCases:
    """Tests for data reader edge cases and boundary conditions."""

    def test_missing_values_filled_with_null(self, tmp_path: Path) -> None:
        """Test that short data lines fill remaining curves with null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A  DEPT  DT  GR\n"
            "100.0  50.0  75.0\n"
            "101.0  51.0\n"
            "102.0\n"
        )
        test_file = tmp_path / "short_lines.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Line 1: all values present
        assert data["logs"]["GR"][0] == 75.0
        # Line 2: GR missing -> should be null_value, NOT 0.0
        assert data["logs"]["GR"][1] == -999.25
        # Line 3: DT and GR missing -> both should be null_value
        assert data["logs"]["DT"][2] == -999.25
        assert data["logs"]["GR"][2] == -999.25

    def test_section_after_ascii_not_parsed_as_data(self, tmp_path: Path) -> None:
        """Test that sections appearing after ~A don't corrupt data."""
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
            "Some free text after data.\n"
        )
        test_file = tmp_path / "section_after_a.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Should have exactly 2 data points, not 3+
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])

    def test_duplicate_curve_names_renamed_with_warning(self, tmp_path: Path) -> None:
        """Test that duplicate curve mnemonics are renamed with a warning."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray 1\n"
            " GR.GAPI  :  Gamma Ray 2\n"
            "~A  DEPT  GR  GR\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "dup_curves.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Duplicate curve mnemonic" in str(x.message) for x in w)

        # Second GR should be renamed to GR_2
        assert "GR" in data["logs"]
        assert "GR_2" in data["logs"]
        np.testing.assert_array_equal(data["logs"]["GR"], [10.0, 11.0])
        np.testing.assert_array_equal(data["logs"]["GR_2"], [20.0, 21.0])

    def test_duplicate_curves_metadata_synced(self, tmp_path: Path) -> None:
        """Test that CurveDefinition objects are synced with renamed curves_order."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray 1\n"
            " GR.GAPI  :  Gamma Ray 2\n"
            "~A  DEPT  GR  GR\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "dup_meta.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)

        # curves_order and curves mnemonics must match
        assert las.curves_order == ["DEPT", "GR", "GR_2"]
        assert [c.mnemonic for c in las.curves] == ["DEPT", "GR", "GR_2"]

        # get_curve_by_mnemonic must find renamed curve
        gr2 = las.get_curve_by_mnemonic("GR_2")
        assert gr2 is not None
        assert gr2.unit == "GAPI"
        assert gr2.original_mnemonic == "GR"

        # to_dict should have consistent curves and curves_order
        d = las.to_dict()
        dict_mnemonics = [c["mnemonic"] for c in d["curves"]]
        assert dict_mnemonics == d["curves_order"]

    @pytest.mark.parametrize("vers", ["4.0", "5.0"])
    def test_unsupported_version_warns_but_reads(self, tmp_path: Path, vers: str) -> None:
        """An unsupported VERS warns once per read and still parses.

        MERGED (register): the dict-API test (VERS 4.0) and the object-API twin
        (VERS 5.0) asserted the same single warning on the same reader path
        (reader.py:203-209); both version literals and both API entry points are
        kept in one test.
        """
        content = (
            "~VERSION INFORMATION\n"
            f" VERS.   {vers}  : FUTURE VERSION\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / f"unsupported_{vers}.las"
        test_file.write_text(content, encoding="utf-8")

        for api in (read_las_file, read_las_file_as_object):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = api(test_file)
            unsupported = [
                str(x.message) for x in w if "not officially supported" in str(x.message)
            ]
            assert len(unsupported) == 1, (
                f"{api.__name__}/VERS {vers}: expected exactly one unsupported-version "
                f"warning, got {unsupported}"
            )
            assert f"LAS version {vers} is not officially supported" in unsupported[0]
            logs = result["logs"] if isinstance(result, dict) else result.logs
            assert logs["DEPT"][0] == 100.0, f"{api.__name__}: data must still be read"

    # --- T6: Non-numeric VERS branch (reader.py:129-134) ---
    def test_non_numeric_vers(self, tmp_path: Path) -> None:
        """A non-numeric VERS reads without error and keeps the raw text.

        MERGED (register): the dict-API test and its object-API twin exercised
        the same ValueError handler (reader.py:210-215); the stronger
        ``version.vers`` preservation assertion from the object twin is kept
        and applied to both entry points.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   CWLS LOG ASCII STANDARD  : VERSION 1.2\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "non_numeric_vers.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["version"]["VERS"] == "CWLS LOG ASCII STANDARD"
        assert data["logs"]["DEPT"][0] == 100.0

        las = read_las_file_as_object(test_file)
        assert las.version.vers == "CWLS LOG ASCII STANDARD"
        assert las.logs["DEPT"][0] == 100.0

    def test_max_file_size_limit(self, tmp_path: Path) -> None:
        """Both directions of the max_file_size boundary (exclusive cap).

        HARDENED + MERGED (register): the former body used 10 MB vs 10 bytes,
        which never exercised the boundary, and TestENC03's twin asserted the
        identical exception with the identical message.  The at-limit case
        (limit == file size) must be ACCEPTED and one byte less REJECTED, and
        the raised exception must be LASReadError carrying the file path.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "size_test.las"
        test_file.write_text(content, encoding="utf-8")
        size = test_file.stat().st_size

        # At the limit (and above) the file is accepted.
        assert read_las_file(test_file, max_file_size=size)["logs"]["DEPT"][0] == 100.0
        assert read_las_file(test_file, max_file_size=size + 1)["logs"]["DEPT"][0] == 100.0

        # One byte below the limit is rejected with the documented exception.
        with pytest.raises(LASReadError, match="Cannot read file") as exc_info:
            read_las_file(test_file, max_file_size=size - 1)
        assert str(test_file) in str(exc_info.value), (
            f"the failure must name the file: {exc_info.value}"
        )

    # --- TEST-02: Non-numeric data triggers ValueError handler in _read_normal ---
    def test_non_numeric_data_normal_mode(self, tmp_path: Path) -> None:
        """Test that non-numeric values in normal mode trigger null_value substitution."""
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
            "100.0  BAD\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "bad_data.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # BAD should be replaced with null_value (-999.25)
        assert data["logs"]["DT"][0] == -999.25
        # DEPT is fine
        assert data["logs"]["DEPT"][0] == 100.0
        # Next row should be fine
        assert data["logs"]["DT"][1] == 51.0

    # --- TEST-03: Wrapped-mode incomplete depth step padding ---
    def test_wrapped_incomplete_step_padding(self, tmp_path: Path) -> None:
        """Wrapped-mode incomplete final step (accumulation contract).

        II-4/R-6: under the n_curves-accumulation rewrite, a trailing
        partial buffer (fewer than curve_count values at EOF) cannot form
        a complete step — the orphan values are DISCARDED with the
        N-I-08-style "not accounted for" warning instead of being padded
        into a phantom step.  Only the complete first step survives; all
        arrays stay equal-length (the equal-length invariant is preserved
        for the steps that exist)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            # Only 2 values for the second step → incomplete trailing buffer
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_short.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
            # The N-I-08-style trailing-step warning replaces the old
            # per-curve "Padding" warnings (the partial step is discarded,
            # not padded).
            assert any("under-filled" in m or "not accounted for" in m for m in messages), (
                f"Expected trailing-step warning, got: {messages[-3:]}"
            )

        # All arrays should be same length after trimming
        sizes = [len(data["logs"][c]) for c in data["curves_order"]]
        assert len(set(sizes)) == 1
        # Only the complete first step survives
        assert len(data["logs"]["DEPT"]) == 1
        assert data["logs"]["DEPT"][0] == 100.0

    # --- TEST-04: Wrapped-mode depth line has >1 value ---
    def test_wrapped_depth_line_extra_values(self, tmp_path: Path) -> None:
        """Wrapped mode with a multi-value "depth" line (accumulation).

        II-4/R-6 (accepted accumulation contract): the depth-line flag
        protocol's "depth line has 2 values → warn+discard the extra"
        behavior is gone — _read_wrapped now uses an n_curves
        pending-value buffer, so `101.0 99.0` contributes BOTH values to
        the step stream: step 2 = [101.0, 99.0] (DT=99.0, not 51.0) and
        the trailing `51.0` is an incomplete final step that is discarded
        with the N-I-08-style "not accounted for" warning.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            # First depth line: 1 value (< curve_count) -> _detect_actual_wrap returns True
            "100.0\n"
            # DT for first step
            "50.0\n"
            # Second "depth" line: 2 values -> under accumulation both are data
            "101.0  99.0\n"
            # Trailing single value -> incomplete final step, discarded
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_extra_depth.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
            # The depth-line warn+discard is gone; the trailing incomplete
            # step emits the N-I-08-style warning instead.
            assert any("under-filled" in m or "not accounted for" in m for m in messages), (
                f"Expected trailing-step warning, got: {messages[-3:]}"
            )

        # DEPT gets the two flushed steps' depth values (100.0, 101.0)
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DEPT"][1] == 101.0
        # DT: step 2 carries 99.0 (the "extra" depth-line value is now data)
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DT"][1] == 99.0

    # --- TEST-11: Zero-curve early return ---
    def test_zero_curves_early_return(self, tmp_path: Path) -> None:
        """A file with no ~C definitions returns empty logs AND warns.

        HARDENED (register F32): the data-discard branch was silent in the
        assertions — only the empty result was pinned, so a read that dropped
        the rows without a diagnostic stayed green.  The discard warning is
        now asserted alongside the empty result.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            "~A\n"
            "100.0\n"
        )
        test_file = tmp_path / "no_curves.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        assert data["logs"] == {}
        assert data["curves_order"] == []
        discard = [str(x.message) for x in caught if "no curves are defined" in str(x.message)]
        assert len(discard) == 1, (
            f"the dropped ~A rows must be announced exactly once, got {discard}"
        )

    # --- TEST-12: _detect_actual_wrap with no data after ~A ---
    def test_detect_wrap_no_data(self, tmp_path: Path) -> None:
        """_detect_actual_wrap returns True when no data lines exist.

        HARDENED (register F9): the docstring claimed a lock the assertions did
        not provide — the structural dict checks would pass whatever the
        detector decided.  The detector verdict itself is now asserted, so the
        documented "defaults to wrapped when there is nothing to classify"
        contract is pinned.
        """
        from pylasdev.data_reader import _detect_actual_wrap

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A\n"
            "# Just a comment, no data\n"
        )
        test_file = tmp_path / "wrap_no_data.las"
        test_file.write_text(content, encoding="utf-8")

        # No data line to classify → the detector defaults to wrapped.
        assert _detect_actual_wrap(["~A", "# Just a comment, no data"], 1, " ") is True

        data = read_las_file(test_file)
        assert isinstance(data, dict)
        # F-M-21: the returned dict keeps the documented structure.
        assert "version" in data
        assert "well" in data
        assert "logs" in data
        assert "curves_order" in data
        # No data lines were parsed — logs contain pre-allocated empty arrays.
        assert data["logs"] != {}
        assert len(data["logs"]["DEPT"]) == 0
        # curves_order should still list the declared curve
        assert data["curves_order"] == ["DEPT"]
        # Version section must reflect WRAP=YES
        assert data["version"]["VERS"] == "1.2"
        assert data["version"]["WRAP"] == "YES"
        # Well section should contain the NULL value
        assert data["well"]["NULL"] == "-999.25"

    # --- TEST-13: Wrapped-mode malformed data handling ---
    def test_wrapped_malformed_data_handlers(self, tmp_path: Path) -> None:
        """Non-numeric / non-finite values in wrapped-mode data are
        substituted with null_value."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            # Depth line with non-numeric value (ValueError)
            "BAD\n"
            "50.0\n"
            "75.0\n"
        )
        test_file = tmp_path / "wrapped_bad.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # BAD in depth should become null_value
        assert data["logs"]["DEPT"][0] == -999.25
        # DT and GR should be fine
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["GR"][0] == 75.0

    # --- TEST-03: Non-numeric NULL value fallback in data_reader.py (lines 124-125, 189-190) ---
    def test_non_numeric_null_fallback_normal(self, tmp_path: Path) -> None:
        """A non-numeric NULL falls back to the -999.25 sentinel in normal mode.

        HARDENED (register F-03): the former fixture had no null-producing
        value, so the fallback was never observed — any sentinel would have
        passed.  A non-numeric data value now shows the fallback sentinel in
        the data while the well entry keeps its raw text.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   NONE  : NON-NUMERIC NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  BAD\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "non_numeric_null.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DT"][0] == -999.25, "fallback sentinel not applied"
        assert data["logs"]["DT"][1] == 51.0
        # NULL value itself should be in the well section as string "NONE"
        assert data["well"]["NULL"] == "NONE"

    def test_non_numeric_null_fallback_wrapped(self, tmp_path: Path) -> None:
        """A non-numeric NULL falls back to -999.25 in wrapped mode too.

        HARDENED (register F-03, same class): the fallback sentinel is now
        observed on the wrapped path via a non-numeric data value.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.   BADNULL  : NON-NUMERIC NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "BAD\n"
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "wrapped_non_numeric_null.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DEPT"][1] == 101.0
        assert data["logs"]["DT"][0] == -999.25, "fallback sentinel not applied (wrapped)"
        assert data["logs"]["DT"][1] == 51.0
        assert data["well"]["NULL"] == "NON-NUMERIC NULL VALUE"

    # --- F-19: Wrapped-mode section transition break ---
    def test_wrapped_section_after_ascii_stops_data(self, tmp_path: Path) -> None:
        """Test that a new section after ~A in wrapped mode stops data collection.

        Exercises the break in _read_wrapped when a section
        other than ~A is encountered during ASCII data reading.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "101.0\n"
            "51.0\n"
            "~OTHER\n"
            "Some free text after data.\n"
        )
        test_file = tmp_path / "wrapped_section_break.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        # Should have 2 data points per curve (stopped before ~OTHER)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])

    # --- F-22: Curve deduplication name collision ---
    def test_curve_dedup_already_has_suffix(self, tmp_path: Path) -> None:
        """Test curve deduplication when curves already have _N suffix.

        Exercises data_reader.py:86 — name collision edge case where
        curves already use _2 suffix notation.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " GR_2.API  :  Pre-existing suffix 1\n"
            " GR_2.API  :  Pre-existing suffix 2\n"
            "~A  GR_2  GR_2\n"
            "100.0  10.0\n"
        )
        test_file = tmp_path / "already_suffixed.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Duplicate curve mnemonic" in str(x.message) for x in w)

        # First GR_2 keeps its name, second becomes GR_2_2
        assert "GR_2" in data["logs"]
        assert "GR_2_2" in data["logs"]
        np.testing.assert_array_equal(data["logs"]["GR_2"], [100.0])
        np.testing.assert_array_equal(data["logs"]["GR_2_2"], [10.0])

    # --- T7: Cross-base collision while-loop (data_reader.py:149-165) ---
    def test_cross_base_collision_dedup(self, tmp_path: Path) -> None:
        """Test dedup when original name matches a previously generated suffix.

        Input curves_order: ["DEPT", "DEPT", "DEPT_2"]
        Expected: ["DEPT", "DEPT_2", "DEPT_2_2"] — the second DEPT becomes
        DEPT_2 which collides with the third original name DEPT_2,
        so the third gets renamed to DEPT_2_2.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth 1\n"
            " DEPT.M   :  Depth 2\n"
            " DEPT_2.M :  Pre-named curve\n"
            "~A  DEPT  DEPT  DEPT_2\n"
            "100.0  10.0  20.0\n"
            "101.0  11.0  21.0\n"
        )
        test_file = tmp_path / "cross_base_collision.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            # Should have 2 warnings: one for duplicate DEPT, one for cross-base
            dup_warnings = [x for x in w if "Duplicate curve mnemonic" in str(x.message)]
            assert len(dup_warnings) >= 2

        # Expected order: DEPT, DEPT_2, DEPT_2_2
        assert "DEPT" in data["logs"]
        assert "DEPT_2" in data["logs"]
        assert "DEPT_2_2" in data["logs"]
        # All three must be distinct keys
        assert len(data["logs"]) == 3
        # Verify data values
        np.testing.assert_array_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(data["logs"]["DEPT_2"], [10.0, 11.0])
        np.testing.assert_array_equal(data["logs"]["DEPT_2_2"], [20.0, 21.0])


class TestMaxTotalElementsGuard:
    """F-023: MAX_TOTAL_ELEMENTS allocation guard tests."""

    def test_max_total_elements_normal_mode(self, tmp_path: Path) -> None:
        """Test that curve_count * data_line_count > MAX_TOTAL_ELEMENTS raises error."""
        from unittest import mock

        # Build a file with 5 curves and 5 data lines
        curves_block = "\n".join(f" C{i:02d}.UNIT  :  Curve {i}" for i in range(5))
        # 5 data lines to get 5*5=25 elements
        data_block = "\n".join(f"{100.0 + i}  1.0  2.0  3.0  4.0" for i in range(5))
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n" + curves_block + "\n~A" + "  C00" * 5 + "\n" + data_block + "\n"
        )
        test_file = tmp_path / "max_total.las"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_TOTAL_ELEMENTS to 20: 5 curves * 5 lines = 25 > 20
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 20):
            with pytest.raises(LASParseError, match="Total allocation"):
                read_las_file(test_file)

    def test_max_total_elements_wrapped_mode(self, tmp_path: Path) -> None:
        """Test MAX_TOTAL_ELEMENTS in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " C00.M  :  Curve 0\n"
            " C01.M  :  Curve 1\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "101.0\n"
            "51.0\n"
            "102.0\n"
            "52.0\n"
        )
        test_file = tmp_path / "max_total_wrap.las"
        test_file.write_text(content, encoding="utf-8")

        # Monkey-patch MAX_TOTAL_ELEMENTS to a low value
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 5):
            with pytest.raises(LASParseError, match="Total allocation"):
                read_las_file(test_file)


class TestPermissionErrorHandler:
    """F-024: PermissionError handler tests."""

    def test_unreadable_file_raises_las_read_error(self, tmp_path: Path) -> None:
        """Test that an unreadable file raises LASReadError, not PermissionError."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "noperm.las"
        test_file.write_text(content, encoding="utf-8")

        try:
            # Make file unreadable
            os.chmod(test_file, 0o000)
            # KEPT + HARDENED (register DISPUTED-DELETE resolved conservatively):
            # the contract is "LASReadError, NOT PermissionError/OSError", and the
            # message must name the file — an assertion on the type alone cannot
            # see a leaked OSError or a message that loses the path.
            with pytest.raises(LASReadError, match="Cannot read") as exc_info:
                read_las_file(test_file)
            assert not isinstance(exc_info.value, OSError), (
                f"an OS-level error leaked through the documented exception: {exc_info.value!r}"
            )
            assert str(test_file) in str(exc_info.value), (
                f"the failure must name the unreadable file: {exc_info.value}"
            )
        finally:
            # Restore permissions so tmp_path cleanup works
            os.chmod(test_file, 0o644)

    def test_unreadable_file_as_object(self, tmp_path: Path) -> None:
        """Test read_las_file_as_object with unreadable file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "noperm_obj.las"
        test_file.write_text(content, encoding="utf-8")

        try:
            os.chmod(test_file, 0o000)
            with pytest.raises(LASReadError, match="Cannot read"):
                read_las_file_as_object(test_file)
        finally:
            os.chmod(test_file, 0o644)


class TestFortranDExponent:
    """T2/F-11: Fortran D-exponent format (e.g., '1.0D+03') in data."""

    def test_d_exponent_values_parsed_correctly(self, tmp_path: Path) -> None:
        """Test that Fortran D-exponent notation is converted via _to_finite_float.

        Some scientific software writes numbers in D-exponent format
        (e.g., '1.0D+03' instead of '1.0E+03').  _to_finite_float
        replaces D/d with E/e before calling float().
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
            " GR.GAPI  :  Gamma\n"
            "~A  DEPT  DT  GR\n"
            "100.0  1.0D+03  75.5\n"
            "101.0  2.5D-01  76.0\n"
            "102.0  3.14D0   77.0\n"
        )
        test_file = tmp_path / "d_exponent.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DT"], [1000.0, 0.25, 3.14], rtol=1e-10)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.5, 76.0, 77.0])

    def test_lowercase_d_exponent(self, tmp_path: Path) -> None:
        """Test that lowercase 'd' exponent is also handled."""
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
            "100.0  1.0d+03\n"
        )
        test_file = tmp_path / "d_lower.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == 1000.0

    def test_mixed_case_d_exponent_in_wrapped_mode(self, tmp_path: Path) -> None:
        """Test D-exponent in wrapped mode."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "1.0D+03\n"
            "101.0\n"
            "5.0d+02\n"
        )
        test_file = tmp_path / "d_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == 1000.0
        assert data["logs"]["DT"][1] == 500.0

    def test_d_exponent_null_value_different_value(self, tmp_path: Path) -> None:
        """F-06: Test that a non-default D-notation NULL value is used correctly.

        NULL. -99.99D0 with BAD data → DT should be -99.99, NOT the
        hardcoded default -999.25. This verifies the D-notation path
        actually produces the correct value.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -99.99D0       : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  BAD\n"
        )
        test_file = tmp_path / "d_null_99.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # -99.99D0 = -99.99, not -999.25
        assert data["logs"]["DT"][0] == -99.99


class TestWrappedPathologicalMisalignment:
    """T6/G-09: _read_wrapped pathological misalignment path."""

    def test_pathological_misalignment_parses_cleanly(self, tmp_path: Path) -> None:
        """II-4 (accepted): the F-06 pathological-misalignment hard-fail is
        removed by the n_curves-accumulation rewrite — the "extra" values
        on the second depth line are now data, and the whole section
        accumulates into 2 clean steps instead of raising LASParseError.

        The F-06 guard's premise ("depth line had extra values" = certain
        misalignment) only held under the depth-line flag protocol; under
        accumulation the same values align into complete steps."""
        # Curve count = 4 (DEPT + 3 non-depth curves)
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            " SP.MV    :  Spontaneous Potential\n"
            "~A\n"
            # Step 1: depth + 3 continuation values
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "10.0\n"
            # Step 2: "depth" line with 3 values + 1 continuation value —
            # previously depth_had_extra + short next line → LASParseError
            "101.0  99.0  88.0\n"
            "77.0\n"
        )
        test_file = tmp_path / "pathological.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # 2 clean steps: [100,50,75,10] and [101,99,88,77]
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 99.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 88.0])
        np.testing.assert_allclose(data["logs"]["SP"], [10.0, 77.0])


class TestDetectActualWrapDLM:
    """T8/G-11: _detect_actual_wrap DLM-aware branch."""

    def test_wrap_yes_with_tab_delimiter(self, tmp_path: Path) -> None:
        """Test truly-wrapped data with DLM=TAB and one value per line.

        When WRAP=YES and DLM=TAB, the DLM-aware branch at
        data_reader.py:190-191 uses strip-and-filter split to handle
        trailing delimiters. The first data line has 1 value (< curve_count)
        → _detect_actual_wrap returns True and parsing uses _read_wrapped.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.    YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM .   TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n50.0\n101.0\n51.0\n"
        )
        test_file = tmp_path / "wrapped_tab_dlm.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])

    def test_wrap_yes_with_mislabeled_data_tab_dlm(self, tmp_path: Path) -> None:
        """Test WRAP=YES with mislabeled non-wrapped data and TAB delimiter.

        The file claims WRAP=YES but each data line has full column set
        separated by TAB.  _detect_actual_wrap should detect non-wrapped
        and route to _read_normal.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            " DLM .   TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0\t50.0\n"
            "101.0\t51.0\n"
        )
        test_file = tmp_path / "mislabeled_tab_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])


class TestReaderLAS12WellExtraction:
    """T1/F-10: LAS 1.2 well section value extraction via reader."""

    def test_las12_well_values_read_correctly(self, tmp_path: Path) -> None:
        """End-to-end test of LAS 1.2 well value extraction through reader.

        Verifies that read_las_file correctly extracts both numeric and
        non-numeric well values from LAS 1.2 files using auto-detection.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            # Numeric fields in spec format (value before colon)
            " STRT.M   1670.0000 : START DEPTH\n"
            " STOP.M   1660.0000 : STOP DEPTH\n"
            " STEP.M   -0.1250    : STEP\n"
            " NULL.    -999.25    : NULL VALUE\n"
            # Non-numeric fields in lasio convention (value after colon)
            " COMP.    COMPANY : ANY OIL COMPANY LTD.\n"
            " WELL.    WELL : ANY ET AL OIL WELL #12\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "1670.0\n"
        )
        test_file = tmp_path / "las12_well.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["well"]["STRT"] == "1670.0000"
        assert data["well"]["STOP"] == "1660.0000"
        assert data["well"]["NULL"] == "-999.25"
        assert data["well"]["COMP"] == "ANY OIL COMPANY LTD."
        assert data["well"]["WELL"] == "ANY ET AL OIL WELL #12"


class TestMaxLimitsGuards:
    """F-026: MAX_DATA_LINES / MAX_CURVES guard tests."""

    def test_max_data_lines_normal_mode(self, tmp_path: Path) -> None:
        """Test MAX_DATA_LINES guard raises error."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "max_lines.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 0):
            with pytest.raises(LASParseError, match="Data line count"):
                read_las_file(test_file)

    def test_max_curves_normal_mode(self, tmp_path: Path) -> None:
        """Test MAX_CURVES guard raises error."""
        from unittest import mock

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
        )
        test_file = tmp_path / "max_curves.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_CURVES", 1):
            with pytest.raises(LASParseError, match=r"(?i)curve count"):
                read_las_file(test_file)

    def test_max_curves_wrapped_mode(self, tmp_path: Path) -> None:
        """Test MAX_CURVES guard in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
        )
        test_file = tmp_path / "max_curves_wrap.las"
        test_file.write_text(content, encoding="utf-8")

        with mock.patch("pylasdev.data_reader.MAX_CURVES", 1):
            with pytest.raises(LASParseError, match=r"(?i)curve count"):
                read_las_file(test_file)


class TestToFiniteFloat:
    """F-029: _to_finite_float non-finite value guard tests.

    MERGED (register): the four single-token tests (nan / inf / -inf / 1e309)
    exercised the same isfinite guard with the same oracle; they are one
    parametrized test so a new non-finite shape is a one-line addition.
    """

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf", "1e309"])
    def test_non_finite_value_returns_null(self, tmp_path: Path, token: str) -> None:
        """Non-finite data tokens are replaced by the declared null value."""
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
            f"100.0  {token}\n"
        )
        test_file = tmp_path / f"non_finite_{token}.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["logs"]["DT"][0] == -999.25, f"{token!r} was not nulled"


class TestMetadataOnlyLas:
    """CF-021: Metadata-only LAS file (no ~A section)."""

    def test_las_without_ascii_section_parses(self, tmp_path: Path) -> None:
        """Test that LAS file with ~V and ~C but no ~A parses correctly."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            " STOP.M   200.0 : STOP DEPTH\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
        )
        test_file = tmp_path / "no_ascii.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file)
        assert data["curves_order"] == ["DEPT", "DT"]
        # logs contain zero-length arrays (pre-allocated with np.zeros(0))
        assert len(data["logs"]) == 2
        assert len(data["logs"]["DEPT"]) == 0
        assert len(data["logs"]["DT"]) == 0

    def test_las_without_ascii_section_as_object(self, tmp_path: Path) -> None:
        """Test LASFile object from metadata-only file."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : START DEPTH\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
        )
        test_file = tmp_path / "no_ascii_obj.las"
        test_file.write_text(content, encoding="utf-8")

        las = read_las_file_as_object(test_file)
        assert las.curves_order == ["DEPT"]
        # logs contains zero-length array (pre-allocated with np.zeros(0))
        assert "DEPT" in las.logs
        assert len(las.logs["DEPT"]) == 0
        assert las.well["STRT"] == "100.0"


class TestArrayTrimmingOvercount:
    """F24: Tests for array trimming when pre-scan over-counts data lines.

    The trimming branch (data_reader.py:354-359) runs when current_line <
    data_line_count. This happens when the parser's _pre_scan counts data
    lines across ALL ~A sections but _read_normal stops at the first
    non-A section header (~OTHER, ~P, etc.).
    """

    def test_multi_a_section_trimming(self, tmp_path: Path) -> None:
        """Dup-~A-drop: a second ~A block after a non-~A section must NOT
        be ingested; only the first ~A block's data survives.

        The parser's pre-scan counts only the FIRST contiguous ~A block
        (data_line_count == 2), so the trim branch never fires — the
        primary guard here is the first-block-only ingestion contract
        (F-EX-02 / I2-06): data in later ~A blocks after a non-~A
        section is DROPPED, and the reader emits the
        "Multiple ~A data sections encountered" warning.
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
            "Freeform text.\n"
            "~A  DEPT  DT\n"
            "200.0  60.0\n"
        )
        test_file = tmp_path / "multi_a_trim.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
        # Only first section's 2 data lines should be present
        assert len(data["logs"]["DEPT"]) == 2
        assert len(data["logs"]["DT"]) == 2
        np.testing.assert_array_almost_equal(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_almost_equal(data["logs"]["DT"], [50.0, 51.0])
        # Dup-~A-drop guard: the second ~A block's values (200.0 / 60.0)
        # must never appear in the ingested arrays.
        assert 200.0 not in data["logs"]["DEPT"]
        assert 60.0 not in data["logs"]["DT"]
        # The reader must announce the drop via the warnings API.
        assert any("Multiple ~A data sections" in str(x.message) for x in w)


class TestExtraColumnWarning:
    """F34: Tests for extra-column warning in _read_normal.

    The warning at data_reader.py:330-331 triggers when a data line has
    MORE columns than curve_count. Exercises the `warned_extra` flag path.
    """

    def test_extra_columns_warning_normal(self, tmp_path: Path) -> None:
        """Test extra-column warning triggered in non-wrapped mode."""
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
            "100.0  50.0  75.0  99.0\n"  # 4 values but only 2 curves
            "101.0  51.0\n"
        )
        test_file = tmp_path / "extra_cols.las"
        test_file.write_text(content, encoding="utf-8")

        # F-089: warnings now issued via warnings.warn, not logger.warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            assert any("Extra columns are discarded" in str(x.message) for x in w)

        # Data should be read correctly: extra values are silently skipped
        assert data["logs"]["DEPT"][0] == 100.0
        assert data["logs"]["DT"][0] == 50.0
        assert data["logs"]["DEPT"][1] == 101.0
        assert data["logs"]["DT"][1] == 51.0

    def test_extra_columns_only_first_row_warns(self, tmp_path: Path) -> None:
        """Extra columns warn once per file, with a counted summary line.

        HARDENED (register F7): the former body pinned only the per-line
        warning; the end-of-section summary that carries the count
        (``extra_col_count > 1``) was unasserted, so a lost count or a summary
        that reports 1 would have passed.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0  1.0  2.0\n"  # 3 values, 1 curve
            "101.0  3.0  4.0\n"  # 3 values, 1 curve — should NOT warn again
        )
        test_file = tmp_path / "extra_once.las"
        test_file.write_text(content, encoding="utf-8")

        # F-089: warnings now issued via warnings.warn, not logger.warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_las_file(test_file)
            warning_texts = [str(x.message) for x in w]
            per_line = [t for t in warning_texts if "Extra columns are discarded" in t]
            assert len(per_line) == 1, f"expected one per-line warning, got {per_line}"
            assert "has 3 values but only 1 curves declared" in per_line[0]
            summaries = [t for t in warning_texts if "had more values than the 1 declared" in t]
            assert len(summaries) == 1, f"expected one counted summary, got {summaries}"
            assert summaries[0].startswith("2 data line(s)"), (
                f"the summary must count BOTH rows, got: {summaries[0]}"
            )


class TestLAS30StructuredDataValues:
    """F-15: Value validation for structured data sections in sample_las3.0_spec.las.

    Tests that data_sections values from the spec-conformant LAS 3.0 file
    are parsed correctly on INITIAL READ (not roundtrip). The roundtrip
    path has a known key-name corruption issue (per-section curve names
    become global curve names on re-read), so only first-read validation
    is performed here.
    """

    def assert_section_values(
        self,
        sections: list[dict],
        expected_name: str,
        expected_type: str,
        expected_data: dict[str, list[float]],
    ) -> dict:
        """Find a section by name and assert its numeric data."""
        for section in sections:
            if section["name"] == expected_name:
                assert section["section_type"] == expected_type
                data = section["data"]
                for key, expected_values in expected_data.items():
                    assert key in data, (
                        f"Key '{key}' missing in {expected_name} section. "
                        f"Available keys: {list(data.keys())}"
                    )
                    np.testing.assert_array_almost_equal(
                        data[key],
                        np.array(expected_values),
                        err_msg=f"{expected_name}.{key} value mismatch",
                    )
                return section
        pytest.fail(f"Section '{expected_name}' not found in data_sections")

    def test_las30_spec_drilling_data(self, test_data_dir: Path) -> None:
        """Validate Drilling data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "DRILLING",
            "DRILLING_DATA",
            {
                "DEPT": [322.02, 323.05],
                "DIST": [1.02, 2.05],
                "HRS": [0.0, 0.1],
                "ROP": [24.0, 37.5],
                "WOB": [3.0, 2.0],
                "RPM": [59.0, 69.0],
                "TQ": [111.0, 118.0],
                "PUMP": [1199.0, 1182.0],
                "TSPM": [179.0, 175.0],
                "GPM": [879.0, 861.0],
                "ECD": [8.73, 8.73],
                "TBR": [39.0, 202.0],
            },
        )

    def test_las30_spec_core1_data(self, test_data_dir: Path) -> None:
        """Validate CORE[1] data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "CORE[1]",
            "CORE_DATA",
            {
                "CORET": [545.50, 551.20, 575.00],
                "COREB": [550.60, 554.90, 595.00],
            },
        )

    def test_las30_spec_core2_data(self, test_data_dir: Path) -> None:
        """Validate CORE[2] data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "CORE[2]",
            "CORE_DATA",
            {
                "CORET": [655.50, 661.20, 675.00],
                "COREB": [660.60, 664.90, 695.00],
            },
        )

    def test_las30_spec_inclinometry_data(self, test_data_dir: Path) -> None:
        """Validate Inclinometry data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "INCLINOMETRY",
            "INCLINOMETRY_DATA",
            {
                "MD": [0.00, 100.00, 200.00, 300.00, 400.00, 500.00, 600.00],
                "TVD": [0.00, 100.00, 198.34, 295.44, 390.71, 482.85, 571.90],
                "AZIM": [290.00, 234.00, 284.86, 234.21, 224.04, 224.64, 204.39],
                "DEVI": [0.00, 0.00, 1.43, 2.04, 3.93, 5.88, 7.41],
            },
        )

    def test_las30_spec_tops_data(self, test_data_dir: Path) -> None:
        """Validate Tops data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "TOPS",
            "TOPS_DATA",
            {
                "TOPT": [545.50, 602.00, 615.00],
                "TOPB": [602.00, 615.00, 655.00],
            },
        )

    def test_las30_spec_perforations_data(self, test_data_dir: Path) -> None:
        """Validate Perforations data section values from sample_las3.0_spec.las."""
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)
        sections = [ds.to_dict() for ds in las.data_sections]

        self.assert_section_values(
            sections,
            "PERFORATIONS",
            "PERFORATIONS_DATA",
            {
                "PERFT": [545.50, 551.20, 575.00],
                "PERFB": [550.60, 554.90, 595.00],
                "PERFD": [12.0, 12.0, 12.0],
            },
        )

    def test_las30_spec_string_data_in_sections(self, test_data_dir: Path) -> None:
        """Validate string_data within structured data sections.

        CORE[1] has CDES with string descriptions; Core[2] has same.
        Perforations has PERFT (charge type) as string.
        """
        spec_file = test_data_dir / "sample_las3.0_spec.las"
        assert spec_file.exists(), f"Required test data missing: {spec_file}"
        las = read_las_file_as_object(spec_file)

        # Find CORE[1] section (parser uppercases section names)
        core1 = None
        for ds in las.data_sections:
            if ds.name == "CORE[1]":
                core1 = ds
                break
        assert core1 is not None, "CORE[1] section not found"
        assert "CDES" in core1.string_data, (
            f"CDES missing from CORE[1] string_data. Keys: {list(core1.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            core1.string_data["CDES"],
            np.array(["Long cylindrical hunk of rock", "Long broken hunk of rock", "Debris only"]),
        )

        # Verify section count and names (parser uppercases all section names)
        section_names = [ds.name for ds in las.data_sections]
        # Should include 8 sections: DRILLING, CORE[1], CORE[2], INCLINOMETRY, TEST, TOPS, PERFORATIONS, ASCII
        assert "DRILLING" in section_names
        assert "CORE[1]" in section_names
        assert "CORE[2]" in section_names
        assert "INCLINOMETRY" in section_names
        assert "TEST" in section_names
        assert "TOPS" in section_names
        assert "PERFORATIONS" in section_names
        assert len(las.data_sections) == 8

        # Validate CORE[2] CDES string_data
        core2 = None
        for ds in las.data_sections:
            if ds.name == "CORE[2]":
                core2 = ds
                break
        assert core2 is not None, "CORE[2] section not found"
        assert "CDES" in core2.string_data, (
            f"CDES missing from CORE[2] string_data. Keys: {list(core2.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            core2.string_data["CDES"],
            np.array(["Long cylindrical hunk of rock", "Long broken hunk of rock", "Debris only"]),
        )

        # Validate TEST BLOWD string_data
        test_section = None
        for ds in las.data_sections:
            if ds.name == "TEST":
                test_section = ds
                break
        assert test_section is not None, "TEST section not found"
        assert "BLOWD" in test_section.string_data, (
            f"BLOWD missing from TEST string_data. Keys: {list(test_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            test_section.string_data["BLOWD"],
            np.array(["Weak Blow", "Strong Blow", "Blow Out"]),
        )

        # Validate TOPS TOPN string_data
        tops_section = None
        for ds in las.data_sections:
            if ds.name == "TOPS":
                tops_section = ds
                break
        assert tops_section is not None, "TOPS section not found"
        assert "TOPN" in tops_section.string_data, (
            f"TOPN missing from TOPS string_data. Keys: {list(tops_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            tops_section.string_data["TOPN"],
            np.array(["Viking", "Colony", "Basal Quartz"]),
        )

        # Validate PERFORATIONS PERFT_2 string_data (PERFT appears twice:
        # once as float depth, once as string charge type — the string
        # one gets renamed to PERFT_2 by per-section dedup)
        perfs_section = None
        for ds in las.data_sections:
            if ds.name == "PERFORATIONS":
                perfs_section = ds
                break
        assert perfs_section is not None, "PERFORATIONS section not found"
        assert "PERFT_2" in perfs_section.string_data, (
            f"PERFT_2 missing from PERFORATIONS string_data. Keys: {list(perfs_section.string_data.keys())}"
        )
        np.testing.assert_array_equal(
            perfs_section.string_data["PERFT_2"],
            np.array(["BIG HOLE", "BIG HOLE", "BIG HOLE"]),
        )


class TestGetNullValue:
    """F-06: Direct unit tests for _get_null_value Fortran D-notation handling."""

    def test_standard_null_value(self) -> None:
        """Standard -999.25 null value parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "-999.25"}
        result = _get_null_value(well)
        assert result == -999.25

    def test_d_notation_with_exponent(self) -> None:
        """Fortran D-notation with explicit exponent (+03) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "1.0D+03"}
        result = _get_null_value(well)
        assert result == 1000.0

    def test_d_notation_negative_exponent(self) -> None:
        """Fortran D-notation with negative exponent (D-02) parses correctly."""
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "3.14D-02"}
        result = _get_null_value(well)
        assert result == 0.0314

    def test_missing_null_key_returns_default(self) -> None:
        """Missing NULL key returns default_float."""
        from pylasdev.data_reader import _get_null_value

        well: dict[str, str] = {}
        result = _get_null_value(well)
        assert result == -999.25


class TestMaxTokensPerLineGuard:
    """M68: Tests for MAX_TOKENS_PER_LINE DoS guard.

    MAX_TOKENS_PER_LINE (data_reader.py:37, =MAX_CURVES=100_000)
    caps the number of tokens parsed from a single data line to
    prevent resource exhaustion from pathological input. Used at
    7 call sites across data_reader.py and dev_reader.py.
    Previously had zero test coverage (grep confirmed).
    """

    def test_token_count_capped_in_normal_mode(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE caps tokens in normal (non-wrapped) mode.

        Mock MAX_TOKENS_PER_LINE to a low value and verify that extra
        tokens on a data line are silently truncated instead of causing
        resource exhaustion.
        """
        from unittest import mock

        # Build a file with 3 declared curves but 4+ space-separated
        # tokens on a data line.
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A  DEPT  DT  GR\n"
            "100.0  50.0  75.0  99.0\n"
            "101.0  51.0  76.0  88.0\n"
        )
        test_file = tmp_path / "token_cap_normal.las"
        test_file.write_text(content, encoding="utf-8")

        # Patch MAX_TOKENS_PER_LINE to 2: split() produces at most 3 tokens.
        # Data line "100.0  50.0  75.0  99.0" → "100.0", "50.0", "75.0  99.0"
        # The 4th value is merged into the 3rd token, causing a float parse
        # failure → substituted with null_value.
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_las_file(test_file)
            # Extra 4th value on each line should be silently handled
            assert len(data["logs"]["DEPT"]) == 2
            assert data["logs"]["DEPT"][0] == 100.0
            # Discriminator: with the cap at 2, the line's 3rd token is
            # "75.0  99.0" (4th value merged into it) — the float parse
            # fails and GR is null-filled.  Removing the cap guard leaves
            # GR[0] == 75.0, so this assertion fails on cap deletion.
            assert data["logs"]["GR"][0] == -999.25

    def test_token_count_capped_in_wrapped_mode(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE caps tokens in wrapped mode."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0  99.0  88.0\n"
            "101.0\n"
            "51.0\n"
        )
        test_file = tmp_path / "token_cap_wrapped.las"
        test_file.write_text(content, encoding="utf-8")

        # Patch MAX_TOKENS_PER_LINE low: wrapped-mode split also capped
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 1):
            data = read_las_file(test_file)
            assert data["logs"]["DEPT"][0] == 100.0
            # Discriminator: with cap=1 the "50.0  99.0  88.0" line splits
            # to ["50.0"] (99.0/88.0 truncated away), so step 2's DEPT
            # value is lost to the null fill.  Removing the cap guard
            # leaves DEPT == [100.0, 99.0, 101.0], so this assertion
            # (DEPT[1] == -999.25) fails on cap deletion.
            assert data["logs"]["DEPT"][1] == -999.25

    def test_token_cap_dev_reader(self, tmp_path: Path) -> None:
        """MAX_TOKENS_PER_LINE capped in DEV reader (dev_reader.py:719)."""
        from unittest import mock

        content = "MD TVD X Y Z\n0.0 0.0 100.0 200.0 300.0\n100.0 99.0 101.0 201.0 301.0\n"
        test_file = tmp_path / "token_cap_dev.dev"
        test_file.write_text(content, encoding="utf-8")

        # Patch data_reader.MAX_TOKENS_PER_LINE: dev_reader fetches at
        # runtime via _resolve_max_tokens_per_line (F-DVR-01 fix).
        with mock.patch("pylasdev.data_reader.MAX_TOKENS_PER_LINE", 2):
            data = read_dev_file(test_file)
            # split(maxsplit=2) produces at most 3 tokens from 5-space line
            assert "MD" in data
            assert "TVD" in data
            # Discriminator: with cap=2 the last three header words collapse
            # into a single "X Y Z" column; removing the cap guard splits
            # them back into X/Y/Z, so "X Y Z" disappears and this
            # assertion fails on cap deletion.
            assert "X Y Z" in data


class TestMaxLimitsAtLimit:
    """M69: At-limit tests for MAX_* reader guards.

    F-M01 changed data_reader.py operator from ``>`` to ``>=`` for
    MAX_CURVES and MAX_DATA_LINES (matching parser.py and models.py).
    At-limit cases now REJECT instead of passing — the limit is exclusive.
    MAX_TOTAL_ELEMENTS still uses ``>`` so at-limit still passes.
    """

    def test_max_curves_at_limit_rejected(self, tmp_path: Path) -> None:
        """MAX_CURVES set exactly to the file's curve count — must reject."""
        from unittest import mock

        # File has exactly 2 curves
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
        )
        test_file = tmp_path / "at_limit_curves.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_CURVES=2 matches actual curve count — F-M01 now rejects at-limit
        with mock.patch("pylasdev.data_reader.MAX_CURVES", 2):
            with pytest.raises(LASParseError, match="exceeds maximum"):
                read_las_file(test_file)

    def test_max_data_lines_at_limit_accepted(self, tmp_path: Path) -> None:
        """MAX_DATA_LINES set exactly to the file's data line count — must accept.

        F-MDR-01: Changed _read_normal from ``>=`` to ``>`` for consistency
        with models.py which uses ``>`` at all 8 MAX_DATA_LINES guard sites
        (accepts at exactly the limit).  The F-M01 fix comment that claimed
        models.py uses ``>=`` was incorrect.
        """
        from unittest import mock

        # File has exactly 1 data line
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )
        test_file = tmp_path / "at_limit_lines.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_DATA_LINES=1 matches actual data line count — F-MDR-01 accepts at-limit
        with mock.patch("pylasdev.data_reader.MAX_DATA_LINES", 1):
            data = read_las_file(test_file)
            assert len(data["logs"]["DEPT"]) == 1
            assert data["logs"]["DEPT"][0] == 100.0

    def test_max_total_elements_at_limit_passes(self, tmp_path: Path) -> None:
        """MAX_TOTAL_ELEMENTS set exactly to curves*lines — must pass."""
        from unittest import mock

        # File has 2 curves x 1 data line = 2 elements
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
        )
        test_file = tmp_path / "at_limit_total.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_TOTAL_ELEMENTS=2 matches curves*lines — should pass
        with mock.patch("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 2):
            data = read_las_file(test_file)
            assert len(data["logs"]["DEPT"]) == 1
            assert data["logs"]["DEPT"][0] == 100.0

    def test_max_curves_wrapped_at_limit_rejected(self, tmp_path: Path) -> None:
        """MAX_CURVES at limit in wrapped mode — must reject."""
        from unittest import mock

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
        )
        test_file = tmp_path / "at_limit_wrap_curves.las"
        test_file.write_text(content, encoding="utf-8")

        # MAX_CURVES=2 matches actual — F-M01 now rejects at-limit
        with mock.patch("pylasdev.data_reader.MAX_CURVES", 2):
            with pytest.raises(LASParseError, match="exceeds maximum"):
                read_las_file(test_file)


# ============================================================
# Production Check Regression Tests
# ============================================================


class TestProductionCheckReaderFixes:
    """Regression tests for production check fixes in data_reader.py and reader.py."""

    # --- F-027 (MEDIUM): _get_null_value silences LASParseError ---

    @pytest.mark.parametrize("token", ["nan", "inf"])
    def test_non_finite_null_value_raises_las_parse_error(self, token: str) -> None:
        """F-027: a non-finite NULL value raises LASParseError.

        Before the fix, LASParseError was caught by the except clause
        (lines 240), falling back to -999.25 silently. Now the error
        propagates so callers know the NULL is corrupt.

        MERGED (register): the 'nan' and 'inf' twins asserted the same guard
        with the same message; both tokens stay as parameters.
        """
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": token}
        with pytest.raises(LASParseError, match="NULL value must be a finite number"):
            _get_null_value(well)

    def test_non_numeric_null_still_falls_back_to_default(self) -> None:
        """F-027: Non-numeric NULL ('NOT_A_NUMBER') still falls back to default.

        Regression check: the except clause still catches ValueError from
        float conversion. Only LASParseError was removed.
        """
        from pylasdev.data_reader import _get_null_value

        well = {"NULL": "NONSENSE"}
        result = _get_null_value(well)
        assert result == -999.25

    # --- F-211 (MEDIUM): _read_wrapped depth-step underestimation ---

    def test_wrapped_compact_all_zero(self, tmp_path: Path) -> None:
        """F-211: Compact wrapped file (2 lines/step, all zero data).

        The depth-step estimate max(1, ceil(N/curve_count)) must not
        undercount steps in compact wrapped format.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            # Depth step 1: 2 lines
            "100.0\n"
            "50.0  75.0\n"
            # Depth step 2: 2 lines
            "101.0\n"
            "51.0  76.0\n"
            # Depth step 3: 2 lines
            "102.0\n"
            "52.0  77.0\n"
        )
        test_file = tmp_path / "compact_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # All 3 depth steps must be parsed correctly
        assert len(data["logs"]["DEPT"]) == 3
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0, 52.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 76.0, 77.0])

    # --- E-F-018 (HIGH): depth_line state machine corruption regression test ---

    def test_wrapped_depth_line_state_machine_non_pathological_ef018(self, tmp_path: Path) -> None:
        """E-F-018 — ACCEPTED CONTRACT CHANGE (II-4/R-6): the n_curves-
        accumulation rewrite removes the depth-line state machine's
        "unrecoverable data misalignment" hard-fail.  The file previously
        hard-failed; it now parses as 3 valid steps.  The second "depth"
        line's extra value (1050.5) is data under accumulation, not
        corrupting junk.

        Test scenario (curve_count=4, DEPT + DT, GR, SP):
          Step 1: depth=1000.0, data=200.0,300.0,400.0  (normal baseline)
          Step 2: depth=1010.0 + extra 1050.5, data=210.0,310.0
          Step 3: depth=1020.0, data=220.0,320.0,420.0
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            " SP.MV    :  Spontaneous Potential\n"
            "~A\n"
            # Step 1 (normal baseline): one value per line
            "1000.0\n"
            "200.0  300.0  400.0\n"
            # Step 2: previously triggered the F-032 hard-fail (depth line
            # with 2 values + short next line); under accumulation the
            # 5 values flow into step 2 = [1010.0, 1050.5, 210.0, 310.0]
            "1010.0  1050.5\n"
            "210.0  310.0\n"
            # Step 3
            "1020.0\n"
            "220.0  320.0  420.0\n"
        )
        test_file = tmp_path / "ef018_wrapped_state_machine.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # 3 clean steps under accumulation (previously LASParseError)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1010.0, 1020.0])
        np.testing.assert_allclose(data["logs"]["DT"], [200.0, 1050.5, 220.0])
        np.testing.assert_allclose(data["logs"]["GR"], [300.0, 210.0, 320.0])
        np.testing.assert_allclose(data["logs"]["SP"], [400.0, 310.0, 420.0])

    # --- F-212 (MEDIUM): _desanitize_las_value unconditional _# strip ---

    def test_desanitize_disabled_preserves_hash_prefix(self) -> None:
        """F-212: With _DESANITIZE_ENABLED=False, _# values are preserved.

        When reading non-pylasdev files that genuinely contain _# in
        their data, the desanitize function must not strip the underscore.
        """
        import pylasdev.data_reader as dr_mod
        import pylasdev.parser as _parser_mod

        prior = _parser_mod._DESANITIZE_ENABLED
        try:
            # F-088: _DESANITIZE_ENABLED is now unified in parser.py;
            # setting parser's flag affects data_reader._desanitize_las_value.
            _parser_mod._DESANITIZE_ENABLED = False
            # Value starts with _# — should be preserved as-is
            result = dr_mod._desanitize_las_value("_#original_value")
            assert result == "_#original_value", (
                f"Expected '_#original_value' preserved, got {result!r}"
            )
            # Non-_# value unchanged
            assert dr_mod._desanitize_las_value("normal_value") == "normal_value"
        finally:
            _parser_mod._DESANITIZE_ENABLED = prior

    def test_desanitize_enabled_strips_hash_prefix(self) -> None:
        """F-212: Default behavior (enabled) strips _# prefix for roundtrip.

        When _DESANITIZE_ENABLED=True (default), _# prefix is stripped
        to restore the original #-prefixed value from the writer escape.
        """
        import pylasdev.data_reader as dr_mod
        import pylasdev.parser as _parser_mod

        # F-088: _DESANITIZE_ENABLED is now unified in parser.py.
        assert _parser_mod._DESANITIZE_ENABLED is True
        result = dr_mod._desanitize_las_value("_#test_value")
        assert result == "#test_value", f"Expected '#test_value' after desanitize, got {result!r}"

    # --- E-04 (MEDIUM): thread-local _DESANITIZE_ENABLED never reset ---

    def test_desanitize_flag_restored_after_false_read(self, tmp_path: Path) -> None:
        """E-04: read_las_file_as_object(desanitize=False) restores the flag.

        The thread-local _DESANITIZE_ENABLED flag is set before parser
        header parsing and must be restored afterwards.  Previously a
        desanitize=False read left the flag False on the current thread,
        silently changing the behavior of subsequent direct
        LASParser.parse() users that never passed desanitize=False.

        HARDENED (register F2): the flag-level assertion alone let a read that
        restored the flag but ignored the parameter pass.  A data-level ``_#``
        escape is now read with ``desanitize=False`` and must come back
        untouched.
        """
        import pylasdev.parser as _parser_mod

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " STRT.M   100.0   : START DEPTH\n"
            " STOP.M   200.0   : STOP DEPTH\n"
            " STEP.M   0.5     : STEP DEPTH\n"
            " WELL.    _#escaped : WELL NAME\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.API   :  Gamma\n"
            "~A  DEPT  GR\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
        )
        test_file = tmp_path / "e04_desanitize_restore.las"
        test_file.write_text(content, encoding="utf-8")

        # Record the thread-local default before the read.
        prior = _parser_mod._DESANITIZE_ENABLED
        las_raw = read_las_file_as_object(test_file, desanitize=False)
        # Data-level contract: with desanitize=False the writer escape is kept.
        assert las_raw.well["WELL"] == "_#escaped", (
            f"desanitize=False must keep the raw escape, got {las_raw.well['WELL']!r}"
        )
        # The flag must be restored — the read must not leak desanitize=False
        # to subsequent same-thread callers.
        assert _parser_mod._DESANITIZE_ENABLED is prior, (
            "E-04: _DESANITIZE_ENABLED not restored after desanitize=False "
            f"read (got {_parser_mod._DESANITIZE_ENABLED}, expected {prior})"
        )
        # And a subsequent read with the default desanitize=True unescapes it.
        las_unescaped = read_las_file_as_object(test_file)
        assert las_unescaped.well["WELL"] == "#escaped", (
            f"the restored default must desanitize, got {las_unescaped.well['WELL']!r}"
        )

    def test_desanitize_flag_restored_on_parse_error(self, tmp_path: Path) -> None:
        """E-04: flag restored even when parsing fails (finally semantics)."""
        import pylasdev.parser as _parser_mod

        test_file = tmp_path / "e04_parse_error.las"
        test_file.write_text("not a las file at all\n", encoding="utf-8")

        prior = _parser_mod._DESANITIZE_ENABLED
        with pytest.raises(LASParseError):
            read_las_file_as_object(test_file)
        assert _parser_mod._DESANITIZE_ENABLED is prior

    # --- F-219 / ENC-03: LASEncodingError propagation ---

    def test_encoding_error_propagates_as_las_encoding_error(self, tmp_path: Path) -> None:
        """F-219/ENC-03: a genuine decoding failure propagates as
        LASEncodingError with an accurate message.

        F-219 previously pinned LASEncodingError → LASReadError.  ENC-03
        corrected that contract: read_with_encoding raises LASEncodingError
        with the file path and codec cause, and reader.py must NOT relabel
        it as a misleading 'size exceeded' LASReadError.  Callers now
        receive LASEncodingError, matching read_with_encoding's contract.
        """
        from pylasdev.exceptions import LASEncodingError

        content = b"\xff\xfe\x00\x01"  # Invalid encoding bytes
        test_file = tmp_path / "bad_encoding.las"
        test_file.write_bytes(content)

        # Without chardet and empty fallback chain, encoding fails
        from unittest import mock

        with mock.patch("pylasdev.encoding.FALLBACK_ENCODINGS", []):
            with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
                with pytest.raises(LASEncodingError, match="Failed to decode"):
                    read_las_file(test_file)


class TestG6DataReaderFixes:
    """G6 regression tests: D-01/D-02/D-03 wrap detection, L-03 {I}
    precision, N-I-08 mid-file under-fill, IT3-F-01 desanitize hoist,
    IT3-F-02 math.isfinite swap, IT3-F-03 wrapped pre-allocation."""

    # --- D-01 (HIGH): COMMA/TAB wrap misdetection ---

    @pytest.mark.parametrize(
        ("rows", "expected_dept", "expected_gr"),
        [
            (
                "100.0,\n101.0,50.0,75.0\n102.0,51.0,76.0\n103.0,52.0,77.0\n",
                [100.0, 101.0, 102.0, 103.0],
                [-999.25, 75.0, 76.0, 77.0],
            ),
            (
                "100.0,\n101.0,50.0,75.0\n102.0,51.0,76.0\n",
                [100.0, 101.0, 102.0],
                [-999.25, 75.0, 76.0],
            ),
        ],
        ids=["four_rows", "three_rows"],
    )
    def test_d01_comma_sparse_first_row_not_wrapped(
        self, tmp_path: Path, rows: str, expected_dept: list[float], expected_gr: list[float]
    ) -> None:
        """D-01: WRAP=NO comma file with a sparse first row (trailing comma)
        must NOT be misdetected as wrapped.

        Pre-fix, the comma/tab path returned wrapped on any 1-value first
        line without corroboration — 50% of rows were lost and the depth
        value landed in the GR column.  The protocol-based detector sees the
        full rows that follow and classifies the file non-wrapped.

        MERGED (register): TestG6's EXT-01 sparse-row guard was the same
        fixture shape with one fewer row; it is the second parameter here.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A DEPT DT GR\n" + rows
        )
        test_file = tmp_path / f"d01_comma_{len(expected_dept)}rows.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)

        assert len(data["logs"]["DEPT"]) == len(expected_dept), (
            f"Expected {len(expected_dept)} rows, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], expected_dept)
        np.testing.assert_allclose(data["logs"]["GR"], expected_gr)

    def test_d01_tab_sparse_first_row_not_wrapped(self, tmp_path: Path) -> None:
        """D-01: same trigger with TAB delimiter."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A DEPT DT\n"
            "100.0\n"
            "101.0\t50.0\n"
            "102.0\t51.0\n"
        )
        test_file = tmp_path / "d01_tab.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 3
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])

    # --- D-02 (MEDIUM): two consecutive sparse rows ---

    def test_d02_two_sparse_rows_not_wrapped(self, tmp_path: Path) -> None:
        """D-02: WRAP=NO space file with TWO consecutive sparse rows must
        not be misdetected as wrapped (pre-fix, the 2-line F-M16
        corroboration returned wrapped on two sparse rows)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A DEPT DT GR\n"
            "100.0\n"
            "101.0\n"
            "102.0 50.0 75.0\n"
            "103.0 51.0 76.0\n"
            "104.0 52.0 77.0\n"
        )
        test_file = tmp_path / "d02_two_sparse.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 5, (
            f"Expected 5 rows, got {len(data['logs']['DEPT'])} — file was misdetected as wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0, 103.0, 104.0])

    # --- D-03 (MEDIUM): genuine WRAP=YES with overfull second line ---

    def test_d03_genuine_wrap_overfull_second_line(self, tmp_path: Path) -> None:
        """D-03: a genuine WRAP=YES file whose second data line has
        >= curve_count values must stay wrapped (pre-fix, the second-line
        corroboration misdetected it as non-wrapped and the F2 overflow
        handler became unreachable)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0 75.0 99.0\n"
            "101.0\n"
            "51.0 76.0\n"
        )
        test_file = tmp_path / "d03_overfull.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Still WRAPPED (declared YES + depth evidence in window [1,3,1,2]) —
        # the D-03 detection outcome is preserved.  The READ semantics
        # changed under the accepted n_curves-accumulation rewrite: the
        # overfull continuation's extra value (99.0) "simply starts the next
        # step" (iter-2 design), so step 2 = [99.0, 101.0, 51.0] and the
        # trailing 76.0 is an incomplete final step (discarded with warning).
        assert len(data["logs"]["DEPT"]) == 2, (
            f"Expected 2 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 99.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 101.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 51.0])

    # --- X-2 (O(n²) pending slicing): wide-line wrapped files ---

    def _build_wide_wrapped_file(self, tmp_path: Path, n_tokens: int) -> Path:
        """Build a WRAP=YES file whose first 4 data lines are 1-value
        depth evidence (window [1,1,1,1] → depth-later arm → wrapped) and
        whose 6th data line is a single *n_tokens*-wide continuation line.

        The wide line is what the pre-fix pending-buffer rewrite handled
        quadratically (``pending = pending[curve_count:]`` per flush).
        """
        giant = " ".join(str(i) for i in range(n_tokens))
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "100.0\n1.0\n101.0\n2.0\n"
            "102.0\n" + giant + "\n"
        )
        path = tmp_path / f"x2_wide_{n_tokens}.las"
        path.write_text(content, encoding="utf-8")
        return path

    def _measure_pending_work(
        self, monkeypatch: pytest.MonkeyPatch, test_file: Path
    ) -> tuple[int, list[int], int, dict[str, Any]]:
        """Deterministically measure the pending-buffer work of one wrapped read.

        Uses the production M-10 trace seam (``_read_wrapped_trace_hook``,
        called once per data line while ``pending`` is live) — no wall clock is
        read.  Returns ``(rebuilds, rebuild_sizes, values_processed, data)``:

        * ``rebuilds`` — how many line boundaries saw a NEW ``pending`` list
          object.  A per-flush ``pending = pending[curve_count:]`` rebuild
          produces one per flush (≈n/2 for a 90K-token line); the in-place
          ``del pending[:read_idx]`` frontier produces none.
        * ``rebuild_sizes`` — the length of each such rebuilt buffer.  Their sum
          is the number of value slots the pre-fix path would have copied, so
          the metric is a *work counter*: linear (bounded by the input) for the
          frontier, quadratic for the rebuild.
        * ``values_processed`` — the largest live buffer seen, i.e. the number
          of values that entered the pending protocol.
        * ``data`` — the parsed dict from the same read.
        """
        import pylasdev.data_reader as dr

        rebuild_sizes: list[int] = []
        held: list[list[str]] = []
        state = {"values": 0}

        def hook(pending: list[str]) -> None:
            if not held or held[-1] is not pending:
                rebuild_sizes.append(len(pending))
                if held:
                    held[-1] = pending
                else:
                    held.append(pending)
            state["values"] = max(state["values"], len(pending))

        monkeypatch.setattr(dr, "_read_wrapped_trace_hook", hook)
        try:
            data = read_las_file(test_file)
        finally:
            monkeypatch.setattr(dr, "_read_wrapped_trace_hook", None)
        return len(rebuild_sizes), rebuild_sizes, state["values"], data

    def test_x2_wide_wrapped_line_values_correct_and_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """X-2: a crafted wide-line wrapped file must parse with the SAME
        value-order accumulation the pending protocol defines, and the
        accumulation work must stay LINEAR in the token count.

        The pre-fix protocol re-copied the whole buffer on every flush
        (``pending = pending[curve_count:]``): for a 90K-token line that is
        ≈45K buffer rebuilds copying ≈2·10⁹ value slots.  The work counter
        below (rebuilds of the live buffer + their summed size) is
        deterministic — a reintroduced per-flush rebuild fails it without any
        clock, sampling or machine-speed dependence."""
        n_tokens = 90_000
        test_file = self._build_wide_wrapped_file(tmp_path, n_tokens)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rebuilds, rebuild_sizes, values_processed, data = self._measure_pending_work(
                monkeypatch, test_file
            )

        # Value-order accumulation: every flushed step takes the next
        # curve_count buffered values in order (line boundaries are
        # irrelevant to the protocol).  values = [100,1,101,2,102] +
        # range(n_tokens); steps = len(values) // 2.
        values = [100.0, 1.0, 101.0, 2.0, 102.0] + [float(i) for i in range(n_tokens)]
        n_steps = len(values) // 2
        np.testing.assert_allclose(data["logs"]["DEPT"], values[0::2][:n_steps])
        np.testing.assert_allclose(data["logs"]["DT"], values[1::2][:n_steps])
        assert len(data["logs"]["DEPT"]) == n_steps == 45002
        trail = [str(w.message) for w in rec if "not accounted for" in str(w.message)]
        assert trail, "expected the N-I-08 trailing-partial warning (1 leftover value)"

        # Deterministic complexity proxy (no clock): the buffer is never
        # rebuilt per flush, and the summed rebuild work stays bounded by the
        # input size instead of growing with the number of flushes.
        assert values_processed == len(values), (
            f"pending protocol saw {values_processed} values, expected {len(values)}"
        )
        assert rebuilds <= 1, (
            f"{rebuilds} live-buffer rebuilds during the read — a per-flush "
            f"pending slice is back (sizes: {rebuild_sizes[:5]}…)"
        )
        assert sum(rebuild_sizes) <= len(values), (
            f"pending-buffer copy work {sum(rebuild_sizes)} exceeds the input size "
            f"{len(values)} — accumulation is no longer linear in the token count"
        )

    def test_x2_wide_wrapped_line_scales_linearly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """X-2: doubling the wide-line token count must double the *work*
        (values processed) and must not multiply the number of buffer
        rebuilds — the pre-fix per-flush ``pending[curve_count:]`` slice grew
        the rebuild count with the flush count (quadratic total work).

        The comparison is an exact work-ratio over a deterministic counter,
        not a ratio of measured durations."""
        sizes = (30_000, 60_000)
        measured: dict[int, tuple[int, list[int], int, dict[str, Any]]] = {}
        for n in sizes:
            test_file = self._build_wide_wrapped_file(tmp_path, n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                measured[n] = self._measure_pending_work(monkeypatch, test_file)

        small_rebuilds, small_sizes, small_values, _ = measured[sizes[0]]
        large_rebuilds, large_sizes, large_values, _ = measured[sizes[1]]

        # Linear work: the values that entered the protocol scale exactly with
        # the input (small = 5 + 30 000, large = 5 + 60 000).
        assert small_values == sizes[0] + 5
        assert large_values == sizes[1] + 5
        assert large_values == 2 * small_values - 5
        # Anti-quadratic: one extra input value must not cost an extra
        # full-buffer copy — rebuilds stay at the constant (0 or 1) level while
        # the input doubles, and the summed copy work stays linear.
        assert large_rebuilds <= small_rebuilds + 1, (
            f"rebuild count grew {small_rebuilds} → {large_rebuilds} with 2x tokens — "
            f"a per-flush buffer slice is back"
        )
        assert sum(large_sizes) <= 2 * sum(small_sizes) + (sizes[1] - sizes[0]), (
            f"copy work grew super-linearly: {sum(small_sizes)} → {sum(large_sizes)}"
        )

    # --- EXT-01 (regression): wrapped COMMA/TAB >=3 curves ---

    def test_ext01_wrapped_comma_three_curves_parses_wrapped(self, tmp_path: Path) -> None:
        """EXT-01: a genuine WRAP=YES COMMA file with 3 curves (depth line
        alone, data lines carrying curve_count-1 = 2 values) must be
        detected as wrapped.

        Pre-fix, the D-01/D-02/D-03 protocol rewrite defined "full" for
        COMMA/TAB as ``n > 1``, so continuation lines carrying 2 values
        counted as full rows → full_count >= 2 → misdetected non-wrapped →
        DEPT received curve values, columns shifted, null padding.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.      :  Curve 1\n"
            " C2.      :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0,\n"
            "1.0,2.0,\n"
            "200.0,\n"
            "3.0,4.0,\n"
            "300.0,\n"
            "5.0,6.0,\n"
        )
        test_file = tmp_path / "ext01_comma_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Wrapped: 3 depth steps with DEPT holding depth values only.
        assert len(data["logs"]["DEPT"]) == 3, (
            f"Expected 3 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 200.0, 300.0])
        np.testing.assert_allclose(data["logs"]["C1"], [1.0, 3.0, 5.0])
        np.testing.assert_allclose(data["logs"]["C2"], [2.0, 4.0, 6.0])

    def test_ext01_wrapped_tab_four_curves_parses_wrapped(self, tmp_path: Path) -> None:
        """EXT-01: same misdetection with TAB delimiter and 4 curves
        (data lines carry curve_count-1 = 3 values)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    TAB  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.      :  Curve 1\n"
            " C2.      :  Curve 2\n"
            " C3.      :  Curve 3\n"
            "~A DEPT C1 C2 C3\n"
            "100.0\n"
            "1.0\t2.0\t3.0\n"
            "200.0\n"
            "4.0\t5.0\t6.0\n"
        )
        test_file = tmp_path / "ext01_tab_wrap.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2, (
            f"Expected 2 wrapped steps, got {len(data['logs']['DEPT'])} — "
            f"file was misdetected as non-wrapped"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 200.0])
        np.testing.assert_allclose(data["logs"]["C1"], [1.0, 4.0])
        np.testing.assert_allclose(data["logs"]["C2"], [2.0, 5.0])
        np.testing.assert_allclose(data["logs"]["C3"], [3.0, 6.0])

    # MERGED (register): the EXT-01 sparse-first-row COMMA guard was the same
    # fixture shape as test_d01_comma_sparse_first_row_not_wrapped above
    # (3 curves, sparse first row, identical assertions); it is that test's
    # "three_rows" parameter.

    # --- L-03 (MEDIUM): {I} integer precision ---

    def test_l03_integer_curve_precision_las30(self, tmp_path: Path) -> None:
        """L-03: {I}-format curves must be parsed via int() and stored as
        int64, preserving values above 2^53 that float64 cannot represent.

        Pre-fix, float('9007199254740993') rounded to 9007199254740992.0.
        """
        content = (
            "~VERSION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL\n"
            " NULL.   -999 : NULL VALUE\n"
            "~CURVE\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 9007199254740994\n"
        )
        test_file = tmp_path / "l03_int.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["RUN_NO"].dtype == np.int64, (
            f"Expected int64 dtype, got {data['logs']['RUN_NO'].dtype}"
        )
        assert data["logs"]["RUN_NO"][0] == 9007199254740993, (
            f"Precision lost: {data['logs']['RUN_NO'][0]}"
        )
        assert data["logs"]["RUN_NO"][1] == 9007199254740994

    def test_l03_integer_curve_precision_las20(self, tmp_path: Path) -> None:
        """L-03: {I} precision also preserved on the LAS 1.2/2.0
        data_reader path (_read_normal)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 9007199254740994\n"
        )
        test_file = tmp_path / "l03_int20.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["logs"]["RUN_NO"].dtype == np.int64
        assert data["logs"]["RUN_NO"][0] == 9007199254740993
        assert data["logs"]["RUN_NO"][1] == 9007199254740994

    def test_l03_fractional_null_uses_object_dtype(self, tmp_path: Path) -> None:
        """L-03/EXT-04 trap: when the declared NULL is non-integral
        (e.g. -999.25), int64 allocation would truncate it to -999 — the
        {I} branch stores an object array instead: data values stay exact
        Python ints, null cells keep the fractional sentinel."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 5\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "l03_frac_null.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        # Fractional NULL → object dtype; the null cell keeps -999.25.
        assert data["logs"]["RUN_NO"].dtype == np.object_, (
            f"Expected object dtype, got {data['logs']['RUN_NO'].dtype}"
        )
        assert data["logs"]["RUN_NO"][0] == 5
        assert data["logs"]["RUN_NO"][1] == -999.25

    def test_l03_roundtrip_from_dict_preserves_int64(self) -> None:
        """L-03: from_dict must not coerce {I} log arrays back to float64
        (roundtrip precision preservation)."""
        from pylasdev.models import LASFile

        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO"},
                "well": {"NULL": "-999"},
                "curves": [
                    {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                    {"mnemonic": "RUN_NO", "data_format": "I"},
                ],
                "curves_order": ["DEPT", "RUN_NO"],
                "logs": {
                    "DEPT": np.array([100.0, 101.0]),
                    "RUN_NO": np.array([9007199254740993, 9007199254740994]),
                },
            }
        )
        assert las.logs["RUN_NO"].dtype == np.int64
        assert las.logs["RUN_NO"][0] == 9007199254740993

    def test_l03_roundtrip_from_dict_fractional_null_preserves_object(
        self,
    ) -> None:
        """L-03/EXT-04: from_dict must NOT coerce {I} arrays to int64 when
        the declared NULL is fractional (-999.25) — int64 would truncate
        the null sentinel.  The object dtype (exact ints + float sentinel)
        is preserved so to_dict → from_dict keeps >2^53 precision."""
        from pylasdev.models import LASFile

        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO"},
                "well": {"NULL": "-999.25"},
                "curves": [
                    {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                    {"mnemonic": "RUN_NO", "data_format": "I"},
                ],
                "curves_order": ["DEPT", "RUN_NO"],
                "logs": {
                    "DEPT": np.array([100.0, 101.0]),
                    "RUN_NO": np.array([5, -999.25]),
                },
            }
        )
        assert las.logs["RUN_NO"].dtype == np.object_, (
            f"Expected object dtype, got {las.logs['RUN_NO'].dtype}"
        )
        assert las.logs["RUN_NO"][0] == 5
        assert las.logs["RUN_NO"][1] == -999.25

    def test_ext04_integer_precision_fractional_null_las20(self, tmp_path: Path) -> None:
        """EXT-04: {I} curve value 9007199254740993 (2^53+1) survives
        exactly on the LAS 1.2/2.0 path with the default/fractional NULL
        -999.25 (pre-fix it rounded to 9007199254740992.0 float64)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_las20.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert int(data["logs"]["RUN_NO"][0]) == 9007199254740993, (
            f"Precision lost: {data['logs']['RUN_NO'][0]}"
        )
        assert data["logs"]["RUN_NO"][1] == -999.25, "null cell must keep the fractional sentinel"

    def test_ext04_integer_precision_fractional_null_las30(self, tmp_path: Path) -> None:
        """EXT-04: same exactness guarantee on the LAS 3.0 path with the
        default/fractional NULL -999.25."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_las30.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        run_no = las.data_sections[0].data["RUN_NO"]
        assert int(run_no[0]) == 9007199254740993, f"Precision lost: {run_no[0]}"
        assert run_no[1] == -999.25, "null cell must keep the fractional sentinel"

    def test_ext04_fractional_null_no_spurious_dtype_validate_issue(self, tmp_path: Path) -> None:
        """EXT-04 convergence: parsing a valid LAS 3.0 file with an {I}
        curve and a fractional declared NULL must NOT produce a spurious
        non-numeric-dtype validation issue.

        The object-dtype exemption must cover DataSection.validate (which
        LASFile.validate(complete=True) delegates to, and which the parser
        runs on every parse), not just the top-level logs loop — otherwise
        every valid {I} + fractional NULL file reports a false-positive
        "non-numeric dtype (object)" issue.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n"
            " STOP.M   101.0 : \n"
            " STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " RUN_NO.      :  Run Number {I}\n"
            "~A ASCII DEPT RUN_NO\n"
            " 100.0 9007199254740993\n"
            " 101.0 -999.25\n"
        )
        test_file = tmp_path / "ext04_validate.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        issues = las.validate(complete=True)
        dtype_issues = [i for i in issues if "non-numeric dtype" in i]
        assert dtype_issues == [], (
            "Spurious non-numeric dtype issue for a valid {I} + fractional "
            f"NULL file: {dtype_issues}"
        )

    # --- N-I-08 (MEDIUM): wrapped-mode trailing EOF under-fill ---

    def test_n08_trailing_eof_under_fill_warns(self, tmp_path: Path) -> None:
        """N-I-08: a wrapped file whose FINAL step is under-filled (the file
        ends with a partial step — leftover values not accounted for by the
        curve count) must emit the N-I-08 EOF warning instead of silently
        misaligning depth and curve columns.

        This fixture ends with 2 leftover values (8 total, 3 curves), so
        the N-I-08 trailing-partial warning fires.  The mid-file aligned
        under-fill case (total is a multiple of curve_count, depth column
        non-monotonic) is a DIFFERENT diagnostic — see
        test_m72_aligned_total_mid_file_advisory below."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "101.0\n"
            "51.0\n"
            "102.0\n"
            "52.0\n"
            "76.0\n"
        )
        test_file = tmp_path / "n08_underfill.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
        assert any("not accounted for by the curve count" in m for m in messages), (
            f"No N-I-08 trailing-partial warning emitted. Got: {messages[-3:]}"
        )
        # HARDENED (register): the diagnostic alone did not pin the outcome —
        # the R-6 discard contract is that the orphan values are DROPPED, not
        # padded into a phantom step.
        assert len(data["logs"]["DEPT"]) == 2, (
            f"the partial trailing step must be discarded, got {data['logs']['DEPT'].tolist()}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])
        # Every flushed step takes exactly 3 values in order; the third value of
        # step 2 is the "depth" of the truncated tail (102.0), and the orphan
        # 52.0/76.0 pair is the part that is discarded.
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 102.0])

    def test_m72_aligned_total_mid_file_advisory(self, tmp_path: Path) -> None:
        """M-72: a wrapped file whose TOTAL is a multiple of curve_count
        (aligned total) but whose depth column is non-monotonic must emit
        the M-72 non-monotonic-depth advisory.

        N-I-08 only fires on a trailing partial buffer; it is blind to a
        non-monotonic depth column that leaves the total aligned.  In that
        case every step "completes" from the reader's perspective but a
        backward depth step produces a non-monotonic depth column — the
        ONLY diagnostic is the "depth column is not monotonic" advisory at
        data_reader.py:1660-1677.

        Fixture: 2 curves, 6 values (3 steps of 2) — total 6 is a multiple
        of 2, so N-I-08 does NOT fire; the depth column [1000, 1001, 999]
        is non-monotonic (backward step), so M-72 fires.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "51.0\n"
            "999.0\n"
            "52.0\n"
        )
        test_file = tmp_path / "m72_aligned.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]

        # The shifted depth column is observable: [1000, 1001, 999].
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 999.0])
        assert any("depth column is not monotonic" in m for m in messages), (
            f"M-72 advisory not emitted. Got: {messages[-3:]}"
        )
        # N-I-08 must NOT fire — the total is a multiple of curve_count.
        assert not any("not accounted for by the curve count" in m for m in messages), (
            f"N-I-08 fired on an aligned total — wrong diagnostic. Got: {messages[-3:]}"
        )

        # HARDENED (register): only the positive direction was pinned, so an
        # ``or`` → ``and`` mutation in the M-72 gate survived.  The negative
        # direction: a file whose ONLY defect is a trailing partial step must
        # emit N-I-08 and NOT the M-72 advisory.
        partial = tmp_path / "m72_negative.las"
        partial.write_text(
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n"
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "51.0\n"
            "1002.0\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            read_las_file(partial)
            partial_messages = [str(x.message) for x in w2]
        assert any("not accounted for by the curve count" in m for m in partial_messages)
        assert not any("depth column is not monotonic" in m for m in partial_messages), (
            f"M-72 must not fire on a trailing-partial-only file. Got: {partial_messages[-3:]}"
        )

    # --- IT3-F-01 (MEDIUM): desanitize flag hoist ---

    def test_it3f01_desanitize_flag_hoist_semantics(self) -> None:
        """IT3-F-01: the hoisted desanitize flag must preserve the exact
        _# escape semantics (value-start and leading-whitespace unescape;
        M11: internal " _#" content is preserved, mirroring F-25)."""
        from pylasdev import parser as _parser_mod
        from pylasdev.data_reader import _desanitize_las_value

        prior = _parser_mod._DESANITIZE_ENABLED
        _parser_mod._DESANITIZE_ENABLED = True
        try:
            assert _desanitize_las_value("_#comment") == "#comment"
            assert _desanitize_las_value("_#comment", False) == "_#comment"
            # M11: the data_reader copy mirrors the parser's F-25 scope —
            # the writer escapes '#' ONLY at value start / after LEADING
            # whitespace, so an internal mid-value " _#" is preserved, while
            # a leading-whitespace writer escape is still restored.
            assert _desanitize_las_value("abc _#def", True) == "abc _#def"
            assert _desanitize_las_value("abc _#def", False) == "abc _#def"
            assert _desanitize_las_value(" _#comment", True) == " #comment"
            assert _desanitize_las_value("plain", True) == "plain"
            # HARDENED (register): the None-fallback case was vacuous while the
            # module flag was True (default) — it asserted the same thing as the
            # cached-True path.  With the module flag flipped, the None call must
            # follow the module flag, not a default.
            _parser_mod._DESANITIZE_ENABLED = False
            assert _desanitize_las_value("_#x") == "_#x", (
                "the None-cache fallback must read the module flag (False here)"
            )
            _parser_mod._DESANITIZE_ENABLED = True
            assert _desanitize_las_value("_#x") == "#x"
        finally:
            _parser_mod._DESANITIZE_ENABLED = prior

    # --- IT3-F-02 (MEDIUM): math.isfinite swap ---

    def test_it3f02_to_finite_float_non_finite(self) -> None:
        """IT3-F-02: _to_finite_float must still null non-finite values
        after the np.isfinite → math.isfinite swap."""
        from pylasdev.data_reader import _to_finite_float

        for bad in ("nan", "inf", "-inf", "1e309", "1.0D309"):
            assert _to_finite_float(bad, -999.25) == -999.25, f"{bad} not nulled"
        assert _to_finite_float("50.5", -999.25) == 50.5
        assert _to_finite_float("", -999.25) == -999.25

    def test_it3f02_writer_math_isfinite(self) -> None:
        """IT3-F-02: writer _format_number must still route NaN/Inf to the
        null sentinel after the np.isnan/np.isinf → math.isfinite swap.

        KEPT + HARDENED (register DISPUTED-DELETE resolved conservatively): the
        finite side asserted only one value; a sentinel that leaked into the
        zero/negative paths would have passed.  Finite zero and a negative value
        are pinned exactly alongside the non-finite routing.
        """
        from pylasdev._writer_base import _format_number

        for bad in (float("nan"), float("inf"), float("-inf")):
            assert _format_number(bad, ".8g", -999.25) == "-999.25", (
                f"{bad} not routed to null sentinel"
            )
        assert _format_number(50.5, ".8g", -999.25) == "50.5"
        assert _format_number(-1.5, ".8g", -999.25) == "-1.5"
        assert _format_number(0.0, ".8g", -999.25) == "0", "a finite 0.0 is not a null"

    # --- IT3-F-03 (MEDIUM): wrapped pre-allocation ---

    def test_it3f03_wrapped_prealloc_correct(self, tmp_path: Path) -> None:
        """IT3-F-03: _read_wrapped pre-allocated numeric columns must
        produce identical results to the previous list-accumulation path."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            "100.0\n"
            "50.0\n"
            "75.0\n"
            "101.0\n"
            "51.0\n"
            "76.0\n"
        )
        test_file = tmp_path / "it3f03_wrapped.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])
        np.testing.assert_allclose(data["logs"]["GR"], [75.0, 76.0])

    def test_it3f03_wrapped_growth_capacity(self, tmp_path: Path) -> None:
        """IT3-F-03: pre-allocated wrapped columns must grow beyond the
        depth-step estimate (files denser than the ceil estimate)."""
        # 2 curves, 22 data lines: 10 single-value steps (20 lines) plus 2
        # two-value steps (2 lines).  The depth-step estimate
        # ceil(22/2)=11 undercounts the actual 12 steps — the two-value
        # lines make actual steps (12) exceed the line-count estimate (11),
        # forcing the pre-allocated column to grow (growth body fires 2x).
        lines: list[str] = []
        for step in range(10):
            lines.append(f"{100.0 + step}")
            lines.append(f"{10.0 + step}")
        # Two-value lines MUST go at the END: placing them first flips the
        # wrap detector to non-wrapped (the leading [2,2,2,2] window reads
        # as full rows) and misaligns the columns.  The leading single-value
        # window is what the test relies on for wrapped classification.
        lines.append("110.0 20.0")
        lines.append("111.0 21.0")
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A\n" + "\n".join(lines) + "\n"
        )
        test_file = tmp_path / "it3f03_growth.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0 + s for s in range(12)])
        np.testing.assert_allclose(data["logs"]["DT"], [10.0 + s for s in range(12)])

    def test_it3f03_wrapped_string_depth_curve(self, tmp_path: Path) -> None:
        """IT3-F-03: _read_wrapped must handle a wrapped file whose DEPTH
        curve (index 0) is a {S} string curve — the pre-allocated numeric
        columns path must not KeyError on a string depth line."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " HOLE.M   :  Hole {S}\n"
            " GR.GAPI  :  Gamma Ray {F}\n"
            "~A\n"
            "A\n50.0\nB\n51.0\n"
        )
        test_file = tmp_path / "it3f03_string_depth.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert data["string_data"]["HOLE"].tolist() == ["A", "B"]
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])


# ──────────────────────────────────────────────────────────────
# P-16 (reader-side, MEDIUM): unrecognized ~section header inside
# ~A must terminate data reading (matching the parser's
# section-boundary classification).
# ──────────────────────────────────────────────────────────────


class TestP16ReaderUnrecognizedSection:
    """P-16: `_iter_ascii_data_lines` must STOP reading data when it
    encounters an unrecognized ~section header (break, not
    skip-and-continue), matching the parser's section-boundary
    classification.  Previously the body of an unrecognized section
    (e.g. ~CUSTOMSECT) was consumed as data rows — the same physical
    lines landed in BOTH logs AND other_lines (garbage rows +
    duplicated text)."""

    def test_customsect_body_not_in_logs(self, tmp_path: Path) -> None:
        """~CUSTOMSECT after ~A: body lines go to other_lines only, loudly.

        HARDENED (register, primary review): the routing assertions were strong
        but the "Data reading stops at this section boundary" diagnostic was
        never asserted, so a silent stop would have passed.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1670.0  : START DEPTH\n"
            " STOP.M   1660.0  : STOP DEPTH\n"
            " STEP.M   0.1     : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            " 10.0  50.0\n"
            " 11.0  51.0\n"
            "~CUSTOMSECT\n"
            " 30.0  70.0\n"
            " 40.0  80.0\n"
        )
        test_file = tmp_path / "p16_customsect.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)

        # Only the ~A rows may land in logs.
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])
        # The ~CUSTOMSECT body goes to other_lines (parser-side routing).
        assert "~CUSTOMSECT" in data["other"]
        assert "30.0  70.0" in data["other"]
        assert "40.0  80.0" in data["other"]
        # ... and the stop is announced, naming the header.
        stops = [
            str(x.message)
            for x in caught
            if "Data reading stops at this section boundary" in str(x.message)
        ]
        assert len(stops) == 1, f"expected one section-stop diagnostic, got {stops}"
        assert "'~CUSTOMSECT'" in stops[0], stops[0]

    def test_unrecognized_section_with_trailing_data(self, tmp_path: Path) -> None:
        """Data after the unrecognized section header is not consumed as
        ~A rows either (the section boundary ends the ASCII block)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~A\n"
            " 10.0  50.0\n"
            "~Units\n"
            " 99.0  99.0\n"
        )
        test_file = tmp_path / "p16_units.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0])
        assert "~Units" in data["other"]

    def test_tilde_garbage_line_not_data_row(self, tmp_path: Path) -> None:
        """~-prefixed lines that are NOT section headers (e.g. ~., ~#)
        must not be consumed as data rows — the parser routes them to
        other_lines (N-2 adjacent defect, same divergence class)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A\n"
            " 10.0\n"
            "~.\n"
            " 11.0\n"
        )
        test_file = tmp_path / "p16_garbage.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        assert "~." in data["other"]


# ──────────────────────────────────────────────────────────────
# P-05 (reader-side, MEDIUM): LAS 1.2/2.0 no-space pipe ASCII
# header (~ASCII|CURVE) must be recognized by the reader.
# ──────────────────────────────────────────────────────────────


class TestP05ReaderNoSpacePipeAscii:
    """P-05: `_is_ascii_section` (and data_reader's section-word
    detection) must strip the `| <target>` pipe suffix so a LAS 1.2/2.0
    `~ASCII|CURVE` header is recognized as a data section.  Previously
    the parser recognized it but the reader silently read ZERO rows."""

    @pytest.mark.parametrize("version", ["1.2", "2.0"])
    def test_ascii_pipe_curve_parses_data(self, tmp_path: Path, version: str) -> None:
        """A ``~ASCII|CURVE`` header produces non-empty data rows.

        MERGED (register): the 1.2 and 2.0 twins exercise the same
        version-independent ``_get_section_word`` strip — both inputs are kept
        as parameters.
        """
        content = (
            "~VERSION INFORMATION\n"
            f" VERS.   {version}  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma Ray\n"
            "~ASCII|CURVE\n"
            " 10.0  50.0\n"
            " 11.0  51.0\n"
        )
        test_file = tmp_path / f"p05_ascii_pipe_{version}.las"
        test_file.write_text(content, encoding="utf-8")
        data = read_las_file(test_file)
        assert len(data["logs"]["DEPT"]) == 2, (
            f"~ASCII|CURVE header must produce data rows — got {len(data['logs']['DEPT'])}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])


# ──────────────────────────────────────────────────────────────
# M-11 (MEDIUM): mandatory_well_fields aligned with lascheck —
# UWI removed (optional), COMP/FLD/DATE added (required).
# ──────────────────────────────────────────────────────────────


class TestM11MandatoryWellFields:
    """M-11: `_LASVersionSpec.mandatory_well_fields` must match lascheck's
    10-field ~W set for LAS 2.0 (STRT, STOP, STEP, NULL, COMP, WELL,
    FLD, LOC, SRVC, DATE).  UWI is optional — a common LAS file with
    no UWI must NOT warn.  LAS 1.2 is narrower (I2-07): COMP/FLD/DATE
    are 2.0-era requirements, NOT mandatory for 1.2, so a 1.2 file
    carrying only the four numeric fields must NOT warn either.

    MERGED (register): the "no UWI" and "no COMP/FLD/DATE" tests asserted the
    same 1.2 contract on the same parser path (the second was a strict subset
    of the first); both field sets stay as parameters.
    """

    @pytest.mark.parametrize("include_uwi", [True, False], ids=["with_uwi", "without_uwi"])
    def test_las12_mandatory_set_draws_no_warning(self, include_uwi: bool) -> None:
        """A LAS 1.2 file with the four numeric fields draws ZERO
        mandatory-field warnings, whether or not UWI is present."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.5 : START DEPTH\n"
            " STOP.M   500.0  : STOP DEPTH\n"
            " STEP.M   -0.125 : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " WELL.    W-1 : WELL NAME\n"
            " LOC.     12-34 : LOCATION\n"
            " SRVC.    LOGCO : SERVICE COMPANY\n"
            + (
                " UWI.     10006170502W500 : UNIQUE WELL ID\n"
                if include_uwi
                else " COMP.    ACME : COMPANY\n"
                " FLD.     NORTH : FIELD\n"
                " DATE.    15/01/2001 : LOG DATE\n"
            )
            + "~CURVE INFORMATION\n"
            " DEPT.M   : DEPTH\n"
            "~A\n"
            " 100.0\n"
        )
        parser = LASParser()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parser.parse(content)
            mandatory_warnings = [
                str(x.message) for x in w if "missing mandatory well field" in str(x.message)
            ]
            assert mandatory_warnings == [], (
                f"LAS 1.2 (UWI present: {include_uwi}) must not warn about any "
                f"mandatory field; got: {mandatory_warnings}"
            )


# ──────────────────────────────────────────────────────────────
# ENC-03 (reader, MEDIUM): max_file_size ValueError conversion +
# LASEncodingError mislabeling.  README.md:346-360 documented ValueError
# (catch separately) but code raises LASReadError — the README was the
# sole outlier (code+docstrings+tests agree on LASReadError) and has been
# corrected.  A genuine LASEncodingError must also NOT be swallowed under
# a misleading "size exceeded" message: it propagates as LASEncodingError.
# ──────────────────────────────────────────────────────────────


class TestENC03SizeLimitErrorContract:
    """ENC-03: size-limit failures raise LASReadError (documented
    contract); genuine encoding failures raise LASEncodingError with an
    accurate message — never a misleading 'size exceeded' LASReadError.

    MERGED (register): the former ``test_max_file_size_raises_las_read_error``
    asserted the identical exception + message as
    ``TestDataReaderEdgeCases::test_max_file_size_limit`` (which now also pins
    the accepted at-limit side and the file path in the message), so it was
    consolidated into that survivor.
    """

    def test_genuine_decode_failure_raises_las_encoding_error(self, tmp_path: Path) -> None:
        """A genuine decoding failure (explicit encoding that cannot decode
        the bytes) raises LASEncodingError with an accurate message — NOT a
        misleading 'size exceeded or invalid parameter' LASReadError."""
        from pylasdev.exceptions import LASEncodingError

        test_file = tmp_path / "enc03_bad_encoding.las"
        # CP1252 bytes are invalid UTF-8, so forcing utf-8 must fail to decode.
        test_file.write_bytes("Caf\u00e9 r\u00e9sum\u00e9".encode("cp1252"))
        with pytest.raises(LASEncodingError, match="Failed to decode"):
            read_las_file(test_file, encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# F-02 (data_reader, MEDIUM, REGRESSION): the F-07 depth-line
# evidence rule's ``window[1] == 1`` arm fired for ANY single
# 1-value row after a full first row when curve_count >= 3,
# misclassifying a ragged NON-wrapped nc>=3 file with a single
# 1-value middle row ([3,1,3] / [3,1,2]) as WRAPPED → silent
# column-shift corruption (the genuine depth value swallowed into
# curve 1, trailing values discarded).  A single 1-value row is
# ragged-row evidence (graceful short-row null-fill), not
# unambiguous depth evidence — only TWO+ 1-value rows trigger the
# wrapped arm.  Mirrored on the LAS 3.0 path (_las30_data.py) and
# covered there by TestF02Las30RaggedMiddleRowNotWrapped.
# ──────────────────────────────────────────────────────────────


class TestF02ThreeCurveShortMiddleRowNotWrapped:
    """F-02: 3-curve WRAP=NO files with a single 1-value middle row must
    NOT be classified wrapped (regression — pre-fix the nc>=3
    window[1]==1 arm silently column-shifted the file)."""

    def test_las20_three_curve_short_middle_row_not_wrapped(self, tmp_path: Path) -> None:
        """3-curve LAS 2.0 WRAP=NO `100.0 50.0 30.0 / 101.0 / 102.0 60.0 40.0`
        (window [3,1,3]): DEPT=[100,101,102], C1=[50,-999.25,60],
        C2=[30,-999.25,40].  Pre-fix: WRAPPED=True → DEPT=[100,101],
        C1=[50,102], C2=[30,60] — the depth value 102.0 swallowed into
        C1 and the genuine 40.0 discarded."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0 40.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_short_middle.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, -999.25, 40.0])

    def test_las20_three_curve_short_middle_two_value_row_not_wrapped(self, tmp_path: Path) -> None:
        """[3,1,2] shape: `100.0 50.0 30.0 / 101.0 / 102.0 60.0` — the
        single 1-value middle row must stay ragged (non-wrapped):
        DEPT=[100,101,102], C1=[50,-999.25,60], C2=[30,-999.25,-999.25].
        Pre-fix this silently produced DEPT=[100,101], C1=[50,102],
        C2=[30,60] with no warning at all."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_short_middle2.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, -999.25, 60.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, -999.25, -999.25])

    def test_las20_three_curve_wrap_yes_short_middle_row_still_wrapped(
        self, tmp_path: Path
    ) -> None:
        """Control: the [3,1,3] shape with WRAP=YES declared must STILL
        be classified wrapped (declared-YES + depth evidence → wrapped;
        the gate must not regress genuine wrapped detection)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " C1.GAPI  :  Curve 1\n"
            " C2.K/M3  :  Curve 2\n"
            "~A DEPT C1 C2\n"
            "100.0 50.0 30.0\n"
            "101.0\n"
            "102.0 60.0 40.0\n"
        )
        test_file = tmp_path / "f02_las20_three_curve_wrap_yes.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        # Wrapped parse: DEPT holds the depth-line values only.
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["C1"], [50.0, 102.0])
        np.testing.assert_allclose(data["logs"]["C2"], [30.0, 60.0])

    def test_detector_single_one_value_row_never_wrapped(self) -> None:
        """Detector-level audit: a SINGLE 1-value row (in any window
        position) after a full first row is ragged — never wrapped.
        Two+ 1-value rows (masquerade/mixed-wrap) stay wrapped."""
        from pylasdev.data_reader import _detect_actual_wrap

        def lines_from_window(window: list[int]) -> list[str]:
            out = ["~A DEPT GR RHOB"]
            for n in window:
                out.append(" ".join(str(i) for i in range(1, n + 1)))
            return out

        # (window, curve_count, declared_wrap, expected_wrapped)
        shapes = [
            ([3, 1, 3], 3, "NO", False),  # F-02: ragged single middle 1-value row
            ([3, 1, 2], 3, "NO", False),  # F-02: ragged single middle 1-value row
            ([3, 1, 3], 3, None, False),  # absent declaration behaves like NO
            ([3, 1, 2], 3, None, False),  # absent declaration behaves like NO
            ([3, 1, 1], 3, "NO", True),  # mnemonic-header masquerade (2 one-value rows)
            ([3, 1, 2, 1], 3, "NO", True),  # I2-03 mixed-wrap (2 one-value rows)
            ([3, 1, 2, 1], 3, None, True),  # mixed-wrap without declaration
            ([3, 1, 3], 3, "YES", True),  # declared YES stays wrapped
            ([3, 2, 1], 3, "NO", False),  # ragged trailing 1-value row (unchanged)
        ]
        for window, nc, decl, expected in shapes:
            got = _detect_actual_wrap(lines_from_window(window), nc, " ", declared_wrap=decl)
            assert got is expected, (
                f"window={window} nc={nc} decl={decl}: expected wrapped={expected}, got {got}"
            )


class TestIsMnemonicHeaderRowSingleCurve:
    """PSR-1 (Stage 11): _is_mnemonic_header_row's token-count gate must
    use min(2, curve_count), not a flat 2.  A SINGLE-curve section must
    still recognize its 1-token standalone mnemonic header row
    ("~A\\nDEPT\\n1670.0\\n...") — the DR-M2 `< 2` gate consumed it as
    data, producing a phantom all-null first row + value shift.  The
    2-token minimum is preserved for multi-curve sections (M-02), and the
    all-string exclusion still protects single-curve STRING sections
    (M-03/F-19)."""

    @staticmethod
    def _las(mnemonics: list[str]) -> LASFile:
        las = LASFile()
        las.curves_order = list(mnemonics)
        for name in mnemonics:
            las.curves.append(CurveDefinition(mnemonic=name, unit="M"))
        return las

    def test_single_curve_one_token_mnemonic_is_header(self) -> None:
        """PSR-1: a 1-token row equal to the sole curve's mnemonic is a
        header in a single-curve section (pre-fix `< 2` returned False —
        the phantom-row defect)."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["DEPT"], las, 1, set()) is True

    def test_single_curve_one_token_non_mnemonic_is_data(self) -> None:
        """A 1-token numeric data row is not a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["1670.0"], las, 1, set()) is False

    def test_single_curve_two_tokens_not_header(self) -> None:
        """A 2-token row in a 1-curve section exceeds curve_count — it is
        an extra-column data row, never a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT"])
        assert _is_mnemonic_header_row(["DEPT", "GR"], las, 1, set()) is False

    def test_two_curve_one_token_not_header(self) -> None:
        """M-02: the 2-token minimum stays intact for multi-curve
        sections — a 1-token row cannot be a full header signature."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR"])
        assert _is_mnemonic_header_row(["DEPT"], las, 2, set()) is False

    def test_single_curve_string_section_one_token_not_header(self) -> None:
        """M-03/F-19: in a single-curve all-STRING section the all-string
        exclusion fires — a mnemonic-coincident value is data, not a
        header (min(2, 1) alone must NOT turn string sections into header
        droppers)."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["LITH"])
        assert _is_mnemonic_header_row(["LITH"], las, 1, {0}) is False

    def test_multi_curve_partial_header_still_header(self) -> None:
        """M12/DR-M2: the partial-header relaxation (2..curve_count) stays
        intact — "DEPT GR" with 3 declared curves is still a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR", "RHOB"])
        assert _is_mnemonic_header_row(["DEPT", "GR"], las, 3, set()) is True

    def test_multi_curve_full_header_still_header(self) -> None:
        """The full-width header remains a header."""
        from pylasdev.data_reader import _is_mnemonic_header_row

        las = self._las(["DEPT", "GR", "RHOB"])
        assert _is_mnemonic_header_row(["DEPT", "GR", "RHOB"], las, 3, set()) is True


# ──────────────────────────────────────────────────────────────
# W-A / W-2 / H-1 / F-02 regression tests (IMPLEMENT read-path pass)
# Each test FAILS on the pre-refactor library and PASSES on the new code.
# ──────────────────────────────────────────────────────────────


def _write_las(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _flowing_content(version: str, wrap: str, dlm: str = "SPACE") -> str:
    """Flowing [6,6] nc=3 content: 2 complete depth steps per line."""
    dlm_line = f" DLM.    {dlm} :\n" if dlm != "SPACE" else ""
    if dlm == "COMMA":
        rows = "1000.0,10.0,20.0,1001.0,11.0,21.0\n1002.0,12.0,22.0,1003.0,13.0,23.0\n"
    elif dlm == "TAB":
        rows = "1000.0\t10.0\t20.0\t1001.0\t11.0\t21.0\n1002.0\t12.0\t22.0\t1003.0\t13.0\t23.0\n"
    else:
        rows = "1000.0 10.0 20.0 1001.0 11.0 21.0\n1002.0 12.0 22.0 1003.0 13.0 23.0\n"
    return (
        "~VERSION INFORMATION\n"
        f" VERS.   {version}  : CWLS LOG ASCII STANDARD\n"
        f" WRAP.   {wrap}  :\n"
        f"{dlm_line}"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        " GR.GAPI  :  Gamma\n"
        " RHOB.K/M3:  Density\n"
        "~A\n"
        f"{rows}"
    )


class TestWrapFlowingAccumulation:
    """W-A (HIGH): the flowing layout (depth NOT on its own line — 2+
    complete depth steps per line) must parse with ALL steps preserved.

    Pre-fix the reader had NO working path for flowing data: the detector
    misrouted all-full windows to _read_normal (2 of 4 steps silently
    lost) and _read_wrapped's depth-line protocol corrupted it worse (3
    of 4 lost).  The n_curves-accumulation rewrite + the multiple-of
    detection rule fix both."""

    @pytest.mark.parametrize("version", ["1.2", "2.0"])
    def test_flowing_wrap_yes(self, tmp_path: Path, version: str) -> None:
        """A flowing layout parses with ALL steps preserved.

        MERGED (register): the LAS 1.2 and 2.0 twins were byte-identical
        except the VERS line and the reader receives no version — one
        parametrized test keeps both inputs.
        """
        test_file = _write_las(tmp_path, f"flow_yes_{version}.las", _flowing_content(version, "YES"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, 21.0, 22.0, 23.0])

    def test_flowing_wrap_no_declaration_independent(self, tmp_path: Path) -> None:
        """W-A's trigger is declaration-INDEPENDENT: a WRAP=NO header must
        not hide the flowing signature (the multiple-of rule is content-
        based, placed before the declared-header fall-through)."""
        test_file = _write_las(tmp_path, "flow_no.las", _flowing_content("2.0", "NO"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_with_string_curve(self, tmp_path: Path) -> None:
        """Flowing with a {S} string curve: step-position dispatch must
        preserve string values in order (pre-fix S12: 2 of 4 lost)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "1000.0 SAND 10.0 1001.0 SHALE 11.0\n"
            "1002.0 LIME 12.0 1003.0 DOLO 13.0\n"
        )
        test_file = _write_las(tmp_path, "flow_str.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"],
            np.array(["SAND", "SHALE", "LIME", "DOLO"], dtype=object),
        )
        np.testing.assert_allclose(las.logs["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_comma_delimiter(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_comma.las", _flowing_content("2.0", "YES", "COMMA"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_tab_delimiter(self, tmp_path: Path) -> None:
        test_file = _write_las(tmp_path, "flow_tab.las", _flowing_content("1.2", "YES", "TAB"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0, 12.0, 13.0])

    def test_flowing_trailing_partial_step_warns(self, tmp_path: Path) -> None:
        """A flowing section ending with an incomplete step (2 trailing
        values, nc=3) emits the N-I-08-style "not accounted for" warning
        and the orphan values are discarded (R-6 contract)."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3:  Density\n"
            "~A\n"
            "1000.0 10.0 20.0 1001.0 11.0 21.0\n"
            "1002.0 12.0\n"
        )
        test_file = _write_las(tmp_path, "flow_partial.las", content)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = read_las_file(test_file)
            messages = [str(x.message) for x in w]
        assert any("under-filled" in m or "not accounted for" in m for m in messages), (
            f"No trailing-step warning emitted. Got: {messages[-3:]}"
        )
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, 11.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, 21.0])


class TestWrapDetectorFlowingRule:
    """Detector-unit locks for the R-1 multiple-of flowing rule and the
    R-3/II-5 discriminator (all FAIL pre-fix, PASS post-fix)."""

    def _detect(self, window: list[int], nc: int, decl: str | None) -> bool:
        from pylasdev.data_reader import _detect_actual_wrap

        lines = ["~A DEPT GR"]
        for n in window:
            lines.append(" ".join(str(i) for i in range(1, n + 1)))
        return _detect_actual_wrap(lines, nc, " ", declared_wrap=decl)

    def test_flowing_all_full_detected_wrapped(self) -> None:
        """[6,6] nc=3 → True (W-A): a full first line carrying 2 complete
        depth steps is flowing, regardless of the declared header."""
        assert self._detect([6, 6], 3, "YES") is True
        assert self._detect([6, 6], 3, "NO") is True

    def test_mislabeled_all_full_stays_unwrapped(self) -> None:
        """[3,3,3,3] nc=3 YES → False (test_regression.py:4708 lock): one
        complete step per line is NOT flowing."""
        assert self._detect([3, 3, 3, 3], 3, "YES") is False

    def test_extra_columns_aligned_stays_unwrapped(self) -> None:
        """[4,4] nc=3 → False (4 % 3 != 0): the extra-columns control —
        a non-aligned all-full window stays non-wrapped so _read_normal's
        extra-column discard is correct (QA N2 hazard)."""
        assert self._detect([4, 4], 3, "YES") is False

    def test_nc2_aligned_extra_columns_stays_unwrapped(self) -> None:
        """[4,4] nc=2 → False: for nc=2 the multiple-of rule is disabled
        (curve_count >= 3 guard) so test-locked extra-columns files are
        never misrouted to the wrapped reader."""
        assert self._detect([4, 4], 2, "NO") is False
        assert self._detect([4, 4], 2, "YES") is False

    def test_ragged_two_one_value_rows_stays_unwrapped(self) -> None:
        """W-2/II-5: [3,1,3,1] and [3,1,1,3] WRAP=NO → False — a full row
        immediately after a 1-value row is impossible in genuine wrapped
        data, so the ≥2-one-value-rows arm must not fire."""
        assert self._detect([3, 1, 3, 1], 3, "NO") is False
        assert self._detect([3, 1, 1, 3], 3, "NO") is False


class TestWrapRaggedNullFill:
    """W-2 (MEDIUM): ragged WRAP=NO files with ≥2 one-value rows must
    null-fill (not corrupt via the wrapped reader)."""

    _base = (
        "~VERSION INFORMATION\n"
        " VERS.   1.2  : CWLS LOG ASCII STANDARD\n"
        " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
        "~WELL INFORMATION\n"
        " NULL.    -999.25 : NULL VALUE\n"
        "~CURVE INFORMATION\n"
        " DEPT.M   :  Depth\n"
        " GR.GAPI  :  Gamma\n"
        " RHOB.K/M3:  Density\n"
        "~A\n"
    )

    def test_ragged_3131_wrap_no_null_fill(self, tmp_path: Path) -> None:
        content = self._base + ("1000.0 10.0 20.0\n1001.0\n1002.0 12.0 22.0\n1003.0\n")
        test_file = _write_las(tmp_path, "r3131.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, -999.25, 12.0, -999.25])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, -999.25, 22.0, -999.25])

    def test_ragged_3113_wrap_no_null_fill(self, tmp_path: Path) -> None:
        content = self._base + ("1000.0 10.0 20.0\n1001.0\n1002.0\n1003.0 13.0 23.0\n")
        test_file = _write_las(tmp_path, "r3113.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(data["logs"]["GR"], [10.0, -999.25, -999.25, 13.0])
        np.testing.assert_allclose(data["logs"]["RHOB"], [20.0, -999.25, -999.25, 23.0])


class TestWrapLas30FlowingRejection:
    """R-5: the LAS 3.0 fix is DETECTION-ONLY — the shared gate's True
    feeds the existing loud WRAP=YES rejection (no accumulation in
    process_ascii_data); ragged WRAP=NO files parse instead of being
    falsely rejected (NEW-1)."""

    def test_las30_flowing_six_six_raises_loudly(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
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
        test_file = _write_las(tmp_path, "l30_flow.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(LASParseError, match="WRAP=YES is not supported"):
                read_las_file(test_file)

    def test_las30_ragged_3131_wrap_no_parses(self, tmp_path: Path) -> None:
        """NEW-1: a ragged WRAP=NO file must NOT be falsely rejected as
        "LAS 3.0 WRAP=YES" — the II-5 discriminator keeps it non-wrapped
        and it parses with null-fill."""
        content = (
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
            "1000.0 10.0 20.0\n"
            "1001.0\n"
            "1002.0 12.0 22.0\n"
            "1003.0\n"
        )
        test_file = _write_las(tmp_path, "l30_ragged.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0, 1003.0])
        np.testing.assert_allclose(las.logs["GR"], [10.0, -999.25, 12.0, -999.25])


class TestHeaderSkipSupersetSplit:
    """H-1 (MEDIUM): the DLM-aware reader must recognize a space-separated
    mnemonic header row in a DLM=COMMA file (superset split at all 3
    predicate sites), and the II-11 collision class (a genuine first-row
    string value containing a space in a mixed COMMA section) must NOT be
    skipped as a header."""

    def test_space_separated_header_in_comma_file_skipped(self, tmp_path: Path) -> None:
        """Pre-fix: the reader split the header row with the DLM-aware
        tokenizer → one token "DEPT GR" → consumed as data → phantom
        all-null first row + one-row shift."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "100.0,50.0\n"
            "101.0,51.0\n"
        )
        test_file = _write_las(tmp_path, "h1_comma_header.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = read_las_file(test_file)
        np.testing.assert_allclose(data["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(data["logs"]["GR"], [50.0, 51.0])

    def test_space_in_string_first_row_not_header(self, tmp_path: Path) -> None:
        """II-11 collision class: a genuine first-row {S} string value
        containing a space ("LITH SHALE") in a COMMA-DLM mixed section is
        split by the superset tokenizer into 4 tokens (> curve_count), so
        the count bound keeps it DATA — never skipped as a header."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " LITH.    :  Lithology {S}\n"
            "~A DEPT GR LITH\n"
            "100.0,10.0,LITH SHALE\n"
            "101.0,11.0,SAND STONE\n"
        )
        test_file = _write_las(tmp_path, "h1_collision.las", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"],
            np.array(["LITH SHALE", "SAND STONE"], dtype=object),
        )


class TestHeaderTildeRoundtrip:
    """F-02/II-26: header fields must NOT restore the writer-only '_~'
    data escape — a genuine '_~'-prefixed header value roundtrips
    unchanged (parser header call sites use restore_tilde=False, II-13)."""

    def test_well_value_underscore_tilde_roundtrips_unchanged(self, tmp_path: Path) -> None:
        from pylasdev.writer import write_las_file

        las = LASFile()
        las.version.wrap = "NO"
        las.well["NULL"] = "-999.25"
        las.well["WELL"] = "_~Acme"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        out = tmp_path / "tilde_well.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = read_las_file_as_object(out)
        # Pre-fix: parser._desanitize_las_value restored '_~' → '~' at the
        # header site, corrupting the genuine value to '~Acme'.
        assert back.well["WELL"] == "_~Acme", (
            f"F-02: '_~' header escape wrongly restored, got {back.well['WELL']!r}"
        )


class TestE20WrappedPendingBufferBounded:
    """E-20 (CONFIRMED MEDIUM): _read_wrapped's pending buffer retained
    every consumed token string until EOF (~58 GB on crafted WRAP=YES
    files before the MAX_TOTAL_ELEMENTS guard fires).  The fix trims the
    consumed prefix once read_idx crosses _PENDING_TRIM_THRESHOLD
    (amortized O(1) per token) while keeping the O(1)-per-step
    extraction and the N-I-08 trailing-step diagnostic."""

    @staticmethod
    def _wrapped_content(n_steps: int) -> str:
        rows = []
        for i in range(n_steps):
            rows.append(f"{1000.0 + i:.1f}")
            rows.append(f"{50.0 + 5 * i:.1f} {1.0 + i:.1f}")
        return (
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
            + "\n".join(rows)
            + "\n"
        )

    def test_trim_preserves_exact_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrapped file crossing the trim threshold many times must parse
        to exactly the same values as an untrimmed read (the trim must not
        shift, drop, or duplicate buffered values)."""
        from pylasdev import data_reader as dr

        # Force a trim every few tokens so the test exercises many trims.
        monkeypatch.setattr(dr, "_PENDING_TRIM_THRESHOLD", 8)
        n_steps = 40
        test_file = _write_las(tmp_path, "e20_trim.las", self._wrapped_content(n_steps))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(
            las.logs["DEPT"], [1000.0 + i for i in range(n_steps)]
        )
        np.testing.assert_allclose(
            las.logs["GR"], [50.0 + 5 * i for i in range(n_steps)]
        )
        np.testing.assert_allclose(
            las.logs["RHOB"], [1.0 + i for i in range(n_steps)]
        )

    def test_pending_retention_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """E-20/M-10: consumed tokens must not stay alive until EOF.  With
        the pre-fix code every token string from _split_data_line stays
        referenced by `pending` (a 12K-token file retains ~600 KB of token
        objects at peak); post-fix the buffer holds only the unconsumed
        tail.  Retention is measured DURING parse via the
        _read_wrapped_trace_hook seam (while `pending` is still live) — a
        snapshot AFTER _read_wrapped returns sees the freed list and
        cannot distinguish trimmed from untrimmed behavior (M-10: the old
        post-parse snapshot passed both ways, so removing the E-20 trim
        went undetected by CI)."""
        import tracemalloc

        from pylasdev import data_reader as dr

        monkeypatch.setattr(dr, "_PENDING_TRIM_THRESHOLD", 64)
        n_steps = 4000  # 12_000 tokens — pre-fix retains every one of them
        test_file = _write_las(tmp_path, "e20_retention.las", self._wrapped_content(n_steps))

        snapshots: list[Any] = []
        calls = 0

        def _trace_hook(_pending: list[str]) -> None:
            nonlocal calls
            calls += 1
            # Sample every 128th line (~62 snapshots for 8_000 lines) —
            # tracemalloc.take_snapshot is too expensive for every line.
            if calls % 128 == 1:
                snapshots.append(tracemalloc.take_snapshot())

        monkeypatch.setattr(dr, "_read_wrapped_trace_hook", _trace_hook)

        tracemalloc.start()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                las = read_las_file_as_object(test_file)
        finally:
            tracemalloc.stop()

        # Sanity: the file parsed completely.
        np.testing.assert_allclose(las.logs["DEPT"][-1], 1000.0 + n_steps - 1)

        def _retained(snapshot: Any) -> int:
            retained = 0
            for stat in snapshot.statistics("lineno"):
                fname = str(stat.traceback[0].filename)
                if "_data_section_reader" in fname or "data_reader" in fname:
                    retained += stat.size
            return retained

        assert snapshots, "M-10: trace hook never fired — the seam is not being invoked"
        peak = max(_retained(s) for s in snapshots)
        # Pre-fix (no trim): at peak `pending` holds all ~12_000 tokens
        # (~600 KB).  Post-fix: the buffer holds only the unconsumed tail
        # (< threshold + one line) at every sampled point of the parse.
        assert peak < 200_000, (
            f"E-20: {peak} bytes of token objects still retained during "
            f"parse — the pending buffer is not being trimmed"
        )


class TestE41WrapDetectorHeaderRowNotCounted:
    """E-41 (CONFIRMED MEDIUM): _detect_actual_wrap's 4-line window counts
    the standalone mnemonic header row as a full data line, flipping
    genuinely wrapped >=3-curve data to non-wrapped → silent column shift
    of the whole depth log.  The fix skips mnemonic-header rows while the
    window is empty.  MUST NOT regress test_regression.py:2377
    ([3,1,2,1] wrapped lock)."""

    @staticmethod
    def _content(with_header: bool, wrap: str = "NO") -> str:
        header_row = "DEPT  GR  RHOB\n" if with_header else ""
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            f" WRAP.   {wrap}  : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A\n"
            + header_row
            # Data: continuation lines of 2 values + depth lines of 1 value.
            # Window WITHOUT the header row: [2,2,1,2] → partial majority →
            # wrapped.  WITH the header row counted: [3,2,2,1] → first-line
            # full + WRAP=NO → non-wrapped (pre-fix silent shift).
            + "50.0 1.0\n"
            + "1000.0\n"
            + "55.0 2.0\n"
            + "1001.0\n"
            + "60.0 3.0\n"
            + "1002.0\n"
        )

    def test_wrapped_with_header_parses_identically_to_without(
        self, tmp_path: Path
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with_header = read_las_file_as_object(
                _write_las(tmp_path, "e41_with_header.las", self._content(True))
            )
            without_header = read_las_file_as_object(
                _write_las(tmp_path, "e41_without_header.las", self._content(False))
            )
        # Pre-fix: the header row (full width) + WRAP=NO → non-wrapped →
        # DEPT=[50,1000,55,1001,60,1002] (silent shift).  Post-fix both
        # variants are wrapped and identical.
        np.testing.assert_allclose(with_header.logs["DEPT"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(with_header.logs["GR"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(with_header.logs["RHOB"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(without_header.logs["DEPT"], with_header.logs["DEPT"])
        np.testing.assert_allclose(without_header.logs["GR"], with_header.logs["GR"])
        np.testing.assert_allclose(without_header.logs["RHOB"], with_header.logs["RHOB"])

    def test_comma_dlm_space_header_not_flipped_to_wrapped(self, tmp_path: Path) -> None:
        """E-41 iter-3 NEW TRIGGER VARIANT: DLM=COMMA + space-separated
        header row → the comma split sees ONE token ("DEPT GR RHOB NPHI")
        → the window starts [1,3,3,3] which is NOT uniform (the header's
        1-token entry breaks the H-02 shape) → partial majority → WRAPPED
        verdict (pre-fix) → silent column shift of every row.  Post-fix
        the header row is not counted → window [3,3,3] → H-02 uniform-short
        → non-wrapped → graceful short-row null-fill."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A\n"
            "DEPT  GR  RHOB  NPHI\n"
            "1000.0,50.0,1.0\n"
            "1001.0,55.0,2.0\n"
            "1002.0,60.0,3.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "e41_comma.las", content))
        # Pre-fix: wrapped verdict → 4-value steps from 3-token rows →
        # DEPT=[1000,50,1] (silent shift).  Post-fix: non-wrapped →
        # short-row null-fill preserves every genuine value.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 55.0, 60.0])
        np.testing.assert_allclose(las.logs["RHOB"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(las.logs["NPHI"], [-999.25, -999.25, -999.25])


class TestE42WrappedHeaderGateClosesAfterFirstDataLine:
    """E-42 (CONFIRMED MEDIUM): _read_wrapped's header-skip gate was keyed
    to step completion (total_elements == 0), staying open across multiple
    lines in depth-first wrapped layouts — a string continuation value
    coinciding with a mnemonic on line 2+ was silently dropped as a
    "header" (data loss + column shift).  The gate is now keyed to the
    line position (current_line == 0), like _read_normal."""

    def test_string_continuation_on_line_two_not_dropped(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " LITH.    :  Lithology {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT LITH GR\n"
            "1000.0\n"
            "LITH GR\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "e42_gate.las", content))
        # The second data line "LITH GR" is a genuine wrapped continuation
        # (LITH string value + GR value) that coincides with the curve
        # mnemonics.  Pre-fix: the gate (total_elements == 0) is still open
        # → the row is skipped as a "header" → the step never completes →
        # DEPT=[] (all values lost).  Post-fix: gate closed after the first
        # data line → the continuation is consumed → one complete step.
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0])
        np.testing.assert_array_equal(
            las.string_data["LITH"], np.array(["LITH"], dtype=object)
        )


class TestM13UnitsRowAfterMnemonicHeaderSkipped:
    """M-13 (CONFIRMED MEDIUM): a units row emitted directly after the
    standalone mnemonic header row inside ~A ("~A\\nDEPT GR\\nM GAPI\\n...")
    was consumed as a DATA row → phantom all-null first row + one-row shift
    of the whole depth log.  The fix skips an optional units row (all
    letters-only tokens, shared is_units_header_row predicate) on the
    first data line only, gated on a mnemonic header row having been
    skipped."""

    def test_units_row_skipped(self, tmp_path: Path) -> None:
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
            "DEPT GR\n"
            "M GAPI\n"
            "1000.0 50.0\n"
            "1001.0 51.0\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "m13_units.las", content))
        # Pre-fix: "M GAPI" consumed as the first data row →
        # DEPT=[-999.25, 1000.0, 1001.0] (phantom all-null first row).
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 51.0])
        # M-02 pin: the parser pre-scan subtracts the units row itself via
        # the shared is_units_header_row predicate, so NO spurious
        # "Pre-scan overcount" warning may fire on an M-13 file (pre-fix
        # the pre-scan counted the units row → declared 3 data lines for 2
        # actual → spurious warning; the old simplefilter("ignore") mask
        # hid it from the test).
        pre_scan_msgs = [
            str(w.message) for w in caught if "Pre-scan overcount" in str(w.message)
        ]
        assert not pre_scan_msgs, (
            f"M-02: spurious pre-scan overcount warning on M-13 file: {pre_scan_msgs}"
        )

    def test_units_row_skipped_wrapped(self, tmp_path: Path) -> None:
        """M-03: the M-13 units-row skip must also apply in WRAP=YES
        (wrapped) mode.  Pre-fix the wrapped path skipped the mnemonic
        header row but consumed the units row as a data step → phantom
        all-null first step + one-step depth shift
        (DEPT=[-999.25, 1000.0, 1001.0, 1002.0])."""
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
            "1000.0\n"
            "50.0\n"
            "1001.0\n"
            "55.0\n"
            "1002.0\n"
            "60.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(
                _write_las(tmp_path, "m13_units_wrapped.las", content)
            )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 55.0, 60.0])

    def test_units_like_first_data_row_not_skipped_without_header(
        self, tmp_path: Path
    ) -> None:
        """Control: without a preceding mnemonic header row, a first row of
        letters-only tokens is DATA (an all-string section), never a units
        row — the skip requires the mnemonic header to have been seen first
        (position gate).  The all-string exclusion already stops the header
        skip; the units skip must not fire either."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " LITH.    :  Lithology {S}\n"
            " FORM.    :  Formation {S}\n"
            "~A\n"
            "ACME SAND\n"
            "SHALE GRN\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m13_control.las", content))
        # Both letters-only rows are genuine data rows: no mnemonic header
        # row precedes them, so nothing may be skipped as a units row.
        np.testing.assert_array_equal(
            las.string_data["LITH"], np.array(["ACME", "SHALE"], dtype=object)
        )
        np.testing.assert_array_equal(
            las.string_data["FORM"], np.array(["SAND", "GRN"], dtype=object)
        )


class TestM30CommaThousandsRecombined:
    """M-30 (CONFIRMED MEDIUM): LAS DLM=COMMA data lines with
    comma-grouped thousands ("1,234.5") split into fragments that silently
    mis-assign every subsequent column when the token count matches
    curve_count (zero warnings).  The LAS path now ports the DEV reader's
    recombination with the same loud per-pair warnings, including the
    equal-token-count case (E-24's hole)."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A DEPT RHOB NPHI\n"
            + rows
        )

    def test_equal_token_count_recombined_with_warning(self, tmp_path: Path) -> None:
        """Equal-token-count case (the E-24 hole): 3 tokens == 3 curves with
        a comma-grouped DEPT.  Pre-fix: DEPT=[1.0,5.0], RHOB=[234.5,60.0],
        NPHI=[50.0,0.2] with ZERO warnings.  Post-fix: recombined with a
        loud warning → DEPT=[1234.5,5.0], RHOB=[50.0,60.0], NPHI null-filled
        (short row)."""
        test_file = _write_las(
            tmp_path, "m30_equal.las", self._content("1,234.5,50\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert thousands_msgs, "M-30: no thousands-separator recombination warning"
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [-999.25, 0.2])

    def test_surplus_tokens_recombined(self, tmp_path: Path) -> None:
        """Surplus case: 4 tokens vs 3 curves — the run merges when the
        recombined count exactly satisfies the declared columns.  Pre-fix:
        DEPT=[1.0,5.0] with the 60 discarded as an extra column."""
        test_file = _write_las(
            tmp_path, "m30_surplus.las", self._content("1,234.5,50,60\n5,60,0.2\n")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [50.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [60.0, 0.2])

    def test_bare_fragment_row_not_false_recombined(self, tmp_path: Path) -> None:
        """M-23 control: a row of bare 3-digit fragments ("100,450") is a
        genuine multi-column row, never thousands — no merge, no warning
        (protects the equal-count widening from over-merging)."""
        test_file = _write_las(
            tmp_path, "m30_bare.las", self._content("100,450,20\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert not thousands_msgs, "M-30: bare fragment row must not recombine"
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [450.0, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [20.0, 0.2])

    def test_three_digit_leading_equal_count_not_false_recombined(
        self, tmp_path: Path
    ) -> None:
        """Widening control: at equal token count a THREE-digit leading
        group ("100,234.5,50" for 3 curves) stays ambiguous with genuine
        columns (DEPT=100, GR=234.5, RHOB=50 is plausible) — the merge is
        limited to 1-2 digit leading groups.  No merge, no warning."""
        test_file = _write_las(
            tmp_path, "m30_leading3.las", self._content("100,234.5,50\n5,60,0.2\n")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        thousands_msgs = [
            str(w.message) for w in caught if "thousands separator" in str(w.message)
        ]
        assert not thousands_msgs, "M-30: 3-digit leading group must not recombine"
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 5.0])
        np.testing.assert_allclose(las.logs["RHOB"], [234.5, 60.0])
        np.testing.assert_allclose(las.logs["NPHI"], [50.0, 0.2])

    def test_wrapped_comma_continuation_recombined(self, tmp_path: Path) -> None:
        """M-30 in wrapped mode: a comma-DLM WRAP=YES file whose
        continuation line carries a comma-grouped thousands value
        ("1,234.5,5.0" = GR 1234.5 + RHOB 5.0 for step 1).  Pre-fix the
        split fragments enter the step buffer as [1,234.5,5.0] → a full
        step of WRONG values (DEPT=1, GR=234.5, RHOB=5.0).  Post-fix the
        recombination restores the true step alignment."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT GR RHOB\n"
            "1000.0\n"
            "1,234.5,5.0\n"
            "1001.0\n"
            "55.0,2.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(
                _write_las(tmp_path, "m30_wrapped.las", content)
            )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(las.logs["GR"], [1234.5, 55.0])
        np.testing.assert_allclose(las.logs["RHOB"], [5.0, 2.0])


# ──────────────────────────────────────────────────────────────
# Stage 12 fix regression pins — LAS read paths (M-32 LAS port,
# M-33 LAS port, F-02 shape (b) LAS port, M-06, M-07).  Each FAILS
# on pre-fix code and PASSES on post-fix.  Adversarial evidence:
# tmp/s11-adv-m2-report.md, tmp/s11-adv-m5-report.md.
# ──────────────────────────────────────────────────────────────


class TestM32LasPortDecimalExponent:
    """M-32 LAS port (CONFIRMED MEDIUM): the LAS 1.2/2.0 port of
    _THOUSANDS_FRAG_RE had the same decimal-OR-exponent grammar drift —
    "1,234.5E3" was detected but its fragment could not merge → silent
    column shift.  The fragment grammar now accepts decimal+exponent."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT RHOB\n"
            + rows
        )

    def test_decimal_exponent_recombined(self, tmp_path: Path) -> None:
        test_file = _write_las(
            tmp_path, "m32_las_decexp.las", self._content("1,234.5E3,5.0\n2,345.6E3,6.0\n")
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234500.0, 2345600.0])
        np.testing.assert_allclose(las.logs["RHOB"], [5.0, 6.0])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 1, f"M-32 LAS no thousands warning: {[str(w.message) for w in rec]}"


class TestM33LasPortTwoSingleGroup:
    """M-33 LAS port (CONFIRMED MEDIUM): two single-comma-group
    thousands values in the LAS 1.2/2.0 DLM=COMMA path failed the
    per-run exact-fit gate individually — full-row recombination now
    drives the gate (same fix as the DEV twin)."""

    @staticmethod
    def _content(rows: str) -> str:
        return (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            "~A DEPT RHOB\n"
            + rows
        )

    def test_two_single_group_recombined(self, tmp_path: Path) -> None:
        test_file = _write_las(
            tmp_path, "m33_las_two_single.las", self._content("1,234.5,2,345.6\n3,456.7,4,567.8\n")
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(test_file)
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 3456.7])
        np.testing.assert_allclose(las.logs["RHOB"], [2345.6, 4567.8])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 2, f"M-33 LAS fewer than 2 warnings: {[str(w.message) for w in rec]}"


class TestF02LasPortShapeB:
    """F-02 shape (b) LAS port (pass-2, CONFIRMED): the hybrid gate
    restores the surplus-row mixed-run shape on the LAS port — the
    unambiguous run merges while the genuine 3-digit pair stays."""

    def test_surplus_mixed_run(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " RHOB.K/M3 :  Density\n"
            " NPHI.V/V :  Porosity\n"
            "~A DEPT RHOB NPHI\n"
            "1,234.5,100,250.5\n"
            "2.0,3.0,4.0\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "f02_las_surplus.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [1234.5, 2.0])
        np.testing.assert_allclose(las.logs["RHOB"], [100.0, 3.0])
        np.testing.assert_allclose(las.logs["NPHI"], [250.5, 4.0])
        thousands = [str(w.message) for w in rec if "thousands separator" in str(w.message)]
        assert len(thousands) >= 1, f"F-02 LAS no thousands warning: {[str(w.message) for w in rec]}"


class TestM06LettersRowPreservedOnReadPaths:
    """M-06 (CONFIRMED MEDIUM): the shared is_units_header_row predicate
    classified a genuine letters-only first DATA row directly after the
    mnemonic header (no units row) as units and dropped it on all 4 read
    paths.  The predicate now uses a units-form token pattern (every
    token letters-only AND <=4 chars AND >=1 token <=2 chars) so genuine
    data words ("ACME SAND") are preserved."""

    def test_las20_normal_letters_row_preserved(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_normal.las", content))
        # The letters row is preserved as a null-filled first data row.
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [-999.25, 150.0, 155.0])

    def test_units_row_still_skipped(self, tmp_path: Path) -> None:
        """Control: the standard M-13 units row ("M GAPI") is still
        skipped.

        KEPT + HARDENED (register DISPUTED-DELETE resolved conservatively):
        the end-to-end assertion stays, and the discriminating predicate is now
        asserted at its own level — the units form is what makes "M GAPI" a
        units row and "ACME SAND" data, so a predicate widened back to
        "letters only" fails here rather than only through a downstream count.
        """
        from pylasdev._data_section_reader import is_units_header_row

        assert is_units_header_row(["M", "GAPI"]) is True
        assert is_units_header_row(["ACME", "SAND"]) is False

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "M GAPI\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_units.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [2000.0, 2001.0])
        np.testing.assert_allclose(las.logs["GR"], [150.0, 155.0])
        assert len(las.logs["DEPT"]) == 2, "the units row must not become a data row"

    def test_las30_letters_row_preserved(self, tmp_path: Path) -> None:
        """LAS 3.0 path: same letters-only-first-row shape preserved."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_las30.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])

    def test_las30_deferred_letters_row_preserved(self, tmp_path: Path) -> None:
        """LAS 3.0 DEFERRED path (data section before ~VERSION): the
        same letters-only-first-row shape is preserved on replay, with no
        spurious pre-scan overcount warning."""
        content = (
            "~A\n"
            "DEPT GR\n"
            "ACME SAND\n"
            "2000.0 150.0\n"
            "2001.0 155.0\n"
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    SPACE :\n"
            "~WELL INFORMATION\n"
            " STRT.M   2000.0 : START DEPTH\n"
            " STOP.M   2001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "m06_las30_def.las", content))
        np.testing.assert_allclose(las.logs["DEPT"], [-999.25, 2000.0, 2001.0])
        assert not any("overcount" in str(w.message) for w in rec), (
            f"spurious pre-scan overcount: {[str(w.message) for w in rec]}"
        )


class TestM07WrappedEmbeddedDelimiterWarning:
    """M-07 (CONFIRMED MEDIUM): the wrapped LAS 1.2/2.0 path had NO
    I2-02 embedded-delimiter warning (present only in _read_normal) — an
    embedded comma in a {S} string value silently corrupted columns with
    only a misleading N-I-08 diagnostic.  _read_wrapped now mirrors the
    I2-02 detection (step-boundary-overshoot signal)."""

    def test_wrapped_embedded_comma_warns(self, tmp_path: Path) -> None:
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : MULTIPLE LINES PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   1000.0 : START DEPTH\n"
            " STOP.M   1001.0 : STOP DEPTH\n"
            " STEP.M   1.0    : STEP\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " NAME.    :  Well Name {S}\n"
            " GR.GAPI  :  Gamma\n"
            "~A\n"
            "1000.0\n"
            "WELL, INC,50.0\n"
            "1001.0\n"
            "WELL, INC2,51.0\n"
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            las = read_las_file_as_object(_write_las(tmp_path, "m07_wrapped.las", content))
        # The warning must name the delimiter-in-string mechanism (mirror
        # of the I2-02 contract asserted at test_regression.py:2612-2613).
        assert any("delimiter" in str(w.message) and "string" in str(w.message) for w in rec), (
            f"M-07 no embedded-delimiter warning: {[str(w.message) for w in rec]}"
        )
        # HARDENED (register): the warning alone left the CORRUPTION unpinned —
        # the whole point of the diagnostic is the column shift it reports.  The
        # observed outcome (verified against the current reader) is that the
        # split string truncates to its first token, the depth value that
        # follows lands in the string column, and the numeric column null-fills.
        assert np.asarray(las.string_data["NAME"]).tolist() == ["WELL", "1001.0"], (
            f"unexpected string-column outcome: {np.asarray(las.string_data['NAME']).tolist()}"
        )
        np.testing.assert_allclose(las.logs["DEPT"], [1000.0, 50.0])
        assert np.all(np.asarray(las.logs["GR"]) == -999.25), (
            "the shifted gamma column must be null-filled once the string splits"
        )


# ══════════════════════════════════════════════════════════════
# TEST-UPDATE regression pins for the fixes that had none
# (s4 extraction backlog rows for tests/test_reader.py).  Each test
# names its finding id and asserts the post-fix behaviour that the
# pre-fix code did not produce.
# ══════════════════════════════════════════════════════════════

_M30_WELL_HEADER = (
    "~VERSION INFORMATION\n"
    " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
    " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
)


class TestS4RegressionPinsReader:
    """Regression pins for landed reader fixes that had no test."""

    def test_f001_tilde_letter_data_row_reported_as_lost(self, tmp_path: Path) -> None:
        """F-001: a data row whose first token looks like a section header is
        announced as a lost row, and a genuine ~O section is not.

        Pre-fix, ``~C 11.0 51.0`` inside ``~A`` was consumed as a section
        header with NO diagnostic — the row's data values simply vanished.
        Post-fix the parser names the row and the discarded count.  The control
        keeps the fix from over-reporting: a real ``~O`` section ends the data
        block loudly but must NOT claim a data row was lost.
        """
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
            " 100.0 50.0\n"
            "~C 11.0 51.0\n"
            " 102.0 52.0\n"
        )
        lost_file = tmp_path / "f001_header_shaped_row.las"
        lost_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lost = read_las_file(lost_file)
        messages = [str(x.message) for x in caught]
        lost_rows = [m for m in messages if "discarded data row(s)" in m]
        assert len(lost_rows) == 1, f"F-001 diagnostic missing: {messages[-4:]}"
        assert "'~C 11.0 51.0'" in lost_rows[0], lost_rows[0]
        assert "1 discarded data row(s)" in lost_rows[0]
        # The row's own value is gone from the depth column (the reported harm).
        np.testing.assert_allclose(lost["logs"]["DEPT"], [100.0])

        other_content = content.replace("~C 11.0 51.0\n", "~O SOME OTHER TEXT\n")
        other_file = tmp_path / "f001_other_section.las"
        other_file.write_text(other_content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_las_file(other_file)
        other_messages = [str(x.message) for x in caught]
        assert not any("discarded data row(s)" in m for m in other_messages), (
            f"a genuine ~O section must not be reported as a lost data row: {other_messages}"
        )
        assert any("Data reading stops at this section boundary" in m for m in other_messages)

    def test_f007_semicolon_array_forms_stay_numeric(self, tmp_path: Path) -> None:
        """F-007: the ``{A;N}`` / ``{AFw.d;N}`` semicolon array forms are array
        curves — their values stay numeric, not string_data.

        Pre-fix only the colon form was accepted, so ``{A;5ms}`` silently routed
        the curve to string_data.  Post-fix both curves carry ``array_info`` and
        their values land in the numeric log, with the AF form's time offset
        parsed out of the ``;5ms`` suffix.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n STOP.M   101.0 : \n STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " NMR[2].PU    :  Echo {A;5ms}\n"
            " AF10.US      :  Amp {AF10.4;5ms}\n"
            "~ASCII DEPT NMR[2] AF10\n"
            " 100.0,50.0,60.0,1.5\n"
        )
        test_file = tmp_path / "f007_semicolon.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)

        assert las.curves_order == ["DEPT", "NMR[2]", "AF10[1]"]
        nmr = las.curves[1]
        af = las.curves[2]
        assert nmr.array_info is not None, "NMR[2] is not an array curve"
        assert nmr.array_info.index == 2
        assert af.array_info is not None, "AF10[1] is not an array curve"
        assert af.array_info.index == 1
        assert af.array_info.time_offset == 5.0, (
            f"the ;5ms suffix was not parsed: {af.array_info.time_offset}"
        )
        np.testing.assert_allclose(las.logs["NMR[2]"], [50.0])
        np.testing.assert_allclose(las.logs["AF10[1]"], [60.0])
        assert "NMR[2]" not in las.string_data and "AF10[1]" not in las.string_data, (
            f"semicolon array curves leaked into string_data: {list(las.string_data)}"
        )

    def test_f011_quoted_string_value_splits_into_its_own_column(
        self, tmp_path: Path
    ) -> None:
        """F-011 / M-1: a quoted string value containing the delimiter is ONE
        column, and the writer replaces the delimiter LOUDLY on write.

        Pre-fix the comma inside ``"WELL, A"`` was not quoted-aware: the row
        produced 4 tokens, the string truncated and every later column shifted.
        The parse half asserts the 3-column result; the write half pins the
        documented, warned substitution (the writer replaces the active
        delimiter with ';' rather than emitting an unreadable row).
        """
        from pylasdev import write_las_file
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n STOP.M   101.0 : \n STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " WELLN.       :  Well {S}\n"
            " GR.GAPI      :  Gamma {F}\n"
            "~ASCII DEPT WELLN GR\n"
            ' 100.0,"WELL, A",50.0\n'
            ' 101.0,"WELL, B",51.0\n'
        )
        test_file = tmp_path / "f011_quoted.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)

        assert np.asarray(las.string_data["WELLN"]).tolist() == ["WELL, A", "WELL, B"]
        np.testing.assert_allclose(las.logs["DEPT"], [100.0, 101.0])
        np.testing.assert_allclose(las.logs["GR"], [50.0, 51.0])

        out = tmp_path / "f011_quoted_rt.las"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_las_file(out, las)
        substitutions = [
            str(x.message)
            for x in caught
            if "delimiter will be replaced" in str(x.message)
        ]
        assert len(substitutions) == 1, (
            f"the delimiter substitution must be announced: {[str(x.message) for x in caught]}"
        )
        assert "COMMA" in substitutions[0]
        back = read_las_file_as_object(out)
        assert np.asarray(back.string_data["WELLN"]).tolist() == ["WELL; A", "WELL; B"], (
            "the warned substitution must be what actually lands on disk"
        )

    @pytest.mark.parametrize("n_curves", [3, 2])
    def test_f074_grouped_comma_thousands_recombined(
        self, tmp_path: Path, n_curves: int
    ) -> None:
        """F-074 / M-10: a comma-grouped thousands value in COMMA data is
        recombined, loudly, whether or not the token count happens to match.

        Pre-fix the grouped value split into two columns: with 3 curves the row
        produced 4 tokens (extra column discarded, every later column shifted),
        and the equal-token-count 2-curve shape was silently mis-assigned
        (E-24's hole).  Post-fix both shapes map to the right columns and the
        recombination is warned once per affected value.

        HEAD A/B (measured this run): this test PASSES on the pre-fix tree as
        well, so it is a BEHAVIOUR pin, not a fix witness — the LAS-path
        recombination predates HEAD.  Kept because the backlog row names it and
        it pins the loud per-value warning.
        """
        curves = " DEPT.M   :  Depth\n GR.GAPI  :  Gamma\n"
        if n_curves == 3:
            curves += " RHOB.K/M3:  Density\n"
            rows = "1000.0,1,234.5,2.5\n1001.0,2,345.5,3.5\n"
        else:
            rows = "1000.0,1,234.5\n1001.0,2,345.5\n"
        content = (
            _M30_WELL_HEADER.replace(
                " WRAP.   NO   : ONE LINE PER DEPTH STEP\n",
                " WRAP.   NO   : ONE LINE PER DEPTH STEP\n DLM.    COMMA :\n",
            )
            + "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            + curves
            + "~A DEPT GR" + (" RHOB" if n_curves == 3 else "") + "\n" + rows
        )
        test_file = tmp_path / f"f074_grouped_{n_curves}.las"
        test_file.write_text(content, encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = read_las_file(test_file)

        np.testing.assert_allclose(data["logs"]["DEPT"], [1000.0, 1001.0])
        np.testing.assert_allclose(data["logs"]["GR"], [1234.5, 2345.5])
        if n_curves == 3:
            np.testing.assert_allclose(data["logs"]["RHOB"], [2.5, 3.5])
        recombined = [
            str(x.message) for x in caught if "contains a thousands separator" in str(x.message)
        ]
        assert len(recombined) == 2, f"expected one warning per value, got {recombined}"
        assert "'1,234.5'" in recombined[0] and "'1234.5'" in recombined[0]

    def test_f129_d_format_is_string_data(self, tmp_path: Path) -> None:
        """F-129: a ``{D}`` date curve keeps its text in string_data.

        Pre-fix ``{D}`` was accepted but routed numeric, so every date became
        the null sentinel.  A Fortran D-exponent numeric column stays numeric —
        the fix must not reclassify numeric D notation as a date.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n STOP.M   101.0 : \n STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " DATE.        :  Date {D}\n"
            " VAL.         :  Value {F}\n"
            "~ASCII DEPT DATE VAL\n"
            " 100.0,2024-01-31,1.5\n"
            " 101.0,2024-02-01,2.5\n"
        )
        test_file = tmp_path / "f129_date.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        assert np.asarray(las.string_data["DATE"]).tolist() == ["2024-01-31", "2024-02-01"]
        assert "DATE" not in las.logs
        np.testing.assert_allclose(las.logs["VAL"], [1.5, 2.5])

        fortran = tmp_path / "f129_fortran_d.las"
        fortran.write_text(
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A DEPT DT\n"
            " 100.0 1.5D+03\n",
            encoding="utf-8",
        )
        numeric = read_las_file(fortran)
        assert numeric["logs"]["DT"].dtype == np.float64
        np.testing.assert_allclose(numeric["logs"]["DT"], [1500.0])

    def test_f075_duplicate_array_curve_raises_parse_error(self, tmp_path: Path) -> None:
        """F-075 / F-077: a duplicate array-curve definition fails through the
        documented exception boundary.

        The dedup rename produced ``NMR_2[1]`` and the contiguity check raised a
        bare ``ValueError``, which escapes the library's ``LASParseError``
        boundary that callers are documented to catch.  Post-fix the failure is
        a ``LASParseError`` naming the contiguity problem.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n STOP.M   101.0 : \n STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " NMR[1].PU    :  Echo A {A}\n"
            " NMR[1].PU    :  Echo B {A}\n"
            "~ASCII DEPT NMR[1] NMR[1]\n"
            " 100.0,1.0,2.0\n"
        )
        test_file = tmp_path / "f075_duplicate_array.las"
        test_file.write_text(content, encoding="utf-8")
        with pytest.raises(LASParseError, match="Non-contiguous array indices") as exc_info:
            read_las_file_as_object(test_file)
        assert not isinstance(exc_info.value, ValueError), (
            "the failure must stay inside the documented LASParseError boundary"
        )
        assert "NMR" in str(exc_info.value)

    def test_f225_nbsp_and_control_deletion_are_counted(self, tmp_path: Path) -> None:
        """F-225 / F-231 / M-9 / LD-08: NBSP thousands padding and true control
        characters are removed with a counted warning naming what was joined.

        Pre-fix the reader substituted a space (or injected the delimiter) with
        no diagnostic, shifting values between columns.  Post-fix the digits are
        joined, the value is preserved, and the warning counts the removals and
        names the fused token.

        The LD-08 control: an NBSP between two COMPLETE numeric values is a
        token separator, never a thousands pad — it must not be deleted.
        """
        body = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " GR.GAPI  :  Gamma\n"
            "~A DEPT GR\n"
        )
        nbsp_file = tmp_path / "f225_nbsp.las"
        nbsp_file.write_bytes(
            body.encode() + " 1000.0 1\xa0234.5\n 1001.0 2\xa0345.5\n".encode()
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nbsp = read_las_file(nbsp_file)
        nbsp_messages = [
            str(x.message) for x in caught if "non-breaking space" in str(x.message)
        ]
        assert len(nbsp_messages) == 1, f"NBSP removal not counted: {nbsp_messages}"
        assert "Removed 2 non-breaking space(s)" in nbsp_messages[0]
        assert "'1234.5'" in nbsp_messages[0] and "'2345.5'" in nbsp_messages[0]
        np.testing.assert_allclose(nbsp["logs"]["GR"], [1234.5, 2345.5])

        # LD-08 control: NBSP as a token separator splits two complete values.
        control_file = tmp_path / "f225_nbsp_separator.las"
        control_file.write_bytes(
            body.encode() + " 1000.0 50\xa075\n 1001.0 51\xa076\n".encode()
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            control = read_las_file(control_file)
        control_messages = [
            str(x.message) for x in caught if "non-breaking space" in str(x.message)
        ]
        assert not control_messages, (
            f"an NBSP between two complete values must not be deleted: {control_messages}"
        )
        # The separator splits the line into two tokens; the surplus one is
        # discarded as an extra column, never fused into 5075.0.
        np.testing.assert_allclose(control["logs"]["GR"], [50.0, 51.0])
        assert not np.any(np.asarray(control["logs"]["GR"]) == 5075.0), (
            "the NBSP separator must not fuse two complete values"
        )

        ctrl_file = tmp_path / "f231_controls.las"
        ctrl_file.write_bytes(
            body.encode("utf-8") + b" 1000.0 50\x000.0\n 1001.0 51\x000.0\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctrl = read_las_file(ctrl_file)
        ctrl_messages = [
            str(x.message) for x in caught if "control character(s)" in str(x.message)
        ]
        assert len(ctrl_messages) == 1, f"control deletion not counted: {ctrl_messages}"
        assert "Removed 2 control character(s)" in ctrl_messages[0]
        assert "U+0000" in ctrl_messages[0]
        np.testing.assert_allclose(ctrl["logs"]["GR"], [500.0, 510.0])

    def test_f014_numeric_first_string_second_views_agree(self, tmp_path: Path) -> None:
        """F-014: the numeric and string views stay in their own columns and
        survive ``to_dict`` → ``from_dict``.

        Pre-fix two independent gates could route the same column into both
        views (or neither), so the top-level ``logs`` and ``string_data`` views
        desynchronised from the parsed section data.

        HEAD A/B (measured this run): passes on the pre-fix tree too — a
        BEHAVIOUR pin for the two-view contract, not a fix witness for this
        fixture shape.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            " DLM.    COMMA :\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0 : \n STOP.M   101.0 : \n STEP.M   1.0 : \n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth {F}\n"
            " VAL.         :  Numeric {F}\n"
            " LAB.         :  Label {S}\n"
            "~ASCII DEPT VAL LAB\n"
            " 100.0,1.5,SAND\n"
            " 101.0,2.5,SHALE\n"
        )
        test_file = tmp_path / "f014_views.las"
        test_file.write_text(content, encoding="utf-8")
        las = read_las_file_as_object(test_file)
        assert set(las.logs) == {"DEPT", "VAL"}
        assert set(las.string_data) == {"LAB"}
        # The {S} column must not leak into the numeric view at all: an
        # ``assert_allclose(las.logs["LAB"] if "LAB" in las.logs else [], [])``
        # here compared [] to [] whenever LAB was absent (vacuous) and could
        # never run the present branch (the set assertion above excludes it).
        assert "LAB" not in las.logs, "the {S} column leaked into the numeric logs view"

        round_tripped = type(las).from_dict(las.to_dict())
        assert set(round_tripped.logs) == {"DEPT", "VAL"}
        assert set(round_tripped.string_data) == {"LAB"}
        np.testing.assert_allclose(round_tripped.logs["VAL"], [1.5, 2.5])
        assert np.asarray(round_tripped.string_data["LAB"]).tolist() == ["SAND", "SHALE"]
