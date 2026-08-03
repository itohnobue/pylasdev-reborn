"""lasio differential harness — pylasdev vs lasio 0.32 as an oracle.

The mechanical-enforcement harness (Stage B, ultimate-solution run).

Parses the same test_data files with pylasdev AND lasio, and asserts the
data-model contract documented by the s1-research-lasio report:

- ``lasio.data`` is (n_points, n_curves), rows = depth samples; pylasdev
  ``logs`` is per-curve 1-D.  Compare ``las.data`` DIRECTLY against
  ``np.column_stack([p['logs'][k] for k in p['curves_order']])`` — NO
  transpose.
- lasio replaces the ~W NULL sentinel with NaN at read time for non-index
  float curves (las.py:478-484); pylasdev keeps the raw sentinel.  The null
  mapping formula is
  ``(isnan_l & (pdata == nullv)) | (~isnan_l & np.isclose(ldata, pdata, atol=1e-9))``
  — never ``np.isclose(equal_nan=True)`` alone.
- lasio has NO ``null_value`` attribute in 0.32; the sentinel lives at
  ``las.well['NULL'].value``.
- Both sides uppercase keys by default, so ``curves_order`` and
  ``las.keys()`` compare directly.
- Numeric well fields: ``float(pylasdev) == float(lasio)``; non-numeric:
  exact string compare.

Documented divergences (asserted as EXPECTED, not failures):
- ragged wrapped: lasio raises ValueError / pylasdev accumulates by curve
  count — a short step is MISALIGNED (the next depth line supplies the
  missing value) and a trailing partial step is discarded with the N-I-08
  warning; pylasdev does NOT null-fill ragged wrapped input;
- missing ~V: lasio defaults to 2.0 / pylasdev raises LASParseError;
- extra data columns: lasio makes 'UNKNOWN' / pylasdev discards;
- duplicate mnemonics: lasio ``DT:1`` / pylasdev ``DT_2`` suffix schemes;
- encoding labels differ but content matches (compare decoded content).

The whole module is gated by ``pytest.importorskip("lasio")`` so it SKIPs
cleanly when lasio is absent.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

lasio = pytest.importorskip("lasio", reason="lasio is not installed")

from pylasdev import read_las_file  # noqa: E402
from pylasdev.exceptions import LASParseError  # noqa: E402

# ---------------------------------------------------------------------------


@contextmanager
def _suppress_warnings():
    """Context manager suppressing parser/writer convention warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Test files: which participate in which comparison tier
# ---------------------------------------------------------------------------

# Files BOTH sides read fully and agree on the data model (0 mismatch cells
# verified during design).
_FULL_COMPARE_FILES = [
    "sample.las",
    "sample_2.0.las",
    "sample_wrapped.las",
    "petrel2.0.las",
    "sample_big.las",
    "5_1.las",
    "comment_test.las",
    "sample_2.0_based.las",
    "sample_2.0_minimal.las",
    "sample_2.0_wrapped.las",
    "sample_curve_api.las",
    "sample_minimal.las",
    "sample_las3.0_tab.las",
]

# Encoding divergence: lasio's chardet misdetects cp866 as cp1006 on
# 1475IBK3.las, producing mojibake curve names / well text.  DATA still
# matches.  Handled by a dedicated documented-divergence test.
_ENCODING_DIVERGENT_FILE = "1475IBK3.las"

# 4ALS.las contains BOTH a duplicate mnemonic (IK appears twice -> pylasdev
# IK/IK_2, lasio IK:1/IK:2) AND a 0xFF byte in the last curve mnemonic
# (SP\xff: lasio decodes 0xFF -> U+02D9, pylasdev -> U+0178).  These are
# documented divergences; the DATA still matches.
_DIVERGENT_KEYS_FILE = "4ALS.las"

# Empty-data: sample_3.0.las has no ~A rows; both sides parse curves but
# produce zero data rows.  Assert both agree on emptiness.
_EMPTY_DATA_FILE = "sample_3.0.las"

# lasio raises LASHeaderError on the LAS 3.0 spec file; pylasdev reads it.
_LASIO_UNREADABLE_FILE = "sample_las3.0_spec.las"


def _load_pair(test_data_dir: Path, name: str) -> tuple[dict, object]:
    """Parse one file with pylasdev and lasio; return (pdict, lasfile)."""
    path = test_data_dir / name
    with _suppress_warnings():
        p = read_las_file(path)
    las = lasio.read(str(path))
    return p, las


def _compare_data(p: dict, las: object) -> tuple[int, int]:
    """Compare lasio.data vs pylasdev column_stack; return (mismatch, total)."""
    pdata = np.column_stack([p["logs"][k] for k in p["curves_order"]])
    ldata = las.data
    assert pdata.shape == ldata.shape, f"shape drift: pylasdev {pdata.shape} vs lasio {ldata.shape}"
    nullv = float(p["well"]["NULL"])
    isnan_l = np.isnan(ldata)
    match = (isnan_l & (pdata == nullv)) | (~isnan_l & np.isclose(ldata, pdata, atol=1e-9))
    return int((~match).sum()), int(pdata.size)


def _compare_well(p: dict, las: object) -> int:
    """Compare well fields; return count of mismatches.

    The key SETS must be identical before any value comparison: comparing
    only the intersection would silently hide a drift that drops or adds a
    well field on either side (e.g. pylasdev stops reading STOP, or lasio
    starts emitting a new key) — exactly the drift this harness exists to
    catch (X-4).
    """
    assert set(p["well"].keys()) == set(las.well.keys()), (
        f"well key-set drift: pylasdev {sorted(p['well'])} vs lasio {sorted(las.well.keys())}"
    )
    mismatches = 0
    for key in sorted(p["well"].keys()):
        pv = p["well"].get(key, "")
        # L-12: lasio stores an empty well field as None; str(None) == "None"
        # would false-fail the exact-string compare below.  No corpus file
        # currently has one, but guard the latent trap.
        lv_raw = las.well[key].value
        if lv_raw is None:
            continue
        lv = str(lv_raw)
        if _is_numeric(lv):
            try:
                if not np.isclose(float(pv), float(lv), rtol=1e-9, atol=1e-9):
                    mismatches += 1
            except (TypeError, ValueError):
                mismatches += 1
        else:
            if pv.strip() != lv.strip():
                mismatches += 1
    return mismatches


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


class TestFullDataContract:
    """Files both sides read fully: strict data + key + well comparison."""

    @pytest.mark.parametrize("fname", _FULL_COMPARE_FILES)
    def test_data_agrees(self, test_data_dir: Path, fname: str) -> None:
        p, las = _load_pair(test_data_dir, fname)
        # Both sides have the same number of data rows (else shapes differ).
        if not p["logs"]:
            pytest.fail(f"pylasdev read no logs for {fname} (expected data)")
        mismatches, total = _compare_data(p, las)
        assert mismatches == 0, f"{fname}: {mismatches}/{total} data cells differ from lasio"

    @pytest.mark.parametrize("fname", _FULL_COMPARE_FILES)
    def test_curve_keys_agree(self, test_data_dir: Path, fname: str) -> None:
        p, las = _load_pair(test_data_dir, fname)
        # Both sides uppercase by default — compare directly, no normalization.
        assert p["curves_order"] == list(las.keys()), (
            f"{fname}: curve mnemonic drift — pylasdev {p['curves_order']} "
            f"vs lasio {list(las.keys())}"
        )

    @pytest.mark.parametrize("fname", _FULL_COMPARE_FILES)
    def test_well_fields_agree(self, test_data_dir: Path, fname: str) -> None:
        p, las = _load_pair(test_data_dir, fname)
        nbad = _compare_well(p, las)
        assert nbad == 0, f"{fname}: {nbad} well field(s) differ from lasio"

    @pytest.mark.parametrize("fname", _FULL_COMPARE_FILES)
    def test_index_curve_agrees(self, test_data_dir: Path, fname: str) -> None:
        p, las = _load_pair(test_data_dir, fname)
        index_name = p["curves_order"][0]
        # lasio .index == curves[0].data; pylasdev first log.  Index curve is
        # never NaN-replaced by lasio, so direct isclose comparison suffices.
        p_index = p["logs"][index_name]
        np.testing.assert_allclose(p_index, las.index, rtol=1e-6, atol=1e-9, err_msg=fname)


class TestNullConvention:
    """lasio null semantics: NULL sentinel location + NaN mapping."""

    def test_null_value_lives_in_well_section(self, test_data_dir: Path) -> None:
        """lasio 0.32 has NO null_value attribute; sentinel is well['NULL']."""
        p, las = _load_pair(test_data_dir, "sample.las")
        assert not hasattr(las, "null_value"), (
            "lasio gained a null_value attribute — re-verify harness contract"
        )
        assert float(las.well["NULL"].value) == float(p["well"]["NULL"])

    def test_data_is_not_masked_array(self, test_data_dir: Path) -> None:
        """lasio 0.32 returns a plain float64 ndarray, not np.ma.MaskedArray."""
        _, las = _load_pair(test_data_dir, "sample.las")
        assert not isinstance(las.data, np.ma.MaskedArray), (
            "lasio returned a masked array — re-verify harness contract"
        )

    def test_wrapped_null_mapping_formula(self, test_data_dir: Path) -> None:
        """sample_wrapped.las has NaN cells in lasio; NULL sentinel in pylasdev.

        The formula (isnan_l & (pdata == nullv)) | (~isnan_l & isclose) must
        yield 0 mismatches; np.isclose(equal_nan=True) alone would fail.
        """
        p, las = _load_pair(test_data_dir, "sample_wrapped.las")
        assert np.isnan(las.data).any(), "test premise: lasio should have NaN cells"
        mismatches, total = _compare_data(p, las)
        assert mismatches == 0, f"{mismatches}/{total} cells differ from lasio"


class TestDocumentedDivergences:
    """Expected behavioral differences — asserted, not failed."""

    def test_missing_version_section(self, tmp_path: Path) -> None:
        """lasio defaults VERS to 2.0; pylasdev raises LASParseError."""
        content = (
            "~WELL INFORMATION\n"
            " STRT.M   100.0  : START DEPTH\n"
            " STOP.M   102.0  : STOP DEPTH\n"
            " STEP.M   1.0  : STEP\n"
            " NULL.    -999.25  : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : DEPTH\n"
            "~A DEPTH\n"
            "100.0\n"
        )
        f = tmp_path / "no_version.las"
        f.write_text(content, encoding="utf-8")
        # pylasdev raises
        with pytest.raises(LASParseError):
            read_las_file(f)
        # lasio defaults to 2.0 and reads fine
        las = lasio.read(str(f))
        assert float(las.version["VERS"].value) == 2.0

    def test_extra_columns_diverge(self, tmp_path: Path) -> None:
        """Extra data columns: lasio names 'UNKNOWN'; pylasdev discards."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO  : One line per depth step\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0  : START DEPTH\n"
            " STOP.M   102.0  : STOP DEPTH\n"
            " STEP.M   1.0  : STEP\n"
            " NULL.    -999.25  : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : DEPTH\n"
            " GR.API  : GAMMA RAY\n"
            "~A DEPTH GR\n"
            "100.0 10.0 999.0\n"
            "101.0 20.0 999.0\n"
        )
        f = tmp_path / "extra_col.las"
        f.write_text(content, encoding="utf-8")
        with _suppress_warnings():
            p = read_las_file(f)
        las = lasio.read(str(f))
        # pylasdev: 2 curves, extra column discarded
        assert p["curves_order"] == ["DEPT", "GR"]
        # lasio: keeps extra column under 'UNKNOWN'
        assert "UNKNOWN" in las.keys()
        assert len(las.keys()) == 3

    def test_ragged_wrapped_diverge(self, tmp_path: Path) -> None:
        """Ragged wrapped rows: lasio raises ValueError; pylasdev misaligns.

        pylasdev does NOT null-fill a ragged wrapped step.  The
        value-accumulation reader consumes curve_count values per step, so
        the missing RHOB value at depth 101.0 is silently supplied by the
        NEXT depth line (102.0), and the trailing under-filled pair
        (30.0 2552.0) is discarded with the N-I-08 warning ("data section
        ended with ... value(s) not accounted for").  This warn+discard
        behavior is the accepted accumulation contract (II-4/R-6); the exact
        arrays below pin it so any change to ragged wrapped handling fails
        this test deliberately.
        """
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   YES  : Multiple lines per depth step\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0  : START DEPTH\n"
            " STOP.M   102.0  : STOP DEPTH\n"
            " STEP.M   1.0  : STEP\n"
            " NULL.    -999.25  : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : DEPTH\n"
            " GR.API  : GAMMA RAY\n"
            " RHOB.K/M3  : BULK DENSITY\n"
            "~A DEPTH GR RHOB\n"
            "100.0\n"
            "10.0 2550.0\n"
            "101.0\n"
            "20.0\n"  # ragged: only 1 of 2 values for this depth step
            "102.0\n"
            "30.0 2552.0\n"
        )
        f = tmp_path / "ragged_wrap.las"
        f.write_text(content, encoding="utf-8")
        # pylasdev reads successfully but MISALIGNED: the missing RHOB value
        # is filled from the next depth line, and the trailing under-filled
        # pair is discarded — with the N-I-08 warning, never a NULL fill.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p = read_las_file(f)
        assert any("not accounted for" in str(m.message) for m in caught), (
            "expected N-I-08 trailing-partial warning for ragged wrapped input"
        )
        assert p["curves_order"] == ["DEPT", "GR", "RHOB"]
        np.testing.assert_array_equal(p["logs"]["DEPT"], [100.0, 101.0])
        np.testing.assert_array_equal(p["logs"]["GR"], [10.0, 20.0])
        np.testing.assert_array_equal(p["logs"]["RHOB"], [2550.0, 102.0])
        # lasio raises ValueError ("Cannot reshape ~A data size ...")
        with pytest.raises(ValueError):
            lasio.read(str(f))

    def test_duplicate_mnemonic_suffixes_differ(self, tmp_path: Path) -> None:
        """Duplicate mnemonics: lasio colon suffix; pylasdev underscore suffix."""
        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO  : One line per depth step\n"
            "~WELL INFORMATION\n"
            " STRT.M   100.0  : START DEPTH\n"
            " STOP.M   102.0  : STOP DEPTH\n"
            " STEP.M   1.0  : STEP\n"
            " NULL.    -999.25  : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M  : DEPTH\n"
            " GR.API  : GAMMA RAY\n"
            " GR.API  : SECOND GAMMA\n"
            "~A DEPTH GR GR\n"
            "100.0 10.0 11.0\n"
            "101.0 20.0 21.0\n"
        )
        f = tmp_path / "dup_mnem.las"
        f.write_text(content, encoding="utf-8")
        with _suppress_warnings():
            p = read_las_file(f)
        las = lasio.read(str(f))
        # Both rename duplicates; schemes differ by design.
        assert len(p["curves_order"]) == 3
        assert len(las.keys()) == 3
        assert set(p["curves_order"]) != set(las.keys()), (
            f"expected divergent suffix schemes, got pylasdev {p['curves_order']} "
            f"lasio {list(las.keys())}"
        )


class TestSpecialFiles:
    """Files requiring dedicated handling."""

    def test_encoding_divergent_file_data_matches(self, test_data_dir: Path) -> None:
        """1475IBK3.las: lasio misdetects cp866 as cp1006 (mojibake text),
        but the DATA still matches under the null-mapping formula."""
        p, las = _load_pair(test_data_dir, _ENCODING_DIVERGENT_FILE)
        # curve count and data shapes agree
        assert len(p["curves_order"]) == len(las.keys())
        mismatches, total = _compare_data(p, las)
        assert mismatches == 0, f"{mismatches}/{total} cells differ from lasio"

    def test_encoding_divergent_file_curve_names_differ(self, test_data_dir: Path) -> None:
        """Documented: lasio's mojibake curve names differ from pylasdev's
        correctly-decoded Cyrillic names on 1475IBK3.las."""
        p, las = _load_pair(test_data_dir, _ENCODING_DIVERGENT_FILE)
        assert p["curves_order"] != list(las.keys()), (
            "expected divergence on this encoding-misdetected file"
        )

    def test_duplicate_mnemonic_file_data_matches(self, test_data_dir: Path) -> None:
        """4ALS.las: duplicate IK mnemonic + 0xFF byte in SP curve make the
        KEY SETS diverge (documented), but the DATA still matches under the
        null-mapping formula."""
        p, las = _load_pair(test_data_dir, _DIVERGENT_KEYS_FILE)
        assert len(p["curves_order"]) == len(las.keys())
        mismatches, total = _compare_data(p, las)
        assert mismatches == 0, f"{mismatches}/{total} cells differ from lasio"

    def test_duplicate_mnemonic_file_keys_differ(self, test_data_dir: Path) -> None:
        """Documented: 4ALS.las keys diverge (IK_2 vs IK:1/IK:2 suffix
        scheme; SP\xff decoded to different codepoints)."""
        p, las = _load_pair(test_data_dir, _DIVERGENT_KEYS_FILE)
        assert p["curves_order"] != list(las.keys()), (
            "expected divergence on duplicate-mnemonic + 0xFF-byte file"
        )

    def test_empty_data_file(self, test_data_dir: Path) -> None:
        """sample_3.0.las has no ~A rows: both sides produce zero data rows."""
        p, las = _load_pair(test_data_dir, _EMPTY_DATA_FILE)
        assert not p["logs"], "pylasdev should have no logs for the empty-data file"
        assert las.data.shape[0] == 0, f"lasio should have 0 data rows, got {las.data.shape}"

    def test_lasio_unreadable_spec_file(self, test_data_dir: Path) -> None:
        """sample_las3.0_spec.las: pylasdev reads it (it is the CWLS spec's own
        example); lasio's partial LAS 3.0 support raises LASHeaderError."""
        f = test_data_dir / _LASIO_UNREADABLE_FILE
        assert f.exists()
        with _suppress_warnings():
            p = read_las_file(f)
        assert "DEPT" in p["curves_order"]
        with pytest.raises(lasio.exceptions.LASHeaderError):
            lasio.read(str(f))
