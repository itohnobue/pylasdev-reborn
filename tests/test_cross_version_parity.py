"""Cross-version parity harness — LAS 2.0, LAS 3.0, LAS 1.2 agreements.

The mechanical-enforcement harness (Stage B, ultimate-solution run).

The same logical well (curve mnemonics, data arrays, well fields) written
as LAS 2.0 and as LAS 3.0 must parse back to the SAME logical content —
modulo version-specific section structure.  Likewise LAS 1.2 vs LAS 2.0.

This enforces the "two paths must agree" contract between the version
writers/readers forever: if the LAS 3.0 path drifts from the LAS 2.0 path
(wrap handling, section naming, header layout, case normalization), a test
here fails without a human having to find the specific edge.

Strategy: build a LASFile model once, write it through each version's
writer, parse each output, and compare curves_order, data arrays, well
fields, and curve metadata across versions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import (
    LASFile,
    VersionSection,
    read_las_file,
    write_las_file,
)
from pylasdev.models import CurveDefinition

# ---------------------------------------------------------------------------
# Logical-well builders
# ---------------------------------------------------------------------------


def _build_well(las: LASFile) -> None:
    for key, value in [
        ("STRT", "1670.0"),
        ("STOP", "1660.0"),
        ("STEP", "-0.125"),
        ("NULL", "-999.25"),
        ("WELL", "PARITY WELL"),
        ("COMP", "TEST COMPANY"),
    ]:
        las.well[key] = value


def _build_numeric_well(las: LASFile, curves_order: list[str]) -> LASFile:
    """A small numeric-only logical well (DEPT index + 3 curves)."""
    _build_well(las)
    las.curves_order = list(curves_order)
    for mnemonic in curves_order:
        unit = "M" if mnemonic == "DEPT" else "US/M"
        las.curves.append(
            CurveDefinition(
                mnemonic=mnemonic,
                unit=unit,
                description=f"{mnemonic} DESC",
                data_format="F",
            )
        )
    las.logs["DEPT"] = np.array([1670.0, 1669.875, 1669.75, 1669.625])
    las.logs["DT"] = np.array([123.45, 123.50, 123.55, 123.60])
    las.logs["RHOB"] = np.array([2550.0, 2551.0, 2552.0, 2553.0])
    return las


def _write_and_parse(las: LASFile, tmp_path: Path, name: str) -> dict:
    out = tmp_path / f"{name}.las"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_las_file(out, las)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_las_file(out)


def _assert_parity(first: dict, second: dict, label: str) -> None:
    """Compare two version-parsed dicts of the same logical well."""
    assert first["curves_order"] == second["curves_order"], (
        f"{label}: curves_order drift {first['curves_order']} != {second['curves_order']}"
    )
    assert first["well"] == second["well"], f"{label}: well drift"
    for curve in first["curves_order"]:
        np.testing.assert_allclose(
            first["logs"][curve],
            second["logs"][curve],
            rtol=1e-6,
            err_msg=f"{label}: data drift for {curve!r}",
        )
    # curve metadata (mnemonic, unit, description, and data_format where both
    # sides declare one) agrees.  data_format is version-structural: LAS 2.0
    # parses to None (no format specifiers in LAS 2.0) while LAS 3.0 carries
    # {F}/{E}/{S} — a None-vs-format difference is NOT a parity regression,
    # so formats are compared only when present on both sides (F-109).
    for c1, c2 in zip(first["curves"], second["curves"], strict=True):
        assert c1["mnemonic"] == c2["mnemonic"], f"{label}: mnemonic drift"
        assert c1["unit"] == c2["unit"], f"{label}: unit drift for {c1['mnemonic']}"
        assert c1["description"] == c2["description"], (
            f"{label}: description drift for {c1['mnemonic']}"
        )
        f1 = c1.get("data_format")
        f2 = c2.get("data_format")
        if f1 is not None and f2 is not None:
            assert f1 == f2, f"{label}: data_format drift for {c1['mnemonic']}: {f1!r} != {f2!r}"


_CURVES_20 = ["DEPT", "DT", "RHOB"]
_CURVES_30 = ["DEPT", "DT", "RHOB"]


class TestLas20VsLas30Parity:
    """The same logical well through LAS 2.0 and LAS 3.0 writers/readers."""

    def test_numeric_well_parity(self, tmp_path: Path) -> None:
        las20 = LASFile()
        las20.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        _build_numeric_well(las20, _CURVES_20)

        las30 = LASFile()
        las30.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        _build_numeric_well(las30, _CURVES_30)

        p20 = _write_and_parse(las20, tmp_path, "parity20")
        p30 = _write_and_parse(las30, tmp_path, "parity30")

        _assert_parity(p20, p30, "LAS2.0-vs-LAS3.0")

    def test_las20_vs_las30_data_formats_agree(self, tmp_path: Path) -> None:
        """The LAS 3.0 writer emits the same numeric data even with format
        specifiers on the curve definitions (F/E formats stay numeric).

        HARDEN (test-register row, M3): ``_assert_parity`` compares
        ``data_format`` only when BOTH sides declare one, and a parsed LAS 2.0
        curve has ``data_format=None`` — so the cross-version comparison
        could not see a 3.0 writer that collapsed {E} to {F} (the M3
        mutation).  The 3.0 side's declared formats are therefore asserted
        directly here.
        """
        las20 = LASFile()
        las20.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        _build_well(las20)
        las20.curves_order = ["DEPT", "DT"]
        las20.curves.append(CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH"))
        las20.curves.append(CurveDefinition(mnemonic="DT", unit="US/M", description="SONIC"))
        las20.logs["DEPT"] = np.array([1670.0, 1669.875])
        las20.logs["DT"] = np.array([123.45, 123.50])

        las30 = LASFile()
        las30.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        _build_well(las30)
        las30.curves_order = ["DEPT", "DT"]
        las30.curves.append(
            CurveDefinition(mnemonic="DEPT", unit="M", description="DEPTH", data_format="F")
        )
        las30.curves.append(
            CurveDefinition(mnemonic="DT", unit="US/M", description="SONIC", data_format="E")
        )
        las30.logs["DEPT"] = np.array([1670.0, 1669.875])
        las30.logs["DT"] = np.array([123.45, 123.50])

        p20 = _write_and_parse(las20, tmp_path, "fmt20")
        p30 = _write_and_parse(las30, tmp_path, "fmt30")

        # The 3.0 side must round-trip the DECLARED format specifiers.
        formats = {curve["mnemonic"]: curve.get("data_format") for curve in p30["curves"]}
        assert formats == {"DEPT": "F", "DT": "E"}, formats

        _assert_parity(p20, p30, "formats-agree")


class TestLas12VsLas20Parity:
    """The same logical well through LAS 1.2 and LAS 2.0 writers/readers."""

    def test_numeric_well_parity(self, tmp_path: Path) -> None:
        las12 = LASFile()
        las12.version = VersionSection(vers="1.2", wrap="NO", dlm="SPACE")
        _build_numeric_well(las12, _CURVES_20)

        las20 = LASFile()
        las20.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        _build_numeric_well(las20, _CURVES_20)

        p12 = _write_and_parse(las12, tmp_path, "parity12")
        p20 = _write_and_parse(las20, tmp_path, "parity20b")

        _assert_parity(p12, p20, "LAS1.2-vs-LAS2.0")


# ---------------------------------------------------------------------------
# F-5 / G-E: wrap + case dimensions, and a falsifiability control for the
# parity helper itself (test-register HARDEN + ADD-GAP rows).
# ---------------------------------------------------------------------------


def _run_parity(las: LASFile, tmp_path: Path, name: str) -> dict:
    return _write_and_parse(las, tmp_path, name)


class TestParityDimensionCoverage:
    """G-E / HARDEN: the parity guard must not be a WRAP=NO-uppercase-only claim."""

    def test_case_variant_mnemonics_parity(self, tmp_path: Path) -> None:
        """Lowercase mnemonics on one side must parse to the same uppercase names.

        The 2.0 model declares ``dept``/``dt``/``rhob``; the reader uppercases,
        so parity with the uppercase 3.0 model holds — and a reader that
        stopped normalizing case would fail here.
        """
        las20 = LASFile()
        las20.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        _build_numeric_well(las20, ["DEPT", "DT", "RHOB"])
        for curve in las20.curves:
            curve.mnemonic = curve.mnemonic.lower()
        las20.curves_order = [name.lower() for name in las20.curves_order]
        las20.logs = {key.lower(): value for key, value in las20.logs.items()}

        las30 = LASFile()
        las30.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        _build_numeric_well(las30, ["DEPT", "DT", "RHOB"])

        p20 = _run_parity(las20, tmp_path, "case20")
        p30 = _run_parity(las30, tmp_path, "case30")
        assert p20["curves_order"] == ["DEPT", "DT", "RHOB"], p20["curves_order"]
        _assert_parity(p20, p30, "case-variant")

    def test_declared_wrap_yes_parity(self, tmp_path: Path) -> None:
        """A LAS 2.0 model declaring WRAP=YES still agrees with LAS 3.0.

        The writer emits the unwrapped form and overrides the header to
        WRAP=NO (documented writer behaviour, F-01), so both versions parse
        to the same logical well and both report WRAP=NO.  This pins the
        wrap-override liveness on the reader/writer boundary: a writer that
        emitted WRAP=YES while writing unwrapped data would drift here.
        """
        las20 = LASFile()
        las20.version = VersionSection(vers="2.0", wrap="YES", dlm="SPACE")
        _build_numeric_well(las20, ["DEPT", "DT", "RHOB"])

        las30 = LASFile()
        las30.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        _build_numeric_well(las30, ["DEPT", "DT", "RHOB"])

        p20 = _run_parity(las20, tmp_path, "wrap20")
        p30 = _run_parity(las30, tmp_path, "wrap30")
        assert p20["version"]["WRAP"] == "NO", p20["version"]
        assert p30["version"]["WRAP"] == "NO", p30["version"]
        _assert_parity(p20, p30, "declared-wrap-yes")


class TestParityHarnessFalsifiability:
    """G-C: _assert_parity must FAIL on each contract dimension it claims.

    Every cross-version test in this module relies on this helper.  A helper
    with an unreachable or weakened dimension makes the whole harness
    decorative, so each dimension gets a negative control: mutate one field of
    a synthetic pair and require an AssertionError.
    """

    @staticmethod
    def _baseline() -> dict:
        return {
            "curves_order": ["DEPT", "DT"],
            "well": {"STRT": "100.0", "STOP": "200.0", "NULL": "-999.25"},
            "logs": {"DEPT": np.array([100.0, 101.0]), "DT": np.array([50.0, 51.0])},
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "description": "DEPTH", "data_format": "F"},
                {"mnemonic": "DT", "unit": "US/M", "description": "SONIC", "data_format": "F"},
            ],
        }

    def test_identical_dicts_pass(self) -> None:
        _assert_parity(self._baseline(), self._baseline(), "control")

    def test_curves_order_drift_fails(self) -> None:
        second = self._baseline()
        second["curves_order"] = ["DT", "DEPT"]
        with pytest.raises(AssertionError, match="curves_order drift"):
            _assert_parity(self._baseline(), second, "order")

    def test_well_drift_fails(self) -> None:
        second = self._baseline()
        second["well"]["STOP"] = "250.0"
        with pytest.raises(AssertionError, match="well drift"):
            _assert_parity(self._baseline(), second, "well")

    def test_data_drift_fails(self) -> None:
        second = self._baseline()
        second["logs"]["DT"] = np.array([50.0, 99.0])
        with pytest.raises(AssertionError):
            _assert_parity(self._baseline(), second, "data")

    def test_unit_drift_fails(self) -> None:
        second = self._baseline()
        second["curves"][1]["unit"] = "OHMM"
        with pytest.raises(AssertionError, match="unit drift"):
            _assert_parity(self._baseline(), second, "unit")

    def test_description_drift_fails(self) -> None:
        second = self._baseline()
        second["curves"][1]["description"] = "OTHER"
        with pytest.raises(AssertionError, match="description drift"):
            _assert_parity(self._baseline(), second, "description")

    def test_data_format_drift_fails_when_both_declare(self) -> None:
        second = self._baseline()
        second["curves"][1]["data_format"] = "E"
        with pytest.raises(AssertionError, match="data_format drift"):
            _assert_parity(self._baseline(), second, "format")
