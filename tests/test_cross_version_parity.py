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
    # curve metadata (mnemonic, unit, description) agrees
    for c1, c2 in zip(first["curves"], second["curves"], strict=True):
        assert c1["mnemonic"] == c2["mnemonic"], f"{label}: mnemonic drift"
        assert c1["unit"] == c2["unit"], f"{label}: unit drift for {c1['mnemonic']}"
        assert c1["description"] == c2["description"], (
            f"{label}: description drift for {c1['mnemonic']}"
        )


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
        specifiers on the curve definitions (F/E formats stay numeric)."""
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

    def test_wrapped_vs_unwrapped_parity(self, tmp_path: Path) -> None:
        """WRAP=YES declaration must not change the logical data content:
        the writer emits non-wrapped output, and both parse to the same data."""
        las_wrap = LASFile()
        las_wrap.version = VersionSection(vers="2.0", wrap="YES", dlm="SPACE")
        _build_numeric_well(las_wrap, _CURVES_20)

        las_nowrap = LASFile()
        las_nowrap.version = VersionSection(vers="2.0", wrap="NO", dlm="SPACE")
        _build_numeric_well(las_nowrap, _CURVES_20)

        p_wrap = _write_and_parse(las_wrap, tmp_path, "wrap_yes")
        p_nowrap = _write_and_parse(las_nowrap, tmp_path, "wrap_no")

        _assert_parity(p_wrap, p_nowrap, "WRAP-YES-vs-WRAP-NO")
