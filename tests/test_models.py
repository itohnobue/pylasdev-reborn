"""Tests for LAS data models."""

from __future__ import annotations

from typing import Any

import numpy as np

from pylasdev.models import (
    CurveDefinition,
    DevFile,
    LASFile,
    ParameterEntry,
    VersionSection,
    WellSection,
)


class TestVersionSection:
    """Tests for VersionSection dataclass."""

    def test_defaults(self) -> None:
        v = VersionSection()
        assert v.vers == "2.0"
        assert v.wrap == "NO"
        assert v.dlm == "SPACE"

    def test_to_dict(self) -> None:
        v = VersionSection(vers="1.2", wrap="YES", dlm="COMMA")
        d = v.to_dict()
        assert d == {"VERS": "1.2", "WRAP": "YES", "DLM": "COMMA"}

    def test_is_las30(self) -> None:
        assert VersionSection(vers="3.0").is_las30 is True
        assert VersionSection(vers="2.0").is_las30 is False
        assert VersionSection(vers="1.2").is_las30 is False

    def test_delimiter_char(self) -> None:
        assert VersionSection(dlm="SPACE").delimiter_char == " "
        assert VersionSection(dlm="TAB").delimiter_char == "\t"
        assert VersionSection(dlm="COMMA").delimiter_char == ","


class TestWellSection:
    """Tests for WellSection dataclass."""

    def test_getitem_setitem(self) -> None:
        w = WellSection()
        w["STRT"] = "100.0"
        assert w["STRT"] == "100.0"

    def test_contains(self) -> None:
        w = WellSection(entries={"STRT": "100"})
        assert "STRT" in w
        assert "MISSING" not in w

    def test_get_with_default(self) -> None:
        w = WellSection(entries={"STRT": "100"})
        assert w.get("STRT") == "100"
        assert w.get("MISSING", "default") == "default"

    def test_to_dict(self) -> None:
        w = WellSection(entries={"A": "1", "B": "2"})
        d = w.to_dict()
        assert d == {"A": "1", "B": "2"}
        # Verify it's a copy
        d["C"] = "3"
        assert "C" not in w.entries


class TestCurveDefinition:
    """Tests for CurveDefinition dataclass."""

    def test_basic(self) -> None:
        c = CurveDefinition(mnemonic="DEPT", unit="M", description="Depth")
        assert c.mnemonic == "DEPT"
        assert c.unit == "M"

    def test_to_dict(self) -> None:
        c = CurveDefinition(mnemonic="DT", unit="US/M", description="Sonic")
        d = c.to_dict()
        assert d["mnemonic"] == "DT"
        assert d["unit"] == "US/M"

    def test_is_array_element(self) -> None:
        from pylasdev.models import ArrayElementInfo

        c1 = CurveDefinition(mnemonic="DEPT")
        assert c1.is_array_element is False

        c2 = CurveDefinition(
            mnemonic="NMR[1]",
            array_info=ArrayElementInfo(base_name="NMR", index=1),
        )
        assert c2.is_array_element is True
        assert c2.base_mnemonic == "NMR"


class TestParameterEntry:
    """Tests for ParameterEntry dataclass."""

    def test_basic(self) -> None:
        p = ParameterEntry(mnemonic="BHT", unit="DEGC", value="35.5", description="Temp")
        assert p.mnemonic == "BHT"
        assert p.value == "35.5"

    def test_to_dict(self) -> None:
        p = ParameterEntry(mnemonic="BS", value="200")
        d = p.to_dict()
        assert d["mnemonic"] == "BS"
        assert d["value"] == "200"


class TestLASFile:
    """Tests for LASFile dataclass."""

    def test_to_dict_structure(self) -> None:
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["STRT"] = "100"
        las.curves_order = ["DEPT"]
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.parameters.append(ParameterEntry(mnemonic="BHT", value="35"))

        d = las.to_dict()
        assert d["version"]["VERS"] == "2.0"
        assert d["well"]["STRT"] == "100"
        assert d["curves_order"] == ["DEPT"]
        assert np.array_equal(d["logs"]["DEPT"], np.array([100.0, 101.0]))
        assert d["parameters"]["BHT"] == "35"

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200"},
            "curves_order": ["DEPT", "DT"],
            "parameters": {"BHT": "35"},
            "logs": {
                "DEPT": np.array([100.0, 101.0]),
                "DT": np.array([50.0, 51.0]),
            },
        }
        las = LASFile.from_dict(data)
        assert las.version.vers == "2.0"
        assert las.well["STRT"] == "100"
        assert las.curves_order == ["DEPT", "DT"]
        assert len(las.curves) == 2
        assert len(las.parameters) == 1
        assert np.array_equal(las.logs["DEPT"], np.array([100.0, 101.0]))

    def test_roundtrip_dict(self) -> None:
        """Test that from_dict(to_dict()) preserves data."""
        las = LASFile()
        las.version = VersionSection(vers="2.0", wrap="NO")
        las.well["STRT"] = "100"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.logs["DEPT"] = np.array([1.0, 2.0])
        las.parameters.append(ParameterEntry(mnemonic="BHT", value="35"))

        d = las.to_dict()
        las2 = LASFile.from_dict(d)
        d2 = las2.to_dict()

        assert d["version"] == d2["version"]
        assert d["well"] == d2["well"]
        assert d["curves_order"] == d2["curves_order"]
        assert d["parameters"] == d2["parameters"]
        np.testing.assert_array_equal(d["logs"]["DEPT"], d2["logs"]["DEPT"])

    def test_get_curve_by_mnemonic(self) -> None:
        las = LASFile()
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.curves.append(CurveDefinition(mnemonic="DT"))
        assert las.get_curve_by_mnemonic("DT") is not None
        assert las.get_curve_by_mnemonic("MISSING") is None


class TestDevFile:
    """Tests for DevFile dataclass."""

    def test_to_dict(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0])
        dev.columns["TVD"] = np.array([0.0, 99.0])
        d = dev.to_dict()
        assert "MD" in d
        np.testing.assert_array_equal(d["MD"], np.array([0.0, 100.0]))
        # Verify it's a copy
        d["MD"][0] = 999.0
        assert dev.columns["MD"][0] == 0.0
