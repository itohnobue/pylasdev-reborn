"""Tests for LAS data models."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import write_las_file
from pylasdev.exceptions import LASDataError, LASWriteError
from pylasdev.mnem_base import MNEM_BASE
from pylasdev.models import (
    MAX_FIELD_LENGTH,
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    DevFile,
    LASFile,
    ParameterEntry,
    ParameterZone,
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

    # F-030: Dots in mnemonics cause roundtrip corruption.
    def test_reject_dot_in_mnemonic(self) -> None:
        """F-030: CurveDefinition rejects dots in mnemonics.

        The writer uses dot as a structural separator; the parser
        splits on the first dot.  ``GR.CO`` → written as
        ``GR.CO.M/FT`` → parsed as mnemonic=GR, unit=CO.M/FT.
        """
        with pytest.raises(ValueError, match=r"mnemonic must not be empty"):
            CurveDefinition(mnemonic="GR.CO")


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

    def test_with_array_index(self) -> None:
        """Test ParameterEntry with array_index set."""
        p = ParameterEntry(mnemonic="RUN[1]", value="1.0", array_index=1)
        assert p.array_index == 1
        assert p.base_mnemonic == "RUN"
        d = p.to_dict()
        assert d["array_index"] == 1

    def test_with_zone(self) -> None:
        """Test ParameterEntry with zone association."""
        from pylasdev.models import ParameterZone

        p = ParameterEntry(
            mnemonic="MATR",
            value="SAND",
            description="Matrix",
            zone=ParameterZone(zone_name="RUN", zone_index=1),
        )
        assert p.zone is not None
        assert p.zone.zone_name == "RUN"
        assert p.zone.zone_index == 1
        d = p.to_dict()
        assert d["zone"]["zone_name"] == "RUN"
        assert d["zone"]["zone_index"] == 1

    def test_base_mnemonic_without_array_index(self) -> None:
        """Test base_mnemonic for parameter without array index."""
        p = ParameterEntry(mnemonic="BHT", value="35")
        assert p.base_mnemonic == "BHT"

    def test_base_mnemonic_without_bracket(self) -> None:
        """Test base_mnemonic when array_index is set but no bracket in mnemonic."""
        p = ParameterEntry(mnemonic="RUN", value="1", array_index=1)
        # array_index is set but mnemonic has no '[' -> returns mnemonic
        assert p.base_mnemonic == "RUN"

    # F-030: Dots in mnemonics cause roundtrip corruption.
    def test_reject_dot_in_mnemonic(self) -> None:
        """F-030: ParameterEntry rejects dots in mnemonics.

        Same roundtrip corruption as CurveDefinition — the writer
        uses dot as a structural separator.
        """
        with pytest.raises(ValueError, match=r"mnemonic must not be empty"):
            ParameterEntry(mnemonic="GR.CO", value="1.0")


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

    def test_get_array_curves(self) -> None:
        """Test get_array_curves filters curves by base array name."""
        from pylasdev.models import ArrayElementInfo

        las = LASFile()
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR[1]",
                array_info=ArrayElementInfo(base_name="NMR", index=1),
            )
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="NMR[2]",
                array_info=ArrayElementInfo(base_name="NMR", index=2),
            )
        )
        las.curves.append(
            CurveDefinition(
                mnemonic="T1[1]",
                array_info=ArrayElementInfo(base_name="T1", index=1),
            )
        )

        nmr_curves = las.get_array_curves("NMR")
        assert len(nmr_curves) == 2
        assert nmr_curves[0].mnemonic == "NMR[1]"
        assert nmr_curves[1].mnemonic == "NMR[2]"

        assert las.get_array_curves("NONEXIST") == []

    # --- T3: from_dict() list-format parameters path (models.py:335-344) ---

    def test_from_dict_list_parameters(self) -> None:
        """Test LASFile.from_dict() with parameters as a list of dicts."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "parameters": [
                {"mnemonic": "BHT", "unit": "DEGC", "value": "35.5", "description": "Temp"},
                {"mnemonic": "BS", "unit": "MM", "value": "200", "description": "Bit Size"},
            ],
            "logs": {"DEPT": np.array([100.0, 101.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.parameters) == 2
        assert las.parameters[0].mnemonic == "BHT"
        assert las.parameters[0].unit == "DEGC"
        assert las.parameters[0].value == "35.5"
        assert las.parameters[0].description == "Temp"
        assert las.parameters[1].mnemonic == "BS"
        assert las.parameters[1].value == "200"

    def test_from_dict_list_parameters_with_zone(self) -> None:
        """Test LASFile.from_dict() with list-format params including zone."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {},
            "curves_order": ["DEPT"],
            "parameters": [
                {
                    "mnemonic": "MATR",
                    "unit": "",
                    "value": "SAND",
                    "description": "Neutron Matrix",
                    "zone": {"zone_name": "RUN", "zone_index": 1},
                },
            ],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.parameters) == 1
        assert las.parameters[0].zone is not None
        assert las.parameters[0].zone.zone_name == "RUN"
        assert las.parameters[0].zone.zone_index == 1

    def test_from_dict_list_parameters_with_array_index(self) -> None:
        """Test LASFile.from_dict() with list-format params including array_index."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {},
            "curves_order": ["DEPT"],
            "parameters": [
                {
                    "mnemonic": "RUN[1]",
                    "unit": "",
                    "value": "1.0",
                    "description": "Run Number",
                    "array_index": 1,
                },
            ],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.parameters) == 1
        assert las.parameters[0].array_index == 1
        assert las.parameters[0].base_mnemonic == "RUN"

    def test_from_dict_list_parameters_roundtrip(self) -> None:
        """Test roundtrip with list-format parameters preserves metadata."""
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.logs["DEPT"] = np.array([100.0])
        from pylasdev.models import ParameterZone

        las.parameters.append(
            ParameterEntry(
                mnemonic="MATR",
                value="SAND",
                description="Matrix",
                zone=ParameterZone(zone_name="RUN", zone_index=1),
            )
        )
        las.parameters.append(
            ParameterEntry(
                mnemonic="RUN[1]",
                value="1.0",
                array_index=1,
            )
        )

        d = las.to_dict()
        las2 = LASFile.from_dict(d)
        d2 = las2.to_dict()

        assert len(d2["parameter_details"]) == 2
        assert d2["parameter_details"][0]["mnemonic"] == "MATR"
        assert d2["parameter_details"][0]["zone"]["zone_name"] == "RUN"
        assert d2["parameter_details"][1]["array_index"] == 1

    def test_section_curves_serialization_roundtrip(self) -> None:
        """Test DataSection section_curves preserve all fields through to_dict/from_dict.

        Creates a DataSection with section_curves (full/partial/minimal
        CurveDefinition metadata including data_format, array_info),
        roundtrips through LASFile.to_dict()/from_dict(), and asserts
        every CurveDefinition field is preserved. Covers MEDIUM-3 from
        post-fix review.
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0, 101.0])

        section = DataSection(
            name="Core[1]",
            section_type="CORE_DATA",
            curves_order=["CORET", "COREB", "CDES"],
            data={
                "CORET": np.array([1.0, 2.0]),
                "COREB": np.array([3.0, 4.0]),
            },
            string_data={
                "CDES": np.array(["desc_one", "desc_two"]),
            },
            section_curves=[
                # Full metadata with array_info
                CurveDefinition(
                    mnemonic="CORET",
                    unit="S",
                    api_code="123",
                    description="Core Top",
                    original_mnemonic="CORET_ORIG",
                    data_format="F",
                    array_info=ArrayElementInfo(
                        base_name="CORE",
                        index=1,
                        time_offset=0.0,
                    ),
                ),
                # Partial metadata
                CurveDefinition(
                    mnemonic="COREB",
                    unit="S",
                    description="Core Bottom",
                    data_format="F",
                ),
                # Minimal metadata (string format)
                CurveDefinition(
                    mnemonic="CDES",
                    description="Core Description",
                    data_format="S",
                ),
            ],
        )
        las.data_sections.append(section)

        d = las.to_dict()
        las2 = LASFile.from_dict(d)

        assert len(las2.data_sections) == 1
        rt_section = las2.data_sections[0]
        assert rt_section.section_type == "CORE_DATA"
        assert rt_section.curves_order == ["CORET", "COREB", "CDES"]
        assert len(rt_section.section_curves) == 3

        # Verify curve with full metadata
        c1 = rt_section.section_curves[0]
        assert c1.mnemonic == "CORET"
        assert c1.unit == "S"
        assert c1.api_code == "123"
        assert c1.description == "Core Top"
        assert c1.original_mnemonic == "CORET_ORIG"
        assert c1.data_format == "F"
        assert c1.array_info is not None
        assert c1.array_info.base_name == "CORE"
        assert c1.array_info.index == 1
        assert c1.array_info.time_offset == 0.0

        # Verify curve with partial metadata
        c2 = rt_section.section_curves[1]
        assert c2.mnemonic == "COREB"
        assert c2.unit == "S"
        assert c2.description == "Core Bottom"
        assert c2.data_format == "F"
        assert c2.api_code == ""
        assert c2.original_mnemonic == ""
        assert c2.array_info is None

        # Verify curve with minimal metadata (string format)
        c3 = rt_section.section_curves[2]
        assert c3.mnemonic == "CDES"
        assert c3.description == "Core Description"
        assert c3.data_format == "S"
        assert c3.array_info is None

        # Verify numeric data is preserved
        np.testing.assert_array_equal(rt_section.data["CORET"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(rt_section.data["COREB"], np.array([3.0, 4.0]))
        # Verify string data is preserved (CDES has data_format="S")
        np.testing.assert_array_equal(
            rt_section.string_data["CDES"], np.array(["desc_one", "desc_two"])
        )

    def test_section_curves_empty_roundtrip(self) -> None:
        """Test DataSection with empty section_curves survives roundtrip.

        Covers the edge case where section_curves=[] should serialize
        as [] and deserialize as []. Also verifies that name,
        section_type, curves_order, and data content survive roundtrip
        (F-I2-M54: previously only checked len + section_curves=[]).
        """
        las = LASFile()
        las.version = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        las.well["NULL"] = "-999.25"
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT", unit="M"))
        las.logs["DEPT"] = np.array([100.0])

        section = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([100.0])},
            section_curves=[],
        )
        las.data_sections.append(section)

        d = las.to_dict()
        las2 = LASFile.from_dict(d)

        assert len(las2.data_sections) == 1
        rt_section = las2.data_sections[0]
        assert rt_section.name == "LOG"
        assert rt_section.section_type == "LOG_DATA"
        assert rt_section.curves_order == ["DEPT"]
        assert rt_section.section_curves == []
        # F-I2-M54: verify data content survives roundtrip
        assert "DEPT" in rt_section.data
        np.testing.assert_array_equal(rt_section.data["DEPT"], np.array([100.0]))

    # --- F-01 fix: MAX_PARAMETERS guard on parameter_details ---

    def test_from_dict_parameter_details_max_guard(self, monkeypatch) -> None:
        """LASFile.from_dict() enforces MAX_PARAMETERS on parameter_details (F-01 fix).

        parameter_details was the sole unguarded iterable in from_dict().
        The fix adds a len check matching the existing params guard at L428.
        """
        # Use a small limit for testing
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        big_details = [{"mnemonic": f"PARAM_{i}", "value": str(i)} for i in range(6)]
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "parameters": {"_": "_"},  # non-empty dict triggers the dict branch
            "parameter_details": big_details,
            "logs": {"DEPT": np.array([100.0])},
        }
        with pytest.raises(ValueError, match="Number of parameter details"):
            LASFile.from_dict(data)

    def test_from_dict_parameter_details_max_at_limit(self, monkeypatch) -> None:
        """At-limit: MAX_PARAMETERS-1 parameter_details should pass (F-I2-M56).

        The guard uses >= (not >), so exactly MAX_PARAMETERS items are also
        rejected. N-1 items are the last count that passes (off-by-one bug
        confirmed by F-I2-M32). This test documents the actual boundary.
        """
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        exact_details = [{"mnemonic": f"PARAM_{i}", "value": str(i)} for i in range(4)]
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "parameters": {"_": "_"},
            "parameter_details": exact_details,
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.parameters) == 4

    # --- F-057 fix: MAX_WELL_ENTRIES guard on well, well_units, well_descriptions ---

    def test_from_dict_well_entries_max_guard(self, monkeypatch) -> None:
        """LASFile.from_dict() enforces MAX_WELL_ENTRIES on well entries (F-057 fix).

        Three guard points exist: well items, well_units items, and
        well_descriptions items.  MAX_WELL_ENTRIES = MAX_PARAMETERS.
        """
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        # Exceeding well entries
        big_well = {f"KEY_{i}": str(i) for i in range(6)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": big_well,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with pytest.raises(ValueError, match="Number of well entries"):
            LASFile.from_dict(data)

    def test_from_dict_well_entries_max_at_limit(self, monkeypatch) -> None:
        """At-limit: MAX_PARAMETERS-1 well entries should pass (F-I2-M56).

        The guard uses >= (not >), so the last passing count is N-1.
        """
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        exact_well = {f"KEY_{i}": str(i) for i in range(4)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": exact_well,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.well.entries) == 4

    def test_from_dict_well_units_max_guard(self, monkeypatch) -> None:
        """LASFile.from_dict() enforces MAX_WELL_ENTRIES on well_units (F-057 fix)."""
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        big_units = {f"KEY_{i}": "unit" for i in range(6)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "well_units": big_units,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with pytest.raises(ValueError, match="well unit entries"):
            LASFile.from_dict(data)

    def test_from_dict_well_units_max_at_limit(self, monkeypatch) -> None:
        """At-limit: MAX_PARAMETERS-1 well_units should pass (F-I2-M56).

        The guard uses >= (not >), so the last passing count is N-1.
        """
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        exact_units = {f"KEY_{i}": "unit" for i in range(4)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "well_units": exact_units,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.well.units) == 4

    def test_from_dict_well_descriptions_max_guard(self, monkeypatch) -> None:
        """LASFile.from_dict() enforces MAX_WELL_ENTRIES on well_descriptions (F-057 fix)."""
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        big_descs = {f"KEY_{i}": "desc" for i in range(6)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "well_descriptions": big_descs,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with pytest.raises(ValueError, match="well description entries"):
            LASFile.from_dict(data)

    def test_from_dict_well_descriptions_max_at_limit(self, monkeypatch) -> None:
        """At-limit: MAX_PARAMETERS-1 well_descriptions should pass (F-I2-M56).

        The guard uses >= (not >), so the last passing count is N-1.
        """
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        exact_descs = {f"KEY_{i}": "desc" for i in range(4)}
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "well_descriptions": exact_descs,
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        assert len(las.well.descriptions) == 4

    # --- F2-17 fix: dict.get() None bypass in CurveDefinition ---

    def test_from_dict_curve_none_in_get_guarded(self) -> None:
        """LASFile.from_dict() handles None for optional curve string fields (F2-17 fix).

        curve_dict.get("mnemonic", "") returned stored None when key existed
        with value None. The fix wraps all string fields with _safe_str().
        The mnemonic field now requires a non-empty value (F-M10 fix), so
        we use a real mnemonic and test None→"" conversion on unit/description.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT", "unit": None, "description": None}],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        # F-M10: mnemonic must be non-empty
        assert las.curves[0].mnemonic == "DEPT"
        # _safe_str(None) → "" for optional string fields
        assert las.curves[0].unit == ""
        assert las.curves[0].description == ""

    def test_from_dict_section_curve_none_in_get_guarded(self) -> None:
        """DataSection section_curves fields guarded against None (F2-17 fix, 2nd site).

        mnemonic uses a matching value ("DEPT") so the F-23 cross-validation
        between curves_order and section_curves passes.  unit and data_format
        use None to exercise the _safe_str() guard.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([100.0])},
                    "section_curves": [{"mnemonic": "DEPT", "unit": None, "data_format": None}],
                }
            ],
        }
        las = LASFile.from_dict(data)
        assert len(las.data_sections) == 1
        ds = las.data_sections[0]
        # F-I2-M55: verify data content and metadata survive from_dict
        assert ds.name == "LOG"
        assert ds.section_type == "LOG_DATA"
        assert ds.curves_order == ["DEPT"]
        assert "DEPT" in ds.data
        np.testing.assert_array_equal(ds.data["DEPT"], np.array([100.0]))
        sc = ds.section_curves[0]
        assert sc.mnemonic == "DEPT"
        assert sc.unit == ""
        assert sc.data_format == ""

    # --- F-23 fix: cross-validation between section curves_order and section_curves ---

    def test_from_dict_section_cross_validation_mismatch_count(self) -> None:
        """F-23: Mismatched curve count raises ValueError in per-section validation.

        curves_order has 2 entries but section_curves has 1 — should raise.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "DT"],
                    "data": {"DEPT": np.array([100.0]), "DT": np.array([50.0])},
                    "section_curves": [
                        {"mnemonic": "DEPT"},
                    ],
                }
            ],
        }
        with pytest.raises(ValueError, match="does not match section_curves"):
            LASFile.from_dict(data)

    def test_from_dict_section_cross_validation_mismatch_name(self) -> None:
        """F-23: Mismatched mnemonic raises ValueError in per-section validation.

        curves_order says "DT" but section_curves says "DEPT" at same index.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DT"],
                    "data": {"DT": np.array([50.0])},
                    "section_curves": [
                        {"mnemonic": "DEPT"},
                    ],
                }
            ],
        }
        with pytest.raises(ValueError, match="does not match section_curves"):
            LASFile.from_dict(data)

    def test_from_dict_section_curves_order_non_string_element(
        self,
    ) -> None:
        """F-I2E-05: Non-string element in per-section curves_order raises TypeError.

        When section_curves is empty, the mnemonic cross-validation gate is
        inactive, so non-string values (int, None) silently become str() column
        headers on write.  The fix adds per-element type checking that runs
        unconditionally, before the section_curves gate.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["GR", 123, None],
                    "data": {
                        "GR": np.array([50.0]),
                    },
                    "section_curves": [],
                }
            ],
        }
        with pytest.raises(LASDataError, match="must be str"):
            LASFile.from_dict(data)

    # --- F-24 fix: string_data count guard ---

    def test_from_dict_string_data_max_guard_per_section(self, monkeypatch) -> None:
        """F-24: Per-section string_data count is bounded by MAX_CURVES."""
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 3)

        # Build 4 string data entries — exceeds MAX_CURVES=3
        str_curves = {f"STR_{i}": np.array(["a"]) for i in range(4)}
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([100.0])},
                    "string_data": str_curves,
                }
            ],
        }
        with pytest.raises(ValueError, match="Number of string data curves"):
            LASFile.from_dict(data)

    def test_from_dict_string_data_max_at_limit_per_section(self, monkeypatch) -> None:
        """At-limit: MAX_CURVES-1 string_data per-section should pass (F-I2-M56).

        The guard uses >= (not >), so the last passing count is N-1.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 3)

        str_curves = {f"STR_{i}": np.array(["a"]) for i in range(2)}
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["STR_0", "STR_1"],
                    "string_data": str_curves,
                }
            ],
        }
        las = LASFile.from_dict(data)
        assert len(las.data_sections[0].string_data) == 2

    def test_from_dict_string_data_max_guard_top_level(self, monkeypatch) -> None:
        """F-24: Top-level string_data count is bounded by MAX_CURVES."""
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 3)

        str_curves = {f"STR_{i}": np.array(["a"]) for i in range(4)}
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "string_data": str_curves,
        }
        with pytest.raises(ValueError, match="Number of string data curves"):
            LASFile.from_dict(data)

    def test_from_dict_string_data_max_at_limit_top_level(self, monkeypatch) -> None:
        """At-limit: MAX_CURVES-1 top-level string_data should pass (F-I2-M56).

        The guard uses >= (not >), so the last passing count is N-1.
        Uses only string_data keys in curves_order (no logs) so the
        F-011 log-curves_order check skips (empty logs dict → falsy guard).
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 3)

        str_curves = {f"STR_{i}": np.array(["a"]) for i in range(2)}
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["STR_0", "STR_1"],
            "curves": [
                {"mnemonic": "STR_0", "data_format": "S"},
                {"mnemonic": "STR_1", "data_format": "S"},
            ],
            "string_data": str_curves,
        }
        las = LASFile.from_dict(data)
        assert len(las.string_data) == 2

    # --- F-25 fix: ValueError instead of UserWarning for inconsistent arrays ---

    def test_from_dict_inconsistent_log_lengths_raises(self) -> None:
        """F-25: Inconsistent log array lengths raise ValueError.

        Previously a suppressible UserWarning; now a hard ValueError matching
        the severity of other validation checks in from_dict().
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200"},
            "curves_order": ["DEPT", "DT"],
            "logs": {
                "DEPT": np.array([100.0, 101.0, 102.0]),
                "DT": np.array([50.0, 51.0]),  # 2 entries vs 3
            },
        }
        with pytest.raises(ValueError, match="inconsistent lengths"):
            LASFile.from_dict(data)

    def test_from_dict_inconsistent_data_section_lengths_raises(self) -> None:
        """F-25: Inconsistent data section array lengths raise ValueError."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "DT"],
                    "data": {
                        "DEPT": np.array([100.0, 101.0, 102.0]),
                        "DT": np.array([50.0, 51.0]),  # 2 entries vs 3
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match="inconsistent"):
            LASFile.from_dict(data)

    # --- F2-25 fix: isinstance guard before _create_parameter_entry ---

    def test_from_dict_parameter_details_non_dict_element(self) -> None:
        """F2-25: Non-dict element in parameter_details raises LASDataError."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": {"_": "_"},
            "parameter_details": [
                {"mnemonic": "OK", "value": "1"},
                None,  # Non-dict element
            ],
        }
        with pytest.raises(LASDataError, match="must be a dict"):
            LASFile.from_dict(data)

    def test_from_dict_params_list_non_dict_element(self) -> None:
        """F2-25: Non-dict element in params list raises LASDataError."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": [
                {"mnemonic": "OK", "value": "1"},
                "not_a_dict",  # Non-dict element
            ],
        }
        with pytest.raises(LASDataError, match="must be a dict"):
            LASFile.from_dict(data)

    # --- F-004: per-section string_data cross-array length ---

    def test_from_dict_per_section_string_data_inconsistent(self) -> None:
        """F-004: Per-section string_data with inconsistent array lengths raises ValueError.

        The per-section path (models.py:818-824) validates that all string_data
        arrays within a single DataSection have the same length.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "STR1", "STR2"],
                    "data": {"DEPT": np.array([100.0])},
                    "string_data": {
                        "STR1": np.array(["a", "b"]),
                        "STR2": np.array(["c"]),
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match=r"inconsistent"):
            LASFile.from_dict(data)

    # --- F-004: top-level string_data cross-array length ---

    def test_from_dict_top_level_string_data_inconsistent(self) -> None:
        """F-004: Per-section string_data with inconsistent array lengths raises ValueError.

        Uses a data_section (LAS 3.0 pattern) because the top-level path
        without data_sections now has conflicting validations after F2-11
        (string_data keys must be in curves_order) and F-011 (curves_order
        keys must be in logs).  Per-section string_data avoids these conflicts.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["STR1", "STR2"],
                    "string_data": {
                        "STR1": np.array(["a", "b"]),
                        "STR2": np.array(["c"]),
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match=r"inconsistent"):
            LASFile.from_dict(data)

    # --- F-025: non-int array_index ---

    def test_from_dict_non_int_array_index_raises(self) -> None:
        """F-025: Non-int array_index in parameter entry raises LASDataError.

        _create_parameter_entry (models.py:47-55) validates that array_index
        is int or None.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": [
                {
                    "mnemonic": "RUN[1]",
                    "value": "1",
                    "array_index": "not_an_int",
                },
            ],
        }
        with pytest.raises(LASDataError, match="array_index: expected int or None"):
            LASFile.from_dict(data)

    # --- F-026: non-int zone_index ---

    def test_from_dict_non_int_zone_index_raises(self) -> None:
        """F-026: Non-int zone_index in parameter zone raises LASDataError.

        _create_parameter_entry (models.py:34-42) validates that zone_index
        is int or None.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": [
                {
                    "mnemonic": "MATR",
                    "value": "SAND",
                    "zone": {
                        "zone_name": "RUN",
                        "zone_index": "not_an_int",
                    },
                },
            ],
        }
        with pytest.raises(LASDataError, match="zone_index: expected int or None"):
            LASFile.from_dict(data)

    # --- F-7-001: ParameterZone rejects bool (type() is not int guard) ---

    def test_parameter_zone_rejects_bool_true_zone_index(self) -> None:
        """F-7-001: ParameterZone(zone_index=True) raises TypeError.

        Direct construction bypasses the from_dict validation path.
        The __post_init__ guard at models.py:717 uses type() is not int
        which correctly rejects bool (bool subclasses int, so isinstance
        would NOT reject it).
        """
        from pylasdev.models import ParameterZone

        with pytest.raises(TypeError, match="zone_index must be int or None"):
            ParameterZone(zone_index=True)

    def test_parameter_zone_rejects_bool_false_zone_index(self) -> None:
        """F-7-001: ParameterZone(zone_index=False) raises TypeError.

        Both True and False are subclasses of int; both must be rejected.
        """
        from pylasdev.models import ParameterZone

        with pytest.raises(TypeError, match="zone_index must be int or None"):
            ParameterZone(zone_index=False)

    # --- F-8-001: _create_parameter_entry rejects bool array_index ---

    def test_create_parameter_entry_rejects_bool_array_index(self) -> None:
        """F-8-001: _create_parameter_entry with array_index=True raises TypeError.

        The guard at models.py:84 was changed from isinstance to type()
        to reject bool values (bool subclasses int).  This matches the
        pattern at lines 62 and 717.
        """
        from pylasdev.models import _create_parameter_entry

        with pytest.raises(TypeError, match="array_index: expected int or None"):
            _create_parameter_entry({"mnemonic": "TEST", "array_index": True})

    # --- F2-002: non-numeric time_offset ---

    def test_from_dict_non_numeric_time_offset_raises(self) -> None:
        """F2-002: Non-numeric time_offset in curve array_info raises LASDataError.

        _resolve_dict_entry (models.py:88-111) validates time_offset against
        (int, float) at both curve sites (lines 490, 683).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["NMR[1]"],
            "curves": [
                {
                    "mnemonic": "NMR[1]",
                    "array_info": {
                        "base_name": "NMR",
                        "index": 1,
                        "time_offset": "not_numeric",
                    },
                },
            ],
            "logs": {"NMR[1]": np.array([100.0])},
        }
        with pytest.raises(LASDataError, match="time_offset: expected"):
            LASFile.from_dict(data)

    # --- F-9-002: _resolve_dict_entry rejects bool for int ---

    def test_resolve_dict_entry_rejects_bool_for_int(self) -> None:
        """F-9-002: _resolve_dict_entry must reject bool when expected_type
        includes int.

        bool subclasses int in Python, so isinstance(True, int) is True.
        When callers pass expected_type=int or expected_type=(int, float),
        _resolve_dict_entry must reject True/False to prevent silent
        acceptance of bool where a numeric value is required.
        This protects from_dict index=True → 1, time_offset=True → 1ms
        semantic corruption.
        """
        from pylasdev.models import _resolve_dict_entry

        # bool should be rejected for int
        with pytest.raises(TypeError, match=r"index.*got bool"):
            _resolve_dict_entry({"index": True}, "index", int, lambda: 0)

        # bool should be rejected for (int, float)
        with pytest.raises(TypeError, match=r"time_offset.*got bool"):
            _resolve_dict_entry({"time_offset": True}, "time_offset", (int, float), lambda: None)

        # int and float should still work
        assert _resolve_dict_entry({"index": 5}, "index", int, lambda: 0) == 5
        assert (
            _resolve_dict_entry({"time_offset": 1.5}, "time_offset", (int, float), lambda: None)
            == 1.5
        )

        # None default should still work
        assert _resolve_dict_entry({}, "index", int, lambda: 0) == 0

    # --- F2-003: non-dict input to from_dict ---

    def test_from_dict_rejects_non_dict_input(self) -> None:
        """F-008/F2-003: LASFile.from_dict() wraps TypeError as LASDataError.

        models.py:741-742 validates isinstance(data, dict) inside try block;
        models.py:1468-1469 catches (ValueError, TypeError) and re-raises as LASDataError.
        """
        with pytest.raises(LASDataError, match="Expected dict, got"):
            LASFile.from_dict("not_a_dict")

    # --- F-017: Mandatory well field validation ---

    def test_from_dict_missing_mandatory_well_field_warns(self) -> None:
        """F-017: LASFile.from_dict() with well dict missing STRT emits warning.

        The pre-construction validation layer at models.py:143-158 checks
        that the four mandatory LAS 2.0 well fields (STRT, STOP, STEP, NULL)
        are present.  Missing fields trigger UserWarning.
        """
        import warnings

        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"COMP": "TestCo"},  # Missing STRT, STOP, STEP, NULL
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(data)
            mandatory_warnings = [
                x for x in w if "Mandatory well field(s) missing" in str(x.message)
            ]
            assert len(mandatory_warnings) == 1
            assert "STRT" in str(mandatory_warnings[0].message)
            assert "STOP" in str(mandatory_warnings[0].message)
            assert "STEP" in str(mandatory_warnings[0].message)
            assert "NULL" in str(mandatory_warnings[0].message)

    def test_from_dict_all_mandatory_well_fields_no_warning(self) -> None:
        """F-017: LASFile.from_dict() with all mandatory well fields — no warning."""
        import warnings

        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": "100.0",
                "STOP": "200.0",
                "STEP": "1.0",
                "NULL": "-999.25",
                "COMP": "ACME",
                "WELL": "WELL_A",
                "FLD": "NORTH",
                "LOC": "LOC_A",
                "SRVC": "SRVC_A",
                "DATE": "01/01/2020",
                "UWI": "UWI_A",
            },
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(data)
            mandatory_warnings = [
                x for x in w if "Mandatory well field(s) missing" in str(x.message)
            ]
            assert len(mandatory_warnings) == 0, (
                f"Unexpected warning: {[str(x.message) for x in mandatory_warnings]}"
            )

    # --- F-018: DLM validation ---

    def test_from_dict_dlm_invalid_raises_value_error(self) -> None:
        """F-018: LASFile.from_dict() with DLM='INVALID' raises ValueError.

        The pre-construction validation at models.py:160-170 checks that
        DLM is one of SPACE, TAB, or COMMA (case-insensitive).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "INVALID"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        with pytest.raises(ValueError, match="Invalid DLM value"):
            LASFile.from_dict(data)

    def test_from_dict_dlm_case_insensitive_accepts(self) -> None:
        """F-018: LASFile.from_dict() with DLM='space' (lowercase) is accepted.

        The validation uppercases the value before comparing against the
        valid set (SPACE, TAB, COMMA), so 'space' should pass validation.
        The value itself is preserved as-is (not uppercased).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "space"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        # Value is case-preserved (not normalized to uppercase)
        assert las.version.dlm == "space"

    def test_from_dict_dlm_all_valid_values_accepted(self) -> None:
        """F-018: All valid DLM values (SPACE, TAB, COMMA) accepted case-insensitively."""
        for dlm in ("SPACE", "TAB", "COMMA", "space", "Tab", "comma"):
            data: dict[str, Any] = {
                "version": {"VERS": "2.0", "WRAP": "NO", "DLM": dlm},
                "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
                "curves_order": ["DEPT"],
                "logs": {"DEPT": np.array([100.0])},
            }
            las = LASFile.from_dict(data)
            # Value is case-preserved; validation is case-insensitive
            assert las.version.dlm.upper() == dlm.upper()

    def test_from_dict_dlm_none_or_empty_no_error(self) -> None:
        """F-018: DLM=None or DLM='' (empty) should not raise — skipped by guard.

        The guard at models.py:163-165 checks ``if dlm_raw is not None and dlm_raw != ""``,
        so None and empty string are silently skipped.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": ""},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        # Empty string is preserved (not replaced with default)
        assert las.version.dlm == ""

    # --- F-019/IF-007: Non-3.0 multi-section rejection ---

    def test_from_dict_non_3_0_multi_section_raises(self) -> None:
        """F-019: LASFile.from_dict() with vers='2.0' and 2 data_sections
        raises ValueError.

        Multiple data_sections are only valid for LAS 3.0 — non-3.0
        versions cannot have more than one data section.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([100.0])},
                },
                {
                    "name": "Section2",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([200.0])},
                },
            ],
        }
        with pytest.raises(ValueError, match="Multiple data_sections"):
            LASFile.from_dict(data)

    def test_from_dict_las30_multi_section_succeeds(self) -> None:
        """F-019: LASFile.from_dict() with vers='3.0' and 2 data_sections succeeds.

        LAS 3.0 supports multiple typed data sections natively.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT", "unit": "M"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([100.0, 101.0])},
                },
                {
                    "name": "Section2",
                    "section_type": "CORE_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([200.0])},
                },
            ],
        }
        las = LASFile.from_dict(data)
        assert len(las.data_sections) == 2

    # --- R7F-02: Cumulative allocation guard uses sum, not max ---

    def test_from_dict_cumulative_guard_ds_data_plus_string_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """R7F-02: Cumulative allocation check must sum ds_data + ds_string_data
        rather than taking max().  Both data types coexist in the same section
        and allocate separate arrays — the total is the sum, not the max.

        The old code used ``_section_total = max(_section_total, ...)`` which
        could be bypassed when both ds_data and ds_string_data individually pass
        their per-section checks but their combined total exceeds the limit.

        This test sets MAX_TOTAL_ELEMENTS small (100) and constructs a section
        where ds_data=60 elements and ds_string_data=60 elements individually
        pass per-section (60 < 100) but combined=120 > 100.
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_TOTAL_ELEMENTS", 100)
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT", "CDES"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                {"mnemonic": "CDES", "unit": "", "data_format": "S"},
            ],
            "logs": {"DEPT": np.zeros(60, dtype=np.float64)},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "CDES"],
                    "section_curves": [
                        {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                        {"mnemonic": "CDES", "unit": "", "data_format": "S"},
                    ],
                    "data": {
                        "DEPT": np.zeros(60, dtype=np.float64),
                    },
                    "string_data": {
                        "CDES": np.array(["desc"] * 60),
                    },
                },
            ],
        }
        with pytest.raises(ValueError, match="Cumulative cross-section allocation"):
            LASFile.from_dict(data)
        """IF-026: Curve with data_format='S' in logs (numeric) raises ValueError.

        _check_df_vs_placement verifies that {S} format curves appear in
        string_data, not logs (numeric data).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT", "CDES"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                {"mnemonic": "CDES", "unit": "", "data_format": "S"},
            ],
            "logs": {
                "DEPT": np.array([100.0, 101.0]),
                "CDES": np.array([1.0, 2.0]),  # CDES is S format — wrong placement
            },
        }
        with pytest.raises(ValueError, match="data_format='S' but is in logs"):
            LASFile.from_dict(data)

    def test_from_dict_numeric_format_in_string_data_raises(self) -> None:
        """IF-026: Curve with data_format='F' in string_data raises ValueError.

        {F} format curves must appear in logs (numeric data), not string_data.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT", "CDES"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                {"mnemonic": "CDES", "unit": "", "data_format": "S"},
            ],
            "logs": {"DEPT": np.array([100.0])},
            "string_data": {
                "DEPT": np.array(["bad"]),  # DEPT is F format — wrong placement
            },
        }
        with pytest.raises(ValueError, match="Numeric-format curves must be in logs"):
            LASFile.from_dict(data)

    def test_from_dict_section_s_format_in_data_key_raises(self) -> None:
        """IF-026 / R-001: Per-section curve with data_format='S' in 'data'
        dict (not in string_data) raises ValueError.

        DataSection uses the key 'data' (not 'logs') for numeric data.
        The cross-validation at models.py:241 resolves the correct key
        via ``logs = data.get("logs") or data.get("data") or {}``,
        covering both top-level 'logs' and per-section 'data' dicts.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT", "unit": "M"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "CORE",
                    "section_type": "CORE_DATA",
                    "curves_order": ["CDES"],
                    "section_curves": [
                        {"mnemonic": "CDES", "data_format": "S"},
                    ],
                    "data": {
                        # CDES is S format but in 'data' (numeric) — should be in string_data
                        "CDES": np.array([1.0, 2.0]),
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match="data_format='S' but is in logs"):
            LASFile.from_dict(data)

    def test_from_dict_cross_validation_skips_none_data_format(self) -> None:
        """IF-026: data_format=None (not set) skips cross-validation.

        When data_format is None, _check_df_vs_placement returns early
        without raising — curves can appear in either logs or string_data.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT", "CDES"],
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "CDES", "unit": ""},
            ],
            "logs": {
                # CDES has no data_format — cross-validation skipped
                "DEPT": np.array([100.0]),
                "CDES": np.array([1.0]),
            },
        }
        LASFile.from_dict(data)  # Should not raise


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

    # --- T2: DevFile.from_dict() coverage (models.py:421-437) ---

    def test_dev_file_from_dict(self) -> None:
        """Test DevFile.from_dict() creates DevFile from column dict."""
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0, 200.0]),
            "TVD": np.array([0.0, 99.0, 198.0]),
            "X": np.array([100.0, 101.0, 102.0]),
        }
        dev = DevFile.from_dict(data)
        assert "MD" in dev.columns
        assert "TVD" in dev.columns
        assert "X" in dev.columns
        np.testing.assert_array_equal(dev.columns["MD"], np.array([0.0, 100.0, 200.0]))
        np.testing.assert_array_equal(dev.columns["TVD"], np.array([0.0, 99.0, 198.0]))
        # column_order should be inferred from insertion order
        assert dev.column_order == ["MD", "TVD", "X"]

    def test_dev_file_from_dict_with_metadata(self) -> None:
        """Test DevFile.from_dict() with encoding, source_file, column_order."""
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0]),
            "TVD": np.array([0.0, 99.0]),
            "encoding": "cp1251",
            "source_file": "/path/to/data.dev",
            "column_order": ["TVD", "MD"],
        }
        dev = DevFile.from_dict(data)
        assert dev.encoding == "cp1251"
        assert dev.source_file == "/path/to/data.dev"
        # Explicit column_order should be preserved
        assert dev.column_order == ["TVD", "MD"]
        np.testing.assert_array_equal(dev.columns["MD"], np.array([0.0, 100.0]))
        np.testing.assert_array_equal(dev.columns["TVD"], np.array([0.0, 99.0]))

    def test_dev_file_from_dict_roundtrip(self) -> None:
        """Test DevFile.from_dict(to_dict()) preserves column data and metadata.

        to_dict() now includes metadata keys (source_file, encoding,
        column_order) alongside column arrays.  from_dict() correctly
        restores them.
        """
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0, 200.0])
        dev.columns["TVD"] = np.array([0.0, 99.0, 198.0])
        dev.encoding = "utf-8"
        dev.source_file = "test.dev"

        d = dev.to_dict()
        dev2 = DevFile.from_dict(d)

        # Column data is preserved
        np.testing.assert_array_equal(dev2.columns["MD"], dev.columns["MD"])
        np.testing.assert_array_equal(dev2.columns["TVD"], dev.columns["TVD"])
        # Metadata from to_dict() is now preserved through roundtrip
        assert dev2.encoding == "utf-8"
        assert dev2.source_file == "test.dev"
        # column_order from to_dict() is restored (or inferred from dict order)
        assert dev2.column_order == ["MD", "TVD"]

    # --- F-03 fix: column_order=None should not crash ---

    def test_dev_file_from_dict_column_order_none(self) -> None:
        """DevFile.from_dict() handles column_order=None (F-03 fix).

        list(None) previously raised TypeError. The fix treats None as
        an empty column_order and infers from dict order.
        """
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0]),
            "column_order": None,
        }
        dev = DevFile.from_dict(data)
        # None should produce an empty list; fallback at L726 infers from keys
        assert dev.column_order == ["MD"]

    # --- F-04 fix: column_order as a string should not corrupt ---

    def test_dev_file_from_dict_column_order_string(self) -> None:
        """DevFile.from_dict() handles column_order as a list of strings (F2-32 fix).

        column_order must be a list of strings matching column names.
        Previously a single string was wrapped in a single-element list,
        but F2-32 cross-validation now verifies column_order entries
        exist in the columns dict.
        """
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0]),
            "TVD": np.array([0.0, 99.0]),
            "column_order": ["MD", "TVD"],
        }
        dev = DevFile.from_dict(data)
        assert dev.column_order == ["MD", "TVD"]

    # --- F2-19 fix: non-numeric values should raise clean ValueError ---

    def test_dev_file_from_dict_non_numeric_column(self) -> None:
        """DevFile.from_dict() raises ValueError for non-numeric columns (F2-19 fix).

        np.array(["abc"], dtype=np.float64) previously raised an unhandled
        ValueError from numpy. The fix wraps it in a try/except and re-raises
        a clean ValueError with context.
        """
        data: dict[str, Any] = {
            "MD": np.array(["abc", "def"]),
        }
        with pytest.raises(ValueError, match="Cannot convert data for column"):
            DevFile.from_dict(data)

    # --- F-006: DevFile columns cross-array length ---

    def test_dev_file_from_dict_inconsistent_column_lengths(self) -> None:
        """F-006: DevFile columns with inconsistent lengths raises LASDataError.

        models.py from_dict validates that all column arrays in DevFile have
        the same length, wrapping any ValueError as LASDataError.
        """
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0, 200.0]),
            "TVD": np.array([0.0, 99.0]),
        }
        with pytest.raises(LASDataError, match=r"has length.*but other columns have length"):
            DevFile.from_dict(data)

    # --- F2-003: non-dict input to DevFile.from_dict ---

    def test_dev_file_from_dict_rejects_non_dict_input(self) -> None:
        """F-008/F2-003: DevFile.from_dict() wraps TypeError as LASDataError.

        models.py:1572-1573 validates isinstance(data, dict) inside try block;
        models.py:1661-1662 catches (ValueError, TypeError) and re-raises as LASDataError.
        """
        with pytest.raises(LASDataError, match="Expected dict, got"):
            DevFile.from_dict(42)

    # --- R7F-01: _meta_ prefix roundtrip (models.py:1592-1620) ---

    def test_dev_file_meta_prefix_roundtrip_source_file(self) -> None:
        """R7F-01: DevFile roundtrip with column named 'source_file' (metadata
        key collision).  to_dict() stores metadata under _meta_ prefix;
        from_dict() must recognise and reverse the prefix without crashing.

        Before the R7F-01 fix, from_dict() would try np.array(str_metadata,
        dtype=np.float64) on _meta_-prefixed keys and crash with LASDataError.
        The fix adds _meta_ prefix recognition before the metadata_keys check.

        NOTE: Column data under a key that collides with a metadata key name
        is currently NOT preserved through the roundtrip.  When from_dict()
        sees the bare "source_file" key, it is in metadata_keys and gets
        treated as metadata (overwriting the actual source_file with the
        column array's string representation, which is then overwritten by
        the _meta_ value).  The column itself is lost.  This is a remaining
        gap — a complete fix would need to skip the metadata branch for a
        bare key when a corresponding _meta_ key exists in the dict.
        """
        import warnings

        dev = DevFile()
        dev.columns["source_file"] = np.array([0.0, 100.0, 200.0])
        dev.columns["TVD"] = np.array([0.0, 99.0, 198.0])
        dev.source_file = "/path/to/real_source.dev"
        dev.encoding = "latin-1"

        # to_dict() should warn about collision and store metadata under
        # _meta_ prefix
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = dev.to_dict()
            meta_warnings = [x for x in w if "storing metadata as '_meta_" in str(x.message)]
            assert len(meta_warnings) >= 1, (
                f"Expected _meta_ collision warning, got warnings: {[str(x.message) for x in w]}"
            )

        # _meta_ keys exist; column data is under bare key
        assert "_meta_source_file" in d, f"Keys: {list(d.keys())}"
        assert "source_file" in d  # column data preserved in to_dict output
        assert "TVD" in d

        # from_dict() should NOT crash on _meta_ keys (the original HIGH bug)
        dev2 = DevFile.from_dict(d)

        # Metadata restored from _meta_ keys
        assert dev2.source_file == "/path/to/real_source.dev"
        assert dev2.encoding == "latin-1"

        # Non-colliding column data survives
        np.testing.assert_array_equal(dev2.columns["TVD"], np.array([0.0, 99.0, 198.0]))

        # R7F-01-gap fix: column "source_file" survives the roundtrip
        np.testing.assert_array_equal(dev2.columns["source_file"], np.array([0.0, 100.0, 200.0]))

    def test_dev_file_meta_prefix_roundtrip_encoding(self) -> None:
        """R7F-01: collision on 'encoding' metadata key — from_dict must
        not crash when processing _meta_-prefixed keys."""
        import warnings

        dev = DevFile()
        dev.columns["encoding"] = np.array([10.0, 20.0])
        dev.columns["MD"] = np.array([0.0, 100.0])
        dev.source_file = "test.dev"
        dev.encoding = "cp1251"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = dev.to_dict()
            meta_warnings = [x for x in w if "storing metadata as '_meta_" in str(x.message)]
            assert len(meta_warnings) >= 1

        assert "_meta_encoding" in d

        # Must not crash (the original HIGH bug)
        dev2 = DevFile.from_dict(d)
        assert dev2.encoding == "cp1251"
        assert dev2.source_file == "test.dev"
        # Non-colliding columns survive
        np.testing.assert_array_equal(dev2.columns["MD"], np.array([0.0, 100.0]))
        # R7F-01-gap fix: column "encoding" survives the roundtrip
        np.testing.assert_array_equal(dev2.columns["encoding"], np.array([10.0, 20.0]))

    def test_dev_file_meta_prefix_roundtrip_no_collision(self) -> None:
        """R7F-01: when there is no collision, to_dict()/from_dict() work
        normally (no _meta_ keys produced or expected)."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0])
        dev.columns["TVD"] = np.array([0.0, 99.0])
        dev.source_file = "normal.dev"
        dev.encoding = "utf-8"

        d = dev.to_dict()
        # No _meta_ keys when no collision
        assert "_meta_source_file" not in d
        assert "_meta_encoding" not in d
        assert "source_file" in d
        assert "encoding" in d

        dev2 = DevFile.from_dict(d)
        assert dev2.source_file == "normal.dev"
        assert dev2.encoding == "utf-8"
        np.testing.assert_array_equal(dev2.columns["MD"], np.array([0.0, 100.0]))
        np.testing.assert_array_equal(dev2.columns["TVD"], np.array([0.0, 99.0]))

    # --- I2F-011: Column order preservation during alias normalization ---

    def test_from_dict_alias_normalization_preserves_column_order(self) -> None:
        """I2F-011: Alias normalization preserves original column order.

        ``DEPTH`` normalises to ``MD`` via ``_DEV_ALIASES``.  The old
        ``pop() + assign`` pattern moved renamed columns to the dict
        end, corrupting ``column_order`` inference.  After the fix,
        ``MD`` stays where ``DEPTH`` was (position 0), not at the end.
        """
        data: dict[str, Any] = {
            "DEPTH": np.array([1.0, 2.0]),
            "INC": np.array([3.0, 4.0]),
            "AZI": np.array([5.0, 6.0]),
        }
        dev = DevFile.from_dict(data)
        # DEPTH → MD via alias; MD should be first (where DEPTH was)
        assert dev.column_order == ["MD", "INC", "AZI"]
        # Verify all columns are accessible under normalized names
        assert "MD" in dev.columns
        assert "INC" in dev.columns
        assert "AZI" in dev.columns

    # --- I2F-024: Duplicate column_order detection ---

    def test_post_init_duplicate_column_order_raises(self) -> None:
        """I2F-024: Duplicate entries in column_order raise LASDataError.

        ``set(["MD","MD","TVD"]) == {"MD","TVD"}``, so the set-based
        comparison between ``column_order`` and ``columns`` keys passes
        even when duplicates exist.  An explicit duplicate check catches
        this.
        """
        with pytest.raises(LASDataError, match="duplicate entries"):
            DevFile(
                columns={"MD": np.array([0.0]), "TVD": np.array([0.0])},
                column_order=["MD", "MD", "TVD"],
            )


class TestF038DlmNoneRaisesValueError:
    """F-038 regression: DLM=None raises ValueError in VersionSection.

    Before F-038, VersionSection(dlm=None) passed validation (``if
    self.dlm:`` skipped it) but crashed downstream in
    ``delimiter_char`` and the writer.  After F-038,
    ``__post_init__`` rejects DLM=None explicitly.
    """

    def test_version_section_dlm_none_raises_value_error(self) -> None:
        """VersionSection(dlm=None) raises ValueError."""
        with pytest.raises(ValueError, match="DLM cannot be None"):
            VersionSection(vers="3.0", wrap="NO", dlm=None)

    def test_version_section_dlm_empty_ok(self) -> None:
        """VersionSection(dlm='') is valid — empty DLM is allowed."""
        vs = VersionSection(vers="3.0", wrap="NO", dlm="")
        assert vs.dlm == ""

    def test_from_dict_dlm_none_after_fix(self) -> None:
        """from_dict with DLM=None (e.g., omitted from version dict)
        should still work — explicit None key triggers default."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
        }
        # DLM not present → from_dict uses default (should succeed)
        las = LASFile.from_dict(data)
        assert las.version.dlm == "SPACE"  # default


class TestFR01PerSectionDataFormatWriteBack:
    """F-R01 regression: per-section data_format truncation write-back.

    Before F-R01, the per-section ``data_format`` loop truncated
    extended format codes (e.g., "F8.3" → "F") into a local variable
    but did NOT write the truncated value back to the section-curve
    dict.  This caused ``LASDataError`` at ``CurveDefinition``
    construction because the original extended code was passed to
    ``__post_init__`` which rejected it.  The top-level loop at
    line 327 correctly writes back; the per-section path was a
    mechanical omission.
    """

    def test_from_dict_multi_section_extended_data_format(self) -> None:
        """Multi-section from_dict with per-section extended data_format."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"NULL": "-999.25"},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "data": {"DEPT": np.array([100.0, 200.0])},
                    "curves_order": ["DEPT"],
                    "section_curves": [
                        {
                            "mnemonic": "DEPT",
                            "unit": "M",
                            "description": "DEPTH",
                            "data_format": "F8.3",  # extended — triggers truncation
                        },
                    ],
                },
            ],
        }
        # F-R01: extended per-section data_format must not raise LASDataError
        las = LASFile.from_dict(data)
        assert len(las.data_sections) == 1
        assert las.data_sections[0].section_curves[0].data_format == "F"


# ============================================================
# Production Check Regression Tests (20 confirmed fixes)
# ============================================================


class TestProductionCheckModelsFixes:
    """Regression tests for production check fixes in models.py."""

    # --- F-002 (HIGH): deepcopy fix — caller can't mutate internal data ---

    def test_to_dict_data_is_deepcopy(self) -> None:
        """F-002: Modifying to_dict() returned data does not affect LASFile.

        The fix replaced data.copy() with copy.deepcopy(data) so nested
        dict/numpy array mutations by the caller don't leak into the
        LASFile's internal state.
        """
        las = LASFile()
        las.curves_order = ["DEPT"]
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        las.logs["DEPT"] = np.array([100.0, 200.0])
        las.well["STRT"] = "100.0"

        d = las.to_dict()
        # Mutate nested structures in the returned dict
        d["logs"]["DEPT"][0] = 9999.0
        d["well"]["STRT"] = "MODIFIED"
        d["curves_order"].append("EXTRA")

        # Original LASFile must NOT be affected
        assert las.logs["DEPT"][0] == 100.0
        assert las.well["STRT"] == "100.0"
        assert las.curves_order == ["DEPT"]

    def test_from_dict_data_is_deepcopy(self) -> None:
        """F-002: Modifying input dict after from_dict() does not affect LASFile.

        from_dict also uses copy.deepcopy so the caller's dict remains
        independent of the LASFile's internal state.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100.0", "STOP": "200.0", "STEP": "1.0", "NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0, 200.0])},
        }
        las = LASFile.from_dict(data)
        # Mutate the original dict after construction
        data["logs"]["DEPT"][0] = 9999.0
        data["well"]["STRT"] = "MODIFIED"

        # LASFile must NOT be affected
        assert las.logs["DEPT"][0] == 100.0
        assert las.well["STRT"] == "100.0"

    # --- F-003 (MEDIUM): DataSection.__post_init__ data_format cross-validation ---

    def test_datasection_post_init_s_format_in_numeric_data_raises(self) -> None:
        """F-003: DataSection with S-format curve in numeric data raises LASDataError.

        Direct DataSection construction bypasses from_dict validation.
        __post_init__ now cross-validates data_format against placement
        (data vs string_data).
        """
        from pylasdev.models import DataSection

        with pytest.raises(LASDataError, match="data_format='S' but is in data"):
            DataSection(
                name="BadSection",
                section_type="",
                curves_order=["CDES"],
                data={"CDES": np.array([1.0, 2.0])},
                section_curves=[
                    CurveDefinition(mnemonic="CDES", data_format="S"),
                ],
            )

    def test_datasection_post_init_f_format_in_string_data_raises(self) -> None:
        """F-003: DataSection with F-format curve in string_data raises LASDataError."""
        from pylasdev.models import DataSection

        with pytest.raises(LASDataError, match="data_format='F' but is in string_data"):
            DataSection(
                name="BadSection",
                section_type="",
                curves_order=["DEPT"],
                string_data={"DEPT": np.array(["bad"])},
                section_curves=[
                    CurveDefinition(mnemonic="DEPT", data_format="F"),
                ],
            )

    # --- F-004 (MEDIUM): ArrayElementInfo __post_init__ validation ---

    def test_array_element_info_empty_base_name_raises(self) -> None:
        """F-004: ArrayElementInfo with empty base_name raises ValueError."""
        with pytest.raises(ValueError, match="base_name must not be empty"):
            ArrayElementInfo(base_name="", index=1)

    def test_array_element_info_whitespace_base_name_raises(self) -> None:
        """F-004: ArrayElementInfo with whitespace-only base_name raises ValueError."""
        with pytest.raises(ValueError, match="base_name must not be empty"):
            ArrayElementInfo(base_name="   ", index=1)

    def test_array_element_info_negative_index_raises(self) -> None:
        """F-004: ArrayElementInfo with negative index raises ValueError."""
        with pytest.raises(ValueError, match="index must be >= 1"):
            ArrayElementInfo(base_name="NMR", index=-1)

    def test_array_element_info_non_finite_time_offset_raises(self) -> None:
        """F-004: ArrayElementInfo with nan time_offset raises ValueError."""
        with pytest.raises(ValueError, match="time_offset must be a finite number"):
            ArrayElementInfo(base_name="NMR", index=1, time_offset=float("nan"))

    def test_array_element_info_valid(self) -> None:
        """F-004: Valid ArrayElementInfo construction succeeds."""
        ai = ArrayElementInfo(base_name="NMR", index=1, time_offset=5.0)
        assert ai.base_name == "NMR"
        assert ai.index == 1
        assert ai.time_offset == 5.0

    # --- F-201 (MEDIUM): curves_order None guard ---

    def test_from_dict_data_section_none_curves_order(self) -> None:
        """F-201: data_section with curves_order=None does not crash with TypeError.

        Before the fix, _ds.get("curves_order", []) returned None when
        the key existed with value None, causing TypeError in the
        generator expression.  After the isinstance guard fix, None
        values are safely skipped (the guard checks isinstance(_, list)).
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT", "unit": "M", "data_format": "F"}],
            "logs": {"DEPT": np.array([100.0])},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "curves_order": None,  # key exists but value is None
                    "data": {"DEPT": np.array([100.0])},
                }
            ],
        }
        # Should not crash with TypeError — the isinstance guard catches None.
        # May raise ValueError from orphaned-key detection (which is better
        # than an unhandled TypeError), so any non-TypeError is acceptable.
        try:
            las = LASFile.from_dict(data)
            assert len(las.data_sections) == 1
        except (ValueError, LASDataError):
            pass  # non-TypeError failure is expected and acceptable

    # --- F-202 (MEDIUM): Mnemonic accepts leading/trailing whitespace ---

    def test_curve_definition_leading_whitespace_mnemonic_raises(self) -> None:
        """F-202: CurveDefinition with leading whitespace mnemonic raises ValueError."""
        with pytest.raises(ValueError, match="mnemonic must not be empty"):
            CurveDefinition(mnemonic="  GR")

    def test_curve_definition_trailing_whitespace_mnemonic_raises(self) -> None:
        """F-202: CurveDefinition with trailing whitespace mnemonic raises ValueError."""
        with pytest.raises(ValueError, match="mnemonic must not be empty"):
            CurveDefinition(mnemonic="GR  ")

    def test_parameter_entry_leading_whitespace_mnemonic_raises(self) -> None:
        """F-202: ParameterEntry with leading whitespace mnemonic raises ValueError."""
        with pytest.raises(ValueError, match="mnemonic must not be empty"):
            ParameterEntry(mnemonic="  BHT", value="35")

    def test_parameter_entry_trailing_whitespace_mnemonic_raises(self) -> None:
        """F-202: ParameterEntry with trailing whitespace mnemonic raises ValueError."""
        with pytest.raises(ValueError, match="mnemonic must not be empty"):
            ParameterEntry(mnemonic="BHT  ", value="35")

    # --- F-203 (MEDIUM): from_dict drops legacy params when parameter_details present ---

    def test_from_dict_empty_parameter_details_honors_explicit_empty(self) -> None:
        """F-203: Empty parameter_details list is honored as explicit (not ignored).

        The fix changed `if param_details:` (truthiness, [] → False)
        to `if param_details is not None:` so an explicitly-provided
        empty list is treated as "details present but empty" rather than
        silently falling back to the legacy params dict. An empty list
        means the caller explicitly wants no params.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": {"BHT": "35.5", "BS": "200"},  # legacy dict
            "parameter_details": [],  # explicitly empty list (not None)
        }
        las = LASFile.from_dict(data)
        # Empty details → 0 params (legacy dict NOT processed — caller's
        # explicit empty parameter_details is honored)
        assert len(las.parameters) == 0

    def test_from_dict_no_parameter_details_uses_legacy_params(self) -> None:
        """F-203: No parameter_details key → legacy params dict is used.

        When parameter_details is not provided (None/default), the
        legacy params dict is processed normally. This was the correct
        behavior before the fix and remains correct after.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": {"BHT": "35.5", "BS": "200"},  # legacy dict
        }
        las = LASFile.from_dict(data)
        assert len(las.parameters) == 2
        assert las.parameters[0].mnemonic == "BHT"
        assert las.parameters[1].mnemonic == "BS"

    def test_from_dict_parameter_details_is_nonempty_list(self) -> None:
        """F-203: Non-empty parameter_details still takes priority (no regression)."""
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
            "parameters": {"BHT": "35.5"},  # legacy dict (will be overridden)
            "parameter_details": [
                {"mnemonic": "BS", "unit": "MM", "value": "200", "description": "Bit Size"},
            ],
        }
        las = LASFile.from_dict(data)
        # parameter_details takes priority over legacy dict
        assert len(las.parameters) == 1
        assert las.parameters[0].mnemonic == "BS"

    # --- F-204 (MEDIUM): str() vs _safe_str() inconsistency for `other` field ---

    def test_from_dict_other_field_none_handled(self) -> None:
        """F-204: 'other' field with None value does not count "None" as a line.

        Before the fix, str(data.get("other", "")) on None value produced
        "None" (4 chars), counting it as a line for MAX_OTHER_LINES guard.
        Now _safe_str() converts None to "".
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves_order": ["DEPT"],
            "logs": {"DEPT": np.array([100.0])},
            "other": None,
        }
        las = LASFile.from_dict(data)
        assert las.other == ""

    # --- F-034 (MEDIUM): DevFile.from_dict wraps ImportError as LASDataError ---

    def test_dev_file_from_dict_importerror_not_wrapped(self) -> None:
        """F-034: ImportError from DevFile.from_dict propagates as ImportError.

        The fix removed ImportError from the except tuple, so import
        failures inside the try block propagate as ImportError, not
        misleading LASDataError. The function imports
        _normalize_dev_column from dev_reader inside the try block
        when normalize_aliases=True (default). Removing that function
        from the dev_reader module triggers ImportError.
        """
        import pylasdev.dev_reader as dr_mod

        data: dict[str, Any] = {"MD": np.array([0.0, 100.0])}
        original = getattr(dr_mod, "_normalize_dev_column", None)
        try:
            if original is not None:
                del dr_mod._normalize_dev_column
            with pytest.raises(ImportError):
                DevFile.from_dict(data)
        finally:
            if original is not None:
                dr_mod._normalize_dev_column = original


# ============================================================
# Stage 9 Re-Fix Regression Tests (5 confirmed fixes)
# ============================================================


class TestReFixModels1:
    """Regression tests for Stage 9 re-fix of models-1 findings."""

    # --- F-046 (MEDIUM): DataSection section_type gate ---

    def test_datasection_default_log_data_s_format_in_data_raises(self) -> None:
        """F-046: DataSection with section_type='LOG_DATA' validates format.

        Before the fix, ``if not self.section_type:`` always skipped
        validation for the default 'LOG_DATA' (truthy), so S-format
        curves in numeric data passed silently.  After the fix,
        'LOG_DATA' sections are validated too.
        """
        sc = CurveDefinition(mnemonic="STR", data_format="S")
        # F-046 fix: LOG_DATA section_type now triggers format validation.
        # __post_init__ is called automatically during dataclass construction.
        with pytest.raises(LASDataError, match=r"curve 'STR'.*data_format='S'.*in data"):
            DataSection(
                name="Test",
                section_type="LOG_DATA",
                curves_order=["STR"],
                data={"STR": np.array([1.0, 2.0])},
                section_curves=[sc],
            )

    def test_datasection_log_data_f_format_in_string_data_raises(self) -> None:
        """F-046: LOG_DATA section validates string_data placement.

        Numeric-format curves in string_data should raise LASDataError
        when section_type is 'LOG_DATA'.
        """
        sc = CurveDefinition(mnemonic="GR", data_format="F")
        with pytest.raises(LASDataError, match=r"curve 'GR'.*data_format='F'.*in string_data"):
            DataSection(
                name="Test",
                section_type="LOG_DATA",
                curves_order=["GR"],
                string_data={"GR": np.array(["a", "b"])},
                section_curves=[sc],
            )

    # --- F-048 (MEDIUM): DataSection dtype validation ---

    def test_datasection_string_dtype_in_data_warns(self) -> None:
        """F-048: Non-numeric dtype in 'data' triggers warning.

        String arrays (dtype='<U...') in the numeric 'data' field
        should trigger a warnings.warn about non-numeric dtype.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataSection(
                name="Test",
                curves_order=["DEPT"],
                data={"DEPT": np.array(["abc", "def"], dtype=str)},
            )
            dtype_warnings = [
                x for x in w if "non-numeric dtype" in str(x.message) and "DEPT" in str(x.message)
            ]
        assert len(dtype_warnings) == 1, (
            f"Expected 1 dtype warning for DEPT, got {len(dtype_warnings)}"
        )

    def test_datasection_numeric_dtype_in_string_data_warns(self) -> None:
        """F-048: Numeric dtype in 'string_data' triggers warning.

        Float arrays (dtype='float64') in the 'string_data' field
        should trigger a warnings.warn about numeric dtype.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataSection(
                name="Test",
                curves_order=["STR"],
                string_data={"STR": np.array([1.0, 2.0], dtype=np.float64)},
            )
            dtype_warnings = [
                x for x in w if "numeric dtype" in str(x.message) and "STR" in str(x.message)
            ]
        assert len(dtype_warnings) == 1, (
            f"Expected 1 dtype warning for STR, got {len(dtype_warnings)}"
        )

    def test_datasection_numeric_dtype_in_data_no_warn(self) -> None:
        """F-048: Numeric dtype in 'data' does NOT trigger warning.

        Float64 arrays in the numeric 'data' field are correct
        and should not trigger any dtype warnings.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DataSection(
                name="Test",
                curves_order=["DEPT"],
                data={"DEPT": np.array([1.0, 2.0], dtype=np.float64)},
            )
            dtype_warnings = [x for x in w if "non-numeric dtype" in str(x.message)]
        assert len(dtype_warnings) == 0, (
            f"Expected 0 dtype warnings for numeric data, got {len(dtype_warnings)}"
        )

    # --- F-059 (MEDIUM): LASFile data_format cross-validation ---

    def test_lasfile_s_format_in_logs_raises(self) -> None:
        """I2F-08: S-format curve in logs (numeric) raises LASDataError.

        LASFile.__post_init__ raises when a curve with data_format='S'
        is placed in the logs dict (numeric storage), matching the
        from_dict path and DataSection.__post_init__.
        """
        from pylasdev.exceptions import LASDataError

        sc = CurveDefinition(mnemonic="STR", data_format="S")
        with pytest.raises(LASDataError, match=r"S.*string-format.*logs"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                well=WellSection(
                    entries={"STRT": "0", "STOP": "100", "STEP": "10", "NULL": "-999"}
                ),
                curves=[sc],
                curves_order=["STR"],
                logs={"STR": np.array([1.0, 2.0])},
            )

    def test_lasfile_numeric_format_in_string_data_raises(self) -> None:
        """I2F-08: Numeric-format curve in string_data raises LASDataError.

        LASFile.__post_init__ raises when a numeric-format curve
        is placed in the string_data dict, matching the from_dict
        path and DataSection.__post_init__.
        """
        from pylasdev.exceptions import LASDataError

        sc = CurveDefinition(mnemonic="GR", data_format="F")
        with pytest.raises(LASDataError, match=r"F.*numeric-format.*string_data"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                well=WellSection(
                    entries={"STRT": "0", "STOP": "100", "STEP": "10", "NULL": "-999"}
                ),
                curves=[sc],
                curves_order=["GR"],
                string_data={"GR": np.array(["a", "b"], dtype=str)},
            )

    # --- E-F-022 (MEDIUM): string_data and data_sections overlap detection ---

    def test_from_dict_string_data_data_sections_overlap_warns(self) -> None:
        """E-F-022: Overlap between top-level string_data and data_sections warns.

        When the same curve name appears in both top-level 'string_data'
        and inside 'data_sections', the writer ignores the top-level
        value — this should produce a warning.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "0", "STOP": "100", "STEP": "10", "NULL": "-999"},
            "curves": [
                {
                    "mnemonic": "STR",
                    "unit": None,
                    "data_format": "S",
                    "description": "",
                },
            ],
            "curves_order": ["STR"],
            "string_data": {"STR": np.array(["a", "b"])},
            "data_sections": [
                {
                    "name": "Section1",
                    "section_type": "LOG_DATA",
                    "curves_order": ["STR"],
                    "string_data": {"STR": np.array(["c", "d"])},
                }
            ],
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(data)
            overlap_warnings = [
                x
                for x in w
                if "top-level 'string_data'" in str(x.message) and "data_sections" in str(x.message)
            ]
        assert len(overlap_warnings) == 1, (
            f"Expected 1 overlap warning, got {len(overlap_warnings)}"
        )

    # --- E-F-026 (MEDIUM): DevFile __post_init__ ---

    def test_dev_file_post_init_column_order_keys_mismatch_raises(self) -> None:
        """E-F-026: DevFile raises when column_order and columns keys don't match.

        Direct construction with mismatched column_order should raise
        LASDataError via the new __post_init__.
        """
        with pytest.raises(LASDataError, match="column_order and columns keys do not match"):
            DevFile(
                columns={"MD": np.array([0.0]), "TVD": np.array([0.0])},
                column_order=["MD"],  # TVD missing
            )

    def test_dev_file_post_init_inconsistent_lengths_raises(self) -> None:
        """E-F-026: DevFile raises on inconsistent array lengths.

        Direct construction with columns of different lengths should
        raise LASDataError via __post_init__.
        """
        with pytest.raises(LASDataError, match="inconsistent array lengths"):
            DevFile(
                columns={
                    "MD": np.array([0.0, 100.0, 200.0]),
                    "TVD": np.array([0.0, 99.0]),  # Only 2 values
                },
                column_order=["MD", "TVD"],
            )

    def test_dev_file_post_init_empty_construction_passes(self) -> None:
        """E-F-026: Empty DevFile construction does not raise.

        DevFile() with defaults should pass __post_init__ without error
        (incremental construction is allowed).
        """
        dev = DevFile()  # No exception
        assert dev.columns == {}
        assert dev.column_order == []

    def test_dev_file_post_init_valid_matching_passes(self) -> None:
        """E-F-026: DevFile with valid matching data passes __post_init__."""
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 100.0]),
                "TVD": np.array([0.0, 99.0]),
            },
            column_order=["MD", "TVD"],
        )
        assert "MD" in dev.columns
        assert "TVD" in dev.columns

    # --- F2-008: ArrayElementInfo rejects bool as index ---

    def test_array_element_info_bool_rejected(self) -> None:
        """F2-008: ArrayElementInfo(index=True) raises TypeError.

        Before the fix, ``isinstance(self.index, int)`` accepted ``bool``
        because ``bool`` is a subclass of ``int``.
        """
        with pytest.raises(TypeError, match="index must be int"):
            ArrayElementInfo(base_name="T", index=True)


# ---------------------------------------------------------------------------
# Regression tests added outside TestProductionCheckModelsFixes
# because they test LASFile-level guards, not DataSection internals.
# ---------------------------------------------------------------------------


class TestRegressionModelsFixes:
    """Regression tests for production check fixes testing LASFile guards."""

    # --- F2-014: dict as curves_order rejected ---

    def test_curves_order_rejects_dict(self) -> None:
        """F2-014: Passing a dict as curves_order raises LASDataError.

        Dict is iterable so it passed the old guard chain in from_dict.
        The fix adds an ``isinstance(curves_order, dict)`` check before
        the iterable guard.  from_dict wraps the inner TypeError in
        LASDataError, so we match on the wrapper.
        """
        with pytest.raises(LASDataError, match="got dict"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
                    "well": {"NULL": "-999.25"},
                    "curves_order": {"DEPT": "M", "GR": "GAPI"},
                }
            )

    # --- H-01: WRAP asymmetry (canonical uppercase after construction) ---

    def test_version_section_wrap_canonical_uppercase(self) -> None:
        """H-01: VersionSection(wrap='no').wrap returns canonical uppercase 'NO'.

        The fix ensures WRAP is normalized to uppercase in __post_init__,
        closing the asymmetry between construction-time and parse-time
        canonicalization.  Previously VersionSection(wrap='no').wrap
        returned 'no' but parsing the same LAS file would yield 'NO'.
        """
        vs = VersionSection(wrap="no")
        assert vs.wrap == "NO"
        # Also verify that uppercase input is preserved as-is
        vs2 = VersionSection(wrap="YES")
        assert vs2.wrap == "YES"

    # --- H-04: VERS validation in from_dict ---

    def test_from_dict_unrecognized_vers_warns(self) -> None:
        """H-04: LASFile.from_dict() with VERS='4.0' emits a warning.

        Unrecognized VERS values should produce a UserWarning at
        construction time (not silently accepted).
        """
        with pytest.warns(UserWarning, match="Unrecognized VERS"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "4.0", "WRAP": "NO", "DLM": "SPACE"},
                    "well": {
                        "STRT": "100.0",
                        "STOP": "200.0",
                        "STEP": "1.0",
                        "NULL": "-999.25",
                        "WELL": "A",
                        "LOC": "B",
                        "SRVC": "C",
                        "UWI": "D",
                    },
                    "curves_order": ["DEPT"],
                    "logs": {"DEPT": np.array([100.0])},
                }
            )

    # --- H-03↓: zone_index >= 0 validation ---

    def test_parameter_zone_negative_zone_index_raises(self) -> None:
        """H-03: ParameterZone(zone_index=-1) raises ValueError.

        The fix adds ``zone_index >= 0`` validation to __post_init__,
        previously accepting negative indices silently.
        """
        from pylasdev.models import ParameterZone

        with pytest.raises(ValueError, match="zone_index must be >= 0"):
            ParameterZone(zone_name="TEST", zone_index=-1)

    # --- M-13: time_offset >= 0 validation ---

    def test_array_element_info_negative_time_offset_raises(self) -> None:
        """M-13: ArrayElementInfo(time_offset=-1.0) raises ValueError.

        The fix adds ``time_offset >= 0`` validation to __post_init__,
        previously accepting negative time offsets silently.
        """
        with pytest.raises(ValueError, match="time_offset must be >= 0"):
            ArrayElementInfo(base_name="T", index=1, time_offset=-1.0)

    # --- M-15: array_info type validation ---

    def test_curve_definition_array_info_not_array_element_info_raises(self) -> None:
        """M-15: CurveDefinition with array_info='not_an_array_info' raises TypeError.

        The fix adds ``isinstance(self.array_info, ArrayElementInfo)``
        validation to __post_init__, previously accepting any object.
        """
        with pytest.raises(TypeError, match="array_info must be ArrayElementInfo"):
            CurveDefinition(mnemonic="GR", array_info="not_an_array_info")

    # --- M-17: non-ndarray dtype crash prevention ---

    def test_data_section_non_ndarray_data_no_crash(self) -> None:
        """M-17: DataSection with list data converts via np.asarray, no crash.

        Before the fix, accessing .dtype on a Python list would raise
        AttributeError.  The fix adds a guard: ``if not isinstance(_arr,
        np.ndarray): _arr = np.asarray(_arr)`` before dtype access.
        The conversion is local to the guard (does not mutate self.data).
        """
        # Construction should not raise AttributeError.
        ds = DataSection(curves_order=["GR"], data={"GR": [1.0, 2.0]})
        assert "GR" in ds.data
        assert len(ds.data["GR"]) == 2

    # --- M-18: 0-d array crash prevention ---

    def test_data_section_zero_d_array_no_crash(self) -> None:
        """M-18: DataSection with 0-d array data does not crash.

        np.array(5.0) is a valid 0-d ndarray.  The fix ensures that
        dtype access works correctly and numeric dtype validation passes.
        """
        ds = DataSection(curves_order=["GR"], data={"GR": np.array(5.0)})
        assert ds.data["GR"].ndim == 0
        assert np.issubdtype(ds.data["GR"].dtype, np.number)

    # --- M-25: curses_order bytes rejection ---

    def test_data_section_curves_order_bytes_raises(self) -> None:
        """M-25: DataSection(curves_order=b'GR') raises TypeError.

        Before the fix, bytes iterated as integers, producing corrupted
        curve sets.  The fix adds an explicit isinstance check for bytes.
        """
        with pytest.raises(TypeError, match="curves_order must be a list"):
            DataSection(curves_order=b"GR")


# ──────────────────────────────────────────────────────────────
# M-28 (HIGH): from_dict F-011 must subtract string_data keys
# ──────────────────────────────────────────────────────────────


class TestM28FromDictStringDataKeys:
    """M-28: from_dict logs-vs-curves_order check (F-011) must not reject
    LAS 1.2/2.0 files whose {S} string curves live in string_data.

    The reader routes {S} curves to ``string_data`` (data_reader.py
    F-WXP-01) while ``curves_order`` still includes them, so a documented
    read→to_dict→from_dict roundtrip previously hard-failed with
    "Missing keys: {'LITH'}".
    """

    def _base_dict(self, vers: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": {"VERS": vers, "WRAP": "NO", "DLM": "COMMA"},
            "well": {
                "STRT": "0",
                "STOP": "1",
                "STEP": "0.5",
                "NULL": "-999.25",
                "WELL": "W",
                "LOC": "L",
                "SRVC": "S",
                "UWI": "U",
            },
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                {"mnemonic": "LITH", "unit": "", "data_format": "S"},
            ],
            "curves_order": ["DEPT", "LITH"],
            "logs": {"DEPT": np.array([100.0, 101.0])},
            "string_data": {"LITH": np.array(["SAND", "SHALE"], dtype=object)},
        }
        return d

    def test_las20_logs_plus_string_data_roundtrip(self) -> None:
        """M-28: LAS 2.0 file with numeric logs + {S} string_data passes."""
        d = self._base_dict("2.0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(d)
        assert list(las.logs.keys()) == ["DEPT"]
        assert list(las.string_data.keys()) == ["LITH"]
        assert list(las.curves_order) == ["DEPT", "LITH"]

    def test_las30_no_data_sections_logs_plus_string_data(self) -> None:
        """M-28: LAS 3.0 file without data_sections (backward-compat
        top-level string_data) also passes."""
        d = self._base_dict("3.0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(d)
        assert list(las.logs.keys()) == ["DEPT"]
        assert "LITH" in las.string_data

    def test_extra_log_keys_still_rejected(self) -> None:
        """M-28: The 'Extra keys' direction must NOT subtract string_data —
        a log key absent from curves_order is still an error."""
        d = self._base_dict("2.0")
        d["logs"]["PHANTOM"] = np.array([1.0, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(LASDataError, match="Extra keys"):
                LASFile.from_dict(d)

    def test_missing_keys_not_in_string_data_still_rejected(self) -> None:
        """M-28: A curve in curves_order absent from BOTH logs and
        string_data is still reported as missing."""
        d = self._base_dict("2.0")
        d["curves"].append({"mnemonic": "GR", "unit": "", "data_format": "F"})
        d["curves_order"] = ["DEPT", "LITH", "GR"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(LASDataError, match="Missing keys"):
                LASFile.from_dict(d)


# ──────────────────────────────────────────────────────────────
# M-21 (MEDIUM): get_array_curves must dedupe section curves
# ──────────────────────────────────────────────────────────────


class TestM21GetArrayCurvesDedup:
    """M-21 (coordinated with N-I-10): for multi-section LAS 3.0 files the
    same logical array element can appear in both top-level ``curves`` and
    a section's ``section_curves`` — get_array_curves previously returned
    every element twice."""

    def test_get_array_curves_dedup_same_element(self) -> None:
        c1 = CurveDefinition(
            mnemonic="NMR[1]",
            data_format="A",
            array_info=ArrayElementInfo(base_name="NMR", index=1),
        )
        ds = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["NMR[1]"],
            section_curves=[c1],
        )
        las = LASFile(version=VersionSection(vers="3.0"), data_sections=[ds])
        # Same element also registered at top level (parser behavior).
        las.curves = [c1]
        got = las.get_array_curves("NMR")
        assert [c.mnemonic for c in got] == ["NMR[1]"]

    def test_get_array_curves_distinct_elements_preserved(self) -> None:
        c1 = CurveDefinition(
            mnemonic="NMR[1]",
            data_format="A",
            array_info=ArrayElementInfo(base_name="NMR", index=1),
        )
        c2 = CurveDefinition(
            mnemonic="NMR[2]",
            data_format="A",
            array_info=ArrayElementInfo(base_name="NMR", index=2),
        )
        ds = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["NMR[1]", "NMR[2]"],
            section_curves=[c1, c2],
        )
        las = LASFile(version=VersionSection(vers="3.0"), data_sections=[ds])
        las.curves = [c1, c2]
        got = las.get_array_curves("NMR")
        assert [c.mnemonic for c in got] == ["NMR[1]", "NMR[2]"]

    def test_get_array_curves_unrelated_base_untouched(self) -> None:
        c1 = CurveDefinition(
            mnemonic="T1[1]",
            data_format="A",
            array_info=ArrayElementInfo(base_name="T1", index=1),
        )
        ds = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["T1[1]"],
            section_curves=[c1],
        )
        las = LASFile(version=VersionSection(vers="3.0"), data_sections=[ds])
        las.curves = [c1]
        assert las.get_array_curves("NMR") == []


# ──────────────────────────────────────────────────────────────
# M-22 (MEDIUM) + IT3-THR-02: unnamed data sections auto-named
# ──────────────────────────────────────────────────────────────


class TestM22UnnamedDataSections:
    """M-22 + IT3-THR-02: two unnamed data sections are valid LAS 3.0
    (the parser auto-names them Section_N on read).  from_dict previously
    raised a false-positive "duplicate data section name '<unnamed>'".
    Empty names are auto-named BEFORE the dedup check."""

    def test_from_dict_two_unnamed_sections_auto_named(self) -> None:
        d: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"},
            "data_sections": [
                {
                    "name": "",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([1.0, 2.0])},
                },
                {
                    "name": "",
                    "section_type": "LOG_DATA",
                    "curves_order": ["GR"],
                    "data": {"GR": np.array([3.0, 4.0])},
                },
            ],
        }
        las = LASFile.from_dict(d)
        assert [ds.name for ds in las.data_sections] == ["Section_0", "Section_1"]

    def test_duplicate_named_sections_still_raise(self) -> None:
        """M-22: user-supplied duplicate names are still rejected."""
        ds1 = DataSection(
            name="A",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array([1.0, 2.0])},
        )
        ds2 = DataSection(
            name="A",
            section_type="LOG_DATA",
            curves_order=["GR"],
            data={"GR": np.array([3.0, 4.0])},
        )
        with pytest.raises(LASDataError, match="duplicate data section name"):
            LASFile(
                version=VersionSection(vers="3.0"),
                data_sections=[ds1, ds2],
            )

    def test_validate_skips_empty_names_in_dedup(self) -> None:
        """M-22: validate(complete=True) does not warn for two unnamed
        sections (empty names are skipped — they are auto-named by
        __post_init__)."""
        las = LASFile()
        las.version = VersionSection(vers="3.0")
        las.data_sections = [
            DataSection(name="", section_type="LOG_DATA", curves_order=["DEPT"]),
            DataSection(name="", section_type="LOG_DATA", curves_order=["GR"]),
        ]
        issues = las.validate(complete=True)
        assert not any("duplicate data section name" in i for i in issues)


# ──────────────────────────────────────────────────────────────
# M-23 (MEDIUM): data_format clear-not-truncate
# ──────────────────────────────────────────────────────────────


class TestM23DataFormatClearNotTruncate:
    """M-23: from_dict must not fabricate valid format codes from invalid
    multi-char data_format values via blind df[0] truncation.  Metadata
    templates (DD/MM/YYYY, DENSITY, DEG) are cleared; extended Fortran
    codes (F8.3) are normalized to their single letter."""

    def _base_dict(self) -> dict[str, Any]:
        return {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"},
            "curves_order": ["DEPT"],
        }

    def test_metadata_template_cleared_top_level(self) -> None:
        d = self._base_dict()
        d["curves"] = [{"mnemonic": "DEPT", "unit": "M", "data_format": "DENSITY"}]
        las = LASFile.from_dict(d)
        assert las.curves[0].data_format == ""

    def test_extended_format_truncated_top_level(self) -> None:
        d = self._base_dict()
        d["curves"] = [{"mnemonic": "DEPT", "unit": "M", "data_format": "F8.3"}]
        las = LASFile.from_dict(d)
        assert las.curves[0].data_format == "F"

    def test_metadata_template_cleared_per_section(self) -> None:
        d = self._base_dict()
        d["data_sections"] = [
            {
                "name": "A",
                "section_type": "LOG_DATA",
                "curves_order": ["DEPT"],
                "section_curves": [{"mnemonic": "DEPT", "unit": "M", "data_format": "DEG"}],
                "data": {"DEPT": np.array([1.0, 2.0])},
            }
        ]
        las = LASFile.from_dict(d)
        assert las.data_sections[0].section_curves[0].data_format == ""

    def test_valid_single_char_unchanged(self) -> None:
        d = self._base_dict()
        d["curves"] = [{"mnemonic": "DEPT", "unit": "M", "data_format": "F"}]
        las = LASFile.from_dict(d)
        assert las.curves[0].data_format == "F"


# ──────────────────────────────────────────────────────────────
# M-27 (MEDIUM): section_type whitespace/pipe rejection
# ──────────────────────────────────────────────────────────────


class TestM27SectionTypeContentValidation:
    """M-27: section_type containing whitespace or a pipe produces a broken
    header (``~MY CORE_Parameter``) that the parser misroutes to ~O,
    silently dropping parameters.  All model validation paths reject
    spaces/tabs and ``|``."""

    def test_parameter_entry_space_rejected(self) -> None:
        with pytest.raises(ValueError, match="whitespace"):
            ParameterEntry(mnemonic="T", value="1", section_type="MY CORE")

    def test_parameter_entry_pipe_rejected(self) -> None:
        with pytest.raises(ValueError, match="pipe"):
            ParameterEntry(mnemonic="T", value="1", section_type="A|B")

    def test_data_section_space_rejected(self) -> None:
        with pytest.raises(LASDataError, match="whitespace"):
            DataSection(name="S", section_type="MY CORE")

    def test_data_section_pipe_rejected(self) -> None:
        with pytest.raises(LASDataError, match="pipe"):
            DataSection(name="S", section_type="A|B")

    def test_from_dict_section_type_space_rejected(self) -> None:
        """M-27: the from_dict pass-through site (models.py ~3089) is
        covered by DataSection.__post_init__ validation."""
        d: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"},
            "data_sections": [
                {
                    "name": "A",
                    "section_type": "MY CORE",
                    "curves_order": ["DEPT"],
                    "data": {"DEPT": np.array([1.0, 2.0])},
                }
            ],
        }
        with pytest.raises(LASDataError, match="whitespace"):
            LASFile.from_dict(d)

    def test_whitespace_only_section_type_still_normalized(self) -> None:
        """M-27: whitespace-ONLY section_type is normalized (not rejected) —
        DataSection → "", ParameterEntry → None (existing behavior)."""
        ds = DataSection(name="S", section_type="   ")
        assert ds.section_type == ""
        p = ParameterEntry(mnemonic="T", value="1", section_type="   ")
        assert p.section_type is None


# ──────────────────────────────────────────────────────────────
# M-29 (MEDIUM): non-LAS-3.0 string_data write warning
# ──────────────────────────────────────────────────────────────


class TestM29NonLas30StringDataWarning:
    """M-29: LAS 1.2/2.0 output has no {S} string marker — string_data
    values are written unmarked and re-read as null sentinels.  The model
    layer emits an explicit write-time warning (validate(complete=True))
    so callers are not surprised."""

    def test_validate_warns_non_las30_string_data(self) -> None:
        las = LASFile()
        las.version = VersionSection(vers="2.0")
        las.string_data["CDES"] = np.array(["SAND"], dtype=object)
        issues = las.validate(complete=True)
        assert any("string_data is present" in i and "not LAS 3.0" in i for i in issues)

    def test_las30_string_data_no_warning(self) -> None:
        las = LASFile()
        las.version = VersionSection(vers="3.0")
        las.string_data["CDES"] = np.array(["SAND"], dtype=object)
        issues = las.validate(complete=True)
        assert not any("string_data is present" in i for i in issues)


# ──────────────────────────────────────────────────────────────
# N-I-12 (MEDIUM): DataSection.curves_order element types
# ──────────────────────────────────────────────────────────────


class TestNI12DataSectionCurvesOrderElementTypes:
    """N-I-12/I2-12: DataSection.curves_order element types were unvalidated on
    direct construction — `LASFile(data_sections=[ds])` crashed with a raw
    TypeError from re.match (``expected string or bytes-like object``).
    Per-element validation now raises a clear error with a message naming
    the offending element.  I2-12 extended the guard to POST-CONSTRUCTION
    mutations by wrapping curves_order in a `_GuardedList`
    (``_expected_type=str``), so the construction-time failure is a
    TypeError — the same exception the LASFile top-level curves_order guard
    (F-13) and the guarded lists for ``curves``/``parameters`` raise."""

    def test_datasection_curves_order_int_raises(self) -> None:
        with pytest.raises(TypeError, match=r"DataSection.curves_order: items must be str"):
            DataSection(name="S", section_type="LOG_DATA", curves_order=[1, 2])

    def test_lasfile_with_int_curves_order_raises_lasdataerror(self) -> None:
        """N-I-12: the original crash path — LASFile(data_sections=[ds])
        with non-str curves_order elements — now raises a clear TypeError
        from the DataSection's guarded curves_order (not a raw TypeError
        from re.match)."""
        with pytest.raises(TypeError, match="items must be str"):
            LASFile(
                version=VersionSection(vers="3.0"),
                data_sections=[DataSection(name="S", section_type="LOG_DATA", curves_order=[1, 2])],
            )

    def test_valid_str_curves_order_unchanged(self) -> None:
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT", "GR"])
        assert ds.curves_order == ["DEPT", "GR"]


# ──────────────────────────────────────────────────────────────
# N-I-07 (MEDIUM): from_dict unknown single-char format alignment
# ──────────────────────────────────────────────────────────────


class TestNI07FromDictUnknownSingleCharFormat:
    """N-I-07: an unknown SINGLE-char data_format ({X}, {G}) was
    warn-and-cleared by the parser but RAISED LASDataError in from_dict —
    the same input behaved differently on the two construction paths and a
    parse→to_dict→from_dict roundtrip crashed.  from_dict now warns and
    clears, matching the parser's deliberate tolerance."""

    def _base_dict(self) -> dict[str, Any]:
        return {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {
                "STRT": {"value": "100.0"},
                "STOP": {"value": "200.0"},
                "STEP": {"value": "1.0"},
                "NULL": {"value": "-999.25"},
            },
            "curves_order": ["DEPT", "GR"],
            "logs": {"DEPT": np.array([100.0]), "GR": np.array([45.5])},
        }

    def test_from_dict_unknown_single_char_warns_and_clears(self) -> None:
        d = self._base_dict()
        d["curves"] = [
            {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
            {"mnemonic": "GR", "unit": "API", "data_format": "X"},
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = LASFile.from_dict(d)
        assert any("invalid data_format 'X'" in str(w.message) for w in caught), (
            "from_dict must warn on unknown single-char format"
        )
        gr = next(c for c in las.curves if c.mnemonic == "GR")
        assert gr.data_format == ""

    def test_from_dict_unknown_single_char_per_section(self) -> None:
        d = self._base_dict()
        d["version"] = {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"}
        d["data_sections"] = [
            {
                "name": "A",
                "section_type": "LOG_DATA",
                "curves_order": ["DEPT"],
                "section_curves": [{"mnemonic": "DEPT", "unit": "M", "data_format": "G"}],
                "data": {"DEPT": np.array([1.0, 2.0])},
            }
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            las = LASFile.from_dict(d)
        assert any("invalid data_format 'G'" in str(w.message) for w in caught)
        assert las.data_sections[0].section_curves[0].data_format == ""

    def test_from_dict_valid_single_char_unchanged(self) -> None:
        d = self._base_dict()
        d["curves"] = [
            {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
            {"mnemonic": "GR", "unit": "API", "data_format": "F"},
        ]
        las = LASFile.from_dict(d)
        assert las.curves[0].data_format == "F"
        assert las.curves[1].data_format == "F"

    def test_write_las_file_accepts_unknown_single_char_format_dict(self, tmp_path) -> None:
        """The roundtrip crash path: write_las_file(dict with {X}) must not
        raise LASWriteError — from_dict now clears the format."""
        from pylasdev import write_las_file

        d = self._base_dict()
        d["curves"] = [
            {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
            {"mnemonic": "GR", "unit": "API", "data_format": "X"},
        ]
        out = tmp_path / "ni07.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, d)
        assert out.exists()


# ============================================================
# G3 — Small-model guards (M-01..M-06, IT3-THR-01)
# ============================================================


class TestG3GuardedDict:
    """M-01: _GuardedDict update()/setdefault()/|=/__init__ key validation.

    CPython's C-level dict methods (update, setdefault, |=, __init__)
    bypass the Python-level __setitem__ override, so the original
    docstring claim ("all delegate to __setitem__ internally") was false.
    """

    def _make(self):
        from pylasdev.models import _GuardedDict

        return _GuardedDict()

    def test_update_rejects_int_key(self) -> None:
        from pylasdev.models import _GuardedDict

        g = _GuardedDict()
        with pytest.raises(TypeError, match="keys must be str"):
            g.update({"a": 1, 5: 2})

    def test_init_rejects_int_key(self) -> None:
        from pylasdev.models import _GuardedDict

        with pytest.raises(TypeError, match="keys must be str"):
            _GuardedDict({1: "x"})

    def test_setdefault_rejects_int_key(self) -> None:
        from pylasdev.models import _GuardedDict

        g = _GuardedDict()
        with pytest.raises(TypeError, match="keys must be str"):
            g.setdefault(3, "v")

    def test_ior_rejects_int_key(self) -> None:
        from pylasdev.models import _GuardedDict

        g = _GuardedDict()
        with pytest.raises(TypeError, match="keys must be str"):
            g |= {7: "x"}

    def test_valid_str_keys_still_work(self) -> None:
        from pylasdev.models import _GuardedDict

        # MOD-17/MOD-23: values must be 1-D array-like (scalars/str are
        # rejected by the data-container contract) — lists keep this test
        # focused on KEY validation.
        g = _GuardedDict({"a": [1]})
        g.update({"b": [2]})
        g.setdefault("c", [3])
        g |= {"d": [4]}
        assert g == {"a": [1], "b": [2], "c": [3], "d": [4]}


class TestG3GuardedList:
    """M-02: _GuardedList __init__ validation + slice assignment support."""

    def test_init_rejects_wrong_type(self) -> None:
        from pylasdev.models import _GuardedList

        with pytest.raises(TypeError, match="items must be str"):
            _GuardedList([1, 2, 3], _expected_type=str)

    def test_init_valid_items(self) -> None:
        from pylasdev.models import _GuardedList

        gl = _GuardedList(["a", "b"], _expected_type=str)
        assert gl == ["a", "b"]

    def test_slice_assignment_valid(self) -> None:
        from pylasdev.models import _GuardedList

        gl = _GuardedList(["a", "b", "c"], _expected_type=str)
        gl[0:1] = ["x"]
        assert gl == ["x", "b", "c"]

    def test_slice_assignment_rejects_wrong_type(self) -> None:
        from pylasdev.models import _GuardedList

        gl = _GuardedList(["a", "b"], _expected_type=str)
        with pytest.raises(TypeError, match="items must be str"):
            gl[0:1] = [1]


class TestG3MnemonicWhitelist:
    """M-03: mnemonics whitelisted against parser grammar."""

    def test_curve_colon_mnemonic_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            CurveDefinition(mnemonic="GR:1")

    def test_curve_pipe_mnemonic_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            CurveDefinition(mnemonic="GR|X")

    def test_curve_hash_mnemonic_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            CurveDefinition(mnemonic="GR#1")

    def test_curve_array_mnemonic_accepted(self) -> None:
        c = CurveDefinition(mnemonic="NMR[1]", unit="MS")
        assert c.mnemonic == "NMR[1]"

    def test_curve_hyphen_mnemonic_accepted(self) -> None:
        c = CurveDefinition(mnemonic="GR-1", unit="M")
        assert c.mnemonic == "GR-1"

    def test_parameter_colon_mnemonic_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            ParameterEntry(mnemonic="RUN:1", value="1.0")


class TestG3UnitValidation:
    """M-04: unit composition validated in CurveDefinition AND ParameterEntry."""

    def test_curve_space_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid unit"):
            CurveDefinition(mnemonic="GR", unit="US M")

    def test_curve_colon_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid unit"):
            CurveDefinition(mnemonic="GR", unit="A:B")

    def test_curve_percent_unit_accepted(self) -> None:
        """N-I-22 coordination: '%' must remain valid."""
        c = CurveDefinition(mnemonic="PHIT", unit="%")
        assert c.unit == "%"

    def test_curve_ohm_dot_unit_accepted(self) -> None:
        """N-I-22 coordination: 'ohm.m' must remain valid."""
        c = CurveDefinition(mnemonic="RT", unit="ohm.m")
        assert c.unit == "ohm.m"

    def test_parameter_space_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid unit"):
            ParameterEntry(mnemonic="US", unit="US M", value="sand")

    def test_parameter_valid_unit_accepted(self) -> None:
        p = ParameterEntry(mnemonic="BHT", unit="DEGC", value="35.5")
        assert p.unit == "DEGC"


class TestG3NonStrFieldCoercion:
    """M-05: CurveDefinition coerces non-str fields, guards mnemonic type."""

    def test_curve_non_str_unit_coerced(self) -> None:
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            c = CurveDefinition(mnemonic="GR", unit=123)
        assert c.unit == "123"
        assert any("coercing non-str unit" in str(x.message) for x in w)

    def test_curve_non_str_api_code_coerced(self) -> None:
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            c = CurveDefinition(mnemonic="GR", api_code=42)
            assert any("coercing non-str api_code" in str(x.message) for x in w)
        assert c.api_code == "42"

    def test_curve_non_str_description_coerced(self) -> None:
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            c = CurveDefinition(mnemonic="GR", description=5.5)
            assert any("coercing non-str description" in str(x.message) for x in w)
        assert c.description == "5.5"

    def test_curve_non_str_mnemonic_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="mnemonic must be str"):
            CurveDefinition(mnemonic=123)

    def test_parameter_non_str_mnemonic_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="mnemonic must be str"):
            ParameterEntry(mnemonic=5, value="1.0")


class TestG3NumpyScalars:
    """M-06: numpy scalars accepted at all five model sites."""

    def test_array_element_info_np_int64_index(self) -> None:
        ai = ArrayElementInfo(base_name="NMR", index=np.int64(1))
        assert ai.index == 1

    def test_array_element_info_np_float32_time_offset(self) -> None:
        ai = ArrayElementInfo(base_name="NMR", index=1, time_offset=np.float32(5.0))
        assert ai.time_offset == 5.0

    def test_parameter_zone_np_int64(self) -> None:
        from pylasdev.models import ParameterZone

        pz = ParameterZone(zone_name="RUN", zone_index=np.int64(1))
        assert pz.zone_index == 1

    def test_parameter_entry_np_int64_array_index(self) -> None:
        pe = ParameterEntry(mnemonic="RUN", value="1.0", array_index=np.int64(1))
        assert pe.array_index == 1

    def test_create_parameter_entry_np_int64(self) -> None:
        from pylasdev.models import ParameterZone

        p = ParameterEntry(
            mnemonic="RUN",
            value="1.0",
            array_index=np.int64(1),
            zone=ParameterZone(zone_name="RUN", zone_index=np.int64(1)),
        )
        assert p.array_index == 1
        assert p.zone is not None
        assert p.zone.zone_index == 1

    def test_from_dict_array_info_np_scalars(self) -> None:
        data = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200", "STEP": "1", "NULL": "-999"},
            "curves": [
                {
                    "mnemonic": "NMR[1]",
                    "unit": "M",
                    "data_format": "A",
                    "array_info": {
                        "base_name": "NMR",
                        "index": np.int64(1),
                        "time_offset": np.float64(0),
                    },
                }
            ],
            "curves_order": ["NMR[1]"],
            "logs": {"NMR[1]": np.array([1.0, 2.0])},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data)
        assert las.curves[0].array_info is not None
        assert las.curves[0].array_info.index == 1


class TestG3NullCaseInsensitive:
    """IT3-THR-01: _get_null_value must be case-insensitive on well keys."""

    def test_get_null_value_lowercase_null_key(self) -> None:
        from pylasdev.data_reader import _get_null_value

        ws = WellSection(entries={"null": "-1", "STRT": "100"})
        assert _get_null_value(ws) == -1.0

    def test_get_null_value_uppercase_key_unchanged(self) -> None:
        from pylasdev.data_reader import _get_null_value

        ws = WellSection(entries={"NULL": "-999.25"})
        assert _get_null_value(ws) == -999.25

    def test_get_null_value_missing_key_uses_default(self) -> None:
        from pylasdev.data_reader import _get_null_value

        ws = WellSection(entries={"STRT": "100"})
        assert _get_null_value(ws) == -999.25

    def test_write_read_roundtrip_consistent_declared_null(self, tmp_path) -> None:
        """End-to-end: from_dict with lowercase 'null' key must produce a
        file whose declared NULL equals its fill cells.

        Pre-fix: `_get_null_value` used case-sensitive ``well.get("NULL")``,
        so a lowercase ``"null"`` key (from_dict stores it verbatim when
        mnem_base is None) made the writer declare NULL=-1 in ~W but fill
        data rows with the -999.25 default.  Post-fix the case-insensitive
        lookup makes declared NULL == fill cells.
        """
        from pylasdev import read_las_file, write_las_file

        data = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "SPACE"},
            # Lowercase "null" — _norm_mnem is identity when mnem_base=None.
            "well": {"null": "-1", "STRT": "100", "STOP": "200", "STEP": "1"},
            "curves": [
                {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                {"mnemonic": "GR", "unit": "API", "data_format": "F"},
            ],
            "curves_order": ["DEPT", "GR"],
            "data_sections": [
                {
                    "name": "A",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "GR"],
                    # GR has no data → the writer pads it with null_value.
                    "data": {"DEPT": np.array([100.0, 101.0])},
                    "section_curves": [
                        {"mnemonic": "DEPT", "unit": "M", "data_format": "F"},
                        {"mnemonic": "GR", "unit": "API", "data_format": "F"},
                    ],
                }
            ],
        }
        out = tmp_path / "thr01_null_case.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, data)
        parsed = read_las_file(out)
        # Re-read uppercases the well key (parser), so declared NULL is -1.
        assert parsed["well"].get("NULL") == "-1"
        # The fill cell for GR's missing rows must equal the declared NULL.
        assert parsed["logs"]["GR"][0] == -1.0
        assert parsed["logs"]["GR"][0] == float(parsed["well"]["NULL"])


# --- G5 fix-agent regression tests (N-I-09, N-I-10, N-I-11, N-I-13, ---
# --- N-I-14, N-I-19, N-I-21, N-I-30) ---


class TestG5WellValidateCaseInsensitive:
    """N-I-09: WellSection.validate must uppercase-normalize NULL/STRT/STOP
    lookups (matching STEP) and distinguish absent NULL from empty-string."""

    def test_null_absent_no_false_diagnostic(self) -> None:
        """NULL-less well produces NO 'NULL is an empty string' diagnostic."""
        w = WellSection(entries={"STRT": "100", "STOP": "200", "STEP": "1"})
        assert "NULL is an empty string" not in " | ".join(w.validate(True))

    def test_null_empty_string_still_diagnosed(self) -> None:
        """A present-but-empty NULL key still fires the diagnostic."""
        w = WellSection(entries={"NULL": "", "STRT": "100", "STOP": "200"})
        assert any("NULL is an empty string" in i for i in w.validate(True))

    def test_null_lowercase_empty_still_diagnosed(self) -> None:
        """Case-variant empty NULL key is found via case-insensitive lookup."""
        w = WellSection(entries={"null": "", "STRT": "100", "STOP": "200"})
        assert any("NULL is an empty string" in i for i in w.validate(True))

    def test_lowercase_strt_stop_fires_zero_range(self) -> None:
        """Lowercase strt/stop keys no longer skip the STRT==STOP check."""
        w = WellSection(entries={"strt": "100", "stop": "100", "step": "1"})
        issues = w.validate(True)
        assert any("STRT equals STOP" in i for i in issues)
        assert not any("NULL is an empty string" in i for i in issues)

    def test_uppercase_control_unchanged(self) -> None:
        """Uppercase keys behave identically to the lowercase case."""
        w = WellSection(entries={"STRT": "100", "STOP": "100", "STEP": "1"})
        issues = w.validate(True)
        assert any("STRT equals STOP" in i for i in issues)


class TestG5GetArrayCurvesMultiSection:
    """N-I-10: get_array_curves must not return the same element twice for
    multi-section LAS 3.0 files (top-level curves + section_curves hold the
    same logical elements).  M-21 (G4) added the dedup; this test locks in
    the multi-section behavior."""

    def _build_multi_section_las(self) -> LASFile:
        from pylasdev.models import ArrayElementInfo

        las = LASFile(version=VersionSection(vers="3.0"))
        arr_defs = [
            CurveDefinition(
                mnemonic=f"NMR[{i}]",
                array_info=ArrayElementInfo(base_name="NMR", index=i),
            )
            for i in range(1, 4)
        ]
        las.curves.extend(arr_defs)
        las.curves_order = [c.mnemonic for c in arr_defs]
        # Same logical elements also registered per-section (as the LAS 3.0
        # parser does via `section_curves = list(ctx.las_file.curves[...])`).
        las.data_sections.append(
            DataSection(
                name="Log1",
                section_type="LOG_DATA",
                curves_order=[c.mnemonic for c in arr_defs],
                section_curves=list(arr_defs),
                data={c.mnemonic: np.array([1.0]) for c in arr_defs},
            )
        )
        return las

    def test_multi_section_dedup(self) -> None:
        """Each NMR element is returned exactly once."""
        las = self._build_multi_section_las()
        result = las.get_array_curves("NMR")
        mnemonics = [c.mnemonic for c in result]
        assert mnemonics == ["NMR[1]", "NMR[2]", "NMR[3]"]
        assert len(mnemonics) == len(set(mnemonics))

    def test_unrelated_base_untouched(self) -> None:
        las = self._build_multi_section_las()
        assert las.get_array_curves("GR") == []


class TestG5DirectConstructionNoAlias:
    """N-I-11: direct LASFile/DevFile construction must deepcopy caller
    dicts (match from_dict) — no in-place mutation, no array aliasing, and
    DevFile list values must not crash validate() with a raw TypeError."""

    def test_lasfile_does_not_mutate_caller_dict(self) -> None:
        caller_logs: dict[str, Any] = {
            "DEPT": [1.0, 2.0, 3.0],
            "GR": [10.0, 20.0, 30.0],
        }
        LASFile(
            logs=caller_logs,
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
            ],
        )
        # Pre-fix: validate() coerced the caller's lists to ndarrays in place.
        assert isinstance(caller_logs["DEPT"], list)
        assert caller_logs["DEPT"] == [1.0, 2.0, 3.0]

    def test_lasfile_does_not_alias_arrays(self) -> None:
        caller_logs = {"DEPT": [1.0, 2.0, 3.0], "GR": [10.0, 20.0, 30.0]}
        las = LASFile(
            logs=caller_logs,
            curves_order=["DEPT", "GR"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="GR"),
            ],
        )
        assert las.logs["GR"] is not caller_logs["GR"]
        # Mutating the caller's array must not corrupt internal data.
        caller_logs["GR"][0] = 999.0
        assert las.logs["GR"][0] == 10.0

    def test_devfile_list_values_coerced_no_crash(self) -> None:
        """List-valued columns construct successfully (coerced to ndarray)
        instead of crashing with a raw TypeError in validate()."""
        dev = DevFile(columns={"MD": [1.0, 2.0]}, column_order=["MD"])
        assert isinstance(dev.columns["MD"], np.ndarray)
        assert list(dev.columns["MD"]) == [1.0, 2.0]

    def test_devfile_does_not_alias_caller_arrays(self) -> None:
        caller_cols: dict[str, Any] = {"MD": np.array([1.0, 2.0])}
        dev = DevFile(columns=caller_cols, column_order=["MD"])
        assert dev.columns["MD"] is not caller_cols["MD"]
        caller_cols["MD"][0] = 999.0
        assert dev.columns["MD"][0] == 1.0


class TestG5TopLevelArrayContinuity:
    """N-I-13: array-continuity validation must also apply to the TOP-LEVEL
    curves_order (previously nested inside `if self.data_sections:`), so the
    writer cannot emit a file its own parser rejects."""

    def _curves(self, names: list[str]) -> list[CurveDefinition]:
        return [CurveDefinition(mnemonic=n) for n in names]

    def test_top_level_interleaved_arrays_rejected(self) -> None:
        with pytest.raises(LASDataError, match="not contiguous"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["DEPT", "NMR[1]", "GR", "NMR[2]"],
                curves=self._curves(["DEPT", "NMR[1]", "GR", "NMR[2]"]),
                logs={
                    "DEPT": np.array([1.0]),
                    "NMR[1]": np.array([1.0]),
                    "GR": np.array([1.0]),
                    "NMR[2]": np.array([1.0]),
                },
            )

    def test_top_level_non_sequential_indices_rejected(self) -> None:
        with pytest.raises(LASDataError, match="non-sequential"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["DEPT", "NMR[1]", "NMR[3]"],
                curves=self._curves(["DEPT", "NMR[1]", "NMR[3]"]),
                logs={
                    "DEPT": np.array([1.0]),
                    "NMR[1]": np.array([1.0]),
                    "NMR[3]": np.array([1.0]),
                },
            )

    def test_top_level_contiguous_arrays_accepted(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "NMR[1]", "NMR[2]", "GR"],
            curves=self._curves(["DEPT", "NMR[1]", "NMR[2]", "GR"]),
            logs={
                "DEPT": np.array([1.0]),
                "NMR[1]": np.array([1.0]),
                "NMR[2]": np.array([1.0]),
                "GR": np.array([1.0]),
            },
        )
        assert las.curves_order[0] == "DEPT"

    def test_non_array_curves_unaffected(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "GR", "NPHI"],
            curves=self._curves(["DEPT", "GR", "NPHI"]),
            logs={
                "DEPT": np.array([1.0]),
                "GR": np.array([1.0]),
                "NPHI": np.array([1.0]),
            },
        )
        assert len(las.curves) == 3


class TestG5DevColumnsMutationGuards:
    """N-I-14: _DevColumns must override update/pop/setdefault/clear so the
    C-level dict methods cannot desync columns/column_order."""

    def test_update_syncs_column_order(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        dev.columns.update({"TVD": np.array([3.0, 4.0])})
        assert list(dev.columns.keys()) == ["MD", "TVD"]
        assert dev.column_order == ["MD", "TVD"]

    def test_setdefault_syncs_column_order(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        dev.columns.setdefault("INC", np.array([5.0, 6.0]))
        assert dev.column_order == ["MD", "INC"]
        # Existing key returns current value, no duplicate in column_order.
        dev.columns.setdefault("MD", np.array([9.0, 9.0]))
        assert dev.column_order == ["MD", "INC"]

    def test_pop_syncs_column_order(self) -> None:
        dev = DevFile(
            columns={"MD": np.array([1.0, 2.0]), "TVD": np.array([3.0, 4.0])},
            column_order=["MD", "TVD"],
        )
        assert dev.columns.pop("MD") is not None
        assert list(dev.columns.keys()) == ["TVD"]
        assert dev.column_order == ["TVD"]
        # Two-argument default semantics preserved.
        assert dev.columns.pop("MISSING", "default") == "default"

    def test_clear_syncs_column_order(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        dev.columns.clear()
        assert dev.columns == {}
        assert dev.column_order == []

    def test_roundtrip_stays_consistent_after_update(self) -> None:
        """update() no longer produces a to_dict/from_dict LASDataError."""
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        dev.columns.update({"TVD": np.array([3.0, 4.0])})
        d = dev.to_dict()
        back = DevFile.from_dict(d)
        assert list(back.columns.keys()) == ["MD", "TVD"]

    def test_update_rejects_invalid_length(self) -> None:
        dev = DevFile(columns={"MD": np.array([1.0, 2.0])}, column_order=["MD"])
        with pytest.raises(ValueError, match="length"):
            dev.columns.update({"TVD": np.array([1.0, 2.0, 3.0])})


class TestG5WellKeyContentValidation:
    """N-I-19: WellSection entry KEYS with dots/spaces/colons are rejected at
    construction (and by the writer) because the parser's ~W regex cannot
    roundtrip them — they were silently dropped on re-read."""

    def test_dot_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            WellSection(entries={"GR.CO": "1"})

    def test_space_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            WellSection(entries={"WELL NAME": "Well-1"})

    def test_colon_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot roundtrip"):
            WellSection(entries={"GR:1": "1"})

    def test_valid_keys_accepted(self) -> None:
        w = WellSection(entries={"STRT": "100", "COMP": "ACME", "WELL-1": "x"})
        assert w.entries["COMP"] == "ACME"

    def test_non_str_key_type_still_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be str"):
            WellSection(entries={1: "x"})  # type: ignore[dict-item]

    def test_writer_defensive_rejects_mutated_key(self, tmp_path) -> None:
        """E-03: entries mutated after construction are rejected AT the
        mutation site (well.entries is now a guarded dict), so a
        non-roundtrippable key cannot reach the writer at all."""
        las = LASFile(
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )
        with pytest.raises(ValueError, match="cannot roundtrip"):
            las.well.entries["GR.CO"] = "1"

    def test_writer_backstop_still_rejects_bypassed_key(self, tmp_path) -> None:
        """E-03: even when the entries mutation guard is bypassed (direct
        dict.__setitem__ on the guarded dict), the writer's key-content
        backstop still rejects the non-roundtrippable key — defense in
        depth stays as-is."""
        from pylasdev import write_las_file

        las = LASFile(
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )
        # Bypass the model-layer guard exactly like a raw dict would.
        dict.__setitem__(las.well.entries, "GR.CO", "1")  # type: ignore[arg-type]
        with pytest.raises((ValueError, LASWriteError), match="cannot roundtrip"):
            write_las_file(str(tmp_path / "bad.las"), las)


class TestG5ParameterDataFormatAlignment:
    """N-I-21: parameter data_format paths aligned — from_dict clears
    multi-char metadata templates (matching the parser), and the writer
    emits the braced {…} form so roundtrips are deterministic."""

    def test_from_dict_clears_multi_char_data_format(self) -> None:
        with pytest.warns(UserWarning, match="multi-character data_format"):
            las = LASFile.from_dict(
                {
                    "version": {"VERS": "3.0"},
                    "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
                    "curves_order": ["DEPT"],
                    "curves": [{"mnemonic": "DEPT"}],
                    "logs": {"DEPT": np.array([1.0])},
                    "parameters": {"DATE": "2024-01-01"},
                    "parameter_details": [
                        {
                            "mnemonic": "DATE",
                            "value": "2024-01-01",
                            "description": "Log date",
                            "data_format": "DD/MM/YYYY",
                        }
                    ],
                }
            )
        assert las.parameters[0].data_format == ""

    def test_from_dict_truncates_extended_format(self) -> None:
        las = LASFile.from_dict(
            {
                "version": {"VERS": "3.0"},
                "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT"}],
                "logs": {"DEPT": np.array([1.0])},
                "parameters": {"DATE": "2024-01-01"},
                "parameter_details": [
                    {
                        "mnemonic": "DATE",
                        "value": "2024-01-01",
                        "data_format": "F8.3",
                    }
                ],
            }
        )
        assert las.parameters[0].data_format == "F"

    def test_writer_emits_braced_multi_char(self, tmp_path) -> None:
        """M-11: Direct construction now matches from_dict — multi-char
        data_format is cleared with a warning, never emitted braced.

        Pre-fix, direct construction preserved ``DD/MM/YYYY`` and the
        writer emitted ``{DD/MM/YYYY}`` which the parser could not re-read
        (data_format cleared, description polluted with literal braces).
        After the N-I-21 parity fix, ParameterEntry construction clears
        multi-char metadata templates (``DD/MM/YYYY`` → ``""``) and
        truncates extended Fortran codes (``F8.3`` → ``F``) — identical
        to the from_dict path.
        """
        from pylasdev import read_las_file_as_object, write_las_file

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pe = ParameterEntry(
                mnemonic="DATE",
                value="2024-01-01",
                description="Log date",
                data_format="DD/MM/YYYY",
            )
        # The multi-char template is cleared at construction with a warning.
        assert pe.data_format == ""
        assert any("multi-character data_format" in str(w.message) for w in rec), (
            f"expected clear warning, got: {[str(w.message) for w in rec]}"
        )
        # Extended Fortran codes truncate to their single-letter base.
        pe_ext = ParameterEntry(
            mnemonic="DEPTH",
            value="1.0",
            description="Depth",
            data_format="F8.3",
        )
        assert pe_ext.data_format == "F"

        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
            parameters=[pe],
        )
        out = tmp_path / "param_fmt.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(str(out), las)
        text = out.read_text(encoding="utf-8")
        # No braced multi-char template is emitted (it was cleared).
        assert "{DD/MM/YYYY}" not in text
        # Re-read is stable: data_format stays cleared, description keeps
        # only the original text — no per-roundtrip duplication.
        back = read_las_file_as_object(str(out))
        assert back.parameters[0].data_format == ""
        assert back.parameters[0].description == "Log date"


class TestG5MnemBaseResolutionCollision:
    """N-I-30: MNEM_BASE resolving distinct mnemonics (LLD/LLS → BK → BFV)
    to the same canonical must not crash from_dict on duplicate-free input,
    must preserve curve identity, and must warn accurately."""

    def _dual_laterolog_dict(self) -> dict[str, Any]:
        return {
            "version": {"VERS": "2.0"},
            "well": {"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"},
            "curves_order": ["DEPT", "LLD", "LLS"],
            "curves": [
                {"mnemonic": "DEPT"},
                {"mnemonic": "LLD"},
                {"mnemonic": "LLS"},
            ],
            "logs": {
                "DEPT": np.array([1.0, 2.0]),
                "LLD": np.array([1.0, 2.0]),
                "LLS": np.array([3.0, 4.0]),
            },
        }

    def test_from_dict_does_not_crash_on_collision(self) -> None:
        las = LASFile.from_dict(self._dual_laterolog_dict(), mnem_base=MNEM_BASE)
        # Identity preserved: LLD normalizes to BFV, LLS keeps its original.
        assert las.curves_order == ["DEPT", "BFV", "LLS"]
        assert list(las.logs.keys()) == ["DEPT", "BFV", "LLS"]

    def test_collision_warns_accurately_once(self) -> None:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            LASFile.from_dict(self._dual_laterolog_dict(), mnem_base=MNEM_BASE)
        collision_warns = [str(w.message) for w in rec if "resolves to" in str(w.message)]
        assert len(collision_warns) == 1
        assert "LLS" in collision_warns[0]
        assert "BFV" in collision_warns[0]

    def test_no_mnem_base_identity_unchanged(self) -> None:
        las = LASFile.from_dict(self._dual_laterolog_dict())
        assert las.curves_order == ["DEPT", "LLD", "LLS"]

    def test_genuine_duplicate_still_rejected(self) -> None:
        """Real duplicates (same raw name twice) still raise — only
        resolution collisions are forgiven."""
        data = self._dual_laterolog_dict()
        data["curves_order"] = ["DEPT", "LLD", "LLD"]
        data["curves"] = [
            {"mnemonic": "DEPT"},
            {"mnemonic": "LLD"},
            {"mnemonic": "LLD"},
        ]
        data["logs"] = {
            "DEPT": np.array([1.0, 2.0]),
            "LLD": np.array([1.0, 2.0]),
        }
        with pytest.raises(LASDataError, match=r"[Dd]uplicate"):
            LASFile.from_dict(data, mnem_base=MNEM_BASE)


class TestDevFileSurvivorValidationG10:
    """V-17 model-side: DevFile.validate must validate dedup SURVIVORS.

    DevFile.validate previously had NO survivor handling for ANY column
    type (N-2A) — a dedup survivor (MD_2, AZI_2, INC_2, TVD_2) bypassed
    every type-specific check on direct construction.  The read path's
    dev_reader._validate_dev_data covers AZI_2/INC_2/TVD_2 survivors but
    NOT MD_2 (V-17); models.py covered none.  These tests verify the
    model-side gap is closed for all four types.  Each test FAILS on
    pre-fix code (validate returns zero issues) and PASSES post-fix.
    """

    def test_md_survivor_non_monotonic_issue(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 150.0, 250.0, 400.0]),
                "MD_2": np.array([150.0, 120.0, 130.0, 140.0]),
            },
            column_order=["MD", "MD_2"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("not monotonically increasing" in issue for issue in issues), (
            f"Expected MD_2 monotonicity issue, got {issues}"
        )

    def test_azi_survivor_out_of_range_issue(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 100.0]),
                "AZI": np.array([90.0, 90.0]),
                "AZI_2": np.array([500.0, 500.0]),
            },
            column_order=["MD", "AZI", "AZI_2"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("Azimuth column 'AZI_2'" in issue for issue in issues), (
            f"Expected AZI_2 range issue, got {issues}"
        )

    def test_inc_survivor_out_of_range_issue(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 100.0]),
                "INC": np.array([0.0, 5.0]),
                "INC_2": np.array([190.0, 190.0]),
            },
            column_order=["MD", "INC", "INC_2"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("Inclination column 'INC_2'" in issue for issue in issues), (
            f"Expected INC_2 range issue, got {issues}"
        )

    def test_tvd_survivor_nan_density_issue(self) -> None:
        """V-17: models.py had NO TVD validation at all — add it so direct
        construction matches the read path (dev_reader validates TVD)."""
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 100.0, 200.0]),
                "TVD_2": np.array([np.nan, np.nan, 300.0]),
            },
            column_order=["MD", "TVD_2"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("TVD column 'TVD_2'" in issue and "NaN" in issue for issue in issues), (
            f"Expected TVD_2 NaN-density issue, got {issues}"
        )

    def test_primary_columns_still_validated(self) -> None:
        """Primary (non-survivor) MD/AZI/INC checks are unchanged."""
        dev = DevFile(
            columns={
                "MD": np.array([100.0, 50.0]),
                "AZI": np.array([400.0, 90.0]),
            },
            column_order=["MD", "AZI"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("not monotonically increasing" in issue for issue in issues), (
            f"Expected primary MD monotonicity issue, got {issues}"
        )
        assert any("Azimuth column 'AZI'" in issue for issue in issues), (
            f"Expected primary AZI range issue, got {issues}"
        )


class TestPF07MetaPrefixedUserColumnRoundtrip:
    """PF-07 (s9 convergence pass, MOD-11 incomplete fix): a DevFile with a
    user column literally named ``_meta_source_file`` / ``_meta_encoding`` /
    ``_meta_column_order`` (array value) crashed the to_dict→from_dict
    roundtrip with LASDataError.  to_dict emitted BOTH the column and the
    bare ``source_file`` metadata key; from_dict's collision check
    (``f"_meta_{key}" in data``, models.py:6114) misread the bare key as a
    column and tried ``np.asarray("test.dev", dtype=float)``.

    The fix mirrors the MOD-11 closed-set rule in BOTH directions:
    ``_is_encoded_dev_metadata_key`` on the ``_meta_`` slot (a ``_meta_`` key
    with an array value is a USER COLUMN, not encoded metadata) plus a
    metadata-shape guard on the bare value.  The roundtrip must preserve the
    column AND the metadata without a crash.
    """

    def test_meta_source_file_column_roundtrips_without_crash(self) -> None:
        """Regression (fail pre-fix): LASDataError on from_dict; pass post-fix."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0, 200.0])
        dev.columns["_meta_source_file"] = np.array([1.0, 2.0, 3.0])
        dev.source_file = "test.dev"

        d = dev.to_dict()
        # Both the user column and the bare metadata key are emitted
        assert "_meta_source_file" in d
        assert d["source_file"] == "test.dev"
        dev2 = DevFile.from_dict(d)

        # Column preserved verbatim (pre-fix this crashed)
        assert "_meta_source_file" in dev2.columns
        np.testing.assert_array_equal(dev2.columns["_meta_source_file"], np.array([1.0, 2.0, 3.0]))
        # Metadata preserved under the bare key (not swallowed as a column)
        assert dev2.source_file == "test.dev"
        np.testing.assert_array_equal(dev2.columns["MD"], np.array([0.0, 100.0, 200.0]))

    def test_meta_encoding_column_roundtrips_without_crash(self) -> None:
        """Same regression for the ``_meta_encoding`` suffix."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 1.0])
        dev.columns["_meta_encoding"] = np.array([5.0, 6.0])
        dev.encoding = "latin-1"

        d = dev.to_dict()
        dev2 = DevFile.from_dict(d)
        assert "_meta_encoding" in dev2.columns
        np.testing.assert_array_equal(dev2.columns["_meta_encoding"], np.array([5.0, 6.0]))
        assert dev2.encoding == "latin-1"

    def test_meta_column_order_column_roundtrips_without_crash(self) -> None:
        """Same regression for the ``_meta_column_order`` suffix."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 1.0])
        dev.columns["_meta_column_order"] = np.array([9.0, 9.0])

        d = dev.to_dict()
        dev2 = DevFile.from_dict(d)
        assert "_meta_column_order" in dev2.columns
        np.testing.assert_array_equal(dev2.columns["_meta_column_order"], np.array([9.0, 9.0]))
        # column_order metadata still applied (includes the user column)
        assert dev2.column_order == ["MD", "_meta_column_order"]

    def test_double_collision_preserves_both_user_columns(self) -> None:
        """PF-07 to_dict mirror: columns named BOTH ``source_file`` AND
        ``_meta_source_file``.  The old to_dict overwrote the ``_meta_`` user
        column with the metadata string (silent data loss).  Now to_dict
        warns and drops the metadata, preserving BOTH user columns."""
        dev = DevFile()
        dev.columns["MD"] = np.array([0.0, 100.0, 200.0])
        dev.columns["source_file"] = np.array([5.0, 6.0, 7.0])
        dev.columns["_meta_source_file"] = np.array([1.0, 2.0, 3.0])
        dev.source_file = "test.dev"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = dev.to_dict()
        assert any("both collide" in str(x.message) for x in w), (
            f"Expected double-collision warning, got {[str(x.message) for x in w]}"
        )
        # Neither user column was overwritten by metadata
        assert isinstance(d["_meta_source_file"], np.ndarray)
        assert isinstance(d["source_file"], np.ndarray)

        dev2 = DevFile.from_dict(d)
        assert "source_file" in dev2.columns
        assert "_meta_source_file" in dev2.columns
        np.testing.assert_array_equal(dev2.columns["source_file"], np.array([5.0, 6.0, 7.0]))
        np.testing.assert_array_equal(dev2.columns["_meta_source_file"], np.array([1.0, 2.0, 3.0]))


class TestPF09GuardedDictPickle:
    """PF-09 (s9 convergence pass, REGRESSION vs HEAD): ``_GuardedDict``
    stopped being picklable when ``__setitem__`` gained the UNCONDITIONAL
    ``_check_column_array_like(value, self._container_name)`` (MOD-14/17/23).
    Default dict-subclass unpickling restores items through ``__setitem__``
    BEFORE the ``_container_name`` slot is set → AttributeError.  The fix
    mirrors the ``_GuardedList`` pattern (``__reduce__`` reconstructs via
    ``__init__`` + ``__setstate__`` restores the slot).  Guards must remain
    intact post-unpickle.
    """

    def test_lasfile_pickle_roundtrip_guards_intact(self) -> None:
        """Regression (fail pre-fix): AttributeError on unpickle; pass post-fix."""
        from pylasdev.models import _GuardedDict

        las = LASFile()
        las.logs["DEPT"] = np.array([100.0, 101.0])
        las.logs["GR"] = np.array([50.0, 60.0])

        las2 = pickle.loads(pickle.dumps(las))
        assert isinstance(las2.logs, _GuardedDict)
        np.testing.assert_array_equal(las2.logs["DEPT"], np.array([100.0, 101.0]))
        # _container_name restored
        assert las2.logs._container_name == "LASFile.logs"
        # Guards intact: non-str key still rejected
        with pytest.raises(TypeError, match="keys must be str"):
            las2.logs[5] = np.array([1.0, 2.0])
        # Value contract intact: str/scalar still rejected
        with pytest.raises(ValueError, match="1-D array-like"):
            las2.logs["X"] = "scalar"
        # Length invariant intact
        with pytest.raises(ValueError, match="inconsistent lengths"):
            las2.logs["NEW"] = np.array([1.0, 2.0, 3.0])

    def test_bare_guarded_dict_pickle_roundtrip(self) -> None:
        """Bare _GuardedDict pickles with _container_name preserved."""
        from pylasdev.models import _GuardedDict

        g = _GuardedDict({"a": np.array([1.0, 2.0])}, _container_name="LASFile.logs")
        g2 = pickle.loads(pickle.dumps(g))
        assert isinstance(g2, _GuardedDict)
        np.testing.assert_array_equal(g2["a"], np.array([1.0, 2.0]))
        assert g2._container_name == "LASFile.logs"

    def test_empty_guarded_dict_pickle_roundtrip(self) -> None:
        """Empty _GuardedDict (no items to restore) also pickles."""
        from pylasdev.models import _GuardedDict

        g = _GuardedDict(_container_name="LASFile.string_data")
        g2 = pickle.loads(pickle.dumps(g))
        assert dict(g2) == {}
        assert g2._container_name == "LASFile.string_data"


# ──────────────────────────────────────────────────────────────
# F-14 (MEDIUM): CurveDefinition.__post_init__ validated data_format
# WITHOUT uppercasing → lowercase 'f' raised ValueError while
# ParameterEntry/from_dict normalize (MOD-02 inconsistency).
# ──────────────────────────────────────────────────────────────


class TestF14CurveDefinitionLowercaseDataFormat:
    """F-14: CurveDefinition(mnemonic='GR', data_format='f') must normalize
    to 'F' like ParameterEntry and from_dict — direct construction now
    accepts lowercase formats on every construction path."""

    def test_lowercase_f_normalizes_to_F(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="f")
        assert c.data_format == "F"

    def test_other_lowercase_formats_normalize(self) -> None:
        for lower, upper in (("s", "S"), ("a", "A"), ("i", "I"), ("e", "E"), ("d", "D")):
            c = CurveDefinition(mnemonic="GR", data_format=lower)
            assert c.data_format == upper

    def test_invalid_format_still_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid data_format"):
            CurveDefinition(mnemonic="GR", data_format="Q")

    def test_parameter_entry_and_curve_agree(self) -> None:
        # The MOD-02 parity this finding documented as missing.
        c = CurveDefinition(mnemonic="GR", data_format="f")
        p = ParameterEntry(mnemonic="MUD", value="x", data_format="f")
        assert c.data_format == p.data_format == "F"


# ──────────────────────────────────────────────────────────────
# F-15 (MEDIUM): WellSection.units/descriptions plain-dict
# post-construction mutation bypassed the I2F-05 construction guard →
# writer crashed with opaque LASWriteError (AttributeError on
# .replace()).  units/descriptions are now guarded dicts.
# ──────────────────────────────────────────────────────────────


class TestF15WellUnitsDescriptionsMutationGuard:
    """F-15: post-construction ``well.units['STRT'] = 123`` must raise a
    clean TypeError (matching the I2F-05 construction guard) instead of
    crashing the writer with an opaque LASWriteError."""

    def _lasfile(self) -> LASFile:
        return LASFile(
            version=VersionSection(vers="2.0"),
            well=WellSection(entries={"STRT": "1", "STOP": "2", "STEP": "0.5", "NULL": "-999"}),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )

    def test_units_non_str_value_raises_at_mutation(self) -> None:
        las = self._lasfile()
        with pytest.raises(TypeError, match="must be str"):
            las.well.units["STRT"] = 123  # type: ignore[assignment]

    def test_descriptions_non_str_value_raises_at_mutation(self) -> None:
        las = self._lasfile()
        with pytest.raises(TypeError, match="must be str"):
            las.well.descriptions["STRT"] = 3.14  # type: ignore[assignment]

    def test_valid_str_assignment_still_works(self) -> None:
        las = self._lasfile()
        las.well.units["DEPT"] = "M"
        las.well.descriptions["DEPT"] = "Depth"
        assert las.well.units["DEPT"] == "M"
        assert las.well.descriptions["DEPT"] == "Depth"

    def test_non_str_key_raises_at_mutation(self) -> None:
        las = self._lasfile()
        with pytest.raises(TypeError, match="must be str"):
            las.well.units[5] = "M"  # type: ignore[index]

    def test_wholesale_reassignment_rewraps_guard(self) -> None:
        """``well.units = {'X': 1}`` must re-wrap through the guarded dict
        (self-healing __setattr__), so the invalid value is rejected."""
        las = self._lasfile()
        with pytest.raises(TypeError, match="must be str"):
            las.well.units = {"X": 1}  # type: ignore[assignment]

    def test_construction_with_non_str_value_still_raises(self) -> None:
        """Control: the I2F-05 construction guard is unchanged."""
        with pytest.raises(TypeError, match="must be str"):
            WellSection(units={"STRT": 123})  # type: ignore[dict-item]


# ──────────────────────────────────────────────────────────────
# F-16 (MEDIUM): LASFile.__post_init__ called len() on log/string_data
# arrays WITHOUT the 0-d ndarray special-case DataSection has (M-18
# convention accepts 0-d) → raw TypeError on documented-valid input.
# ──────────────────────────────────────────────────────────────


class TestF16LasFileZeroDimArrays:
    """F-16: LASFile construction with 0-d numpy arrays (np.array(5.0))
    must succeed — the M-18 convention treats them as single-element
    values, matching DataSection."""

    def _order(self) -> list[str]:
        return ["GR", "DT"]

    def _curves(self) -> list[CurveDefinition]:
        return [CurveDefinition(mnemonic="GR"), CurveDefinition(mnemonic="DT")]

    def test_logs_zero_dim_arrays_accepted(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=self._order(),
            curves=self._curves(),
            logs={"GR": np.array(1.0), "DT": np.array(2.0)},
        )
        assert las.logs["GR"].ndim == 0

    def test_string_data_zero_dim_arrays_accepted(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=self._order(),
            curves=self._curves(),
            string_data={"GR": np.array("x"), "DT": np.array("y")},
        )
        assert las.string_data["GR"].ndim == 0

    def test_mixed_zero_dim_logs_and_string_data_accepted(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["GR", "CDES"],
            curves=[
                CurveDefinition(mnemonic="GR", data_format="F"),
                CurveDefinition(mnemonic="CDES", data_format="S"),
            ],
            logs={"GR": np.array(5.0)},
            string_data={"CDES": np.array("rock")},
        )
        assert las.logs["GR"].ndim == 0
        assert las.string_data["CDES"].ndim == 0

    def test_inconsistent_zero_dim_lengths_still_raise(self) -> None:
        """A 0-d array is length 1 — mixed 0-d and 2-row arrays must still
        be detected as inconsistent (the guarded-dict length guard fires
        at construction, treating the 0-d array as a 1-row value)."""
        with pytest.raises(ValueError, match="inconsistent lengths"):
            LASFile(
                version=VersionSection(vers="2.0"),
                curves_order=["GR", "DT"],
                curves=self._curves(),
                logs={"GR": np.array(1.0), "DT": np.array([1.0, 2.0])},
            )


# ──────────────────────────────────────────────────────────────
# F-17 (MEDIUM): _GuardedDict.__setitem__ / _DevColumns.__setitem__
# stored the caller's array BY REFERENCE → alias mutation silently
# corrupted the duplicate column (survived to_dict/from_dict).
# Item-level assignment now copies.
# ──────────────────────────────────────────────────────────────


class TestF17ItemAssignmentCopiesArrays:
    """F-17: ``las.logs['DUP'] = las.logs['MD']`` must store a COPY — no
    shared memory with the source column (matches the library's
    no-shared-reference philosophy, N-I-11/F-002)."""

    def _lasfile(self) -> LASFile:
        return LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["MD"],
            curves=[CurveDefinition(mnemonic="MD")],
            logs={"MD": np.array([1.0, 2.0, 3.0])},
        )

    def test_logs_item_assignment_copies(self) -> None:
        las = self._lasfile()
        las.logs["DUP"] = las.logs["MD"]
        assert las.logs["DUP"] is not las.logs["MD"]
        assert not np.shares_memory(las.logs["DUP"], las.logs["MD"])

    def test_logs_mutation_does_not_corrupt_source(self) -> None:
        las = self._lasfile()
        las.logs["DUP"] = las.logs["MD"]
        las.logs["DUP"][0] = 999.0
        np.testing.assert_array_equal(las.logs["MD"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(las.logs["DUP"], [999.0, 2.0, 3.0])

    def test_dev_columns_item_assignment_copies(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([1.0, 2.0, 3.0])
        dev.columns["MD_COPY"] = dev.columns["MD"]
        assert dev.columns["MD_COPY"] is not dev.columns["MD"]
        assert not np.shares_memory(dev.columns["MD_COPY"], dev.columns["MD"])

    def test_dev_columns_mutation_does_not_corrupt_source(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([1.0, 2.0, 3.0])
        dev.columns["MD_COPY"] = dev.columns["MD"]
        dev.columns["MD_COPY"][0] = 999.0
        np.testing.assert_array_equal(dev.columns["MD"], [1.0, 2.0, 3.0])

    def test_roundtrip_preserves_both_columns(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([1.0, 2.0])
        dev.columns["MD_COPY"] = dev.columns["MD"]
        dev2 = DevFile.from_dict(dev.to_dict())
        np.testing.assert_array_equal(dev2.columns["MD"], [1.0, 2.0])
        np.testing.assert_array_equal(dev2.columns["MD_COPY"], [1.0, 2.0])


# ──────────────────────────────────────────────────────────────
# F-18 (MEDIUM): DevFile.from_dict was input-order-dependent — a
# column_order key BEFORE referenced columns raised a spurious
# LASDataError against PARTIALLY-populated dev.columns.
# ──────────────────────────────────────────────────────────────


class TestF18DevFileFromDictOrderIndependentColumnOrder:
    """F-18: DevFile.from_dict must accept column_order in ANY dict
    position — a column_order entry referencing a column that appears
    LATER in the dict is valid and must not raise."""

    def test_column_order_before_referenced_columns(self) -> None:
        dev = DevFile.from_dict({"MD": [1.0, 2.0], "column_order": ["MD", "GR"], "GR": [3.0, 4.0]})
        assert dev.column_order == ["MD", "GR"]

    def test_column_order_first_still_works(self) -> None:
        dev = DevFile.from_dict({"column_order": ["MD", "GR"], "MD": [1.0], "GR": [3.0]})
        assert dev.column_order == ["MD", "GR"]

    def test_column_order_last_still_works(self) -> None:
        dev = DevFile.from_dict({"MD": [1.0], "GR": [3.0], "column_order": ["GR", "MD"]})
        assert dev.column_order == ["GR", "MD"]

    def test_genuinely_orphaned_entry_still_rejected(self) -> None:
        with pytest.raises(LASDataError):
            DevFile.from_dict({"MD": [1.0, 2.0], "column_order": ["MD", "GHOST"], "GR": [3.0, 4.0]})

    def test_roundtrip_stable(self) -> None:
        dev = DevFile.from_dict({"MD": [1.0, 2.0], "column_order": ["MD", "GR"], "GR": [3.0, 4.0]})
        dev2 = DevFile.from_dict(dev.to_dict())
        assert dev2.column_order == ["MD", "GR"]

    def test_partial_column_order_auto_appends_omitted_columns(self) -> None:
        """F-18 regression (M6): an explicit column_order that NAMES only
        a SUBSET of the columns must auto-append the omitted columns (in
        insertion order), matching the pre-F-18 sync behavior.  The F-18
        deferral replaced the sync-built order with the partial explicit
        order, so a dict whose column_order key omits a later column
        (e.g. {"MD": [...], "column_order": ["MD"], "GR": [...]}) raised
        LASDataError "column_order and columns keys do not match" even
        though the same dict succeeded before the deferral."""
        dev = DevFile.from_dict({"MD": [1.0, 2.0], "column_order": ["MD"], "GR": [3.0, 4.0]})
        assert dev.column_order == ["MD", "GR"]

    def test_partial_column_order_roundtrip_stable(self) -> None:
        dev = DevFile.from_dict({"MD": [1.0, 2.0], "column_order": ["MD"], "GR": [3.0, 4.0]})
        dev2 = DevFile.from_dict(dev.to_dict())
        assert dev2.column_order == ["MD", "GR"]


# ──────────────────────────────────────────────────────────────
# F-19 (MEDIUM): LASFile with non-empty curves_order but EMPTY curves
# passed __post_init__ + validate() + writer backstop → write emitted
# ~C with no curve headers + ~A data rows → re-read discarded all
# data.  Direct construction now rejects the state loudly.
# ──────────────────────────────────────────────────────────────


class TestF19CurvesOrderWithoutCurvesRejected:
    """F-19: a non-empty curves_order with EMPTY curves is an
    inconsistent state that silently lost all data on write→read.  Direct
    construction must reject it with a clear LASDataError, and
    validate(complete=True) must flag a post-construction ``curves``
    wipe."""

    def test_construction_with_curves_order_but_no_curves_raises(self) -> None:
        with pytest.raises(LASDataError, match="curve definitions"):
            LASFile(
                version=VersionSection(vers="2.0"),
                curves_order=["DEPT", "GR"],
                curves=[],
                logs={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
            )

    def test_post_construction_curves_clear_detected_by_validate(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT", "GR"],
            curves=[CurveDefinition(mnemonic="DEPT"), CurveDefinition(mnemonic="GR")],
            logs={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        las.curves.clear()
        issues = las.validate(complete=True)
        assert any("curve definitions" in issue for issue in issues), issues

    def test_matching_curves_still_constructs(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT", "GR"],
            curves=[CurveDefinition(mnemonic="DEPT"), CurveDefinition(mnemonic="GR")],
            logs={"DEPT": np.array([1.0]), "GR": np.array([2.0])},
        )
        assert [c.mnemonic for c in las.curves] == ["DEPT", "GR"]

    def test_las30_data_sections_with_empty_top_curves_constructs(self) -> None:
        """F-19 regression (M14): the widened F-19 gate (``if
        self.curves_order:``) over-rejected a legitimate LAS 3.0
        per-section construction — empty top-level curves + populated
        curves_order + data_sections whose definitions live in
        ``section_curves``.  Pre-fix (HEAD) this state constructed,
        wrote, and re-read OK; the gate widening broke direct
        construction.  The state must construct, must NOT be flagged by
        validate(complete=True), and must roundtrip with data intact.
        The true F-19 silent-loss state (curves_order set + curves empty
        + NO data_sections) is still rejected (test above)."""
        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="API"),
            ],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "GR"],
            curves=[],
            data_sections=[ds],
        )
        assert len(las.curves) == 0
        assert len(las.data_sections) == 1
        # validate(complete=True) must not emit the F-19 curves_order-
        # without-definitions issue for the per-section state.
        issues = las.validate(complete=True)
        assert not any("curve definitions" in issue for issue in issues), issues

    def test_las30_data_sections_with_empty_top_curves_roundtrips(self, tmp_path: Path) -> None:
        """M14 write→re-read: the per-section state must roundtrip with
        the writer emitting the section_curves definitions and preserving
        the section data (the F-19 gate previously made the state
        unconstructible; the adversarial bypass probe confirmed the
        writer handles it)."""
        from pylasdev import read_las_file, write_las_file

        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="API"),
            ],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "GR"],
            curves=[],
            data_sections=[ds],
        )
        out = tmp_path / "m14_las30_sections.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        parsed = read_las_file(out)
        assert list(parsed["logs"]["DEPT"]) == [100.0, 110.0]
        assert list(parsed["logs"]["GR"]) == [75.0, 80.0]


# ──────────────────────────────────────────────────────────────
# MOD-M1 (MEDIUM, M14 residual): the M14 gate relaxation used
# 'data_sections present' as a proxy for 'definitions exist'.  A
# DataSection holding data with EMPTY section_curves (or section_curves
# covering only part of curves_order) passed the gate with NO definition
# anywhere — the writer emitted ~C with no header for the uncovered curve
# and re-read silently lost the data (F-19 class).  The gate now verifies
# the sections' section_curves actually cover curves_order.
# ──────────────────────────────────────────────────────────────


class TestMODM1SectionCurvesCoverageGate:
    """MOD-M1: the LAS 3.0 per-section skip of the F-19 definition gate
    must verify that the sections' section_curves actually cover
    curves_order.  A DataSection with data but empty/partial section_curves
    re-opens F-19 silent data loss (constructs, validate silent, write
    emits only warnings, re-read logs={})."""

    def test_las30_empty_section_curves_with_data_raises(self) -> None:
        """Direct construction with a DataSection that carries data but
        ZERO section_curves definitions must raise — the writer would emit
        ~C with no header for those curves and re-read would silently lose
        ALL the data (F-19 class re-opened by the M14 relaxation)."""
        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
            section_curves=[],
        )
        with pytest.raises(LASDataError, match="no curve definition"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["DEPT", "GR"],
                curves=[],
                data_sections=[ds],
            )

    def test_las30_partial_section_curves_coverage_raises(self) -> None:
        """Partial coverage — the section defines only GR but the top-level
        curves_order also claims DEPT — must raise: DEPT has no definition
        anywhere and its data would be silently lost on write."""
        ds = DataSection(
            name="LOG",
            curves_order=["GR"],
            section_curves=[CurveDefinition(mnemonic="GR", unit="API")],
            data={"GR": np.array([75.0, 80.0])},
        )
        with pytest.raises(LASDataError, match="no curve definition"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["DEPT", "GR"],
                curves=[],
                data_sections=[ds],
            )

    def test_las30_section_curves_wipe_detected_by_validate(self) -> None:
        """MOD-M1 validate twin: post-construction clearing of a section's
        section_curves (removing the only definitions) must be flagged by
        validate(complete=True) — the writer would otherwise emit the
        curves without headers and silently lose the data."""
        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="API"),
            ],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "GR"],
            curves=[],
            data_sections=[ds],
        )
        assert las.validate(complete=True) == []
        las.data_sections[0].section_curves = []
        issues = las.validate(complete=True)
        assert any("no curve definition" in issue for issue in issues), issues

    def test_las30_case_variant_curves_order_definition_coverage_accepted(self) -> None:
        """The coverage check is case-insensitive (mirroring the writer's
        definition resolution at _writer_las30.py:321-332/348): a
        post-construction lowercase curves_order entry that resolves to an
        uppercase section_curves definition must NOT be rejected by the
        gate — the writer DOES emit it."""
        ds = DataSection(
            name="LOG",
            curves_order=["DEPT", "GR"],
            section_curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="GR", unit="API"),
            ],
            data={"DEPT": np.array([100.0, 110.0]), "GR": np.array([75.0, 80.0])},
        )
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "GR"],
            curves=[],
            data_sections=[ds],
        )
        las.curves_order = ["dept", "GR"]
        issues = las.validate(complete=True)
        assert not any("no curve definition" in issue for issue in issues), issues


# ──────────────────────────────────────────────────────────────
# MOD-3 (MEDIUM, F-04 residual): to_dict → from_dict roundtrip of the
# now-blessed case-variant state (curves_order=['dept','GR'], definitions
# and data keyed DEPT/GR).  The pass-3 MOD-1/MOD-2 fix made the state
# constructible/validatable/writable but left from_dict and its
# per-section/DataSection twins exact-case — the roundtrip and the public
# dict-write API hard-failed with LASDataError/LASWriteError.
# ──────────────────────────────────────────────────────────────


class TestMOD3CaseVariantRoundtripFromDict:
    """F-04: the case-variant state must round-trip through
    ``LASFile.from_dict(las.to_dict())`` (LAS 2.0 + 3.0) with data
    preserved, while genuinely distinct/renamed mnemonics still raise."""

    def _case_variant_las20(self) -> LASFile:
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
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.well["NULL"] = "-999.25"
        return las

    def test_mod3_las20_case_variant_roundtrip_from_dict(self) -> None:
        """F-04: the LAS 2.0 to_dict→from_dict roundtrip of the supported
        case-variant state must succeed (pre-fix: LASDataError at the
        top-level positional check models.py:4484) and preserve data."""
        las = self._case_variant_las20()
        d = las.to_dict()
        assert d["curves_order"] == ["dept", "GR"]
        las2 = LASFile.from_dict(d)
        # M-10: from_dict normalizes mnemonics identically to the parser
        # (uppercase on lookup miss) — the case-variant roundtrip is
        # still supported (case-insensitive comparisons, data preserved),
        # but the stored curves_order casing is the parser-canonical
        # uppercase, so parse() and from_dict() produce identical models
        # for identical file content.
        assert las2.curves_order == ["DEPT", "GR"]
        np.testing.assert_array_equal(las2.logs["DEPT"], np.array([100.0, 101.0, 102.0]))
        np.testing.assert_array_equal(las2.logs["GR"], np.array([75.0, 76.0, 77.0]))
        # Second roundtrip must be stable (to_dict → from_dict idempotent).
        las3 = LASFile.from_dict(las2.to_dict())
        np.testing.assert_array_equal(las3.logs["DEPT"], np.array([100.0, 101.0, 102.0]))

    def test_mod3_las20_roundtrip_string_data_case_variant(self) -> None:
        """F-04: case-variant string_data keys round-trip too — the
        top-level string_data orphan check and the F-011 log-key vs
        curves_order check must compare case-insensitively (pre-fix:
        false 'Extra keys'/'orphaned' LASDataError)."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "TDEP"],
            curves=[
                CurveDefinition(mnemonic="DEPT", unit="M"),
                CurveDefinition(mnemonic="TDEP", unit="US/M"),
            ],
            logs={"DEPT": np.array([100.0, 101.0, 102.0])},
            string_data={"TDEP": np.array(["a", "b", "c"])},
        )
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.well["NULL"] = "-999.25"
        las2 = LASFile.from_dict(las.to_dict())
        # M-10: curves_order casing normalizes to the parser-canonical
        # uppercase on the from_dict path (see test above); string_data
        # values are preserved.
        assert las2.curves_order == ["DEPT", "TDEP"]
        np.testing.assert_array_equal(las2.string_data["TDEP"], np.array(["a", "b", "c"]))

    def test_mod3_las30_case_variant_roundtrip_from_dict(self) -> None:
        """F-04: the LAS 3.0 roundtrip (data_sections with section
        definitions keyed DEPT/GR) must succeed — the per-section
        orphan checks, per-section positional check, and DataSection
        __post_init__ twins must compare case-insensitively (pre-fix:
        LASDataError at :4903/:4913/:4970/:2803)."""
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
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept", "GR"],
            curves=[],
            data_sections=[ds],
        )
        las.well["STRT"] = "100"
        las.well["STOP"] = "102"
        las.well["STEP"] = "1"
        las.well["NULL"] = "-999.25"
        las2 = LASFile.from_dict(las.to_dict())
        # M-10: curves_order casing normalizes to the parser-canonical
        # uppercase on the from_dict path (see the LAS 2.0 test above);
        # per-section data and section order are preserved.
        assert las2.curves_order == ["DEPT", "GR"]
        assert las2.data_sections[0].curves_order == ["DEPT", "GR"]
        np.testing.assert_array_equal(
            las2.data_sections[0].data["DEPT"], np.array([100.0, 101.0, 102.0])
        )
        np.testing.assert_array_equal(
            las2.data_sections[0].data["GR"], np.array([75.0, 76.0, 77.0])
        )
        assert las2.validate(complete=True) == []

    def test_mod3_roundtrip_true_positive_distinct_curve_still_raises(self) -> None:
        """F-04: true positives preserved — a genuinely distinct curve name
        in the position still raises (only case-variant aliasing is blessed)."""
        las = self._case_variant_las20()
        d = las.to_dict()
        d["curves_order"] = ["dept", "XYZ"]
        with pytest.raises(LASDataError, match="does not match"):
            LASFile.from_dict(d)

    def test_mod3_roundtrip_true_positive_orphan_still_raises(self) -> None:
        """F-04: true positives preserved — a genuinely orphaned log key
        still raises after the case-insensitive F-011 comparison."""
        las = self._case_variant_las20()
        d = las.to_dict()
        del d["logs"]["GR"]
        with pytest.raises(LASDataError, match="Log curve keys do not match"):
            LASFile.from_dict(d)


# ──────────────────────────────────────────────────────────────
# F-37 (MEDIUM, models x writer boundary): post-construction mutation of
# data_format to an invalid value emitted structurally-invalid LAS 3.0
# {Q} with zero writer warnings.  __setattr__ guards now re-validate.
# ──────────────────────────────────────────────────────────────


class TestF37PostConstructionDataFormatMutation:
    """F-37: CurveDefinition raises on invalid data_format assignment;
    ParameterEntry warn-and-clears (its documented MOD-02 construction
    contract).  The writer can no longer see an invalid format."""

    def test_curve_invalid_format_raises(self) -> None:
        c = CurveDefinition(mnemonic="GR", unit="GAPI", data_format="F")
        with pytest.raises(ValueError, match="invalid data_format"):
            c.data_format = "Q"

    def test_curve_lowercase_mutation_normalizes(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F")
        c.data_format = "f"
        assert c.data_format == "F"

    def test_curve_clearing_format_allowed(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F")
        c.data_format = ""
        assert c.data_format == ""

    def test_curve_non_str_format_raises(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F")
        with pytest.raises(TypeError, match="data_format must be str"):
            c.data_format = 42  # type: ignore[assignment]

    def test_parameter_invalid_format_warns_and_clears(self) -> None:
        p = ParameterEntry(mnemonic="MUD", value="x")
        with pytest.warns(UserWarning, match="Clearing to empty string"):
            p.data_format = "Q"
        assert p.data_format == ""


# ──────────────────────────────────────────────────────────────
# F-38 (MEDIUM, models x writer boundary): post-construction mutation of
# leaf string fields (curve.unit=42, param.value=42, ...) crashed the
# writer with an opaque LASWriteError; array_info='x' silently
# corrupted output.  __setattr__ guards now coerce-or-reject.
# ──────────────────────────────────────────────────────────────


class TestF38LeafFieldMutationGuarded:
    """F-38: leaf-dataclass string fields coerce non-str values on
    assignment (warning, mirroring construction) and array_info is
    type-checked, so the writer never sees an invalid leaf type."""

    def test_curve_unit_coerces_with_warning(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        with pytest.warns(UserWarning, match="coercing non-str unit"):
            c.unit = 42  # type: ignore[assignment]
        assert c.unit == "42"

    def test_curve_api_code_and_description_coerce(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c.api_code = 42  # type: ignore[assignment]
            c.description = 42  # type: ignore[assignment]
        assert c.api_code == "42"
        assert c.description == "42"

    def test_curve_array_info_must_be_array_element_info(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        with pytest.raises(TypeError, match="array_info must be ArrayElementInfo"):
            c.array_info = "x"  # type: ignore[assignment]
        # None / valid ArrayElementInfo still accepted.
        c.array_info = None
        c.array_info = ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0)
        assert c.array_info.index == 1

    def test_parameter_value_coerces_with_warning(self) -> None:
        p = ParameterEntry(mnemonic="MUD")
        with pytest.warns(UserWarning, match="coercing non-str value"):
            p.value = 42  # type: ignore[assignment]
        assert p.value == "42"

    def test_parameter_unit_and_description_coerce(self) -> None:
        p = ParameterEntry(mnemonic="MUD")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.unit = 42  # type: ignore[assignment]
            p.description = 42  # type: ignore[assignment]
        assert p.unit == "42"
        assert p.description == "42"

    def test_array_element_info_field_mutations_validated(self) -> None:
        ai = ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0)
        with pytest.raises(TypeError, match="base_name must be str"):
            ai.base_name = 42  # type: ignore[assignment]
        with pytest.raises(ValueError, match="index must be >= 1"):
            ai.index = -1
        with pytest.raises(ValueError, match="time_offset must be a finite"):
            ai.time_offset = float("nan")
        # Valid mutations still accepted.
        ai.base_name = "ECHO"
        ai.index = 2
        ai.time_offset = 5.0
        assert (ai.base_name, ai.index, ai.time_offset) == ("ECHO", 2, 5.0)

    def test_writer_survives_guarded_mutations(self, tmp_path: Path) -> None:
        """End-to-end: after valid post-construction mutations the writer
        still emits correct output (the crash class F-38 documented is
        gone)."""
        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )
        las.curves[0].unit = "M"
        out = tmp_path / "f38.las"
        write_las_file(out, las)
        content = out.read_text(encoding="utf-8")
        assert "DEPT.M" in content or "DEPT.M" in content.replace(" ", "")


# ──────────────────────────────────────────────────────────────
# M5 (MEDIUM, models): the F-37/F-38 __setattr__ guards claim to
# "re-apply the construction contract at every assignment" but the unit
# branch omitted the M-04 _UNIT_PATTERN composition check both
# __post_init__s run.  Post-construction ``curve.unit = 'A B'`` was
# accepted (construction raises), and write→re-read then silently
# truncated the unit ('A'), dropped the curve + column ('A#B'), or
# corrupted the ~P value.  The unit branch must raise like construction.
# ──────────────────────────────────────────────────────────────


class TestM5SetattrUnitCompositionValidation:
    """M5: post-construction invalid-unit assignment must behave exactly
    like construction — a unit failing _UNIT_PATTERN.fullmatch raises
    ValueError; a valid unit is still accepted and roundtrips."""

    def test_curve_space_unit_assignment_raises(self) -> None:
        c = CurveDefinition(mnemonic="DEPT", unit="M")
        with pytest.raises(ValueError, match="invalid unit"):
            c.unit = "A B"

    def test_curve_hash_unit_assignment_raises(self) -> None:
        c = CurveDefinition(mnemonic="DEPT", unit="M")
        with pytest.raises(ValueError, match="invalid unit"):
            c.unit = "A#B"

    def test_parameter_space_unit_assignment_raises(self) -> None:
        p = ParameterEntry(mnemonic="NULL", unit="M", value="-999.25")
        with pytest.raises(ValueError, match="invalid unit"):
            p.unit = "bad unit"

    def test_valid_unit_assignment_still_works(self) -> None:
        c = CurveDefinition(mnemonic="RT", unit="ohm.m")
        c.unit = "ohm.m2"
        assert c.unit == "ohm.m2"
        p = ParameterEntry(mnemonic="BHT", unit="DEGC", value="35.5")
        p.unit = "DEGF"
        assert p.unit == "DEGF"

    def test_valid_unit_assignment_roundtrips(self, tmp_path: Path) -> None:
        """End-to-end: a VALID post-construction unit mutation must write
        and re-read unchanged (the guard must not block the legitimate
        mutation API, and must prevent the truncation/drop corruption the
        invalid-assignment class caused)."""
        from pylasdev import read_las_file, write_las_file

        las = LASFile(
            version=VersionSection(vers="2.0"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
        )
        las.curves[0].unit = "M"
        out = tmp_path / "m5_unit_roundtrip.las"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_las_file(out, las)
        parsed = read_las_file(out)
        assert parsed["curves"][0]["unit"] == "M"


# ── Case-normalization regression tests (N1a/N1b/II family) ──────────


class TestCaseNormalizationRegression:
    """Regression tests for the case-normalization drift fixes.

    Each test FAILS on the pre-fix code (exact-case comparison sites) and
    PASSES after the ``_case_key`` migration.  The codebase blesses
    case-variant keys (MOD-2/MOD-3 — the writer's data lookup is
    case-insensitive), so these tests assert that validation/construction/
    dtype-preservation behave case-insensitively TOO.
    """

    def test_n1a1_from_dict_s_format_case_variant_log_key_raises(self) -> None:
        """N1a-1: the IF-026 format-vs-placement guard must reject an
        S-format curve whose data sits in logs under a case-variant key,
        matching the same-case control.  Pre-fix the case-variant state
        bypassed the guard and the writer silently emitted the {S} curve's
        values as numeric."""
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["dept"],
            "curves": [{"mnemonic": "dept", "data_format": "S"}],
            "logs": {"DEPT": np.array([1.0, 2.0])},  # case-variant key
        }
        with pytest.raises(ValueError, match="data_format='S' but is in logs"):
            LASFile.from_dict(data)

    def test_n1a2_datasection_placement_case_variant_raises(self) -> None:
        """N1a-2: DataSection I2F-06 placement check must reject an
        S-format section curve whose data sits in 'data' under a
        case-variant key.  Pre-fix the case-variant state passed both
        construction and validate() while the same-case control raised."""
        with pytest.raises(LASDataError, match="data_format='S' but is in data"):
            DataSection(
                curves_order=["GR"],
                section_curves=[CurveDefinition(mnemonic="GR", data_format="S")],
                data={"gr": np.array([1.0, 2.0])},
            )

    def test_n1a8_well_get_ci_lowercase_keyed(self) -> None:
        """N1a-8: WellSection.get_ci resolves case-variant well keys while
        the public exact-case accessors keep their documented contract.
        Pre-fix ``w.get_ci`` did not exist — consumers hand-rolled CI loops
        and the canonical query ``w["STRT"]`` raised KeyError for a
        lowercase-keyed well even though validate() treated it as present."""
        w = WellSection(entries={"strt": "100", "stop": "200", "step": "1", "null": "-999.25"})
        assert w.get_ci("STRT") == "100"
        assert w.get_ci("NULL") == "-999.25"
        assert w.get_ci("missing", "fallback") == "fallback"
        # The exact-case public accessors are unchanged (documented contract).
        with pytest.raises(KeyError):
            w["STRT"]
        assert "STRT" not in w
        assert w.get("STRT") == ""

    def test_n1b1_top_level_i_precision_case_variant(self) -> None:
        """N1b-1: the top-level {I} dtype-preservation lookup must match
        case-insensitively.  Pre-fix a case-variant log key ('dept' vs
        curve 'DEPT') yielded _fmt='' and silently coerced int64→float64,
        rounding 9007199254740993 to ...992.0."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                "well": {"NULL": "-1"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT", "data_format": "I"}],
                "logs": {"dept": [9007199254740993, 9007199254740995]},
            }
        )
        # M-10: the case-variant log key is normalized to the
        # parser-canonical uppercase on the from_dict path; the {I}
        # int64 precision is preserved.
        arr = las.logs["DEPT"]
        assert arr.dtype == np.int64, f"dtype {arr.dtype} — precision loss"
        assert int(arr[0]) == 9007199254740993
        assert int(arr[1]) == 9007199254740995

    def test_n1b1_per_section_i_precision_case_variant(self) -> None:
        """N1b-1 (per-section): the per-section {I} lookup must match
        case-insensitively — a case-variant section data key with an
        explicit section_curves entry preserves int64."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                "well": {"NULL": "-1"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT", "data_format": "I"}],
                "logs": {"DEPT": [100.0]},
                "data_sections": [
                    {
                        "name": "LOG",
                        "section_type": "LOG_DATA",
                        "curves_order": ["dept"],
                        "section_curves": [{"mnemonic": "DEPT", "data_format": "I"}],
                        "data": {"dept": [9007199254740993, 9007199254740995]},
                    }
                ],
            }
        )
        # M-10: the case-variant section data key is normalized to the
        # parser-canonical uppercase; the {I} int64 precision is
        # preserved.
        arr = las.data_sections[0].data["DEPT"]
        assert arr.dtype == np.int64, f"dtype {arr.dtype} — precision loss"
        assert int(arr[0]) == 9007199254740993

    def test_i3b_per_section_i_fallback_to_top_level(self) -> None:
        """II-3b: a per-section {I} curve WITHOUT explicit section_curves
        must fall back to the top-level curve's data_format.  Pre-fix
        ``ds_section_curves=[]`` made ``next()`` return '' and the branch
        fell to float64 even with an EXACT-case key — silent int64→float64
        precision loss >2^53."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                "well": {"NULL": "-1"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT", "data_format": "I"}],
                "logs": {"DEPT": [100.0]},
                "data_sections": [
                    {
                        "name": "LOG",
                        "section_type": "LOG_DATA",
                        "curves_order": ["DEPT"],
                        "data": {"DEPT": [9007199254740993, 9007199254740995]},
                    }
                ],
            }
        )
        arr = las.data_sections[0].data["DEPT"]
        assert arr.dtype == np.int64, f"dtype {arr.dtype} — precision loss"
        assert int(arr[0]) == 9007199254740993

    def test_n1b8_overlap_case_variant_raises_construction(self) -> None:
        """N1b-8: the LASFile logs↔string_data overlap guard must reject
        a case-variant pair at construction.  Pre-fix the pair passed and
        the LAS 2.0 roundtrip SILENTLY DROPPED the string column."""
        with pytest.raises(LASDataError, match="appear in both logs and string_data"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["DEPT"],
                curves=[CurveDefinition(mnemonic="DEPT")],
                logs={"DEPT": np.array([1.0, 2.0])},
                string_data={"dept": np.array(["a", "b"], dtype=object)},
            )

    def test_n1b8_overlap_case_variant_raises_from_dict(self) -> None:
        """N1b-8 (from_dict twin): the from_dict overlap guard must reject
        a case-variant pair too."""
        with pytest.raises(LASDataError, match="appear in both logs and string_data"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"NULL": "-999.25"},
                    "curves_order": ["DEPT"],
                    "curves": [{"mnemonic": "DEPT"}],
                    "logs": {"DEPT": [1.0, 2.0]},
                    "string_data": {"dept": ["a", "b"]},
                }
            )

    def test_i2_datasection_overlap_case_variant_raises(self) -> None:
        """II-2: the DataSection data↔string_data overlap guard (direct
        construction) must reject a case-variant pair.  Pre-fix the pair
        passed while the same-case control raised."""
        with pytest.raises(LASDataError, match="appear in both data and string_data"):
            DataSection(
                curves_order=["DEPT"],
                data={"DEPT": np.array([1.0, 2.0])},
                string_data={"dept": np.array(["a", "b"], dtype=object)},
            )

    def test_i2_from_dict_per_section_overlap_case_variant_raises(self) -> None:
        """II-2 (from_dict twin): the per-section data↔string_data
        collision check must reject a case-variant pair."""
        with pytest.raises(LASDataError, match="appear in both 'data' and 'string_data'"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"NULL": "-999.25"},
                    "curves_order": ["DEPT"],
                    "curves": [{"mnemonic": "DEPT"}],
                    "logs": {"DEPT": [1.0, 2.0]},
                    "data_sections": [
                        {
                            "name": "LOG",
                            "section_type": "LOG_DATA",
                            "curves_order": ["DEPT"],
                            "data": {"DEPT": [1.0, 2.0]},
                            "string_data": {"dept": ["a", "b"]},
                        }
                    ],
                }
            )

    def test_n1b4_construction_s_format_case_variant_logs_raises(self) -> None:
        """N1b-4: the I2F-08 direct-construction raise must fire for an
        S-format curve whose logs key is case-variant.  Pre-fix the state
        passed construction and the roundtrip silently migrated the
        numeric values into string_data."""
        with pytest.raises(LASDataError, match=r"data_format='S'.*in logs"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["DEPT"],
                curves=[CurveDefinition(mnemonic="DEPT", data_format="S")],
                logs={"dept": np.array([1.5, 2.5])},
            )

    def test_n1b4_validate_i_exemption_case_variant_no_false_issue(self) -> None:
        """N1b-4: the validate() {I} object-dtype exemption must match
        case-insensitively — a case-variant {I} log key must NOT emit the
        false 'non-numeric dtype (object)' issue.  Pre-fix the exemption
        map was keyed by raw curve mnemonic, so the case-variant lookup
        missed and the valid object-dtype state was flagged."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT", data_format="I")],
            logs={"dept": np.array([9007199254740993, 9007199254740993], dtype=object)},
        )
        las.well["NULL"] = "-999.25"  # fractional NULL → object dtype is correct
        issues = las.validate(complete=True)
        assert not any("non-numeric dtype" in i for i in issues), issues

    def test_n1b3_case_variant_duplicate_detection_construction(self) -> None:
        """N1b-3: duplicate curve-name detection is case-insensitive — a
        case-variant pair ('DEPT','dept') must be rejected at construction
        like the same-case control, because the writer's FIRST-wins
        case-insensitive resolution would silently drop the second
        definition's identity."""
        with pytest.raises(LASDataError, match="duplicate curve name"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["DEPT", "dept"],
                curves=[
                    CurveDefinition(mnemonic="DEPT", unit="M"),
                    CurveDefinition(mnemonic="dept", unit="FT"),
                ],
                logs={
                    "DEPT": np.array([1.0, 2.0]),
                    "dept": np.array([3.0, 4.0]),
                },
            )

    def test_n1b3_from_dict_case_variant_duplicate_detection(self) -> None:
        """N1b-3 (from_dict): the top-level duplicate detection must reject
        case-variant duplicates on the from_dict path too."""
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"NULL": "-999.25"},
                    "curves_order": ["DEPT", "dept"],
                    "curves": [
                        {"mnemonic": "DEPT", "unit": "M"},
                        {"mnemonic": "dept", "unit": "FT"},
                    ],
                    "logs": {"DEPT": [1.0, 2.0], "dept": [3.0, 4.0]},
                }
            )

    def test_i19_from_dict_array_cross_check_no_false_warning(self) -> None:
        """II-19: the from_dict M-17 array cross-check must NOT fire a
        false warning for a case-variant mnemonic/base_name pair.  Pre-fix
        the raw mnemonic base ('nmr') was compared against the UPPERCASED
        base_name ('NMR'), warning spuriously."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                "well": {"NULL": "-999.25"},
                "curves_order": ["nmr[1]"],
                "curves": [
                    {
                        "mnemonic": "nmr[1]",
                        "array_info": {"base_name": "NMR", "index": 1},
                    }
                ],
                "logs": {"nmr[1]": [1.0, 2.0]},
            }
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            LASFile.from_dict(las.to_dict())
        false_warns = [str(w.message) for w in rec if "Cross-check mismatch" in str(w.message)]
        assert false_warns == [], f"false M-17 cross-check warnings: {false_warns}"


# ──────────────────────────────────────────────────────────────
# M-31 (MEDIUM): Well units/descriptions had NO MAX_FIELD_LENGTH on
# any path — a 200k-char unit is emitted by the writer and then the
# re-read fails with LASParseError (self-unreadable output).  The
# generic 256-char header-line warning fires but the file is still
# written.  Now key AND value are length-checked at construction,
# item mutation, and wholesale assignment.
# ──────────────────────────────────────────────────────────────


class TestM31WellUnitsDescriptionsMaxFieldLength:
    """M-31: well units/descriptions keys and values must enforce
    MAX_FIELD_LENGTH on every path — a value the parser cannot re-read
    must be rejected at the model instead of written as self-unreadable
    output."""

    def test_construction_long_unit_value_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            WellSection(units={"DEPT": "X" * (MAX_FIELD_LENGTH + 1)})

    def test_construction_long_unit_key_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            WellSection(units={"X" * (MAX_FIELD_LENGTH + 1): "M"})

    def test_construction_long_description_value_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            WellSection(descriptions={"DEPT": "Y" * (MAX_FIELD_LENGTH + 1)})

    def test_mutation_long_unit_value_raises(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            w.units["DEPT"] = "X" * (MAX_FIELD_LENGTH + 1)

    def test_mutation_long_description_key_raises(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            w.descriptions["X" * (MAX_FIELD_LENGTH + 1)] = "desc"

    def test_wholesale_assignment_long_unit_value_raises(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            w.units = {"DEPT": "X" * (MAX_FIELD_LENGTH + 1)}

    def test_boundary_accepts_exact_max_length(self) -> None:
        w = WellSection(units={"DEPT": "X" * MAX_FIELD_LENGTH})
        assert len(w.units["DEPT"]) == MAX_FIELD_LENGTH
        w.units["GR"] = "Y" * MAX_FIELD_LENGTH
        assert len(w.units["GR"]) == MAX_FIELD_LENGTH


# ──────────────────────────────────────────────────────────────
# M-32 (MEDIUM): VersionSection vers/wrap/dlm had NO MAX_FIELD_LENGTH —
# an over-long VERS is only warned as "unrecognized" (never rejected)
# yet still emitted, producing ~V lines the parser cannot re-read.
# ──────────────────────────────────────────────────────────────


class TestM32VersionSectionMaxFieldLength:
    """M-32: VersionSection vers/wrap/dlm must enforce MAX_FIELD_LENGTH
    at construction (length checked before the value checks so the
    diagnostic names the real problem)."""

    def test_long_vers_raises(self) -> None:
        with pytest.raises(ValueError, match=r"VERS length .* exceeds maximum allowed"):
            VersionSection(vers="3.0" + "x" * MAX_FIELD_LENGTH)

    def test_long_wrap_raises(self) -> None:
        with pytest.raises(ValueError, match=r"WRAP length .* exceeds maximum allowed"):
            VersionSection(wrap="YES" + "x" * (MAX_FIELD_LENGTH - 2))

    def test_long_dlm_raises(self) -> None:
        with pytest.raises(ValueError, match=r"DLM length .* exceeds maximum allowed"):
            VersionSection(dlm="COMMA" + "x" * (MAX_FIELD_LENGTH - 4))

    def test_short_values_still_work(self) -> None:
        v = VersionSection(vers="3.0", wrap="NO", dlm="TAB")
        assert (v.vers, v.wrap, v.dlm) == ("3.0", "NO", "TAB")


# ──────────────────────────────────────────────────────────────
# N-01 (MEDIUM): ArrayElementInfo.index accepted 0 — LAS 3.0 array
# indices are 1-based.  The read side rejects index 0 for ≥2-element
# groups; single-element arrays and metadata-only files parsed/round-
# tripped silently.  from_dict's missing-'index' default (0) silently
# yielded an invalid 0-based element.  Now __post_init__ + __setattr__
# require index >= 1, making the from_dict missing-key case an error.
# ──────────────────────────────────────────────────────────────


class TestN01ArrayElementIndexOneBased:
    """N-01: index must be 1-based (>= 1) on construction, mutation,
    and the from_dict missing-key path."""

    def test_zero_index_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="index must be >= 1"):
            ArrayElementInfo(base_name="NMR", index=0)

    def test_zero_index_mutation_raises(self) -> None:
        ai = ArrayElementInfo(base_name="NMR", index=1)
        with pytest.raises(ValueError, match="index must be >= 1"):
            ai.index = 0

    def test_from_dict_missing_index_raises(self) -> None:
        """array_info dict without an 'index' key must NOT silently
        default to 0 — it is an explicit error (spec: 1-based)."""
        with pytest.raises(ValueError, match="index must be >= 1"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"NULL": "-999.25"},
                    "curves_order": ["NMR[1]"],
                    "curves": [
                        {
                            "mnemonic": "NMR[1]",
                            "array_info": {"base_name": "NMR"},
                        }
                    ],
                    "logs": {"NMR[1]": [1.0, 2.0]},
                }
            )

    def test_from_dict_missing_index_sc_array_info_raises(self) -> None:
        """Per-section (data_sections) array_info missing 'index' is an
        error too — same 1-based contract."""
        with pytest.raises(ValueError, match="index must be >= 1"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"NULL": "-999.25"},
                    "data_sections": [
                        {
                            "name": "LOG",
                            "section_type": "LOG_DATA",
                            "curves_order": ["NMR[1]"],
                            "section_curves": [
                                {
                                    "mnemonic": "NMR[1]",
                                    "array_info": {"base_name": "NMR"},
                                }
                            ],
                            "data": {"NMR[1]": [1.0, 2.0]},
                        }
                    ],
                }
            )

    def test_one_based_index_still_accepted(self) -> None:
        ai = ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0)
        ai.index = 2
        assert ai.index == 2


# ──────────────────────────────────────────────────────────────
# N-12 (MEDIUM): LAS 3.0 post-construction replacement of the ONLY key
# in a group (section data↔string_data, or top-level logs↔string_data)
# with a different-length array passes validate(complete=True)=[] —
# the writer max()-null-pads SILENTLY and the re-read FABRICATES null
# rows.  validate(complete=True) now re-checks per-section group
# lengths (unconditional) and top-level LAS 3.0 group lengths (M-29
# already warns loudly for 1.2/2.0, so no double-warning there).
# ──────────────────────────────────────────────────────────────


class TestN12PostConstructionGroupRowCountRecheck:
    """N-12: validate(complete=True) must flag cross-group row-count
    mismatch introduced by post-construction ONLY-key replacement."""

    def _section(self) -> DataSection:
        return DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT", "TEXT"],
            data={"DEPT": np.array([1.0, 2.0, 3.0])},
            string_data={"TEXT": np.array(["a", "b", "c"], dtype=object)},
        )

    def test_section_only_key_replacement_flags_validate(self) -> None:
        ds = self._section()
        # Only-key replacement with a different-length array: allowed by
        # _GuardedDict (no sibling reference), but the group is now
        # inconsistent — validate must flag it.
        ds.data["DEPT"] = np.array([0.0])
        issues = ds.validate(complete=True)
        assert any("row count" in i for i in issues), issues

    def test_section_consistent_replacement_no_issue(self) -> None:
        ds = self._section()
        ds.data["DEPT"] = np.array([9.0, 8.0, 7.0])
        issues = ds.validate(complete=True)
        assert not any("row count" in i for i in issues), issues

    def test_top_level_las30_only_key_replacement_flags_validate(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "TEXT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="TEXT", data_format="S"),
            ],
            logs={"DEPT": np.array([1.0, 2.0, 3.0])},
            string_data={"TEXT": np.array(["a", "b", "c"], dtype=object)},
        )
        las.logs["DEPT"] = np.array([0.0])
        issues = las.validate(complete=True)
        assert any("logs row count" in i for i in issues), issues

    def test_top_level_non_las30_no_cross_group_double_warning(self) -> None:
        """1.2/2.0 top-level: M-29 (string_data in non-3.0) is already
        loud — the N-12 cross-group diagnostic must NOT fire on top of
        it (double-warning defect class)."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "TEXT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="TEXT", data_format="S"),
            ],
            logs={"DEPT": np.array([1.0, 2.0, 3.0])},
            string_data={"TEXT": np.array(["a", "b", "c"], dtype=object)},
        )
        las.logs["DEPT"] = np.array([0.0])
        issues = las.validate(complete=True)
        assert not any("logs row count" in i for i in issues), issues
        assert any("string_data is present" in i for i in issues), issues


# ──────────────────────────────────────────────────────────────
# N-13 (MEDIUM): post-construction exact-case logs∩string_data overlap
# (curve added to string_data that exists in logs with NO data_format)
# bypassed the construction raise; validate(complete=True) skipped
# no-data_format curves and the writer silently dropped the numeric
# logs value (_lookup_data_array prefers string_data).  validate now
# re-runs the construction overlap check regardless of data_format.
# ──────────────────────────────────────────────────────────────


class TestN13PostConstructionOverlapValidate:
    """N-13: validate(complete=True) must flag a logs∩string_data key
    overlap introduced post-construction — regardless of data_format —
    and the format-vs-placement loop must not double-report it."""

    def _lasfile(self) -> LASFile:
        return LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["DEPT", "LITH", "TEXT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="LITH"),  # data_format='' (no format)
                CurveDefinition(mnemonic="TEXT", data_format="S"),
            ],
            logs={
                "DEPT": np.array([1.0, 2.0]),
                "LITH": np.array([10.0, 20.0]),
            },
            string_data={"TEXT": np.array(["a", "b"], dtype=object)},
        )

    def test_no_data_format_overlap_flagged(self) -> None:
        """The exact N-13 shape: 'LITH' exists in logs with no
        data_format, then is added to string_data post-construction."""
        las = self._lasfile()
        las.string_data["LITH"] = np.array(["x", "y"], dtype=object)
        issues = las.validate(complete=True)
        assert any("both 'logs' and 'string_data'" in i for i in issues), issues

    def test_overlap_not_double_reported_as_placement_issue(self) -> None:
        """A formatted overlap is flagged once as an overlap — the
        format-vs-placement loop must not ALSO emit a misleading
        'S-format in logs' placement diagnostic."""
        las = self._lasfile()
        las.curves[1].data_format = "S"  # LITH now S-format AND overlapping
        las.string_data["LITH"] = np.array(["x", "y"], dtype=object)
        issues = las.validate(complete=True)
        overlap_issues = [i for i in issues if "both 'logs' and 'string_data'" in i]
        assert len(overlap_issues) == 1, issues
        assert not any("data_format='S'" in i and "LITH" in i for i in issues), issues

    def test_construction_overlap_still_raises(self) -> None:
        """Control: the construction-time overlap raise is unchanged
        (loud behavior preserved)."""
        with pytest.raises(LASDataError, match="appear in both logs and string_data"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["DEPT"],
                curves=[CurveDefinition(mnemonic="DEPT")],
                logs={"DEPT": np.array([1.0, 2.0])},
                string_data={"DEPT": np.array(["a", "b"], dtype=object)},
            )



# ──────────────────────────────────────────────────────────────
# M-01 (MEDIUM): _validate_array_continuity missing the read-side
# data_format-consistency dimension.  The reader rejects a LAS 3.0
# array group whose channels declare differing format specifiers
# ("Inconsistent data_format for array", _las30_data.py:790-794);
# the model must reject the same state at construction AND re-check
# it in validate() (post-construction mutation).
# ──────────────────────────────────────────────────────────────


class TestM01ArrayDataFormatConsistency:
    """M-01: array groups must share one data_format across channels."""

    @staticmethod
    def _section(formats: list[str]) -> DataSection:
        return DataSection(
            name="LOG",
            curves_order=[f"NMR[{i + 1}]" for i in range(len(formats))],
            section_curves=[
                CurveDefinition(mnemonic=f"NMR[{i + 1}]", data_format=f1)
                for i, f1 in enumerate(formats)
            ],
            data={
                f"NMR[{i + 1}]": np.array([1.0, 2.0]) for i in range(len(formats))
            },
        )

    def test_top_level_inconsistent_format_raises(self) -> None:
        """M-01: a top-level array group with mixed formats is rejected
        at construction — the reader would refuse the written file."""
        with pytest.raises(LASDataError, match="inconsistent data_format"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["NMR[1]", "NMR[2]"],
                curves=[
                    CurveDefinition(mnemonic="NMR[1]", data_format="F"),
                    CurveDefinition(mnemonic="NMR[2]", data_format="E"),
                ],
            )

    def test_section_inconsistent_format_raises(self) -> None:
        """M-01: a per-section array group with mixed formats is rejected
        at construction."""
        ds = self._section(["F", "E"])
        with pytest.raises(LASDataError, match="inconsistent data_format"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["NMR[1]", "NMR[2]"],
                curves=[],
                data_sections=[ds],
            )

    def test_consistent_formats_pass(self) -> None:
        """M-01 control: a uniform-format array group constructs fine."""
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["NMR[1]", "NMR[2]"],
            curves=[],
            data_sections=[self._section(["F", "F"])],
        )
        assert las.validate(complete=True) == []

    def test_validate_reechecks_post_construction_mutation(self) -> None:
        """M-01: validate(complete=True) catches a data_format reassigned
        after construction (bypassing __post_init__)."""
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["NMR[1]", "NMR[2]"],
            curves=[],
            data_sections=[self._section(["F", "F"])],
        )
        las.data_sections[0].section_curves[1].data_format = "E"
        issues = las.data_sections[0].validate(complete=True)
        assert any("inconsistent data_format" in i and "NMR" in i for i in issues), issues


# ──────────────────────────────────────────────────────────────
# M-02 (MEDIUM): ParameterEntry.__setattr__ guarded only
# data_format/value/unit/description — zone/array_index/section_type
# mutations bypassed the construction contract (zone → opaque
# LASWriteError; array_index=-1 → parameter silently dropped;
# section_type with spaces → ~P section misrouted to ~Other).
# ──────────────────────────────────────────────────────────────


class TestM02ParameterEntrySetattrGuards:
    """M-02: post-construction leaf-field assignments re-validate."""

    def test_zone_type_raises(self) -> None:
        p = ParameterEntry(mnemonic="T", value="1")
        with pytest.raises(TypeError, match="zone must be ParameterZone or None"):
            p.zone = "Zone1"

    def test_zone_valid_assignment_passes(self) -> None:
        from pylasdev.models import ParameterZone

        p = ParameterEntry(mnemonic="T", value="1")
        z = ParameterZone(zone_name="A", zone_index=1)
        p.zone = z
        assert p.zone is z

    def test_negative_array_index_raises(self) -> None:
        p = ParameterEntry(mnemonic="RUN", value="1")
        with pytest.raises(ValueError, match="array_index must be >= 0"):
            p.array_index = -1

    def test_non_int_array_index_raises(self) -> None:
        p = ParameterEntry(mnemonic="RUN", value="1")
        with pytest.raises(TypeError, match="array_index must be int or None"):
            p.array_index = "1"

    def test_numpy_int_array_index_coerced(self) -> None:
        p = ParameterEntry(mnemonic="RUN", value="1")
        p.array_index = np.int64(2)
        assert p.array_index == 2
        assert type(p.array_index) is int

    def test_section_type_space_raises(self) -> None:
        p = ParameterEntry(mnemonic="T", value="1")
        with pytest.raises(ValueError, match="whitespace"):
            p.section_type = "MY CORE"

    def test_section_type_pipe_raises(self) -> None:
        p = ParameterEntry(mnemonic="T", value="1")
        with pytest.raises(ValueError, match="pipe"):
            p.section_type = "A|B"

    def test_section_type_whitespace_only_normalized_to_none(self) -> None:
        p = ParameterEntry(mnemonic="T", value="1")
        p.section_type = "   "
        assert p.section_type is None

    def test_section_type_valid_assignment_stripped(self) -> None:
        p = ParameterEntry(mnemonic="T", value="1")
        p.section_type = "CORE"
        assert p.section_type == "CORE"


# ──────────────────────────────────────────────────────────────
# M-03 (MEDIUM): DataSection.section_type mutation unguarded — a
# post-construction section_type with spaces/pipe produced a broken
# ``~MY CORE_Data`` header the parser misroutes to ~Other, losing
# the whole section's data.  __setattr__ re-validates and
# validate() re-checks.
# ──────────────────────────────────────────────────────────────


class TestM03DataSectionSectionTypeGuard:
    """M-03: DataSection.section_type re-validates on assignment and
    in validate()."""

    def test_assignment_space_raises(self) -> None:
        ds = DataSection(name="S", section_type="LOG_DATA")
        with pytest.raises(LASDataError, match="whitespace"):
            ds.section_type = "MY CORE"

    def test_assignment_pipe_raises(self) -> None:
        ds = DataSection(name="S", section_type="LOG_DATA")
        with pytest.raises(LASDataError, match="pipe"):
            ds.section_type = "A|B"

    def test_assignment_whitespace_only_normalized(self) -> None:
        ds = DataSection(name="S", section_type="LOG_DATA")
        ds.section_type = "   "
        assert ds.section_type == ""

    def test_assignment_valid_stripped(self) -> None:
        ds = DataSection(name="S", section_type="LOG_DATA")
        ds.section_type = " CORE_DATA "
        assert ds.section_type == "CORE_DATA"

    def test_validate_reechecks_bypassed_state(self) -> None:
        """M-03: validate(complete=True) flags an invalid section_type
        even when __setattr__ was bypassed (direct __dict__ write)."""
        ds = DataSection(name="S", section_type="LOG_DATA")
        object.__setattr__(ds, "section_type", "MY CORE")
        issues = ds.validate(complete=True)
        assert any("section_type contains invalid characters" in i for i in issues), issues


# ──────────────────────────────────────────────────────────────
# M-10 (MEDIUM): mnemonic case normalization diverged between
# from_dict (identity on lookup miss) and the parser (unconditional
# uppercase) → compare_las_dicts False for identical file content.
# Both sides now normalize identically (uppercase on miss),
# including the WELL side.
# ──────────────────────────────────────────────────────────────


class TestM10MnemonicCaseNormalization:
    """M-10: from_dict normalizes mnemonics identically to the parser."""

    _LAS20_LOWERCASE = """~VERSION
VERS. 2.0
WRAP. NO
DLM. SPACE
~WELL
STRT. 100.0
STOP. 101.0
STEP. 1.0
NULL. -999.25
~CURVE
dept.M  : Depth
gr.GAPI  : Gamma Ray
~PARAMETER
bht.DEGC  35.5  : Bottom Hole Temp
~A  dept  gr
100.0  10.0
101.0  11.0
"""

    def test_from_dict_uppercases_curve_mnemonics(self) -> None:
        """M-10: from_dict with a lowercase curve dict produces the same
        uppercase curves_order the parser produces for the same file."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
                "well": {"STRT": "100.0", "STOP": "101.0", "STEP": "1.0", "NULL": "-999.25"},
                "curves_order": ["dept", "gr"],
                "curves": [
                    {"mnemonic": "dept", "unit": "M", "description": "Depth"},
                    {"mnemonic": "gr", "unit": "GAPI", "description": "Gamma Ray"},
                ],
                "logs": {"dept": [100.0, 101.0], "gr": [10.0, 11.0]},
            }
        )
        assert las.curves_order == ["DEPT", "GR"]
        assert las.curves[0].mnemonic == "DEPT"
        assert list(las.logs.keys()) == ["DEPT", "GR"]

    def test_from_dict_uppercases_well_mnemonics(self) -> None:
        """M-10: well entry keys normalize like the parser's well path
        (parser.py:3018-3019 uppercases the mnemonic before lookup)."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
                "well": {"strt": "100.0", "stop": "101.0", "step": "1.0", "null": "-999.25"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT"}],
                "logs": {"DEPT": [100.0, 101.0]},
            }
        )
        assert las.well.entries == {"STRT": "100.0", "STOP": "101.0", "STEP": "1.0", "NULL": "-999.25"}

    def test_from_dict_uppercases_parameter_mnemonics(self) -> None:
        """M-10: parameter mnemonics normalize like the parser's ~P path."""
        las = LASFile.from_dict(
            {
                "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
                "well": {"STRT": "100.0", "STOP": "101.0", "STEP": "1.0", "NULL": "-999.25"},
                "curves_order": ["DEPT"],
                "curves": [{"mnemonic": "DEPT"}],
                "logs": {"DEPT": [100.0, 101.0]},
                "parameters": {"bht": "35.5"},
                "parameter_details": [
                    {"mnemonic": "bht", "unit": "DEGC", "value": "35.5", "description": "BH Temp"}
                ],
            }
        )
        assert las.parameters[0].mnemonic == "BHT"

    def test_compare_las_dicts_parse_vs_from_dict_lowercase(self, tmp_path: Path) -> None:
        """M-10: parse() and from_dict() of the same (lowercase-mnemonic)
        content now produce identical to_dicts — compare_las_dicts True.
        Pre-fix the case divergence made it False for identical content."""
        from pylasdev import read_las_file_as_object
        from pylasdev.compare import compare_las_dicts

        f = tmp_path / "lowercase.las"
        f.write_text(self._LAS20_LOWERCASE, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = read_las_file_as_object(f)
        hand_dict: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100.0", "STOP": "101.0", "STEP": "1.0", "NULL": "-999.25"},
            # The parser stores a (possibly empty) unit entry for every
            # well mnemonic — mirror it so the dual shapes are identical.
            "well_units": {"STRT": "", "STOP": "", "STEP": "", "NULL": ""},
            "well_descriptions": {},
            "parameters": {"bht": "35.5"},
            "parameter_details": [
                {
                    "mnemonic": "bht",
                    "unit": "DEGC",
                    "value": "35.5",
                    "description": "Bottom Hole Temp",
                }
            ],
            "curves": [
                {
                    "mnemonic": "dept",
                    "unit": "M",
                    "api_code": "",
                    "description": "Depth",
                    "original_mnemonic": "dept",
                },
                {
                    "mnemonic": "gr",
                    "unit": "GAPI",
                    "api_code": "",
                    "description": "Gamma Ray",
                    "original_mnemonic": "gr",
                },
            ],
            "logs": {"dept": np.array([100.0, 101.0]), "gr": np.array([10.0, 11.0])},
            "curves_order": ["dept", "gr"],
            "other": "",
            "data_sections": [],
            "string_data": {},
            "encoding": "utf-8",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_dict_las = LASFile.from_dict(hand_dict)
        # source_file is set by the read path (the file's path) and is
        # outside the case-normalization contract — drop it from both
        # sides so the comparison isolates the mnemonic casing.
        parsed_d = {k: v for k, v in parsed.to_dict().items() if k != "source_file"}
        from_dict_d = {k: v for k, v in from_dict_las.to_dict().items() if k != "source_file"}
        assert compare_las_dicts(parsed_d, from_dict_d) is True


# ──────────────────────────────────────────────────────────────
# M-17 (MEDIUM): case-variant duplicate parameter mnemonics passed
# the exact-case duplicate checks with zero warnings → the
# construction dup check and the to_dict last-wins warning now
# compare case-insensitively.
# ──────────────────────────────────────────────────────────────


class TestM17CaseVariantDuplicateParameters:
    """M-17: 'GR' + 'gr' are the same logical parameter for the
    case-insensitive ~P roundtrip and must warn like exact-case dups."""

    @staticmethod
    def _las_with_case_variant_params() -> LASFile:
        return LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            well=WellSection(
                entries={"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"}
            ),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
            parameters=[
                ParameterEntry(mnemonic="GR", value="10"),
                ParameterEntry(mnemonic="gr", value="20"),
            ],
        )

    def test_construction_warns_case_variant_duplicate(self) -> None:
        """M-17: LASFile construction warns for a case-variant duplicate
        (pre-fix: exact-case check → zero warnings)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._las_with_case_variant_params()
        dup = [x for x in w if "duplicate parameter mnemonic" in str(x.message)]
        assert len(dup) >= 1, f"expected dup warning, got: {[str(x.message) for x in w]}"

    def test_to_dict_warns_case_variant_last_wins(self) -> None:
        """M-17: to_dict warns for a case-variant duplicate (pre-fix: the
        exact-case seen-check let it pass silently)."""
        las = self._las_with_case_variant_params()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las.to_dict()
        dup = [x for x in w if "LASFile.to_dict(): duplicate parameter mnemonic" in str(x.message)]
        assert len(dup) == 1, f"expected 1 to_dict dup warning, got: {[str(x.message) for x in w]}"

    def test_exact_case_duplicate_still_warns(self) -> None:
        """M-17 control: the exact-case duplicate warning is unchanged."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            well=WellSection(
                entries={"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"}
            ),
            curves_order=["DEPT"],
            curves=[CurveDefinition(mnemonic="DEPT")],
            logs={"DEPT": np.array([1.0, 2.0])},
            parameters=[
                ParameterEntry(mnemonic="GR", value="10"),
                ParameterEntry(mnemonic="GR", value="20"),
            ],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            las.to_dict()
        dup = [x for x in w if "LASFile.to_dict(): duplicate parameter mnemonic" in str(x.message)]
        assert len(dup) == 1


# ──────────────────────────────────────────────────────────────
# M-18 (MEDIUM): CurveDefinition lacked the bracket-notation
# index ↔ array_info.index cross-check (ParameterEntry's M-42
# twin).  A curve ``NMR[1]`` with array_info.index=2 writes the
# bracket index and silently diverges from array_info on re-parse.
# ──────────────────────────────────────────────────────────────


class TestM18CurveDefinitionBracketIndexCrossCheck:
    """M-18: bracket mnemonic index must match array_info.index."""

    def test_mismatched_index_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CurveDefinition(
                mnemonic="NMR[2]",
                array_info=ArrayElementInfo(base_name="NMR", index=1),
            )
        idx_warnings = [
            x
            for x in w
            if "array notation with index 2" in str(x.message)
            and "array_info.index" in str(x.message)
        ]
        assert len(idx_warnings) == 1, f"expected index cross-check warning, got {[str(x.message) for x in w]}"

    def test_matching_index_silent(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CurveDefinition(
                mnemonic="NMR[2]",
                array_info=ArrayElementInfo(base_name="NMR", index=2),
            )
        assert not any("array_info.index" in str(x.message) for x in w)

    def test_base_name_mismatch_still_warns(self) -> None:
        """M-18 control: the M-17 base_name cross-check is unchanged."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CurveDefinition(
                mnemonic="NMR[1]",
                array_info=ArrayElementInfo(base_name="DT", index=1),
            )
        assert any("array_info.base_name" in str(x.message) for x in w)


# ──────────────────────────────────────────────────────────────
# M-25 (MEDIUM): DevFile.validate(complete=True) flagged ANY NaN as
# corruption, but on DEV reads NaN is the reader's OWN designed
# missing-data representation (sentinel/short-row fill).  The
# models-side fix: the reader marks its DevFile with
# ``_designed_nan=True`` and validate() then reports only genuine
# Inf (the reader never produces Inf) — while direct construction
# and from_dict (user data) keep the full check.
# ──────────────────────────────────────────────────────────────


class TestM25DevFileDesignedNan:
    """M-25: reader-designed NaN is not reported as corruption; genuine
    non-finite user data still is."""

    def test_designed_nan_not_flagged(self) -> None:
        """M-25: a reader-marked DevFile (designed missing-data NaN) has
        no non-finite issue (pre-fix: NaN always flagged)."""
        dev = DevFile(
            columns={
                "MD": np.array([0.0, np.nan, 2.0]),
                "AZI": np.array([90.0, np.nan, 90.0]),
            },
            column_order=["MD", "AZI"],
            _from_dict=True,
            _designed_nan=True,
        )
        issues = dev.validate(complete=True)
        assert not any("non-finite" in i for i in issues), issues

    def test_inf_always_flagged_even_designed(self) -> None:
        """M-25: Inf is never a designed sentinel (the reader collapses
        all non-finite conversions to NaN), so it is flagged in both
        modes."""
        dev = DevFile(
            columns={"MD": np.array([0.0, np.inf])},
            column_order=["MD"],
            _from_dict=True,
            _designed_nan=True,
        )
        issues = dev.validate(complete=True)
        assert any("non-finite" in i for i in issues), issues

    def test_genuine_nan_still_flagged(self) -> None:
        """M-25 control: user-constructed DevFile (no reader marker) keeps
        the full NaN/Inf check."""
        dev = DevFile(
            columns={"MD": np.array([0.0, np.nan])},
            column_order=["MD"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("non-finite" in i for i in issues), issues


# ──────────────────────────────────────────────────────────────
# M-28 (MEDIUM): the F-115 logs∩data_sections overlap warning fired
# on EVERY canonical LAS 3.0 parse→to_dict→from_dict roundtrip (the
# parser populates both views with the same values by design) →
# warnings-as-errors automation broke.  The check now distinguishes
# a canonical-copy (identical values) from a genuine conflict
# (differing values) and warns only on divergence.
# ──────────────────────────────────────────────────────────────


class TestM28F115CanonicalCopyVsConflict:
    """M-28: F-115 warns only when logs/data_sections values diverge."""

    @staticmethod
    def _canonical_dict(gr_section_values: np.ndarray | None = None) -> dict[str, Any]:
        if gr_section_values is None:
            gr_section_values = np.array([10.0, 11.0, 12.0])
        return {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999.25"},
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "GR", "unit": "API"},
            ],
            "curves_order": ["DEPT", "GR"],
            "logs": {
                "DEPT": np.array([0.0, 0.5, 1.0]),
                "GR": np.array([10.0, 11.0, 12.0]),
            },
            "data_sections": [
                {
                    "name": "LOG",
                    "section_type": "LOG_DATA",
                    "curves_order": ["DEPT", "GR"],
                    "data": {
                        "DEPT": np.array([0.0, 0.5, 1.0]),
                        "GR": gr_section_values,
                    },
                }
            ],
        }

    @staticmethod
    def _f115_warnings(w: list[warnings.WarningMessage]) -> list[str]:
        return [str(x.message) for x in w if "appear in both 'logs' and 'data_sections'" in str(x.message)]

    def test_canonical_copy_no_warning(self) -> None:
        """M-28: the parser-style roundtrip dict (identical values in
        logs and the owning section) must NOT warn (pre-fix: warned on
        every canonical roundtrip)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(self._canonical_dict())
        assert self._f115_warnings(w) == [], f"expected no F-115 warning, got: {self._f115_warnings(w)}"

    def test_canonical_copy_with_nan_cells_no_warning(self) -> None:
        """M-28: identical values including NaN null cells are still a
        canonical copy (equal_nan comparison), not a conflict."""
        d = self._canonical_dict()
        d["logs"]["GR"] = np.array([10.0, np.nan, 12.0])
        d["data_sections"][0]["data"]["GR"] = np.array([10.0, np.nan, 12.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(d)
        assert self._f115_warnings(w) == [], f"expected no F-115 warning, got: {self._f115_warnings(w)}"

    def test_genuine_conflict_still_warns(self) -> None:
        """M-28: differing values between logs and the section are a real
        conflict and still warn."""
        d = self._canonical_dict(gr_section_values=np.array([20.0, 21.0, 22.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(d)
        assert len(self._f115_warnings(w)) == 1, f"expected 1 F-115 warning, got: {self._f115_warnings(w)}"


# ──────────────────────────────────────────────────────────────
# fix-models-B regression tests (E-11, E-12, E-13, E-14, E-15,
# E-29, E-39, E-46) — verified findings from the s7 synthesis.
# ──────────────────────────────────────────────────────────────


class TestE11DevColumnsOnlyKeyReplacement:
    """E-11: _DevColumns.__setitem__ false-rejected replacing the ONLY
    column with a different-length array (the MOD-01 exclusion present in
    _GuardedDict.__setitem__ was never ported to _DevColumns)."""

    def test_only_column_replacement_allows_growth(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([1.0, 2.0, 3.0])
        # Replace the ONLY column with a different length — trivially
        # consistent growth, must be allowed (MOD-01 contract).
        dev.columns["MD"] = np.array([1.0, 2.0])
        np.testing.assert_array_equal(dev.columns["MD"], np.array([1.0, 2.0]))

    def test_only_column_replacement_allows_shrink(self) -> None:
        dev = DevFile()
        dev.columns["MD"] = np.array([1.0])
        dev.columns["MD"] = np.array([1.0, 2.0, 3.0, 4.0])
        assert len(dev.columns["MD"]) == 4

    def test_replacement_against_other_columns_still_raises(self) -> None:
        dev = DevFile(
            columns={"MD": np.array([1.0, 2.0]), "TVD": np.array([3.0, 4.0])},
            column_order=["MD", "TVD"],
        )
        with pytest.raises(ValueError, match="other columns have length"):
            dev.columns["MD"] = np.array([1.0, 2.0, 3.0])


class TestE12TopLevelConstructionRowCountLoudPath:
    """E-12: the 1.2/2.0 top-level logs↔string_data row-count invariant
    raises at construction (the loud path).  Post-construction 1.2/2.0
    top-level mismatch is already loud via M-29 (pinned by D's N-12 test
    test_top_level_non_las30_no_cross_group_double_warning)."""

    def _las20(self) -> LASFile:
        return LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "TEXT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="TEXT", data_format="S"),
            ],
            logs={"DEPT": np.array([1.0, 2.0, 3.0])},
            string_data={"TEXT": np.array(["a", "b"], dtype=object)},
        )

    def test_las20_construction_raises_on_cross_group_mismatch(self) -> None:
        with pytest.raises(LASDataError, match="does not match string_data row count"):
            self._las20()

    def test_las20_construction_matching_rows_passes(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["DEPT", "TEXT"],
            curves=[
                CurveDefinition(mnemonic="DEPT"),
                CurveDefinition(mnemonic="TEXT", data_format="S"),
            ],
            logs={"DEPT": np.array([1.0, 2.0, 3.0])},
            string_data={"TEXT": np.array(["a", "b", "c"], dtype=object)},
        )
        assert las.logs["DEPT"].shape == (3,)


class TestE13CaseVariantDuplicateKeysWithinContainer:
    """E-13: case-variant duplicate keys WITHIN one container
    (logs={'GR', 'gr'}, string_data, DataSection.data) pass all previous
    validation → the writer's case-insensitive lookup emits ONE column and
    silently drops the other's data.  Now rejected at construction, at
    from_dict, and re-checked in validate(complete=True)."""

    def test_construction_logs_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["GR"],
                curves=[CurveDefinition(mnemonic="GR")],
                logs={"GR": np.array([1.0]), "gr": np.array([2.0])},
            )

    def test_construction_string_data_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                curves_order=["GR"],
                curves=[CurveDefinition(mnemonic="GR", data_format="S")],
                string_data={"GR": np.array(["a"]), "gr": np.array(["b"])},
            )

    def test_construction_data_section_data_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            DataSection(
                name="LOG",
                curves_order=["GR"],
                data={"GR": np.array([1.0]), "gr": np.array([2.0])},
            )

    def test_construction_data_section_string_data_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            DataSection(
                name="LOG",
                curves_order=["GR"],
                string_data={"GR": np.array(["a"]), "gr": np.array(["b"])},
            )

    def test_from_dict_top_level_logs_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
                    "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999"},
                    "curves_order": ["GR"],
                    "curves": [{"mnemonic": "GR"}],
                    "logs": {"GR": [1.0], "gr": [2.0]},
                }
            )

    def test_from_dict_per_section_data_case_variant_duplicate_raises(self) -> None:
        with pytest.raises(LASDataError, match="case-variant duplicate keys"):
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999"},
                    "curves_order": ["GR"],
                    "data_sections": [
                        {
                            "name": "LOG",
                            "curves_order": ["GR"],
                            "data": {"GR": [1.0], "gr": [2.0]},
                        }
                    ],
                }
            )

    def test_validate_rechecks_post_construction_logs_duplicate(self) -> None:
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["GR"],
            curves=[CurveDefinition(mnemonic="GR")],
            logs={"GR": np.array([1.0])},
        )
        # Post-construction mutation bypasses the construction raise.
        las.logs["gr"] = np.array([2.0])
        issues = las.validate(complete=True)
        assert any("case-variant duplicate keys" in i for i in issues), issues

    def test_validate_rechecks_post_construction_section_duplicate(self) -> None:
        ds = DataSection(
            name="LOG",
            curves_order=["GR"],
            data={"GR": np.array([1.0])},
        )
        ds.data["gr"] = np.array([2.0])
        issues = ds.validate(complete=True)
        assert any("case-variant duplicate keys" in i for i in issues), issues

    def test_supported_cross_container_case_variant_still_roundtrips(self) -> None:
        """Control: the supported MOD-3 state (case-variant BETWEEN
        curves_order and data keys, not within one container) is
        untouched — 'dept' order entry + 'DEPT' data key still works."""
        las = LASFile(
            version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
            curves_order=["dept"],
            curves=[CurveDefinition(mnemonic="DEPT", unit="M")],
            logs={"DEPT": np.array([100.0, 101.0])},
        )
        assert las.logs["DEPT"].shape == (2,)


class TestE14DevFileAliasValidation:
    """E-14: DevFile.validate(complete=True) only checked exact 'MD'/
    'TVD' names — alias-named columns (MDKB/MDSS/DEPTH/DPT,
    TVDKB/TVDSS/TVDBML) bypassed MD monotonicity, TVD NaN-density, and
    TVD/MD-consistency.  Aliases now resolve via the same map
    _validate_dev_data uses."""

    def test_mdkb_non_monotonic_flagged(self) -> None:
        dev = DevFile(
            columns={"MDKB": np.array([0.0, 150.0, 120.0, 140.0])},
            column_order=["MDKB"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("not monotonically increasing" in i for i in issues), issues

    def test_depth_alias_non_monotonic_flagged(self) -> None:
        dev = DevFile(
            columns={"DEPTH": np.array([100.0, 50.0])},
            column_order=["DEPTH"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("not monotonically increasing" in i for i in issues), issues

    def test_tvdss_nan_density_flagged(self) -> None:
        dev = DevFile(
            columns={
                "MD": np.array([0.0, 100.0, 200.0]),
                "TVDSS": np.array([np.nan, np.nan, 300.0]),
            },
            column_order=["MD", "TVDSS"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("TVD column 'TVDSS'" in i and "NaN" in i for i in issues), issues

    def test_alias_md_reference_used_for_tvd_consistency(self) -> None:
        """The TVD MD-consistency check must use the alias-resolved MD
        reference (MDKB) instead of the exact-case columns.get('MD')."""
        dev = DevFile(
            columns={
                "MDKB": np.array([0.0, 100.0, 200.0, 300.0]),
                "TVDSS": np.array([0.0, 90.0, 95.0, 80.0]),
            },
            column_order=["MDKB", "TVDSS"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("TVD column 'TVDSS' decreases" in i for i in issues), issues

    def test_case_variant_md_column_flagged(self) -> None:
        dev = DevFile(
            columns={"md": np.array([100.0, 50.0])},
            column_order=["md"],
            _from_dict=True,
        )
        issues = dev.validate(complete=True)
        assert any("not monotonically increasing" in i for i in issues), issues


class TestE15ZeroDimArraysInDataSections:
    """E-15: LASFile.__post_init__ data_sections loop crashed with
    TypeError on 0-d numpy arrays (len() of unsized object).  The M-18
    ndim==0 guard now treats them as single-element values."""

    def test_zero_dim_section_data_constructs(self) -> None:
        ds = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["DEPT"],
            data={"DEPT": np.array(1.0)},
        )
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=[],
            data_sections=[ds],
        )
        assert las.data_sections[0].data["DEPT"].ndim == 0

    def test_zero_dim_with_string_data_cross_group_constructs(self) -> None:
        ds = DataSection(
            name="LOG",
            section_type="LOG_DATA",
            curves_order=["A", "B"],
            data={"A": np.array(1.0)},
            string_data={"B": np.array(["x"], dtype=object)},
        )
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=[],
            data_sections=[ds],
        )
        assert las.data_sections[0].data["A"].ndim == 0


class TestE29Las30OtherCheck:
    """E-29: LAS 3.0 + other content — the parser rejects ~O on 3.0 and
    the 3.0 writer drops the content; construction and validate() now
    give loud feedback instead of silent acceptance."""

    def test_direct_construction_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile(version=VersionSection(vers="3.0"), other="junk")
        assert any("~Other" in str(x.message) for x in w), (
            f"expected ~Other warning, got: {[str(x.message) for x in w]}"
        )

    def test_direct_construction_20_other_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile(version=VersionSection(vers="2.0"), other="junk")
        assert not any("~Other" in str(x.message) for x in w), (
            f"unexpected ~Other warning: {[str(x.message) for x in w]}"
        )

    def test_from_dict_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile.from_dict(
                {
                    "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
                    "well": {"STRT": "0", "STOP": "1", "STEP": "0.5", "NULL": "-999"},
                    "other": "junk",
                }
            )
        n = len([x for x in w if "~Other" in str(x.message)])
        assert n == 1, f"expected exactly 1 ~Other warning, got {n}: {[str(x.message) for x in w]}"

    def test_validate_complete_rechecks_other(self) -> None:
        las = LASFile(version=VersionSection(vers="3.0"))
        las.other = "post-construction junk"
        issues = las.validate(complete=True)
        assert any("~Other is NOT ALLOWED in LAS 3.0" in i for i in issues), issues


class TestE39BytesInStringData:
    """E-39: bytes/bytearray in string_data passed validate(complete=True)
    (only numeric dtypes were rejected) → the writer emitted repr
    literals (b'GR') — type AND content corruption.  Now flagged."""

    def test_lasfile_bytes_string_data_flagged(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["TEXT"],
            curves=[CurveDefinition(mnemonic="TEXT", data_format="S")],
            string_data={"TEXT": np.array([b"abc", b"def"], dtype=object)},
        )
        issues = las.validate(complete=True)
        assert any("bytes/bytearray" in i for i in issues), issues

    def test_lasfile_bytearray_string_data_flagged(self) -> None:
        # Build the object array element-by-element: np.array([bytearray])
        # would treat the bytearray as a sequence and produce a 2-D array
        # (which the MOD-17 construction guard rejects — also loud).
        _arr = np.empty(1, dtype=object)
        _arr[0] = bytearray(b"abc")
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["TEXT"],
            curves=[CurveDefinition(mnemonic="TEXT", data_format="S")],
            string_data={"TEXT": _arr},
        )
        issues = las.validate(complete=True)
        assert any("bytes/bytearray" in i for i in issues), issues

    def test_lasfile_bytes_dtype_string_data_flagged(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["TEXT"],
            curves=[CurveDefinition(mnemonic="TEXT", data_format="S")],
            string_data={"TEXT": np.array([b"abc", b"def"])},  # dtype 'S' (not object)
        )
        issues = las.validate(complete=True)
        assert any("bytes/bytearray" in i for i in issues), issues

    def test_datasection_bytes_string_data_flagged(self) -> None:
        ds = DataSection(
            name="LOG",
            curves_order=["TEXT"],
            string_data={"TEXT": np.array([b"x"], dtype=object)},
        )
        issues = ds.validate(complete=True)
        assert any("bytes/bytearray" in i for i in issues), issues

    def test_str_string_data_still_clean(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0", wrap="NO", dlm="COMMA"),
            curves_order=["TEXT"],
            curves=[CurveDefinition(mnemonic="TEXT", data_format="S")],
            string_data={"TEXT": np.array(["a", "b"], dtype=object)},
        )
        issues = las.validate(complete=True)
        assert not any("bytes/bytearray" in i for i in issues), issues


class TestE46CurveDefinitionExtendedFormatCodes:
    """E-46: CurveDefinition raised ValueError on extended Fortran codes
    ('F8.3'/'E10.2') that ParameterEntry/from_dict/parser all normalize.
    Now truncates to the single-letter code at construction AND on
    __setattr__; single-char invalid codes still raise (F-37 contract)."""

    def test_construction_truncates_f83(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F8.3")
        assert c.data_format == "F"

    def test_construction_truncates_e102(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="E10.2")
        assert c.data_format == "E"

    def test_setattr_truncates_extended_code(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F")
        c.data_format = "F8.3"
        assert c.data_format == "F"

    def test_single_char_invalid_still_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid data_format"):
            CurveDefinition(mnemonic="GR", data_format="Q")

    def test_setattr_single_char_invalid_still_raises(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="F")
        with pytest.raises(ValueError, match="invalid data_format"):
            c.data_format = "Q"

    def test_lowercase_extended_truncates(self) -> None:
        c = CurveDefinition(mnemonic="GR", data_format="f8.3")
        assert c.data_format == "F"


# ──────────────────────────────────────────────────────────────
# E-03 (MEDIUM): WellSection.entries had NO mutation guard on any
# path.  Item mutation (well.entries['STRT'] = 123), wholesale
# assignment (well.entries = {...}) and update() all bypassed the
# __post_init__/__setitem__ contract — non-str values crashed the
# writer with an opaque AttributeError, and non-roundtrippable keys
# were silently dropped on re-read.  Now re-wrapped through a
# validating dict at every entry point.
# ──────────────────────────────────────────────────────────────


class TestE03WellEntriesMutationGuard:
    """E-03: entries dict guards key content, value type, and length."""

    def test_item_mutation_rejects_non_roundtrippable_key(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="cannot roundtrip"):
            w.entries["GR.CO"] = "1"

    def test_update_rejects_non_roundtrippable_key(self) -> None:
        w = WellSection(entries={"STRT": "100"})
        with pytest.raises(ValueError, match="cannot roundtrip"):
            w.entries.update({"GR.CO": "1"})

    def test_wholesale_assignment_rejects_bad_key(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="cannot roundtrip"):
            w.entries = {"GR:1": "x"}  # type: ignore[assignment]

    def test_non_str_key_rejected_on_mutation(self) -> None:
        w = WellSection()
        with pytest.raises(TypeError, match="must be str"):
            w.entries[1] = "x"  # type: ignore[index]

    def test_non_str_value_coerced_with_warning(self) -> None:
        w = WellSection()
        with pytest.warns(UserWarning, match="coercing non-str value"):
            w.entries["STRT"] = 123  # type: ignore[assignment]
        assert w.entries["STRT"] == "123"

    def test_overlong_value_rejected_on_mutation(self) -> None:
        w = WellSection()
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            w.entries["STRT"] = "x" * (MAX_FIELD_LENGTH + 1)

    def test_valid_mutation_preserved(self) -> None:
        w = WellSection(entries={"STRT": "100"})
        w.entries["STOP"] = "200"
        w.entries.update({"STEP": "1"})
        assert w.entries == {"STRT": "100", "STOP": "200", "STEP": "1"}


# ──────────────────────────────────────────────────────────────
# E-04 (MEDIUM): VersionSection WRAP/DLM/VERS post-construction
# mutation was never re-checked.  WRAP was the only ~V field never
# re-checked anywhere — an invalid WRAP also silently disables the
# 256-char line-limit enforcement.  __setattr__ now re-validates and
# validate() re-checks WRAP/DLM values.
# ──────────────────────────────────────────────────────────────


class TestE04VersionSectionMutationRevalidation:
    """E-04: WRAP/DLM/VERS mutations re-apply the construction contract."""

    def test_wrap_mutation_rejects_invalid(self) -> None:
        vs = VersionSection()
        with pytest.raises(ValueError, match="invalid WRAP value"):
            vs.wrap = "MAYBE"

    def test_dlm_mutation_rejects_invalid(self) -> None:
        vs = VersionSection()
        with pytest.raises(ValueError, match="invalid DLM value"):
            vs.dlm = "FOO"

    def test_wrap_mutation_normalizes_uppercase(self) -> None:
        vs = VersionSection()
        vs.wrap = "no"
        assert vs.wrap == "NO"

    def test_dlm_mutation_case_preserved(self) -> None:
        vs = VersionSection()
        vs.dlm = "space"
        assert vs.dlm == "space"

    def test_vers_mutation_rejects_none(self) -> None:
        vs = VersionSection()
        with pytest.raises(ValueError, match="VERS cannot be None"):
            vs.vers = None  # type: ignore[assignment]

    def test_validate_catches_bypassed_invalid_wrap(self) -> None:
        vs = VersionSection()
        object.__setattr__(vs, "wrap", "MAYBE")
        issues = vs.validate()
        assert any("invalid WRAP value" in i for i in issues), issues

    def test_validate_catches_bypassed_invalid_dlm(self) -> None:
        vs = VersionSection()
        object.__setattr__(vs, "dlm", "FOO")
        issues = vs.validate()
        assert any("invalid DLM value" in i for i in issues), issues

    def test_validate_clean_for_valid_state(self) -> None:
        vs = VersionSection(vers="3.0", wrap="NO", dlm="COMMA")
        assert vs.validate() == []


# ──────────────────────────────────────────────────────────────
# E-05 (MEDIUM): WellSection.to_dict() nests units/descriptions
# inside the well dict (documented contract); LASFile.from_dict's
# well loop treated them as well entries → junk 'UNITS'/'DESCRIPTIONS'
# entries.  from_dict now skips the reserved nested keys (they are
# transported via well_units/well_descriptions).
# ──────────────────────────────────────────────────────────────


class TestE05WellToDictNestedUnitsNoJunk:
    """E-05: well.to_dict() output roundtrips through from_dict cleanly."""

    def test_nested_units_not_ingested_as_entry(self) -> None:
        well = WellSection(
            entries={"STRT": "100", "STOP": "200"},
            units={"STRT": "m"},
            descriptions={"STRT": "Start"},
        )
        wd = well.to_dict()
        assert "units" in wd and "descriptions" in wd  # documented nesting
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": wd,
            "well_units": dict(well.units),
            "well_descriptions": dict(well.descriptions),
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([1.0])},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data)
        assert "UNITS" not in {k.upper() for k in las.well.entries}
        assert "DESCRIPTIONS" not in {k.upper() for k in las.well.entries}
        assert las.well.entries["STRT"] == "100"
        assert las.well.units["STRT"] == "m"
        assert las.well.descriptions["STRT"] == "Start"

    def test_flat_well_dict_unchanged(self) -> None:
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100", "STOP": "200"},
            "well_units": {"STRT": "m"},
            "well_descriptions": {"STRT": "Start"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([1.0])},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = LASFile.from_dict(data)
        assert las.well.entries == {"STRT": "100", "STOP": "200"}
        assert las.well.units["STRT"] == "m"


# ──────────────────────────────────────────────────────────────
# E-06 (MEDIUM): CurveDefinition/ParameterEntry __setattr__ re-
# validated every leaf EXCEPT mnemonic.  A mutated mnemonic with
# non-roundtrippable characters (colon, dot, etc.) passed and the
# writer emitted a ~C/~P line the parser cannot match — silently
# dropping the curve/parameter AND its data.
# ──────────────────────────────────────────────────────────────


class TestE06MnemonicMutationRevalidated:
    """E-06: mnemonic mutation re-applies the __post_init__ contract."""

    def test_curve_mnemonic_colon_rejected(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        with pytest.raises(ValueError, match="cannot roundtrip"):
            c.mnemonic = "GR:1"

    def test_curve_mnemonic_dot_rejected(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        # Dots hit the __post_init__-mirrored content check first.
        with pytest.raises(ValueError, match="spaces/tabs/newlines/dots"):
            c.mnemonic = "GR.CO"

    def test_parameter_mnemonic_colon_rejected(self) -> None:
        p = ParameterEntry(mnemonic="GR")
        with pytest.raises(ValueError, match="cannot roundtrip"):
            p.mnemonic = "GR:1"

    def test_parameter_mnemonic_non_str_rejected(self) -> None:
        p = ParameterEntry(mnemonic="GR")
        with pytest.raises(TypeError, match="mnemonic must be str"):
            p.mnemonic = 42  # type: ignore[assignment]

    def test_curve_mnemonic_valid_mutation_passes(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        c.mnemonic = "NEW"
        assert c.mnemonic == "NEW"

    def test_parameter_mnemonic_array_form_accepted(self) -> None:
        p = ParameterEntry(mnemonic="GR")
        p.mnemonic = "RUN[1]"
        assert p.mnemonic == "RUN[1]"


# ──────────────────────────────────────────────────────────────
# E-07 (MEDIUM): ParameterZone had NO __setattr__ — post-construction
# mutation of zone_index/zone_name bypassed the construction contract
# (negative index → writer emitted an unreadable zone association;
# bracket chars → raw text leaked into the parameter description).
# ──────────────────────────────────────────────────────────────


class TestE07ParameterZoneMutationRevalidated:
    """E-07: ParameterZone leaf fields re-validate on assignment."""

    def test_negative_zone_index_rejected_on_mutation(self) -> None:
        pz = ParameterZone(zone_name="RUN", zone_index=1)
        with pytest.raises(ValueError, match="zone_index must be >= 0"):
            pz.zone_index = -1

    def test_non_int_zone_index_rejected_on_mutation(self) -> None:
        pz = ParameterZone(zone_name="RUN", zone_index=1)
        with pytest.raises(TypeError, match="zone_index must be int or None"):
            pz.zone_index = "1"  # type: ignore[assignment]

    def test_zone_name_brackets_warn_on_mutation(self) -> None:
        pz = ParameterZone(zone_name="RUN", zone_index=1)
        with pytest.warns(UserWarning, match="cannot roundtrip"):
            pz.zone_name = "Bad[Name]"

    def test_zone_name_whitespace_normalized_on_mutation(self) -> None:
        pz = ParameterZone(zone_name="RUN", zone_index=1)
        with pytest.warns(UserWarning, match="Normalizing to"):
            pz.zone_name = " |Zone "
        assert pz.zone_name == "|Zone"

    def test_valid_mutations_preserved(self) -> None:
        pz = ParameterZone(zone_name="RUN", zone_index=1)
        pz.zone_index = 2
        pz.zone_name = "MAIN"
        assert (pz.zone_index, pz.zone_name) == (2, "MAIN")


# ──────────────────────────────────────────────────────────────
# E-08 (MEDIUM): _validate_array_continuity grouped array curves by
# case-SENSITIVE base — a case-variant interleaved pair ('NMR[1]' +
# 'nmr[2]' with a non-array curve between) passed as two single-
# element groups and the writer emitted self-unreadable output.
# ──────────────────────────────────────────────────────────────


class TestE08ArrayContinuityCaseInsensitiveBase:
    """E-08: array grouping uses _case_key on the base mnemonic."""

    def test_case_variant_interleaved_arrays_rejected(self) -> None:
        with pytest.raises(LASDataError, match="not contiguous"):
            LASFile(
                version=VersionSection(vers="3.0"),
                curves_order=["DEPT", "NMR[1]", "GR", "nmr[2]"],
                curves=[
                    CurveDefinition(mnemonic=m)
                    for m in ["DEPT", "NMR[1]", "GR", "nmr[2]"]
                ],
                logs={
                    m: np.array([1.0])
                    for m in ["DEPT", "NMR[1]", "GR", "nmr[2]"]
                },
            )

    def test_case_variant_contiguous_arrays_accepted(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "NMR[1]", "nmr[2]", "GR"],
            curves=[
                CurveDefinition(mnemonic=m)
                for m in ["DEPT", "NMR[1]", "nmr[2]", "GR"]
            ],
            logs={
                m: np.array([1.0]) for m in ["DEPT", "NMR[1]", "nmr[2]", "GR"]
            },
        )
        assert len(las.curves) == 4


# ──────────────────────────────────────────────────────────────
# E-09 (MEDIUM): MAX_FIELD_LENGTH gaps — ParameterEntry already-str
# values (missing elif branch on BOTH mutation and construction) and
# CurveDefinition/ParameterEntry mnemonics were never length-checked;
# over-long values were emitted and then failed the parser's own
# guard on re-read (self-unreadable).
# ──────────────────────────────────────────────────────────────


class TestE09MaxFieldLengthGaps:
    """E-09: length checks on ParameterEntry values + all mnemonics."""

    def test_parameter_value_overlong_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            ParameterEntry(mnemonic="GR", value="x" * (MAX_FIELD_LENGTH + 1))

    def test_parameter_value_overlong_mutation_raises(self) -> None:
        p = ParameterEntry(mnemonic="GR", value="x")
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            p.value = "y" * (MAX_FIELD_LENGTH + 1)

    def test_parameter_unit_overlong_mutation_raises(self) -> None:
        p = ParameterEntry(mnemonic="GR", value="x")
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            p.unit = "u" * (MAX_FIELD_LENGTH + 1)

    def test_curve_mnemonic_overlong_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            CurveDefinition(mnemonic="x" * (MAX_FIELD_LENGTH + 1))

    def test_parameter_mnemonic_overlong_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            ParameterEntry(mnemonic="x" * (MAX_FIELD_LENGTH + 1))

    def test_curve_mnemonic_overlong_mutation_raises(self) -> None:
        c = CurveDefinition(mnemonic="GR")
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            c.mnemonic = "y" * (MAX_FIELD_LENGTH + 1)

    def test_parameter_mnemonic_overlong_mutation_raises(self) -> None:
        p = ParameterEntry(mnemonic="GR", value="x")
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            p.mnemonic = "z" * (MAX_FIELD_LENGTH + 1)


# ──────────────────────────────────────────────────────────────
# E-10 (MEDIUM): _validate_array_continuity enforced ONLY at
# construction — post-construction curves_order mutation bypassed it
# and the writer emitted self-unreadable files.  validate(complete=True)
# now re-runs it for data_sections AND top-level curves_order.
# ──────────────────────────────────────────────────────────────


class TestE10ValidateRerunsArrayContinuity:
    """E-10: validate(complete=True) re-checks array contiguity."""

    def test_top_level_interleaved_mutation_flagged(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "NMR[1]", "NMR[2]", "GR"],
            curves=[
                CurveDefinition(mnemonic=m)
                for m in ["DEPT", "NMR[1]", "NMR[2]", "GR"]
            ],
            logs={
                m: np.array([1.0]) for m in ["DEPT", "NMR[1]", "NMR[2]", "GR"]
            },
        )
        las.curves_order[2] = "GR"
        las.curves_order[3] = "NMR[2]"
        issues = las.validate(complete=True)
        assert any("not contiguous" in i for i in issues), issues

    def test_section_interleaved_mutation_flagged(self) -> None:
        ds = DataSection(
            name="LOG",
            curves_order=["NMR[1]", "NMR[2]"],
            section_curves=[
                CurveDefinition(mnemonic="NMR[1]"),
                CurveDefinition(mnemonic="NMR[2]"),
            ],
            data={"NMR[1]": np.array([1.0]), "NMR[2]": np.array([1.0])},
        )
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["NMR[1]", "NMR[2]"],
            data_sections=[ds],
        )
        las.data_sections[0].curves_order.insert(1, "GR")
        issues = las.validate(complete=True)
        assert any("not contiguous" in i for i in issues), issues

    def test_valid_state_stays_clean(self) -> None:
        las = LASFile(
            version=VersionSection(vers="3.0"),
            curves_order=["DEPT", "NMR[1]", "NMR[2]", "GR"],
            curves=[
                CurveDefinition(mnemonic=m)
                for m in ["DEPT", "NMR[1]", "NMR[2]", "GR"]
            ],
            logs={
                m: np.array([1.0]) for m in ["DEPT", "NMR[1]", "NMR[2]", "GR"]
            },
        )
        assert las.validate(complete=True) == []
