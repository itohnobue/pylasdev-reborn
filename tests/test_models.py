"""Tests for LAS data models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pylasdev.exceptions import LASDataError
from pylasdev.models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
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
        np.testing.assert_array_equal(
            rt_section.data["DEPT"], np.array([100.0])
        )

    # --- F-01 fix: MAX_PARAMETERS guard on parameter_details ---

    def test_from_dict_parameter_details_max_guard(self, monkeypatch) -> None:
        """LASFile.from_dict() enforces MAX_PARAMETERS on parameter_details (F-01 fix).

        parameter_details was the sole unguarded iterable in from_dict().
        The fix adds a len check matching the existing params guard at L428.
        """
        # Use a small limit for testing
        monkeypatch.setattr("pylasdev.parser.MAX_PARAMETERS", 5)

        big_details = [
            {"mnemonic": f"PARAM_{i}", "value": str(i)} for i in range(6)
        ]
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

        exact_details = [
            {"mnemonic": f"PARAM_{i}", "value": str(i)} for i in range(4)
        ]
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
        """LASFile.from_dict() handles None values in curve dict fields (F2-17 fix).

        curve_dict.get("mnemonic", "") returned stored None when key existed
        with value None. The fix wraps all string fields with _safe_str().
        """
        data: dict[str, Any] = {
            "version": {"VERS": "2.0", "WRAP": "NO", "DLM": "SPACE"},
            "well": {"STRT": "100"},
            "curves_order": [""],  # _safe_str(None) → "" matches this
            "curves": [
                {"mnemonic": None, "unit": None, "description": None}
            ],
            "logs": {"": np.array([100.0])},
        }
        las = LASFile.from_dict(data)
        # _safe_str(None) → "" for all string fields
        assert las.curves[0].mnemonic == ""
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
                    "section_curves": [
                        {"mnemonic": "DEPT", "unit": None, "data_format": None}
                    ],
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
        """
        monkeypatch.setattr("pylasdev.data_reader.MAX_CURVES", 3)

        str_curves = {f"STR_{i}": np.array(["a"]) for i in range(2)}
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
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
        with pytest.raises(ValueError, match="inconsistent array"):
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
        with pytest.raises(ValueError, match=r"inconsistent.*string_data"):
            LASFile.from_dict(data)

    # --- F-004: top-level string_data cross-array length ---

    def test_from_dict_top_level_string_data_inconsistent(self) -> None:
        """F-004: Top-level string_data with inconsistent array lengths raises ValueError.

        The top-level path (models.py:926-933) validates that all arrays in
        the top-level string_data dict have the same length.
        """
        data: dict[str, Any] = {
            "version": {"VERS": "3.0", "WRAP": "NO", "DLM": "COMMA"},
            "well": {"NULL": "-999.25"},
            "curves_order": ["DEPT"],
            "curves": [{"mnemonic": "DEPT"}],
            "logs": {"DEPT": np.array([100.0])},
            "string_data": {
                "STR1": np.array(["a", "b"]),
                "STR2": np.array(["c"]),
            },
        }
        with pytest.raises(ValueError, match="inconsistent lengths"):
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
        self, monkeypatch: pytest.MonkeyPatch,
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
            "string_data": {"CDES": np.array(["desc"])},
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
        """DevFile.from_dict() handles column_order as a string (F-04 fix).

        list("MD,TVD") previously produced ['M','D',',','T','V','D'].
        The fix wraps a string value in a single-element list.
        """
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0]),
            "TVD": np.array([0.0, 99.0]),
            "column_order": "MD,TVD",
        }
        dev = DevFile.from_dict(data)
        # String is wrapped in list; "MD,TVD" stays as a single element
        assert dev.column_order == ["MD,TVD"]

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
        """F-006: DevFile columns with inconsistent lengths raises ValueError.

        models.py:1078-1086 validates that all column arrays in DevFile have
        the same length.
        """
        data: dict[str, Any] = {
            "MD": np.array([0.0, 100.0, 200.0]),
            "TVD": np.array([0.0, 99.0]),
        }
        with pytest.raises(ValueError, match="inconsistent lengths"):
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
            meta_warnings = [
                x for x in w
                if "storing metadata as '_meta_" in str(x.message)
            ]
            assert len(meta_warnings) >= 1, (
                f"Expected _meta_ collision warning, got warnings: "
                f"{[str(x.message) for x in w]}"
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
        np.testing.assert_array_equal(
            dev2.columns["TVD"], np.array([0.0, 99.0, 198.0])
        )

        # R7F-01-gap fix: column "source_file" survives the roundtrip
        np.testing.assert_array_equal(
            dev2.columns["source_file"], np.array([0.0, 100.0, 200.0])
        )

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
            meta_warnings = [
                x for x in w
                if "storing metadata as '_meta_" in str(x.message)
            ]
            assert len(meta_warnings) >= 1

        assert "_meta_encoding" in d

        # Must not crash (the original HIGH bug)
        dev2 = DevFile.from_dict(d)
        assert dev2.encoding == "cp1251"
        assert dev2.source_file == "test.dev"
        # Non-colliding columns survive
        np.testing.assert_array_equal(
            dev2.columns["MD"], np.array([0.0, 100.0])
        )
        # R7F-01-gap fix: column "encoding" survives the roundtrip
        np.testing.assert_array_equal(
            dev2.columns["encoding"], np.array([10.0, 20.0])
        )

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
