"""Tests for LAS data models."""

from __future__ import annotations

import warnings
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
            "curves": [
                {"mnemonic": "DEPT", "unit": None, "description": None}
            ],
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
        with pytest.raises(ValueError, match=r"inconsistent.*string_data.*lengths"):
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
            _resolve_dict_entry(
                {"time_offset": True}, "time_offset", (int, float), lambda: None
            )

        # int and float should still work
        assert _resolve_dict_entry({"index": 5}, "index", int, lambda: 0) == 5
        assert (
            _resolve_dict_entry(
                {"time_offset": 1.5}, "time_offset", (int, float), lambda: None
            )
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
        with pytest.raises(ValueError, match="index must be >= 0"):
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
        with pytest.raises(
            LASDataError, match=r"curve 'GR'.*data_format='F'.*in string_data"
        ):
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
                x for x in w
                if "non-numeric dtype" in str(x.message)
                and "DEPT" in str(x.message)
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
                x for x in w
                if "numeric dtype" in str(x.message)
                and "STR" in str(x.message)
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
            dtype_warnings = [
                x for x in w
                if "non-numeric dtype" in str(x.message)
            ]
        assert len(dtype_warnings) == 0, (
            f"Expected 0 dtype warnings for numeric data, got {len(dtype_warnings)}"
        )

    # --- F-059 (MEDIUM): LASFile data_format cross-validation ---

    def test_lasfile_s_format_in_logs_warns(self) -> None:
        """F-059: S-format curve in logs (numeric) triggers warning.

        LASFile.__post_init__ should warn when a curve with data_format='S'
        is placed in the logs dict (numeric storage).
        """
        sc = CurveDefinition(mnemonic="STR", data_format="S")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                well=WellSection(entries={"STRT": "0", "STOP": "100", "STEP": "10", "NULL": "-999"}),
                curves=[sc],
                curves_order=["STR"],
                logs={"STR": np.array([1.0, 2.0])},
            )
            s_format_warnings = [
                x for x in w
                if "string-format" in str(x.message)
                and "STR" in str(x.message)
            ]
        assert len(s_format_warnings) == 1, (
            f"Expected 1 S-format warning for STR, got {len(s_format_warnings)}"
        )

    def test_lasfile_numeric_format_in_string_data_warns(self) -> None:
        """F-059: Numeric-format curve in string_data triggers warning.

        LASFile.__post_init__ should warn when a numeric-format curve
        is placed in the string_data dict.
        """
        sc = CurveDefinition(mnemonic="GR", data_format="F")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LASFile(
                version=VersionSection(vers="2.0", wrap="NO", dlm="SPACE"),
                well=WellSection(entries={"STRT": "0", "STOP": "100", "STEP": "10", "NULL": "-999"}),
                curves=[sc],
                curves_order=["GR"],
                string_data={"GR": np.array(["a", "b"], dtype=str)},
            )
            num_format_warnings = [
                x for x in w
                if "numeric-format" in str(x.message)
                and "GR" in str(x.message)
            ]
        assert len(num_format_warnings) == 1, (
            f"Expected 1 numeric-format warning for GR, got {len(num_format_warnings)}"
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
                x for x in w
                if "top-level 'string_data'" in str(x.message)
                and "data_sections" in str(x.message)
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
