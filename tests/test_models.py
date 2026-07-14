"""Tests for LAS data models."""

from __future__ import annotations

from typing import Any

import numpy as np

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
                "CDES": np.array([5.0, 6.0]),
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

    def test_section_curves_empty_roundtrip(self) -> None:
        """Test DataSection with empty section_curves survives roundtrip.

        Covers the edge case where section_curves=[] should serialize
        as [] and deserialize as [].
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
        assert rt_section.section_curves == []


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
