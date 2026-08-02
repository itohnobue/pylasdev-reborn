"""Coverage-gap tests for the TEST gate.

The production-check FIX stage changed 9 source files; the existing suite
is fully green (1135 passed / 0 failed / 1 intentional skip) but the
coverage gate (85%) was at 83.75%.  These tests exercise the uncovered
edge-case branches in the fixed modules — error paths, guard clauses,
and invariant checks that the FIX stage added or hardened but that had no
direct test coverage.  Each test asserts real behavior (no coverage-only
probes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev._data_section_reader import (
    _get_section_word,
    _is_recognized_section_word,
)
from pylasdev._las30_data import (
    _NULL_FILL_CELLS_ATTR,
    _NULL_FILL_SENTINEL_ATTR,
    _NULL_LOGS_OWNER_ATTR,
    AsciiDataContext,
    _build_spec_form_array_info,
    _deduplicate_curves,
    _detect_actual_wrap_las30,
    _reconcile_null_sentinels,
    _spec_form_group_data_is_numeric,
)
from pylasdev._parser_state import _ParserState
from pylasdev.compare import (
    _coerce_and_compare,
    _compare_data_sections,
    _compare_lists,
    _has_nan,
    _list_to_numeric_array,
    _scalars_equal,
    compare_las_dicts,
)
from pylasdev.encoding import read_with_encoding
from pylasdev.exceptions import LASEncodingError
from pylasdev.models import (
    ArrayElementInfo,
    CurveDefinition,
    DataSection,
    LASFile,
    _coerce_numpy_scalar,
    _data_is_integral,
    _safe_str,
)

# ──────────────────────────────────────────────────────────────
# compare.py — scalar/array edge-case guards
# ──────────────────────────────────────────────────────────────


class TestScalarsEqualGuards:
    """_scalars_equal ndarray size guards and fallbacks."""

    def test_empty_ndarray_first_operand(self) -> None:
        assert _scalars_equal(np.array([]), 5) is False

    def test_multi_element_ndarray_first_operand(self) -> None:
        assert _scalars_equal(np.array([1.0, 2.0]), 5) is False

    def test_empty_ndarray_second_operand(self) -> None:
        assert _scalars_equal(5, np.array([])) is False

    def test_multi_element_ndarray_second_operand(self) -> None:
        assert _scalars_equal(5, np.array([1.0, 2.0])) is False

    def test_nan_equal_nan(self) -> None:
        assert _scalars_equal(np.nan, np.nan) is True

    def test_type_error_fallback_false(self) -> None:
        # object() comparison raises TypeError -> caught -> False
        assert _scalars_equal(object(), object()) is False


class TestHasNanGuards:
    """_has_nan masked-array and TypeError branches."""

    def test_masked_array_with_mask_is_nan(self) -> None:
        ma = np.ma.array([1.0, 2.0], mask=[False, True])
        assert _has_nan(ma) is True

    def test_masked_array_no_mask(self) -> None:
        ma = np.ma.array([1.0, 2.0])
        assert _has_nan(ma) is False

    def test_object_dtype_returns_false(self) -> None:
        assert _has_nan(np.array(["a", "b"], dtype=object)) is False

    def test_string_dtype_returns_false(self) -> None:
        assert _has_nan(np.array(["a", "b"], dtype="U")) is False

    def test_list_of_dicts_with_nan(self) -> None:
        assert _has_nan([{"a": np.nan}]) is True


class TestListToNumericArrayGuards:
    """_list_to_numeric_array rejection branches."""

    def test_empty_list_returns_none(self) -> None:
        assert _list_to_numeric_array([]) is None

    def test_non_numeric_ndarray_item_returns_none(self) -> None:
        assert _list_to_numeric_array([np.array(["a"])]) is None

    def test_none_element_returns_none(self) -> None:
        assert _list_to_numeric_array([1.0, None]) is None

    def test_dict_element_returns_none(self) -> None:
        assert _list_to_numeric_array([{"a": 1}]) is None

    def test_numeric_list_converts(self) -> None:
        arr = _list_to_numeric_array([1, 2.5])
        assert arr is not None and arr.dtype.kind == "f"

    def test_masked_element_filled_nan(self) -> None:
        arr = _list_to_numeric_array([np.ma.array(1.0, mask=True)])
        assert arr is not None and np.isnan(arr[0])


class TestCompareListsEdgeCases:
    """_compare_lists fallback branches (non-numeric lists)."""

    def test_length_mismatch_via_per_element_path(self) -> None:
        # list with a dict element -> _list_to_numeric_array returns None ->
        # _has_nan(dict) False -> l1 != l2 shortcut False -> per-element path
        assert _compare_lists([{"a": 1}], [{"a": 1}, {"b": 2}], "x", 1e-7, 0.0) is False

    def test_type_mismatch_non_list_first(self) -> None:
        # l1 is a dict (not a list) whose value contains NaN -> _has_nan
        # raises ValueError -> except branch -> non-list type-mismatch warning.
        assert _compare_lists({"a": np.nan}, [np.nan], "x", 1e-7, 0.0) is False

    def test_list_of_dicts_equal(self) -> None:
        assert _compare_lists([{"a": 1}], [{"a": 1}], "x", 1e-7, 0.0) is True

    def test_length_mismatch_with_nan_routes_to_len_check(self) -> None:
        # Both lists contain a NaN and a string -> _list_to_numeric_array
        # returns None for both -> _has_nan raises -> except -> length check.
        assert _compare_lists(["a", np.nan], ["a"], "x", 1e-7, 0.0) is False


class TestCompareDataSectionsGuards:
    """_compare_data_sections symmetric type guards."""

    def test_sections1_not_list(self) -> None:
        assert _compare_data_sections(None, [], 1e-7, 0.0) is False

    def test_sections2_not_list(self) -> None:
        assert _compare_data_sections([], {}, 1e-7, 0.0) is False

    def test_keys_only_in_first_section_dict(self) -> None:
        sections1 = [{"a": 1.0, "b": 2.0}]
        sections2 = [{"a": 1.0}]
        assert _compare_data_sections(sections1, sections2, 1e-7, 0.0) is False

    def test_keys_only_in_second_section_dict(self) -> None:
        sections1 = [{"a": 1.0}]
        sections2 = [{"a": 1.0, "b": 2.0}]
        assert _compare_data_sections(sections1, sections2, 1e-7, 0.0) is False


class TestCoerceAndCompareMasked:
    """_coerce_and_compare 0-d MaskedArray coercion."""

    def test_0d_masked_scalar_vs_scalar_mismatch(self) -> None:
        # masked 0-d -> np.ma.masked -> type mismatch with plain scalar
        ma = np.ma.array(42.0, mask=True)
        assert _coerce_and_compare(ma, 42.0, "x", 1e-7, 0.0) is False

    def test_0d_masked_vs_masked_equal(self) -> None:
        ma1 = np.ma.array(42.0, mask=True)
        ma2 = np.ma.array(99.0, mask=True)
        assert _coerce_and_compare(ma1, ma2, "x", 1e-7, 0.0) is True


class TestCompareArraysGuardBranches:
    """_compare_arrays type/shape guard branches via public API."""

    def test_scalar_vs_ndarray_top_level(self) -> None:
        assert compare_las_dicts({"d": 5}, {"d": np.array([1.0])}) is False

    def test_non_numeric_dtype_arrays_equal(self) -> None:
        d1 = {"d": np.array(["a", "b"], dtype="S")}
        d2 = {"d": np.array([b"a", b"b"], dtype="S")}
        assert compare_las_dicts(d1, d2) is True

    def test_0d_ndarray_scalar_matches_scalar(self) -> None:
        assert compare_las_dicts({"d": np.array(42.0)}, {"d": 42.0}) is True


# ──────────────────────────────────────────────────────────────
# _parser_state.py — validate() invariant checks
# ──────────────────────────────────────────────────────────────


class TestParserStateValidate:
    """_ParserState.validate() invariant detection."""

    def _las(self) -> LASFile:
        return LASFile()

    def test_las30_sections_seen_but_not_las30(self) -> None:
        st = _ParserState(las30_sections_seen=True)
        issues = st.validate(self._las())
        assert any("LAS 3.0 structured data sections found" in i for i in issues)

    def test_data_sections_present_but_not_las30(self) -> None:
        las = self._las()
        las.data_sections.append(DataSection(name="S", section_type="LOG_DATA"))
        st = _ParserState()
        issues = st.validate(las)
        assert any("but is_las30 is False" in i for i in issues)

    def test_dangling_ascii_data_lines(self) -> None:
        st = _ParserState(current_data_section_idx=1, ascii_data_lines=["1 2"])
        issues = st.validate(self._las())
        assert any("dangling data" in i for i in issues)

    def test_deferred_well_entries_leftover(self) -> None:
        st = _ParserState(deferred_well_entries=[{"STRT": "100"}])
        issues = st.validate(self._las())
        assert any("deferred_well_entries still contains" in i for i in issues)

    def test_deferred_ascii_leftover(self) -> None:
        st = _ParserState(deferred_ascii_data_lines=[("a", "b", 1, "c", 2, None)])
        issues = st.validate(self._las())
        assert any("deferred_ascii_data_lines still contains" in i for i in issues)

    def test_section_sequences_out_of_sync(self) -> None:
        st = _ParserState(section_sequence=["A"], section_type_sequence=[])
        issues = st.validate(self._las())
        assert any("out of sync" in i for i in issues)

    def test_cumulative_counter_warning_flag_inconsistent(self) -> None:
        st = _ParserState(
            cumulative_data_lines=5,
            cumulative_data_lines_warned=True,
            current_data_section_idx=0,
        )
        issues = st.validate(self._las())
        assert any("warning threshold requires multiple sections" in i for i in issues)

    def test_main_curve_end_below_minus_one(self) -> None:
        st = _ParserState(main_curve_end=-2)
        issues = st.validate(self._las())
        assert any("main_curve_end is -2" in i for i in issues)

    def test_main_curve_end_exceeds_curves(self) -> None:
        las = self._las()
        las.curves.append(CurveDefinition(mnemonic="DEPT"))
        st = _ParserState(main_curve_end=5)
        issues = st.validate(las)
        assert any("index exceeds curve count" in i for i in issues)

    def test_section_curve_start_idx_negative(self) -> None:
        st = _ParserState(section_curve_start_idx=-1)
        issues = st.validate(self._las())
        assert any("section_curve_start_idx is -1" in i for i in issues)

    def test_section_curve_end_idx_negative(self) -> None:
        st = _ParserState(section_curve_end_idx=-1)
        issues = st.validate(self._las())
        assert any("section_curve_end_idx is -1" in i for i in issues)

    def test_definition_curve_range_start_negative(self) -> None:
        st = _ParserState(definition_curve_ranges={"D1": (-1, 2)})
        issues = st.validate(self._las())
        assert any("start=-1 is negative" in i for i in issues)

    def test_definition_curve_range_start_gt_end(self) -> None:
        st = _ParserState(definition_curve_ranges={"D1": (3, 1)})
        issues = st.validate(self._las())
        assert any("must be non-decreasing" in i for i in issues)

    def test_definition_curve_range_end_exceeds_curves(self) -> None:
        st = _ParserState(definition_curve_ranges={"D1": (0, 5)})
        issues = st.validate(self._las())
        assert any("exceeds 0 curves" in i for i in issues)

    def test_clean_state_no_issues(self) -> None:
        assert _ParserState().validate(self._las()) == []


# ──────────────────────────────────────────────────────────────
# _las30_data.py — dedup / wrap detection / spec-form / null reconcile
# ──────────────────────────────────────────────────────────────


def _ctx(
    las: LASFile,
    data_lines: list[str] | None = None,
    start: int = 0,
    end: int | None = None,
) -> AsciiDataContext:
    return AsciiDataContext(
        las_file=las,
        ascii_data_lines=data_lines or [],
        section_curve_start_idx=start,
        section_curve_end_idx=end,
        current_section_name="ASCII",
        current_data_section_type="LOG_DATA",
        current_data_section_idx=0,
    )


class TestDeduplicateCurvesEdges:
    """_deduplicate_curves collision and writeback edge cases."""

    def test_second_collision_while_loop_used(self) -> None:
        # X, X_2, X_2 collision -> X_3 (both while loops exercised)
        las = LASFile()
        las.curves = [
            CurveDefinition(mnemonic="X"),
            CurveDefinition(mnemonic="X"),
            CurveDefinition(mnemonic="X"),
        ]
        las.curves_order = ["X", "X", "X"]
        ctx = _ctx(las)
        sc = [
            CurveDefinition(mnemonic="X"),
            CurveDefinition(mnemonic="X"),
            CurveDefinition(mnemonic="X"),
        ]
        order = _deduplicate_curves(ctx, sc, is_first_section=True)
        assert order == ["X", "X_2", "X_3"]

    def test_first_section_global_writeback(self) -> None:
        las = LASFile()
        las.curves = [
            CurveDefinition(mnemonic="A"),
            CurveDefinition(mnemonic="A"),
        ]
        las.curves_order = ["A", "A"]
        ctx = _ctx(las)
        sc = [CurveDefinition(mnemonic="A"), CurveDefinition(mnemonic="A")]
        order = _deduplicate_curves(ctx, sc, is_first_section=True)
        assert order == ["A", "A_2"]
        assert las.curves[1].mnemonic == "A_2"
        assert las.curves_order == ["A", "A_2"]

    def test_second_collision_branch(self) -> None:
        # name in output_names but not in seen -> suffix=2 path
        las = LASFile()
        las.curves = [
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
        ]
        las.curves_order = ["B", "B", "B", "B"]
        ctx = _ctx(las)
        sc = [
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
            CurveDefinition(mnemonic="B"),
        ]
        order = _deduplicate_curves(ctx, sc, is_first_section=True)
        assert order == ["B", "B_2", "B_3", "B_4"]


class TestDetectActualWrapLas30Edges:
    """_detect_actual_wrap_las30 majority-vote branches."""

    def test_comment_and_blank_only(self) -> None:
        assert _detect_actual_wrap_las30(["# c", "   "], 2, " ", None) is False

    def test_declared_yes_but_complete_rows(self) -> None:
        assert _detect_actual_wrap_las30(["1 2", "3 4"], 2, " ", "YES") is False

    def test_declared_yes_with_partial_continuation(self) -> None:
        assert _detect_actual_wrap_las30(["1 2 3 4", "5"], 2, " ", "YES") is True

    def test_two_full_rows_not_wrapped(self) -> None:
        assert _detect_actual_wrap_las30(["1 2", "3 4"], 2, " ", None) is False

    def test_three_partial_rows_wrapped(self) -> None:
        assert _detect_actual_wrap_las30(["1", "2", "3"], 2, " ", None) is True

    def test_tie_uses_declared_wrap(self) -> None:
        assert _detect_actual_wrap_las30(["1", "2 3"], 2, " ", "YES") is True
        assert _detect_actual_wrap_las30(["1", "2 3"], 2, " ", "NO") is False

    def test_tie_no_declared_defaults_wrapped(self) -> None:
        assert _detect_actual_wrap_las30(["1", "2 3"], 2, " ", None) is True

    def test_uniform_short_rows_not_wrapped(self) -> None:
        assert _detect_actual_wrap_las30(["1 2", "3 4", "5 6"], 3, " ", None) is False


class TestSpecFormGroupDataIsNumeric:
    """_spec_form_group_data_is_numeric discriminator."""

    def test_numeric_group_true(self) -> None:
        assert _spec_form_group_data_is_numeric(["1.0 2.0"], " ", [0, 1]) is True

    def test_string_group_false(self) -> None:
        assert _spec_form_group_data_is_numeric(["SAND SHALE"], " ", [0, 1]) is False

    def test_no_data_false(self) -> None:
        assert _spec_form_group_data_is_numeric([], " ", [0, 1]) is False

    def test_none_delimiter_false(self) -> None:
        assert _spec_form_group_data_is_numeric(["1 2"], None, [0, 1]) is False

    def test_comment_and_blank_lines_skipped(self) -> None:
        assert _spec_form_group_data_is_numeric(["# c", "   ", "1.0 2.0"], " ", [0, 1]) is True

    def test_short_row_ignored(self) -> None:
        assert _spec_form_group_data_is_numeric(["1.0"], " ", [0, 1]) is True


class TestBuildSpecFormArrayInfo:
    """_build_spec_form_array_info synthesis and guards."""

    def _pair(self) -> list[CurveDefinition]:
        return [
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:0}"),
            CurveDefinition(mnemonic="NMR", unit="ms", data_format="A", description="Echo {A:5}"),
        ]

    def test_synthesis_with_numeric_data(self) -> None:
        out = _build_spec_form_array_info(self._pair(), ["1 2"], " ")
        assert [c.mnemonic for c in out] == ["NMR[1]", "NMR[2]"]
        assert out[0].array_info is not None
        assert out[0].array_info.index == 1
        assert out[0].array_info.time_offset == 0.0
        assert out[1].array_info.time_offset == 5.0

    def test_string_data_leaves_duplicates_untouched(self) -> None:
        out = _build_spec_form_array_info(self._pair(), ["SAND SHALE"], " ")
        assert [c.mnemonic for c in out] == ["NMR", "NMR"]

    def test_no_data_returns_unchanged(self) -> None:
        out = _build_spec_form_array_info(self._pair(), None, None)
        assert [c.mnemonic for c in out] == ["NMR", "NMR"]

    def test_single_group_member_untouched(self) -> None:
        sc = [CurveDefinition(mnemonic="GR", data_format="A")]
        out = _build_spec_form_array_info(sc, ["1"], " ")
        assert [c.mnemonic for c in out] == ["GR"]

    def test_invalid_offset_falls_back_none(self) -> None:
        sc = [
            CurveDefinition(mnemonic="NMR", data_format="A", description="Echo {A:zz}"),
            CurveDefinition(mnemonic="NMR", data_format="A", description="Echo {A:zz}"),
        ]
        out = _build_spec_form_array_info(sc, ["1 2"], " ")
        assert out[0].array_info is not None
        assert out[0].array_info.time_offset is None

    def test_interleaved_non_member_breaks_group(self) -> None:
        sc = [
            CurveDefinition(mnemonic="GR", data_format="F"),
            CurveDefinition(mnemonic="NMR", data_format="A"),
            CurveDefinition(mnemonic="NMR", data_format="A"),
        ]
        out = _build_spec_form_array_info(sc, ["1 2 3"], " ")
        # GR stays first; NMR pair synthesized
        assert out[0].mnemonic == "GR"
        assert out[1].mnemonic == "NMR[1]"
        assert out[2].mnemonic == "NMR[2]"

    def test_existing_array_info_members_skip_grouping(self) -> None:
        sc = [
            CurveDefinition(
                mnemonic="NMR[1]",
                data_format="A",
                array_info=ArrayElementInfo(base_name="NMR", index=1, time_offset=0.0),
            ),
            CurveDefinition(
                mnemonic="NMR",
                data_format="A",
            ),
        ]
        out = _build_spec_form_array_info(sc, ["1 2"], " ")
        # single remaining plain member -> no synthesis
        assert [c.mnemonic for c in out] == ["NMR[1]", "NMR"]


class TestReconcileNullSentinelsEdges:
    """_reconcile_null_sentinels guard branches."""

    def test_non_finite_declared_null_returns_early(self) -> None:
        las = LASFile()
        las.well["NULL"] = "nan"
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT"])
        ds.data["DEPT"] = np.array([-999.25, 1.0])
        setattr(ds, _NULL_FILL_CELLS_ATTR, [(0, 0)])
        setattr(ds, _NULL_FILL_SENTINEL_ATTR, -999.25)
        las.data_sections.append(ds)
        _reconcile_null_sentinels(las)
        assert ds.data["DEPT"].tolist() == [-999.25, 1.0]  # untouched

    def test_sentinel_equals_declared_clears_cells(self) -> None:
        las = LASFile()
        las.well["NULL"] = "-999.25"
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT"])
        ds.data["DEPT"] = np.array([-999.25])
        setattr(ds, _NULL_FILL_CELLS_ATTR, [(0, 0)])
        setattr(ds, _NULL_FILL_SENTINEL_ATTR, -999.25)
        las.data_sections.append(ds)
        _reconcile_null_sentinels(las)
        assert getattr(ds, _NULL_FILL_CELLS_ATTR) == []

    def test_col_out_of_range_skipped(self) -> None:
        las = LASFile()
        las.well["NULL"] = "-999.0"
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT"])
        ds.data["DEPT"] = np.array([-999.25])
        setattr(ds, _NULL_FILL_CELLS_ATTR, [(0, 5)])
        setattr(ds, _NULL_FILL_SENTINEL_ATTR, -999.25)
        las.data_sections.append(ds)
        _reconcile_null_sentinels(las)
        assert ds.data["DEPT"].tolist() == [-999.25]  # out-of-range col skipped

    def test_logs_owner_sync(self) -> None:
        las = LASFile()
        las.well["NULL"] = "-999.0"
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT"])
        ds.data["DEPT"] = np.array([-999.25])
        setattr(ds, _NULL_FILL_CELLS_ATTR, [(0, 0)])
        setattr(ds, _NULL_FILL_SENTINEL_ATTR, -999.25)
        setattr(ds, _NULL_LOGS_OWNER_ATTR, True)
        las.data_sections.append(ds)
        las.logs["DEPT"] = np.array([-999.25])
        _reconcile_null_sentinels(las)
        assert ds.data["DEPT"].tolist() == [-999.0]
        assert las.logs["DEPT"].tolist() == [-999.0]

    def test_non_owner_section_does_not_touch_logs(self) -> None:
        las = LASFile()
        las.well["NULL"] = "-999.0"
        ds = DataSection(name="S", section_type="LOG_DATA", curves_order=["DEPT"])
        ds.data["DEPT"] = np.array([-999.25])
        setattr(ds, _NULL_FILL_CELLS_ATTR, [(0, 0)])
        setattr(ds, _NULL_FILL_SENTINEL_ATTR, -999.25)
        # _NULL_LOGS_OWNER_ATTR not set -> False
        las.data_sections.append(ds)
        las.logs["DEPT"] = np.array([999.0])  # another section's logs
        _reconcile_null_sentinels(las)
        assert ds.data["DEPT"].tolist() == [-999.0]
        assert las.logs["DEPT"].tolist() == [999.0]  # untouched


# ──────────────────────────────────────────────────────────────
# _data_section_reader.py — section-word helpers
# ──────────────────────────────────────────────────────────────


class TestSectionWordHelpers:
    """_get_section_word / _is_recognized_section_word branches."""

    def test_no_match_returns_empty(self) -> None:
        assert _get_section_word("hello world") == ""

    def test_pipe_target_stripped(self) -> None:
        assert _get_section_word("~ASCII|CURVE") == "ASCII"

    def test_empty_word_not_recognized(self) -> None:
        assert _is_recognized_section_word("") is False

    def test_bracketed_word_stripped(self) -> None:
        assert _is_recognized_section_word("CORE[1]") is True

    def test_definition_suffix_recognized(self) -> None:
        assert _is_recognized_section_word("CORE_DEFINITION") is True

    def test_numbered_definition_suffix_recognized(self) -> None:
        assert _is_recognized_section_word("CORE_DEFINITION_2") is True

    def test_parameter_suffix_recognized(self) -> None:
        assert _is_recognized_section_word("CORE_PARAMETER") is True

    def test_data_suffix_recognized(self) -> None:
        assert _is_recognized_section_word("MUD_DATA") is True

    def test_unknown_word_not_recognized(self) -> None:
        assert _is_recognized_section_word("FOOBAR") is False


# ──────────────────────────────────────────────────────────────
# encoding.py — guard and BOM branches
# ──────────────────────────────────────────────────────────────


class TestReadWithEncodingGuards:
    """read_with_encoding guard and BOM paths."""

    def test_non_regular_file_raises(self, tmp_path: Path) -> None:
        # A directory is not a regular file.
        with pytest.raises(LASEncodingError):
            read_with_encoding(tmp_path)

    def test_negative_max_file_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_file_size"):
            read_with_encoding(Path("README.md"), max_file_size=-1)

    def test_utf32_le_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "u32le.las"
        p.write_bytes(b"\xff\xfe\x00\x00" + "Hello".encode("utf-32-le"))
        enc, content = read_with_encoding(p, max_file_size=100_000)
        assert enc == "utf-32-le"
        assert content == "Hello"

    def test_utf32_be_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "u32be.las"
        p.write_bytes(b"\x00\x00\xfe\xff" + "Hi".encode("utf-32-be"))
        enc, content = read_with_encoding(p, max_file_size=100_000)
        assert enc == "utf-32-be"
        assert content == "Hi"

    def test_utf16_le_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "u16le.las"
        p.write_bytes(b"\xff\xfe" + "Hello".encode("utf-16-le"))
        enc, content = read_with_encoding(p, max_file_size=100_000)
        assert enc == "utf-16-le"
        assert content == "Hello"

    def test_utf16_be_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "u16be.las"
        p.write_bytes(b"\xfe\xff" + "Hello".encode("utf-16-be"))
        enc, content = read_with_encoding(p, max_file_size=100_000)
        assert enc == "utf-16-be"
        assert content == "Hello"


# ──────────────────────────────────────────────────────────────
# models.py — _safe_str / _coerce_numpy_scalar / _data_is_integral
# ──────────────────────────────────────────────────────────────


class TestModelHelpers:
    """models.py helper edge cases."""

    def test_safe_str_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _safe_str(float("nan"))

    def test_safe_str_rejects_bytes(self) -> None:
        with pytest.raises(TypeError, match="Decode to str"):
            _safe_str(b"x")

    def test_safe_str_max_length(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum"):
            _safe_str("x" * 300, max_length=10)

    def test_coerce_numpy_scalars(self) -> None:
        assert _coerce_numpy_scalar(np.bool_(True)) is True
        assert _coerce_numpy_scalar(np.int64(5)) == 5
        assert _coerce_numpy_scalar(np.float32(1.5)) == 1.5
        assert _coerce_numpy_scalar("x") == "x"

    def test_data_is_integral_object_dtype(self) -> None:
        assert _data_is_integral(np.array([1, 2], dtype=object)) is True
        assert _data_is_integral(np.array([1.5], dtype=object)) is False
        assert _data_is_integral(np.array(["a"], dtype=object)) is False

    def test_data_is_integral_float_paths(self) -> None:
        assert _data_is_integral([1.0, 2.0]) is True
        assert _data_is_integral([1.5]) is False
        assert _data_is_integral([1.0, np.nan]) is False

    def test_data_is_integral_unconvertible_scalar(self) -> None:
        assert _data_is_integral([object()]) is False
