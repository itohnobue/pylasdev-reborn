"""Tests for the DEV (deviation survey) writer."""

from __future__ import annotations

import os
import re
import stat
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pylasdev import read_dev_file_as_object, write_dev_file
from pylasdev._writer_base import _format_number
from pylasdev.dev_reader import _DEV_NUMERIC_SENTINELS
from pylasdev.exceptions import DEVWriteError, LASDataError
from pylasdev.models import DevFile

TEST_DATA = Path(__file__).parent.parent / "test_data"

# Substrings of the READER's format-detection diagnostics — the only
# warnings AC-4 scores.  Data-quality diagnostics (MD monotonicity, NaN
# density, TVD reversals) are explicitly out of AC-4's scope.
_FORMAT_DIAGNOSTIC_MARKERS = (
    "Auto-detected headerless format",
    "Duplicate DEV column name",
    "Delimiter mismatch",
    "Delimiter consistency warning",
    "Extra columns",
    "Delimiter must be",
    "No data lines found",
    "fewer values than expected",
)


def _dev(columns: dict[str, Any], column_order: list[str] | None = None) -> DevFile:
    """Build a model without the model's own construction-time diagnostics.

    ``DevFile.__post_init__`` warns about data-quality issues in the test
    data (a missing MD column, non-finite cells, a repeated station).
    Those warnings come from the MODEL, not from the writer, so they are
    silenced here to keep each test's warning assertions about the writer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DevFile(columns=columns, column_order=column_order or [])


def _read(path: Path, **kwargs: Any) -> tuple[DevFile, list[str]]:
    """Read a DEV file and return it with the format-detection warnings seen."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dev = read_dev_file_as_object(path, **kwargs)
    messages = [str(w.message) for w in caught]
    format_diagnostics = [
        message
        for message in messages
        if any(marker in message for marker in _FORMAT_DIAGNOSTIC_MARKERS)
    ]
    return dev, format_diagnostics


def _read_with_aliases(path: Path) -> tuple[DevFile, list[str]]:
    """Re-read without alias normalization (H-02's preservation route)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dev = read_dev_file_as_object(path, normalize_aliases=False)
    return dev, [str(w.message) for w in caught]


def _write(path: Path, dev_data: Any, **kwargs: Any) -> list[str]:
    """Write a DEV file and return every warning the writer emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_dev_file(path, dev_data, **kwargs)
    return [str(w.message) for w in caught]


class TestDEVPublicAPI:
    """AC-1/AC-2: the public surface and its error contract."""

    def test_public_imports(self) -> None:
        import pylasdev

        assert "write_dev_file" in pylasdev.__all__
        assert "DEVWriteError" in pylasdev.__all__
        assert "write_dev_file()" in pylasdev.__doc__
        assert "DEVWriteError " in pylasdev.__doc__
        assert issubclass(pylasdev.DEVWriteError, pylasdev.PylasdevError)

    def test_returns_none_and_accepts_str_path(self, tmp_path: Path) -> None:
        target = tmp_path / "out.dev"
        result = write_dev_file(str(target), _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}))
        assert result is None
        assert target.exists()

    def test_accepts_dict_with_alias_normalization(self, tmp_path: Path) -> None:
        target = tmp_path / "dict.dev"
        _write(target, {"MDKB": [0.0, 10.0], "TVDSS": [0.0, -10.0]})
        assert target.read_text().splitlines()[0] == "MD TVD"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "out.dev"
        write_dev_file(target, _dev({"MD": [0.0], "TVD": [1.0]}))
        assert target.exists()

    @pytest.mark.parametrize(
        ("dev_data", "match"),
        [
            (["MD"], "expects a DevFile or dict"),
            (None, "expects a DevFile or dict"),
            (42, "expects a DevFile or dict"),
        ],
    )
    def test_rejects_non_devfile_input(self, tmp_path: Path, dev_data: Any, match: str) -> None:
        with pytest.raises(DEVWriteError, match=match):
            write_dev_file(tmp_path / "x.dev", dev_data)

    def test_rejects_invalid_dict(self, tmp_path: Path) -> None:
        with pytest.raises(DEVWriteError, match="Cannot create DevFile from dict"):
            write_dev_file(tmp_path / "x.dev", {"MD": [1.0], "TVD": [1.0, 2.0]})

    @pytest.mark.parametrize("precision", [8, None, b".8g"])
    def test_rejects_non_str_precision(self, tmp_path: Path, precision: Any) -> None:
        with pytest.raises(DEVWriteError, match="precision must be a string"):
            write_dev_file(tmp_path / "x.dev", _dev({"MD": [0.0]}), precision=precision)

    @pytest.mark.parametrize("precision", [".8x", ".8d", "", ".8g.2f"])
    def test_rejects_unparseable_precision(self, tmp_path: Path, precision: str) -> None:
        with pytest.raises(DEVWriteError, match="Invalid precision"):
            write_dev_file(tmp_path / "x.dev", _dev({"MD": [0.0]}), precision=precision)

    @pytest.mark.parametrize("precision", [".2%", ".3n"])
    def test_rejects_locale_and_percent_precision_without_warning_first(
        self, tmp_path: Path, precision: str
    ) -> None:
        # R-09: the LAS helper only WARNS for these codes; the DEV entry
        # point must refuse them, and must not emit the helper's warning
        # first (a warn-then-raise double signal).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(DEVWriteError, match="not supported for DEV output"):
                write_dev_file(tmp_path / "x.dev", _dev({"MD": [0.0]}), precision=precision)
        assert [str(w.message) for w in caught] == []

    @pytest.mark.parametrize("delimiter", [",,", "", "\n", "\r", ";,", 7])
    def test_rejects_invalid_delimiter(self, tmp_path: Path, delimiter: Any) -> None:
        with pytest.raises(DEVWriteError, match="delimiter must be a single character"):
            write_dev_file(tmp_path / "x.dev", _dev({"MD": [0.0]}), delimiter=delimiter)

    @pytest.mark.parametrize("delimiter", ["\x00", "\x01", "\x7f"])
    def test_rejects_a_delimiter_the_reader_deletes(self, tmp_path: Path, delimiter: str) -> None:
        """C-2: the reader deletes this class, so header and rows would fuse."""
        target = tmp_path / "deleted.dev"
        with pytest.raises(DEVWriteError) as excinfo:
            write_dev_file(
                target, _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}), delimiter=delimiter
            )
        message = str(excinfo.value)
        assert repr(delimiter) in message
        assert f"U+{ord(delimiter):04X}" in message
        assert "deletes that control character" in message
        assert not target.exists()

    @pytest.mark.parametrize("delimiter", [" ", ",", ";", "\t"])
    def test_accepts_delimiters_the_reader_keeps(self, tmp_path: Path, delimiter: str) -> None:
        """C-2: the four auto-detected separators still write and round-trip."""
        target = tmp_path / "kept.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}), delimiter=delimiter)
        reread, format_diagnostics = _read(target, delimiter=delimiter)
        assert format_diagnostics == []
        assert list(reread.column_order) == ["MD", "TVD"]
        np.testing.assert_array_equal(reread.columns["TVD"], [0.0, 1.0])

    @pytest.mark.parametrize("delimiter", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e"])
    def test_accepts_control_delimiters_the_reader_turns_into_a_space(
        self, tmp_path: Path, delimiter: str
    ) -> None:
        """C-2: splitlines()-class controls are NOT in the delete class.

        The reader substitutes them with a space before tokenizing, so their
        round-trip is through the default read rather than an explicit one.
        """
        target = tmp_path / "spaceclass.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}), delimiter=delimiter)
        reread, _ = _read(target)
        assert list(reread.column_order) == ["MD", "TVD"]
        np.testing.assert_array_equal(reread.columns["TVD"], [0.0, 1.0])

    def test_unwritable_path_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(DEVWriteError, match="Cannot create output directory"):
            write_dev_file(blocker / "nested" / "x.dev", _dev({"MD": [0.0]}))

    def test_unknown_encoding_raises_and_leaves_no_temp_file(self, tmp_path: Path) -> None:
        target = tmp_path / "enc.dev"
        with pytest.raises(DEVWriteError, match="Cannot write to"):
            write_dev_file(target, _dev({"MD": [0.0], "TVD": [1.0]}), encoding="no-such-codec")
        assert not target.exists()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp_")] == []


class TestDEVRoundTripSample:
    """AC-3/AC-4: `test_data/sample.dev` round-trips through the writer."""

    @staticmethod
    def _source() -> DevFile:
        dev, format_diagnostics = _read(TEST_DATA / "sample.dev")
        assert format_diagnostics == []
        return dev

    @pytest.mark.parametrize("delimiter", [" ", ",", "\t"], ids=["space", "comma", "tab"])
    def test_sample_roundtrip_preserves_order_and_values(
        self, tmp_path: Path, delimiter: str
    ) -> None:
        """AC-3: identical column_order; index exact; data at token precision.

        The measured baseline on this fixture at ``.8g`` is 341/1490 cells
        changing (AZI 141 + INC 144 + DLS 56) — eight SIGNIFICANT digits vs
        the export's eight DECIMALS.  That is the amended AC-3 tolerance:
        every non-index cell must equal the value its own emitted token
        carries.
        """
        source = self._source()
        target = tmp_path / f"rt{'-' if delimiter != ' ' else ''}.dev"
        assert _write(target, source, delimiter=delimiter) == []

        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert list(reread.column_order) == list(source.column_order)

        changed = 0
        for name in source.column_order:
            assert name in reread.columns
            original = source.columns[name]
            actual = reread.columns[name]
            assert len(actual) == len(original)
            for before, after in zip(original, actual, strict=True):
                expected = float(_format_number(float(before), ".8g", -999.25))
                assert not np.isnan(before)
                assert float(after) == expected, (name, float(before), float(after), expected)
                if float(after) != float(before):
                    changed += 1
        assert changed == 341

    @pytest.mark.parametrize("delimiter", [" ", ",", "\t"], ids=["space", "comma", "tab"])
    def test_sample_roundtrip_is_exact_on_the_index_column(
        self, tmp_path: Path, delimiter: str
    ) -> None:
        """AC-3: the index column is never rounded at the default precision."""
        source = self._source()
        target = tmp_path / "idx.dev"
        _write(target, source, delimiter=delimiter)
        reread, _ = _read(target)
        np.testing.assert_array_equal(reread.columns["MD"], source.columns["MD"])

    def test_emitted_layout_is_simple_header_plus_one_row_per_station(self, tmp_path: Path) -> None:
        """AC-4: one header line, one row per station, no extra bare line."""
        source = self._source()
        target = tmp_path / "layout.dev"
        _write(target, source)
        text = target.read_text(encoding="utf-8")
        assert "\r" not in text
        assert text.endswith("\n")
        lines = text.splitlines()
        assert lines[0] == "MD X Y Z TVD DX DY AZI INC DLS"
        assert len(lines) == 1 + 149
        for line in lines[1:]:
            assert len(line.split(" ")) == 10
            assert "" not in line.split(" ")
        assert not any(line.lstrip().startswith("#") for line in lines)
        assert not any(line.lstrip().startswith("sep=") for line in lines)

    def test_index_value_keeps_nine_significant_digits(self, tmp_path: Path) -> None:
        """AC-3: `_format_index_value` on column 0 (`1234.56789` at `.8g`)."""
        target = tmp_path / "idx9.dev"
        _write(target, _dev({"MD": [1234.56789, 1235.56789], "INC": [1.0, 2.0]}))
        lines = target.read_text().splitlines()
        assert lines[1].split(" ")[0] == "1234.56789"
        reread, _ = _read(target)
        np.testing.assert_array_equal(reread.columns["MD"], np.array([1234.56789, 1235.56789]))


class TestDEVTokenShape:
    """R-03/R-04: no emitted token is a bare integer."""

    def test_one_column_roundtrip(self, tmp_path: Path) -> None:
        """R-04: a one-column file's header is not eaten as a title."""
        target = tmp_path / "one.dev"
        _write(target, _dev({"AZIMUTH_MSL": [1, 6, 7, 8]}))
        assert target.read_text() == "AZIMUTH_MSL\n1.0\n6.0\n7.0\n8.0\n"
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert list(reread.column_order) == ["AZIMUTH_MSL"]
        np.testing.assert_array_equal(reread.columns["AZIMUTH_MSL"], [1.0, 6.0, 7.0, 8.0])

    @pytest.mark.parametrize("delimiter", [" ", ",", "\t"], ids=["space", "comma", "tab"])
    def test_every_data_token_carries_a_decimal_point(self, tmp_path: Path, delimiter: str) -> None:
        """R-03: a bare integer feeds the comma thousands recombination."""
        target = tmp_path / "ints.dev"
        _write(target, _dev({"MD": [12, 13346], "TVD": [345.6, -82]}), delimiter=delimiter)
        for line in target.read_text().splitlines()[1:]:
            for token in line.split(delimiter):
                assert re.fullmatch(r"[+-]?\d+\.\d+", token), token

    def test_comma_mode_does_not_recombine_a_thousands_pair(self, tmp_path: Path) -> None:
        """R-03: `MD=12, TVD=345.6` must not re-read as `MD=12345.6, TVD=NaN`."""
        target = tmp_path / "merge.dev"
        _write(target, _dev({"MD": [12, 13346], "TVD": [345.6, 345.7]}), delimiter=",")
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        np.testing.assert_array_equal(reread.columns["MD"], [12.0, 13346.0])
        np.testing.assert_array_equal(reread.columns["TVD"], [345.6, 345.7])

    def test_single_row_survey_roundtrips(self, tmp_path: Path) -> None:
        """A one-station file must not be read as a count-prefix header."""
        target = tmp_path / "single.dev"
        _write(target, _dev({"MD": [0.0], "TVD": [1.0], "AZI": [45.0]}))
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert list(reread.column_order) == ["MD", "TVD", "AZI"]
        assert len(reread.columns["MD"]) == 1

    def test_entirely_missing_column_stays_readable(self, tmp_path: Path) -> None:
        """R-07: a >50 %-missing column re-reads with the density warning."""
        target = tmp_path / "allnull.dev"
        _write(
            target,
            _dev({"MD": [0.0, 10.0, 20.0], "TVD": np.array([np.nan] * 3)}),
        )
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert len(reread.columns["MD"]) == 3
        assert np.isnan(reread.columns["TVD"]).all()

    def test_non_autodetected_delimiter_roundtrips_with_an_explicit_delimiter(
        self, tmp_path: Path
    ) -> None:
        """A '|' file is readable, but only with the same explicit delimiter.

        The reader auto-detects space, tab, comma and semicolon only; any
        other separator has to be declared on both sides.  The writer emits
        exactly what it is asked for, so the pair stays symmetric.
        """
        target = tmp_path / "pipe.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}), delimiter="|")
        assert target.read_text() == "MD|TVD\n0.0|0.0\n10.0|1.0\n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reread = read_dev_file_as_object(target, delimiter="|")
        assert list(reread.column_order) == ["MD", "TVD"]
        np.testing.assert_array_equal(reread.columns["MD"], [0.0, 10.0])


class TestDEVNumericEmission:
    """AC-5: precision, null handling and sentinel collisions."""

    def test_precision_is_honoured(self, tmp_path: Path) -> None:
        target = tmp_path / "p2f.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "INC": [1.5, 2.25]}), precision=".2f")
        assert target.read_text() == "MD INC\n0.00 1.50\n10.00 2.25\n"

    def test_bare_precision_is_normalized(self, tmp_path: Path) -> None:
        target = tmp_path / "p5.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "INC": [1.5, 2.25]}), precision=".6")
        assert target.read_text() == "MD INC\n0.0 1.5\n10.0 2.25\n"

    def test_negative_zero_keeps_its_sign(self, tmp_path: Path) -> None:
        target = tmp_path / "negzero.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "X": np.array([-0.0, 1.0])}))
        assert target.read_text().splitlines()[1].split(" ")[1] == "-0.0"
        reread, _ = _read(target)
        assert np.signbit(reread.columns["X"][0])

    def test_inexact_index_at_an_explicit_precision_is_warned(self, tmp_path: Path) -> None:
        """C-4: an explicit precision may round a depth — say so, keep the precision."""
        target = tmp_path / "roundedidx.dev"
        messages = _write(target, _dev({"MD": [1234.56789]}), precision=".5g")
        index_warnings = [message for message in messages if "Index curve" in message]
        assert len(index_warnings) == 1
        assert "'MD'" in index_warnings[0]
        assert "'1234.6'" in index_warnings[0]
        assert "1234.56789" in index_warnings[0]
        # the caller's precision is honoured, never overridden
        assert target.read_text() == "MD\n1234.6\n"

    def test_inexact_index_warning_does_not_cite_the_las_specification(
        self, tmp_path: Path
    ) -> None:
        """The DEV warning is format-true: no LAS authority inside a DEV write.

        ``_writer_base._warn_inexact_index``'s message asserts that CWLS
        LAS 2.0 §5.4 forbids rounding depth values.  A DEV file has no
        specification of its own, so that citation claims a conformance
        requirement the artifact cannot carry; the DEV obligation is only to
        name the column, the source value and the emitted token.
        """
        target = tmp_path / "roundedidxmsg.dev"
        messages = _write(target, _dev({"MD": [1234.56789]}), precision=".5g")
        assert len(messages) == 1
        message = messages[0]
        assert "CWLS" not in message
        assert "LAS" not in message
        assert "§5.4" not in message
        assert "'MD'" in message
        assert "1234.56789" in message
        assert "'1234.6'" in message
        assert target.read_text() == "MD\n1234.6\n"

    def test_exact_index_at_the_default_precision_emits_no_warning(self, tmp_path: Path) -> None:
        """C-4: at `.8g` the index is exact, so there is nothing to warn about."""
        target = tmp_path / "exactidx.dev"
        assert _write(target, _dev({"MD": [1234.56789]})) == []
        assert target.read_text() == "MD\n1234.56789\n"

    def test_non_index_rounding_never_raises_the_index_warning(self, tmp_path: Path) -> None:
        """C-4: only the index column carries the index-rounding obligation."""
        target = tmp_path / "nonindexidx.dev"
        messages = _write(
            target, _dev({"MD": [0.0, 10.0], "TVD": [1234.56789, 2.0]}), precision=".5g"
        )
        assert not [message for message in messages if "Index curve" in message]
        assert target.read_text() == "MD TVD\n0.0 1234.6\n10.0 2.0\n"

    def test_nan_cell_becomes_the_numeric_sentinel(self, tmp_path: Path) -> None:
        target = tmp_path / "nan.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "TVD": np.array([0.0, np.nan])}))
        assert target.read_text() == "MD TVD\n0.0 0.0\n10.0 -999.25\n"

    def test_nan_index_cell_is_numeric_not_text(self, tmp_path: Path) -> None:
        """R-08: an alphabetic null token promotes the first row to a header.

        ``_format_index_value``'s exactness fallback is ``repr(value)``,
        which turns a missing index cell into the literal ``nan``; the
        writer routes non-finite index cells through the null formatter, so
        the header survives.
        """
        target = tmp_path / "nanmd.dev"
        _write(
            target,
            _dev(
                {
                    "AZIMUTH_MSL_LONG": np.array([np.nan, 1.0, 2.0]),
                    "TVD": np.array([np.nan, 1.0, 2.0]),
                }
            ),
        )
        assert target.read_text().splitlines()[1] == "-999.25 -999.25"
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert list(reread.column_order) == ["AZIMUTH_MSL_LONG", "TVD"]
        assert len(reread.columns["TVD"]) == 3

    def test_nan_provenance_warning_when_the_model_is_not_reader_produced(
        self, tmp_path: Path
    ) -> None:
        """C-04: NaN becomes declared missing data — say so, never raise."""
        messages = _write(
            tmp_path / "launder.dev",
            _dev({"MD": [0.0, 10.0], "TVD": np.array([0.0, np.nan])}),
        )
        assert len(messages) == 1
        assert "NaN" in messages[0]
        assert "MISSING" in messages[0]
        assert "DECLARED MISSING DATA" in messages[0]

    def test_non_sentinel_null_reports_the_number_it_reads_back_as(self, tmp_path: Path) -> None:
        """C-5: a non-sentinel null is a fabricated MEASUREMENT, not missing data."""
        target = tmp_path / "numericnull.dev"
        messages = _write(
            target,
            _dev({"MD": [0.0, 10.0], "TVD": np.array([0.0, np.nan])}),
            null_value=-1234.5,
        )
        assert len(messages) == 1
        assert "DECLARED MISSING DATA" not in messages[0]
        assert "-1234.5" in messages[0]
        assert "NUMBER" in messages[0]
        reread, _ = _read(target)
        assert float(reread.columns["TVD"][1]) == -1234.5

    def test_reader_produced_nan_does_not_warn(self, tmp_path: Path) -> None:
        """C-04: `_designed_nan` models describe their own gaps — no warning."""
        source_dev = _dev({"MD": [0.0, 10.0], "TVD": np.array([0.0, np.nan])})
        source_dev._designed_nan = True
        assert _write(tmp_path / "designed.dev", source_dev) == []

    def test_exact_sentinel_collision_warns_per_column(self, tmp_path: Path) -> None:
        """R-01: a real `-999.25` is emitted and reads back as missing."""
        target = tmp_path / "collide.dev"
        messages = _write(target, _dev({"MD": [0.0, 10.0], "X": np.array([-999.25, 5.0])}))
        collisions = [message for message in messages if "NO-DATA" in message]
        assert len(collisions) == 1
        assert "'X'" in collisions[0]
        assert "-999.25" in collisions[0]
        assert target.read_text() == "MD X\n0.0 -999.25\n10.0 5.0\n"
        reread, _ = _read(target)
        assert np.isnan(reread.columns["X"][0])

    def test_precision_rounded_into_a_sentinel_warns(self, tmp_path: Path) -> None:
        """R-01: the gate is the emitted TOKEN, not value == null_value."""
        messages = _write(
            tmp_path / "rounded.dev",
            _dev({"MD": [0.0, 10.0], "TVDSS": np.array([-999.001, 5.0])}),
            precision=".2f",
        )
        collisions = [message for message in messages if "NO-DATA" in message]
        assert len(collisions) == 1
        assert "'TVDSS'" in collisions[0]
        assert "'-999.00'" in collisions[0]

    def test_clean_survey_emits_no_collision_warning(self, tmp_path: Path) -> None:
        messages = _write(
            tmp_path / "clean.dev",
            _dev({"MD": [0.0, 10.0], "INC": [0.0, 1.5], "AZI": [0.0, 45.0]}),
        )
        assert messages == []

    @pytest.mark.parametrize(
        ("null_value", "match"),
        [
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (float("-inf"), "must be finite"),
            (np.float64("nan"), "must be finite"),
            (None, "must be a finite real number"),
            (True, "must be a finite real number"),
            ("-999.25", "must be a finite real number"),
            (b"-999.25", "must be a finite real number"),
            ([], "must be a finite real number"),
            (Decimal("-999.25"), "must be a finite real number"),
            (complex(-999.25, 0), "must be a finite real number"),
        ],
    )
    def test_rejects_non_finite_or_non_numeric_null(
        self, tmp_path: Path, null_value: Any, match: str
    ) -> None:
        """C-06/R-08: a non-finite null emits literal text — refuse it."""
        with pytest.raises(DEVWriteError, match=match):
            write_dev_file(
                tmp_path / "null.dev", _dev({"MD": [0.0], "TVD": [1.0]}), null_value=null_value
            )

    def test_integral_null_value_is_emitted_as_a_decimal_token(self, tmp_path: Path) -> None:
        target = tmp_path / "intnull.dev"
        messages = _write(
            target,
            _dev({"MD": [0.0, 10.0], "TVD": np.array([0.0, np.nan])}),
            null_value=-999,
        )
        assert target.read_text() == "MD TVD\n0.0 0.0\n10.0 -999.0\n"
        assert not [message for message in messages if "NO-DATA" in message]


class TestDEVColumnNameGrammar:
    """AC-8: names the DEV header grammar cannot represent are refused."""

    @pytest.mark.parametrize(
        ("columns", "match"),
        [
            ({"": [1.0]}, "empty or whitespace-only"),
            ({"   ": [1.0]}, "empty or whitespace-only"),
            ({" MD": [1.0]}, "leading or trailing whitespace"),
            ({"MD ": [1.0]}, "leading or trailing whitespace"),
            ({"MD TVD": [1.0]}, "contains whitespace"),
            ({"M\u00a0D": [1.0]}, "contains whitespace"),
            ({"MD,X": [1.0]}, "field separator"),
            ({"MD;X": [1.0]}, "field separator"),
            ({"MD\tX": [1.0]}, "field separator"),
            ({"MD\x00X": [1.0]}, "control character"),
            ({"MD\x07X": [1.0]}, "control character"),
            ({'"MD': [1.0]}, "double quote"),
            ({"#MD": [1.0], "TVD": [2.0]}, "begins with '#'"),
            ({"*COLUMNSX": [1.0], "TVD": [2.0]}, "'*COLUMNS'"),
            ({"sep=|": [1.0]}, "sep=<char>"),
            ({"=====": [1.0]}, "bare separator rule"),
            ({"1": [1.0], "2": [2.0]}, "parses entirely as numbers"),
            ({"1.5": [1.0]}, "parses entirely as numbers"),
        ],
    )
    def test_rejects_unrepresentable_name(
        self, tmp_path: Path, columns: dict[str, list[float]], match: str
    ) -> None:
        with pytest.raises(DEVWriteError, match=match):
            write_dev_file(tmp_path / "bad.dev", _dev(columns))
        assert not (tmp_path / "bad.dev").exists()

    @pytest.mark.parametrize(
        ("columns", "delimiter"),
        [
            ({"A|B": [1.0, 2.0], "X": [3.0, 4.0]}, "|"),
            ({"A.B": [1.0, 2.0], "X": [3.0, 4.0]}, "."),
            ({"A1B": [1.0, 2.0], "X": [3.0, 4.0]}, "1"),
        ],
    )
    def test_rejects_a_name_containing_the_active_delimiter(
        self, tmp_path: Path, columns: dict[str, list[float]], delimiter: str
    ) -> None:
        """C-1: the name grammar is POSITIONAL — it reads the delimiter in effect.

        Pre-fix these names passed the grammar (it consulted a fixed separator
        set) and the file was written with the name split into phantom columns;
        the `.`/digit cases were only caught later by the internal row assert.
        """
        target = tmp_path / "delimited.dev"
        name = next(iter(columns))
        with pytest.raises(DEVWriteError) as excinfo:
            write_dev_file(target, _dev(columns), delimiter=delimiter)
        message = str(excinfo.value)
        assert repr(name) in message
        assert repr(delimiter) in message
        assert "active field delimiter" in message
        assert "Internal emission error" not in message
        assert not target.exists()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp_")] == []

    def test_the_delimiter_test_is_not_applied_under_the_default_space(
        self, tmp_path: Path
    ) -> None:
        """C-1: `A|B` is refused under `delimiter='|'` and accepted under space."""
        target = tmp_path / "pipe-name.dev"
        _write(target, _dev({"A|B": [1.0, 2.0], "X": [3.0, 4.0]}))
        assert target.read_text() == "A|B X\n1.0 3.0\n2.0 4.0\n"
        reread, format_diagnostics = _read(target)
        assert format_diagnostics == []
        assert list(reread.column_order) == ["A|B", "X"]

    @pytest.mark.parametrize("name", ["AZIM", "INCL", "TVDSS", "MDKB", "DEPT", "md", "AZIMUTH_MSL"])
    def test_accepts_legitimate_dev_vocabulary(self, tmp_path: Path, name: str) -> None:
        """AC-8: alias/lowercase spellings are warned about, never refused.

        Each name is emitted as the ONLY column: pairing it with ``MD``
        would be an alias collision (``MDKB``/``DEPT`` normalize to ``MD``)
        and is refused by H-02's separate check.
        """
        target = tmp_path / "vocab.dev"
        _write(target, _dev({name: [1.0, 2.0]}))
        assert target.read_text().splitlines()[0] == name

    def test_hash_name_is_allowed_outside_the_first_position(self, tmp_path: Path) -> None:
        target = tmp_path / "hash.dev"
        _write(target, _dev({"MD": [0.0, 10.0], "#X": [1.0, 2.0]}))
        assert target.read_text().splitlines()[0] == "MD #X"


class TestDEVOrderingAuthority:
    """C-05/I-02/RE-3/H-02: `column_order` is the only emission authority."""

    def test_permuted_column_order_is_emitted_and_reread(self, tmp_path: Path) -> None:
        dev = _dev(
            {"MD": [0.0, 10.0], "TVD": [0.0, -10.0], "INC": [0.0, 1.0]},
            column_order=["INC", "MD", "TVD"],
        )
        target = tmp_path / "perm.dev"
        _write(target, dev)
        assert target.read_text().splitlines()[0] == "INC MD TVD"
        assert target.read_text().splitlines()[1] == "0.0 0.0 0.0"
        reread, _ = _read(target)
        assert list(reread.column_order) == ["INC", "MD", "TVD"]

    def test_depth_column_is_selected_by_name_not_by_position(self, tmp_path: Path) -> None:
        """C-3 (amended AC-3): the index is the first `MD`-family column."""
        dev = _dev(
            {"X": [1.0, 2.0], "MD": [1234.56789, 1250.123456789], "TVD": [5.0, 6.0]},
            column_order=["X", "MD", "TVD"],
        )
        target = tmp_path / "named.dev"
        assert _write(target, dev) == []
        assert target.read_text().splitlines()[1] == "1.0 1234.56789 5.0"
        assert target.read_text().splitlines()[2] == "2.0 1250.123456789 6.0"
        reread, _ = _read(target)
        assert list(reread.column_order) == ["X", "MD", "TVD"]
        np.testing.assert_array_equal(reread.columns["MD"], dev.columns["MD"])

    def test_depth_alias_is_recognised_as_the_index_column(self, tmp_path: Path) -> None:
        """C-3: `DEPT` re-reads as `MD`, so it is the depth the reader uses."""
        dev = _dev(
            {"X": [1.0, 2.0], "DEPT": [1234.56789, 1250.123456789]},
            column_order=["X", "DEPT"],
        )
        target = tmp_path / "dept.dev"
        messages = _write(target, dev)
        assert not [message for message in messages if "Index curve" in message]
        assert target.read_text().splitlines()[1] == "1.0 1234.56789"
        assert any("'DEPT' will be renamed to 'MD'" in message for message in messages)

    def test_md_first_emission_is_byte_identical(self, tmp_path: Path) -> None:
        """C-3 control: an `MD`-first model keeps its historical bytes."""
        target = tmp_path / "mdfirst.dev"
        assert _write(target, _dev({"MD": [1234.56789, 1250.123456789], "X": [1.0, 2.0]})) == []
        assert target.read_text() == "MD X\n1234.56789 1.0\n1250.123456789 2.0\n"

    def test_md_less_model_keeps_the_position_zero_fallback(self, tmp_path: Path) -> None:
        """C-3 fallback: with no `MD`-family column, position 0 stays the index."""
        target = tmp_path / "nomd.dev"
        _write(target, _dev({"X": [1234.56789, 2.0], "TVD": [5.0, 6.0]}))
        assert target.read_text().splitlines()[1] == "1234.56789 5.0"

    def test_column_order_is_never_mutated(self, tmp_path: Path) -> None:
        dev = _dev(
            {"MD": [0.0, 10.0], "TVD": [0.0, -10.0], "INC": [0.0, 1.0]},
            column_order=["INC", "MD", "TVD"],
        )
        before = list(dev.column_order)
        _write(tmp_path / "nomutate.dev", dev)
        assert list(dev.column_order) == before

    def test_duplicate_order_entries_raise(self, tmp_path: Path) -> None:
        """RE-3: the state `column_order *= 2` leaves after its re-wrap raise."""
        dev = _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]})
        with pytest.raises(LASDataError):
            dev.column_order *= 2
        assert list(dev.column_order) == ["MD", "TVD", "MD", "TVD"]
        with pytest.raises(DEVWriteError, match="duplicate entries"):
            write_dev_file(tmp_path / "dup.dev", dev)

    def test_orphan_order_entry_raises(self, tmp_path: Path) -> None:
        """I-02: an order entry with no column (validate() reports, write refuses)."""
        dev = _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]})
        object.__setattr__(dev, "column_order", ["MD", "GHOST"])
        with pytest.raises(DEVWriteError, match="GHOST"):
            write_dev_file(tmp_path / "orphan.dev", dev)

    def test_alias_pair_raises_naming_both_columns(self, tmp_path: Path) -> None:
        """H-02: the reader would rename the second column to `AZI_2`."""
        with pytest.raises(DEVWriteError, match="'AZI' and 'AZIM'"):
            write_dev_file(tmp_path / "alias.dev", _dev({"AZI": [1.0], "AZIM": [2.0]}))

    def test_alias_name_warns_and_spelling_can_be_preserved(self, tmp_path: Path) -> None:
        """H-02: names are emitted verbatim; the rename is reported."""
        target = tmp_path / "verbatim.dev"
        messages = _write(target, _dev({"MD": [0.0, 10.0], "AZIM": [1.0, 2.0]}))
        assert any("'AZIM' will be renamed to 'AZI'" in message for message in messages)
        assert target.read_text().splitlines()[0] == "MD AZIM"
        preserved, _ = _read_with_aliases(target)
        assert list(preserved.column_order) == ["MD", "AZIM"]


class TestDEVDegenerateInput:
    """AC-6: warn, then refuse — never a silently unreadable file."""

    def test_empty_model_warns_then_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.dev"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(DEVWriteError, match="no columns"):
                write_dev_file(target, _dev({}))
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1
        assert "not written" in messages[0]
        assert not target.exists()

    def test_zero_row_columns_warn_then_raise(self, tmp_path: Path) -> None:
        target = tmp_path / "zerorow.dev"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(DEVWriteError, match="all empty"):
                write_dev_file(target, _dev({"MD": []}))
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1
        assert "0 rows" in messages[0]
        assert not target.exists()

    def test_rejected_name_raises_without_the_degenerate_warning(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(DEVWriteError, match="empty or whitespace-only"):
                write_dev_file(tmp_path / "x.dev", _dev({"": [1.0]}))
        assert [str(w.message) for w in caught] == []


class TestDEVAtomicWrite:
    """AC-7: temp file + os.replace, mode preserved, target never opened."""

    def test_overwrite_preserves_existing_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "mode.dev"
        dev = _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]})
        write_dev_file(target, dev)
        os.chmod(target, 0o640)
        write_dev_file(target, _dev({"MD": [0.0], "TVD": [5.0]}))
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o640
        assert target.read_text() == "MD TVD\n0.0 5.0\n"

    def test_write_protected_target_is_replaced_and_keeps_its_mode(self, tmp_path: Path) -> None:
        """I-01 (documented): replace, not open('w') — write-protect the dir."""
        target = tmp_path / "readonly.dev"
        write_dev_file(target, _dev({"MD": [0.0], "TVD": [1.0]}))
        os.chmod(target, 0o444)
        write_dev_file(target, _dev({"MD": [0.0, 10.0], "TVD": [3.0, 4.0]}))
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o444
        assert target.read_text() == "MD TVD\n0.0 3.0\n10.0 4.0\n"

    def test_new_file_mode_follows_the_umask(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh.dev"
        write_dev_file(target, _dev({"MD": [0.0], "TVD": [1.0]}))
        current_umask = os.umask(0o022)
        os.umask(current_umask)
        assert stat.S_IMODE(os.stat(target).st_mode) == (0o666 & ~current_umask)

    def test_late_failure_leaves_the_target_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "atomic.dev"
        write_dev_file(target, _dev({"MD": [0.0], "TVD": [1.0]}))
        before = target.read_bytes()

        def fail_replace(src: str, dst: str) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("pylasdev.dev_writer.os.replace", fail_replace)
        with pytest.raises(DEVWriteError, match="simulated replace failure"):
            write_dev_file(target, _dev({"MD": [0.0, 50.0], "TVD": [9.0, 9.0]}))
        assert target.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp_")] == []

    def test_written_bytes_are_lf_terminated(self, tmp_path: Path) -> None:
        target = tmp_path / "eol.dev"
        write_dev_file(target, _dev({"MD": [0.0, 10.0], "TVD": [0.0, 1.0]}))
        data = target.read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"\n")


class TestDEVSentinelSetContract:
    """The writer and the reader must agree on the numeric sentinel set."""

    def test_writer_imports_the_reader_sentinel_set(self) -> None:
        from pylasdev import dev_writer

        assert dev_writer._DEV_NUMERIC_SENTINELS is _DEV_NUMERIC_SENTINELS
        assert dev_writer._DEV_NUMERIC_SENTINELS == frozenset({-999.0, -999.25, -9999.0, -9999.25})
