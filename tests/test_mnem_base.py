"""Tests for mnemonic base database."""

from __future__ import annotations

from pylasdev.mnem_base import MNEM_BASE


class TestMnemBase:
    """Tests for the mnemonic alias database."""

    def test_mnem_base_is_dict(self) -> None:
        """Test that MNEM_BASE is a dict."""
        assert isinstance(MNEM_BASE, dict)

    def test_mnem_base_not_empty(self) -> None:
        """Test that MNEM_BASE has entries."""
        assert len(MNEM_BASE) > 100

    def test_mnem_base_values_are_strings(self) -> None:
        """Test that all values are strings."""
        for key, value in MNEM_BASE.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value for {key!r} is not a string"

    def test_known_mappings(self) -> None:
        """Test some known mnemonic mappings."""
        assert MNEM_BASE.get("AK") == "DT"
        assert MNEM_BASE.get("AKDT") == "DT"

    def test_mnem_base_used_in_reader(self) -> None:
        """Test that mnem_base can be used with read_las_file."""
        from pathlib import Path

        from pylasdev import read_las_file

        test_data = Path(__file__).parent.parent / "test_data"
        las_files = list(test_data.glob("*.las"))
        if las_files:
            # Just verify it doesn't crash with mnem_base
            data = read_las_file(las_files[0], mnem_base=MNEM_BASE)
            assert "logs" in data
