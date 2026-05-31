"""Tests for encoding detection and file reading utilities."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from pylasdev.encoding import FALLBACK_ENCODINGS, detect_encoding, read_with_encoding


class TestDetectEncoding:
    """Tests for encoding detection."""

    def test_detect_utf8_file(self, tmp_path: Path) -> None:
        """Test detecting UTF-8 encoded file."""
        test_file = tmp_path / "test.las"
        test_file.write_text("Hello UTF-8", encoding="utf-8")
        enc = detect_encoding(test_file)
        assert enc is not None

    def test_detect_returns_string(self, tmp_path: Path) -> None:
        """Test that detect_encoding returns a string."""
        test_file = tmp_path / "test.las"
        test_file.write_text("Simple ASCII text\n", encoding="utf-8")
        result = detect_encoding(test_file)
        assert isinstance(result, str)

    # --- TEST-14: HAS_CHARDET=False path ---
    def test_detect_without_chardet(self, tmp_path: Path) -> None:
        """Test detect_encoding when chardet is not available."""
        test_file = tmp_path / "test.las"
        test_file.write_text("Simple text", encoding="utf-8")

        # Mock HAS_CHARDET to False to exercise the fallback path
        with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
            enc = detect_encoding(test_file)
            # Without chardet, should return utf-8 (fallback)
            assert enc == "utf-8"


class TestReadWithEncoding:
    """Tests for read_with_encoding."""

    def test_read_utf8(self, tmp_path: Path) -> None:
        """Test reading UTF-8 file."""
        test_file = tmp_path / "test.las"
        test_file.write_text("Hello UTF-8", encoding="utf-8")
        _enc, content = read_with_encoding(test_file)
        assert content == "Hello UTF-8"

    def test_read_with_explicit_encoding(self, tmp_path: Path) -> None:
        """Test reading with explicit encoding override."""
        test_file = tmp_path / "test.las"
        test_file.write_bytes(b"Hello")
        enc, content = read_with_encoding(test_file, encoding="utf-8")
        assert enc == "utf-8"
        assert content == "Hello"

    def test_read_cp1251(self, tmp_path: Path) -> None:
        """Test reading CP1251 encoded file (Russian Windows)."""
        test_file = tmp_path / "test.las"
        russian_text = "\u041f\u0440\u0438\u0432\u0435\u0442"  # "Привет"
        test_file.write_bytes(russian_text.encode("cp1251"))
        _enc, content = read_with_encoding(test_file)
        assert russian_text in content

    def test_read_cp866(self, tmp_path: Path) -> None:
        """Test reading CP866 encoded file (Russian DOS)."""
        test_file = tmp_path / "test.las"
        russian_text = "\u041f\u0420\u0418\u0412\u0415\u0422"  # "ПРИВЕТ"
        test_file.write_bytes(russian_text.encode("cp866"))
        _enc, content = read_with_encoding(test_file)
        # Should be readable (either via chardet or fallback chain)
        assert len(content) > 0

    def test_read_latin1(self, tmp_path: Path) -> None:
        """Test reading Latin-1 encoded file."""
        test_file = tmp_path / "test.las"
        text = "Caf\u00e9 r\u00e9sum\u00e9"
        test_file.write_bytes(text.encode("latin-1"))
        _enc, content = read_with_encoding(test_file)
        assert len(content) > 0

    def test_fallback_chain_exists(self) -> None:
        """Test that fallback encodings are defined."""
        assert len(FALLBACK_ENCODINGS) >= 4
        assert "utf-8" in FALLBACK_ENCODINGS
        assert "cp1251" in FALLBACK_ENCODINGS
        assert "cp866" in FALLBACK_ENCODINGS

    def test_read_real_las_file(self, test_data_dir: Path) -> None:
        """Test reading real LAS files from test_data."""
        las_files = list(test_data_dir.glob("*.las"))
        assert len(las_files) > 0
        for las_file in las_files:
            enc, content = read_with_encoding(las_file)
            assert len(content) > 0
            assert isinstance(enc, str)

    # --- F-40: Explicit encoding failure ---
    def test_explicit_encoding_fails_with_unicode_decode_error(self, tmp_path: Path) -> None:
        """Test that explicit encoding parameter raises UnicodeDecodeError on mismatch.

        Exercises encoding.py:83-85 — the direct read_text(encoding=encoding) path
        that occurs when an explicit encoding is provided.
        When the file content cannot be decoded with the given encoding,
        UnicodeDecodeError should propagate.
        """
        test_file = tmp_path / "cp1251.las"
        # Write Russian text in CP1251 (Windows Cyrillic)
        russian_text = "\u041f\u0440\u0438\u0432\u0435\u0442"  # "Привет"
        test_file.write_bytes(russian_text.encode("cp1251"))

        # Trying to decode CP1251 as ASCII should fail
        with pytest.raises(UnicodeDecodeError):
            read_with_encoding(test_file, encoding="ascii")
