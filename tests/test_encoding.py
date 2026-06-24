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
        """Test that explicit encoding parameter raises LASEncodingError on mismatch.

        Exercises the explicit-encoding decode path in read_with_encoding.
        When the file content cannot be decoded with the given encoding,
        LASEncodingError should be raised wrapping the UnicodeDecodeError.
        """
        test_file = tmp_path / "cp1251.las"
        # Write Russian text in CP1251 (Windows Cyrillic)
        russian_text = "\u041f\u0440\u0438\u0432\u0435\u0442"  # "Привет"
        test_file.write_bytes(russian_text.encode("cp1251"))

        # Trying to decode CP1251 as ASCII should fail
        from pylasdev.exceptions import LASEncodingError

        with pytest.raises(LASEncodingError, match="Failed to decode"):
            read_with_encoding(test_file, encoding="ascii")

    # --- T5: LASEncodingError unreachable path (encoding.py:110) ---
    def test_fallback_chain_empty_raises_encoding_error(self, tmp_path: Path) -> None:
        """Test LASEncodingError when fallback chain is exhausted.

        Exercises encoding.py:110-112 — the LASEncodingError raise at the end
        of read_with_encoding when all fallback encodings fail.
        This path is normally unreachable because latin-1 can decode any byte,
        but it guards against the case where FALLBACK_ENCODINGS is modified.
        """
        test_file = tmp_path / "bad.las"
        # Write bytes that are invalid UTF-8 and will fail with standard encodings
        test_file.write_bytes(b"\xff\xfe\x00\x01")

        # Mock fallback chain to be empty so we hit the error path
        with mock.patch("pylasdev.encoding.FALLBACK_ENCODINGS", []):
            with mock.patch("pylasdev.encoding.HAS_CHARDET", False):
                from pylasdev.exceptions import LASEncodingError

                with pytest.raises(LASEncodingError, match="Failed to decode"):
                    read_with_encoding(test_file)


class TestLowConfidenceChardetFallback:
    """F-027: Low-confidence chardet fallback tests."""

    def test_confidence_none_falls_back_to_utf8(self) -> None:
        """Test that confidence=None falls back to utf-8."""
        from pylasdev.encoding import _detect_encoding_from_bytes

        with mock.patch("pylasdev.encoding.HAS_CHARDET", True):
            with mock.patch("pylasdev.encoding.chardet", create=True) as mock_chardet:
                mock_chardet.detect.return_value = {
                    "encoding": "cp1251",
                    "confidence": None,
                }
                result = _detect_encoding_from_bytes(b"hello")
                assert result == "utf-8"

    def test_confidence_falsy_falls_back_to_utf8(self) -> None:
        """Test that confidence=0.0 (falsy) falls back to utf-8."""
        from pylasdev.encoding import _detect_encoding_from_bytes

        with mock.patch("pylasdev.encoding.HAS_CHARDET", True):
            with mock.patch("pylasdev.encoding.chardet", create=True) as mock_chardet:
                mock_chardet.detect.return_value = {
                    "encoding": "cp1251",
                    "confidence": 0.0,
                }
                result = _detect_encoding_from_bytes(b"hello")
                assert result == "utf-8"

    def test_confidence_below_threshold_falls_back_to_utf8(self) -> None:
        """Test that confidence=0.7 falls back to utf-8 (not > 0.7)."""
        from pylasdev.encoding import _detect_encoding_from_bytes

        with mock.patch("pylasdev.encoding.HAS_CHARDET", True):
            with mock.patch("pylasdev.encoding.chardet", create=True) as mock_chardet:
                mock_chardet.detect.return_value = {
                    "encoding": "cp1251",
                    "confidence": 0.7,
                }
                result = _detect_encoding_from_bytes(b"hello")
                assert result == "utf-8"

    def test_confidence_above_threshold_uses_detected_encoding(self) -> None:
        """Test that confidence=0.8 uses the detected encoding."""
        from pylasdev.encoding import _detect_encoding_from_bytes

        with mock.patch("pylasdev.encoding.HAS_CHARDET", True):
            with mock.patch("pylasdev.encoding.chardet", create=True) as mock_chardet:
                mock_chardet.detect.return_value = {
                    "encoding": "cp1251",
                    "confidence": 0.9,
                }
                result = _detect_encoding_from_bytes(b"hello")
                assert result == "cp1251"

    def test_confidence_above_threshold_encoding_none_falls_back(self) -> None:
        """Test that high confidence but encoding=None falls back to utf-8."""
        from pylasdev.encoding import _detect_encoding_from_bytes

        with mock.patch("pylasdev.encoding.HAS_CHARDET", True):
            with mock.patch("pylasdev.encoding.chardet", create=True) as mock_chardet:
                mock_chardet.detect.return_value = {
                    "encoding": None,
                    "confidence": 0.9,
                }
                result = _detect_encoding_from_bytes(b"hello")
                assert result == "utf-8"


class TestBOMHandling:
    """Tests for UTF-8 BOM (Byte Order Mark) handling.

    LAS files may inadvertently contain a UTF-8 BOM at the start of
    the file. The encoding module strips it via content.lstrip("\\ufeff")
    on all decode paths so that the BOM does not interfere with section
    header detection.
    """

    def test_bom_stripped_from_las_file_read(self, tmp_path: Path) -> None:
        """Test that a LAS file with UTF-8 BOM is parsed correctly.

        The BOM should be stripped during decoding so that the leading
        ``~VERSION`` section header is recognised and the file content
        is parsed as valid LAS.
        """
        from pylasdev import read_las_file_as_object

        # Build valid LAS content (no BOM in the string)
        las_body = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            " STRT.M   100.0   : START DEPTH\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            " DT.US/M  :  Sonic\n"
            "~A  DEPT  DT\n"
            "100.0  50.0\n"
            "101.0  51.0\n"
        )

        # Prepend UTF-8 BOM and write as raw bytes
        bom = b"\xef\xbb\xbf"
        raw = bom + las_body.encode("utf-8")

        test_file = tmp_path / "bom_test.las"
        test_file.write_bytes(raw)

        # Read via read_las_file_as_object (exercises all decode paths)
        las = read_las_file_as_object(test_file)

        # Verify the BOM was stripped — version info must be correct
        assert las.version.vers == "2.0"

        # Verify curves were parsed
        assert len(las.curves) == 2
        assert las.curves_order == ["DEPT", "DT"]

        # Verify logs were parsed with correct data
        assert len(las.logs["DEPT"]) == 2
        assert las.logs["DEPT"][0] == 100.0
        assert las.logs["DEPT"][1] == 101.0
        assert las.logs["DT"][0] == 50.0
        assert las.logs["DT"][1] == 51.0

        # Verify well section entries present
        assert "STRT" in las.well

    def test_bom_with_explicit_encoding(self, tmp_path: Path) -> None:
        """Test BOM stripping works when an explicit encoding is passed.

        Exercises the explicit-encoding decode path at encoding.py:117-124
        where ``content.lstrip("\\ufeff")`` is called after decoding.
        """
        from pylasdev import read_las_file

        las_body = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M   :  Depth\n"
            "~A  DEPT\n"
            "100.0\n"
        )

        bom = b"\xef\xbb\xbf"
        raw = bom + las_body.encode("utf-8")

        test_file = tmp_path / "bom_explicit.las"
        test_file.write_bytes(raw)

        # Read with explicit encoding — exercises the encoding != None path
        data = read_las_file(test_file, encoding="utf-8")

        assert data["version"]["VERS"] == "2.0"
        assert "DEPT" in data["logs"]
        assert data["logs"]["DEPT"][0] == 100.0
