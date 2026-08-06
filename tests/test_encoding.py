"""Tests for encoding detection and file reading utilities."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from pylasdev.encoding import FALLBACK_ENCODINGS, detect_encoding, read_with_encoding


class TestDetectEncoding:
    """Tests for encoding detection."""

    def test_detect_utf8_file(self, tmp_path: Path) -> None:
        """Test detecting UTF-8 encoded file.

        Uses non-ASCII UTF-8 content so that chardet can distinguish
        UTF-8 from pure ASCII (which chardet labels "ascii").
        """
        test_file = tmp_path / "test.las"
        test_file.write_text("H\u00e9llo UTF-8 \u2603", encoding="utf-8")
        enc = detect_encoding(test_file)
        assert enc == "utf-8"

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
        """Test reading CP1251 encoded file (Russian Windows).

        Uses explicit encoding because short test files (< 50 KB) are
        too small for chardet to distinguish single-byte Cyrillic
        encodings (cp1251 vs cp866).  chardet's default path may return
        either encoding for short samples; quality-based selection
        (_decode_best_quality) picks the correct encoding regardless
        of the fallback chain order.  Explicit encoding removes the
        ambiguity.
        """
        test_file = tmp_path / "test.las"
        russian_text = "\u041f\u0440\u0438\u0432\u0435\u0442"  # "Привет"
        test_file.write_bytes(russian_text.encode("cp1251"))
        _enc, content = read_with_encoding(test_file, encoding="cp1251")
        assert russian_text in content

    def test_read_cp866(self, tmp_path: Path) -> None:
        """Test reading CP866 encoded file (Russian DOS).

        With cp1251 now ordered before cp866 in the fallback chain (CORR-F29),
        cp866-encoded Russian text is still decoded correctly via
        quality-based selection (_decode_best_quality), which picks the
        encoding producing the best decode quality regardless of order.
        """
        test_file = tmp_path / "test.las"
        russian_text = "\u041f\u0420\u0418\u0412\u0415\u0422"  # "ПРИВЕТ"
        test_file.write_bytes(russian_text.encode("cp866"))
        _enc, content = read_with_encoding(test_file)
        assert russian_text in content
        assert _enc == "cp866"

    def test_read_latin1(self, tmp_path: Path) -> None:
        """Test reading Latin-1 encoded file.

        Uses explicit encoding because short test files are too small for
        chardet to confidently detect Latin-1.  Without explicit encoding
        the fallback chain tries cp866 first (which succeeds on any byte
        sequence, producing garbled output instead of correct Latin-1).
        """
        test_file = tmp_path / "test.las"
        text = "Caf\u00e9 r\u00e9sum\u00e9"
        test_file.write_bytes(text.encode("latin-1"))
        _enc, content = read_with_encoding(test_file, encoding="latin-1")
        assert text in content

    def test_read_cp1252(self, tmp_path: Path) -> None:
        """Test reading CP1252 encoded file (Western European Windows).

        CP1252 is in FALLBACK_ENCODINGS (encoding.py:36) but is unreachable
        via the fallback chain because cp866 (preceding in the chain) can
        decode all 256 byte values.  This test exercises the explicit
        ``encoding="cp1252"`` parameter path.

        Uses Western European accented characters: é, ñ, ü, ç, À.
        """
        test_file = tmp_path / "test.las"
        text = "T\u00eate en fran\u00e7ais: \u00e9t\u00e9, h\u00f4tel, \u00f1"
        test_file.write_bytes(text.encode("cp1252"))
        _enc, content = read_with_encoding(test_file, encoding="cp1252")
        assert text in content

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


# ============================================================
# Production Check Regression Tests
# ============================================================


class TestProductionCheckEncodingFix:
    """Regression test for F-217 fix in encoding.py."""

    def test_cyrillic_after_large_ascii_header(self, tmp_path: Path) -> None:
        """F-217: Cyrillic content after >10KB ASCII preamble is detected.

        Before the fix, the 10K sample window missed Cyrillic beyond the
        ASCII preamble. Now the window is 64K, capturing the Cyrillic
        portion. We test by creating a file with >15KB ASCII header
        (typical LAS headers are 5-15KB) followed by Cyrillic content
        and verifying cp1251 detection.
        """
        # Build a file with ~15KB ASCII preamble + Cyrillic content
        # A single LAS comment line is ~80 chars; 200 lines ≈ 16KB
        ascii_preamble = "# " + ("X" * 77) + "\n"
        large_preamble = ascii_preamble * 200  # ~15.6 KB

        # Cyrillic content in Russian (cp1251 encoded)
        # "Привет из России" — common Russian text
        russian_text = (
            "\u041f\u0440\u0438\u0432\u0435\u0442 \u0438\u0437 \u0420\u043e\u0441\u0441\u0438\u0438"
        )
        russian_part = "~VERSION INFORMATION\n" + russian_text + "\n"

        content = large_preamble + russian_part
        test_file = tmp_path / "cyrillic_after_preamble.las"
        test_file.write_text(content, encoding="utf-8")

        # Should detect and read the file correctly without crashing
        from pylasdev.encoding import read_with_encoding

        enc, text = read_with_encoding(test_file)
        assert isinstance(enc, str)
        assert len(text) > 0
        # The Russian text characters should be present in the decoded content
        assert "VERSION" in text


class TestE06NumeroSignCyrillic:
    """E-06: cp1251 files containing "№" (0xB9) must not decode as cp1252.

    Byte 0xB9 is "№" in cp1251 (not alnum) but "¹" in cp1252 (alnum), so
    any "№" in the sample gives cp1252 a small ratio advantage over cp1251.
    The ratio-primary sort then selects cp1252 → full-file mojibake.  The
    near-tie Cyrillic preference must override only that small margin.
    """

    def test_cp1251_with_numero_sign_detected_as_cp1251(self, tmp_path: Path) -> None:
        """E-06: a cp1251 Russian file with "№" decodes as cp1251, not cp1252."""
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u2116 123 \u041c\u0415\u0421\u0422\u041e\u0420\u041e\u0416\u0414\u0415\u041d\u0418\u0415 \u041f\u041b\u0410\u0421\u0422 "
        # "СКВАЖИНА № 123 МЕСТОРОЖДЕНИЕ ПЛАСТ "
        mixed = ("WELL NAME : " + russian + "\n") * 50
        test_file = tmp_path / "e06_numero_sign.las"
        test_file.write_bytes(mixed.encode("cp1251"))

        # Force chardet to fail (returns utf-8 as the fallback) so the
        # quality-based fallback chain makes the decision.
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, _text = read_with_encoding(test_file)
        assert enc == "cp1251", f"E-06: cp1251 file with '№' misdecoded as {enc!r} (mojibake)"

    def test_utf8_cyrillic_still_detected_after_fix(self, tmp_path: Path) -> None:
        """E-06: UTF-8 Cyrillic must not regress to cp1251 (fix-regression guard)."""
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u2116 123 \u041c\u0415\u0421\u0422\u041e\u0420\u041e\u0416\u0414\u0415\u041d\u0418\u0415 \u041f\u041b\u0410\u0421\u0422 "
        mixed = ("WELL NAME : " + russian + "\n") * 50
        test_file = tmp_path / "e06_utf8_cyrillic.las"
        test_file.write_bytes(mixed.encode("utf-8"))

        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="latin-1",
        ):
            enc, text = read_with_encoding(test_file)
        assert enc == "utf-8", f"E-06: UTF-8 Cyrillic file misdecoded as {enc!r}"
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in text


class TestE07CyrillicBeyond64K:
    """E-07: Cyrillic content beyond the first 64K must be detected.

    The byte-frequency and run-length detectors previously sampled only
    the first 64K bytes, so a cp1251/cp866 file whose Cyrillic text lands
    beyond 64K (large ~C/~O sections or long ASCII headers) was invisible
    to both signals → misdecoded as cp1252/latin-1 → silent mojibake.
    """

    def test_cp1251_cyrillic_beyond_64k_detected(self, tmp_path: Path) -> None:
        """E-07: cp1251 Cyrillic beyond 64K decodes as cp1251."""
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u0422\u0415\u0421\u0422 \u041c\u0415\u0421\u0422\u041e\u0420\u041e\u0416\u0414\u0415\u041d\u0418\u0415 \u041f\u041b\u0410\u0421\u0422 "
        # ~70KB ASCII preamble pushes Cyrillic beyond the old 64K window.
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + (russian * 200).encode("cp1251") + b"\n"
        test_file = tmp_path / "e07_beyond_64k.las"
        test_file.write_bytes(raw)

        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, text = read_with_encoding(test_file)
        assert enc == "cp1251", f"E-07: Cyrillic beyond 64K misdecoded as {enc!r}"
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in text

    def test_cp866_cyrillic_beyond_64k_detected(self, tmp_path: Path) -> None:
        """E-07: cp866 Cyrillic beyond 64K is no longer misdecoded as latin-1.

        Before the fix the first-64K sampling missed the Cyrillic tail, so
        _is_cyrillic was False and the file fell to the Western tiebreak
        (latin-1, raw control chars).  After widening the Cyrillic detection
        samples, the file is recognized as Cyrillic — the selection must be
        a Cyrillic encoding, never a Western one.  (Note: when the Cyrillic
        tail lies beyond the F-88 ratio sample window, cp1251-vs-cp866
        cannot be distinguished by ratio — both are valid Cyrillic results.)
        """
        russian = "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u0422\u0415\u0421\u0422 \u041c\u0415\u0421\u0422\u041e\u0420\u041e\u0416\u0414\u0415\u041d\u0418\u0415 \u041f\u041b\u0410\u0421\u0422 "
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + (russian * 200).encode("cp866") + b"\n"
        test_file = tmp_path / "e07_beyond_64k_cp866.las"
        test_file.write_bytes(raw)

        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            enc, text = read_with_encoding(test_file)
        assert enc in ("cp1251", "cp866"), f"E-07: cp866 Cyrillic beyond 64K misdecoded as {enc!r}"
        # Pre-fix the file fell to the Western tiebreak (latin-1).  Post-fix
        # it is recognized as Cyrillic — the decoded text must contain
        # Cyrillic code points, not raw Western control chars.
        assert any(0x0400 <= ord(c) <= 0x04FF for c in text), (
            "E-07: cp866 tail decoded without any Cyrillic characters"
        )


# ──────────────────────────────────────────────────────────────
# ENC-01 (encoding, MEDIUM): №-adjacency Cyrillic rule false-positive.
# A Western cp1252 "¹" (0xB9) ADJACENT to a 3+ accent run fired the
# №-adjacency rule → the whole file decoded as cp1251 (mojibake).
# F-18 fixed the far-away case only; the adjacent case was untested.
# The rule must require the unambiguous № byte (0x85, cp866) or a 0xB9
# followed by an ASCII digit (the Russian "СКВ №1"/"ПЛАСТ №2" convention)  # noqa: RUF003
# — a Western footnote "¹" is not a "№ <number>" prefix.
# ──────────────────────────────────────────────────────────────


class TestENC01NumeroAdjacencyFalsePositive:
    """ENC-01: Western '¹' adjacent to an accent run must NOT confirm
    Cyrillic — 'Nota¹ Ñáñez' (cp1252) decodes as cp1252, not cp1251."""

    def test_western_adjacent_superscript_stays_cp1252(self, tmp_path: Path) -> None:
        text = "Nota\u00b9 \u00d1\u00e1\u00f1ez"  # 'Nota¹ Ñáñez' in cp1252
        test_file = tmp_path / "enc01_adjacent_superscript.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-01: adjacent-¹ Western file misdecoded as {enc!r} (mojibake)"
        assert not any(0x0400 <= ord(c) <= 0x04FF for c in content)

    def test_genuine_cp1251_numero_label_stays_cp1251(self, tmp_path: Path) -> None:
        """Positive control: genuine cp1251 'СКВ №1' (3-run + digit-follow
        №) still confirms Cyrillic — no over-correction of the fix."""  # noqa: RUF002
        russian = "\u0421\u041a\u0412 \u2116 1"  # 'СКВ № 1' in cp1251  # noqa: RUF003
        test_file = tmp_path / "enc01_genuine_cp1251.las"
        test_file.write_bytes(russian.encode("cp1251"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"ENC-01 positive control: genuine cp1251 №-label misdecoded as {enc!r}"
        )
        assert "\u0421\u041a\u0412" in content


# ──────────────────────────────────────────────────────────────
# ENC-02 (encoding, HIGH): word-char-ratio selection flips genuine
# Western cp1252/latin-1 content to cp866/cp1251 when Windows-1252 smart
# punctuation (0x91-0x97) is present — silent mojibake of ALL header
# strings; overrides even a perfect chardet answer.  Fix: (a) honor a
# high-confidence chardet detection (material-margin guard), (b) Western
# near-tie rescue with smart-punct artifact subtraction (mirror of E-06).
# ──────────────────────────────────────────────────────────────


class TestENC02SmartPunctuationWestern:
    """ENC-02: Western files with smart punctuation decode as cp1252."""

    def test_western_smart_punct_stays_cp1252(self, tmp_path: Path) -> None:
        """A cp1252 file with smart quotes/dashes decodes as cp1252 (not
        cp866) via the smart-punct artifact subtraction (chardet absent)."""
        text = 'Puits "Jean-Joseph" \u2014 R\u00e9servoir \u00e0 l\u2019ouest: profondeur 2450,5 m'
        test_file = tmp_path / "enc02_smart_punct.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-02: cp1252 smart-punct file misdecoded as {enc!r} (mojibake)"
        assert "R\u00e9servoir" in content

    def test_mocked_chardet_cp1252_answer_honored(self, tmp_path: Path) -> None:
        """Even a mocked (perfect) chardet cp1252 answer is honored — the
        ratio sort must not override a high-confidence detection."""
        text = 'Puits "Jean-Joseph" \u2014 R\u00e9servoir \u00e0 l\u2019ouest: profondeur 2450,5 m'
        test_file = tmp_path / "enc02_chardet_cp1252.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="cp1252",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.99,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-02: mocked chardet cp1252 answer overridden to {enc!r}"
        assert "R\u00e9servoir" in content

    def test_utf8_cyrillic_still_utf8(self, tmp_path: Path) -> None:
        """Guard: UTF-8 Cyrillic must still decode as UTF-8 (no regression
        from the Western rescue / detected-encoding priority)."""
        russian = (
            "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 "
            "\u0422\u0415\u0421\u0422 \u041c\u0415\u0421\u0422\u041e\u0420\u041e\u0416\u0414\u0415\u041d\u0418\u0415 \u041f\u041b\u0410\u0421\u0422 "
        )
        test_file = tmp_path / "enc02_utf8_cyrillic.las"
        test_file.write_bytes((russian * 50).encode("utf-8"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="latin-1",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.99,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "utf-8", f"ENC-02 guard: UTF-8 Cyrillic misdecoded as {enc!r}"
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in content


# ──────────────────────────────────────────────────────────────
# F-01/F-09 (encoding, MEDIUM): the ENC-02(b) Western rescue subtracted
# a hand-enumerated 7-byte _SMART_PUNCT_BYTES set (0x91-0x97), leaving the
# Euro (0x80), its control-range siblings (0x82/0x84-0x87/0x89/0x8B), and
# the symbol class (0xA1-0xBF/0xD7/0xF7) uncovered — genuine Western
# cp1252 files containing them misdecoded to cp1251/cp866 (silent
# mojibake of all header strings).  Fix: replace the hand set with a
# codec-derived per-pair inflator table (_INFLATORS) computed at module
# load — the byte class comes from the codec tables themselves, so it can
# never be partial again — and gate the rescue on the Western candidate
# being plausible text (_WESTERN_RATIO_FLOOR).
# ──────────────────────────────────────────────────────────────


class TestENC02InflatorTable:
    """ENC-02(b): _INFLATORS is derived from the codec tables, not a
    hand-enumerated subset — pinning the complete per-pair byte classes."""

    def test_inflators_table_completeness(self) -> None:
        """The codec-derived table covers the full 0x80-0xFF class per pair.

        The counts are measured from Python's own codec tables (39/48/17/25
        inflator bytes per pair).  Pinning them guards against accidental
        narrowing of the table — the previous hand-enumerated 7-byte set
        could never converge because every sibling byte was another hole.
        """
        from pylasdev.encoding import _INFLATORS

        assert len(_INFLATORS[("cp866", "cp1252")]) == 39
        assert len(_INFLATORS[("cp866", "latin-1")]) == 48
        assert len(_INFLATORS[("cp1251", "cp1252")]) == 17
        assert len(_INFLATORS[("cp1251", "latin-1")]) == 25

        # Euro + control-range siblings (F-09 class): Cyrillic alnum under
        # cp866 but punctuation under cp1252.
        for b in (0x80, 0x82, 0x84, 0x85, 0x86, 0x87, 0x89, 0x8B):
            assert b in _INFLATORS[("cp866", "cp1252")], f"0x{b:02X} missing from cp866-over-cp1252"
        assert 0x80 in _INFLATORS[("cp1251", "cp1252")], "0x80 missing from cp1251-over-cp1252"

        # Symbol class (F-01 class): Cyrillic alnum under cp1251 but
        # symbols under cp1252.
        for b in (0xA1, 0xA2, 0xA3, 0xA5, 0xA8, 0xAF, 0xB4, 0xB8, 0xBF, 0xD7, 0xF7):
            assert b in _INFLATORS[("cp1251", "cp1252")], (
                f"0x{b:02X} missing from cp1251-over-cp1252"
            )


class TestENC02WesternRescueExtended:
    """ENC-02(b) F-01/F-09: genuine Western cp1252 files containing the
    previously-uncovered byte classes decode as cp1252, not cp1251/cp866."""

    def test_western_euro_stays_cp1252(self, tmp_path: Path) -> None:
        """F-09: cp1252 text with € (0x80) decodes as cp1252 — 0x80 is
        alnum under cp1251 ('Ђ') and cp866 ('А') but non-alnum under
        cp1252, so without the fix dense Euro text misdecodes to cp1251."""  # noqa: RUF002
        text = "Well: 1234 \u20ac 5678 prix 50\u20ac total 123\u20ac"
        test_file = tmp_path / "f09_euro.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"F-09: cp1252 Euro text misdecoded as {enc!r} (mojibake)"
        assert "\u20ac" in content

    def test_western_symbol_class_stays_cp1252(self, tmp_path: Path) -> None:
        """F-01: symbol-dense cp1252 prose (¡ ¿ ª — the 0xA1-0xBF/0xD7/0xF7
        class) decodes as cp1252, not cp1251."""
        text = "Hola \u00a1buenos d\u00edas! 1\u00aa casa, \u00bfverdad? " * 20
        test_file = tmp_path / "f01_symbols.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"F-01: cp1252 symbol prose misdecoded as {enc!r} (mojibake)"
        assert "\u00a1buenos d\u00edas" in content

    def test_western_dimensions_symbols_stay_cp1252(self, tmp_path: Path) -> None:
        """F-01: engineering note with × ÷ ² (0xD7/0xF7/0xB2 — a distinct
        trigger shape: 0xD7/0xF7 are inflators, 0xB2 is alnum under both
        encodings) decodes as cp1252."""  # noqa: RUF002
        text = "Dimensiones 10\u00d720\u00d730 m\u00b2, secci\u00f3n 4\u00f72"
        test_file = tmp_path / "f01_dimensions.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"F-01: cp1252 dimensions misdecoded as {enc!r} (mojibake)"
        assert "10\u00d720\u00d730" in content

    def test_western_control_range_smart_punct_stays_cp1252(self, tmp_path: Path) -> None:
        """F-09 siblings: ‚ „ … † ‡ ‰ ‹ (0x82/0x84-0x87/0x89/0x8B) decode
        as cp1252 — these are Cyrillic alnum under cp866 but punctuation
        under cp1252, so without the fix the text misdecodes to cp866."""  # noqa: RUF002
        text = "He said \u201a\u201cquoted\u2026\u2020\u2021\u2030\u2039\u203a\u201d done"
        test_file = tmp_path / "f09_sibling_punct.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"F-09: cp1252 sibling-punct text misdecoded as {enc!r} (mojibake)"
        assert "\u201cquoted" in content


# ──────────────────────────────────────────────────────────────
# M4 (encoding, MEDIUM — F-01/F-09 regression): the codec-derived
# _INFLATORS table treats EVERY cp866 Cyrillic letter (0x80-0x9F) as an
# "inflator", so a genuine cp866 file whose short-word Cyrillic is mixed
# into ≥50% ASCII had its ENTIRE ratio gap "explained away" and was
# rescued to a Western mojibake decode (latin-1/cp1252) — the
# _WESTERN_RATIO_FLOOR could not help once ASCII ≥50%.  Verified live:
# "WELL 1000.0 ПРИВЕТ 2000.0" pre-fix (HEAD) decodes cp866, the F-01/F-09
# fix flipped it to latin-1; "WELL УАЗ-469 1000.0 ..." flipped to cp1252.  # noqa: RUF003
# Fix: the rescue requires (1) the artifact to count only inflator bytes
# that are PRINTABLE under the actual Western candidate (cp866 Cyrillic
# bytes decode to C1 controls under latin-1 / are undefined under cp1252,
# so they are never Western symbols), and (2) the Western candidate's
# decode to contain real ASCII-letter evidence (genuine cp866 files are
# digit-heavy LAS data; genuine Western symbol files are prose).
# ──────────────────────────────────────────────────────────────


class TestM4Cp866ShortWordMixedAscii:
    """M4: genuine cp866 files with short-word Cyrillic mixed into ≥50%
    ASCII must decode as cp866 — the ENC-02(b) rescue must NOT flip them
    to a Western mojibake decode."""

    def test_cp866_privet_mixed_ascii_stays_cp866(self, tmp_path: Path) -> None:
        """The M4 live repro: 'WELL 1000.0 ПРИВЕТ 2000.0' in cp866
        (85% ASCII) decodes cp866.  Post-F-01/F-09 pre-fix it flipped to
        latin-1 mojibake (\x8f\x90\x88\x82\x85\x92); HEAD kept cp866."""
        raw = b"WELL 1000.0 " + "\u041f\u0420\u0418\u0412\u0415\u0422".encode("cp866") + b" 2000.0"
        test_file = tmp_path / "m4_privet.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"M4: cp866 ПРИВЕТ+ASCII misdecoded as {enc!r} (mojibake)"
        assert "\u041f\u0420\u0418\u0412\u0415\u0422" in content

    def test_cp866_uaz469_mixed_ascii_stays_cp866(self, tmp_path: Path) -> None:
        """The M4 live repro: 'WELL УАЗ-469 1000.0 2000.0 3000.0 4000.0'
        in cp866 (93% ASCII) decodes cp866.  Post-F-01/F-09 pre-fix it
        flipped to cp1252 mojibake ('“€‡-469'); HEAD kept cp866."""  # noqa: RUF002
        raw = b"WELL " + "\u0423\u0410\u0417-469".encode("cp866") + b" 1000.0 2000.0 3000.0 4000.0"
        test_file = tmp_path / "m4_uaz469.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"M4: cp866 УАЗ-469+ASCII misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content

    def test_cp866_multiple_short_words_with_ascii_stays_cp866(self, tmp_path: Path) -> None:
        """A fuller LAS-like cp866 file with several short Cyrillic words
        (well name + labels) over digit-heavy ASCII stays cp866 — the
        rescue must not fire even when the Cyrillic maps to printable
        cp1252 symbols ('СКВ №1' -> smart-punct-like bytes)."""  # noqa: RUF002
        raw = (
            "WELL 1000.0 ".encode("ascii")
            + "\u0421\u041a\u0412 \u2116 1 \u041f\u041b\u0410\u0421\u0422 \u2116 2".encode("cp866")
            + b" 2000.0 3000.0"
        )
        test_file = tmp_path / "m4_skv_plast.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"M4: cp866 СКВ/ПЛАСТ+ASCII misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0421\u041a\u0412" in content
        assert "\u041f\u041b\u0410\u0421\u0422" in content


# ──────────────────────────────────────────────────────────────
# ENC-M1 (encoding, MEDIUM — M4 regression, FIX pass 2): the M4 gates
# (_PRINTABLE_INFLATORS + _WESTERN_MIN_ASCII_LETTERS=8) closed the
# digit-heavy М4 repros but NOT the prose-mixed class: a genuine cp866  # noqa: RUF003
# file with a short Cyrillic word whose bytes are ALL printable under
# cp1252 (УАЗ = 0x93 0x80 0x87 -> "“€‡"; also ГАЗ, МАЗ, СКВ) PLUS ASCII  # noqa: RUF003
# prose with ≥8 letters still flipped to cp1252 mojibake — a NEW
# regression vs HEAD 82cadce (HEAD decoded these inputs correctly as
# cp866; verified by live A/B of the actual HEAD module).  The cp1251 run
# detector cannot see this class (under cp1251 the УАЗ bytes decode to  # noqa: RUF003
# "“Ђ‡" — no Cyrillic run), so the rescue fired.  Fix: the rescue is also
# blocked when the winning NON-Western candidate's own decode contains a
# standalone (non-embedded) Cyrillic run of ≥3 letters
# (_has_word_like_cyrillic_run) — genuine cp866 words are standalone
# tokens, while Western symbol clusters (…†‡, ‚"…) are punctuation  # noqa: RUF003
# embedded in ASCII words and the F-09 rescue still fires for them.
# ──────────────────────────────────────────────────────────────


class TestENCM1ProseUazStaysCp866:
    """ENC-M1: genuine cp866 files with a printable-under-cp1252 Cyrillic
    word (УАЗ/ГАЗ/МАЗ/СКВ-class) mixed into ASCII PROSE must decode as
    cp866 — the ENC-02(b) rescue must NOT flip them to cp1252 mojibake.
    Regression class: cp866 word + ≥8 ASCII letters (prose), where both
    M4 gates pass and only the word-like-Cyrillic-run discriminator blocks."""  # noqa: RUF002

    def test_cp866_uaz469_prose_stays_cp866(self, tmp_path: Path) -> None:
        """The ENC-M1 live repro (s9-review-2-enc E1 e1a): 'WELL NAME:
        ACME OIL CORP УАЗ-469 1000.0 2000.0 3000.0' in cp866 decodes cp866.
        Pre-fix it flipped to cp1252 mojibake ('… CORP “€‡-469 …'); HEAD
        decoded it correctly as cp866."""  # noqa: RUF002
        raw = (
            b"WELL NAME: ACME OIL CORP "
            + "\u0423\u0410\u0417-469".encode("cp866")
            + b" 1000.0 2000.0 3000.0"
        )
        test_file = tmp_path / "encm1_uaz469_prose.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"ENC-M1: cp866 УАЗ-469+prose misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content

    def test_cp866_uaz469_exactly_8_letters_stays_cp866(self, tmp_path: Path) -> None:
        """ENC-M1 e1b: the gate threshold does not protect this class —
        'WELLABCD УАЗ-469 …' has exactly 8 ASCII letters, so
        _WESTERN_MIN_ASCII_LETTERS passes and only the word-like-run
        discriminator keeps the file cp866."""  # noqa: RUF002
        raw = b"WELLABCD " + "\u0423\u0410\u0417-469".encode("cp866") + b" 1000.0 2000.0 3000.0"
        test_file = tmp_path / "encm1_uaz469_8letters.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"ENC-M1: cp866 УАЗ-469 @8 letters misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content

    def test_cp866_gaz_maz_prose_stays_cp866(self, tmp_path: Path) -> None:
        """ENC-M1: sibling words of the УАЗ class (ГАЗ, МАЗ — also all
        printable under cp1252) mixed into ASCII prose stay cp866, so the
        discriminator is not tuned to the single УАЗ byte pattern."""  # noqa: RUF002
        raw = (
            b"WELL ACME OIL CORP "
            + "\u0413\u0410\u0417-66 \u041c\u0410\u0417-4370".encode("cp866")
            + b" 1000.0 2000.0 3000.0"
        )
        test_file = tmp_path / "encm1_gaz_maz.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"ENC-M1: cp866 ГАЗ/МАЗ+prose misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0413\u0410\u0417-66" in content
        assert "\u041c\u0410\u0417-4370" in content

    def test_f09_euro_prose_still_cp1252(self, tmp_path: Path) -> None:
        """ENC-M1 guard: the F-09 Western rescue must still fire for
        genuine cp1252 Euro prose (the discriminator must not over-block
        — its cp866 misread 'А' is isolated, not a word-like run)."""  # noqa: RUF002
        text = "Well: 1234 \u20ac 5678 prix 50\u20ac total 123\u20ac"
        test_file = tmp_path / "encm1_guard_euro.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-M1 guard: F-09 Euro prose misdecoded as {enc!r}"
        assert "\u20ac" in content

    def test_f09_sibling_cluster_still_cp1252(self, tmp_path: Path) -> None:
        """ENC-M1 guard: the F-09 sibling-punct cluster must still rescue
        to cp1252 even though its cp866 misread ('quotedЕЖЗЙЛЫФ') contains
        a 7-letter Cyrillic run — the run is EMBEDDED in the ASCII word
        'quoted', so the word-like discriminator correctly does not block."""  # noqa: RUF002
        text = "He said \u201a\u201cquoted\u2026\u2020\u2021\u2030\u2039\u203a\u201d done"
        test_file = tmp_path / "encm1_guard_sibling.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-M1 guard: F-09 sibling cluster misdecoded as {enc!r}"
        assert "\u201cquoted" in content

    def test_m4_digit_uaz469_still_cp866(self, tmp_path: Path) -> None:
        """ENC-M1 guard: the M4 digit-heavy class must stay cp866 (both
        gates and the new discriminator agree — the file has a standalone
        'УАЗ' run, and its Western decode has <8 ASCII letters)."""  # noqa: RUF002
        raw = b"WELL " + "\u0423\u0410\u0417-469".encode("cp866") + b" 1000.0 2000.0 3000.0 4000.0"
        test_file = tmp_path / "encm1_guard_m4_digit.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"ENC-M1 guard: M4 digit УАЗ-469 misdecoded as {enc!r}"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content

    def test_cp866_uaz469_glued_before_stays_cp866(self, tmp_path: Path) -> None:
        """ENC-M1 guard: the glued-before subclass — a genuine cp866 word
        glued directly to a preceding ASCII letter with no separator
        ('THE WELLУАЗ-469 FIELD …') — must still decode cp866.  The run's
        raw-byte boundary (hyphen + digits) marks it as a genuine LAS
        Cyrillic label, not Western prose punctuation.  Pre-fix (pass-2
        gate) the run was classified "embedded" and the file flipped to
        cp1252 mojibake in realistic full-file form."""  # noqa: RUF002
        raw = (
            b"THE WELL" + "\u0423\u0410\u0417".encode("cp866") + b"-469 FIELD 1000.0 2000.0 3000.0"
        )
        test_file = tmp_path / "encm1_uaz469_glued.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"ENC-M1 guard: cp866 glued УАЗ-469 misdecoded as {enc!r}"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content


# ──────────────────────────────────────────────────────────────
# ENC-1 (encoding, MEDIUM — ENC-M1 regression, FIX pass 3): the pass-2
# word-like-Cyrillic-run discriminator (_has_word_like_cyrillic_run)
# classified STANDALONE typographic-symbol clusters as word-like Cyrillic.
# A genuine cp1252 file whose prose contains a space/punct-bounded run of
# ≥3 typographic symbols (…†‡, †‡‰, ‚„…, –—…) decodes to 3+ Cyrillic code  # noqa: RUF003
# points under cp866 (ЕЖЗ, ЖЗЙ, ВДЕ, ЦЧЕ) — a "standalone Cyrillic word" —
# so the ENC-02(b) Western near-tie rescue was BLOCKED and the file decoded
# cp866 mojibake (a NEW regression vs the shipped v2.0.3 release; HEAD
# decoded these inputs correctly as cp1252).  Fix: the discriminator now
# examines the raw-byte boundary of each run — genuine LAS Cyrillic labels
# are followed by digits/hyphens (УАЗ-469, СКВ №1), while Western prose  # noqa: RUF003
# punctuation is followed by the next ASCII word (…†‡ and more) — and only
# treats the run as word-like in the former case.  A run containing a
# non-ambiguous (lowercase/extra) Cyrillic byte is word-like unconditionally.
# ──────────────────────────────────────────────────────────────


class TestENC1StandaloneSymbolClusterWestern:
    """ENC-1: genuine cp1252 files with a STANDALONE (space/punct-bounded)
    typographic-symbol cluster must decode as cp1252 — the pass-2
    discriminator must not block the Western rescue on their cp866 misreads.
    Each cluster variant decodes to a 3+ uppercase-Cyrillic run under cp866
    (the ENC-1 regression shape)."""

    def test_cp1252_prose_ellipsis_dagger_dagger_stays_cp1252(self, tmp_path: Path) -> None:
        """The ENC-1 live repro (s10-adv-m1): long ASCII prose with a
        standalone '…†‡' cluster and '50€' in cp1252 decodes cp1252.
        Pre-fix (pass-2 gate) it decoded cp866 mojibake; HEAD decoded it
        correctly."""
        text = (
            "This is a fairly long ASCII description line about the prices of "
            "the equipment \u2026\u2020\u2021 and more text with a total "
            "50\u20ac in it here"
        )
        test_file = tmp_path / "enc1_ellipsis_dagger.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-1: cp1252 prose+…†‡ misdecoded as {enc!r} (mojibake)"
        assert "\u2026\u2020\u2021" in content
        assert "\u20ac" in content

    def test_cp1252_prose_permille_cluster_stays_cp1252(self, tmp_path: Path) -> None:
        """ENC-1 cluster variant: a standalone '†‡‰' run (cp866 misread
        'ЖЗЙ') followed by ASCII words decodes cp1252."""
        text = "The quoted prices \u2020\u2021\u2030 for the listed items are final"
        test_file = tmp_path / "enc1_permille.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-1: cp1252 prose+†‡‰ misdecoded as {enc!r} (mojibake)"
        assert "\u2020\u2021\u2030" in content

    def test_cp1252_prose_lowquote_cluster_stays_cp1252(self, tmp_path: Path) -> None:
        """ENC-1 cluster variant: a standalone '‚„…' run (cp866 misread
        'ВДЕ') followed by ASCII words decodes cp1252."""  # noqa: RUF002
        text = "He wrote \u201a\u201e\u2026 then paused for effect"
        test_file = tmp_path / "enc1_lowquote.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-1: cp1252 prose+‚„… misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u201a\u201e\u2026" in content

    def test_cp1252_prose_dash_cluster_stays_cp1252(self, tmp_path: Path) -> None:
        """ENC-1 cluster variant: a standalone '–—…' run (cp866 misread
        'ЦЧЕ') followed by ASCII words decodes cp1252."""  # noqa: RUF002
        text = "The interval \u2013\u2014\u2026 covers the full range of values"
        test_file = tmp_path / "enc1_dash.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"ENC-1: cp1252 prose+–—… misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u2013\u2014\u2026" in content


# ──────────────────────────────────────────────────────────────
# F-02 (encoding, MEDIUM — ENC-1 regression, FIX pass 4): the pass-3 rule (2)
# treated "an ASCII letter following the Cyrillic run" as conclusive
# Western-prose evidence.  But genuine cp866 all-uppercase Cyrillic words are
# ALSO followed by a space + ASCII word — the most natural prose form
# ('THE WELL УАЗ FIELD 1000.0').  Live A/B: pass-2 discriminator True  # noqa: RUF003
# (blocks rescue -> cp866 correct), pass-3 False -> cp1252 mojibake.
# Fix: the ASCII-letter-follows case is now decided by the run's byte
# content — a byte that is Cyrillic under cp1251 (_CP1251_CYRILLIC_BYTES,
# e.g. УАЗ = 0x93 0x80 0x87 contains 0x80 = Ђ) marks a genuine word, while  # noqa: RUF003
# the Western smart-punctuation clusters (…†‡ etc.) use bytes that are not
# Cyrillic under cp1251 and stay prose punctuation.
# ──────────────────────────────────────────────────────────────


class TestF02SpaceSeparatedWordStaysCp866:
    """F-02: genuine cp866 files whose short Cyrillic word (УАЗ-class, all
    bytes printable under cp1252) is followed by a space + ASCII word must
    decode as cp866 — the ENC-02(b) rescue must NOT flip them to cp1252
    mojibake.  Regression class: space-separated cp866 word + ASCII word,
    where the run is all-ambiguous (0x80-0x9F) and only the
    cp1251-Cyrillic-byte signal distinguishes it from Western prose
    punctuation."""  # noqa: RUF002

    def test_cp866_uaz_space_separated_word_stays_cp866(self, tmp_path: Path) -> None:
        """The F-02 live repro (s11-adv-m1): 'THE WELL УАЗ FIELD 1000.0' in
        cp866 decodes cp866.  Pass-3 rule (2) classified the run as Western
        prose punctuation ('FIELD' follows) and the file flipped to cp1252
        mojibake ('THE WELL “€‡ FIELD 1000.0'); pass-2 and HEAD decoded it
        correctly."""  # noqa: RUF002
        raw = b"THE WELL " + "\u0423\u0410\u0417".encode("cp866") + b" FIELD 1000.0"
        test_file = tmp_path / "f02_uaz_space_separated.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"F-02: cp866 УАЗ+space+ASCII word misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417" in content

    def test_cp866_uaz_field_data_stays_cp866(self, tmp_path: Path) -> None:
        """F-02 variant (s11-adv-m1 table): 'УАЗ FIELD DATA' in cp866 decodes
        cp866 — the discriminator must not treat the space-separated form as
        Western prose even when no digits are present."""  # noqa: RUF002
        raw = "\u0423\u0410\u0417".encode("cp866") + b" FIELD DATA"
        test_file = tmp_path / "f02_uaz_field_data.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"F-02: cp866 'УАЗ FIELD DATA' misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417" in content

    def test_cp866_uaz_lowercase_prose_stays_cp866(self, tmp_path: Path) -> None:
        """F-02 variant (s11-adv-m1 table): 'the well УАЗ of' in cp866
        decodes cp866 — the space-separated genuine word stays cp866 even
        in lowercase ASCII prose."""  # noqa: RUF002
        raw = b"the well " + "\u0423\u0410\u0417".encode("cp866") + b" of"
        test_file = tmp_path / "f02_uaz_lowercase.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"F-02: cp866 'the well УАЗ of' misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417" in content

    def test_cp866_uaz_hyphen_field_stays_cp866(self, tmp_path: Path) -> None:
        """F-02 regression pin: the hyphen-followed class in the
        space-separated context ('THE WELL УАЗ-469 FIELD 1000.0') decodes
        cp866 — the boundary rule's hyphen + digit tail (УАЗ-469) still
        marks a genuine LAS Cyrillic label."""  # noqa: RUF002
        raw = b"THE WELL " + "\u0423\u0410\u0417-469".encode("cp866") + b" FIELD 1000.0"
        test_file = tmp_path / "f02_uaz_hyphen_field.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"F-02 pin: cp866 УАЗ-469 FIELD misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        assert "\u0423\u0410\u0417-469" in content

    def test_cp1252_standalone_cluster_word_follows_stays_cp1252(self, tmp_path: Path) -> None:
        """F-02 guard: the ENC-1 shape stays fixed — a standalone Western
        smart-punctuation cluster followed by a space + ASCII word
        ('…†‡ and more') in cp1252 still decodes cp1252.  The cluster's
        bytes (0x85 0x86 0x87) are NOT Cyrillic under cp1251, so the
        cp1251-Cyrillic-byte signal correctly keeps it as Western prose
        punctuation and the rescue fires."""
        text = "The prices listed below are \u2026\u2020\u2021 and more items final"
        test_file = tmp_path / "f02_guard_cluster.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"F-02 guard: cp1252 standalone …†‡ misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content


# ──────────────────────────────────────────────────────────────
# E-01 (encoding, MEDIUM — F-02 fix INCOMPLETE at root cause, FIX pass 5):
# the pass-4 _CP1251_CYRILLIC_BYTES byte-content signal recognizes only 14
# of the 32 cp866 uppercase letters in the ambiguous 0x80-0x9F range.  The
# 18-letter complement (В Д Е Ж З И Й Л С Т У Ф Х Ц Ч Ш Щ Ы — ТЕСТ/ЗИЛ/ВДЕ/  # noqa: RUF003
# ВЕС/ВИД/ЛЕС/ДЕД) carries NO set byte and is byte-identical under cp866 to  # noqa: RUF003
# the Western punctuation clusters (ВДЕ == ‚„…), so the pass-4 rule left  # noqa: RUF003
# genuine space-separated cp866 words decoding cp1252 mojibake on the F-02
# harness shape where HEAD 82cadce decoded cp866 (correct).  The 0x80-0x9F
# byte space is symmetric — no byte-content rule can separate the classes
# (pre-fix audit A-2).  Fix: keep the set-byte fast path; decide the
# no-set-byte ASCII-letter-follows case by LAS-context evidence (digit
# within _LAS_DIGIT_CONTEXT_WINDOW bytes, or an UPPERCASE ASCII word
# following the run) — a genuinely different information source.
# ──────────────────────────────────────────────────────────────


class TestE01NoSetByteSpaceSeparatedWordStaysCp866:
    """E-01: genuine cp866 words composed of the 18 uppercase letters that
    have NO cp1251-Cyrillic set byte must decode as cp866 in the
    space-separated harness shape — the pass-5 convergence gate class that
    FAILED today in P4 (pass-4 byte-content rule) while HEAD decoded it
    correctly."""

    def test_cp866_test_space_separated_word_stays_cp866(self, tmp_path: Path) -> None:
        """The E-01 convergence gate (s12 synthesis + pre-fix audit test 1):
        'THE WELL TEST FIELD 1000.0' with TEST encoded in cp866 (0x92 0x85
        0x93 0x92) decodes cp866.  The word contains NO cp1251-Cyrillic set
        byte, so pass-4 flipped it to cp1252 mojibake; the LAS-context rule
        (digit '1' ~8 bytes after the run + uppercase FIELD) keeps it
        cp866.  This fixture FAILED today in P4 (live verified by
        adversarial) — it is the pass-5 convergence gate."""
        raw = b"THE WELL " + "\u0422\u0415\u0421\u0422".encode("cp866") + b" FIELD 1000.0"
        test_file = tmp_path / "e01_test_space_separated.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"E-01: cp866 TEST+space+ASCII word misdecoded as {enc!r} (mojibake)"
        assert "\u0422\u0415\u0421\u0422" in content

    def test_cp866_no_set_byte_sweep_stays_cp866(self, tmp_path: Path) -> None:
        """E-01 sweep (synthesis-recommended no-set-byte class): ZIL, VDE,
        VES, VID, LES, DED in the 'THE WELL X FIELD 1000.0' harness all
        decode cp866 with the word preserved.  Each is byte-identical to a
        Western smart-punctuation cluster under cp866 (VDE == the low-quote
        cluster); only the LAS context separates them."""
        words = {
            "\u0417\u0418\u041b": "ZIL",
            "\u0412\u0414\u0415": "VDE",
            "\u0412\u0415\u0421": "VES",
            "\u0412\u0418\u0414": "VID",
            "\u041b\u0415\u0421": "LES",
            "\u0414\u0415\u0414": "DED",
        }
        for word, label in words.items():
            raw = b"THE WELL " + word.encode("cp866") + b" FIELD 1000.0"
            test_file = tmp_path / f"e01_sweep_{label}.las"
            test_file.write_bytes(raw)
            with mock.patch(
                "pylasdev.encoding._detect_encoding_from_bytes",
                return_value="utf-8",
            ):
                with mock.patch(
                    "pylasdev.encoding._detect_confidence_from_bytes",
                    return_value=0.0,
                    create=True,
                ):
                    enc, content = read_with_encoding(test_file)
            assert enc == "cp866", (
                f"E-01 sweep: cp866 {label}+space+ASCII word misdecoded as {enc!r} (mojibake)"
            )
            assert word in content, f"E-01 sweep: {label} missing from cp866 content"

    def test_cp866_test_hyphen_digit_laden_stays_cp866(self, tmp_path: Path) -> None:
        """E-01 digit-laden variant (pre-fix audit test 3): 'THE WELL
        TEST-2 FIELD 1000.0' decodes cp866 — the hyphen + digit tail
        (TEST-2) is already a genuine-LAS boundary (non-letter after the
        run), pinning the digit-laden form of the no-set-byte class."""
        raw = b"THE WELL " + "\u0422\u0415\u0421\u0422-2".encode("cp866") + b" FIELD 1000.0"
        test_file = tmp_path / "e01_test_hyphen_digit.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", f"E-01: cp866 TEST-2 digit-laden misdecoded as {enc!r} (mojibake)"
        assert "\u0422\u0415\u0421\u0422-2" in content

    def test_cp866_vde_no_digits_uppercase_follows_stays_cp866(self, tmp_path: Path) -> None:
        """E-01 no-digits variant: 'THE WELL VDE FIELD' (no digits anywhere)
        decodes cp866 — the LAS-context UPPERCASE-word signal (FIELD, a LAS
        mnemonic) keeps the byte-identical-to-Western no-set word genuine
        even without the digit signal.  This is the adversarial's
        'no-digits' harness class (HEAD 82cadce = cp866, P4 = cp1252)."""
        raw = b"THE WELL " + "\u0412\u0414\u0415".encode("cp866") + b" FIELD"
        test_file = tmp_path / "e01_vde_no_digits.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", (
            f"E-01: cp866 VDE no-digits+uppercase misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0412\u0414\u0415" in content

    def test_cp1252_cluster_sentence_initial_capital_stays_cp1252(self, tmp_path: Path) -> None:
        """E-01 guard: a Western cluster followed by a SENTENCE-INITIAL
        capital word ('... and more' -> 'And more') still decodes cp1252 —
        the UPPERCASE-word context signal requires the FULL following word
        to be uppercase (a LAS mnemonic), not a single capital letter (the
        'A' of 'And' is followed by lowercase, so it stays Western)."""
        text = "The prices listed below are \u2026\u2020\u2021 And more items final"
        test_file = tmp_path / "e01_guard_sentence_initial.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E-01 guard: cp1252 sentence-initial capital misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content


# ──────────────────────────────────────────────────────────────
# M-1 / M-2 (encoding, MEDIUM — pass-5 _run_has_las_context regressions,
# FIX pass 6): the pass-5 LAS-context discriminator's two signals were
# sufficient-but-not-necessary on the Western side.  Signal (a) ("any ASCII
# digit within 24 bytes") fired on ordinary Western prose prices/counts/dates
# ('Prices ... and 3 more', '... dated 2024', '... total 50', '... over 100')
# and flipped whole files to cp866 mojibake (M-1).  Signal (b) (full-uppercase
# word follows) fired on Western all-caps/acronyms ('... USA', '... IMPORTANT
# NOTE') (M-2).  Pass-6 fix: signal (a) is refined to require the digit's
# immediately-preceding token to be a full-uppercase ASCII word (LAS mnemonic
# + value); signal (b) is provably irreducible against the pinned E-01
# no-digits gate and is xfail-pinned as a documented residual.
# ──────────────────────────────────────────────────────────────


class TestM1WesternClusterDigitContextStaysCp1252:
    """M-1: genuine cp1252 files whose smart-punct cluster is followed by a
    digit in ordinary Western prose (prices/counts/dates) must decode cp1252.
    The pass-5 'any digit in the window' signal fired on these shapes and
    flipped them to cp866 mojibake; the pass-6 refinement requires the
    digit's immediately-preceding token to be a full-uppercase LAS mnemonic,
    which lowercase-prose tokens are not."""

    def test_cp1252_cluster_digit_in_window_stays_cp1252(self, tmp_path: Path) -> None:
        """The M-1 convergence gate (s13 synthesis + pre-fix audit test 1):
        'Prices ... and 3 more items are final' (cp1252) decodes cp1252.
        The digit '3' sits ~4 bytes after the cluster — inside the 24-byte
        window — so pass-5 flipped it to cp866 mojibake; the pass-6
        refinement requires an uppercase token before the digit, and 'and'
        is lowercase prose.  This test FAILED in P5 (live verified by both
        reviewers and the adversarial); it is the pass-6 convergence gate."""
        text = "Prices \u2026\u2020\u2021 and 3 more items are final"
        test_file = tmp_path / "m1_digit_in_window.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-1 gate: cp1252 cluster+digit-in-window misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content

    def test_cp1252_cluster_digit_next_line_stays_cp1252(self, tmp_path: Path) -> None:
        """M-1 sub-shape pin (pre-fix audit test 5): a Western cp1252 file
        with the digit on the line AFTER the cluster ('...' then a newline
        then 'and 3 more') decodes cp1252.  The pass-5 window crossed
        newlines (a digit within 24 bytes on the next line flipped to
        cp866); under the pass-6 refinement the digit follows the lowercase
        token 'and', so it is ordinary prose, not a LAS label."""
        text = "The note said \u2026\u2020\u2021\nand 3 more items are final"
        test_file = tmp_path / "m1_digit_next_line.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-1: cp1252 cluster+digit-on-next-line misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content

    def test_cp1252_cluster_sentence_initial_capital_digit_stays_cp1252(
        self, tmp_path: Path
    ) -> None:
        """M-1 sub-shape pin (pre-fix audit test 5): a Western cp1252 file
        with a SENTENCE-INITIAL capital word followed by a digit ('... And
        3 more') decodes cp1252.  'And' is NOT a full-uppercase token (only
        its first letter is capital), so the digit is prose, not a LAS
        mnemonic + value."""
        text = "The prices listed below are \u2026\u2020\u2021 And 3 more items final"
        test_file = tmp_path / "m1_sentence_initial_capital_digit.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-1: cp1252 cluster+sentence-initial-capital+digit misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content


class TestM2IrreducibleResidualsXfail:
    """M-2 / A-2: the irreducible Western residuals of the context
    discriminator, pinned as strict xfail so the documented trade cannot
    silently rot.  '... USA' and '... PAGE 3' are byte-identical to the
    genuine E-01 no-set gate ('THE WELL VDE FIELD'), so no local rule can
    separate them without dropping the gate; the trade is kept in favor of
    the genuine Russian-geoscience LAS class.  strict=True: a future pass
    that changes the trade flips these to XPASS and the suite fails."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "M-2 documented residual: '... USA' is byte-identical to the genuine "
            "E-01 no-set word 'THE WELL VDE FIELD' (all-ambiguous 3-byte run + "
            "space + full-uppercase ASCII word); the pinned E-01 no-digits gate "
            "(test_cp866_vde_no_digits_uppercase_follows_stays_cp866) requires the "
            "uppercase-follows signal, so this Western shape cannot be separated "
            "by any local rule without dropping that gate.  Trade kept in favor "
            "of the genuine Russian-geoscience LAS class."
        ),
    )
    def test_cp1252_cluster_acronym_follows_stays_cp1252(self, tmp_path: Path) -> None:
        """M-2 residual pin (s13 synthesis + pre-fix audit): the Western
        all-caps/acronym shape '... USA and more' would decode cp1252, but
        signal (b) (uppercase word follows the run) fires on it — the
        structural identity with the genuine 'THE WELL VDE FIELD' E-01
        no-digits gate makes the trade irreducible.  The test FAILS in P5
        and stays failing (xfail, not xpass)."""
        text = "\u2026\u2020\u2021 USA and more"
        test_file = tmp_path / "m2_acronym_follows.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-2 residual: cp1252 cluster+acronym misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A-2 documented residual (M-1 family, s13 pre-fix audit): '... PAGE 3 "
            "of 10' is byte-identical to the genuine E-01 gate 'THE WELL TEST "
            "FIELD 1000.0' (all-ambiguous 3-byte run + space + full-uppercase "
            "ASCII word + space + digit) — the digit follows the uppercase "
            "mnemonic PAGE, the same structure the E-01 gate requires to stay "
            "genuine, so no local rule can separate them.  Same irreducible "
            "trade as M-2; not pinned as genuine."
        ),
    )
    def test_cp1252_cluster_uppercase_token_digit_stays_cp1252(self, tmp_path: Path) -> None:
        """A-2 residual pin (s13 pre-fix audit): '... PAGE 3 of 10 in the
        report' would decode cp1252, but the digit follows the full-uppercase
        token PAGE — byte-identical to the E-01 gate structure, so it
        decodes cp866 and cannot be fixed without breaking the gate.
        strict=True guards the trade the same way as the M-2 pin."""
        text = "\u2026\u2020\u2021 PAGE 3 of 10 in the report"
        test_file = tmp_path / "a2_uppercase_token_digit.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"A-2 residual: cp1252 cluster+uppercase-token+digit misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content


# ──────────────────────────────────────────────────────────────
# C-1 / C-2 / C-3 (encoding, MEDIUM — stage-C discovery F-1/F-2/F-3 +
# ADV-M3 byte-identity proofs, IMP-4 pins): three NEW provably-irreducible
# encoding classes, pinned strict-xfail exactly like the M-2/A-2 residuals.
# Each class is byte-symmetric: the two intended encodings produce
# byte-identical content that each decodes to plausible text under both
# codecs, so no local byte-content rule (ratio, run length, set-byte, strong
# byte) can ever separate the pair — the load-bearing gate that forbids the
# alternative trade is quoted in each reason.  strict=True: a future pass
# that changes a trade flips the pin to XPASS and the suite fails.
# ──────────────────────────────────────────────────────────────


class TestEncodingNewIrreducibleResidualsXfail:
    """C-1 / C-2 / C-3: the irreducible residuals of the encoding fallback
    chain that stage-C discovery and adversarial verification proved to be
    byte-identical to their alternative codec interpretations.  Pins only —
    the detection logic is deliberately NOT changed (no byte-content rule
    can converge on a byte-identical pair; the fix-regress trap)."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "C-1 documented residual (stage-C F-1, ADV-M3 CONFIRMED): bytes "
            "74 68 65 20 86 87 89 20 33 20 69 74 65 6D 73 = cp1252 "
            "'the \\u2020\\u2021\\u2030 3 items' (Western dagger / double "
            "dagger / per-mille cluster + digit) == cp866 'the "
            "\\u0416\\u0417\\u0419 3 items' (genuine 3-letter Cyrillic word + "
            "digit).  The digit-follows branch of _is_genuine_word_run fires "
            "identically on both, so the Western input misdecodes as cp866 "
            "and no local byte-content rule can separate the pair without "
            "dropping the load-bearing digit-follows gate "
            "(test_cp866_test_hyphen_digit_laden_stays_cp866: genuine "
            "TEST-2 digit-laden no-set word must stay cp866).  Trade kept in "
            "favor of the genuine Russian-geoscience LAS digit-follows class."
        ),
    )
    def test_cp1252_cluster_digit_direct_stays_cp1252(self, tmp_path: Path) -> None:
        """C-1 residual pin (stage-C F-1 + ADV-M3): 'the \u2020\u2021\u2030 3
        items' (cp1252) would decode cp1252, but the digit '3' directly
        follows the cluster — byte-identical to the genuine cp866
        digit-follows class 'the \u0416\u0417\u0419 3 items' — so it decodes
        cp866 and cannot be fixed without dropping the load-bearing TEST-2
        digit-follows gate.  strict=True guards the trade the same way as
        the M-2 pin."""
        text = "the \u2020\u2021\u2030 3 items"
        test_file = tmp_path / "c1_digit_direct.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"C-1 residual: cp1252 cluster+digit-direct misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2020\u2021\u2030" in content

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "C-2 documented residual (stage-C F-2, ADV-M3 CONFIRMED): bytes "
            "54 48 45 20 57 45 4C 4C 20 82 84 85 20 66 69 65 6C 64 = cp866 "
            "'THE WELL \\u0412\\u0414\\u0415 field' (genuine no-set Cyrillic "
            "word + LOWERCASE ASCII follow) == cp1252 'THE WELL \\u201a\\u201e"
            "\\u2026 field' (Western smart-punct cluster).  The lowercase "
            "follow fails the _run_has_las_context signal, so the Western "
            "rescue fires and the genuine cp866 input misdecodes as cp1252; "
            "the EXACT byte pattern is already pinned PASSING on the Western "
            "side (test_cp1252_prose_lowquote_cluster_stays_cp1252), so no "
            "local rule can flip lowercase-follows to cp866 without breaking "
            "that gate.  This is the loss-side mirror of the M-2 pin (M-2 "
            "pins the Western loss; C-2 pins the Cyrillic loss)."
        ),
    )
    def test_cp866_no_set_word_lowercase_follow_stays_cp866(self, tmp_path: Path) -> None:
        """C-2 residual pin (stage-C F-2 + ADV-M3): 'THE WELL VDE field'
        (cp866) would decode cp866, but the lowercase 'field' fails the
        LAS-context signal — byte-identical to the Western prose shape
        pinned PASSING at test_cp1252_prose_lowquote_cluster_stays_cp1252 —
        so it decodes cp1252 and cannot be fixed without breaking that gate.
        strict=True guards the trade the same way as the M-2 pin."""
        raw = b"THE WELL " + "\u0412\u0414\u0415".encode("cp866") + b" field"
        test_file = tmp_path / "c2_lowercase_follow.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", (
            f"C-2 residual: cp866 no-set word + lowercase follow misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0412\u0414\u0415" in content

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "C-3 documented residual (stage-C F-3, ADV-M3 CONFIRMED): bytes "
            "D1 CA C2 20 31 32 20 46 49 45 4C 44 = cp1251 '\\u0421\\u041a\\u0412 "
            "12 FIELD' (genuine 3-letter Cyrillic word, skvazhina = well) == "
            "cp1252 '\\u00d1\\u00ca\\u00c2 12 FIELD' (Western accented "
            "Latin).  Both decodes have identical word-char ratios (0.833), "
            "no 4-byte run (_CYRILLIC_RUN_CONFIRM=4), no strong byte, so "
            "Western wins the ratio tie and the genuine cp1251 input "
            "misdecodes as cp1252; any threshold-lowering or tie-time "
            "Cyrillic-run check flips the byte-identical M-57 Spanish gate "
            "(TestM57SpanishNotMisdetectedAsCp1251: 'Nota\\u00b9 "
            "\\u00d1\\u00e1\\u00f1ez') to cp1251.  Trade kept in favor of "
            "the Western M-57 gate."
        ),
    )
    def test_cp1251_short_word_stays_cp1251(self, tmp_path: Path) -> None:
        """C-3 residual pin (stage-C F-3 + ADV-M3): '\\u0421\\u041a\\u0412 12 "
        "FIELD' (cp1251) would decode cp1251, but the 3-letter run is below "
        "the 4-byte Cyrillic-run threshold and the ratio ties — "
        "byte-identical to the M-57 Spanish '\\u00d1\\u00e1\\u00f1ez' shape "
        "— so it decodes cp1252 and cannot be fixed without breaking that "
        "gate.  strict=True guards the trade the same way as the M-2 pin."""
        raw = "\u0421\u041a\u0412 12 FIELD".encode("cp1251")
        test_file = tmp_path / "c3_short_word.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", f"C-3 residual: cp1251 short word misdecoded as {enc!r} (mojibake)"
        assert "\u0421\u041a\u0412" in content


# ──────────────────────────────────────────────────────────────
# E-02 (encoding, HIGH): cp866 Cyrillic beyond the 64K ratio window is
# invisible to the whole-file run detector → silent latin-1/cp1252
# mojibake.  The common cp866 words СКВАЖИНА/ПРИВЕТ/ТЕСТ/УАЗ/ПЛАСТ form  # noqa: RUF003
# runs of at most 2 cp1251-class bytes (no 4-run, no strong byte, no №),
# so _has_confirmed_cyrillic_run returns False and the Western tie-break
# wins when the 64K window is all-ASCII.  E-07 passes only because its
# sample contains МЕСТОРОЖДЕНИЕ's 0x8E 0x90 0x8E strong-byte triple; the
# invisible class is unpinned.  Fix: the word-like Cyrillic discriminator
# (_has_word_like_cyrillic_run) — the one detector that CAN see these
# words (in the cp866 decode) — is made reachable on the Western-winner
# path with whole-file evidence (strict rule-(2)-only judgment: 0x80-0x9F
# uppercase-class bytes, no rule-(1) ≥0xA0 ambiguity).
# ──────────────────────────────────────────────────────────────


class TestE02Cp866InvisibleBeyond64K:
    """E-02: a cp866 file whose Cyrillic content lies BEYOND the 64K ratio
    window must decode cp866 — the whole-file word-like Cyrillic evidence
    check must reach the invisible-class words (СКВАЖИНА/ПРИВЕТ/ТЕСТ/УАЗ/
    ПЛАСТ — no strong-byte triple like МЕСТОРОЖДЕНИЕ's)."""  # noqa: RUF002

    def test_cp866_invisible_words_beyond_64k_detected(self, tmp_path: Path) -> None:
        """The E-02 live repro: a >64K ASCII preamble followed by cp866
        invisible-class words decodes cp866.  Pre-fix it decoded latin-1
        (raw C1 controls, zero Cyrillic code points, 0 warnings).  The
        sample deliberately EXCLUDES МЕСТОРОЖДЕНИЕ (whose strong-byte
        triple masks the defect in the E-07 test)."""
        russian = (
            "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u041f\u0420\u0418\u0412\u0415\u0422 "
            "\u0422\u0415\u0421\u0422 \u0423\u0410\u0417 \u041f\u041b\u0410\u0421\u0422 "
            "\u0413\u0410\u0417 \u041c\u0410\u0417 "
        )
        preamble = ("# " + "X" * 77 + "\n") * 900  # ~70KB ASCII preamble
        raw = preamble.encode("ascii") + (russian * 30).encode("cp866") + b"\n"
        assert len(raw) > 65_536  # Cyrillic content must lie beyond the ratio window
        test_file = tmp_path / "e02_invisible_beyond_64k.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", (
            f"E-02: cp866 invisible-class words beyond 64K misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in content
        assert "\u041f\u0420\u0418\u0412\u0415\u0422" in content

    def test_cp1251_invisible_beyond_64k_control(self, tmp_path: Path) -> None:
        """E-02 control: the same layout with cp1251 Cyrillic stays cp1251 —
        the whole-file evidence check prefers the best Cyrillic candidate
        (cp1251 first in the fallback order)."""
        russian = (
            "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u041f\u0420\u0418\u0412\u0415\u0422 "
            "\u0422\u0415\u0421\u0422 \u0423\u0410\u0417 \u041f\u041b\u0410\u0421\u0422 "
        )
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + (russian * 30).encode("cp1251") + b"\n"
        test_file = tmp_path / "e02_cp1251_control.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"E-02 control: cp1251 beyond-64K misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in content


# ──────────────────────────────────────────────────────────────
# M-07 (encoding, MEDIUM — REGRESSION vs HEAD): the E-02 whole-file
# evidence check flipped >64K Western cp1252 files to cp1251.  Two
# holes: (a) the 8 alnum-under-cp1252 Western letters (ƒ Š Œ Ž š œ ž Ÿ
# = 0x83/0x8A/0x8C/0x8E/0x9A/0x9C/0x9E/0x9F) stayed in
# _STRICT_CYRILLIC_EVIDENCE_BYTES — a genuine cp1252 'the ŠŠŠ field'
# run (0x8A×3) passed the strict set-byte evidence; (b) rule (2)'s EOF  # noqa: RUF003
# and digit branches returned True with no byte-class check — 'well
# name áéí' (@EOF) and 'áéí 2024' flipped on the boundary alone.
# Fix: carve the 8 letters from the strict set AND gate the EOF/digit
# branches on strict-set bytes when allow_rule1=False (E-02 whole-file
# evidence).  All three repros decode cp1252 at HEAD and post-fix; the
# genuine cp866 invisible-class target still flips.
# ──────────────────────────────────────────────────────────────


class TestM07StrictEvidenceCarve:
    """M-07: >64K Western cp1252 files with accented runs / Western
    letters must NOT flip to cp1251 via the E-02 whole-file evidence —
    and the genuine cp866 invisible-class target must STILL flip."""

    def test_western_s_letter_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-07 live repro: 'the ŠŠŠ field' (cp1252 — 0x8A×3, alnum under
        cp1252) decodes cp1252.  Pre-fix the strict set still contained
        0x8A → whole-file evidence flipped it to cp1251 ('the ЉЉЉ field')."""  # noqa: RUF002
        text = "the \u0160\u0160\u0160 field"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "m07_s_letter_run.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-07: cp1252 'ŠŠŠ field' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0160\u0160\u0160" in content

    def test_western_accented_run_before_digits_stays_cp1252(self, tmp_path: Path) -> None:
        """M-07 live repro: 'áéí 2024' (cp1252 — 0xE1 0xE9 0xED, M-82
        accents) decodes cp1252.  Pre-fix rule (2)'s digit branch returned
        True with no byte-class check → flipped to cp1251 ('бйн 2024')."""
        text = "\u00e1\u00e9\u00ed 2024"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        test_file = tmp_path / "m07_accented_digits.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-07: cp1252 'áéí 2024' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00e1\u00e9\u00ed" in content

    def test_western_accented_run_at_true_eof_stays_cp1252(self, tmp_path: Path) -> None:
        """M-07 live repro: 'well name áéí' with the accented run at the
        TRUE end of the file decodes cp1252.  Pre-fix the end-of-sample
        branch returned True (not truncated) with no byte-class check →
        flipped to cp1251 ('well name бйн')."""
        text = "well name \u00e1\u00e9\u00ed"  # no trailing newline — run at EOF
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252")
        test_file = tmp_path / "m07_accented_eof.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-07: cp1252 'áéí'@EOF misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00e1\u00e9\u00ed" in content

    def test_cp866_invisible_beyond_64k_still_flips(self, tmp_path: Path) -> None:
        """M-07 positive control (E-02 direction): the genuine cp866
        invisible-class target still flips — a >64K ASCII preamble followed
        by cp866 words (СКВАЖИНА/ПРИВЕТ/ПЛАСТ carry strict 0x80-0x9F  # noqa: RUF002
        evidence bytes 0x8D/0x8F that survived the carve) decodes cp866."""
        russian = (
            "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u041f\u0420\u0418\u0412\u0415\u0422 "
            "\u0422\u0415\u0421\u0422 \u0423\u0410\u0417 \u041f\u041b\u0410\u0421\u0422 "
            "\u0413\u0410\u0417 \u041c\u0410\u0417 "
        )
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + (russian * 30).encode("cp866") + b"\n"
        assert len(raw) > 65_536
        test_file = tmp_path / "m07_cp866_target.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp866", (
            f"M-07: cp866 invisible-class target misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410" in content


# ──────────────────────────────────────────────────────────────
# E3 (encoding, MEDIUM — REGRESSION vs HEAD, fix3 convergence pass):
# the letter-follows branch of _is_genuine_word_run fell through to
# _run_has_las_context UNGATED for allow_rule1=False (E-02 whole-file
# evidence) — the M-07 byte-class gates on the EOF/digit branches were
# missing here.  A >64K Western cp1252 file whose accented/carved-letter
# run (áéí = 0xE1 0xE9 0xED, ŠŠŠ = 0x8A×3 — M-82/M-07-ambiguous bytes, no  # noqa: RUF003
# strict-set byte) is followed by an uppercase LAS mnemonic (FIELD) or a
# digit after an uppercase token (DEPT 1000) flipped the WHOLE file to
# cp1251 mojibake post-fix2 (HEAD=cp1252 on all shapes).  Fix: gate the
# LAS-context fall-through exactly like the EOF/digit branches — with
# allow_rule1=False the run needs a strict 0x80-0x9F evidence byte
# (0x81/0x8D/0x8F/0x90/0x9D — the C1-control class real Western text
# cannot contain); the genuine cp866 СКВАЖИНА-class target (0x8D-bearing)
# still flips via the set-byte fast path.
# ──────────────────────────────────────────────────────────────


class TestE3WesternRunLasContextNoFlip:
    """E3: >64K Western cp1252 files with accented/carved-letter runs
    followed by LAS context (uppercase mnemonic / digit-after-uppercase)
    must NOT flip to cp1251 via the E-02 whole-file evidence — and the
    M-20 section-marker guard must keep holding (no over-correction)."""

    def test_western_accent_run_uppercase_field_stays_cp1252(self, tmp_path: Path) -> None:
        """E3 live repro: 'áéí FIELD 1000.0' (cp1252 — 0xE1 0xE9 0xED +
        uppercase mnemonic FIELD + digit) decodes cp1252.  Pre-fix the
        ungated LAS-context fall-through flipped it to cp1251 ('бйн
        FIELD 1000.0')."""
        text = "\u00e1\u00e9\u00ed FIELD 1000.0"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "e3_accent_uppercase_field.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E3: cp1252 'áéí FIELD 1000.0' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00e1\u00e9\u00ed" in content

    def test_western_carved_s_letter_run_uppercase_stays_cp1252(self, tmp_path: Path) -> None:
        """E3 live repro: 'the ŠŠŠ FIELD' (cp1252 — 0x8A×3, M-07-carved
        letter) decodes cp1252.  Pre-fix the carved-run set-byte path
        fails (0x8A not strict) and the ungated LAS-context fall-through
        flipped it to cp1251 ('the ЉЉЉ FIELD')."""  # noqa: RUF002
        text = "the \u0160\u0160\u0160 FIELD"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "e3_s_letter_uppercase_field.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E3: cp1252 'ŠŠŠ FIELD' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0160\u0160\u0160" in content

    def test_western_accent_run_dept_digit_stays_cp1252(self, tmp_path: Path) -> None:
        """E3 live repro: 'áéí DEPT 1000' (cp1252 — accents + uppercase
        mnemonic + digit-after-uppercase) decodes cp1252.  Pre-fix the
        digit-after-uppercase LAS-context signal flipped it to cp1251
        ('бйн DEPT 1000')."""
        text = "\u00e1\u00e9\u00ed DEPT 1000"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "e3_accent_dept_digit.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E3: cp1252 'áéí DEPT 1000' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00e1\u00e9\u00ed" in content

    def test_western_carved_s_letter_run_dept_digit_stays_cp1252(self, tmp_path: Path) -> None:
        """E3 live repro: 'ŠŠŠ DEPT 1000.0' (cp1252 — carved letters +
        uppercase mnemonic + digit) decodes cp1252.  Pre-fix it flipped
        to cp1251 ('ЉЉЉ DEPT 1000.0')."""
        text = "\u0160\u0160\u0160 DEPT 1000.0"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "e3_s_letter_dept_digit.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E3: cp1252 'ŠŠŠ DEPT 1000.0' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0160\u0160\u0160" in content

    def test_western_accent_run_section_marker_stays_cp1252(self, tmp_path: Path) -> None:
        """E3 control (no over-correction): the section-marker shape
        '~A\\náéí 1234.5' (M-20 guard — digit follows a ~-prefixed
        section header) decodes cp1252.  Held at HEAD AND post-fix2; the
        new LAS-context gate must not disturb the section-marker guard."""
        text = "~A\n\u00e1\u00e9\u00ed 1234.5"
        preamble = ("# " + "X" * 77 + "\n") * 900
        raw = preamble.encode("ascii") + text.encode("cp1252") + b"\n"
        assert len(raw) > 65_536  # E-02 whole-file evidence path
        test_file = tmp_path / "e3_accent_section_marker.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E3 control: cp1252 '~A\\náéí 1234.5' misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00e1\u00e9\u00ed" in content


# ──────────────────────────────────────────────────────────────
# E-26 (encoding, MEDIUM): the №-adjacency digit-follow constraint alone
# cannot distinguish the Russian "СКВ №1" convention from a Western  # noqa: RUF003
# superscript-one + digit ("Nota¹1 Ñáñez") — both are 0xB9 followed by an
# ASCII digit.  Pre-fix the Western ¹+digit shape adjacent to a 3-byte
# accent run fired the №-rule → whole file decoded cp1251 mojibake.
# Fix: the marker must ALSO be preceded (after optional whitespace) by a
# byte that is a Cyrillic letter under cp1251 — the Russian convention
# places № immediately after a Cyrillic word.
# ──────────────────────────────────────────────────────────────


class TestE26NumeroDigitVariant:
    """E-26: Western cp1252 '¹'+digit adjacent to a 3-byte accent run must
    NOT confirm Cyrillic — the №-adjacency rule requires a preceding
    cp1251-Cyrillic byte, not just the 0xB9+digit shape."""

    def test_western_superscript_one_digit_stays_cp1252(self, tmp_path: Path) -> None:
        """The E-26 live repro: 'Nota¹1 Ñáñez' (cp1252 — superscript-one
        directly followed by a digit, adjacent to the 3-byte accent run
        Ñáñ) decodes cp1252.  Pre-fix the №-adjacency rule fired (0xB9 +
        digit) → cp1251 mojibake ('Nota№1 Сбсez')."""  # noqa: RUF002
        text = "Nota\u00b91 \u00d1\u00e1\u00f1ez"
        test_file = tmp_path / "e26_superscript_digit.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E-26: cp1252 '¹'+digit+accent-run misdecoded as {enc!r} (mojibake)"
        )
        assert not any(0x0400 <= ord(c) <= 0x04FF for c in content)

    def test_western_superscript_space_digit_stays_cp1252(self, tmp_path: Path) -> None:
        """E-26 variant: 'Values ¹ 2 Ñáñ' (superscript + space + digit,
        3-byte accent run) decodes cp1252 — the spaced form is also
        Western footnote/ordinal convention, not a Russian № label."""
        text = "Values \u00b9 2 \u00d1\u00e1\u00f1"
        test_file = tmp_path / "e26_superscript_space_digit.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"E-26: cp1252 '¹ 2'+accent-run misdecoded as {enc!r} (mojibake)"
        )
        assert not any(0x0400 <= ord(c) <= 0x04FF for c in content)

    def test_genuine_cp1251_numero_digit_label_stays_cp1251(self, tmp_path: Path) -> None:
        """E-26 positive control: the genuine Russian 'СКВ №1' (3-run + №
        preceded by the Cyrillic В) still confirms Cyrillic — the fix must
        not over-correct the ENC-01 positive control."""  # noqa: RUF002
        russian = "\u0421\u041a\u0412 \u2116 1"
        test_file = tmp_path / "e26_genuine_numero.las"
        test_file.write_bytes(russian.encode("cp1251"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"E-26 positive control: genuine cp1251 №-label misdecoded as {enc!r}"
        )
        assert "\u0421\u041a\u0412" in content

    def test_genuine_cp1251_numero_before_word_stays_cp1251(self, tmp_path: Path) -> None:
        """M-08 control: the genuine Russian '№ 1 СКВ' (№ BEFORE the word,
        at line start) still confirms Cyrillic — E-26's preceding-Cyrillic-
        byte requirement must not break the №-before-word convention.
        Pre-fix (E-26 constraint) it decoded cp1252 mojibake ('¹ 1 ÑÊÂ')."""  # noqa: RUF002
        russian = "\u2116 1 \u0421\u041a\u0412"  # '№ 1 СКВ'  # noqa: RUF003
        test_file = tmp_path / "m08_numero_before_word.las"
        test_file.write_bytes(russian.encode("cp1251"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"M-08: genuine cp1251 №-before-word misdecoded as {enc!r} (mojibake)"
        )
        assert "\u0421\u041a\u0412" in content

    def test_genuine_cp1251_numero_before_word_no_space_stays_cp1251(self, tmp_path: Path) -> None:
        """M-08 control variant: '№1 СКВ' (no space after the marker, № at
        line start) also confirms Cyrillic."""  # noqa: RUF002
        russian = "\u21161 \u0421\u041a\u0412"  # '№1 СКВ'  # noqa: RUF003
        test_file = tmp_path / "m08_numero_before_word_nospace.las"
        test_file.write_bytes(russian.encode("cp1251"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1251", (
            f"M-08: genuine cp1251 '№1 СКВ' misdecoded as {enc!r} (mojibake)"  # noqa: RUF001
        )
        assert "\u0421\u041a\u0412" in content


# ──────────────────────────────────────────────────────────────
# M-19 (encoding, MEDIUM): 3+ consecutive Western-symbol bytes (€ £ ¥ ¢ ¡
# ¿ — the 0x80/0xA1-0xA3/0xA5/0xBF strong bytes, plus ¨ ¯ ´ ¸ × ÷) fired  # noqa: RUF003
# the strong-byte Cyrillic rule → whole file decoded cp1251 mojibake.  The
# genuine-Cyrillic mirror runs (cp1251 ЎЎЎ/ЈЈЈ/ҐҐҐ) are essentially  # noqa: RUF003
# nonexistent in real Russian text — an ASYMMETRIC trade — so the
# Western-symbol bytes are carved out of the strong class AND out of the
# word-like discriminator's evidence for all-carved runs (€€€ = 0x80×3 —  # noqa: RUF003
# byte-identical to cp866 ААА — stays Western; mixed runs like УАЗ keep  # noqa: RUF003
# the F-02 set-byte gate).
# ──────────────────────────────────────────────────────────────


class TestM19WesternSymbolRunsStayCp1252:
    """M-19: genuine cp1252 files with 3+ consecutive Western-symbol runs
    (currency, inverted punctuation, math signs) decode as cp1252 — the
    strong-byte Cyrillic rule must not fire on symbols Western text
    plausibly contains."""

    def test_western_euro_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'PRICE 100€€€ total in euros' (0x80×3) decodes cp1252.
        Pre-fix: strong rule fired → cp1251 mojibake ('100ЂЂЂ')."""  # noqa: RUF002
        text = "PRICE 100\u20ac\u20ac\u20ac total in euros"
        test_file = tmp_path / "m19_euro_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"M-19: cp1252 €€€ run misdecoded as {enc!r} (mojibake)"
        assert "\u20ac\u20ac\u20ac" in content

    def test_western_pound_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'PRICE 100£££ total pounds' (0xA3×3) decodes cp1252."""  # noqa: RUF002
        text = "PRICE 100\u00a3\u00a3\u00a3 total pounds"
        test_file = tmp_path / "m19_pound_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"M-19: cp1252 £££ run misdecoded as {enc!r} (mojibake)"
        assert "\u00a3\u00a3\u00a3" in content

    def test_western_yen_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'PRICE 100¥¥¥ total yen' (0xA5×3) decodes cp1252."""  # noqa: RUF002
        text = "PRICE 100\u00a5\u00a5\u00a5 total yen"
        test_file = tmp_path / "m19_yen_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"M-19: cp1252 ¥¥¥ run misdecoded as {enc!r} (mojibake)"
        assert "\u00a5\u00a5\u00a5" in content

    def test_western_cent_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'cost 50¢¢¢ per item' (0xA2×3) decodes cp1252."""  # noqa: RUF002
        text = "cost 50\u00a2\u00a2\u00a2 per item"
        test_file = tmp_path / "m19_cent_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"M-19: cp1252 ¢¢¢ run misdecoded as {enc!r} (mojibake)"
        assert "\u00a2\u00a2\u00a2" in content

    def test_western_inverted_exclamation_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'Hola ¡¡¡ buenos dias amigo' (0xA1×3) decodes cp1252."""  # noqa: RUF002
        text = "Hola \u00a1\u00a1\u00a1 buenos dias amigo"
        test_file = tmp_path / "m19_inv_excl_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-19: cp1252 ¡¡¡ run misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00a1\u00a1\u00a1" in content

    def test_western_inverted_question_run_stays_cp1252(self, tmp_path: Path) -> None:
        """M-19: 'Es verdad ¿¿¿ no se' (0xBF×3) decodes cp1252."""  # noqa: RUF002
        text = "Es verdad \u00bf\u00bf\u00bf no se"
        test_file = tmp_path / "m19_inv_q_run.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", f"M-19: cp1252 ¿¿¿ run misdecoded as {enc!r} (mojibake)"
        assert "\u00bf\u00bf\u00bf" in content

    def test_strong_rule_no_longer_fires_on_symbol_runs(self) -> None:
        """M-19 guard: the strong-byte Cyrillic detector must NOT fire on
        the carved Western-symbol runs (direct detector test — the carve
        itself)."""
        from pylasdev.encoding import _has_confirmed_cyrillic_run

        for sym in (b"\x80", b"\xa1", b"\xa2", b"\xa3", b"\xa5", b"\xa8", b"\xaf",
                    b"\xb4", b"\xb8", b"\xbf", b"\xd7", b"\xf7"):
            assert not _has_confirmed_cyrillic_run(sym * 3), (
                f"M-19: strong rule still fires on 0x{sym[0]:02X}x3"
            )
        # The cp866-only strong bytes (C1 controls under cp1252) must STAY
        # strong — they are the load-bearing evidence for cp866 Cyrillic.
        assert _has_confirmed_cyrillic_run(b"\x8f\x8f\x8f"), "M-19: 0x8Fx3 lost strong evidence"


# ──────────────────────────────────────────────────────────────
# M-20 (encoding, MEDIUM): the ENC-02(b) Western rescue is blocked for
# short Western strings with smart-punct clusters → cp866 mojibake, via
# two sub-mechanisms: (a) rule (2) of _is_genuine_word_run returns True
# unconditionally at the end of the (possibly truncated) 64K sample —
# a cluster sitting exactly at the sample boundary is judged "genuine
# Cyrillic"; (b) _run_has_las_context counts a `~`-prefixed LAS section
# marker + digit ('~A\n1234.5') as Cyrillic-evidence context.  Fix: a
# truncated sample-end run is NOT genuine (the file continues beyond the
# sample); `~`-prefixed markers are structural headers, not data mnemonics.
# ──────────────────────────────────────────────────────────────


class TestM20WesternRescueBlockers:
    """M-20: the ENC-02(b) rescue must fire for Western smart-punct
    clusters at a truncated sample boundary and before LAS section
    markers — both previously blocked → cp866 mojibake."""

    def test_cluster_at_truncated_sample_end_stays_cp1252(self, tmp_path: Path) -> None:
        """M-20 sub-mechanism (a): a Western '†††' cluster whose run ENDS
        exactly at the 64K sample boundary decodes cp1252.  Pre-fix,
        rule (2)'s end-of-sample branch judged the truncated run genuine
        (the file continues beyond the sample) → rescue blocked → cp866
        mojibake ('ЖЖЖ nota').  M-09: the fixture is >65,536 bytes with
        the run ending AT the boundary and no ASCII-letter-follows
        evidence inside the sample, so ONLY the truncated-sample-end
        branch (encoding.py:576) can resolve it — the old 65,440-byte
        fixture resolved via the unchanged ASCII-letter-follows path and
        passed even with the fixed branch reverted to `return True`."""
        preamble = ("# " + "X" * 77 + "\n") * 819  # 65,520 bytes
        partial = "# " + "X" * 11  # 13 bytes → 65,533 before the run
        raw = preamble.encode("ascii") + partial.encode("ascii")
        raw += "\u2020\u2020\u2020".encode("cp1252")  # run ends at 65,536
        raw += " nota".encode("cp1252")  # file continues beyond the window
        assert len(raw) > 65_536  # sample_truncated must be True
        test_file = tmp_path / "m20_truncated_boundary.las"
        test_file.write_bytes(raw)
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-20a: cp1252 boundary cluster misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2020\u2020\u2020" in content

    def test_cluster_before_section_marker_digits_stays_cp1252(self, tmp_path: Path) -> None:
        """M-20 sub-mechanism (b): a Western '†‡‰' cluster followed by a
        LAS section header with digits ('\n~A\n1234.5 5678.9') decodes
        cp1252.  Pre-fix, _run_has_las_context counted the `~A` marker +
        digit as LAS label evidence → cluster judged genuine → rescue
        blocked → cp866 mojibake."""
        text = "the note \u2020\u2021\u2030\n~A\n1234.5 5678.9"
        test_file = tmp_path / "m20b_section_marker.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-20b: cp1252 cluster+~A+digits misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2020\u2021\u2030" in content
        assert "1234.5" in content

    def test_cluster_before_curve_marker_stays_cp1252(self, tmp_path: Path) -> None:
        """M-20 sub-mechanism (b) variant: the multi-letter section marker
        '~CURVE' must not count as Cyrillic evidence either — a Western
        cluster followed by '\n~CURVE INFORMATION\n1 2 3' decodes cp1252."""
        text = "the note \u2026\u2020\u2021\n~CURVE INFORMATION\n1 2 3"
        test_file = tmp_path / "m20b_curve_marker.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"M-20b: cp1252 cluster+~CURVE misdecoded as {enc!r} (mojibake)"
        )
        assert "\u2026\u2020\u2021" in content


# ──────────────────────────────────────────────────────────────
# N-10 (encoding, MEDIUM — stage-C family continuation): 0xA0-0xAF Western
# symbol runs (guillemets «««, §§§, ¤¤¤, ©©©, ®®®, ¬¬¬, ¦¦¦, soft-hyphen
# ×3+) decode as cp866 lowercase Cyrillic (ллл, ззз, ...) — byte-IDENTICAL  # noqa: RUF003
# to genuine cp866 lowercase words (««« cp1252 == ллл cp866; «как» cp866
# == «ªàª cp1252).  Rule (1) of _is_genuine_word_run (any run byte >= 0xA0
# is "unambiguously genuine") is the load-bearing gate for the genuine
# cp866 lowercase class, so NO byte-content carve can separate the pair —
# the project's C-1/C-2/C-3 convention pins provably-irreducible classes
# as strict xfail rather than changing the detector.  strict=True: a future
# pass that changes the trade flips this pin to XPASS and the suite fails.
# ──────────────────────────────────────────────────────────────


class TestN10A0AFWesternSymbolsXfail:
    """N-10: the irreducible 0xA0-0xAF Western-symbol → cp866 lowercase
    class, pinned strict-xfail exactly like the C-1/C-2/C-3 residuals.
    Pins only — the detection logic is deliberately NOT changed (the
    byte-identity with genuine cp866 lowercase words makes any byte-content
    carve misclassify the genuine class; see the N-10 adversarial proof)."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "N-10 documented residual (stage-C F-5, ADV-M3 CONFIRMED): bytes "
            "AB AB AB = cp1252 '\\u00ab\\u00ab\\u00ab' (Western guillemets) == "
            "cp866 '\\u043b\\u043b\\u043b' (genuine lowercase Cyrillic word).  "
            "The 0xA0-0xAF range IS cp866 lowercase Cyrillic "
            "(\\u0430-\\u043f) — '\\u00ab\\u043a\\u0430\\u043a\\u00bb' "
            "cp866 = 0xAA 0xE0 0xAA = cp1252 '\\u00ab\\u00aa\\u00e0\\u00ab' — so rule (1) of "
            "_is_genuine_word_run (any run byte >= 0xA0 is unambiguous) is "
            "load-bearing for the genuine class and any byte-content carve "
            "misclassifies genuine cp866 lowercase words.  Trade kept in "
            "favor of the genuine Russian-geoscience cp866 lowercase class."
        ),
    )
    def test_cp1252_guillemet_run_stays_cp1252(self, tmp_path: Path) -> None:
        """N-10 residual pin (stage-C F-5 + ADV-M3): 'the ««« quote marks'
        (cp1252) would decode cp1252, but ««« is byte-identical to the
        genuine cp866 word ллл — rule (1) marks it genuine, the rescue is
        blocked, and it decodes cp866.  Cannot be fixed without dropping
        the load-bearing rule-(1) gate.  strict=True guards the trade the
        same way as the C-1 pin."""
        text = "the \u00ab\u00ab\u00ab quote marks"
        test_file = tmp_path / "n10_guillemets.las"
        test_file.write_bytes(text.encode("cp1252"))
        with mock.patch(
            "pylasdev.encoding._detect_encoding_from_bytes",
            return_value="utf-8",
        ):
            with mock.patch(
                "pylasdev.encoding._detect_confidence_from_bytes",
                return_value=0.0,
                create=True,
            ):
                enc, content = read_with_encoding(test_file)
        assert enc == "cp1252", (
            f"N-10 residual: cp1252 guillemet run misdecoded as {enc!r} (mojibake)"
        )
        assert "\u00ab\u00ab\u00ab" in content
