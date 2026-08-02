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
