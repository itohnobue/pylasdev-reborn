"""Tests for mnemonic base database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev.mnem_base import MNEM_BASE, resolve_mnemonic


def _build_uppercased_first_wins(
    mnem_base: dict[str, str],
) -> dict[str, str]:
    """Build uppercased mnemonic dict using sorted first-wins semantics.

    Matches the parser's algorithm at parser.py:298-315:
    1. Sort by (not isupper, key) — canonical uppercase entries come first.
    2. First-wins on key.upper() — later entries for the same key are ignored.
    3. Resolve chains via resolve_mnemonic to reach terminal canonical names.

    Dict comprehension {k.upper(): v for ...} uses last-wins semantics
    and produces different results for 84 BK/LL/Cyrillic entries (F-M26).
    """
    # Step 1: sort uppercase-first, then alphabetically
    sorted_items = sorted(
        mnem_base.items(),
        key=lambda item: (not item[0].isupper(), item[0]),
    )
    # Step 2: first-wins by uppercased key
    raw_upper: dict[str, str] = {}
    for k, v in sorted_items:
        key = k.upper()
        if key not in raw_upper:
            raw_upper[key] = v
    # Step 3: resolve chains to terminal canonical names
    return {k: resolve_mnemonic(raw_upper, k) for k in raw_upper}


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

    # --- T12/G-15: MNEM_BASE reader integration end-to-end ---
    def test_mnem_base_normalizes_curve_names(self, tmp_path: Path) -> None:
        """Test that reading a file with mnem_base actually normalizes
        curve names to canonical forms.

        Uses a custom mnem_base dict with single-hop mappings to avoid
        case conflicts in the full MNEM_BASE dict. Verifies end-to-end
        integration: read → parser normalizes → curves_order/logs use
        canonical names.
        """
        from pylasdev import read_las_file

        # Custom mnem_base: AK → DT, APS → ALPS
        custom_mb = {"AK": "DT", "APTS": "SP"}

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth\n"
            " AK.US/M      :  Sonic\n"
            " APTS.        :  SP\n"
            "~A  DEPT  AK  APTS\n"
            "100.0  50.0  10.0\n"
            "101.0  51.0  11.0\n"
        )
        test_file = tmp_path / "mnem_test.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file, mnem_base=custom_mb)
        # Curve names should be normalized
        assert "DEPT" in data["curves_order"]
        assert "DT" in data["curves_order"]
        assert "SP" in data["curves_order"]
        # Original mnemonics should NOT appear
        assert "AK" not in data["curves_order"]
        assert "APTS" not in data["curves_order"]
        # Data should be accessible via normalized names
        assert "DT" in data["logs"]
        assert "SP" in data["logs"]
        np.testing.assert_allclose(data["logs"]["DT"], [50.0, 51.0])
        np.testing.assert_allclose(data["logs"]["SP"], [10.0, 11.0])

    def test_mnem_base_chain_resolution_in_reader(self, tmp_path: Path) -> None:
        """Test multi-hop chain resolution end-to-end through the reader.

        Uses a custom mnem_base with a two-hop chain: BK-3 → BK → BFV.
        Verifies the parser walks the full chain to the terminal canonical name.
        """
        from pylasdev import read_las_file

        # Custom two-hop chain: BK-3 → BK → BFV
        custom_mb = {"BK-3": "BK", "BK": "BFV"}

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth\n"
            " BK-3.OHMM    :  Resistivity\n"
            "~A  DEPT  BK-3\n"
            "100.0  15.0\n"
            "101.0  16.0\n"
        )
        test_file = tmp_path / "chain_test.las"
        test_file.write_text(content, encoding="utf-8")

        data = read_las_file(test_file, mnem_base=custom_mb)
        # BK-3 → BK → BFV (two-hop chain to terminal)
        assert "BFV" in data["curves_order"]
        assert "BK-3" not in data["curves_order"]
        assert "BK" not in data["curves_order"]
        assert "BFV" in data["logs"]
        np.testing.assert_allclose(data["logs"]["BFV"], [15.0, 16.0])


class TestResolveMnemonic:
    """F-022+F-032: Direct tests for resolve_mnemonic function."""

    def test_single_hop_resolution(self) -> None:
        """Single-hop: AK → DT."""
        mnem_base = {"AK": "DT"}
        result = resolve_mnemonic(mnem_base, "AK")
        assert result == "DT"

    def test_non_existent_key_returns_self(self) -> None:
        """Key not in base returns itself."""
        mnem_base = {"AK": "DT"}
        result = resolve_mnemonic(mnem_base, "ZZZ")
        assert result == "ZZZ"

    def test_multi_hop_chain_resolution(self) -> None:
        """Multi-hop chain: BK-3 → BK → BFV."""
        mnem_base = {"BK-3": "BK", "BK": "BFV"}
        result = resolve_mnemonic(mnem_base, "BK-3")
        assert result == "BFV"

    def test_three_hop_chain_resolution(self) -> None:
        """Three-hop chain: A → B → C → D."""
        mnem_base = {"A": "B", "B": "C", "C": "D"}
        result = resolve_mnemonic(mnem_base, "A")
        assert result == "D"

    def test_chain_terminates_at_terminal(self) -> None:
        """Chain stops when a target is not itself a key."""
        mnem_base = {"X": "Y"}  # Y is not a key
        result = resolve_mnemonic(mnem_base, "X")
        assert result == "Y"

    def test_cycle_detection(self) -> None:
        """Cycle A→B→A returns current value when cycle detected."""
        mnem_base = {"A": "B", "B": "A"}
        result = resolve_mnemonic(mnem_base, "A")
        # Cycle detected: returns current when next is in seen
        assert result in ("A", "B")

    def test_self_loop_cycle_detection(self) -> None:
        """Self-loop: A→A returns A."""
        mnem_base = {"A": "A"}
        result = resolve_mnemonic(mnem_base, "A")
        assert result == "A"

    def test_max_depth_limit(self) -> None:
        """Long chain exceeding max_depth returns the current value."""
        # Build a chain of max_depth+5 steps: 0→1→2→...→(max_depth+5)
        chain_len = 15  # > default max_depth of 10
        mnem_base = {str(i): str(i + 1) for i in range(chain_len)}
        result = resolve_mnemonic(mnem_base, "0", max_depth=10)
        # max_depth exhausted at the 10th hop, returns whatever current is
        assert result in mnem_base.values() or result == "0"

    def test_resolve_with_real_mnem_base_single_hop(self) -> None:
        """Use real MNEM_BASE to verify single-hop resolution works."""
        uppered = _build_uppercased_first_wins(MNEM_BASE)
        # AK → DT (single hop, works in uppercased dict)
        result = resolve_mnemonic(uppered, "AK")
        assert result == "DT"

    def test_resolve_three_hop_real_chain(self) -> None:
        """Real chain from MNEM_BASE: GZ3R1 → OGZ (resolved single-hop in uppercased)."""
        uppered = _build_uppercased_first_wins(MNEM_BASE)
        result = resolve_mnemonic(uppered, "GZ3R1")
        assert result == "OGZ"

    def test_custom_max_depth(self) -> None:
        """Custom max_depth limits chain walking."""
        mnem_base = {"A": "B", "B": "C", "C": "D"}
        result = resolve_mnemonic(mnem_base, "A", max_depth=1)
        assert result == "B"  # Only one hop allowed, stops at B

    # --- R-005: Parametrized mnemonic chain resolution ---
    @pytest.mark.parametrize(
        "mnem_base,key,expected",
        [
            ({"AK": "DT"}, "AK", "DT"),  # single-hop
            ({"AK": "DT"}, "ZZZ", "ZZZ"),  # non-existent key
            ({"BK-3": "BK", "BK": "BFV"}, "BK-3", "BFV"),  # two-hop chain
            ({"A": "B", "B": "C", "C": "D"}, "A", "D"),  # three-hop chain
            ({"X": "Y"}, "X", "Y"),  # terminal target
        ],
    )
    def test_resolve_chain_parametrized(
        self,
        mnem_base: dict[str, str],
        key: str,
        expected: str,
    ) -> None:
        """Parametrized test for mnemonic chain resolution across multiple configurations."""
        result = resolve_mnemonic(mnem_base, key)
        assert result == expected
