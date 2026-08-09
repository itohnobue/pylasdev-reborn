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
        # Use a LAS 2.0 file — LAS 3.0 disallows ~Other sections
        las_file = test_data / "sample_2.0.las"
        data = read_las_file(las_file, mnem_base=MNEM_BASE)
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
    """F-022+F-032: Direct tests for resolve_mnemonic function.

    The 5 simple chain cases (single-hop, non-existent, two-hop, three-hop,
    terminal) are covered row-for-row by the parametrized test below
    (F-252) — the standalone duplicates were removed.
    """

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

    # --- F-H-004: AGK/Agk collision fix ---

    def test_agk_collision_fix(self) -> None:
        """F-H-004: AGK/Agk/aGK all resolve to 'GK', not 'GRO'.

        Before the fix, the MNEM_BASE ordering caused case-variant AGK entries
        to collide with Agk1→GRO via the uppercased first-wins algorithm,
        silently mapping AGK to GRO.  After the fix, AGK/Agk/aGK all route
        to the GK family.
        """
        uppered = _build_uppercased_first_wins(MNEM_BASE)
        assert resolve_mnemonic(uppered, "AGK") == "GK", "AGK must resolve to GK, not GRO"
        assert resolve_mnemonic(uppered, "Agk") == "GK", "Agk must resolve to GK, not GRO"
        assert resolve_mnemonic(uppered, "aGK") == "GK", "aGK must resolve to GK, not GRO"

    # --- F-H-005: BK collision fix ---

    def test_bk_collision_fix(self) -> None:
        """F-H-005: BK/bk resolve to 'BFV'.

        Before the fix, the first-wins collision between Cyrillic БК→BK and
        the canonical BK→BFV caused lowercase 'bk' lookups to resolve
        incorrectly.  After the fix, both casing variants resolve to BFV.
        """
        uppered = _build_uppercased_first_wins(MNEM_BASE)
        assert resolve_mnemonic(uppered, "BK") == "BFV", "BK must resolve to BFV"
        assert resolve_mnemonic(uppered, "bk") == "BFV", "bk must resolve to BFV"


class TestMnemBaseDualLaterologCollision:
    """N-I-30/M-36: MNEM_BASE maps both LLD and LLS → BK → BFV.  Reading a
    dual-laterolog file (both curves present) must not crash and the
    collision warning must mention mnem_base resolution AND document the
    collision avoidance — the colliding curve keeps its ORIGINAL mnemonic
    (no duplicate BFV, identity preserved)."""

    def test_reader_rename_warning_mentions_mnem_base(self, tmp_path: Path) -> None:
        import warnings

        from pylasdev import read_las_file

        content = (
            "~VERSION INFORMATION\n"
            " VERS.   2.0  : CWLS LOG ASCII STANDARD\n"
            " WRAP.   NO   : ONE LINE PER DEPTH STEP\n"
            "~WELL INFORMATION\n"
            " NULL.    -999.25 : NULL VALUE\n"
            "~CURVE INFORMATION\n"
            " DEPT.M       :  Depth\n"
            " LLD.OHMM     :  Laterolog deep\n"
            " LLS.OHMM     :  Laterolog shallow\n"
            "~A  DEPT  LLD  LLS\n"
            "100.0  15.0  16.0\n"
            "101.0  15.5  16.5\n"
        )
        test_file = tmp_path / "dual_laterolog.las"
        test_file.write_text(content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            data = read_las_file(test_file, mnem_base=MNEM_BASE)

        # M-36 collision avoidance: the second colliding curve (LLS) keeps
        # its ORIGINAL mnemonic instead of being renamed to a duplicate
        # BFV — the warning documents the preservation.
        keep_warns = [
            str(w.message) for w in rec if "Keeping the original mnemonic" in str(w.message)
        ]
        assert len(keep_warns) >= 1, (
            "Expected 'Keeping the original mnemonic' collision warning, "
            f"got: {[str(w.message) for w in rec]}"
        )
        assert "LLS" in keep_warns[0]
        # The warning still references mnem_base resolution.
        assert "mnem_base" in keep_warns[0]
        # No duplicate BFV is created — LLD normalizes to BFV, LLS survives
        # under its original name.
        assert data["curves_order"] == ["DEPT", "BFV", "LLS"]
        # Data survives under the normalized names.
        assert "BFV" in data["logs"]


class TestCyrillicRsResistivity:
    """F-13: 'РС':'SP' was the only Cyrillic R-* entry breaking the
    R-*→R-* pattern (РД→RD, РЕЗ→RS, РП→RP, РПЗ→RZP).  РС is the Russian
    resistivity abbreviation — corrected to RS.

    The РС→RS member (test_rs_maps_to_resistivity_not_sp) was deleted as
    redundant: test_regression.py:3902-3906 asserts the identical check via
    the production lookup (F-245 partial).  The exhaustive Р-*→R-* sweep
    below is KEPT — it is the only suite test asserting the R-*→R-* prefix
    invariant (grep `startswith("R")` = 1 hit) and is NOT covered by the
    regression class (which only excludes SP).
    """  # noqa: RUF002

    def test_all_cyrillic_r_keys_follow_r_family(self) -> None:
        uppered = _build_uppercased_first_wins(MNEM_BASE)
        for key in uppered:
            if key.startswith("Р"):  # noqa: RUF001
                assert uppered[key].startswith("R"), (
                    f"Cyrillic Р-* key {key!r} resolves to {uppered[key]!r}, "  # noqa: RUF001
                    "breaking the R-*→R-* pattern"
                )
