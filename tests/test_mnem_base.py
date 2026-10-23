"""Tests for mnemonic base database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pylasdev.mnem_base import MNEM_BASE, build_mnemonic_lookup, resolve_mnemonic

# HARDEN (test-register rows for TestResolveMnemonic / TestCyrillicRsResistivity):
# the file used to build its own copy of the uppercased first-wins lookup,
# so a regression in the PRODUCTION builder never failed these tests.  All
# real-MNEM_BASE assertions now use build_mnemonic_lookup (mnem_base.py:2072),
# the function the parser (parser.py:1191) and models (models.py:6522) use.


class TestMnemBase:
    """Tests for the mnemonic alias database."""

    def test_mnem_base_shape_and_content(self) -> None:
        """MNEM_BASE is a non-empty dict of str -> str.

        MERGE (test-register row): the former test_mnem_base_is_dict,
        test_mnem_base_not_empty and test_mnem_base_values_are_strings each
        asserted one shape property of the same literal dict; they are one
        test now, keeping every assertion.
        """
        assert isinstance(MNEM_BASE, dict)
        assert len(MNEM_BASE) > 100
        for key, value in MNEM_BASE.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value for {key!r} is not a string"

    def test_known_mappings(self) -> None:
        """Test some known mnemonic mappings."""
        assert MNEM_BASE.get("AK") == "DT"
        assert MNEM_BASE.get("AKDT") == "DT"

    def test_mnem_base_used_in_reader(self) -> None:
        """Test that mnem_base can be used with read_las_file.

        HARDEN (test-register DISPUTED-DELETE, resolved KEEP + HARDEN): the
        old assertion ``assert "logs" in data`` cannot fail for a parseable
        file.  The end-to-end contract this test names is that the reader
        applies the base, so the exact normalized curve order and the index
        values are asserted.
        """
        from pylasdev import read_las_file

        test_data = Path(__file__).parent.parent / "test_data"
        # Use a LAS 2.0 file — LAS 3.0 disallows ~Other sections
        las_file = test_data / "sample_2.0.las"
        data = read_las_file(las_file, mnem_base=MNEM_BASE)
        assert data["curves_order"] == ["DEPT", "DT", "GGK", "KP", "SFLU", "SFLA", "ILM", "ILD"]
        np.testing.assert_allclose(data["logs"]["DEPT"][:3], [1670.0, 1669.875, 1669.75])

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
        """Cycle A→B→A returns the current value when the cycle is detected.

        HARDEN (test-register row): ``result in ("A", "B")`` accepted either
        exit, so it could not detect a change in the cycle policy.  The
        documented behaviour (nodes.py: resolve_mnemonic "returns current when
        next is in seen") makes the exact terminal value B.
        """
        mnem_base = {"A": "B", "B": "A"}
        assert resolve_mnemonic(mnem_base, "A") == "B"

    def test_self_loop_cycle_detection(self) -> None:
        """Self-loop: A→A returns A."""
        mnem_base = {"A": "A"}
        result = resolve_mnemonic(mnem_base, "A")
        assert result == "A"

    def test_max_depth_limit(self) -> None:
        """Long chain exceeding max_depth stops at the depth-limited node.

        HARDEN (test-register row): ``result in mnem_base.values() or
        result == "0"`` accepted almost any string.  Chain 0→1→…→15 with
        max_depth=10 walks exactly 10 hops and returns "10".
        """
        chain_len = 15  # > default max_depth of 10
        mnem_base = {str(i): str(i + 1) for i in range(chain_len)}
        assert resolve_mnemonic(mnem_base, "0", max_depth=10) == "10"

    def test_resolve_with_real_mnem_base_single_hop(self) -> None:
        """Use the real MNEM_BASE lookup to verify single-hop resolution.

        HARDEN (test-register row): the oracle is now the PRODUCTION
        build_mnemonic_lookup rather than a test-local reimplementation
        (a regression in the builder could not fail the old test).
        """
        lookup = build_mnemonic_lookup(MNEM_BASE)
        assert lookup["AK"] == "DT"

    def test_resolve_real_chain_terminates_at_ogz(self) -> None:
        """Real base: GZ3R1 resolves to the terminal canonical name OGZ.

        HARDEN (test-register row): the old name claimed a "three hop" chain
        while the docstring described a single hop (the production builder
        resolves chains eagerly, so no multi-hop chain survives in its
        output — probed: 0 entries of build_mnemonic_lookup resolve to
        another key).  The name and docstring now state what is pinned.
        """
        lookup = build_mnemonic_lookup(MNEM_BASE)
        assert lookup["GZ3R1"] == "OGZ"

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

        HARDEN (test-register row): the lookup comes from the PRODUCTION
        build_mnemonic_lookup, so a regression in the builder fails this pin.
        """
        lookup = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lookup, "AGK") == "GK", "AGK must resolve to GK, not GRO"
        assert resolve_mnemonic(lookup, "Agk") == "GK", "Agk must resolve to GK, not GRO"
        assert resolve_mnemonic(lookup, "aGK") == "GK", "aGK must resolve to GK, not GRO"

    # --- F-H-005: BK collision fix ---

    def test_bk_collision_fix(self) -> None:
        """F-H-005: BK/bk resolve to 'BFV'.

        Before the fix, the first-wins collision between Cyrillic БК→BK and
        the canonical BK→BFV caused lowercase 'bk' lookups to resolve
        incorrectly.  After the fix, both casing variants resolve to BFV.

        HARDEN (test-register row): production build_mnemonic_lookup oracle.
        """
        lookup = build_mnemonic_lookup(MNEM_BASE)
        assert resolve_mnemonic(lookup, "BK") == "BFV", "BK must resolve to BFV"
        assert resolve_mnemonic(lookup, "bk") == "BFV", "bk must resolve to BFV"


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
        """Every Cyrillic R-family key resolves to an R-prefixed canonical name.

        HARDEN (test-register row): the sweep now iterates the PRODUCTION
        lookup (a builder regression fails it) and asserts the sweep is not
        vacuous — a base that lost its Cyrillic entries would otherwise pass
        with zero iterations.
        """
        lookup = build_mnemonic_lookup(MNEM_BASE)
        cyrillic_r_keys = [key for key in lookup if key.startswith("Р")]  # noqa: RUF001
        assert len(cyrillic_r_keys) >= 10, (
            f"expected the Cyrillic R family in the production lookup, got {cyrillic_r_keys}"
        )
        for key in cyrillic_r_keys:
            assert lookup[key].startswith("R"), (
                f"Cyrillic Р-* key {key!r} resolves to {lookup[key]!r}, "  # noqa: RUF001
                "breaking the R-*→R-* pattern"
            )
