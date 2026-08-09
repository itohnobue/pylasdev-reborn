"""Property-based roundtrip fuzz harness.

The mechanical-enforcement harness (Stage B, ultimate-solution run).

Generates random LAS files (varying curve count, row count, data types,
mnemonic case, WRAP mode, delimiter, format specifiers, NULL sentinel),
then asserts that **parse -> write -> reparse** preserves the structural
contract: curves_order, data arrays, well fields, mnemonics, and formats.

This converts the recurring drift-class bugs (wrap detection, header-skip
predicate, case-normalization, sanitize) from a *detection* problem into a
*continuously-running guard*: any future divergence between the reader and
the writer fails here, without a human needing to find the specific edge.

Limitation — roundtrip self-consistency: parse -> write -> reparse applies
the SAME reader on both sides, so a deterministic READER-side bug (wrap
misdetection, case-normalization loss, standalone-header-row misparse) is
reproduced identically on both parses and PASSES — first == second even
with corrupted data.  These scenarios therefore guard the reader<->writer
*contract* (writer regressions and reader/writer asymmetry), NOT reader
correctness in isolation.  The correctness oracle for the wrap/case/header
reader classes is the lasio differential (tests/test_lasio_differential.py
— an independent implementation) plus the reader unit tests in
tests/test_reader.py; see the comment on TestDeterministicScenarios.

Two layers:
  1. Deterministic scenarios (``pytest.mark.parametrize``) — hand-built files
     targeting the known drift classes.
  2. Seeded random batch — a ``random.Random(seed)`` generator producing
     structurally-valid LAS text with varied parameters, asserted under the
     same structural contract.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import pytest

from pylasdev import read_las_file, write_las_file

# ---------------------------------------------------------------------------
# Structural contract helpers
# ---------------------------------------------------------------------------


def _parse(path: Path) -> dict:
    """Parse with warnings suppressed (writer/parser emit convention warnings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_las_file(path)


def _write(path: Path, data: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_las_file(path, data)


def _roundtrip(content: str, tmp_path: Path, name: str) -> tuple[dict, dict]:
    """Parse content -> write -> reparse; return (first, second) dicts."""
    src = tmp_path / f"{name}_src.las"
    out = tmp_path / f"{name}_out.las"
    src.write_text(content, encoding="utf-8")
    first = _parse(src)
    _write(out, first)
    second = _parse(out)
    return first, second


def assert_structural_equal(first: dict, second: dict) -> None:
    """Assert the roundtrip preserved the structural contract.

    The writer is permitted to normalize the version header (WRAP=YES is
    always emitted as NO — a documented writer behavior), so the *version*
    dict is deliberately NOT compared.  Everything that represents the
    logical well content must agree exactly:

    - curves_order (mnemonic identity + order)
    - curve metadata (mnemonic, unit, description, data_format)
    - data arrays (numeric, rtol-matched for float formatting)
    - string_data arrays (LAS 3.0 {S} curves)
    - well fields (exact string dict)
    """
    # curves_order — strict list equality (order is part of the contract)
    assert first["curves_order"] == second["curves_order"], (
        f"curves_order drift: {first['curves_order']} != {second['curves_order']}"
    )
    # curve metadata
    assert first["curves"] == second["curves"], (
        f"curve metadata drift: {first['curves']} != {second['curves']}"
    )
    # well fields
    assert first["well"] == second["well"], f"well field drift: {first['well']} != {second['well']}"
    # numeric data arrays
    for curve in first["curves_order"]:
        if curve in first["logs"]:
            assert curve in second["logs"], f"log {curve!r} missing after roundtrip"
            a = first["logs"][curve]
            b = second["logs"][curve]
            # LAS 3.0 {I} integer curves arrive as object-dtype arrays of
            # Python ints (consistent across the roundtrip); coerce both to
            # float for numeric comparison.
            np.testing.assert_allclose(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                rtol=1e-6,
                err_msg=f"data drift for curve {curve!r}",
            )
    # string data arrays (LAS 3.0 {S})
    first_str = first.get("string_data", {})
    second_str = second.get("string_data", {})
    assert set(first_str) == set(second_str), (
        f"string_data key drift: {sorted(first_str)} != {sorted(second_str)}"
    )
    for key, arr in first_str.items():
        np.testing.assert_array_equal(
            arr,
            second_str[key],
            err_msg=f"string_data drift for {key!r}",
        )


# ---------------------------------------------------------------------------
# Deterministic scenarios targeting known drift classes
# ---------------------------------------------------------------------------

_LAS20_BASE = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO  : One line per depth step
~WELL INFORMATION
 STRT.M   100.0  : START DEPTH
 STOP.M   102.0  : STOP DEPTH
 STEP.M   1.0  : STEP
 NULL.    -999.25  : NULL VALUE
"""

# Wrap detection: depth on its own line, values flowing on following lines
_SCEN_WRAP_DEPTH_OWN_LINE = (
    _LAS20_BASE.replace(" WRAP.   NO", " WRAP.   YES")
    + """~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
 RHOB.K/M3  : BULK DENSITY
~A DEPTH GR RHOB
100.0
10.0 2550.0
101.0
20.0 2551.0
102.0
30.0 2552.0
"""
)

# Wrap detection: values flowing continuously (multiples of curve count per line)
# NOTE (X-6, F-111): this scenario is ROUNDTRIP-BLIND to a wrap-detection
# reader regression — a broken `detect_actual_wrap_from_window` reproduces
# identically on both parses, so first == second even with corrupted data.
# Wrap-detection correctness is delegated to the lasio differential
# (tests/test_lasio_differential.py) and the reader unit tests in
# tests/test_reader.py; this scenario only guards reader<->writer asymmetry.
_SCEN_WRAP_FLOWING = (
    _LAS20_BASE.replace(" WRAP.   NO", " WRAP.   YES")
    + """~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A DEPTH GR
100.0 10.0 101.0 20.0 102.0 30.0
"""
)

# Header-skip predicate: standalone mnemonic header row on its OWN line
# immediately after ~A (a real-world LAS 2.0 variant).  The predicate
# _is_mnemonic_header_row (data_reader.py:721) fires only on the section's
# first line(s), so the mnemonic row here is skipped, not consumed as data.
# The row must contain the DECLARED mnemonics (DEPT, not the description
# DEPTH) for the predicate to match.
# NOTE (X-6, F-111): this scenario is ROUNDTRIP-BLIND to a
# _is_mnemonic_header_row regression — a broken predicate consumes the
# header row as data on BOTH parses identically, so first == second still
# passes with a phantom null row.  The predicate is directly pinned by
# test_reader.py:3771-3822 and test_regression.py:2142; this scenario only
# guards reader<->writer asymmetry.
_SCEN_HEADER_ROW = (
    _LAS20_BASE
    + """~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A
DEPT GR
100.0 10.0
101.0 20.0
102.0 30.0
"""
)

# Case normalization: lowercase and mixed-case mnemonics
_SCEN_CASE_VARIANTS = (
    _LAS20_BASE
    + """~CURVE INFORMATION
 dept.M  : DEPTH
 Gr.API  : GAMMA RAY
 pHit.V/V  : POROSITY
~A DEPTH GR PHIT
100.0 10.0 0.1
101.0 20.0 0.2
102.0 30.0 0.3
"""
)

# Sanitize: # and ~ characters in well values
_SCEN_SANITIZE = (
    _LAS20_BASE
    + """ WELL.   WELL #1 with ~ tilde  : WELL NAME
~CURVE INFORMATION
 DEPT.M  : DEPTH
~A DEPTH
100.0
101.0
"""
)

# NULL sentinel inside data
_SCEN_NULL_IN_DATA = (
    _LAS20_BASE
    + """~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A DEPTH GR
100.0 10.0
101.0 -999.25
102.0 30.0
"""
)

# Tab-delimited LAS 2.0
_SCEN_TAB = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO  : One line per depth step
 DLM .   TAB  : DELIMITING CHARACTER
~WELL INFORMATION
 STRT.M   100.0  : START DEPTH
 STOP.M   102.0  : STOP DEPTH
 STEP.M   1.0  : STEP
 NULL.    -999.25  : NULL VALUE
~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A DEPTH GR
100.0\t10.0
101.0\t20.0
102.0\t30.0
"""

# Comma-delimited LAS 2.0
_SCEN_COMMA20 = """~VERSION INFORMATION
 VERS.   2.0  : CWLS LOG ASCII STANDARD
 WRAP.   NO  : One line per depth step
 DLM .   COMMA  : DELIMITING CHARACTER
~WELL INFORMATION
 STRT.M   100.0  : START DEPTH
 STOP.M   102.0  : STOP DEPTH
 STEP.M   1.0  : STEP
 NULL.    -999.25  : NULL VALUE
~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A DEPTH GR
100.0,10.0
101.0,20.0
102.0,30.0
"""

# LAS 1.2 baseline
_SCEN_LAS12 = """~VERSION INFORMATION
 VERS.   1.2  : CWLS LOG ASCII STANDARD
 WRAP.   NO  : One line per depth step
~WELL INFORMATION
 STRT.M   100.0  : START DEPTH
 STOP.M   102.0  : STOP DEPTH
 STEP.M   1.0  : STEP
 NULL.    -999.25  : NULL VALUE
~CURVE INFORMATION
 DEPT.M  : DEPTH
 GR.API  : GAMMA RAY
~A DEPTH GR
100.0 10.0
101.0 20.0
102.0 30.0
"""

# LAS 3.0 with {S} string curve and comma delimiter
_SCEN_LAS30_STRING = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO  : ONE LINE PER DEPTH STEP
 DLM .   COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS
~Well Information
 STRT .M   100.0  : First Index Value
 STOP .M   102.0  : Last Index Value
 STEP .M   1.0  : STEP
 NULL .    -999.25  : NULL VALUE
~CURVE INFORMATION
 DEPT .M  : DEPTH  {F}
 GR   .API  : GAMMA RAY  {F}
 LITH .  : LITHOLOGY  {S}
~A DEPT GR LITH
100.0,10.0,SAND
101.0,20.0,SHALE
102.0,30.0,LIME
"""

# LAS 3.0 with {E} exponential and {I} integer formats
_SCEN_LAS30_FORMATS = """~VERSION INFORMATION
 VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0
 WRAP.   NO  : ONE LINE PER DEPTH STEP
 DLM .   COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS
~Well Information
 STRT .M   100.0  : First Index Value
 STOP .M   102.0  : Last Index Value
 STEP .M   1.0  : STEP
 NULL .    -999.25  : NULL VALUE
~CURVE INFORMATION
 DEPT .M  : DEPTH  {F}
 YME .PA  : YOUNGS MODULES  {E}
 CNT .    : COUNTS  {I}
~A DEPT YME CNT
100.0,1.45E+12,42
101.0,1.47E+12,43
102.0,2.85E+12,44
"""

_SCENARIOS = [
    pytest.param(_SCEN_WRAP_DEPTH_OWN_LINE, id="wrap-depth-on-own-line"),
    pytest.param(_SCEN_WRAP_FLOWING, id="wrap-flowing-multiples"),
    pytest.param(_SCEN_HEADER_ROW, id="mnemonic-header-row"),
    pytest.param(_SCEN_CASE_VARIANTS, id="case-variant-mnemonics"),
    pytest.param(_SCEN_SANITIZE, id="sanitize-hash-tilde"),
    pytest.param(_SCEN_NULL_IN_DATA, id="null-in-data"),
    pytest.param(_SCEN_TAB, id="tab-delimiter"),
    pytest.param(_SCEN_COMMA20, id="comma-delimiter-las20"),
    pytest.param(_SCEN_LAS12, id="las12-baseline"),
    pytest.param(_SCEN_LAS30_STRING, id="las30-string-curve"),
    pytest.param(_SCEN_LAS30_FORMATS, id="las30-exponential-integer"),
]


class TestDeterministicScenarios:
    """Hand-built files targeting the known drift classes.

    NOTE (self-consistency blindness, X-6): a roundtrip cannot catch a
    deterministic READER bug — the same misparse applies on both parses, so
    first == second passes with corrupted data.  These scenarios catch
    writer regressions and reader/writer asymmetry.  The correctness oracle
    for the wrap/case/header reader classes is the lasio differential
    (tests/test_lasio_differential.py — independent implementation); the
    reader unit tests in tests/test_reader.py are the second guard.
    """

    @pytest.mark.parametrize("content", _SCENARIOS)
    def test_roundtrip_structural_equality(self, content: str, tmp_path: Path) -> None:
        first, second = _roundtrip(content, tmp_path, "scen")
        assert_structural_equal(first, second)


# ---------------------------------------------------------------------------
# Seeded random batch
# ---------------------------------------------------------------------------


def _rand_curve_name(rng: random.Random) -> str:
    """Random mnemonic; sometimes lowercase/mixed-case (case normalization)."""
    base = [
        "DEPT",
        "GR",
        "PHIT",
        "RHOB",
        "DT",
        "NPHI",
        "RES",
        "CALI",
        "SP",
        "ILM",
        "ILD",
        "MSFL",
        "SFLA",
        "SFLU",
        "PEF",
        "DEN",
    ]
    name = rng.choice(base)
    mode = rng.random()
    if mode < 0.25:
        return name.lower()
    if mode < 0.35:
        return name[:1] + name[1:].lower() + name[-1:]
    return name


def _build_las20(rng: random.Random) -> str:
    """Generate a structurally-valid LAS 2.0 text with random parameters."""
    ncurves = rng.randint(2, 8)
    nrows = rng.randint(1, 50)
    wrap = rng.choice(["NO", "YES"])
    dlm = rng.choice(["SPACE", "TAB", "COMMA"])
    nullv = rng.choice(["-999.25", "-9999", "-999.0"])
    start = round(rng.uniform(100.0, 5000.0), 3)
    step = rng.choice([0.1, 0.25, 0.5, 1.0, 2.0])
    # avoid float accumulation error: generate depth as integer index * step
    depths = [start + i * step for i in range(nrows)]

    names: list[str] = []
    while len(names) < ncurves:
        cand = _rand_curve_name(rng)
        if cand.upper() not in [n.upper() for n in names]:
            names.append(cand)

    lines: list[str] = []
    lines.append("~VERSION INFORMATION")
    lines.append(" VERS.   2.0  : CWLS LOG ASCII STANDARD")
    wrap_desc = "Multiple lines per depth step" if wrap == "YES" else "One line per depth step"
    lines.append(f" WRAP.   {wrap}  : {wrap_desc}")
    if dlm != "SPACE":
        lines.append(f" DLM .   {dlm}  : DELIMITING CHARACTER")
    lines.append("~WELL INFORMATION")
    lines.append(f" STRT.M   {depths[0]}  : START DEPTH")
    lines.append(f" STOP.M   {depths[-1]}  : STOP DEPTH")
    lines.append(f" STEP.M   {step}  : STEP")
    lines.append(f" NULL.    {nullv}  : NULL VALUE")
    lines.append(" WELL.   FUZZ WELL  : WELL NAME")

    # units per curve
    units = [rng.choice(["M", "FT", "API", "US/M", "V/V", "OHMM", "K/M3"]) for _ in names]
    lines.append("~CURVE INFORMATION")
    for name, unit in zip(names, units, strict=True):
        lines.append(f" {name}.{unit}  : CURVE DESCRIPTION")

    sep = {"SPACE": " ", "TAB": "\t", "COMMA": ","}[dlm]
    header = " ".join(n.upper() for n in names)
    lines.append(f"~A {header}")

    # generate per-curve data columns
    columns: list[list[float]] = []
    for _i, _name in enumerate(names):
        col: list[float] = []
        for _row in range(nrows):
            if rng.random() < 0.12:
                col.append(float(nullv))
            elif rng.random() < 0.3:
                col.append(float(rng.randint(-50, 5000)))
            else:
                col.append(round(rng.uniform(-10.0, 5000.0), 4))
        columns.append(col)

    for r in range(nrows):
        values = [f"{depths[r]:.6g}"] + [f"{columns[i][r]:.6g}" for i in range(1, ncurves)]
        if wrap == "YES" and rng.random() < 0.5:
            # depth on its own line, remaining values on the next line
            lines.append(f"{depths[r]:.6g}")
            lines.append(sep.join(f"{columns[i][r]:.6g}" for i in range(1, ncurves)))
        else:
            lines.append(sep.join(values))
    return "\n".join(lines) + "\n"


def _build_las30(rng: random.Random) -> str:
    """Generate a structurally-valid LAS 3.0 text with random parameters."""
    ncurves = rng.randint(2, 8)
    nrows = rng.randint(1, 50)
    nullv = rng.choice(["-999.25", "-9999"])
    start = round(rng.uniform(100.0, 5000.0), 3)
    step = rng.choice([0.1, 0.5, 1.0, 2.0])
    depths = [start + i * step for i in range(nrows)]

    names: list[str] = []
    while len(names) < ncurves:
        cand = _rand_curve_name(rng)
        if cand.upper() not in [n.upper() for n in names]:
            names.append(cand)

    formats: list[str] = []
    for i, _n in enumerate(names):
        if i == 0:
            formats.append("F")
        else:
            formats.append(rng.choice(["F", "E", "I", "S"]))

    lines: list[str] = []
    lines.append("~VERSION INFORMATION")
    lines.append(" VERS.   3.0  : CWLS LOG ASCII STANDARD -VERSION 3.0")
    lines.append(" WRAP.   NO  : ONE LINE PER DEPTH STEP")
    lines.append(" DLM .   COMMA : DELIMITING CHARACTER BETWEEN DATA COLUMNS")
    lines.append("~Well Information")
    lines.append(f" STRT .M   {depths[0]}  : First Index Value")
    lines.append(f" STOP .M   {depths[-1]}  : Last Index Value")
    lines.append(f" STEP .M   {step}  : STEP")
    lines.append(f" NULL .    {nullv}  : NULL VALUE")
    lines.append(" WELL .   FUZZ WELL  : WELL")

    lines.append("~CURVE INFORMATION")
    for name, fmt in zip(names, formats, strict=True):
        lines.append(f" {name} .  : CURVE DESCRIPTION  {{{fmt}}}")

    header = " ".join(n.upper() for n in names)
    lines.append(f"~A {header}")

    string_columns: dict[str, list[str]] = {}
    for _i, (name, fmt) in enumerate(zip(names, formats, strict=True)):
        if fmt == "S":
            col: list[str] = []
            for _row in range(nrows):
                col.append(rng.choice(["SAND", "SHALE", "LIME", "DOLOMITE"]))
            string_columns[name] = col

    for r in range(nrows):
        row: list[str] = [f"{depths[r]:.6g}"]
        for i, (name, fmt) in enumerate(zip(names, formats, strict=True)):
            if i == 0:
                continue
            if fmt == "S":
                row.append(string_columns[name][r])
            elif fmt == "E":
                row.append(f"{rng.uniform(1e10, 1e14):.6e}")
            elif fmt == "I":
                row.append(str(rng.randint(0, 5000)))
            else:
                if rng.random() < 0.12:
                    row.append(nullv)
                else:
                    row.append(f"{round(rng.uniform(-10.0, 5000.0), 4):.6g}")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


# Fixed seeds so failures are deterministic and reproducible
_RANDOM_SEEDS = list(range(30))


class TestSeededRandomBatch:
    """Seeded random LAS generation under the structural contract."""

    @pytest.mark.parametrize("seed", _RANDOM_SEEDS)
    def test_las20_random_roundtrip(self, seed: int, tmp_path: Path) -> None:
        rng = random.Random(seed)
        content = _build_las20(rng)
        first, second = _roundtrip(content, tmp_path, f"r20_{seed}")
        assert_structural_equal(first, second)

    @pytest.mark.parametrize("seed", _RANDOM_SEEDS)
    def test_las30_random_roundtrip(self, seed: int, tmp_path: Path) -> None:
        rng = random.Random(seed)
        content = _build_las30(rng)
        first, second = _roundtrip(content, tmp_path, f"r30_{seed}")
        assert_structural_equal(first, second)
