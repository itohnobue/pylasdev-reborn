# Changelog

All notable changes to pylasdev will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.5] — 2026-08-03

### Production Check

This is the "ultimate-solution" run — a structural refactor that eliminates the four
recurring drift-class problem families that reappeared across the v2.0.1–v2.0.4 production
checks, plus a mechanical enforcement harness and irreducible-class pinning. The regression
suite now passes 1840 tests (1846 collected, 0 failures, 1 skipped DEV-writer placeholder,
5 xfail-pinned documented encoding residuals); ruff and mypy are clean.

### Fixed

- **Drift-class structural refactor**: single shared wrap-detection source of truth — the
  LAS 1.2/2.0 and LAS 3.0 paths now share one gate; flowing-wrapped data (depth + curves on
  one line) now parses correctly via n_curves-accumulation (was silent depth-step loss);
  ragged WRAP=NO shapes no longer corrupt or false-reject; single shared mnemonic-header
  predicate across pre-scan/reader/LAS 3.0; unified sanitize/desanitize helper (`_sanitize.py`)
  with thread-local state preserved
- **Case-normalization**: centralized `.upper()` matching at the container/validation layer
  (single `_case_key`/`_mnem_key` helpers); `{S}` string marker preserved for case-variant
  and emitted-name mnemonics (was silent string-value destruction); `{I}` int64 precision
  preserved beyond 2^53 in all paths; case-variant duplicate curves handled consistently
  (distinct-data refused with accurate error, no silent data loss); `WellSection.get_ci` added
- **Header/parse hygiene**: superset tokenizer unified across header-skip predicate sites
  (fixes phantom all-null first row + data shift in DLM=COMMA files); bare ValueErrors from
  LAS 3.0 spec-form arrays and bracket-array dedup wrapped as LASParseError
- **Writer**: LAS 1.2 well units/descriptions now looked up case-insensitively; linear-time
  wrapped reading (eliminated quadratic list-slicing)
- **Encoding**: three new irreducible classes (Western cluster+digit-direct, cp866 no-set
  word + lowercase follow, cp1251 short word) pinned as strict-xfail documented residuals

### Added

- Mechanical enforcement harness: property-based roundtrip fuzz (deterministic scenarios +
  seeded random), lasio differential tests (lasio>=0.32 as oracle, gated with
  importorskip), LAS 2.0↔3.0 / 1.2↔2.0 cross-version parity tests
- `lasio>=0.32` added to dev optional-dependencies (needed for the differential harness)
- Regression tests locking every fixed drift-class behavior

## [2.0.4] — 2026-08-03

### Production Check

A fourth full production-check audit (34-agent discovery across 11 scopes plus 5
boundary intersections, adversarial verification, implementation, review, and a 6-pass
FIX convergence loop) fixed **35 verified defects** plus **27 post-fix convergence
findings** across the source tree, with additional regression tests. The regression
suite now passes 1655 tests (0 failures, 1 skipped DEV-writer placeholder, 2
xfail-pinned documented encoding residuals) at 87% coverage; ruff and mypy are clean.

### Fixed

- **Parser**: lasio-convention line shapes (no-period, colon-in-unit) now parsed;
  pre-scan/reader header-skip predicates aligned (partial mnemonic headers no longer
  produce phantom rows or data shift); deferred replay and pipe-scoped sections fixed;
  CPU-exhaustion slice-before-check eliminated; sanitize `_#` scope corrected;
  direct-parse duplicate-curve dedup added; guarded-container preservation on EXT-03 dedup
- **Models**: guarded-dict/deepcopy correctness, 0-d array handling, `data_format`
  normalization parity, leaf-field mutation guards, `DevFile.from_dict`
  order-independence, silent data-loss gates, case-insensitive roundtrip parity
- **DEV reader**: DUG Pattern B sentinel/count-match guards, numeric null sentinels,
  EW/NS offset semantics, thousands-separator decimal variants
- **Data reader**: wrap-detection gate corrected on both LAS 2.0/3.0 paths, partial
  mnemonic-header recognition, `_#` desanitize parity
- **Writer**: `~C`/`~A` divergence warning, dedup-key parity, case-insensitive data
  lookup, empty `curves_order` data-loss guard, LAS 3.0 metadata-only curve
  preservation, FIRST-wins curve resolution, VERS normalization
- **Encoding**: codec-derived inflator classification, cp866/cp1252
  Western-vs-Cyrillic rescue corrections, context-based discriminator
- **Compare**: int64 precision beyond 2^53, masked-vs-NaN path consistency
- **Mnemonic base**: GZ2-GZ5 chain correction, Cyrillic РС→RS resistivity mapping
- **LAS 3.0**: spec-form array `{A:N }` tolerance

## [2.0.3] — 2026-08-02

### Production Check

A third full production-check audit fixed **60 verified defects** plus **8 post-fix
convergence findings** across the source tree, with additional regression tests. The
regression suite now passes 1420 tests (0 failures, 1 skipped) at 86.44% coverage;
ruff and mypy are clean.

### Fixed

- **Parser**: deferred/replay and pipe-scoping corrections — phantom rows, discarded
  data, and forward pipes no longer mis-parse
- **Data reader**: wrap-detection contract unified across the LAS 2.0 and 3.0 paths;
  string-object caps and error-boundary corrections applied
- **DEV reader**: thousands-separator recombination, semicolon/locale handling, and
  format detection hardened
- **Models**: guarded-container invariants completed; pickle support and `from_dict`
  symmetry fixed
- **Writer**: duplicate-curve dedup scoping, mutation guards, precision preservation,
  and `~O` roundtrip corrected
- **Encoding**: smart-punctuation and №-adjacency detection fixes
- **ReDoS/security hardening**: regex parsing paths bounded against pathological inputs

## [2.0.2] — 2026-08-02

### Production Check

A second full production-check audit (3 discovery iterations + 2 convergence passes)
fixed **85 verified defects** across the source tree, with ~150 new regression tests. The regression suite
now passes 1231 tests (0 failures, 1 skipped) at 86.07% coverage; ruff and mypy are clean.

### Fixed

- **Parser**: three-segment version normalization (`1.2.0` → `1.2`, `2.0.1` → `2.0`) and VERS re-entry
  guards hardened; section/header edge cases closed
- **Data reader**: wrap-mode and curve-count handling hardened; integer-precision and under-fill paths fixed
- **DEV reader**: format detection (DUG/Petrel/headerless) and MD dedup-survivor validation hardened
- **Models**: mutation-entry validation completed; `from_dict` roundtrip and numpy-scalar acceptance fixed
- **Writer**: duplicate-curve emission scoped per section; mutation guards and precision preservation restored
- **Encoding/comparison**: encoding fallback and compare guards hardened
- **ReDoS/security hardening**: regex/parsing hot paths bounded against pathological inputs

## [2.0.1] — 2026-08-01

### Production Check

A full production check audit (3 discovery iterations, 63 specialized agents) fixed **88 verified defects**
across the source tree, with ~200 new regression tests. Coverage now passes the 85% gate (85.6%).

### Fixed

- **Parser**: deferred pre-~V data sections no longer merge (P-03); well-format swap logic corrected (P-01);
  LAS 3.0 WRAP=YES detection restored (P-02); colon-in-description truncation fixed (P-04);
  `~ASCII|CURVE` pipe headers parse data rows (P-05); exception escapes normalized to `LASParseError` (P-11)
- **Data reader**: wrap-mode detection rewritten to be curve-count-aware and corroborated over ≥3 lines (D-01/D-02/D-03);
  `{I}` integer curves preserve exact precision beyond 2^53 (L-03); wrapped-mode under-fill detection (N-I-08);
  desanitize flag hoisted from the per-value hot path (IT3-F-01); scalar finite checks use `math.*` (IT3-F-02);
  wrapped reads pre-allocate instead of Python-float accumulation (IT3-F-03)
- **Models**: guarded-container wrappers now validate all mutation entry points (M-01/M-02);
  identifier/unit validation whitelisted against the parser grammar (M-03/M-04); numpy scalars accepted (M-06);
  `from_dict` string-data roundtrip fixed (M-28); unnamed data sections auto-named (M-22); direct construction
  deep-copies caller dicts (N-I-11); `_DevColumns` mutation guards completed (N-I-14)
- **Writer**: duplicate curve emission fixed with per-section scoping (W-01/N-I-20); bare precision crash fixed (W-02);
  mutation guards re-established after write (W-06); array `array_index`/`array_info` preserved (W-08/W-09);
  integer precision preserved on write (EXT-04)
- **DEV reader**: DUG/Petrel/headerless format detection hardened (V-01..V-08); all-integer first-row data preserved (V-13);
  MD dedup-survivor validation added (V-17); empty middle header cells rejected (V-18); count-prefix handling corrected (N-I-24)
- **Encoding/comparison**: Cyrillic detection widened and near-tie preference fixed (E-06/E-07); symmetric compare guards (E-08);
  thread-local desanitize flag restored in `finally` (E-04)
- **Null sentinel**: declared NULL now reconciles with baked fill sentinel consistently (N-I-31/IT3-THR-01)

## [2.0.0] — 2026-07-23

### Stable Release

pylasdev 2.0.0 is the first production-stable release. After 35+ rounds of adversarial
security audits, 4 major refactoring phases, and hundreds of verified fixes, the library
is declared **Production/Stable**.

### Added

- **Object-oriented API**: `read_las_file_as_object()` returns typed `LASFile`; `read_dev_file_as_object()` returns `DevFile`
- **Data model dataclasses**: `LASFile`, `DevFile`, `VersionSection`, `WellSection`, `CurveDefinition`, `ParameterEntry`, `ParameterZone`, `ArrayElementInfo`, `DataSection`
- **`_LASVersionSpec`**: centralized version-specific rules (mandatory well fields, delimiter characters, version detection)
- **Writer mutation guard** (`_WriterMutationGuard`): prevents data corruption during write operations
- **`_ParserState`**: extracted parser state management with clean transitions
- **Section transition handler** (`_SectionTransitionHandler`): dedicated section boundary logic
- **Model validation**: `validate()` method on `VersionSection`, `WellSection`, `LASFile` for data integrity checks
- **Combinatorial test matrix**: comprehensive cross-version, cross-mode, cross-encoding test coverage
- **Unified comparison**: `compare_las_dicts()` with type-dispatch pattern and configurable tolerances
- **`LASDataError`**: new exception for data validation failures in `from_dict()`

### Changed

- **Writer refactored** into version-specific modules: `_writer_las12.py`, `_writer_las20.py`, `_writer_las30.py`, `_writer_base.py`
- **LAS 3.0 data handling** extracted into dedicated `_las30_data.py` and `_data_section_reader.py`
- **Version rules centralized** in `_LASVersionSpec` — all version-dependent logic has one authoritative source
- **Classifier upgraded** from `Development Status :: 4 - Beta` to `5 - Production/Stable`
- Version-aware mandatory well fields replace hardcoded LAS 1.2 field set for all versions
- Non-trivial float conversion failures now emit warnings (diagnostic `F-PXR-03`)
- Minimum Python version: 3.12 (was indirectly 3.8+)

### Fixed

#### Data Integrity
- LAS 3.0 per-section curve ordering preserved on roundtrip
- LAS 3.0 custom section type roundtrip preservation (e.g., `~Tops`)
- LAS 1.2 field placement heuristic for files missing mandatory well fields
- WRAP mode auto-detection: files claiming `WRAP=YES` but containing non-wrapped data are handled correctly
- Version string whitespace stripped before version checks (`"  3.0"` no longer misidentified)
- DLM (delimiter) field preserved and validated for LAS 2.0+
- Parameter zone associations survived reassembly

#### Security
- Section injection guard prevents malformed `~` lines from creating phantom sections
- `~Log` alias handling prevents duplicate section registration
- maxsplit guards prevent runaway parsing on pathological data lines
- DoS protections: file size limits, bounded token processing
- BOM stripping for UTF-8 files with byte order marks
- NaN/Inf guards on null values and array bounds

#### Robustness
- Splitlines sanitization prevents header injection via newline-containing field values
- ValueError consistently wrapped as PylasdevError
- Empty-file detection with appropriate warnings
- Encoding fallback chain hardened (cp1251, cp1252, cp866, latin-1)
- DEV format auto-detection for DUG and headerless formats
- Mnemonic collision detection in curve name normalization
- Array element metadata correctly indexed for LAS 3.0
- Format specifier regex handles trailing spaces (e.g., `{A:0 }`)
- Data reader stops at section boundaries (prevents reading garbage after `~A`)
- Depth section reset on re-encounter (prevents double-parsing)

#### Roundtrip Fidelity
- Writer preserves original units/descriptions (not hardcoded `.X`)
- Writer outputs WRAP=NO by default (matches actual non-wrapped output)
- LAS 3.0 string curves stored separately from numeric logs and roundtripped correctly
- LAS 1.2 other-section data roundtripped with correct field order
- `~Other` text preserved during read/write

### Performance
- Wrapped mode: O(n²) → O(n) (fixed `numpy.append()` bug)
- Regex parser: 450+ → 170 lines (replaces PLY-based parser)
- Mnemonic database deduplicated: 5,577 → 2,020 unique entries

### Removed
- PLY-based parser (replaced with regex parser)
- CI/CD infrastructure (`.github/`, workflows) — per project policy
- Community governance docs (`CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`)

---

## [1.6.0] — 2026-06-16

Bug fixes and production hardening:
- NaN/Inf guard for null_value and array bounds checks
- Case-sensitive section detection, strict mypy config, CI pipeline
- Encoding I/O hardening, mnemonic chain normalization, None-to-string conversion
- Parameter normalization and error handling improvements
- Eliminated redundant file read path
- Test coverage increased from 184 to 206

## [1.5.0] — 2026-06-02

Production stability release with 58 adversarially verified fixes:
- Code quality fixes: deduplication, security null guards, dead code removal
- LAS 3.0 data section handling hardened
- Writer output verification improved
- Parser robustness for edge-case LAS files
- Total: +795/-75 lines, 184 passing tests, ruff + mypy clean

## [1.4.0] — 2026-05-28

Second production check with 32 adversarially verified fixes:
- Deep security review with null value guards
- Test coverage expanded (+13 tests)
- Production configuration hardening

## [1.3.0] — 2026-05-20

First production check passes:
- Code quality: CQ deduplication, dead code removal
- Security: null value guards across parser and data reader
- Test coverage expansion (+24 tests)

## [1.2.0] — 2026-05-12

Installation and infrastructure fixes:
- Installation section updated for local clone-and-install workflow
- Package is not published on PyPI (source install only) — clarified in README
- Extras install syntax corrected to PEP 508 direct reference
- All repository URLs fixed to github.com/itohnobue/pylasdev-reborn

## [1.1.0] — 2026-05-01

General cleanup and initialization:
- Module names standardized with underscore convention
- README and LICENSE files added
- Unicode support for curve names and well names
- Initial GitHub repository setup

## [1.0.0] — 2026-02-12

Complete rewrite from Python 2 to Python 3.12+.

### Added
- LAS 3.0 support: array notation, format specifiers ({F}, {E}, {S}, {A:x}), multiple data sections, delimiters
- Type hints on all public APIs
- Object-oriented API: `read_las_file_as_object()` returns typed `LASFile`
- Encoding detection with chardet + fallback chain (cp1251, cp1252, cp866, latin-1)
- Custom exception hierarchy: `LASReadError`, `LASWriteError`, `LASParseError`, `LASVersionError`, `LASEncodingError`, `DEVReadError`
- Comprehensive pytest suite (184 tests)

### Performance
- Wrapped mode: O(n²) → O(n) (fixed `numpy.append()` bug)
- Regex parser: 450+ → 170 lines (replaces PLY)

### Fixed
- Writer preserves original units/descriptions (old writer hardcoded `.X`)
- Writer always outputs WRAP=NO (matches actual non-wrapped output format)
- Parser handles spaces between mnemonic and dot (e.g., `DT  .US/M`)
- Parser supports LAS 3.0 array notation in mnemonic names (e.g., `NMR[1].ms`)
- Auto-detection of mislabeled WRAP headers (files claiming WRAP=YES with non-wrapped data)
- Parser only processes ASCII data inline for LAS 3.0 (prevents double-parsing for 1.2/2.0)
- Roundtrip handles LAS 3.0 string curves stored separately from numeric logs
- Thread-safe parser (no global state)
- Format specifier regex handles trailing spaces (e.g., `{A:0 }`)
- Data reader stops at section boundaries (prevents reading garbage after `~A`)
- Mnemonic database deduplicated: 5,577 → 2,020 unique entries
