# Changelog

All notable changes to pylasdev will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
