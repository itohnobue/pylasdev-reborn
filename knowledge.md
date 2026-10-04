# Knowledge Base
Last updated: 2026-10-04T20:19:58.625054

## [got-20260630231738-c9b671]
Category: gotcha
Tags: pylasdev, las-spec, writer
Changed: 2026-06-30T23:17:38.034753

pylasdev-reborn LAS 1.2 writer: numeric well fields (STRT, STOP, STEP, NULL) must have value BEFORE colon per CWLS spec. Non-numeric fields use value AFTER colon (lasio convention). Parser auto-detects which convention by trying float(value).

## [pat-20260630231743-c4cd8c]
Category: pattern
Tags: pylasdev, security, dos-protection
Changed: 2026-06-30T23:17:43.531378

pylasdev-reborn DoS protection constants: MAX_DATA_SECTIONS=1000, MAX_CURVES=100000, MAX_PARAMETERS=100000, MAX_DATA_LINES=10M, MAX_TOTAL_ELEMENTS=1B, MAX_TOKENS_PER_LINE=MAX_CURVES. All module-level, overridable. Applied to parser, data_reader, dev_reader.

## [con-20260701005900-8cc26b]
Category: reference
Tags: dev-format, research, pylasdev, specification, las
Changed: 2026-10-04T20:19:18.636195

DEV format reference: (1) no single authoritative DEV standard - LAS 3.0 Inclinometry and
Petrel .dev are closest; (2) two paradigms - MD/INC/AZI and MD/TVD/X/Y; (3) multiple delimiter
conventions; (4) null sentinel = -999.25 in LAS, -999 common; (5) 5 calculation methods but NONE
encoded in the file format; (6) structural comparison table with parsing implications for pylasdev.

## [got-20260717091300-1b2681]
Category: gotcha
Tags: python, models, deserialization, validation
Changed: 2026-07-17T09:13:00.888950

Type validation: checking only the first element of an iterable (isinstance on element[0]) and assuming all subsequent elements match is a false sense of security. In pylasdev models.py: `curves_data` checked `curves_data[0]` with isinstance but looped over all elements without per-element checks (F-012). `_sc_raw` had zero isinstance checks (F-013). And `from_dict` checked length consistency between logs and curves_order but not key-set membership — extra/phantom keys silently created inconsistent state (F-011). Fix: every deserialized iterable element must be independently validated; structural consistency (key-set equality, cross-reference integrity) is as important as type validation.

## [got-20260717091303-1d7974]
Category: gotcha
Tags: python, parser, format-detection, dev-reader
Changed: 2026-07-17T09:13:03.994657

Heuristic format detection is brittle: detecting file formats by inspecting data content (spaces in values, whitespace as delimiter) fails systematically. In pylasdev: CWLS/lasio auto-detection used `' ' in value` — fails for DATE fields where descriptions naturally contain spaces (F-005). DEV reader format detection used whitespace-only `split()` before detecting the actual delimiter — comma-delimited headerless files misdetected, first data row consumed as header (F-019). Fix: use structural markers (known delimiters from file extension/metadata, explicit format parameters) before falling back to data-content heuristics. When heuristics are unavoidable, make them overridable via explicit parameter.

## [pat-20260717164610-581fab]
Category: pattern
Tags: python, input-validation, dict, truthiness, guard, deserialization, pylasdev
Changed: 2026-10-04T20:19:18.636204

Input-validation guard gotchas (truthiness / falsy / wrong-type):
- Truthiness dispatch `value = data.get('key') or default` conflates 'key missing/None' with 'key
  present but wrong type/falsy' - a truthy non-dict `parameters` value produces zero parameters, a
  truthy non-list `curves_data` silently falls into a legacy branch with data loss. Fix: a
  type-validating helper (_resolve_dict_entry) doing `in` + None + isinstance, raising TypeError on
  wrong type; add a mandatory else clause to dispatch.
- `if x:` conflates None (invalid) with '' (valid): for a three-state field (None=invalid,
  ''=not-specified, 'COMMA'=value) use `if x is not None:`, not `if x:`; downstream attribute access
  (.upper()) on None crashes with AttributeError.
- Negative/zero bypass after a None-default: `if x is None: x = DEFAULT` then `if x > limit:` lets
  x=-1000000 through (not None, not above a negative limit). Add `elif x <= 0: raise ValueError`
  immediately after the None-default check, before any downstream guard uses the value.
- Normalize case ONCE at the extraction site (.upper()), not at each comparison site.
In pylasdev-reborn: models.py from_dict had 5 'or'-pattern sites (a function-level regression hotspot);
F-038, F-010.

## [got-20260717164613-322d5f]
Category: gotcha
Tags: python, parser, validation, state-machine
Changed: 2026-07-17T16:46:13.747235

Premature state flag before validation: Setting boolean success flags (`self._version_found = True`) at function entry before any line validation causes silent suppression of error conditions. When a flag is set unconditionally before `_match_data_line` returns, sections containing only garbage/non-matching lines are still marked 'found' — suppressing the 'missing mandatory section' error and producing spurious downstream warnings against incorrect defaults. Fix: move the flag set INSIDE the successful match branch, so it only fires when actual valid data was parsed. In pylasdev-reborn: `_parse_version()` at parser.py:861 set `_version_found = True` before any validation; if all lines in the ~V section failed to match, the flag stayed True, suppressing the real error.

## [got-20260717164628-d88987]
Category: gotcha
Tags: python, numpy, isinstance, NaN, float32
Changed: 2026-07-17T16:46:28.130623

`isinstance(x, float)` returns False for numpy float types (float32, float16, float128) in numpy 1.20+. These types' MRO no longer includes Python's built-in `float`. Code like `if isinstance(a, float) and math.isnan(a):` silently misses NaN values stored as np.float32 — producing incorrect equality comparisons (NaN != NaN returns True, but the guard never fires to handle it). Fix: use `isinstance(a, (float, np.floating))` or just call `math.isnan()` directly (it accepts all numpy float types). This is the 3rd instance of numpy interop gaps in pylasdev-reborn across multiple production check runs — prior fix ae07093 added the check but only for float64.

## [got-20260717164638-f84852]
Category: gotcha
Tags: python, parser, line-parsing, dispatch, fallthrough
Changed: 2026-07-17T16:46:38.810002

Prefix-based parser dispatch fallthrough: In line-oriented parsers using prefix-based dispatch (e.g., `~` for sections, `#` for comments, empty for blanks), unmatched prefixes that bypass ALL handler checks silently fall into the default data handler — producing corrupt data rows. In pylasdev-reborn `_parse_line()`: `~.`, `~#`, `~/` lines didn't match SECTION_PATTERN (requires letter after `~`), COMMENT_PATTERN (requires `#`), or EMPTY_PATTERN (requires blank) — they routed to `_parse_ascii_data` and created phantom data rows. Fix: after all handler checks, add an explicit `if line.strip().startswith('~'):` guard that routes unmatched `~`-prefixed lines to `_other_lines` instead of the data handler. Any prefix that doesn't match a known handler is NOT data.

## [pat-20260717164645-4bf6a0]
Category: pattern
Tags: python, encoding, memory, optimization, candidate
Changed: 2026-07-17T16:46:45.773023

Candidate scoring: store samples, not full content. When selecting among N candidates in a decode/transform pipeline, store only a validation sample (e.g., first 10K chars) for quality scoring; after selecting the winner, re-decode the full content. Storing full content for all candidates multiplies peak memory by N (N x file_size). Fix reduces peak memory to file_size + N x sample_size. In pylasdev-reborn `_decode_best_quality()`: up to 6 encodings each stored full decoded content → ~7x file size peak memory with no benefit (only the winner's content is returned). Fix: store `content[:_MIN_VALIDATION_CHARS]` in candidates list; after selection, `raw_bytes.decode(best_enc).lstrip(BOM)`.

## [got-20260717164648-8dfe29]
Category: gotcha
Tags: python, numerical, allocation, estimation, ceil
Changed: 2026-07-17T16:46:48.895795

Integer division underestimation for capacity estimates: `count // divisor` can undercount by divisor-1 elements (50% error for small divisors), causing resource exhaustion guards to allocate insufficiently or let through excessive input. Use `math.ceil(count / divisor)` for conservative (safe) upper-bound estimates. In pylasdev-reborn `data_reader.py`: `depth_steps = _count // curve_count` produced 50,000x underestimation in wrapped mode, allowing a 500-billion-element allocation to bypass the resource guard. Fix: `math.ceil(_count / curve_count)` + dynamic element counter in the append loop for real-time overflow detection.

## [got-20260718003425-9c2bfd]
Category: gotcha
Tags: python, parser, writer, case-normalization, data-integrity
Changed: 2026-07-18T00:34:25.445450

Case normalization across transform paths: When string values from external input are parsed and compared against internal canonical values, normalize case immediately after extraction. In pylasdev-reborn: parser.py FORMAT_SPEC_PATTERN captured data_format as-is (no .upper()), then 3 comparison sites did case-sensitive checks against uppercase canonical values. Writer.py assumed uppercase section_type (checked via == LOG_DATA) but from_dict input never uppercased it. Fix: apply .upper() once at the extraction site; all comparison sites then naturally match. Verify that ALL code paths touching a parsed string value apply the same case normalization.

## [got-20260718003436-244c67]
Category: gotcha
Tags: python, csv, delimiter, quoting, splitting, data-integrity
Changed: 2026-07-18T00:34:36.836161

Delimiter-split without quoting corrupts data with embedded delimiters: str.split(delimiter) has no concept of quoting or escaping - a value containing the delimiter character silently produces extra tokens. In pylasdev-reborn: DLM=COMMA data with values containing commas produced ghost columns; DLM=TAB with tab-containing string data caused column shift. All 5 str.split(delimiter) call sites across parser.py, data_reader.py, dev_reader.py were affected. Fix: use Python csv.reader with delimiter=char and quoting=csv.QUOTE_NONE for LAS-compatible behavior (LAS spec does not define quoting), then strip each token. The csv module handles edge cases that raw split() cannot: trailing delimiters, empty fields, and proper RFC 4180 semantics. When the format spec does not define quoting, use the csv module for robustness anyway - it is more correct than raw str.split() even in QUOTE_NONE mode because it handles field count consistency.

## [got-20260718003508-6f56a5]
Category: gotcha
Tags: python, numpy, comparison, list, broadcast, ValueError, pylasdev
Changed: 2026-10-04T20:19:58.624978

numpy array comparison in list/dict containers: (a) `v1 != v2` on Python lists containing numpy
arrays raises ValueError (truth value of an array with more than one element is ambiguous) - list
comparison broadcasts to element-wise booleans. Wrap list comparison paths in try/except (ValueError,
TypeError) with an element-by-element ndarray fallback, and replicate the parent's safety pattern in
every delegate/helper: parent compare_las_dicts had a fallback while the list-comparison delegate did
not. (b) np.allclose / np.array_equal / np.array_equiv ALSO raise ValueError for incompatible broadcast
shapes ((3,) vs (2,3)) - pre-validate a.shape == b.shape, or wrap those calls in their own try/except.
Related type-dispatch gotchas (both operands, 0-d arrays, NaN fast-path) are covered by
got-20260718050212-863dce, got-20260721043454-ef9e98, got-20260720104601-5a5b73.
In pylasdev-reborn: compare.py _compare_data_sections / _compare_lists; F-I2-XCM-01, F-01-H, F-43.

## [got-20260718050159-a0e2b3]
Category: gotcha
Tags: python, parser, state-machine, invariant, pylasdev
Changed: 2026-07-18T05:01:59.597258

Parser invariant: _process_ascii_data() → _current_data_section_idx increment. Every call to _process_ascii_data() in the main parser loop MUST be followed by _current_data_section_idx += 1. The 3 existing mid-loop call sites all follow this pattern. Any new call site added during a bug fix or feature must replicate it. Missing the increment causes duplicate DataSection names and MAX_DATA_SECTIONS under-count. See finding F-S9-01 in s9-synth-report.md.

## [got-20260718050201-d378f8]
Category: gotcha
Tags: python, parser, state-machine, invariant, pylasdev
Changed: 2026-07-18T05:02:01.031787

Parser invariant: save/swap/restore all section tracking attributes. When the parser saves/restores state around a deferred operation, ALL section tracking attributes must be included: _section_curve_start_idx, _section_curve_end_idx, _current_data_section_type, and _current_section_name. Missing any attribute leaves stale state from the deferred section bleeding into post-restore context. The consecutive-~A handler (lines 786-810) is the reference implementation — any new save/swap block must enumerate ALL mutable parser state attributes. See finding F-S9-02 in s9-synth-report.md.

## [got-20260718050203-fc830d]
Category: gotcha
Tags: python, parser, state-machine, las30, pre-V, pylasdev
Changed: 2026-07-18T05:02:03.305565

Parser invariant: buffer and replay pre-~V data with section context. LAS 3.0 files can have data sections before the ~VERSION section. (a) Data lines before ~V must be buffered and replayed after version is known. (b) Replay must create SEPARATE DataSections per original section boundary — never merge all deferred lines into one section. Missing (a) silently discards data; missing (b) merges sections with different depth ranges and curve assignments into one corrupted DataSection. Also: deferred well entries (~W before ~V) must be replayed BEFORE any intra-loop _process_ascii_data() to ensure correct null_value. See findings F-M12, F-I2-H01, F-EX-01 in s3/s5/s8-synth-report.md.

## [got-20260718050205-f39e4a]
Category: gotcha
Tags: writer, parser, regex, roundtrip, colon, pylasdev
Changed: 2026-07-18T05:02:05.695861

Writer/Reader contract: colon escaping must handle ALL parser regex alternatives. The parser colon-separator regex has three alternatives: left-whitespace (\\s+:\\s*), right-whitespace (\\s*:\\s+), and trailing (:\\s*$) in DATA_LINE_PATTERN. When the writer escapes colons in values to prevent parser splitting on re-read, ALL alternatives must be handled. Fixing only right-whitespace (': ') while leaving left-whitespace (' : ') unescaped still corrupts on roundtrip. Escape both patterns. Add test with value ' : ' (space-colon-space) to catch left-whitespace escapes. See findings F-H04, F-EX-03 in s3/s8-synth-report.md.

## [got-20260718050207-d85b7e]
Category: gotcha
Tags: parser, data-reader, contract, multi-block, cross-module, pylasdev
Changed: 2026-07-18T05:02:07.992625

Cross-module contract: parser _pre_scan and data-reader MUST agree on multi-block semantics. For files with multiple ~A sections, both modules must use the SAME semantics: either both count/read only the first contiguous block, or both count/read all blocks cumulatively. When they disagree, pre-allocation size mismatches the actual read and the overflow guard silently discards data. Cross-validate with a live integration test using an exact multi-~A file layout, and DOCUMENT the contract explicitly — do not rely on comments that may be factually wrong. Multi-module fixes are fragile: if both sides are fixed independently in opposite directions, the original bug is restored. See findings F-I2-M16, F-EX-02 in s5/s8-synth-report.md.

## [got-20260718050212-863dce]
Category: gotcha
Tags: numerical, numpy, compare, type-dispatch, pylasdev
Changed: 2026-07-18T05:02:12.024344

Numerical safety: comparison dispatch must check BOTH operand types. Type-dispatch branches that check only ONE operand (e.g. isinstance(val2, np.ndarray)) crash when val1 is the ndarray and val2 is a scalar — _scalars_equal(ndarray, scalar) produces a multi-element boolean array, and bool() on it raises ValueError. Every type-dispatch branch must check BOTH operands: isinstance(a, T1) and isinstance(b, T2). The inner list/dict comparison paths in compare.py already use symmetric guards; the scalar fallthrough paths do not. See findings F-I2-M24, F-I2-M25 in s5-synth-report.md.

## [got-20260718050213-8d4156]
Category: gotcha
Tags: parser, sections, duplicate-detection, data-integrity, pylasdev
Changed: 2026-07-18T05:02:13.721046

Section handling: all key-based section types need duplicate detection. Well entries use raw dict assignment (self.well[mnen] = value) — duplicate mnemonic silently overwrites prior value with no warning. Curves have _deduplicate_curves() (60+ lines with renaming logic). Parameters use list.append(). When adding a new section type, follow the STRONGEST convention (dedup + warning), not the weakest (silent overwrite). At minimum: log a warning on duplicate mnemonics in any key-based section. See finding F-I2-M10 in s5-synth-report.md.

## [got-20260718050215-6ccbbc]
Category: gotcha
Tags: data-reader, wrapped-mode, column-alignment, pylasdev
Changed: 2026-07-18T05:02:15.455313

Data-reader wrapped-mode: guard against consecutive single-value lines. Two consecutive single-value lines in wrapped mode cause permanent DEPTH↔C1 column swap. With curve_count=2: line1 (1 value) → data_lists[0], line2 (1 value) → data_lists[1] (C1 treated as depth), line3+ → all columns shifted. Wrapped-mode is a state machine with implicit column expectations. After a single-value line, the NEXT line MUST complete the pair (curve_count - 1 values). Two consecutive short lines = state machine corruption. Add explicit guard. See finding F-I2-M15 in s5-synth-report.md.

## [pat-20260718050217-736991]
Category: pattern
Tags: parser, sections, label-collision, abbreviation, pylasdev
Changed: 2026-07-18T05:02:17.628889

Section label abbreviation collisions: When section labels have abbreviated forms (e.g. ~C for ~CURVE, ~P for ~PARAMETER), duplicate detection must account for label equivalence. Both labels collapse to the same internal key, so valid LAS 3.0 multi-section files with multiple curve sections trigger false-positive 'Duplicate section header' warnings. Solution: use full section names for dedup, or track whether a collision is from the same structural section type. This applies to any parser that normalizes/abbreviates input labels before internal storage. See finding F-I2-M11 in s5-synth-report.md.

## [got-20260718223055-580847]
Category: gotcha
Tags: guard, off-by-one, unit-mismatch, dos, resource-limits, pylasdev
Changed: 2026-10-04T20:19:18.636215

Guard operator and unit consistency across code paths: when the same semantic limit is enforced
in multiple code paths (parser vs from_dict, pre-scan vs read), both the comparison operator (>= vs >)
and the unit of measurement (lines vs characters) must be consistent. >= in one path and > in another
creates non-commutative roundtrip behavior (from_dict accepts what the parser rejects); counting lines
in one path and characters in another against the same constant is a unit error. Also: when a guard
expression multiplies two counters (num_curves * actual_count), a zero counter neutralizes the entire
guard - every multiplier needs independent minimum validation (> 0). In pylasdev-reborn: parser used
>= for 4 MAX guards while from_dict used > for 6 (F-I2-M32); MAX_OTHER_LINES checked len(list[str]) in
parser but len(str) in from_dict (F-I2-M33); actual_count==0 made num_curves*0=0 always pass
MAX_TOTAL_ELEMENTS (F-I2-M09); parser.py _process_ascii_data mixed > and >= in one function (line
2411 vs 2416). Checklist: grep the same semantic limit across all paths; compare operator AND unit
verbatim.

## [got-20260718223103-8ee17f]
Category: gotcha
Tags: sanitization, parser, writer, symmetry, pylasdev
Changed: 2026-07-18T22:31:03.206955

Symmetric reader/writer sanitization: characters stripped, escaped, or sanitized by the writer must also be handled by the reader, and vice versa. Asymmetry creates injection surfaces (characters the writer strips but the reader passes through) and roundtrip corruption (characters the writer escapes that the reader cannot un-escape). Checklist: (a) enumerate every character class stripped by the writer's sanitizer regex, (b) verify the reader's input-splitting regex strips or handles the SAME character class — a single character in one but not the other is an asymmetry, (c) when one side adds escape characters, the other side must remove them (lossy escaping without un-escaping permanently reduces roundtrip fidelity). In pylasdev-reborn: writer's _CONTROL_CHARS_RE strips \x00 and 25 other control chars, but reader's _SPLITLINES_CHARS_RE does not strip \x00 (F-I2-M04, F-I2-M05); writer's _escape_colons_for_las_value adds permanent _ characters around colons with no parser-side un-escaping (R9-019).

## [got-20260718223114-4d4bbf]
Category: gotcha
Tags: regex, parser, format-specifier, data-corruption, pylasdev
Changed: 2026-07-18T22:31:14.014545

Over-broad regex silently corrupts data in format-sensitive parsers: when a regex is designed to match a specific syntax (format specifiers like {F}, {F10.4}) but the pattern is over-broad (matches any {letter...}), non-format brace content gets consumed as format specifiers and stripped from the original text, corrupting descriptions and metadata. Similarly, format validation by first-character check is fragile — {DEG} and {DD/MM/YYYY} both start with D, passing a first-char check for 'decimal format' while containing non-numeric data. Fix: (a) constrain regex to the exact syntax needed (for format specifiers: {[FED]\d*(\.\d+)?} not {[A-Za-z][^}:]*?}), (b) validate format specifiers against a whitelist of known codes, not by first-character heuristics, (c) when regex routing determines downstream processing (numeric vs string curves), the regex character class must exactly match the routing logic — a character in the regex class but not handled by routing = silent data corruption. In pylasdev-reborn: FORMAT_SPEC_PATTERN \{[A-Za-z][^}:]*?...\} matched {well A12} as a format specifier, stripping it from descriptions (F-I2-M08); _KNOWN_CURVE_FORMATS used fmt[0] check, passing {DEG} and {DD/MM/YYYY} as numeric formats (F-M03); _FORMAT_SPEC_RE accepted S10 as numeric-width format while string_curves classified S as string, producing NaN for string values (R9-018).

## [got-20260718223138-e73af5]
Category: gotcha
Tags: normalization, case-sensitivity, reference-data, lookup, pylasdev
Changed: 2026-07-18T22:31:38.744877

Reference data case normalization completeness: when case-insensitive lookup is critical (mnemonic resolution, canonical value matching), ALL entries in the reference database must be normalized to a single case. Even a small fraction of unnormalized entries breaks lookup for that entire subset — quiet failures that are very hard to detect without explicit testing. In pylasdev-reborn: 9 of 1849 canonical mnemonic values (~0.5%) were mixed-case ("Density", "GKst", "NKTst", "Koll") while all other values were uppercase. The parser uppercased lookup KEYS but stored VALUES as-is, so resolution of these 9 entries against uppercase keys silently failed. Checklist: (a) mechanically verify that EVERY value in the reference dataset passes `v == v.upper()` (or the chosen normalization), (b) add a regression test that iterates over all entries and asserts normalization, (c) when a reference dataset is large (1000+ entries), use automated validation — one-off manual fixes will miss entries.

## [pat-20260719022332-c28c46]
Category: pattern
Tags: from-dict, validation, construction, parser, roundtrip, data-model, pylasdev
Changed: 2026-10-04T20:19:18.636218

from_dict / direct-construction / parser parity: from_dict() and direct construction are the
primary bypass vectors around parser guarantees. The parser builds objects through a controlled pipeline
(key types, cross-field validation, dedup, resource limits, normalization); every other construction
path must replicate EVERY validation, post-processing step, and guard.
Checklist: (a) validate ALL dict key types (isinstance(key, str) - parser always produces str keys);
(b) cross-validate data keys against curve mnemonics (numeric AND string_data); (c) deduplicate
curves/sections/entries; (d) catch ALL exception types internal helpers raise (ValueError AND TypeError
at every try, matching sibling classes); (e) apply the SAME resource limits with the SAME operators
(>= vs >, lines vs chars); (f) detect key collisions across parallel dicts (data vs string_data,
metadata vs column names); (g) validate data_sections against the version (parser only produces them for
3.0); (h) cross-validate data_format against actual placement (data_format='F' with string values
silently changes np.float64->np.str_ on roundtrip); (i) apply the SAME preprocessing/normalization
(case folding, mnem_base key normalization) - from_dict that skips parser normalization produces
different internal state for the same logical input.
Direct construction: __post_init__ is the universal gate (runs for ALL paths) - it must apply the SAME
mandatory-field checks as parser/from_dict, or delegate to a shared validation function; silently
accepting empty/invalid values (Empty VERS, missing mandatory well fields) yields an invalid object.
Nested models: the parent must replicate or delegate to child validation (LASFile vs DataSection dtype
checks). Deepcopy mutable caller input on direct-construction paths too, not just from_dict.
When a fix widens what construction/validate accepts, trace the SAME state through to_dict->from_dict,
write->read, AND the public dict-write API - from_dict exact-case checks and DataSection direct
construction must accept it too, or the newly-reachable state fails the public API (F-04).
In pylasdev-reborn: F-004/F-011/F-012/F-013/F-058/F-036/F-067, IF-007/IF-026, F-04 - 9+ CONFIRMED
MEDIUM+ across iterations.

## [pat-20260719022337-4248dc]
Category: pattern
Tags: python, format-spec, input-validation, writer, data-integrity, pylasdev
Changed: 2026-07-19T02:23:37.207020

Validate user-supplied format specifiers before use: When users supply Python format specifiers (e.g., precision strings like '.4f'), validate that the specifier is compatible with the data type being formatted. format(1.5, '.8x') produces hex float notation; format(1.5, '.4%') multiplies by 100 and appends '%' suffix producing unparseable '150.0000%'; format(1.5, '.4n') uses locale-dependent decimal separator and grouping producing e.g. '1,500.0' — all silently corrupt output. Only 'e','E','f','F','g','G' are safe for decimal float formatting. Fix: validate format specifiers against a whitelist at the entry point (writer constructor, write() call), raising ValueError for unsupported codes including % (percentage) and n (locale). This is analogous to SQL injection via unsanitized format strings — the format() function accepts ALL Python format codes regardless of data type. In pylasdev-reborn: _validate_precision regex accepted % and n format codes, causing 100x multiplier corruption and locale-dependent corruption respectively (IF-016, IF-017 in 2026-07-19 run).

## [pat-20260719022344-ba2fd5]
Category: pattern
Tags: python, parser, version, validation, las-format, pylasdev
Changed: 2026-07-19T02:23:44.927421

Validate parsed version/format strings at parse time: When a format branches parsing behavior based on a version string field (e.g., VERS in LAS files), validate the version value IMMEDIATELY after extraction against a known set of supported values. Silent acceptance of non-standard values (e.g., '1,2' instead of '1.2', or '2,0' instead of '2.0') passes through downstream startswith() checks without error, routing the file through the wrong parsing code path and silently corrupting data. In pylasdev-reborn: parser.py _parse_version stored any string value without validation; WRAP and DLM had guards but VERS did not. Non-standard VERS values silently failed startswith('1.2') and startswith('2.0') checks, causing the file to be parsed under the wrong version rules. Fix: maintain a SUPPORTED_VERSIONS set and raise ValueError immediately if the parsed value is not in the set. This applies to any parser that branches on a version discriminator — validate before storing, not at consumption time when context is lost. (F-005/IF-003, DOUBLE CONFIRMED across two iterations in 2026-07-19 run)

## [pat-20260719022349-35310c]
Category: pattern
Tags: python, data-model, roundtrip, structured-format, las30, pylasdev
Changed: 2026-07-19T02:23:49.005777

Preserve section/type provenance in flat data models: When a structured file format supports per-section typed metadata (e.g., LAS 3.0 ~Core_Parameter, ~Inclinometry_Parameter), the data model MUST preserve the section-type association. Merging heterogeneous typed sections into a single flat list permanently loses provenance — on roundtrip, all entries are emitted under a single generic section header, destroying the original per-type grouping. In pylasdev-reborn: parser routed per-section parameters to a flat las_file.parameters list; ParameterEntry had no section_type field; writer emitted a single ~PARAMETER INFORMATION section for all parameters. A file with separate ~Core_Parameter and ~Inclinometry_Parameter sections roundtrips to a single merged parameter section. Fix: add a provenance field (section_type) to each entry so the writer can reconstruct per-section blocks. Applies to any format with typed sections — the type IS part of the data and must survive roundtrip. (F-053/IF-011, DOUBLE CONFIRMED across two iterations in 2026-07-19 run)

## [pat-20260719022353-f93a69]
Category: pattern
Tags: python, writer, sanitization, escape, output, roundtrip, pylasdev
Changed: 2026-10-04T20:19:18.636221

Output/field sanitization completeness: when N of M output fields/call-sites apply the sanitizer,
the one unsanitized field is the injection/corruption vector - a single gap negates all others. In
pylasdev-reborn 19/19 call sites used _sanitize_las_value() except one (`actual_wrap` writer.py:168); a
crafted WRAP value containing '\n~' injected fake section headers. In delimited output, curve
name/unit/description used the colon-escaper but api_code did not - a colon in api_code mis-split the
line on re-read.
Checklist: enumerate ALL output fields rendered to text (grep every .format()/interpolation site in the
writer) and mechanically verify each passes through the same sanitizer/escaper; when adding an escape,
verify the parser reverses EXACTLY the same positions (sanitize/desanitize scope symmetry); test the one
omitted field explicitly.

## [got-20260719022356-224371]
Category: gotcha
Tags: python, dict-key, contract, consistency, data-section, pylasdev
Changed: 2026-07-19T02:23:56.591857

Cross-file dict key contract mismatch — 'data' vs 'logs': In pylasdev-reborn, DataSection objects store curve data under the key 'data' (e.g., section['data']), but LASFile-level validation functions accessed it as 'logs' (e.g., data.get('logs')). This key inconsistency caused the per-section data_format validation (_check_df_vs_placement) to silently read nothing from DataSection dicts, making the validation inert for all per-section data. When two types in the same codebase use different dict keys for the same semantic concept, every cross-type function that accesses both is at risk of key mismatch. Fix: define a shared constant (DATA_KEY = 'data') and use it consistently across all types that store or access curve data. Grep for the literal string across all consumers before finalizing the design. (R-001 in 2026-07-19 post-fix review)

## [pat-20260719060334-b04a13]
Category: pattern
Tags: python, memory, allocation, validation, resource-guards, structural
Changed: 2026-07-19T06:03:34.204076

Validate before allocating: order guard checks before resource calls. Guards placed AFTER np.array()/allocation cannot prevent MemoryError. Checklist: (1) move all size/count/product/finiteness checks before allocation, (2) except clause must catch MemoryError if allocations remain after guards, (3) test combined-excess scenario. In pylasdev-reborn: two from_dict paths allocated np.array() BEFORE per-array and product guards, allowing 79GB/800MB allocations before detection. F-012 + I2F-14, DOUBLE CONFIRMED.

## [got-20260719060336-bdbeb2]
Category: gotcha
Tags: python, numerical, float, int-conversion, inf, nan, isinstance
Changed: 2026-07-19T06:03:36.420946

float('inf') passes isinstance checks but crashes int(): float('inf') and float('nan') pass isinstance(x, (int, float)) but int(x) raises OverflowError/ValueError. Notably float('inf').is_integer() returns True (Python doc quirk) — it is NOT a sufficient guard. Every int(float_val) conversion needs math.isfinite() before conversion. Checklist: (1) grep for int(float_val), (2) add finiteness check before each, (3) test with inf/-inf/nan. In pylasdev-reborn: writer crashed on int(float('inf')) for time_offset; from_dict accepted inf/nan values that crash downstream. F-024 + I2F-13, DOUBLE CONFIRMED.

## [got-20260719060338-113b9f]
Category: gotcha
Tags: python, memory, allocation, cumulative-guard, arithmetic, resource-limits
Changed: 2026-07-19T06:03:38.659666

Disjoint allocation pools need sum, not max, for cumulative guards: When two allocation pools are disjoint (mutually exclusive by construction), the cumulative guard must use += (sum), not max(). max() only tracks the larger pool, letting combined allocations bypass the limit. Overlap → max(); disjoint → sum. Checklist: (1) determine pool semantics, (2) verify arithmetic operator, (3) test combined-bypass scenario. In pylasdev-reborn: max() on disjoint ds_data+ds_string_data allowed 600M+600M=1.2B through a 1B limit. I2F-15 + R7F-02, DOUBLE CONFIRMED.

## [pat-20260719092855-df2103]
Category: pattern
Tags: validation, parser, from-dict, roundtrip, format-specifier, las-spec
Changed: 2026-07-19T09:28:55.333585

Format validation gate symmetry between parser and from_dict: When a parser validates format specifiers at parse time (via a regex like _FORMAT_SPEC_RE accepting F8.3, E10.2, I5, etc.), the from_dict constructor's validation gate (_VALID_DATA_FORMATS) must accept the SAME set. Asymmetric validation causes silent roundtrip failures: parse→to_dict()→from_dict() breaks because the parser accepts what from_dict rejects. Checklist: (a) extract format validation regex/constant to a shared location used by both parser and from_dict, (b) move format validation to parse time — validation gated on data-section presence misses metadata-only files, (c) validate ALL format-bearing fields (curve format, parameter data_format) not just the primary one, (d) when the parser captures format specifiers but defers validation to data-processing time, prevent early-return code paths from skipping validation on files without data sections. In pylasdev-reborn: parser's _FORMAT_SPEC_RE accepted F8.3/I5/E10.2; from_dict's _VALID_DATA_FORMATS=frozenset({'F','E','D','A','S'}) rejected them — broke 6 format variants. Parameter data_format captured by broad regex with zero validation; curve format validation gated on _ascii_data_lines being non-empty, skipped for metadata-only LAS 3.0 files. _VALID_DATA_FORMATS missing 'I' caused separate roundtrip failure. 4 confirmed MEDIUM+ findings (F-088, F-101, F-102, F-207) across 2 production-check iterations.

## [pat-20260719092907-5aba39]
Category: pattern
Tags: python, dataclass, validation, post-init, data-model
Changed: 2026-07-19T09:29:07.576030

Dataclass __post_init__ cross-field consistency validation: When a dataclass is a public API type that can be directly constructed by users, __post_init__ must validate ALL cross-field consistency constraints — not just type checks and required-field presence. Checklist: (a) validate within-group array/sequence lengths (all arrays in self.data should have identical lengths), (b) validate cross-group row counts (numeric data and string_data within the same section should have matching row counts), (c) validate collection uniqueness constraints (no duplicate entries in curves_order, labels, identifiers), (d) document that mutation after __post_init__ (e.g., dict key insertion) bypasses validation — Python dataclasses provide no mutation guard. Direct construction bypasses parser/from_dict validation pipelines; the dataclass itself is the last defense. In pylasdev-reborn: DataSection.__post_init__ validated orphaned keys, disjointness, and section_curves length but missed array-length consistency (F-027/F-079), cross-group row-count parity (F-037), and duplicate curves_order entries (F-105). 4 confirmed MEDIUM+ findings across 2 iterations.

## [pat-20260719092917-951b5b]
Category: pattern
Tags: parser, duplicate-detection, label, section, las-spec
Changed: 2026-07-19T09:29:17.831930

Semantic-type-based duplicate detection beats label-based: In parsers where section headers can have multiple label variants mapping to the same semantic type (e.g., ~V, ~VERSION, ~V INFORMATION all map to VERSION section), duplicate detection must normalize by semantic type, not label string. Label-based dedup produces false negatives — ~V and ~VERSION produce different label keys and are NOT flagged as duplicates, causing the second occurrence to silently overwrite the first. The parser's dispatch code correctly routes all variants to the same handler, confirming the semantic equivalence that duplicate detection should mirror. Checklist: (a) identify all sections with multiple label variants (full name, abbreviation, name-with-whitespace), (b) build a label→type normalization map, (c) track seen types (not labels) in duplicate detection, (d) consider whether multi-occurrence sections (~C/~CURVE in LAS 3.0) should be exempt from dedup — preserve the parser's current behavior for those. In pylasdev-reborn: section duplicate detection used exact string keys ('V:' vs 'VERSION:INFORMATION'); both labels dispatched to _parse_version via new_section='V', but duplicate counting saw different keys → no warning → silent version overwrite. Section-name variation (~V INFORMATION) extended the bypass surface. 2 confirmed MEDIUM+ findings (F-048, F-103) across 2 iterations.

## [pat-20260719092928-18c87b]
Category: pattern
Tags: writer, dispatch, legacy, data-model, bridge, copy-back
Changed: 2026-07-19T09:29:28.943319

Legacy↔new data model bridge: write-path dispatch must copy new-model fields to legacy attributes: When a data model evolves by adding a new representation (data_sections) alongside legacy flat fields (logs, string_data, curves_order), the writer's dispatch logic that reads legacy fields for certain code paths must include copy-back logic from the new model. Checklist: (a) enumerate ALL fields that legacy code paths read (logs, string_data, curves_order, etc.), (b) when new-model construction (from_dict) produces data exclusively in new fields, verify the writer copies them to legacy fields before dispatch reads from the legacy side, (c) when both new and old fields can be populated simultaneously (parser produces both), add mutual-exclusivity enforcement — reject ambiguous input rather than silently preferring one, (d) NEVER emit a warning claiming data preservation if the code immediately below destroys the data. The parser path often masks this gap because the parser copies new→old at construction time; from_dict path is the unprotected entry point. In pylasdev-reborn: writer's non-LAS-3.0 path read las_file.logs (empty for from_dict construction) while data resided in data_sections[0].data — silent data loss with misleading preservation warning (F-067). curves_order also lost (F-111). For LAS 3.0, both logs and data_sections could be populated with zero mutual-exclusivity enforcement — modified logs values silently discarded (F-115). 3 confirmed MEDIUM+ findings across 2 iterations.

## [got-20260719093002-019e72]
Category: gotcha
Tags: python, injection, validation, sanitization, section-header, identifier
Changed: 2026-07-19T09:30:02.596613

Identifier field content validation — without it, injected control characters become structural delimiters: When a field like section_type becomes part of structured output (writer interpolates it into a section header line like ~{section_type}_Parameter, newline and tilde characters in the value produce injected section headers that the parser routes as real sections. Three compounding gaps enable end-to-end injection: (a) input validation checks only type and length, not character content — 'CORE\n~VERSION' passes isinstance(str) + length check, (b) writer sanitization converts \n to space, creating a word boundary where the embedded ~VERSION survives, (c) the parser's leading-tilde regex only strips tilde at string START — embedded tildes after whitespace pass through and match section patterns. Defense-in-depth checklist: (1) input validation layer — reject control characters (\n, \r, \0), tildes, and any character that is meaningful in the target format, (2) sanitization layer — strip embedded format-significant characters (not just leading), (3) verification — test that roundtrip through write→parse does not produce phantom sections. In pylasdev-reborn: section_type = 'CORE\n~VERSION' → writer emits ~CORE ~VERSION_Parameter → parser routes as DATA section → parameter data consumed as numeric data rows → silent corruption. 1 confirmed MEDIUM+ finding (F-118), verified end-to-end by cross-domain adversarial agent.

## [got-20260719093011-67b36b]
Category: gotcha
Tags: python, mypy, type-checking, set, comprehension
Changed: 2026-07-19T09:30:11.272763

mypy strict rejects set.add() return value in boolean context: set.add() returns None, and using it as a boolean guard in comprehensions (e.g., seen.add(x) or True) triggers mypy strict mode error. Common idiom 'seen.add(key) or True' in list comprehensions relies on None being falsy for short-circuit evaluation — mypy strict correctly identifies that set.add()'s None return is being consumed as a boolean. This is typically the project's sole mypy error when strict=true is configured. Fix: replace the comprehension with an explicit loop that checks key not in seen before adding, or use a dedicated helper that returns True after adding. In pylasdev-reborn: models.py used seen.add(curve) or True in a list comprehension dedup loop — the only mypy strict error in the entire project after pyproject.toml configured strict=true. 1 confirmed MEDIUM+ finding (F-205).

## [got-20260719191457-721022]
Category: gotcha
Tags: python, module, import, constant, stale
Changed: 2026-07-19T19:14:57.344383

Import-time module-level constant snapshot: When a module-level constant is defined by evaluating another constant at import time (e.g., MAX_TOKENS_PER_LINE = MAX_CURVES), overriding the source constant at runtime does NOT propagate to the dependent constant. The dependent constant captures the value at import time — a stale snapshot. Documented overrideable behavior is silently broken. This chains across modules: when module B does from .module_a import MAX_CURVES, it captures a snapshot at B's import time too — two independent stale copies. Fix: use lazy evaluation (a property or function returning the source constant) instead of import-time assignment for derived constants. In pylasdev-reborn: 3 files affected across 2 iterations — data_reader.py:37, dev_reader.py:18-22, parser.py:23-27 (F-MDR-03, F-DVR-01, F-I2-XMD-02). All three documented as 'overridable' but override mechanism silently broken at import time.

## [got-20260720004351-133d1d]
Category: gotcha
Tags: python, type-safety, bool, int, isinstance, pylasdev
Changed: 2026-07-20T00:43:51.614871

Python bool-as-int isinstance vulnerability: bool subclasses int in Python, so isinstance(True, int) returns True and isinstance(False, int) returns True. Any isinstance(value, int) guard in validation or coercion code silently accepts bool values, which then propagate as True/False where integers are expected — f-string formatting outputs True instead of 1, dict lookups succeed unexpectedly, and type contracts are violated with zero warning. Fix: use type(value) is int for exact type matching when bool should be rejected. Do NOT use isinstance(value, int) if bool values need to be caught. Checklist: (1) grep for isinstance(*, int) in validation code, (2) replace with type() is not int where bool is invalid input, (3) add regression tests with True/False inputs. In pylasdev-reborn: 3 confirmed MEDIUM+ findings across 3 fix convergence passes (F-7-001 ParameterZone zone_index, F-8-001 array_index at line 84, F-9-002 _resolve_dict_entry) — each missed instance was found in a separate stage because the fix agent fixed one site but missed the others. Centralize the bool rejection in a shared guard function to prevent recurrence.

## [got-20260720004359-f2916a]
Category: gotcha
Tags: numpy, dtype, string-truncation, np.str_, data-integrity, pylasdev
Changed: 2026-07-20T00:43:59.487357

numpy str_ dtype produces fixed-width strings with silent truncation: np.empty(N, dtype=np.str_) defaults to <U1 (1-character Unicode), silently truncating all multi-character string values to their first character. np.array(['hello', 'world'], dtype=np.str_) also produces <U5 (fixed-width, determined by longest input at construction time) — subsequent assignments of longer strings are truncated. This is distinct from Python's native str behavior and causes silent data corruption that passes all type checks and shape validation. Fix: use dtype=object for variable-length string data when storing arbitrary strings. For string arrays that need numpy's vectorized operations, explicitly specify the max string width with dtype='<U{max_width}'. Checklist: (1) grep for dtype=np.str_ in numpy allocation code, (2) for variable-length string columns use dtype=object, (3) for fixed-width text columns provide explicit max width, (4) add tests with strings longer than expected width to catch truncation. In pylasdev-reborn: np.empty((data_lines, num_curves_str), dtype=np.str_) at data_reader.py:634-635 produced <U1 arrays that truncated all string curve values to single characters — silent data loss that required adversarial verification to confirm reachability. 1 confirmed MEDIUM finding (F-H-006).

## [pat-20260720004406-402cd7]
Category: pattern
Tags: python, dataclass, validation, lazy-init, deferred-population, pylasdev
Changed: 2026-07-20T00:44:06.685243

Deferred population validation timing: when an object is constructed with empty/default state and populated later via deferred/lazy methods, validation checks in __post_init__ or constructor fire against the empty state before data arrives — producing spurious warnings or errors for conditions that will be satisfied after population completes. Checklist: (1) identify objects that support deferred population (constructed first, populated later), (2) add emptiness-guards before validation checks — skip cross-field validation when both/all related fields are still empty/default (indicating not-yet-populated state), (3) ensure the deferred-population method performs the same validation AFTER populating data, (4) do NOT remove validation from __post_init__ entirely — it must still run when data is provided at construction time. In pylasdev-reborn: parser constructed DataSection at parser.py:2418 without data/string_data kwargs, then populated them at line 2496-2497 — __post_init__ fired uncovered-curve validation against empty dicts, producing spurious warnings for all curves. Fix: early-return guard in __post_init__ when both data and string_data are empty (deferred population indicator). 1 confirmed MEDIUM finding (F-7-004), cross-domain verified.

## [pat-20260720004413-7b4db0]
Category: pattern
Tags: python, csv, delimiter, validation, input-guard, pylasdev
Changed: 2026-07-20T00:44:13.494813

Single-character delimiter validation before csv.reader: when a public API accepts a delimiter character and passes it to Python's csv.reader(), validate that the delimiter is exactly one character BEFORE the csv.reader call. csv.reader(delimiter='::') raises TypeError ('delimiter' must be a 1-character string) which is often unhandled — the csv.Error catch does not cover TypeError. Even when TypeError is caught, multi-char delimiters produce ambiguous parsing: csv.reader treats each character of a multi-char string as a separate delimiter possibility. Checklist: (1) every API entry point that accepts a delimiter parameter MUST validate len(delimiter) == 1 before any csv.reader call, (2) catch BOTH csv.Error AND TypeError at csv.reader call sites — csv.Error covers malformed input, TypeError covers bad delimiter length, (3) consider also rejecting delimiter characters that are alphanumeric or conflict with format-significant characters. In pylasdev-reborn: dev_reader.py accepted arbitrary delimiter strings from the DEV file specification and passed them directly to csv.reader without length validation — a multi-char delimiter raised unhandled TypeError bypassing the csv.Error catch. 1 confirmed MEDIUM finding (F-H-007), adversarial-verified.

## [got-20260720040315-5c2cb9]
Category: gotcha
Tags: numpy, masked-array, integer, crash, comparison, pylasdev
Changed: 2026-07-20T04:03:15.268997

numpy MaskedArray .filled(np.nan) crashes on integer-dtype arrays: np.ma.MaskedArray IS an ndarray subclass (isinstance(arr, np.ndarray) returns True), but .filled(np.nan) raises TypeError: 'Cannot convert fill_value nan to dtype int32' for integer-dtype (kind 'i'/'u') MaskedArrays. Must call .astype(np.float64).filled(np.nan) first. The dtype exclusion guard must include integer kinds ('i','u'), not just string/object kinds. When comparing/computing on MaskedArrays, always check for masked arrays BEFORE isinstance(arr, np.ndarray) because MaskedArray passes that check. In pylasdev-reborn: compare.py F-023 + R8-002, 2 CONFIRMED findings across 3 synthesis stages.

## [got-20260720040321-a7ecee]
Category: gotcha
Tags: python, str, bytes, repr, corruption, type-safety, pylasdev
Changed: 2026-07-20T04:03:21.487403

str(bytes) produces Python repr form, NOT decoded text: str(b'utf-8') → "b'utf-8'" (with b-prefix and quotes), not 'utf-8'. This is silent corruption when bytes values flow through generic str() conversion in helper functions (e.g., _safe_str). The resulting repr-encoded string passes all type checks and propagates as if valid → downstream crashes (LookupError for encoding lookup, KeyError for dict keys). Fix: add isinstance(value, bytes) guard before str(value), raising TypeError with 'Decode to str first: value.decode()' message. Following the existing codebase pattern is preferred over implicit decode-in-place. In pylasdev-reborn: models.py _safe_str had 42 call sites with zero bytes guards; 11 other locations in the same file had explicit bytes guards (F-017, 1 CONFIRMED MEDIUM finding).

## [pat-20260720040325-3f7d26]
Category: pattern
Tags: python, save-restore, try-finally, mutation, exception-safety, mutable-state, pylasdev
Changed: 2026-10-04T20:19:18.636231

Write/serialize operations must not permanently mutate the input model, and shared-state
save/restore must use try/finally.
(a) Functions that mutate shared mutable objects for internal computation (copy-back of data_sections to
    las_file, overriding version.wrap for format compliance) must restore the original state before
    returning - the model may be reused (double-write: first write mutates logs, second sees stale
    state) or inspected post-write. Save BEFORE the try block, restore as the ONLY code in a finally
    block, and use finally even when operations appear infallible.
(b) The INVERSE: a field the writer deliberately overrides for format compliance (wrap='NO' for LAS 1.2
    WRAP=YES) must NOT be restored - the post-write model must reflect disk truth, else to_dict()
    claims a value the file does not contain. An override+restore pair is always a bug (either the
    override is right and the restore wrong, or vice versa).
Checklist: classify each writer mutation as internal-computation (restore) vs compliance-override (do
not restore); enumerate every operation between save and restore; recompute derived values after
mutation (a limit computed pre-mutation is stale); assert post-write model state matches the written
file. In pylasdev-reborn: writer.py; G-018, R8-007, S10-001, F2-26, M-06.

## [got-20260720040336-9b5fc1]
Category: gotcha
Tags: python, float, OverflowError, exception, Python3.12, pylasdev
Changed: 2026-07-20T04:03:36.037034

float() can raise OverflowError (NOT ValueError) for extreme values: On Python 3.12.0-3.12.1 specifically, float('1e500') raises OverflowError instead of returning inf. The behavior was reverted in 3.12.2, but libraries with requires-python >= 3.12 must still handle it. OverflowError is NOT a subclass of ValueError — except ValueError clauses silently miss it. Checklist: (a) every except ValueError around float() conversion should also catch OverflowError: except (ValueError, OverflowError), (b) check the same for int() conversion (also raises OverflowError on float('inf') → int()). In pylasdev-reborn: _to_finite_float() caught only ValueError; 5 np.array() sites missed OverflowError. 2 CONFIRMED MEDIUM findings (G-014, G-017) across 2 different exception-handling patterns.

## [got-20260720040338-93d5d9]
Category: gotcha
Tags: python, numpy, OverflowError, exception, array, pylasdev
Changed: 2026-07-20T04:03:38.641261

np.array() with extreme values raises OverflowError: np.array([10**400], dtype=np.float64) raises OverflowError, which is NOT a subclass of ValueError, TypeError, or MemoryError. Common except clauses like 'except (ValueError, TypeError, MemoryError)' silently miss it. OverflowError escapes through nested try/except wrappers. Fix: add OverflowError to ALL except tuples guarding np.array() calls: except (ValueError, TypeError, MemoryError, OverflowError). Also add to outer wrappers that re-raise as domain exceptions. Checklist: (a) grep for np.array() in except blocks, (b) verify OverflowError is in every tuple, (c) verify it's also in outer re-raising wrappers — missed in inner = missed everywhere. In pylasdev-reborn: 5 np.array() sites + 2 outer wrappers = 7 locations missing OverflowError (G-017, 1 CONFIRMED MEDIUM finding).

## [got-20260720040358-cdb557]
Category: gotcha
Tags: python, exception, dead-code, regression, pass, pylasdev
Changed: 2026-07-20T04:03:58.884452

Dead except blocks suppress future regressions: When a try/except is guarded by multiple independent pre-checks that make the exception provably unreachable (e.g., range checks, type checks, pre-allocation guards), the dead except block silently swallows any exception if a future refactor breaks ONE of the guards. The silent pass suppresses the regression with zero diagnostics. Checklist: (a) remove dead except blocks — let unexpected exceptions crash loudly, (b) if the exception COULD become reachable through a future code path, convert to except ExceptionType: raise DomainError(...) with context — this surfaces the regression with diagnostics, (c) never leave bare 'except: pass' or 'except SpecificType: pass' at guarded assignment sites. In pylasdev-reborn: 4 identical 'except IndexError: pass' sites in dev_reader.py had 3 independent pre-checks each (data_lines, range, np.full pre-allocation) making IndexError unreachable. All 4 survived adversarial review as CONFIRMED (G-012, 1 CONFIRMED MEDIUM finding).

## [got-20260720040405-f9625b]
Category: gotcha
Tags: python, reader, empty, validation, silent-failure, pylasdev
Changed: 2026-07-20T04:04:05.617494

Data readers silently returning empty output on all-comment/empty input hide upstream data issues: When a file parser completes parsing with zero content lines (all lines are comments, whitespace, or comments-only), returning an empty data object without warning or error masks upstream data problems. The caller receives valid output with zero rows/columns and no indication the input was empty. Fix: after content scanning, check if any content lines were found. If data_lines == 0, raise a domain error (e.g., 'No data lines found in file') or emit a clear warning. Empty output with zero diagnostics defeats downstream validation — all subsequent operations on the empty result may succeed silently with zero data. Checklist: (a) count non-comment, non-blank lines during pre-scan, (b) after pre-scan, verify count > 0 — if zero, raise or warn before constructing the output object, (c) test with files containing only comments, only whitespace, and truly empty files. In pylasdev-reborn: dev_reader.py returned empty DevFile for all-comment files — zero content seen, zero columns, zero data, zero warnings (G-019, 1 CONFIRMED MEDIUM finding).

## [pat-20260720071838-171ab7]
Category: pattern
Tags: pylasdev, parallel-code-path, mirror, fix, regression, audit
Changed: 2026-10-04T20:19:18.636234

Parallel/sibling code-path omission - THE #1 recurring defect class: when a bug, fix, guard, or
validation is applied to one code path, the same defect propagates through every structurally-similar
path. Highest-risk structures: fork-in-code (if/elif/try branches), per-section for loops,
auto-detection vs explicit-parameter branches, fallback/default else-branches, and duplicated mirrored
modules (parser vs data_reader, _writer_base vs _writer_las30, legacy vs LAS 3.0 twin).
Evidence across runs: sibling guard asymmetry (_write_version_section computed is_las30 but not is_las12
while sibling _write_well_section computed both; data_sections silently continued on non-dict elements
while sibling paths raised TypeError); incomplete fix across explicit + auto-detected CWLS branches (the
auto-detected branches kept the identical value/description inversion); heuristic variant parity (DUG
Pattern A had three fallback layers, Pattern B none -> all-numeric column names failed); validation
guard asymmetry between sibling functions (_compare_lists had type guards, _compare_data_sections did
not); cross-domain fix-asymmetry (desanitize fixed on the parser copy only, data_reader copy still
blanket-unescapes; wrap fixed on one of two twins; LAS 3.0 section-uncovered warning stayed exact-case
while the legacy twin was fixed); repeated leaf patches on a regressing function while sibling shapes
stayed broken.
Fix-coordination corollary: a single logical fix spanning two files (a marker/flag SET in one module and
CONSUMED in another) fails repeatedly when split across fix agents - keep the full producer+consumer
change in ONE agent's writable scope, or the post-fix review must grep both sides (see
pat-20260806084858-5ea18b).
Checklist: after EVERY fix or validation addition, grep for structurally-similar code in sibling
functions, per-section/branch siblings, parallel call sites, and mirrored module pairs - fix ALL of them
in the SAME pass with cross-domain post-fix review. Compare gate expressions verbatim (git diff the two
bodies). When copies cannot be merged, extract ONE shared helper or add a verbatim-mirror contract test
on a shared fixture matrix. Also enumerate construction/wholesale/mutation/pickle/deepcopy entry points
and top-level-vs-section views; a fix at one entry point or view leaves the defect live at the twins
(see pat-20260809134744-17c419).

## [got-20260720071943-a34bd7]
Category: gotcha
Tags: pylasdev, normalization, canonicalization, collision, case-insensitive, lookup, validation
Changed: 2026-10-04T20:19:18.636236

Normalization/canonicalization collision detection: when two distinct raw keys normalize to the
same canonical key, normalization MUST detect and reject the collision rather than silently
overwriting. (a) Dict alias normalization (MDKB->MD, md->MD) silently overwrites when input has both a
canonical key and an alias (models.py:2466-2470) - before `data[norm_key] = data.pop(raw_key)`, check
`if norm_key in data: raise ValueError` naming both keys. (b) Case-insensitive index build: 'aGK':'GK'
and 'Agk':'GRO' both uppercase to 'AGK'; sorted first-wins silently resolves to the wrong canonical
value and makes the other entry dead code through the parser path. Detect collisions at build time
(len(raw) vs len(upper), or a collision map) - sorted first-wins is the WORST tiebreaker
(non-deterministic across Python versions and data-dependent). Prefer explicit collision resolution
(reject ambiguous entries, or keep both with a disambiguation mechanism). Reference-data case
normalization must cover ALL entries - even 0.5% mixed-case entries break lookup for that subset
silently; add a regression test iterating all entries.
Checklist: every normalization/alias map must check collisions between normalized keys and pre-existing
keys AND between two raw keys normalizing to the same target. In pylasdev-reborn: F-030, F-013.

## [got-20260720071943-b7686e]
Category: gotcha
Tags: pylasdev, aggregation, data-loss, string, numeric, max-len
Changed: 2026-07-20T07:19:43.629913

Aggregation over heterogeneous data types — when computing an aggregate (max, min, sum) across typed data, consider ALL types, not just the primary type: F-007 (data_reader.py:1198-1240): _max_len computed from float curve indices only via 'max(len(data_lists[i]) for i in _float_indices)'. When all curves are string-formatted, _float_indices is empty, _max_len = 0, and ALL string data is silently truncated to empty arrays. Fix: also compute max from string curve lengths via 'max(_string_lists.values())'. Checklist: for every aggregate over indexed/typed collections, enumerate all possible types/indices and verify each contributes to the aggregate. An empty primary type should not zero out the aggregate — use 'default=0' with explicit fallback to secondary types.

## [got-20260720071943-c7f9db]
Category: gotcha
Tags: pylasdev, numpy, dtype, exclusion, drift, data-loss
Changed: 2026-07-20T07:19:43.705669

Dtype exclusion/gate set drift — when a guarded conversion path excludes certain dtypes from a lossy conversion, new dtypes added later bypass the exclusion silently: F-027 (compare.py:251): _compare_arrays excluded string/object/void dtypes from .astype(np.float64) path but missed 'c' (complex). Complex arrays routed through .astype(np.float64) — silently dropping imaginary components. Fix: after every addition of a new supported dtype, verify all exclusion/guard sets in the same module include it. When the exclusion set uses dtype.kind letters, also check for 'c' (complex) and 'i'/'u' (integer, for MaskedArray .filled(nan) crash — see got-20260720040315-5c2cb9).

## [got-20260720071943-d42702]
Category: gotcha
Tags: pylasdev, validation, dict, mutation, write-back, truncation
Changed: 2026-07-20T07:19:43.780663

Validation-by-mutation of dict inputs must write back the mutated value: when a validation function mutates a dict key value (e.g., truncating 'F8.3' to 'F') and passes the dict to a downstream constructor, the mutated value MUST be written back to the dict. F-017 (models.py:326) and F-R01 (models.py:348): _validate_from_dict_input truncated extended format codes 'df = df[0]' for validation but didn't write back 'cd[data_format] = df' or 'sc[data_format] = df'. The untruncated value passed through to CurveDefinition.__post_init__ which rejected 'F8.3' as not in _VALID_DATA_FORMATS. Fix: after every dict value mutation during validation, write back the mutated value to the dict key. The top-level loop and per-section loop are parallel paths — fix both.

## [pat-20260720071943-585efe]
Category: pattern
Tags: pylasdev, validation, gating, early-return, boolean-flag, coverage
Changed: 2026-07-20T07:19:43.855634

Validation gating — use boolean flags instead of early return in multi-stage validation functions: When a validation function has multiple independent stages (e.g., MD checks, azimuth checks, inclination checks), an early 'return' in one stage silently skips all subsequent stages. F-019 (dev_reader.py:546,562,566): three 'return' statements inside the MD block of _validate_dev_data exited the entire function, skipping azimuth (lines 600-622) and inclination (lines 624-646) range checks. Fix: replace 'return' with a boolean flag '_md_check_ok = False' that gates only the MD-specific checks. The function then falls through to the remaining stages. Checklist: every 'return' inside a validation function that has code after it should be a boolean gate, not a function exit.

## [pat-20260720104554-d19a5e]
Category: pattern
Tags: python, from-dict, input-mutation, defensive-copy, pylasdev
Changed: 2026-07-20T10:45:54.002844

Don't mutate caller's input dict — defensive copy first: from_dict() and constructor methods that accept mutable input containers (dicts, lists) must NOT mutate the caller's input. Always perform defensive copy before any mutation, or document the mutation explicitly in the docstring. F-40: 4 mutation sites in LASFile.from_dict (models.py:327,349,1658,1679) — data_format truncation, data.pop(), mnemonic normalization all mutated the caller's dict. F2-12: 5th mutation site in DevFile.from_dict (models.py:2492) — data.pop() with normalization mutated a completely different from_dict method. Both fix approaches converged on: copy data before mutating. Checklist: (a) for every from_dict/constructor: grep for data.pop(), key reassignment, or data[key] = ... in validation paths, (b) add data = data.copy() (shallow) at the top of the method before any mutation, (c) document in the docstring if mutation is intentional (prefer copy-and-mutate over mutation-and-document), (d) check sibling from_dict methods — F2-12 was a DevFile.from_dict mutation found in iter 2 that iter 1 missed because it only audited LASFile.from_dict.

## [got-20260720104601-5a5b73]
Category: gotcha
Tags: python, nan, comparison, ordering, float, pylasdev
Changed: 2026-07-20T10:46:01.007981

NaN comparison shortcut bypasses NaN-aware fallback: When a comparison function has a fast-path shortcut that compares entire objects (e.g., l1 != l2 for lists) before a NaN-aware element-by-element fallback, NaN-containing data silently takes the shortcut. NaN != NaN returns True, so l1 != l2 evaluates True for NaN lists → returns False without ever reaching the NaN-aware code below. F2-19 (compare.py:357): _compare_lists had if l1 != l2: return False as a fast-path shortcut at line 357, then a NaN-aware _scalars_equal fallback at line 378. When both lists were [float('nan')], the shortcut returned False — NaN lists compared NOT EQUAL despite NaN==NaN being the module's convention. Fix: either (a) check for NaN before the generic shortcut: if any math.isnan(x) for x in l1 ... route to per-element, or (b) remove the generic shortcut and always go through per-element comparison for list types. Checklist: in every comparison function, check whether any fast-path shortcut could produce NaN-dependent results — if NaN!=NaN would make the shortcut produce the wrong answer, move NaN detection before the shortcut.

## [pat-20260720104605-c03639]
Category: pattern
Tags: python, validation, mutation, side-effect, ordering, pylasdev
Changed: 2026-07-20T10:46:05.130857

Validate before mutating global/shared state: When a function must both validate an operation and mutate shared state, perform all validation checks BEFORE any mutation. Mutating state first and then discovering the operation is invalid leaves state permanently corrupted with no recovery path. F2-07 (parser.py:2472-2553): _deduplicate_curves mutated global las_file.curves/curves_order at line 2473 BEFORE checking actual_count > 0 at line 2544. If the first data section contained only comments/blanks, actual_count == 0 → early return WITHOUT saving a DataSection, but global curves were already mutated to deduped names with no backing data. Fix: move mutations AFTER all validation gates pass. The validator should be pure — it inspects state, returns a boolean, and the caller mutates only if validation succeeds. Checklist: (a) in any function that both validates and mutates: verify that every mutation site is after every validation check, (b) if validation is complex, split into validate() → mutate() → commit() stages, (c) ensure early-return paths don't leave partial mutations — check every return/raise between first mutation and last mutation.

## [got-20260720104609-a00537]
Category: gotcha
Tags: python, dataclass, validation, post-init, mutation, pylasdev
Changed: 2026-07-20T10:46:09.232734

Post-construction attribute assignment bypasses __post_init__ validation: Python dataclass __post_init__ runs exactly once at construction time. Any code that assigns attributes after construction (obj.attr = value on an already-constructed object) bypasses all __post_init__ validation — the guard fires against the constructor values, not the post-construction values. F-28 (parser.py:2214-2232, models.py:793-801): ParameterEntry was constructed with default data_format, then data_format was assigned post-construction — __post_init__ validated the default, not the actual value. When data_format was set to an invalid value, no validation caught it. Fix: (a) prefer passing all values through the constructor so __post_init__ validates them, (b) if post-construction assignment is unavoidable, add a setter method that re-runs validation: def set_data_format(self, df): validate(df); self._data_format = df, (c) at minimum, add an explicit validation call immediately after post-construction assignment with a comment: self.data_format = df; self._validate_data_format()  # bypasses __post_init__. Checklist: grep for self.attr = ... after dataclass construction in the same function — each such line is a __post_init__ bypass.

## [pat-20260720104613-1546fe]
Category: pattern
Tags: python, metadata, validation, consistency, pylasdev
Changed: 2026-07-20T10:46:13.834126

Metadata ordering lists must be cross-validated against data keys: When a data model has a separate ordering list (column_order, curves_order, section_order) alongside a data container (columns dict, curves dict), the ordering list MUST be cross-validated against the container's keys. Orphaned entries in the ordering list (keys not present in the data container) create silent inconsistency — downstream consumers may trust the ordering as authoritative and either crash (KeyError) or produce incomplete output. F2-32 (models.py:2590-2651): DevFile.from_dict set column_order from user input but never checked entries against dev.columns.keys(). Inference only ran when column_order was empty/falsy. If user passed column_order=['MD','GR'] but columns={'MD': [1,2,3]}, 'GR' survives as an orphan — dev.columns['GR'] raises KeyError, writer silently skips it. Fix: after column_order is set: for col in column_order: if col not in columns: raise ValueError(f'column {col} in column_order but not in columns'). Checklist: (a) identify every data model field that is an ordering list paired with a data container, (b) add a cross-validation check in __post_init__ or from_dict that every ordering entry exists as a key in the container, (c) also validate the reverse: warn if any container key is NOT in the ordering list (unless ordering is optional/auto-inferred).

## [pat-20260720104619-db1479]
Category: pattern
Tags: python, exception, error-handling, api-contract, module-boundary, third-party, pylasdev
Changed: 2026-10-04T20:19:18.636242

Exception-handling contract consistency:
- Module boundary: every raise site in a module's public API must use the module's documented exception
  type - a general exception from another domain (LASDataError in a parser that documents LASParseError,
  RuntimeError anywhere) escapes the caller's catch block. grep `^\s*raise\s+\w+Error` per module; each
  match must be the module's documented type (F-01, F2-20).
- Catch the FULL parent hierarchy: `except UnicodeDecodeError` misses LookupError (parent of
  codecs.LookupError, raised for invalid encoding names) - use except (UnicodeDecodeError, LookupError)
  at every entry point (reader, dev_reader, writer, encoding) (F-acc693).
- Wrap ALL third-party exceptions (csv.Error, numpy MemoryError/OverflowError, encoding errors) in a
  domain exception before the public API boundary; csv.reader is high-risk (csv.Error on malformed
  input).
- Code placed BEFORE the try: block is NOT covered by the except clause - move validation inside the
  try, or the documented wrapping contract silently fails.
- When adding a raise inside a try, update the downstream except clauses to catch it (LASParseError
  inherits PylasdevError->Exception, NOT ValueError/TypeError); verify ALL outer re-raising wrappers
  catch it too - a missed inner catch is a missed catch everywhere.
- Dead except blocks (provably unreachable due to pre-checks) silently swallow a future regression when
  one guard breaks - remove it, or re-raise as a domain error with context; never leave bare
  `except: pass`.
- Self-defeating handler: an except that retries the SAME operation with identical inputs is a
  guaranteed crash path - change the input or re-raise.
- Docstring @raises contract: every listed exception must have a reachable raise path - an unreachable
  documented exception creates dead catch blocks in consumers (F-219).
In pylasdev-reborn: parser.py, dev_reader.py, models.py; F-01, F2-20, F-026, F-219, acc693, F-I2-DV-05.

## [got-20260720104633-378dcc]
Category: gotcha
Tags: python, writer, serialize, model-state, compliance, pylasdev
Changed: 2026-07-20T10:46:33.670890

Writer model must reflect disk truth for compliance-overridden fields — don't restore pre-write values that the writer deliberately changed: When a writer overrides a model field for format compliance during serialization (e.g., forcing wrap='NO' for LAS 1.2 WRAP=YES files), the model state after write must reflect what was actually written to disk — NOT the pre-write value. Restoring the compliance-overridden field to its pre-write value creates a model-disk contradiction: the model claims one value (wrap='YES') but the file contains another (WRAP=NO). F2-26 (writer.py:366-425): the writer correctly set las_file.version.wrap = 'NO' at line 389 to match what was written, but the finally block at line 425 unconditionally restored original_wrap — for wrap='YES' input, model claims YES but file says NO. Compounding: to_dict() returns {'WRAP': 'YES'} but the actual file is WRAP=NO. For wrap=None: model restored to None, to_dict() violates dict[str,str] contract. This is the inverse of pat-20260720040325-3f7d26 (which correctly advises saving/restoring internal computation state): for compliance-overridden fields the writer intentionally changes, DO NOT restore — the model must be honest about what was written. Checklist: (a) identify which model mutations during write are internal computation (restore these) vs. compliance overrides (do NOT restore these), (b) if a field has two purposes, flag the conflict — an override+restore pair is always a bug (either the override is correct and restore is wrong, or vice versa), (c) after write, assert las_file.version.wrap == actual_written_value — verify model-disk consistency.

## [got-20260720203918-eae414]
Category: gotcha
Tags: python, copy, mutation, defensive-copy, shallow-copy, deepcopy, aliasing, pylasdev
Changed: 2026-10-04T20:19:18.636244

Shared mutable references across object boundaries: (a) a property returning an internal numpy
ndarray as a mutable view lets caller mutations silently propagate to internal state and vice versa -
return .copy() for public mutable data (read-only view + documented aliasing only in internal hot
paths); (b) a shallow dict.copy() leaks nested mutable values (nested dict/list/DataSection) - use
copy.deepcopy() when nested values are mutable, or immutable nested types (tuple of NamedTuples,
frozenset) to make shallow copy safe; (c) direct-construction paths (LASFile(logs=)/DevFile(columns=))
must deepcopy caller input, not alias it (from_dict already does), and the deepcopy must not skip
already-guarded dicts (that reintroduced aliasing, M-14). Document shallow-vs-deep copy semantics as a
contract.
In pylasdev-reborn: models.py - models() returned a shallow copy of _curves but the nested DataSection
values dict was a shared reference (F-002 HIGH, both-found); N-I-11; M-14.

## [got-20260720203931-de7ced]
Category: gotcha
Tags: python, validation, hardening, inconsistent, guard, half-fix, pylasdev
Changed: 2026-07-20T20:39:31.321361

Guard/store hardening asymmetry: When a data flow has a guard (input validation) and a store (internal storage), and they use different hardening functions — the guard uses str() while the store uses _safe_str() — the hardening is only as strong as the weaker function. The guard passes values that the store would reject, or the store silently accepts values the guard would reject, creating two divergent truth paths. A prior fix may have hardened the store (adding _safe_str()) but left the guard using bare str() — classic half-fix where only the downstream symptom was treated, not the upstream cause. In pylasdev-reborn: F-204 (MEDIUM, PRIOR_FIX_ATTEMPT at da1f696) — prior fix da1f696 added _safe_str() hardening to the store path at models.py:1769 but left the guard at models.py:1762 using bare str(). The guard accepted None and non-finite floats that str() silently converts but _safe_str() rejects — producing an inconsistent validation/inconsistency between guard approval and store rejection. Fix: (a) after adding a hardening function at any downstream site, grep for ALL upstream validation sites on the same data flow and verify they use the same function, (b) extract hardening into a single shared function used by both guard and store, (c) when documenting a fix, explicitly list ALL sites that the hardening applies to — a site not listed is a gap. Checklist: for every _safe_* or _validate_* helper, grep for bare str()/float()/int() calls on the same data attribute in the same module — each bare call is a potential hardening gap.

## [got-20260720203938-5168b4]
Category: gotcha
Tags: python, docstring, raises, contract, exception, library, pylasdev
Changed: 2026-07-20T20:39:38.199973

Docstring @raises contract violation — documented exception never propagated: When a function's docstring lists an exception in its Raises section but the exception is never actually raised or propagated to the caller, consumers who catch that exception type based on the documented contract have dead code — their except block will never execute, and the actual error propagates uncaught. This creates a false sense of safety: the documentation says the function can raise X, but it never does. In pylasdev-reborn: F-219 (MEDIUM) — LASFile constructor's __init__ at reader.py:123-124 documented Raises: LASEncodingError but the encoding error was caught and converted to a warning at line 142, never propagated as an exception. Callers checking for LASEncodingError would have unreachable except blocks. Fix: (a) after the final except block in every function, grep for the Raises section and mechanically verify every listed exception type has at least one reachable raise path, (b) remove unreachable exception types from the Raises section, (c) alternatively: if the exception SHOULD be raised, restructure the error handling to propagate it instead of silently downgrading to a warning. Checklist: for every function with a Raises docstring section, grep for every exception type listed — each must have at least one raise SomeError statement reachable from the function entry, not gated by an always-false condition and not consumed by a catch-all except that never re-raises.

## [got-20260721022227-286b1e]
Category: gotcha
Tags: pylasdev, parser, regression
Changed: 2026-07-21T02:22:27.924526

pylasdev F-031 (las_file.logs multi-section population) is DEFECTIVE: removing is_first_section guards at parser.py:2873/2885 causes multi-section LAS 3.0 roundtrip regression. Sections have heterogeneous row counts, blending them into las_file.logs breaks from_dict F-25 validation. Multi-section data already accessible via data_sections[i].data. Do NOT re-apply F-031.

## [pat-20260721022228-f9ba72]
Category: pattern
Tags: pylasdev, data-reader, state-machine
Changed: 2026-07-21T02:22:28.005048

pylasdev data_reader _read_wrapped state machine: after non-pathological depth_had_extra warning at L1151, MUST reset depth_line=True AND counter=0 in addition to depth_had_extra=False. Missing these resets causes next depth line to enter data branch → silent permanent cross-curve value shifting (E-F-018 HIGH severity).

## [got-20260721043454-ef9e98]
Category: gotcha
Tags: python, numpy, dispatch, 0-d-array, isinstance, comparison
Changed: 2026-07-21T04:34:54.676450

0-d numpy ndarray passes isinstance(arr, np.ndarray) check but should be treated as scalar — ndarray.item() conversion is unreachable: numpy 0-dimensional arrays (np.array(3.14), np.float64(42)) are genuine ndarray instances — isinstance(arr, np.ndarray) returns True. This means a type-dispatch chain that checks np.ndarray BEFORE scalar types routes 0-d arrays into array-comparison code paths that call .astype(), .filled(), or broadcast operations — operations that work on 0-d arrays but produce subtly wrong results (broadcast of 0-d array against scalar produces 0-d result — not what the scalar path would produce). The existing scalar-handling code with .item() conversion sits dead because the isinstance(np.ndarray) gate fires first. This is the COMPLEMENT of got-20260718050212-863dce (which covers 'check BOTH operands'): even with perfect two-operand checks, 0-d arrays still pass the ndarray gate unless you explicitly check .ndim == 0 first. Fix: before routing to array-comparison code, add 'if val.ndim == 0: val = val.item()' (or route to scalar path). Alternative: check 'isinstance(val, np.ndarray) and val.ndim > 0' to split 0-d from proper arrays. Checklist: (a) grep for isinstance(*, np.ndarray) in dispatch/comparison code, (b) verify 0-d arrays are caught before the array path, (c) test with np.float64(3.14) and np.array(42) as inputs. In pylasdev-reborn: compare.py:183 — isinstance(val2, np.ndarray) caught 0-d arrays before _scalars_equal path at line 201, making the existing 0-d handling unreachable. Confirmed MEDIUM (F-028).

## [pat-20260721043501-1b33bd]
Category: pattern
Tags: python, validation, mutation, write-back, dataclass, sanitization
Changed: 2026-07-21T04:35:01.287172

Validate-then-store: sanitized/transformed value computed locally but never assigned back to the object: A validation __post_init__ or processing function computes a sanitized value (e.g., .strip(), .lower(), .replace()) into a local variable but never writes it back to the target attribute — the original unsanitized value persists. The function 'validates' but the validation has zero effect on stored state. This is mechanically different from dict-mutation write-back (got-20260720071943-d42702) — here the local variable IS the sanitized value, computed directly from the attribute, but the assignment back is missing. The pattern: '_stripped = self.field.strip()' with no 'self.field = _stripped'. The function appears to validate (strip is called) but the field retains leading/trailing whitespace. Fix: after every local sanitization variable, mechanically verify it is assigned back: self.field = _sanitized. Checklist: (a) grep for '_stripped\|_sanitized\|_normalized\|_cleaned' local variable assignments in __post_init__ and validation functions, (b) verify each has a corresponding 'self.attr = _var' assignment after the transform, (c) prefer inline assignment: self.field = self.field.strip() — no intermediate variable to forget. In pylasdev-reborn: models.py ParameterEntry.__post_init__ and DataSection.__post_init__ — _stripped = self.section_type.strip() computed but never assigned back; whitespace survived validation. Confirmed MEDIUM (F-030).

## [pat-20260721043509-65432f]
Category: pattern
Tags: python, resource-guard, cumulative, parser, streaming
Changed: 2026-07-21T04:35:09.018131

Per-instance resource guard needs cumulative cross-instance counter: When a resource limit is enforced per-section/per-item but the resource cost sums across ALL instances, the per-instance guard passes each individually while the combined total exceeds the limit. The from_dict path (which processes all sections at once) has a cumulative counter; the streaming parser path (which processes sections sequentially) only checks each section against MAX_TOTAL_ELEMENTS independently — N sections each at the limit produce N × LIMIT elements through a guard designed to allow LIMIT total. Fix: add a cumulative counter that persists across all sections/items and check pre-increment, matching the from_dict implementation. This is the streaming equivalent of got-20260719060338-113b9f (disjoint pools need sum not max) — both are about cumulative semantics but the gap here is structural (no counter exists at all, not wrong arithmetic). Checklist: (a) for every MAX_* guard in a streaming parser, grep for the same guard in from_dict — if from_dict has a cumulative counter and the parser doesn't, add one, (b) verify the accumulator crosses ALL sections (not just data sections — parameter sections, well sections all contribute), (c) the counter must be pre-increment checked (raise BEFORE adding) to reject exactly-at-limit inputs. In pylasdev-reborn: from_dict had cumulative _total_elements at models.py:2286-2310; parser checked MAX_TOTAL_ELEMENTS per-section at parser.py:2746 with no cross-section counter. Confirmed MEDIUM (F2-006).

## [got-20260721043517-20fe2e]
Category: gotcha
Tags: python, type-checking, dict, iterable, bytes, generator, guard, pylasdev
Changed: 2026-10-04T20:19:18.636248

Type-guard scope traps when distinguishing single values from sequences:
- isinstance(x, str) misses bytes: list(b'GR') -> [71, 82] (integers) propagate as curve names /
  identifiers and silently fail string-keyed lookups, producing empty output with no error. Guard
  single-value inputs with isinstance(value, (str, bytes)) (F-110).
- isinstance(x, Iterable) accepts dict: a guard chain None->str/bytes->Iterable lets a dict through,
  and downstream iteration uses dict KEYS (silently wrong-but-valid output). Add an explicit dict
  rejection BEFORE the Iterable branch; prefer explicit list acceptance over Iterable-as-catch-all
  (F2-014).
- Generator exhaustion: an iterable validated in one loop then re-iterated by a second loop / list
  comprehension yields zero elements - materialize with list() immediately after the type check; grep
  for any function iterating the same parameter twice (F-029).
In pylasdev-reborn: models.py guard chains.

## [pat-20260721043524-c318eb]
Category: pattern
Tags: python, defensive-programming, cap-bypass, min, max
Changed: 2026-07-21T04:35:24.439037

Sequential cap cancellation — min(x, CAP) undone by downstream max(x, uncapped): When a value is capped with min(value, CAP) and then a downstream operation takes max(value, other_value), the cap is structurally negated if other_value can exceed CAP. Pattern: 'capped = min(raw, 100)' followed later by 'result = max(capped, other)' — if 'other > 100', result exceeds the cap. The max() undoes the min() cap. This is an emergent property — neither function is wrong in isolation, the bug is in the ordering and the assumption that a downstream max() will not re-exceed the cap. Fix: apply the cap at the FINAL computation site (after all arithmetic), or cap the downstream input too: 'other = min(other, CAP)'. Alternative: cap after max: 'result = min(max(capped, other), CAP)'. Checklist: (a) trace every capped value through all downstream arithmetic — any operation that can increase the value beyond CAP is a bypass, (b) max(), addition, multiplication, and exponentiation are the most common bypass vectors, (c) prefer single-point capping: apply min(..., CAP) as the LAST operation before use, not the first. In pylasdev-reborn: writer.py:1419 capped decimal_places = min(raw, 100), but line 1434 used max(decimal_places, sig_digits) where sig_digits had no cap → when sig_digits > 100, result exceeded cap. Confirmed MEDIUM (F2-016).

## [pat-20260721043540-57993d]
Category: pattern
Tags: python, encoding, bom, utf-16, utf-32, parity
Changed: 2026-07-21T04:35:40.158181

Encoding BOM detection parity — when UTF-8 BOM is detected, also detect UTF-16/32 BOMs: An encoding detection system that checks for UTF-8 BOM (\xEF\xBB\xBF) but not UTF-16 LE/BE BOM (\xFF\xFE, \xFE\xFF) or UTF-32 LE/BE BOM (\xFF\xFE\x00\x00, \x00\x00\xFE\xFF) silently fails on UTF-16/32 files. Without chardet, a UTF-16LE file without BOM detection will be decoded as ASCII/Latin-1 → every other byte is \x00, producing garbled output with no BOM to signal the correct encoding. The fix: add BOM detection for all UTF variants in the same code path that checks for UTF-8 BOM, and add utf-16, utf-16-le, utf-16-be to FALLBACK_ENCODINGS. The UTF-16/32 BOMs are unambiguous (no overlap with valid ASCII content) and the detection is mechanical — a few bytes at offset 0. Checklist: (a) for every '\xEF\xBB\xBF' (UTF-8 BOM) check in encoding detection code, verify UTF-16 LE/BE and UTF-32 LE/BE BOMs are also checked, (b) add the corresponding encoding names to FALLBACK_ENCODINGS, (c) the BOM bytes are: UTF-16 LE = b'\xFF\xFE', UTF-16 BE = b'\xFE\xFF', UTF-32 LE = b'\xFF\xFE\x00\x00', UTF-32 BE = b'\x00\x00\xFE\xFF'. In pylasdev-reborn: encoding.py detected UTF-8 BOM at line 46 but had zero UTF-16/32 BOM detection; FALLBACK_ENCODINGS had utf-8, cp1252, latin-1 only. Confirmed MEDIUM (F2-025).

## [pat-20260721043546-e1c5f5]
Category: pattern
Tags: python, design, accessor-interface, coupling, encapsulation
Changed: 2026-07-21T04:35:46.798377

Incomplete accessor interface forces external modules to directly access internal attributes — coupling hotspot: When a model class provides public accessor methods (__getitem__, __setitem__, __contains__, get()) but omits key attributes from the interface (e.g., .units, .descriptions on a section class), external modules that need those attributes are forced to bypass the interface and access internal dict attributes directly (.entries, ._units, ._descriptions). This creates 23+ coupling points across multiple modules (writer, parser) that all reach into internal storage — any change to the internal representation requires coordinated changes in every coupled module. The accessor interface being incomplete is the root cause; fixing it surgically is impossible because each coupling point must be individually rewritten to use the new accessor. Fix (when designing): enumerate ALL attributes external modules need from a class and expose them through the public interface BEFORE external modules start using internal attributes. Fix (when retrofitting, NOTE-ONLY due to scope): add the missing accessor methods, then rewrite ALL external call sites — this is a structural refactoring, not a surgical fix. In pylasdev-reborn: WellSection exposed .entries via accessor but not .units or .descriptions; writer access at 8+ call sites, parser at 15+ call sites both reached directly into .entries dict. Confirmed MEDIUM, NOTE-ONLY due to refactoring scope (F2-029).

## [pat-20260721065811-1c7cde]
Category: pattern
Tags: python, performance, regex, optimization, pylasdev
Changed: 2026-07-21T06:58:11.043035

Module-level constant hoisting for hot paths — regex, frozenset, compiled objects: When regex patterns, frozen sets, or other constructed objects are created per-call in hot functions (called up to MAX_CURVES=100K times in a loop), hoist them to module-level constants. Each per-call construction allocates and compiles the object, compounding across the call count. In pylasdev-reborn: `_FORMAT_SPEC_RE` regex and `_KNOWN_CURVE_FORMATS` frozenset in parser.py:418 were reallocated on every call to `_parse_format_spec()`, called in a loop over every curve in every ~C section. Fix: move to module-level constants after the function definition or at module top. Checklist: (a) grep for `re.compile` inside function bodies — any in hot paths should be module-level, (b) grep for `frozenset(`, `set(`, `tuple(` constructing from literal iterables inside loops, (c) prefer module-level for any construction of known-size objects with no dynamic input

## [pat-20260721065815-b01a91]
Category: pattern
Tags: python, spec-compliance, las-spec, validation, pylasdev
Changed: 2026-07-21T06:58:15.902376

Format specification completeness — mechanically verify ALL mandatory constraints: When implementing a data format specification (LAS 1.2, LAS 2.0, LAS 3.0, CSV, DEV, etc.), mechanically enumerate every mandatory constraint from the spec and verify each one is enforced somewhere in the codebase. A single missing constraint silently accepts invalid data that downstream tools will reject. In pylasdev-reborn: the LAS 2.0 constraint that the first curve must be the index curve (depth/measurement channel) was completely absent across the entire codebase — not enforced by parser, writer, from_dict, or `__post_init__`. Any LAS 2.0 file with a non-index first curve was accepted and written without error. Checklist: (a) after reading the spec, build a table of constraints (M rows) vs enforcement code paths (N columns: parser/writer/from_dict/__post_init__), (b) mark each cell as enforced/missing with file:line reference, (c) every row with all-missing cells is a finding, (d) constraints that are partially enforced (some paths yes, some no) are also findings — partial enforcement is false safety

## [pat-20260721065823-376e3e]
Category: pattern
Tags: python, dataclass, validation, design, pylasdev
Changed: 2026-07-21T06:58:23.155231

Public validate() method for dataclass models beyond `__post_init__`: Dataclass models with complex cross-field validation should expose a public `validate()` method that can be called by consumers (writer, from_dict, direct construction) at any point. `__post_init__` runs exactly once at construction time; it skips empty collections (deferred population); it doesn't fire after mutation; and it can't be re-invoked for roundtrip verification. A public method centralizes all validation logic and allows both eager (`__post_init__` calls it) and deferred (writer calls it before emitting) use. In pylasdev-reborn: models.py had validation logic scattered across `__post_init__` (skipped empty collections), `_validate_from_dict_input` (from_dict only), and writer had its own checks — 3 independent code paths with gaps between them (e.g., writer didn't re-validate after deferred population). No single entry point for 'is this model valid?' Checklist: (a) when implementing a data model, expose `validate(complete: bool = True) -> list[str]` that returns a list of error messages (empty list = valid), (b) `__post_init__` calls `validate(complete=not self._deferred)` to skip cross-field checks when collections are empty, (c) each write/emit path calls `validate()` before using model data, (d) the validate method covers: type checks, range checks, cross-field consistency, format specifier validity, mandatory field presence — all validation logic in one method

## [got-20260721180533-81f338]
Category: gotcha
Tags: python, pylasdev, exception, comparison, numpy, bare-except
Changed: 2026-07-21T18:05:33.403987

Bare except in comparison/validation code catches unexpected exception types — numpy ValueErrors from array comparison, OverflowErrors from float-to-int conversion. When the intent is to catch AttributeError/TypeError for missing methods, a bare 'except:' or 'except Exception:' catches numpy broadcast errors, non-finite float errors, and array shape mismatches — all of which should propagate as failures, not be silently swallowed. Fix: narrow bare except to the specific exception types the handler is designed for. Alternative: add isinstance guards before comparison to prevent the unexpected exception from being raised in the first place. Checklist: (a) grep for 'except:' and 'except Exception:' in comparison/validation functions, (b) verify the handler only catches exception types that the code explicitly handles, (c) prefer isinstance/pre-condition checks over try/except for type-dispatch decisions. In pylasdev-reborn: compare.py _scalars_equal at line 164 had bare except that swallowed numpy ValueError from list-of-arrays comparison; F-01-H (HIGH) + F-43 (MEDIUM) — two structurally identical list dispatch gaps at compare.py:164 and compare.py:547.

## [pat-20260721180542-06bcb4]
Category: pattern
Tags: python, pylasdev, numpy, validation, nan, inf, data-integrity
Changed: 2026-07-21T18:05:42.409821

NaN/Inf validation for numeric array data in model constructors: Direct construction of data containers (DataSection, LASFile, DevFile) should validate that array data values are finite (np.isfinite) before storage. NaN and Inf values in numpy arrays pass all type checks and shape validation silently, then cause silent data corruption downstream — writer format strings produce 'NaN'/'Inf' literals, comparison functions produce incorrect results (NaN != NaN), and downstream numerical operations propagate NaNs. Fix: add np.isfinite() checks in __post_init__ for data arrays, or in the write path before emitting. This is distinct from NaN sentinel handling (where NaN IS a valid missing-data marker) — those should be explicitly handled with a dedicated null_value field. Checklist: (a) for every data container with numpy array fields, verify NaN/Inf is either explicitly handled (null_value sentinel) or rejected, (b) test with np.array([1.0, np.nan, 3.0]) and np.array([1.0, np.inf]) — both should raise or warn, not pass silently, (c) the check should cover both numeric data (float arrays) and string data (should not contain NaN objects). In pylasdev-reborn: models.py had zero NaN/Inf validation for array data — F-20 (MEDIUM).

## [pat-20260721180548-719513]
Category: pattern
Tags: python, pylasdev, dataclass, validation, type-guard, post-init
Changed: 2026-07-21T18:05:48.131677

Complete type validation of ALL fields in __post_init__: Dataclass __post_init__ should validate the type of every field that has a public setter surface — not just the ones that have caused past bugs. When ParameterEntry has fields unit: str, value: str, description: str, and only some of them have isinstance(value, str) checks, the unchecked fields are open bypass vectors. A non-str value assigned to unit (e.g., int 42) passes __post_init__ silently and surfaces as a crash in writer when str methods are called. Fix: mechanically enumerate all fields in __post_init__ and verify each has a type check matching its annotation. This is mechanical — not every field needs a guard, but every unguarded field is an intentional decision that should be documented with a comment. Checklist: (a) for every dataclass with a __post_init__, list all fields and verify each either has an isinstance check or has a documented reason why it's unnecessary, (b) particularly high-risk: fields annotated as str that consumers call .upper()/.lower()/.strip() on — None or non-str values crash on string method calls, (c) fields that are Optional[str] need both 'is not None' AND 'isinstance(value, str)' guards. In pylasdev-reborn: ParameterEntry.__post_init__ had no isinstance checks for unit, value, description — F-21 (MEDIUM).

## [pat-20260721180552-fd9182]
Category: pattern
Tags: python, pylasdev, sanitization, security, control-characters
Changed: 2026-07-21T18:05:52.852767

Sanitization functions should strip control characters: When a sanitization function (_safe_str, _sanitize_value) prepares strings for output or validation, it should strip or reject control characters (\x00-\x1F, \x7F DEL). A sanitization function that passes control characters unchanged creates a defense-in-depth gap — the output layer (writer) may strip them, but any consumer that reads directly from the dataclass fields (comparison functions, test assertions, API consumers) gets raw control characters. Writer-side sanitization is a mitigation, not a complete defense. Fix: add control character stripping or rejection in the lowest-level sanitization function, so ALL consumers are protected regardless of output path. Checklist: (a) grep for the sanitization function's regex or character handling, (b) verify control characters (\x00-\x1f, \x7f) are either stripped, replaced, or cause rejection, (c) test with a string containing \x00, \x1b, \x7f — verify they don't pass through. In pylasdev-reborn: _safe_str at models.py:25-53 passed control characters unchanged; writer sanitized on output but comparison functions got raw chars — F-22 (MEDIUM).

## [pat-20260721180558-02238a]
Category: pattern
Tags: python, pylasdev, dataclass, validation, nested-model, hierarchy
Changed: 2026-07-21T18:05:58.291881

Validation completeness across nested model hierarchy: When a parent model contains child models and the child has validation (__post_init__ type/dtype checks), the parent MUST replicate equivalent validation for the same fields at its level. DataSection validates its own data arrays for dtype correctness but LASFile.__post_init__ does NOT validate self.logs and self.string_data — the child's validation runs on child instances but the parent accesses the same data through separate attributes. This creates a gap where direct LASFile construction bypasses DataSection-level validation. Fix: for every child model field that has validation, verify the parent model's __post_init__ applies equivalent checks on the parent-level accessor. Alternatively: always route through child construction so child validation runs, never directly assign to parent-level attributes. Checklist: (a) enumerate every parent-child model relationship where both levels expose overlapping data, (b) verify validation exists at BOTH levels or the parent delegates to child construction, (c) test direct parent construction with invalid data — verify it fails with an error, not passes silently. In pylasdev-reborn: LASFile.__post_init__ had zero numpy dtype validation for self.logs and self.string_data while DataSection.__post_init__ had thorough dtype checks — F-26 (MEDIUM).

## [pat-20260721180602-3fb725]
Category: pattern
Tags: python, pylasdev, csv, delimiter, auto-detection, i18n
Changed: 2026-07-21T18:06:02.844736

Delimiter auto-detection should cover locale-common delimiters: When format auto-detection tries to determine the delimiter (comma vs tab vs other), the set of delimiters tested should include semicolon (;). Comma (,) is the most common CSV delimiter in English-locale files, but semicolon is the CSV delimiter in many European locales (German, French, Italian, Spanish, etc. — where comma is the decimal separator). Auto-detection that checks comma and tab but not semicolon silently misparses European CSV files as single-column. Fix: add semicolon to the delimiter detection logic as the third candidate after comma and tab. Checklist: (a) enumerate all delimiter auto-detection paths, (b) verify semicolon is tested as a candidate, (c) test with semicolon-delimited files from European locales — verify correct multi-column parsing. In pylasdev-reborn: dev_reader.py delimiter auto-detection checked comma and tab only — semicolons silently misparsed as single-column. F-30 (MEDIUM, both-found — reported independently by both primary and second opinion agents).

## [pat-20260721180610-02f26b]
Category: pattern
Tags: python, pylasdev, validation, columns, silent-failure
Changed: 2026-07-21T18:06:10.736888

Validation must warn when expected columns are absent: When a data validation function checks specific named columns (e.g., 'MD' for measured depth), the function should emit a warning or error when the column is entirely absent from the data. Silent success with zero warnings when the primary index column is missing creates false confidence — the data passes validation but is structurally invalid. All validation blocks gated on 'if "MD" in columns' skip silently when MD is absent. Fix: before the gated checks, add an explicit check: if 'MD' not in columns: log warning or raise — before any other validation runs. Checklist: (a) for every validation function with column-name-gated checks, verify an absent-column guard fires BEFORE the gate, (b) the guard should be an explicit check, not a default behavior of the gate (gating silently on absent columns is the bug), (c) test with data that has no index column — verify a warning or error is produced. In pylasdev-reborn: dev_reader.py _validate_dev_data had all validation blocks gated on '"MD" in dev.columns' — zero warnings when MD column was missing entirely. F-33 (MEDIUM).

## [got-20260721180616-43ced2]
Category: gotcha
Tags: python, pylasdev, validation, numeric, monotonic, ranges
Changed: 2026-07-21T18:06:16.329499

Monotonicity checks must validate both direction AND value range — checking diffs alone is insufficient: When validating that a sequence is monotonically increasing (ascending depth values), checking diffs < 0 only catches decreases. An ascending sequence of negative values (e.g., MD = [-100, -50, 0]) has diffs that are all positive — the decreasing check passes, but the values are in a physically invalid range. This is a semantic gap: the check verifies consistency (no decreases) but not correctness (values are in valid range). Fix: add a value range check alongside the monotonicity check — verify values are in the expected domain (e.g., MD >= 0 for measured depth from surface). Alternatively: check that diffs > 0 (strictly increasing) AND first value is in valid range. Checklist: (a) for every monotonicity/ordering check, verify there is an accompanying value range/direction check, (b) monotonicity alone ('diffs < 0' for ascending) does not distinguish between valid ascending positive values and invalid ascending negative values, (c) test with entirely negative increasing sequence — verify it fails validation. In pylasdev-reborn: dev_reader.py monotonicity check used 'diffs < 0' only — negative ascending MD values passed silently. F-34 (MEDIUM).

## [pat-20260721180622-4f55ad]
Category: pattern
Tags: python, pylasdev, testing, regression, coverage, falsifiability
Changed: 2026-10-04T20:19:18.636256

Test quality (falsifiability): every test - especially a bugfix regression test - must FAIL on
pre-fix code and PASS on post-fix code. A test that passes on both is vacuous and provides zero
protection. Failure modes found across runs:
(a) weak assertions - assert len>0 / assert key-in-dict pass identically pre/post-fix; assert the
    SPECIFIC expected value, or the absence of the specific problematic character;
(b) default-value no-op - testing a non-default path with the default input (wrap='NO') bypasses all
    mutation branches, so the save/restore under test never fires; use a value that triggers the branch;
(c) wrong input shape/type - H-04 ReDoS test used many lines x one brace (linear) instead of many
    unclosed braces on ONE line (the quadratic trigger); Python list instead of ndarray (tolerance
    asymmetry);
(d) branch routing - the test routed around the changed branch (pipe-qualified ~LOG_DATA avoided the
    bare-~LOG_DATA branch); wrong assertion target (asserted section data while the bug corrupted
    top-level logs);
(e) false contract - asserted a recombination the code cannot perform (zero separators);
(f) string-filter mismatch - a warnings filter string that does not match actual emitted text captures
    zero warnings and passes silently; add assert len(filtered) > 0;
(g) guard/boundary coverage - every resource guard needs a test AND both sides of the boundary
    (at-limit passes, limit+1 fails); regex optional components, null-value paths, and sanitizers each
    need dedicated tests;
(h) content verification - roundtrip tests asserting only structure (counts/types/headers) can pass
    with corrupted data; include at least one content-level assertion.
MECHANICAL GATE: run every new regression test against the PRE-FIX tree (git archive HEAD); if it
passes, it is vacuous - fix input shape/branch/type/target until it fails pre-fix. Behavior-changing
fixes additionally need dual-direction A/B verification (old correct inputs stay correct, target class
flips) - see pat-20260806084853-48f126.
In pylasdev-reborn: F-I2-*, R8-006, S10-004/5, H-04, H-01, M-32, M-76, M-04 - 6+ CONFIRMED.

## [pat-20260721180628-fe7758]
Category: pattern
Tags: python, pylasdev, validation, deduplication, layered-construction
Changed: 2026-07-21T18:06:28.849529

Deduplicate warnings across layered construction: When an object is constructed through multiple validation layers (pre-validation like _validate_from_dict_input + construction-time __post_init__), the same validation logic executed at both layers produces duplicate warnings for the same issue. A VERS format warning emitted by _validate_from_dict_input AND again by VersionSection.__post_init__ creates 2 warnings for 1 issue — misleading the user into thinking there are 2 problems. Fix: add a context flag (e.g., _from_dict=True) that inner validation layers check before emitting warnings — suppress non-critical warnings when the flag indicates pre-validation already ran. Critical errors (ValueError, TypeError raises) should still fire at both layers. Alternatively: remove the duplicate validation from the pre-validation layer and let __post_init__ be the single source of truth — but only if __post_init__ runs AFTER all fields are populated (see deferred-population pattern). Checklist: (a) enumerate all validation call sites in the object construction path, (b) for each pair of overlapping checks, verify only one emits the warning, (c) the context flag pattern (_from_dict, _skip_warnings) should be checked before every warnings.warn() call in inner models. In pylasdev-reborn: s8 found M-04 (duplicate VERS/DLM warnings from _validate_from_dict_input + VersionSection.__post_init__ — up to 4 warnings for 2 validation issues) and M-05 (DataSection NaN/Inf warning fires unconditionally during from_dict(), lacks _from_dict guard that LASFile equivalent has). Both MEDIUM.

## [got-20260721180635-1974fb]
Category: gotcha
Tags: python, pylasdev, mutation, derived-values, staleness
Changed: 2026-07-21T18:06:35.454570

Recompute derived values after state mutation — stale pre-mutation values used after mutation: When a boolean, limit, or threshold is computed from model state BEFORE the state is mutated, and then used AFTER mutation, the derived value is stale — it reflects pre-mutation state while the actual state has changed. In writer.py, check_line_limit was computed from las_file.version.wrap before the WRAP was mutated to 'NO' for LAS 1.2 output. After mutation, the line-limit check evaluated the stale pre-mutation value — the 256-char line-limit warning was silently skipped for LAS 2.0 WRAP=YES files despite the actual output being WRAP=NO. Fix: recompute derived values AFTER all mutations, or compute them from the actual value (not from a saved pre-mutation copy). If the value is computed in a function that receives both pre- and post-mutation state, the function should use the post-mutation state for its logic. Checklist: (a) when a function mutates model state mid-execution, grep for any variables computed before the mutation that are used after it, (b) verify each such variable is either recomputed after mutation or derived from the mutated state, (c) prefer computing derived values inline at point of use rather than caching them early. In pylasdev-reborn: s8 found M-06 — check_line_limit computed from pre-mutation _wrap at writer.py:887-888, used after WRAP mutation at line 932. LAS 2.0 WRAP=YES output had WRAP=NO on disk but line-limit was checked against pre-mutation YES. MEDIUM.

## [got-20260722035622-e9feab]
Category: gotcha
Tags: python, encoding, utf-16, bom, endianness, cross-platform, pylasdev
Changed: 2026-07-22T03:56:22.188806

BOM-stripped UTF-16/32 decode is endianness-dependent: decoding UTF-16/32 with BOM only after stripping the BOM produces silent garbage on BE files read on LE systems (and vice versa). A BOM-stripped UTF-16LE stream on a system where the default endianness is BE decodes as garbage with no error — all code unit values are byte-swapped. The BOM exists specifically to signal endianness; stripping it before decode removes the only endianness indicator. Fix: pass the BOM-detected encoding with explicit endianness suffix (UTF-16LE/UTF-16BE) to the codec, not the generic UTF-16/32 codec. Alternatively: decode with BOM first via codecs.decode(obj, encoding, errors) which consumes the BOM and picks correct endianness. Checklist: (a) grep for .decode('utf-16') or .decode('utf-32') — each is potentially endianness-dependent, (b) verify the byte source signals endianness (BOM, MIME charset, protocol header), (c) on non-BOM sources, explicitly specify endianness: 'utf-16-le' or 'utf-16-be', (d) test with sample data from both endiannesses. In pylasdev-reborn: encoding.py:378-380 stripped BOM before decode, then used generic 'utf-16' codec — BE files on LE systems decode to garbage. 1 confirmed HIGH finding (F-04).

## [got-20260722035628-b50c61]
Category: gotcha
Tags: python, unicode, whitespace, sanitization, asymmetry, control-characters, pylasdev
Changed: 2026-10-04T20:19:18.636258

Control-char / Unicode-whitespace sanitization must be symmetric between reader and writer, and
must DELETE vs REPLACE the right characters:
- Regexes targeting ASCII control chars ([\x00-\x08\x0b\x0c\x0e-\x1f\x7f]) as 'characters to strip'
  silently DELETE Unicode whitespace outside that range - NBSP (\u00A0), en/em/hair spaces
  (\u2000-\u200A), narrow NBSP (\u202F), math space (\u205F), ideographic space (\u3000) - concatenating
  adjacent tokens ('ABC\u00A0DEF' -> 'ABCDEF') (F-05 HIGH). Separate 'delete' (true control chars) from
  'replace with space' (Unicode whitespace).
- Reader/writer asymmetry: characters the writer strips/escapes must be handled by the reader and vice
  versa. writer._CONTROL_CHARS_RE strips \x00 + 26 control chars and Unicode spaces, but
  reader._SPLITLINES_CHARS_RE did not strip \x00; a Unicode NBSP from pasted content can create phantom
  section headers. Enumerate the writer's character classes and verify the reader handles the SAME
  class; a single asymmetric character is an injection/corruption vector.
In pylasdev-reborn: _writer_base.py, parser.py, reader.py, dev_reader.py; F-05 HIGH, F-I2-M04/M05.

## [got-20260722035635-f9576c]
Category: gotcha
Tags: python, concurrency, module-level, shared-state, thread-safety, pylasdev
Changed: 2026-07-22T03:56:35.413290

Module-level mutable flags as per-instance configuration: A module-level variable used as a configuration toggle (e.g., _DESANITIZE_ENABLED) creates a race condition when multiple callers set different values concurrently. Two threads both calling read_las_file() with different desanitize parameters will race on the single module-level flag — the second caller's value overwrites the first's before the first finishes, causing the first to process data with the wrong sanitization setting. This is invisible in single-threaded tests because only one caller exists at a time, but breaks under any concurrent use (web server, thread pool). Fix: thread the configuration through function parameters or per-instance state instead of module-level globals. Preference order: (1) pass as parameter to every function that needs it, (2) store on a per-parser-instance attribute (self._desanitize) not on the module, (3) if a module-level constant is truly needed, use threading.local() or contextvars.ContextVar. Checklist: (a) grep for module-level variables set inside functions (assignment in function body to global/module-level name), (b) verify each is either read-only constant or thread-safe, (c) for any variable that changes per-call, trace all concurrent call paths — if two threads can set different values, it's a race. In pylasdev-reborn: parser.py _DESANITIZE_ENABLED was set to True/False per read_las_file call at line 501, but all parsing functions read it from module level — concurrent callers with different desanitize values race. 1 confirmed MEDIUM finding (F-21).

## [got-20260722035642-bcb034]
Category: gotcha
Tags: python, version, dispatch, compatibility, order-dependent, las-spec, pylasdev
Changed: 2026-10-04T20:19:18.636260

Version-aware dispatch/config:
(a) never hardcode a single version's constants when a version-aware structure exists - the parser
    hardcoded the 8 LAS 1.2 mandatory well fields for ALL versions while _LASVersionSpec
    .mandatory_well_fields held correct per-version data; LAS 2.0 files got spurious 1.2 warnings and
    3.0 files satisfied the wrong check. Grep for hardcoded references and redirect through the
    version-aware lookup (F-25).
(b) never use a catch-all else/default branch to route unknown/future versions to a specific version's
    handler - reject with a clear error or warn; unknown versions silently get wrong-version semantics
    (F-020).
(c) version-gated parse classification is ORDER-DEPENDENT: sections parsed before ~V get
    default-version rules, so the same file parses differently by section order - defer/re-run
    classification when ~V arrives after the section (full treatment: pat-20260806084847-299ec6;
    extends the pre-~V replay invariant got-20260718050203-fc830d).
Checklist: grep every is_las30/is_las12-gated classification at parse time and verify it is deferred
when ~V is late; order-invariance tests parse the same file with ~V first and last.

## [got-20260722035655-a02f76]
Category: gotcha
Tags: python, ieee754, float, formatting, sign-loss, numerical, pylasdev
Changed: 2026-07-22T03:56:55.973232

IEEE 754 negative zero (-0.0) sign silently lost in numeric formatting: Python's int(-0.0) == 0 is True, and subsequent formatting via format(0, '.8g') produces '0' instead of '-0'. The -0.0 sign is semantically meaningful in well-log data (e.g., TVD above reference datum, directional survey sign conventions) but is silently lost when numeric values pass through integer conversion or float formatting. Python's native format(-0.0, '.8g') correctly produces '-0', but any intermediate int() or round() call destroys the sign before formatting. Fix: (a) avoid int() on float values destined for formatting — use the float directly, (b) use math.copysign or explicit sign check: '-' if math.copysign(1.0, value) < 0 else '' before formatting the magnitude, (c) after formatting, verify negative zero preservation: assert format(-0.0, fmt) in ('-0', '-0.0'), (d) numpy float scalars (np.float32, np.float64) have the same behavior — np.int64(np.float64(-0.0)) == 0. Checklist: (a) grep for int(float_val) in numeric formatting functions, (b) verify -0.0 roundtrip: value → format → parse returns sign-preserving value, (c) test with -0.0 and np.float64(-0.0) explicitly. In pylasdev-reborn: _writer_base.py _format_number used int(-0.0)==0 early-exit → '0' instead of '-0'. 1 confirmed MEDIUM finding (I2F-19).

## [got-20260722035702-f6167d]
Category: gotcha
Tags: python, writer, silent-mutation, data-integrity, null-value, padding, pylasdev
Changed: 2026-07-22T03:57:02.695345

Writer silent null-value padding for uncovered curves: When a writer encounters a DataSection where curves_order specifies curves that are absent from the data dict, silently padding the missing columns with null_value conceals data truncation from the user. The output appears valid (all curves present, correct row count) but some columns are entirely fabricated from the null_value. This is a data-loss scenario masked as a valid output — the user receives a file that passes all structural validation but contains zero real data for the missing curves. The legacy writer path warns about uncovered curves; the LAS 3.0 writer path is silent. Fix: (a) before writing, verify every mnemonic in curves_order has a corresponding key in data or string_data, (b) for missing curves, emit a warning (not just debug log) listing the missing mnemonics, (c) consider raising an error for the case where ALL data for a curve is missing (distinct from partial nulls within a present column). Checklist: (a) grep for null_value usage in writer output loops — each use site should be paired with a warning when the curve column is entirely absent, (b) test with DataSection(curves_order=['A','B','C'], data={'A':[...], 'B':[...]}) — verify 'C' absence triggers a warning, not silent padding. In pylasdev-reborn: _writer_las30.py silently padded missing curves with null_value; legacy writer at _writer_base.py:564-571 correctly warned. 1 confirmed MEDIUM finding (I2F-20).

## [got-20260722035708-7f552a]
Category: gotcha
Tags: python, validation, asymmetric, limit, enforcement, pylasdev
Changed: 2026-07-22T03:57:08.115629

Asymmetric limit enforcement — validation guard applied in one code path but not in a sibling path: When a validation limit (e.g., 256-char line length for LAS 1.2) is enforced in one output path (data rows) but silently skipped in another (header section lines), the invariant is partially enforced — headers can exceed the limit while data rows cannot. The user sees data-row truncation but header truncation is silently absent, creating a format-compliance gap. Fix: (a) identify ALL code paths that produce output for the same format/version, (b) apply the same limit enforcement in every path regardless of output type (header, data, comments), (c) if different limits apply to different output types, document why explicitly and enforce the per-type limit. Never silently skip a limit in one path that's enforced in another — the asymmetry IS the bug. Checklist: (a) grep for every format limit check, (b) enumerate all output-producing code paths for that format, (c) verify the limit is checked in every path or explicitly documented as intentionally not enforced. In pylasdev-reborn: _writer_base.py:300 enforced 256-char line limit for data rows but header section lines had zero enforcement. 1 confirmed MEDIUM finding (F-34).

## [got-20260723180816-b4fbc3]
Category: gotcha
Tags: python, pylasdev, auto-detection, parameters, asymmetric, validation
Changed: 2026-07-23T18:08:16.533020

Auto-detection validation bypassed by explicit parameters: When auto-detection logic (e.g., delimiter inference from file content) performs data-quality checks or corrections, code paths that accept explicit parameters skip those checks — the user who specifies a value gets silently worse behavior than the user who lets the library auto-detect. Fix: run the same validation/correction logic regardless of whether the parameter was auto-detected or explicitly provided. Checklist: (a) for every function with auto-detected defaults, grep for validation/correction that runs only in the auto-detect branch, (b) extract the validation into a shared function called by both paths. In pylasdev-reborn: Delimiter auto-correction and cross-validation only ran when delimiter=None; explicit wrong delimiter produced silent data corruption (F-013).

## [got-20260723180822-2aa2ef]
Category: gotcha
Tags: python, pylasdev, api, parameter, dead-code, validation
Changed: 2026-07-23T18:08:22.164789

Dead parameter in public API: A function parameter that is accepted, documented, and passed by callers but never evaluated internally creates a silent API contract violation — callers believe they're controlling behavior (e.g., validate(complete=True)) but the parameter has zero effect. This is worse than a missing parameter because it gives false confidence. Fix: either implement the parameter's promised behavior or remove it from the API (breaking change with deprecation period). Checklist: (a) for every public API parameter, grep its usage inside the function body, (b) if zero usages found, it's a dead parameter — either wire it up or deprecate it, (c) add a test that verifies the parameter actually changes behavior. In pylasdev-reborn: DataSection.validate() accepted a 'complete' parameter but never checked it — all 5 check groups ran unconditionally (F-032).

## [got-20260723180824-8a15bb]
Category: gotcha
Tags: python, pylasdev, query, search, traversal, nested, collections
Changed: 2026-07-23T18:08:24.865758

Query methods missing nested same-type collections: When query/search methods iterate over a primary collection but skip nested sub-collections of the same type (e.g., querying curves in the top-level list but missing curves inside data_sections[].section_curves), results are silently incomplete. Fix: when a data model supports nested same-type collections (e.g., curves at both file-level and section-level), every query method must traverse all nesting levels. Checklist: (a) identify all nesting levels for each collection type, (b) verify every query method traverses all levels, (c) add a test with data present only in nested collections. In pylasdev-reborn: get_curve_by_mnemonic and get_array_curves did not search data_sections[].section_curves for LAS 3.0 — both-found confidence (F-038).

## [got-20260723180827-71c263]
Category: gotcha
Tags: python, pylasdev, loops, state, conditional, stale, scope
Changed: 2026-07-23T18:08:27.559024

Stale loop variable from skipped conditional branch: When a variable is set inside a conditional within a loop body, and the condition is false on a given iteration, the variable retains its value from the PREVIOUS iteration. Subsequent code using that variable operates on stale data with no indication. Fix: explicitly reset the variable to a known sentinel (e.g., None) at the top of each loop iteration before the conditional, then check for the sentinel before use. Checklist: (a) grep for variable assignments inside if-blocks within loop bodies, (b) for each, check if the variable is used after the if-block on the same iteration, (c) verify no iteration inherits the previous iteration's value. In pylasdev-reborn: LOG_DATA curve scope went stale after typed data section when __MAIN__ not populated — no else clause fallback (F-044).

## [got-20260801115852-c9d67c]
Category: gotcha
Tags: pylasdev, fix-coordination, models
Changed: 2026-08-01T11:58:52.086466

M-11 fix requires editing src/pylasdev/_version_spec.py:91-93 (mandatory_well_fields property) which was NOT in G3 fix agent's writable files list; no fix agent owns _version_spec.py. Exact change: LAS 1.2 tuple to lascheck 10-field set (drop UWI, add COMP/FLD/DATE). Also test_parser.py:1554 test_las12_no_mandatory_field_warning asserts 4 warnings (would become 6) and test_combinatorial.py:38 docstring stale.

## [pat-20260801161338-93f466]
Category: pattern
Tags: python, null-sentinel, data-integrity, roundtrip
Changed: 2026-08-01T16:13:38.423842

Null-sentinel consistency: the declared NULL value and the fill sentinel baked into data cells MUST agree. LAS 3.0 data sections processed before ~Well bake DEFAULT -999.25; if the file later declares NULL=-999 with no cross-check, downstream consumers read fill cells as real data. Case-variant well keys (null vs NULL) make declared NULL != fill sentinel. Fix: cross-check declared NULL vs baked default; reconcile after ~Well known AND at end-of-parse (single trailing-section case — reconcile only firing on a LATER data call misses it, EXT-02); case-insensitive _get_null_value lookup / well-key normalization; distinguish absent-from empty-string NULL. In pylasdev-reborn: IT3-THR-01, N-I-31, EXT-02, N-I-09 — 4 CONFIRMED MEDIUM, coordinated fix.

## [pat-20260801161434-05fd52]
Category: pattern
Tags: python, validation, whitelist, parser, roundtrip
Changed: 2026-08-01T16:14:34.604521

Identifiers and units must be validated against a WHITELIST matching the parser grammar, not a punctuation blacklist — blacklists silently miss novel characters. M-03: ALL punctuation (#, |, ;, (, @, $, ,, /, \, +, =, ~, [abc]) silently drops curves, not just colons — whitelist against parser grammar ^\w[\w\-]*(\[\d+\])?$. M-04/N-I-22: unit char class [\w\-/]* rejects %/degC/ohm.m → whole curve + data column silently dropped (HIGH, data irrecoverable). M-27: whitespace section_type misrouted to ~O — unify sentinel, reject spaces AND |. N-I-19: well keys with dots/spaces/colons ENTIRELY DROPPED on re-read. N-I-07: unknown single-char format {X} — parser warns-and-clears vs from_dict RAISES, roundtrip crashes — align gates. Checklist: every identifier field (curve mnemonic, unit, well key, section_type, data_format) needs a whitelist regex matching what the parser accepts; validate on BOTH model layer and parser layer (CurveDefinition AND ParameterEntry). In pylasdev-reborn: M-03, M-04, N-I-22(HIGH), M-27, N-I-19, N-I-07 — 6 CONFIRMED.

## [pat-20260801161434-5a047d]
Category: pattern
Tags: python, roundtrip, writer, parser, symmetry, format, pylasdev
Changed: 2026-10-04T20:19:18.636267

Roundtrip writer/parser token symmetry: every token the writer emits must have a parser path that
consumes it with the same meaning, and vice versa - otherwise every roundtrip is silently lossy. Cases:
the writer emits CSV quoting that the parser strips; the writer emits a :offset suffix for all
data_formats but the parser processes offset only for 'A'; the writer appends {F} at END while the
parser takes the FIRST format-spec match (user text {F} stripped from a description); multi-char
data_format divergence (parser clears, from_dict truncates to first char, writer emits unbraced -> one
duplication on roundtrip); array bracket mnemonics dropped without them -> curve reclassified to
string_data (double-bracketing caution); version-specific patterns run unconditionally with no version
gate while the writer never re-emits them (text lost).
Checklist: for every token the writer emits, verify a parser path consumes it and vice versa;
version-gate version-specific patterns; bracket mnemonics must align data dict keys/curves_order. In
pylasdev-reborn: F-004, F-016, N-I-18, M-23/N-I-21, W-08/W-09, N-I-02 - 6 CONFIRMED.

## [pat-20260801161434-098911]
Category: pattern
Tags: python, container-guard, mutation, writer, validation, pickle, pylasdev
Changed: 2026-10-04T20:19:18.636269

Guard wrappers around mutable containers must be re-established after write and cover ALL
mutation entry points.
(a) Writer finally blocks that strip _GuardedDict/_GuardedList permanently disable guards after any
    write (W-06) - re-wrap is required; failure-path wrap leaks need a success-flag restore (W-07).
(b) _GuardedDict/_DevColumns must override update/pop/setdefault/clear - C-level dict methods bypass
    __setitem__/__delitem__ (verified empirically) -> columns/column_order desync.
(c) A class wrapping a mutable container and exposing __setitem__/public attributes must enforce
    type/value constraints at EVERY mutation entry point, not only at validate() time (non-str keys or
    values, plain dicts/lists exposed with zero guards).
(d) Whole-container reconcile (trim/grow ALL arrays to a common length) must NOT trip per-key length
    guards - it needs an explicit bypass; the invariant (equal length) and the guard (per-key) must be
    separated and documented.
(e) Every __slots__ container subclass with __setitem__-time validation breaks pickle (loads raises
    AttributeError: the guard fires before the slot is restored) unless __reduce__/__setstate__ is added
    IN THE SAME change.
Checklist: grep finally blocks that unwrap guarded containers (re-wrap); override ALL C-level dict
methods; guard every construction path (__init__, slicing); add a pickle roundtrip test (non-empty AND
empty) verifying guards still raise post-unpickle. In pylasdev-reborn: models.py; W-06, W-07, N-I-14,
M-01/M-02, PF-09 - 5 CONFIRMED.

## [pat-20260801161434-337672]
Category: pattern
Tags: python, resource-limit, validation, parser, gating
Changed: 2026-08-01T16:14:34.968485

Resource limits must be validated against the ACTUAL input being processed, and completeness checks must not be gated on collection presence. N-I-01: parse() missing-~V validation gated on content.strip() even when lines= parsed — gate on the actual parsed source. N-I-13: array-continuity validation gated `if self.data_sections:` → top-level interleaved arrays pass validation, writer emits SELF-UNREADABLE output (library's own parser raises LASParseError). P-03: F-040 increment must widen to EVERY pre-~V data entry; merged sections must not bypass MAX_DATA_SECTIONS. Checklist: validation gates must inspect the real input shape/source, not a sibling attribute; merged/aggregated records must still hit resource counters; validate top-level state even when the collection is empty. In pylasdev-reborn: N-I-01, N-I-13, P-03 — 3 CONFIRMED (P-03 HIGH).

## [pat-20260801161435-69ee23]
Category: pattern
Tags: python, integer-precision, data-integrity, roundtrip, numerical
Changed: 2026-08-01T16:14:35.059704

Integer-format ({I}) curves lose precision when parsed via float(): values beyond 2^53 (9007199254740993 → ...992.0) silently corrupt. L-03: {I} stored float64 at data_reader.py:560-562/1027 + _las30_data.py:488 + models.py:2152 — fix TRAP: precision loss happens at float() conversion BEFORE dtype branch, and int64 allocation truncates fractional null -999.25→-999 — parse via int() and handle non-integral NULL. EXT-04: int64 branch only engages when declared NULL is integral; default fractional NULL -999.25 → float64 path persists (defensible tradeoff, must be documented + tested >2^53). EXT-09: from_dict int64 coercion silently corrupts non-integral values (NaN→0, 1.5→1) — reader is safe, from_dict isn't. Checklist: parse {I} with int(); dtype branch in ALL allocation sites; handle non-integral NULL; test values >2^53 AND fractional-NULL path. In pylasdev-reborn: L-03, EXT-04, EXT-09 — 3 CONFIRMED.

## [pat-20260801161435-e8a493]
Category: pattern
Tags: python, writer, null, string-data, roundtrip
Changed: 2026-08-01T16:14:35.151047

String-branch null/NaN handling must match numeric-branch sentinel routing, and string data written to formats that cannot represent it must warn. N-I-17: string curve None/NaN written as literal 'None'/'nan' (string branch has no null guard; numeric branch sends to sentinel) → re-read fabricates literal values. M-29: non-LAS-3.0 string_data write → null sentinel on re-read — needs reader-side all-non-numeric column detection OR explicit writer warning (no {S} in 1.2/2.0). Checklist: every writer branch (string/numeric) must route null/NaN to the sentinel; when a format version lacks a data type, detect or warn — never silently convert. In pylasdev-reborn: N-I-17, M-29 — 2 CONFIRMED.

## [pat-20260801161435-3b6fe9]
Category: pattern
Tags: python, writer, diagnostics, warning, consistency
Changed: 2026-08-01T16:14:35.241618

Writer diagnostics and output structure must match reality across versions and paths. L-01: LAS 3.0 diagnostics use logger.warning while LAS 2.0 uses warnings.warn — users cannot intercept 3.0 warnings; unify to warnings.warn (also _fc summary + NULL warning). W-04: false warning text 'Single-section data will be preserved' when drop is intended — conditional on actual copy-back outcome. W-05: empty ~CURVE emitted when top-level curves empty + single data_sections — copy-back must run BEFORE ~C emission. W-02: bare precision .5/.8 → format(int(v)) ValueError → LASWriteError — require trailing code letter or normalize; crash fires exactly when real data exists. Checklist: grep all warning mechanisms in sibling paths; warning text must match actual behavior; emission order must precede dependent sections; validate format specifiers at write time. In pylasdev-reborn: L-01, W-04, W-05, W-02 — 4 CONFIRMED.

## [pat-20260801161435-32eca4]
Category: pattern
Tags: python, dev-reader, format-detection, heuristics, locale
Changed: 2026-08-01T16:14:35.333370

DEV reader format-detection heuristics must guard against all-numeric/all-integer misclassification and delimiter/locale ambiguity. V-01/V-03: DUG Pattern B missing all-float guard — I2F-001 fall-through does NOT generalize; validate candidate header against documented DUG column sets. V-02: comma count-prefix misdetection — BOTH comma branches (float 435-446 + integer 405-416) need fallthrough, not just whitespace twin. V-13: headerless all-integer first row consumed as names — fix the F-92 integer heuristic generally. V-04: headerless semicolon first row consumed as names. V-07: comma-decimal locale values → all-NaN, no comma→dot conversion. V-08: thousands separator 1,234.5 silently corrupts in comma mode. V-18: empty MIDDLE header cell → column shift — distinguish trailing empties (drop) from middle empties (reject/pad). Checklist: all-numeric header guard on EVERY detection branch (pattern/type/delimiter); locale-decimal handling when comma is delimiter; empty-token filter position-aware. In pylasdev-reborn: V-01, V-02, V-03, V-13, V-04, V-07, V-08, V-18 — 8 CONFIRMED (2 HIGH).

## [pat-20260801161435-cb4e99]
Category: pattern
Tags: python, deepcopy, from_dict, construction, data-integrity
Changed: 2026-08-01T16:14:35.423190

Deepcopy caller input on direct construction paths — not just from_dict. N-I-11: direct LASFile(logs=)/DevFile(columns=) MUTATES caller dict (list→ndarray coercion, no deepcopy, contrast from_dict 2360) + aliases array storage; DevFile list path raw TypeError crash. M-28: from_dict F-011 must SUBTRACT string_data keys from MISSING-side key set (sibling pattern at models.py:1772-1786 proves correct pattern was missed; producer at data_reader.py:544-563 is test-documented intent — fix the consumer). Checklist: every constructor path (direct, from_dict, parser) must deepcopy mutable caller input; missing-keys validation must account for keys routed to alternate storage (string_data vs data). In pylasdev-reborn: N-I-11, M-28 — 2 CONFIRMED (M-28 HIGH).

## [pat-20260801161518-ecfb93]
Category: pattern
Tags: python, writer, dedup, per-section, roundtrip, parser
Changed: 2026-08-01T16:15:18.966679

Writer-side cross-section curve dedup must be per-section scoped, dedup keys must include distinguishing attributes, and the fix must land on BOTH writer and parser. W-01 (HIGH): duplicate curve emission — dedup alone INSUFFICIENT: per-section curve scoping required; dedup-by-mnemonic silently drops differing definitions (DEPT.M vs DEPT.FT); root cause spans writer AND parser (parser registers curves from top-level ~C AND each ~_Definition — writer-only dedup still inflates on re-read, EXT-03 confirmed 3→7 curve inflation). L-02: non-first-section dedup never writes back global curves (inner writeback blocks are dead code — only call site passes False). N-I-20: emit ~{name} | {section_target} per LOG_DATA section, not hardcoded | CURVE. Checklist: dedup key = mnemonic+unit+format; dedup during first extension + warn on unit/format mismatch; parser-side registration dedup; per-section scope preserved through merge (P-03). In pylasdev-reborn: W-01(HIGH), L-02, N-I-20, EXT-03 — 4 CONFIRMED (co-located G8/G4b fix family).

## [pat-20260801161519-2c365e]
Category: pattern
Tags: python, performance, hot-path, memory, benchmark, dos, pylasdev
Changed: 2026-10-04T20:19:18.636273

Per-value hot-path costs in data reading:
(a) hoist thread-local/import-machinery flag lookups out of per-value loops (_desanitize_las_value used
    a thread-local __getattr__ at 1.04-1.07 us/value -> 2.05x end-to-end slowdown; cache the flag once
    per read);
(b) prefer math.isfinite/isnan/isinf over np.* for Python scalars (15.3x faster, semantically
    identical) - keep np.* for array-vectorized checks;
(c) hoist regex/frozenset/compiled objects to module level when constructed per-call in a loop (e.g.
    over MAX_CURVES=100K curves);
(d) pre-allocate numeric columns and cap intermediate accumulation phases - MAX_TOTAL_ELEMENTS counted
    final arrays only, so a Python-float intermediate list bypassed the OOM guard's intent (1.80x
    peak-RSS at 3M values);
(e) bound every str.split() with maxsplit in parsing/detection code - 9 unbounded sites in
    delimiter-detection gave 102.8MB RSS on a 0.58MB file (177x amplification).
Checklist: profile per-value loops; grep thread-local/import lookups in hot loops; math.* for Python
scalar checks, np.* only for arrays; every str.split() bounded. In pylasdev-reborn: IT3-F-01/02/03,
N-I-26 - 4 CONFIRMED (measured).

## [pat-20260802053925-61425d]
Category: pattern
Tags: python, regression, fix-coordination, structural-fix, hotspot
Changed: 2026-08-02T05:39:25.152354

Regressing-function structural hardening: when a function is re-fixed for the same defect class (2+ PRIOR_FIX_ATTEMPT findings in the same ~45-line block), leaf patches are the wrong tool — apply a structural fix satisfying the FULL decision contract, not just the current repro. In pylasdev-reborn _is_mnemonic_header_row (data_reader.py:781-853) was re-fixed 3x in ONE session (F-01, F-03, F-19 then M-01/M-02/M-03): each leaf patch fixed one repro shape while sibling shapes stayed broken or a new regression appeared (M-02 string-row drop at the new wrapped call site). The structural fix that ended the cycle: (a) token-count equality — only a full-width row (len(values) == curve_count) can be a header; (b) all-string exclusion — a section with only string curves is never a header; (c) match set = resolved ∪ original_mnemonics (mnem_base-aware); (d) replace drop-with-warning on pre-scan undercount with geometric grow-and-continue (_allocated doubles; whole-container growth via dict.__setitem__ preserving the equal-length invariant). Process requirements: localized pre-fix audit BEFORE the fix pass + second-opinion reviewer on the fix agent (hotspot rule). Checklist: after a function is fixed twice for one defect class, stop patching leaves; enumerate the full contract; fix the root structure; regression-test EVERY previously-broken input class (M-01/M-02/M-03 repro shapes: mnem_base header, wrapped mixed-section, all-string section).

## [pat-20260802053927-4cb0be]
Category: pattern
Tags: python, regex, performance, dos-protection, family, pre-fix-audit
Changed: 2026-08-02T05:39:27.751449

Regex ReDoS families require whole-family hardening: catastrophic-backtracking regexes come in FAMILIES within one module — fixing one pattern's ReDoS leaves sibling patterns with the same backtracking structure quadratic. In pylasdev-reborn, commit 3f18608 fixed DATA_LINE_PATTERN's O(n^3) and added _SAFE_REGEX_LINE_LENGTH but left FORMAT_SPEC_PATTERN (H-04: 22-44s per line at 48KB, 712x amplification, ~61-122h CPU per 500MB file), VALUE_ONLY fallback (M-60), and unescape-colons re.sub (M-61) quadratic — asymmetric-fix regression verified via git show (3f18608 never touched FORMAT_SPEC_PATTERN). The fix required a localized pre-fix audit of the whole regex family + unified hardening (bounded matcher / fast-path scan / line-length guard), landing on all siblings together. Checklist: when fixing a regex ReDoS, grep the module for ALL sibling regexes with the same backtracking structure (optional quantifiers, nested repetition, alternation over long runs); run a pre-fix audit covering every sibling before fixing; fix the family together or add a shared guard; regression-test each pattern with adversarial inputs proving linear time. This is the regex-specific instance of the parallel-code-path omission pattern (pat-20260720071838-171ab7).

## [pat-20260802053933-1c5213]
Category: pattern
Tags: workflow, fix-coordination, edit-density, dispatch, process
Changed: 2026-08-02T05:39:33.391567

Fix-dispatch completeness: when splitting confirmed findings across fix agents (e.g., by edit-density 8-per-file / 12-per-agent caps), a confirmed finding can be silently DROPPED — never assigned to any fix agent. In pylasdev-reborn, M-76 (multi-thousands-separator corruption) was CONFIRMED and on the fix list but absent from ALL 14 fix task prompts (grep 0 hits) — discovered only by post-fix VERIFY (F-02, HIGH as filed: fix-completeness). The edit-density split is a coordination artifact that can lose findings; F-04 (M-38 las30-side) was a second dispatch gap — the 2.0-side fix was dispatched, the 3.0-side implementation never received it. Fix: after assembling fix-agent task prompts, mechanically verify EVERY confirmed finding ID appears in at least one task prompt (grep the prompt corpus for each ID); a finding with 0 hits is a dispatch gap. Cross-file findings need one hit per file-owner agent. Checklist: build the confirmed-finding ID list; grep every fix task prompt for each ID; every ID must have >=1 hit; cross-file findings >=2 hits; re-check after any scope split.

## [pat-20260802053943-dd5df5]
Category: pattern
Tags: python, container-guard, reconcile, trim, grow, interaction
Changed: 2026-08-02T05:39:43.175885

Whole-container reconcile vs per-key guards: internal whole-container reconciliation (trimming or growing ALL curve arrays to a common length) must NOT trip per-assignment length guards — it needs an explicit bypass path. In pylasdev-reborn, the F-01 fix's F36 whole-container trim (las_file.logs[curve_name] = arr[:current_line]) tripped the M-43 per-key _check_value_length guard -> spurious crash (HIGH F-01); the FIX-CONV-2 G-04 grow uses dict.__setitem__ whole-container growth to preserve the equal-length invariant, mirroring _GuardedDict.trim_all's bypass (models.py:302-303). Design rule: when a guarded container enforces equal-length via per-key validation, whole-container resize operations (trim/grow/sync-all) bypass per-key guards; per-key mutation stays guarded. The invariant (equal-length) and the guard (per-key) must be separated. Checklist: when adding a per-key length guard, verify internal reconcile paths have a bypass; when adding a reconcile, use the bypass not per-key assignment; document the invariant-vs-guard separation; test both trim (overcount) and grow (undercount) directions.

## [pat-20260802053951-54b7bd]
Category: pattern
Tags: python, encoding, cyrillic, detection, mojibake, false-positive, adjacency, byte-analysis, pylasdev
Changed: 2026-10-04T20:19:18.636276

Encoding detection: decide from RAW BYTES before decoding. Character-based analysis on decoded
samples is unreliable (cp1251 maps Western accented bytes 0xC0-0xFF to Cyrillic code points -> false
Cyrillic). Cyrillic-vs-Western uses THREE independent detectors, each with a distinct failure mode;
adding a detector without testing the others' false-positive classes misroutes realistic files:
- byte-frequency on raw bytes: count bytes for the top-10 Russian letters in cp1251
  {0xE0,0xE2,0xE5,0xE8,0xEB,0xED,0xEE,0xF0,0xF1,0xF2}; genuine Russian text is 25-57%, Western 2-5%.
  The ~10% threshold is a HEURISTIC, not a separator: accent-dense Western cp1252 (n-dense) can reach
  15.7% and is misdecoded as cp1251 (mojibake, zero warnings) (M-57).
- run-length: >=3 consecutive accented bytes false-positives on Western cp1252 (e.g. 'Nanez') (M-82).
- numero-sign adjacency: require byte 0xB9 ADJACENT to the Cyrillic run, not whole-file membership -
  a lone Western '1-superscript' (also 0xB9 in cp1252) plus any accented run elsewhere flips the file
  (_NUMERO_ADJACENCY_WINDOW) (F-18). Numero-rich genuine cp1251 can still fail (M-81).
Sampling: first-64K sampling misses Cyrillic beyond 64K -> mojibake; widen BOTH the byte-frequency and
run-length samples (sample the whole file or a size-proportional window). Score by ratio; do NOT switch
the primary sort to preferred-encoding-first without checking the dominant-encoding (UTF-8 Cyrillic)
regression (E-07, E-06). Break ties by domain prevalence (cp1252/latin-1 > cp1251 for Western data);
always prefer chardet when available.
Byte-identical classes (e.g. cp866 Cyrillic vs cp1252 smart-punctuation, both using 0x80-0x9F
symmetrically) cannot be separated by byte content - move the decisive signal to CONTEXT (digit
density, uppercase-mnemonic position) and pin provably-irreducible residuals as strict xfail; adding
another byte rule restarts the fix-regress cycle (see pat-20260803092027-428275).
Checklist: every added detector must be tested against the OTHER detectors' known false-positive classes
(accent-dense Western, numero-dense Cyrillic, multi-consecutive-accent Western); per-char byte
signatures need positional (adjacency) constraints. In pylasdev-reborn: encoding.py; E-06/E-07, M-57,
M-81, M-82, F-18 - 5+ CONFIRMED MEDIUM+.

## [pat-20260802053958-275951]
Category: pattern
Tags: python, writer, parser, frozenset, order, column-swap, scoping
Changed: 2026-08-02T05:39:58.691584

Order-insensitive set comparison for order-sensitive output: using set/frozenset equality for scoping or identity decisions when column ORDER is semantically meaningful silently swaps columns. M-66/M-68 (_writer_las30.py): writer compares frozensets of curve names to decide whether a section's curves match the main block — {GR, DEPT} == {DEPT, GR}, so a section with the SAME curve-name set but DIFFERENT column order roundtrips with GR/DEPT data silently SWAPPED, zero corruption warnings. M-83: same dedup region, mnemonic-only key drops the second section's desc/api_code (W-01 compares unit/format only). The parser-side scope resolution (main_curve_end pipe branch, M-67/M-69) must mirror the writer's per-section scoping — H-01's fix did not cover the pipe branch, giving two failure directions (frozen-at-0 -> whole LOG_DATA discarded; unfrozen -> None -> phantom columns). Fix: when output ordering matters (per-section column order, scoping), compare order-sensitive structures (tuples, sequences) or emit explicit per-section definitions — never frozenset/set equality on ordered data. Checklist: grep frozenset/set comparisons in writer/parser scoping and dedup; verify ordering is preserved; test with same-set-different-order sections; dedup keys must include distinguishing attributes (unit/format/desc) not just the name.

## [got-20260802145747-ca3ff7]
Category: gotcha
Tags: pylasdev, parser, models, mnem_base, pxm, gotcha
Changed: 2026-08-02T14:57:47.477148

pylasdev Parser/Models boundary: mnem_base (incl. shipped MNEM_BASE) is OPT-IN on both read_las_file (default None) and LASFile.from_dict (default None) — default paths apply NO mnemonic normalization, so PXM-01/PXM-06 collision bugs (well last-wins, curve alias-first swap GK/GK_2) only trigger when caller passes mnem_base. from_dict curve-collision failure is order-dependent: [GR,GK] alias-first raises LASDataError (curves_order vs curves mismatch via shared _norm_curve_mnem closure state), canonical-first [GK,GR] passes. Well path has raw==resolved re-key branch (models.py:3306-3316); curve paths lack it. Parser VERS normalizes '1,2'->'2.0' but from_dict keeps verbatim -> write_las_file raises LASWriteError.

## [pat-20260802161321-0c72c1]
Category: pattern
Tags: memory-cap, string, data_reader
Changed: 2026-08-02T16:13:21.471050

DR-05 string-object cap: MAX_TOTAL_ELEMENTS accounts 8B/element but Python str objects cost 50-100B; LAS 1.2/2.0 paths now have MAX_STRING_VALUES = MAX_TOTAL_ELEMENTS // 12 mirroring _las30_data._MAX_STRING_VALUES, enforced at _read_normal store + 3 _read_wrapped append sites with >=cap-raise semantics.

## [dec-20260802161321-285958]
Category: decision
Tags: dlm, comma, string
Changed: 2026-08-02T16:13:21.552543

I2-02 embedded-comma-in-string fix: csv.reader quote-awareness REJECTED (F2-015: writer emits raw delimiter.join(), no CSV quotes; quote parsing breaks writer roundtrips). Chose loud warning at _read_normal extra-columns site + count summary.

## [pat-20260802195933-a62aa6]
Category: pattern
Tags: python, parser, writer, sanitization, escape, roundtrip, symmetry
Changed: 2026-08-02T19:59:33.551940

Sanitize/desanitize scope symmetry: a desanitize (unescape) step must reverse EXACTLY the escape positions the paired writer emits — no more, no less. PF-02 (MEDIUM regression): the s8 fix made _parse_other run EVERY ~O line through blanket _desanitize_las_value, which unescaped '_~' -> '~' (a data-row escape the ~O writer NEVER emits) and mid-line '_#' (the ~O writer escapes ONLY line-start '#'-prefixed content) — genuine '_~weird line' and mid-line '_#literal_under' silently corrupted on write->read. Fix: scoped _desanitize_other_line (parser.py:612-655) reverses only '_#' at position 0 or preceded exclusively by leading whitespace. Fundamental limitation: a genuine line-start '_#literal_under' is byte-identical to a writer-escaped '#literal_under', so line-start '_#' CANNOT be both preserved and restored — document the achievable scope. Checklist: enumerate the writer's ACTUAL escape positions (grep _sanitize_las_value); the unescape must match them exactly; test genuine content that merely RESEMBLES escapes ('_~', mid-line '_#'); roundtrip both writer-produced escapes AND genuine lookalikes.

## [pat-20260802195938-ae151e]
Category: pattern
Tags: python, parser, state-capture, deferral, section-transition, pipe-target
Changed: 2026-08-02T19:59:38.564668

Deferred-state snapshot completeness: when a parser snapshots state for deferred processing (capture/flush), the snapshot MUST include every attribute the deferred flush path reads — a missing field silently misroutes data. PF-01 (MEDIUM, incomplete PARS-06 fix): _CapturedState (_section_transition.py:62-70) had no pipe-target field; capture_current_state() snapshotted BEFORE classification but _process_consecutive_data restored curve indices + data-section type and NOT _current_pipe_target, so on A->A consecutive data sections the first section's forward '| X_Definition' pipe was never recorded in _deferred_pipe_targets and its replay scoped to __MAIN__ (DEPT/GR) instead of the piped _Definition (RHOB) — silent data mislabeling. Fix: add current_pipe_target to _CapturedState, capture it before classification, swap in/out in _process_consecutive_data like the other fields. Checklist: for every deferred-flush attribute the flush function reads, verify the snapshot struct has a field AND the restore path restores it; grep the flush path for all self._* reads and cross-check against the snapshot struct; A->A consecutive-section tests must exercise the full captured-state contract.

## [pat-20260802195943-aafe5a]
Category: pattern
Tags: python, models, metadata, prefix, collision, to_dict, from_dict, roundtrip
Changed: 2026-08-02T19:59:43.866343

Reserved-prefix metadata vs user columns: when a metadata namespace prefix (e.g. _meta_) is reserved, to_dict AND from_dict must BOTH disambiguate by VALUE SHAPE (str/bytes/list[str]/None = metadata; array = user column), not by key presence alone — and BOTH directions must agree. PF-07 (MEDIUM, incomplete MOD-11 fix): to_dict collision check tested only the bare name (models.py:5827-5842) while from_dict tested f'_meta_{key}' in data (:6049), so a user column literally named _meta_source_file + bare source_file metadata caused LASDataError on roundtrip, and a DOUBLE collision (columns named both source_file AND _meta_source_file) silently OVERWROTE the _meta_source_file user column with the metadata string (data loss). Fix: extract a metadata-SHAPE predicate (_is_dev_metadata_shaped, models.py:172-210); apply it on BOTH sides; to_dict gains a double-collision guard (preserve both user columns, warn, skip metadata). Checklist: reserved-prefix handling must be shape-based, not name-based; every to_dict emission branch must mirror the from_dict classification branch; test the double-collision case (bare + prefixed user columns) — it must warn, not silently overwrite.

## [pat-20260802195954-b51168]
Category: pattern
Tags: python, writer, writeback, las30, copy-back, roundtrip, to_dict
Changed: 2026-08-02T19:59:54.100880

Copy-back/writeback must transfer ALL distinguishing fields, not just identity fields: when a fix propagates a curve/attribute to a top-level structure (writeback), copying only mnemonic+array_info but NOT the stripped description leaks state on roundtrip. PF-19 (MEDIUM, incomplete L30-01 fix): F2-07 writeback (_las30_data.py:891-905) copied mnemonic/array_info but not the stripped description -> to_dict leaked the {A:0} marker and the no-data_sections write path double-emitted '{A:0} {A:0}'. Fix: extend writeback to copy the stripped description (same source the section curve uses). Checklist: when writeback/copy-back copies a curve, enumerate ALL fields the roundtrip (parse->to_dict, write) depends on — identity (mnemonic/array_info) AND descriptive (description, unit, format) fields; test the no-data_sections write path AND parse->to_dict; the marker/format text must not leak into output twice.

## [pat-20260803092027-428275]
Category: pattern
Tags: encoding, cyrillic, byte-identical, discriminator, convergence, xfail, mirror-symmetric
Changed: 2026-08-03T09:20:27.394866

Byte-identical input classes: byte-content discriminators provably cannot converge — when two input classes differ only in bytes that are symmetric between two codecs/encodings (0x80-0x9F fully symmetric between genuine cp866 Cyrillic and Western cp1252 smart-punctuation: ВДЕ = 0x82 0x84 0x85 = ‚„…), every byte-content rule is necessary-but-not-sufficient for a sibling class and the fix-regress cycle is STRUCTURAL, not accidental. Evidence: ENC-02(b) Western-vs-Cyrillic rescue took a 6-pass convergence loop (F-01/F-09 -> M4 -> ENC-M1 -> ENC-1 -> F-02 -> E-01 -> M-1/M-2), each pass adding one more byte signal that re-broke the sibling class (14-byte set -> 18-letter complement; digit-in-24B -> Western prices/dates; all-caps-follows -> Western headings/acronyms). Resolution: (1) MOVE the decisive signal OFF the byte-identical bytes onto CONTEXT evidence (LAS digit density / uppercase-mnemonic position / preceding-token structure); (2) accept PROVABLY-IRREDUCIBLE residuals as documented xfail pins with strict=True (M-2: '...IMPORTANT' == 'THE WELL ВДЕ FIELD' structurally — no local rule can separate them; the xfail flips to XPASS if anyone changes the trade); (3) do NOT add another byte-content rule (the fix-regress trap). Checklist: before discriminating two classes, verify their bytes are NOT symmetric (enumerate the byte tables); if symmetric, context is the only separation axis; every new discriminator must be tested against the sibling class (not just the class it fixes); terminal state is pinned residuals, not perfect separation. In pylasdev-reborn: encoding.py _is_genuine_word_run/_run_has_las_context, 6 passes, 10+ CONFIRMED MEDIUM findings.

## [pat-20260803092040-932ff9]
Category: pattern
Tags: models, case-insensitive, mnemonic, curves_order, validate, from_dict, writer, roundtrip, data-loss, pylasdev
Changed: 2026-10-04T20:19:18.636283

Case-variant mnemonic / curves_order resolution: exact-case comparisons where resolution
elsewhere is case-insensitive cause silent data loss or false diagnostics - data null-filled / ~A
skipped (F-32), false 'will pad' diagnostics while data IS emitted, and newly-reachable state that
roundtrip rejects (F-04).
Fix: route ALL mnemonic/service-key matching through ONE centralized case-insensitive helper
(_case_key/_mnem_key), applied at EVERY mirrored site in ONE pass: validate() positional + orphan checks,
construction twins, from_dict, DataSection direct construction, writer emission + diagnostics, {S}/{I}
marker membership set-builds, and warning guards.
- {S} marker membership must test the EMITTED mnemonic (_emit_mnem), not curve.mnemonic - a curve with
  original_mnemonic='LLD' emits markerless; upper-case ALL string_mnemonics set-build sites + the
  warning guard.
- {I} integer-dtype preservation lookup exact-case -> silent int64->float64 precision loss; the
  per-section branch ALSO fails when section_curves is absent -> fall back to top-level curve data_format.
- Lookup win-semantics must be consistent: a by_upper dict built LAST-wins vs setdefault FIRST-wins
  silently resolves a duplicate uppercase key to a different CurveDefinition (silent unit corruption
  M->FT, F-31).
- Case-insensitive dedup must never silently discard a duplicate's DISTINCT data on write->re-read -
  refuse loudly or preserve (X-1: deduping 'DEPT'+'dept' then emitting both columns discards dept data
  with a false warning).
- Metadata (~C) and data rows must emit from the SAME live order source (curves_order); a cached curves
  list silently swaps columns after post-construction mutation (PF-22), and every fallback lookup loop
  needs the same normalization (PF-21).
In pylasdev-reborn: models.py, _writer_base.py, _writer_las30.py; F-32, M13, WL-M1, MOD-1/2, F-04,
F-31, N2b-1, N1b-1, X-1 - 5+ CONFIRMED, 3+ consecutive convergence passes. Checklist: grep EVERY
exact-case site in the mnemonic family; a fix at one site leaves the defect at the twins; roundtrip
tests for case-variant, emitted-name, and duplicate-distinct-data states.

## [pat-20260803092055-fdd6de]
Category: pattern
Tags: parser, data_reader, pre-scan, mirror, predicate, header, phantom-row, tokenizer, regression, pylasdev
Changed: 2026-10-04T20:19:18.636285

Pre-scan/reader predicate mirror contract: when a parser's PRE-SCAN (fast forward pass to
count/skip) duplicates the reader's header-skip predicate, the mirror must be a VERBATIM copy -
count-equality + all-string exclusion + order-independent mnemonic collection + the SAME lower bound
(min(2, curve_count), NOT flat len>=2) + first-line-of-section-only (current_line==0) + a dedicated
distinct-curve counter (NEVER len(curve_mnems) - it includes original_mnemonic aliases like LLD->BFV).
Evidence: 3-copy duplication (data_reader._is_mnemonic_header_row, parser._is_standalone_mnemonic_header,
parser _pre_scan inline skip) produced a 5-pass regression family (F-24 phantom row + data shift;
M9/DR-M1 overcount warning; DR-M2 LAS 3.0 phantom row; DR-M3 mid-section false positives; PSR-1
single-curve phantom row; F-01 stale bound). Each fix aligned ONE copy and the mirror drifted again.
Fix: extract ONE shared pure predicate _is_mnemonic_header_tokens(tokens, curve_count, declared,
string_curve_count) consumed by reader, pre-scan, and the LAS 3.0 twin.
The TOKENIZER is part of the contract: DLM-aware splitting misses a space-separated header in a COMMA
file -> phantom row; unify on a superset split ([\s,]+) at ALL sites - but add a regression test for the
converse (a genuine first-row string value containing a space in a COMMA mixed section would be
space-split and skipped as a header) (H-1, II-11).
Related header-row predicates (also shared-predicate contracts): units-row predicates need UNITS-FORM
token evidence (every token a 1-4 char letter abbreviation AND at least one <=2-char token
M/FT/MS/MV/IN/CM/MM/US), not position/looks-like-letters, else a genuine letters-only data row is
dropped (got-20260809134756-d69576); one-shot skip flags need 4 lifecycle sites init/set/gate/reset on
EVERY path (pat-20260806084817-cce2a1).

## [got-20260803092217-289178]
Category: gotcha
Tags: models, validation, gate, data-loss, curves_order, regression, write-read
Changed: 2026-08-03T09:22:17.522696

Guard-widening must not re-open the defect it was widened to avoid: when a validation gate is RELAXED to accept a legitimate state (M14: F-19 gate widened from 'curves_order non-empty' to allow empty top-level curves + populated curves_order + data_sections), the relaxation can re-open the ORIGINAL silent-data-loss defect the gate was built to block (MOD-M1: the widened state constructs+validates silently, then write emits only no-definition warnings and re-read loses ALL data). Fix: when widening a guard, trace the ORIGINAL defect class end-to-end (construct -> validate(complete=True) -> write -> re-read) on the newly-accepted state; the widened gate must verify data_sections[].section_curves ACTUALLY cover curves_order, not just that both are non-empty. Checklist: for every gate relaxation, re-run the original finding's full repro chain on the relaxed state; a state that constructs+validates silently but loses data on write->read is a gate hole, not a feature; coverage verification beats emptiness checks.

## [pat-20260803092241-235682]
Category: pattern
Tags: compare, numpy, int64, uint64, overflow, precision, float64, allclose, regression
Changed: 2026-08-03T09:22:41.136796

int64/uint64 comparison precision and overflow in hand-rolled allclose: compare.py _allclose_symmetric has TWO distinct failure modes (both confirmed): (1) int64 SUBTRACTION overflow — diff/abs in native dtype WRAPS on two's complement: [-2**63] vs [0] compares EQUAL (F-17). (2) int64/uint64->float64 PROMOTION collapses >2^53: [2**53] vs [2**53+1] -> True (F-07, CONFIRMED MEDIUM) — the float64-promotion fix for (1) creates (2); float64 cannot represent integers >2^53 exactly. Both array and list paths affected; MaskedArray path (filled(np.nan)) must also promote. M2 (s8): masked-vs-NaN path divergence — array path unwraps mask (False), list path retains NaN-fill (True) — both paths must agree on mask semantics. Fix: for integer dtypes, compare exactly (e.g. via int64 arithmetic / object dtype / bit-exact comparison) rather than float64 promotion, or document the >2^53 precision limit and test sentinel values; ensure array/list/masked paths use IDENTICAL comparison semantics. Checklist: any comparison computing diff=a-b on integer dtypes must avoid BOTH overflow-wrap AND float64 precision loss; test [-2**63] vs [0], [2**53] vs [2**53+1], [2**63-1] vs [0]; symmetric argument swap; masked-vs-NaN equivalence on all paths.

## [pat-20260803195621-cfa5ed]
Category: pattern
Tags: python, wrap, flowing, accumulation, curve_count, data_reader, las30, regression, pylasdev
Changed: 2026-10-04T20:19:18.636288

Wrap detection (LAS 1.2/2.0 data_reader._detect_actual_wrap AND LAS 3.0
_las30_data._detect_actual_wrap_las30) must be ONE shared gate, curve_count-aware and
ragged-shape-aware; line-shape heuristics misclassify FLOWING layout.
- Two paths drift: a wrap fix must land on BOTH. W-A (HIGH): all-full window + declared WRAP=YES
  misrouted to _read_normal -> silent depth-step loss; _read_wrapped ALSO fails on flowing data, so the
  fix is n_curves-accumulation in the wrapped reader + LAS 3.0 counterpart, NOT a routing flip.
- Corroborate over >=3 lines (single/first-line-only checks are defeated by two consecutive sparse rows
  or a genuinely overfull second line); detection must be curve_count-aware (a blanket >=2-values rule
  breaks 2-curve wrapped files).
- Declared WRAP=YES fall-through required in BOTH paths (M-05, F-04); content-based detection also on
  the 3.0 path (M-07 made the decision on the first data line only).
- Depth-line rule: (curve_count >= 3 and window[1] == 1) OR sum(1 for n in window[1:] if n == 1) >= 2,
  mirrored identically on both paths - must classify the [2,1,2] nc=2 ambiguity AND the [3,1,3] ragged
  non-wrapped nc>=3 shape correctly (F-02).
- A genuine wrapped continuation carries exactly curve_count-1 values, never curve_count (W-2: the
  >=2-one-value-rows arm overrode a correct WRAP=NO header -> column corruption on [3,1,3,1]).
- Under-fill guard must cover mid-file steps, not just trailing EOF (N-I-08); single-curve
  WRAP=YES-before-VERS needs a post-VERS re-check (P-15).
Checklist: every wrap rule arm must be curve_count-aware AND distinguish a single-short-row ragged shape
from genuine wrap; two-path parity verified by comparing gate expressions verbatim; regression-test
[2,1,2] nc=2 AND [3,1,3] nc=3 WRAP=NO on LAS 1.2/2.0 (+ string padding); a detection rewrite must keep
previously-correct input classes passing (EXt-01 was a rewrite regression).

## [got-20260803195638-d9e020]
Category: gotcha
Tags: performance, quadratic, slicing, accumulation, dos, data_reader, dev-reader, pylasdev
Changed: 2026-10-04T20:19:18.636290

Accumulation/recombination loops must be linear - re-slicing a growing pending list
(`pending = pending[curve_count:]` per flush) or re-scanning/pair-restarting is O(n^2) and
DoS-amplifiable on a public API. Evidence: the _read_wrapped accumulation rewrite re-sliced per flushed
step (100K-token line ~6.1s at nc=2; ~80min CPU per 500MB crafted file via read_las_file;
thousands-separator recombination re-scan ran 0.10s -> 27.65s at 16K tokens, ~18min at 100K). Fix: an
index pointer (read_idx) or collections.deque.popleft() -> O(1) per step with a single trim at EOF; a
single linear pass with first-run exact-fit for recombination. MAX_TOKENS_PER_LINE caps per-line cost
but the attacker scales LINE COUNT.
Checklist: grep accumulation loops for `list[k:]` re-assignment inside the loop; benchmark on
>=16K-token rows; O(n) per flush is the contract. In pylasdev-reborn: data_reader.py, dev_reader.py;
X-2, I2-18 CONFIRMED.

## [pat-20260803195651-8956e2]
Category: pattern
Tags: testing, roundtrip, differential, oracle, key-set, harness, parity
Changed: 2026-10-04T20:19:18.636292

Differential/parity and roundtrip test limitations:
(a) parse->write->reparse is SELF-CONSISTENT under deterministic reader bugs - the same buggy reader
    parses both sides, so corrupted data compares equal and the test PASSES. Deterministic roundtrips
    cannot detect reader bugs the reader itself introduces: pair them with an INDEPENDENT oracle (e.g.
    a lasio-differential corpus) plus reader unit tests.
(b) Differential tests comparing two dicts must assert KEY-SET EQUALITY (set(p) == set(q)) BEFORE any
    value comparison - intersection-only comparison makes dropped/added keys invisible (deleting the
    STOP well key or adding BOGUS produced 0 mismatches and the test still passed).
(c) A scenario fixture must actually trigger the claimed code path - a 'mnemonic-header-row' scenario
    containing a plain unwrapped file never exercises _is_mnemonic_header_row.
Checklist: mutation-test the harness by deleting a real key / adding a bogus one - the test must fail.
In pylasdev-reborn: test_lasio_differential.py, test_property_roundtrip.py; X-4, X-5, X-6 CONFIRMED.

## [pat-20260806084817-cce2a1]
Category: pattern
Tags: parser, reader, flag, one-shot, header-skip, units-row, mirror, regression, data-loss
Changed: 2026-08-06T08:48:17.032756

One-shot skip-flag lifecycle: a header/units-skip flag set when a mnemonic header or units row is consumed, then left live, silently drops the next letters-only data row as a header/units row. This run's fix loop found the SAME family 3 consecutive cycles: fix3-P1 added the one-shot reset to the non-deferred LAS 3.0 branch (parser.py:4491-4503), fix4-F1 found the reader twins (_read_normal data_reader.py:1074/:1082-1083, _read_wrapped :1555/:1563-1564) still set-without-reset, fix5-F-01 found the deferred pre-~V replay loop (parser.py:3251-3290) as a third set-without-reset site (local flag, 3 refs, no reset). s10-F1: wrapped path is FULLY SILENT while the normal path warns. Root cause: when a one-shot gate is added to one path, sibling paths with the same lifecycle (set/gate/consume) are neither grepped for resets nor checked for gate parity. Detection: grep every flag for init/set/gate/reset sites — a set-without-reset site is a latent silent drop; the units-row skip must ALSO use ONE shared predicate (is_units_header_row, _data_section_reader.py:604) consumed by parser pre-scan subtraction, 3.0 accumulation, and wrapped reader. Checklist: every one-shot skip flag needs 4 lifecycle sites (init/set/gate/reset); reset must fire when the units row is consumed AND when the first data row is consumed; grep sibling paths (non-deferred/deferred parser, normal/wrapped reader) for the same flag pattern; letters-only-first-row files are the trigger class — test single AND double letters rows on every path. In pylasdev-reborn: P1 (s9), F1 (s10), F-01 (s11) — 3 CONFIRMED MEDIUM, same class, consecutive fix passes; family includes M-13/M-03/M-04 (units row consumed as data on 3 paths) and E-42 (header gate stays open).

## [pat-20260806084823-856b98]
Category: pattern
Tags: models, mutation, setattr, validate, post-construction, writer, invariant, data-loss
Changed: 2026-08-06T08:48:23.474729

Construction-only invariants regress: any invariant enforced only in __init__/__post_init__/construction-time validate() silently dies when the model is mutated post-construction — every mutation path (attribute assignment, container __setitem__/append/del, model-mutating methods) needs its own guard (__setattr__ override) OR validate() must re-check the full invariant, and the WRITER must re-validate before emitting. This run: 25 CONFIRMED in one family — E-04 (VersionSection WRAP/DLM post-construction mutation never re-checked → invalid values written verbatim), E-06 (mnemonic mutation bypasses __setattr__ → curve+data dropped), E-07 (ParameterZone no __setattr__ → zone dropped + text leaks), E-10/M-01/s8-M-06 (array-continuity enforced only at construction; top-level re-check passes no data_formats → validate [] → write 0 warnings → own parser LASParseError), E-12/N-12 (cross-container row-count invariant only at construction → writer SILENTLY null-pads → re-read fabricates -999.25 rows, 0 warnings; E-32 same fabrication via unguarded del las.logs['GR']), M-02 (ParameterEntry.__setattr__ guards only 4 of N fields), M-03 (DataSection.section_type mutation unguarded → whole section misrouted, data lost), M-18 (CurveDefinition bracket-index cross-check missing), N-01 (ArrayElementInfo.index accepts 0; from_dict missing key silently yields 0), N-13 (post-construction logs∩string_data overlap → numeric value silently dropped), E-15/E-16/N-11 (0-d ndarrays crash __post_init__ and write, bypass compare tolerance), E-09/M-31/M-32 (MAX_FIELD_LENGTH gaps on emitted string fields → self-unreadable files), E-29 (from_dict accepts LAS 3.0 'other' with zero validate feedback; 3.0 write drops). Checklist: for every model invariant, grep ALL mutation entry points (plain attr assignment sites, __setattr__/__setitem__/append/del overrides, container C-level methods); a validate() that runs at construction but not after mutation is a hole; the writer must re-validate the full contract before emit (write-side validate(complete=True) catches what construction missed); 0-d arrays need consistent handling at construction, write, and compare. Extends got-20260720104609-a00537 and pat-20260721065823-376e3e.

## [pat-20260806084841-d62554]
Category: pattern
Tags: cache, memoization, invalidation, parser, dedup, staleness, match-set
Changed: 2026-08-06T08:48:41.771527

Memoized match-set/scope caches keyed on mutable namespace state go stale when earlier processing mutates that state — a set-without-invalidate site is a latent stale-read defect that survives sibling fixes. N-18 (CONFIRMED, parser.py:3836-3859): the scope cache '_mnemonic_header_scope_cache' keyed (start, end, len(curves)) has exactly 3 references (init/get/set) and is NEVER cleared; dedup writeback (_las30_data.py:815-839) renames global curves AFTER section 1, so section 2 with the same scope key hits the stale pre-dedup match set → post-dedup header consumed as data → phantom null row + shift. The M-22 fix (rebuild the accumulation-time match set post-dedup, parser.py:3853-3858) does NOT clear this cache — cache invalidation is a separate requirement that survives a sibling fix. Checklist: for every cache/memoized set whose computation reads names/indices that later processing mutates (dedup renames, writeback, re-keying), the cache key must cover ALL inputs OR the cache must be versioned/invalidated on every such mutation; grep for cache sites (init/get/set) and verify a clear/version site exists — 3 refs with no clear is the signature; test with a second section sharing the same scope key after a dedup rename.

## [pat-20260806084847-299ec6]
Category: pattern
Tags: parser, version-gating, order-dependent, pre-V, deferral, classification, is_las30
Changed: 2026-08-06T08:48:47.409465

Version-gated parse classification is order-dependent: when classification of a section (curve data_format extraction, parameter-zone extraction, suffix dispatch) is gated on is_las30/is_las12 AT PARSE TIME, sections parsed BEFORE ~V is known are classified with default-version rules and the same file parses differently by section order. N-15 (parser.py:3225-3234): curve parsing gates FORMAT_SPEC_PATTERN on is_las30 — curves parsed before ~V lose {F}/{E}/{D} → data_format='' while ~V-first files get 'F'; compare_las_dicts False, format lost from metadata on write-back. N-16 (parser.py:3621-3645): same for ZONE_ASSOC_PATTERN → pre-~V parameters get zone=None + raw '| Zone[1]' in description. N-07/N-08 (parser.py:1514-1562): endswith('_DATA')/_DEFINITION dispatch version-agnostic → LAS 1.2/2.0 files silently DISCARD ~{Name}_DATA sections or create PHANTOM curves. E-18 (parser.py:1547-1558): ~O rejection order-dependent → read-OK → write-drops asymmetry. Extends the pre-~V replay invariant (got-20260718050203-fc830d): deferral must cover ALL pre-~V content — data lines AND metadata classification (curve formats, zones, suffix dispatch) — not just data rows. Checklist: grep every is_las30/is_las12-gated classification at parse time and verify it is re-run or deferred when ~V arrives after the section; order-invariance tests must parse the same file with ~V first and last and assert identical models; version-agnostic dispatch of 3.0-only suffixes must be gated on the parsed version.

## [pat-20260806084853-48f126]
Category: pattern
Tags: regression, verification, HEAD-parity, behavior-change, encoding, fix-process, AB-test
Changed: 2026-08-06T08:48:53.474595

Behavior-changing fixes require dual-direction verification (A/B vs HEAD): when a fix changes detection/decoding/escape behavior, it MUST be verified against BOTH pre-fix (HEAD) behavior and post-fix intended behavior — run the OLD repro inputs on the NEW code; regressions introduced by fixes are the dominant fix-loop failure mode and are only caught by executing pre-fix inputs (git archive HEAD A/B), not by the new test suite. This run: s8 post-fix review found M-07 (E-02 carve left 8 cp1252 letters in the strict-Cyrillic byte set → >64K Western files flip to cp1251, proven vs HEAD), M-08 (E-26 №-context requirement broke genuine №-before-word → cp1252 mojibake, proven vs HEAD), H-01 (brace-unescape moved INSIDE the format-matches gate → 3.0 digit-led braces regress to progressive backslash corruption); s9 E1/E2 REJECTED by HEAD parity (identical decode at HEAD → zero behavior delta) and E3 CONFIRMED by a 14-shape A/B table (all 14 flip post-fix, cp1252 at HEAD). Method: for every behavior-changing fix, build an A/B matrix — old inputs that were CORRECT pre-fix must stay correct post-fix, AND the fix's own target class must flip; verify against git archive HEAD, not memory. Checklist: before shipping a fix that changes decoding/detection/escape/warning behavior, execute the pre-fix repro corpus against the new code; encode both directions as pinned tests (see pat-20260721180622-4f55ad for the vacuous-test complement); a behavior-change fix without A/B verification is incomplete — this run's M-07/M-08/E3 are the proof.

## [pat-20260806084858-5ea18b]
Category: pattern
Tags: workflow, fix-coordination, cross-file, hand-off, writable-scope, agent, process
Changed: 2026-08-06T08:48:58.704510

A single logical fix spanning two files (a marker/flag SET in one module and CONSUMED in another) fails repeatedly when split across fix agents — each agent completes its own side and the bridge is dropped; keep the full cross-domain change in ONE agent's writable scope. M-01 (s8): the '_designed_nan' marker — the M-25 fix failed TWICE (s5/s7) because models.py (setter side) and dev_reader.py (consumer side) were owned by different agents and the hand-off was dropped on both sides; resolved only by giving ONE agent both files (models.py:7185/7325 + dev_reader.py:1674-1686/2641) and pinning the end-to-end marker flow (reader constructs DevFile with _designed_nan=True → models.py gate activates → read-path NaN block suppressed). Related: got-20260801115852-c9d67c (fix agent lacked a required file in its writable list). Checklist: when a fix requires a producer-consumer pair across files, assign BOTH files to one agent's WRITABLE FILES; if a split is unavoidable, the post-fix review must verify the marker is set AND consumed (grep both sides) — a set-without-consumer or consumer-without-set is an incomplete fix; review must trace the full cross-file flow, not each side in isolation.

## [pat-20260806084904-440e2d]
Category: pattern
Tags: las30, string_data, classification, dedup, marker, data-loss, roundtrip
Changed: 2026-08-06T08:49:04.653727

LAS 3.0 section classification and dedup must be string_data-aware at EVERY site: classification/dedup logic that treats string_data columns as numeric (or vice versa) silently destroys data — either genuine string values become -999.25 nulls or numeric columns are reclassified as strings and misrouted. This run, 6 CONFIRMED in one family: E-22 (lone {A:0} array channel misrouted to string_data; format mutated A→S on roundtrip), M-06 (NaN/Inf in a ≥2-element spec-form group → 'string evidence' → duplicate STRING curves; gate contradicts the fill loop's NaN null-fill), M-16 (duplicate plain-mnemonic {A} numeric-looking groups reclassified to float arrays — data-type mutation on external files), N-04 (main ~C union-forced {S} reclassifies numeric LOG_DATA column as string on roundtrip; the | CURVE pipe decision ignores the {S}-forcing), M-29 (shared-Definition dedup string_data-blind → genuine string values destroyed to -999.25 nulls, 0 write warnings; re-read raises LASParseError — SELF-UNREADABLE output), N-13 (post-construction exact-case logs∩string_data overlap → numeric value silently dropped; _lookup_data_array prefers string_data). Related: {S} marker membership must test the EMITTED mnemonic (pat-20260803092040-932ff9). Checklist: every set-build/membership/dedup/classification decision on the logs↔string_data boundary must test the string_data key set (CI-normalized AND emitted-name-aware); a dedup or classification that can reclassify a numeric column as string (or destroy string values) must warn loudly or refuse; roundtrip tests must cover marker-less, marker-bearing, case-variant, and shared-definition states.

## [pat-20260809134726-678f6e]
Category: pattern
Tags: parser, from-dict, token-extraction, header, VERS, WRAP, DLM, data-corruption
Changed: 2026-08-09T13:47:26.276028

Colon-free header value extraction must take the FIRST TOKEN at EVERY site: LAS header lines without a colon ('WRAP. YES data wrapped', 'DLM. COMMA comma delimited', '1.2 CWLS LOG ASCII STANDARD') carry the VALUE as the first whitespace-delimited token and description text after it. When a site consumes the raw value (value.upper()) instead of the first token, the description corrupts the parsed value: WRAP='NO' instead of 'YES', DLM='SPACE' instead of 'COMMA' (comma data re-read all-null), VERS misread as 2.0 with spurious mandatory-field warnings + write re-label. The parser's VERS branch had F-151-style first-token extraction; WRAP/DLM branches (parser.py:2574-2628) and from_dict normalization (models.py:5669-5696) did not. Fix: extract the first token at EVERY colon-free header site (VERS/WRAP/DLM/from_dict), not just the one that shipped the original fix. Checklist: for every colon-free header field, grep ALL parse sites (parser branches + from_dict + normalization); the first whitespace-delimited token is the value, the remainder is description; test 'WRAP. YES data wrapped' / 'DLM. COMMA comma delimited' / '1.2 CWLS LOG ASCII STANDARD' shapes on every site; from_dict must mirror the parser's leading-token rule (a 3-segment VERS + trailing text must extract '1.2'). In pylasdev-reborn: F-151 (VERS), M-01 (WRAP/DLM), M-21 (from_dict) — 3 CONFIRMED MEDIUM, same family across 2 modules.

## [pat-20260809134732-c273a1]
Category: pattern
Tags: writer, las30, string-data, format-suppression, mixed-placement, self-unreadable, roundtrip
Changed: 2026-08-09T13:47:32.299429

Mixed string/numeric placement format suppression must cover ALL explicit numeric formats, not just 'S': when a mnemonic is string in one section and numeric in another, the main ~C block must not carry the curve's top-level data_format token for the numeric placement — a hard-wired suppression predicate keyed to data_format=='S' misses explicit 'F'/'E'/'A' formats, emitting a self-unreadable file (own parser raises LASParseError on re-read). H-01 (HIGH, cross-boundary): _all_string_mnemonics() union forces {S} on any mnemonic string anywhere → numeric-first mixed S/N → own parser rejects; H-02 (HIGH): _suppress_s_marker hard-wired to (df).upper()=='S' misses data_format='F'/'E'/'A' with mixed placement — silent numeric→string re-read corruption; F-01 (HIGH, s12 residual): suppression must extend to ANY explicit numeric format when string_union_mnemonics shows mixed placement (emit markerless so the parser's 'if not _df: continue' rescues). Fix: per-section format consistency — the marker decision must consult the string_union (mixed-placement signal), not just the format code; the consumer-side rescue is not a substitute — the writer must not emit what it cannot re-read. Checklist: format matrix (top-level df vs section df) must round-trip in BOTH orders (numeric-first AND string-first) and from_dict path with empty section_curves; suppression predicate = mixed-placement signal OR 'S'; a fix landing only on the empty-format case leaves the explicit-format sub-variant live (F-01 was the H-01 residual).

## [got-20260809134738-13fc48]
Category: gotcha
Tags: dev-reader, thousands, separator, recombination, gate, las-port, grammar-drift, regression, pylasdev
Changed: 2026-10-04T20:19:18.636298

Thousands-separator recombination (dev_reader._recombine_thousands_separators and the LAS port
_data_section_reader): the gate must be HYBRID - full-row exact-fit as PRIMARY, with a per-run snapshot
fallback when the full-row count != expected. A pure full-row gate silently rejects individually-valid
runs (headerless first row with 2+ runs; a surplus row mixing an unambiguous run with a genuine 3-digit
pair) and regressed previously-correct shapes (M-33 vs F-02).
Corruption family: multi-separator values only PARTIALLY recombined (M-76: the len==expected+1 gate
merges only the first pair; a 6-token value never recombines); signed values ('-1,234.5') fail an
isdigit gate and merge a later wrong pair with a warning citing the WRONG pair (M-53); headerless files
with the -1 derivation (_expected_cols = len(values)-1) merge genuine multi-column rows (H-03); the
delimiter-blind gate gives NaN with a misleading warning for semicolon locales (F-13).
Detection/conversion grammar drift: detection recognizes decimal/exponent thousands forms while the
conversion gate stays narrow -> parse-to-NaN with only the generic failure counter. Use ONE shared
token-normalization helper for detection AND conversion with a never-silent contract (any recognized
token is converted or counted in the specific counter with a loud warning).
Checklist: iterate consecutive pairs (not just the first), use a numeric-aware check (try float, not
isdigit), gate on unambiguous evidence (delimiter-aware AND not-headerless-first-row), warn with the
ACTUAL merged pair; mirror every gate change to the LAS port; test multi-separator, signed,
headerless-comma, semicolon-locale, and short-row shapes. In pylasdev-reborn: M-33, M-76, M-53, H-03,
F-13, M-32/F-06 - 7+ CONFIRMED.

## [pat-20260809134744-17c419]
Category: pattern
Tags: models, fix-residual, wholesale, mutation, entry-points, views, data-loss
Changed: 2026-08-09T13:47:44.228890

Fix residuals — when fixing wholesale/mutation paths, verify ALL entry points AND all views in the same pass. A fix applied to one entry point (construction, wholesale assignment, mutation, pickle, deepcopy) or one view (top-level LASFile.logs vs per-section data) leaves the defect live at the twins. This run's own fixes left residuals (M-13..M-18, all CONFIRMED): F-152 reconcile heals only section curves_order/data/string_data, not top-level LASFile.logs (M-13); F-75 deepcopy skips ALREADY-guarded dicts → aliasing (M-14); F-105 duplicate check construction-only, no validate() re-check (M-15); MOD-B-PROD wholesale branch wraps _GuardedList but never deepcopies DataSection objects (M-16); DataSection.validate lacks the data∩string_data overlap check LASFile.validate has (M-17); __setattr__ mnemonic branches re-validate str/pattern/length but not bracket-vs-index (M-18). Meta-lesson: the fix-residual class is the #1 regression source — when fixing a wholesale/mutation path, mechanically enumerate construction/wholesale/mutation/pickle/deepcopy entry points AND top-level-vs-section views, and re-verify the ORIGINAL repro through every entry point and view. Checklist: after any fix, grep all sibling entry points (from_dict/wholesale/__setattr__/validate/pickle) for the same pattern; verify both views (top-level logs vs data_sections) expose the same heal/reconcile; construction-only checks need a validate()-time twin; a sibling method that HAS the check (LASFile.validate) while its twin (DataSection.validate) lacks it is the signature.

## [got-20260809134750-8bb145]
Category: gotcha
Tags: writer, sanitization, warning-gate, raw-vs-emitted, roundtrip, name
Changed: 2026-08-09T13:47:50.552303

Warning gates must test the SANITIZED/EMITTED value, not the raw input: when a writer sanitizes a value before emitting (_sanitize_las_value(name).replace('|','')) but the warning gate tests the RAW name, silent name alteration passes with zero warning. M-38 (MEDIUM, 8-name matrix): name='~A' → no warning yet re-read Section_0; 'A|B'→'AB'; '~CURVE'→'CURVE' — silent name alteration. S8-1 (MEDIUM, s8): explicit DataSection name 'A'/'ASCII' collides with the section-type keyword — writer emits ~A A | CURVE, re-read auto-names Section_0; helper docstring falsely claims 'explicit name preserved exactly' while the collision path silently loses it. Fix: test the SANITIZED value in the warning gate (warn on pipe/tilde stripping); warn loudly when an explicit name collides with a reserved section keyword and will be auto-named on re-read; correct docstrings that overclaim preservation. Checklist: for every writer warning gate, compute the SANITIZED value FIRST and gate on it; reserved-keyword collisions (A/ASCII) need a writer-site warning; the discriminating regression pin asserts the warning fires (pre-fix no warning exists).

## [got-20260809134756-d69576]
Category: gotcha
Tags: reader, parser, units-row, predicate, units-form, token-pattern, data-loss
Changed: 2026-08-09T13:47:56.014812

Shared units-row predicates need UNITS-FORM token patterns, not position heuristics: a predicate that classifies any letters-only row after a mnemonic header as a units row drops genuine letters-only first DATA rows when no units row exists. M-06 (MEDIUM): is_units_header_row (_data_section_reader.py:604-629) classified 'ACME SAND' (a real data row) as units on ALL 4 paths (data_reader.py:1064-1075, :1544-1556; parser.py:4523-4536, :3308-3321) — the row survives when a units row intervenes (one-shot gate closes), so the drop is order/adjacency-dependent. Fix: narrow the predicate to units-form token evidence (every token a plain 1-4 char letter abbreviation AND at least one very short <=2-letter token — the canonical M/FT/MS/MV/IN/CM/MM/US units), NOT position/looks-like-letters; the shared predicate is consumed by every reader/parser path, so refine it ONCE and verify all 4 paths + the pre-scan agree (shared-predicate contract). Checklist: letters-only first data row with no units row must be preserved on all paths; a real units row 'M GAPI' must still be skipped; a predicate refinement must land on every consumer in the same pass; docstring protection claims must be verified against the actual narrow-match logic.

